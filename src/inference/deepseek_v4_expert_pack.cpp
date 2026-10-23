#include "vastloom/inference/deepseek_v4_expert_pack.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

#include "vastloom/util/json.hpp"
#include "vastloom/util/sha256.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

constexpr ByteCount kMaximumExpertPackManifestBytes = 4U * MiB;
constexpr std::size_t kMaximumExpertPackSourceArtifacts = 64U;

std::filesystem::path utf8_path(std::string_view value) {
    std::u8string encoded;
    encoded.reserve(value.size());
    for (const char byte : value) {
        encoded.push_back(static_cast<char8_t>(static_cast<unsigned char>(byte)));
    }
    return std::filesystem::path(encoded);
}

bool safe_relative_path(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return false;
    }
    for (const auto& component : path) {
        if (component.empty() || component == "." || component == "..") return false;
    }
    return true;
}

Result<std::filesystem::path> resolve_regular_file(const std::filesystem::path& root,
                                                   const std::filesystem::path& relative,
                                                   std::string_view description) {
    if (!safe_relative_path(relative)) {
        return make_error(ErrorCode::unsafe_path,
                          std::string(description) + " contains an unsafe relative path",
                          relative.string());
    }
    const auto candidate = root / relative;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(candidate, error);
    if (error || status.type() == std::filesystem::file_type::not_found) {
        return make_error(ErrorCode::not_found,
                          std::string(description) + " is missing",
                          candidate.string());
    }
    if (std::filesystem::is_symlink(status)) {
        return make_error(ErrorCode::unsafe_path,
                          std::string(description) + " must not be a symlink",
                          candidate.string());
    }
    if (!std::filesystem::is_regular_file(status)) {
        return make_error(ErrorCode::integrity_error,
                          std::string(description) + " is not a regular file",
                          candidate.string());
    }
    const auto canonical_root = std::filesystem::weakly_canonical(root, error);
    if (error) {
        return make_error(ErrorCode::io_error,
                          "Unable to canonicalize expert pack root",
                          root.string());
    }
    const auto canonical_candidate = std::filesystem::weakly_canonical(candidate, error);
    if (error) {
        return make_error(ErrorCode::io_error,
                          "Unable to canonicalize expert pack payload",
                          candidate.string());
    }
    auto root_iterator = canonical_root.begin();
    auto candidate_iterator = canonical_candidate.begin();
    for (; root_iterator != canonical_root.end(); ++root_iterator, ++candidate_iterator) {
        if (candidate_iterator == canonical_candidate.end() ||
            *candidate_iterator != *root_iterator) {
            return make_error(ErrorCode::unsafe_path,
                              std::string(description) + " escapes the expert pack root",
                              candidate.string());
        }
    }
    return canonical_candidate;
}

Result<std::string> required_string(const util::JsonValue& object,
                                    std::string_view key,
                                    std::size_t maximum = 4'096U) {
    const auto* value = object.find(key);
    if (value == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Expert pack manifest is missing a string field",
                          std::string(key));
    }
    auto parsed = value->as_string();
    if (!parsed || parsed.value().empty() || parsed.value().size() > maximum) {
        return make_error(ErrorCode::invalid_format,
                          "Expert pack manifest string field is invalid",
                          std::string(key));
    }
    return std::string(parsed.value());
}

Result<ByteCount> required_bytes(const util::JsonValue& object,
                                 std::string_view key,
                                 bool allow_zero = false) {
    const auto* value = object.find(key);
    if (value == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Expert pack manifest is missing a numeric field",
                          std::string(key));
    }
    auto parsed = value->as_u64();
    if (!parsed || (!allow_zero && parsed.value() == 0U)) {
        return make_error(ErrorCode::invalid_format,
                          "Expert pack manifest numeric field is invalid",
                          std::string(key));
    }
    return parsed.value();
}

Result<bool> required_bool(const util::JsonValue& object, std::string_view key) {
    const auto* value = object.find(key);
    if (value == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Expert pack manifest is missing a boolean field",
                          std::string(key));
    }
    auto parsed = value->as_bool();
    if (!parsed) {
        return make_error(ErrorCode::invalid_format,
                          "Expert pack manifest boolean field is invalid",
                          std::string(key));
    }
    return parsed.value();
}

Result<const model::ExpertBundle*> find_bundle(const model::LogicalTensorCatalog& catalog,
                                               std::uint32_t layer,
                                               std::uint32_t expert) {
    const model::ExpertBundle* found = nullptr;
    for (const auto& bundle : catalog.experts) {
        if (bundle.layer != layer || bundle.expert != expert) continue;
        if (found != nullptr) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack expert identity is ambiguous in the catalog",
                              std::to_string(layer) + ":" + std::to_string(expert));
        }
        found = &bundle;
    }
    if (found == nullptr) {
        return make_error(ErrorCode::not_found,
                          "Expert pack expert is absent from the catalog",
                          std::to_string(layer) + ":" + std::to_string(expert));
    }
    return found;
}

bool same_shape(const std::vector<std::uint64_t>& left, const std::vector<std::uint64_t>& right) {
    return left == right;
}

Result<void> validate_record_against_bundle(const model::LogicalTensorCatalog& catalog,
                                            const model::ExpertBundle& bundle,
                                            const DeepseekV4ExpertPackRecord& record) {
    if (bundle.bytes != record.bytes) {
        return make_error(ErrorCode::integrity_error,
                          "Expert pack record size does not match the catalog bundle",
                          std::to_string(bundle.expert));
    }
    if (record.tensors.size() != bundle.tensor_slices.size()) {
        return make_error(ErrorCode::integrity_error,
                          "Expert pack tensor count does not match the catalog bundle",
                          std::to_string(bundle.expert));
    }
    ByteCount total = 0U;
    for (std::size_t index = 0U; index < record.tensors.size(); ++index) {
        const auto& tensor = record.tensors[index];
        const auto& slice = bundle.tensor_slices[index];
        if (slice.tensor_index >= catalog.tensors.size()) {
            return make_error(ErrorCode::integrity_error,
                              "Catalog expert tensor slice is out of range",
                              std::to_string(bundle.expert));
        }
        const auto& descriptor = catalog.tensors[slice.tensor_index];
        if (tensor.name != descriptor.name ||
            tensor.packed_offset != slice.packed_offset ||
            tensor.bytes != slice.bytes ||
            tensor.ggml_type != descriptor.ggml_type ||
            !same_shape(tensor.shape, descriptor.shape)) {
            return make_error(ErrorCode::integrity_error,
                              "Expert pack tensor metadata does not match the catalog",
                              tensor.name);
        }
        if (slice.bytes > std::numeric_limits<ByteCount>::max() - total) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack tensor byte total overflows",
                              tensor.name);
        }
        total += slice.bytes;
    }
    if (total != record.bytes) {
        return make_error(ErrorCode::integrity_error,
                          "Expert pack tensor bytes do not fill the record",
                          std::to_string(bundle.expert));
    }
    return {};
}

}  // namespace

Result<DeepseekV4ExpertPack> verify_deepseek_v4_expert_pack(
    const std::filesystem::path& requested_root,
    const DeepseekV4GgufCatalogReport& report,
    std::uint32_t layer,
    const DeepseekV4ExpertPackVerificationOptions& options) {
    if (options.max_records == 0U ||
        options.max_records > kMaximumDeepseekV4ExpertPackRecords ||
        options.max_payload_bytes == 0U ||
        options.max_payload_bytes > kMaximumDeepseekV4ExpertPackBytes) {
        return make_error(ErrorCode::invalid_argument,
                          "Expert pack verification bounds are invalid");
    }
    if (report.catalog.model_id.empty() || report.catalog.revision.empty() ||
        report.layers == 0U || layer >= report.layers ||
        report.experts_per_layer == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Expert pack requires a pinned catalog and a valid layer");
    }
    try {
        std::error_code error;
        const auto root = std::filesystem::absolute(requested_root, error).lexically_normal();
        if (error) {
            return make_error(ErrorCode::io_error,
                              "Unable to resolve expert pack directory",
                              requested_root.string());
        }
        const auto root_status = std::filesystem::symlink_status(root, error);
        if (error || !std::filesystem::is_directory(root_status)) {
            return make_error(ErrorCode::not_found,
                              "Expert pack directory does not exist",
                              requested_root.string());
        }
        if (std::filesystem::is_symlink(root_status)) {
            return make_error(ErrorCode::unsafe_path,
                              "Expert pack root must not be a symlink",
                              root.string());
        }

        auto manifest_path = resolve_regular_file(
            root, std::filesystem::path(kDeepseekV4ExpertPackManifestName),
            "Expert pack manifest");
        if (!manifest_path) return manifest_path.error();
        const auto manifest_bytes = std::filesystem::file_size(manifest_path.value(), error);
        if (error || manifest_bytes == 0U || manifest_bytes > kMaximumExpertPackManifestBytes) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack manifest size is outside its safety bound",
                              manifest_path.value().string());
        }

        util::JsonLimits limits;
        limits.max_input_bytes = kMaximumExpertPackManifestBytes;
        limits.max_depth = 16U;
        limits.max_nodes = 200'000U;
        limits.max_string_bytes = 1U * MiB;
        limits.max_array_elements = kMaximumDeepseekV4ExpertPackRecords;
        limits.max_object_members = 65'536U;
        auto manifest = util::load_json_file(manifest_path.value(), limits);
        if (!manifest) return manifest.error();

        auto format = required_string(manifest.value(), "format", 128U);
        auto model_id = required_string(manifest.value(), "model_id");
        auto revision = required_string(manifest.value(), "revision");
        auto layer_value = required_bytes(manifest.value(), "layer", true);
        auto alignment = required_bytes(manifest.value(), "alignment");
        auto routing_authority = required_bool(manifest.value(), "routing_authority_verified");
        if (!format) return format.error();
        if (!model_id) return model_id.error();
        if (!revision) return revision.error();
        if (!layer_value) return layer_value.error();
        if (!alignment) return alignment.error();
        if (!routing_authority) return routing_authority.error();

        if (format.value() != kDeepseekV4ExpertPackFormat ||
            model_id.value() != report.catalog.model_id ||
            revision.value() != report.catalog.revision ||
            layer_value.value() != layer ||
            alignment.value() != kDeepseekV4ExpertPackAlignment ||
            !routing_authority.value()) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack identity or alignment is invalid",
                              root.string());
        }

        const auto* payload_value = manifest.value().find("payload");
        if (payload_value == nullptr) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack manifest is missing its payload");
        }
        auto payload_object = payload_value->as_object();
        if (!payload_object) return payload_object.error();
        util::JsonValue payload_json{*payload_object.value()};
        auto payload_name = required_string(payload_json, "path", 512U);
        auto payload_bytes = required_bytes(payload_json, "bytes");
        auto payload_sha = required_string(payload_json, "sha256", 64U);
        if (!payload_name) return payload_name.error();
        if (!payload_bytes) return payload_bytes.error();
        if (!payload_sha) return payload_sha.error();
        if (payload_bytes.value() > options.max_payload_bytes ||
            !util::is_sha256_hex(payload_sha.value())) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack payload identity is invalid",
                              root.string());
        }

        auto payload_relative = utf8_path(payload_name.value());
        if (!safe_relative_path(payload_relative) ||
            payload_relative.filename().generic_string() != payload_name.value()) {
            return make_error(ErrorCode::unsafe_path,
                              "Expert pack payload path must be a plain relative file name",
                              payload_name.value());
        }
        auto payload_path = resolve_regular_file(root, payload_relative, "Expert pack payload");
        if (!payload_path) return payload_path.error();
        const auto actual_payload_bytes = std::filesystem::file_size(payload_path.value(), error);
        if (error || actual_payload_bytes != payload_bytes.value()) {
            return make_error(ErrorCode::integrity_error,
                              "Expert pack payload size does not match its manifest",
                              payload_path.value().string());
        }
        bool payload_hash_verified = false;
        if (options.verify_payload_hash) {
            auto actual_hash = util::sha256_file(payload_path.value());
            if (!actual_hash) return actual_hash.error();
            if (actual_hash.value() != payload_sha.value()) {
                return make_error(ErrorCode::integrity_error,
                                  "Expert pack payload SHA-256 does not match its manifest",
                                  payload_path.value().string());
            }
            payload_hash_verified = true;
        }

        DeepseekV4ExpertPack result;
        result.root = std::filesystem::weakly_canonical(root, error);
        if (error) {
            return make_error(ErrorCode::io_error,
                              "Unable to canonicalize expert pack root",
                              root.string());
        }
        result.payload_path = payload_path.take_value();
        result.model_id = model_id.take_value();
        result.revision = revision.take_value();
        result.layer = layer;
        result.payload_bytes = payload_bytes.value();
        result.routing_authority_verified = true;
        result.full_hash_verified = payload_hash_verified;

        const auto* sources_value = manifest.value().find("source_artifacts");
        if (sources_value != nullptr) {
            auto sources = sources_value->as_array();
            if (!sources || sources.value()->size() > kMaximumExpertPackSourceArtifacts) {
                return make_error(ErrorCode::invalid_format,
                                  "Expert pack source artifact list is invalid",
                                  root.string());
            }
            for (const auto& source : *sources.value()) {
                auto source_object = source.as_object();
                if (!source_object) return source_object.error();
                util::JsonValue source_json{*source_object.value()};
                auto path = required_string(source_json, "path");
                auto manifest_sha = required_string(source_json, "manifest_sha256", 64U);
                if (!path) return path.error();
                if (!manifest_sha) return manifest_sha.error();
                if (!util::is_sha256_hex(manifest_sha.value())) {
                    return make_error(ErrorCode::invalid_format,
                                      "Expert pack source artifact hash is invalid",
                                      path.value());
                }
                result.source_artifact_manifests.push_back(manifest_sha.value());
            }
        }

        const auto* experts_value = manifest.value().find("experts");
        if (experts_value == nullptr) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack manifest is missing its expert records",
                              root.string());
        }
        auto experts = experts_value->as_array();
        if (!experts || experts.value()->empty() ||
            experts.value()->size() > options.max_records ||
            experts.value()->size() > report.experts_per_layer) {
            return make_error(ErrorCode::invalid_format,
                              "Expert pack expert list is empty or outside its bound",
                              root.string());
        }

        std::set<std::uint32_t> seen_experts;
        std::vector<std::pair<ByteCount, ByteCount>> spans;
        result.records.reserve(experts.value()->size());
        for (const auto& expert_value : *experts.value()) {
            auto expert_object = expert_value.as_object();
            if (!expert_object) return expert_object.error();
            util::JsonValue expert_json{*expert_object.value()};
            auto expert_id = required_bytes(expert_json, "expert", true);
            auto offset = required_bytes(expert_json, "offset", true);
            auto bytes = required_bytes(expert_json, "bytes");
            auto sha256 = required_string(expert_json, "sha256", 64U);
            if (!expert_id) return expert_id.error();
            if (!offset) return offset.error();
            if (!bytes) return bytes.error();
            if (!sha256) return sha256.error();
            if (expert_id.value() >= report.experts_per_layer ||
                !seen_experts.insert(static_cast<std::uint32_t>(expert_id.value())).second ||
                offset.value() % kDeepseekV4ExpertPackAlignment != 0U ||
                offset.value() > result.payload_bytes ||
                bytes.value() > result.payload_bytes - offset.value() ||
                !util::is_sha256_hex(sha256.value())) {
                return make_error(ErrorCode::invalid_format,
                                  "Expert pack expert record identity or span is invalid",
                                  std::to_string(expert_id.value()));
            }
            auto bundle = find_bundle(report.catalog, layer,
                                      static_cast<std::uint32_t>(expert_id.value()));
            if (!bundle) return bundle.error();

            DeepseekV4ExpertPackRecord record;
            record.expert = static_cast<std::uint32_t>(expert_id.value());
            record.offset = offset.value();
            record.bytes = bytes.value();
            record.sha256 = sha256.take_value();

            const auto* tensors_value = expert_json.find("tensors");
            if (tensors_value == nullptr) {
                return make_error(ErrorCode::invalid_format,
                                  "Expert pack record is missing its tensors",
                                  std::to_string(record.expert));
            }
            auto tensors = tensors_value->as_array();
            if (!tensors) return tensors.error();
            record.tensors.reserve(tensors.value()->size());
            for (const auto& tensor_value : *tensors.value()) {
                auto tensor_object = tensor_value.as_object();
                if (!tensor_object) return tensor_object.error();
                util::JsonValue tensor_json{*tensor_object.value()};
                auto name = required_string(tensor_json, "name");
                auto packed_offset = required_bytes(tensor_json, "packed_offset", true);
                auto tensor_bytes = required_bytes(tensor_json, "bytes");
                if (!name) return name.error();
                if (!packed_offset) return packed_offset.error();
                if (!tensor_bytes) return tensor_bytes.error();
                DeepseekV4ExpertPackTensor tensor;
                tensor.name = name.take_value();
                tensor.packed_offset = packed_offset.value();
                tensor.bytes = tensor_bytes.value();
                const auto* ggml = tensor_json.find("ggml_type");
                if (ggml != nullptr) {
                    auto parsed = ggml->as_u64();
                    if (!parsed || parsed.value() > std::numeric_limits<std::uint32_t>::max()) {
                        return make_error(ErrorCode::invalid_format,
                                          "Expert pack tensor storage type is invalid",
                                          tensor.name);
                    }
                    tensor.ggml_type = static_cast<std::uint32_t>(parsed.value());
                }
                const auto* shape_value = tensor_json.find("shape");
                if (shape_value != nullptr) {
                    auto shape = shape_value->as_array();
                    if (!shape) return shape.error();
                    tensor.shape.reserve(shape.value()->size());
                    for (const auto& dimension : *shape.value()) {
                        auto parsed = dimension.as_u64();
                        if (!parsed) {
                            return make_error(ErrorCode::invalid_format,
                                              "Expert pack tensor shape is invalid",
                                              tensor.name);
                        }
                        tensor.shape.push_back(parsed.value());
                    }
                }
                record.tensors.push_back(std::move(tensor));
            }
            auto validated = validate_record_against_bundle(report.catalog, *bundle.value(),
                                                            record);
            if (!validated) return validated.error();
            spans.emplace_back(record.offset, record.bytes);
            result.records.push_back(std::move(record));
        }

        std::sort(spans.begin(), spans.end());
        for (std::size_t index = 1U; index < spans.size(); ++index) {
            const auto previous_end = spans[index - 1U].first + spans[index - 1U].second;
            if (spans[index].first < previous_end) {
                return make_error(ErrorCode::integrity_error,
                                  "Expert pack contains overlapping records",
                                  root.string());
            }
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate bounded expert pack metadata",
                          requested_root.string());
    } catch (const std::filesystem::filesystem_error& exception) {
        return make_error(ErrorCode::io_error,
                          "Unable to inspect expert pack filesystem state",
                          exception.what());
    }
}

Result<std::size_t> apply_deepseek_v4_expert_pack(model::LogicalTensorCatalog& catalog,
                                                  const DeepseekV4ExpertPack& pack) {
    if (pack.records.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "Expert pack has no applicable records");
    }
    std::size_t applied = 0U;
    for (const auto& record : pack.records) {
        model::ExpertBundle* target = nullptr;
        for (auto& bundle : catalog.experts) {
            if (bundle.layer != pack.layer || bundle.expert != record.expert) continue;
            if (target != nullptr) {
                return make_error(ErrorCode::invalid_format,
                                  "Expert pack expert identity is ambiguous in the catalog",
                                  std::to_string(record.expert));
            }
            target = &bundle;
        }
        if (target == nullptr) {
            return make_error(ErrorCode::not_found,
                              "Expert pack expert is absent from the catalog",
                              std::to_string(record.expert));
        }
        auto validated = validate_record_against_bundle(catalog, *target, record);
        if (!validated) return validated.error();
        target->read_ranges = {model::TensorSourceRange{
            pack.payload_path,
            record.offset,
            record.bytes,
        }};
        ++applied;
    }
    return applied;
}

std::string to_json(const DeepseekV4ExpertPack& pack) {
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"" << kDeepseekV4ExpertPackFormat
           << "\",\"model_id\":\"" << util::json_escape(pack.model_id)
           << "\",\"revision\":\"" << util::json_escape(pack.revision)
           << "\",\"layer\":" << pack.layer
           << ",\"payload_bytes\":" << pack.payload_bytes
           << ",\"record_count\":" << pack.records.size()
           << ",\"source_artifacts\":" << pack.source_artifact_manifests.size()
           << ",\"routing_authority_verified\":"
           << (pack.routing_authority_verified ? "true" : "false")
           << ",\"payload_hash_verified\":"
           << (pack.full_hash_verified ? "true" : "false") << '}';
    return output.str();
}

}  // namespace vastloom::inference
