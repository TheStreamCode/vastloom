#include "vastloom/model/range_artifact.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <string_view>

#include "vastloom/util/json.hpp"
#include "vastloom/util/sha256.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::model {
namespace {

constexpr ByteCount kMaximumManifestBytes = 16U * MiB;

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
                          "Unable to canonicalize range artifact root",
                          root.string());
    }
    const auto canonical_candidate = std::filesystem::weakly_canonical(candidate, error);
    if (error) {
        return make_error(ErrorCode::io_error,
                          "Unable to canonicalize range artifact payload",
                          candidate.string());
    }
    auto root_iterator = canonical_root.begin();
    auto candidate_iterator = canonical_candidate.begin();
    for (; root_iterator != canonical_root.end(); ++root_iterator, ++candidate_iterator) {
        if (candidate_iterator == canonical_candidate.end() ||
            *candidate_iterator != *root_iterator) {
            return make_error(ErrorCode::unsafe_path,
                              std::string(description) + " escapes the artifact root",
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
                          "Range artifact manifest is missing a string field",
                          std::string(key));
    }
    auto parsed = value->as_string();
    if (!parsed || parsed.value().empty() || parsed.value().size() > maximum) {
        return make_error(ErrorCode::invalid_format,
                          "Range artifact manifest string field is invalid",
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
                          "Range artifact manifest is missing an integer field",
                          std::string(key));
    }
    auto parsed = value->as_u64();
    if (!parsed || (!allow_zero && parsed.value() == 0U)) {
        return make_error(ErrorCode::invalid_format,
                          "Range artifact manifest integer field is invalid",
                          std::string(key));
    }
    return parsed.value();
}

Result<bool> required_bool(const util::JsonValue& object, std::string_view key) {
    const auto* value = object.find(key);
    if (value == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Range artifact manifest is missing a Boolean field",
                          std::string(key));
    }
    auto parsed = value->as_bool();
    if (!parsed) {
        return make_error(ErrorCode::invalid_format,
                          "Range artifact manifest Boolean field is invalid",
                          std::string(key));
    }
    return parsed.value();
}

Result<void> validate_closed_layout(const std::filesystem::path& root,
                                    const std::set<std::filesystem::path>& expected_ranges) {
    std::error_code error;
    const auto ranges_directory = root / "ranges";
    const auto ranges_status = std::filesystem::symlink_status(ranges_directory, error);
    if (error || !std::filesystem::is_directory(ranges_status) ||
        std::filesystem::is_symlink(ranges_status)) {
        return make_error(ErrorCode::unsafe_path,
                          "Range artifact ranges directory is missing, unsafe or not a directory",
                          ranges_directory.string());
    }

    std::set<std::filesystem::path> actual_ranges;
    for (std::filesystem::directory_iterator iterator(ranges_directory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        const auto status = iterator->symlink_status(error);
        if (error) break;
        if (std::filesystem::is_symlink(status) || !std::filesystem::is_regular_file(status)) {
            return make_error(ErrorCode::unsafe_path,
                              "Range artifact contains a non-regular payload entry",
                              iterator->path().string());
        }
        actual_ranges.insert(std::filesystem::path("ranges") / iterator->path().filename());
    }
    if (error) {
        return make_error(ErrorCode::io_error,
                          "Unable to enumerate range artifact payloads",
                          ranges_directory.string());
    }
    if (actual_ranges != expected_ranges) {
        return make_error(ErrorCode::integrity_error,
                          "Range artifact payload set does not exactly match its manifest",
                          root.string());
    }

    std::set<std::filesystem::path> expected_root{
        std::filesystem::path(kRangeArtifactManifestName), std::filesystem::path("ranges")};
    std::set<std::filesystem::path> actual_root;
    for (std::filesystem::directory_iterator iterator(root, error), end;
         !error && iterator != end; iterator.increment(error)) {
        actual_root.insert(iterator->path().filename());
    }
    if (error) {
        return make_error(ErrorCode::io_error,
                          "Unable to enumerate range artifact root",
                          root.string());
    }
    if (actual_root != expected_root) {
        return make_error(ErrorCode::integrity_error,
                          "Range artifact contains unexpected root entries",
                          root.string());
    }
    return {};
}

}  // namespace

Result<RangeArtifact> verify_range_artifact(
    const std::filesystem::path& requested_root,
    const RangeArtifactVerificationOptions& options) {
    if (options.max_ranges == 0U || options.max_ranges > kMaximumRangeArtifactRanges ||
        options.max_range_bytes == 0U ||
        options.max_range_bytes > kMaximumRangeArtifactRangeBytes ||
        options.max_payload_bytes == 0U ||
        options.max_payload_bytes > kMaximumRangeArtifactPayloadBytes) {
        return make_error(ErrorCode::invalid_argument,
                          "Range artifact verification bounds are invalid");
    }
    try {
        std::error_code error;
        const auto root = std::filesystem::absolute(requested_root, error).lexically_normal();
        if (error) {
            return make_error(ErrorCode::io_error,
                              "Unable to resolve range artifact directory",
                              requested_root.string());
        }
        const auto root_status = std::filesystem::symlink_status(root, error);
        if (error || !std::filesystem::is_directory(root_status)) {
            return make_error(ErrorCode::not_found,
                              "Range artifact directory does not exist",
                              requested_root.string());
        }
        if (std::filesystem::is_symlink(root_status)) {
            return make_error(ErrorCode::unsafe_path,
                              "Range artifact root must not be a symlink",
                              root.string());
        }

        const auto manifest_relative = std::filesystem::path(kRangeArtifactManifestName);
        auto manifest_path = resolve_regular_file(root, manifest_relative, "Range artifact manifest");
        if (!manifest_path) return manifest_path.error();
        const auto manifest_bytes = std::filesystem::file_size(manifest_path.value(), error);
        if (error || manifest_bytes == 0U || manifest_bytes > kMaximumManifestBytes) {
            return make_error(ErrorCode::invalid_format,
                              "Range artifact manifest size is outside its safety bound",
                              manifest_path.value().string());
        }

        util::JsonLimits limits;
        limits.max_input_bytes = kMaximumManifestBytes;
        limits.max_depth = 16U;
        limits.max_nodes = 100'000U;
        limits.max_string_bytes = 1U * MiB;
        limits.max_array_elements = kMaximumRangeArtifactRanges;
        limits.max_object_members = 65'536U;
        auto manifest = util::load_json_file(manifest_path.value(), limits);
        if (!manifest) return manifest.error();

        auto format = required_string(manifest.value(), "format", 128U);
        auto model_id = required_string(manifest.value(), "model_id");
        auto revision = required_string(manifest.value(), "revision");
        auto scope = required_string(manifest.value(), "scope", 64U);
        auto source_plan_format = required_string(manifest.value(), "source_plan_format", 128U);
        auto source_plan_sha256 = required_string(manifest.value(), "source_plan_sha256", 64U);
        auto verification = required_string(manifest.value(), "verification", 128U);
        auto layer = required_bytes(manifest.value(), "layer", true);
        auto range_count = required_bytes(manifest.value(), "range_count");
        auto payload_bytes = required_bytes(manifest.value(), "payload_bytes");
        auto routing_authority = required_bool(manifest.value(), "routing_authority_verified");
        auto complete_checkpoint = required_bool(manifest.value(), "complete_checkpoint_verified");
        auto execution_supported = required_bool(manifest.value(), "execution_supported");
        if (!format) return format.error();
        if (!model_id) return model_id.error();
        if (!revision) return revision.error();
        if (!scope) return scope.error();
        if (!source_plan_format) return source_plan_format.error();
        if (!source_plan_sha256) return source_plan_sha256.error();
        if (!verification) return verification.error();
        if (!layer) return layer.error();
        if (!range_count) return range_count.error();
        if (!payload_bytes) return payload_bytes.error();
        if (!routing_authority) return routing_authority.error();
        if (!complete_checkpoint) return complete_checkpoint.error();
        if (!execution_supported) return execution_supported.error();

        if (format.value() != kRangeArtifactFormat ||
            !util::is_sha256_hex(source_plan_sha256.value()) ||
            verification.value() != "pinned-revision-content-range-plus-local-sha256" ||
            layer.value() > std::numeric_limits<std::uint32_t>::max() ||
            range_count.value() > options.max_ranges ||
            payload_bytes.value() > options.max_payload_bytes ||
            complete_checkpoint.value() || execution_supported.value()) {
            return make_error(ErrorCode::invalid_format,
                              "Range artifact identity or non-execution contract is invalid",
                              root.string());
        }
        if (scope.value() != "trunk" && scope.value() != "selected-experts" &&
            scope.value() != "all" && scope.value() != "globals") {
            return make_error(ErrorCode::invalid_format,
                              "Range artifact scope is unsupported",
                              scope.value());
        }
        if (scope.value() != "trunk" && scope.value() != "globals" &&
            !routing_authority.value()) {
            return make_error(ErrorCode::integrity_error,
                              "Expert range artifact lacks verified routing authority",
                              scope.value());
        }
        if ((scope.value() == "trunk" || scope.value() == "globals") &&
            routing_authority.value()) {
            return make_error(ErrorCode::integrity_error,
                              "Non-routed range artifact must not claim routing authority",
                              scope.value());
        }

        const auto* ranges_value = manifest.value().find("ranges");
        if (ranges_value == nullptr) {
            return make_error(ErrorCode::invalid_format,
                              "Range artifact manifest is missing ranges",
                              root.string());
        }
        auto ranges = ranges_value->as_array();
        if (!ranges || ranges.value()->empty() ||
            ranges.value()->size() != range_count.value()) {
            return make_error(ErrorCode::invalid_format,
                              "Range artifact range list does not match its count",
                              root.string());
        }

        RangeArtifact result;
        result.root = std::filesystem::weakly_canonical(root, error);
        if (error) {
            return make_error(ErrorCode::io_error,
                              "Unable to canonicalize range artifact root",
                              root.string());
        }
        result.model_id = model_id.take_value();
        result.revision = revision.take_value();
        result.scope = scope.take_value();
        result.layer = static_cast<std::uint32_t>(layer.value());
        result.routing_authority_verified = routing_authority.value();
        result.source_plan_format = source_plan_format.take_value();
        result.source_plan_sha256 = source_plan_sha256.take_value();
        result.payload_bytes = payload_bytes.value();
        result.full_hash_verified = options.verify_hashes;
        result.complete_checkpoint_verified = false;
        result.execution_supported = false;
        result.ranges.reserve(ranges.value()->size());

        std::set<std::filesystem::path> expected_files;
        std::map<std::string, std::vector<std::pair<ByteCount, ByteCount>>, std::less<>> spans;
        struct SourceIdentity {
            std::string object;
            ByteCount bytes{0U};
            std::string sha256;
        };
        std::map<std::string, SourceIdentity, std::less<>> source_identities;
        ByteCount total_bytes = 0U;
        for (std::size_t index = 0U; index < ranges.value()->size(); ++index) {
            const auto& entry = (*ranges.value())[index];
            auto role = required_string(entry, "role", 64U);
            auto label = required_string(entry, "label");
            auto artifact_name = required_string(entry, "artifact", 512U);
            auto bytes = required_bytes(entry, "bytes");
            auto sha256 = required_string(entry, "sha256", 64U);
            auto source_object = required_string(entry, "source_object", 512U);
            auto source_path = required_string(entry, "source_path");
            auto source_object_bytes = required_bytes(entry, "source_object_bytes");
            auto source_object_sha256 = required_string(entry, "source_object_sha256", 64U);
            auto source_offset = required_bytes(entry, "source_offset", true);
            auto source_bytes = required_bytes(entry, "source_bytes");
            if (!role) return role.error();
            if (!label) return label.error();
            if (!artifact_name) return artifact_name.error();
            if (!bytes) return bytes.error();
            if (!sha256) return sha256.error();
            if (!source_object) return source_object.error();
            if (!source_path) return source_path.error();
            if (!source_object_bytes) return source_object_bytes.error();
            if (!source_object_sha256) return source_object_sha256.error();
            if (!source_offset) return source_offset.error();
            if (!source_bytes) return source_bytes.error();

            const auto relative = utf8_path(artifact_name.value());
            const auto original_relative = utf8_path(source_path.value());
            const auto expected_relative = std::filesystem::path("ranges") /
                (std::string(4U - std::min<std::size_t>(4U, std::to_string(index).size()), '0') +
                 std::to_string(index) + ".bin");
            if (relative.generic_string() != expected_relative.generic_string() ||
                !safe_relative_path(original_relative) ||
                original_relative.filename().generic_string() != source_object.value() ||
                bytes.value() > options.max_range_bytes ||
                source_bytes.value() != bytes.value() ||
                source_offset.value() > source_object_bytes.value() ||
                source_bytes.value() > source_object_bytes.value() - source_offset.value() ||
                !util::is_sha256_hex(sha256.value()) ||
                !util::is_sha256_hex(source_object_sha256.value()) ||
                (role.value() != "trunk" && role.value() != "selected-expert" &&
                 role.value() != "global") ||
                (result.scope == "trunk" && role.value() != "trunk") ||
                (result.scope == "selected-experts" && role.value() != "selected-expert") ||
                (result.scope == "globals" && role.value() != "global") ||
                (result.scope == "all" && role.value() == "global")) {
                return make_error(ErrorCode::invalid_format,
                                  "Range artifact entry identity or source span is invalid",
                                  label.value());
            }
            if (!expected_files.insert(relative).second) {
                return make_error(ErrorCode::invalid_format,
                                  "Range artifact contains a duplicate payload path",
                                  relative.string());
            }
            auto path = resolve_regular_file(result.root, relative, "Range artifact payload");
            if (!path) return path.error();
            const auto actual_bytes = std::filesystem::file_size(path.value(), error);
            if (error || actual_bytes != bytes.value()) {
                return make_error(ErrorCode::integrity_error,
                                  "Range artifact payload size does not match its manifest",
                                  path.value().string());
            }
            if (options.verify_hashes) {
                auto actual_hash = util::sha256_file(path.value());
                if (!actual_hash) return actual_hash.error();
                if (actual_hash.value() != sha256.value()) {
                    return make_error(ErrorCode::integrity_error,
                                      "Range artifact payload SHA-256 does not match its manifest",
                                      path.value().string());
                }
            }
            if (bytes.value() > options.max_payload_bytes - total_bytes) {
                return make_error(ErrorCode::resource_exhausted,
                                  "Range artifact payload total exceeds its verification bound",
                                  root.string());
            }
            total_bytes += bytes.value();
            auto& object_spans = spans[source_path.value()];
            object_spans.emplace_back(source_offset.value(), source_bytes.value());
            const SourceIdentity identity{
                source_object.value(), source_object_bytes.value(), source_object_sha256.value()};
            const auto [identity_iterator, inserted] =
                source_identities.emplace(source_path.value(), identity);
            if (!inserted &&
                (identity_iterator->second.object != identity.object ||
                 identity_iterator->second.bytes != identity.bytes ||
                 identity_iterator->second.sha256 != identity.sha256)) {
                return make_error(ErrorCode::integrity_error,
                                  "Range artifact source object identity is inconsistent",
                                  source_path.value());
            }
            result.ranges.push_back(RangeArtifactEntry{
                role.take_value(),
                label.take_value(),
                relative,
                path.take_value(),
                bytes.value(),
                sha256.take_value(),
                source_object.take_value(),
                original_relative,
                source_object_bytes.value(),
                source_object_sha256.take_value(),
                source_offset.value(),
                source_bytes.value(),
            });
        }
        if (total_bytes != result.payload_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "Range artifact payload byte total does not match its manifest",
                              root.string());
        }
        for (auto& [source, object_spans] : spans) {
            std::sort(object_spans.begin(), object_spans.end());
            for (std::size_t index = 1U; index < object_spans.size(); ++index) {
                const auto previous_end = object_spans[index - 1U].first +
                                          object_spans[index - 1U].second;
                if (object_spans[index].first < previous_end) {
                    return make_error(ErrorCode::integrity_error,
                                      "Range artifact contains overlapping source spans",
                                      source);
                }
            }
        }
        auto closed = validate_closed_layout(result.root, expected_files);
        if (!closed) return closed.error();
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate bounded range artifact metadata",
                          requested_root.string());
    } catch (const std::filesystem::filesystem_error& exception) {
        return make_error(ErrorCode::io_error,
                          "Unable to inspect range artifact filesystem state",
                          exception.what());
    }
}

std::string to_json(const RangeArtifact& artifact) {
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"" << kRangeArtifactFormat
           << "\",\"model_id\":\"" << util::json_escape(artifact.model_id)
           << "\",\"revision\":\"" << util::json_escape(artifact.revision)
           << "\",\"scope\":\"" << util::json_escape(artifact.scope)
           << "\",\"layer\":" << artifact.layer
           << ",\"payload_bytes\":" << artifact.payload_bytes
           << ",\"range_count\":" << artifact.ranges.size()
           << ",\"source_plan_format\":\"" << util::json_escape(artifact.source_plan_format)
           << "\",\"source_plan_sha256\":\"" << artifact.source_plan_sha256 << '"'
           << ",\"routing_authority_verified\":"
           << (artifact.routing_authority_verified ? "true" : "false")
           << ",\"local_range_hashes_verified\":"
           << (artifact.full_hash_verified ? "true" : "false")
           << ",\"complete_checkpoint_verified\":"
           << (artifact.complete_checkpoint_verified ? "true" : "false")
           << ",\"execution_supported\":"
           << (artifact.execution_supported ? "true" : "false") << '}';
    return output.str();
}

std::string to_text(const RangeArtifact& artifact) {
    std::ostringstream output;
    output << "Verified range artifact: " << artifact.model_id << '\n'
           << "Revision:             " << artifact.revision << '\n'
           << "Scope / layer:        " << artifact.scope << " / " << artifact.layer << '\n'
           << "Local ranges:         " << artifact.ranges.size() << " ("
           << artifact.payload_bytes << " bytes)" << '\n'
           << "Local SHA-256:        "
           << (artifact.full_hash_verified ? "verified" : "not checked") << '\n'
           << "Complete checkpoint:  no" << '\n'
           << "Execution:            unsupported";
    return output.str();
}

}  // namespace vastloom::model
