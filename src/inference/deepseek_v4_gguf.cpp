#include "vastloom/inference/deepseek_v4_gguf.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <new>
#include <optional>
#include <sstream>
#include <string_view>

#include "vastloom/util/json.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

enum class ExpertProjection : std::uint8_t { down, gate, up };

struct ParsedExpertTensor {
    std::uint32_t layer{0U};
    ExpertProjection projection{ExpertProjection::down};
};

std::optional<ParsedExpertTensor> parse_expert_tensor_name(std::string_view name) {
    constexpr std::string_view prefix = "blk.";
    if (!util::starts_with(name, prefix)) return std::nullopt;
    const auto separator = name.find('.', prefix.size());
    if (separator == std::string_view::npos || separator == prefix.size()) return std::nullopt;
    std::uint32_t layer = 0U;
    const auto converted = std::from_chars(name.data() + prefix.size(), name.data() + separator, layer);
    if (converted.ec != std::errc{} || converted.ptr != name.data() + separator) return std::nullopt;
    const auto suffix = name.substr(separator);
    if (suffix == ".ffn_down_exps.weight") return ParsedExpertTensor{layer, ExpertProjection::down};
    if (suffix == ".ffn_gate_exps.weight") return ParsedExpertTensor{layer, ExpertProjection::gate};
    if (suffix == ".ffn_up_exps.weight") return ParsedExpertTensor{layer, ExpertProjection::up};
    return std::nullopt;
}

model::TensorDType logical_dtype(std::uint32_t ggml_type) noexcept {
    switch (ggml_type) {
        case 0U: return model::TensorDType::f32;
        case 1U: return model::TensorDType::f16;
        case 24U: return model::TensorDType::i8;
        case 25U: return model::TensorDType::i16;
        case 26U: return model::TensorDType::i32;
        case 27U: return model::TensorDType::i64;
        case 28U: return model::TensorDType::f64;
        case 30U: return model::TensorDType::bf16;
        default: return model::TensorDType::quantized;
    }
}

model::TensorRole classify_always_active(std::string_view name) noexcept {
    if (name == "token_embd.weight") return model::TensorRole::embedding;
    if (name == "output.weight") return model::TensorRole::output_head;
    if (name.find("ffn_gate_inp") != std::string_view::npos ||
        name.find("ffn_gate_tid2eid") != std::string_view::npos) {
        return model::TensorRole::router;
    }
    if (name.find("_shexp") != std::string_view::npos) return model::TensorRole::shared_expert;
    if (name.find("hc_") != std::string_view::npos ||
        name.find("_hc_") != std::string_view::npos) {
        return model::TensorRole::hyper_connection;
    }
    if (name.find("norm") != std::string_view::npos) return model::TensorRole::normalization;
    if (name.find("attn") != std::string_view::npos) return model::TensorRole::attention;
    return model::TensorRole::other;
}

Result<ByteCount> add_bytes(ByteCount current, ByteCount added, std::string_view context) {
    if (added > std::numeric_limits<ByteCount>::max() - current) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek GGUF byte total overflows",
                          std::string(context));
    }
    return current + added;
}

}  // namespace

Result<DeepseekV4GgufCatalogReport> build_deepseek_v4_gguf_catalog(
    const model::GgufSplitCatalog& split,
    const DeepseekV4GgufCatalogOptions& options) {
    try {
        if (split.architecture != "deepseek4" || split.shards.empty() || split.tensors.empty() ||
            split.unknown_tensor_sizes != 0U || options.model_id.empty() || options.revision.empty() ||
            options.layers == 0U || options.layers > 1'024U || options.experts_per_layer == 0U ||
            options.experts_per_layer > 65'536U || options.top_k == 0U ||
            options.top_k > options.experts_per_layer ||
            (!options.source_shards.empty() && options.source_shards.size() != split.shards.size())) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek GGUF catalog identity, shape or source mapping is invalid",
                              options.model_id);
        }
        const bool contains_header_snapshot = std::any_of(
            split.shards.begin(), split.shards.end(), [](const model::GgufFile& shard) {
                return shard.header_only();
            });
        if (contains_header_snapshot && options.source_shards.empty() && !options.metadata_only) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek GGUF header snapshots cannot be executable tensor sources",
                              options.model_id);
        }
        const std::uint64_t routed_descriptor_count =
            static_cast<std::uint64_t>(options.layers) * options.experts_per_layer * 3U;
        if (routed_descriptor_count > 1'000'000U ||
            split.tensors.size() > 1'000'000U - static_cast<std::size_t>(routed_descriptor_count)) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek GGUF logical tensor count exceeds the bounded catalog limit",
                              options.model_id);
        }

        model::LogicalTensorCatalog logical;
        logical.model_id = options.model_id;
        logical.revision = options.revision;
        logical.metadata_only = options.metadata_only;
        logical.tensors.reserve(split.tensors.size() + static_cast<std::size_t>(routed_descriptor_count));
        std::vector<std::uint8_t> projection_masks(options.layers, 0U);

        for (const auto& [name, location] : split.tensors) {
            if (location.shard_index >= split.shards.size() ||
                location.tensor_index >= split.shards[location.shard_index].tensors().size()) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek GGUF tensor location is outside its split catalog",
                                  name);
            }
            const auto& shard = split.shards[location.shard_index];
            const auto& tensor = shard.tensors()[location.tensor_index];
            if (!tensor.encoded_bytes.has_value()) {
                return make_error(ErrorCode::unsupported,
                                  "DeepSeek GGUF tensor uses an unknown GGML storage type",
                                  tensor.name);
            }
            const auto& source_path = options.source_shards.empty()
                                          ? shard.path()
                                          : options.source_shards[location.shard_index];
            if (source_path.empty()) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek GGUF source shard path is empty",
                                  tensor.name);
            }
            const auto parsed = parse_expert_tensor_name(tensor.name);
            if (!parsed.has_value()) {
                logical.tensors.push_back(model::LogicalTensorDescriptor{
                    tensor.name,
                    logical_dtype(tensor.ggml_type),
                    tensor.shape,
                    classify_always_active(tensor.name),
                    std::nullopt,
                    std::nullopt,
                    false,
                    model::TensorSourceRange{source_path, tensor.absolute_offset, *tensor.encoded_bytes},
                    tensor.ggml_type,
                });
                continue;
            }
            if (parsed->layer >= options.layers || tensor.shape.size() != 3U ||
                tensor.shape.back() != options.experts_per_layer ||
                *tensor.encoded_bytes % options.experts_per_layer != 0U) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF packed expert tensor shape is invalid",
                                  tensor.name);
            }
            const auto projection_bit = static_cast<std::uint8_t>(
                1U << static_cast<unsigned>(parsed->projection));
            if ((projection_masks[parsed->layer] & projection_bit) != 0U) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF duplicates an expert projection",
                                  tensor.name);
            }
            projection_masks[parsed->layer] |= projection_bit;
            const ByteCount expert_bytes = *tensor.encoded_bytes / options.experts_per_layer;
            if (expert_bytes == 0U) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF expert slice is empty",
                                  tensor.name);
            }
            std::vector<std::uint64_t> expert_shape(tensor.shape.begin(), tensor.shape.end() - 1);
            for (std::uint32_t expert = 0U; expert < options.experts_per_layer; ++expert) {
                const ByteCount expert_index = expert;
                if (expert_index > std::numeric_limits<ByteCount>::max() / expert_bytes ||
                    expert_index * expert_bytes >
                        std::numeric_limits<ByteCount>::max() - tensor.absolute_offset) {
                    return make_error(ErrorCode::invalid_format,
                                      "DeepSeek GGUF expert slice offset overflows",
                                      tensor.name);
                }
                logical.tensors.push_back(model::LogicalTensorDescriptor{
                    tensor.name + ".expert." + std::to_string(expert),
                    logical_dtype(tensor.ggml_type),
                    expert_shape,
                    model::TensorRole::routed_expert,
                    parsed->layer,
                    expert,
                    false,
                    model::TensorSourceRange{
                        source_path,
                        tensor.absolute_offset + expert_index * expert_bytes,
                        expert_bytes,
                    },
                    tensor.ggml_type,
                });
            }
        }
        if (!std::all_of(projection_masks.begin(), projection_masks.end(), [](std::uint8_t mask) {
                return mask == 0x07U;
            })) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek GGUF does not provide exactly three expert projections per layer",
                              options.model_id);
        }
        auto finalized = model::finalize_logical_tensor_catalog(logical);
        if (!finalized) return finalized.error();
        if (logical.tensor_bytes != split.known_tensor_bytes ||
            logical.experts.size() != static_cast<std::size_t>(options.layers) *
                                          options.experts_per_layer) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek GGUF logical catalog does not cover the encoded split tensors",
                              options.model_id);
        }

        std::vector<std::optional<ByteCount>> layer_bundle_bytes(options.layers);
        ByteCount minimum = std::numeric_limits<ByteCount>::max();
        ByteCount maximum = 0U;
        for (const auto& bundle : logical.experts) {
            if (bundle.layer >= options.layers || bundle.expert >= options.experts_per_layer ||
                bundle.tensor_indices.size() != 3U || bundle.tensor_slices.size() != 3U) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek GGUF expert bundle shape is inconsistent",
                                  options.model_id);
            }
            auto& per_layer = layer_bundle_bytes[bundle.layer];
            if (per_layer.has_value() && *per_layer != bundle.bytes) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek GGUF experts in one layer have different byte sizes",
                                  std::to_string(bundle.layer));
            }
            per_layer = bundle.bytes;
            minimum = std::min(minimum, bundle.bytes);
            maximum = std::max(maximum, bundle.bytes);
        }
        ByteCount no_cache_bytes = 0U;
        for (std::uint32_t layer = 0U; layer < options.layers; ++layer) {
            if (!layer_bundle_bytes[layer].has_value() ||
                *layer_bundle_bytes[layer] > std::numeric_limits<ByteCount>::max() / options.top_k) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek GGUF layer bundle total is missing or overflows",
                                  std::to_string(layer));
            }
            auto total = add_bytes(no_cache_bytes,
                                   *layer_bundle_bytes[layer] * options.top_k,
                                   std::to_string(layer));
            if (!total) return total.error();
            no_cache_bytes = total.value();
        }

        DeepseekV4GgufCatalogReport report;
        report.catalog = std::move(logical);
        report.layers = options.layers;
        report.experts_per_layer = options.experts_per_layer;
        report.top_k = options.top_k;
        report.no_cache_expert_bytes_per_token = no_cache_bytes;
        report.minimum_expert_bundle_bytes = minimum;
        report.maximum_expert_bundle_bytes = maximum;
        report.split_shards = split.shards.size();
        report.physical_tensors = split.tensors.size();
        report.total_file_bytes = split.total_file_bytes;
        report.total_header_bytes = split.total_header_bytes;
        report.total_payload_bytes = split.total_payload_bytes;
        return report;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded DeepSeek GGUF logical catalog",
                          options.model_id);
    }
}

std::string to_json(const DeepseekV4GgufCatalogReport& report) {
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"GGUFv3\",\"architecture\":\"deepseek4\""
           << ",\"model_id\":\"" << util::json_escape(report.catalog.model_id)
           << "\",\"revision\":\"" << util::json_escape(report.catalog.revision)
           << "\",\"metadata_only\":" << (report.catalog.metadata_only ? "true" : "false")
           << ",\"split_shards\":" << report.split_shards
           << ",\"physical_tensors\":" << report.physical_tensors
           << ",\"logical_tensors\":" << report.catalog.tensors.size()
           << ",\"layers\":" << report.layers
           << ",\"experts_per_layer\":" << report.experts_per_layer
           << ",\"top_k\":" << report.top_k
           << ",\"logical_experts\":" << report.catalog.experts.size()
           << ",\"total_file_bytes\":" << report.total_file_bytes
           << ",\"total_header_bytes\":" << report.total_header_bytes
           << ",\"total_payload_bytes\":" << report.total_payload_bytes
           << ",\"tensor_bytes\":" << report.catalog.tensor_bytes
           << ",\"always_active_bytes\":" << report.catalog.always_active_bytes
           << ",\"routed_expert_bytes\":" << report.catalog.routed_expert_bytes
           << ",\"minimum_expert_bundle_bytes\":" << report.minimum_expert_bundle_bytes
           << ",\"maximum_expert_bundle_bytes\":" << report.maximum_expert_bundle_bytes
           << ",\"no_cache_expert_bytes_per_token\":"
           << report.no_cache_expert_bytes_per_token << '}';
    return output.str();
}

std::string to_text(const DeepseekV4GgufCatalogReport& report) {
    std::ostringstream output;
    output << "DeepSeek GGUF:       " << report.catalog.model_id << '\n'
           << "Revision label:      " << report.catalog.revision << '\n'
           << "Mode:                "
           << (report.catalog.metadata_only ? "metadata only" : "executable sources") << '\n'
           << "Split shards:        " << report.split_shards << '\n'
           << "Physical tensors:    " << report.physical_tensors << '\n'
           << "Logical tensors:     " << report.catalog.tensors.size() << '\n'
           << "Logical experts:     " << report.catalog.experts.size() << '\n'
           << "Known tensor bytes:  " << format_bytes(report.catalog.tensor_bytes) << '\n'
           << "Always-active bytes: " << format_bytes(report.catalog.always_active_bytes) << '\n'
           << "Routed expert bytes: " << format_bytes(report.catalog.routed_expert_bytes) << '\n'
           << "Expert bundle range: " << format_bytes(report.minimum_expert_bundle_bytes)
           << " - " << format_bytes(report.maximum_expert_bundle_bytes) << '\n'
           << "No-cache bytes/token:" << ' '
           << format_bytes(report.no_cache_expert_bytes_per_token) << '\n'
           << "Execution:           unavailable (inspection/catalog only)";
    return output.str();
}

}  // namespace vastloom::inference
