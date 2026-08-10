#include "vastloom/inference/deepseek_v4_gguf.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <initializer_list>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <sstream>
#include <string_view>

#include "vastloom/model/imported_checkpoint.hpp"
#include "vastloom/model/layer_streaming.hpp"
#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/util/json.hpp"
#include "vastloom/util/sha256.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

enum class ExpertProjection : std::uint8_t { down, gate, up };

struct ParsedExpertTensor {
    std::uint32_t layer{0U};
    ExpertProjection projection{ExpertProjection::down};
};

std::optional<std::uint32_t> parse_block_layer(std::string_view name) {
    constexpr std::string_view prefix = "blk.";
    if (!util::starts_with(name, prefix)) return std::nullopt;
    const auto separator = name.find('.', prefix.size());
    if (separator == std::string_view::npos || separator == prefix.size()) return std::nullopt;
    std::uint32_t layer = 0U;
    const auto converted = std::from_chars(name.data() + prefix.size(), name.data() + separator, layer);
    if (converted.ec != std::errc{} || converted.ptr != name.data() + separator) return std::nullopt;
    return layer;
}

std::optional<ParsedExpertTensor> parse_expert_tensor_name(std::string_view name) {
    const auto layer = parse_block_layer(name);
    if (!layer.has_value()) return std::nullopt;
    constexpr std::string_view prefix = "blk.";
    const auto separator = name.find('.', prefix.size());
    const auto suffix = name.substr(separator);
    if (suffix == ".ffn_down_exps.weight") return ParsedExpertTensor{*layer, ExpertProjection::down};
    if (suffix == ".ffn_gate_exps.weight") return ParsedExpertTensor{*layer, ExpertProjection::gate};
    if (suffix == ".ffn_up_exps.weight") return ParsedExpertTensor{*layer, ExpertProjection::up};
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

Result<ByteCount> multiply_bytes(std::initializer_list<ByteCount> factors,
                                 std::string_view context) {
    ByteCount result = 1U;
    for (const ByteCount factor : factors) {
        if (factor != 0U && result > std::numeric_limits<ByteCount>::max() / factor) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek GGUF working-set product overflows",
                              std::string(context));
        }
        result *= factor;
    }
    return result;
}

Result<std::uint32_t> required_u32(const model::GgufFile& metadata,
                                   std::string_view key) {
    const auto* entry = metadata.find_metadata(key);
    if (entry == nullptr || !entry->unsigned_value.has_value() ||
        *entry->unsigned_value > std::numeric_limits<std::uint32_t>::max()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek GGUF required unsigned metadata is missing or invalid",
                          std::string(key));
    }
    return static_cast<std::uint32_t>(*entry->unsigned_value);
}

Result<std::vector<std::uint32_t>> required_compress_ratios(
    const model::GgufFile& metadata,
    std::uint32_t expected_count) {
    constexpr std::string_view key = "deepseek4.attention.compress_ratios";
    const auto* entry = metadata.find_metadata(key);
    if (entry == nullptr || entry->type != model::GgufValueType::array ||
        entry->array_elements != expected_count) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek GGUF compression schedule is missing or has the wrong length",
                          std::string(key));
    }
    std::vector<std::uint32_t> result;
    result.reserve(expected_count);
    if (!entry->unsigned_array_values.empty()) {
        if (entry->unsigned_array_values.size() != expected_count) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek GGUF compression schedule was not retained completely",
                              std::string(key));
        }
        for (const auto value : entry->unsigned_array_values) {
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF compression ratio is outside safe bounds",
                                  std::string(key));
            }
            result.push_back(static_cast<std::uint32_t>(value));
        }
    } else {
        if (entry->signed_array_values.size() != expected_count) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek GGUF compression schedule was not retained completely",
                              std::string(key));
        }
        for (const auto value : entry->signed_array_values) {
            if (value < 0 || static_cast<std::uint64_t>(value) >
                                 std::numeric_limits<std::uint32_t>::max()) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF compression ratio is outside safe bounds",
                                  std::string(key));
            }
            result.push_back(static_cast<std::uint32_t>(value));
        }
    }
    return result;
}

struct WorkingSetSummary {
    std::uint32_t maximum_context_tokens{0U};
    ByteCount persistent_kv_bytes{0U};
    ByteCount compression_state_bytes{0U};
    ByteCount rope_table_bytes{0U};
    ByteCount activation_scratch_bytes{0U};
    ByteCount runtime_state_bytes{0U};
    ByteCount minimum_host_resident_bytes{0U};
    ByteCount minimum_streamed_trunk_cpu_ram_bytes{0U};
    ByteCount double_buffer_streamed_trunk_cpu_ram_bytes{0U};
    ByteCount minimum_cpu_streaming_ram_bytes{0U};
    ByteCount double_buffer_cpu_streaming_ram_bytes{0U};
};

Result<WorkingSetSummary> derive_working_set(
    const model::GgufSplitCatalog& split,
    const model::LogicalTensorCatalog& catalog,
    const DeepseekV4GgufCatalogOptions& options,
    ByteCount maximum_expert_bundle_bytes) {
    if (split.shards.empty()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek GGUF split has no primary metadata shard");
    }
    const auto& metadata = split.shards.front();
    auto layers = required_u32(metadata, "deepseek4.block_count");
    auto hash_layers = required_u32(metadata, "deepseek4.hash_layer_count");
    auto maximum_context = required_u32(metadata, "deepseek4.context_length");
    auto hidden = required_u32(metadata, "deepseek4.embedding_length");
    auto heads = required_u32(metadata, "deepseek4.attention.head_count");
    auto head_dim = required_u32(metadata, "deepseek4.attention.key_length");
    auto value_dim = required_u32(metadata, "deepseek4.attention.value_length");
    auto rope_dim = required_u32(metadata, "deepseek4.rope.dimension_count");
    auto window = required_u32(metadata, "deepseek4.attention.sliding_window");
    auto experts = required_u32(metadata, "deepseek4.expert_count");
    auto top_k = required_u32(metadata, "deepseek4.expert_used_count");
    auto expert_ffn = required_u32(metadata, "deepseek4.expert_feed_forward_length");
    auto index_heads = required_u32(metadata, "deepseek4.attention.indexer.head_count");
    auto index_dim = required_u32(metadata, "deepseek4.attention.indexer.key_length");
    auto hyper_connections = required_u32(metadata, "deepseek4.hyper_connection.count");
    const std::array<Result<std::uint32_t>*, 14U> required{{
        &layers, &hash_layers, &maximum_context, &hidden, &heads, &head_dim, &value_dim,
        &rope_dim, &window, &experts, &top_k, &expert_ffn, &index_heads, &index_dim,
    }};
    for (const auto* value : required) {
        if (!*value) return value->error();
    }
    if (!hyper_connections) return hyper_connections.error();
    if (layers.value() != options.layers || experts.value() != options.experts_per_layer ||
        top_k.value() != options.top_k || options.context_tokens == 0U ||
        options.context_tokens > maximum_context.value() || options.batch_size == 0U ||
        options.batch_size > 8U || maximum_context.value() == 0U ||
        maximum_context.value() > 16'777'216U || hidden.value() == 0U ||
        hidden.value() > 262'144U || heads.value() == 0U || heads.value() > 65'536U ||
        head_dim.value() == 0U || head_dim.value() > 65'536U ||
        value_dim.value() != head_dim.value() || rope_dim.value() == 0U ||
        rope_dim.value() > head_dim.value() || window.value() == 0U ||
        window.value() > maximum_context.value() || expert_ffn.value() == 0U ||
        expert_ffn.value() > 16'777'216U || index_heads.value() == 0U ||
        index_heads.value() > 65'536U || index_dim.value() == 0U ||
        index_dim.value() > 65'536U || hyper_connections.value() == 0U ||
        hyper_connections.value() > 1'024U ||
        hash_layers.value() > 64U || layers.value() >
            std::numeric_limits<std::uint32_t>::max() - hash_layers.value()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek GGUF architecture metadata disagrees with the requested plan",
                          options.model_id);
    }
    const std::uint32_t schedule_count = layers.value() + hash_layers.value();
    auto ratios = required_compress_ratios(metadata, schedule_count);
    if (!ratios) return ratios.error();

    constexpr ByteCount kv_scalar_bytes = 2U;       // Native baseline stores persistent KV in BF16.
    constexpr ByteCount state_scalar_bytes = 4U;    // Official compressor decode state is FP32.
    constexpr ByteCount activation_scalar_bytes = 4U;
    constexpr ByteCount kernel_tile_bytes = 64U * MiB;
    ByteCount persistent_kv = 0U;
    ByteCount compression_state = 0U;
    ByteCount maximum_attention_slots = 0U;
    ByteCount maximum_index_slots = 0U;
    for (std::uint32_t layer = 0U; layer < layers.value(); ++layer) {
        const std::uint32_t ratio = ratios.value()[layer];
        if (ratio != 0U && (ratio > options.context_tokens ||
                            options.context_tokens % ratio != 0U)) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek GGUF context must be divisible by every compression ratio",
                              std::to_string(ratio));
        }
        const ByteCount compressed_slots = ratio == 0U ? 0U : options.context_tokens / ratio;
        auto attention_slots = add_bytes(window.value(), compressed_slots, "attention slots");
        if (!attention_slots) return attention_slots.error();
        maximum_attention_slots = std::max(maximum_attention_slots, attention_slots.value());
        auto main_cache = multiply_bytes({options.batch_size,
                                          attention_slots.value(),
                                          head_dim.value(),
                                          kv_scalar_bytes},
                                         "main KV cache");
        if (!main_cache) return main_cache.error();
        auto main_total = add_bytes(persistent_kv, main_cache.value(), "main KV cache");
        if (!main_total) return main_total.error();
        persistent_kv = main_total.value();
        if (ratio == 0U) continue;

        const ByteCount coefficient = ratio == 4U ? 2U : 1U;
        auto main_state = multiply_bytes({2U,
                                          options.batch_size,
                                          coefficient,
                                          ratio,
                                          coefficient,
                                          head_dim.value(),
                                          state_scalar_bytes},
                                         "main compressor state");
        if (!main_state) return main_state.error();
        auto state_total = add_bytes(compression_state,
                                     main_state.value(),
                                     "main compressor state");
        if (!state_total) return state_total.error();
        compression_state = state_total.value();

        if (ratio == 4U) {
            const ByteCount index_slots = options.context_tokens / ratio;
            maximum_index_slots = std::max(maximum_index_slots, index_slots);
            auto index_cache = multiply_bytes({options.batch_size,
                                               index_slots,
                                               index_dim.value(),
                                               kv_scalar_bytes},
                                              "indexer KV cache");
            if (!index_cache) return index_cache.error();
            auto kv_total = add_bytes(persistent_kv, index_cache.value(), "indexer KV cache");
            if (!kv_total) return kv_total.error();
            persistent_kv = kv_total.value();
            auto index_state = multiply_bytes({2U,
                                               options.batch_size,
                                               2U,
                                               ratio,
                                               2U,
                                               index_dim.value(),
                                               state_scalar_bytes},
                                              "indexer compressor state");
            if (!index_state) return index_state.error();
            state_total = add_bytes(compression_state,
                                    index_state.value(),
                                    "indexer compressor state");
            if (!state_total) return state_total.error();
            compression_state = state_total.value();
        }
    }

    auto rope_table = multiply_bytes(
        {options.context_tokens, rope_dim.value(), activation_scalar_bytes}, "RoPE table");
    if (!rope_table) return rope_table.error();

    ByteCount vocabulary = hidden.value();
    const auto output_location = split.tensors.find("output.weight");
    if (output_location != split.tensors.end()) {
        const auto& location = output_location->second;
        if (location.shard_index < split.shards.size() &&
            location.tensor_index < split.shards[location.shard_index].tensors().size()) {
            const auto& shape = split.shards[location.shard_index].tensors()[location.tensor_index].shape;
            if (shape.size() == 2U) vocabulary = std::max(shape[0], shape[1]);
        }
    }
    ByteCount activation_elements = 0U;
    const std::array<Result<ByteCount>, 9U> activation_parts{{
        multiply_bytes({hidden.value(), hyper_connections.value() + 2U}, "hyper-connection activations"),
        multiply_bytes({heads.value(), head_dim.value()}, "attention query"),
        multiply_bytes({2U, heads.value(), maximum_attention_slots}, "attention scores"),
        multiply_bytes({index_heads.value(), index_dim.value()}, "indexer query"),
        multiply_bytes({2U, index_heads.value(), maximum_index_slots}, "indexer scores"),
        multiply_bytes({2U, expert_ffn.value()}, "expert gate/up activations"),
        multiply_bytes({2U, hidden.value()}, "expert reduction activations"),
        multiply_bytes({experts.value() + top_k.value()}, "router activations"),
        Result<ByteCount>(vocabulary),
    }};
    for (const auto& part : activation_parts) {
        if (!part) return part.error();
        auto total = add_bytes(activation_elements, part.value(), "activation scratch");
        if (!total) return total.error();
        activation_elements = total.value();
    }
    auto activation_bytes = multiply_bytes(
        {activation_elements, activation_scalar_bytes}, "activation scratch");
    if (!activation_bytes) return activation_bytes.error();
    auto scratch = add_bytes(kernel_tile_bytes, activation_bytes.value(), "kernel tile scratch");
    if (!scratch) return scratch.error();

    ByteCount runtime_state = 0U;
    for (const auto component : {persistent_kv,
                                 compression_state,
                                 rope_table.value(),
                                 scratch.value()}) {
        auto total = add_bytes(runtime_state, component, "minimum runtime state");
        if (!total) return total.error();
        runtime_state = total.value();
    }
    auto resident_total = add_bytes(runtime_state,
                                    catalog.always_active_bytes,
                                    "resident trunk host set");
    if (!resident_total) return resident_total.error();
    const ByteCount host_resident = resident_total.value();
    auto streamed_minimum = add_bytes(runtime_state,
                                      maximum_expert_bundle_bytes,
                                      "minimum streamed-trunk CPU RAM");
    if (!streamed_minimum) return streamed_minimum.error();
    auto cpu_minimum = add_bytes(host_resident,
                                 maximum_expert_bundle_bytes,
                                 "minimum CPU streaming RAM");
    if (!cpu_minimum) return cpu_minimum.error();
    auto double_expert = multiply_bytes(
        {2U, maximum_expert_bundle_bytes}, "double-buffer expert tier");
    if (!double_expert) return double_expert.error();
    auto streamed_double_minimum = add_bytes(runtime_state,
                                             double_expert.value(),
                                             "double-buffer streamed-trunk CPU RAM");
    if (!streamed_double_minimum) return streamed_double_minimum.error();
    auto double_minimum = add_bytes(host_resident,
                                    double_expert.value(),
                                    "double-buffer CPU streaming RAM");
    if (!double_minimum) return double_minimum.error();

    return WorkingSetSummary{
        maximum_context.value(), persistent_kv, compression_state, rope_table.value(),
        scratch.value(), runtime_state, host_resident, streamed_minimum.value(),
        streamed_double_minimum.value(), cpu_minimum.value(), double_minimum.value()};
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
            options.top_k > options.experts_per_layer || options.context_tokens == 0U ||
            options.batch_size == 0U || options.batch_size > 8U ||
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
                const auto block_layer = parse_block_layer(tensor.name);
                if ((util::starts_with(tensor.name, "blk.") && !block_layer.has_value()) ||
                    (block_layer.has_value() && *block_layer >= options.layers)) {
                    return make_error(ErrorCode::invalid_format,
                                      "DeepSeek GGUF block tensor layer identity is invalid",
                                      tensor.name);
                }
                logical.tensors.push_back(model::LogicalTensorDescriptor{
                    tensor.name,
                    logical_dtype(tensor.ggml_type),
                    tensor.shape,
                    classify_always_active(tensor.name),
                    block_layer,
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

        ByteCount minimum_layer_trunk_bytes = std::numeric_limits<ByteCount>::max();
        ByteCount maximum_layer_trunk_bytes = 0U;
        ByteCount maximum_layer_trunk_tensor_bytes = 0U;
        ByteCount maximum_layer_trunk_tile_bytes = 0U;
        std::size_t layer_trunk_planned_layers = 0U;
        std::size_t layer_trunk_tensor_count = 0U;
        std::size_t layer_trunk_tile_count = 0U;
        std::size_t covered_storage_tensor_count = 0U;
        ByteCount covered_storage_bytes = 0U;
        std::size_t unsupported_storage_tensor_count = 0U;
        ByteCount unsupported_storage_bytes = 0U;
        for (const auto& tensor : logical.tensors) {
            if (tensor.role == model::TensorRole::routed_expert || !tensor.layer.has_value()) {
                continue;
            }
            const bool covered = tensor.ggml_type.has_value() &&
                                 supports_reference_storage_ggml_type(*tensor.ggml_type);
            auto& count = covered ? covered_storage_tensor_count
                                  : unsupported_storage_tensor_count;
            auto& bytes = covered ? covered_storage_bytes : unsupported_storage_bytes;
            if (count == std::numeric_limits<std::size_t>::max() ||
                tensor.source.bytes > std::numeric_limits<ByteCount>::max() - bytes) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF storage-kernel coverage totals overflow",
                                  tensor.name);
            }
            ++count;
            bytes += tensor.source.bytes;
        }
        for (std::uint32_t layer = 0U; layer < options.layers; ++layer) {
            auto plan = model::build_layer_streaming_plan(logical, layer);
            if (!plan) {
                if (plan.error().code == ErrorCode::not_found) continue;
                return plan.error();
            }
            if (layer_trunk_tensor_count > std::numeric_limits<std::size_t>::max() -
                                               plan.value().tensor_count ||
                layer_trunk_tile_count > std::numeric_limits<std::size_t>::max() -
                                             plan.value().tiles.size()) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF layer streaming totals overflow",
                                  options.model_id);
            }
            ++layer_trunk_planned_layers;
            layer_trunk_tensor_count += plan.value().tensor_count;
            layer_trunk_tile_count += plan.value().tiles.size();
            minimum_layer_trunk_bytes =
                std::min(minimum_layer_trunk_bytes, plan.value().trunk_tensor_bytes);
            maximum_layer_trunk_bytes =
                std::max(maximum_layer_trunk_bytes, plan.value().trunk_tensor_bytes);
            maximum_layer_trunk_tensor_bytes =
                std::max(maximum_layer_trunk_tensor_bytes, plan.value().maximum_tensor_bytes);
            maximum_layer_trunk_tile_bytes =
                std::max(maximum_layer_trunk_tile_bytes, plan.value().maximum_tile_bytes);
        }
        if (layer_trunk_planned_layers == 0U) minimum_layer_trunk_bytes = 0U;

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
        auto working_set = derive_working_set(split, logical, options, maximum);
        if (!working_set) return working_set.error();

        DeepseekV4GgufCatalogReport report;
        report.catalog = std::move(logical);
        report.layers = options.layers;
        report.experts_per_layer = options.experts_per_layer;
        report.top_k = options.top_k;
        report.no_cache_expert_bytes_per_token = no_cache_bytes;
        report.minimum_expert_bundle_bytes = minimum;
        report.maximum_expert_bundle_bytes = maximum;
        report.layer_streaming_tile_limit_bytes = model::kDefaultLayerStreamingTileBytes;
        report.minimum_layer_trunk_bytes = minimum_layer_trunk_bytes;
        report.maximum_layer_trunk_bytes = maximum_layer_trunk_bytes;
        report.maximum_layer_trunk_tensor_bytes = maximum_layer_trunk_tensor_bytes;
        report.maximum_layer_trunk_tile_bytes = maximum_layer_trunk_tile_bytes;
        report.layer_trunk_planned_layers = layer_trunk_planned_layers;
        report.layer_trunk_tensor_count = layer_trunk_tensor_count;
        report.layer_trunk_tile_count = layer_trunk_tile_count;
        report.reference_storage_type_covered_tensor_count = covered_storage_tensor_count;
        report.reference_storage_type_covered_bytes = covered_storage_bytes;
        report.reference_storage_type_unsupported_tensor_count = unsupported_storage_tensor_count;
        report.reference_storage_type_unsupported_bytes = unsupported_storage_bytes;
        report.reference_storage_types_complete =
            layer_trunk_tensor_count != 0U &&
            covered_storage_tensor_count == layer_trunk_tensor_count &&
            unsupported_storage_tensor_count == 0U;
        report.context_tokens = options.context_tokens;
        report.maximum_context_tokens = working_set.value().maximum_context_tokens;
        report.batch_size = options.batch_size;
        report.persistent_kv_bytes = working_set.value().persistent_kv_bytes;
        report.compression_state_bytes = working_set.value().compression_state_bytes;
        report.rope_table_bytes = working_set.value().rope_table_bytes;
        report.activation_scratch_bytes = working_set.value().activation_scratch_bytes;
        report.runtime_state_bytes = working_set.value().runtime_state_bytes;
        report.minimum_host_resident_bytes = working_set.value().minimum_host_resident_bytes;
        report.minimum_expert_tier_bytes = maximum;
        report.minimum_streamed_trunk_cpu_ram_bytes =
            working_set.value().minimum_streamed_trunk_cpu_ram_bytes;
        report.double_buffer_streamed_trunk_cpu_ram_bytes =
            working_set.value().double_buffer_streamed_trunk_cpu_ram_bytes;
        report.minimum_cpu_streaming_ram_bytes =
            working_set.value().minimum_cpu_streaming_ram_bytes;
        report.double_buffer_cpu_streaming_ram_bytes =
            working_set.value().double_buffer_cpu_streaming_ram_bytes;
        report.safe_ram_cap_bytes = options.safe_ram_cap_bytes;
        report.safe_vram_cap_bytes = options.safe_vram_cap_bytes;
        report.admission_evaluated = options.safe_ram_cap_bytes != 0U;
        if (report.admission_evaluated &&
            report.runtime_state_bytes <= report.safe_ram_cap_bytes) {
            const ByteCount ram_expert_capacity =
                report.safe_ram_cap_bytes - report.runtime_state_bytes;
            report.expert_fits_ram_tier = maximum <= ram_expert_capacity;
            report.expert_fits_vram_tier = options.safe_vram_cap_bytes != 0U &&
                                           maximum <= options.safe_vram_cap_bytes;
            report.capacity_plan_candidate = report.expert_fits_ram_tier ||
                                             report.expert_fits_vram_tier;
            if (report.minimum_host_resident_bytes <= report.safe_ram_cap_bytes) {
                const ByteCount resident_ram_expert_capacity =
                    report.safe_ram_cap_bytes - report.minimum_host_resident_bytes;
                report.resident_trunk_capacity_candidate =
                    maximum <= resident_ram_expert_capacity || report.expert_fits_vram_tier;
            }
        }
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

const DeepseekV4GgufCheckpointSpec& pinned_deepseek_v4_flash_0731_iq1s_checkpoint() {
    static const DeepseekV4GgufCheckpointSpec spec{
        "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        "fbbb5b93fb787c21338159b0af3318bb3f4d9768",
        {
            {
                "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf",
                5'257'664U,
                "2fa152e25a14500e42e6d98f20d200eec1e945f34dc1cfb981b651b9c86b97be",
                "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf",
                5'257'664U,
                "2fa152e25a14500e42e6d98f20d200eec1e945f34dc1cfb981b651b9c86b97be",
            },
            {
                "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf",
                49'093'726'624U,
                "e9fc77544cd48aa0c31fbf32fdf5432f2b560b92a9aeffd9f690d99867cf0e5f",
                "headers/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf.header",
                52'320U,
                "7128e5c180640517eb10a2ba2eb22a5153f80b3d8316ec70122246a5a2f3db47",
            },
            {
                "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf",
                33'440'253'504U,
                "fd99d5cd0503d668b6febc2eedcc9db68e264cfb79dd355efc88c15eabe4248d",
                "headers/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf.header",
                33'600U,
                "48c9b417ec3c389e834bb299a3273376e8dabbc2855116d12cadb92977091717",
            },
        },
        43U,
        256U,
        6U,
    };
    return spec;
}

Result<DeepseekV4GgufCatalogReport> inspect_deepseek_v4_gguf_checkpoint(
    const std::filesystem::path& checkpoint_root,
    const DeepseekV4GgufCheckpointSpec& spec,
    const DeepseekV4GgufCheckpointInspectOptions& inspect_options) {
    try {
        if (spec.model_id.empty() || spec.revision.empty() || spec.shards.empty() ||
            spec.shards.size() > 1'024U || spec.layers == 0U ||
            spec.experts_per_layer == 0U || spec.top_k == 0U ||
            spec.top_k > spec.experts_per_layer || inspect_options.context_tokens == 0U ||
            inspect_options.batch_size == 0U || inspect_options.batch_size > 8U ||
            inspect_options.safe_ram_cap_bytes == 0U) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek GGUF checkpoint inspection options are invalid",
                              checkpoint_root.string());
        }

        auto imported = model::verify_imported_checkpoint(
            checkpoint_root, inspect_options.verify_hashes);
        if (!imported) return imported.error();
        if (imported.value().model_id != spec.model_id ||
            imported.value().revision != spec.revision ||
            imported.value().files.size() != spec.shards.size()) {
            return make_error(ErrorCode::integrity_error,
                              "Imported DeepSeek GGUF identity or shard count does not match the pinned profile",
                              checkpoint_root.string());
        }

        std::map<std::string, const model::ImportedFile*, std::less<>> imported_by_name;
        for (const auto& file : imported.value().files) {
            const auto name = file.relative_path.generic_string();
            if (!imported_by_name.emplace(name, &file).second) {
                return make_error(ErrorCode::integrity_error,
                                  "Imported DeepSeek GGUF contains a duplicate object identity",
                                  name);
            }
        }

        std::vector<std::filesystem::path> selected_paths;
        selected_paths.reserve(spec.shards.size());
        for (const auto& shard : spec.shards) {
            const auto& relative = imported.value().metadata_only
                                       ? shard.metadata_relative_path
                                       : shard.full_relative_path;
            const ByteCount expected_bytes = imported.value().metadata_only
                                                 ? shard.metadata_bytes
                                                 : shard.full_bytes;
            const auto& expected_sha = imported.value().metadata_only
                                           ? shard.metadata_sha256
                                           : shard.full_sha256;
            if (relative.empty() || expected_bytes == 0U ||
                !util::is_sha256_hex(expected_sha)) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek GGUF checkpoint profile contains an invalid object identity",
                                  relative.string());
            }
            const auto found = imported_by_name.find(relative.generic_string());
            if (found == imported_by_name.end() || found->second->bytes != expected_bytes ||
                found->second->sha256 != expected_sha) {
                return make_error(ErrorCode::integrity_error,
                                  "Imported DeepSeek GGUF object does not match the pinned path, size and SHA-256",
                                  relative.generic_string());
            }
            selected_paths.push_back(imported.value().root / relative);
        }

        model::GgufReaderLimits reader_limits;
        reader_limits.max_header_bytes = 16U * MiB;
        reader_limits.retained_integral_array_keys = {
            "deepseek4.attention.compress_ratios"};
        std::vector<model::GgufFile> shards;
        shards.reserve(spec.shards.size());
        for (std::size_t index = 0U; index < spec.shards.size(); ++index) {
            const auto& identity = spec.shards[index];
            Result<model::GgufFile> opened = imported.value().metadata_only &&
                                                   identity.metadata_bytes < identity.full_bytes
                                               ? model::GgufFile::open_header_prefix(
                                                     selected_paths[index],
                                                     identity.full_bytes,
                                                     reader_limits)
                                               : model::GgufFile::open(
                                                     selected_paths[index], reader_limits);
            if (!opened) return opened.error();
            shards.push_back(opened.take_value());
        }
        auto split = model::finalize_gguf_split_catalog(std::move(shards));
        if (!split) return split.error();

        DeepseekV4GgufCatalogOptions options;
        options.model_id = spec.model_id;
        options.revision = spec.revision;
        options.layers = spec.layers;
        options.experts_per_layer = spec.experts_per_layer;
        options.top_k = spec.top_k;
        options.context_tokens = inspect_options.context_tokens;
        options.batch_size = inspect_options.batch_size;
        options.safe_ram_cap_bytes = inspect_options.safe_ram_cap_bytes;
        options.safe_vram_cap_bytes = inspect_options.safe_vram_cap_bytes;
        options.metadata_only = imported.value().metadata_only ||
                                !imported.value().full_hash_verified;
        // Logical source ranges always name the immutable full checkpoint objects. Metadata-only
        // header snapshots supply offsets and shapes, but must never become payload identities.
        options.source_shards.reserve(spec.shards.size());
        for (const auto& identity : spec.shards) {
            options.source_shards.push_back(imported.value().root / identity.full_relative_path);
        }
        auto report = build_deepseek_v4_gguf_catalog(split.value(), options);
        if (!report) return report.error();
        report.value().imported_manifest_verified = true;
        report.value().imported_file_hashes_verified = imported.value().full_hash_verified;
        report.value().full_payload_present = !imported.value().metadata_only;

        if (!imported.value().metadata_only && imported.value().full_hash_verified &&
            !report.value().capacity_plan_candidate) {
            return make_error(
                ErrorCode::resource_exhausted,
                "DeepSeek GGUF execution refused: one complete expert bundle does not fit a single safe RAM or VRAM tier",
                std::to_string(report.value().maximum_expert_bundle_bytes));
        }
        report.value().executable_source_binding_verified =
            !imported.value().metadata_only && imported.value().full_hash_verified &&
            report.value().capacity_plan_candidate;
        return report;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded DeepSeek GGUF checkpoint binding",
                          checkpoint_root.string());
    }
}

std::string to_json(const DeepseekV4GgufCatalogReport& report) {
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"GGUFv3\",\"architecture\":\"deepseek4\""
           << ",\"model_id\":\"" << util::json_escape(report.catalog.model_id)
           << "\",\"revision\":\"" << util::json_escape(report.catalog.revision)
           << "\",\"metadata_only\":" << (report.catalog.metadata_only ? "true" : "false")
           << ",\"execution_supported\":false"
           << ",\"checkpoint\":{\"imported_manifest_verified\":"
           << (report.imported_manifest_verified ? "true" : "false")
           << ",\"imported_file_hashes_verified\":"
           << (report.imported_file_hashes_verified ? "true" : "false")
           << ",\"full_payload_present\":"
           << (report.full_payload_present ? "true" : "false")
           << ",\"executable_source_binding_verified\":"
           << (report.executable_source_binding_verified ? "true" : "false") << '}'
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
           << ",\"layer_streaming\":{\"tile_limit_bytes\":"
           << report.layer_streaming_tile_limit_bytes
           << ",\"planned_layers\":" << report.layer_trunk_planned_layers
           << ",\"tensor_count\":" << report.layer_trunk_tensor_count
           << ",\"tile_count\":" << report.layer_trunk_tile_count
           << ",\"minimum_layer_trunk_bytes\":" << report.minimum_layer_trunk_bytes
           << ",\"maximum_layer_trunk_bytes\":" << report.maximum_layer_trunk_bytes
           << ",\"maximum_tensor_bytes\":" << report.maximum_layer_trunk_tensor_bytes
           << ",\"maximum_tile_bytes\":" << report.maximum_layer_trunk_tile_bytes
           << ",\"reference_storage_type_covered_tensor_count\":"
           << report.reference_storage_type_covered_tensor_count
           << ",\"reference_storage_type_covered_bytes\":"
           << report.reference_storage_type_covered_bytes
           << ",\"reference_storage_type_unsupported_tensor_count\":"
           << report.reference_storage_type_unsupported_tensor_count
           << ",\"reference_storage_type_unsupported_bytes\":"
           << report.reference_storage_type_unsupported_bytes
           << ",\"reference_storage_types_complete\":"
           << (report.reference_storage_types_complete ? "true" : "false") << '}'
           << ",\"working_set\":{\"context_tokens\":" << report.context_tokens
           << ",\"maximum_context_tokens\":" << report.maximum_context_tokens
           << ",\"batch_size\":" << report.batch_size
           << ",\"persistent_kv_bytes\":" << report.persistent_kv_bytes
           << ",\"compression_state_bytes\":" << report.compression_state_bytes
           << ",\"rope_table_bytes\":" << report.rope_table_bytes
           << ",\"activation_scratch_bytes\":" << report.activation_scratch_bytes
           << ",\"runtime_state_bytes\":" << report.runtime_state_bytes
           << ",\"minimum_host_resident_bytes\":" << report.minimum_host_resident_bytes
           << ",\"minimum_expert_tier_bytes\":" << report.minimum_expert_tier_bytes
           << ",\"minimum_streamed_trunk_cpu_ram_bytes\":"
           << report.minimum_streamed_trunk_cpu_ram_bytes
           << ",\"double_buffer_streamed_trunk_cpu_ram_bytes\":"
           << report.double_buffer_streamed_trunk_cpu_ram_bytes
           << ",\"minimum_cpu_streaming_ram_bytes\":"
           << report.minimum_cpu_streaming_ram_bytes
           << ",\"double_buffer_cpu_streaming_ram_bytes\":"
           << report.double_buffer_cpu_streaming_ram_bytes
           << ",\"mtp_enabled\":false}"
           << ",\"admission\":{\"evaluated\":"
           << (report.admission_evaluated ? "true" : "false")
           << ",\"safe_ram_cap_bytes\":" << report.safe_ram_cap_bytes
           << ",\"safe_vram_cap_bytes\":" << report.safe_vram_cap_bytes
           << ",\"expert_fits_ram_tier\":"
           << (report.expert_fits_ram_tier ? "true" : "false")
           << ",\"expert_fits_vram_tier\":"
           << (report.expert_fits_vram_tier ? "true" : "false")
           << ",\"resident_trunk_capacity_candidate\":"
           << (report.resident_trunk_capacity_candidate ? "true" : "false")
           << ",\"capacity_plan_candidate\":"
           << (report.capacity_plan_candidate ? "true" : "false") << '}'
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
           << "Import manifest:     "
           << (report.imported_manifest_verified ? "verified" : "not bound") << '\n'
           << "Imported file hashes:" << ' '
           << (report.imported_file_hashes_verified ? "verified" : "not verified") << '\n'
           << "Executable binding:  "
           << (report.executable_source_binding_verified ? "verified" : "unavailable") << '\n'
           << "RAM + VRAM pooling:  forbidden for one expert bundle" << '\n'
           << "Split shards:        " << report.split_shards << '\n'
           << "Physical tensors:    " << report.physical_tensors << '\n'
           << "Logical tensors:     " << report.catalog.tensors.size() << '\n'
           << "Logical experts:     " << report.catalog.experts.size() << '\n'
           << "Known tensor bytes:  " << format_bytes(report.catalog.tensor_bytes) << '\n'
           << "Always-active bytes: " << format_bytes(report.catalog.always_active_bytes) << '\n'
           << "Routed expert bytes: " << format_bytes(report.catalog.routed_expert_bytes) << '\n'
           << "Expert bundle range: " << format_bytes(report.minimum_expert_bundle_bytes)
           << " - " << format_bytes(report.maximum_expert_bundle_bytes) << '\n'
           << "Layer trunk plans:   " << report.layer_trunk_planned_layers << " / "
           << report.layers << ", " << report.layer_trunk_tensor_count << " tensors, "
           << report.layer_trunk_tile_count << " tiles" << '\n'
           << "Layer trunk range:   " << format_bytes(report.minimum_layer_trunk_bytes)
           << " - " << format_bytes(report.maximum_layer_trunk_bytes) << '\n'
           << "Largest trunk tensor:" << ' '
           << format_bytes(report.maximum_layer_trunk_tensor_bytes) << '\n'
           << "Bounded tile maximum:" << ' '
           << format_bytes(report.maximum_layer_trunk_tile_bytes) << " / "
           << format_bytes(report.layer_streaming_tile_limit_bytes) << '\n'
           << "Storage type kernels:" << ' '
           << report.reference_storage_type_covered_tensor_count << " covered / "
           << report.reference_storage_type_unsupported_tensor_count << " unsupported"
           << (report.reference_storage_types_complete ? " (complete)" : " (incomplete)")
           << '\n'
           << "Context assumption:  " << report.context_tokens << " / "
           << report.maximum_context_tokens << " tokens, batch " << report.batch_size << '\n'
           << "Persistent KV:       " << format_bytes(report.persistent_kv_bytes) << '\n'
           << "Compression state:   " << format_bytes(report.compression_state_bytes) << '\n'
           << "RoPE table:          " << format_bytes(report.rope_table_bytes) << '\n'
           << "Activation scratch:  " << format_bytes(report.activation_scratch_bytes) << '\n'
           << "Runtime-state floor: " << format_bytes(report.runtime_state_bytes) << '\n'
           << "Resident-trunk floor:" << ' ' << format_bytes(report.minimum_host_resident_bytes) << '\n'
           << "Expert tier floor:   " << format_bytes(report.minimum_expert_tier_bytes) << '\n'
           << "Streamed-trunk CPU:  "
           << format_bytes(report.minimum_streamed_trunk_cpu_ram_bytes) << '\n'
           << "Resident-trunk CPU:  " << format_bytes(report.minimum_cpu_streaming_ram_bytes) << '\n'
           << "Resident double buf: "
           << format_bytes(report.double_buffer_cpu_streaming_ram_bytes) << '\n'
           << "Streamed admission:  "
           << (report.admission_evaluated
                   ? (report.capacity_plan_candidate ? "candidate" : "rejected")
                   : "not evaluated") << '\n'
           << "Resident admission:  "
           << (report.admission_evaluated
                   ? (report.resident_trunk_capacity_candidate ? "candidate" : "rejected")
                   : "not evaluated") << '\n'
           << "No-cache bytes/token:" << ' '
           << format_bytes(report.no_cache_expert_bytes_per_token) << '\n'
           << "Execution:           unavailable (inspection/catalog only)";
    return output.str();
}

}  // namespace vastloom::inference
