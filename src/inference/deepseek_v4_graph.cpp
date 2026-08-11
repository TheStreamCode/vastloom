#include "vastloom/inference/deepseek_v4_reference.hpp"

#include "vastloom/inference/deepseek_v4_gguf.hpp"
#include "vastloom/inference/deepseek_v4_range_artifact.hpp"
#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/model/expert_loader.hpp"
#include "vastloom/model/layer_streaming.hpp"
#include "vastloom/safety/runtime_policy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <limits>
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>

namespace vastloom::inference {
namespace {

constexpr std::size_t kMaximumGraphElements = 64U * 1024U * 1024U;
// Packed weights are not materialized as FP32 graph tensors. This ceiling admits the pinned
// 129,280 x 4,096 Q4_K embedding/head while still rejecting unbounded descriptors.
constexpr std::size_t kMaximumPackedMatrixElements = 1U * 1024U * 1024U * 1024U;
constexpr std::size_t kMaximumPackedFloatElements = 1U * 1024U * 1024U;
constexpr std::size_t kMaximumReferenceTokens = kDeepseekV4ReferenceMaxTokens;
// Sized for the real checkpoint rather than the reduced fixtures: DeepSeek V4 Flash 0731 has 43
// main layers plus one MTP block. The bound stays well below anything that could exhaust memory on
// its own - depth is separately gated by the Governor and by deepseek-forward's preflight.
constexpr std::size_t kMaximumReferenceLayers = 64U;

Result<void> execution_checkpoint(
    const DeepseekV4ReferenceExecutionControl* control) {
    if (control == nullptr || !control->checkpoint) return {};
    try {
        return control->checkpoint();
    } catch (const std::bad_alloc&) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Unable to run the bounded DeepSeek execution checkpoint");
    } catch (const std::exception& error) {
        return make_error(
            ErrorCode::internal_error,
            "DeepSeek execution checkpoint raised an exception", error.what());
    } catch (...) {
        return make_error(
            ErrorCode::internal_error,
            "DeepSeek execution checkpoint raised an unknown exception");
    }
}

Result<std::uint32_t> sample_reference_token(
    std::span<const float> logits,
    const DeepseekV4ReferenceGenerationOptions& options,
    std::mt19937_64& random) {
    if (logits.empty()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek generation logits are empty");
    }
    for (const auto value : logits) {
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek generation logits contain a non-finite value");
        }
    }
    if (options.temperature == 0.0F || options.top_k == 1U) {
        return static_cast<std::uint32_t>(std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    const std::size_t count = std::min(options.top_k, logits.size());
    std::vector<std::size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(count),
        indices.end(),
        [logits](std::size_t left, std::size_t right) {
            return logits[left] == logits[right] ? left < right
                                                 : logits[left] > logits[right];
        });
    indices.resize(count);

    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto index : indices) {
        maximum = std::max(maximum, static_cast<double>(logits[index]));
    }
    std::vector<double> weights(count);
    double total = 0.0;
    for (std::size_t index = 0U; index < count; ++index) {
        weights[index] = std::exp(
            (static_cast<double>(logits[indices[index]]) - maximum) /
            static_cast<double>(options.temperature));
        total += weights[index];
    }
    if (!std::isfinite(total) || total <= 0.0) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek generation sampling weights are invalid");
    }

    // Use mt19937_64 bits directly so seeded sampling is identical across standard libraries.
    constexpr double kInverseTwoTo53 = 1.0 / 9007199254740992.0;
    const double threshold =
        static_cast<double>(random() >> 11U) * kInverseTwoTo53 * total;
    double cumulative = 0.0;
    for (std::size_t index = 0U; index < count; ++index) {
        cumulative += weights[index];
        if (threshold < cumulative) {
            return static_cast<std::uint32_t>(indices[index]);
        }
    }
    return static_cast<std::uint32_t>(indices.back());
}

Result<void> accumulate_generation_telemetry(
    const DeepseekV4ReferenceForwardResult& forward,
    DeepseekV4ReferenceGenerationStats& stats) {
    const auto record_layers = [&](std::span<const DeepseekV4LayerTrace> layers)
        -> Result<void> {
        const auto add_duration = [](double& total, double value) {
            if (!std::isfinite(value) || value < 0.0 ||
                total > std::numeric_limits<double>::max() - value) {
                return false;
            }
            total += value;
            return std::isfinite(total);
        };
        for (const auto& layer : layers) {
            for (const auto& access : layer.expert_accesses) {
                if (stats.expert_requests == std::numeric_limits<std::size_t>::max() ||
                    (access.cache_hit && stats.expert_cache_hits ==
                                             std::numeric_limits<std::size_t>::max()) ||
                    (!access.cache_hit && stats.expert_cache_misses ==
                                              std::numeric_limits<std::size_t>::max()) ||
                    access.resident_bytes >
                        std::numeric_limits<std::uint64_t>::max() -
                            stats.resident_expert_bytes ||
                    access.logical_bytes_read >
                        std::numeric_limits<std::uint64_t>::max() -
                            stats.logical_expert_bytes_read) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "DeepSeek generation expert telemetry is invalid or overflows");
                }
                ++stats.expert_requests;
                if (access.cache_hit) ++stats.expert_cache_hits;
                else ++stats.expert_cache_misses;
                stats.resident_expert_bytes += access.resident_bytes;
                stats.logical_expert_bytes_read += access.logical_bytes_read;
                if (!add_duration(stats.expert_load_service_ms,
                                  access.load_service_ms) ||
                    !add_duration(stats.expert_compute_ms, access.compute_ms) ||
                    !add_duration(stats.io_compute_overlap_ms,
                                  access.io_compute_overlap_ms)) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "DeepSeek generation expert telemetry is invalid or overflows");
                }
            }
        }
        return {};
    };
    auto recorded = record_layers(forward.layers);
    if (!recorded) return recorded.error();
    return record_layers(forward.mtp_layers);
}

Result<std::span<const std::byte>> packed_payload(
    const std::vector<std::byte>& encoded,
    const std::shared_ptr<const model::ResidentByteBuffer>& resident,
    std::size_t resident_offset,
    std::size_t resident_bytes,
    std::string_view context) {
    if (resident != nullptr) {
        if (!encoded.empty() || resident_bytes == 0U) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek packed tensor has ambiguous resident storage",
                              std::string(context));
        }
        const auto payload = resident->bytes();
        if (resident_offset > payload.size() ||
            resident_bytes > payload.size() - resident_offset) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek packed tensor resident range is out of bounds",
                              std::string(context));
        }
        return payload.subspan(resident_offset, resident_bytes);
    }
    if (resident_offset != 0U || resident_bytes != 0U || encoded.empty()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek packed tensor storage is empty or malformed",
                          std::string(context));
    }
    return std::span<const std::byte>(encoded);
}

Result<DeepseekV4PackedMatrix> bind_resident_matrix(
    std::shared_ptr<const model::ResidentByteBuffer> resident,
    std::size_t offset,
    std::size_t available_bytes,
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns) {
    if (resident == nullptr || logical_columns == 0U ||
        logical_columns > storage_columns) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek resident matrix binding is invalid");
    }
    auto required = ggml_storage_bytes(
        ggml_type, rows, storage_columns,
        QuantizedKernelLimits{kMaximumPackedMatrixElements});
    if (!required) return required.error();
    if (available_bytes != required.value()) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek resident matrix byte count does not match its descriptor");
    }
    const auto payload = resident->bytes();
    if (offset > payload.size() || available_bytes > payload.size() - offset) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek resident matrix range exceeds its retained buffer");
    }
    DeepseekV4PackedMatrix matrix;
    matrix.ggml_type = ggml_type;
    matrix.rows = rows;
    matrix.logical_columns = logical_columns;
    matrix.storage_columns = storage_columns;
    matrix.resident = std::move(resident);
    matrix.resident_offset = offset;
    matrix.resident_bytes = available_bytes;
    return matrix;
}

Result<DeepseekV4PackedIntegerTensor> bind_resident_i32(
    std::shared_ptr<const model::ResidentByteBuffer> resident,
    std::size_t offset,
    std::size_t available_bytes,
    std::size_t elements) {
    if (resident == nullptr) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek resident I32 binding has no retained buffer");
    }
    auto required = ggml_storage_bytes(
        kGgmlTypeI32, 1U, elements,
        QuantizedKernelLimits{kMaximumGraphElements});
    if (!required) return required.error();
    if (available_bytes != required.value()) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek resident I32 byte count does not match its descriptor");
    }
    const auto payload = resident->bytes();
    if (offset > payload.size() || available_bytes > payload.size() - offset) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek resident I32 range exceeds its retained buffer");
    }
    DeepseekV4PackedIntegerTensor tensor;
    tensor.elements = elements;
    tensor.resident = std::move(resident);
    tensor.resident_offset = offset;
    tensor.resident_bytes = available_bytes;
    return tensor;
}

Result<DeepseekV4PackedFloatTensor> bind_resident_float(
    std::shared_ptr<const model::ResidentByteBuffer> resident,
    std::size_t offset,
    std::size_t available_bytes,
    std::uint32_t ggml_type,
    std::size_t elements) {
    if (resident == nullptr ||
        (ggml_type != kGgmlTypeF32 && ggml_type != kGgmlTypeBf16)) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek resident float binding is invalid");
    }
    auto required = ggml_storage_bytes(
        ggml_type, 1U, elements,
        QuantizedKernelLimits{kMaximumPackedFloatElements});
    if (!required) return required.error();
    if (available_bytes != required.value()) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek resident float byte count does not match its descriptor");
    }
    const auto payload = resident->bytes();
    if (offset > payload.size() || available_bytes > payload.size() - offset) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek resident float range exceeds its retained buffer");
    }
    DeepseekV4PackedFloatTensor tensor;
    tensor.ggml_type = ggml_type;
    tensor.elements = elements;
    tensor.resident = std::move(resident);
    tensor.resident_offset = offset;
    tensor.resident_bytes = available_bytes;
    return tensor;
}

Result<std::size_t> bounded_product(std::initializer_list<std::size_t> factors,
                                    std::string context) {
    std::size_t result = 1U;
    for (const auto factor : factors) {
        if (factor != 0U && result > std::numeric_limits<std::size_t>::max() / factor) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference graph shape overflows", std::move(context));
        }
        result *= factor;
    }
    if (result > kMaximumGraphElements) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek reference graph exceeds its element limit",
                          std::move(context));
    }
    return result;
}

Result<std::size_t> bounded_packed_product(
    std::initializer_list<std::size_t> factors,
    std::string context) {
    std::size_t result = 1U;
    for (const auto factor : factors) {
        if (factor == 0U || result > std::numeric_limits<std::size_t>::max() / factor) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek packed matrix shape overflows", std::move(context));
        }
        result *= factor;
    }
    if (result > kMaximumPackedMatrixElements) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek packed matrix exceeds its element limit",
                          std::move(context));
    }
    return result;
}

Result<std::size_t> bounded_catalog_product(
    std::span<const std::uint64_t> factors,
    std::string context) {
    if (factors.empty()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek catalog tensor shape is empty",
                          std::move(context));
    }
    std::size_t result = 1U;
    for (const auto factor : factors) {
        if (factor == 0U || factor > std::numeric_limits<std::size_t>::max() ||
            result > std::numeric_limits<std::size_t>::max() /
                         static_cast<std::size_t>(factor)) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek catalog tensor shape overflows",
                              std::move(context));
        }
        result *= static_cast<std::size_t>(factor);
    }
    if (result > kMaximumGraphElements) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek catalog tensor exceeds the reference graph limit",
                          std::move(context));
    }
    return result;
}

bool storage_type_matches_dtype(std::uint32_t ggml_type,
                                model::TensorDType dtype) noexcept {
    switch (ggml_type) {
        case kGgmlTypeF32: return dtype == model::TensorDType::f32;
        case kGgmlTypeBf16: return dtype == model::TensorDType::bf16;
        case kGgmlTypeI32: return dtype == model::TensorDType::i32;
        case kGgmlTypeQ8_0:
        case kGgmlTypeQ4K:
        case kGgmlTypeQ5K:
        case kGgmlTypeQ6K:
        case kGgmlTypeIq2Xxs:
        case kGgmlTypeIq3Xxs:
        case kGgmlTypeIq1S:
        case kGgmlTypeMxfp4: return dtype == model::TensorDType::quantized;
        default: return false;
    }
}

struct Layer0MatrixSpec {
    std::string_view source;
    std::string_view consumer;
    std::size_t rows{0U};
    std::size_t columns{0U};
    model::TensorRole role{model::TensorRole::other};
};

struct Layer0FloatSpec {
    std::string_view source;
    std::string_view consumer;
    std::size_t elements{0U};
    model::TensorRole role{model::TensorRole::other};
};

struct Layer0TrunkLayout {
    std::array<Layer0MatrixSpec, 11U> matrices;
    std::array<Layer0FloatSpec, 9U> floats;
    std::size_t routing_table_elements{0U};
};

Result<Layer0TrunkLayout> make_layer0_trunk_layout(
    const DeepseekV4ReferenceConfig& config) {
    if (config.layer_count == 0U || config.width == 0U ||
        config.intermediate_width == 0U || config.head_count == 0U ||
        config.head_width == 0U || config.query_lora_rank == 0U ||
        config.output_groups == 0U || config.output_lora_rank == 0U ||
        config.expert_count == 0U || config.experts_per_token == 0U ||
        config.experts_per_token > config.expert_count ||
        config.vocabulary_size == 0U || config.hyper_multiplier == 0U ||
        config.head_count > std::numeric_limits<std::size_t>::max() /
                                config.head_width ||
        (config.head_count * config.head_width) % config.output_groups != 0U ||
        config.hyper_multiplier > std::numeric_limits<std::size_t>::max() - 2U) {
        return make_error(
            ErrorCode::invalid_argument,
            "DeepSeek 0731 layer-0 trunk binding configuration is invalid");
    }
    auto mix_width = bounded_product(
        {2U + config.hyper_multiplier, config.hyper_multiplier},
        "DeepSeek 0731 hyper-connection mix width");
    if (!mix_width) return mix_width.error();
    auto flattened_width = bounded_product(
        {config.hyper_multiplier, config.width},
        "DeepSeek 0731 hyper-connection input width");
    if (!flattened_width) return flattened_width.error();
    auto attention_width = bounded_product(
        {config.head_count, config.head_width},
        "DeepSeek 0731 attention width");
    if (!attention_width) return attention_width.error();
    auto output_rank_width = bounded_product(
        {config.output_groups, config.output_lora_rank},
        "DeepSeek 0731 output rank width");
    if (!output_rank_width) return output_rank_width.error();
    auto routing_table_elements = bounded_product(
        {config.vocabulary_size, config.experts_per_token},
        "DeepSeek 0731 hash routing table");
    if (!routing_table_elements) return routing_table_elements.error();

    Layer0TrunkLayout result{{{
        {"blk.0.attn_kv.weight", "layers.0.attn.wkv.weight",
         config.head_width, config.width, model::TensorRole::attention},
        {"blk.0.attn_output_a.weight", "layers.0.attn.wo_a.weight",
         output_rank_width.value(), attention_width.value() / config.output_groups,
         model::TensorRole::attention},
        {"blk.0.attn_output_b.weight", "layers.0.attn.wo_b.weight",
         config.width, output_rank_width.value(), model::TensorRole::attention},
        {"blk.0.attn_q_a.weight", "layers.0.attn.wq_a.weight",
         config.query_lora_rank, config.width, model::TensorRole::attention},
        {"blk.0.attn_q_b.weight", "layers.0.attn.wq_b.weight",
         attention_width.value(), config.query_lora_rank,
         model::TensorRole::attention},
        {"blk.0.ffn_down_shexp.weight", "layers.0.ffn.shared_experts.w2.weight",
         config.width, config.intermediate_width, model::TensorRole::shared_expert},
        {"blk.0.ffn_gate_inp.weight", "layers.0.ffn.gate.weight",
         config.expert_count, config.width, model::TensorRole::router},
        {"blk.0.ffn_gate_shexp.weight", "layers.0.ffn.shared_experts.w1.weight",
         config.intermediate_width, config.width, model::TensorRole::shared_expert},
        {"blk.0.ffn_up_shexp.weight", "layers.0.ffn.shared_experts.w3.weight",
         config.intermediate_width, config.width, model::TensorRole::shared_expert},
        {"blk.0.hc_attn_fn.weight", "layers.0.hc_attn_fn",
         mix_width.value(), flattened_width.value(),
         model::TensorRole::hyper_connection},
        {"blk.0.hc_ffn_fn.weight", "layers.0.hc_ffn_fn",
         mix_width.value(), flattened_width.value(),
         model::TensorRole::hyper_connection},
    }}, {{
        {"blk.0.attn_kv_a_norm.weight", "layers.0.attn.kv_norm.weight",
         config.head_width, model::TensorRole::normalization},
        {"blk.0.attn_norm.weight", "layers.0.attn_norm.weight", config.width,
         model::TensorRole::normalization},
        {"blk.0.attn_q_a_norm.weight", "layers.0.attn.q_norm.weight",
         config.query_lora_rank, model::TensorRole::normalization},
        {"blk.0.attn_sinks.weight", "layers.0.attn.attn_sink", config.head_count,
         model::TensorRole::attention},
        {"blk.0.ffn_norm.weight", "layers.0.ffn_norm.weight", config.width,
         model::TensorRole::normalization},
        {"blk.0.hc_attn_base.weight", "layers.0.hc_attn_base", mix_width.value(),
         model::TensorRole::hyper_connection},
        {"blk.0.hc_attn_scale.weight", "layers.0.hc_attn_scale", 3U,
         model::TensorRole::hyper_connection},
        {"blk.0.hc_ffn_base.weight", "layers.0.hc_ffn_base", mix_width.value(),
         model::TensorRole::hyper_connection},
        {"blk.0.hc_ffn_scale.weight", "layers.0.hc_ffn_scale", 3U,
         model::TensorRole::hyper_connection},
    }}, routing_table_elements.value()};
    return result;
}

Result<void> finite_values(std::span<const float> values, std::string_view context) {
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek graph tensor contains a non-finite value",
                              std::string(context) + "[" + std::to_string(index) + "]");
        }
    }
    return {};
}

class ResolvedFloatTensor {
public:
    explicit ResolvedFloatTensor(std::span<const float> borrowed) noexcept
        : borrowed_(borrowed) {}

    explicit ResolvedFloatTensor(std::vector<float> decoded) noexcept
        : decoded_(std::move(decoded)) {}

    [[nodiscard]] std::span<const float> values() const noexcept {
        if (!decoded_.empty()) return decoded_;
        return borrowed_;
    }

    [[nodiscard]] const float& operator[](std::size_t index) const noexcept {
        return values()[index];
    }

    [[nodiscard]] operator std::span<const float>() const noexcept { return values(); }

private:
    std::span<const float> borrowed_;
    std::vector<float> decoded_;
};

Result<ResolvedFloatTensor> float_tensor(const DeepseekV4ReferenceGraph& graph,
                                         std::string_view name,
                                         std::initializer_list<std::size_t> shape) {
    auto expected = bounded_product(shape, std::string(name));
    if (!expected) return expected.error();
    const auto packed = graph.packed_float_tensors.find(name);
    if (packed != graph.packed_float_tensors.end()) {
        if (packed->second.elements != expected.value() ||
            packed->second.elements > kMaximumPackedFloatElements ||
            (packed->second.ggml_type != kGgmlTypeF32 &&
             packed->second.ggml_type != kGgmlTypeBf16)) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek packed float tensor shape or type does not match",
                              std::string(name));
        }
        auto payload = packed_payload(
            packed->second.encoded, packed->second.resident,
            packed->second.resident_offset, packed->second.resident_bytes, name);
        if (!payload) return payload.error();
        std::vector<float> decoded(expected.value());
        Result<void> status = packed->second.ggml_type == kGgmlTypeF32
            ? dequantize_f32_le(payload.value(), decoded)
            : dequantize_bf16(payload.value(), decoded);
        if (!status) return status.error();
        auto finite = finite_values(decoded, name);
        if (!finite) return finite.error();
        return ResolvedFloatTensor(std::move(decoded));
    }
    const auto found = graph.float_tensors.find(name);
    if (found == graph.float_tensors.end()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph is missing a float tensor", std::string(name));
    }
    if (found->second.size() != expected.value()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph tensor shape does not match", std::string(name));
    }
    auto finite = finite_values(found->second, name);
    if (!finite) return finite.error();
    return ResolvedFloatTensor(std::span<const float>(found->second));
}

Result<std::vector<std::uint32_t>> integer_tensor(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view name,
    std::initializer_list<std::size_t> shape) {
    auto expected = bounded_product(shape, std::string(name));
    if (!expected) return expected.error();
    const auto packed = graph.packed_integer_tensors.find(name);
    if (packed != graph.packed_integer_tensors.end()) {
        if (packed->second.elements != expected.value()) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek packed integer tensor shape does not match",
                              std::string(name));
        }
        auto payload = packed_payload(
            packed->second.encoded, packed->second.resident,
            packed->second.resident_offset, packed->second.resident_bytes, name);
        if (!payload) return payload.error();
        std::vector<std::int32_t> decoded(expected.value());
        auto status = decode_i32_le(payload.value(), decoded);
        if (!status) return status.error();
        std::vector<std::uint32_t> result(decoded.size());
        for (std::size_t index = 0U; index < decoded.size(); ++index) {
            if (decoded[index] < 0) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek packed integer tensor contains a negative index",
                                  std::string(name) + "[" + std::to_string(index) + "]");
            }
            result[index] = static_cast<std::uint32_t>(decoded[index]);
        }
        return result;
    }
    const auto found = graph.integer_tensors.find(name);
    if (found == graph.integer_tensors.end()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph is missing an integer tensor", std::string(name));
    }
    if (found->second.size() != expected.value()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph integer tensor shape does not match",
                          std::string(name));
    }
    return found->second;
}

Result<std::vector<float>> linear(std::span<const float> input,
                                  std::span<const float> weight,
                                  std::size_t rows,
                                  std::size_t input_width,
                                  std::size_t output_width,
                                  std::string_view context) {
    auto input_count = bounded_product({rows, input_width}, std::string(context) + " input");
    if (!input_count) return input_count.error();
    auto weight_count = bounded_product({output_width, input_width},
                                        std::string(context) + " weight");
    if (!weight_count) return weight_count.error();
    auto output_count = bounded_product({rows, output_width}, std::string(context) + " output");
    if (!output_count) return output_count.error();
    if (input.size() != input_count.value() || weight.size() != weight_count.value()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek graph linear tensor shape does not match",
                          std::string(context));
    }
    std::vector<float> output(output_count.value());
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t destination = 0U; destination < output_width; ++destination) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < input_width; ++column) {
                sum += static_cast<double>(input[row * input_width + column]) *
                       static_cast<double>(weight[destination * input_width + column]);
            }
            output[row * output_width + destination] = static_cast<float>(sum);
        }
    }
    auto finite = finite_values(output, context);
    if (!finite) return finite.error();
    return output;
}

Result<std::vector<float>> linear_packed_matrix(
    const DeepseekV4PackedMatrix& matrix,
    std::string_view name,
    std::span<const float> input,
    std::size_t rows,
    std::size_t input_width,
    std::size_t output_width,
    std::string_view context,
    unsigned cpu_threads = 1U) {
    if (!supports_reference_matvec_ggml_type(matrix.ggml_type)) {
        return make_error(ErrorCode::unsupported,
                          "DeepSeek packed matrix type has no reference matvec",
                          std::string(name));
    }
    if (matrix.rows != output_width || matrix.logical_columns != input_width ||
        matrix.storage_columns < matrix.logical_columns || matrix.storage_columns == 0U) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek packed matrix descriptor does not match the graph",
                          std::string(name));
    }
    auto input_count = bounded_product({rows, input_width}, std::string(context) + " input");
    if (!input_count) return input_count.error();
    auto output_count = bounded_product({rows, output_width}, std::string(context) + " output");
    if (!output_count) return output_count.error();
    auto storage_count = bounded_packed_product(
        {matrix.rows, matrix.storage_columns}, std::string(name) + " storage shape");
    if (!storage_count) return storage_count.error();
    auto payload = deepseek_v4_packed_matrix_payload(matrix, name);
    if (!payload) return payload.error();
    if (input.size() != input_count.value()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek packed matrix input shape does not match",
                          std::string(context));
    }
    auto finite = finite_values(input, context);
    if (!finite) return finite.error();
    std::vector<float> padded_input(matrix.storage_columns, 0.0F);
    std::vector<float> scratch(output_width);
    std::vector<float> row_output(output_width);
    std::vector<float> output(output_count.value());
    QuantizedKernelLimits limits;
    limits.max_elements = storage_count.value();
    limits.max_threads = cpu_threads;
    for (std::size_t row = 0U; row < rows; ++row) {
        std::fill(padded_input.begin(), padded_input.end(), 0.0F);
        std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(row * input_width),
                    input_width, padded_input.begin());
        auto multiplied = matvec_ggml_f32_transactional(
            matrix.ggml_type, payload.value(), matrix.rows, matrix.storage_columns,
            padded_input, scratch, row_output, limits);
        if (!multiplied) {
            return make_error(multiplied.error().code, multiplied.error().message,
                              std::string(name) + ": " + multiplied.error().context);
        }
        std::copy(row_output.begin(), row_output.end(),
                  output.begin() + static_cast<std::ptrdiff_t>(row * output_width));
    }
    finite = finite_values(output, context);
    if (!finite) return finite.error();
    return output;
}

Result<std::vector<float>> embedding_rows(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> token_ids) {
    const auto& config = graph.config;
    auto output_count = bounded_product(
        {token_ids.size(), config.width}, "DeepSeek embedding output");
    if (!output_count) return output_count.error();

    const auto packed = graph.packed_matrices.find("embed.weight");
    if (packed == graph.packed_matrices.end()) {
        auto embedding = float_tensor(
            graph, "embed.weight", {config.vocabulary_size, config.width});
        if (!embedding) return embedding.error();
        std::vector<float> output(output_count.value());
        for (std::size_t token = 0U; token < token_ids.size(); ++token) {
            std::copy_n(
                embedding.value().values().begin() + static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(token_ids[token]) * config.width),
                config.width,
                output.begin() + static_cast<std::ptrdiff_t>(token * config.width));
        }
        return output;
    }

    const auto& matrix = packed->second;
    if (!supports_reference_matvec_ggml_type(matrix.ggml_type)) {
        return make_error(ErrorCode::unsupported,
                          "DeepSeek packed embedding type has no reference decoder",
                          "embed.weight");
    }
    if (matrix.rows != config.vocabulary_size ||
        matrix.logical_columns != config.width ||
        matrix.storage_columns < matrix.logical_columns ||
        matrix.storage_columns > kMaximumGraphElements) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek packed embedding descriptor does not match the graph",
                          "embed.weight");
    }
    auto storage_count = bounded_packed_product(
        {matrix.rows, matrix.storage_columns}, "embed.weight storage shape");
    if (!storage_count) return storage_count.error();
    auto payload = deepseek_v4_packed_matrix_payload(matrix, "embed.weight");
    if (!payload) return payload.error();

    QuantizedKernelLimits limits{storage_count.value()};
    auto required = ggml_storage_bytes(
        matrix.ggml_type, matrix.rows, matrix.storage_columns, limits);
    if (!required) return required.error();
    if (payload.value().size() != required.value()) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek packed embedding byte count does not match",
                          "embed.weight");
    }

    std::vector<float> decoded_row(matrix.storage_columns);
    std::vector<float> output(output_count.value());
    for (std::size_t token = 0U; token < token_ids.size(); ++token) {
        auto decoded = dequantize_ggml_row_f32(
            matrix.ggml_type, payload.value(), matrix.rows, matrix.storage_columns,
            token_ids[token], decoded_row, limits);
        if (!decoded) {
            return make_error(decoded.error().code, decoded.error().message,
                              "embed.weight: " + decoded.error().context);
        }
        std::copy_n(decoded_row.begin(), config.width,
                    output.begin() + static_cast<std::ptrdiff_t>(token * config.width));
    }
    auto finite = finite_values(output, "embed.weight");
    if (!finite) return finite.error();
    return output;
}

Result<std::vector<float>> linear_tensor(const DeepseekV4ReferenceGraph& graph,
                                         std::string_view name,
                                         std::span<const float> input,
                                         std::size_t rows,
                                         std::size_t input_width,
                                         std::size_t output_width,
                                         std::string_view context) {
    const auto packed = graph.packed_matrices.find(name);
    if (packed != graph.packed_matrices.end()) {
        return linear_packed_matrix(packed->second, name, input, rows,
                                    input_width, output_width, context,
                                    graph.config.cpu_threads);
    }
    auto weight = float_tensor(graph, name, {output_width, input_width});
    if (!weight) return weight.error();
    return linear(input, weight.value(), rows, input_width, output_width, context);
}

double rope_frequency(std::size_t pair,
                      std::size_t rope_width,
                      double theta,
                      std::size_t original_context,
                      double factor,
                      double beta_fast,
                      double beta_slow) {
    double frequency = std::pow(theta, -2.0 * static_cast<double>(pair) /
                                           static_cast<double>(rope_width));
    if (original_context == 0U) return frequency;
    const auto correction = [&](double rotations) {
        return static_cast<double>(rope_width) *
               std::log(static_cast<double>(original_context) /
                        (rotations * 2.0 * std::numbers::pi)) /
               (2.0 * std::log(theta));
    };
    const double low = std::max(0.0, std::floor(correction(beta_fast)));
    const double high = std::min(static_cast<double>(rope_width - 1U),
                                 std::ceil(correction(beta_slow)));
    double denominator = high - low;
    if (denominator == 0.0) denominator = 0.001;
    const double ramp = std::clamp((static_cast<double>(pair) - low) / denominator,
                                   0.0, 1.0);
    const double smooth = 1.0 - ramp;
    frequency = frequency / factor * (1.0 - smooth) + frequency * smooth;
    return frequency;
}

Result<void> apply_rope_tail(std::vector<float>& values,
                             std::size_t rows,
                             std::size_t heads,
                             std::size_t head_width,
                             std::size_t rope_width,
                             float theta,
                             std::size_t original_context,
                             float factor,
                              float beta_fast,
                              float beta_slow,
                              std::size_t position_stride,
                              bool inverse = false,
                              std::size_t position_offset = 0U) {
    if (rope_width == 0U || rope_width > head_width || rope_width % 2U != 0U ||
        !std::isfinite(theta) || theta <= 0.0F || !std::isfinite(factor) || factor <= 0.0F ||
        !std::isfinite(beta_fast) || beta_fast <= 0.0F ||
        !std::isfinite(beta_slow) || beta_slow <= 0.0F) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek graph RoPE configuration is invalid");
    }
    auto expected = bounded_product({rows, heads, head_width}, "DeepSeek graph RoPE values");
    if (!expected) return expected.error();
    if (values.size() != expected.value()) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek graph RoPE value shape does not match");
    }
    const std::size_t rope_offset = head_width - rope_width;
    for (std::size_t row = 0U; row < rows; ++row) {
        const double position = static_cast<double>(position_offset + row * position_stride);
        for (std::size_t head = 0U; head < heads; ++head) {
            const std::size_t offset = (row * heads + head) * head_width + rope_offset;
            for (std::size_t pair = 0U; pair < rope_width / 2U; ++pair) {
                double angle = position * rope_frequency(
                    pair, rope_width, static_cast<double>(theta), original_context,
                    static_cast<double>(factor), static_cast<double>(beta_fast),
                    static_cast<double>(beta_slow));
                if (inverse) angle = -angle;
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const double real = static_cast<double>(values[offset + pair * 2U]);
                const double imaginary = static_cast<double>(values[offset + pair * 2U + 1U]);
                values[offset + pair * 2U] =
                    static_cast<float>(real * cosine - imaginary * sine);
                values[offset + pair * 2U + 1U] =
                    static_cast<float>(imaginary * cosine + real * sine);
            }
        }
    }
    return {};
}

Result<void> normalize_attention_queries(std::vector<float>& query,
                                         std::size_t tokens,
                                         std::size_t heads,
                                         std::size_t width,
                                         float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek query normalization epsilon is invalid");
    }
    for (std::size_t token = 0U; token < tokens; ++token) {
        for (std::size_t head = 0U; head < heads; ++head) {
            const std::size_t offset = (token * heads + head) * width;
            double squares = 0.0;
            for (std::size_t column = 0U; column < width; ++column) {
                const double value = static_cast<double>(query[offset + column]);
                squares += value * value;
            }
            const double inverse = 1.0 / std::sqrt(
                squares / static_cast<double>(width) + static_cast<double>(epsilon));
            for (std::size_t column = 0U; column < width; ++column) {
                query[offset + column] = static_cast<float>(
                    static_cast<double>(query[offset + column]) * inverse);
            }
        }
    }
    return {};
}

struct CompressionResult {
    std::vector<float> values;
    std::size_t count{0};
};

Result<CompressionResult> compress_prefill(const DeepseekV4ReferenceGraph& graph,
                                            std::string_view prefix,
                                            std::span<const float> input,
                                            std::size_t token_count,
                                            std::size_t ratio,
                                            std::size_t output_width,
                                            std::size_t rope_width,
                                            DeepseekV4ReferenceCompressorState* state = nullptr) {
    const auto& config = graph.config;
    if (ratio == 0U || output_width == 0U || rope_width > output_width) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek compressor configuration is invalid");
    }
    const bool overlap = ratio == 4U;
    const std::size_t coefficient = overlap ? 2U : 1U;
    const std::size_t projected_width = coefficient * output_width;
    const std::string base(prefix);
    auto ape = float_tensor(graph, base + ".ape", {ratio, projected_width});
    if (!ape) return ape.error();
    auto norm = float_tensor(graph, base + ".norm.weight", {output_width});
    if (!norm) return norm.error();
    auto projected = linear_tensor(graph, base + ".wkv.weight", input, token_count,
                                   config.width, projected_width, base + ".wkv");
    if (!projected) return projected.error();
    auto scores = linear_tensor(graph, base + ".wgate.weight", input, token_count,
                                config.width, projected_width, base + ".wgate");
    if (!scores) return scores.error();
    const std::size_t cutoff = token_count - token_count % ratio;
    if (state != nullptr) {
        auto state_size = bounded_product(
            {coefficient, ratio, projected_width}, base + " incremental state");
        if (!state_size) return state_size.error();
        state->key_values.assign(state_size.value(), 0.0F);
        state->scores.assign(state_size.value(),
                             -std::numeric_limits<float>::infinity());
        state->compressed.clear();
        if (overlap && cutoff >= ratio) {
            for (std::size_t position = 0U; position < ratio; ++position) {
                const std::size_t source = cutoff - ratio + position;
                for (std::size_t column = 0U; column < projected_width; ++column) {
                    state->key_values[position * projected_width + column] =
                        projected.value()[source * projected_width + column];
                    state->scores[position * projected_width + column] =
                        scores.value()[source * projected_width + column] +
                        ape.value()[position * projected_width + column];
                }
            }
        }
        const std::size_t remainder = token_count - cutoff;
        const std::size_t offset = overlap ? ratio : 0U;
        for (std::size_t position = 0U; position < remainder; ++position) {
            const std::size_t source = cutoff + position;
            const std::size_t destination = offset + position;
            for (std::size_t column = 0U; column < projected_width; ++column) {
                state->key_values[destination * projected_width + column] =
                    projected.value()[source * projected_width + column];
                state->scores[destination * projected_width + column] =
                    scores.value()[source * projected_width + column] +
                    ape.value()[position * projected_width + column];
            }
        }
    }
    CompressionResult result;
    result.count = cutoff / ratio;
    auto result_size = bounded_product({result.count, output_width}, base + " output");
    if (!result_size) return result_size.error();
    result.values.resize(result_size.value());
    for (std::size_t window = 0U; window < result.count; ++window) {
        for (std::size_t dimension = 0U; dimension < output_width; ++dimension) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t candidate = 0U; candidate < coefficient * ratio; ++candidate) {
                const bool previous = overlap && candidate < ratio;
                if (previous && window == 0U) continue;
                const std::size_t position = candidate % ratio;
                const std::size_t source_token =
                    (previous ? window - 1U : window) * ratio + position;
                const std::size_t projected_column =
                    (previous ? 0U : overlap ? output_width : 0U) + dimension;
                const double score = static_cast<double>(
                    scores.value()[source_token * projected_width + projected_column]) +
                    static_cast<double>(ape.value()[position * projected_width + projected_column]);
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            double accumulated = 0.0;
            for (std::size_t candidate = 0U; candidate < coefficient * ratio; ++candidate) {
                const bool previous = overlap && candidate < ratio;
                if (previous && window == 0U) continue;
                const std::size_t position = candidate % ratio;
                const std::size_t source_token =
                    (previous ? window - 1U : window) * ratio + position;
                const std::size_t projected_column =
                    (previous ? 0U : overlap ? output_width : 0U) + dimension;
                const double exponential = std::exp(
                    static_cast<double>(scores.value()[source_token * projected_width + projected_column]) +
                    static_cast<double>(ape.value()[position * projected_width + projected_column]) -
                    maximum);
                denominator += exponential;
                accumulated += exponential * static_cast<double>(
                    projected.value()[source_token * projected_width + projected_column]);
            }
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek compressor normalization is invalid", base);
            }
            result.values[window * output_width + dimension] =
                static_cast<float>(accumulated / denominator);
        }
    }
    if (result.count == 0U) return result;
    auto normalized = deepseek_v4_rms_norm(result.values, norm.value(), result.count,
                                            output_width, config.norm_epsilon);
    if (!normalized) return normalized.error();
    result.values = normalized.take_value();
    auto rope = apply_rope_tail(result.values, result.count, 1U, output_width, rope_width,
                                config.compression_rope_theta, config.original_context,
                                config.rope_factor, config.beta_fast, config.beta_slow,
                                ratio);
    if (!rope) return rope.error();
    if (state != nullptr) state->compressed = result.values;
    return result;
}

Result<bool> compress_incremental(const DeepseekV4ReferenceGraph& graph,
                                  std::string_view prefix,
                                  std::span<const float> input,
                                  std::size_t start_position,
                                  std::size_t ratio,
                                  std::size_t output_width,
                                  std::size_t rope_width,
                                  DeepseekV4ReferenceCompressorState& state) {
    const auto& config = graph.config;
    const bool overlap = ratio == 4U;
    const std::size_t coefficient = overlap ? 2U : 1U;
    const std::size_t projected_width = coefficient * output_width;
    const std::string base(prefix);
    auto expected_state = bounded_product(
        {coefficient, ratio, projected_width}, base + " incremental state");
    if (!expected_state) return expected_state.error();
    if (input.size() != config.width || state.key_values.size() != expected_state.value() ||
        state.scores.size() != expected_state.value() ||
        state.compressed.size() % output_width != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek compressor incremental state is malformed", base);
    }
    auto ape = float_tensor(graph, base + ".ape", {ratio, projected_width});
    if (!ape) return ape.error();
    auto norm = float_tensor(graph, base + ".norm.weight", {output_width});
    if (!norm) return norm.error();
    auto projected = linear_tensor(graph, base + ".wkv.weight", input, 1U,
                                   config.width, projected_width,
                                   base + ".wkv incremental");
    if (!projected) return projected.error();
    auto scores = linear_tensor(graph, base + ".wgate.weight", input, 1U,
                                config.width, projected_width,
                                base + ".wgate incremental");
    if (!scores) return scores.error();
    const std::size_t position = start_position % ratio;
    const std::size_t destination = (overlap ? ratio : 0U) + position;
    for (std::size_t column = 0U; column < projected_width; ++column) {
        state.key_values[destination * projected_width + column] = projected.value()[column];
        state.scores[destination * projected_width + column] =
            scores.value()[column] + ape.value()[position * projected_width + column];
    }
    if ((start_position + 1U) % ratio != 0U) return false;

    std::vector<float> compressed(output_width);
    for (std::size_t dimension = 0U; dimension < output_width; ++dimension) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t candidate = 0U; candidate < coefficient * ratio; ++candidate) {
            const bool previous = overlap && candidate < ratio;
            const std::size_t row = previous ? candidate :
                (overlap ? ratio + candidate % ratio : candidate);
            const std::size_t column =
                (previous ? 0U : overlap ? output_width : 0U) + dimension;
            maximum = std::max(maximum, static_cast<double>(
                state.scores[row * projected_width + column]));
        }
        double denominator = 0.0;
        double accumulated = 0.0;
        for (std::size_t candidate = 0U; candidate < coefficient * ratio; ++candidate) {
            const bool previous = overlap && candidate < ratio;
            const std::size_t row = previous ? candidate :
                (overlap ? ratio + candidate % ratio : candidate);
            const std::size_t column =
                (previous ? 0U : overlap ? output_width : 0U) + dimension;
            const double exponential = std::exp(
                static_cast<double>(state.scores[row * projected_width + column]) - maximum);
            denominator += exponential;
            accumulated += exponential * static_cast<double>(
                state.key_values[row * projected_width + column]);
        }
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek incremental compressor normalization is invalid",
                              base);
        }
        compressed[dimension] = static_cast<float>(accumulated / denominator);
    }
    if (overlap) {
        std::copy_n(state.key_values.begin() +
                        static_cast<std::ptrdiff_t>(ratio * projected_width),
                    ratio * projected_width, state.key_values.begin());
        std::copy_n(state.scores.begin() +
                        static_cast<std::ptrdiff_t>(ratio * projected_width),
                    ratio * projected_width, state.scores.begin());
    }
    auto normalized = deepseek_v4_rms_norm(compressed, norm.value(), 1U, output_width,
                                            config.norm_epsilon);
    if (!normalized) return normalized.error();
    compressed = normalized.take_value();
    auto rope = apply_rope_tail(compressed, 1U, 1U, output_width, rope_width,
                                config.compression_rope_theta, config.original_context,
                                config.rope_factor, config.beta_fast, config.beta_slow,
                                ratio, false, start_position + 1U - ratio);
    if (!rope) return rope.error();
    if (state.compressed.size() > kMaximumGraphElements - compressed.size()) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek compressed cache exceeds its element limit", base);
    }
    state.compressed.insert(state.compressed.end(), compressed.begin(), compressed.end());
    return true;
}

Result<std::vector<std::int32_t>> window_indices(std::size_t token_count,
                                                 std::size_t window_size) {
    const std::size_t selected = std::min(token_count, window_size);
    auto count = bounded_product({token_count, selected}, "DeepSeek window indices");
    if (!count) return count.error();
    std::vector<std::int32_t> result(count.value(), -1);
    for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t base = token >= window_size ? token - window_size + 1U : 0U;
        for (std::size_t slot = 0U; slot < selected; ++slot) {
            const std::size_t candidate = base + slot;
            if (candidate <= token) {
                result[token * selected + slot] = static_cast<std::int32_t>(candidate);
            }
        }
    }
    return result;
}

Result<std::vector<std::int32_t>> indexer_indices(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view prefix,
    std::span<const float> input,
    std::span<const float> query_rank,
    std::size_t token_count,
    std::size_t ratio,
    std::size_t compressed_count,
    DeepseekV4ReferenceCompressorState* state = nullptr) {
    const auto& config = graph.config;
    const std::string base(prefix);
    auto projected_query = linear_tensor(
        graph, base + ".wq_b.weight", query_rank, token_count,
        config.query_lora_rank, config.index_head_count * config.index_head_width,
        base + ".wq_b");
    if (!projected_query) return projected_query.error();
    auto query_rope = apply_rope_tail(projected_query.value(), token_count,
                                      config.index_head_count, config.index_head_width,
                                      config.rope_width, config.compression_rope_theta,
                                      config.original_context, config.rope_factor,
                                      config.beta_fast, config.beta_slow, 1U);
    if (!query_rope) return query_rope.error();
    auto compressed = compress_prefill(graph, base + ".compressor", input, token_count,
                                       ratio, config.index_head_width, config.rope_width,
                                       state);
    if (!compressed) return compressed.error();
    if (compressed.value().count != compressed_count) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek indexer compression count does not match");
    }
    auto head_weights = linear_tensor(
        graph, base + ".weights_proj.weight", input, token_count, config.width,
        config.index_head_count, base + ".weights_proj");
    if (!head_weights) return head_weights.error();
    const double head_scale = 1.0 / std::sqrt(static_cast<double>(config.index_head_width)) /
                              std::sqrt(static_cast<double>(config.index_head_count));
    const std::size_t selected = std::min(config.index_top_k, compressed_count);
    auto output_count = bounded_product({token_count, selected}, base + " indices");
    if (!output_count) return output_count.error();
    std::vector<std::int32_t> result(output_count.value(), -1);
    std::vector<std::pair<double, std::size_t>> ranking(compressed_count);
    for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t allowed = std::min((token + 1U) / ratio, compressed_count);
        for (std::size_t compressed_index = 0U; compressed_index < compressed_count;
             ++compressed_index) {
            double score = -std::numeric_limits<double>::infinity();
            if (compressed_index < allowed) {
                score = 0.0;
                for (std::size_t head = 0U; head < config.index_head_count; ++head) {
                    double dot = 0.0;
                    for (std::size_t column = 0U; column < config.index_head_width; ++column) {
                        dot += static_cast<double>(projected_query.value()[
                                   (token * config.index_head_count + head) *
                                       config.index_head_width + column]) *
                               static_cast<double>(compressed.value().values[
                                   compressed_index * config.index_head_width + column]);
                    }
                    score += std::max(dot, 0.0) *
                             static_cast<double>(head_weights.value()[
                                 token * config.index_head_count + head]) *
                             head_scale;
                }
            }
            ranking[compressed_index] = {score, compressed_index};
        }
        std::partial_sort(ranking.begin(), ranking.begin() + static_cast<std::ptrdiff_t>(selected),
                          ranking.end(), [](const auto& left, const auto& right) {
                              return left.first == right.first ? left.second < right.second
                                                               : left.first > right.first;
                          });
        for (std::size_t slot = 0U; slot < selected; ++slot) {
            if (std::isfinite(ranking[slot].first)) {
                result[token * selected + slot] = static_cast<std::int32_t>(
                    token_count + ranking[slot].second);
            }
        }
    }
    return result;
}

Result<std::vector<std::int32_t>> indexer_indices_incremental(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view prefix,
    std::span<const float> input,
    std::span<const float> query_rank,
    std::size_t start_position,
    std::size_t ratio,
    DeepseekV4ReferenceCompressorState& state) {
    const auto& config = graph.config;
    const std::string base(prefix);
    auto projected_query = linear_tensor(
        graph, base + ".wq_b.weight", query_rank, 1U, config.query_lora_rank,
        config.index_head_count * config.index_head_width,
        base + ".wq_b incremental");
    if (!projected_query) return projected_query.error();
    auto query_rope = apply_rope_tail(
        projected_query.value(), 1U, config.index_head_count, config.index_head_width,
        config.rope_width, config.compression_rope_theta, config.original_context,
        config.rope_factor, config.beta_fast, config.beta_slow, 1U, false,
        start_position);
    if (!query_rope) return query_rope.error();
    auto compressed = compress_incremental(graph, base + ".compressor", input,
                                           start_position, ratio,
                                           config.index_head_width, config.rope_width,
                                           state);
    if (!compressed) return compressed.error();
    const std::size_t compressed_count = (start_position + 1U) / ratio;
    auto expected_cache = bounded_product({compressed_count, config.index_head_width},
                                          base + " incremental cache");
    if (!expected_cache) return expected_cache.error();
    if (state.compressed.size() != expected_cache.value()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek indexer incremental cache length is malformed", base);
    }
    auto head_weights = linear_tensor(
        graph, base + ".weights_proj.weight", input, 1U, config.width,
        config.index_head_count, base + ".weights_proj incremental");
    if (!head_weights) return head_weights.error();
    const double head_scale = 1.0 / std::sqrt(static_cast<double>(config.index_head_width)) /
                              std::sqrt(static_cast<double>(config.index_head_count));
    const std::size_t selected = std::min(config.index_top_k, compressed_count);
    std::vector<std::pair<double, std::size_t>> ranking(compressed_count);
    for (std::size_t compressed_index = 0U; compressed_index < compressed_count;
         ++compressed_index) {
        double score = 0.0;
        for (std::size_t head = 0U; head < config.index_head_count; ++head) {
            double dot = 0.0;
            for (std::size_t column = 0U; column < config.index_head_width; ++column) {
                dot += static_cast<double>(projected_query.value()[
                           head * config.index_head_width + column]) *
                       static_cast<double>(state.compressed[
                           compressed_index * config.index_head_width + column]);
            }
            score += std::max(dot, 0.0) *
                     static_cast<double>(head_weights.value()[head]) * head_scale;
        }
        ranking[compressed_index] = {score, compressed_index};
    }
    std::partial_sort(ranking.begin(),
                      ranking.begin() + static_cast<std::ptrdiff_t>(selected),
                      ranking.end(), [](const auto& left, const auto& right) {
                          return left.first == right.first ? left.second < right.second
                                                           : left.first > right.first;
                      });
    std::vector<std::int32_t> result(selected);
    for (std::size_t slot = 0U; slot < selected; ++slot) {
        result[slot] = static_cast<std::int32_t>(config.window_size + ranking[slot].second);
    }
    return result;
}

struct AttentionResult {
    std::vector<float> output;
    std::vector<std::int32_t> indices;
    std::size_t selected_per_token{0};
};

Result<AttentionResult> finish_attention(const DeepseekV4ReferenceGraph& graph,
                                         std::string_view base,
                                         std::span<const float> query,
                                         std::span<const float> key_value,
                                         std::vector<std::int32_t> indices,
                                         std::size_t token_count,
                                         std::size_t selected_per_token,
                                         float theta,
                                         std::size_t original_context,
                                         std::size_t position_offset) {
    const auto& config = graph.config;
    AttentionResult result;
    result.indices = std::move(indices);
    result.selected_per_token = selected_per_token;
    auto sink = float_tensor(graph, std::string(base) + ".attn_sink", {config.head_count});
    if (!sink) return sink.error();
    auto attended = deepseek_v4_sparse_attention(
        query, key_value, sink.value(), result.indices, token_count,
        key_value.size() / config.head_width, config.head_count, config.head_width,
        selected_per_token,
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(config.head_width))));
    if (!attended) return attended.error();
    auto inverse_rope = apply_rope_tail(
        attended.value(), token_count, config.head_count, config.head_width,
        config.rope_width, theta, original_context, config.rope_factor,
        config.beta_fast, config.beta_slow, 1U, true, position_offset);
    if (!inverse_rope) return inverse_rope.error();
    const std::size_t group_input =
        config.head_count * config.head_width / config.output_groups;
    const std::string wo_a_name = std::string(base) + ".wo_a.weight";
    auto group_count = bounded_product(
        {token_count, config.output_groups, config.output_lora_rank},
        std::string(base) + ".wo_a output");
    if (!group_count) return group_count.error();
    std::vector<float> grouped(group_count.value());
    if (graph.packed_matrices.contains(wo_a_name)) {
        for (std::size_t token = 0U; token < token_count; ++token) {
            for (std::size_t group = 0U; group < config.output_groups; ++group) {
                const std::span<const float> group_values(
                    attended.value().data() + static_cast<std::ptrdiff_t>(
                        token * config.head_count * config.head_width +
                        group * group_input),
                    group_input);
                auto projected_group = linear_tensor(
                    graph, wo_a_name, group_values, 1U, group_input,
                    config.output_groups * config.output_lora_rank, wo_a_name);
                if (!projected_group) return projected_group.error();
                std::copy_n(
                    projected_group.value().begin() + static_cast<std::ptrdiff_t>(
                        group * config.output_lora_rank),
                    config.output_lora_rank,
                    grouped.begin() + static_cast<std::ptrdiff_t>(
                        (token * config.output_groups + group) *
                        config.output_lora_rank));
            }
        }
    } else {
        auto wo_a = float_tensor(
            graph, wo_a_name,
            {config.output_groups * config.output_lora_rank, group_input});
        if (!wo_a) return wo_a.error();
        for (std::size_t token = 0U; token < token_count; ++token) {
            for (std::size_t group = 0U; group < config.output_groups; ++group) {
                for (std::size_t rank = 0U; rank < config.output_lora_rank; ++rank) {
                    double sum = 0.0;
                    for (std::size_t column = 0U; column < group_input; ++column) {
                        sum += static_cast<double>(attended.value()[
                                   token * config.head_count * config.head_width +
                                   group * group_input + column]) *
                               static_cast<double>(wo_a.value()[
                                   (group * config.output_lora_rank + rank) * group_input +
                                   column]);
                    }
                    grouped[(token * config.output_groups + group) *
                                config.output_lora_rank + rank] = static_cast<float>(sum);
                }
            }
        }
    }
    auto projected = linear_tensor(
        graph, std::string(base) + ".wo_b.weight", grouped, token_count,
        config.output_groups * config.output_lora_rank, config.width,
        std::string(base) + ".wo_b");
    if (!projected) return projected.error();
    result.output = projected.take_value();
    return result;
}

Result<AttentionResult> attention_prefill(const DeepseekV4ReferenceGraph& graph,
                                          std::string_view block_prefix,
                                          std::size_t ratio,
                                          std::span<const float> input,
                                          std::size_t token_count,
                                          DeepseekV4ReferenceAttentionState* state = nullptr) {
    const auto& config = graph.config;
    const std::string base = std::string(block_prefix) + ".attn";
    auto query_rank = linear_tensor(graph, base + ".wq_a.weight", input,
                                    token_count, config.width,
                                    config.query_lora_rank, base + ".wq_a");
    if (!query_rank) return query_rank.error();
    auto query_norm_weight = float_tensor(graph, base + ".q_norm.weight",
                                          {config.query_lora_rank});
    if (!query_norm_weight) return query_norm_weight.error();
    auto normalized_query_rank = deepseek_v4_rms_norm(
        query_rank.value(), query_norm_weight.value(), token_count,
        config.query_lora_rank, config.norm_epsilon);
    if (!normalized_query_rank) return normalized_query_rank.error();
    auto query = linear_tensor(
        graph, base + ".wq_b.weight", normalized_query_rank.value(), token_count,
        config.query_lora_rank, config.head_count * config.head_width,
        base + ".wq_b");
    if (!query) return query.error();
    auto query_normalized = normalize_attention_queries(query.value(), token_count,
                                                        config.head_count,
                                                        config.head_width,
                                                        config.norm_epsilon);
    if (!query_normalized) return query_normalized.error();
    const std::size_t original_context = ratio == 0U ? 0U : config.original_context;
    const float theta = ratio == 0U ? config.rope_theta : config.compression_rope_theta;
    auto query_rope = apply_rope_tail(query.value(), token_count, config.head_count,
                                      config.head_width, config.rope_width, theta,
                                      original_context, config.rope_factor,
                                      config.beta_fast, config.beta_slow, 1U);
    if (!query_rope) return query_rope.error();

    auto key_value = linear_tensor(graph, base + ".wkv.weight", input,
                                   token_count, config.width, config.head_width,
                                   base + ".wkv");
    if (!key_value) return key_value.error();
    auto kv_norm_weight = float_tensor(graph, base + ".kv_norm.weight",
                                       {config.head_width});
    if (!kv_norm_weight) return kv_norm_weight.error();
    auto normalized_kv = deepseek_v4_rms_norm(key_value.value(), kv_norm_weight.value(),
                                              token_count, config.head_width,
                                              config.norm_epsilon);
    if (!normalized_kv) return normalized_kv.error();
    key_value = normalized_kv.take_value();
    auto kv_rope = apply_rope_tail(key_value.value(), token_count, 1U, config.head_width,
                                   config.rope_width, theta, original_context,
                                   config.rope_factor, config.beta_fast,
                                   config.beta_slow, 1U);
    if (!kv_rope) return kv_rope.error();
    if (state != nullptr) {
        auto window_size = bounded_product({config.window_size, config.head_width},
                                           base + " window state");
        if (!window_size) return window_size.error();
        state->window_key_values.assign(window_size.value(), 0.0F);
        state->compressor = {};
        state->indexer = {};
        for (std::size_t token = 0U; token < token_count; ++token) {
            const std::size_t destination = token % config.window_size;
            std::copy_n(key_value.value().begin() +
                            static_cast<std::ptrdiff_t>(token * config.head_width),
                        config.head_width,
                        state->window_key_values.begin() +
                            static_cast<std::ptrdiff_t>(destination * config.head_width));
        }
    }

    auto window = window_indices(token_count, config.window_size);
    if (!window) return window.error();
    const std::size_t window_selected = std::min(token_count, config.window_size);
    std::vector<std::int32_t> compressed_indices;
    std::size_t compressed_selected = 0U;
    if (ratio != 0U) {
        const std::size_t compressed_count = token_count / ratio;
        if (ratio == 4U) {
            auto learned = indexer_indices(graph, base + ".indexer", input,
                                           normalized_query_rank.value(), token_count,
                                           ratio, compressed_count,
                                           state == nullptr ? nullptr : &state->indexer);
            if (!learned) return learned.error();
            compressed_indices = learned.take_value();
            compressed_selected = std::min(config.index_top_k, compressed_count);
        } else {
            compressed_selected = compressed_count;
            auto compressed_index_count = bounded_product(
                {token_count, compressed_selected}, base + " compressed indices");
            if (!compressed_index_count) return compressed_index_count.error();
            compressed_indices.assign(compressed_index_count.value(), -1);
            for (std::size_t token = 0U; token < token_count; ++token) {
                const std::size_t allowed = (token + 1U) / ratio;
                for (std::size_t slot = 0U; slot < compressed_selected; ++slot) {
                    if (slot < allowed) {
                        compressed_indices[token * compressed_selected + slot] =
                            static_cast<std::int32_t>(token_count + slot);
                    }
                }
            }
        }
        auto compressed = compress_prefill(graph, base + ".compressor", input,
                                            token_count, ratio, config.head_width,
                                            config.rope_width,
                                            state == nullptr ? nullptr : &state->compressor);
        if (!compressed) return compressed.error();
        key_value.value().insert(key_value.value().end(), compressed.value().values.begin(),
                                 compressed.value().values.end());
    }
    const std::size_t selected_per_token = window_selected + compressed_selected;
    auto index_count = bounded_product({token_count, selected_per_token},
                                       base + " attention indices");
    if (!index_count) return index_count.error();
    std::vector<std::int32_t> indices(index_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        std::copy_n(window.value().begin() + static_cast<std::ptrdiff_t>(token * window_selected),
                    window_selected,
                    indices.begin() +
                        static_cast<std::ptrdiff_t>(token * selected_per_token));
        if (compressed_selected != 0U) {
            std::copy_n(compressed_indices.begin() +
                            static_cast<std::ptrdiff_t>(token * compressed_selected),
                        compressed_selected,
                        indices.begin() + static_cast<std::ptrdiff_t>(
                            token * selected_per_token + window_selected));
        }
    }
    return finish_attention(graph, base, query.value(), key_value.value(),
                            std::move(indices), token_count, selected_per_token,
                            theta, original_context, 0U);
}

Result<AttentionResult> attention_incremental(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view block_prefix,
    std::size_t ratio,
    std::span<const float> input,
    std::size_t start_position,
    DeepseekV4ReferenceAttentionState& state) {
    const auto& config = graph.config;
    const std::string base = std::string(block_prefix) + ".attn";
    auto expected_window = bounded_product({config.window_size, config.head_width},
                                           base + " window state");
    if (!expected_window) return expected_window.error();
    if (input.size() != config.width ||
        state.window_key_values.size() != expected_window.value()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek incremental attention state is malformed", base);
    }
    auto query_rank = linear_tensor(graph, base + ".wq_a.weight", input, 1U,
                                    config.width, config.query_lora_rank,
                                    base + ".wq_a incremental");
    if (!query_rank) return query_rank.error();
    auto query_norm_weight = float_tensor(graph, base + ".q_norm.weight",
                                          {config.query_lora_rank});
    if (!query_norm_weight) return query_norm_weight.error();
    auto normalized_query_rank = deepseek_v4_rms_norm(
        query_rank.value(), query_norm_weight.value(), 1U, config.query_lora_rank,
        config.norm_epsilon);
    if (!normalized_query_rank) return normalized_query_rank.error();
    auto query = linear_tensor(
        graph, base + ".wq_b.weight", normalized_query_rank.value(), 1U,
        config.query_lora_rank, config.head_count * config.head_width,
        base + ".wq_b incremental");
    if (!query) return query.error();
    auto query_normalized = normalize_attention_queries(
        query.value(), 1U, config.head_count, config.head_width, config.norm_epsilon);
    if (!query_normalized) return query_normalized.error();
    const std::size_t original_context = ratio == 0U ? 0U : config.original_context;
    const float theta = ratio == 0U ? config.rope_theta : config.compression_rope_theta;
    auto query_rope = apply_rope_tail(
        query.value(), 1U, config.head_count, config.head_width, config.rope_width,
        theta, original_context, config.rope_factor, config.beta_fast, config.beta_slow,
        1U, false, start_position);
    if (!query_rope) return query_rope.error();

    auto key_value = linear_tensor(graph, base + ".wkv.weight", input, 1U,
                                   config.width, config.head_width,
                                   base + ".wkv incremental");
    if (!key_value) return key_value.error();
    auto kv_norm_weight = float_tensor(graph, base + ".kv_norm.weight",
                                       {config.head_width});
    if (!kv_norm_weight) return kv_norm_weight.error();
    auto normalized_kv = deepseek_v4_rms_norm(
        key_value.value(), kv_norm_weight.value(), 1U, config.head_width,
        config.norm_epsilon);
    if (!normalized_kv) return normalized_kv.error();
    key_value = normalized_kv.take_value();
    auto kv_rope = apply_rope_tail(
        key_value.value(), 1U, 1U, config.head_width, config.rope_width, theta,
        original_context, config.rope_factor, config.beta_fast, config.beta_slow,
        1U, false, start_position);
    if (!kv_rope) return kv_rope.error();
    const std::size_t window_slot = start_position % config.window_size;
    std::copy_n(key_value.value().begin(), config.head_width,
                state.window_key_values.begin() +
                    static_cast<std::ptrdiff_t>(window_slot * config.head_width));

    std::vector<std::int32_t> indices(config.window_size, -1);
    if (start_position >= config.window_size - 1U) {
        std::size_t slot = 0U;
        for (std::size_t index = window_slot + 1U; index < config.window_size; ++index) {
            indices[slot++] = static_cast<std::int32_t>(index);
        }
        for (std::size_t index = 0U; index <= window_slot; ++index) {
            indices[slot++] = static_cast<std::int32_t>(index);
        }
    } else {
        for (std::size_t index = 0U; index <= start_position; ++index) {
            indices[index] = static_cast<std::int32_t>(index);
        }
    }

    std::vector<std::int32_t> compressed_indices;
    if (ratio != 0U) {
        if (ratio == 4U) {
            auto learned = indexer_indices_incremental(
                graph, base + ".indexer", input, normalized_query_rank.value(),
                start_position, ratio, state.indexer);
            if (!learned) return learned.error();
            compressed_indices = learned.take_value();
        } else {
            const std::size_t compressed_count = (start_position + 1U) / ratio;
            compressed_indices.resize(compressed_count);
            for (std::size_t index = 0U; index < compressed_count; ++index) {
                compressed_indices[index] =
                    static_cast<std::int32_t>(config.window_size + index);
            }
        }
        auto compressed = compress_incremental(
            graph, base + ".compressor", input, start_position, ratio,
            config.head_width, config.rope_width, state.compressor);
        if (!compressed) return compressed.error();
        const std::size_t compressed_count = (start_position + 1U) / ratio;
        auto expected_compressed = bounded_product(
            {compressed_count, config.head_width}, base + " compressed cache");
        if (!expected_compressed) return expected_compressed.error();
        if (state.compressor.compressed.size() != expected_compressed.value()) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek compressed attention cache length is malformed",
                              base);
        }
        indices.insert(indices.end(), compressed_indices.begin(), compressed_indices.end());
    }
    std::vector<float> cache = state.window_key_values;
    cache.insert(cache.end(), state.compressor.compressed.begin(),
                 state.compressor.compressed.end());
    return finish_attention(graph, base, query.value(), cache, std::move(indices), 1U,
                            config.window_size + compressed_indices.size(), theta,
                            original_context, start_position);
}

Result<std::vector<float>> execute_expert(const DeepseekV4ReferenceGraph& graph,
                                          std::string_view prefix,
                                          std::span<const float> input,
                                          std::span<const float> routing,
                                          std::size_t tokens,
                                          float swiglu_limit,
                                          const DeepseekV4ResolvedExpert* resolved = nullptr) {
    const auto& config = graph.config;
    const std::string base(prefix);
    if (tokens == 0U || !std::isfinite(swiglu_limit) || swiglu_limit < 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek graph expert dimensions or clamp are invalid", base);
    }
    if (!routing.empty() && routing.size() != tokens) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek graph expert routing shape does not match", base);
    }
    auto routing_finite = finite_values(routing, base + " routing");
    if (!routing_finite) return routing_finite.error();
    const auto expert_linear = [&](const DeepseekV4PackedMatrix* matrix,
                                   std::string_view suffix,
                                   std::span<const float> values,
                                   std::size_t input_columns,
                                   std::size_t output_rows) {
        const std::string name = base + std::string(suffix) + ".weight";
        if (matrix != nullptr) {
            return linear_packed_matrix(*matrix, name, values, tokens,
                                        input_columns, output_rows,
                                        base + std::string(suffix),
                                        graph.config.cpu_threads);
        }
        return linear_tensor(graph, name, values, tokens, input_columns,
                             output_rows, base + std::string(suffix));
    };
    auto gate = expert_linear(resolved == nullptr ? nullptr : &resolved->w1,
                              ".w1", input, config.width,
                              config.intermediate_width);
    if (!gate) return gate.error();
    auto up = expert_linear(resolved == nullptr ? nullptr : &resolved->w3,
                            ".w3", input, config.width,
                            config.intermediate_width);
    if (!up) return up.error();
    auto activated_count = bounded_product(
        {tokens, config.intermediate_width}, base + " activated values");
    if (!activated_count) return activated_count.error();
    std::vector<float> activated(activated_count.value());
    for (std::size_t token = 0U; token < tokens; ++token) {
        for (std::size_t column = 0U; column < config.intermediate_width; ++column) {
            double gate_value = static_cast<double>(
                gate.value()[token * config.intermediate_width + column]);
            double up_value = static_cast<double>(
                up.value()[token * config.intermediate_width + column]);
            if (swiglu_limit > 0.0F) {
                gate_value = std::min(gate_value,
                                      static_cast<double>(swiglu_limit));
                up_value = std::clamp(up_value,
                                      -static_cast<double>(swiglu_limit),
                                      static_cast<double>(swiglu_limit));
            }
            double value = gate_value / (1.0 + std::exp(-gate_value)) * up_value;
            if (!routing.empty()) value *= static_cast<double>(routing[token]);
            activated[token * config.intermediate_width + column] =
                static_cast<float>(value);
        }
    }
    return expert_linear(resolved == nullptr ? nullptr : &resolved->w2,
                         ".w2", activated, config.intermediate_width,
                         config.width);
}

struct MoeResult {
    std::vector<float> output;
    DeepseekV4RouteResult route;
    std::vector<DeepseekV4LayerTrace::ExpertAccess> expert_accesses;
};

Result<MoeResult> moe_forward(const DeepseekV4ReferenceGraph& graph,
                              std::string_view block_prefix,
                              std::size_t clamp_layer,
                              bool hash_route,
                              std::span<const float> input,
                              std::span<const std::uint32_t> token_ids,
                              std::size_t token_count) {
    const auto& config = graph.config;
    const std::string base = std::string(block_prefix) + ".ffn";
    const bool has_schedules = !config.routed_expert_swiglu_limits.empty() ||
                               !config.shared_expert_swiglu_limits.empty();
    const std::size_t expected_schedule_count =
        config.layer_count + config.mtp_layer_count;
    if (has_schedules &&
        (config.routed_expert_swiglu_limits.size() != expected_schedule_count ||
         config.shared_expert_swiglu_limits.size() != expected_schedule_count ||
         clamp_layer >= expected_schedule_count)) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek graph SwiGLU clamp schedule is malformed", base);
    }
    const float routed_swiglu_limit = has_schedules
        ? config.routed_expert_swiglu_limits[clamp_layer]
        : config.swiglu_limit;
    const float shared_swiglu_limit = has_schedules
        ? config.shared_expert_swiglu_limits[clamp_layer]
        : config.swiglu_limit;
    auto gate_logits = linear_tensor(graph, base + ".gate.weight", input,
                                     token_count, config.width,
                                     config.expert_count, base + ".gate");
    if (!gate_logits) return gate_logits.error();
    Result<DeepseekV4RouteResult> route = make_error(
        ErrorCode::internal_error, "DeepSeek route was not initialized");
    if (hash_route) {
        auto table = integer_tensor(graph, base + ".gate.tid2eid",
                                    {config.vocabulary_size, config.experts_per_token});
        if (!table) return table.error();
        route = deepseek_v4_route_hash_logits(
            gate_logits.value(), token_ids, table.value(), token_count,
            config.vocabulary_size, config.expert_count,
            config.experts_per_token, config.route_scale);
    } else {
        auto bias = float_tensor(graph, base + ".gate.bias", {config.expert_count});
        if (!bias) return bias.error();
        route = deepseek_v4_route_learned_logits(
            gate_logits.value(), bias.value(), token_count, config.expert_count,
            config.experts_per_token, config.route_scale);
    }
    if (!route) return route.error();
    std::vector<std::uint32_t> requested_experts;
    try {
        requested_experts.reserve(std::min(route.value().indices.size(),
                                           config.expert_count));
        for (const auto expert_id : route.value().indices) {
            if (expert_id >= config.expert_count) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek router selected an out-of-range expert",
                                  base);
            }
            if (std::find(requested_experts.begin(), requested_experts.end(),
                          expert_id) == requested_experts.end()) {
                requested_experts.push_back(expert_id);
            }
        }
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded selected-expert set",
                          base);
    }

    std::vector<DeepseekV4ResolvedExpert> resolved_experts;
    std::unique_ptr<DeepseekV4ExpertBatch> pending_batch;
    std::vector<std::pair<std::uint32_t, DeepseekV4LayerTrace::ExpertAccess>>
        recorded_accesses;
    std::vector<std::pair<std::chrono::steady_clock::time_point,
                          std::chrono::steady_clock::time_point>>
        compute_windows;
    const auto provider = graph.expert_providers.find(base);
    if (provider != graph.expert_providers.end()) {
        if ((!provider->second.resolve && !provider->second.begin) ||
            provider->second.max_experts_per_request == 0U ||
            requested_experts.size() > provider->second.max_experts_per_request) {
            return make_error(ErrorCode::resource_exhausted,
                              "Selected-expert request exceeds its configured bound",
                              base);
        }
        try {
            recorded_accesses.reserve(requested_experts.size());
            compute_windows.reserve(requested_experts.size());
            if (provider->second.begin) {
                auto begun = provider->second.begin(base, requested_experts);
                if (!begun) return begun.error();
                pending_batch = begun.take_value();
                if (pending_batch == nullptr) {
                    return make_error(ErrorCode::integrity_error,
                                      "Selected-expert provider returned an empty batch",
                                      base);
                }
            } else {
                auto provided = provider->second.resolve(base, requested_experts);
                if (!provided) return provided.error();
                resolved_experts = provided.take_value();
                if (resolved_experts.size() != requested_experts.size()) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "Selected-expert provider returned the wrong bundle count",
                        base);
                }
                for (const auto expert_id : requested_experts) {
                    if (std::count_if(
                            resolved_experts.begin(), resolved_experts.end(),
                            [expert_id](const DeepseekV4ResolvedExpert& expert) {
                                return expert.expert_id == expert_id;
                            }) != 1) {
                        return make_error(
                            ErrorCode::integrity_error,
                            "Selected-expert provider returned duplicate identities",
                            base + ".experts." + std::to_string(expert_id));
                    }
                }
            }
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted,
                              "Selected-expert provider exhausted memory",
                              base);
        } catch (const std::exception& exception) {
            return make_error(ErrorCode::internal_error,
                              "Selected-expert provider failed",
                              exception.what());
        } catch (...) {
            return make_error(ErrorCode::internal_error,
                              "Selected-expert provider failed",
                              base);
        }
    }

    const auto validate_resolved = [&](const DeepseekV4ResolvedExpert& expert,
                                       std::uint32_t expected_id) -> Result<void> {
        if (expert.expert_id != expected_id || expert.resident_bytes == 0U ||
            expert.logical_bytes_read > expert.resident_bytes ||
            (expert.cache_hit && expert.logical_bytes_read != 0U) ||
            !std::isfinite(expert.load_queue_ms) || expert.load_queue_ms < 0.0 ||
            !std::isfinite(expert.load_service_ms) || expert.load_service_ms < 0.0 ||
            !std::isfinite(expert.load_total_ms) || expert.load_total_ms < 0.0 ||
            expert.load_total_ms + 0.01 < expert.load_queue_ms ||
            expert.load_total_ms + 0.01 < expert.load_service_ms ||
            (expert.asynchronous_load &&
             (expert.load_request_sequence == 0U ||
              expert.load_service_started_at ==
                  std::chrono::steady_clock::time_point{} ||
              expert.load_completed_at ==
                  std::chrono::steady_clock::time_point{} ||
              expert.load_completed_at < expert.load_service_started_at))) {
            return make_error(ErrorCode::integrity_error,
                              "Selected-expert provider returned malformed telemetry",
                              base + ".experts." + std::to_string(expected_id));
        }
        return {};
    };

    auto output_count = bounded_product({token_count, config.width}, base + " output");
    if (!output_count) return output_count.error();
    std::vector<float> output(output_count.value(), 0.0F);
    for (std::size_t expert = 0U; expert < config.expert_count; ++expert) {
        const bool selected =
            std::find(requested_experts.begin(), requested_experts.end(), expert) !=
            requested_experts.end();
        if (provider != graph.expert_providers.end() && !selected) continue;

        std::optional<DeepseekV4ResolvedExpert> acquired_expert;
        const DeepseekV4ResolvedExpert* resolved = nullptr;
        if (provider != graph.expert_providers.end()) {
            const auto wait_started = std::chrono::steady_clock::now();
            if (pending_batch != nullptr) {
                auto acquired = [&]() -> Result<DeepseekV4ResolvedExpert> {
                    try {
                        return pending_batch->acquire(
                            static_cast<std::uint32_t>(expert));
                    } catch (const std::bad_alloc&) {
                        return make_error(ErrorCode::resource_exhausted,
                                          "Selected-expert acquisition exhausted memory",
                                          std::to_string(expert));
                    } catch (const std::exception& exception) {
                        return make_error(ErrorCode::internal_error,
                                          "Selected-expert acquisition failed",
                                          exception.what());
                    } catch (...) {
                        return make_error(ErrorCode::internal_error,
                                          "Selected-expert acquisition failed",
                                          std::to_string(expert));
                    }
                }();
                if (!acquired) return acquired.error();
                acquired_expert = acquired.take_value();
                resolved = &*acquired_expert;
            } else {
                const auto found = std::find_if(
                    resolved_experts.begin(), resolved_experts.end(),
                    [expert](const DeepseekV4ResolvedExpert& value) {
                        return value.expert_id == expert;
                    });
                if (found == resolved_experts.end()) {
                    return make_error(ErrorCode::integrity_error,
                                      "Selected-expert provider omitted an identity",
                                      std::to_string(expert));
                }
                resolved = &*found;
            }
            const auto wait_completed = std::chrono::steady_clock::now();
            auto valid = validate_resolved(
                *resolved, static_cast<std::uint32_t>(expert));
            if (!valid) return valid.error();

            DeepseekV4LayerTrace::ExpertAccess access{
                resolved->expert_id, resolved->resident_bytes,
                resolved->logical_bytes_read, resolved->cache_hit};
            access.asynchronous_load = resolved->asynchronous_load;
            access.load_request_sequence = resolved->load_request_sequence;
            access.load_queue_ms = resolved->load_queue_ms;
            access.load_service_ms = resolved->load_service_ms;
            access.load_total_ms = resolved->load_total_ms;
            access.acquire_wait_ms = std::chrono::duration<double, std::milli>(
                                         wait_completed - wait_started)
                                         .count();
            if (resolved->asynchronous_load) {
                for (const auto& window : compute_windows) {
                    const auto overlap_start = std::max(
                        resolved->load_service_started_at, window.first);
                    const auto overlap_end = std::min(
                        resolved->load_completed_at, window.second);
                    if (overlap_end > overlap_start) {
                        access.io_compute_overlap_ms +=
                            std::chrono::duration<double, std::milli>(
                                overlap_end - overlap_start)
                                .count();
                    }
                }
            }
            recorded_accesses.emplace_back(
                static_cast<std::uint32_t>(expert), std::move(access));
        }
        const auto compute_started = std::chrono::steady_clock::now();
        for (std::size_t token = 0U; token < token_count; ++token) {
            for (std::size_t slot = 0U; slot < config.experts_per_token; ++slot) {
                if (route.value().indices[token * config.experts_per_token + slot] != expert) {
                    continue;
                }
                const std::span<const float> token_input(
                    input.data() + static_cast<std::ptrdiff_t>(token * config.width),
                    config.width);
                const float route_weight =
                    route.value().weights[token * config.experts_per_token + slot];
                auto expert_output = execute_expert(
                    graph, base + ".experts." + std::to_string(expert), token_input,
                    std::span<const float>(&route_weight, 1U), 1U,
                    routed_swiglu_limit, resolved);
                if (!expert_output) return expert_output.error();
                for (std::size_t column = 0U; column < config.width; ++column) {
                    output[token * config.width + column] += expert_output.value()[column];
                }
            }
        }
        const auto compute_completed = std::chrono::steady_clock::now();
        if (provider != graph.expert_providers.end()) {
            recorded_accesses.back().second.compute_ms =
                std::chrono::duration<double, std::milli>(
                    compute_completed - compute_started)
                    .count();
            compute_windows.emplace_back(compute_started, compute_completed);
        }
    }

    std::vector<DeepseekV4LayerTrace::ExpertAccess> expert_accesses;
    try {
        if (provider != graph.expert_providers.end()) {
            expert_accesses.reserve(requested_experts.size());
            for (const auto expert_id : requested_experts) {
                const auto access = std::find_if(
                    recorded_accesses.begin(), recorded_accesses.end(),
                    [expert_id](const auto& candidate) {
                        return candidate.first == expert_id;
                    });
                if (access == recorded_accesses.end()) {
                    return make_error(ErrorCode::integrity_error,
                                      "Selected-expert telemetry omitted an identity",
                                      std::to_string(expert_id));
                }
                expert_accesses.push_back(access->second);
            }
        }
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to retain selected-expert telemetry",
                          base);
    }
    auto shared = execute_expert(graph, base + ".shared_experts", input, {}, token_count,
                                 shared_swiglu_limit);
    if (!shared) return shared.error();
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] += shared.value()[index];
    }
    auto finite = finite_values(output, base + " output");
    if (!finite) return finite.error();
    MoeResult result;
    result.output = std::move(output);
    result.route = route.take_value();
    result.expert_accesses = std::move(expert_accesses);
    return result;
}

Result<DeepseekV4HyperConnectionPreResult> hc_pre(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view prefix,
    std::span<const float> input,
    std::size_t token_count) {
    const auto& config = graph.config;
    const std::string base(prefix);
    const std::size_t mix_width = (2U + config.hyper_multiplier) * config.hyper_multiplier;
    auto scale = float_tensor(graph, base + "_scale", {3U});
    if (!scale) return scale.error();
    auto bias = float_tensor(graph, base + "_base", {mix_width});
    if (!bias) return bias.error();
    const std::string function_name = base + "_fn";
    if (!graph.packed_matrices.contains(function_name)) {
        auto function = float_tensor(
            graph, function_name,
            {mix_width, config.hyper_multiplier * config.width});
        if (!function) return function.error();
        return deepseek_v4_hc_pre(input, function.value(), scale.value(), bias.value(),
                                  token_count, config.hyper_multiplier, config.width,
                                  config.sinkhorn_iterations, config.norm_epsilon,
                                  config.hyper_epsilon);
    }
    const std::size_t flattened_width = config.hyper_multiplier * config.width;
    auto input_count = bounded_product({token_count, flattened_width}, base + " input");
    if (!input_count) return input_count.error();
    if (input.size() != input_count.value() || !std::isfinite(config.norm_epsilon) ||
        config.norm_epsilon <= 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek packed Hyper-Connection input is invalid", base);
    }
    auto input_finite = finite_values(input, base + " input");
    if (!input_finite) return input_finite.error();
    std::vector<float> normalized(input.size());
    for (std::size_t token = 0U; token < token_count; ++token) {
        double squares = 0.0;
        for (std::size_t column = 0U; column < flattened_width; ++column) {
            const double value = static_cast<double>(
                input[token * flattened_width + column]);
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(
            squares / static_cast<double>(flattened_width) +
            static_cast<double>(config.norm_epsilon));
        for (std::size_t column = 0U; column < flattened_width; ++column) {
            normalized[token * flattened_width + column] = static_cast<float>(
                static_cast<double>(input[token * flattened_width + column]) * inverse);
        }
    }
    auto mixes = linear_tensor(graph, function_name, normalized, token_count,
                               flattened_width, mix_width, base + " mixes");
    if (!mixes) return mixes.error();
    auto split = deepseek_v4_hc_split_sinkhorn(
        mixes.value(), scale.value(), bias.value(), token_count,
        config.hyper_multiplier, config.sinkhorn_iterations, config.hyper_epsilon);
    if (!split) return split.error();
    DeepseekV4HyperConnectionPreResult result;
    result.width = config.width;
    result.split = split.take_value();
    auto reduced_count = bounded_product({token_count, config.width}, base + " reduced");
    if (!reduced_count) return reduced_count.error();
    result.reduced.resize(reduced_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t column = 0U; column < config.width; ++column) {
            double value = 0.0;
            for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
                value += static_cast<double>(
                             result.split.pre[token * config.hyper_multiplier + copy]) *
                         static_cast<double>(input[
                             (token * config.hyper_multiplier + copy) * config.width +
                             column]);
            }
            result.reduced[token * config.width + column] = static_cast<float>(value);
        }
    }
    auto reduced_finite = finite_values(result.reduced, base + " reduced");
    if (!reduced_finite) return reduced_finite.error();
    return result;
}

struct BlockResult {
    std::vector<float> hidden;
    DeepseekV4LayerTrace trace;
};

Result<BlockResult> block_prefill(const DeepseekV4ReferenceGraph& graph,
                                  std::string_view block_prefix,
                                  std::size_t trace_layer,
                                  std::size_t compression_ratio,
                                   bool hash_route,
                                   std::span<const float> input,
                                   std::span<const std::uint32_t> token_ids,
                                   DeepseekV4ReferenceAttentionState* state = nullptr) {
    const auto& config = graph.config;
    const std::size_t token_count = token_ids.size();
    const std::string base(block_prefix);
    const std::vector<float> residual_attention(input.begin(), input.end());
    auto attention_mix = hc_pre(graph, base + ".hc_attn", input, token_count);
    if (!attention_mix) return attention_mix.error();
    auto attention_norm_weight = float_tensor(graph, base + ".attn_norm.weight",
                                              {config.width});
    if (!attention_norm_weight) return attention_norm_weight.error();
    auto normalized_attention = deepseek_v4_rms_norm(
        attention_mix.value().reduced, attention_norm_weight.value(), token_count,
        config.width, config.norm_epsilon);
    if (!normalized_attention) return normalized_attention.error();
    auto attention = attention_prefill(graph, base, compression_ratio,
                                       normalized_attention.value(), token_count, state);
    if (!attention) return attention.error();
    auto attention_post = deepseek_v4_hc_post(
        attention.value().output, residual_attention, attention_mix.value().split,
        config.width);
    if (!attention_post) return attention_post.error();

    const auto residual_ffn = attention_post.value();
    auto ffn_mix = hc_pre(graph, base + ".hc_ffn", attention_post.value(), token_count);
    if (!ffn_mix) return ffn_mix.error();
    auto ffn_norm_weight = float_tensor(graph, base + ".ffn_norm.weight", {config.width});
    if (!ffn_norm_weight) return ffn_norm_weight.error();
    auto normalized_ffn = deepseek_v4_rms_norm(
        ffn_mix.value().reduced, ffn_norm_weight.value(), token_count,
        config.width, config.norm_epsilon);
    if (!normalized_ffn) return normalized_ffn.error();
    const std::size_t clamp_layer = std::string_view(block_prefix).starts_with("mtp.")
        ? config.layer_count + trace_layer
        : trace_layer;
    auto moe = moe_forward(graph, base, clamp_layer, hash_route,
                           normalized_ffn.value(), token_ids,
                           token_count);
    if (!moe) return moe.error();
    auto ffn_post = deepseek_v4_hc_post(moe.value().output, residual_ffn,
                                        ffn_mix.value().split, config.width);
    if (!ffn_post) return ffn_post.error();

    BlockResult result;
    result.hidden = ffn_post.take_value();
    result.trace.layer = trace_layer;
    result.trace.route = std::move(moe.value().route);
    result.trace.expert_accesses = std::move(moe.value().expert_accesses);
    result.trace.attention_indices = std::move(attention.value().indices);
    result.trace.attention_selected_per_token = attention.value().selected_per_token;
    return result;
}

Result<BlockResult> block_incremental(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view block_prefix,
    std::size_t trace_layer,
    std::size_t compression_ratio,
    bool hash_route,
    std::span<const float> input,
    std::uint32_t token_id,
    std::size_t start_position,
    DeepseekV4ReferenceAttentionState& state) {
    const auto& config = graph.config;
    const std::string base(block_prefix);
    const std::vector<float> residual_attention(input.begin(), input.end());
    auto attention_mix = hc_pre(graph, base + ".hc_attn", input, 1U);
    if (!attention_mix) return attention_mix.error();
    auto attention_norm_weight = float_tensor(graph, base + ".attn_norm.weight",
                                              {config.width});
    if (!attention_norm_weight) return attention_norm_weight.error();
    auto normalized_attention = deepseek_v4_rms_norm(
        attention_mix.value().reduced, attention_norm_weight.value(), 1U,
        config.width, config.norm_epsilon);
    if (!normalized_attention) return normalized_attention.error();
    auto attention = attention_incremental(graph, base, compression_ratio,
                                           normalized_attention.value(), start_position,
                                           state);
    if (!attention) return attention.error();
    auto attention_post = deepseek_v4_hc_post(
        attention.value().output, residual_attention, attention_mix.value().split,
        config.width);
    if (!attention_post) return attention_post.error();

    const auto residual_ffn = attention_post.value();
    auto ffn_mix = hc_pre(graph, base + ".hc_ffn", attention_post.value(), 1U);
    if (!ffn_mix) return ffn_mix.error();
    auto ffn_norm_weight = float_tensor(graph, base + ".ffn_norm.weight", {config.width});
    if (!ffn_norm_weight) return ffn_norm_weight.error();
    auto normalized_ffn = deepseek_v4_rms_norm(
        ffn_mix.value().reduced, ffn_norm_weight.value(), 1U,
        config.width, config.norm_epsilon);
    if (!normalized_ffn) return normalized_ffn.error();
    const std::array<std::uint32_t, 1U> token_ids{token_id};
    const std::size_t clamp_layer = std::string_view(block_prefix).starts_with("mtp.")
        ? config.layer_count + trace_layer
        : trace_layer;
    auto moe = moe_forward(graph, base, clamp_layer, hash_route,
                           normalized_ffn.value(), token_ids, 1U);
    if (!moe) return moe.error();
    auto ffn_post = deepseek_v4_hc_post(moe.value().output, residual_ffn,
                                        ffn_mix.value().split, config.width);
    if (!ffn_post) return ffn_post.error();

    BlockResult result;
    result.hidden = ffn_post.take_value();
    result.trace.layer = trace_layer;
    result.trace.route = std::move(moe.value().route);
    result.trace.expert_accesses = std::move(moe.value().expert_accesses);
    result.trace.attention_indices = std::move(attention.value().indices);
    result.trace.attention_selected_per_token = attention.value().selected_per_token;
    return result;
}

Result<std::vector<float>> hc_head(const DeepseekV4ReferenceGraph& graph,
                                   std::span<const float> input,
                                   std::size_t token_count,
                                   std::string_view tensor_prefix) {
    const auto& config = graph.config;
    const std::size_t flattened_width = config.hyper_multiplier * config.width;
    const std::string prefix(tensor_prefix);
    auto scale = float_tensor(graph, prefix + "hc_head_scale", {1U});
    if (!scale) return scale.error();
    auto base = float_tensor(graph, prefix + "hc_head_base", {config.hyper_multiplier});
    if (!base) return base.error();
    auto mixes = linear_tensor(graph, prefix + "hc_head_fn", input, token_count,
                               flattened_width, config.hyper_multiplier,
                               "DeepSeek HC head");
    if (!mixes) return mixes.error();
    auto output_count = bounded_product({token_count, config.width}, "DeepSeek HC head output");
    if (!output_count) return output_count.error();
    std::vector<float> output(output_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        double squares = 0.0;
        for (std::size_t column = 0U; column < flattened_width; ++column) {
            const double value = static_cast<double>(input[token * flattened_width + column]);
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(
            squares / static_cast<double>(flattened_width) +
            static_cast<double>(config.norm_epsilon));
        for (std::size_t column = 0U; column < config.width; ++column) {
            double value = 0.0;
            for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
                const double mixed = static_cast<double>(
                    mixes.value()[token * config.hyper_multiplier + copy]) * inverse;
                const double sigmoid = 1.0 / (1.0 + std::exp(-(
                    mixed * static_cast<double>(scale.value()[0]) +
                    static_cast<double>(base.value()[copy]))));
                value += (sigmoid + static_cast<double>(config.hyper_epsilon)) *
                         static_cast<double>(input[
                             (token * config.hyper_multiplier + copy) * config.width + column]);
            }
            output[token * config.width + column] = static_cast<float>(value);
        }
    }
    return output;
}

Result<std::vector<float>> final_logits(const DeepseekV4ReferenceGraph& graph,
                                        std::span<const float> hidden,
                                        std::size_t token_count,
                                        std::string_view tensor_prefix) {
    const auto& config = graph.config;
    const std::string prefix(tensor_prefix);
    auto reduced = hc_head(graph, hidden, token_count, prefix);
    if (!reduced) return reduced.error();
    auto norm_weight = float_tensor(graph, prefix + "norm.weight", {config.width});
    if (!norm_weight) return norm_weight.error();
    auto normalized = deepseek_v4_rms_norm(reduced.value(), norm_weight.value(), token_count,
                                           config.width, config.norm_epsilon);
    if (!normalized) return normalized.error();
    const std::span<const float> final_hidden(
        normalized.value().data() +
            static_cast<std::ptrdiff_t>((token_count - 1U) * config.width),
        config.width);
    return linear_tensor(graph, "head.weight", final_hidden, 1U, config.width,
                         config.vocabulary_size, prefix + "LM head");
}

Result<void> validate_graph(const DeepseekV4ReferenceGraph& graph,
                            std::span<const std::uint32_t> token_ids) {
    const auto& config = graph.config;
    if (token_ids.empty() || token_ids.size() > kMaximumReferenceTokens ||
        config.vocabulary_size == 0U || config.width == 0U ||
        config.intermediate_width == 0U || config.layer_count == 0U ||
        config.layer_count > kMaximumReferenceLayers ||
        config.mtp_layer_count > kMaximumReferenceLayers - config.layer_count ||
        config.mtp_layer_count > 1U ||
        config.hash_layer_count > config.layer_count || config.head_count == 0U ||
        config.expert_count == 0U || config.experts_per_token == 0U ||
        config.experts_per_token > config.expert_count || config.query_lora_rank == 0U ||
        config.head_width == 0U || config.rope_width == 0U ||
        config.rope_width > config.head_width || config.rope_width % 2U != 0U ||
        config.output_groups == 0U || config.output_lora_rank == 0U ||
        config.window_size == 0U || config.compression_ratios.size() != config.layer_count ||
        config.mtp_compression_ratios.size() != config.mtp_layer_count ||
        config.index_head_count == 0U || config.index_head_width < config.rope_width ||
        config.index_top_k == 0U || config.hyper_multiplier == 0U ||
        config.hyper_multiplier > 16U || config.sinkhorn_iterations == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference graph configuration is invalid");
    }
    auto attention_width = bounded_product({config.head_count, config.head_width},
                                           "DeepSeek attention width");
    if (!attention_width) return attention_width.error();
    if (attention_width.value() % config.output_groups != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek attention width is not divisible by output groups");
    }
    auto indexer_width = bounded_product({config.index_head_count, config.index_head_width},
                                         "DeepSeek indexer width");
    if (!indexer_width) return indexer_width.error();
    auto hyper_width = bounded_product({config.hyper_multiplier, config.width},
                                       "DeepSeek Hyper-Connection width");
    if (!hyper_width) return hyper_width.error();
    auto output_lora_width = bounded_product({config.output_groups,
                                              config.output_lora_rank},
                                             "DeepSeek output LoRA width");
    if (!output_lora_width) return output_lora_width.error();
    auto overlap_width = bounded_product({2U, config.head_width},
                                         "DeepSeek overlap compressor width");
    if (!overlap_width) return overlap_width.error();
    auto overlap_indexer_width = bounded_product({2U, config.index_head_width},
                                                 "DeepSeek overlap indexer width");
    if (!overlap_indexer_width) return overlap_indexer_width.error();
    for (const auto ratio : config.compression_ratios) {
        if (ratio != 0U && ratio != 4U && ratio != 128U) {
            return make_error(ErrorCode::unsupported,
                              "DeepSeek reference graph compression ratio is unsupported",
                              std::to_string(ratio));
        }
    }
    for (const auto ratio : config.mtp_compression_ratios) {
        if (ratio != 0U && ratio != 4U && ratio != 128U) {
            return make_error(ErrorCode::unsupported,
                              "DeepSeek MTP compression ratio is unsupported",
                              std::to_string(ratio));
        }
    }
    for (const auto value : {config.compression_rope_theta, config.rope_theta,
                             config.rope_factor, config.beta_fast, config.beta_slow,
                             config.norm_epsilon, config.hyper_epsilon,
                             config.route_scale}) {
        if (!std::isfinite(value) || value <= 0.0F) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek reference graph scalar is invalid");
        }
    }
    if (!std::isfinite(config.swiglu_limit) || config.swiglu_limit < 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference graph SwiGLU clamp is invalid");
    }
    const bool has_swiglu_schedules =
        !config.routed_expert_swiglu_limits.empty() ||
        !config.shared_expert_swiglu_limits.empty();
    const std::size_t expected_swiglu_values =
        config.layer_count + config.mtp_layer_count;
    if (has_swiglu_schedules &&
        (config.routed_expert_swiglu_limits.size() != expected_swiglu_values ||
         config.shared_expert_swiglu_limits.size() != expected_swiglu_values)) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference graph SwiGLU schedules are malformed");
    }
    for (const auto& schedule : {std::span<const float>(
                                      config.routed_expert_swiglu_limits),
                                  std::span<const float>(
                                      config.shared_expert_swiglu_limits)}) {
        for (const float value : schedule) {
            if (!std::isfinite(value) || value < 0.0F) {
                return make_error(
                    ErrorCode::invalid_argument,
                    "DeepSeek reference graph SwiGLU schedule value is invalid");
            }
        }
    }
    for (const auto token : token_ids) {
        if (token >= config.vocabulary_size) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek reference graph token exceeds the vocabulary",
                              std::to_string(token));
        }
    }
    std::size_t total = 0U;
    for (const auto& [name, values] : graph.float_tensors) {
        if (values.size() > kMaximumGraphElements - total) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference graph tensor store exceeds its limit", name);
        }
        total += values.size();
        auto finite = finite_values(values, name);
        if (!finite) return finite.error();
    }
    for (const auto& [name, values] : graph.integer_tensors) {
        if (values.size() > kMaximumGraphElements - total) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference graph tensor store exceeds its limit", name);
        }
        total += values.size();
    }
    return {};
}

Result<void> validate_state_budget(const DeepseekV4ReferenceConfig& config,
                                   std::size_t position) {
    std::size_t total = 0U;
    const auto add = [&](std::initializer_list<std::size_t> factors,
                         std::string context) -> Result<void> {
        auto count = bounded_product(factors, std::move(context));
        if (!count) return count.error();
        if (count.value() > kMaximumGraphElements - total) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek incremental state exceeds its element limit");
        }
        total += count.value();
        return {};
    };
    const auto add_attention = [&](std::size_t ratio,
                                   std::string_view context) -> Result<void> {
        auto valid = add({config.window_size, config.head_width},
                         std::string(context) + " window");
        if (!valid) return valid.error();
        if (ratio == 0U) return {};
        const std::size_t coefficient = ratio == 4U ? 2U : 1U;
        valid = add({2U, coefficient, ratio, coefficient, config.head_width},
                    std::string(context) + " compressor state");
        if (!valid) return valid.error();
        valid = add({position / ratio, config.head_width},
                    std::string(context) + " compressed cache");
        if (!valid) return valid.error();
        if (ratio != 4U) return {};
        valid = add({2U, coefficient, ratio, coefficient, config.index_head_width},
                    std::string(context) + " indexer state");
        if (!valid) return valid.error();
        return add({position / ratio, config.index_head_width},
                   std::string(context) + " indexer cache");
    };
    for (std::size_t layer = 0U; layer < config.layer_count; ++layer) {
        auto valid = add_attention(config.compression_ratios[layer],
                                   "main layer " + std::to_string(layer));
        if (!valid) return valid.error();
    }
    for (std::size_t layer = 0U; layer < config.mtp_layer_count; ++layer) {
        auto valid = add_attention(config.mtp_compression_ratios[layer],
                                   "MTP layer " + std::to_string(layer));
        if (!valid) return valid.error();
    }
    return {};
}

Result<void> validate_reference_state(const DeepseekV4ReferenceGraph& graph,
                                      const DeepseekV4ReferenceState& state) {
    const auto& config = graph.config;
    if (state.next_position == 0U || state.next_position > kMaximumReferenceTokens ||
        state.layers.size() != config.layer_count ||
        state.mtp_layers.size() != config.mtp_layer_count) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek incremental state header is malformed");
    }
    std::size_t total = 0U;
    const auto account = [&](std::span<const float> values,
                             std::string_view context,
                             bool allow_negative_infinity = false) -> Result<void> {
        if (values.size() > kMaximumGraphElements - total) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek incremental state exceeds its element limit",
                              std::string(context));
        }
        total += values.size();
        for (const auto value : values) {
            if (std::isfinite(value)) continue;
            if (allow_negative_infinity && value ==
                    -std::numeric_limits<float>::infinity()) {
                continue;
            }
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek incremental state contains an invalid value",
                              std::string(context));
        }
        return {};
    };
    const auto validate_compressor = [&](const DeepseekV4ReferenceCompressorState& compressor,
                                         std::size_t ratio,
                                         std::size_t width,
                                         std::string_view context) -> Result<void> {
        if (ratio == 0U) {
            if (!compressor.key_values.empty() || !compressor.scores.empty() ||
                !compressor.compressed.empty()) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek unused compressor state is not empty",
                                  std::string(context));
            }
            return {};
        }
        const std::size_t coefficient = ratio == 4U ? 2U : 1U;
        auto expected_state = bounded_product(
            {coefficient, ratio, coefficient, width},
            std::string(context) + " compressor state");
        if (!expected_state) return expected_state.error();
        auto expected_cache = bounded_product(
            {state.next_position / ratio, width},
            std::string(context) + " compressed cache");
        if (!expected_cache) return expected_cache.error();
        if (compressor.key_values.size() != expected_state.value() ||
            compressor.scores.size() != expected_state.value() ||
            compressor.compressed.size() != expected_cache.value()) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek compressor state shape is malformed",
                              std::string(context));
        }
        auto valid = account(compressor.key_values, context);
        if (!valid) return valid.error();
        valid = account(compressor.scores, context, true);
        if (!valid) return valid.error();
        return account(compressor.compressed, context);
    };
    const auto validate_attention = [&](const DeepseekV4ReferenceAttentionState& attention,
                                        std::size_t ratio,
                                        std::string_view context) -> Result<void> {
        auto expected_window = bounded_product(
            {config.window_size, config.head_width},
            std::string(context) + " window state");
        if (!expected_window) return expected_window.error();
        if (attention.window_key_values.size() != expected_window.value()) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek window state shape is malformed",
                              std::string(context));
        }
        auto valid = account(attention.window_key_values, context);
        if (!valid) return valid.error();
        valid = validate_compressor(attention.compressor, ratio, config.head_width,
                                    context);
        if (!valid) return valid.error();
        return validate_compressor(attention.indexer, ratio == 4U ? ratio : 0U,
                                   config.index_head_width, context);
    };
    for (std::size_t layer = 0U; layer < config.layer_count; ++layer) {
        auto valid = validate_attention(state.layers[layer], config.compression_ratios[layer],
                                        "main layer " + std::to_string(layer));
        if (!valid) return valid.error();
    }
    for (std::size_t layer = 0U; layer < config.mtp_layer_count; ++layer) {
        auto valid = validate_attention(
            state.mtp_layers[layer], config.mtp_compression_ratios[layer],
            "MTP layer " + std::to_string(layer));
        if (!valid) return valid.error();
    }
    return {};
}

}  // namespace

Result<std::span<const std::byte>> deepseek_v4_packed_matrix_payload(
    const DeepseekV4PackedMatrix& matrix,
    std::string_view identity) {
    return packed_payload(matrix.encoded, matrix.resident, matrix.resident_offset,
                          matrix.resident_bytes, identity);
}

Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const model::LoadedTensorTile& tile,
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns) {
    const auto& identity = tile.identity();
    if (identity.tensor_offset != 0U || !identity.final_tile ||
        tile.bytes().size() != identity.source.bytes) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek matrix binding requires one complete tensor tile");
    }
    return bind_resident_matrix(tile.retain_buffer(), 0U, tile.bytes().size(),
                                ggml_type, rows, logical_columns, storage_columns);
}

Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const model::LoadedExpert& expert,
    std::size_t catalog_tensor_index,
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns) {
    const auto found = std::find_if(
        expert.tensor_slices().begin(), expert.tensor_slices().end(),
        [catalog_tensor_index](const model::ExpertTensorSlice& slice) {
            return slice.tensor_index == catalog_tensor_index;
        });
    if (found == expert.tensor_slices().end()) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek expert matrix tensor is unavailable");
    }
    return bind_resident_matrix(
        expert.retain_buffer(), static_cast<std::size_t>(found->packed_offset),
        static_cast<std::size_t>(found->bytes), ggml_type, rows,
        logical_columns, storage_columns);
}

Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const model::LoadedTensorTile& tile,
    std::size_t elements) {
    const auto& identity = tile.identity();
    if (identity.tensor_offset != 0U || !identity.final_tile ||
        tile.bytes().size() != identity.source.bytes) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek I32 binding requires one complete tensor tile");
    }
    return bind_resident_i32(tile.retain_buffer(), 0U, tile.bytes().size(), elements);
}

Result<DeepseekV4PackedFloatTensor> deepseek_v4_bind_packed_float(
    const model::LoadedTensorTile& tile,
    std::uint32_t ggml_type,
    std::size_t elements) {
    const auto& identity = tile.identity();
    if (identity.tensor_offset != 0U || !identity.final_tile ||
        tile.bytes().size() != identity.source.bytes) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek float binding requires one complete tensor tile");
    }
    return bind_resident_float(tile.retain_buffer(), 0U, tile.bytes().size(), ggml_type,
                               elements);
}

Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const model::LoadedExpert& expert,
    std::size_t catalog_tensor_index,
    std::size_t elements) {
    const auto found = std::find_if(
        expert.tensor_slices().begin(), expert.tensor_slices().end(),
        [catalog_tensor_index](const model::ExpertTensorSlice& slice) {
            return slice.tensor_index == catalog_tensor_index;
        });
    if (found == expert.tensor_slices().end()) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek expert I32 tensor is unavailable");
    }
    return bind_resident_i32(
        expert.retain_buffer(), static_cast<std::size_t>(found->packed_offset),
        static_cast<std::size_t>(found->bytes), elements);
}

Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    std::string_view name,
    std::size_t logical_columns) {
    const auto* tensor = tensors.find(name);
    if (tensor == nullptr) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek trunk matrix tensor is unavailable",
                          std::string(name));
    }
    if (!tensor->ggml_type.has_value() || tensor->shape.size() != 2U ||
        *tensor->ggml_type == kGgmlTypeI32 ||
        !storage_type_matches_dtype(*tensor->ggml_type, tensor->dtype) ||
        tensor->shape[0] > std::numeric_limits<std::size_t>::max() ||
        tensor->shape[1] > std::numeric_limits<std::size_t>::max() ||
        tensor->packed_offset > std::numeric_limits<std::size_t>::max() ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek trunk matrix catalog descriptor is unsupported or malformed",
                          tensor->name);
    }
    return bind_resident_matrix(
        tensors.retain_buffer(), static_cast<std::size_t>(tensor->packed_offset),
        static_cast<std::size_t>(tensor->bytes), *tensor->ggml_type,
        static_cast<std::size_t>(tensor->shape[1]), logical_columns,
        static_cast<std::size_t>(tensor->shape[0]));
}

Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    std::string_view name) {
    const auto* tensor = tensors.find(name);
    if (tensor == nullptr) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek trunk I32 tensor is unavailable",
                          std::string(name));
    }
    if (tensor->dtype != model::TensorDType::i32 ||
        tensor->ggml_type != kGgmlTypeI32 ||
        tensor->packed_offset > std::numeric_limits<std::size_t>::max() ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek trunk I32 catalog descriptor is unsupported or malformed",
                          tensor->name);
    }
    auto elements = bounded_catalog_product(tensor->shape, tensor->name);
    if (!elements) return elements.error();
    return bind_resident_i32(
        tensors.retain_buffer(), static_cast<std::size_t>(tensor->packed_offset),
        static_cast<std::size_t>(tensor->bytes), elements.value());
}

Result<DeepseekV4PackedFloatTensor> deepseek_v4_bind_packed_float(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    std::string_view name) {
    const auto* tensor = tensors.find(name);
    if (tensor == nullptr) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek trunk float tensor is unavailable",
                          std::string(name));
    }
    if (!tensor->ggml_type.has_value() ||
        (*tensor->ggml_type != kGgmlTypeF32 &&
         *tensor->ggml_type != kGgmlTypeBf16) ||
        !storage_type_matches_dtype(*tensor->ggml_type, tensor->dtype) ||
        tensor->packed_offset > std::numeric_limits<std::size_t>::max() ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek trunk float catalog descriptor is unsupported or malformed",
                          tensor->name);
    }
    auto elements = bounded_catalog_product(tensor->shape, tensor->name);
    if (!elements) return elements.error();
    return bind_resident_float(
        tensors.retain_buffer(), static_cast<std::size_t>(tensor->packed_offset),
        static_cast<std::size_t>(tensor->bytes), *tensor->ggml_type,
        elements.value());
}

Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const DeepseekV4LoadedGlobalTensorSet& tensors,
    std::string_view name,
    std::size_t logical_columns) {
    const auto* tensor = tensors.find(name);
    if (tensor == nullptr) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek global matrix tensor is unavailable",
                          std::string(name));
    }
    if (!tensor->ggml_type.has_value() || tensor->shape.size() != 2U ||
        *tensor->ggml_type == kGgmlTypeI32 ||
        !storage_type_matches_dtype(*tensor->ggml_type, tensor->dtype) ||
        tensor->shape[0] > std::numeric_limits<std::size_t>::max() ||
        tensor->shape[1] > std::numeric_limits<std::size_t>::max() ||
        tensor->packed_offset > std::numeric_limits<std::size_t>::max() ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        return make_error(
            ErrorCode::invalid_format,
            "DeepSeek global matrix catalog descriptor is unsupported or malformed",
            tensor->name);
    }
    return bind_resident_matrix(
        tensors.retain_buffer(), static_cast<std::size_t>(tensor->packed_offset),
        static_cast<std::size_t>(tensor->bytes), *tensor->ggml_type,
        static_cast<std::size_t>(tensor->shape[1]), logical_columns,
        static_cast<std::size_t>(tensor->shape[0]));
}

Result<DeepseekV4PackedFloatTensor> deepseek_v4_bind_packed_float(
    const DeepseekV4LoadedGlobalTensorSet& tensors,
    std::string_view name) {
    const auto* tensor = tensors.find(name);
    if (tensor == nullptr) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek global float tensor is unavailable",
                          std::string(name));
    }
    if (!tensor->ggml_type.has_value() ||
        (*tensor->ggml_type != kGgmlTypeF32 &&
         *tensor->ggml_type != kGgmlTypeBf16) ||
        !storage_type_matches_dtype(*tensor->ggml_type, tensor->dtype) ||
        tensor->packed_offset > std::numeric_limits<std::size_t>::max() ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        return make_error(
            ErrorCode::invalid_format,
            "DeepSeek global float catalog descriptor is unsupported or malformed",
            tensor->name);
    }
    auto elements = bounded_catalog_product(tensor->shape, tensor->name);
    if (!elements) return elements.error();
    return bind_resident_float(
        tensors.retain_buffer(), static_cast<std::size_t>(tensor->packed_offset),
        static_cast<std::size_t>(tensor->bytes), *tensor->ggml_type,
        elements.value());
}

Result<DeepseekV4ReferenceConfig>
deepseek_v4_reference_config_from_0731_gguf(
    const DeepseekV4GgufCatalogReport& report) {
    try {
        const auto& pinned = pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
        if (!report.imported_manifest_verified ||
            !report.pinned_0731_checkpoint_spec_verified ||
            report.catalog.model_id != pinned.model_id ||
            report.catalog.revision != pinned.revision ||
            !report.reference_architecture.has_value()) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek 0731 reference configuration requires the pinned verified metadata identity");
        }
        const auto& source = *report.reference_architecture;
        if (source.layer_count != report.layers ||
            source.expert_count != report.experts_per_layer ||
            source.experts_per_token != report.top_k ||
            source.compression_ratios.size() !=
                static_cast<std::size_t>(source.layer_count) + source.hash_layer_count ||
            source.routed_expert_swiglu_limits.size() != source.layer_count ||
            source.shared_expert_swiglu_limits.size() != source.layer_count ||
            source.key_value_head_count != 1U || source.shared_expert_count != 1U ||
            source.expert_gating_function != 4U ||
            !source.expert_weights_normalized || source.rope_scaling_type != "yarn" ||
            source.vocabulary_size == 0U || source.maximum_context_tokens == 0U ||
            source.width == 0U || source.layer_count == 0U ||
            source.hash_layer_count > source.layer_count || source.head_count == 0U ||
            source.expert_count == 0U || source.experts_per_token == 0U ||
            source.experts_per_token > source.expert_count ||
            source.intermediate_width == 0U || source.query_lora_rank == 0U ||
            source.head_width == 0U || source.rope_width == 0U ||
            source.rope_width > source.head_width || source.rope_width % 2U != 0U ||
            source.output_groups == 0U || source.output_lora_rank == 0U ||
            source.window_size == 0U ||
            source.window_size > source.maximum_context_tokens ||
            source.original_context_tokens == 0U ||
            source.original_context_tokens > source.maximum_context_tokens ||
            source.index_head_count == 0U ||
            source.index_head_width < source.rope_width || source.index_top_k == 0U ||
            source.hyper_multiplier == 0U || source.hyper_multiplier > 16U ||
            source.sinkhorn_iterations == 0U) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 verified architecture snapshot is incomplete or inconsistent");
        }
        const std::array<float, 8U> positive_scalars{{
            source.compression_rope_theta, source.rope_theta, source.rope_factor,
            source.beta_fast, source.beta_slow, source.norm_epsilon,
            source.hyper_epsilon, source.route_scale,
        }};
        if (!std::all_of(positive_scalars.begin(), positive_scalars.end(),
                         [](float value) {
                             return std::isfinite(value) && value > 0.0F;
                         }) ||
            !std::all_of(source.compression_ratios.begin(),
                         source.compression_ratios.end(),
                         [](std::uint32_t value) {
                             return value == 0U || value == 4U || value == 128U;
                         })) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 verified architecture schedule or scalar is unsupported");
        }
        for (const auto& limits : {
                 std::span<const float>(source.routed_expert_swiglu_limits),
                 std::span<const float>(source.shared_expert_swiglu_limits)}) {
            if (!std::all_of(limits.begin(), limits.end(), [](float value) {
                    return std::isfinite(value) && value >= 0.0F;
                })) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 verified SwiGLU schedule contains an invalid value");
            }
        }
        DeepseekV4ReferenceConfig result;
        result.vocabulary_size = source.vocabulary_size;
        result.width = source.width;
        result.intermediate_width = source.intermediate_width;
        result.layer_count = source.layer_count;
        result.hash_layer_count = source.hash_layer_count;
        result.mtp_layer_count = 0U;
        result.head_count = source.head_count;
        result.expert_count = source.expert_count;
        result.experts_per_token = source.experts_per_token;
        result.query_lora_rank = source.query_lora_rank;
        result.head_width = source.head_width;
        result.rope_width = source.rope_width;
        result.output_groups = source.output_groups;
        result.output_lora_rank = source.output_lora_rank;
        result.window_size = source.window_size;
        result.compression_ratios.assign(
            source.compression_ratios.begin(),
            source.compression_ratios.begin() +
                static_cast<std::ptrdiff_t>(source.layer_count));
        result.compression_rope_theta = source.compression_rope_theta;
        result.original_context = source.original_context_tokens;
        result.rope_theta = source.rope_theta;
        result.rope_factor = source.rope_factor;
        result.beta_fast = source.beta_fast;
        result.beta_slow = source.beta_slow;
        result.index_head_count = source.index_head_count;
        result.index_head_width = source.index_head_width;
        result.index_top_k = source.index_top_k;
        result.hyper_multiplier = source.hyper_multiplier;
        result.sinkhorn_iterations = source.sinkhorn_iterations;
        result.norm_epsilon = source.norm_epsilon;
        result.hyper_epsilon = source.hyper_epsilon;
        result.route_scale = source.route_scale;
        result.routed_expert_swiglu_limits = source.routed_expert_swiglu_limits;
        result.shared_expert_swiglu_limits = source.shared_expert_swiglu_limits;
        result.swiglu_limit = 0.0F;
        auto layout = make_layer0_trunk_layout(result);
        if (!layout) return layout.error();
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate the bounded DeepSeek 0731 reference configuration");
    }
}

Result<void> deepseek_v4_validate_0731_gguf_layer0_catalog(
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4ReferenceConfig& config) {
    auto expected = deepseek_v4_reference_config_from_0731_gguf(report);
    if (!expected) return expected.error();
    const auto& canonical = expected.value();
    if (config.vocabulary_size != canonical.vocabulary_size ||
        config.width != canonical.width ||
        config.intermediate_width != canonical.intermediate_width ||
        config.layer_count != canonical.layer_count ||
        config.hash_layer_count != canonical.hash_layer_count ||
        config.mtp_layer_count != canonical.mtp_layer_count ||
        config.head_count != canonical.head_count ||
        config.expert_count != canonical.expert_count ||
        config.experts_per_token != canonical.experts_per_token ||
        config.query_lora_rank != canonical.query_lora_rank ||
        config.head_width != canonical.head_width ||
        config.rope_width != canonical.rope_width ||
        config.output_groups != canonical.output_groups ||
        config.output_lora_rank != canonical.output_lora_rank ||
        config.window_size != canonical.window_size ||
        config.compression_ratios != canonical.compression_ratios ||
        config.mtp_compression_ratios != canonical.mtp_compression_ratios ||
        config.compression_rope_theta != canonical.compression_rope_theta ||
        config.original_context != canonical.original_context ||
        config.rope_theta != canonical.rope_theta ||
        config.rope_factor != canonical.rope_factor ||
        config.beta_fast != canonical.beta_fast ||
        config.beta_slow != canonical.beta_slow ||
        config.index_head_count != canonical.index_head_count ||
        config.index_head_width != canonical.index_head_width ||
        config.index_top_k != canonical.index_top_k ||
        config.hyper_multiplier != canonical.hyper_multiplier ||
        config.sinkhorn_iterations != canonical.sinkhorn_iterations ||
        config.norm_epsilon != canonical.norm_epsilon ||
        config.hyper_epsilon != canonical.hyper_epsilon ||
        config.route_scale != canonical.route_scale ||
        config.routed_expert_swiglu_limits !=
            canonical.routed_expert_swiglu_limits ||
        config.shared_expert_swiglu_limits !=
            canonical.shared_expert_swiglu_limits ||
        config.swiglu_limit != canonical.swiglu_limit) {
        return make_error(
            ErrorCode::invalid_argument,
            "DeepSeek 0731 layer-0 catalog configuration differs from verified metadata");
    }
    auto layout = make_layer0_trunk_layout(config);
    if (!layout) return layout.error();

    const auto find_tensor = [&](std::string_view name)
        -> const model::LogicalTensorDescriptor* {
        const auto found = std::find_if(
            report.catalog.tensors.begin(), report.catalog.tensors.end(),
            [name](const model::LogicalTensorDescriptor& tensor) {
                return tensor.name == name;
            });
        return found == report.catalog.tensors.end() ? nullptr : &*found;
    };
    const auto validate_storage = [&](const model::LogicalTensorDescriptor& tensor,
                                      std::size_t rows,
                                      std::size_t storage_columns) -> Result<void> {
        if (!tensor.ggml_type.has_value() ||
            !storage_type_matches_dtype(*tensor.ggml_type, tensor.dtype)) {
            return make_error(
                ErrorCode::unsupported,
                "DeepSeek 0731 layer-0 tensor storage type has no reference kernel",
                tensor.name);
        }
        auto bytes = ggml_storage_bytes(
            *tensor.ggml_type, rows, storage_columns,
            QuantizedKernelLimits{kMaximumGraphElements});
        if (!bytes) return bytes.error();
        if (bytes.value() != tensor.source.bytes) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek 0731 layer-0 encoded byte count disagrees with its shape",
                tensor.name);
        }
        return {};
    };

    std::size_t layer0_tensor_count = 0U;
    for (const auto& tensor : report.catalog.tensors) {
        if (tensor.layer == 0U &&
            tensor.role != model::TensorRole::routed_expert) {
            ++layer0_tensor_count;
        }
    }
    if (layer0_tensor_count != 21U) {
        return make_error(
            ErrorCode::invalid_format,
            "DeepSeek 0731 layer 0 does not contain exactly 21 non-routed tensors",
            std::to_string(layer0_tensor_count));
    }
    for (const auto& mapping : layout.value().matrices) {
        const auto* tensor = find_tensor(mapping.source);
        if (tensor == nullptr || tensor->layer != 0U || tensor->expert.has_value() ||
            tensor->role != mapping.role ||
            tensor->shape.size() != 2U || tensor->shape[1] != mapping.rows ||
            tensor->shape[0] < mapping.columns ||
            tensor->shape[0] > std::numeric_limits<std::size_t>::max()) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 layer-0 matrix descriptor does not match its consumer",
                std::string(mapping.source));
        }
        auto valid = validate_storage(
            *tensor, mapping.rows, static_cast<std::size_t>(tensor->shape[0]));
        if (!valid) return valid.error();
    }
    for (const auto& mapping : layout.value().floats) {
        const auto* tensor = find_tensor(mapping.source);
        if (tensor == nullptr || tensor->layer != 0U || tensor->expert.has_value() ||
            tensor->role != mapping.role ||
            !tensor->ggml_type.has_value() ||
            (*tensor->ggml_type != kGgmlTypeF32 &&
             *tensor->ggml_type != kGgmlTypeBf16)) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 layer-0 float descriptor does not match its consumer",
                std::string(mapping.source));
        }
        auto elements = bounded_catalog_product(tensor->shape, tensor->name);
        if (!elements) return elements.error();
        if (elements.value() != mapping.elements) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 layer-0 float element count does not match its consumer",
                tensor->name);
        }
        auto valid = validate_storage(*tensor, 1U, elements.value());
        if (!valid) return valid.error();
    }
    const auto* table = find_tensor("blk.0.ffn_gate_tid2eid.weight");
    if (table == nullptr || table->layer != 0U || table->expert.has_value() ||
        table->dtype != model::TensorDType::i32 ||
        table->ggml_type != kGgmlTypeI32) {
        return make_error(
            ErrorCode::invalid_format,
            "DeepSeek 0731 layer-0 routing table descriptor is missing or mistyped");
    }
    auto table_elements = bounded_catalog_product(table->shape, table->name);
    if (!table_elements) return table_elements.error();
    if (table_elements.value() != layout.value().routing_table_elements) {
        return make_error(
            ErrorCode::invalid_format,
            "DeepSeek 0731 layer-0 routing table has the wrong element count",
            table->name);
    }
    auto table_storage = validate_storage(*table, 1U, table_elements.value());
    if (!table_storage) return table_storage.error();
    return {};
}

Result<DeepseekV4ReferenceCatalogValidation>
deepseek_v4_validate_0731_gguf_layer_catalogs(
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4ReferenceConfig& config) {
    try {
        auto layer0 = deepseek_v4_validate_0731_gguf_layer0_catalog(report, config);
        if (!layer0) return layer0.error();
        if (config.layer_count != report.layers ||
            config.compression_ratios.size() != config.layer_count) {
            return make_error(
                ErrorCode::invalid_argument,
                "DeepSeek 0731 all-layer catalog configuration is inconsistent");
        }
        auto common = make_layer0_trunk_layout(config);
        if (!common) return common.error();

        DeepseekV4ReferenceCatalogValidation result;
        const auto find_tensor = [&](std::string_view name)
            -> const model::LogicalTensorDescriptor* {
            const auto found = std::find_if(
                report.catalog.tensors.begin(), report.catalog.tensors.end(),
                [name](const model::LogicalTensorDescriptor& tensor) {
                    return tensor.name == name;
                });
            return found == report.catalog.tensors.end() ? nullptr : &*found;
        };
        const auto validate_storage = [&](const model::LogicalTensorDescriptor& tensor,
                                          std::size_t rows,
                                          std::size_t storage_columns) -> Result<void> {
            if (!tensor.ggml_type.has_value() ||
                !storage_type_matches_dtype(*tensor.ggml_type, tensor.dtype)) {
                return make_error(
                    ErrorCode::unsupported,
                    "DeepSeek 0731 layer tensor storage type has no reference kernel",
                    tensor.name);
            }
            auto bytes = ggml_storage_bytes(
                *tensor.ggml_type, rows, storage_columns,
                QuantizedKernelLimits{kMaximumGraphElements});
            if (!bytes) return bytes.error();
            if (bytes.value() != tensor.source.bytes) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek 0731 layer tensor encoded bytes disagree with shape/storage",
                    tensor.name);
            }
            return {};
        };
        const auto record_tensor = [&](const model::LogicalTensorDescriptor& tensor)
            -> Result<void> {
            if (result.tensors == std::numeric_limits<std::size_t>::max()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek 0731 validated tensor count overflows");
            }
            auto total = checked_add(result.bytes, tensor.source.bytes);
            if (!total) return total.error();
            ++result.tensors;
            result.bytes = total.value();
            return {};
        };

        for (std::size_t layer = 0U; layer < config.layer_count; ++layer) {
            const auto prefix = "blk." + std::to_string(layer);
            const auto layer_name = [&](std::string_view layer0_name) {
                return prefix + std::string(layer0_name.substr(5U));
            };
            const auto validate_matrix = [&](const std::string& name,
                                             std::size_t rows,
                                             std::size_t columns,
                                             model::TensorRole role) -> Result<void> {
                const auto* tensor = find_tensor(name);
                if (tensor == nullptr || !tensor->layer.has_value() ||
                    *tensor->layer != layer || tensor->expert.has_value() ||
                    tensor->role != role || tensor->shape.size() != 2U ||
                    tensor->shape[1] != rows || tensor->shape[0] < columns ||
                    tensor->shape[0] > std::numeric_limits<std::size_t>::max()) {
                    return make_error(
                        ErrorCode::invalid_format,
                        "DeepSeek 0731 layer matrix descriptor does not match its consumer",
                        name);
                }
                auto valid = validate_storage(
                    *tensor, rows, static_cast<std::size_t>(tensor->shape[0]));
                if (!valid) return valid.error();
                return record_tensor(*tensor);
            };
            const auto validate_float = [&](const std::string& name,
                                            std::initializer_list<std::uint64_t> shape,
                                            model::TensorRole role) -> Result<void> {
                const auto* tensor = find_tensor(name);
                if (tensor == nullptr || !tensor->layer.has_value() ||
                    *tensor->layer != layer || tensor->expert.has_value() ||
                    tensor->role != role || !tensor->ggml_type.has_value() ||
                    (*tensor->ggml_type != kGgmlTypeF32 &&
                     *tensor->ggml_type != kGgmlTypeBf16) ||
                    tensor->shape.size() != shape.size() ||
                    !std::equal(tensor->shape.begin(), tensor->shape.end(),
                                shape.begin(), shape.end())) {
                    return make_error(
                        ErrorCode::invalid_format,
                        "DeepSeek 0731 layer float descriptor does not match its consumer",
                        name);
                }
                auto elements = bounded_catalog_product(tensor->shape, tensor->name);
                if (!elements) return elements.error();
                auto valid = validate_storage(*tensor, 1U, elements.value());
                if (!valid) return valid.error();
                return record_tensor(*tensor);
            };

            const std::size_t before_layer = result.tensors;
            for (const auto& mapping : common.value().matrices) {
                auto valid = validate_matrix(layer_name(mapping.source), mapping.rows,
                                             mapping.columns, mapping.role);
                if (!valid) return valid.error();
            }
            for (const auto& mapping : common.value().floats) {
                auto valid = validate_float(layer_name(mapping.source),
                                            {mapping.elements}, mapping.role);
                if (!valid) return valid.error();
            }

            if (layer < config.hash_layer_count) {
                const auto name = prefix + ".ffn_gate_tid2eid.weight";
                const auto* table = find_tensor(name);
                if (table == nullptr || !table->layer.has_value() ||
                    *table->layer != layer || table->expert.has_value() ||
                    table->role != model::TensorRole::router ||
                    table->dtype != model::TensorDType::i32 ||
                    table->ggml_type != kGgmlTypeI32 || table->shape.size() != 2U ||
                    table->shape[0] != config.experts_per_token ||
                    table->shape[1] != config.vocabulary_size) {
                    return make_error(
                        ErrorCode::invalid_format,
                        "DeepSeek 0731 hash routing descriptor does not match its consumer",
                        name);
                }
                auto storage = validate_storage(
                    *table, 1U, common.value().routing_table_elements);
                if (!storage) return storage.error();
                auto recorded = record_tensor(*table);
                if (!recorded) return recorded.error();
            } else {
                auto valid = validate_float(prefix + ".exp_probs_b.bias",
                                            {config.expert_count},
                                            model::TensorRole::other);
                if (!valid) return valid.error();
            }

            const std::size_t ratio = config.compression_ratios[layer];
            if (ratio != 0U) {
                if (ratio != 4U && ratio != 128U) {
                    return make_error(
                        ErrorCode::unsupported,
                        "DeepSeek 0731 layer compression ratio is unsupported",
                        std::to_string(layer));
                }
                const std::size_t coefficient = ratio == 4U ? 2U : 1U;
                auto projected_width = bounded_product(
                    {coefficient, config.head_width},
                    "DeepSeek 0731 attention compressor width");
                if (!projected_width) return projected_width.error();
                for (const auto suffix : {".attn_compressor_gate.weight",
                                          ".attn_compressor_kv.weight"}) {
                    auto valid = validate_matrix(prefix + suffix,
                                                 projected_width.value(),
                                                 config.width,
                                                 model::TensorRole::attention);
                    if (!valid) return valid.error();
                }
                auto ape = validate_float(prefix + ".attn_compressor_ape.weight",
                                          {projected_width.value(), ratio},
                                          model::TensorRole::attention);
                if (!ape) return ape.error();
                auto norm = validate_float(prefix + ".attn_compressor_norm.weight",
                                           {config.head_width},
                                           model::TensorRole::normalization);
                if (!norm) return norm.error();

                if (ratio == 4U) {
                    auto index_width = bounded_product(
                        {config.index_head_count, config.index_head_width},
                        "DeepSeek 0731 indexer query width");
                    if (!index_width) return index_width.error();
                    auto index_projected_width = bounded_product(
                        {2U, config.index_head_width},
                        "DeepSeek 0731 indexer compressor width");
                    if (!index_projected_width) return index_projected_width.error();
                    auto query = validate_matrix(prefix + ".indexer.attn_q_b.weight",
                                                 index_width.value(),
                                                 config.query_lora_rank,
                                                 model::TensorRole::attention);
                    if (!query) return query.error();
                    auto projection = validate_matrix(prefix + ".indexer.proj.weight",
                                                      config.index_head_count,
                                                      config.width,
                                                      model::TensorRole::other);
                    if (!projection) return projection.error();
                    for (const auto suffix : {".indexer_compressor_gate.weight",
                                              ".indexer_compressor_kv.weight"}) {
                        auto valid = validate_matrix(prefix + suffix,
                                                     index_projected_width.value(),
                                                     config.width,
                                                     model::TensorRole::other);
                        if (!valid) return valid.error();
                    }
                    auto index_ape = validate_float(
                        prefix + ".indexer_compressor_ape.weight",
                        {index_projected_width.value(), ratio},
                        model::TensorRole::other);
                    if (!index_ape) return index_ape.error();
                    auto index_norm = validate_float(
                        prefix + ".indexer_compressor_norm.weight",
                        {config.index_head_width},
                        model::TensorRole::normalization);
                    if (!index_norm) return index_norm.error();
                }
            }

            std::size_t actual_layer_tensors = 0U;
            for (const auto& tensor : report.catalog.tensors) {
                if (tensor.layer.has_value() && *tensor.layer == layer &&
                    tensor.role != model::TensorRole::routed_expert) {
                    ++actual_layer_tensors;
                }
            }
            const std::size_t expected_layer_tensors = result.tensors - before_layer;
            if (actual_layer_tensors != expected_layer_tensors) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 layer contains an unexpected non-routed tensor set",
                    std::to_string(layer));
            }
            ++result.layers;
        }

        if ((report.layer_trunk_planned_layers != 0U &&
             report.layer_trunk_planned_layers != result.layers) ||
            (report.layer_trunk_tensor_count != 0U &&
             report.layer_trunk_tensor_count != result.tensors) ||
            (report.reference_storage_types_complete &&
             (report.reference_storage_type_covered_tensor_count != result.tensors ||
              report.reference_storage_type_covered_bytes != result.bytes ||
              report.reference_storage_type_unsupported_tensor_count != 0U ||
              report.reference_storage_type_unsupported_bytes != 0U))) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek 0731 validated layer catalog disagrees with planning totals");
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate the bounded DeepSeek 0731 all-layer catalog validation");
    }
}

Result<DeepseekV4ReferenceCheckpointCatalogValidation>
deepseek_v4_validate_0731_gguf_checkpoint_catalog(
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4ReferenceConfig& config) {
    try {
        auto layers = deepseek_v4_validate_0731_gguf_layer_catalogs(report, config);
        if (!layers) return layers.error();
        if (config.layer_count == 0U || config.expert_count == 0U ||
            config.width == 0U || config.intermediate_width == 0U ||
            config.vocabulary_size == 0U || config.hyper_multiplier == 0U ||
            config.layer_count > std::numeric_limits<std::size_t>::max() /
                                     config.expert_count) {
            return make_error(
                ErrorCode::invalid_argument,
                "DeepSeek 0731 checkpoint catalog configuration is invalid");
        }

        DeepseekV4ReferenceCheckpointCatalogValidation result;
        result.layers = layers.value();
        const auto find_tensor = [&](std::string_view name)
            -> const model::LogicalTensorDescriptor* {
            const auto found = std::find_if(
                report.catalog.tensors.begin(), report.catalog.tensors.end(),
                [name](const model::LogicalTensorDescriptor& tensor) {
                    return tensor.name == name;
                });
            return found == report.catalog.tensors.end() ? nullptr : &*found;
        };
        const auto validate_storage = [&](const model::LogicalTensorDescriptor& tensor,
                                          std::size_t rows,
                                          std::size_t storage_columns) -> Result<void> {
            if (!tensor.ggml_type.has_value() || rows == 0U || storage_columns == 0U ||
                rows > std::numeric_limits<std::size_t>::max() / storage_columns ||
                !storage_type_matches_dtype(*tensor.ggml_type, tensor.dtype)) {
                return make_error(
                    ErrorCode::unsupported,
                    "DeepSeek 0731 checkpoint tensor storage layout is unsupported",
                    tensor.name);
            }
            const std::size_t elements = rows * storage_columns;
            auto bytes = ggml_storage_bytes(
                *tensor.ggml_type, rows, storage_columns,
                QuantizedKernelLimits{elements});
            if (!bytes) return bytes.error();
            if (bytes.value() != tensor.source.bytes) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek 0731 checkpoint tensor bytes disagree with shape/storage",
                    tensor.name);
            }
            return {};
        };
        const auto record_global = [&](const model::LogicalTensorDescriptor& tensor)
            -> Result<void> {
            if (result.global_tensors == std::numeric_limits<std::size_t>::max()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek 0731 global tensor count overflows");
            }
            auto total = checked_add(result.global_bytes, tensor.source.bytes);
            if (!total) return total.error();
            ++result.global_tensors;
            result.global_bytes = total.value();
            return {};
        };
        const auto validate_global_matrix = [&](std::string_view name,
                                                std::size_t rows,
                                                std::size_t columns,
                                                model::TensorRole role) -> Result<void> {
            const auto* tensor = find_tensor(name);
            if (tensor == nullptr || tensor->layer.has_value() ||
                tensor->expert.has_value() || tensor->role != role ||
                tensor->shape.size() != 2U || tensor->shape[1] != rows ||
                tensor->shape[0] < columns ||
                tensor->shape[0] > std::numeric_limits<std::size_t>::max()) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 global matrix descriptor does not match its consumer",
                    std::string(name));
            }
            auto storage = validate_storage(
                *tensor, rows, static_cast<std::size_t>(tensor->shape[0]));
            if (!storage) return storage.error();
            return record_global(*tensor);
        };
        const auto validate_global_float = [&](std::string_view name,
                                               std::initializer_list<std::uint64_t> shape,
                                               model::TensorRole role) -> Result<void> {
            const auto* tensor = find_tensor(name);
            if (tensor == nullptr || tensor->layer.has_value() ||
                tensor->expert.has_value() || tensor->role != role ||
                !tensor->ggml_type.has_value() ||
                (*tensor->ggml_type != kGgmlTypeF32 &&
                 *tensor->ggml_type != kGgmlTypeBf16) ||
                tensor->shape.size() != shape.size() ||
                !std::equal(tensor->shape.begin(), tensor->shape.end(),
                            shape.begin(), shape.end())) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 global float descriptor does not match its consumer",
                    std::string(name));
            }
            std::size_t elements = 1U;
            for (const auto dimension : shape) {
                if (dimension == 0U ||
                    dimension > std::numeric_limits<std::size_t>::max() ||
                    elements > std::numeric_limits<std::size_t>::max() /
                                   static_cast<std::size_t>(dimension)) {
                    return make_error(
                        ErrorCode::resource_exhausted,
                        "DeepSeek 0731 global float shape overflows",
                        std::string(name));
                }
                elements *= static_cast<std::size_t>(dimension);
            }
            auto storage = validate_storage(*tensor, 1U, elements);
            if (!storage) return storage.error();
            return record_global(*tensor);
        };

        auto embedding = validate_global_matrix(
            "token_embd.weight", config.vocabulary_size, config.width,
            model::TensorRole::embedding);
        if (!embedding) return embedding.error();
        auto output = validate_global_matrix(
            "output.weight", config.vocabulary_size, config.width,
            model::TensorRole::output_head);
        if (!output) return output.error();
        auto output_norm = validate_global_float(
            "output_norm.weight", {config.width}, model::TensorRole::normalization);
        if (!output_norm) return output_norm.error();
        auto head_base = validate_global_float(
            "output_hc_base.weight", {config.hyper_multiplier},
            model::TensorRole::hyper_connection);
        if (!head_base) return head_base.error();
        auto head_scale = validate_global_float(
            "output_hc_scale.weight", {1U}, model::TensorRole::hyper_connection);
        if (!head_scale) return head_scale.error();
        if (config.hyper_multiplier > std::numeric_limits<std::size_t>::max() /
                                          config.width) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek 0731 HC head width overflows");
        }
        auto head_fn = validate_global_matrix(
            "output_hc_fn.weight", config.hyper_multiplier,
            config.hyper_multiplier * config.width,
            model::TensorRole::hyper_connection);
        if (!head_fn) return head_fn.error();
        const auto* head_fn_tensor = find_tensor("output_hc_fn.weight");
        if (head_fn_tensor == nullptr ||
            (head_fn_tensor->ggml_type != kGgmlTypeF32 &&
             head_fn_tensor->ggml_type != kGgmlTypeBf16)) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 HC head matrix must use bounded float storage",
                "output_hc_fn.weight");
        }

        std::size_t actual_global_tensors = 0U;
        for (const auto& tensor : report.catalog.tensors) {
            if (!tensor.layer.has_value()) ++actual_global_tensors;
        }
        if (result.global_tensors != 6U || actual_global_tensors != 6U) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 checkpoint does not contain exactly six global tensors",
                std::to_string(actual_global_tensors));
        }

        const std::size_t expected_bundles =
            config.layer_count * config.expert_count;
        if (report.catalog.experts.size() != expected_bundles ||
            expected_bundles > std::numeric_limits<std::size_t>::max() / 3U) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 checkpoint expert bundle count is inconsistent",
                std::to_string(report.catalog.experts.size()));
        }
        const std::size_t expected_expert_slices = expected_bundles * 3U;
        std::vector<bool> claimed(report.catalog.tensors.size(), false);
        std::uint64_t routed_bytes = 0U;
        for (std::size_t layer = 0U; layer < config.layer_count; ++layer) {
            for (std::size_t expert = 0U; expert < config.expert_count; ++expert) {
                const std::size_t bundle_index = layer * config.expert_count + expert;
                const auto& bundle = report.catalog.experts[bundle_index];
                if (bundle.layer != layer || bundle.expert != expert ||
                    bundle.speculative || bundle.tensor_indices.size() != 3U ||
                    bundle.tensor_slices.size() != 3U || bundle.read_ranges.empty()) {
                    return make_error(
                        ErrorCode::invalid_format,
                        "DeepSeek 0731 routed expert bundle identity/layout is invalid",
                        std::to_string(bundle_index));
                }
                const std::string prefix = "blk." + std::to_string(layer);
                const std::string suffix = ".weight.expert." + std::to_string(expert);
                const std::array<std::string, 3U> expected_names{{
                    prefix + ".ffn_down_exps" + suffix,
                    prefix + ".ffn_gate_exps" + suffix,
                    prefix + ".ffn_up_exps" + suffix,
                }};
                const std::array<std::size_t, 3U> expected_rows{{
                    config.width, config.intermediate_width,
                    config.intermediate_width,
                }};
                const std::array<std::size_t, 3U> expected_columns{{
                    config.intermediate_width, config.width, config.width,
                }};
                std::array<bool, 3U> projections{};
                std::optional<std::uint32_t> gate_up_type;
                std::uint64_t bundle_bytes = 0U;
                for (const auto tensor_index : bundle.tensor_indices) {
                    if (tensor_index >= report.catalog.tensors.size() ||
                        claimed[tensor_index]) {
                        return make_error(
                            ErrorCode::invalid_format,
                            "DeepSeek 0731 routed expert tensor index is invalid or duplicated",
                            std::to_string(tensor_index));
                    }
                    const auto& tensor = report.catalog.tensors[tensor_index];
                    const auto projection = std::find(
                        expected_names.begin(), expected_names.end(), tensor.name);
                    if (projection == expected_names.end()) {
                        return make_error(
                            ErrorCode::invalid_format,
                            "DeepSeek 0731 routed expert tensor name is not canonical",
                            tensor.name);
                    }
                    const std::size_t projection_index = static_cast<std::size_t>(
                        std::distance(expected_names.begin(), projection));
                    const bool expected_profile_type = projection_index == 0U
                        ? (tensor.ggml_type == kGgmlTypeIq3Xxs ||
                           tensor.ggml_type == kGgmlTypeMxfp4)
                        : (tensor.ggml_type == kGgmlTypeIq1S ||
                           tensor.ggml_type == kGgmlTypeIq2Xxs);
                    if (projections[projection_index] ||
                        tensor.role != model::TensorRole::routed_expert ||
                        tensor.speculative || tensor.layer != layer ||
                        tensor.expert != expert || tensor.shape.size() != 2U ||
                        tensor.shape[1] != expected_rows[projection_index] ||
                        tensor.shape[0] < expected_columns[projection_index] ||
                        tensor.shape[0] > std::numeric_limits<std::size_t>::max() ||
                        !expected_profile_type) {
                        return make_error(
                            ErrorCode::invalid_format,
                            "DeepSeek 0731 routed expert descriptor does not match its consumer",
                            tensor.name);
                    }
                    auto storage = validate_storage(
                        tensor, expected_rows[projection_index],
                        static_cast<std::size_t>(tensor.shape[0]));
                    if (!storage) return storage.error();
                    if (projection_index != 0U) {
                        if (gate_up_type.has_value() &&
                            gate_up_type != tensor.ggml_type) {
                            return make_error(
                                ErrorCode::invalid_format,
                                "DeepSeek 0731 gate/up expert storage types disagree",
                                tensor.name);
                        }
                        gate_up_type = tensor.ggml_type;
                    }
                    auto bundle_total = checked_add(bundle_bytes, tensor.source.bytes);
                    if (!bundle_total) return bundle_total.error();
                    bundle_bytes = bundle_total.value();
                    projections[projection_index] = true;
                    claimed[tensor_index] = true;
                    ++result.expert_tensor_slices;
                }
                if (!std::all_of(projections.begin(), projections.end(),
                                 [](bool present) { return present; }) ||
                    bundle_bytes != bundle.bytes) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "DeepSeek 0731 routed expert bundle byte/layout closure failed",
                        std::to_string(bundle_index));
                }
                for (const auto& slice : bundle.tensor_slices) {
                    if (slice.tensor_index >= report.catalog.tensors.size() ||
                        std::find(bundle.tensor_indices.begin(),
                                  bundle.tensor_indices.end(),
                                  slice.tensor_index) == bundle.tensor_indices.end() ||
                        slice.bytes !=
                            report.catalog.tensors[slice.tensor_index].source.bytes ||
                        slice.packed_offset > bundle.bytes ||
                        slice.bytes > bundle.bytes - slice.packed_offset) {
                        return make_error(
                            ErrorCode::integrity_error,
                            "DeepSeek 0731 routed expert packed slice is invalid",
                            std::to_string(bundle_index));
                    }
                }
                for (const auto tensor_index : bundle.tensor_indices) {
                    const auto matching_slices = std::count_if(
                        bundle.tensor_slices.begin(), bundle.tensor_slices.end(),
                        [tensor_index](const model::ExpertTensorSlice& slice) {
                            return slice.tensor_index == tensor_index;
                        });
                    if (matching_slices != 1) {
                        return make_error(
                            ErrorCode::integrity_error,
                            "DeepSeek 0731 routed expert packed slices are not one-to-one",
                            std::to_string(bundle_index));
                    }
                }
                std::uint64_t read_bytes = 0U;
                for (const auto& range : bundle.read_ranges) {
                    auto read_total = checked_add(read_bytes, range.bytes);
                    if (!read_total) return read_total.error();
                    read_bytes = read_total.value();
                }
                if (read_bytes != bundle.bytes) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "DeepSeek 0731 routed expert read ranges do not close",
                        std::to_string(bundle_index));
                }
                auto routed_total = checked_add(routed_bytes, bundle.bytes);
                if (!routed_total) return routed_total.error();
                routed_bytes = routed_total.value();
                ++result.expert_bundles;
            }
        }

        std::size_t actual_routed_slices = 0U;
        for (std::size_t index = 0U; index < report.catalog.tensors.size(); ++index) {
            if (report.catalog.tensors[index].role == model::TensorRole::routed_expert) {
                ++actual_routed_slices;
                if (!claimed[index]) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "DeepSeek 0731 routed expert tensor is outside every bundle",
                        report.catalog.tensors[index].name);
                }
            } else if (claimed[index]) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek 0731 non-routed tensor was claimed by an expert bundle",
                    report.catalog.tensors[index].name);
            }
        }
        if (result.expert_tensor_slices != expected_expert_slices ||
            actual_routed_slices != expected_expert_slices ||
            routed_bytes != report.catalog.routed_expert_bytes) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek 0731 routed expert catalog totals do not close");
        }

        if (config.layer_count > std::numeric_limits<std::size_t>::max() / 3U) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek 0731 physical expert count overflows");
        }
        result.physical_expert_tensors = config.layer_count * 3U;
        if (result.layers.tensors > std::numeric_limits<std::size_t>::max() -
                                        result.global_tensors ||
            result.layers.tensors + result.global_tensors >
                std::numeric_limits<std::size_t>::max() -
                    result.physical_expert_tensors) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek 0731 physical tensor closure overflows");
        }
        const std::size_t expected_physical_tensors =
            result.layers.tensors + result.global_tensors +
            result.physical_expert_tensors;
        if (report.physical_tensors != expected_physical_tensors) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek 0731 physical tensor partition does not close",
                std::to_string(report.physical_tensors));
        }

        if (result.layers.tensors > std::numeric_limits<std::size_t>::max() -
                                        result.global_tensors ||
            result.layers.tensors + result.global_tensors >
                std::numeric_limits<std::size_t>::max() -
                    result.expert_tensor_slices) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek 0731 logical tensor closure overflows");
        }
        result.logical_tensors = result.layers.tensors + result.global_tensors +
                                 result.expert_tensor_slices;
        auto always_active = checked_add(result.layers.bytes, result.global_bytes);
        if (!always_active) return always_active.error();
        auto tensor_bytes = checked_add(always_active.value(), routed_bytes);
        if (!tensor_bytes) return tensor_bytes.error();
        result.tensor_bytes = tensor_bytes.value();
        if (result.logical_tensors != report.catalog.tensors.size() ||
            always_active.value() != report.catalog.always_active_bytes ||
            result.tensor_bytes != report.catalog.tensor_bytes) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek 0731 complete logical tensor/byte partition does not close");
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate the bounded DeepSeek 0731 checkpoint catalog validation");
    }
}

Result<DeepseekV4ReferenceLayerTrunk>
deepseek_v4_bind_0731_gguf_layer0_trunk(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    const DeepseekV4ReferenceConfig& config) {
    try {
        constexpr std::size_t expected_tensor_count = 21U;
        if (tensors.tensors().size() != expected_tensor_count) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 layer-0 trunk binding requires exactly 21 tensors");
        }
        auto layout = make_layer0_trunk_layout(config);
        if (!layout) return layout.error();

        DeepseekV4ReferenceLayerTrunk result;
        result.layer = 0U;
        for (const auto& mapping : layout.value().matrices) {
            auto matrix = deepseek_v4_bind_packed_matrix(
                tensors, mapping.source, mapping.columns);
            if (!matrix) return matrix.error();
            if (matrix.value().rows != mapping.rows) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 trunk matrix shape does not match its consumer",
                    std::string(mapping.source));
            }
            result.matrices.emplace(mapping.consumer, matrix.take_value());
        }
        for (const auto& mapping : layout.value().floats) {
            auto tensor = deepseek_v4_bind_packed_float(tensors, mapping.source);
            if (!tensor) return tensor.error();
            if (tensor.value().elements != mapping.elements) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 trunk float shape does not match its consumer",
                    std::string(mapping.source));
            }
            result.float_tensors.emplace(mapping.consumer, tensor.take_value());
        }
        auto table = deepseek_v4_bind_packed_i32(
            tensors, "blk.0.ffn_gate_tid2eid.weight");
        if (!table) return table.error();
        if (table.value().elements != layout.value().routing_table_elements) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 hash routing table shape does not match its consumer",
                "blk.0.ffn_gate_tid2eid.weight");
        }
        result.integer_tensors.emplace(
            "layers.0.ffn.gate.tid2eid", table.take_value());
        if (result.matrices.size() != layout.value().matrices.size() ||
            result.float_tensors.size() != layout.value().floats.size() ||
            result.integer_tensors.size() != 1U) {
            return make_error(
                ErrorCode::internal_error,
                "DeepSeek 0731 layer-0 trunk mapping is not one-to-one");
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate the bounded DeepSeek 0731 layer-0 trunk binding");
    }
}

namespace {

// Layer-0 names carry the layer in one place: "blk.0.x" and "layers.0.x". Retargeting them keeps a
// single source of truth for the common layout instead of duplicating it per layer.
Result<std::string> retarget_layer_name(std::string_view layer0_name, std::size_t layer) {
    const auto marker = layer0_name.find(".0.");
    if (marker == std::string_view::npos) {
        return make_error(ErrorCode::internal_error,
                          "DeepSeek layer name has no layer component",
                          std::string(layer0_name));
    }
    try {
        std::string result(layer0_name.substr(0U, marker));
        result += '.';
        result += std::to_string(layer);
        result += layer0_name.substr(marker + 2U);
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate a retargeted DeepSeek layer name");
    }
}

}  // namespace

Result<DeepseekV4ReferenceLayerTrunk> deepseek_v4_build_layer_trunk(
    const DeepseekV4LayerTrunkSource& source_binder,
    const DeepseekV4ReferenceConfig& config,
    std::size_t layer) {
    if (layer >= config.layer_count || config.compression_ratios.size() != config.layer_count ||
        source_binder.matrix == nullptr || source_binder.vector == nullptr ||
        source_binder.table == nullptr) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek 0731 layer trunk binding names a layer outside the checkpoint",
                          std::to_string(layer));
    }
    try {
        auto layout = make_layer0_trunk_layout(config);
        if (!layout) return layout.error();

        DeepseekV4ReferenceLayerTrunk result;
        result.layer = layer;
        std::size_t consumed = 0U;
        const auto bind_matrix = [&](std::string_view source, std::string_view consumer,
                                     std::size_t rows, std::size_t columns) -> Result<void> {
            auto matrix = source_binder.matrix(source, columns);
            if (!matrix) return matrix.error();
            if (matrix.value().rows != rows) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek 0731 trunk matrix shape does not match its consumer",
                                  std::string(source));
            }
            result.matrices.emplace(std::string(consumer), matrix.take_value());
            ++consumed;
            return {};
        };
        const auto bind_float = [&](std::string_view source, std::string_view consumer,
                                    std::size_t elements) -> Result<void> {
            auto tensor = source_binder.vector(source);
            if (!tensor) return tensor.error();
            if (tensor.value().elements != elements) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek 0731 trunk float shape does not match its consumer",
                                  std::string(source));
            }
            result.float_tensors.emplace(std::string(consumer), tensor.take_value());
            ++consumed;
            return {};
        };

        for (const auto& mapping : layout.value().matrices) {
            auto source = retarget_layer_name(mapping.source, layer);
            if (!source) return source.error();
            auto consumer = retarget_layer_name(mapping.consumer, layer);
            if (!consumer) return consumer.error();
            auto bound = bind_matrix(source.value(), consumer.value(), mapping.rows,
                                     mapping.columns);
            if (!bound) return bound.error();
        }
        for (const auto& mapping : layout.value().floats) {
            auto source = retarget_layer_name(mapping.source, layer);
            if (!source) return source.error();
            auto consumer = retarget_layer_name(mapping.consumer, layer);
            if (!consumer) return consumer.error();
            auto bound = bind_float(source.value(), consumer.value(), mapping.elements);
            if (!bound) return bound.error();
        }

        const std::string prefix = "blk." + std::to_string(layer);
        const std::string consumer_prefix = "layers." + std::to_string(layer);
        if (layer < config.hash_layer_count) {
            auto table = source_binder.table(prefix + ".ffn_gate_tid2eid.weight",
                                             layout.value().routing_table_elements);
            if (!table) return table.error();
            if (table.value().elements != layout.value().routing_table_elements) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 hash routing table shape does not match its consumer",
                    prefix + ".ffn_gate_tid2eid.weight");
            }
            result.integer_tensors.emplace(consumer_prefix + ".ffn.gate.tid2eid",
                                           table.take_value());
            ++consumed;
        } else {
            // Learned-router layers carry a per-expert bias instead of a lookup table.
            auto bound = bind_float(prefix + ".exp_probs_b.bias",
                                    consumer_prefix + ".ffn.gate.bias", config.expert_count);
            if (!bound) return bound.error();
        }

        const std::size_t ratio = config.compression_ratios[layer];
        if (ratio != 0U) {
            if (ratio != 4U && ratio != 128U) {
                return make_error(ErrorCode::unsupported,
                                  "DeepSeek 0731 layer compression ratio is unsupported",
                                  std::to_string(layer));
            }
            const std::size_t coefficient = ratio == 4U ? 2U : 1U;
            auto projected_width = bounded_product(
                {coefficient, config.head_width}, "DeepSeek 0731 attention compressor width");
            if (!projected_width) return projected_width.error();
            const std::string compressor = consumer_prefix + ".attn.compressor";
            auto bound = bind_matrix(prefix + ".attn_compressor_kv.weight",
                                     compressor + ".wkv.weight", projected_width.value(),
                                     config.width);
            if (!bound) return bound.error();
            bound = bind_matrix(prefix + ".attn_compressor_gate.weight",
                                compressor + ".wgate.weight", projected_width.value(),
                                config.width);
            if (!bound) return bound.error();
            auto ape_elements = bounded_product(
                {ratio, projected_width.value()}, "DeepSeek 0731 compressor APE");
            if (!ape_elements) return ape_elements.error();
            bound = bind_float(prefix + ".attn_compressor_ape.weight", compressor + ".ape",
                               ape_elements.value());
            if (!bound) return bound.error();
            bound = bind_float(prefix + ".attn_compressor_norm.weight",
                               compressor + ".norm.weight", config.head_width);
            if (!bound) return bound.error();

            if (ratio == 4U) {
                auto index_width = bounded_product(
                    {config.index_head_count, config.index_head_width},
                    "DeepSeek 0731 indexer query width");
                if (!index_width) return index_width.error();
                auto index_projected_width = bounded_product(
                    {2U, config.index_head_width}, "DeepSeek 0731 indexer compressor width");
                if (!index_projected_width) return index_projected_width.error();
                const std::string indexer = consumer_prefix + ".attn.indexer";
                bound = bind_matrix(prefix + ".indexer.attn_q_b.weight",
                                    indexer + ".wq_b.weight", index_width.value(),
                                    config.query_lora_rank);
                if (!bound) return bound.error();
                bound = bind_matrix(prefix + ".indexer.proj.weight",
                                    indexer + ".weights_proj.weight",
                                    config.index_head_count, config.width);
                if (!bound) return bound.error();
                bound = bind_matrix(prefix + ".indexer_compressor_kv.weight",
                                    indexer + ".compressor.wkv.weight",
                                    index_projected_width.value(), config.width);
                if (!bound) return bound.error();
                bound = bind_matrix(prefix + ".indexer_compressor_gate.weight",
                                    indexer + ".compressor.wgate.weight",
                                    index_projected_width.value(), config.width);
                if (!bound) return bound.error();
                auto index_ape_elements = bounded_product(
                    {ratio, index_projected_width.value()}, "DeepSeek 0731 indexer APE");
                if (!index_ape_elements) return index_ape_elements.error();
                bound = bind_float(prefix + ".indexer_compressor_ape.weight",
                                   indexer + ".compressor.ape", index_ape_elements.value());
                if (!bound) return bound.error();
                bound = bind_float(prefix + ".indexer_compressor_norm.weight",
                                   indexer + ".compressor.norm.weight",
                                   config.index_head_width);
                if (!bound) return bound.error();
            }
        }

        const std::size_t bound = result.matrices.size() + result.float_tensors.size() +
                                  result.integer_tensors.size();
        if (bound != consumed) {
            return make_error(ErrorCode::internal_error,
                              "DeepSeek 0731 layer trunk mapping is not one-to-one",
                              std::to_string(layer));
        }
        result.tensor_count = consumed;
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded DeepSeek 0731 layer trunk binding");
    }
}

namespace {

// One complete tensor, read under its own reservation and retained by whatever descriptor binds it.
Result<std::pair<model::LoadedTensorTile, const model::LogicalTensorDescriptor*>>
load_catalog_tensor_tile(const model::LogicalTensorCatalog& catalog,
                         std::string_view name,
                         std::optional<std::size_t> expected_layer,
                         safety::ResourceGovernor& governor) {
    const auto found = std::find_if(
        catalog.tensors.begin(), catalog.tensors.end(),
        [name](const model::LogicalTensorDescriptor& tensor) { return tensor.name == name; });
    if (found == catalog.tensors.end() || found->expert.has_value() ||
        found->layer.has_value() != expected_layer.has_value() ||
        (expected_layer.has_value() && *found->layer != *expected_layer)) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek tensor is absent from the catalog", std::string(name));
    }
    model::LayerTensorTile tile;
    tile.tensor_index = static_cast<std::size_t>(std::distance(catalog.tensors.begin(), found));
    tile.layer = found->layer.has_value() ? *found->layer : 0U;
    tile.role = found->role;
    tile.tensor_offset = 0U;
    tile.source = found->source;
    tile.final_tile = true;
    tile.checkpoint_global = !found->layer.has_value();
    auto residency = model::ResidencyObject::create(
        catalog.model_id + "/tensor/" + found->name, found->source.bytes);
    if (!residency) return residency.error();
    model::TensorTileLoadOptions options;
    options.max_tile_bytes = found->source.bytes;
    options.measure_process_io = false;
    auto loaded = model::load_tensor_tile(catalog, tile, residency.value(), governor, options);
    if (!loaded) return loaded.error();
    return std::pair<model::LoadedTensorTile, const model::LogicalTensorDescriptor*>(
        loaded.take_value(), &*found);
}

}  // namespace

Result<DeepseekV4ReferenceGlobals> deepseek_v4_load_0731_gguf_globals(
    const model::LogicalTensorCatalog& catalog,
    const DeepseekV4ReferenceConfig& config,
    safety::ResourceGovernor& governor) {
    if (catalog.metadata_only) {
        return make_error(ErrorCode::unsupported,
                          "Loading globals from a metadata-only catalog is forbidden",
                          catalog.model_id);
    }
    try {
        struct MatrixMapping {
            std::string_view source;
            std::string_view consumer;
            std::size_t rows;
            std::size_t columns;
        };
        const std::array<MatrixMapping, 3U> matrices{{
            {"token_embd.weight", "embed.weight", config.vocabulary_size, config.width},
            {"output.weight", "head.weight", config.vocabulary_size, config.width},
            {"output_hc_fn.weight", "hc_head_fn", config.hyper_multiplier,
             config.hyper_multiplier * config.width},
        }};
        struct FloatMapping {
            std::string_view source;
            std::string_view consumer;
            std::size_t elements;
        };
        const std::array<FloatMapping, 3U> floats{{
            {"output_norm.weight", "norm.weight", config.width},
            {"output_hc_base.weight", "hc_head_base", config.hyper_multiplier},
            {"output_hc_scale.weight", "hc_head_scale", 1U},
        }};

        DeepseekV4ReferenceGlobals result;
        for (const auto& mapping : matrices) {
            auto tile = load_catalog_tensor_tile(catalog, mapping.source, std::nullopt, governor);
            if (!tile) return tile.error();
            const auto* descriptor = tile.value().second;
            if (!descriptor->ggml_type.has_value() || descriptor->shape.size() != 2U ||
                static_cast<std::size_t>(descriptor->shape[1]) != mapping.rows) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek global matrix shape does not match its consumer",
                                  std::string(mapping.source));
            }
            auto matrix = deepseek_v4_bind_packed_matrix(
                tile.value().first, *descriptor->ggml_type, mapping.rows, mapping.columns,
                static_cast<std::size_t>(descriptor->shape[0]));
            if (!matrix) return matrix.error();
            result.matrices.emplace(mapping.consumer, matrix.take_value());
        }
        for (const auto& mapping : floats) {
            auto tile = load_catalog_tensor_tile(catalog, mapping.source, std::nullopt, governor);
            if (!tile) return tile.error();
            const auto* descriptor = tile.value().second;
            if (!descriptor->ggml_type.has_value()) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek global vector descriptor is malformed",
                                  std::string(mapping.source));
            }
            auto elements = bounded_catalog_product(descriptor->shape, descriptor->name);
            if (!elements) return elements.error();
            if (elements.value() != mapping.elements) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek global vector shape does not match its consumer",
                                  std::string(mapping.source));
            }
            auto tensor = deepseek_v4_bind_packed_float(
                tile.value().first, *descriptor->ggml_type, elements.value());
            if (!tensor) return tensor.error();
            result.float_tensors.emplace(mapping.consumer, tensor.take_value());
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the streamed DeepSeek globals");
    }
}

Result<DeepseekV4ReferenceLayerTrunk> deepseek_v4_load_0731_gguf_layer_trunk(
    const model::LogicalTensorCatalog& catalog,
    const DeepseekV4ReferenceConfig& config,
    std::size_t layer,
    safety::ResourceGovernor& governor) {
    if (catalog.metadata_only) {
        return make_error(ErrorCode::unsupported,
                          "Loading a layer trunk from a metadata-only catalog is forbidden",
                          catalog.model_id);
    }
    // Each always-active tensor is read as one complete tile and retained by the descriptor it
    // produces, so the layer's residency survives exactly as long as the graph that uses it.
    const auto load_tile = [&](std::string_view name) {
        return load_catalog_tensor_tile(catalog, name, layer, governor);
    };

    DeepseekV4LayerTrunkSource binder;
    binder.matrix = [&load_tile](std::string_view name, std::size_t logical_columns)
        -> Result<DeepseekV4PackedMatrix> {
        auto tile = load_tile(name);
        if (!tile) return tile.error();
        const auto* descriptor = tile.value().second;
        if (!descriptor->ggml_type.has_value() || descriptor->shape.size() != 2U) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek layer matrix descriptor is malformed", descriptor->name);
        }
        return deepseek_v4_bind_packed_matrix(
            tile.value().first, *descriptor->ggml_type,
            static_cast<std::size_t>(descriptor->shape[1]), logical_columns,
            static_cast<std::size_t>(descriptor->shape[0]));
    };
    binder.vector = [&load_tile](std::string_view name) -> Result<DeepseekV4PackedFloatTensor> {
        auto tile = load_tile(name);
        if (!tile) return tile.error();
        const auto* descriptor = tile.value().second;
        if (!descriptor->ggml_type.has_value()) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek layer vector descriptor is malformed", descriptor->name);
        }
        auto elements = bounded_catalog_product(descriptor->shape, descriptor->name);
        if (!elements) return elements.error();
        return deepseek_v4_bind_packed_float(tile.value().first, *descriptor->ggml_type,
                                             elements.value());
    };
    binder.table = [&load_tile](std::string_view name, std::size_t elements)
        -> Result<DeepseekV4PackedIntegerTensor> {
        auto tile = load_tile(name);
        if (!tile) return tile.error();
        return deepseek_v4_bind_packed_i32(tile.value().first, elements);
    };
    return deepseek_v4_build_layer_trunk(binder, config, layer);
}

Result<DeepseekV4ReferenceLayerTrunk> deepseek_v4_bind_0731_gguf_layer_trunk(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    const DeepseekV4ReferenceConfig& config,
    std::size_t layer) {
    DeepseekV4LayerTrunkSource binder;
    binder.matrix = [&tensors](std::string_view name, std::size_t logical_columns) {
        return deepseek_v4_bind_packed_matrix(tensors, name, logical_columns);
    };
    binder.vector = [&tensors](std::string_view name) {
        return deepseek_v4_bind_packed_float(tensors, name);
    };
    binder.table = [&tensors](std::string_view name, std::size_t) {
        return deepseek_v4_bind_packed_i32(tensors, name);
    };
    auto trunk = deepseek_v4_build_layer_trunk(binder, config, layer);
    if (!trunk) return trunk.error();
    if (trunk.value().tensor_count != tensors.tensors().size()) {
        return make_error(
            ErrorCode::invalid_format,
            "DeepSeek 0731 layer trunk binding did not consume every loaded tensor",
            std::to_string(layer));
    }
    return trunk;
}

Result<DeepseekV4ReferenceGlobals> deepseek_v4_bind_0731_gguf_globals(
    const DeepseekV4LoadedGlobalTensorSet& tensors,
    const DeepseekV4ReferenceConfig& config) {
    try {
        if (tensors.tensors().size() != 6U || config.vocabulary_size == 0U ||
            config.width == 0U || config.hyper_multiplier == 0U ||
            config.hyper_multiplier >
                std::numeric_limits<std::size_t>::max() / config.width) {
            return make_error(
                ErrorCode::invalid_format,
                "DeepSeek 0731 global binding requires six tensors and valid consumer dimensions");
        }
        struct MatrixMapping {
            std::string_view source;
            std::string_view consumer;
            std::size_t rows;
            std::size_t columns;
        };
        const std::array<MatrixMapping, 3U> matrices{{
            {"token_embd.weight", "embed.weight", config.vocabulary_size,
             config.width},
            {"output.weight", "head.weight", config.vocabulary_size,
             config.width},
            {"output_hc_fn.weight", "hc_head_fn", config.hyper_multiplier,
             config.hyper_multiplier * config.width},
        }};
        struct FloatMapping {
            std::string_view source;
            std::string_view consumer;
            std::size_t elements;
        };
        const std::array<FloatMapping, 3U> floats{{
            {"output_norm.weight", "norm.weight", config.width},
            {"output_hc_base.weight", "hc_head_base", config.hyper_multiplier},
            {"output_hc_scale.weight", "hc_head_scale", 1U},
        }};

        DeepseekV4ReferenceGlobals result;
        for (const auto& mapping : matrices) {
            auto matrix = deepseek_v4_bind_packed_matrix(
                tensors, mapping.source, mapping.columns);
            if (!matrix) return matrix.error();
            if (matrix.value().rows != mapping.rows) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 global matrix shape does not match its consumer",
                    std::string(mapping.source));
            }
            result.matrices.emplace(mapping.consumer, matrix.take_value());
        }
        for (const auto& mapping : floats) {
            auto tensor = deepseek_v4_bind_packed_float(tensors, mapping.source);
            if (!tensor) return tensor.error();
            if (tensor.value().elements != mapping.elements) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek 0731 global float shape does not match its consumer",
                    std::string(mapping.source));
            }
            result.float_tensors.emplace(mapping.consumer, tensor.take_value());
        }
        if (result.matrices.size() != matrices.size() ||
            result.float_tensors.size() != floats.size()) {
            return make_error(
                ErrorCode::internal_error,
                "DeepSeek 0731 global mapping is not one-to-one");
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate bounded DeepSeek 0731 global binding");
    }
}

Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_prefill(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> token_ids,
    DeepseekV4ReferenceState* state,
    const DeepseekV4ReferenceExecutionControl* control) {
    auto valid = validate_graph(graph, token_ids);
    if (!valid) return valid.error();
    valid = execution_checkpoint(control);
    if (!valid) return valid.error();
    const auto& config = graph.config;
    const std::size_t token_count = token_ids.size();
    if (state != nullptr) {
        auto budget = validate_state_budget(config, token_count);
        if (!budget) return budget.error();
    }
    DeepseekV4ReferenceState working_state;
    if (state != nullptr) {
        working_state.layers.resize(config.layer_count);
        working_state.mtp_layers.resize(config.mtp_layer_count);
    }
    auto embedding = embedding_rows(graph, token_ids);
    if (!embedding) return embedding.error();
    auto hidden_count = bounded_product(
        {token_count, config.hyper_multiplier, config.width}, "DeepSeek hidden state");
    if (!hidden_count) return hidden_count.error();
    std::vector<float> hidden(hidden_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
            std::copy_n(embedding.value().begin() +
                            static_cast<std::ptrdiff_t>(token * config.width),
                        config.width,
                        hidden.begin() + static_cast<std::ptrdiff_t>(
                            (token * config.hyper_multiplier + copy) * config.width));
        }
    }
    DeepseekV4ReferenceForwardResult result;
    result.layers.reserve(config.layer_count);
    for (std::size_t layer = 0U; layer < config.layer_count; ++layer) {
        valid = execution_checkpoint(control);
        if (!valid) return valid.error();
        const std::string layer_base = "layers." + std::to_string(layer);
        auto block = block_prefill(graph, layer_base, layer,
                                   config.compression_ratios[layer],
                                   layer < config.hash_layer_count, hidden, token_ids,
                                   state == nullptr ? nullptr : &working_state.layers[layer]);
        if (!block) return block.error();
        hidden = std::move(block.value().hidden);
        result.layers.push_back(std::move(block.value().trace));
    }
    valid = execution_checkpoint(control);
    if (!valid) return valid.error();
    auto logits = final_logits(graph, hidden, token_count, "");
    if (!logits) return logits.error();
    result.logits = logits.take_value();
    const auto maximum = std::max_element(result.logits.begin(), result.logits.end());
    result.greedy_token_id = static_cast<std::uint32_t>(
        std::distance(result.logits.begin(), maximum));

    if (config.mtp_layer_count != 0U) {
        valid = execution_checkpoint(control);
        if (!valid) return valid.error();
        const std::string mtp_base = "mtp.0";
        auto embedding_norm_weight = float_tensor(graph, mtp_base + ".enorm.weight",
                                                  {config.width});
        if (!embedding_norm_weight) return embedding_norm_weight.error();
        auto normalized_embedding = deepseek_v4_rms_norm(
            embedding.value(), embedding_norm_weight.value(), token_count, config.width,
            config.norm_epsilon);
        if (!normalized_embedding) return normalized_embedding.error();
        auto hidden_norm_weight = float_tensor(graph, mtp_base + ".hnorm.weight",
                                               {config.width});
        if (!hidden_norm_weight) return hidden_norm_weight.error();
        auto normalized_hidden = deepseek_v4_rms_norm(
            hidden, hidden_norm_weight.value(), token_count * config.hyper_multiplier,
            config.width, config.norm_epsilon);
        if (!normalized_hidden) return normalized_hidden.error();
        auto embedding_projection = linear_tensor(
            graph, mtp_base + ".e_proj.weight", normalized_embedding.value(),
            token_count, config.width, config.width, mtp_base + ".e_proj");
        if (!embedding_projection) return embedding_projection.error();
        auto hidden_projection = linear_tensor(
            graph, mtp_base + ".h_proj.weight", normalized_hidden.value(),
            token_count * config.hyper_multiplier, config.width, config.width,
            mtp_base + ".h_proj");
        if (!hidden_projection) return hidden_projection.error();
        std::vector<float> mtp_input = hidden_projection.take_value();
        for (std::size_t token = 0U; token < token_count; ++token) {
            for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
                for (std::size_t column = 0U; column < config.width; ++column) {
                    mtp_input[(token * config.hyper_multiplier + copy) * config.width + column] +=
                        embedding_projection.value()[token * config.width + column];
                }
            }
        }
        auto mtp = block_prefill(graph, mtp_base, 0U,
                                 config.mtp_compression_ratios[0], false,
                                 mtp_input, token_ids,
                                 state == nullptr ? nullptr : &working_state.mtp_layers[0]);
        if (!mtp) return mtp.error();
        result.mtp_layers.push_back(std::move(mtp.value().trace));
        auto mtp_logits = final_logits(graph, mtp.value().hidden, token_count,
                                       mtp_base + ".");
        if (!mtp_logits) return mtp_logits.error();
        result.mtp_logits = mtp_logits.take_value();
        const auto mtp_maximum = std::max_element(result.mtp_logits.begin(),
                                                  result.mtp_logits.end());
        result.mtp_greedy_token_id = static_cast<std::uint32_t>(
            std::distance(result.mtp_logits.begin(), mtp_maximum));
    }
    if (state != nullptr) {
        working_state.next_position = token_count;
        auto state_valid = validate_reference_state(graph, working_state);
        if (!state_valid) return state_valid.error();
        state_valid = execution_checkpoint(control);
        if (!state_valid) return state_valid.error();
        *state = std::move(working_state);
    }
    return result;
}

Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_decode(
    const DeepseekV4ReferenceGraph& graph,
    std::uint32_t token_id,
    DeepseekV4ReferenceState& state,
    const DeepseekV4ReferenceExecutionControl* control) {
    const std::array<std::uint32_t, 1U> token_ids{token_id};
    auto graph_valid = validate_graph(graph, token_ids);
    if (!graph_valid) return graph_valid.error();
    auto state_valid = validate_reference_state(graph, state);
    if (!state_valid) return state_valid.error();
    state_valid = execution_checkpoint(control);
    if (!state_valid) return state_valid.error();
    if (state.next_position >= kMaximumReferenceTokens) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek incremental state reached its token limit");
    }
    const auto& config = graph.config;
    const std::size_t start_position = state.next_position;
    auto budget = validate_state_budget(config, start_position + 1U);
    if (!budget) return budget.error();
    DeepseekV4ReferenceState working_state = state;
    auto embedding = embedding_rows(graph, token_ids);
    if (!embedding) return embedding.error();
    std::vector<float> embedded = embedding.take_value();
    auto hidden_count = bounded_product(
        {config.hyper_multiplier, config.width}, "DeepSeek incremental hidden state");
    if (!hidden_count) return hidden_count.error();
    std::vector<float> hidden(hidden_count.value());
    for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
        std::copy_n(embedded.begin(), config.width,
                    hidden.begin() +
                        static_cast<std::ptrdiff_t>(copy * config.width));
    }

    DeepseekV4ReferenceForwardResult result;
    result.layers.reserve(config.layer_count);
    for (std::size_t layer = 0U; layer < config.layer_count; ++layer) {
        state_valid = execution_checkpoint(control);
        if (!state_valid) return state_valid.error();
        const std::string layer_base = "layers." + std::to_string(layer);
        auto block = block_incremental(
            graph, layer_base, layer, config.compression_ratios[layer],
            layer < config.hash_layer_count, hidden, token_id, start_position,
            working_state.layers[layer]);
        if (!block) return block.error();
        hidden = std::move(block.value().hidden);
        result.layers.push_back(std::move(block.value().trace));
    }
    state_valid = execution_checkpoint(control);
    if (!state_valid) return state_valid.error();
    auto logits = final_logits(graph, hidden, 1U, "");
    if (!logits) return logits.error();
    result.logits = logits.take_value();
    const auto maximum = std::max_element(result.logits.begin(), result.logits.end());
    result.greedy_token_id = static_cast<std::uint32_t>(
        std::distance(result.logits.begin(), maximum));

    if (config.mtp_layer_count != 0U) {
        state_valid = execution_checkpoint(control);
        if (!state_valid) return state_valid.error();
        const std::string mtp_base = "mtp.0";
        auto embedding_norm_weight = float_tensor(graph, mtp_base + ".enorm.weight",
                                                  {config.width});
        if (!embedding_norm_weight) return embedding_norm_weight.error();
        auto normalized_embedding = deepseek_v4_rms_norm(
            embedded, embedding_norm_weight.value(), 1U, config.width,
            config.norm_epsilon);
        if (!normalized_embedding) return normalized_embedding.error();
        auto hidden_norm_weight = float_tensor(graph, mtp_base + ".hnorm.weight",
                                               {config.width});
        if (!hidden_norm_weight) return hidden_norm_weight.error();
        auto normalized_hidden = deepseek_v4_rms_norm(
            hidden, hidden_norm_weight.value(), config.hyper_multiplier,
            config.width, config.norm_epsilon);
        if (!normalized_hidden) return normalized_hidden.error();
        auto embedding_projection = linear_tensor(
            graph, mtp_base + ".e_proj.weight", normalized_embedding.value(), 1U,
            config.width, config.width, mtp_base + ".e_proj incremental");
        if (!embedding_projection) return embedding_projection.error();
        auto hidden_projection = linear_tensor(
            graph, mtp_base + ".h_proj.weight", normalized_hidden.value(),
            config.hyper_multiplier, config.width, config.width,
            mtp_base + ".h_proj incremental");
        if (!hidden_projection) return hidden_projection.error();
        std::vector<float> mtp_input = hidden_projection.take_value();
        for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
            for (std::size_t column = 0U; column < config.width; ++column) {
                mtp_input[copy * config.width + column] +=
                    embedding_projection.value()[column];
            }
        }
        auto mtp = block_incremental(
            graph, mtp_base, 0U, config.mtp_compression_ratios[0], false,
            mtp_input, token_id, start_position, working_state.mtp_layers[0]);
        if (!mtp) return mtp.error();
        result.mtp_layers.push_back(std::move(mtp.value().trace));
        auto mtp_logits = final_logits(graph, mtp.value().hidden, 1U, mtp_base + ".");
        if (!mtp_logits) return mtp_logits.error();
        result.mtp_logits = mtp_logits.take_value();
        const auto mtp_maximum = std::max_element(result.mtp_logits.begin(),
                                                  result.mtp_logits.end());
        result.mtp_greedy_token_id = static_cast<std::uint32_t>(
            std::distance(result.mtp_logits.begin(), mtp_maximum));
    }
    working_state.next_position = start_position + 1U;
    state_valid = validate_reference_state(graph, working_state);
    if (!state_valid) return state_valid.error();
    state_valid = execution_checkpoint(control);
    if (!state_valid) return state_valid.error();
    state = std::move(working_state);
    return result;
}

Result<DeepseekV4ReferenceGenerationResult> deepseek_v4_reference_generate(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> prompt_ids,
    const DeepseekV4ReferenceGenerationOptions& options,
    const safety::ProcessSafetyGuard* process_guard) {
    if (options.max_new_tokens == 0U ||
        options.max_new_tokens > kDeepseekV4ReferenceMaxTokens ||
        options.context_limit_tokens == 0U ||
        options.context_limit_tokens > kDeepseekV4ReferenceMaxTokens ||
        prompt_ids.empty() || prompt_ids.size() >= options.context_limit_tokens ||
        !std::isfinite(options.temperature) || options.temperature < 0.0F ||
        options.temperature > 10.0F || options.top_k == 0U ||
        options.top_k > graph.config.vocabulary_size ||
        options.timeout < std::chrono::milliseconds(1) ||
        options.timeout > std::chrono::hours(1)) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference generation options are invalid");
    }
    if (options.stop_token_ids.size() > 64U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek generation has too many stop tokens");
    }
    for (std::size_t index = 0U; index < options.stop_token_ids.size(); ++index) {
        if (options.stop_token_ids[index] >= graph.config.vocabulary_size ||
            std::find(options.stop_token_ids.begin(),
                      options.stop_token_ids.begin() +
                          static_cast<std::ptrdiff_t>(index),
                      options.stop_token_ids[index]) !=
                options.stop_token_ids.begin() +
                    static_cast<std::ptrdiff_t>(index)) {
            return make_error(
                ErrorCode::invalid_argument,
                "DeepSeek generation stop-token list is invalid or duplicated");
        }
    }

    enum class RequestedStop { none, cancelled, timeout };
    RequestedStop requested_stop = RequestedStop::none;
    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + options.timeout;
    DeepseekV4ReferenceExecutionControl control;
    control.checkpoint = [&]() -> Result<void> {
        if (std::chrono::steady_clock::now() >= deadline) {
            requested_stop = RequestedStop::timeout;
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference generation timed out safely");
        }
        if (options.cancelled && options.cancelled()) {
            requested_stop = RequestedStop::cancelled;
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference generation was cancelled safely");
        }
        if (process_guard != nullptr) {
            auto safe = process_guard->sample_and_check();
            if (!safe) return safe.error();
        }
        return {};
    };

    DeepseekV4ReferenceGenerationResult result;
    result.stats.prompt_tokens = prompt_ids.size();
    result.stats.requested_max_tokens = options.max_new_tokens;
    result.stats.effective_max_tokens = std::min(
        options.max_new_tokens, options.context_limit_tokens - prompt_ids.size());
    const auto finish_requested_stop = [&]() {
        if (requested_stop == RequestedStop::cancelled) {
            result.finish_reason = DeepseekV4ReferenceFinishReason::cancelled;
        } else if (requested_stop == RequestedStop::timeout) {
            result.finish_reason = DeepseekV4ReferenceFinishReason::timeout;
        }
    };

    try {
        result.prompt_ids.assign(prompt_ids.begin(), prompt_ids.end());
        result.generated_ids.reserve(result.stats.effective_max_tokens);
        DeepseekV4ReferenceState state;
        auto prefill = deepseek_v4_reference_prefill(
            graph, prompt_ids, &state, &control);
        const auto prompt_finished = std::chrono::steady_clock::now();
        result.stats.prompt_eval_ms = std::chrono::duration<double, std::milli>(
            prompt_finished - started).count();
        if (!prefill) {
            if (requested_stop != RequestedStop::none) {
                finish_requested_stop();
                return result;
            }
            return prefill.error();
        }
        result.stats.forward_passes = 1U;
        auto telemetry = accumulate_generation_telemetry(
            prefill.value(), result.stats);
        if (!telemetry) return telemetry.error();

        std::mt19937_64 random(options.seed);
        DeepseekV4ReferenceForwardResult current = prefill.take_value();
        for (std::size_t generated = 0U;
             generated < result.stats.effective_max_tokens; ++generated) {
            auto safe = execution_checkpoint(&control);
            if (!safe) {
                if (requested_stop != RequestedStop::none) {
                    finish_requested_stop();
                    break;
                }
                return safe.error();
            }
            auto sampled = sample_reference_token(current.logits, options, random);
            if (!sampled) return sampled.error();
            const std::uint32_t token = sampled.value();
            if (std::find(options.stop_token_ids.begin(),
                          options.stop_token_ids.end(), token) !=
                options.stop_token_ids.end()) {
                result.finish_reason = DeepseekV4ReferenceFinishReason::stop_token;
                result.stop_token_id = token;
                break;
            }

            result.generated_ids.push_back(token);
            if (options.on_token) {
                try {
                    options.on_token(token);
                } catch (const std::bad_alloc&) {
                    return make_error(
                        ErrorCode::resource_exhausted,
                        "DeepSeek generation token callback exhausted memory");
                } catch (const std::exception& error) {
                    return make_error(
                        ErrorCode::internal_error,
                        "DeepSeek generation token callback raised an exception",
                        error.what());
                } catch (...) {
                    return make_error(
                        ErrorCode::internal_error,
                        "DeepSeek generation token callback raised an unknown exception");
                }
            }

            if (generated + 1U == result.stats.effective_max_tokens) break;
            auto decoded = deepseek_v4_reference_decode(
                graph, token, state, &control);
            if (!decoded) {
                if (requested_stop != RequestedStop::none) {
                    finish_requested_stop();
                    break;
                }
                return decoded.error();
            }
            if (result.stats.forward_passes ==
                std::numeric_limits<std::size_t>::max()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek generation forward count overflows");
            }
            ++result.stats.forward_passes;
            telemetry = accumulate_generation_telemetry(
                decoded.value(), result.stats);
            if (!telemetry) return telemetry.error();
            current = decoded.take_value();
        }

        result.stats.generated_tokens = result.generated_ids.size();
        if (requested_stop == RequestedStop::none &&
            result.finish_reason != DeepseekV4ReferenceFinishReason::stop_token) {
            result.finish_reason =
                result.stats.effective_max_tokens < options.max_new_tokens
                    ? DeepseekV4ReferenceFinishReason::context_limit
                    : DeepseekV4ReferenceFinishReason::max_tokens;
        }
        result.stats.generation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prompt_finished).count();
        if (result.stats.generation_ms > 0.0) {
            result.stats.tokens_per_second =
                static_cast<double>(result.stats.generated_tokens) * 1000.0 /
                result.stats.generation_ms;
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate the bounded DeepSeek reference generation state");
    }
}

std::string_view to_string(DeepseekV4ReferenceFinishReason reason) noexcept {
    switch (reason) {
        case DeepseekV4ReferenceFinishReason::stop_token: return "stop_token";
        case DeepseekV4ReferenceFinishReason::context_limit: return "context_limit";
        case DeepseekV4ReferenceFinishReason::max_tokens: return "max_tokens";
        case DeepseekV4ReferenceFinishReason::cancelled: return "cancelled";
        case DeepseekV4ReferenceFinishReason::timeout: return "timeout";
    }
    return "unknown";
}

}  // namespace vastloom::inference
