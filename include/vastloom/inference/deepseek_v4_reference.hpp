#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::model {
class LoadedExpert;
class LoadedTensorTile;
class ResidentByteBuffer;
}  // namespace vastloom::model

namespace vastloom::inference {

struct DeepseekV4RouteResult {
    std::vector<float> weights;
    std::vector<std::uint32_t> indices;
    std::size_t token_count{0};
    std::size_t top_k{0};
};

struct DeepseekV4HyperConnectionSplit {
    std::vector<float> pre;
    std::vector<float> post;
    std::vector<float> combination;
    std::size_t token_count{0};
    std::size_t multiplier{0};
};

struct DeepseekV4HyperConnectionPreResult {
    std::vector<float> reduced;
    DeepseekV4HyperConnectionSplit split;
    std::size_t width{0};
};

struct DeepseekV4ReferenceConfig {
    std::size_t vocabulary_size{0};
    std::size_t width{0};
    std::size_t intermediate_width{0};
    std::size_t layer_count{0};
    std::size_t hash_layer_count{0};
    std::size_t mtp_layer_count{0};
    std::size_t head_count{0};
    std::size_t expert_count{0};
    std::size_t experts_per_token{0};
    std::size_t query_lora_rank{0};
    std::size_t head_width{0};
    std::size_t rope_width{0};
    std::size_t output_groups{0};
    std::size_t output_lora_rank{0};
    std::size_t window_size{0};
    std::vector<std::size_t> compression_ratios;
    std::vector<std::size_t> mtp_compression_ratios;
    float compression_rope_theta{0.0F};
    std::size_t original_context{0};
    float rope_theta{0.0F};
    float rope_factor{0.0F};
    float beta_fast{0.0F};
    float beta_slow{0.0F};
    std::size_t index_head_count{0};
    std::size_t index_head_width{0};
    std::size_t index_top_k{0};
    std::size_t hyper_multiplier{0};
    std::size_t sinkhorn_iterations{0};
    float norm_epsilon{0.0F};
    float hyper_epsilon{0.0F};
    float route_scale{0.0F};
    float swiglu_limit{0.0F};
};

struct DeepseekV4PackedMatrix {
    DeepseekV4PackedMatrix() = default;
    DeepseekV4PackedMatrix(std::uint32_t type,
                           std::size_t output_rows,
                           std::size_t input_columns,
                           std::size_t physical_columns,
                           std::vector<std::byte> bytes)
        : ggml_type(type),
          rows(output_rows),
          logical_columns(input_columns),
          storage_columns(physical_columns),
          encoded(std::move(bytes)) {}

    std::uint32_t ggml_type{0U};
    std::size_t rows{0U};
    std::size_t logical_columns{0U};
    std::size_t storage_columns{0U};
    std::vector<std::byte> encoded;
    std::shared_ptr<const model::ResidentByteBuffer> resident;
    std::size_t resident_offset{0U};
    std::size_t resident_bytes{0U};
};

struct DeepseekV4PackedIntegerTensor {
    DeepseekV4PackedIntegerTensor() = default;
    DeepseekV4PackedIntegerTensor(std::size_t element_count,
                                  std::vector<std::byte> bytes)
        : elements(element_count), encoded(std::move(bytes)) {}

    std::size_t elements{0U};
    std::vector<std::byte> encoded;
    std::shared_ptr<const model::ResidentByteBuffer> resident;
    std::size_t resident_offset{0U};
    std::size_t resident_bytes{0U};
};

// Bind exact complete tensors from bounded file loaders without copying their payload. The
// returned descriptor retains the loader's ResidencyUse through a shared immutable buffer.
[[nodiscard]] Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const model::LoadedTensorTile& tile,
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns);

[[nodiscard]] Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const model::LoadedExpert& expert,
    std::size_t catalog_tensor_index,
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns);

[[nodiscard]] Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const model::LoadedTensorTile& tile,
    std::size_t elements);

[[nodiscard]] Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const model::LoadedExpert& expert,
    std::size_t catalog_tensor_index,
    std::size_t elements);

struct DeepseekV4ReferenceGraph {
    DeepseekV4ReferenceConfig config;
    std::map<std::string, std::vector<float>, std::less<>> float_tensors;
    std::map<std::string, std::vector<std::uint32_t>, std::less<>> integer_tensors;
    std::map<std::string, DeepseekV4PackedMatrix, std::less<>> packed_matrices;
    std::map<std::string, DeepseekV4PackedIntegerTensor, std::less<>> packed_integer_tensors;
};

struct DeepseekV4ReferenceCompressorState {
    std::vector<float> key_values;
    std::vector<float> scores;
    std::vector<float> compressed;
};

struct DeepseekV4ReferenceAttentionState {
    std::vector<float> window_key_values;
    DeepseekV4ReferenceCompressorState compressor;
    DeepseekV4ReferenceCompressorState indexer;
};

struct DeepseekV4ReferenceState {
    std::size_t next_position{0};
    std::vector<DeepseekV4ReferenceAttentionState> layers;
    std::vector<DeepseekV4ReferenceAttentionState> mtp_layers;
};

struct DeepseekV4LayerTrace {
    std::size_t layer{0};
    DeepseekV4RouteResult route;
    std::vector<std::int32_t> attention_indices;
    std::size_t attention_selected_per_token{0};
};

struct DeepseekV4ReferenceForwardResult {
    std::vector<float> logits;
    std::vector<DeepseekV4LayerTrace> layers;
    std::uint32_t greedy_token_id{0};
    std::vector<float> mtp_logits;
    std::vector<DeepseekV4LayerTrace> mtp_layers;
    std::uint32_t mtp_greedy_token_id{0};
};

[[nodiscard]] Result<std::vector<float>> deepseek_v4_rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t row_count,
    std::size_t width,
    float epsilon);

[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_learned(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const float> selection_bias,
    std::size_t token_count,
    std::size_t width,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_hash(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> token_to_experts,
    std::size_t token_count,
    std::size_t width,
    std::size_t vocabulary_size,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

// These entry points accept the raw router projection before sqrt-softplus. They let the
// architecture graph consume a packed GGML matrix without materializing a float weight matrix.
[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_learned_logits(
    std::span<const float> logits,
    std::span<const float> selection_bias,
    std::size_t token_count,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_hash_logits(
    std::span<const float> logits,
    std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> token_to_experts,
    std::size_t token_count,
    std::size_t vocabulary_size,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_expert_swiglu(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const float> down_weight,
    std::span<const float> up_weight,
    std::span<const float> routing_weights,
    std::size_t token_count,
    std::size_t width,
    std::size_t intermediate_width,
    float swiglu_limit);

[[nodiscard]] Result<DeepseekV4HyperConnectionSplit> deepseek_v4_hc_split_sinkhorn(
    std::span<const float> mixes,
    std::span<const float> scale,
    std::span<const float> base,
    std::size_t token_count,
    std::size_t multiplier,
    std::size_t sinkhorn_iterations,
    float epsilon);

[[nodiscard]] Result<DeepseekV4HyperConnectionPreResult> deepseek_v4_hc_pre(
    std::span<const float> input,
    std::span<const float> mixing_weight,
    std::span<const float> scale,
    std::span<const float> base,
    std::size_t token_count,
    std::size_t multiplier,
    std::size_t width,
    std::size_t sinkhorn_iterations,
    float norm_epsilon,
    float hc_epsilon);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_hc_post(
    std::span<const float> sublayer,
    std::span<const float> residual,
    const DeepseekV4HyperConnectionSplit& split,
    std::size_t width);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_apply_rope(
    std::span<const float> input,
    std::span<const std::uint32_t> positions,
    std::size_t row_count,
    std::size_t head_count,
    std::size_t head_width,
    float theta,
    bool inverse = false);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_sparse_attention(
    std::span<const float> query,
    std::span<const float> key_value,
    std::span<const float> attention_sink,
    std::span<const std::int32_t> selected_indices,
    std::size_t query_count,
    std::size_t key_value_count,
    std::size_t head_count,
    std::size_t head_width,
    std::size_t selected_per_query,
    float softmax_scale);

// Bounded single-batch oracle for reduced float fixtures. Passing a state captures the
// raw/compressed caches needed by deepseek_v4_reference_decode. Quantized checkpoint
// kernels remain outside this reference path.
[[nodiscard]] Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_prefill(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> token_ids,
    DeepseekV4ReferenceState* state = nullptr);

[[nodiscard]] Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_decode(
    const DeepseekV4ReferenceGraph& graph,
    std::uint32_t token_id,
    DeepseekV4ReferenceState& state);

}  // namespace vastloom::inference
