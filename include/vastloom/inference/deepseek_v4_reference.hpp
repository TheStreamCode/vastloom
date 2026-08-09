#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/status.hpp"

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

struct DeepseekV4ReferenceGraph {
    DeepseekV4ReferenceConfig config;
    std::map<std::string, std::vector<float>, std::less<>> float_tensors;
    std::map<std::string, std::vector<std::uint32_t>, std::less<>> integer_tensors;
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

// Bounded single-batch prefill oracle for reduced float fixtures. This deliberately
// excludes incremental cache mutation, quantized checkpoint kernels and MTP execution.
[[nodiscard]] Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_prefill(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> token_ids);

}  // namespace vastloom::inference
