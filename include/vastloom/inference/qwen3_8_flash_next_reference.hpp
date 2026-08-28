#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

// Qwen3.8-Flash-Next reference operators — pure, deterministic, no I/O.
//
// These are the ground-truth scalar kernels that the streaming implementation
// must match bit-exactly. They are intentionally slow and single-threaded.

// FP8 E4M3 (OCP) -> F32. 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
// Exponent 15 is NaN (no Inf in E4M3). Subnormals are supported.
[[nodiscard]] float qwen38_fp8_e4m3_to_f32(std::uint8_t bits) noexcept;

// BF16 (1-8-7) -> F32. Bit-exact: payload is the high 16 bits of an F32.
[[nodiscard]] float qwen38_bf16_to_f32(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t qwen38_f32_to_bf16(float value) noexcept;

// Decode a BF16 tensor payload into F32. Payload must contain exactly
// product(shape) * 2 bytes. Returns F32 vector in row-major order.
[[nodiscard]] Result<std::vector<float>> qwen38_decode_bf16(
    std::span<const std::byte> payload,
    const std::vector<std::int64_t>& shape);

// Decode an FP8 block-scaled tensor. Weight payload is F8_E4M3 bytes,
// scale payload is BF16 per block (block_rows x block_cols, default 128x128).
// Shape is the logical shape of the weight tensor (e.g., [640, 2560]).
// Returns F32 vector: weight_f32[i] = fp8_to_f32(weight_bytes[i]) * bf16_to_f32(scale_for_block(i))
[[nodiscard]] Result<std::vector<float>> qwen38_decode_fp8_block_scale(
    std::span<const std::byte> weight_payload,
    std::span<const std::byte> scale_payload,
    const std::vector<std::int64_t>& weight_shape,
    const std::vector<std::int64_t>& scale_shape,
    std::size_t block_rows = 128,
    std::size_t block_cols = 128);

// Convenience: decode a tensor whose descriptor declares F8_E4M3 + companion
// scale_inv. Reads both payload spans and dispatches to the block-scale decoder.
// If the tensor is BF16, dispatches to qwen38_decode_bf16.
[[nodiscard]] Result<std::vector<float>> qwen38_decode_tensor_payload(
    std::span<const std::byte> payload,
    std::span<const std::byte> scale_payload,
    const std::string& dtype,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>& scale_shape);

// RMSNorm reference: y = x / sqrt(mean(x^2) + eps) * weight
[[nodiscard]] Result<std::vector<float>> qwen38_rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon);

// Qwen4ExpTextRMSNorm: grouped last-dim RMS, then y *= (1 + weight).
// group_size == 0 means a single group over the whole vector.
[[nodiscard]] Result<std::vector<float>> qwen38_rms_norm_affine(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon,
    std::size_t group_size = 0);

struct Qwen38GatedResidualView {
    std::span<const float> hc_norm;
    std::span<const float> mix_down;
    std::span<const float> mix_up;
    std::span<const float> block_inject;
};

struct Qwen38GatedResidualResult {
    std::vector<float> mixed_input;
    std::vector<float> residual;
    std::vector<float> injection_weights;
};

[[nodiscard]] Result<Qwen38GatedResidualResult> qwen38_gated_residual(
    std::span<const float> hyper_input,
    const Qwen38GatedResidualView& weights,
    std::uint32_t hc_count,
    std::uint32_t hidden_size,
    std::uint32_t hc_lowrank,
    float epsilon);

[[nodiscard]] Result<std::vector<float>> qwen38_gated_residual_combine(
    std::span<const float> residual,
    std::span<const float> sublayer_out,
    std::span<const float> injection_weights,
    std::uint32_t hc_count,
    std::uint32_t hidden_size);

struct Qwen38DeltaNetView {
    std::span<const float> in_proj_qkv;
    std::span<const float> in_proj_z;
    std::span<const float> in_proj_a;
    std::span<const float> in_proj_b;
    std::span<const float> conv1d;
    std::span<const float> a_log;
    std::span<const float> dt_bias;
    std::span<const float> norm;
    std::span<const float> out_proj;
    std::uint32_t hidden_size{0};
    std::uint32_t num_k_heads{0};
    std::uint32_t num_v_heads{0};
    std::uint32_t key_head_dim{0};
    std::uint32_t value_head_dim{0};
    std::uint32_t conv_kernel{0};
    float rms_eps{1.0e-6F};
};

struct Qwen38DeltaNetState {
    std::vector<float> conv;
    std::vector<float> ssm;
};

[[nodiscard]] Result<std::vector<float>> qwen38_delta_net_decode(
    std::span<const float> hidden,
    const Qwen38DeltaNetView& weights,
    Qwen38DeltaNetState& state);

struct Qwen38QsaSelectView {
    std::span<const float> query_heads;
    std::span<const float> raw_keys;
    std::span<const float> k_norm;
    std::uint32_t n_query_heads{0};
    std::uint32_t head_dim{0};
    std::uint32_t token_count{0};
    std::uint32_t budget{0};
    std::uint32_t compress_ratio{0};
    float rms_eps{1.0e-6F};
};

struct Qwen38QsaSelectResult {
    std::vector<std::int32_t> selected;
    std::vector<float> block_scores;
};

[[nodiscard]] Result<Qwen38QsaSelectResult> qwen38_qsa_select(const Qwen38QsaSelectView& view);

[[nodiscard]] std::uint64_t qwen38_splitmix64(std::uint64_t value) noexcept;
[[nodiscard]] Result<std::uint32_t> qwen38_nth_prime_after(std::uint32_t start, std::uint32_t count);

[[nodiscard]] Result<std::vector<std::int64_t>> qwen38_ngram_multipliers(
    std::uint32_t unigram_vocab,
    std::uint32_t ngram_size,
    std::uint32_t ple_layer_index,
    std::int32_t seed);

[[nodiscard]] Result<std::vector<std::int64_t>> qwen38_ngram_hash_ids(
    std::span<const std::int32_t> token_ids,
    std::span<const std::int32_t> previous_context,
    std::span<const std::int64_t> multipliers,
    std::span<const std::int64_t> head_vocab_sizes,
    std::span<const std::int64_t> head_offsets,
    std::uint32_t ngram_size,
    std::uint32_t heads_per_ngram,
    std::int32_t eos_token_id);

[[nodiscard]] Result<std::vector<float>> qwen38_ngram_lookup(
    std::span<const std::int64_t> ngram_ids,
    std::span<const float> table,
    std::uint32_t table_rows,
    std::uint32_t head_dim);

[[nodiscard]] Result<std::vector<float>> qwen38_linear(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t out_dim,
    std::size_t in_dim);

[[nodiscard]] Result<std::vector<float>> qwen38_shared_mlp(
    std::span<const float> hidden,
    std::span<const float> gate_proj,
    std::span<const float> up_proj,
    std::span<const float> down_proj,
    std::span<const float> shared_gate,
    std::uint32_t hidden_size,
    std::uint32_t intermediate);

[[nodiscard]] std::vector<float> qwen38_expand_stream(
    std::span<const float> hidden,
    std::uint32_t hc_count);

struct Qwen38QsaLayerView {
    std::span<const float> index_qk;
    std::span<const float> q_layernorm;
    std::span<const float> k_layernorm;
    std::span<const float> q_proj;
    std::span<const float> k_proj;
    std::span<const float> v_proj;
    std::span<const float> o_proj;
    std::span<const float> q_norm;
    std::span<const float> k_norm;
    std::uint32_t hidden_size{0};
    std::uint32_t indexer_heads{0};
    std::uint32_t indexer_dim{0};
    std::uint32_t budget{0};
    std::uint32_t compress_ratio{0};
    std::uint32_t attn_heads{0};
    std::uint32_t attn_kv_heads{0};
    std::uint32_t attn_head_dim{0};
    float rms_eps{1.0e-6F};
};

struct Qwen38QsaCache {
    std::vector<float> index_keys;
    std::vector<float> attn_k;
    std::vector<float> attn_v;
};

[[nodiscard]] Result<std::vector<float>> qwen38_qsa_layer_token(
    std::span<const float> hidden,
    const Qwen38QsaLayerView& weights,
    Qwen38QsaCache& cache);

[[nodiscard]] Result<std::vector<float>> qwen38_qsa_sparse_attention(
    std::span<const float> query,
    std::span<const float> k_cache,
    std::span<const float> v_cache,
    std::span<const std::int32_t> selected,
    std::span<const float> gate,
    std::uint32_t attn_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim);

struct Qwen38OwnedResidual {
    std::vector<float> hc_norm;
    std::vector<float> mix_down;
    std::vector<float> mix_up;
    std::vector<float> block_inject;
};

struct Qwen38OwnedDeltaNet {
    std::vector<float> in_proj_qkv;
    std::vector<float> in_proj_z;
    std::vector<float> in_proj_a;
    std::vector<float> in_proj_b;
    std::vector<float> conv1d;
    std::vector<float> a_log;
    std::vector<float> dt_bias;
    std::vector<float> norm;
    std::vector<float> out_proj;
};

struct Qwen38OwnedQsa {
    std::vector<float> index_qk;
    std::vector<float> q_layernorm;
    std::vector<float> k_layernorm;
    std::vector<float> q_proj;
    std::vector<float> k_proj;
    std::vector<float> v_proj;
    std::vector<float> o_proj;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
};

struct Qwen38OwnedSharedMlp {
    std::vector<float> gate_proj;
    std::vector<float> up_proj;
    std::vector<float> down_proj;
    std::vector<float> shared_gate;
};

struct Qwen38OwnedExpert {
    std::vector<float> gate_proj;
    std::vector<float> up_proj;
    std::vector<float> down_proj;
};

struct Qwen38OwnedMoe {
    std::vector<float> router;
    std::vector<Qwen38OwnedExpert> experts;
};

struct Qwen38MoeRoute {
    std::vector<std::uint32_t> indices;
    std::vector<float> weights;
    std::vector<float> logits;
};

struct Qwen38ExpertCacheTelemetry {
    std::size_t requests{0};
    std::size_t hits{0};
    std::size_t misses{0};
    std::size_t evictions{0};
    std::uint64_t logical_bytes{0};
};

struct Qwen38ExpertLru {
    std::size_t capacity{0};
    std::vector<std::uint32_t> keys;
    std::vector<Qwen38OwnedExpert> values;
    Qwen38ExpertCacheTelemetry telemetry;
};

[[nodiscard]] Result<Qwen38MoeRoute> qwen38_moe_route(
    std::span<const float> hidden,
    std::span<const float> router,
    std::uint32_t num_experts,
    std::uint32_t top_k,
    bool norm_topk_prob = true);

[[nodiscard]] Result<std::vector<float>> qwen38_expert_mlp(
    std::span<const float> hidden,
    std::span<const float> gate_proj,
    std::span<const float> up_proj,
    std::span<const float> down_proj,
    std::uint32_t hidden_size,
    std::uint32_t intermediate);

[[nodiscard]] Result<const Qwen38OwnedExpert*> qwen38_expert_cache_acquire(
    Qwen38ExpertLru& cache,
    std::uint32_t key,
    const Qwen38OwnedExpert& source);

[[nodiscard]] Result<std::vector<float>> qwen38_moe_block(
    std::span<const float> hidden,
    const Qwen38OwnedMoe& moe,
    const Qwen38OwnedSharedMlp& shared,
    Qwen38ExpertLru& cache,
    std::uint32_t layer_id,
    std::uint32_t hidden_size,
    std::uint32_t intermediate,
    std::uint32_t top_k,
    bool norm_topk_prob);

struct Qwen38ReducedDims {
    std::uint32_t hidden_size{4};
    std::uint32_t hc_count{2};
    std::uint32_t hc_lowrank{2};
    std::uint32_t vocab_size{4};
    std::uint32_t dn_k_heads{1};
    std::uint32_t dn_v_heads{2};
    std::uint32_t dn_key_dim{2};
    std::uint32_t dn_value_dim{2};
    std::uint32_t dn_kernel{2};
    std::uint32_t indexer_heads{2};
    std::uint32_t indexer_dim{2};
    std::uint32_t budget{2};
    std::uint32_t compress_ratio{2};
    std::uint32_t attn_heads{2};
    std::uint32_t attn_kv_heads{1};
    std::uint32_t attn_head_dim{2};
    std::uint32_t moe_intermediate{4};
    std::uint32_t num_experts{4};
    std::uint32_t experts_per_token{2};
    std::uint32_t expert_cache_slots{2};
    bool norm_topk_prob{true};
    float rms_eps{1.0e-6F};
};

struct Qwen38ReducedModel {
    Qwen38ReducedDims dims;
    std::vector<float> embed;
    std::vector<float> lm_head;
    Qwen38OwnedResidual mixer;
    Qwen38OwnedResidual layer0_attn_hc;
    Qwen38OwnedDeltaNet layer0_delta;
    Qwen38OwnedResidual layer0_mlp_hc;
    Qwen38OwnedSharedMlp layer0_mlp;
    Qwen38OwnedMoe layer0_moe;
    Qwen38OwnedResidual layer1_attn_hc;
    Qwen38OwnedQsa layer1_qsa;
    Qwen38OwnedResidual layer1_mlp_hc;
    Qwen38OwnedSharedMlp layer1_mlp;
    Qwen38OwnedMoe layer1_moe;
    Qwen38OwnedResidual mtp_attn_hc;
    Qwen38OwnedQsa mtp_qsa;
    Qwen38OwnedResidual mtp_mlp_hc;
    Qwen38OwnedSharedMlp mtp_mlp;
    Qwen38OwnedMoe mtp_moe;
    Qwen38OwnedResidual mtp_mixer;
    std::vector<float> mtp_fc_hidden;
    std::vector<float> mtp_fc_embed;
    std::vector<float> mtp_pre_hidden;
    std::vector<float> mtp_pre_embed;
};

struct Qwen38ReducedState {
    Qwen38DeltaNetState delta;
    Qwen38QsaCache layer1;
    Qwen38QsaCache mtp;
    Qwen38ExpertLru experts;
};

struct Qwen38ReducedOutput {
    std::vector<float> hidden;
    std::vector<float> logits;
    std::vector<float> mtp_logits;
    Qwen38ExpertCacheTelemetry expert_cache;
};

[[nodiscard]] Result<Qwen38ReducedState> qwen38_reduced_zero_state(const Qwen38ReducedDims& dims);

[[nodiscard]] Result<Qwen38ReducedOutput> qwen38_reduced_prefill(
    const Qwen38ReducedModel& model,
    std::span<const std::int32_t> token_ids,
    Qwen38ReducedState& state);

[[nodiscard]] Result<Qwen38ReducedOutput> qwen38_reduced_decode(
    const Qwen38ReducedModel& model,
    std::int32_t token_id,
    Qwen38ReducedState& state);

inline constexpr std::size_t kQwen38ReducedMaxTokens = 4'096U;

enum class Qwen38ReducedFinishReason {
    stop_token,
    context_limit,
    max_tokens,
    cancelled,
    timeout,
};

struct Qwen38ReducedGenerationOptions {
    std::size_t max_new_tokens{64U};
    std::size_t context_limit_tokens{kQwen38ReducedMaxTokens};
    float temperature{0.0F};
    std::size_t top_k{40U};
    std::uint64_t seed{1U};
    std::chrono::milliseconds timeout{std::chrono::minutes(5)};
    std::vector<std::int32_t> stop_token_ids;
    std::function<bool()> cancelled;
    std::function<void(std::int32_t)> on_token;
};

struct Qwen38ReducedGenerationStats {
    std::size_t prompt_tokens{0U};
    std::size_t generated_tokens{0U};
    std::size_t requested_max_tokens{0U};
    std::size_t effective_max_tokens{0U};
    std::size_t forward_passes{0U};
    std::size_t expert_requests{0U};
    std::size_t expert_cache_hits{0U};
    std::size_t expert_cache_misses{0U};
    std::uint64_t logical_expert_bytes_read{0U};
    double prompt_eval_ms{0.0};
    double generation_ms{0.0};
    double tokens_per_second{0.0};
};

struct Qwen38ReducedGenerationResult {
    std::vector<std::int32_t> prompt_ids;
    std::vector<std::int32_t> generated_ids;
    Qwen38ReducedFinishReason finish_reason{Qwen38ReducedFinishReason::max_tokens};
    std::optional<std::int32_t> stop_token_id;
    Qwen38ReducedGenerationStats stats;
};

[[nodiscard]] Result<Qwen38ReducedGenerationResult> qwen38_reduced_generate(
    const Qwen38ReducedModel& model,
    std::span<const std::int32_t> prompt_ids,
    const Qwen38ReducedGenerationOptions& options);

// --- Q11c: real layer-0 forward over the imported FP8 checkpoint ---

struct Qwen38Layer0Weights {
    Qwen38OwnedResidual attn_hyper;
    Qwen38OwnedDeltaNet delta_net;
    Qwen38OwnedResidual mlp_hyper;
    Qwen38OwnedSharedMlp shared_mlp;
    std::vector<float> router;
    std::vector<float> embed_row;        // embedding row for the input token
    std::vector<float> token_embed;      // embed_tokens.weight rows read (row-major, 1 row)
};

struct Qwen38Layer0Telemetry {
    std::uint64_t payload_bytes_read{0U};
    std::uint64_t tensors_read{0U};
    std::uint64_t expert_payload_bytes{0U};
    std::size_t selected_experts{0U};
    std::chrono::steady_clock::duration load_ms{};
    std::chrono::steady_clock::duration forward_ms{};
};

struct Qwen38Layer0ForwardResult {
    // Post-MLP-block layer output stream (hc_count * hidden_size), available
    // only after routed experts are added by the caller via combine_layer0.
    std::vector<float> output;
    // Intermediate pieces of the layer-0 forward:
    std::vector<float> mixed_input;      // MoE input (hidden_size)
    std::vector<float> residual;         // mlp hyper residual (hc_count * hidden_size)
    std::vector<float> injection_weights;// hc_count gate weights
    std::vector<float> shared_out;       // shared expert contribution (hidden_size)
    Qwen38MoeRoute route;                // router top-k result
    Qwen38Layer0Telemetry telemetry;
};

// Folds the routed-expert contributions into the layer-0 result and applies
// the final gated-residual combine. expert_sum must already contain
// sum_k route_weight_k * expert_k(mixed_input).
[[nodiscard]] Result<std::vector<float>> qwen38_layer0_finish(
    const Qwen38Layer0ForwardResult& partial,
    std::span<const float> expert_sum);

// Loads the 22 canonical non-routed layer-0 tensors plus the embedding row for
// `token_id` from the imported checkpoint (safetensors shards). Every tensor
// payload is SHA-256 hashed at read time; hashes are recorded in telemetry.
[[nodiscard]] Result<Qwen38Layer0Weights> qwen38_load_layer0_weights(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t token_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Loads one routed expert (gate/up/down FP8 block-scale) for layer 0 from the
// imported checkpoint. Returns the decoded F32 weights.
[[nodiscard]] Result<Qwen38OwnedExpert> qwen38_load_layer0_expert(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expert_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// One-token forward through real layer-0 weights: embed row -> attn gated
// residual -> Gated DeltaNet -> combine -> mlp gated residual -> router +
// shared expert + top-k routed experts (FP8 block-scale decoded) -> combine.
// DeltaNet state starts zeroed; the layer output is the post-MLP hidden state.
[[nodiscard]] Result<Qwen38Layer0ForwardResult> qwen38_layer0_forward(
    const Qwen38Layer0Weights& weights,
    std::uint32_t token_id,
    std::uint32_t num_experts,
    std::uint32_t top_k);

[[nodiscard]] std::string_view to_string(Qwen38ReducedFinishReason reason) noexcept;

}  // namespace vastloom::inference
