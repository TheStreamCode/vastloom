#pragma once

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/model/gguf.hpp"

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

// A dense weight that can live either as F32 values or as contiguous Q8_0
// blocks ([out, in] row major). linear_w consumes both forms with the same
// per-row sequential double accumulation, so a Q8_0 weight is bit-identical
// to its dequantized F32 form while costing a quarter of the RAM.
struct Qwen38WeightRef {
    std::span<const float> f32;
    std::span<const std::byte> q8;

    Qwen38WeightRef() = default;
    Qwen38WeightRef(const Qwen38WeightRef&) = default;
    Qwen38WeightRef& operator=(const Qwen38WeightRef&) = default;
    // Single user-defined conversion from any float container so existing
    // span-style construction keeps compiling.
    template <class Vector>
        requires (!std::same_as<std::remove_cvref_t<Vector>, Qwen38WeightRef>)
    Qwen38WeightRef(const Vector& values) : f32(values) {}
    Qwen38WeightRef(std::span<const float> f, std::span<const std::byte> q)
        : f32(f), q8(q) {}
    bool is_q8() const noexcept { return !q8.empty(); }
    // Logical element count (Q8_0 blocks carry 32 values in 34 bytes).
    [[nodiscard]] std::size_t size() const noexcept {
        return is_q8() ? q8.size() / 34U * 32U : f32.size();
    }
    [[nodiscard]] bool empty() const noexcept { return size() == 0U; }
};

struct Qwen38Weight {
    std::vector<float> f32;
    std::vector<std::byte> q8;
    [[nodiscard]] Qwen38WeightRef ref() const noexcept { return {f32, q8}; }
    // Logical element count: valid for either representation.
    [[nodiscard]] std::size_t size() const noexcept {
        return q8.empty() ? f32.size() : q8.size() / 34U * 32U;
    }
    [[nodiscard]] bool empty() const noexcept { return size() == 0U; }
};
struct Qwen38GatedResidualView {
    std::span<const float> hc_norm;
    Qwen38WeightRef mix_down;
    Qwen38WeightRef mix_up;
    Qwen38WeightRef block_inject;
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
    Qwen38WeightRef in_proj_qkv;
    Qwen38WeightRef in_proj_z;
    Qwen38WeightRef in_proj_a;
    Qwen38WeightRef in_proj_b;
    std::span<const float> conv1d;
    std::span<const float> a_log;
    std::span<const float> dt_bias;
    std::span<const float> norm;
    Qwen38WeightRef out_proj;
    std::uint32_t hidden_size{0};
    std::uint32_t num_k_heads{0};
    std::uint32_t num_v_heads{0};
    std::uint32_t key_head_dim{0};
    std::uint32_t value_head_dim{0};
    std::uint32_t conv_kernel{0};
    float rms_eps{1.0e-6F};
    // false (default): HF repeat_interleave, v-head vh reads k-head vh/group.
    // true: ggml-repeat tile semantics, v-head vh reads k-head vh % k_heads —
    // the mapping the converted GGUF checkpoint is validated against.
    bool tile_k_heads{false};
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
    // Partial RoPE (used by the official qwen4exp indexer): rotated dim is the
    // first `rope_dim` elements of each head with rotate_half pairing;
    // query_position is the absolute position of the current token, block b
    // starts at token b * compress_ratio. rope_theta 0 disables RoPE.
    float rope_theta{0.0F};
    std::uint32_t rope_dim{0U};
    std::uint32_t query_position{0U};
};

// Partial rotate-half RoPE over the first `rope_dim` elements of a head
// vector: freqs = base^(-2i/rope_dim), pairs (i, i + rope_dim/2), the
// tail beyond rope_dim is left untouched.
void qwen38_apply_partial_rope(
    std::span<float> head,
    std::uint64_t position,
    std::uint64_t rope_dim,
    float theta);

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

// Owned (possibly Q8_0) variant used by the layer forward.
struct Qwen38OwnedSharedMlp;
[[nodiscard]] Result<std::vector<float>> qwen38_shared_mlp(
    std::span<const float> hidden,
    const Qwen38OwnedSharedMlp& weights,
    std::uint32_t hidden_size,
    std::uint32_t intermediate);

[[nodiscard]] std::vector<float> qwen38_expand_stream(
    std::span<const float> hidden,
    std::uint32_t hc_count);

struct Qwen38QsaLayerView {
    std::span<const float> index_qk;
    std::span<const float> q_layernorm;
    std::span<const float> k_layernorm;
    Qwen38WeightRef q_proj;
    Qwen38WeightRef k_proj;
    Qwen38WeightRef v_proj;
    Qwen38WeightRef o_proj;
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
    // Partial RoPE over attention q/k and the indexer (theta 0 disables).
    float rope_theta{0.0F};
    std::uint32_t rope_dim{0U};
    std::uint32_t position{0U};
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
    Qwen38Weight mix_down;
    Qwen38Weight mix_up;
    Qwen38Weight block_inject;
};

struct Qwen38OwnedDeltaNet {
    Qwen38Weight in_proj_qkv;
    Qwen38Weight in_proj_z;
    Qwen38Weight in_proj_a;
    Qwen38Weight in_proj_b;
    std::vector<float> conv1d;
    std::vector<float> a_log;
    std::vector<float> dt_bias;
    std::vector<float> norm;
    Qwen38Weight out_proj;
};

struct Qwen38OwnedQsa {
    std::vector<float> index_qk;
    std::vector<float> q_layernorm;
    std::vector<float> k_layernorm;
    Qwen38Weight q_proj;
    Qwen38Weight k_proj;
    Qwen38Weight v_proj;
    Qwen38Weight o_proj;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
};

struct Qwen38OwnedSharedMlp {
    Qwen38Weight gate_proj;
    Qwen38Weight up_proj;
    Qwen38Weight down_proj;
    std::vector<float> shared_gate;
};

struct Qwen38OwnedExpert {
    std::vector<float> gate_proj;
    std::vector<float> up_proj;
    std::vector<float> down_proj;
};

// Routed expert held as raw contiguous Q8_0 blocks ([out, in] row major).
// Consumed by qwen38_expert_mlp_q8; ~4x smaller than the F32 form.
struct Qwen38OwnedExpertQ8 {
    std::vector<std::byte> gate;
    std::vector<std::byte> up;
    std::vector<std::byte> down;
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

// Bit-identical to qwen38_expert_mlp for the same logical weights, but the
// matvec reads Q8_0 bytes directly (no F32 materialization).
[[nodiscard]] Result<std::vector<float>> qwen38_expert_mlp_q8(
    std::span<const float> hidden,
    const Qwen38OwnedExpertQ8& expert,
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

struct Qwen38GgufCheckpoint {
    std::filesystem::path root;
    model::GgufSplitCatalog catalog;
};

// --- Q11c/Q11d/Q11e: real layer forward over the imported FP8 checkpoint ---

// Always-active weights of one language-model layer plus the shard map needed
// to load routed experts on demand. The layer is either a Gated DeltaNet layer
// (is_qsa == false) or a QSA full-attention layer (is_qsa == true). The token
// embedding row is NOT part of the trunk: it is loaded per token with a
// bounded read.
struct Qwen38LayerTrunk {
    std::uint32_t layer{0U};
    bool is_qsa{false};
    Qwen38OwnedResidual attn_hyper;
    Qwen38OwnedDeltaNet delta_net;   // valid when !is_qsa
    Qwen38OwnedQsa qsa;              // valid when is_qsa
    Qwen38OwnedResidual mlp_hyper;
    Qwen38OwnedSharedMlp shared_mlp;
    std::vector<float> router;
    // expert id -> {gate shard, up shard, down shard} file names (safetensors path).
    std::unordered_map<std::uint32_t, std::array<std::string, 3U>> expert_shards;
    // GGUF path: experts live in fused [in, out, 512] tensors, one expert slice
    // at a fixed byte stride.
    bool is_gguf{false};
    std::filesystem::path gguf_root;
    struct Qwen38GgufExpertSource {
        std::filesystem::path shard;         // shard file holding the fused tensor
        std::uint64_t data_offset{0U};       // fused tensor start within the shard
        std::uint64_t expert_stride_bytes{0U};
    };
    Qwen38GgufExpertSource gguf_gate;
    Qwen38GgufExpertSource gguf_up;
    Qwen38GgufExpertSource gguf_down;
    // PLE n-gram block (layer 1 only). Weights come from the GGUF tensors;
    // multipliers/offsets/vocab-sizes from the GGUF metadata.
    bool has_ple{false};
    struct Qwen38PleWeights {
        std::vector<float> key_proj;      // [10240, 160] row-major
        std::vector<float> value_proj;    // [2560, 160] row-major
        std::vector<float> norm_key;      // [10240]
        std::vector<float> norm_query;    // [10240]
        std::vector<float> norm_conv;     // [10240]
        std::vector<float> conv1d;        // [10240 * 4] (k contiguous per channel)
        std::uint64_t multipliers[3]{};
        std::uint64_t head_offsets[16]{};
        std::uint64_t head_vocab_sizes[16]{};
    };
    Qwen38PleWeights ple;
};

// PLE runtime state: 2-token n-gram context (init EOS) and the dilated
// depthwise short-conv state (10240 channels x 9 history values).
struct Qwen38PleRuntime {
    std::uint32_t previous_context[2]{0U, 0U};
    bool context_initialized{false};
    std::vector<float> conv_state;  // 10240 * 9
};

// Caller-owned per-layer runtime state. The DeltaNet branch carries conv
// history and SSM across tokens; the QSA branch carries the indexer-key and
// KV caches; the PLE branch carries the n-gram context and short-conv state.
struct Qwen38LayerRuntimeState {
    Qwen38DeltaNetState delta_net;
    Qwen38QsaCache qsa;
    Qwen38PleRuntime ple;
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
    // Intermediate pieces of the layer forward:
    std::vector<float> mixed_input;      // MoE input (hidden_size)
    std::vector<float> residual;         // mlp hyper residual (hc_count * hidden_size)
    std::vector<float> injection_weights;// hc_count gate weights
    std::vector<float> shared_out;       // shared expert contribution (hidden_size)
    Qwen38MoeRoute route;                // router top-k result
    Qwen38Layer0Telemetry telemetry;
};

// Loads the 22 canonical non-routed tensors of `layer` (Gated DeltaNet for
// layers where layer % full_attention_interval != 3, QSA otherwise) plus the
// expert->shard map for that layer.
[[nodiscard]] Result<Qwen38LayerTrunk> qwen38_load_layer_trunk(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t layer,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Bounded read of exactly one embedding row (5 KiB) for `token_id`.
[[nodiscard]] Result<std::vector<float>> qwen38_load_embed_row(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t token_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Loads one routed expert (gate/up/down FP8 block-scale) for the trunk's layer
// from the imported checkpoint using the trunk's shard map. Returns F32.
[[nodiscard]] Result<Qwen38OwnedExpert> qwen38_load_layer_expert(
    const Qwen38LayerTrunk& trunk,
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expert_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Q8_0-native variant: only the GGUF layout provides Q8_0 routed experts; the
// safetensors/FP8 path has no equivalent raw block and errors out.
[[nodiscard]] Result<Qwen38OwnedExpertQ8> qwen38_load_layer_expert_q8(
    const Qwen38LayerTrunk& trunk,
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expert_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// One-token forward through real layer weights: input stream (hc_count *
// hidden_size; the expanded embedding row for the first layer of a chain, the
// previous layer's output otherwise) -> attn gated residual -> Gated DeltaNet
// or QSA -> combine -> mlp gated residual -> router + shared expert. The
// runtime state is caller-owned: pass a fresh state for the first token and
// reuse it for subsequent tokens (multi-token prefill). token_id drives the
// PLE n-gram lookup when the layer carries one (layer 1). Routed experts are
// resolved by the caller (qwen38_load_layer_expert + qwen38_expert_mlp) and
// folded back with qwen38_layer0_finish.
[[nodiscard]] Result<Qwen38Layer0ForwardResult> qwen38_layer_forward(
    const Qwen38LayerTrunk& trunk,
    std::span<const float> input_stream,
    Qwen38LayerRuntimeState& state,
    std::uint32_t token_id,
    std::uint32_t num_experts,
    std::uint32_t top_k);

// PLE n-gram forward (layer 1): hashed n-gram lookup over the per-layer
// embedding table, key/value projections, sigmoid gate against the current
// stream, norm-conv and dilated short-conv. Returns the [10240] delta to add
// to the input stream.
[[nodiscard]] Result<std::vector<float>> qwen38_ple_forward(
    const Qwen38LayerTrunk& trunk,
    const Qwen38GgufCheckpoint& checkpoint,
    Qwen38PleRuntime& state,
    std::span<const float> input_stream,
    std::uint32_t token_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Folds the routed-expert contributions into the layer result and applies
// the final gated-residual combine. expert_sum must already contain
// sum_k route_weight_k * expert_k(mixed_input).
[[nodiscard]] Result<std::vector<float>> qwen38_layer0_finish(
    const Qwen38Layer0ForwardResult& partial,
    std::span<const float> expert_sum);

// --- Q11g: GGUF (qwen4exp) checkpoint source ---

enum class Qwen38CheckpointKind {
    safetensors,
    gguf,
};

// Detects the checkpoint layout: model.safetensors.index.json (FP8 release)
// or one or more *.gguf shards (llama.cpp Q8_0 and friends).
[[nodiscard]] Result<Qwen38CheckpointKind> qwen38_detect_checkpoint_kind(
    const std::filesystem::path& checkpoint_root);

// IEEE half -> F32 (GGUF Q8_0 block scales are F16).
[[nodiscard]] float qwen38_f16_to_f32(std::uint16_t bits) noexcept;

// Decodes Q8_0 payload (F16 scale + 32 int8 per block) into F32.
[[nodiscard]] Result<std::vector<float>> qwen38_decode_q8_0(
    std::span<const std::byte> payload,
    std::uint64_t element_count);

// Opens every *.gguf shard under the checkpoint root (depth 2), finalizes the
// split catalog and validates general.architecture == "qwen4exp".
[[nodiscard]] Result<Qwen38GgufCheckpoint> qwen38_open_gguf_checkpoint(
    const std::filesystem::path& checkpoint_root);

// Enables/disables the resident raw Q8_0 payload cache (default on). The
// greedy CLI turns it off when trunk residency is active, so the two caches
// never compete for the same RAM.
void qwen38_set_raw_cache_enabled(bool enabled);

// Cache diagnostics: [fetches, hits].
std::pair<std::uint64_t, std::uint64_t> qwen38_raw_cache_stats();

// Reads one GGUF tensor by exact name, decodes it (Q8_0/BF16/F32) and, when
// transpose is set, converts the ggml [in, out] layout to row-major [out, in].
// expected_dims is checked against the GGUF shape verbatim.
[[nodiscard]] Result<std::vector<float>> qwen38_read_gguf_tensor_f32(
    const Qwen38GgufCheckpoint& checkpoint,
    std::string_view tensor_name,
    const std::vector<std::uint64_t>& expected_dims,
    bool transpose,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// GGUF variants of the canonical safetensors loaders. Same semantics, same
// output structures; tensors arrive transposed (ggml [in, out]) and Q8_0
// quantized, and routed experts are read as fixed-stride slices of the fused
// [in, out, 512] tensors.
[[nodiscard]] Result<Qwen38LayerTrunk> qwen38_load_gguf_layer_trunk(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t layer,
    Qwen38Layer0Telemetry* telemetry = nullptr);

[[nodiscard]] Result<std::vector<float>> qwen38_load_gguf_embed_row(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t token_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

[[nodiscard]] Result<Qwen38OwnedExpert> qwen38_load_gguf_expert(
    const Qwen38LayerTrunk& trunk,
    const Qwen38GgufCheckpoint& checkpoint,
    std::uint32_t expert_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Same slice as qwen38_load_gguf_expert but returns the raw Q8_0 blocks.
[[nodiscard]] Result<Qwen38OwnedExpertQ8> qwen38_load_gguf_expert_q8(
    const Qwen38LayerTrunk& trunk,
    const Qwen38GgufCheckpoint& checkpoint,
    std::uint32_t expert_id,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Final collapse + LM head over the GGUF checkpoint. The official text model
// ends with the global hyper-connection mixer (Qwen4ExpTextGatedResidual with
// use_combine=false) followed directly by lm_head: there is NO final RMSNorm.
struct Qwen38GgufHead {
    Qwen38OwnedResidual mixer;   // output_hc_norm/down/up (block_inject empty)
    std::vector<float> lm_head;  // [248320, 2560] row-major
};

[[nodiscard]] Result<Qwen38GgufHead> qwen38_load_gguf_head(
    const std::filesystem::path& checkpoint_root,
    Qwen38Layer0Telemetry* telemetry = nullptr);

// Collapses the final 4-branch stream through the global mixer and projects
// the 2560 hidden state onto the vocabulary (248320 logits).
[[nodiscard]] Result<std::vector<float>> qwen38_gguf_logits(
    const Qwen38GgufHead& head,
    std::span<const float> final_stream);

// Detokenizer: loads tokenizer.ggml.tokens from the GGUF header (gpt2-style
// byte-level pieces) and maps them back to raw bytes.
[[nodiscard]] Result<std::vector<std::string>> qwen38_load_gguf_vocab(
    const std::filesystem::path& checkpoint_root);

// Byte-level gpt2 piece -> raw bytes (Ġ -> ' ', Ċ -> '\n', ...).
[[nodiscard]] std::string qwen38_gpt2_piece_to_bytes(std::string_view piece);

[[nodiscard]] std::string_view to_string(Qwen38ReducedFinishReason reason) noexcept;

}  // namespace vastloom::inference
