#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::inference {

// Qwen3.8-Flash-Next (Q4 architecture preview)
//
// Reference: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
// Budget: docs/planning/QWEN38_BUDGET_ESTIMATE.md
//
// This header is a Phase 0 skeleton. It declares the data types the inspector
// needs to parse a Qwen3.8-Flash-Next checkpoint metadata. It does NOT yet
// open any GGUF/Safetensors payload — the test that exercises this header
// uses mock metadata only. Real-payload support lands in Q3.

struct Qwen38FlashNextConfig {
    // Core layout (from text_config)
    std::uint32_t vocab_size{0};
    std::uint32_t hidden_size{0};
    std::uint32_t num_hidden_layers{0};

    // Full attention (QSA layers — every 4th)
    std::uint32_t num_attention_heads{0};
    std::uint32_t num_key_value_heads{0};
    std::uint32_t head_dim{0};
    std::uint32_t rope_dimension{0};

    // Linear attention (Gated DeltaNet — 3 per block)
    std::uint32_t linear_num_value_heads{0};
    std::uint32_t linear_num_key_heads{0};
    std::uint32_t linear_value_head_dim{0};
    std::uint32_t linear_key_head_dim{0};
    std::uint32_t linear_conv_kernel_dim{0};

    // QSA sparse attention indexer
    std::uint32_t indexer_n_heads{0};
    std::uint32_t indexer_kv_heads{0};
    std::uint32_t indexer_head_dim{0};
    std::uint32_t indexer_budget{0};
    std::uint32_t indexer_compress_ratio{0};

    // MoE
    std::uint32_t num_experts{0};
    std::uint32_t num_experts_per_tok{0};
    std::uint32_t moe_intermediate_size{0};
    std::uint32_t shared_expert_intermediate_size{0};

    // Gated Residual (4 branches)
    std::uint32_t hc_count{0};
    std::uint32_t hc_lowrank{0};

    // N-gram embedding (PLE)
    std::uint32_t ngram_size{0};
    std::uint32_t ngram_vocab_size_base{0};
    std::uint32_t ple_embed_dim{0};
    std::uint32_t ple_conv_kernel_size{0};
    std::uint32_t split_ngram_parts{0};
    // Layer indices at which PLE is inserted (1-based). From config: ple_layer_ids.
    std::vector<std::uint32_t> ple_layer_ids;

    // Block layout
    std::uint32_t full_attention_interval{0};

    // MTP
    std::uint32_t mtp_num_hidden_layers{0};
    std::uint32_t nextn_predict_layers{0};

    // Position embeddings
    std::uint32_t max_position_embeddings{0};
    float rope_theta{0.0F};
    float partial_rotary_factor{0.0F};
    float rms_norm_eps{0.0F};

    // Tokens
    std::uint32_t bos_token_id{0};
    std::uint32_t eos_token_id{0};
    std::uint32_t pad_token_id{0};

    // Data types (informational — actual per-tensor dtype comes from the GGUF/Safetensors header)
    std::string ssm_dtype;
    std::string default_dtype;

    // Vision (ignored in text-only execution path; parsed for metadata completeness)
    bool has_vision{false};
};

struct Qwen38DTypeSummary {
    std::string dtype;
    std::size_t tensor_count{0};
    ByteCount bytes{0};
};

struct Qwen38FlashNextInspection {
    std::filesystem::path root;
    std::string model_id;
    std::string revision;
    Qwen38FlashNextConfig config;
    bool metadata_only{true};
    bool is_qwen4_exp{false};
    bool has_ngram_embedding{false};
    bool has_gated_residual{false};
    bool has_qsa{false};
    bool has_deltanet{false};
    bool has_mtp{false};
    bool has_vision{false};
    std::size_t shard_count{0};
    std::size_t tensor_count{0};
    std::vector<Qwen38DTypeSummary> dtypes;
    model::LogicalTensorCatalog logical_catalog;
};

// Parses a Hugging Face config.json for Qwen3.8-Flash-Next.
// This is read-only and does not access any tensor payload.
[[nodiscard]] Result<Qwen38FlashNextConfig> load_qwen38_flash_next_config(
    const std::filesystem::path& config_path,
    bool require_text_config = true);

// Validates a parsed config against the invariants documented in
// docs/planning/QWEN38_BUDGET_ESTIMATE.md. Returns the populated Inspection
// with booleans for each primitive architecture feature.
[[nodiscard]] Result<Qwen38FlashNextInspection> inspect_qwen38_flash_next_config(
    const Qwen38FlashNextConfig& config);

// Safetensors header parsing: a single shard file begins with a 64-bit
// little-endian byte length, followed by that many bytes of UTF-8 JSON
// describing every tensor in the shard. The payload bytes follow the JSON.
struct Qwen38TensorDescriptor {
    std::string name;       // logical name, e.g. "model.language_model.layers.0.linear_attn.A_log"
    std::string dtype;      // "BF16", "F8_E4M3", "F16", ...
    std::vector<std::int64_t> shape;
    std::uint64_t data_offset_begin{0};
    std::uint64_t data_offset_end{0};
    // Optional block-scale companion (FP8 block-scale quantization).
    // When `weight` is FP8 (F8_E4M3), `scale_inv_name` carries the matching
    // `weight_scale_inv` tensor that is read alongside the weight tensor.
    std::string scale_inv_name;
    std::uint64_t scale_inv_offset_begin{0};
    std::uint64_t scale_inv_offset_end{0};
};

struct Qwen38ShardHeader {
    std::filesystem::path path;
    std::uint64_t json_header_bytes{0};   // size of the JSON header section
    std::uint64_t total_bytes{0};          // total shard bytes (file size)
    std::vector<Qwen38TensorDescriptor> tensors;
};

struct Qwen38ShardIndexEntry {
    std::string tensor_name;
    std::filesystem::path shard_path;      // relative to checkpoint root
};

struct Qwen38LayerSummary {
    std::size_t layer{0};
    std::size_t total{0};
    std::size_t linear_attn{0};      // Gated DeltaNet tensors (9 per linear layer)
    std::size_t self_attn{0};        // full attention tensors (9 per full layer)
    std::size_t attn_hyper{0};       // per-layer Gated Residual attn (4)
    std::size_t mlp_hyper{0};        // per-layer Gated Residual mlp (4)
    std::size_t mlp_gate{0};         // router gate (1)
    std::size_t mlp_shared_gate{0};  // shared expert gate (1)
    std::size_t mlp_shared_expert{0}; // shared expert weights (3)
    std::size_t mlp_experts{0};      // routed experts (3072 per layer)
    std::size_t ple{0};              // N-gram PLE tensors (0 or 138 for layer 1)
};

struct Qwen38InspectionResult {
    std::filesystem::path root;
    std::string model_id;
    Qwen38FlashNextConfig config;
    // Index-level results: full catalog of tensors across all shards.
    std::vector<Qwen38ShardIndexEntry> weight_map;
    std::size_t total_tensors{0};
    std::size_t total_shards{0};
    std::size_t fp8_block_scaled_experts{0};
    std::size_t ngram_tensor_count{0};
    bool has_gated_residual_global{false};   // hyper_connection_mixer
    bool has_gated_residual_per_layer{false}; // attn_hyper_connection, mlp_hyper_connection
    // Per-layer breakdown (48 entries for language_model.layers.0..47).
    std::vector<Qwen38LayerSummary> layers;
    // Non-layer aggregates.
    std::size_t vision_tensors{0};
    std::size_t mtp_tensors{0};
    std::size_t global_tensors{0};
    // Header-level result for a single loaded shard.
    Qwen38ShardHeader shard_header;
    bool shard_loaded{false};
};

struct Qwen38ExpectedCatalog {
    static constexpr std::size_t total_tensors = 152089;
    static constexpr std::size_t total_shards = 131;
    static constexpr std::size_t language_layers = 48;
    static constexpr std::size_t language_tensors = 148650; // 47*3094 + 3232
    static constexpr std::size_t per_layer_linear = 3094;
    static constexpr std::size_t per_layer_full = 3094;
    static constexpr std::size_t per_layer_ple_extra = 138; // added to layer 1
    static constexpr std::size_t vision_tensors = 333;
    static constexpr std::size_t mtp_tensors = 3101;
    static constexpr std::size_t global_tensors = 5; // embed + lm_head + norm + 3 hyper global
    static constexpr std::size_t ngram_tensors = 129; // 128 shard weights + weight_scale
    static constexpr std::size_t ngram_with_proj = 138; // + ple_input_proj etc
    static constexpr std::size_t fp8_expert_scale_inv = 75264; // counted as fp8_block_scaled_experts
    static constexpr std::size_t linear_layers = 36;
    static constexpr std::size_t qsa_layers = 12;
    static constexpr std::size_t mtp_layers = 1;
    static constexpr std::size_t per_layer_non_routed = 22;
    static constexpr std::size_t mtp_non_routed = 29;
    static constexpr std::size_t language_non_routed = 1194; // 48*22 + 138 PLE
    static constexpr std::size_t text_non_routed = 1228;     // 1194 + 5 global + 29 mtp
};

enum class Qwen38LayerKind {
    linear_attention,
    qwen_sparse_attention,
};

enum class Qwen38Consumer {
    delta_net,
    qsa_indexer,
    qsa_attention,
    gated_residual_attn,
    gated_residual_mlp,
    moe_router,
    moe_shared,
    ngram_ple,
    mtp_head,
    global_embed,
    global_lm_head,
    global_mixer,
};

struct Qwen38PlannedTensor {
    std::string name;
    Qwen38Consumer consumer{Qwen38Consumer::delta_net};
    std::filesystem::path shard_path;
    bool bound{false};
};

struct Qwen38LayerPlan {
    std::uint32_t layer{0};
    bool is_mtp{false};
    Qwen38LayerKind kind{Qwen38LayerKind::linear_attention};
    bool has_ple{false};
    std::vector<Qwen38PlannedTensor> tensors;
};

struct Qwen38GraphPlan {
    std::uint32_t language_layers{0};
    std::uint32_t linear_layers{0};
    std::uint32_t qsa_layers{0};
    std::uint32_t mtp_layers{0};
    std::vector<Qwen38LayerPlan> layers;
    std::vector<Qwen38PlannedTensor> globals;
    std::size_t non_routed_tensor_count{0};
};

[[nodiscard]] bool qwen38_is_qsa_layer(std::uint32_t layer, std::uint32_t full_attention_interval) noexcept;

[[nodiscard]] Result<Qwen38GraphPlan> qwen38_plan_non_routed_graph(const Qwen38FlashNextConfig& config);

[[nodiscard]] Result<Qwen38GraphPlan> qwen38_validate_non_routed_plan(const Qwen38GraphPlan& plan);

[[nodiscard]] Result<Qwen38GraphPlan> qwen38_bind_non_routed_plan(
    const Qwen38GraphPlan& plan,
    const std::vector<Qwen38ShardIndexEntry>& weight_map);

static constexpr ByteCount qwen38_ngram_scenario_b_bytes = 4U * ::vastloom::GiB;
static constexpr ByteCount qwen38_reference_live_ram_gate = (123ULL * ::vastloom::GiB) / 10ULL;

struct Qwen38PersistBudget {
    std::uint32_t linear_layers{0};
    ByteCount deltanet_conv_bytes{0};
    ByteCount deltanet_ssm_bytes{0};
    ByteCount deltanet_bytes{0};
    ByteCount ngram_working_set{0};
    ByteCount total_resident{0};
    bool under_reference_ram_gate{false};
};

[[nodiscard]] Result<Qwen38PersistBudget> qwen38_plan_persist_budget(
    const Qwen38FlashNextConfig& config,
    ByteCount ngram_working_set = qwen38_ngram_scenario_b_bytes);

struct Qwen38DeltaNetLayerState {
    std::vector<float> conv;
    std::vector<float> ssm;
};

struct Qwen38DeltaNetPersist {
    std::uint32_t linear_layers{0};
    std::vector<Qwen38DeltaNetLayerState> layers;
    safety::Reservation ram;
};

[[nodiscard]] Result<Qwen38DeltaNetPersist> qwen38_deltanet_persist_create(
    const Qwen38FlashNextConfig& config,
    safety::ResourceGovernor& governor);

struct Qwen38NgramPageTelemetry {
    std::size_t hits{0};
    std::size_t misses{0};
    std::size_t evictions{0};
    std::size_t prefetches{0};
    std::uint64_t logical_bytes{0};
};

struct Qwen38NgramPageCache {
    std::filesystem::path path;
    std::uint32_t rows{0};
    std::uint32_t dim{0};
    std::uint32_t rows_per_page{0};
    std::uint32_t max_pages{0};
    std::vector<std::uint32_t> lru;
    std::vector<std::vector<float>> pages;
    Qwen38NgramPageTelemetry telemetry;
    safety::Reservation ram;
};

struct Qwen38NgramPageOptions {
    std::filesystem::path table_path;
    std::uint32_t rows{0};
    std::uint32_t dim{0};
    std::uint32_t rows_per_page{0};
    std::uint32_t max_pages{0};
};

[[nodiscard]] Result<Qwen38NgramPageCache> qwen38_ngram_page_cache_open(
    const Qwen38NgramPageOptions& options,
    safety::ResourceGovernor& governor);

[[nodiscard]] Result<std::vector<float>> qwen38_ngram_page_gather(
    Qwen38NgramPageCache& cache,
    std::span<const std::int64_t> row_ids);

// Validates a full inspection result against Qwen38ExpectedCatalog.
// Returns the result on success, error with context on mismatch.
[[nodiscard]] Result<Qwen38InspectionResult> qwen38_validate_catalog(
    const Qwen38InspectionResult& result);

// Builds a per-layer summary from a weight_map without touching shard files.
[[nodiscard]] std::vector<Qwen38LayerSummary> qwen38_summarize_layers(
    const std::vector<Qwen38ShardIndexEntry>& weight_map);

// Parses model.safetensors.index.json and walks the JSON object {name -> shard_path}.
[[nodiscard]] Result<Qwen38InspectionResult> inspect_qwen38_flash_next_safetensors(
    const std::filesystem::path& index_json_path,
    const Qwen38FlashNextConfig& config,
    bool load_first_shard_header = false);

// Parses the safetensors header of a single shard file (does not read
// tensor payload bytes). Caller passes the file path and the file size
// (Content-Length from HTTP HEAD or filesystem stat).
[[nodiscard]] Result<Qwen38ShardHeader> read_qwen38_safetensors_header(
    const std::filesystem::path& shard_path,
    std::uint64_t file_size_bytes,
    std::uint64_t json_header_bytes_to_read = 256U * 1024U);

// --- Q4: payload read + three-object import transaction ---

struct Qwen38TensorPayload {
    std::string tensor_name;
    std::string dtype;
    std::vector<std::byte> bytes;           // weight bytes (and scale bytes appended if present)
    std::vector<std::byte> scale_bytes;     // weight_scale_inv companion, empty if none
    std::string sha256_hex;                 // hash of `bytes` only (weight payload)
    ByteCount payload_bytes{0U};
};

// Reads exactly one tensor's payload from a shard file using the parsed
// header. Bounds are checked against the header JSON length and file size.
// If the descriptor has a scale_inv companion, the companion payload is also
// read and returned separately. No hashing is performed here.
[[nodiscard]] Result<Qwen38TensorPayload> qwen38_read_tensor_payload(
    const std::filesystem::path& shard_path,
    const Qwen38ShardHeader& header,
    std::string_view tensor_name);

struct Qwen38TensorImportRequest {
    std::filesystem::path source_shard;     // safetensors shard file
    std::string tensor_name;
    std::string expected_sha256;            // hex of the weight payload bytes
    std::filesystem::path destination_dir;  // artifact root (must exist)
    std::string artifact_name;              // output file name (without extension)
};

struct Qwen38TensorImportResult {
    std::string tensor_name;
    std::filesystem::path artifact_path;
    ByteCount payload_bytes{0U};
    ByteCount scale_bytes{0U};
    std::string sha256_hex;
    bool scale_included{false};
};

// Three-object import transaction for a single safetensors tensor range:
//   (1) source range from the shard (tensor + optional scale_inv),
//   (2) expected SHA-256 of the weight payload,
//   (3) bounded destination write with atomic rename.
// The Governor must admit the aggregate payload (weight + scale) as a RAM
// reservation before any byte is written. Fails atomically on: missing
// shard/header entry, out-of-bounds range, hash mismatch, destination
// collision or governor refusal (ErrorCode::resource_exhausted).
[[nodiscard]] Result<Qwen38TensorImportResult> qwen38_import_tensor_transaction(
    const Qwen38TensorImportRequest& request,
    const Qwen38ShardHeader& header,
    safety::ResourceGovernor& governor);

// Returns the strict expected values for Qwen3.8-Flash-Next (committed 2026-08-26).
// Used by tests as a "this is the model we're targeting" constant, decoupled
// from any parsed JSON. If the official config drifts, this constant does not
// change — tests will then flag the drift.
struct Qwen38ExpectedInvariants {
    static constexpr std::uint32_t hidden_size = 2560;
    static constexpr std::uint32_t num_hidden_layers = 48;
    static constexpr std::uint32_t num_experts = 512;
    static constexpr std::uint32_t num_experts_per_tok = 10;
    static constexpr std::uint32_t moe_intermediate_size = 640;
    static constexpr std::uint32_t shared_expert_intermediate_size = 640;
    static constexpr std::uint32_t linear_num_value_heads = 48;
    static constexpr std::uint32_t linear_num_key_heads = 16;
    static constexpr std::uint32_t linear_value_head_dim = 128;
    static constexpr std::uint32_t linear_key_head_dim = 128;
    static constexpr std::uint32_t linear_conv_kernel_dim = 4;
    static constexpr std::uint32_t num_attention_heads = 24;
    static constexpr std::uint32_t num_key_value_heads = 2;
    static constexpr std::uint32_t head_dim = 256;
    static constexpr std::uint32_t indexer_n_heads = 4;
    static constexpr std::uint32_t indexer_kv_heads = 1;
    static constexpr std::uint32_t indexer_head_dim = 128;
    static constexpr std::uint32_t indexer_budget = 2048;
    static constexpr std::uint32_t hc_count = 4;
    static constexpr std::uint32_t hc_lowrank = 320;
    static constexpr std::uint32_t ngram_size = 3;
    static constexpr std::uint32_t ngram_vocab_size_base = 20000000;
    static constexpr std::uint32_t ple_embed_dim = 2560;
    static constexpr std::uint32_t split_ngram_parts = 128;
    static constexpr std::uint32_t full_attention_interval = 4;
    static constexpr std::uint32_t vocab_size = 248320;
    static constexpr std::uint32_t bos_token_id = 248044;
    static constexpr std::uint32_t eos_token_id = 248044;
    static constexpr const char* ssm_dtype = "float32";
    static constexpr const char* default_dtype = "bfloat16";
    static constexpr const char* model_type = "qwen4_exp";
};

}  // namespace vastloom::inference
