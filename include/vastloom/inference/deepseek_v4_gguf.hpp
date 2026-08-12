#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/model/gguf.hpp"
#include "vastloom/model/layer_streaming.hpp"
#include "vastloom/model/tensor_catalog.hpp"

namespace vastloom::inference {

struct DeepseekV4GgufCatalogOptions {
    std::string model_id{"unsloth/DeepSeek-V4-Flash-0731-GGUF"};
    std::string revision;
    std::uint32_t layers{43U};
    std::uint32_t experts_per_layer{256U};
    std::uint32_t top_k{6U};
    std::uint32_t context_tokens{4'096U};
    std::uint32_t batch_size{1U};
    ByteCount safe_ram_cap_bytes{0U};
    ByteCount safe_vram_cap_bytes{0U};
    bool metadata_only{false};
    // Require the complete architecture metadata needed to construct the native reference
    // configuration. Synthetic catalog tests may leave this disabled.
    bool require_reference_architecture{false};
    // Optional immutable full-shard paths corresponding to split order. Header-prefix paths are
    // used only when this vector is empty.
    std::vector<std::filesystem::path> source_shards;
};

struct DeepseekV4GgufArchitectureConfig {
    std::uint32_t vocabulary_size{0U};
    std::uint32_t maximum_context_tokens{0U};
    std::uint32_t width{0U};
    std::uint32_t layer_count{0U};
    std::uint32_t hash_layer_count{0U};
    std::uint32_t head_count{0U};
    std::uint32_t key_value_head_count{0U};
    std::uint32_t expert_count{0U};
    std::uint32_t experts_per_token{0U};
    std::uint32_t expert_gating_function{0U};
    std::uint32_t shared_expert_count{0U};
    std::uint32_t intermediate_width{0U};
    std::uint32_t query_lora_rank{0U};
    std::uint32_t head_width{0U};
    std::uint32_t rope_width{0U};
    std::uint32_t output_groups{0U};
    std::uint32_t output_lora_rank{0U};
    std::uint32_t window_size{0U};
    std::uint32_t original_context_tokens{0U};
    std::uint32_t index_head_count{0U};
    std::uint32_t index_head_width{0U};
    std::uint32_t index_top_k{0U};
    std::uint32_t hyper_multiplier{0U};
    std::uint32_t sinkhorn_iterations{0U};
    float compression_rope_theta{0.0F};
    float rope_theta{0.0F};
    float rope_factor{0.0F};
    float beta_fast{0.0F};
    float beta_slow{0.0F};
    float norm_epsilon{0.0F};
    float hyper_epsilon{0.0F};
    float route_scale{0.0F};
    bool expert_weights_normalized{false};
    std::string rope_scaling_type;
    std::vector<std::uint32_t> compression_ratios;
    std::vector<float> routed_expert_swiglu_limits;
    std::vector<float> shared_expert_swiglu_limits;
};

struct DeepseekV4GgufCatalogReport {
    model::LogicalTensorCatalog catalog;
    std::optional<DeepseekV4GgufArchitectureConfig> reference_architecture;
    std::uint32_t layers{0U};
    std::uint32_t experts_per_layer{0U};
    std::uint32_t top_k{0U};
    ByteCount no_cache_expert_bytes_per_token{0U};
    ByteCount minimum_expert_bundle_bytes{0U};
    ByteCount maximum_expert_bundle_bytes{0U};
    ByteCount layer_streaming_tile_limit_bytes{0U};
    ByteCount minimum_layer_trunk_bytes{0U};
    ByteCount maximum_layer_trunk_bytes{0U};
    ByteCount maximum_layer_trunk_tensor_bytes{0U};
    ByteCount maximum_layer_trunk_tile_bytes{0U};
    std::size_t layer_trunk_planned_layers{0U};
    std::size_t layer_trunk_tensor_count{0U};
    std::size_t layer_trunk_tile_count{0U};
    std::size_t reference_storage_type_covered_tensor_count{0U};
    ByteCount reference_storage_type_covered_bytes{0U};
    std::size_t reference_storage_type_unsupported_tensor_count{0U};
    ByteCount reference_storage_type_unsupported_bytes{0U};
    bool reference_storage_types_complete{false};
    std::uint32_t context_tokens{0U};
    std::uint32_t maximum_context_tokens{0U};
    std::uint32_t batch_size{0U};
    ByteCount persistent_kv_bytes{0U};
    ByteCount compression_state_bytes{0U};
    ByteCount rope_table_bytes{0U};
    ByteCount activation_scratch_bytes{0U};
    ByteCount runtime_state_bytes{0U};
    ByteCount minimum_host_resident_bytes{0U};
    ByteCount minimum_expert_tier_bytes{0U};
    ByteCount minimum_streamed_trunk_cpu_ram_bytes{0U};
    ByteCount double_buffer_streamed_trunk_cpu_ram_bytes{0U};
    ByteCount minimum_cpu_streaming_ram_bytes{0U};
    ByteCount double_buffer_cpu_streaming_ram_bytes{0U};
    ByteCount safe_ram_cap_bytes{0U};
    ByteCount safe_vram_cap_bytes{0U};
    bool admission_evaluated{false};
    bool expert_fits_ram_tier{false};
    bool expert_fits_vram_tier{false};
    bool resident_trunk_capacity_candidate{false};
    bool capacity_plan_candidate{false};
    std::size_t split_shards{0U};
    std::size_t physical_tensors{0U};
    ByteCount total_file_bytes{0U};
    ByteCount total_header_bytes{0U};
    ByteCount total_payload_bytes{0U};
    bool imported_manifest_verified{false};
    bool pinned_0731_checkpoint_spec_verified{false};
    bool reference_config_verified{false};
    bool layer0_reference_catalog_verified{false};
    bool all_layer_reference_catalogs_verified{false};
    bool checkpoint_reference_catalog_verified{false};
    std::size_t reference_catalog_validated_layers{0U};
    std::size_t reference_catalog_validated_tensors{0U};
    ByteCount reference_catalog_validated_bytes{0U};
    std::size_t reference_catalog_global_tensors{0U};
    ByteCount reference_catalog_global_bytes{0U};
    std::size_t reference_catalog_expert_bundles{0U};
    std::size_t reference_catalog_expert_tensor_slices{0U};
    std::size_t reference_catalog_physical_expert_tensors{0U};
    bool imported_file_hashes_verified{false};
    bool full_payload_present{false};
    bool executable_source_binding_verified{false};
};

struct DeepseekV4GgufShardIdentity {
    std::filesystem::path full_relative_path;
    ByteCount full_bytes{0U};
    std::string full_sha256;
    std::filesystem::path metadata_relative_path;
    ByteCount metadata_bytes{0U};
    std::string metadata_sha256;
};

struct DeepseekV4GgufCheckpointSpec {
    std::string model_id;
    std::string revision;
    std::vector<DeepseekV4GgufShardIdentity> shards;
    std::uint32_t layers{43U};
    std::uint32_t experts_per_layer{256U};
    std::uint32_t top_k{6U};
};

struct DeepseekV4GgufCheckpointInspectOptions {
    bool verify_hashes{true};
    std::uint32_t context_tokens{4'096U};
    std::uint32_t batch_size{1U};
    ByteCount safe_ram_cap_bytes{0U};
    ByteCount safe_vram_cap_bytes{0U};
};

struct DeepseekV4ProbeTrunkRange {
    std::string tensor_name;
    model::TensorRole role{model::TensorRole::other};
    ByteCount tensor_offset{0U};
    model::TensorSourceRange source;
    bool final_tile{false};
    std::optional<std::uint32_t> ggml_type;
};

struct DeepseekV4ProbeTensorSlice {
    std::string tensor_name;
    std::vector<std::uint64_t> shape;
    ByteCount packed_offset{0U};
    ByteCount bytes{0U};
    model::TensorSourceRange source;
    std::optional<std::uint32_t> ggml_type;
};

struct DeepseekV4ProbeExpert {
    std::uint32_t expert{0U};
    ByteCount bytes{0U};
    std::vector<model::TensorSourceRange> read_ranges;
    std::vector<DeepseekV4ProbeTensorSlice> tensors;
};

// Exact, non-executing range plan for one checkpoint-shaped layer/token probe. The caller must
// provide the top-k result after the trunk/router phase; this function validates the IDs and
// ranges but cannot prove routing authority. Metadata-only catalogs may be planned but never loaded.
//
// A hash-routed layer is the one case where authority can be established without executing the
// model: its selection is a pure lookup in the pinned `tid2eid` table. When such a plan is built
// from a verified table read, `routing_authority_verified` is set and `routing_token` records the
// token the selection belongs to.
struct DeepseekV4LayerProbePlan {
    std::string model_id;
    std::string revision;
    bool routing_authority_verified{false};
    std::optional<std::uint32_t> routing_token;
    bool metadata_only{true};
    bool imported_file_hashes_verified{false};
    bool full_payload_present{false};
    bool executable_source_binding_verified{false};
    std::uint32_t layer{0U};
    std::uint32_t expected_top_k{0U};
    ByteCount tile_limit_bytes{0U};
    ByteCount trunk_bytes{0U};
    ByteCount selected_expert_bytes{0U};
    ByteCount total_payload_bytes{0U};
    ByteCount maximum_read_range_bytes{0U};
    ByteCount minimum_expert_tier_bytes{0U};
    std::size_t trunk_range_count{0U};
    std::size_t selected_expert_range_count{0U};
    std::size_t total_range_count{0U};
    std::size_t unique_source_objects{0U};
    std::vector<std::uint32_t> selected_experts;
    std::vector<DeepseekV4ProbeTrunkRange> trunk_ranges;
    std::vector<DeepseekV4ProbeExpert> experts;
};

struct DeepseekV4GlobalTensorRange {
    std::string tensor_name;
    model::TensorRole role{model::TensorRole::other};
    std::vector<std::uint64_t> shape;
    model::TensorSourceRange source;
    std::optional<std::uint32_t> ggml_type;
};

// Exact, non-executing plan for the six checkpoint-global tensors consumed by the pinned 0731
// graph. Each tensor remains one indivisible range so an imported artifact can be rebound to the
// verified catalog before one aggregate governed RAM reservation is attempted.
struct DeepseekV4GlobalTensorPlan {
    std::string model_id;
    std::string revision;
    bool metadata_only{true};
    bool imported_file_hashes_verified{false};
    bool full_payload_present{false};
    bool executable_source_binding_verified{false};
    ByteCount payload_bytes{0U};
    ByteCount maximum_read_range_bytes{0U};
    std::size_t tensor_count{0U};
    std::size_t unique_source_objects{0U};
    std::vector<DeepseekV4GlobalTensorRange> tensors;
};

[[nodiscard]] Result<DeepseekV4GgufCatalogReport> build_deepseek_v4_gguf_catalog(
    const model::GgufSplitCatalog& split,
    const DeepseekV4GgufCatalogOptions& options);

[[nodiscard]] const DeepseekV4GgufCheckpointSpec&
pinned_deepseek_v4_flash_0731_iq1s_checkpoint();

// The 2-bit sibling from the same repository and revision. Same architecture; only the routed
// experts' storage differs.
[[nodiscard]] const DeepseekV4GgufCheckpointSpec&
pinned_deepseek_v4_flash_0731_iq2xxs_checkpoint();

// Picks the pinned identity whose shards are the ones present under this root.
//
// Each quantization is pinned separately on purpose - a checkpoint must not be able to become a
// different one quietly - so a caller holding only a directory needs to be told which pin applies
// rather than guessing. Returns nothing when the directory matches no known identity, which is a
// refusal to execute rather than a licence to skip the check.
[[nodiscard]] const DeepseekV4GgufCheckpointSpec* pinned_deepseek_v4_flash_0731_checkpoint_for(
    const std::filesystem::path& checkpoint_root);

// Verifies an immutable importer manifest, exact object identities and (by default) every file
// hash before exposing payload paths. A full verified checkpoint is rejected unless one complete
// expert bundle fits in either the safe RAM tier or the safe VRAM tier. The two capacities are
// intentionally never pooled.
[[nodiscard]] Result<DeepseekV4GgufCatalogReport> inspect_deepseek_v4_gguf_checkpoint(
    const std::filesystem::path& checkpoint_root,
    const DeepseekV4GgufCheckpointSpec& spec,
    const DeepseekV4GgufCheckpointInspectOptions& options);

[[nodiscard]] Result<DeepseekV4LayerProbePlan> build_deepseek_v4_layer_probe_plan(
    const DeepseekV4GgufCatalogReport& report,
    std::uint32_t layer,
    std::span<const std::uint32_t> selected_experts,
    ByteCount tile_limit_bytes = model::kDefaultLayerStreamingTileBytes,
    std::size_t max_ranges = 4'096U);

[[nodiscard]] Result<DeepseekV4GlobalTensorPlan> build_deepseek_v4_global_tensor_plan(
    const DeepseekV4GgufCatalogReport& report);

[[nodiscard]] std::string to_json(const DeepseekV4GgufCatalogReport& report);
[[nodiscard]] std::string to_text(const DeepseekV4GgufCatalogReport& report);
[[nodiscard]] std::string to_json(const DeepseekV4LayerProbePlan& plan);
[[nodiscard]] std::string to_text(const DeepseekV4LayerProbePlan& plan);
[[nodiscard]] std::string to_json(const DeepseekV4GlobalTensorPlan& plan);
[[nodiscard]] std::string to_text(const DeepseekV4GlobalTensorPlan& plan);

}  // namespace vastloom::inference
