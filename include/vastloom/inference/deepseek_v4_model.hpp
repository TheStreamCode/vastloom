#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/model/tensor_catalog.hpp"

namespace vastloom::inference {

struct DeepseekV4Config {
    std::uint32_t vocab_size{0};
    std::uint32_t hidden_size{0};
    std::uint32_t moe_intermediate_size{0};
    std::uint32_t num_hidden_layers{0};
    std::uint32_t num_attention_heads{0};
    std::uint32_t num_key_value_heads{0};
    std::uint32_t head_dim{0};
    std::uint32_t max_position_embeddings{0};
    std::uint32_t num_routed_experts{0};
    std::uint32_t num_shared_experts{0};
    std::uint32_t num_experts_per_token{0};
    std::uint32_t num_hash_layers{0};
    std::uint32_t num_nextn_predict_layers{0};
    std::uint32_t q_lora_rank{0};
    std::uint32_t o_lora_rank{0};
    std::uint32_t index_head_dim{0};
    std::uint32_t index_heads{0};
    std::uint32_t index_top_k{0};
    std::uint32_t sliding_window{0};
    std::uint32_t eos_token_id{0};
    std::uint32_t bos_token_id{0};
    float rms_norm_eps{0.0F};
    float rope_theta{0.0F};
    float routed_scaling_factor{0.0F};
};

struct DeepseekDTypeSummary {
    std::string dtype;
    std::size_t tensor_count{0};
    ByteCount bytes{0};
};

struct DeepseekV4Inspection {
    std::filesystem::path root;
    std::string model_id;
    std::string revision;
    DeepseekV4Config config;
    bool metadata_only{false};
    bool hashes_verified{false};
    bool header_catalog_complete{false};
    bool logical_catalog_valid{false};
    bool execution_supported{false};
    std::size_t shard_count{0};
    std::size_t tensor_count{0};
    std::size_t routed_expert_tensor_count{0};
    ByteCount imported_file_bytes{0};
    ByteCount checkpoint_file_bytes{0};
    ByteCount header_snapshot_bytes{0};
    ByteCount tensor_payload_bytes{0};
    ByteCount routed_expert_bytes{0};
    ByteCount always_active_bytes{0};
    ByteCount bytes_per_expert{0};
    ByteCount active_expert_bytes_per_layer{0};
    ByteCount no_cache_expert_bytes_per_token{0};
    ByteCount safe_ram_cap_bytes{0};
    std::vector<DeepseekDTypeSummary> dtypes;
    model::LogicalTensorCatalog logical_catalog;
};

[[nodiscard]] Result<DeepseekV4Config> load_deepseek_v4_config(
    const std::filesystem::path& config_path);
[[nodiscard]] Result<DeepseekV4Inspection> inspect_deepseek_v4_checkpoint(
    const std::filesystem::path& model_directory,
    bool verify_hashes = true,
    ByteCount safe_ram_cap_bytes = 0);
[[nodiscard]] std::string to_json(const DeepseekV4Inspection& inspection);
[[nodiscard]] std::string to_text(const DeepseekV4Inspection& inspection);

}  // namespace vastloom::inference
