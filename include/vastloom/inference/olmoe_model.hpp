#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::inference {

struct OlmoeConfig {
    std::uint32_t vocab_size{0};
    std::uint32_t hidden_size{0};
    std::uint32_t intermediate_size{0};
    std::uint32_t num_hidden_layers{0};
    std::uint32_t num_attention_heads{0};
    std::uint32_t num_key_value_heads{0};
    std::uint32_t max_position_embeddings{0};
    std::uint32_t num_experts{0};
    std::uint32_t num_experts_per_token{0};
    std::uint32_t eos_token_id{0};
    std::uint32_t pad_token_id{0};
    float rms_norm_eps{1.0e-5F};
    float rope_theta{10'000.0F};
};

struct OlmoeInspection {
    std::filesystem::path root;
    std::string model_id;
    std::string revision;
    OlmoeConfig config;
    bool metadata_only{false};
    bool hashes_verified{false};
    std::size_t tensor_count{0};
    ByteCount imported_file_bytes{0};
    ByteCount tensor_payload_bytes{0};
    ByteCount routed_expert_bytes{0};
    ByteCount always_active_bytes{0};
    ByteCount bytes_per_expert{0};
    ByteCount active_expert_bytes_per_layer{0};
    ByteCount no_cache_expert_bytes_per_token{0};
    ByteCount estimated_minimum_streaming_ram_bytes{0};
    ByteCount safe_ram_cap_bytes{0};
    bool resident_load_supported{false};
    bool streaming_plan_candidate{false};
};

[[nodiscard]] Result<OlmoeConfig> load_olmoe_config(const std::filesystem::path& config_path);
[[nodiscard]] Result<OlmoeInspection> inspect_olmoe_checkpoint(
    const std::filesystem::path& model_directory,
    bool verify_hashes = true,
    ByteCount safe_ram_cap_bytes = 0);
[[nodiscard]] std::string to_json(const OlmoeInspection& inspection);
[[nodiscard]] std::string to_text(const OlmoeInspection& inspection);

}  // namespace vastloom::inference
