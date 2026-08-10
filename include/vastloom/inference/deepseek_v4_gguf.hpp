#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/model/gguf.hpp"
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
    // Optional immutable full-shard paths corresponding to split order. Header-prefix paths are
    // used only when this vector is empty.
    std::vector<std::filesystem::path> source_shards;
};

struct DeepseekV4GgufCatalogReport {
    model::LogicalTensorCatalog catalog;
    std::uint32_t layers{0U};
    std::uint32_t experts_per_layer{0U};
    std::uint32_t top_k{0U};
    ByteCount no_cache_expert_bytes_per_token{0U};
    ByteCount minimum_expert_bundle_bytes{0U};
    ByteCount maximum_expert_bundle_bytes{0U};
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
};

[[nodiscard]] Result<DeepseekV4GgufCatalogReport> build_deepseek_v4_gguf_catalog(
    const model::GgufSplitCatalog& split,
    const DeepseekV4GgufCatalogOptions& options);

[[nodiscard]] std::string to_json(const DeepseekV4GgufCatalogReport& report);
[[nodiscard]] std::string to_text(const DeepseekV4GgufCatalogReport& report);

}  // namespace vastloom::inference
