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
