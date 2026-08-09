#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::model {

enum class ChunkRole {
    dense,
    expert,
    router,
    embedding,
    output,
    tokenizer,
    config,
    support,
    other,
};

struct ChunkDescriptor {
    std::filesystem::path relative_path;
    ByteCount size_bytes{0};
    std::string sha256;
    ChunkRole role{ChunkRole::other};
    std::string dtype;
    std::optional<unsigned> layer;
    std::optional<unsigned> expert;
    bool required{true};
};

struct ModelManifest {
    std::filesystem::path source_path;
    std::string format;
    std::string model_id;
    std::string revision;
    std::string architecture;
    std::string tokenizer_family;
    bool metadata_only{false};
    bool requires_avx2{false};
    bool supports_speculation{false};

    ByteCount model_total_bytes{0};
    ByteCount required_storage_bytes{0};
    ByteCount dense_bytes{0};
    ByteCount routed_expert_bytes{0};
    ByteCount active_bytes_per_token{0};
    ByteCount estimated_expert_bytes{0};
    ByteCount minimum_ram_bytes{0};
    ByteCount recommended_ram_bytes{0};
    ByteCount minimum_vram_bytes{0};
    ByteCount recommended_vram_bytes{0};

    unsigned layer_count{0};
    unsigned expert_count{0};
    unsigned selected_experts_per_token{0};
    unsigned max_context_tokens{0};

    std::vector<ChunkDescriptor> chunks;
    std::map<std::string, std::string> extras;
    std::vector<std::string> warnings;
};

[[nodiscard]] Result<ModelManifest> load_manifest(const std::filesystem::path& path);
[[nodiscard]] bool is_safe_relative_path(const std::filesystem::path& path) noexcept;
[[nodiscard]] std::string_view to_string(ChunkRole role) noexcept;
[[nodiscard]] Result<ChunkRole> parse_chunk_role(std::string_view value);
[[nodiscard]] std::string to_text(const ModelManifest& manifest);
[[nodiscard]] std::string to_json(const ModelManifest& manifest);

}  // namespace vastloom::model
