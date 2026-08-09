#include "stargate/model/manifest.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

#include "stargate/util/sha256.hpp"
#include "stargate/util/string.hpp"
#include "stargate/version.hpp"

namespace stargate::model {
namespace {

constexpr std::uintmax_t kMaximumManifestBytes = 16U * 1024U * 1024U;

Result<unsigned> parse_unsigned(std::string_view value, std::string_view key) {
    const std::string normalized = util::trim(value);
    unsigned parsed = 0;
    const auto result = std::from_chars(normalized.data(), normalized.data() + normalized.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != normalized.data() + normalized.size()) {
        return make_error(ErrorCode::invalid_format, "Invalid unsigned integer", std::string(key));
    }
    return parsed;
}

Result<ByteCount> parse_byte_field(std::string_view value, std::string_view key) {
    auto parsed = parse_bytes(value);
    if (!parsed) {
        return make_error(ErrorCode::invalid_format, parsed.error().message, std::string(key));
    }
    return parsed.take_value();
}

Result<bool> parse_bool_field(std::string_view value, std::string_view key) {
    bool parsed = false;
    if (!util::parse_bool(value, &parsed)) {
        return make_error(ErrorCode::invalid_format, "Invalid boolean", std::string(key));
    }
    return parsed;
}

Result<std::optional<unsigned>> parse_optional_index(std::string_view value, std::string_view key) {
    const auto normalized = util::trim(value);
    if (normalized.empty() || normalized == "-") {
        return std::optional<unsigned>{};
    }
    auto parsed = parse_unsigned(normalized, key);
    if (!parsed) {
        return parsed.error();
    }
    return std::optional<unsigned>{parsed.take_value()};
}

Result<ChunkDescriptor> parse_chunk_line(std::string_view line, std::size_t line_number) {
    const auto fields = util::split(line, '\t', true);
    if (fields.size() != 9U || fields[0] != "chunk") {
        return make_error(ErrorCode::invalid_format,
                          "Chunk lines must contain exactly 9 tab-separated fields",
                          "line " + std::to_string(line_number));
    }

    ChunkDescriptor chunk;
    chunk.relative_path = std::filesystem::path(fields[1]);
    if (!is_safe_relative_path(chunk.relative_path)) {
        return make_error(ErrorCode::unsafe_path, "Unsafe chunk path", fields[1]);
    }

    auto size = parse_byte_field(fields[2], "chunk.size_bytes");
    if (!size || size.value() == 0) {
        return make_error(ErrorCode::invalid_format, "Chunk size must be greater than zero", fields[1]);
    }
    chunk.size_bytes = size.take_value();

    chunk.sha256 = util::to_lower(util::trim(fields[3]));
    if (!util::is_sha256_hex(chunk.sha256)) {
        return make_error(ErrorCode::invalid_format, "Chunk SHA-256 must contain exactly 64 hexadecimal characters", fields[1]);
    }

    auto role = parse_chunk_role(fields[4]);
    if (!role) {
        return role.error();
    }
    chunk.role = role.take_value();
    chunk.dtype = util::trim(fields[5]);
    if (chunk.dtype.empty()) {
        return make_error(ErrorCode::invalid_format, "Chunk dtype must not be empty", fields[1]);
    }

    auto layer = parse_optional_index(fields[6], "chunk.layer");
    if (!layer) {
        return layer.error();
    }
    chunk.layer = layer.take_value();

    auto expert = parse_optional_index(fields[7], "chunk.expert");
    if (!expert) {
        return expert.error();
    }
    chunk.expert = expert.take_value();

    auto required = parse_bool_field(fields[8], "chunk.required");
    if (!required) {
        return required.error();
    }
    chunk.required = required.take_value();
    return chunk;
}

}  // namespace

Result<ModelManifest> load_manifest(const std::filesystem::path& path) {
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error) {
        return make_error(ErrorCode::not_found, "Unable to stat model manifest", path.string());
    }
    if (file_size > kMaximumManifestBytes) {
        return make_error(ErrorCode::invalid_format, "Manifest exceeds the 16 MiB safety limit", path.string());
    }

    std::ifstream input(path);
    if (!input) {
        return make_error(ErrorCode::io_error, "Unable to open model manifest", path.string());
    }

    ModelManifest manifest;
    manifest.source_path = path;
    std::set<std::string> seen_keys;
    std::set<std::filesystem::path> seen_chunk_paths;
    std::optional<unsigned> declared_chunk_count;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.size() > 64U * 1024U) {
            return make_error(ErrorCode::invalid_format, "Manifest line exceeds the 64 KiB safety limit", "line " + std::to_string(line_number));
        }
        const std::string trimmed = util::trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        if (util::starts_with(trimmed, "chunk\t")) {
            auto chunk = parse_chunk_line(trimmed, line_number);
            if (!chunk) {
                return chunk.error();
            }
            if (!seen_chunk_paths.insert(chunk.value().relative_path.lexically_normal()).second) {
                return make_error(ErrorCode::invalid_format, "Duplicate chunk path", chunk.value().relative_path.string());
            }
            manifest.chunks.push_back(chunk.take_value());
            continue;
        }

        const auto separator = trimmed.find('=');
        if (separator == std::string::npos) {
            return make_error(ErrorCode::invalid_format, "Expected key=value or a chunk line", "line " + std::to_string(line_number));
        }
        const std::string key = util::trim(std::string_view(trimmed).substr(0, separator));
        const std::string value = util::trim(std::string_view(trimmed).substr(separator + 1));
        if (key.empty()) {
            return make_error(ErrorCode::invalid_format, "Manifest key must not be empty", "line " + std::to_string(line_number));
        }
        if (!seen_keys.insert(key).second) {
            return make_error(ErrorCode::invalid_format, "Duplicate manifest key", key);
        }

        if (key == "format") manifest.format = value;
        else if (key == "model_id") manifest.model_id = value;
        else if (key == "revision") manifest.revision = value;
        else if (key == "architecture") manifest.architecture = value;
        else if (key == "tokenizer_family") manifest.tokenizer_family = value;
        else if (key == "metadata_only") {
            auto parsed = parse_bool_field(value, key); if (!parsed) return parsed.error(); manifest.metadata_only = parsed.take_value();
        } else if (key == "requires_avx2") {
            auto parsed = parse_bool_field(value, key); if (!parsed) return parsed.error(); manifest.requires_avx2 = parsed.take_value();
        } else if (key == "supports_speculation") {
            auto parsed = parse_bool_field(value, key); if (!parsed) return parsed.error(); manifest.supports_speculation = parsed.take_value();
        } else if (key == "model_total_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.model_total_bytes = parsed.take_value();
        } else if (key == "required_storage_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.required_storage_bytes = parsed.take_value();
        } else if (key == "dense_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.dense_bytes = parsed.take_value();
        } else if (key == "routed_expert_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.routed_expert_bytes = parsed.take_value();
        } else if (key == "active_bytes_per_token") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.active_bytes_per_token = parsed.take_value();
        } else if (key == "estimated_expert_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.estimated_expert_bytes = parsed.take_value();
        } else if (key == "minimum_ram_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.minimum_ram_bytes = parsed.take_value();
        } else if (key == "recommended_ram_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.recommended_ram_bytes = parsed.take_value();
        } else if (key == "minimum_vram_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.minimum_vram_bytes = parsed.take_value();
        } else if (key == "recommended_vram_bytes") {
            auto parsed = parse_byte_field(value, key); if (!parsed) return parsed.error(); manifest.recommended_vram_bytes = parsed.take_value();
        } else if (key == "layer_count") {
            auto parsed = parse_unsigned(value, key); if (!parsed) return parsed.error(); manifest.layer_count = parsed.take_value();
        } else if (key == "expert_count") {
            auto parsed = parse_unsigned(value, key); if (!parsed) return parsed.error(); manifest.expert_count = parsed.take_value();
        } else if (key == "selected_experts_per_token") {
            auto parsed = parse_unsigned(value, key); if (!parsed) return parsed.error(); manifest.selected_experts_per_token = parsed.take_value();
        } else if (key == "max_context_tokens") {
            auto parsed = parse_unsigned(value, key); if (!parsed) return parsed.error(); manifest.max_context_tokens = parsed.take_value();
        } else if (key == "chunk_count") {
            auto parsed = parse_unsigned(value, key); if (!parsed) return parsed.error(); declared_chunk_count = parsed.take_value();
        } else {
            manifest.extras.emplace(key, value);
            manifest.warnings.emplace_back("Unknown manifest key preserved for forward compatibility: " + key);
        }
    }

    if (!input.eof()) {
        return make_error(ErrorCode::io_error, "I/O error while reading model manifest", path.string());
    }
    if (manifest.format != kManifestFormat) {
        return make_error(ErrorCode::invalid_format, "Unsupported manifest format", manifest.format);
    }
    if (manifest.model_id.empty() || manifest.architecture.empty()) {
        return make_error(ErrorCode::invalid_format, "model_id and architecture are required", path.string());
    }
    if (declared_chunk_count.has_value() && *declared_chunk_count != manifest.chunks.size()) {
        return make_error(ErrorCode::integrity_error, "Declared chunk_count does not match parsed chunk count", path.string());
    }
    if (!manifest.metadata_only && manifest.chunks.empty()) {
        return make_error(ErrorCode::invalid_format, "A non-metadata manifest must contain at least one chunk", path.string());
    }

    ByteCount chunk_sum = 0;
    for (const auto& chunk : manifest.chunks) {
        auto sum = checked_add(chunk_sum, chunk.size_bytes);
        if (!sum) {
            return make_error(ErrorCode::integrity_error, "Chunk sizes overflow uint64", path.string());
        }
        chunk_sum = sum.take_value();
    }
    if (manifest.model_total_bytes == 0) {
        manifest.model_total_bytes = chunk_sum;
    } else if (!manifest.chunks.empty() && chunk_sum > manifest.model_total_bytes) {
        return make_error(ErrorCode::integrity_error, "Chunk bytes exceed model_total_bytes", path.string());
    } else if (!manifest.chunks.empty() && chunk_sum < manifest.model_total_bytes) {
        manifest.warnings.emplace_back("Manifest chunks do not yet account for all model_total_bytes.");
    }
    if (manifest.required_storage_bytes == 0) {
        manifest.required_storage_bytes = manifest.model_total_bytes;
    }
    if (manifest.dense_bytes > manifest.model_total_bytes || manifest.routed_expert_bytes > manifest.model_total_bytes) {
        return make_error(ErrorCode::integrity_error, "Dense or expert byte count exceeds model_total_bytes", path.string());
    }
    if (manifest.recommended_ram_bytes != 0 && manifest.recommended_ram_bytes < manifest.minimum_ram_bytes) {
        return make_error(ErrorCode::invalid_format, "recommended_ram_bytes is below minimum_ram_bytes", path.string());
    }
    if (manifest.recommended_vram_bytes != 0 && manifest.recommended_vram_bytes < manifest.minimum_vram_bytes) {
        return make_error(ErrorCode::invalid_format, "recommended_vram_bytes is below minimum_vram_bytes", path.string());
    }
    return manifest;
}

bool is_safe_relative_path(const std::filesystem::path& path) noexcept {
    if (path.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return false;
    }
    for (const auto& component : path) {
        if (component == ".." || component == "." || component.empty()) {
            return false;
        }
    }
    return true;
}

std::string_view to_string(ChunkRole role) noexcept {
    switch (role) {
        case ChunkRole::dense: return "dense";
        case ChunkRole::expert: return "expert";
        case ChunkRole::router: return "router";
        case ChunkRole::embedding: return "embedding";
        case ChunkRole::output: return "output";
        case ChunkRole::tokenizer: return "tokenizer";
        case ChunkRole::config: return "config";
        case ChunkRole::support: return "support";
        case ChunkRole::other: return "other";
    }
    return "other";
}

Result<ChunkRole> parse_chunk_role(std::string_view value) {
    const auto normalized = util::to_lower(util::trim(value));
    if (normalized == "dense") return ChunkRole::dense;
    if (normalized == "expert") return ChunkRole::expert;
    if (normalized == "router") return ChunkRole::router;
    if (normalized == "embedding") return ChunkRole::embedding;
    if (normalized == "output") return ChunkRole::output;
    if (normalized == "tokenizer") return ChunkRole::tokenizer;
    if (normalized == "config") return ChunkRole::config;
    if (normalized == "support") return ChunkRole::support;
    if (normalized == "other") return ChunkRole::other;
    return make_error(ErrorCode::invalid_format, "Unknown chunk role", normalized);
}

std::string to_text(const ModelManifest& manifest) {
    std::ostringstream output;
    output << "Model:        " << manifest.model_id << '\n'
           << "Revision:     " << (manifest.revision.empty() ? "unspecified" : manifest.revision) << '\n'
           << "Architecture: " << manifest.architecture << '\n'
           << "Footprint:    " << format_bytes(manifest.model_total_bytes) << '\n'
           << "Dense:        " << format_bytes(manifest.dense_bytes) << '\n'
           << "Experts:      " << format_bytes(manifest.routed_expert_bytes) << '\n'
           << "Active/token: " << format_bytes(manifest.active_bytes_per_token) << '\n'
           << "RAM minimum:  " << format_bytes(manifest.minimum_ram_bytes) << '\n'
           << "VRAM minimum: " << format_bytes(manifest.minimum_vram_bytes) << '\n'
           << "Chunks:       " << manifest.chunks.size() << (manifest.metadata_only ? " (metadata-only)" : "") << '\n';
    for (const auto& warning : manifest.warnings) {
        output << "WARNING:      " << warning << '\n';
    }
    return output.str();
}

std::string to_json(const ModelManifest& manifest) {
    std::ostringstream output;
    output << '{'
           << "\"format\":\"" << util::json_escape(manifest.format) << "\"," 
           << "\"model_id\":\"" << util::json_escape(manifest.model_id) << "\"," 
           << "\"revision\":\"" << util::json_escape(manifest.revision) << "\"," 
           << "\"architecture\":\"" << util::json_escape(manifest.architecture) << "\"," 
           << "\"metadata_only\":" << (manifest.metadata_only ? "true" : "false") << ','
           << "\"model_total_bytes\":" << manifest.model_total_bytes << ','
           << "\"required_storage_bytes\":" << manifest.required_storage_bytes << ','
           << "\"dense_bytes\":" << manifest.dense_bytes << ','
           << "\"routed_expert_bytes\":" << manifest.routed_expert_bytes << ','
           << "\"active_bytes_per_token\":" << manifest.active_bytes_per_token << ','
           << "\"minimum_ram_bytes\":" << manifest.minimum_ram_bytes << ','
           << "\"recommended_ram_bytes\":" << manifest.recommended_ram_bytes << ','
           << "\"minimum_vram_bytes\":" << manifest.minimum_vram_bytes << ','
           << "\"recommended_vram_bytes\":" << manifest.recommended_vram_bytes << ','
           << "\"layer_count\":" << manifest.layer_count << ','
           << "\"expert_count\":" << manifest.expert_count << ','
           << "\"selected_experts_per_token\":" << manifest.selected_experts_per_token << ','
           << "\"chunk_count\":" << manifest.chunks.size() << ','
           << "\"warnings\":[";
    for (std::size_t index = 0; index < manifest.warnings.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << util::json_escape(manifest.warnings[index]) << '"';
    }
    output << "]}";
    return output.str();
}

}  // namespace stargate::model
