#include "vastloom/model/model_store.hpp"

#include <array>
#include <chrono>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

#include "vastloom/util/string.hpp"

namespace vastloom::model {
namespace {

constexpr std::string_view kMarkerName = ".vastloom-store";
constexpr std::string_view kLegacyMarkerName = ".stargate-store";
constexpr std::string_view kMarkerContents =
    "format=vastloom.store.v1\n"
    "original_weights=immutable\n"
    "cache=reconstructable\n";
constexpr std::string_view kLegacyMarkerContents =
    "format=stargate.store.v1\n"
    "original_weights=immutable\n"
    "cache=reconstructable\n";

Result<ModelStoreLayout> make_layout(const std::filesystem::path& requested, ModelStoreOptions options) {
    if (requested.empty() || options.minimum_free_after_write < 20U * GiB) {
        return make_error(ErrorCode::invalid_argument, "Invalid model-store path or disk reserve", requested.string());
    }
    std::error_code error;
    const auto root = std::filesystem::absolute(requested, error).lexically_normal();
    if (error || root.empty() || root == root.root_path()) {
        return make_error(ErrorCode::unsafe_path, "Refusing unsafe model-store root", requested.string());
    }
    std::filesystem::path cursor;
    for (const auto& component : root) {
        cursor /= component;
        error.clear();
        if (std::filesystem::exists(cursor, error) && !error && std::filesystem::is_symlink(cursor, error)) {
            return make_error(ErrorCode::unsafe_path, "Model-store path traverses a symlink", cursor.string());
        }
    }
    return ModelStoreLayout{
        root,
        root / "models" / "original",
        root / "models" / "optimized",
        root / "cache",
        root / "scratch",
        root / "state",
        root / "telemetry",
        options.minimum_free_after_write,
    };
}

Result<void> validate_marker(const std::filesystem::path& marker, std::string_view expected_contents) {
    std::error_code error;
    const auto size = std::filesystem::file_size(marker, error);
    if (error || size > 4096U) return make_error(ErrorCode::integrity_error, "Invalid model-store marker", marker.string());
    std::ifstream input(marker, std::ios::binary);
    if (!input) return make_error(ErrorCode::io_error, "Unable to read model-store marker", marker.string());
    std::string contents(static_cast<std::size_t>(size), '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (input.gcount() != static_cast<std::streamsize>(contents.size()) || contents != expected_contents) {
        return make_error(ErrorCode::integrity_error, "Model-store marker contents do not match", marker.string());
    }
    return {};
}

Result<std::optional<std::filesystem::path>> find_existing_marker(const std::filesystem::path& root) {
    const auto current = root / kMarkerName;
    const auto legacy = root / kLegacyMarkerName;
    std::error_code error;
    const bool current_exists = std::filesystem::exists(current, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to inspect model-store marker", current.string());
    if (current_exists && std::filesystem::is_symlink(current, error)) {
        return make_error(ErrorCode::unsafe_path, "Model-store marker must not be a symlink", current.string());
    }
    if (error) return make_error(ErrorCode::io_error, "Unable to inspect model-store marker", current.string());
    if (current_exists && !std::filesystem::is_regular_file(current, error)) {
        return make_error(ErrorCode::integrity_error, "Model-store marker is not a regular file", current.string());
    }
    if (error) return make_error(ErrorCode::io_error, "Unable to inspect model-store marker", current.string());
    error.clear();
    const bool legacy_exists = std::filesystem::exists(legacy, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to inspect model-store marker", legacy.string());
    if (legacy_exists && std::filesystem::is_symlink(legacy, error)) {
        return make_error(ErrorCode::unsafe_path, "Model-store marker must not be a symlink", legacy.string());
    }
    if (error) return make_error(ErrorCode::io_error, "Unable to inspect model-store marker", legacy.string());
    if (legacy_exists && !std::filesystem::is_regular_file(legacy, error)) {
        return make_error(ErrorCode::integrity_error, "Model-store marker is not a regular file", legacy.string());
    }
    if (error) return make_error(ErrorCode::io_error, "Unable to inspect model-store marker", legacy.string());
    if (current_exists && legacy_exists) {
        return make_error(ErrorCode::integrity_error,
                          "Model store contains conflicting current and legacy markers",
                          root.string());
    }
    if (current_exists) {
        auto valid = validate_marker(current, kMarkerContents);
        if (!valid) return valid.error();
        return std::optional<std::filesystem::path>{current};
    }
    if (legacy_exists) {
        auto valid = validate_marker(legacy, kLegacyMarkerContents);
        if (!valid) return valid.error();
        return std::optional<std::filesystem::path>{legacy};
    }
    return std::optional<std::filesystem::path>{};
}

bool safe_component(std::string_view value) {
    if (value.empty() || value == "." || value == "..") return false;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        const bool allowed = (character >= 'a' && character <= 'z') ||
                             (character >= 'A' && character <= 'Z') ||
                             (character >= '0' && character <= '9') || character == '-' || character == '_' || character == '.';
        if (!allowed) return false;
    }
    return true;
}

std::string normalize_model_id(std::string_view model_id) {
    std::string result;
    result.reserve(model_id.size());
    for (const char character : model_id) result.push_back(character == '/' ? '-' : character);
    return result;
}

}  // namespace

Result<ModelStore> ModelStore::initialize(const std::filesystem::path& root, ModelStoreOptions options) {
    auto layout = make_layout(root, options);
    if (!layout) return layout.error();
    std::error_code error;
    const auto space = std::filesystem::space(layout.value().root.parent_path(), error);
    if (error || space.available < options.minimum_free_after_write) {
        return make_error(ErrorCode::resource_exhausted,
                          "Model store would violate the minimum free-space reserve",
                          layout.value().root.string());
    }
    const std::array<std::filesystem::path, 7> directories{
        layout.value().root,
        layout.value().original,
        layout.value().optimized,
        layout.value().cache,
        layout.value().scratch,
        layout.value().state,
        layout.value().telemetry,
    };
    for (const auto& directory : directories) {
        error.clear();
        std::filesystem::create_directories(directory, error);
        if (error) return make_error(ErrorCode::io_error, "Unable to create model-store directory", directory.string());
    }
    auto existing_marker = find_existing_marker(layout.value().root);
    if (!existing_marker) return existing_marker.error();
    if (!existing_marker.value().has_value()) {
        const auto marker = layout.value().root / kMarkerName;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto temporary = layout.value().root / (".vastloom-store.tmp-" + std::to_string(stamp));
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) return make_error(ErrorCode::io_error, "Unable to create temporary model-store marker", temporary.string());
            output << kMarkerContents;
            output.flush();
            if (!output) return make_error(ErrorCode::io_error, "Unable to flush temporary model-store marker", temporary.string());
        }
        std::filesystem::rename(temporary, marker, error);
        if (error) {
            std::error_code cleanup_error;
            std::filesystem::remove(temporary, cleanup_error);
            return make_error(ErrorCode::io_error, "Unable to atomically publish model-store marker", marker.string());
        }
    }
    return ModelStore(layout.take_value());
}

Result<ModelStore> ModelStore::open(const std::filesystem::path& root, ModelStoreOptions options) {
    auto layout = make_layout(root, options);
    if (!layout) return layout.error();
    auto marker = find_existing_marker(layout.value().root);
    if (!marker) return marker.error();
    if (!marker.value().has_value()) {
        return make_error(ErrorCode::not_found, "Model-store marker is missing", layout.value().root.string());
    }
    const std::array<std::filesystem::path, 6> directories{
        layout.value().original,
        layout.value().optimized,
        layout.value().cache,
        layout.value().scratch,
        layout.value().state,
        layout.value().telemetry,
    };
    std::error_code error;
    for (const auto& directory : directories) {
        if (!std::filesystem::is_directory(directory, error) || error) {
            return make_error(ErrorCode::integrity_error, "Model-store directory is missing", directory.string());
        }
    }
    return ModelStore(layout.take_value());
}

Result<void> ModelStore::preflight_write(ByteCount bytes) const {
    std::error_code error;
    const auto space = std::filesystem::space(layout_.root, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to query model-store free space", layout_.root.string());
    if (bytes > std::numeric_limits<ByteCount>::max() - layout_.minimum_free_after_write ||
        space.available < bytes + layout_.minimum_free_after_write) {
        return make_error(ErrorCode::resource_exhausted,
                          "Write would violate the model-store free-space reserve",
                          layout_.root.string());
    }
    return {};
}

Result<std::filesystem::path> ModelStore::immutable_model_path(std::string_view model_id,
                                                               std::string_view revision) const {
    std::size_t start = 0;
    while (start <= model_id.size()) {
        const std::size_t separator = model_id.find('/', start);
        const std::string_view component = model_id.substr(
            start, separator == std::string_view::npos ? std::string_view::npos : separator - start);
        if (!safe_component(component)) {
            return make_error(ErrorCode::unsafe_path, "Unsafe model id or revision", std::string(model_id));
        }
        if (separator == std::string_view::npos) break;
        start = separator + 1U;
    }
    const std::string normalized = normalize_model_id(model_id);
    if (!safe_component(normalized) || !safe_component(revision)) {
        return make_error(ErrorCode::unsafe_path, "Unsafe model id or revision", std::string(model_id));
    }
    return layout_.original / (normalized + "--" + std::string(revision));
}

std::string to_json(const ModelStoreLayout& layout) {
    std::ostringstream output;
    output << "{\"ok\":true,\"root\":\"" << util::json_escape(layout.root.string())
           << "\",\"original\":\"" << util::json_escape(layout.original.string())
           << "\",\"optimized\":\"" << util::json_escape(layout.optimized.string())
           << "\",\"cache\":\"" << util::json_escape(layout.cache.string())
           << "\",\"scratch\":\"" << util::json_escape(layout.scratch.string())
           << "\",\"state\":\"" << util::json_escape(layout.state.string())
           << "\",\"telemetry\":\"" << util::json_escape(layout.telemetry.string())
           << "\",\"minimum_free_after_write\":" << layout.minimum_free_after_write << '}';
    return output.str();
}

std::string to_text(const ModelStoreLayout& layout) {
    std::ostringstream output;
    output << "Vastloom model store: " << layout.root.string() << '\n'
           << "Original weights:     " << layout.original.string() << '\n'
           << "Derived models:       " << layout.optimized.string() << '\n'
           << "Cache:                " << layout.cache.string() << '\n'
           << "Scratch:              " << layout.scratch.string() << '\n'
           << "State:                " << layout.state.string() << '\n'
           << "Telemetry:            " << layout.telemetry.string() << '\n'
           << "Free-space reserve:   " << format_bytes(layout.minimum_free_after_write);
    return output.str();
}

}  // namespace vastloom::model
