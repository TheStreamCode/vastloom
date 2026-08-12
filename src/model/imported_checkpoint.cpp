#include "vastloom/model/imported_checkpoint.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vastloom/util/json.hpp"
#include "vastloom/util/string.hpp"
#include "vastloom/util/sha256.hpp"
#include "vastloom/version.hpp"

namespace vastloom::model {
namespace {

Result<std::filesystem::path> safe_file_path(const std::filesystem::path& root,
                                             const std::filesystem::path& relative) {
    if (relative.empty() || relative.is_absolute() || relative.has_root_name() || relative.has_root_directory()) {
        return make_error(ErrorCode::unsafe_path, "Imported checkpoint contains an unsafe file path", relative.string());
    }
    for (const auto& component : relative) {
        if (component == ".." || component == "." || component.empty()) {
            return make_error(ErrorCode::unsafe_path, "Imported checkpoint contains path traversal", relative.string());
        }
    }
    const auto candidate = root / relative;
    std::error_code error;
    if (std::filesystem::is_symlink(candidate, error)) {
        return make_error(ErrorCode::unsafe_path, "Imported checkpoint file is a symlink", candidate.string());
    }
    error.clear();
    const auto canonical_root = std::filesystem::weakly_canonical(root, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to canonicalize checkpoint root", root.string());
    const auto canonical_candidate = std::filesystem::weakly_canonical(candidate, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to canonicalize checkpoint file", candidate.string());
    auto root_iterator = canonical_root.begin();
    auto candidate_iterator = canonical_candidate.begin();
    for (; root_iterator != canonical_root.end(); ++root_iterator, ++candidate_iterator) {
        if (candidate_iterator == canonical_candidate.end() || *candidate_iterator != *root_iterator) {
            return make_error(ErrorCode::unsafe_path, "Imported checkpoint file escapes its root", candidate.string());
        }
    }
    return canonical_candidate;
}

// Hashing 82.54 GB takes minutes, and a command that must do it before every prompt is unusable.
// A receipt records what a completed verification saw: each file's size and last-write time beside
// the hash that was confirmed. If both still match, the bytes on disk are the bytes that were
// verified and the hash is not recomputed.
//
// This trusts the filesystem's own metadata, which is weaker than rehashing: a file edited in place
// while preserving size and timestamp would pass. That is the same trade every build system makes,
// and it is bounded here - any mismatch, a missing or malformed receipt, or a changed manifest
// forces the full verification again and rewrites the receipt.
constexpr std::string_view kVerificationReceiptName = "vastloom-verified.json";

struct ReceiptEntry {
    ByteCount bytes{0U};
    std::int64_t modified_ns{0};
    std::string sha256;
};

std::int64_t last_write_ns(const std::filesystem::path& path, std::error_code& error) {
    const auto stamp = std::filesystem::last_write_time(path, error);
    if (error) return 0;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stamp.time_since_epoch()).count();
}

std::map<std::string, ReceiptEntry, std::less<>> read_verification_receipt(
    const std::filesystem::path& root, std::string_view manifest_hash) {
    std::map<std::string, ReceiptEntry, std::less<>> entries;
    std::error_code error;
    const auto path = root / std::filesystem::path(std::string(kVerificationReceiptName));
    if (!std::filesystem::is_regular_file(path, error) || error) return entries;
    std::ifstream input(path, std::ios::binary);
    if (!input) return entries;
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    util::JsonLimits limits;
    limits.max_input_bytes = 4U * MiB;
    limits.max_nodes = 16'384;
    limits.max_array_elements = 1024;
    limits.max_object_members = 4096;
    auto document = util::parse_json(text, limits);
    if (!document) return entries;
    const auto* manifest = document.value().find("manifest_sha256");
    if (manifest == nullptr) return entries;
    auto manifest_value = manifest->as_string();
    if (!manifest_value || manifest_value.value() != manifest_hash) return entries;
    const auto* files = document.value().find("files");
    if (files == nullptr) return entries;
    auto array = files->as_array();
    if (!array) return entries;
    for (const auto& entry : *array.value()) {
        const auto* name = entry.find("name");
        const auto* bytes = entry.find("size");
        const auto* modified = entry.find("modified_ns");
        const auto* hash = entry.find("sha256");
        if (name == nullptr || bytes == nullptr || modified == nullptr || hash == nullptr) continue;
        auto name_value = name->as_string();
        auto bytes_value = bytes->as_u64();
        auto modified_value = modified->as_i64();
        auto hash_value = hash->as_string();
        if (!name_value || !bytes_value || !modified_value || !hash_value ||
            !util::is_sha256_hex(hash_value.value())) {
            continue;
        }
        entries.emplace(std::string(name_value.value()),
                        ReceiptEntry{bytes_value.value(), modified_value.value(),
                                     std::string(hash_value.value())});
    }
    return entries;
}

void write_verification_receipt(const std::filesystem::path& root,
                                std::string_view manifest_hash,
                                const std::vector<std::pair<std::string, ReceiptEntry>>& entries) {
    std::ostringstream document;
    document << "{\"format\":\"vastloom.verification-receipt.v1\",\"manifest_sha256\":\""
             << util::json_escape(std::string(manifest_hash)) << "\",\"files\":[";
    for (std::size_t index = 0U; index < entries.size(); ++index) {
        if (index != 0U) document << ',';
        document << "{\"name\":\"" << util::json_escape(entries[index].first)
                 << "\",\"size\":" << entries[index].second.bytes
                 << ",\"modified_ns\":" << entries[index].second.modified_ns
                 << ",\"sha256\":\"" << entries[index].second.sha256 << "\"}";
    }
    document << "]}";

    std::error_code error;
    const auto final_path = root / std::filesystem::path(std::string(kVerificationReceiptName));
    const auto staging = final_path.string() + ".partial";
    {
        std::ofstream output(staging, std::ios::binary | std::ios::trunc);
        if (!output) return;
        output << document.str();
        if (!output) return;
    }
    std::filesystem::rename(staging, final_path, error);
    if (error) std::filesystem::remove(staging, error);
}

}  // namespace

Result<ImportedCheckpoint> verify_imported_checkpoint(const std::filesystem::path& requested_root,
                                                       bool full_hash_verification) {
    std::error_code error;
    const auto root = std::filesystem::absolute(requested_root, error).lexically_normal();
    if (error || !std::filesystem::is_directory(root, error)) {
        return make_error(ErrorCode::not_found, "Imported checkpoint directory does not exist", requested_root.string());
    }
    util::JsonLimits limits;
    limits.max_input_bytes = 4U * MiB;
    limits.max_nodes = 16'384;
    limits.max_array_elements = 1024;
    limits.max_object_members = 4096;
    const auto current_manifest_path = root / "vastloom-import.json";
    const auto legacy_manifest_path = root / "stargate-import.json";
    error.clear();
    const bool current_exists = std::filesystem::exists(current_manifest_path, error);
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to inspect imported checkpoint manifest", current_manifest_path.string());
    }
    if (current_exists && std::filesystem::is_symlink(current_manifest_path, error)) {
        return make_error(ErrorCode::unsafe_path, "Imported checkpoint manifest must not be a symlink", current_manifest_path.string());
    }
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to inspect imported checkpoint manifest", current_manifest_path.string());
    }
    if (current_exists && !std::filesystem::is_regular_file(current_manifest_path, error)) {
        return make_error(ErrorCode::integrity_error, "Imported checkpoint manifest is not a regular file", current_manifest_path.string());
    }
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to inspect imported checkpoint manifest", current_manifest_path.string());
    }
    error.clear();
    const bool legacy_exists = std::filesystem::exists(legacy_manifest_path, error);
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to inspect imported checkpoint manifest", legacy_manifest_path.string());
    }
    if (legacy_exists && std::filesystem::is_symlink(legacy_manifest_path, error)) {
        return make_error(ErrorCode::unsafe_path, "Imported checkpoint manifest must not be a symlink", legacy_manifest_path.string());
    }
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to inspect imported checkpoint manifest", legacy_manifest_path.string());
    }
    if (legacy_exists && !std::filesystem::is_regular_file(legacy_manifest_path, error)) {
        return make_error(ErrorCode::integrity_error, "Imported checkpoint manifest is not a regular file", legacy_manifest_path.string());
    }
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to inspect imported checkpoint manifest", legacy_manifest_path.string());
    }
    if (current_exists == legacy_exists) {
        return make_error(
            ErrorCode::integrity_error,
            current_exists ? "Imported checkpoint contains conflicting current and legacy manifests"
                           : "Imported checkpoint manifest is missing",
            root.string());
    }
    const auto& manifest_path = current_exists ? current_manifest_path : legacy_manifest_path;
    auto manifest = util::load_json_file(manifest_path, limits);
    if (!manifest) return manifest.error();
    const auto* format_value = manifest.value().find("format");
    const auto* model_value = manifest.value().find("model_id");
    const auto* revision_value = manifest.value().find("revision");
    const auto* metadata_value = manifest.value().find("metadata_only");
    const auto* files_value = manifest.value().find("files");
    if (format_value == nullptr || model_value == nullptr || revision_value == nullptr ||
        metadata_value == nullptr || files_value == nullptr) {
        return make_error(ErrorCode::invalid_format, "Imported checkpoint manifest is missing required fields", root.string());
    }
    auto format = format_value->as_string();
    auto model_id = model_value->as_string();
    auto revision = revision_value->as_string();
    auto metadata_only = metadata_value->as_bool();
    auto files = files_value->as_array();
    if (!format ||
        (format.value() != kImportManifestFormat && format.value() != kLegacyImportManifestFormat) ||
        !model_id || model_id.value().empty() ||
        !revision || revision.value().empty() || !metadata_only || !files || files.value()->empty() || files.value()->size() > 1024U) {
        return make_error(ErrorCode::invalid_format, "Imported checkpoint manifest identity is invalid", root.string());
    }

    ImportedCheckpoint result;
    result.root = root;
    result.model_id = model_id.value();
    result.revision = revision.value();
    result.metadata_only = metadata_only.value();
    result.full_hash_verified = full_hash_verification;
    std::set<std::filesystem::path> seen;
    result.files.reserve(files.value()->size());

    // The receipt is keyed on the manifest's own hash, so a re-import that changes any expected
    // value invalidates every cached result at once.
    std::string manifest_hash;
    std::map<std::string, ReceiptEntry, std::less<>> receipt;
    std::vector<std::pair<std::string, ReceiptEntry>> confirmed;
    if (full_hash_verification) {
        auto hashed = util::sha256_file(manifest_path);
        if (hashed) {
            manifest_hash = hashed.take_value();
            receipt = read_verification_receipt(root, manifest_hash);
        }
        confirmed.reserve(files.value()->size());
    }

    for (const auto& entry : *files.value()) {
        const auto* name_value = entry.find("name");
        const auto* size_value = entry.find("size");
        const auto* hash_value = entry.find("sha256");
        if (name_value == nullptr || size_value == nullptr || hash_value == nullptr) {
            return make_error(ErrorCode::invalid_format, "Imported file entry is missing required fields", root.string());
        }
        auto name = name_value->as_string();
        auto bytes = size_value->as_u64();
        auto expected_hash = hash_value->as_string();
        if (!name || !bytes || !expected_hash || !util::is_sha256_hex(expected_hash.value())) {
            return make_error(ErrorCode::invalid_format, "Imported file entry is invalid", root.string());
        }
        const std::filesystem::path relative{std::string(name.value())};
        if (!seen.insert(relative).second) return make_error(ErrorCode::invalid_format, "Duplicate imported file entry", relative.string());
        auto candidate = safe_file_path(root, relative);
        if (!candidate) return candidate.error();
        const auto actual_bytes = std::filesystem::file_size(candidate.value(), error);
        if (error || actual_bytes != bytes.value()) {
            return make_error(ErrorCode::integrity_error, "Imported file size does not match manifest", candidate.value().string());
        }
        if (full_hash_verification) {
            const std::string key(name.value());
            std::error_code stamp_error;
            const auto modified_ns = last_write_ns(candidate.value(), stamp_error);
            const auto cached = receipt.find(key);
            const bool unchanged =
                !stamp_error && cached != receipt.end() && cached->second.bytes == bytes.value() &&
                cached->second.modified_ns == modified_ns &&
                cached->second.sha256 == expected_hash.value();
            if (!unchanged) {
                auto actual_hash = util::sha256_file(candidate.value());
                if (!actual_hash) return actual_hash.error();
                if (actual_hash.value() != expected_hash.value()) {
                    return make_error(ErrorCode::integrity_error, "Imported file SHA-256 does not match manifest", candidate.value().string());
                }
            }
            if (!manifest_hash.empty() && !stamp_error) {
                confirmed.emplace_back(
                    key, ReceiptEntry{bytes.value(), modified_ns,
                                      std::string(expected_hash.value())});
            }
        }
        if (bytes.value() > std::numeric_limits<ByteCount>::max() - result.total_bytes) {
            return make_error(ErrorCode::invalid_format, "Imported checkpoint byte count overflows", root.string());
        }
        result.total_bytes += bytes.value();
        result.files.push_back(ImportedFile{relative, bytes.value(), std::string(expected_hash.value())});
    }

    // Written only after every file passed, so a failed verification never leaves a receipt that
    // would let the next run skip the work.
    if (full_hash_verification && !manifest_hash.empty() &&
        confirmed.size() == result.files.size()) {
        write_verification_receipt(root, manifest_hash, confirmed);
    }
    return result;
}

}  // namespace vastloom::model
