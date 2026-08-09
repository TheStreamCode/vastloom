#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::model {

struct ImportedFile {
    std::filesystem::path relative_path;
    ByteCount bytes{0};
    std::string sha256;
};

struct ImportedCheckpoint {
    std::filesystem::path root;
    std::string model_id;
    std::string revision;
    bool metadata_only{false};
    std::vector<ImportedFile> files;
    ByteCount total_bytes{0};
    bool full_hash_verified{false};
};

[[nodiscard]] Result<ImportedCheckpoint> verify_imported_checkpoint(const std::filesystem::path& root,
                                                                     bool full_hash_verification = true);

}  // namespace vastloom::model
