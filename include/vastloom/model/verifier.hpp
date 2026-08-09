#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/model/manifest.hpp"

namespace vastloom::model {

enum class VerificationMode {
    quick,
    full,
};

enum class VerificationSeverity {
    warning,
    error,
};

struct VerificationIssue {
    VerificationSeverity severity{VerificationSeverity::error};
    std::filesystem::path path;
    std::string message;
};

struct VerificationReport {
    bool ok{true};
    VerificationMode mode{VerificationMode::quick};
    std::size_t chunks_checked{0};
    ByteCount bytes_checked{0};
    std::size_t missing_chunks{0};
    std::size_t size_mismatches{0};
    std::size_t hash_mismatches{0};
    std::size_t unsafe_paths{0};
    std::vector<VerificationIssue> issues;
};

[[nodiscard]] Result<VerificationReport> verify_model(const ModelManifest& manifest,
                                                      const std::filesystem::path& model_root,
                                                      VerificationMode mode);
[[nodiscard]] std::string to_text(const VerificationReport& report);
[[nodiscard]] std::string to_json(const VerificationReport& report);

}  // namespace vastloom::model
