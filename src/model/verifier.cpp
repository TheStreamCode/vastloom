#include "vastloom/model/verifier.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "vastloom/util/sha256.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::model {
namespace {

bool path_is_within(const std::filesystem::path& root, const std::filesystem::path& candidate) {
    auto root_iterator = root.begin();
    auto candidate_iterator = candidate.begin();
    for (; root_iterator != root.end(); ++root_iterator, ++candidate_iterator) {
        if (candidate_iterator == candidate.end() || *root_iterator != *candidate_iterator) {
            return false;
        }
    }
    return true;
}

void add_issue(VerificationReport& report,
               VerificationSeverity severity,
               const std::filesystem::path& path,
               std::string message) {
    report.issues.push_back(VerificationIssue{severity, path, std::move(message)});
    if (severity == VerificationSeverity::error) {
        report.ok = false;
    }
}

}  // namespace

Result<VerificationReport> verify_model(const ModelManifest& manifest,
                                        const std::filesystem::path& model_root,
                                        VerificationMode mode) {
    VerificationReport report;
    report.mode = mode;
    std::error_code error;
    const auto canonical_root = std::filesystem::weakly_canonical(model_root, error);
    if (error || !std::filesystem::exists(canonical_root)) {
        return make_error(ErrorCode::not_found, "Model root does not exist", model_root.string());
    }
    if (!std::filesystem::is_directory(canonical_root, error) || error) {
        return make_error(ErrorCode::invalid_argument, "Model root is not a directory", canonical_root.string());
    }

    if (manifest.metadata_only || manifest.chunks.empty()) {
        add_issue(report, VerificationSeverity::warning, manifest.source_path,
                  "The manifest is metadata-only; no weight chunks were verified.");
        return report;
    }

    for (const auto& chunk : manifest.chunks) {
        ++report.chunks_checked;
        if (!is_safe_relative_path(chunk.relative_path)) {
            ++report.unsafe_paths;
            add_issue(report, VerificationSeverity::error, chunk.relative_path, "Unsafe relative path in manifest");
            continue;
        }

        const auto candidate = canonical_root / chunk.relative_path;
        if (!std::filesystem::exists(candidate, error) || error) {
            error.clear();
            ++report.missing_chunks;
            add_issue(report,
                      chunk.required ? VerificationSeverity::error : VerificationSeverity::warning,
                      chunk.relative_path,
                      chunk.required ? "Required chunk is missing" : "Optional chunk is missing");
            continue;
        }

        const auto canonical_candidate = std::filesystem::weakly_canonical(candidate, error);
        if (error || !path_is_within(canonical_root, canonical_candidate)) {
            error.clear();
            ++report.unsafe_paths;
            add_issue(report, VerificationSeverity::error, chunk.relative_path,
                      "Chunk resolves outside model root (possible symlink escape)");
            continue;
        }
        if (!std::filesystem::is_regular_file(canonical_candidate, error) || error) {
            error.clear();
            add_issue(report, VerificationSeverity::error, chunk.relative_path, "Chunk is not a regular file");
            continue;
        }

        const auto actual_size = std::filesystem::file_size(canonical_candidate, error);
        if (error) {
            error.clear();
            add_issue(report, VerificationSeverity::error, chunk.relative_path, "Unable to read chunk size");
            continue;
        }
        if (actual_size != chunk.size_bytes) {
            ++report.size_mismatches;
            add_issue(report, VerificationSeverity::error, chunk.relative_path,
                      "Chunk size mismatch: expected " + std::to_string(chunk.size_bytes) +
                          ", found " + std::to_string(actual_size));
            continue;
        }
        report.bytes_checked += static_cast<ByteCount>(actual_size);

        if (mode == VerificationMode::full) {
            auto actual_hash = util::sha256_file(canonical_candidate);
            if (!actual_hash) {
                add_issue(report, VerificationSeverity::error, chunk.relative_path, actual_hash.error().message);
                continue;
            }
            if (util::to_lower(actual_hash.value()) != util::to_lower(chunk.sha256)) {
                ++report.hash_mismatches;
                add_issue(report, VerificationSeverity::error, chunk.relative_path,
                          "SHA-256 mismatch: expected " + chunk.sha256 + ", found " + actual_hash.value());
            }
        }
    }
    return report;
}

std::string to_text(const VerificationReport& report) {
    std::ostringstream output;
    output << "Verification: " << (report.ok ? "PASS" : "FAIL")
           << " (" << (report.mode == VerificationMode::full ? "full SHA-256" : "quick size") << ")\n"
           << "Chunks:       " << report.chunks_checked << '\n'
           << "Bytes:        " << format_bytes(report.bytes_checked) << '\n'
           << "Missing:      " << report.missing_chunks << '\n'
           << "Size errors:  " << report.size_mismatches << '\n'
           << "Hash errors:  " << report.hash_mismatches << '\n'
           << "Unsafe paths: " << report.unsafe_paths << '\n';
    for (const auto& issue : report.issues) {
        output << (issue.severity == VerificationSeverity::error ? "ERROR:   " : "WARNING: ")
               << issue.path.string() << ": " << issue.message << '\n';
    }
    return output.str();
}

std::string to_json(const VerificationReport& report) {
    std::ostringstream output;
    output << '{'
           << "\"ok\":" << (report.ok ? "true" : "false") << ','
           << "\"mode\":\"" << (report.mode == VerificationMode::full ? "full" : "quick") << "\"," 
           << "\"chunks_checked\":" << report.chunks_checked << ','
           << "\"bytes_checked\":" << report.bytes_checked << ','
           << "\"missing_chunks\":" << report.missing_chunks << ','
           << "\"size_mismatches\":" << report.size_mismatches << ','
           << "\"hash_mismatches\":" << report.hash_mismatches << ','
           << "\"unsafe_paths\":" << report.unsafe_paths << ','
           << "\"issues\":[";
    for (std::size_t index = 0; index < report.issues.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& issue = report.issues[index];
        output << '{'
               << "\"severity\":\"" << (issue.severity == VerificationSeverity::error ? "error" : "warning") << "\"," 
               << "\"path\":\"" << util::json_escape(issue.path.string()) << "\"," 
               << "\"message\":\"" << util::json_escape(issue.message) << "\"}";
    }
    output << "]}";
    return output.str();
}

}  // namespace vastloom::model
