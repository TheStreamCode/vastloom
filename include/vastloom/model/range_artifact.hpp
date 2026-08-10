#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::model {

inline constexpr const char* kRangeArtifactFormat = "vastloom.range-artifact.v1";
inline constexpr const char* kRangeArtifactManifestName = "vastloom-range-artifact.json";
inline constexpr std::size_t kMaximumRangeArtifactRanges = 4'096U;
inline constexpr ByteCount kMaximumRangeArtifactRangeBytes = 1U * GiB;
inline constexpr ByteCount kMaximumRangeArtifactPayloadBytes = 16U * GiB;

struct RangeArtifactEntry {
    std::string role;
    std::string label;
    std::filesystem::path relative_path;
    std::filesystem::path path;
    ByteCount bytes{0U};
    std::string sha256;
    std::string source_object;
    std::filesystem::path source_path;
    ByteCount source_object_bytes{0U};
    std::string source_object_sha256;
    ByteCount source_offset{0U};
    ByteCount source_bytes{0U};
};

// A range artifact is a closed, immutable derived container. Verification never promotes it to a
// complete checkpoint and never makes it executable; an architecture adapter must additionally
// bind every source span to an independently verified catalog/plan.
struct RangeArtifact {
    std::filesystem::path root;
    std::string model_id;
    std::string revision;
    std::string scope;
    std::uint32_t layer{0U};
    bool routing_authority_verified{false};
    std::string source_plan_format;
    std::string source_plan_sha256;
    ByteCount payload_bytes{0U};
    std::vector<RangeArtifactEntry> ranges;
    bool full_hash_verified{false};
    bool complete_checkpoint_verified{false};
    bool execution_supported{false};
};

struct RangeArtifactVerificationOptions {
    bool verify_hashes{true};
    std::size_t max_ranges{kMaximumRangeArtifactRanges};
    ByteCount max_range_bytes{kMaximumRangeArtifactRangeBytes};
    ByteCount max_payload_bytes{kMaximumRangeArtifactPayloadBytes};
};

[[nodiscard]] Result<RangeArtifact> verify_range_artifact(
    const std::filesystem::path& root,
    const RangeArtifactVerificationOptions& options = {});

[[nodiscard]] std::string to_json(const RangeArtifact& artifact);
[[nodiscard]] std::string to_text(const RangeArtifact& artifact);

}  // namespace vastloom::model
