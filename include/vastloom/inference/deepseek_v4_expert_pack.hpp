#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/inference/deepseek_v4_gguf.hpp"
#include "vastloom/model/tensor_catalog.hpp"

namespace vastloom::inference {

inline constexpr const char* kDeepseekV4ExpertPackFormat =
    "vastloom.deepseek-v4-expert-pack.v1";
inline constexpr const char* kDeepseekV4ExpertPackManifestName = "vastloom-expert-pack.json";
inline constexpr std::size_t kMaximumDeepseekV4ExpertPackRecords = 256U;
inline constexpr ByteCount kMaximumDeepseekV4ExpertPackBytes = 2U * GiB;
inline constexpr ByteCount kDeepseekV4ExpertPackAlignment = 4U * 1024U;

struct DeepseekV4ExpertPackTensor {
    std::string name;
    ByteCount packed_offset{0U};
    ByteCount bytes{0U};
    std::optional<std::uint32_t> ggml_type;
    std::vector<std::uint64_t> shape;
};

// One expert's complete routed payload in one contiguous, aligned record. The record bytes are the
// same bytes the catalog bundle would have read from its separate ranges, in the same order, so a
// single read fills the same packed buffer the kernels already consume.
struct DeepseekV4ExpertPackRecord {
    std::uint32_t expert{0U};
    ByteCount offset{0U};
    ByteCount bytes{0U};
    std::string sha256;
    std::vector<DeepseekV4ExpertPackTensor> tensors;
};

struct DeepseekV4ExpertPack {
    std::filesystem::path root;
    std::filesystem::path payload_path;
    std::string model_id;
    std::string revision;
    std::uint32_t layer{0U};
    ByteCount payload_bytes{0U};
    std::vector<DeepseekV4ExpertPackRecord> records;
    std::vector<std::string> source_artifact_manifests;
    bool routing_authority_verified{false};
    bool full_hash_verified{false};
};

struct DeepseekV4ExpertPackVerificationOptions {
    bool verify_payload_hash{true};
    std::size_t max_records{kMaximumDeepseekV4ExpertPackRecords};
    ByteCount max_payload_bytes{kMaximumDeepseekV4ExpertPackBytes};
};

// Verifies a derived expert pack against the pinned catalog: format and identity, payload bounds,
// per-record alignment/no-overlap, exact tensor order/offsets/shapes/storage types, and (by
// default) the whole-payload SHA-256. The pack never becomes a complete checkpoint and does not
// change the checkpoint contract.
[[nodiscard]] Result<DeepseekV4ExpertPack> verify_deepseek_v4_expert_pack(
    const std::filesystem::path& root,
    const DeepseekV4GgufCatalogReport& report,
    std::uint32_t layer,
    const DeepseekV4ExpertPackVerificationOptions& options = {});

// Rewrites only the experts present in the pack so each bundle reads one contiguous local span.
// Tensors, packed offsets, bundle byte counts and kernel bindings are unchanged. Returns the number
// of bundles rewritten; a missing catalog bundle or a metadata mismatch fails the whole operation.
[[nodiscard]] Result<std::size_t> apply_deepseek_v4_expert_pack(
    model::LogicalTensorCatalog& catalog,
    const DeepseekV4ExpertPack& pack);

[[nodiscard]] std::string to_json(const DeepseekV4ExpertPack& pack);

}  // namespace vastloom::inference
