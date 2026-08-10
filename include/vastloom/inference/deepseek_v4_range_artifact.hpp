#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/inference/deepseek_v4_gguf.hpp"
#include "vastloom/model/bound_range_loader.hpp"
#include "vastloom/model/range_artifact.hpp"

namespace vastloom::inference {

struct DeepseekV4BoundTrunkRange {
    std::size_t artifact_index{0U};
    std::string tensor_name;
    model::TensorRole role{model::TensorRole::other};
    ByteCount tensor_offset{0U};
    bool final_tile{false};
    std::optional<std::uint32_t> ggml_type;
    model::TensorSourceRange original_source;
    model::TensorSourceRange local_source;
    std::string local_sha256;
};

// This proves that each hash-verified local trunk range is an exact projection of the pinned
// checkpoint catalog and deterministic layer plan. It deliberately does not mark the artifact as
// a complete checkpoint or authorize execution of the official DeepSeek graph.
struct DeepseekV4TrunkArtifactBinding {
    std::filesystem::path artifact_root;
    std::string model_id;
    std::string revision;
    std::string source_plan_sha256;
    std::uint32_t layer{0U};
    ByteCount payload_bytes{0U};
    ByteCount maximum_local_range_bytes{0U};
    std::vector<DeepseekV4BoundTrunkRange> ranges;
    bool metadata_catalog_hashes_verified{false};
    bool local_range_hashes_verified{false};
    bool exact_source_binding_verified{false};
    bool complete_checkpoint_verified{false};
    bool execution_supported{false};
};

[[nodiscard]] Result<DeepseekV4TrunkArtifactBinding>
bind_deepseek_v4_trunk_range_artifact(
    const model::RangeArtifact& artifact,
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4LayerProbePlan& plan,
    const DeepseekV4GgufCheckpointSpec& checkpoint_spec);

[[nodiscard]] Result<std::string> deepseek_v4_bound_trunk_range_logical_id(
    const DeepseekV4TrunkArtifactBinding& binding,
    std::size_t range_index);

// Revalidates the selected binding identity and loads exactly one local range through a governed
// RAM transaction. The in-memory bytes are hashed again before RAM_READY is published.
[[nodiscard]] Result<model::LoadedBoundRange> load_deepseek_v4_bound_trunk_range(
    const DeepseekV4TrunkArtifactBinding& binding,
    std::size_t range_index,
    model::ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const model::BoundRangeLoadOptions& options = {});

[[nodiscard]] std::string to_json(const DeepseekV4TrunkArtifactBinding& binding);
[[nodiscard]] std::string to_text(const DeepseekV4TrunkArtifactBinding& binding);

}  // namespace vastloom::inference
