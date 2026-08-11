#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

struct DeepseekV4BoundTrunkTensor {
    std::size_t catalog_tensor_index{0U};
    std::string name;
    model::TensorDType dtype{model::TensorDType::f32};
    std::vector<std::uint64_t> shape;
    model::TensorRole role{model::TensorRole::other};
    std::optional<std::uint32_t> ggml_type;
    ByteCount packed_offset{0U};
    ByteCount bytes{0U};
};

// A closed, complete non-routed layer tensor set backed by one Governor-accounted immutable RAM
// buffer. It remains non-executing by itself; graph consumers must bind and validate each storage
// type and logical shape they understand.
class DeepseekV4LoadedTrunkTensorSet {
public:
    DeepseekV4LoadedTrunkTensorSet(const DeepseekV4LoadedTrunkTensorSet&) = delete;
    DeepseekV4LoadedTrunkTensorSet& operator=(const DeepseekV4LoadedTrunkTensorSet&) = delete;
    DeepseekV4LoadedTrunkTensorSet(DeepseekV4LoadedTrunkTensorSet&&) noexcept = default;
    DeepseekV4LoadedTrunkTensorSet& operator=(DeepseekV4LoadedTrunkTensorSet&&) = delete;

    [[nodiscard]] const std::string& logical_id() const noexcept { return logical_id_; }
    [[nodiscard]] ByteCount payload_bytes() const noexcept { return payload_bytes_; }
    [[nodiscard]] const std::vector<DeepseekV4BoundTrunkTensor>& tensors() const noexcept {
        return tensors_;
    }
    [[nodiscard]] const DeepseekV4BoundTrunkTensor* find(std::string_view name) const noexcept;
    [[nodiscard]] Result<std::span<const std::byte>> tensor(std::string_view name) const;
    [[nodiscard]] std::shared_ptr<const model::ResidentByteBuffer> retain_buffer() const noexcept {
        return buffer_;
    }
    [[nodiscard]] const model::BoundRangeSetLoadTelemetry& telemetry() const noexcept {
        return telemetry_;
    }

private:
    friend Result<DeepseekV4LoadedTrunkTensorSet>
    load_deepseek_v4_bound_trunk_tensor_set(
        const DeepseekV4TrunkArtifactBinding&,
        const DeepseekV4GgufCatalogReport&,
        model::ResidencyObject&,
        safety::ResourceGovernor&,
        const model::BoundRangeSetLoadOptions&);
    DeepseekV4LoadedTrunkTensorSet(
        std::string logical_id,
        ByteCount payload_bytes,
        std::vector<DeepseekV4BoundTrunkTensor> tensors,
        std::shared_ptr<const model::ResidentByteBuffer> buffer,
        model::BoundRangeSetLoadTelemetry telemetry);

    std::string logical_id_;
    ByteCount payload_bytes_{0U};
    std::vector<DeepseekV4BoundTrunkTensor> tensors_;
    std::shared_ptr<const model::ResidentByteBuffer> buffer_;
    model::BoundRangeSetLoadTelemetry telemetry_;
};

struct DeepseekV4BoundGlobalRange {
    std::size_t artifact_index{0U};
    std::string tensor_name;
    model::TensorRole role{model::TensorRole::other};
    std::optional<std::uint32_t> ggml_type;
    model::TensorSourceRange original_source;
    model::TensorSourceRange local_source;
    std::string local_sha256;
};

// Exact projection of a hash-verified six-range artifact onto the pinned checkpoint catalog.
// The binding remains non-executing until every global consumer validates its logical shape.
struct DeepseekV4GlobalArtifactBinding {
    std::filesystem::path artifact_root;
    std::string model_id;
    std::string revision;
    std::string source_plan_sha256;
    ByteCount payload_bytes{0U};
    ByteCount maximum_local_range_bytes{0U};
    std::vector<DeepseekV4BoundGlobalRange> ranges;
    bool metadata_catalog_hashes_verified{false};
    bool local_range_hashes_verified{false};
    bool exact_source_binding_verified{false};
    bool complete_checkpoint_verified{false};
    bool execution_supported{false};
};

struct DeepseekV4BoundGlobalTensor {
    std::size_t catalog_tensor_index{0U};
    std::string name;
    model::TensorDType dtype{model::TensorDType::f32};
    std::vector<std::uint64_t> shape;
    model::TensorRole role{model::TensorRole::other};
    std::optional<std::uint32_t> ggml_type;
    ByteCount packed_offset{0U};
    ByteCount bytes{0U};
};

// The six globals share one immutable buffer and therefore one ResidencyUse. Consumer descriptors
// retain that buffer without duplicating its reservation or payload.
class DeepseekV4LoadedGlobalTensorSet {
public:
    DeepseekV4LoadedGlobalTensorSet(const DeepseekV4LoadedGlobalTensorSet&) = delete;
    DeepseekV4LoadedGlobalTensorSet& operator=(const DeepseekV4LoadedGlobalTensorSet&) = delete;
    DeepseekV4LoadedGlobalTensorSet(DeepseekV4LoadedGlobalTensorSet&&) noexcept = default;
    DeepseekV4LoadedGlobalTensorSet& operator=(DeepseekV4LoadedGlobalTensorSet&&) = delete;

    [[nodiscard]] const std::string& logical_id() const noexcept { return logical_id_; }
    [[nodiscard]] ByteCount payload_bytes() const noexcept { return payload_bytes_; }
    [[nodiscard]] const std::vector<DeepseekV4BoundGlobalTensor>& tensors() const noexcept {
        return tensors_;
    }
    [[nodiscard]] const DeepseekV4BoundGlobalTensor* find(std::string_view name) const noexcept;
    [[nodiscard]] Result<std::span<const std::byte>> tensor(std::string_view name) const;
    [[nodiscard]] std::shared_ptr<const model::ResidentByteBuffer> retain_buffer() const noexcept {
        return buffer_;
    }
    [[nodiscard]] const model::BoundRangeSetLoadTelemetry& telemetry() const noexcept {
        return telemetry_;
    }

private:
    friend Result<DeepseekV4LoadedGlobalTensorSet>
    load_deepseek_v4_bound_global_tensor_set(
        const DeepseekV4GlobalArtifactBinding&,
        const DeepseekV4GgufCatalogReport&,
        model::ResidencyObject&,
        safety::ResourceGovernor&,
        const model::BoundRangeSetLoadOptions&);
    DeepseekV4LoadedGlobalTensorSet(
        std::string logical_id,
        ByteCount payload_bytes,
        std::vector<DeepseekV4BoundGlobalTensor> tensors,
        std::shared_ptr<const model::ResidentByteBuffer> buffer,
        model::BoundRangeSetLoadTelemetry telemetry);

    std::string logical_id_;
    ByteCount payload_bytes_{0U};
    std::vector<DeepseekV4BoundGlobalTensor> tensors_;
    std::shared_ptr<const model::ResidentByteBuffer> buffer_;
    model::BoundRangeSetLoadTelemetry telemetry_;
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

[[nodiscard]] Result<std::string> deepseek_v4_bound_trunk_tensor_set_logical_id(
    const DeepseekV4TrunkArtifactBinding& binding);

// Revalidates the selected binding identity and loads exactly one local range through a governed
// RAM transaction. The in-memory bytes are hashed again before RAM_READY is published.
[[nodiscard]] Result<model::LoadedBoundRange> load_deepseek_v4_bound_trunk_range(
    const DeepseekV4TrunkArtifactBinding& binding,
    std::size_t range_index,
    model::ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const model::BoundRangeLoadOptions& options = {});

// Requires exactly one complete local range for every non-routed tensor in the bound layer.
// Missing, duplicate, split or catalog/type/source mismatched sets fail before any residency
// change.
[[nodiscard]] Result<DeepseekV4LoadedTrunkTensorSet>
load_deepseek_v4_bound_trunk_tensor_set(
    const DeepseekV4TrunkArtifactBinding& binding,
    const DeepseekV4GgufCatalogReport& report,
    model::ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const model::BoundRangeSetLoadOptions& options = {});

// The authoritative expert selection of one token in a hash-routed layer.
struct DeepseekV4HashRouteResolution {
    std::uint32_t layer{0U};
    std::uint32_t token{0U};
    std::uint32_t expert_count{0U};
    std::uint32_t experts_per_token{0U};
    std::string table_tensor_name;
    std::string table_range_sha256;
    ByteCount table_bytes{0U};
    std::vector<std::uint32_t> slot_experts;
    std::vector<std::uint32_t> unique_experts;
};

// Resolves which experts a token actually selects, from the imported trunk payload.
//
// This is the one routing decision that can be proven without executing the model. In a hash-routed
// layer the selection is a pure lookup: `indices[slot] = tid2eid[token * top_k + slot]`. The gate
// logits weight those experts but cannot change which ones are chosen. Layers with a learned router
// are refused: their selection depends on activations that only execution can produce.
//
// The table is read through the same governed, rehashed range path as any other payload, so a
// result here carries the artifact's own integrity guarantee.
// The routing table is read through its own bounded residency, which this function owns: only it
// knows which range holds the table, and the loader requires a residency created for exactly that
// range's logical identity and byte count.
[[nodiscard]] Result<DeepseekV4HashRouteResolution> deepseek_v4_resolve_hash_route(
    const DeepseekV4TrunkArtifactBinding& binding,
    const DeepseekV4GgufCatalogReport& report,
    std::uint32_t token,
    safety::ResourceGovernor& governor);

[[nodiscard]] Result<DeepseekV4GlobalArtifactBinding>
bind_deepseek_v4_global_range_artifact(
    const model::RangeArtifact& artifact,
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4GlobalTensorPlan& plan,
    const DeepseekV4GgufCheckpointSpec& checkpoint_spec);

[[nodiscard]] Result<std::string> deepseek_v4_bound_global_tensor_set_logical_id(
    const DeepseekV4GlobalArtifactBinding& binding);

// The caller must explicitly admit the verified aggregate payload. The generic 512 MiB default
// intentionally refuses the current official 0731 globals until a larger bounded option and
// matching Governor budget are supplied.
[[nodiscard]] Result<DeepseekV4LoadedGlobalTensorSet>
load_deepseek_v4_bound_global_tensor_set(
    const DeepseekV4GlobalArtifactBinding& binding,
    const DeepseekV4GgufCatalogReport& report,
    model::ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const model::BoundRangeSetLoadOptions& options = {});

[[nodiscard]] std::string to_json(const DeepseekV4TrunkArtifactBinding& binding);
[[nodiscard]] std::string to_text(const DeepseekV4TrunkArtifactBinding& binding);
[[nodiscard]] std::string to_json(const DeepseekV4GlobalArtifactBinding& binding);
[[nodiscard]] std::string to_text(const DeepseekV4GlobalArtifactBinding& binding);

}  // namespace vastloom::inference
