#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/model/safetensors.hpp"

namespace vastloom::model {

enum class TensorRole : std::uint8_t {
    embedding,
    output_head,
    normalization,
    attention,
    router,
    shared_expert,
    routed_expert,
    hyper_connection,
    speculative,
    other,
};

struct TensorSourceRange {
    std::filesystem::path shard;
    ByteCount offset{0};
    ByteCount bytes{0};
};

struct LogicalTensorDescriptor {
    std::string name;
    TensorDType dtype{TensorDType::f32};
    std::vector<std::uint64_t> shape;
    TensorRole role{TensorRole::other};
    std::optional<std::uint32_t> layer;
    std::optional<std::uint32_t> expert;
    bool speculative{false};
    TensorSourceRange source;
};

struct ExpertBundle {
    std::uint32_t layer{0};
    std::uint32_t expert{0};
    bool speculative{false};
    ByteCount bytes{0};
    std::vector<std::size_t> tensor_indices;
    std::vector<TensorSourceRange> read_ranges;
};

struct LogicalTensorCatalog {
    std::string model_id;
    std::string revision;
    bool metadata_only{false};
    std::vector<LogicalTensorDescriptor> tensors;
    std::vector<ExpertBundle> experts;
    ByteCount tensor_bytes{0};
    ByteCount always_active_bytes{0};
    ByteCount routed_expert_bytes{0};
    std::size_t expert_read_range_count{0};
};

[[nodiscard]] Result<void> finalize_logical_tensor_catalog(
    LogicalTensorCatalog& catalog,
    std::size_t max_tensors = 1'000'000U,
    std::size_t max_experts = 1'000'000U);

[[nodiscard]] std::string_view to_string(TensorRole role) noexcept;

}  // namespace vastloom::model
