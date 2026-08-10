#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/io/device_io_counters.hpp"
#include "vastloom/io/process_io_counters.hpp"
#include "vastloom/model/residency.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::model {

inline constexpr ByteCount kDefaultLayerStreamingTileBytes = 64U * MiB;

struct LayerTensorTile {
    std::size_t tensor_index{0U};
    std::uint32_t layer{0U};
    TensorRole role{TensorRole::other};
    ByteCount tensor_offset{0U};
    TensorSourceRange source;
    bool final_tile{false};
};

struct LayerStreamingPlan {
    std::string model_id;
    std::uint32_t layer{0U};
    ByteCount tile_limit_bytes{0U};
    ByteCount trunk_tensor_bytes{0U};
    ByteCount maximum_tensor_bytes{0U};
    ByteCount maximum_tile_bytes{0U};
    std::size_t tensor_count{0U};
    std::vector<LayerTensorTile> tiles;
};

// Produces a deterministic tensor-by-tensor plan for one Transformer block. Routed experts are
// excluded: their complete bundle has a separate, indivisible residency contract. Planning is
// valid for metadata-only catalogs, but loading a tile from one is forbidden.
[[nodiscard]] Result<LayerStreamingPlan> build_layer_streaming_plan(
    const LogicalTensorCatalog& catalog,
    std::uint32_t layer,
    ByteCount tile_limit_bytes = kDefaultLayerStreamingTileBytes,
    std::size_t max_tensors = 4'096U,
    std::size_t max_tiles = 65'536U);

struct TensorTileLoadOptions {
    ByteCount max_tile_bytes{kDefaultLayerStreamingTileBytes};
    bool measure_process_io{true};
    bool measure_device_io{false};
    std::function<bool()> cancelled;
};

struct TensorTileLoadTelemetry {
    ByteCount logical_bytes{0U};
    double elapsed_ms{0.0};
    std::optional<io::ProcessIoDelta> process_io;
    std::optional<io::DeviceIoDelta> device_io;
};

class LoadedTensorTile {
public:
    LoadedTensorTile(const LoadedTensorTile&) = delete;
    LoadedTensorTile& operator=(const LoadedTensorTile&) = delete;
    LoadedTensorTile(LoadedTensorTile&&) noexcept = default;
    LoadedTensorTile& operator=(LoadedTensorTile&&) = delete;

    [[nodiscard]] const LayerTensorTile& identity() const noexcept { return identity_; }
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return buffer_ == nullptr ? std::span<const std::byte>{} : buffer_->bytes();
    }
    [[nodiscard]] std::shared_ptr<const ResidentByteBuffer> retain_buffer() const noexcept {
        return buffer_;
    }
    [[nodiscard]] const TensorTileLoadTelemetry& telemetry() const noexcept { return telemetry_; }

private:
    friend Result<LoadedTensorTile> load_tensor_tile(const LogicalTensorCatalog&,
                                                     const LayerTensorTile&,
                                                     ResidencyObject&,
                                                     safety::ResourceGovernor&,
                                                     const TensorTileLoadOptions&);
    LoadedTensorTile(ResidencyUse lease,
                     LayerTensorTile identity,
                     std::vector<std::byte> bytes,
                     TensorTileLoadTelemetry telemetry);

    LayerTensorTile identity_;
    std::shared_ptr<const ResidentByteBuffer> buffer_;
    TensorTileLoadTelemetry telemetry_;
};

[[nodiscard]] Result<LoadedTensorTile> load_tensor_tile(
    const LogicalTensorCatalog& catalog,
    const LayerTensorTile& tile,
    ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const TensorTileLoadOptions& options = {});

}  // namespace vastloom::model
