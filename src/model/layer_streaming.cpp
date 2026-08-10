#include "vastloom/model/layer_streaming.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <new>
#include <utility>

#include "vastloom/io/bounded_io_pool.hpp"

namespace vastloom::model {
namespace {

Result<bool> cancellation_requested(const std::function<bool()>& cancelled) {
    if (!cancelled) return false;
    try {
        return cancelled();
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "Tensor-tile cancellation callback failed",
                          exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "Tensor-tile cancellation callback failed");
    }
}

Result<void> validate_tile_identity(const LogicalTensorCatalog& catalog,
                                    const LayerTensorTile& tile,
                                    ByteCount max_tile_bytes) {
    if (catalog.model_id.empty() || catalog.metadata_only || tile.tensor_index >= catalog.tensors.size() ||
        tile.source.shard.empty() || tile.source.bytes == 0U || max_tile_bytes == 0U ||
        tile.source.bytes > max_tile_bytes ||
        tile.source.offset > std::numeric_limits<ByteCount>::max() - tile.source.bytes) {
        return make_error(ErrorCode::invalid_argument,
                          catalog.metadata_only
                              ? "Metadata-only catalogs cannot load tensor payload tiles"
                              : "Tensor tile identity or load bound is invalid",
                          catalog.model_id);
    }
    const auto& tensor = catalog.tensors[tile.tensor_index];
    if (tensor.role == TensorRole::routed_expert || !tensor.layer.has_value() ||
        *tensor.layer != tile.layer || tensor.role != tile.role ||
        tensor.source.shard != tile.source.shard || tensor.source.bytes == 0U ||
        tensor.source.offset > std::numeric_limits<ByteCount>::max() - tensor.source.bytes ||
        tile.tensor_offset >= tensor.source.bytes ||
        tile.source.offset != tensor.source.offset + tile.tensor_offset ||
        tile.source.bytes > tensor.source.bytes - tile.tensor_offset ||
        tile.final_tile != (tile.source.bytes == tensor.source.bytes - tile.tensor_offset)) {
        return make_error(ErrorCode::integrity_error,
                          "Tensor tile no longer matches its immutable catalog identity",
                          tensor.name);
    }
    return {};
}

}  // namespace

Result<LayerStreamingPlan> build_layer_streaming_plan(const LogicalTensorCatalog& catalog,
                                                       std::uint32_t layer,
                                                       ByteCount tile_limit_bytes,
                                                       std::size_t max_tensors,
                                                       std::size_t max_tiles) {
    if (catalog.model_id.empty() || catalog.tensors.empty() || tile_limit_bytes == 0U ||
        tile_limit_bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max()) ||
        max_tensors == 0U || max_tiles == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Layer streaming plan requires a catalog and finite non-zero bounds",
                          catalog.model_id);
    }
    try {
        LayerStreamingPlan plan;
        plan.model_id = catalog.model_id;
        plan.layer = layer;
        plan.tile_limit_bytes = tile_limit_bytes;
        for (std::size_t tensor_index = 0U; tensor_index < catalog.tensors.size(); ++tensor_index) {
            const auto& tensor = catalog.tensors[tensor_index];
            if (tensor.role == TensorRole::routed_expert || !tensor.layer.has_value() ||
                *tensor.layer != layer) {
                continue;
            }
            if (plan.tensor_count == max_tensors) {
                return make_error(ErrorCode::resource_exhausted,
                                  "Layer trunk tensor count exceeds the configured bound",
                                  std::to_string(layer));
            }
            if (tensor.source.shard.empty() || tensor.source.bytes == 0U ||
                tensor.source.offset > std::numeric_limits<ByteCount>::max() - tensor.source.bytes ||
                tensor.source.bytes > std::numeric_limits<ByteCount>::max() - plan.trunk_tensor_bytes) {
                return make_error(ErrorCode::invalid_format,
                                  "Layer trunk tensor has an invalid source range",
                                  tensor.name);
            }
            ++plan.tensor_count;
            plan.trunk_tensor_bytes += tensor.source.bytes;
            plan.maximum_tensor_bytes = std::max(plan.maximum_tensor_bytes, tensor.source.bytes);
            ByteCount tensor_offset = 0U;
            while (tensor_offset < tensor.source.bytes) {
                if (plan.tiles.size() == max_tiles) {
                    return make_error(ErrorCode::resource_exhausted,
                                      "Layer streaming tile count exceeds the configured bound",
                                      std::to_string(layer));
                }
                const ByteCount bytes = std::min(tile_limit_bytes, tensor.source.bytes - tensor_offset);
                if (tensor_offset > std::numeric_limits<ByteCount>::max() - tensor.source.offset) {
                    return make_error(ErrorCode::invalid_format,
                                      "Layer streaming tile offset overflows",
                                      tensor.name);
                }
                plan.tiles.push_back(LayerTensorTile{
                    tensor_index,
                    layer,
                    tensor.role,
                    tensor_offset,
                    TensorSourceRange{tensor.source.shard, tensor.source.offset + tensor_offset, bytes},
                    bytes == tensor.source.bytes - tensor_offset,
                });
                plan.maximum_tile_bytes = std::max(plan.maximum_tile_bytes, bytes);
                tensor_offset += bytes;
            }
        }
        if (plan.tensor_count == 0U || plan.tiles.empty() || plan.trunk_tensor_bytes == 0U) {
            return make_error(ErrorCode::not_found,
                              "Catalog has no non-routed trunk tensors for the requested layer",
                              std::to_string(layer));
        }
        return plan;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded layer streaming plan",
                          std::to_string(layer));
    }
}

LoadedTensorTile::LoadedTensorTile(ResidencyUse lease,
                                   LayerTensorTile identity,
                                   std::vector<std::byte> bytes,
                                   TensorTileLoadTelemetry telemetry)
    : identity_(std::move(identity)),
      buffer_(std::shared_ptr<const ResidentByteBuffer>(
          new ResidentByteBuffer(std::move(lease), std::move(bytes)))),
      telemetry_(std::move(telemetry)) {}

Result<LoadedTensorTile> load_tensor_tile(const LogicalTensorCatalog& catalog,
                                          const LayerTensorTile& tile,
                                          ResidencyObject& residency,
                                          safety::ResourceGovernor& governor,
                                          const TensorTileLoadOptions& options) {
    auto valid = validate_tile_identity(catalog, tile, options.max_tile_bytes);
    if (!valid) return valid.error();
    auto snapshot = residency.snapshot();
    if (!snapshot) return snapshot.error();
    if (snapshot.value().object_bytes != tile.source.bytes ||
        snapshot.value().state != ResidencyState::nvme) {
        return make_error(ErrorCode::integrity_error,
                          "Tensor tile residency does not match the planned payload",
                          snapshot.value().logical_id);
    }
    auto cancelled = cancellation_requested(options.cancelled);
    if (!cancelled) return cancelled.error();
    if (cancelled.value()) {
        return make_error(ErrorCode::resource_exhausted, "Tensor tile load cancelled safely");
    }
    auto begun = residency.begin_read(governor, tile.source.bytes);
    if (!begun) return begun.error();
    auto transfer = residency.acquire_transfer();
    if (!transfer) {
        auto failed = residency.fail(transfer.error().message);
        return failed ? Result<LoadedTensorTile>(transfer.error())
                      : Result<LoadedTensorTile>(failed.error());
    }

    const auto started = std::chrono::steady_clock::now();
    std::optional<io::ProcessIoSnapshot> process_before;
    std::optional<io::DeviceIoSnapshot> device_before;
    if (options.measure_process_io) {
        auto sampled = io::sample_process_io();
        if (sampled) process_before = sampled.take_value();
    }
    if (options.measure_device_io) {
        auto sampled = io::sample_device_io(tile.source.shard);
        if (sampled) device_before = sampled.take_value();
    }

    auto loaded = [&]() -> Result<LoadedTensorTile> {
        try {
            std::vector<std::byte> bytes(static_cast<std::size_t>(tile.source.bytes));
            auto read = io::read_buffered_range_into(tile.source.shard, tile.source.offset, bytes);
            if (!read) return read.error();
            auto after_cancel = cancellation_requested(options.cancelled);
            if (!after_cancel) return after_cancel.error();
            if (after_cancel.value()) {
                return make_error(ErrorCode::resource_exhausted, "Tensor tile load cancelled safely");
            }
            auto published = residency.publish_ram_ready();
            if (!published) return published.error();
            auto lease = residency.acquire_use();
            if (!lease) return lease.error();
            TensorTileLoadTelemetry telemetry;
            telemetry.logical_bytes = tile.source.bytes;
            telemetry.elapsed_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
            if (process_before.has_value()) {
                auto after = io::sample_process_io();
                if (after) {
                    auto delta = io::process_io_delta(*process_before, after.value());
                    if (delta) telemetry.process_io = delta.take_value();
                }
            }
            if (device_before.has_value()) {
                auto after = io::sample_device_io(tile.source.shard);
                if (after) {
                    auto delta = io::device_io_delta(*device_before, after.value());
                    if (delta) telemetry.device_io = delta.take_value();
                }
            }
            return LoadedTensorTile(lease.take_value(), tile, std::move(bytes), std::move(telemetry));
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to allocate the bounded tensor tile buffer");
        } catch (const std::exception& exception) {
            return make_error(ErrorCode::internal_error,
                              "Unexpected tensor tile load failure",
                              exception.what());
        } catch (...) {
            return make_error(ErrorCode::internal_error,
                              "Unexpected tensor tile load failure");
        }
    }();
    transfer.value().release();
    if (!loaded) {
        auto failed = residency.fail(loaded.error().message);
        if (!failed) return failed.error();
    }
    return loaded;
}

}  // namespace vastloom::model
