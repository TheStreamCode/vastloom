#include "vastloom/model/bound_range_loader.hpp"

#include <chrono>
#include <exception>
#include <limits>
#include <new>
#include <utility>

#include "vastloom/io/bounded_io_pool.hpp"
#include "vastloom/util/sha256.hpp"

namespace vastloom::model {
namespace {

Result<bool> cancellation_requested(const std::function<bool()>& cancelled) {
    if (!cancelled) return false;
    try {
        return cancelled();
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "Bound-range cancellation callback failed",
                          exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "Bound-range cancellation callback failed");
    }
}

Result<void> validate_identity(const BoundRangeIdentity& identity,
                               ByteCount max_range_bytes) {
    if (identity.logical_id.empty() || identity.source.shard.empty() ||
        !util::is_sha256_hex(identity.sha256) ||
        identity.source.bytes == 0U || max_range_bytes == 0U ||
        max_range_bytes > kMaximumBoundRangeLoadBytes ||
        identity.source.bytes > max_range_bytes ||
        identity.source.bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max()) ||
        identity.source.offset >
            std::numeric_limits<ByteCount>::max() - identity.source.bytes) {
        return make_error(ErrorCode::invalid_argument,
                          "Bound range identity or load bound is invalid",
                          identity.logical_id);
    }
    std::error_code error;
    const auto status = std::filesystem::symlink_status(identity.source.shard, error);
    if (error || status.type() == std::filesystem::file_type::not_found) {
        return make_error(ErrorCode::not_found,
                          "Bound range payload is missing",
                          identity.source.shard.string());
    }
    if (std::filesystem::is_symlink(status)) {
        return make_error(ErrorCode::unsafe_path,
                          "Bound range payload must not be a symlink",
                          identity.source.shard.string());
    }
    if (!std::filesystem::is_regular_file(status)) {
        return make_error(ErrorCode::integrity_error,
                          "Bound range payload is not a regular file",
                          identity.source.shard.string());
    }
    const auto file_bytes = std::filesystem::file_size(identity.source.shard, error);
    if (error || identity.source.offset > file_bytes ||
        identity.source.bytes > file_bytes - identity.source.offset) {
        return make_error(ErrorCode::integrity_error,
                          "Bound range payload is shorter than its verified identity",
                          identity.source.shard.string());
    }
    return {};
}

}  // namespace

LoadedBoundRange::LoadedBoundRange(ResidencyUse lease,
                                   BoundRangeIdentity identity,
                                   std::vector<std::byte> bytes,
                                   BoundRangeLoadTelemetry telemetry)
    : identity_(std::move(identity)),
      buffer_(std::shared_ptr<const ResidentByteBuffer>(
          new ResidentByteBuffer(std::move(lease), std::move(bytes)))),
      telemetry_(std::move(telemetry)) {}

Result<LoadedBoundRange> load_bound_range(
    const BoundRangeIdentity& identity,
    ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const BoundRangeLoadOptions& options) {
    auto valid = validate_identity(identity, options.max_range_bytes);
    if (!valid) return valid.error();
    auto snapshot = residency.snapshot();
    if (!snapshot) return snapshot.error();
    if (snapshot.value().logical_id != identity.logical_id ||
        snapshot.value().object_bytes != identity.source.bytes ||
        snapshot.value().state != ResidencyState::nvme) {
        return make_error(ErrorCode::integrity_error,
                          "Bound range residency does not match its immutable identity",
                          identity.logical_id);
    }
    auto cancelled = cancellation_requested(options.cancelled);
    if (!cancelled) return cancelled.error();
    if (cancelled.value()) {
        return make_error(ErrorCode::resource_exhausted,
                          "Bound range load cancelled safely",
                          identity.logical_id);
    }
    auto begun = residency.begin_read(governor, identity.source.bytes);
    if (!begun) return begun.error();
    auto transfer = residency.acquire_transfer();
    if (!transfer) {
        auto failed = residency.fail(transfer.error().message);
        return failed ? Result<LoadedBoundRange>(transfer.error())
                      : Result<LoadedBoundRange>(failed.error());
    }

    const auto started = std::chrono::steady_clock::now();
    std::optional<io::ProcessIoSnapshot> process_before;
    std::optional<io::DeviceIoSnapshot> device_before;
    if (options.measure_process_io) {
        auto sampled = io::sample_process_io();
        if (sampled) process_before = sampled.take_value();
    }
    if (options.measure_device_io) {
        auto sampled = io::sample_device_io(identity.source.shard);
        if (sampled) device_before = sampled.take_value();
    }

    auto loaded = [&]() -> Result<LoadedBoundRange> {
        try {
            std::vector<std::byte> bytes(static_cast<std::size_t>(identity.source.bytes));
            auto read = io::read_buffered_range_into(
                identity.source.shard, identity.source.offset, bytes);
            if (!read) return read.error();
            util::Sha256 hasher;
            hasher.update(bytes);
            if (hasher.finalize_hex() != identity.sha256) {
                return make_error(ErrorCode::integrity_error,
                                  "Bound range payload changed after artifact verification",
                                  identity.logical_id);
            }
            auto after_cancel = cancellation_requested(options.cancelled);
            if (!after_cancel) return after_cancel.error();
            if (after_cancel.value()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "Bound range load cancelled safely",
                                  identity.logical_id);
            }
            auto published = residency.publish_ram_ready();
            if (!published) return published.error();
            auto lease = residency.acquire_use();
            if (!lease) return lease.error();
            BoundRangeLoadTelemetry telemetry;
            telemetry.logical_bytes = identity.source.bytes;
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
                auto after = io::sample_device_io(identity.source.shard);
                if (after) {
                    auto delta = io::device_io_delta(*device_before, after.value());
                    if (delta) telemetry.device_io = delta.take_value();
                }
            }
            return LoadedBoundRange(lease.take_value(), identity, std::move(bytes),
                                    std::move(telemetry));
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to allocate the bounded range RAM buffer",
                              identity.logical_id);
        } catch (const std::exception& exception) {
            return make_error(ErrorCode::internal_error,
                              "Unexpected bound range load failure",
                              exception.what());
        } catch (...) {
            return make_error(ErrorCode::internal_error,
                              "Unexpected bound range load failure",
                              identity.logical_id);
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
