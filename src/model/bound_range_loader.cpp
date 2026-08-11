#include "vastloom/model/bound_range_loader.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <new>
#include <set>
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

struct BoundRangeSetPreflight {
    ByteCount largest_range_bytes{0U};
    std::vector<BoundRangeSetSlice> slices;
};

Result<BoundRangeSetPreflight> validate_set_identity(
    const BoundRangeSetIdentity& identity,
    const BoundRangeSetLoadOptions& options) {
    if (identity.logical_id.empty() || identity.total_bytes == 0U ||
        identity.ranges.empty() || options.max_total_bytes == 0U ||
        options.max_total_bytes > kMaximumBoundRangeSetLoadBytes ||
        identity.total_bytes > options.max_total_bytes ||
        identity.total_bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max()) ||
        options.max_range_bytes == 0U ||
        options.max_range_bytes > kMaximumBoundRangeLoadBytes ||
        options.max_ranges == 0U ||
        options.max_ranges > kMaximumBoundRangeSetRanges ||
        identity.ranges.size() > options.max_ranges) {
        return make_error(ErrorCode::invalid_argument,
                          "Bound range set identity or load bounds are invalid",
                          identity.logical_id);
    }
    try {
        BoundRangeSetPreflight preflight;
        preflight.slices.reserve(identity.ranges.size());
        std::set<std::string, std::less<>> logical_ids;
        ByteCount total = 0U;
        for (const auto& range : identity.ranges) {
            auto valid = validate_identity(range, options.max_range_bytes);
            if (!valid) return valid.error();
            if (!logical_ids.emplace(range.logical_id).second) {
                return make_error(ErrorCode::integrity_error,
                                  "Bound range set contains a duplicate immutable identity",
                                  range.logical_id);
            }
            if (range.source.bytes > std::numeric_limits<ByteCount>::max() - total) {
                return make_error(ErrorCode::invalid_format,
                                  "Bound range set byte total overflows",
                                  identity.logical_id);
            }
            preflight.slices.push_back(
                BoundRangeSetSlice{range.logical_id, total, range.source.bytes});
            total += range.source.bytes;
            preflight.largest_range_bytes =
                std::max(preflight.largest_range_bytes, range.source.bytes);
        }
        if (total != identity.total_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "Bound range set ranges do not match its aggregate byte identity",
                              identity.logical_id);
        }
        return preflight;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate bound range set preflight state",
                          identity.logical_id);
    }
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

LoadedBoundRangeSet::LoadedBoundRangeSet(
    ResidencyUse lease,
    BoundRangeSetIdentity identity,
    std::vector<BoundRangeSetSlice> slices,
    std::vector<std::byte> bytes,
    BoundRangeSetLoadTelemetry telemetry)
    : identity_(std::move(identity)),
      slices_(std::move(slices)),
      buffer_(std::shared_ptr<const ResidentByteBuffer>(
          new ResidentByteBuffer(std::move(lease), std::move(bytes)))),
      telemetry_(std::move(telemetry)) {}

Result<std::span<const std::byte>> LoadedBoundRangeSet::range(std::size_t index) const {
    const auto payload = bytes();
    if (index >= slices_.size() ||
        slices_[index].packed_offset > payload.size() ||
        slices_[index].bytes > payload.size() - slices_[index].packed_offset) {
        return make_error(ErrorCode::not_found,
                          "Packed bound range is unavailable",
                          identity_.logical_id);
    }
    return payload.subspan(static_cast<std::size_t>(slices_[index].packed_offset),
                           static_cast<std::size_t>(slices_[index].bytes));
}

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

Result<LoadedBoundRangeSet> load_bound_range_set(
    const BoundRangeSetIdentity& identity,
    ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const BoundRangeSetLoadOptions& options) {
    auto preflight = validate_set_identity(identity, options);
    if (!preflight) return preflight.error();
    auto snapshot = residency.snapshot();
    if (!snapshot) return snapshot.error();
    if (snapshot.value().logical_id != identity.logical_id ||
        snapshot.value().object_bytes != identity.total_bytes ||
        snapshot.value().state != ResidencyState::nvme) {
        return make_error(ErrorCode::integrity_error,
                          "Bound range set residency does not match its immutable identity",
                          identity.logical_id);
    }
    auto cancelled = cancellation_requested(options.cancelled);
    if (!cancelled) return cancelled.error();
    if (cancelled.value()) {
        return make_error(ErrorCode::resource_exhausted,
                          "Bound range set load cancelled safely",
                          identity.logical_id);
    }
    auto begun = residency.begin_read(governor, preflight.value().largest_range_bytes);
    if (!begun) return begun.error();
    auto transfer = residency.acquire_transfer();
    if (!transfer) {
        auto failed = residency.fail(transfer.error().message);
        return failed ? Result<LoadedBoundRangeSet>(transfer.error())
                      : Result<LoadedBoundRangeSet>(failed.error());
    }

    const auto started = std::chrono::steady_clock::now();
    std::optional<io::ProcessIoSnapshot> process_before;
    std::optional<io::DeviceIoSnapshot> device_before;
    if (options.measure_process_io) {
        auto sampled = io::sample_process_io();
        if (sampled) process_before = sampled.take_value();
    }
    if (options.measure_device_io) {
        auto sampled = io::sample_device_io(identity.ranges.front().source.shard);
        if (sampled) device_before = sampled.take_value();
    }

    auto loaded = [&]() -> Result<LoadedBoundRangeSet> {
        try {
            std::vector<std::byte> bytes(static_cast<std::size_t>(identity.total_bytes));
            for (std::size_t index = 0U; index < identity.ranges.size(); ++index) {
                auto before_read = cancellation_requested(options.cancelled);
                if (!before_read) return before_read.error();
                if (before_read.value()) {
                    return make_error(ErrorCode::resource_exhausted,
                                      "Bound range set load cancelled safely",
                                      identity.ranges[index].logical_id);
                }
                const auto& range = identity.ranges[index];
                const auto& slice = preflight.value().slices[index];
                auto destination = std::span<std::byte>(bytes).subspan(
                    static_cast<std::size_t>(slice.packed_offset),
                    static_cast<std::size_t>(slice.bytes));
                auto read = io::read_buffered_range_into(
                    range.source.shard, range.source.offset, destination);
                if (!read) return read.error();
                util::Sha256 hasher;
                hasher.update(destination);
                if (hasher.finalize_hex() != range.sha256) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "Bound range set payload changed after artifact verification",
                        range.logical_id);
                }
            }
            auto after_reads = cancellation_requested(options.cancelled);
            if (!after_reads) return after_reads.error();
            if (after_reads.value()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "Bound range set load cancelled safely",
                                  identity.logical_id);
            }
            auto published = residency.publish_ram_ready();
            if (!published) return published.error();
            auto lease = residency.acquire_use();
            if (!lease) return lease.error();
            BoundRangeSetLoadTelemetry telemetry;
            telemetry.logical_bytes = identity.total_bytes;
            telemetry.range_count = identity.ranges.size();
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
                auto after = io::sample_device_io(identity.ranges.front().source.shard);
                if (after) {
                    auto delta = io::device_io_delta(*device_before, after.value());
                    if (delta) telemetry.device_io = delta.take_value();
                }
            }
            return LoadedBoundRangeSet(
                lease.take_value(), identity, std::move(preflight.value().slices),
                std::move(bytes), std::move(telemetry));
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to allocate the bounded range set RAM buffer",
                              identity.logical_id);
        } catch (const std::exception& exception) {
            return make_error(ErrorCode::internal_error,
                              "Unexpected bound range set load failure",
                              exception.what());
        } catch (...) {
            return make_error(ErrorCode::internal_error,
                              "Unexpected bound range set load failure",
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
