#include "vastloom/model/expert_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string_view>
#include <utility>

#include "vastloom/io/bounded_io_pool.hpp"

namespace vastloom::model {
namespace {

Result<bool> cancellation_requested(const std::function<bool()>& cancelled) {
    if (!cancelled) return false;
    try {
        return cancelled();
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error, "Expert-load cancellation callback failed", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error, "Expert-load cancellation callback failed");
    }
}

Result<ByteCount> validate_bundle(const ExpertBundle& bundle, const ExpertLoadOptions& options) {
    if (bundle.bytes == 0U || bundle.tensor_indices.empty() || bundle.read_ranges.empty() ||
        bundle.tensor_slices.size() != bundle.tensor_indices.size() ||
        options.max_expert_bytes == 0U || options.max_range_bytes == 0U || options.max_ranges == 0U) {
        return make_error(ErrorCode::invalid_argument, "Expert bundle or load limits are empty");
    }
    if (bundle.bytes > options.max_expert_bytes || bundle.read_ranges.size() > options.max_ranges) {
        return make_error(ErrorCode::resource_exhausted, "Expert bundle exceeds configured load limits");
    }
    ByteCount total = 0U;
    ByteCount largest = 0U;
    for (std::size_t index = 0U; index < bundle.read_ranges.size(); ++index) {
        const auto& range = bundle.read_ranges[index];
        if (range.shard.empty() || range.bytes == 0U ||
            range.offset > std::numeric_limits<ByteCount>::max() - range.bytes ||
            range.bytes > std::numeric_limits<ByteCount>::max() - total) {
            return make_error(ErrorCode::invalid_format, "Expert bundle contains an invalid source range");
        }
        if (range.bytes > options.max_range_bytes) {
            return make_error(ErrorCode::resource_exhausted,
                              "Expert source range exceeds the configured read limit");
        }
        if (index != 0U) {
            const auto& previous = bundle.read_ranges[index - 1U];
            if (range.shard < previous.shard ||
                (range.shard == previous.shard &&
                 range.offset <= previous.offset + previous.bytes)) {
                return make_error(ErrorCode::invalid_format,
                                  "Expert bundle ranges are overlapping, adjacent or not canonical");
            }
        }
        total += range.bytes;
        largest = std::max(largest, range.bytes);
    }
    if (total != bundle.bytes || total > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
        return make_error(ErrorCode::integrity_error, "Expert bundle ranges do not match its byte identity");
    }
    ByteCount slice_total = 0U;
    for (std::size_t index = 0U; index < bundle.tensor_slices.size(); ++index) {
        const auto& slice = bundle.tensor_slices[index];
        if (slice.bytes == 0U || slice.packed_offset > bundle.bytes ||
            slice.bytes > bundle.bytes - slice.packed_offset ||
            std::find(bundle.tensor_indices.begin(), bundle.tensor_indices.end(), slice.tensor_index) ==
                bundle.tensor_indices.end() ||
            slice.bytes > std::numeric_limits<ByteCount>::max() - slice_total) {
            return make_error(ErrorCode::invalid_format, "Expert bundle contains an invalid packed tensor slice");
        }
        for (std::size_t previous = 0U; previous < index; ++previous) {
            const auto& other = bundle.tensor_slices[previous];
            const bool overlaps = slice.packed_offset < other.packed_offset + other.bytes &&
                                  other.packed_offset < slice.packed_offset + slice.bytes;
            if (slice.tensor_index == other.tensor_index || overlaps) {
                return make_error(ErrorCode::invalid_format,
                                  "Expert bundle packed tensor slices overlap or duplicate an identity");
            }
        }
        slice_total += slice.bytes;
    }
    if (slice_total != bundle.bytes) {
        return make_error(ErrorCode::integrity_error,
                          "Expert bundle packed tensor slices do not cover its complete payload");
    }
    return largest;
}

}  // namespace

LoadedExpert::LoadedExpert(ResidencyUse lease,
                           std::vector<std::byte> bytes,
                           std::vector<ExpertTensorSlice> tensor_slices,
                           ExpertLoadTelemetry telemetry)
    : lease_(std::move(lease)),
      bytes_(std::move(bytes)),
      tensor_slices_(std::move(tensor_slices)),
      telemetry_(std::move(telemetry)) {}

Result<std::span<const std::byte>> LoadedExpert::tensor(std::size_t catalog_tensor_index) const {
    const auto iterator = std::find_if(tensor_slices_.begin(), tensor_slices_.end(),
                                       [&](const ExpertTensorSlice& slice) {
                                           return slice.tensor_index == catalog_tensor_index;
                                       });
    if (iterator == tensor_slices_.end() || iterator->packed_offset > bytes_.size() ||
        iterator->bytes > bytes_.size() - iterator->packed_offset) {
        return make_error(ErrorCode::not_found, "Packed expert tensor is unavailable");
    }
    return std::span<const std::byte>(bytes_.data() + static_cast<std::size_t>(iterator->packed_offset),
                                      static_cast<std::size_t>(iterator->bytes));
}

Result<LoadedExpert> load_expert_bundle(const ExpertBundle& bundle,
                                        ResidencyObject& residency,
                                        safety::ResourceGovernor& governor,
                                        const ExpertLoadOptions& options) {
    auto largest = validate_bundle(bundle, options);
    if (!largest) return largest.error();
    auto begun = residency.begin_read(governor, largest.value());
    if (!begun) return begun.error();

    const auto started = std::chrono::steady_clock::now();
    std::optional<io::ProcessIoSnapshot> process_before;
    std::optional<io::DeviceIoSnapshot> device_before;
    if (options.measure_process_io) {
        auto sampled = io::sample_process_io();
        if (sampled) process_before = sampled.take_value();
    }
    if (options.measure_device_io) {
        auto sampled = io::sample_device_io(bundle.read_ranges.front().shard);
        if (sampled) device_before = sampled.take_value();
    }

    auto loaded = [&]() -> Result<LoadedExpert> {
        try {
            std::vector<std::byte> destination(static_cast<std::size_t>(bundle.bytes));
            std::vector<std::byte> scratch(static_cast<std::size_t>(largest.value()));
            std::size_t cursor = 0U;
            for (const auto& range : bundle.read_ranges) {
                auto cancelled = cancellation_requested(options.cancelled);
                if (!cancelled) return cancelled.error();
                if (cancelled.value()) {
                    return make_error(ErrorCode::resource_exhausted, "Expert load cancelled safely");
                }
                const auto range_size = static_cast<std::size_t>(range.bytes);
                auto read = io::read_buffered_range_into(
                    range.shard, range.offset, std::span<std::byte>(scratch.data(), range_size));
                if (!read) return read.error();
                std::memcpy(destination.data() + cursor, scratch.data(), range_size);
                cursor += range_size;
            }
            auto cancelled = cancellation_requested(options.cancelled);
            if (!cancelled) return cancelled.error();
            if (cancelled.value()) {
                return make_error(ErrorCode::resource_exhausted, "Expert load cancelled safely");
            }
            if (cursor != destination.size()) {
                return make_error(ErrorCode::integrity_error, "Expert load did not fill its destination buffer");
            }
            std::vector<std::byte>().swap(scratch);
            auto published = residency.publish_ram_ready();
            if (!published) return published.error();
            auto lease = residency.acquire_use();
            if (!lease) return lease.error();

            ExpertLoadTelemetry telemetry;
            telemetry.logical_bytes = bundle.bytes;
            telemetry.range_count = bundle.read_ranges.size();
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
                auto after = io::sample_device_io(bundle.read_ranges.front().shard);
                if (after) {
                    auto delta = io::device_io_delta(*device_before, after.value());
                    if (delta) telemetry.device_io = delta.take_value();
                }
            }
            return LoadedExpert(lease.take_value(), std::move(destination), bundle.tensor_slices,
                                std::move(telemetry));
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted, "Unable to allocate bounded expert-load buffers");
        }
    }();

    if (!loaded) {
        auto failed = residency.fail(loaded.error().message);
        if (!failed) return failed.error();
    }
    return loaded;
}

}  // namespace vastloom::model
