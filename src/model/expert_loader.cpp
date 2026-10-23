#include "vastloom/model/expert_loader.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>
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

Result<LoadedExpert> load_prepared_expert_bundle(const ExpertBundle& bundle,
                                                  ResidencyObject& residency,
                                                  ResidencyTransfer& transfer,
                                                  const ExpertLoadOptions& options,
                                                  ByteCount largest_range) {
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
            (void)largest_range;
            std::vector<std::byte> destination(static_cast<std::size_t>(bundle.bytes));
            std::size_t cursor = 0U;
            std::size_t read_calls = 0U;
            for (const auto& range : bundle.read_ranges) {
                auto cancelled = cancellation_requested(options.cancelled);
                if (!cancelled) return cancelled.error();
                if (cancelled.value()) {
                    return make_error(ErrorCode::resource_exhausted, "Expert load cancelled safely");
                }
                const auto range_size = static_cast<std::size_t>(range.bytes);
                // Read straight into the published buffer. A bounce copy through a second
                // scratch allocation was doubling every expert byte in RAM (about 2.37 GB per
                // token) without changing the Governor staging reservation, which still bounds
                // the largest range. Buffered is the default because unbuffered reads of the same
                // small recurring ranges were three times slower when the page cache held them;
                // `io_cache_mode` is the explicit bypass for cold/large or coalesced records.
                const auto dest = std::span<std::byte>(destination).subspan(cursor, range_size);
                const bool use_session = options.io_cache_mode == io::IoCacheMode::buffered &&
                                         options.read_session != nullptr;
                auto read = use_session
                                ? options.read_session->read_into(range.shard, range.offset, dest)
                                : io::read_range_into(range.shard, range.offset, dest,
                                                      options.io_cache_mode);
                if (!read) return read.error();
                ++read_calls;
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
            auto published = residency.publish_ram_ready();
            if (!published) return published.error();
            auto lease = residency.acquire_use();
            if (!lease) return lease.error();

            ExpertLoadTelemetry telemetry;
            const auto completed = std::chrono::steady_clock::now();
            telemetry.logical_bytes = bundle.bytes;
            telemetry.range_count = bundle.read_ranges.size();
            telemetry.read_calls = read_calls;
            telemetry.elapsed_ms = std::chrono::duration<double, std::milli>(
                                       completed - started)
                                       .count();
            telemetry.service_ms = telemetry.elapsed_ms;
            telemetry.total_ms = telemetry.elapsed_ms;
            telemetry.submitted_at = started;
            telemetry.service_started_at = started;
            telemetry.completed_at = completed;
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
        } catch (const std::exception& exception) {
            return make_error(ErrorCode::internal_error,
                              "Unexpected expert-load failure",
                              exception.what());
        } catch (...) {
            return make_error(ErrorCode::internal_error, "Unexpected expert-load failure");
        }
    }();

    transfer.release();
    if (!loaded) {
        auto failed = residency.fail(loaded.error().message);
        if (!failed) return failed.error();
    }
    return loaded;
}

LoadedExpert::LoadedExpert(ResidencyUse lease,
                           std::vector<std::byte> bytes,
                           std::vector<ExpertTensorSlice> tensor_slices,
                           ExpertLoadTelemetry telemetry)
    : buffer_(std::shared_ptr<const ResidentByteBuffer>(
          new ResidentByteBuffer(std::move(lease), std::move(bytes)))),
      tensor_slices_(std::move(tensor_slices)),
      telemetry_(std::move(telemetry)) {}

Result<std::span<const std::byte>> LoadedExpert::tensor(std::size_t catalog_tensor_index) const {
    const auto iterator = std::find_if(tensor_slices_.begin(), tensor_slices_.end(),
                                       [&](const ExpertTensorSlice& slice) {
                                           return slice.tensor_index == catalog_tensor_index;
                                       });
    const auto payload = bytes();
    if (iterator == tensor_slices_.end() || iterator->packed_offset > payload.size() ||
        iterator->bytes > payload.size() - iterator->packed_offset) {
        return make_error(ErrorCode::not_found, "Packed expert tensor is unavailable");
    }
    return std::span<const std::byte>(payload.data() + static_cast<std::size_t>(iterator->packed_offset),
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
    auto transfer = residency.acquire_transfer();
    if (!transfer) {
        auto failed = residency.fail(transfer.error().message);
        return failed ? Result<LoadedExpert>(transfer.error()) : Result<LoadedExpert>(failed.error());
    }
    return load_prepared_expert_bundle(bundle, residency, transfer.value(), options, largest.value());
}

struct BoundedExpertLoadPool::Task {
    Task(ExpertBundle source_bundle,
         ResidencyObject source_residency,
         ResidencyTransfer source_transfer,
         ExpertLoadOptions source_options,
         std::string source_tag)
        : bundle(std::move(source_bundle)),
          residency(std::move(source_residency)),
          transfer(std::move(source_transfer)),
          options(std::move(source_options)),
          tag(std::move(source_tag)) {}

    ExpertBundle bundle;
    ResidencyObject residency;
    ResidencyTransfer transfer;
    ExpertLoadOptions options;
    std::string tag;
    ByteCount largest_range{0U};
    std::size_t sequence{0U};
    std::chrono::steady_clock::time_point submitted;
    std::promise<Result<LoadedExpert>> promise;
};

BoundedExpertLoadPool::BoundedExpertLoadPool(ExpertLoadPoolOptions options,
                                             safety::ResourceGovernor& governor)
    : options_(std::move(options)), governor_(governor) {
    options_.workers = std::clamp(options_.workers, 1U, 4U);
    // A batch must be able to submit its whole selected set without waiting on itself, and a
    // prefill selects up to one bundle per token per slot. At 64 the clamp silently cut a caller
    // that the cache had already validated as legal, and the surplus submissions failed the batch
    // mid-layer. Every request is still reserved through the Governor before enqueue, so this
    // ceiling bounds the queue rather than the memory.
    options_.max_queued_requests = std::clamp<std::size_t>(options_.max_queued_requests, 1U, 256U);
    try {
        workers_.reserve(options_.workers);
        for (unsigned index = 0U; index < options_.workers; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    } catch (...) {
        {
            std::scoped_lock lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        workers_.clear();
        throw;
    }
}

BoundedExpertLoadPool::~BoundedExpertLoadPool() {
    shutdown();
}

Result<std::future<Result<LoadedExpert>>> BoundedExpertLoadPool::submit(
    const ExpertBundle& bundle,
    ResidencyObject& residency,
    std::string tag) {
    if (tag.empty()) return make_error(ErrorCode::invalid_argument, "Expert-load task tag is empty");
    auto largest = validate_bundle(bundle, options_.load);
    if (!largest) return largest.error();
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) return make_error(ErrorCode::internal_error, "Expert-load pool is shutting down", tag);
        if (queue_.size() >= options_.max_queued_requests) {
            return make_error(ErrorCode::resource_exhausted, "Expert-load queue is full", tag);
        }
    }

    auto begun = residency.begin_read(governor_, largest.value());
    if (!begun) return begun.error();
    auto transfer = residency.acquire_transfer();
    if (!transfer) {
        auto failed = residency.fail(transfer.error().message);
        return failed ? Result<std::future<Result<LoadedExpert>>>(transfer.error())
                      : Result<std::future<Result<LoadedExpert>>>(failed.error());
    }

    std::unique_ptr<Task> task;
    std::future<Result<LoadedExpert>> future;
    try {
        task = std::make_unique<Task>(bundle, residency, transfer.take_value(), options_.load, tag);
        task->largest_range = largest.value();
        task->submitted = std::chrono::steady_clock::now();
        future = task->promise.get_future();
    } catch (const std::bad_alloc&) {
        if (task != nullptr) task->transfer.release();
        else transfer.value().release();
        auto failed = residency.fail("Unable to allocate bounded expert-load task");
        return failed ? Result<std::future<Result<LoadedExpert>>>(make_error(
                            ErrorCode::resource_exhausted, "Unable to allocate bounded expert-load task", tag))
                      : Result<std::future<Result<LoadedExpert>>>(failed.error());
    } catch (const std::future_error& error) {
        if (task != nullptr) task->transfer.release();
        else transfer.value().release();
        auto failed = residency.fail("Unable to create expert-load future");
        return failed ? Result<std::future<Result<LoadedExpert>>>(make_error(
                            ErrorCode::internal_error, "Unable to create expert-load future", error.what()))
                      : Result<std::future<Result<LoadedExpert>>>(failed.error());
    } catch (const std::exception& error) {
        if (task != nullptr) task->transfer.release();
        else transfer.value().release();
        auto failed = residency.fail("Unable to create bounded expert-load task");
        return failed ? Result<std::future<Result<LoadedExpert>>>(make_error(
                            ErrorCode::internal_error,
                            "Unable to create bounded expert-load task",
                            error.what()))
                      : Result<std::future<Result<LoadedExpert>>>(failed.error());
    } catch (...) {
        if (task != nullptr) task->transfer.release();
        else transfer.value().release();
        auto failed = residency.fail("Unable to create bounded expert-load task");
        return failed ? Result<std::future<Result<LoadedExpert>>>(make_error(
                            ErrorCode::internal_error, "Unable to create bounded expert-load task", tag))
                      : Result<std::future<Result<LoadedExpert>>>(failed.error());
    }

    try {
        {
            std::scoped_lock lock(mutex_);
            if (stopping_ || queue_.size() >= options_.max_queued_requests) {
                task->transfer.release();
                auto failed = task->residency.fail(stopping_ ? "Expert-load pool is shutting down"
                                                             : "Expert-load queue became full");
                return failed ? Result<std::future<Result<LoadedExpert>>>(make_error(
                                    stopping_ ? ErrorCode::internal_error : ErrorCode::resource_exhausted,
                                    stopping_ ? "Expert-load pool is shutting down" : "Expert-load queue is full",
                                    tag))
                              : Result<std::future<Result<LoadedExpert>>>(failed.error());
            }
            task->sequence = next_sequence_++;
            queue_.push_back(std::move(task));
        }
    } catch (const std::bad_alloc&) {
        if (task != nullptr) task->transfer.release();
        auto failed = residency.fail("Unable to enqueue bounded expert-load task");
        return failed ? Result<std::future<Result<LoadedExpert>>>(make_error(
                            ErrorCode::resource_exhausted, "Unable to enqueue bounded expert-load task", tag))
                      : Result<std::future<Result<LoadedExpert>>>(failed.error());
    }
    condition_.notify_one();
    return future;
}

void BoundedExpertLoadPool::shutdown() noexcept {
    std::deque<std::unique_ptr<Task>> cancelled;
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) return;
        stopping_ = true;
        cancelled.swap(queue_);
    }
    for (auto& task : cancelled) {
        task->transfer.release();
        static_cast<void>(task->residency.fail("Expert-load pool stopped before task execution"));
        try {
            task->promise.set_value(make_error(ErrorCode::internal_error,
                                               "Expert-load pool stopped before task execution",
                                               task->tag));
        } catch (...) {
            // Destroying the promise below still wakes a waiter with a broken-promise error.
        }
    }
    condition_.notify_all();
    workers_.clear();
}

std::size_t BoundedExpertLoadPool::pending_requests() const {
    std::scoped_lock lock(mutex_);
    return queue_.size();
}

void BoundedExpertLoadPool::worker_loop() {
    while (true) {
        std::unique_ptr<Task> task;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) return;
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        const auto service_started = std::chrono::steady_clock::now();
        thread_local io::BufferedReadSession session;
        auto options = task->options;
        options.read_session = &session;
        auto loaded = load_prepared_expert_bundle(
            task->bundle, task->residency, task->transfer, options, task->largest_range);
        const auto completed = std::chrono::steady_clock::now();
        if (loaded) {
            auto& telemetry = loaded.value().telemetry_;
            telemetry.request_sequence = task->sequence;
            telemetry.queue_ms = std::chrono::duration<double, std::milli>(service_started - task->submitted).count();
            telemetry.service_ms = std::chrono::duration<double, std::milli>(completed - service_started).count();
            telemetry.total_ms = std::chrono::duration<double, std::milli>(completed - task->submitted).count();
            telemetry.submitted_at = task->submitted;
            telemetry.service_started_at = service_started;
            telemetry.completed_at = completed;
        }
        try {
            task->promise.set_value(std::move(loaded));
        } catch (...) {
            // Keep the bounded worker alive. Promise destruction reports a broken promise.
        }
    }
}

}  // namespace vastloom::model
