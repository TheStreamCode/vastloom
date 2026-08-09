#include "stargate/io/bounded_io_pool.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <new>
#include <span>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace stargate::io {
namespace {

Result<bool> cancellation_requested(const std::function<bool()>& cancelled) {
    if (!cancelled) return false;
    try {
        return cancelled();
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error, "I/O cancellation callback failed", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error, "I/O cancellation callback failed");
    }
}

Result<std::size_t> read_file_range(const std::filesystem::path& path,
                                    ByteCount offset,
                                    std::span<std::byte> destination,
                                    std::size_t requested) {
#if defined(_WIN32)
    class FileHandle {
    public:
        explicit FileHandle(HANDLE handle) : handle_(handle) {}
        FileHandle(const FileHandle&) = delete;
        FileHandle& operator=(const FileHandle&) = delete;
        ~FileHandle() {
            if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
        }
        [[nodiscard]] HANDLE get() const noexcept { return handle_; }

    private:
        HANDLE handle_{INVALID_HANDLE_VALUE};
    };

    FileHandle file(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                                nullptr));
    if (file.get() == INVALID_HANDLE_VALUE) {
        return make_error(ErrorCode::io_error, "Unable to open I/O source",
                          path.string() + ": Windows error " + std::to_string(GetLastError()));
    }
    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(offset);
    if (SetFilePointerEx(file.get(), position, nullptr, FILE_BEGIN) == 0) {
        return make_error(ErrorCode::io_error, "Unable to seek I/O source",
                          path.string() + ": Windows error " + std::to_string(GetLastError()));
    }
    std::size_t total = 0U;
    while (total < requested) {
        const auto remaining = requested - total;
        const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(remaining, std::numeric_limits<DWORD>::max()));
        DWORD completed = 0U;
        if (ReadFile(file.get(), destination.data() + total, chunk, &completed, nullptr) == 0) {
            return make_error(ErrorCode::io_error, "Unable to read I/O source",
                              path.string() + ": Windows error " + std::to_string(GetLastError()));
        }
        total += completed;
        if (completed == 0U) break;
    }
    return total;
#else
    std::ifstream input(path, std::ios::binary);
    if (!input) return make_error(ErrorCode::io_error, "Unable to open I/O source", path.string());
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) return make_error(ErrorCode::io_error, "Unable to seek I/O source", path.string());
    input.read(reinterpret_cast<char*>(destination.data()), static_cast<std::streamsize>(requested));
    return static_cast<std::size_t>(input.gcount());
#endif
}

}  // namespace

BoundedIoPool::BoundedIoPool(IoPoolOptions options, safety::ResourceGovernor& governor)
    : options_(options), governor_(governor) {
    options_.workers = std::max(1U, options_.workers);
    options_.max_queued_requests = std::max<std::size_t>(1U, options_.max_queued_requests);
    options_.max_request_bytes = std::max<ByteCount>(1U, options_.max_request_bytes);
    if (options_.request_timeout < std::chrono::milliseconds::zero()) {
        options_.request_timeout = std::chrono::milliseconds::zero();
    }
    options_.request_timeout = std::min(
        options_.request_timeout, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::hours(1)));
    options_.fault.delay = std::clamp(options_.fault.delay, std::chrono::milliseconds::zero(),
                                      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(1)));
    if (options_.fault.kind != IoFaultKind::none && options_.fault.request_sequence == 0U) {
        options_.fault.request_sequence = 1U;
    }
    workers_.reserve(options_.workers);
    for (unsigned index = 0; index < options_.workers; ++index) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

BoundedIoPool::~BoundedIoPool() {
    shutdown();
}

Result<std::future<Result<IoBuffer>>> BoundedIoPool::submit_read(const std::filesystem::path& path,
                                                                 ByteCount offset,
                                                                 ByteCount size,
                                                                 std::string tag) {
    if (path.empty()) {
        return make_error(ErrorCode::invalid_argument, "I/O path must not be empty", std::move(tag));
    }
    if (size == 0 || size > options_.max_request_bytes) {
        return make_error(ErrorCode::invalid_argument, "I/O request size is outside the configured safe range", std::move(tag));
    }
    if (offset > std::numeric_limits<ByteCount>::max() - size) {
        return make_error(ErrorCode::invalid_argument, "I/O offset plus size overflows uint64", std::move(tag));
    }
    if (size > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
        return make_error(ErrorCode::unsupported, "I/O request does not fit this platform's address space", std::move(tag));
    }

    auto reservation = governor_.try_reserve(safety::ResourceKind::staging, size, tag);
    if (!reservation) {
        return reservation.error();
    }

    Task task;
    task.path = path;
    task.offset = offset;
    task.size = size;
    task.tag = tag;
    task.submitted = std::chrono::steady_clock::now();
    task.reservation = reservation.take_value();
    auto future = task.promise.get_future();

    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            return make_error(ErrorCode::internal_error, "I/O pool is shutting down", std::move(tag));
        }
        if (queue_.size() >= options_.max_queued_requests) {
            return make_error(ErrorCode::resource_exhausted, "I/O queue is full", std::move(tag));
        }
        task.sequence = next_sequence_++;
        queue_.push_back(std::move(task));
    }
    condition_.notify_one();
    return future;
}

void BoundedIoPool::shutdown() noexcept {
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            return;
        }
        stopping_ = true;
        while (!queue_.empty()) {
            auto task = std::move(queue_.front());
            queue_.pop_front();
            task.promise.set_value(make_error(ErrorCode::internal_error, "I/O pool stopped before request execution", task.path.string()));
        }
    }
    condition_.notify_all();
    workers_.clear();
}

std::size_t BoundedIoPool::pending_requests() const {
    std::scoped_lock lock(mutex_);
    return queue_.size();
}

void BoundedIoPool::worker_loop() {
    while (true) {
        Task task;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
        }

        const auto service_started = std::chrono::steady_clock::now();
        const bool injected = options_.fault.kind != IoFaultKind::none &&
                              options_.fault.request_sequence == task.sequence;
        if (injected && options_.fault.delay > std::chrono::milliseconds::zero()) {
            std::this_thread::sleep_for(options_.fault.delay);
        }
        auto cancelled = cancellation_requested(options_.cancelled);
        if (!cancelled) {
            task.promise.set_value(cancelled.error());
            continue;
        }
        if (cancelled.value() || (injected && options_.fault.kind == IoFaultKind::cancelled)) {
            task.promise.set_value(make_error(ErrorCode::resource_exhausted, "I/O request cancelled safely", task.tag));
            continue;
        }
        const auto timed_out = [&] {
            return options_.request_timeout > std::chrono::milliseconds::zero() &&
                   std::chrono::steady_clock::now() - task.submitted >= options_.request_timeout;
        };
        if (timed_out() || (injected && options_.fault.kind == IoFaultKind::timeout)) {
            task.promise.set_value(make_error(ErrorCode::resource_exhausted, "I/O request timed out safely", task.tag));
            continue;
        }
        if (injected && options_.fault.kind == IoFaultKind::fail) {
            task.promise.set_value(make_error(ErrorCode::io_error, "Injected I/O worker failure", task.tag));
            continue;
        }
        if (injected && options_.fault.kind == IoFaultKind::allocation) {
            task.promise.set_value(make_error(ErrorCode::resource_exhausted, "Injected I/O allocation failure", task.tag));
            continue;
        }

        std::error_code error;
        const auto file_size = std::filesystem::file_size(task.path, error);
        if (error) {
            task.promise.set_value(make_error(ErrorCode::io_error, "Unable to stat I/O source", task.path.string()));
            continue;
        }
        if (task.offset > file_size || task.size > file_size - task.offset) {
            task.promise.set_value(make_error(ErrorCode::invalid_argument, "I/O request exceeds file bounds", task.path.string()));
            continue;
        }
        if (task.offset > static_cast<ByteCount>(std::numeric_limits<std::streamoff>::max())) {
            task.promise.set_value(make_error(ErrorCode::unsupported, "I/O offset exceeds streamoff range", task.path.string()));
            continue;
        }

        std::vector<std::byte> bytes;
        try {
            bytes.resize(static_cast<std::size_t>(task.size));
        } catch (const std::bad_alloc&) {
            task.promise.set_value(make_error(ErrorCode::resource_exhausted, "Unable to allocate bounded I/O buffer", task.tag));
            continue;
        }
        const std::size_t requested = injected && options_.fault.kind == IoFaultKind::short_read
                                          ? bytes.size() - 1U
                                          : bytes.size();
        auto read = read_file_range(task.path, task.offset, bytes, requested);
        if (!read) {
            task.promise.set_value(read.error());
            continue;
        }
        if (read.value() != bytes.size()) {
            task.promise.set_value(make_error(ErrorCode::io_error, "Short read from I/O source", task.path.string()));
            continue;
        }
        cancelled = cancellation_requested(options_.cancelled);
        if (!cancelled) {
            task.promise.set_value(cancelled.error());
            continue;
        }
        if (cancelled.value()) {
            task.promise.set_value(make_error(ErrorCode::resource_exhausted, "I/O request cancelled safely", task.tag));
            continue;
        }
        if (timed_out()) {
            task.promise.set_value(make_error(ErrorCode::resource_exhausted, "I/O request timed out safely", task.tag));
            continue;
        }
        const auto completed = std::chrono::steady_clock::now();
        const IoTiming timing{
            task.submitted,
            service_started,
            completed,
            std::chrono::duration<double, std::milli>(service_started - task.submitted).count(),
            std::chrono::duration<double, std::milli>(completed - service_started).count(),
            std::chrono::duration<double, std::milli>(completed - task.submitted).count(),
        };
        task.promise.set_value(IoBuffer{std::move(bytes), std::move(task.reservation), timing, task.sequence});
    }
}

}  // namespace stargate::io
