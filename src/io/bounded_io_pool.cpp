#include "vastloom/io/bounded_io_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <span>
#include <thread>

#include "vastloom/util/string.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#include <malloc.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace vastloom::io {
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
                                    std::size_t requested,
                                    IoCacheMode cache_mode) {
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

    const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS |
                        (cache_mode == IoCacheMode::bypass ? FILE_FLAG_NO_BUFFERING : 0U);
    FileHandle file(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, flags,
                                nullptr));
    if (file.get() == INVALID_HANDLE_VALUE) {
        return make_error(ErrorCode::io_error, "Unable to open I/O source",
                          path.string() + ": Windows error " + std::to_string(GetLastError()));
    }
    std::byte* read_destination = destination.data();
    void* aligned_buffer = nullptr;
    std::size_t offset_alignment = 1U;
    std::size_t memory_alignment = 1U;
    std::size_t prefix_bytes = 0U;
    std::size_t read_size = requested;
    ByteCount read_offset = offset;
    if (cache_mode == IoCacheMode::bypass) {
        STORAGE_PROPERTY_QUERY query{};
        query.PropertyId = StorageAccessAlignmentProperty;
        query.QueryType = PropertyStandardQuery;
        STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR descriptor{};
        DWORD returned = 0U;
        if (DeviceIoControl(file.get(), IOCTL_STORAGE_QUERY_PROPERTY,
                            &query, static_cast<DWORD>(sizeof(query)),
                            &descriptor, static_cast<DWORD>(sizeof(descriptor)),
                            &returned, nullptr) != 0 &&
            returned >= static_cast<DWORD>(sizeof(descriptor))) {
            offset_alignment = descriptor.BytesPerLogicalSector;
            memory_alignment = descriptor.BytesPerPhysicalSector;
        }
        offset_alignment = std::max<std::size_t>(offset_alignment, 512U);
        memory_alignment = std::max<std::size_t>({memory_alignment, offset_alignment, 4096U});
        constexpr std::size_t maximum_supported_alignment = 64U * 1024U;
        if ((offset_alignment & (offset_alignment - 1U)) != 0U ||
            (memory_alignment & (memory_alignment - 1U)) != 0U ||
            offset_alignment > maximum_supported_alignment ||
            memory_alignment > maximum_supported_alignment) {
            return make_error(ErrorCode::unsupported,
                              "Windows direct-I/O alignment is unsupported", path.string());
        }
        prefix_bytes = static_cast<std::size_t>(offset % offset_alignment);
        read_offset = offset - prefix_bytes;
        if (requested > std::numeric_limits<std::size_t>::max() - prefix_bytes - (offset_alignment - 1U)) {
            return make_error(ErrorCode::invalid_argument, "Windows direct-I/O envelope overflows", path.string());
        }
        const std::size_t required = prefix_bytes + requested;
        read_size = ((required + offset_alignment - 1U) / offset_alignment) * offset_alignment;
        aligned_buffer = _aligned_malloc(read_size, memory_alignment);
        if (aligned_buffer == nullptr) {
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to allocate aligned unbuffered I/O memory", path.string());
        }
        read_destination = static_cast<std::byte*>(aligned_buffer);
    }

    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(read_offset);
    if (SetFilePointerEx(file.get(), position, nullptr, FILE_BEGIN) == 0) {
        const DWORD seek_error = GetLastError();
        if (aligned_buffer != nullptr) _aligned_free(aligned_buffer);
        return make_error(ErrorCode::io_error, "Unable to seek I/O source",
                          path.string() + ": Windows error " + std::to_string(seek_error));
    }

    const auto release_aligned = [&] {
        if (aligned_buffer != nullptr) _aligned_free(aligned_buffer);
    };
    std::size_t total = 0U;
    while (total < read_size) {
        const auto remaining = read_size - total;
        std::size_t maximum_chunk = std::numeric_limits<DWORD>::max();
        if (cache_mode == IoCacheMode::bypass) maximum_chunk -= maximum_chunk % offset_alignment;
        const DWORD chunk = static_cast<DWORD>(std::min(remaining, maximum_chunk));
        DWORD completed = 0U;
        if (ReadFile(file.get(), read_destination + total, chunk, &completed, nullptr) == 0) {
            const DWORD read_error = GetLastError();
            release_aligned();
            return make_error(ErrorCode::io_error, "Unable to read I/O source",
                              path.string() + ": Windows error " + std::to_string(read_error));
        }
        total += completed;
        if (completed == 0U) break;
    }
    std::size_t delivered = total;
    if (cache_mode == IoCacheMode::bypass) {
        delivered = total > prefix_bytes ? std::min(requested, total - prefix_bytes) : 0U;
        if (delivered != 0U) {
            std::memcpy(destination.data(), read_destination + prefix_bytes, delivered);
        }
    }
    release_aligned();
    return delivered;
#elif defined(__linux__)
    if (cache_mode == IoCacheMode::bypass) {
        constexpr std::size_t alignment = 4096U;
        const std::size_t prefix_bytes = static_cast<std::size_t>(offset % alignment);
        const ByteCount read_offset = offset - prefix_bytes;
        if (requested > std::numeric_limits<std::size_t>::max() - prefix_bytes - (alignment - 1U)) {
            return make_error(ErrorCode::invalid_argument, "Linux direct-I/O envelope overflows", path.string());
        }
        const std::size_t required = prefix_bytes + requested;
        const std::size_t read_size = ((required + alignment - 1U) / alignment) * alignment;
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (descriptor < 0) {
            return make_error(ErrorCode::unsupported, "Unable to open direct I/O source",
                              path.string() + ": errno " + std::to_string(errno));
        }
        void* aligned_buffer = nullptr;
        if (posix_memalign(&aligned_buffer, alignment, read_size) != 0 || aligned_buffer == nullptr) {
            ::close(descriptor);
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to allocate aligned direct-I/O memory", path.string());
        }
        ssize_t completed = -1;
        do {
            completed = ::pread(descriptor, aligned_buffer, read_size, static_cast<off_t>(read_offset));
        } while (completed < 0 && errno == EINTR);
        const int read_error = errno;
        ::close(descriptor);
        if (completed < 0) {
            std::free(aligned_buffer);
            return make_error(ErrorCode::unsupported, "Unable to read direct I/O source",
                              path.string() + ": errno " + std::to_string(read_error));
        }
        const auto total = static_cast<std::size_t>(completed);
        const std::size_t delivered = total > prefix_bytes
                                          ? std::min(requested, total - prefix_bytes)
                                          : 0U;
        if (delivered != 0U) {
            std::memcpy(destination.data(), static_cast<std::byte*>(aligned_buffer) + prefix_bytes, delivered);
        }
        std::free(aligned_buffer);
        return delivered;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) return make_error(ErrorCode::io_error, "Unable to open I/O source", path.string());
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) return make_error(ErrorCode::io_error, "Unable to seek I/O source", path.string());
    input.read(reinterpret_cast<char*>(destination.data()), static_cast<std::streamsize>(requested));
    return static_cast<std::size_t>(input.gcount());
#else
    (void)path;
    (void)offset;
    (void)destination;
    (void)requested;
    (void)cache_mode;
    return make_error(ErrorCode::unsupported, "Bounded file reads are unsupported on this operating system");
#endif
}

}  // namespace

Result<IoCacheMode> parse_io_cache_mode(std::string_view value) {
    const auto normalized = util::to_lower(util::trim(value));
    if (normalized == "buffered" || normalized == "system") return IoCacheMode::buffered;
    if (normalized == "bypass" || normalized == "direct" || normalized == "unbuffered") {
        return IoCacheMode::bypass;
    }
    return make_error(ErrorCode::invalid_argument, "Unknown I/O cache mode", std::string(value));
}

std::string_view to_string(IoCacheMode mode) noexcept {
    switch (mode) {
        case IoCacheMode::buffered: return "buffered";
        case IoCacheMode::bypass: return "bypass";
    }
    return "unknown";
}

Result<void> read_buffered_range_into(const std::filesystem::path& path,
                                      ByteCount offset,
                                      std::span<std::byte> destination) {
    if (path.empty() || destination.empty()) {
        return make_error(ErrorCode::invalid_argument, "Buffered range read requires a path and destination");
    }
    const ByteCount bytes = static_cast<ByteCount>(destination.size());
    if (offset > std::numeric_limits<ByteCount>::max() - bytes ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
        offset > static_cast<ByteCount>(std::numeric_limits<std::streamoff>::max())) {
        return make_error(ErrorCode::unsupported, "Buffered range read exceeds platform limits", path.string());
    }
    std::error_code error;
    const bool symlink = std::filesystem::is_symlink(path, error);
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to inspect buffered I/O source", path.string());
    }
    if (symlink) {
        return make_error(ErrorCode::unsafe_path, "Refusing a symlinked buffered I/O source", path.string());
    }
    const auto file_size = std::filesystem::file_size(path, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to stat buffered I/O source", path.string());
    if (offset > file_size || bytes > file_size - offset) {
        return make_error(ErrorCode::invalid_argument, "Buffered range read exceeds file bounds", path.string());
    }
    auto read = read_file_range(path, offset, destination, destination.size(), IoCacheMode::buffered);
    if (!read) return read.error();
    if (read.value() != destination.size()) {
        return make_error(ErrorCode::io_error, "Short buffered range read", path.string());
    }
    return {};
}

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

    ByteCount reservation_bytes = size;
    if (options_.cache_mode == IoCacheMode::bypass) {
        constexpr ByteCount envelope_allowance = 128U * KiB;
        if (size > (std::numeric_limits<ByteCount>::max() - envelope_allowance) / 2U) {
            return make_error(ErrorCode::invalid_argument,
                              "Unbuffered I/O staging reservation overflows", std::move(tag));
        }
        reservation_bytes = size * 2U + envelope_allowance;
    }
    auto reservation = governor_.try_reserve(safety::ResourceKind::staging, reservation_bytes, tag);
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
        auto read = read_file_range(task.path, task.offset, bytes, requested, options_.cache_mode);
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

}  // namespace vastloom::io
