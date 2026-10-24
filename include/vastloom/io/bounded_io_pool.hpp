#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::io {

enum class IoCacheMode : std::uint8_t {
    buffered,
    bypass,
};

[[nodiscard]] Result<IoCacheMode> parse_io_cache_mode(std::string_view value);
[[nodiscard]] std::string_view to_string(IoCacheMode mode) noexcept;
[[nodiscard]] Result<void> read_buffered_range_into(const std::filesystem::path& path,
                                                     ByteCount offset,
                                                     std::span<std::byte> destination);

// Keeps a few shard handles open for the life of one worker. Expert generate hits the same
// two or three GGUF files thousands of times; this is not a process-wide cache, so a later
// overwrite of a fixture file still goes through a fresh open.
class BufferedReadSession {
public:
    BufferedReadSession();
    BufferedReadSession(const BufferedReadSession&) = delete;
    BufferedReadSession& operator=(const BufferedReadSession&) = delete;
    ~BufferedReadSession();

    [[nodiscard]] Result<void> read_into(const std::filesystem::path& path,
                                         ByteCount offset,
                                         std::span<std::byte> destination);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Reads a range with the caller's choice of page-cache behaviour. When `actual_mode` is non-null it
// receives the path that delivered the bytes: a bypass request that the platform refuses (short
// read, alignment rejection) falls back to a buffered read and is reported as such.
//
// Expert payload is the reason this is exposed. A decode step reads gigabytes of expert weights
// that will not be read again - a different token routes to different experts - so filling the page
// cache with them evicts what the host was using and buys nothing. `bypass` also avoids the copy
// through that cache, and measured effective bandwidth on this path was well under what the device
// can sustain. Falls back to a buffered read when the platform refuses an unbuffered one, because
// alignment requirements are a property of the device and not something a caller can guarantee.
[[nodiscard]] Result<void> read_range_into(const std::filesystem::path& path,
                                           ByteCount offset,
                                           std::span<std::byte> destination,
                                           IoCacheMode cache_mode,
                                           IoCacheMode* actual_mode = nullptr);

enum class IoFaultKind : std::uint8_t {
    none,
    fail,
    short_read,
    allocation,
    cancelled,
    timeout,
};

struct IoFaultInjection {
    IoFaultKind kind{IoFaultKind::none};
    std::size_t request_sequence{0};
    std::chrono::milliseconds delay{0};
};

struct IoTiming {
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point service_started_at;
    std::chrono::steady_clock::time_point completed_at;
    double queue_ms{0.0};
    double service_ms{0.0};
    double total_ms{0.0};
};

struct IoBuffer {
    std::vector<std::byte> bytes;
    safety::Reservation reservation;
    IoTiming timing;
    std::size_t request_sequence{0};
};

struct IoPoolOptions {
    unsigned workers{1};
    std::size_t max_queued_requests{32};
    ByteCount max_request_bytes{512U * MiB};
    IoCacheMode cache_mode{IoCacheMode::buffered};
    std::chrono::milliseconds request_timeout{std::chrono::seconds(30)};
    std::function<bool()> cancelled;
    IoFaultInjection fault;
};

class BoundedIoPool {
public:
    BoundedIoPool(IoPoolOptions options, safety::ResourceGovernor& governor);
    BoundedIoPool(const BoundedIoPool&) = delete;
    BoundedIoPool& operator=(const BoundedIoPool&) = delete;
    ~BoundedIoPool();

    [[nodiscard]] Result<std::future<Result<IoBuffer>>> submit_read(const std::filesystem::path& path,
                                                                    ByteCount offset,
                                                                    ByteCount size,
                                                                    std::string tag);
    void shutdown() noexcept;
    [[nodiscard]] std::size_t pending_requests() const;

private:
    struct Task {
        std::filesystem::path path;
        ByteCount offset{0};
        ByteCount size{0};
        std::string tag;
        std::size_t sequence{0};
        std::chrono::steady_clock::time_point submitted;
        safety::Reservation reservation;
        std::promise<Result<IoBuffer>> promise;
    };

    void worker_loop();

    IoPoolOptions options_;
    safety::ResourceGovernor& governor_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Task> queue_;
    std::vector<std::jthread> workers_;
    std::size_t next_sequence_{1};
    bool stopping_{false};
};

}  // namespace vastloom::io
