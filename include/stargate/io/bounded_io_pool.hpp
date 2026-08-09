#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "stargate/core/bytes.hpp"
#include "stargate/core/status.hpp"
#include "stargate/safety/resource_governor.hpp"

namespace stargate::io {

struct IoBuffer {
    std::vector<std::byte> bytes;
    safety::Reservation reservation;
};

struct IoPoolOptions {
    unsigned workers{1};
    std::size_t max_queued_requests{32};
    ByteCount max_request_bytes{512U * MiB};
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
    bool stopping_{false};
};

}  // namespace stargate::io
