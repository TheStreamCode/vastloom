#include "stargate/io/bounded_io_pool.hpp"

#include <fstream>
#include <limits>

namespace stargate::io {

BoundedIoPool::BoundedIoPool(IoPoolOptions options, safety::ResourceGovernor& governor)
    : options_(options), governor_(governor) {
    options_.workers = std::max(1U, options_.workers);
    options_.max_queued_requests = std::max<std::size_t>(1U, options_.max_queued_requests);
    options_.max_request_bytes = std::max<ByteCount>(1U, options_.max_request_bytes);
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

        std::ifstream input(task.path, std::ios::binary);
        if (!input) {
            task.promise.set_value(make_error(ErrorCode::io_error, "Unable to open I/O source", task.path.string()));
            continue;
        }
        input.seekg(static_cast<std::streamoff>(task.offset), std::ios::beg);
        if (!input) {
            task.promise.set_value(make_error(ErrorCode::io_error, "Unable to seek I/O source", task.path.string()));
            continue;
        }

        std::vector<std::byte> bytes(static_cast<std::size_t>(task.size));
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
            task.promise.set_value(make_error(ErrorCode::io_error, "Short read from I/O source", task.path.string()));
            continue;
        }
        task.promise.set_value(IoBuffer{std::move(bytes), std::move(task.reservation)});
    }
}

}  // namespace stargate::io
