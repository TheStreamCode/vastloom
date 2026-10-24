#pragma once

#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/io/bounded_io_pool.hpp"
#include "vastloom/io/device_io_counters.hpp"
#include "vastloom/io/process_io_counters.hpp"
#include "vastloom/model/residency.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::model {

struct ExpertLoadOptions {
    ByteCount max_expert_bytes{512U * MiB};
    ByteCount max_range_bytes{128U * MiB};
    std::size_t max_ranges{64U};
    bool measure_process_io{true};
    bool measure_device_io{false};
    std::function<bool()> cancelled;
    io::BufferedReadSession* read_session{nullptr};
    io::IoCacheMode io_cache_mode{io::IoCacheMode::buffered};
};

struct ExpertLoadTelemetry {
    ByteCount logical_bytes{0};
    std::size_t range_count{0};
    std::size_t read_calls{0};
    std::size_t bypass_read_calls{0};
    std::size_t buffered_read_calls{0};
    std::size_t request_sequence{0};
    double queue_ms{0.0};
    double service_ms{0.0};
    double total_ms{0.0};
    double elapsed_ms{0.0};
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point service_started_at;
    std::chrono::steady_clock::time_point completed_at;
    std::optional<io::ProcessIoDelta> process_io;
    std::optional<io::DeviceIoDelta> device_io;
};

class LoadedExpert {
public:
    LoadedExpert(const LoadedExpert&) = delete;
    LoadedExpert& operator=(const LoadedExpert&) = delete;
    LoadedExpert(LoadedExpert&&) noexcept = default;
    LoadedExpert& operator=(LoadedExpert&&) = delete;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return buffer_ == nullptr ? std::span<const std::byte>{} : buffer_->bytes();
    }
    [[nodiscard]] std::shared_ptr<const ResidentByteBuffer> retain_buffer() const noexcept {
        return buffer_;
    }
    [[nodiscard]] Result<std::span<const std::byte>> tensor(std::size_t catalog_tensor_index) const;
    [[nodiscard]] std::span<const ExpertTensorSlice> tensor_slices() const noexcept { return tensor_slices_; }
    [[nodiscard]] const ExpertLoadTelemetry& telemetry() const noexcept { return telemetry_; }

private:
    friend class BoundedExpertLoadPool;
    friend Result<LoadedExpert> load_expert_bundle(const ExpertBundle&,
                                                    ResidencyObject&,
                                                    safety::ResourceGovernor&,
                                                    const ExpertLoadOptions&);
    friend Result<LoadedExpert> load_prepared_expert_bundle(const ExpertBundle&,
                                                             ResidencyObject&,
                                                             ResidencyTransfer&,
                                                             const ExpertLoadOptions&,
                                                             ByteCount);
    LoadedExpert(ResidencyUse lease,
                 std::vector<std::byte> bytes,
                 std::vector<ExpertTensorSlice> tensor_slices,
                 ExpertLoadTelemetry telemetry);

    std::shared_ptr<const ResidentByteBuffer> buffer_;
    std::vector<ExpertTensorSlice> tensor_slices_;
    ExpertLoadTelemetry telemetry_;
};

[[nodiscard]] Result<LoadedExpert> load_expert_bundle(
    const ExpertBundle& bundle,
    ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const ExpertLoadOptions& options = {});

struct ExpertLoadPoolOptions {
    unsigned workers{1U};
    std::size_t max_queued_requests{4U};
    ExpertLoadOptions load;
};

class BoundedExpertLoadPool {
public:
    // The governor must outlive the pool and every LoadedExpert returned by it.
    BoundedExpertLoadPool(ExpertLoadPoolOptions options, safety::ResourceGovernor& governor);
    BoundedExpertLoadPool(const BoundedExpertLoadPool&) = delete;
    BoundedExpertLoadPool& operator=(const BoundedExpertLoadPool&) = delete;
    ~BoundedExpertLoadPool();

    [[nodiscard]] Result<std::future<Result<LoadedExpert>>> submit(
        const ExpertBundle& bundle,
        ResidencyObject& residency,
        std::string tag);
    void shutdown() noexcept;
    [[nodiscard]] std::size_t pending_requests() const;

private:
    struct Task;
    void worker_loop();

    ExpertLoadPoolOptions options_;
    safety::ResourceGovernor& governor_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::unique_ptr<Task>> queue_;
    std::vector<std::jthread> workers_;
    std::size_t next_sequence_{1U};
    bool stopping_{false};
};

}  // namespace vastloom::model
