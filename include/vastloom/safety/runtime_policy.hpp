#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/hardware/hardware_probe.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::safety {

struct RuntimePolicy {
    ResourceLimits resources;
    unsigned cpu_threads{1};
    unsigned io_workers{1};
    std::size_t max_queued_io{1};
    ByteCount max_io_request_bytes{0};
    ByteCount storage_reserve_bytes{0};
    bool lower_process_priority{true};
};

[[nodiscard]] Result<RuntimePolicy> make_safe_runtime_policy(const hardware::HardwareSnapshot& hardware);
[[nodiscard]] std::string to_json(const RuntimePolicy& policy);
[[nodiscard]] std::string to_text(const RuntimePolicy& policy);

class ProcessSafetyGuard {
public:
    ProcessSafetyGuard() = default;
    ProcessSafetyGuard(const ProcessSafetyGuard&) = delete;
    ProcessSafetyGuard& operator=(const ProcessSafetyGuard&) = delete;
    ProcessSafetyGuard(ProcessSafetyGuard&& other) noexcept;
    ProcessSafetyGuard& operator=(ProcessSafetyGuard&& other) noexcept;
    ~ProcessSafetyGuard();

    [[nodiscard]] static Result<ProcessSafetyGuard> apply(const RuntimePolicy& policy);
    [[nodiscard]] Result<void> sample_and_check() const;
    [[nodiscard]] ByteCount commit_limit_bytes() const noexcept { return commit_limit_bytes_; }

private:
    void release() noexcept;
    void* native_job_{nullptr};
    ByteCount commit_limit_bytes_{0};
    ByteCount minimum_system_available_{0};
};

}  // namespace vastloom::safety
