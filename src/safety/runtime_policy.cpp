#include "vastloom/safety/runtime_policy.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#include "vastloom/util/string.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace vastloom::safety {
namespace {

constexpr ByteCount kMinimumPhysicalRam = 8U * GiB;
constexpr ByteCount kSystemHeadroom = 8U * GiB;
constexpr ByteCount kStorageReserve = 300U * GiB;

Result<ByteCount> current_commit_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)) == 0) {
        return make_error(ErrorCode::internal_error, "Unable to query current process memory");
    }
    return static_cast<ByteCount>(counters.PrivateUsage);
#else
    std::ifstream input("/proc/self/status");
    std::string key;
    ByteCount kibibytes = 0;
    std::string unit;
    while (input >> key) {
        if (key == "VmRSS:") {
            if (!(input >> kibibytes >> unit)) break;
            return kibibytes * KiB;
        }
        std::string rest;
        std::getline(input, rest);
    }
    return make_error(ErrorCode::internal_error, "Unable to query current process resident memory");
#endif
}

}  // namespace

Result<ByteCount> process_peak_commit_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters)) == 0) {
        return make_error(ErrorCode::internal_error, "Unable to query peak process memory");
    }
    return static_cast<ByteCount>(counters.PeakPagefileUsage);
#else
    // VmHWM is the peak resident set, the closest counterpart to what the Windows path reports.
    std::ifstream input("/proc/self/status");
    std::string key;
    ByteCount kibibytes = 0;
    std::string unit;
    while (input >> key) {
        if (key == "VmHWM:") {
            if (!(input >> kibibytes >> unit)) break;
            return kibibytes * KiB;
        }
        std::string rest;
        std::getline(input, rest);
    }
    return make_error(ErrorCode::internal_error, "Unable to query peak process resident memory");
#endif
}

namespace {

Result<ByteCount> current_available_ram() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) == 0) return make_error(ErrorCode::internal_error, "Unable to query available system RAM");
    return static_cast<ByteCount>(status.ullAvailPhys);
#else
    std::ifstream input("/proc/meminfo");
    std::string key;
    ByteCount kibibytes = 0;
    std::string unit;
    while (input >> key) {
        if (key == "MemAvailable:") {
            if (!(input >> kibibytes >> unit)) break;
            return kibibytes * KiB;
        }
        std::string rest;
        std::getline(input, rest);
    }
    return make_error(ErrorCode::internal_error, "Unable to query available system RAM");
#endif
}

}  // namespace

Result<RuntimePolicy> make_safe_runtime_policy(const hardware::HardwareSnapshot& hardware) {
    if (hardware.memory.total_bytes < kMinimumPhysicalRam) {
        return make_error(ErrorCode::resource_exhausted, "Real-model execution requires at least 8 GiB of physical RAM");
    }
    if (hardware.memory.available_bytes <= kSystemHeadroom) {
        return make_error(ErrorCode::resource_exhausted, "Less than 8 GiB of RAM headroom is currently available");
    }
    if (hardware.storage.available_bytes <= kStorageReserve) {
        return make_error(ErrorCode::resource_exhausted, "Less than the 300 GiB storage reserve is currently available");
    }
    const ByteCount half_physical = hardware.memory.total_bytes / 2U;
    const ByteCount available_budget = hardware.memory.available_bytes - kSystemHeadroom;
    const ByteCount ram_limit = std::min(half_physical, available_budget);
    if (ram_limit < 512U * MiB) {
        return make_error(ErrorCode::resource_exhausted, "Safe real-model RAM budget is too small");
    }
    RuntimePolicy policy;
    policy.resources.host_memory_hard_limit = ram_limit;
    policy.resources.ram_hard_limit = ram_limit;
    policy.resources.minimum_system_ram_available = kSystemHeadroom;
    policy.resources.minimum_storage_available = kStorageReserve;
    policy.resources.staging_hard_limit = std::min<ByteCount>(512U * MiB, ram_limit / 4U);
    policy.resources.scratch_hard_limit = std::min<ByteCount>(2U * GiB, ram_limit / 4U);
    if (!hardware.gpus.empty() && hardware.gpus.front().free_vram_bytes > 1U * GiB) {
        policy.resources.vram_hard_limit = std::min(
            percentage(hardware.gpus.front().total_vram_bytes, 0.80),
            hardware.gpus.front().free_vram_bytes - 1U * GiB);
    }
    // Half the logical processors, capped. The old fixed ceiling of six predated any threaded
    // kernel and left a 24-thread machine using a quarter of itself for inference; half still
    // leaves the host as much CPU as the model takes, which is what keeps it usable during a run.
    policy.cpu_threads = std::clamp(hardware.cpu.logical_cores / 2U, 1U, 12U);
    policy.io_workers = 2U;
    policy.max_queued_io = 4U;
    policy.max_io_request_bytes = 64U * MiB;
    policy.storage_reserve_bytes = kStorageReserve;
    return policy;
}

std::string to_json(const RuntimePolicy& policy) {
    std::ostringstream output;
    output << "{\"host_memory_hard_limit\":" << policy.resources.host_memory_hard_limit
           << ",\"ram_hard_limit\":" << policy.resources.ram_hard_limit
           << ",\"vram_hard_limit\":" << policy.resources.vram_hard_limit
           << ",\"staging_hard_limit\":" << policy.resources.staging_hard_limit
           << ",\"scratch_hard_limit\":" << policy.resources.scratch_hard_limit
           << ",\"minimum_system_ram_available\":" << policy.resources.minimum_system_ram_available
           << ",\"storage_reserve_bytes\":" << policy.storage_reserve_bytes
           << ",\"cpu_threads\":" << policy.cpu_threads
           << ",\"io_workers\":" << policy.io_workers
           << ",\"max_queued_io\":" << policy.max_queued_io
           << ",\"max_io_request_bytes\":" << policy.max_io_request_bytes << '}';
    return output.str();
}

std::string to_text(const RuntimePolicy& policy) {
    std::ostringstream output;
    output << "Safe runtime policy\n"
           << "Host-memory limit:    " << format_bytes(policy.resources.host_memory_hard_limit) << '\n'
           << "RAM hard limit:       " << format_bytes(policy.resources.ram_hard_limit) << '\n'
           << "VRAM hard limit:      " << format_bytes(policy.resources.vram_hard_limit) << '\n'
           << "System RAM headroom:  " << format_bytes(policy.resources.minimum_system_ram_available) << '\n'
           << "Disk reserve:         " << format_bytes(policy.storage_reserve_bytes) << '\n'
           << "CPU threads:          " << policy.cpu_threads << '\n'
           << "I/O workers/queue:    " << policy.io_workers << '/' << policy.max_queued_io << '\n'
           << "Maximum I/O request:  " << format_bytes(policy.max_io_request_bytes);
    return output.str();
}

ProcessSafetyGuard::ProcessSafetyGuard(ProcessSafetyGuard&& other) noexcept
    : native_job_(std::exchange(other.native_job_, nullptr)),
      commit_limit_bytes_(std::exchange(other.commit_limit_bytes_, 0)),
      minimum_system_available_(std::exchange(other.minimum_system_available_, 0)) {}

ProcessSafetyGuard& ProcessSafetyGuard::operator=(ProcessSafetyGuard&& other) noexcept {
    if (this != &other) {
        release();
        native_job_ = std::exchange(other.native_job_, nullptr);
        commit_limit_bytes_ = std::exchange(other.commit_limit_bytes_, 0);
        minimum_system_available_ = std::exchange(other.minimum_system_available_, 0);
    }
    return *this;
}

ProcessSafetyGuard::~ProcessSafetyGuard() { release(); }

Result<ProcessSafetyGuard> ProcessSafetyGuard::apply(const RuntimePolicy& policy) {
    if (policy.resources.ram_hard_limit == 0U) return make_error(ErrorCode::invalid_argument, "Process memory limit must be nonzero");
    ProcessSafetyGuard guard;
    guard.commit_limit_bytes_ = policy.resources.ram_hard_limit;
    guard.minimum_system_available_ = policy.resources.minimum_system_ram_available;
#if defined(_WIN32)
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job == nullptr) return make_error(ErrorCode::internal_error, "Unable to create Windows memory Job Object");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.ProcessMemoryLimit = static_cast<SIZE_T>(policy.resources.ram_hard_limit);
    if (SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) == 0 ||
        AssignProcessToJobObject(job, GetCurrentProcess()) == 0) {
        CloseHandle(job);
        return make_error(ErrorCode::permission_denied, "Unable to enforce Windows process memory Job Object");
    }
    guard.native_job_ = job;
    if (policy.lower_process_priority && SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS) == 0) {
        guard.release();
        return make_error(ErrorCode::permission_denied, "Unable to lower Windows process priority");
    }
#else
    if (policy.lower_process_priority && setpriority(PRIO_PROCESS, 0, 5) != 0) {
        return make_error(ErrorCode::permission_denied, "Unable to lower process priority");
    }
#endif
    return guard;
}

Result<void> ProcessSafetyGuard::sample_and_check() const {
    auto commit = current_commit_bytes();
    if (!commit) return commit.error();
    if (commit.value() >= commit_limit_bytes_) {
        return make_error(ErrorCode::resource_exhausted, "Process reached its hard memory budget; generation must stop");
    }
    auto available = current_available_ram();
    if (!available) return available.error();
    if (available.value() <= minimum_system_available_ / 4U) {
        return make_error(ErrorCode::resource_exhausted, "System RAM entered EMERGENCY state; generation must stop");
    }
    return {};
}

void ProcessSafetyGuard::release() noexcept {
#if defined(_WIN32)
    if (native_job_ != nullptr) CloseHandle(static_cast<HANDLE>(native_job_));
#endif
    native_job_ = nullptr;
    commit_limit_bytes_ = 0;
    minimum_system_available_ = 0;
}

}  // namespace vastloom::safety
