#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::safety {

enum class ResourceKind : std::uint8_t {
    ram = 0,
    vram = 1,
    staging = 2,
    scratch = 3,
};

enum class SafetyState : std::uint8_t {
    normal = 0,
    pressure = 1,
    critical = 2,
    emergency = 3,
};

struct ResourceLimits {
    // Optional aggregate ceiling for all host-memory resources (RAM destination,
    // staging and scratch). A zero value preserves legacy per-kind-only behavior.
    ByteCount host_memory_hard_limit{0};
    ByteCount ram_hard_limit{0};
    ByteCount vram_hard_limit{0};
    ByteCount staging_hard_limit{0};
    ByteCount scratch_hard_limit{0};
    ByteCount minimum_system_ram_available{2U * GiB};
    ByteCount minimum_storage_available{20U * GiB};
    double pressure_ratio{0.82};
    double critical_ratio{0.93};
    unsigned recovery_samples{3};
};

struct RuntimePressure {
    ByteCount system_ram_available{0};
    ByteCount gpu_vram_available{0};
    ByteCount storage_available{0};
};

struct GovernorSnapshot {
    SafetyState state{SafetyState::normal};
    std::array<ByteCount, 4> reserved{};
    ResourceLimits limits;
    unsigned recovery_counter{0};
    std::string last_reason;
};

class ResourceGovernor;

class Reservation {
public:
    Reservation() = default;
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    Reservation(Reservation&& other) noexcept;
    Reservation& operator=(Reservation&& other) noexcept;
    ~Reservation();

    [[nodiscard]] ByteCount bytes() const noexcept { return bytes_; }
    [[nodiscard]] ResourceKind kind() const noexcept { return kind_; }
    [[nodiscard]] std::string_view tag() const noexcept { return tag_; }
    [[nodiscard]] bool valid() const noexcept { return governor_ != nullptr && bytes_ != 0; }
    void release() noexcept;

    // Keeps the bytes charged to the Governor while detaching this RAII handle. Use only when an
    // external allocation could not be released: making the budget unavailable is safer than
    // pretending that leaked memory is free.
    void abandon_accounting_on_external_release_failure() noexcept;

private:
    friend class ResourceGovernor;
    Reservation(ResourceGovernor* governor, ResourceKind kind, ByteCount bytes, std::string tag);

    ResourceGovernor* governor_{nullptr};
    ResourceKind kind_{ResourceKind::ram};
    ByteCount bytes_{0};
    std::string tag_;
};

class ResourceGovernor {
public:
    explicit ResourceGovernor(ResourceLimits limits);
    ResourceGovernor(const ResourceGovernor&) = delete;
    ResourceGovernor& operator=(const ResourceGovernor&) = delete;

    [[nodiscard]] Result<Reservation> try_reserve(ResourceKind kind, ByteCount bytes, std::string tag);
    [[nodiscard]] SafetyState update_pressure(const RuntimePressure& pressure);
    [[nodiscard]] GovernorSnapshot snapshot() const;
    [[nodiscard]] bool allows_new_work() const;

private:
    friend class Reservation;
    void release(ResourceKind kind, ByteCount bytes) noexcept;
    [[nodiscard]] ByteCount limit_for(ResourceKind kind) const noexcept;
    [[nodiscard]] static std::size_t index_for(ResourceKind kind) noexcept;

    ResourceLimits limits_;
    mutable std::mutex mutex_;
    std::array<ByteCount, 4> reserved_{};
    SafetyState state_{SafetyState::normal};
    unsigned recovery_counter_{0};
    std::string last_reason_;
};

[[nodiscard]] std::string_view to_string(ResourceKind kind) noexcept;
[[nodiscard]] std::string_view to_string(SafetyState state) noexcept;

}  // namespace vastloom::safety
