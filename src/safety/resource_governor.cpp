#include "stargate/safety/resource_governor.hpp"

#include <algorithm>
#include <utility>

namespace stargate::safety {
namespace {

int severity(SafetyState state) noexcept {
    return static_cast<int>(state);
}

SafetyState max_state(SafetyState left, SafetyState right) noexcept {
    return severity(left) >= severity(right) ? left : right;
}

}  // namespace

Reservation::Reservation(ResourceGovernor* governor, ResourceKind kind, ByteCount bytes, std::string tag)
    : governor_(governor), kind_(kind), bytes_(bytes), tag_(std::move(tag)) {}

Reservation::Reservation(Reservation&& other) noexcept
    : governor_(std::exchange(other.governor_, nullptr)),
      kind_(other.kind_),
      bytes_(std::exchange(other.bytes_, 0)),
      tag_(std::move(other.tag_)) {}

Reservation& Reservation::operator=(Reservation&& other) noexcept {
    if (this != &other) {
        release();
        governor_ = std::exchange(other.governor_, nullptr);
        kind_ = other.kind_;
        bytes_ = std::exchange(other.bytes_, 0);
        tag_ = std::move(other.tag_);
    }
    return *this;
}

Reservation::~Reservation() {
    release();
}

void Reservation::release() noexcept {
    if (governor_ != nullptr && bytes_ != 0) {
        governor_->release(kind_, bytes_);
        governor_ = nullptr;
        bytes_ = 0;
        tag_.clear();
    }
}

ResourceGovernor::ResourceGovernor(ResourceLimits limits) : limits_(limits) {
    limits_.pressure_ratio = std::clamp(limits_.pressure_ratio, 0.10, 0.99);
    limits_.critical_ratio = std::clamp(limits_.critical_ratio, limits_.pressure_ratio + 0.01, 1.0);
    limits_.recovery_samples = std::max(1U, limits_.recovery_samples);
}

Result<Reservation> ResourceGovernor::try_reserve(ResourceKind kind, ByteCount bytes, std::string tag) {
    if (bytes == 0) {
        return make_error(ErrorCode::invalid_argument, "A zero-byte reservation is not allowed", std::move(tag));
    }

    std::scoped_lock lock(mutex_);
    if (state_ == SafetyState::critical || state_ == SafetyState::emergency) {
        return make_error(ErrorCode::resource_exhausted,
                          "Safety governor rejected new work in " + std::string(to_string(state_)) + " state",
                          std::move(tag));
    }

    const std::size_t index = index_for(kind);
    const ByteCount limit = limit_for(kind);
    if (limit == 0 || bytes > limit || reserved_[index] > limit - bytes) {
        return make_error(ErrorCode::resource_exhausted,
                          "Reservation would exceed the hard " + std::string(to_string(kind)) + " budget",
                          std::move(tag));
    }

    reserved_[index] += bytes;
    return Reservation(this, kind, bytes, std::move(tag));
}

SafetyState ResourceGovernor::update_pressure(const RuntimePressure& pressure) {
    std::scoped_lock lock(mutex_);
    SafetyState raw_state = SafetyState::normal;
    std::string reason = "Resource pressure is within configured limits";

    const auto consider_ratio = [&](ResourceKind kind, ByteCount limit) {
        if (limit == 0) {
            return SafetyState::normal;
        }
        const double ratio = static_cast<double>(reserved_[index_for(kind)]) / static_cast<double>(limit);
        if (ratio >= limits_.critical_ratio) {
            return SafetyState::critical;
        }
        if (ratio >= limits_.pressure_ratio) {
            return SafetyState::pressure;
        }
        return SafetyState::normal;
    };

    const SafetyState ram_ratio_state = consider_ratio(ResourceKind::ram, limits_.ram_hard_limit);
    const SafetyState vram_ratio_state = consider_ratio(ResourceKind::vram, limits_.vram_hard_limit);
    const SafetyState staging_ratio_state = consider_ratio(ResourceKind::staging, limits_.staging_hard_limit);
    const SafetyState scratch_ratio_state = consider_ratio(ResourceKind::scratch, limits_.scratch_hard_limit);
    raw_state = max_state(max_state(ram_ratio_state, vram_ratio_state), max_state(staging_ratio_state, scratch_ratio_state));

    if (pressure.system_ram_available <= limits_.minimum_system_ram_available / 4U) {
        raw_state = SafetyState::emergency;
        reason = "System RAM emergency reserve is nearly exhausted";
    } else if (pressure.system_ram_available <= limits_.minimum_system_ram_available) {
        raw_state = max_state(raw_state, SafetyState::critical);
        reason = "System RAM is below the configured emergency reserve";
    } else if (pressure.system_ram_available <= limits_.minimum_system_ram_available * 2U) {
        raw_state = max_state(raw_state, SafetyState::pressure);
        reason = "System RAM is approaching the configured emergency reserve";
    }

    if (pressure.storage_available <= limits_.minimum_storage_available / 4U) {
        raw_state = SafetyState::emergency;
        reason = "Storage emergency reserve is nearly exhausted";
    } else if (pressure.storage_available <= limits_.minimum_storage_available) {
        raw_state = max_state(raw_state, SafetyState::critical);
        reason = "Storage is below the configured minimum free-space reserve";
    } else if (pressure.storage_available <= limits_.minimum_storage_available * 2U) {
        raw_state = max_state(raw_state, SafetyState::pressure);
        reason = "Storage is approaching the configured minimum free-space reserve";
    }

    if (severity(raw_state) > severity(state_)) {
        state_ = raw_state;
        recovery_counter_ = 0;
    } else if (severity(raw_state) < severity(state_)) {
        ++recovery_counter_;
        if (recovery_counter_ >= limits_.recovery_samples) {
            state_ = static_cast<SafetyState>(severity(state_) - 1);
            recovery_counter_ = 0;
        }
    } else {
        recovery_counter_ = 0;
    }

    last_reason_ = std::move(reason);
    return state_;
}

GovernorSnapshot ResourceGovernor::snapshot() const {
    std::scoped_lock lock(mutex_);
    return GovernorSnapshot{state_, reserved_, limits_, recovery_counter_, last_reason_};
}

bool ResourceGovernor::allows_new_work() const {
    std::scoped_lock lock(mutex_);
    return state_ == SafetyState::normal || state_ == SafetyState::pressure;
}

void ResourceGovernor::release(ResourceKind kind, ByteCount bytes) noexcept {
    std::scoped_lock lock(mutex_);
    const std::size_t index = index_for(kind);
    reserved_[index] = bytes >= reserved_[index] ? 0 : reserved_[index] - bytes;
}

ByteCount ResourceGovernor::limit_for(ResourceKind kind) const noexcept {
    switch (kind) {
        case ResourceKind::ram: return limits_.ram_hard_limit;
        case ResourceKind::vram: return limits_.vram_hard_limit;
        case ResourceKind::staging: return limits_.staging_hard_limit;
        case ResourceKind::scratch: return limits_.scratch_hard_limit;
    }
    return 0;
}

std::size_t ResourceGovernor::index_for(ResourceKind kind) noexcept {
    return static_cast<std::size_t>(kind);
}

std::string_view to_string(ResourceKind kind) noexcept {
    switch (kind) {
        case ResourceKind::ram: return "RAM";
        case ResourceKind::vram: return "VRAM";
        case ResourceKind::staging: return "staging";
        case ResourceKind::scratch: return "scratch";
    }
    return "unknown";
}

std::string_view to_string(SafetyState state) noexcept {
    switch (state) {
        case SafetyState::normal: return "NORMAL";
        case SafetyState::pressure: return "PRESSURE";
        case SafetyState::critical: return "CRITICAL";
        case SafetyState::emergency: return "EMERGENCY";
    }
    return "UNKNOWN";
}

}  // namespace stargate::safety
