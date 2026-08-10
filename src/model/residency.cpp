#include "vastloom/model/residency.hpp"

#include <limits>
#include <mutex>
#include <new>
#include <utility>

namespace vastloom::model {

struct ResidencyControl {
    mutable std::mutex mutex;
    std::string logical_id;
    ResidencyState state{ResidencyState::nvme};
    ByteCount object_bytes{0};
    std::size_t active_uses{0};
    bool pinned{false};
    std::string failure;
    safety::Reservation ram;
    safety::Reservation vram;
    safety::Reservation staging;
};

namespace {

Error state_error(const ResidencyControl& control, std::string_view operation) {
    return make_error(ErrorCode::invalid_argument,
                      std::string(operation) + " is illegal from residency state " +
                          std::string(to_string(control.state)),
                      control.logical_id);
}

bool ready_state(ResidencyState state) noexcept {
    return state == ResidencyState::ram_ready || state == ResidencyState::vram_ready;
}

}  // namespace

ResidencyUse::ResidencyUse(std::shared_ptr<ResidencyControl> control) : control_(std::move(control)) {}

ResidencyUse::ResidencyUse(ResidencyUse&& other) noexcept : control_(std::move(other.control_)) {}

ResidencyUse& ResidencyUse::operator=(ResidencyUse&& other) noexcept {
    if (this != &other) {
        release();
        control_ = std::move(other.control_);
    }
    return *this;
}

ResidencyUse::~ResidencyUse() {
    release();
}

void ResidencyUse::release() noexcept {
    if (control_ != nullptr) {
        {
            std::scoped_lock lock(control_->mutex);
            if (control_->active_uses != 0U) --control_->active_uses;
        }
        control_.reset();
    }
}

ResidencyObject::ResidencyObject(std::shared_ptr<ResidencyControl> control)
    : control_(std::move(control)) {}

Result<ResidencyObject> ResidencyObject::create(std::string logical_id, ByteCount bytes) {
    if (logical_id.empty() || bytes == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Residency object requires a non-empty identity and byte count",
                          std::move(logical_id));
    }
    try {
        auto control = std::make_shared<ResidencyControl>();
        control->logical_id = std::move(logical_id);
        control->object_bytes = bytes;
        return ResidencyObject(std::move(control));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded residency control object",
                          std::move(logical_id));
    }
}

Result<std::shared_ptr<ResidencyControl>> ResidencyObject::require_control() const {
    if (control_ == nullptr) {
        return make_error(ErrorCode::invalid_argument, "Residency object has been moved from");
    }
    return control_;
}

Result<ResidencySnapshot> ResidencyObject::snapshot() const {
    auto required = require_control();
    if (!required) return required.error();
    const auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    return ResidencySnapshot{
        control.logical_id,
        control.state,
        control.object_bytes,
        control.ram.bytes(),
        control.vram.bytes(),
        control.staging.bytes(),
        control.active_uses,
        control.pinned,
        ready_state(control.state) && !control.pinned && control.active_uses == 0U,
        control.failure,
    };
}

Result<void> ResidencyObject::begin_read(safety::ResourceGovernor& governor,
                                         ByteCount staging_bytes) {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (control.state != ResidencyState::nvme) return state_error(control, "begin_read");
    if (staging_bytes == 0U || staging_bytes > control.object_bytes) {
        return make_error(ErrorCode::invalid_argument,
                          "Residency read staging bytes must be within the object size",
                          control.logical_id);
    }

    auto ram = governor.try_reserve(safety::ResourceKind::ram,
                                    control.object_bytes,
                                    control.logical_id + " RAM destination");
    if (!ram) return ram.error();
    auto staging = governor.try_reserve(safety::ResourceKind::staging,
                                        staging_bytes,
                                        control.logical_id + " bounded read staging");
    if (!staging) return staging.error();

    control.ram = ram.take_value();
    control.staging = staging.take_value();
    control.state = ResidencyState::reading;
    return {};
}

Result<void> ResidencyObject::publish_ram_ready() {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (control.state != ResidencyState::reading || !control.ram.valid() || !control.staging.valid()) {
        return state_error(control, "publish_ram_ready");
    }
    control.staging.release();
    control.state = ResidencyState::ram_ready;
    return {};
}

Result<void> ResidencyObject::begin_vram_copy(safety::ResourceGovernor& governor) {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (control.state != ResidencyState::ram_ready || !control.ram.valid() || control.active_uses != 0U) {
        return state_error(control, "begin_vram_copy");
    }
    auto vram = governor.try_reserve(safety::ResourceKind::vram,
                                     control.object_bytes,
                                     control.logical_id + " VRAM destination");
    if (!vram) return vram.error();
    control.vram = vram.take_value();
    control.state = ResidencyState::copying_to_vram;
    return {};
}

Result<void> ResidencyObject::publish_vram_ready(bool keep_ram_copy) {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (control.state != ResidencyState::copying_to_vram || !control.ram.valid() || !control.vram.valid()) {
        return state_error(control, "publish_vram_ready");
    }
    if (!keep_ram_copy) control.ram.release();
    control.state = ResidencyState::vram_ready;
    return {};
}

Result<ResidencyUse> ResidencyObject::acquire_use() {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (!ready_state(control.state) || control.active_uses == std::numeric_limits<std::size_t>::max()) {
        return state_error(control, "acquire_use");
    }
    ++control.active_uses;
    return ResidencyUse(required.take_value());
}

Result<void> ResidencyObject::set_pinned(bool pinned) {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (pinned && !ready_state(control.state)) return state_error(control, "set_pinned");
    control.pinned = pinned;
    return {};
}

Result<void> ResidencyObject::evict_ram() {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (control.pinned || control.active_uses != 0U || !control.ram.valid() ||
        (control.state != ResidencyState::ram_ready && control.state != ResidencyState::vram_ready)) {
        return state_error(control, "evict_ram");
    }
    control.ram.release();
    if (control.state == ResidencyState::ram_ready) control.state = ResidencyState::nvme;
    return {};
}

Result<void> ResidencyObject::evict_vram() {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (control.state != ResidencyState::vram_ready || control.pinned ||
        control.active_uses != 0U || !control.vram.valid()) {
        return state_error(control, "evict_vram");
    }
    control.vram.release();
    control.state = control.ram.valid() ? ResidencyState::ram_ready : ResidencyState::nvme;
    return {};
}

Result<void> ResidencyObject::fail(std::string reason) {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (reason.empty()) {
        return make_error(ErrorCode::invalid_argument, "Residency failure requires a reason", control.logical_id);
    }
    if (control.active_uses != 0U) return state_error(control, "fail");
    control.staging.release();
    control.ram.release();
    control.vram.release();
    control.pinned = false;
    control.failure = std::move(reason);
    control.state = ResidencyState::failed;
    return {};
}

Result<void> ResidencyObject::retry() {
    auto required = require_control();
    if (!required) return required.error();
    auto& control = *required.value();
    std::scoped_lock lock(control.mutex);
    if (control.state != ResidencyState::failed || control.active_uses != 0U ||
        control.ram.valid() || control.vram.valid() || control.staging.valid()) {
        return state_error(control, "retry");
    }
    control.failure.clear();
    control.state = ResidencyState::nvme;
    return {};
}

Result<std::vector<ResidencyObject>> make_expert_residencies(const LogicalTensorCatalog& catalog,
                                                              std::size_t max_experts) {
    if (catalog.model_id.empty() || catalog.experts.empty() || max_experts == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Logical catalog cannot produce a bounded expert residency table",
                          catalog.model_id);
    }
    if (catalog.experts.size() > max_experts) {
        return make_error(ErrorCode::resource_exhausted,
                          "Logical catalog exceeds the configured expert residency limit",
                          catalog.model_id);
    }
    try {
        std::vector<ResidencyObject> result;
        result.reserve(catalog.experts.size());
        ByteCount total = 0U;
        std::size_t range_count = 0U;
        for (const auto& expert : catalog.experts) {
            if (expert.bytes == 0U || expert.tensor_indices.empty() || expert.read_ranges.empty() ||
                expert.tensor_slices.size() != expert.tensor_indices.size() ||
                expert.bytes > std::numeric_limits<ByteCount>::max() - total ||
                range_count > std::numeric_limits<std::size_t>::max() - expert.read_ranges.size()) {
                return make_error(ErrorCode::invalid_format,
                                  "Logical expert bundle cannot produce a residency object",
                                  catalog.model_id);
            }
            total += expert.bytes;
            range_count += expert.read_ranges.size();
            std::string logical_id = catalog.model_id + "/";
            logical_id += expert.speculative ? "mtp/" : "layer/";
            logical_id += std::to_string(expert.layer) + "/expert/" + std::to_string(expert.expert);
            auto object = ResidencyObject::create(std::move(logical_id), expert.bytes);
            if (!object) return object.error();
            result.push_back(object.take_value());
        }
        if (total != catalog.routed_expert_bytes || range_count != catalog.expert_read_range_count) {
            return make_error(ErrorCode::integrity_error,
                              "Logical expert residency totals differ from the finalized catalog",
                              catalog.model_id);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded expert residency table",
                          catalog.model_id);
    }
}

std::string_view to_string(ResidencyState state) noexcept {
    switch (state) {
        case ResidencyState::nvme: return "NVME";
        case ResidencyState::reading: return "READING";
        case ResidencyState::ram_ready: return "RAM_READY";
        case ResidencyState::copying_to_vram: return "COPYING_TO_VRAM";
        case ResidencyState::vram_ready: return "VRAM_READY";
        case ResidencyState::failed: return "FAILED";
    }
    return "UNKNOWN";
}

}  // namespace vastloom::model
