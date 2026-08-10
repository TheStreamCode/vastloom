#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::model {

enum class ResidencyState : std::uint8_t {
    nvme,
    reading,
    ram_ready,
    copying_to_vram,
    vram_ready,
    failed,
};

struct ResidencySnapshot {
    std::string logical_id;
    ResidencyState state{ResidencyState::nvme};
    ByteCount object_bytes{0};
    ByteCount ram_reserved_bytes{0};
    ByteCount vram_reserved_bytes{0};
    ByteCount staging_reserved_bytes{0};
    std::size_t active_uses{0};
    bool pinned{false};
    bool evictable{false};
    std::string failure;
};

struct ResidencyControl;

class ResidencyUse {
public:
    ResidencyUse() = default;
    ResidencyUse(const ResidencyUse&) = delete;
    ResidencyUse& operator=(const ResidencyUse&) = delete;
    ResidencyUse(ResidencyUse&& other) noexcept;
    ResidencyUse& operator=(ResidencyUse&& other) noexcept;
    ~ResidencyUse();

    [[nodiscard]] bool valid() const noexcept { return control_ != nullptr; }
    void release() noexcept;

private:
    friend class ResidencyObject;
    explicit ResidencyUse(std::shared_ptr<ResidencyControl> control);

    std::shared_ptr<ResidencyControl> control_;
};

class ResidencyObject {
public:
    // Residency transitions reserve capacity but do not allocate or expose tensor buffers.
    // Any ResourceGovernor passed to a transition must outlive every copy of this object
    // and every ResidencyUse lease that can retain its reservations.
    [[nodiscard]] static Result<ResidencyObject> create(std::string logical_id, ByteCount bytes);

    ResidencyObject(const ResidencyObject&) = default;
    ResidencyObject& operator=(const ResidencyObject&) = default;
    ResidencyObject(ResidencyObject&&) noexcept = default;
    ResidencyObject& operator=(ResidencyObject&&) noexcept = default;

    [[nodiscard]] Result<ResidencySnapshot> snapshot() const;
    [[nodiscard]] Result<void> begin_read(safety::ResourceGovernor& governor,
                                          ByteCount staging_bytes);
    [[nodiscard]] Result<void> publish_ram_ready();
    [[nodiscard]] Result<void> begin_vram_copy(safety::ResourceGovernor& governor);
    [[nodiscard]] Result<void> publish_vram_ready(bool keep_ram_copy);
    [[nodiscard]] Result<ResidencyUse> acquire_use();
    [[nodiscard]] Result<void> set_pinned(bool pinned);
    [[nodiscard]] Result<void> evict_ram();
    [[nodiscard]] Result<void> evict_vram();
    [[nodiscard]] Result<void> fail(std::string reason);
    [[nodiscard]] Result<void> retry();

private:
    explicit ResidencyObject(std::shared_ptr<ResidencyControl> control);
    [[nodiscard]] Result<std::shared_ptr<ResidencyControl>> require_control() const;

    std::shared_ptr<ResidencyControl> control_;
};

[[nodiscard]] Result<std::vector<ResidencyObject>> make_expert_residencies(
    const LogicalTensorCatalog& catalog,
    std::size_t max_experts = 1'000'000U);

[[nodiscard]] std::string_view to_string(ResidencyState state) noexcept;

}  // namespace vastloom::model
