#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/io/device_io_counters.hpp"
#include "vastloom/io/process_io_counters.hpp"
#include "vastloom/model/residency.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::model {

inline constexpr ByteCount kMaximumBoundRangeLoadBytes = 1U * GiB;

struct BoundRangeIdentity {
    std::string logical_id;
    TensorSourceRange source;
    std::string sha256;
};

struct BoundRangeLoadOptions {
    ByteCount max_range_bytes{64U * MiB};
    bool measure_process_io{true};
    bool measure_device_io{false};
    std::function<bool()> cancelled;
};

struct BoundRangeLoadTelemetry {
    ByteCount logical_bytes{0U};
    double elapsed_ms{0.0};
    std::optional<io::ProcessIoDelta> process_io;
    std::optional<io::DeviceIoDelta> device_io;
};

class LoadedBoundRange {
public:
    LoadedBoundRange(const LoadedBoundRange&) = delete;
    LoadedBoundRange& operator=(const LoadedBoundRange&) = delete;
    LoadedBoundRange(LoadedBoundRange&&) noexcept = default;
    LoadedBoundRange& operator=(LoadedBoundRange&&) = delete;

    [[nodiscard]] const BoundRangeIdentity& identity() const noexcept { return identity_; }
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return buffer_ == nullptr ? std::span<const std::byte>{} : buffer_->bytes();
    }
    [[nodiscard]] std::shared_ptr<const ResidentByteBuffer> retain_buffer() const noexcept {
        return buffer_;
    }
    [[nodiscard]] const BoundRangeLoadTelemetry& telemetry() const noexcept { return telemetry_; }

private:
    friend Result<LoadedBoundRange> load_bound_range(const BoundRangeIdentity&,
                                                     ResidencyObject&,
                                                     safety::ResourceGovernor&,
                                                     const BoundRangeLoadOptions&);
    LoadedBoundRange(ResidencyUse lease,
                     BoundRangeIdentity identity,
                     std::vector<std::byte> bytes,
                     BoundRangeLoadTelemetry telemetry);

    BoundRangeIdentity identity_;
    std::shared_ptr<const ResidentByteBuffer> buffer_;
    BoundRangeLoadTelemetry telemetry_;
};

// Loads one already-verified immutable local range through the standard NVMe -> READING ->
// RAM_READY transaction. Architecture-specific callers remain responsible for proving that the
// range identity came from a trusted catalog binding before invoking this bounded primitive.
[[nodiscard]] Result<LoadedBoundRange> load_bound_range(
    const BoundRangeIdentity& identity,
    ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const BoundRangeLoadOptions& options = {});

}  // namespace vastloom::model
