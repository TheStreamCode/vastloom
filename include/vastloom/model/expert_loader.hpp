#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
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
};

struct ExpertLoadTelemetry {
    ByteCount logical_bytes{0};
    std::size_t range_count{0};
    double elapsed_ms{0.0};
    std::optional<io::ProcessIoDelta> process_io;
    std::optional<io::DeviceIoDelta> device_io;
};

class LoadedExpert {
public:
    LoadedExpert(const LoadedExpert&) = delete;
    LoadedExpert& operator=(const LoadedExpert&) = delete;
    LoadedExpert(LoadedExpert&&) noexcept = default;
    LoadedExpert& operator=(LoadedExpert&&) = delete;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }
    [[nodiscard]] Result<std::span<const std::byte>> tensor(std::size_t catalog_tensor_index) const;
    [[nodiscard]] std::span<const ExpertTensorSlice> tensor_slices() const noexcept { return tensor_slices_; }
    [[nodiscard]] const ExpertLoadTelemetry& telemetry() const noexcept { return telemetry_; }

private:
    friend Result<LoadedExpert> load_expert_bundle(const ExpertBundle&,
                                                    ResidencyObject&,
                                                    safety::ResourceGovernor&,
                                                    const ExpertLoadOptions&);
    LoadedExpert(ResidencyUse lease,
                 std::vector<std::byte> bytes,
                 std::vector<ExpertTensorSlice> tensor_slices,
                 ExpertLoadTelemetry telemetry);

    // Reverse destruction releases the physical bytes before the lease can make the
    // associated Governor reservation evictable.
    ResidencyUse lease_;
    std::vector<std::byte> bytes_;
    std::vector<ExpertTensorSlice> tensor_slices_;
    ExpertLoadTelemetry telemetry_;
};

[[nodiscard]] Result<LoadedExpert> load_expert_bundle(
    const ExpertBundle& bundle,
    ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const ExpertLoadOptions& options = {});

}  // namespace vastloom::model
