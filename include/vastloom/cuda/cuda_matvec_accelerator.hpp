#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/cuda/cuda_driver.hpp"
#include "vastloom/inference/deepseek_v4_reference.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::cuda {

struct CudaMatvecAcceleratorOptions {
    int device_ordinal{0};
    ByteCount minimum_free_vram_after{1U * GiB};
    // Ceiling on resident weight bytes. Matrices beyond it stay on the CPU path rather than
    // evicting an existing one: a decode step revisits the same matrices, so thrashing here would
    // cost more than it saves.
    ByteCount maximum_resident_bytes{6U * GiB};
    std::size_t maximum_matrices{512U};
};

struct CudaMatvecAcceleratorStats {
    std::size_t offered{0U};
    std::size_t accelerated{0U};
    std::size_t declined{0U};
    std::size_t resident_matrices{0U};
    ByteCount resident_bytes{0U};
    ByteCount uploaded_bytes{0U};
    double kernel_ms{0.0};
    double upload_ms{0.0};
};

// Runs Q8_0 matvecs on the GPU, keeping each matrix resident in VRAM after its first use.
//
// Residency is what makes this worth doing: a matrix costs one upload and is then reused by every
// later token, whereas copying per launch would spend more on PCIe than the kernel saves. Anything
// this cannot handle - another storage type, a shape it does not support, no VRAM left - is
// declined so the caller runs the CPU kernel, which stays the correctness reference.
class CudaMatvecAccelerator final : public inference::DeepseekV4MatvecAccelerator {
public:
    [[nodiscard]] static Result<std::unique_ptr<CudaMatvecAccelerator>> create(
        CudaDriver driver,
        safety::ResourceGovernor& governor,
        const CudaMatvecAcceleratorOptions& options = {});

    CudaMatvecAccelerator(CudaMatvecAccelerator&&) = delete;
    CudaMatvecAccelerator& operator=(CudaMatvecAccelerator&&) = delete;
    ~CudaMatvecAccelerator() override = default;

    [[nodiscard]] Result<bool> matvec(
        const inference::DeepseekV4PackedMatrix& matrix,
        std::span<const float> input,
        std::span<float> output) override;

    [[nodiscard]] const CudaMatvecAcceleratorStats& stats() const noexcept { return stats_; }

private:
    struct Resident {
        CudaAllocation allocation;
        ByteCount matrix_bytes{0U};
        ByteCount grid_offset{0U};
        ByteCount input_offset{0U};
        ByteCount output_offset{0U};
        std::size_t rows{0U};
        std::size_t blocks_per_row{0U};
        std::uint32_t ggml_type{0U};
    };

    CudaMatvecAccelerator(CudaDriver driver,
                          safety::ResourceGovernor& governor,
                          CudaMatvecAcceleratorOptions options);

    CudaDriver driver_;
    safety::ResourceGovernor* governor_{nullptr};
    CudaMatvecAcceleratorOptions options_;
    // Keyed on the descriptor's stable identity, never on a host address: see the note on
    // DeepseekV4PackedMatrix::cache_identity for what address-keying did to the model's output.
    std::map<std::uint64_t, Resident> resident_;
    CudaMatvecAcceleratorStats stats_;
};

}  // namespace vastloom::cuda
