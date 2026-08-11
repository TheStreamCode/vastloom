#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/cuda/cuda_driver.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::cuda {

inline constexpr std::size_t kCudaRmsNormMaximumElements = 4U * 1'024U * 1'024U;

struct CudaRmsNormOptions {
    int device_ordinal{0};
    ByteCount minimum_free_vram_after{1U * GiB};
};

struct CudaRmsNormStats {
    std::size_t row_count{0U};
    std::size_t width{0U};
    ByteCount device_buffer_bytes{0U};
    ByteCount host_to_device_bytes{0U};
    ByteCount device_to_host_bytes{0U};
};

struct CudaRmsNormResult {
    // Declared before output so destruction frees the vector before releasing its accounting.
    safety::Reservation host_output_reservation;
    std::vector<float> output;
    CudaRmsNormStats stats;
};

// Bounded FP32 reference kernel path. The caller owns/accounted input and weight storage; the
// returned output keeps its own Governor scratch reservation alive.
[[nodiscard]] Result<CudaRmsNormResult> cuda_rms_norm(
    CudaDriver& driver,
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t row_count,
    std::size_t width,
    float epsilon,
    safety::ResourceGovernor& governor,
    const CudaRmsNormOptions& options = {});

}  // namespace vastloom::cuda
