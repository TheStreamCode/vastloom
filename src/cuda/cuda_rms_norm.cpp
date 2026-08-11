#include "vastloom/cuda/cuda_rms_norm.hpp"

#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace vastloom::cuda {
namespace {

Result<ByteCount> checked_bytes(std::size_t elements, std::string_view label) {
    if (elements > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
        return make_error(ErrorCode::resource_exhausted,
                          "CUDA RMSNorm byte count overflows", std::string(label));
    }
    return static_cast<ByteCount>(elements) * sizeof(float);
}

Result<ByteCount> checked_add(ByteCount left, ByteCount right, std::string_view label) {
    if (right > std::numeric_limits<ByteCount>::max() - left) {
        return make_error(ErrorCode::resource_exhausted,
                          "CUDA RMSNorm packed layout overflows", std::string(label));
    }
    return left + right;
}

}  // namespace

Result<CudaRmsNormResult> cuda_rms_norm(
    CudaDriver& driver,
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t row_count,
    std::size_t width,
    float epsilon,
    safety::ResourceGovernor& governor,
    const CudaRmsNormOptions& options) {
    if (row_count == 0U || width == 0U || options.device_ordinal < 0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA RMSNorm dimensions, device or epsilon are invalid");
    }
    if (width > std::numeric_limits<std::size_t>::max() / row_count) {
        return make_error(ErrorCode::resource_exhausted,
                          "CUDA RMSNorm element count overflows");
    }
    const std::size_t element_count = row_count * width;
    if (element_count > kCudaRmsNormMaximumElements) {
        return make_error(ErrorCode::resource_exhausted,
                          "CUDA RMSNorm exceeds the bounded reference element limit",
                          std::to_string(element_count));
    }
    if (input.size() != element_count || weight.size() != width) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA RMSNorm input or weight shape is invalid");
    }
    for (const float value : input) {
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA RMSNorm input contains a non-finite value");
        }
    }
    for (const float value : weight) {
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA RMSNorm weight contains a non-finite value");
        }
    }

    auto matrix_bytes = checked_bytes(element_count, "input/output");
    auto weight_bytes = checked_bytes(width, "weight");
    if (!matrix_bytes) return matrix_bytes.error();
    if (!weight_bytes) return weight_bytes.error();
    const ByteCount input_offset = 0U;
    const ByteCount weight_offset = matrix_bytes.value();
    auto output_offset = checked_add(weight_offset, weight_bytes.value(), "output offset");
    if (!output_offset) return output_offset.error();
    auto device_bytes = checked_add(output_offset.value(), matrix_bytes.value(), "device buffer");
    if (!device_bytes) return device_bytes.error();

    auto host_reservation = governor.try_reserve(
        safety::ResourceKind::scratch, matrix_bytes.value(), "cuda-rmsnorm-output");
    if (!host_reservation) return host_reservation.error();
    std::vector<float> output;
    try {
        output.resize(element_count);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "CUDA RMSNorm host output allocation failed");
    } catch (const std::length_error&) {
        return make_error(ErrorCode::resource_exhausted,
                          "CUDA RMSNorm host output length is invalid");
    }

    auto device = driver.allocate(options.device_ordinal, device_bytes.value(),
                                  options.minimum_free_vram_after, governor,
                                  "cuda-rmsnorm-packed");
    if (!device) return device.error();
    auto allocation = device.take_value();
    auto copied_input = driver.copy_from_host(
        allocation, input_offset, std::as_bytes(input));
    if (!copied_input) return copied_input.error();
    auto copied_weight = driver.copy_from_host(
        allocation, weight_offset, std::as_bytes(weight));
    if (!copied_weight) return copied_weight.error();
    auto launched = driver.launch_rms_norm(
        allocation, input_offset, weight_offset, output_offset.value(), row_count, width,
        epsilon);
    if (!launched) return launched.error();
    auto copied_output = driver.copy_to_host(
        allocation, output_offset.value(),
        std::as_writable_bytes(std::span<float>(output)));
    if (!copied_output) return copied_output.error();
    for (const float value : output) {
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::integrity_error,
                              "CUDA RMSNorm kernel produced a non-finite output");
        }
    }
    auto released = allocation.release();
    if (!released) return released.error();

    CudaRmsNormStats stats;
    stats.row_count = row_count;
    stats.width = width;
    stats.device_buffer_bytes = device_bytes.value();
    stats.host_to_device_bytes = matrix_bytes.value() + weight_bytes.value();
    stats.device_to_host_bytes = matrix_bytes.value();
    return CudaRmsNormResult{host_reservation.take_value(), std::move(output), stats};
}

}  // namespace vastloom::cuda
