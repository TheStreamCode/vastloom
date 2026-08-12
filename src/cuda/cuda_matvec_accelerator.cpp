#include "vastloom/cuda/cuda_matvec_accelerator.hpp"

#include <chrono>
#include <limits>
#include <new>

#include "vastloom/inference/ggml_quantization.hpp"

namespace vastloom::cuda {
namespace {

constexpr std::uint32_t kGgmlTypeQ8_0 = 8U;
constexpr ByteCount kQ8BlockBytes = 34U;
constexpr std::size_t kQ8BlockElements = 32U;

}  // namespace

CudaMatvecAccelerator::CudaMatvecAccelerator(CudaDriver driver,
                                             safety::ResourceGovernor& governor,
                                             CudaMatvecAcceleratorOptions options)
    : driver_(std::move(driver)), governor_(&governor), options_(options) {}

Result<std::unique_ptr<CudaMatvecAccelerator>> CudaMatvecAccelerator::create(
    CudaDriver driver,
    safety::ResourceGovernor& governor,
    const CudaMatvecAcceleratorOptions& options) {
    if (options.device_ordinal < 0 || options.maximum_resident_bytes == 0U ||
        options.maximum_matrices == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA matvec accelerator options are invalid");
    }
    auto probe = driver.probe();
    if (!probe) return probe.error();
    if (probe.value().devices.empty()) {
        return make_error(ErrorCode::not_found,
                          "CUDA matvec acceleration requires at least one device");
    }
    try {
        return std::unique_ptr<CudaMatvecAccelerator>(
            new CudaMatvecAccelerator(std::move(driver), governor, options));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the CUDA matvec accelerator");
    }
}

Result<bool> CudaMatvecAccelerator::matvec(
    const inference::DeepseekV4PackedMatrix& matrix,
    std::span<const float> input,
    std::span<float> output) {
    ++stats_.offered;
    // Only Q8_0 has a device kernel today. Everything else is declined rather than failed, so the
    // graph keeps producing the same numbers through the CPU reference.
    if (matrix.ggml_type != kGgmlTypeQ8_0 || matrix.rows == 0U ||
        matrix.storage_columns == 0U || matrix.storage_columns % kQ8BlockElements != 0U ||
        input.size() != matrix.storage_columns || output.size() != matrix.rows) {
        ++stats_.declined;
        return false;
    }
    const auto* key_pointer = matrix.resident == nullptr
        ? static_cast<const void*>(matrix.encoded.data())
        : static_cast<const void*>(matrix.resident.get());
    const auto key = std::make_pair(key_pointer, static_cast<ByteCount>(matrix.resident_offset));

    auto entry = resident_.find(key);
    if (entry == resident_.end()) {
        if (resident_.size() >= options_.maximum_matrices ||
            stats_.resident_bytes >= options_.maximum_resident_bytes) {
            ++stats_.declined;
            return false;
        }
        auto payload = inference::deepseek_v4_packed_matrix_payload(matrix, "cuda-matvec");
        if (!payload) return payload.error();
        const std::size_t blocks_per_row = matrix.storage_columns / kQ8BlockElements;
        if (blocks_per_row == 0U ||
            matrix.rows > std::numeric_limits<ByteCount>::max() / blocks_per_row) {
            ++stats_.declined;
            return false;
        }
        const ByteCount matrix_bytes =
            static_cast<ByteCount>(matrix.rows) * blocks_per_row * kQ8BlockBytes;
        if (payload.value().size() != matrix_bytes) {
            ++stats_.declined;
            return false;
        }
        const ByteCount input_bytes =
            static_cast<ByteCount>(matrix.storage_columns) * sizeof(float);
        const ByteCount output_bytes = static_cast<ByteCount>(matrix.rows) * sizeof(float);
        const ByteCount total = matrix_bytes + input_bytes + output_bytes;
        if (total > options_.maximum_resident_bytes - stats_.resident_bytes) {
            ++stats_.declined;
            return false;
        }

        auto allocation = driver_.allocate(options_.device_ordinal, total,
                                           options_.minimum_free_vram_after, *governor_,
                                           "cuda-matvec-resident");
        if (!allocation) {
            // Out of VRAM is a reason to stop offering, not to fail the forward pass.
            ++stats_.declined;
            return false;
        }
        Resident resident;
        resident.allocation = allocation.take_value();
        resident.matrix_bytes = matrix_bytes;
        resident.input_offset = matrix_bytes;
        resident.output_offset = matrix_bytes + input_bytes;
        resident.rows = matrix.rows;
        resident.blocks_per_row = blocks_per_row;

        const auto upload_started = std::chrono::steady_clock::now();
        auto uploaded = driver_.copy_from_host(resident.allocation, 0U, payload.value());
        stats_.upload_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - upload_started).count();
        if (!uploaded) {
            auto released = resident.allocation.release();
            if (!released) return released.error();
            return uploaded.error();
        }
        stats_.uploaded_bytes += matrix_bytes;
        stats_.resident_bytes += total;
        try {
            entry = resident_.emplace(key, std::move(resident)).first;
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to record a resident CUDA matrix");
        }
        stats_.resident_matrices = resident_.size();
    }

    auto& live = entry->second;
    auto copied_input = driver_.copy_from_host(
        live.allocation, live.input_offset, std::as_bytes(input));
    if (!copied_input) return copied_input.error();
    const auto started = std::chrono::steady_clock::now();
    auto launched = driver_.launch_matvec_q8_0(
        live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
        live.blocks_per_row);
    if (!launched) return launched.error();
    stats_.kernel_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    auto copied_output = driver_.copy_to_host(
        live.allocation, live.output_offset, std::as_writable_bytes(output));
    if (!copied_output) return copied_output.error();
    ++stats_.accelerated;
    return true;
}

}  // namespace vastloom::cuda
