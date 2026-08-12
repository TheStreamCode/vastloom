#include "vastloom/cuda/cuda_matvec_accelerator.hpp"

#include <chrono>
#include <limits>
#include <new>
#include <optional>

#include "vastloom/inference/ggml_quantization.hpp"

namespace vastloom::cuda {
namespace {

constexpr std::uint32_t kGgmlTypeQ8_0 = 8U;
constexpr ByteCount kQ8BlockBytes = 34U;
constexpr std::size_t kQ8BlockElements = 32U;
constexpr std::uint32_t kGgmlTypeIq1S = 19U;
constexpr ByteCount kIq1SBlockBytes = 50U;
constexpr std::size_t kIq1SBlockElements = 256U;

struct Layout {
    ByteCount block_bytes{0U};
    std::size_t block_elements{0U};
};

std::optional<Layout> layout_for(std::uint32_t ggml_type) {
    if (ggml_type == kGgmlTypeQ8_0) return Layout{kQ8BlockBytes, kQ8BlockElements};
    if (ggml_type == kGgmlTypeIq1S) return Layout{kIq1SBlockBytes, kIq1SBlockElements};
    return std::nullopt;
}

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
    // Q8_0 and IQ1_S have device kernels. Everything else is declined rather than failed, so the
    // graph keeps producing the same numbers through the CPU reference.
    const auto layout = layout_for(matrix.ggml_type);
    if (!layout.has_value() || matrix.rows == 0U || matrix.storage_columns == 0U ||
        matrix.storage_columns % layout->block_elements != 0U ||
        input.size() != matrix.storage_columns || output.size() != matrix.rows) {
        ++stats_.declined;
        return false;
    }
    // No stable identity means there is no safe way to recognize this matrix again later, so it
    // goes to the CPU rather than into a cache that could confuse it with another.
    if (matrix.cache_identity == 0U) {
        ++stats_.declined;
        return false;
    }
    const std::uint64_t key = matrix.cache_identity;

    auto entry = resident_.find(key);
    if (entry == resident_.end()) {
        if (resident_.size() >= options_.maximum_matrices ||
            stats_.resident_bytes >= options_.maximum_resident_bytes) {
            ++stats_.declined;
            return false;
        }
        auto payload = inference::deepseek_v4_packed_matrix_payload(matrix, "cuda-matvec");
        if (!payload) return payload.error();
        const std::size_t blocks_per_row = matrix.storage_columns / layout->block_elements;
        if (blocks_per_row == 0U ||
            matrix.rows > std::numeric_limits<ByteCount>::max() / blocks_per_row) {
            ++stats_.declined;
            return false;
        }
        const ByteCount matrix_bytes =
            static_cast<ByteCount>(matrix.rows) * blocks_per_row * layout->block_bytes;
        if (payload.value().size() != matrix_bytes) {
            ++stats_.declined;
            return false;
        }
        const ByteCount input_bytes =
            static_cast<ByteCount>(matrix.storage_columns) * sizeof(float);
        const ByteCount output_bytes = static_cast<ByteCount>(matrix.rows) * sizeof(float);
        // IQ1_S decodes through a shared 16 KiB grid, which rides along in the same allocation so
        // the kernel needs no second pointer to manage.
        const auto grid = inference::iq1_s_grid_table();
        const ByteCount grid_bytes =
            matrix.ggml_type == kGgmlTypeIq1S ? grid.size_bytes() : 0U;
        const ByteCount total = matrix_bytes + grid_bytes + input_bytes + output_bytes;
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
        resident.grid_offset = matrix_bytes;
        resident.input_offset = matrix_bytes + grid_bytes;
        resident.output_offset = matrix_bytes + grid_bytes + input_bytes;
        resident.rows = matrix.rows;
        resident.blocks_per_row = blocks_per_row;
        resident.ggml_type = matrix.ggml_type;

        const auto upload_started = std::chrono::steady_clock::now();
        auto uploaded = driver_.copy_from_host(resident.allocation, 0U, payload.value());
        if (uploaded && grid_bytes != 0U) {
            uploaded = driver_.copy_from_host(resident.allocation, resident.grid_offset,
                                              std::as_bytes(grid));
        }
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

    // Upload, launch and download in one driver entry: three separate calls each push and pop the
    // context and synchronize, which costs more than the kernel itself at this call rate.
    auto& live = entry->second;
    const auto started = std::chrono::steady_clock::now();
    auto executed = live.ggml_type == kGgmlTypeIq1S
        ? driver_.matvec_iq1_s_with_transfer(
              live.allocation, 0U, live.grid_offset, live.input_offset, live.output_offset,
              live.rows, live.blocks_per_row, std::as_bytes(input),
              std::as_writable_bytes(output))
        : driver_.matvec_q8_0_with_transfer(
              live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
              live.blocks_per_row, std::as_bytes(input), std::as_writable_bytes(output));
    if (!executed) return executed.error();
    stats_.kernel_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++stats_.accelerated;
    return true;
}

}  // namespace vastloom::cuda
