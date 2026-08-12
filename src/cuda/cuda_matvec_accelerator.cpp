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
constexpr std::uint32_t kGgmlTypeIq3Xxs = 18U;
constexpr ByteCount kIq3XxsBlockBytes = 98U;
constexpr std::size_t kIq3XxsBlockElements = 256U;
// Dense formats have no block structure; one element is its own block.
constexpr std::uint32_t kGgmlTypeF32 = 0U;
constexpr std::uint32_t kGgmlTypeBf16 = 30U;
constexpr std::uint32_t kGgmlTypeQ4K = 12U;
constexpr ByteCount kQ4KBlockBytes = 144U;
constexpr std::size_t kQ4KBlockElements = 256U;
constexpr std::uint32_t kGgmlTypeMxfp4 = 39U;
constexpr ByteCount kMxfp4BlockBytes = 17U;
constexpr std::size_t kMxfp4BlockElements = 32U;
constexpr std::uint32_t kGgmlTypeQ6K = 14U;
constexpr ByteCount kQ6KBlockBytes = 210U;
constexpr std::size_t kQ6KBlockElements = 256U;
constexpr std::uint32_t kGgmlTypeQ5K = 13U;
constexpr ByteCount kQ5KBlockBytes = 176U;
constexpr std::size_t kQ5KBlockElements = 256U;
constexpr std::uint32_t kGgmlTypeIq2Xxs = 16U;
constexpr ByteCount kIq2XxsBlockBytes = 66U;
constexpr std::size_t kIq2XxsBlockElements = 256U;

struct Layout {
    ByteCount block_bytes{0U};
    std::size_t block_elements{0U};
};

std::optional<Layout> layout_for(std::uint32_t ggml_type) {
    if (ggml_type == kGgmlTypeQ8_0) return Layout{kQ8BlockBytes, kQ8BlockElements};
    if (ggml_type == kGgmlTypeIq1S) return Layout{kIq1SBlockBytes, kIq1SBlockElements};
    if (ggml_type == kGgmlTypeIq3Xxs) return Layout{kIq3XxsBlockBytes, kIq3XxsBlockElements};
    if (ggml_type == kGgmlTypeIq2Xxs) return Layout{kIq2XxsBlockBytes, kIq2XxsBlockElements};
    if (ggml_type == kGgmlTypeQ5K) return Layout{kQ5KBlockBytes, kQ5KBlockElements};
    if (ggml_type == kGgmlTypeQ6K) return Layout{kQ6KBlockBytes, kQ6KBlockElements};
    if (ggml_type == kGgmlTypeQ4K) return Layout{kQ4KBlockBytes, kQ4KBlockElements};
    if (ggml_type == kGgmlTypeMxfp4) return Layout{kMxfp4BlockBytes, kMxfp4BlockElements};
    if (ggml_type == kGgmlTypeF32) return Layout{4U, 1U};
    if (ggml_type == kGgmlTypeBf16) return Layout{2U, 1U};
    return std::nullopt;
}

// Bytes of decode table this storage type needs beside its weights, and the table itself.
std::span<const std::byte> grid_for(std::uint32_t ggml_type) {
    if (ggml_type == kGgmlTypeIq1S) return std::as_bytes(inference::iq1_s_grid_table());
    if (ggml_type == kGgmlTypeIq3Xxs) return std::as_bytes(inference::iq3_xxs_grid_table());
    if (ggml_type == kGgmlTypeIq2Xxs) return std::as_bytes(inference::iq2_xxs_grid_table());
    return {};
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

// Returns the resident entry for this matrix, uploading it on first sight. A null pointer means the
// accelerator will not take it - unsupported type, no stable identity, or no VRAM left - which is a
// refusal rather than a failure, so the caller runs the CPU reference instead.
Result<CudaMatvecAccelerator::Resident*> CudaMatvecAccelerator::ensure_resident(
    const inference::DeepseekV4PackedMatrix& matrix) {
    const auto layout = layout_for(matrix.ggml_type);
    if (!layout.has_value()) {
        try {
            ++stats_.declined_by_type[matrix.ggml_type];
        } catch (const std::bad_alloc&) {
            // Losing a statistic is not a reason to fail a forward pass.
        }
        return nullptr;
    }
    if (matrix.rows == 0U || matrix.storage_columns == 0U ||
        matrix.storage_columns % layout->block_elements != 0U) {
        return nullptr;
    }
    // No stable identity means there is no safe way to recognize this matrix again later, so it
    // goes to the CPU rather than into a cache that could confuse it with another.
    if (matrix.cache_identity == 0U) return nullptr;

    auto entry = resident_.find(matrix.cache_identity);
    if (entry != resident_.end()) {
        entry->second.last_used = ++clock_;
        return &entry->second;
    }
    auto payload = inference::deepseek_v4_packed_matrix_payload(matrix, "cuda-matvec");
    if (!payload) return payload.error();
    const std::size_t blocks_per_row = matrix.storage_columns / layout->block_elements;
    if (blocks_per_row == 0U ||
        matrix.rows > std::numeric_limits<ByteCount>::max() / blocks_per_row) {
        return nullptr;
    }
    const ByteCount matrix_bytes =
        static_cast<ByteCount>(matrix.rows) * blocks_per_row * layout->block_bytes;
    if (payload.value().size() != matrix_bytes) return nullptr;

    const ByteCount input_bytes = static_cast<ByteCount>(matrix.storage_columns) * sizeof(float);
    const ByteCount output_bytes = static_cast<ByteCount>(matrix.rows) * sizeof(float);
    // The IQ formats decode through a shared grid, which rides along in the same allocation so the
    // kernel needs no second pointer to manage.
    const auto grid = grid_for(matrix.ggml_type);
    const auto grid_bytes = static_cast<ByteCount>(grid.size());
    // The grid is read with 64-bit loads, which fault on a misaligned address. Block strides of 50
    // and 66 bytes leave the matrix ending wherever they please, so the grid starts at the next
    // multiple of eight rather than immediately after it.
    constexpr ByteCount kGridAlignment = 8U;
    const ByteCount grid_offset =
        (matrix_bytes + kGridAlignment - 1U) / kGridAlignment * kGridAlignment;
    const ByteCount total = grid_offset + grid_bytes + input_bytes + output_bytes;
    if (total > options_.maximum_resident_bytes) return nullptr;

    // Make room by releasing the least recently used matrices rather than refusing this one.
    //
    // Refusing was the original behaviour and it was badly wrong: whatever arrived first kept the
    // space forever, and every later matrix ran on the CPU. The arithmetic says otherwise - a
    // routed expert costs about 0.29 ms to upload and 0.88 ms to multiply against roughly 8 ms on
    // twelve CPU threads, so re-uploading a matrix on every use still wins by around sevenfold.
    // Space is worth spending on whatever is being used now, not on whatever was used first.
    while ((stats_.resident_bytes + total > options_.maximum_resident_bytes ||
            resident_.size() >= options_.maximum_matrices) &&
           !resident_.empty()) {
        auto victim = std::min_element(
            resident_.begin(), resident_.end(),
            [](const auto& left, const auto& right) {
                return left.second.last_used < right.second.last_used;
            });
        const ByteCount reclaimed = victim->second.total_bytes;
        auto released = victim->second.allocation.release();
        if (!released) return released.error();
        resident_.erase(victim);
        stats_.resident_bytes -= std::min(reclaimed, stats_.resident_bytes);
        ++stats_.evicted;
    }
    if (total > options_.maximum_resident_bytes - stats_.resident_bytes) return nullptr;

    auto allocation = driver_.allocate(options_.device_ordinal, total,
                                       options_.minimum_free_vram_after, *governor_,
                                       "cuda-matvec-resident");
    // Out of VRAM is a reason to stop offering, not to fail the forward pass.
    if (!allocation) return nullptr;

    Resident resident;
    resident.allocation = allocation.take_value();
    resident.total_bytes = total;
    resident.last_used = ++clock_;
    resident.matrix_bytes = matrix_bytes;
    resident.grid_offset = grid_offset;
    resident.input_offset = grid_offset + grid_bytes;
    resident.output_offset = grid_offset + grid_bytes + input_bytes;
    resident.rows = matrix.rows;
    resident.blocks_per_row = blocks_per_row;
    resident.ggml_type = matrix.ggml_type;

    const auto upload_started = std::chrono::steady_clock::now();
    auto uploaded = driver_.copy_from_host(resident.allocation, 0U, payload.value());
    if (uploaded && grid_bytes != 0U) {
        uploaded = driver_.copy_from_host(resident.allocation, resident.grid_offset, grid);
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
        entry = resident_.emplace(matrix.cache_identity, std::move(resident)).first;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to record a resident CUDA matrix");
    }
    stats_.resident_matrices = resident_.size();
    return &entry->second;
}

Result<bool> CudaMatvecAccelerator::matvec(
    const inference::DeepseekV4PackedMatrix& matrix,
    std::span<const float> input,
    std::span<float> output) {
    ++stats_.offered;
    if (input.size() != matrix.storage_columns || output.size() != matrix.rows) {
        ++stats_.declined;
        return false;
    }
    auto made = ensure_resident(matrix);
    if (!made) return made.error();
    if (made.value() == nullptr) {
        ++stats_.declined;
        return false;
    }

    // Upload, launch and download in one driver entry: three separate calls each push and pop the
    // context and synchronize, which costs more than the kernel itself at this call rate.
    auto& live = *made.value();
    const auto started = std::chrono::steady_clock::now();
    auto executed = [&] {
        switch (live.ggml_type) {
            case kGgmlTypeIq1S:
                return driver_.matvec_iq1_s_with_transfer(
                    live.allocation, 0U, live.grid_offset, live.input_offset, live.output_offset,
                    live.rows, live.blocks_per_row, std::as_bytes(input),
                    std::as_writable_bytes(output));
            case kGgmlTypeIq3Xxs:
                return driver_.matvec_iq3_xxs_with_transfer(
                    live.allocation, 0U, live.grid_offset, live.input_offset, live.output_offset,
                    live.rows, live.blocks_per_row, std::as_bytes(input),
                    std::as_writable_bytes(output));
            case kGgmlTypeIq2Xxs:
                return driver_.matvec_iq2_xxs_with_transfer(
                    live.allocation, 0U, live.grid_offset, live.input_offset, live.output_offset,
                    live.rows, live.blocks_per_row, std::as_bytes(input),
                    std::as_writable_bytes(output));
            case kGgmlTypeQ4K:
                return driver_.matvec_q4_k_with_transfer(
                    live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
                    live.blocks_per_row, std::as_bytes(input), std::as_writable_bytes(output));
            case kGgmlTypeMxfp4:
                return driver_.matvec_mxfp4_with_transfer(
                    live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
                    live.blocks_per_row, std::as_bytes(input), std::as_writable_bytes(output));
            case kGgmlTypeQ6K:
                return driver_.matvec_q6_k_with_transfer(
                    live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
                    live.blocks_per_row, std::as_bytes(input), std::as_writable_bytes(output));
            case kGgmlTypeQ5K:
                return driver_.matvec_q5_k_with_transfer(
                    live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
                    live.blocks_per_row, std::as_bytes(input), std::as_writable_bytes(output));
            case kGgmlTypeF32:
            case kGgmlTypeBf16:
                // blocks_per_row is the column count for these: one element per block.
                return driver_.matvec_dense_with_transfer(
                    live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
                    live.blocks_per_row, live.ggml_type == kGgmlTypeF32 ? 4U : 2U,
                    std::as_bytes(input), std::as_writable_bytes(output));
            default:
                return driver_.matvec_q8_0_with_transfer(
                    live.allocation, 0U, live.input_offset, live.output_offset, live.rows,
                    live.blocks_per_row, std::as_bytes(input), std::as_writable_bytes(output));
        }
    }();
    if (!executed) return executed.error();
    stats_.kernel_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++stats_.accelerated;
    return true;
}

}  // namespace vastloom::cuda
