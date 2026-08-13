#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::cuda {

struct CudaDeviceCapabilities {
    int ordinal{0};
    std::string name;
    ByteCount total_vram_bytes{0U};
    int compute_capability_major{0};
    int compute_capability_minor{0};
    int multiprocessor_count{0};
    int maximum_threads_per_block{0};
    bool unified_addressing{false};
};

struct CudaDriverProbe {
    std::string library;
    int driver_version{0};
    std::vector<CudaDeviceCapabilities> devices;
};

struct CudaBackendAllocation {
    std::uintptr_t context{0U};
    std::uint64_t device_pointer{0U};
    ByteCount bytes{0U};
    int device_ordinal{0};
};

// Backend seam used by the native dynamic CUDA implementation and deterministic fault tests.
// Runtime callers should construct CudaDriver through load().
class CudaDriverBackend {
public:
    virtual ~CudaDriverBackend() = default;

    [[nodiscard]] virtual Result<CudaDriverProbe> probe() = 0;
    [[nodiscard]] virtual Result<CudaBackendAllocation> allocate(
        int device_ordinal,
        ByteCount bytes,
        ByteCount minimum_free_vram_after) = 0;
    [[nodiscard]] virtual Result<void> release(const CudaBackendAllocation& allocation) = 0;
    [[nodiscard]] virtual Result<void> copy_from_host(
        const CudaBackendAllocation& allocation,
        ByteCount destination_offset,
        std::span<const std::byte> source) = 0;
    [[nodiscard]] virtual Result<void> copy_to_host(
        const CudaBackendAllocation& allocation,
        ByteCount source_offset,
        std::span<std::byte> destination) = 0;
    [[nodiscard]] virtual Result<void> launch_rms_norm(
        const CudaBackendAllocation& allocation,
        ByteCount input_offset,
        ByteCount weight_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t width,
        float epsilon) = 0;
    // One thread per output row over a ggml Q8_0 matrix: 34-byte blocks of one FP16 scale and
    // 32 int8 weights. Blocks accumulate in the CPU reference's order but in FP32, so the result
    // matches to a declared tolerance rather than bit for bit.
    [[nodiscard]] virtual Result<void> launch_matvec_q8_0(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row) = 0;
    // Upload, launch and download inside one context push. Called thousands of times per generated
    // token, where three separate entries cost more in round trips than the kernel does in
    // arithmetic.
    // IQ1_S needs its 2,048-entry grid in device memory; the caller places it in the same
    // allocation and passes its offset.
    [[nodiscard]] virtual Result<void> matvec_iq1_s_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    // IQ2_XXS carries a 256-entry 64-bit grid, placed in the allocation the same way.
    [[nodiscard]] virtual Result<void> matvec_iq2_xxs_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    // Q4_K super-blocks: 144 bytes per 256 weights. MXFP4 is 17 bytes per 32, with an E8M0 scale.
    [[nodiscard]] virtual Result<void> matvec_q4_k_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    [[nodiscard]] virtual Result<void> matvec_mxfp4_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    // Q6_K super-blocks: 210 bytes per 256 weights, no shared grid.
    [[nodiscard]] virtual Result<void> matvec_q6_k_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    // Q5_K super-blocks: 176 bytes per 256 weights, no shared grid.
    [[nodiscard]] virtual Result<void> matvec_q5_k_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    // Dense little-endian rows with no block structure: 4 bytes per element for FP32, 2 for BF16.
    [[nodiscard]] virtual Result<void> matvec_dense_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t columns,
        ByteCount element_bytes,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    // IQ3_XXS carries a 256-entry grid, placed in the allocation the same way.
    [[nodiscard]] virtual Result<void> matvec_iq3_xxs_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
    [[nodiscard]] virtual Result<void> matvec_q8_0_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) = 0;
};

class CudaAllocation {
public:
    CudaAllocation() = default;
    CudaAllocation(const CudaAllocation&) = delete;
    CudaAllocation& operator=(const CudaAllocation&) = delete;
    CudaAllocation(CudaAllocation&& other) noexcept;
    CudaAllocation& operator=(CudaAllocation&& other) noexcept;
    ~CudaAllocation();

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t device_pointer() const noexcept;
    [[nodiscard]] ByteCount bytes() const noexcept;
    [[nodiscard]] int device_ordinal() const noexcept;

    // Explicit release reports CUDA cleanup errors. If cleanup fails, the allocation and its
    // Governor reservation remain owned so the caller may retry safely.
    [[nodiscard]] Result<void> release();

private:
    friend class CudaDriver;
    CudaAllocation(std::shared_ptr<CudaDriverBackend> backend,
                   CudaBackendAllocation allocation,
                   safety::Reservation reservation,
                   safety::Reservation host_reservation);
    void release_for_destruction() noexcept;

    std::shared_ptr<CudaDriverBackend> backend_;
    CudaBackendAllocation allocation_;
    safety::Reservation reservation_;
    // Device memory is not free of host memory where WDDM charges commit for it. Measured on the
    // reference machine at one for one - 6 GB of VRAM cost 6.26 GB of host commit - so the same
    // bytes are reserved twice, once against each budget. Without it the Governor approves a device
    // allocation the host cannot back, and the failure surfaces later in whatever asks for host
    // memory next: the tile loader died on bad_alloc after a preflight that had said yes.
    safety::Reservation host_reservation_;
};

class CudaDriver {
public:
    [[nodiscard]] static Result<CudaDriver> load();

    // Explicit test seam. Production code must use load(), which binds the platform driver.
    [[nodiscard]] static Result<CudaDriver> from_backend_for_testing(
        std::shared_ptr<CudaDriverBackend> backend);

    [[nodiscard]] Result<CudaDriverProbe> probe() const;

    // The VRAM reservation is acquired before the backend creates a CUDA allocation. The
    // backend independently checks live device headroom after making the device context current.
    [[nodiscard]] Result<CudaAllocation> allocate(
        int device_ordinal,
        ByteCount bytes,
        ByteCount minimum_free_vram_after,
        safety::ResourceGovernor& governor,
        std::string tag) const;

    [[nodiscard]] Result<void> copy_from_host(
        CudaAllocation& allocation,
        ByteCount destination_offset,
        std::span<const std::byte> source) const;
    [[nodiscard]] Result<void> copy_to_host(
        CudaAllocation& allocation,
        ByteCount source_offset,
        std::span<std::byte> destination) const;
    [[nodiscard]] Result<void> launch_rms_norm(
        CudaAllocation& allocation,
        ByteCount input_offset,
        ByteCount weight_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t width,
        float epsilon) const;
    [[nodiscard]] Result<void> launch_matvec_q8_0(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row) const;
    [[nodiscard]] Result<void> matvec_iq1_s_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_iq2_xxs_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_q4_k_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_mxfp4_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_q6_k_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_q5_k_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_dense_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t columns,
        ByteCount element_bytes,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_iq3_xxs_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;
    [[nodiscard]] Result<void> matvec_q8_0_with_transfer(
        CudaAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) const;

private:
    explicit CudaDriver(std::shared_ptr<CudaDriverBackend> backend);

    std::shared_ptr<CudaDriverBackend> backend_;
};

[[nodiscard]] std::string cuda_driver_version_text(int encoded_version);
[[nodiscard]] std::string to_json(const CudaDriverProbe& probe);
[[nodiscard]] std::string to_text(const CudaDriverProbe& probe);

}  // namespace vastloom::cuda
