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
                   safety::Reservation reservation);
    void release_for_destruction() noexcept;

    std::shared_ptr<CudaDriverBackend> backend_;
    CudaBackendAllocation allocation_;
    safety::Reservation reservation_;
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

private:
    explicit CudaDriver(std::shared_ptr<CudaDriverBackend> backend);

    std::shared_ptr<CudaDriverBackend> backend_;
};

[[nodiscard]] std::string cuda_driver_version_text(int encoded_version);
[[nodiscard]] std::string to_json(const CudaDriverProbe& probe);
[[nodiscard]] std::string to_text(const CudaDriverProbe& probe);

}  // namespace vastloom::cuda
