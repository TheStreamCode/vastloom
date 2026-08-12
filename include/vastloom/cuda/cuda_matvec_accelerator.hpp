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
    std::size_t evicted{0U};
    // Declines that happened because the storage type has no device kernel, counted per type, so
    // the next kernel is chosen from what the model actually asks for rather than from guesswork.
    std::map<std::uint32_t, std::size_t> declined_by_type;
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

    // Matrices are uploaded on first use, deliberately. Uploading them eagerly as each layer loads
    // was measured and was 25% slower at full depth: it spends VRAM on trunk matrices before the
    // routed experts - which do most of the arithmetic - can ask for any, so first-come beats
    // most-used. On-demand residency lets the busiest matrices win the space.

    [[nodiscard]] const CudaMatvecAcceleratorStats& stats() const noexcept { return stats_; }

    // Whether these exact weights are in device memory right now.
    //
    // Residency is evictable, so this is a snapshot and not a promise: a caller must still be able
    // to supply the host bytes afterwards. Reporting only.
    [[nodiscard]] bool holds(std::uint64_t cache_identity) const noexcept {
        return cache_identity != 0U && resident_.contains(cache_identity);
    }

private:
    struct Resident {
        CudaAllocation allocation;
        ByteCount total_bytes{0U};
        std::uint64_t last_used{0U};
        ByteCount matrix_bytes{0U};
        ByteCount grid_offset{0U};
        ByteCount input_offset{0U};
        ByteCount output_offset{0U};
        std::size_t rows{0U};
        std::size_t blocks_per_row{0U};
        std::uint32_t ggml_type{0U};
        // Whether the next pass will ask for these weights again. See the eviction comment in
        // ensure_resident: recency alone cannot separate weights every token needs from experts
        // this token happened to pick.
        bool recurring{false};
    };

    CudaMatvecAccelerator(CudaDriver driver,
                          safety::ResourceGovernor& governor,
                          CudaMatvecAcceleratorOptions options);

    [[nodiscard]] Result<Resident*> ensure_resident(
        const inference::DeepseekV4PackedMatrix& matrix);

    CudaDriver driver_;
    safety::ResourceGovernor* governor_{nullptr};
    CudaMatvecAcceleratorOptions options_;
    // Keyed on the descriptor's stable identity, never on a host address: see the note on
    // DeepseekV4PackedMatrix::cache_identity for what address-keying did to the model's output.
    std::map<std::uint64_t, Resident> resident_;
    // Monotonic tick, stamped on every use so the least recently used entry can be identified.
    std::uint64_t clock_{0U};
    CudaMatvecAcceleratorStats stats_;
};

}  // namespace vastloom::cuda
