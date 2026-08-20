#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vastloom/core/status.hpp"

namespace vastloom::safety {
class ResourceGovernor;
}

namespace vastloom::inference {

// Opt-in compute backend that delegates matrix-vector products to a pinned external ggml
// checkout. Disabled by default: the bounded scalar reference kernels remain the project's
// validated path and parity gates. When enabled, supported matvecs run on ggml's SIMD kernels
// over the same streamed GGML-format bytes; unsupported types, shapes and buffer alignments
// fall back to the scalar kernels call by call. ggml accumulates in float in its own order,
// so backend results differ from the f64 scalar reference by accumulation rounding only - the
// decoded weights are identical by construction (same block bytes, ggml's own decoder).
struct GgmlComputeOptions {
    unsigned threads{1U};
    // Governor-charged scratch for the backend's per-call graph pool and work buffer.
    std::size_t pool_bytes{8U * 1024U * 1024U};
};

// True when the build was compiled against a ggml checkout (VASTLOOM_GGML_ROOT).
[[nodiscard]] bool ggml_compute_compiled() noexcept;

// True between a successful ggml_compute_enable and ggml_compute_disable.
[[nodiscard]] bool ggml_compute_enabled() noexcept;

// Creates the backend's thread pool and graph pool after reserving their memory with the
// Governor. Refuses when the build has no ggml, when already enabled, or when the reservation
// fails. Thread count is the Governor's decision, passed by the caller.
[[nodiscard]] Result<void> ggml_compute_enable(
    safety::ResourceGovernor& governor, const GgmlComputeOptions& options = {});

// Releases the thread pool, graph pool and Governor reservation. Safe when not enabled.
void ggml_compute_disable() noexcept;

// True for storage types the backend can multiply.
[[nodiscard]] bool ggml_compute_supports(std::uint32_t ggml_type) noexcept;

// True when the backend can take this exact call: enabled, supported type, dimensions within
// ggml's block constraints and buffer alignment it can consume. Calls that do not pass stay on
// the scalar reference path.
[[nodiscard]] bool ggml_compute_can_run(
    std::uint32_t ggml_type,
    const std::byte* encoded_matrix,
    std::size_t rows,
    std::size_t columns) noexcept;

// One ggml-backed matrix-vector product for `rows` consecutive GGML-format rows of `columns`
// elements each. `input` must have exactly `columns` elements and `output` exactly `rows`.
[[nodiscard]] Result<void> matvec_ggml_backend(
    std::uint32_t ggml_type,
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output);

// Counters for measurement: matvecs the backend ran, and calls that reached the dispatch point
// but fell back to the scalar path (type, shape or alignment). The fallback note belongs to the
// dispatch point, not to the backend. Input-quantize counts how many times an F32 activation was
// converted to the weight's vec_dot type; hits are later matvecs that reused that encoding.
[[nodiscard]] std::uint64_t ggml_compute_accelerated_count() noexcept;
[[nodiscard]] std::uint64_t ggml_compute_fallback_count() noexcept;
[[nodiscard]] std::uint64_t ggml_compute_input_quantize_count() noexcept;
[[nodiscard]] std::uint64_t ggml_compute_input_quantize_hits() noexcept;
void ggml_compute_note_fallback() noexcept;

}  // namespace vastloom::inference
