#pragma once

#include <cstddef>
#include <span>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

inline constexpr std::size_t kIq1SBlockElements = 256U;
inline constexpr std::size_t kIq1SBlockBytes = 50U;

struct QuantizedKernelLimits {
    std::size_t max_elements{64'000'000U};
};

// Decodes complete little-endian GGML IQ1_S blocks into caller-owned FP32 storage.
[[nodiscard]] Result<void> dequantize_iq1_s(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for a GGML matrix whose first/contiguous dimension is
// `columns`. The encoded matrix contains `rows` consecutive IQ1_S rows.
[[nodiscard]] Result<void> matvec_iq1_s_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

}  // namespace vastloom::inference
