#pragma once

#include <cstddef>
#include <span>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

inline constexpr std::size_t kIq1SBlockElements = 256U;
inline constexpr std::size_t kIq1SBlockBytes = 50U;
inline constexpr std::size_t kIq3XxsBlockElements = 256U;
inline constexpr std::size_t kIq3XxsBlockBytes = 98U;

struct QuantizedKernelLimits {
    std::size_t max_elements{64'000'000U};
};

struct RoutedExpertScratch {
    std::span<float> gate;
    std::span<float> up;
    std::span<float> down;
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

// Decodes complete little-endian GGML IQ3_XXS blocks into caller-owned FP32 storage.
[[nodiscard]] Result<void> dequantize_iq3_xxs(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for consecutive GGML IQ3_XXS rows.
[[nodiscard]] Result<void> matvec_iq3_xxs_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Executes one gate/up IQ1_S plus down IQ3_XXS SwiGLU expert. All temporary
// storage is caller-owned, and `output` is published only after the complete
// operation succeeds.
[[nodiscard]] Result<void> routed_expert_iq1_s_iq3_xxs_f32(
    std::span<const std::byte> gate_matrix,
    std::span<const std::byte> up_matrix,
    std::span<const std::byte> down_matrix,
    std::span<const float> input,
    RoutedExpertScratch scratch,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

}  // namespace vastloom::inference
