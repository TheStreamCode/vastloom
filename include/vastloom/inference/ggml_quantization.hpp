#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

inline constexpr std::size_t kF32ElementBytes = 4U;
inline constexpr std::size_t kQ8_0BlockElements = 32U;
inline constexpr std::size_t kQ8_0BlockBytes = 34U;
inline constexpr std::size_t kQ4KBlockElements = 256U;
inline constexpr std::size_t kQ4KBlockBytes = 144U;
inline constexpr std::size_t kQ5KBlockElements = 256U;
inline constexpr std::size_t kQ5KBlockBytes = 176U;
inline constexpr std::size_t kQ6KBlockElements = 256U;
inline constexpr std::size_t kQ6KBlockBytes = 210U;
inline constexpr std::size_t kIq2XxsBlockElements = 256U;
inline constexpr std::size_t kIq2XxsBlockBytes = 66U;
inline constexpr std::size_t kIq1SBlockElements = 256U;
inline constexpr std::size_t kIq1SBlockBytes = 50U;
inline constexpr std::size_t kIq3XxsBlockElements = 256U;
inline constexpr std::size_t kIq3XxsBlockBytes = 98U;
inline constexpr std::size_t kI32ElementBytes = 4U;
inline constexpr std::size_t kBf16ElementBytes = 2U;
inline constexpr std::size_t kMxfp4BlockElements = 32U;
inline constexpr std::size_t kMxfp4BlockBytes = 17U;

inline constexpr std::uint32_t kGgmlTypeF32 = 0U;
inline constexpr std::uint32_t kGgmlTypeQ8_0 = 8U;
inline constexpr std::uint32_t kGgmlTypeQ4K = 12U;
inline constexpr std::uint32_t kGgmlTypeQ5K = 13U;
inline constexpr std::uint32_t kGgmlTypeQ6K = 14U;
inline constexpr std::uint32_t kGgmlTypeIq2Xxs = 16U;
inline constexpr std::uint32_t kGgmlTypeIq3Xxs = 18U;
inline constexpr std::uint32_t kGgmlTypeIq1S = 19U;
inline constexpr std::uint32_t kGgmlTypeI32 = 26U;
inline constexpr std::uint32_t kGgmlTypeBf16 = 30U;
inline constexpr std::uint32_t kGgmlTypeMxfp4 = 39U;

struct QuantizedKernelLimits {
    std::size_t max_elements{64'000'000U};
};

struct RoutedExpertScratch {
    std::span<float> gate;
    std::span<float> up;
    std::span<float> down;
};

// Deterministic ggml-compatible reference encoder used by conversion/fixture tooling. Logical
// rows are zero-padded to storage_columns; no file is written by this function.
[[nodiscard]] Result<std::vector<std::byte>> quantize_q8_0_reference(
    std::span<const float> matrix,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns,
    const QuantizedKernelLimits& limits = {});

// True only for storage formats with a bounded scalar reference matvec below.
[[nodiscard]] bool supports_reference_matvec_ggml_type(std::uint32_t ggml_type) noexcept;

// Includes non-matvec integer routing tables that have a bounded native decoder.
[[nodiscard]] bool supports_reference_storage_ggml_type(std::uint32_t ggml_type) noexcept;

// Returns the exact encoded byte count for one known GGML layout after validating the shape and
// configured element ceiling. A known layout does not imply that a native matvec exists; query
// supports_reference_matvec_ggml_type before execution.
[[nodiscard]] Result<std::size_t> ggml_storage_bytes(
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t columns,
    const QuantizedKernelLimits& limits = {});

// Decodes complete little-endian GGML Q8_0 blocks into caller-owned FP32 storage.
[[nodiscard]] Result<void> dequantize_q8_0(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Decodes complete little-endian GGML Q4_K blocks into caller-owned FP32 storage.
[[nodiscard]] Result<void> dequantize_q4_k(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for consecutive GGML Q4_K rows.
[[nodiscard]] Result<void> matvec_q4_k_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for consecutive GGML Q8_0 rows.
[[nodiscard]] Result<void> matvec_q8_0_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Decodes complete little-endian GGML Q5_K blocks into caller-owned FP32 storage.
[[nodiscard]] Result<void> dequantize_q5_k(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for consecutive GGML Q5_K rows.
[[nodiscard]] Result<void> matvec_q5_k_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Decodes complete little-endian GGML Q6_K blocks into caller-owned FP32 storage.
[[nodiscard]] Result<void> dequantize_q6_k(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for consecutive GGML Q6_K rows.
[[nodiscard]] Result<void> matvec_q6_k_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Decodes complete little-endian GGML IQ2_XXS blocks into caller-owned FP32 storage.
[[nodiscard]] Result<void> dequantize_iq2_xxs(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for consecutive GGML IQ2_XXS rows.
[[nodiscard]] Result<void> matvec_iq2_xxs_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

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

// Decodes little-endian BF16 elements without assuming host alignment or endianness.
[[nodiscard]] Result<void> dequantize_bf16(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Reference matrix-vector product for consecutive little-endian BF16 rows.
[[nodiscard]] Result<void> matvec_bf16_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Decodes and multiplies little-endian F32 tensors without unchecked casts.
[[nodiscard]] Result<void> dequantize_f32_le(
    std::span<const std::byte> encoded,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

[[nodiscard]] Result<void> matvec_f32_le_f32(
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Decodes little-endian I32 routing tables into caller-owned integer storage.
[[nodiscard]] Result<void> decode_i32_le(
    std::span<const std::byte> encoded,
    std::span<std::int32_t> output,
    const QuantizedKernelLimits& limits = {});

// Decodes exactly one physical row from a complete supported GGML matrix. The caller supplies
// storage-width output, so large embedding tables can be accessed without materializing every
// row. Matrix byte arithmetic remains subject to the explicit element ceiling.
[[nodiscard]] Result<void> dequantize_ggml_row_f32(
    std::uint32_t ggml_type,
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::size_t row,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Dispatches a supported GGML storage type into caller-owned scratch and publishes
// `output` only after the whole matrix-vector product succeeds. Scratch, input and
// output must be pairwise disjoint.
[[nodiscard]] Result<void> matvec_ggml_f32_transactional(
    std::uint32_t ggml_type,
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> scratch,
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
