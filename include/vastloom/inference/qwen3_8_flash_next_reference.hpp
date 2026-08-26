#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

// Qwen3.8-Flash-Next reference operators — pure, deterministic, no I/O.
//
// These are the ground-truth scalar kernels that the streaming implementation
// must match bit-exactly. They are intentionally slow and single-threaded.

// FP8 E4M3 (OCP) -> F32. 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
// Exponent 15 is NaN (no Inf in E4M3). Subnormals are supported.
[[nodiscard]] float qwen38_fp8_e4m3_to_f32(std::uint8_t bits) noexcept;

// BF16 (1-8-7) -> F32. Bit-exact: payload is the high 16 bits of an F32.
[[nodiscard]] float qwen38_bf16_to_f32(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t qwen38_f32_to_bf16(float value) noexcept;

// Decode a BF16 tensor payload into F32. Payload must contain exactly
// product(shape) * 2 bytes. Returns F32 vector in row-major order.
[[nodiscard]] Result<std::vector<float>> qwen38_decode_bf16(
    std::span<const std::byte> payload,
    const std::vector<std::int64_t>& shape);

// Decode an FP8 block-scaled tensor. Weight payload is F8_E4M3 bytes,
// scale payload is BF16 per block (block_rows x block_cols, default 128x128).
// Shape is the logical shape of the weight tensor (e.g., [640, 2560]).
// Returns F32 vector: weight_f32[i] = fp8_to_f32(weight_bytes[i]) * bf16_to_f32(scale_for_block(i))
[[nodiscard]] Result<std::vector<float>> qwen38_decode_fp8_block_scale(
    std::span<const std::byte> weight_payload,
    std::span<const std::byte> scale_payload,
    const std::vector<std::int64_t>& weight_shape,
    const std::vector<std::int64_t>& scale_shape,
    std::size_t block_rows = 128,
    std::size_t block_cols = 128);

// Convenience: decode a tensor whose descriptor declares F8_E4M3 + companion
// scale_inv. Reads both payload spans and dispatches to the block-scale decoder.
// If the tensor is BF16, dispatches to qwen38_decode_bf16.
[[nodiscard]] Result<std::vector<float>> qwen38_decode_tensor_payload(
    std::span<const std::byte> payload,
    std::span<const std::byte> scale_payload,
    const std::string& dtype,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>& scale_shape);

// RMSNorm reference: y = x / sqrt(mean(x^2) + eps) * weight
[[nodiscard]] Result<std::vector<float>> qwen38_rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon);

}  // namespace vastloom::inference
