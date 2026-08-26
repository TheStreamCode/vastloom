#include "vastloom/inference/qwen3_8_flash_next_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

float qwen38_fp8_e4m3_to_f32(std::uint8_t bits) noexcept {
    const std::uint32_t sign = (bits >> 7) & 0x1U;
    const std::uint32_t exp = (bits >> 3) & 0xFU;
    const std::uint32_t mant = bits & 0x7U;

    // E4M3 special: exp==15 => NaN (no Inf).
    if (exp == 15U) {
        // OCP: 0x7F and 0xFF are NaN, but we treat any exp==15 as NaN for safety.
        // Preserve sign in NaN payload if needed, but return quiet NaN.
        return std::numeric_limits<float>::quiet_NaN();
    }

    float value = 0.0F;
    if (exp == 0U) {
        // Subnormal: mant/8 * 2^(1-bias) where bias=7 => 2^-6
        // value = (mant / 8.0) * 2^-6
        if (mant == 0U) {
            value = 0.0F;
        } else {
            value = static_cast<float>(mant) / 8.0F * std::exp2(-6.0F);
        }
    } else {
        // Normal: (1 + mant/8) * 2^(exp - bias)
        value = (1.0F + static_cast<float>(mant) / 8.0F) * std::exp2(static_cast<float>(static_cast<int>(exp) - 7));
    }

    return sign ? -value : value;
}

float qwen38_bf16_to_f32(std::uint16_t bits) noexcept {
    std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
    float out = 0.0F;
    std::memcpy(&out, &f32_bits, sizeof(out));
    return out;
}

std::uint16_t qwen38_f32_to_bf16(float value) noexcept {
    std::uint32_t f32_bits = 0U;
    std::memcpy(&f32_bits, &value, sizeof(f32_bits));
    // Round to nearest even: add 0x7FFF + ((f32_bits>>16)&1) before truncation.
    // For the reference we use simple truncation (round toward zero) — bit-exact
    // for the values we pin in tests. Use truncation to keep the reference trivial.
    // Callers that need RNE should implement it separately.
    // Here we do RNE to match the common hardware path.
    std::uint32_t rounding_bias = 0x7FFFU + ((f32_bits >> 16) & 1U);
    f32_bits += rounding_bias;
    return static_cast<std::uint16_t>(f32_bits >> 16);
}

Result<std::vector<float>> qwen38_decode_bf16(
    std::span<const std::byte> payload,
    const std::vector<std::int64_t>& shape) {
    std::size_t elements = 1;
    for (auto d : shape) {
        if (d <= 0) {
            return make_error(ErrorCode::invalid_argument,
                "shape dim must be positive", "bf16_decode shape");
        }
        elements *= static_cast<std::size_t>(d);
    }
    if (payload.size() != elements * 2U) {
        return make_error(ErrorCode::invalid_argument,
            "payload size does not match shape for BF16",
            "expected=" + std::to_string(elements * 2U) + " got=" + std::to_string(payload.size()));
    }
    std::vector<float> out;
    out.reserve(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        std::uint16_t bits = 0U;
        std::memcpy(&bits, payload.data() + i * 2U, 2U);
        // Payload is little-endian BF16 as stored in safetensors (native LE).
        // On LE hosts memcpy gives the correct value directly.
        out.push_back(qwen38_bf16_to_f32(bits));
    }
    return out;
}

Result<std::vector<float>> qwen38_decode_fp8_block_scale(
    std::span<const std::byte> weight_payload,
    std::span<const std::byte> scale_payload,
    const std::vector<std::int64_t>& weight_shape,
    const std::vector<std::int64_t>& scale_shape,
    std::size_t block_rows,
    std::size_t block_cols) {
    if (weight_shape.size() != 2) {
        return make_error(ErrorCode::invalid_argument,
            "FP8 block-scale requires 2D weight shape", "fp8_decode");
    }
    if (scale_shape.size() != 2) {
        return make_error(ErrorCode::invalid_argument,
            "FP8 block-scale requires 2D scale shape", "fp8_decode");
    }
    const std::size_t rows = static_cast<std::size_t>(weight_shape[0]);
    const std::size_t cols = static_cast<std::size_t>(weight_shape[1]);
    const std::size_t scale_rows = static_cast<std::size_t>(scale_shape[0]);
    const std::size_t scale_cols = static_cast<std::size_t>(scale_shape[1]);

    const std::size_t expected_scale_rows = (rows + block_rows - 1) / block_rows;
    const std::size_t expected_scale_cols = (cols + block_cols - 1) / block_cols;
    if (scale_rows != expected_scale_rows || scale_cols != expected_scale_cols) {
        return make_error(ErrorCode::invalid_argument,
            "scale shape does not match block tiling",
            "weight=[" + std::to_string(rows) + "," + std::to_string(cols) + "]"
            + " scale=[" + std::to_string(scale_rows) + "," + std::to_string(scale_cols) + "]"
            + " expected=[" + std::to_string(expected_scale_rows) + "," + std::to_string(expected_scale_cols) + "]");
    }

    const std::size_t weight_elements = rows * cols;
    const std::size_t scale_elements = scale_rows * scale_cols;
    if (weight_payload.size() != weight_elements) {
        return make_error(ErrorCode::invalid_argument,
            "weight payload size mismatch for F8_E4M3",
            "expected=" + std::to_string(weight_elements) + " got=" + std::to_string(weight_payload.size()));
    }
    if (scale_payload.size() != scale_elements * 2U) {
        return make_error(ErrorCode::invalid_argument,
            "scale payload size mismatch for BF16",
            "expected=" + std::to_string(scale_elements * 2U) + " got=" + std::to_string(scale_payload.size()));
    }

    // Decode scales to F32 once.
    std::vector<float> scales_f32;
    scales_f32.reserve(scale_elements);
    for (std::size_t i = 0; i < scale_elements; ++i) {
        std::uint16_t bits = 0U;
        std::memcpy(&bits, scale_payload.data() + i * 2U, 2U);
        scales_f32.push_back(qwen38_bf16_to_f32(bits));
    }

    std::vector<float> out;
    out.reserve(weight_elements);
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t block_row = r / block_rows;
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t block_col = c / block_cols;
            const std::size_t scale_idx = block_row * scale_cols + block_col;
            const float scale = scales_f32[scale_idx];
            const std::size_t weight_idx = r * cols + c;
            std::uint8_t bits = 0U;
            std::memcpy(&bits, weight_payload.data() + weight_idx, 1U);
            const float w = qwen38_fp8_e4m3_to_f32(bits);
            // NaN * scale => NaN, which is the correct OCP behavior (propagate NaN).
            out.push_back(w * scale);
        }
    }
    return out;
}

Result<std::vector<float>> qwen38_decode_tensor_payload(
    std::span<const std::byte> payload,
    std::span<const std::byte> scale_payload,
    const std::string& dtype,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>& scale_shape) {
    if (dtype == "BF16") {
        if (!scale_payload.empty()) {
            return make_error(ErrorCode::invalid_argument,
                "BF16 tensor should not have scale payload", dtype);
        }
        return qwen38_decode_bf16(payload, shape);
    }
    if (dtype == "F8_E4M3") {
        return qwen38_decode_fp8_block_scale(payload, scale_payload, shape, scale_shape, 128, 128);
    }
    if (dtype == "F32") {
        std::size_t elements = 1;
        for (auto d : shape) elements *= static_cast<std::size_t>(d);
        if (payload.size() != elements * 4U) {
            return make_error(ErrorCode::invalid_argument,
                "payload size mismatch for F32", dtype);
        }
        std::vector<float> out(elements);
        std::memcpy(out.data(), payload.data(), payload.size());
        return out;
    }
    return make_error(ErrorCode::unsupported,
        "unsupported dtype for decode", dtype);
}

Result<std::vector<float>> qwen38_rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon) {
    if (input.size() != weight.size()) {
        return make_error(ErrorCode::invalid_argument,
            "rms_norm input and weight must have same size",
            "input=" + std::to_string(input.size()) + " weight=" + std::to_string(weight.size()));
    }
    if (input.empty()) {
        return make_error(ErrorCode::invalid_argument,
            "rms_norm input must be non-empty", "rms_norm");
    }
    double sum_sq = 0.0;
    for (float v : input) sum_sq += static_cast<double>(v) * static_cast<double>(v);
    const double mean_sq = sum_sq / static_cast<double>(input.size());
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(epsilon));
    std::vector<float> out;
    out.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        out.push_back(static_cast<float>(static_cast<double>(input[i]) * inv_rms * static_cast<double>(weight[i])));
    }
    return out;
}

}  // namespace vastloom::inference
