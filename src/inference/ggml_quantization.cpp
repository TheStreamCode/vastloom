#include "vastloom/inference/ggml_quantization.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace vastloom::inference {
namespace {

// The immutable IQ codebooks are derived from ggml commit
// 30bf8685ed4eb0a47f2b06229543327749904150 under its MIT license. See NOTICE and
// third_party/ggml/LICENSE. Keeping them local avoids a runtime dependency.
constexpr std::array<std::uint64_t, 2'048U> kIq1SGrid{
#include "iq1s_grid.inc"
};

constexpr std::array<std::uint64_t, 256U> kIq2XxsGrid{
#include "iq2xxs_grid.inc"
};

constexpr std::array<std::uint32_t, 256U> kIq3XxsGrid{
#include "iq3xxs_grid.inc"
};

// Doubled E2M1 values used by GGML MXFP4. The scale decoder below supplies the
// matching one-half factor. Pinned to the same ggml revision as the IQ tables.
constexpr std::array<std::int8_t, 16U> kMxfp4Values{
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

consteval std::uint64_t iq1_s_grid_fingerprint() {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const std::uint64_t value : kIq1SGrid) {
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((value >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static_assert(iq1_s_grid_fingerprint() == UINT64_C(0x6703ed863501ae2e));

consteval std::uint64_t iq2_xxs_grid_fingerprint() {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const std::uint64_t value : kIq2XxsGrid) {
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((value >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static_assert(iq2_xxs_grid_fingerprint() == UINT64_C(0xbb4ee025b5ac6e8e));

consteval std::uint64_t iq3_xxs_grid_fingerprint() {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const std::uint32_t value : kIq3XxsGrid) {
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((value >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static_assert(iq3_xxs_grid_fingerprint() == UINT64_C(0xdfa5dc83d6a155d5));

float fp16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    std::uint32_t exponent = static_cast<std::uint32_t>((value >> 10U) & 0x1fU);
    std::uint32_t mantissa = static_cast<std::uint32_t>(value & 0x03ffU);
    std::uint32_t bits = sign;
    if (exponent == 0U) {
        if (mantissa != 0U) {
            exponent = 113U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03ffU;
            bits |= exponent << 23U;
            bits |= mantissa << 13U;
        }
    } else if (exponent == 0x1fU) {
        bits |= 0x7f800000U | (mantissa << 13U);
    } else {
        bits |= (exponent + 112U) << 23U;
        bits |= mantissa << 13U;
    }
    return std::bit_cast<float>(bits);
}

float e8m0_to_float_half(std::uint8_t value) noexcept {
    // This is the exact scalar bit construction used by the pinned GGML MXFP4
    // decoder. Values 0 and 1 map to the two smallest half-scaled powers;
    // all other byte values map to a normalized exponent with no mantissa.
    const std::uint32_t bits = value < 2U
                                   ? UINT32_C(0x00200000) << value
                                   : static_cast<std::uint32_t>(value - 1U) << 23U;
    return std::bit_cast<float>(bits);
}

std::uint16_t float_to_fp16(float value) noexcept {
    constexpr float scale_to_infinity = 0x1.0p+112F;
    constexpr float scale_to_zero = 0x1.0p-110F;
    float base = (std::abs(value) * scale_to_infinity) * scale_to_zero;
    const std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t shifted = word + word;
    const std::uint32_t sign = word & UINT32_C(0x80000000);
    std::uint32_t bias = shifted & UINT32_C(0xff000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
    base = std::bit_cast<float>((bias >> 1U) + UINT32_C(0x07800000)) + base;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(base);
    const std::uint32_t exponent = (bits >> 13U) & UINT32_C(0x00007c00);
    const std::uint32_t mantissa = bits & UINT32_C(0x00000fff);
    const std::uint32_t non_sign = exponent + mantissa;
    return static_cast<std::uint16_t>(
        (sign >> 16U) |
        (shifted > UINT32_C(0xff000000) ? UINT16_C(0x7e00) : non_sign));
}

void write_u16_le(std::byte* bytes, std::uint16_t value) noexcept {
    bytes[0] = static_cast<std::byte>(value & 0xffU);
    bytes[1] = static_cast<std::byte>(value >> 8U);
}

std::uint16_t read_u16_le(const std::byte* bytes) noexcept {
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[0])) |
           static_cast<std::uint16_t>(
               static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[1])) << 8U);
}

std::uint32_t read_u32_le(const std::byte* bytes) noexcept {
    std::uint32_t value = 0U;
    for (unsigned index = 0U; index < 4U; ++index) {
        value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[index]))
                 << (index * 8U);
    }
    return value;
}

Result<std::size_t> checked_matrix_elements(std::size_t rows,
                                            std::size_t columns,
                                            std::size_t block_elements,
                                            const QuantizedKernelLimits& limits) {
    if (limits.max_elements == 0U || rows == 0U || columns == 0U ||
        block_elements == 0U || columns % block_elements != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Quantized matrix dimensions or kernel limits are invalid");
    }
    if (rows > limits.max_elements / columns) {
        return make_error(ErrorCode::resource_exhausted,
                          "Quantized matrix exceeds the configured element limit");
    }
    return rows * columns;
}

Result<std::size_t> checked_encoded_bytes(std::size_t elements,
                                          std::size_t block_elements,
                                          std::size_t block_bytes) {
    if (block_elements == 0U || block_bytes == 0U || elements % block_elements != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Quantized element count or block layout is invalid");
    }
    const std::size_t blocks = elements / block_elements;
    if (blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
        return make_error(ErrorCode::resource_exhausted,
                          "Quantized encoded byte count overflows");
    }
    return blocks * block_bytes;
}

Result<void> validate_block_scales(std::span<const std::byte> encoded,
                                   std::size_t block_bytes) {
    for (std::size_t offset = 0U; offset < encoded.size(); offset += block_bytes) {
        if (!std::isfinite(fp16_to_float(read_u16_le(encoded.data() + offset)))) {
            return make_error(ErrorCode::invalid_format,
                              "Quantized block scale is not finite");
        }
    }
    return {};
}

Result<void> decode_iq1_s_block(const std::byte* encoded,
                                std::span<float, kIq1SBlockElements> output) {
    const float scale = fp16_to_float(read_u16_le(encoded));
    if (!std::isfinite(scale)) {
        return make_error(ErrorCode::invalid_format,
                          "IQ1_S block scale is not finite");
    }
    constexpr float delta_magnitude = 0.125F;
    const std::byte* qs = encoded + 2U;
    const std::byte* qh = encoded + 34U;
    for (std::size_t group = 0U; group < 8U; ++group) {
        const std::uint16_t high = read_u16_le(qh + group * 2U);
        const float local_scale = scale * static_cast<float>(
            2U * static_cast<std::uint32_t>((high >> 12U) & 7U) + 1U);
        const float delta = (high & 0x8000U) != 0U ? -delta_magnitude : delta_magnitude;
        for (std::size_t subblock = 0U; subblock < 4U; ++subblock) {
            const auto low = std::to_integer<std::uint8_t>(qs[group * 4U + subblock]);
            const auto high_index = static_cast<std::uint16_t>(
                (high >> static_cast<unsigned>(3U * subblock)) & 7U);
            const std::size_t grid_index = static_cast<std::size_t>(low) |
                                           (static_cast<std::size_t>(high_index) << 8U);
            const std::uint64_t grid = kIq1SGrid[grid_index];
            for (std::size_t lane = 0U; lane < 8U; ++lane) {
                const auto raw = static_cast<std::uint8_t>((grid >> (lane * 8U)) & 0xffU);
                const auto value = std::bit_cast<std::int8_t>(raw);
                output[group * 32U + subblock * 8U + lane] =
                    local_scale * (static_cast<float>(value) + delta);
            }
        }
    }
    return {};
}

Result<void> decode_iq2_xxs_block(
    const std::byte* encoded,
    std::span<float, kIq2XxsBlockElements> output) {
    const float scale = fp16_to_float(read_u16_le(encoded));
    if (!std::isfinite(scale)) {
        return make_error(ErrorCode::invalid_format,
                          "IQ2_XXS block scale is not finite");
    }
    const std::byte* quantized = encoded + 2U;
    for (std::size_t group = 0U; group < 8U; ++group) {
        const std::byte* packed = quantized + group * 8U;
        const std::uint32_t auxiliary = read_u32_le(packed + 4U);
        const float local_scale = scale *
                                  (0.5F + static_cast<float>(auxiliary >> 28U)) * 0.25F;
        for (std::size_t subblock = 0U; subblock < 4U; ++subblock) {
            const auto grid_index = std::to_integer<std::uint8_t>(packed[subblock]);
            const std::uint64_t grid = kIq2XxsGrid[grid_index];
            const std::uint32_t sign_index =
                (auxiliary >> static_cast<unsigned>(7U * subblock)) & 127U;
            const std::uint8_t signs = static_cast<std::uint8_t>(
                sign_index | ((static_cast<std::uint32_t>(std::popcount(sign_index)) & 1U)
                              << 7U));
            const std::size_t base = group * 32U + subblock * 8U;
            for (std::size_t lane = 0U; lane < 8U; ++lane) {
                const auto magnitude = static_cast<std::uint8_t>(
                    (grid >> static_cast<unsigned>(lane * 8U)) & 0xffU);
                const float sign = (signs & (1U << lane)) != 0U ? -1.0F : 1.0F;
                output[base + lane] =
                    local_scale * static_cast<float>(magnitude) * sign;
            }
        }
    }
    return {};
}

Result<void> decode_iq3_xxs_block(
    const std::byte* encoded,
    std::span<float, kIq3XxsBlockElements> output) {
    const float scale = fp16_to_float(read_u16_le(encoded));
    if (!std::isfinite(scale)) {
        return make_error(ErrorCode::invalid_format,
                          "IQ3_XXS block scale is not finite");
    }
    const std::byte* quantized = encoded + 2U;
    const std::byte* scales_and_signs = quantized + 64U;
    for (std::size_t group = 0U; group < 8U; ++group) {
        const std::uint32_t auxiliary = read_u32_le(scales_and_signs + group * 4U);
        const float local_scale = scale *
                                  (0.5F + static_cast<float>(auxiliary >> 28U)) * 0.5F;
        for (std::size_t subblock = 0U; subblock < 4U; ++subblock) {
            const std::uint32_t sign_index =
                (auxiliary >> static_cast<unsigned>(7U * subblock)) & 127U;
            const std::uint8_t signs = static_cast<std::uint8_t>(
                sign_index | ((static_cast<std::uint32_t>(std::popcount(sign_index)) & 1U)
                              << 7U));
            const auto first_index = std::to_integer<std::uint8_t>(
                quantized[group * 8U + subblock * 2U]);
            const auto second_index = std::to_integer<std::uint8_t>(
                quantized[group * 8U + subblock * 2U + 1U]);
            const std::uint32_t first_grid = kIq3XxsGrid[first_index];
            const std::uint32_t second_grid = kIq3XxsGrid[second_index];
            for (std::size_t lane = 0U; lane < 4U; ++lane) {
                const float first_sign = (signs & (1U << lane)) != 0U ? -1.0F : 1.0F;
                const float second_sign =
                    (signs & (1U << (lane + 4U))) != 0U ? -1.0F : 1.0F;
                const auto first = static_cast<std::uint8_t>(
                    (first_grid >> static_cast<unsigned>(lane * 8U)) & 0xffU);
                const auto second = static_cast<std::uint8_t>(
                    (second_grid >> static_cast<unsigned>(lane * 8U)) & 0xffU);
                const std::size_t base = group * 32U + subblock * 8U;
                output[base + lane] =
                    local_scale * static_cast<float>(first) * first_sign;
                output[base + lane + 4U] =
                    local_scale * static_cast<float>(second) * second_sign;
            }
        }
    }
    return {};
}

Result<void> decode_mxfp4_block(
    const std::byte* encoded,
    std::span<float, kMxfp4BlockElements> output) {
    const float scale = e8m0_to_float_half(
        std::to_integer<std::uint8_t>(encoded[0]));
    for (std::size_t index = 0U; index < kMxfp4BlockElements / 2U; ++index) {
        const auto packed = std::to_integer<std::uint8_t>(encoded[1U + index]);
        const float low = static_cast<float>(kMxfp4Values[packed & 15U]) * scale;
        const float high = static_cast<float>(kMxfp4Values[packed >> 4U]) * scale;
        if (!std::isfinite(low) || !std::isfinite(high)) {
            return make_error(ErrorCode::invalid_format,
                              "MXFP4 block produces a non-finite value");
        }
        output[index] = low;
        output[index + kMxfp4BlockElements / 2U] = high;
    }
    return {};
}

Result<void> validate_mxfp4_blocks(std::span<const std::byte> encoded) {
    std::array<float, kMxfp4BlockElements> decoded{};
    for (std::size_t offset = 0U; offset < encoded.size(); offset += kMxfp4BlockBytes) {
        auto status = decode_mxfp4_block(
            encoded.data() + offset,
            std::span<float, kMxfp4BlockElements>(decoded));
        if (!status) return status.error();
    }
    return {};
}

float bf16_to_float(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

float read_f32_le(const std::byte* bytes) noexcept {
    return std::bit_cast<float>(read_u32_le(bytes));
}

Result<void> decode_q8_0_block(
    const std::byte* encoded,
    std::span<float, kQ8_0BlockElements> output) {
    const float scale = fp16_to_float(read_u16_le(encoded));
    if (!std::isfinite(scale)) {
        return make_error(ErrorCode::invalid_format,
                          "Q8_0 block scale is not finite");
    }
    for (std::size_t index = 0U; index < output.size(); ++index) {
        const auto raw = std::to_integer<std::uint8_t>(encoded[2U + index]);
        const auto quantized = std::bit_cast<std::int8_t>(raw);
        output[index] = scale * static_cast<float>(quantized);
    }
    return {};
}

// Q4_K/Q5_K scale unpacking follows ggml commit
// 30bf8685ed4eb0a47f2b06229543327749904150 under its MIT license.
void qk_scale_min(std::size_t index,
                  const std::byte* encoded,
                  std::uint8_t& scale,
                  std::uint8_t& minimum) noexcept {
    const auto at = [encoded](std::size_t offset) {
        return std::to_integer<std::uint8_t>(encoded[offset]);
    };
    if (index < 4U) {
        scale = static_cast<std::uint8_t>(at(index) & 63U);
        minimum = static_cast<std::uint8_t>(at(index + 4U) & 63U);
        return;
    }
    scale = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(at(index + 4U)) & 15U) |
        ((static_cast<std::uint32_t>(at(index - 4U)) >> 6U) << 4U));
    minimum = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(at(index + 4U)) >> 4U) |
        ((static_cast<std::uint32_t>(at(index)) >> 6U) << 4U));
}

Result<void> decode_q4_k_block(
    const std::byte* encoded,
    std::span<float, kQ4KBlockElements> output) {
    const float scale = fp16_to_float(read_u16_le(encoded));
    const float minimum = fp16_to_float(read_u16_le(encoded + 2U));
    if (!std::isfinite(scale) || !std::isfinite(minimum)) {
        return make_error(ErrorCode::invalid_format,
                          "Q4_K block scale or minimum is not finite");
    }

    const std::byte* scales = encoded + 4U;
    const std::byte* quantized = encoded + 16U;
    std::size_t scale_index = 0U;
    for (std::size_t group = 0U; group < 4U; ++group) {
        std::uint8_t first_scale = 0U;
        std::uint8_t first_minimum = 0U;
        std::uint8_t second_scale = 0U;
        std::uint8_t second_minimum = 0U;
        qk_scale_min(scale_index, scales, first_scale, first_minimum);
        qk_scale_min(scale_index + 1U, scales, second_scale, second_minimum);
        const float d1 = scale * static_cast<float>(first_scale);
        const float m1 = minimum * static_cast<float>(first_minimum);
        const float d2 = scale * static_cast<float>(second_scale);
        const float m2 = minimum * static_cast<float>(second_minimum);
        const std::size_t base = group * 64U;
        for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const auto value = std::to_integer<std::uint8_t>(quantized[lane]);
            output[base + lane] =
                d1 * static_cast<float>(value & 15U) - m1;
            output[base + 32U + lane] =
                d2 * static_cast<float>(value >> 4U) - m2;
        }
        quantized += 32U;
        scale_index += 2U;
    }
    return {};
}

Result<void> decode_q5_k_block(
    const std::byte* encoded,
    std::span<float, kQ5KBlockElements> output) {
    const float scale = fp16_to_float(read_u16_le(encoded));
    const float minimum = fp16_to_float(read_u16_le(encoded + 2U));
    if (!std::isfinite(scale) || !std::isfinite(minimum)) {
        return make_error(ErrorCode::invalid_format,
                          "Q5_K block scale or minimum is not finite");
    }

    const std::byte* scales = encoded + 4U;
    const std::byte* high = encoded + 16U;
    const std::byte* low = encoded + 48U;
    std::size_t scale_index = 0U;
    std::uint8_t first_high_mask = 1U;
    std::uint8_t second_high_mask = 2U;
    for (std::size_t group = 0U; group < 4U; ++group) {
        std::uint8_t first_scale = 0U;
        std::uint8_t first_minimum = 0U;
        std::uint8_t second_scale = 0U;
        std::uint8_t second_minimum = 0U;
        qk_scale_min(scale_index, scales, first_scale, first_minimum);
        qk_scale_min(scale_index + 1U, scales, second_scale, second_minimum);
        const float d1 = scale * static_cast<float>(first_scale);
        const float m1 = minimum * static_cast<float>(first_minimum);
        const float d2 = scale * static_cast<float>(second_scale);
        const float m2 = minimum * static_cast<float>(second_minimum);
        for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const auto low_byte = std::to_integer<std::uint8_t>(low[lane]);
            const auto high_byte = std::to_integer<std::uint8_t>(high[lane]);
            const std::uint8_t first = static_cast<std::uint8_t>(
                (low_byte & 15U) + ((high_byte & first_high_mask) != 0U ? 16U : 0U));
            const std::uint8_t second = static_cast<std::uint8_t>(
                (low_byte >> 4U) + ((high_byte & second_high_mask) != 0U ? 16U : 0U));
            const std::size_t base = group * 64U;
            output[base + lane] = d1 * static_cast<float>(first) - m1;
            output[base + 32U + lane] = d2 * static_cast<float>(second) - m2;
        }
        low += 32U;
        scale_index += 2U;
        first_high_mask = static_cast<std::uint8_t>(first_high_mask << 2U);
        second_high_mask = static_cast<std::uint8_t>(second_high_mask << 2U);
    }
    return {};
}

Result<void> decode_q6_k_block(
    const std::byte* encoded,
    std::span<float, kQ6KBlockElements> output) {
    const float scale = fp16_to_float(read_u16_le(encoded + 208U));
    if (!std::isfinite(scale)) {
        return make_error(ErrorCode::invalid_format,
                          "Q6_K block scale is not finite");
    }

    const std::byte* low = encoded;
    const std::byte* high = encoded + 128U;
    const std::byte* scales = encoded + 192U;
    for (std::size_t half = 0U; half < 2U; ++half) {
        for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t scale_index = lane / 16U;
            const auto low_first = std::to_integer<std::uint8_t>(low[lane]);
            const auto low_second = std::to_integer<std::uint8_t>(low[lane + 32U]);
            const auto high_byte = std::to_integer<std::uint8_t>(high[lane]);
            const auto q1 = static_cast<std::int32_t>(
                (low_first & 15U) | (((high_byte >> 0U) & 3U) << 4U)) - 32;
            const auto q2 = static_cast<std::int32_t>(
                (low_second & 15U) | (((high_byte >> 2U) & 3U) << 4U)) - 32;
            const auto q3 = static_cast<std::int32_t>(
                (low_first >> 4U) | (((high_byte >> 4U) & 3U) << 4U)) - 32;
            const auto q4 = static_cast<std::int32_t>(
                (low_second >> 4U) | (((high_byte >> 6U) & 3U) << 4U)) - 32;
            const auto signed_scale = [scales](std::size_t index) {
                return std::bit_cast<std::int8_t>(
                    std::to_integer<std::uint8_t>(scales[index]));
            };
            const std::size_t base = half * 128U;
            output[base + lane] = scale * static_cast<float>(signed_scale(scale_index)) *
                                  static_cast<float>(q1);
            output[base + 32U + lane] =
                scale * static_cast<float>(signed_scale(scale_index + 2U)) *
                static_cast<float>(q2);
            output[base + 64U + lane] =
                scale * static_cast<float>(signed_scale(scale_index + 4U)) *
                static_cast<float>(q3);
            output[base + 96U + lane] =
                scale * static_cast<float>(signed_scale(scale_index + 6U)) *
                static_cast<float>(q4);
        }
        low += 64U;
        high += 32U;
        scales += 8U;
    }
    return {};
}

Result<void> validate_input(std::span<const float> input) {
    for (const float value : input) {
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_argument,
                              "Reference matvec input contains a non-finite value");
        }
    }
    return {};
}

template <std::size_t BlockElements, std::size_t BlockBytes, typename Validator,
          typename Decoder>
Result<void> dequantize_blocked(std::span<const std::byte> encoded,
                                std::span<float> output,
                                const QuantizedKernelLimits& limits,
                                Validator&& validator,
                                Decoder&& decoder) {
    if (output.empty() || output.size() % BlockElements != 0U ||
        output.size() > limits.max_elements) {
        return make_error(ErrorCode::invalid_argument,
                          "Blocked output size is empty, unaligned or outside its limit");
    }
    auto required = checked_encoded_bytes(output.size(), BlockElements, BlockBytes);
    if (!required) return required.error();
    if (encoded.size() != required.value()) {
        return make_error(ErrorCode::invalid_format,
                          "Blocked encoded bytes do not match the output element count");
    }
    auto validated = std::invoke(std::forward<Validator>(validator), encoded);
    if (!validated) return validated.error();
    for (std::size_t block = 0U; block < output.size() / BlockElements; ++block) {
        auto status = std::invoke(
            std::forward<Decoder>(decoder), encoded.data() + block * BlockBytes,
            std::span<float, BlockElements>(output.data() + block * BlockElements,
                                            BlockElements));
        if (!status) return status.error();
    }
    return {};
}

// Fixed set of worker threads shared by every matvec in the process.
//
// Bounded by construction: the worker count is set on first use and never grows, there is no queue -
// one dispatch is in flight at a time - and every dispatch joins before returning, so no work
// outlives the call that submitted it. Workers idle on a condition variable between dispatches.
class RowWorkPool {
public:
    explicit RowWorkPool(unsigned workers) {
        try {
            workers_.reserve(workers);
            for (unsigned index = 0U; index < workers; ++index) {
                workers_.emplace_back([this, index]() { serve(index); });
            }
        } catch (const std::system_error&) {
            // Fewer workers than asked for is fine; callers fall back to running inline.
        }
    }

    RowWorkPool(const RowWorkPool&) = delete;
    RowWorkPool& operator=(const RowWorkPool&) = delete;

    ~RowWorkPool() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

    // Runs `job(index)` for index in [0, helpers) on pool workers while the caller runs `inline_job`
    // itself, then waits for all of them. Returns false when the pool cannot serve the request.
    bool run(unsigned helpers,
             const std::function<void(unsigned)>& job,
             const std::function<void()>& inline_job) {
        if (helpers == 0U || helpers > workers_.size()) return false;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || job_ != nullptr) return false;
            job_ = &job;
            helpers_ = helpers;
            claimed_ = 0U;
            finished_ = 0U;
            ++generation_;
        }
        wake_.notify_all();
        inline_job();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this, helpers]() { return finished_ == helpers; });
        job_ = nullptr;
        return true;
    }

private:
    void serve(unsigned) {
        std::uint64_t seen = 0U;
        while (true) {
            unsigned index = 0U;
            const std::function<void(unsigned)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this, seen]() {
                    return stopping_ || (job_ != nullptr && generation_ != seen &&
                                         claimed_ < helpers_);
                });
                if (stopping_) return;
                if (job_ == nullptr || claimed_ >= helpers_) continue;
                index = claimed_++;
                job = job_;
                if (claimed_ >= helpers_) seen = generation_;
            }
            (*job)(index);
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                ++finished_;
                seen = generation_;
            }
            done_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::vector<std::thread> workers_;
    const std::function<void(unsigned)>* job_{nullptr};
    unsigned helpers_{0U};
    unsigned claimed_{0U};
    unsigned finished_{0U};
    std::uint64_t generation_{0U};
    bool stopping_{false};
};

// Sized on first use to half the logical processors, matching the largest budget the runtime policy
// will ever ask for. Sizing it to the full machine would leave the surplus workers waking on every
// dispatch only to find no share left for them.
RowWorkPool& row_work_pool() {
    static RowWorkPool pool(std::max(1U, std::thread::hardware_concurrency() / 2U));
    return pool;
}

template <std::size_t BlockElements, std::size_t BlockBytes, typename Validator,
          typename Decoder>
Result<void> matvec_blocked(std::span<const std::byte> encoded_matrix,
                            std::size_t rows,
                            std::size_t columns,
                            std::span<const float> input,
                            std::span<float> output,
                            const QuantizedKernelLimits& limits,
                            Validator&& validator,
                            Decoder&& decoder) {
    auto elements = checked_matrix_elements(rows, columns, BlockElements, limits);
    if (!elements) return elements.error();
    auto required = checked_encoded_bytes(elements.value(), BlockElements, BlockBytes);
    if (!required) return required.error();
    if (encoded_matrix.size() != required.value() || input.size() != columns ||
        output.size() != rows) {
        return make_error(ErrorCode::invalid_argument,
                          "Blocked matrix, input or output span has the wrong size");
    }
    auto input_status = validate_input(input);
    if (!input_status) return input_status.error();
    auto validated = std::invoke(std::forward<Validator>(validator), encoded_matrix);
    if (!validated) return validated.error();

    const std::size_t blocks_per_row = columns / BlockElements;

    // One row's work: decoded into a caller-owned scratch block so each worker keeps its own.
    const auto compute_rows = [&](std::size_t first, std::size_t last,
                                  std::array<float, BlockElements>& decoded) -> Result<void> {
        for (std::size_t row = first; row < last; ++row) {
            double sum = 0.0;
            for (std::size_t block = 0U; block < blocks_per_row; ++block) {
                const std::size_t encoded_block = row * blocks_per_row + block;
                auto status = std::invoke(
                    decoder, encoded_matrix.data() + encoded_block * BlockBytes,
                    std::span<float, BlockElements>(decoded));
                if (!status) return status.error();
                const std::size_t input_base = block * BlockElements;
                for (std::size_t column = 0U; column < BlockElements; ++column) {
                    sum += static_cast<double>(decoded[column]) *
                           static_cast<double>(input[input_base + column]);
                }
            }
            const float value = static_cast<float>(sum);
            if (!std::isfinite(value)) {
                return make_error(ErrorCode::invalid_format,
                                  "Blocked matvec produced a non-finite result");
            }
            output[row] = value;
        }
        return {};
    };

    // Below this a worker costs more than the rows it would take.
    constexpr std::size_t kMinimumRowsPerWorker = 64U;
    const unsigned requested = std::max(1U, limits.max_threads);
    const auto affordable = static_cast<unsigned>(
        std::min<std::size_t>(rows / kMinimumRowsPerWorker, requested));
    const unsigned workers = std::max(1U, affordable);

    if (workers == 1U) {
        std::array<float, BlockElements> decoded{};
        return compute_rows(0U, rows, decoded);
    }

    // A generation issues tens of thousands of matvecs, so spawning threads per call spends more on
    // starting workers than on arithmetic. The pool's workers are created once and reused.
    std::vector<Result<void>> outcomes(workers, Result<void>{});
    const std::size_t rows_per_worker = (rows + workers - 1U) / workers;
    const bool dispatched = row_work_pool().run(workers - 1U, [&](unsigned worker) {
        const std::size_t first = std::min(rows, (worker + 1U) * rows_per_worker);
        const std::size_t last = std::min(rows, first + rows_per_worker);
        if (first >= last) return;
        std::array<float, BlockElements> decoded{};
        outcomes[worker + 1U] = compute_rows(first, last, decoded);
    }, [&]() {
        std::array<float, BlockElements> decoded{};
        outcomes[0] = compute_rows(0U, std::min(rows, rows_per_worker), decoded);
    });
    if (!dispatched) {
        // No pool available: finish every row here rather than leaving the output half written.
        std::array<float, BlockElements> decoded{};
        return compute_rows(0U, rows, decoded);
    }
    for (auto& outcome : outcomes) {
        if (!outcome) return outcome.error();
    }
    return {};
}

template <std::size_t ElementBytes, typename Decoder>
Result<void> validate_dense_values(std::span<const std::byte> encoded, Decoder&& decoder) {
    if (encoded.size() % ElementBytes != 0U) {
        return make_error(ErrorCode::invalid_format,
                          "Dense encoded bytes are not element-aligned");
    }
    for (std::size_t offset = 0U; offset < encoded.size(); offset += ElementBytes) {
        if (!std::isfinite(std::invoke(std::forward<Decoder>(decoder),
                                       encoded.data() + offset))) {
            return make_error(ErrorCode::invalid_format,
                              "Dense tensor contains a non-finite value");
        }
    }
    return {};
}

template <std::size_t ElementBytes, typename Decoder>
Result<void> dequantize_dense(std::span<const std::byte> encoded,
                              std::span<float> output,
                              const QuantizedKernelLimits& limits,
                              Decoder&& decoder) {
    if (output.empty() || output.size() > limits.max_elements) {
        return make_error(ErrorCode::invalid_argument,
                          "Dense output size is empty or outside its limit");
    }
    auto required = checked_encoded_bytes(output.size(), 1U, ElementBytes);
    if (!required) return required.error();
    if (encoded.size() != required.value()) {
        return make_error(ErrorCode::invalid_format,
                          "Dense encoded bytes do not match the output element count");
    }
    auto validated = validate_dense_values<ElementBytes>(encoded, decoder);
    if (!validated) return validated.error();
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] = std::invoke(std::forward<Decoder>(decoder),
                                    encoded.data() + index * ElementBytes);
    }
    return {};
}

template <std::size_t ElementBytes, typename Decoder>
Result<void> matvec_dense(std::span<const std::byte> encoded_matrix,
                          std::size_t rows,
                          std::size_t columns,
                          std::span<const float> input,
                          std::span<float> output,
                          const QuantizedKernelLimits& limits,
                          Decoder&& decoder) {
    auto elements = checked_matrix_elements(rows, columns, 1U, limits);
    if (!elements) return elements.error();
    auto required = checked_encoded_bytes(elements.value(), 1U, ElementBytes);
    if (!required) return required.error();
    if (encoded_matrix.size() != required.value() || input.size() != columns ||
        output.size() != rows) {
        return make_error(ErrorCode::invalid_argument,
                          "Dense matrix, input or output span has the wrong size");
    }
    auto input_status = validate_input(input);
    if (!input_status) return input_status.error();
    auto validated = validate_dense_values<ElementBytes>(encoded_matrix, decoder);
    if (!validated) return validated.error();
    for (std::size_t row = 0U; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0U; column < columns; ++column) {
            const float weight = std::invoke(
                std::forward<Decoder>(decoder),
                encoded_matrix.data() + (row * columns + column) * ElementBytes);
            sum += static_cast<double>(weight) * static_cast<double>(input[column]);
        }
        const float value = static_cast<float>(sum);
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_format,
                              "Dense matvec produced a non-finite result");
        }
        output[row] = value;
    }
    return {};
}

bool overlaps(std::span<const float> first, std::span<const float> second) noexcept {
    if (first.empty() || second.empty()) return false;
    const std::less<const float*> less;
    return less(first.data(), second.data() + second.size()) &&
           less(second.data(), first.data() + first.size());
}

}  // namespace

Result<std::vector<std::byte>> quantize_q8_0_reference(
    std::span<const float> matrix,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns,
    const QuantizedKernelLimits& limits) {
    if (logical_columns == 0U || logical_columns > storage_columns ||
        storage_columns % kQ8_0BlockElements != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Q8_0 reference quantization dimensions are invalid");
    }
    auto elements = checked_matrix_elements(
        rows, storage_columns, kQ8_0BlockElements, limits);
    if (!elements) return elements.error();
    if (rows > std::numeric_limits<std::size_t>::max() / logical_columns ||
        matrix.size() != rows * logical_columns) {
        return make_error(ErrorCode::invalid_argument,
                          "Q8_0 reference source matrix shape does not match");
    }
    auto encoded_size = checked_encoded_bytes(
        elements.value(), kQ8_0BlockElements, kQ8_0BlockBytes);
    if (!encoded_size) return encoded_size.error();
    std::vector<std::byte> encoded;
    try {
        encoded.resize(encoded_size.value());
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded Q8_0 encoded buffer");
    }
    const std::size_t blocks_per_row = storage_columns / kQ8_0BlockElements;
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t block = 0U; block < blocks_per_row; ++block) {
            std::array<float, kQ8_0BlockElements> values{};
            float maximum = 0.0F;
            for (std::size_t index = 0U; index < values.size(); ++index) {
                const std::size_t column = block * values.size() + index;
                const float value = column < logical_columns
                                        ? matrix[row * logical_columns + column]
                                        : 0.0F;
                if (!std::isfinite(value)) {
                    return make_error(ErrorCode::invalid_argument,
                                      "Q8_0 reference source contains a non-finite value",
                                      std::to_string(row * logical_columns +
                                                     std::min(column, logical_columns - 1U)));
                }
                values[index] = value;
                maximum = std::max(maximum, std::abs(value));
            }
            const float scale = maximum / 127.0F;
            const float inverse = scale != 0.0F ? 1.0F / scale : 0.0F;
            const std::size_t encoded_offset =
                (row * blocks_per_row + block) * kQ8_0BlockBytes;
            const std::uint16_t encoded_scale = float_to_fp16(scale);
            if (!std::isfinite(fp16_to_float(encoded_scale))) {
                return make_error(ErrorCode::invalid_argument,
                                  "Q8_0 reference scale is not representable as FP16",
                                  std::to_string(row));
            }
            write_u16_le(encoded.data() + encoded_offset, encoded_scale);
            for (std::size_t index = 0U; index < values.size(); ++index) {
                const float rounded = std::round(values[index] * inverse);
                const auto quantized = static_cast<std::int8_t>(
                    std::clamp(rounded, -127.0F, 127.0F));
                encoded[encoded_offset + 2U + index] =
                    static_cast<std::byte>(std::bit_cast<std::uint8_t>(quantized));
            }
        }
    }
    return encoded;
}

bool supports_reference_matvec_ggml_type(std::uint32_t ggml_type) noexcept {
    switch (ggml_type) {
        case kGgmlTypeF32:
        case kGgmlTypeQ8_0:
        case kGgmlTypeQ4K:
        case kGgmlTypeQ5K:
        case kGgmlTypeQ6K:
        case kGgmlTypeIq2Xxs:
        case kGgmlTypeIq3Xxs:
        case kGgmlTypeIq1S:
        case kGgmlTypeMxfp4:
        case kGgmlTypeBf16: return true;
        default: return false;
    }
}

bool supports_reference_storage_ggml_type(std::uint32_t ggml_type) noexcept {
    return supports_reference_matvec_ggml_type(ggml_type) || ggml_type == kGgmlTypeI32;
}

Result<std::size_t> ggml_storage_bytes(std::uint32_t ggml_type,
                                       std::size_t rows,
                                       std::size_t columns,
                                       const QuantizedKernelLimits& limits) {
    std::size_t block_elements = 0U;
    std::size_t block_bytes = 0U;
    switch (ggml_type) {
        case kGgmlTypeF32:
            block_elements = 1U;
            block_bytes = kF32ElementBytes;
            break;
        case kGgmlTypeQ8_0:
            block_elements = kQ8_0BlockElements;
            block_bytes = kQ8_0BlockBytes;
            break;
        case kGgmlTypeQ4K:
            block_elements = kQ4KBlockElements;
            block_bytes = kQ4KBlockBytes;
            break;
        case kGgmlTypeQ5K:
            block_elements = kQ5KBlockElements;
            block_bytes = kQ5KBlockBytes;
            break;
        case kGgmlTypeQ6K:
            block_elements = kQ6KBlockElements;
            block_bytes = kQ6KBlockBytes;
            break;
        case kGgmlTypeIq2Xxs:
            block_elements = kIq2XxsBlockElements;
            block_bytes = kIq2XxsBlockBytes;
            break;
        case kGgmlTypeIq3Xxs:
            block_elements = kIq3XxsBlockElements;
            block_bytes = kIq3XxsBlockBytes;
            break;
        case kGgmlTypeIq1S:
            block_elements = kIq1SBlockElements;
            block_bytes = kIq1SBlockBytes;
            break;
        case kGgmlTypeI32:
            block_elements = 1U;
            block_bytes = kI32ElementBytes;
            break;
        case kGgmlTypeBf16:
            block_elements = 1U;
            block_bytes = kBf16ElementBytes;
            break;
        case kGgmlTypeMxfp4:
            block_elements = kMxfp4BlockElements;
            block_bytes = kMxfp4BlockBytes;
            break;
        default:
            return make_error(ErrorCode::unsupported,
                              "GGML storage type has no reference byte layout");
    }
    auto elements = checked_matrix_elements(rows, columns, block_elements, limits);
    if (!elements) return elements.error();
    return checked_encoded_bytes(elements.value(), block_elements, block_bytes);
}

Result<void> dequantize_q8_0(std::span<const std::byte> encoded,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) {
        return validate_block_scales(bytes, kQ8_0BlockBytes);
    };
    return dequantize_blocked<kQ8_0BlockElements, kQ8_0BlockBytes>(
        encoded, output, limits, validator, decode_q8_0_block);
}

Result<void> matvec_q8_0_f32(std::span<const std::byte> encoded_matrix,
                             std::size_t rows,
                             std::size_t columns,
                             std::span<const float> input,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) {
        return validate_block_scales(bytes, kQ8_0BlockBytes);
    };
    return matvec_blocked<kQ8_0BlockElements, kQ8_0BlockBytes>(
        encoded_matrix, rows, columns, input, output, limits, validator,
        decode_q8_0_block);
}

Result<void> dequantize_q4_k(std::span<const std::byte> encoded,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        for (std::size_t offset = 0U; offset < bytes.size(); offset += kQ4KBlockBytes) {
            const float scale = fp16_to_float(read_u16_le(bytes.data() + offset));
            const float minimum = fp16_to_float(read_u16_le(bytes.data() + offset + 2U));
            if (!std::isfinite(scale) || !std::isfinite(minimum)) {
                return make_error(ErrorCode::invalid_format,
                                  "Q4_K block scale or minimum is not finite");
            }
        }
        return {};
    };
    return dequantize_blocked<kQ4KBlockElements, kQ4KBlockBytes>(
        encoded, output, limits, validator, decode_q4_k_block);
}

Result<void> matvec_q4_k_f32(std::span<const std::byte> encoded_matrix,
                             std::size_t rows,
                             std::size_t columns,
                             std::span<const float> input,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        for (std::size_t offset = 0U; offset < bytes.size(); offset += kQ4KBlockBytes) {
            const float scale = fp16_to_float(read_u16_le(bytes.data() + offset));
            const float minimum = fp16_to_float(read_u16_le(bytes.data() + offset + 2U));
            if (!std::isfinite(scale) || !std::isfinite(minimum)) {
                return make_error(ErrorCode::invalid_format,
                                  "Q4_K block scale or minimum is not finite");
            }
        }
        return {};
    };
    return matvec_blocked<kQ4KBlockElements, kQ4KBlockBytes>(
        encoded_matrix, rows, columns, input, output, limits, validator,
        decode_q4_k_block);
}

Result<void> dequantize_q5_k(std::span<const std::byte> encoded,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        for (std::size_t offset = 0U; offset < bytes.size(); offset += kQ5KBlockBytes) {
            const float scale = fp16_to_float(read_u16_le(bytes.data() + offset));
            const float minimum = fp16_to_float(read_u16_le(bytes.data() + offset + 2U));
            if (!std::isfinite(scale) || !std::isfinite(minimum)) {
                return make_error(ErrorCode::invalid_format,
                                  "Q5_K block scale or minimum is not finite");
            }
        }
        return {};
    };
    return dequantize_blocked<kQ5KBlockElements, kQ5KBlockBytes>(
        encoded, output, limits, validator, decode_q5_k_block);
}

Result<void> matvec_q5_k_f32(std::span<const std::byte> encoded_matrix,
                             std::size_t rows,
                             std::size_t columns,
                             std::span<const float> input,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        for (std::size_t offset = 0U; offset < bytes.size(); offset += kQ5KBlockBytes) {
            const float scale = fp16_to_float(read_u16_le(bytes.data() + offset));
            const float minimum = fp16_to_float(read_u16_le(bytes.data() + offset + 2U));
            if (!std::isfinite(scale) || !std::isfinite(minimum)) {
                return make_error(ErrorCode::invalid_format,
                                  "Q5_K block scale or minimum is not finite");
            }
        }
        return {};
    };
    return matvec_blocked<kQ5KBlockElements, kQ5KBlockBytes>(
        encoded_matrix, rows, columns, input, output, limits, validator,
        decode_q5_k_block);
}

Result<void> dequantize_q6_k(std::span<const std::byte> encoded,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        for (std::size_t offset = 0U; offset < bytes.size(); offset += kQ6KBlockBytes) {
            if (!std::isfinite(fp16_to_float(read_u16_le(bytes.data() + offset + 208U)))) {
                return make_error(ErrorCode::invalid_format,
                                  "Q6_K block scale is not finite");
            }
        }
        return {};
    };
    return dequantize_blocked<kQ6KBlockElements, kQ6KBlockBytes>(
        encoded, output, limits, validator, decode_q6_k_block);
}

Result<void> matvec_q6_k_f32(std::span<const std::byte> encoded_matrix,
                             std::size_t rows,
                             std::size_t columns,
                             std::span<const float> input,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        for (std::size_t offset = 0U; offset < bytes.size(); offset += kQ6KBlockBytes) {
            if (!std::isfinite(fp16_to_float(read_u16_le(bytes.data() + offset + 208U)))) {
                return make_error(ErrorCode::invalid_format,
                                  "Q6_K block scale is not finite");
            }
        }
        return {};
    };
    return matvec_blocked<kQ6KBlockElements, kQ6KBlockBytes>(
        encoded_matrix, rows, columns, input, output, limits, validator,
        decode_q6_k_block);
}

Result<void> dequantize_iq2_xxs(std::span<const std::byte> encoded,
                                std::span<float> output,
                                const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        return validate_block_scales(bytes, kIq2XxsBlockBytes);
    };
    return dequantize_blocked<kIq2XxsBlockElements, kIq2XxsBlockBytes>(
        encoded, output, limits, validator, decode_iq2_xxs_block);
}

Result<void> matvec_iq2_xxs_f32(std::span<const std::byte> encoded_matrix,
                                std::size_t rows,
                                std::size_t columns,
                                std::span<const float> input,
                                std::span<float> output,
                                const QuantizedKernelLimits& limits) {
    const auto validator = [](std::span<const std::byte> bytes) -> Result<void> {
        return validate_block_scales(bytes, kIq2XxsBlockBytes);
    };
    return matvec_blocked<kIq2XxsBlockElements, kIq2XxsBlockBytes>(
        encoded_matrix, rows, columns, input, output, limits, validator,
        decode_iq2_xxs_block);
}

Result<void> dequantize_iq1_s(std::span<const std::byte> encoded,
                              std::span<float> output,
                              const QuantizedKernelLimits& limits) {
    if (output.empty() || output.size() % kIq1SBlockElements != 0U ||
        output.size() > limits.max_elements) {
        return make_error(ErrorCode::invalid_argument,
                          "IQ1_S output size is empty, unaligned or outside its limit");
    }
    auto required = checked_encoded_bytes(
        output.size(), kIq1SBlockElements, kIq1SBlockBytes);
    if (!required) return required.error();
    if (encoded.size() != required.value()) {
        return make_error(ErrorCode::invalid_format,
                          "IQ1_S encoded bytes do not match the output element count");
    }
    auto validated = validate_block_scales(encoded, kIq1SBlockBytes);
    if (!validated) return validated.error();
    for (std::size_t block = 0U; block < output.size() / kIq1SBlockElements; ++block) {
        auto decoded = decode_iq1_s_block(
            encoded.data() + block * kIq1SBlockBytes,
            std::span<float, kIq1SBlockElements>(
                output.data() + block * kIq1SBlockElements, kIq1SBlockElements));
        if (!decoded) return decoded.error();
    }
    return {};
}

Result<void> matvec_iq1_s_f32(std::span<const std::byte> encoded_matrix,
                              std::size_t rows,
                              std::size_t columns,
                              std::span<const float> input,
                              std::span<float> output,
                              const QuantizedKernelLimits& limits) {
    auto elements = checked_matrix_elements(rows, columns, kIq1SBlockElements, limits);
    if (!elements) return elements.error();
    auto required = checked_encoded_bytes(
        elements.value(), kIq1SBlockElements, kIq1SBlockBytes);
    if (!required) return required.error();
    if (encoded_matrix.size() != required.value() || input.size() != columns ||
        output.size() != rows) {
        return make_error(ErrorCode::invalid_argument,
                          "IQ1_S matrix, input or output span has the wrong size");
    }
    for (const float value : input) {
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_argument,
                              "IQ1_S matvec input contains a non-finite value");
        }
    }
    auto validated = validate_block_scales(encoded_matrix, kIq1SBlockBytes);
    if (!validated) return validated.error();

    const std::size_t blocks_per_row = columns / kIq1SBlockElements;
    std::array<float, kIq1SBlockElements> decoded{};
    for (std::size_t row = 0U; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t block = 0U; block < blocks_per_row; ++block) {
            const std::size_t encoded_block = row * blocks_per_row + block;
            auto status = decode_iq1_s_block(
                encoded_matrix.data() + encoded_block * kIq1SBlockBytes,
                std::span<float, kIq1SBlockElements>(decoded));
            if (!status) return status.error();
            const std::size_t input_base = block * kIq1SBlockElements;
            for (std::size_t column = 0U; column < kIq1SBlockElements; ++column) {
                sum += static_cast<double>(decoded[column]) *
                       static_cast<double>(input[input_base + column]);
            }
        }
        const float value = static_cast<float>(sum);
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_format,
                              "IQ1_S matvec produced a non-finite result");
        }
        output[row] = value;
    }
    return {};
}

Result<void> dequantize_iq3_xxs(std::span<const std::byte> encoded,
                                std::span<float> output,
                                const QuantizedKernelLimits& limits) {
    if (output.empty() || output.size() % kIq3XxsBlockElements != 0U ||
        output.size() > limits.max_elements) {
        return make_error(ErrorCode::invalid_argument,
                          "IQ3_XXS output size is empty, unaligned or outside its limit");
    }
    auto required = checked_encoded_bytes(
        output.size(), kIq3XxsBlockElements, kIq3XxsBlockBytes);
    if (!required) return required.error();
    if (encoded.size() != required.value()) {
        return make_error(ErrorCode::invalid_format,
                          "IQ3_XXS encoded bytes do not match the output element count");
    }
    auto validated = validate_block_scales(encoded, kIq3XxsBlockBytes);
    if (!validated) return validated.error();
    for (std::size_t block = 0U; block < output.size() / kIq3XxsBlockElements; ++block) {
        auto decoded = decode_iq3_xxs_block(
            encoded.data() + block * kIq3XxsBlockBytes,
            std::span<float, kIq3XxsBlockElements>(
                output.data() + block * kIq3XxsBlockElements, kIq3XxsBlockElements));
        if (!decoded) return decoded.error();
    }
    return {};
}

Result<void> matvec_iq3_xxs_f32(std::span<const std::byte> encoded_matrix,
                                std::size_t rows,
                                std::size_t columns,
                                std::span<const float> input,
                                std::span<float> output,
                                const QuantizedKernelLimits& limits) {
    auto elements = checked_matrix_elements(rows, columns, kIq3XxsBlockElements, limits);
    if (!elements) return elements.error();
    auto required = checked_encoded_bytes(
        elements.value(), kIq3XxsBlockElements, kIq3XxsBlockBytes);
    if (!required) return required.error();
    if (encoded_matrix.size() != required.value() || input.size() != columns ||
        output.size() != rows) {
        return make_error(ErrorCode::invalid_argument,
                          "IQ3_XXS matrix, input or output span has the wrong size");
    }
    for (const float value : input) {
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_argument,
                              "IQ3_XXS matvec input contains a non-finite value");
        }
    }
    auto validated = validate_block_scales(encoded_matrix, kIq3XxsBlockBytes);
    if (!validated) return validated.error();

    const std::size_t blocks_per_row = columns / kIq3XxsBlockElements;
    std::array<float, kIq3XxsBlockElements> decoded{};
    for (std::size_t row = 0U; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t block = 0U; block < blocks_per_row; ++block) {
            const std::size_t encoded_block = row * blocks_per_row + block;
            auto status = decode_iq3_xxs_block(
                encoded_matrix.data() + encoded_block * kIq3XxsBlockBytes,
                std::span<float, kIq3XxsBlockElements>(decoded));
            if (!status) return status.error();
            const std::size_t input_base = block * kIq3XxsBlockElements;
            for (std::size_t column = 0U; column < kIq3XxsBlockElements; ++column) {
                sum += static_cast<double>(decoded[column]) *
                       static_cast<double>(input[input_base + column]);
            }
        }
        const float value = static_cast<float>(sum);
        if (!std::isfinite(value)) {
            return make_error(ErrorCode::invalid_format,
                              "IQ3_XXS matvec produced a non-finite result");
        }
        output[row] = value;
    }
    return {};
}

Result<void> dequantize_mxfp4(std::span<const std::byte> encoded,
                              std::span<float> output,
                              const QuantizedKernelLimits& limits) {
    return dequantize_blocked<kMxfp4BlockElements, kMxfp4BlockBytes>(
        encoded, output, limits, validate_mxfp4_blocks, decode_mxfp4_block);
}

Result<void> matvec_mxfp4_f32(std::span<const std::byte> encoded_matrix,
                              std::size_t rows,
                              std::size_t columns,
                              std::span<const float> input,
                              std::span<float> output,
                              const QuantizedKernelLimits& limits) {
    return matvec_blocked<kMxfp4BlockElements, kMxfp4BlockBytes>(
        encoded_matrix, rows, columns, input, output, limits,
        validate_mxfp4_blocks, decode_mxfp4_block);
}

Result<void> dequantize_bf16(std::span<const std::byte> encoded,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto decoder = [](const std::byte* bytes) {
        return bf16_to_float(read_u16_le(bytes));
    };
    return dequantize_dense<kBf16ElementBytes>(encoded, output, limits, decoder);
}

Result<void> matvec_bf16_f32(std::span<const std::byte> encoded_matrix,
                             std::size_t rows,
                             std::size_t columns,
                             std::span<const float> input,
                             std::span<float> output,
                             const QuantizedKernelLimits& limits) {
    const auto decoder = [](const std::byte* bytes) {
        return bf16_to_float(read_u16_le(bytes));
    };
    return matvec_dense<kBf16ElementBytes>(
        encoded_matrix, rows, columns, input, output, limits, decoder);
}

Result<void> dequantize_f32_le(std::span<const std::byte> encoded,
                               std::span<float> output,
                               const QuantizedKernelLimits& limits) {
    return dequantize_dense<kF32ElementBytes>(encoded, output, limits, read_f32_le);
}

Result<void> matvec_f32_le_f32(std::span<const std::byte> encoded_matrix,
                               std::size_t rows,
                               std::size_t columns,
                               std::span<const float> input,
                               std::span<float> output,
                               const QuantizedKernelLimits& limits) {
    return matvec_dense<kF32ElementBytes>(
        encoded_matrix, rows, columns, input, output, limits, read_f32_le);
}

Result<void> decode_i32_le(std::span<const std::byte> encoded,
                           std::span<std::int32_t> output,
                           const QuantizedKernelLimits& limits) {
    if (output.empty() || output.size() > limits.max_elements) {
        return make_error(ErrorCode::invalid_argument,
                          "I32 output size is empty or outside its limit");
    }
    auto required = checked_encoded_bytes(output.size(), 1U, kI32ElementBytes);
    if (!required) return required.error();
    if (encoded.size() != required.value()) {
        return make_error(ErrorCode::invalid_format,
                          "I32 encoded bytes do not match the output element count");
    }
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] = std::bit_cast<std::int32_t>(
            read_u32_le(encoded.data() + index * kI32ElementBytes));
    }
    return {};
}

Result<void> dequantize_ggml_row_f32(
    std::uint32_t ggml_type,
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::size_t row,
    std::span<float> output,
    const QuantizedKernelLimits& limits) {
    if (!supports_reference_matvec_ggml_type(ggml_type)) {
        return make_error(ErrorCode::unsupported,
                          "GGML storage type has no reference row decoder");
    }
    if (row >= rows || output.size() != columns) {
        return make_error(ErrorCode::invalid_argument,
                          "GGML row index or output span has the wrong size");
    }
    auto total_bytes = ggml_storage_bytes(ggml_type, rows, columns, limits);
    if (!total_bytes) return total_bytes.error();
    if (encoded_matrix.size() != total_bytes.value()) {
        return make_error(ErrorCode::invalid_format,
                          "GGML matrix bytes do not match the row descriptor");
    }
    auto row_bytes = ggml_storage_bytes(
        ggml_type, 1U, columns, QuantizedKernelLimits{columns});
    if (!row_bytes) return row_bytes.error();
    if (row > std::numeric_limits<std::size_t>::max() / row_bytes.value()) {
        return make_error(ErrorCode::resource_exhausted,
                          "GGML row byte offset overflows");
    }
    const auto encoded_row = encoded_matrix.subspan(row * row_bytes.value(),
                                                     row_bytes.value());
    switch (ggml_type) {
        case kGgmlTypeF32: return dequantize_f32_le(encoded_row, output, limits);
        case kGgmlTypeQ8_0: return dequantize_q8_0(encoded_row, output, limits);
        case kGgmlTypeQ4K: return dequantize_q4_k(encoded_row, output, limits);
        case kGgmlTypeQ5K: return dequantize_q5_k(encoded_row, output, limits);
        case kGgmlTypeQ6K: return dequantize_q6_k(encoded_row, output, limits);
        case kGgmlTypeIq2Xxs: return dequantize_iq2_xxs(encoded_row, output, limits);
        case kGgmlTypeIq3Xxs: return dequantize_iq3_xxs(encoded_row, output, limits);
        case kGgmlTypeIq1S: return dequantize_iq1_s(encoded_row, output, limits);
        case kGgmlTypeMxfp4: return dequantize_mxfp4(encoded_row, output, limits);
        case kGgmlTypeBf16: return dequantize_bf16(encoded_row, output, limits);
        default:
            return make_error(ErrorCode::unsupported,
                              "GGML storage type has no reference row decoder");
    }
}

Result<void> matvec_ggml_f32_transactional(
    std::uint32_t ggml_type,
    std::span<const std::byte> encoded_matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const float> input,
    std::span<float> scratch,
    std::span<float> output,
    const QuantizedKernelLimits& limits) {
    if (!supports_reference_matvec_ggml_type(ggml_type)) {
        return make_error(ErrorCode::unsupported,
                          "GGML storage type has no reference matvec kernel");
    }
    if (rows == 0U || scratch.size() != rows || output.size() != rows) {
        return make_error(ErrorCode::invalid_argument,
                          "Transactional matvec scratch or output has the wrong size");
    }
    const std::span<const float> scratch_view = scratch;
    const std::span<const float> output_view = output;
    if (overlaps(input, scratch_view) || overlaps(input, output_view) ||
        overlaps(scratch_view, output_view)) {
        return make_error(ErrorCode::invalid_argument,
                          "Transactional matvec buffers overlap");
    }

    Result<void> status;
    switch (ggml_type) {
        case kGgmlTypeF32:
            status = matvec_f32_le_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeQ8_0:
            status = matvec_q8_0_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeQ4K:
            status = matvec_q4_k_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeQ5K:
            status = matvec_q5_k_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeQ6K:
            status = matvec_q6_k_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeIq2Xxs:
            status = matvec_iq2_xxs_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeIq3Xxs:
            status = matvec_iq3_xxs_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeIq1S:
            status = matvec_iq1_s_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeMxfp4:
            status = matvec_mxfp4_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        case kGgmlTypeBf16:
            status = matvec_bf16_f32(
                encoded_matrix, rows, columns, input, scratch, limits);
            break;
        default:
            return make_error(ErrorCode::unsupported,
                              "GGML storage type has no reference matvec kernel");
    }
    if (!status) return status.error();
    std::copy(scratch.begin(), scratch.end(), output.begin());
    return {};
}

Result<void> routed_expert_iq1_s_iq3_xxs_f32(
    std::span<const std::byte> gate_matrix,
    std::span<const std::byte> up_matrix,
    std::span<const std::byte> down_matrix,
    std::span<const float> input,
    RoutedExpertScratch scratch,
    std::span<float> output,
    const QuantizedKernelLimits& limits) {
    if (input.empty() || output.empty() || scratch.gate.empty() ||
        scratch.up.size() != scratch.gate.size() || scratch.down.size() != output.size()) {
        return make_error(ErrorCode::invalid_argument,
                          "Routed expert input, output or scratch dimensions are invalid");
    }
    const std::span<const float> gate_scratch = scratch.gate;
    const std::span<const float> up_scratch = scratch.up;
    const std::span<const float> down_scratch = scratch.down;
    const std::span<const float> final_output = output;
    if (overlaps(input, gate_scratch) || overlaps(input, up_scratch) ||
        overlaps(input, down_scratch) || overlaps(gate_scratch, up_scratch) ||
        overlaps(gate_scratch, down_scratch) || overlaps(up_scratch, down_scratch) ||
        overlaps(final_output, gate_scratch) || overlaps(final_output, up_scratch) ||
        overlaps(final_output, down_scratch)) {
        return make_error(ErrorCode::invalid_argument,
                          "Routed expert scratch buffers overlap live input or output");
    }

    const std::size_t expert_width = scratch.gate.size();
    auto gate_status = matvec_iq1_s_f32(
        gate_matrix, expert_width, input.size(), input, scratch.gate, limits);
    if (!gate_status) return gate_status.error();
    auto up_status = matvec_iq1_s_f32(
        up_matrix, expert_width, input.size(), input, scratch.up, limits);
    if (!up_status) return up_status.error();
    for (std::size_t index = 0U; index < expert_width; ++index) {
        const float gate = scratch.gate[index];
        const float activated = gate / (1.0F + std::exp(-gate)) * scratch.up[index];
        if (!std::isfinite(activated)) {
            return make_error(ErrorCode::invalid_format,
                              "Routed expert SwiGLU produced a non-finite activation");
        }
        scratch.gate[index] = activated;
    }
    auto down_status = matvec_iq3_xxs_f32(
        down_matrix, output.size(), expert_width, scratch.gate, scratch.down, limits);
    if (!down_status) return down_status.error();
    std::copy(scratch.down.begin(), scratch.down.end(), output.begin());
    return {};
}

}  // namespace vastloom::inference
