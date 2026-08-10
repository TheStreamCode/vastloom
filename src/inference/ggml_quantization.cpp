#include "vastloom/inference/ggml_quantization.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>

namespace vastloom::inference {
namespace {

// The IQ1_S codebook is derived from ggml commit
// 30bf8685ed4eb0a47f2b06229543327749904150 under its MIT license. See NOTICE and
// third_party/ggml/LICENSE. Keeping the immutable codebook local avoids a runtime dependency.
constexpr std::array<std::uint64_t, 2'048U> kIq1SGrid{
#include "iq1s_grid.inc"
};

constexpr std::array<std::uint32_t, 256U> kIq3XxsGrid{
#include "iq3xxs_grid.inc"
};

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

bool overlaps(std::span<const float> first, std::span<const float> second) noexcept {
    if (first.empty() || second.empty()) return false;
    const std::less<const float*> less;
    return less(first.data(), second.data() + second.size()) &&
           less(second.data(), first.data() + first.size());
}

}  // namespace

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
