#include "vastloom/inference/ggml_quantization.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
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

Result<std::size_t> checked_matrix_elements(std::size_t rows,
                                            std::size_t columns,
                                            const QuantizedKernelLimits& limits) {
    if (limits.max_elements == 0U || rows == 0U || columns == 0U ||
        columns % kIq1SBlockElements != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "IQ1_S matrix dimensions or kernel limits are invalid");
    }
    if (rows > limits.max_elements / columns) {
        return make_error(ErrorCode::resource_exhausted,
                          "IQ1_S matrix exceeds the configured element limit");
    }
    return rows * columns;
}

Result<std::size_t> checked_encoded_bytes(std::size_t elements) {
    const std::size_t blocks = elements / kIq1SBlockElements;
    if (blocks > std::numeric_limits<std::size_t>::max() / kIq1SBlockBytes) {
        return make_error(ErrorCode::resource_exhausted,
                          "IQ1_S encoded byte count overflows");
    }
    return blocks * kIq1SBlockBytes;
}

Result<void> validate_block_scales(std::span<const std::byte> encoded) {
    for (std::size_t offset = 0U; offset < encoded.size(); offset += kIq1SBlockBytes) {
        if (!std::isfinite(fp16_to_float(read_u16_le(encoded.data() + offset)))) {
            return make_error(ErrorCode::invalid_format,
                              "IQ1_S block scale is not finite");
        }
    }
    return {};
}

Result<void> decode_block(const std::byte* encoded, std::span<float, kIq1SBlockElements> output) {
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

}  // namespace

Result<void> dequantize_iq1_s(std::span<const std::byte> encoded,
                              std::span<float> output,
                              const QuantizedKernelLimits& limits) {
    if (output.empty() || output.size() % kIq1SBlockElements != 0U ||
        output.size() > limits.max_elements) {
        return make_error(ErrorCode::invalid_argument,
                          "IQ1_S output size is empty, unaligned or outside its limit");
    }
    auto required = checked_encoded_bytes(output.size());
    if (!required) return required.error();
    if (encoded.size() != required.value()) {
        return make_error(ErrorCode::invalid_format,
                          "IQ1_S encoded bytes do not match the output element count");
    }
    auto validated = validate_block_scales(encoded);
    if (!validated) return validated.error();
    for (std::size_t block = 0U; block < output.size() / kIq1SBlockElements; ++block) {
        auto decoded = decode_block(
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
    auto elements = checked_matrix_elements(rows, columns, limits);
    if (!elements) return elements.error();
    auto required = checked_encoded_bytes(elements.value());
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
    auto validated = validate_block_scales(encoded_matrix);
    if (!validated) return validated.error();

    const std::size_t blocks_per_row = columns / kIq1SBlockElements;
    std::array<float, kIq1SBlockElements> decoded{};
    for (std::size_t row = 0U; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t block = 0U; block < blocks_per_row; ++block) {
            const std::size_t encoded_block = row * blocks_per_row + block;
            auto status = decode_block(
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

}  // namespace vastloom::inference
