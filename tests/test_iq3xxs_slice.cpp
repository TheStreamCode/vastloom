#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/util/sha256.hpp"

namespace {

constexpr std::size_t kRows = 4'096U;
constexpr std::size_t kColumns = 2'048U;
constexpr std::size_t kEncodedBytes =
    kRows * (kColumns / vastloom::inference::kIq3XxsBlockElements) *
    vastloom::inference::kIq3XxsBlockBytes;
constexpr std::string_view kExpectedSha256 =
    "65b9cdc96f9e753584dc6eac4723616a494a98113bf25836fcda5ec72c400204";
constexpr std::uint64_t kExpectedOutputFnv64 = UINT64_C(0x15774ed1fd7eb1a0);

std::uint64_t output_fingerprint(std::span<const float> values) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : values) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

int fail(std::string_view message) {
    std::cerr << "IQ3_XXS real-slice test failed: " << message << '\n';
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return fail("expected exactly one slice path");
    const std::filesystem::path path(argv[1]);
    std::error_code error;
    if (std::filesystem::is_symlink(path, error) || error ||
        !std::filesystem::is_regular_file(path, error) || error) {
        return fail("slice must be a regular non-symlink file");
    }
    if (std::filesystem::file_size(path, error) != kEncodedBytes || error) {
        return fail("slice byte size does not match the pinned tensor");
    }
    auto digest = vastloom::util::sha256_file(path);
    if (!digest || digest.value() != kExpectedSha256) {
        return fail("slice SHA-256 does not match the pinned range");
    }

    std::vector<std::byte> encoded(kEncodedBytes);
    std::ifstream input_file(path, std::ios::binary);
    input_file.read(reinterpret_cast<char*>(encoded.data()),
                    static_cast<std::streamsize>(encoded.size()));
    if (input_file.gcount() != static_cast<std::streamsize>(encoded.size()) ||
        input_file.peek() != std::ifstream::traits_type::eof()) {
        return fail("slice read was short or contained trailing data");
    }

    std::vector<float> input(kColumns);
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const int centered = static_cast<int>(column % 29U) - 14;
        input[column] = static_cast<float>(centered) * 0.03125F +
                        static_cast<float>(column % 3U) * 0.0078125F;
    }
    std::vector<float> output(kRows, std::numeric_limits<float>::quiet_NaN());
    auto multiplied = vastloom::inference::matvec_iq3_xxs_f32(
        encoded, kRows, kColumns, input, output);
    if (!multiplied) return fail(multiplied.error().message);
    if (output_fingerprint(output) != kExpectedOutputFnv64 ||
        output.front() != 0.156733781F || output.back() != 0.393542588F) {
        return fail("matrix-vector output differs from the pinned ggml oracle");
    }

    std::cout << "DeepSeek V4 Flash 0731 IQ3_XXS real slice: PASS\n";
    return 0;
}
