#include "stargate/util/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace stargate::util {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants{{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
}};

constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned amount) noexcept {
    return (value >> amount) | (value << (32U - amount));
}

}  // namespace

Sha256::Sha256()
    : state_{{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}} {}

void Sha256::update(std::span<const std::byte> data) {
    update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data.data()), data.size()));
}

void Sha256::update(std::span<const std::uint8_t> data) {
    if (finalized_ || data.empty()) {
        return;
    }
    total_bytes_ += static_cast<std::uint64_t>(data.size());
    std::size_t offset = 0;
    if (buffer_size_ != 0) {
        const auto copy_size = std::min(buffer_.size() - buffer_size_, data.size());
        std::copy_n(data.data(), copy_size, buffer_.data() + buffer_size_);
        buffer_size_ += copy_size;
        offset += copy_size;
        if (buffer_size_ == buffer_.size()) {
            transform(buffer_.data());
            buffer_size_ = 0;
        }
    }
    while (offset + buffer_.size() <= data.size()) {
        transform(data.data() + offset);
        offset += buffer_.size();
    }
    if (offset < data.size()) {
        buffer_size_ = data.size() - offset;
        std::copy_n(data.data() + offset, buffer_size_, buffer_.data());
    }
}

std::array<std::uint8_t, 32> Sha256::finalize() {
    if (!finalized_) {
        const std::uint64_t bit_length = total_bytes_ * 8ULL;
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56U) {
            std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                      buffer_.end(),
                      std::uint8_t{0});
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                  buffer_.begin() + 56,
                  std::uint8_t{0});
        for (std::size_t index = 0; index < 8U; ++index) {
            buffer_[63U - index] = static_cast<std::uint8_t>((bit_length >> (index * 8U)) & 0xffU);
        }
        transform(buffer_.data());
        finalized_ = true;
    }

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t index = 0; index < state_.size(); ++index) {
        digest[index * 4U] = static_cast<std::uint8_t>((state_[index] >> 24U) & 0xffU);
        digest[index * 4U + 1U] = static_cast<std::uint8_t>((state_[index] >> 16U) & 0xffU);
        digest[index * 4U + 2U] = static_cast<std::uint8_t>((state_[index] >> 8U) & 0xffU);
        digest[index * 4U + 3U] = static_cast<std::uint8_t>(state_[index] & 0xffU);
    }
    return digest;
}

std::string Sha256::finalize_hex() {
    const auto digest = finalize();
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : digest) {
        output << std::setw(2) << static_cast<unsigned>(byte);
    }
    return output.str();
}

void Sha256::transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
        const std::size_t base = index * 4U;
        words[index] = (static_cast<std::uint32_t>(block[base]) << 24U) |
                       (static_cast<std::uint32_t>(block[base + 1U]) << 16U) |
                       (static_cast<std::uint32_t>(block[base + 2U]) << 8U) |
                       static_cast<std::uint32_t>(block[base + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
        const std::uint32_t s0 = rotate_right(words[index - 15U], 7U) ^ rotate_right(words[index - 15U], 18U) ^ (words[index - 15U] >> 3U);
        const std::uint32_t s1 = rotate_right(words[index - 2U], 17U) ^ rotate_right(words[index - 2U], 19U) ^ (words[index - 2U] >> 10U);
        words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const std::uint32_t choose = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + sum1 + choose + kRoundConstants[index] + words[index];
        const std::uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

Result<std::string> sha256_file(const std::filesystem::path& path, std::size_t buffer_size) {
    if (buffer_size == 0U || buffer_size > 64U * 1024U * 1024U) {
        return make_error(ErrorCode::invalid_argument, "SHA-256 buffer size is outside the safe range", path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return make_error(ErrorCode::io_error, "Unable to open file for SHA-256", path.string());
    }
    std::vector<std::uint8_t> buffer(buffer_size);
    Sha256 hash;
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto read = input.gcount();
        if (read > 0) {
            hash.update(std::span<const std::uint8_t>(buffer.data(), static_cast<std::size_t>(read)));
        }
    }
    if (!input.eof()) {
        return make_error(ErrorCode::io_error, "I/O error while hashing file", path.string());
    }
    return hash.finalize_hex();
}

bool is_sha256_hex(std::string_view value) noexcept {
    if (value.size() != 64U) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f') ||
               (character >= 'A' && character <= 'F');
    });
}

}  // namespace stargate::util
