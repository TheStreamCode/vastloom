#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "vastloom/core/status.hpp"

namespace vastloom::util {

class Sha256 {
public:
    Sha256();

    void update(std::span<const std::byte> data);
    void update(std::span<const std::uint8_t> data);
    [[nodiscard]] std::array<std::uint8_t, 32> finalize();
    [[nodiscard]] std::string finalize_hex();

private:
    void transform(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t total_bytes_{0};
    std::size_t buffer_size_{0};
    bool finalized_{false};
};

[[nodiscard]] Result<std::string> sha256_file(const std::filesystem::path& path,
                                              std::size_t buffer_size = 4U * 1024U * 1024U);
[[nodiscard]] bool is_sha256_hex(std::string_view value) noexcept;

}  // namespace vastloom::util
