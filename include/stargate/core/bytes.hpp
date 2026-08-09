#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "stargate/core/status.hpp"

namespace stargate {

using ByteCount = std::uint64_t;

inline constexpr ByteCount KiB = 1024ULL;
inline constexpr ByteCount MiB = 1024ULL * KiB;
inline constexpr ByteCount GiB = 1024ULL * MiB;
inline constexpr ByteCount TiB = 1024ULL * GiB;

[[nodiscard]] Result<ByteCount> parse_bytes(std::string_view input);
[[nodiscard]] std::string format_bytes(ByteCount bytes, int precision = 2);
[[nodiscard]] Result<ByteCount> checked_add(ByteCount left, ByteCount right);
[[nodiscard]] Result<ByteCount> checked_sub(ByteCount left, ByteCount right);
[[nodiscard]] ByteCount percentage(ByteCount value, double ratio) noexcept;

}  // namespace stargate
