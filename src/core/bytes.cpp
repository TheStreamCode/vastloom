#include "vastloom/core/bytes.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>

#include "vastloom/util/string.hpp"

namespace vastloom {

Result<ByteCount> parse_bytes(std::string_view input) {
    std::string normalized = util::trim(input);
    normalized.erase(std::remove(normalized.begin(), normalized.end(), '_'), normalized.end());
    if (normalized.empty()) {
        return make_error(ErrorCode::invalid_argument, "Byte value is empty");
    }

    char* end = nullptr;
    errno = 0;
    const long double numeric = std::strtold(normalized.c_str(), &end);
    if (end == normalized.c_str() || errno == ERANGE || !std::isfinite(static_cast<double>(numeric)) || numeric < 0.0L) {
        return make_error(ErrorCode::invalid_argument, "Invalid byte value", normalized);
    }

    std::string suffix = util::to_lower(util::trim(std::string_view(end)));
    long double multiplier = 1.0L;
    if (suffix.empty() || suffix == "b") {
        multiplier = 1.0L;
    } else if (suffix == "kb") {
        multiplier = 1'000.0L;
    } else if (suffix == "mb") {
        multiplier = 1'000'000.0L;
    } else if (suffix == "gb") {
        multiplier = 1'000'000'000.0L;
    } else if (suffix == "tb") {
        multiplier = 1'000'000'000'000.0L;
    } else if (suffix == "kib" || suffix == "k") {
        multiplier = static_cast<long double>(KiB);
    } else if (suffix == "mib" || suffix == "m") {
        multiplier = static_cast<long double>(MiB);
    } else if (suffix == "gib" || suffix == "g") {
        multiplier = static_cast<long double>(GiB);
    } else if (suffix == "tib" || suffix == "t") {
        multiplier = static_cast<long double>(TiB);
    } else {
        return make_error(ErrorCode::invalid_argument, "Unknown byte suffix", suffix);
    }

    const long double bytes = numeric * multiplier;
    if (bytes > static_cast<long double>(std::numeric_limits<ByteCount>::max())) {
        return make_error(ErrorCode::invalid_argument, "Byte value exceeds uint64 range", normalized);
    }
    return static_cast<ByteCount>(std::llround(bytes));
}

std::string format_bytes(ByteCount bytes, int precision) {
    struct Unit {
        ByteCount divisor;
        const char* suffix;
    };
    constexpr std::array<Unit, 5> units{{
        {TiB, "TiB"},
        {GiB, "GiB"},
        {MiB, "MiB"},
        {KiB, "KiB"},
        {1, "B"},
    }};

    const auto selected = std::find_if(units.begin(), units.end(), [bytes](const Unit& unit) {
        return bytes >= unit.divisor;
    });
    const Unit& unit = selected == units.end() ? units.back() : *selected;
    std::ostringstream output;
    output << std::fixed << std::setprecision(unit.divisor == 1 ? 0 : precision)
           << static_cast<long double>(bytes) / static_cast<long double>(unit.divisor)
           << ' ' << unit.suffix;
    return output.str();
}

Result<ByteCount> checked_add(ByteCount left, ByteCount right) {
    if (right > std::numeric_limits<ByteCount>::max() - left) {
        return make_error(ErrorCode::resource_exhausted, "Byte count addition overflow");
    }
    return left + right;
}

Result<ByteCount> checked_sub(ByteCount left, ByteCount right) {
    if (right > left) {
        return make_error(ErrorCode::resource_exhausted, "Byte count subtraction underflow");
    }
    return left - right;
}

ByteCount percentage(ByteCount value, double ratio) noexcept {
    if (ratio <= 0.0) {
        return 0;
    }
    if (ratio >= 1.0) {
        return value;
    }
    return static_cast<ByteCount>(static_cast<long double>(value) * static_cast<long double>(ratio));
}

}  // namespace vastloom
