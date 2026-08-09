#pragma once

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace stargate::util {

inline std::string trim(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

inline std::string to_lower(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return result;
}

inline std::vector<std::string> split(std::string_view value, char delimiter, bool keep_empty = true) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(delimiter, start);
        const auto length = end == std::string_view::npos ? value.size() - start : end - start;
        if (keep_empty || length != 0) {
            parts.emplace_back(value.substr(start, length));
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return parts;
}

inline bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

inline bool is_utf8_continuation(unsigned char value) noexcept {
    return value >= 0x80U && value <= 0xbfU;
}

inline std::size_t valid_utf8_sequence_length(std::string_view value, std::size_t index) noexcept {
    const auto current = static_cast<unsigned char>(value[index]);
    const auto has = [&](std::size_t offset) { return index + offset < value.size(); };
    const auto byte = [&](std::size_t offset) { return static_cast<unsigned char>(value[index + offset]); };

    if (current <= 0x7fU) return 1U;
    if (current >= 0xc2U && current <= 0xdfU) {
        return has(1U) && is_utf8_continuation(byte(1U)) ? 2U : 0U;
    }
    if (current == 0xe0U) {
        return has(2U) && byte(1U) >= 0xa0U && byte(1U) <= 0xbfU &&
                       is_utf8_continuation(byte(2U))
                   ? 3U
                   : 0U;
    }
    if ((current >= 0xe1U && current <= 0xecU) || (current >= 0xeeU && current <= 0xefU)) {
        return has(2U) && is_utf8_continuation(byte(1U)) && is_utf8_continuation(byte(2U)) ? 3U : 0U;
    }
    if (current == 0xedU) {
        return has(2U) && byte(1U) >= 0x80U && byte(1U) <= 0x9fU &&
                       is_utf8_continuation(byte(2U))
                   ? 3U
                   : 0U;
    }
    if (current == 0xf0U) {
        return has(3U) && byte(1U) >= 0x90U && byte(1U) <= 0xbfU &&
                       is_utf8_continuation(byte(2U)) && is_utf8_continuation(byte(3U))
                   ? 4U
                   : 0U;
    }
    if (current >= 0xf1U && current <= 0xf3U) {
        return has(3U) && is_utf8_continuation(byte(1U)) && is_utf8_continuation(byte(2U)) &&
                       is_utf8_continuation(byte(3U))
                   ? 4U
                   : 0U;
    }
    if (current == 0xf4U) {
        return has(3U) && byte(1U) >= 0x80U && byte(1U) <= 0x8fU &&
                       is_utf8_continuation(byte(2U)) && is_utf8_continuation(byte(3U))
                   ? 4U
                   : 0U;
    }
    return 0U;
}

inline std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (std::size_t index = 0; index < value.size();) {
        const auto character = static_cast<unsigned char>(value[index]);
        if (character >= 0x80U) {
            const std::size_t sequence_length = valid_utf8_sequence_length(value, index);
            if (sequence_length == 0U) {
                output << "\\ufffd";
                ++index;
            } else {
                output.write(value.data() + index, static_cast<std::streamsize>(sequence_length));
                index += sequence_length;
            }
            continue;
        }

        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    constexpr char digits[] = "0123456789abcdef";
                    output << "\\u00" << digits[(character >> 4U) & 0x0fU] << digits[character & 0x0fU];
                } else {
                    output << static_cast<char>(character);
                }
        }
        ++index;
    }
    return output.str();
}

inline bool parse_bool(std::string_view value, bool* output) {
    const auto normalized = to_lower(trim(value));
    if (normalized == "true" || normalized == "1" || normalized == "yes") {
        *output = true;
        return true;
    }
    if (normalized == "false" || normalized == "0" || normalized == "no") {
        *output = false;
        return true;
    }
    return false;
}

}  // namespace stargate::util
