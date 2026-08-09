#include "stargate/util/json.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <system_error>

namespace stargate::util {
namespace {

Error json_error(std::string message, std::size_t offset) {
    return make_error(ErrorCode::invalid_format, std::move(message), "JSON byte " + std::to_string(offset));
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
}

class JsonParser {
public:
    JsonParser(std::string_view input, JsonLimits limits) : input_(input), limits_(limits) {}

    Result<JsonValue> parse() {
        skip_space();
        auto value = parse_value(0);
        if (!value) return value.error();
        skip_space();
        if (position_ != input_.size()) return json_error("Unexpected data after JSON value", position_);
        return value;
    }

private:
    Result<JsonValue> parse_value(std::size_t depth) {
        if (depth > limits_.max_depth) return json_error("JSON nesting limit exceeded", position_);
        if (++nodes_ > limits_.max_nodes) return json_error("JSON node limit exceeded", position_);
        if (position_ >= input_.size()) return json_error("Unexpected end of JSON input", position_);
        switch (input_[position_]) {
            case 'n': {
                auto consumed = consume_literal("null");
                if (!consumed) return consumed.error();
                return JsonValue(nullptr);
            }
            case 't': {
                auto consumed = consume_literal("true");
                if (!consumed) return consumed.error();
                return JsonValue(true);
            }
            case 'f': {
                auto consumed = consume_literal("false");
                if (!consumed) return consumed.error();
                return JsonValue(false);
            }
            case '"': {
                auto string = parse_string();
                if (!string) return string.error();
                return JsonValue(string.take_value());
            }
            case '[': return parse_array(depth + 1U);
            case '{': return parse_object(depth + 1U);
            default:
                if (input_[position_] == '-' || (input_[position_] >= '0' && input_[position_] <= '9')) {
                    return parse_number();
                }
                return json_error("Unexpected JSON token", position_);
        }
    }

    Result<void> consume_literal(std::string_view literal) {
        if (input_.substr(position_, literal.size()) != literal) {
            return json_error("Invalid JSON literal", position_);
        }
        position_ += literal.size();
        return {};
    }

    Result<JsonValue> parse_number() {
        const std::size_t start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) return json_error("Truncated JSON number", position_);
        if (input_[position_] == '0') {
            ++position_;
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
        } else {
            return json_error("Invalid JSON number", position_);
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const std::size_t digits = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (digits == position_) return json_error("JSON fraction has no digits", position_);
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const std::size_t digits = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (digits == position_) return json_error("JSON exponent has no digits", position_);
        }
        return JsonValue(JsonNumber{std::string(input_.substr(start, position_ - start))});
    }

    Result<std::uint32_t> parse_hex4() {
        if (input_.size() - position_ < 4U) return json_error("Truncated JSON Unicode escape", position_);
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < 4U; ++index) {
            const char character = input_[position_++];
            value <<= 4U;
            if (character >= '0' && character <= '9') value |= static_cast<std::uint32_t>(character - '0');
            else if (character >= 'a' && character <= 'f') value |= static_cast<std::uint32_t>(character - 'a' + 10);
            else if (character >= 'A' && character <= 'F') value |= static_cast<std::uint32_t>(character - 'A' + 10);
            else return json_error("Invalid hexadecimal digit in JSON escape", position_ - 1U);
        }
        return value;
    }

    Result<std::string> parse_string() {
        if (input_[position_] != '"') return json_error("Expected JSON string", position_);
        ++position_;
        std::string output;
        while (position_ < input_.size()) {
            const unsigned char character = static_cast<unsigned char>(input_[position_++]);
            if (character == '"') return output;
            if (character < 0x20U) return json_error("Unescaped control byte in JSON string", position_ - 1U);
            if (character != '\\') {
                output.push_back(static_cast<char>(character));
            } else {
                if (position_ >= input_.size()) return json_error("Truncated JSON escape", position_);
                const char escape = input_[position_++];
                switch (escape) {
                    case '"': output.push_back('"'); break;
                    case '\\': output.push_back('\\'); break;
                    case '/': output.push_back('/'); break;
                    case 'b': output.push_back('\b'); break;
                    case 'f': output.push_back('\f'); break;
                    case 'n': output.push_back('\n'); break;
                    case 'r': output.push_back('\r'); break;
                    case 't': output.push_back('\t'); break;
                    case 'u': {
                        auto first = parse_hex4();
                        if (!first) return first.error();
                        std::uint32_t codepoint = first.value();
                        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                            if (input_.size() - position_ < 6U || input_[position_] != '\\' || input_[position_ + 1U] != 'u') {
                                return json_error("JSON high surrogate is missing its low surrogate", position_);
                            }
                            position_ += 2U;
                            auto second = parse_hex4();
                            if (!second) return second.error();
                            if (second.value() < 0xdc00U || second.value() > 0xdfffU) {
                                return json_error("Invalid JSON low surrogate", position_ - 4U);
                            }
                            codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + (second.value() - 0xdc00U);
                        } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                            return json_error("Unexpected JSON low surrogate", position_ - 4U);
                        }
                        append_utf8(output, codepoint);
                        break;
                    }
                    default: return json_error("Unknown JSON escape", position_ - 1U);
                }
            }
            if (output.size() > limits_.max_string_bytes) return json_error("JSON string limit exceeded", position_);
        }
        return json_error("Unterminated JSON string", position_);
    }

    Result<JsonValue> parse_array(std::size_t depth) {
        ++position_;
        skip_space();
        JsonValue::Array values;
        if (position_ < input_.size() && input_[position_] == ']') {
            ++position_;
            return JsonValue(std::move(values));
        }
        while (true) {
            if (values.size() >= limits_.max_array_elements) return json_error("JSON array limit exceeded", position_);
            auto value = parse_value(depth);
            if (!value) return value.error();
            values.push_back(value.take_value());
            skip_space();
            if (position_ >= input_.size()) return json_error("Unterminated JSON array", position_);
            if (input_[position_] == ']') {
                ++position_;
                return JsonValue(std::move(values));
            }
            if (input_[position_] != ',') return json_error("Expected comma in JSON array", position_);
            ++position_;
            skip_space();
        }
    }

    Result<JsonValue> parse_object(std::size_t depth) {
        ++position_;
        skip_space();
        JsonValue::Object values;
        if (position_ < input_.size() && input_[position_] == '}') {
            ++position_;
            return JsonValue(std::move(values));
        }
        while (true) {
            if (values.size() >= limits_.max_object_members) return json_error("JSON object limit exceeded", position_);
            if (position_ >= input_.size() || input_[position_] != '"') return json_error("Expected JSON object key", position_);
            auto key = parse_string();
            if (!key) return key.error();
            skip_space();
            if (position_ >= input_.size() || input_[position_] != ':') return json_error("Expected colon after JSON object key", position_);
            ++position_;
            skip_space();
            auto value = parse_value(depth);
            if (!value) return value.error();
            const auto [iterator, inserted] = values.emplace(key.take_value(), value.take_value());
            static_cast<void>(iterator);
            if (!inserted) return json_error("Duplicate JSON object key", position_);
            skip_space();
            if (position_ >= input_.size()) return json_error("Unterminated JSON object", position_);
            if (input_[position_] == '}') {
                ++position_;
                return JsonValue(std::move(values));
            }
            if (input_[position_] != ',') return json_error("Expected comma in JSON object", position_);
            ++position_;
            skip_space();
        }
    }

    void skip_space() noexcept {
        while (position_ < input_.size()) {
            const char character = input_[position_];
            if (character != ' ' && character != '\t' && character != '\n' && character != '\r') return;
            ++position_;
        }
    }

    std::string_view input_;
    JsonLimits limits_;
    std::size_t position_{0};
    std::size_t nodes_{0};
};

const JsonNumber* number_value(const JsonValue::Storage& storage) {
    return std::get_if<JsonNumber>(&storage);
}

}  // namespace

bool JsonValue::is_null() const noexcept { return std::holds_alternative<std::nullptr_t>(storage_); }
bool JsonValue::is_bool() const noexcept { return std::holds_alternative<bool>(storage_); }
bool JsonValue::is_number() const noexcept { return std::holds_alternative<JsonNumber>(storage_); }
bool JsonValue::is_string() const noexcept { return std::holds_alternative<std::string>(storage_); }
bool JsonValue::is_array() const noexcept { return std::holds_alternative<Array>(storage_); }
bool JsonValue::is_object() const noexcept { return std::holds_alternative<Object>(storage_); }

Result<bool> JsonValue::as_bool() const {
    const auto* value = std::get_if<bool>(&storage_);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "JSON value is not a boolean");
    return *value;
}

Result<std::string_view> JsonValue::as_string() const {
    const auto* value = std::get_if<std::string>(&storage_);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "JSON value is not a string");
    return std::string_view(*value);
}

Result<double> JsonValue::as_double() const {
    const auto* value = number_value(storage_);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "JSON value is not a number");
    double result = 0.0;
    const auto converted = std::from_chars(value->lexeme.data(), value->lexeme.data() + value->lexeme.size(), result);
    if (converted.ec != std::errc{} || converted.ptr != value->lexeme.data() + value->lexeme.size() || !std::isfinite(result)) {
        return make_error(ErrorCode::invalid_format, "JSON number is outside the supported finite range", value->lexeme);
    }
    return result;
}

Result<std::uint64_t> JsonValue::as_u64() const {
    const auto* value = number_value(storage_);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "JSON value is not an unsigned integer");
    std::uint64_t result = 0;
    const auto converted = std::from_chars(value->lexeme.data(), value->lexeme.data() + value->lexeme.size(), result);
    if (converted.ec != std::errc{} || converted.ptr != value->lexeme.data() + value->lexeme.size()) {
        return make_error(ErrorCode::invalid_format, "JSON value is not an unsigned integer", value->lexeme);
    }
    return result;
}

Result<std::int64_t> JsonValue::as_i64() const {
    const auto* value = number_value(storage_);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "JSON value is not an integer");
    std::int64_t result = 0;
    const auto converted = std::from_chars(value->lexeme.data(), value->lexeme.data() + value->lexeme.size(), result);
    if (converted.ec != std::errc{} || converted.ptr != value->lexeme.data() + value->lexeme.size()) {
        return make_error(ErrorCode::invalid_format, "JSON value is not an integer", value->lexeme);
    }
    return result;
}

Result<const JsonValue::Array*> JsonValue::as_array() const {
    const auto* value = std::get_if<Array>(&storage_);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "JSON value is not an array");
    return value;
}

Result<const JsonValue::Object*> JsonValue::as_object() const {
    const auto* value = std::get_if<Object>(&storage_);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "JSON value is not an object");
    return value;
}

const JsonValue* JsonValue::find(std::string_view key) const noexcept {
    const auto* object = std::get_if<Object>(&storage_);
    if (object == nullptr) return nullptr;
    const auto iterator = object->find(key);
    return iterator == object->end() ? nullptr : &iterator->second;
}

Result<JsonValue> parse_json(std::string_view input, JsonLimits limits) {
    if (static_cast<ByteCount>(input.size()) > limits.max_input_bytes) {
        return make_error(ErrorCode::resource_exhausted, "JSON input exceeds the configured byte limit");
    }
    try {
        return JsonParser(input, limits).parse();
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate memory while parsing JSON");
    }
}

Result<JsonValue> load_json_file(const std::filesystem::path& path, JsonLimits limits) {
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to stat JSON file", path.string());
    if (bytes > limits.max_input_bytes || bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
        return make_error(ErrorCode::resource_exhausted, "JSON file exceeds the configured byte limit", path.string());
    }
    try {
        std::string input(static_cast<std::size_t>(bytes), '\0');
        std::ifstream stream(path, std::ios::binary);
        if (!stream) return make_error(ErrorCode::io_error, "Unable to open JSON file", path.string());
        stream.read(input.data(), static_cast<std::streamsize>(input.size()));
        if (stream.gcount() != static_cast<std::streamsize>(input.size())) {
            return make_error(ErrorCode::io_error, "Short read while loading JSON file", path.string());
        }
        return parse_json(input, limits);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate memory for JSON file", path.string());
    }
}

}  // namespace stargate::util
