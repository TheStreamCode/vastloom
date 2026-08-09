#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "stargate/core/bytes.hpp"
#include "stargate/core/status.hpp"

namespace stargate::util {

struct JsonNumber {
    std::string lexeme;
};

class JsonValue {
public:
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue, std::less<>>;
    using Storage = std::variant<std::nullptr_t, bool, JsonNumber, std::string, Array, Object>;

    JsonValue() = default;
    explicit JsonValue(std::nullptr_t) : storage_(nullptr) {}
    explicit JsonValue(bool value) : storage_(value) {}
    explicit JsonValue(JsonNumber value) : storage_(std::move(value)) {}
    explicit JsonValue(std::string value) : storage_(std::move(value)) {}
    explicit JsonValue(Array value) : storage_(std::move(value)) {}
    explicit JsonValue(Object value) : storage_(std::move(value)) {}

    [[nodiscard]] bool is_null() const noexcept;
    [[nodiscard]] bool is_bool() const noexcept;
    [[nodiscard]] bool is_number() const noexcept;
    [[nodiscard]] bool is_string() const noexcept;
    [[nodiscard]] bool is_array() const noexcept;
    [[nodiscard]] bool is_object() const noexcept;

    [[nodiscard]] Result<bool> as_bool() const;
    [[nodiscard]] Result<std::string_view> as_string() const;
    [[nodiscard]] Result<double> as_double() const;
    [[nodiscard]] Result<std::uint64_t> as_u64() const;
    [[nodiscard]] Result<std::int64_t> as_i64() const;
    [[nodiscard]] Result<const Array*> as_array() const;
    [[nodiscard]] Result<const Object*> as_object() const;
    [[nodiscard]] const JsonValue* find(std::string_view key) const noexcept;

private:
    Storage storage_{nullptr};
};

struct JsonLimits {
    ByteCount max_input_bytes{64U * MiB};
    std::size_t max_depth{64};
    std::size_t max_nodes{1'000'000};
    std::size_t max_string_bytes{16U * MiB};
    std::size_t max_array_elements{1'000'000};
    std::size_t max_object_members{1'000'000};
};

[[nodiscard]] Result<JsonValue> parse_json(std::string_view input, JsonLimits limits = {});
[[nodiscard]] Result<JsonValue> load_json_file(const std::filesystem::path& path, JsonLimits limits = {});

}  // namespace stargate::util
