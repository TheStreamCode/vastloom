#include "vastloom/model/gguf.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <span>
#include <utility>

#include "vastloom/util/string.hpp"
#include "vastloom/util/json.hpp"

namespace vastloom::model {
namespace {

class BoundedReader {
public:
    BoundedReader(std::ifstream& input, ByteCount physical_bytes, ByteCount max_bytes)
        : input_(input), physical_bytes_(physical_bytes), max_bytes_(max_bytes) {}

    [[nodiscard]] ByteCount position() const noexcept { return position_; }

    [[nodiscard]] Result<std::uint8_t> read_u8() {
        std::array<std::byte, 1> bytes{};
        auto read = read_exact(bytes);
        if (!read) return read.error();
        return std::to_integer<std::uint8_t>(bytes[0]);
    }

    [[nodiscard]] Result<std::uint16_t> read_u16() {
        std::array<std::byte, 2> bytes{};
        auto read = read_exact(bytes);
        if (!read) return read.error();
        return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[0])) |
               static_cast<std::uint16_t>(static_cast<std::uint16_t>(
                   std::to_integer<std::uint8_t>(bytes[1])) << 8U);
    }

    [[nodiscard]] Result<std::uint32_t> read_u32() {
        std::array<std::byte, 4> bytes{};
        auto read = read_exact(bytes);
        if (!read) return read.error();
        std::uint32_t value = 0U;
        for (std::size_t index = 0U; index < bytes.size(); ++index) {
            value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[index]))
                     << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] Result<std::uint64_t> read_u64() {
        std::array<std::byte, 8> bytes{};
        auto read = read_exact(bytes);
        if (!read) return read.error();
        std::uint64_t value = 0U;
        for (std::size_t index = 0U; index < bytes.size(); ++index) {
            value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[index]))
                     << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] Result<std::string> read_string(ByteCount max_string_bytes, bool retain) {
        auto length = read_u64();
        if (!length) return length.error();
        if (length.value() > max_string_bytes ||
            length.value() > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
            return make_error(ErrorCode::resource_exhausted, "GGUF string exceeds configured bounds");
        }
        if (!retain) {
            auto skipped = skip(length.value());
            if (!skipped) return skipped.error();
            return std::string{};
        }
        std::string result(static_cast<std::size_t>(length.value()), '\0');
        auto read = read_exact(std::as_writable_bytes(std::span(result)));
        if (!read) return read.error();
        return result;
    }

    [[nodiscard]] Result<void> skip(ByteCount bytes) {
        auto bounded = ensure(bytes);
        if (!bounded) return bounded.error();
        if (bytes > static_cast<ByteCount>(std::numeric_limits<std::streamsize>::max())) {
            return make_error(ErrorCode::unsupported, "GGUF skip exceeds platform stream limits");
        }
        input_.ignore(static_cast<std::streamsize>(bytes));
        if (input_.gcount() != static_cast<std::streamsize>(bytes)) {
            return make_error(ErrorCode::io_error, "Short skip in bounded GGUF header");
        }
        position_ += bytes;
        return {};
    }

private:
    [[nodiscard]] Result<void> ensure(ByteCount bytes) const {
        if (bytes > std::numeric_limits<ByteCount>::max() - position_ ||
            position_ + bytes > physical_bytes_ || position_ + bytes > max_bytes_) {
            return make_error(ErrorCode::invalid_format, "GGUF header exceeds declared or configured bounds");
        }
        return {};
    }

    [[nodiscard]] Result<void> read_exact(std::span<std::byte> destination) {
        const ByteCount bytes = static_cast<ByteCount>(destination.size());
        auto bounded = ensure(bytes);
        if (!bounded) return bounded.error();
        if (destination.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            return make_error(ErrorCode::unsupported, "GGUF read exceeds platform stream limits");
        }
        input_.read(reinterpret_cast<char*>(destination.data()),
                    static_cast<std::streamsize>(destination.size()));
        if (input_.gcount() != static_cast<std::streamsize>(destination.size())) {
            return make_error(ErrorCode::io_error, "Short read in bounded GGUF header");
        }
        position_ += bytes;
        return {};
    }

    std::ifstream& input_;
    ByteCount physical_bytes_{0U};
    ByteCount max_bytes_{0U};
    ByteCount position_{0U};
};

struct ArrayBudget {
    std::uint64_t remaining_elements{0U};
};

bool valid_utf8(std::string_view value) noexcept {
    for (std::size_t index = 0U; index < value.size();) {
        const auto length = util::valid_utf8_sequence_length(value, index);
        if (length == 0U) return false;
        index += length;
    }
    return true;
}

bool valid_metadata_key(std::string_view key) noexcept {
    if (key.empty() || key.size() > 65'535U || key.front() == '.' || key.back() == '.') return false;
    bool previous_dot = false;
    for (const char raw_character : key) {
        const auto character = static_cast<unsigned char>(raw_character);
        const bool dot = character == static_cast<unsigned char>('.');
        if (dot && previous_dot) return false;
        if (!dot && !(character >= static_cast<unsigned char>('a') &&
                      character <= static_cast<unsigned char>('z')) &&
            !(character >= static_cast<unsigned char>('0') &&
              character <= static_cast<unsigned char>('9')) &&
            character != static_cast<unsigned char>('_')) {
            return false;
        }
        previous_dot = dot;
    }
    return true;
}

Result<GgufValueType> read_value_type(BoundedReader& reader) {
    auto raw = reader.read_u32();
    if (!raw) return raw.error();
    if (raw.value() > static_cast<std::uint32_t>(GgufValueType::float64)) {
        return make_error(ErrorCode::invalid_format, "GGUF metadata uses an unknown value type");
    }
    return static_cast<GgufValueType>(raw.value());
}

std::optional<ByteCount> fixed_value_bytes(GgufValueType type) noexcept {
    switch (type) {
        case GgufValueType::uint8:
        case GgufValueType::int8:
        case GgufValueType::boolean: return 1U;
        case GgufValueType::uint16:
        case GgufValueType::int16: return 2U;
        case GgufValueType::uint32:
        case GgufValueType::int32:
        case GgufValueType::float32: return 4U;
        case GgufValueType::uint64:
        case GgufValueType::int64:
        case GgufValueType::float64: return 8U;
        case GgufValueType::string:
        case GgufValueType::array: return std::nullopt;
    }
    return std::nullopt;
}

Result<void> parse_value(BoundedReader& reader,
                         GgufValueType type,
                         const GgufReaderLimits& limits,
                         ArrayBudget& budget,
                         ArrayBudget& retained_budget,
                         std::uint32_t depth,
                         GgufMetadataEntry* summary,
                         bool retain_integral_array = false) {
    switch (type) {
        case GgufValueType::uint8: {
            auto value = reader.read_u8();
            if (!value) return value.error();
            if (summary != nullptr) summary->unsigned_value = value.value();
            return {};
        }
        case GgufValueType::int8: {
            auto value = reader.read_u8();
            if (!value) return value.error();
            if (summary != nullptr) summary->signed_value = std::bit_cast<std::int8_t>(value.value());
            return {};
        }
        case GgufValueType::uint16: {
            auto value = reader.read_u16();
            if (!value) return value.error();
            if (summary != nullptr) summary->unsigned_value = value.value();
            return {};
        }
        case GgufValueType::int16: {
            auto value = reader.read_u16();
            if (!value) return value.error();
            if (summary != nullptr) summary->signed_value = std::bit_cast<std::int16_t>(value.value());
            return {};
        }
        case GgufValueType::uint32: {
            auto value = reader.read_u32();
            if (!value) return value.error();
            if (summary != nullptr) summary->unsigned_value = value.value();
            return {};
        }
        case GgufValueType::int32: {
            auto value = reader.read_u32();
            if (!value) return value.error();
            if (summary != nullptr) summary->signed_value = std::bit_cast<std::int32_t>(value.value());
            return {};
        }
        case GgufValueType::float32: {
            auto value = reader.read_u32();
            if (!value) return value.error();
            if (summary != nullptr) summary->floating_value = std::bit_cast<float>(value.value());
            return {};
        }
        case GgufValueType::boolean: {
            auto value = reader.read_u8();
            if (!value) return value.error();
            if (value.value() > 1U) {
                return make_error(ErrorCode::invalid_format, "GGUF boolean metadata is not zero or one");
            }
            if (summary != nullptr) summary->boolean_value = value.value() != 0U;
            return {};
        }
        case GgufValueType::string: {
            auto value = reader.read_string(limits.max_string_bytes, summary != nullptr);
            if (!value) return value.error();
            if (summary != nullptr) {
                if (!valid_utf8(value.value())) {
                    return make_error(ErrorCode::invalid_format, "GGUF scalar string is not valid UTF-8");
                }
                summary->string_value = value.take_value();
            }
            return {};
        }
        case GgufValueType::array: {
            if (depth >= limits.max_array_depth) {
                return make_error(ErrorCode::resource_exhausted, "GGUF metadata array nesting is too deep");
            }
            auto element_type = read_value_type(reader);
            auto length = reader.read_u64();
            if (!element_type) return element_type.error();
            if (!length) return length.error();
            if (length.value() > budget.remaining_elements) {
                return make_error(ErrorCode::resource_exhausted,
                                  "GGUF metadata arrays exceed the aggregate element limit");
            }
            budget.remaining_elements -= length.value();
            if (summary != nullptr) {
                summary->array_type = element_type.value();
                summary->array_elements = length.value();
            }
            if (retain_integral_array) {
                if (summary == nullptr || depth != 0U ||
                    length.value() > retained_budget.remaining_elements) {
                    return make_error(ErrorCode::resource_exhausted,
                                      "GGUF retained metadata arrays exceed the configured bound");
                }
                retained_budget.remaining_elements -= length.value();
                switch (element_type.value()) {
                    case GgufValueType::uint8:
                    case GgufValueType::uint16:
                    case GgufValueType::uint32:
                    case GgufValueType::uint64:
                        summary->unsigned_array_values.reserve(
                            static_cast<std::size_t>(length.value()));
                        for (std::uint64_t index = 0U; index < length.value(); ++index) {
                            GgufMetadataEntry item;
                            auto parsed = parse_value(reader,
                                                      element_type.value(),
                                                      limits,
                                                      budget,
                                                      retained_budget,
                                                      depth + 1U,
                                                      &item);
                            if (!parsed) return parsed.error();
                            if (!item.unsigned_value.has_value()) {
                                return make_error(ErrorCode::invalid_format,
                                                  "GGUF retained unsigned array value is invalid");
                            }
                            summary->unsigned_array_values.push_back(*item.unsigned_value);
                        }
                        return {};
                    case GgufValueType::int8:
                    case GgufValueType::int16:
                    case GgufValueType::int32:
                    case GgufValueType::int64:
                        summary->signed_array_values.reserve(
                            static_cast<std::size_t>(length.value()));
                        for (std::uint64_t index = 0U; index < length.value(); ++index) {
                            GgufMetadataEntry item;
                            auto parsed = parse_value(reader,
                                                      element_type.value(),
                                                      limits,
                                                      budget,
                                                      retained_budget,
                                                      depth + 1U,
                                                      &item);
                            if (!parsed) return parsed.error();
                            if (!item.signed_value.has_value()) {
                                return make_error(ErrorCode::invalid_format,
                                                  "GGUF retained signed array value is invalid");
                            }
                            summary->signed_array_values.push_back(*item.signed_value);
                        }
                        return {};
                    default:
                        return make_error(ErrorCode::invalid_format,
                                          "GGUF requested retained array is not integral");
                }
            }
            const auto width = fixed_value_bytes(element_type.value());
            if (width.has_value() && element_type.value() != GgufValueType::boolean) {
                if (length.value() != 0U && *width > std::numeric_limits<ByteCount>::max() / length.value()) {
                    return make_error(ErrorCode::invalid_format, "GGUF metadata array byte size overflows");
                }
                return reader.skip(length.value() * *width);
            }
            for (std::uint64_t index = 0U; index < length.value(); ++index) {
                auto parsed = parse_value(reader,
                                          element_type.value(),
                                          limits,
                                          budget,
                                          retained_budget,
                                          depth + 1U,
                                          nullptr);
                if (!parsed) return parsed.error();
            }
            return {};
        }
        case GgufValueType::uint64: {
            auto value = reader.read_u64();
            if (!value) return value.error();
            if (summary != nullptr) summary->unsigned_value = value.value();
            return {};
        }
        case GgufValueType::int64: {
            auto value = reader.read_u64();
            if (!value) return value.error();
            if (summary != nullptr) summary->signed_value = std::bit_cast<std::int64_t>(value.value());
            return {};
        }
        case GgufValueType::float64: {
            auto value = reader.read_u64();
            if (!value) return value.error();
            if (summary != nullptr) summary->floating_value = std::bit_cast<double>(value.value());
            return {};
        }
    }
    return make_error(ErrorCode::invalid_format, "GGUF metadata value type is invalid");
}

Result<std::uint32_t> resolve_alignment(const GgufFile& file, const GgufReaderLimits& limits) {
    std::uint64_t alignment = 32U;
    const auto* entry = file.find_metadata("general.alignment");
    if (entry != nullptr) {
        if (!entry->unsigned_value.has_value()) {
            return make_error(ErrorCode::invalid_format, "GGUF general.alignment is not unsigned");
        }
        alignment = *entry->unsigned_value;
    }
    if (alignment < 8U || alignment > limits.max_alignment || alignment % 8U != 0U ||
        alignment > std::numeric_limits<std::uint32_t>::max()) {
        return make_error(ErrorCode::invalid_format, "GGUF general.alignment is outside safe bounds");
    }
    return static_cast<std::uint32_t>(alignment);
}

Result<ByteCount> checked_element_count(const std::vector<std::uint64_t>& shape,
                                        std::string_view name) {
    ByteCount count = 1U;
    for (const auto dimension : shape) {
        if (dimension == 0U || count > std::numeric_limits<ByteCount>::max() / dimension) {
            return make_error(ErrorCode::invalid_format,
                              "GGUF tensor shape is empty or overflows",
                              std::string(name));
        }
        count *= dimension;
    }
    return count;
}

std::optional<std::uint64_t> unsigned_metadata(const GgufFile& file,
                                               std::string_view key) noexcept {
    const auto* entry = file.find_metadata(key);
    return entry == nullptr ? std::nullopt : entry->unsigned_value;
}

struct GgmlStorageLayout {
    ByteCount block_elements{0U};
    ByteCount block_bytes{0U};
};

std::optional<GgmlStorageLayout> ggml_storage_layout(std::uint32_t type) noexcept {
    switch (type) {
        case 0U: return GgmlStorageLayout{1U, 4U};       // F32
        case 1U: return GgmlStorageLayout{1U, 2U};       // F16
        case 2U: return GgmlStorageLayout{32U, 18U};     // Q4_0
        case 3U: return GgmlStorageLayout{32U, 20U};     // Q4_1
        case 6U: return GgmlStorageLayout{32U, 22U};     // Q5_0
        case 7U: return GgmlStorageLayout{32U, 24U};     // Q5_1
        case 8U: return GgmlStorageLayout{32U, 34U};     // Q8_0
        case 9U: return GgmlStorageLayout{32U, 40U};     // Q8_1
        case 10U: return GgmlStorageLayout{256U, 84U};   // Q2_K
        case 11U: return GgmlStorageLayout{256U, 110U};  // Q3_K
        case 12U: return GgmlStorageLayout{256U, 144U};  // Q4_K
        case 13U: return GgmlStorageLayout{256U, 176U};  // Q5_K
        case 14U: return GgmlStorageLayout{256U, 210U};  // Q6_K
        case 15U: return GgmlStorageLayout{256U, 292U};  // Q8_K
        case 16U: return GgmlStorageLayout{256U, 66U};   // IQ2_XXS
        case 17U: return GgmlStorageLayout{256U, 74U};   // IQ2_XS
        case 18U: return GgmlStorageLayout{256U, 98U};   // IQ3_XXS
        case 19U: return GgmlStorageLayout{256U, 50U};   // IQ1_S
        case 20U: return GgmlStorageLayout{32U, 18U};    // IQ4_NL
        case 21U: return GgmlStorageLayout{256U, 110U};  // IQ3_S
        case 22U: return GgmlStorageLayout{256U, 82U};   // IQ2_S
        case 23U: return GgmlStorageLayout{256U, 136U};  // IQ4_XS
        case 24U: return GgmlStorageLayout{1U, 1U};      // I8
        case 25U: return GgmlStorageLayout{1U, 2U};      // I16
        case 26U: return GgmlStorageLayout{1U, 4U};      // I32
        case 27U: return GgmlStorageLayout{1U, 8U};      // I64
        case 28U: return GgmlStorageLayout{1U, 8U};      // F64
        case 29U: return GgmlStorageLayout{256U, 56U};   // IQ1_M
        case 30U: return GgmlStorageLayout{1U, 2U};      // BF16
        case 34U: return GgmlStorageLayout{256U, 54U};   // TQ1_0
        case 35U: return GgmlStorageLayout{256U, 66U};   // TQ2_0
        case 39U: return GgmlStorageLayout{32U, 17U};    // MXFP4
        default: return std::nullopt;
    }
}

}  // namespace

Result<GgufFile> GgufFile::open(const std::filesystem::path& path,
                                const GgufReaderLimits& limits) {
    return open_with_declared_size(path, 0U, false, limits);
}

Result<GgufFile> GgufFile::open_header_prefix(const std::filesystem::path& path,
                                              ByteCount declared_file_bytes,
                                              const GgufReaderLimits& limits) {
    return open_with_declared_size(path, declared_file_bytes, true, limits);
}

Result<GgufFile> GgufFile::open_with_declared_size(const std::filesystem::path& path,
                                                   ByteCount declared_file_bytes,
                                                   bool header_snapshot,
                                                   const GgufReaderLimits& limits) {
    try {
        if (path.empty() || limits.max_header_bytes < 24U || limits.max_metadata_entries == 0U ||
            limits.max_tensors == 0U || limits.max_array_elements == 0U ||
            limits.max_string_bytes == 0U || limits.max_string_bytes > limits.max_header_bytes ||
            limits.max_alignment < 8U || limits.max_dimensions == 0U || limits.max_dimensions > 64U ||
            limits.max_array_depth == 0U || limits.max_array_depth > 32U ||
            limits.max_retained_array_elements == 0U ||
            limits.retained_integral_array_keys.size() > 64U) {
            return make_error(ErrorCode::invalid_argument, "Invalid GGUF reader limits", path.string());
        }
        for (std::size_t index = 0U; index < limits.retained_integral_array_keys.size(); ++index) {
            const auto& key = limits.retained_integral_array_keys[index];
            if (!valid_metadata_key(key) ||
                std::find(limits.retained_integral_array_keys.begin(),
                          limits.retained_integral_array_keys.begin() +
                              static_cast<std::ptrdiff_t>(index),
                          key) != limits.retained_integral_array_keys.begin() +
                                      static_cast<std::ptrdiff_t>(index)) {
                return make_error(ErrorCode::invalid_argument,
                                  "GGUF retained metadata key is invalid or duplicated",
                                  key);
            }
        }
        std::error_code error;
        if (std::filesystem::is_symlink(path, error)) {
            return make_error(ErrorCode::unsafe_path, "Refusing a symlinked GGUF file", path.string());
        }
        error.clear();
        const ByteCount physical_bytes = std::filesystem::file_size(path, error);
        if (error) return make_error(ErrorCode::io_error, "Unable to stat GGUF file", path.string());
        if (physical_bytes < 24U) {
            return make_error(ErrorCode::invalid_format, "GGUF file is too small", path.string());
        }
        const ByteCount file_bytes = header_snapshot ? declared_file_bytes : physical_bytes;
        if (file_bytes < physical_bytes) {
            return make_error(ErrorCode::invalid_argument,
                              "GGUF declared size is below the header snapshot size",
                              path.string());
        }

        std::ifstream input(path, std::ios::binary);
        if (!input) return make_error(ErrorCode::io_error, "Unable to open GGUF file", path.string());
        BoundedReader reader(input, physical_bytes, limits.max_header_bytes);
        auto magic = reader.read_u32();
        auto version = reader.read_u32();
        auto tensor_count = reader.read_u64();
        auto metadata_count = reader.read_u64();
        if (!magic) return magic.error();
        if (!version) return version.error();
        if (!tensor_count) return tensor_count.error();
        if (!metadata_count) return metadata_count.error();
        if (magic.value() != 0x46554747U) {
            return make_error(ErrorCode::invalid_format, "GGUF magic is invalid", path.string());
        }
        if (version.value() != 3U) {
            return make_error(ErrorCode::unsupported, "Only little-endian GGUF v3 is supported", path.string());
        }
        if (tensor_count.value() > limits.max_tensors ||
            tensor_count.value() > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            metadata_count.value() > limits.max_metadata_entries ||
            metadata_count.value() > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return make_error(ErrorCode::resource_exhausted, "GGUF header counts exceed configured limits", path.string());
        }

        GgufFile result;
        result.file_bytes_ = file_bytes;
        result.version_ = version.value();
        result.header_only_ = header_snapshot;
        result.metadata_.reserve(static_cast<std::size_t>(metadata_count.value()));
        result.tensors_.reserve(static_cast<std::size_t>(tensor_count.value()));
        ArrayBudget array_budget{limits.max_array_elements};
        ArrayBudget retained_array_budget{limits.max_retained_array_elements};

        for (std::uint64_t index = 0U; index < metadata_count.value(); ++index) {
            auto key = reader.read_string(65'535U, true);
            if (!key) return key.error();
            if (!valid_metadata_key(key.value()) || result.metadata_by_key_.contains(key.value())) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF metadata key is invalid or duplicated",
                                  key.value());
            }
            auto type = read_value_type(reader);
            if (!type) return type.error();
            GgufMetadataEntry entry;
            entry.key = key.take_value();
            entry.type = type.value();
            const bool retain_integral_array =
                std::find(limits.retained_integral_array_keys.begin(),
                          limits.retained_integral_array_keys.end(),
                          entry.key) != limits.retained_integral_array_keys.end();
            if (retain_integral_array && entry.type != GgufValueType::array) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF requested retained metadata value is not an array",
                                  entry.key);
            }
            const ByteCount value_start = reader.position();
            auto parsed = parse_value(reader,
                                      entry.type,
                                      limits,
                                      array_budget,
                                      retained_array_budget,
                                      0U,
                                      &entry,
                                      retain_integral_array);
            if (!parsed) return parsed.error();
            entry.encoded_value_bytes = reader.position() - value_start;
            result.metadata_by_key_.emplace(entry.key, result.metadata_.size());
            result.metadata_.push_back(std::move(entry));
        }

        auto alignment = resolve_alignment(result, limits);
        if (!alignment) return alignment.error();
        result.alignment_ = alignment.value();

        for (std::uint64_t index = 0U; index < tensor_count.value(); ++index) {
            auto name = reader.read_string(64U, true);
            auto dimensions = reader.read_u32();
            if (!name) return name.error();
            if (!dimensions) return dimensions.error();
            if (name.value().empty() || !valid_utf8(name.value()) ||
                result.tensors_by_name_.contains(name.value()) || dimensions.value() == 0U ||
                dimensions.value() > limits.max_dimensions) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF tensor name or dimension count is invalid",
                                  name.value());
            }
            GgufTensorDescriptor tensor;
            tensor.name = name.take_value();
            tensor.shape.reserve(dimensions.value());
            for (std::uint32_t dimension = 0U; dimension < dimensions.value(); ++dimension) {
                auto value = reader.read_u64();
                if (!value) return value.error();
                tensor.shape.push_back(value.value());
            }
            auto element_count = checked_element_count(tensor.shape, tensor.name);
            if (!element_count) return element_count.error();
            tensor.element_count = element_count.value();
            auto ggml_type = reader.read_u32();
            auto relative_offset = reader.read_u64();
            if (!ggml_type) return ggml_type.error();
            if (!relative_offset) return relative_offset.error();
            if (ggml_type.value() > 1024U ||
                relative_offset.value() % result.alignment_ != 0U) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF tensor type or relative offset is invalid",
                                  tensor.name);
            }
            tensor.ggml_type = ggml_type.value();
            tensor.relative_offset = relative_offset.value();
            result.tensors_by_name_.emplace(tensor.name, result.tensors_.size());
            result.tensors_.push_back(std::move(tensor));
        }

        const ByteCount remainder = reader.position() % result.alignment_;
        const ByteCount padding = remainder == 0U ? 0U : result.alignment_ - remainder;
        for (ByteCount index = 0U; index < padding; ++index) {
            auto value = reader.read_u8();
            if (!value) return value.error();
            if (value.value() != 0U) {
                return make_error(ErrorCode::invalid_format, "GGUF header padding is nonzero", path.string());
            }
        }
        result.data_offset_ = reader.position();
        if (result.data_offset_ > file_bytes || (header_snapshot && physical_bytes < result.data_offset_) ||
            (!header_snapshot && physical_bytes < result.data_offset_)) {
            return make_error(ErrorCode::invalid_format,
                              "GGUF tensor-data boundary is inconsistent with the file",
                              path.string());
        }

        const ByteCount payload_bytes = file_bytes - result.data_offset_;
        std::vector<std::pair<ByteCount, std::size_t>> offsets;
        offsets.reserve(result.tensors_.size());
        for (std::size_t index = 0U; index < result.tensors_.size(); ++index) {
            const auto relative = result.tensors_[index].relative_offset;
            if (relative >= payload_bytes || relative > std::numeric_limits<ByteCount>::max() - result.data_offset_) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF tensor offset exceeds the declared payload",
                                  result.tensors_[index].name);
            }
            result.tensors_[index].absolute_offset = result.data_offset_ + relative;
            offsets.emplace_back(relative, index);
        }
        std::sort(offsets.begin(), offsets.end());
        for (std::size_t index = 0U; index < offsets.size(); ++index) {
            if (index != 0U && offsets[index].first == offsets[index - 1U].first) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF tensor offsets are duplicated",
                                  result.tensors_[offsets[index].second].name);
            }
            const ByteCount end = index + 1U < offsets.size() ? offsets[index + 1U].first : payload_bytes;
            if (end <= offsets[index].first) {
                return make_error(ErrorCode::invalid_format, "GGUF tensor storage span is empty", path.string());
            }
            auto& tensor = result.tensors_[offsets[index].second];
            tensor.available_span_bytes = end - offsets[index].first;
            const auto layout = ggml_storage_layout(tensor.ggml_type);
            if (layout.has_value()) {
                if (tensor.shape.front() % layout->block_elements != 0U ||
                    tensor.element_count % layout->block_elements != 0U ||
                    tensor.element_count / layout->block_elements >
                        std::numeric_limits<ByteCount>::max() / layout->block_bytes) {
                    return make_error(ErrorCode::invalid_format,
                                      "GGUF tensor shape is incompatible with its GGML block type",
                                      tensor.name);
                }
                const ByteCount encoded = tensor.element_count / layout->block_elements *
                                          layout->block_bytes;
                if (encoded > tensor.available_span_bytes ||
                    tensor.available_span_bytes - encoded >= result.alignment_) {
                    return make_error(ErrorCode::invalid_format,
                                      "GGUF tensor byte span disagrees with its GGML block type",
                                      tensor.name);
                }
                tensor.encoded_bytes = encoded;
            }
        }

        auto architecture = result.architecture();
        if (architecture) {
            if (architecture.value().empty() || architecture.value().size() > 64U ||
                !std::all_of(architecture.value().begin(), architecture.value().end(), [](char raw_character) {
                    const auto character = static_cast<unsigned char>(raw_character);
                    return (character >= static_cast<unsigned char>('a') &&
                            character <= static_cast<unsigned char>('z')) ||
                           (character >= static_cast<unsigned char>('0') &&
                            character <= static_cast<unsigned char>('9'));
                })) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF general.architecture is invalid",
                                  path.string());
            }
        } else {
            const auto* split_no = result.find_metadata("split.no");
            const auto* split_count = result.find_metadata("split.count");
            if (split_no == nullptr || split_count == nullptr ||
                !split_no->unsigned_value.has_value() || !split_count->unsigned_value.has_value() ||
                *split_no->unsigned_value == 0U ||
                *split_no->unsigned_value >= *split_count->unsigned_value) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF general.architecture is absent outside a continuation split",
                                  path.string());
            }
        }

        result.path_ = std::filesystem::absolute(path, error).lexically_normal();
        if (error) result.path_ = path;
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate within bounded GGUF inspection",
                          path.string());
    }
}

const GgufMetadataEntry* GgufFile::find_metadata(std::string_view key) const noexcept {
    const auto iterator = metadata_by_key_.find(key);
    return iterator == metadata_by_key_.end() ? nullptr : &metadata_[iterator->second];
}

const GgufTensorDescriptor* GgufFile::find_tensor(std::string_view name) const noexcept {
    const auto iterator = tensors_by_name_.find(name);
    return iterator == tensors_by_name_.end() ? nullptr : &tensors_[iterator->second];
}

Result<GgufSplitCatalog> finalize_gguf_split_catalog(std::vector<GgufFile> shards,
                                                     std::size_t max_shards,
                                                     std::size_t max_tensors) {
    try {
        if (shards.empty() || max_shards == 0U || max_tensors == 0U ||
            shards.size() > max_shards) {
            return make_error(ErrorCode::invalid_argument,
                              "GGUF split catalog has no shards or exceeds configured limits");
        }
        std::vector<std::pair<std::size_t, GgufFile>> ordered;
        ordered.reserve(shards.size());
        std::optional<std::uint64_t> declared_split_count;
        for (auto& shard : shards) {
            const auto split_no = unsigned_metadata(shard, "split.no");
            const auto split_count = unsigned_metadata(shard, "split.count");
            if (shards.size() == 1U && !split_no.has_value() && !split_count.has_value()) {
                ordered.emplace_back(0U, std::move(shard));
                continue;
            }
            if (!split_no.has_value() || !split_count.has_value() ||
                *split_count != static_cast<std::uint64_t>(shards.size()) ||
                *split_no >= *split_count ||
                *split_no > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF shard has inconsistent split metadata",
                                  shard.path().string());
            }
            if (declared_split_count.has_value() && *declared_split_count != *split_count) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF shards disagree on split count",
                                  shard.path().string());
            }
            declared_split_count = *split_count;
            ordered.emplace_back(static_cast<std::size_t>(*split_no), std::move(shard));
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        for (std::size_t index = 0U; index < ordered.size(); ++index) {
            if (ordered[index].first != index) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF split sequence has a duplicate or missing shard number");
            }
        }

        auto primary_architecture = ordered.front().second.architecture();
        if (!primary_architecture) {
            return make_error(ErrorCode::invalid_format,
                              "GGUF split zero does not define general.architecture",
                              ordered.front().second.path().string());
        }
        GgufSplitCatalog result;
        result.architecture = std::string(primary_architecture.value());
        result.shards.reserve(ordered.size());
        for (auto& [split_no, shard] : ordered) {
            static_cast<void>(split_no);
            auto architecture = shard.architecture();
            if (architecture && architecture.value() != result.architecture) {
                return make_error(ErrorCode::invalid_format,
                                  "GGUF shards disagree on architecture",
                                  shard.path().string());
            }
            if (result.tensors.size() > max_tensors ||
                shard.tensors().size() > max_tensors - result.tensors.size()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "GGUF split tensor count exceeds configured limits");
            }
            if (shard.file_bytes() > std::numeric_limits<ByteCount>::max() - result.total_file_bytes ||
                shard.data_offset() > std::numeric_limits<ByteCount>::max() - result.total_header_bytes ||
                shard.file_bytes() - shard.data_offset() >
                    std::numeric_limits<ByteCount>::max() - result.total_payload_bytes) {
                return make_error(ErrorCode::invalid_format, "GGUF split byte totals overflow");
            }
            result.total_file_bytes += shard.file_bytes();
            result.total_header_bytes += shard.data_offset();
            result.total_payload_bytes += shard.file_bytes() - shard.data_offset();
            const std::size_t shard_index = result.shards.size();
            for (std::size_t tensor_index = 0U; tensor_index < shard.tensors().size(); ++tensor_index) {
                const auto& tensor = shard.tensors()[tensor_index];
                if (result.tensors.contains(tensor.name)) {
                    return make_error(ErrorCode::integrity_error,
                                      "GGUF tensor name is duplicated across split shards",
                                      tensor.name);
                }
                if (tensor.encoded_bytes.has_value()) {
                    if (*tensor.encoded_bytes >
                        std::numeric_limits<ByteCount>::max() - result.known_tensor_bytes) {
                        return make_error(ErrorCode::invalid_format,
                                          "GGUF encoded tensor byte total overflows",
                                          tensor.name);
                    }
                    result.known_tensor_bytes += *tensor.encoded_bytes;
                } else {
                    ++result.unknown_tensor_sizes;
                }
                result.tensors.emplace(tensor.name, GgufTensorLocation{shard_index, tensor_index});
            }
            result.shards.push_back(std::move(shard));
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded GGUF split catalog");
    }
}

Result<std::string_view> GgufFile::architecture() const {
    const auto* entry = find_metadata("general.architecture");
    if (entry == nullptr || entry->type != GgufValueType::string || !entry->string_value.has_value()) {
        return make_error(ErrorCode::invalid_format, "GGUF general.architecture is missing or not a string");
    }
    return std::string_view(*entry->string_value);
}

std::string_view to_string(GgufValueType type) noexcept {
    switch (type) {
        case GgufValueType::uint8: return "UINT8";
        case GgufValueType::int8: return "INT8";
        case GgufValueType::uint16: return "UINT16";
        case GgufValueType::int16: return "INT16";
        case GgufValueType::uint32: return "UINT32";
        case GgufValueType::int32: return "INT32";
        case GgufValueType::float32: return "FLOAT32";
        case GgufValueType::boolean: return "BOOL";
        case GgufValueType::string: return "STRING";
        case GgufValueType::array: return "ARRAY";
        case GgufValueType::uint64: return "UINT64";
        case GgufValueType::int64: return "INT64";
        case GgufValueType::float64: return "FLOAT64";
    }
    return "UNKNOWN";
}

std::string to_json(const GgufFile& file, std::size_t tensor_limit) {
    const auto architecture = file.architecture();
    std::map<std::uint32_t, std::size_t> types;
    for (const auto& tensor : file.tensors()) ++types[tensor.ggml_type];
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"GGUFv3\",\"path\":\""
           << util::json_escape(file.path().string()) << "\",\"header_only\":"
           << (file.header_only() ? "true" : "false") << ",\"version\":" << file.version()
           << ",\"architecture\":";
    if (architecture) output << '\"' << util::json_escape(architecture.value()) << '\"';
    else output << "null";
    output << ",\"alignment\":" << file.alignment()
           << ",\"file_bytes\":" << file.file_bytes()
           << ",\"header_bytes\":" << file.data_offset()
           << ",\"payload_bytes\":" << (file.file_bytes() - file.data_offset())
           << ",\"metadata_count\":" << file.metadata().size()
           << ",\"tensor_count\":" << file.tensors().size()
           << ",\"quantization_version\":";
    if (const auto value = unsigned_metadata(file, "general.quantization_version")) output << *value;
    else output << "null";
    output << ",\"split\":{\"no\":";
    if (const auto value = unsigned_metadata(file, "split.no")) output << *value;
    else output << "null";
    output << ",\"count\":";
    if (const auto value = unsigned_metadata(file, "split.count")) output << *value;
    else output << "null";
    output << ",\"tensor_count\":";
    if (const auto value = unsigned_metadata(file, "split.tensors.count")) output << *value;
    else output << "null";
    output << "},\"ggml_types\":{";
    bool first = true;
    for (const auto& [type, count] : types) {
        if (!first) output << ',';
        first = false;
        output << '\"' << type << "\":" << count;
    }
    output << '}';
    if (tensor_limit != 0U) {
        output << ",\"tensors\":[";
        const auto count = std::min(tensor_limit, file.tensors().size());
        for (std::size_t index = 0U; index < count; ++index) {
            if (index != 0U) output << ',';
            const auto& tensor = file.tensors()[index];
            output << "{\"name\":\"" << util::json_escape(tensor.name) << "\",\"shape\":[";
            for (std::size_t dimension = 0U; dimension < tensor.shape.size(); ++dimension) {
                if (dimension != 0U) output << ',';
                output << tensor.shape[dimension];
            }
            output << "],\"ggml_type\":" << tensor.ggml_type
                   << ",\"elements\":" << tensor.element_count
                   << ",\"relative_offset\":" << tensor.relative_offset
                   << ",\"absolute_offset\":" << tensor.absolute_offset
                   << ",\"encoded_bytes\":";
            if (tensor.encoded_bytes.has_value()) output << *tensor.encoded_bytes;
            else output << "null";
            output
                   << ",\"available_span_bytes\":" << tensor.available_span_bytes << '}';
        }
        output << ']';
    }
    output << '}';
    return output.str();
}

std::string to_text(const GgufFile& file, std::size_t tensor_limit) {
    const auto architecture = file.architecture();
    std::map<std::uint32_t, std::size_t> types;
    for (const auto& tensor : file.tensors()) ++types[tensor.ggml_type];
    std::ostringstream output;
    output << "GGUF:          " << file.path().string() << '\n'
           << "Version:       " << file.version() << '\n'
           << "Architecture:  " << (architecture ? architecture.value() : "unknown") << '\n'
           << "Header mode:   " << (file.header_only() ? "snapshot" : "complete file") << '\n'
           << "Alignment:     " << file.alignment() << '\n'
           << "Metadata:      " << file.metadata().size() << '\n'
           << "Tensors:       " << file.tensors().size() << '\n'
           << "File bytes:    " << format_bytes(file.file_bytes()) << '\n'
           << "Header bytes:  " << format_bytes(file.data_offset()) << '\n'
           << "Payload span:  " << format_bytes(file.file_bytes() - file.data_offset()) << '\n'
           << "GGML types:";
    for (const auto& [type, count] : types) output << ' ' << type << '=' << count;
    const auto count = std::min(tensor_limit, file.tensors().size());
    for (std::size_t index = 0U; index < count; ++index) {
        const auto& tensor = file.tensors()[index];
        output << "\n  " << tensor.name << " type=" << tensor.ggml_type << " shape=";
        for (std::size_t dimension = 0U; dimension < tensor.shape.size(); ++dimension) {
            if (dimension != 0U) output << 'x';
            output << tensor.shape[dimension];
        }
        output << " offset=" << tensor.absolute_offset << " bytes=";
        if (tensor.encoded_bytes.has_value()) output << *tensor.encoded_bytes;
        else output << "unknown";
        output
               << " span<=" << tensor.available_span_bytes;
    }
    return output.str();
}

}  // namespace vastloom::model
