#include "stargate/model/safetensors.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>

#include "stargate/util/json.hpp"
#include "stargate/util/string.hpp"

namespace stargate::model {
namespace {

std::uint64_t read_u64_le(const std::array<std::byte, 8>& bytes) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[index])) << (index * 8U);
    }
    return value;
}

Result<TensorDType> parse_dtype(std::string_view value) {
    if (value == "BOOL") return TensorDType::boolean;
    if (value == "U8") return TensorDType::u8;
    if (value == "I8") return TensorDType::i8;
    if (value == "U16") return TensorDType::u16;
    if (value == "I16") return TensorDType::i16;
    if (value == "F16") return TensorDType::f16;
    if (value == "BF16") return TensorDType::bf16;
    if (value == "U32") return TensorDType::u32;
    if (value == "I32") return TensorDType::i32;
    if (value == "F32") return TensorDType::f32;
    if (value == "F64") return TensorDType::f64;
    if (value == "U64") return TensorDType::u64;
    if (value == "I64") return TensorDType::i64;
    return make_error(ErrorCode::unsupported, "Unsupported Safetensors dtype", std::string(value));
}

ByteCount dtype_bytes(TensorDType dtype) noexcept {
    switch (dtype) {
        case TensorDType::boolean:
        case TensorDType::u8:
        case TensorDType::i8: return 1;
        case TensorDType::u16:
        case TensorDType::i16:
        case TensorDType::f16:
        case TensorDType::bf16: return 2;
        case TensorDType::u32:
        case TensorDType::i32:
        case TensorDType::f32: return 4;
        case TensorDType::f64:
        case TensorDType::u64:
        case TensorDType::i64: return 8;
    }
    return 0;
}

Result<ByteCount> checked_multiply(ByteCount left, ByteCount right, std::string_view tensor_name) {
    if (left != 0 && right > std::numeric_limits<ByteCount>::max() / left) {
        return make_error(ErrorCode::invalid_format, "Safetensors shape byte count overflows", std::string(tensor_name));
    }
    return left * right;
}

Result<const util::JsonValue*> required_field(const util::JsonValue& value,
                                              std::string_view field,
                                              std::string_view tensor_name) {
    const auto* result = value.find(field);
    if (result == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Safetensors tensor is missing required field " + std::string(field),
                          std::string(tensor_name));
    }
    return result;
}

bool has_safetensors_extension(const std::filesystem::path& path) {
    return util::to_lower(path.extension().string()) == ".safetensors";
}

}  // namespace

Result<SafetensorsFile> SafetensorsFile::open(const std::filesystem::path& path,
                                               ByteCount max_header_bytes,
                                               std::size_t max_tensors) {
    if (path.empty() || max_header_bytes < 2U || max_tensors == 0U ||
        max_tensors > (std::numeric_limits<std::size_t>::max() - 64U) / 8U) {
        return make_error(ErrorCode::invalid_argument, "Invalid Safetensors reader limits", path.string());
    }
    std::error_code error;
    if (std::filesystem::is_symlink(path, error)) {
        return make_error(ErrorCode::unsafe_path, "Refusing a symlinked Safetensors shard", path.string());
    }
    error.clear();
    const ByteCount file_bytes = std::filesystem::file_size(path, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to stat Safetensors shard", path.string());
    if (file_bytes < 10U) return make_error(ErrorCode::invalid_format, "Safetensors shard is too small", path.string());

    std::ifstream input(path, std::ios::binary);
    if (!input) return make_error(ErrorCode::io_error, "Unable to open Safetensors shard", path.string());
    std::array<std::byte, 8> prefix{};
    input.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    if (input.gcount() != static_cast<std::streamsize>(prefix.size())) {
        return make_error(ErrorCode::io_error, "Unable to read Safetensors header length", path.string());
    }
    const ByteCount header_bytes = read_u64_le(prefix);
    if (header_bytes < 2U || header_bytes > max_header_bytes || header_bytes > file_bytes - 8U) {
        return make_error(ErrorCode::invalid_format, "Safetensors header length is outside safe bounds", path.string());
    }
    if (header_bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
        return make_error(ErrorCode::unsupported, "Safetensors header does not fit the platform address space", path.string());
    }

    std::string header;
    try {
        header.resize(static_cast<std::size_t>(header_bytes));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate the bounded Safetensors header", path.string());
    }
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size())) {
        return make_error(ErrorCode::io_error, "Short read in Safetensors header", path.string());
    }

    util::JsonLimits limits;
    limits.max_input_bytes = max_header_bytes;
    limits.max_depth = 16;
    limits.max_nodes = max_tensors * 8U + 64U;
    limits.max_object_members = max_tensors + 16U;
    limits.max_array_elements = 64;
    auto parsed = util::parse_json(header, limits);
    if (!parsed) return parsed.error();
    auto root = parsed.value().as_object();
    if (!root) return make_error(ErrorCode::invalid_format, "Safetensors header root is not an object", path.string());

    SafetensorsFile result;
    result.path_ = std::filesystem::absolute(path, error).lexically_normal();
    if (error) result.path_ = path;
    result.file_bytes_ = file_bytes;
    result.header_bytes_ = header_bytes;
    result.tensors_.reserve(std::min(root.value()->size(), max_tensors));
    const ByteCount data_start = 8U + header_bytes;
    const ByteCount payload_bytes = file_bytes - data_start;

    for (const auto& [name, definition] : *root.value()) {
        if (name == "__metadata__") continue;
        if (name.empty() || result.tensors_.size() >= max_tensors) {
            return make_error(ErrorCode::invalid_format, "Safetensors tensor name/count is invalid", path.string());
        }
        auto object = definition.as_object();
        if (!object) return make_error(ErrorCode::invalid_format, "Safetensors tensor definition is not an object", name);
        auto dtype_field = required_field(definition, "dtype", name);
        auto shape_field = required_field(definition, "shape", name);
        auto offsets_field = required_field(definition, "data_offsets", name);
        if (!dtype_field) return dtype_field.error();
        if (!shape_field) return shape_field.error();
        if (!offsets_field) return offsets_field.error();
        auto dtype_text = dtype_field.value()->as_string();
        if (!dtype_text) return make_error(ErrorCode::invalid_format, "Safetensors dtype is not a string", name);
        auto dtype = parse_dtype(dtype_text.value());
        if (!dtype) return dtype.error();
        auto shape_array = shape_field.value()->as_array();
        auto offsets_array = offsets_field.value()->as_array();
        if (!shape_array || !offsets_array || offsets_array.value()->size() != 2U || shape_array.value()->size() > 16U) {
            return make_error(ErrorCode::invalid_format, "Safetensors shape or data_offsets is invalid", name);
        }

        TensorDescriptor tensor;
        tensor.name = name;
        tensor.dtype = dtype.value();
        ByteCount element_count = 1U;
        tensor.shape.reserve(shape_array.value()->size());
        for (const auto& dimension_value : *shape_array.value()) {
            auto dimension = dimension_value.as_u64();
            if (!dimension) return make_error(ErrorCode::invalid_format, "Safetensors shape dimension is invalid", name);
            tensor.shape.push_back(dimension.value());
            auto multiplied = checked_multiply(element_count, dimension.value(), name);
            if (!multiplied) return multiplied.error();
            element_count = multiplied.value();
        }
        auto begin = (*offsets_array.value())[0].as_u64();
        auto end = (*offsets_array.value())[1].as_u64();
        if (!begin || !end || begin.value() > end.value() || end.value() > payload_bytes) {
            return make_error(ErrorCode::invalid_format, "Safetensors data_offsets exceeds the shard payload", name);
        }
        auto expected_bytes = checked_multiply(element_count, dtype_bytes(tensor.dtype), name);
        if (!expected_bytes) return expected_bytes.error();
        tensor.byte_count = end.value() - begin.value();
        if (tensor.byte_count != expected_bytes.value()) {
            return make_error(ErrorCode::invalid_format, "Safetensors tensor byte count does not match shape and dtype", name);
        }
        tensor.absolute_offset = data_start + begin.value();
        result.by_name_.emplace(tensor.name, result.tensors_.size());
        result.tensors_.push_back(std::move(tensor));
    }

    std::vector<std::pair<ByteCount, ByteCount>> ranges;
    ranges.reserve(result.tensors_.size());
    for (const auto& tensor : result.tensors_) ranges.emplace_back(tensor.absolute_offset, tensor.absolute_offset + tensor.byte_count);
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].first < ranges[index - 1U].second) {
            return make_error(ErrorCode::invalid_format, "Safetensors tensor byte ranges overlap", path.string());
        }
    }
    return result;
}

const TensorDescriptor* SafetensorsFile::find(std::string_view name) const noexcept {
    const auto iterator = by_name_.find(name);
    return iterator == by_name_.end() ? nullptr : &tensors_[iterator->second];
}

Result<std::vector<std::byte>> SafetensorsFile::read_range(const TensorDescriptor& tensor,
                                                           ByteCount tensor_offset,
                                                           ByteCount bytes,
                                                           ByteCount max_request_bytes) const {
    const auto* owned = find(tensor.name);
    if (owned == nullptr || owned != &tensor) {
        return make_error(ErrorCode::invalid_argument, "Tensor descriptor does not belong to this shard", tensor.name);
    }
    if (bytes == 0U || bytes > max_request_bytes || tensor_offset > tensor.byte_count || bytes > tensor.byte_count - tensor_offset) {
        return make_error(ErrorCode::invalid_argument, "Safetensors read is outside the configured bounded range", tensor.name);
    }
    if (bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max()) ||
        tensor.absolute_offset + tensor_offset > static_cast<ByteCount>(std::numeric_limits<std::streamoff>::max())) {
        return make_error(ErrorCode::unsupported, "Safetensors read exceeds platform limits", tensor.name);
    }
    try {
        std::vector<std::byte> output(static_cast<std::size_t>(bytes));
        auto read = read_range_into(tensor, tensor_offset, output, max_request_bytes);
        if (!read) return read.error();
        return output;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate bounded Safetensors read buffer", tensor.name);
    }
}

Result<void> SafetensorsFile::read_range_into(const TensorDescriptor& tensor,
                                              ByteCount tensor_offset,
                                              std::span<std::byte> destination,
                                              ByteCount max_request_bytes) const {
    const auto* owned = find(tensor.name);
    const ByteCount bytes = static_cast<ByteCount>(destination.size());
    if (owned == nullptr || owned != &tensor) {
        return make_error(ErrorCode::invalid_argument, "Tensor descriptor does not belong to this shard", tensor.name);
    }
    if (bytes == 0U || bytes > max_request_bytes || tensor_offset > tensor.byte_count || bytes > tensor.byte_count - tensor_offset) {
        return make_error(ErrorCode::invalid_argument, "Safetensors read is outside the configured bounded range", tensor.name);
    }
    if (tensor.absolute_offset + tensor_offset > static_cast<ByteCount>(std::numeric_limits<std::streamoff>::max()) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        return make_error(ErrorCode::unsupported, "Safetensors read exceeds platform limits", tensor.name);
    }
    std::ifstream input(path_, std::ios::binary);
    if (!input) return make_error(ErrorCode::io_error, "Unable to open Safetensors shard", path_.string());
    input.seekg(static_cast<std::streamoff>(tensor.absolute_offset + tensor_offset));
    if (!input) return make_error(ErrorCode::io_error, "Unable to seek Safetensors shard", tensor.name);
    input.read(reinterpret_cast<char*>(destination.data()), static_cast<std::streamsize>(destination.size()));
    if (input.gcount() != static_cast<std::streamsize>(destination.size())) {
        return make_error(ErrorCode::io_error, "Short bounded read from Safetensors shard", tensor.name);
    }
    return {};
}

Result<CheckpointCatalog> inspect_safetensors_checkpoint(const std::filesystem::path& path) {
    std::error_code error;
    std::vector<std::filesystem::path> paths;
    if (std::filesystem::is_regular_file(path, error)) {
        if (!has_safetensors_extension(path)) return make_error(ErrorCode::invalid_argument, "Checkpoint file is not Safetensors", path.string());
        paths.push_back(path);
    } else if (!error && std::filesystem::is_directory(path, error)) {
        for (std::filesystem::directory_iterator iterator(path, std::filesystem::directory_options::skip_permission_denied, error), end;
             !error && iterator != end; iterator.increment(error)) {
            const auto& entry = *iterator;
            std::error_code entry_error;
            if (entry.is_regular_file(entry_error) && !entry_error && has_safetensors_extension(entry.path())) paths.push_back(entry.path());
            if (paths.size() > 1024U) return make_error(ErrorCode::resource_exhausted, "Checkpoint has too many Safetensors shards", path.string());
        }
        if (error) return make_error(ErrorCode::io_error, "Unable to enumerate checkpoint directory", path.string());
    } else {
        return make_error(ErrorCode::not_found, "Checkpoint path does not exist", path.string());
    }
    if (paths.empty()) return make_error(ErrorCode::not_found, "No Safetensors shards found", path.string());
    std::sort(paths.begin(), paths.end());

    CheckpointCatalog result;
    result.root = std::filesystem::is_directory(path, error) ? std::filesystem::absolute(path, error).lexically_normal()
                                                            : std::filesystem::absolute(path.parent_path(), error).lexically_normal();
    for (const auto& shard_path : paths) {
        auto shard = SafetensorsFile::open(shard_path);
        if (!shard) return shard.error();
        const std::size_t shard_index = result.shards.size();
        if (shard.value().file_bytes() > std::numeric_limits<ByteCount>::max() - result.total_file_bytes) {
            return make_error(ErrorCode::invalid_format, "Checkpoint file size overflows", path.string());
        }
        result.total_file_bytes += shard.value().file_bytes();
        for (std::size_t tensor_index = 0; tensor_index < shard.value().tensors().size(); ++tensor_index) {
            const auto& tensor = shard.value().tensors()[tensor_index];
            if (result.tensors.contains(tensor.name)) {
                return make_error(ErrorCode::integrity_error, "Duplicate tensor across Safetensors shards", tensor.name);
            }
            if (tensor.byte_count > std::numeric_limits<ByteCount>::max() - result.total_tensor_bytes) {
                return make_error(ErrorCode::invalid_format, "Checkpoint tensor size overflows", tensor.name);
            }
            result.total_tensor_bytes += tensor.byte_count;
            result.tensors.emplace(tensor.name, TensorLocation{shard_index, tensor_index});
        }
        result.shards.push_back(shard.take_value());
    }
    return result;
}

std::string_view to_string(TensorDType dtype) noexcept {
    switch (dtype) {
        case TensorDType::boolean: return "BOOL";
        case TensorDType::u8: return "U8";
        case TensorDType::i8: return "I8";
        case TensorDType::u16: return "U16";
        case TensorDType::i16: return "I16";
        case TensorDType::f16: return "F16";
        case TensorDType::bf16: return "BF16";
        case TensorDType::u32: return "U32";
        case TensorDType::i32: return "I32";
        case TensorDType::f32: return "F32";
        case TensorDType::f64: return "F64";
        case TensorDType::u64: return "U64";
        case TensorDType::i64: return "I64";
    }
    return "UNKNOWN";
}

std::string to_json(const CheckpointCatalog& catalog) {
    std::ostringstream output;
    output << "{\"ok\":true,\"root\":\"" << util::json_escape(catalog.root.string())
           << "\",\"shard_count\":" << catalog.shards.size()
           << ",\"tensor_count\":" << catalog.tensors.size()
           << ",\"total_file_bytes\":" << catalog.total_file_bytes
           << ",\"total_tensor_bytes\":" << catalog.total_tensor_bytes << ",\"dtypes\":{";
    std::map<std::string_view, std::size_t> dtypes;
    for (const auto& shard : catalog.shards) {
        for (const auto& tensor : shard.tensors()) ++dtypes[to_string(tensor.dtype)];
    }
    bool first = true;
    for (const auto& [dtype, count] : dtypes) {
        if (!first) output << ',';
        first = false;
        output << '\"' << dtype << "\":" << count;
    }
    output << "}}";
    return output.str();
}

std::string to_text(const CheckpointCatalog& catalog) {
    std::ostringstream output;
    output << "Safetensors checkpoint: " << catalog.root.string() << '\n'
           << "Shards:       " << catalog.shards.size() << '\n'
           << "Tensors:      " << catalog.tensors.size() << '\n'
           << "File bytes:   " << format_bytes(catalog.total_file_bytes) << '\n'
           << "Tensor bytes: " << format_bytes(catalog.total_tensor_bytes);
    return output.str();
}

}  // namespace stargate::model
