#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::model {

enum class TensorDType : std::uint8_t {
    boolean,
    u8,
    i8,
    u16,
    i16,
    f16,
    bf16,
    u32,
    i32,
    f32,
    f64,
    u64,
    i64,
};

struct TensorDescriptor {
    std::string name;
    TensorDType dtype{TensorDType::f32};
    std::vector<std::uint64_t> shape;
    ByteCount absolute_offset{0};
    ByteCount byte_count{0};
};

class SafetensorsFile {
public:
    [[nodiscard]] static Result<SafetensorsFile> open(
        const std::filesystem::path& path,
        ByteCount max_header_bytes = 128U * MiB,
        std::size_t max_tensors = 1'000'000);

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] ByteCount file_bytes() const noexcept { return file_bytes_; }
    [[nodiscard]] ByteCount header_bytes() const noexcept { return header_bytes_; }
    [[nodiscard]] const std::vector<TensorDescriptor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const TensorDescriptor* find(std::string_view name) const noexcept;
    [[nodiscard]] Result<std::vector<std::byte>> read_range(
        const TensorDescriptor& tensor,
        ByteCount tensor_offset,
        ByteCount bytes,
        ByteCount max_request_bytes = 64U * MiB) const;
    [[nodiscard]] Result<void> read_range_into(const TensorDescriptor& tensor,
                                               ByteCount tensor_offset,
                                               std::span<std::byte> destination,
                                               ByteCount max_request_bytes = 64U * MiB) const;

private:
    std::filesystem::path path_;
    ByteCount file_bytes_{0};
    ByteCount header_bytes_{0};
    std::vector<TensorDescriptor> tensors_;
    std::map<std::string, std::size_t, std::less<>> by_name_;
};

struct TensorLocation {
    std::size_t shard_index{0};
    std::size_t tensor_index{0};
};

struct CheckpointCatalog {
    std::filesystem::path root;
    std::vector<SafetensorsFile> shards;
    std::map<std::string, TensorLocation, std::less<>> tensors;
    ByteCount total_file_bytes{0};
    ByteCount total_tensor_bytes{0};
};

[[nodiscard]] Result<CheckpointCatalog> inspect_safetensors_checkpoint(const std::filesystem::path& path);
[[nodiscard]] std::string_view to_string(TensorDType dtype) noexcept;
[[nodiscard]] std::string to_json(const CheckpointCatalog& catalog);
[[nodiscard]] std::string to_text(const CheckpointCatalog& catalog);

}  // namespace vastloom::model
