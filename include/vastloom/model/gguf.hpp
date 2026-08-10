#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::model {

enum class GgufValueType : std::uint32_t {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

struct GgufReaderLimits {
    ByteCount max_header_bytes{256U * MiB};
    std::size_t max_metadata_entries{1'000'000U};
    std::size_t max_tensors{1'000'000U};
    std::uint64_t max_array_elements{16'000'000U};
    ByteCount max_string_bytes{16U * MiB};
    std::uint32_t max_alignment{1U * MiB};
    std::uint32_t max_dimensions{8U};
    std::uint32_t max_array_depth{8U};
    // Array payloads are skipped by default. Callers may retain only explicitly named integral
    // arrays, within this separate aggregate bound, when architecture metadata is needed for a
    // safe execution plan.
    std::vector<std::string> retained_integral_array_keys;
    std::uint64_t max_retained_array_elements{65'536U};
};

struct GgufMetadataEntry {
    std::string key;
    GgufValueType type{GgufValueType::uint8};
    std::optional<GgufValueType> array_type;
    std::uint64_t array_elements{0U};
    ByteCount encoded_value_bytes{0U};
    std::optional<std::string> string_value;
    std::optional<std::uint64_t> unsigned_value;
    std::optional<std::int64_t> signed_value;
    std::optional<double> floating_value;
    std::optional<bool> boolean_value;
    std::vector<std::uint64_t> unsigned_array_values;
    std::vector<std::int64_t> signed_array_values;
};

struct GgufTensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t ggml_type{0U};
    std::uint64_t element_count{0U};
    ByteCount relative_offset{0U};
    ByteCount absolute_offset{0U};
    std::optional<ByteCount> encoded_bytes;
    // Upper bound ending at the next tensor offset or file end. It may include GGUF padding.
    ByteCount available_span_bytes{0U};
};

class GgufFile {
public:
    [[nodiscard]] static Result<GgufFile> open(
        const std::filesystem::path& path,
        const GgufReaderLimits& limits = {});
    // Parses a local prefix that contains the complete GGUF header/table. Bytes after the
    // aligned tensor-data boundary are ignored; declared_file_bytes describes the immutable
    // remote/full shard and is used to validate every relative tensor offset.
    [[nodiscard]] static Result<GgufFile> open_header_prefix(
        const std::filesystem::path& path,
        ByteCount declared_file_bytes,
        const GgufReaderLimits& limits = {});

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] ByteCount file_bytes() const noexcept { return file_bytes_; }
    [[nodiscard]] ByteCount data_offset() const noexcept { return data_offset_; }
    [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
    [[nodiscard]] std::uint32_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] bool header_only() const noexcept { return header_only_; }
    [[nodiscard]] const std::vector<GgufMetadataEntry>& metadata() const noexcept { return metadata_; }
    [[nodiscard]] const std::vector<GgufTensorDescriptor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const GgufMetadataEntry* find_metadata(std::string_view key) const noexcept;
    [[nodiscard]] const GgufTensorDescriptor* find_tensor(std::string_view name) const noexcept;
    [[nodiscard]] Result<std::string_view> architecture() const;

private:
    [[nodiscard]] static Result<GgufFile> open_with_declared_size(
        const std::filesystem::path& path,
        ByteCount declared_file_bytes,
        bool header_snapshot,
        const GgufReaderLimits& limits);

    std::filesystem::path path_;
    ByteCount file_bytes_{0U};
    ByteCount data_offset_{0U};
    std::uint32_t version_{0U};
    std::uint32_t alignment_{32U};
    bool header_only_{false};
    std::vector<GgufMetadataEntry> metadata_;
    std::vector<GgufTensorDescriptor> tensors_;
    std::map<std::string, std::size_t, std::less<>> metadata_by_key_;
    std::map<std::string, std::size_t, std::less<>> tensors_by_name_;
};

struct GgufTensorLocation {
    std::size_t shard_index{0U};
    std::size_t tensor_index{0U};
};

struct GgufSplitCatalog {
    std::string architecture;
    std::vector<GgufFile> shards;
    std::map<std::string, GgufTensorLocation, std::less<>> tensors;
    ByteCount total_file_bytes{0U};
    ByteCount total_header_bytes{0U};
    ByteCount total_payload_bytes{0U};
    ByteCount known_tensor_bytes{0U};
    std::size_t unknown_tensor_sizes{0U};
};

[[nodiscard]] Result<GgufSplitCatalog> finalize_gguf_split_catalog(
    std::vector<GgufFile> shards,
    std::size_t max_shards = 1'024U,
    std::size_t max_tensors = 1'000'000U);

[[nodiscard]] std::string_view to_string(GgufValueType type) noexcept;
[[nodiscard]] std::string to_json(const GgufFile& file, std::size_t tensor_limit = 0U);
[[nodiscard]] std::string to_text(const GgufFile& file, std::size_t tensor_limit = 0U);

}  // namespace vastloom::model
