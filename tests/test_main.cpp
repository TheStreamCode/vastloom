#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "vastloom/api/openai_protocol.hpp"
#include "vastloom/core/bytes.hpp"
#include "vastloom/hardware/hardware_probe.hpp"
#include "vastloom/inference/gpt2_tokenizer.hpp"
#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/inference/deepseek_v4_gguf.hpp"
#include "vastloom/inference/deepseek_v4_expert_provider.hpp"
#include "vastloom/inference/deepseek_v4_model.hpp"
#include "vastloom/inference/deepseek_v4_range_artifact.hpp"
#include "vastloom/inference/deepseek_v4_reference.hpp"
#include "vastloom/inference/llama_model.hpp"
#include "vastloom/inference/olmoe_model.hpp"
#include "vastloom/inference/olmoe_runtime.hpp"
#include "vastloom/inference/tiny_model.hpp"
#include "vastloom/io/bounded_io_pool.hpp"
#include "vastloom/io/device_io_counters.hpp"
#include "vastloom/io/process_io_counters.hpp"
#include "vastloom/model/manifest.hpp"
#include "vastloom/model/imported_checkpoint.hpp"
#include "vastloom/model/expert_loader.hpp"
#include "vastloom/model/gguf.hpp"
#include "vastloom/model/layer_streaming.hpp"
#include "vastloom/model/model_store.hpp"
#include "vastloom/model/residency.hpp"
#include "vastloom/model/safetensors.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/model/verifier.hpp"
#include "vastloom/planner/execution_planner.hpp"
#include "vastloom/research/trace_simulator.hpp"
#include "vastloom/safety/heartbeat.hpp"
#include "vastloom/safety/resource_governor.hpp"
#include "vastloom/safety/runtime_policy.hpp"
#include "vastloom/util/json.hpp"
#include "vastloom/util/sha256.hpp"
#include "vastloom/util/string.hpp"

namespace {

int failures = 0;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " #condition "\n"; \
            ++failures; \
        } \
    } while (false)

std::filesystem::path make_temp_dir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("vastloom-tests-" + std::to_string(stamp));
    std::filesystem::create_directories(path);
    return path;
}

void test_bytes() {
    auto parsed = vastloom::parse_bytes("32GiB");
    CHECK(parsed && parsed.value() == 32U * vastloom::GiB);
    auto decimal = vastloom::parse_bytes("1.5 GB");
    CHECK(decimal && decimal.value() == 1'500'000'000ULL);
    CHECK(!vastloom::parse_bytes("-1GiB"));
    CHECK(!vastloom::parse_bytes("10 bananas"));
    CHECK(vastloom::format_bytes(2U * vastloom::GiB) == "2.00 GiB");
}

void test_sha256() {
    const std::string input = "abc";
    vastloom::util::Sha256 hash;
    hash.update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(input.data()), input.size()));
    CHECK(hash.finalize_hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}


void test_json_escape() {
    CHECK(vastloom::util::json_escape("quote\" slash\\ newline\n") ==
          "quote\\\" slash\\\\ newline\\n");
    const std::string valid_utf8 = "Qual è la capitale?";
    CHECK(vastloom::util::json_escape(valid_utf8) == valid_utf8);
    const std::string invalid_utf8{static_cast<char>(0xf1), 'x', static_cast<char>(0x80)};
    CHECK(vastloom::util::json_escape(invalid_utf8) == "\\ufffdx\\ufffd");
}

void test_bounded_json() {
    auto parsed = vastloom::util::parse_json(R"({"name":"Vastloom","value":42,"unicode":"\u00e8","items":[true,null]})");
    CHECK(parsed && parsed.value().is_object());
    if (parsed) {
        const auto* name = parsed.value().find("name");
        const auto* value = parsed.value().find("value");
        const auto* unicode = parsed.value().find("unicode");
        CHECK(name != nullptr && name->as_string() && name->as_string().value() == "Vastloom");
        CHECK(value != nullptr && value->as_u64() && value->as_u64().value() == 42U);
        CHECK(unicode != nullptr && unicode->as_string() && unicode->as_string().value() == "è");
    }
    CHECK(!vastloom::util::parse_json(R"({"duplicate":1,"duplicate":2})"));
    CHECK(!vastloom::util::parse_json(R"([1,])"));
    vastloom::util::JsonLimits limits;
    limits.max_input_bytes = 4U;
    CHECK(!vastloom::util::parse_json("[1,2]", limits));
    limits = {};
    limits.max_depth = 2U;
    CHECK(!vastloom::util::parse_json("[[[0]]]", limits));
}

void write_u64_le(std::ofstream& output, std::uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<unsigned char>((value >> (index * 8U)) & 0xffU);
    }
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void append_u32_le(std::vector<unsigned char>& output, std::uint32_t value) {
    for (std::size_t index = 0U; index < 4U; ++index) {
        output.push_back(static_cast<unsigned char>((value >> (index * 8U)) & 0xffU));
    }
}

void append_u64_le(std::vector<unsigned char>& output, std::uint64_t value) {
    for (std::size_t index = 0U; index < 8U; ++index) {
        output.push_back(static_cast<unsigned char>((value >> (index * 8U)) & 0xffU));
    }
}

void append_gguf_string(std::vector<unsigned char>& output, std::string_view value) {
    append_u64_le(output, value.size());
    output.insert(output.end(), value.begin(), value.end());
}

void write_bytes(const std::filesystem::path& path, std::span<const unsigned char> bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

void write_bytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

void test_safetensors(const std::filesystem::path& root) {
    const auto checkpoint = root / "safetensors";
    std::filesystem::create_directories(checkpoint);
    const std::string header = R"({"weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}})";
    {
        std::ofstream output(checkpoint / "model.safetensors", std::ios::binary);
        write_u64_le(output, header.size());
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        const std::array<unsigned char, 8> payload{0x80U, 0x3fU, 0x00U, 0x40U, 0x40U, 0x40U, 0x80U, 0x40U};
        output.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
    auto shard = vastloom::model::SafetensorsFile::open(checkpoint / "model.safetensors");
    CHECK(shard && shard.value().tensors().size() == 1U);
    if (shard) {
        const auto* tensor = shard.value().find("weight");
        CHECK(tensor != nullptr && tensor->byte_count == 8U);
        if (tensor != nullptr) {
            auto bytes = shard.value().read_range(*tensor, 0U, 8U);
            CHECK(bytes && bytes.value().size() == 8U);
            CHECK(!shard.value().read_range(*tensor, 0U, 8U, 4U));
            std::array<std::byte, 8> destination{};
            CHECK(shard.value().read_range_into(*tensor, 0U, destination));
        }
    }
    auto catalog = vastloom::model::inspect_safetensors_checkpoint(checkpoint);
    CHECK(catalog && catalog.value().shards.size() == 1U && catalog.value().tensors.size() == 1U);

    const auto snapshot_path = checkpoint / "model.safetensors.header";
    {
        std::ofstream output(snapshot_path, std::ios::binary);
        write_u64_le(output, header.size());
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    auto snapshot = vastloom::model::SafetensorsFile::open_header_snapshot(
        snapshot_path, 8U + header.size() + 8U);
    CHECK(snapshot && snapshot.value().header_only() && snapshot.value().file_bytes() == 8U + header.size() + 8U);
    if (snapshot) {
        const auto* tensor = snapshot.value().find("weight");
        CHECK(tensor != nullptr && tensor->byte_count == 8U);
        if (tensor != nullptr) CHECK(!snapshot.value().read_range(*tensor, 0U, 8U));
    }
    CHECK(!vastloom::model::SafetensorsFile::open_header_snapshot(snapshot_path, 8U + header.size() + 7U));

    const std::string fp8_header = R"({"data":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]},"scale":{"dtype":"F8_E8M0","shape":[1],"data_offsets":[1,2]}})";
    const auto fp8_snapshot_path = checkpoint / "fp8.safetensors.header";
    {
        std::ofstream output(fp8_snapshot_path, std::ios::binary);
        write_u64_le(output, fp8_header.size());
        output.write(fp8_header.data(), static_cast<std::streamsize>(fp8_header.size()));
    }
    auto fp8_snapshot = vastloom::model::SafetensorsFile::open_header_snapshot(
        fp8_snapshot_path, 8U + fp8_header.size() + 2U);
    CHECK(fp8_snapshot && fp8_snapshot.value().tensors().size() == 2U);
    if (fp8_snapshot) {
        CHECK(fp8_snapshot.value().find("data") != nullptr &&
              fp8_snapshot.value().find("data")->dtype == vastloom::model::TensorDType::f8_e4m3);
        CHECK(fp8_snapshot.value().find("scale") != nullptr &&
              fp8_snapshot.value().find("scale")->dtype == vastloom::model::TensorDType::f8_e8m0);
    }

    const std::string invalid_header = R"({"weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,9]}})";
    {
        std::ofstream output(checkpoint / "invalid.safetensors", std::ios::binary);
        write_u64_le(output, invalid_header.size());
        output.write(invalid_header.data(), static_cast<std::streamsize>(invalid_header.size()));
        output.write("123456789", 9);
    }
    CHECK(!vastloom::model::SafetensorsFile::open(checkpoint / "invalid.safetensors"));
}

void test_gguf(const std::filesystem::path& root) {
    const auto directory = root / std::filesystem::path(std::u8string(u8"gguf-àccènt"));
    std::filesystem::create_directories(directory);
    const auto& pinned = vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
    CHECK(pinned.model_id == "unsloth/DeepSeek-V4-Flash-0731-GGUF" &&
          pinned.revision == "fbbb5b93fb787c21338159b0af3318bb3f4d9768" &&
          pinned.shards.size() == 3U && pinned.layers == 43U &&
          pinned.experts_per_layer == 256U && pinned.top_k == 6U);
    if (pinned.shards.size() == 3U) {
        CHECK(pinned.shards[0].full_bytes + pinned.shards[1].full_bytes +
                  pinned.shards[2].full_bytes ==
              82'539'237'792U);
        CHECK(pinned.shards[0].metadata_bytes + pinned.shards[1].metadata_bytes +
                  pinned.shards[2].metadata_bytes ==
              5'343'584U);
    }
    std::vector<unsigned char> header;
    append_u32_le(header, 0x46554747U);
    append_u32_le(header, 3U);
    append_u64_le(header, 3U);
    append_u64_le(header, 23U);

    append_gguf_string(header, "general.architecture");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::string));
    append_gguf_string(header, "deepseek4");

    append_gguf_string(header, "general.alignment");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::uint32));
    const std::size_t alignment_value_offset = header.size();
    append_u32_le(header, 32U);

    append_gguf_string(header, "general.quantization_version");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::uint32));
    append_u32_le(header, 2U);

    append_gguf_string(header, "split.no");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::uint16));
    header.push_back(0U);
    header.push_back(0U);
    append_gguf_string(header, "split.count");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::uint16));
    header.push_back(2U);
    header.push_back(0U);

    append_gguf_string(header, "tokenizer.ggml.tokens");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::array));
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::string));
    append_u64_le(header, 2U);
    append_gguf_string(header, "a");
    append_gguf_string(header, "\xc3\xa8");

    const std::array<std::pair<std::string_view, std::uint32_t>, 15U> deepseek_metadata{{
        {"deepseek4.block_count", 1U},
        {"deepseek4.hash_layer_count", 0U},
        {"deepseek4.context_length", 128U},
        {"deepseek4.embedding_length", 8U},
        {"deepseek4.attention.head_count", 2U},
        {"deepseek4.attention.key_length", 4U},
        {"deepseek4.attention.value_length", 4U},
        {"deepseek4.rope.dimension_count", 2U},
        {"deepseek4.attention.sliding_window", 4U},
        {"deepseek4.expert_count", 2U},
        {"deepseek4.expert_used_count", 1U},
        {"deepseek4.expert_feed_forward_length", 4U},
        {"deepseek4.attention.indexer.head_count", 2U},
        {"deepseek4.attention.indexer.key_length", 2U},
        {"deepseek4.hyper_connection.count", 2U},
    }};
    std::optional<std::size_t> hidden_metadata_value_offset;
    for (const auto& [key, value] : deepseek_metadata) {
        append_gguf_string(header, key);
        append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::uint32));
        if (key == "deepseek4.embedding_length") {
            hidden_metadata_value_offset = header.size();
        }
        append_u32_le(header, value);
    }
    append_gguf_string(header, "deepseek4.attention.compress_ratios");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::array));
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::int32));
    append_u64_le(header, 1U);
    append_u32_le(header, 0U);

    append_gguf_string(header, "deepseek4.swiglu_clamp_exp");
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::array));
    append_u32_le(header, static_cast<std::uint32_t>(vastloom::model::GgufValueType::float32));
    append_u64_le(header, 2U);
    append_u32_le(header, std::bit_cast<std::uint32_t>(0.5F));
    append_u32_le(header, std::bit_cast<std::uint32_t>(1.25F));

    append_gguf_string(header, "blk.0.ffn_down_exps.weight");
    append_u32_le(header, 3U);
    append_u64_le(header, 32U);
    append_u64_le(header, 1U);
    append_u64_le(header, 2U);
    append_u32_le(header, 8U);
    append_u64_le(header, 0U);

    append_gguf_string(header, "blk.0.ffn_gate_exps.weight");
    append_u32_le(header, 3U);
    append_u64_le(header, 32U);
    append_u64_le(header, 1U);
    append_u64_le(header, 2U);
    append_u32_le(header, 8U);
    const std::size_t second_tensor_offset = header.size();
    append_u64_le(header, 96U);

    append_gguf_string(header, "blk.0.ffn_up_exps.weight");
    append_u32_le(header, 3U);
    append_u64_le(header, 32U);
    append_u64_le(header, 1U);
    append_u64_le(header, 2U);
    append_u32_le(header, 8U);
    append_u64_le(header, 192U);

    while (header.size() % 32U != 0U) header.push_back(0U);
    const vastloom::ByteCount data_offset = header.size();
    std::vector<unsigned char> full = header;
    full.resize(full.size() + 288U, 0U);
    const auto full_path = directory / "model.gguf";
    write_bytes(full_path, full);

    vastloom::model::GgufReaderLimits deepseek_limits;
    deepseek_limits.retained_integral_array_keys = {
        "deepseek4.attention.compress_ratios"};
    deepseek_limits.retained_floating_array_keys = {
        "deepseek4.swiglu_clamp_exp"};
    auto gguf = vastloom::model::GgufFile::open(full_path, deepseek_limits);
    CHECK(hidden_metadata_value_offset.has_value());
    auto oversized_architecture = full;
    if (hidden_metadata_value_offset.has_value()) {
        std::fill(oversized_architecture.begin() +
                      static_cast<std::ptrdiff_t>(*hidden_metadata_value_offset),
                  oversized_architecture.begin() +
                      static_cast<std::ptrdiff_t>(*hidden_metadata_value_offset + 4U),
                  static_cast<unsigned char>(0xffU));
    }
    const auto oversized_architecture_path = directory / "oversized-architecture.gguf";
    write_bytes(oversized_architecture_path, oversized_architecture);
    auto oversized_architecture_file = vastloom::model::GgufFile::open(
        oversized_architecture_path, deepseek_limits);
    auto malformed_layer = full;
    constexpr std::string_view layer_name = "blk.0.ffn_down_exps.weight";
    const auto layer_name_position = std::search(malformed_layer.begin(), malformed_layer.end(),
                                                 layer_name.begin(), layer_name.end());
    CHECK(layer_name_position != malformed_layer.end());
    if (layer_name_position != malformed_layer.end()) {
        *(layer_name_position + 4) = static_cast<unsigned char>('x');
    }
    const auto malformed_layer_path = directory / "malformed-layer.gguf";
    write_bytes(malformed_layer_path, malformed_layer);
    auto malformed_layer_file = vastloom::model::GgufFile::open(
        malformed_layer_path, deepseek_limits);
    CHECK(gguf && gguf.value().version() == 3U && gguf.value().alignment() == 32U &&
          gguf.value().data_offset() == data_offset && !gguf.value().header_only() &&
          gguf.value().metadata().size() == 23U && gguf.value().tensors().size() == 3U);
    if (gguf) {
        auto architecture = gguf.value().architecture();
        CHECK(architecture && architecture.value() == "deepseek4");
        const auto* tokens = gguf.value().find_metadata("tokenizer.ggml.tokens");
        CHECK(tokens != nullptr && tokens->type == vastloom::model::GgufValueType::array &&
              tokens->array_type == vastloom::model::GgufValueType::string &&
              tokens->array_elements == 2U);
        const auto* ratios = gguf.value().find_metadata(
            "deepseek4.attention.compress_ratios");
        CHECK(ratios != nullptr && ratios->signed_array_values.size() == 1U &&
              ratios->signed_array_values[0] == 0);
        const auto* clamps = gguf.value().find_metadata(
            "deepseek4.swiglu_clamp_exp");
        CHECK(clamps != nullptr && clamps->floating_array_values.size() == 2U &&
              clamps->floating_array_values[0] == 0.5 &&
              clamps->floating_array_values[1] == 1.25);
        const auto* down = gguf.value().find_tensor("blk.0.ffn_down_exps.weight");
        const auto* gate = gguf.value().find_tensor("blk.0.ffn_gate_exps.weight");
        const auto* up = gguf.value().find_tensor("blk.0.ffn_up_exps.weight");
        CHECK(down != nullptr && down->element_count == 64U && down->encoded_bytes == 68U &&
              down->absolute_offset == data_offset && down->available_span_bytes == 96U);
        CHECK(gate != nullptr && gate->relative_offset == 96U && gate->encoded_bytes == 68U &&
              gate->absolute_offset == data_offset + 96U && gate->available_span_bytes == 96U);
        CHECK(up != nullptr && up->relative_offset == 192U && up->encoded_bytes == 68U &&
              up->available_span_bytes == 96U);
        CHECK(gguf.value().find_tensor("missing") == nullptr);
    }

    const auto snapshot_path = directory / "model.gguf.header";
    write_bytes(snapshot_path, header);
    auto snapshot = vastloom::model::GgufFile::open_header_prefix(
        snapshot_path, full.size(), deepseek_limits);
    CHECK(snapshot && snapshot.value().header_only() && snapshot.value().file_bytes() == full.size() &&
          snapshot.value().data_offset() == data_offset && snapshot.value().tensors().size() == 3U);

    auto bad_magic = full;
    bad_magic[0] = 0U;
    const auto bad_magic_path = directory / "bad-magic.gguf";
    write_bytes(bad_magic_path, bad_magic);
    CHECK(!vastloom::model::GgufFile::open(bad_magic_path));

    auto bad_count = full;
    std::fill(bad_count.begin() + 8, bad_count.begin() + 16,
              static_cast<unsigned char>(0xffU));
    const auto bad_count_path = directory / "bad-count.gguf";
    write_bytes(bad_count_path, bad_count);
    CHECK(!vastloom::model::GgufFile::open(bad_count_path));

    auto bad_alignment = full;
    bad_alignment[alignment_value_offset] = 7U;
    bad_alignment[alignment_value_offset + 1U] = 0U;
    bad_alignment[alignment_value_offset + 2U] = 0U;
    bad_alignment[alignment_value_offset + 3U] = 0U;
    const auto bad_alignment_path = directory / "bad-alignment.gguf";
    write_bytes(bad_alignment_path, bad_alignment);
    CHECK(!vastloom::model::GgufFile::open(bad_alignment_path));

    auto bad_offset = full;
    bad_offset[second_tensor_offset] = 97U;
    const auto bad_offset_path = directory / "bad-offset.gguf";
    write_bytes(bad_offset_path, bad_offset);
    CHECK(!vastloom::model::GgufFile::open(bad_offset_path));

    vastloom::model::GgufReaderLimits array_limits;
    array_limits.max_array_elements = 1U;
    CHECK(!vastloom::model::GgufFile::open(full_path, array_limits));
    vastloom::model::GgufReaderLimits non_integral_retention;
    non_integral_retention.retained_integral_array_keys = {"tokenizer.ggml.tokens"};
    CHECK(!vastloom::model::GgufFile::open(full_path, non_integral_retention));
    vastloom::model::GgufReaderLimits scalar_retention;
    scalar_retention.retained_integral_array_keys = {"general.alignment"};
    CHECK(!vastloom::model::GgufFile::open(full_path, scalar_retention));
    scalar_retention = {};
    scalar_retention.retained_floating_array_keys = {"general.alignment"};
    CHECK(!vastloom::model::GgufFile::open(full_path, scalar_retention));
    vastloom::model::GgufReaderLimits duplicate_retention;
    duplicate_retention.retained_integral_array_keys = {
        "deepseek4.attention.compress_ratios",
        "deepseek4.attention.compress_ratios"};
    CHECK(!vastloom::model::GgufFile::open(full_path, duplicate_retention));
    vastloom::model::GgufReaderLimits wrong_float_retention;
    wrong_float_retention.retained_floating_array_keys = {
        "deepseek4.attention.compress_ratios"};
    CHECK(!vastloom::model::GgufFile::open(full_path, wrong_float_retention));
    vastloom::model::GgufReaderLimits wrong_integral_retention;
    wrong_integral_retention.retained_integral_array_keys = {
        "deepseek4.swiglu_clamp_exp"};
    CHECK(!vastloom::model::GgufFile::open(full_path, wrong_integral_retention));
    vastloom::model::GgufReaderLimits cross_duplicate_retention;
    cross_duplicate_retention.retained_integral_array_keys = {
        "deepseek4.attention.compress_ratios"};
    cross_duplicate_retention.retained_floating_array_keys = {
        "deepseek4.attention.compress_ratios"};
    CHECK(!vastloom::model::GgufFile::open(full_path, cross_duplicate_retention));
    vastloom::model::GgufReaderLimits retained_array_budget;
    retained_array_budget.retained_integral_array_keys = {
        "deepseek4.attention.compress_ratios"};
    retained_array_budget.retained_floating_array_keys = {
        "deepseek4.swiglu_clamp_exp"};
    retained_array_budget.max_retained_array_elements = 2U;
    CHECK(!vastloom::model::GgufFile::open(full_path, retained_array_budget));

    auto truncated_header = header;
    truncated_header.pop_back();
    const auto truncated_path = directory / "truncated.gguf.header";
    write_bytes(truncated_path, truncated_header);
    CHECK(!vastloom::model::GgufFile::open_header_prefix(truncated_path, full.size()));

    std::vector<unsigned char> continuation;
    append_u32_le(continuation, 0x46554747U);
    append_u32_le(continuation, 3U);
    append_u64_le(continuation, 1U);
    append_u64_le(continuation, 2U);
    append_gguf_string(continuation, "split.no");
    append_u32_le(continuation, static_cast<std::uint32_t>(vastloom::model::GgufValueType::uint16));
    continuation.push_back(1U);
    continuation.push_back(0U);
    append_gguf_string(continuation, "split.count");
    append_u32_le(continuation, static_cast<std::uint32_t>(vastloom::model::GgufValueType::uint16));
    continuation.push_back(2U);
    continuation.push_back(0U);
    append_gguf_string(continuation, "blk.0.attn_output.weight");
    append_u32_le(continuation, 1U);
    append_u64_le(continuation, 1U);
    append_u32_le(continuation, 0U);
    append_u64_le(continuation, 0U);
    while (continuation.size() % 32U != 0U) continuation.push_back(0U);
    continuation.resize(continuation.size() + 4U, 0U);
    const auto continuation_path = directory / "continuation.gguf";
    write_bytes(continuation_path, continuation);
    auto continuation_file = vastloom::model::GgufFile::open(continuation_path);
    CHECK(continuation_file && !continuation_file.value().architecture() &&
          continuation_file.value().tensors().size() == 1U);
    if (gguf && continuation_file) {
        std::vector<vastloom::model::GgufFile> parts;
        parts.push_back(gguf.value());
        parts.push_back(continuation_file.value());
        auto catalog = vastloom::model::finalize_gguf_split_catalog(std::move(parts));
        CHECK(catalog && catalog.value().architecture == "deepseek4" &&
              catalog.value().shards.size() == 2U && catalog.value().tensors.size() == 4U &&
              catalog.value().known_tensor_bytes == 208U &&
              catalog.value().unknown_tensor_sizes == 0U);
        if (catalog) {
            vastloom::inference::DeepseekV4GgufCatalogOptions options;
            options.model_id = "test/deepseek4-gguf";
            options.revision = "fixture";
            options.layers = 1U;
            options.experts_per_layer = 2U;
            options.top_k = 1U;
            options.context_tokens = 128U;
            auto mapped = vastloom::inference::build_deepseek_v4_gguf_catalog(
                catalog.value(), options);
            CHECK(mapped && mapped.value().catalog.tensors.size() == 7U &&
                  mapped.value().catalog.experts.size() == 2U &&
                  mapped.value().catalog.routed_expert_bytes == 204U &&
                  mapped.value().catalog.always_active_bytes == 4U &&
                  mapped.value().layer_trunk_planned_layers == 1U &&
                  mapped.value().layer_trunk_tensor_count == 1U &&
                  mapped.value().layer_trunk_tile_count == 1U &&
                  mapped.value().minimum_layer_trunk_bytes == 4U &&
                  mapped.value().maximum_layer_trunk_bytes == 4U &&
                  mapped.value().maximum_layer_trunk_tensor_bytes == 4U &&
                  mapped.value().maximum_layer_trunk_tile_bytes == 4U &&
                  mapped.value().reference_storage_type_covered_tensor_count == 1U &&
                  mapped.value().reference_storage_type_covered_bytes == 4U &&
                  mapped.value().reference_storage_type_unsupported_tensor_count == 0U &&
                  mapped.value().reference_storage_type_unsupported_bytes == 0U &&
                  mapped.value().reference_storage_types_complete &&
                  mapped.value().minimum_expert_bundle_bytes == 102U &&
                  mapped.value().maximum_expert_bundle_bytes == 102U &&
                  mapped.value().no_cache_expert_bytes_per_token == 102U &&
                  mapped.value().persistent_kv_bytes == 32U &&
                  mapped.value().compression_state_bytes == 0U &&
                  mapped.value().rope_table_bytes == 1'024U &&
                  mapped.value().runtime_state_bytes == 67'110'300U &&
                  mapped.value().minimum_host_resident_bytes == 67'110'304U &&
                  mapped.value().minimum_streamed_trunk_cpu_ram_bytes == 67'110'402U &&
                  mapped.value().minimum_cpu_streaming_ram_bytes == 67'110'406U);
            if (mapped) {
                CHECK(mapped.value().catalog.experts.front().read_ranges.size() == 3U &&
                      mapped.value().catalog.experts.front().tensor_slices.size() == 3U);
                const std::array<std::uint32_t, 1U> selected{1U};
                auto probe = vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    mapped.value(), 0U, selected, 4U, 8U);
                CHECK(probe && !probe.value().metadata_only &&
                      probe.value().layer == 0U && probe.value().expected_top_k == 1U &&
                      probe.value().selected_experts == std::vector<std::uint32_t>{1U} &&
                      probe.value().trunk_bytes == 4U &&
                      probe.value().selected_expert_bytes == 102U &&
                      probe.value().total_payload_bytes == 106U &&
                      probe.value().trunk_range_count == 1U &&
                      probe.value().selected_expert_range_count == 3U &&
                      probe.value().total_range_count == 4U &&
                      probe.value().unique_source_objects == 2U &&
                      probe.value().maximum_read_range_bytes == 34U &&
                      probe.value().minimum_expert_tier_bytes == 102U &&
                      probe.value().experts.size() == 1U &&
                      probe.value().experts.front().tensors.size() == 3U);
                if (probe) {
                    const auto json = vastloom::inference::to_json(probe.value());
                    CHECK(json.find("\"plan_only\":true") != std::string::npos &&
                          json.find("\"execution_supported\":false") != std::string::npos &&
                          json.find("\"routing_authority_verified\":false") !=
                              std::string::npos &&
                          json.find("\"total_payload_bytes\":106") != std::string::npos &&
                          json.find("model.gguf") != std::string::npos &&
                          vastloom::inference::to_text(probe.value()).find(
                              "planner cannot prove it") != std::string::npos);
                }
                const std::array<std::uint32_t, 1U> out_of_range{2U};
                const std::array<std::uint32_t, 2U> wrong_count{0U, 1U};
                const std::array<std::uint32_t, 2U> duplicate{1U, 1U};
                CHECK(!vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    mapped.value(), 0U, out_of_range, 4U, 8U));
                CHECK(!vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    mapped.value(), 0U, wrong_count, 4U, 8U));
                CHECK(!vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    mapped.value(), 0U, selected, 4U, 3U));
                CHECK(!vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    mapped.value(), 1U, selected, 4U, 8U));
                auto duplicate_report = mapped.value();
                duplicate_report.top_k = 2U;
                CHECK(!vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    duplicate_report, 0U, duplicate, 4U, 8U));
                auto missing_expert = mapped.value();
                missing_expert.catalog.experts.pop_back();
                CHECK(!vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    missing_expert, 0U, selected, 4U, 8U));
            }
            options.safe_ram_cap_bytes = 67'110'360U;
            options.safe_vram_cap_bytes = 60U;
            auto cannot_split_expert = vastloom::inference::build_deepseek_v4_gguf_catalog(
                catalog.value(), options);
            CHECK(cannot_split_expert && cannot_split_expert.value().admission_evaluated &&
                  !cannot_split_expert.value().expert_fits_ram_tier &&
                  !cannot_split_expert.value().expert_fits_vram_tier &&
                  !cannot_split_expert.value().capacity_plan_candidate);
            options.safe_vram_cap_bytes = 102U;
            auto vram_expert = vastloom::inference::build_deepseek_v4_gguf_catalog(
                catalog.value(), options);
            CHECK(vram_expert && vram_expert.value().expert_fits_vram_tier &&
                  vram_expert.value().capacity_plan_candidate);
            options.top_k = 2U;
            CHECK(!vastloom::inference::build_deepseek_v4_gguf_catalog(
                catalog.value(), options));
        }
    }
    if (snapshot && continuation_file) {
        std::vector<vastloom::model::GgufFile> header_parts;
        header_parts.push_back(snapshot.value());
        header_parts.push_back(continuation_file.value());
        auto header_catalog = vastloom::model::finalize_gguf_split_catalog(
            std::move(header_parts));
        CHECK(header_catalog);
        if (header_catalog) {
            vastloom::inference::DeepseekV4GgufCatalogOptions options;
            options.model_id = "test/deepseek4-header-only";
            options.revision = "fixture";
            options.layers = 1U;
            options.experts_per_layer = 2U;
            options.top_k = 1U;
            options.context_tokens = 128U;
            CHECK(!vastloom::inference::build_deepseek_v4_gguf_catalog(
                header_catalog.value(), options));
            options.metadata_only = true;
            CHECK(vastloom::inference::build_deepseek_v4_gguf_catalog(
                header_catalog.value(), options));
        }
    }
    if (oversized_architecture_file && continuation_file) {
        std::vector<vastloom::model::GgufFile> oversized_parts;
        oversized_parts.push_back(oversized_architecture_file.take_value());
        oversized_parts.push_back(continuation_file.value());
        auto oversized_catalog = vastloom::model::finalize_gguf_split_catalog(
            std::move(oversized_parts));
        CHECK(oversized_catalog);
        if (oversized_catalog) {
            vastloom::inference::DeepseekV4GgufCatalogOptions options;
            options.model_id = "test/deepseek4-oversized";
            options.revision = "fixture";
            options.layers = 1U;
            options.experts_per_layer = 2U;
            options.top_k = 1U;
            options.context_tokens = 128U;
            CHECK(!vastloom::inference::build_deepseek_v4_gguf_catalog(
                oversized_catalog.value(), options));
        }
    }

    if (malformed_layer_file && continuation_file) {
        std::vector<vastloom::model::GgufFile> malformed_parts;
        malformed_parts.push_back(malformed_layer_file.take_value());
        malformed_parts.push_back(continuation_file.value());
        auto malformed_catalog = vastloom::model::finalize_gguf_split_catalog(
            std::move(malformed_parts));
        CHECK(malformed_catalog);
        if (malformed_catalog) {
            vastloom::inference::DeepseekV4GgufCatalogOptions options;
            options.model_id = "test/deepseek4-malformed-layer";
            options.revision = "fixture";
            options.layers = 1U;
            options.experts_per_layer = 2U;
            options.top_k = 1U;
            options.context_tokens = 128U;
            CHECK(!vastloom::inference::build_deepseek_v4_gguf_catalog(
                malformed_catalog.value(), options));
        }
    }

    if (continuation_file) {
        const auto full_import = directory / "checkpoint-full";
        const auto metadata_import = directory / "checkpoint-metadata";
        std::filesystem::create_directories(full_import);
        std::filesystem::create_directories(metadata_import / "headers");
        const auto full_part_1 = full_import / "part-1.gguf";
        const auto full_part_2 = full_import / "part-2.gguf";
        const auto header_part_1 = metadata_import / "headers" / "part-1.gguf.header";
        const auto header_part_2 = metadata_import / "headers" / "part-2.gguf.header";
        write_bytes(full_part_1, full);
        write_bytes(full_part_2, continuation);
        write_bytes(header_part_1, header);
        const auto continuation_header_bytes = static_cast<std::size_t>(
            continuation_file.value().data_offset());
        std::vector<unsigned char> continuation_header(
            continuation.begin(),
            continuation.begin() + static_cast<std::ptrdiff_t>(continuation_header_bytes));
        write_bytes(header_part_2, continuation_header);

        const auto full_hash_1 = vastloom::util::sha256_file(full_part_1);
        const auto full_hash_2 = vastloom::util::sha256_file(full_part_2);
        const auto header_hash_1 = vastloom::util::sha256_file(header_part_1);
        const auto header_hash_2 = vastloom::util::sha256_file(header_part_2);
        CHECK(full_hash_1 && full_hash_2 && header_hash_1 && header_hash_2);
        if (full_hash_1 && full_hash_2 && header_hash_1 && header_hash_2) {
            const vastloom::inference::DeepseekV4GgufCheckpointSpec checkpoint_spec{
                "test/deepseek4-gguf-checkpoint",
                "fixture-revision",
                {
                    {"part-1.gguf", full.size(), full_hash_1.value(),
                     "headers/part-1.gguf.header", header.size(), header_hash_1.value()},
                    {"part-2.gguf", continuation.size(), full_hash_2.value(),
                     "headers/part-2.gguf.header", continuation_header.size(),
                     header_hash_2.value()},
                },
                1U,
                2U,
                1U,
            };
            {
                std::ofstream manifest(full_import / "vastloom-import.json");
                manifest << "{\"format\":\"vastloom.hf-import.v1\","
                            "\"model_id\":\"test/deepseek4-gguf-checkpoint\","
                            "\"revision\":\"fixture-revision\",\"metadata_only\":false,"
                            "\"files\":[{\"name\":\"part-1.gguf\",\"size\":"
                         << full.size() << ",\"sha256\":\"" << full_hash_1.value()
                         << "\"},{\"name\":\"part-2.gguf\",\"size\":"
                         << continuation.size() << ",\"sha256\":\"" << full_hash_2.value()
                         << "\"}]}";
            }
            {
                std::ofstream manifest(metadata_import / "vastloom-import.json");
                manifest << "{\"format\":\"vastloom.hf-import.v1\","
                            "\"model_id\":\"test/deepseek4-gguf-checkpoint\","
                            "\"revision\":\"fixture-revision\",\"metadata_only\":true,"
                            "\"files\":[{\"name\":\"headers/part-1.gguf.header\",\"size\":"
                         << header.size() << ",\"sha256\":\"" << header_hash_1.value()
                         << "\"},{\"name\":\"headers/part-2.gguf.header\",\"size\":"
                         << continuation_header.size() << ",\"sha256\":\""
                         << header_hash_2.value() << "\"}]}";
            }

            vastloom::inference::DeepseekV4GgufCheckpointInspectOptions options;
            options.context_tokens = 128U;
            options.safe_ram_cap_bytes = 80U * vastloom::MiB;
            options.safe_vram_cap_bytes = 102U;
            auto bound = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
                full_import, checkpoint_spec, options);
            CHECK(bound && bound.value().imported_manifest_verified &&
                  bound.value().imported_file_hashes_verified &&
                  bound.value().full_payload_present &&
                  bound.value().executable_source_binding_verified &&
                  !bound.value().catalog.metadata_only &&
                  bound.value().expert_fits_vram_tier);
            if (bound) {
                CHECK(bound.value().catalog.experts.front().read_ranges.front().shard.parent_path() ==
                      full_import);
            }

            options.verify_hashes = false;
            auto quick = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
                full_import, checkpoint_spec, options);
            CHECK(quick && quick.value().full_payload_present &&
                  !quick.value().imported_file_hashes_verified &&
                  !quick.value().executable_source_binding_verified &&
                  quick.value().catalog.metadata_only);

            options.verify_hashes = true;
            options.safe_ram_cap_bytes = 67'110'360U;
            options.safe_vram_cap_bytes = 60U;
            auto rejected = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
                full_import, checkpoint_spec, options);
            CHECK(!rejected && rejected.error().code == vastloom::ErrorCode::resource_exhausted);

            options.safe_ram_cap_bytes = 80U * vastloom::MiB;
            options.safe_vram_cap_bytes = 102U;
            auto metadata = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
                metadata_import, checkpoint_spec, options);
            CHECK(metadata && metadata.value().imported_manifest_verified &&
                  metadata.value().imported_file_hashes_verified &&
                  !metadata.value().full_payload_present &&
                  !metadata.value().executable_source_binding_verified &&
                  metadata.value().catalog.metadata_only);
            if (metadata) {
                const std::array<std::uint32_t, 1U> selected{0U};
                auto probe = vastloom::inference::build_deepseek_v4_layer_probe_plan(
                    metadata.value(), 0U, selected, 4U, 8U);
                CHECK(probe && probe.value().imported_file_hashes_verified &&
                      !probe.value().full_payload_present &&
                      !probe.value().executable_source_binding_verified &&
                      probe.value().trunk_ranges.front().source.shard ==
                          metadata_import / "part-2.gguf" &&
                      probe.value().experts.front().read_ranges.front().shard ==
                          metadata_import / "part-1.gguf");
            }

            auto mismatched_spec = checkpoint_spec;
            mismatched_spec.shards.front().full_sha256 = std::string(64U, '0');
            CHECK(!vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
                full_import, mismatched_spec, options));

            const auto missing_import = directory / "checkpoint-missing-shard";
            std::filesystem::create_directories(missing_import);
            {
                std::ofstream manifest(missing_import / "vastloom-import.json");
                manifest << "{\"format\":\"vastloom.hf-import.v1\","
                            "\"model_id\":\"test/deepseek4-gguf-checkpoint\","
                            "\"revision\":\"fixture-revision\",\"metadata_only\":false,"
                            "\"files\":[{\"name\":\"part-1.gguf\",\"size\":"
                         << full.size() << ",\"sha256\":\"" << full_hash_1.value()
                         << "\"},{\"name\":\"part-2.gguf\",\"size\":"
                         << continuation.size() << ",\"sha256\":\"" << full_hash_2.value()
                         << "\"}]}";
            }
            CHECK(!vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
                missing_import, checkpoint_spec, options));

            auto corrupted = full;
            corrupted.back() ^= 1U;
            write_bytes(full_part_1, corrupted);
            CHECK(!vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
                full_import, checkpoint_spec, options));
        }
    }
}

void test_trunk_storage_reference_kernels() {
    const auto write_u16_le = [](std::span<std::byte> bytes,
                                 std::size_t offset,
                                 std::uint16_t value) {
        bytes[offset] = static_cast<std::byte>(value & 0xffU);
        bytes[offset + 1U] = static_cast<std::byte>(value >> 8U);
    };
    const auto write_u32_le = [](std::span<std::byte> bytes,
                                 std::size_t offset,
                                 std::uint32_t value) {
        for (unsigned index = 0U; index < 4U; ++index) {
            bytes[offset + index] = static_cast<std::byte>(
                (value >> (index * 8U)) & 0xffU);
        }
    };
    const auto fingerprint = [](std::span<const float> values) {
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (const float value : values) {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
            for (unsigned shift = 0U; shift < 32U; shift += 8U) {
                hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
                hash *= UINT64_C(1099511628211);
            }
        }
        return hash;
    };
    const auto make_input = [](std::size_t columns) {
        std::vector<float> input(columns);
        for (std::size_t column = 0U; column < columns; ++column) {
            const int centered = static_cast<int>(column % 29U) - 14;
            input[column] = static_cast<float>(centered) * 0.03125F +
                            static_cast<float>(column % 3U) * 0.0078125F;
        }
        return input;
    };
    const auto byte_fingerprint = [](std::span<const std::byte> values) {
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (const std::byte value : values) {
            hash ^= std::to_integer<std::uint8_t>(value);
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    };
    std::vector<float> q8_source(3U * 17U);
    for (std::size_t row = 0U; row < 3U; ++row) {
        for (std::size_t column = 0U; column < 17U; ++column) {
            const int centered = static_cast<int>((row * 19U + column * 7U) % 31U) - 15;
            q8_source[row * 17U + column] =
                static_cast<float>(centered) * 0.017578125F +
                static_cast<float>(column % 5U) * 0.00390625F;
        }
    }
    auto q8_encoded = vastloom::inference::quantize_q8_0_reference(
        q8_source, 3U, 17U, 32U);
    CHECK(q8_encoded && q8_encoded.value().size() ==
                            3U * vastloom::inference::kQ8_0BlockBytes);
    if (q8_encoded) {
        CHECK(byte_fingerprint(q8_encoded.value()) == UINT64_C(0x083a653165a92c21));
        std::vector<float> padded(3U * 32U);
        CHECK(vastloom::inference::dequantize_q8_0(q8_encoded.value(), padded));
        for (std::size_t row = 0U; row < 3U; ++row) {
            CHECK(std::all_of(
                padded.begin() + static_cast<std::ptrdiff_t>(row * 32U + 17U),
                padded.begin() + static_cast<std::ptrdiff_t>((row + 1U) * 32U),
                [](float value) { return value == 0.0F; }));
        }
    }
    CHECK(!vastloom::inference::quantize_q8_0_reference(
        q8_source, 3U, 17U, 31U));
    auto non_finite_source = q8_source;
    non_finite_source[3] = std::numeric_limits<float>::infinity();
    CHECK(!vastloom::inference::quantize_q8_0_reference(
        non_finite_source, 3U, 17U, 32U));
    auto unrepresentable_scale = q8_source;
    unrepresentable_scale[3] = std::numeric_limits<float>::max();
    CHECK(!vastloom::inference::quantize_q8_0_reference(
        unrepresentable_scale, 3U, 17U, 32U));
    vastloom::inference::QuantizedKernelLimits encoder_limit;
    encoder_limit.max_elements = 95U;
    CHECK(!vastloom::inference::quantize_q8_0_reference(
        q8_source, 3U, 17U, 32U, encoder_limit));
    constexpr std::array<std::uint16_t, 6U> scales{
        0x3c00U, 0xb800U, 0x3555U, 0x4000U, 0x0400U, 0x0001U};
    constexpr std::array<std::uint16_t, 6U> minimum_scales{
        0x3800U, 0xbc00U, 0x3000U, 0x3c00U, 0x0001U, 0x0400U};

    std::vector<std::byte> q8(scales.size() * vastloom::inference::kQ8_0BlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kQ8_0BlockBytes;
        write_u16_le(q8, base, scales[block]);
        for (std::size_t index = 0U;
             index < vastloom::inference::kQ8_0BlockElements; ++index) {
            q8[base + 2U + index] = static_cast<std::byte>(
                (block * 37U + index * 53U + 11U) & 0xffU);
        }
    }
    std::vector<float> q8_decoded(
        scales.size() * vastloom::inference::kQ8_0BlockElements);
    CHECK(vastloom::inference::dequantize_q8_0(q8, q8_decoded));
    CHECK(fingerprint(q8_decoded) == UINT64_C(0x25edd1a40129ee99));
    CHECK(q8_decoded[0] == 11.0F && q8_decoded[7] == 126.0F &&
          q8_decoded[8] == -77.0F && q8_decoded[31] == 118.0F &&
          q8_decoded[32] == -24.0F && q8_decoded[63] == 50.5F &&
          q8_decoded[127] == -54.0F);
    auto q8_input = make_input(64U);
    std::array<float, 3U> q8_output{};
    CHECK(vastloom::inference::matvec_q8_0_f32(
        q8, 3U, 64U, q8_input, q8_output));
    CHECK(q8_output[0] == -168.55859375F);
    CHECK(q8_output[1] == -2.8284435272216797F);
    CHECK(q8_output[2] == 0.0058367089368402958F);

    // Raw blocks and expected values are pinned to ggml commit
    // 30bf8685ed4eb0a47f2b06229543327749904150, not generated by Vastloom.
    std::vector<std::byte> q4(scales.size() * vastloom::inference::kQ4KBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kQ4KBlockBytes;
        write_u16_le(q4, base, scales[block]);
        write_u16_le(q4, base + 2U, minimum_scales[block]);
        for (std::size_t index = 0U; index < 12U; ++index) {
            q4[base + 4U + index] = static_cast<std::byte>(
                (block * 29U + index * 47U + 7U) & 0xffU);
        }
        for (std::size_t index = 0U; index < 128U; ++index) {
            q4[base + 16U + index] = static_cast<std::byte>(
                (block * 37U + index * 53U + 11U) & 0xffU);
        }
    }
    std::vector<float> q4_decoded(
        scales.size() * vastloom::inference::kQ4KBlockElements);
    CHECK(vastloom::inference::dequantize_q4_k(q4, q4_decoded));
    CHECK(fingerprint(q4_decoded) == UINT64_C(0x84e8e84336ea0ad7));
    CHECK(q4_decoded[0] == 75.5F && q4_decoded[7] == 96.5F &&
          q4_decoded[8] == 19.5F && q4_decoded[31] == 40.5F &&
          q4_decoded[32] == -25.0F && q4_decoded[63] == 353.0F &&
          q4_decoded[127] == 12.0F && q4_decoded[255] == 212.0F &&
          q4_decoded[256] == 32.0F && q4_decoded[511] == -125.5F);
    auto q4_input = make_input(512U);
    std::array<float, 3U> q4_output{};
    CHECK(vastloom::inference::matvec_q4_k_f32(
        q4, 3U, 512U, q4_input, q4_output));
    CHECK(q4_output[0] == 281.6796875F);
    CHECK(q4_output[1] == -792.12872314453125F);
    CHECK(q4_output[2] == 0.0096547976136207581F);
    std::array<float, 512U> q4_row{};
    CHECK(vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeQ4K, q4, 3U, 512U, 1U, q4_row));
    CHECK(std::equal(q4_row.begin(), q4_row.end(),
                     q4_decoded.begin() + 512));
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeQ4K, q4, 3U, 512U, 3U, q4_row));
    std::array<float, 511U> short_q4_row{};
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeQ4K, q4, 3U, 512U, 1U,
        short_q4_row));
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeI32, q4, 3U, 512U, 1U, q4_row));

    std::vector<std::byte> q5(scales.size() * vastloom::inference::kQ5KBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kQ5KBlockBytes;
        write_u16_le(q5, base, scales[block]);
        write_u16_le(q5, base + 2U, minimum_scales[block]);
        for (std::size_t index = 0U; index < 12U; ++index) {
            q5[base + 4U + index] = static_cast<std::byte>(
                (block * 29U + index * 47U + 7U) & 0xffU);
        }
        for (std::size_t index = 0U; index < 32U; ++index) {
            q5[base + 16U + index] = static_cast<std::byte>(
                (block * 41U + index * 31U + 13U) & 0xffU);
        }
        for (std::size_t index = 0U; index < 128U; ++index) {
            q5[base + 48U + index] = static_cast<std::byte>(
                (block * 37U + index * 53U + 11U) & 0xffU);
        }
    }
    std::vector<float> q5_decoded(
        scales.size() * vastloom::inference::kQ5KBlockElements);
    CHECK(vastloom::inference::dequantize_q5_k(q5, q5_decoded));
    CHECK(fingerprint(q5_decoded) == UINT64_C(0x3ec4a59589931118));
    CHECK(q5_decoded[0] == 187.5F && q5_decoded[7] == 96.5F &&
          q5_decoded[8] == 131.5F && q5_decoded[31] == 40.5F &&
          q5_decoded[32] == -25.0F && q5_decoded[63] == 1217.0F &&
          q5_decoded[127] == 332.0F && q5_decoded[255] == 916.0F &&
          q5_decoded[256] == 32.0F && q5_decoded[511] == -453.5F);
    auto q5_input = make_input(512U);
    std::array<float, 3U> q5_output{};
    CHECK(vastloom::inference::matvec_q5_k_f32(
        q5, 3U, 512U, q5_input, q5_output));
    CHECK(q5_output[0] == 668.2421875F);
    CHECK(q5_output[1] == 1631.9100341796875F);
    CHECK(q5_output[2] == -0.046117834746837616F);

    std::vector<std::byte> q6(scales.size() * vastloom::inference::kQ6KBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kQ6KBlockBytes;
        for (std::size_t index = 0U; index < 128U; ++index) {
            q6[base + index] = static_cast<std::byte>(
                (block * 37U + index * 53U + 11U) & 0xffU);
        }
        for (std::size_t index = 0U; index < 64U; ++index) {
            q6[base + 128U + index] = static_cast<std::byte>(
                (block * 41U + index * 31U + 13U) & 0xffU);
        }
        for (std::size_t index = 0U; index < 16U; ++index) {
            q6[base + 192U + index] = static_cast<std::byte>(
                (block * 59U + index * 43U + 17U) & 0xffU);
        }
        write_u16_le(q6, base + 208U, scales[block]);
    }
    std::vector<float> q6_decoded(
        scales.size() * vastloom::inference::kQ6KBlockElements);
    CHECK(vastloom::inference::dequantize_q6_k(q6, q6_decoded));
    CHECK(fingerprint(q6_decoded) == UINT64_C(0xcd39c3f1219afbc9));
    CHECK(q6_decoded[0] == -85.0F && q6_decoded[7] == 238.0F &&
          q6_decoded[8] == -221.0F && q6_decoded[31] == 360.0F &&
          q6_decoded[32] == 2781.0F && q6_decoded[63] == -2420.0F &&
          q6_decoded[127] == 1054.0F && q6_decoded[255] == -530.0F &&
          q6_decoded[511] == 540.5F);
    auto q6_input = make_input(512U);
    std::array<float, 3U> q6_output{};
    CHECK(vastloom::inference::matvec_q6_k_f32(
        q6, 3U, 512U, q6_input, q6_output));
    CHECK(q6_output[0] == -4684.79296875F);
    CHECK(q6_output[1] == 7494.287109375F);
    CHECK(q6_output[2] == 0.5950310230255127F);

    constexpr std::array<float, 8U> dense_values{
        1.0F, -2.0F, 0.5F, 4.0F, -1.0F, 3.0F, -0.25F, 2.0F};
    constexpr std::array<std::uint16_t, 8U> bf16_values{
        0x3f80U, 0xc000U, 0x3f00U, 0x4080U, 0xbf80U, 0x4040U, 0xbe80U, 0x4000U};
    std::vector<std::byte> bf16(bf16_values.size() * vastloom::inference::kBf16ElementBytes);
    std::vector<std::byte> f32(dense_values.size() * vastloom::inference::kF32ElementBytes);
    for (std::size_t index = 0U; index < dense_values.size(); ++index) {
        write_u16_le(bf16, index * 2U, bf16_values[index]);
        write_u32_le(f32, index * 4U, std::bit_cast<std::uint32_t>(dense_values[index]));
    }
    std::array<float, 8U> dense_decoded{};
    CHECK(vastloom::inference::dequantize_bf16(bf16, dense_decoded));
    CHECK(std::equal(dense_decoded.begin(), dense_decoded.end(), dense_values.begin()));
    std::fill(dense_decoded.begin(), dense_decoded.end(), 0.0F);
    CHECK(vastloom::inference::dequantize_f32_le(f32, dense_decoded));
    CHECK(std::equal(dense_decoded.begin(), dense_decoded.end(), dense_values.begin()));
    constexpr std::array<float, 4U> dense_input{0.25F, -0.5F, 2.0F, 1.0F};
    std::array<float, 2U> dense_output{};
    CHECK(vastloom::inference::matvec_bf16_f32(
        bf16, 2U, 4U, dense_input, dense_output));
    CHECK(dense_output[0] == 6.25F && dense_output[1] == -0.25F);
    CHECK(vastloom::inference::matvec_f32_le_f32(
        f32, 2U, 4U, dense_input, dense_output));
    CHECK(dense_output[0] == 6.25F && dense_output[1] == -0.25F);

    constexpr std::array<std::int32_t, 5U> i32_values{
        0, 1, -1, std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max()};
    std::vector<std::byte> i32(i32_values.size() * vastloom::inference::kI32ElementBytes);
    for (std::size_t index = 0U; index < i32_values.size(); ++index) {
        write_u32_le(i32, index * vastloom::inference::kI32ElementBytes,
                     std::bit_cast<std::uint32_t>(i32_values[index]));
    }
    std::array<std::int32_t, 5U> i32_decoded{};
    CHECK(vastloom::inference::decode_i32_le(i32, i32_decoded));
    CHECK(i32_decoded == i32_values);
    auto truncated_i32 = i32;
    truncated_i32.pop_back();
    CHECK(!vastloom::inference::decode_i32_le(truncated_i32, i32_decoded));

    for (const std::uint32_t type : {
             vastloom::inference::kGgmlTypeF32,
             vastloom::inference::kGgmlTypeQ8_0,
             vastloom::inference::kGgmlTypeQ4K,
             vastloom::inference::kGgmlTypeQ5K,
             vastloom::inference::kGgmlTypeQ6K,
             vastloom::inference::kGgmlTypeIq2Xxs,
             vastloom::inference::kGgmlTypeIq2S,
             vastloom::inference::kGgmlTypeIq3Xxs,
             vastloom::inference::kGgmlTypeIq1S,
             vastloom::inference::kGgmlTypeMxfp4,
             vastloom::inference::kGgmlTypeBf16}) {
        CHECK(vastloom::inference::supports_reference_matvec_ggml_type(type));
        CHECK(vastloom::inference::supports_reference_storage_ggml_type(type));
    }
    CHECK(!vastloom::inference::supports_reference_matvec_ggml_type(1024U));
    CHECK(!vastloom::inference::supports_reference_matvec_ggml_type(
        vastloom::inference::kGgmlTypeI32));
    CHECK(vastloom::inference::supports_reference_storage_ggml_type(
        vastloom::inference::kGgmlTypeI32));
    CHECK(!vastloom::inference::supports_reference_storage_ggml_type(1024U));
    auto q8_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeQ8_0, 3U, 64U);
    auto q5_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeQ5K, 3U, 512U);
    auto q4_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeQ4K, 2U, 512U);
    auto q6_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeQ6K, 3U, 512U);
    auto iq2_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeIq2Xxs, 2U, 256U);
    auto mxfp4_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeMxfp4, 2U, 64U);
    auto bf16_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeBf16, 2U, 4U);
    auto f32_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeF32, 2U, 4U);
    auto i32_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeI32, 1U, i32_values.size());
    auto iq1_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeIq1S, 2U, 256U);
    auto iq3_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeIq3Xxs, 2U, 256U);
    CHECK(q8_bytes && q8_bytes.value() == q8.size());
    CHECK(q5_bytes && q5_bytes.value() == q5.size());
    CHECK(q4_bytes && q4_bytes.value() ==
                          4U * vastloom::inference::kQ4KBlockBytes);
    CHECK(q6_bytes && q6_bytes.value() == q6.size());
    CHECK(iq2_bytes && iq2_bytes.value() ==
                           2U * vastloom::inference::kIq2XxsBlockBytes);
    CHECK(mxfp4_bytes && mxfp4_bytes.value() ==
                            4U * vastloom::inference::kMxfp4BlockBytes);
    CHECK(bf16_bytes && bf16_bytes.value() == bf16.size());
    CHECK(f32_bytes && f32_bytes.value() == f32.size());
    CHECK(i32_bytes && i32_bytes.value() == i32.size());
    CHECK(iq1_bytes && iq1_bytes.value() == 2U * vastloom::inference::kIq1SBlockBytes);
    CHECK(iq3_bytes && iq3_bytes.value() == 2U * vastloom::inference::kIq3XxsBlockBytes);
    CHECK(!vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeQ8_0, 1U, 31U));
    CHECK(!vastloom::inference::ggml_storage_bytes(1024U, 1U, 1U));

    std::array<float, 3U> scratch{};
    std::array<float, 3U> published{17.0F, 17.0F, 17.0F};
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeQ8_0, q8, 3U, 64U, q8_input,
        scratch, published));
    CHECK(published == q8_output);
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeQ4K, q4, 3U, 512U, q4_input,
        scratch, published));
    CHECK(published == q4_output);
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeQ5K, q5, 3U, 512U, q5_input,
        scratch, published));
    CHECK(published == q5_output);
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeQ6K, q6, 3U, 512U, q6_input,
        scratch, published));
    CHECK(published == q6_output);
    std::array<float, 2U> dense_scratch{};
    std::array<float, 2U> dense_published{17.0F, 17.0F};
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeBf16, bf16, 2U, 4U, dense_input,
        dense_scratch, dense_published));
    CHECK(dense_published[0] == 6.25F && dense_published[1] == -0.25F);
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeF32, f32, 2U, 4U, dense_input,
        dense_scratch, dense_published));
    CHECK(dense_published[0] == 6.25F && dense_published[1] == -0.25F);
    std::fill(published.begin(), published.end(), 17.0F);
    auto invalid_q8 = q8;
    write_u16_le(invalid_q8, 4U * vastloom::inference::kQ8_0BlockBytes, 0x7c00U);
    CHECK(!vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeQ8_0, invalid_q8, 3U, 64U, q8_input,
        scratch, published));
    CHECK(std::all_of(published.begin(), published.end(),
                      [](float value) { return value == 17.0F; }));
    CHECK(!vastloom::inference::matvec_ggml_f32_transactional(
        1024U, q8, 3U, 64U, q8_input, scratch, published));
    CHECK(!vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeQ8_0, q8, 3U, 64U, q8_input,
        published, published));

    auto truncated_q4 = q4;
    truncated_q4.pop_back();
    CHECK(!vastloom::inference::dequantize_q4_k(truncated_q4, q4_decoded));
    auto unchanged_q4 = q4_decoded;
    auto invalid_q4 = q4;
    write_u16_le(invalid_q4,
                 5U * vastloom::inference::kQ4KBlockBytes + 2U, 0x7c00U);
    CHECK(!vastloom::inference::dequantize_q4_k(invalid_q4, q4_decoded));
    CHECK(q4_decoded == unchanged_q4);
    const auto unchanged_q4_row = q4_row;
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeQ4K, invalid_q4, 3U, 512U, 2U,
        q4_row));
    CHECK(q4_row == unchanged_q4_row);
    std::fill(published.begin(), published.end(), 17.0F);
    CHECK(!vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeQ4K, invalid_q4, 3U, 512U, q4_input,
        scratch, published));
    CHECK(std::all_of(published.begin(), published.end(),
                      [](float value) { return value == 17.0F; }));
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeQ4K, truncated_q4, 3U, 512U, 0U,
        q4_row));

    auto truncated = q5;
    truncated.pop_back();
    CHECK(!vastloom::inference::dequantize_q5_k(truncated, q5_decoded));
    auto unchanged = q5_decoded;
    auto invalid_q5 = q5;
    write_u16_le(invalid_q5, 5U * vastloom::inference::kQ5KBlockBytes + 2U, 0x7c00U);
    CHECK(!vastloom::inference::dequantize_q5_k(invalid_q5, q5_decoded));
    CHECK(q5_decoded == unchanged);
    auto invalid_q6 = q6;
    write_u16_le(invalid_q6, 5U * vastloom::inference::kQ6KBlockBytes + 208U, 0x7c00U);
    unchanged = q6_decoded;
    CHECK(!vastloom::inference::dequantize_q6_k(invalid_q6, q6_decoded));
    CHECK(q6_decoded == unchanged);
    vastloom::inference::QuantizedKernelLimits small_limits;
    small_limits.max_elements = 255U;
    CHECK(!vastloom::inference::dequantize_q4_k(q4, q4_decoded, small_limits));
    CHECK(!vastloom::inference::dequantize_q5_k(q5, q5_decoded, small_limits));
    small_limits.max_elements = 4U;
    CHECK(!vastloom::inference::decode_i32_le(i32, i32_decoded, small_limits));
    CHECK(!vastloom::inference::matvec_q8_0_f32(
        q8, 3U, 63U, q8_input, q8_output));
    q8_input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::matvec_q8_0_f32(
        q8, 3U, 64U, q8_input, q8_output));

    auto invalid_bf16 = bf16;
    write_u16_le(invalid_bf16, 6U, 0x7f80U);
    std::fill(dense_decoded.begin(), dense_decoded.end(), 17.0F);
    CHECK(!vastloom::inference::dequantize_bf16(invalid_bf16, dense_decoded));
    CHECK(std::all_of(dense_decoded.begin(), dense_decoded.end(),
                      [](float value) { return value == 17.0F; }));
    auto invalid_f32 = f32;
    write_u32_le(invalid_f32, 24U, 0x7f800000U);
    CHECK(!vastloom::inference::dequantize_f32_le(invalid_f32, dense_decoded));
    CHECK(std::all_of(dense_decoded.begin(), dense_decoded.end(),
                      [](float value) { return value == 17.0F; }));
}

void test_iq1_s_reference_kernel() {
    constexpr std::array<std::uint16_t, 6U> scales{
        0x3c00U, 0xb800U, 0x3555U, 0x4000U, 0x0400U, 0x0001U};
    std::vector<std::byte> encoded(scales.size() * vastloom::inference::kIq1SBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kIq1SBlockBytes;
        encoded[base] = static_cast<std::byte>(scales[block] & 0xffU);
        encoded[base + 1U] = static_cast<std::byte>(scales[block] >> 8U);
        for (std::size_t index = 0U; index < 32U; ++index) {
            encoded[base + 2U + index] = static_cast<std::byte>(
                (block * 37U + index * 53U + 11U) & 0xffU);
        }
        for (std::size_t group = 0U; group < 8U; ++group) {
            const auto high = static_cast<std::uint16_t>(
                (((group * 5U + block) & 7U) << 12U) |
                (((group + block) & 1U) != 0U ? 0x8000U : 0U) |
                ((block * 733U + group * 97U + 0x123U) & 0x0fffU));
            encoded[base + 34U + group * 2U] = static_cast<std::byte>(high & 0xffU);
            encoded[base + 35U + group * 2U] = static_cast<std::byte>(high >> 8U);
        }
    }

    std::vector<float> decoded(scales.size() * vastloom::inference::kIq1SBlockElements);
    auto status = vastloom::inference::dequantize_iq1_s(encoded, decoded);
    CHECK(status);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : decoded) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    CHECK(hash == UINT64_C(0x9b66de452b7f3c8d));
    CHECK(decoded[0] == 0.125F && decoded[7] == 0.125F && decoded[8] == -0.875F &&
          decoded[31] == -0.875F && decoded[32] == 9.625F && decoded[255] == -7.875F &&
          decoded[256] == 1.6875F && decoded[511] == -0.5625F);

    std::vector<float> input(512U);
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const int centered = static_cast<int>(column % 29U) - 14;
        input[column] = static_cast<float>(centered) * 0.03125F +
                        static_cast<float>(column % 3U) * 0.0078125F;
    }
    std::array<float, 3U> output{};
    auto multiplied = vastloom::inference::matvec_iq1_s_f32(
        encoded, 3U, 512U, input, output);
    CHECK(multiplied);
    CHECK(output[0] == -4.8994140625F);
    CHECK(std::abs(output[1] - 22.142812728881836F) <= 1.0e-6F);
    CHECK(std::abs(output[2] - (-0.0021024104207754135F)) <= 1.0e-9F);

    auto truncated = encoded;
    truncated.pop_back();
    CHECK(!vastloom::inference::dequantize_iq1_s(truncated, decoded));
    CHECK(!vastloom::inference::dequantize_iq1_s(encoded, std::span<float>{}));
    vastloom::inference::QuantizedKernelLimits small_limits;
    small_limits.max_elements = 256U;
    CHECK(!vastloom::inference::dequantize_iq1_s(encoded, decoded, small_limits));
    CHECK(!vastloom::inference::matvec_iq1_s_f32(encoded, 3U, 255U, input, output));
    CHECK(!vastloom::inference::matvec_iq1_s_f32(truncated, 3U, 512U, input, output));

    auto invalid_scale = encoded;
    invalid_scale[0] = std::byte{0x00U};
    invalid_scale[1] = std::byte{0x7cU};
    auto unchanged = decoded;
    CHECK(!vastloom::inference::dequantize_iq1_s(invalid_scale, decoded));
    CHECK(decoded == unchanged);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::matvec_iq1_s_f32(encoded, 3U, 512U, input, output));
}

void test_iq2_xxs_reference_kernel() {
    constexpr std::array<std::uint16_t, 6U> scales{
        0x3c00U, 0xb800U, 0x3555U, 0x4000U, 0x0400U, 0x0001U};
    std::vector<std::byte> encoded(
        scales.size() * vastloom::inference::kIq2XxsBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kIq2XxsBlockBytes;
        encoded[base] = static_cast<std::byte>(scales[block] & 0xffU);
        encoded[base + 1U] = static_cast<std::byte>(scales[block] >> 8U);
        for (std::size_t index = 0U; index < 32U; ++index) {
            const auto packed = static_cast<std::uint16_t>(
                (block * 733U + index * 977U + 0x1234U) & 0xffffU);
            encoded[base + 2U + index * 2U] =
                static_cast<std::byte>(packed & 0xffU);
            encoded[base + 3U + index * 2U] =
                static_cast<std::byte>(packed >> 8U);
        }
    }

    std::vector<float> decoded(scales.size() * vastloom::inference::kIq2XxsBlockElements);
    auto status = vastloom::inference::dequantize_iq2_xxs(encoded, decoded);
    CHECK(status);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : decoded) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    CHECK(hash == UINT64_C(0x9bca8e03f48dafe9));
    CHECK(decoded[0] == 9.375F && decoded[7] == 3.0F && decoded[8] == -16.125F &&
          decoded[31] == -3.0F && decoded[32] == 5.0F && decoded[63] == -5.0F &&
          decoded[127] == -9.0F && decoded[255] == 17.0F &&
          decoded[256] == 2.5F && decoded[511] == 8.5F);

    std::vector<float> input(512U);
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const int centered = static_cast<int>(column % 29U) - 14;
        input[column] = static_cast<float>(centered) * 0.03125F +
                        static_cast<float>(column % 3U) * 0.0078125F;
    }
    std::array<float, 3U> output{};
    auto multiplied = vastloom::inference::matvec_iq2_xxs_f32(
        encoded, 3U, 512U, input, output);
    CHECK(multiplied);
    CHECK(output[0] == -32.86767578125F);
    CHECK(output[1] == -284.89828491210938F);
    CHECK(std::abs(output[2] - (-0.011499286629259586F)) <= 1.0e-9F);

    std::vector<float> row(512U);
    CHECK(vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeIq2Xxs, encoded, 3U, 512U, 1U, row));
    CHECK(std::equal(row.begin(), row.end(), decoded.begin() + 512));

    std::array<float, 3U> scratch{};
    std::array<float, 3U> published{17.0F, 17.0F, 17.0F};
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeIq2Xxs, encoded, 3U, 512U, input,
        scratch, published));
    CHECK(published == output);

    auto truncated = encoded;
    truncated.pop_back();
    CHECK(!vastloom::inference::dequantize_iq2_xxs(truncated, decoded));
    CHECK(!vastloom::inference::dequantize_iq2_xxs(encoded, std::span<float>{}));
    vastloom::inference::QuantizedKernelLimits small_limits;
    small_limits.max_elements = 256U;
    CHECK(!vastloom::inference::dequantize_iq2_xxs(encoded, decoded, small_limits));
    CHECK(!vastloom::inference::matvec_iq2_xxs_f32(encoded, 3U, 255U, input, output));
    CHECK(!vastloom::inference::matvec_iq2_xxs_f32(truncated, 3U, 512U, input, output));
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeIq2Xxs, encoded, 3U, 512U, 3U, row));

    auto invalid_scale = encoded;
    invalid_scale[4U * vastloom::inference::kIq2XxsBlockBytes] = std::byte{0x00U};
    invalid_scale[4U * vastloom::inference::kIq2XxsBlockBytes + 1U] = std::byte{0x7cU};
    auto unchanged = decoded;
    CHECK(!vastloom::inference::dequantize_iq2_xxs(invalid_scale, decoded));
    CHECK(decoded == unchanged);
    const auto unchanged_row = row;
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeIq2Xxs, invalid_scale, 3U, 512U, 2U, row));
    CHECK(row == unchanged_row);
    std::fill(published.begin(), published.end(), 17.0F);
    CHECK(!vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeIq2Xxs, invalid_scale, 3U, 512U, input,
        scratch, published));
    CHECK(std::all_of(published.begin(), published.end(),
                      [](float value) { return value == 17.0F; }));
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::matvec_iq2_xxs_f32(
        encoded, 3U, 512U, input, output));
}

// Checked against ggml's own dequantize_row_iq2_s, run under WSL on these exact bytes, rather than
// against this implementation's own output. IQ2_S is not a variant of IQ2_XXS: 82 bytes rather than
// 66, a 1,024-entry grid addressed by ten bits of which two come from a per-group high-bit byte,
// two four-bit scales per group of 32, and explicit sign bytes instead of a seven-bit index with a
// parity bit.
// A request arriving over a socket decides how much work this process does, so the parser is
// checked for what it refuses at least as hard as for what it accepts.
void test_openai_protocol() {
    const std::string body =
        R"({"model":"deepseek-v4-flash","messages":[{"role":"system","content":"Sei conciso."},)"
        R"({"role":"user","content":"Qual e la capitale d'Italia?"}],"max_tokens":24,)"
        R"("stream":true,"temperature":0,"stop":["\n\n","FINE"]})";
    auto parsed = vastloom::api::parse_chat_completion_request(body);
    CHECK(parsed);
    CHECK(parsed.value().model == "deepseek-v4-flash");
    CHECK(parsed.value().messages.size() == 2U);
    CHECK(parsed.value().messages[0].role == "system");
    CHECK(parsed.value().messages[1].content == "Qual e la capitale d'Italia?");
    CHECK(parsed.value().max_tokens == 24U);
    CHECK(parsed.value().stream);
    CHECK(parsed.value().temperature.has_value() && *parsed.value().temperature == 0.0);
    CHECK(parsed.value().stop.size() == 2U && parsed.value().stop[1] == "FINE");

    // The newer field wins when a client sends both, which some send during migration.
    auto both = vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"x"}],)"
        R"("max_tokens":8,"max_completion_tokens":16})");
    CHECK(both && both.value().max_tokens == 16U);

    // A single string is as legal as an array for `stop`, and clients use both.
    auto one_stop = vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"x"}],"stop":"###"})");
    CHECK(one_stop && one_stop.value().stop.size() == 1U && one_stop.value().stop[0] == "###");

    // Absent optionals keep their bounded defaults rather than becoming unbounded.
    auto minimal = vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    CHECK(minimal && minimal.value().max_tokens == 128U && !minimal.value().stream);
    CHECK(!minimal.value().temperature.has_value());

    // Refusals. Each of these would otherwise become work this process performs on request.
    CHECK(!vastloom::api::parse_chat_completion_request(R"({"messages":[]})"));
    CHECK(!vastloom::api::parse_chat_completion_request(R"({"model":"m","messages":[]})"));
    CHECK(!vastloom::api::parse_chat_completion_request(R"({"model":"m"})"));
    CHECK(!vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user"}]})"));
    CHECK(!vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"content":"x"}]})"));
    CHECK(!vastloom::api::parse_chat_completion_request("not json"));
    CHECK(!vastloom::api::parse_chat_completion_request(R"(["array","root"])"));
    // Out of bounds is refused, not clamped: a clamped request is one the caller believes it made.
    CHECK(!vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":1000000})"));
    CHECK(!vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":0})"));
    vastloom::api::ChatCompletionLimits tight;
    tight.max_messages = 1U;
    CHECK(!vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]})",
        tight));
    tight = {};
    tight.max_body_bytes = 16U;
    CHECK(!vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"x"}]})", tight));
    tight = {};
    tight.max_stop_sequences = 1U;
    CHECK(!vastloom::api::parse_chat_completion_request(
        R"({"model":"m","messages":[{"role":"user","content":"x"}],"stop":["a","b"]})", tight));

    vastloom::api::ChatCompletionResponse response;
    response.id = "chatcmpl-1";
    response.model = "deepseek-v4-flash";
    response.content = "La capitale d'Italia è \"Roma\".\nDavvero.";
    response.finish_reason = "stop";
    response.created = 1'700'000'000U;
    response.prompt_tokens = 12U;
    response.completion_tokens = 10U;
    auto serialized = vastloom::api::serialize_chat_completion_response(response);
    CHECK(serialized);
    // The payload must survive a round trip through the project's own parser, which is what any
    // client will do to it. Quotes and newlines in generated text are the usual way that breaks.
    auto reparsed = vastloom::util::parse_json(serialized.value());
    CHECK(reparsed);
    {
        auto root = reparsed.value().as_object();
        CHECK(root);
        const auto* choices = reparsed.value().find("choices");
        CHECK(choices != nullptr);
        auto array = choices->as_array();
        CHECK(array && array.value()->size() == 1U);
        const auto* message = (*array.value())[0].find("message");
        CHECK(message != nullptr);
        const auto* content = message->find("content");
        CHECK(content != nullptr);
        auto text = content->as_string();
        CHECK(text && text.value() == response.content);
        const auto* usage = reparsed.value().find("usage");
        CHECK(usage != nullptr);
        const auto* total = usage->find("total_tokens");
        CHECK(total != nullptr);
        auto total_value = total->as_u64();
        CHECK(total_value && total_value.value() == 22U);
    }

    auto first_chunk = vastloom::api::serialize_chat_completion_chunk(response, "La", true, false);
    CHECK(first_chunk && vastloom::util::parse_json(first_chunk.value()));
    CHECK(first_chunk.value().find("\"role\":\"assistant\"") != std::string::npos);
    CHECK(first_chunk.value().find("\"finish_reason\":null") != std::string::npos);
    auto last_chunk = vastloom::api::serialize_chat_completion_chunk(response, "", false, true);
    CHECK(last_chunk && vastloom::util::parse_json(last_chunk.value()));
    CHECK(last_chunk.value().find("\"finish_reason\":\"stop\"") != std::string::npos);

    auto listing = vastloom::api::serialize_model_list("deepseek-v4-flash", 1'700'000'000U);
    CHECK(listing && vastloom::util::parse_json(listing.value()));

    auto error_body = vastloom::api::serialize_error("no room", "insufficient_resources", 503);
    CHECK(vastloom::util::parse_json(error_body));

    // A Governor refusal is a 503, not a 500: the request was fine and the host was not able.
    CHECK(vastloom::api::http_status_for(vastloom::ErrorCode::resource_exhausted) == 503);
    CHECK(vastloom::api::http_status_for(vastloom::ErrorCode::invalid_argument) == 400);
    CHECK(vastloom::api::http_status_for(vastloom::ErrorCode::not_found) == 404);
    CHECK(vastloom::api::http_status_for(vastloom::ErrorCode::unsupported) == 501);
    CHECK(vastloom::api::http_status_for(vastloom::ErrorCode::internal_error) == 500);
}

// Q4_K and MXFP4 were the two storage types with no standalone parity gate: their correctness had
// rested on the end-to-end result, which is the thing now known to be wrong. Q4_K in particular
// carries the LM head, where a decoding defect would misplace individual vocabulary rows.
//
// Checked against ggml's own dequantize_row_q4_K and dequantize_row_mxfp4, run under WSL on these
// exact bytes.
void test_q4k_mxfp4_reference_parity() {
    constexpr std::array<std::uint16_t, 6U> scales{
        0x3c00U, 0xb800U, 0x3555U, 0x4000U, 0x0400U, 0x0001U};
    constexpr std::array<std::uint16_t, 6U> minimums{
        0x3800U, 0x3555U, 0xbc00U, 0x0001U, 0x4200U, 0x0400U};
    std::vector<std::byte> q4k(scales.size() * vastloom::inference::kQ4KBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kQ4KBlockBytes;
        q4k[base] = static_cast<std::byte>(scales[block] & 0xffU);
        q4k[base + 1U] = static_cast<std::byte>(scales[block] >> 8U);
        q4k[base + 2U] = static_cast<std::byte>(minimums[block] & 0xffU);
        q4k[base + 3U] = static_cast<std::byte>(minimums[block] >> 8U);
        for (std::size_t index = 0U; index < 12U; ++index) {
            q4k[base + 4U + index] =
                static_cast<std::byte>((block * 37U + index * 53U + 0x2bU) & 0xffU);
        }
        for (std::size_t index = 0U; index < 128U; ++index) {
            q4k[base + 16U + index] =
                static_cast<std::byte>((block * 733U + index * 977U + 0x1234U) & 0xffU);
        }
    }
    std::vector<float> decoded(scales.size() * vastloom::inference::kQ4KBlockElements);
    CHECK(vastloom::inference::dequantize_q4_k(q4k, decoded));
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : decoded) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    CHECK(hash == UINT64_C(0x4a61f0f8094737cc));
    CHECK(decoded[0] == 140.5F && decoded[1] == 183.5F && decoded[31] == 97.5F &&
          decoded[32] == 70.0F && decoded[127] == 85.0F && decoded[255] == 680.5F);
    CHECK(std::abs(decoded[256] - (-19.9970703F)) <= 1.0e-4F);
    CHECK(std::abs(decoded[511] - (-348.995361F)) <= 1.0e-3F);
    CHECK(std::abs(decoded[1024] - (-56.9692383F)) <= 1.0e-4F);

    std::vector<std::byte> mxfp4(scales.size() * vastloom::inference::kMxfp4BlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kMxfp4BlockBytes;
        mxfp4[base] = static_cast<std::byte>(120U + block * 3U);
        for (std::size_t index = 0U; index < 16U; ++index) {
            mxfp4[base + 1U + index] =
                static_cast<std::byte>((block * 91U + index * 167U + 0x5aU) & 0xffU);
        }
    }
    std::vector<float> mx(scales.size() * vastloom::inference::kMxfp4BlockElements);
    CHECK(vastloom::inference::dequantize_mxfp4(mxfp4, mx));
    std::uint64_t mx_hash = UINT64_C(14695981039346656037);
    for (const float value : mx) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            mx_hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            mx_hash *= UINT64_C(1099511628211);
        }
    }
    CHECK(mx_hash == UINT64_C(0xce2b9aa19ef0e448));
    CHECK(mx[0] == -0.0078125F && mx[1] == 0.00390625F && mx[2] == 0.0F &&
          mx[3] == -0.046875F && mx[4] == 0.03125F);
}

void test_iq2_s_reference_kernel() {
    // Scales vary in exponent and mantissa, including a subnormal, and the input below is
    // non-dyadic. A gate built from dyadic values cannot separate f32 from f64 accumulation and has
    // already let two wrong kernels through.
    constexpr std::array<std::uint16_t, 6U> scales{
        0x3c00U, 0xb800U, 0x3555U, 0x4000U, 0x0400U, 0x0001U};
    std::vector<std::byte> encoded(
        scales.size() * vastloom::inference::kIq2SBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kIq2SBlockBytes;
        encoded[base] = static_cast<std::byte>(scales[block] & 0xffU);
        encoded[base + 1U] = static_cast<std::byte>(scales[block] >> 8U);
        for (std::size_t index = 0U; index < 32U; ++index) {
            encoded[base + 2U + index] = static_cast<std::byte>(
                (block * 733U + index * 977U + 0x1234U) & 0xffU);
        }
        for (std::size_t index = 0U; index < 32U; ++index) {
            encoded[base + 34U + index] = static_cast<std::byte>(
                (block * 251U + index * 131U + 0x5aa5U) & 0xffU);
        }
        for (std::size_t index = 0U; index < 8U; ++index) {
            encoded[base + 66U + index] = static_cast<std::byte>(
                (block * 97U + index * 43U + 0x3cU) & 0xffU);
        }
        for (std::size_t index = 0U; index < 8U; ++index) {
            encoded[base + 74U + index] = static_cast<std::byte>(
                (block * 61U + index * 29U + 0x17U) & 0xffU);
        }
    }

    std::vector<float> decoded(scales.size() * vastloom::inference::kIq2SBlockElements);
    auto status = vastloom::inference::dequantize_iq2_s(encoded, decoded);
    CHECK(status);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : decoded) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    CHECK(hash == UINT64_C(0x785968d455de0695));
    CHECK(decoded[0] == -15.0F && decoded[7] == -15.0F && decoded[8] == 15.0F &&
          decoded[31] == 3.0F && decoded[32] == -9.0F && decoded[63] == 21.875F &&
          decoded[127] == 40.625F && decoded[255] == -29.0F &&
          decoded[256] == -4.5F && decoded[511] == -8.0625F);

    std::vector<float> input(512U);
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const int centered = static_cast<int>(column % 29U) - 14;
        input[column] = static_cast<float>(centered) * 0.03125F +
                        static_cast<float>(column % 3U) * 0.0078125F;
    }
    std::array<float, 3U> output{};
    auto multiplied = vastloom::inference::matvec_iq2_s_f32(
        encoded, 3U, 512U, input, output);
    CHECK(multiplied);
    CHECK(output[0] == 209.540527F);
    CHECK(output[1] == 274.050537F);
    CHECK(std::abs(output[2] - 0.0192997903F) <= 1.0e-9F);

    std::vector<float> row(512U);
    CHECK(vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeIq2S, encoded, 3U, 512U, 1U, row));
    CHECK(std::equal(row.begin(), row.end(), decoded.begin() + 512));

    std::array<float, 3U> scratch{};
    std::array<float, 3U> published{17.0F, 17.0F, 17.0F};
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeIq2S, encoded, 3U, 512U, input,
        scratch, published));
    CHECK(published == output);

    auto truncated = encoded;
    truncated.pop_back();
    CHECK(!vastloom::inference::dequantize_iq2_s(truncated, decoded));
    CHECK(!vastloom::inference::dequantize_iq2_s(encoded, std::span<float>{}));
    vastloom::inference::QuantizedKernelLimits small_limits;
    small_limits.max_elements = 256U;
    CHECK(!vastloom::inference::dequantize_iq2_s(encoded, decoded, small_limits));
    CHECK(!vastloom::inference::matvec_iq2_s_f32(encoded, 3U, 255U, input, output));
    CHECK(!vastloom::inference::matvec_iq2_s_f32(truncated, 3U, 512U, input, output));
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeIq2S, encoded, 3U, 512U, 3U, row));

    auto invalid_scale = encoded;
    invalid_scale[4U * vastloom::inference::kIq2SBlockBytes] = std::byte{0x00U};
    invalid_scale[4U * vastloom::inference::kIq2SBlockBytes + 1U] = std::byte{0x7cU};
    auto unchanged = decoded;
    CHECK(!vastloom::inference::dequantize_iq2_s(invalid_scale, decoded));
    CHECK(decoded == unchanged);
    std::fill(published.begin(), published.end(), 17.0F);
    CHECK(!vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeIq2S, invalid_scale, 3U, 512U, input,
        scratch, published));
    CHECK(std::all_of(published.begin(), published.end(),
                      [](float value) { return value == 17.0F; }));
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::matvec_iq2_s_f32(encoded, 3U, 512U, input, output));

    CHECK(vastloom::inference::iq2_s_grid_table().size() == 1'024U);
    auto iq2s_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeIq2S, 2U, 256U);
    CHECK(iq2s_bytes &&
          iq2s_bytes.value() == 2U * vastloom::inference::kIq2SBlockBytes);
}

void test_iq3_xxs_reference_kernel() {
    constexpr std::array<std::uint16_t, 6U> scales{
        0x3c00U, 0xb800U, 0x3555U, 0x4000U, 0x0400U, 0x0001U};
    std::vector<std::byte> encoded(scales.size() * vastloom::inference::kIq3XxsBlockBytes);
    for (std::size_t block = 0U; block < scales.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kIq3XxsBlockBytes;
        encoded[base] = static_cast<std::byte>(scales[block] & 0xffU);
        encoded[base + 1U] = static_cast<std::byte>(scales[block] >> 8U);
        for (std::size_t index = 0U; index < 64U; ++index) {
            encoded[base + 2U + index] = static_cast<std::byte>(
                (block * 37U + index * 53U + 11U) & 0xffU);
        }
        for (std::size_t group = 0U; group < 8U; ++group) {
            std::uint32_t auxiliary = static_cast<std::uint32_t>(
                (block * 3U + group * 5U) & 15U) << 28U;
            for (std::size_t subblock = 0U; subblock < 4U; ++subblock) {
                const auto sign_index = static_cast<std::uint32_t>(
                    (block * 19U + group * 13U + subblock * 23U + 7U) & 127U);
                auxiliary |= sign_index << (7U * subblock);
            }
            for (std::size_t byte = 0U; byte < 4U; ++byte) {
                encoded[base + 66U + group * 4U + byte] = static_cast<std::byte>(
                    (auxiliary >> (8U * byte)) & 0xffU);
            }
        }
    }

    std::vector<float> decoded(scales.size() * vastloom::inference::kIq3XxsBlockElements);
    auto status = vastloom::inference::dequantize_iq3_xxs(encoded, decoded);
    CHECK(status);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : decoded) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    CHECK(hash == UINT64_C(0xf36989dcc58b1664));
    CHECK(decoded[0] == -11.0F && decoded[7] == -3.0F && decoded[8] == 5.0F &&
          decoded[31] == -5.0F && decoded[32] == 170.5F && decoded[255] == 7.0F &&
          decoded[256] == -54.25F && decoded[511] == -19.5F);

    std::vector<float> input(512U);
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const int centered = static_cast<int>(column % 29U) - 14;
        input[column] = static_cast<float>(centered) * 0.03125F +
                        static_cast<float>(column % 3U) * 0.0078125F;
    }
    std::array<float, 3U> output{};
    auto multiplied = vastloom::inference::matvec_iq3_xxs_f32(
        encoded, 3U, 512U, input, output);
    CHECK(multiplied);
    CHECK(output[0] == -347.998046875F);
    CHECK(std::abs(output[1] - 2887.319580078125F) <= 1.0e-4F);
    CHECK(std::abs(output[2] - 0.064146831631660461F) <= 1.0e-8F);

    auto truncated = encoded;
    truncated.pop_back();
    CHECK(!vastloom::inference::dequantize_iq3_xxs(truncated, decoded));
    CHECK(!vastloom::inference::dequantize_iq3_xxs(encoded, std::span<float>{}));
    vastloom::inference::QuantizedKernelLimits small_limits;
    small_limits.max_elements = 256U;
    CHECK(!vastloom::inference::dequantize_iq3_xxs(encoded, decoded, small_limits));
    CHECK(!vastloom::inference::matvec_iq3_xxs_f32(encoded, 3U, 255U, input, output));
    CHECK(!vastloom::inference::matvec_iq3_xxs_f32(truncated, 3U, 512U, input, output));

    auto invalid_scale = encoded;
    invalid_scale[0] = std::byte{0x00U};
    invalid_scale[1] = std::byte{0x7cU};
    auto unchanged = decoded;
    CHECK(!vastloom::inference::dequantize_iq3_xxs(invalid_scale, decoded));
    CHECK(decoded == unchanged);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::matvec_iq3_xxs_f32(encoded, 3U, 512U, input, output));
}

void test_mxfp4_reference_kernel() {
    constexpr std::array<std::uint8_t, 6U> exponents{
        128U, 127U, 0U, 1U, 120U, 136U};
    std::vector<std::byte> encoded(
        exponents.size() * vastloom::inference::kMxfp4BlockBytes);
    for (std::size_t block = 0U; block < exponents.size(); ++block) {
        const std::size_t base = block * vastloom::inference::kMxfp4BlockBytes;
        encoded[base] = static_cast<std::byte>(exponents[block]);
        for (std::size_t index = 0U;
             index < vastloom::inference::kMxfp4BlockElements / 2U; ++index) {
            encoded[base + 1U + index] = static_cast<std::byte>(
                (block * 37U + index * 53U + 11U) & 0xffU);
        }
    }

    std::vector<float> decoded(
        exponents.size() * vastloom::inference::kMxfp4BlockElements);
    auto status = vastloom::inference::dequantize_mxfp4(encoded, decoded);
    CHECK(status);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : decoded) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    CHECK(hash == UINT64_C(0x53a1edd1080feb82));
    CHECK(decoded[0] == -3.0F && decoded[7] == -8.0F &&
          decoded[15] == 8.0F && decoded[16] == 0.0F &&
          decoded[31] == 2.0F && decoded[32] == 0.0F &&
          decoded[63] == 2.0F &&
          std::bit_cast<std::uint32_t>(decoded[64]) == UINT32_C(0x00c00000) &&
          std::bit_cast<std::uint32_t>(decoded[95]) == UINT32_C(0x01400000) &&
          std::bit_cast<std::uint32_t>(decoded[96]) == UINT32_C(0x80800000) &&
          std::bit_cast<std::uint32_t>(decoded[127]) == UINT32_C(0x80400000) &&
          decoded[160] == 1024.0F && decoded[191] == -1536.0F);

    std::vector<float> input(64U);
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const int centered = static_cast<int>(column % 29U) - 14;
        input[column] = static_cast<float>(centered) * 0.03125F +
                        static_cast<float>(column % 3U) * 0.0078125F;
    }
    std::array<float, 3U> output{};
    auto multiplied = vastloom::inference::matvec_mxfp4_f32(
        encoded, 3U, 64U, input, output);
    CHECK(multiplied);
    CHECK(output[0] == 13.18359375F);
    CHECK(std::bit_cast<std::uint32_t>(output[1]) == UINT32_C(0x01372000));
    CHECK(output[2] == -213.97833251953125F);

    std::vector<float> row(64U);
    CHECK(vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeMxfp4, encoded, 3U, 64U, 1U, row));
    CHECK(std::equal(row.begin(), row.end(), decoded.begin() + 64));

    std::array<float, 3U> scratch{};
    std::array<float, 3U> published{17.0F, 17.0F, 17.0F};
    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeMxfp4, encoded, 3U, 64U, input,
        scratch, published));
    CHECK(published == output);

    auto truncated = encoded;
    truncated.pop_back();
    CHECK(!vastloom::inference::dequantize_mxfp4(truncated, decoded));
    CHECK(!vastloom::inference::dequantize_mxfp4(encoded, std::span<float>{}));
    vastloom::inference::QuantizedKernelLimits small_limits;
    small_limits.max_elements = 32U;
    CHECK(!vastloom::inference::dequantize_mxfp4(encoded, decoded, small_limits));
    CHECK(!vastloom::inference::matvec_mxfp4_f32(encoded, 3U, 31U, input, output));
    CHECK(!vastloom::inference::matvec_mxfp4_f32(truncated, 3U, 64U, input, output));
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeMxfp4, encoded, 3U, 64U, 3U, row));

    auto overflowing = encoded;
    overflowing[4U * vastloom::inference::kMxfp4BlockBytes] = std::byte{0xfdU};
    overflowing[4U * vastloom::inference::kMxfp4BlockBytes + 1U] = std::byte{0x77U};
    auto unchanged = decoded;
    CHECK(!vastloom::inference::dequantize_mxfp4(overflowing, decoded));
    CHECK(decoded == unchanged);
    const auto unchanged_row = row;
    CHECK(!vastloom::inference::dequantize_ggml_row_f32(
        vastloom::inference::kGgmlTypeMxfp4, overflowing, 3U, 64U, 2U, row));
    CHECK(row == unchanged_row);
    std::fill(published.begin(), published.end(), 17.0F);
    CHECK(!vastloom::inference::matvec_ggml_f32_transactional(
        vastloom::inference::kGgmlTypeMxfp4, overflowing, 3U, 64U, input,
        scratch, published));
    CHECK(std::all_of(published.begin(), published.end(),
                      [](float value) { return value == 17.0F; }));
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::matvec_mxfp4_f32(
        encoded, 3U, 64U, input, output));
}

void test_quantized_routed_expert_contract() {
    constexpr std::size_t dimension = vastloom::inference::kIq1SBlockElements;
    std::vector<std::byte> gate(dimension * vastloom::inference::kIq1SBlockBytes);
    std::vector<std::byte> up(dimension * vastloom::inference::kIq1SBlockBytes);
    std::vector<std::byte> down(dimension * vastloom::inference::kIq3XxsBlockBytes);
    std::vector<float> input(dimension, 0.25F);
    std::vector<float> gate_scratch(dimension);
    std::vector<float> up_scratch(dimension);
    std::vector<float> down_scratch(dimension);
    std::vector<float> output(dimension, 17.0F);

    auto executed = vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
        gate, up, down, input, {gate_scratch, up_scratch, down_scratch}, output);
    CHECK(executed);
    CHECK(std::all_of(output.begin(), output.end(), [](float value) { return value == 0.0F; }));

    std::fill(output.begin(), output.end(), 17.0F);
    auto truncated_gate = gate;
    truncated_gate.pop_back();
    CHECK(!vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
        truncated_gate, up, down, input, {gate_scratch, up_scratch, down_scratch}, output));
    CHECK(std::all_of(output.begin(), output.end(), [](float value) { return value == 17.0F; }));

    auto invalid_down = down;
    invalid_down[0] = std::byte{0x00U};
    invalid_down[1] = std::byte{0x7cU};
    CHECK(!vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
        gate, up, invalid_down, input, {gate_scratch, up_scratch, down_scratch}, output));
    CHECK(std::all_of(output.begin(), output.end(), [](float value) { return value == 17.0F; }));

    CHECK(!vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
        gate, up, down, input, {output, up_scratch, down_scratch}, output));
    CHECK(!vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
        gate, up, down, input, {gate_scratch, gate_scratch, down_scratch}, output));

    vastloom::inference::QuantizedKernelLimits small_limits;
    small_limits.max_elements = dimension * dimension - 1U;
    CHECK(!vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
        gate, up, down, input, {gate_scratch, up_scratch, down_scratch}, output, small_limits));
    CHECK(std::all_of(output.begin(), output.end(), [](float value) { return value == 17.0F; }));

    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
        gate, up, down, input, {gate_scratch, up_scratch, down_scratch}, output));
    CHECK(std::all_of(output.begin(), output.end(), [](float value) { return value == 17.0F; }));
}

void test_logical_tensor_catalog() {
    using vastloom::model::LogicalTensorDescriptor;
    using vastloom::model::TensorDType;
    using vastloom::model::TensorRole;
    using vastloom::model::TensorSourceRange;

    vastloom::model::LogicalTensorCatalog catalog;
    catalog.model_id = "test/moe";
    catalog.revision = "revision";
    catalog.tensors = {
        LogicalTensorDescriptor{"embed", TensorDType::f32, {2U}, TensorRole::embedding,
                                std::nullopt, std::nullopt, false, TensorSourceRange{"shard-a", 0U, 8U}, std::nullopt},
        LogicalTensorDescriptor{"expert.weight", TensorDType::f32, {1U}, TensorRole::routed_expert,
                                2U, 7U, false, TensorSourceRange{"shard-a", 8U, 4U}, std::nullopt},
        LogicalTensorDescriptor{"expert.scale", TensorDType::f32, {1U}, TensorRole::routed_expert,
                                2U, 7U, false, TensorSourceRange{"shard-a", 12U, 4U}, std::nullopt},
        LogicalTensorDescriptor{"expert.bias", TensorDType::f16, {1U}, TensorRole::routed_expert,
                                2U, 7U, false, TensorSourceRange{"shard-b", 0U, 2U}, std::nullopt},
    };
    auto finalized = vastloom::model::finalize_logical_tensor_catalog(catalog);
    CHECK(finalized);
    CHECK(catalog.tensor_bytes == 18U && catalog.always_active_bytes == 8U &&
          catalog.routed_expert_bytes == 10U);
    CHECK(catalog.experts.size() == 1U && catalog.expert_read_range_count == 2U);
    if (catalog.experts.size() == 1U) {
        CHECK(catalog.experts[0].layer == 2U && catalog.experts[0].expert == 7U &&
              !catalog.experts[0].speculative && catalog.experts[0].bytes == 10U);
        CHECK(catalog.experts[0].tensor_indices.size() == 3U && catalog.experts[0].read_ranges.size() == 2U);
        CHECK(catalog.experts[0].tensor_slices.size() == 3U);
        if (catalog.experts[0].read_ranges.size() == 2U) {
            CHECK(catalog.experts[0].read_ranges[0].shard == "shard-a" &&
                  catalog.experts[0].read_ranges[0].offset == 8U &&
                  catalog.experts[0].read_ranges[0].bytes == 8U);
        }
        if (catalog.experts[0].tensor_slices.size() == 3U) {
            CHECK(catalog.experts[0].tensor_slices[0].tensor_index == 1U &&
                  catalog.experts[0].tensor_slices[0].packed_offset == 0U &&
                  catalog.experts[0].tensor_slices[1].packed_offset == 4U &&
                  catalog.experts[0].tensor_slices[2].packed_offset == 8U);
        }
    }
    CHECK(vastloom::model::to_string(TensorRole::routed_expert) == "routed_expert");
    auto residencies = vastloom::model::make_expert_residencies(catalog);
    CHECK(residencies && residencies.value().size() == 1U);
    if (residencies && residencies.value().size() == 1U) {
        auto residency = residencies.value()[0].snapshot();
        CHECK(residency && residency.value().state == vastloom::model::ResidencyState::nvme &&
              residency.value().object_bytes == 10U);
    }
    CHECK(!vastloom::model::make_expert_residencies(catalog, 0U));

    auto duplicate = catalog;
    duplicate.tensors.push_back(LogicalTensorDescriptor{
        "embed", TensorDType::f32, {1U}, TensorRole::other,
        std::nullopt, std::nullopt, false, TensorSourceRange{"shard-a", 20U, 4U}, std::nullopt});
    CHECK(!vastloom::model::finalize_logical_tensor_catalog(duplicate));

    auto missing_identity = catalog;
    missing_identity.tensors[1].expert.reset();
    CHECK(!vastloom::model::finalize_logical_tensor_catalog(missing_identity));

    auto overlapping = catalog;
    overlapping.tensors[3].source = TensorSourceRange{"shard-a", 10U, 2U};
    CHECK(!vastloom::model::finalize_logical_tensor_catalog(overlapping));

    auto overflowing = catalog;
    overflowing.tensors[0].source.offset = std::numeric_limits<vastloom::ByteCount>::max();
    CHECK(!vastloom::model::finalize_logical_tensor_catalog(overflowing));
    CHECK(!vastloom::model::finalize_logical_tensor_catalog(catalog, 2U, 1U));
}

void test_deepseek_v4_gguf_expert_bindings() {
    using vastloom::model::LogicalTensorDescriptor;
    using vastloom::model::TensorDType;
    using vastloom::model::TensorRole;
    using vastloom::model::TensorSourceRange;

    constexpr std::size_t hidden_width = 8U;
    constexpr std::size_t intermediate_width = 6U;
    constexpr std::size_t storage_columns =
        vastloom::inference::kQ8_0BlockElements;
    auto gate_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeQ8_0, intermediate_width,
        storage_columns);
    auto down_bytes = vastloom::inference::ggml_storage_bytes(
        vastloom::inference::kGgmlTypeQ8_0, hidden_width, storage_columns);
    CHECK(gate_bytes && down_bytes);
    if (!gate_bytes || !down_bytes) return;

    vastloom::model::LogicalTensorCatalog catalog;
    catalog.model_id = "test/deepseek-gguf-bindings";
    catalog.revision = "fixture";
    catalog.tensors = {
        LogicalTensorDescriptor{
            "blk.0.ffn_gate_exps.weight.expert.0", TensorDType::quantized,
            {storage_columns, intermediate_width}, TensorRole::routed_expert,
            0U, 0U, false, TensorSourceRange{"gate.bin", 0U, gate_bytes.value()},
            vastloom::inference::kGgmlTypeQ8_0},
        LogicalTensorDescriptor{
            "blk.0.ffn_up_exps.weight.expert.0", TensorDType::quantized,
            {storage_columns, intermediate_width}, TensorRole::routed_expert,
            0U, 0U, false, TensorSourceRange{"up.bin", 0U, gate_bytes.value()},
            vastloom::inference::kGgmlTypeQ8_0},
        LogicalTensorDescriptor{
            "blk.0.ffn_down_exps.weight.expert.0", TensorDType::quantized,
            {storage_columns, hidden_width}, TensorRole::routed_expert,
            0U, 0U, false, TensorSourceRange{"down.bin", 0U, down_bytes.value()},
            vastloom::inference::kGgmlTypeQ8_0},
    };
    CHECK(vastloom::model::finalize_logical_tensor_catalog(catalog));
    auto bindings = vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        catalog, 0U, 1U, hidden_width, intermediate_width);
    CHECK(bindings && bindings.value().size() == 1U);
    if (bindings && bindings.value().size() == 1U) {
        const auto& binding = bindings.value().front();
        CHECK(binding.expert_id == 0U && binding.bundle.bytes ==
                  2U * gate_bytes.value() + down_bytes.value());
        const auto& w1 = binding.matrices[static_cast<std::size_t>(
            vastloom::inference::DeepseekV4ExpertWeight::w1)];
        const auto& w2 = binding.matrices[static_cast<std::size_t>(
            vastloom::inference::DeepseekV4ExpertWeight::w2)];
        const auto& w3 = binding.matrices[static_cast<std::size_t>(
            vastloom::inference::DeepseekV4ExpertWeight::w3)];
        CHECK(w1.catalog_tensor_index == 0U &&
              w1.rows == intermediate_width &&
              w1.logical_columns == hidden_width &&
              w1.storage_columns == storage_columns);
        CHECK(w2.catalog_tensor_index == 2U && w2.rows == hidden_width &&
              w2.logical_columns == intermediate_width &&
              w2.storage_columns == storage_columns);
        CHECK(w3.catalog_tensor_index == 1U &&
              w3.rows == intermediate_width &&
              w3.logical_columns == hidden_width &&
              w3.storage_columns == storage_columns);
    }

    auto metadata_only = catalog;
    metadata_only.metadata_only = true;
    CHECK(!vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        metadata_only, 0U, 1U, hidden_width, intermediate_width));
    CHECK(!vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        catalog, 0U, 2U, hidden_width, intermediate_width));
    CHECK(!vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        catalog, 0U, 1U, hidden_width, intermediate_width, 0U));

    auto malformed_name = catalog;
    malformed_name.tensors[0].name = "blk.0.ffn_gate.weight.expert.0";
    CHECK(!vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        malformed_name, 0U, 1U, hidden_width, intermediate_width));
    auto malformed_shape = catalog;
    malformed_shape.tensors[2].shape[1] = hidden_width - 1U;
    CHECK(!vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        malformed_shape, 0U, 1U, hidden_width, intermediate_width));
    auto malformed_bytes = catalog;
    --malformed_bytes.tensors[1].source.bytes;
    CHECK(!vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        malformed_bytes, 0U, 1U, hidden_width, intermediate_width));
}

void test_residency_state_machine() {
    using vastloom::model::ResidencyState;
    CHECK(!vastloom::model::ResidencyObject::create("", 8U));
    CHECK(!vastloom::model::ResidencyObject::create("expert", 0U));

    vastloom::safety::ResourceLimits limits;
    limits.ram_hard_limit = 16U * vastloom::MiB;
    limits.vram_hard_limit = 16U * vastloom::MiB;
    limits.staging_hard_limit = 4U * vastloom::MiB;
    limits.scratch_hard_limit = 4U * vastloom::MiB;
    vastloom::safety::ResourceGovernor governor(limits);

    auto created = vastloom::model::ResidencyObject::create("test/layer/2/expert/7", 8U * vastloom::MiB);
    CHECK(created);
    if (!created) return;
    auto object = created.take_value();
    auto initial = object.snapshot();
    CHECK(initial && initial.value().state == ResidencyState::nvme && !initial.value().evictable);
    CHECK(!object.acquire_use());
    CHECK(!object.begin_read(governor, 0U));
    CHECK(!object.begin_read(governor, 9U * vastloom::MiB));
    CHECK(object.begin_read(governor, 4U * vastloom::MiB));
    CHECK(!object.begin_read(governor, 4U * vastloom::MiB));
    auto reading = object.snapshot();
    CHECK(reading && reading.value().state == ResidencyState::reading &&
          reading.value().ram_reserved_bytes == 8U * vastloom::MiB &&
          reading.value().staging_reserved_bytes == 4U * vastloom::MiB);
    CHECK(governor.snapshot().reserved[0] == 8U * vastloom::MiB &&
          governor.snapshot().reserved[2] == 4U * vastloom::MiB);
    CHECK(object.publish_ram_ready());
    auto ram_ready = object.snapshot();
    CHECK(ram_ready && ram_ready.value().state == ResidencyState::ram_ready &&
          ram_ready.value().staging_reserved_bytes == 0U && ram_ready.value().evictable);

    auto use = object.acquire_use();
    CHECK(use);
    CHECK(!object.evict_ram() && !object.begin_vram_copy(governor));
    if (use) use.value().release();
    CHECK(object.set_pinned(true));
    CHECK(!object.evict_ram());
    CHECK(object.set_pinned(false));

    CHECK(object.begin_vram_copy(governor));
    auto copying = object.snapshot();
    CHECK(copying && copying.value().state == ResidencyState::copying_to_vram &&
          copying.value().ram_reserved_bytes == 8U * vastloom::MiB &&
          copying.value().vram_reserved_bytes == 8U * vastloom::MiB);
    CHECK(object.publish_vram_ready(true));
    CHECK(object.evict_ram());
    auto vram_only = object.snapshot();
    CHECK(vram_only && vram_only.value().state == ResidencyState::vram_ready &&
          vram_only.value().ram_reserved_bytes == 0U &&
          vram_only.value().vram_reserved_bytes == 8U * vastloom::MiB);
    auto gpu_use = object.acquire_use();
    CHECK(gpu_use && !object.evict_vram() && !object.fail("must not free an active expert"));
    if (gpu_use) gpu_use.value().release();
    CHECK(object.evict_vram());
    CHECK(object.snapshot() && object.snapshot().value().state == ResidencyState::nvme);

    CHECK(object.fail("injected I/O failure"));
    auto failed = object.snapshot();
    CHECK(failed && failed.value().state == ResidencyState::failed &&
          failed.value().failure == "injected I/O failure" &&
          failed.value().ram_reserved_bytes == 0U && failed.value().vram_reserved_bytes == 0U);
    CHECK(!object.begin_read(governor, 4U * vastloom::MiB));
    CHECK(object.retry());
    CHECK(object.snapshot() && object.snapshot().value().state == ResidencyState::nvme);

    vastloom::safety::ResourceLimits small_limits = limits;
    small_limits.ram_hard_limit = 4U * vastloom::MiB;
    vastloom::safety::ResourceGovernor small_governor(small_limits);
    auto too_large = vastloom::model::ResidencyObject::create("too-large", 8U * vastloom::MiB);
    CHECK(too_large && !too_large.value().begin_read(small_governor, 4U * vastloom::MiB));
    CHECK(small_governor.snapshot().reserved[0] == 0U && small_governor.snapshot().reserved[2] == 0U);

    vastloom::safety::ResourceLimits staging_limits = limits;
    staging_limits.staging_hard_limit = 1U * vastloom::MiB;
    vastloom::safety::ResourceGovernor staging_governor(staging_limits);
    auto no_staging = vastloom::model::ResidencyObject::create("no-staging", 8U * vastloom::MiB);
    CHECK(no_staging && !no_staging.value().begin_read(staging_governor, 2U * vastloom::MiB));
    CHECK(staging_governor.snapshot().reserved[0] == 0U &&
          staging_governor.snapshot().reserved[2] == 0U);
    CHECK(no_staging && no_staging.value().snapshot() &&
          no_staging.value().snapshot().value().state == ResidencyState::nvme);

    vastloom::safety::ResourceLimits copy_limits = limits;
    copy_limits.vram_hard_limit = 4U * vastloom::MiB;
    vastloom::safety::ResourceGovernor copy_governor(copy_limits);
    auto no_vram = vastloom::model::ResidencyObject::create("no-vram", 8U * vastloom::MiB);
    CHECK(no_vram && no_vram.value().begin_read(copy_governor, 2U * vastloom::MiB) &&
          no_vram.value().publish_ram_ready());
    CHECK(no_vram && !no_vram.value().begin_vram_copy(copy_governor));
    CHECK(no_vram && no_vram.value().snapshot() &&
          no_vram.value().snapshot().value().state == ResidencyState::ram_ready &&
          no_vram.value().snapshot().value().ram_reserved_bytes == 8U * vastloom::MiB &&
          no_vram.value().snapshot().value().vram_reserved_bytes == 0U);
    CHECK(no_vram && no_vram.value().evict_ram());
    CHECK(copy_governor.snapshot().reserved[0] == 0U && copy_governor.snapshot().reserved[1] == 0U);

    auto concurrent = vastloom::model::ResidencyObject::create("concurrent", 4U * vastloom::MiB);
    CHECK(concurrent);
    if (concurrent) {
        auto first_handle = concurrent.value();
        auto second_handle = concurrent.value();
        std::array<int, 2> successes{};
        std::thread first_reader([&] {
            successes[0] = first_handle.begin_read(governor, 2U * vastloom::MiB) ? 1 : 0;
        });
        std::thread second_reader([&] {
            successes[1] = second_handle.begin_read(governor, 2U * vastloom::MiB) ? 1 : 0;
        });
        first_reader.join();
        second_reader.join();
        CHECK(successes[0] + successes[1] == 1);
        auto concurrent_snapshot = concurrent.value().snapshot();
        CHECK(concurrent_snapshot && concurrent_snapshot.value().state == ResidencyState::reading &&
              concurrent_snapshot.value().ram_reserved_bytes == 4U * vastloom::MiB &&
              concurrent_snapshot.value().staging_reserved_bytes == 2U * vastloom::MiB);
        CHECK(concurrent.value().fail("cancel concurrent read"));
    }
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);

    auto transferring = vastloom::model::ResidencyObject::create("exclusive-transfer", 2U * vastloom::MiB);
    CHECK(transferring && transferring.value().begin_read(governor, 1U * vastloom::MiB));
    if (transferring) {
        auto transfer = transferring.value().acquire_transfer();
        CHECK(transfer);
        auto snapshot = transferring.value().snapshot();
        CHECK(snapshot && snapshot.value().active_transfers == 1U &&
              !snapshot.value().evictable);
        CHECK(!transferring.value().acquire_transfer());
        CHECK(!transferring.value().fail("must not release an active transfer"));
        if (transfer) transfer.value().release();
        CHECK(transferring.value().fail("injected transfer failure"));
    }
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);

    vastloom::safety::ResourceGovernor pressure_governor(limits);
    CHECK(pressure_governor.update_pressure(
              {1U, 16U * vastloom::MiB, limits.minimum_storage_available * 4U}) ==
          vastloom::safety::SafetyState::emergency);
    auto pressure_object = vastloom::model::ResidencyObject::create("pressure", 2U * vastloom::MiB);
    CHECK(pressure_object && !pressure_object.value().begin_read(pressure_governor, 1U * vastloom::MiB));
    CHECK(pressure_object && pressure_object.value().snapshot() &&
          pressure_object.value().snapshot().value().state == ResidencyState::nvme);

    std::optional<vastloom::model::ResidencyUse> retained_use;
    {
        auto lifetime = vastloom::model::ResidencyObject::create("lease-lifetime", 2U * vastloom::MiB);
        CHECK(lifetime && lifetime.value().begin_read(governor, 1U * vastloom::MiB) &&
              lifetime.value().publish_ram_ready());
        if (lifetime) {
            auto acquired = lifetime.value().acquire_use();
            CHECK(acquired);
            if (acquired) retained_use.emplace(acquired.take_value());
        }
    }
    CHECK(governor.snapshot().reserved[0] == 2U * vastloom::MiB);
    retained_use.reset();
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[1] == 0U &&
          governor.snapshot().reserved[2] == 0U);
    CHECK(vastloom::model::to_string(ResidencyState::copying_to_vram) == "COPYING_TO_VRAM");
}

void test_packed_expert_loader(const std::filesystem::path& root) {
    const auto path = root / std::filesystem::path(std::u8string(u8"expert-àccènt.bin"));
    {
        std::ofstream output(path, std::ios::binary);
        for (unsigned value = 0U; value < 32U; ++value) output.put(static_cast<char>(value));
    }

    vastloom::model::ExpertBundle bundle;
    bundle.layer = 2U;
    bundle.expert = 7U;
    bundle.bytes = 10U;
    bundle.tensor_indices = {1U, 2U};
    bundle.read_ranges = {{path, 4U, 6U}, {path, 16U, 4U}};
    bundle.tensor_slices = {{1U, 0U, 6U}, {2U, 6U, 4U}};

    vastloom::safety::ResourceLimits limits;
    limits.ram_hard_limit = 1U * vastloom::MiB;
    limits.vram_hard_limit = 1U * vastloom::MiB;
    limits.staging_hard_limit = 1U * vastloom::MiB;
    limits.scratch_hard_limit = 1U * vastloom::MiB;
    vastloom::safety::ResourceGovernor governor(limits);
    auto residency = vastloom::model::ResidencyObject::create("test/layer/2/expert/7", bundle.bytes);
    CHECK(residency);
    if (!residency) return;

    vastloom::model::ExpertLoadOptions options;
    options.max_expert_bytes = 64U;
    options.max_range_bytes = 32U;
    options.max_ranges = 4U;
    options.measure_process_io = false;
    {
        auto loaded = vastloom::model::load_expert_bundle(bundle, residency.value(), governor, options);
        CHECK(loaded && loaded.value().bytes().size() == 10U);
        if (loaded && loaded.value().bytes().size() == 10U) {
            for (std::size_t index = 0U; index < 6U; ++index) {
                CHECK(std::to_integer<unsigned char>(loaded.value().bytes()[index]) ==
                      static_cast<unsigned char>(index + 4U));
            }
            for (std::size_t index = 0U; index < 4U; ++index) {
                CHECK(std::to_integer<unsigned char>(loaded.value().bytes()[index + 6U]) ==
                      static_cast<unsigned char>(index + 16U));
            }
            CHECK(loaded.value().telemetry().logical_bytes == 10U &&
                  loaded.value().telemetry().range_count == 2U &&
                  loaded.value().telemetry().elapsed_ms >= 0.0);
            auto first_tensor = loaded.value().tensor(1U);
            auto second_tensor = loaded.value().tensor(2U);
            CHECK(first_tensor && first_tensor.value().size() == 6U &&
                  std::to_integer<unsigned char>(first_tensor.value().front()) == 4U);
            CHECK(second_tensor && second_tensor.value().size() == 4U &&
                  std::to_integer<unsigned char>(second_tensor.value().front()) == 16U);
            CHECK(!loaded.value().tensor(999U));
        }
        auto ready = residency.value().snapshot();
        CHECK(ready && ready.value().state == vastloom::model::ResidencyState::ram_ready &&
              ready.value().ram_reserved_bytes == 10U && ready.value().staging_reserved_bytes == 0U &&
              ready.value().active_uses == 1U && !ready.value().evictable);
        CHECK(!residency.value().evict_ram());
    }
    CHECK(residency.value().snapshot() && residency.value().snapshot().value().active_uses == 0U);
    CHECK(residency.value().evict_ram());
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);

    auto limited_residency = vastloom::model::ResidencyObject::create("limited", bundle.bytes);
    auto limited_options = options;
    limited_options.max_expert_bytes = 9U;
    CHECK(limited_residency &&
          !vastloom::model::load_expert_bundle(bundle, limited_residency.value(), governor, limited_options));
    CHECK(limited_residency && limited_residency.value().snapshot() &&
          limited_residency.value().snapshot().value().state == vastloom::model::ResidencyState::nvme);

    auto invalid_bundle = bundle;
    invalid_bundle.read_ranges[1] = {path, 10U, 4U};
    auto invalid_residency = vastloom::model::ResidencyObject::create("invalid", invalid_bundle.bytes);
    CHECK(invalid_residency &&
          !vastloom::model::load_expert_bundle(invalid_bundle, invalid_residency.value(), governor, options));

    auto short_bundle = bundle;
    short_bundle.read_ranges = {{path, 30U, 10U}};
    auto short_residency = vastloom::model::ResidencyObject::create("short", short_bundle.bytes);
    CHECK(short_residency &&
          !vastloom::model::load_expert_bundle(short_bundle, short_residency.value(), governor, options));
    CHECK(short_residency && short_residency.value().snapshot() &&
          short_residency.value().snapshot().value().state == vastloom::model::ResidencyState::failed &&
          short_residency.value().snapshot().value().ram_reserved_bytes == 0U &&
          short_residency.value().snapshot().value().staging_reserved_bytes == 0U);

    auto cancelled_residency = vastloom::model::ResidencyObject::create("cancelled", bundle.bytes);
    auto cancelled_options = options;
    cancelled_options.cancelled = [] { return true; };
    CHECK(cancelled_residency &&
          !vastloom::model::load_expert_bundle(bundle, cancelled_residency.value(), governor, cancelled_options));
    CHECK(cancelled_residency && cancelled_residency.value().snapshot() &&
          cancelled_residency.value().snapshot().value().state == vastloom::model::ResidencyState::failed);
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);

    std::atomic<bool> worker_entered{false};
    std::atomic<bool> release_worker{false};
    auto asynchronous_options = options;
    asynchronous_options.cancelled = [&] {
        worker_entered.store(true, std::memory_order_release);
        while (!release_worker.load(std::memory_order_acquire)) std::this_thread::yield();
        return false;
    };
    vastloom::model::ExpertLoadPoolOptions pool_options;
    pool_options.workers = 1U;
    pool_options.max_queued_requests = 1U;
    pool_options.load = asynchronous_options;
    vastloom::model::BoundedExpertLoadPool pool(pool_options, governor);

    auto first_residency = vastloom::model::ResidencyObject::create("async-first", bundle.bytes);
    auto second_residency = vastloom::model::ResidencyObject::create("async-second", bundle.bytes);
    auto rejected_residency = vastloom::model::ResidencyObject::create("async-rejected", bundle.bytes);
    CHECK(first_residency && second_residency && rejected_residency);
    if (!first_residency || !second_residency || !rejected_residency) return;

    auto first_submission = pool.submit(bundle, first_residency.value(), "first");
    CHECK(first_submission);
    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!worker_entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::yield();
    }
    CHECK(worker_entered.load(std::memory_order_acquire));
    auto active = first_residency.value().snapshot();
    CHECK(active && active.value().state == vastloom::model::ResidencyState::reading &&
          active.value().active_transfers == 1U && active.value().ram_reserved_bytes == bundle.bytes);
    CHECK(!first_residency.value().fail("must not fail an active asynchronous read"));

    auto second_submission = pool.submit(bundle, second_residency.value(), "second");
    CHECK(second_submission && pool.pending_requests() == 1U);
    auto rejected_submission = pool.submit(bundle, rejected_residency.value(), "rejected");
    CHECK(!rejected_submission);
    auto rejected = rejected_residency.value().snapshot();
    CHECK(rejected && rejected.value().state == vastloom::model::ResidencyState::nvme &&
          rejected.value().ram_reserved_bytes == 0U && rejected.value().staging_reserved_bytes == 0U);

    release_worker.store(true, std::memory_order_release);
    if (first_submission && second_submission) {
        {
            auto first_loaded = first_submission.value().get();
            auto second_loaded = second_submission.value().get();
            CHECK(first_loaded && second_loaded);
            if (first_loaded) {
                const auto& telemetry = first_loaded.value().telemetry();
                CHECK(telemetry.request_sequence == 1U && telemetry.queue_ms >= 0.0 &&
                      telemetry.service_ms >= 0.0 && telemetry.total_ms >= telemetry.queue_ms &&
                      telemetry.total_ms >= telemetry.service_ms);
            }
            if (second_loaded) {
                const auto& telemetry = second_loaded.value().telemetry();
                CHECK(telemetry.request_sequence == 2U && telemetry.queue_ms >= 0.0 &&
                      telemetry.service_ms >= 0.0 && telemetry.total_ms >= telemetry.queue_ms &&
                      telemetry.total_ms >= telemetry.service_ms);
            }
            auto first_ready = first_residency.value().snapshot();
            auto second_ready = second_residency.value().snapshot();
            CHECK(first_ready && first_ready.value().active_transfers == 0U &&
                  first_ready.value().active_uses == 1U);
            CHECK(second_ready && second_ready.value().active_transfers == 0U &&
                  second_ready.value().active_uses == 1U);
        }
        CHECK(first_residency.value().evict_ram());
        CHECK(second_residency.value().evict_ram());
    }
    pool.shutdown();
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);

    std::atomic<bool> shutdown_worker_entered{false};
    std::atomic<bool> shutdown_release_worker{false};
    auto shutdown_options = options;
    shutdown_options.cancelled = [&] {
        shutdown_worker_entered.store(true, std::memory_order_release);
        while (!shutdown_release_worker.load(std::memory_order_acquire)) std::this_thread::yield();
        return false;
    };
    pool_options.load = shutdown_options;
    vastloom::model::BoundedExpertLoadPool shutdown_pool(pool_options, governor);
    auto running_residency = vastloom::model::ResidencyObject::create("shutdown-running", bundle.bytes);
    auto queued_residency = vastloom::model::ResidencyObject::create("shutdown-queued", bundle.bytes);
    CHECK(running_residency && queued_residency);
    if (!running_residency || !queued_residency) return;
    auto running_submission = shutdown_pool.submit(bundle, running_residency.value(), "running");
    CHECK(running_submission);
    const auto shutdown_wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!shutdown_worker_entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < shutdown_wait_deadline) {
        std::this_thread::yield();
    }
    CHECK(shutdown_worker_entered.load(std::memory_order_acquire));
    auto queued_submission = shutdown_pool.submit(bundle, queued_residency.value(), "queued");
    CHECK(queued_submission && shutdown_pool.pending_requests() == 1U);

    std::thread shutdown_thread([&] { shutdown_pool.shutdown(); });
    if (queued_submission) {
        CHECK(queued_submission.value().wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    }
    shutdown_release_worker.store(true, std::memory_order_release);
    shutdown_thread.join();
    if (queued_submission) {
        auto queued_result = queued_submission.value().get();
        CHECK(!queued_result);
    }
    auto queued_snapshot = queued_residency.value().snapshot();
    CHECK(queued_snapshot && queued_snapshot.value().state == vastloom::model::ResidencyState::failed &&
          queued_snapshot.value().ram_reserved_bytes == 0U &&
          queued_snapshot.value().staging_reserved_bytes == 0U);
    if (running_submission) {
        {
            auto running_result = running_submission.value().get();
            CHECK(running_result);
        }
        CHECK(running_residency.value().evict_ram());
    }
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);
}

void test_layer_streaming(const std::filesystem::path& root) {
    using vastloom::model::LayerTensorTile;
    using vastloom::model::LogicalTensorCatalog;
    using vastloom::model::LogicalTensorDescriptor;
    using vastloom::model::ResidencyObject;
    using vastloom::model::ResidencyState;
    using vastloom::model::TensorDType;
    using vastloom::model::TensorRole;
    using vastloom::model::TensorSourceRange;

    const auto path = root / std::filesystem::path(std::u8string(u8"trunk-àccènt.bin"));
    {
        std::ofstream output(path, std::ios::binary);
        for (unsigned value = 0U; value < 32U; ++value) output.put(static_cast<char>(value));
    }
    LogicalTensorCatalog catalog;
    catalog.model_id = "test/layer-streaming";
    catalog.revision = "fixture";
    catalog.tensors = {
        LogicalTensorDescriptor{"blk.3.attn.weight", TensorDType::quantized, {10U},
                                TensorRole::attention, 3U, std::nullopt, false,
                                TensorSourceRange{path, 0U, 10U}, 8U},
        LogicalTensorDescriptor{"blk.3.ffn_norm.weight", TensorDType::f32, {5U},
                                TensorRole::normalization, 3U, std::nullopt, false,
                                TensorSourceRange{path, 12U, 5U}, 0U},
        LogicalTensorDescriptor{"blk.3.ffn_gate_exps.weight.expert.0", TensorDType::quantized,
                                {4U}, TensorRole::routed_expert, 3U, 0U, false,
                                TensorSourceRange{path, 20U, 4U}, 19U},
        LogicalTensorDescriptor{"output.weight", TensorDType::f32, {4U},
                                TensorRole::output_head, std::nullopt, std::nullopt, false,
                                TensorSourceRange{path, 24U, 4U}, 0U},
    };
    CHECK(vastloom::model::finalize_logical_tensor_catalog(catalog));

    auto plan = vastloom::model::build_layer_streaming_plan(catalog, 3U, 4U);
    CHECK(plan && plan.value().tensor_count == 2U && plan.value().tiles.size() == 5U &&
          plan.value().trunk_tensor_bytes == 15U && plan.value().maximum_tensor_bytes == 10U &&
          plan.value().maximum_tile_bytes == 4U);
    if (!plan || plan.value().tiles.size() != 5U) return;
    CHECK(plan.value().tiles[0].tensor_index == 0U &&
          plan.value().tiles[0].tensor_offset == 0U &&
          plan.value().tiles[0].source.offset == 0U &&
          plan.value().tiles[0].source.bytes == 4U && !plan.value().tiles[0].final_tile);
    CHECK(plan.value().tiles[2].tensor_index == 0U &&
          plan.value().tiles[2].tensor_offset == 8U &&
          plan.value().tiles[2].source.bytes == 2U && plan.value().tiles[2].final_tile);
    CHECK(plan.value().tiles[4].tensor_index == 1U &&
          plan.value().tiles[4].tensor_offset == 4U &&
          plan.value().tiles[4].source.offset == 16U &&
          plan.value().tiles[4].source.bytes == 1U && plan.value().tiles[4].final_tile);
    CHECK(!vastloom::model::build_layer_streaming_plan(catalog, 2U, 4U));
    CHECK(!vastloom::model::build_layer_streaming_plan(catalog, 3U, 0U));
    CHECK(!vastloom::model::build_layer_streaming_plan(catalog, 3U, 4U, 1U, 10U));
    CHECK(!vastloom::model::build_layer_streaming_plan(catalog, 3U, 4U, 10U, 4U));

    vastloom::safety::ResourceLimits limits;
    limits.host_memory_hard_limit = 16U;
    limits.ram_hard_limit = 8U;
    limits.vram_hard_limit = 8U;
    limits.staging_hard_limit = 8U;
    limits.scratch_hard_limit = 8U;
    vastloom::safety::ResourceGovernor governor(limits);
    auto residency = ResidencyObject::create("test/layer/3/tensor/0/tile/0", 4U);
    CHECK(residency);
    if (!residency) return;
    vastloom::model::TensorTileLoadOptions load_options;
    load_options.max_tile_bytes = 4U;
    load_options.measure_process_io = false;
    {
        auto loaded = vastloom::model::load_tensor_tile(
            catalog, plan.value().tiles[0], residency.value(), governor, load_options);
        CHECK(loaded && loaded.value().bytes().size() == 4U &&
              loaded.value().identity().tensor_index == 0U &&
              loaded.value().telemetry().logical_bytes == 4U);
        if (loaded) {
            for (std::size_t index = 0U; index < loaded.value().bytes().size(); ++index) {
                CHECK(std::to_integer<unsigned char>(loaded.value().bytes()[index]) == index);
            }
        }
        auto ready = residency.value().snapshot();
        CHECK(ready && ready.value().state == ResidencyState::ram_ready &&
              ready.value().ram_reserved_bytes == 4U &&
              ready.value().staging_reserved_bytes == 0U && ready.value().active_uses == 1U);
    }
    CHECK(residency.value().snapshot() && residency.value().snapshot().value().active_uses == 0U);
    CHECK(residency.value().evict_ram());
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);

    auto metadata = catalog;
    metadata.metadata_only = true;
    auto metadata_residency = ResidencyObject::create("metadata", 4U);
    CHECK(metadata_residency &&
          !vastloom::model::load_tensor_tile(metadata, plan.value().tiles[0],
                                             metadata_residency.value(), governor, load_options));
    CHECK(metadata_residency && metadata_residency.value().snapshot() &&
          metadata_residency.value().snapshot().value().state == ResidencyState::nvme);

    auto tampered = plan.value().tiles[0];
    ++tampered.source.offset;
    auto tampered_residency = ResidencyObject::create("tampered", 4U);
    CHECK(tampered_residency &&
          !vastloom::model::load_tensor_tile(catalog, tampered, tampered_residency.value(),
                                             governor, load_options));

    auto bounded_options = load_options;
    bounded_options.max_tile_bytes = 3U;
    auto bounded_residency = ResidencyObject::create("tile-too-large", 4U);
    CHECK(bounded_residency &&
          !vastloom::model::load_tensor_tile(catalog, plan.value().tiles[0],
                                             bounded_residency.value(), governor,
                                             bounded_options));
    auto wrong_size_residency = ResidencyObject::create("wrong-tile-size", 3U);
    CHECK(wrong_size_residency &&
          !vastloom::model::load_tensor_tile(catalog, plan.value().tiles[0],
                                             wrong_size_residency.value(), governor,
                                             load_options));

    auto cancelled_residency = ResidencyObject::create("cancelled-tile", 4U);
    auto cancelled_options = load_options;
    cancelled_options.cancelled = [] { return true; };
    CHECK(cancelled_residency &&
          !vastloom::model::load_tensor_tile(catalog, plan.value().tiles[0],
                                             cancelled_residency.value(), governor,
                                             cancelled_options));
    CHECK(cancelled_residency && cancelled_residency.value().snapshot() &&
          cancelled_residency.value().snapshot().value().state == ResidencyState::nvme);

    unsigned cancellation_checks = 0U;
    auto late_cancel_options = load_options;
    late_cancel_options.cancelled = [&] { return ++cancellation_checks >= 2U; };
    auto late_cancel_residency = ResidencyObject::create("late-cancelled-tile", 4U);
    CHECK(late_cancel_residency &&
          !vastloom::model::load_tensor_tile(catalog, plan.value().tiles[0],
                                             late_cancel_residency.value(), governor,
                                             late_cancel_options));
    CHECK(late_cancel_residency && late_cancel_residency.value().snapshot() &&
          late_cancel_residency.value().snapshot().value().state == ResidencyState::failed &&
          late_cancel_residency.value().snapshot().value().ram_reserved_bytes == 0U &&
          late_cancel_residency.value().snapshot().value().staging_reserved_bytes == 0U);

    vastloom::safety::ResourceLimits small_limits = limits;
    small_limits.host_memory_hard_limit = 7U;
    vastloom::safety::ResourceGovernor small_governor(small_limits);
    auto refused_residency = ResidencyObject::create("budget-refused-tile", 4U);
    CHECK(refused_residency &&
          !vastloom::model::load_tensor_tile(catalog, plan.value().tiles[0],
                                             refused_residency.value(), small_governor,
                                             load_options));
    CHECK(small_governor.snapshot().reserved[0] == 0U &&
          small_governor.snapshot().reserved[2] == 0U);

    auto short_catalog = catalog;
    short_catalog.tensors[0].source = TensorSourceRange{path, 30U, 10U};
    auto short_plan = vastloom::model::build_layer_streaming_plan(short_catalog, 3U, 4U);
    CHECK(short_plan);
    if (short_plan) {
        auto short_residency = ResidencyObject::create("short-tile", 4U);
        CHECK(short_residency &&
              !vastloom::model::load_tensor_tile(short_catalog, short_plan.value().tiles[0],
                                                 short_residency.value(), governor, load_options));
        CHECK(short_residency && short_residency.value().snapshot() &&
              short_residency.value().snapshot().value().state == ResidencyState::failed &&
              short_residency.value().snapshot().value().ram_reserved_bytes == 0U &&
              short_residency.value().snapshot().value().staging_reserved_bytes == 0U);
    }
    CHECK(governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);
}

void test_imported_checkpoint_and_llama_config(const std::filesystem::path& root) {
    const auto imported_root = root / "imported-checkpoint";
    std::filesystem::create_directories(imported_root);
    {
        std::ofstream(imported_root / "data.bin", std::ios::binary) << "hello";
        std::ofstream manifest(imported_root / "vastloom-import.json");
        manifest << R"({"format":"vastloom.hf-import.v1","model_id":"test/model","revision":"abc","metadata_only":false,"files":[{"name":"data.bin","size":5,"sha256":"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}]})";
    }
    auto verified = vastloom::model::verify_imported_checkpoint(imported_root, true);
    CHECK(verified && verified.value().full_hash_verified && verified.value().total_bytes == 5U);
    // The first verification leaves a receipt keyed on each file's size and last-write time, so a
    // same-size edit is only caught if the filesystem moved the timestamp. NTFS usually does, but
    // not reliably enough to assert on: this test is about the hash gate rejecting changed bytes,
    // so the receipt is removed to make that the only thing under test. The receipt's own
    // behaviour - including that it is written at all - is asserted separately below.
    CHECK(std::filesystem::exists(imported_root / "vastloom-verified.json"));
    std::filesystem::remove(imported_root / "vastloom-verified.json");
    std::ofstream(imported_root / "data.bin", std::ios::binary | std::ios::trunc) << "HELLO";
    CHECK(!vastloom::model::verify_imported_checkpoint(imported_root, true));

    const auto legacy_imported_root = root / "legacy-imported-checkpoint";
    std::filesystem::create_directories(legacy_imported_root);
    std::ofstream(legacy_imported_root / "data.bin", std::ios::binary) << "hello";
    std::ofstream legacy_manifest(legacy_imported_root / "stargate-import.json");
    legacy_manifest << R"({"format":"stargate.hf-import.v1","model_id":"test/legacy","revision":"abc","metadata_only":false,"files":[{"name":"data.bin","size":5,"sha256":"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}]})";
    legacy_manifest.close();
    auto legacy_verified = vastloom::model::verify_imported_checkpoint(legacy_imported_root, true);
    CHECK(legacy_verified && legacy_verified.value().model_id == "test/legacy");
    std::ofstream(legacy_imported_root / "vastloom-import.json") << "{}";
    CHECK(!vastloom::model::verify_imported_checkpoint(legacy_imported_root, false));

    const auto config_path = root / "llama-config.json";
    {
        std::ofstream config(config_path);
        config << R"({"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_act":"silu","torch_dtype":"bfloat16","vocab_size":49152,"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,"max_position_embeddings":8192,"bos_token_id":1,"eos_token_id":2,"pad_token_id":2,"rms_norm_eps":0.00001,"rope_theta":100000,"tie_word_embeddings":true,"attention_bias":false,"mlp_bias":false,"rope_interleaved":false,"rope_scaling":null})";
    }
    auto llama = vastloom::inference::load_llama_config(config_path);
    CHECK(llama && llama.value().hidden_size == 576U && llama.value().num_key_value_heads == 3U);
    {
        std::ofstream config(config_path, std::ios::trunc);
        config << R"({"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_act":"silu","torch_dtype":"bfloat16"})";
    }
    CHECK(!vastloom::inference::load_llama_config(config_path));
}

void test_olmoe_config(const std::filesystem::path& root) {
    const auto path = root / "olmoe-config.json";
    {
        std::ofstream config(path);
        config << R"({"architectures":["OlmoeForCausalLM"],"attention_bias":false,"clip_qkv":null,"eos_token_id":50279,"hidden_act":"silu","hidden_size":2048,"intermediate_size":1024,"max_position_embeddings":4096,"model_type":"olmoe","norm_topk_prob":false,"num_attention_heads":16,"num_experts":64,"num_experts_per_tok":8,"num_hidden_layers":16,"num_key_value_heads":16,"pad_token_id":1,"rope_scaling":null,"rope_theta":10000.0,"tie_word_embeddings":false,"torch_dtype":"bfloat16","vocab_size":50304})";
    }
    auto config = vastloom::inference::load_olmoe_config(path);
    CHECK(config && config.value().num_experts == 64U && config.value().num_experts_per_token == 8U);
    {
        std::ofstream invalid(path, std::ios::trunc);
        invalid << R"({"architectures":["OlmoeForCausalLM"],"model_type":"olmoe","hidden_act":"silu","torch_dtype":"bfloat16"})";
    }
    CHECK(!vastloom::inference::load_olmoe_config(path));
    CHECK(vastloom::inference::parse_expert_cache_policy("no-cache") &&
          vastloom::inference::parse_expert_cache_policy("LFU") &&
          vastloom::inference::parse_expert_cache_policy("static-hot") &&
          !vastloom::inference::parse_expert_cache_policy("random"));
    CHECK(vastloom::inference::parse_expert_read_mode("whole") &&
          vastloom::inference::parse_expert_read_mode("staged") &&
          !vastloom::inference::parse_expert_read_mode("unbounded"));
    CHECK(vastloom::io::parse_io_cache_mode("buffered") &&
          vastloom::io::parse_io_cache_mode("DIRECT") &&
          !vastloom::io::parse_io_cache_mode("global-flush"));
}

void test_deepseek_v4_config(const std::filesystem::path& root) {
    const auto path = root / "deepseek-v4-config.json";
    {
        std::ofstream config(path);
        config << R"({"architectures":["DeepseekV4ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"bos_token_id":0,"compress_ratios":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0],"compress_rope_theta":160000,"eos_token_id":1,"expert_dtype":"fp4","hc_eps":0.000001,"hc_mult":4,"hc_sinkhorn_iters":20,"head_dim":512,"hidden_act":"silu","hidden_size":4096,"index_head_dim":128,"index_n_heads":64,"index_topk":512,"initializer_range":0.02,"max_position_embeddings":1048576,"model_type":"deepseek_v4","moe_intermediate_size":2048,"n_routed_experts":256,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,"num_experts_per_tok":6,"num_hash_layers":3,"num_hidden_layers":43,"num_key_value_heads":1,"num_nextn_predict_layers":1,"o_groups":8,"o_lora_rank":1024,"q_lora_rank":1024,"qk_rope_head_dim":64,"quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},"rms_norm_eps":0.000001,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},"rope_theta":10000,"routed_scaling_factor":1.5,"scoring_func":"sqrtsoftplus","sliding_window":128,"swiglu_limit":10.0,"tie_word_embeddings":false,"topk_method":"noaux_tc","torch_dtype":"bfloat16","transformers_version":"4.57.1","use_cache":true,"vocab_size":129280})";
    }
    auto loaded = vastloom::inference::load_deepseek_v4_config(path);
    CHECK(loaded && loaded.value().num_hidden_layers == 43U && loaded.value().num_routed_experts == 256U &&
          loaded.value().num_experts_per_token == 6U && loaded.value().max_position_embeddings == 1'048'576U);
    {
        std::ofstream config(path, std::ios::trunc);
        config << R"({"architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4"})";
    }
    CHECK(!vastloom::inference::load_deepseek_v4_config(path));
}

void test_deepseek_v4_reference_operators() {
    const auto path = std::filesystem::path(VASTLOOM_SOURCE_DIR) / "tests" / "fixtures" /
                      "deepseek_v4_operators.json";
    vastloom::util::JsonLimits limits;
    limits.max_input_bytes = 1U * vastloom::MiB;
    limits.max_nodes = 20'000U;
    limits.max_array_elements = 10'000U;
    auto fixture = vastloom::util::load_json_file(path, limits);
    CHECK(fixture && fixture.value().is_object());
    if (!fixture || !fixture.value().is_object()) return;
    const auto string_value = [&](const vastloom::util::JsonValue& object,
                                  std::string_view key) -> std::string {
        const auto* value = object.find(key);
        if (value == nullptr || !value->as_string()) {
            CHECK(false);
            return {};
        }
        return std::string(value->as_string().value());
    };
    const auto section = [&](std::string_view key) -> const vastloom::util::JsonValue* {
        const auto* value = fixture.value().find(key);
        CHECK(value != nullptr && value->is_object());
        return value != nullptr && value->is_object() ? value : nullptr;
    };
    const auto size_value = [&](const vastloom::util::JsonValue& object,
                                std::string_view key) -> std::size_t {
        const auto* value = object.find(key);
        auto parsed = value == nullptr
                          ? vastloom::Result<std::uint64_t>(vastloom::make_error(vastloom::ErrorCode::invalid_format, "missing fixture integer"))
                          : value->as_u64();
        CHECK(parsed && parsed.value() <= std::numeric_limits<std::size_t>::max());
        if (!parsed || parsed.value() > std::numeric_limits<std::size_t>::max()) return 0U;
        return static_cast<std::size_t>(parsed.value());
    };
    const auto float_value = [&](const vastloom::util::JsonValue& object,
                                 std::string_view key) -> float {
        const auto* value = object.find(key);
        auto parsed = value == nullptr
                          ? vastloom::Result<double>(vastloom::make_error(vastloom::ErrorCode::invalid_format, "missing fixture scalar"))
                          : value->as_double();
        CHECK(parsed && std::isfinite(parsed.value()));
        if (!parsed || !std::isfinite(parsed.value())) return 0.0F;
        return static_cast<float>(parsed.value());
    };
    const auto floats = [&](const vastloom::util::JsonValue& object,
                            std::string_view key) -> std::vector<float> {
        std::vector<float> result;
        const auto* value = object.find(key);
        auto array = value == nullptr
                         ? vastloom::Result<const vastloom::util::JsonValue::Array*>(vastloom::make_error(vastloom::ErrorCode::invalid_format, "missing fixture array"))
                         : value->as_array();
        CHECK(array);
        if (!array) return result;
        result.reserve(array.value()->size());
        for (const auto& item : *array.value()) {
            auto parsed = item.as_double();
            CHECK(parsed && std::isfinite(parsed.value()));
            if (!parsed) return {};
            result.push_back(static_cast<float>(parsed.value()));
        }
        return result;
    };
    const auto unsigneds = [&](const vastloom::util::JsonValue& object,
                               std::string_view key) -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> result;
        const auto* value = object.find(key);
        auto array = value == nullptr
                         ? vastloom::Result<const vastloom::util::JsonValue::Array*>(vastloom::make_error(vastloom::ErrorCode::invalid_format, "missing fixture array"))
                         : value->as_array();
        CHECK(array);
        if (!array) return result;
        result.reserve(array.value()->size());
        for (const auto& item : *array.value()) {
            auto parsed = item.as_u64();
            CHECK(parsed && parsed.value() <= std::numeric_limits<std::uint32_t>::max());
            if (!parsed || parsed.value() > std::numeric_limits<std::uint32_t>::max()) return {};
            result.push_back(static_cast<std::uint32_t>(parsed.value()));
        }
        return result;
    };
    const auto signed_indices = [&](const vastloom::util::JsonValue& object,
                                    std::string_view key) -> std::vector<std::int32_t> {
        std::vector<std::int32_t> result;
        const auto* value = object.find(key);
        auto array = value == nullptr
                         ? vastloom::Result<const vastloom::util::JsonValue::Array*>(vastloom::make_error(vastloom::ErrorCode::invalid_format, "missing fixture array"))
                         : value->as_array();
        CHECK(array);
        if (!array) return result;
        result.reserve(array.value()->size());
        for (const auto& item : *array.value()) {
            auto parsed = item.as_i64();
            CHECK(parsed && parsed.value() >= std::numeric_limits<std::int32_t>::min() &&
                  parsed.value() <= std::numeric_limits<std::int32_t>::max());
            if (!parsed || parsed.value() < std::numeric_limits<std::int32_t>::min() ||
                parsed.value() > std::numeric_limits<std::int32_t>::max()) {
                return {};
            }
            result.push_back(static_cast<std::int32_t>(parsed.value()));
        }
        return result;
    };
    const float tolerance = float_value(fixture.value(), "absolute_tolerance");
    const auto check_near = [&](const std::vector<float>& actual,
                                const std::vector<float>& expected) {
        CHECK(actual.size() == expected.size());
        if (actual.size() != expected.size()) return;
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            if (std::abs(actual[index] - expected[index]) > tolerance) {
                std::cerr << "DeepSeek operator mismatch at " << index << ": "
                          << actual[index] << " versus " << expected[index] << '\n';
                ++failures;
                return;
            }
        }
    };

    CHECK(string_value(fixture.value(), "format") == "vastloom.deepseek_v4.operator_fixture.v1");
    CHECK(string_value(fixture.value(), "official_revision") ==
          "60d8d70770c6776ff598c94bb586a859a38244f1");

    if (const auto* rms = section("rms_norm")) {
        auto result = vastloom::inference::deepseek_v4_rms_norm(
            floats(*rms, "input"), floats(*rms, "weight"), size_value(*rms, "rows"),
            size_value(*rms, "width"), float_value(*rms, "epsilon"));
        CHECK(result);
        if (result) check_near(result.value(), floats(*rms, "expected"));
        auto invalid = vastloom::inference::deepseek_v4_rms_norm({}, {}, 1U, 1U, 0.0F);
        CHECK(!invalid);
    }

    if (const auto* route = section("learned_route")) {
        auto result = vastloom::inference::deepseek_v4_route_learned(
            floats(*route, "input"), floats(*route, "gate_weight"),
            floats(*route, "selection_bias"), size_value(*route, "tokens"),
            size_value(*route, "width"), size_value(*route, "experts"),
            size_value(*route, "top_k"), float_value(*route, "route_scale"));
        CHECK(result);
        if (result) {
            CHECK(result.value().indices == unsigneds(*route, "expected_indices"));
            check_near(result.value().weights, floats(*route, "expected_weights"));
        }
        auto bias = floats(*route, "selection_bias");
        bias[0] = std::numeric_limits<float>::quiet_NaN();
        CHECK(!vastloom::inference::deepseek_v4_route_learned(
            floats(*route, "input"), floats(*route, "gate_weight"), bias,
            size_value(*route, "tokens"), size_value(*route, "width"),
            size_value(*route, "experts"), size_value(*route, "top_k"),
            float_value(*route, "route_scale")));
    }

    if (const auto* route = section("hash_route")) {
        auto result = vastloom::inference::deepseek_v4_route_hash(
            floats(*route, "input"), floats(*route, "gate_weight"),
            unsigneds(*route, "token_ids"), unsigneds(*route, "token_to_experts"),
            size_value(*route, "tokens"), size_value(*route, "width"),
            size_value(*route, "vocabulary"), size_value(*route, "experts"),
            size_value(*route, "top_k"), float_value(*route, "route_scale"));
        CHECK(result);
        if (result) {
            CHECK(result.value().indices == unsigneds(*route, "expected_indices"));
            check_near(result.value().weights, floats(*route, "expected_weights"));
        }
        auto ids = unsigneds(*route, "token_ids");
        ids[0] = static_cast<std::uint32_t>(size_value(*route, "vocabulary"));
        CHECK(!vastloom::inference::deepseek_v4_route_hash(
            floats(*route, "input"), floats(*route, "gate_weight"), ids,
            unsigneds(*route, "token_to_experts"), size_value(*route, "tokens"),
            size_value(*route, "width"), size_value(*route, "vocabulary"),
            size_value(*route, "experts"), size_value(*route, "top_k"),
            float_value(*route, "route_scale")));
    }

    if (const auto* expert = section("expert")) {
        auto result = vastloom::inference::deepseek_v4_expert_swiglu(
            floats(*expert, "input"), floats(*expert, "gate_weight"),
            floats(*expert, "down_weight"), floats(*expert, "up_weight"),
            floats(*expert, "routing_weights"), size_value(*expert, "tokens"),
            size_value(*expert, "width"), size_value(*expert, "intermediate"),
            float_value(*expert, "swiglu_limit"));
        CHECK(result);
        if (result) check_near(result.value(), floats(*expert, "expected"));
        CHECK(!vastloom::inference::deepseek_v4_expert_swiglu(
            floats(*expert, "input"), {}, floats(*expert, "down_weight"),
            floats(*expert, "up_weight"), floats(*expert, "routing_weights"),
            size_value(*expert, "tokens"), size_value(*expert, "width"),
            size_value(*expert, "intermediate"), float_value(*expert, "swiglu_limit")));
    }

    if (const auto* hc = section("hyper_connection")) {
        auto result = vastloom::inference::deepseek_v4_hc_pre(
            floats(*hc, "input"), floats(*hc, "mixing_weight"), floats(*hc, "scale"),
            floats(*hc, "base"), size_value(*hc, "tokens"),
            size_value(*hc, "multiplier"), size_value(*hc, "width"),
            size_value(*hc, "sinkhorn_iterations"), float_value(*hc, "norm_epsilon"),
            float_value(*hc, "hc_epsilon"));
        CHECK(result);
        if (result) {
            check_near(result.value().reduced, floats(*hc, "expected_reduced"));
            check_near(result.value().split.pre, floats(*hc, "expected_pre"));
            check_near(result.value().split.post, floats(*hc, "expected_post"));
            check_near(result.value().split.combination, floats(*hc, "expected_combination"));
            auto posted = vastloom::inference::deepseek_v4_hc_post(
                floats(*hc, "sublayer"), floats(*hc, "input"), result.value().split,
                size_value(*hc, "width"));
            CHECK(posted);
            if (posted) check_near(posted.value(), floats(*hc, "expected_output"));
        }
        CHECK(!vastloom::inference::deepseek_v4_hc_split_sinkhorn(
            {}, {}, {}, 1U, 2U, 0U, 1.0e-6F));
    }

    if (const auto* rope = section("rope")) {
        auto result = vastloom::inference::deepseek_v4_apply_rope(
            floats(*rope, "input"), unsigneds(*rope, "positions"),
            size_value(*rope, "rows"), size_value(*rope, "heads"),
            size_value(*rope, "width"), float_value(*rope, "theta"));
        CHECK(result);
        if (result) {
            check_near(result.value(), floats(*rope, "expected"));
            auto inverse = vastloom::inference::deepseek_v4_apply_rope(
                result.value(), unsigneds(*rope, "positions"), size_value(*rope, "rows"),
                size_value(*rope, "heads"), size_value(*rope, "width"),
                float_value(*rope, "theta"), true);
            CHECK(inverse);
            if (inverse) check_near(inverse.value(), floats(*rope, "expected_inverse"));
        }
        CHECK(!vastloom::inference::deepseek_v4_apply_rope(
            floats(*rope, "input"), unsigneds(*rope, "positions"),
            size_value(*rope, "rows"), size_value(*rope, "heads"), 3U,
            float_value(*rope, "theta")));
    }

    if (const auto* attention = section("sparse_attention")) {
        auto result = vastloom::inference::deepseek_v4_sparse_attention(
            floats(*attention, "query"), floats(*attention, "key_value"),
            floats(*attention, "attention_sink"), signed_indices(*attention, "selected_indices"),
            size_value(*attention, "queries"), size_value(*attention, "kv_count"),
            size_value(*attention, "heads"), size_value(*attention, "width"),
            size_value(*attention, "selected"), float_value(*attention, "softmax_scale"));
        CHECK(result);
        if (result) check_near(result.value(), floats(*attention, "expected"));
        auto invalid_indices = signed_indices(*attention, "selected_indices");
        invalid_indices[0] = static_cast<std::int32_t>(size_value(*attention, "kv_count"));
        CHECK(!vastloom::inference::deepseek_v4_sparse_attention(
            floats(*attention, "query"), floats(*attention, "key_value"),
            floats(*attention, "attention_sink"), invalid_indices,
            size_value(*attention, "queries"), size_value(*attention, "kv_count"),
            size_value(*attention, "heads"), size_value(*attention, "width"),
            size_value(*attention, "selected"), float_value(*attention, "softmax_scale")));
    }
}

void test_deepseek_v4_reference_graph(const std::filesystem::path& root) {
    const auto path = std::filesystem::path(VASTLOOM_SOURCE_DIR) / "tests" / "fixtures" /
                      "deepseek_v4_graph.json";
    vastloom::util::JsonLimits limits;
    limits.max_input_bytes = 2U * vastloom::MiB;
    limits.max_nodes = 100'000U;
    limits.max_array_elements = 100'000U;
    auto fixture = vastloom::util::load_json_file(path, limits);
    CHECK(fixture && fixture.value().is_object());
    if (!fixture || !fixture.value().is_object()) return;

    const auto object = [&](const vastloom::util::JsonValue& parent,
                            std::string_view key) -> const vastloom::util::JsonValue::Object* {
        const auto* value = parent.find(key);
        auto parsed = value == nullptr
                          ? vastloom::Result<const vastloom::util::JsonValue::Object*>(
                                vastloom::make_error(vastloom::ErrorCode::invalid_format,
                                                     "missing fixture object"))
                          : value->as_object();
        CHECK(parsed);
        return parsed ? parsed.value() : nullptr;
    };
    const auto array = [&](const vastloom::util::JsonValue& parent,
                           std::string_view key) -> const vastloom::util::JsonValue::Array* {
        const auto* value = parent.find(key);
        auto parsed = value == nullptr
                          ? vastloom::Result<const vastloom::util::JsonValue::Array*>(
                                vastloom::make_error(vastloom::ErrorCode::invalid_format,
                                                     "missing fixture array"))
                          : value->as_array();
        CHECK(parsed);
        return parsed ? parsed.value() : nullptr;
    };
    const auto size_value = [&](const vastloom::util::JsonValue& parent,
                                std::string_view key) -> std::size_t {
        const auto* value = parent.find(key);
        auto parsed = value == nullptr
                          ? vastloom::Result<std::uint64_t>(vastloom::make_error(
                                vastloom::ErrorCode::invalid_format, "missing fixture integer"))
                          : value->as_u64();
        CHECK(parsed && parsed.value() <= std::numeric_limits<std::size_t>::max());
        if (!parsed || parsed.value() > std::numeric_limits<std::size_t>::max()) return 0U;
        return static_cast<std::size_t>(parsed.value());
    };
    const auto float_value = [&](const vastloom::util::JsonValue& parent,
                                 std::string_view key) -> float {
        const auto* value = parent.find(key);
        auto parsed = value == nullptr
                          ? vastloom::Result<double>(vastloom::make_error(
                                vastloom::ErrorCode::invalid_format, "missing fixture scalar"))
                          : value->as_double();
        CHECK(parsed && std::isfinite(parsed.value()));
        if (!parsed || !std::isfinite(parsed.value())) return 0.0F;
        return static_cast<float>(parsed.value());
    };
    const auto floats = [&](const vastloom::util::JsonValue& value) -> std::vector<float> {
        std::vector<float> result;
        auto parsed = value.as_array();
        CHECK(parsed);
        if (!parsed) return result;
        result.reserve(parsed.value()->size());
        for (const auto& item : *parsed.value()) {
            auto number = item.as_double();
            CHECK(number && std::isfinite(number.value()));
            if (!number || !std::isfinite(number.value())) return {};
            result.push_back(static_cast<float>(number.value()));
        }
        return result;
    };
    const auto unsigneds = [&](const vastloom::util::JsonValue& value) {
        std::vector<std::uint32_t> result;
        auto parsed = value.as_array();
        CHECK(parsed);
        if (!parsed) return result;
        result.reserve(parsed.value()->size());
        for (const auto& item : *parsed.value()) {
            auto number = item.as_u64();
            CHECK(number && number.value() <= std::numeric_limits<std::uint32_t>::max());
            if (!number || number.value() > std::numeric_limits<std::uint32_t>::max()) return std::vector<std::uint32_t>{};
            result.push_back(static_cast<std::uint32_t>(number.value()));
        }
        return result;
    };
    const auto signed_indices = [&](const vastloom::util::JsonValue& value) {
        std::vector<std::int32_t> result;
        auto parsed = value.as_array();
        CHECK(parsed);
        if (!parsed) return result;
        result.reserve(parsed.value()->size());
        for (const auto& item : *parsed.value()) {
            auto number = item.as_i64();
            CHECK(number && number.value() >= std::numeric_limits<std::int32_t>::min() &&
                  number.value() <= std::numeric_limits<std::int32_t>::max());
            if (!number || number.value() < std::numeric_limits<std::int32_t>::min() ||
                number.value() > std::numeric_limits<std::int32_t>::max()) {
                return std::vector<std::int32_t>{};
            }
            result.push_back(static_cast<std::int32_t>(number.value()));
        }
        return result;
    };

    const auto* config_object = object(fixture.value(), "config");
    const auto* tensor_object = object(fixture.value(), "tensors");
    const auto* dtype_object = object(fixture.value(), "tensor_dtypes");
    const auto* expected_object = object(fixture.value(), "expected");
    if (config_object == nullptr || tensor_object == nullptr || dtype_object == nullptr ||
        expected_object == nullptr) {
        return;
    }
    const vastloom::util::JsonValue config_value(*config_object);
    vastloom::inference::DeepseekV4ReferenceGraph graph;
    auto& config = graph.config;
    config.vocabulary_size = size_value(config_value, "vocabulary_size");
    config.width = size_value(config_value, "width");
    config.intermediate_width = size_value(config_value, "intermediate_width");
    config.layer_count = size_value(config_value, "layers");
    config.hash_layer_count = size_value(config_value, "hash_layers");
    config.mtp_layer_count = size_value(config_value, "mtp_layers");
    config.head_count = size_value(config_value, "heads");
    config.expert_count = size_value(config_value, "experts");
    config.experts_per_token = size_value(config_value, "experts_per_token");
    config.query_lora_rank = size_value(config_value, "query_lora_rank");
    config.head_width = size_value(config_value, "head_width");
    config.rope_width = size_value(config_value, "rope_width");
    config.output_groups = size_value(config_value, "output_groups");
    config.output_lora_rank = size_value(config_value, "output_lora_rank");
    config.window_size = size_value(config_value, "window_size");
    const auto* ratios = array(config_value, "compression_ratios");
    if (ratios == nullptr) return;
    for (const auto& ratio : *ratios) {
        auto parsed = ratio.as_u64();
        CHECK(parsed && parsed.value() <= std::numeric_limits<std::size_t>::max());
        if (!parsed || parsed.value() > std::numeric_limits<std::size_t>::max()) return;
        config.compression_ratios.push_back(static_cast<std::size_t>(parsed.value()));
    }
    const auto* mtp_ratios = array(config_value, "mtp_compression_ratios");
    if (mtp_ratios == nullptr) return;
    for (const auto& ratio : *mtp_ratios) {
        auto parsed = ratio.as_u64();
        CHECK(parsed && parsed.value() <= std::numeric_limits<std::size_t>::max());
        if (!parsed || parsed.value() > std::numeric_limits<std::size_t>::max()) return;
        config.mtp_compression_ratios.push_back(static_cast<std::size_t>(parsed.value()));
    }
    config.compression_rope_theta = float_value(config_value, "compression_rope_theta");
    config.original_context = size_value(config_value, "original_context");
    config.rope_theta = float_value(config_value, "rope_theta");
    config.rope_factor = float_value(config_value, "rope_factor");
    config.beta_fast = float_value(config_value, "beta_fast");
    config.beta_slow = float_value(config_value, "beta_slow");
    config.index_head_count = size_value(config_value, "index_heads");
    config.index_head_width = size_value(config_value, "index_head_width");
    config.index_top_k = size_value(config_value, "index_top_k");
    config.hyper_multiplier = size_value(config_value, "hyper_multiplier");
    config.sinkhorn_iterations = size_value(config_value, "sinkhorn_iterations");
    config.norm_epsilon = float_value(config_value, "norm_epsilon");
    config.hyper_epsilon = float_value(config_value, "hyper_epsilon");
    config.route_scale = float_value(config_value, "route_scale");
    config.swiglu_limit = float_value(config_value, "swiglu_limit");

    for (const auto& [name, value] : *tensor_object) {
        const auto dtype = dtype_object->find(name);
        CHECK(dtype != dtype_object->end());
        if (dtype == dtype_object->end()) return;
        auto type = dtype->second.as_string();
        CHECK(type);
        if (!type) return;
        if (type.value() == "F32") graph.float_tensors.emplace(name, floats(value));
        else if (type.value() == "I32") graph.integer_tensors.emplace(name, unsigneds(value));
        else CHECK(false);
    }
    const auto* token_value = fixture.value().find("token_ids");
    CHECK(token_value != nullptr);
    if (token_value == nullptr) return;
    const auto token_ids = unsigneds(*token_value);
    vastloom::inference::DeepseekV4ReferenceState incremental_state;
    auto result = vastloom::inference::deepseek_v4_reference_prefill(
        graph, token_ids, &incremental_state);
    CHECK(result);
    if (!result) {
        std::cerr << "DeepSeek graph reference error: " << result.error().message << " ("
                  << result.error().context << ")\n";
        return;
    }
    auto scheduled_graph = graph;
    const std::size_t scheduled_blocks =
        config.layer_count + config.mtp_layer_count;
    scheduled_graph.config.routed_expert_swiglu_limits.assign(
        scheduled_blocks, config.swiglu_limit);
    scheduled_graph.config.shared_expert_swiglu_limits.assign(
        scheduled_blocks, config.swiglu_limit);
    scheduled_graph.config.swiglu_limit = 0.0F;
    auto scheduled_result = vastloom::inference::deepseek_v4_reference_prefill(
        scheduled_graph, token_ids);
    CHECK(scheduled_result && scheduled_result.value().logits == result.value().logits &&
          scheduled_result.value().mtp_logits == result.value().mtp_logits);
    auto malformed_schedule = scheduled_graph;
    malformed_schedule.config.shared_expert_swiglu_limits.pop_back();
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
        malformed_schedule, token_ids));
    CHECK(incremental_state.next_position == token_ids.size());
    const float tolerance = float_value(fixture.value(), "absolute_tolerance");
    const auto expected_logits_value = expected_object->find("logits");
    CHECK(expected_logits_value != expected_object->end());
    if (expected_logits_value == expected_object->end()) return;
    const auto expected_logits = floats(expected_logits_value->second);
    CHECK(result.value().logits.size() == expected_logits.size());
    for (std::size_t index = 0U;
         index < std::min(result.value().logits.size(), expected_logits.size()); ++index) {
        if (std::abs(result.value().logits[index] - expected_logits[index]) > tolerance) {
            std::cerr << "DeepSeek graph logit mismatch at " << index << ": "
                      << result.value().logits[index] << " versus " << expected_logits[index]
                      << '\n';
            ++failures;
            break;
        }
    }
    const vastloom::util::JsonValue expected_value(*expected_object);
    CHECK(result.value().greedy_token_id == size_value(expected_value, "greedy_token_id"));
    const auto* expected_routes = array(expected_value, "routes");
    const auto* expected_attention = array(expected_value, "attention_indices");
    if (expected_routes == nullptr || expected_attention == nullptr) return;
    CHECK(result.value().layers.size() == expected_routes->size());
    CHECK(result.value().layers.size() == expected_attention->size());
    for (std::size_t layer = 0U;
         layer < std::min(result.value().layers.size(), expected_routes->size()); ++layer) {
        auto route_object = (*expected_routes)[layer].as_object();
        auto attention_object = (*expected_attention)[layer].as_object();
        CHECK(route_object && attention_object);
        if (!route_object || !attention_object) return;
        const vastloom::util::JsonValue route_value(*route_object.value());
        const vastloom::util::JsonValue attention_value(*attention_object.value());
        const auto* route_indices_value = route_value.find("indices");
        const auto* route_weights_value = route_value.find("weights");
        const auto* attention_indices_value = attention_value.find("indices");
        CHECK(route_indices_value != nullptr && route_weights_value != nullptr &&
              attention_indices_value != nullptr);
        if (route_indices_value == nullptr || route_weights_value == nullptr ||
            attention_indices_value == nullptr) {
            return;
        }
        CHECK(result.value().layers[layer].route.indices == unsigneds(*route_indices_value));
        const auto expected_weights = floats(*route_weights_value);
        CHECK(result.value().layers[layer].route.weights.size() == expected_weights.size());
        for (std::size_t index = 0U;
             index < std::min(result.value().layers[layer].route.weights.size(),
                              expected_weights.size()); ++index) {
            CHECK(std::abs(result.value().layers[layer].route.weights[index] -
                           expected_weights[index]) <= tolerance);
        }
        CHECK(result.value().layers[layer].attention_indices ==
              signed_indices(*attention_indices_value));
    }

    const auto expected_mtp_logits_value = expected_object->find("mtp_logits");
    CHECK(expected_mtp_logits_value != expected_object->end());
    if (expected_mtp_logits_value == expected_object->end()) return;
    const auto expected_mtp_logits = floats(expected_mtp_logits_value->second);
    CHECK(result.value().mtp_logits.size() == expected_mtp_logits.size());
    for (std::size_t index = 0U;
         index < std::min(result.value().mtp_logits.size(), expected_mtp_logits.size()); ++index) {
        if (std::abs(result.value().mtp_logits[index] - expected_mtp_logits[index]) > tolerance) {
            std::cerr << "DeepSeek graph MTP logit mismatch at " << index << ": "
                      << result.value().mtp_logits[index] << " versus "
                      << expected_mtp_logits[index] << '\n';
            ++failures;
            break;
        }
    }
    CHECK(result.value().mtp_greedy_token_id ==
          size_value(expected_value, "mtp_greedy_token_id"));
    const auto* expected_mtp_routes = array(expected_value, "mtp_routes");
    const auto* expected_mtp_attention = array(expected_value, "mtp_attention_indices");
    if (expected_mtp_routes == nullptr || expected_mtp_attention == nullptr) return;
    CHECK(result.value().mtp_layers.size() == expected_mtp_routes->size());
    CHECK(result.value().mtp_layers.size() == expected_mtp_attention->size());
    for (std::size_t layer = 0U;
         layer < std::min(result.value().mtp_layers.size(), expected_mtp_routes->size()); ++layer) {
        auto route_object = (*expected_mtp_routes)[layer].as_object();
        auto attention_object = (*expected_mtp_attention)[layer].as_object();
        CHECK(route_object && attention_object);
        if (!route_object || !attention_object) return;
        const vastloom::util::JsonValue route_value(*route_object.value());
        const vastloom::util::JsonValue attention_value(*attention_object.value());
        const auto* route_indices_value = route_value.find("indices");
        const auto* route_weights_value = route_value.find("weights");
        const auto* attention_indices_value = attention_value.find("indices");
        CHECK(route_indices_value != nullptr && route_weights_value != nullptr &&
              attention_indices_value != nullptr);
        if (route_indices_value == nullptr || route_weights_value == nullptr ||
            attention_indices_value == nullptr) {
            return;
        }
        CHECK(result.value().mtp_layers[layer].route.indices == unsigneds(*route_indices_value));
        const auto expected_weights = floats(*route_weights_value);
        CHECK(result.value().mtp_layers[layer].route.weights.size() == expected_weights.size());
        for (std::size_t index = 0U;
             index < std::min(result.value().mtp_layers[layer].route.weights.size(),
                              expected_weights.size()); ++index) {
            CHECK(std::abs(result.value().mtp_layers[layer].route.weights[index] -
                           expected_weights[index]) <= tolerance);
        }
        CHECK(result.value().mtp_layers[layer].attention_indices ==
              signed_indices(*attention_indices_value));
    }

    const auto compare_incremental_logits = [&](std::span<const float> actual,
                                                const vastloom::util::JsonValue& step,
                                                std::string_view key,
                                                std::string_view label) {
        const auto* expected_item = step.find(key);
        CHECK(expected_item != nullptr);
        if (expected_item == nullptr) return;
        const auto expected_items = floats(*expected_item);
        CHECK(actual.size() == expected_items.size());
        for (std::size_t index = 0U;
             index < std::min(actual.size(), expected_items.size()); ++index) {
            if (std::abs(actual[index] - expected_items[index]) > tolerance) {
                std::cerr << "DeepSeek incremental " << label << " mismatch at " << index
                          << ": " << actual[index] << " versus " << expected_items[index]
                          << '\n';
                ++failures;
                break;
            }
        }
    };
    const auto compare_incremental_layers = [&] (
        const std::vector<vastloom::inference::DeepseekV4LayerTrace>& actual,
        const vastloom::util::JsonValue& step,
        std::string_view route_key,
        std::string_view attention_key) {
        const auto* routes = array(step, route_key);
        const auto* attention = array(step, attention_key);
        CHECK(routes != nullptr && attention != nullptr);
        if (routes == nullptr || attention == nullptr) return;
        CHECK(actual.size() == routes->size());
        CHECK(actual.size() == attention->size());
        for (std::size_t layer = 0U;
             layer < std::min(actual.size(), routes->size()); ++layer) {
            auto route_object = (*routes)[layer].as_object();
            auto attention_object = (*attention)[layer].as_object();
            CHECK(route_object && attention_object);
            if (!route_object || !attention_object) return;
            const vastloom::util::JsonValue route_value(*route_object.value());
            const vastloom::util::JsonValue attention_value(*attention_object.value());
            const auto* route_indices = route_value.find("indices");
            const auto* route_weights = route_value.find("weights");
            const auto* attention_indices = attention_value.find("indices");
            CHECK(route_indices != nullptr && route_weights != nullptr &&
                  attention_indices != nullptr);
            if (route_indices == nullptr || route_weights == nullptr ||
                attention_indices == nullptr) {
                return;
            }
            CHECK(actual[layer].route.indices == unsigneds(*route_indices));
            const auto weights = floats(*route_weights);
            CHECK(actual[layer].route.weights.size() == weights.size());
            for (std::size_t index = 0U;
                 index < std::min(actual[layer].route.weights.size(), weights.size()); ++index) {
                CHECK(std::abs(actual[layer].route.weights[index] - weights[index]) <= tolerance);
            }
            CHECK(actual[layer].attention_indices == signed_indices(*attention_indices));
        }
    };
    const auto* decode_tokens_value = fixture.value().find("decode_token_ids");
    const auto* incremental_expected = array(expected_value, "incremental");
    CHECK(decode_tokens_value != nullptr && incremental_expected != nullptr);
    if (decode_tokens_value == nullptr || incremental_expected == nullptr) return;
    const auto decode_tokens = unsigneds(*decode_tokens_value);
    CHECK(decode_tokens.size() == incremental_expected->size());
    for (std::size_t step_index = 0U;
         step_index < std::min(decode_tokens.size(), incremental_expected->size()); ++step_index) {
        auto step_object = (*incremental_expected)[step_index].as_object();
        CHECK(step_object);
        if (!step_object) return;
        const vastloom::util::JsonValue step(*step_object.value());
        CHECK(size_value(step, "position") == incremental_state.next_position);
        CHECK(size_value(step, "token_id") == decode_tokens[step_index]);
        auto decoded = vastloom::inference::deepseek_v4_reference_decode(
            graph, decode_tokens[step_index], incremental_state);
        CHECK(decoded);
        if (!decoded) {
            std::cerr << "DeepSeek incremental reference error: "
                      << decoded.error().message << " (" << decoded.error().context << ")\n";
            return;
        }
        CHECK(incremental_state.next_position == token_ids.size() + step_index + 1U);
        compare_incremental_logits(decoded.value().logits, step, "logits", "logit");
        compare_incremental_logits(decoded.value().mtp_logits, step, "mtp_logits",
                                   "MTP logit");
        CHECK(decoded.value().greedy_token_id == size_value(step, "greedy_token_id"));
        CHECK(decoded.value().mtp_greedy_token_id ==
              size_value(step, "mtp_greedy_token_id"));
        compare_incremental_layers(decoded.value().layers, step, "routes",
                                   "attention_indices");
        compare_incremental_layers(decoded.value().mtp_layers, step, "mtp_routes",
                                   "mtp_attention_indices");
    }

    const auto* greedy_generation_value = expected_value.find("greedy_generation");
    CHECK(greedy_generation_value != nullptr);
    if (greedy_generation_value == nullptr) return;
    const auto expected_generation = unsigneds(*greedy_generation_value);
    CHECK(expected_generation.size() == 8U);
    vastloom::inference::DeepseekV4ReferenceGenerationOptions generation_options;
    generation_options.max_new_tokens = expected_generation.size();
    generation_options.top_k = 1U;
    generation_options.temperature = 0.0F;
    std::vector<std::uint32_t> streamed_tokens;
    generation_options.on_token = [&](std::uint32_t token) {
        streamed_tokens.push_back(token);
    };
    auto generated = vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, generation_options);
    CHECK(generated);
    if (generated) {
        CHECK(generated.value().prompt_ids == token_ids);
        CHECK(generated.value().generated_ids == expected_generation);
        CHECK(streamed_tokens == expected_generation);
        CHECK(generated.value().finish_reason ==
              vastloom::inference::DeepseekV4ReferenceFinishReason::max_tokens);
        CHECK(vastloom::inference::to_string(generated.value().finish_reason) ==
              "max_tokens");
        CHECK(!generated.value().stop_token_id.has_value());
        CHECK(generated.value().stats.prompt_tokens == token_ids.size());
        CHECK(generated.value().stats.generated_tokens == expected_generation.size());
        CHECK(generated.value().stats.requested_max_tokens ==
              expected_generation.size());
        CHECK(generated.value().stats.effective_max_tokens ==
              expected_generation.size());
        CHECK(generated.value().stats.forward_passes == expected_generation.size());
        CHECK(generated.value().stats.expert_requests == 0U);
        CHECK(generated.value().stats.expert_cache_hits == 0U);
        CHECK(generated.value().stats.expert_cache_misses == 0U);
        CHECK(generated.value().stats.logical_expert_bytes_read == 0U);
        CHECK(generated.value().stats.prompt_eval_ms >= 0.0);
        CHECK(generated.value().stats.generation_ms >= 0.0);
        CHECK(generated.value().stats.tokens_per_second >= 0.0);
    }

    auto stop_options = generation_options;
    stop_options.on_token = {};
    stop_options.stop_token_ids = {expected_generation.front()};
    auto stopped = vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, stop_options);
    CHECK(stopped && stopped.value().generated_ids.empty() &&
          stopped.value().finish_reason ==
              vastloom::inference::DeepseekV4ReferenceFinishReason::stop_token &&
          stopped.value().stop_token_id == expected_generation.front() &&
          stopped.value().stats.forward_passes == 1U);

    auto context_options = generation_options;
    context_options.on_token = {};
    context_options.max_new_tokens = 8U;
    context_options.context_limit_tokens = token_ids.size() + 2U;
    auto context_limited = vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, context_options);
    CHECK(context_limited &&
          context_limited.value().generated_ids ==
              std::vector<std::uint32_t>(expected_generation.begin(),
                                         expected_generation.begin() + 2) &&
          context_limited.value().finish_reason ==
              vastloom::inference::DeepseekV4ReferenceFinishReason::context_limit &&
          context_limited.value().stats.effective_max_tokens == 2U &&
          context_limited.value().stats.forward_passes == 2U);

    auto sampled_options = generation_options;
    sampled_options.on_token = {};
    sampled_options.max_new_tokens = 6U;
    sampled_options.temperature = 0.7F;
    sampled_options.top_k = 5U;
    sampled_options.seed = 0x564153544c4f4f4dULL;
    auto sampled_first = vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, sampled_options);
    auto sampled_second = vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, sampled_options);
    const std::vector<std::uint32_t> expected_sampled_generation{
        4U, 11U, 13U, 2U, 13U, 11U};
    CHECK(sampled_first && sampled_second &&
          sampled_first.value().generated_ids == sampled_second.value().generated_ids &&
          sampled_first.value().generated_ids == expected_sampled_generation);
    auto tiny_temperature_options = sampled_options;
    tiny_temperature_options.max_new_tokens = 2U;
    tiny_temperature_options.temperature =
        std::numeric_limits<float>::denorm_min();
    auto tiny_temperature =
        vastloom::inference::deepseek_v4_reference_generate(
            graph, token_ids, tiny_temperature_options);
    CHECK(tiny_temperature &&
          tiny_temperature.value().generated_ids ==
              std::vector<std::uint32_t>(expected_generation.begin(),
                                         expected_generation.begin() + 2));

    bool cancel_after_first = false;
    auto generation_cancel_options = generation_options;
    generation_cancel_options.max_new_tokens = 4U;
    generation_cancel_options.on_token = [&](std::uint32_t) {
        cancel_after_first = true;
    };
    generation_cancel_options.cancelled = [&] { return cancel_after_first; };
    auto cancelled = vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, generation_cancel_options);
    CHECK(cancelled && cancelled.value().generated_ids.size() == 1U &&
          cancelled.value().generated_ids.front() == expected_generation.front() &&
          cancelled.value().finish_reason ==
              vastloom::inference::DeepseekV4ReferenceFinishReason::cancelled &&
          cancelled.value().stats.forward_passes == 1U);

    auto timeout_options = generation_options;
    timeout_options.on_token = {};
    timeout_options.timeout = std::chrono::milliseconds(1);
    timeout_options.cancelled = [] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return false;
    };
    auto timed_out = vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, timeout_options);
    CHECK(timed_out && timed_out.value().generated_ids.empty() &&
          timed_out.value().finish_reason ==
              vastloom::inference::DeepseekV4ReferenceFinishReason::timeout);

    auto invalid_generation = generation_options;
    invalid_generation.max_new_tokens = 0U;
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.temperature = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.top_k = config.vocabulary_size + 1U;
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.context_limit_tokens = token_ids.size();
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.stop_token_ids = {1U, 1U};
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.stop_token_ids = {
        static_cast<std::uint32_t>(config.vocabulary_size)};
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.stop_token_ids.assign(65U, 1U);
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.timeout = std::chrono::milliseconds::zero();
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.cancelled = []() -> bool {
        throw std::runtime_error("injected cancellation failure");
    };
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));
    invalid_generation = generation_options;
    invalid_generation.on_token = [](std::uint32_t) {
        throw std::runtime_error("injected callback failure");
    };
    CHECK(!vastloom::inference::deepseek_v4_reference_generate(
        graph, token_ids, invalid_generation));

    vastloom::inference::DeepseekV4ReferenceState checkpoint_state;
    CHECK(vastloom::inference::deepseek_v4_reference_prefill(
        graph, token_ids, &checkpoint_state));
    const auto checkpoint_state_before = checkpoint_state;
    std::size_t checkpoint_calls = 0U;
    vastloom::inference::DeepseekV4ReferenceExecutionControl failing_control;
    failing_control.checkpoint = [&]() -> vastloom::Result<void> {
        if (++checkpoint_calls == 3U) {
            return vastloom::make_error(
                vastloom::ErrorCode::resource_exhausted,
                "injected layer checkpoint failure");
        }
        return {};
    };
    CHECK(!vastloom::inference::deepseek_v4_reference_decode(
        graph, expected_generation.front(), checkpoint_state, &failing_control));
    CHECK(checkpoint_state.next_position == checkpoint_state_before.next_position);
    CHECK(checkpoint_state.layers[0].window_key_values ==
          checkpoint_state_before.layers[0].window_key_values);
    CHECK(checkpoint_state.layers[1].compressor.compressed ==
          checkpoint_state_before.layers[1].compressor.compressed);
    auto unpublished_prefill_state = checkpoint_state_before;
    checkpoint_calls = 0U;
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
        graph, token_ids, &unpublished_prefill_state, &failing_control));
    CHECK(unpublished_prefill_state.next_position ==
          checkpoint_state_before.next_position);
    CHECK(unpublished_prefill_state.layers[0].window_key_values ==
          checkpoint_state_before.layers[0].window_key_values);
    CHECK(unpublished_prefill_state.layers[1].compressor.compressed ==
          checkpoint_state_before.layers[1].compressor.compressed);

    // Exercise embedding lookup and the output head from bounded Q4_K storage. The scalar
    // Q4_K test above is pinned independently to ggml; this graph oracle contains its decoded
    // logical columns and therefore verifies both prefill and incremental integration.
    auto q4_oracle_graph = graph;
    auto q4_packed_graph = graph;
    const auto write_q4_u16 = [](std::span<std::byte> bytes,
                                 std::size_t offset,
                                 std::uint16_t value) {
        bytes[offset] = static_cast<std::byte>(value & 0xffU);
        bytes[offset + 1U] = static_cast<std::byte>(value >> 8U);
    };
    const auto install_q4_matrix = [&](std::string_view name,
                                       std::size_t rows,
                                       std::uint32_t seed) {
        constexpr std::size_t storage_columns =
            vastloom::inference::kQ4KBlockElements;
        std::vector<std::byte> bytes(
            rows * vastloom::inference::kQ4KBlockBytes);
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t base =
                row * vastloom::inference::kQ4KBlockBytes;
            write_q4_u16(bytes, base, 0x2800U);
            write_q4_u16(bytes, base + 2U, 0x2400U);
            for (std::size_t index = 0U; index < 12U; ++index) {
                bytes[base + 4U + index] = static_cast<std::byte>(
                    (seed + row * 29U + index * 47U + 7U) & 0xffU);
            }
            for (std::size_t index = 0U; index < 128U; ++index) {
                bytes[base + 16U + index] = static_cast<std::byte>(
                    (seed * 3U + row * 37U + index * 53U + 11U) & 0xffU);
            }
        }
        std::vector<float> physical(rows * storage_columns);
        vastloom::inference::QuantizedKernelLimits q4_limits;
        q4_limits.max_elements = physical.size();
        auto decoded = vastloom::inference::dequantize_q4_k(
            bytes, physical, q4_limits);
        CHECK(decoded);
        if (!decoded) return false;
        std::vector<float> logical(rows * config.width);
        for (std::size_t row = 0U; row < rows; ++row) {
            std::copy_n(
                physical.begin() + static_cast<std::ptrdiff_t>(row * storage_columns),
                config.width,
                logical.begin() + static_cast<std::ptrdiff_t>(row * config.width));
        }
        q4_oracle_graph.float_tensors[std::string(name)] = std::move(logical);
        q4_packed_graph.float_tensors.erase(std::string(name));
        q4_packed_graph.packed_matrices.emplace(
            name, vastloom::inference::DeepseekV4PackedMatrix{
                      vastloom::inference::kGgmlTypeQ4K, rows, config.width,
                      storage_columns, std::move(bytes)});
        return true;
    };
    CHECK(install_q4_matrix("embed.weight", config.vocabulary_size, 17U));
    CHECK(install_q4_matrix("head.weight", config.vocabulary_size, 71U));
    CHECK(!q4_packed_graph.float_tensors.contains("embed.weight") &&
          !q4_packed_graph.float_tensors.contains("head.weight"));

    vastloom::inference::DeepseekV4ReferenceState q4_oracle_state;
    vastloom::inference::DeepseekV4ReferenceState q4_packed_state;
    auto q4_oracle_result = vastloom::inference::deepseek_v4_reference_prefill(
        q4_oracle_graph, token_ids, &q4_oracle_state);
    auto q4_packed_result = vastloom::inference::deepseek_v4_reference_prefill(
        q4_packed_graph, token_ids, &q4_packed_state);
    CHECK(q4_oracle_result && q4_packed_result);
    if (q4_oracle_result && q4_packed_result) {
        CHECK(q4_packed_result.value().logits == q4_oracle_result.value().logits);
        CHECK(q4_packed_result.value().mtp_logits ==
              q4_oracle_result.value().mtp_logits);
        CHECK(q4_packed_result.value().greedy_token_id ==
              q4_oracle_result.value().greedy_token_id);
        CHECK(q4_packed_result.value().mtp_greedy_token_id ==
              q4_oracle_result.value().mtp_greedy_token_id);
    }
    CHECK(q4_packed_state.next_position == q4_oracle_state.next_position);
    CHECK(q4_packed_state.layers.size() == q4_oracle_state.layers.size());
    for (std::size_t layer = 0U;
         layer < std::min(q4_packed_state.layers.size(),
                          q4_oracle_state.layers.size()); ++layer) {
        CHECK(q4_packed_state.layers[layer].window_key_values ==
              q4_oracle_state.layers[layer].window_key_values);
        CHECK(q4_packed_state.layers[layer].compressor.compressed ==
              q4_oracle_state.layers[layer].compressor.compressed);
        CHECK(q4_packed_state.layers[layer].indexer.compressed ==
              q4_oracle_state.layers[layer].indexer.compressed);
    }
    auto q4_oracle_decode = vastloom::inference::deepseek_v4_reference_decode(
        q4_oracle_graph, decode_tokens.front(), q4_oracle_state);
    auto q4_packed_decode = vastloom::inference::deepseek_v4_reference_decode(
        q4_packed_graph, decode_tokens.front(), q4_packed_state);
    CHECK(q4_oracle_decode && q4_packed_decode);
    if (q4_oracle_decode && q4_packed_decode) {
        CHECK(q4_packed_decode.value().logits == q4_oracle_decode.value().logits);
        CHECK(q4_packed_decode.value().mtp_logits ==
              q4_oracle_decode.value().mtp_logits);
    }

    auto malformed_q4_embedding = q4_packed_graph;
    malformed_q4_embedding.packed_matrices["embed.weight"].encoded.pop_back();
    vastloom::inference::DeepseekV4ReferenceState q4_failed_state;
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
        malformed_q4_embedding, token_ids, &q4_failed_state));
    CHECK(q4_failed_state.next_position == 0U && q4_failed_state.layers.empty());
    malformed_q4_embedding = q4_packed_graph;
    malformed_q4_embedding.packed_matrices["embed.weight"].storage_columns = 255U;
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
        malformed_q4_embedding, token_ids));
    malformed_q4_embedding = q4_packed_graph;
    auto& malformed_embedding_bytes =
        malformed_q4_embedding.packed_matrices["embed.weight"].encoded;
    write_q4_u16(
        malformed_embedding_bytes,
        static_cast<std::size_t>(token_ids.front()) *
            vastloom::inference::kQ4KBlockBytes,
        0x7c00U);
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
        malformed_q4_embedding, token_ids));

    const auto state_before_failure = incremental_state;
    auto failing_decode_graph = graph;
    failing_decode_graph.float_tensors.erase("head.weight");
    CHECK(!vastloom::inference::deepseek_v4_reference_decode(
        failing_decode_graph, decode_tokens.back(), incremental_state));
    CHECK(incremental_state.next_position == state_before_failure.next_position);
    CHECK(incremental_state.layers.size() == state_before_failure.layers.size());
    for (std::size_t layer = 0U;
         layer < std::min(incremental_state.layers.size(),
                          state_before_failure.layers.size()); ++layer) {
        CHECK(incremental_state.layers[layer].window_key_values ==
              state_before_failure.layers[layer].window_key_values);
        CHECK(incremental_state.layers[layer].compressor.compressed ==
              state_before_failure.layers[layer].compressor.compressed);
        CHECK(incremental_state.layers[layer].indexer.compressed ==
              state_before_failure.layers[layer].indexer.compressed);
    }
    auto malformed_state = incremental_state;
    malformed_state.layers[0].window_key_values.clear();
    const auto malformed_position = malformed_state.next_position;
    CHECK(!vastloom::inference::deepseek_v4_reference_decode(
        graph, decode_tokens.back(), malformed_state));
    CHECK(malformed_state.next_position == malformed_position);

    auto malformed = graph;
    malformed.float_tensors.erase("head.weight");
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(malformed, token_ids));
    for (const auto& [name, unused] : graph.float_tensors) {
        static_cast<void>(unused);
        malformed = graph;
        malformed.float_tensors.erase(name);
        auto missing = vastloom::inference::deepseek_v4_reference_prefill(malformed, token_ids);
        if (missing) {
            std::cerr << "DeepSeek graph did not consume fixture tensor: " << name << '\n';
            ++failures;
            break;
        }
    }
    for (const auto& [name, unused] : graph.integer_tensors) {
        static_cast<void>(unused);
        malformed = graph;
        malformed.integer_tensors.erase(name);
        auto missing = vastloom::inference::deepseek_v4_reference_prefill(malformed, token_ids);
        if (missing) {
            std::cerr << "DeepSeek graph did not consume fixture tensor: " << name << '\n';
            ++failures;
            break;
        }
    }
    malformed = graph;
    malformed.float_tensors["embed.weight"][0] =
        std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(malformed, token_ids));
    auto invalid_tokens = token_ids;
    invalid_tokens[0] = static_cast<std::uint32_t>(config.vocabulary_size);
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(graph, invalid_tokens));
    malformed = graph;
    malformed.config.head_count = std::numeric_limits<std::size_t>::max();
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(malformed, token_ids));
    malformed = graph;
    malformed.config.window_size = std::numeric_limits<std::size_t>::max();
    vastloom::inference::DeepseekV4ReferenceState oversized_state;
    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
        malformed, token_ids, &oversized_state));

    // Replace every matrix consumed by the first main layer with deterministic packed
    // storage. Trunk/router/shared-expert matrices use Q8_0; routed W1/W3 use IQ1_S and
    // routed W2 uses IQ3_XXS, matching the real 0731 mixed-format expert contract. The
    // float oracle uses the independently verified GGML dequantization of exactly those
    // bytes, so this validates full graph composition rather than standalone kernels.
    auto quantized_oracle = graph;
    auto packed_graph = graph;
    struct MatrixSpec {
        std::string name;
        std::size_t rows;
        std::size_t columns;
        vastloom::model::TensorRole role;
        std::optional<std::uint32_t> expert;
        std::uint32_t ggml_type{vastloom::inference::kGgmlTypeQ8_0};
        std::size_t storage_columns{vastloom::inference::kQ8_0BlockElements};

        MatrixSpec(std::string tensor_name,
                   std::size_t output_rows,
                   std::size_t input_columns,
                   vastloom::model::TensorRole tensor_role =
                       vastloom::model::TensorRole::attention,
                   std::optional<std::uint32_t> expert_id = std::nullopt)
            : name(std::move(tensor_name)),
              rows(output_rows),
              columns(input_columns),
              role(tensor_role),
              expert(expert_id) {
            if (expert.has_value()) {
                ggml_type = name.ends_with(".w2.weight")
                                ? vastloom::inference::kGgmlTypeIq3Xxs
                                : vastloom::inference::kGgmlTypeIq1S;
                storage_columns = vastloom::inference::kIq1SBlockElements;
            }
        }
    };
    const std::size_t mix_width =
        (2U + config.hyper_multiplier) * config.hyper_multiplier;
    const std::size_t flattened_width = config.hyper_multiplier * config.width;
    std::vector<MatrixSpec> layer_matrices{
        {"layers.0.hc_attn_fn", mix_width, flattened_width,
         vastloom::model::TensorRole::hyper_connection},
        {"layers.0.attn.wq_a.weight", config.query_lora_rank, config.width},
        {"layers.0.attn.wq_b.weight", config.head_count * config.head_width,
         config.query_lora_rank},
        {"layers.0.attn.wkv.weight", config.head_width, config.width},
        {"layers.0.attn.wo_a.weight",
         config.output_groups * config.output_lora_rank,
         config.head_count * config.head_width / config.output_groups},
        {"layers.0.attn.wo_b.weight", config.width,
         config.output_groups * config.output_lora_rank},
        {"layers.0.hc_ffn_fn", mix_width, flattened_width,
         vastloom::model::TensorRole::hyper_connection},
        {"layers.0.ffn.gate.weight", config.expert_count, config.width,
         vastloom::model::TensorRole::router},
    };
    for (std::size_t expert = 0U; expert < config.expert_count; ++expert) {
        const std::string base =
            "layers.0.ffn.experts." + std::to_string(expert);
        layer_matrices.push_back(
            {base + ".w1.weight", config.intermediate_width, config.width,
             vastloom::model::TensorRole::routed_expert,
             static_cast<std::uint32_t>(expert)});
        layer_matrices.push_back(
            {base + ".w2.weight", config.width, config.intermediate_width,
             vastloom::model::TensorRole::routed_expert,
             static_cast<std::uint32_t>(expert)});
        layer_matrices.push_back(
            {base + ".w3.weight", config.intermediate_width, config.width,
             vastloom::model::TensorRole::routed_expert,
             static_cast<std::uint32_t>(expert)});
    }
    layer_matrices.push_back(
        {"layers.0.ffn.shared_experts.w1.weight", config.intermediate_width,
         config.width, vastloom::model::TensorRole::shared_expert});
    layer_matrices.push_back(
        {"layers.0.ffn.shared_experts.w2.weight", config.width,
         config.intermediate_width, vastloom::model::TensorRole::shared_expert});
    layer_matrices.push_back(
        {"layers.0.ffn.shared_experts.w3.weight", config.intermediate_width,
         config.width, vastloom::model::TensorRole::shared_expert});
    struct FloatVectorSpec {
        std::string name;
        vastloom::model::TensorRole role;
    };
    const std::array<FloatVectorSpec, 9U> layer_float_vectors{{
        {"layers.0.attn.attn_sink", vastloom::model::TensorRole::attention},
        {"layers.0.attn.kv_norm.weight", vastloom::model::TensorRole::normalization},
        {"layers.0.attn.q_norm.weight", vastloom::model::TensorRole::normalization},
        {"layers.0.attn_norm.weight", vastloom::model::TensorRole::normalization},
        {"layers.0.ffn_norm.weight", vastloom::model::TensorRole::normalization},
        {"layers.0.hc_attn_base", vastloom::model::TensorRole::hyper_connection},
        {"layers.0.hc_attn_scale", vastloom::model::TensorRole::hyper_connection},
        {"layers.0.hc_ffn_base", vastloom::model::TensorRole::hyper_connection},
        {"layers.0.hc_ffn_scale", vastloom::model::TensorRole::hyper_connection},
    }};
    const std::map<std::string, std::string, std::less<>> layer0_gguf_names{{
        {"layers.0.attn.wkv.weight", "blk.0.attn_kv.weight"},
        {"layers.0.attn.kv_norm.weight", "blk.0.attn_kv_a_norm.weight"},
        {"layers.0.attn_norm.weight", "blk.0.attn_norm.weight"},
        {"layers.0.attn.wo_a.weight", "blk.0.attn_output_a.weight"},
        {"layers.0.attn.wo_b.weight", "blk.0.attn_output_b.weight"},
        {"layers.0.attn.wq_a.weight", "blk.0.attn_q_a.weight"},
        {"layers.0.attn.q_norm.weight", "blk.0.attn_q_a_norm.weight"},
        {"layers.0.attn.wq_b.weight", "blk.0.attn_q_b.weight"},
        {"layers.0.attn.attn_sink", "blk.0.attn_sinks.weight"},
        {"layers.0.ffn.shared_experts.w2.weight", "blk.0.ffn_down_shexp.weight"},
        {"layers.0.ffn.gate.weight", "blk.0.ffn_gate_inp.weight"},
        {"layers.0.ffn.shared_experts.w1.weight", "blk.0.ffn_gate_shexp.weight"},
        {"layers.0.ffn.gate.tid2eid", "blk.0.ffn_gate_tid2eid.weight"},
        {"layers.0.ffn_norm.weight", "blk.0.ffn_norm.weight"},
        {"layers.0.ffn.shared_experts.w3.weight", "blk.0.ffn_up_shexp.weight"},
        {"layers.0.hc_attn_base", "blk.0.hc_attn_base.weight"},
        {"layers.0.hc_attn_fn", "blk.0.hc_attn_fn.weight"},
        {"layers.0.hc_attn_scale", "blk.0.hc_attn_scale.weight"},
        {"layers.0.hc_ffn_base", "blk.0.hc_ffn_base.weight"},
        {"layers.0.hc_ffn_fn", "blk.0.hc_ffn_fn.weight"},
        {"layers.0.hc_ffn_scale", "blk.0.hc_ffn_scale.weight"},
    }};
    CHECK(layer0_gguf_names.size() == 21U);
    const auto make_iq1_s = [](std::size_t rows, std::size_t salt) {
        std::vector<std::byte> encoded(
            rows * vastloom::inference::kIq1SBlockBytes);
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t base =
                row * vastloom::inference::kIq1SBlockBytes;
            encoded[base] = std::byte{0x00U};
            encoded[base + 1U] = std::byte{0x24U};
            for (std::size_t index = 0U; index < 32U; ++index) {
                encoded[base + 2U + index] = static_cast<std::byte>(
                    (salt * 29U + row * 37U + index * 53U + 11U) & 0xffU);
            }
            for (std::size_t group = 0U; group < 8U; ++group) {
                const auto high = static_cast<std::uint16_t>(
                    ((((group * 5U + row + salt) & 7U) << 12U) |
                     (((group + row + salt) & 1U) != 0U ? 0x8000U : 0U) |
                     ((salt * 431U + row * 733U + group * 97U + 0x123U) &
                      0x0fffU)));
                encoded[base + 34U + group * 2U] =
                    static_cast<std::byte>(high & 0xffU);
                encoded[base + 35U + group * 2U] =
                    static_cast<std::byte>(high >> 8U);
            }
        }
        return encoded;
    };
    const auto make_iq2_xxs = [](std::size_t rows, std::size_t salt) {
        std::vector<std::byte> encoded(
            rows * vastloom::inference::kIq2XxsBlockBytes);
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t base =
                row * vastloom::inference::kIq2XxsBlockBytes;
            encoded[base] = std::byte{0x00U};
            encoded[base + 1U] = std::byte{0x20U};
            for (std::size_t index = 0U; index < 32U; ++index) {
                const auto packed = static_cast<std::uint16_t>(
                    (salt * 431U + row * 733U + index * 977U + 0x1234U) &
                    0xffffU);
                encoded[base + 2U + index * 2U] =
                    static_cast<std::byte>(packed & 0xffU);
                encoded[base + 3U + index * 2U] =
                    static_cast<std::byte>(packed >> 8U);
            }
        }
        return encoded;
    };
    const auto make_mxfp4 = [](std::size_t rows, std::size_t salt) {
        std::vector<std::byte> encoded(
            rows * vastloom::inference::kMxfp4BlockBytes);
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t base =
                row * vastloom::inference::kMxfp4BlockBytes;
            encoded[base] = static_cast<std::byte>(
                120U + ((salt + row) % 9U));
            for (std::size_t index = 0U;
                 index < vastloom::inference::kMxfp4BlockElements / 2U;
                 ++index) {
                encoded[base + 1U + index] = static_cast<std::byte>(
                    (salt * 29U + row * 37U + index * 53U + 11U) & 0xffU);
            }
        }
        return encoded;
    };
    const auto make_iq3_xxs = [](std::size_t rows, std::size_t salt) {
        std::vector<std::byte> encoded(
            rows * vastloom::inference::kIq3XxsBlockBytes);
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t base =
                row * vastloom::inference::kIq3XxsBlockBytes;
            encoded[base] = std::byte{0x00U};
            encoded[base + 1U] = std::byte{0x18U};
            for (std::size_t index = 0U; index < 64U; ++index) {
                encoded[base + 2U + index] = static_cast<std::byte>(
                    (salt * 29U + row * 37U + index * 53U + 11U) & 0xffU);
            }
            for (std::size_t group = 0U; group < 8U; ++group) {
                std::uint32_t auxiliary = static_cast<std::uint32_t>(
                    (salt + row * 3U + group * 5U) & 15U)
                                          << 28U;
                for (std::size_t subblock = 0U; subblock < 4U; ++subblock) {
                    const auto sign_index = static_cast<std::uint32_t>(
                        (salt * 17U + row * 19U + group * 13U +
                         subblock * 23U + 7U) &
                        127U);
                    auxiliary |= sign_index << (7U * subblock);
                }
                for (std::size_t byte = 0U; byte < 4U; ++byte) {
                    encoded[base + 66U + group * 4U + byte] =
                        static_cast<std::byte>(
                            (auxiliary >> (8U * byte)) & 0xffU);
                }
            }
        }
        return encoded;
    };
    std::size_t packed_matrix_bytes = 0U;
    std::size_t iq1_s_matrix_count = 0U;
    std::size_t iq3_xxs_matrix_count = 0U;
    bool packed_ok = true;
    for (std::size_t matrix_index = 0U;
         matrix_index < layer_matrices.size(); ++matrix_index) {
        const auto& spec = layer_matrices[matrix_index];
        const auto found = graph.float_tensors.find(spec.name);
        CHECK(found != graph.float_tensors.end());
        if (found == graph.float_tensors.end()) {
            packed_ok = false;
            break;
        }
        CHECK(found->second.size() == spec.rows * spec.columns);
        auto encoded = [&]() -> vastloom::Result<std::vector<std::byte>> {
            if (spec.ggml_type == vastloom::inference::kGgmlTypeQ8_0) {
                return vastloom::inference::quantize_q8_0_reference(
                    found->second, spec.rows, spec.columns,
                    spec.storage_columns);
            }
            if (spec.ggml_type == vastloom::inference::kGgmlTypeIq1S) {
                ++iq1_s_matrix_count;
                return make_iq1_s(spec.rows, matrix_index + 1U);
            }
            if (spec.ggml_type == vastloom::inference::kGgmlTypeIq3Xxs) {
                ++iq3_xxs_matrix_count;
                return make_iq3_xxs(spec.rows, matrix_index + 1U);
            }
            return vastloom::make_error(
                vastloom::ErrorCode::unsupported,
                "Unexpected mixed-format DeepSeek fixture matrix", spec.name);
        }();
        CHECK(encoded);
        if (!encoded) {
            packed_ok = false;
            break;
        }
        std::vector<float> padded(spec.rows * spec.storage_columns);
        auto decoded = [&]() -> vastloom::Result<void> {
            if (spec.ggml_type == vastloom::inference::kGgmlTypeQ8_0) {
                return vastloom::inference::dequantize_q8_0(
                    encoded.value(), padded);
            }
            if (spec.ggml_type == vastloom::inference::kGgmlTypeIq1S) {
                return vastloom::inference::dequantize_iq1_s(
                    encoded.value(), padded);
            }
            return vastloom::inference::dequantize_iq3_xxs(
                encoded.value(), padded);
        }();
        CHECK(decoded);
        if (!decoded) {
            packed_ok = false;
            break;
        }
        std::vector<float> logical(spec.rows * spec.columns);
        for (std::size_t row = 0U; row < spec.rows; ++row) {
            std::copy_n(
                padded.begin() + static_cast<std::ptrdiff_t>(
                    row * spec.storage_columns),
                spec.columns,
                logical.begin() + static_cast<std::ptrdiff_t>(row * spec.columns));
        }
        quantized_oracle.float_tensors[spec.name] = std::move(logical);
        packed_matrix_bytes += encoded.value().size();
        packed_graph.packed_matrices.emplace(
            spec.name,
            vastloom::inference::DeepseekV4PackedMatrix{
                spec.ggml_type, spec.rows, spec.columns,
                spec.storage_columns, encoded.take_value()});
        packed_graph.float_tensors.erase(spec.name);
    }
    CHECK(layer_matrices.size() == 23U);
    CHECK(iq1_s_matrix_count == 8U && iq3_xxs_matrix_count == 4U);
    CHECK(packed_matrix_bytes == 8'052U);
    CHECK(packed_graph.packed_matrices.at(
              "layers.0.ffn.experts.0.w1.weight").ggml_type ==
              vastloom::inference::kGgmlTypeIq1S &&
          packed_graph.packed_matrices.at(
              "layers.0.ffn.experts.0.w2.weight").ggml_type ==
              vastloom::inference::kGgmlTypeIq3Xxs &&
          packed_graph.packed_matrices.at(
              "layers.0.ffn.experts.0.w3.weight").ggml_type ==
              vastloom::inference::kGgmlTypeIq1S);

    const std::string hash_table_name = "layers.0.ffn.gate.tid2eid";
    const auto hash_table = graph.integer_tensors.find(hash_table_name);
    CHECK(hash_table != graph.integer_tensors.end());
    if (hash_table != graph.integer_tensors.end()) {
        std::vector<std::byte> encoded(hash_table->second.size() * 4U);
        for (std::size_t index = 0U; index < hash_table->second.size(); ++index) {
            for (unsigned byte = 0U; byte < 4U; ++byte) {
                encoded[index * 4U + byte] = static_cast<std::byte>(
                    (hash_table->second[index] >> (byte * 8U)) & 0xffU);
            }
        }
        packed_graph.packed_integer_tensors.emplace(
            hash_table_name,
            vastloom::inference::DeepseekV4PackedIntegerTensor{
                hash_table->second.size(), std::move(encoded)});
        packed_graph.integer_tensors.erase(hash_table_name);
    } else {
        packed_ok = false;
    }
    CHECK(packed_graph.packed_matrices.size() == 23U);
    CHECK(packed_graph.packed_integer_tensors.size() == 1U);

    const auto compare_packed_values = [](std::span<const float> actual,
                                          std::span<const float> expected,
                                          float allowed) {
        CHECK(actual.size() == expected.size());
        for (std::size_t index = 0U;
             index < std::min(actual.size(), expected.size()); ++index) {
            CHECK(std::abs(actual[index] - expected[index]) <= allowed);
        }
    };
    const auto compare_packed_forward = [&](const auto& actual,
                                            const auto& expected) {
        compare_packed_values(actual.logits, expected.logits, 3.0e-5F);
        compare_packed_values(actual.mtp_logits, expected.mtp_logits, 3.0e-5F);
        CHECK(actual.greedy_token_id == expected.greedy_token_id);
        CHECK(actual.mtp_greedy_token_id == expected.mtp_greedy_token_id);
        const auto compare_layers = [&](const auto& actual_layers,
                                        const auto& expected_layers) {
            CHECK(actual_layers.size() == expected_layers.size());
            for (std::size_t layer = 0U;
                 layer < std::min(actual_layers.size(), expected_layers.size()); ++layer) {
                CHECK(actual_layers[layer].route.indices ==
                      expected_layers[layer].route.indices);
                compare_packed_values(actual_layers[layer].route.weights,
                                      expected_layers[layer].route.weights,
                                      3.0e-5F);
                CHECK(actual_layers[layer].attention_indices ==
                      expected_layers[layer].attention_indices);
            }
        };
        compare_layers(actual.layers, expected.layers);
        compare_layers(actual.mtp_layers, expected.mtp_layers);
    };

    if (packed_ok) {
        vastloom::inference::DeepseekV4ReferenceState oracle_packed_state;
        vastloom::inference::DeepseekV4ReferenceState packed_state;
        auto oracle_packed = vastloom::inference::deepseek_v4_reference_prefill(
            quantized_oracle, token_ids, &oracle_packed_state);
        auto packed = vastloom::inference::deepseek_v4_reference_prefill(
            packed_graph, token_ids, &packed_state);
        CHECK(oracle_packed && packed);
        if (oracle_packed && packed) {
            compare_packed_forward(packed.value(), oracle_packed.value());
            CHECK(packed_state.next_position == oracle_packed_state.next_position);
            auto oracle_decoded = vastloom::inference::deepseek_v4_reference_decode(
                quantized_oracle, 10U, oracle_packed_state);
            auto packed_decoded = vastloom::inference::deepseek_v4_reference_decode(
                packed_graph, 10U, packed_state);
            CHECK(oracle_decoded && packed_decoded);
            if (oracle_decoded && packed_decoded) {
                compare_packed_forward(packed_decoded.value(), oracle_decoded.value());
            }

            auto truncated_graph = packed_graph;
            truncated_graph.packed_matrices.begin()->second.encoded.pop_back();
            const auto state_before_truncation = packed_state;
            CHECK(!vastloom::inference::deepseek_v4_reference_decode(
                truncated_graph, 12U, packed_state));
            CHECK(packed_state.next_position == state_before_truncation.next_position);
            CHECK(packed_state.layers[0].window_key_values ==
                  state_before_truncation.layers[0].window_key_values);

            auto intact_iq_state = packed_state;
            auto intact_iq = vastloom::inference::deepseek_v4_reference_decode(
                packed_graph, 12U, intact_iq_state);
            CHECK(intact_iq &&
                  std::find(intact_iq.value().layers[0].route.indices.begin(),
                            intact_iq.value().layers[0].route.indices.end(), 0U) !=
                      intact_iq.value().layers[0].route.indices.end());
            auto truncated_iq = packed_graph;
            truncated_iq.packed_matrices.at(
                "layers.0.ffn.experts.0.w1.weight").encoded.pop_back();
            const auto state_before_iq_truncation = packed_state;
            CHECK(!vastloom::inference::deepseek_v4_reference_decode(
                truncated_iq, 12U, packed_state));
            CHECK(packed_state.next_position ==
                      state_before_iq_truncation.next_position &&
                  packed_state.layers[0].window_key_values ==
                      state_before_iq_truncation.layers[0].window_key_values);

            auto malformed_descriptor = packed_graph;
            malformed_descriptor.packed_matrices.begin()->second.logical_columns += 1U;
            CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                malformed_descriptor, token_ids));

            auto negative_route = packed_graph;
            auto& route_bytes =
                negative_route.packed_integer_tensors.at(hash_table_name).encoded;
            std::fill_n(route_bytes.begin(), 4U, std::byte{0xffU});
            CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                negative_route, token_ids));

            auto truncated_route = packed_graph;
            truncated_route.packed_integer_tensors.at(hash_table_name).encoded.pop_back();
            CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                truncated_route, token_ids));
        }

        // Exercise the alternate IQ2_XXS gate/up variant admitted by the official profile.
        // Replacing every routed W1/W3 matrix guarantees that every selected route traverses
        // IQ2_XXS; the independently dequantized graph remains the numerical oracle.
        auto iq2_graph = packed_graph;
        auto iq2_oracle = quantized_oracle;
        bool iq2_graph_ok = true;
        for (std::size_t expert = 0U; expert < config.expert_count; ++expert) {
            for (const std::string_view weight : {"w1", "w3"}) {
                const std::string name =
                    "layers.0.ffn.experts." + std::to_string(expert) + "." +
                    std::string(weight) + ".weight";
                const auto spec = std::find_if(
                    layer_matrices.begin(), layer_matrices.end(),
                    [&name](const MatrixSpec& candidate) {
                        return candidate.name == name;
                    });
                CHECK(spec != layer_matrices.end());
                if (spec == layer_matrices.end()) {
                    iq2_graph_ok = false;
                    break;
                }
                auto encoded = make_iq2_xxs(
                    spec->rows, expert * 2U + (weight == "w3" ? 2U : 1U));
                std::vector<float> padded(
                    spec->rows * vastloom::inference::kIq2XxsBlockElements);
                auto decoded = vastloom::inference::dequantize_iq2_xxs(
                    encoded, padded);
                CHECK(decoded);
                if (!decoded) {
                    iq2_graph_ok = false;
                    break;
                }
                std::vector<float> logical(spec->rows * spec->columns);
                for (std::size_t row = 0U; row < spec->rows; ++row) {
                    std::copy_n(
                        padded.begin() + static_cast<std::ptrdiff_t>(
                            row * vastloom::inference::kIq2XxsBlockElements),
                        spec->columns,
                        logical.begin() + static_cast<std::ptrdiff_t>(
                            row * spec->columns));
                }
                iq2_oracle.float_tensors[name] = std::move(logical);
                iq2_graph.packed_matrices[name] =
                    vastloom::inference::DeepseekV4PackedMatrix{
                        vastloom::inference::kGgmlTypeIq2Xxs,
                        spec->rows, spec->columns,
                        vastloom::inference::kIq2XxsBlockElements,
                        std::move(encoded)};
            }
            if (!iq2_graph_ok) break;
        }
        if (iq2_graph_ok) {
            vastloom::inference::DeepseekV4ReferenceState iq2_oracle_state;
            vastloom::inference::DeepseekV4ReferenceState iq2_state;
            auto expected_iq2 = vastloom::inference::deepseek_v4_reference_prefill(
                iq2_oracle, token_ids, &iq2_oracle_state);
            auto actual_iq2 = vastloom::inference::deepseek_v4_reference_prefill(
                iq2_graph, token_ids, &iq2_state);
            CHECK(expected_iq2 && actual_iq2);
            if (expected_iq2 && actual_iq2) {
                compare_packed_forward(actual_iq2.value(), expected_iq2.value());
                auto expected_iq2_decode =
                    vastloom::inference::deepseek_v4_reference_decode(
                        iq2_oracle, 10U, iq2_oracle_state);
                auto actual_iq2_decode =
                    vastloom::inference::deepseek_v4_reference_decode(
                        iq2_graph, 10U, iq2_state);
                CHECK(expected_iq2_decode && actual_iq2_decode);
                if (expected_iq2_decode && actual_iq2_decode) {
                    compare_packed_forward(actual_iq2_decode.value(),
                                           expected_iq2_decode.value());
                }
            }
            auto corrupt_iq2 = iq2_graph;
            for (auto& entry : corrupt_iq2.packed_matrices) {
                auto& matrix = entry.second;
                if (matrix.ggml_type == vastloom::inference::kGgmlTypeIq2Xxs) {
                    matrix.encoded[0] = std::byte{0x00U};
                    matrix.encoded[1] = std::byte{0x7cU};
                }
            }
            CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                corrupt_iq2, token_ids));
        }

        // Exercise the official profile's alternate MXFP4 down projections. Every routed W2
        // is replaced, so every selected expert must cross the new kernel. The float graph is
        // built by decoding the exact packed bytes before graph execution.
        auto mxfp4_graph = packed_graph;
        auto mxfp4_oracle = quantized_oracle;
        bool mxfp4_graph_ok = true;
        for (std::size_t expert = 0U; expert < config.expert_count; ++expert) {
            const std::string name =
                "layers.0.ffn.experts." + std::to_string(expert) +
                ".w2.weight";
            const auto spec = std::find_if(
                layer_matrices.begin(), layer_matrices.end(),
                [&name](const MatrixSpec& candidate) {
                    return candidate.name == name;
                });
            CHECK(spec != layer_matrices.end());
            if (spec == layer_matrices.end()) {
                mxfp4_graph_ok = false;
                break;
            }
            auto encoded = make_mxfp4(spec->rows, expert + 1U);
            std::vector<float> padded(
                spec->rows * vastloom::inference::kMxfp4BlockElements);
            auto decoded = vastloom::inference::dequantize_mxfp4(
                encoded, padded);
            CHECK(decoded);
            if (!decoded) {
                mxfp4_graph_ok = false;
                break;
            }
            std::vector<float> logical(spec->rows * spec->columns);
            for (std::size_t row = 0U; row < spec->rows; ++row) {
                std::copy_n(
                    padded.begin() + static_cast<std::ptrdiff_t>(
                        row * vastloom::inference::kMxfp4BlockElements),
                    spec->columns,
                    logical.begin() + static_cast<std::ptrdiff_t>(
                        row * spec->columns));
            }
            mxfp4_oracle.float_tensors[name] = std::move(logical);
            mxfp4_graph.packed_matrices[name] =
                vastloom::inference::DeepseekV4PackedMatrix{
                    vastloom::inference::kGgmlTypeMxfp4,
                    spec->rows, spec->columns,
                    vastloom::inference::kMxfp4BlockElements,
                    std::move(encoded)};
        }
        if (mxfp4_graph_ok) {
            vastloom::inference::DeepseekV4ReferenceState mxfp4_oracle_state;
            vastloom::inference::DeepseekV4ReferenceState mxfp4_state;
            auto expected_mxfp4 = vastloom::inference::deepseek_v4_reference_prefill(
                mxfp4_oracle, token_ids, &mxfp4_oracle_state);
            auto actual_mxfp4 = vastloom::inference::deepseek_v4_reference_prefill(
                mxfp4_graph, token_ids, &mxfp4_state);
            CHECK(expected_mxfp4 && actual_mxfp4);
            if (expected_mxfp4 && actual_mxfp4) {
                compare_packed_forward(actual_mxfp4.value(),
                                       expected_mxfp4.value());
                auto expected_mxfp4_decode =
                    vastloom::inference::deepseek_v4_reference_decode(
                        mxfp4_oracle, 10U, mxfp4_oracle_state);
                auto actual_mxfp4_decode =
                    vastloom::inference::deepseek_v4_reference_decode(
                        mxfp4_graph, 10U, mxfp4_state);
                CHECK(expected_mxfp4_decode && actual_mxfp4_decode);
                if (expected_mxfp4_decode && actual_mxfp4_decode) {
                    compare_packed_forward(actual_mxfp4_decode.value(),
                                           expected_mxfp4_decode.value());
                }
            }
            auto corrupt_mxfp4 = mxfp4_graph;
            for (auto& entry : corrupt_mxfp4.packed_matrices) {
                auto& matrix = entry.second;
                if (matrix.ggml_type == vastloom::inference::kGgmlTypeMxfp4) {
                    matrix.encoded[0] = std::byte{0xfdU};
                    matrix.encoded[1] = std::byte{0x77U};
                }
            }
            CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                corrupt_mxfp4, token_ids));
        }

        // Persist the exact packed fixture, load trunk tensors as complete bounded tiles and
        // routed tensors as indivisible expert bundles, then let the graph retain only the
        // Governor-owned immutable buffers. The loader result wrappers intentionally die before
        // inference to prove that the graph itself owns the accounting lifetime.
        const auto packed_path =
            root / std::filesystem::path(std::u8string(u8"deepseek-packed-àccènt.bin"));
        vastloom::model::LogicalTensorCatalog packed_catalog;
        packed_catalog.model_id = "test/deepseek-v4-packed-layer";
        packed_catalog.revision = "fixture-q8-0-iq1-s-iq3-xxs";
        vastloom::ByteCount file_offset = 0U;
        bool file_fixture_ok = true;
        {
            std::ofstream output(packed_path, std::ios::binary | std::ios::trunc);
            const auto append = [&](std::span<const std::byte> bytes) {
                if (!output || bytes.empty() ||
                    bytes.size() > static_cast<std::size_t>(
                                       std::numeric_limits<std::streamsize>::max())) {
                    return false;
                }
                output.write(reinterpret_cast<const char*>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()));
                return static_cast<bool>(output);
            };
            for (const auto& spec : layer_matrices) {
                const auto packed_matrix = packed_graph.packed_matrices.find(spec.name);
                CHECK(packed_matrix != packed_graph.packed_matrices.end());
                if (packed_matrix == packed_graph.packed_matrices.end() ||
                    !append(packed_matrix->second.encoded)) {
                    file_fixture_ok = false;
                    break;
                }
                vastloom::model::LogicalTensorDescriptor descriptor;
                descriptor.name = spec.name;
                descriptor.dtype = vastloom::model::TensorDType::quantized;
                descriptor.shape = {
                    static_cast<std::uint64_t>(spec.storage_columns),
                    static_cast<std::uint64_t>(spec.rows)};
                descriptor.role = spec.role;
                descriptor.layer = 0U;
                descriptor.expert = spec.expert;
                descriptor.source = vastloom::model::TensorSourceRange{
                    packed_path, file_offset,
                    static_cast<vastloom::ByteCount>(packed_matrix->second.encoded.size())};
                descriptor.ggml_type = spec.ggml_type;
                packed_catalog.tensors.push_back(std::move(descriptor));
                file_offset += static_cast<vastloom::ByteCount>(
                    packed_matrix->second.encoded.size());
            }
            if (file_fixture_ok) {
                const auto route =
                    packed_graph.packed_integer_tensors.find(hash_table_name);
                CHECK(route != packed_graph.packed_integer_tensors.end());
                if (route == packed_graph.packed_integer_tensors.end() ||
                    !append(route->second.encoded)) {
                    file_fixture_ok = false;
                } else {
                    vastloom::model::LogicalTensorDescriptor descriptor;
                    descriptor.name = hash_table_name;
                    descriptor.dtype = vastloom::model::TensorDType::i32;
                    descriptor.shape = {
                        static_cast<std::uint64_t>(route->second.elements)};
                    descriptor.role = vastloom::model::TensorRole::router;
                    descriptor.layer = 0U;
                    descriptor.source = vastloom::model::TensorSourceRange{
                        packed_path, file_offset,
                        static_cast<vastloom::ByteCount>(route->second.encoded.size())};
                    descriptor.ggml_type = vastloom::inference::kGgmlTypeI32;
                    packed_catalog.tensors.push_back(std::move(descriptor));
                    file_offset += static_cast<vastloom::ByteCount>(
                        route->second.encoded.size());
                }
            }
            output.flush();
            file_fixture_ok = file_fixture_ok && static_cast<bool>(output);
        }
        CHECK(file_fixture_ok && file_offset == 8'180U);
        auto finalized = vastloom::model::finalize_logical_tensor_catalog(packed_catalog);
        CHECK(finalized && packed_catalog.tensor_bytes == file_offset &&
              packed_catalog.experts.size() == config.expert_count &&
              packed_catalog.routed_expert_bytes == 5'536U &&
              packed_catalog.always_active_bytes == 2'644U);
        auto file_plan = vastloom::model::build_layer_streaming_plan(
            packed_catalog, 0U, 16U * vastloom::KiB);
        const std::size_t routed_matrix_count = config.expert_count * 3U;
        CHECK(file_plan &&
              file_plan.value().tensor_count ==
                  layer_matrices.size() - routed_matrix_count + 1U &&
              file_plan.value().tiles.size() == file_plan.value().tensor_count);

        // The legacy per-object streaming fixture above remains byte-for-byte stable. Extend a
        // separate catalog over the same immutable file with every F32 vector consumed by layer
        // zero so the aggregate artifact path can prove a complete 21-object trunk boundary.
        auto artifact_catalog = packed_catalog;
        std::map<std::string, std::vector<std::byte>, std::less<>> artifact_float_bytes;
        vastloom::ByteCount artifact_file_offset = file_offset;
        bool artifact_catalog_ok = true;
        for (auto& descriptor : artifact_catalog.tensors) {
            if (descriptor.role == vastloom::model::TensorRole::routed_expert ||
                descriptor.layer != 0U) {
                continue;
            }
            const auto canonical = layer0_gguf_names.find(descriptor.name);
            CHECK(canonical != layer0_gguf_names.end());
            if (canonical == layer0_gguf_names.end()) {
                artifact_catalog_ok = false;
                break;
            }
            descriptor.name = canonical->second;
        }
        {
            std::ofstream output(packed_path, std::ios::binary | std::ios::app);
            for (const auto& spec : layer_float_vectors) {
                const auto found = graph.float_tensors.find(spec.name);
                CHECK(found != graph.float_tensors.end() && !found->second.empty());
                if (!output || found == graph.float_tensors.end() || found->second.empty()) {
                    artifact_catalog_ok = false;
                    break;
                }
                std::vector<std::byte> encoded(found->second.size() *
                                               vastloom::inference::kF32ElementBytes);
                for (std::size_t index = 0U; index < found->second.size(); ++index) {
                    const auto bits = std::bit_cast<std::uint32_t>(found->second[index]);
                    for (unsigned byte = 0U; byte < 4U; ++byte) {
                        encoded[index * 4U + byte] = static_cast<std::byte>(
                            (bits >> (byte * 8U)) & 0xffU);
                    }
                }
                output.write(reinterpret_cast<const char*>(encoded.data()),
                             static_cast<std::streamsize>(encoded.size()));
                if (!output) {
                    artifact_catalog_ok = false;
                    break;
                }
                vastloom::model::LogicalTensorDescriptor descriptor;
                descriptor.name = layer0_gguf_names.at(spec.name);
                descriptor.dtype = vastloom::model::TensorDType::f32;
                descriptor.shape = {
                    static_cast<std::uint64_t>(found->second.size())};
                descriptor.role = spec.role;
                descriptor.layer = 0U;
                descriptor.source = vastloom::model::TensorSourceRange{
                    packed_path, artifact_file_offset,
                    static_cast<vastloom::ByteCount>(encoded.size())};
                descriptor.ggml_type = vastloom::inference::kGgmlTypeF32;
                artifact_catalog.tensors.push_back(std::move(descriptor));
                artifact_file_offset += static_cast<vastloom::ByteCount>(encoded.size());
                artifact_float_bytes.emplace(
                    layer0_gguf_names.at(spec.name), std::move(encoded));
            }
            output.flush();
            artifact_catalog_ok = artifact_catalog_ok && static_cast<bool>(output);
        }
        auto artifact_finalized = artifact_catalog_ok
            ? vastloom::model::finalize_logical_tensor_catalog(artifact_catalog)
            : vastloom::Result<void>(vastloom::make_error(
                  vastloom::ErrorCode::internal_error,
                  "unable to build packed float-vector fixture"));
        CHECK(artifact_catalog_ok && artifact_finalized &&
              artifact_float_bytes.size() == layer_float_vectors.size() &&
              artifact_file_offset == 8'380U &&
              artifact_catalog.tensor_bytes == 8'380U &&
              artifact_catalog.routed_expert_bytes == 5'536U &&
              artifact_catalog.always_active_bytes == 2'844U);

        if (file_fixture_ok && finalized && file_plan && artifact_catalog_ok &&
            artifact_finalized && !file_plan.value().tiles.empty()) {
            vastloom::model::TensorTileLoadOptions tile_options;
            tile_options.max_tile_bytes = 16U * vastloom::KiB;
            tile_options.measure_process_io = false;

            const auto& probe_tile = file_plan.value().tiles.front();
            vastloom::safety::ResourceLimits refused_limits;
            refused_limits.host_memory_hard_limit = probe_tile.source.bytes * 2U - 1U;
            refused_limits.ram_hard_limit = probe_tile.source.bytes;
            refused_limits.vram_hard_limit = 1U;
            refused_limits.staging_hard_limit = probe_tile.source.bytes;
            refused_limits.scratch_hard_limit = 1U;
            vastloom::safety::ResourceGovernor refused_governor(refused_limits);
            auto refused_residency = vastloom::model::ResidencyObject::create(
                "deepseek/refused/trunk", probe_tile.source.bytes);
            CHECK(refused_residency &&
                  !vastloom::model::load_tensor_tile(
                      packed_catalog, probe_tile, refused_residency.value(),
                      refused_governor, tile_options));
            CHECK(refused_governor.snapshot().reserved[0] == 0U &&
                  refused_governor.snapshot().reserved[2] == 0U);

            vastloom::safety::ResourceLimits resident_limits;
            resident_limits.host_memory_hard_limit = 64U * vastloom::KiB;
            resident_limits.ram_hard_limit = 32U * vastloom::KiB;
            resident_limits.vram_hard_limit = 1U;
            resident_limits.staging_hard_limit = 16U * vastloom::KiB;
            resident_limits.scratch_hard_limit = 16U * vastloom::KiB;
            vastloom::safety::ResourceGovernor governor(resident_limits);

            unsigned cancellation_checks = 0U;
            auto cancelled_options = tile_options;
            cancelled_options.cancelled = [&] { return ++cancellation_checks >= 2U; };
            auto cancelled_residency = vastloom::model::ResidencyObject::create(
                "deepseek/cancelled/trunk", probe_tile.source.bytes);
            CHECK(cancelled_residency &&
                  !vastloom::model::load_tensor_tile(
                      packed_catalog, probe_tile, cancelled_residency.value(),
                      governor, cancelled_options));
            CHECK(cancelled_residency && cancelled_residency.value().snapshot() &&
                  cancelled_residency.value().snapshot().value().state ==
                      vastloom::model::ResidencyState::failed &&
                  governor.snapshot().reserved[0] == 0U &&
                  governor.snapshot().reserved[2] == 0U);

            const auto find_spec = [&](std::string_view name) -> const MatrixSpec* {
                const auto found = std::find_if(
                    layer_matrices.begin(), layer_matrices.end(),
                    [name](const MatrixSpec& spec) { return spec.name == name; });
                return found == layer_matrices.end() ? nullptr : &*found;
            };

            auto governed_graph = graph;
            std::vector<vastloom::model::ResidencyObject> resident_objects;
            resident_objects.reserve(file_plan.value().tiles.size() +
                                     packed_catalog.experts.size());
            bool bound_ok = true;
            for (const auto& tile : file_plan.value().tiles) {
                auto residency = vastloom::model::ResidencyObject::create(
                    "deepseek/trunk/" + std::to_string(tile.tensor_index),
                    tile.source.bytes);
                CHECK(residency);
                if (!residency) {
                    bound_ok = false;
                    break;
                }
                resident_objects.push_back(residency.take_value());
                auto loaded = vastloom::model::load_tensor_tile(
                    packed_catalog, tile, resident_objects.back(), governor,
                    tile_options);
                CHECK(loaded);
                if (!loaded) {
                    bound_ok = false;
                    break;
                }
                const auto& descriptor = packed_catalog.tensors[tile.tensor_index];
                if (descriptor.ggml_type == vastloom::inference::kGgmlTypeI32) {
                    auto bound = vastloom::inference::deepseek_v4_bind_packed_i32(
                        loaded.value(), descriptor.shape.front());
                    CHECK(bound);
                    if (!bound) {
                        bound_ok = false;
                        break;
                    }
                    governed_graph.packed_integer_tensors.emplace(
                        descriptor.name, bound.take_value());
                    governed_graph.integer_tensors.erase(descriptor.name);
                } else {
                    const auto* spec = find_spec(descriptor.name);
                    CHECK(spec != nullptr && descriptor.ggml_type.has_value());
                    if (spec == nullptr || !descriptor.ggml_type.has_value()) {
                        bound_ok = false;
                        break;
                    }
                    auto wrong_shape =
                        vastloom::inference::deepseek_v4_bind_packed_matrix(
                            loaded.value(), *descriptor.ggml_type,
                            spec->rows + 1U,
                            spec->columns, spec->storage_columns);
                    CHECK(!wrong_shape);
                    auto bound = vastloom::inference::deepseek_v4_bind_packed_matrix(
                        loaded.value(), *descriptor.ggml_type, spec->rows,
                        spec->columns, spec->storage_columns);
                    CHECK(bound);
                    if (!bound) {
                        bound_ok = false;
                        break;
                    }
                    governed_graph.packed_matrices.emplace(
                        descriptor.name, bound.take_value());
                    governed_graph.float_tensors.erase(descriptor.name);
                }
            }

            vastloom::model::ExpertLoadOptions expert_options;
            expert_options.max_expert_bytes = 16U * vastloom::KiB;
            expert_options.max_range_bytes = 16U * vastloom::KiB;
            expert_options.max_ranges = 4U;
            expert_options.measure_process_io = false;
            for (const auto& bundle : packed_catalog.experts) {
                if (!bound_ok) break;
                auto residency = vastloom::model::ResidencyObject::create(
                    "deepseek/expert/" + std::to_string(bundle.expert),
                    bundle.bytes);
                CHECK(residency);
                if (!residency) {
                    bound_ok = false;
                    break;
                }
                resident_objects.push_back(residency.take_value());
                auto loaded = vastloom::model::load_expert_bundle(
                    bundle, resident_objects.back(), governor, expert_options);
                CHECK(loaded);
                if (!loaded) {
                    bound_ok = false;
                    break;
                }
                CHECK(!vastloom::inference::deepseek_v4_bind_packed_matrix(
                    loaded.value(), packed_catalog.tensors.size() + 1U,
                    vastloom::inference::kGgmlTypeQ8_0, 1U, 1U,
                    vastloom::inference::kQ8_0BlockElements));
                for (const auto tensor_index : bundle.tensor_indices) {
                    const auto& descriptor = packed_catalog.tensors[tensor_index];
                    const auto* spec = find_spec(descriptor.name);
                    CHECK(spec != nullptr && descriptor.ggml_type.has_value());
                    if (spec == nullptr || !descriptor.ggml_type.has_value()) {
                        bound_ok = false;
                        break;
                    }
                    auto bound = vastloom::inference::deepseek_v4_bind_packed_matrix(
                        loaded.value(), tensor_index, *descriptor.ggml_type,
                        spec->rows, spec->columns, spec->storage_columns);
                    CHECK(bound);
                    if (!bound) {
                        bound_ok = false;
                        break;
                    }
                    governed_graph.packed_matrices.emplace(
                        descriptor.name, bound.take_value());
                    governed_graph.float_tensors.erase(descriptor.name);
                }
            }

            CHECK(bound_ok && governed_graph.packed_matrices.size() == 23U &&
                  governed_graph.packed_integer_tensors.size() == 1U &&
                  governor.snapshot().reserved[0] == file_offset &&
                  governor.snapshot().reserved[2] == 0U);
            for (auto& residency : resident_objects) {
                auto snapshot = residency.snapshot();
                CHECK(snapshot && snapshot.value().active_uses == 1U &&
                      !snapshot.value().evictable && !residency.evict_ram());
            }

            if (bound_ok) {
                vastloom::inference::DeepseekV4ReferenceState oracle_file_state;
                vastloom::inference::DeepseekV4ReferenceState governed_state;
                auto oracle_file = vastloom::inference::deepseek_v4_reference_prefill(
                    quantized_oracle, token_ids, &oracle_file_state);
                auto governed = vastloom::inference::deepseek_v4_reference_prefill(
                    governed_graph, token_ids, &governed_state);
                CHECK(oracle_file && governed);
                if (oracle_file && governed) {
                    compare_packed_forward(governed.value(), oracle_file.value());
                    auto oracle_file_decode =
                        vastloom::inference::deepseek_v4_reference_decode(
                            quantized_oracle, 10U, oracle_file_state);
                    auto governed_decode =
                        vastloom::inference::deepseek_v4_reference_decode(
                            governed_graph, 10U, governed_state);
                    CHECK(oracle_file_decode && governed_decode);
                    if (oracle_file_decode && governed_decode) {
                        compare_packed_forward(governed_decode.value(),
                                               oracle_file_decode.value());
                    }

                    {
                        auto invalid_range = governed_graph;
                        auto& matrix = invalid_range.packed_matrices.begin()->second;
                        matrix.resident_offset = matrix.resident->bytes().size();
                        const auto before = governed_state;
                        CHECK(!vastloom::inference::deepseek_v4_reference_decode(
                            invalid_range, 12U, governed_state));
                        CHECK(governed_state.next_position == before.next_position &&
                              governed_state.layers[0].window_key_values ==
                                  before.layers[0].window_key_values);
                    }
                    {
                        auto ambiguous = governed_graph;
                        ambiguous.packed_matrices.begin()->second.encoded.push_back(
                            std::byte{0U});
                        CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                            ambiguous, token_ids));
                    }
                }
            }

            // Repackage the complete non-routed layer tensors as independently hashed artifact
            // ranges, load them into one aggregate-governed buffer and bind the reduced graph
            // directly from that retained tensor set. Routed experts remain indivisible bundles.
            const auto trunk_artifact_root = root / "deepseek-packed-trunk-artifact";
            std::filesystem::create_directories(trunk_artifact_root / "ranges");
            vastloom::inference::DeepseekV4TrunkArtifactBinding trunk_binding;
            trunk_binding.artifact_root = trunk_artifact_root;
            trunk_binding.model_id = artifact_catalog.model_id;
            trunk_binding.revision = artifact_catalog.revision;
            trunk_binding.source_plan_sha256 = std::string(64U, 'a');
            trunk_binding.layer = 0U;
            trunk_binding.metadata_catalog_hashes_verified = true;
            trunk_binding.local_range_hashes_verified = true;
            trunk_binding.exact_source_binding_verified = true;
            bool trunk_artifact_ok = true;
            for (std::size_t tensor_index = 0U;
                 tensor_index < artifact_catalog.tensors.size(); ++tensor_index) {
                const auto& descriptor = artifact_catalog.tensors[tensor_index];
                if (descriptor.role == vastloom::model::TensorRole::routed_expert ||
                    !descriptor.layer.has_value() || *descriptor.layer != 0U) {
                    continue;
                }
                const auto consumer = std::find_if(
                    layer0_gguf_names.begin(), layer0_gguf_names.end(),
                    [&](const auto& mapping) {
                        return mapping.second == descriptor.name;
                    });
                CHECK(consumer != layer0_gguf_names.end());
                if (consumer == layer0_gguf_names.end()) {
                    trunk_artifact_ok = false;
                    break;
                }
                const std::vector<std::byte>* encoded = nullptr;
                if (descriptor.ggml_type == vastloom::inference::kGgmlTypeI32) {
                    const auto found =
                        packed_graph.packed_integer_tensors.find(consumer->first);
                    if (found != packed_graph.packed_integer_tensors.end()) {
                        encoded = &found->second.encoded;
                    }
                } else {
                    const auto found = packed_graph.packed_matrices.find(consumer->first);
                    if (found != packed_graph.packed_matrices.end()) {
                        encoded = &found->second.encoded;
                    } else {
                        const auto float_found = artifact_float_bytes.find(descriptor.name);
                        if (float_found != artifact_float_bytes.end()) {
                            encoded = &float_found->second;
                        }
                    }
                }
                CHECK(encoded != nullptr && !encoded->empty() &&
                      encoded->size() == descriptor.source.bytes);
                if (encoded == nullptr || encoded->empty() ||
                    encoded->size() != descriptor.source.bytes) {
                    trunk_artifact_ok = false;
                    break;
                }
                const auto range_index = trunk_binding.ranges.size();
                const auto local_path = trunk_artifact_root / "ranges" /
                                        (std::to_string(range_index) + ".bin");
                write_bytes(local_path, *encoded);
                auto local_hash = vastloom::util::sha256_file(local_path);
                CHECK(local_hash);
                if (!local_hash) {
                    trunk_artifact_ok = false;
                    break;
                }
                trunk_binding.ranges.push_back(
                    vastloom::inference::DeepseekV4BoundTrunkRange{
                        range_index,
                        descriptor.name,
                        descriptor.role,
                        0U,
                        true,
                        descriptor.ggml_type,
                        descriptor.source,
                        vastloom::model::TensorSourceRange{
                            local_path, 0U, descriptor.source.bytes},
                        local_hash.take_value(),
                    });
                trunk_binding.payload_bytes += descriptor.source.bytes;
                trunk_binding.maximum_local_range_bytes = std::max(
                    trunk_binding.maximum_local_range_bytes,
                    descriptor.source.bytes);
            }
            vastloom::inference::DeepseekV4GgufCatalogReport trunk_report;
            trunk_report.catalog = artifact_catalog;
            trunk_report.layers = 1U;
            trunk_report.experts_per_layer =
                static_cast<std::uint32_t>(config.expert_count);
            trunk_report.top_k = static_cast<std::uint32_t>(config.experts_per_token);
            trunk_report.imported_manifest_verified = true;
            trunk_report.imported_file_hashes_verified = true;
            auto reference_report = trunk_report;
            const auto& pinned_0731 =
                vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
            reference_report.catalog.model_id = pinned_0731.model_id;
            reference_report.catalog.revision = pinned_0731.revision;
            reference_report.layers = static_cast<std::uint32_t>(config.layer_count);
            reference_report.experts_per_layer =
                static_cast<std::uint32_t>(config.expert_count);
            reference_report.top_k =
                static_cast<std::uint32_t>(config.experts_per_token);
            reference_report.pinned_0731_checkpoint_spec_verified = true;
            vastloom::inference::DeepseekV4GgufArchitectureConfig architecture;
            architecture.vocabulary_size =
                static_cast<std::uint32_t>(config.vocabulary_size);
            architecture.maximum_context_tokens = 128U;
            architecture.width = static_cast<std::uint32_t>(config.width);
            architecture.layer_count =
                static_cast<std::uint32_t>(config.layer_count);
            architecture.hash_layer_count =
                static_cast<std::uint32_t>(config.hash_layer_count);
            architecture.head_count = static_cast<std::uint32_t>(config.head_count);
            architecture.key_value_head_count = 1U;
            architecture.expert_count =
                static_cast<std::uint32_t>(config.expert_count);
            architecture.experts_per_token =
                static_cast<std::uint32_t>(config.experts_per_token);
            architecture.expert_gating_function = 4U;
            architecture.shared_expert_count = 1U;
            architecture.intermediate_width =
                static_cast<std::uint32_t>(config.intermediate_width);
            architecture.query_lora_rank =
                static_cast<std::uint32_t>(config.query_lora_rank);
            architecture.head_width = static_cast<std::uint32_t>(config.head_width);
            architecture.rope_width = static_cast<std::uint32_t>(config.rope_width);
            architecture.output_groups =
                static_cast<std::uint32_t>(config.output_groups);
            architecture.output_lora_rank =
                static_cast<std::uint32_t>(config.output_lora_rank);
            architecture.window_size = static_cast<std::uint32_t>(config.window_size);
            architecture.original_context_tokens =
                static_cast<std::uint32_t>(config.original_context);
            architecture.index_head_count =
                static_cast<std::uint32_t>(config.index_head_count);
            architecture.index_head_width =
                static_cast<std::uint32_t>(config.index_head_width);
            architecture.index_top_k =
                static_cast<std::uint32_t>(config.index_top_k);
            architecture.hyper_multiplier =
                static_cast<std::uint32_t>(config.hyper_multiplier);
            architecture.sinkhorn_iterations =
                static_cast<std::uint32_t>(config.sinkhorn_iterations);
            architecture.compression_rope_theta = config.compression_rope_theta;
            architecture.rope_theta = config.rope_theta;
            architecture.rope_factor = config.rope_factor;
            architecture.beta_fast = config.beta_fast;
            architecture.beta_slow = config.beta_slow;
            architecture.norm_epsilon = config.norm_epsilon;
            architecture.hyper_epsilon = config.hyper_epsilon;
            architecture.route_scale = config.route_scale;
            architecture.expert_weights_normalized = true;
            architecture.rope_scaling_type = "yarn";
            for (const auto ratio : config.compression_ratios) {
                architecture.compression_ratios.push_back(
                    static_cast<std::uint32_t>(ratio));
            }
            architecture.compression_ratios.push_back(0U);
            architecture.routed_expert_swiglu_limits.assign(
                config.layer_count, config.swiglu_limit);
            architecture.shared_expert_swiglu_limits.assign(
                config.layer_count, config.swiglu_limit);
            reference_report.reference_architecture = architecture;
            auto derived_config = vastloom::inference::
                deepseek_v4_reference_config_from_0731_gguf(reference_report);
            CHECK(derived_config && derived_config.value().mtp_layer_count == 0U &&
                  derived_config.value().compression_ratios ==
                      config.compression_ratios &&
                  derived_config.value().routed_expert_swiglu_limits.size() ==
                      config.layer_count &&
                  derived_config.value().shared_expert_swiglu_limits.size() ==
                      config.layer_count &&
                  derived_config.value().swiglu_limit == 0.0F);
            if (derived_config) {
                CHECK(vastloom::inference::deepseek_v4_validate_0731_gguf_layer0_catalog(
                    reference_report, derived_config.value()));
                auto one_layer_report = reference_report;
                one_layer_report.layers = 1U;
                one_layer_report.reference_architecture->layer_count = 1U;
                one_layer_report.reference_architecture->hash_layer_count = 1U;
                one_layer_report.reference_architecture->compression_ratios = {
                    static_cast<std::uint32_t>(config.compression_ratios.front()), 0U};
                one_layer_report.reference_architecture->routed_expert_swiglu_limits = {
                    config.swiglu_limit};
                one_layer_report.reference_architecture->shared_expert_swiglu_limits = {
                    config.swiglu_limit};
                const auto one_layer_table = std::find_if(
                    one_layer_report.catalog.tensors.begin(),
                    one_layer_report.catalog.tensors.end(),
                    [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                        return tensor.name == "blk.0.ffn_gate_tid2eid.weight";
                    });
                CHECK(one_layer_table != one_layer_report.catalog.tensors.end());
                if (one_layer_table != one_layer_report.catalog.tensors.end()) {
                    one_layer_table->shape = {
                        static_cast<std::uint64_t>(config.experts_per_token),
                        static_cast<std::uint64_t>(config.vocabulary_size)};
                }
                auto one_layer_config = vastloom::inference::
                    deepseek_v4_reference_config_from_0731_gguf(one_layer_report);
                CHECK(one_layer_config);
                if (one_layer_config) {
                    auto all_layers = vastloom::inference::
                        deepseek_v4_validate_0731_gguf_layer_catalogs(
                            one_layer_report, one_layer_config.value());
                    CHECK(all_layers && all_layers.value().layers == 1U &&
                          all_layers.value().tensors == 21U &&
                          all_layers.value().bytes == 2'844U);
                    auto checkpoint_report = one_layer_report;
                    bool checkpoint_fixture_ok = true;
                    for (auto& tensor : checkpoint_report.catalog.tensors) {
                        if (tensor.role !=
                            vastloom::model::TensorRole::routed_expert) {
                            continue;
                        }
                        CHECK(tensor.layer == 0U && tensor.expert.has_value());
                        if (tensor.layer != 0U || !tensor.expert.has_value()) {
                            checkpoint_fixture_ok = false;
                            break;
                        }
                        const std::string prefix = "blk.0.";
                        const std::string suffix = ".weight.expert." +
                                                   std::to_string(*tensor.expert);
                        if (tensor.name.ends_with(".w1.weight")) {
                            tensor.name = prefix + "ffn_gate_exps" + suffix;
                        } else if (tensor.name.ends_with(".w2.weight")) {
                            tensor.name = prefix + "ffn_down_exps" + suffix;
                        } else if (tensor.name.ends_with(".w3.weight")) {
                            tensor.name = prefix + "ffn_up_exps" + suffix;
                        } else {
                            checkpoint_fixture_ok = false;
                            break;
                        }
                    }
                    const auto global_path =
                        packed_path.parent_path() / "checkpoint-globals.bin";
                    vastloom::ByteCount global_offset = 0U;
                    const auto append_global =
                        [&](std::string name,
                            vastloom::model::TensorDType dtype,
                            std::vector<std::uint64_t> shape,
                            vastloom::model::TensorRole role,
                            std::uint32_t ggml_type,
                            std::size_t rows,
                            std::size_t columns) {
                            auto bytes = vastloom::inference::ggml_storage_bytes(
                                ggml_type, rows, columns,
                                vastloom::inference::QuantizedKernelLimits{
                                    rows * columns});
                            CHECK(bytes);
                            if (!bytes) {
                                checkpoint_fixture_ok = false;
                                return;
                            }
                            checkpoint_report.catalog.tensors.push_back(
                                vastloom::model::LogicalTensorDescriptor{
                                    std::move(name), dtype, std::move(shape), role,
                                    std::nullopt, std::nullopt, false,
                                    vastloom::model::TensorSourceRange{
                                        global_path, global_offset, bytes.value()},
                                    ggml_type});
                            global_offset += bytes.value();
                        };
                    const std::size_t embedding_columns =
                        vastloom::inference::kQ8_0BlockElements;
                    append_global(
                        "token_embd.weight",
                        vastloom::model::TensorDType::quantized,
                        {embedding_columns, config.vocabulary_size},
                        vastloom::model::TensorRole::embedding,
                        vastloom::inference::kGgmlTypeQ8_0,
                        config.vocabulary_size, embedding_columns);
                    append_global(
                        "output.weight",
                        vastloom::model::TensorDType::quantized,
                        {embedding_columns, config.vocabulary_size},
                        vastloom::model::TensorRole::output_head,
                        vastloom::inference::kGgmlTypeQ8_0,
                        config.vocabulary_size, embedding_columns);
                    append_global(
                        "output_hc_base.weight",
                        vastloom::model::TensorDType::f32,
                        {config.hyper_multiplier},
                        vastloom::model::TensorRole::hyper_connection,
                        vastloom::inference::kGgmlTypeF32, 1U,
                        config.hyper_multiplier);
                    append_global(
                        "output_hc_fn.weight",
                        vastloom::model::TensorDType::f32,
                        {config.hyper_multiplier * config.width,
                         config.hyper_multiplier},
                        vastloom::model::TensorRole::hyper_connection,
                        vastloom::inference::kGgmlTypeF32,
                        config.hyper_multiplier,
                        config.hyper_multiplier * config.width);
                    append_global(
                        "output_hc_scale.weight",
                        vastloom::model::TensorDType::f32, {1U},
                        vastloom::model::TensorRole::hyper_connection,
                        vastloom::inference::kGgmlTypeF32, 1U, 1U);
                    append_global(
                        "output_norm.weight",
                        vastloom::model::TensorDType::f32, {config.width},
                        vastloom::model::TensorRole::normalization,
                        vastloom::inference::kGgmlTypeF32, 1U, config.width);
                    auto checkpoint_finalized = checkpoint_fixture_ok
                        ? vastloom::model::finalize_logical_tensor_catalog(
                              checkpoint_report.catalog)
                        : vastloom::Result<void>(vastloom::make_error(
                              vastloom::ErrorCode::internal_error,
                              "unable to build checkpoint closure fixture"));
                    checkpoint_report.physical_tensors = 21U + 6U + 3U;
                    CHECK(checkpoint_fixture_ok && checkpoint_finalized);
                    if (checkpoint_fixture_ok && checkpoint_finalized) {
                        auto checkpoint = vastloom::inference::
                            deepseek_v4_validate_0731_gguf_checkpoint_catalog(
                                checkpoint_report, one_layer_config.value());
                        CHECK(checkpoint && checkpoint.value().layers.layers == 1U &&
                              checkpoint.value().layers.tensors == 21U &&
                              checkpoint.value().global_tensors == 6U &&
                              checkpoint.value().global_bytes == global_offset &&
                              checkpoint.value().expert_bundles ==
                                  config.expert_count &&
                              checkpoint.value().expert_tensor_slices ==
                                  config.expert_count * 3U &&
                              checkpoint.value().physical_expert_tensors == 3U &&
                              checkpoint.value().logical_tensors == 39U &&
                              checkpoint.value().tensor_bytes ==
                                  checkpoint_report.catalog.tensor_bytes);

                        auto global_plan = vastloom::inference::
                            build_deepseek_v4_global_tensor_plan(checkpoint_report);
                        CHECK(global_plan && global_plan.value().tensor_count == 6U &&
                              global_plan.value().payload_bytes == global_offset &&
                              global_plan.value().tensors.front().tensor_name ==
                                  "token_embd.weight" &&
                              global_plan.value().tensors.back().tensor_name ==
                                  "output_hc_fn.weight" &&
                              vastloom::inference::to_json(global_plan.value()).find(
                                  "vastloom.deepseek_v4.global_tensor_plan.v1") !=
                                  std::string::npos);
                        if (checkpoint && global_plan) {
                            const std::map<std::string, std::string, std::less<>>
                                global_consumers{
                                    {"token_embd.weight", "embed.weight"},
                                    {"output.weight", "head.weight"},
                                    {"output_norm.weight", "norm.weight"},
                                    {"output_hc_base.weight", "hc_head_base"},
                                    {"output_hc_scale.weight", "hc_head_scale"},
                                    {"output_hc_fn.weight", "hc_head_fn"},
                                };
                            std::map<std::string, std::vector<std::byte>, std::less<>>
                                global_encoded;
                            bool global_payload_ok = true;
                            for (const auto& planned : global_plan.value().tensors) {
                                const auto consumer = global_consumers.find(
                                    planned.tensor_name);
                                const auto values = consumer == global_consumers.end()
                                    ? graph.float_tensors.end()
                                    : graph.float_tensors.find(consumer->second);
                                CHECK(consumer != global_consumers.end() &&
                                      values != graph.float_tensors.end());
                                if (consumer == global_consumers.end() ||
                                    values == graph.float_tensors.end()) {
                                    global_payload_ok = false;
                                    break;
                                }
                                if (planned.tensor_name == "token_embd.weight" ||
                                    planned.tensor_name == "output.weight") {
                                    auto encoded = vastloom::inference::
                                        quantize_q8_0_reference(
                                            values->second,
                                            config.vocabulary_size,
                                            config.width,
                                            embedding_columns);
                                    CHECK(encoded &&
                                          encoded.value().size() ==
                                              planned.source.bytes);
                                    if (!encoded || encoded.value().size() !=
                                                        planned.source.bytes) {
                                        global_payload_ok = false;
                                        break;
                                    }
                                    global_encoded.emplace(
                                        planned.tensor_name,
                                        encoded.take_value());
                                    continue;
                                }
                                std::vector<std::byte> encoded(
                                    values->second.size() *
                                    vastloom::inference::kF32ElementBytes);
                                for (std::size_t index = 0U;
                                     index < values->second.size(); ++index) {
                                    const auto bits = std::bit_cast<std::uint32_t>(
                                        values->second[index]);
                                    for (unsigned byte = 0U; byte < 4U; ++byte) {
                                        encoded[index * 4U + byte] =
                                            static_cast<std::byte>(
                                                (bits >> (byte * 8U)) & 0xffU);
                                    }
                                }
                                CHECK(encoded.size() == planned.source.bytes);
                                if (encoded.size() != planned.source.bytes) {
                                    global_payload_ok = false;
                                    break;
                                }
                                global_encoded.emplace(
                                    planned.tensor_name, std::move(encoded));
                            }

                            std::vector<std::byte> global_payload(
                                static_cast<std::size_t>(global_offset));
                            for (const auto& descriptor :
                                 checkpoint_report.catalog.tensors) {
                                if (descriptor.layer.has_value()) continue;
                                const auto encoded = global_encoded.find(
                                    descriptor.name);
                                CHECK(encoded != global_encoded.end() &&
                                      encoded->second.size() ==
                                          descriptor.source.bytes &&
                                      descriptor.source.offset <=
                                          global_payload.size() &&
                                      descriptor.source.bytes <=
                                          global_payload.size() -
                                              descriptor.source.offset);
                                if (encoded == global_encoded.end() ||
                                    encoded->second.size() !=
                                        descriptor.source.bytes ||
                                    descriptor.source.offset >
                                        global_payload.size() ||
                                    descriptor.source.bytes >
                                        global_payload.size() -
                                            descriptor.source.offset) {
                                    global_payload_ok = false;
                                    break;
                                }
                                std::copy(
                                    encoded->second.begin(), encoded->second.end(),
                                    global_payload.begin() +
                                        static_cast<std::ptrdiff_t>(
                                            descriptor.source.offset));
                            }
                            if (global_payload_ok) {
                                write_bytes(global_path, global_payload);
                            }
                            auto global_source_hash = global_payload_ok
                                ? vastloom::util::sha256_file(global_path)
                                : vastloom::Result<std::string>(
                                      vastloom::make_error(
                                          vastloom::ErrorCode::internal_error,
                                          "missing global fixture payload"));
                            CHECK(global_payload_ok && global_source_hash);

                            vastloom::inference::DeepseekV4GgufCheckpointSpec
                                global_spec;
                            global_spec.model_id = checkpoint_report.catalog.model_id;
                            global_spec.revision = checkpoint_report.catalog.revision;
                            global_spec.layers = 1U;
                            global_spec.experts_per_layer =
                                static_cast<std::uint32_t>(config.expert_count);
                            global_spec.top_k = static_cast<std::uint32_t>(
                                config.experts_per_token);
                            global_spec.shards = {{
                                "weights/checkpoint-globals.bin",
                                global_offset,
                                global_source_hash
                                    ? global_source_hash.value()
                                    : std::string(64U, '0'),
                                "headers/checkpoint-globals.header",
                                1U,
                                std::string(64U, 'd'),
                            }};
                            const auto global_artifact_root =
                                root / "deepseek-global-artifact";
                            std::filesystem::create_directories(
                                global_artifact_root / "ranges");
                            vastloom::model::RangeArtifact global_artifact;
                            global_artifact.root = global_artifact_root;
                            global_artifact.model_id = global_spec.model_id;
                            global_artifact.revision = global_spec.revision;
                            global_artifact.scope = "globals";
                            global_artifact.layer = 0U;
                            global_artifact.source_plan_format =
                                "vastloom.deepseek_v4.global_tensor_plan.v1";
                            global_artifact.source_plan_sha256 =
                                std::string(64U, 'c');
                            global_artifact.payload_bytes = global_offset;
                            global_artifact.full_hash_verified = true;
                            for (std::size_t index = 0U;
                                 index < global_plan.value().tensors.size();
                                 ++index) {
                                const auto& planned =
                                    global_plan.value().tensors[index];
                                const auto local_path =
                                    global_artifact_root / "ranges" /
                                    (std::to_string(index) + ".bin");
                                const auto encoded = global_encoded.find(
                                    planned.tensor_name);
                                CHECK(encoded != global_encoded.end());
                                if (encoded == global_encoded.end()) {
                                    global_payload_ok = false;
                                    break;
                                }
                                write_bytes(local_path, encoded->second);
                                auto local_hash =
                                    vastloom::util::sha256_file(local_path);
                                CHECK(local_hash);
                                if (!local_hash) {
                                    global_payload_ok = false;
                                    break;
                                }
                                global_artifact.ranges.push_back(
                                    vastloom::model::RangeArtifactEntry{
                                        "global",
                                        "global[" + std::to_string(index) +
                                            "] " + planned.tensor_name,
                                        std::filesystem::path("ranges") /
                                            (std::to_string(index) + ".bin"),
                                        local_path,
                                        planned.source.bytes,
                                        local_hash.take_value(),
                                        "checkpoint-globals.bin",
                                        "weights/checkpoint-globals.bin",
                                        global_offset,
                                        global_source_hash
                                            ? global_source_hash.value()
                                            : std::string(64U, '0'),
                                        planned.source.offset,
                                        planned.source.bytes,
                                    });
                            }
                            auto global_binding = global_payload_ok
                                ? vastloom::inference::
                                      bind_deepseek_v4_global_range_artifact(
                                          global_artifact,
                                          checkpoint_report,
                                          global_plan.value(),
                                          global_spec)
                                : vastloom::Result<vastloom::inference::
                                      DeepseekV4GlobalArtifactBinding>(
                                      vastloom::make_error(
                                          vastloom::ErrorCode::internal_error,
                                          "unable to construct global fixture"));
                            CHECK(global_binding &&
                                  global_binding.value().ranges.size() == 6U &&
                                  global_binding.value().payload_bytes ==
                                      global_offset &&
                                  global_binding.value().
                                      exact_source_binding_verified &&
                                  !global_binding.value().execution_supported &&
                                  vastloom::inference::to_json(
                                      global_binding.value()).find(
                                      "global_artifact_binding.v1") !=
                                      std::string::npos);
                            if (global_binding) {
                                auto set_id = vastloom::inference::
                                    deepseek_v4_bound_global_tensor_set_logical_id(
                                        global_binding.value());
                                CHECK(set_id);
                                const std::string global_set_id = set_id
                                    ? set_id.value()
                                    : "deepseek/invalid/globals/tensor-set";
                                vastloom::safety::ResourceLimits global_limits;
                                global_limits.host_memory_hard_limit =
                                    global_offset +
                                    global_binding.value().
                                        maximum_local_range_bytes;
                                global_limits.ram_hard_limit = global_offset;
                                global_limits.vram_hard_limit = 1U;
                                global_limits.staging_hard_limit =
                                    global_binding.value().
                                        maximum_local_range_bytes;
                                global_limits.scratch_hard_limit = 1U;
                                vastloom::safety::ResourceGovernor
                                    global_governor(global_limits);
                                auto global_residency = set_id
                                    ? vastloom::model::ResidencyObject::create(
                                          global_set_id, global_offset)
                                    : vastloom::Result<vastloom::model::
                                          ResidencyObject>(
                                          vastloom::make_error(
                                              vastloom::ErrorCode::internal_error,
                                              "missing global set identity"));
                                vastloom::model::BoundRangeSetLoadOptions
                                    global_options;
                                global_options.max_total_bytes = global_offset;
                                global_options.max_range_bytes =
                                    global_binding.value().
                                        maximum_local_range_bytes;
                                global_options.max_ranges = 6U;
                                global_options.measure_process_io = false;
                                auto loaded_graph = graph;
                                bool loaded_globals_ok = false;
                                CHECK(global_residency);
                                if (global_residency) {
                                    {
                                        auto loaded = vastloom::inference::
                                            load_deepseek_v4_bound_global_tensor_set(
                                                global_binding.value(),
                                                checkpoint_report,
                                                global_residency.value(),
                                                global_governor,
                                                global_options);
                                        CHECK(loaded &&
                                              loaded.value().tensors().size() == 6U &&
                                              loaded.value().telemetry().range_count ==
                                                  6U &&
                                              loaded.value().telemetry().
                                                      logical_bytes ==
                                                  global_offset &&
                                              loaded.value().find(
                                                  "token_embd.weight") != nullptr &&
                                              !loaded.value().tensor(
                                                  "missing.weight"));
                                        auto globals = loaded
                                            ? vastloom::inference::
                                                  deepseek_v4_bind_0731_gguf_globals(
                                                      loaded.value(),
                                                      one_layer_config.value())
                                            : vastloom::Result<vastloom::inference::
                                                  DeepseekV4ReferenceGlobals>(
                                                  vastloom::make_error(
                                                      vastloom::ErrorCode::
                                                          internal_error,
                                                      "missing loaded globals"));
                                        CHECK(globals &&
                                              globals.value().matrices.size() == 3U &&
                                              globals.value().float_tensors.size() ==
                                                  3U);
                                        auto wrong_global_config =
                                            one_layer_config.value();
                                        ++wrong_global_config.vocabulary_size;
                                        CHECK(!vastloom::inference::
                                                  deepseek_v4_bind_0731_gguf_globals(
                                                      loaded.value(),
                                                      wrong_global_config));
                                        if (globals) {
                                            for (auto& [name, matrix] :
                                                 globals.value().matrices) {
                                                loaded_graph.float_tensors.erase(name);
                                                loaded_graph.packed_matrices.insert_or_assign(
                                                    name, std::move(matrix));
                                            }
                                            for (auto& [name, tensor] :
                                                 globals.value().float_tensors) {
                                                loaded_graph.float_tensors.erase(name);
                                                loaded_graph.packed_float_tensors.
                                                    insert_or_assign(
                                                        name, std::move(tensor));
                                            }
                                            loaded_globals_ok = true;
                                        }
                                    }
                                    auto ready =
                                        global_residency.value().snapshot();
                                    CHECK(loaded_globals_ok && ready &&
                                          ready.value().state ==
                                              vastloom::model::ResidencyState::
                                                  ram_ready &&
                                          ready.value().active_uses == 1U &&
                                          ready.value().ram_reserved_bytes ==
                                              global_offset &&
                                          ready.value().staging_reserved_bytes ==
                                              0U &&
                                          !global_residency.value().evict_ram());
                                }

                                auto direct_graph = graph;
                                for (const auto& planned :
                                     global_plan.value().tensors) {
                                    const auto& consumer =
                                        global_consumers.at(planned.tensor_name);
                                    const auto& encoded =
                                        global_encoded.at(planned.tensor_name);
                                    direct_graph.float_tensors.erase(consumer);
                                    if (planned.tensor_name ==
                                            "token_embd.weight" ||
                                        planned.tensor_name == "output.weight" ||
                                        planned.tensor_name ==
                                            "output_hc_fn.weight") {
                                        const auto logical_columns =
                                            planned.tensor_name ==
                                                    "output_hc_fn.weight"
                                                ? config.hyper_multiplier *
                                                      config.width
                                                : config.width;
                                        direct_graph.packed_matrices.insert_or_assign(
                                            consumer,
                                            vastloom::inference::
                                                DeepseekV4PackedMatrix{
                                                    *planned.ggml_type,
                                                    static_cast<std::size_t>(
                                                        planned.shape[1]),
                                                    logical_columns,
                                                    static_cast<std::size_t>(
                                                        planned.shape[0]),
                                                    encoded});
                                    } else {
                                        std::size_t elements = 1U;
                                        for (const auto dimension : planned.shape) {
                                            elements *= static_cast<std::size_t>(
                                                dimension);
                                        }
                                        direct_graph.packed_float_tensors.
                                            insert_or_assign(
                                                consumer,
                                                [&] {
                                                    vastloom::inference::
                                                        DeepseekV4PackedFloatTensor
                                                            tensor;
                                                    tensor.ggml_type =
                                                        *planned.ggml_type;
                                                    tensor.elements = elements;
                                                    tensor.encoded = encoded;
                                                    return tensor;
                                                }());
                                    }
                                }
                                if (loaded_globals_ok) {
                                    vastloom::inference::
                                        DeepseekV4ReferenceState direct_state;
                                    vastloom::inference::
                                        DeepseekV4ReferenceState loaded_state;
                                    auto direct_prefill = vastloom::inference::
                                        deepseek_v4_reference_prefill(
                                            direct_graph, token_ids,
                                            &direct_state);
                                    auto loaded_prefill = vastloom::inference::
                                        deepseek_v4_reference_prefill(
                                            loaded_graph, token_ids,
                                            &loaded_state);
                                    CHECK(direct_prefill && loaded_prefill);
                                    if (direct_prefill && loaded_prefill) {
                                        compare_packed_forward(
                                            loaded_prefill.value(),
                                            direct_prefill.value());
                                        auto direct_decode = vastloom::inference::
                                            deepseek_v4_reference_decode(
                                                direct_graph,
                                                decode_tokens.front(),
                                                direct_state);
                                        auto loaded_decode = vastloom::inference::
                                            deepseek_v4_reference_decode(
                                                loaded_graph,
                                                decode_tokens.front(),
                                                loaded_state);
                                        CHECK(direct_decode && loaded_decode);
                                        if (direct_decode && loaded_decode) {
                                            compare_packed_forward(
                                                loaded_decode.value(),
                                                direct_decode.value());
                                        }
                                    }
                                }
                                for (const auto& name : {
                                         std::string("embed.weight"),
                                         std::string("head.weight"),
                                         std::string("hc_head_fn")}) {
                                    loaded_graph.packed_matrices.erase(name);
                                }
                                for (const auto& name : {
                                         std::string("norm.weight"),
                                         std::string("hc_head_base"),
                                         std::string("hc_head_scale")}) {
                                    loaded_graph.packed_float_tensors.erase(name);
                                }
                                CHECK(global_residency &&
                                      global_residency.value().evict_ram() &&
                                      global_governor.snapshot().reserved[0] == 0U &&
                                      global_governor.snapshot().reserved[2] == 0U);

                                auto insufficient_limits = global_limits;
                                --insufficient_limits.host_memory_hard_limit;
                                vastloom::safety::ResourceGovernor
                                    insufficient_governor(insufficient_limits);
                                auto insufficient_residency =
                                    vastloom::model::ResidencyObject::create(
                                        global_set_id, global_offset);
                                CHECK(insufficient_residency &&
                                      !vastloom::inference::
                                          load_deepseek_v4_bound_global_tensor_set(
                                              global_binding.value(),
                                              checkpoint_report,
                                              insufficient_residency.value(),
                                              insufficient_governor,
                                              global_options) &&
                                      insufficient_governor.snapshot().reserved[0] ==
                                          0U &&
                                      insufficient_governor.snapshot().reserved[2] ==
                                          0U);

                                auto cancel_options = global_options;
                                cancel_options.cancelled = [] { return true; };
                                auto global_cancelled_residency =
                                    vastloom::model::ResidencyObject::create(
                                        global_set_id, global_offset);
                                CHECK(global_cancelled_residency &&
                                      !vastloom::inference::
                                          load_deepseek_v4_bound_global_tensor_set(
                                              global_binding.value(),
                                              checkpoint_report,
                                              global_cancelled_residency.value(),
                                              global_governor,
                                              cancel_options) &&
                                      global_cancelled_residency.value().snapshot() &&
                                      global_cancelled_residency.value().snapshot().value().
                                              state ==
                                          vastloom::model::ResidencyState::nvme &&
                                      global_governor.snapshot().reserved[0] == 0U &&
                                      global_governor.snapshot().reserved[2] == 0U);

                                const auto corrupt_path =
                                    global_binding.value().ranges.back().
                                        local_source.shard;
                                auto corrupt_payload = global_encoded.at(
                                    global_binding.value().ranges.back().tensor_name);
                                corrupt_payload.back() ^= std::byte{1U};
                                write_bytes(corrupt_path, corrupt_payload);
                                auto corrupt_residency =
                                    vastloom::model::ResidencyObject::create(
                                        global_set_id, global_offset);
                                CHECK(corrupt_residency &&
                                      !vastloom::inference::
                                          load_deepseek_v4_bound_global_tensor_set(
                                              global_binding.value(),
                                              checkpoint_report,
                                              corrupt_residency.value(),
                                              global_governor,
                                              global_options) &&
                                      corrupt_residency.value().snapshot() &&
                                      corrupt_residency.value().snapshot().value().state ==
                                          vastloom::model::ResidencyState::failed &&
                                      global_governor.snapshot().reserved[0] == 0U &&
                                      global_governor.snapshot().reserved[2] == 0U);
                                write_bytes(
                                    corrupt_path,
                                    global_encoded.at(
                                        global_binding.value().ranges.back().tensor_name));

                                auto incomplete_binding =
                                    global_binding.value();
                                incomplete_binding.ranges.pop_back();
                                auto incomplete_residency =
                                    vastloom::model::ResidencyObject::create(
                                        global_set_id, global_offset);
                                CHECK(incomplete_residency &&
                                      !vastloom::inference::
                                          load_deepseek_v4_bound_global_tensor_set(
                                              incomplete_binding,
                                              checkpoint_report,
                                              incomplete_residency.value(),
                                              global_governor,
                                              global_options));

                                auto tampered_plan = global_plan.value();
                                tampered_plan.tensors.front().tensor_name =
                                    "output.weight";
                                CHECK(!vastloom::inference::
                                          bind_deepseek_v4_global_range_artifact(
                                              global_artifact,
                                              checkpoint_report,
                                              tampered_plan,
                                              global_spec));
                            }
                        }

                        auto wrong_global_role = checkpoint_report;
                        const auto embedding_tensor = std::find_if(
                            wrong_global_role.catalog.tensors.begin(),
                            wrong_global_role.catalog.tensors.end(),
                            [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                                return tensor.name == "token_embd.weight";
                            });
                        CHECK(embedding_tensor !=
                              wrong_global_role.catalog.tensors.end());
                        if (embedding_tensor !=
                            wrong_global_role.catalog.tensors.end()) {
                            embedding_tensor->role =
                                vastloom::model::TensorRole::other;
                            CHECK(!vastloom::inference::
                                      deepseek_v4_validate_0731_gguf_checkpoint_catalog(
                                          wrong_global_role,
                                          one_layer_config.value()));
                        }

                        auto unexpected_global = checkpoint_report;
                        const auto norm_tensor = std::find_if(
                            unexpected_global.catalog.tensors.begin(),
                            unexpected_global.catalog.tensors.end(),
                            [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                                return tensor.name == "output_norm.weight";
                            });
                        CHECK(norm_tensor !=
                              unexpected_global.catalog.tensors.end());
                        if (norm_tensor !=
                            unexpected_global.catalog.tensors.end()) {
                            auto extra = *norm_tensor;
                            extra.name = "unexpected_global.weight";
                            unexpected_global.catalog.tensors.push_back(
                                std::move(extra));
                            CHECK(!vastloom::inference::
                                      deepseek_v4_validate_0731_gguf_checkpoint_catalog(
                                          unexpected_global,
                                          one_layer_config.value()));
                        }

                        auto wrong_expert_name = checkpoint_report;
                        const auto expert_tensor = std::find_if(
                            wrong_expert_name.catalog.tensors.begin(),
                            wrong_expert_name.catalog.tensors.end(),
                            [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                                return tensor.role ==
                                    vastloom::model::TensorRole::routed_expert;
                            });
                        CHECK(expert_tensor !=
                              wrong_expert_name.catalog.tensors.end());
                        if (expert_tensor !=
                            wrong_expert_name.catalog.tensors.end()) {
                            expert_tensor->name = "blk.0.invalid_expert.weight";
                            CHECK(!vastloom::inference::
                                      deepseek_v4_validate_0731_gguf_checkpoint_catalog(
                                          wrong_expert_name,
                                          one_layer_config.value()));
                        }

                        auto duplicate_expert_index = checkpoint_report;
                        CHECK(!duplicate_expert_index.catalog.experts.empty());
                        if (!duplicate_expert_index.catalog.experts.empty()) {
                            auto& indices = duplicate_expert_index.catalog.experts
                                                .front()
                                                .tensor_indices;
                            CHECK(indices.size() == 3U);
                            if (indices.size() == 3U) indices[1U] = indices[0U];
                            CHECK(!vastloom::inference::
                                      deepseek_v4_validate_0731_gguf_checkpoint_catalog(
                                          duplicate_expert_index,
                                          one_layer_config.value()));
                        }

                        auto duplicate_packed_slice = checkpoint_report;
                        CHECK(!duplicate_packed_slice.catalog.experts.empty());
                        if (!duplicate_packed_slice.catalog.experts.empty()) {
                            auto& slices = duplicate_packed_slice.catalog.experts
                                               .front()
                                               .tensor_slices;
                            CHECK(slices.size() == 3U);
                            if (slices.size() == 3U) slices[1U] = slices[0U];
                            CHECK(!vastloom::inference::
                                      deepseek_v4_validate_0731_gguf_checkpoint_catalog(
                                          duplicate_packed_slice,
                                          one_layer_config.value()));
                        }

                        auto wrong_physical_count = checkpoint_report;
                        ++wrong_physical_count.physical_tensors;
                        CHECK(!vastloom::inference::
                                  deepseek_v4_validate_0731_gguf_checkpoint_catalog(
                                      wrong_physical_count,
                                      one_layer_config.value()));
                    }
                    auto wrong_table_shape = one_layer_report;
                    const auto flattened_table = std::find_if(
                        wrong_table_shape.catalog.tensors.begin(),
                        wrong_table_shape.catalog.tensors.end(),
                        [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                            return tensor.name == "blk.0.ffn_gate_tid2eid.weight";
                        });
                    CHECK(flattened_table != wrong_table_shape.catalog.tensors.end());
                    if (flattened_table != wrong_table_shape.catalog.tensors.end()) {
                        flattened_table->shape = {
                            static_cast<std::uint64_t>(
                                config.experts_per_token * config.vocabulary_size)};
                        CHECK(!vastloom::inference::
                                   deepseek_v4_validate_0731_gguf_layer_catalogs(
                                       wrong_table_shape, one_layer_config.value()));
                    }
                    auto unexpected_tensor = one_layer_report;
                    const auto first_non_routed = std::find_if(
                        unexpected_tensor.catalog.tensors.begin(),
                        unexpected_tensor.catalog.tensors.end(),
                        [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                            return tensor.layer == 0U &&
                                   tensor.role !=
                                       vastloom::model::TensorRole::routed_expert;
                        });
                    CHECK(first_non_routed != unexpected_tensor.catalog.tensors.end());
                    if (first_non_routed != unexpected_tensor.catalog.tensors.end()) {
                        auto extra = *first_non_routed;
                        extra.name = "blk.0.unexpected.weight";
                        unexpected_tensor.catalog.tensors.push_back(std::move(extra));
                        CHECK(!vastloom::inference::
                                   deepseek_v4_validate_0731_gguf_layer_catalogs(
                                       unexpected_tensor, one_layer_config.value()));
                    }
                    auto wrong_role = one_layer_report;
                    const auto query = std::find_if(
                        wrong_role.catalog.tensors.begin(),
                        wrong_role.catalog.tensors.end(),
                        [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                            return tensor.name == "blk.0.attn_q_a.weight";
                        });
                    CHECK(query != wrong_role.catalog.tensors.end());
                    if (query != wrong_role.catalog.tensors.end()) {
                        query->role = vastloom::model::TensorRole::other;
                        CHECK(!vastloom::inference::
                                   deepseek_v4_validate_0731_gguf_layer_catalogs(
                                       wrong_role, one_layer_config.value()));
                    }
                }
                auto wrong_config = derived_config.value();
                ++wrong_config.query_lora_rank;
                CHECK(!vastloom::inference::deepseek_v4_validate_0731_gguf_layer0_catalog(
                    reference_report, wrong_config));
                auto wrong_bytes = reference_report;
                const auto matrix = std::find_if(
                    wrong_bytes.catalog.tensors.begin(),
                    wrong_bytes.catalog.tensors.end(),
                    [](const vastloom::model::LogicalTensorDescriptor& tensor) {
                        return tensor.name == "blk.0.attn_q_a.weight";
                    });
                CHECK(matrix != wrong_bytes.catalog.tensors.end());
                if (matrix != wrong_bytes.catalog.tensors.end()) {
                    ++matrix->source.bytes;
                    CHECK(!vastloom::inference::
                              deepseek_v4_validate_0731_gguf_layer0_catalog(
                                  wrong_bytes, derived_config.value()));
                }
            }
            auto missing_clamps = reference_report;
            missing_clamps.reference_architecture->routed_expert_swiglu_limits.pop_back();
            CHECK(!vastloom::inference::deepseek_v4_reference_config_from_0731_gguf(
                missing_clamps));
            auto unpinned_report = reference_report;
            unpinned_report.pinned_0731_checkpoint_spec_verified = false;
            CHECK(!vastloom::inference::deepseek_v4_reference_config_from_0731_gguf(
                unpinned_report));
            CHECK(trunk_artifact_ok && trunk_binding.ranges.size() == 21U &&
                  trunk_binding.payload_bytes == artifact_catalog.always_active_bytes &&
                  trunk_binding.payload_bytes == 2'844U);

            vastloom::safety::ResourceLimits trunk_limits;
            trunk_limits.host_memory_hard_limit =
                trunk_binding.payload_bytes + trunk_binding.maximum_local_range_bytes;
            trunk_limits.ram_hard_limit = trunk_binding.payload_bytes;
            trunk_limits.vram_hard_limit = 1U;
            trunk_limits.staging_hard_limit = trunk_binding.maximum_local_range_bytes;
            trunk_limits.scratch_hard_limit = 1U;
            vastloom::safety::ResourceGovernor trunk_governor(trunk_limits);
            auto trunk_logical_id =
                vastloom::inference::deepseek_v4_bound_trunk_tensor_set_logical_id(
                    trunk_binding);
            auto trunk_residency = trunk_logical_id
                ? vastloom::model::ResidencyObject::create(
                      trunk_logical_id.value(), trunk_binding.payload_bytes)
                : vastloom::Result<vastloom::model::ResidencyObject>(
                      vastloom::make_error(vastloom::ErrorCode::internal_error,
                                           "missing packed trunk identity"));
            vastloom::model::BoundRangeSetLoadOptions trunk_options;
            trunk_options.max_total_bytes = trunk_binding.payload_bytes;
            trunk_options.max_range_bytes = trunk_binding.maximum_local_range_bytes;
            trunk_options.max_ranges = trunk_binding.ranges.size();
            trunk_options.measure_process_io = false;
            auto artifact_graph = graph;
            bool artifact_graph_ok = trunk_artifact_ok && bound_ok && trunk_residency;
            if (artifact_graph_ok) {
                {
                    auto loaded =
                        vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
                            trunk_binding, trunk_report, trunk_residency.value(),
                            trunk_governor, trunk_options);
                    CHECK(loaded && loaded.value().tensors().size() == 21U &&
                          loaded.value().telemetry().range_count == 21U &&
                          loaded.value().telemetry().logical_bytes == 2'844U);
                    if (!loaded) {
                        artifact_graph_ok = false;
                    } else {
                        auto bound_layer =
                            vastloom::inference::deepseek_v4_bind_0731_gguf_layer0_trunk(
                                loaded.value(), config);
                        CHECK(bound_layer && bound_layer.value().layer == 0U &&
                              bound_layer.value().matrices.size() == 11U &&
                              bound_layer.value().integer_tensors.size() == 1U &&
                              bound_layer.value().float_tensors.size() == 9U);
                        auto wrong_config = config;
                        ++wrong_config.query_lora_rank;
                        CHECK(!vastloom::inference::
                                  deepseek_v4_bind_0731_gguf_layer0_trunk(
                                      loaded.value(), wrong_config));
                        if (!bound_layer) {
                            artifact_graph_ok = false;
                        } else {
                            for (auto& [name, matrix] :
                                 bound_layer.value().matrices) {
                                artifact_graph.packed_matrices.emplace(
                                    name, std::move(matrix));
                                artifact_graph.float_tensors.erase(name);
                            }
                            for (auto& [name, tensor] :
                                 bound_layer.value().integer_tensors) {
                                artifact_graph.packed_integer_tensors.emplace(
                                    name, std::move(tensor));
                                artifact_graph.integer_tensors.erase(name);
                            }
                            for (auto& [name, tensor] :
                                 bound_layer.value().float_tensors) {
                                artifact_graph.packed_float_tensors.emplace(
                                    name, std::move(tensor));
                                artifact_graph.float_tensors.erase(name);
                            }
                        }
                        for (const auto& spec : layer_matrices) {
                            if (!spec.expert.has_value()) continue;
                            const auto found =
                                governed_graph.packed_matrices.find(spec.name);
                            CHECK(found != governed_graph.packed_matrices.end());
                            if (found == governed_graph.packed_matrices.end()) {
                                artifact_graph_ok = false;
                                break;
                            }
                            artifact_graph.packed_matrices.emplace(
                                spec.name, found->second);
                            artifact_graph.float_tensors.erase(spec.name);
                        }
                        CHECK(!vastloom::inference::deepseek_v4_bind_packed_float(
                                  loaded.value(), layer0_gguf_names.at(hash_table_name)) &&
                              !vastloom::inference::deepseek_v4_bind_packed_i32(
                                  loaded.value(), layer0_gguf_names.at(
                                                      layer_float_vectors.front().name)) &&
                              !vastloom::inference::deepseek_v4_bind_packed_matrix(
                                  loaded.value(), layer0_gguf_names.at(
                                                      layer_float_vectors.front().name),
                                  1U));
                    }
                }
                auto retained = trunk_residency.value().snapshot();
                CHECK(retained && retained.value().state ==
                                      vastloom::model::ResidencyState::ram_ready &&
                      retained.value().active_uses == 1U &&
                      retained.value().ram_reserved_bytes == 2'844U &&
                      retained.value().staging_reserved_bytes == 0U &&
                      !trunk_residency.value().evict_ram());
            }
            CHECK(artifact_graph_ok && artifact_graph.packed_matrices.size() == 23U &&
                  artifact_graph.packed_integer_tensors.size() == 1U &&
                  artifact_graph.packed_float_tensors.size() == 9U);
            if (artifact_graph_ok) {
                vastloom::inference::DeepseekV4ReferenceState oracle_artifact_state;
                vastloom::inference::DeepseekV4ReferenceState artifact_state;
                auto oracle_artifact =
                    vastloom::inference::deepseek_v4_reference_prefill(
                        quantized_oracle, token_ids, &oracle_artifact_state);
                auto artifact = vastloom::inference::deepseek_v4_reference_prefill(
                    artifact_graph, token_ids, &artifact_state);
                CHECK(oracle_artifact && artifact);
                if (oracle_artifact && artifact) {
                    compare_packed_forward(artifact.value(), oracle_artifact.value());
                    auto oracle_decode =
                        vastloom::inference::deepseek_v4_reference_decode(
                            quantized_oracle, 10U, oracle_artifact_state);
                    auto artifact_decode =
                        vastloom::inference::deepseek_v4_reference_decode(
                            artifact_graph, 10U, artifact_state);
                    CHECK(oracle_decode && artifact_decode);
                    if (oracle_decode && artifact_decode) {
                        compare_packed_forward(artifact_decode.value(),
                                               oracle_decode.value());
                    }
                }

                const auto& float_name = layer_float_vectors.front().name;
                {
                    auto wrong_elements = artifact_graph;
                    ++wrong_elements.packed_float_tensors.at(float_name).elements;
                    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                        wrong_elements, token_ids));
                }
                {
                    auto wrong_type = artifact_graph;
                    wrong_type.packed_float_tensors.at(float_name).ggml_type =
                        vastloom::inference::kGgmlTypeI32;
                    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                        wrong_type, token_ids));
                }
                {
                    auto truncated = artifact_graph;
                    --truncated.packed_float_tensors.at(float_name).resident_bytes;
                    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                        truncated, token_ids));
                }
                {
                    auto ambiguous = artifact_graph;
                    ambiguous.packed_float_tensors.at(float_name).encoded.push_back(
                        std::byte{0U});
                    CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                        ambiguous, token_ids));
                }
                {
                    auto non_finite = artifact_graph;
                    auto& tensor = non_finite.packed_float_tensors.at(float_name);
                    tensor.resident.reset();
                    tensor.resident_offset = 0U;
                    tensor.resident_bytes = 0U;
                    tensor.encoded = artifact_float_bytes.at(
                        layer0_gguf_names.at(float_name));
                    const std::array<std::byte, 4U> nan_bits{
                        std::byte{0x00U}, std::byte{0x00U},
                        std::byte{0xc0U}, std::byte{0x7fU}};
                    std::copy(nan_bits.begin(), nan_bits.end(), tensor.encoded.begin());
                    const auto before = artifact_state;
                    CHECK(!vastloom::inference::deepseek_v4_reference_decode(
                        non_finite, 12U, artifact_state));
                    CHECK(artifact_state.next_position == before.next_position &&
                          artifact_state.layers[0].window_key_values ==
                              before.layers[0].window_key_values);
                }
            }
            artifact_graph.packed_matrices.clear();
            artifact_graph.packed_integer_tensors.clear();
            artifact_graph.packed_float_tensors.clear();
            if (trunk_residency) {
                auto released = trunk_residency.value().snapshot();
                CHECK(released && released.value().active_uses == 0U &&
                      released.value().evictable &&
                      trunk_residency.value().evict_ram() &&
                      trunk_governor.snapshot().reserved[0] == 0U &&
                      trunk_governor.snapshot().reserved[2] == 0U);
            }

            // Keep only the trunk buffers and replace static expert tensors with a router-first,
            // two-entry file-backed cache. One-token prefill selects experts 2/3, the next decode
            // hits them, and the following decode selects 0/1 and evicts the old pair.
            auto dynamic_graph = governed_graph;
            for (const auto& spec : layer_matrices) {
                if (spec.expert.has_value()) {
                    dynamic_graph.packed_matrices.erase(spec.name);
                    governed_graph.packed_matrices.erase(spec.name);
                }
            }
            governed_graph.packed_matrices.clear();
            governed_graph.packed_integer_tensors.clear();
            const std::size_t trunk_residency_count = file_plan.value().tiles.size();
            CHECK(resident_objects.size() ==
                  trunk_residency_count + packed_catalog.experts.size());
            for (std::size_t index = trunk_residency_count;
                 index < resident_objects.size(); ++index) {
                auto snapshot = resident_objects[index].snapshot();
                CHECK(snapshot && snapshot.value().active_uses == 0U &&
                      snapshot.value().evictable &&
                      resident_objects[index].evict_ram());
            }

            std::vector<vastloom::inference::DeepseekV4ExpertBundleBinding>
                expert_bindings;
            bool expert_bindings_ok = true;
            for (const auto& bundle : packed_catalog.experts) {
                vastloom::inference::DeepseekV4ExpertBundleBinding binding;
                binding.expert_id = bundle.expert;
                binding.bundle = bundle;
                std::size_t matrix_count = 0U;
                for (const auto tensor_index : bundle.tensor_indices) {
                    const auto& descriptor = packed_catalog.tensors[tensor_index];
                    const auto* spec = find_spec(descriptor.name);
                    std::optional<vastloom::inference::DeepseekV4ExpertWeight>
                        weight;
                    if (descriptor.name.ends_with(".w1.weight")) {
                        weight = vastloom::inference::DeepseekV4ExpertWeight::w1;
                    } else if (descriptor.name.ends_with(".w2.weight")) {
                        weight = vastloom::inference::DeepseekV4ExpertWeight::w2;
                    } else if (descriptor.name.ends_with(".w3.weight")) {
                        weight = vastloom::inference::DeepseekV4ExpertWeight::w3;
                    }
                    if (matrix_count >= binding.matrices.size() || spec == nullptr ||
                        !weight.has_value() || !descriptor.ggml_type.has_value()) {
                        expert_bindings_ok = false;
                        break;
                    }
                    binding.matrices[matrix_count++] =
                        vastloom::inference::DeepseekV4ExpertMatrixBinding{
                            *weight, tensor_index, *descriptor.ggml_type,
                            spec->rows, spec->columns, spec->storage_columns};
                }
                if (!expert_bindings_ok || matrix_count != binding.matrices.size()) {
                    expert_bindings_ok = false;
                    break;
                }
                expert_bindings.push_back(std::move(binding));
            }
            CHECK(expert_bindings_ok &&
                  expert_bindings.size() == packed_catalog.experts.size());

            vastloom::inference::DeepseekV4ExpertCacheOptions cache_options;
            cache_options.block_prefix = "layers.0.ffn";
            cache_options.max_cached_experts = 2U;
            cache_options.max_experts_per_request = 2U;
            cache_options.load = expert_options;
            if (expert_bindings_ok) {
                auto invalid_options = cache_options;
                invalid_options.max_experts_per_request = 3U;
                CHECK(!vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, invalid_options));
                auto invalid_async_queue = cache_options;
                invalid_async_queue.asynchronous_workers = 1U;
                invalid_async_queue.asynchronous_queue_capacity = 1U;
                CHECK(!vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, invalid_async_queue));
                auto invalid_async_workers = cache_options;
                invalid_async_workers.asynchronous_workers = 5U;
                invalid_async_workers.asynchronous_queue_capacity = 2U;
                CHECK(!vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, invalid_async_workers));
                auto ignored_async_queue = cache_options;
                ignored_async_queue.asynchronous_queue_capacity = 2U;
                CHECK(!vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, ignored_async_queue));
                auto duplicate_bindings = expert_bindings;
                duplicate_bindings.push_back(expert_bindings.front());
                CHECK(!vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    std::move(duplicate_bindings), governor, cache_options));
                auto malformed_bindings = expert_bindings;
                malformed_bindings.front().matrices.front().logical_columns =
                    malformed_bindings.front().matrices.front().storage_columns + 1U;
                CHECK(!vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    std::move(malformed_bindings), governor, cache_options));
                auto malformed_mixed_format = expert_bindings;
                auto malformed_w2 = std::find_if(
                    malformed_mixed_format.front().matrices.begin(),
                    malformed_mixed_format.front().matrices.end(),
                    [](const auto& matrix) {
                        return matrix.weight ==
                               vastloom::inference::DeepseekV4ExpertWeight::w2;
                    });
                CHECK(malformed_w2 !=
                      malformed_mixed_format.front().matrices.end());
                if (malformed_w2 !=
                    malformed_mixed_format.front().matrices.end()) {
                    malformed_w2->ggml_type =
                        vastloom::inference::kGgmlTypeIq1S;
                    CHECK(!vastloom::inference::DeepseekV4BoundedExpertCache::create(
                        std::move(malformed_mixed_format), governor,
                        cache_options));
                }
            }
            auto cache_result =
                vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, cache_options);
            CHECK(cache_result);
            std::shared_ptr<vastloom::inference::DeepseekV4BoundedExpertCache>
                expert_cache;
            if (cache_result) {
                expert_cache = cache_result.take_value();
                auto validation_provider = expert_cache->provider_binding();
                const std::array<std::uint32_t, 2U> duplicate_request{0U, 0U};
                const std::array<std::uint32_t, 1U> unknown_request{
                    static_cast<std::uint32_t>(config.expert_count)};
                const std::array<std::uint32_t, 3U> oversized_request{0U, 1U, 2U};
                CHECK(!validation_provider.resolve(
                          "layers.1.ffn", unknown_request) &&
                      !validation_provider.resolve(
                          "layers.0.ffn", duplicate_request) &&
                      !validation_provider.resolve(
                          "layers.0.ffn", unknown_request) &&
                      !validation_provider.resolve(
                          "layers.0.ffn", oversized_request));
                const auto validation_snapshot = expert_cache->snapshot();
                CHECK(validation_snapshot.requests == 0U &&
                      validation_snapshot.cached_experts == 0U &&
                      validation_snapshot.resident_bytes == 0U);

                vastloom::safety::ResourceLimits cache_refused_limits;
                cache_refused_limits.host_memory_hard_limit = 2U;
                cache_refused_limits.ram_hard_limit = 1U;
                cache_refused_limits.vram_hard_limit = 1U;
                cache_refused_limits.staging_hard_limit = 1U;
                cache_refused_limits.scratch_hard_limit = 1U;
                vastloom::safety::ResourceGovernor cache_refused_governor(
                    cache_refused_limits);
                auto refused_cache =
                    vastloom::inference::DeepseekV4BoundedExpertCache::create(
                        expert_bindings, cache_refused_governor, cache_options);
                CHECK(refused_cache);
                if (refused_cache) {
                    auto refused_provider =
                        refused_cache.value()->provider_binding();
                    const std::array<std::uint32_t, 1U> one_expert{0U};
                    CHECK(!refused_provider.resolve(
                        "layers.0.ffn", one_expert));
                    CHECK(refused_cache.value()->snapshot().cached_experts == 0U &&
                          cache_refused_governor.snapshot().reserved[0] == 0U &&
                          cache_refused_governor.snapshot().reserved[2] == 0U);
                }
                auto refused_async_options = cache_options;
                refused_async_options.asynchronous_workers = 1U;
                refused_async_options.asynchronous_queue_capacity = 2U;
                auto refused_async_cache =
                    vastloom::inference::DeepseekV4BoundedExpertCache::create(
                        expert_bindings, cache_refused_governor,
                        refused_async_options);
                CHECK(refused_async_cache);
                if (refused_async_cache) {
                    auto refused_async_provider =
                        refused_async_cache.value()->provider_binding();
                    const std::array<std::uint32_t, 1U> one_expert{0U};
                    CHECK(!refused_async_provider.begin(
                        "layers.0.ffn", one_expert));
                    CHECK(refused_async_cache.value()->snapshot().cached_experts ==
                              0U &&
                          cache_refused_governor.snapshot().reserved[0] == 0U &&
                          cache_refused_governor.snapshot().reserved[2] == 0U);
                }
                dynamic_graph.expert_providers.emplace(
                    "layers.0.ffn", expert_cache->provider_binding());
            }

            const std::array<std::uint32_t, 1U> dynamic_prompt{token_ids.front()};
            vastloom::inference::DeepseekV4ReferenceState dynamic_oracle_state;
            vastloom::inference::DeepseekV4ReferenceState dynamic_state;
            auto dynamic_oracle = vastloom::inference::deepseek_v4_reference_prefill(
                quantized_oracle, dynamic_prompt, &dynamic_oracle_state);
            auto dynamic = vastloom::inference::deepseek_v4_reference_prefill(
                dynamic_graph, dynamic_prompt, &dynamic_state);
            CHECK(dynamic_oracle && dynamic);
            if (dynamic_oracle && dynamic) {
                compare_packed_forward(dynamic.value(), dynamic_oracle.value());
                CHECK(dynamic.value().layers[0].expert_accesses.size() == 2U);
                for (const auto& access :
                     dynamic.value().layers[0].expert_accesses) {
                    CHECK(!access.cache_hit && access.logical_bytes_read ==
                                                  access.resident_bytes);
                }
            }
            const auto first_cache_snapshot =
                expert_cache != nullptr
                    ? expert_cache->snapshot()
                    : vastloom::inference::DeepseekV4ExpertCacheSnapshot{};
            CHECK(first_cache_snapshot.requests == 1U &&
                  first_cache_snapshot.requested_experts == 2U &&
                  first_cache_snapshot.misses == 2U &&
                  first_cache_snapshot.hits == 0U &&
                  first_cache_snapshot.evictions == 0U &&
                  first_cache_snapshot.cached_experts == 2U);
            const auto requests_before_oversized_prefill =
                first_cache_snapshot.requests;
            CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                dynamic_graph, token_ids));
            CHECK(expert_cache != nullptr &&
                  expert_cache->snapshot().requests ==
                      requests_before_oversized_prefill);

            auto dynamic_oracle_hit =
                vastloom::inference::deepseek_v4_reference_decode(
                    quantized_oracle, 10U, dynamic_oracle_state);
            auto dynamic_hit = vastloom::inference::deepseek_v4_reference_decode(
                dynamic_graph, 10U, dynamic_state);
            CHECK(dynamic_oracle_hit && dynamic_hit);
            if (dynamic_oracle_hit && dynamic_hit) {
                compare_packed_forward(dynamic_hit.value(),
                                       dynamic_oracle_hit.value());
                CHECK(dynamic_hit.value().layers[0].expert_accesses.size() == 2U);
                for (const auto& access :
                     dynamic_hit.value().layers[0].expert_accesses) {
                    CHECK(access.cache_hit && access.logical_bytes_read == 0U);
                }
            }
            const auto hit_cache_snapshot =
                expert_cache != nullptr
                    ? expert_cache->snapshot()
                    : vastloom::inference::DeepseekV4ExpertCacheSnapshot{};
            CHECK(hit_cache_snapshot.misses == 2U &&
                  hit_cache_snapshot.hits == 2U &&
                  hit_cache_snapshot.evictions == 0U &&
                  hit_cache_snapshot.logical_bytes_read ==
                      first_cache_snapshot.logical_bytes_read);

            auto dynamic_oracle_miss =
                vastloom::inference::deepseek_v4_reference_decode(
                    quantized_oracle, 12U, dynamic_oracle_state);
            auto dynamic_miss = vastloom::inference::deepseek_v4_reference_decode(
                dynamic_graph, 12U, dynamic_state);
            CHECK(dynamic_oracle_miss && dynamic_miss);
            if (dynamic_oracle_miss && dynamic_miss) {
                compare_packed_forward(dynamic_miss.value(),
                                       dynamic_oracle_miss.value());
                CHECK(dynamic_miss.value().layers[0].expert_accesses.size() == 2U);
                for (const auto& access :
                     dynamic_miss.value().layers[0].expert_accesses) {
                    CHECK(!access.cache_hit && access.logical_bytes_read ==
                                                  access.resident_bytes);
                }
            }
            const auto miss_cache_snapshot =
                expert_cache != nullptr
                    ? expert_cache->snapshot()
                    : vastloom::inference::DeepseekV4ExpertCacheSnapshot{};
            CHECK(miss_cache_snapshot.misses == 4U &&
                  miss_cache_snapshot.hits == 2U &&
                  miss_cache_snapshot.evictions == 2U &&
                  miss_cache_snapshot.logical_bytes_read ==
                      packed_catalog.routed_expert_bytes &&
                  miss_cache_snapshot.cached_experts == 2U);

            {
                auto wrong_count = dynamic_graph;
                wrong_count.expert_providers["layers.0.ffn"].resolve =
                    [](std::string_view,
                       std::span<const std::uint32_t>)
                        -> vastloom::Result<std::vector<
                            vastloom::inference::DeepseekV4ResolvedExpert>> {
                        return std::vector<
                            vastloom::inference::DeepseekV4ResolvedExpert>{};
                    };
                const auto before = dynamic_state;
                CHECK(!vastloom::inference::deepseek_v4_reference_decode(
                    wrong_count, 14U, dynamic_state));
                CHECK(dynamic_state.next_position == before.next_position &&
                      dynamic_state.layers[0].window_key_values ==
                          before.layers[0].window_key_values);
            }
            {
                auto throwing = dynamic_graph;
                throwing.expert_providers["layers.0.ffn"].resolve =
                    [](std::string_view,
                       std::span<const std::uint32_t>)
                        -> vastloom::Result<std::vector<
                            vastloom::inference::DeepseekV4ResolvedExpert>> {
                        throw 7;
                    };
                CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                    throwing, dynamic_prompt));
            }
            {
                class ThrowingBatch final
                    : public vastloom::inference::DeepseekV4ExpertBatch {
                public:
                    [[nodiscard]] vastloom::Result<
                        vastloom::inference::DeepseekV4ResolvedExpert>
                    acquire(std::uint32_t) override {
                        throw 7;
                    }
                };
                auto throwing_batch = dynamic_graph;
                auto& binding =
                    throwing_batch.expert_providers["layers.0.ffn"];
                binding.resolve = {};
                binding.begin = [](
                                    std::string_view,
                                    std::span<const std::uint32_t>)
                    -> vastloom::Result<std::unique_ptr<
                        vastloom::inference::DeepseekV4ExpertBatch>> {
                    return std::unique_ptr<
                        vastloom::inference::DeepseekV4ExpertBatch>(
                        std::make_unique<ThrowingBatch>());
                };
                CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                    throwing_batch, dynamic_prompt));
            }
            {
                auto empty_batch = dynamic_graph;
                auto& binding =
                    empty_batch.expert_providers["layers.0.ffn"];
                binding.resolve = {};
                binding.begin = [](
                                    std::string_view,
                                    std::span<const std::uint32_t>)
                    -> vastloom::Result<std::unique_ptr<
                        vastloom::inference::DeepseekV4ExpertBatch>> {
                    return std::unique_ptr<
                        vastloom::inference::DeepseekV4ExpertBatch>{};
                };
                CHECK(!vastloom::inference::deepseek_v4_reference_prefill(
                    empty_batch, dynamic_prompt));
            }

            const vastloom::ByteCount dynamic_cache_bytes =
                miss_cache_snapshot.resident_bytes;
            CHECK(governor.snapshot().reserved[0] ==
                  packed_catalog.always_active_bytes + dynamic_cache_bytes);

            // A second request cannot evict a bundle still retained by an in-flight
            // consumer. The failed clear/request leave the cache and Governor accounting
            // unchanged; once the retained result dies, explicit clear succeeds.
            if (expert_cache != nullptr) {
                auto direct_provider = expert_cache->provider_binding();
                const std::array<std::uint32_t, 2U> cached_ids{0U, 1U};
                const std::array<std::uint32_t, 2U> replacement_ids{2U, 3U};
                {
                    auto retained = direct_provider.resolve(
                        "layers.0.ffn", cached_ids);
                    CHECK(retained &&
                          !direct_provider.resolve(
                              "layers.0.ffn", replacement_ids));
                    CHECK(!expert_cache->clear());
                    const auto retained_snapshot = expert_cache->snapshot();
                    CHECK(retained_snapshot.cached_experts == 2U &&
                          retained_snapshot.evictions ==
                              miss_cache_snapshot.evictions &&
                          retained_snapshot.resident_bytes == dynamic_cache_bytes &&
                          governor.snapshot().reserved[0] ==
                              packed_catalog.always_active_bytes +
                                  dynamic_cache_bytes);
                }
            }

            dynamic_graph.expert_providers.clear();
            CHECK(expert_cache != nullptr && expert_cache->clear());
            expert_cache.reset();

            // The token-level driver must aggregate the exact cache telemetry exposed by every
            // routed expert access rather than inferring traffic from model dimensions.
            auto generation_cache_result =
                vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, cache_options);
            CHECK(generation_cache_result);
            if (generation_cache_result) {
                auto generation_cache = generation_cache_result.take_value();
                auto generation_graph = dynamic_graph;
                generation_graph.expert_providers.emplace(
                    "layers.0.ffn", generation_cache->provider_binding());
                vastloom::inference::DeepseekV4ReferenceGenerationOptions
                    cache_generation_options;
                cache_generation_options.max_new_tokens = 3U;
                cache_generation_options.top_k = 1U;
                auto oracle_generation =
                    vastloom::inference::deepseek_v4_reference_generate(
                        quantized_oracle, dynamic_prompt,
                        cache_generation_options);
                auto cached_generation =
                    vastloom::inference::deepseek_v4_reference_generate(
                        generation_graph, dynamic_prompt,
                        cache_generation_options);
                CHECK(oracle_generation && cached_generation &&
                      cached_generation.value().generated_ids ==
                          oracle_generation.value().generated_ids);
                if (cached_generation) {
                    const auto cache_snapshot = generation_cache->snapshot();
                    const auto& stats = cached_generation.value().stats;
                    CHECK(stats.forward_passes == 3U &&
                          stats.expert_requests == 6U &&
                          stats.expert_cache_hits == cache_snapshot.hits &&
                          stats.expert_cache_misses == cache_snapshot.misses &&
                          stats.expert_cache_hits + stats.expert_cache_misses ==
                              stats.expert_requests &&
                          stats.logical_expert_bytes_read ==
                              cache_snapshot.logical_bytes_read &&
                          stats.resident_expert_bytes ==
                              stats.expert_requests *
                                  expert_bindings.front().bundle.bytes &&
                          stats.expert_load_service_ms >= 0.0 &&
                          stats.expert_compute_ms >= 0.0 &&
                          stats.io_compute_overlap_ms >= 0.0);
                }
                generation_graph.expert_providers.clear();
                CHECK(generation_cache->clear());
            }

            // The asynchronous provider submits the complete selected miss set, but the graph
            // acquires one expert only when its authoritative numeric compute turn begins. An
            // abandoned batch drains its futures and releases reservations without publishing
            // partial cache entries.
            auto async_options = cache_options;
            async_options.asynchronous_workers = 1U;
            async_options.asynchronous_queue_capacity = 2U;
            std::atomic<bool> enable_overlap_barrier{false};
            std::atomic<bool> second_service_waiting{false};
            std::atomic<bool> release_second_service{false};
            std::atomic<std::size_t> async_cancellation_checks{0U};
            const std::size_t cancellation_checks_before_second =
                expert_bindings.front().bundle.read_ranges.size() + 2U;
            async_options.load.cancelled = [&] {
                const auto check =
                    async_cancellation_checks.fetch_add(
                        1U, std::memory_order_acq_rel) +
                    1U;
                if (enable_overlap_barrier.load(std::memory_order_acquire) &&
                    check == cancellation_checks_before_second) {
                    second_service_waiting.store(true, std::memory_order_release);
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(2);
                    while (!release_second_service.load(
                               std::memory_order_acquire) &&
                           std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::yield();
                    }
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50));
                }
                return false;
            };
            auto async_cache_result =
                vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, async_options);
            CHECK(async_cache_result);
            std::shared_ptr<
                vastloom::inference::DeepseekV4BoundedExpertCache> async_cache;
            if (async_cache_result) {
                async_cache = async_cache_result.take_value();
                auto async_provider = async_cache->provider_binding();
                CHECK(async_provider.begin);
                const std::array<std::uint32_t, 2U> async_ids{2U, 3U};
                {
                    auto abandoned = async_provider.begin(
                        "layers.0.ffn", async_ids);
                    CHECK(abandoned && abandoned.value() != nullptr &&
                          !async_provider.begin("layers.0.ffn", async_ids) &&
                          !async_provider.resolve("layers.0.ffn", async_ids) &&
                          !async_cache->clear());
                }
                CHECK(async_cache->snapshot().cached_experts == 0U &&
                      async_cache->snapshot().resident_bytes == 0U &&
                      async_cache->snapshot().abandoned_loads == 2U &&
                      async_cache->snapshot().logical_bytes_read ==
                          2U * expert_bindings.front().bundle.bytes &&
                      async_cache->snapshot().wasted_prefetch_bytes ==
                          2U * expert_bindings.front().bundle.bytes &&
                      !async_cache->snapshot().telemetry_overflow &&
                      governor.snapshot().reserved[0] ==
                          packed_catalog.always_active_bytes &&
                      governor.snapshot().reserved[2] == 0U);
                async_cancellation_checks.store(0U, std::memory_order_release);
                second_service_waiting.store(false, std::memory_order_release);
                release_second_service.store(false, std::memory_order_release);
                enable_overlap_barrier.store(true, std::memory_order_release);
                class CoordinatedBatch final
                    : public vastloom::inference::DeepseekV4ExpertBatch {
                public:
                    CoordinatedBatch(
                        std::unique_ptr<
                            vastloom::inference::DeepseekV4ExpertBatch> inner,
                        std::atomic<bool>& second_waiting,
                        std::atomic<bool>& release_second)
                        : inner_(std::move(inner)),
                          second_waiting_(second_waiting),
                          release_second_(release_second) {}

                    [[nodiscard]] vastloom::Result<
                        vastloom::inference::DeepseekV4ResolvedExpert>
                    acquire(std::uint32_t expert_id) override {
                        auto acquired = inner_->acquire(expert_id);
                        if (!acquired || first_acquired_) return acquired;
                        first_acquired_ = true;
                        const auto deadline = std::chrono::steady_clock::now() +
                                              std::chrono::seconds(2);
                        while (!second_waiting_.load(std::memory_order_acquire) &&
                               std::chrono::steady_clock::now() < deadline) {
                            std::this_thread::yield();
                        }
                        if (!second_waiting_.load(std::memory_order_acquire)) {
                            release_second_.store(true, std::memory_order_release);
                            return vastloom::make_error(
                                vastloom::ErrorCode::internal_error,
                                "Controlled overlap service did not start");
                        }
                        release_second_.store(true, std::memory_order_release);
                        return acquired;
                    }

                private:
                    std::unique_ptr<
                        vastloom::inference::DeepseekV4ExpertBatch> inner_;
                    std::atomic<bool>& second_waiting_;
                    std::atomic<bool>& release_second_;
                    bool first_acquired_{false};
                };
                auto coordinated_provider = async_cache->provider_binding();
                auto begin = coordinated_provider.begin;
                coordinated_provider.begin =
                    [begin = std::move(begin), &second_service_waiting,
                     &release_second_service](
                        std::string_view block_prefix,
                        std::span<const std::uint32_t> expert_ids)
                        -> vastloom::Result<std::unique_ptr<
                            vastloom::inference::DeepseekV4ExpertBatch>> {
                    auto inner = begin(block_prefix, expert_ids);
                    if (!inner) return inner.error();
                    return std::unique_ptr<
                        vastloom::inference::DeepseekV4ExpertBatch>(
                        std::make_unique<CoordinatedBatch>(
                            inner.take_value(), second_service_waiting,
                            release_second_service));
                };
                dynamic_graph.expert_providers.emplace(
                    "layers.0.ffn", std::move(coordinated_provider));
            }

            vastloom::inference::DeepseekV4ReferenceState async_oracle_state;
            vastloom::inference::DeepseekV4ReferenceState async_state;
            auto async_oracle = vastloom::inference::deepseek_v4_reference_prefill(
                quantized_oracle, dynamic_prompt, &async_oracle_state);
            auto asynchronous = vastloom::inference::deepseek_v4_reference_prefill(
                dynamic_graph, dynamic_prompt, &async_state);
            enable_overlap_barrier.store(false, std::memory_order_release);
            release_second_service.store(true, std::memory_order_release);
            CHECK(async_oracle && asynchronous);
            if (async_oracle && asynchronous) {
                compare_packed_forward(asynchronous.value(), async_oracle.value());
                CHECK(asynchronous.value().layers[0].expert_accesses.size() == 2U);
                bool saw_io_compute_overlap = false;
                std::size_t previous_request_sequence = 0U;
                for (const auto& access :
                     asynchronous.value().layers[0].expert_accesses) {
                    CHECK(!access.cache_hit && access.asynchronous_load &&
                          access.load_request_sequence >
                              previous_request_sequence &&
                          access.logical_bytes_read == access.resident_bytes &&
                          access.load_queue_ms >= 0.0 &&
                          access.load_service_ms >= 0.0 &&
                          access.load_total_ms >= access.load_queue_ms &&
                          access.load_total_ms >= access.load_service_ms &&
                          access.acquire_wait_ms >= 0.0 &&
                          access.compute_ms >= 0.0 &&
                          access.io_compute_overlap_ms >= 0.0);
                    saw_io_compute_overlap = saw_io_compute_overlap ||
                                             access.io_compute_overlap_ms > 0.0;
                    previous_request_sequence = access.load_request_sequence;
                }
                CHECK(saw_io_compute_overlap);
            }
            const auto async_miss_snapshot =
                async_cache != nullptr
                    ? async_cache->snapshot()
                    : vastloom::inference::DeepseekV4ExpertCacheSnapshot{};
            CHECK(async_miss_snapshot.cached_experts == 2U &&
                  async_miss_snapshot.misses == 2U &&
                  async_miss_snapshot.hits == 0U &&
                  async_miss_snapshot.logical_bytes_read ==
                      4U * expert_bindings.front().bundle.bytes &&
                  async_miss_snapshot.wasted_prefetch_bytes ==
                      2U * expert_bindings.front().bundle.bytes &&
                  async_miss_snapshot.abandoned_loads == 2U &&
                  !async_miss_snapshot.telemetry_overflow);

            auto async_oracle_hit =
                vastloom::inference::deepseek_v4_reference_decode(
                    quantized_oracle, 10U, async_oracle_state);
            auto asynchronous_hit =
                vastloom::inference::deepseek_v4_reference_decode(
                    dynamic_graph, 10U, async_state);
            CHECK(async_oracle_hit && asynchronous_hit);
            if (async_oracle_hit && asynchronous_hit) {
                compare_packed_forward(asynchronous_hit.value(),
                                       async_oracle_hit.value());
                for (const auto& access :
                     asynchronous_hit.value().layers[0].expert_accesses) {
                    CHECK(access.cache_hit && !access.asynchronous_load &&
                          access.load_request_sequence == 0U &&
                          access.logical_bytes_read == 0U);
                }
            }
            CHECK(async_cache != nullptr &&
                  async_cache->snapshot().hits == 2U);

            dynamic_graph.expert_providers.clear();
            CHECK(async_cache != nullptr && async_cache->clear());
            async_cache.reset();

            auto cancelled_async_options = async_options;
            cancelled_async_options.load.cancelled = [] { return true; };
            auto cancelled_async_cache =
                vastloom::inference::DeepseekV4BoundedExpertCache::create(
                    expert_bindings, governor, cancelled_async_options);
            CHECK(cancelled_async_cache);
            if (cancelled_async_cache) {
                auto cancelled_provider =
                    cancelled_async_cache.value()->provider_binding();
                const std::array<std::uint32_t, 1U> cancelled_id{0U};
                {
                    auto cancelled_batch = cancelled_provider.begin(
                        "layers.0.ffn", cancelled_id);
                    CHECK(cancelled_batch &&
                          !cancelled_batch.value()->acquire(0U));
                }
                CHECK(cancelled_async_cache.value()->snapshot().cached_experts ==
                          0U &&
                      cancelled_async_cache.value()->snapshot().misses == 0U &&
                      cancelled_async_cache.value()->clear() &&
                      governor.snapshot().reserved[0] ==
                          packed_catalog.always_active_bytes &&
                      governor.snapshot().reserved[2] == 0U);
            }

            dynamic_graph.packed_matrices.clear();
            dynamic_graph.packed_integer_tensors.clear();
            for (std::size_t index = 0U; index < trunk_residency_count; ++index) {
                auto snapshot = resident_objects[index].snapshot();
                CHECK(snapshot && snapshot.value().active_uses == 0U &&
                      snapshot.value().evictable &&
                      resident_objects[index].evict_ram());
            }
            CHECK(governor.snapshot().reserved[0] == 0U &&
                  governor.snapshot().reserved[1] == 0U &&
                  governor.snapshot().reserved[2] == 0U);
        }
    }
}

void test_model_store_and_runtime_policy(const std::filesystem::path& root) {
    constexpr vastloom::ByteCount test_reserve = 20U * vastloom::GiB;
    std::error_code space_error;
    const auto available = std::filesystem::space(root, space_error).available;
    auto store = vastloom::model::ModelStore::initialize(root / "model-store", {test_reserve});
    if (!space_error && available >= test_reserve) {
        CHECK(store);
    } else {
        CHECK(!store);
        if (!store) CHECK(store.error().code == vastloom::ErrorCode::resource_exhausted);
    }
    if (store) {
        CHECK(std::filesystem::is_directory(store.value().layout().original));
        CHECK(store.value().immutable_model_path("org/model", "revision-1"));
        CHECK(!store.value().immutable_model_path("../escape", "revision-1"));
        CHECK(!store.value().preflight_write(std::numeric_limits<vastloom::ByteCount>::max()));
        CHECK(vastloom::model::ModelStore::open(root / "model-store", {test_reserve}));
    }

    const auto legacy_store = root / "legacy-model-store";
    const std::array legacy_directories{
        legacy_store / "models" / "original",
        legacy_store / "models" / "optimized",
        legacy_store / "cache",
        legacy_store / "scratch",
        legacy_store / "state",
        legacy_store / "telemetry",
    };
    for (const auto& directory : legacy_directories) std::filesystem::create_directories(directory);
    std::ofstream(legacy_store / ".stargate-store", std::ios::binary)
        << "format=stargate.store.v1\noriginal_weights=immutable\ncache=reconstructable\n";
    CHECK(vastloom::model::ModelStore::open(legacy_store, {test_reserve}));

    const auto symlink_store = root / "symlink-model-store";
    for (const auto& legacy_directory : legacy_directories) {
        std::filesystem::create_directories(
            symlink_store / legacy_directory.lexically_relative(legacy_store));
    }
    std::error_code marker_symlink_error;
    std::filesystem::create_symlink(
        legacy_store / ".stargate-store",
        symlink_store / ".vastloom-store",
        marker_symlink_error);
    if (!marker_symlink_error) {
        CHECK(!vastloom::model::ModelStore::open(symlink_store, {20U * vastloom::GiB}));
    }

    vastloom::hardware::HardwareSnapshot hardware;
    hardware.cpu.logical_cores = 24U;
    hardware.memory.total_bytes = 32U * vastloom::GiB;
    hardware.memory.available_bytes = 20U * vastloom::GiB;
    hardware.storage.available_bytes = 1U * vastloom::TiB;
    hardware.gpus.push_back({"CUDA", "test", "test", 12U * vastloom::GiB, 10U * vastloom::GiB, {}, {}});
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware);
    CHECK(policy && policy.value().resources.ram_hard_limit == 12U * vastloom::GiB);
    if (policy) {
        CHECK(policy.value().resources.host_memory_hard_limit == 12U * vastloom::GiB);
        // Half of 24 logical processors, which leaves the host as much CPU as inference takes.
        CHECK(policy.value().cpu_threads == 12U);
        CHECK(policy.value().io_workers == 2U && policy.value().max_queued_io == 4U);
        CHECK(policy.value().max_io_request_bytes == 64U * vastloom::MiB);
        CHECK(policy.value().storage_reserve_bytes == 300U * vastloom::GiB);
    }
    hardware.memory.available_bytes = 8U * vastloom::GiB;
    CHECK(!vastloom::safety::make_safe_runtime_policy(hardware));
}

void test_gpt2_tokenizer(const std::filesystem::path& root) {
    const auto tokenizer_root = root / "tokenizer";
    std::filesystem::create_directories(tokenizer_root);
    {
        std::ofstream vocab(tokenizer_root / "vocab.json");
        vocab << R"({"<|endoftext|>":0,"<|im_start|>":1,"<|im_end|>":2,"H":3,"e":4,"l":5,"o":6,"He":7,"Hel":8,"Hell":9,"Hello":10})";
    }
    {
        std::ofstream merges(tokenizer_root / "merges.txt");
        merges << "#version: 0.2\nH e\nHe l\nHel l\nHell o\n";
    }
    {
        std::ofstream config(tokenizer_root / "tokenizer_config.json");
        config << R"({"bos_token":"<|im_start|>","eos_token":"<|im_end|>","pad_token":"<|im_end|>","added_tokens_decoder":{"0":{"content":"<|endoftext|>","special":true},"1":{"content":"<|im_start|>","special":true},"2":{"content":"<|im_end|>","special":true}}})";
    }
    auto tokenizer = vastloom::inference::Gpt2Tokenizer::load(tokenizer_root);
    CHECK(tokenizer && tokenizer.value().vocab_size() == 11U);
    if (tokenizer) {
        auto hello = tokenizer.value().encode("Hello");
        CHECK(hello && hello.value() == std::vector<std::uint32_t>{10U});
        auto special = tokenizer.value().encode("<|im_start|>Hello<|im_end|>");
        CHECK(special && special.value() == (std::vector<std::uint32_t>{1U, 10U, 2U}));
        if (special) {
            auto decoded = tokenizer.value().decode(special.value(), false);
            CHECK(decoded && decoded.value() == "<|im_start|>Hello<|im_end|>");
        }
        auto templated = tokenizer.value().apply_smol_chat_template({{"user", "Hello"}}, true);
        CHECK(templated && templated.value() ==
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
    }
}

void test_manifest_and_verifier(const std::filesystem::path& root) {
    std::ofstream(root / "weights.bin", std::ios::binary) << "hello";
    std::ofstream manifest(root / "model.sgm");
    manifest << "format=vastloom.manifest.v1\n"
             << "model_id=test/model\n"
             << "revision=v1\n"
             << "architecture=test_moe\n"
             << "metadata_only=false\n"
             << "requires_avx2=false\n"
             << "supports_speculation=false\n"
             << "model_total_bytes=5\n"
             << "required_storage_bytes=5\n"
             << "minimum_ram_bytes=1GiB\n"
             << "chunk_count=1\n"
             << "chunk\tweights.bin\t5\t2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824\tdense\tfp16\t-\t-\ttrue\n";
    manifest.close();

    auto loaded = vastloom::model::load_manifest(root / "model.sgm");
    CHECK(loaded);
    if (loaded) {
        auto quick = vastloom::model::verify_model(loaded.value(), root, vastloom::model::VerificationMode::quick);
        CHECK(quick && quick.value().ok);
        auto full = vastloom::model::verify_model(loaded.value(), root, vastloom::model::VerificationMode::full);
        CHECK(full && full.value().ok && full.value().bytes_checked == 5U);

        std::ofstream(root / "weights.bin", std::ios::binary | std::ios::trunc) << "HELLO";
        auto corrupted = vastloom::model::verify_model(loaded.value(), root, vastloom::model::VerificationMode::full);
        CHECK(corrupted && !corrupted.value().ok && corrupted.value().hash_mismatches == 1U);
        std::ofstream(root / "weights.bin", std::ios::binary | std::ios::trunc) << "hello";

        const auto outside = root.parent_path() / (root.filename().string() + "-outside.bin");
        std::ofstream(outside, std::ios::binary) << "hello";
        std::error_code symlink_error;
        std::filesystem::create_symlink(outside, root / "escape-link.bin", symlink_error);
        if (!symlink_error) {
            std::ofstream symlink_manifest(root / "symlink.sgm");
            symlink_manifest << "format=vastloom.manifest.v1\nmodel_id=x\narchitecture=x\nmetadata_only=false\nmodel_total_bytes=5\nchunk_count=1\n"
                             << "chunk\tescape-link.bin\t5\t2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824\tdense\tfp16\t-\t-\ttrue\n";
            symlink_manifest.close();
            auto symlink_loaded = vastloom::model::load_manifest(root / "symlink.sgm");
            CHECK(symlink_loaded);
            if (symlink_loaded) {
                auto escaped = vastloom::model::verify_model(symlink_loaded.value(), root, vastloom::model::VerificationMode::quick);
                CHECK(escaped && !escaped.value().ok && escaped.value().unsafe_paths == 1U);
            }
        }
        std::filesystem::remove(outside, symlink_error);
    }


    std::ofstream legacy_manifest(root / "legacy-model.sgm");
    legacy_manifest << "format=stargate.manifest.v1\n"
                    << "model_id=test/legacy\n"
                    << "architecture=test_moe\n"
                    << "metadata_only=true\n"
                    << "chunk_count=0\n";
    legacy_manifest.close();
    auto legacy_loaded = vastloom::model::load_manifest(root / "legacy-model.sgm");
    CHECK(legacy_loaded && !legacy_loaded.value().warnings.empty());

    std::ofstream legacy_incomplete_moe(root / "legacy-incomplete-moe.sgm");
    legacy_incomplete_moe << "format=stargate.manifest.v1\n"
                          << "model_id=test/legacy-incomplete-moe\n"
                          << "architecture=test_moe\n"
                          << "metadata_only=true\n"
                          << "model_total_bytes=1GiB\n"
                          << "routed_expert_bytes=1GiB\n"
                          << "minimum_ram_bytes=1GiB\n"
                          << "expert_count=8\n"
                          << "selected_experts_per_token=2\n"
                          << "chunk_count=0\n";
    legacy_incomplete_moe.close();
    auto legacy_incomplete_loaded = vastloom::model::load_manifest(root / "legacy-incomplete-moe.sgm");
    CHECK(legacy_incomplete_loaded && legacy_incomplete_loaded.value().warnings.size() >= 2U);

    std::ofstream incomplete_moe(root / "incomplete-moe.sgm");
    incomplete_moe << "format=vastloom.manifest.v1\n"
                   << "model_id=test/incomplete-moe\n"
                   << "architecture=test_moe\n"
                   << "metadata_only=true\n"
                   << "model_total_bytes=1GiB\n"
                   << "routed_expert_bytes=1GiB\n"
                   << "minimum_ram_bytes=1GiB\n"
                   << "expert_count=8\n"
                   << "selected_experts_per_token=2\n"
                   << "chunk_count=0\n";
    incomplete_moe.close();
    CHECK(!vastloom::model::load_manifest(root / "incomplete-moe.sgm"));

    std::ofstream unsafe(root / "unsafe.sgm");
    unsafe << "format=vastloom.manifest.v1\nmodel_id=x\narchitecture=x\nmetadata_only=false\nchunk_count=1\n"
           << "chunk\t../escape.bin\t1\t0000000000000000000000000000000000000000000000000000000000000000\tdense\tfp16\t-\t-\ttrue\n";
    unsafe.close();
    CHECK(!vastloom::model::load_manifest(root / "unsafe.sgm"));
}

void test_governor() {
    vastloom::safety::ResourceLimits limits;
    limits.ram_hard_limit = 8U * vastloom::GiB;
    limits.vram_hard_limit = 4U * vastloom::GiB;
    limits.staging_hard_limit = 1U * vastloom::GiB;
    limits.scratch_hard_limit = 2U * vastloom::GiB;
    limits.minimum_system_ram_available = 2U * vastloom::GiB;
    limits.minimum_storage_available = 20U * vastloom::GiB;
    limits.recovery_samples = 2;
    vastloom::safety::ResourceGovernor governor(limits);

    auto first = governor.try_reserve(vastloom::safety::ResourceKind::ram, 6U * vastloom::GiB, "first");
    CHECK(first);
    auto denied = governor.try_reserve(vastloom::safety::ResourceKind::ram, 3U * vastloom::GiB, "oversubscribe");
    CHECK(!denied);
    if (first) first.value().release();
    auto second = governor.try_reserve(vastloom::safety::ResourceKind::ram, 3U * vastloom::GiB, "after-release");
    CHECK(second);

    auto state = governor.update_pressure({400U * vastloom::MiB, 2U * vastloom::GiB, 100U * vastloom::GiB});
    CHECK(state == vastloom::safety::SafetyState::emergency);
    CHECK(!governor.allows_new_work());
    state = governor.update_pressure({8U * vastloom::GiB, 2U * vastloom::GiB, 100U * vastloom::GiB});
    state = governor.update_pressure({8U * vastloom::GiB, 2U * vastloom::GiB, 100U * vastloom::GiB});
    CHECK(state == vastloom::safety::SafetyState::critical);

    vastloom::safety::ResourceLimits aggregate_limits;
    aggregate_limits.host_memory_hard_limit = 10U * vastloom::MiB;
    aggregate_limits.ram_hard_limit = 8U * vastloom::MiB;
    aggregate_limits.vram_hard_limit = 8U * vastloom::MiB;
    aggregate_limits.staging_hard_limit = 4U * vastloom::MiB;
    aggregate_limits.scratch_hard_limit = 4U * vastloom::MiB;
    aggregate_limits.minimum_system_ram_available = 1U * vastloom::MiB;
    aggregate_limits.minimum_storage_available = 1U * vastloom::MiB;
    vastloom::safety::ResourceGovernor aggregate_governor(aggregate_limits);
    auto aggregate_ram = aggregate_governor.try_reserve(
        vastloom::safety::ResourceKind::ram, 8U * vastloom::MiB, "aggregate-ram");
    CHECK(aggregate_ram);
    CHECK(!aggregate_governor.try_reserve(
        vastloom::safety::ResourceKind::staging, 3U * vastloom::MiB, "aggregate-overflow"));
    auto aggregate_staging = aggregate_governor.try_reserve(
        vastloom::safety::ResourceKind::staging, 2U * vastloom::MiB, "aggregate-staging");
    CHECK(aggregate_staging);
    CHECK(!aggregate_governor.try_reserve(
        vastloom::safety::ResourceKind::scratch, 1U, "aggregate-full"));
    auto aggregate_vram = aggregate_governor.try_reserve(
        vastloom::safety::ResourceKind::vram, 8U * vastloom::MiB, "separate-vram-tier");
    CHECK(aggregate_vram);
    CHECK(aggregate_governor.snapshot().reserved[0] == 8U * vastloom::MiB &&
          aggregate_governor.snapshot().reserved[2] == 2U * vastloom::MiB);
    if (aggregate_staging) aggregate_staging.value().release();
    auto aggregate_scratch = aggregate_governor.try_reserve(
        vastloom::safety::ResourceKind::scratch, 1U * vastloom::MiB, "aggregate-scratch");
    CHECK(aggregate_scratch);

    aggregate_limits.ram_hard_limit = 10U * vastloom::MiB;
    aggregate_limits.staging_hard_limit = 10U * vastloom::MiB;
    aggregate_limits.scratch_hard_limit = 10U * vastloom::MiB;
    vastloom::safety::ResourceGovernor aggregate_pressure_governor(aggregate_limits);
    auto pressure_ram = aggregate_pressure_governor.try_reserve(
        vastloom::safety::ResourceKind::ram, 8U * vastloom::MiB, "pressure-ram");
    auto pressure_staging = aggregate_pressure_governor.try_reserve(
        vastloom::safety::ResourceKind::staging, 1U * vastloom::MiB, "pressure-staging");
    CHECK(pressure_ram && pressure_staging);
    const auto aggregate_state = aggregate_pressure_governor.update_pressure(
        {10U * vastloom::GiB, 10U * vastloom::GiB, 10U * vastloom::GiB});
    CHECK(aggregate_state == vastloom::safety::SafetyState::pressure);
    CHECK(aggregate_pressure_governor.snapshot().last_reason.find("Aggregate host-memory") !=
          std::string::npos);
}

void test_planner() {
    vastloom::hardware::HardwareSnapshot hardware;
    hardware.operating_system = "test";
    hardware.cpu = {"Ryzen 9 7900X", 24, true, false, true};
    hardware.memory = {32U * vastloom::GiB, 26U * vastloom::GiB};
    hardware.gpus.push_back({"CUDA", "RTX 4070 SUPER", "test", 12U * vastloom::GiB, 11U * vastloom::GiB, 60.0, 220.0});
    hardware.storage = {".", 2U * vastloom::TiB, 1500U * vastloom::GiB, 1500U * vastloom::GiB, std::nullopt};

    vastloom::model::ModelManifest manifest;
    manifest.model_id = "synthetic";
    manifest.architecture = "moe";
    manifest.metadata_only = true;
    manifest.requires_avx2 = true;
    manifest.supports_speculation = true;
    manifest.model_total_bytes = 180U * vastloom::GiB;
    manifest.required_storage_bytes = 200U * vastloom::GiB;
    manifest.dense_bytes = 18U * vastloom::GiB;
    manifest.routed_expert_bytes = 150U * vastloom::GiB;
    manifest.estimated_expert_bytes = 48U * vastloom::MiB;
    manifest.minimum_ram_bytes = 16U * vastloom::GiB;
    manifest.recommended_ram_bytes = 32U * vastloom::GiB;
    manifest.recommended_vram_bytes = 12U * vastloom::GiB;
    manifest.layer_count = 48U;
    manifest.expert_count = 256U;
    manifest.selected_experts_per_token = 6U;
    manifest.max_context_tokens = 131'072U;

    auto plan = vastloom::planner::build_plan(hardware, manifest, vastloom::planner::Profile::balanced);
    CHECK(plan && plan.value().compatible);
    if (plan) {
        CHECK(plan.value().safe_ram_budget < hardware.memory.total_bytes);
        CHECK(plan.value().safe_vram_budget < hardware.gpus.front().total_vram_bytes);
        CHECK(plan.value().governor_limits.ram_hard_limit == plan.value().safe_ram_budget);
        CHECK(plan.value().governor_limits.host_memory_hard_limit == plan.value().safe_ram_budget);
        CHECK(plan.value().minimum_ram_working_set_bytes == 16U * vastloom::GiB);
        CHECK(plan.value().expert_bundle_bytes == 48U * vastloom::MiB);
        CHECK(plan.value().expert_admission_tier == "RAM");
        CHECK(plan.value().scratch_ram_bytes != 0U);
        CHECK(plan.value().governor_limits.scratch_hard_limit == plan.value().scratch_ram_bytes);
        CHECK(plan.value().dense_ram_bytes + plan.value().expert_ram_bytes +
                  plan.value().kv_ram_bytes + plan.value().staging_ram_bytes +
                  plan.value().scratch_ram_bytes <=
              plan.value().safe_ram_budget);
        CHECK(plan.value().strategy == "multitier-predictive");
    }

    auto pressured_hardware = hardware;
    pressured_hardware.memory.available_bytes = 17U * vastloom::GiB;
    auto pressured = vastloom::planner::build_plan(
        pressured_hardware, manifest, vastloom::planner::Profile::balanced);
    CHECK(pressured && !pressured.value().compatible);

    auto oversized_expert = manifest;
    oversized_expert.estimated_expert_bytes = 20U * vastloom::GiB;
    auto rejected_expert = vastloom::planner::build_plan(
        hardware, oversized_expert, vastloom::planner::Profile::balanced);
    CHECK(rejected_expert && !rejected_expert.value().compatible &&
          rejected_expert.value().expert_admission_tier.empty());

    auto vram_expert = manifest;
    vram_expert.estimated_expert_bytes = 6U * vastloom::GiB;
    auto admitted_to_vram = vastloom::planner::build_plan(
        hardware, vram_expert, vastloom::planner::Profile::balanced);
    CHECK(admitted_to_vram && admitted_to_vram.value().compatible &&
          admitted_to_vram.value().expert_admission_tier == "VRAM" &&
          admitted_to_vram.value().expert_tier_budget_bytes ==
              admitted_to_vram.value().gpu_resident_bytes);
    auto cpu_only_hardware = hardware;
    cpu_only_hardware.gpus.clear();
    auto rejected_without_tier = vastloom::planner::build_plan(
        cpu_only_hardware, vram_expert, vastloom::planner::Profile::balanced);
    CHECK(rejected_without_tier && !rejected_without_tier.value().compatible);

    auto missing_expert_identity = manifest;
    missing_expert_identity.estimated_expert_bytes = 0U;
    auto rejected_missing = vastloom::planner::build_plan(
        hardware, missing_expert_identity, vastloom::planner::Profile::balanced);
    CHECK(rejected_missing && !rejected_missing.value().compatible);

    auto missing_expert_count = manifest;
    missing_expert_count.expert_count = 0U;
    missing_expert_count.selected_experts_per_token = 0U;
    auto rejected_incomplete_moe = vastloom::planner::build_plan(
        hardware, missing_expert_count, vastloom::planner::Profile::balanced);
    CHECK(rejected_incomplete_moe && !rejected_incomplete_moe.value().compatible);

    auto overflow_hardware = hardware;
    overflow_hardware.memory.total_bytes = std::numeric_limits<vastloom::ByteCount>::max();
    overflow_hardware.memory.available_bytes = std::numeric_limits<vastloom::ByteCount>::max();
    overflow_hardware.gpus.front().total_vram_bytes = std::numeric_limits<vastloom::ByteCount>::max();
    overflow_hardware.gpus.front().free_vram_bytes = std::numeric_limits<vastloom::ByteCount>::max();
    CHECK(!vastloom::planner::build_plan(
        overflow_hardware, manifest, vastloom::planner::Profile::balanced));
}

void test_io_pool(const std::filesystem::path& root) {
    const auto path = root / std::filesystem::path(std::u8string(u8"io-àccènt.bin"));
    {
        std::ofstream output(path, std::ios::binary);
        for (int index = 0; index < 4096; ++index) output.put(static_cast<char>(index % 251));
    }
    vastloom::safety::ResourceLimits limits;
    limits.ram_hard_limit = 64U * vastloom::MiB;
    limits.vram_hard_limit = 64U * vastloom::MiB;
    limits.staging_hard_limit = 8U * vastloom::MiB;
    limits.scratch_hard_limit = 64U * vastloom::MiB;
    vastloom::safety::ResourceGovernor governor(limits);
    vastloom::io::IoPoolOptions pool_options;
    pool_options.workers = 2U;
    pool_options.max_queued_requests = 4U;
    pool_options.max_request_bytes = 4U * vastloom::MiB;
    vastloom::io::BoundedIoPool pool(pool_options, governor);
    auto io_before = vastloom::io::sample_process_io();
    auto request = pool.submit_read(path, 1024, 2048, "test-read");
    CHECK(request);
    if (request) {
        auto result = request.value().get();
        CHECK(result && result.value().bytes.size() == 2048U);
        if (result) {
            CHECK(std::to_integer<unsigned char>(result.value().bytes.front()) == static_cast<unsigned char>(1024 % 251));
            CHECK(result.value().request_sequence == 1U);
            CHECK(result.value().timing.total_ms >= result.value().timing.service_ms);
        }
    }
    auto io_after = vastloom::io::sample_process_io();
    CHECK(io_before && io_after);
    if (io_before && io_after) {
        auto delta = vastloom::io::process_io_delta(io_before.value(), io_after.value());
        CHECK(delta);
        if (delta) {
            CHECK(delta.value().read_operations >= 1U);
            CHECK(delta.value().process_read_bytes >= 2048U);
#if defined(_WIN32)
            CHECK(!delta.value().storage_read_bytes_available);
#elif defined(__linux__)
            CHECK(delta.value().storage_read_bytes_available);
#endif
        }
    }
    auto out_of_bounds = pool.submit_read(path, 4090, 64, "out-of-bounds");
    CHECK(out_of_bounds);
    if (out_of_bounds) {
        auto result = out_of_bounds.value().get();
        CHECK(!result);
    }
    auto too_large = pool.submit_read(path, 0, 5U * vastloom::MiB, "too-large");
    CHECK(!too_large);

    vastloom::io::IoPoolOptions bypass_options;
    bypass_options.workers = 1U;
    bypass_options.max_queued_requests = 1U;
    bypass_options.max_request_bytes = 4U * vastloom::KiB;
    bypass_options.cache_mode = vastloom::io::IoCacheMode::bypass;
    vastloom::io::BoundedIoPool bypass_pool(bypass_options, governor);
    auto bypass_request = bypass_pool.submit_read(path, 0U, 4U * vastloom::KiB, "cache-bypass-read");
    CHECK(bypass_request);
    if (bypass_request) {
        auto bypass_result = bypass_request.value().get();
#if defined(_WIN32)
        CHECK(bypass_result && bypass_result.value().bytes.size() == 4U * vastloom::KiB);
#elif defined(__linux__)
        CHECK(bypass_result || bypass_result.error().code == vastloom::ErrorCode::unsupported);
#endif
    }
    auto unaligned_bypass = bypass_pool.submit_read(
        path, 1U, 2048U, "unaligned-cache-bypass-read");
    CHECK(unaligned_bypass);
    if (unaligned_bypass) {
        auto unaligned_result = unaligned_bypass.value().get();
#if defined(_WIN32)
        CHECK(unaligned_result && unaligned_result.value().bytes.size() == 2048U);
        if (unaligned_result) CHECK(std::to_integer<unsigned char>(unaligned_result.value().bytes.front()) == 1U);
#elif defined(__linux__)
        CHECK(unaligned_result || unaligned_result.error().code == vastloom::ErrorCode::unsupported);
#endif
    }

    const std::array fault_kinds{
        vastloom::io::IoFaultKind::fail,
        vastloom::io::IoFaultKind::short_read,
        vastloom::io::IoFaultKind::allocation,
        vastloom::io::IoFaultKind::cancelled,
        vastloom::io::IoFaultKind::timeout,
    };
    for (const auto kind : fault_kinds) {
        vastloom::io::IoPoolOptions options;
        options.workers = 1U;
        options.max_queued_requests = 1U;
        options.max_request_bytes = 4U * vastloom::KiB;
        options.request_timeout = kind == vastloom::io::IoFaultKind::timeout
                                      ? std::chrono::milliseconds(1)
                                      : std::chrono::seconds(1);
        options.fault = {kind, 1U, kind == vastloom::io::IoFaultKind::timeout
                                       ? std::chrono::milliseconds(5)
                                       : std::chrono::milliseconds(0)};
        vastloom::io::BoundedIoPool fault_pool(options, governor);
        auto fault_request = fault_pool.submit_read(path, 0U, 1024U, "injected-read");
        CHECK(fault_request);
        if (fault_request) {
            auto result = fault_request.value().get();
            CHECK(!result);
        }
        fault_pool.shutdown();
        CHECK(governor.snapshot().reserved[2] == 0U);
    }
    vastloom::io::IoPoolOptions cancelled_options;
    cancelled_options.max_request_bytes = 4U * vastloom::KiB;
    cancelled_options.cancelled = [] { return true; };
    vastloom::io::BoundedIoPool cancelled_pool(cancelled_options, governor);
    auto cancelled_request = cancelled_pool.submit_read(path, 0U, 1024U, "cancelled-read");
    CHECK(cancelled_request);
    if (cancelled_request) CHECK(!cancelled_request.value().get());
    cancelled_pool.shutdown();
    CHECK(governor.snapshot().reserved[2] == 0U);
}

void test_device_io_counters(const std::filesystem::path& root) {
    const vastloom::io::DeviceIoSnapshot before{
        "test-device-counter", "disk0", "whole-device", 17U, 4096U};
    const vastloom::io::DeviceIoSnapshot after{
        "test-device-counter", "disk0", "whole-device", 20U, 12'288U};
    auto delta = vastloom::io::device_io_delta(before, after);
    CHECK(delta);
    if (delta) {
        CHECK(delta.value().source == "test-device-counter");
        CHECK(delta.value().device_id == "disk0");
        CHECK(delta.value().scope == "whole-device");
        CHECK(delta.value().read_operations == 3U);
        CHECK(delta.value().read_bytes == 8192U);
    }

    auto incompatible = after;
    incompatible.device_id = "disk1";
    CHECK(!vastloom::io::device_io_delta(before, incompatible));
    auto backwards = before;
    backwards.read_operations = 16U;
    CHECK(!vastloom::io::device_io_delta(before, backwards));

    auto live_before = vastloom::io::sample_device_io(root);
    if (live_before) {
        CHECK(!live_before.value().source.empty());
        CHECK(!live_before.value().device_id.empty());
        CHECK(!live_before.value().scope.empty());

        const auto path = root / "device-counter-read.bin";
        {
            std::ofstream output(path, std::ios::binary);
            std::array<char, 4096> bytes{};
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        {
            std::ifstream input(path, std::ios::binary);
            std::array<char, 4096> bytes{};
            input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            CHECK(input.gcount() == static_cast<std::streamsize>(bytes.size()));
        }
        auto live_after = vastloom::io::sample_device_io(root);
        CHECK(live_after);
        if (live_after) CHECK(vastloom::io::device_io_delta(live_before.value(), live_after.value()));
    }
}

void test_trace(const std::filesystem::path& root) {
    const auto path = root / "trace.csv";
    std::ofstream trace(path);
    trace << "token,layer,expert,size_bytes\n"
          << "0,0,1,64MiB\n0,0,2,64MiB\n1,0,1,64MiB\n1,0,3,64MiB\n2,0,1,64MiB\n";
    trace.close();
    vastloom::research::TraceSimulationConfig config;
    config.vram_cache_bytes = 128U * vastloom::MiB;
    config.ram_cache_bytes = 128U * vastloom::MiB;
    auto report = vastloom::research::simulate_trace(path, config);
    CHECK(report && report.value().records == 5U);
    if (report) {
        CHECK(report.value().vram_hits >= 1U);
        CHECK(report.value().storage_misses >= 3U);
        CHECK(report.value().tokens == 3U);
        CHECK(report.value().unique_experts == 3U);
        CHECK(report.value().cold_expert_accesses == 3U);
        CHECK(report.value().reused_expert_accesses == 2U);
        CHECK(std::abs(report.value().reuse_access_ratio - 0.4) < 1e-12);
        CHECK(std::abs(report.value().average_unique_reuse_distance - 1.0) < 1e-12);
        CHECK(report.value().p50_unique_reuse_distance == 1U);
        CHECK(report.value().p95_unique_reuse_distance == 1U);
        CHECK(report.value().maximum_unique_reuse_distance == 1U);
    }
    config.policy = vastloom::research::CachePolicy::none;
    config.vram_cache_bytes = 0U;
    config.ram_cache_bytes = 0U;
    auto no_cache = vastloom::research::simulate_trace(path, config);
    CHECK(no_cache && no_cache.value().storage_misses == 5U && no_cache.value().nvme_bytes == 320U * vastloom::MiB);
    config.policy = vastloom::research::CachePolicy::lfu;
    config.ram_cache_bytes = 128U * vastloom::MiB;
    auto lfu = vastloom::research::simulate_trace(path, config);
    CHECK(lfu && lfu.value().records == 5U);

    const auto telemetry_path = root / "telemetry-trace.csv";
    std::ofstream telemetry(telemetry_path);
    telemetry << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms\n"
              << "0,0,1,12MiB,0.2,nvme,0,12582912,1.5\n";
    telemetry.close();
    auto telemetry_report = vastloom::research::simulate_trace(telemetry_path, config);
    CHECK(telemetry_report && telemetry_report.value().records == 1U);
    const auto prefetch_path = root / "prefetch-telemetry-trace.csv";
    std::ofstream prefetch(prefetch_path);
    prefetch << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms,"
                "prefetched,prefetch_wait_ms,prefetch_overlap_ms\n"
             << "0,0,1,12MiB,0.2,nvme,0,12582912,1.5,1,0.5,1.0\n";
    prefetch.close();
    auto prefetch_report = vastloom::research::simulate_trace(prefetch_path, config);
    CHECK(prefetch_report && prefetch_report.value().records == 1U);
    const auto readiness_path = root / "readiness-telemetry-trace.csv";
    std::ofstream readiness(readiness_path);
    readiness << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms,"
                 "prefetched,prefetch_wait_ms,prefetch_overlap_ms,submitted_ms,queue_ms,ready_ms,"
                 "compute_started_ms,compute_completed_ms,route_order,reduction_order\n"
              << "0,0,1,12MiB,0.3,nvme,0,12582912,1,1,0,0,0,0,10,10,14,0,0\n"
              << "0,0,2,12MiB,0.2,nvme,0,12582912,1,1,0,0,0,0,1,14,17,1,1\n"
              << "0,0,3,12MiB,0.1,nvme,0,12582912,1,1,0,0,0,0,2,17,19,2,2\n";
    readiness.close();
    auto readiness_report = vastloom::research::simulate_trace(readiness_path, config);
    CHECK(readiness_report && readiness_report.value().ready_order_available);
    if (readiness_report) {
        CHECK(readiness_report.value().ready_order_groups == 1U);
        CHECK(readiness_report.value().ready_order_reordered_experts == 3U);
        CHECK(std::abs(readiness_report.value().fixed_order_makespan_ms - 19.0) < 1e-12);
        CHECK(std::abs(readiness_report.value().ready_order_makespan_ms - 14.0) < 1e-12);
        CHECK(std::abs(readiness_report.value().fixed_order_wait_ms - 10.0) < 1e-12);
        CHECK(std::abs(readiness_report.value().ready_order_wait_ms - 5.0) < 1e-12);
        CHECK(std::abs(readiness_report.value().ready_order_saved_ms - 5.0) < 1e-12);
    }
    const auto malformed_telemetry_path = root / "malformed-telemetry-trace.csv";
    std::ofstream malformed_telemetry(malformed_telemetry_path);
    malformed_telemetry << "# actual trace\n"
                        << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms\n"
                        << "0,0,1,12MiB,0.2,ram,1,12582912,1.5\n";
    malformed_telemetry.close();
    CHECK(!vastloom::research::simulate_trace(malformed_telemetry_path, config));
    const auto malformed_prefetch_path = root / "malformed-prefetch-trace.csv";
    std::ofstream malformed_prefetch(malformed_prefetch_path);
    malformed_prefetch << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms,"
                          "prefetched,prefetch_wait_ms,prefetch_overlap_ms\n"
                       << "0,0,1,12MiB,0.2,ram,1,0,0,1,0,0\n";
    malformed_prefetch.close();
    CHECK(!vastloom::research::simulate_trace(malformed_prefetch_path, config));
    const auto malformed_readiness_path = root / "malformed-readiness-trace.csv";
    std::ofstream malformed_readiness(malformed_readiness_path);
    malformed_readiness << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms,"
                           "prefetched,prefetch_wait_ms,prefetch_overlap_ms,submitted_ms,queue_ms,ready_ms,"
                           "compute_started_ms,compute_completed_ms,route_order,reduction_order\n"
                        << "0,0,1,12MiB,0.2,nvme,0,12582912,1,1,0,0,5,0,4,4,5,0,0\n";
    malformed_readiness.close();
    CHECK(!vastloom::research::simulate_trace(malformed_readiness_path, config));
}


void test_tiny_model(const std::filesystem::path& root) {
#ifndef VASTLOOM_SOURCE_DIR
#error "VASTLOOM_SOURCE_DIR must be defined for TinyModel fixture tests"
#endif
    const auto model_path = std::filesystem::path(VASTLOOM_SOURCE_DIR) /
                            "examples/models/vastloom-tinychat-v1.sgtm";
    vastloom::safety::ResourceLimits limits;
    limits.ram_hard_limit = 128U * vastloom::MiB;
    limits.vram_hard_limit = 1U * vastloom::MiB;
    limits.staging_hard_limit = 16U * vastloom::MiB;
    limits.scratch_hard_limit = 64U * vastloom::MiB;
    vastloom::safety::ResourceGovernor governor(limits);

    {
        auto model = vastloom::inference::TinyModel::load(model_path, governor);
        CHECK(model);
        if (!model) return;
        CHECK(model.value().config().model_id == "vastloom/tinychat-v1");
        CHECK(model.value().config().context_length == 128U);
        CHECK(model.value().config().d_model == 64U);
        CHECK(model.value().resident_bytes() > 300U * vastloom::KiB);
        CHECK(governor.snapshot().reserved[0] > model.value().resident_bytes());

        std::string streamed;
        vastloom::inference::GenerationOptions greedy;
        greedy.max_new_tokens = 64U;
        greedy.temperature = 0.0F;
        greedy.top_k = 1U;
        greedy.on_text = [&](std::string_view piece) { streamed.append(piece); };
        auto answer = model.value().generate("Quanto fa 2+2?", greedy, governor);
        CHECK(answer);
        if (answer) {
            CHECK(answer.value().text == "2+2 fa 4.");
            CHECK(streamed == answer.value().text);
            CHECK(answer.value().stats.generated_tokens == 9U);
            CHECK(answer.value().stats.stopped_on_eot);
            CHECK(!answer.value().stats.stopped_on_timeout);
            CHECK(answer.value().stats.prompt_tokens > 0U);
        }

        auto one_token = greedy;
        one_token.max_new_tokens = 1U;
        auto limited = model.value().generate("Ciao", one_token, governor);
        CHECK(limited && limited.value().stats.stopped_on_max_tokens &&
              limited.value().stats.generated_tokens == 1U);

        vastloom::inference::GenerationOptions sampled;
        sampled.max_new_tokens = 32U;
        sampled.temperature = 0.7F;
        sampled.top_k = 20U;
        sampled.seed = 42U;
        auto first = model.value().generate("Ciao", sampled, governor);
        auto second = model.value().generate("Ciao", sampled, governor);
        CHECK(first && second);
        if (first && second) CHECK(first.value().text == second.value().text);

        auto tiny_temperature = sampled;
        tiny_temperature.temperature = std::numeric_limits<float>::denorm_min();
        auto tiny_temperature_result = model.value().generate("Ciao", tiny_temperature, governor);
        CHECK(tiny_temperature_result);

        auto self_test = vastloom::inference::run_tiny_model_self_test(model.value(), governor);
        CHECK(self_test && self_test.value().passed() && self_test.value().cases.size() == 5U);

        std::atomic<unsigned> concurrent_successes{0U};
        std::vector<std::thread> workers;
        for (unsigned index = 0; index < 8U; ++index) {
            workers.emplace_back([&] {
                vastloom::inference::GenerationOptions options;
                options.max_new_tokens = 64U;
                options.temperature = 0.0F;
                options.top_k = 1U;
                auto generated = model.value().generate("Quanto fa 2+2?", options, governor);
                if (generated && generated.value().text == "2+2 fa 4.") {
                    concurrent_successes.fetch_add(1U, std::memory_order_relaxed);
                }
            });
        }
        for (auto& worker : workers) worker.join();
        CHECK(concurrent_successes.load(std::memory_order_relaxed) == 8U);

        std::string too_long(512U, 'x');
        CHECK(!model.value().generate(too_long, greedy, governor));
        CHECK(!model.value().generate("", greedy, governor));

        auto invalid = greedy;
        invalid.max_new_tokens = 0U;
        CHECK(!model.value().generate("Ciao", invalid, governor));
        invalid = greedy;
        invalid.top_k = 0U;
        CHECK(!model.value().generate("Ciao", invalid, governor));
        invalid = greedy;
        invalid.temperature = std::numeric_limits<float>::quiet_NaN();
        CHECK(!model.value().generate("Ciao", invalid, governor));
        invalid = greedy;
        invalid.timeout = std::chrono::milliseconds(0);
        CHECK(!model.value().generate("Ciao", invalid, governor));

        const auto truncated = root / "truncated.sgtm";
        std::filesystem::copy_file(model_path, truncated, std::filesystem::copy_options::overwrite_existing);
        const auto size = std::filesystem::file_size(truncated);
        std::filesystem::resize_file(truncated, size - 17U);
        CHECK(!vastloom::inference::TinyModel::load(truncated, governor));

        const auto empty_id = root / "empty-id.sgtm";
        std::filesystem::copy_file(model_path, empty_id, std::filesystem::copy_options::overwrite_existing);
        {
            std::fstream file(empty_id, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(8 + 13 * 4);
            const std::array<char, 4> zero{};
            file.write(zero.data(), static_cast<std::streamsize>(zero.size()));
        }
        CHECK(!vastloom::inference::TinyModel::load(empty_id, governor));

        const auto non_finite = root / "non-finite.sgtm";
        std::filesystem::copy_file(model_path, non_finite, std::filesystem::copy_options::overwrite_existing);
        {
            std::fstream file(non_finite, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(-4, std::ios::end);
            const std::array<unsigned char, 4> quiet_nan{0x00U, 0x00U, 0xc0U, 0x7fU};
            file.write(reinterpret_cast<const char*>(quiet_nan.data()),
                       static_cast<std::streamsize>(quiet_nan.size()));
        }
        CHECK(!vastloom::inference::TinyModel::load(non_finite, governor));
    }

    CHECK(governor.snapshot().reserved[0] == 0U);

    vastloom::safety::ResourceLimits tiny_limits;
    tiny_limits.ram_hard_limit = 1U * vastloom::MiB;
    tiny_limits.vram_hard_limit = 1U * vastloom::MiB;
    tiny_limits.staging_hard_limit = 1U * vastloom::MiB;
    tiny_limits.scratch_hard_limit = 1U * vastloom::MiB;
    vastloom::safety::ResourceGovernor tiny_governor(tiny_limits);
    CHECK(!vastloom::inference::TinyModel::load(model_path, tiny_governor));
}

void test_heartbeat(const std::filesystem::path& root) {
    const auto path = root / "heartbeat";
    vastloom::safety::HeartbeatRecord heartbeat;
    heartbeat.pid = 1234;
    heartbeat.sequence = 9;
    heartbeat.unix_time_ms = vastloom::safety::unix_time_ms();
    heartbeat.state = vastloom::safety::SafetyState::pressure;
    heartbeat.ram_reserved = 2U * vastloom::GiB;
    CHECK(vastloom::safety::write_heartbeat_atomic(path, heartbeat));
    auto loaded = vastloom::safety::read_heartbeat(path);
    CHECK(loaded && loaded.value().pid == 1234U && loaded.value().sequence == 9U);
    if (loaded) CHECK(loaded.value().state == vastloom::safety::SafetyState::pressure);
}

}  // namespace

int main() {
    const auto root = make_temp_dir();
    test_bytes();
    test_sha256();
    test_json_escape();
    test_bounded_json();
    test_safetensors(root);
    test_gguf(root);
    test_trunk_storage_reference_kernels();
    test_iq1_s_reference_kernel();
    test_iq2_xxs_reference_kernel();
    test_iq2_s_reference_kernel();
    test_q4k_mxfp4_reference_parity();
    test_openai_protocol();
    test_iq3_xxs_reference_kernel();
    test_mxfp4_reference_kernel();
    test_quantized_routed_expert_contract();
    test_logical_tensor_catalog();
    test_deepseek_v4_gguf_expert_bindings();
    test_residency_state_machine();
    test_packed_expert_loader(root);
    test_layer_streaming(root);
    test_imported_checkpoint_and_llama_config(root);
    test_olmoe_config(root);
    test_deepseek_v4_config(root);
    test_deepseek_v4_reference_operators();
    test_deepseek_v4_reference_graph(root);
    test_model_store_and_runtime_policy(root);
    test_gpt2_tokenizer(root);
    test_manifest_and_verifier(root);
    test_governor();
    test_planner();
    test_io_pool(root);
    test_device_io_counters(root);
    test_trace(root);
    test_tiny_model(root);
    test_heartbeat(root);
    std::error_code error;
    std::filesystem::remove_all(root, error);

    if (failures == 0) {
        std::cout << "ALL VASTLOOM TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " TEST(S) FAILED\n";
    return 1;
}
