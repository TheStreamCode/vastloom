#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/hardware/hardware_probe.hpp"
#include "vastloom/inference/gpt2_tokenizer.hpp"
#include "vastloom/inference/deepseek_v4_model.hpp"
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
#include "vastloom/model/model_store.hpp"
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
                                std::nullopt, std::nullopt, false, TensorSourceRange{"shard-a", 0U, 8U}},
        LogicalTensorDescriptor{"expert.weight", TensorDType::f32, {1U}, TensorRole::routed_expert,
                                2U, 7U, false, TensorSourceRange{"shard-a", 8U, 4U}},
        LogicalTensorDescriptor{"expert.scale", TensorDType::f32, {1U}, TensorRole::routed_expert,
                                2U, 7U, false, TensorSourceRange{"shard-a", 12U, 4U}},
        LogicalTensorDescriptor{"expert.bias", TensorDType::f16, {1U}, TensorRole::routed_expert,
                                2U, 7U, false, TensorSourceRange{"shard-b", 0U, 2U}},
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
        if (catalog.experts[0].read_ranges.size() == 2U) {
            CHECK(catalog.experts[0].read_ranges[0].shard == "shard-a" &&
                  catalog.experts[0].read_ranges[0].offset == 8U &&
                  catalog.experts[0].read_ranges[0].bytes == 8U);
        }
    }
    CHECK(vastloom::model::to_string(TensorRole::routed_expert) == "routed_expert");

    auto duplicate = catalog;
    duplicate.tensors.push_back(LogicalTensorDescriptor{
        "embed", TensorDType::f32, {1U}, TensorRole::other,
        std::nullopt, std::nullopt, false, TensorSourceRange{"shard-a", 20U, 4U}});
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

void test_deepseek_v4_reference_graph() {
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
}

void test_model_store_and_runtime_policy(const std::filesystem::path& root) {
    auto store = vastloom::model::ModelStore::initialize(root / "model-store", {20U * vastloom::GiB});
    CHECK(store);
    if (store) {
        CHECK(std::filesystem::is_directory(store.value().layout().original));
        CHECK(store.value().immutable_model_path("org/model", "revision-1"));
        CHECK(!store.value().immutable_model_path("../escape", "revision-1"));
        CHECK(!store.value().preflight_write(std::numeric_limits<vastloom::ByteCount>::max()));
        CHECK(vastloom::model::ModelStore::open(root / "model-store", {20U * vastloom::GiB}));
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
    CHECK(vastloom::model::ModelStore::open(legacy_store, {20U * vastloom::GiB}));

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
        CHECK(policy.value().cpu_threads == 6U);
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
    manifest.estimated_expert_bytes = 48U * vastloom::MiB;
    manifest.minimum_ram_bytes = 16U * vastloom::GiB;
    manifest.recommended_ram_bytes = 32U * vastloom::GiB;
    manifest.recommended_vram_bytes = 12U * vastloom::GiB;

    auto plan = vastloom::planner::build_plan(hardware, manifest, vastloom::planner::Profile::balanced);
    CHECK(plan && plan.value().compatible);
    if (plan) {
        CHECK(plan.value().safe_ram_budget < hardware.memory.total_bytes);
        CHECK(plan.value().safe_vram_budget < hardware.gpus.front().total_vram_bytes);
        CHECK(plan.value().governor_limits.ram_hard_limit == plan.value().safe_ram_budget);
        CHECK(plan.value().strategy == "multitier-predictive");
    }
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
    test_logical_tensor_catalog();
    test_imported_checkpoint_and_llama_config(root);
    test_olmoe_config(root);
    test_deepseek_v4_config(root);
    test_deepseek_v4_reference_operators();
    test_deepseek_v4_reference_graph();
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
