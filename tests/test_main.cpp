#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "stargate/core/bytes.hpp"
#include "stargate/hardware/hardware_probe.hpp"
#include "stargate/inference/gpt2_tokenizer.hpp"
#include "stargate/inference/llama_model.hpp"
#include "stargate/inference/olmoe_model.hpp"
#include "stargate/inference/olmoe_runtime.hpp"
#include "stargate/inference/tiny_model.hpp"
#include "stargate/io/bounded_io_pool.hpp"
#include "stargate/io/process_io_counters.hpp"
#include "stargate/model/manifest.hpp"
#include "stargate/model/imported_checkpoint.hpp"
#include "stargate/model/model_store.hpp"
#include "stargate/model/safetensors.hpp"
#include "stargate/model/verifier.hpp"
#include "stargate/planner/execution_planner.hpp"
#include "stargate/research/trace_simulator.hpp"
#include "stargate/safety/heartbeat.hpp"
#include "stargate/safety/resource_governor.hpp"
#include "stargate/safety/runtime_policy.hpp"
#include "stargate/util/json.hpp"
#include "stargate/util/sha256.hpp"
#include "stargate/util/string.hpp"

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
    const auto path = std::filesystem::temp_directory_path() / ("stargate-tests-" + std::to_string(stamp));
    std::filesystem::create_directories(path);
    return path;
}

void test_bytes() {
    auto parsed = stargate::parse_bytes("32GiB");
    CHECK(parsed && parsed.value() == 32U * stargate::GiB);
    auto decimal = stargate::parse_bytes("1.5 GB");
    CHECK(decimal && decimal.value() == 1'500'000'000ULL);
    CHECK(!stargate::parse_bytes("-1GiB"));
    CHECK(!stargate::parse_bytes("10 bananas"));
    CHECK(stargate::format_bytes(2U * stargate::GiB) == "2.00 GiB");
}

void test_sha256() {
    const std::string input = "abc";
    stargate::util::Sha256 hash;
    hash.update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(input.data()), input.size()));
    CHECK(hash.finalize_hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}


void test_json_escape() {
    CHECK(stargate::util::json_escape("quote\" slash\\ newline\n") ==
          "quote\\\" slash\\\\ newline\\n");
    const std::string valid_utf8 = "Qual è la capitale?";
    CHECK(stargate::util::json_escape(valid_utf8) == valid_utf8);
    const std::string invalid_utf8{static_cast<char>(0xf1), 'x', static_cast<char>(0x80)};
    CHECK(stargate::util::json_escape(invalid_utf8) == "\\ufffdx\\ufffd");
}

void test_bounded_json() {
    auto parsed = stargate::util::parse_json(R"({"name":"Stargate","value":42,"unicode":"\u00e8","items":[true,null]})");
    CHECK(parsed && parsed.value().is_object());
    if (parsed) {
        const auto* name = parsed.value().find("name");
        const auto* value = parsed.value().find("value");
        const auto* unicode = parsed.value().find("unicode");
        CHECK(name != nullptr && name->as_string() && name->as_string().value() == "Stargate");
        CHECK(value != nullptr && value->as_u64() && value->as_u64().value() == 42U);
        CHECK(unicode != nullptr && unicode->as_string() && unicode->as_string().value() == "è");
    }
    CHECK(!stargate::util::parse_json(R"({"duplicate":1,"duplicate":2})"));
    CHECK(!stargate::util::parse_json(R"([1,])"));
    stargate::util::JsonLimits limits;
    limits.max_input_bytes = 4U;
    CHECK(!stargate::util::parse_json("[1,2]", limits));
    limits = {};
    limits.max_depth = 2U;
    CHECK(!stargate::util::parse_json("[[[0]]]", limits));
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
    auto shard = stargate::model::SafetensorsFile::open(checkpoint / "model.safetensors");
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
    auto catalog = stargate::model::inspect_safetensors_checkpoint(checkpoint);
    CHECK(catalog && catalog.value().shards.size() == 1U && catalog.value().tensors.size() == 1U);

    const std::string invalid_header = R"({"weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,9]}})";
    {
        std::ofstream output(checkpoint / "invalid.safetensors", std::ios::binary);
        write_u64_le(output, invalid_header.size());
        output.write(invalid_header.data(), static_cast<std::streamsize>(invalid_header.size()));
        output.write("123456789", 9);
    }
    CHECK(!stargate::model::SafetensorsFile::open(checkpoint / "invalid.safetensors"));
}

void test_imported_checkpoint_and_llama_config(const std::filesystem::path& root) {
    const auto imported_root = root / "imported-checkpoint";
    std::filesystem::create_directories(imported_root);
    {
        std::ofstream(imported_root / "data.bin", std::ios::binary) << "hello";
        std::ofstream manifest(imported_root / "stargate-import.json");
        manifest << R"({"format":"stargate.hf-import.v1","model_id":"test/model","revision":"abc","metadata_only":false,"files":[{"name":"data.bin","size":5,"sha256":"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}]})";
    }
    auto verified = stargate::model::verify_imported_checkpoint(imported_root, true);
    CHECK(verified && verified.value().full_hash_verified && verified.value().total_bytes == 5U);
    std::ofstream(imported_root / "data.bin", std::ios::binary | std::ios::trunc) << "HELLO";
    CHECK(!stargate::model::verify_imported_checkpoint(imported_root, true));

    const auto config_path = root / "llama-config.json";
    {
        std::ofstream config(config_path);
        config << R"({"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_act":"silu","torch_dtype":"bfloat16","vocab_size":49152,"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,"max_position_embeddings":8192,"bos_token_id":1,"eos_token_id":2,"pad_token_id":2,"rms_norm_eps":0.00001,"rope_theta":100000,"tie_word_embeddings":true,"attention_bias":false,"mlp_bias":false,"rope_interleaved":false,"rope_scaling":null})";
    }
    auto llama = stargate::inference::load_llama_config(config_path);
    CHECK(llama && llama.value().hidden_size == 576U && llama.value().num_key_value_heads == 3U);
    {
        std::ofstream config(config_path, std::ios::trunc);
        config << R"({"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_act":"silu","torch_dtype":"bfloat16"})";
    }
    CHECK(!stargate::inference::load_llama_config(config_path));
}

void test_olmoe_config(const std::filesystem::path& root) {
    const auto path = root / "olmoe-config.json";
    {
        std::ofstream config(path);
        config << R"({"architectures":["OlmoeForCausalLM"],"attention_bias":false,"clip_qkv":null,"eos_token_id":50279,"hidden_act":"silu","hidden_size":2048,"intermediate_size":1024,"max_position_embeddings":4096,"model_type":"olmoe","norm_topk_prob":false,"num_attention_heads":16,"num_experts":64,"num_experts_per_tok":8,"num_hidden_layers":16,"num_key_value_heads":16,"pad_token_id":1,"rope_scaling":null,"rope_theta":10000.0,"tie_word_embeddings":false,"torch_dtype":"bfloat16","vocab_size":50304})";
    }
    auto config = stargate::inference::load_olmoe_config(path);
    CHECK(config && config.value().num_experts == 64U && config.value().num_experts_per_token == 8U);
    {
        std::ofstream invalid(path, std::ios::trunc);
        invalid << R"({"architectures":["OlmoeForCausalLM"],"model_type":"olmoe","hidden_act":"silu","torch_dtype":"bfloat16"})";
    }
    CHECK(!stargate::inference::load_olmoe_config(path));
    CHECK(stargate::inference::parse_expert_cache_policy("no-cache") &&
          stargate::inference::parse_expert_cache_policy("LFU") &&
          stargate::inference::parse_expert_cache_policy("static-hot") &&
          !stargate::inference::parse_expert_cache_policy("random"));
}

void test_model_store_and_runtime_policy(const std::filesystem::path& root) {
    auto store = stargate::model::ModelStore::initialize(root / "model-store", {20U * stargate::GiB});
    CHECK(store);
    if (store) {
        CHECK(std::filesystem::is_directory(store.value().layout().original));
        CHECK(store.value().immutable_model_path("org/model", "revision-1"));
        CHECK(!store.value().immutable_model_path("../escape", "revision-1"));
        CHECK(!store.value().preflight_write(std::numeric_limits<stargate::ByteCount>::max()));
        CHECK(stargate::model::ModelStore::open(root / "model-store", {20U * stargate::GiB}));
    }

    stargate::hardware::HardwareSnapshot hardware;
    hardware.cpu.logical_cores = 24U;
    hardware.memory.total_bytes = 32U * stargate::GiB;
    hardware.memory.available_bytes = 20U * stargate::GiB;
    hardware.storage.available_bytes = 1U * stargate::TiB;
    hardware.gpus.push_back({"CUDA", "test", "test", 12U * stargate::GiB, 10U * stargate::GiB, {}, {}});
    auto policy = stargate::safety::make_safe_runtime_policy(hardware);
    CHECK(policy && policy.value().resources.ram_hard_limit == 12U * stargate::GiB);
    if (policy) {
        CHECK(policy.value().cpu_threads == 6U);
        CHECK(policy.value().io_workers == 2U && policy.value().max_queued_io == 4U);
        CHECK(policy.value().max_io_request_bytes == 64U * stargate::MiB);
        CHECK(policy.value().storage_reserve_bytes == 300U * stargate::GiB);
    }
    hardware.memory.available_bytes = 8U * stargate::GiB;
    CHECK(!stargate::safety::make_safe_runtime_policy(hardware));
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
    auto tokenizer = stargate::inference::Gpt2Tokenizer::load(tokenizer_root);
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
    manifest << "format=stargate.manifest.v1\n"
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

    auto loaded = stargate::model::load_manifest(root / "model.sgm");
    CHECK(loaded);
    if (loaded) {
        auto quick = stargate::model::verify_model(loaded.value(), root, stargate::model::VerificationMode::quick);
        CHECK(quick && quick.value().ok);
        auto full = stargate::model::verify_model(loaded.value(), root, stargate::model::VerificationMode::full);
        CHECK(full && full.value().ok && full.value().bytes_checked == 5U);

        std::ofstream(root / "weights.bin", std::ios::binary | std::ios::trunc) << "HELLO";
        auto corrupted = stargate::model::verify_model(loaded.value(), root, stargate::model::VerificationMode::full);
        CHECK(corrupted && !corrupted.value().ok && corrupted.value().hash_mismatches == 1U);
        std::ofstream(root / "weights.bin", std::ios::binary | std::ios::trunc) << "hello";

        const auto outside = root.parent_path() / (root.filename().string() + "-outside.bin");
        std::ofstream(outside, std::ios::binary) << "hello";
        std::error_code symlink_error;
        std::filesystem::create_symlink(outside, root / "escape-link.bin", symlink_error);
        if (!symlink_error) {
            std::ofstream symlink_manifest(root / "symlink.sgm");
            symlink_manifest << "format=stargate.manifest.v1\nmodel_id=x\narchitecture=x\nmetadata_only=false\nmodel_total_bytes=5\nchunk_count=1\n"
                             << "chunk\tescape-link.bin\t5\t2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824\tdense\tfp16\t-\t-\ttrue\n";
            symlink_manifest.close();
            auto symlink_loaded = stargate::model::load_manifest(root / "symlink.sgm");
            CHECK(symlink_loaded);
            if (symlink_loaded) {
                auto escaped = stargate::model::verify_model(symlink_loaded.value(), root, stargate::model::VerificationMode::quick);
                CHECK(escaped && !escaped.value().ok && escaped.value().unsafe_paths == 1U);
            }
        }
        std::filesystem::remove(outside, symlink_error);
    }

    std::ofstream unsafe(root / "unsafe.sgm");
    unsafe << "format=stargate.manifest.v1\nmodel_id=x\narchitecture=x\nmetadata_only=false\nchunk_count=1\n"
           << "chunk\t../escape.bin\t1\t0000000000000000000000000000000000000000000000000000000000000000\tdense\tfp16\t-\t-\ttrue\n";
    unsafe.close();
    CHECK(!stargate::model::load_manifest(root / "unsafe.sgm"));
}

void test_governor() {
    stargate::safety::ResourceLimits limits;
    limits.ram_hard_limit = 8U * stargate::GiB;
    limits.vram_hard_limit = 4U * stargate::GiB;
    limits.staging_hard_limit = 1U * stargate::GiB;
    limits.scratch_hard_limit = 2U * stargate::GiB;
    limits.minimum_system_ram_available = 2U * stargate::GiB;
    limits.minimum_storage_available = 20U * stargate::GiB;
    limits.recovery_samples = 2;
    stargate::safety::ResourceGovernor governor(limits);

    auto first = governor.try_reserve(stargate::safety::ResourceKind::ram, 6U * stargate::GiB, "first");
    CHECK(first);
    auto denied = governor.try_reserve(stargate::safety::ResourceKind::ram, 3U * stargate::GiB, "oversubscribe");
    CHECK(!denied);
    if (first) first.value().release();
    auto second = governor.try_reserve(stargate::safety::ResourceKind::ram, 3U * stargate::GiB, "after-release");
    CHECK(second);

    auto state = governor.update_pressure({400U * stargate::MiB, 2U * stargate::GiB, 100U * stargate::GiB});
    CHECK(state == stargate::safety::SafetyState::emergency);
    CHECK(!governor.allows_new_work());
    state = governor.update_pressure({8U * stargate::GiB, 2U * stargate::GiB, 100U * stargate::GiB});
    state = governor.update_pressure({8U * stargate::GiB, 2U * stargate::GiB, 100U * stargate::GiB});
    CHECK(state == stargate::safety::SafetyState::critical);
}

void test_planner() {
    stargate::hardware::HardwareSnapshot hardware;
    hardware.operating_system = "test";
    hardware.cpu = {"Ryzen 9 7900X", 24, true, false, true};
    hardware.memory = {32U * stargate::GiB, 26U * stargate::GiB};
    hardware.gpus.push_back({"CUDA", "RTX 4070 SUPER", "test", 12U * stargate::GiB, 11U * stargate::GiB, 60.0, 220.0});
    hardware.storage = {".", 2U * stargate::TiB, 1500U * stargate::GiB, 1500U * stargate::GiB, std::nullopt};

    stargate::model::ModelManifest manifest;
    manifest.model_id = "synthetic";
    manifest.architecture = "moe";
    manifest.metadata_only = true;
    manifest.requires_avx2 = true;
    manifest.supports_speculation = true;
    manifest.model_total_bytes = 180U * stargate::GiB;
    manifest.required_storage_bytes = 200U * stargate::GiB;
    manifest.dense_bytes = 18U * stargate::GiB;
    manifest.estimated_expert_bytes = 48U * stargate::MiB;
    manifest.minimum_ram_bytes = 16U * stargate::GiB;
    manifest.recommended_ram_bytes = 32U * stargate::GiB;
    manifest.recommended_vram_bytes = 12U * stargate::GiB;

    auto plan = stargate::planner::build_plan(hardware, manifest, stargate::planner::Profile::balanced);
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
    stargate::safety::ResourceLimits limits;
    limits.ram_hard_limit = 64U * stargate::MiB;
    limits.vram_hard_limit = 64U * stargate::MiB;
    limits.staging_hard_limit = 8U * stargate::MiB;
    limits.scratch_hard_limit = 64U * stargate::MiB;
    stargate::safety::ResourceGovernor governor(limits);
    stargate::io::IoPoolOptions pool_options;
    pool_options.workers = 2U;
    pool_options.max_queued_requests = 4U;
    pool_options.max_request_bytes = 4U * stargate::MiB;
    stargate::io::BoundedIoPool pool(pool_options, governor);
    auto io_before = stargate::io::sample_process_io();
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
    auto io_after = stargate::io::sample_process_io();
    CHECK(io_before && io_after);
    if (io_before && io_after) {
        auto delta = stargate::io::process_io_delta(io_before.value(), io_after.value());
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
    auto too_large = pool.submit_read(path, 0, 5U * stargate::MiB, "too-large");
    CHECK(!too_large);

    const std::array fault_kinds{
        stargate::io::IoFaultKind::fail,
        stargate::io::IoFaultKind::short_read,
        stargate::io::IoFaultKind::allocation,
        stargate::io::IoFaultKind::cancelled,
        stargate::io::IoFaultKind::timeout,
    };
    for (const auto kind : fault_kinds) {
        stargate::io::IoPoolOptions options;
        options.workers = 1U;
        options.max_queued_requests = 1U;
        options.max_request_bytes = 4U * stargate::KiB;
        options.request_timeout = kind == stargate::io::IoFaultKind::timeout
                                      ? std::chrono::milliseconds(1)
                                      : std::chrono::seconds(1);
        options.fault = {kind, 1U, kind == stargate::io::IoFaultKind::timeout
                                       ? std::chrono::milliseconds(5)
                                       : std::chrono::milliseconds(0)};
        stargate::io::BoundedIoPool fault_pool(options, governor);
        auto fault_request = fault_pool.submit_read(path, 0U, 1024U, "injected-read");
        CHECK(fault_request);
        if (fault_request) {
            auto result = fault_request.value().get();
            CHECK(!result);
        }
        fault_pool.shutdown();
        CHECK(governor.snapshot().reserved[2] == 0U);
    }
    stargate::io::IoPoolOptions cancelled_options;
    cancelled_options.max_request_bytes = 4U * stargate::KiB;
    cancelled_options.cancelled = [] { return true; };
    stargate::io::BoundedIoPool cancelled_pool(cancelled_options, governor);
    auto cancelled_request = cancelled_pool.submit_read(path, 0U, 1024U, "cancelled-read");
    CHECK(cancelled_request);
    if (cancelled_request) CHECK(!cancelled_request.value().get());
    cancelled_pool.shutdown();
    CHECK(governor.snapshot().reserved[2] == 0U);
}

void test_trace(const std::filesystem::path& root) {
    const auto path = root / "trace.csv";
    std::ofstream trace(path);
    trace << "token,layer,expert,size_bytes\n"
          << "0,0,1,64MiB\n0,0,2,64MiB\n1,0,1,64MiB\n1,0,3,64MiB\n2,0,1,64MiB\n";
    trace.close();
    stargate::research::TraceSimulationConfig config;
    config.vram_cache_bytes = 128U * stargate::MiB;
    config.ram_cache_bytes = 128U * stargate::MiB;
    auto report = stargate::research::simulate_trace(path, config);
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
    config.policy = stargate::research::CachePolicy::none;
    config.vram_cache_bytes = 0U;
    config.ram_cache_bytes = 0U;
    auto no_cache = stargate::research::simulate_trace(path, config);
    CHECK(no_cache && no_cache.value().storage_misses == 5U && no_cache.value().nvme_bytes == 320U * stargate::MiB);
    config.policy = stargate::research::CachePolicy::lfu;
    config.ram_cache_bytes = 128U * stargate::MiB;
    auto lfu = stargate::research::simulate_trace(path, config);
    CHECK(lfu && lfu.value().records == 5U);

    const auto telemetry_path = root / "telemetry-trace.csv";
    std::ofstream telemetry(telemetry_path);
    telemetry << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms\n"
              << "0,0,1,12MiB,0.2,nvme,0,12582912,1.5\n";
    telemetry.close();
    auto telemetry_report = stargate::research::simulate_trace(telemetry_path, config);
    CHECK(telemetry_report && telemetry_report.value().records == 1U);
    const auto prefetch_path = root / "prefetch-telemetry-trace.csv";
    std::ofstream prefetch(prefetch_path);
    prefetch << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms,"
                "prefetched,prefetch_wait_ms,prefetch_overlap_ms\n"
             << "0,0,1,12MiB,0.2,nvme,0,12582912,1.5,1,0.5,1.0\n";
    prefetch.close();
    auto prefetch_report = stargate::research::simulate_trace(prefetch_path, config);
    CHECK(prefetch_report && prefetch_report.value().records == 1U);
    const auto malformed_telemetry_path = root / "malformed-telemetry-trace.csv";
    std::ofstream malformed_telemetry(malformed_telemetry_path);
    malformed_telemetry << "# actual trace\n"
                        << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms\n"
                        << "0,0,1,12MiB,0.2,ram,1,12582912,1.5\n";
    malformed_telemetry.close();
    CHECK(!stargate::research::simulate_trace(malformed_telemetry_path, config));
    const auto malformed_prefetch_path = root / "malformed-prefetch-trace.csv";
    std::ofstream malformed_prefetch(malformed_prefetch_path);
    malformed_prefetch << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms,"
                          "prefetched,prefetch_wait_ms,prefetch_overlap_ms\n"
                       << "0,0,1,12MiB,0.2,ram,1,0,0,1,0,0\n";
    malformed_prefetch.close();
    CHECK(!stargate::research::simulate_trace(malformed_prefetch_path, config));
}


void test_tiny_model(const std::filesystem::path& root) {
#ifndef STARGATE_SOURCE_DIR
#error "STARGATE_SOURCE_DIR must be defined for TinyModel fixture tests"
#endif
    const auto model_path = std::filesystem::path(STARGATE_SOURCE_DIR) /
                            "examples/models/stargate-tinychat-v1.sgtm";
    stargate::safety::ResourceLimits limits;
    limits.ram_hard_limit = 128U * stargate::MiB;
    limits.vram_hard_limit = 1U * stargate::MiB;
    limits.staging_hard_limit = 16U * stargate::MiB;
    limits.scratch_hard_limit = 64U * stargate::MiB;
    stargate::safety::ResourceGovernor governor(limits);

    {
        auto model = stargate::inference::TinyModel::load(model_path, governor);
        CHECK(model);
        if (!model) return;
        CHECK(model.value().config().model_id == "stargate/tinychat-v1");
        CHECK(model.value().config().context_length == 128U);
        CHECK(model.value().config().d_model == 64U);
        CHECK(model.value().resident_bytes() > 300U * stargate::KiB);
        CHECK(governor.snapshot().reserved[0] > model.value().resident_bytes());

        std::string streamed;
        stargate::inference::GenerationOptions greedy;
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

        stargate::inference::GenerationOptions sampled;
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

        auto self_test = stargate::inference::run_tiny_model_self_test(model.value(), governor);
        CHECK(self_test && self_test.value().passed() && self_test.value().cases.size() == 5U);

        std::atomic<unsigned> concurrent_successes{0U};
        std::vector<std::thread> workers;
        for (unsigned index = 0; index < 8U; ++index) {
            workers.emplace_back([&] {
                stargate::inference::GenerationOptions options;
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
        CHECK(!stargate::inference::TinyModel::load(truncated, governor));

        const auto empty_id = root / "empty-id.sgtm";
        std::filesystem::copy_file(model_path, empty_id, std::filesystem::copy_options::overwrite_existing);
        {
            std::fstream file(empty_id, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(8 + 13 * 4);
            const std::array<char, 4> zero{};
            file.write(zero.data(), static_cast<std::streamsize>(zero.size()));
        }
        CHECK(!stargate::inference::TinyModel::load(empty_id, governor));

        const auto non_finite = root / "non-finite.sgtm";
        std::filesystem::copy_file(model_path, non_finite, std::filesystem::copy_options::overwrite_existing);
        {
            std::fstream file(non_finite, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(-4, std::ios::end);
            const std::array<unsigned char, 4> quiet_nan{0x00U, 0x00U, 0xc0U, 0x7fU};
            file.write(reinterpret_cast<const char*>(quiet_nan.data()),
                       static_cast<std::streamsize>(quiet_nan.size()));
        }
        CHECK(!stargate::inference::TinyModel::load(non_finite, governor));
    }

    CHECK(governor.snapshot().reserved[0] == 0U);

    stargate::safety::ResourceLimits tiny_limits;
    tiny_limits.ram_hard_limit = 1U * stargate::MiB;
    tiny_limits.vram_hard_limit = 1U * stargate::MiB;
    tiny_limits.staging_hard_limit = 1U * stargate::MiB;
    tiny_limits.scratch_hard_limit = 1U * stargate::MiB;
    stargate::safety::ResourceGovernor tiny_governor(tiny_limits);
    CHECK(!stargate::inference::TinyModel::load(model_path, tiny_governor));
}

void test_heartbeat(const std::filesystem::path& root) {
    const auto path = root / "heartbeat";
    stargate::safety::HeartbeatRecord heartbeat;
    heartbeat.pid = 1234;
    heartbeat.sequence = 9;
    heartbeat.unix_time_ms = stargate::safety::unix_time_ms();
    heartbeat.state = stargate::safety::SafetyState::pressure;
    heartbeat.ram_reserved = 2U * stargate::GiB;
    CHECK(stargate::safety::write_heartbeat_atomic(path, heartbeat));
    auto loaded = stargate::safety::read_heartbeat(path);
    CHECK(loaded && loaded.value().pid == 1234U && loaded.value().sequence == 9U);
    if (loaded) CHECK(loaded.value().state == stargate::safety::SafetyState::pressure);
}

}  // namespace

int main() {
    const auto root = make_temp_dir();
    test_bytes();
    test_sha256();
    test_json_escape();
    test_bounded_json();
    test_safetensors(root);
    test_imported_checkpoint_and_llama_config(root);
    test_olmoe_config(root);
    test_model_store_and_runtime_policy(root);
    test_gpt2_tokenizer(root);
    test_manifest_and_verifier(root);
    test_governor();
    test_planner();
    test_io_pool(root);
    test_trace(root);
    test_tiny_model(root);
    test_heartbeat(root);
    std::error_code error;
    std::filesystem::remove_all(root, error);

    if (failures == 0) {
        std::cout << "ALL STARGATE TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " TEST(S) FAILED\n";
    return 1;
}
