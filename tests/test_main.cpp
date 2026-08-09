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
#include "stargate/inference/tiny_model.hpp"
#include "stargate/io/bounded_io_pool.hpp"
#include "stargate/model/manifest.hpp"
#include "stargate/model/verifier.hpp"
#include "stargate/planner/execution_planner.hpp"
#include "stargate/research/trace_simulator.hpp"
#include "stargate/safety/heartbeat.hpp"
#include "stargate/safety/resource_governor.hpp"
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
    const auto path = root / "io.bin";
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
    stargate::io::BoundedIoPool pool({2, 4, 4U * stargate::MiB}, governor);
    auto request = pool.submit_read(path, 1024, 2048, "test-read");
    CHECK(request);
    if (request) {
        auto result = request.value().get();
        CHECK(result && result.value().bytes.size() == 2048U);
        CHECK(std::to_integer<unsigned char>(result.value().bytes.front()) == static_cast<unsigned char>(1024 % 251));
    }
    auto out_of_bounds = pool.submit_read(path, 4090, 64, "out-of-bounds");
    CHECK(out_of_bounds);
    if (out_of_bounds) {
        auto result = out_of_bounds.value().get();
        CHECK(!result);
    }
    auto too_large = pool.submit_read(path, 0, 5U * stargate::MiB, "too-large");
    CHECK(!too_large);
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
    }
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
