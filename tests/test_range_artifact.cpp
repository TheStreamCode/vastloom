#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "vastloom/inference/deepseek_v4_gguf.hpp"
#include "vastloom/inference/deepseek_v4_range_artifact.hpp"
#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/model/bound_range_loader.hpp"
#include "vastloom/model/range_artifact.hpp"
#include "vastloom/model/residency.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"
#include "vastloom/util/sha256.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ << ": " #condition    \
                      << '\n';                                                                     \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

std::filesystem::path make_temp_dir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
                      ("vastloom-range-artifact-tests-" + std::to_string(stamp));
    std::filesystem::create_directories(root);
    return root;
}

std::string range_name(std::size_t index) {
    std::ostringstream output;
    output << "ranges/" << std::setfill('0') << std::setw(4) << index << ".bin";
    return output.str();
}

void write_bytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::byte> read_range(const std::filesystem::path& path,
                                  vastloom::ByteCount offset,
                                  vastloom::ByteCount bytes) {
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
    return result;
}

struct Fixture {
    std::filesystem::path root;
    std::filesystem::path source;
    std::filesystem::path artifact;
    vastloom::inference::DeepseekV4GgufCatalogReport report;
    vastloom::inference::DeepseekV4LayerProbePlan plan;
    vastloom::inference::DeepseekV4GgufCheckpointSpec spec;
    std::string source_hash;
};

std::string artifact_manifest(const Fixture& fixture,
                              std::uint64_t first_offset_delta = 0U,
                              std::optional<std::string> source_hash_override = std::nullopt,
                              bool execution_supported = false,
                              std::string_view first_artifact = "ranges/0000.bin") {
    std::ostringstream output;
    output << "{\"complete_checkpoint_verified\":false,"
              "\"execution_supported\":"
           << (execution_supported ? "true" : "false")
           << ",\"format\":\"vastloom.range-artifact.v1\",\"layer\":0,"
              "\"model_id\":\"test/deepseek-range-artifact\",\"payload_bytes\":"
           << fixture.plan.trunk_bytes
           << ",\"range_count\":" << fixture.plan.trunk_ranges.size()
           << ",\"ranges\":[";
    for (std::size_t index = 0U; index < fixture.plan.trunk_ranges.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& planned = fixture.plan.trunk_ranges[index];
        const auto local = fixture.artifact / range_name(index);
        const auto local_hash = vastloom::util::sha256_file(local);
        CHECK(local_hash);
        output << "{\"artifact\":\""
               << (index == 0U ? first_artifact : range_name(index))
               << "\",\"bytes\":" << planned.source.bytes
               << ",\"label\":\"trunk[" << index << "] " << planned.tensor_name
               << "\",\"role\":\"trunk\",\"sha256\":\""
               << (local_hash ? local_hash.value() : std::string(64U, '0'))
               << "\",\"source_bytes\":" << planned.source.bytes
               << ",\"source_object\":\"model-00001.gguf\","
                  "\"source_object_bytes\":64,\"source_object_sha256\":\""
               << source_hash_override.value_or(fixture.source_hash)
               << "\",\"source_offset\":"
               << planned.source.offset + (index == 0U ? first_offset_delta : 0U)
               << ",\"source_path\":\"weights/model-00001.gguf\"}";
    }
    output << "],\"revision\":\"fixture-revision\","
              "\"routing_authority_verified\":false,\"scope\":\"trunk\","
              "\"source_plan_format\":\"vastloom.deepseek_v4.layer_probe_plan.v1\","
              "\"source_plan_sha256\":\""
           << std::string(64U, 'a')
           << "\",\"verification\":"
              "\"pinned-revision-content-range-plus-local-sha256\"}";
    return output.str();
}

void write_manifest(const Fixture& fixture, const std::string& document) {
    std::ofstream output(fixture.artifact / vastloom::model::kRangeArtifactManifestName,
                         std::ios::binary | std::ios::trunc);
    output << document;
}

Fixture make_fixture(const std::filesystem::path& root,
                     vastloom::ByteCount tile_limit = 4U) {
    Fixture fixture;
    fixture.root = root;
    std::filesystem::create_directories(root);
    fixture.source = root / "model-00001.gguf";
    std::array<std::byte, 64U> source_bytes{};
    for (std::size_t index = 0U; index < source_bytes.size(); ++index) {
        source_bytes[index] = static_cast<std::byte>(index);
    }
    source_bytes[36U] = std::byte{0x00U};
    source_bytes[37U] = std::byte{0x00U};
    source_bytes[38U] = std::byte{0xc0U};
    source_bytes[39U] = std::byte{0x3fU};
    write_bytes(fixture.source, source_bytes);
    auto source_hash = vastloom::util::sha256_file(fixture.source);
    CHECK(source_hash);
    fixture.source_hash = source_hash ? source_hash.value() : std::string(64U, '0');

    vastloom::model::LogicalTensorCatalog catalog;
    catalog.model_id = "test/deepseek-range-artifact";
    catalog.revision = "fixture-revision";
    catalog.metadata_only = true;
    catalog.tensors = {
        {"blk.0.attn.weight", vastloom::model::TensorDType::f32, {10U},
         vastloom::model::TensorRole::attention, 0U, std::nullopt, false,
         {fixture.source, 4U, 10U}, 0U},
        {"blk.0.ffn_norm.weight", vastloom::model::TensorDType::f32, {1U},
         vastloom::model::TensorRole::normalization, 0U, std::nullopt, false,
         {fixture.source, 36U, 4U}, 0U},
        {"blk.0.ffn_gate_exps.weight.expert.0", vastloom::model::TensorDType::f32, {4U},
         vastloom::model::TensorRole::routed_expert, 0U, 0U, false,
         {fixture.source, 20U, 4U}, 0U},
        {"blk.0.ffn_up_exps.weight.expert.0", vastloom::model::TensorDType::f32, {4U},
         vastloom::model::TensorRole::routed_expert, 0U, 0U, false,
         {fixture.source, 24U, 4U}, 0U},
        {"blk.0.ffn_down_exps.weight.expert.0", vastloom::model::TensorDType::f32, {4U},
         vastloom::model::TensorRole::routed_expert, 0U, 0U, false,
         {fixture.source, 28U, 4U}, 0U},
    };
    CHECK(vastloom::model::finalize_logical_tensor_catalog(catalog));
    fixture.report.catalog = std::move(catalog);
    fixture.report.layers = 1U;
    fixture.report.experts_per_layer = 1U;
    fixture.report.top_k = 1U;
    fixture.report.imported_manifest_verified = true;
    fixture.report.imported_file_hashes_verified = true;
    fixture.report.full_payload_present = false;
    fixture.report.executable_source_binding_verified = false;
    const std::array<std::uint32_t, 1U> selected{0U};
    auto plan = vastloom::inference::build_deepseek_v4_layer_probe_plan(
        fixture.report, 0U, selected, tile_limit, 16U);
    CHECK(plan);
    if (plan) fixture.plan = plan.take_value();

    fixture.spec.model_id = fixture.report.catalog.model_id;
    fixture.spec.revision = fixture.report.catalog.revision;
    fixture.spec.layers = 1U;
    fixture.spec.experts_per_layer = 1U;
    fixture.spec.top_k = 1U;
    fixture.spec.shards = {{"weights/model-00001.gguf", 64U, fixture.source_hash,
                            "headers/model-00001.gguf.header", 16U,
                            std::string(64U, 'b')}};

    fixture.artifact = root / "artifact";
    std::filesystem::create_directories(fixture.artifact / "ranges");
    for (std::size_t index = 0U; index < fixture.plan.trunk_ranges.size(); ++index) {
        const auto& range = fixture.plan.trunk_ranges[index].source;
        const auto payload = read_range(fixture.source, range.offset, range.bytes);
        write_bytes(fixture.artifact / range_name(index), payload);
    }
    write_manifest(fixture, artifact_manifest(fixture));
    return fixture;
}

void test_range_artifact(const std::filesystem::path& root) {
    auto fixture = make_fixture(root);
    CHECK(fixture.plan.trunk_ranges.size() == 4U && fixture.plan.trunk_bytes == 14U);

    auto verified = vastloom::model::verify_range_artifact(fixture.artifact);
    CHECK(verified && verified.value().full_hash_verified &&
          verified.value().ranges.size() == 4U && verified.value().payload_bytes == 14U &&
          !verified.value().complete_checkpoint_verified &&
          !verified.value().execution_supported);
    if (!verified) return;
    auto binding = vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
        verified.value(), fixture.report, fixture.plan, fixture.spec);
    CHECK(binding && binding.value().exact_source_binding_verified &&
          binding.value().local_range_hashes_verified && binding.value().ranges.size() == 4U &&
          binding.value().maximum_local_range_bytes == 4U &&
          binding.value().ranges.front().local_source.offset == 0U &&
          binding.value().ranges.front().local_source.bytes == 4U &&
          !binding.value().complete_checkpoint_verified && !binding.value().execution_supported);
    if (binding) {
        CHECK(vastloom::inference::to_json(binding.value()).find(
                  "\"exact_source_binding_verified\":true") != std::string::npos);
        CHECK(vastloom::inference::to_text(binding.value()).find(
                  "Execution:            unsupported") != std::string::npos);

        auto set_logical_id =
            vastloom::inference::deepseek_v4_bound_trunk_tensor_set_logical_id(
                binding.value());
        CHECK(set_logical_id);
        if (set_logical_id) {
            vastloom::model::BoundRangeSetIdentity set_identity;
            set_identity.logical_id = set_logical_id.value();
            set_identity.total_bytes = binding.value().payload_bytes;
            for (const auto& range : binding.value().ranges) {
                set_identity.ranges.push_back(vastloom::model::BoundRangeIdentity{
                    set_identity.logical_id + "/" + std::to_string(range.artifact_index),
                    range.local_source,
                    range.local_sha256});
            }
            vastloom::safety::ResourceLimits set_limits;
            set_limits.host_memory_hard_limit = 18U;
            set_limits.ram_hard_limit = 14U;
            set_limits.vram_hard_limit = 1U;
            set_limits.staging_hard_limit = 4U;
            set_limits.scratch_hard_limit = 1U;
            vastloom::safety::ResourceGovernor set_governor(set_limits);
            auto set_residency = vastloom::model::ResidencyObject::create(
                set_identity.logical_id, set_identity.total_bytes);
            vastloom::model::BoundRangeSetLoadOptions set_options;
            set_options.max_total_bytes = 14U;
            set_options.max_range_bytes = 4U;
            set_options.max_ranges = 4U;
            set_options.measure_process_io = false;
            CHECK(set_residency);
            if (set_residency) {
                std::shared_ptr<const vastloom::model::ResidentByteBuffer> retained;
                {
                    auto loaded = vastloom::model::load_bound_range_set(
                        set_identity, set_residency.value(), set_governor, set_options);
                    CHECK(loaded && loaded.value().bytes().size() == 14U &&
                          loaded.value().slices().size() == 4U &&
                          loaded.value().telemetry().logical_bytes == 14U &&
                          loaded.value().telemetry().range_count == 4U &&
                          !loaded.value().range(4U));
                    if (loaded) {
                        for (std::size_t index = 0U;
                             index < loaded.value().slices().size(); ++index) {
                            auto bytes = loaded.value().range(index);
                            CHECK(bytes && bytes.value().size() == 4U - (index == 2U ? 2U : 0U));
                        }
                        retained = loaded.value().retain_buffer();
                    }
                }
                auto ready = set_residency.value().snapshot();
                CHECK(ready && ready.value().state ==
                                   vastloom::model::ResidencyState::ram_ready &&
                      ready.value().ram_reserved_bytes == 14U &&
                      ready.value().staging_reserved_bytes == 0U &&
                      ready.value().active_uses == 1U &&
                      !set_residency.value().evict_ram());
                retained.reset();
                CHECK(set_residency.value().evict_ram() &&
                      set_governor.snapshot().reserved[0] == 0U &&
                      set_governor.snapshot().reserved[2] == 0U);
            }

            auto duplicate_identity = set_identity;
            duplicate_identity.ranges[1U].logical_id =
                duplicate_identity.ranges[0U].logical_id;
            auto duplicate_residency = vastloom::model::ResidencyObject::create(
                duplicate_identity.logical_id, duplicate_identity.total_bytes);
            CHECK(duplicate_residency &&
                  !vastloom::model::load_bound_range_set(
                      duplicate_identity, duplicate_residency.value(), set_governor,
                      set_options));
            CHECK(duplicate_residency && duplicate_residency.value().snapshot() &&
                  duplicate_residency.value().snapshot().value().state ==
                      vastloom::model::ResidencyState::nvme);

            auto split_tensor_residency = vastloom::model::ResidencyObject::create(
                set_identity.logical_id, set_identity.total_bytes);
            CHECK(split_tensor_residency &&
                  !vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
                      binding.value(), fixture.report, split_tensor_residency.value(),
                      set_governor, set_options));
            CHECK(split_tensor_residency && split_tensor_residency.value().snapshot() &&
                  split_tensor_residency.value().snapshot().value().state ==
                      vastloom::model::ResidencyState::nvme);
        }

        const auto range_index = binding.value().ranges.size() - 1U;
        auto logical_id = vastloom::inference::deepseek_v4_bound_trunk_range_logical_id(
            binding.value(), range_index);
        CHECK(logical_id);
        vastloom::safety::ResourceLimits limits;
        limits.host_memory_hard_limit = 8U;
        limits.ram_hard_limit = 4U;
        limits.vram_hard_limit = 4U;
        limits.staging_hard_limit = 4U;
        limits.scratch_hard_limit = 4U;
        vastloom::safety::ResourceGovernor governor(limits);
        auto residency = logical_id
            ? vastloom::model::ResidencyObject::create(logical_id.value(), 4U)
            : vastloom::Result<vastloom::model::ResidencyObject>(
                  vastloom::make_error(vastloom::ErrorCode::internal_error, "missing id"));
        CHECK(residency);
        if (residency) {
            std::shared_ptr<const vastloom::model::ResidentByteBuffer> retained;
            {
                vastloom::model::BoundRangeLoadOptions load_options;
                load_options.max_range_bytes = 4U;
                load_options.measure_process_io = false;
                auto loaded = vastloom::inference::load_deepseek_v4_bound_trunk_range(
                    binding.value(), range_index, residency.value(), governor, load_options);
                CHECK(loaded && loaded.value().bytes().size() == 4U &&
                      loaded.value().telemetry().logical_bytes == 4U);
                if (loaded) {
                    std::array<float, 1U> input{2.0F};
                    std::array<float, 1U> scratch{};
                    std::array<float, 1U> output{-1.0F};
                    CHECK(vastloom::inference::matvec_ggml_f32_transactional(
                        vastloom::inference::kGgmlTypeF32,
                        loaded.value().bytes(), 1U, 1U, input, scratch, output));
                    CHECK(output[0] == 3.0F);
                    retained = loaded.value().retain_buffer();
                }
            }
            auto ready = residency.value().snapshot();
            CHECK(ready && ready.value().state == vastloom::model::ResidencyState::ram_ready &&
                  ready.value().active_uses == 1U &&
                  ready.value().ram_reserved_bytes == 4U &&
                  ready.value().staging_reserved_bytes == 0U);
            CHECK(!residency.value().evict_ram());
            retained.reset();
            CHECK(residency.value().evict_ram());
            CHECK(governor.snapshot().reserved[0] == 0U &&
                  governor.snapshot().reserved[2] == 0U);
        }

        vastloom::safety::ResourceLimits small_limits = limits;
        small_limits.host_memory_hard_limit = 7U;
        vastloom::safety::ResourceGovernor small_governor(small_limits);
        auto refused_residency = vastloom::model::ResidencyObject::create(
            logical_id.value(), 4U);
        vastloom::model::BoundRangeLoadOptions load_options;
        load_options.max_range_bytes = 4U;
        load_options.measure_process_io = false;
        CHECK(refused_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, refused_residency.value(), small_governor,
                  load_options));
        CHECK(small_governor.snapshot().reserved[0] == 0U &&
              small_governor.snapshot().reserved[2] == 0U);

        auto oversized_options = load_options;
        oversized_options.max_range_bytes =
            vastloom::model::kMaximumBoundRangeLoadBytes + 1U;
        auto oversized_residency = vastloom::model::ResidencyObject::create(
            logical_id.value(), 4U);
        CHECK(oversized_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, oversized_residency.value(), governor,
                  oversized_options));
        auto wrong_identity_residency = vastloom::model::ResidencyObject::create(
            "deepseek/wrong-range", 4U);
        CHECK(wrong_identity_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, wrong_identity_residency.value(), governor,
                  load_options));

        auto cancelled_residency = vastloom::model::ResidencyObject::create(
            logical_id.value(), 4U);
        auto cancelled_options = load_options;
        cancelled_options.cancelled = [] { return true; };
        CHECK(cancelled_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, cancelled_residency.value(), governor,
                  cancelled_options));
        CHECK(cancelled_residency && cancelled_residency.value().snapshot() &&
              cancelled_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::nvme);

        auto callback_failure_residency = vastloom::model::ResidencyObject::create(
            logical_id.value(), 4U);
        auto callback_failure_options = load_options;
        callback_failure_options.cancelled = []() -> bool {
            throw std::runtime_error("synthetic cancellation callback fault");
        };
        auto callback_failure = callback_failure_residency
            ? vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, callback_failure_residency.value(), governor,
                  callback_failure_options)
            : vastloom::Result<vastloom::model::LoadedBoundRange>(
                  vastloom::make_error(vastloom::ErrorCode::internal_error, "missing residency"));
        CHECK(!callback_failure &&
              callback_failure.error().code == vastloom::ErrorCode::internal_error &&
              callback_failure_residency && callback_failure_residency.value().snapshot() &&
              callback_failure_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::nvme);

        unsigned cancellation_checks = 0U;
        auto late_options = load_options;
        late_options.cancelled = [&] { return ++cancellation_checks >= 2U; };
        auto late_residency = vastloom::model::ResidencyObject::create(logical_id.value(), 4U);
        CHECK(late_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, late_residency.value(), governor, late_options));
        CHECK(late_residency && late_residency.value().snapshot() &&
              late_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::failed &&
              late_residency.value().snapshot().value().ram_reserved_bytes == 0U &&
              late_residency.value().snapshot().value().staging_reserved_bytes == 0U);

        const auto local_path = binding.value().ranges[range_index].local_source.shard;
        const auto original_payload = read_range(fixture.source, 36U, 4U);
        auto corrupted_payload = original_payload;
        corrupted_payload[0] ^= std::byte{1U};
        write_bytes(local_path, corrupted_payload);
        auto corrupt_residency = vastloom::model::ResidencyObject::create(logical_id.value(), 4U);
        CHECK(corrupt_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, corrupt_residency.value(), governor,
                  load_options));
        CHECK(corrupt_residency && corrupt_residency.value().snapshot() &&
              corrupt_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::failed &&
              governor.snapshot().reserved[0] == 0U && governor.snapshot().reserved[2] == 0U);
        write_bytes(local_path, original_payload);

        write_bytes(local_path, std::span<const std::byte>(original_payload).first(3U));
        auto short_residency = vastloom::model::ResidencyObject::create(logical_id.value(), 4U);
        CHECK(short_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_range(
                  binding.value(), range_index, short_residency.value(), governor, load_options));
        CHECK(short_residency && short_residency.value().snapshot() &&
              short_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::nvme);
        write_bytes(local_path, original_payload);
    }
    CHECK(vastloom::model::to_json(verified.value()).find(
              "\"local_range_hashes_verified\":true") != std::string::npos);

    vastloom::model::RangeArtifactVerificationOptions quick_options;
    quick_options.verify_hashes = false;
    auto quick = vastloom::model::verify_range_artifact(fixture.artifact, quick_options);
    CHECK(quick && !quick.value().full_hash_verified);
    if (quick) {
        CHECK(!vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
            quick.value(), fixture.report, fixture.plan, fixture.spec));
    }

    auto invalid_bounds = quick_options;
    invalid_bounds.max_ranges = 0U;
    CHECK(!vastloom::model::verify_range_artifact(fixture.artifact, invalid_bounds));
    auto tiny_bound = quick_options;
    tiny_bound.max_payload_bytes = 13U;
    CHECK(!vastloom::model::verify_range_artifact(fixture.artifact, tiny_bound));

    const auto first_path = fixture.artifact / range_name(0U);
    auto first_payload = read_range(first_path, 0U, 4U);
    first_payload[0] ^= std::byte{1U};
    write_bytes(first_path, first_payload);
    CHECK(!vastloom::model::verify_range_artifact(fixture.artifact));
    CHECK(vastloom::model::verify_range_artifact(fixture.artifact, quick_options));
    const auto original_first = read_range(fixture.source,
                                           fixture.plan.trunk_ranges[0].source.offset,
                                           fixture.plan.trunk_ranges[0].source.bytes);
    write_bytes(first_path, original_first);
    write_manifest(fixture, artifact_manifest(fixture));

    std::ofstream(fixture.artifact / "ranges" / "extra.bin", std::ios::binary) << "x";
    CHECK(!vastloom::model::verify_range_artifact(fixture.artifact));
    std::filesystem::remove(fixture.artifact / "ranges" / "extra.bin");

    write_manifest(fixture, artifact_manifest(fixture, 44U));
    auto shifted = vastloom::model::verify_range_artifact(fixture.artifact);
    CHECK(shifted);
    if (shifted) {
        CHECK(!vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
            shifted.value(), fixture.report, fixture.plan, fixture.spec));
    }

    write_manifest(fixture, artifact_manifest(fixture, 0U, std::string(64U, '0')));
    auto wrong_source_hash = vastloom::model::verify_range_artifact(fixture.artifact);
    CHECK(wrong_source_hash);
    if (wrong_source_hash) {
        CHECK(!vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
            wrong_source_hash.value(), fixture.report, fixture.plan, fixture.spec));
    }

    write_manifest(fixture, artifact_manifest(fixture, 0U, std::nullopt, true));
    CHECK(!vastloom::model::verify_range_artifact(fixture.artifact));
    write_manifest(fixture, artifact_manifest(fixture, 0U, std::nullopt, false,
                                               "../outside.bin"));
    CHECK(!vastloom::model::verify_range_artifact(fixture.artifact));
    write_manifest(fixture, artifact_manifest(fixture));

    auto tampered_plan = fixture.plan;
    tampered_plan.trunk_ranges.front().tensor_name = "blk.0.forged.weight";
    auto restored = vastloom::model::verify_range_artifact(fixture.artifact);
    CHECK(restored);
    if (restored) {
        CHECK(!vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
            restored.value(), fixture.report, tampered_plan, fixture.spec));
        auto wrong_spec = fixture.spec;
        wrong_spec.shards.front().full_sha256 = std::string(64U, '0');
        CHECK(!vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
            restored.value(), fixture.report, fixture.plan, wrong_spec));
        auto unverified_report = fixture.report;
        unverified_report.imported_file_hashes_verified = false;
        CHECK(!vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
            restored.value(), unverified_report, fixture.plan, fixture.spec));
    }

    auto complete_fixture = make_fixture(root / "complete", 64U);
    CHECK(complete_fixture.plan.trunk_ranges.size() == 2U &&
          complete_fixture.plan.trunk_bytes == 14U);
    auto complete_artifact =
        vastloom::model::verify_range_artifact(complete_fixture.artifact);
    auto complete_binding = complete_artifact
        ? vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
              complete_artifact.value(), complete_fixture.report,
              complete_fixture.plan, complete_fixture.spec)
        : vastloom::Result<vastloom::inference::DeepseekV4TrunkArtifactBinding>(
              vastloom::make_error(vastloom::ErrorCode::internal_error,
                                   "missing complete artifact"));
    CHECK(complete_artifact && complete_binding);
    if (complete_binding) {
        auto complete_id =
            vastloom::inference::deepseek_v4_bound_trunk_tensor_set_logical_id(
                complete_binding.value());
        CHECK(complete_id);
        vastloom::safety::ResourceLimits limits;
        limits.host_memory_hard_limit = 24U;
        limits.ram_hard_limit = 14U;
        limits.vram_hard_limit = 1U;
        limits.staging_hard_limit = 10U;
        limits.scratch_hard_limit = 1U;
        vastloom::safety::ResourceGovernor governor(limits);
        vastloom::model::BoundRangeSetLoadOptions options;
        options.max_total_bytes = 14U;
        options.max_range_bytes = 10U;
        options.max_ranges = 2U;
        options.measure_process_io = false;
        auto residency = vastloom::model::ResidencyObject::create(
            complete_id.value(), complete_binding.value().payload_bytes);
        std::shared_ptr<const vastloom::model::ResidentByteBuffer> retained;
        CHECK(residency);
        if (residency) {
            {
                auto loaded =
                    vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
                        complete_binding.value(), complete_fixture.report,
                        residency.value(), governor, options);
                CHECK(loaded && loaded.value().payload_bytes() == 14U &&
                      loaded.value().tensors().size() == 2U &&
                      loaded.value().find("blk.0.attn.weight") != nullptr &&
                      loaded.value().tensor("blk.0.ffn_norm.weight") &&
                      !loaded.value().tensor("missing.weight") &&
                      loaded.value().telemetry().range_count == 2U);
                if (loaded) retained = loaded.value().retain_buffer();
            }
            CHECK(residency.value().snapshot() &&
                  residency.value().snapshot().value().active_uses == 1U &&
                  !residency.value().evict_ram());
            retained.reset();
            CHECK(residency.value().evict_ram());
        }

        auto check_preflight_rejection = [&](auto mutate) {
            auto rejected = complete_binding.value();
            mutate(rejected);
            auto rejected_residency = vastloom::model::ResidencyObject::create(
                complete_id.value(), complete_binding.value().payload_bytes);
            CHECK(rejected_residency &&
                  !vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
                      rejected, complete_fixture.report, rejected_residency.value(),
                      governor, options));
            CHECK(rejected_residency && rejected_residency.value().snapshot() &&
                  rejected_residency.value().snapshot().value().state ==
                      vastloom::model::ResidencyState::nvme &&
                  governor.snapshot().reserved[0] == 0U &&
                  governor.snapshot().reserved[2] == 0U);
        };
        check_preflight_rejection([](auto& rejected) {
            rejected.ranges.pop_back();
        });
        check_preflight_rejection([](auto& rejected) {
            rejected.ranges[1U].tensor_name = rejected.ranges[0U].tensor_name;
        });
        check_preflight_rejection([](auto& rejected) {
            rejected.ranges[0U].ggml_type = 8U;
        });
        check_preflight_rejection([](auto& rejected) {
            rejected.ranges[0U].role = vastloom::model::TensorRole::normalization;
        });
        check_preflight_rejection([](auto& rejected) {
            ++rejected.ranges[0U].original_source.offset;
        });
        check_preflight_rejection([](auto& rejected) {
            rejected.ranges[0U].final_tile = false;
        });

        vastloom::safety::ResourceLimits insufficient_limits = limits;
        insufficient_limits.host_memory_hard_limit = 23U;
        vastloom::safety::ResourceGovernor insufficient_governor(insufficient_limits);
        auto insufficient_residency = vastloom::model::ResidencyObject::create(
            complete_id.value(), complete_binding.value().payload_bytes);
        CHECK(insufficient_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
                  complete_binding.value(), complete_fixture.report,
                  insufficient_residency.value(), insufficient_governor, options));
        CHECK(insufficient_residency && insufficient_residency.value().snapshot() &&
              insufficient_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::nvme &&
              insufficient_governor.snapshot().reserved[0] == 0U &&
              insufficient_governor.snapshot().reserved[2] == 0U);

        auto corrupted = complete_binding.value();
        const auto corrupt_path = corrupted.ranges.back().local_source.shard;
        auto original = read_range(corrupt_path, 0U, corrupted.ranges.back().local_source.bytes);
        auto changed = original;
        changed.back() ^= std::byte{1U};
        write_bytes(corrupt_path, changed);
        auto corrupt_residency = vastloom::model::ResidencyObject::create(
            complete_id.value(), corrupted.payload_bytes);
        CHECK(corrupt_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
                  corrupted, complete_fixture.report, corrupt_residency.value(),
                  governor, options));
        CHECK(corrupt_residency && corrupt_residency.value().snapshot() &&
              corrupt_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::failed &&
              governor.snapshot().reserved[0] == 0U &&
              governor.snapshot().reserved[2] == 0U);
        write_bytes(corrupt_path, original);

        unsigned cancellation_checks = 0U;
        auto cancel_options = options;
        cancel_options.cancelled = [&] { return ++cancellation_checks >= 3U; };
        auto cancel_residency = vastloom::model::ResidencyObject::create(
            complete_id.value(), complete_binding.value().payload_bytes);
        CHECK(cancel_residency &&
              !vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
                  complete_binding.value(), complete_fixture.report,
                  cancel_residency.value(), governor, cancel_options));
        CHECK(cancel_residency && cancel_residency.value().snapshot() &&
              cancel_residency.value().snapshot().value().state ==
                  vastloom::model::ResidencyState::failed &&
              governor.snapshot().reserved[0] == 0U &&
              governor.snapshot().reserved[2] == 0U);
    }
}

}  // namespace

int main() {
    const auto root = make_temp_dir();
    test_range_artifact(root);
    std::error_code error;
    std::filesystem::remove_all(root, error);
    if (failures != 0) {
        std::cerr << failures << " range artifact test(s) failed\n";
        return 1;
    }
    std::cout << "Range artifact tests passed\n";
    return 0;
}
