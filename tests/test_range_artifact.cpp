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
#include <string>
#include <vector>

#include "vastloom/inference/deepseek_v4_gguf.hpp"
#include "vastloom/inference/deepseek_v4_range_artifact.hpp"
#include "vastloom/model/range_artifact.hpp"
#include "vastloom/model/tensor_catalog.hpp"
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

Fixture make_fixture(const std::filesystem::path& root) {
    Fixture fixture;
    fixture.root = root;
    fixture.source = root / "model-00001.gguf";
    std::array<std::byte, 64U> source_bytes{};
    for (std::size_t index = 0U; index < source_bytes.size(); ++index) {
        source_bytes[index] = static_cast<std::byte>(index);
    }
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
        fixture.report, 0U, selected, 4U, 16U);
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
    CHECK(fixture.plan.trunk_ranges.size() == 3U && fixture.plan.trunk_bytes == 10U);

    auto verified = vastloom::model::verify_range_artifact(fixture.artifact);
    CHECK(verified && verified.value().full_hash_verified &&
          verified.value().ranges.size() == 3U && verified.value().payload_bytes == 10U &&
          !verified.value().complete_checkpoint_verified &&
          !verified.value().execution_supported);
    if (!verified) return;
    auto binding = vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
        verified.value(), fixture.report, fixture.plan, fixture.spec);
    CHECK(binding && binding.value().exact_source_binding_verified &&
          binding.value().local_range_hashes_verified && binding.value().ranges.size() == 3U &&
          binding.value().maximum_local_range_bytes == 4U &&
          binding.value().ranges.front().local_source.offset == 0U &&
          binding.value().ranges.front().local_source.bytes == 4U &&
          !binding.value().complete_checkpoint_verified && !binding.value().execution_supported);
    if (binding) {
        CHECK(vastloom::inference::to_json(binding.value()).find(
                  "\"exact_source_binding_verified\":true") != std::string::npos);
        CHECK(vastloom::inference::to_text(binding.value()).find(
                  "Execution:            unsupported") != std::string::npos);
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
    tiny_bound.max_payload_bytes = 9U;
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

    write_manifest(fixture, artifact_manifest(fixture, 32U));
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
