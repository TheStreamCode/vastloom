#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <span>
#include <string>
#include <vector>

#include "vastloom/inference/deepseek_v4_expert_pack.hpp"
#include "vastloom/inference/deepseek_v4_gguf.hpp"
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

struct PackTensor {
    std::string name;
    std::uint64_t packed_offset{0U};
    std::uint64_t bytes{0U};
};

struct PackExpert {
    std::uint32_t expert{0U};
    std::uint64_t offset{0U};
    std::uint64_t bytes{0U};
    std::vector<PackTensor> tensors;
    std::string sha256;
};

std::filesystem::path make_temp_dir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base = std::filesystem::current_path() / "vastloom-expert-pack-test-tmp";
    std::filesystem::create_directories(base);
    const auto root = base / std::to_string(stamp);
    std::filesystem::create_directories(root);
    return root;
}

std::vector<std::byte> make_payload(std::size_t bytes) {
    std::vector<std::byte> payload(bytes);
    for (std::size_t index = 0U; index < bytes; ++index) {
        payload[index] = static_cast<std::byte>((index * 13U + 7U) & 0xFFU);
    }
    return payload;
}

std::string sha256_span(std::span<const std::byte> bytes) {
    vastloom::util::Sha256 hasher;
    hasher.update(bytes);
    return hasher.finalize_hex();
}

std::vector<PackExpert> make_experts(std::span<const std::byte> payload) {
    PackExpert expert;
    expert.expert = 0U;
    expert.offset = 0U;
    expert.bytes = payload.size();
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        PackTensor tensor;
        tensor.name = "expert.0.t" + std::to_string(index);
        tensor.packed_offset = index * 4U;
        tensor.bytes = 4U;
        expert.tensors.push_back(tensor);
    }
    expert.sha256 = sha256_span(payload);
    return {expert};
}

void write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

std::string tensor_json(const PackTensor& tensor) {
    std::ostringstream output;
    output << "{\"name\":\"" << tensor.name << "\",\"packed_offset\":" << tensor.packed_offset
           << ",\"bytes\":" << tensor.bytes << ",\"shape\":[1,4]}";
    return output.str();
}

std::string manifest_json(const std::string& model_id, const std::string& revision,
                          std::uint64_t layer, const std::string& payload_sha256,
                          std::uint64_t payload_bytes, bool routing_authority,
                          std::uint64_t alignment, const std::vector<PackExpert>& experts,
                          const std::string& format = "vastloom.deepseek-v4-expert-pack.v1") {
    std::ostringstream output;
    output << "{\"format\":\"" << format << "\",\"model_id\":\"" << model_id
           << "\",\"revision\":\"" << revision << "\",\"layer\":" << layer
           << ",\"routing_authority_verified\":"
           << (routing_authority ? "true" : "false") << ",\"alignment\":" << alignment
           << ",\"payload\":{\"path\":\"experts-L0.bin\",\"bytes\":" << payload_bytes
           << ",\"sha256\":\"" << payload_sha256 << "\"},\"source_artifacts\":[],\"experts\":[";
    for (std::size_t expert_index = 0U; expert_index < experts.size(); ++expert_index) {
        if (expert_index != 0U) output << ',';
        const auto& expert = experts[expert_index];
        output << "{\"expert\":" << expert.expert << ",\"offset\":" << expert.offset
               << ",\"bytes\":" << expert.bytes << ",\"sha256\":\"" << expert.sha256
               << "\",\"tensors\":[";
        for (std::size_t tensor_index = 0U; tensor_index < expert.tensors.size(); ++tensor_index) {
            if (tensor_index != 0U) output << ',';
            output << tensor_json(expert.tensors[tensor_index]);
        }
        output << "]}";
    }
    output << "]}";
    return output.str();
}

bool verify_fixture(const std::string& manifest, std::span<const std::byte> payload) {
    const auto root = make_temp_dir();
    std::ofstream manifest_file(root / "vastloom-expert-pack.json",
                                std::ios::binary | std::ios::trunc);
    manifest_file << manifest;
    manifest_file.close();
    write_file(root / "experts-L0.bin", payload);
    auto report = vastloom::inference::DeepseekV4GgufCatalogReport{};
    report.catalog.model_id = "test/model";
    report.catalog.revision = "rev1";
    report.catalog.metadata_only = true;
    report.layers = 1U;
    report.experts_per_layer = 1U;
    report.top_k = 1U;
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        vastloom::model::LogicalTensorDescriptor tensor;
        tensor.name = "expert.0.t" + std::to_string(index);
        tensor.dtype = vastloom::model::TensorDType::f32;
        tensor.shape = {1U, 4U};
        tensor.role = vastloom::model::TensorRole::other;
        tensor.layer = 0U;
        tensor.expert = 0U;
        report.catalog.tensors.push_back(tensor);
    }
    vastloom::model::ExpertBundle bundle;
    bundle.layer = 0U;
    bundle.expert = 0U;
    bundle.bytes = 12U;
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        bundle.tensor_indices.push_back(index);
        bundle.tensor_slices.push_back(vastloom::model::ExpertTensorSlice{
            index, index * 4U, 4U});
    }
    bundle.read_ranges.push_back(vastloom::model::TensorSourceRange{
        "original/shard.gguf", 100U, 4U});
    report.catalog.experts.push_back(std::move(bundle));
    const auto verified = vastloom::inference::verify_deepseek_v4_expert_pack(root, report, 0U);
    std::filesystem::remove_all(root);
    return static_cast<bool>(verified);
}

std::string valid_manifest(std::span<const std::byte> payload) {
    const auto experts = make_experts(payload);
    return manifest_json("test/model", "rev1", 0U, sha256_span(payload), payload.size(), true,
                         4096U, experts);
}

void test_valid_pack_applies_one_range() {
    const auto payload = make_payload(12U);
    const auto root = make_temp_dir();
    std::ofstream manifest_file(root / "vastloom-expert-pack.json",
                                std::ios::binary | std::ios::trunc);
    manifest_file << valid_manifest(payload);
    manifest_file.close();
    write_file(root / "experts-L0.bin", payload);

    auto report = vastloom::inference::DeepseekV4GgufCatalogReport{};
    report.catalog.model_id = "test/model";
    report.catalog.revision = "rev1";
    report.layers = 1U;
    report.experts_per_layer = 1U;
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        vastloom::model::LogicalTensorDescriptor tensor;
        tensor.name = "expert.0.t" + std::to_string(index);
        tensor.dtype = vastloom::model::TensorDType::f32;
        tensor.shape = {1U, 4U};
        report.catalog.tensors.push_back(tensor);
    }
    vastloom::model::ExpertBundle bundle;
    bundle.layer = 0U;
    bundle.expert = 0U;
    bundle.bytes = 12U;
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        bundle.tensor_indices.push_back(index);
        bundle.tensor_slices.push_back(vastloom::model::ExpertTensorSlice{
            index, index * 4U, 4U});
    }
    report.catalog.experts.push_back(std::move(bundle));

    auto verified = vastloom::inference::verify_deepseek_v4_expert_pack(root, report, 0U);
    CHECK(static_cast<bool>(verified));
    if (verified) {
        CHECK(verified.value().records.size() == 1U);
        CHECK(verified.value().records[0].expert == 0U);
        CHECK(verified.value().payload_bytes == 12U);
        CHECK(verified.value().full_hash_verified);
        CHECK(verified.value().routing_authority_verified);
        CHECK(vastloom::inference::to_json(verified.value()).find("record_count") !=
              std::string::npos);
        auto applied = vastloom::inference::apply_deepseek_v4_expert_pack(
            report.catalog, verified.value());
        CHECK(static_cast<bool>(applied));
        CHECK(applied && applied.value() == 1U);
        if (applied) {
            CHECK(report.catalog.experts.size() == 1U);
            CHECK(report.catalog.experts[0].read_ranges.size() == 1U);
            CHECK(report.catalog.experts[0].read_ranges[0].bytes == 12U);
            CHECK(report.catalog.experts[0].read_ranges[0].offset == 0U);
            CHECK(report.catalog.experts[0].tensor_slices.size() == 3U);
            CHECK(report.catalog.experts[0].tensor_slices[1].packed_offset == 4U);
        }
    }
    std::filesystem::remove_all(root);
}

void test_malformed_packs_are_refused() {
    const auto payload = make_payload(12U);
    const auto experts = make_experts(payload);
    const auto payload_sha = sha256_span(payload);

    CHECK(!verify_fixture(
        manifest_json("other/model", "rev1", 0U, payload_sha, payload.size(), true, 4096U,
                      experts),
        payload));
    CHECK(!verify_fixture(
        manifest_json("test/model", "rev1", 0U, std::string(64U, '0'), payload.size(), true,
                      4096U, experts),
        payload));
    CHECK(!verify_fixture(
        manifest_json("test/model", "rev1", 0U, payload_sha, payload.size(), false, 4096U,
                      experts),
        payload));
    CHECK(!verify_fixture(
        manifest_json("test/model", "rev1", 0U, payload_sha, payload.size(), true, 4096U,
                      experts, "vastloom.deepseek-v4-expert-pack.v0"),
        payload));
    CHECK(!verify_fixture(
        manifest_json("test/model", "rev1", 1U, payload_sha, payload.size(), true, 4096U,
                      experts),
        payload));

    auto misaligned = experts;
    misaligned[0].offset = 1U;
    CHECK(!verify_fixture(
        manifest_json("test/model", "rev1", 0U, payload_sha, payload.size(), true, 4096U,
                      misaligned),
        payload));

    auto wrong_tensor_bytes = experts;
    wrong_tensor_bytes[0].tensors[0].bytes = 5U;
    CHECK(!verify_fixture(
        manifest_json("test/model", "rev1", 0U, payload_sha, payload.size(), true, 4096U,
                      wrong_tensor_bytes),
        payload));

    auto duplicate = experts;
    duplicate.push_back(experts[0]);
    CHECK(!verify_fixture(
        manifest_json("test/model", "rev1", 0U, payload_sha, payload.size(), true, 4096U,
                      duplicate),
        payload));

    CHECK(!verify_fixture(valid_manifest(payload), make_payload(8U)));
}

}  // namespace

int main() {
    try {
        test_valid_pack_applies_one_range();
        test_malformed_packs_are_refused();
    } catch (const std::exception& exception) {
        std::cerr << "uncaught exception: " << exception.what() << '\n';
        return 2;
    } catch (...) {
        std::cerr << "uncaught non-standard exception\n";
        return 2;
    }
    std::error_code cleanup_error;
    std::filesystem::remove_all(
        std::filesystem::current_path() / "vastloom-expert-pack-test-tmp", cleanup_error);
    if (failures != 0) {
        std::cerr << failures << " expert-pack test(s) failed\n";
        return 1;
    }
    std::cout << "Expert-pack tests PASS\n";
    return 0;
}
