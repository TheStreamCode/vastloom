#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/model/expert_loader.hpp"
#include "vastloom/model/residency.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"
#include "vastloom/util/sha256.hpp"

namespace {

constexpr std::size_t kHiddenSize = 4'096U;
constexpr std::size_t kExpertWidth = 2'048U;
constexpr std::size_t kIq1SBytes =
    kExpertWidth * (kHiddenSize / vastloom::inference::kIq1SBlockElements) *
    vastloom::inference::kIq1SBlockBytes;
constexpr std::size_t kIq3XxsBytes =
    kHiddenSize * (kExpertWidth / vastloom::inference::kIq3XxsBlockElements) *
    vastloom::inference::kIq3XxsBlockBytes;
constexpr vastloom::ByteCount kBundleBytes = 2U * kIq1SBytes + kIq3XxsBytes;
constexpr std::size_t kGateTensor = 0U;
constexpr std::size_t kUpTensor = 1U;
constexpr std::size_t kDownTensor = 2U;
constexpr std::uint64_t kExpectedOutputFnv64 = UINT64_C(0x90388543cbcbab52);

struct SourceTensor {
    std::filesystem::path path;
    std::size_t tensor_index{0U};
    vastloom::ByteCount bytes{0U};
};

int fail(std::string_view message) {
    std::cerr << "DeepSeek real-expert test failed: " << message << '\n';
    return 1;
}

bool verify_slice(const std::filesystem::path& path,
                  vastloom::ByteCount expected_bytes,
                  std::string_view expected_sha256) {
    std::error_code error;
    if (std::filesystem::is_symlink(path, error) || error ||
        !std::filesystem::is_regular_file(path, error) || error ||
        std::filesystem::file_size(path, error) != expected_bytes || error) {
        return false;
    }
    auto digest = vastloom::util::sha256_file(path);
    return digest && digest.value() == expected_sha256;
}

std::uint64_t output_fingerprint(std::span<const float> values) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : values) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0U; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

vastloom::model::ExpertBundle make_bundle(std::vector<SourceTensor> sources) {
    std::sort(sources.begin(), sources.end(), [](const auto& left, const auto& right) {
        return left.path < right.path;
    });

    vastloom::model::ExpertBundle bundle;
    bundle.layer = 0U;
    bundle.expert = 0U;
    bundle.bytes = kBundleBytes;
    vastloom::ByteCount packed_offset = 0U;
    for (const auto& source : sources) {
        bundle.tensor_indices.push_back(source.tensor_index);
        bundle.read_ranges.push_back({source.path, 0U, source.bytes});
        bundle.tensor_slices.push_back({source.tensor_index, packed_offset, source.bytes});
        packed_offset += source.bytes;
    }
    return bundle;
}

vastloom::safety::ResourceLimits make_limits(vastloom::ByteCount ram_limit) {
    vastloom::safety::ResourceLimits limits;
    limits.ram_hard_limit = ram_limit;
    limits.vram_hard_limit = 1U * vastloom::MiB;
    limits.staging_hard_limit = 4U * vastloom::MiB;
    limits.scratch_hard_limit = 128U * vastloom::KiB;
    return limits;
}

vastloom::model::ExpertLoadPoolOptions make_pool_options() {
    vastloom::model::ExpertLoadPoolOptions options;
    options.workers = 1U;
    options.max_queued_requests = 1U;
    options.load.max_expert_bytes = 8U * vastloom::MiB;
    options.load.max_range_bytes = 4U * vastloom::MiB;
    options.load.max_ranges = 3U;
    options.load.measure_process_io = true;
    options.load.measure_device_io = false;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) return fail("expected gate, up and down slice paths");

    const std::filesystem::path gate_path(argv[1]);
    const std::filesystem::path up_path(argv[2]);
    const std::filesystem::path down_path(argv[3]);
    if (!verify_slice(gate_path, kIq1SBytes,
                      "4710ef29f496c13acabd1371d1d2a17be3f321f85a0b12287145e7c0b409c4f9") ||
        !verify_slice(up_path, kIq1SBytes,
                      "7be122f1839feb8fdd84f57b9d20d5b57566227f8ba7d8ffc50123b158bf762f") ||
        !verify_slice(down_path, kIq3XxsBytes,
                      "65b9cdc96f9e753584dc6eac4723616a494a98113bf25836fcda5ec72c400204")) {
        return fail("a slice is not the pinned regular file with the expected size and SHA-256");
    }

    const auto bundle = make_bundle({
        {gate_path, kGateTensor, kIq1SBytes},
        {up_path, kUpTensor, kIq1SBytes},
        {down_path, kDownTensor, kIq3XxsBytes},
    });

    // A one-byte RAM shortfall must reject the request synchronously, before the
    // worker can allocate or read any model payload.
    {
        vastloom::safety::ResourceGovernor limited_governor(make_limits(kBundleBytes - 1U));
        vastloom::model::BoundedExpertLoadPool limited_pool(make_pool_options(), limited_governor);
        auto limited_residency = vastloom::model::ResidencyObject::create(
            "deepseek-v4/layer/0/expert/0/limited", kBundleBytes);
        if (!limited_residency) return fail(limited_residency.error().message);
        auto rejected = limited_pool.submit(
            bundle, limited_residency.value(), "deepseek-v4-expert-0-limited");
        if (rejected) return fail("insufficient RAM budget unexpectedly accepted the load");
        auto snapshot = limited_residency.value().snapshot();
        if (!snapshot || snapshot.value().state != vastloom::model::ResidencyState::nvme ||
            snapshot.value().ram_reserved_bytes != 0U ||
            snapshot.value().staging_reserved_bytes != 0U ||
            limited_governor.snapshot().reserved[0] != 0U ||
            limited_governor.snapshot().reserved[2] != 0U) {
            return fail("rejected load changed residency state or leaked a reservation");
        }
    }

    vastloom::safety::ResourceGovernor governor(make_limits(8U * vastloom::MiB));
    vastloom::model::BoundedExpertLoadPool pool(make_pool_options(), governor);
    auto residency = vastloom::model::ResidencyObject::create(
        "deepseek-v4/layer/0/expert/0", kBundleBytes);
    if (!residency) return fail(residency.error().message);

    auto submission = pool.submit(bundle, residency.value(), "deepseek-v4-expert-0");
    if (!submission) return fail(submission.error().message);
    auto loaded_result = submission.value().get();
    if (!loaded_result) return fail(loaded_result.error().message);

    double load_ms = 0.0;
    double compute_ms = 0.0;
    std::uint64_t fingerprint = 0U;
    std::uint64_t process_read_operations = 0U;
    vastloom::ByteCount process_read_bytes = 0U;
    bool storage_read_bytes_available = false;
    vastloom::ByteCount storage_read_bytes = 0U;
    {
        auto loaded = loaded_result.take_value();
        const auto& telemetry = loaded.telemetry();
        if (telemetry.logical_bytes != kBundleBytes || telemetry.range_count != 3U ||
            telemetry.request_sequence != 1U || telemetry.service_ms < 0.0 ||
            telemetry.total_ms < telemetry.service_ms) {
            return fail("bounded loader telemetry is inconsistent");
        }
        load_ms = telemetry.total_ms;
        if (telemetry.process_io.has_value()) {
            process_read_operations = telemetry.process_io->read_operations;
            process_read_bytes = telemetry.process_io->process_read_bytes;
            storage_read_bytes_available = telemetry.process_io->storage_read_bytes_available;
            storage_read_bytes = telemetry.process_io->storage_read_bytes;
        }

        auto gate = loaded.tensor(kGateTensor);
        auto up = loaded.tensor(kUpTensor);
        auto down = loaded.tensor(kDownTensor);
        if (!gate || !up || !down || gate.value().size() != kIq1SBytes ||
            up.value().size() != kIq1SBytes || down.value().size() != kIq3XxsBytes) {
            return fail("packed tensor addressing did not recover the three exact matrices");
        }

        auto ready = residency.value().snapshot();
        if (!ready || ready.value().state != vastloom::model::ResidencyState::ram_ready ||
            ready.value().ram_reserved_bytes != kBundleBytes ||
            ready.value().staging_reserved_bytes != 0U ||
            ready.value().active_uses != 1U || ready.value().evictable ||
            residency.value().evict_ram()) {
            return fail("loaded expert is not protected by its RAM residency lease");
        }

        constexpr vastloom::ByteCount kComputeBytes =
            (3U * kHiddenSize + 2U * kExpertWidth) * sizeof(float);
        auto compute_reservation = governor.try_reserve(
            vastloom::safety::ResourceKind::scratch, kComputeBytes,
            "deepseek-v4 expert input, output and scratch");
        if (!compute_reservation) return fail(compute_reservation.error().message);

        std::vector<float> input(kHiddenSize);
        std::vector<float> gate_scratch(kExpertWidth);
        std::vector<float> up_scratch(kExpertWidth);
        std::vector<float> down_scratch(kHiddenSize);
        std::vector<float> output(kHiddenSize, std::numeric_limits<float>::quiet_NaN());
        for (std::size_t column = 0U; column < input.size(); ++column) {
            const int centered = static_cast<int>(column % 29U) - 14;
            input[column] = static_cast<float>(centered) * 0.03125F +
                            static_cast<float>(column % 3U) * 0.0078125F;
        }

        const auto compute_started = std::chrono::steady_clock::now();
        auto executed = vastloom::inference::routed_expert_iq1_s_iq3_xxs_f32(
            gate.value(), up.value(), down.value(), input,
            {gate_scratch, up_scratch, down_scratch}, output);
        compute_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - compute_started)
                         .count();
        if (!executed) return fail(executed.error().message);
        fingerprint = output_fingerprint(output);
        if (fingerprint != kExpectedOutputFnv64 || output.front() != -0.0709934607F ||
            output.back() != 0.105151489F) {
            return fail("full expert output differs from the pinned ggml oracle");
        }
    }

    auto idle = residency.value().snapshot();
    if (!idle || idle.value().active_uses != 0U || !idle.value().evictable ||
        !residency.value().evict_ram()) {
        return fail("expert RAM could not be evicted after releasing its use lease");
    }
    const auto final_governor = governor.snapshot();
    if (final_governor.reserved[0] != 0U || final_governor.reserved[2] != 0U ||
        final_governor.reserved[3] != 0U) {
        return fail("expert execution leaked a Governor reservation");
    }

    std::cout << "DeepSeek V4 Flash 0731 full real expert: PASS"
              << " bytes=" << kBundleBytes
              << " ranges=3 load_ms=" << load_ms
              << " compute_ms=" << compute_ms
              << " output_fnv64=" << std::hex << fingerprint << std::dec
              << " process_read_operations=" << process_read_operations
              << " process_read_bytes=" << process_read_bytes
              << " storage_read_bytes=";
    if (storage_read_bytes_available) std::cout << storage_read_bytes;
    else std::cout << "unavailable";
    std::cout << '\n';
    return 0;
}
