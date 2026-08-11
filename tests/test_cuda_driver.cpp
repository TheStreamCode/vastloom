#include <cstddef>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/cuda/cuda_driver.hpp"
#include "vastloom/cuda/cuda_rms_norm.hpp"
#include "vastloom/inference/deepseek_v4_reference.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__     \
                      << ": " #condition << '\n';                             \
            ++failures;                                                        \
        }                                                                       \
    } while (false)

class FakeCudaBackend final : public vastloom::cuda::CudaDriverBackend {
public:
    vastloom::safety::ResourceGovernor* governor{nullptr};
    bool fail_probe{false};
    bool fail_allocate{false};
    bool fail_release{false};
    bool malformed_handle{false};
    bool throw_allocate{false};
    bool fail_copy_from{false};
    bool fail_copy_to{false};
    bool fail_launch{false};
    bool throw_copy_from{false};
    bool throw_copy_to{false};
    bool throw_launch{false};
    bool reservation_visible{false};
    int probe_calls{0};
    int allocate_calls{0};
    int release_calls{0};
    int copy_from_calls{0};
    int copy_to_calls{0};
    int launch_calls{0};
    vastloom::ByteCount last_minimum_free{0U};
    std::vector<std::byte> memory;

    [[nodiscard]] vastloom::Result<vastloom::cuda::CudaDriverProbe> probe() override {
        ++probe_calls;
        if (fail_probe) {
            return vastloom::make_error(vastloom::ErrorCode::io_error,
                                        "injected CUDA probe failure");
        }
        vastloom::cuda::CudaDriverProbe result;
        result.library = "fake-cuda";
        result.driver_version = 12'080;
        result.devices.push_back({0, "Fake RTX", 12U * vastloom::GiB, 8, 9, 56, 1'024, true});
        return result;
    }

    [[nodiscard]] vastloom::Result<vastloom::cuda::CudaBackendAllocation> allocate(
        int device_ordinal,
        vastloom::ByteCount bytes,
        vastloom::ByteCount minimum_free_vram_after) override {
        ++allocate_calls;
        last_minimum_free = minimum_free_vram_after;
        if (governor != nullptr) {
            const auto snapshot = governor->snapshot();
            reservation_visible = snapshot.reserved[static_cast<std::size_t>(
                                      vastloom::safety::ResourceKind::vram)] == bytes;
        }
        if (throw_allocate) throw std::runtime_error("injected CUDA allocation exception");
        if (fail_allocate) {
            return vastloom::make_error(vastloom::ErrorCode::resource_exhausted,
                                        "injected CUDA allocation failure");
        }
        if (malformed_handle) {
            memory.resize(static_cast<std::size_t>(bytes));
            return vastloom::cuda::CudaBackendAllocation{
                1U, UINT64_C(0x2000), bytes + 1U, device_ordinal};
        }
        memory.resize(static_cast<std::size_t>(bytes));
        return vastloom::cuda::CudaBackendAllocation{
            1U, UINT64_C(0x1000) + static_cast<std::uint64_t>(allocate_calls), bytes,
            device_ordinal};
    }

    [[nodiscard]] vastloom::Result<void> release(
        const vastloom::cuda::CudaBackendAllocation&) override {
        ++release_calls;
        if (fail_release) {
            return vastloom::make_error(vastloom::ErrorCode::io_error,
                                        "injected CUDA release failure");
        }
        memory.clear();
        return {};
    }

    [[nodiscard]] vastloom::Result<void> copy_from_host(
        const vastloom::cuda::CudaBackendAllocation& allocation,
        vastloom::ByteCount destination_offset,
        std::span<const std::byte> source) override {
        ++copy_from_calls;
        if (throw_copy_from) throw std::runtime_error("injected CUDA H2D exception");
        if (fail_copy_from) {
            return vastloom::make_error(vastloom::ErrorCode::io_error,
                                        "injected CUDA H2D failure");
        }
        if (destination_offset > allocation.bytes ||
            source.size() > allocation.bytes - destination_offset ||
            destination_offset > memory.size() ||
            source.size() > memory.size() - static_cast<std::size_t>(destination_offset)) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                        "fake CUDA H2D range is invalid");
        }
        std::memcpy(memory.data() + static_cast<std::size_t>(destination_offset),
                    source.data(), source.size());
        return {};
    }

    [[nodiscard]] vastloom::Result<void> copy_to_host(
        const vastloom::cuda::CudaBackendAllocation& allocation,
        vastloom::ByteCount source_offset,
        std::span<std::byte> destination) override {
        ++copy_to_calls;
        if (throw_copy_to) throw std::runtime_error("injected CUDA D2H exception");
        if (fail_copy_to) {
            return vastloom::make_error(vastloom::ErrorCode::io_error,
                                        "injected CUDA D2H failure");
        }
        if (source_offset > allocation.bytes ||
            destination.size() > allocation.bytes - source_offset ||
            source_offset > memory.size() ||
            destination.size() > memory.size() - static_cast<std::size_t>(source_offset)) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                        "fake CUDA D2H range is invalid");
        }
        std::memcpy(destination.data(),
                    memory.data() + static_cast<std::size_t>(source_offset),
                    destination.size());
        return {};
    }

    [[nodiscard]] vastloom::Result<void> launch_rms_norm(
        const vastloom::cuda::CudaBackendAllocation& allocation,
        vastloom::ByteCount input_offset,
        vastloom::ByteCount weight_offset,
        vastloom::ByteCount output_offset,
        std::size_t row_count,
        std::size_t width,
        float epsilon) override {
        ++launch_calls;
        if (throw_launch) throw std::runtime_error("injected CUDA kernel exception");
        if (fail_launch) {
            return vastloom::make_error(vastloom::ErrorCode::io_error,
                                        "injected CUDA kernel failure");
        }
        if (row_count == 0U || width == 0U || width > SIZE_MAX / row_count) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                        "fake CUDA RMSNorm shape is invalid");
        }
        const std::size_t elements = row_count * width;
        if (elements > SIZE_MAX / sizeof(float)) {
            return vastloom::make_error(vastloom::ErrorCode::resource_exhausted,
                                        "fake CUDA RMSNorm bytes overflow");
        }
        const std::size_t matrix_bytes = elements * sizeof(float);
        const std::size_t weight_bytes = width * sizeof(float);
        const auto valid = [&](vastloom::ByteCount offset, std::size_t bytes) {
            return offset <= allocation.bytes && bytes <= allocation.bytes - offset &&
                   offset <= memory.size() &&
                   bytes <= memory.size() - static_cast<std::size_t>(offset);
        };
        if (!valid(input_offset, matrix_bytes) || !valid(weight_offset, weight_bytes) ||
            !valid(output_offset, matrix_bytes)) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                        "fake CUDA RMSNorm range is invalid");
        }
        const auto read_float = [&](vastloom::ByteCount offset) {
            float value = 0.0F;
            std::memcpy(&value, memory.data() + static_cast<std::size_t>(offset), sizeof(value));
            return value;
        };
        const auto write_float = [&](vastloom::ByteCount offset, float value) {
            std::memcpy(memory.data() + static_cast<std::size_t>(offset), &value, sizeof(value));
        };
        for (std::size_t row = 0U; row < row_count; ++row) {
            double squares = 0.0;
            for (std::size_t column = 0U; column < width; ++column) {
                const auto index = row * width + column;
                const double value = static_cast<double>(read_float(
                    input_offset + static_cast<vastloom::ByteCount>(index * sizeof(float))));
                squares += value * value;
            }
            const double inverse = 1.0 / std::sqrt(
                squares / static_cast<double>(width) + static_cast<double>(epsilon));
            for (std::size_t column = 0U; column < width; ++column) {
                const auto index = row * width + column;
                const float input_value = read_float(
                    input_offset + static_cast<vastloom::ByteCount>(index * sizeof(float)));
                const float weight_value = read_float(
                    weight_offset + static_cast<vastloom::ByteCount>(column * sizeof(float)));
                write_float(
                    output_offset + static_cast<vastloom::ByteCount>(index * sizeof(float)),
                    static_cast<float>(static_cast<double>(input_value) * inverse *
                                       static_cast<double>(weight_value)));
            }
        }
        return {};
    }
};

vastloom::safety::ResourceLimits limits(vastloom::ByteCount vram) {
    vastloom::safety::ResourceLimits result;
    result.host_memory_hard_limit = 8U * vastloom::MiB;
    result.ram_hard_limit = 4U * vastloom::MiB;
    result.vram_hard_limit = vram;
    result.staging_hard_limit = 2U * vastloom::MiB;
    result.scratch_hard_limit = 2U * vastloom::MiB;
    result.minimum_system_ram_available = 1U;
    result.minimum_storage_available = 1U;
    return result;
}

vastloom::ByteCount reserved_vram(const vastloom::safety::ResourceGovernor& governor) {
    return governor.snapshot().reserved[static_cast<std::size_t>(
        vastloom::safety::ResourceKind::vram)];
}

vastloom::ByteCount reserved_scratch(const vastloom::safety::ResourceGovernor& governor) {
    return governor.snapshot().reserved[static_cast<std::size_t>(
        vastloom::safety::ResourceKind::scratch)];
}

void test_probe_and_formatting() {
    CHECK(!vastloom::cuda::CudaDriver::from_backend_for_testing(nullptr));
    auto backend = std::make_shared<FakeCudaBackend>();
    auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
    CHECK(driver);
    if (!driver) return;
    auto probe = driver.value().probe();
    CHECK(probe && backend->probe_calls == 1 && probe.value().devices.size() == 1U);
    if (probe) {
        CHECK(probe.value().devices.front().compute_capability_major == 8);
        CHECK(probe.value().devices.front().compute_capability_minor == 9);
        const auto json = vastloom::cuda::to_json(probe.value());
        const auto text = vastloom::cuda::to_text(probe.value());
        CHECK(json.find("\"compute_capability_major\":8") != std::string::npos);
        CHECK(text.find("8.9") != std::string::npos);
    }
    CHECK(vastloom::cuda::cuda_driver_version_text(12'080) == "12.8");
    CHECK(vastloom::cuda::cuda_driver_version_text(0) == "unknown");
    backend->fail_probe = true;
    CHECK(!driver.value().probe());
}

void test_governed_allocation_success_and_move() {
    vastloom::safety::ResourceGovernor governor(limits(16U * vastloom::MiB));
    auto backend = std::make_shared<FakeCudaBackend>();
    backend->governor = &governor;
    auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
    CHECK(driver);
    if (!driver) return;
    auto allocated = driver.value().allocate(0, 4U * vastloom::MiB, 2U * vastloom::MiB,
                                             governor, "cuda-test");
    CHECK(allocated && allocated.value().valid() && backend->reservation_visible);
    CHECK(backend->last_minimum_free == 2U * vastloom::MiB);
    CHECK(reserved_vram(governor) == 4U * vastloom::MiB);
    if (!allocated) return;
    vastloom::cuda::CudaAllocation moved = allocated.take_value();
    CHECK(moved.valid() && moved.bytes() == 4U * vastloom::MiB &&
          moved.device_ordinal() == 0 && moved.device_pointer() != 0U);
    auto released = moved.release();
    CHECK(released && !moved.valid() && backend->release_calls == 1);
    CHECK(reserved_vram(governor) == 0U);
    CHECK(moved.release());
}

void test_failures_are_transactional() {
    vastloom::safety::ResourceGovernor governor(limits(4U * vastloom::MiB));
    auto backend = std::make_shared<FakeCudaBackend>();
    backend->governor = &governor;
    auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
    CHECK(driver);
    if (!driver) return;
    CHECK(!driver.value().allocate(-1, 1U, 0U, governor, "negative-device"));
    CHECK(!driver.value().allocate(0, 0U, 0U, governor, "zero"));
    CHECK(!driver.value().allocate(0, 1U, 0U, governor, ""));
    CHECK(backend->allocate_calls == 0);
    CHECK(!driver.value().allocate(0, 5U * vastloom::MiB, 0U, governor, "too-large"));
    CHECK(backend->allocate_calls == 0 && reserved_vram(governor) == 0U);

    backend->fail_allocate = true;
    CHECK(!driver.value().allocate(0, 2U * vastloom::MiB, 1U, governor, "driver-fail"));
    CHECK(backend->allocate_calls == 1 && backend->reservation_visible &&
          reserved_vram(governor) == 0U);
    backend->fail_allocate = false;
    backend->throw_allocate = true;
    CHECK(!driver.value().allocate(0, 2U * vastloom::MiB, 1U, governor, "driver-throw"));
    CHECK(reserved_vram(governor) == 0U);
    backend->throw_allocate = false;
    backend->malformed_handle = true;
    CHECK(!driver.value().allocate(0, 2U * vastloom::MiB, 1U, governor, "bad-handle"));
    CHECK(backend->release_calls == 1 && reserved_vram(governor) == 0U);
    backend->fail_release = true;
    CHECK(!driver.value().allocate(0, 2U * vastloom::MiB, 1U, governor,
                                   "bad-handle-cleanup-fail"));
    CHECK(reserved_vram(governor) == 2U * vastloom::MiB);
}

void test_release_retry_and_fail_closed_destruction() {
    {
        vastloom::safety::ResourceGovernor governor(limits(8U * vastloom::MiB));
        auto backend = std::make_shared<FakeCudaBackend>();
        auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
        CHECK(driver);
        if (!driver) return;
        auto allocated = driver.value().allocate(0, 2U * vastloom::MiB, 1U, governor,
                                                 "release-retry");
        CHECK(allocated);
        if (!allocated) return;
        auto allocation = allocated.take_value();
        backend->fail_release = true;
        CHECK(!allocation.release() && allocation.valid() &&
              reserved_vram(governor) == 2U * vastloom::MiB);
        backend->fail_release = false;
        CHECK(allocation.release() && reserved_vram(governor) == 0U);
    }
    {
        vastloom::safety::ResourceGovernor governor(limits(8U * vastloom::MiB));
        auto backend = std::make_shared<FakeCudaBackend>();
        backend->fail_release = true;
        auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
        CHECK(driver);
        if (!driver) return;
        {
            auto allocated = driver.value().allocate(0, 2U * vastloom::MiB, 1U, governor,
                                                     "release-leak");
            CHECK(allocated);
            if (!allocated) return;
            auto allocation = allocated.take_value();
            CHECK(allocation.valid());
        }
        CHECK(backend->release_calls == 1);
        CHECK(reserved_vram(governor) == 2U * vastloom::MiB);
        CHECK(!driver.value().allocate(0, 7U * vastloom::MiB, 1U, governor,
                                      "accounting-stays-closed"));
    }
}

void test_copy_and_kernel_driver_contract() {
    vastloom::safety::ResourceGovernor governor(limits(8U * vastloom::MiB));
    auto backend = std::make_shared<FakeCudaBackend>();
    auto other_backend = std::make_shared<FakeCudaBackend>();
    auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
    auto other_driver = vastloom::cuda::CudaDriver::from_backend_for_testing(other_backend);
    CHECK(driver && other_driver);
    if (!driver || !other_driver) return;
    auto allocated = driver.value().allocate(0, 256U, 0U, governor, "copy-contract");
    CHECK(allocated);
    if (!allocated) return;
    auto allocation = allocated.take_value();
    const std::vector<float> source{1.0F, -2.0F, 3.0F, -4.0F};
    std::vector<float> destination(source.size(), 0.0F);
    CHECK(driver.value().copy_from_host(allocation, 0U, std::as_bytes(std::span(source))));
    CHECK(driver.value().copy_to_host(
        allocation, 0U, std::as_writable_bytes(std::span(destination))));
    CHECK(destination == source);
    CHECK(backend->copy_from_calls == 1 && backend->copy_to_calls == 1);

    CHECK(!other_driver.value().copy_from_host(
        allocation, 0U, std::as_bytes(std::span(source))));
    CHECK(!driver.value().copy_from_host(allocation, 0U, {}));
    CHECK(!driver.value().copy_from_host(
        allocation, 250U, std::as_bytes(std::span(source))));
    CHECK(!driver.value().copy_to_host(allocation, 0U, {}));
    CHECK(!driver.value().copy_to_host(
        allocation, 250U, std::as_writable_bytes(std::span(destination))));
    CHECK(!driver.value().launch_rms_norm(allocation, 0U, 32U, 64U, 0U, 4U, 1.0e-6F));
    CHECK(!driver.value().launch_rms_norm(allocation, 0U, 32U, 64U, 2U, 4U, 0.0F));
    CHECK(!driver.value().launch_rms_norm(allocation, 0U, 16U, 64U, 2U, 4U, 1.0e-6F));
    CHECK(!driver.value().launch_rms_norm(allocation, 0U, 32U, 250U, 2U, 4U, 1.0e-6F));
    CHECK(!driver.value().launch_rms_norm(allocation, 1U, 32U, 64U, 2U, 4U, 1.0e-6F));
    CHECK(backend->launch_calls == 0);
    CHECK(allocation.release());
    CHECK(reserved_vram(governor) == 0U);
}

void test_governed_rms_norm_success() {
    vastloom::safety::ResourceGovernor governor(limits(8U * vastloom::MiB));
    auto backend = std::make_shared<FakeCudaBackend>();
    auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
    CHECK(driver);
    if (!driver) return;
    const std::vector<float> input{0.25F, -0.5F, 1.0F, 2.0F,
                                   -1.5F, 0.75F, 0.125F, -0.25F};
    const std::vector<float> weight{0.5F, 1.0F, 1.5F, 2.0F};
    auto expected = vastloom::inference::deepseek_v4_rms_norm(
        input, weight, 2U, 4U, 1.0e-6F);
    CHECK(expected);
    if (!expected) return;
    {
        auto actual = vastloom::cuda::cuda_rms_norm(
            driver.value(), input, weight, 2U, 4U, 1.0e-6F, governor,
            {.device_ordinal = 0, .minimum_free_vram_after = 1U * vastloom::MiB});
        CHECK(actual);
        if (!actual) return;
        CHECK(actual.value().output == expected.value());
        CHECK(actual.value().stats.row_count == 2U && actual.value().stats.width == 4U);
        CHECK(actual.value().stats.host_to_device_bytes == 48U);
        CHECK(actual.value().stats.device_to_host_bytes == 32U);
        CHECK(actual.value().stats.device_buffer_bytes == 80U);
        CHECK(backend->copy_from_calls == 2 && backend->launch_calls == 1 &&
              backend->copy_to_calls == 1 && backend->release_calls == 1);
        CHECK(reserved_vram(governor) == 0U);
        CHECK(reserved_scratch(governor) == 32U);
    }
    CHECK(reserved_scratch(governor) == 0U);
}

void test_rms_norm_failures_are_transactional() {
    const std::vector<float> input{0.25F, -0.5F, 1.0F, 2.0F,
                                   -1.5F, 0.75F, 0.125F, -0.25F};
    const std::vector<float> weight{0.5F, 1.0F, 1.5F, 2.0F};
    for (int failure_stage = 0; failure_stage < 6; ++failure_stage) {
        vastloom::safety::ResourceGovernor governor(limits(8U * vastloom::MiB));
        auto backend = std::make_shared<FakeCudaBackend>();
        backend->fail_copy_from = failure_stage == 0;
        backend->fail_launch = failure_stage == 1;
        backend->fail_copy_to = failure_stage == 2;
        backend->throw_copy_from = failure_stage == 3;
        backend->throw_launch = failure_stage == 4;
        backend->throw_copy_to = failure_stage == 5;
        auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
        CHECK(driver);
        if (!driver) return;
        CHECK(!vastloom::cuda::cuda_rms_norm(
            driver.value(), input, weight, 2U, 4U, 1.0e-6F, governor,
            {.device_ordinal = 0, .minimum_free_vram_after = 0U}));
        CHECK(backend->release_calls == 1);
        CHECK(reserved_vram(governor) == 0U && reserved_scratch(governor) == 0U);
    }

    vastloom::safety::ResourceGovernor governor(limits(8U * vastloom::MiB));
    auto backend = std::make_shared<FakeCudaBackend>();
    auto driver = vastloom::cuda::CudaDriver::from_backend_for_testing(backend);
    CHECK(driver);
    if (!driver) return;
    CHECK(!vastloom::cuda::cuda_rms_norm(
        driver.value(), input, weight, 0U, 4U, 1.0e-6F, governor));
    CHECK(!vastloom::cuda::cuda_rms_norm(
        driver.value(), input, weight, 2U, 5U, 1.0e-6F, governor));
    CHECK(!vastloom::cuda::cuda_rms_norm(
        driver.value(), input, weight, 2U, 4U, 0.0F, governor));
    auto non_finite_input = input;
    non_finite_input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!vastloom::cuda::cuda_rms_norm(
        driver.value(), non_finite_input, weight, 2U, 4U, 1.0e-6F, governor));
    CHECK(!vastloom::cuda::cuda_rms_norm(
        driver.value(), {}, {}, 1U,
        vastloom::cuda::kCudaRmsNormMaximumElements + 1U, 1.0e-6F, governor));
    CHECK(backend->allocate_calls == 0);

    auto constrained = limits(8U * vastloom::MiB);
    constrained.scratch_hard_limit = 16U;
    vastloom::safety::ResourceGovernor constrained_governor(constrained);
    CHECK(!vastloom::cuda::cuda_rms_norm(
        driver.value(), input, weight, 2U, 4U, 1.0e-6F, constrained_governor));
    CHECK(backend->allocate_calls == 0);
    CHECK(reserved_scratch(constrained_governor) == 0U);

    vastloom::safety::ResourceGovernor release_governor(limits(8U * vastloom::MiB));
    auto release_backend = std::make_shared<FakeCudaBackend>();
    release_backend->fail_release = true;
    auto release_driver = vastloom::cuda::CudaDriver::from_backend_for_testing(release_backend);
    CHECK(release_driver);
    if (!release_driver) return;
    CHECK(!vastloom::cuda::cuda_rms_norm(
        release_driver.value(), input, weight, 2U, 4U, 1.0e-6F, release_governor,
        {.device_ordinal = 0, .minimum_free_vram_after = 0U}));
    CHECK(release_backend->release_calls == 2);
    CHECK(reserved_vram(release_governor) == 80U);
    CHECK(reserved_scratch(release_governor) == 0U);
}

}  // namespace

int main() {
    test_probe_and_formatting();
    test_governed_allocation_success_and_move();
    test_failures_are_transactional();
    test_release_retry_and_fail_closed_destruction();
    test_copy_and_kernel_driver_contract();
    test_governed_rms_norm_success();
    test_rms_norm_failures_are_transactional();
    if (failures != 0) {
        std::cerr << failures << " CUDA driver test(s) failed\n";
        return 1;
    }
    std::cout << "CUDA driver probe/allocator/RMSNorm tests: PASS\n";
    return 0;
}
