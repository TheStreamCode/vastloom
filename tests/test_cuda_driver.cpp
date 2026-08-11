#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "vastloom/core/bytes.hpp"
#include "vastloom/cuda/cuda_driver.hpp"
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
    bool reservation_visible{false};
    int probe_calls{0};
    int allocate_calls{0};
    int release_calls{0};
    vastloom::ByteCount last_minimum_free{0U};

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
            return vastloom::cuda::CudaBackendAllocation{
                1U, UINT64_C(0x2000), bytes + 1U, device_ordinal};
        }
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

}  // namespace

int main() {
    test_probe_and_formatting();
    test_governed_allocation_success_and_move();
    test_failures_are_transactional();
    test_release_retry_and_fail_closed_destruction();
    if (failures != 0) {
        std::cerr << failures << " CUDA driver test(s) failed\n";
        return 1;
    }
    std::cout << "CUDA driver probe/allocator tests: PASS\n";
    return 0;
}
