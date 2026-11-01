#include <cstdint>
#include <iostream>

#include "vastloom/hardware/hardware_probe.hpp"

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

constexpr vastloom::ByteCount GiB = 1024ULL * 1024ULL * 1024ULL;

vastloom::hardware::MemoryInfo host_memory() {
    return vastloom::hardware::MemoryInfo{32ULL * GiB, 24ULL * GiB};
}

void test_container_limit_clamps_total_and_available() {
    const auto clamped =
        vastloom::hardware::clamp_memory_to_cgroup_limit(host_memory(), "8589934592\n", "6442450944\n");
    CHECK(clamped.total_bytes == 8ULL * GiB);
    CHECK(clamped.available_bytes == 2ULL * GiB);
}

void test_unlimited_markers_keep_host_values() {
    const auto host = host_memory();
    for (const auto* limit_text : {"max\n", "9223372036854771712\n", "\n", "not-a-number\n", "0\n"}) {
        const auto clamped = vastloom::hardware::clamp_memory_to_cgroup_limit(host, limit_text, "0\n");
        CHECK(clamped.total_bytes == host.total_bytes);
        CHECK(clamped.available_bytes == host.available_bytes);
    }
}

void test_usage_edges_stay_bounded() {
    const auto over_limit =
        vastloom::hardware::clamp_memory_to_cgroup_limit(host_memory(), "4294967296\n", "9999999999\n");
    CHECK(over_limit.total_bytes == 4ULL * GiB);
    CHECK(over_limit.available_bytes == 0U);
    for (const auto* usage_text : {"", "\n", "garbage\n"}) {
        const auto unknown_usage =
            vastloom::hardware::clamp_memory_to_cgroup_limit(host_memory(), "4294967296\n", usage_text);
        CHECK(unknown_usage.total_bytes == 4ULL * GiB);
        CHECK(unknown_usage.available_bytes == 4ULL * GiB);
    }
}

void test_host_below_limit_is_unchanged() {
    const vastloom::hardware::MemoryInfo small_host{2ULL * GiB, 1ULL * GiB};
    const auto clamped =
        vastloom::hardware::clamp_memory_to_cgroup_limit(small_host, "8589934592\n", "0\n");
    CHECK(clamped.total_bytes == small_host.total_bytes);
    CHECK(clamped.available_bytes == small_host.available_bytes);
}

}  // namespace

int main() {
    test_container_limit_clamps_total_and_available();
    test_unlimited_markers_keep_host_values();
    test_usage_edges_stay_bounded();
    test_host_below_limit_is_unchanged();
    if (failures != 0) {
        std::cerr << failures << " cgroup memory-limit check(s) failed\n";
        return 1;
    }
    std::cout << "Hardware cgroup memory-limit tests passed\n";
    return 0;
}
