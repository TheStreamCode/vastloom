#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>

#include "vastloom/core/bytes.hpp"
#include "vastloom/research/trace_simulator.hpp"

#include "test_temp_dir.hpp"

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
    return vastloom_test::make_temp_dir("vastloom-trace-adaptive-tests-");
}

std::filesystem::path write_scan_trace(const std::filesystem::path& root) {
    const auto path = root / "adaptive-scan-trace.csv";
    std::ofstream trace(path);
    trace << "token,layer,expert,size_bytes\n"
          << "0,0,0,64MiB\n"
          << "0,0,0,64MiB\n";
    for (std::uint32_t expert = 100U; expert < 150U; ++expert) {
        trace << "1,0," << expert << ",64MiB\n";
    }
    trace << "2,0,0,64MiB\n";
    for (std::uint32_t expert = 150U; expert < 200U; ++expert) {
        trace << "3,0," << expert << ",64MiB\n";
    }
    trace << "4,0,0,64MiB\n";
    trace.close();
    return path;
}

void test_policy_round_trip() {
    const auto parsed = vastloom::research::parse_cache_policy("Adaptive");
    CHECK(parsed && parsed.value() == vastloom::research::CachePolicy::adaptive);
    CHECK(vastloom::research::to_string(vastloom::research::CachePolicy::adaptive) == "adaptive");
    CHECK(!vastloom::research::parse_cache_policy("unbounded"));
}

void test_scan_resistance_on_a_scan_trace() {
    // 104 records: a hot expert seen twice, 100 one-shot experts around it, and two later
    // reuses of the hot expert. One 64 MiB slot holds exactly one expert.
    const auto root = make_temp_dir();
    const auto path = write_scan_trace(root);

    vastloom::research::TraceSimulationConfig config;
    config.ram_cache_bytes = 64U * vastloom::MiB;

    config.policy = vastloom::research::CachePolicy::none;
    const auto no_cache = vastloom::research::simulate_trace(path, config);
    CHECK(no_cache && no_cache.value().records == 104U);
    CHECK(no_cache && no_cache.value().storage_misses == 104U);

    config.policy = vastloom::research::CachePolicy::lru;
    const auto lru = vastloom::research::simulate_trace(path, config);
    CHECK(lru && lru.value().ram_hits == 1U);
    CHECK(lru && lru.value().storage_misses == 103U);

    config.policy = vastloom::research::CachePolicy::adaptive;
    const auto adaptive = vastloom::research::simulate_trace(path, config);
    CHECK(adaptive && adaptive.value().ram_hits == 2U);
    CHECK(adaptive && adaptive.value().storage_misses == 102U);
    CHECK(adaptive && adaptive.value().nvme_bytes == 102U * 64U * vastloom::MiB);

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

}  // namespace

int main() {
    test_policy_round_trip();
    test_scan_resistance_on_a_scan_trace();
    if (failures != 0) {
        std::cerr << failures << " adaptive trace-simulation check(s) failed\n";
        return 1;
    }
    std::cout << "Adaptive trace-simulation tests passed\n";
    return 0;
}
