#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

#include "stargate/core/bytes.hpp"
#include "stargate/core/status.hpp"

namespace stargate::research {

enum class CachePolicy {
    none,
    lru,
    lfu,
    static_hot,
};

struct TraceSimulationConfig {
    ByteCount vram_cache_bytes{0};
    ByteCount ram_cache_bytes{0};
    CachePolicy policy{CachePolicy::lru};
    std::size_t max_records{10'000'000};
};

struct TraceSimulationReport {
    CachePolicy policy{CachePolicy::lru};
    std::size_t records{0};
    std::size_t tokens{0};
    std::size_t unique_experts{0};
    std::size_t cold_expert_accesses{0};
    std::size_t reused_expert_accesses{0};
    double reuse_access_ratio{0.0};
    double average_unique_reuse_distance{0.0};
    std::size_t p50_unique_reuse_distance{0};
    std::size_t p95_unique_reuse_distance{0};
    std::size_t maximum_unique_reuse_distance{0};
    std::size_t vram_hits{0};
    std::size_t ram_hits{0};
    std::size_t storage_misses{0};
    std::size_t evictions{0};
    ByteCount nvme_bytes{0};
    ByteCount pcie_bytes{0};
    ByteCount warmup_nvme_bytes{0};
    double vram_hit_ratio{0.0};
    double ram_hit_ratio{0.0};
    double storage_miss_ratio{0.0};
    double average_nvme_bytes_per_token{0.0};
    ByteCount p50_nvme_bytes_per_token{0};
    ByteCount p95_nvme_bytes_per_token{0};
    ByteCount maximum_nvme_bytes_per_token{0};
};

[[nodiscard]] Result<CachePolicy> parse_cache_policy(std::string_view value);
[[nodiscard]] std::string_view to_string(CachePolicy policy) noexcept;
[[nodiscard]] Result<TraceSimulationReport> simulate_trace(const std::filesystem::path& trace_path,
                                                            const TraceSimulationConfig& config);
[[nodiscard]] std::string to_text(const TraceSimulationReport& report);
[[nodiscard]] std::string to_json(const TraceSimulationReport& report);

}  // namespace stargate::research
