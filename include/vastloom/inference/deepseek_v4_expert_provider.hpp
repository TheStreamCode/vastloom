#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/inference/deepseek_v4_reference.hpp"
#include "vastloom/model/expert_loader.hpp"
#include "vastloom/model/tensor_catalog.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::inference {

enum class DeepseekV4ExpertWeight : std::uint8_t {
    w1,
    w2,
    w3,
};

struct DeepseekV4ExpertMatrixBinding {
    DeepseekV4ExpertWeight weight{DeepseekV4ExpertWeight::w1};
    std::size_t catalog_tensor_index{0U};
    std::uint32_t ggml_type{0U};
    std::size_t rows{0U};
    std::size_t logical_columns{0U};
    std::size_t storage_columns{0U};
};

struct DeepseekV4ExpertBundleBinding {
    std::uint32_t expert_id{0U};
    model::ExpertBundle bundle;
    std::array<DeepseekV4ExpertMatrixBinding, 3U> matrices;
};

struct DeepseekV4ExpertCacheOptions {
    std::string block_prefix;
    std::size_t max_cached_experts{2U};
    std::size_t max_experts_per_request{2U};
    model::ExpertLoadOptions load;
};

struct DeepseekV4ExpertCacheSnapshot {
    std::size_t cached_experts{0U};
    ByteCount resident_bytes{0U};
    std::size_t requests{0U};
    std::size_t requested_experts{0U};
    std::size_t hits{0U};
    std::size_t misses{0U};
    std::size_t evictions{0U};
    ByteCount logical_bytes_read{0U};
};

// Synchronous correctness baseline for router-first expert acquisition. The cache owns complete,
// indivisible RAM bundles and never splits one across tiers. Its callback is bounded by both the
// request and cache limits; asynchronous prefetch/overlap belongs to a later scheduler.
class DeepseekV4BoundedExpertCache final
    : public std::enable_shared_from_this<DeepseekV4BoundedExpertCache> {
public:
    [[nodiscard]] static Result<std::shared_ptr<DeepseekV4BoundedExpertCache>> create(
        std::vector<DeepseekV4ExpertBundleBinding> bindings,
        safety::ResourceGovernor& governor,
        DeepseekV4ExpertCacheOptions options);

    DeepseekV4BoundedExpertCache(const DeepseekV4BoundedExpertCache&) = delete;
    DeepseekV4BoundedExpertCache& operator=(const DeepseekV4BoundedExpertCache&) = delete;
    ~DeepseekV4BoundedExpertCache();

    [[nodiscard]] DeepseekV4ExpertProviderBinding provider_binding();
    [[nodiscard]] DeepseekV4ExpertCacheSnapshot snapshot() const;
    [[nodiscard]] Result<void> clear();

private:
    struct CacheEntry {
        DeepseekV4ResolvedExpert expert;
        model::ResidencyObject residency;
        std::size_t last_use{0U};
    };

    DeepseekV4BoundedExpertCache(
        std::map<std::uint32_t, DeepseekV4ExpertBundleBinding> bindings,
        safety::ResourceGovernor& governor,
        DeepseekV4ExpertCacheOptions options);

    [[nodiscard]] Result<std::vector<DeepseekV4ResolvedExpert>> resolve(
        std::string_view block_prefix,
        std::span<const std::uint32_t> unique_expert_ids);
    [[nodiscard]] Result<void> evict_one(std::span<const std::uint32_t> protected_ids);

    std::map<std::uint32_t, DeepseekV4ExpertBundleBinding> bindings_;
    safety::ResourceGovernor& governor_;
    DeepseekV4ExpertCacheOptions options_;
    mutable std::mutex mutex_;
    std::map<std::uint32_t, CacheEntry> cache_;
    DeepseekV4ExpertCacheSnapshot telemetry_;
    std::size_t tick_{0U};
};

}  // namespace vastloom::inference
