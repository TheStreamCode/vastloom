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
#include "vastloom/inference/ggml_quantization.hpp"
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
    unsigned asynchronous_workers{0U};
    std::size_t asynchronous_queue_capacity{0U};
    model::ExpertLoadOptions load;
};

struct DeepseekV4ExpertCacheSnapshot {
    std::size_t cached_experts{0U};
    ByteCount resident_bytes{0U};
    std::size_t requests{0U};
    std::size_t requested_experts{0U};
    std::size_t hits{0U};
    std::size_t misses{0U};
    // Selected identities that were also selected on this cache's previous begin.
    // Independent of whether the payload is still resident: release-after-layer
    // zeros hits but this still says whether a decode persist would pay.
    std::size_t recurrent_experts{0U};
    std::size_t evictions{0U};
    std::size_t abandoned_loads{0U};
    ByteCount logical_bytes_read{0U};
    std::size_t read_calls{0U};
    std::size_t bypass_read_calls{0U};
    std::size_t buffered_read_calls{0U};
    ByteCount wasted_prefetch_bytes{0U};
    bool telemetry_overflow{false};
};

// Converts one executable DeepSeek GGUF layer from the universal catalog into strict W1/W2/W3
// bindings. Physical columns may include quantization padding, but logical graph dimensions and
// complete expert count must match exactly.
[[nodiscard]] Result<std::vector<DeepseekV4ExpertBundleBinding>>
deepseek_v4_make_gguf_expert_bindings(
    const model::LogicalTensorCatalog& catalog,
    std::uint32_t layer,
    std::uint32_t expected_experts,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    std::size_t max_experts = 1'024U);

// Architecture-specific bridge from one cache-resolved official IQ1_S/IQ3_XXS bundle to the
// existing caller-buffered reference kernel. No allocation or routing decision occurs here.
[[nodiscard]] Result<void> deepseek_v4_execute_resolved_iq1_s_iq3_xxs_f32(
    const DeepseekV4ResolvedExpert& expert,
    std::span<const float> input,
    RoutedExpertScratch scratch,
    std::span<float> output,
    const QuantizedKernelLimits& limits = {});

// Router-first complete-bundle RAM cache. Synchronous mode is the correctness baseline; bounded
// asynchronous mode submits all selected misses and lets the graph acquire each expert at its
// compute boundary. Neither mode splits an expert across tiers.
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
    // Drops everyone except the last begin's identities. Decode uses this instead of
    // clear so the next token can hit without holding a whole prefill union.
    [[nodiscard]] Result<void> retain_last_selected();

private:
    class AsyncBatch;

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
    [[nodiscard]] Result<std::unique_ptr<DeepseekV4ExpertBatch>> begin_resolve(
        std::string_view block_prefix,
        std::span<const std::uint32_t> unique_expert_ids);
    [[nodiscard]] Result<void> evict_one(std::span<const std::uint32_t> protected_ids);

    std::map<std::uint32_t, DeepseekV4ExpertBundleBinding> bindings_;
    safety::ResourceGovernor& governor_;
    DeepseekV4ExpertCacheOptions options_;
    mutable std::mutex mutex_;
    std::map<std::uint32_t, CacheEntry> cache_;
    std::unique_ptr<model::BoundedExpertLoadPool> load_pool_;
    DeepseekV4ExpertCacheSnapshot telemetry_;
    std::vector<std::uint32_t> last_selected_ids_;
    std::size_t tick_{0U};
    bool async_batch_active_{false};
};

}  // namespace vastloom::inference
