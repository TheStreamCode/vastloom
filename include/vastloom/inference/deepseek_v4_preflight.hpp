#pragma once

#include <cstddef>

#include "vastloom/core/bytes.hpp"

namespace vastloom::inference {

// Exact resident cost of a deepseek-forward run, derived from the catalog before anything is
// allocated. The runtime charges every one of these quantities against the same aggregate host
// budget through the Governor, so an approved preflight must already include what the loader,
// the bounded reads and the compute backend will reserve later; a refusal that comes too late
// is worse than one that comes too early.
struct DeepseekV4ForwardPreflightInput {
    ByteCount trunk_bytes{0U};
    ByteCount global_bytes{0U};
    ByteCount largest_expert_bundle{0U};
    // Largest single source range any bounded read will move; staging is reserved per load for
    // the load's largest range, so the preflight charges the worst case once.
    ByteCount largest_source_range{0U};
    std::size_t depth{0U};
    std::size_t experts_per_token{0U};
    std::size_t expert_count{0U};
    std::size_t prompt_tokens{0U};
    std::size_t expert_cache_entries{0U};
    std::size_t trunk_window{2U};
    std::size_t max_tokens{1U};
    ByteCount ram_budget{0U};
    ByteCount staging_hard_limit{0U};
    // Zero when the ggml backend is off: the pool is reserved as scratch only when enabled.
    ByteCount ggml_pool_bytes{0U};
};

struct DeepseekV4ForwardPreflightPlan {
    ByteCount resident_trunk_bytes{0U};
    ByteCount expert_cache_bytes{0U};
    ByteCount staging_charge{0U};
    ByteCount ggml_charge{0U};
    ByteCount required_bytes{0U};
    bool persist_decode{false};
    bool fits{false};
};

[[nodiscard]] DeepseekV4ForwardPreflightPlan
deepseek_v4_forward_preflight(const DeepseekV4ForwardPreflightInput& input);

}  // namespace vastloom::inference
