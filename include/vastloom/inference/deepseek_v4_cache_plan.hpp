#pragma once

#include <cstddef>
#include <string>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::inference {

struct DeepseekV4CachePlanInput {
    ByteCount ram_budget{0U};
    ByteCount resident_floor{0U};
    ByteCount max_expert_bytes{0U};
    // The cache must be able to hold one layer request (the selected-expert union).
    std::size_t request_entries{0U};
    // 0 asks the planner to use the automatic capacity passed in auto_entries.
    std::size_t requested_entries{0U};
    std::size_t auto_entries{0U};
    std::size_t max_entries{128U};
};

struct DeepseekV4CachePlan {
    std::size_t floor_entries{0U};
    std::size_t ceiling_entries{0U};
    std::size_t chosen_entries{0U};
    ByteCount resident_floor_bytes{0U};
    ByteCount headroom_bytes{0U};
    ByteCount ceiling_bytes{0U};
    ByteCount chosen_bytes{0U};
    bool clamped{false};
    bool requested_exceeds_ceiling{false};
    std::string reason;
};

// Bounded cache planner for the DeepSeek chain. It keeps the cache above one request's expert
// union and below the live RAM headroom, and refuses a plan that cannot satisfy both. The ceiling
// is derived from the current process budget, not from a machine-independent constant, so a larger
// machine can admit a larger cache while a tight one fails closed instead of paging.
[[nodiscard]] Result<DeepseekV4CachePlan> plan_deepseek_v4_expert_cache(
    const DeepseekV4CachePlanInput& input);

}  // namespace vastloom::inference
