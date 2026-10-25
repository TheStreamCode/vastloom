#include "vastloom/inference/deepseek_v4_cache_plan.hpp"

#include <algorithm>
#include <limits>

namespace vastloom::inference {
namespace {

std::string describe(ByteCount bytes) {
    return std::to_string(bytes) + " B";
}

}  // namespace

Result<DeepseekV4CachePlan> plan_deepseek_v4_expert_cache(
    const DeepseekV4CachePlanInput& input) {
    if (input.ram_budget == 0U || input.max_expert_bytes == 0U ||
        input.request_entries == 0U || input.max_entries == 0U ||
        input.requested_entries > input.max_entries ||
        input.auto_entries > input.max_entries) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek cache plan input is invalid");
    }
    if (input.resident_floor >= input.ram_budget) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek cache plan has no RAM headroom after the resident floor",
                          describe(input.resident_floor) + " >= " +
                              describe(input.ram_budget));
    }
    DeepseekV4CachePlan plan;
    plan.resident_floor_bytes = input.resident_floor;
    plan.headroom_bytes = input.ram_budget - input.resident_floor;
    const auto ceiling_by_bytes = plan.headroom_bytes / input.max_expert_bytes;
    plan.ceiling_entries = static_cast<std::size_t>(
        std::min<ByteCount>(ceiling_by_bytes,
                            static_cast<ByteCount>(input.max_entries)));
    if (plan.ceiling_entries < input.request_entries) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek cache plan cannot hold one layer request in the safe headroom",
                          "ceiling " + std::to_string(plan.ceiling_entries) +
                              " < request " + std::to_string(input.request_entries));
    }
    plan.ceiling_bytes = static_cast<ByteCount>(plan.ceiling_entries) *
                         input.max_expert_bytes;
    plan.floor_entries = input.request_entries;

    const std::size_t requested =
        input.requested_entries == 0U ? input.auto_entries : input.requested_entries;
    plan.requested_exceeds_ceiling = requested > plan.ceiling_entries;
    plan.chosen_entries = std::min(requested, plan.ceiling_entries);
    plan.clamped = plan.chosen_entries < requested;
    if (plan.chosen_entries < input.request_entries) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek cache plan is smaller than one layer request",
                          "chosen " + std::to_string(plan.chosen_entries) +
                              " < request " + std::to_string(input.request_entries));
    }
    plan.chosen_bytes = static_cast<ByteCount>(plan.chosen_entries) *
                        input.max_expert_bytes;
    plan.reason = plan.clamped
                      ? "clamped to the live RAM headroom"
                      : "fits the live RAM headroom";
    return plan;
}

}  // namespace vastloom::inference
