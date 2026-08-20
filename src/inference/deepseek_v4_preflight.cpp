#include "vastloom/inference/deepseek_v4_preflight.hpp"

#include <algorithm>

namespace vastloom::inference {

DeepseekV4ForwardPreflightPlan
deepseek_v4_forward_preflight(const DeepseekV4ForwardPreflightInput& input) {
    DeepseekV4ForwardPreflightPlan plan;

    // Trunks are streamed through a bounded window rather than all held, so what a pass needs
    // is a few layers' worth, not the whole depth: the floor scales with one layer, not with
    // the model. Charged as the window's capacity plus one spare layer, against the largest
    // layer rather than the average, because a refusal that comes too late is worse than one
    // that comes too early.
    const auto average_trunk =
        input.depth == 0U ? ByteCount{0U} : input.trunk_bytes / input.depth;
    plan.resident_trunk_bytes =
        std::min(input.trunk_bytes, average_trunk * (input.trunk_window + 1U) * 2U);

    plan.staging_charge = std::min(input.largest_source_range, input.staging_hard_limit);
    plan.ggml_charge = input.ggml_pool_bytes;
    const ByteCount overhead = plan.staging_charge + plan.ggml_charge;

    // Prefill keeps one selected union and drops it after the layer. Decode keeps the last
    // experts in every layer so the next token can hit. Charge the larger of the two sets.
    const auto per_layer_experts =
        std::min(input.prompt_tokens * input.experts_per_token, input.expert_count);
    const auto cache_size = std::max(input.expert_cache_entries, per_layer_experts);
    plan.expert_cache_bytes = input.largest_expert_bundle * cache_size;

    // Persisting the decode experts is only worth what it costs: enable it when the whole
    // resident set, overhead included, still fits the live budget, never by stealing the trunk
    // window that pays for it.
    if (input.max_tokens > 1U) {
        const ByteCount persist_bytes =
            input.largest_expert_bundle * input.experts_per_token * input.depth;
        const ByteCount persist_experts = std::max(plan.expert_cache_bytes, persist_bytes);
        if (plan.resident_trunk_bytes + input.global_bytes + persist_experts + overhead <=
            input.ram_budget) {
            plan.expert_cache_bytes = persist_experts;
            plan.persist_decode = true;
        }
    }

    plan.required_bytes = plan.resident_trunk_bytes + input.global_bytes +
                          plan.expert_cache_bytes + overhead;
    plan.fits = plan.required_bytes <= input.ram_budget;
    return plan;
}

}  // namespace vastloom::inference
