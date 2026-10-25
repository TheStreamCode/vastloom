#include <cstddef>
#include <cstdint>
#include <iostream>

#include "vastloom/inference/deepseek_v4_cache_plan.hpp"

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

vastloom::inference::DeepseekV4CachePlanInput base_input() {
    vastloom::inference::DeepseekV4CachePlanInput input;
    input.ram_budget = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    input.resident_floor = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    input.max_expert_bytes = 8ULL * 1024ULL * 1024ULL;
    input.request_entries = 6U;
    input.auto_entries = 8U;
    input.max_entries = 128U;
    return input;
}

void test_auto_plan_fits_headroom() {
    const auto plan = vastloom::inference::plan_deepseek_v4_expert_cache(base_input());
    CHECK(static_cast<bool>(plan));
    if (plan) {
        CHECK(plan.value().ceiling_entries == 128U);
        CHECK(plan.value().chosen_entries == 8U);
        CHECK(plan.value().floor_entries == 6U);
        CHECK(plan.value().ceiling_bytes == 128ULL * 8ULL * 1024ULL * 1024ULL);
        CHECK(!plan.value().clamped);
        CHECK(!plan.value().requested_exceeds_ceiling);
    }
}

void test_requested_capacity_is_bounded() {
    auto input = base_input();
    input.requested_entries = 10U;
    const auto plan = vastloom::inference::plan_deepseek_v4_expert_cache(input);
    CHECK(static_cast<bool>(plan));
    if (plan) CHECK(plan.value().chosen_entries == 10U);

    input.requested_entries = 1U;
    const auto too_small = vastloom::inference::plan_deepseek_v4_expert_cache(input);
    CHECK(!too_small);
}

void test_clamp_and_no_headroom() {
    auto input = base_input();
    input.ram_budget = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    input.resident_floor = 100ULL * 1024ULL * 1024ULL;
    input.max_expert_bytes = 64ULL * 1024ULL * 1024ULL;
    input.request_entries = 1U;
    input.auto_entries = 8U;
    const auto plan = vastloom::inference::plan_deepseek_v4_expert_cache(input);
    CHECK(static_cast<bool>(plan));
    if (plan) {
        CHECK(plan.value().ceiling_entries == 14U);
        CHECK(plan.value().chosen_entries == 8U);
        CHECK(plan.value().chosen_entries <= plan.value().ceiling_entries);
    }

    auto no_headroom = input;
    no_headroom.resident_floor = no_headroom.ram_budget;
    CHECK(!vastloom::inference::plan_deepseek_v4_expert_cache(no_headroom));

    auto no_request_fit = input;
    no_request_fit.request_entries = 20U;
    CHECK(!vastloom::inference::plan_deepseek_v4_expert_cache(no_request_fit));
}

void test_ceiling_is_monotonic_in_budget() {
    std::size_t previous = 0U;
    for (std::uint64_t gib = 1U; gib <= 8U; ++gib) {
        auto input = base_input();
        input.ram_budget = gib * 1024ULL * 1024ULL * 1024ULL;
        input.resident_floor = 128ULL * 1024ULL * 1024ULL;
        input.max_expert_bytes = 8ULL * 1024ULL * 1024ULL;
        input.request_entries = 2U;
        const auto plan = vastloom::inference::plan_deepseek_v4_expert_cache(input);
        CHECK(static_cast<bool>(plan));
        if (plan) {
            CHECK(plan.value().ceiling_entries >= previous);
            previous = plan.value().ceiling_entries;
        }
    }
}

}  // namespace

int main() {
    test_auto_plan_fits_headroom();
    test_requested_capacity_is_bounded();
    test_clamp_and_no_headroom();
    test_ceiling_is_monotonic_in_budget();
    if (failures != 0) {
        std::cerr << failures << " cache-plan test(s) failed\n";
        return 1;
    }
    std::cout << "Cache-plan tests PASS\n";
    return 0;
}
