#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_set>

namespace vastloom::inference {

// Admission filter for routed-expert caches: an expert seen for the first time is
// streamed and dropped instead of occupying a long-lived cache entry, and a repeated
// miss earns admission. The ghost history stores identifiers only, never weights, and
// is bounded so a scan over cold experts cannot displace a resident hot expert.
// This is an admission heuristic, not a route predictor: it never chooses which expert
// executes, and it makes no claim about hit rate until a real trace measures one.
class AdaptiveExpertAdmission {
public:
    explicit AdaptiveExpertAdmission(std::size_t cache_slots)
        : ghost_limit_(std::clamp(cache_slots, kMinimumSlots, kMaximumSlots) * kGhostMultiplier) {}

    // Returns true when `key` was already recorded in the bounded ghost history, which
    // means the cache entry is worth its RAM on this miss.
    [[nodiscard]] bool should_admit_miss(std::uint32_t key) {
        if (ghost_.contains(key)) {
            return true;
        }
        if (fifo_.size() == ghost_limit_) {
            ghost_.erase(fifo_.front());
            fifo_.pop_front();
        }
        fifo_.push_back(key);
        ghost_.insert(key);
        return false;
    }

    [[nodiscard]] std::size_t ghost_size() const noexcept { return fifo_.size(); }
    [[nodiscard]] std::size_t ghost_limit() const noexcept { return ghost_limit_; }

private:
    static constexpr std::size_t kMinimumSlots = 4U;
    static constexpr std::size_t kMaximumSlots = 1024U;
    static constexpr std::size_t kGhostMultiplier = 4U;

    const std::size_t ghost_limit_;
    std::deque<std::uint32_t> fifo_;
    std::unordered_set<std::uint32_t> ghost_;
};

}  // namespace vastloom::inference
