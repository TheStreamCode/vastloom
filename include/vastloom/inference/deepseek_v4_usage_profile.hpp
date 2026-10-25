#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

inline constexpr const char* kDeepseekV4UsageProfileFormat =
    "vastloom.deepseek-v4-usage-profile.v1";

struct DeepseekV4UsageProfile {
    std::string model_id;
    std::string revision;
    std::uint64_t observations{0U};
    std::map<std::uint32_t, std::map<std::uint32_t, std::uint64_t>> counts;
};

struct DeepseekV4UsageProfileOptions {
    std::uint32_t max_layers{64U};
    std::uint32_t max_experts{1'024U};
    std::uint64_t max_observations{1'000'000'000ULL};
};

// A missing file is an empty profile, not an error: the first run creates it. A present file must
// match the pinned model identity and the format version, or the load fails closed.
[[nodiscard]] Result<DeepseekV4UsageProfile> load_deepseek_v4_usage_profile(
    const std::filesystem::path& path,
    std::string_view model_id,
    std::string_view revision,
    const DeepseekV4UsageProfileOptions& options = {});

[[nodiscard]] Result<void> record_deepseek_v4_usage(
    DeepseekV4UsageProfile& profile,
    std::uint32_t layer,
    std::span<const std::uint32_t> experts,
    const DeepseekV4UsageProfileOptions& options = {});

// Writes through a temporary file and renames it; the profile is a local sidecar, never a model
// artifact.
[[nodiscard]] Result<void> save_deepseek_v4_usage_profile(
    const std::filesystem::path& path,
    const DeepseekV4UsageProfile& profile);

// Highest-count experts for a layer, descending by count then expert id.
[[nodiscard]] std::vector<std::pair<std::uint32_t, std::uint64_t>>
top_deepseek_v4_usage(const DeepseekV4UsageProfile& profile,
                      std::uint32_t layer,
                      std::size_t limit);

}  // namespace vastloom::inference
