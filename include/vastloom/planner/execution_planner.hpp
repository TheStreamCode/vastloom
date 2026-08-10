#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/hardware/hardware_probe.hpp"
#include "vastloom/model/manifest.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::planner {

enum class Profile {
    safe,
    balanced,
    performance,
    low_memory,
    quiet,
};

struct ExecutionPlan {
    Profile profile{Profile::balanced};
    bool compatible{false};
    std::string strategy;
    std::string model_id;
    std::string selected_gpu;

    ByteCount os_ram_reserve{0};
    ByteCount emergency_ram_reserve{0};
    ByteCount safe_ram_budget{0};
    ByteCount vram_reserve{0};
    ByteCount safe_vram_budget{0};
    ByteCount storage_reserve{0};
    ByteCount additional_storage_required{0};

    ByteCount minimum_ram_working_set_bytes{0};
    ByteCount expert_bundle_bytes{0};
    ByteCount expert_tier_budget_bytes{0};
    std::string expert_admission_tier;

    ByteCount dense_ram_bytes{0};
    ByteCount expert_ram_bytes{0};
    ByteCount kv_ram_bytes{0};
    ByteCount staging_ram_bytes{0};
    ByteCount scratch_ram_bytes{0};
    ByteCount gpu_resident_bytes{0};
    ByteCount gpu_kv_bytes{0};
    ByteCount gpu_staging_bytes{0};
    ByteCount nvme_backing_bytes{0};

    unsigned io_workers{1};
    unsigned prefetch_depth{1};
    unsigned speculative_window{0};

    safety::ResourceLimits governor_limits;
    std::vector<std::string> blockers;
    std::vector<std::string> warnings;
};

[[nodiscard]] Result<Profile> parse_profile(std::string_view value);
[[nodiscard]] std::string_view to_string(Profile profile) noexcept;
[[nodiscard]] Result<ExecutionPlan> build_plan(const hardware::HardwareSnapshot& hardware,
                                               const model::ModelManifest& manifest,
                                               Profile profile);
[[nodiscard]] std::string to_text(const ExecutionPlan& plan);
[[nodiscard]] std::string to_json(const ExecutionPlan& plan);

}  // namespace vastloom::planner
