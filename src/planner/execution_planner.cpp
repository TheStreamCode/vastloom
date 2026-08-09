#include "stargate/planner/execution_planner.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>

#include "stargate/util/string.hpp"

namespace stargate::planner {
namespace {

struct ProfilePolicy {
    double ram_cap_ratio;
    double os_reserve_ratio;
    ByteCount minimum_os_reserve;
    double emergency_ratio;
    ByteCount minimum_emergency_reserve;
    double vram_reserve_ratio;
    ByteCount minimum_vram_reserve;
    double storage_reserve_ratio;
    ByteCount minimum_storage_reserve;
    ByteCount maximum_storage_reserve;
    double staging_ratio;
    double kv_ratio;
    unsigned max_io_workers;
    unsigned speculative_window;
};

ProfilePolicy policy_for(Profile profile) {
    switch (profile) {
        case Profile::safe:
            return {0.64, 0.25, 6U * GiB, 0.08, 2U * GiB, 0.16, 1536U * MiB,
                    0.06, 24U * GiB, 100U * GiB, 0.055, 0.12, 6, 2};
        case Profile::balanced:
            return {0.72, 0.20, 5U * GiB, 0.06, 2U * GiB, 0.13, 1280U * MiB,
                    0.05, 20U * GiB, 100U * GiB, 0.065, 0.14, 10, 4};
        case Profile::performance:
            return {0.80, 0.15, 4U * GiB, 0.05, 1536U * MiB, 0.10, 1024U * MiB,
                    0.04, 20U * GiB, 80U * GiB, 0.08, 0.16, 16, 5};
        case Profile::low_memory:
            return {0.56, 0.28, 6U * GiB, 0.09, 2U * GiB, 0.18, 1536U * MiB,
                    0.07, 24U * GiB, 120U * GiB, 0.05, 0.09, 5, 1};
        case Profile::quiet:
            return {0.68, 0.22, 5U * GiB, 0.07, 2U * GiB, 0.15, 1536U * MiB,
                    0.06, 24U * GiB, 100U * GiB, 0.055, 0.12, 4, 2};
    }
    return policy_for(Profile::balanced);
}

ByteCount clamp_bytes(ByteCount value, ByteCount minimum, ByteCount maximum) {
    return std::min(std::max(value, minimum), maximum);
}

const hardware::GpuInfo* select_gpu(const hardware::HardwareSnapshot& hardware) {
    if (hardware.gpus.empty()) {
        return nullptr;
    }
    return &*std::max_element(hardware.gpus.begin(), hardware.gpus.end(), [](const auto& left, const auto& right) {
        return left.free_vram_bytes < right.free_vram_bytes;
    });
}

ByteCount safe_subtract(ByteCount left, ByteCount right) {
    return left > right ? left - right : 0;
}

ByteCount existing_size_matched_bytes(const model::ModelManifest& manifest) {
    if (manifest.metadata_only || manifest.chunks.empty() || manifest.source_path.empty()) {
        return 0;
    }
    ByteCount present = 0;
    const auto root = manifest.source_path.parent_path().empty()
                          ? std::filesystem::path(".")
                          : manifest.source_path.parent_path();
    for (const auto& chunk : manifest.chunks) {
        std::error_code error;
        const auto path = root / chunk.relative_path;
        const auto size = std::filesystem::file_size(path, error);
        if (!error && size == chunk.size_bytes && present <= std::numeric_limits<ByteCount>::max() - chunk.size_bytes) {
            present += chunk.size_bytes;
        }
    }
    return present;
}

}  // namespace

Result<Profile> parse_profile(std::string_view value) {
    const auto normalized = util::to_lower(util::trim(value));
    if (normalized == "safe") return Profile::safe;
    if (normalized == "balanced") return Profile::balanced;
    if (normalized == "performance") return Profile::performance;
    if (normalized == "low-memory" || normalized == "low_memory") return Profile::low_memory;
    if (normalized == "quiet") return Profile::quiet;
    return make_error(ErrorCode::invalid_argument, "Unknown execution profile", normalized);
}

std::string_view to_string(Profile profile) noexcept {
    switch (profile) {
        case Profile::safe: return "safe";
        case Profile::balanced: return "balanced";
        case Profile::performance: return "performance";
        case Profile::low_memory: return "low-memory";
        case Profile::quiet: return "quiet";
    }
    return "balanced";
}

Result<ExecutionPlan> build_plan(const hardware::HardwareSnapshot& hardware,
                                 const model::ModelManifest& manifest,
                                 Profile profile) {
    if (hardware.memory.total_bytes == 0 || hardware.storage.capacity_bytes == 0) {
        return make_error(ErrorCode::invalid_argument, "Hardware snapshot is incomplete");
    }

    const ProfilePolicy policy = policy_for(profile);
    ExecutionPlan plan;
    plan.profile = profile;
    plan.model_id = manifest.model_id;

    plan.os_ram_reserve = std::max(policy.minimum_os_reserve,
                                   percentage(hardware.memory.total_bytes, policy.os_reserve_ratio));
    plan.emergency_ram_reserve = std::max(policy.minimum_emergency_reserve,
                                          percentage(hardware.memory.total_bytes, policy.emergency_ratio));

    const ByteCount physical_budget = safe_subtract(
        safe_subtract(hardware.memory.total_bytes, plan.os_ram_reserve), plan.emergency_ram_reserve);
    const ByteCount available_budget = safe_subtract(hardware.memory.available_bytes, plan.emergency_ram_reserve);
    const ByteCount cap_budget = percentage(hardware.memory.total_bytes, policy.ram_cap_ratio);
    plan.safe_ram_budget = std::min({physical_budget, available_budget, cap_budget});

    const hardware::GpuInfo* gpu = select_gpu(hardware);
    if (gpu != nullptr) {
        plan.selected_gpu = gpu->name;
        plan.vram_reserve = std::max(policy.minimum_vram_reserve,
                                     percentage(gpu->total_vram_bytes, policy.vram_reserve_ratio));
        const ByteCount total_vram_budget = safe_subtract(gpu->total_vram_bytes, plan.vram_reserve);
        const ByteCount free_vram_budget = safe_subtract(gpu->free_vram_bytes, plan.vram_reserve / 2U);
        plan.safe_vram_budget = std::min(total_vram_budget, free_vram_budget);
    }

    plan.storage_reserve = clamp_bytes(percentage(hardware.storage.capacity_bytes, policy.storage_reserve_ratio),
                                       policy.minimum_storage_reserve,
                                       policy.maximum_storage_reserve);
    const ByteCount existing_bytes = existing_size_matched_bytes(manifest);
    plan.additional_storage_required = safe_subtract(manifest.required_storage_bytes, existing_bytes);

    if (manifest.requires_avx2 && !hardware.cpu.avx2) {
        plan.blockers.emplace_back("The model adapter requires AVX2, but AVX2 was not detected.");
    }
    if (manifest.minimum_ram_bytes != 0 && hardware.memory.total_bytes < manifest.minimum_ram_bytes) {
        plan.blockers.emplace_back("Physical RAM is below the model's minimum requirement.");
    }
    if (plan.safe_ram_budget < 4U * GiB) {
        plan.blockers.emplace_back("Less than 4 GiB can be assigned safely after OS and emergency reserves.");
    }
    if (manifest.minimum_vram_bytes != 0 && (gpu == nullptr || gpu->total_vram_bytes < manifest.minimum_vram_bytes)) {
        plan.blockers.emplace_back("GPU VRAM is below the model's mandatory minimum.");
    }
    if (manifest.required_storage_bytes > hardware.storage.capacity_bytes) {
        plan.blockers.emplace_back("The model is larger than the selected storage device.");
    } else if (plan.additional_storage_required > safe_subtract(hardware.storage.available_bytes, plan.storage_reserve)) {
        plan.blockers.emplace_back("Insufficient free storage for missing model bytes after preserving Stargate's safety reserve.");
    }

    if (hardware.memory.available_bytes < hardware.memory.total_bytes / 3U) {
        plan.warnings.emplace_back("Current RAM availability is low relative to installed RAM; close memory-heavy applications before inference.");
    }
    if (gpu == nullptr) {
        plan.warnings.emplace_back("No GPU backend was detected. The current plan is CPU/storage-oriented and will not meet the intended interactive target.");
    } else if (manifest.recommended_vram_bytes != 0 && gpu->total_vram_bytes < manifest.recommended_vram_bytes) {
        plan.warnings.emplace_back("VRAM is below the model's recommended value; more experts will remain in RAM/NVMe tiers.");
    }
    if (manifest.recommended_ram_bytes != 0 && hardware.memory.total_bytes < manifest.recommended_ram_bytes) {
        plan.warnings.emplace_back("RAM is below the model's recommended value; predictive streaming is required.");
    }

    if (plan.safe_ram_budget != 0) {
        plan.staging_ram_bytes = clamp_bytes(percentage(plan.safe_ram_budget, policy.staging_ratio),
                                             512U * MiB,
                                             std::min<ByteCount>(2U * GiB, plan.safe_ram_budget / 4U));
        plan.kv_ram_bytes = clamp_bytes(percentage(plan.safe_ram_budget, policy.kv_ratio),
                                        512U * MiB,
                                        std::min<ByteCount>(4U * GiB, plan.safe_ram_budget / 3U));
        ByteCount remaining = safe_subtract(plan.safe_ram_budget, plan.staging_ram_bytes + plan.kv_ram_bytes);
        if (manifest.dense_bytes != 0) {
            plan.dense_ram_bytes = std::min(manifest.dense_bytes, percentage(remaining, 0.80));
        } else {
            plan.dense_ram_bytes = percentage(remaining, 0.55);
        }
        remaining = safe_subtract(remaining, plan.dense_ram_bytes);
        plan.expert_ram_bytes = remaining;
    }

    if (plan.safe_vram_budget != 0) {
        plan.gpu_staging_bytes = clamp_bytes(percentage(plan.safe_vram_budget, 0.06),
                                             256U * MiB,
                                             std::min<ByteCount>(768U * MiB, plan.safe_vram_budget / 5U));
        plan.gpu_kv_bytes = clamp_bytes(percentage(plan.safe_vram_budget, 0.16),
                                        512U * MiB,
                                        std::min<ByteCount>(3U * GiB, plan.safe_vram_budget / 3U));
        plan.gpu_resident_bytes = safe_subtract(plan.safe_vram_budget,
                                                plan.gpu_staging_bytes + plan.gpu_kv_bytes);
    }

    plan.nvme_backing_bytes = manifest.model_total_bytes;
    const ByteCount fast_memory = plan.safe_ram_budget + plan.safe_vram_budget;
    if (manifest.model_total_bytes != 0 && manifest.model_total_bytes <= fast_memory) {
        plan.strategy = "resident";
    } else if (manifest.dense_bytes == 0 || manifest.dense_bytes <= plan.dense_ram_bytes + plan.gpu_resident_bytes) {
        plan.strategy = "multitier-predictive";
    } else {
        plan.strategy = "extreme-streaming";
        plan.warnings.emplace_back("The dense working set cannot be fully resident; token latency may be dominated by storage.");
    }

    const unsigned half_cores = std::max(1U, hardware.cpu.logical_cores / 2U);
    plan.io_workers = std::min(half_cores, policy.max_io_workers);
    if (profile == Profile::quiet) {
        plan.io_workers = std::min(plan.io_workers, 4U);
    }

    const ByteCount expert_size = manifest.estimated_expert_bytes == 0 ? 64U * MiB : manifest.estimated_expert_bytes;
    const ByteCount staging_slots = expert_size == 0 ? 1 : plan.staging_ram_bytes / expert_size;
    plan.prefetch_depth = static_cast<unsigned>(std::clamp<ByteCount>(staging_slots / 2U, 1U, 16U));
    plan.speculative_window = manifest.supports_speculation ? policy.speculative_window : 0U;

    plan.governor_limits.ram_hard_limit = plan.safe_ram_budget;
    plan.governor_limits.vram_hard_limit = plan.safe_vram_budget;
    plan.governor_limits.staging_hard_limit = plan.staging_ram_bytes;
    plan.governor_limits.scratch_hard_limit = std::min<ByteCount>(4U * GiB, safe_subtract(hardware.storage.available_bytes, plan.storage_reserve));
    plan.governor_limits.minimum_system_ram_available = plan.emergency_ram_reserve;
    plan.governor_limits.minimum_storage_available = plan.storage_reserve;
    plan.governor_limits.pressure_ratio = profile == Profile::performance ? 0.86 : 0.82;
    plan.governor_limits.critical_ratio = profile == Profile::performance ? 0.95 : 0.92;
    plan.governor_limits.recovery_samples = 3;

    plan.compatible = plan.blockers.empty();
    return plan;
}

std::string to_text(const ExecutionPlan& plan) {
    std::ostringstream output;
    output << "Stargate execution plan\n"
           << "Model:       " << plan.model_id << '\n'
           << "Profile:     " << to_string(plan.profile) << '\n'
           << "Compatible:  " << (plan.compatible ? "yes" : "no") << '\n'
           << "Strategy:    " << plan.strategy << '\n'
           << "GPU:         " << (plan.selected_gpu.empty() ? "none" : plan.selected_gpu) << '\n'
           << "\nSafety budgets\n"
           << "  OS reserve:        " << format_bytes(plan.os_ram_reserve) << '\n'
           << "  Emergency reserve: " << format_bytes(plan.emergency_ram_reserve) << '\n'
           << "  Stargate RAM max:  " << format_bytes(plan.safe_ram_budget) << '\n'
           << "  VRAM reserve:      " << format_bytes(plan.vram_reserve) << '\n'
           << "  Stargate VRAM max: " << format_bytes(plan.safe_vram_budget) << '\n'
           << "  Storage reserve:   " << format_bytes(plan.storage_reserve) << '\n'
           << "  Storage to add:    " << format_bytes(plan.additional_storage_required) << '\n'
           << "\nPlacement\n"
           << "  Dense in RAM:      " << format_bytes(plan.dense_ram_bytes) << '\n'
           << "  Expert RAM cache:  " << format_bytes(plan.expert_ram_bytes) << '\n'
           << "  KV in RAM:         " << format_bytes(plan.kv_ram_bytes) << '\n'
           << "  RAM staging:       " << format_bytes(plan.staging_ram_bytes) << '\n'
           << "  GPU resident:      " << format_bytes(plan.gpu_resident_bytes) << '\n'
           << "  GPU KV:            " << format_bytes(plan.gpu_kv_bytes) << '\n'
           << "  GPU staging:       " << format_bytes(plan.gpu_staging_bytes) << '\n'
           << "  NVMe backing:      " << format_bytes(plan.nvme_backing_bytes) << '\n'
           << "\nScheduler\n"
           << "  I/O workers:       " << plan.io_workers << '\n'
           << "  Prefetch depth:    " << plan.prefetch_depth << '\n'
           << "  Speculative window:" << ' ' << plan.speculative_window << '\n';
    for (const auto& blocker : plan.blockers) {
        output << "BLOCKER:     " << blocker << '\n';
    }
    for (const auto& warning : plan.warnings) {
        output << "WARNING:     " << warning << '\n';
    }
    return output.str();
}

std::string to_json(const ExecutionPlan& plan) {
    std::ostringstream output;
    output << '{'
           << "\"model_id\":\"" << util::json_escape(plan.model_id) << "\"," 
           << "\"profile\":\"" << to_string(plan.profile) << "\"," 
           << "\"compatible\":" << (plan.compatible ? "true" : "false") << ','
           << "\"strategy\":\"" << util::json_escape(plan.strategy) << "\"," 
           << "\"selected_gpu\":\"" << util::json_escape(plan.selected_gpu) << "\"," 
           << "\"budgets\":{"
           << "\"os_ram_reserve\":" << plan.os_ram_reserve << ','
           << "\"emergency_ram_reserve\":" << plan.emergency_ram_reserve << ','
           << "\"safe_ram\":" << plan.safe_ram_budget << ','
           << "\"vram_reserve\":" << plan.vram_reserve << ','
           << "\"safe_vram\":" << plan.safe_vram_budget << ','
           << "\"storage_reserve\":" << plan.storage_reserve << ','
           << "\"additional_storage_required\":" << plan.additional_storage_required << "},"
           << "\"placement\":{"
           << "\"dense_ram\":" << plan.dense_ram_bytes << ','
           << "\"expert_ram\":" << plan.expert_ram_bytes << ','
           << "\"kv_ram\":" << plan.kv_ram_bytes << ','
           << "\"staging_ram\":" << plan.staging_ram_bytes << ','
           << "\"gpu_resident\":" << plan.gpu_resident_bytes << ','
           << "\"gpu_kv\":" << plan.gpu_kv_bytes << ','
           << "\"gpu_staging\":" << plan.gpu_staging_bytes << ','
           << "\"nvme_backing\":" << plan.nvme_backing_bytes << "},"
           << "\"scheduler\":{"
           << "\"io_workers\":" << plan.io_workers << ','
           << "\"prefetch_depth\":" << plan.prefetch_depth << ','
           << "\"speculative_window\":" << plan.speculative_window << "},"
           << "\"blockers\":[";
    for (std::size_t index = 0; index < plan.blockers.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << util::json_escape(plan.blockers[index]) << '"';
    }
    output << "],\"warnings\":[";
    for (std::size_t index = 0; index < plan.warnings.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << util::json_escape(plan.warnings[index]) << '"';
    }
    output << "]}";
    return output.str();
}

}  // namespace stargate::planner
