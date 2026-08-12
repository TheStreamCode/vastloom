#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/cli/tiny_commands.hpp"
#include "vastloom/core/bytes.hpp"
#include "vastloom/cuda/cuda_driver.hpp"
#include "vastloom/cuda/cuda_rms_norm.hpp"
#include "vastloom/hardware/hardware_probe.hpp"
#include "vastloom/inference/gpt2_tokenizer.hpp"
#include "vastloom/cuda/cuda_matvec_accelerator.hpp"
#include "vastloom/inference/deepseek_v4_expert_provider.hpp"
#include "vastloom/inference/deepseek_v4_gguf.hpp"
#include "vastloom/inference/deepseek_v4_range_artifact.hpp"
#include "vastloom/inference/deepseek_v4_model.hpp"
#include "vastloom/inference/deepseek_v4_reference.hpp"
#include "vastloom/inference/llama_model.hpp"
#include "vastloom/inference/olmoe_model.hpp"
#include "vastloom/inference/olmoe_runtime.hpp"
#include "vastloom/io/bounded_io_pool.hpp"
#include "vastloom/model/manifest.hpp"
#include "vastloom/model/imported_checkpoint.hpp"
#include "vastloom/model/gguf.hpp"
#include "vastloom/model/model_store.hpp"
#include "vastloom/model/range_artifact.hpp"
#include "vastloom/model/safetensors.hpp"
#include "vastloom/model/verifier.hpp"
#include "vastloom/planner/execution_planner.hpp"
#include "vastloom/research/trace_simulator.hpp"
#include "vastloom/safety/heartbeat.hpp"
#include "vastloom/safety/resource_governor.hpp"
#include "vastloom/safety/runtime_policy.hpp"
#include "vastloom/util/string.hpp"
#include "vastloom/version.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

using vastloom::ByteCount;
using vastloom::Error;
using vastloom::Result;

volatile std::sig_atomic_t real_generation_cancelled = 0;

void handle_real_interrupt(int) {
    real_generation_cancelled = 1;
}

int exit_code(const Error& error) {
    using vastloom::ErrorCode;
    switch (error.code) {
        case ErrorCode::invalid_argument:
        case ErrorCode::invalid_format: return 64;
        case ErrorCode::not_found: return 66;
        case ErrorCode::permission_denied: return 77;
        case ErrorCode::integrity_error:
        case ErrorCode::unsafe_path: return 65;
        case ErrorCode::resource_exhausted: return 75;
        case ErrorCode::unsupported: return 69;
        case ErrorCode::io_error:
        case ErrorCode::internal_error: return 74;
    }
    return 1;
}

int print_error(const Error& error, bool json = false) {
    if (json) {
        std::cerr << "{\"ok\":false,\"error\":\"" << vastloom::util::json_escape(error.message)
                  << "\",\"context\":\"" << vastloom::util::json_escape(error.context) << "\"}\n";
    } else {
        std::cerr << "ERROR: " << error.message;
        if (!error.context.empty()) std::cerr << " [" << error.context << ']';
        std::cerr << '\n';
    }
    return exit_code(error);
}

bool has_flag(const std::vector<std::string>& args, std::string_view flag) {
    return std::any_of(args.begin(), args.end(), [flag](const std::string& argument) {
        return std::string_view(argument) == flag;
    });
}

std::filesystem::path cli_path(std::string_view utf8) {
    std::u8string encoded;
    encoded.reserve(utf8.size());
    for (const char byte : utf8) {
        encoded.push_back(static_cast<char8_t>(static_cast<unsigned char>(byte)));
    }
    return std::filesystem::path(encoded);
}

std::optional<std::string> option_value(const std::vector<std::string>& args, std::string_view name) {
    for (std::size_t index = 0; index < args.size(); ++index) {
        if (args[index] == name && index + 1U < args.size()) return args[index + 1U];
    }
    return std::nullopt;
}

std::filesystem::path option_path(const std::vector<std::string>& args,
                                  std::string_view name,
                                  const std::filesystem::path& fallback) {
    const auto value = option_value(args, name);
    return value.has_value() ? cli_path(*value) : fallback;
}

Result<ByteCount> optional_bytes(const std::vector<std::string>& args, std::string_view name, ByteCount fallback) {
    const auto value = option_value(args, name);
    if (!value.has_value()) return fallback;
    return vastloom::parse_bytes(*value);
}

Result<std::size_t> optional_size(const std::vector<std::string>& args, std::string_view name, std::size_t fallback) {
    const auto value = option_value(args, name);
    if (!value) return fallback;
    std::size_t parsed = 0;
    const auto converted = std::from_chars(value->data(), value->data() + value->size(), parsed);
    if (converted.ec != std::errc{} || converted.ptr != value->data() + value->size()) {
        return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Invalid unsigned integer option", std::string(name));
    }
    return parsed;
}

Result<std::uint64_t> optional_u64(const std::vector<std::string>& args, std::string_view name, std::uint64_t fallback) {
    const auto value = option_value(args, name);
    if (!value) return fallback;
    std::uint64_t parsed = 0;
    const auto converted = std::from_chars(value->data(), value->data() + value->size(), parsed);
    if (converted.ec != std::errc{} || converted.ptr != value->data() + value->size()) {
        return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Invalid uint64 option", std::string(name));
    }
    return parsed;
}

Result<float> optional_float(const std::vector<std::string>& args, std::string_view name, float fallback) {
    const auto value = option_value(args, name);
    if (!value) return fallback;
    float parsed = 0.0F;
    const auto converted = std::from_chars(value->data(), value->data() + value->size(), parsed);
    if (converted.ec != std::errc{} || converted.ptr != value->data() + value->size() || !std::isfinite(parsed)) {
        return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Invalid floating-point option", std::string(name));
    }
    return parsed;
}

Result<vastloom::io::IoFaultInjection> parse_io_fault(std::string_view value) {
    const auto normalized = vastloom::util::to_lower(vastloom::util::trim(value));
    vastloom::io::IoFaultInjection result;
    result.request_sequence = 1U;
    if (normalized.empty() || normalized == "none") result.kind = vastloom::io::IoFaultKind::none;
    else if (normalized == "fail") result.kind = vastloom::io::IoFaultKind::fail;
    else if (normalized == "short-read") result.kind = vastloom::io::IoFaultKind::short_read;
    else if (normalized == "allocation") result.kind = vastloom::io::IoFaultKind::allocation;
    else if (normalized == "cancelled") result.kind = vastloom::io::IoFaultKind::cancelled;
    else if (normalized == "timeout") result.kind = vastloom::io::IoFaultKind::timeout;
    else return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                     "Unknown OLMoE I/O fault injection mode", std::string(value));
    return result;
}

Result<void> validate_options(const std::vector<std::string>& args,
                              std::size_t positional_count,
                              std::initializer_list<std::string_view> value_options,
                              std::initializer_list<std::string_view> flags) {
    if (args.size() < positional_count) return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Missing positional argument");
    std::set<std::string> seen;
    for (std::size_t index = positional_count; index < args.size(); ++index) {
        const std::string_view argument = args[index];
        const bool is_value = std::find(value_options.begin(), value_options.end(), argument) != value_options.end();
        const bool is_flag = std::find(flags.begin(), flags.end(), argument) != flags.end();
        if (!is_value && !is_flag) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Unknown or unexpected option", std::string(argument));
        }
        if (!seen.insert(std::string(argument)).second) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Duplicate option", std::string(argument));
        }
        if (is_value) {
            if (index + 1U >= args.size()) return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Option requires a value", std::string(argument));
            ++index;
        }
    }
    return {};
}

std::uint64_t current_pid() {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(_getpid());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

void print_help() {
    std::cout
        << "Vastloom Local AI " << vastloom::kVersion << "\n"
        << "Safety-first local inference research runtime for oversized MoE models.\n\n"
        << "Commands:\n"
        << "  chat MODEL_OR_MANIFEST [--prompt TEXT] [--max-tokens N] [--temperature F]\n"
        << "       [--top-k N] [--seed N] [--timeout-ms N] [--system TEXT] [--no-stream]\n"
        << "       [--no-stats] [--no-verify] [--json]\n"
        << "      Start interactive chat or run one prompt with TinyChat or pinned SmolLM2.\n\n"
        << "  model-info MODEL_OR_MANIFEST [--no-verify] [--json]\n"
        << "      Inspect a runnable TinyChat or imported SmolLM2 checkpoint.\n\n"
        << "  self-test MODEL_OR_MANIFEST [--no-verify] [--json]\n"
        << "      Run adapter-specific deterministic end-to-end generation probes.\n\n"
        << "  doctor [--storage PATH] [--json]\n"
        << "      Detect CPU, RAM, NVIDIA VRAM and storage without allocating model memory.\n\n"
        << "  cuda-info [--allocation-smoke SIZE|--rmsnorm-smoke]"
           " [--minimum-free-vram SIZE] [--json]\n"
        << "      Probe CUDA; optionally run a governed allocation or CPU-verified RMSNorm.\n\n"
        << "  inspect MANIFEST_OR_MODEL_HEADERS [--declared-size SIZE] [--tensor-limit N] [--json]\n"
        << "      Inspect a manifest, bounded Safetensors checkpoint or GGUF v3 header.\n\n"
        << "  verify MANIFEST [--model-root PATH] [--full] [--json]\n"
        << "      Verify chunk paths and sizes; --full also checks SHA-256.\n\n"
        << "  plan MANIFEST [--profile safe|balanced|performance|low-memory|quiet]\n"
        << "       [--storage PATH] [--reference-pc] [--simulate-ram SIZE]\n"
        << "       [--simulate-vram SIZE] [--simulate-storage SIZE] [--simulate-free SIZE] [--json]\n"
        << "      Build a fail-closed VRAM/RAM/NVMe execution plan.\n\n"
        << "  trace-sim TRACE.csv --vram SIZE --ram SIZE [--policy no-cache|lru|lfu|static-hot] [--json]\n"
        << "      Measure expert-cache hit rates, NVMe bytes/token and PCIe traffic.\n\n"
        << "  io-smoke FILE [--offset BYTES] [--size SIZE]\n"
        << "      Exercise the bounded asynchronous I/O path under the resource governor.\n\n"
        << "  store-init PATH [--reserve SIZE] [--json]\n"
        << "      Create the immutable-model/cache/scratch/state store with a disk reserve.\n\n"
        << "  store-info PATH [--reserve SIZE] [--json]\n"
        << "      Validate and display an existing Vastloom model store.\n\n"
        << "  runtime-policy [--storage PATH] [--json]\n"
        << "      Show the fail-closed RAM/VRAM/CPU/I/O policy for this host.\n\n"
        << "  tokenize MODEL_DIRECTORY --text TEXT [--chat] [--system TEXT] [--json]\n"
        << "      Exercise a supported byte-BPE tokenizer and its native chat template.\n\n"
        << "  real-logits MODEL_DIRECTORY --text TEXT [--chat] [--top N] [--no-verify] [--json]\n"
        << "      Run the pinned SmolLM2 CPU reference path and report final-token logits.\n\n"
        << "  moe-inspect MODEL_DIRECTORY [--no-verify] [--json]\n"
        << "      Validate the pinned OLMoE metadata and estimate safe streaming budgets.\n\n"
        << "  deepseek-inspect MODEL_DIRECTORY [--no-verify] [--json]\n"
        << "      Validate pinned DeepSeek V4 Flash metadata and Safetensors header snapshots.\n\n"
        << "  deepseek-gguf-inspect SHARD... [--declared-sizes CSV] [--revision ID]"
           " [--context N] [--batch-size N] [--json]\n"
        << "      Build a metadata-only catalog and derived resident/streamed working-set plan.\n\n"
        << "  deepseek-gguf-checkpoint MODEL_DIRECTORY [--context N] [--batch-size N]"
           " [--no-verify] [--json]\n"
        << "      Verify and bind the pinned 0731 UD-IQ1_S import under single-tier admission.\n\n"
        << "  deepseek-gguf-probe-plan MODEL_DIRECTORY --layer N"
           " [--experts CSV | --route-artifact DIR --token ID]"
           " [--tile-size SIZE] [--context N] [--batch-size N] [--no-verify] [--json]\n"
        << "      Plan exact trunk and selected-expert ranges; never fetch or execute them.\n"
        << "      --route-artifact reads the pinned hash-routing table from an imported trunk and\n"
        << "      lets the checkpoint name the experts, which caller-supplied IDs cannot prove.\n\n"
        << "  deepseek-gguf-global-plan MODEL_DIRECTORY [--context N] [--batch-size N]"
           " [--no-verify] [--json]\n"
        << "      Plan the exact six checkpoint-global tensors; never fetch or execute them.\n\n"
        << "  range-artifact-inspect ARTIFACT_DIRECTORY [--no-verify] [--json]\n"
        << "      Validate a bounded immutable range artifact; never promotes it to execution.\n\n"
        << "  deepseek-forward MODEL_DIRECTORY [--layers N] [--token ID] [--expert-cache N]"
           " [--dry-run] [--json]\n"
        << "      Report exactly what a forward pass to that depth would make resident and whether\n"
        << "      it fits the safe RAM budget, before allocating anything.\n\n"
        << "  deepseek-layer-probe MODEL_DIRECTORY TRUNK_ARTIFACT EXPERT_ARTIFACT GLOBAL_ARTIFACT"
           " [--token ID] [--json]\n"
        << "      Execute one real layer from imported payload. This is one layer of 43, not the\n"
        << "      model's output; the logits it prints are the first layer's alone.\n\n"
        << "  deepseek-gguf-range-bind MODEL_DIRECTORY ARTIFACT_DIRECTORY --layer N --experts CSV"
           " [--tile-size SIZE] [--context N] [--batch-size N] [--json]\n"
        << "      Rehash and bind a trunk artifact to pinned DeepSeek metadata; still non-executing.\n\n"
        << "  deepseek-gguf-global-bind MODEL_DIRECTORY ARTIFACT_DIRECTORY"
           " [--context N] [--batch-size N] [--json]\n"
        << "      Rehash and bind a six-range globals artifact to pinned metadata.\n\n"
        << "  moe-logits MODEL_DIRECTORY --text TEXT [--chat] [--top N]\n"
        << "       [--cache no-cache|lru|lfu|static-hot] [--cache-bytes SIZE]\n"
        << "       [--io sync|async] [--io-cache buffered|bypass] [--expert-read whole|staged]\n"
        << "       [--prefetch-depth 1|2]\n"
        << "       [--io-timeout-ms N]\n"
        << "       [--timeout-ms N] [--trace PATH] [--no-verify] [--json]\n"
        << "      Run the OLMoE CPU oracle with bounded NVMe expert streaming.\n\n"
        << "  heartbeat-write PATH [--state normal|pressure|critical|emergency]\n"
        << "      Atomically write one watchdog heartbeat for integration tests.\n\n"
        << "  version\n";
}

vastloom::hardware::HardwareSnapshot reference_pc_snapshot(const std::filesystem::path& storage_path) {
    vastloom::hardware::HardwareSnapshot hardware;
    hardware.operating_system = "Windows 11 (reference simulation)";
    hardware.cpu.name = "AMD Ryzen 9 7900X";
    hardware.cpu.logical_cores = 24;
    hardware.cpu.avx2 = true;
    hardware.cpu.fma = true;
    hardware.memory.total_bytes = 32U * vastloom::GiB;
    hardware.memory.available_bytes = 26U * vastloom::GiB;
    hardware.gpus.push_back(vastloom::hardware::GpuInfo{
        "CUDA", "NVIDIA GeForce RTX 4070 SUPER", "simulated", 12U * vastloom::GiB,
        11U * vastloom::GiB, 60.0, 220.0});
    hardware.storage.path = storage_path;
    hardware.storage.capacity_bytes = 2U * vastloom::TiB;
    hardware.storage.free_bytes = 1536U * vastloom::GiB;
    hardware.storage.available_bytes = 1536U * vastloom::GiB;
    return hardware;
}

Result<vastloom::safety::SafetyState> parse_safety_state(std::string_view value) {
    const auto normalized = vastloom::util::to_lower(vastloom::util::trim(value));
    if (normalized == "normal") return vastloom::safety::SafetyState::normal;
    if (normalized == "pressure") return vastloom::safety::SafetyState::pressure;
    if (normalized == "critical") return vastloom::safety::SafetyState::critical;
    if (normalized == "emergency") return vastloom::safety::SafetyState::emergency;
    return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Unknown safety state", normalized);
}

int command_doctor(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    const auto storage = option_path(args, "--storage", std::filesystem::current_path());
    auto hardware = vastloom::hardware::HardwareProbe::collect(storage);
    if (!hardware) return print_error(hardware.error(), json);
    std::cout << (json ? vastloom::hardware::to_json(hardware.value()) : vastloom::hardware::to_text(hardware.value())) << '\n';
    return 0;
}

int command_cuda_info(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (has_flag(args, "--help")) {
        std::cout
            << "Usage: vastloom cuda-info [--allocation-smoke SIZE|--rmsnorm-smoke]"
               " [--minimum-free-vram SIZE] [--json]\n"
            << "Loads the CUDA Driver API dynamically; no CUDA toolkit is required. The optional\n"
            << "allocation is limited to 64 MiB, reserves Governor VRAM first and preserves at\n"
            << "least 1 GiB of live device headroom unless a larger reserve is requested.\n"
            << "--rmsnorm-smoke transfers two 4096-wide FP32 rows, launches native PTX, copies\n"
            << "the output back and compares it with the bounded CPU reference.\n";
        return 0;
    }
    auto valid = validate_options(args, 0U,
                                  {"--allocation-smoke", "--minimum-free-vram"},
                                  {"--json", "--help", "--rmsnorm-smoke", "--matvec-smoke",
                                   "--iq1s-smoke"});
    if (!valid) return print_error(valid.error(), json);
    const bool allocation_smoke_requested =
        option_value(args, "--allocation-smoke").has_value();
    const bool rmsnorm_smoke_requested = has_flag(args, "--rmsnorm-smoke");
    const bool matvec_smoke_requested = has_flag(args, "--matvec-smoke");
    const bool iq1s_smoke_requested = has_flag(args, "--iq1s-smoke");
    if ((allocation_smoke_requested || rmsnorm_smoke_requested) &&
        (matvec_smoke_requested || iq1s_smoke_requested)) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "--matvec-smoke and --iq1s-smoke run on their own"),
            json);
    }
    if (allocation_smoke_requested && rmsnorm_smoke_requested) {
        return print_error(vastloom::make_error(
                               vastloom::ErrorCode::invalid_argument,
                               "--allocation-smoke and --rmsnorm-smoke are mutually exclusive"),
                           json);
    }
    const bool operation_requested = allocation_smoke_requested || rmsnorm_smoke_requested ||
                                     matvec_smoke_requested || iq1s_smoke_requested;
    if (!operation_requested && option_value(args, "--minimum-free-vram").has_value()) {
        return print_error(vastloom::make_error(
                               vastloom::ErrorCode::invalid_argument,
                               "--minimum-free-vram requires a CUDA smoke operation"),
                           json);
    }
    auto smoke_bytes = optional_bytes(args, "--allocation-smoke", 0U);
    auto minimum_free = optional_bytes(args, "--minimum-free-vram", 1U * vastloom::GiB);
    if (!smoke_bytes) return print_error(smoke_bytes.error(), json);
    if (!minimum_free) return print_error(minimum_free.error(), json);
    if (allocation_smoke_requested &&
        (smoke_bytes.value() == 0U || smoke_bytes.value() > 64U * vastloom::MiB)) {
        return print_error(vastloom::make_error(
                               vastloom::ErrorCode::invalid_argument,
                               "--allocation-smoke must be between 1 byte and 64 MiB"),
                           json);
    }

    auto driver = vastloom::cuda::CudaDriver::load();
    if (!driver) return print_error(driver.error(), json);
    auto probe = driver.value().probe();
    if (!probe) return print_error(probe.error(), json);

    bool allocation_smoke_passed = false;
    bool rmsnorm_smoke_passed = false;
    double rmsnorm_max_abs_error = 0.0;
    vastloom::cuda::CudaRmsNormStats rmsnorm_stats;
    bool matvec_smoke_passed = false;
    double matvec_max_abs_error = 0.0;
    double matvec_kernel_ms = 0.0;
    double matvec_cpu_ms = 0.0;
    bool iq1s_smoke_passed = false;
    double iq1s_max_abs_error = 0.0;
    double iq1s_kernel_ms = 0.0;
    double iq1s_cpu_ms = 0.0;
    std::size_t matvec_rows_executed = 0U;
    std::size_t matvec_columns_executed = 0U;
    ByteCount matvec_device_bytes = 0U;
    if (operation_requested) {
        if (probe.value().devices.empty()) {
            return print_error(vastloom::make_error(
                                   vastloom::ErrorCode::not_found,
                                   "CUDA smoke operation requires device ordinal 0"),
                               json);
        }
        auto hardware = vastloom::hardware::HardwareProbe::collect(
            std::filesystem::current_path());
        if (!hardware) return print_error(hardware.error(), json);
        auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
        if (!policy) return print_error(policy.error(), json);
        if (policy.value().resources.vram_hard_limit == 0U) {
            return print_error(vastloom::make_error(
                                   vastloom::ErrorCode::resource_exhausted,
                                   "Runtime policy exposes no safe VRAM allocation budget"),
                               json);
        }
        auto process_guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
        if (!process_guard) return print_error(process_guard.error(), json);
        auto safe = process_guard.value().sample_and_check();
        if (!safe) return print_error(safe.error(), json);
        vastloom::safety::ResourceGovernor governor(policy.value().resources);
        if (allocation_smoke_requested) {
            auto allocation = driver.value().allocate(
                0, smoke_bytes.value(), minimum_free.value(), governor, "cuda-info-smoke");
            if (!allocation) return print_error(allocation.error(), json);
            auto released = allocation.value().release();
            if (!released) return print_error(released.error(), json);
            allocation_smoke_passed = true;
        } else {
            constexpr std::size_t rows = 2U;
            constexpr std::size_t width = 4'096U;
            constexpr float epsilon = 1.0e-6F;
            constexpr double tolerance = 3.0e-5;
            constexpr std::size_t elements = rows * width;
            constexpr ByteCount matrix_bytes = elements * sizeof(float);
            constexpr ByteCount weight_bytes = width * sizeof(float);
            constexpr ByteCount caller_host_bytes = 2U * matrix_bytes + weight_bytes;
            auto caller_host = governor.try_reserve(
                vastloom::safety::ResourceKind::scratch, caller_host_bytes,
                "cuda-rmsnorm-smoke-input-reference");
            if (!caller_host) return print_error(caller_host.error(), json);
            std::vector<float> input(elements);
            std::vector<float> weight(width);
            for (std::size_t index = 0U; index < input.size(); ++index) {
                input[index] = static_cast<float>(
                    static_cast<int>(index % 257U) - 128) / 64.0F;
            }
            for (std::size_t index = 0U; index < weight.size(); ++index) {
                weight[index] = 0.75F + static_cast<float>(index % 17U) / 32.0F;
            }
            auto expected = vastloom::inference::deepseek_v4_rms_norm(
                input, weight, rows, width, epsilon);
            if (!expected) return print_error(expected.error(), json);
            auto actual = vastloom::cuda::cuda_rms_norm(
                driver.value(), input, weight, rows, width, epsilon, governor,
                {.device_ordinal = 0,
                 .minimum_free_vram_after = minimum_free.value()});
            if (!actual) return print_error(actual.error(), json);
            if (actual.value().output.size() != expected.value().size()) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA RMSNorm returned an invalid output size"),
                                   json);
            }
            for (std::size_t index = 0U; index < actual.value().output.size(); ++index) {
                rmsnorm_max_abs_error = std::max(
                    rmsnorm_max_abs_error,
                    std::abs(static_cast<double>(actual.value().output[index]) -
                             static_cast<double>(expected.value()[index])));
            }
            if (rmsnorm_max_abs_error > tolerance) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA RMSNorm does not match the CPU reference",
                                       std::to_string(rmsnorm_max_abs_error)),
                                   json);
            }
            rmsnorm_stats = actual.value().stats;
            rmsnorm_smoke_passed = true;
        }
        if (matvec_smoke_requested) {
            // The first quantized kernel on the GPU, checked against the CPU matvec that is itself
            // validated bit-for-bit against pinned ggml. The GPU accumulates in FP32 where the
            // reference uses FP64, so this is a tolerance gate, not a parity gate.
            constexpr std::size_t matvec_rows = 4'096U;
            constexpr std::size_t matvec_columns = 4'096U;
            constexpr std::size_t block_elements = 32U;
            constexpr std::size_t block_bytes = 34U;
            constexpr double matvec_tolerance = 2.0e-2;
            const std::size_t blocks_per_row = matvec_columns / block_elements;
            const ByteCount matrix_bytes =
                static_cast<ByteCount>(matvec_rows) * blocks_per_row * block_bytes;
            const ByteCount input_bytes =
                static_cast<ByteCount>(matvec_columns) * sizeof(float);
            const ByteCount output_bytes = static_cast<ByteCount>(matvec_rows) * sizeof(float);

            auto caller_host = governor.try_reserve(
                vastloom::safety::ResourceKind::scratch,
                matrix_bytes + input_bytes + 2U * output_bytes, "cuda-matvec-smoke-host");
            if (!caller_host) return print_error(caller_host.error(), json);

            std::vector<float> dense(matvec_rows * matvec_columns);
            std::vector<float> input(matvec_columns);
            for (std::size_t index = 0U; index < dense.size(); ++index) {
                dense[index] = static_cast<float>(static_cast<int>(index % 251U) - 125) / 512.0F;
            }
            for (std::size_t index = 0U; index < input.size(); ++index) {
                input[index] = static_cast<float>(static_cast<int>(index % 199U) - 99) / 256.0F;
            }
            auto encoded = vastloom::inference::quantize_q8_0_reference(
                dense, matvec_rows, matvec_columns, matvec_columns);
            if (!encoded) return print_error(encoded.error(), json);
            if (encoded.value().size() != matrix_bytes) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "Q8_0 smoke encoding has an unexpected size"),
                                   json);
            }

            std::vector<float> expected(matvec_rows);
            vastloom::inference::QuantizedKernelLimits kernel_limits;
            kernel_limits.max_elements = matvec_rows * matvec_columns;
            kernel_limits.max_threads = policy.value().cpu_threads;
            const auto cpu_started = std::chrono::steady_clock::now();
            auto cpu = vastloom::inference::matvec_q8_0_f32(
                encoded.value(), matvec_rows, matvec_columns, input, expected, kernel_limits);
            matvec_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cpu_started).count();
            if (!cpu) return print_error(cpu.error(), json);

            const ByteCount input_offset = matrix_bytes;
            const ByteCount output_offset = input_offset + input_bytes;
            auto device = driver.value().allocate(
                0, matrix_bytes + input_bytes + output_bytes, minimum_free.value(), governor,
                "cuda-matvec-q8-0-smoke");
            if (!device) return print_error(device.error(), json);
            auto allocation = device.take_value();
            auto copied_matrix = driver.value().copy_from_host(allocation, 0U, encoded.value());
            if (!copied_matrix) return print_error(copied_matrix.error(), json);
            auto copied_input = driver.value().copy_from_host(
                allocation, input_offset, std::as_bytes(std::span<const float>(input)));
            if (!copied_input) return print_error(copied_input.error(), json);
            const auto started = std::chrono::steady_clock::now();
            auto launched = driver.value().launch_matvec_q8_0(
                allocation, 0U, input_offset, output_offset, matvec_rows, blocks_per_row);
            if (!launched) return print_error(launched.error(), json);
            matvec_kernel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            std::vector<float> actual_output(matvec_rows);
            auto copied_output = driver.value().copy_to_host(
                allocation, output_offset,
                std::as_writable_bytes(std::span<float>(actual_output)));
            if (!copied_output) return print_error(copied_output.error(), json);
            auto released = allocation.release();
            if (!released) return print_error(released.error(), json);

            for (std::size_t index = 0U; index < actual_output.size(); ++index) {
                if (!std::isfinite(actual_output[index])) {
                    return print_error(vastloom::make_error(
                                           vastloom::ErrorCode::integrity_error,
                                           "CUDA Q8_0 matvec produced a non-finite value"),
                                       json);
                }
                matvec_max_abs_error = std::max(
                    matvec_max_abs_error,
                    std::abs(static_cast<double>(actual_output[index]) -
                             static_cast<double>(expected[index])));
            }
            if (matvec_max_abs_error > matvec_tolerance) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA Q8_0 matvec does not match the CPU reference",
                                       std::to_string(matvec_max_abs_error)),
                                   json);
            }
            matvec_rows_executed = matvec_rows;
            matvec_columns_executed = matvec_columns;
            matvec_device_bytes = matrix_bytes + input_bytes + output_bytes;
            matvec_smoke_passed = true;
        }
        if (iq1s_smoke_requested) {
            // The expert quantization, which telemetry says carries most of the arithmetic. Its
            // grid must be resident on the device, so it shares the allocation with the matrix.
            constexpr std::size_t iq1s_rows = 2'048U;
            constexpr std::size_t iq1s_columns = 4'096U;
            constexpr std::size_t iq1s_block_elements = 256U;
            constexpr std::size_t iq1s_block_bytes = 50U;
            // The kernel accumulates in f64 like the reference, so agreement should be to the last
            // bit of the f32 result. This leaves room for one rounding step, not for a different
            // summation order.
            constexpr double iq1s_tolerance = 1.0e-3;
            const std::size_t iq1s_blocks_per_row = iq1s_columns / iq1s_block_elements;
            const ByteCount iq1s_matrix_bytes =
                static_cast<ByteCount>(iq1s_rows) * iq1s_blocks_per_row * iq1s_block_bytes;
            const auto grid = vastloom::inference::iq1_s_grid_table();
            const ByteCount grid_bytes = grid.size_bytes();
            const ByteCount iq1s_input_bytes =
                static_cast<ByteCount>(iq1s_columns) * sizeof(float);
            const ByteCount iq1s_output_bytes = static_cast<ByteCount>(iq1s_rows) * sizeof(float);

            auto caller_host = governor.try_reserve(
                vastloom::safety::ResourceKind::scratch,
                iq1s_matrix_bytes + iq1s_input_bytes + 2U * iq1s_output_bytes,
                "cuda-iq1s-smoke-host");
            if (!caller_host) return print_error(caller_host.error(), json);

            // Deterministic encoded bytes: every field is exercised, including the sign bit and
            // the per-group scale, without needing a quantizer for this format.
            std::vector<std::byte> encoded(static_cast<std::size_t>(iq1s_matrix_bytes));
            for (std::size_t index = 0U; index < encoded.size(); ++index) {
                encoded[index] = static_cast<std::byte>((index * 37U + 11U) % 251U);
            }
            for (std::size_t block = 0U; block * iq1s_block_bytes < encoded.size(); ++block) {
                // Scales must be finite - the byte pattern above could spell NaN - but they must
                // not be tidy. An earlier version of this gate pinned every scale to 0.25, which
                // combined with a k/256 input to make every product exactly representable; f32 and
                // f64 accumulation then agreed to the bit and the gate passed a kernel that was
                // accumulating in the wrong precision. Vary the exponent and fill the mantissa so
                // the arithmetic is as awkward as a real expert's.
                const auto exponent = static_cast<std::uint16_t>(0x0CU + (block % 5U));
                const auto mantissa = static_cast<std::uint16_t>((block * 7'919U + 13U) & 0x03FFU);
                const auto scale_bits =
                    static_cast<std::uint16_t>((exponent << 10U) | mantissa);
                std::memcpy(encoded.data() + block * iq1s_block_bytes, &scale_bits,
                            sizeof(scale_bits));
            }
            std::vector<float> iq1s_input(iq1s_columns);
            for (std::size_t index = 0U; index < iq1s_input.size(); ++index) {
                // Non-dyadic by construction, for the same reason.
                const auto scrambled =
                    static_cast<std::uint32_t>((index * 2'654'435'761ULL + 97ULL) % 1'000'003ULL);
                iq1s_input[index] =
                    static_cast<float>(scrambled) / 1'000'003.0F - 0.5F;
            }

            std::vector<float> iq1s_expected(iq1s_rows);
            vastloom::inference::QuantizedKernelLimits iq1s_limits;
            iq1s_limits.max_elements = iq1s_rows * iq1s_columns;
            iq1s_limits.max_threads = policy.value().cpu_threads;
            const auto iq1s_cpu_started = std::chrono::steady_clock::now();
            auto iq1s_cpu = vastloom::inference::matvec_iq1_s_f32(
                encoded, iq1s_rows, iq1s_columns, iq1s_input, iq1s_expected, iq1s_limits);
            iq1s_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - iq1s_cpu_started).count();
            if (!iq1s_cpu) return print_error(iq1s_cpu.error(), json);

            const ByteCount grid_offset = iq1s_matrix_bytes;
            const ByteCount iq1s_input_offset = grid_offset + grid_bytes;
            const ByteCount iq1s_output_offset = iq1s_input_offset + iq1s_input_bytes;
            auto iq1s_device = driver.value().allocate(
                0, iq1s_output_offset + iq1s_output_bytes, minimum_free.value(), governor,
                "cuda-matvec-iq1-s-smoke");
            if (!iq1s_device) return print_error(iq1s_device.error(), json);
            auto iq1s_allocation = iq1s_device.take_value();
            auto copied = driver.value().copy_from_host(iq1s_allocation, 0U, encoded);
            if (!copied) return print_error(copied.error(), json);
            copied = driver.value().copy_from_host(
                iq1s_allocation, grid_offset, std::as_bytes(grid));
            if (!copied) return print_error(copied.error(), json);

            std::vector<float> iq1s_actual(iq1s_rows);
            const auto iq1s_started = std::chrono::steady_clock::now();
            auto iq1s_launched = driver.value().matvec_iq1_s_with_transfer(
                iq1s_allocation, 0U, grid_offset, iq1s_input_offset, iq1s_output_offset,
                iq1s_rows, iq1s_blocks_per_row,
                std::as_bytes(std::span<const float>(iq1s_input)),
                std::as_writable_bytes(std::span<float>(iq1s_actual)));
            if (!iq1s_launched) return print_error(iq1s_launched.error(), json);
            iq1s_kernel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - iq1s_started).count();
            auto iq1s_released = iq1s_allocation.release();
            if (!iq1s_released) return print_error(iq1s_released.error(), json);

            for (std::size_t index = 0U; index < iq1s_actual.size(); ++index) {
                if (!std::isfinite(iq1s_actual[index])) {
                    return print_error(vastloom::make_error(
                                           vastloom::ErrorCode::integrity_error,
                                           "CUDA IQ1_S matvec produced a non-finite value"),
                                       json);
                }
                iq1s_max_abs_error = std::max(
                    iq1s_max_abs_error,
                    std::abs(static_cast<double>(iq1s_actual[index]) -
                             static_cast<double>(iq1s_expected[index])));
            }
            if (iq1s_max_abs_error > iq1s_tolerance) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA IQ1_S matvec does not match the CPU reference",
                                       std::to_string(iq1s_max_abs_error)),
                                   json);
            }
            iq1s_smoke_passed = true;
        }
        const auto snapshot = governor.snapshot();
        if (snapshot.reserved[static_cast<std::size_t>(
                vastloom::safety::ResourceKind::vram)] != 0U) {
            return print_error(vastloom::make_error(
                                   vastloom::ErrorCode::integrity_error,
                                   "CUDA smoke left a residual VRAM reservation"),
                               json);
        }
    }

    if (json) {
        std::string output = vastloom::cuda::to_json(probe.value());
        if (operation_requested) {
            if (output.empty() || output.back() != '}') {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::internal_error,
                                       "CUDA JSON serialization produced an invalid object"),
                                   true);
            }
            output.pop_back();
            if (allocation_smoke_requested) {
                output += ",\"allocation_smoke\":{\"passed\":";
                output += allocation_smoke_passed ? "true" : "false";
                output += ",\"bytes\":" + std::to_string(smoke_bytes.value()) +
                          ",\"minimum_free_vram_after\":" +
                          std::to_string(minimum_free.value()) + "}}";
            } else if (iq1s_smoke_requested) {
                output += ",\"matvec_iq1_s_smoke\":{\"passed\":";
                output += iq1s_smoke_passed ? "true" : "false";
                output += ",\"rows\":2048,\"columns\":4096,\"max_abs_error\":" +
                          std::to_string(iq1s_max_abs_error) +
                          ",\"kernel_ms\":" + std::to_string(iq1s_kernel_ms) +
                          ",\"cpu_ms\":" + std::to_string(iq1s_cpu_ms) + "}}";
            } else if (matvec_smoke_requested) {
                output += ",\"matvec_q8_0_smoke\":{\"passed\":";
                output += matvec_smoke_passed ? "true" : "false";
                output += ",\"rows\":" + std::to_string(matvec_rows_executed) +
                          ",\"columns\":" + std::to_string(matvec_columns_executed) +
                          ",\"max_abs_error\":" + std::to_string(matvec_max_abs_error) +
                          ",\"device_buffer_bytes\":" + std::to_string(matvec_device_bytes) +
                          ",\"kernel_ms\":" + std::to_string(matvec_kernel_ms) +
                          ",\"cpu_ms\":" + std::to_string(matvec_cpu_ms) +
                          ",\"minimum_free_vram_after\":" +
                          std::to_string(minimum_free.value()) + "}}";
            } else {
                output += ",\"rmsnorm_smoke\":{\"passed\":";
                output += rmsnorm_smoke_passed ? "true" : "false";
                output += ",\"rows\":" + std::to_string(rmsnorm_stats.row_count) +
                          ",\"width\":" + std::to_string(rmsnorm_stats.width) +
                          ",\"max_abs_error\":" + std::to_string(rmsnorm_max_abs_error) +
                          ",\"device_buffer_bytes\":" +
                          std::to_string(rmsnorm_stats.device_buffer_bytes) +
                          ",\"host_to_device_bytes\":" +
                          std::to_string(rmsnorm_stats.host_to_device_bytes) +
                          ",\"device_to_host_bytes\":" +
                          std::to_string(rmsnorm_stats.device_to_host_bytes) +
                          ",\"minimum_free_vram_after\":" +
                          std::to_string(minimum_free.value()) + "}}";
            }
        }
        std::cout << output << '\n';
    } else {
        std::cout << vastloom::cuda::to_text(probe.value());
        if (allocation_smoke_requested) {
            std::cout << "Allocation:     PASS (" << vastloom::format_bytes(smoke_bytes.value())
                      << ", released; live reserve "
                      << vastloom::format_bytes(minimum_free.value()) << ")\n";
        } else if (rmsnorm_smoke_requested) {
            std::cout << "RMSNorm:        PASS (" << rmsnorm_stats.row_count << "x"
                      << rmsnorm_stats.width << ", max abs error "
                      << rmsnorm_max_abs_error << ", released)\n";
        } else if (iq1s_smoke_requested) {
            std::cout << "IQ1_S matvec:   " << (iq1s_smoke_passed ? "PASS" : "FAIL")
                      << " (2048x4096, max abs error " << iq1s_max_abs_error << ", GPU "
                      << iq1s_kernel_ms << " ms vs CPU " << iq1s_cpu_ms << " ms, released)\n";
        } else if (matvec_smoke_requested) {
            std::cout << "Q8_0 matvec:    PASS (" << matvec_rows_executed << "x"
                      << matvec_columns_executed << ", max abs error " << matvec_max_abs_error
                      << ", GPU " << matvec_kernel_ms << " ms vs CPU " << matvec_cpu_ms
                      << " ms, released)\n";
        }
    }
    return 0;
}

int command_inspect(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "inspect requires a manifest or checkpoint path"));
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(args, 1U, {"--declared-size", "--tensor-limit"}, {"--json"});
    if (!valid) return print_error(valid.error(), json);
    std::error_code error;
    const std::filesystem::path path = cli_path(args[0]);
    if (vastloom::util::to_lower(path.extension().string()) == ".gguf") {
        auto tensor_limit = optional_size(args, "--tensor-limit", 0U);
        if (!tensor_limit || tensor_limit.value() > 1'000U) {
            return print_error(tensor_limit ? vastloom::make_error(
                                                  vastloom::ErrorCode::invalid_argument,
                                                  "GGUF --tensor-limit cannot exceed 1000")
                                            : tensor_limit.error(),
                               json);
        }
        Result<vastloom::model::GgufFile> gguf = vastloom::make_error(
            vastloom::ErrorCode::internal_error, "GGUF inspection was not initialized");
        if (option_value(args, "--declared-size").has_value()) {
            auto declared_size = optional_bytes(args, "--declared-size", 0U);
            if (!declared_size || declared_size.value() == 0U) {
                return print_error(declared_size ? vastloom::make_error(
                                                       vastloom::ErrorCode::invalid_argument,
                                                       "GGUF --declared-size must be positive")
                                                 : declared_size.error(),
                                   json);
            }
            gguf = vastloom::model::GgufFile::open_header_prefix(path, declared_size.value());
        } else {
            gguf = vastloom::model::GgufFile::open(path);
        }
        if (!gguf) return print_error(gguf.error(), json);
        std::cout << (json ? vastloom::model::to_json(gguf.value(), tensor_limit.value())
                          : vastloom::model::to_text(gguf.value(), tensor_limit.value())) << '\n';
        return 0;
    }
    if (option_value(args, "--declared-size").has_value() ||
        option_value(args, "--tensor-limit").has_value()) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "GGUF inspection options require a .gguf path"),
                           json);
    }
    if (std::filesystem::is_directory(path, error) ||
        vastloom::util::to_lower(path.extension().string()) == ".safetensors") {
        auto catalog = vastloom::model::inspect_safetensors_checkpoint(path);
        if (!catalog) return print_error(catalog.error(), json);
        std::cout << (json ? vastloom::model::to_json(catalog.value()) : vastloom::model::to_text(catalog.value())) << '\n';
        return 0;
    }
    auto manifest = vastloom::model::load_manifest(cli_path(args[0]));
    if (!manifest) return print_error(manifest.error(), json);
    std::cout << (json ? vastloom::model::to_json(manifest.value()) : vastloom::model::to_text(manifest.value())) << '\n';
    return 0;
}

int command_verify(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "verify requires a manifest path"));
    const bool json = has_flag(args, "--json");
    auto manifest = vastloom::model::load_manifest(cli_path(args[0]));
    if (!manifest) return print_error(manifest.error(), json);
    const std::filesystem::path default_root = manifest.value().source_path.parent_path().empty()
        ? std::filesystem::path(".") : manifest.value().source_path.parent_path();
    const std::filesystem::path root = option_path(args, "--model-root", default_root);
    const auto mode = has_flag(args, "--full") ? vastloom::model::VerificationMode::full : vastloom::model::VerificationMode::quick;
    auto report = vastloom::model::verify_model(manifest.value(), root, mode);
    if (!report) return print_error(report.error(), json);
    std::cout << (json ? vastloom::model::to_json(report.value()) : vastloom::model::to_text(report.value())) << '\n';
    return report.value().ok ? 0 : 65;
}

int command_plan(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "plan requires a manifest path"));
    const bool json = has_flag(args, "--json");
    auto manifest = vastloom::model::load_manifest(cli_path(args[0]));
    if (!manifest) return print_error(manifest.error(), json);
    auto profile = vastloom::planner::parse_profile(option_value(args, "--profile").value_or("balanced"));
    if (!profile) return print_error(profile.error(), json);
    const auto storage_path = option_path(args, "--storage", std::filesystem::current_path());

    vastloom::hardware::HardwareSnapshot hardware;
    if (has_flag(args, "--reference-pc")) {
        hardware = reference_pc_snapshot(storage_path);
    } else {
        auto probed = vastloom::hardware::HardwareProbe::collect(storage_path);
        if (!probed) return print_error(probed.error(), json);
        hardware = probed.take_value();
    }

    auto simulated_ram = optional_bytes(args, "--simulate-ram", hardware.memory.total_bytes);
    auto simulated_vram = optional_bytes(args, "--simulate-vram", hardware.gpus.empty() ? 0 : hardware.gpus.front().total_vram_bytes);
    auto simulated_storage = optional_bytes(args, "--simulate-storage", hardware.storage.capacity_bytes);
    auto simulated_free = optional_bytes(args, "--simulate-free", hardware.storage.available_bytes);
    if (!simulated_ram) return print_error(simulated_ram.error(), json);
    if (!simulated_vram) return print_error(simulated_vram.error(), json);
    if (!simulated_storage) return print_error(simulated_storage.error(), json);
    if (!simulated_free) return print_error(simulated_free.error(), json);

    if (option_value(args, "--simulate-ram").has_value()) {
        hardware.memory.total_bytes = simulated_ram.value();
        hardware.memory.available_bytes = vastloom::percentage(simulated_ram.value(), 0.82);
    }
    if (option_value(args, "--simulate-vram").has_value()) {
        hardware.gpus.clear();
        if (simulated_vram.value() != 0) {
            hardware.gpus.push_back(vastloom::hardware::GpuInfo{
                "CUDA", "Simulated GPU", "simulated", simulated_vram.value(),
                vastloom::percentage(simulated_vram.value(), 0.90), std::nullopt, std::nullopt});
        }
    }
    hardware.storage.capacity_bytes = simulated_storage.value();
    hardware.storage.free_bytes = simulated_free.value();
    hardware.storage.available_bytes = simulated_free.value();

    auto plan = vastloom::planner::build_plan(hardware, manifest.value(), profile.value());
    if (!plan) return print_error(plan.error(), json);
    std::cout << (json ? vastloom::planner::to_json(plan.value()) : vastloom::planner::to_text(plan.value())) << '\n';
    return plan.value().compatible ? 0 : 78;
}

int command_trace_sim(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "trace-sim requires a CSV trace path"));
    const bool json = has_flag(args, "--json");
    auto vram = optional_bytes(args, "--vram", 0);
    auto ram = optional_bytes(args, "--ram", 0);
    if (!vram) return print_error(vram.error(), json);
    if (!ram) return print_error(ram.error(), json);
    auto policy = vastloom::research::parse_cache_policy(option_value(args, "--policy").value_or("lru"));
    if (!policy) return print_error(policy.error(), json);
    vastloom::research::TraceSimulationConfig config;
    config.vram_cache_bytes = vram.value();
    config.ram_cache_bytes = ram.value();
    config.policy = policy.value();
    auto report = vastloom::research::simulate_trace(cli_path(args[0]), config);
    if (!report) return print_error(report.error(), json);
    std::cout << (json ? vastloom::research::to_json(report.value()) : vastloom::research::to_text(report.value())) << '\n';
    return 0;
}

int command_io_smoke(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "io-smoke requires a file path"));
    auto offset = optional_bytes(args, "--offset", 0);
    auto size = optional_bytes(args, "--size", 1U * vastloom::MiB);
    if (!offset) return print_error(offset.error());
    if (!size) return print_error(size.error());

    vastloom::safety::ResourceLimits limits;
    limits.host_memory_hard_limit = 2U * vastloom::GiB;
    limits.ram_hard_limit = 2U * vastloom::GiB;
    limits.vram_hard_limit = 1U * vastloom::GiB;
    limits.staging_hard_limit = std::max<ByteCount>(size.value(), 16U * vastloom::MiB);
    limits.scratch_hard_limit = 1U * vastloom::GiB;
    vastloom::safety::ResourceGovernor governor(limits);
    vastloom::io::IoPoolOptions pool_options;
    pool_options.workers = 2U;
    pool_options.max_queued_requests = 8U;
    pool_options.max_request_bytes = limits.staging_hard_limit;
    vastloom::io::BoundedIoPool pool(pool_options, governor);
    auto submitted = pool.submit_read(cli_path(args[0]), offset.value(), size.value(), "cli-io-smoke");
    if (!submitted) return print_error(submitted.error());
    auto result = submitted.value().get();
    if (!result) return print_error(result.error());
    std::cout << "Read " << vastloom::format_bytes(result.value().bytes.size())
              << " under a bounded staging reservation.\n";
    return 0;
}

int command_store_init(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "store-init requires a path"));
    const bool json = has_flag(args, "--json");
    auto reserve = optional_bytes(args, "--reserve", 300U * vastloom::GiB);
    if (!reserve) return print_error(reserve.error(), json);
    auto store = vastloom::model::ModelStore::initialize(cli_path(args[0]), {reserve.value()});
    if (!store) return print_error(store.error(), json);
    std::cout << (json ? vastloom::model::to_json(store.value().layout())
                      : vastloom::model::to_text(store.value().layout())) << '\n';
    return 0;
}

int command_store_info(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "store-info requires a path"));
    const bool json = has_flag(args, "--json");
    auto reserve = optional_bytes(args, "--reserve", 300U * vastloom::GiB);
    if (!reserve) return print_error(reserve.error(), json);
    auto store = vastloom::model::ModelStore::open(cli_path(args[0]), {reserve.value()});
    if (!store) return print_error(store.error(), json);
    std::cout << (json ? vastloom::model::to_json(store.value().layout())
                      : vastloom::model::to_text(store.value().layout())) << '\n';
    return 0;
}

int command_runtime_policy(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    const auto storage = option_path(args, "--storage", std::filesystem::current_path());
    auto hardware = vastloom::hardware::HardwareProbe::collect(storage);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);
    std::cout << (json ? vastloom::safety::to_json(policy.value())
                      : vastloom::safety::to_text(policy.value())) << '\n';
    return 0;
}

int command_moe_inspect(const std::vector<std::string>& args) {
    auto valid = validate_options(args, 1U, {}, {"--no-verify", "--json"});
    const bool json = has_flag(args, "--json");
    if (!valid) return print_error(valid.error(), json);

    ByteCount safe_ram_cap = 0U;
    const auto model_path = cli_path(args[0]);
    auto hardware = vastloom::hardware::HardwareProbe::collect(model_path);
    if (hardware) {
        auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
        if (policy) safe_ram_cap = policy.value().resources.ram_hard_limit;
    }
    auto inspection = vastloom::inference::inspect_olmoe_checkpoint(
        model_path, !has_flag(args, "--no-verify"), safe_ram_cap);
    if (!inspection) return print_error(inspection.error(), json);
    std::cout << (json ? vastloom::inference::to_json(inspection.value())
                      : vastloom::inference::to_text(inspection.value())) << '\n';
    return 0;
}

int command_deepseek_inspect(const std::vector<std::string>& args) {
    auto valid = validate_options(args, 1U, {}, {"--no-verify", "--json"});
    const bool json = has_flag(args, "--json");
    if (!valid) return print_error(valid.error(), json);

    ByteCount safe_ram_cap = 0U;
    const auto model_path = cli_path(args[0]);
    auto hardware = vastloom::hardware::HardwareProbe::collect(model_path);
    if (hardware) {
        auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
        if (policy) safe_ram_cap = policy.value().resources.ram_hard_limit;
    }
    auto inspection = vastloom::inference::inspect_deepseek_v4_checkpoint(
        model_path, !has_flag(args, "--no-verify"), safe_ram_cap);
    if (!inspection) return print_error(inspection.error(), json);
    std::cout << (json ? vastloom::inference::to_json(inspection.value())
                      : vastloom::inference::to_text(inspection.value())) << '\n';
    return 0;
}

Result<std::vector<ByteCount>> parse_declared_sizes(std::string_view value,
                                                    std::size_t expected_count) {
    std::vector<ByteCount> sizes;
    sizes.reserve(expected_count);
    std::size_t begin = 0U;
    while (begin <= value.size()) {
        const auto separator = value.find(',', begin);
        const auto item = value.substr(
            begin, separator == std::string_view::npos ? value.size() - begin : separator - begin);
        auto parsed = vastloom::parse_bytes(item);
        if (!parsed || parsed.value() == 0U) {
            return parsed ? vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                  "DeepSeek GGUF declared sizes must be positive")
                          : parsed.error();
        }
        sizes.push_back(parsed.value());
        if (separator == std::string_view::npos) break;
        begin = separator + 1U;
    }
    if (sizes.size() != expected_count) {
        return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                    "DeepSeek GGUF declared-size count differs from shard count");
    }
    return sizes;
}

Result<std::vector<std::uint32_t>> parse_expert_ids(std::string_view value) {
    const auto items = vastloom::util::split(value, ',', true);
    if (items.empty() || items.size() > 64U) {
        return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                    "DeepSeek probe expert list is empty or exceeds 64 IDs");
    }
    std::vector<std::uint32_t> experts;
    experts.reserve(items.size());
    for (const auto& item : items) {
        const auto trimmed = vastloom::util::trim(item);
        std::uint32_t expert = 0U;
        const auto converted = std::from_chars(
            trimmed.data(), trimmed.data() + trimmed.size(), expert);
        if (trimmed.empty() || converted.ec != std::errc{} ||
            converted.ptr != trimmed.data() + trimmed.size()) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                        "DeepSeek probe expert IDs must be unsigned integers",
                                        item);
        }
        experts.push_back(expert);
    }
    return experts;
}

int command_deepseek_gguf_inspect(const std::vector<std::string>& args) {
    std::size_t positional_count = 0U;
    while (positional_count < args.size() && !vastloom::util::starts_with(args[positional_count], "--")) {
        ++positional_count;
    }
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(args, positional_count,
                                  {"--declared-sizes", "--revision", "--context", "--batch-size"},
                                  {"--json"});
    if (!valid) return print_error(valid.error(), json);
    if (positional_count == 0U || positional_count > 1'024U) {
        return print_error(vastloom::make_error(
                               vastloom::ErrorCode::invalid_argument,
                               "deepseek-gguf-inspect requires between 1 and 1024 shard paths"),
                           json);
    }

    std::vector<ByteCount> declared_sizes;
    if (const auto values = option_value(args, "--declared-sizes")) {
        auto parsed = parse_declared_sizes(*values, positional_count);
        if (!parsed) return print_error(parsed.error(), json);
        declared_sizes = parsed.take_value();
    }
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U) {
        return print_error(vastloom::make_error(
                               vastloom::ErrorCode::invalid_argument,
                               "DeepSeek context or batch size is outside safe bounds"),
                           json);
    }

    std::vector<vastloom::model::GgufFile> shards;
    shards.reserve(positional_count);
    vastloom::model::GgufReaderLimits reader_limits;
    reader_limits.retained_integral_array_keys = {
        "deepseek4.attention.compress_ratios"};
    reader_limits.retained_floating_array_keys = {
        "deepseek4.swiglu_clamp_exp",
        "deepseek4.swiglu_clamp_shexp"};
    for (std::size_t index = 0U; index < positional_count; ++index) {
        Result<vastloom::model::GgufFile> opened = declared_sizes.empty()
            ? vastloom::model::GgufFile::open(cli_path(args[index]), reader_limits)
            : vastloom::model::GgufFile::open_header_prefix(
                  cli_path(args[index]), declared_sizes[index], reader_limits);
        if (!opened) return print_error(opened.error(), json);
        shards.push_back(opened.take_value());
    }
    auto split = vastloom::model::finalize_gguf_split_catalog(std::move(shards));
    if (!split) return print_error(split.error(), json);
    vastloom::inference::DeepseekV4GgufCatalogOptions options;
    options.revision = option_value(args, "--revision").value_or("unverified-local-headers");
    options.context_tokens = static_cast<std::uint32_t>(context.value());
    options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    auto hardware = vastloom::hardware::HardwareProbe::collect(cli_path(args[0]));
    if (hardware) {
        auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
        if (policy) {
            options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
            options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
        }
    }
    options.metadata_only = true;
    auto report = vastloom::inference::build_deepseek_v4_gguf_catalog(split.value(), options);
    if (!report) return print_error(report.error(), json);
    std::cout << (json ? vastloom::inference::to_json(report.value())
                      : vastloom::inference::to_text(report.value())) << '\n';
    return 0;
}

int command_deepseek_gguf_checkpoint(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args, 1U, {"--context", "--batch-size"}, {"--no-verify", "--json"});
    if (!valid) return print_error(valid.error(), json);
    if (args.empty() || vastloom::util::starts_with(args.front(), "--")) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-gguf-checkpoint requires one imported checkpoint directory"),
            json);
    }
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "DeepSeek context or batch size is outside safe bounds"),
            json);
    }

    const auto checkpoint_root = cli_path(args.front());
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);

    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions options;
    options.verify_hashes = !has_flag(args, "--no-verify");
    options.context_tokens = static_cast<std::uint32_t>(context.value());
    options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root,
        vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint(),
        options);
    if (!report) return print_error(report.error(), json);
    std::cout << (json ? vastloom::inference::to_json(report.value())
                      : vastloom::inference::to_text(report.value())) << '\n';
    return 0;
}

int command_deepseek_gguf_probe_plan(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args,
        1U,
        {"--layer", "--experts", "--tile-size", "--context", "--batch-size",
         "--route-artifact", "--token"},
        {"--no-verify", "--json"});
    if (!valid) return print_error(valid.error(), json);
    if (args.empty() || vastloom::util::starts_with(args.front(), "--")) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-gguf-probe-plan requires one imported checkpoint directory"),
            json);
    }
    const auto layer_option = option_value(args, "--layer");
    const auto expert_option = option_value(args, "--experts");
    const auto route_option = option_value(args, "--route-artifact");
    const auto token_option = option_value(args, "--token");
    // Either the caller supplies the top-k it obtained elsewhere, or the planner derives it from an
    // imported hash-routing table. Supplying both would leave it ambiguous which one has authority.
    if (route_option.has_value() != token_option.has_value()) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "--route-artifact and --token must be supplied together"),
            json);
    }
    if (route_option.has_value() == expert_option.has_value()) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-gguf-probe-plan requires either --experts or --route-artifact with --token"),
            json);
    }
    if (!layer_option.has_value()) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "deepseek-gguf-probe-plan requires --layer"),
            json);
    }
    auto layer = optional_size(args, "--layer", 0U);
    auto experts = expert_option.has_value()
        ? parse_expert_ids(*expert_option)
        : vastloom::Result<std::vector<std::uint32_t>>{std::vector<std::uint32_t>{}};
    auto routing_token = optional_size(args, "--token", 0U);
    auto tile_size = optional_bytes(
        args, "--tile-size", vastloom::model::kDefaultLayerStreamingTileBytes);
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!layer) return print_error(layer.error(), json);
    if (!experts) return print_error(experts.error(), json);
    if (!routing_token) return print_error(routing_token.error(), json);
    if (!tile_size) return print_error(tile_size.error(), json);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (layer.value() > std::numeric_limits<std::uint32_t>::max() ||
        routing_token.value() > std::numeric_limits<std::uint32_t>::max() ||
        tile_size.value() == 0U || tile_size.value() > 1U * vastloom::GiB ||
        context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "DeepSeek probe layer, token, tile, context or batch bound is invalid"),
            json);
    }

    const auto checkpoint_root = cli_path(args.front());
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);

    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions options;
    options.verify_hashes = !has_flag(args, "--no-verify");
    options.context_tokens = static_cast<std::uint32_t>(context.value());
    options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root,
        vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint(),
        options);
    if (!report) return print_error(report.error(), json);

    // Authoritative mode: read the pinned routing table out of the already imported trunk and let
    // the checkpoint itself name the experts. The intermediate plan exists only to bind the
    // artifact; its caller-supplied expert list is discarded once the table has spoken.
    std::optional<vastloom::inference::DeepseekV4HashRouteResolution> route;
    if (route_option.has_value()) {
        // The trunk binding is independent of which experts are selected, but the planner still
        // requires a well-formed top-k. Use the lowest legal IDs to reach the routing table; the
        // real selection replaces them below and is the only one that reaches the final plan.
        std::vector<std::uint32_t> placeholder;
        try {
            placeholder.reserve(report.value().top_k);
            for (std::uint32_t index = 0U; index < report.value().top_k; ++index) {
                placeholder.push_back(index);
            }
        } catch (const std::bad_alloc&) {
            return print_error(
                vastloom::make_error(vastloom::ErrorCode::resource_exhausted,
                                     "Unable to allocate the routing bootstrap plan"),
                json);
        }
        auto binding_plan = vastloom::inference::build_deepseek_v4_layer_probe_plan(
            report.value(),
            static_cast<std::uint32_t>(layer.value()),
            placeholder,
            tile_size.value());
        if (!binding_plan) return print_error(binding_plan.error(), json);
        auto artifact = vastloom::model::verify_range_artifact(cli_path(*route_option));
        if (!artifact) return print_error(artifact.error(), json);
        auto binding = vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
            artifact.value(), report.value(), binding_plan.value(),
            vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint());
        if (!binding) return print_error(binding.error(), json);
        vastloom::safety::ResourceGovernor governor(policy.value().resources);
        auto resolved = vastloom::inference::deepseek_v4_resolve_hash_route(
            binding.value(), report.value(),
            static_cast<std::uint32_t>(routing_token.value()), governor);
        if (!resolved) return print_error(resolved.error(), json);
        route = resolved.take_value();
        experts = route->unique_experts;
    }

    auto plan = vastloom::inference::build_deepseek_v4_layer_probe_plan(
        report.value(),
        static_cast<std::uint32_t>(layer.value()),
        experts.value(),
        tile_size.value());
    if (!plan) return print_error(plan.error(), json);
    if (route.has_value()) {
        plan.value().routing_authority_verified = true;
        plan.value().routing_token = route->token;
    }
    std::cout << (json ? vastloom::inference::to_json(plan.value())
                      : vastloom::inference::to_text(plan.value())) << '\n';
    return 0;
}

int command_deepseek_gguf_global_plan(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args, 1U, {"--context", "--batch-size"}, {"--no-verify", "--json"});
    if (!valid) return print_error(valid.error(), json);
    if (args.empty() || vastloom::util::starts_with(args.front(), "--")) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-gguf-global-plan requires one imported checkpoint directory"),
            json);
    }
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "DeepSeek global plan context or batch bound is invalid"),
            json);
    }

    const auto checkpoint_root = cli_path(args.front());
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);
    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions options;
    options.verify_hashes = !has_flag(args, "--no-verify");
    options.context_tokens = static_cast<std::uint32_t>(context.value());
    options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root,
        vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint(),
        options);
    if (!report) return print_error(report.error(), json);
    auto plan = vastloom::inference::build_deepseek_v4_global_tensor_plan(
        report.value());
    if (!plan) return print_error(plan.error(), json);
    std::cout << (json ? vastloom::inference::to_json(plan.value())
                      : vastloom::inference::to_text(plan.value())) << '\n';
    return 0;
}

int command_range_artifact_inspect(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(args, 1U, {}, {"--no-verify", "--json"});
    if (!valid) return print_error(valid.error(), json);
    if (args.empty() || vastloom::util::starts_with(args.front(), "--")) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "range-artifact-inspect requires one artifact directory"),
            json);
    }
    vastloom::model::RangeArtifactVerificationOptions options;
    options.verify_hashes = !has_flag(args, "--no-verify");
    auto artifact = vastloom::model::verify_range_artifact(cli_path(args.front()), options);
    if (!artifact) return print_error(artifact.error(), json);
    std::cout << (json ? vastloom::model::to_json(artifact.value())
                      : vastloom::model::to_text(artifact.value())) << '\n';
    return 0;
}

int command_deepseek_gguf_range_bind(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args,
        2U,
        {"--layer", "--experts", "--tile-size", "--context", "--batch-size"},
        {"--json", "--load-tensor-set"});
    if (!valid) return print_error(valid.error(), json);
    if (args.size() < 2U || vastloom::util::starts_with(args[0], "--") ||
        vastloom::util::starts_with(args[1], "--")) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-gguf-range-bind requires checkpoint and artifact directories"),
            json);
    }
    const auto layer_option = option_value(args, "--layer");
    const auto expert_option = option_value(args, "--experts");
    if (!layer_option.has_value() || !expert_option.has_value()) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-gguf-range-bind requires --layer and the plan's --experts IDs"),
            json);
    }
    auto layer = optional_size(args, "--layer", 0U);
    auto experts = parse_expert_ids(*expert_option);
    auto tile_size = optional_bytes(
        args, "--tile-size", vastloom::model::kDefaultLayerStreamingTileBytes);
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!layer) return print_error(layer.error(), json);
    if (!experts) return print_error(experts.error(), json);
    if (!tile_size) return print_error(tile_size.error(), json);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (layer.value() > std::numeric_limits<std::uint32_t>::max() ||
        tile_size.value() == 0U || tile_size.value() > 1U * vastloom::GiB ||
        context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "DeepSeek range binding bounds are invalid"),
            json);
    }

    const auto checkpoint_root = cli_path(args[0]);
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);

    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions inspect_options;
    inspect_options.verify_hashes = true;
    inspect_options.context_tokens = static_cast<std::uint32_t>(context.value());
    inspect_options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    inspect_options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    inspect_options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    const auto& spec = vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root, spec, inspect_options);
    if (!report) return print_error(report.error(), json);
    auto plan = vastloom::inference::build_deepseek_v4_layer_probe_plan(
        report.value(),
        static_cast<std::uint32_t>(layer.value()),
        experts.value(),
        tile_size.value());
    if (!plan) return print_error(plan.error(), json);
    auto artifact = vastloom::model::verify_range_artifact(cli_path(args[1]));
    if (!artifact) return print_error(artifact.error(), json);
    auto binding = vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
        artifact.value(), report.value(), plan.value(), spec);
    if (!binding) return print_error(binding.error(), json);

    if (!has_flag(args, "--load-tensor-set")) {
        std::cout << (json ? vastloom::inference::to_json(binding.value())
                          : vastloom::inference::to_text(binding.value())) << '\n';
        return 0;
    }

    // Binding proves the local ranges are an exact projection of the pinned catalog. Loading proves
    // the payload those ranges hold is something the native graph can actually consume: every
    // matrix, vector and table reaches a declared consumer at its declared shape and storage type.
    auto config = vastloom::inference::deepseek_v4_reference_config_from_0731_gguf(report.value());
    if (!config) return print_error(config.error(), json);
    auto guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    if (!guard) return print_error(guard.error(), json);
    vastloom::safety::ResourceGovernor governor(policy.value().resources);
    auto set_id = vastloom::inference::deepseek_v4_bound_trunk_tensor_set_logical_id(
        binding.value());
    if (!set_id) return print_error(set_id.error(), json);
    auto residency = vastloom::model::ResidencyObject::create(
        set_id.take_value(), binding.value().payload_bytes);
    if (!residency) return print_error(residency.error(), json);
    vastloom::model::BoundRangeSetLoadOptions load_options;
    load_options.max_total_bytes = binding.value().payload_bytes;
    load_options.max_range_bytes = binding.value().maximum_local_range_bytes;
    load_options.max_ranges = binding.value().ranges.size();
    auto loaded = vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
        binding.value(), report.value(), residency.value(), governor, load_options);
    if (!loaded) return print_error(loaded.error(), json);
    auto trunk = vastloom::inference::deepseek_v4_bind_0731_gguf_layer0_trunk(
        loaded.value(), config.value());
    if (!trunk) return print_error(trunk.error(), json);

    const auto& telemetry = loaded.value().telemetry();
    if (json) {
        std::cout << "{\"ok\":true,\"layer\":" << binding.value().layer
                  << ",\"resident_bytes\":" << loaded.value().payload_bytes()
                  << ",\"tensors\":" << loaded.value().tensors().size()
                  << ",\"graph_matrices\":" << trunk.value().matrices.size()
                  << ",\"graph_float_tensors\":" << trunk.value().float_tensors.size()
                  << ",\"graph_integer_tensors\":" << trunk.value().integer_tensors.size()
                  << ",\"ranges_read\":" << telemetry.range_count
                  << ",\"elapsed_ms\":" << telemetry.elapsed_ms
                  << ",\"execution_supported\":false}" << '\n';
    } else {
        std::cout << "Layer trunk loaded:   " << loaded.value().payload_bytes()
                  << " bytes in " << telemetry.range_count << " ranges\n"
                  << "Graph consumers:      " << trunk.value().matrices.size()
                  << " matrices, " << trunk.value().float_tensors.size()
                  << " vectors, " << trunk.value().integer_tensors.size() << " table(s)\n"
                  << "Execution:            unsupported; this binds payload, it does not run it\n";
    }
    return 0;
}

int command_deepseek_gguf_global_bind(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args, 2U, {"--context", "--batch-size"}, {"--json", "--load-tensor-set"});
    if (!valid) return print_error(valid.error(), json);
    if (args.size() < 2U || vastloom::util::starts_with(args[0], "--") ||
        vastloom::util::starts_with(args[1], "--")) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-gguf-global-bind requires checkpoint and artifact directories"),
            json);
    }
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "DeepSeek global binding bounds are invalid"),
            json);
    }

    const auto checkpoint_root = cli_path(args[0]);
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);
    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions inspect_options;
    inspect_options.verify_hashes = true;
    inspect_options.context_tokens = static_cast<std::uint32_t>(context.value());
    inspect_options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    inspect_options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    inspect_options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    const auto& spec =
        vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root, spec, inspect_options);
    if (!report) return print_error(report.error(), json);
    auto plan = vastloom::inference::build_deepseek_v4_global_tensor_plan(
        report.value());
    if (!plan) return print_error(plan.error(), json);
    auto artifact = vastloom::model::verify_range_artifact(cli_path(args[1]));
    if (!artifact) return print_error(artifact.error(), json);
    auto binding = vastloom::inference::bind_deepseek_v4_global_range_artifact(
        artifact.value(), report.value(), plan.value(), spec);
    if (!binding) return print_error(binding.error(), json);

    if (!has_flag(args, "--load-tensor-set")) {
        std::cout << (json ? vastloom::inference::to_json(binding.value())
                          : vastloom::inference::to_text(binding.value())) << '\n';
        return 0;
    }

    // The globals are dominated by two 297,861,120-byte Q4_K matrices, so the default 64 MiB range
    // ceiling refuses them by design. Raise the bound to exactly what this verified binding holds
    // rather than to a round number, and let the Governor reserve the whole set once.
    auto config = vastloom::inference::deepseek_v4_reference_config_from_0731_gguf(report.value());
    if (!config) return print_error(config.error(), json);
    auto guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    if (!guard) return print_error(guard.error(), json);
    vastloom::safety::ResourceGovernor governor(policy.value().resources);
    auto set_id = vastloom::inference::deepseek_v4_bound_global_tensor_set_logical_id(
        binding.value());
    if (!set_id) return print_error(set_id.error(), json);
    auto residency = vastloom::model::ResidencyObject::create(
        set_id.take_value(), binding.value().payload_bytes);
    if (!residency) return print_error(residency.error(), json);
    vastloom::model::BoundRangeSetLoadOptions load_options;
    load_options.max_total_bytes = binding.value().payload_bytes;
    load_options.max_range_bytes = binding.value().maximum_local_range_bytes;
    load_options.max_ranges = binding.value().ranges.size();
    auto loaded = vastloom::inference::load_deepseek_v4_bound_global_tensor_set(
        binding.value(), report.value(), residency.value(), governor, load_options);
    if (!loaded) return print_error(loaded.error(), json);
    auto globals = vastloom::inference::deepseek_v4_bind_0731_gguf_globals(
        loaded.value(), config.value());
    if (!globals) return print_error(globals.error(), json);

    const auto& telemetry = loaded.value().telemetry();
    if (json) {
        std::cout << "{\"ok\":true,\"resident_bytes\":" << loaded.value().payload_bytes()
                  << ",\"tensors\":" << loaded.value().tensors().size()
                  << ",\"graph_matrices\":" << globals.value().matrices.size()
                  << ",\"graph_float_tensors\":" << globals.value().float_tensors.size()
                  << ",\"ranges_read\":" << telemetry.range_count
                  << ",\"elapsed_ms\":" << telemetry.elapsed_ms
                  << ",\"execution_supported\":false}" << '\n';
    } else {
        std::cout << "Globals loaded:       " << loaded.value().payload_bytes()
                  << " bytes in " << telemetry.range_count << " ranges\n"
                  << "Graph consumers:      " << globals.value().matrices.size()
                  << " matrices, " << globals.value().float_tensors.size() << " vectors\n"
                  << "Execution:            unsupported; this binds payload, it does not run it\n";
    }
    return 0;
}

// Executes one real DeepSeek layer from imported payload.
//
// The graph this builds is a single layer plus the checkpoint's own embedding, normalization and
// output head. That is deliberately not the model: the real network is 43 layers deep, and the
// logits below are what the first layer alone produces. What it does establish is that the official
// quantized weights - Q8_0 attention, Q5_K/Q6_K shared expert, BF16 router, IQ1_S/IQ3_XXS routed
// experts, the I32 routing table and the Q4_K embedding - execute together through the native
// kernels on real payload, with the router choosing the experts.
int command_deepseek_layer_probe(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args, 4U, {"--token", "--context", "--batch-size"}, {"--json"});
    if (!valid) return print_error(valid.error(), json);
    if (args.size() < 4U) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::invalid_argument,
                "deepseek-layer-probe requires MODEL_DIRECTORY TRUNK_ARTIFACT EXPERT_ARTIFACT "
                "GLOBAL_ARTIFACT"),
            json);
    }
    for (std::size_t index = 0U; index < 4U; ++index) {
        if (vastloom::util::starts_with(args[index], "--")) {
            return print_error(
                vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                     "deepseek-layer-probe requires four directories first"),
                json);
        }
    }
    auto token = optional_size(args, "--token", 0U);
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!token) return print_error(token.error(), json);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (token.value() > std::numeric_limits<std::uint32_t>::max() ||
        context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "DeepSeek layer probe bounds are invalid"),
            json);
    }

    const auto checkpoint_root = cli_path(args[0]);
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);
    auto process_guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    if (!process_guard) return print_error(process_guard.error(), json);

    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions inspect_options;
    inspect_options.verify_hashes = true;
    inspect_options.context_tokens = static_cast<std::uint32_t>(context.value());
    inspect_options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    inspect_options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    inspect_options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    const auto& spec = vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root, spec, inspect_options);
    if (!report) return print_error(report.error(), json);
    auto config = vastloom::inference::deepseek_v4_reference_config_from_0731_gguf(report.value());
    if (!config) return print_error(config.error(), json);

    std::vector<vastloom::model::RangeArtifact> artifacts;
    for (std::size_t index = 1U; index < 4U; ++index) {
        auto artifact = vastloom::model::verify_range_artifact(cli_path(args[index]));
        if (!artifact) return print_error(artifact.error(), json);
        artifacts.push_back(artifact.take_value());
    }
    auto relocated = vastloom::inference::deepseek_v4_relocate_catalog_to_artifacts(
        report.value().catalog, artifacts);
    if (!relocated) return print_error(relocated.error(), json);

    vastloom::safety::ResourceGovernor governor(policy.value().resources);

    // Bind the trunk and globals through their verified artifact paths, exactly as the two
    // --load-tensor-set gates do, then hand the packed results to a one-layer graph.
    auto trunk_plan = vastloom::inference::build_deepseek_v4_layer_probe_plan(
        report.value(), 0U, std::array<std::uint32_t, 6U>{35U, 53U, 200U, 222U, 245U, 254U},
        vastloom::model::kDefaultLayerStreamingTileBytes);
    if (!trunk_plan) return print_error(trunk_plan.error(), json);
    auto trunk_binding = vastloom::inference::bind_deepseek_v4_trunk_range_artifact(
        artifacts[0], report.value(), trunk_plan.value(), spec);
    if (!trunk_binding) return print_error(trunk_binding.error(), json);
    auto trunk_set_id = vastloom::inference::deepseek_v4_bound_trunk_tensor_set_logical_id(
        trunk_binding.value());
    if (!trunk_set_id) return print_error(trunk_set_id.error(), json);
    auto trunk_residency = vastloom::model::ResidencyObject::create(
        trunk_set_id.take_value(), trunk_binding.value().payload_bytes);
    if (!trunk_residency) return print_error(trunk_residency.error(), json);
    vastloom::model::BoundRangeSetLoadOptions trunk_options;
    trunk_options.max_total_bytes = trunk_binding.value().payload_bytes;
    trunk_options.max_range_bytes = trunk_binding.value().maximum_local_range_bytes;
    trunk_options.max_ranges = trunk_binding.value().ranges.size();
    auto trunk_tensors = vastloom::inference::load_deepseek_v4_bound_trunk_tensor_set(
        trunk_binding.value(), report.value(), trunk_residency.value(), governor, trunk_options);
    if (!trunk_tensors) return print_error(trunk_tensors.error(), json);
    auto trunk = vastloom::inference::deepseek_v4_bind_0731_gguf_layer0_trunk(
        trunk_tensors.value(), config.value());
    if (!trunk) return print_error(trunk.error(), json);

    auto global_plan = vastloom::inference::build_deepseek_v4_global_tensor_plan(report.value());
    if (!global_plan) return print_error(global_plan.error(), json);
    auto global_binding = vastloom::inference::bind_deepseek_v4_global_range_artifact(
        artifacts[2], report.value(), global_plan.value(), spec);
    if (!global_binding) return print_error(global_binding.error(), json);
    auto global_set_id = vastloom::inference::deepseek_v4_bound_global_tensor_set_logical_id(
        global_binding.value());
    if (!global_set_id) return print_error(global_set_id.error(), json);
    auto global_residency = vastloom::model::ResidencyObject::create(
        global_set_id.take_value(), global_binding.value().payload_bytes);
    if (!global_residency) return print_error(global_residency.error(), json);
    vastloom::model::BoundRangeSetLoadOptions global_options;
    global_options.max_total_bytes = global_binding.value().payload_bytes;
    global_options.max_range_bytes = global_binding.value().maximum_local_range_bytes;
    global_options.max_ranges = global_binding.value().ranges.size();
    auto global_tensors = vastloom::inference::load_deepseek_v4_bound_global_tensor_set(
        global_binding.value(), report.value(), global_residency.value(), governor,
        global_options);
    if (!global_tensors) return print_error(global_tensors.error(), json);
    auto globals = vastloom::inference::deepseek_v4_bind_0731_gguf_globals(
        global_tensors.value(), config.value());
    if (!globals) return print_error(globals.error(), json);

    // The routed experts come from the relocated catalog, so the bounded cache streams them out of
    // the imported artifact instead of the absent 82 GB checkpoint.
    auto expert_bindings = vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        relocated.value().catalog, 0U, report.value().experts_per_layer, config.value().width,
        config.value().intermediate_width);
    if (!expert_bindings) return print_error(expert_bindings.error(), json);
    vastloom::inference::DeepseekV4ExpertCacheOptions cache_options;
    cache_options.block_prefix = "layers.0.ffn";
    cache_options.max_cached_experts = 6U;
    cache_options.max_experts_per_request = 6U;
    auto cache = vastloom::inference::DeepseekV4BoundedExpertCache::create(
        expert_bindings.take_value(), governor, cache_options);
    if (!cache) return print_error(cache.error(), json);

    // One layer of the real configuration: every dimension, schedule and epsilon stays as the
    // checkpoint declares it, only the depth is cut to what has been imported. Layer 0 is
    // hash-routed, so the reduced hash-layer count stays 1.
    auto layer_config = config.value();
    layer_config.cpu_threads = policy.value().cpu_threads;
    layer_config.layer_count = 1U;
    layer_config.hash_layer_count = std::min<std::size_t>(layer_config.hash_layer_count, 1U);
    layer_config.mtp_layer_count = 0U;
    layer_config.mtp_compression_ratios.clear();
    layer_config.compression_ratios.resize(1U);
    if (!layer_config.routed_expert_swiglu_limits.empty()) {
        layer_config.routed_expert_swiglu_limits.resize(1U);
    }
    if (!layer_config.shared_expert_swiglu_limits.empty()) {
        layer_config.shared_expert_swiglu_limits.resize(1U);
    }

    vastloom::inference::DeepseekV4ReferenceGraph graph;
    graph.config = std::move(layer_config);
    for (auto& [name, matrix] : trunk.value().matrices) {
        graph.packed_matrices.emplace(name, std::move(matrix));
    }
    for (auto& [name, tensor] : trunk.value().float_tensors) {
        graph.packed_float_tensors.emplace(name, std::move(tensor));
    }
    for (auto& [name, tensor] : trunk.value().integer_tensors) {
        graph.packed_integer_tensors.emplace(name, std::move(tensor));
    }
    for (auto& [name, matrix] : globals.value().matrices) {
        graph.packed_matrices.emplace(name, std::move(matrix));
    }
    for (auto& [name, tensor] : globals.value().float_tensors) {
        graph.packed_float_tensors.emplace(name, std::move(tensor));
    }
    graph.expert_providers.emplace("layers.0.ffn", cache.value()->provider_binding());

    const std::array<std::uint32_t, 1U> token_ids{static_cast<std::uint32_t>(token.value())};
    const auto started = std::chrono::steady_clock::now();
    auto forward = vastloom::inference::deepseek_v4_reference_prefill(graph, token_ids);
    if (!forward) return print_error(forward.error(), json);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    const auto snapshot = cache.value()->snapshot();
    std::vector<std::uint32_t> executed_experts;
    if (!forward.value().layers.empty()) {
        for (const auto& access : forward.value().layers.front().expert_accesses) {
            executed_experts.push_back(access.expert_id);
        }
    }
    double logit_minimum = 0.0;
    double logit_maximum = 0.0;
    if (!forward.value().logits.empty()) {
        const auto [low, high] = std::minmax_element(forward.value().logits.begin(),
                                                     forward.value().logits.end());
        logit_minimum = *low;
        logit_maximum = *high;
    }

    if (json) {
        std::cout << "{\"ok\":true,\"complete_model\":false,\"executed_layers\":1"
                  << ",\"model_layers\":" << config.value().layer_count
                  << ",\"token\":" << token.value()
                  << ",\"logits\":" << forward.value().logits.size()
                  << ",\"logit_minimum\":" << logit_minimum
                  << ",\"logit_maximum\":" << logit_maximum
                  << ",\"argmax_token\":" << forward.value().greedy_token_id
                  << ",\"experts_executed\":[";
        for (std::size_t index = 0U; index < executed_experts.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << executed_experts[index];
        }
        std::cout << "],\"expert_bytes_read\":" << snapshot.logical_bytes_read
                  << ",\"expert_misses\":" << snapshot.misses
                  << ",\"trunk_bytes\":" << trunk_tensors.value().payload_bytes()
                  << ",\"global_bytes\":" << global_tensors.value().payload_bytes()
                  << ",\"relocated_tensors\":" << relocated.value().relocated_tensors
                  << ",\"relocated_expert_bundles\":" << relocated.value().relocated_expert_bundles
                  << ",\"elapsed_ms\":" << elapsed << '}' << '\n';
    } else {
        std::cout << "DeepSeek layer probe:  token " << token.value() << '\n'
                  << "Executed layers:       1 of " << config.value().layer_count
                  << " (this is not the model's output)\n"
                  << "Routed experts:        ";
        for (std::size_t index = 0U; index < executed_experts.size(); ++index) {
            if (index != 0U) std::cout << ", ";
            std::cout << executed_experts[index];
        }
        std::cout << '\n'
                  << "Expert bytes read:     " << snapshot.logical_bytes_read << '\n'
                  << "Logits:                " << forward.value().logits.size()
                  << " values in [" << logit_minimum << ", " << logit_maximum << "]\n"
                  << "Elapsed:               " << elapsed << " ms\n";
    }
    return 0;
}

// Runs the real graph to a chosen depth, streaming each layer's always-active tensors out of the
// complete checkpoint.
//
// Depth is explicit and capacity is checked before anything is allocated. The Safety Governor would
// refuse an oversized load anyway, but it would do so partway through, after minutes of reading; a
// preflight that compares the exact resident total against the live budget refuses in advance and
// says by how much. Nothing here can grow without a reservation: trunk tensors, globals and expert
// bundles all pass through the Governor, the host keeps its 8 GiB headroom, and the process guard is
// sampled at every layer boundary.
int command_deepseek_forward(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args, 1U,
        {"--token", "--layers", "--context", "--batch-size", "--expert-cache", "--prompt",
         "--tokenizer", "--max-tokens"},
        {"--json", "--dry-run", "--chat", "--cpu-only"});
    if (!valid) return print_error(valid.error(), json);
    if (args.empty() || vastloom::util::starts_with(args.front(), "--")) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "deepseek-forward requires one imported checkpoint directory"),
            json);
    }
    auto token = optional_size(args, "--token", 0U);
    auto requested_layers = optional_size(args, "--layers", 1U);
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    auto expert_cache = optional_size(args, "--expert-cache", 8U);
    auto max_tokens = optional_size(args, "--max-tokens", 1U);
    if (!token) return print_error(token.error(), json);
    if (!requested_layers) return print_error(requested_layers.error(), json);
    if (!context) return print_error(context.error(), json);
    if (!batch_size) return print_error(batch_size.error(), json);
    if (!expert_cache) return print_error(expert_cache.error(), json);
    if (!max_tokens) return print_error(max_tokens.error(), json);
    if (token.value() > std::numeric_limits<std::uint32_t>::max() ||
        requested_layers.value() == 0U || requested_layers.value() > 128U ||
        context.value() == 0U || context.value() > 1'048'576U ||
        batch_size.value() == 0U || batch_size.value() > 8U ||
        max_tokens.value() == 0U || max_tokens.value() > 512U ||
        expert_cache.value() == 0U || expert_cache.value() > 64U) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "DeepSeek forward bounds are invalid"),
            json);
    }

    const auto checkpoint_root = cli_path(args.front());
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);
    auto process_guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    if (!process_guard) return print_error(process_guard.error(), json);

    // Executable source ranges require the full-hash gate: a checkpoint is only readable here after
    // every complete file has been verified against its pinned SHA-256. That costs one pass over the
    // payload and is the difference between planning and running.
    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions inspect_options;
    inspect_options.verify_hashes = true;
    inspect_options.context_tokens = static_cast<std::uint32_t>(context.value());
    inspect_options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    inspect_options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    inspect_options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    const auto& spec = vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root, spec, inspect_options);
    if (!report) return print_error(report.error(), json);
    auto config = vastloom::inference::deepseek_v4_reference_config_from_0731_gguf(report.value());
    if (!config) return print_error(config.error(), json);
    const std::size_t depth = std::min<std::size_t>(requested_layers.value(),
                                                    config.value().layer_count);

    // The prompt has to be encoded before the preflight, because how many experts a layer must hold
    // at once depends on how many tokens are routed through it.
    const auto prompt = option_value(args, "--prompt");
    const auto tokenizer_path = option_value(args, "--tokenizer");
    std::vector<std::uint32_t> token_ids;
    std::optional<vastloom::inference::Gpt2Tokenizer> tokenizer;
    if (prompt.has_value()) {
        if (!tokenizer_path.has_value()) {
            return print_error(
                vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                     "--prompt requires --tokenizer naming a store with the pinned "
                                     "DeepSeek tokenizer"),
                json);
        }
        auto loaded = vastloom::inference::Gpt2Tokenizer::load(cli_path(*tokenizer_path));
        if (!loaded) return print_error(loaded.error(), json);
        if (!loaded.value().is_deepseek_v4()) {
            return print_error(
                vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                     "--tokenizer must name the DeepSeek V4 tokenizer"),
                json);
        }
        tokenizer = loaded.take_value();
        std::string text = *prompt;
        if (has_flag(args, "--chat")) {
            auto framed = tokenizer->apply_deepseek_v4_chat_template({{"user", *prompt}}, true);
            if (!framed) return print_error(framed.error(), json);
            text = framed.take_value();
        }
        auto encoded = tokenizer->encode(text);
        if (!encoded) return print_error(encoded.error(), json);
        if (encoded.value().empty()) {
            return print_error(
                vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                     "The prompt encoded to no tokens"),
                json);
        }
        token_ids = encoded.take_value();
    } else {
        token_ids.push_back(static_cast<std::uint32_t>(token.value()));
    }

    // Exact resident cost of what is about to be loaded, from the catalog rather than an estimate.
    vastloom::ByteCount trunk_bytes = 0U;
    vastloom::ByteCount global_bytes = 0U;
    vastloom::ByteCount largest_expert = 0U;
    for (const auto& tensor : report.value().catalog.tensors) {
        if (tensor.role == vastloom::model::TensorRole::routed_expert) continue;
        if (!tensor.layer.has_value()) {
            global_bytes += tensor.source.bytes;
        } else if (*tensor.layer < depth) {
            trunk_bytes += tensor.source.bytes;
        }
    }
    for (const auto& bundle : report.value().catalog.experts) {
        largest_expert = std::max(largest_expert, bundle.bytes);
    }
    // A synchronous resolve hands the graph every expert the layer selected at once, so one layer's
    // cache must hold all of them: up to one distinct bundle per token per slot. Earlier layers
    // release theirs as the graph advances, so the peak is two layers' worth rather than all 43 -
    // the outgoing one may still be releasing while the incoming one loads.
    const auto per_layer_experts = std::min<std::size_t>(
        token_ids.size() * config.value().experts_per_token, config.value().expert_count);
    const auto cache_size = std::max<std::size_t>(expert_cache.value(), per_layer_experts);
    const auto expert_bytes = largest_expert * cache_size * std::min<std::size_t>(depth, 2U);
    const auto required = trunk_bytes + global_bytes + expert_bytes;
    const auto budget = policy.value().resources.ram_hard_limit;

    // Planning works from metadata alone; executing does not. Say which of the two this store
    // supports rather than letting a green preflight imply a runnable model.
    const bool payload_present = !report.value().catalog.metadata_only;
    if (json) {
        std::cout << "{\"ok\":" << (required <= budget ? "true" : "false")
                  << ",\"preflight\":{\"layers\":" << depth
                  << ",\"model_layers\":" << config.value().layer_count
                  << ",\"trunk_bytes\":" << trunk_bytes
                  << ",\"global_bytes\":" << global_bytes
                  << ",\"expert_cache_bytes\":" << expert_bytes
                  << ",\"required_bytes\":" << required
                  << ",\"ram_budget_bytes\":" << budget
                  << ",\"system_headroom_bytes\":"
                  << policy.value().resources.minimum_system_ram_available
                  << ",\"payload_present\":" << (payload_present ? "true" : "false")
                  << ",\"fits\":" << (required <= budget ? "true" : "false") << "}}" << '\n';
    } else {
        std::cout << "DeepSeek forward preflight\n"
                  << "Layers:               " << depth << " of " << config.value().layer_count
                  << '\n'
                  << "Layer trunks:         " << trunk_bytes << " bytes\n"
                  << "Globals:              " << global_bytes << " bytes\n"
                  << "Expert cache ceiling: " << expert_bytes << " bytes ("
                  << expert_cache.value() << " bundles)\n"
                  << "Required:             " << required << " bytes\n"
                  << "Safe RAM budget:      " << budget << " bytes\n"
                  << "Host headroom kept:   "
                  << policy.value().resources.minimum_system_ram_available << " bytes\n"
                  << "Weight payload:       "
                  << (payload_present ? "present" : "absent (planning only)") << '\n'
                  << "Verdict:              " << (required <= budget ? "fits" : "REFUSED") << '\n';
    }
    if (required > budget) {
        return print_error(
            vastloom::make_error(
                vastloom::ErrorCode::resource_exhausted,
                "This depth does not fit the safe RAM budget; reduce --layers or free memory",
                std::to_string(required) + " required, " + std::to_string(budget) + " available"),
            json);
    }
    if (has_flag(args, "--dry-run")) return 0;

    vastloom::safety::ResourceGovernor governor(policy.value().resources);
    auto layer_config = config.value();
    // The Governor's thread budget, not the machine's core count: the policy already decided how
    // much of the CPU this process may take while leaving the host usable.
    layer_config.cpu_threads = policy.value().cpu_threads;
    layer_config.layer_count = depth;
    layer_config.hash_layer_count = std::min(layer_config.hash_layer_count, depth);
    layer_config.mtp_layer_count = 0U;
    layer_config.mtp_compression_ratios.clear();
    layer_config.compression_ratios.resize(depth);
    if (layer_config.routed_expert_swiglu_limits.size() > depth) {
        layer_config.routed_expert_swiglu_limits.resize(depth);
    }
    if (layer_config.shared_expert_swiglu_limits.size() > depth) {
        layer_config.shared_expert_swiglu_limits.resize(depth);
    }

    vastloom::inference::DeepseekV4ReferenceGraph graph;
    graph.config = std::move(layer_config);

    // A GPU is optional. When one is present the Q8_0 matrices - the large attention projections -
    // run there and stay resident in VRAM; everything else keeps using the CPU reference. If the
    // driver is missing or the device refuses, the pass runs exactly as before.
    std::unique_ptr<vastloom::cuda::CudaMatvecAccelerator> accelerator;
    if (!has_flag(args, "--cpu-only")) {
        auto cuda_driver = vastloom::cuda::CudaDriver::load();
        if (cuda_driver) {
            vastloom::cuda::CudaMatvecAcceleratorOptions accelerator_options;
            accelerator_options.minimum_free_vram_after = 1U * vastloom::GiB;
            accelerator_options.maximum_resident_bytes =
                policy.value().resources.vram_hard_limit;
            // Attention contributes a handful of matrices per layer, but every routed expert is
            // two more, and a decode step may select different experts each token. Let VRAM be the
            // limit rather than a count sized for attention alone.
            accelerator_options.maximum_matrices = 8'192U;
            auto created = vastloom::cuda::CudaMatvecAccelerator::create(
                cuda_driver.take_value(), governor, accelerator_options);
            if (created) {
                accelerator = created.take_value();
                graph.accelerator = accelerator.get();
            }
        }
    }

    auto globals = vastloom::inference::deepseek_v4_load_0731_gguf_globals(
        report.value().catalog, config.value(), governor);
    if (!globals) return print_error(globals.error(), json);
    for (auto& [name, matrix] : globals.value().matrices) {
        graph.packed_matrices.emplace(name, std::move(matrix));
    }
    for (auto& [name, tensor] : globals.value().float_tensors) {
        graph.packed_float_tensors.emplace(name, std::move(tensor));
    }

    // Each layer keeps its own bounded expert cache: the 256 bundles of layer N are a different set
    // from layer N+1, and a shared cache would let one layer evict another's working set.
    std::vector<std::shared_ptr<vastloom::inference::DeepseekV4BoundedExpertCache>> caches;
    const auto load_started = std::chrono::steady_clock::now();
    for (std::size_t layer = 0U; layer < depth; ++layer) {
        auto safe = process_guard.value().sample_and_check();
        if (!safe) return print_error(safe.error(), json);
        auto trunk = vastloom::inference::deepseek_v4_load_0731_gguf_layer_trunk(
            report.value().catalog, graph.config, layer, governor);
        if (!trunk) return print_error(trunk.error(), json);
        for (auto& [name, matrix] : trunk.value().matrices) {
            graph.packed_matrices.emplace(name, std::move(matrix));
        }
        for (auto& [name, tensor] : trunk.value().float_tensors) {
            graph.packed_float_tensors.emplace(name, std::move(tensor));
        }
        for (auto& [name, tensor] : trunk.value().integer_tensors) {
            graph.packed_integer_tensors.emplace(name, std::move(tensor));
        }
        auto bindings = vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
            report.value().catalog, static_cast<std::uint32_t>(layer),
            report.value().experts_per_layer, graph.config.width,
            graph.config.intermediate_width);
        if (!bindings) return print_error(bindings.error(), json);
        vastloom::inference::DeepseekV4ExpertCacheOptions cache_options;
        cache_options.block_prefix = "layers." + std::to_string(layer) + ".ffn";
        cache_options.max_cached_experts = cache_size;
        cache_options.max_experts_per_request = per_layer_experts;
        auto cache = vastloom::inference::DeepseekV4BoundedExpertCache::create(
            bindings.take_value(), governor, cache_options);
        if (!cache) return print_error(cache.error(), json);

        // A layer's experts are dead the moment the next layer asks for its own, and the graph
        // walks layers in order. Releasing the earlier caches here caps expert residency at one
        // layer's working set instead of letting it accumulate across all 43 - which is what
        // otherwise makes the cost scale with prompt length. A release that fails leaves the memory
        // charged rather than pretending it is free; the Governor then refuses later work.
        auto binding = cache.value()->provider_binding();
        vastloom::inference::DeepseekV4ExpertProviderBinding staged;
        staged.max_experts_per_request = binding.max_experts_per_request;
        auto* live = &caches;
        staged.resolve = [live, layer, resolve = binding.resolve](
                             std::string_view block_prefix,
                             std::span<const std::uint32_t> ids) {
            for (std::size_t earlier = 0U; earlier < layer && earlier < live->size(); ++earlier) {
                [[maybe_unused]] const auto released = (*live)[earlier]->clear();
            }
            return resolve(block_prefix, ids);
        };
        graph.expert_providers.emplace(cache_options.block_prefix, std::move(staged));
        caches.push_back(cache.take_value());
    }
    const auto load_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_started).count();

    const auto started = std::chrono::steady_clock::now();
    const bool complete = depth == config.value().layer_count;

    // More than one new token means the decode loop, which reuses the KV state instead of
    // re-running the prompt. Each step routes a single token, so its expert working set is the
    // smallest the model ever needs.
    if (max_tokens.value() > 1U) {
        vastloom::inference::DeepseekV4ReferenceGenerationOptions options;
        options.max_new_tokens = max_tokens.value();
        options.temperature = 0.0F;
        // The driver caps a run at one hour, which at CPU speed is the real ceiling on how many
        // tokens a request can produce.
        options.timeout = std::chrono::hours(1);
        if (tokenizer.has_value()) {
            options.stop_token_ids.push_back(1U);  // <|end of sentence|>
        }
        auto generated = vastloom::inference::deepseek_v4_reference_generate(
            graph, token_ids, options, &process_guard.value());
        if (!generated) return print_error(generated.error(), json);
        const auto generate_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::string text;
        if (tokenizer.has_value()) {
            auto decoded = tokenizer->decode(generated.value().generated_ids, false);
            if (decoded) text = decoded.take_value();
        }
        const auto& stats = generated.value().stats;
        vastloom::cuda::CudaMatvecAcceleratorStats gpu_stats;
        if (accelerator != nullptr) gpu_stats = accelerator->stats();
        if (json) {
            std::cout << "{\"ok\":true,\"complete_model\":" << (complete ? "true" : "false")
                      << ",\"executed_layers\":" << depth
                      << ",\"model_layers\":" << config.value().layer_count
                      << ",\"prompt_tokens\":" << stats.prompt_tokens
                      << ",\"generated_tokens\":" << stats.generated_tokens
                      << ",\"finish_reason\":\""
                      << vastloom::inference::to_string(generated.value().finish_reason)
                      << "\",\"text\":\"" << vastloom::util::json_escape(text)
                      << "\",\"expert_bytes_read\":" << stats.logical_expert_bytes_read
                      << ",\"forward_passes\":" << stats.forward_passes
                      << ",\"resident_bytes\":" << required
                      << ",\"gpu_matvecs\":" << gpu_stats.accelerated
                      << ",\"gpu_declined\":" << gpu_stats.declined
                      << ",\"gpu_resident_bytes\":" << gpu_stats.resident_bytes
                      << ",\"gpu_kernel_ms\":" << gpu_stats.kernel_ms
                      << ",\"load_ms\":" << load_elapsed
                      << ",\"generate_ms\":" << generate_elapsed
                      << ",\"ms_per_token\":"
                      << (stats.generated_tokens == 0U
                              ? 0.0
                              : generate_elapsed / static_cast<double>(stats.generated_tokens))
                      << '}' << '\n';
        } else {
            std::cout << "Executed layers:      " << depth << " of "
                      << config.value().layer_count << '\n'
                      << "Prompt / generated:   " << stats.prompt_tokens << " / "
                      << stats.generated_tokens << " tokens ("
                      << vastloom::inference::to_string(generated.value().finish_reason) << ")\n"
                      << "Output:               " << text << '\n'
                      << "GPU matvecs:          " << gpu_stats.accelerated << " accelerated, "
                      << gpu_stats.declined << " on CPU ("
                      << vastloom::format_bytes(gpu_stats.resident_bytes) << " resident)\n"
                      << "Expert bytes read:    " << stats.logical_expert_bytes_read << '\n'
                      << "Weight load:          " << load_elapsed << " ms\n"
                      << "Generation:           " << generate_elapsed << " ms";
            if (stats.generated_tokens != 0U) {
                std::cout << " (" << generate_elapsed / static_cast<double>(stats.generated_tokens)
                          << " ms/token)";
            }
            std::cout << '\n';
        }
        return 0;
    }

    auto forward = vastloom::inference::deepseek_v4_reference_prefill(graph, token_ids);
    if (!forward) return print_error(forward.error(), json);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::string next_text;
    if (tokenizer.has_value()) {
        auto decoded = tokenizer->decode({forward.value().greedy_token_id}, false);
        if (decoded) next_text = decoded.take_value();
    }

    vastloom::ByteCount expert_bytes_read = 0U;
    std::size_t expert_misses = 0U;
    for (const auto& cache : caches) {
        const auto snapshot = cache->snapshot();
        expert_bytes_read += snapshot.logical_bytes_read;
        expert_misses += snapshot.misses;
    }

    if (json) {
        std::cout << "{\"ok\":true,\"complete_model\":" << (complete ? "true" : "false")
                  << ",\"executed_layers\":" << depth
                  << ",\"model_layers\":" << config.value().layer_count
                  << ",\"prompt_tokens\":" << token_ids.size()
                  << ",\"next_token\":" << forward.value().greedy_token_id
                  << ",\"next_text\":\"" << vastloom::util::json_escape(next_text)
                  << "\",\"logits\":" << forward.value().logits.size()
                  << ",\"expert_bytes_read\":" << expert_bytes_read
                  << ",\"expert_misses\":" << expert_misses
                  << ",\"resident_bytes\":" << required
                  << ",\"load_ms\":" << load_elapsed
                  << ",\"forward_ms\":" << elapsed << '}' << '\n';
    } else {
        std::cout << "Executed layers:      " << depth << " of " << config.value().layer_count
                  << (complete ? " (complete model)" : " (partial: not the model's output)") << '\n'
                  << "Prompt tokens:        " << token_ids.size() << '\n'
                  << "Next token:           " << forward.value().greedy_token_id;
        if (!next_text.empty()) std::cout << "  " << next_text;
        std::cout << '\n'
                  << "Expert bytes read:    " << expert_bytes_read << " in " << expert_misses
                  << " misses\n"
                  << "Weight load:          " << load_elapsed << " ms\n"
                  << "Forward:              " << elapsed << " ms\n";
    }
    return 0;
}

Result<void> write_olmoe_trace_atomic(const std::filesystem::path& path,
                                      const std::vector<vastloom::inference::ExpertTraceEvent>& events) {
    if (path.empty()) return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "OLMoE trace path is empty");
    std::error_code error;
    if (std::filesystem::exists(path, error)) {
        return vastloom::make_error(vastloom::ErrorCode::permission_denied, "Refusing to overwrite an existing OLMoE trace", path.string());
    }
    const auto partial = path.string() + ".partial-" + std::to_string(current_pid());
    {
        std::ofstream output(partial, std::ios::binary | std::ios::trunc);
        if (!output) return vastloom::make_error(vastloom::ErrorCode::io_error, "Unable to create OLMoE trace staging file", partial);
        output << "token,layer,expert,size_bytes,routing_weight,tier,hit,bytes_read,io_ms,"
                  "prefetched,prefetch_wait_ms,prefetch_overlap_ms,submitted_ms,queue_ms,ready_ms,"
                  "compute_started_ms,compute_completed_ms,route_order,reduction_order\n";
        output.precision(9);
        for (const auto& event : events) {
            output << event.token_position << ',' << event.layer << ',' << event.expert << ',' << (12U * vastloom::MiB) << ','
                   << event.routing_weight << ',' << (event.ram_cache_hit ? "ram" : "nvme") << ','
                   << (event.ram_cache_hit ? 1 : 0) << ',' << event.bytes_read << ',' << event.io_ms << ','
                   << (event.prefetched ? 1 : 0) << ',' << event.prefetch_wait_ms << ','
                   << event.prefetch_overlap_ms << ',' << event.submitted_ms << ',' << event.queue_ms << ','
                   << event.ready_ms << ',' << event.compute_started_ms << ',' << event.compute_completed_ms << ','
                   << event.route_order << ',' << event.reduction_order << '\n';
        }
        output.flush();
        if (!output) {
            std::filesystem::remove(partial, error);
            return vastloom::make_error(vastloom::ErrorCode::io_error, "Unable to flush OLMoE trace staging file", partial);
        }
    }
    std::filesystem::rename(partial, path, error);
    if (error) {
        std::filesystem::remove(partial, error);
        return vastloom::make_error(vastloom::ErrorCode::io_error, "Unable to publish OLMoE trace atomically", path.string());
    }
    return {};
}

int command_moe_logits(const std::vector<std::string>& args) {
    auto valid = validate_options(args, 1U,
                                  {"--text", "--top", "--cache", "--cache-bytes", "--io", "--io-cache", "--prefetch-depth",
                                   "--expert-read", "--io-timeout-ms", "--inject-io-fault", "--timeout-ms", "--trace"},
                                  {"--chat", "--no-verify", "--json"});
    const bool json = has_flag(args, "--json");
    if (!valid) return print_error(valid.error(), json);
    const auto text = option_value(args, "--text");
    if (!text) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "moe-logits requires --text"), json);
    auto top = optional_size(args, "--top", 10U);
    auto timeout = optional_size(args, "--timeout-ms", 600'000U);
    auto cache_bytes = optional_bytes(args, "--cache-bytes", 256U * vastloom::MiB);
    auto cache_policy = vastloom::inference::parse_expert_cache_policy(option_value(args, "--cache").value_or("lru"));
    auto io_mode = vastloom::inference::parse_expert_io_mode(option_value(args, "--io").value_or("sync"));
    auto io_cache_mode = vastloom::io::parse_io_cache_mode(
        option_value(args, "--io-cache").value_or("buffered"));
    auto read_mode = vastloom::inference::parse_expert_read_mode(
        option_value(args, "--expert-read").value_or("whole"));
    auto prefetch_depth = optional_size(args, "--prefetch-depth", 1U);
    auto io_timeout = optional_size(args, "--io-timeout-ms", 30'000U);
    auto io_fault = parse_io_fault(option_value(args, "--inject-io-fault").value_or("none"));
    if (!top) return print_error(top.error(), json);
    if (!timeout) return print_error(timeout.error(), json);
    if (!cache_bytes) return print_error(cache_bytes.error(), json);
    if (!cache_policy) return print_error(cache_policy.error(), json);
    if (!io_mode) return print_error(io_mode.error(), json);
    if (!io_cache_mode) return print_error(io_cache_mode.error(), json);
    if (!read_mode) return print_error(read_mode.error(), json);
    if (!prefetch_depth) return print_error(prefetch_depth.error(), json);
    if (!io_timeout) return print_error(io_timeout.error(), json);
    if (!io_fault) return print_error(io_fault.error(), json);
    if (top.value() == 0U || top.value() > 100U || timeout.value() == 0U || timeout.value() > 3'600'000U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "OLMoE top/timeout option is outside safe bounds"), json);
    }
    if (prefetch_depth.value() == 0U || prefetch_depth.value() > 2U ||
        io_timeout.value() == 0U || io_timeout.value() > 60'000U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "OLMoE prefetch/I/O-timeout option is outside safe bounds"), json);
    }
    if (io_mode.value() == vastloom::inference::ExpertIoMode::synchronous &&
        io_fault.value().kind != vastloom::io::IoFaultKind::none) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--inject-io-fault requires --io async"), json);
    }
    if (read_mode.value() == vastloom::inference::ExpertReadMode::staged &&
        io_mode.value() != vastloom::inference::ExpertIoMode::asynchronous) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--expert-read staged requires --io async"), json);
    }
    if (io_cache_mode.value() == vastloom::io::IoCacheMode::bypass &&
        io_mode.value() != vastloom::inference::ExpertIoMode::asynchronous) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--io-cache bypass requires --io async"), json);
    }
    if (io_cache_mode.value() == vastloom::io::IoCacheMode::bypass &&
        cache_policy.value() != vastloom::inference::ExpertCachePolicy::none) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--io-cache bypass requires --cache no-cache"), json);
    }

    const auto model_path = cli_path(args[0]);
    auto hardware = vastloom::hardware::HardwareProbe::collect(model_path);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);
    auto guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    if (!guard) return print_error(guard.error(), json);
    vastloom::safety::ResourceGovernor governor(policy.value().resources);
    auto model = vastloom::inference::OlmoeRuntimeModel::load(
        model_path, governor, {!has_flag(args, "--no-verify"), policy.value().cpu_threads});
    if (!model) return print_error(model.error(), json);
    auto checked = guard.value().sample_and_check();
    if (!checked) return print_error(checked.error(), json);

    std::string input = *text;
    if (has_flag(args, "--chat")) {
        auto formatted = model.value().tokenizer().apply_olmoe_chat_template({{"user", input}}, true);
        if (!formatted) return print_error(formatted.error(), json);
        input = formatted.take_value();
    }
    auto tokens = model.value().tokenizer().encode(input);
    if (!tokens) return print_error(tokens.error(), json);
    vastloom::inference::OlmoeForwardOptions options;
    options.cache_policy = cache_policy.value();
    options.expert_cache_bytes = cache_bytes.value();
    options.io_mode = io_mode.value();
    options.io_cache_mode = io_cache_mode.value();
    options.read_mode = read_mode.value();
    options.prefetch_depth = prefetch_depth.value();
    options.io_request_timeout = std::chrono::milliseconds(io_timeout.value());
    options.io_fault = io_fault.value();
    options.timeout = std::chrono::milliseconds(timeout.value());
    options.cancelled = [] { return real_generation_cancelled != 0; };
    real_generation_cancelled = 0;
    const auto previous_handler = std::signal(SIGINT, handle_real_interrupt);
    auto forwarded = model.value().forward(tokens.value(), options, governor, &guard.value());
    std::signal(SIGINT, previous_handler);
    if (!forwarded) return print_error(forwarded.error(), json);

    const auto trace_path = option_value(args, "--trace");
    if (trace_path) {
        auto written = write_olmoe_trace_atomic(cli_path(*trace_path), forwarded.value().trace);
        if (!written) return print_error(written.error(), json);
    }
    std::vector<std::size_t> order(forwarded.value().logits.size());
    std::iota(order.begin(), order.end(), 0U);
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top.value()), order.end(),
                      [&](std::size_t left, std::size_t right) {
                          if (forwarded.value().logits[left] != forwarded.value().logits[right]) {
                              return forwarded.value().logits[left] > forwarded.value().logits[right];
                          }
                          return left < right;
                      });
    const auto greedy_id = static_cast<std::uint32_t>(order.front());
    auto greedy_text = model.value().tokenizer().decode({greedy_id}, false);
    if (!greedy_text) return print_error(greedy_text.error(), json);
    if (json) {
        std::cout << "{\"ok\":true,\"input_tokens\":" << tokens.value().size()
                  << ",\"greedy_id\":" << greedy_id << ",\"greedy_text\":\""
                  << vastloom::util::json_escape(greedy_text.value()) << "\",\"top_logits\":[";
        for (std::size_t index = 0U; index < top.value(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << "{\"id\":" << order[index] << ",\"logit\":"
                      << forwarded.value().logits[order[index]] << '}';
        }
        std::cout << "],\"cache_policy\":\"" << vastloom::inference::to_string(options.cache_policy)
                   << "\",\"io_mode\":\"" << vastloom::inference::to_string(options.io_mode)
                   << "\",\"io_cache_mode\":\"" << vastloom::io::to_string(options.io_cache_mode)
                   << "\",\"io_cache_protocol\":\""
                   << (options.io_cache_mode == vastloom::io::IoCacheMode::bypass
                           ? "direct-unbuffered-v1"
                           : "system-buffered")
                   << "\",\"expert_read_mode\":\"" << vastloom::inference::to_string(options.read_mode)
                   << "\",\"prefetch_depth\":" << options.prefetch_depth
                  << ",\"expert_cache_hits\":" << forwarded.value().stats.expert_cache_hits
                  << ",\"expert_cache_misses\":" << forwarded.value().stats.expert_cache_misses
                  << ",\"expert_prefetches\":" << forwarded.value().stats.expert_prefetches
                  << ",\"expert_bytes_read\":" << forwarded.value().stats.expert_bytes_read
                  << ",\"expert_io_ms\":" << forwarded.value().stats.expert_io_ms
                   << ",\"expert_prefetch_wait_ms\":" << forwarded.value().stats.expert_prefetch_wait_ms
                   << ",\"expert_prefetch_overlap_ms\":" << forwarded.value().stats.expert_prefetch_overlap_ms
                   << ",\"staged_expert_reads\":" << forwarded.value().stats.staged_expert_reads
                   << ",\"staged_gate_up_bytes_read\":" << forwarded.value().stats.staged_gate_up_bytes_read
                   << ",\"staged_down_bytes_read\":" << forwarded.value().stats.staged_down_bytes_read
                   << ",\"staged_down_compute_overlap_ms\":"
                   << forwarded.value().stats.staged_down_compute_overlap_ms
                  << ",\"load_ms\":" << forwarded.value().stats.load_ms
                  << ",\"forward_ms\":" << forwarded.value().stats.forward_ms
                  << ",\"trace_events\":" << forwarded.value().trace.size()
                  << ",\"process_io\":{";
        if (forwarded.value().stats.process_io_available) {
            const auto& process_io = forwarded.value().stats.process_io;
            std::cout << "\"available\":true,\"source\":\""
                      << vastloom::util::json_escape(process_io.source)
                      << "\",\"read_operations\":" << process_io.read_operations
                      << ",\"process_read_bytes\":" << process_io.process_read_bytes
                      << ",\"storage_read_bytes_available\":"
                      << (process_io.storage_read_bytes_available ? "true" : "false")
                      << ",\"storage_read_bytes\":";
            if (process_io.storage_read_bytes_available) std::cout << process_io.storage_read_bytes;
            else std::cout << "null";
        } else {
            std::cout << "\"available\":false,\"error\":\""
                      << vastloom::util::json_escape(forwarded.value().stats.process_io_error) << '"';
        }
        std::cout << "},\"device_io\":{";
        if (forwarded.value().stats.device_io_available) {
            const auto& device_io = forwarded.value().stats.device_io;
            std::cout << "\"available\":true,\"source\":\""
                      << vastloom::util::json_escape(device_io.source)
                      << "\",\"device_id\":\""
                      << vastloom::util::json_escape(device_io.device_id)
                      << "\",\"scope\":\""
                      << vastloom::util::json_escape(device_io.scope)
                      << "\",\"read_operations\":" << device_io.read_operations
                      << ",\"read_bytes\":" << device_io.read_bytes;
        } else {
            std::cout << "\"available\":false,\"error\":\""
                      << vastloom::util::json_escape(forwarded.value().stats.device_io_error) << '"';
        }
        std::cout << "}}\n";
    } else {
        std::cout << "Greedy token: " << greedy_id << " (" << greedy_text.value() << ")\nTop logits:\n";
        for (std::size_t index = 0U; index < top.value(); ++index) {
            std::cout << "  " << order[index] << "  " << forwarded.value().logits[order[index]] << '\n';
        }
        std::cout << "Expert I/O: " << vastloom::format_bytes(forwarded.value().stats.expert_bytes_read)
                  << ", hits/misses " << forwarded.value().stats.expert_cache_hits << '/'
                   << forwarded.value().stats.expert_cache_misses << ", "
                   << vastloom::inference::to_string(options.io_mode) << '/'
                   << vastloom::io::to_string(options.io_cache_mode) << '/'
                   << vastloom::inference::to_string(options.read_mode) << " prefetches "
                  << forwarded.value().stats.expert_prefetches << ", overlap "
                  << forwarded.value().stats.expert_prefetch_overlap_ms << " ms, forward "
                  << forwarded.value().stats.forward_ms << " ms\n";
        if (forwarded.value().stats.process_io_available) {
            const auto& process_io = forwarded.value().stats.process_io;
            std::cout << "Process I/O: " << vastloom::format_bytes(process_io.process_read_bytes)
                      << " read-transfer bytes via " << process_io.source;
            if (process_io.storage_read_bytes_available) {
                std::cout << ", storage-layer reads "
                          << vastloom::format_bytes(process_io.storage_read_bytes);
            } else {
                std::cout << ", storage-layer bytes unavailable";
            }
            std::cout << '\n';
        } else {
            std::cout << "Process I/O counters unavailable: "
                      << forwarded.value().stats.process_io_error << '\n';
        }
        if (forwarded.value().stats.device_io_available) {
            const auto& device_io = forwarded.value().stats.device_io;
            std::cout << "Device I/O: " << vastloom::format_bytes(device_io.read_bytes)
                      << " read bytes via " << device_io.source << " ("
                      << device_io.device_id << ", " << device_io.scope << ")\n";
        } else {
            std::cout << "Device I/O counters unavailable: "
                      << forwarded.value().stats.device_io_error << '\n';
        }
    }
    return 0;
}

int command_tokenize(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "tokenize requires a model directory"), json);
    auto valid = validate_options(args, 1U, {"--text", "--system"}, {"--chat", "--json"});
    if (!valid) return print_error(valid.error(), json);
    const auto text = option_value(args, "--text");
    if (!text) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "tokenize requires --text"), json);
    const auto model_path = cli_path(args[0]);
    auto tokenizer = vastloom::inference::Gpt2Tokenizer::load(model_path);
    if (!tokenizer) return print_error(tokenizer.error(), json);
    std::string input = *text;
    if (has_flag(args, "--chat")) {
        std::vector<vastloom::inference::ChatMessage> messages;
        if (const auto system = option_value(args, "--system")) messages.push_back({"system", *system});
        messages.push_back({"user", input});
        const auto config_path = model_path / "config.json";
        auto olmoe = vastloom::inference::load_olmoe_config(config_path);
        auto formatted = tokenizer.value().is_deepseek_v4()
                             ? tokenizer.value().apply_deepseek_v4_chat_template(messages, true)
                             : (olmoe ? tokenizer.value().apply_olmoe_chat_template(messages, true)
                                      : tokenizer.value().apply_smol_chat_template(messages, true));
        if (!formatted) return print_error(formatted.error(), json);
        input = formatted.take_value();
    } else if (option_value(args, "--system")) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "tokenize --system requires --chat"), json);
    }
    auto tokens = tokenizer.value().encode(input);
    if (!tokens) return print_error(tokens.error(), json);
    auto decoded = tokenizer.value().decode(tokens.value(), false);
    if (!decoded) return print_error(decoded.error(), json);
    if (json) {
        std::cout << "{\"ok\":true,\"family\":\""
                  << (tokenizer.value().is_deepseek_v4() ? "deepseek_v4_byte_bpe" : "gpt2_byte_bpe")
                  << "\",\"token_count\":" << tokens.value().size() << ",\"ids\":[";
        for (std::size_t index = 0; index < tokens.value().size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << tokens.value()[index];
        }
        std::cout << "],\"decoded\":\"" << vastloom::util::json_escape(decoded.value()) << "\"}\n";
    } else {
        std::cout << "Tokens (" << tokens.value().size() << "):";
        for (const auto token : tokens.value()) std::cout << ' ' << token;
        std::cout << "\nDecoded: " << decoded.value() << '\n';
    }
    return 0;
}

int command_real_model_info(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "model-info requires a model directory"), json);
    auto valid = validate_options(args, 1U, {}, {"--no-verify", "--json"});
    if (!valid) return print_error(valid.error(), json);
    const auto model_path = cli_path(args[0]);
    auto imported = vastloom::model::verify_imported_checkpoint(model_path, !has_flag(args, "--no-verify"));
    if (!imported) return print_error(imported.error(), json);
    auto config = vastloom::inference::load_llama_config(model_path / "config.json");
    if (!config) return print_error(config.error(), json);
    auto tokenizer = vastloom::inference::Gpt2Tokenizer::load(model_path);
    if (!tokenizer) return print_error(tokenizer.error(), json);
    auto catalog = vastloom::model::inspect_safetensors_checkpoint(model_path);
    if (!catalog) return print_error(catalog.error(), json);
    if (json) {
        std::cout << "{\"ok\":true,\"model_id\":\"" << vastloom::util::json_escape(imported.value().model_id)
                  << "\",\"revision\":\"" << vastloom::util::json_escape(imported.value().revision)
                  << "\",\"hash_verified\":" << (imported.value().full_hash_verified ? "true" : "false")
                  << ",\"architecture\":\"LlamaForCausalLM\",\"vocab_size\":" << config.value().vocab_size
                  << ",\"hidden_size\":" << config.value().hidden_size
                  << ",\"intermediate_size\":" << config.value().intermediate_size
                  << ",\"layers\":" << config.value().num_hidden_layers
                  << ",\"attention_heads\":" << config.value().num_attention_heads
                  << ",\"kv_heads\":" << config.value().num_key_value_heads
                  << ",\"context_length\":" << config.value().max_position_embeddings
                  << ",\"tensor_count\":" << catalog.value().tensors.size()
                  << ",\"tensor_bytes\":" << catalog.value().total_tensor_bytes << "}\n";
    } else {
        std::cout << "Model:         " << imported.value().model_id << '\n'
                  << "Revision:      " << imported.value().revision << '\n'
                  << "Architecture:  LlamaForCausalLM\n"
                  << "Layers/width:  " << config.value().num_hidden_layers << '/' << config.value().hidden_size << '\n'
                  << "Heads/KV:      " << config.value().num_attention_heads << '/' << config.value().num_key_value_heads << '\n'
                  << "Context:       " << config.value().max_position_embeddings << '\n'
                  << "Tensors:       " << catalog.value().tensors.size() << '\n'
                  << "Tensor bytes:  " << vastloom::format_bytes(catalog.value().total_tensor_bytes) << '\n'
                  << "SHA-256:       " << (imported.value().full_hash_verified ? "verified" : "skipped") << '\n';
    }
    return 0;
}

struct RealRuntime {
    vastloom::safety::RuntimePolicy policy;
    std::unique_ptr<vastloom::safety::ResourceGovernor> governor;
    vastloom::safety::ProcessSafetyGuard process_guard;
};

Result<RealRuntime> start_real_runtime(const std::filesystem::path& model_path) {
    auto hardware = vastloom::hardware::HardwareProbe::collect(model_path);
    if (!hardware) return hardware.error();
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return policy.error();
    auto guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    if (!guard) return guard.error();
    RealRuntime runtime;
    runtime.policy = policy.value();
    runtime.governor = std::make_unique<vastloom::safety::ResourceGovernor>(runtime.policy.resources);
    runtime.process_guard = guard.take_value();
    return runtime;
}

Result<vastloom::inference::LlamaModel> load_real_model(const std::filesystem::path& model_path,
                                                        RealRuntime& runtime,
                                                        bool verify) {
    return vastloom::inference::LlamaModel::load(
        model_path, *runtime.governor, {verify, runtime.policy.cpu_threads});
}

int command_real_logits(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "real-logits requires a model directory"), json);
    auto valid = validate_options(args, 1U, {"--text", "--top"}, {"--chat", "--no-verify", "--json"});
    if (!valid) return print_error(valid.error(), json);
    const auto text = option_value(args, "--text");
    if (!text) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "real-logits requires --text"), json);
    auto top = optional_size(args, "--top", 10U);
    if (!top || top.value() == 0U || top.value() > 100U) return print_error(top ? vastloom::make_error(vastloom::ErrorCode::invalid_argument, "--top must be between 1 and 100") : top.error(), json);
    const auto model_path = cli_path(args[0]);
    auto runtime = start_real_runtime(model_path);
    if (!runtime) return print_error(runtime.error(), json);
    auto model = load_real_model(model_path, runtime.value(), !has_flag(args, "--no-verify"));
    if (!model) return print_error(model.error(), json);
    std::string input = *text;
    if (has_flag(args, "--chat")) {
        auto formatted = model.value().tokenizer().apply_smol_chat_template({{"user", input}}, true);
        if (!formatted) return print_error(formatted.error(), json);
        input = formatted.take_value();
    }
    auto tokens = model.value().tokenizer().encode(input);
    if (!tokens) return print_error(tokens.error(), json);
    auto logits = model.value().logits(tokens.value(), *runtime.value().governor, &runtime.value().process_guard);
    if (!logits) return print_error(logits.error(), json);
    std::vector<std::uint32_t> indices(logits.value().size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(top.value()), indices.end(),
                      [&logits](std::uint32_t left, std::uint32_t right) { return logits.value()[left] > logits.value()[right]; });
    indices.resize(top.value());
    if (json) {
        std::cout << "{\"ok\":true,\"prompt_tokens\":" << tokens.value().size()
                  << ",\"load_ms\":" << model.value().load_ms() << ",\"top_logits\":[";
        for (std::size_t index = 0; index < indices.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto token = indices[index];
            std::cout << "{\"id\":" << token << ",\"logit\":" << logits.value()[token] << '}';
        }
        std::cout << "]}\n";
    } else {
        std::cout << "Prompt tokens: " << tokens.value().size() << "\nLoad: " << model.value().load_ms() << " ms\n";
        for (const auto token : indices) std::cout << token << '\t' << logits.value()[token] << '\n';
    }
    return 0;
}

int command_real_chat(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "chat requires a model directory"), json);
    auto valid = validate_options(args, 1U,
                                  {"--prompt", "--max-tokens", "--top-k", "--seed", "--temperature", "--timeout-ms", "--system"},
                                  {"--json", "--no-stream", "--no-stats", "--no-verify"});
    if (!valid) return print_error(valid.error(), json);
    const auto prompt = option_value(args, "--prompt");
    if (!prompt && json) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Interactive real-model chat does not support --json"), json);
    auto max_tokens = optional_size(args, "--max-tokens", 64U);
    auto top_k = optional_size(args, "--top-k", 40U);
    auto seed = optional_u64(args, "--seed", 1U);
    auto temperature = optional_float(args, "--temperature", 0.0F);
    auto timeout_ms = optional_size(args, "--timeout-ms", 300'000U);
    if (!max_tokens) return print_error(max_tokens.error(), json);
    if (!top_k) return print_error(top_k.error(), json);
    if (!seed) return print_error(seed.error(), json);
    if (!temperature) return print_error(temperature.error(), json);
    if (!timeout_ms || timeout_ms.value() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return print_error(timeout_ms ? vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Invalid --timeout-ms") : timeout_ms.error(), json);
    }
    const auto model_path = cli_path(args[0]);
    auto runtime = start_real_runtime(model_path);
    if (!runtime) return print_error(runtime.error(), json);
    auto model = load_real_model(model_path, runtime.value(), !has_flag(args, "--no-verify"));
    if (!model) return print_error(model.error(), json);
    std::vector<vastloom::inference::ChatMessage> messages;
    if (const auto system = option_value(args, "--system")) messages.push_back({"system", *system});
    vastloom::inference::LlamaGenerationOptions options;
    options.max_new_tokens = max_tokens.value();
    options.top_k = top_k.value();
    options.seed = seed.value();
    options.temperature = temperature.value();
    options.timeout = std::chrono::milliseconds(static_cast<std::int64_t>(timeout_ms.value()));
    options.cancelled = [] { return real_generation_cancelled != 0; };
    const bool stream_enabled = !json && !has_flag(args, "--no-stream");
    static_cast<void>(std::signal(SIGINT, handle_real_interrupt));

    const auto run_generation = [&](std::vector<vastloom::inference::ChatMessage>& history,
                                    bool interactive) -> Result<vastloom::inference::LlamaGenerationResult> {
        real_generation_cancelled = 0;
        if (interactive) std::cout << "assistant> " << std::flush;
        auto generated = model.value().generate(history, options, *runtime.value().governor,
                                                &runtime.value().process_guard,
                                                stream_enabled ? [](std::string_view piece) { std::cout << piece << std::flush; }
                                                               : std::function<void(std::string_view)>{});
        if (generated && stream_enabled) std::cout << '\n';
        if (generated && !stream_enabled && interactive) std::cout << generated.value().text << '\n';
        return generated;
    };

    const auto print_stats = [&args](const vastloom::inference::LlamaGenerationResult& generated) {
        if (has_flag(args, "--no-stats")) return;
        const std::string finish = generated.stats.stopped_on_eos ? "end_of_turn" :
                                   (generated.stats.stopped_on_context_limit ? "context_limit" : "max_tokens");
        std::cerr << "[load " << generated.stats.load_ms << " ms; prefill "
                  << generated.stats.prompt_eval_ms << " ms; decode "
                  << generated.stats.tokens_per_second << " tok/s; " << finish << "]\n";
    };

    if (prompt) {
        messages.push_back({"user", *prompt});
        auto generated = run_generation(messages, false);
        if (!generated) return print_error(generated.error(), json);
        const std::string finish = generated.value().stats.stopped_on_eos ? "end_of_turn" :
                                   (generated.value().stats.stopped_on_context_limit ? "context_limit" : "max_tokens");
        if (json) {
            std::cout << "{\"ok\":true,\"text\":\"" << vastloom::util::json_escape(generated.value().text)
                      << "\",\"generated_ids\":[";
            for (std::size_t index = 0; index < generated.value().generated_ids.size(); ++index) {
                if (index != 0U) std::cout << ',';
                std::cout << generated.value().generated_ids[index];
            }
            std::cout << "],\"finish_reason\":\"" << finish
                      << "\",\"stats\":{\"prompt_tokens\":" << generated.value().stats.prompt_tokens
                      << ",\"generated_tokens\":" << generated.value().stats.generated_tokens
                      << ",\"load_ms\":" << generated.value().stats.load_ms
                      << ",\"prompt_eval_ms\":" << generated.value().stats.prompt_eval_ms
                      << ",\"generation_ms\":" << generated.value().stats.generation_ms
                      << ",\"tokens_per_second\":" << generated.value().stats.tokens_per_second << "}}\n";
        } else {
            if (!stream_enabled) std::cout << generated.value().text << '\n';
            print_stats(generated.value());
        }
        return 0;
    }

    const std::vector<vastloom::inference::ChatMessage> initial_messages = messages;
    std::optional<vastloom::inference::LlamaGenerationStats> last_stats;
    std::cout << "Vastloom real-model chat (SmolLM2 CPU reference)\n"
              << "Commands: /exit, /clear, /stats, /model, /help\n";
    while (true) {
        std::cout << "you> " << std::flush;
        std::string input;
        if (!std::getline(std::cin, input)) break;
        if (input == "/exit" || input == "/quit") break;
        if (input == "/clear") {
            messages = initial_messages;
            std::cout << "Conversation history cleared.\n";
            continue;
        }
        if (input == "/stats") {
            if (!last_stats) std::cout << "No generation statistics yet.\n";
            else std::cout << "Prompt tokens: " << last_stats->prompt_tokens << ", generated: "
                           << last_stats->generated_tokens << ", decode: " << last_stats->tokens_per_second << " tok/s\n";
            continue;
        }
        if (input == "/model") {
            std::cout << "HuggingFaceTB/SmolLM2-135M-Instruct @ 12fd25f (CPU BF16/FP32 reference)\n";
            continue;
        }
        if (input == "/help") {
            std::cout << "Enter a message, or use /exit, /clear, /stats, /model. Ctrl+C cancels generation safely.\n";
            continue;
        }
        if (input.empty()) continue;
        messages.push_back({"user", std::move(input)});
        auto generated = run_generation(messages, true);
        if (!generated) {
            messages.pop_back();
            if (real_generation_cancelled != 0) {
                std::cout << "Generation cancelled; history was not modified.\n";
                continue;
            }
            return print_error(generated.error());
        }
        messages.push_back({"assistant", generated.value().text});
        last_stats = generated.value().stats;
        print_stats(generated.value());
    }
    return 0;
}

int command_real_self_test(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "self-test requires a model directory"), json);
    auto valid = validate_options(args, 1U, {}, {"--no-verify", "--json"});
    if (!valid) return print_error(valid.error(), json);
    const auto model_path = cli_path(args[0]);
    auto runtime = start_real_runtime(model_path);
    if (!runtime) return print_error(runtime.error(), json);
    auto model = load_real_model(model_path, runtime.value(), !has_flag(args, "--no-verify"));
    if (!model) return print_error(model.error(), json);

    struct Case {
        std::string name;
        std::vector<vastloom::inference::ChatMessage> messages;
        std::string expected;
    };
    const std::vector<Case> cases{
        {"hello", {{"user", "Hello"}}, "Hello! How can I help you today?"},
        {"italian", {{"user", "Qual è la capitale d'Italia?"}}, "Italia."},
        {"moon", {{"user", "Write one short sentence about the Moon."}},
         "The Moon, a celestial body, is a mysterious and awe-inspiring sight that captivates our imagination and inspires awe in us."},
        {"multi-turn",
         {{"user", "Hello"}, {"assistant", "Hello! How can I help you today?"}, {"user", "What was my first word?"}},
         "My first word was \"I\"."},
    };
    vastloom::inference::LlamaGenerationOptions options;
    options.max_new_tokens = 64U;
    options.temperature = 0.0F;
    options.top_k = 40U;
    options.timeout = std::chrono::minutes(2);
    bool passed = true;
    struct Outcome { std::string name; std::string expected; std::string actual; bool passed; };
    std::vector<Outcome> outcomes;
    const auto started = std::chrono::steady_clock::now();
    for (const auto& test : cases) {
        auto generated = model.value().generate(test.messages, options, *runtime.value().governor, &runtime.value().process_guard);
        if (!generated) return print_error(generated.error(), json);
        const bool case_passed = generated.value().text == test.expected;
        passed = passed && case_passed;
        outcomes.push_back({test.name, test.expected, generated.value().text, case_passed});
    }
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (json) {
        std::cout << "{\"ok\":" << (passed ? "true" : "false") << ",\"model_id\":\"HuggingFaceTB/SmolLM2-135M-Instruct\""
                  << ",\"elapsed_ms\":" << elapsed << ",\"cases\":[";
        for (std::size_t index = 0; index < outcomes.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto& outcome = outcomes[index];
            std::cout << "{\"name\":\"" << outcome.name << "\",\"expected\":\""
                      << vastloom::util::json_escape(outcome.expected) << "\",\"actual\":\""
                      << vastloom::util::json_escape(outcome.actual) << "\",\"passed\":"
                      << (outcome.passed ? "true" : "false") << '}';
        }
        std::cout << "]}\n";
    } else {
        for (const auto& outcome : outcomes) {
            std::cout << (outcome.passed ? "PASS " : "FAIL ") << outcome.name << ": " << outcome.actual << '\n';
        }
        std::cout << "Result: " << (passed ? "PASS" : "FAIL") << " (" << elapsed << " ms)\n";
    }
    return passed ? 0 : 1;
}

int command_heartbeat_write(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument, "heartbeat-write requires a path"));
    auto state = parse_safety_state(option_value(args, "--state").value_or("normal"));
    if (!state) return print_error(state.error());
    vastloom::safety::HeartbeatRecord heartbeat;
    heartbeat.pid = current_pid();
    heartbeat.sequence = 1;
    heartbeat.unix_time_ms = vastloom::safety::unix_time_ms();
    heartbeat.state = state.value();
    auto written = vastloom::safety::write_heartbeat_atomic(cli_path(args[0]), heartbeat);
    if (!written) return print_error(written.error());
    std::cout << "Wrote atomic heartbeat to " << args[0] << '\n';
    return 0;
}

int dispatch_cli(const std::vector<std::string>& arguments) {
    try {
        if (arguments.size() < 2U) {
            print_help();
            return 0;
        }
        const std::string& command = arguments[1];
        const std::vector<std::string> args(arguments.begin() + 2, arguments.end());

        if (command == "chat") {
            std::error_code error;
            if (!args.empty() && std::filesystem::is_directory(cli_path(args[0]), error)) return command_real_chat(args);
            return vastloom::cli::command_chat(args);
        }
        if (command == "model-info") {
            std::error_code error;
            if (!args.empty() && std::filesystem::is_directory(cli_path(args[0]), error)) return command_real_model_info(args);
            return vastloom::cli::command_model_info(args);
        }
        if (command == "self-test") {
            std::error_code error;
            if (!args.empty() && std::filesystem::is_directory(cli_path(args[0]), error)) return command_real_self_test(args);
            return vastloom::cli::command_self_test(args);
        }
        if (command == "doctor") return command_doctor(args);
        if (command == "cuda-info") return command_cuda_info(args);
        if (command == "inspect") return command_inspect(args);
        if (command == "verify") return command_verify(args);
        if (command == "plan") return command_plan(args);
        if (command == "trace-sim") return command_trace_sim(args);
        if (command == "io-smoke") return command_io_smoke(args);
        if (command == "store-init") return command_store_init(args);
        if (command == "store-info") return command_store_info(args);
        if (command == "runtime-policy") return command_runtime_policy(args);
        if (command == "tokenize") return command_tokenize(args);
        if (command == "real-logits") return command_real_logits(args);
        if (command == "moe-inspect") return command_moe_inspect(args);
        if (command == "deepseek-inspect") return command_deepseek_inspect(args);
        if (command == "deepseek-gguf-inspect") return command_deepseek_gguf_inspect(args);
        if (command == "deepseek-gguf-checkpoint") return command_deepseek_gguf_checkpoint(args);
        if (command == "deepseek-gguf-probe-plan") return command_deepseek_gguf_probe_plan(args);
        if (command == "deepseek-gguf-global-plan") return command_deepseek_gguf_global_plan(args);
        if (command == "range-artifact-inspect") return command_range_artifact_inspect(args);
        if (command == "deepseek-gguf-range-bind") return command_deepseek_gguf_range_bind(args);
        if (command == "deepseek-layer-probe") return command_deepseek_layer_probe(args);
        if (command == "deepseek-forward") return command_deepseek_forward(args);
        if (command == "deepseek-gguf-global-bind") return command_deepseek_gguf_global_bind(args);
        if (command == "moe-logits") return command_moe_logits(args);
        if (command == "heartbeat-write") return command_heartbeat_write(args);
        if (command == "version" || command == "--version") {
            std::cout << vastloom::kVersion << '\n';
            return 0;
        }
        if (command == "help" || command == "--help" || command == "-h") {
            print_help();
            return 0;
        }
        std::cerr << "Unknown command: " << command << "\n\n";
        print_help();
        return 64;
    } catch (const std::exception& exception) {
        std::cerr << "FATAL: " << exception.what() << '\n';
        return 70;
    }
}

#if defined(_WIN32)
std::string wide_to_utf8(std::wstring_view value) {
    if (value.empty()) return {};
    if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Windows command-line argument is too long");
    }
    const int input_size = static_cast<int>(value.size());
    const int required = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), input_size, nullptr, 0, nullptr, nullptr);
    if (required <= 0) {
        throw std::runtime_error("Unable to convert a Windows command-line argument to UTF-8");
    }
    std::string result(static_cast<std::size_t>(required), '\0');
    const int written = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        value.data(),
        input_size,
        result.data(),
        required,
        nullptr,
        nullptr);
    if (written != required) {
        throw std::runtime_error("Unable to convert a Windows command-line argument to UTF-8");
    }
    return result;
}
#endif

}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
    static_cast<void>(SetConsoleCP(CP_UTF8));
    static_cast<void>(SetConsoleOutputCP(CP_UTF8));
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            const std::wstring_view argument =
                argv[index] == nullptr ? std::wstring_view{} : std::wstring_view(argv[index]);
            arguments.push_back(wide_to_utf8(argument));
        }
        return dispatch_cli(arguments);
    } catch (const std::exception& exception) {
        std::cerr << "FATAL: " << exception.what() << '\n';
        return 70;
    }
}
#else
int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            arguments.emplace_back(argv[index] == nullptr ? "" : argv[index]);
        }
        return dispatch_cli(arguments);
    } catch (const std::exception& exception) {
        std::cerr << "FATAL: " << exception.what() << '\n';
        return 70;
    }
}
#endif
