#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
#include "vastloom/api/loopback_server.hpp"
#include "vastloom/api/openai_protocol.hpp"
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
#include "vastloom/inference/ggml_compute.hpp"
#include "vastloom/inference/deepseek_v4_preflight.hpp"
#include "vastloom/inference/qwen3_8_flash_next.hpp"
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
#include "vastloom/util/json.hpp"
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
                              std::initializer_list<std::string_view> flags,
                              std::initializer_list<std::string_view> repeatable_value_options = {}) {
    if (args.size() < positional_count) return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Missing positional argument");
    std::set<std::string> seen;
    for (std::size_t index = positional_count; index < args.size(); ++index) {
        const std::string_view argument = args[index];
        const bool is_value = std::find(value_options.begin(), value_options.end(), argument) != value_options.end();
        const bool is_flag = std::find(flags.begin(), flags.end(), argument) != flags.end();
        const bool is_repeatable = std::find(repeatable_value_options.begin(), repeatable_value_options.end(), argument) != repeatable_value_options.end();
        if (!is_value && !is_flag && !is_repeatable) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Unknown or unexpected option", std::string(argument));
        }
        if (!is_repeatable && !seen.insert(std::string(argument)).second) {
            return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Duplicate option", std::string(argument));
        }
        if (is_value || is_repeatable) {
            if (index + 1U >= args.size()) return vastloom::make_error(vastloom::ErrorCode::invalid_argument, "Option requires a value", std::string(argument));
            ++index;
        }
    }
    return {};
}

std::vector<std::string> all_option_values(const std::vector<std::string>& args, std::string_view name) {
    std::vector<std::string> values;
    for (std::size_t index = 0; index + 1U < args.size(); ++index) {
        if (args[index] == name) values.push_back(args[index + 1U]);
    }
    return values;
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
        << "  serve MODEL_OR_MANIFEST [--port N] [--no-verify]\n"
        << "      Expose an OpenAI-compatible chat-completions endpoint on 127.0.0.1 only.\n\n"
        << "  module-init MODEL OUT --name NAME [--rank N] [--scale F] [--seed N] [--layers N]\n"
        << "      [--targets LIST] [--zero-a] [--json]\n"
        << "      Write a deterministic synthetic capability module sized to a pinned SmolLM2 checkpoint.\n\n"
        << "  module-fit MODEL TRANSCRIPT OUT --name NAME [--rank N] [--steps N] [--lr F] [--reg F]\n"
        << "      [--seed N] [--max-records N] [--site lm-head|hidden] [--no-verify] [--json]\n"
        << "      Distil a capability module from a teacher transcript artifact.\n\n"
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
        << "  deepseek-expert-redundancy MODEL_DIRECTORY [--layer N] [--experts N] [--rows N]"
           " [--json]\n"
        << "      Measure whether one layer's routed experts share a component worth factoring"
           " out.\n\n"
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
                                   "--iq1s-smoke", "--iq3xxs-smoke", "--iq2xxs-smoke", "--q5k-smoke", "--q6k-smoke"});
    if (!valid) return print_error(valid.error(), json);
    const bool allocation_smoke_requested =
        option_value(args, "--allocation-smoke").has_value();
    const bool rmsnorm_smoke_requested = has_flag(args, "--rmsnorm-smoke");
    const bool matvec_smoke_requested = has_flag(args, "--matvec-smoke");
    const bool iq1s_smoke_requested = has_flag(args, "--iq1s-smoke");
    const bool iq3xxs_smoke_requested = has_flag(args, "--iq3xxs-smoke");
    const bool iq2xxs_smoke_requested = has_flag(args, "--iq2xxs-smoke");
    const bool q5k_smoke_requested = has_flag(args, "--q5k-smoke");
    const bool q6k_smoke_requested = has_flag(args, "--q6k-smoke");
    if ((allocation_smoke_requested || rmsnorm_smoke_requested) &&
        (matvec_smoke_requested || iq1s_smoke_requested || iq3xxs_smoke_requested ||
         iq2xxs_smoke_requested || q5k_smoke_requested || q6k_smoke_requested)) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "the matvec parity gates run on their own"),
            json);
    }
    if (allocation_smoke_requested && rmsnorm_smoke_requested) {
        return print_error(vastloom::make_error(
                               vastloom::ErrorCode::invalid_argument,
                               "--allocation-smoke and --rmsnorm-smoke are mutually exclusive"),
                           json);
    }
    const bool operation_requested = allocation_smoke_requested || rmsnorm_smoke_requested ||
                                     matvec_smoke_requested || iq1s_smoke_requested ||
                                     iq3xxs_smoke_requested || iq2xxs_smoke_requested ||
                                     q5k_smoke_requested || q6k_smoke_requested;
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
    // The ceiling is a sanity bound on the argument, not the safety mechanism: every byte still
    // passes the Governor, which refuses in advance anything the host cannot afford. It was 64 MiB,
    // which made the smoke unable to answer the question it is now needed for - how much device
    // memory this platform will actually part with, and what that costs in host commit.
    if (allocation_smoke_requested &&
        (smoke_bytes.value() == 0U || smoke_bytes.value() > 24U * vastloom::GiB)) {
        return print_error(vastloom::make_error(
                               vastloom::ErrorCode::invalid_argument,
                               "--allocation-smoke must be between 1 byte and 24 GiB"),
                           json);
    }

    auto driver = vastloom::cuda::CudaDriver::load();
    if (!driver) return print_error(driver.error(), json);
    auto probe = driver.value().probe();
    if (!probe) return print_error(probe.error(), json);

    bool allocation_smoke_passed = false;
    vastloom::ByteCount smoke_commit_before = 0U;
    vastloom::ByteCount smoke_commit_after = 0U;
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
    bool q6k_smoke_passed = false;
    double q6k_max_abs_error = 0.0;
    double q6k_kernel_ms = 0.0;
    double q6k_cpu_ms = 0.0;
    bool q5k_smoke_passed = false;
    double q5k_max_abs_error = 0.0;
    double q5k_kernel_ms = 0.0;
    double q5k_cpu_ms = 0.0;
    bool iq2xxs_smoke_passed = false;
    double iq2xxs_max_abs_error = 0.0;
    double iq2xxs_kernel_ms = 0.0;
    double iq2xxs_cpu_ms = 0.0;
    bool iq3xxs_smoke_passed = false;
    double iq3xxs_max_abs_error = 0.0;
    double iq3xxs_kernel_ms = 0.0;
    double iq3xxs_cpu_ms = 0.0;
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
            // Host commit either side of a device allocation, because on Windows the two are
            // believed to move together roughly one for one - and that belief is what caps the
            // accelerator's ceiling and keeps the 6.39 GB trunk from ever being resident. Measured
            // rather than assumed: the difference between the two samples is the coupling.
            const auto commit_before = vastloom::safety::process_peak_commit_bytes();
            auto allocation = driver.value().allocate(
                0, smoke_bytes.value(), minimum_free.value(), governor, "cuda-info-smoke");
            if (!allocation) return print_error(allocation.error(), json);
            const auto commit_after = vastloom::safety::process_peak_commit_bytes();
            if (commit_before && commit_after) {
                smoke_commit_before = commit_before.value();
                smoke_commit_after = commit_after.value();
            }
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
        if (iq3xxs_smoke_requested) {
            // The experts' down projection, the last storage type in the routed path without a
            // device kernel.
            constexpr std::size_t iq3_rows = 2'048U;
            constexpr std::size_t iq3_columns = 4'096U;
            constexpr std::size_t iq3_block_elements = 256U;
            constexpr std::size_t iq3_block_bytes = 98U;
            constexpr double iq3_tolerance = 1.0e-3;
            const std::size_t iq3_blocks_per_row = iq3_columns / iq3_block_elements;
            const ByteCount iq3_matrix_bytes =
                static_cast<ByteCount>(iq3_rows) * iq3_blocks_per_row * iq3_block_bytes;
            const auto iq3_grid = vastloom::inference::iq3_xxs_grid_table();
            const ByteCount iq3_grid_bytes = iq3_grid.size_bytes();
            const ByteCount iq3_input_bytes =
                static_cast<ByteCount>(iq3_columns) * sizeof(float);
            const ByteCount iq3_output_bytes = static_cast<ByteCount>(iq3_rows) * sizeof(float);

            auto caller_host = governor.try_reserve(
                vastloom::safety::ResourceKind::scratch,
                iq3_matrix_bytes + iq3_input_bytes + 2U * iq3_output_bytes,
                "cuda-iq3xxs-smoke-host");
            if (!caller_host) return print_error(caller_host.error(), json);

            std::vector<std::byte> encoded(static_cast<std::size_t>(iq3_matrix_bytes));
            for (std::size_t index = 0U; index < encoded.size(); ++index) {
                encoded[index] = static_cast<std::byte>((index * 53U + 7U) % 251U);
            }
            for (std::size_t block = 0U; block * iq3_block_bytes < encoded.size(); ++block) {
                // Finite but untidy, for the reason spelled out in the IQ1_S gate above.
                const auto exponent = static_cast<std::uint16_t>(0x0CU + (block % 5U));
                const auto mantissa = static_cast<std::uint16_t>((block * 6'421U + 29U) & 0x03FFU);
                const auto scale_bits =
                    static_cast<std::uint16_t>((exponent << 10U) | mantissa);
                std::memcpy(encoded.data() + block * iq3_block_bytes, &scale_bits,
                            sizeof(scale_bits));
            }
            std::vector<float> iq3_input(iq3_columns);
            for (std::size_t index = 0U; index < iq3_input.size(); ++index) {
                const auto scrambled =
                    static_cast<std::uint32_t>((index * 2'246'822'519ULL + 61ULL) % 1'000'003ULL);
                iq3_input[index] = static_cast<float>(scrambled) / 1'000'003.0F - 0.5F;
            }

            std::vector<float> iq3_expected(iq3_rows);
            vastloom::inference::QuantizedKernelLimits iq3_limits;
            iq3_limits.max_elements = iq3_rows * iq3_columns;
            iq3_limits.max_threads = policy.value().cpu_threads;
            const auto iq3_cpu_started = std::chrono::steady_clock::now();
            auto iq3_cpu = vastloom::inference::matvec_iq3_xxs_f32(
                encoded, iq3_rows, iq3_columns, iq3_input, iq3_expected, iq3_limits);
            iq3xxs_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - iq3_cpu_started).count();
            if (!iq3_cpu) return print_error(iq3_cpu.error(), json);

            const ByteCount iq3_grid_offset = iq3_matrix_bytes;
            const ByteCount iq3_input_offset = iq3_grid_offset + iq3_grid_bytes;
            const ByteCount iq3_output_offset = iq3_input_offset + iq3_input_bytes;
            auto iq3_device = driver.value().allocate(
                0, iq3_output_offset + iq3_output_bytes, minimum_free.value(), governor,
                "cuda-matvec-iq3-xxs-smoke");
            if (!iq3_device) return print_error(iq3_device.error(), json);
            auto iq3_allocation = iq3_device.take_value();
            auto copied = driver.value().copy_from_host(iq3_allocation, 0U, encoded);
            if (!copied) return print_error(copied.error(), json);
            copied = driver.value().copy_from_host(
                iq3_allocation, iq3_grid_offset, std::as_bytes(iq3_grid));
            if (!copied) return print_error(copied.error(), json);

            std::vector<float> iq3_actual(iq3_rows);
            const auto iq3_started = std::chrono::steady_clock::now();
            auto iq3_launched = driver.value().matvec_iq3_xxs_with_transfer(
                iq3_allocation, 0U, iq3_grid_offset, iq3_input_offset, iq3_output_offset,
                iq3_rows, iq3_blocks_per_row,
                std::as_bytes(std::span<const float>(iq3_input)),
                std::as_writable_bytes(std::span<float>(iq3_actual)));
            if (!iq3_launched) return print_error(iq3_launched.error(), json);
            iq3xxs_kernel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - iq3_started).count();
            auto iq3_released = iq3_allocation.release();
            if (!iq3_released) return print_error(iq3_released.error(), json);

            for (std::size_t index = 0U; index < iq3_actual.size(); ++index) {
                if (!std::isfinite(iq3_actual[index])) {
                    return print_error(vastloom::make_error(
                                           vastloom::ErrorCode::integrity_error,
                                           "CUDA IQ3_XXS matvec produced a non-finite value"),
                                       json);
                }
                iq3xxs_max_abs_error = std::max(
                    iq3xxs_max_abs_error,
                    std::abs(static_cast<double>(iq3_actual[index]) -
                             static_cast<double>(iq3_expected[index])));
            }
            if (iq3xxs_max_abs_error > iq3_tolerance) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA IQ3_XXS matvec does not match the CPU reference",
                                       std::to_string(iq3xxs_max_abs_error)),
                                   json);
            }
            iq3xxs_smoke_passed = true;
        }
        if (iq2xxs_smoke_requested) {
            // The most-declined storage type in the measured breakdown.
            constexpr std::size_t iq2_rows = 2'048U;
            constexpr std::size_t iq2_columns = 4'096U;
            constexpr std::size_t iq2_block_elements = 256U;
            constexpr std::size_t iq2_block_bytes = 66U;
            constexpr double iq2_tolerance = 1.0e-3;
            const std::size_t iq2_blocks_per_row = iq2_columns / iq2_block_elements;
            const ByteCount iq2_matrix_bytes =
                static_cast<ByteCount>(iq2_rows) * iq2_blocks_per_row * iq2_block_bytes;
            const auto iq2_grid = vastloom::inference::iq2_xxs_grid_table();
            const ByteCount iq2_grid_bytes = iq2_grid.size_bytes();
            const ByteCount iq2_input_bytes =
                static_cast<ByteCount>(iq2_columns) * sizeof(float);
            const ByteCount iq2_output_bytes = static_cast<ByteCount>(iq2_rows) * sizeof(float);

            auto caller_host = governor.try_reserve(
                vastloom::safety::ResourceKind::scratch,
                iq2_matrix_bytes + iq2_input_bytes + 2U * iq2_output_bytes,
                "cuda-iq2xxs-smoke-host");
            if (!caller_host) return print_error(caller_host.error(), json);

            std::vector<std::byte> encoded(static_cast<std::size_t>(iq2_matrix_bytes));
            for (std::size_t index = 0U; index < encoded.size(); ++index) {
                encoded[index] = static_cast<std::byte>((index * 61U + 17U) % 251U);
            }
            for (std::size_t block = 0U; block * iq2_block_bytes < encoded.size(); ++block) {
                // Finite but untidy, for the reason spelled out in the IQ1_S gate above.
                const auto exponent = static_cast<std::uint16_t>(0x0CU + (block % 5U));
                const auto mantissa = static_cast<std::uint16_t>((block * 5'791U + 41U) & 0x03FFU);
                const auto scale_bits =
                    static_cast<std::uint16_t>((exponent << 10U) | mantissa);
                std::memcpy(encoded.data() + block * iq2_block_bytes, &scale_bits,
                            sizeof(scale_bits));
            }
            std::vector<float> iq2_input(iq2_columns);
            for (std::size_t index = 0U; index < iq2_input.size(); ++index) {
                const auto scrambled =
                    static_cast<std::uint32_t>((index * 3'141'592'653ULL + 79ULL) % 1'000'003ULL);
                iq2_input[index] = static_cast<float>(scrambled) / 1'000'003.0F - 0.5F;
            }

            std::vector<float> iq2_expected(iq2_rows);
            vastloom::inference::QuantizedKernelLimits iq2_limits;
            iq2_limits.max_elements = iq2_rows * iq2_columns;
            iq2_limits.max_threads = policy.value().cpu_threads;
            const auto iq2_cpu_started = std::chrono::steady_clock::now();
            auto iq2_cpu = vastloom::inference::matvec_iq2_xxs_f32(
                encoded, iq2_rows, iq2_columns, iq2_input, iq2_expected, iq2_limits);
            iq2xxs_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - iq2_cpu_started).count();
            if (!iq2_cpu) return print_error(iq2_cpu.error(), json);

            // The grid is read with 64-bit loads, so it starts on an eight-byte boundary.
            const ByteCount iq2_grid_offset = (iq2_matrix_bytes + 7U) / 8U * 8U;
            const ByteCount iq2_input_offset = iq2_grid_offset + iq2_grid_bytes;
            const ByteCount iq2_output_offset = iq2_input_offset + iq2_input_bytes;
            auto iq2_device = driver.value().allocate(
                0, iq2_output_offset + iq2_output_bytes, minimum_free.value(), governor,
                "cuda-matvec-iq2-xxs-smoke");
            if (!iq2_device) return print_error(iq2_device.error(), json);
            auto iq2_allocation = iq2_device.take_value();
            auto copied = driver.value().copy_from_host(iq2_allocation, 0U, encoded);
            if (!copied) return print_error(copied.error(), json);
            copied = driver.value().copy_from_host(
                iq2_allocation, iq2_grid_offset, std::as_bytes(iq2_grid));
            if (!copied) return print_error(copied.error(), json);

            std::vector<float> iq2_actual(iq2_rows);
            const auto iq2_started = std::chrono::steady_clock::now();
            auto iq2_launched = driver.value().matvec_iq2_xxs_with_transfer(
                iq2_allocation, 0U, iq2_grid_offset, iq2_input_offset, iq2_output_offset,
                iq2_rows, iq2_blocks_per_row,
                std::as_bytes(std::span<const float>(iq2_input)),
                std::as_writable_bytes(std::span<float>(iq2_actual)));
            if (!iq2_launched) return print_error(iq2_launched.error(), json);
            iq2xxs_kernel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - iq2_started).count();
            auto iq2_released = iq2_allocation.release();
            if (!iq2_released) return print_error(iq2_released.error(), json);

            for (std::size_t index = 0U; index < iq2_actual.size(); ++index) {
                if (!std::isfinite(iq2_actual[index])) {
                    return print_error(vastloom::make_error(
                                           vastloom::ErrorCode::integrity_error,
                                           "CUDA IQ2_XXS matvec produced a non-finite value"),
                                       json);
                }
                iq2xxs_max_abs_error = std::max(
                    iq2xxs_max_abs_error,
                    std::abs(static_cast<double>(iq2_actual[index]) -
                             static_cast<double>(iq2_expected[index])));
            }
            if (iq2xxs_max_abs_error > iq2_tolerance) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA IQ2_XXS matvec does not match the CPU reference",
                                       std::to_string(iq2xxs_max_abs_error)),
                                   json);
            }
            iq2xxs_smoke_passed = true;
        }
        if (q5k_smoke_requested) {
            // The shared expert's storage, active on every token of every layer.
            constexpr std::size_t q5k_rows = 2'048U;
            constexpr std::size_t q5k_columns = 4'096U;
            constexpr std::size_t q5k_block_elements = 256U;
            constexpr std::size_t q5k_block_bytes = 176U;
            constexpr double q5k_tolerance = 1.0e-3;
            const std::size_t q5k_blocks_per_row = q5k_columns / q5k_block_elements;
            const ByteCount q5k_matrix_bytes =
                static_cast<ByteCount>(q5k_rows) * q5k_blocks_per_row * q5k_block_bytes;
            const ByteCount q5k_input_bytes =
                static_cast<ByteCount>(q5k_columns) * sizeof(float);
            const ByteCount q5k_output_bytes = static_cast<ByteCount>(q5k_rows) * sizeof(float);

            auto caller_host = governor.try_reserve(
                vastloom::safety::ResourceKind::scratch,
                q5k_matrix_bytes + q5k_input_bytes + 2U * q5k_output_bytes,
                "cuda-q5k-smoke-host");
            if (!caller_host) return print_error(caller_host.error(), json);

            std::vector<std::byte> encoded(static_cast<std::size_t>(q5k_matrix_bytes));
            for (std::size_t index = 0U; index < encoded.size(); ++index) {
                encoded[index] = static_cast<std::byte>((index * 47U + 23U) % 251U);
            }
            for (std::size_t block = 0U; block * q5k_block_bytes < encoded.size(); ++block) {
                // Both the scale and the minimum are FP16 and both must be finite; vary them for
                // the reason spelled out in the IQ1_S gate above.
                const auto exponent = static_cast<std::uint16_t>(0x0CU + (block % 5U));
                const auto mantissa = static_cast<std::uint16_t>((block * 4'513U + 7U) & 0x03FFU);
                const auto scale_bits =
                    static_cast<std::uint16_t>((exponent << 10U) | mantissa);
                const auto minimum_bits = static_cast<std::uint16_t>(
                    ((0x0BU + (block % 4U)) << 10U) | ((block * 2'207U + 19U) & 0x03FFU));
                std::memcpy(encoded.data() + block * q5k_block_bytes, &scale_bits,
                            sizeof(scale_bits));
                std::memcpy(encoded.data() + block * q5k_block_bytes + 2U, &minimum_bits,
                            sizeof(minimum_bits));
            }
            std::vector<float> q5k_input(q5k_columns);
            for (std::size_t index = 0U; index < q5k_input.size(); ++index) {
                const auto scrambled =
                    static_cast<std::uint32_t>((index * 1'664'525ULL + 1'013ULL) % 1'000'003ULL);
                q5k_input[index] = static_cast<float>(scrambled) / 1'000'003.0F - 0.5F;
            }

            std::vector<float> q5k_expected(q5k_rows);
            vastloom::inference::QuantizedKernelLimits q5k_limits;
            q5k_limits.max_elements = q5k_rows * q5k_columns;
            q5k_limits.max_threads = policy.value().cpu_threads;
            const auto q5k_cpu_started = std::chrono::steady_clock::now();
            auto q5k_cpu = vastloom::inference::matvec_q5_k_f32(
                encoded, q5k_rows, q5k_columns, q5k_input, q5k_expected, q5k_limits);
            q5k_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - q5k_cpu_started).count();
            if (!q5k_cpu) return print_error(q5k_cpu.error(), json);

            const ByteCount q5k_input_offset = q5k_matrix_bytes;
            const ByteCount q5k_output_offset = q5k_input_offset + q5k_input_bytes;
            auto q5k_device = driver.value().allocate(
                0, q5k_output_offset + q5k_output_bytes, minimum_free.value(), governor,
                "cuda-matvec-q5-k-smoke");
            if (!q5k_device) return print_error(q5k_device.error(), json);
            auto q5k_allocation = q5k_device.take_value();
            auto copied = driver.value().copy_from_host(q5k_allocation, 0U, encoded);
            if (!copied) return print_error(copied.error(), json);

            std::vector<float> q5k_actual(q5k_rows);
            const auto q5k_started = std::chrono::steady_clock::now();
            auto q5k_launched = driver.value().matvec_q5_k_with_transfer(
                q5k_allocation, 0U, q5k_input_offset, q5k_output_offset,
                q5k_rows, q5k_blocks_per_row,
                std::as_bytes(std::span<const float>(q5k_input)),
                std::as_writable_bytes(std::span<float>(q5k_actual)));
            if (!q5k_launched) return print_error(q5k_launched.error(), json);
            q5k_kernel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - q5k_started).count();
            auto q5k_released = q5k_allocation.release();
            if (!q5k_released) return print_error(q5k_released.error(), json);

            for (std::size_t index = 0U; index < q5k_actual.size(); ++index) {
                if (!std::isfinite(q5k_actual[index])) {
                    return print_error(vastloom::make_error(
                                           vastloom::ErrorCode::integrity_error,
                                           "CUDA Q5_K matvec produced a non-finite value"),
                                       json);
                }
                q5k_max_abs_error = std::max(
                    q5k_max_abs_error,
                    std::abs(static_cast<double>(q5k_actual[index]) -
                             static_cast<double>(q5k_expected[index])));
            }
            if (q5k_max_abs_error > q5k_tolerance) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA Q5_K matvec does not match the CPU reference",
                                       std::to_string(q5k_max_abs_error)),
                                   json);
            }
            q5k_smoke_passed = true;
        }
        if (q6k_smoke_requested) {
            // The last trunk storage type without a kernel, and with it every trunk matrix the
            // streaming window reloads becomes one the device can hold instead.
            constexpr std::size_t q6k_rows = 2'048U;
            constexpr std::size_t q6k_columns = 4'096U;
            constexpr std::size_t q6k_block_elements = 256U;
            constexpr std::size_t q6k_block_bytes = 210U;
            constexpr double q6k_tolerance = 1.0e-3;
            const std::size_t q6k_blocks_per_row = q6k_columns / q6k_block_elements;
            const ByteCount q6k_matrix_bytes =
                static_cast<ByteCount>(q6k_rows) * q6k_blocks_per_row * q6k_block_bytes;
            const ByteCount q6k_input_bytes =
                static_cast<ByteCount>(q6k_columns) * sizeof(float);
            const ByteCount q6k_output_bytes = static_cast<ByteCount>(q6k_rows) * sizeof(float);

            auto caller_host = governor.try_reserve(
                vastloom::safety::ResourceKind::scratch,
                q6k_matrix_bytes + q6k_input_bytes + 2U * q6k_output_bytes,
                "cuda-q6k-smoke-host");
            if (!caller_host) return print_error(caller_host.error(), json);

            std::vector<std::byte> encoded(static_cast<std::size_t>(q6k_matrix_bytes));
            for (std::size_t index = 0U; index < encoded.size(); ++index) {
                encoded[index] = static_cast<std::byte>((index * 43U + 31U) % 251U);
            }
            for (std::size_t block = 0U; block * q6k_block_bytes < encoded.size(); ++block) {
                // The FP16 scale lives at the end of a Q6_K block, not the start.
                const auto exponent = static_cast<std::uint16_t>(0x0BU + (block % 5U));
                const auto mantissa = static_cast<std::uint16_t>((block * 3'187U + 11U) & 0x03FFU);
                const auto scale_bits =
                    static_cast<std::uint16_t>((exponent << 10U) | mantissa);
                std::memcpy(encoded.data() + block * q6k_block_bytes + 208U, &scale_bits,
                            sizeof(scale_bits));
            }
            std::vector<float> q6k_input(q6k_columns);
            for (std::size_t index = 0U; index < q6k_input.size(); ++index) {
                const auto scrambled =
                    static_cast<std::uint32_t>((index * 22'695'477ULL + 53ULL) % 1'000'003ULL);
                q6k_input[index] = static_cast<float>(scrambled) / 1'000'003.0F - 0.5F;
            }

            std::vector<float> q6k_expected(q6k_rows);
            vastloom::inference::QuantizedKernelLimits q6k_limits;
            q6k_limits.max_elements = q6k_rows * q6k_columns;
            q6k_limits.max_threads = policy.value().cpu_threads;
            const auto q6k_cpu_started = std::chrono::steady_clock::now();
            auto q6k_cpu = vastloom::inference::matvec_q6_k_f32(
                encoded, q6k_rows, q6k_columns, q6k_input, q6k_expected, q6k_limits);
            q6k_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - q6k_cpu_started).count();
            if (!q6k_cpu) return print_error(q6k_cpu.error(), json);

            const ByteCount q6k_input_offset = q6k_matrix_bytes;
            const ByteCount q6k_output_offset = q6k_input_offset + q6k_input_bytes;
            auto q6k_device = driver.value().allocate(
                0, q6k_output_offset + q6k_output_bytes, minimum_free.value(), governor,
                "cuda-matvec-q6-k-smoke");
            if (!q6k_device) return print_error(q6k_device.error(), json);
            auto q6k_allocation = q6k_device.take_value();
            auto copied = driver.value().copy_from_host(q6k_allocation, 0U, encoded);
            if (!copied) return print_error(copied.error(), json);

            std::vector<float> q6k_actual(q6k_rows);
            const auto q6k_started = std::chrono::steady_clock::now();
            auto q6k_launched = driver.value().matvec_q6_k_with_transfer(
                q6k_allocation, 0U, q6k_input_offset, q6k_output_offset,
                q6k_rows, q6k_blocks_per_row,
                std::as_bytes(std::span<const float>(q6k_input)),
                std::as_writable_bytes(std::span<float>(q6k_actual)));
            if (!q6k_launched) return print_error(q6k_launched.error(), json);
            q6k_kernel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - q6k_started).count();
            auto q6k_released = q6k_allocation.release();
            if (!q6k_released) return print_error(q6k_released.error(), json);

            for (std::size_t index = 0U; index < q6k_actual.size(); ++index) {
                if (!std::isfinite(q6k_actual[index])) {
                    return print_error(vastloom::make_error(
                                           vastloom::ErrorCode::integrity_error,
                                           "CUDA Q6_K matvec produced a non-finite value"),
                                       json);
                }
                q6k_max_abs_error = std::max(
                    q6k_max_abs_error,
                    std::abs(static_cast<double>(q6k_actual[index]) -
                             static_cast<double>(q6k_expected[index])));
            }
            if (q6k_max_abs_error > q6k_tolerance) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::integrity_error,
                                       "CUDA Q6_K matvec does not match the CPU reference",
                                       std::to_string(q6k_max_abs_error)),
                                   json);
            }
            q6k_smoke_passed = true;
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
                          std::to_string(minimum_free.value()) +
                          ",\"host_commit_before\":" + std::to_string(smoke_commit_before) +
                          ",\"host_commit_after\":" + std::to_string(smoke_commit_after) + "}}";
            } else if (iq1s_smoke_requested) {
                output += ",\"matvec_iq1_s_smoke\":{\"passed\":";
                output += iq1s_smoke_passed ? "true" : "false";
                output += ",\"rows\":2048,\"columns\":4096,\"max_abs_error\":" +
                          std::to_string(iq1s_max_abs_error) +
                          ",\"kernel_ms\":" + std::to_string(iq1s_kernel_ms) +
                          ",\"cpu_ms\":" + std::to_string(iq1s_cpu_ms) + "}}";
            } else if (iq3xxs_smoke_requested) {
                output += ",\"matvec_iq3_xxs_smoke\":{\"passed\":";
                output += iq3xxs_smoke_passed ? "true" : "false";
                output += ",\"rows\":2048,\"columns\":4096,\"max_abs_error\":" +
                          std::to_string(iq3xxs_max_abs_error) +
                          ",\"kernel_ms\":" + std::to_string(iq3xxs_kernel_ms) +
                          ",\"cpu_ms\":" + std::to_string(iq3xxs_cpu_ms) + "}}";
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
        } else if (q6k_smoke_requested) {
            std::cout << "Q6_K matvec:    " << (q6k_smoke_passed ? "PASS" : "FAIL")
                      << " (2048x4096, max abs error " << q6k_max_abs_error << ", GPU "
                      << q6k_kernel_ms << " ms vs CPU " << q6k_cpu_ms << " ms, released)\n";
        } else if (q5k_smoke_requested) {
            std::cout << "Q5_K matvec:    " << (q5k_smoke_passed ? "PASS" : "FAIL")
                      << " (2048x4096, max abs error " << q5k_max_abs_error << ", GPU "
                      << q5k_kernel_ms << " ms vs CPU " << q5k_cpu_ms << " ms, released)\n";
        } else if (iq2xxs_smoke_requested) {
            std::cout << "IQ2_XXS matvec: " << (iq2xxs_smoke_passed ? "PASS" : "FAIL")
                      << " (2048x4096, max abs error " << iq2xxs_max_abs_error << ", GPU "
                      << iq2xxs_kernel_ms << " ms vs CPU " << iq2xxs_cpu_ms << " ms, released)\n";
        } else if (iq3xxs_smoke_requested) {
            std::cout << "IQ3_XXS matvec: " << (iq3xxs_smoke_passed ? "PASS" : "FAIL")
                      << " (2048x4096, max abs error " << iq3xxs_max_abs_error << ", GPU "
                      << iq3xxs_kernel_ms << " ms vs CPU " << iq3xxs_cpu_ms << " ms, released)\n";
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

// Qwen3.8-Flash-Next checkpoint inspection over the canonical Q1-Q4 reader
// path: config.json -> model.safetensors.index.json -> optional first-shard
// header. Metadata only; no tensor payload is read.
int command_qwen38_inspect(const std::vector<std::string>& args) {
    auto valid = validate_options(args, 1U, {}, {"--json", "--load-header"});
    const bool json = has_flag(args, "--json");
    if (!valid) return print_error(valid.error(), json);

    const auto model_path = cli_path(args[0]);
    const auto config_path = model_path / "config.json";
    auto cfg = vastloom::inference::load_qwen38_flash_next_config(config_path);
    if (!cfg) return print_error(cfg.error(), json);

    const auto index_path = model_path / "model.safetensors.index.json";
    auto inspected = vastloom::inference::inspect_qwen38_flash_next_safetensors(
        index_path, cfg.value(), has_flag(args, "--load-header"));
    if (!inspected) return print_error(inspected.error(), json);

    auto validated = vastloom::inference::qwen38_validate_catalog(inspected.value());
    if (!validated) return print_error(validated.error(), json);
    const auto& r = validated.value();

    if (json) {
        std::cout << "{\n"
                  << "  \"model_id\": \"" << r.model_id << "\",\n"
                  << "  \"shards\": " << r.total_shards << ",\n"
                  << "  \"tensors\": " << r.total_tensors << ",\n"
                  << "  \"fp8_block_scaled_experts\": " << r.fp8_block_scaled_experts << ",\n"
                  << "  \"ngram_tensors\": " << r.ngram_tensor_count << ",\n"
                  << "  \"gated_residual_global\": " << (r.has_gated_residual_global ? "true" : "false") << ",\n"
                  << "  \"gated_residual_per_layer\": " << (r.has_gated_residual_per_layer ? "true" : "false") << ",\n"
                  << "  \"layers\": " << r.layers.size() << ",\n"
                  << "  \"vision_tensors\": " << r.vision_tensors << ",\n"
                  << "  \"mtp_tensors\": " << r.mtp_tensors << ",\n"
                  << "  \"global_tensors\": " << r.global_tensors << ",\n"
                  << "  \"shard_header_loaded\": " << (r.shard_loaded ? "true" : "false") << "\n"
                  << "}\n";
    } else {
        std::cout << "Qwen3.8-Flash-Next checkpoint inspection\n"
                  << "  model: " << r.model_id << '\n'
                  << "  shards: " << r.total_shards << '\n'
                  << "  tensors: " << r.total_tensors << '\n'
                  << "  fp8 block-scaled experts: " << r.fp8_block_scaled_experts << '\n'
                  << "  ngram tensors: " << r.ngram_tensor_count << '\n'
                  << "  gated residual: "
                  << (r.has_gated_residual_global && r.has_gated_residual_per_layer ? "yes" : "no") << '\n'
                  << "  layers: " << r.layers.size() << '\n'
                  << "  vision tensors: " << r.vision_tensors << '\n'
                  << "  mtp tensors: " << r.mtp_tensors << '\n'
                  << "  global tensors: " << r.global_tensors << '\n'
                  << "  shard header loaded: " << (r.shard_loaded ? "yes" : "no") << '\n';
    }
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

// Picks the pinned checkpoint identity matching what is on disk.
//
// Each quantization of the same model is pinned separately, on purpose: the pin exists so a
// checkpoint cannot quietly become a different one, and that promise is per-quantization. A
// directory holding UD-IQ2_XXS is a different identity from one holding UD-IQ1_S, not a
// corrupted version of it. When nothing matches, the first identity is returned so the failure
// comes from the hash gate, which names the object that did not match, rather than from a vaguer
// complaint about the directory.
const vastloom::inference::DeepseekV4GgufCheckpointSpec& pinned_spec_for(
    const std::filesystem::path& checkpoint_root) {
    const auto* found =
        vastloom::inference::pinned_deepseek_v4_flash_0731_checkpoint_for(checkpoint_root);
    return found != nullptr
        ? *found
        : vastloom::inference::pinned_deepseek_v4_flash_0731_iq1s_checkpoint();
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
    const auto& spec = pinned_spec_for(checkpoint_root);
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
    const auto& spec = pinned_spec_for(checkpoint_root);
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
// Asks whether the routed experts of one layer share a common component worth factoring out.
//
// The reason to ask: a token reads six experts of two hundred and fifty-six per layer, and that
// traffic is what sets the cost of any higher-precision checkpoint. If an expert were a shared base
// plus a small individual part, the base could stay resident and only the parts would stream - and
// the saving would grow with precision, because a two-bit expert is already near its entropy while
// an eight-bit one carries its redundancy intact.
//
// It is a question about the weights, not about inference, so it is answerable without a correct
// forward pass and without downloading anything. The answer is a correlation: experts that share
// structure have large off-diagonal cosines, experts that do not have cosines near zero, and the
// second answer closes the whole direction.
int command_deepseek_expert_redundancy(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args, 1U, {"--layer", "--experts", "--rows", "--context", "--batch-size"}, {"--json"});
    if (!valid) return print_error(valid.error(), json);
    if (args.empty() || vastloom::util::starts_with(args.front(), "--")) {
        return print_error(
            vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                 "deepseek-expert-redundancy requires one checkpoint directory"),
            json);
    }
    auto layer = optional_size(args, "--layer", 0U);
    auto experts = optional_size(args, "--experts", 16U);
    // A sample of rows rather than the whole matrix. Correlation between 8.4-million-element
    // matrices is estimated to three digits from a few hundred rows, and the whole thing would cost
    // 33 MB per expert to hold.
    auto rows = optional_size(args, "--rows", 256U);
    auto context = optional_size(args, "--context", 4'096U);
    auto batch_size = optional_size(args, "--batch-size", 1U);
    if (!layer || !experts || !rows || !context || !batch_size) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "Expert redundancy bounds are invalid"),
                           json);
    }
    if (layer.value() >= 128U || experts.value() < 2U || experts.value() > 64U ||
        rows.value() == 0U || rows.value() > 4'096U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "Expert redundancy bounds are outside their limits"),
                           json);
    }

    const auto checkpoint_root = cli_path(args.front());
    auto hardware = vastloom::hardware::HardwareProbe::collect(checkpoint_root);
    if (!hardware) return print_error(hardware.error(), json);
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    if (!policy) return print_error(policy.error(), json);
    auto process_guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    if (!process_guard) return print_error(process_guard.error(), json);
    vastloom::safety::ResourceGovernor governor(policy.value().resources);

    vastloom::inference::DeepseekV4GgufCheckpointInspectOptions inspect_options;
    inspect_options.verify_hashes = true;
    inspect_options.context_tokens = static_cast<std::uint32_t>(context.value());
    inspect_options.batch_size = static_cast<std::uint32_t>(batch_size.value());
    inspect_options.safe_ram_cap_bytes = policy.value().resources.host_memory_hard_limit;
    inspect_options.safe_vram_cap_bytes = policy.value().resources.vram_hard_limit;
    const auto& spec = pinned_spec_for(checkpoint_root);
    auto report = vastloom::inference::inspect_deepseek_v4_gguf_checkpoint(
        checkpoint_root, spec, inspect_options);
    if (!report) return print_error(report.error(), json);
    auto config = vastloom::inference::deepseek_v4_reference_config_from_0731_gguf(report.value());
    if (!config) return print_error(config.error(), json);
    if (layer.value() >= config.value().layer_count) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "Layer is outside the checkpoint",
                                                std::to_string(layer.value())),
                           json);
    }

    auto bindings = vastloom::inference::deepseek_v4_make_gguf_expert_bindings(
        report.value().catalog, static_cast<std::uint32_t>(layer.value()),
        report.value().experts_per_layer, config.value().width,
        config.value().intermediate_width);
    if (!bindings) return print_error(bindings.error(), json);
    vastloom::inference::DeepseekV4ExpertCacheOptions cache_options;
    cache_options.block_prefix = "layers." + std::to_string(layer.value()) + ".ffn";
    cache_options.max_cached_experts = 1U;
    cache_options.max_experts_per_request = 1U;
    auto cache = vastloom::inference::DeepseekV4BoundedExpertCache::create(
        bindings.take_value(), governor, cache_options);
    if (!cache) return print_error(cache.error(), json);
    auto resolve = cache.value()->provider_binding().resolve;

    // One expert at a time, keeping only the sampled rows. Peak residency is the sample rather than
    // the experts, so this asks its question inside a few hundred megabytes.
    const std::size_t sampled = std::min<std::size_t>(rows.value(), config.value().intermediate_width);
    const std::size_t columns = config.value().width;
    std::vector<std::vector<float>> samples;
    std::string storage_type;
    try {
        samples.reserve(experts.value());
        for (std::size_t index = 0U; index < experts.value(); ++index) {
            auto safe = process_guard.value().sample_and_check();
            if (!safe) return print_error(safe.error(), json);
            const std::array<std::uint32_t, 1U> wanted{static_cast<std::uint32_t>(index)};
            auto resolved = resolve(cache_options.block_prefix, wanted);
            if (!resolved) return print_error(resolved.error(), json);
            if (resolved.value().empty()) {
                return print_error(vastloom::make_error(vastloom::ErrorCode::not_found,
                                                        "Expert bundle did not resolve",
                                                        std::to_string(index)),
                                   json);
            }
            const auto& w1 = resolved.value().front().w1;
            if (storage_type.empty()) storage_type = std::to_string(w1.ggml_type);
            auto payload = vastloom::inference::deepseek_v4_packed_matrix_payload(w1, "redundancy");
            if (!payload) return print_error(payload.error(), json);
            std::vector<float> flat(sampled * columns, 0.0F);
            std::vector<float> row(w1.storage_columns, 0.0F);
            // Rows spread across the matrix rather than the first few, so the sample is not a
            // corner of it.
            const std::size_t stride = std::max<std::size_t>(1U, w1.rows / sampled);
            for (std::size_t taken = 0U; taken < sampled; ++taken) {
                const std::size_t source_row = std::min(taken * stride, w1.rows - 1U);
                auto decoded = vastloom::inference::dequantize_ggml_row_f32(
                    w1.ggml_type, payload.value(), w1.rows, w1.storage_columns, source_row, row);
                if (!decoded) return print_error(decoded.error(), json);
                std::copy_n(row.begin(), columns, flat.begin() +
                            static_cast<std::ptrdiff_t>(taken * columns));
            }
            samples.push_back(std::move(flat));
        }
    } catch (const std::bad_alloc&) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::resource_exhausted,
                                                "Unable to hold the expert sample"),
                           json);
    }

    const std::size_t elements = sampled * columns;
    std::vector<double> norms(samples.size(), 0.0);
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        double sum = 0.0;
        for (const float value : samples[index]) sum += static_cast<double>(value) * value;
        norms[index] = std::sqrt(sum);
    }

    // The shared component, if there is one, is what every expert has in common: their mean.
    std::vector<double> mean(elements, 0.0);
    for (const auto& sample : samples) {
        for (std::size_t index = 0U; index < elements; ++index) mean[index] += sample[index];
    }
    for (double& value : mean) value /= static_cast<double>(samples.size());
    double mean_energy = 0.0;
    for (const double value : mean) mean_energy += value * value;

    // What is left of each expert once the shared component is removed. Near 1.0 means nothing was
    // shared and there is no base to factor out; well below 1.0 means there is.
    double residual_min = 2.0;
    double residual_max = 0.0;
    double residual_sum = 0.0;
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        double sum = 0.0;
        for (std::size_t position = 0U; position < elements; ++position) {
            const double delta = static_cast<double>(samples[index][position]) - mean[position];
            sum += delta * delta;
        }
        const double ratio = norms[index] > 0.0 ? sum / (norms[index] * norms[index]) : 0.0;
        residual_min = std::min(residual_min, ratio);
        residual_max = std::max(residual_max, ratio);
        residual_sum += ratio;
    }
    const double residual_mean = residual_sum / static_cast<double>(samples.size());

    double cosine_min = 1.0;
    double cosine_max = -1.0;
    double cosine_sum = 0.0;
    std::size_t pairs = 0U;
    for (std::size_t left = 0U; left < samples.size(); ++left) {
        for (std::size_t right = left + 1U; right < samples.size(); ++right) {
            double dot = 0.0;
            for (std::size_t position = 0U; position < elements; ++position) {
                dot += static_cast<double>(samples[left][position]) *
                       static_cast<double>(samples[right][position]);
            }
            const double denominator = norms[left] * norms[right];
            const double cosine = denominator > 0.0 ? dot / denominator : 0.0;
            cosine_min = std::min(cosine_min, cosine);
            cosine_max = std::max(cosine_max, cosine);
            cosine_sum += cosine;
            ++pairs;
        }
    }
    const double cosine_mean = pairs == 0U ? 0.0 : cosine_sum / static_cast<double>(pairs);

    if (json) {
        std::cout << "{\"ok\":true,\"layer\":" << layer.value()
                  << ",\"experts\":" << samples.size()
                  << ",\"rows_sampled\":" << sampled
                  << ",\"columns\":" << columns
                  << ",\"ggml_type\":" << storage_type
                  << ",\"mean_energy\":" << mean_energy
                  << ",\"residual_energy_mean\":" << residual_mean
                  << ",\"residual_energy_min\":" << residual_min
                  << ",\"residual_energy_max\":" << residual_max
                  << ",\"cosine_mean\":" << cosine_mean
                  << ",\"cosine_min\":" << cosine_min
                  << ",\"cosine_max\":" << cosine_max << "}\n";
    } else {
        std::cout << "Expert redundancy, layer " << layer.value() << '\n'
                  << "Experts sampled:      " << samples.size() << '\n'
                  << "Rows x columns:       " << sampled << " x " << columns << '\n'
                  << "Residual after mean:  " << residual_mean << " (min " << residual_min
                  << ", max " << residual_max << ")\n"
                  << "Pairwise cosine:      " << cosine_mean << " (min " << cosine_min
                  << ", max " << cosine_max << ")\n"
                  << "Near 1.0 residual and near 0.0 cosine mean the experts share nothing.\n";
    }
    return 0;
}

int command_deepseek_forward(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    auto valid = validate_options(
        args, 1U,
        {"--token", "--layers", "--context", "--batch-size", "--expert-cache", "--trunk-window", "--prompt",
         "--tokenizer", "--max-tokens", "--continue-ids", "--gpu-limit", "--gpu-reserve",
         "--dump-hidden"},
        {"--json", "--dry-run", "--chat", "--cpu-only", "--ggml-compute"});
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
    const auto& spec = pinned_spec_for(checkpoint_root);
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
        // Extra tokens appended after the template, so a partially generated answer can be replayed
        // as a prompt. That is what separates a model choosing an odd token from an incremental
        // decode feeding it the wrong state: prefilling the same prefix has to predict the same
        // next token the decode loop produced, and if it does not, the loop is wrong.
        if (const auto continuation = option_value(args, "--continue-ids")) {
            std::string_view remaining = *continuation;
            while (!remaining.empty()) {
                const auto comma = remaining.find(',');
                const auto field = remaining.substr(0U, comma);
                std::uint32_t value = 0U;
                const auto* const first = field.data();
                const auto* const last = field.data() + field.size();
                const auto parsed = std::from_chars(first, last, value);
                if (parsed.ec != std::errc{} || parsed.ptr != last) {
                    return print_error(
                        vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                             "--continue-ids takes a comma-separated token list",
                                             std::string(field)),
                        json);
                }
                token_ids.push_back(value);
                if (comma == std::string_view::npos) break;
                remaining.remove_prefix(comma + 1U);
            }
        }
    } else {
        token_ids.push_back(static_cast<std::uint32_t>(token.value()));
    }

    // Exact resident cost of what is about to be loaded, from the catalog rather than an estimate.
    vastloom::ByteCount trunk_bytes = 0U;
    vastloom::ByteCount global_bytes = 0U;
    vastloom::ByteCount largest_expert = 0U;
    vastloom::ByteCount largest_source_range = 0U;
    for (const auto& tensor : report.value().catalog.tensors) {
        if (tensor.role == vastloom::model::TensorRole::routed_expert) continue;
        if (!tensor.layer.has_value()) {
            global_bytes += tensor.source.bytes;
        } else if (*tensor.layer < depth) {
            trunk_bytes += tensor.source.bytes;
        }
        largest_source_range = std::max(largest_source_range, tensor.source.bytes);
    }
    for (const auto& bundle : report.value().catalog.experts) {
        largest_expert = std::max(largest_expert, bundle.bytes);
        for (const auto& range : bundle.read_ranges) {
            largest_source_range = std::max(largest_source_range, range.bytes);
        }
    }
    auto trunk_window = optional_size(args, "--trunk-window", 2U);
    if (!trunk_window) return print_error(trunk_window.error(), json);
    std::size_t trunk_window_capacity = std::max<std::size_t>(1U, trunk_window.value());
    const auto budget = policy.value().resources.ram_hard_limit;

    // The runtime reserves, against the same host budget this preflight checks, more than the
    // weights themselves: bounded reads hold staging for their largest range, and the opt-in
    // ggml backend holds its graph pool as scratch. Charge both up front so an approved
    // preflight cannot be surprised by a reservation it never counted - the failure mode that
    // used to die in the expert loader with bad_alloc after a green preflight.
    vastloom::inference::DeepseekV4ForwardPreflightInput preflight_input;
    preflight_input.trunk_bytes = trunk_bytes;
    preflight_input.global_bytes = global_bytes;
    preflight_input.largest_expert_bundle = largest_expert;
    preflight_input.largest_source_range = largest_source_range;
    preflight_input.depth = depth;
    preflight_input.experts_per_token = config.value().experts_per_token;
    preflight_input.expert_count = config.value().expert_count;
    preflight_input.prompt_tokens = token_ids.size();
    preflight_input.expert_cache_entries = expert_cache.value();
    preflight_input.trunk_window = trunk_window_capacity;
    preflight_input.max_tokens = max_tokens.value();
    preflight_input.ram_budget = budget;
    preflight_input.staging_hard_limit = policy.value().resources.staging_hard_limit;
    preflight_input.ggml_pool_bytes =
        has_flag(args, "--ggml-compute")
            ? static_cast<vastloom::ByteCount>(vastloom::inference::GgmlComputeOptions{}.pool_bytes)
            : 0U;
    const auto preflight = vastloom::inference::deepseek_v4_forward_preflight(preflight_input);
    const auto expert_bytes = preflight.expert_cache_bytes;
    const bool persist_decode = preflight.persist_decode;
    const auto required = preflight.required_bytes;
    const auto per_layer_experts = std::min<std::size_t>(
        token_ids.size() * config.value().experts_per_token, config.value().expert_count);
    const auto cache_size = std::max<std::size_t>(expert_cache.value(), per_layer_experts);

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
                  << ",\"trunk_window\":" << trunk_window_capacity
                  << ",\"persist_decode\":" << (persist_decode ? "true" : "false")
                  << ",\"staging_bytes\":" << preflight.staging_charge
                  << ",\"ggml_pool_bytes\":" << preflight.ggml_charge
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
                  << "Read staging:         " << preflight.staging_charge << " bytes\n"
                  << "Compute pool:         " << preflight.ggml_charge << " bytes\n"
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
    // Opt-in: delegate CPU matvecs to the pinned ggml checkout instead of the scalar reference
    // kernels. The Governor charges the backend's fixed pool up front, before anything allocates.
    struct GgmlComputeScope {
        bool active{false};
        ~GgmlComputeScope() {
            if (active) vastloom::inference::ggml_compute_disable();
        }
    } ggml_scope;
    if (has_flag(args, "--ggml-compute")) {
        vastloom::inference::GgmlComputeOptions ggml_options;
        ggml_options.threads = policy.value().cpu_threads;
        auto ggml_ready = vastloom::inference::ggml_compute_enable(governor, ggml_options);
        if (!ggml_ready) return print_error(ggml_ready.error(), json);
        ggml_scope.active = true;
    }
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
    vastloom::ByteCount gpu_resident_limit = 0U;
    if (!has_flag(args, "--cpu-only")) {
        auto cuda_driver = vastloom::cuda::CudaDriver::load();
        if (cuda_driver) {
            vastloom::cuda::CudaMatvecAcceleratorOptions accelerator_options;
            accelerator_options.minimum_free_vram_after = 1U * vastloom::GiB;
            // VRAM is not free of host memory on Windows: a device allocation carries commit
            // charge, so resident matrices compete with the weights for the same budget the
            // preflight just checked. At shallow depths there is room for the full VRAM limit; at
            // full depth there is not, and a 43-layer pass that ignored this passed preflight and
            // then died in the expert loader with bad_alloc. Spend only what the preflight left
            // over, keeping a gigabyte of that headroom for the allocations it does not model.
            // Covers what the preflight does not model and the accelerator must not crowd out: the
            // expert loader's own read buffers, graph scratch and KV state, and the window's trunk
            // reloads. One gigabyte was enough while the weights themselves left little room; with
            // trunks streamed the ceiling would otherwise claim nearly all the remaining commit and
            // starve the very reloads that made the room.
            // Overridable so the residency threshold can be crossed under measurement without
            // changing what every other run reserves. The default stands; a smaller value is a
            // deliberate experiment, and the Governor still refuses any allocation the host cannot
            // afford, so this widens what may be attempted rather than what may be exceeded.
            vastloom::ByteCount kUnmodelledHostHeadroom = 3U * vastloom::GiB;
            if (option_value(args, "--gpu-reserve").has_value()) {
                auto reserve = optional_size(args, "--gpu-reserve", 3072U);
                if (!reserve) return print_error(reserve.error(), json);
                kUnmodelledHostHeadroom =
                    static_cast<vastloom::ByteCount>(reserve.value()) * vastloom::MiB;
            }
            vastloom::ByteCount host_headroom =
                budget > required + kUnmodelledHostHeadroom
                    ? budget - required - kUnmodelledHostHeadroom
                    : 0U;
            // Do not let a generous reserve become a cliff.
            //
            // Zero here is not a smaller GPU, it is no GPU: a full-depth token costs 55,788 ms
            // instead of 8,405, sixfold, with an idle device beside it. And the reserve turns out
            // to be far larger than what it stands in for. Measured on a 43-layer pass, the process
            // peaked at 5.12 GB against 2.56 charged and 2.33 held by the accelerator, so the part
            // nobody models is 0.23 GB - a thirteenth of the three gigabytes held back for it.
            //
            // So when the full reserve would switch the device off, fall back to a gigabyte - still
            // four times the measured need - and grant a gigabyte of ceiling, no more.
            //
            // No more, because more buys nothing. On one prompt a 512 MiB ceiling cost 22,781 ms
            // per token and a 1.41 GB one 20,682, a 9% spread inside the run-to-run noise, and six
            // earlier runs between 1.05 and 2.34 GB showed no trend either. Only zero is different,
            // and it is very different: 55,788 against 8,405 on the same prompt. This removes a
            // cliff rather than climbing a slope, so it claims the least commit that does so.
            //
            // Safe with room to spare: at a 1.41 GB ceiling the pass peaked at 4.27 GB against a
            // 6.15 GB budget, and the 3.74 GB it charged was itself 0.78 GB more than it touched.
            constexpr vastloom::ByteCount kMinimumUsefulCeiling = 1U * vastloom::GiB;
            constexpr vastloom::ByteCount kMeasuredUnmodelledFloor = 1U * vastloom::GiB;
            if (host_headroom < kMinimumUsefulCeiling &&
                budget > required + kMeasuredUnmodelledFloor) {
                host_headroom = std::min<vastloom::ByteCount>(
                    kMinimumUsefulCeiling, budget - required - kMeasuredUnmodelledFloor);
            }
            accelerator_options.maximum_resident_bytes = std::min<vastloom::ByteCount>(
                policy.value().resources.vram_hard_limit, host_headroom);
            // An explicit ceiling, for comparing two runs rather than two machine states.
            //
            // The derived ceiling moves with whatever else the host is doing - it varied between
            // 1.05 and 2.34 GB across one afternoon - and it is the single largest influence on the
            // time per token, so an A/B that does not hold it fixed measures the machine. Only ever
            // lowers: the Governor's limit still bounds it, and this cannot raise it.
            if (option_value(args, "--gpu-limit").has_value()) {
                auto mebibytes = optional_size(args, "--gpu-limit", 0U);
                if (!mebibytes) return print_error(mebibytes.error(), json);
                accelerator_options.maximum_resident_bytes = std::min<vastloom::ByteCount>(
                    accelerator_options.maximum_resident_bytes,
                    static_cast<vastloom::ByteCount>(mebibytes.value()) * vastloom::MiB);
            }
            gpu_resident_limit = accelerator_options.maximum_resident_bytes;
            // A ceiling of zero means the GPU is present and unusable, which costs about fourfold
            // and otherwise looks like the model simply being slow. The Governor is right to refuse
            // - the host has no commit to spare - but the user is owed the reason, because the
            // remedy is theirs: free host memory, not a faster disk or a better GPU.
            if (gpu_resident_limit == 0U && !json) {
                std::cout << "Note: the GPU is present but has no host memory budget left, so every "
                             "matrix runs on the CPU.\n"
                          << "      Free host RAM to use it: the budget is what the RAM limit "
                          << vastloom::format_bytes(budget) << " leaves after the pass needs "
                          << vastloom::format_bytes(required) << " and "
                          << vastloom::format_bytes(kUnmodelledHostHeadroom)
                          << " is reserved for what the preflight does not model.\n";
            }
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

    // A bounded window of layer trunks, refilled as the graph walks layers.
    //
    // Holding all 43 is what sets the host memory floor, and that floor decides both whether the
    // model runs on a given machine and how much is left for the accelerator - which on this
    // hardware is the difference between 90% of matvecs declined and 28%. Two layers are kept
    // because the graph asks for one while the previous one's matrices may still be in flight.
    // The unit is one layer, and a measurement says it should stay that way.
    //
    // Reading only the matrices actually missing looks obviously right - a layer is 149 MB and a
    // matrix is about 5 - and it is not. The device holds roughly half the matrices but barely a
    // fifteenth of the bytes, because small matrices are what fit: a paired run cut trunk traffic
    // by 7%, not the predicted 70%, and paid 7,029 governed reads where 473 sequential ones had
    // done. 19% slower. The whole-layer read stays until the device holds bytes rather than counts.
    struct TrunkWindow {
        // Where a matrix comes from and what shape it has, without its bytes. Enough for the
        // accelerator to recognise weights it already holds, which is what lets a resident matrix
        // skip its NVMe read entirely.
        struct Descriptor {
            std::size_t layer{0U};
            vastloom::inference::DeepseekV4PackedMatrix shape;
        };

        const vastloom::model::LogicalTensorCatalog* catalog{nullptr};
        const vastloom::inference::DeepseekV4ReferenceConfig* config{nullptr};
        vastloom::safety::ResourceGovernor* governor{nullptr};
        vastloom::cuda::CudaMatvecAccelerator* accelerator{nullptr};
        std::size_t capacity{2U};
        // Layers that keep their place, and the one slot everything else passes through.
        std::map<std::size_t, vastloom::inference::DeepseekV4ReferenceLayerTrunk> live;
        std::map<std::size_t, vastloom::inference::DeepseekV4ReferenceLayerTrunk> transient;
        std::map<std::string, Descriptor, std::less<>> index;
        std::uint64_t bytes_read{0U};
        std::size_t reloads{0U};
        std::size_t device_hits{0U};

        [[nodiscard]] vastloom::Result<const vastloom::inference::DeepseekV4PackedMatrix*> get(
            std::string_view name) {
            const auto found = index.find(name);
            if (found == index.end()) return nullptr;
            // Already on the device: hand back the shape without its bytes and read nothing.
            //
            // Safe despite residency being evictable: the caller multiplies by this matrix
            // immediately and nothing else is uploaded in between, so it cannot be evicted before
            // use. Were that ever to change, the CPU fallback fails loudly on an empty payload
            // rather than computing against the wrong weights.
            if (accelerator != nullptr && accelerator->holds(found->second.shape.cache_identity)) {
                ++device_hits;
                return &found->second.shape;
            }
            const std::size_t layer = found->second.layer;
            // The spare slot counts as held, or every matrix of a streamed layer reloads the whole
            // layer: 5,805 reads and 810 GB where 473 and 65.3 had been.
            if (!live.contains(layer) && !transient.contains(layer)) {
                auto trunk = vastloom::inference::deepseek_v4_load_0731_gguf_layer_trunk(
                    *catalog, *config, layer, *governor);
                if (!trunk) return trunk.error();
                ++reloads;
                try {
                    for (const auto& [matrix_name, matrix] : trunk.value().matrices) {
                        (void)matrix_name;
                        bytes_read += matrix.resident != nullptr
                                          ? static_cast<std::uint64_t>(matrix.resident_bytes)
                                          : static_cast<std::uint64_t>(matrix.encoded.size());
                    }
                    // Retain a fixed set, and stream the rest through one slot.
                    //
                    // Discarding the oldest was worth nothing: the pass walks layers in a cycle, so
                    // by the time it returns to layer 0 an eight-deep window has replaced it, and
                    // eight resident layers gave exactly the reload count of two. Under a cyclic
                    // reference pattern a window smaller than the cycle has a hit rate of zero
                    // however it chooses its victim - but it has a hit rate of capacity/cycle if it
                    // simply stops choosing. Whichever layers arrive first keep their place; the
                    // others pass through the spare slot, which exists so a matrix stays alive
                    // while the caller multiplies by it.
                    if (live.size() < capacity) {
                        live.emplace(layer, trunk.take_value());
                    } else {
                        transient.clear();
                        transient.emplace(layer, trunk.take_value());
                    }
                } catch (const std::bad_alloc&) {
                    return vastloom::make_error(vastloom::ErrorCode::resource_exhausted,
                                                "Unable to record a streamed layer trunk");
                }
            }
            const auto& held = live.contains(layer) ? live : transient;
            const auto& matrices = held.at(layer).matrices;
            const auto matrix = matrices.find(std::string(name));
            if (matrix == matrices.end()) return nullptr;
            return &matrix->second;
        }
    };
    TrunkWindow window;
    window.catalog = &report.value().catalog;
    window.config = &graph.config;
    window.governor = &governor;
    window.capacity = trunk_window_capacity;
    window.accelerator = accelerator.get();

    // Each layer keeps its own bounded expert cache: the 256 bundles of layer N are a different set
    // from layer N+1, and a shared cache would let one layer evict another's working set.
    std::vector<std::shared_ptr<vastloom::inference::DeepseekV4BoundedExpertCache>> caches;
    const auto load_started = std::chrono::steady_clock::now();
    for (std::size_t layer = 0U; layer < depth; ++layer) {
        auto safe = process_guard.value().sample_and_check();
        if (!safe) return print_error(safe.error(), json);
        // Matrices are indexed from metadata and never read here; the window fetches each one when
        // the graph first asks for it. Normalization vectors, router biases and routing tables are
        // loaded and stay resident - they are what the graph consults to decide shapes and routes,
        // and together they are a rounding error against the matrices.
        //
        // Describing rather than loading is what removes the startup pass over all 6.39 GB of trunk
        // weights, which existed only to learn shapes the catalog already knew.
        auto mappings = vastloom::inference::deepseek_v4_layer_trunk_mappings(graph.config, layer);
        if (!mappings) return print_error(mappings.error(), json);
        for (const auto& mapping : mappings.value()) {
            using Kind = vastloom::inference::DeepseekV4TrunkTensorMapping::Kind;
            try {
                if (mapping.kind == Kind::matrix) {
                    // Shape, type and identity; no bytes. This is what a token hands back when the
                    // accelerator already holds the weights, so nothing is read from NVMe.
                    auto shape = vastloom::inference::deepseek_v4_describe_0731_gguf_trunk_matrix(
                        report.value().catalog, mapping.source, layer, mapping.elements);
                    if (!shape) return print_error(shape.error(), json);
                    if (shape.value().rows != mapping.rows) {
                        return print_error(
                            vastloom::make_error(
                                vastloom::ErrorCode::invalid_format,
                                "DeepSeek 0731 trunk matrix shape does not match its consumer",
                                mapping.source),
                            json);
                    }
                    TrunkWindow::Descriptor descriptor;
                    descriptor.layer = layer;
                    descriptor.shape = shape.take_value();
                    window.index.emplace(mapping.consumer, std::move(descriptor));
                    graph.provided_matrix_names.insert(mapping.consumer);
                } else if (mapping.kind == Kind::vector) {
                    auto tensor = vastloom::inference::deepseek_v4_load_0731_gguf_trunk_vector(
                        report.value().catalog, mapping.source, layer, governor);
                    if (!tensor) return print_error(tensor.error(), json);
                    if (tensor.value().elements != mapping.elements) {
                        return print_error(
                            vastloom::make_error(
                                vastloom::ErrorCode::invalid_format,
                                "DeepSeek 0731 trunk float shape does not match its consumer",
                                mapping.source),
                            json);
                    }
                    graph.packed_float_tensors.emplace(mapping.consumer, tensor.take_value());
                } else {
                    auto table = vastloom::inference::deepseek_v4_load_0731_gguf_trunk_table(
                        report.value().catalog, mapping.source, layer, mapping.elements, governor);
                    if (!table) return print_error(table.error(), json);
                    if (table.value().elements != mapping.elements) {
                        return print_error(
                            vastloom::make_error(
                                vastloom::ErrorCode::invalid_format,
                                "DeepSeek 0731 hash routing table shape does not match its consumer",
                                mapping.source),
                            json);
                    }
                    graph.packed_integer_tensors.emplace(mapping.consumer, table.take_value());
                }
            } catch (const std::bad_alloc&) {
                return print_error(vastloom::make_error(
                                       vastloom::ErrorCode::resource_exhausted,
                                       "Unable to index a streamed layer trunk"),
                                   json);
            }
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
        // Four workers submit every selected miss after routing. The graph then computes the
        // shared expert (trunk-resident) while those loads run, and acquires each routed
        // expert at its compute turn. Two workers left expert I/O on the critical path
        // (24 s of a 33 s IQ2 generate); four is the cache bound. Queue equals the per-layer
        // request cap so a batch never waits on itself.
        cache_options.asynchronous_workers = 4U;
        cache_options.asynchronous_queue_capacity = per_layer_experts;
        auto cache = vastloom::inference::DeepseekV4BoundedExpertCache::create(
            bindings.take_value(), governor, cache_options);
        if (!cache) return print_error(cache.error(), json);

        // Release happens after the layer finishes, not when the next layer begins, so a
        // hash-routed N+1 prefetch can load while N still holds its experts. Peak residency is
        // two layers (current plus one prefetched hash layer), which the preflight already
        // budgets. A release that fails leaves the memory charged.
        graph.expert_providers.emplace(cache_options.block_prefix,
                                       cache.value()->provider_binding());
        caches.push_back(cache.take_value());
    }
    graph.release_layer_experts = [&caches, persist_decode](std::size_t layer, bool persist) {
        if (layer >= caches.size()) return;
        if (persist && persist_decode) {
            [[maybe_unused]] const auto kept = caches[layer]->retain_last_selected();
        } else {
            [[maybe_unused]] const auto released = caches[layer]->clear();
        }
    };
    graph.matrix_provider = [&window](std::string_view name) { return window.get(name); };
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
        // Physical device reads across the generation, which is the one quantity the byte counters
        // cannot supply. `expert_bytes_read` and `trunk_bytes_read` are logical: they say what the
        // engine asked for, not what the disk delivered, and the difference is page cache. Without
        // separating them, an expert rate of 1.74 GB/s could equally be a saturated SSD or a warm
        // cache being consumed slowly, and those call for opposite work.
        auto device_before = vastloom::io::sample_device_io(checkpoint_root);
        auto generated = vastloom::inference::deepseek_v4_reference_generate(
            graph, token_ids, options, &process_guard.value());
        if (!generated) return print_error(generated.error(), json);
        vastloom::ByteCount device_read_bytes = 0U;
        std::uint64_t device_read_operations = 0U;
        if (device_before) {
            if (auto device_after = vastloom::io::sample_device_io(checkpoint_root)) {
                if (auto delta = vastloom::io::device_io_delta(device_before.value(),
                                                              device_after.value())) {
                    device_read_bytes = delta.value().read_bytes;
                    device_read_operations = delta.value().read_operations;
                }
            }
        }
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
        vastloom::ByteCount peak_commit = 0U;
        if (auto peak = vastloom::safety::process_peak_commit_bytes()) {
            peak_commit = peak.value();
        }
        if (json) {
            std::cout << "{\"ok\":true,\"complete_model\":" << (complete ? "true" : "false")
                      << ",\"executed_layers\":" << depth
                      << ",\"model_layers\":" << config.value().layer_count
                      << ",\"prompt_tokens\":" << stats.prompt_tokens
                      << ",\"generated_tokens\":" << stats.generated_tokens
                      // The tokens themselves, not just their rendering. A decoded string cannot
                      // distinguish a model that chose an odd token from a decoder that mangled a
                      // reasonable one, and telling those apart is most of diagnosing bad output.
                      << ",\"generated_ids\":[";
            {
                bool first_id = true;
                for (const std::uint32_t id : generated.value().generated_ids) {
                    if (!first_id) std::cout << ',';
                    first_id = false;
                    std::cout << id;
                }
            }
            std::cout << "]"
                      << ",\"finish_reason\":\""
                      << vastloom::inference::to_string(generated.value().finish_reason)
                      << "\",\"text\":\"" << vastloom::util::json_escape(text)
                      << "\",\"expert_bytes_read\":" << stats.logical_expert_bytes_read;
        {
            std::size_t recurrent = 0U;
            std::size_t requested = 0U;
            for (const auto& cache : caches) {
                const auto snap = cache->snapshot();
                recurrent += snap.recurrent_experts;
                requested += snap.requested_experts;
            }
            std::cout << ",\"expert_recurrent\":" << recurrent
                      << ",\"expert_requested\":" << requested;
        }
        std::cout
                      // Where a token's time actually goes: reading experts off NVMe against
                      // multiplying by them. Optimising the wrong one of these is easy to do blind.
                      << ",\"expert_load_ms\":" << stats.expert_load_service_ms
                      << ",\"expert_compute_ms\":" << stats.expert_compute_ms
                      << ",\"io_compute_overlap_ms\":" << stats.io_compute_overlap_ms
                      << ",\"forward_passes\":" << stats.forward_passes
                      << ",\"resident_bytes\":" << required
                      // What the process actually peaked at, against what the preflight charged.
                      // The gap between them, less the accelerator's own resident bytes, is the
                      // part no one is modelling - and a flat reserve stands in for it today at the
                      // cost of the whole GPU on a busy host.
                      << ",\"peak_commit_bytes\":" << peak_commit
                      << ",\"gpu_matvecs\":" << gpu_stats.accelerated
                      << ",\"gpu_declined\":" << gpu_stats.declined
                      << ",\"gpu_resident_bytes\":" << gpu_stats.resident_bytes
                      // What the accelerator was allowed to hold, so a run that declines most of
                      // its matvecs shows whether it ran out of room or out of supported types.
                      << ",\"gpu_resident_limit\":" << gpu_resident_limit
                      // What the opt-in ggml backend took, against what fell back to the scalar
                      // reference kernels. Zero and zero means the backend was not enabled.
                      << ",\"ggml_matvecs\":" << vastloom::inference::ggml_compute_accelerated_count()
                      << ",\"ggml_fallbacks\":" << vastloom::inference::ggml_compute_fallback_count()
                      << ",\"ggml_input_quantizes\":"
                      << vastloom::inference::ggml_compute_input_quantize_count()
                      << ",\"ggml_input_quantize_hits\":"
                      << vastloom::inference::ggml_compute_input_quantize_hits()
                      // Trunk reads that happened, against requests served from the device without
                      // one. The second number is I/O the streaming window did not have to pay.
                      << ",\"trunk_reloads\":" << window.reloads
                      << ",\"trunk_device_hits\":" << window.device_hits
                      // Bytes the window actually read, which is what a change to its granularity
                      // has to move. Reload counts alone cannot say whether one is better than
                      // another: 7,029 reads of one matrix and 473 of a whole layer differ by a
                      // factor of fifteen in count and by 7% in traffic.
                      << ",\"trunk_bytes_read\":" << window.bytes_read
                      // The four quantities a bandwidth budget needs kept apart. Logical bytes are
                      // what the engine consumed; device bytes are what the disk actually delivered,
                      // and their difference is page cache; uploaded bytes are what crossed PCIe,
                      // which is far less than the logical total because resident matrices are not
                      // re-sent. Each divides by the target time into a different requirement.
                      << ",\"device_read_bytes\":" << device_read_bytes
                      << ",\"device_read_operations\":" << device_read_operations
                      << ",\"gpu_uploaded_bytes\":" << gpu_stats.uploaded_bytes
                      << ",\"gpu_upload_ms\":" << gpu_stats.upload_ms
                      << ",\"gpu_evicted\":" << gpu_stats.evicted
                      << ",\"gpu_declined_by_type\":{";
        {
            bool first_type = true;
            for (const auto& [ggml_type, count] : gpu_stats.declined_by_type) {
                if (!first_type) std::cout << ',';
                first_type = false;
                std::cout << '"' << ggml_type << "\":" << count;
            }
        }
        std::cout << "}"
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
                       << vastloom::format_bytes(gpu_stats.resident_bytes) << " resident)\n";
            if (vastloom::inference::ggml_compute_enabled() ||
                vastloom::inference::ggml_compute_accelerated_count() != 0U) {
                std::cout << "ggml matvecs:         "
                          << vastloom::inference::ggml_compute_accelerated_count()
                          << " accelerated, "
                          << vastloom::inference::ggml_compute_fallback_count()
                          << " fell back to scalar\n"
                          << "ggml activations:     "
                          << vastloom::inference::ggml_compute_input_quantize_count()
                          << " quantized, "
                          << vastloom::inference::ggml_compute_input_quantize_hits()
                          << " reused\n";
            }
            std::cout << "Expert bytes read:    " << stats.logical_expert_bytes_read << '\n'
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

    // The hidden state the LM head multiplied, written where another implementation's can be laid
    // beside it. Logits cannot localize a disagreement - they are this vector seen through the head,
    // so a fault in either looks identical from outside.
    if (const auto destination = option_value(args, "--dump-hidden")) {
        std::ofstream out(cli_path(*destination), std::ios::binary | std::ios::trunc);
        if (!out) {
            return print_error(vastloom::make_error(vastloom::ErrorCode::io_error,
                                                    "Unable to open the hidden-state destination",
                                                    *destination),
                               json);
        }
        out << std::setprecision(9);
        for (const float value : forward.value().final_hidden) out << value << '\n';
        if (!out) {
            return print_error(vastloom::make_error(vastloom::ErrorCode::io_error,
                                                    "Unable to write the hidden state", *destination),
                               json);
        }
    }

    // The eight strongest candidates and their exact logits.
    //
    // A greedy token on its own cannot say whether a disagreement with another implementation is a
    // defect or the last bit of arithmetic: two logits a hair apart are decided by accumulation
    // order, and this reference sums in double where ggml sums in float. The margin is what tells
    // those apart, so report it rather than the winner alone.
    std::vector<std::pair<float, std::uint32_t>> ranked;
    {
        const auto& logits = forward.value().logits;
        ranked.reserve(logits.size());
        for (std::size_t index = 0U; index < logits.size(); ++index) {
            ranked.emplace_back(logits[index], static_cast<std::uint32_t>(index));
        }
        const std::size_t wanted = std::min<std::size_t>(8U, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(wanted),
                          ranked.end(), [](const auto& left, const auto& right) {
                              return left.first > right.first;
                          });
        ranked.resize(wanted);
    }

    if (json) {
        std::cout << "{\"ok\":true,\"complete_model\":" << (complete ? "true" : "false")
                  << ",\"executed_layers\":" << depth
                  << ",\"model_layers\":" << config.value().layer_count
                  << ",\"prompt_tokens\":" << token_ids.size()
                  << ",\"next_token\":" << forward.value().greedy_token_id
                  << ",\"next_text\":\"" << vastloom::util::json_escape(next_text)
                  << "\",\"top_logits\":[";
        {
            bool first_rank = true;
            for (const auto& [value, id] : ranked) {
                if (!first_rank) std::cout << ',';
                first_rank = false;
                std::cout << "{\"id\":" << id << ",\"logit\":"
                          << std::setprecision(9) << value << '}';
            }
        }
        std::cout << "],\"logits\":" << forward.value().logits.size()
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
                                  {"--json", "--no-stream", "--no-stats", "--no-verify"},
                                  {"--capability-module"});
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

    auto capability_paths = all_option_values(args, "--capability-module");
    if (capability_paths.size() > 8U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "At most eight capability modules can be registered"), json);
    }
    if (!capability_paths.empty() && !prompt) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--capability-module currently requires --prompt"), json);
    }
    std::vector<vastloom::inference::CapabilityModule> capability_modules;
    for (const auto& path_text : capability_paths) {
        auto module = vastloom::inference::load_capability_module(cli_path(path_text));
        if (!module) return print_error(module.error(), json);
        capability_modules.push_back(module.take_value());
    }

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
                                    bool interactive,
                                    const vastloom::inference::CapabilityOverlay* capability_overlay) -> Result<vastloom::inference::LlamaGenerationResult> {
        real_generation_cancelled = 0;
        if (interactive) std::cout << "assistant> " << std::flush;
        auto generated = model.value().generate(history, options, *runtime.value().governor,
                                                &runtime.value().process_guard,
                                                stream_enabled ? [](std::string_view piece) { std::cout << piece << std::flush; }
                                                               : std::function<void(std::string_view)>{},
                                                capability_overlay);
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
        std::optional<vastloom::inference::CapabilityOverlay> capability_overlay;
        if (!capability_modules.empty()) {
            const auto routed = vastloom::inference::route_modules(capability_modules, *prompt, 1U);
            const auto& selected = capability_modules.at(routed.front().index);
            auto bound = vastloom::inference::build_capability_overlay(model.value().config(), selected);
            if (!bound) return print_error(bound.error(), json);
            capability_overlay.emplace(bound.take_value());
        }
        auto generated = run_generation(messages, false,
                                        capability_overlay.has_value() ? &capability_overlay.value()
                                                                       : nullptr);
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
                      << ",\"tokens_per_second\":" << generated.value().stats.tokens_per_second
                      << "}";
        if (capability_overlay.has_value()) {
            std::cout << ",\"capability\":{\"selected\":\""
                      << vastloom::util::json_escape(capability_overlay->selected_name())
                      << "\",\"considered\":" << capability_modules.size()
                      << ",\"bindings\":" << capability_overlay->binding_count()
                      << ",\"added_macs\":" << capability_overlay->applied_macs()
                      << ",\"module_bytes\":" << capability_overlay->module_bytes() << "}";
        }
        std::cout << "}\n";
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
        auto generated = run_generation(messages, true, nullptr);
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

int command_module_init(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (!args.empty() && (args[0] == "--help" || args[0] == "-h")) {
        std::cout << "Usage: vastloom module-init MODEL OUT --name NAME [--rank N] [--scale F]\n"
                  << "                     [--seed N] [--layers N] [--targets LIST] [--zero-a] [--json]\n";
        return 0;
    }
    auto valid = validate_options(args, 2U,
                                  {"--name", "--rank", "--scale", "--seed", "--layers", "--targets"},
                                  {"--zero-a", "--json"});
    if (!valid) return print_error(valid.error(), json);

    const auto name = option_value(args, "--name");
    if (!name || name->empty() || name->size() > 128U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "module-init requires a --name of 1 to 128 bytes"), json);
    }
    auto rank = optional_size(args, "--rank", 16U);
    auto scale = optional_float(args, "--scale", 0.05F);
    auto seed = optional_u64(args, "--seed", 0x564153544C4F4F4DULL);
    auto layers = optional_size(args, "--layers", 0U);
    if (!rank || rank.value() < 1U || rank.value() > vastloom::inference::kMaxModuleRank) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--rank must be between 1 and 64"), json);
    }
    if (!scale || !std::isfinite(scale.value()) || scale.value() < 0.0F || scale.value() > 1.0e6F) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--scale must be a finite number between 0 and 1e6"), json);
    }
    if (!seed) return print_error(seed.error(), json);

    const auto model_path = cli_path(args[0]);
    const auto output_path = cli_path(args[1]);
    auto config = vastloom::inference::load_llama_config(model_path / "config.json");
    if (!config) return print_error(config.error(), json);
    std::size_t layer_count = layers.value() != 0U ? layers.value()
                                                   : config.value().num_hidden_layers;
    if (layer_count > config.value().num_hidden_layers) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--layers exceeds the model's layer count"), json);
    }

    // --targets is a comma-separated list of projection names; lm_head is opt-in because it is
    // the largest matrix in the module and E1 mechanics do not need it by default.
    std::vector<vastloom::inference::CapabilityTarget> selected_targets;
    bool include_lm_head = false;
    {
        const std::string raw = option_value(args, "--targets").value_or("q,o,gate,up,down");
        std::string lower;
        lower.reserve(raw.size());
        for (const char character : raw) {
            lower.push_back(character >= 'A' && character <= 'Z'
                                ? static_cast<char>(character - 'A' + 'a')
                                : character);
        }
        std::set<std::string> seen_pieces;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = lower.find(',', start);
            const std::string piece =
                lower.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (piece.empty()) {
                return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                        "--targets contains an empty name"), json);
            }
            if (!seen_pieces.insert(piece).second) {
                return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                        "--targets repeats a name", piece), json);
            }
            if (piece == "lm_head") {
                include_lm_head = true;
            } else {
                const auto match = std::find_if(
                    vastloom::inference::kAllCapabilityTargets.begin(),
                    vastloom::inference::kAllCapabilityTargets.end(),
                    [&piece](vastloom::inference::CapabilityTarget candidate) {
                        return vastloom::inference::to_string(candidate) == piece;
                    });
                if (match == vastloom::inference::kAllCapabilityTargets.end()) {
                    return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                            "Unknown target in --targets", piece),
                                       json);
                }
                selected_targets.push_back(*match);
            }
            if (comma == std::string::npos) break;
            start = comma + 1U;
        }
    }
    if (selected_targets.empty() && !include_lm_head) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--targets produced no usable projections"), json);
    }

    vastloom::inference::CapabilityModule module;
    module.name = *name;
    module.rank = static_cast<std::uint32_t>(rank.value());
    module.scale = scale.value();
    // The routing embedding is derived from the module's own name: distinct names route
    // distinctly, deterministically, with no extra flag to forget.
    module.routing_embedding = vastloom::inference::embed_request(module.name);

    // splitmix64 is fully specified, so the same seed produces the same bytes everywhere.
    std::uint64_t state = seed.value();
    const auto next_random = [&state]() {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94D049BB133111EBULL;
        return mixed ^ (mixed >> 31U);
    };
    const auto next_weight = [&next_random]() {
        return static_cast<float>(
            static_cast<double>(static_cast<std::int64_t>(next_random() % 2'000'001U) - 1'000'000) / 1.0e9);
    };
    const bool zero_a = has_flag(args, "--zero-a");

    const auto append_target = [&](std::uint32_t layer_id,
                                   vastloom::inference::CapabilityTarget target) {
        vastloom::inference::ModuleTarget entry;
        entry.spec.layer = layer_id;
        entry.spec.target = target;
        const auto dims =
            vastloom::inference::capability_target_dims(config.value(), target);
        entry.spec.rows = dims.first;
        entry.spec.cols = dims.second;
        entry.a.resize(static_cast<std::size_t>(module.rank) * entry.spec.cols);
        entry.b.resize(static_cast<std::size_t>(entry.spec.rows) * module.rank);
        for (float& value : entry.a) value = zero_a ? 0.0F : next_weight();
        for (float& value : entry.b) value = next_weight();
        module.targets.push_back(std::move(entry));
    };
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        for (const auto target : selected_targets) {
            append_target(static_cast<std::uint32_t>(layer), target);
        }
    }
    if (include_lm_head) {
        append_target(vastloom::inference::kCapabilityLmHeadLayer,
                      vastloom::inference::CapabilityTarget::lm_head);
    }

    auto written = vastloom::inference::write_capability_module(output_path, module);
    if (!written) return print_error(written.error(), json);
    if (json) {
        std::error_code size_error;
        const auto bytes = std::filesystem::file_size(output_path, size_error);
        std::cout << "{\"ok\":true,\"path\":\"" << vastloom::util::json_escape(output_path.string())
                  << "\",\"file_bytes\":" << (size_error ? 0U : bytes)
                  << ",\"rank\":" << module.rank
                  << ",\"targets\":" << module.targets.size()
                  << ",\"scale\":" << module.scale << "}\n";
    } else {
        std::cout << "Wrote capability module " << output_path.string() << '\n';
    }
    return 0;
}

int command_module_fit(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    if (!args.empty() && (args[0] == "--help" || args[0] == "-h")) {
        std::cout << "Usage: vastloom module-fit MODEL TRANSCRIPT OUT --name NAME [--rank N]"
                     " [--steps N] [--lr F] [--reg F] [--seed N] [--max-records N]"
                     " [--site lm-head|hidden] [--no-verify] [--json]\n";
        return 0;
    }
    auto valid = validate_options(args, 3U,
                                  {"--name", "--rank", "--steps", "--lr", "--reg", "--seed",
                                   "--max-records", "--site"},
                                  {"--json", "--no-verify"});
    if (!valid) return print_error(valid.error(), json);

    const auto name = option_value(args, "--name");
    if (!name || name->empty() || name->size() > 128U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "module-fit requires a --name of 1 to 128 bytes"), json);
    }
    auto rank = optional_size(args, "--rank", 8U);
    auto steps = optional_size(args, "--steps", 400U);
    auto lr = optional_float(args, "--lr", 0.5F);
    auto reg = optional_float(args, "--reg", 0.05F);
    auto seed = optional_u64(args, "--seed", 0xD157111CULL);
    auto max_records = optional_size(args, "--max-records", 16U);
    if (!rank || rank.value() < 1U || rank.value() > vastloom::inference::kMaxModuleRank) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--rank must be between 1 and 64"), json);
    }
    if (!steps || steps.value() == 0U || steps.value() > 100'000U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--steps must be between 1 and 100000"), json);
    }
    if (!lr || !std::isfinite(lr.value()) || lr.value() <= 0.0F || lr.value() > 10.0F) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--lr must be between 0 and 10"), json);
    }
    if (!reg || !std::isfinite(reg.value()) || reg.value() < 0.0F || reg.value() > 1.0e6F) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--reg must be a finite number between 0 and 1e6"), json);
    }
    if (!seed) return print_error(seed.error(), json);
    if (!max_records || max_records.value() == 0U || max_records.value() > 64U) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--max-records must be between 1 and 64"), json);
    }
    const auto site = option_value(args, "--site");
    if (!site) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "module-fit requires --site lm-head or --site hidden"), json);
    }
    const bool hidden_site = *site == "hidden";
    if (!hidden_site && *site != "lm-head") {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "--site must be 'lm-head' or 'hidden'"), json);
    }

    // The transcript is untrusted input like everything else that crosses the CLI: bounded
    // parse, strict format identity, teacher-incorrect records skipped rather than trained on.
    vastloom::util::JsonLimits limits;
    limits.max_input_bytes = 8U * vastloom::MiB;
    limits.max_nodes = 16'384U;
    auto transcript = vastloom::util::load_json_file(cli_path(args[1]), limits);
    if (!transcript) return print_error(transcript.error(), json);
    const auto* format = transcript.value().find("format");
    if (format == nullptr || !format->as_string() ||
        format->as_string().value() != "vastloom.distill_transcript.v1") {
        return print_error(vastloom::make_error(vastloom::ErrorCode::unsupported,
                                                "Transcript format is unrecognized",
                                                args[1]), json);
    }
    const auto* records_value = transcript.value().find("records");
    auto records = records_value == nullptr ? vastloom::make_error(vastloom::ErrorCode::invalid_format,
                                                                   "Transcript has no records")
                                            : records_value->as_array();
    if (!records) return print_error(records.error(), json);

    struct UsableRecord { std::string id; std::string prompt; std::string completion; };
    std::vector<UsableRecord> usable;
    std::size_t skipped_uncorrect = 0U;
    for (const auto& entry : *records.value()) {
        if (usable.size() >= max_records.value()) break;
        const auto* identifier = entry.find("id");
        const auto* prompt = entry.find("prompt");
        const auto* completion = entry.find("completion_text");
        if (identifier == nullptr || prompt == nullptr || completion == nullptr) {
            return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_format,
                                                    "A transcript record is missing fields"), json);
        }
        auto identifier_text = identifier->as_string();
        auto prompt_text = prompt->as_string();
        auto completion_text = completion->as_string();
        if (!identifier_text || !prompt_text || !completion_text) {
            return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_format,
                                                    "A transcript record field is not a string"), json);
        }
        const auto* correct = entry.find("teacher_correct");
        if (correct != nullptr && correct->as_bool() && !correct->as_bool().value()) {
            ++skipped_uncorrect;
            continue;
        }
        usable.push_back(UsableRecord{std::string(identifier_text.value()),
                                      std::string(prompt_text.value()),
                                      std::string(completion_text.value())});
    }
    if (usable.empty()) {
        return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                "No usable transcript records remain after gating"),
                           json);
    }

    const auto model_path = cli_path(args[0]);
    const auto output_path = cli_path(args[2]);
    auto runtime = start_real_runtime(model_path);
    if (!runtime) return print_error(runtime.error(), json);
    auto model = load_real_model(model_path, runtime.value(), !has_flag(args, "--no-verify"));
    if (!model) return print_error(model.error(), json);
    const auto& config = model.value().config();

    std::vector<vastloom::inference::CapabilityFitPair> pairs;
    std::vector<vastloom::inference::HiddenFitSample> hidden_samples;
    for (const auto& record : usable) {
        auto framed = model.value().tokenizer().apply_smol_chat_template(
            {{vastloom::inference::ChatMessage{"user", record.prompt}}}, true);
        if (!framed) return print_error(framed.error(), json);
        auto framed_ids = model.value().tokenizer().encode(framed.value());
        if (!framed_ids) return print_error(framed_ids.error(), json);
        auto target_ids = model.value().tokenizer().encode(record.completion);
        if (!target_ids) return print_error(target_ids.error(), json);
        if (framed_ids.value().empty() || target_ids.value().empty()) continue;
        // Teacher forcing: predicting targets[k] needs every token before it, so the last
        // target itself is never an input.
        std::vector<std::uint32_t> sequence = framed_ids.value();
        sequence.insert(sequence.end(), target_ids.value().begin(),
                        target_ids.value().end() - 1U);
        if (sequence.size() > config.max_position_embeddings) {
            return print_error(vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                    "A training sequence exceeds the model context",
                                                    record.id), json);
        }
        auto states = model.value().hidden_states(sequence, *runtime.value().governor,
                                                  &runtime.value().process_guard);
        if (!states) return print_error(states.error(), json);
        const std::size_t offset = framed_ids.value().size() - 1U;
        if (states.value().size() < offset + target_ids.value().size()) {
            return print_error(vastloom::make_error(vastloom::ErrorCode::internal_error,
                                                    "Hidden states do not cover the supervised positions"),
                               json);
        }
        for (std::size_t k = 0; k < target_ids.value().size(); ++k) {
            if (hidden_site) {
                vastloom::inference::HiddenFitSample sample;
                sample.hidden = states.value()[offset + k];
                sample.target_id = target_ids.value()[k];
                hidden_samples.push_back(std::move(sample));
                continue;
            }
            auto head = model.value().head_dot(states.value()[offset + k]);
            if (!head) return print_error(head.error(), json);
            vastloom::inference::CapabilityFitPair pair;
            pair.hidden = states.value()[offset + k];
            pair.base_logits = head.take_value();
            pair.target_id = target_ids.value()[k];
            pairs.push_back(std::move(pair));
        }
    }

    std::optional<vastloom::safety::Reservation> head_copy_reservation;
    std::vector<float> head_matrix;
    if (hidden_site) {
        // The hidden-site fitter reads the frozen head every step; the copy is Governor-
        // charged like any other large resident buffer and released when the fit returns.
        const ByteCount copy_bytes = static_cast<ByteCount>(config.vocab_size) *
                                     config.hidden_size * sizeof(float);
        auto reserved = runtime.value().governor->try_reserve(
            vastloom::safety::ResourceKind::ram, copy_bytes,
            "Final-hidden capability head matrix copy");
        if (!reserved) return print_error(reserved.error(), json);
        head_copy_reservation = reserved.take_value();
        head_matrix.resize(static_cast<std::size_t>(config.vocab_size) * config.hidden_size);
        auto filled = model.value().head_weights_into(head_matrix);
        if (!filled) return print_error(filled.error(), json);
    }

    Result<vastloom::inference::CapabilityFitResult> fitted =
        vastloom::make_error(vastloom::ErrorCode::internal_error, "unreachable");
    if (hidden_site) {
        fitted = vastloom::inference::fit_final_hidden_lora(
            config.vocab_size, config.hidden_size, rank.value(), steps.value(),
            static_cast<double>(lr.value()), static_cast<double>(reg.value()),
            seed.value(), head_matrix, hidden_samples);
    } else {
        fitted = vastloom::inference::fit_lm_head_lora(
            config.vocab_size, config.hidden_size, rank.value(), steps.value(),
            static_cast<double>(lr.value()), static_cast<double>(reg.value()),
            seed.value(), pairs);
    }
    head_copy_reservation.reset();
    if (!fitted) return print_error(fitted.error(), json);

    vastloom::inference::CapabilityModule module;
    module.name = *name;
    module.rank = static_cast<std::uint32_t>(rank.value());
    module.scale = 1.0F;
    module.routing_embedding = vastloom::inference::embed_request(module.name);
    vastloom::inference::ModuleTarget trained_target;
    if (hidden_site) {
        trained_target.spec.layer = vastloom::inference::kCapabilityFinalHiddenLayer;
        trained_target.spec.target = vastloom::inference::CapabilityTarget::final_hidden;
        trained_target.spec.rows = config.hidden_size;
        trained_target.spec.cols = config.hidden_size;
    } else {
        trained_target.spec.layer = vastloom::inference::kCapabilityLmHeadLayer;
        trained_target.spec.target = vastloom::inference::CapabilityTarget::lm_head;
        trained_target.spec.rows = config.vocab_size;
        trained_target.spec.cols = config.hidden_size;
    }
    trained_target.a = fitted.value().a;
    trained_target.b = fitted.value().b;
    module.targets.push_back(std::move(trained_target));

    auto written = vastloom::inference::write_capability_module(output_path, module);
    if (!written) return print_error(written.error(), json);
    std::error_code size_error;
    const auto bytes = std::filesystem::file_size(output_path, size_error);
    if (json) {
        std::cout << "{\"ok\":true,\"name\":\"" << vastloom::util::json_escape(*name)
                  << "\",\"site\":\"" << *site << "\""
                  << ",\"file_bytes\":" << (size_error ? 0U : bytes)
                  << ",\"records_used\":" << usable.size()
                  << ",\"skipped_teacher_incorrect\":" << skipped_uncorrect
                  << ",\"positions\":" << fitted.value().positions
                  << ",\"initial_loss\":" << fitted.value().initial_loss
                  << ",\"final_loss\":" << fitted.value().final_loss
                  << ",\"regularization\":" << reg.value()
                  << ",\"steps\":" << steps.value()
                  << ",\"rank\":" << rank.value() << "}\n";
    } else {
        std::cout << "Fitted " << usable.size() << " transcripts (" << fitted.value().positions
                  << " positions): loss " << fitted.value().initial_loss << " -> "
                  << fitted.value().final_loss << "\nWrote capability module "
                  << output_path.string() << '\n';
    }
    return 0;
}

int command_serve_real(const std::vector<std::string>& args) {
    auto valid = validate_options(args, 1U, {"--port"}, {"--no-verify"});
    if (!valid) return print_error(valid.error());
    auto port = optional_size(args, "--port", 8099U);
    if (!port || port.value() > 65'535U) {
        return print_error(port ? vastloom::make_error(vastloom::ErrorCode::invalid_argument,
                                                 "--port must be between 0 and 65535")
                                : port.error());
    }
    const auto model_path = cli_path(args[0]);
    auto runtime = start_real_runtime(model_path);
    if (!runtime) return print_error(runtime.error());
    auto model = load_real_model(model_path, runtime.value(), !has_flag(args, "--no-verify"));
    if (!model) return print_error(model.error());

    // The session is the whole point of serving a stateful dense model: one live workspace
    // for the single allowed connection, sized to a bounded slice of the model context.
    const std::size_t session_capacity =
        std::min<std::size_t>(model.value().config().max_position_embeddings, 4096U);
    auto created = model.value().create_session(*runtime.value().governor, session_capacity);
    if (!created) return print_error(created.error());
    auto conversation = created.take_value();

    vastloom::inference::LlamaSession* session_ptr = &conversation;
    const std::uint64_t started_at = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    vastloom::api::LoopbackHandler handler = [&](const vastloom::api::HttpRequest& request)
                                      -> vastloom::api::HttpResponse {
        vastloom::api::HttpResponse response;
        if (request.method == "GET" && request.path == "/v1/models") {
            auto body = vastloom::api::serialize_model_list(
                std::string(model.value().model_id()), started_at);
            if (!body) {
                response.status = 500;
                response.body = vastloom::api::serialize_error(body.error().message,
                                                               "api_error", 500);
                return response;
            }
            response.body = body.take_value();
            return response;
        }
        if (request.method == "POST" && request.path == "/v1/chat/completions") {
            auto parsed = vastloom::api::parse_chat_completion_request(request.body);
            if (!parsed) {
                response.status = vastloom::api::http_status_for(parsed.error().code);
                response.body = vastloom::api::serialize_error(parsed.error().message,
                                                               "invalid_request_error",
                                                               response.status);
                return response;
            }
            std::vector<vastloom::inference::ChatMessage> messages;
            messages.reserve(parsed.value().messages.size());
            for (const auto& message : parsed.value().messages) {
                messages.push_back({message.role, message.content});
            }
            vastloom::inference::LlamaGenerationOptions options;
            options.max_new_tokens = parsed.value().max_tokens;
            options.temperature = parsed.value().temperature.has_value()
                                      ? static_cast<float>(*parsed.value().temperature)
                                      : 0.0F;
            options.top_k = options.temperature > 0.0F ? 40U : 1U;

            const bool wants_stream = parsed.value().stream;
            if (wants_stream) {
                vastloom::api::ChatCompletionResponse skeleton;
                skeleton.id = "chatcmpl-vastloom-real";
                skeleton.model = std::string(model.value().model_id());
                skeleton.created = started_at;
                response.streaming = true;
                response.status = 200;
                response.produce = [&, skeleton](const std::function<bool(std::string_view)>& emit) {
                    auto skel = skeleton;
                    bool first = true;
                    bool alive = true;
                    vastloom::inference::LlamaGenerationOptions stream_options = options;
                    stream_options.cancelled = [&alive] { return !alive; };
                    auto result = model.value().generate(messages, stream_options,
                                                         *runtime.value().governor,
                                                         &runtime.value().process_guard,
                                                         [&](std::string_view piece) {
                                                             if (!alive) return;
                                                             auto chunk = vastloom::api::serialize_chat_completion_chunk(skel, piece, first, false);
                                                             first = false;
                                                             if (!chunk) { alive = false; return; }
                                                             alive = emit("data: " + chunk.take_value() + "\n\n");
                                                         },
                                                         nullptr,
                                                         session_ptr);
                    skel.finish_reason =
                        (result && result.value().stats.stopped_on_max_tokens) ? "length" : "stop";
                    if (alive) {
                        auto last = vastloom::api::serialize_chat_completion_chunk(skel, "", false, true);
                        if (last) alive = emit("data: " + last.take_value() + "\n\n");
                    }
                    if (alive) emit("data: [DONE]\n\n");
                };
                return response;
            }

            auto generated = model.value().generate(messages, options,
                                                    *runtime.value().governor,
                                                    &runtime.value().process_guard,
                                                    {}, nullptr, session_ptr);
            if (!generated) {
                response.status = vastloom::api::http_status_for(generated.error().code);
                response.body = vastloom::api::serialize_error(generated.error().message,
                                                               "api_error", response.status);
                return response;
            }
            vastloom::api::ChatCompletionResponse completion;
            completion.id = "chatcmpl-vastloom-real";
            completion.model = std::string(model.value().model_id());
            completion.content = generated.value().text;
            completion.finish_reason =
                generated.value().stats.stopped_on_max_tokens ? "length" : "stop";
            completion.created = started_at;
            completion.prompt_tokens = generated.value().stats.prompt_tokens;
            completion.completion_tokens = generated.value().stats.generated_tokens;
            auto body = vastloom::api::serialize_chat_completion_response(completion);
            if (!body) {
                response.status = 500;
                response.body = vastloom::api::serialize_error(body.error().message, "api_error", 500);
                return response;
            }
            // Non-standard observability field: how many prompt tokens were served from the
            // session's KV cache instead of re-prefilled. Zero on the first turn.
            std::string payload = body.take_value();
            payload.insert(payload.size() - 1U,
                           std::string(",\"vastloom_cached_prefix_tokens\":") +
                               std::to_string(generated.value().stats.cached_prefix_tokens));
            response.body = std::move(payload);
            return response;
        }
        response.status = 404;
        response.body = vastloom::api::serialize_error("unknown route", "not_found", 404);
        return response;
    };

    vastloom::api::LoopbackServer server;
    auto started = server.start(static_cast<std::uint16_t>(port.value()), std::move(handler),
                                1U * 1024U * 1024U);    if (!started) return print_error(started.error());
    std::cout << "Vastloom real-model API serving " << model.value().model_id()
              << " (prefix-KV session: " << session_capacity << " tokens) on http://127.0.0.1:"
              << server.port() << " (Ctrl+C to stop)\n";
    while (server.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
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
        if (command == "serve") {
            std::error_code serve_error;
            if (!args.empty() && std::filesystem::is_directory(cli_path(args[0]), serve_error)) {
                return command_serve_real(args);
            }
            return vastloom::cli::command_serve(args);
        }
        if (command == "module-init") return command_module_init(args);
        if (command == "module-fit") return command_module_fit(args);
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
        if (command == "qwen38-inspect") return command_qwen38_inspect(args);
        if (command == "deepseek-gguf-inspect") return command_deepseek_gguf_inspect(args);
        if (command == "deepseek-gguf-checkpoint") return command_deepseek_gguf_checkpoint(args);
        if (command == "deepseek-gguf-probe-plan") return command_deepseek_gguf_probe_plan(args);
        if (command == "deepseek-gguf-global-plan") return command_deepseek_gguf_global_plan(args);
        if (command == "range-artifact-inspect") return command_range_artifact_inspect(args);
        if (command == "deepseek-gguf-range-bind") return command_deepseek_gguf_range_bind(args);
        if (command == "deepseek-layer-probe") return command_deepseek_layer_probe(args);
        if (command == "deepseek-forward") return command_deepseek_forward(args);
        if (command == "deepseek-expert-redundancy") {
            return command_deepseek_expert_redundancy(args);
        }
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
