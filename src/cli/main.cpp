#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "stargate/cli/tiny_commands.hpp"
#include "stargate/core/bytes.hpp"
#include "stargate/hardware/hardware_probe.hpp"
#include "stargate/io/bounded_io_pool.hpp"
#include "stargate/model/manifest.hpp"
#include "stargate/model/verifier.hpp"
#include "stargate/planner/execution_planner.hpp"
#include "stargate/research/trace_simulator.hpp"
#include "stargate/safety/heartbeat.hpp"
#include "stargate/safety/resource_governor.hpp"
#include "stargate/util/string.hpp"
#include "stargate/version.hpp"

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

using stargate::ByteCount;
using stargate::Error;
using stargate::Result;

int exit_code(const Error& error) {
    using stargate::ErrorCode;
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
        std::cerr << "{\"ok\":false,\"error\":\"" << stargate::util::json_escape(error.message)
                  << "\",\"context\":\"" << stargate::util::json_escape(error.context) << "\"}\n";
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

std::optional<std::string> option_value(const std::vector<std::string>& args, std::string_view name) {
    for (std::size_t index = 0; index < args.size(); ++index) {
        if (args[index] == name && index + 1U < args.size()) return args[index + 1U];
    }
    return std::nullopt;
}

Result<ByteCount> optional_bytes(const std::vector<std::string>& args, std::string_view name, ByteCount fallback) {
    const auto value = option_value(args, name);
    if (!value.has_value()) return fallback;
    return stargate::parse_bytes(*value);
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
        << "Stargate Local AI " << stargate::kVersion << "\n"
        << "Safety-first local inference research runtime for oversized MoE models.\n\n"
        << "Commands:\n"
        << "  chat MODEL_OR_MANIFEST [--prompt TEXT] [--max-tokens N] [--temperature F]\n"
        << "       [--top-k N] [--seed N] [--timeout-ms N] [--system TEXT] [--no-stream]\n"
        << "       [--no-stats] [--no-verify] [--json]\n"
        << "      Start the interactive TinyTransformer chat CLI or run one prompt.\n\n"
        << "  model-info MODEL_OR_MANIFEST [--no-verify] [--json]\n"
        << "      Inspect a runnable TinyTransformer model and its safe memory budget.\n\n"
        << "  self-test MODEL_OR_MANIFEST [--no-verify] [--json]\n"
        << "      Run deterministic end-to-end generation probes against the validation model.\n\n"
        << "  doctor [--storage PATH] [--json]\n"
        << "      Detect CPU, RAM, NVIDIA VRAM and storage without allocating model memory.\n\n"
        << "  inspect MANIFEST [--json]\n"
        << "      Parse and validate a Stargate model manifest.\n\n"
        << "  verify MANIFEST [--model-root PATH] [--full] [--json]\n"
        << "      Verify chunk paths and sizes; --full also checks SHA-256.\n\n"
        << "  plan MANIFEST [--profile safe|balanced|performance|low-memory|quiet]\n"
        << "       [--storage PATH] [--reference-pc] [--simulate-ram SIZE]\n"
        << "       [--simulate-vram SIZE] [--simulate-storage SIZE] [--simulate-free SIZE] [--json]\n"
        << "      Build a fail-closed VRAM/RAM/NVMe execution plan.\n\n"
        << "  trace-sim TRACE.csv --vram SIZE --ram SIZE [--policy lru|static-hot] [--json]\n"
        << "      Measure expert-cache hit rates, NVMe bytes/token and PCIe traffic.\n\n"
        << "  io-smoke FILE [--offset BYTES] [--size SIZE]\n"
        << "      Exercise the bounded asynchronous I/O path under the resource governor.\n\n"
        << "  store-init PATH\n"
        << "      Create the immutable-model/cache/scratch/state store layout.\n\n"
        << "  heartbeat-write PATH [--state normal|pressure|critical|emergency]\n"
        << "      Atomically write one watchdog heartbeat for integration tests.\n\n"
        << "  version\n";
}

stargate::hardware::HardwareSnapshot reference_pc_snapshot(const std::filesystem::path& storage_path) {
    stargate::hardware::HardwareSnapshot hardware;
    hardware.operating_system = "Windows 11 (reference simulation)";
    hardware.cpu.name = "AMD Ryzen 9 7900X";
    hardware.cpu.logical_cores = 24;
    hardware.cpu.avx2 = true;
    hardware.cpu.fma = true;
    hardware.memory.total_bytes = 32U * stargate::GiB;
    hardware.memory.available_bytes = 26U * stargate::GiB;
    hardware.gpus.push_back(stargate::hardware::GpuInfo{
        "CUDA", "NVIDIA GeForce RTX 4070 SUPER", "simulated", 12U * stargate::GiB,
        11U * stargate::GiB, 60.0, 220.0});
    hardware.storage.path = storage_path;
    hardware.storage.capacity_bytes = 2U * stargate::TiB;
    hardware.storage.free_bytes = 1536U * stargate::GiB;
    hardware.storage.available_bytes = 1536U * stargate::GiB;
    return hardware;
}

Result<stargate::safety::SafetyState> parse_safety_state(std::string_view value) {
    const auto normalized = stargate::util::to_lower(stargate::util::trim(value));
    if (normalized == "normal") return stargate::safety::SafetyState::normal;
    if (normalized == "pressure") return stargate::safety::SafetyState::pressure;
    if (normalized == "critical") return stargate::safety::SafetyState::critical;
    if (normalized == "emergency") return stargate::safety::SafetyState::emergency;
    return stargate::make_error(stargate::ErrorCode::invalid_argument, "Unknown safety state", normalized);
}

int command_doctor(const std::vector<std::string>& args) {
    const bool json = has_flag(args, "--json");
    const auto storage = option_value(args, "--storage").value_or(std::filesystem::current_path().string());
    auto hardware = stargate::hardware::HardwareProbe::collect(storage);
    if (!hardware) return print_error(hardware.error(), json);
    std::cout << (json ? stargate::hardware::to_json(hardware.value()) : stargate::hardware::to_text(hardware.value())) << '\n';
    return 0;
}

int command_inspect(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(stargate::make_error(stargate::ErrorCode::invalid_argument, "inspect requires a manifest path"));
    const bool json = has_flag(args, "--json");
    auto manifest = stargate::model::load_manifest(args[0]);
    if (!manifest) return print_error(manifest.error(), json);
    std::cout << (json ? stargate::model::to_json(manifest.value()) : stargate::model::to_text(manifest.value())) << '\n';
    return 0;
}

int command_verify(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(stargate::make_error(stargate::ErrorCode::invalid_argument, "verify requires a manifest path"));
    const bool json = has_flag(args, "--json");
    auto manifest = stargate::model::load_manifest(args[0]);
    if (!manifest) return print_error(manifest.error(), json);
    const std::filesystem::path root = option_value(args, "--model-root").value_or(
        manifest.value().source_path.parent_path().empty() ? "." : manifest.value().source_path.parent_path().string());
    const auto mode = has_flag(args, "--full") ? stargate::model::VerificationMode::full : stargate::model::VerificationMode::quick;
    auto report = stargate::model::verify_model(manifest.value(), root, mode);
    if (!report) return print_error(report.error(), json);
    std::cout << (json ? stargate::model::to_json(report.value()) : stargate::model::to_text(report.value())) << '\n';
    return report.value().ok ? 0 : 65;
}

int command_plan(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(stargate::make_error(stargate::ErrorCode::invalid_argument, "plan requires a manifest path"));
    const bool json = has_flag(args, "--json");
    auto manifest = stargate::model::load_manifest(args[0]);
    if (!manifest) return print_error(manifest.error(), json);
    auto profile = stargate::planner::parse_profile(option_value(args, "--profile").value_or("balanced"));
    if (!profile) return print_error(profile.error(), json);
    const auto storage_path = option_value(args, "--storage").value_or(std::filesystem::current_path().string());

    stargate::hardware::HardwareSnapshot hardware;
    if (has_flag(args, "--reference-pc")) {
        hardware = reference_pc_snapshot(storage_path);
    } else {
        auto probed = stargate::hardware::HardwareProbe::collect(storage_path);
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
        hardware.memory.available_bytes = stargate::percentage(simulated_ram.value(), 0.82);
    }
    if (option_value(args, "--simulate-vram").has_value()) {
        hardware.gpus.clear();
        if (simulated_vram.value() != 0) {
            hardware.gpus.push_back(stargate::hardware::GpuInfo{
                "CUDA", "Simulated GPU", "simulated", simulated_vram.value(),
                stargate::percentage(simulated_vram.value(), 0.90), std::nullopt, std::nullopt});
        }
    }
    hardware.storage.capacity_bytes = simulated_storage.value();
    hardware.storage.free_bytes = simulated_free.value();
    hardware.storage.available_bytes = simulated_free.value();

    auto plan = stargate::planner::build_plan(hardware, manifest.value(), profile.value());
    if (!plan) return print_error(plan.error(), json);
    std::cout << (json ? stargate::planner::to_json(plan.value()) : stargate::planner::to_text(plan.value())) << '\n';
    return plan.value().compatible ? 0 : 78;
}

int command_trace_sim(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(stargate::make_error(stargate::ErrorCode::invalid_argument, "trace-sim requires a CSV trace path"));
    const bool json = has_flag(args, "--json");
    auto vram = optional_bytes(args, "--vram", 0);
    auto ram = optional_bytes(args, "--ram", 0);
    if (!vram) return print_error(vram.error(), json);
    if (!ram) return print_error(ram.error(), json);
    auto policy = stargate::research::parse_cache_policy(option_value(args, "--policy").value_or("lru"));
    if (!policy) return print_error(policy.error(), json);
    stargate::research::TraceSimulationConfig config;
    config.vram_cache_bytes = vram.value();
    config.ram_cache_bytes = ram.value();
    config.policy = policy.value();
    auto report = stargate::research::simulate_trace(args[0], config);
    if (!report) return print_error(report.error(), json);
    std::cout << (json ? stargate::research::to_json(report.value()) : stargate::research::to_text(report.value())) << '\n';
    return 0;
}

int command_io_smoke(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(stargate::make_error(stargate::ErrorCode::invalid_argument, "io-smoke requires a file path"));
    auto offset = optional_bytes(args, "--offset", 0);
    auto size = optional_bytes(args, "--size", 1U * stargate::MiB);
    if (!offset) return print_error(offset.error());
    if (!size) return print_error(size.error());

    stargate::safety::ResourceLimits limits;
    limits.ram_hard_limit = 2U * stargate::GiB;
    limits.vram_hard_limit = 1U * stargate::GiB;
    limits.staging_hard_limit = std::max<ByteCount>(size.value(), 16U * stargate::MiB);
    limits.scratch_hard_limit = 1U * stargate::GiB;
    stargate::safety::ResourceGovernor governor(limits);
    stargate::io::BoundedIoPool pool({2, 8, limits.staging_hard_limit}, governor);
    auto submitted = pool.submit_read(args[0], offset.value(), size.value(), "cli-io-smoke");
    if (!submitted) return print_error(submitted.error());
    auto result = submitted.value().get();
    if (!result) return print_error(result.error());
    std::cout << "Read " << stargate::format_bytes(result.value().bytes.size())
              << " under a bounded staging reservation.\n";
    return 0;
}

int command_store_init(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(stargate::make_error(stargate::ErrorCode::invalid_argument, "store-init requires a path"));
    std::error_code error;
    const auto root = std::filesystem::absolute(args[0], error).lexically_normal();
    if (error || root == root.root_path()) {
        return print_error(stargate::make_error(stargate::ErrorCode::unsafe_path, "Refusing unsafe model-store root", args[0]));
    }
    for (const auto& relative : {"models/original", "models/optimized", "cache", "scratch", "state", "telemetry"}) {
        std::filesystem::create_directories(root / relative, error);
        if (error) return print_error(stargate::make_error(stargate::ErrorCode::io_error, "Unable to create model-store directory", (root / relative).string()));
    }
    const auto marker = root / ".stargate-store";
    if (!std::filesystem::exists(marker)) {
        std::ofstream output(marker, std::ios::trunc);
        if (!output) return print_error(stargate::make_error(stargate::ErrorCode::io_error, "Unable to create model-store marker", marker.string()));
        output << "format=stargate.store.v1\n"
               << "original_weights=immutable\n"
               << "cache=reconstructable\n";
    }
    std::cout << "Initialized Stargate model store at " << root.string() << '\n';
    return 0;
}

int command_heartbeat_write(const std::vector<std::string>& args) {
    if (args.empty()) return print_error(stargate::make_error(stargate::ErrorCode::invalid_argument, "heartbeat-write requires a path"));
    auto state = parse_safety_state(option_value(args, "--state").value_or("normal"));
    if (!state) return print_error(state.error());
    stargate::safety::HeartbeatRecord heartbeat;
    heartbeat.pid = current_pid();
    heartbeat.sequence = 1;
    heartbeat.unix_time_ms = stargate::safety::unix_time_ms();
    heartbeat.state = state.value();
    auto written = stargate::safety::write_heartbeat_atomic(args[0], heartbeat);
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

        if (command == "chat") return stargate::cli::command_chat(args);
        if (command == "model-info") return stargate::cli::command_model_info(args);
        if (command == "self-test") return stargate::cli::command_self_test(args);
        if (command == "doctor") return command_doctor(args);
        if (command == "inspect") return command_inspect(args);
        if (command == "verify") return command_verify(args);
        if (command == "plan") return command_plan(args);
        if (command == "trace-sim") return command_trace_sim(args);
        if (command == "io-smoke") return command_io_smoke(args);
        if (command == "store-init") return command_store_init(args);
        if (command == "heartbeat-write") return command_heartbeat_write(args);
        if (command == "version" || command == "--version") {
            std::cout << stargate::kVersion << '\n';
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
