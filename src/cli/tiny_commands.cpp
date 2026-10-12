#include "vastloom/cli/tiny_commands.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/hardware/hardware_probe.hpp"
#include "vastloom/inference/tiny_model.hpp"
#include "vastloom/api/loopback_server.hpp"
#include "vastloom/api/openai_protocol.hpp"
#include "vastloom/model/manifest.hpp"
#include "vastloom/model/verifier.hpp"
#include "vastloom/safety/resource_governor.hpp"
#include "vastloom/util/string.hpp"
#include "vastloom/version.hpp"

namespace vastloom::cli {
namespace {

using inference::GenerationOptions;
using inference::GenerationResult;

int error_exit_code(const Error& error) {
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
        std::cerr << "{\"ok\":false,\"error\":\"" << util::json_escape(error.message)
                  << "\",\"context\":\"" << util::json_escape(error.context) << "\"}\n";
    } else {
        std::cerr << "ERROR: " << error.message;
        if (!error.context.empty()) std::cerr << " [" << error.context << ']';
        std::cerr << '\n';
    }
    return error_exit_code(error);
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

Result<std::size_t> parse_size_option(
    const std::vector<std::string>& args,
    std::string_view name,
    std::size_t fallback,
    std::size_t minimum,
    std::size_t maximum) {
    const auto raw = option_value(args, name);
    if (!raw) return fallback;
    std::uint64_t parsed = 0;
    const auto conversion = std::from_chars(raw->data(), raw->data() + raw->size(), parsed);
    if (conversion.ec != std::errc{} || conversion.ptr != raw->data() + raw->size() ||
        parsed < minimum || parsed > maximum || parsed > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::invalid_argument, "Invalid numeric option", std::string(name));
    }
    return static_cast<std::size_t>(parsed);
}

Result<std::uint64_t> parse_u64_option(
    const std::vector<std::string>& args,
    std::string_view name,
    std::uint64_t fallback) {
    const auto raw = option_value(args, name);
    if (!raw) return fallback;
    std::uint64_t parsed = 0;
    const auto conversion = std::from_chars(raw->data(), raw->data() + raw->size(), parsed);
    if (conversion.ec != std::errc{} || conversion.ptr != raw->data() + raw->size()) {
        return make_error(ErrorCode::invalid_argument, "Invalid unsigned integer option", std::string(name));
    }
    return parsed;
}

Result<float> parse_float_option(
    const std::vector<std::string>& args,
    std::string_view name,
    float fallback,
    float minimum,
    float maximum) {
    const auto raw = option_value(args, name);
    if (!raw) return fallback;
    try {
        std::size_t consumed = 0;
        const float parsed = std::stof(*raw, &consumed);
        if (consumed != raw->size() || !std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
            return make_error(ErrorCode::invalid_argument, "Invalid floating-point option", std::string(name));
        }
        return parsed;
    } catch (const std::exception&) {
        return make_error(ErrorCode::invalid_argument, "Invalid floating-point option", std::string(name));
    }
}


Result<void> validate_options(
    const std::vector<std::string>& args,
    const std::vector<std::string_view>& value_options,
    const std::vector<std::string_view>& flag_options) {
    if (args.empty()) return {};
    std::vector<std::string_view> seen;
    seen.reserve(args.size());
    for (std::size_t index = 1U; index < args.size(); ++index) {
        const std::string_view argument = args[index];
        const bool is_flag = std::find(flag_options.begin(), flag_options.end(), argument) != flag_options.end();
        const bool takes_value = std::find(value_options.begin(), value_options.end(), argument) != value_options.end();
        if (!is_flag && !takes_value) {
            return make_error(ErrorCode::invalid_argument, "Unknown option", std::string(argument));
        }
        if (std::find(seen.begin(), seen.end(), argument) != seen.end()) {
            return make_error(ErrorCode::invalid_argument, "Duplicate option", std::string(argument));
        }
        seen.push_back(argument);
        if (!takes_value) continue;
        if (index + 1U >= args.size() || util::starts_with(args[index + 1U], "--")) {
            return make_error(ErrorCode::invalid_argument, "Missing value for option", std::string(argument));
        }
        ++index;
    }
    return {};
}

struct ResolvedModel {
    std::filesystem::path path;
    bool manifest_verified{false};
    std::optional<std::string> expected_model_id;
    std::optional<unsigned> expected_layer_count;
    std::optional<unsigned> expected_context_tokens;
};

Result<ResolvedModel> resolve_model_input(const std::filesystem::path& input, bool verify_hash) {
    if (util::to_lower(input.extension().string()) != ".sgm") {
        return ResolvedModel{input, false, std::nullopt, std::nullopt, std::nullopt};
    }

    auto manifest = model::load_manifest(input);
    if (!manifest) return manifest.error();
    if (manifest.value().metadata_only) {
        return make_error(ErrorCode::invalid_format, "Cannot run a metadata-only model manifest", input.string());
    }
    if (manifest.value().architecture != "vastloom_tiny_transformer" &&
        manifest.value().architecture != "stargate_tiny_transformer") {
        return make_error(
            ErrorCode::unsupported,
            "This command currently supports the Vastloom TinyTransformer adapter",
            manifest.value().architecture);
    }
    if (manifest.value().chunks.size() != 1U || !manifest.value().chunks.front().required) {
        return make_error(
            ErrorCode::invalid_format,
            "TinyTransformer manifests must contain exactly one required model chunk",
            input.string());
    }
    const auto& chunk = manifest.value().chunks.front();
    if (chunk.role != model::ChunkRole::dense || util::to_lower(chunk.dtype) != "fp32" ||
        chunk.layer.has_value() || chunk.expert.has_value()) {
        return make_error(
            ErrorCode::invalid_format,
            "TinyTransformer manifest chunk must be an unscoped dense FP32 payload",
            input.string());
    }
    if (manifest.value().model_total_bytes != chunk.size_bytes ||
        manifest.value().dense_bytes != chunk.size_bytes || manifest.value().routed_expert_bytes != 0U) {
        return make_error(
            ErrorCode::invalid_format,
            "TinyTransformer manifest byte accounting is inconsistent",
            input.string());
    }

    const std::filesystem::path root = input.parent_path().empty() ? "." : input.parent_path();
    auto verification = model::verify_model(
        manifest.value(),
        root,
        verify_hash ? model::VerificationMode::full : model::VerificationMode::quick);
    if (!verification) return verification.error();
    if (!verification.value().ok) {
        return make_error(
            ErrorCode::integrity_error,
            "Model manifest verification failed",
            model::to_text(verification.value()));
    }
    return ResolvedModel{
        root / chunk.relative_path,
        verify_hash,
        manifest.value().model_id,
        manifest.value().layer_count == 0U ? std::nullopt
                                           : std::optional<unsigned>(manifest.value().layer_count),
        manifest.value().max_context_tokens == 0U
            ? std::nullopt
            : std::optional<unsigned>(manifest.value().max_context_tokens)};
}

Result<std::unique_ptr<safety::ResourceGovernor>> make_governor(const std::filesystem::path& model_path) {
    auto hardware = hardware::HardwareProbe::collect(model_path.parent_path().empty() ? "." : model_path.parent_path());
    if (!hardware) return hardware.error();

    std::error_code filesystem_error;
    const auto raw_size = std::filesystem::file_size(model_path, filesystem_error);
    if (filesystem_error) {
        return make_error(ErrorCode::not_found, "Unable to stat model for resource planning", model_path.string());
    }
    const ByteCount model_size = static_cast<ByteCount>(raw_size);
    const ByteCount available = hardware.value().memory.available_bytes;
    const ByteCount reserve = std::min<ByteCount>(2U * GiB, hardware.value().memory.total_bytes / 4U);
    auto required = checked_add(model_size, 32U * MiB);
    if (!required) return required.error();
    if (available <= reserve || required.value() > available - reserve) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Insufficient safe RAM headroom for the model and generation workspace",
            format_bytes(available) + " available");
    }

    const ByteCount safe_available = available - reserve;
    const ByteCount desired = std::max<ByteCount>(128U * MiB, required.value() * 2U);
    safety::ResourceLimits limits;
    limits.host_memory_hard_limit = std::min(safe_available, desired);
    limits.ram_hard_limit = std::min(safe_available, desired);
    limits.vram_hard_limit = 1U * MiB;
    limits.staging_hard_limit = 64U * MiB;
    limits.scratch_hard_limit = std::min<ByteCount>(safe_available, 512U * MiB);
    limits.minimum_system_ram_available = reserve;
    limits.minimum_storage_available = std::min<ByteCount>(5U * GiB, hardware.value().storage.capacity_bytes / 20U);
    return std::make_unique<safety::ResourceGovernor>(limits);
}

struct LoadedModel {
    // TinyModel owns a Reservation that points at the governor. Keep the
    // governor declared first so member destruction releases the model before
    // destroying the governor.
    std::unique_ptr<safety::ResourceGovernor> governor;
    inference::TinyModel model;
    bool manifest_verified{false};
    double load_ms{0.0};
};

Result<LoadedModel> load_model(const std::filesystem::path& input, bool verify_hash) {
    auto resolved = resolve_model_input(input, verify_hash);
    if (!resolved) return resolved.error();
    auto governor = make_governor(resolved.value().path);
    if (!governor) return governor.error();
    const auto started = std::chrono::steady_clock::now();
    auto model = inference::TinyModel::load(resolved.value().path, *governor.value());
    if (!model) return model.error();
    if (resolved.value().expected_model_id &&
        model.value().config().model_id != *resolved.value().expected_model_id) {
        return make_error(
            ErrorCode::integrity_error,
            "Model id does not match its verified manifest",
            model.value().config().model_id + " != " + *resolved.value().expected_model_id);
    }
    if (resolved.value().expected_layer_count &&
        model.value().config().n_layers != *resolved.value().expected_layer_count) {
        return make_error(
            ErrorCode::integrity_error,
            "Model layer count does not match its verified manifest",
            std::to_string(model.value().config().n_layers) + " != " +
                std::to_string(*resolved.value().expected_layer_count));
    }
    if (resolved.value().expected_context_tokens &&
        model.value().config().context_length != *resolved.value().expected_context_tokens) {
        return make_error(
            ErrorCode::integrity_error,
            "Model context length does not match its verified manifest",
            std::to_string(model.value().config().context_length) + " != " +
                std::to_string(*resolved.value().expected_context_tokens));
    }
    const double load_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    return LoadedModel{
        governor.take_value(), model.take_value(), resolved.value().manifest_verified, load_ms};
}

void print_stats(const GenerationResult& result, double load_ms, std::ostream& output) {
    const auto& stats = result.stats;
    output << std::fixed << std::setprecision(2);
    if (load_ms > 0.0) output << "load=" << load_ms << "ms ";
    output << "prompt=" << stats.prompt_tokens << "tok/" << stats.prompt_eval_ms << "ms "
           << "generated=" << stats.generated_tokens << "tok/" << stats.generation_ms << "ms "
           << "speed=" << stats.tokens_per_second << " tok/s";
    if (stats.stopped_on_context_limit) output << " context-limit";
    if (stats.stopped_on_max_tokens) output << " max-tokens";
    if (stats.stopped_on_invalid_token) output << " invalid-token";
    if (stats.stopped_on_timeout) output << " timeout";
    output << '\n';
}

Result<GenerationOptions> parse_generation_options(const std::vector<std::string>& args) {
    auto max_tokens = parse_size_option(args, "--max-tokens", 96U, 1U, 1'000'000U);
    auto top_k = parse_size_option(args, "--top-k", 40U, 1U, 65'536U);
    auto seed = parse_u64_option(args, "--seed", 0x5354415247415445ULL);
    auto timeout_ms = parse_size_option(args, "--timeout-ms", 30'000U, 1U, 3'600'000U);
    auto temperature = parse_float_option(args, "--temperature", 0.0F, 0.0F, 10.0F);
    if (!max_tokens) return max_tokens.error();
    if (!top_k) return top_k.error();
    if (!seed) return seed.error();
    if (!timeout_ms) return timeout_ms.error();
    if (!temperature) return temperature.error();

    GenerationOptions options;
    options.max_new_tokens = max_tokens.value();
    options.top_k = top_k.value();
    options.seed = seed.value();
    options.timeout = std::chrono::milliseconds(timeout_ms.value());
    options.temperature = temperature.value();
    options.system_prompt_override = option_value(args, "--system").value_or("");
    return options;
}

void print_chat_usage() {
    std::cout
        << "Usage: vastloom chat MODEL_OR_MANIFEST [options]\n"
        << "\nOptions:\n"
        << "  --prompt TEXT       Run one prompt instead of the interactive REPL.\n"
        << "  --max-tokens N      Maximum generated byte tokens (default: 96).\n"
        << "  --temperature F     0 for greedy decoding; maximum 10 (default: 0).\n"
        << "  --top-k N           Sampling candidate count (default: 40).\n"
        << "  --seed N            Deterministic sampling seed.\n"
        << "  --timeout-ms N      Per-generation timeout (default: 30000).\n"
        << "  --system TEXT       Override the model system prompt.\n"
        << "  --no-stream         Buffer text until generation completes.\n"
        << "  --no-stats          Hide human-readable timing statistics.\n"
        << "  --no-verify         Skip SHA-256 but keep manifest path/size checks.\n"
        << "  --json              JSON output; requires --prompt.\n";
}

void print_chat_help() {
    std::cout
        << "Interactive commands:\n"
        << "  /help             Show this help.\n"
        << "  /model            Show loaded model information.\n"
        << "  /stats            Show statistics for the previous response.\n"
        << "  /system TEXT      Override the system prompt for later turns.\n"
        << "  /reset            Restore the model's default system prompt.\n"
        << "  /exit or /quit    End the session.\n";
}

}  // namespace

int command_model_info(const std::vector<std::string>& args) {
    if (args.size() == 1U && (args.front() == "--help" || args.front() == "-h")) {
        std::cout << "Usage: vastloom model-info MODEL_OR_MANIFEST [--no-verify] [--json]\n";
        return 0;
    }
    if (args.empty() || util::starts_with(args.front(), "--") || args.front() == "-h") {
        return print_error(make_error(ErrorCode::invalid_argument, "model-info requires a model or manifest path"),
                           has_flag(args, "--json"));
    }
    const bool json = has_flag(args, "--json");
    auto validation = validate_options(args, {}, {"--json", "--no-verify"});
    if (!validation) return print_error(validation.error(), json);
    const bool verify_hash = !has_flag(args, "--no-verify");
    auto loaded = load_model(args[0], verify_hash);
    if (!loaded) return print_error(loaded.error(), json);
    if (json) {
        std::cout << "{\"ok\":true,\"model\":"
                  << inference::tiny_model_config_to_json(loaded.value().model.config())
                  << ",\"resident_bytes\":" << loaded.value().model.resident_bytes()
                  << ",\"scratch_budget_bytes\":"
                  << loaded.value().model.estimated_generation_scratch_bytes()
                  << ",\"load_ms\":" << std::fixed << std::setprecision(3)
                  << loaded.value().load_ms
                  << ",\"manifest_hash_verified\":"
                  << (loaded.value().manifest_verified ? "true" : "false") << "}\n";
    } else {
        std::cout << inference::tiny_model_config_to_text(loaded.value().model.config()) << '\n'
                  << "Resident RAM:    " << format_bytes(loaded.value().model.resident_bytes()) << '\n'
                  << "Scratch budget:  " << format_bytes(loaded.value().model.estimated_generation_scratch_bytes()) << '\n'
                  << "Load time:       " << std::fixed << std::setprecision(2) << loaded.value().load_ms << " ms\n"
                  << "Manifest hash:   " << (loaded.value().manifest_verified ? "verified" : "not verified") << '\n';
    }
    return 0;
}

int command_self_test(const std::vector<std::string>& args) {
    if (args.size() == 1U && (args.front() == "--help" || args.front() == "-h")) {
        std::cout << "Usage: vastloom self-test MODEL_OR_MANIFEST [--no-verify] [--json]\n";
        return 0;
    }
    if (args.empty() || util::starts_with(args.front(), "--") || args.front() == "-h") {
        return print_error(make_error(ErrorCode::invalid_argument, "self-test requires a model or manifest path"),
                           has_flag(args, "--json"));
    }
    const bool json = has_flag(args, "--json");
    auto validation = validate_options(args, {}, {"--json", "--no-verify"});
    if (!validation) return print_error(validation.error(), json);
    const bool verify_hash = !has_flag(args, "--no-verify");
    auto loaded = load_model(args[0], verify_hash);
    if (!loaded) return print_error(loaded.error(), json);
    auto report = inference::run_tiny_model_self_test(loaded.value().model, *loaded.value().governor);
    if (!report) return print_error(report.error(), json);
    std::cout << (json ? inference::self_test_report_to_json(report.value())
                       : inference::self_test_report_to_text(report.value()))
              << '\n';
    return report.value().passed() ? 0 : 1;
}

int command_chat(const std::vector<std::string>& args) {
    if (args.size() == 1U && (args.front() == "--help" || args.front() == "-h")) {
        print_chat_usage();
        return 0;
    }
    if (args.empty() || util::starts_with(args.front(), "--") || args.front() == "-h") {
        return print_error(make_error(ErrorCode::invalid_argument, "chat requires a model or manifest path"),
                           has_flag(args, "--json"));
    }
    auto validation = validate_options(
        args,
        {"--prompt", "--max-tokens", "--temperature", "--top-k", "--seed", "--timeout-ms", "--system"},
        {"--json", "--no-stream", "--no-stats", "--no-verify"});
    if (!validation) return print_error(validation.error(), has_flag(args, "--json"));
    const bool json = has_flag(args, "--json");
    const bool no_stream = has_flag(args, "--no-stream");
    const bool no_stats = has_flag(args, "--no-stats");
    const bool verify_hash = !has_flag(args, "--no-verify");
    const auto one_shot_prompt = option_value(args, "--prompt");
    if (json && !one_shot_prompt) {
        return print_error(make_error(ErrorCode::invalid_argument, "--json is only supported with --prompt"), true);
    }

    auto options = parse_generation_options(args);
    if (!options) return print_error(options.error(), json);
    auto loaded = load_model(args[0], verify_hash);
    if (!loaded) return print_error(loaded.error(), json);
    if (options.value().top_k > loaded.value().model.config().vocab_size) {
        return print_error(make_error(ErrorCode::invalid_argument, "--top-k exceeds this model's vocabulary"), json);
    }

    if (one_shot_prompt) {
        if (!json && !no_stream) {
            options.value().on_text = [](std::string_view piece) {
                std::cout.write(piece.data(), static_cast<std::streamsize>(piece.size()));
                std::cout.flush();
            };
        }
        auto result = loaded.value().model.generate(*one_shot_prompt, options.value(), *loaded.value().governor);
        if (!result) return print_error(result.error(), json);
        if (json) {
            std::cout << inference::generation_result_to_json(result.value()) << '\n';
        } else {
            if (no_stream) std::cout << result.value().text;
            std::cout << '\n';
            if (!no_stats) {
                std::cout.flush();
                print_stats(result.value(), loaded.value().load_ms, std::cerr);
            }
        }
        return result.value().stats.stopped_on_timeout ? 75 : 0;
    }

    std::cout << "Vastloom Local AI " << kVersion << " — TinyChat CLI\n"
              << "Model: " << loaded.value().model.config().model_id << " ("
              << loaded.value().model.config().context_length << " token context)\n"
              << "Weights: " << (loaded.value().manifest_verified ? "SHA-256 verified" : "direct file; no manifest hash") << "\n"
              << "This validation model is stateless: each line is an independent prompt.\n"
              << "Type /help for commands.\n\n";

    std::optional<GenerationResult> last_result;
    std::string active_system_prompt = options.value().system_prompt_override;
    for (;;) {
        std::cout << "you> " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) {
            std::cout << '\n';
            break;
        }
        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/help") {
            print_chat_help();
            continue;
        }
        if (line == "/model") {
            std::cout << inference::tiny_model_config_to_text(loaded.value().model.config()) << '\n';
            continue;
        }
        if (line == "/stats") {
            if (last_result) print_stats(*last_result, 0.0, std::cout);
            else std::cout << "No response statistics are available yet.\n";
            continue;
        }
        if (line == "/reset") {
            active_system_prompt.clear();
            last_result.reset();
            std::cout << "System prompt restored to the model default.\n";
            continue;
        }
        if (util::starts_with(line, "/system ")) {
            active_system_prompt = line.substr(8U);
            std::cout << "System prompt override updated.\n";
            continue;
        }
        if (line == "/system") {
            std::cout << "Current system prompt: "
                      << (active_system_prompt.empty() ? loaded.value().model.config().system_prompt : active_system_prompt)
                      << '\n';
            continue;
        }

        GenerationOptions turn_options = options.value();
        turn_options.system_prompt_override = active_system_prompt;
        std::cout << "model> " << std::flush;
        if (!no_stream) {
            turn_options.on_text = [](std::string_view piece) {
                std::cout.write(piece.data(), static_cast<std::streamsize>(piece.size()));
                std::cout.flush();
            };
        }
        auto result = loaded.value().model.generate(line, turn_options, *loaded.value().governor);
        if (!result) {
            std::cout << '\n';
            print_error(result.error());
            continue;
        }
        if (no_stream) std::cout << result.value().text;
        std::cout << '\n';
        if (!no_stats) print_stats(result.value(), 0.0, std::cerr);
        if (result.value().stats.stopped_on_timeout) {
            std::cerr << "Generation stopped safely after reaching the configured timeout.\n";
        }
        last_result = result.take_value();
    }
    return 0;
}

int command_serve(const std::vector<std::string>& args) {
    if (args.size() == 1U && (args.front() == "--help" || args.front() == "-h")) {
        std::cout << "Usage: vastloom serve MODEL_OR_MANIFEST [--port N] [--no-verify]\n";
        return 0;
    }
    if (args.empty() || util::starts_with(args.front(), "--") || args.front() == "-h") {
        return print_error(make_error(ErrorCode::invalid_argument, "serve requires a model or manifest path"));
    }
    auto validation = validate_options(args, {"--port"}, {"--no-verify"});
    if (!validation) return print_error(validation.error());
    const bool verify_hash = !has_flag(args, "--no-verify");
    auto port = parse_size_option(args, "--port", 8099U, 0U, 65535U);
    if (!port) return print_error(port.error());
    auto loaded = load_model(args[0], verify_hash);
    if (!loaded) return print_error(loaded.error());

    auto* model = &loaded.value().model;
    auto* governor = loaded.value().governor.get();
    const std::string model_id = model->config().model_id;
    api::LoopbackHandler handler = [model, governor, model_id](const api::HttpRequest& request) {
        api::HttpResponse response;
        const auto now = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
        if (request.method == "GET" && request.path == "/v1/models") {
            auto body = api::serialize_model_list(model_id, now);
            if (!body) { response.status = 500; response.body = api::serialize_error(body.error().message, "api_error", 500); return response; }
            response.body = body.take_value();
            return response;
        }
        if (request.method == "POST" && request.path == "/v1/chat/completions") {
            auto parsed = api::parse_chat_completion_request(request.body);
            if (!parsed) {
                response.status = api::http_status_for(parsed.error().code);
                response.body = api::serialize_error(parsed.error().message, "invalid_request_error", response.status);
                return response;
            }
            std::string prompt;
            for (const auto& message : parsed.value().messages) {
                if (!prompt.empty()) prompt += '\n';
                prompt += message.content;
            }
                if (parsed.value().stream) {
                    api::ChatCompletionResponse skeleton;
                    skeleton.id = "chatcmpl-vastloom";
                    skeleton.model = model_id;
                    skeleton.created = now;
                    const std::size_t stream_max = parsed.value().max_tokens;
                    const float stream_temperature =
                        parsed.value().temperature.has_value()
                            ? static_cast<float>(*parsed.value().temperature)
                            : 0.0F;
                    response.streaming = true;
                    response.status = 200;
                    response.produce = [model, governor, prompt, skeleton, stream_max,
                                        stream_temperature, stops = parsed.value().stop](const std::function<bool(std::string_view)>& emit) {
                        GenerationOptions options;
                        options.max_new_tokens = stream_max;
                        options.temperature = stream_temperature;
                    auto skel = skeleton;
                    bool first = true;
                    bool alive = true;
                    api::StopSequenceFilter stop_filter(stops);
                    options.on_text = [&](std::string_view piece) {
                        if (!alive) return;
                        const auto safe = stop_filter.feed(piece);
                        if (safe.empty()) return;
                        auto chunk = api::serialize_chat_completion_chunk(skel, safe, first, false);
                        first = false;
                        if (!chunk) { alive = false; return; }
                        alive = emit("data: " + chunk.take_value() + "\n\n");
                    };
                    auto result = model->generate(prompt, options, *governor);
                    if (alive && !stop_filter.stopped() && result) {
                        const auto tail = stop_filter.flush();
                        if (!tail.empty()) {
                            auto chunk = api::serialize_chat_completion_chunk(skel, tail, first, false);
                            first = false;
                            if (!chunk) {
                                alive = false;
                            } else {
                                alive = emit("data: " + chunk.take_value() + "\n\n");
                            }
                        }
                    }
                    skel.finish_reason = (result && !stop_filter.stopped() &&
                                          result.value().stats.stopped_on_max_tokens)
                                                 ? "length" : "stop";
                    if (alive) {
                        auto last = api::serialize_chat_completion_chunk(skel, "", false, true);
                        if (last) alive = emit("data: " + last.take_value() + "\n\n");
                    }
                    if (alive) emit("data: [DONE]\n\n");
                };
                return response;
            }
            GenerationOptions options;
            options.max_new_tokens = parsed.value().max_tokens;
            if (parsed.value().temperature.has_value()) {
                options.temperature = static_cast<float>(*parsed.value().temperature);
            }
            auto result = model->generate(prompt, options, *governor);
            if (!result) {
                response.status = api::http_status_for(result.error().code);
                response.body = api::serialize_error(result.error().message, "api_error", response.status);
                return response;
            }
            api::ChatCompletionResponse completion;
            completion.id = "chatcmpl-vastloom";
            completion.model = model_id;
            api::StopSequenceFilter stop_filter(parsed.value().stop);
            completion.content = stop_filter.apply(result.value().text);
            completion.finish_reason =
                (result.value().stats.stopped_on_max_tokens && !stop_filter.stopped()) ? "length" : "stop";
            completion.created = now;
            completion.prompt_tokens = result.value().stats.prompt_tokens;
            completion.completion_tokens = result.value().stats.generated_tokens;
            auto body = api::serialize_chat_completion_response(completion);
            if (!body) { response.status = 500; response.body = api::serialize_error(body.error().message, "api_error", 500); return response; }
            response.body = body.take_value();
            return response;
        }
        response.status = 404;
        response.body = api::serialize_error("unknown route", "not_found", 404);
        return response;
    };

    api::LoopbackServer server;
    auto started = server.start(static_cast<std::uint16_t>(port.value()), std::move(handler), 1U * 1024U * 1024U);
    if (!started) return print_error(started.error());
    std::cout << "Vastloom loopback API serving model " << model_id << " on http://127.0.0.1:"
              << server.port() << " (Ctrl+C to stop)\n"
              << std::flush;
    while (server.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}

}  // namespace vastloom::cli
