#include "stargate/safety/heartbeat.hpp"

#include <charconv>
#include <chrono>
#include <fstream>
#include <map>
#include <sstream>

#include "stargate/util/string.hpp"
#include "stargate/version.hpp"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace stargate::safety {
namespace {

Result<std::uint64_t> parse_u64(std::string_view value, std::string_view key) {
    const std::string normalized = util::trim(value);
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(normalized.data(), normalized.data() + normalized.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != normalized.data() + normalized.size()) {
        return make_error(ErrorCode::invalid_format, "Invalid heartbeat integer", std::string(key));
    }
    return parsed;
}

Result<SafetyState> parse_state(std::string_view value) {
    const auto normalized = util::to_lower(util::trim(value));
    if (normalized == "normal") return SafetyState::normal;
    if (normalized == "pressure") return SafetyState::pressure;
    if (normalized == "critical") return SafetyState::critical;
    if (normalized == "emergency") return SafetyState::emergency;
    return make_error(ErrorCode::invalid_format, "Unknown heartbeat safety state", normalized);
}

}  // namespace

std::uint64_t unix_time_ms() noexcept {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

Result<void> write_heartbeat_atomic(const std::filesystem::path& path,
                                    const HeartbeatRecord& heartbeat) {
    if (path.empty()) {
        return make_error(ErrorCode::invalid_argument, "Heartbeat path must not be empty");
    }
    std::error_code error;
    const auto parent = path.parent_path().empty() ? std::filesystem::current_path(error) : path.parent_path();
    if (error || !std::filesystem::exists(parent)) {
        return make_error(ErrorCode::not_found, "Heartbeat parent directory does not exist", parent.string());
    }

    const auto temporary = path.string() + ".tmp." + std::to_string(heartbeat.pid);
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            return make_error(ErrorCode::io_error, "Unable to create temporary heartbeat", temporary);
        }
        output << kHeartbeatFormat << '\n'
               << "pid=" << heartbeat.pid << '\n'
               << "sequence=" << heartbeat.sequence << '\n'
               << "unix_time_ms=" << heartbeat.unix_time_ms << '\n'
               << "state=" << util::to_lower(to_string(heartbeat.state)) << '\n'
               << "ram_reserved=" << heartbeat.ram_reserved << '\n'
               << "vram_reserved=" << heartbeat.vram_reserved << '\n';
        output.flush();
        if (!output) {
            return make_error(ErrorCode::io_error, "Unable to flush temporary heartbeat", temporary);
        }
    }

#if defined(_WIN32)
    if (MoveFileExW(std::filesystem::path(temporary).wstring().c_str(), path.wstring().c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        std::filesystem::remove(temporary, error);
        return make_error(ErrorCode::io_error, "Atomic heartbeat replacement failed", path.string());
    }
#else
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return make_error(ErrorCode::io_error, "Atomic heartbeat replacement failed", path.string());
    }
#endif
    return {};
}

Result<HeartbeatRecord> read_heartbeat(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        return make_error(ErrorCode::not_found, "Unable to stat heartbeat", path.string());
    }
    if (size > 64U * 1024U) {
        return make_error(ErrorCode::invalid_format, "Heartbeat exceeds the 64 KiB safety limit", path.string());
    }
    std::ifstream input(path);
    if (!input) {
        return make_error(ErrorCode::io_error, "Unable to open heartbeat", path.string());
    }
    std::string magic;
    if (!std::getline(input, magic) || util::trim(magic) != kHeartbeatFormat) {
        return make_error(ErrorCode::invalid_format, "Invalid heartbeat magic", path.string());
    }

    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(input, line)) {
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            return make_error(ErrorCode::invalid_format, "Invalid heartbeat key/value line", path.string());
        }
        const auto key = util::trim(std::string_view(line).substr(0, separator));
        const auto value = util::trim(std::string_view(line).substr(separator + 1));
        if (!values.emplace(key, value).second) {
            return make_error(ErrorCode::invalid_format, "Duplicate heartbeat key", key);
        }
    }

    for (const auto* required : {"pid", "sequence", "unix_time_ms", "state", "ram_reserved", "vram_reserved"}) {
        if (!values.contains(required)) {
            return make_error(ErrorCode::invalid_format, "Missing heartbeat key", required);
        }
    }

    HeartbeatRecord heartbeat;
    auto pid = parse_u64(values["pid"], "pid"); if (!pid) return pid.error(); heartbeat.pid = pid.take_value();
    auto sequence = parse_u64(values["sequence"], "sequence"); if (!sequence) return sequence.error(); heartbeat.sequence = sequence.take_value();
    auto timestamp = parse_u64(values["unix_time_ms"], "unix_time_ms"); if (!timestamp) return timestamp.error(); heartbeat.unix_time_ms = timestamp.take_value();
    auto state = parse_state(values["state"]); if (!state) return state.error(); heartbeat.state = state.take_value();
    auto ram = parse_u64(values["ram_reserved"], "ram_reserved"); if (!ram) return ram.error(); heartbeat.ram_reserved = ram.take_value();
    auto vram = parse_u64(values["vram_reserved"], "vram_reserved"); if (!vram) return vram.error(); heartbeat.vram_reserved = vram.take_value();
    return heartbeat;
}

}  // namespace stargate::safety
