#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>

namespace vastloom_test {

// Creates a unique scratch directory for one test binary. The system temporary directory is
// the default, but a host policy (for example a Controlled Folder Access rule) can deny writes
// there while the build tree stays writable, so the current directory is the fallback. The
// tests keep the same assertions either way; only the location changes.
inline std::filesystem::path make_temp_dir(const std::string& prefix) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string name = prefix + std::to_string(stamp);
    std::error_code error;
    const auto system_temp = std::filesystem::temp_directory_path(error);
    if (!error) {
        const auto candidate = system_temp / name;
        std::filesystem::create_directories(candidate, error);
        if (!error && std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    const auto fallback = std::filesystem::current_path() / name;
    std::filesystem::create_directories(fallback);
    return fallback;
}

}  // namespace vastloom_test
