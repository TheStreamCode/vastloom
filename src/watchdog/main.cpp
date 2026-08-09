#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

#include "vastloom/safety/heartbeat.hpp"
#include "vastloom/version.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

namespace {

bool process_alive(std::uint64_t pid) {
#if defined(_WIN32)
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (process == nullptr) return false;
    DWORD exit_code = 0;
    const bool alive = GetExitCodeProcess(process, &exit_code) != 0 && exit_code == STILL_ACTIVE;
    CloseHandle(process);
    return alive;
#else
    if (pid > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) return false;
    const int result = kill(static_cast<pid_t>(pid), 0);
    return result == 0 || errno == EPERM;
#endif
}

bool terminate_process(std::uint64_t pid, std::chrono::milliseconds grace) {
#if defined(_WIN32)
    HANDLE process = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
    if (process == nullptr) return false;
    const bool result = TerminateProcess(process, 86) != 0;
    WaitForSingleObject(process, static_cast<DWORD>(grace.count()));
    CloseHandle(process);
    return result;
#else
    if (pid > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) return false;
    const auto native_pid = static_cast<pid_t>(pid);
    if (kill(native_pid, SIGTERM) != 0 && errno != ESRCH) return false;
    const auto deadline = std::chrono::steady_clock::now() + grace;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!process_alive(pid)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return kill(native_pid, SIGKILL) == 0 || errno == ESRCH;
#endif
}

void usage() {
    std::cout << "vastloom-watchdog " << vastloom::kVersion << "\n"
              << "Usage: vastloom-watchdog --pid PID --heartbeat PATH [--timeout-ms 5000]"
                 " [--interval-ms 500] [--grace-ms 3000] [--enforce]\n"
              << "Without --enforce, the watchdog only reports a stale heartbeat and exits.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::uint64_t pid = 0;
    std::string heartbeat_path;
    std::uint64_t timeout_ms = 5000;
    std::uint64_t interval_ms = 500;
    std::uint64_t grace_ms = 3000;
    bool enforce = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto require_value = [&](const char* name) -> std::string {
            if (index + 1 >= argc) {
                std::cerr << "Missing value for " << name << '\n';
                std::exit(64);
            }
            return argv[++index];
        };
        try {
            if (argument == "--pid") pid = std::stoull(require_value("--pid"));
            else if (argument == "--heartbeat") heartbeat_path = require_value("--heartbeat");
            else if (argument == "--timeout-ms") timeout_ms = std::stoull(require_value("--timeout-ms"));
            else if (argument == "--interval-ms") interval_ms = std::stoull(require_value("--interval-ms"));
            else if (argument == "--grace-ms") grace_ms = std::stoull(require_value("--grace-ms"));
            else if (argument == "--enforce") enforce = true;
            else if (argument == "--help" || argument == "-h") { usage(); return 0; }
            else { std::cerr << "Unknown argument: " << argument << '\n'; return 64; }
        } catch (const std::exception&) {
            std::cerr << "Invalid numeric argument for " << argument << '\n';
            return 64;
        }
    }

    if (pid == 0 || heartbeat_path.empty() || timeout_ms < 250 || interval_ms < 50) {
        usage();
        return 64;
    }

    while (process_alive(pid)) {
        auto heartbeat = vastloom::safety::read_heartbeat(heartbeat_path);
        if (heartbeat && heartbeat.value().pid != pid) {
            std::cerr << "Heartbeat PID does not match watched PID; refusing to act.\n";
            return 65;
        }
        const auto now = vastloom::safety::unix_time_ms();
        const bool stale = !heartbeat || now < heartbeat.value().unix_time_ms ||
                           now - heartbeat.value().unix_time_ms > timeout_ms;
        if (stale) {
            std::cerr << "Vastloom heartbeat is stale or unreadable.";
            if (!enforce) {
                std::cerr << " Observe-only mode: no process signal was sent.\n";
                return 2;
            }
            std::cerr << " Terminating the runtime to protect the host.\n";
            return terminate_process(pid, std::chrono::milliseconds(grace_ms)) ? 3 : 4;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    return 0;
}
