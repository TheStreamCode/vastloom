#include "vastloom/io/process_io_counters.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <string_view>

#include "vastloom/util/string.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace vastloom::io {
namespace {

#if defined(__linux__)
Result<std::uint64_t> parse_counter(std::string_view value, std::string_view field) {
    const std::string normalized = util::trim(value);
    std::uint64_t result = 0U;
    const auto parsed = std::from_chars(normalized.data(), normalized.data() + normalized.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != normalized.data() + normalized.size()) {
        return make_error(ErrorCode::invalid_format, "Invalid /proc/self/io counter", std::string(field));
    }
    return result;
}
#endif

}  // namespace

Result<ProcessIoSnapshot> sample_process_io() {
#if defined(_WIN32)
    IO_COUNTERS counters{};
    if (GetProcessIoCounters(GetCurrentProcess(), &counters) == 0) {
        return make_error(ErrorCode::unsupported, "Windows process I/O counters are unavailable",
                          std::to_string(GetLastError()));
    }
    return ProcessIoSnapshot{
        "windows-get-process-io-counters",
        static_cast<std::uint64_t>(counters.ReadOperationCount),
        static_cast<ByteCount>(counters.ReadTransferCount),
        false,
        0U,
    };
#elif defined(__linux__)
    std::ifstream input("/proc/self/io");
    if (!input) {
        return make_error(ErrorCode::unsupported, "Linux process I/O counters are unavailable", "/proc/self/io");
    }
    std::map<std::string, std::uint64_t, std::less<>> counters;
    std::string line;
    std::size_t lines = 0U;
    while (std::getline(input, line)) {
        if (++lines > 32U || line.size() > 256U) {
            return make_error(ErrorCode::invalid_format, "Linux process I/O counters exceed bounded syntax");
        }
        const auto delimiter = line.find(':');
        if (delimiter == std::string::npos) continue;
        const std::string key = util::trim(std::string_view(line).substr(0U, delimiter));
        auto value = parse_counter(std::string_view(line).substr(delimiter + 1U), key);
        if (!value) return value.error();
        if (!counters.emplace(key, value.value()).second) {
            return make_error(ErrorCode::invalid_format, "Duplicate Linux process I/O counter", key);
        }
    }
    if (!input.eof()) return make_error(ErrorCode::io_error, "Unable to read Linux process I/O counters");
    const auto rchar = counters.find("rchar");
    const auto syscr = counters.find("syscr");
    const auto read_bytes = counters.find("read_bytes");
    if (rchar == counters.end() || syscr == counters.end() || read_bytes == counters.end()) {
        return make_error(ErrorCode::invalid_format, "Linux process I/O counters are incomplete");
    }
    return ProcessIoSnapshot{
        "linux-proc-self-io",
        syscr->second,
        rchar->second,
        true,
        read_bytes->second,
    };
#else
    return make_error(ErrorCode::unsupported, "Process I/O counters are unsupported on this operating system");
#endif
}

Result<ProcessIoDelta> process_io_delta(const ProcessIoSnapshot& before,
                                        const ProcessIoSnapshot& after) {
    if (before.source.empty() || before.source != after.source ||
        before.storage_read_bytes_available != after.storage_read_bytes_available) {
        return make_error(ErrorCode::invalid_argument, "Process I/O snapshots are incompatible");
    }
    if (after.read_operations < before.read_operations ||
        after.process_read_bytes < before.process_read_bytes ||
        (before.storage_read_bytes_available && after.storage_read_bytes < before.storage_read_bytes)) {
        return make_error(ErrorCode::integrity_error, "Process I/O counters moved backwards");
    }
    return ProcessIoDelta{
        after.source,
        after.read_operations - before.read_operations,
        after.process_read_bytes - before.process_read_bytes,
        after.storage_read_bytes_available,
        after.storage_read_bytes - before.storage_read_bytes,
    };
}

}  // namespace vastloom::io
