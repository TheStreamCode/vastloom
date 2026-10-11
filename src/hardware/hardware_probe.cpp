#include "vastloom/hardware/hardware_probe.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>
#include <thread>

#include "vastloom/util/string.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <cpuid.h>
#include <sys/utsname.h>
#endif

namespace vastloom::hardware {
namespace {

std::string detect_os() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    struct utsname info {};
    if (uname(&info) == 0) {
        return std::string("Linux ") + info.release;
    }
    return "Linux";
#else
    return "Unknown";
#endif
}

std::string first_cpu_model_line() {
#if defined(__linux__)
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        const auto key = util::to_lower(util::trim(std::string_view(line).substr(0, separator)));
        if (key == "model name" || key == "hardware") {
            return util::trim(std::string_view(line).substr(separator + 1));
        }
    }
#elif defined(_WIN32)
    char* value = nullptr;
    std::size_t value_size = 0;
    if (_dupenv_s(&value, &value_size, "PROCESSOR_IDENTIFIER") == 0 && value != nullptr) {
        const std::string result(value);
        std::free(value);
        return result;
    }
#elif defined(__APPLE__)
    std::size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) == 0 && size > 1U) {
        std::string value(size, '\0');
        if (sysctlbyname("machdep.cpu.brand_string", value.data(), &size, nullptr, 0) == 0) {
            while (!value.empty() && value.back() == '\0') {
                value.pop_back();
            }
            return value;
        }
    }
#endif
    return "Unknown CPU";
}

CpuInfo detect_cpu() {
    CpuInfo cpu;
    cpu.name = first_cpu_model_line();
    cpu.logical_cores = std::max(1U, std::thread::hardware_concurrency());

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4]{};
    __cpuidex(registers, 1, 0);
    cpu.fma = (registers[2] & (1 << 12)) != 0;
    __cpuidex(registers, 7, 0);
    cpu.avx2 = (registers[1] & (1 << 5)) != 0;
    cpu.avx512f = (registers[1] & (1 << 16)) != 0;
#elif (defined(__x86_64__) || defined(__i386__)) && !defined(_WIN32)
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0) {
        cpu.fma = (ecx & bit_FMA) != 0U;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0) {
        cpu.avx2 = (ebx & bit_AVX2) != 0U;
#ifdef bit_AVX512F
        cpu.avx512f = (ebx & bit_AVX512F) != 0U;
#endif
    }
#endif
    return cpu;
}

Result<MemoryInfo> detect_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) == 0) {
        return make_error(ErrorCode::io_error, "GlobalMemoryStatusEx failed");
    }
    return MemoryInfo{static_cast<ByteCount>(status.ullTotalPhys), static_cast<ByteCount>(status.ullAvailPhys)};
#elif defined(__APPLE__)
    std::uint64_t total = 0;
    std::size_t total_size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &total_size, nullptr, 0) != 0) {
        return make_error(ErrorCode::io_error, "Unable to query macOS physical memory");
    }
    // vm_statistics is intentionally avoided in this first portable probe. Conservatively expose 70% as available.
    return MemoryInfo{total, percentage(total, 0.70)};
#elif defined(__linux__)
    std::ifstream input("/proc/meminfo");
    if (!input) {
        return make_error(ErrorCode::io_error, "Unable to open /proc/meminfo");
    }
    ByteCount total = 0;
    ByteCount available = 0;
    std::string key;
    ByteCount value_kib = 0;
    std::string unit;
    while (input >> key >> value_kib >> unit) {
        if (key == "MemTotal:") {
            total = value_kib * KiB;
        } else if (key == "MemAvailable:") {
            available = value_kib * KiB;
        }
    }
    if (total == 0) {
        return make_error(ErrorCode::io_error, "MemTotal is missing from /proc/meminfo");
    }
    if (available == 0) {
        available = percentage(total, 0.50);
    }
    return MemoryInfo{total, available};
#else
    return make_error(ErrorCode::unsupported, "Memory probing is unsupported on this platform");
#endif
}

Result<StorageInfo> detect_storage(const std::filesystem::path& input_path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(input_path, error);
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to resolve storage path", input_path.string());
    }
    auto existing = absolute;
    while (!std::filesystem::exists(existing, error) && existing.has_parent_path() && existing.parent_path() != existing) {
        error.clear();
        existing = existing.parent_path();
    }
    if (error || !std::filesystem::exists(existing)) {
        return make_error(ErrorCode::not_found, "No existing parent for storage path", absolute.string());
    }
    const auto info = std::filesystem::space(existing, error);
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to query storage capacity", existing.string());
    }
    return StorageInfo{absolute,
                       static_cast<ByteCount>(info.capacity),
                       static_cast<ByteCount>(info.free),
                       static_cast<ByteCount>(info.available),
                       std::nullopt};
}

std::string capture_command(const char* command) {
#if defined(_WIN32)
    // A wedged driver must not wedge the whole CLI: every shell-out here is bounded by a
    // hard deadline and terminated, mirroring the timeout(1) guard on POSIX. _popen cannot
    // be given a deadline, so the child runs under CreateProcess with an explicit pipe.
    constexpr DWORD kCommandDeadlineMs = 3000U;
    SECURITY_ATTRIBUTES inheritable{};
    inheritable.nLength = sizeof(inheritable);
    inheritable.bInheritHandle = TRUE;
    HANDLE read_end = nullptr;
    HANDLE write_end = nullptr;
    if (CreatePipe(&read_end, &write_end, &inheritable, 0U) == 0) {
        return {};
    }
    // The parent's read end must not leak into the child.
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0U);

    std::string command_line = std::string("cmd.exe /c ") + command;
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_end;
    startup.hStdError = write_end;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    std::string output;
    const bool spawned = CreateProcessA(nullptr, command_line.data(), nullptr, nullptr, TRUE,
                                         CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                                         &process) != 0;
    // The child owns its end of the pipe now.
    CloseHandle(write_end);
    if (spawned) {
        std::thread reader([&] {
            std::array<char, 4096> buffer{};
            DWORD read_bytes = 0U;
            while (ReadFile(read_end, buffer.data(), static_cast<DWORD>(buffer.size()),
                            &read_bytes, nullptr) != 0 &&
                   read_bytes > 0U) {
                output.append(buffer.data(), static_cast<std::size_t>(read_bytes));
                if (output.size() > 1024U * 1024U) break;
            }
        });
        if (WaitForSingleObject(process.hProcess, kCommandDeadlineMs) == WAIT_TIMEOUT) {
            TerminateProcess(process.hProcess, 1U);
            WaitForSingleObject(process.hProcess, 500U);
        }
        reader.join();
        CloseHandle(process.hProcess);
        CloseHandle(process.hThread);
    }
    CloseHandle(read_end);
    return output;
#else
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) {
        return {};
    }
    std::string output;
    std::array<char, 4096> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output.append(buffer.data());
        if (output.size() > 1024U * 1024U) {
            break;
        }
    }
    pclose(pipe);
    return output;
#endif
}

std::optional<double> parse_double(std::string_view value) {
    const auto normalized = util::trim(value);
    if (normalized.empty() || util::to_lower(normalized) == "n/a") {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(normalized.c_str(), &end);
    if (end == normalized.c_str() || errno == ERANGE) {
        return std::nullopt;
    }
    return parsed;
}

std::vector<GpuInfo> detect_nvidia_gpus() {
#if defined(_WIN32)
    constexpr const char* command = "nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,power.limit,driver_version --format=csv,noheader,nounits 2>NUL";
#else
    constexpr const char* command = "timeout 3s nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,power.limit,driver_version --format=csv,noheader,nounits 2>/dev/null";
#endif
    const std::string output = capture_command(command);
    std::vector<GpuInfo> gpus;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        const auto fields = util::split(line, ',', true);
        if (fields.size() != 6U) {
            continue;
        }
        const auto total_mib = parse_double(fields[1]);
        const auto free_mib = parse_double(fields[2]);
        if (!total_mib.has_value() || !free_mib.has_value() || *total_mib < 0.0 || *free_mib < 0.0) {
            continue;
        }
        GpuInfo gpu;
        gpu.backend = "CUDA";
        gpu.name = util::trim(fields[0]);
        gpu.total_vram_bytes = static_cast<ByteCount>(*total_mib * static_cast<double>(MiB));
        gpu.free_vram_bytes = static_cast<ByteCount>(*free_mib * static_cast<double>(MiB));
        gpu.temperature_celsius = parse_double(fields[3]);
        gpu.power_limit_watts = parse_double(fields[4]);
        gpu.driver_version = util::trim(fields[5]);
        gpus.push_back(std::move(gpu));
    }
    return gpus;
}

}  // namespace

Result<HardwareSnapshot> HardwareProbe::collect(const std::filesystem::path& storage_path) {
    HardwareSnapshot snapshot;
    snapshot.operating_system = detect_os();
    snapshot.cpu = detect_cpu();

    auto memory = detect_memory();
    if (!memory) {
        return memory.error();
    }
    snapshot.memory = memory.take_value();

    auto storage = detect_storage(storage_path);
    if (!storage) {
        return storage.error();
    }
    snapshot.storage = storage.take_value();
    snapshot.gpus = detect_nvidia_gpus();

    if (snapshot.gpus.empty()) {
        snapshot.warnings.emplace_back("No NVIDIA GPU was detected through nvidia-smi; CPU/Vulkan/ROCm/Metal probing is planned for later milestones.");
    }
    if (!snapshot.cpu.avx2) {
        snapshot.warnings.emplace_back("AVX2 was not detected; large-model CPU kernels may be unavailable or significantly slower.");
    }
    if (snapshot.memory.available_bytes < 4U * GiB) {
        snapshot.warnings.emplace_back("Less than 4 GiB of RAM is currently available. Vastloom must not start a model in this state.");
    }
    return snapshot;
}

std::string to_json(const HardwareSnapshot& snapshot) {
    std::ostringstream output;
    output << '{'
           << "\"operating_system\":\"" << util::json_escape(snapshot.operating_system) << "\"," 
           << "\"cpu\":{"
           << "\"name\":\"" << util::json_escape(snapshot.cpu.name) << "\"," 
           << "\"logical_cores\":" << snapshot.cpu.logical_cores << ','
           << "\"avx2\":" << (snapshot.cpu.avx2 ? "true" : "false") << ','
           << "\"avx512f\":" << (snapshot.cpu.avx512f ? "true" : "false") << ','
           << "\"fma\":" << (snapshot.cpu.fma ? "true" : "false") << "},"
           << "\"memory\":{"
           << "\"total_bytes\":" << snapshot.memory.total_bytes << ','
           << "\"available_bytes\":" << snapshot.memory.available_bytes << "},"
           << "\"storage\":{"
           << "\"path\":\"" << util::json_escape(snapshot.storage.path.string()) << "\"," 
           << "\"capacity_bytes\":" << snapshot.storage.capacity_bytes << ','
           << "\"free_bytes\":" << snapshot.storage.free_bytes << ','
           << "\"available_bytes\":" << snapshot.storage.available_bytes << "},"
           << "\"gpus\":[";
    for (std::size_t index = 0; index < snapshot.gpus.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        const auto& gpu = snapshot.gpus[index];
        output << '{'
               << "\"backend\":\"" << util::json_escape(gpu.backend) << "\"," 
               << "\"name\":\"" << util::json_escape(gpu.name) << "\"," 
               << "\"driver_version\":\"" << util::json_escape(gpu.driver_version) << "\"," 
               << "\"total_vram_bytes\":" << gpu.total_vram_bytes << ','
               << "\"free_vram_bytes\":" << gpu.free_vram_bytes;
        if (gpu.temperature_celsius.has_value()) {
            output << ",\"temperature_celsius\":" << *gpu.temperature_celsius;
        }
        if (gpu.power_limit_watts.has_value()) {
            output << ",\"power_limit_watts\":" << *gpu.power_limit_watts;
        }
        output << '}';
    }
    output << "],\"warnings\":[";
    for (std::size_t index = 0; index < snapshot.warnings.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << '"' << util::json_escape(snapshot.warnings[index]) << '"';
    }
    output << "]}";
    return output.str();
}

std::string to_text(const HardwareSnapshot& snapshot) {
    std::ostringstream output;
    output << "Vastloom hardware report\n"
           << "OS:       " << snapshot.operating_system << '\n'
           << "CPU:      " << snapshot.cpu.name << " (" << snapshot.cpu.logical_cores << " logical cores)\n"
           << "ISA:      AVX2=" << (snapshot.cpu.avx2 ? "yes" : "no")
           << " FMA=" << (snapshot.cpu.fma ? "yes" : "no")
           << " AVX-512F=" << (snapshot.cpu.avx512f ? "yes" : "no") << '\n'
           << "RAM:      " << format_bytes(snapshot.memory.total_bytes)
           << " total, " << format_bytes(snapshot.memory.available_bytes) << " available\n"
           << "Storage:  " << snapshot.storage.path.string() << '\n'
           << "          " << format_bytes(snapshot.storage.capacity_bytes)
           << " capacity, " << format_bytes(snapshot.storage.available_bytes) << " available\n";
    if (snapshot.gpus.empty()) {
        output << "GPU:      not detected\n";
    } else {
        for (const auto& gpu : snapshot.gpus) {
            output << "GPU:      " << gpu.name << " [" << gpu.backend << "] — "
                   << format_bytes(gpu.total_vram_bytes) << " total, "
                   << format_bytes(gpu.free_vram_bytes) << " free";
            if (gpu.temperature_celsius.has_value()) {
                output << ", " << *gpu.temperature_celsius << " °C";
            }
            output << '\n';
        }
    }
    for (const auto& warning : snapshot.warnings) {
        output << "WARNING:  " << warning << '\n';
    }
    return output.str();
}

}  // namespace vastloom::hardware
