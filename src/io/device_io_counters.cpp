#include "vastloom/io/device_io_counters.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#elif defined(__linux__)
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

namespace vastloom::io {
namespace {

#if defined(_WIN32)
class UniqueHandle {
public:
    explicit UniqueHandle(HANDLE handle = INVALID_HANDLE_VALUE) noexcept : handle_(handle) {}
    ~UniqueHandle() {
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
    }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    [[nodiscard]] HANDLE get() const noexcept { return handle_; }
    [[nodiscard]] bool valid() const noexcept { return handle_ != INVALID_HANDLE_VALUE; }

private:
    HANDLE handle_;
};

Result<std::wstring> volume_device_path(const std::filesystem::path& input) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(input, error);
    if (error) {
        return make_error(ErrorCode::io_error, "Unable to resolve device-counter path", input.string());
    }
    while (!std::filesystem::exists(absolute, error) && absolute.has_parent_path()) {
        absolute = absolute.parent_path();
        error.clear();
    }
    if (error || !std::filesystem::exists(absolute, error)) {
        return make_error(ErrorCode::not_found, "No existing path for device counter", input.string());
    }

    std::vector<wchar_t> root(32'768U, L'\0');
    if (GetVolumePathNameW(absolute.c_str(), root.data(), static_cast<DWORD>(root.size())) == 0) {
        return make_error(ErrorCode::unsupported, "Unable to map path to a Windows volume",
                          std::to_string(GetLastError()));
    }
    const std::wstring volume_root(root.data());
    if (volume_root.size() < 2U || volume_root[1] != L':') {
        return make_error(ErrorCode::unsupported,
                          "Device counters currently require a local drive-letter volume");
    }
    return std::wstring(L"\\\\.\\") + volume_root.substr(0U, 2U);
}

Result<DeviceIoSnapshot> sample_windows_device(const std::filesystem::path& path) {
    auto volume_path = volume_device_path(path);
    if (!volume_path) return volume_path.error();

    UniqueHandle volume(CreateFileW(volume_path.value().c_str(), 0U,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, 0U, nullptr));
    if (!volume.valid()) {
        return make_error(ErrorCode::unsupported, "Unable to open Windows checkpoint volume",
                          std::to_string(GetLastError()));
    }

    STORAGE_DEVICE_NUMBER number{};
    DWORD returned = 0U;
    if (DeviceIoControl(volume.get(), IOCTL_STORAGE_GET_DEVICE_NUMBER, nullptr, 0U,
                        &number, static_cast<DWORD>(sizeof(number)), &returned, nullptr) == 0 ||
        returned < static_cast<DWORD>(sizeof(number))) {
        return make_error(ErrorCode::unsupported, "Unable to identify checkpoint physical drive",
                          std::to_string(GetLastError()));
    }
    if (number.DeviceType != FILE_DEVICE_DISK) {
        return make_error(ErrorCode::unsupported, "Checkpoint volume is not backed by a disk device");
    }

    const std::wstring disk_path = L"\\\\.\\PhysicalDrive" + std::to_wstring(number.DeviceNumber);
    UniqueHandle disk(CreateFileW(disk_path.c_str(), 0U,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING, 0U, nullptr));
    if (!disk.valid()) {
        return make_error(ErrorCode::unsupported, "Unable to open Windows physical-drive counter",
                          std::to_string(GetLastError()));
    }

    DISK_PERFORMANCE performance{};
    returned = 0U;
    if (DeviceIoControl(disk.get(), IOCTL_DISK_PERFORMANCE, nullptr, 0U,
                        &performance, static_cast<DWORD>(sizeof(performance)), &returned, nullptr) == 0 ||
        returned < static_cast<DWORD>(sizeof(performance))) {
        return make_error(ErrorCode::unsupported, "Windows physical-drive counters are unavailable",
                          std::to_string(GetLastError()));
    }
    if (performance.BytesRead.QuadPart < 0) {
        return make_error(ErrorCode::integrity_error, "Windows physical-drive byte counter is negative");
    }
    return DeviceIoSnapshot{
        "windows-ioctl-disk-performance",
        "PhysicalDrive" + std::to_string(number.DeviceNumber),
        "whole-device",
        static_cast<std::uint64_t>(performance.ReadCount),
        static_cast<ByteCount>(performance.BytesRead.QuadPart),
    };
}
#elif defined(__linux__)
Result<std::uint64_t> parse_u64(std::string_view value, std::string_view field) {
    std::uint64_t parsed_value = 0U;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), parsed_value);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        return make_error(ErrorCode::invalid_format, "Invalid Linux block-device counter", std::string(field));
    }
    return parsed_value;
}

Result<DeviceIoSnapshot> sample_linux_device(const std::filesystem::path& input) {
    std::error_code error;
    auto existing = std::filesystem::absolute(input, error);
    if (error) return make_error(ErrorCode::io_error, "Unable to resolve device-counter path", input.string());
    while (!std::filesystem::exists(existing, error) && existing.has_parent_path()) {
        existing = existing.parent_path();
        error.clear();
    }
    if (error || !std::filesystem::exists(existing, error)) {
        return make_error(ErrorCode::not_found, "No existing path for device counter", input.string());
    }

    struct stat status {};
    if (::stat(existing.c_str(), &status) != 0) {
        return make_error(ErrorCode::unsupported, "Unable to identify Linux checkpoint device",
                          existing.string());
    }
    const auto sysfs_link = std::filesystem::path("/sys/dev/block") /
                            (std::to_string(major(status.st_dev)) + ":" + std::to_string(minor(status.st_dev)));
    auto sysfs_device = std::filesystem::canonical(sysfs_link, error);
    if (error) {
        return make_error(ErrorCode::unsupported, "Linux checkpoint filesystem has no block-device counter",
                          sysfs_link.string());
    }
    const std::string canonical = sysfs_device.generic_string();
    if (canonical.find("/virtual/") != std::string::npos) {
        return make_error(ErrorCode::unsupported,
                          "Linux checkpoint filesystem maps to a virtual block device", canonical);
    }

    std::ifstream input_stat(sysfs_device / "stat");
    if (!input_stat) {
        return make_error(ErrorCode::unsupported, "Linux block-device statistics are unavailable",
                          (sysfs_device / "stat").string());
    }
    std::string line;
    if (!std::getline(input_stat, line) || line.empty() || line.size() > 1024U) {
        return make_error(ErrorCode::invalid_format, "Linux block-device statistics are invalid",
                          (sysfs_device / "stat").string());
    }
    std::istringstream values(line);
    std::vector<std::string> fields;
    std::string field;
    while (values >> field) {
        if (fields.size() >= 32U) {
            return make_error(ErrorCode::invalid_format, "Linux block-device statistics exceed bounded syntax");
        }
        fields.push_back(field);
    }
    if (fields.size() < 11U) {
        return make_error(ErrorCode::invalid_format, "Linux block-device statistics are incomplete");
    }
    auto reads = parse_u64(fields[0], "reads_completed");
    if (!reads) return reads.error();
    auto sectors = parse_u64(fields[2], "sectors_read");
    if (!sectors) return sectors.error();
    if (sectors.value() > std::numeric_limits<ByteCount>::max() / 512U) {
        return make_error(ErrorCode::invalid_format, "Linux block-device byte counter overflows");
    }
    const bool partition = std::filesystem::exists(sysfs_device / "partition", error) && !error;
    return DeviceIoSnapshot{
        "linux-sysfs-block-stat",
        sysfs_device.filename().string(),
        partition ? "partition" : "whole-device",
        reads.value(),
        sectors.value() * 512U,
    };
}
#endif

}  // namespace

Result<DeviceIoSnapshot> sample_device_io(const std::filesystem::path& path) {
#if defined(_WIN32)
    return sample_windows_device(path);
#elif defined(__linux__)
    return sample_linux_device(path);
#else
    (void)path;
    return make_error(ErrorCode::unsupported, "Device I/O counters are unsupported on this operating system");
#endif
}

Result<DeviceIoDelta> device_io_delta(const DeviceIoSnapshot& before,
                                      const DeviceIoSnapshot& after) {
    if (before.source.empty() || before.device_id.empty() || before.scope.empty() ||
        before.source != after.source || before.device_id != after.device_id || before.scope != after.scope) {
        return make_error(ErrorCode::invalid_argument, "Device I/O snapshots are incompatible");
    }
    if (after.read_operations < before.read_operations || after.read_bytes < before.read_bytes) {
        return make_error(ErrorCode::integrity_error, "Device I/O counters moved backwards");
    }
    return DeviceIoDelta{
        after.source,
        after.device_id,
        after.scope,
        after.read_operations - before.read_operations,
        after.read_bytes - before.read_bytes,
    };
}

}  // namespace vastloom::io
