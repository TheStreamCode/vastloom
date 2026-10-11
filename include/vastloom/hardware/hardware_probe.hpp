#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::hardware {

struct CpuInfo {
    std::string name{"Unknown CPU"};
    unsigned logical_cores{1};
    bool avx2{false};
    bool avx512f{false};
    bool fma{false};
};

struct MemoryInfo {
    ByteCount total_bytes{0};
    ByteCount available_bytes{0};
};

struct GpuInfo {
    std::string backend;
    std::string name;
    std::string driver_version;
    ByteCount total_vram_bytes{0};
    ByteCount free_vram_bytes{0};
    std::optional<double> temperature_celsius;
    std::optional<double> power_limit_watts;
};

struct StorageInfo {
    std::filesystem::path path;
    ByteCount capacity_bytes{0};
    ByteCount free_bytes{0};
    ByteCount available_bytes{0};
    std::optional<double> measured_read_gib_per_second;
};

struct HardwareSnapshot {
    std::string operating_system;
    CpuInfo cpu;
    MemoryInfo memory;
    std::vector<GpuInfo> gpus;
    StorageInfo storage;
    std::vector<std::string> warnings;
};

class HardwareProbe {
public:
    [[nodiscard]] static Result<HardwareSnapshot> collect(const std::filesystem::path& storage_path);
};

// Parses the exact `nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,
// power.limit,driver_version --format=csv,noheader,nounits` output into GpuInfo entries.
// Exposed as a pure seam so the parsing has an always-on test without a physical GPU;
// malformed rows are skipped exactly as the live probe skips them.
[[nodiscard]] std::vector<GpuInfo> parse_nvidia_smi_csv(std::string_view output);

[[nodiscard]] std::string to_json(const HardwareSnapshot& snapshot);
[[nodiscard]] std::string to_text(const HardwareSnapshot& snapshot);

}  // namespace vastloom::hardware
