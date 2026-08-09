#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::io {

struct DeviceIoSnapshot {
    std::string source;
    std::string device_id;
    std::string scope;
    std::uint64_t read_operations{0U};
    ByteCount read_bytes{0U};
};

struct DeviceIoDelta {
    std::string source;
    std::string device_id;
    std::string scope;
    std::uint64_t read_operations{0U};
    ByteCount read_bytes{0U};
};

[[nodiscard]] Result<DeviceIoSnapshot> sample_device_io(const std::filesystem::path& path);
[[nodiscard]] Result<DeviceIoDelta> device_io_delta(const DeviceIoSnapshot& before,
                                                     const DeviceIoSnapshot& after);

}  // namespace vastloom::io
