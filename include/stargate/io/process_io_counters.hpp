#pragma once

#include <cstdint>
#include <string>

#include "stargate/core/bytes.hpp"
#include "stargate/core/status.hpp"

namespace stargate::io {

struct ProcessIoSnapshot {
    std::string source;
    std::uint64_t read_operations{0U};
    ByteCount process_read_bytes{0U};
    bool storage_read_bytes_available{false};
    ByteCount storage_read_bytes{0U};
};

struct ProcessIoDelta {
    std::string source;
    std::uint64_t read_operations{0U};
    ByteCount process_read_bytes{0U};
    bool storage_read_bytes_available{false};
    ByteCount storage_read_bytes{0U};
};

[[nodiscard]] Result<ProcessIoSnapshot> sample_process_io();
[[nodiscard]] Result<ProcessIoDelta> process_io_delta(const ProcessIoSnapshot& before,
                                                       const ProcessIoSnapshot& after);

}  // namespace stargate::io
