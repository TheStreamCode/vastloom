#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "stargate/core/bytes.hpp"
#include "stargate/core/status.hpp"
#include "stargate/safety/resource_governor.hpp"

namespace stargate::safety {

struct HeartbeatRecord {
    std::uint64_t pid{0};
    std::uint64_t sequence{0};
    std::uint64_t unix_time_ms{0};
    SafetyState state{SafetyState::normal};
    ByteCount ram_reserved{0};
    ByteCount vram_reserved{0};
};

[[nodiscard]] std::uint64_t unix_time_ms() noexcept;
[[nodiscard]] Result<void> write_heartbeat_atomic(const std::filesystem::path& path,
                                                  const HeartbeatRecord& heartbeat);
[[nodiscard]] Result<HeartbeatRecord> read_heartbeat(const std::filesystem::path& path);

}  // namespace stargate::safety
