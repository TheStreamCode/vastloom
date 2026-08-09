#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::safety {

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

}  // namespace vastloom::safety
