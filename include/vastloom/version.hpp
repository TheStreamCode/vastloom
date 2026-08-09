#pragma once

#include <string_view>

namespace vastloom {
inline constexpr std::string_view kVersion = "0.4.0-alpha.4";
inline constexpr std::string_view kManifestFormat = "vastloom.manifest.v1";
inline constexpr std::string_view kLegacyManifestFormat = "stargate.manifest.v1";
inline constexpr std::string_view kImportManifestFormat = "vastloom.hf-import.v1";
inline constexpr std::string_view kLegacyImportManifestFormat = "stargate.hf-import.v1";
inline constexpr std::string_view kHeartbeatFormat = "SGHB1";
}  // namespace vastloom
