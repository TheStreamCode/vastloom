#include "vastloom/inference/deepseek_v4_usage_profile.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <string_view>

#include "vastloom/util/json.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

constexpr ByteCount kMaximumUsageProfileBytes = 4U * MiB;

Result<std::string> required_string(const util::JsonValue& object,
                                    std::string_view key,
                                    std::size_t maximum = 4'096U) {
    const auto* value = object.find(key);
    if (value == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile is missing a string field",
                          std::string(key));
    }
    auto parsed = value->as_string();
    if (!parsed || parsed.value().empty() || parsed.value().size() > maximum) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile string field is invalid",
                          std::string(key));
    }
    return std::string(parsed.value());
}

Result<std::uint64_t> required_u64(const util::JsonValue& object, std::string_view key) {
    const auto* value = object.find(key);
    if (value == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile is missing a numeric field",
                          std::string(key));
    }
    auto parsed = value->as_u64();
    if (!parsed) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile numeric field is invalid",
                          std::string(key));
    }
    return parsed.value();
}

}  // namespace

Result<DeepseekV4UsageProfile> load_deepseek_v4_usage_profile(
    const std::filesystem::path& path,
    std::string_view model_id,
    std::string_view revision,
    const DeepseekV4UsageProfileOptions& options) {
    if (model_id.empty() || revision.empty() || options.max_layers == 0U ||
        options.max_experts == 0U || options.max_observations == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "Usage profile identity or bounds are invalid");
    }
    DeepseekV4UsageProfile profile;
    profile.model_id = std::string(model_id);
    profile.revision = std::string(revision);
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        if (error) {
            return make_error(ErrorCode::io_error,
                              "Unable to inspect usage profile path",
                              path.string());
        }
        return profile;
    }
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes == 0U || bytes > kMaximumUsageProfileBytes) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile size is outside its safety bound",
                          path.string());
    }
    util::JsonLimits limits;
    limits.max_input_bytes = kMaximumUsageProfileBytes;
    limits.max_depth = 16U;
    limits.max_nodes = 200'000U;
    limits.max_string_bytes = 1U * MiB;
    limits.max_array_elements = options.max_layers * options.max_experts;
    limits.max_object_members = options.max_layers * options.max_experts;
    auto manifest = util::load_json_file(path, limits);
    if (!manifest) return manifest.error();

    auto format = required_string(manifest.value(), "format", 128U);
    auto file_model = required_string(manifest.value(), "model_id");
    auto file_revision = required_string(manifest.value(), "revision");
    auto observations = required_u64(manifest.value(), "observations");
    if (!format) return format.error();
    if (!file_model) return file_model.error();
    if (!file_revision) return file_revision.error();
    if (!observations) return observations.error();
    if (format.value() != kDeepseekV4UsageProfileFormat ||
        file_model.value() != model_id || file_revision.value() != revision ||
        observations.value() > options.max_observations) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile identity or observation count is invalid",
                          path.string());
    }
    profile.observations = observations.value();

    const auto* layers_value = manifest.value().find("layers");
    if (layers_value == nullptr) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile is missing its layer map",
                          path.string());
    }
    auto layers = layers_value->as_object();
    if (!layers) return layers.error();
    if (layers.value()->size() > options.max_layers) {
        return make_error(ErrorCode::invalid_format,
                          "Usage profile has too many layers",
                          path.string());
    }
    for (const auto& [layer_key, layer_value] : *layers.value()) {
        std::uint64_t layer_number = 0U;
        try {
            layer_number = std::stoull(layer_key);
        } catch (const std::exception&) {
            return make_error(ErrorCode::invalid_format,
                              "Usage profile layer key is not an integer",
                              layer_key);
        }
        if (layer_number >= options.max_layers) {
            return make_error(ErrorCode::invalid_format,
                              "Usage profile layer is outside its bound",
                              layer_key);
        }
        auto experts = layer_value.as_object();
        if (!experts) return experts.error();
        if (experts.value()->size() > options.max_experts) {
            return make_error(ErrorCode::invalid_format,
                              "Usage profile has too many experts in a layer",
                              layer_key);
        }
        auto& counts = profile.counts[static_cast<std::uint32_t>(layer_number)];
        for (const auto& [expert_key, count_value] : *experts.value()) {
            std::uint64_t expert_number = 0U;
            try {
                expert_number = std::stoull(expert_key);
            } catch (const std::exception&) {
                return make_error(ErrorCode::invalid_format,
                                  "Usage profile expert key is not an integer",
                                  expert_key);
            }
            if (expert_number >= options.max_experts) {
                return make_error(ErrorCode::invalid_format,
                                  "Usage profile expert is outside its bound",
                                  expert_key);
            }
            auto count = count_value.as_u64();
            if (!count || count.value() == 0U) {
                return make_error(ErrorCode::invalid_format,
                                  "Usage profile expert count is invalid",
                                  expert_key);
            }
            counts[static_cast<std::uint32_t>(expert_number)] = count.value();
        }
    }
    return profile;
}

Result<void> record_deepseek_v4_usage(
    DeepseekV4UsageProfile& profile,
    std::uint32_t layer,
    std::span<const std::uint32_t> experts,
    const DeepseekV4UsageProfileOptions& options) {
    if (layer >= options.max_layers || experts.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "Usage profile record is outside its bounds",
                          std::to_string(layer));
    }
    if (profile.observations == std::numeric_limits<std::uint64_t>::max()) {
        return make_error(ErrorCode::resource_exhausted,
                          "Usage profile observation counter overflow");
    }
    ++profile.observations;
    auto& counts = profile.counts[layer];
    for (const auto expert : experts) {
        if (expert >= options.max_experts) {
            return make_error(ErrorCode::invalid_argument,
                              "Usage profile expert is outside its bound",
                              std::to_string(expert));
        }
        auto& count = counts[expert];
        if (count == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::resource_exhausted,
                              "Usage profile expert counter overflow",
                              std::to_string(expert));
        }
        ++count;
    }
    return {};
}

Result<void> save_deepseek_v4_usage_profile(const std::filesystem::path& path,
                                            const DeepseekV4UsageProfile& profile) {
    if (profile.model_id.empty() || profile.revision.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "Usage profile identity is missing");
    }
    std::ostringstream output;
    output << "{\"format\":\"" << kDeepseekV4UsageProfileFormat
           << "\",\"model_id\":\"" << util::json_escape(profile.model_id)
           << "\",\"revision\":\"" << util::json_escape(profile.revision)
           << "\",\"observations\":" << profile.observations << ",\"layers\":{";
    bool first_layer = true;
    for (const auto& [layer, counts] : profile.counts) {
        if (!first_layer) output << ',';
        first_layer = false;
        output << '"' << layer << "\":{";
        bool first_expert = true;
        for (const auto& [expert, count] : counts) {
            if (!first_expert) output << ',';
            first_expert = false;
            output << '"' << expert << "\":" << count;
        }
        output << '}';
    }
    output << "}}\n";
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            return make_error(ErrorCode::io_error,
                              "Unable to create usage profile temporary file",
                              temporary.string());
        }
        file << output.str();
        if (!file) {
            return make_error(ErrorCode::io_error,
                              "Unable to write usage profile temporary file",
                              temporary.string());
        }
    }
    std::error_code error;
    std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path, error);
    if (error) {
        return make_error(ErrorCode::io_error,
                          "Unable to publish usage profile",
                          path.string());
    }
    return {};
}

std::vector<std::pair<std::uint32_t, std::uint64_t>> top_deepseek_v4_usage(
    const DeepseekV4UsageProfile& profile,
    std::uint32_t layer,
    std::size_t limit) {
    std::vector<std::pair<std::uint32_t, std::uint64_t>> result;
    const auto found = profile.counts.find(layer);
    if (found == profile.counts.end() || limit == 0U) return result;
    result.reserve(found->second.size());
    for (const auto& [expert, count] : found->second) {
        result.emplace_back(expert, count);
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) return left.second > right.second;
        return left.first < right.first;
    });
    if (result.size() > limit) result.resize(limit);
    return result;
}

}  // namespace vastloom::inference
