#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "stargate/core/bytes.hpp"
#include "stargate/core/status.hpp"
#include "stargate/inference/gpt2_tokenizer.hpp"
#include "stargate/inference/olmoe_model.hpp"
#include "stargate/io/bounded_io_pool.hpp"
#include "stargate/model/safetensors.hpp"
#include "stargate/safety/resource_governor.hpp"
#include "stargate/safety/runtime_policy.hpp"

namespace stargate::inference {

enum class ExpertCachePolicy : std::uint8_t {
    none,
    lru,
    lfu,
    static_hot,
};

enum class ExpertIoMode : std::uint8_t {
    synchronous,
    asynchronous,
};

[[nodiscard]] Result<ExpertCachePolicy> parse_expert_cache_policy(std::string_view value);
[[nodiscard]] std::string_view to_string(ExpertCachePolicy policy) noexcept;
[[nodiscard]] Result<ExpertIoMode> parse_expert_io_mode(std::string_view value);
[[nodiscard]] std::string_view to_string(ExpertIoMode mode) noexcept;

struct OlmoeRuntimeLoadOptions {
    bool verify_hashes{true};
    unsigned cpu_threads{1};
};

struct OlmoeForwardOptions {
    ExpertCachePolicy cache_policy{ExpertCachePolicy::lru};
    ByteCount expert_cache_bytes{256U * MiB};
    ExpertIoMode io_mode{ExpertIoMode::synchronous};
    std::size_t prefetch_depth{1U};
    std::chrono::milliseconds io_request_timeout{std::chrono::seconds(30)};
    io::IoFaultInjection io_fault;
    std::chrono::milliseconds timeout{std::chrono::minutes(10)};
    std::function<bool()> cancelled;
};

struct ExpertTraceEvent {
    std::size_t token_position{0};
    std::uint32_t layer{0};
    std::uint32_t expert{0};
    float routing_weight{0.0F};
    bool ram_cache_hit{false};
    bool prefetched{false};
    ByteCount bytes_read{0};
    double io_ms{0.0};
    double prefetch_wait_ms{0.0};
    double prefetch_overlap_ms{0.0};
};

struct OlmoeForwardStats {
    std::size_t input_tokens{0};
    double load_ms{0.0};
    double forward_ms{0.0};
    ByteCount expert_bytes_read{0};
    double expert_io_ms{0.0};
    std::size_t expert_cache_hits{0};
    std::size_t expert_cache_misses{0};
    std::size_t expert_prefetches{0};
    double expert_prefetch_wait_ms{0.0};
    double expert_prefetch_overlap_ms{0.0};
};

struct OlmoeForwardResult {
    std::vector<float> logits;
    std::vector<ExpertTraceEvent> trace;
    OlmoeForwardStats stats;
};

class OlmoeRuntimeModel {
public:
    struct WeightView {
        std::size_t offset{0};
        std::size_t rows{0};
        std::size_t columns{0};
    };
    struct StoredTensor {
        std::size_t offset{0};
        std::vector<std::uint64_t> shape;
    };
    struct ExpertLocations {
        model::TensorLocation gate;
        model::TensorLocation up;
        model::TensorLocation down;
    };
    struct LayerWeights {
        WeightView input_norm;
        WeightView q;
        WeightView k;
        WeightView v;
        WeightView o;
        WeightView q_norm;
        WeightView k_norm;
        WeightView post_norm;
        WeightView router;
        std::array<ExpertLocations, 64> experts;
    };
    struct ExpertWeights {
        std::vector<std::uint16_t> values;
        WeightView gate;
        WeightView up;
        WeightView down;
    };
    struct ExpertTensorRead {
        std::filesystem::path path;
        ByteCount absolute_offset{0U};
        ByteCount byte_count{0U};
        std::size_t element_offset{0U};
    };

    OlmoeRuntimeModel() = default;
    OlmoeRuntimeModel(const OlmoeRuntimeModel&) = delete;
    OlmoeRuntimeModel& operator=(const OlmoeRuntimeModel&) = delete;
    OlmoeRuntimeModel(OlmoeRuntimeModel&&) noexcept = default;
    OlmoeRuntimeModel& operator=(OlmoeRuntimeModel&&) noexcept = default;

    [[nodiscard]] static Result<OlmoeRuntimeModel> load(
        const std::filesystem::path& model_directory,
        safety::ResourceGovernor& governor,
        OlmoeRuntimeLoadOptions options = {});

    [[nodiscard]] Result<OlmoeForwardResult> forward(
        const std::vector<std::uint32_t>& token_ids,
        const OlmoeForwardOptions& options,
        safety::ResourceGovernor& governor,
        const safety::ProcessSafetyGuard* process_guard = nullptr) const;

    [[nodiscard]] Result<ExpertWeights> read_expert(std::uint32_t layer, std::uint32_t expert) const;
    [[nodiscard]] Result<std::array<ExpertTensorRead, 3>> expert_read_plan(
        std::uint32_t layer, std::uint32_t expert) const;
    [[nodiscard]] const OlmoeConfig& config() const noexcept { return config_; }
    [[nodiscard]] const Gpt2Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    [[nodiscard]] ByteCount resident_bytes() const noexcept { return static_cast<ByteCount>(resident_.size()) * 2U; }
    [[nodiscard]] double load_ms() const noexcept { return load_ms_; }
    [[nodiscard]] unsigned cpu_threads() const noexcept { return cpu_threads_; }
    [[nodiscard]] const std::vector<std::uint16_t>& resident_weights() const noexcept { return resident_; }
    [[nodiscard]] const WeightView& embedding_view() const noexcept { return embedding_; }
    [[nodiscard]] const WeightView& lm_head_view() const noexcept { return lm_head_; }
    [[nodiscard]] const WeightView& final_norm_view() const noexcept { return final_norm_; }
    [[nodiscard]] const std::vector<LayerWeights>& layer_views() const noexcept { return layers_; }

private:
    [[nodiscard]] Result<WeightView> require_resident(
        std::string_view name,
        const std::vector<std::uint64_t>& shape) const;
    [[nodiscard]] Result<model::TensorLocation> require_streamed(
        std::string_view name,
        const std::vector<std::uint64_t>& shape) const;

    OlmoeConfig config_;
    Gpt2Tokenizer tokenizer_;
    model::CheckpointCatalog catalog_;
    std::vector<std::uint16_t> resident_;
    std::map<std::string, StoredTensor, std::less<>> resident_tensors_;
    WeightView embedding_;
    WeightView lm_head_;
    WeightView final_norm_;
    std::vector<LayerWeights> layers_;
    safety::Reservation resident_reservation_;
    double load_ms_{0.0};
    unsigned cpu_threads_{1};
};

}  // namespace stargate::inference
