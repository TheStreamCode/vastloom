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

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/inference/gpt2_tokenizer.hpp"
#include "vastloom/inference/olmoe_model.hpp"
#include "vastloom/io/bounded_io_pool.hpp"
#include "vastloom/io/device_io_counters.hpp"
#include "vastloom/io/process_io_counters.hpp"
#include "vastloom/model/safetensors.hpp"
#include "vastloom/safety/resource_governor.hpp"
#include "vastloom/safety/runtime_policy.hpp"

namespace vastloom::inference {

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

enum class ExpertReadMode : std::uint8_t {
    whole,
    staged,
};

[[nodiscard]] Result<ExpertCachePolicy> parse_expert_cache_policy(std::string_view value);
[[nodiscard]] std::string_view to_string(ExpertCachePolicy policy) noexcept;
[[nodiscard]] Result<ExpertIoMode> parse_expert_io_mode(std::string_view value);
[[nodiscard]] std::string_view to_string(ExpertIoMode mode) noexcept;
[[nodiscard]] Result<ExpertReadMode> parse_expert_read_mode(std::string_view value);
[[nodiscard]] std::string_view to_string(ExpertReadMode mode) noexcept;

struct OlmoeRuntimeLoadOptions {
    bool verify_hashes{true};
    unsigned cpu_threads{1};
};

struct OlmoeForwardOptions {
    ExpertCachePolicy cache_policy{ExpertCachePolicy::lru};
    ByteCount expert_cache_bytes{256U * MiB};
    ExpertIoMode io_mode{ExpertIoMode::synchronous};
    io::IoCacheMode io_cache_mode{io::IoCacheMode::buffered};
    ExpertReadMode read_mode{ExpertReadMode::whole};
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
    double submitted_ms{0.0};
    double queue_ms{0.0};
    double ready_ms{0.0};
    double compute_started_ms{0.0};
    double compute_completed_ms{0.0};
    std::uint32_t route_order{0};
    std::uint32_t reduction_order{0};
};

struct OlmoeForwardStats {
    std::size_t input_tokens{0};
    std::size_t cached_prefix_tokens{0};
    double load_ms{0.0};
    double forward_ms{0.0};
    ByteCount expert_bytes_read{0};
    double expert_io_ms{0.0};
    std::size_t expert_cache_hits{0};
    std::size_t expert_cache_misses{0};
    std::size_t expert_prefetches{0};
    double expert_prefetch_wait_ms{0.0};
    double expert_prefetch_overlap_ms{0.0};
    std::size_t staged_expert_reads{0};
    ByteCount staged_gate_up_bytes_read{0};
    ByteCount staged_down_bytes_read{0};
    double staged_down_compute_overlap_ms{0.0};
    bool process_io_available{false};
    io::ProcessIoDelta process_io;
    std::string process_io_error;
    bool device_io_available{false};
    io::DeviceIoDelta device_io;
    std::string device_io_error;
};

struct OlmoeForwardResult {
    std::vector<float> logits;
    std::vector<ExpertTraceEvent> trace;
    OlmoeForwardStats stats;
};

// A resumable KV workspace for the MoE stream, the M13 session pattern applied to the
// OLMoE path: a caller that resends a shared prefix (a chat template with history, a
// coding agent replaying a transcript) only has the diverging tail evaluated, because the
// Key/Value states of the cached prefix are kept in Governor-reserved buffers owned by
// the session. Divergence, overflow or a cold start degrade to the stateless path and
// never to wrong output.
class OlmoeSession {
public:
    OlmoeSession() = default;
    ~OlmoeSession();
    OlmoeSession(OlmoeSession&&) noexcept;
    OlmoeSession& operator=(OlmoeSession&&) noexcept;
    OlmoeSession(const OlmoeSession&) = delete;
    OlmoeSession& operator=(const OlmoeSession&) = delete;

    [[nodiscard]] bool ready() const noexcept { return capacity_ != 0U; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t cached_token_count() const noexcept { return cached_ids_.size(); }
    void reset();

private:
    friend class OlmoeRuntimeModel;
    safety::Reservation reservation_;
    std::size_t capacity_{0};
    std::vector<float> keys_;
    std::vector<float> values_;
    std::vector<std::uint32_t> cached_ids_;
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
        const safety::ProcessSafetyGuard* process_guard = nullptr,
        OlmoeSession* session = nullptr) const;

    // Reserves the persistent session KV workspace once; the session owns both the
    // reservation and the buffers for its whole lifetime.
    [[nodiscard]] Result<OlmoeSession> create_session(safety::ResourceGovernor& governor,
                                                      std::size_t capacity) const;

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

}  // namespace vastloom::inference
