#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "stargate/core/bytes.hpp"
#include "stargate/core/status.hpp"
#include "stargate/inference/gpt2_tokenizer.hpp"
#include "stargate/safety/resource_governor.hpp"
#include "stargate/safety/runtime_policy.hpp"

namespace stargate::inference {

struct LlamaConfig {
    std::uint32_t vocab_size{0};
    std::uint32_t hidden_size{0};
    std::uint32_t intermediate_size{0};
    std::uint32_t num_hidden_layers{0};
    std::uint32_t num_attention_heads{0};
    std::uint32_t num_key_value_heads{0};
    std::uint32_t max_position_embeddings{0};
    std::uint32_t bos_token_id{0};
    std::uint32_t eos_token_id{0};
    std::uint32_t pad_token_id{0};
    float rms_norm_eps{0.0F};
    float rope_theta{0.0F};
    bool tie_word_embeddings{false};
};

struct LlamaLoadOptions {
    bool verify_hashes{true};
    unsigned cpu_threads{1};
};

struct LlamaGenerationOptions {
    std::size_t max_new_tokens{64};
    float temperature{0.0F};
    std::size_t top_k{40};
    std::uint64_t seed{1};
    std::chrono::milliseconds timeout{std::chrono::minutes(5)};
    std::function<bool()> cancelled;
};

struct LlamaGenerationStats {
    std::size_t prompt_tokens{0};
    std::size_t generated_tokens{0};
    double load_ms{0.0};
    double prompt_eval_ms{0.0};
    double generation_ms{0.0};
    double tokens_per_second{0.0};
    bool stopped_on_eos{false};
    bool stopped_on_context_limit{false};
    bool stopped_on_max_tokens{false};
};

struct LlamaGenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_ids;
    std::vector<std::uint32_t> generated_ids;
    LlamaGenerationStats stats;
};

class LlamaModel {
public:
    LlamaModel() = default;
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;
    LlamaModel(LlamaModel&&) noexcept = default;
    LlamaModel& operator=(LlamaModel&&) noexcept = default;

    [[nodiscard]] static Result<LlamaModel> load(const std::filesystem::path& model_directory,
                                                 safety::ResourceGovernor& governor,
                                                 LlamaLoadOptions options = {});

    [[nodiscard]] Result<std::vector<float>> logits(const std::vector<std::uint32_t>& token_ids,
                                                    safety::ResourceGovernor& governor,
                                                    const safety::ProcessSafetyGuard* process_guard = nullptr) const;
    [[nodiscard]] Result<LlamaGenerationResult> generate(
        const std::vector<ChatMessage>& messages,
        const LlamaGenerationOptions& options,
        safety::ResourceGovernor& governor,
        const safety::ProcessSafetyGuard* process_guard = nullptr,
        const std::function<void(std::string_view)>& stream = {}) const;

    [[nodiscard]] const LlamaConfig& config() const noexcept { return config_; }
    [[nodiscard]] const Gpt2Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    [[nodiscard]] ByteCount resident_bytes() const noexcept { return static_cast<ByteCount>(weights_.size()) * 2U; }
    [[nodiscard]] double load_ms() const noexcept { return load_ms_; }
    [[nodiscard]] unsigned cpu_threads() const noexcept { return cpu_threads_; }

public:
    // Public only so the translation-unit-local reference kernels can use compact,
    // allocation-free views. These are not a stable adapter ABI.
    struct WeightView {
        std::size_t offset{0};
        std::size_t rows{0};
        std::size_t columns{0};
    };
    struct StoredTensor {
        std::size_t offset{0};
        std::vector<std::uint64_t> shape;
    };
    struct LayerWeights {
        WeightView input_norm;
        WeightView q;
        WeightView k;
        WeightView v;
        WeightView o;
        WeightView post_norm;
        WeightView gate;
        WeightView up;
        WeightView down;
    };

    [[nodiscard]] const std::vector<std::uint16_t>& raw_weights() const noexcept { return weights_; }
    [[nodiscard]] const WeightView& embedding_view() const noexcept { return embedding_; }
    [[nodiscard]] const WeightView& final_norm_view() const noexcept { return final_norm_; }
    [[nodiscard]] const std::vector<LayerWeights>& layer_views() const noexcept { return layers_; }

private:
    [[nodiscard]] Result<WeightView> require_weight(std::string_view name,
                                                    const std::vector<std::uint64_t>& shape) const;

    LlamaConfig config_;
    Gpt2Tokenizer tokenizer_;
    std::vector<std::uint16_t> weights_;
    std::map<std::string, StoredTensor, std::less<>> tensors_;
    WeightView embedding_;
    WeightView final_norm_;
    std::vector<LayerWeights> layers_;
    safety::Reservation weight_reservation_;
    double load_ms_{0.0};
    unsigned cpu_threads_{1};
};

[[nodiscard]] Result<LlamaConfig> load_llama_config(const std::filesystem::path& config_path);

}  // namespace stargate::inference
