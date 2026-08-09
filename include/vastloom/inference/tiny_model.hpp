#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace vastloom::inference {

inline constexpr std::string_view kTinyModelMagic = "SGTMV001";
inline constexpr std::uint32_t kTinyModelFormatVersion = 1;
inline constexpr std::uint32_t kTinyModelFlagTiedEmbeddings = 1U << 0U;
inline constexpr std::uint32_t kTinyModelFlagSupportsHistory = 1U << 1U;

struct TinyModelConfig {
    std::string model_id;
    std::string system_prompt;
    std::uint32_t vocab_size{0};
    std::uint32_t context_length{0};
    std::uint32_t d_model{0};
    std::uint32_t n_layers{0};
    std::uint32_t n_heads{0};
    std::uint32_t d_ff{0};
    std::uint32_t bos_token{0};
    std::uint32_t eos_token{0};
    std::uint32_t eot_token{0};
    std::uint32_t byte_token_offset{0};
    float layer_norm_epsilon{1.0e-5F};
    bool tied_embeddings{false};
    bool supports_history{false};
    ByteCount tensor_bytes{0};
};

struct GenerationOptions {
    std::size_t max_new_tokens{96};
    float temperature{0.0F};
    std::size_t top_k{40};
    std::uint64_t seed{0x5354415247415445ULL};
    std::chrono::milliseconds timeout{30'000};
    std::string system_prompt_override;
    std::function<void(std::string_view)> on_text;
};

struct GenerationStats {
    std::size_t prompt_tokens{0};
    std::size_t generated_tokens{0};
    std::size_t requested_max_tokens{0};
    std::size_t effective_max_tokens{0};
    bool stopped_on_eot{false};
    bool stopped_on_eos{false};
    bool stopped_on_context_limit{false};
    bool stopped_on_max_tokens{false};
    bool stopped_on_invalid_token{false};
    bool stopped_on_timeout{false};
    double prompt_eval_ms{0.0};
    double generation_ms{0.0};
    double tokens_per_second{0.0};
};

struct GenerationResult {
    std::string text;
    GenerationStats stats;
};

struct TinyModelSelfTestCase {
    std::string prompt;
    std::string expected;
    std::string actual;
    bool passed{false};
};

struct TinyModelSelfTestReport {
    std::string model_id;
    std::vector<TinyModelSelfTestCase> cases;
    double elapsed_ms{0.0};

    [[nodiscard]] bool passed() const noexcept;
};

class TinyModel {
public:
    TinyModel(const TinyModel&) = delete;
    TinyModel& operator=(const TinyModel&) = delete;
    TinyModel(TinyModel&&) noexcept;
    TinyModel& operator=(TinyModel&&) noexcept;
    ~TinyModel();

    // The governor must outlive the returned model because the model keeps a
    // reservation in that governor for its resident weights.
    [[nodiscard]] static Result<TinyModel> load(
        const std::filesystem::path& path,
        safety::ResourceGovernor& governor);

    [[nodiscard]] const TinyModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::filesystem::path& source_path() const noexcept { return source_path_; }
    [[nodiscard]] ByteCount resident_bytes() const noexcept;
    [[nodiscard]] ByteCount estimated_generation_scratch_bytes() const noexcept;

    [[nodiscard]] Result<std::vector<std::uint32_t>> encode_prompt(
        std::string_view user_prompt,
        std::string_view system_prompt_override = {}) const;
    [[nodiscard]] Result<GenerationResult> generate(
        std::string_view user_prompt,
        const GenerationOptions& options,
        safety::ResourceGovernor& governor) const;

private:
    struct LayerLayout;
    struct InferenceState;

    TinyModel() = default;

    [[nodiscard]] Result<void> forward_token(
        std::uint32_t token,
        InferenceState& state) const;
    [[nodiscard]] std::uint32_t sample_token(
        const std::vector<float>& logits,
        const GenerationOptions& options,
        std::mt19937_64& random) const;
    [[nodiscard]] const float* weights_at(std::size_t offset) const noexcept;

    TinyModelConfig config_;
    std::filesystem::path source_path_;
    std::vector<float> weights_;
    std::vector<LayerLayout> layers_;
    std::size_t token_embedding_offset_{0};
    std::size_t position_embedding_offset_{0};
    std::size_t final_norm_weight_offset_{0};
    std::size_t final_norm_bias_offset_{0};
    safety::Reservation model_reservation_;
};

[[nodiscard]] Result<TinyModelSelfTestReport> run_tiny_model_self_test(
    const TinyModel& model,
    safety::ResourceGovernor& governor);
[[nodiscard]] std::string tiny_model_config_to_text(const TinyModelConfig& config);
[[nodiscard]] std::string tiny_model_config_to_json(const TinyModelConfig& config);
[[nodiscard]] std::string generation_result_to_json(const GenerationResult& result);
[[nodiscard]] std::string self_test_report_to_text(const TinyModelSelfTestReport& report);
[[nodiscard]] std::string self_test_report_to_json(const TinyModelSelfTestReport& report);

}  // namespace vastloom::inference
