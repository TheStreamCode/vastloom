#include "vastloom/inference/tiny_model.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

constexpr std::size_t kFixedHeaderBytes = 96U;
constexpr ByteCount kMaximumTinyModelBytes = 8U * GiB;
constexpr std::uint32_t kMaximumVocabSize = 65'536U;
constexpr std::uint32_t kMaximumContextLength = 32'768U;
constexpr std::uint32_t kMaximumModelWidth = 4'096U;
constexpr std::uint32_t kMaximumLayerCount = 128U;
constexpr std::uint32_t kMaximumHeadCount = 128U;
constexpr std::uint32_t kMaximumFeedForwardWidth = 32'768U;
constexpr std::size_t kMaximumMetadataBytes = 1U * MiB;
constexpr std::size_t kReadChunkBytes = 64U * MiB;
constexpr float kSqrtTwo = 1.4142135623730950488F;

std::uint32_t read_u32_le(const std::byte* data) noexcept {
    return static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[0])) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[1])) << 8U) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[2])) << 16U) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[3])) << 24U);
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8U; ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(data[index])) << (index * 8U);
    }
    return value;
}

float read_f32_le(const std::byte* data) noexcept {
    return std::bit_cast<float>(read_u32_le(data));
}

[[maybe_unused]] std::uint32_t byte_swap_u32(std::uint32_t value) noexcept {
    return ((value & 0x000000ffU) << 24U) |
           ((value & 0x0000ff00U) << 8U) |
           ((value & 0x00ff0000U) >> 8U) |
           ((value & 0xff000000U) >> 24U);
}

Result<std::size_t> checked_size_add(std::size_t left, std::size_t right, std::string context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return make_error(ErrorCode::invalid_format, "Size calculation overflow", std::move(context));
    }
    return left + right;
}

Result<std::size_t> checked_size_multiply(std::size_t left, std::size_t right, std::string context) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        return make_error(ErrorCode::invalid_format, "Size calculation overflow", std::move(context));
    }
    return left * right;
}

Result<std::size_t> expected_float_count(const TinyModelConfig& config) {
    const std::size_t d = config.d_model;
    const std::size_t ff = config.d_ff;
    const std::size_t layers = config.n_layers;
    const std::size_t vocab = config.vocab_size;
    const std::size_t context = config.context_length;

    auto token_embedding = checked_size_multiply(vocab, d, "token embedding");
    auto position_embedding = checked_size_multiply(context, d, "position embedding");
    auto qkv = checked_size_multiply(3U * d, d, "qkv weight");
    auto projection = checked_size_multiply(d, d, "attention projection");
    auto fc1 = checked_size_multiply(ff, d, "feed-forward input weight");
    auto fc2 = checked_size_multiply(d, ff, "feed-forward output weight");
    if (!token_embedding) return token_embedding.error();
    if (!position_embedding) return position_embedding.error();
    if (!qkv) return qkv.error();
    if (!projection) return projection.error();
    if (!fc1) return fc1.error();
    if (!fc2) return fc2.error();

    std::size_t per_layer = 0;
    const std::array<std::size_t, 10> components{
        2U * d,
        qkv.value(),
        3U * d,
        projection.value(),
        d,
        2U * d,
        fc1.value(),
        ff,
        fc2.value(),
        d,
    };
    for (const auto component : components) {
        auto sum = checked_size_add(per_layer, component, "layer tensor layout");
        if (!sum) return sum.error();
        per_layer = sum.value();
    }
    auto all_layers = checked_size_multiply(layers, per_layer, "all layer tensors");
    if (!all_layers) return all_layers.error();

    std::size_t total = token_embedding.value();
    for (const auto component : {position_embedding.value(), all_layers.value(), 2U * d}) {
        auto sum = checked_size_add(total, component, "model tensor layout");
        if (!sum) return sum.error();
        total = sum.value();
    }
    return total;
}

void layer_norm(
    const std::vector<float>& input,
    const float* weight,
    const float* bias,
    float epsilon,
    std::vector<float>& output) {
    const std::size_t width = input.size();
    float mean = 0.0F;
    for (const float value : input) mean += value;
    mean /= static_cast<float>(width);

    float variance = 0.0F;
    for (const float value : input) {
        const float centered = value - mean;
        variance += centered * centered;
    }
    variance /= static_cast<float>(width);
    const float inverse_std = 1.0F / std::sqrt(variance + epsilon);
    for (std::size_t index = 0; index < width; ++index) {
        output[index] = (input[index] - mean) * inverse_std * weight[index] + bias[index];
    }
}

void linear(
    const float* weight,
    const float* bias,
    std::size_t output_width,
    std::size_t input_width,
    const std::vector<float>& input,
    std::vector<float>& output) {
    for (std::size_t row = 0; row < output_width; ++row) {
        float sum = bias == nullptr ? 0.0F : bias[row];
        const float* row_weight = weight + row * input_width;
        for (std::size_t column = 0; column < input_width; ++column) {
            sum += row_weight[column] * input[column];
        }
        output[row] = sum;
    }
}

float gelu(float value) noexcept {
    return 0.5F * value * (1.0F + std::erf(value / kSqrtTwo));
}

std::string bool_json(bool value) {
    return value ? "true" : "false";
}

std::string_view finish_reason(const GenerationStats& stats) noexcept {
    if (stats.stopped_on_timeout) return "timeout";
    if (stats.stopped_on_eot) return "end_of_turn";
    if (stats.stopped_on_eos) return "end_of_sequence";
    if (stats.stopped_on_invalid_token) return "invalid_token";
    if (stats.stopped_on_context_limit) return "context_length";
    if (stats.stopped_on_max_tokens) return "max_tokens";
    return "unknown";
}

}  // namespace

struct TinyModel::LayerLayout {
    std::size_t ln1_weight{0};
    std::size_t ln1_bias{0};
    std::size_t qkv_weight{0};
    std::size_t qkv_bias{0};
    std::size_t projection_weight{0};
    std::size_t projection_bias{0};
    std::size_t ln2_weight{0};
    std::size_t ln2_bias{0};
    std::size_t fc1_weight{0};
    std::size_t fc1_bias{0};
    std::size_t fc2_weight{0};
    std::size_t fc2_bias{0};
};

struct TinyModel::InferenceState {
    explicit InferenceState(const TinyModelConfig& config)
        : key_cache(static_cast<std::size_t>(config.n_layers) * config.context_length * config.d_model),
          value_cache(static_cast<std::size_t>(config.n_layers) * config.context_length * config.d_model),
          hidden(config.d_model),
          normalized(config.d_model),
          qkv(static_cast<std::size_t>(3U) * config.d_model),
          attention(config.d_model),
          projection(config.d_model),
          feed_forward(config.d_ff),
          scores(config.context_length),
          logits(config.vocab_size) {}

    std::vector<float> key_cache;
    std::vector<float> value_cache;
    std::vector<float> hidden;
    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> attention;
    std::vector<float> projection;
    std::vector<float> feed_forward;
    std::vector<float> scores;
    std::vector<float> logits;
    std::size_t position{0};
};

TinyModel::TinyModel(TinyModel&&) noexcept = default;
TinyModel& TinyModel::operator=(TinyModel&&) noexcept = default;
TinyModel::~TinyModel() = default;

Result<TinyModel> TinyModel::load(const std::filesystem::path& path, safety::ResourceGovernor& governor) {
    std::error_code filesystem_error;
    const auto raw_file_size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error) {
        return make_error(ErrorCode::not_found, "Unable to stat TinyModel file", path.string());
    }
    if (raw_file_size < kFixedHeaderBytes) {
        return make_error(ErrorCode::invalid_format, "TinyModel file is smaller than its fixed header", path.string());
    }
    if (raw_file_size > kMaximumTinyModelBytes) {
        return make_error(ErrorCode::invalid_format, "TinyModel file exceeds the 8 GiB safety limit", path.string());
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return make_error(ErrorCode::io_error, "Unable to open TinyModel file", path.string());
    }

    std::array<std::byte, kFixedHeaderBytes> header{};
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size())) {
        return make_error(ErrorCode::io_error, "Unable to read complete TinyModel header", path.string());
    }
    if (std::memcmp(header.data(), kTinyModelMagic.data(), kTinyModelMagic.size()) != 0) {
        return make_error(ErrorCode::invalid_format, "TinyModel magic does not match SGTMV001", path.string());
    }

    const auto u32 = [&](std::size_t index) {
        return read_u32_le(header.data() + 8U + index * 4U);
    };
    const std::uint32_t version = u32(0);
    const std::uint32_t header_bytes = u32(1);
    const std::uint32_t flags = u32(2);
    const std::uint32_t model_id_bytes = u32(13);
    const std::uint32_t system_prompt_bytes = u32(14);
    const std::uint32_t reserved0 = u32(15);
    const float layer_norm_epsilon = read_f32_le(header.data() + 72U);
    const std::uint32_t reserved1 = read_u32_le(header.data() + 76U);
    const std::uint64_t tensor_float_count = read_u64_le(header.data() + 80U);
    const std::uint64_t tensor_bytes = read_u64_le(header.data() + 88U);

    if (version != kTinyModelFormatVersion) {
        return make_error(ErrorCode::unsupported, "Unsupported TinyModel format version", std::to_string(version));
    }
    if ((flags & ~(kTinyModelFlagTiedEmbeddings | kTinyModelFlagSupportsHistory)) != 0U) {
        return make_error(ErrorCode::unsupported, "TinyModel uses unknown feature flags", path.string());
    }
    if ((flags & kTinyModelFlagTiedEmbeddings) == 0U) {
        return make_error(ErrorCode::unsupported, "TinyModel v1 requires tied token/output embeddings", path.string());
    }
    if (reserved0 != 0U || reserved1 != 0U) {
        return make_error(ErrorCode::invalid_format, "TinyModel reserved header fields must be zero", path.string());
    }
    if (!std::isfinite(layer_norm_epsilon) || layer_norm_epsilon <= 0.0F || layer_norm_epsilon > 1.0F) {
        return make_error(ErrorCode::invalid_format, "TinyModel layer-normalization epsilon is invalid", path.string());
    }

    const std::uint64_t metadata_bytes = static_cast<std::uint64_t>(model_id_bytes) + system_prompt_bytes;
    if (model_id_bytes == 0U) {
        return make_error(ErrorCode::invalid_format, "TinyModel model id must not be empty", path.string());
    }
    if (metadata_bytes > kMaximumMetadataBytes ||
        header_bytes != kFixedHeaderBytes + metadata_bytes) {
        return make_error(ErrorCode::invalid_format, "TinyModel metadata/header size is inconsistent", path.string());
    }
    if (header_bytes > raw_file_size) {
        return make_error(ErrorCode::invalid_format, "TinyModel header extends past end of file", path.string());
    }
    if (tensor_float_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        tensor_bytes != tensor_float_count * sizeof(float)) {
        return make_error(ErrorCode::invalid_format, "TinyModel tensor byte count is inconsistent", path.string());
    }
    if (static_cast<std::uint64_t>(header_bytes) + tensor_bytes != raw_file_size) {
        return make_error(ErrorCode::integrity_error, "TinyModel file size does not match its header", path.string());
    }
    if (tensor_float_count > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::resource_exhausted, "TinyModel tensor count exceeds this process address space", path.string());
    }

    TinyModel model;
    model.source_path_ = path;
    model.config_.vocab_size = u32(3);
    model.config_.context_length = u32(4);
    model.config_.d_model = u32(5);
    model.config_.n_layers = u32(6);
    model.config_.n_heads = u32(7);
    model.config_.d_ff = u32(8);
    model.config_.bos_token = u32(9);
    model.config_.eos_token = u32(10);
    model.config_.eot_token = u32(11);
    model.config_.byte_token_offset = u32(12);
    model.config_.layer_norm_epsilon = layer_norm_epsilon;
    model.config_.tied_embeddings = true;
    model.config_.supports_history = (flags & kTinyModelFlagSupportsHistory) != 0U;
    model.config_.tensor_bytes = tensor_bytes;

    const auto& config = model.config_;
    if (config.vocab_size < 259U || config.vocab_size > kMaximumVocabSize ||
        config.context_length < 8U || config.context_length > kMaximumContextLength ||
        config.d_model < 8U || config.d_model > kMaximumModelWidth ||
        config.n_layers < 1U || config.n_layers > kMaximumLayerCount ||
        config.n_heads < 1U || config.n_heads > kMaximumHeadCount ||
        config.d_ff < config.d_model || config.d_ff > kMaximumFeedForwardWidth) {
        return make_error(ErrorCode::invalid_format, "TinyModel dimensions are outside safe supported bounds", path.string());
    }
    if (config.d_model % config.n_heads != 0U) {
        return make_error(ErrorCode::invalid_format, "TinyModel width must be divisible by its head count", path.string());
    }
    if (config.bos_token >= config.vocab_size || config.eos_token >= config.vocab_size ||
        config.eot_token >= config.vocab_size || config.bos_token == config.eos_token ||
        config.bos_token == config.eot_token || config.eos_token == config.eot_token ||
        config.byte_token_offset > config.vocab_size - 256U) {
        return make_error(ErrorCode::invalid_format, "TinyModel special-token configuration is invalid", path.string());
    }

    auto expected_count = expected_float_count(config);
    if (!expected_count) return expected_count.error();
    if (expected_count.value() != tensor_float_count) {
        return make_error(ErrorCode::integrity_error, "TinyModel tensor count does not match its architecture", path.string());
    }

    std::vector<std::byte> metadata(static_cast<std::size_t>(metadata_bytes));
    if (!metadata.empty()) {
        input.read(reinterpret_cast<char*>(metadata.data()), static_cast<std::streamsize>(metadata.size()));
        if (input.gcount() != static_cast<std::streamsize>(metadata.size())) {
            return make_error(ErrorCode::io_error, "Unable to read complete TinyModel metadata", path.string());
        }
    }
    model.config_.model_id.assign(
        reinterpret_cast<const char*>(metadata.data()),
        static_cast<std::size_t>(model_id_bytes));
    if (system_prompt_bytes != 0U) {
        model.config_.system_prompt.assign(
            reinterpret_cast<const char*>(metadata.data() + model_id_bytes),
            static_cast<std::size_t>(system_prompt_bytes));
    }
    if (model.config_.model_id.empty() || model.config_.model_id.find('\0') != std::string::npos ||
        model.config_.system_prompt.find('\0') != std::string::npos) {
        return make_error(ErrorCode::invalid_format, "TinyModel metadata contains an invalid model id or NUL byte", path.string());
    }

    auto reservation_bytes = checked_add(tensor_bytes, 4U * MiB);
    if (!reservation_bytes) return reservation_bytes.error();
    auto reservation = governor.try_reserve(safety::ResourceKind::ram, reservation_bytes.value(), "tiny-model-weights");
    if (!reservation) return reservation.error();

    try {
        model.weights_.resize(static_cast<std::size_t>(tensor_float_count));
        std::size_t bytes_read = 0;
        auto* destination = reinterpret_cast<char*>(model.weights_.data());
        const std::size_t target_bytes = static_cast<std::size_t>(tensor_bytes);
        while (bytes_read < target_bytes) {
            const std::size_t chunk = std::min(kReadChunkBytes, target_bytes - bytes_read);
            input.read(destination + bytes_read, static_cast<std::streamsize>(chunk));
            const auto current = input.gcount();
            if (current != static_cast<std::streamsize>(chunk)) {
                return make_error(ErrorCode::io_error, "TinyModel tensor payload is truncated", path.string());
            }
            bytes_read += chunk;
        }
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate RAM for TinyModel weights", path.string());
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        return make_error(ErrorCode::integrity_error, "TinyModel contains unexpected trailing data", path.string());
    }

    if constexpr (std::endian::native == std::endian::big) {
        for (float& value : model.weights_) {
            auto bits = byte_swap_u32(std::bit_cast<std::uint32_t>(value));
            value = std::bit_cast<float>(bits);
        }
    }
    if (!std::all_of(model.weights_.begin(), model.weights_.end(), [](float value) { return std::isfinite(value); })) {
        return make_error(ErrorCode::integrity_error, "TinyModel tensor payload contains NaN or infinity", path.string());
    }

    std::size_t offset = 0;
    const auto take = [&](std::size_t count, std::size_t* destination_offset) -> Result<void> {
        if (count > model.weights_.size() - offset) {
            return make_error(ErrorCode::integrity_error, "TinyModel tensor layout extends past payload", path.string());
        }
        *destination_offset = offset;
        offset += count;
        return {};
    };
    const std::size_t d = config.d_model;
    const std::size_t ff = config.d_ff;
    auto status = take(static_cast<std::size_t>(config.vocab_size) * d, &model.token_embedding_offset_);
    if (!status) return status.error();
    status = take(static_cast<std::size_t>(config.context_length) * d, &model.position_embedding_offset_);
    if (!status) return status.error();

    model.layers_.resize(config.n_layers);
    for (auto& layer : model.layers_) {
        for (const auto& [count, destination] : std::array<std::pair<std::size_t, std::size_t*>, 12>{
                 std::pair{d, &layer.ln1_weight},
                 std::pair{d, &layer.ln1_bias},
                 std::pair{3U * d * d, &layer.qkv_weight},
                 std::pair{3U * d, &layer.qkv_bias},
                 std::pair{d * d, &layer.projection_weight},
                 std::pair{d, &layer.projection_bias},
                 std::pair{d, &layer.ln2_weight},
                 std::pair{d, &layer.ln2_bias},
                 std::pair{ff * d, &layer.fc1_weight},
                 std::pair{ff, &layer.fc1_bias},
                 std::pair{d * ff, &layer.fc2_weight},
                 std::pair{d, &layer.fc2_bias},
             }) {
            status = take(count, destination);
            if (!status) return status.error();
        }
    }
    status = take(d, &model.final_norm_weight_offset_);
    if (!status) return status.error();
    status = take(d, &model.final_norm_bias_offset_);
    if (!status) return status.error();
    if (offset != model.weights_.size()) {
        return make_error(ErrorCode::integrity_error, "TinyModel tensor layout did not consume the full payload", path.string());
    }

    model.model_reservation_ = reservation.take_value();
    return model;
}

ByteCount TinyModel::resident_bytes() const noexcept {
    return static_cast<ByteCount>(weights_.size()) * sizeof(float);
}

ByteCount TinyModel::estimated_generation_scratch_bytes() const noexcept {
    const ByteCount cache_floats = 2ULL * config_.n_layers * config_.context_length * config_.d_model;
    const ByteCount workspace_floats = 12ULL * config_.d_model + config_.d_ff +
                                       config_.context_length + config_.vocab_size;
    const ByteCount bytes = (cache_floats + workspace_floats) * sizeof(float);
    return bytes + 1U * MiB;
}

Result<std::vector<std::uint32_t>> TinyModel::encode_prompt(
    std::string_view user_prompt,
    std::string_view system_prompt_override) const {
    if (user_prompt.empty()) {
        return make_error(ErrorCode::invalid_argument, "Chat prompt must not be empty");
    }
    const std::string_view system_prompt = system_prompt_override.empty()
                                               ? std::string_view(config_.system_prompt)
                                               : system_prompt_override;
    constexpr std::size_t kPromptFramingBytes = 9U;  // BOS + "S:" + "\nU:" + "\nA:"
    if (system_prompt.size() >= config_.context_length ||
        user_prompt.size() >= config_.context_length ||
        system_prompt.size() + user_prompt.size() >
            static_cast<std::size_t>(config_.context_length) -
                std::min<std::size_t>(kPromptFramingBytes, config_.context_length)) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Prompt leaves no context space for generation",
            std::to_string(kPromptFramingBytes + system_prompt.size() + user_prompt.size()) + "/" +
                std::to_string(config_.context_length) + " tokens");
    }
    std::string formatted;
    formatted.reserve(8U + system_prompt.size() + user_prompt.size());
    formatted.append("S:");
    formatted.append(system_prompt);
    formatted.append("\nU:");
    formatted.append(user_prompt);
    formatted.append("\nA:");

    std::vector<std::uint32_t> tokens;
    tokens.reserve(formatted.size() + 1U);
    tokens.push_back(config_.bos_token);
    for (const char raw_byte : formatted) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        tokens.push_back(config_.byte_token_offset + static_cast<std::uint32_t>(byte));
    }
    if (tokens.size() >= config_.context_length) {
        return make_error(
            ErrorCode::resource_exhausted,
            "Prompt leaves no context space for generation",
            std::to_string(tokens.size()) + "/" + std::to_string(config_.context_length) + " tokens");
    }
    return tokens;
}

Result<void> TinyModel::forward_token(std::uint32_t token, InferenceState& state) const {
    if (token >= config_.vocab_size) {
        return make_error(ErrorCode::invalid_argument, "Token id is outside the TinyModel vocabulary", std::to_string(token));
    }
    if (state.position >= config_.context_length) {
        return make_error(ErrorCode::resource_exhausted, "TinyModel context window is exhausted");
    }

    const std::size_t d = config_.d_model;
    const std::size_t head_width = d / config_.n_heads;
    const float* token_embedding = weights_at(token_embedding_offset_ + static_cast<std::size_t>(token) * d);
    const float* position_embedding = weights_at(position_embedding_offset_ + state.position * d);
    for (std::size_t index = 0; index < d; ++index) {
        state.hidden[index] = token_embedding[index] + position_embedding[index];
    }

    for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        const auto& layer = layers_[layer_index];
        layer_norm(
            state.hidden,
            weights_at(layer.ln1_weight),
            weights_at(layer.ln1_bias),
            config_.layer_norm_epsilon,
            state.normalized);
        linear(
            weights_at(layer.qkv_weight),
            weights_at(layer.qkv_bias),
            3U * d,
            d,
            state.normalized,
            state.qkv);

        const float* query = state.qkv.data();
        const float* key = state.qkv.data() + d;
        const float* value = state.qkv.data() + 2U * d;
        const std::size_t cache_base =
            (layer_index * static_cast<std::size_t>(config_.context_length) + state.position) * d;
        std::copy_n(key, d, state.key_cache.data() + cache_base);
        std::copy_n(value, d, state.value_cache.data() + cache_base);
        std::fill(state.attention.begin(), state.attention.end(), 0.0F);

        const float scale = 1.0F / std::sqrt(static_cast<float>(head_width));
        for (std::size_t head = 0; head < config_.n_heads; ++head) {
            const std::size_t head_offset = head * head_width;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t past = 0; past <= state.position; ++past) {
                const std::size_t past_base =
                    (layer_index * static_cast<std::size_t>(config_.context_length) + past) * d + head_offset;
                float score = 0.0F;
                for (std::size_t column = 0; column < head_width; ++column) {
                    score += query[head_offset + column] * state.key_cache[past_base + column];
                }
                score *= scale;
                state.scores[past] = score;
                maximum = std::max(maximum, score);
            }

            float denominator = 0.0F;
            for (std::size_t past = 0; past <= state.position; ++past) {
                const float probability = std::exp(state.scores[past] - maximum);
                state.scores[past] = probability;
                denominator += probability;
            }
            if (!(denominator > 0.0F) || !std::isfinite(denominator)) {
                return make_error(ErrorCode::internal_error, "TinyModel attention softmax became non-finite");
            }
            const float inverse_denominator = 1.0F / denominator;
            for (std::size_t column = 0; column < head_width; ++column) {
                float sum = 0.0F;
                for (std::size_t past = 0; past <= state.position; ++past) {
                    const std::size_t past_base =
                        (layer_index * static_cast<std::size_t>(config_.context_length) + past) * d + head_offset;
                    sum += state.scores[past] * inverse_denominator * state.value_cache[past_base + column];
                }
                state.attention[head_offset + column] = sum;
            }
        }

        linear(
            weights_at(layer.projection_weight),
            weights_at(layer.projection_bias),
            d,
            d,
            state.attention,
            state.projection);
        for (std::size_t index = 0; index < d; ++index) state.hidden[index] += state.projection[index];

        layer_norm(
            state.hidden,
            weights_at(layer.ln2_weight),
            weights_at(layer.ln2_bias),
            config_.layer_norm_epsilon,
            state.normalized);
        linear(
            weights_at(layer.fc1_weight),
            weights_at(layer.fc1_bias),
            config_.d_ff,
            d,
            state.normalized,
            state.feed_forward);
        for (float& value_at_index : state.feed_forward) value_at_index = gelu(value_at_index);
        linear(
            weights_at(layer.fc2_weight),
            weights_at(layer.fc2_bias),
            d,
            config_.d_ff,
            state.feed_forward,
            state.projection);
        for (std::size_t index = 0; index < d; ++index) state.hidden[index] += state.projection[index];
    }

    layer_norm(
        state.hidden,
        weights_at(final_norm_weight_offset_),
        weights_at(final_norm_bias_offset_),
        config_.layer_norm_epsilon,
        state.normalized);
    linear(
        weights_at(token_embedding_offset_),
        nullptr,
        config_.vocab_size,
        d,
        state.normalized,
        state.logits);
    if (!std::all_of(state.logits.begin(), state.logits.end(), [](float value) { return std::isfinite(value); })) {
        return make_error(ErrorCode::internal_error, "TinyModel logits became non-finite");
    }
    ++state.position;
    return {};
}

std::uint32_t TinyModel::sample_token(
    const std::vector<float>& logits,
    const GenerationOptions& options,
    std::mt19937_64& random) const {
    if (options.temperature <= 0.0F || options.top_k == 1U) {
        return static_cast<std::uint32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    const std::size_t count = std::min(options.top_k, logits.size());
    std::vector<std::size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(count),
        indices.end(),
        [&](std::size_t left, std::size_t right) { return logits[left] > logits[right]; });
    indices.resize(count);

    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto index : indices) {
        const double scaled = static_cast<double>(logits[index]) / static_cast<double>(options.temperature);
        maximum = std::max(maximum, scaled);
    }
    std::vector<double> weights(count);
    double total = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double scaled =
            static_cast<double>(logits[indices[index]]) / static_cast<double>(options.temperature);
        weights[index] = std::exp(scaled - maximum);
        total += weights[index];
    }

    // Convert the standardized mt19937_64 output directly into a 53-bit unit
    // interval value instead of relying on implementation-specific distribution
    // algorithms. This keeps seeded sampling stable across standard libraries.
    constexpr double kInverseTwoTo53 = 1.0 / 9007199254740992.0;
    const double threshold = static_cast<double>(random() >> 11U) * kInverseTwoTo53 * total;
    double cumulative = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        cumulative += weights[index];
        if (threshold < cumulative) return static_cast<std::uint32_t>(indices[index]);
    }
    return static_cast<std::uint32_t>(indices.back());
}

Result<GenerationResult> TinyModel::generate(
    std::string_view user_prompt,
    const GenerationOptions& options,
    safety::ResourceGovernor& governor) const {
    if (options.max_new_tokens == 0U || options.max_new_tokens > 1'000'000U) {
        return make_error(ErrorCode::invalid_argument, "max_new_tokens must be in [1, 1000000]");
    }
    if (!std::isfinite(options.temperature) || options.temperature < 0.0F || options.temperature > 10.0F) {
        return make_error(ErrorCode::invalid_argument, "temperature must be finite and in [0, 10]");
    }
    if (options.top_k == 0U || options.top_k > config_.vocab_size) {
        return make_error(ErrorCode::invalid_argument, "top_k must be in [1, vocab_size]");
    }
    if (options.timeout.count() < 1 || options.timeout > std::chrono::hours(1)) {
        return make_error(ErrorCode::invalid_argument, "generation timeout must be between 1 ms and 1 hour");
    }

    auto prompt = encode_prompt(user_prompt, options.system_prompt_override);
    if (!prompt) return prompt.error();
    const std::size_t available = config_.context_length - prompt.value().size();
    const std::size_t effective_max_tokens = std::min(options.max_new_tokens, available);
    if (effective_max_tokens == 0U) {
        return make_error(ErrorCode::resource_exhausted, "No TinyModel context remains for generation");
    }

    auto scratch = governor.try_reserve(
        safety::ResourceKind::ram,
        estimated_generation_scratch_bytes(),
        "tiny-model-generation");
    if (!scratch) return scratch.error();

    GenerationResult result;
    result.stats.prompt_tokens = prompt.value().size();
    result.stats.requested_max_tokens = options.max_new_tokens;
    result.stats.effective_max_tokens = effective_max_tokens;

    try {
        InferenceState state(config_);
        const auto deadline = std::chrono::steady_clock::now() + options.timeout;
        const auto prompt_started = std::chrono::steady_clock::now();
        for (const auto token : prompt.value()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                result.stats.stopped_on_timeout = true;
                break;
            }
            auto next = forward_token(token, state);
            if (!next) return next.error();
        }
        const auto prompt_finished = std::chrono::steady_clock::now();
        result.stats.prompt_eval_ms =
            std::chrono::duration<double, std::milli>(prompt_finished - prompt_started).count();
        if (result.stats.stopped_on_timeout) return result;

        std::mt19937_64 random(options.seed);
        const auto generation_started = std::chrono::steady_clock::now();
        for (std::size_t generated = 0; generated < effective_max_tokens; ++generated) {
            if (std::chrono::steady_clock::now() >= deadline) {
                result.stats.stopped_on_timeout = true;
                break;
            }
            const std::uint32_t token = sample_token(state.logits, options, random);
            if (token == config_.eot_token) {
                result.stats.stopped_on_eot = true;
                break;
            }
            if (token == config_.eos_token) {
                result.stats.stopped_on_eos = true;
                break;
            }
            if (token < config_.byte_token_offset || token >= config_.byte_token_offset + 256U) {
                result.stats.stopped_on_invalid_token = true;
                break;
            }

            const char byte = static_cast<char>(token - config_.byte_token_offset);
            result.text.push_back(byte);
            ++result.stats.generated_tokens;
            if (options.on_text) {
                const std::string_view piece(&result.text.back(), 1U);
                options.on_text(piece);
            }

            if (generated + 1U < effective_max_tokens) {
                auto next = forward_token(token, state);
                if (!next) return next.error();
            }
        }
        const auto generation_finished = std::chrono::steady_clock::now();
        result.stats.generation_ms =
            std::chrono::duration<double, std::milli>(generation_finished - generation_started).count();
        if (result.stats.generation_ms > 0.0) {
            result.stats.tokens_per_second =
                static_cast<double>(result.stats.generated_tokens) * 1000.0 / result.stats.generation_ms;
        }
        if (!result.stats.stopped_on_eot && !result.stats.stopped_on_eos &&
            !result.stats.stopped_on_invalid_token && !result.stats.stopped_on_timeout &&
            result.stats.generated_tokens == effective_max_tokens) {
            if (effective_max_tokens == available) result.stats.stopped_on_context_limit = true;
            else result.stats.stopped_on_max_tokens = true;
        }
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate TinyModel generation workspace");
    }
    return result;
}

const float* TinyModel::weights_at(std::size_t offset) const noexcept {
    return weights_.data() + offset;
}

bool TinyModelSelfTestReport::passed() const noexcept {
    return !cases.empty() && std::all_of(cases.begin(), cases.end(), [](const auto& test_case) {
        return test_case.passed;
    });
}

Result<TinyModelSelfTestReport> run_tiny_model_self_test(
    const TinyModel& model,
    safety::ResourceGovernor& governor) {
    const bool legacy_fixture = model.config().model_id == "stargate/tinychat-v1";
    if (model.config().model_id != "vastloom/tinychat-v1" && !legacy_fixture) {
        return make_error(
            ErrorCode::unsupported,
            "No built-in self-test vectors are registered for this TinyModel",
            model.config().model_id);
    }
    const std::array<std::pair<std::string_view, std::string_view>, 5> current_vectors{
        std::pair{"Ciao", "Ciao! Sono Vastloom TinyChat."},
        std::pair{"Quanto fa 2+2?", "2+2 fa 4."},
        std::pair{"Qual è la capitale d'Italia?", "La capitale d'Italia è Roma."},
        std::pair{"Sei online?", "No. Funziono completamente in locale."},
        std::pair{"What is 2+2?", "2+2 is 4."},
    };
    const std::array<std::pair<std::string_view, std::string_view>, 5> legacy_vectors{
        std::pair{"Ciao", "Ciao! Sono Stargate TinyChat."},
        std::pair{"Quanto fa 2+2?", "2+2 fa 4."},
        std::pair{"Qual è la capitale d'Italia?", "La capitale d'Italia è Roma."},
        std::pair{"Sei online?", "No. Funziono completamente in locale."},
        std::pair{"What is 2+2?", "2+2 is 4."},
    };
    const auto& vectors = legacy_fixture ? legacy_vectors : current_vectors;

    TinyModelSelfTestReport report;
    report.model_id = model.config().model_id;
    const auto started = std::chrono::steady_clock::now();
    for (const auto& [prompt, expected] : vectors) {
        GenerationOptions options;
        options.max_new_tokens = 64;
        options.temperature = 0.0F;
        options.top_k = 1;
        options.timeout = std::chrono::seconds(10);
        auto generation = model.generate(prompt, options, governor);
        if (!generation) return generation.error();
        TinyModelSelfTestCase test_case;
        test_case.prompt = prompt;
        test_case.expected = expected;
        test_case.actual = generation.value().text;
        test_case.passed = test_case.actual == test_case.expected;
        report.cases.push_back(std::move(test_case));
    }
    report.elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
    return report;
}

std::string tiny_model_config_to_text(const TinyModelConfig& config) {
    std::ostringstream output;
    output << "Model:           " << config.model_id << '\n'
           << "Architecture:    Vastloom TinyTransformer v1\n"
           << "Vocabulary:      " << config.vocab_size << " byte/special tokens\n"
           << "Context:         " << config.context_length << " tokens\n"
           << "Width:           " << config.d_model << '\n'
           << "Layers:          " << config.n_layers << '\n'
           << "Attention heads: " << config.n_heads << '\n'
           << "FFN width:       " << config.d_ff << '\n'
           << "Tensor bytes:    " << format_bytes(config.tensor_bytes) << '\n'
           << "History:         " << (config.supports_history ? "supported" : "stateless validation mode") << '\n'
           << "System prompt:   " << config.system_prompt;
    return output.str();
}

std::string tiny_model_config_to_json(const TinyModelConfig& config) {
    std::ostringstream output;
    output << '{'
           << "\"model_id\":\"" << util::json_escape(config.model_id) << "\","
           << "\"architecture\":\"vastloom_tiny_transformer\","
           << "\"vocab_size\":" << config.vocab_size << ','
           << "\"context_length\":" << config.context_length << ','
           << "\"d_model\":" << config.d_model << ','
           << "\"n_layers\":" << config.n_layers << ','
           << "\"n_heads\":" << config.n_heads << ','
           << "\"d_ff\":" << config.d_ff << ','
           << "\"tensor_bytes\":" << config.tensor_bytes << ','
           << "\"tied_embeddings\":" << bool_json(config.tied_embeddings) << ','
           << "\"supports_history\":" << bool_json(config.supports_history) << ','
           << "\"system_prompt\":\"" << util::json_escape(config.system_prompt) << "\"}";
    return output.str();
}

std::string generation_result_to_json(const GenerationResult& result) {
    const auto& stats = result.stats;
    std::ostringstream output;
    output << std::fixed << std::setprecision(3)
           << '{'
           << "\"ok\":true,"
           << "\"text\":\"" << util::json_escape(result.text) << "\","
           << "\"finish_reason\":\"" << finish_reason(stats) << "\","
           << "\"stats\":{"
           << "\"prompt_tokens\":" << stats.prompt_tokens << ','
           << "\"generated_tokens\":" << stats.generated_tokens << ','
           << "\"requested_max_tokens\":" << stats.requested_max_tokens << ','
           << "\"effective_max_tokens\":" << stats.effective_max_tokens << ','
           << "\"prompt_eval_ms\":" << stats.prompt_eval_ms << ','
           << "\"generation_ms\":" << stats.generation_ms << ','
           << "\"tokens_per_second\":" << stats.tokens_per_second << ','
           << "\"stopped_on_eot\":" << bool_json(stats.stopped_on_eot) << ','
           << "\"stopped_on_eos\":" << bool_json(stats.stopped_on_eos) << ','
           << "\"stopped_on_context_limit\":" << bool_json(stats.stopped_on_context_limit) << ','
           << "\"stopped_on_max_tokens\":" << bool_json(stats.stopped_on_max_tokens) << ','
           << "\"stopped_on_invalid_token\":" << bool_json(stats.stopped_on_invalid_token) << ','
           << "\"stopped_on_timeout\":" << bool_json(stats.stopped_on_timeout)
           << "}}";
    return output.str();
}

std::string self_test_report_to_text(const TinyModelSelfTestReport& report) {
    std::ostringstream output;
    output << "TinyModel self-test: " << report.model_id << '\n';
    for (const auto& test_case : report.cases) {
        output << (test_case.passed ? "PASS" : "FAIL") << "  " << test_case.prompt << '\n';
        if (!test_case.passed) {
            output << "      expected: " << test_case.expected << '\n'
                   << "      actual:   " << test_case.actual << '\n';
        }
    }
    output << "Result: " << (report.passed() ? "PASS" : "FAIL")
           << " (" << std::fixed << std::setprecision(2) << report.elapsed_ms << " ms)";
    return output.str();
}

std::string self_test_report_to_json(const TinyModelSelfTestReport& report) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3)
           << '{'
           << "\"ok\":" << bool_json(report.passed()) << ','
           << "\"model_id\":\"" << util::json_escape(report.model_id) << "\","
           << "\"elapsed_ms\":" << report.elapsed_ms << ','
           << "\"cases\":[";
    for (std::size_t index = 0; index < report.cases.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& test_case = report.cases[index];
        output << '{'
               << "\"prompt\":\"" << util::json_escape(test_case.prompt) << "\","
               << "\"expected\":\"" << util::json_escape(test_case.expected) << "\","
               << "\"actual\":\"" << util::json_escape(test_case.actual) << "\","
               << "\"passed\":" << bool_json(test_case.passed)
               << '}';
    }
    output << "]}";
    return output.str();
}

}  // namespace vastloom::inference
