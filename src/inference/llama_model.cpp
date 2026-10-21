#include "vastloom/inference/llama_model.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <utility>

#include "vastloom/model/imported_checkpoint.hpp"
#include "vastloom/model/safetensors.hpp"
#include "vastloom/util/json.hpp"

namespace vastloom::inference {
namespace {

constexpr std::string_view kModelId = "HuggingFaceTB/SmolLM2-135M-Instruct";
constexpr std::string_view kRevision = "12fd25f77366fa6b3b4b768ec3050bf629380bac";
constexpr ByteCount kLoadChunkBytes = 64U * MiB;

class ParallelExecutor {
public:
    explicit ParallelExecutor(unsigned threads) {
        const unsigned bounded = std::clamp(threads, 1U, 6U);
        workers_.reserve(bounded - 1U);
        for (unsigned index = 1U; index < bounded; ++index) workers_.emplace_back([this] { worker_loop(); });
    }

    ~ParallelExecutor() {
        {
            std::scoped_lock lock(mutex_);
            stopping_ = true;
            ++epoch_;
        }
        start_.notify_all();
        workers_.clear();
    }

    void run(std::size_t count, std::function<void(std::size_t)> task) {
        if (workers_.empty() || count < 256U) {
            for (std::size_t index = 0; index < count; ++index) task(index);
            return;
        }
        {
            std::scoped_lock lock(mutex_);
            task_ = std::move(task);
            count_ = count;
            next_.store(0U, std::memory_order_relaxed);
            completed_ = 0U;
            ++epoch_;
        }
        start_.notify_all();
        execute_current();
        std::unique_lock lock(mutex_);
        done_.wait(lock, [this] { return completed_ == workers_.size(); });
        task_ = {};
    }

private:
    void execute_current() {
        while (true) {
            const std::size_t index = next_.fetch_add(1U, std::memory_order_relaxed);
            if (index >= count_) return;
            task_(index);
        }
    }

    void worker_loop() {
        std::size_t observed_epoch = 0U;
        while (true) {
            {
                std::unique_lock lock(mutex_);
                start_.wait(lock, [this, observed_epoch] { return stopping_ || epoch_ != observed_epoch; });
                if (stopping_) return;
                observed_epoch = epoch_;
            }
            execute_current();
            {
                std::scoped_lock lock(mutex_);
                ++completed_;
                if (completed_ == workers_.size()) done_.notify_one();
            }
        }
    }

    std::vector<std::jthread> workers_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::function<void(std::size_t)> task_;
    std::atomic<std::size_t> next_{0};
    std::size_t count_{0};
    std::size_t completed_{0};
    std::size_t epoch_{0};
    bool stopping_{false};
};

Result<std::uint32_t> required_u32(const util::JsonValue& root, std::string_view key) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "Model config is missing integer field", std::string(key));
    auto parsed = value->as_u64();
    if (!parsed || parsed.value() > std::numeric_limits<std::uint32_t>::max()) {
        return make_error(ErrorCode::invalid_format, "Model config integer is out of range", std::string(key));
    }
    return static_cast<std::uint32_t>(parsed.value());
}

Result<float> required_float(const util::JsonValue& root, std::string_view key) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "Model config is missing numeric field", std::string(key));
    auto parsed = value->as_double();
    if (!parsed || parsed.value() > static_cast<double>(std::numeric_limits<float>::max()) ||
        parsed.value() < -static_cast<double>(std::numeric_limits<float>::max())) {
        return make_error(ErrorCode::invalid_format, "Model config number is out of range", std::string(key));
    }
    return static_cast<float>(parsed.value());
}

Result<bool> required_bool(const util::JsonValue& root, std::string_view key) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "Model config is missing boolean field", std::string(key));
    return value->as_bool();
}

float bf16_to_float(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

Result<ByteCount> kv_cache_bytes(const LlamaConfig& config, std::size_t sequence_length) {
    const ByteCount head_size = config.hidden_size / config.num_attention_heads;
    const ByteCount kv_width = static_cast<ByteCount>(config.num_key_value_heads) * head_size;
    const ByteCount factors[] = {config.num_hidden_layers, static_cast<ByteCount>(sequence_length), kv_width, 2U, sizeof(float)};
    ByteCount result = 1U;
    for (const ByteCount factor : factors) {
        if (factor != 0U && result > std::numeric_limits<ByteCount>::max() / factor) {
            return make_error(ErrorCode::resource_exhausted, "KV cache byte count overflows");
        }
        result *= factor;
    }
    return result;
}

struct InferenceSession {
    ParallelExecutor executor;
    std::size_t maximum_sequence{0};
    std::size_t kv_width{0};
    std::span<float> keys;
    std::span<float> values;

    InferenceSession(unsigned threads, std::size_t sequence, std::size_t width,
                     std::span<float> key_storage, std::span<float> value_storage)
        : executor(threads), maximum_sequence(sequence), kv_width(width),
          keys(key_storage), values(value_storage) {}
};

}  // namespace

Result<LlamaConfig> load_llama_config(const std::filesystem::path& config_path) {
    util::JsonLimits limits;
    limits.max_input_bytes = 1U * MiB;
    limits.max_nodes = 4096;
    auto json = util::load_json_file(config_path, limits);
    if (!json) return json.error();
    auto object = json.value().as_object();
    if (!object) return make_error(ErrorCode::invalid_format, "Llama config root is not an object", config_path.string());
    const auto* architectures_value = json.value().find("architectures");
    const auto* model_type_value = json.value().find("model_type");
    const auto* activation_value = json.value().find("hidden_act");
    const auto* dtype_value = json.value().find("torch_dtype");
    if (architectures_value == nullptr || model_type_value == nullptr || activation_value == nullptr || dtype_value == nullptr) {
        return make_error(ErrorCode::invalid_format, "Llama config identity fields are missing", config_path.string());
    }
    auto architectures = architectures_value->as_array();
    auto model_type = model_type_value->as_string();
    auto activation = activation_value->as_string();
    auto dtype = dtype_value->as_string();
    if (!architectures || architectures.value()->empty() || !(*architectures.value())[0].as_string() ||
        (*architectures.value())[0].as_string().value() != "LlamaForCausalLM" || !model_type || model_type.value() != "llama" ||
        !activation || activation.value() != "silu" || !dtype || dtype.value() != "bfloat16") {
        return make_error(ErrorCode::unsupported, "Checkpoint is not the supported BF16 Llama architecture", config_path.string());
    }

    LlamaConfig config;
    auto vocab = required_u32(json.value(), "vocab_size");
    auto hidden = required_u32(json.value(), "hidden_size");
    auto intermediate = required_u32(json.value(), "intermediate_size");
    auto layers = required_u32(json.value(), "num_hidden_layers");
    auto heads = required_u32(json.value(), "num_attention_heads");
    auto kv_heads = required_u32(json.value(), "num_key_value_heads");
    auto context = required_u32(json.value(), "max_position_embeddings");
    auto bos = required_u32(json.value(), "bos_token_id");
    auto eos = required_u32(json.value(), "eos_token_id");
    auto pad = required_u32(json.value(), "pad_token_id");
    auto epsilon = required_float(json.value(), "rms_norm_eps");
    auto theta = required_float(json.value(), "rope_theta");
    auto tied = required_bool(json.value(), "tie_word_embeddings");
    if (!vocab) return vocab.error();
    if (!hidden) return hidden.error();
    if (!intermediate) return intermediate.error();
    if (!layers) return layers.error();
    if (!heads) return heads.error();
    if (!kv_heads) return kv_heads.error();
    if (!context) return context.error();
    if (!bos) return bos.error();
    if (!eos) return eos.error();
    if (!pad) return pad.error();
    if (!epsilon) return epsilon.error();
    if (!theta) return theta.error();
    if (!tied) return tied.error();
    config = {vocab.value(), hidden.value(), intermediate.value(), layers.value(), heads.value(), kv_heads.value(),
              context.value(), bos.value(), eos.value(), pad.value(), epsilon.value(), theta.value(), tied.value()};
    if (config.vocab_size == 0U || config.vocab_size > 1'000'000U || config.hidden_size == 0U || config.hidden_size > 16'384U ||
        config.intermediate_size == 0U || config.intermediate_size > 131'072U || config.num_hidden_layers == 0U ||
        config.num_hidden_layers > 128U || config.num_attention_heads == 0U || config.num_key_value_heads == 0U ||
        config.num_attention_heads % config.num_key_value_heads != 0U || config.hidden_size % config.num_attention_heads != 0U ||
        (config.hidden_size / config.num_attention_heads) % 2U != 0U || config.max_position_embeddings == 0U ||
        config.max_position_embeddings > 1'000'000U || config.bos_token_id >= config.vocab_size ||
        config.eos_token_id >= config.vocab_size || config.pad_token_id >= config.vocab_size || !std::isfinite(config.rms_norm_eps) ||
        config.rms_norm_eps <= 0.0F || !std::isfinite(config.rope_theta) || config.rope_theta <= 0.0F || !config.tie_word_embeddings) {
        return make_error(ErrorCode::unsupported, "Llama configuration is outside supported safe bounds", config_path.string());
    }
    const auto require_false = [&json](std::string_view key) -> Result<void> {
        const auto* value = json.value().find(key);
        if (value == nullptr) return make_error(ErrorCode::invalid_format, "Llama config is missing required bias/interleave flag", std::string(key));
        auto parsed = value->as_bool();
        if (!parsed || parsed.value()) return make_error(ErrorCode::unsupported, "Llama adapter does not support enabled bias/interleaved RoPE", std::string(key));
        return {};
    };
    auto attention_bias = require_false("attention_bias");
    auto mlp_bias = require_false("mlp_bias");
    auto rope_interleaved = require_false("rope_interleaved");
    if (!attention_bias) return attention_bias.error();
    if (!mlp_bias) return mlp_bias.error();
    if (!rope_interleaved) return rope_interleaved.error();
    const auto* scaling = json.value().find("rope_scaling");
    if (scaling != nullptr && !scaling->is_null()) return make_error(ErrorCode::unsupported, "RoPE scaling is not supported by this reference adapter", config_path.string());
    return config;
}

Result<LlamaModel> LlamaModel::load(const std::filesystem::path& model_directory,
                                    safety::ResourceGovernor& governor,
                                    LlamaLoadOptions options) {
    const auto started = std::chrono::steady_clock::now();
    options.cpu_threads = std::clamp(options.cpu_threads, 1U, 6U);
    auto imported = model::verify_imported_checkpoint(model_directory, options.verify_hashes);
    if (!imported) return imported.error();
    if (imported.value().metadata_only || imported.value().model_id != kModelId || imported.value().revision != kRevision) {
        return make_error(ErrorCode::unsupported, "Llama adapter accepts only the pinned full SmolLM2 checkpoint", model_directory.string());
    }
    auto config = load_llama_config(model_directory / "config.json");
    if (!config) return config.error();
    auto tokenizer = Gpt2Tokenizer::load(model_directory);
    if (!tokenizer) return tokenizer.error();
    if (tokenizer.value().vocab_size() != config.value().vocab_size ||
        tokenizer.value().bos_token_id() != config.value().bos_token_id || tokenizer.value().eos_token_id() != config.value().eos_token_id) {
        return make_error(ErrorCode::integrity_error, "Tokenizer identity does not match Llama config", model_directory.string());
    }
    auto catalog = model::inspect_safetensors_checkpoint(model_directory);
    if (!catalog) return catalog.error();
    if (catalog.value().total_tensor_bytes == 0U || catalog.value().total_tensor_bytes % 2U != 0U ||
        catalog.value().total_tensor_bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
        return make_error(ErrorCode::unsupported, "Checkpoint tensor payload does not fit the BF16 address space", model_directory.string());
    }
    auto reservation = governor.try_reserve(safety::ResourceKind::ram, catalog.value().total_tensor_bytes, "SmolLM2 immutable BF16 weights");
    if (!reservation) return reservation.error();

    LlamaModel result;
    result.config_ = config.value();
    result.tokenizer_ = tokenizer.take_value();
    result.cpu_threads_ = options.cpu_threads;
    result.weight_reservation_ = reservation.take_value();
    try {
        result.weights_.resize(static_cast<std::size_t>(catalog.value().total_tensor_bytes / 2U));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate reserved BF16 weight store", model_directory.string());
    }
    std::size_t cursor_bytes = 0U;
    auto destination = std::span<std::byte>(reinterpret_cast<std::byte*>(result.weights_.data()),
                                            result.weights_.size() * sizeof(std::uint16_t));
    for (const auto& [name, location] : catalog.value().tensors) {
        const auto& shard = catalog.value().shards[location.shard_index];
        const auto& tensor = shard.tensors()[location.tensor_index];
        if (tensor.dtype != model::TensorDType::bf16 || tensor.byte_count % 2U != 0U ||
            tensor.byte_count > destination.size() - cursor_bytes) {
            return make_error(ErrorCode::unsupported, "Llama adapter requires a bounded all-BF16 checkpoint", name);
        }
        const std::size_t tensor_bytes = static_cast<std::size_t>(tensor.byte_count);
        std::size_t copied = 0U;
        while (copied < tensor_bytes) {
            const std::size_t chunk = std::min<std::size_t>(static_cast<std::size_t>(kLoadChunkBytes), tensor_bytes - copied);
            auto read = shard.read_range_into(tensor,
                                              static_cast<ByteCount>(copied),
                                              destination.subspan(cursor_bytes + copied, chunk),
                                              kLoadChunkBytes);
            if (!read) return read.error();
            copied += chunk;
        }
        result.tensors_.emplace(name, StoredTensor{cursor_bytes / 2U, tensor.shape});
        cursor_bytes += tensor_bytes;
    }
    if (cursor_bytes != destination.size()) return make_error(ErrorCode::integrity_error, "Loaded tensor bytes do not fill the reserved store");

    auto embedding = result.require_weight("model.embed_tokens.weight", {result.config_.vocab_size, result.config_.hidden_size});
    auto final_norm = result.require_weight("model.norm.weight", {result.config_.hidden_size});
    if (!embedding) return embedding.error();
    if (!final_norm) return final_norm.error();
    result.embedding_ = embedding.value();
    result.final_norm_ = final_norm.value();
    result.layers_.reserve(result.config_.num_hidden_layers);
    const std::uint64_t head_size = result.config_.hidden_size / result.config_.num_attention_heads;
    const std::uint64_t kv_width = static_cast<std::uint64_t>(result.config_.num_key_value_heads) * head_size;
    for (std::uint32_t layer = 0; layer < result.config_.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + '.';
        auto input_norm = result.require_weight(prefix + "input_layernorm.weight", {result.config_.hidden_size});
        auto q = result.require_weight(prefix + "self_attn.q_proj.weight", {result.config_.hidden_size, result.config_.hidden_size});
        auto k = result.require_weight(prefix + "self_attn.k_proj.weight", {kv_width, result.config_.hidden_size});
        auto v = result.require_weight(prefix + "self_attn.v_proj.weight", {kv_width, result.config_.hidden_size});
        auto o = result.require_weight(prefix + "self_attn.o_proj.weight", {result.config_.hidden_size, result.config_.hidden_size});
        auto post_norm = result.require_weight(prefix + "post_attention_layernorm.weight", {result.config_.hidden_size});
        auto gate = result.require_weight(prefix + "mlp.gate_proj.weight", {result.config_.intermediate_size, result.config_.hidden_size});
        auto up = result.require_weight(prefix + "mlp.up_proj.weight", {result.config_.intermediate_size, result.config_.hidden_size});
        auto down = result.require_weight(prefix + "mlp.down_proj.weight", {result.config_.hidden_size, result.config_.intermediate_size});
        if (!input_norm) return input_norm.error();
        if (!q) return q.error();
        if (!k) return k.error();
        if (!v) return v.error();
        if (!o) return o.error();
        if (!post_norm) return post_norm.error();
        if (!gate) return gate.error();
        if (!up) return up.error();
        if (!down) return down.error();
        result.layers_.push_back({input_norm.value(), q.value(), k.value(), v.value(), o.value(), post_norm.value(),
                                  gate.value(), up.value(), down.value()});
    }
    if (result.tensors_.size() != 2U + static_cast<std::size_t>(result.config_.num_hidden_layers) * 9U) {
        return make_error(ErrorCode::unsupported, "Checkpoint contains tensors outside the pinned SmolLM2 layout", model_directory.string());
    }
    result.load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}

Result<LlamaModel::WeightView> LlamaModel::require_weight(std::string_view name,
                                                          const std::vector<std::uint64_t>& shape) const {
    const auto iterator = tensors_.find(name);
    if (iterator == tensors_.end() || iterator->second.shape != shape) {
        return make_error(ErrorCode::integrity_error, "Missing or incorrectly shaped Llama tensor", std::string(name));
    }
    if (shape.empty() || shape.size() > 2U) return make_error(ErrorCode::unsupported, "Unsupported Llama tensor rank", std::string(name));
    const std::size_t rows = shape.size() == 1U ? 1U : static_cast<std::size_t>(shape[0]);
    const std::size_t columns = shape.size() == 1U ? static_cast<std::size_t>(shape[0]) : static_cast<std::size_t>(shape[1]);
    return WeightView{iterator->second.offset, rows, columns};
}

namespace {

void rms_norm(const LlamaModel& model,
              const LlamaModel::WeightView& weight,
              std::span<const float> input,
              std::span<float> output,
              float epsilon) {
    double squares = 0.0;
    for (const float value : input) squares += static_cast<double>(value) * static_cast<double>(value);
    const float scale = 1.0F / std::sqrt(static_cast<float>(squares / static_cast<double>(input.size())) + epsilon);
    const auto& weights = model.raw_weights();
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * scale * bf16_to_float(weights[weight.offset + index]);
    }
}

void matvec(const LlamaModel& model,
            ParallelExecutor& executor,
            const LlamaModel::WeightView& weight,
            std::span<const float> input,
            std::span<float> output) {
    const auto& weights = model.raw_weights();
    executor.run(weight.rows, [&weights, &weight, input, output](std::size_t row) {
        const std::size_t base = weight.offset + row * weight.columns;
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        std::size_t column = 0U;
        for (; column + 3U < weight.columns; column += 4U) {
            sum0 += static_cast<double>(bf16_to_float(weights[base + column])) * static_cast<double>(input[column]);
            sum1 += static_cast<double>(bf16_to_float(weights[base + column + 1U])) * static_cast<double>(input[column + 1U]);
            sum2 += static_cast<double>(bf16_to_float(weights[base + column + 2U])) * static_cast<double>(input[column + 2U]);
            sum3 += static_cast<double>(bf16_to_float(weights[base + column + 3U])) * static_cast<double>(input[column + 3U]);
        }
        for (; column < weight.columns; ++column) {
            sum0 += static_cast<double>(bf16_to_float(weights[base + column])) * static_cast<double>(input[column]);
        }
        output[row] = static_cast<float>(sum0 + sum1 + sum2 + sum3);
    });
}

// Adds one frozen-target delta on top of a completed base matvec. A null overlay is the
// everyday path: the reference generation is untouched by any of this machinery.
Result<void> inject_capability_delta(const CapabilityOverlay* overlay,
                                     std::uint32_t layer,
                                     CapabilityTarget target,
                                     std::span<const float> input,
                                     std::span<float> scratch,
                                     std::span<float> output) {
    if (overlay == nullptr) return {};
    const CapabilityOverlayBinding* binding = overlay->find(layer, target);
    if (binding == nullptr) return {};
    if (const auto applied = apply_capability_delta(*binding, input, scratch, output); !applied) {
        return applied.error();
    }
    overlay->note_macs(static_cast<std::uint64_t>(binding->rank) *
                       (static_cast<std::uint64_t>(binding->cols) + binding->rows));
    return {};
}

Result<std::vector<float>> evaluate_tokens(const LlamaModel& model,
                                           const std::vector<std::uint32_t>& token_ids,
                                           std::size_t maximum_sequence,
                                           std::size_t start_position,
                                           const safety::ProcessSafetyGuard* process_guard,
                                           const std::function<bool()>* cancelled,
                                           InferenceSession* session,
                                           const CapabilityOverlay* capability_overlay,
                                           std::vector<std::vector<float>>* hidden_capture) {
    const auto& config = model.config();
    const std::size_t hidden_size = config.hidden_size;
    const std::size_t intermediate_size = config.intermediate_size;
    const std::size_t head_count = config.num_attention_heads;
    const std::size_t kv_head_count = config.num_key_value_heads;
    const std::size_t head_size = hidden_size / head_count;
    const std::size_t heads_per_kv = head_count / kv_head_count;
    std::vector<float> hidden(hidden_size);
    std::vector<float> normalized(hidden_size);
    std::vector<float> q(hidden_size);
    std::vector<float> k(session->kv_width);
    std::vector<float> v(session->kv_width);
    std::vector<float> attention(hidden_size);
    std::vector<float> projected(hidden_size);
    std::vector<float> gate(intermediate_size);
    std::vector<float> up(intermediate_size);
    std::vector<float> product(intermediate_size);
    std::vector<float> down(hidden_size);
    std::vector<float> scores(maximum_sequence);
    std::vector<float> logits(config.vocab_size);
    std::vector<float> capability_scratch(static_cast<std::size_t>(kMaxModuleRank));
    std::vector<float> final_hidden_input(hidden_size);

    if (start_position > maximum_sequence || token_ids.size() > maximum_sequence - start_position) {
        return make_error(ErrorCode::invalid_argument, "Incremental Llama evaluation exceeds allocated context");
    }
    for (std::size_t local_position = 0; local_position < token_ids.size(); ++local_position) {
        const std::size_t position = start_position + local_position;
        if (cancelled != nullptr && *cancelled && (*cancelled)()) {
            return make_error(ErrorCode::resource_exhausted, "Llama generation was cancelled safely");
        }
        if (process_guard != nullptr) {
            auto safe = process_guard->sample_and_check();
            if (!safe) return safe.error();
        }
        const auto token = token_ids[local_position];
        if (token >= config.vocab_size) return make_error(ErrorCode::invalid_argument, "Prompt token exceeds Llama vocabulary");
        const std::size_t embedding_offset = model.embedding_view().offset + static_cast<std::size_t>(token) * hidden_size;
        for (std::size_t index = 0; index < hidden_size; ++index) hidden[index] = bf16_to_float(model.raw_weights()[embedding_offset + index]);

        for (std::size_t layer = 0; layer < model.layer_views().size(); ++layer) {
            const auto& weights = model.layer_views()[layer];
            const std::uint32_t target_layer = static_cast<std::uint32_t>(layer);
            rms_norm(model, weights.input_norm, hidden, normalized, config.rms_norm_eps);
            matvec(model, session->executor, weights.q, normalized, q);
            if (const auto injected = inject_capability_delta(capability_overlay, target_layer,
                                                              CapabilityTarget::q, normalized,
                                                              capability_scratch, q);
                !injected) {
                return injected.error();
            }
            matvec(model, session->executor, weights.k, normalized, k);
            if (const auto injected = inject_capability_delta(capability_overlay, target_layer,
                                                              CapabilityTarget::k, normalized,
                                                              capability_scratch, k);
                !injected) {
                return injected.error();
            }
            matvec(model, session->executor, weights.v, normalized, v);
            if (const auto injected = inject_capability_delta(capability_overlay, target_layer,
                                                              CapabilityTarget::v, normalized,
                                                              capability_scratch, v);
                !injected) {
                return injected.error();
            }

            const std::size_t half_head = head_size / 2U;
            for (std::size_t head = 0; head < head_count; ++head) {
                const std::size_t base = head * head_size;
                for (std::size_t index = 0; index < half_head; ++index) {
                    const float frequency = std::pow(config.rope_theta,
                        -2.0F * static_cast<float>(index) / static_cast<float>(head_size));
                    const float angle = static_cast<float>(position) * frequency;
                    const float cosine = std::cos(angle);
                    const float sine = std::sin(angle);
                    const float first = q[base + index];
                    const float second = q[base + half_head + index];
                    q[base + index] = first * cosine - second * sine;
                    q[base + half_head + index] = second * cosine + first * sine;
                }
            }
            for (std::size_t head = 0; head < kv_head_count; ++head) {
                const std::size_t base = head * head_size;
                for (std::size_t index = 0; index < half_head; ++index) {
                    const float frequency = std::pow(config.rope_theta,
                        -2.0F * static_cast<float>(index) / static_cast<float>(head_size));
                    const float angle = static_cast<float>(position) * frequency;
                    const float cosine = std::cos(angle);
                    const float sine = std::sin(angle);
                    const float first = k[base + index];
                    const float second = k[base + half_head + index];
                    k[base + index] = first * cosine - second * sine;
                    k[base + half_head + index] = second * cosine + first * sine;
                }
            }

            const std::size_t cache_base = (layer * maximum_sequence + position) * session->kv_width;
            std::copy(k.begin(), k.end(), session->keys.begin() + static_cast<std::ptrdiff_t>(cache_base));
            std::copy(v.begin(), v.end(), session->values.begin() + static_cast<std::ptrdiff_t>(cache_base));
            const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));
            std::fill(attention.begin(), attention.end(), 0.0F);
            for (std::size_t head = 0; head < head_count; ++head) {
                const std::size_t kv_head = head / heads_per_kv;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t source = 0; source <= position; ++source) {
                    const std::size_t source_base = (layer * maximum_sequence + source) * session->kv_width + kv_head * head_size;
                    double score = 0.0;
                    for (std::size_t index = 0; index < head_size; ++index) {
                        score += static_cast<double>(q[head * head_size + index]) *
                                 static_cast<double>(session->keys[source_base + index]);
                    }
                    scores[source] = static_cast<float>(score) * attention_scale;
                    maximum = std::max(maximum, scores[source]);
                }
                double denominator = 0.0;
                for (std::size_t source = 0; source <= position; ++source) {
                    scores[source] = std::exp(scores[source] - maximum);
                    denominator += static_cast<double>(scores[source]);
                }
                for (std::size_t source = 0; source <= position; ++source) {
                    const float probability = scores[source] / static_cast<float>(denominator);
                    const std::size_t source_base = (layer * maximum_sequence + source) * session->kv_width + kv_head * head_size;
                    for (std::size_t index = 0; index < head_size; ++index) {
                        attention[head * head_size + index] += probability * session->values[source_base + index];
                    }
                }
            }
            matvec(model, session->executor, weights.o, attention, projected);
            if (const auto injected = inject_capability_delta(capability_overlay, target_layer,
                                                              CapabilityTarget::o, attention,
                                                              capability_scratch, projected);
                !injected) {
                return injected.error();
            }
            for (std::size_t index = 0; index < hidden_size; ++index) hidden[index] += projected[index];

            rms_norm(model, weights.post_norm, hidden, normalized, config.rms_norm_eps);
            matvec(model, session->executor, weights.gate, normalized, gate);
            if (const auto injected = inject_capability_delta(capability_overlay, target_layer,
                                                              CapabilityTarget::gate, normalized,
                                                              capability_scratch, gate);
                !injected) {
                return injected.error();
            }
            matvec(model, session->executor, weights.up, normalized, up);
            if (const auto injected = inject_capability_delta(capability_overlay, target_layer,
                                                              CapabilityTarget::up, normalized,
                                                              capability_scratch, up);
                !injected) {
                return injected.error();
            }
            for (std::size_t index = 0; index < intermediate_size; ++index) {
                const float silu = gate[index] / (1.0F + std::exp(-gate[index]));
                product[index] = silu * up[index];
            }
            matvec(model, session->executor, weights.down, product, down);
            if (const auto injected = inject_capability_delta(capability_overlay, target_layer,
                                                              CapabilityTarget::down, product,
                                                              capability_scratch, down);
                !injected) {
                return injected.error();
            }
            for (std::size_t index = 0; index < hidden_size; ++index) hidden[index] += down[index];
        }
        rms_norm(model, model.final_norm_view(), hidden, normalized, config.rms_norm_eps);
        if (capability_overlay != nullptr) {
            const auto* hidden_binding = capability_overlay->find(
                kCapabilityFinalHiddenLayer, CapabilityTarget::final_hidden);
            if (hidden_binding != nullptr) {
                std::copy(normalized.begin(), normalized.end(), final_hidden_input.begin());
                if (const auto applied = apply_capability_delta(
                        *hidden_binding, final_hidden_input, capability_scratch, normalized);
                    !applied) {
                    return applied.error();
                }
                capability_overlay->note_macs(
                    static_cast<std::uint64_t>(hidden_binding->rank) *
                    (static_cast<std::uint64_t>(hidden_binding->cols) + hidden_binding->rows));
            }
        }
        if (hidden_capture != nullptr) {
            hidden_capture->push_back(normalized);
        }
        matvec(model, session->executor, model.embedding_view(), normalized, logits);
        if (const auto injected =
                inject_capability_delta(capability_overlay, kCapabilityLmHeadLayer,
                                        CapabilityTarget::lm_head, normalized, capability_scratch,
                                        logits);
            !injected) {
            return injected.error();
        }
    }
    return logits;
}

std::uint32_t sample_token(const std::vector<float>& logits,
                           const LlamaGenerationOptions& options,
                           std::mt19937_64& generator) {
    if (options.temperature == 0.0F) {
        return static_cast<std::uint32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    const std::size_t count = std::min(options.top_k, logits.size());
    std::vector<std::uint32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(count), indices.end(),
                      [&logits](std::uint32_t left, std::uint32_t right) { return logits[left] > logits[right]; });
    indices.resize(count);
    const float maximum = logits[indices.front()] / options.temperature;
    std::vector<double> weights;
    weights.reserve(indices.size());
    for (const auto index : indices) weights.push_back(std::exp(static_cast<double>(logits[index] / options.temperature - maximum)));
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return indices[distribution(generator)];
}

// Removes disallowed tokens from a copy of the step logits. The original logits
// stay untouched because later metrics and dumps may still need them.
Result<void> apply_token_constraint(std::vector<float>& logits, TokenConstraint& constraint) {
    auto mask = constraint.allowed_mask();
    if (!mask) return mask.error();
    if (mask.value().size() != logits.size()) {
        return make_error(ErrorCode::internal_error,
                          "decoder constraint mask size differs from the model vocabulary");
    }
    bool any_allowed = false;
    for (std::size_t index = 0U; index < logits.size(); ++index) {
        if (mask.value()[index] != 0U) {
            any_allowed = true;
        } else {
            logits[index] = -std::numeric_limits<float>::infinity();
        }
    }
    if (!any_allowed) {
        // A correct mask always allows something: EOS at minimum when the
        // document is complete, otherwise a continuation byte. Reaching this
        // point means the constraint itself is inconsistent, so fail closed.
        return make_error(ErrorCode::internal_error,
                          "decoder constraint allows no token at this position");
    }
    return {};
}

}  // namespace

std::string_view LlamaModel::model_id() const noexcept {
    return kModelId;
}

LlamaSession::LlamaSession() = default;
LlamaSession::~LlamaSession() = default;
LlamaSession::LlamaSession(LlamaSession&&) noexcept = default;
LlamaSession& LlamaSession::operator=(LlamaSession&&) noexcept = default;

bool LlamaSession::ready() const noexcept { return capacity_ != 0U; }
std::size_t LlamaSession::capacity() const noexcept { return capacity_; }
std::size_t LlamaSession::cached_token_count() const noexcept { return cached_ids_.size(); }
void LlamaSession::reset() { cached_ids_.clear(); }

std::size_t common_prefix(std::span<const std::uint32_t> first,
                          std::span<const std::uint32_t> second) {
    const std::size_t limit = std::min(first.size(), second.size());
    std::size_t matched = 0U;
    while (matched < limit && first[matched] == second[matched]) ++matched;
    return matched;
}

Result<LlamaSession> LlamaModel::create_session(safety::ResourceGovernor& governor,
                                                std::size_t capacity) const {
    if (capacity == 0U || capacity > config_.max_position_embeddings) {
        return make_error(ErrorCode::invalid_argument,
                          "Session capacity is outside the model context",
                          std::to_string(capacity));
    }
    auto kv_bytes = kv_cache_bytes(config_, capacity);
    if (!kv_bytes) return kv_bytes.error();
    auto reservation = governor.try_reserve(safety::ResourceKind::ram, kv_bytes.value(),
                                            "Llama persistent session KV cache");
    if (!reservation) return reservation.error();
    try {
        LlamaSession session;
        session.reservation_ = reservation.take_value();
        session.capacity_ = capacity;
        session.kv_width_ = config_.num_key_value_heads *
                            (config_.hidden_size / config_.num_attention_heads);
        session.keys_.assign(config_.num_hidden_layers * capacity * session.kv_width_, 0.0F);
        session.values_.assign(session.keys_.size(), 0.0F);
        return session;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate reserved persistent session buffers");
    }
}

Result<std::vector<float>> LlamaModel::logits(const std::vector<std::uint32_t>& token_ids,
                                              safety::ResourceGovernor& governor,
                                              const safety::ProcessSafetyGuard* process_guard) const {
    if (token_ids.empty() || token_ids.size() > config_.max_position_embeddings) {
        return make_error(ErrorCode::invalid_argument, "Llama logits input is empty or exceeds context");
    }
    auto kv_bytes = kv_cache_bytes(config_, token_ids.size());
    if (!kv_bytes) return kv_bytes.error();
    auto kv_reservation = governor.try_reserve(safety::ResourceKind::ram, kv_bytes.value(), "Llama reference KV cache");
    if (!kv_reservation) return kv_reservation.error();
    auto scratch = governor.try_reserve(safety::ResourceKind::scratch, 8U * MiB, "Llama reference activation scratch");
    if (!scratch) return scratch.error();
    try {
        const std::size_t kv_width = static_cast<std::size_t>(config_.num_key_value_heads) *
                                     (config_.hidden_size / config_.num_attention_heads);
        std::vector<float> keys(config_.num_hidden_layers * token_ids.size() * kv_width);
        std::vector<float> values(keys.size());
        InferenceSession session(cpu_threads_, token_ids.size(), kv_width, keys, values);
        return evaluate_tokens(*this, token_ids, token_ids.size(), 0U, process_guard, nullptr, &session, nullptr, nullptr);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate reserved Llama KV or activation buffers");
    }
}

Result<std::vector<std::vector<float>>> LlamaModel::hidden_states(
    const std::vector<std::uint32_t>& token_ids,
    safety::ResourceGovernor& governor,
    const safety::ProcessSafetyGuard* process_guard) const {
    if (token_ids.empty() || token_ids.size() > config_.max_position_embeddings) {
        return make_error(ErrorCode::invalid_argument,
                          "Llama hidden-states input is empty or exceeds context");
    }
    auto kv_bytes = kv_cache_bytes(config_, token_ids.size());
    if (!kv_bytes) return kv_bytes.error();
    auto kv_reservation = governor.try_reserve(safety::ResourceKind::ram, kv_bytes.value(),
                                               "Llama reference KV cache");
    if (!kv_reservation) return kv_reservation.error();
    auto scratch = governor.try_reserve(safety::ResourceKind::scratch, 8U * MiB,
                                        "Llama reference activation scratch");
    if (!scratch) return scratch.error();
    try {
        const std::size_t kv_width = static_cast<std::size_t>(config_.num_key_value_heads) *
                                     (config_.hidden_size / config_.num_attention_heads);
        std::vector<float> keys(config_.num_hidden_layers * token_ids.size() * kv_width);
        std::vector<float> values(keys.size());
        InferenceSession session(cpu_threads_, token_ids.size(), kv_width, keys, values);
        std::vector<std::vector<float>> capture;
        capture.reserve(token_ids.size());
        auto result = evaluate_tokens(*this, token_ids, token_ids.size(), 0U, process_guard,
                                      nullptr, &session, nullptr, &capture);
        if (!result) return result.error();
        return capture;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate reserved Llama hidden-state buffers");
    }
}

Result<std::vector<float>> LlamaModel::head_dot(const std::vector<float>& normalized) const {
    if (normalized.size() != config_.hidden_size) {
        return make_error(ErrorCode::invalid_argument,
                          "LM-head input does not match the model width");
    }
    std::vector<float> output(config_.vocab_size);
    const auto& weights = raw_weights();
    const auto& head = embedding_view();
    for (std::size_t row = 0; row < config_.vocab_size; ++row) {
        const std::size_t base = head.offset + row * config_.hidden_size;
        double sum = 0.0;
        for (std::size_t column = 0; column < config_.hidden_size; ++column) {
            sum += static_cast<double>(bf16_to_float(weights[base + column])) *
                   static_cast<double>(normalized[column]);
        }
        output[row] = static_cast<float>(sum);
    }
    return output;
}

Result<void> LlamaModel::head_weights_into(std::span<float> destination) const {
    if (destination.size() != static_cast<std::size_t>(config_.vocab_size) * config_.hidden_size) {
        return make_error(ErrorCode::invalid_argument,
                          "Head matrix destination does not match the model dimensions");
    }
    const auto& weights = raw_weights();
    const auto& head = embedding_view();
    for (std::size_t row = 0; row < config_.vocab_size; ++row) {
        const std::size_t source = head.offset + row * config_.hidden_size;
        float* out_row = destination.data() + row * config_.hidden_size;
        for (std::size_t column = 0; column < config_.hidden_size; ++column) {
            out_row[column] = bf16_to_float(weights[source + column]);
        }
    }
    return {};
}

Result<LlamaGenerationResult> LlamaModel::generate(const std::vector<ChatMessage>& messages,
                                                   const LlamaGenerationOptions& options,
                                                   safety::ResourceGovernor& governor,
                                                   const safety::ProcessSafetyGuard* process_guard,
                                                   const std::function<void(std::string_view)>& stream,
                                                   const CapabilityOverlay* capability_overlay,
                                                   LlamaSession* session) const {
    if (options.max_new_tokens == 0U || options.top_k == 0U || !std::isfinite(options.temperature) ||
        options.temperature < 0.0F || options.timeout <= std::chrono::milliseconds::zero()) {
        return make_error(ErrorCode::invalid_argument, "Invalid Llama generation options");
    }
    auto formatted = tokenizer_.apply_smol_chat_template(messages, true);
    if (!formatted) return formatted.error();
    auto prompt = tokenizer_.encode(formatted.value());
    if (!prompt) return prompt.error();
    if (prompt.value().empty() || prompt.value().size() >= config_.max_position_embeddings) {
        return make_error(ErrorCode::invalid_argument, "Formatted prompt is empty or fills the Llama context");
    }
    const std::size_t maximum_new = std::min(options.max_new_tokens,
        static_cast<std::size_t>(config_.max_position_embeddings) - prompt.value().size());

    // Prefix reuse: when a ready session carries the whole framed prefix and the request fits
    // its buffers, only the diverging tail is evaluated. Divergence, a duplicate request with
    // no tail, or an overflow degrade to today's stateless path - never to wrong output.
    std::size_t start_position = 0U;
    std::size_t maximum_sequence = prompt.value().size() + maximum_new;
    bool prefix_reuse = false;
    if (session != nullptr && session->ready()) {
        const std::size_t shared = common_prefix(prompt.value(), session->cached_ids_);
        if (shared == session->cached_ids_.size() && shared < prompt.value().size() &&
            maximum_sequence <= session->capacity_) {
            start_position = shared;
            prefix_reuse = true;
            maximum_sequence = session->capacity_;
        } else {
            session->reset();
        }
    }

    std::optional<safety::Reservation> kv_reservation;
    if (!prefix_reuse) {
        auto kv_bytes = kv_cache_bytes(config_, maximum_sequence);
        if (!kv_bytes) return kv_bytes.error();
        auto reserved = governor.try_reserve(safety::ResourceKind::ram, kv_bytes.value(), "Llama generation KV cache");
        if (!reserved) return reserved.error();
        kv_reservation.emplace(reserved.take_value());
    }
    auto scratch = governor.try_reserve(safety::ResourceKind::scratch, 8U * MiB, "Llama generation activation scratch");
    if (!scratch) return scratch.error();

    const std::vector<std::uint32_t> eval_ids(
        prompt.value().begin() + static_cast<std::ptrdiff_t>(start_position), prompt.value().end());

    LlamaGenerationResult result;
    result.prompt_ids = prompt.value();
    result.stats.prompt_tokens = result.prompt_ids.size();
    result.stats.cached_prefix_tokens = start_position;
    result.stats.load_ms = load_ms_;
    const auto started = std::chrono::steady_clock::now();
    const std::size_t kv_width = static_cast<std::size_t>(config_.num_key_value_heads) *
                                 (config_.hidden_size / config_.num_attention_heads);
    try {
        std::vector<float> fresh_keys;
        std::vector<float> fresh_values;
        std::span<float> keys_storage;
        std::span<float> values_storage;
        if (prefix_reuse && session != nullptr) {
            keys_storage = std::span<float>(session->keys_);
            values_storage = std::span<float>(session->values_);
        } else {
            fresh_keys.assign(config_.num_hidden_layers * maximum_sequence * kv_width, 0.0F);
            fresh_values.assign(fresh_keys.size(), 0.0F);
            keys_storage = std::span<float>(fresh_keys);
            values_storage = std::span<float>(fresh_values);
        }
        InferenceSession active(cpu_threads_, maximum_sequence, kv_width, keys_storage, values_storage);
        const auto* cancelled = options.cancelled ? &options.cancelled : nullptr;
        auto logits_result = evaluate_tokens(*this, eval_ids, maximum_sequence, start_position, process_guard, cancelled, &active, capability_overlay, nullptr);
        if (!logits_result) return logits_result.error();
        std::vector<float> current_logits = logits_result.take_value();
        const auto prompt_finished = std::chrono::steady_clock::now();
        result.stats.prompt_eval_ms = std::chrono::duration<double, std::milli>(prompt_finished - started).count();
        std::mt19937_64 generator(options.seed);
        result.stats.constrained = options.constraint != nullptr;
        for (std::size_t index = 0; index < maximum_new; ++index) {
            if (std::chrono::steady_clock::now() - started > options.timeout) {
                return make_error(ErrorCode::resource_exhausted, "Llama generation timed out safely");
            }
            std::vector<float> constrained_logits;
            const std::vector<float>* sampling_logits = &current_logits;
            if (options.constraint != nullptr) {
                constrained_logits = current_logits;
                auto masked = apply_token_constraint(constrained_logits, *options.constraint);
                if (!masked) return masked.error();
                sampling_logits = &constrained_logits;
            }
            const std::uint32_t next = sample_token(*sampling_logits, options, generator);
            if (next == config_.eos_token_id) {
                if (options.constraint != nullptr && !options.constraint->complete()) {
                    return make_error(ErrorCode::internal_error,
                                      "decoder constraint let EOS through before the document completed");
                }
                result.stats.stopped_on_eos = true;
                break;
            }
            if (options.constraint != nullptr) {
                auto accepted = options.constraint->accept_token(next);
                if (!accepted) return accepted.error();
            }
            result.generated_ids.push_back(next);
            auto piece = tokenizer_.decode({next}, true);
            if (!piece) return piece.error();
            result.text += piece.value();
            if (stream) stream(piece.value());
            if (index + 1U < maximum_new) {
                const std::vector<std::uint32_t> incremental{next};
                const std::size_t token_position = start_position + eval_ids.size() + result.generated_ids.size() - 1U;
                auto evaluated = evaluate_tokens(*this, incremental, maximum_sequence, token_position, process_guard, cancelled, &active, capability_overlay, nullptr);
                if (!evaluated) return evaluated.error();
                current_logits = evaluated.take_value();
            }
        }
        result.stats.generated_tokens = result.generated_ids.size();
        result.stats.stopped_on_context_limit = !result.stats.stopped_on_eos && maximum_new < options.max_new_tokens;
        result.stats.stopped_on_max_tokens = !result.stats.stopped_on_eos && maximum_new == options.max_new_tokens;
        result.stats.constraint_complete = options.constraint != nullptr && options.constraint->complete();
        // Hand the updated Key/Value states and the full evaluated sequence back to the
        // session so the next request can reuse them. Only sequences that fit the session's
        // capacity are cached; anything longer leaves it empty for a clean future start.
        if (session != nullptr) {
            std::vector<std::uint32_t> updated = result.prompt_ids;
            updated.insert(updated.end(), result.generated_ids.begin(), result.generated_ids.end());
            if (updated.size() <= session->capacity_ && prefix_reuse) {
                session->cached_ids_ = std::move(updated);
            } else {
                session->reset();
            }
        }
        result.stats.generation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prompt_finished).count();
        if (result.stats.generation_ms > 0.0) {
            result.stats.tokens_per_second = static_cast<double>(result.stats.generated_tokens) * 1000.0 / result.stats.generation_ms;
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate reserved Llama generation buffers");
    }
}

std::pair<std::uint32_t, std::uint32_t> capability_target_dims(const LlamaConfig& config,
                                                               CapabilityTarget target) {
    const std::uint64_t head_size =
        static_cast<std::uint64_t>(config.hidden_size) / config.num_attention_heads;
    switch (target) {
        case CapabilityTarget::q:
        case CapabilityTarget::o: return {config.hidden_size, config.hidden_size};
        case CapabilityTarget::k:
        case CapabilityTarget::v:
            return {config.num_key_value_heads * static_cast<std::uint32_t>(head_size),
                    config.hidden_size};
        case CapabilityTarget::gate:
        case CapabilityTarget::up: return {config.intermediate_size, config.hidden_size};
        case CapabilityTarget::down: return {config.hidden_size, config.intermediate_size};
        case CapabilityTarget::lm_head: return {config.vocab_size, config.hidden_size};
        case CapabilityTarget::final_hidden: return {config.hidden_size, config.hidden_size};
    }
    return {0U, 0U};
}

Result<CapabilityOverlay> build_capability_overlay(const LlamaConfig& config,
                                                   const CapabilityModule& module) {
    // Bindings borrow the module's storage, so the overlay is only valid while the module lives.
    CapabilityOverlay overlay;
    overlay.set_module_bytes(module.file_bytes);
    overlay.set_selected_name(module.name);
    for (const auto& target : module.targets) {
        if (target.spec.target == CapabilityTarget::lm_head) {
            if (target.spec.layer != kCapabilityLmHeadLayer) {
                return make_error(ErrorCode::invalid_argument,
                                  "Capability module LM-head target must use the reserved layer id",
                                  module.name);
            }
        } else if (target.spec.target == CapabilityTarget::final_hidden) {
            if (target.spec.layer != kCapabilityFinalHiddenLayer) {
                return make_error(ErrorCode::invalid_argument,
                                  "Capability module final-hidden target must use its reserved layer id",
                                  module.name);
            }
        } else if (target.spec.layer >= config.num_hidden_layers) {
            return make_error(ErrorCode::invalid_argument,
                              "Capability module targets a layer outside the model",
                              module.name + " layer " + std::to_string(target.spec.layer));
        }
        const auto [rows, cols] = capability_target_dims(config, target.spec.target);
        if (target.spec.rows != rows || target.spec.cols != cols) {
            return make_error(
                ErrorCode::integrity_error,
                "Capability module target does not match the model's dimensions",
                module.name + " " + std::string(to_string(target.spec.target)) + ": module " +
                    std::to_string(target.spec.rows) + "x" + std::to_string(target.spec.cols) +
                    ", model " + std::to_string(rows) + "x" + std::to_string(cols));
        }
        CapabilityOverlayBinding binding;
        binding.layer = target.spec.layer;
        binding.target = target.spec.target;
        binding.rows = target.spec.rows;
        binding.cols = target.spec.cols;
        binding.rank = module.rank;
        binding.scale = module.scale;
        binding.a = target.a.data();
        binding.b = target.b.data();
        binding.source_name = module.name;
        overlay.add_binding(std::move(binding));
    }
    return overlay;
}

}  // namespace vastloom::inference
