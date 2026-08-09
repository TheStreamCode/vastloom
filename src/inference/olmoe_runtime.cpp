#include "stargate/inference/olmoe_runtime.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <thread>

#include "stargate/model/imported_checkpoint.hpp"
#include "stargate/util/string.hpp"

namespace stargate::inference {
namespace {

constexpr ByteCount kResidentBytes = 953'421'824ULL;
constexpr ByteCount kExpertBytes = 12U * MiB;
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

    void run(std::size_t count, const std::function<void(std::size_t)>& task) {
        if (workers_.empty() || count < 128U) {
            for (std::size_t index = 0; index < count; ++index) task(index);
            return;
        }
        {
            std::scoped_lock lock(mutex_);
            task_ = task;
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
    std::atomic<std::size_t> next_{0U};
    std::size_t count_{0U};
    std::size_t completed_{0U};
    std::size_t epoch_{0U};
    bool stopping_{false};
};

float bf16_to_float(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

void matrix_vector(const std::vector<std::uint16_t>& weights,
                   const OlmoeRuntimeModel::WeightView& view,
                   std::span<const float> input,
                   std::span<float> output,
                   ParallelExecutor& executor) {
    executor.run(view.rows, [&](std::size_t row) {
        const std::size_t base = view.offset + row * view.columns;
        double sum = 0.0;
        for (std::size_t column = 0; column < view.columns; ++column) {
            sum += static_cast<double>(bf16_to_float(weights[base + column])) * static_cast<double>(input[column]);
        }
        output[row] = static_cast<float>(sum);
    });
}

void rms_norm(const std::vector<std::uint16_t>& weights,
              const OlmoeRuntimeModel::WeightView& view,
              std::span<const float> input,
              std::span<float> output,
              float epsilon) {
    double squares = 0.0;
    for (const float value : input) squares += static_cast<double>(value) * static_cast<double>(value);
    const float scale = 1.0F / std::sqrt(static_cast<float>(squares / static_cast<double>(input.size())) + epsilon);
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * scale * bf16_to_float(weights[view.offset + index]);
    }
}

void apply_rope(std::span<float> query,
                std::span<float> key,
                std::size_t heads,
                std::size_t head_size,
                std::size_t position,
                float theta) {
    const std::size_t half = head_size / 2U;
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t base = head * head_size;
        for (std::size_t index = 0; index < half; ++index) {
            const float frequency = std::pow(theta, -2.0F * static_cast<float>(index) / static_cast<float>(head_size));
            const float angle = static_cast<float>(position) * frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float q_first = query[base + index];
            const float q_second = query[base + half + index];
            const float k_first = key[base + index];
            const float k_second = key[base + half + index];
            query[base + index] = q_first * cosine - q_second * sine;
            query[base + half + index] = q_second * cosine + q_first * sine;
            key[base + index] = k_first * cosine - k_second * sine;
            key[base + half + index] = k_second * cosine + k_first * sine;
        }
    }
}

Result<ByteCount> kv_cache_bytes(const OlmoeConfig& config, std::size_t sequence_length) {
    ByteCount result = config.num_hidden_layers;
    constexpr std::array<ByteCount, 4> factors{2U, sizeof(float), 1U, 1U};
    const std::array<ByteCount, 3> dynamic{
        static_cast<ByteCount>(sequence_length), config.hidden_size, factors[0] * factors[1]};
    for (const ByteCount factor : dynamic) {
        if (factor != 0U && result > std::numeric_limits<ByteCount>::max() / factor) {
            return make_error(ErrorCode::resource_exhausted, "OLMoE KV cache size overflows");
        }
        result *= factor;
    }
    return result;
}

struct AcquiredExpert {
    std::shared_ptr<const OlmoeRuntimeModel::ExpertWeights> weights;
    bool hit{false};
    ByteCount bytes_read{0U};
    double io_ms{0.0};
};

class ExpertCache {
public:
    static Result<ExpertCache> create(const OlmoeRuntimeModel& model,
                                      ExpertCachePolicy policy,
                                      ByteCount requested_bytes,
                                      safety::ResourceGovernor& governor) {
        ByteCount capacity = 0U;
        if (policy != ExpertCachePolicy::none) {
            capacity = (requested_bytes / kExpertBytes) * kExpertBytes;
            if (capacity < kExpertBytes || capacity > 2U * GiB) {
                return make_error(ErrorCode::invalid_argument, "OLMoE expert cache must hold 1-170 experts and remain at most 2 GiB");
            }
        }
        ByteCount reservation_bytes = policy == ExpertCachePolicy::none ? kExpertBytes : capacity;
        if (policy == ExpertCachePolicy::static_hot) {
            auto with_cold = checked_add(reservation_bytes, kExpertBytes);
            if (!with_cold) return with_cold.error();
            reservation_bytes = with_cold.value();
        }
        auto reservation = governor.try_reserve(safety::ResourceKind::ram, reservation_bytes, "OLMoE bounded expert cache");
        if (!reservation) return reservation.error();
        return ExpertCache(model, policy, capacity, reservation.take_value());
    }

    ExpertCache(ExpertCache&&) noexcept = default;
    ExpertCache& operator=(ExpertCache&&) noexcept = delete;
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    Result<AcquiredExpert> acquire(std::uint32_t layer, std::uint32_t expert) {
        const std::uint32_t key = layer * model_.config().num_experts + expert;
        ++clock_;
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
            found->second.last_use = clock_;
            ++found->second.frequency;
            return AcquiredExpert{found->second.weights, true, 0U, 0.0};
        }

        bool retain = policy_ == ExpertCachePolicy::lru || policy_ == ExpertCachePolicy::lfu;
        if (policy_ == ExpertCachePolicy::static_hot) {
            const std::size_t slots = static_cast<std::size_t>(capacity_bytes_ / kExpertBytes);
            const std::size_t base = slots / model_.config().num_hidden_layers;
            const std::size_t remainder = slots % model_.config().num_hidden_layers;
            const std::size_t hot_for_layer = base + (layer < remainder ? 1U : 0U);
            retain = expert < hot_for_layer;
        }
        if (retain) evict_for_one();

        const auto started = std::chrono::steady_clock::now();
        auto loaded = model_.read_expert(layer, expert);
        if (!loaded) return loaded.error();
        const double io_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::shared_ptr<const OlmoeRuntimeModel::ExpertWeights> shared;
        try {
            shared = std::make_shared<OlmoeRuntimeModel::ExpertWeights>(loaded.take_value());
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted, "Unable to retain streamed OLMoE expert");
        }
        if (retain) {
            entries_.emplace(key, Entry{shared, clock_, 1U});
            used_bytes_ += kExpertBytes;
        }
        return AcquiredExpert{std::move(shared), false, kExpertBytes, io_ms};
    }

private:
    struct Entry {
        std::shared_ptr<const OlmoeRuntimeModel::ExpertWeights> weights;
        std::uint64_t last_use{0U};
        std::uint64_t frequency{0U};
    };

    ExpertCache(const OlmoeRuntimeModel& model,
                ExpertCachePolicy policy,
                ByteCount capacity,
                safety::Reservation reservation)
        : model_(model), policy_(policy), capacity_bytes_(capacity), reservation_(std::move(reservation)) {}

    void evict_for_one() {
        while (used_bytes_ + kExpertBytes > capacity_bytes_ && !entries_.empty()) {
            auto victim = entries_.begin();
            for (auto iterator = std::next(entries_.begin()); iterator != entries_.end(); ++iterator) {
                const bool replace = policy_ == ExpertCachePolicy::lfu
                                         ? std::pair{iterator->second.frequency, iterator->second.last_use} <
                                               std::pair{victim->second.frequency, victim->second.last_use}
                                         : iterator->second.last_use < victim->second.last_use;
                if (replace) victim = iterator;
            }
            entries_.erase(victim);
            used_bytes_ -= kExpertBytes;
        }
    }

    const OlmoeRuntimeModel& model_;
    ExpertCachePolicy policy_{ExpertCachePolicy::none};
    ByteCount capacity_bytes_{0U};
    ByteCount used_bytes_{0U};
    std::uint64_t clock_{0U};
    std::map<std::uint32_t, Entry> entries_;
    safety::Reservation reservation_;
};

struct Session {
    ParallelExecutor executor;
    std::size_t sequence_length{0U};
    std::size_t hidden_size{0U};
    std::vector<float> keys;
    std::vector<float> values;

    Session(const OlmoeRuntimeModel& model, std::size_t sequence)
        : executor(model.cpu_threads()), sequence_length(sequence), hidden_size(model.config().hidden_size),
          keys(static_cast<std::size_t>(model.config().num_hidden_layers) * sequence * hidden_size), values(keys.size()) {}
};

bool expert_tensor_name(std::string_view name) {
    return name.find(".mlp.experts.") != std::string_view::npos;
}

const model::TensorDescriptor& descriptor_at(const model::CheckpointCatalog& catalog,
                                             const model::TensorLocation& location) {
    return catalog.shards[location.shard_index].tensors()[location.tensor_index];
}

}  // namespace

Result<ExpertCachePolicy> parse_expert_cache_policy(std::string_view value) {
    const auto normalized = util::to_lower(util::trim(value));
    if (normalized == "none" || normalized == "no-cache") return ExpertCachePolicy::none;
    if (normalized == "lru") return ExpertCachePolicy::lru;
    if (normalized == "lfu") return ExpertCachePolicy::lfu;
    if (normalized == "static-hot") return ExpertCachePolicy::static_hot;
    return make_error(ErrorCode::invalid_argument, "Unknown OLMoE expert-cache policy", std::string(value));
}

std::string_view to_string(ExpertCachePolicy policy) noexcept {
    switch (policy) {
        case ExpertCachePolicy::none: return "no-cache";
        case ExpertCachePolicy::lru: return "lru";
        case ExpertCachePolicy::lfu: return "lfu";
        case ExpertCachePolicy::static_hot: return "static-hot";
    }
    return "unknown";
}

Result<OlmoeRuntimeModel> OlmoeRuntimeModel::load(const std::filesystem::path& model_directory,
                                                 safety::ResourceGovernor& governor,
                                                 OlmoeRuntimeLoadOptions options) {
    if constexpr (std::endian::native != std::endian::little) {
        return make_error(ErrorCode::unsupported, "OLMoE BF16 checkpoint loading currently requires little-endian hardware");
    }
    const auto started = std::chrono::steady_clock::now();
    auto inspection = inspect_olmoe_checkpoint(model_directory, options.verify_hashes);
    if (!inspection) return inspection.error();
    if (inspection.value().metadata_only) {
        return make_error(ErrorCode::not_found, "OLMoE runtime requires the complete verified checkpoint", model_directory.string());
    }
    auto tokenizer = Gpt2Tokenizer::load(model_directory);
    if (!tokenizer) return tokenizer.error();
    auto catalog = model::inspect_safetensors_checkpoint(model_directory);
    if (!catalog) return catalog.error();
    if (catalog.value().total_tensor_bytes != inspection.value().tensor_payload_bytes) {
        return make_error(ErrorCode::integrity_error, "OLMoE Safetensors payload differs from the pinned index");
    }

    ByteCount resident_bytes = 0U;
    for (const auto& [name, location] : catalog.value().tensors) {
        if (expert_tensor_name(name)) continue;
        const auto& tensor = descriptor_at(catalog.value(), location);
        if (tensor.dtype != model::TensorDType::bf16) {
            return make_error(ErrorCode::unsupported, "OLMoE runtime requires BF16 tensors", name);
        }
        auto added = checked_add(resident_bytes, tensor.byte_count);
        if (!added) return added.error();
        resident_bytes = added.value();
    }
    if (resident_bytes != kResidentBytes || (resident_bytes % 2U) != 0U) {
        return make_error(ErrorCode::integrity_error, "OLMoE always-active tensor byte count differs from the pinned layout");
    }
    auto reservation = governor.try_reserve(safety::ResourceKind::ram, resident_bytes, "OLMoE immutable always-active BF16 weights");
    if (!reservation) return reservation.error();

    OlmoeRuntimeModel result;
    result.config_ = inspection.value().config;
    result.tokenizer_ = tokenizer.take_value();
    result.catalog_ = catalog.take_value();
    result.cpu_threads_ = std::clamp(options.cpu_threads, 1U, 6U);
    result.resident_reservation_ = reservation.take_value();
    try {
        result.resident_.resize(static_cast<std::size_t>(resident_bytes / 2U));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate reserved OLMoE always-active weight store");
    }

    ByteCount cursor_bytes = 0U;
    for (const auto& [name, location] : result.catalog_.tensors) {
        if (expert_tensor_name(name)) continue;
        const auto& shard = result.catalog_.shards[location.shard_index];
        const auto& tensor = shard.tensors()[location.tensor_index];
        const std::size_t element_offset = static_cast<std::size_t>(cursor_bytes / 2U);
        result.resident_tensors_.emplace(name, StoredTensor{element_offset, tensor.shape});
        for (ByteCount offset = 0U; offset < tensor.byte_count;) {
            const ByteCount bytes = std::min(kLoadChunkBytes, tensor.byte_count - offset);
            auto destination = std::span<std::byte>(
                reinterpret_cast<std::byte*>(result.resident_.data()) + cursor_bytes + offset,
                static_cast<std::size_t>(bytes));
            auto read = shard.read_range_into(tensor, offset, destination, kLoadChunkBytes);
            if (!read) return read.error();
            offset += bytes;
        }
        cursor_bytes += tensor.byte_count;
    }
    if (cursor_bytes != resident_bytes) {
        return make_error(ErrorCode::integrity_error, "OLMoE resident tensor copy did not fill its reservation");
    }

    auto embedding = result.require_resident("model.embed_tokens.weight", {50'304U, 2'048U});
    auto lm_head = result.require_resident("lm_head.weight", {50'304U, 2'048U});
    auto final_norm = result.require_resident("model.norm.weight", {2'048U});
    if (!embedding) return embedding.error();
    if (!lm_head) return lm_head.error();
    if (!final_norm) return final_norm.error();
    result.embedding_ = embedding.value();
    result.lm_head_ = lm_head.value();
    result.final_norm_ = final_norm.value();
    result.layers_.reserve(result.config_.num_hidden_layers);
    for (std::uint32_t layer = 0U; layer < result.config_.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        LayerWeights weights;
        auto input_norm = result.require_resident(prefix + "input_layernorm.weight", {2'048U});
        auto q = result.require_resident(prefix + "self_attn.q_proj.weight", {2'048U, 2'048U});
        auto k = result.require_resident(prefix + "self_attn.k_proj.weight", {2'048U, 2'048U});
        auto v = result.require_resident(prefix + "self_attn.v_proj.weight", {2'048U, 2'048U});
        auto o = result.require_resident(prefix + "self_attn.o_proj.weight", {2'048U, 2'048U});
        auto q_norm = result.require_resident(prefix + "self_attn.q_norm.weight", {2'048U});
        auto k_norm = result.require_resident(prefix + "self_attn.k_norm.weight", {2'048U});
        auto post_norm = result.require_resident(prefix + "post_attention_layernorm.weight", {2'048U});
        auto router = result.require_resident(prefix + "mlp.gate.weight", {64U, 2'048U});
        if (!input_norm) return input_norm.error();
        if (!q) return q.error();
        if (!k) return k.error();
        if (!v) return v.error();
        if (!o) return o.error();
        if (!q_norm) return q_norm.error();
        if (!k_norm) return k_norm.error();
        if (!post_norm) return post_norm.error();
        if (!router) return router.error();
        weights.input_norm = input_norm.value();
        weights.q = q.value();
        weights.k = k.value();
        weights.v = v.value();
        weights.o = o.value();
        weights.q_norm = q_norm.value();
        weights.k_norm = k_norm.value();
        weights.post_norm = post_norm.value();
        weights.router = router.value();
        for (std::uint32_t expert = 0U; expert < result.config_.num_experts; ++expert) {
            const std::string expert_prefix = prefix + "mlp.experts." + std::to_string(expert) + ".";
            auto gate = result.require_streamed(expert_prefix + "gate_proj.weight", {1'024U, 2'048U});
            auto up = result.require_streamed(expert_prefix + "up_proj.weight", {1'024U, 2'048U});
            auto down = result.require_streamed(expert_prefix + "down_proj.weight", {2'048U, 1'024U});
            if (!gate) return gate.error();
            if (!up) return up.error();
            if (!down) return down.error();
            weights.experts[expert] = ExpertLocations{gate.value(), up.value(), down.value()};
        }
        result.layers_.push_back(std::move(weights));
    }
    result.load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}

Result<OlmoeRuntimeModel::WeightView> OlmoeRuntimeModel::require_resident(
    std::string_view name,
    const std::vector<std::uint64_t>& shape) const {
    const auto iterator = resident_tensors_.find(name);
    if (iterator == resident_tensors_.end() || iterator->second.shape != shape) {
        return make_error(ErrorCode::integrity_error, "OLMoE resident tensor is missing or has the wrong shape", std::string(name));
    }
    std::size_t rows = 1U;
    std::size_t columns = static_cast<std::size_t>(shape.front());
    if (shape.size() == 2U) {
        rows = static_cast<std::size_t>(shape[0]);
        columns = static_cast<std::size_t>(shape[1]);
    }
    return WeightView{iterator->second.offset, rows, columns};
}

Result<model::TensorLocation> OlmoeRuntimeModel::require_streamed(
    std::string_view name,
    const std::vector<std::uint64_t>& shape) const {
    const auto iterator = catalog_.tensors.find(name);
    if (iterator == catalog_.tensors.end()) {
        return make_error(ErrorCode::integrity_error, "OLMoE expert tensor is missing", std::string(name));
    }
    const auto& tensor = descriptor_at(catalog_, iterator->second);
    if (tensor.dtype != model::TensorDType::bf16 || tensor.shape != shape) {
        return make_error(ErrorCode::integrity_error, "OLMoE expert tensor has the wrong dtype or shape", std::string(name));
    }
    return iterator->second;
}

Result<OlmoeRuntimeModel::ExpertWeights> OlmoeRuntimeModel::read_expert(std::uint32_t layer,
                                                                       std::uint32_t expert) const {
    if (layer >= layers_.size() || expert >= config_.num_experts) {
        return make_error(ErrorCode::invalid_argument, "OLMoE expert address is out of range");
    }
    ExpertWeights result;
    constexpr std::size_t matrix_elements = 1'024U * 2'048U;
    try {
        result.values.resize(matrix_elements * 3U);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate one bounded OLMoE expert");
    }
    result.gate = WeightView{0U, 1'024U, 2'048U};
    result.up = WeightView{matrix_elements, 1'024U, 2'048U};
    result.down = WeightView{matrix_elements * 2U, 2'048U, 1'024U};
    const auto& locations = layers_[layer].experts[expert];
    const std::array<std::pair<model::TensorLocation, std::size_t>, 3> reads{{
        {locations.gate, result.gate.offset},
        {locations.up, result.up.offset},
        {locations.down, result.down.offset},
    }};
    for (const auto& [location, element_offset] : reads) {
        const auto& shard = catalog_.shards[location.shard_index];
        const auto& tensor = shard.tensors()[location.tensor_index];
        auto destination = std::span<std::byte>(reinterpret_cast<std::byte*>(result.values.data() + element_offset),
                                                static_cast<std::size_t>(tensor.byte_count));
        auto read = shard.read_range_into(tensor, 0U, destination, kLoadChunkBytes);
        if (!read) return read.error();
    }
    return result;
}

Result<OlmoeForwardResult> OlmoeRuntimeModel::forward(const std::vector<std::uint32_t>& token_ids,
                                                      const OlmoeForwardOptions& options,
                                                      safety::ResourceGovernor& governor,
                                                      const safety::ProcessSafetyGuard* process_guard) const {
    if (token_ids.empty() || token_ids.size() > config_.max_position_embeddings) {
        return make_error(ErrorCode::invalid_argument, "OLMoE forward input is empty or exceeds context");
    }
    if (options.timeout <= std::chrono::milliseconds::zero()) {
        return make_error(ErrorCode::invalid_argument, "OLMoE forward timeout must be positive");
    }
    for (const auto token : token_ids) {
        if (token >= config_.vocab_size) return make_error(ErrorCode::invalid_argument, "OLMoE input token exceeds vocabulary");
    }
    auto cache = ExpertCache::create(*this, options.cache_policy, options.expert_cache_bytes, governor);
    if (!cache) return cache.error();
    auto kv_bytes = kv_cache_bytes(config_, token_ids.size());
    if (!kv_bytes) return kv_bytes.error();
    auto kv_reservation = governor.try_reserve(safety::ResourceKind::ram, kv_bytes.value(), "OLMoE incremental KV cache");
    if (!kv_reservation) return kv_reservation.error();
    auto scratch_reservation = governor.try_reserve(safety::ResourceKind::scratch, 64U * MiB, "OLMoE reference activation scratch");
    if (!scratch_reservation) return scratch_reservation.error();

    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + options.timeout;
    OlmoeForwardResult result;
    result.stats.input_tokens = token_ids.size();
    result.stats.load_ms = load_ms_;
    try {
        Session session(*this, token_ids.size());
        const std::size_t hidden_size = config_.hidden_size;
        const std::size_t head_size = hidden_size / config_.num_attention_heads;
        std::vector<float> hidden(hidden_size);
        std::vector<float> residual(hidden_size);
        std::vector<float> normalized(hidden_size);
        std::vector<float> query(hidden_size);
        std::vector<float> key(hidden_size);
        std::vector<float> value(hidden_size);
        std::vector<float> attention(hidden_size);
        std::vector<float> projected(hidden_size);
        std::vector<float> moe(hidden_size);
        std::vector<float> gate(config_.intermediate_size);
        std::vector<float> up(config_.intermediate_size);
        std::vector<float> intermediate(config_.intermediate_size);
        std::vector<float> expert_output(hidden_size);
        std::vector<float> scores(token_ids.size());
        std::array<float, 64> router_logits{};
        std::array<float, 64> routing_weights{};
        std::array<std::uint32_t, 64> expert_order{};
        result.trace.reserve(token_ids.size() * config_.num_hidden_layers * config_.num_experts_per_token);

        for (std::size_t position = 0U; position < token_ids.size(); ++position) {
            const std::size_t embedding_offset = embedding_.offset + static_cast<std::size_t>(token_ids[position]) * hidden_size;
            for (std::size_t index = 0U; index < hidden_size; ++index) {
                hidden[index] = bf16_to_float(resident_[embedding_offset + index]);
            }
            for (std::uint32_t layer = 0U; layer < config_.num_hidden_layers; ++layer) {
                if ((options.cancelled && options.cancelled()) || std::chrono::steady_clock::now() >= deadline) {
                    return make_error(ErrorCode::resource_exhausted, "OLMoE forward was cancelled or timed out safely");
                }
                if (process_guard != nullptr) {
                    auto checked = process_guard->sample_and_check();
                    if (!checked) return checked.error();
                }
                const auto& weights = layers_[layer];
                residual = hidden;
                rms_norm(resident_, weights.input_norm, hidden, normalized, config_.rms_norm_eps);
                matrix_vector(resident_, weights.q, normalized, query, session.executor);
                matrix_vector(resident_, weights.k, normalized, key, session.executor);
                matrix_vector(resident_, weights.v, normalized, value, session.executor);
                rms_norm(resident_, weights.q_norm, query, query, config_.rms_norm_eps);
                rms_norm(resident_, weights.k_norm, key, key, config_.rms_norm_eps);
                apply_rope(query, key, config_.num_attention_heads, head_size, position, config_.rope_theta);

                const std::size_t kv_base = (static_cast<std::size_t>(layer) * token_ids.size() + position) * hidden_size;
                std::copy(key.begin(), key.end(), session.keys.begin() + static_cast<std::ptrdiff_t>(kv_base));
                std::copy(value.begin(), value.end(), session.values.begin() + static_cast<std::ptrdiff_t>(kv_base));
                std::fill(attention.begin(), attention.end(), 0.0F);
                const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));
                for (std::size_t head = 0U; head < config_.num_attention_heads; ++head) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (std::size_t source = 0U; source <= position; ++source) {
                        const std::size_t source_base =
                            (static_cast<std::size_t>(layer) * token_ids.size() + source) * hidden_size + head * head_size;
                        double dot = 0.0;
                        for (std::size_t index = 0U; index < head_size; ++index) {
                            dot += static_cast<double>(query[head * head_size + index]) *
                                   static_cast<double>(session.keys[source_base + index]);
                        }
                        scores[source] = static_cast<float>(dot) * attention_scale;
                        maximum = std::max(maximum, scores[source]);
                    }
                    double denominator = 0.0;
                    for (std::size_t source = 0U; source <= position; ++source) {
                        scores[source] = std::exp(scores[source] - maximum);
                        denominator += static_cast<double>(scores[source]);
                    }
                    for (std::size_t source = 0U; source <= position; ++source) {
                        const float probability = scores[source] / static_cast<float>(denominator);
                        const std::size_t source_base =
                            (static_cast<std::size_t>(layer) * token_ids.size() + source) * hidden_size + head * head_size;
                        for (std::size_t index = 0U; index < head_size; ++index) {
                            attention[head * head_size + index] += probability * session.values[source_base + index];
                        }
                    }
                }
                matrix_vector(resident_, weights.o, attention, projected, session.executor);
                for (std::size_t index = 0U; index < hidden_size; ++index) hidden[index] = residual[index] + projected[index];

                residual = hidden;
                rms_norm(resident_, weights.post_norm, hidden, normalized, config_.rms_norm_eps);
                matrix_vector(resident_, weights.router, normalized, router_logits, session.executor);
                const float maximum_router = *std::max_element(router_logits.begin(), router_logits.end());
                double router_denominator = 0.0;
                for (std::size_t expert = 0U; expert < router_logits.size(); ++expert) {
                    routing_weights[expert] = std::exp(router_logits[expert] - maximum_router);
                    router_denominator += static_cast<double>(routing_weights[expert]);
                    expert_order[expert] = static_cast<std::uint32_t>(expert);
                }
                for (auto& routing_weight : routing_weights) routing_weight /= static_cast<float>(router_denominator);
                std::partial_sort(expert_order.begin(),
                                  expert_order.begin() + config_.num_experts_per_token,
                                  expert_order.end(),
                                  [&](std::uint32_t left, std::uint32_t right) {
                                      if (routing_weights[left] != routing_weights[right]) {
                                          return routing_weights[left] > routing_weights[right];
                                      }
                                      return left < right;
                                  });
                std::fill(moe.begin(), moe.end(), 0.0F);
                for (std::uint32_t route = 0U; route < config_.num_experts_per_token; ++route) {
                    const std::uint32_t expert = expert_order[route];
                    auto acquired = cache.value().acquire(layer, expert);
                    if (!acquired) return acquired.error();
                    result.stats.expert_bytes_read += acquired.value().bytes_read;
                    result.stats.expert_io_ms += acquired.value().io_ms;
                    if (acquired.value().hit) {
                        ++result.stats.expert_cache_hits;
                    } else {
                        ++result.stats.expert_cache_misses;
                    }
                    result.trace.push_back(ExpertTraceEvent{
                        position, layer, expert, routing_weights[expert], acquired.value().hit,
                        acquired.value().bytes_read, acquired.value().io_ms});
                    const auto& expert_weights = *acquired.value().weights;
                    matrix_vector(expert_weights.values, expert_weights.gate, normalized, gate, session.executor);
                    matrix_vector(expert_weights.values, expert_weights.up, normalized, up, session.executor);
                    for (std::size_t index = 0U; index < intermediate.size(); ++index) {
                        intermediate[index] = (gate[index] / (1.0F + std::exp(-gate[index]))) * up[index];
                    }
                    matrix_vector(expert_weights.values, expert_weights.down, intermediate, expert_output, session.executor);
                    const float route_weight = routing_weights[expert];
                    for (std::size_t index = 0U; index < hidden_size; ++index) {
                        moe[index] += route_weight * expert_output[index];
                    }
                }
                for (std::size_t index = 0U; index < hidden_size; ++index) hidden[index] = residual[index] + moe[index];
            }
        }
        rms_norm(resident_, final_norm_, hidden, normalized, config_.rms_norm_eps);
        result.logits.resize(config_.vocab_size);
        matrix_vector(resident_, lm_head_, normalized, result.logits, session.executor);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate reserved OLMoE forward buffers");
    }
    result.stats.forward_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace stargate::inference
