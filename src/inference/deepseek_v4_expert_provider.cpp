#include "vastloom/inference/deepseek_v4_expert_provider.hpp"

#include <algorithm>
#include <exception>
#include <future>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>

#include "vastloom/inference/ggml_quantization.hpp"

namespace vastloom::inference {
namespace {

Result<void> validate_binding(const DeepseekV4ExpertBundleBinding& binding,
                              const DeepseekV4ExpertCacheOptions& options) {
    const auto& bundle = binding.bundle;
    if (bundle.expert != binding.expert_id || bundle.bytes == 0U ||
        bundle.bytes > options.load.max_expert_bytes ||
        bundle.tensor_indices.size() != binding.matrices.size() ||
        bundle.tensor_slices.size() != binding.matrices.size()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek expert cache binding is incomplete",
                          std::to_string(binding.expert_id));
    }
    std::array<bool, 3U> roles{};
    for (const auto& matrix : binding.matrices) {
        const auto role = static_cast<std::size_t>(matrix.weight);
        if (role >= roles.size() || roles[role] ||
            std::find(bundle.tensor_indices.begin(), bundle.tensor_indices.end(),
                      matrix.catalog_tensor_index) == bundle.tensor_indices.end()) {
            return make_error(ErrorCode::invalid_format,
                              "DeepSeek expert cache matrix identity is invalid",
                              std::to_string(binding.expert_id));
        }
        roles[role] = true;
        auto required = ggml_storage_bytes(matrix.ggml_type, matrix.rows,
                                           matrix.storage_columns);
        if (!required) return required.error();
        const auto slice = std::find_if(
            bundle.tensor_slices.begin(), bundle.tensor_slices.end(),
            [&](const model::ExpertTensorSlice& value) {
                return value.tensor_index == matrix.catalog_tensor_index;
            });
        if (matrix.logical_columns == 0U ||
            matrix.logical_columns > matrix.storage_columns ||
            slice == bundle.tensor_slices.end() ||
            slice->bytes != required.value()) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek expert cache matrix layout does not match its bundle",
                              std::to_string(binding.expert_id));
        }
    }
    return {};
}

Result<DeepseekV4ResolvedExpert> bind_loaded_expert(
    const DeepseekV4ExpertBundleBinding& binding,
    const model::LoadedExpert& loaded,
    bool asynchronous_load) {
    DeepseekV4ResolvedExpert expert;
    expert.expert_id = binding.expert_id;
    expert.resident_bytes = binding.bundle.bytes;
    const auto& telemetry = loaded.telemetry();
    expert.logical_bytes_read = telemetry.logical_bytes;
    expert.asynchronous_load = asynchronous_load;
    expert.load_request_sequence = telemetry.request_sequence;
    expert.load_queue_ms = telemetry.queue_ms;
    expert.load_service_ms = telemetry.service_ms;
    expert.load_total_ms = telemetry.total_ms;
    expert.load_service_started_at = telemetry.service_started_at;
    expert.load_completed_at = telemetry.completed_at;
    for (const auto& matrix : binding.matrices) {
        // Layer, expert and catalog tensor identify these weights for as long as the checkpoint
        // exists, which is what a device-side cache needs. The host address does not: this bundle's
        // buffer is released once the layer is done, and the next expert may land on it.
        std::uint64_t identity = UINT64_C(14695981039346656037);
        for (const std::uint64_t part : {static_cast<std::uint64_t>(binding.bundle.layer),
                                         static_cast<std::uint64_t>(binding.bundle.expert),
                                         static_cast<std::uint64_t>(matrix.catalog_tensor_index)}) {
            for (unsigned shift = 0U; shift < 64U; shift += 8U) {
                identity ^= (part >> shift) & 0xffU;
                identity *= UINT64_C(1099511628211);
            }
        }
        auto packed = deepseek_v4_bind_packed_matrix(
            loaded, matrix.catalog_tensor_index, matrix.ggml_type,
            matrix.rows, matrix.logical_columns, matrix.storage_columns, identity | 1ULL);
        if (!packed) return packed.error();
        switch (matrix.weight) {
            case DeepseekV4ExpertWeight::w1:
                expert.w1 = packed.take_value();
                break;
            case DeepseekV4ExpertWeight::w2:
                expert.w2 = packed.take_value();
                break;
            case DeepseekV4ExpertWeight::w3:
                expert.w3 = packed.take_value();
                break;
        }
    }
    if (expert.w1.resident == nullptr ||
        expert.w1.resident != expert.w2.resident ||
        expert.w1.resident != expert.w3.resident) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek expert matrices do not share one indivisible bundle",
                          std::to_string(binding.expert_id));
    }
    return expert;
}

}  // namespace

Result<std::vector<DeepseekV4ExpertBundleBinding>>
deepseek_v4_make_gguf_expert_bindings(
    const model::LogicalTensorCatalog& catalog,
    std::uint32_t layer,
    std::uint32_t expected_experts,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    std::size_t max_experts) {
    if (catalog.model_id.empty() || catalog.revision.empty() || catalog.metadata_only ||
        expected_experts == 0U || expected_experts > max_experts ||
        hidden_width == 0U || intermediate_width == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek GGUF expert binding request is not executable",
                          catalog.model_id);
    }
    try {
        std::vector<DeepseekV4ExpertBundleBinding> result;
        result.reserve(expected_experts);
        for (const auto& bundle : catalog.experts) {
            if (bundle.layer != layer || bundle.speculative) continue;
            if (bundle.expert >= expected_experts || bundle.tensor_indices.size() != 3U ||
                bundle.tensor_slices.size() != 3U) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF layer has an invalid expert bundle",
                                  std::to_string(bundle.expert));
            }

            DeepseekV4ExpertBundleBinding binding;
            binding.expert_id = bundle.expert;
            binding.bundle = bundle;
            std::array<bool, 3U> roles{};
            const std::string prefix =
                "blk." + std::to_string(layer) + ".ffn_";
            const std::string suffix =
                ".weight.expert." + std::to_string(bundle.expert);
            for (const auto tensor_index : bundle.tensor_indices) {
                if (tensor_index >= catalog.tensors.size()) {
                    return make_error(ErrorCode::invalid_format,
                                      "DeepSeek GGUF expert tensor index is out of range",
                                      std::to_string(bundle.expert));
                }
                const auto& descriptor = catalog.tensors[tensor_index];
                DeepseekV4ExpertWeight weight{};
                std::size_t rows = 0U;
                std::size_t logical_columns = 0U;
                if (descriptor.name == prefix + "gate_exps" + suffix) {
                    weight = DeepseekV4ExpertWeight::w1;
                    rows = intermediate_width;
                    logical_columns = hidden_width;
                } else if (descriptor.name == prefix + "down_exps" + suffix) {
                    weight = DeepseekV4ExpertWeight::w2;
                    rows = hidden_width;
                    logical_columns = intermediate_width;
                } else if (descriptor.name == prefix + "up_exps" + suffix) {
                    weight = DeepseekV4ExpertWeight::w3;
                    rows = intermediate_width;
                    logical_columns = hidden_width;
                } else {
                    return make_error(ErrorCode::invalid_format,
                                      "DeepSeek GGUF expert tensor name is not canonical",
                                      descriptor.name);
                }
                const auto role = static_cast<std::size_t>(weight);
                if (roles[role] || descriptor.role != model::TensorRole::routed_expert ||
                    descriptor.layer != layer || descriptor.expert != bundle.expert ||
                    descriptor.speculative || descriptor.shape.size() != 2U ||
                    !descriptor.ggml_type.has_value() ||
                    descriptor.shape[0] >
                        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                    descriptor.shape[1] != rows) {
                    return make_error(ErrorCode::invalid_format,
                                      "DeepSeek GGUF expert tensor metadata is inconsistent",
                                      descriptor.name);
                }
                const auto storage_columns =
                    static_cast<std::size_t>(descriptor.shape[0]);
                if (storage_columns < logical_columns) {
                    return make_error(ErrorCode::invalid_format,
                                      "DeepSeek GGUF expert storage columns are truncated",
                                      descriptor.name);
                }
                auto required = ggml_storage_bytes(
                    *descriptor.ggml_type, rows, storage_columns);
                if (!required || required.value() != descriptor.source.bytes) {
                    return make_error(ErrorCode::integrity_error,
                                      "DeepSeek GGUF expert encoded bytes do not match its shape",
                                      descriptor.name);
                }
                roles[role] = true;
                binding.matrices[role] = DeepseekV4ExpertMatrixBinding{
                    weight, tensor_index, *descriptor.ggml_type, rows,
                    logical_columns, storage_columns};
            }
            if (!std::all_of(roles.begin(), roles.end(), [](bool present) {
                    return present;
                })) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek GGUF expert bundle is incomplete",
                                  std::to_string(bundle.expert));
            }
            DeepseekV4ExpertCacheOptions validation_options;
            validation_options.load.max_expert_bytes =
                std::numeric_limits<ByteCount>::max();
            auto valid = validate_binding(binding, validation_options);
            if (!valid) return valid.error();
            result.push_back(std::move(binding));
        }
        std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
            return left.expert_id < right.expert_id;
        });
        if (result.size() != expected_experts) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek GGUF layer has the wrong complete expert count",
                              std::to_string(layer));
        }
        for (std::size_t expert = 0U; expert < result.size(); ++expert) {
            if (result[expert].expert_id != expert) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek GGUF layer expert identities are not contiguous",
                                  std::to_string(layer));
            }
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate DeepSeek GGUF expert bindings",
                          catalog.model_id);
    }
}

Result<void> deepseek_v4_execute_resolved_iq1_s_iq3_xxs_f32(
    const DeepseekV4ResolvedExpert& expert,
    std::span<const float> input,
    RoutedExpertScratch scratch,
    std::span<float> output,
    const QuantizedKernelLimits& limits) {
    const std::size_t intermediate_width = scratch.gate.size();
    if (expert.w1.ggml_type != kGgmlTypeIq1S ||
        expert.w3.ggml_type != kGgmlTypeIq1S ||
        expert.w2.ggml_type != kGgmlTypeIq3Xxs || input.empty() || output.empty() ||
        input.size() != output.size() || scratch.up.size() != intermediate_width ||
        scratch.down.size() != output.size() ||
        expert.w1.rows != intermediate_width ||
        expert.w3.rows != intermediate_width || expert.w2.rows != output.size() ||
        expert.w1.logical_columns != input.size() ||
        expert.w3.logical_columns != input.size() ||
        expert.w2.logical_columns != intermediate_width ||
        expert.w1.storage_columns != expert.w1.logical_columns ||
        expert.w3.storage_columns != expert.w3.logical_columns ||
        expert.w2.storage_columns != expert.w2.logical_columns) {
        return make_error(ErrorCode::invalid_format,
                          "Resolved DeepSeek IQ1_S/IQ3_XXS expert has invalid storage dimensions",
                          std::to_string(expert.expert_id));
    }
    auto gate = deepseek_v4_packed_matrix_payload(expert.w1, "resolved expert gate");
    if (!gate) return gate.error();
    auto up = deepseek_v4_packed_matrix_payload(expert.w3, "resolved expert up");
    if (!up) return up.error();
    auto down = deepseek_v4_packed_matrix_payload(expert.w2, "resolved expert down");
    if (!down) return down.error();
    return routed_expert_iq1_s_iq3_xxs_f32(
        gate.value(), up.value(), down.value(), input, scratch, output, limits);
}

Result<std::shared_ptr<DeepseekV4BoundedExpertCache>>
DeepseekV4BoundedExpertCache::create(
    std::vector<DeepseekV4ExpertBundleBinding> bindings,
    safety::ResourceGovernor& governor,
    DeepseekV4ExpertCacheOptions options) {
    if (bindings.empty() || options.block_prefix.empty() ||
        options.max_cached_experts == 0U ||
        options.max_experts_per_request == 0U ||
        options.max_experts_per_request > options.max_cached_experts ||
        options.asynchronous_workers > 4U ||
        (options.asynchronous_workers == 0U &&
         options.asynchronous_queue_capacity != 0U) ||
        (options.asynchronous_workers != 0U &&
          (options.asynchronous_queue_capacity < options.max_experts_per_request ||
           options.asynchronous_queue_capacity > 256U)) ||
        options.load.max_expert_bytes == 0U ||
        options.load.max_range_bytes == 0U || options.load.max_ranges == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek expert cache options are invalid",
                          options.block_prefix);
    }
    try {
        std::map<std::uint32_t, DeepseekV4ExpertBundleBinding> indexed;
        for (auto& binding : bindings) {
            auto valid = validate_binding(binding, options);
            if (!valid) return valid.error();
            const auto expert_id = binding.expert_id;
            if (!indexed.emplace(expert_id, std::move(binding)).second) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek expert cache contains a duplicate expert",
                                  std::to_string(expert_id));
            }
        }
        return std::shared_ptr<DeepseekV4BoundedExpertCache>(
            new DeepseekV4BoundedExpertCache(
                std::move(indexed), governor, std::move(options)));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded DeepSeek expert cache");
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to start the bounded DeepSeek expert cache",
                          exception.what());
    } catch (...) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to start the bounded DeepSeek expert cache");
    }
}

DeepseekV4BoundedExpertCache::DeepseekV4BoundedExpertCache(
    std::map<std::uint32_t, DeepseekV4ExpertBundleBinding> bindings,
    safety::ResourceGovernor& governor,
    DeepseekV4ExpertCacheOptions options)
    : bindings_(std::move(bindings)), governor_(governor), options_(std::move(options)) {
    if (options_.asynchronous_workers != 0U) {
        model::ExpertLoadPoolOptions pool_options;
        pool_options.workers = options_.asynchronous_workers;
        pool_options.max_queued_requests = options_.asynchronous_queue_capacity;
        pool_options.load = options_.load;
        load_pool_ = std::make_unique<model::BoundedExpertLoadPool>(
            std::move(pool_options), governor_);
    }
}

DeepseekV4BoundedExpertCache::~DeepseekV4BoundedExpertCache() {
    static_cast<void>(clear());
}

DeepseekV4ExpertProviderBinding DeepseekV4BoundedExpertCache::provider_binding() {
    auto self = shared_from_this();
    DeepseekV4ExpertProviderBinding binding{
        options_.max_experts_per_request,
        [self](
            std::string_view block_prefix,
            std::span<const std::uint32_t> unique_expert_ids) {
            return self->resolve(block_prefix, unique_expert_ids);
        }};
    if (load_pool_ != nullptr) {
        binding.begin = [self = std::move(self)](
                            std::string_view block_prefix,
                            std::span<const std::uint32_t> unique_expert_ids) {
            return self->begin_resolve(block_prefix, unique_expert_ids);
        };
    }
    return binding;
}

DeepseekV4ExpertCacheSnapshot DeepseekV4BoundedExpertCache::snapshot() const {
    std::scoped_lock lock(mutex_);
    auto snapshot = telemetry_;
    snapshot.cached_experts = cache_.size();
    return snapshot;
}

class DeepseekV4BoundedExpertCache::AsyncBatch final
    : public DeepseekV4ExpertBatch {
public:
    AsyncBatch(std::shared_ptr<DeepseekV4BoundedExpertCache> cache,
               std::string block_prefix,
               std::span<const std::uint32_t> expert_ids)
        : cache_(std::move(cache)),
          block_prefix_(std::move(block_prefix)),
          requested_ids_(expert_ids.begin(), expert_ids.end()) {}

    ~AsyncBatch() override { cleanup(); }

    [[nodiscard]] Result<void> prepare() {
        if (cache_ == nullptr || cache_->load_pool_ == nullptr) {
            return make_error(ErrorCode::internal_error,
                              "DeepSeek asynchronous expert pool is unavailable",
                              block_prefix_);
        }
        {
            std::scoped_lock lock(cache_->mutex_);
            if (cache_->async_batch_active_) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek expert cache already has an active asynchronous batch",
                                  block_prefix_);
            }
            if (block_prefix_ != cache_->options_.block_prefix ||
                requested_ids_.empty()) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek selected-expert request does not match its cache binding",
                                  block_prefix_);
            }
            if (requested_ids_.size() >
                cache_->options_.max_experts_per_request) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek selected-expert request exceeds its cache binding",
                                  block_prefix_);
            }
            for (std::size_t index = 0U; index < requested_ids_.size(); ++index) {
                if (!cache_->bindings_.contains(requested_ids_[index]) ||
                    std::find(requested_ids_.begin(),
                              requested_ids_.begin() +
                                  static_cast<std::ptrdiff_t>(index),
                              requested_ids_[index]) !=
                        requested_ids_.begin() +
                            static_cast<std::ptrdiff_t>(index)) {
                    return make_error(
                        ErrorCode::invalid_argument,
                        "DeepSeek selected-expert request contains an unknown or duplicate identity",
                        std::to_string(requested_ids_[index]));
                }
            }
            if (cache_->telemetry_.requests ==
                    std::numeric_limits<std::size_t>::max() ||
                cache_->telemetry_.telemetry_overflow ||
                requested_ids_.size() >
                    std::numeric_limits<std::size_t>::max() -
                        cache_->telemetry_.requested_experts) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek expert cache request telemetry overflow",
                                  block_prefix_);
            }

            cache_->async_batch_active_ = true;
            active_ = true;
            entries_.reserve(requested_ids_.size());
            std::size_t misses = 0U;
            for (const auto expert_id : requested_ids_) {
                if (cache_->tick_ == std::numeric_limits<std::size_t>::max()) {
                    return make_error(ErrorCode::resource_exhausted,
                                      "DeepSeek expert cache LRU clock overflow",
                                      block_prefix_);
                }
                ++cache_->tick_;
                Entry entry;
                entry.expert_id = expert_id;
                entry.last_use = cache_->tick_;
                const auto cached = cache_->cache_.find(expert_id);
                if (cached != cache_->cache_.end()) {
                    if (cache_->telemetry_.hits ==
                        std::numeric_limits<std::size_t>::max()) {
                        return make_error(ErrorCode::resource_exhausted,
                                          "DeepSeek expert cache hit telemetry overflow",
                                          block_prefix_);
                    }
                    cached->second.last_use = entry.last_use;
                    entry.ready = cached->second.expert;
                    entry.ready->cache_hit = true;
                    entry.ready->logical_bytes_read = 0U;
                    entry.ready->asynchronous_load = false;
                    entry.ready->load_request_sequence = 0U;
                    entry.ready->load_queue_ms = 0.0;
                    entry.ready->load_service_ms = 0.0;
                    entry.ready->load_total_ms = 0.0;
                    entry.ready->load_service_started_at = {};
                    entry.ready->load_completed_at = {};
                    ++cache_->telemetry_.hits;
                } else {
                    ++misses;
                }
                entries_.push_back(std::move(entry));
            }
            if (cache_->cache_.size() >
                cache_->options_.max_cached_experts) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek expert cache exceeds its configured capacity",
                                  block_prefix_);
            }
            while (misses > cache_->options_.max_cached_experts -
                                cache_->cache_.size()) {
                auto evicted = cache_->evict_one(requested_ids_);
                if (!evicted) return evicted.error();
            }
            ++cache_->telemetry_.requests;
            cache_->telemetry_.requested_experts += requested_ids_.size();
            for (const auto expert_id : requested_ids_) {
                if (std::find(cache_->last_selected_ids_.begin(),
                              cache_->last_selected_ids_.end(),
                              expert_id) != cache_->last_selected_ids_.end()) {
                    if (cache_->telemetry_.recurrent_experts ==
                        std::numeric_limits<std::size_t>::max()) {
                        return make_error(ErrorCode::resource_exhausted,
                                          "DeepSeek expert cache recurrent telemetry overflow",
                                          block_prefix_);
                    }
                    ++cache_->telemetry_.recurrent_experts;
                }
            }
            cache_->last_selected_ids_ = requested_ids_;
        }

        for (auto& entry : entries_) {
            if (entry.ready.has_value()) continue;
            const auto& binding = cache_->bindings_.at(entry.expert_id);
            auto residency = model::ResidencyObject::create(
                block_prefix_ + ".experts." +
                    std::to_string(entry.expert_id),
                binding.bundle.bytes);
            if (!residency) return residency.error();
            entry.residency = residency.take_value();
        }

        std::vector<Entry*> pending;
        pending.reserve(entries_.size());
        for (auto& entry : entries_) {
            if (!entry.ready.has_value()) pending.push_back(&entry);
        }
        std::sort(pending.begin(), pending.end(), [](const Entry* left,
                                                     const Entry* right) {
            return left->expert_id < right->expert_id;
        });
        for (auto* entry : pending) {
            const auto& binding = cache_->bindings_.at(entry->expert_id);
            auto submitted = cache_->load_pool_->submit(
                binding.bundle, *entry->residency,
                block_prefix_ + ".experts." +
                    std::to_string(entry->expert_id));
            if (!submitted) return submitted.error();
            entry->future = submitted.take_value();
        }
        return {};
    }

    [[nodiscard]] Result<DeepseekV4ResolvedExpert> acquire(
        std::uint32_t expert_id) override {
        std::scoped_lock batch_lock(mutex_);
        auto entry = std::find_if(entries_.begin(), entries_.end(),
                                  [expert_id](const Entry& candidate) {
                                      return candidate.expert_id == expert_id;
                                  });
        if (!active_ || entry == entries_.end() || entry->acquired) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek asynchronous expert acquisition is invalid",
                              std::to_string(expert_id));
        }
        entry->acquired = true;
        if (entry->ready.has_value()) return *entry->ready;
        if (!entry->future.has_value() || !entry->future->valid() ||
            !entry->residency.has_value()) {
            return make_error(ErrorCode::internal_error,
                              "DeepSeek asynchronous expert future is unavailable",
                              std::to_string(expert_id));
        }

        try {
            auto loaded = entry->future->get();
            entry->future.reset();
            if (!loaded) return loaded.error();
            entry->completed_logical_bytes = loaded.value().telemetry().logical_bytes;
            const auto loaded_read_calls = loaded.value().telemetry().read_calls;
            const auto loaded_bypass_read_calls = loaded.value().telemetry().bypass_read_calls;
            const auto loaded_buffered_read_calls = loaded.value().telemetry().buffered_read_calls;
            const auto& binding = cache_->bindings_.at(expert_id);
            auto bound = bind_loaded_expert(binding, loaded.value(), true);
            if (!bound) return bound.error();
            auto expert = bound.take_value();

            std::scoped_lock lock(cache_->mutex_);
            if (!cache_->async_batch_active_) {
                return make_error(ErrorCode::internal_error,
                                  "DeepSeek asynchronous expert batch lost cache ownership",
                                  std::to_string(expert_id));
            }
            auto logical_total = checked_add(
                cache_->telemetry_.logical_bytes_read,
                expert.logical_bytes_read);
            if (!logical_total) return logical_total.error();
            auto resident_total = checked_add(
                cache_->telemetry_.resident_bytes,
                binding.bundle.bytes);
            if (!resident_total) return resident_total.error();
            if (loaded_read_calls > std::numeric_limits<std::size_t>::max() -
                                       cache_->telemetry_.read_calls) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek expert read-call telemetry overflow",
                                  block_prefix_);
            }
            if (cache_->telemetry_.misses ==
                std::numeric_limits<std::size_t>::max()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek expert cache miss telemetry overflow",
                                  block_prefix_);
            }
            auto inserted = cache_->cache_.emplace(
                expert_id,
                CacheEntry{expert, *entry->residency, entry->last_use});
            if (!inserted.second) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek asynchronous expert duplicated a cache entry",
                                  std::to_string(expert_id));
            }
            entry->installed = true;
            ++cache_->telemetry_.misses;
            cache_->telemetry_.read_calls += loaded_read_calls;
            cache_->telemetry_.bypass_read_calls += loaded_bypass_read_calls;
            cache_->telemetry_.buffered_read_calls += loaded_buffered_read_calls;
            cache_->telemetry_.logical_bytes_read = logical_total.value();
            cache_->telemetry_.resident_bytes = resident_total.value();
            return expert;
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to retain an asynchronous DeepSeek expert",
                              std::to_string(expert_id));
        } catch (const std::future_error& exception) {
            entry->future.reset();
            return make_error(ErrorCode::internal_error,
                              "DeepSeek asynchronous expert future failed",
                              exception.what());
        } catch (const std::exception& exception) {
            return make_error(ErrorCode::internal_error,
                              "DeepSeek asynchronous expert acquisition failed",
                              exception.what());
        } catch (...) {
            return make_error(ErrorCode::internal_error,
                              "DeepSeek asynchronous expert acquisition failed",
                              std::to_string(expert_id));
        }
    }

private:
    struct Entry {
        std::uint32_t expert_id{0U};
        std::size_t last_use{0U};
        std::optional<DeepseekV4ResolvedExpert> ready;
        std::optional<model::ResidencyObject> residency;
        std::optional<std::future<Result<model::LoadedExpert>>> future;
        ByteCount completed_logical_bytes{0U};
        bool acquired{false};
        bool installed{false};
    };

    void cleanup() noexcept {
        if (!active_) return;
        ByteCount abandoned_bytes = 0U;
        std::size_t abandoned_loads = 0U;
        bool telemetry_overflow = false;
        for (auto& entry : entries_) {
            if (entry.future.has_value() && entry.future->valid()) {
                try {
                    {
                        auto ignored = entry.future->get();
                        if (ignored) {
                            entry.completed_logical_bytes =
                                ignored.value().telemetry().logical_bytes;
                        }
                    }
                } catch (...) {
                    // The pool contains worker failures; cleanup still releases residency below.
                }
                entry.future.reset();
            }
            if (entry.residency.has_value() && !entry.installed) {
                try {
                    auto snapshot = entry.residency->snapshot();
                    if (snapshot &&
                        snapshot.value().state == model::ResidencyState::ram_ready) {
                        static_cast<void>(entry.residency->evict_ram());
                    }
                } catch (...) {
                    // Residency control releases any remaining reservations on destruction.
                }
            }
            if (!entry.installed && entry.completed_logical_bytes != 0U) {
                auto total = checked_add(abandoned_bytes,
                                         entry.completed_logical_bytes);
                if (!total || abandoned_loads ==
                                  std::numeric_limits<std::size_t>::max()) {
                    telemetry_overflow = true;
                } else {
                    abandoned_bytes = total.value();
                    ++abandoned_loads;
                }
            }
        }
        entries_.clear();
        {
            std::scoped_lock lock(cache_->mutex_);
            auto logical_total = checked_add(
                cache_->telemetry_.logical_bytes_read, abandoned_bytes);
            auto wasted_total = checked_add(
                cache_->telemetry_.wasted_prefetch_bytes, abandoned_bytes);
            if (telemetry_overflow || !logical_total || !wasted_total ||
                abandoned_loads > std::numeric_limits<std::size_t>::max() -
                                      cache_->telemetry_.abandoned_loads) {
                cache_->telemetry_.telemetry_overflow = true;
            } else {
                cache_->telemetry_.logical_bytes_read = logical_total.value();
                cache_->telemetry_.wasted_prefetch_bytes = wasted_total.value();
                cache_->telemetry_.abandoned_loads += abandoned_loads;
            }
            cache_->async_batch_active_ = false;
        }
        active_ = false;
    }

    std::shared_ptr<DeepseekV4BoundedExpertCache> cache_;
    std::string block_prefix_;
    std::vector<std::uint32_t> requested_ids_;
    std::vector<Entry> entries_;
    std::mutex mutex_;
    bool active_{false};
};

Result<std::unique_ptr<DeepseekV4ExpertBatch>>
DeepseekV4BoundedExpertCache::begin_resolve(
    std::string_view block_prefix,
    std::span<const std::uint32_t> unique_expert_ids) {
    try {
        auto batch = std::make_unique<AsyncBatch>(
            shared_from_this(), std::string(block_prefix), unique_expert_ids);
        auto prepared = batch->prepare();
        if (!prepared) return prepared.error();
        return std::unique_ptr<DeepseekV4ExpertBatch>(std::move(batch));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate a DeepSeek asynchronous expert batch",
                          std::string(block_prefix));
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "Unable to create a DeepSeek asynchronous expert batch",
                          exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "Unable to create a DeepSeek asynchronous expert batch",
                          std::string(block_prefix));
    }
}

Result<void> DeepseekV4BoundedExpertCache::evict_one(
    std::span<const std::uint32_t> protected_ids) {
    // The cache owns exactly one descriptor for each of W1/W2/W3, all backed by the same
    // ResidentByteBuffer. A larger shared_ptr count means an inference call still retains the
    // indivisible bundle outside the cache, so that entry must not be evicted yet.
    constexpr long cache_owned_buffer_references = 3L;
    auto victim = cache_.end();
    for (auto iterator = cache_.begin(); iterator != cache_.end(); ++iterator) {
        const bool protected_expert =
            std::find(protected_ids.begin(), protected_ids.end(), iterator->first) !=
            protected_ids.end();
        const auto& expert = iterator->second.expert;
        if (expert.w1.resident == nullptr ||
            expert.w1.resident != expert.w2.resident ||
            expert.w1.resident != expert.w3.resident) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek expert cache entry lost its indivisible backing",
                              std::to_string(iterator->first));
        }
        const bool actively_used =
            expert.w1.resident.use_count() != cache_owned_buffer_references;
        if (!protected_expert && !actively_used &&
            (victim == cache_.end() ||
             iterator->second.last_use < victim->second.last_use)) {
            victim = iterator;
        }
    }
    if (victim == cache_.end()) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek expert cache has no evictable entry",
                          options_.block_prefix);
    }
    if (telemetry_.evictions == std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek expert cache eviction telemetry overflow",
                          options_.block_prefix);
    }
    const ByteCount bytes = victim->second.expert.resident_bytes;
    auto remaining = checked_sub(telemetry_.resident_bytes, bytes);
    if (!remaining) return remaining.error();
    auto residency = victim->second.residency;
    cache_.erase(victim);
    telemetry_.resident_bytes = remaining.value();
    auto evicted = residency.evict_ram();
    if (!evicted) return evicted.error();
    ++telemetry_.evictions;
    return {};
}

Result<std::vector<DeepseekV4ResolvedExpert>> DeepseekV4BoundedExpertCache::resolve(
    std::string_view block_prefix,
    std::span<const std::uint32_t> unique_expert_ids) {
    std::scoped_lock lock(mutex_);
    if (async_batch_active_) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek expert cache has an active asynchronous batch",
                          std::string(block_prefix));
    }
    if (telemetry_.telemetry_overflow) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek expert cache telemetry overflowed",
                          std::string(block_prefix));
    }
    if (block_prefix != options_.block_prefix || unique_expert_ids.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek selected-expert request does not match its cache binding",
                          std::string(block_prefix));
    }
    if (unique_expert_ids.size() > options_.max_experts_per_request) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek selected-expert request exceeds its cache binding",
                          std::string(block_prefix));
    }
    for (std::size_t index = 0U; index < unique_expert_ids.size(); ++index) {
        if (!bindings_.contains(unique_expert_ids[index]) ||
            std::find(unique_expert_ids.begin(),
                      unique_expert_ids.begin() + static_cast<std::ptrdiff_t>(index),
                      unique_expert_ids[index]) !=
                unique_expert_ids.begin() + static_cast<std::ptrdiff_t>(index)) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek selected-expert request contains an unknown or duplicate identity",
                              std::to_string(unique_expert_ids[index]));
        }
    }

    try {
        std::vector<DeepseekV4ResolvedExpert> result;
        result.reserve(unique_expert_ids.size());
        if (telemetry_.requests == std::numeric_limits<std::size_t>::max() ||
            unique_expert_ids.size() >
                std::numeric_limits<std::size_t>::max() -
                    telemetry_.requested_experts) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek expert cache request telemetry overflow",
                              options_.block_prefix);
        }
        ++telemetry_.requests;
        telemetry_.requested_experts += unique_expert_ids.size();
        for (const auto expert_id : unique_expert_ids) {
            if (tick_ == std::numeric_limits<std::size_t>::max()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek expert cache LRU clock overflow",
                                  options_.block_prefix);
            }
            ++tick_;
            auto cached = cache_.find(expert_id);
            if (cached != cache_.end()) {
                cached->second.last_use = tick_;
                auto expert = cached->second.expert;
                expert.cache_hit = true;
                expert.logical_bytes_read = 0U;
                expert.asynchronous_load = false;
                expert.load_request_sequence = 0U;
                expert.load_queue_ms = 0.0;
                expert.load_service_ms = 0.0;
                expert.load_total_ms = 0.0;
                expert.load_service_started_at = {};
                expert.load_completed_at = {};
                if (telemetry_.hits == std::numeric_limits<std::size_t>::max()) {
                    return make_error(ErrorCode::resource_exhausted,
                                      "DeepSeek expert cache hit telemetry overflow",
                                      options_.block_prefix);
                }
                result.push_back(std::move(expert));
                ++telemetry_.hits;
                continue;
            }
            if (cache_.size() == options_.max_cached_experts) {
                auto evicted = evict_one(unique_expert_ids);
                if (!evicted) return evicted.error();
            }

            const auto& binding = bindings_.at(expert_id);
            auto residency = model::ResidencyObject::create(
                options_.block_prefix + ".experts." + std::to_string(expert_id),
                binding.bundle.bytes);
            if (!residency) return residency.error();
            auto loaded = model::load_expert_bundle(
                binding.bundle, residency.value(), governor_, options_.load);
            if (!loaded) return loaded.error();
            const auto loaded_read_calls = loaded.value().telemetry().read_calls;
            const auto loaded_bypass_read_calls = loaded.value().telemetry().bypass_read_calls;
            const auto loaded_buffered_read_calls = loaded.value().telemetry().buffered_read_calls;

            auto bound = bind_loaded_expert(binding, loaded.value(), false);
            if (!bound) return bound.error();
            auto expert = bound.take_value();
            const ByteCount logical_bytes = expert.logical_bytes_read;
            auto logical_total =
                checked_add(telemetry_.logical_bytes_read, logical_bytes);
            if (!logical_total) return logical_total.error();
            auto resident_total =
                checked_add(telemetry_.resident_bytes, binding.bundle.bytes);
            if (!resident_total) return resident_total.error();
            if (loaded_read_calls > std::numeric_limits<std::size_t>::max() - telemetry_.read_calls) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek expert read-call telemetry overflow",
                                  options_.block_prefix);
            }
            if (telemetry_.misses == std::numeric_limits<std::size_t>::max()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek expert cache miss telemetry overflow",
                                  options_.block_prefix);
            }
            cache_.emplace(
                expert_id,
                CacheEntry{expert, residency.take_value(), tick_});
            result.push_back(std::move(expert));
            ++telemetry_.misses;
            telemetry_.read_calls += loaded_read_calls;
            telemetry_.bypass_read_calls += loaded_bypass_read_calls;
            telemetry_.buffered_read_calls += loaded_buffered_read_calls;
            telemetry_.logical_bytes_read = logical_total.value();
            telemetry_.resident_bytes = resident_total.value();
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate selected DeepSeek expert state",
                          options_.block_prefix);
    }
}

Result<void> DeepseekV4BoundedExpertCache::clear() {
    std::scoped_lock lock(mutex_);
    if (async_batch_active_) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek expert cache cannot clear an active asynchronous batch",
                          options_.block_prefix);
    }
    while (!cache_.empty()) {
        std::array<std::uint32_t, 0U> none{};
        auto evicted = evict_one(none);
        if (!evicted) return evicted.error();
    }
    return {};
}

Result<void> DeepseekV4BoundedExpertCache::retain_last_selected() {
    std::scoped_lock lock(mutex_);
    if (async_batch_active_) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek expert cache cannot retain during an active asynchronous batch",
                          options_.block_prefix);
    }
    while (true) {
        bool extra = false;
        for (const auto& [expert_id, entry] : cache_) {
            (void)entry;
            if (std::find(last_selected_ids_.begin(), last_selected_ids_.end(), expert_id) ==
                last_selected_ids_.end()) {
                extra = true;
                break;
            }
        }
        if (!extra) break;
        auto evicted = evict_one(last_selected_ids_);
        if (!evicted) return evicted.error();
    }
    return {};
}

}  // namespace vastloom::inference
