#include "vastloom/inference/deepseek_v4_expert_provider.hpp"

#include <algorithm>
#include <limits>
#include <new>
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
    }
}

DeepseekV4BoundedExpertCache::DeepseekV4BoundedExpertCache(
    std::map<std::uint32_t, DeepseekV4ExpertBundleBinding> bindings,
    safety::ResourceGovernor& governor,
    DeepseekV4ExpertCacheOptions options)
    : bindings_(std::move(bindings)), governor_(governor), options_(std::move(options)) {}

DeepseekV4BoundedExpertCache::~DeepseekV4BoundedExpertCache() {
    static_cast<void>(clear());
}

DeepseekV4ExpertProviderBinding DeepseekV4BoundedExpertCache::provider_binding() {
    auto self = shared_from_this();
    return DeepseekV4ExpertProviderBinding{
        options_.max_experts_per_request,
        [self = std::move(self)](
            std::string_view block_prefix,
            std::span<const std::uint32_t> unique_expert_ids) {
            return self->resolve(block_prefix, unique_expert_ids);
        }};
}

DeepseekV4ExpertCacheSnapshot DeepseekV4BoundedExpertCache::snapshot() const {
    std::scoped_lock lock(mutex_);
    auto snapshot = telemetry_;
    snapshot.cached_experts = cache_.size();
    return snapshot;
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

            DeepseekV4ResolvedExpert expert;
            expert.expert_id = expert_id;
            expert.resident_bytes = binding.bundle.bytes;
            expert.logical_bytes_read = loaded.value().telemetry().logical_bytes;
            for (const auto& matrix : binding.matrices) {
                auto packed = deepseek_v4_bind_packed_matrix(
                    loaded.value(), matrix.catalog_tensor_index, matrix.ggml_type,
                    matrix.rows, matrix.logical_columns, matrix.storage_columns);
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
                                  std::to_string(expert_id));
            }
            const ByteCount logical_bytes = expert.logical_bytes_read;
            auto logical_total =
                checked_add(telemetry_.logical_bytes_read, logical_bytes);
            if (!logical_total) return logical_total.error();
            auto resident_total =
                checked_add(telemetry_.resident_bytes, binding.bundle.bytes);
            if (!resident_total) return resident_total.error();
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
    while (!cache_.empty()) {
        std::array<std::uint32_t, 0U> none{};
        auto evicted = evict_one(none);
        if (!evicted) return evicted.error();
    }
    return {};
}

}  // namespace vastloom::inference
