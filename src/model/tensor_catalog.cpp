#include "vastloom/model/tensor_catalog.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <utility>

namespace vastloom::model {
namespace {

using ExpertKey = std::tuple<bool, std::uint32_t, std::uint32_t>;

Result<ByteCount> add_bytes(ByteCount left, ByteCount right, std::string_view context) {
    if (right > std::numeric_limits<ByteCount>::max() - left) {
        return make_error(ErrorCode::invalid_format,
                          "Logical tensor catalog byte count overflows",
                          std::string(context));
    }
    return left + right;
}

bool source_less(const TensorSourceRange& left, const TensorSourceRange& right) {
    if (left.shard != right.shard) return left.shard < right.shard;
    return left.offset < right.offset;
}

Result<void> finalize_catalog(LogicalTensorCatalog& catalog,
                              std::size_t max_tensors,
                              std::size_t max_experts) {
    if (catalog.model_id.empty() || catalog.revision.empty() || max_tensors == 0U || max_experts == 0U) {
        return make_error(ErrorCode::invalid_argument, "Invalid logical tensor catalog identity or limits");
    }
    if (catalog.tensors.empty() || catalog.tensors.size() > max_tensors) {
        return make_error(ErrorCode::resource_exhausted,
                          "Logical tensor catalog is empty or exceeds the configured tensor limit",
                          catalog.model_id);
    }

    std::set<std::string, std::less<>> names;
    std::map<ExpertKey, ExpertBundle> bundles;
    std::map<std::filesystem::path, std::vector<TensorSourceRange>> shard_ranges;
    ByteCount tensor_bytes = 0U;
    ByteCount always_active_bytes = 0U;
    ByteCount routed_expert_bytes = 0U;

    for (std::size_t index = 0; index < catalog.tensors.size(); ++index) {
        const auto& tensor = catalog.tensors[index];
        if (tensor.name.empty() || !names.insert(tensor.name).second) {
            return make_error(ErrorCode::invalid_format,
                              "Logical tensor catalog contains an empty or duplicate tensor name",
                              tensor.name);
        }
        if (tensor.source.shard.empty() || tensor.source.bytes == 0U ||
            tensor.source.offset > std::numeric_limits<ByteCount>::max() - tensor.source.bytes) {
            return make_error(ErrorCode::invalid_format,
                              "Logical tensor source range is empty or overflows",
                              tensor.name);
        }
        if (tensor.dtype == TensorDType::quantized && !tensor.ggml_type.has_value()) {
            return make_error(ErrorCode::invalid_format,
                              "Quantized logical tensor is missing its GGML storage type",
                              tensor.name);
        }

        auto total = add_bytes(tensor_bytes, tensor.source.bytes, tensor.name);
        if (!total) return total.error();
        tensor_bytes = total.value();
        shard_ranges[tensor.source.shard].push_back(tensor.source);

        if (tensor.role != TensorRole::routed_expert) {
            auto active = add_bytes(always_active_bytes, tensor.source.bytes, tensor.name);
            if (!active) return active.error();
            always_active_bytes = active.value();
            continue;
        }
        if (!tensor.layer.has_value() || !tensor.expert.has_value()) {
            return make_error(ErrorCode::invalid_format,
                              "Routed expert tensor is missing its logical layer or expert identity",
                              tensor.name);
        }

        auto routed = add_bytes(routed_expert_bytes, tensor.source.bytes, tensor.name);
        if (!routed) return routed.error();
        routed_expert_bytes = routed.value();

        const ExpertKey key{tensor.speculative, *tensor.layer, *tensor.expert};
        auto [iterator, inserted] = bundles.try_emplace(
            key, ExpertBundle{*tensor.layer, *tensor.expert, tensor.speculative, 0U, {}, {}, {}});
        if (inserted && bundles.size() > max_experts) {
            return make_error(ErrorCode::resource_exhausted,
                              "Logical tensor catalog exceeds the configured expert limit",
                              catalog.model_id);
        }
        auto bundle_total = add_bytes(iterator->second.bytes, tensor.source.bytes, tensor.name);
        if (!bundle_total) return bundle_total.error();
        iterator->second.bytes = bundle_total.value();
        iterator->second.tensor_indices.push_back(index);
        iterator->second.read_ranges.push_back(tensor.source);
    }

    for (auto& [shard, ranges] : shard_ranges) {
        std::sort(ranges.begin(), ranges.end(), source_less);
        for (std::size_t index = 1U; index < ranges.size(); ++index) {
            const auto previous_end = ranges[index - 1U].offset + ranges[index - 1U].bytes;
            if (ranges[index].offset < previous_end) {
                return make_error(ErrorCode::invalid_format,
                                  "Logical tensor source ranges overlap",
                                  shard.generic_string());
            }
        }
    }

    std::vector<ExpertBundle> finalized_bundles;
    finalized_bundles.reserve(bundles.size());
    std::size_t read_range_count = 0U;
    for (auto& [key, bundle] : bundles) {
        static_cast<void>(key);
        std::sort(bundle.read_ranges.begin(), bundle.read_ranges.end(), source_less);
        std::vector<TensorSourceRange> merged;
        merged.reserve(bundle.read_ranges.size());
        for (const auto& range : bundle.read_ranges) {
            if (!merged.empty() && merged.back().shard == range.shard &&
                merged.back().offset + merged.back().bytes == range.offset) {
                auto combined = add_bytes(merged.back().bytes, range.bytes, range.shard.generic_string());
                if (!combined) return combined.error();
                merged.back().bytes = combined.value();
            } else {
                merged.push_back(range);
            }
        }
        if (read_range_count > std::numeric_limits<std::size_t>::max() - merged.size()) {
            return make_error(ErrorCode::invalid_format,
                              "Logical tensor catalog read range count overflows",
                              catalog.model_id);
        }
        read_range_count += merged.size();
        ByteCount packed_base = 0U;
        bundle.tensor_slices.reserve(bundle.tensor_indices.size());
        for (const auto tensor_index : bundle.tensor_indices) {
            const auto& source = catalog.tensors[tensor_index].source;
            ByteCount range_base = 0U;
            bool found = false;
            for (const auto& range : merged) {
                if (source.shard == range.shard && source.offset >= range.offset &&
                    source.offset - range.offset <= range.bytes &&
                    source.bytes <= range.bytes - (source.offset - range.offset)) {
                    bundle.tensor_slices.push_back(ExpertTensorSlice{
                        tensor_index, range_base + (source.offset - range.offset), source.bytes});
                    found = true;
                    break;
                }
                range_base += range.bytes;
            }
            if (!found) {
                return make_error(ErrorCode::internal_error,
                                  "Logical expert tensor is outside its coalesced read ranges",
                                  catalog.tensors[tensor_index].name);
            }
        }
        for (const auto& range : merged) packed_base += range.bytes;
        if (packed_base != bundle.bytes || bundle.tensor_slices.size() != bundle.tensor_indices.size()) {
            return make_error(ErrorCode::internal_error,
                              "Logical expert packed layout differs from its tensor identity",
                              catalog.model_id);
        }
        bundle.read_ranges = std::move(merged);
        finalized_bundles.push_back(std::move(bundle));
    }

    catalog.experts = std::move(finalized_bundles);
    catalog.tensor_bytes = tensor_bytes;
    catalog.always_active_bytes = always_active_bytes;
    catalog.routed_expert_bytes = routed_expert_bytes;
    catalog.expert_read_range_count = read_range_count;
    return {};
}

}  // namespace

Result<void> finalize_logical_tensor_catalog(LogicalTensorCatalog& catalog,
                                             std::size_t max_tensors,
                                             std::size_t max_experts) {
    try {
        return finalize_catalog(catalog, max_tensors, max_experts);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded logical tensor catalog",
                          catalog.model_id);
    }
}

std::string_view to_string(TensorRole role) noexcept {
    switch (role) {
        case TensorRole::embedding: return "embedding";
        case TensorRole::output_head: return "output_head";
        case TensorRole::normalization: return "normalization";
        case TensorRole::attention: return "attention";
        case TensorRole::router: return "router";
        case TensorRole::shared_expert: return "shared_expert";
        case TensorRole::routed_expert: return "routed_expert";
        case TensorRole::hyper_connection: return "hyper_connection";
        case TensorRole::speculative: return "speculative";
        case TensorRole::other: return "other";
    }
    return "other";
}

}  // namespace vastloom::model
