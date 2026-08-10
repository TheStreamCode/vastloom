#include "vastloom/inference/deepseek_v4_gguf.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

#include "vastloom/model/layer_streaming.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

Result<ByteCount> add_probe_bytes(ByteCount current,
                                  ByteCount added,
                                  std::string_view context) {
    if (added > std::numeric_limits<ByteCount>::max() - current) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek layer probe byte total overflows",
                          std::string(context));
    }
    return current + added;
}

std::string source_name(const std::filesystem::path& path) {
    return path.filename().generic_string();
}

void append_source_json(std::ostringstream& output, const model::TensorSourceRange& source) {
    output << "{\"object\":\"" << util::json_escape(source_name(source.shard))
           << "\",\"offset\":" << source.offset << ",\"bytes\":" << source.bytes << '}';
}

void append_shape_json(std::ostringstream& output, const std::vector<std::uint64_t>& shape) {
    output << '[';
    for (std::size_t index = 0U; index < shape.size(); ++index) {
        if (index != 0U) output << ',';
        output << shape[index];
    }
    output << ']';
}

}  // namespace

Result<DeepseekV4LayerProbePlan> build_deepseek_v4_layer_probe_plan(
    const DeepseekV4GgufCatalogReport& report,
    std::uint32_t layer,
    std::span<const std::uint32_t> selected_experts,
    ByteCount tile_limit_bytes,
    std::size_t max_ranges) {
    try {
        if (report.catalog.model_id.empty() || report.catalog.revision.empty() ||
            report.layers == 0U || report.experts_per_layer == 0U || report.top_k == 0U ||
            report.top_k > report.experts_per_layer || report.top_k > 64U ||
            layer >= report.layers ||
            tile_limit_bytes == 0U || max_ranges == 0U ||
            selected_experts.size() != report.top_k) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek layer probe identity, bounds or top-k selection is invalid",
                              report.catalog.model_id);
        }

        std::set<std::uint32_t> selected_ids;
        for (const auto expert : selected_experts) {
            if (expert >= report.experts_per_layer || !selected_ids.insert(expert).second) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek layer probe experts must be unique and in range",
                                  std::to_string(expert));
            }
        }

        auto trunk = model::build_layer_streaming_plan(
            report.catalog, layer, tile_limit_bytes, 4'096U, max_ranges);
        if (!trunk) return trunk.error();

        DeepseekV4LayerProbePlan plan;
        plan.model_id = report.catalog.model_id;
        plan.revision = report.catalog.revision;
        plan.metadata_only = report.catalog.metadata_only;
        plan.imported_file_hashes_verified = report.imported_file_hashes_verified;
        plan.full_payload_present = report.full_payload_present;
        plan.executable_source_binding_verified = report.executable_source_binding_verified;
        plan.layer = layer;
        plan.expected_top_k = report.top_k;
        plan.tile_limit_bytes = tile_limit_bytes;
        plan.trunk_bytes = trunk.value().trunk_tensor_bytes;
        plan.trunk_range_count = trunk.value().tiles.size();
        plan.selected_experts.assign(selected_experts.begin(), selected_experts.end());
        plan.trunk_ranges.reserve(trunk.value().tiles.size());
        plan.experts.reserve(selected_experts.size());

        std::set<std::filesystem::path> sources;
        for (const auto& tile : trunk.value().tiles) {
            if (tile.tensor_index >= report.catalog.tensors.size()) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek layer probe trunk identity is outside the catalog");
            }
            const auto& tensor = report.catalog.tensors[tile.tensor_index];
            plan.trunk_ranges.push_back(DeepseekV4ProbeTrunkRange{
                tensor.name,
                tensor.role,
                tile.tensor_offset,
                tile.source,
                tile.final_tile,
                tensor.ggml_type,
            });
            sources.insert(tile.source.shard);
            plan.maximum_read_range_bytes =
                std::max(plan.maximum_read_range_bytes, tile.source.bytes);
        }

        for (const auto selected_expert : selected_experts) {
            const auto found = std::find_if(
                report.catalog.experts.begin(), report.catalog.experts.end(),
                [layer, selected_expert](const model::ExpertBundle& bundle) {
                    return !bundle.speculative && bundle.layer == layer &&
                           bundle.expert == selected_expert;
                });
            if (found == report.catalog.experts.end() || found->bytes == 0U ||
                found->read_ranges.empty() || found->tensor_indices.empty() ||
                found->tensor_slices.size() != found->tensor_indices.size()) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek layer probe selected expert bundle is missing or incomplete",
                                  std::to_string(selected_expert));
            }
            if (found->read_ranges.size() > max_ranges - plan.trunk_range_count ||
                plan.selected_expert_range_count >
                    max_ranges - plan.trunk_range_count - found->read_ranges.size()) {
                return make_error(ErrorCode::resource_exhausted,
                                  "DeepSeek layer probe range count exceeds the configured bound",
                                  std::to_string(max_ranges));
            }

            DeepseekV4ProbeExpert expert;
            expert.expert = selected_expert;
            expert.bytes = found->bytes;
            expert.read_ranges = found->read_ranges;
            expert.tensors.reserve(found->tensor_slices.size());
            for (const auto& range : found->read_ranges) {
                if (range.shard.empty() || range.bytes == 0U ||
                    range.offset > std::numeric_limits<ByteCount>::max() - range.bytes) {
                    return make_error(ErrorCode::integrity_error,
                                      "DeepSeek layer probe selected expert range is invalid",
                                      std::to_string(selected_expert));
                }
                sources.insert(range.shard);
                plan.maximum_read_range_bytes =
                    std::max(plan.maximum_read_range_bytes, range.bytes);
            }
            for (const auto& slice : found->tensor_slices) {
                if (slice.tensor_index >= report.catalog.tensors.size()) {
                    return make_error(ErrorCode::integrity_error,
                                      "DeepSeek layer probe expert tensor is outside the catalog",
                                      std::to_string(selected_expert));
                }
                const auto& tensor = report.catalog.tensors[slice.tensor_index];
                if (tensor.role != model::TensorRole::routed_expert ||
                    tensor.layer != layer || tensor.expert != selected_expert ||
                    slice.bytes != tensor.source.bytes ||
                    slice.packed_offset > found->bytes ||
                    slice.bytes > found->bytes - slice.packed_offset) {
                    return make_error(ErrorCode::integrity_error,
                                      "DeepSeek layer probe expert tensor identity is inconsistent",
                                      tensor.name);
                }
                expert.tensors.push_back(DeepseekV4ProbeTensorSlice{
                    tensor.name,
                    tensor.shape,
                    slice.packed_offset,
                    slice.bytes,
                    tensor.source,
                    tensor.ggml_type,
                });
            }
            auto expert_total = add_probe_bytes(
                plan.selected_expert_bytes, found->bytes, "selected experts");
            if (!expert_total) return expert_total.error();
            plan.selected_expert_bytes = expert_total.value();
            plan.minimum_expert_tier_bytes =
                std::max(plan.minimum_expert_tier_bytes, found->bytes);
            plan.selected_expert_range_count += found->read_ranges.size();
            plan.experts.push_back(std::move(expert));
        }

        if (plan.trunk_range_count > max_ranges - plan.selected_expert_range_count) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek layer probe range count exceeds the configured bound",
                              std::to_string(max_ranges));
        }
        plan.total_range_count = plan.trunk_range_count + plan.selected_expert_range_count;
        auto total = add_probe_bytes(plan.trunk_bytes,
                                     plan.selected_expert_bytes,
                                     "complete layer probe");
        if (!total) return total.error();
        plan.total_payload_bytes = total.value();
        plan.unique_source_objects = sources.size();
        return plan;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded DeepSeek layer probe plan",
                          report.catalog.model_id);
    }
}

std::string to_json(const DeepseekV4LayerProbePlan& plan) {
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"vastloom.deepseek_v4.layer_probe_plan.v1\""
           << ",\"model_id\":\"" << util::json_escape(plan.model_id)
           << "\",\"revision\":\"" << util::json_escape(plan.revision)
           << "\",\"plan_only\":true,\"execution_supported\":false"
           << ",\"layer\":" << plan.layer
           << ",\"expected_top_k\":" << plan.expected_top_k
           << ",\"selection_source\":\"caller-supplied-unverified\""
           << ",\"routing_authority_verified\":false"
           << ",\"payload_access\":{\"metadata_only\":"
           << (plan.metadata_only ? "true" : "false")
           << ",\"imported_file_hashes_verified\":"
           << (plan.imported_file_hashes_verified ? "true" : "false")
           << ",\"full_payload_present\":"
           << (plan.full_payload_present ? "true" : "false")
           << ",\"executable_source_binding_verified\":"
           << (plan.executable_source_binding_verified ? "true" : "false") << '}'
           << ",\"tile_limit_bytes\":" << plan.tile_limit_bytes
           << ",\"trunk_bytes\":" << plan.trunk_bytes
           << ",\"selected_expert_bytes\":" << plan.selected_expert_bytes
           << ",\"total_payload_bytes\":" << plan.total_payload_bytes
           << ",\"maximum_read_range_bytes\":" << plan.maximum_read_range_bytes
           << ",\"minimum_expert_tier_bytes\":" << plan.minimum_expert_tier_bytes
           << ",\"trunk_range_count\":" << plan.trunk_range_count
           << ",\"selected_expert_range_count\":" << plan.selected_expert_range_count
           << ",\"total_range_count\":" << plan.total_range_count
           << ",\"unique_source_objects\":" << plan.unique_source_objects
           << ",\"selected_experts\":[";
    for (std::size_t index = 0U; index < plan.selected_experts.size(); ++index) {
        if (index != 0U) output << ',';
        output << plan.selected_experts[index];
    }
    output << "],\"trunk_ranges\":[";
    for (std::size_t index = 0U; index < plan.trunk_ranges.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& range = plan.trunk_ranges[index];
        output << "{\"tensor\":\"" << util::json_escape(range.tensor_name)
               << "\",\"role\":\"" << model::to_string(range.role)
               << "\",\"tensor_offset\":" << range.tensor_offset
               << ",\"final_tile\":" << (range.final_tile ? "true" : "false")
               << ",\"ggml_type\":";
        if (range.ggml_type.has_value()) output << *range.ggml_type;
        else output << "null";
        output << ",\"source\":";
        append_source_json(output, range.source);
        output << '}';
    }
    output << "],\"experts\":[";
    for (std::size_t expert_index = 0U; expert_index < plan.experts.size(); ++expert_index) {
        if (expert_index != 0U) output << ',';
        const auto& expert = plan.experts[expert_index];
        output << "{\"expert\":" << expert.expert << ",\"bytes\":" << expert.bytes
               << ",\"read_ranges\":[";
        for (std::size_t range_index = 0U; range_index < expert.read_ranges.size(); ++range_index) {
            if (range_index != 0U) output << ',';
            append_source_json(output, expert.read_ranges[range_index]);
        }
        output << "],\"tensors\":[";
        for (std::size_t tensor_index = 0U; tensor_index < expert.tensors.size(); ++tensor_index) {
            if (tensor_index != 0U) output << ',';
            const auto& tensor = expert.tensors[tensor_index];
            output << "{\"tensor\":\"" << util::json_escape(tensor.tensor_name)
                   << "\",\"shape\":";
            append_shape_json(output, tensor.shape);
            output << ",\"packed_offset\":" << tensor.packed_offset
                   << ",\"bytes\":" << tensor.bytes << ",\"ggml_type\":";
            if (tensor.ggml_type.has_value()) output << *tensor.ggml_type;
            else output << "null";
            output << ",\"source\":";
            append_source_json(output, tensor.source);
            output << '}';
        }
        output << "]}";
    }
    output << "]}";
    return output.str();
}

std::string to_text(const DeepseekV4LayerProbePlan& plan) {
    std::ostringstream output;
    output << "DeepSeek layer probe: " << plan.model_id << '\n'
           << "Revision:             " << plan.revision << '\n'
           << "Mode:                 plan only; no payload execution" << '\n'
           << "Layer / top-k:        " << plan.layer << " / " << plan.expected_top_k << '\n'
           << "Trunk ranges:         " << plan.trunk_range_count << " ("
           << plan.trunk_bytes << " bytes)" << '\n'
           << "Selected experts:     " << plan.experts.size() << " ("
           << plan.selected_expert_range_count << " ranges, "
           << plan.selected_expert_bytes << " bytes)" << '\n'
           << "Complete range set:   " << plan.total_range_count << " ranges, "
           << plan.total_payload_bytes << " bytes across " << plan.unique_source_objects
           << " source object(s)" << '\n'
           << "Single-tier expert:   " << plan.minimum_expert_tier_bytes << " bytes" << '\n'
           << "Metadata only:        " << (plan.metadata_only ? "yes" : "no") << '\n'
           << "Executable binding:   "
           << (plan.executable_source_binding_verified ? "verified" : "unavailable") << '\n'
           << "Routing contract:     caller must supply authoritative output; planner cannot prove it";
    return output.str();
}

}  // namespace vastloom::inference
