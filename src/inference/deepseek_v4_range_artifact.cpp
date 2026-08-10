#include "vastloom/inference/deepseek_v4_range_artifact.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <new>
#include <sstream>
#include <string_view>

#include "vastloom/util/sha256.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

bool same_source(const model::TensorSourceRange& left,
                 const model::TensorSourceRange& right) {
    return left.shard == right.shard && left.offset == right.offset &&
           left.bytes == right.bytes;
}

bool same_trunk_range(const DeepseekV4ProbeTrunkRange& left,
                      const DeepseekV4ProbeTrunkRange& right) {
    return left.tensor_name == right.tensor_name && left.role == right.role &&
           left.tensor_offset == right.tensor_offset && left.final_tile == right.final_tile &&
           left.ggml_type == right.ggml_type && same_source(left.source, right.source);
}

std::string expected_label(std::size_t index, std::string_view tensor_name) {
    return "trunk[" + std::to_string(index) + "] " + std::string(tensor_name);
}

}  // namespace

Result<DeepseekV4TrunkArtifactBinding> bind_deepseek_v4_trunk_range_artifact(
    const model::RangeArtifact& artifact,
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4LayerProbePlan& plan,
    const DeepseekV4GgufCheckpointSpec& checkpoint_spec) {
    try {
        if (!artifact.full_hash_verified || artifact.complete_checkpoint_verified ||
            artifact.execution_supported || artifact.scope != "trunk" ||
            artifact.routing_authority_verified ||
            artifact.source_plan_format != "vastloom.deepseek_v4.layer_probe_plan.v1" ||
            !util::is_sha256_hex(artifact.source_plan_sha256)) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek trunk binding requires a fully hash-verified, non-executing trunk artifact",
                              artifact.root.string());
        }
        if (!report.imported_manifest_verified || !report.imported_file_hashes_verified ||
            report.catalog.model_id.empty() || report.catalog.revision.empty() ||
            !plan.imported_file_hashes_verified || plan.trunk_ranges.empty() ||
            plan.trunk_range_count != plan.trunk_ranges.size() ||
            plan.trunk_range_count > model::kMaximumRangeArtifactRanges ||
            plan.trunk_bytes == 0U || plan.tile_limit_bytes == 0U) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek trunk binding requires hash-verified catalog metadata and a bounded plan",
                              report.catalog.model_id);
        }
        if (checkpoint_spec.model_id.empty() || checkpoint_spec.revision.empty() ||
            checkpoint_spec.shards.empty() ||
            artifact.model_id != checkpoint_spec.model_id ||
            artifact.revision != checkpoint_spec.revision ||
            artifact.model_id != report.catalog.model_id ||
            artifact.revision != report.catalog.revision ||
            artifact.model_id != plan.model_id || artifact.revision != plan.revision ||
            artifact.layer != plan.layer || artifact.ranges.size() != plan.trunk_range_count ||
            artifact.payload_bytes != plan.trunk_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek trunk artifact, catalog, plan and pinned checkpoint identities differ",
                              artifact.model_id);
        }

        auto rebuilt = build_deepseek_v4_layer_probe_plan(
            report,
            plan.layer,
            plan.selected_experts,
            plan.tile_limit_bytes,
            model::kMaximumRangeArtifactRanges);
        if (!rebuilt || rebuilt.value().trunk_ranges.size() != plan.trunk_ranges.size() ||
            rebuilt.value().trunk_bytes != plan.trunk_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek trunk plan cannot be reproduced from the verified catalog",
                              std::to_string(plan.layer));
        }
        for (std::size_t index = 0U; index < plan.trunk_ranges.size(); ++index) {
            if (!same_trunk_range(plan.trunk_ranges[index], rebuilt.value().trunk_ranges[index])) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek trunk plan differs from the verified catalog",
                                  plan.trunk_ranges[index].tensor_name);
            }
        }

        std::map<std::string, const DeepseekV4GgufShardIdentity*, std::less<>> shards;
        for (const auto& shard : checkpoint_spec.shards) {
            const auto object = shard.full_relative_path.filename().generic_string();
            if (object.empty() || shard.full_bytes == 0U ||
                !util::is_sha256_hex(shard.full_sha256) ||
                !shards.emplace(object, &shard).second) {
                return make_error(ErrorCode::integrity_error,
                                  "Pinned DeepSeek checkpoint has an invalid or ambiguous shard identity",
                                  object);
            }
        }

        DeepseekV4TrunkArtifactBinding binding;
        binding.artifact_root = artifact.root;
        binding.model_id = artifact.model_id;
        binding.revision = artifact.revision;
        binding.source_plan_sha256 = artifact.source_plan_sha256;
        binding.layer = artifact.layer;
        binding.payload_bytes = artifact.payload_bytes;
        binding.metadata_catalog_hashes_verified = true;
        binding.local_range_hashes_verified = true;
        binding.ranges.reserve(artifact.ranges.size());

        ByteCount total_bytes = 0U;
        for (std::size_t index = 0U; index < artifact.ranges.size(); ++index) {
            const auto& local = artifact.ranges[index];
            const auto& expected = plan.trunk_ranges[index];
            const auto object = expected.source.shard.filename().generic_string();
            const auto shard = shards.find(object);
            if (shard == shards.end()) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek trunk range names an object outside the pinned checkpoint",
                                  object);
            }
            const auto& identity = *shard->second;
            if (local.role != "trunk" ||
                local.label != expected_label(index, expected.tensor_name) ||
                local.source_object != object ||
                local.source_path.generic_string() != identity.full_relative_path.generic_string() ||
                local.source_object_bytes != identity.full_bytes ||
                local.source_object_sha256 != identity.full_sha256 ||
                local.source_offset != expected.source.offset ||
                local.source_bytes != expected.source.bytes ||
                local.bytes != expected.source.bytes ||
                expected.source.offset > identity.full_bytes ||
                expected.source.bytes > identity.full_bytes - expected.source.offset) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek local range does not match its pinned source span",
                                  expected.tensor_name);
            }
            if (local.bytes > std::numeric_limits<ByteCount>::max() - total_bytes) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek bound trunk byte total overflows",
                                  expected.tensor_name);
            }
            total_bytes += local.bytes;
            binding.maximum_local_range_bytes =
                std::max(binding.maximum_local_range_bytes, local.bytes);
            binding.ranges.push_back(DeepseekV4BoundTrunkRange{
                index,
                expected.tensor_name,
                expected.role,
                expected.tensor_offset,
                expected.final_tile,
                expected.ggml_type,
                expected.source,
                model::TensorSourceRange{local.path, 0U, local.bytes},
                local.sha256,
            });
        }
        if (total_bytes != binding.payload_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek bound trunk byte total differs from its artifact",
                              artifact.root.string());
        }
        binding.exact_source_binding_verified = true;
        return binding;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate bounded DeepSeek trunk artifact binding",
                          artifact.root.string());
    }
}

Result<std::string> deepseek_v4_bound_trunk_range_logical_id(
    const DeepseekV4TrunkArtifactBinding& binding,
    std::size_t range_index) {
    if (!binding.metadata_catalog_hashes_verified || !binding.local_range_hashes_verified ||
        !binding.exact_source_binding_verified || binding.complete_checkpoint_verified ||
        binding.execution_supported || binding.model_id.empty() || binding.revision.empty() ||
        range_index >= binding.ranges.size()) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek bound trunk load requires an exact non-executing binding",
                          binding.artifact_root.string());
    }
    const auto& range = binding.ranges[range_index];
    if (range.artifact_index != range_index || range.tensor_name.empty() ||
        range.local_source.shard.empty() || range.local_source.offset != 0U ||
        range.local_source.bytes == 0U ||
        range.local_source.bytes != range.original_source.bytes ||
        !util::is_sha256_hex(range.local_sha256)) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek bound trunk local range identity is invalid",
                          range.tensor_name);
    }
    return "deepseek/" + binding.model_id + "/layer/" +
           std::to_string(binding.layer) + "/trunk-range/" +
           std::to_string(range_index) + "/" + range.tensor_name;
}

Result<model::LoadedBoundRange> load_deepseek_v4_bound_trunk_range(
    const DeepseekV4TrunkArtifactBinding& binding,
    std::size_t range_index,
    model::ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const model::BoundRangeLoadOptions& options) {
    auto logical_id = deepseek_v4_bound_trunk_range_logical_id(binding, range_index);
    if (!logical_id) return logical_id.error();
    const auto& range = binding.ranges[range_index];
    return model::load_bound_range(
        model::BoundRangeIdentity{logical_id.take_value(), range.local_source, range.local_sha256},
        residency,
        governor,
        options);
}

std::string to_json(const DeepseekV4TrunkArtifactBinding& binding) {
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"vastloom.deepseek_v4.trunk_artifact_binding.v1\""
           << ",\"model_id\":\"" << util::json_escape(binding.model_id)
           << "\",\"revision\":\"" << util::json_escape(binding.revision)
           << "\",\"layer\":" << binding.layer
           << ",\"payload_bytes\":" << binding.payload_bytes
           << ",\"range_count\":" << binding.ranges.size()
           << ",\"maximum_local_range_bytes\":" << binding.maximum_local_range_bytes
           << ",\"source_plan_sha256\":\"" << binding.source_plan_sha256 << '"'
           << ",\"metadata_catalog_hashes_verified\":"
           << (binding.metadata_catalog_hashes_verified ? "true" : "false")
           << ",\"local_range_hashes_verified\":"
           << (binding.local_range_hashes_verified ? "true" : "false")
           << ",\"exact_source_binding_verified\":"
           << (binding.exact_source_binding_verified ? "true" : "false")
           << ",\"complete_checkpoint_verified\":"
           << (binding.complete_checkpoint_verified ? "true" : "false")
           << ",\"execution_supported\":"
           << (binding.execution_supported ? "true" : "false") << '}';
    return output.str();
}

std::string to_text(const DeepseekV4TrunkArtifactBinding& binding) {
    std::ostringstream output;
    output << "DeepSeek verified trunk artifact: " << binding.model_id << '\n'
           << "Revision:             " << binding.revision << '\n'
           << "Layer:                " << binding.layer << '\n'
           << "Local ranges:         " << binding.ranges.size() << " ("
           << binding.payload_bytes << " bytes)" << '\n'
           << "Largest range:        " << binding.maximum_local_range_bytes << " bytes" << '\n'
           << "Catalog hashes:       verified" << '\n'
           << "Local range hashes:   verified" << '\n'
           << "Exact source binding: verified" << '\n'
           << "Complete checkpoint:  no" << '\n'
           << "Execution:            unsupported";
    return output.str();
}

}  // namespace vastloom::inference
