#include "vastloom/inference/deepseek_v4_range_artifact.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
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

bool same_global_range(const DeepseekV4GlobalTensorRange& left,
                       const DeepseekV4GlobalTensorRange& right) {
    return left.tensor_name == right.tensor_name && left.role == right.role &&
           left.shape == right.shape && left.ggml_type == right.ggml_type &&
           same_source(left.source, right.source);
}

std::string expected_label(std::size_t index, std::string_view tensor_name) {
    return "trunk[" + std::to_string(index) + "] " + std::string(tensor_name);
}

std::string expected_global_label(std::size_t index, std::string_view tensor_name) {
    return "global[" + std::to_string(index) + "] " + std::string(tensor_name);
}

}  // namespace

DeepseekV4LoadedTrunkTensorSet::DeepseekV4LoadedTrunkTensorSet(
    std::string logical_id,
    ByteCount payload_bytes,
    std::vector<DeepseekV4BoundTrunkTensor> tensors,
    std::shared_ptr<const model::ResidentByteBuffer> buffer,
    model::BoundRangeSetLoadTelemetry telemetry)
    : logical_id_(std::move(logical_id)),
      payload_bytes_(payload_bytes),
      tensors_(std::move(tensors)),
      buffer_(std::move(buffer)),
      telemetry_(std::move(telemetry)) {}

const DeepseekV4BoundTrunkTensor* DeepseekV4LoadedTrunkTensorSet::find(
    std::string_view name) const noexcept {
    const auto found = std::find_if(
        tensors_.begin(), tensors_.end(),
        [name](const DeepseekV4BoundTrunkTensor& tensor) {
            return tensor.name == name;
        });
    return found == tensors_.end() ? nullptr : &*found;
}

Result<std::span<const std::byte>> DeepseekV4LoadedTrunkTensorSet::tensor(
    std::string_view name) const {
    const auto* found = find(name);
    const auto payload = buffer_ == nullptr ? std::span<const std::byte>{} : buffer_->bytes();
    if (found == nullptr || found->packed_offset > payload.size() ||
        found->bytes > payload.size() - found->packed_offset) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek bound trunk tensor is unavailable",
                          std::string(name));
    }
    return payload.subspan(static_cast<std::size_t>(found->packed_offset),
                           static_cast<std::size_t>(found->bytes));
}

DeepseekV4LoadedGlobalTensorSet::DeepseekV4LoadedGlobalTensorSet(
    std::string logical_id,
    ByteCount payload_bytes,
    std::vector<DeepseekV4BoundGlobalTensor> tensors,
    std::shared_ptr<const model::ResidentByteBuffer> buffer,
    model::BoundRangeSetLoadTelemetry telemetry)
    : logical_id_(std::move(logical_id)),
      payload_bytes_(payload_bytes),
      tensors_(std::move(tensors)),
      buffer_(std::move(buffer)),
      telemetry_(std::move(telemetry)) {}

const DeepseekV4BoundGlobalTensor* DeepseekV4LoadedGlobalTensorSet::find(
    std::string_view name) const noexcept {
    const auto found = std::find_if(
        tensors_.begin(), tensors_.end(),
        [name](const DeepseekV4BoundGlobalTensor& tensor) {
            return tensor.name == name;
        });
    return found == tensors_.end() ? nullptr : &*found;
}

Result<std::span<const std::byte>> DeepseekV4LoadedGlobalTensorSet::tensor(
    std::string_view name) const {
    const auto* found = find(name);
    const auto payload = buffer_ == nullptr ? std::span<const std::byte>{} : buffer_->bytes();
    if (found == nullptr || found->packed_offset > payload.size() ||
        found->bytes > payload.size() - found->packed_offset) {
        return make_error(ErrorCode::not_found,
                          "DeepSeek bound global tensor is unavailable",
                          std::string(name));
    }
    return payload.subspan(static_cast<std::size_t>(found->packed_offset),
                           static_cast<std::size_t>(found->bytes));
}

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

Result<std::string> deepseek_v4_bound_trunk_tensor_set_logical_id(
    const DeepseekV4TrunkArtifactBinding& binding) {
    if (!binding.metadata_catalog_hashes_verified ||
        !binding.local_range_hashes_verified ||
        !binding.exact_source_binding_verified ||
        binding.complete_checkpoint_verified || binding.execution_supported ||
        binding.model_id.empty() || binding.revision.empty() ||
        binding.artifact_root.empty() ||
        !util::is_sha256_hex(binding.source_plan_sha256) ||
        binding.payload_bytes == 0U || binding.ranges.empty() ||
        binding.ranges.size() > model::kMaximumBoundRangeSetRanges ||
        binding.maximum_local_range_bytes == 0U ||
        binding.maximum_local_range_bytes > binding.payload_bytes) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek bound trunk tensor-set load requires an exact non-executing binding",
                          binding.artifact_root.string());
    }
    return "deepseek/" + binding.model_id + "/layer/" +
           std::to_string(binding.layer) + "/trunk-tensor-set";
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

Result<DeepseekV4CatalogRelocation> deepseek_v4_relocate_catalog_to_artifacts(
    const model::LogicalTensorCatalog& catalog,
    std::span<const model::RangeArtifact> artifacts) {
    if (artifacts.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "Catalog relocation requires at least one verified range artifact",
                          catalog.model_id);
    }
    try {
        // Source identity is (object, offset, bytes); the catalog guarantees it is unique.
        struct SourceKey {
            std::string object;
            ByteCount offset;
            ByteCount bytes;
            bool operator<(const SourceKey& other) const noexcept {
                if (object != other.object) return object < other.object;
                if (offset != other.offset) return offset < other.offset;
                return bytes < other.bytes;
            }
        };
        std::map<SourceKey, std::size_t> by_source;
        for (std::size_t index = 0U; index < catalog.tensors.size(); ++index) {
            const auto& source = catalog.tensors[index].source;
            by_source.emplace(
                SourceKey{source.shard.filename().string(), source.offset, source.bytes}, index);
        }

        DeepseekV4CatalogRelocation result;
        result.catalog = catalog;
        // The source catalog describes a checkpoint whose payload was never fetched. After
        // relocation the imported tensors do have readable payload, so consumers that refuse
        // metadata-only catalogs must be allowed to proceed. This does not claim the whole
        // checkpoint is present: every tensor that was not relocated still points at an absent
        // file and fails on open rather than reading anything.
        result.catalog.metadata_only = false;
        std::set<std::size_t> relocated;
        for (const auto& artifact : artifacts) {
            if (!artifact.full_hash_verified || artifact.execution_supported ||
                artifact.complete_checkpoint_verified ||
                artifact.model_id != catalog.model_id ||
                artifact.revision != catalog.revision) {
                return make_error(
                    ErrorCode::integrity_error,
                    "Range artifact identity does not match the catalog it would relocate",
                    artifact.root.string());
            }
            for (const auto& entry : artifact.ranges) {
                const auto found = by_source.find(
                    SourceKey{entry.source_object, entry.source_offset, entry.source_bytes});
                if (found == by_source.end()) {
                    return make_error(
                        ErrorCode::not_found,
                        "Imported range does not correspond to any catalog tensor", entry.label);
                }
                if (!relocated.insert(found->second).second) {
                    return make_error(ErrorCode::invalid_format,
                                      "Two imported ranges claim the same catalog tensor",
                                      entry.label);
                }
                auto& tensor = result.catalog.tensors[found->second];
                if (entry.bytes != tensor.source.bytes) {
                    return make_error(
                        ErrorCode::integrity_error,
                        "Imported range length differs from its catalog tensor", entry.label);
                }
                tensor.source = model::TensorSourceRange{entry.path, 0U, entry.bytes};
                result.relocated_bytes += entry.bytes;
            }
        }
        result.relocated_tensors = relocated.size();

        // An expert bundle is only usable when every one of its tensors moved: a bundle half in the
        // artifact and half in an absent checkpoint would read across two different worlds.
        for (auto& bundle : result.catalog.experts) {
            const bool complete = std::all_of(
                bundle.tensor_indices.begin(), bundle.tensor_indices.end(),
                [&relocated](std::size_t index) { return relocated.count(index) != 0U; });
            if (!complete) continue;
            // Coalesced bundles cover several tensors per read, so a positional rewrite would be
            // wrong. This checkpoint reads one range per tensor; anything else must not be guessed.
            if (bundle.read_ranges.size() != bundle.tensor_indices.size()) {
                return make_error(
                    ErrorCode::unsupported,
                    "Relocating a coalesced expert bundle is not supported",
                    "layer " + std::to_string(bundle.layer) + " expert " +
                        std::to_string(bundle.expert));
            }
            for (std::size_t slot = 0U; slot < bundle.tensor_indices.size(); ++slot) {
                bundle.read_ranges[slot] = result.catalog.tensors[bundle.tensor_indices[slot]].source;
            }
            ++result.relocated_expert_bundles;
        }
        return result;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the relocated DeepSeek catalog", catalog.model_id);
    }
}

Result<DeepseekV4HashRouteResolution> deepseek_v4_resolve_hash_route(
    const DeepseekV4TrunkArtifactBinding& binding,
    const DeepseekV4GgufCatalogReport& report,
    std::uint32_t token,
    safety::ResourceGovernor& governor) {
    if (!report.reference_architecture.has_value()) {
        return make_error(
            ErrorCode::unsupported,
            "DeepSeek hash-route resolution requires verified architecture metadata",
            binding.model_id);
    }
    const auto& architecture = report.reference_architecture.value();
    if (architecture.hash_layer_count == 0U || binding.layer >= architecture.hash_layer_count) {
        return make_error(
            ErrorCode::unsupported,
            "DeepSeek layer uses a learned router; its selection cannot be derived without execution",
            "layer " + std::to_string(binding.layer));
    }
    if (architecture.vocabulary_size == 0U || architecture.experts_per_token == 0U ||
        architecture.expert_count == 0U ||
        architecture.experts_per_token > architecture.expert_count) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek hash-route architecture dimensions are invalid",
                          binding.model_id);
    }
    if (token >= architecture.vocabulary_size) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek hash-route token id exceeds the pinned vocabulary",
                          std::to_string(token));
    }

    const std::string table_name =
        "blk." + std::to_string(binding.layer) + ".ffn_gate_tid2eid.weight";
    const auto found = std::find_if(
        binding.ranges.begin(), binding.ranges.end(),
        [&table_name](const DeepseekV4BoundTrunkRange& range) {
            return range.tensor_name == table_name;
        });
    if (found == binding.ranges.end()) {
        return make_error(ErrorCode::not_found,
                          "The imported trunk artifact does not contain the routing table",
                          table_name);
    }
    if (found->tensor_offset != 0U || !found->final_tile) {
        return make_error(
            ErrorCode::unsupported,
            "DeepSeek routing tables must be bound as one complete tile", table_name);
    }
    const auto elements =
        static_cast<std::uint64_t>(architecture.vocabulary_size) * architecture.experts_per_token;
    const auto expected_bytes = elements * sizeof(std::int32_t);
    if (found->local_source.bytes != expected_bytes) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek routing table size does not match its pinned dimensions",
                          table_name);
    }

    const auto range_index =
        static_cast<std::size_t>(std::distance(binding.ranges.begin(), found));
    auto logical_id = deepseek_v4_bound_trunk_range_logical_id(binding, range_index);
    if (!logical_id) return logical_id.error();
    auto residency =
        model::ResidencyObject::create(logical_id.take_value(), found->local_source.bytes);
    if (!residency) return residency.error();
    auto loaded =
        load_deepseek_v4_bound_trunk_range(binding, range_index, residency.value(), governor);
    if (!loaded) return loaded.error();
    const auto bytes = loaded.value().bytes();
    if (bytes.size() != expected_bytes) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek routing table payload changed size after verification",
                          table_name);
    }

    try {
        DeepseekV4HashRouteResolution resolution;
        resolution.layer = binding.layer;
        resolution.token = token;
        resolution.expert_count = architecture.expert_count;
        resolution.experts_per_token = architecture.experts_per_token;
        resolution.table_tensor_name = table_name;
        resolution.table_range_sha256 = found->local_sha256;
        resolution.table_bytes = found->local_source.bytes;
        resolution.slot_experts.reserve(architecture.experts_per_token);
        const auto row = static_cast<std::size_t>(token) * architecture.experts_per_token;
        for (std::uint32_t slot = 0U; slot < architecture.experts_per_token; ++slot) {
            std::int32_t value = 0;
            std::memcpy(&value, bytes.data() + (row + slot) * sizeof(std::int32_t),
                        sizeof(value));
            if (value < 0 || static_cast<std::uint32_t>(value) >= architecture.expert_count) {
                return make_error(
                    ErrorCode::invalid_format,
                    "DeepSeek routing table names an out-of-range expert", table_name);
            }
            resolution.slot_experts.push_back(static_cast<std::uint32_t>(value));
        }
        resolution.unique_experts = resolution.slot_experts;
        std::sort(resolution.unique_experts.begin(), resolution.unique_experts.end());
        resolution.unique_experts.erase(
            std::unique(resolution.unique_experts.begin(), resolution.unique_experts.end()),
            resolution.unique_experts.end());
        return resolution;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the bounded DeepSeek hash-route result",
                          table_name);
    }
}

Result<DeepseekV4LoadedTrunkTensorSet>
load_deepseek_v4_bound_trunk_tensor_set(
    const DeepseekV4TrunkArtifactBinding& binding,
    const DeepseekV4GgufCatalogReport& report,
    model::ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const model::BoundRangeSetLoadOptions& options) {
    auto logical_id = deepseek_v4_bound_trunk_tensor_set_logical_id(binding);
    if (!logical_id) return logical_id.error();
    bool published_residency = false;
    try {
        if (!report.imported_manifest_verified ||
            !report.imported_file_hashes_verified ||
            report.catalog.model_id != binding.model_id ||
            report.catalog.revision != binding.revision ||
            binding.layer >= report.layers) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek trunk tensor set differs from its verified catalog identity",
                binding.model_id);
        }

        std::map<std::string_view, std::size_t, std::less<>> catalog_by_name;
        std::size_t expected_tensor_count = 0U;
        ByteCount expected_bytes = 0U;
        for (std::size_t index = 0U; index < report.catalog.tensors.size(); ++index) {
            const auto& descriptor = report.catalog.tensors[index];
            if (descriptor.name.empty() ||
                !catalog_by_name.emplace(descriptor.name, index).second) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek catalog contains duplicate tensor identities",
                                  descriptor.name);
            }
            if (descriptor.role == model::TensorRole::routed_expert ||
                !descriptor.layer.has_value() ||
                *descriptor.layer != binding.layer) {
                continue;
            }
            if (descriptor.source.bytes == 0U ||
                descriptor.source.bytes > std::numeric_limits<ByteCount>::max() - expected_bytes) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek catalog trunk byte identity is invalid",
                                  descriptor.name);
            }
            ++expected_tensor_count;
            expected_bytes += descriptor.source.bytes;
        }
        if (expected_tensor_count == 0U ||
            binding.ranges.size() != expected_tensor_count ||
            binding.payload_bytes != expected_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek trunk binding does not cover the complete layer tensor set",
                              std::to_string(binding.layer));
        }

        model::BoundRangeSetIdentity set_identity;
        set_identity.logical_id = logical_id.value();
        set_identity.total_bytes = binding.payload_bytes;
        set_identity.ranges.reserve(binding.ranges.size());
        std::vector<DeepseekV4BoundTrunkTensor> tensors;
        tensors.reserve(binding.ranges.size());
        std::set<std::string, std::less<>> represented;
        ByteCount packed_offset = 0U;
        ByteCount maximum_range = 0U;
        for (std::size_t index = 0U; index < binding.ranges.size(); ++index) {
            const auto& range = binding.ranges[index];
            const auto catalog_entry = catalog_by_name.find(range.tensor_name);
            if (catalog_entry == catalog_by_name.end() ||
                range.artifact_index != index || range.tensor_offset != 0U ||
                !range.final_tile || range.local_source.offset != 0U ||
                range.local_source.bytes == 0U ||
                range.local_source.bytes != range.original_source.bytes ||
                !util::is_sha256_hex(range.local_sha256) ||
                !represented.emplace(range.tensor_name).second) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek trunk binding contains an incomplete or duplicate tensor",
                                  range.tensor_name);
            }
            const auto& descriptor = report.catalog.tensors[catalog_entry->second];
            if (descriptor.role == model::TensorRole::routed_expert ||
                !descriptor.layer.has_value() ||
                *descriptor.layer != binding.layer ||
                descriptor.role != range.role ||
                descriptor.ggml_type != range.ggml_type ||
                !same_source(descriptor.source, range.original_source) ||
                descriptor.source.bytes != range.local_source.bytes) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek bound trunk tensor differs from its catalog type or source",
                                  range.tensor_name);
            }
            if (range.local_source.bytes >
                std::numeric_limits<ByteCount>::max() - packed_offset) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek packed trunk tensor offsets overflow",
                                  range.tensor_name);
            }
            const auto range_logical_id =
                set_identity.logical_id + "/range/" + std::to_string(index) + "/" +
                range.tensor_name;
            set_identity.ranges.push_back(model::BoundRangeIdentity{
                range_logical_id, range.local_source, range.local_sha256});
            tensors.push_back(DeepseekV4BoundTrunkTensor{
                catalog_entry->second,
                descriptor.name,
                descriptor.dtype,
                descriptor.shape,
                descriptor.role,
                descriptor.ggml_type,
                packed_offset,
                descriptor.source.bytes,
            });
            packed_offset += range.local_source.bytes;
            maximum_range = std::max(maximum_range, range.local_source.bytes);
        }
        if (represented.size() != expected_tensor_count ||
            packed_offset != binding.payload_bytes ||
            maximum_range != binding.maximum_local_range_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek bound trunk tensor-set totals are inconsistent",
                              binding.model_id);
        }

        auto loaded = model::load_bound_range_set(
            set_identity, residency, governor, options);
        if (!loaded) return loaded.error();
        published_residency = true;
        return DeepseekV4LoadedTrunkTensorSet(
            std::move(set_identity.logical_id), binding.payload_bytes,
            std::move(tensors), loaded.value().retain_buffer(),
            loaded.value().telemetry());
    } catch (const std::bad_alloc&) {
        const auto error = make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate bounded DeepSeek trunk tensor-set metadata",
            binding.model_id);
        if (published_residency) {
            auto failed = residency.fail(error.message);
            if (!failed) return failed.error();
        }
        return error;
    }
}

Result<DeepseekV4GlobalArtifactBinding> bind_deepseek_v4_global_range_artifact(
    const model::RangeArtifact& artifact,
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4GlobalTensorPlan& plan,
    const DeepseekV4GgufCheckpointSpec& checkpoint_spec) {
    try {
        if (!artifact.full_hash_verified || artifact.complete_checkpoint_verified ||
            artifact.execution_supported || artifact.scope != "globals" ||
            artifact.layer != 0U || artifact.routing_authority_verified ||
            artifact.source_plan_format !=
                "vastloom.deepseek_v4.global_tensor_plan.v1" ||
            !util::is_sha256_hex(artifact.source_plan_sha256)) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek global binding requires a fully hash-verified, non-executing globals artifact",
                artifact.root.string());
        }
        if (!report.imported_manifest_verified ||
            !report.imported_file_hashes_verified ||
            report.catalog.model_id.empty() || report.catalog.revision.empty() ||
            !plan.imported_file_hashes_verified || plan.tensor_count != 6U ||
            plan.tensors.size() != plan.tensor_count || plan.payload_bytes == 0U ||
            plan.maximum_read_range_bytes == 0U) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek global binding requires a hash-verified catalog and complete six-tensor plan",
                report.catalog.model_id);
        }
        if (checkpoint_spec.model_id.empty() || checkpoint_spec.revision.empty() ||
            checkpoint_spec.shards.empty() ||
            artifact.model_id != checkpoint_spec.model_id ||
            artifact.revision != checkpoint_spec.revision ||
            artifact.model_id != report.catalog.model_id ||
            artifact.revision != report.catalog.revision ||
            artifact.model_id != plan.model_id || artifact.revision != plan.revision ||
            artifact.ranges.size() != plan.tensor_count ||
            artifact.payload_bytes != plan.payload_bytes) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek globals artifact, catalog, plan and pinned checkpoint identities differ",
                artifact.model_id);
        }

        auto rebuilt = build_deepseek_v4_global_tensor_plan(report);
        if (!rebuilt || rebuilt.value().tensors.size() != plan.tensors.size() ||
            rebuilt.value().payload_bytes != plan.payload_bytes ||
            rebuilt.value().maximum_read_range_bytes !=
                plan.maximum_read_range_bytes) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek global plan cannot be reproduced from the verified catalog",
                plan.model_id);
        }
        for (std::size_t index = 0U; index < plan.tensors.size(); ++index) {
            if (!same_global_range(plan.tensors[index], rebuilt.value().tensors[index])) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek global plan differs from the verified catalog",
                    plan.tensors[index].tensor_name);
            }
        }

        std::map<std::string, const DeepseekV4GgufShardIdentity*, std::less<>> shards;
        for (const auto& shard : checkpoint_spec.shards) {
            const auto object = shard.full_relative_path.filename().generic_string();
            if (object.empty() || shard.full_bytes == 0U ||
                !util::is_sha256_hex(shard.full_sha256) ||
                !shards.emplace(object, &shard).second) {
                return make_error(
                    ErrorCode::integrity_error,
                    "Pinned DeepSeek checkpoint has an invalid or ambiguous shard identity",
                    object);
            }
        }

        DeepseekV4GlobalArtifactBinding binding;
        binding.artifact_root = artifact.root;
        binding.model_id = artifact.model_id;
        binding.revision = artifact.revision;
        binding.source_plan_sha256 = artifact.source_plan_sha256;
        binding.payload_bytes = artifact.payload_bytes;
        binding.metadata_catalog_hashes_verified = true;
        binding.local_range_hashes_verified = true;
        binding.ranges.reserve(artifact.ranges.size());
        ByteCount total_bytes = 0U;
        for (std::size_t index = 0U; index < artifact.ranges.size(); ++index) {
            const auto& local = artifact.ranges[index];
            const auto& expected = plan.tensors[index];
            const auto object = expected.source.shard.filename().generic_string();
            const auto shard = shards.find(object);
            if (shard == shards.end()) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek global range names an object outside the pinned checkpoint",
                    object);
            }
            const auto& identity = *shard->second;
            if (local.role != "global" ||
                local.label != expected_global_label(index, expected.tensor_name) ||
                local.source_object != object ||
                local.source_path.generic_string() !=
                    identity.full_relative_path.generic_string() ||
                local.source_object_bytes != identity.full_bytes ||
                local.source_object_sha256 != identity.full_sha256 ||
                local.source_offset != expected.source.offset ||
                local.source_bytes != expected.source.bytes ||
                local.bytes != expected.source.bytes ||
                expected.source.offset > identity.full_bytes ||
                expected.source.bytes > identity.full_bytes - expected.source.offset) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek local global range does not match its pinned source span",
                    expected.tensor_name);
            }
            if (local.bytes > std::numeric_limits<ByteCount>::max() - total_bytes) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek bound global byte total overflows",
                                  expected.tensor_name);
            }
            total_bytes += local.bytes;
            binding.maximum_local_range_bytes =
                std::max(binding.maximum_local_range_bytes, local.bytes);
            binding.ranges.push_back(DeepseekV4BoundGlobalRange{
                index,
                expected.tensor_name,
                expected.role,
                expected.ggml_type,
                expected.source,
                model::TensorSourceRange{local.path, 0U, local.bytes},
                local.sha256,
            });
        }
        if (total_bytes != binding.payload_bytes ||
            binding.maximum_local_range_bytes != plan.maximum_read_range_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek bound global totals differ from their artifact",
                              artifact.root.string());
        }
        binding.exact_source_binding_verified = true;
        return binding;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate bounded DeepSeek global artifact binding",
                          artifact.root.string());
    }
}

Result<std::string> deepseek_v4_bound_global_tensor_set_logical_id(
    const DeepseekV4GlobalArtifactBinding& binding) {
    if (!binding.metadata_catalog_hashes_verified ||
        !binding.local_range_hashes_verified ||
        !binding.exact_source_binding_verified ||
        binding.complete_checkpoint_verified || binding.execution_supported ||
        binding.model_id.empty() || binding.revision.empty() ||
        binding.artifact_root.empty() ||
        !util::is_sha256_hex(binding.source_plan_sha256) ||
        binding.payload_bytes == 0U || binding.ranges.size() != 6U ||
        binding.maximum_local_range_bytes == 0U ||
        binding.maximum_local_range_bytes > binding.payload_bytes) {
        return make_error(
            ErrorCode::integrity_error,
            "DeepSeek bound global tensor-set load requires an exact six-range binding",
            binding.artifact_root.string());
    }
    return "deepseek/" + binding.model_id + "/globals/tensor-set";
}

Result<DeepseekV4LoadedGlobalTensorSet>
load_deepseek_v4_bound_global_tensor_set(
    const DeepseekV4GlobalArtifactBinding& binding,
    const DeepseekV4GgufCatalogReport& report,
    model::ResidencyObject& residency,
    safety::ResourceGovernor& governor,
    const model::BoundRangeSetLoadOptions& options) {
    auto logical_id = deepseek_v4_bound_global_tensor_set_logical_id(binding);
    if (!logical_id) return logical_id.error();
    bool published_residency = false;
    try {
        if (!report.imported_manifest_verified ||
            !report.imported_file_hashes_verified ||
            report.catalog.model_id != binding.model_id ||
            report.catalog.revision != binding.revision) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek global tensor set differs from its verified catalog identity",
                binding.model_id);
        }

        std::map<std::string_view, std::size_t, std::less<>> catalog_by_name;
        std::size_t expected_tensor_count = 0U;
        ByteCount expected_bytes = 0U;
        for (std::size_t index = 0U; index < report.catalog.tensors.size(); ++index) {
            const auto& descriptor = report.catalog.tensors[index];
            if (descriptor.name.empty() ||
                !catalog_by_name.emplace(descriptor.name, index).second) {
                return make_error(ErrorCode::integrity_error,
                                  "DeepSeek catalog contains duplicate tensor identities",
                                  descriptor.name);
            }
            if (descriptor.layer.has_value()) continue;
            if (descriptor.expert.has_value() || descriptor.source.bytes == 0U ||
                descriptor.source.bytes >
                    std::numeric_limits<ByteCount>::max() - expected_bytes) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek catalog global tensor identity is invalid",
                                  descriptor.name);
            }
            ++expected_tensor_count;
            expected_bytes += descriptor.source.bytes;
        }
        if (expected_tensor_count != 6U ||
            binding.ranges.size() != expected_tensor_count ||
            binding.payload_bytes != expected_bytes) {
            return make_error(
                ErrorCode::integrity_error,
                "DeepSeek global binding does not cover the complete six-tensor set",
                binding.model_id);
        }

        model::BoundRangeSetIdentity set_identity;
        set_identity.logical_id = logical_id.value();
        set_identity.total_bytes = binding.payload_bytes;
        set_identity.ranges.reserve(binding.ranges.size());
        std::vector<DeepseekV4BoundGlobalTensor> tensors;
        tensors.reserve(binding.ranges.size());
        std::set<std::string, std::less<>> represented;
        ByteCount packed_offset = 0U;
        ByteCount maximum_range = 0U;
        for (std::size_t index = 0U; index < binding.ranges.size(); ++index) {
            const auto& range = binding.ranges[index];
            const auto catalog_entry = catalog_by_name.find(range.tensor_name);
            if (catalog_entry == catalog_by_name.end() ||
                range.artifact_index != index || range.local_source.offset != 0U ||
                range.local_source.bytes == 0U ||
                range.local_source.bytes != range.original_source.bytes ||
                !util::is_sha256_hex(range.local_sha256) ||
                !represented.emplace(range.tensor_name).second) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek global binding contains an incomplete or duplicate tensor",
                    range.tensor_name);
            }
            const auto& descriptor = report.catalog.tensors[catalog_entry->second];
            if (descriptor.layer.has_value() || descriptor.expert.has_value() ||
                descriptor.role != range.role || descriptor.ggml_type != range.ggml_type ||
                !same_source(descriptor.source, range.original_source) ||
                descriptor.source.bytes != range.local_source.bytes) {
                return make_error(
                    ErrorCode::integrity_error,
                    "DeepSeek bound global tensor differs from its catalog type or source",
                    range.tensor_name);
            }
            if (range.local_source.bytes >
                std::numeric_limits<ByteCount>::max() - packed_offset) {
                return make_error(ErrorCode::invalid_format,
                                  "DeepSeek packed global tensor offsets overflow",
                                  range.tensor_name);
            }
            set_identity.ranges.push_back(model::BoundRangeIdentity{
                set_identity.logical_id + "/range/" + std::to_string(index) + "/" +
                    range.tensor_name,
                range.local_source,
                range.local_sha256});
            tensors.push_back(DeepseekV4BoundGlobalTensor{
                catalog_entry->second,
                descriptor.name,
                descriptor.dtype,
                descriptor.shape,
                descriptor.role,
                descriptor.ggml_type,
                packed_offset,
                descriptor.source.bytes,
            });
            packed_offset += range.local_source.bytes;
            maximum_range = std::max(maximum_range, range.local_source.bytes);
        }
        if (represented.size() != expected_tensor_count ||
            packed_offset != binding.payload_bytes ||
            maximum_range != binding.maximum_local_range_bytes) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek bound global tensor-set totals are inconsistent",
                              binding.model_id);
        }

        auto loaded = model::load_bound_range_set(
            set_identity, residency, governor, options);
        if (!loaded) return loaded.error();
        published_residency = true;
        return DeepseekV4LoadedGlobalTensorSet(
            std::move(set_identity.logical_id), binding.payload_bytes,
            std::move(tensors), loaded.value().retain_buffer(),
            loaded.value().telemetry());
    } catch (const std::bad_alloc&) {
        const auto error = make_error(
            ErrorCode::resource_exhausted,
            "Unable to allocate bounded DeepSeek global tensor-set metadata",
            binding.model_id);
        if (published_residency) {
            auto failed = residency.fail(error.message);
            if (!failed) return failed.error();
        }
        return error;
    }
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

std::string to_json(const DeepseekV4GlobalArtifactBinding& binding) {
    std::ostringstream output;
    output << "{\"ok\":true,\"format\":\"vastloom.deepseek_v4.global_artifact_binding.v1\""
           << ",\"model_id\":\"" << util::json_escape(binding.model_id)
           << "\",\"revision\":\"" << util::json_escape(binding.revision)
           << "\",\"payload_bytes\":" << binding.payload_bytes
           << ",\"range_count\":" << binding.ranges.size()
           << ",\"maximum_local_range_bytes\":"
           << binding.maximum_local_range_bytes
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

std::string to_text(const DeepseekV4GlobalArtifactBinding& binding) {
    std::ostringstream output;
    output << "DeepSeek verified globals artifact: " << binding.model_id << '\n'
           << "Revision:             " << binding.revision << '\n'
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
