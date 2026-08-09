#include "stargate/inference/olmoe_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>

#include "stargate/model/imported_checkpoint.hpp"
#include "stargate/util/json.hpp"
#include "stargate/util/string.hpp"

namespace stargate::inference {
namespace {

constexpr std::string_view kModelId = "allenai/OLMoE-1B-7B-0924-Instruct";
constexpr std::string_view kRevision = "7f1c97f440f06ce36705e4f2b843edb5925f4498";
constexpr ByteCount kTensorPayloadBytes = 13'838'323'712ULL;
constexpr std::size_t kTensorCount = 3'219U;

struct ExpectedFile {
    std::string_view name;
    ByteCount bytes;
    std::string_view sha256;
    bool weight;
};

constexpr std::array<ExpectedFile, 9> kExpectedFiles{{
    {"config.json", 759U, "42ea6f5fe31e9243290ba3c348a4ad615b88a6fbc6e10544e21a87b6dff5da6c", false},
    {"generation_config.json", 120U, "77cf3f0caaa5e6d6972d910c70ca03f64fc47eccb4374082bdc8fc7b2c524a45", false},
    {"model-00001-of-00003.safetensors", 4'997'744'872ULL, "6c79ac0487c3f23e8ee3d38752197d1e4a1a39d6c1438ec5fd7862874bb19321", true},
    {"model-00002-of-00003.safetensors", 4'997'235'176ULL, "b58ca7a9f28f35d76b56b0ec40fc46c356fd84ccf078083b251a3ad6a2da9a35", true},
    {"model-00003-of-00003.safetensors", 3'843'741'912ULL, "e93653ed509223e63eabae6f72ac8ea5a41115c5c7785574b9b2068ca0961c45", true},
    {"model.safetensors.index.json", 287'214U, "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579", false},
    {"special_tokens_map.json", 433U, "cf22141a130514760f6280166b8d1c9a56610578188aad0790eb6eaf6ea1eb08", false},
    {"tokenizer.json", 2'115'598U, "b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167", false},
    {"tokenizer_config.json", 5'986U, "d3079f9c26a02fd28ee8d9bdfea4e96da7f5e803e06707ec8d1511acaaac7864", false},
}};

Result<std::uint32_t> required_u32(const util::JsonValue& root, std::string_view key) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "OLMoE config is missing integer field", std::string(key));
    auto parsed = value->as_u64();
    if (!parsed || parsed.value() > std::numeric_limits<std::uint32_t>::max()) {
        return make_error(ErrorCode::invalid_format, "OLMoE config integer is out of range", std::string(key));
    }
    return static_cast<std::uint32_t>(parsed.value());
}

Result<float> required_float(const util::JsonValue& root, std::string_view key) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "OLMoE config is missing numeric field", std::string(key));
    auto parsed = value->as_double();
    if (!parsed || !std::isfinite(parsed.value()) ||
        std::abs(parsed.value()) > static_cast<double>(std::numeric_limits<float>::max())) {
        return make_error(ErrorCode::invalid_format, "OLMoE config number is out of range", std::string(key));
    }
    return static_cast<float>(parsed.value());
}

Result<bool> required_bool(const util::JsonValue& root, std::string_view key) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "OLMoE config is missing boolean field", std::string(key));
    return value->as_bool();
}

Result<void> validate_imported_files(const model::ImportedCheckpoint& imported) {
    const std::size_t expected_count = imported.metadata_only ? 6U : kExpectedFiles.size();
    if (imported.files.size() != expected_count) {
        return make_error(ErrorCode::integrity_error, "OLMoE imported file set is incomplete or contains extras", imported.root.string());
    }
    for (const auto& expected : kExpectedFiles) {
        if (imported.metadata_only && expected.weight) continue;
        const auto iterator = std::find_if(imported.files.begin(), imported.files.end(), [&](const model::ImportedFile& file) {
            return file.relative_path.generic_string() == expected.name;
        });
        if (iterator == imported.files.end() || iterator->bytes != expected.bytes || iterator->sha256 != expected.sha256) {
            return make_error(ErrorCode::integrity_error, "OLMoE imported file identity does not match the pinned revision", std::string(expected.name));
        }
    }
    return {};
}

std::set<std::string, std::less<>> expected_tensor_names(const OlmoeConfig& config) {
    std::set<std::string, std::less<>> names{
        "lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"};
    constexpr std::array<std::string_view, 9> layer_tensors{{
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate.weight",
    }};
    constexpr std::array<std::string_view, 3> expert_tensors{{
        "down_proj.weight", "gate_proj.weight", "up_proj.weight"}};
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        for (const auto suffix : layer_tensors) names.insert(prefix + std::string(suffix));
        for (std::uint32_t expert = 0; expert < config.num_experts; ++expert) {
            const std::string expert_prefix = prefix + "mlp.experts." + std::to_string(expert) + ".";
            for (const auto suffix : expert_tensors) names.insert(expert_prefix + std::string(suffix));
        }
    }
    return names;
}

Result<ByteCount> validate_weight_index(const std::filesystem::path& path, const OlmoeConfig& config) {
    util::JsonLimits limits;
    limits.max_input_bytes = 1U * MiB;
    limits.max_nodes = 16'384U;
    limits.max_object_members = 4'096U;
    limits.max_array_elements = 16U;
    auto json = util::load_json_file(path, limits);
    if (!json) return json.error();
    const auto* metadata_value = json.value().find("metadata");
    const auto* map_value = json.value().find("weight_map");
    if (metadata_value == nullptr || map_value == nullptr) {
        return make_error(ErrorCode::invalid_format, "OLMoE weight index is missing metadata or weight_map", path.string());
    }
    const auto* total_value = metadata_value->find("total_size");
    auto weight_map = map_value->as_object();
    if (total_value == nullptr || !weight_map) {
        return make_error(ErrorCode::invalid_format, "OLMoE weight index fields are invalid", path.string());
    }
    auto total = total_value->as_u64();
    if (!total || total.value() != kTensorPayloadBytes) {
        return make_error(ErrorCode::integrity_error, "OLMoE tensor payload size does not match the pinned revision", path.string());
    }
    const auto expected = expected_tensor_names(config);
    if (expected.size() != kTensorCount || weight_map.value()->size() != expected.size()) {
        return make_error(ErrorCode::integrity_error, "OLMoE tensor count does not match the pinned architecture", path.string());
    }
    const std::set<std::string_view> shards{
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    };
    for (const auto& name : expected) {
        const auto iterator = weight_map.value()->find(name);
        if (iterator == weight_map.value()->end()) {
            return make_error(ErrorCode::integrity_error, "OLMoE weight index is missing an expected tensor", name);
        }
        auto shard = iterator->second.as_string();
        if (!shard || !shards.contains(shard.value())) {
            return make_error(ErrorCode::integrity_error, "OLMoE tensor maps to an unexpected shard", name);
        }
    }
    return total.value();
}

}  // namespace

Result<OlmoeConfig> load_olmoe_config(const std::filesystem::path& config_path) {
    util::JsonLimits limits;
    limits.max_input_bytes = 1U * MiB;
    limits.max_nodes = 4'096U;
    auto json = util::load_json_file(config_path, limits);
    if (!json) return json.error();
    if (!json.value().is_object()) return make_error(ErrorCode::invalid_format, "OLMoE config root is not an object", config_path.string());
    const auto* architectures_value = json.value().find("architectures");
    const auto* model_type_value = json.value().find("model_type");
    const auto* activation_value = json.value().find("hidden_act");
    const auto* dtype_value = json.value().find("torch_dtype");
    if (architectures_value == nullptr || model_type_value == nullptr || activation_value == nullptr || dtype_value == nullptr) {
        return make_error(ErrorCode::invalid_format, "OLMoE config identity fields are missing", config_path.string());
    }
    auto architectures = architectures_value->as_array();
    auto model_type = model_type_value->as_string();
    auto activation = activation_value->as_string();
    auto dtype = dtype_value->as_string();
    if (!architectures || architectures.value()->size() != 1U || !(*architectures.value())[0].as_string() ||
        (*architectures.value())[0].as_string().value() != "OlmoeForCausalLM" || !model_type || model_type.value() != "olmoe" ||
        !activation || activation.value() != "silu" || !dtype || dtype.value() != "bfloat16") {
        return make_error(ErrorCode::unsupported, "Checkpoint is not the pinned BF16 OLMoE architecture", config_path.string());
    }

    OlmoeConfig config;
    auto vocab = required_u32(json.value(), "vocab_size");
    auto hidden = required_u32(json.value(), "hidden_size");
    auto intermediate = required_u32(json.value(), "intermediate_size");
    auto layers = required_u32(json.value(), "num_hidden_layers");
    auto heads = required_u32(json.value(), "num_attention_heads");
    auto kv_heads = required_u32(json.value(), "num_key_value_heads");
    auto context = required_u32(json.value(), "max_position_embeddings");
    auto experts = required_u32(json.value(), "num_experts");
    auto top_k = required_u32(json.value(), "num_experts_per_tok");
    auto eos = required_u32(json.value(), "eos_token_id");
    auto pad = required_u32(json.value(), "pad_token_id");
    auto theta = required_float(json.value(), "rope_theta");
    auto attention_bias = required_bool(json.value(), "attention_bias");
    auto tied = required_bool(json.value(), "tie_word_embeddings");
    auto normalized = required_bool(json.value(), "norm_topk_prob");
    if (!vocab) return vocab.error();
    if (!hidden) return hidden.error();
    if (!intermediate) return intermediate.error();
    if (!layers) return layers.error();
    if (!heads) return heads.error();
    if (!kv_heads) return kv_heads.error();
    if (!context) return context.error();
    if (!experts) return experts.error();
    if (!top_k) return top_k.error();
    if (!eos) return eos.error();
    if (!pad) return pad.error();
    if (!theta) return theta.error();
    if (!attention_bias) return attention_bias.error();
    if (!tied) return tied.error();
    if (!normalized) return normalized.error();
    config = {vocab.value(), hidden.value(), intermediate.value(), layers.value(), heads.value(), kv_heads.value(),
              context.value(), experts.value(), top_k.value(), eos.value(), pad.value(), 1.0e-5F, theta.value()};

    const auto* clip = json.value().find("clip_qkv");
    const auto* rope_scaling = json.value().find("rope_scaling");
    const auto exact = config.vocab_size == 50'304U && config.hidden_size == 2'048U &&
                       config.intermediate_size == 1'024U && config.num_hidden_layers == 16U &&
                       config.num_attention_heads == 16U && config.num_key_value_heads == 16U &&
                       config.max_position_embeddings == 4'096U && config.num_experts == 64U &&
                       config.num_experts_per_token == 8U && config.eos_token_id == 50'279U &&
                       config.pad_token_id == 1U && std::abs(config.rope_theta - 10'000.0F) < 0.01F;
    if (!exact || attention_bias.value() || tied.value() || normalized.value() || clip == nullptr || !clip->is_null() ||
        rope_scaling == nullptr || !rope_scaling->is_null()) {
        return make_error(ErrorCode::unsupported, "OLMoE config differs from the pinned supported revision", config_path.string());
    }
    return config;
}

Result<OlmoeInspection> inspect_olmoe_checkpoint(const std::filesystem::path& model_directory,
                                                 bool verify_hashes,
                                                 ByteCount safe_ram_cap_bytes) {
    auto imported = model::verify_imported_checkpoint(model_directory, verify_hashes);
    if (!imported) return imported.error();
    if (imported.value().model_id != kModelId || imported.value().revision != kRevision) {
        return make_error(ErrorCode::unsupported, "Imported checkpoint is not the pinned OLMoE revision", model_directory.string());
    }
    auto files_valid = validate_imported_files(imported.value());
    if (!files_valid) return files_valid.error();
    auto config = load_olmoe_config(imported.value().root / "config.json");
    if (!config) return config.error();
    auto payload = validate_weight_index(imported.value().root / "model.safetensors.index.json", config.value());
    if (!payload) return payload.error();

    const ByteCount bytes_per_expert = static_cast<ByteCount>(3U) * config.value().hidden_size *
                                       config.value().intermediate_size * 2U;
    const ByteCount expert_bytes = bytes_per_expert * config.value().num_experts * config.value().num_hidden_layers;
    const ByteCount active_per_layer = bytes_per_expert * config.value().num_experts_per_token;
    const ByteCount no_cache_per_token = active_per_layer * config.value().num_hidden_layers;
    const ByteCount always_active = payload.value() - expert_bytes;
    const ByteCount full_context_kv = static_cast<ByteCount>(config.value().num_hidden_layers) *
                                      config.value().max_position_embeddings * config.value().hidden_size * 2U * sizeof(float);
    const ByteCount minimum_streaming = always_active + active_per_layer + full_context_kv + 1U * GiB;

    OlmoeInspection result;
    result.root = imported.value().root;
    result.model_id = imported.value().model_id;
    result.revision = imported.value().revision;
    result.config = config.value();
    result.metadata_only = imported.value().metadata_only;
    result.hashes_verified = imported.value().full_hash_verified;
    result.tensor_count = kTensorCount;
    result.imported_file_bytes = imported.value().total_bytes;
    result.tensor_payload_bytes = payload.value();
    result.routed_expert_bytes = expert_bytes;
    result.always_active_bytes = always_active;
    result.bytes_per_expert = bytes_per_expert;
    result.active_expert_bytes_per_layer = active_per_layer;
    result.no_cache_expert_bytes_per_token = no_cache_per_token;
    result.estimated_minimum_streaming_ram_bytes = minimum_streaming;
    result.safe_ram_cap_bytes = safe_ram_cap_bytes;
    result.resident_load_supported = !result.metadata_only && safe_ram_cap_bytes != 0U && payload.value() <= safe_ram_cap_bytes;
    result.streaming_plan_candidate = safe_ram_cap_bytes >= minimum_streaming;
    return result;
}

std::string to_json(const OlmoeInspection& inspection) {
    std::ostringstream output;
    output << "{\"ok\":true,\"model_id\":\"" << util::json_escape(inspection.model_id)
           << "\",\"revision\":\"" << util::json_escape(inspection.revision)
           << "\",\"architecture\":\"OlmoeForCausalLM\",\"metadata_only\":"
           << (inspection.metadata_only ? "true" : "false")
           << ",\"hashes_verified\":" << (inspection.hashes_verified ? "true" : "false")
           << ",\"layers\":" << inspection.config.num_hidden_layers
           << ",\"hidden_size\":" << inspection.config.hidden_size
           << ",\"attention_heads\":" << inspection.config.num_attention_heads
           << ",\"experts\":" << inspection.config.num_experts
           << ",\"experts_per_token\":" << inspection.config.num_experts_per_token
           << ",\"tensor_count\":" << inspection.tensor_count
           << ",\"tensor_payload_bytes\":" << inspection.tensor_payload_bytes
           << ",\"routed_expert_bytes\":" << inspection.routed_expert_bytes
           << ",\"always_active_bytes\":" << inspection.always_active_bytes
           << ",\"bytes_per_expert\":" << inspection.bytes_per_expert
           << ",\"active_expert_bytes_per_layer\":" << inspection.active_expert_bytes_per_layer
           << ",\"no_cache_expert_bytes_per_token\":" << inspection.no_cache_expert_bytes_per_token
           << ",\"estimated_minimum_streaming_ram_bytes\":" << inspection.estimated_minimum_streaming_ram_bytes
           << ",\"safe_ram_cap_bytes\":" << inspection.safe_ram_cap_bytes
           << ",\"resident_load_supported\":" << (inspection.resident_load_supported ? "true" : "false")
           << ",\"streaming_plan_candidate\":" << (inspection.streaming_plan_candidate ? "true" : "false") << '}';
    return output.str();
}

std::string to_text(const OlmoeInspection& inspection) {
    std::ostringstream output;
    output << "Model: " << inspection.model_id << '\n'
           << "Revision: " << inspection.revision << '\n'
           << "Architecture: 16 layers, width 2048, 16 heads, 64 experts, top-8\n"
           << "Checkpoint state: " << (inspection.metadata_only ? "verified metadata only" : "complete checkpoint") << '\n'
           << "Tensor payload: " << format_bytes(inspection.tensor_payload_bytes) << " in " << inspection.tensor_count << " tensors\n"
           << "Always-active weights: " << format_bytes(inspection.always_active_bytes) << '\n'
           << "Routed experts: " << format_bytes(inspection.routed_expert_bytes) << '\n'
           << "One expert: " << format_bytes(inspection.bytes_per_expert) << '\n'
           << "Active experts per layer: " << format_bytes(inspection.active_expert_bytes_per_layer) << '\n'
           << "No-cache expert traffic/token: " << format_bytes(inspection.no_cache_expert_bytes_per_token) << '\n'
           << "Estimated minimum streaming RAM: " << format_bytes(inspection.estimated_minimum_streaming_ram_bytes) << '\n'
           << "Safe host RAM cap: " << format_bytes(inspection.safe_ram_cap_bytes) << '\n'
           << "Resident load supported: " << (inspection.resident_load_supported ? "yes" : "no") << '\n'
           << "Streaming plan candidate: " << (inspection.streaming_plan_candidate ? "yes" : "no");
    return output.str();
}

}  // namespace stargate::inference
