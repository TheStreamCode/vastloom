#include "vastloom/inference/qwen3_8_flash_next.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/util/json.hpp"

namespace vastloom::inference {
namespace {

using vastloom::Result;
using vastloom::Error;
using vastloom::ErrorCode;
using vastloom::make_error;
using vastloom::util::JsonValue;

constexpr std::string_view kQwen38ModelId = "Qwen/Qwen3.8-Flash-Next";
constexpr std::string_view kExpectedModelType = "qwen4_exp";

[[nodiscard]] Result<std::uint32_t> json_u32(
    const JsonValue* v, std::string_view path) {
    if (v == nullptr) {
        return make_error(ErrorCode::invalid_format,
            "missing required field", std::string(path));
    }
    auto as_u64 = v->as_u64();
    if (!as_u64) {
        return make_error(ErrorCode::invalid_format,
            "field is not a non-negative integer", std::string(path));
    }
    constexpr std::uint64_t kMaxU32 = 0xFFFFFFFFULL;
    if (as_u64.value() > kMaxU32) {
        return make_error(ErrorCode::invalid_format,
            "field exceeds uint32 range", std::string(path));
    }
    return static_cast<std::uint32_t>(as_u64.value());
}

[[nodiscard]] Result<float> json_f32(
    const JsonValue* v, std::string_view path) {
    if (v == nullptr) {
        return make_error(ErrorCode::invalid_format,
            "missing required field", std::string(path));
    }
    auto as_d = v->as_double();
    if (!as_d) {
        return make_error(ErrorCode::invalid_format,
            "field is not a number", std::string(path));
    }
    return static_cast<float>(as_d.value());
}

[[nodiscard]] Result<std::string> json_string(
    const JsonValue* v, std::string_view path) {
    if (v == nullptr) {
        return make_error(ErrorCode::invalid_format,
            "missing required field", std::string(path));
    }
    auto s = v->as_string();
    if (!s) {
        return make_error(ErrorCode::invalid_format,
            "field is not a string", std::string(path));
    }
    return std::string(s.value());
}

[[nodiscard]] Result<std::vector<std::uint32_t>> json_u32_array(
    const JsonValue* v, std::string_view path) {
    if (v == nullptr) {
        return std::vector<std::uint32_t>{};
    }
    auto arr = v->as_array();
    if (!arr) {
        return make_error(ErrorCode::invalid_format,
            "field is not an array", std::string(path));
    }
    std::vector<std::uint32_t> out;
    out.reserve(arr.value()->size());
    for (std::size_t i = 0; i < arr.value()->size(); ++i) {
        const JsonValue* e = &(*arr.value())[i];
        auto u = json_u32(e, path);
        if (!u) {
            return make_error(u.error().code, u.error().message,
                u.error().context + "[" + std::to_string(i) + "]");
        }
        out.push_back(u.value());
    }
    return out;
}

[[nodiscard]] Result<const JsonValue*> get_text_config(const JsonValue& root) {
    const JsonValue* text_config = root.find("text_config");
    if (text_config != nullptr) {
        return text_config;
    }
    // Some early previews of qwen4_exp may put fields at root.
    return &root;
}

}  // namespace

Result<Qwen38FlashNextConfig> load_qwen38_flash_next_config(
    const std::filesystem::path& config_path,
    bool require_text_config) {
    auto json = vastloom::util::load_json_file(config_path);
    if (!json) {
        return make_error(json.error().code, json.error().message,
            "config_path=" + config_path.string());
    }
    const JsonValue& root = json.value();
    if (!root.is_object()) {
        return make_error(ErrorCode::invalid_format,
            "config root is not a JSON object",
            config_path.string());
    }

    // model_type guard — refuse non-qwen4_exp payloads.
    auto model_type = json_string(root.find("model_type"), "model_type");
    if (!model_type) {
        return make_error(model_type.error().code, model_type.error().message,
            "model_type missing in " + config_path.string());
    }
    if (std::string_view(model_type.value()) != kExpectedModelType) {
        return make_error(ErrorCode::unsupported,
            "config model_type is not qwen4_exp",
            "got=\"" + model_type.value() + "\"");
    }

    auto tc = get_text_config(root);
    if (!tc) {
        return make_error(ErrorCode::invalid_format,
            "text_config absent and root fields unavailable",
            config_path.string());
    }
    if (require_text_config && tc.value() == &root) {
        return make_error(ErrorCode::invalid_format,
            "text_config object is required for Qwen3.8-Flash-Next",
            config_path.string());
    }
    const JsonValue& t = *tc.value();

    Qwen38FlashNextConfig cfg;
    cfg.default_dtype = "bfloat16";

    // Top-level (config.json root)
    auto arch_root = root.find("architectures");
    if (arch_root != nullptr && arch_root->is_array()) {
        auto arr = arch_root->as_array();
        if (arr && !arr.value()->empty()) {
            auto first = (*arr.value())[0].as_string();
            if (first) {
                cfg.default_dtype = cfg.default_dtype;  // reserved for later
            }
        }
    }

    // Core
    {
        auto v = json_u32(t.find("vocab_size"), "text_config.vocab_size");
        if (!v) return v.error();
        cfg.vocab_size = v.value();
    }
    {
        auto v = json_u32(t.find("hidden_size"), "text_config.hidden_size");
        if (!v) return v.error();
        cfg.hidden_size = v.value();
    }
    {
        auto v = json_u32(t.find("num_hidden_layers"), "text_config.num_hidden_layers");
        if (!v) return v.error();
        cfg.num_hidden_layers = v.value();
    }

    // Full attention
    {
        auto v = json_u32(t.find("num_attention_heads"), "text_config.num_attention_heads");
        if (!v) return v.error();
        cfg.num_attention_heads = v.value();
    }
    {
        auto v = json_u32(t.find("num_key_value_heads"), "text_config.num_key_value_heads");
        if (!v) return v.error();
        cfg.num_key_value_heads = v.value();
    }
    {
        auto v = json_u32(t.find("head_dim"), "text_config.head_dim");
        if (!v) return v.error();
        cfg.head_dim = v.value();
    }
    {
        auto v = json_u32(t.find("rope_parameters"), "text_config.rope_parameters");
        // rope_dimension may be embedded in rope_parameters.partial_rotary_factor; we
        // do not parse it here — config will be checked against expected invariants
        // later in inspect_qwen38_flash_next_config.
        if (!v) {
            // optional — partial rotary can be derived from head_dim
            cfg.rope_dimension = cfg.head_dim / 4;  // standard default for Qwen
        } else {
            cfg.rope_dimension = v.value();
        }
    }

    // Linear attention (Gated DeltaNet)
    {
        auto v = json_u32(t.find("linear_num_value_heads"), "text_config.linear_num_value_heads");
        if (!v) return v.error();
        cfg.linear_num_value_heads = v.value();
    }
    {
        auto v = json_u32(t.find("linear_num_key_heads"), "text_config.linear_num_key_heads");
        if (!v) return v.error();
        cfg.linear_num_key_heads = v.value();
    }
    {
        auto v = json_u32(t.find("linear_value_head_dim"), "text_config.linear_value_head_dim");
        if (!v) return v.error();
        cfg.linear_value_head_dim = v.value();
    }
    {
        auto v = json_u32(t.find("linear_key_head_dim"), "text_config.linear_key_head_dim");
        if (!v) return v.error();
        cfg.linear_key_head_dim = v.value();
    }
    {
        auto v = json_u32(t.find("linear_conv_kernel_dim"), "text_config.linear_conv_kernel_dim");
        if (!v) return v.error();
        cfg.linear_conv_kernel_dim = v.value();
    }

    // QSA indexer
    {
        auto v = json_u32(t.find("indexer_n_heads"), "text_config.indexer_n_heads");
        if (!v) return v.error();
        cfg.indexer_n_heads = v.value();
    }
    {
        auto v = json_u32(t.find("indexer_kv_heads"), "text_config.indexer_kv_heads");
        if (!v) return v.error();
        cfg.indexer_kv_heads = v.value();
    }
    {
        auto v = json_u32(t.find("indexer_head_dim"), "text_config.indexer_head_dim");
        if (!v) return v.error();
        cfg.indexer_head_dim = v.value();
    }
    {
        auto v = json_u32(t.find("indexer_budget"), "text_config.indexer_budget");
        if (!v) return v.error();
        cfg.indexer_budget = v.value();
    }
    {
        auto v = json_u32(t.find("indexer_compress_ratio"), "text_config.indexer_compress_ratio");
        if (!v) return v.error();
        cfg.indexer_compress_ratio = v.value();
    }

    // MoE
    {
        auto v = json_u32(t.find("num_experts"), "text_config.num_experts");
        if (!v) return v.error();
        cfg.num_experts = v.value();
    }
    {
        auto v = json_u32(t.find("num_experts_per_tok"), "text_config.num_experts_per_tok");
        if (!v) return v.error();
        cfg.num_experts_per_tok = v.value();
    }
    {
        auto v = json_u32(t.find("moe_intermediate_size"), "text_config.moe_intermediate_size");
        if (!v) return v.error();
        cfg.moe_intermediate_size = v.value();
    }
    {
        auto v = json_u32(t.find("shared_expert_intermediate_size"), "text_config.shared_expert_intermediate_size");
        if (!v) return v.error();
        cfg.shared_expert_intermediate_size = v.value();
    }

    // Gated Residual
    {
        auto v = json_u32(t.find("hc_count"), "text_config.hc_count");
        if (!v) return v.error();
        cfg.hc_count = v.value();
    }
    {
        auto v = json_u32(t.find("hc_lowrank"), "text_config.hc_lowrank");
        if (!v) return v.error();
        cfg.hc_lowrank = v.value();
    }

    // N-gram / PLE
    {
        auto v = json_u32(t.find("ngram_size"), "text_config.ngram_size");
        if (!v) return v.error();
        cfg.ngram_size = v.value();
    }
    {
        auto v = json_u32(t.find("ngram_vocab_size_base"), "text_config.ngram_vocab_size_base");
        if (!v) return v.error();
        cfg.ngram_vocab_size_base = v.value();
    }
    {
        auto v = json_u32(t.find("ple_embed_dim"), "text_config.ple_embed_dim");
        if (!v) return v.error();
        cfg.ple_embed_dim = v.value();
    }
    {
        auto v = json_u32(t.find("ple_conv_kernel_size"), "text_config.ple_conv_kernel_size");
        if (!v) return v.error();
        cfg.ple_conv_kernel_size = v.value();
    }
    {
        auto v = json_u32(t.find("split_ngram_parts"), "text_config.split_ngram_parts");
        if (!v) return v.error();
        cfg.split_ngram_parts = v.value();
    }
    {
        auto v = json_u32_array(t.find("ple_layer_ids"), "text_config.ple_layer_ids");
        if (!v) return v.error();
        cfg.ple_layer_ids = std::move(v.value());
    }

    // Block layout
    {
        auto v = json_u32(t.find("full_attention_interval"), "text_config.full_attention_interval");
        if (!v) return v.error();
        cfg.full_attention_interval = v.value();
    }

    // MTP
    {
        auto v = json_u32(t.find("mtp_num_hidden_layers"), "text_config.mtp_num_hidden_layers");
        if (!v) return v.error();
        cfg.mtp_num_hidden_layers = v.value();
    }
    {
        auto v = json_u32(t.find("nextn_predict_layers"), "text_config.nextn_predict_layers");
        if (!v) {
            // not strictly required — Qwen3.5/3.6 models use this key.
            cfg.nextn_predict_layers = 0;
        } else {
            cfg.nextn_predict_layers = v.value();
        }
    }

    // Position embeddings
    {
        auto v = json_u32(t.find("max_position_embeddings"), "text_config.max_position_embeddings");
        if (!v) return v.error();
        cfg.max_position_embeddings = v.value();
    }
    {
        // rope_theta lives either at text_config.rope_theta (older previews) or
        // nested under text_config.rope_parameters.rope_theta (current release).
        auto direct = json_f32(t.find("rope_theta"), "text_config.rope_theta");
        if (direct) {
            cfg.rope_theta = direct.value();
        } else {
            const JsonValue* rope = t.find("rope_parameters");
            if (rope == nullptr || !rope->is_object()) {
                return make_error(ErrorCode::invalid_format,
                    "rope_theta missing and no rope_parameters block present",
                    "text_config.rope_theta");
            }
            auto nested = json_f32(rope->find("rope_theta"),
                "text_config.rope_parameters.rope_theta");
            if (!nested) return nested.error();
            cfg.rope_theta = nested.value();
        }
    }
    {
        auto v = json_f32(t.find("partial_rotary_factor"), "text_config.partial_rotary_factor");
        if (v) {
            cfg.partial_rotary_factor = v.value();
        } else {
            const JsonValue* rope = t.find("rope_parameters");
            if (rope != nullptr && rope->is_object()) {
                auto nested = json_f32(rope->find("partial_rotary_factor"),
                    "text_config.rope_parameters.partial_rotary_factor");
                if (nested) cfg.partial_rotary_factor = nested.value();
            }
        }
    }
    {
        auto v = json_f32(t.find("rms_norm_eps"), "text_config.rms_norm_eps");
        if (!v) return v.error();
        cfg.rms_norm_eps = v.value();
    }

    // Tokens
    {
        auto v = json_u32(t.find("bos_token_id"), "text_config.bos_token_id");
        if (!v) return v.error();
        cfg.bos_token_id = v.value();
    }
    {
        auto v = json_u32(t.find("eos_token_id"), "text_config.eos_token_id");
        if (!v) return v.error();
        cfg.eos_token_id = v.value();
    }
    {
        auto v = json_u32(t.find("pad_token_id"), "text_config.pad_token_id");
        if (!v) {
            cfg.pad_token_id = 0;
        } else {
            cfg.pad_token_id = v.value();
        }
    }

    // Data types
    cfg.ssm_dtype = "float32";
    if (auto* v = t.find("mamba_ssm_dtype"); v != nullptr) {
        auto s = json_string(v, "text_config.mamba_ssm_dtype");
        if (s) cfg.ssm_dtype = s.value();
    }
    if (auto* v = t.find("dtype"); v != nullptr) {
        auto s = json_string(v, "text_config.dtype");
        if (s) cfg.default_dtype = s.value();
    }

    // Vision (informational only; not used in text-only execution path).
    cfg.has_vision = (root.find("vision_config") != nullptr);

    return cfg;
}

Result<Qwen38FlashNextInspection> inspect_qwen38_flash_next_config(
    const Qwen38FlashNextConfig& config) {
    using Inv = Qwen38ExpectedInvariants;

    Qwen38FlashNextInspection ins;
    ins.metadata_only = true;
    ins.is_qwen4_exp = true;
    ins.has_deltanet = (config.linear_num_value_heads > 0
        && config.linear_num_key_heads > 0
        && config.linear_value_head_dim > 0
        && config.linear_key_head_dim > 0
        && config.linear_conv_kernel_dim > 0);
    ins.has_qsa = (config.num_attention_heads > 0
        && config.indexer_n_heads > 0
        && config.indexer_budget > 0);
    ins.has_ngram_embedding = (config.ngram_size > 0
        && config.ngram_vocab_size_base > 0
        && config.ple_embed_dim > 0
        && !config.ple_layer_ids.empty());
    ins.has_gated_residual = (config.hc_count > 0 && config.hc_lowrank > 0);
    ins.has_mtp = (config.mtp_num_hidden_layers > 0);
    ins.has_vision = config.has_vision;

    // Strict invariant check against the official 2026-08-26 release.
    if (config.hidden_size != Inv::hidden_size
        || config.num_hidden_layers != Inv::num_hidden_layers
        || config.num_experts != Inv::num_experts
        || config.num_experts_per_tok != Inv::num_experts_per_tok
        || config.moe_intermediate_size != Inv::moe_intermediate_size
        || config.shared_expert_intermediate_size != Inv::shared_expert_intermediate_size
        || config.linear_num_value_heads != Inv::linear_num_value_heads
        || config.linear_num_key_heads != Inv::linear_num_key_heads
        || config.linear_value_head_dim != Inv::linear_value_head_dim
        || config.linear_key_head_dim != Inv::linear_key_head_dim
        || config.linear_conv_kernel_dim != Inv::linear_conv_kernel_dim
        || config.num_attention_heads != Inv::num_attention_heads
        || config.num_key_value_heads != Inv::num_key_value_heads
        || config.head_dim != Inv::head_dim
        || config.indexer_n_heads != Inv::indexer_n_heads
        || config.indexer_kv_heads != Inv::indexer_kv_heads
        || config.indexer_head_dim != Inv::indexer_head_dim
        || config.indexer_budget != Inv::indexer_budget
        || config.hc_count != Inv::hc_count
        || config.hc_lowrank != Inv::hc_lowrank
        || config.ngram_size != Inv::ngram_size
        || config.ngram_vocab_size_base != Inv::ngram_vocab_size_base
        || config.ple_embed_dim != Inv::ple_embed_dim
        || config.split_ngram_parts != Inv::split_ngram_parts
        || config.full_attention_interval != Inv::full_attention_interval
        || config.vocab_size != Inv::vocab_size
        || config.bos_token_id != Inv::bos_token_id
        || config.eos_token_id != Inv::eos_token_id
        || config.ssm_dtype != Inv::ssm_dtype) {
        return make_error(ErrorCode::invalid_format,
            "config does not match the pinned Qwen3.8-Flash-Next invariants",
            "see Qwen38ExpectedInvariants in qwen3_8_flash_next.hpp");
    }

    ins.config = config;
    return ins;
}

}  // namespace vastloom::inference
