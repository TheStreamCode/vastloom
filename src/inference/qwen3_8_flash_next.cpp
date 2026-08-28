#include "vastloom/inference/qwen3_8_flash_next.hpp"
#include "vastloom/inference/qwen3_8_flash_next_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/util/json.hpp"
#include "vastloom/util/sha256.hpp"

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

Result<Qwen38ShardHeader> read_qwen38_safetensors_header(
    const std::filesystem::path& shard_path,
    std::uint64_t file_size_bytes,
    std::uint64_t json_header_bytes_to_read) {
    if (!std::filesystem::exists(shard_path)) {
        return make_error(ErrorCode::not_found,
            "shard file not found", shard_path.string());
    }
    std::ifstream in(shard_path, std::ios::binary);
    if (!in) {
        return make_error(ErrorCode::io_error,
            "cannot open shard file", shard_path.string());
    }

    // 8-byte little-endian length prefix.
    unsigned char lenbuf[8];
    in.read(reinterpret_cast<char*>(lenbuf), 8);
    if (in.gcount() != 8) {
        return make_error(ErrorCode::invalid_format,
            "shard too short to contain length prefix", shard_path.string());
    }
    const std::uint64_t header_len = static_cast<std::uint64_t>(lenbuf[0])
        | (static_cast<std::uint64_t>(lenbuf[1]) << 8)
        | (static_cast<std::uint64_t>(lenbuf[2]) << 16)
        | (static_cast<std::uint64_t>(lenbuf[3]) << 24)
        | (static_cast<std::uint64_t>(lenbuf[4]) << 32)
        | (static_cast<std::uint64_t>(lenbuf[5]) << 40)
        | (static_cast<std::uint64_t>(lenbuf[6]) << 48)
        | (static_cast<std::uint64_t>(lenbuf[7]) << 56);

    if (header_len == 0 || header_len > json_header_bytes_to_read) {
        return make_error(ErrorCode::invalid_format,
            "header length out of allowed range",
            "header_len=" + std::to_string(header_len)
                + " max=" + std::to_string(json_header_bytes_to_read));
    }

    std::string header_json(header_len, '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_len));
    if (static_cast<std::uint64_t>(in.gcount()) != header_len) {
        return make_error(ErrorCode::invalid_format,
            "short read of header JSON", shard_path.string());
    }

    auto json = vastloom::util::parse_json(header_json);
    if (!json) {
        return make_error(json.error().code, json.error().message,
            "safetensors header parse failed for " + shard_path.string());
    }
    const auto& root = json.value();
    if (!root.is_object()) {
        return make_error(ErrorCode::invalid_format,
            "safetensors header is not a JSON object", shard_path.string());
    }

    Qwen38ShardHeader hdr;
    hdr.path = shard_path;
    hdr.json_header_bytes = header_len;
    hdr.total_bytes = file_size_bytes;

    // Two-pass: first record all tensors, then attach block-scale companions.
    auto obj_res = root.as_object();
    if (!obj_res) {
        return make_error(ErrorCode::invalid_format,
            "header is not a JSON object", shard_path.string());
    }
    const auto* obj = obj_res.value();
    for (const auto& kv : *obj) {
        const std::string& name = kv.first;
        const auto& value = kv.second;
        if (name == "__metadata__") continue;
        const auto* tp = value.find("dtype");
        const auto* sp = value.find("shape");
        const auto* op = value.find("data_offsets");
        if (tp == nullptr || sp == nullptr || op == nullptr) {
            return make_error(ErrorCode::invalid_format,
                "tensor descriptor missing required fields",
                "name=" + name);
        }
        auto dtype = tp->as_string();
        auto shape = sp->as_array();
        auto offsets = op->as_array();
        if (!dtype || !shape || !offsets) {
            return make_error(ErrorCode::invalid_format,
                "tensor descriptor has wrong types",
                "name=" + name);
        }
        Qwen38TensorDescriptor desc;
        desc.name = name;
        desc.dtype = std::string(dtype.value());
        desc.shape.reserve(shape.value()->size());
        for (const auto& dim : *shape.value()) {
            auto d = dim.as_i64();
            if (!d) {
                return make_error(ErrorCode::invalid_format,
                    "tensor shape dim not int",
                    "name=" + name);
            }
            desc.shape.push_back(d.value());
        }
        auto begin = (*offsets.value())[0].as_u64();
        auto end = (*offsets.value())[1].as_u64();
        if (!begin || !end) {
            return make_error(ErrorCode::invalid_format,
                "tensor data_offsets not uint",
                "name=" + name);
        }
        desc.data_offset_begin = begin.value();
        desc.data_offset_end = end.value();
        hdr.tensors.push_back(std::move(desc));
    }

    // Attach block-scale companions: pair each `*_scale_inv` with the matching
    // `*` tensor of the same base name. Qwen3.8-Flash-Next's FP8 block-scale
    // uses this exact suffix convention for expert weights
    // (`up_proj.weight` + `up_proj.weight_scale_inv`).
    // We need to attach onto the matching non-scale tensor, which means we
    // need a non-const view of the tensors. Iterate twice: first locate the
    // companions, then patch via index.
    for (std::size_t i = 0; i < hdr.tensors.size(); ++i) {
        const std::string& name = hdr.tensors[i].name;
        const std::string suffix = "_scale_inv";
        if (name.size() <= suffix.size()) continue;
        if (std::string_view(name).substr(name.size() - suffix.size())
            != std::string_view(suffix)) continue;
        std::string base_name = name.substr(0, name.size() - suffix.size());
        for (std::size_t j = 0; j < hdr.tensors.size(); ++j) {
            if (i == j) continue;
            if (hdr.tensors[j].name == base_name) {
                hdr.tensors[j].scale_inv_name = name;
                hdr.tensors[j].scale_inv_offset_begin = hdr.tensors[i].data_offset_begin;
                hdr.tensors[j].scale_inv_offset_end = hdr.tensors[i].data_offset_end;
                break;
            }
        }
    }

    return hdr;
}

std::vector<Qwen38LayerSummary> qwen38_summarize_layers(
    const std::vector<Qwen38ShardIndexEntry>& weight_map) {
    std::vector<Qwen38LayerSummary> layers(48);
    for (std::size_t i = 0; i < 48; ++i) layers[i].layer = i;
    for (const auto& e : weight_map) {
        const std::string& name = e.tensor_name;
        if (name.find("language_model.layers.") == std::string::npos) continue;
        // Extract layer index: "model.language_model.layers.<N>."
        const std::string marker = "language_model.layers.";
        auto pos = name.find(marker);
        if (pos == std::string::npos) continue;
        pos += marker.size();
        auto dot = name.find('.', pos);
        if (dot == std::string::npos) continue;
        int layer = 0;
        try {
            layer = std::stoi(name.substr(pos, dot - pos));
        } catch (...) { continue; }
        if (layer < 0 || layer >= 48) continue;
        auto& s = layers[layer];
        ++s.total;
        if (name.find("linear_attn.") != std::string::npos) ++s.linear_attn;
        else if (name.find("self_attn.") != std::string::npos) ++s.self_attn;
        if (name.find("attn_hyper_connection.") != std::string::npos) ++s.attn_hyper;
        if (name.find("mlp_hyper_connection.") != std::string::npos) ++s.mlp_hyper;
        if (name.find("mlp.gate.weight") != std::string::npos) ++s.mlp_gate;
        if (name.find("mlp.shared_expert_gate") != std::string::npos) ++s.mlp_shared_gate;
        if (name.find("mlp.shared_expert.") != std::string::npos) ++s.mlp_shared_expert;
        if (name.find("mlp.experts.") != std::string::npos) ++s.mlp_experts;
        if (name.find("ple") != std::string::npos) ++s.ple;
    }
    return layers;
}

Result<Qwen38InspectionResult> qwen38_validate_catalog(
    const Qwen38InspectionResult& result) {
    using Exp = Qwen38ExpectedCatalog;
    if (result.total_tensors != Exp::total_tensors) {
        return make_error(ErrorCode::invalid_format,
            "catalog total_tensors mismatch",
            "expected=" + std::to_string(Exp::total_tensors) + " got=" + std::to_string(result.total_tensors));
    }
    if (result.total_shards != Exp::total_shards) {
        return make_error(ErrorCode::invalid_format,
            "catalog total_shards mismatch",
            "expected=" + std::to_string(Exp::total_shards) + " got=" + std::to_string(result.total_shards));
    }
    if (result.fp8_block_scaled_experts != Exp::fp8_expert_scale_inv) {
        return make_error(ErrorCode::invalid_format,
            "catalog fp8 scale_inv count mismatch",
            "expected=" + std::to_string(Exp::fp8_expert_scale_inv) + " got=" + std::to_string(result.fp8_block_scaled_experts));
    }
    if (result.ngram_tensor_count != Exp::ngram_tensors) {
        return make_error(ErrorCode::invalid_format,
            "catalog ngram tensor count mismatch",
            "expected=" + std::to_string(Exp::ngram_tensors) + " got=" + std::to_string(result.ngram_tensor_count));
    }
    if (result.layers.size() != Exp::language_layers) {
        return make_error(ErrorCode::invalid_format,
            "catalog layer count mismatch",
            "expected=" + std::to_string(Exp::language_layers) + " got=" + std::to_string(result.layers.size()));
    }
    for (std::size_t i = 0; i < result.layers.size(); ++i) {
        const auto& s = result.layers[i];
        const std::size_t expected = (i == 1 ? Exp::per_layer_linear + Exp::per_layer_ple_extra : Exp::per_layer_linear);
        // Both linear and full layers have 3094; layer 1 is 3232.
        if (s.total != expected) {
            return make_error(ErrorCode::invalid_format,
                "catalog per-layer total mismatch",
                "layer=" + std::to_string(i) + " expected=" + std::to_string(expected) + " got=" + std::to_string(s.total));
        }
    }
    Qwen38InspectionResult copy = result;
    return copy;
}

Result<Qwen38InspectionResult> inspect_qwen38_flash_next_safetensors(
    const std::filesystem::path& index_json_path,
    const Qwen38FlashNextConfig& config,
    bool load_first_shard_header) {
    if (!std::filesystem::exists(index_json_path)) {
        return make_error(ErrorCode::not_found,
            "index.json not found", index_json_path.string());
    }
    auto json = vastloom::util::load_json_file(index_json_path);
    if (!json) {
        return make_error(json.error().code, json.error().message,
            "failed to parse index.json: " + index_json_path.string());
    }
    const auto& root = json.value();
    auto* wm = root.find("weight_map");
    auto* meta = root.find("metadata");
    if (wm == nullptr || !wm->is_object()) {
        return make_error(ErrorCode::invalid_format,
            "weight_map missing or not an object", index_json_path.string());
    }
    auto wm_obj = wm->as_object();
    if (!wm_obj) {
        return make_error(ErrorCode::invalid_format,
            "weight_map not a JSON object", index_json_path.string());
    }
    std::ignore = meta;

    Qwen38InspectionResult result;
    result.root = index_json_path.parent_path();
    result.config = config;
    result.weight_map.reserve(wm_obj.value()->size());
    result.total_tensors = wm_obj.value()->size();
    std::set<std::string> unique_shards;
    std::size_t fp8_experts = 0;
    std::size_t ngram_count = 0;
    bool has_global_hc = false;
    bool has_layer_hc = false;
    for (const auto& kv : *wm_obj.value()) {
        const std::string& name = kv.first;
        auto shard_val = kv.second.as_string();
        if (!shard_val) {
            return make_error(ErrorCode::invalid_format,
                "weight_map entry has non-string shard",
                "name=" + name);
        }
        Qwen38ShardIndexEntry e;
        e.tensor_name = name;
        e.shard_path = std::string(shard_val.value());
        result.weight_map.push_back(std::move(e));
        unique_shards.insert(std::string(shard_val.value()));
        if (name.find("mlp.experts.") != std::string::npos
            && name.find("_scale_inv") != std::string::npos) {
            ++fp8_experts;
        }
        if (name.find("ngram_embedding") != std::string::npos) {
            ++ngram_count;
        }
        if (name.find("hyper_connection_mixer") != std::string::npos) {
            has_global_hc = true;
        }
        if (name.find("attn_hyper_connection") != std::string::npos
            || name.find("mlp_hyper_connection") != std::string::npos) {
            has_layer_hc = true;
        }
    }
    result.total_shards = unique_shards.size();
    result.fp8_block_scaled_experts = fp8_experts;
    result.ngram_tensor_count = ngram_count;
    result.has_gated_residual_global = has_global_hc;
    result.has_gated_residual_per_layer = has_layer_hc;
    result.layers = qwen38_summarize_layers(result.weight_map);
    result.vision_tensors = 0;
    result.mtp_tensors = 0;
    result.global_tensors = 0;
    for (const auto& e : result.weight_map) {
        const std::string& n = e.tensor_name;
        if (n.find("visual.") != std::string::npos) ++result.vision_tensors;
        else if (n.rfind("mtp.", 0) == 0) ++result.mtp_tensors;
        else if (n.find("language_model.layers.") == std::string::npos) ++result.global_tensors;
    }

    if (load_first_shard_header && !unique_shards.empty()) {
        // First shard alphabetically.
        const std::string first_shard = *unique_shards.begin();
        std::filesystem::path shard_full = result.root / first_shard;
        std::error_code ec;
        auto size = std::filesystem::file_size(shard_full, ec);
        if (ec) {
            return make_error(ErrorCode::io_error,
                "cannot stat first shard",
                shard_full.string() + ": " + ec.message());
        }
        auto hdr = read_qwen38_safetensors_header(shard_full, size);
        if (!hdr) {
            return make_error(hdr.error().code, hdr.error().message,
                "first shard header parse failed: " + shard_full.string());
        }
        result.shard_header = hdr.value();
        result.shard_loaded = true;
    }

    return result;
}

Result<Qwen38TensorPayload> qwen38_read_tensor_payload(
    const std::filesystem::path& shard_path,
    const Qwen38ShardHeader& header,
    std::string_view tensor_name) {
    const Qwen38TensorDescriptor* desc = nullptr;
    for (const auto& t : header.tensors) {
        if (t.name == tensor_name) { desc = &t; break; }
    }
    if (desc == nullptr) {
        return make_error(ErrorCode::not_found,
            "tensor not present in shard header", std::string(tensor_name));
    }

    const std::uint64_t payload_offset = 8U + header.json_header_bytes;
    const std::uint64_t weight_begin = payload_offset + desc->data_offset_begin;
    const std::uint64_t weight_end = payload_offset + desc->data_offset_end;
    if (weight_end > header.total_bytes || weight_begin > weight_end) {
        return make_error(ErrorCode::invalid_format,
            "weight range out of bounds",
            "tensor=" + std::string(tensor_name)
                + " range=[" + std::to_string(weight_begin) + "," + std::to_string(weight_end)
                + "] file_size=" + std::to_string(header.total_bytes));
    }

    std::ifstream in(shard_path, std::ios::binary);
    if (!in) {
        return make_error(ErrorCode::io_error,
            "cannot open shard file", shard_path.string());
    }

    Qwen38TensorPayload payload;
    payload.tensor_name = std::string(tensor_name);
    payload.dtype = desc->dtype;
    payload.payload_bytes = desc->data_offset_end - desc->data_offset_begin;

    in.seekg(static_cast<std::streamoff>(weight_begin));
    payload.bytes.resize(static_cast<std::size_t>(payload.payload_bytes));
    in.read(reinterpret_cast<char*>(payload.bytes.data()),
            static_cast<std::streamsize>(payload.bytes.size()));
    if (static_cast<std::uint64_t>(in.gcount()) != payload.bytes.size()) {
        return make_error(ErrorCode::io_error,
            "short read of weight payload", std::string(tensor_name));
    }

    // Optional scale_inv companion: also payload-relative offsets.
    if (!desc->scale_inv_name.empty()) {
        const std::uint64_t sb = payload_offset + desc->scale_inv_offset_begin;
        const std::uint64_t se = payload_offset + desc->scale_inv_offset_end;
        if (se > header.total_bytes || sb > se) {
            return make_error(ErrorCode::invalid_format,
                "scale range out of bounds",
                "tensor=" + std::string(tensor_name));
        }
        in.seekg(static_cast<std::streamoff>(sb));
        payload.scale_bytes.resize(static_cast<std::size_t>(se - sb));
        in.read(reinterpret_cast<char*>(payload.scale_bytes.data()),
                static_cast<std::streamsize>(payload.scale_bytes.size()));
        if (static_cast<std::uint64_t>(in.gcount()) != payload.scale_bytes.size()) {
            return make_error(ErrorCode::io_error,
                "short read of scale payload", std::string(tensor_name));
        }
    }

    vastloom::util::Sha256 hasher;
    hasher.update(payload.bytes);
    payload.sha256_hex = hasher.finalize_hex();

    return payload;
}

Result<Qwen38TensorImportResult> qwen38_import_tensor_transaction(
    const Qwen38TensorImportRequest& request,
    const Qwen38ShardHeader& header,
    safety::ResourceGovernor& governor) {
    if (!std::filesystem::exists(request.source_shard)) {
        return make_error(ErrorCode::not_found,
            "source shard missing", request.source_shard.string());
    }
    if (!std::filesystem::is_directory(request.destination_dir)) {
        return make_error(ErrorCode::not_found,
            "destination dir missing", request.destination_dir.string());
    }
    if (!vastloom::util::is_sha256_hex(request.expected_sha256)) {
        return make_error(ErrorCode::invalid_argument,
            "expected_sha256 is not valid hex", request.expected_sha256);
    }

    auto loaded = qwen38_read_tensor_payload(
        request.source_shard, header, request.tensor_name);
    if (!loaded) return loaded.error();

    // Object (2): hash verification before any write.
    if (loaded.value().sha256_hex != request.expected_sha256) {
        return make_error(ErrorCode::integrity_error,
            "SHA-256 mismatch on tensor payload",
            "tensor=" + request.tensor_name
                + " expected=" + request.expected_sha256
                + " got=" + loaded.value().sha256_hex);
    }

    // Governor must admit the aggregate payload (weight + scale) as RAM.
    const ByteCount aggregate = loaded.value().bytes.size() + loaded.value().scale_bytes.size();
    auto reservation = governor.try_reserve(
        safety::ResourceKind::ram, aggregate,
        "qwen38_import:" + request.tensor_name);
    if (!reservation) {
        return make_error(reservation.error().code, reservation.error().message,
            "governor refused qwen38 import of " + request.tensor_name);
    }

    // Object (3): bounded write to temp then atomic rename.
    const std::filesystem::path artifact = request.destination_dir /
        (request.artifact_name + ".bin");
    const std::filesystem::path tmp = request.destination_dir /
        ("." + request.artifact_name + ".tmp" + std::to_string(
            static_cast<unsigned long long>(std::chrono::steady_clock::now()
                .time_since_epoch().count())));

    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return make_error(ErrorCode::io_error,
                "cannot create temp artifact", tmp.string());
        }
        out.write(reinterpret_cast<const char*>(loaded.value().bytes.data()),
                  static_cast<std::streamsize>(loaded.value().bytes.size()));
        if (!loaded.value().scale_bytes.empty()) {
            out.write(reinterpret_cast<const char*>(loaded.value().scale_bytes.data()),
                      static_cast<std::streamsize>(loaded.value().scale_bytes.size()));
        }
        out.flush();
        if (!out) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return make_error(ErrorCode::io_error,
                "failed writing temp artifact", tmp.string());
        }
    }

    std::error_code ec;
    if (std::filesystem::exists(artifact)) {
        std::filesystem::remove(tmp, ec);
        return make_error(ErrorCode::resource_exhausted,
            "destination artifact already exists",
            artifact.string());
    }
    std::filesystem::rename(tmp, artifact, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return make_error(ErrorCode::io_error,
            "atomic rename failed", artifact.string() + ": " + ec.message());
    }

    Qwen38TensorImportResult result;
    result.tensor_name = request.tensor_name;
    result.artifact_path = artifact;
    result.payload_bytes = loaded.value().bytes.size();
    result.scale_bytes = loaded.value().scale_bytes.size();
    result.sha256_hex = loaded.value().sha256_hex;
    result.scale_included = !loaded.value().scale_bytes.empty();
    return result;
}

bool qwen38_is_qsa_layer(std::uint32_t layer, std::uint32_t full_attention_interval) noexcept {
    return full_attention_interval > 0U && ((layer + 1U) % full_attention_interval) == 0U;
}

namespace {

void append_planned(
    std::vector<Qwen38PlannedTensor>& out,
    const std::string& prefix,
    std::string_view suffix,
    Qwen38Consumer consumer) {
    Qwen38PlannedTensor t;
    t.name = prefix;
    t.name.append(suffix);
    t.consumer = consumer;
    out.push_back(std::move(t));
}

void append_common_non_routed(std::vector<Qwen38PlannedTensor>& out, const std::string& prefix) {
    append_planned(out, prefix, "attn_hyper_connection.hc_norm.weight", Qwen38Consumer::gated_residual_attn);
    append_planned(out, prefix, "attn_hyper_connection.input_mix_weight_down.weight", Qwen38Consumer::gated_residual_attn);
    append_planned(out, prefix, "attn_hyper_connection.input_mix_weight_up.weight", Qwen38Consumer::gated_residual_attn);
    append_planned(out, prefix, "attn_hyper_connection.block_inject_weight.weight", Qwen38Consumer::gated_residual_attn);
    append_planned(out, prefix, "mlp_hyper_connection.hc_norm.weight", Qwen38Consumer::gated_residual_mlp);
    append_planned(out, prefix, "mlp_hyper_connection.input_mix_weight_down.weight", Qwen38Consumer::gated_residual_mlp);
    append_planned(out, prefix, "mlp_hyper_connection.input_mix_weight_up.weight", Qwen38Consumer::gated_residual_mlp);
    append_planned(out, prefix, "mlp_hyper_connection.block_inject_weight.weight", Qwen38Consumer::gated_residual_mlp);
    append_planned(out, prefix, "mlp.gate.weight", Qwen38Consumer::moe_router);
    append_planned(out, prefix, "mlp.shared_expert_gate.weight", Qwen38Consumer::moe_shared);
    append_planned(out, prefix, "mlp.shared_expert.gate_proj.weight", Qwen38Consumer::moe_shared);
    append_planned(out, prefix, "mlp.shared_expert.up_proj.weight", Qwen38Consumer::moe_shared);
    append_planned(out, prefix, "mlp.shared_expert.down_proj.weight", Qwen38Consumer::moe_shared);
}

void append_delta_net(std::vector<Qwen38PlannedTensor>& out, const std::string& prefix) {
    append_planned(out, prefix, "linear_attn.A_log", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.conv1d.weight", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.dt_bias", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.in_proj_a.weight", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.in_proj_b.weight", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.in_proj_qkv.weight", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.in_proj_z.weight", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.norm.weight", Qwen38Consumer::delta_net);
    append_planned(out, prefix, "linear_attn.out_proj.weight", Qwen38Consumer::delta_net);
}

void append_qsa(std::vector<Qwen38PlannedTensor>& out, const std::string& prefix) {
    append_planned(out, prefix, "self_attn.indexer.index_qk_proj.weight", Qwen38Consumer::qsa_indexer);
    append_planned(out, prefix, "self_attn.indexer.q_layernorm.weight", Qwen38Consumer::qsa_indexer);
    append_planned(out, prefix, "self_attn.indexer.k_layernorm.weight", Qwen38Consumer::qsa_indexer);
    append_planned(out, prefix, "self_attn.q_proj.weight", Qwen38Consumer::qsa_attention);
    append_planned(out, prefix, "self_attn.k_proj.weight", Qwen38Consumer::qsa_attention);
    append_planned(out, prefix, "self_attn.v_proj.weight", Qwen38Consumer::qsa_attention);
    append_planned(out, prefix, "self_attn.o_proj.weight", Qwen38Consumer::qsa_attention);
    append_planned(out, prefix, "self_attn.q_norm.weight", Qwen38Consumer::qsa_attention);
    append_planned(out, prefix, "self_attn.k_norm.weight", Qwen38Consumer::qsa_attention);
}

void append_ple(std::vector<Qwen38PlannedTensor>& out, const std::string& prefix, std::uint32_t parts) {
    append_planned(out, prefix, "ple.conv1d.weight", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.key_proj.weight", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.value_proj.weight", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.norm_conv.weight", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.norm_key.weight", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.norm_query.weight", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.ple_embedding.layer_multipliers", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.ple_embedding.ngram_embedding.weight_scale", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.ple_embedding.ngram_heads_offsets", Qwen38Consumer::ngram_ple);
    append_planned(out, prefix, "ple.ple_embedding.ngram_heads_vocab_sizes", Qwen38Consumer::ngram_ple);
    for (std::uint32_t i = 0; i < parts; ++i) {
        append_planned(out, prefix,
            "ple.ple_embedding.ngram_embedding.shard_" + std::to_string(i) + ".weight",
            Qwen38Consumer::ngram_ple);
    }
}

bool layer_has_ple(const Qwen38FlashNextConfig& config, std::uint32_t layer) {
    const std::uint32_t one_based = layer + 1U;
    return std::find(config.ple_layer_ids.begin(), config.ple_layer_ids.end(), one_based)
        != config.ple_layer_ids.end();
}

}  // namespace

Result<Qwen38GraphPlan> qwen38_plan_non_routed_graph(const Qwen38FlashNextConfig& config) {
    if (config.num_hidden_layers == 0U || config.full_attention_interval == 0U) {
        return make_error(ErrorCode::invalid_argument,
            "graph plan requires hidden layers and full_attention_interval", "qwen38_plan");
    }
    if (config.mtp_num_hidden_layers != 1U) {
        return make_error(ErrorCode::invalid_argument,
            "graph plan requires exactly one MTP layer",
            "mtp_num_hidden_layers=" + std::to_string(config.mtp_num_hidden_layers));
    }

    Qwen38GraphPlan plan;
    plan.language_layers = config.num_hidden_layers;
    plan.mtp_layers = 1;
    plan.layers.reserve(static_cast<std::size_t>(config.num_hidden_layers) + 1U);

    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        Qwen38LayerPlan lp;
        lp.layer = layer;
        lp.is_mtp = false;
        lp.has_ple = layer_has_ple(config, layer);
        lp.kind = qwen38_is_qsa_layer(layer, config.full_attention_interval)
            ? Qwen38LayerKind::qwen_sparse_attention
            : Qwen38LayerKind::linear_attention;
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
        if (lp.kind == Qwen38LayerKind::qwen_sparse_attention) {
            append_qsa(lp.tensors, prefix);
            ++plan.qsa_layers;
        } else {
            append_delta_net(lp.tensors, prefix);
            ++plan.linear_layers;
        }
        append_common_non_routed(lp.tensors, prefix);
        if (lp.has_ple) {
            append_ple(lp.tensors, prefix, config.split_ngram_parts);
        }
        plan.non_routed_tensor_count += lp.tensors.size();
        plan.layers.push_back(std::move(lp));
    }

    Qwen38LayerPlan mtp;
    mtp.layer = 0;
    mtp.is_mtp = true;
    mtp.kind = Qwen38LayerKind::qwen_sparse_attention;
    const std::string mtp_prefix = "mtp.layers.0.";
    append_qsa(mtp.tensors, mtp_prefix);
    append_common_non_routed(mtp.tensors, mtp_prefix);
    append_planned(mtp.tensors, "mtp.", "fc_embedding.weight", Qwen38Consumer::mtp_head);
    append_planned(mtp.tensors, "mtp.", "fc_hidden.weight", Qwen38Consumer::mtp_head);
    append_planned(mtp.tensors, "mtp.", "pre_fc_norm_embedding.weight", Qwen38Consumer::mtp_head);
    append_planned(mtp.tensors, "mtp.", "pre_fc_norm_hidden.weight", Qwen38Consumer::mtp_head);
    append_planned(mtp.tensors, "mtp.", "hyper_connection_mixer.hc_norm.weight", Qwen38Consumer::global_mixer);
    append_planned(mtp.tensors, "mtp.", "hyper_connection_mixer.input_mix_weight_down.weight", Qwen38Consumer::global_mixer);
    append_planned(mtp.tensors, "mtp.", "hyper_connection_mixer.input_mix_weight_up.weight", Qwen38Consumer::global_mixer);
    plan.non_routed_tensor_count += mtp.tensors.size();
    plan.layers.push_back(std::move(mtp));

    append_planned(plan.globals, "", "model.language_model.embed_tokens.weight", Qwen38Consumer::global_embed);
    append_planned(plan.globals, "", "lm_head.weight", Qwen38Consumer::global_lm_head);
    append_planned(plan.globals, "", "model.language_model.hyper_connection_mixer.hc_norm.weight", Qwen38Consumer::global_mixer);
    append_planned(plan.globals, "", "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", Qwen38Consumer::global_mixer);
    append_planned(plan.globals, "", "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", Qwen38Consumer::global_mixer);
    plan.non_routed_tensor_count += plan.globals.size();
    return plan;
}

Result<Qwen38GraphPlan> qwen38_validate_non_routed_plan(const Qwen38GraphPlan& plan) {
    using Exp = Qwen38ExpectedCatalog;
    if (plan.language_layers != Exp::language_layers || plan.layers.size() != Exp::language_layers + Exp::mtp_layers) {
        return make_error(ErrorCode::invalid_format,
            "non-routed plan layer count mismatch",
            "layers=" + std::to_string(plan.layers.size()));
    }
    if (plan.linear_layers != Exp::linear_layers || plan.qsa_layers != Exp::qsa_layers || plan.mtp_layers != Exp::mtp_layers) {
        return make_error(ErrorCode::invalid_format,
            "non-routed plan kind counts mismatch",
            "linear=" + std::to_string(plan.linear_layers) + " qsa=" + std::to_string(plan.qsa_layers));
    }
    if (plan.non_routed_tensor_count != Exp::text_non_routed || plan.globals.size() != Exp::global_tensors) {
        return make_error(ErrorCode::invalid_format,
            "non-routed plan tensor count mismatch",
            "count=" + std::to_string(plan.non_routed_tensor_count));
    }
    std::size_t language_tensors = 0;
    for (std::size_t i = 0; i < Exp::language_layers; ++i) {
        const auto& lp = plan.layers[i];
        if (lp.is_mtp || lp.layer != i) {
            return make_error(ErrorCode::invalid_format, "language layer slot mismatch", std::to_string(i));
        }
        const bool expect_qsa = qwen38_is_qsa_layer(static_cast<std::uint32_t>(i), 4);
        if ((lp.kind == Qwen38LayerKind::qwen_sparse_attention) != expect_qsa) {
            return make_error(ErrorCode::invalid_format, "layer kind mismatch", std::to_string(i));
        }
        const std::size_t expected = Exp::per_layer_non_routed + (i == 1 ? Exp::per_layer_ple_extra : 0);
        if (lp.tensors.size() != expected || lp.has_ple != (i == 1)) {
            return make_error(ErrorCode::invalid_format,
                "per-layer non-routed count mismatch",
                "layer=" + std::to_string(i) + " got=" + std::to_string(lp.tensors.size()));
        }
        language_tensors += lp.tensors.size();
    }
    const auto& mtp = plan.layers.back();
    if (!mtp.is_mtp || mtp.kind != Qwen38LayerKind::qwen_sparse_attention
        || mtp.tensors.size() != Exp::mtp_non_routed) {
        return make_error(ErrorCode::invalid_format,
            "MTP non-routed plan mismatch",
            "count=" + std::to_string(mtp.tensors.size()));
    }
    if (language_tensors != Exp::language_non_routed) {
        return make_error(ErrorCode::invalid_format,
            "language non-routed total mismatch",
            std::to_string(language_tensors));
    }
    return plan;
}

Result<Qwen38GraphPlan> qwen38_bind_non_routed_plan(
    const Qwen38GraphPlan& plan,
    const std::vector<Qwen38ShardIndexEntry>& weight_map) {
    std::unordered_map<std::string, std::filesystem::path> by_name;
    by_name.reserve(weight_map.size());
    for (const auto& e : weight_map) {
        auto [it, inserted] = by_name.emplace(e.tensor_name, e.shard_path);
        if (!inserted) {
            return make_error(ErrorCode::invalid_format,
                "duplicate tensor in weight_map", e.tensor_name);
        }
    }
    Qwen38GraphPlan bound = plan;
    auto bind_one = [&](Qwen38PlannedTensor& t) -> Result<int> {
        auto it = by_name.find(t.name);
        if (it == by_name.end()) {
            return make_error(ErrorCode::not_found,
                "planned non-routed tensor missing from catalog", t.name);
        }
        t.shard_path = it->second;
        t.bound = true;
        return 0;
    };
    for (auto& layer : bound.layers) {
        for (auto& t : layer.tensors) {
            auto ok = bind_one(t);
            if (!ok) {
                return ok.error();
            }
        }
    }
    for (auto& t : bound.globals) {
        auto ok = bind_one(t);
        if (!ok) {
            return ok.error();
        }
    }
    return bound;
}

Result<Qwen38PersistBudget> qwen38_plan_persist_budget(
    const Qwen38FlashNextConfig& config,
    ByteCount ngram_working_set) {
    if (config.num_hidden_layers == 0 || config.full_attention_interval == 0
        || config.linear_num_value_heads == 0 || config.linear_num_key_heads == 0
        || config.linear_value_head_dim == 0 || config.linear_key_head_dim == 0
        || config.linear_conv_kernel_dim < 2) {
        return make_error(ErrorCode::invalid_argument, "persist budget config incomplete", "persist");
    }
    std::uint32_t linear = 0;
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        if (!qwen38_is_qsa_layer(i, config.full_attention_interval)) {
            ++linear;
        }
    }
    const ByteCount key_dim = static_cast<ByteCount>(config.linear_key_head_dim) * config.linear_num_key_heads;
    const ByteCount value_dim = static_cast<ByteCount>(config.linear_value_head_dim) * config.linear_num_value_heads;
    const ByteCount conv_dim = key_dim * 2U + value_dim;
    const ByteCount conv_elems = conv_dim * (config.linear_conv_kernel_dim - 1U);
    const ByteCount ssm_elems = static_cast<ByteCount>(config.linear_num_value_heads)
        * config.linear_key_head_dim * config.linear_value_head_dim;
    Qwen38PersistBudget budget;
    budget.linear_layers = linear;
    budget.deltanet_conv_bytes = conv_elems * sizeof(float) * linear;
    budget.deltanet_ssm_bytes = ssm_elems * sizeof(float) * linear;
    budget.deltanet_bytes = budget.deltanet_conv_bytes + budget.deltanet_ssm_bytes;
    budget.ngram_working_set = ngram_working_set;
    budget.total_resident = budget.deltanet_bytes + ngram_working_set;
    budget.under_reference_ram_gate = budget.total_resident <= qwen38_reference_live_ram_gate;
    return budget;
}

Result<Qwen38DeltaNetPersist> qwen38_deltanet_persist_create(
    const Qwen38FlashNextConfig& config,
    safety::ResourceGovernor& governor) {
    auto budget = qwen38_plan_persist_budget(config, 0);
    if (!budget) {
        return budget.error();
    }
    if (budget.value().deltanet_bytes == 0) {
        return make_error(ErrorCode::invalid_argument, "no DeltaNet layers to persist", "persist");
    }
    auto ram = governor.try_reserve(safety::ResourceKind::ram, budget.value().deltanet_bytes, "qwen38_deltanet_persist");
    if (!ram) {
        return ram.error();
    }
    const std::size_t key_dim = static_cast<std::size_t>(config.linear_key_head_dim) * config.linear_num_key_heads;
    const std::size_t value_dim = static_cast<std::size_t>(config.linear_value_head_dim) * config.linear_num_value_heads;
    const std::size_t conv_dim = key_dim * 2U + value_dim;
    const std::size_t conv_elems = conv_dim * (config.linear_conv_kernel_dim - 1U);
    const std::size_t ssm_elems = static_cast<std::size_t>(config.linear_num_value_heads)
        * config.linear_key_head_dim * config.linear_value_head_dim;
    Qwen38DeltaNetPersist persist;
    persist.linear_layers = budget.value().linear_layers;
    persist.layers.resize(persist.linear_layers);
    for (auto& layer : persist.layers) {
        layer.conv.assign(conv_elems, 0.0F);
        layer.ssm.assign(ssm_elems, 0.0F);
    }
    persist.ram = std::move(ram.value());
    return persist;
}

namespace {

Result<void> load_ngram_page(Qwen38NgramPageCache& cache, std::uint32_t page, bool prefetch) {
    const std::uint32_t num_pages = (cache.rows + cache.rows_per_page - 1U) / cache.rows_per_page;
    if (page >= num_pages) {
        return make_error(ErrorCode::invalid_argument, "ngram page out of range", std::to_string(page));
    }
    for (std::size_t i = 0; i < cache.lru.size(); ++i) {
        if (cache.lru[i] == page) {
            auto id = cache.lru[i];
            auto data = std::move(cache.pages[i]);
            cache.lru.erase(cache.lru.begin() + static_cast<std::ptrdiff_t>(i));
            cache.pages.erase(cache.pages.begin() + static_cast<std::ptrdiff_t>(i));
            cache.lru.push_back(id);
            cache.pages.push_back(std::move(data));
            if (!prefetch) {
                ++cache.telemetry.hits;
            }
            return {};
        }
    }
    if (prefetch) {
        ++cache.telemetry.prefetches;
    } else {
        ++cache.telemetry.misses;
    }
    const std::uint32_t row0 = page * cache.rows_per_page;
    const std::uint32_t count = (std::min)(cache.rows_per_page, cache.rows - row0);
    const std::size_t bytes = static_cast<std::size_t>(count) * cache.dim * sizeof(float);
    std::ifstream in(cache.path, std::ios::binary);
    if (!in) {
        return make_error(ErrorCode::io_error, "cannot open ngram table", cache.path.string());
    }
    in.seekg(static_cast<std::streamoff>(static_cast<std::uint64_t>(row0) * cache.dim * sizeof(float)));
    std::vector<float> payload(static_cast<std::size_t>(count) * cache.dim, 0.0F);
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(in.gcount()) != bytes) {
        return make_error(ErrorCode::io_error, "short ngram page read", cache.path.string());
    }
    if (cache.lru.size() >= cache.max_pages) {
        cache.lru.erase(cache.lru.begin());
        cache.pages.erase(cache.pages.begin());
        ++cache.telemetry.evictions;
    }
    cache.lru.push_back(page);
    cache.pages.push_back(std::move(payload));
    cache.telemetry.logical_bytes += bytes;
    return {};
}

}  // namespace

Result<Qwen38NgramPageCache> qwen38_ngram_page_cache_open(
    const Qwen38NgramPageOptions& options,
    safety::ResourceGovernor& governor) {
    if (options.rows == 0 || options.dim == 0 || options.rows_per_page == 0 || options.max_pages == 0) {
        return make_error(ErrorCode::invalid_argument, "ngram page cache options invalid", "ngram_page");
    }
    if (!std::filesystem::exists(options.table_path)) {
        return make_error(ErrorCode::not_found, "ngram table not found", options.table_path.string());
    }
    const ByteCount page_bytes = static_cast<ByteCount>(options.rows_per_page) * options.dim * sizeof(float);
    const ByteCount reserve = page_bytes * options.max_pages;
    auto ram = governor.try_reserve(safety::ResourceKind::ram, reserve, "qwen38_ngram_pages");
    if (!ram) {
        return ram.error();
    }
    Qwen38NgramPageCache cache;
    cache.path = options.table_path;
    cache.rows = options.rows;
    cache.dim = options.dim;
    cache.rows_per_page = options.rows_per_page;
    cache.max_pages = options.max_pages;
    cache.ram = std::move(ram.value());
    return cache;
}

Result<std::vector<float>> qwen38_ngram_page_gather(
    Qwen38NgramPageCache& cache,
    std::span<const std::int64_t> row_ids) {
    if (row_ids.empty()) {
        return make_error(ErrorCode::invalid_argument, "ngram gather ids must be non-empty", "ngram_page");
    }
    std::vector<float> out;
    out.reserve(row_ids.size() * cache.dim);
    std::vector<std::uint32_t> touched;
    for (std::int64_t id : row_ids) {
        if (id < 0 || static_cast<std::uint64_t>(id) >= cache.rows) {
            return make_error(ErrorCode::invalid_argument, "ngram row id out of range", std::to_string(id));
        }
        const auto uid = static_cast<std::uint32_t>(id);
        const std::uint32_t page = uid / cache.rows_per_page;
        auto loaded = load_ngram_page(cache, page, false);
        if (!loaded) {
            return loaded.error();
        }
        std::uint32_t slot = 0;
        for (; slot < cache.lru.size(); ++slot) {
            if (cache.lru[slot] == page) {
                break;
            }
        }
        const std::uint32_t local = uid - page * cache.rows_per_page;
        const float* row = cache.pages[slot].data() + static_cast<std::size_t>(local) * cache.dim;
        out.insert(out.end(), row, row + cache.dim);
        if (std::find(touched.begin(), touched.end(), page) == touched.end()) {
            touched.push_back(page);
        }
    }
    const std::uint32_t num_pages = (cache.rows + cache.rows_per_page - 1U) / cache.rows_per_page;
    for (std::uint32_t page : touched) {
        const std::uint32_t next = page + 1U;
        if (next >= num_pages) {
            continue;
        }
        bool resident = false;
        for (std::uint32_t p : cache.lru) {
            if (p == next) {
                resident = true;
                break;
            }
        }
        if (!resident) {
            auto loaded = load_ngram_page(cache, next, true);
            if (!loaded) {
                return loaded.error();
            }
        }
    }
    return out;
}

// --- Q11c: real layer-0 loader + forward ---

namespace {

bool all_finite(std::span<const float> v) {
    for (const float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

Result<void> require_finite(std::span<const float> v, std::string_view stage) {
    if (!all_finite(v)) {
        return make_error(ErrorCode::invalid_format, "layer0 stage produced non-finite values",
                          std::string(stage));
    }
    return {};
}

// Reads one tensor from a checkpoint shard as F32 using the canonical
// safetensors reader. Records payload bytes and SHA-256 of the weight bytes.
Result<std::vector<float>> load_tensor_f32(
    const std::filesystem::path& root,
    const std::string& shard_name,
    const std::string& tensor_name,
    const std::vector<std::int64_t>& expected_shape,
    Qwen38Layer0Telemetry* telemetry,
    bool is_expert) {
    const auto shard_path = root / shard_name;
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(shard_path, ec);
    if (ec) {
        return make_error(ErrorCode::io_error, "cannot stat shard", shard_path.string());
    }
    // Header reads are bounded; expert-dense shards carry larger JSON headers
    // (hundreds of tensor entries), so allow 4 MiB. The reader still validates
    // the declared length against the real header length.
    auto header = read_qwen38_safetensors_header(shard_path, file_size, 4U * 1024U * 1024U);
    if (!header) {
        return make_error(header.error().code, header.error().message,
                          "layer0 header: " + shard_name);
    }
    auto payload = qwen38_read_tensor_payload(shard_path, header.value(), tensor_name);
    if (!payload) {
        return make_error(payload.error().code, payload.error().message,
                          "layer0 tensor: " + tensor_name);
    }

    const Qwen38TensorDescriptor* desc = nullptr;
    for (const auto& t : header.value().tensors) {
        if (t.name == tensor_name) { desc = &t; break; }
    }
    if (desc == nullptr) {
        return make_error(ErrorCode::not_found, "tensor descriptor vanished", tensor_name);
    }
    if (desc->shape != expected_shape) {
        return make_error(ErrorCode::invalid_format,
            "layer0 tensor shape drift",
            tensor_name + ": expected " + std::to_string(expected_shape.size()) + " dims");
    }

    util::Sha256 digest;
    digest.update(std::span<const std::byte>{payload.value().bytes.data(), payload.value().bytes.size()});
    const std::string hash = digest.finalize_hex();

    // FP8 block-scale companions carry their own shape; locate it so the
    // decoder can validate the 128x128 block tiling.
    std::vector<std::int64_t> scale_shape;
    if (!desc->scale_inv_name.empty()) {
        const Qwen38TensorDescriptor* scale_desc = nullptr;
        for (const auto& t : header.value().tensors) {
            if (t.name == desc->scale_inv_name) { scale_desc = &t; break; }
        }
        if (scale_desc == nullptr) {
            return make_error(ErrorCode::not_found, "scale_inv descriptor missing",
                              desc->scale_inv_name);
        }
        scale_shape = scale_desc->shape;
    }

    auto decoded = qwen38_decode_tensor_payload(
        std::span<const std::byte>{payload.value().bytes.data(), payload.value().bytes.size()},
        std::span<const std::byte>{payload.value().scale_bytes.data(), payload.value().scale_bytes.size()},
        desc->dtype, desc->shape, scale_shape);
    if (!decoded) {
        return make_error(decoded.error().code, decoded.error().message,
                          "layer0 decode: " + tensor_name);
    }

    if (telemetry) {
        telemetry->payload_bytes_read += payload.value().payload_bytes;
        telemetry->tensors_read += 1U;
        if (is_expert) {
            telemetry->expert_payload_bytes += payload.value().payload_bytes;
        }
    }
    std::ignore = hash;
    return decoded;
}

}  // namespace

Result<Qwen38LayerTrunk> qwen38_load_layer_trunk(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t layer,
    Qwen38Layer0Telemetry* telemetry) {
    auto kind = qwen38_detect_checkpoint_kind(checkpoint_root);
    if (!kind) return kind.error();
    if (kind.value() == Qwen38CheckpointKind::gguf) {
        return qwen38_load_gguf_layer_trunk(checkpoint_root, layer, telemetry);
    }
    const auto t0 = std::chrono::steady_clock::now();

    // Verify the checkpoint identity first: config + full catalog validation.
    auto cfg = load_qwen38_flash_next_config(checkpoint_root / "config.json");
    if (!cfg) {
        return make_error(cfg.error().code, cfg.error().message, "layer config");
    }
    if (layer >= cfg.value().num_hidden_layers) {
        return make_error(ErrorCode::invalid_argument, "layer id out of range",
                          std::to_string(layer));
    }
    auto inspected = inspect_qwen38_flash_next_safetensors(
        checkpoint_root / "model.safetensors.index.json", cfg.value(), false);
    if (!inspected) {
        return make_error(inspected.error().code, inspected.error().message, "layer index");
    }
    auto validated = qwen38_validate_catalog(inspected.value());
    if (!validated) {
        return make_error(validated.error().code, validated.error().message, "layer catalog");
    }

    const std::string p = "model.language_model.layers." + std::to_string(layer) + ".";
    const bool is_qsa = qwen38_is_qsa_layer(layer, cfg.value().full_attention_interval);

    Qwen38LayerTrunk w;
    w.layer = layer;
    w.is_qsa = is_qsa;

    // Resolve every non-routed tensor of this layer to its shard file in one
    // pass, so the loads below never guess shard names.
    std::unordered_map<std::string, std::string> shard_of;
    {
        const std::string expert_prefix = p + "mlp.experts.";
        const std::string gate_suffix = ".gate_proj.weight";
        const std::string up_suffix = ".up_proj.weight";
        const std::string down_suffix = ".down_proj.weight";
        for (const auto& e : inspected.value().weight_map) {
            const std::string& n = e.tensor_name;
            if (n.rfind(p, 0U) != 0U) {
                continue;
            }
            if (n.rfind(expert_prefix, 0U) == 0U) {
                const std::size_t id_begin = expert_prefix.size();
                const std::size_t dot = n.find('.', id_begin);
                if (dot == std::string::npos) {
                    continue;
                }
                const std::uint32_t eid = static_cast<std::uint32_t>(
                    std::strtoul(n.c_str() + id_begin, nullptr, 10));
                if (n.size() >= gate_suffix.size()
                    && n.compare(n.size() - gate_suffix.size(), gate_suffix.size(), gate_suffix) == 0) {
                    w.expert_shards[eid][0] = e.shard_path.string();
                } else if (n.size() >= up_suffix.size()
                    && n.compare(n.size() - up_suffix.size(), up_suffix.size(), up_suffix) == 0) {
                    w.expert_shards[eid][1] = e.shard_path.string();
                } else if (n.size() >= down_suffix.size()
                    && n.compare(n.size() - down_suffix.size(), down_suffix.size(), down_suffix) == 0) {
                    w.expert_shards[eid][2] = e.shard_path.string();
                }
                continue;
            }
            shard_of[n] = e.shard_path.string();
        }
        if (w.expert_shards.size() != Qwen38ExpectedInvariants::num_experts) {
            return make_error(ErrorCode::invalid_format,
                "layer expert shard map incomplete",
                "layer=" + std::to_string(layer)
                    + " experts=" + std::to_string(w.expert_shards.size()));
        }
    }
    auto shard_for = [&](const std::string& full_name) -> Result<std::string> {
        const auto it = shard_of.find(full_name);
        if (it == shard_of.end()) {
            return make_error(ErrorCode::not_found, "layer tensor missing from index",
                              full_name);
        }
        return it->second;
    };

    auto load_residual = [&](const std::string& prefix,
                             Qwen38OwnedResidual& out) -> Result<void> {
        const std::string base = p + prefix;
        auto shard = shard_for(base + ".hc_norm.weight");
        if (!shard) return shard.error();
        auto hc_norm = load_tensor_f32(checkpoint_root, shard.value(), base + ".hc_norm.weight",
            {Qwen38ExpectedInvariants::hc_count * Qwen38ExpectedInvariants::hidden_size},
            telemetry, false);
        if (!hc_norm) return hc_norm.error();
        auto mix_down = load_tensor_f32(checkpoint_root, shard.value(), base + ".input_mix_weight_down.weight",
            {Qwen38ExpectedInvariants::hc_lowrank,
             Qwen38ExpectedInvariants::hc_count * Qwen38ExpectedInvariants::hidden_size},
            telemetry, false);
        if (!mix_down) return mix_down.error();
        auto mix_up = load_tensor_f32(checkpoint_root, shard.value(), base + ".input_mix_weight_up.weight",
            {Qwen38ExpectedInvariants::hc_count * Qwen38ExpectedInvariants::hidden_size,
             Qwen38ExpectedInvariants::hc_lowrank},
            telemetry, false);
        if (!mix_up) return mix_up.error();
        auto inject = load_tensor_f32(checkpoint_root, shard.value(), base + ".block_inject_weight.weight",
            {Qwen38ExpectedInvariants::hc_count,
             Qwen38ExpectedInvariants::hc_count * Qwen38ExpectedInvariants::hidden_size},
            telemetry, false);
        if (!inject) return inject.error();
        out.hc_norm = std::move(hc_norm.value());
        out.mix_down = std::move(mix_down.value());
        out.mix_up = std::move(mix_up.value());
        out.block_inject = std::move(inject.value());
        return {};
    };

    if (auto r = load_residual("attn_hyper_connection", w.attn_hyper); !r) {
        return r.error();
    }
    if (auto r = load_residual("mlp_hyper_connection", w.mlp_hyper); !r) {
        return r.error();
    }

    if (!is_qsa) {
        // Gated DeltaNet branch: 9 tensors.
        const std::string b = p + "linear_attn.";
        const std::size_t conv_dim =
            static_cast<std::size_t>(Qwen38ExpectedInvariants::linear_num_key_heads)
                * Qwen38ExpectedInvariants::linear_key_head_dim * 2U
            + static_cast<std::size_t>(Qwen38ExpectedInvariants::linear_num_value_heads)
                * Qwen38ExpectedInvariants::linear_value_head_dim;
        {
            auto shard = shard_for(b + "in_proj_qkv.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "in_proj_qkv.weight",
                {static_cast<std::int64_t>(conv_dim), Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.delta_net.in_proj_qkv = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "in_proj_z.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "in_proj_z.weight",
                {static_cast<std::int64_t>(
                     Qwen38ExpectedInvariants::linear_num_value_heads
                     * Qwen38ExpectedInvariants::linear_value_head_dim),
                 Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.delta_net.in_proj_z = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "in_proj_a.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "in_proj_a.weight",
                {Qwen38ExpectedInvariants::linear_num_value_heads, Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.delta_net.in_proj_a = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "in_proj_b.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "in_proj_b.weight",
                {Qwen38ExpectedInvariants::linear_num_value_heads, Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.delta_net.in_proj_b = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "conv1d.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "conv1d.weight",
                {static_cast<std::int64_t>(conv_dim), 1,
                 Qwen38ExpectedInvariants::linear_conv_kernel_dim},
                telemetry, false);
            if (!v) return v.error();
            w.delta_net.conv1d = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "A_log");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "A_log",
                {Qwen38ExpectedInvariants::linear_num_value_heads}, telemetry, false);
            if (!v) return v.error();
            w.delta_net.a_log = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "dt_bias");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "dt_bias",
                {Qwen38ExpectedInvariants::linear_num_value_heads}, telemetry, false);
            if (!v) return v.error();
            w.delta_net.dt_bias = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "norm.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "norm.weight",
                {Qwen38ExpectedInvariants::linear_value_head_dim}, telemetry, false);
            if (!v) return v.error();
            w.delta_net.norm = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "out_proj.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "out_proj.weight",
                {Qwen38ExpectedInvariants::hidden_size,
                 static_cast<std::int64_t>(
                     Qwen38ExpectedInvariants::linear_num_value_heads
                     * Qwen38ExpectedInvariants::linear_value_head_dim)},
                telemetry, false);
            if (!v) return v.error();
            w.delta_net.out_proj = std::move(v.value());
        }
    } else {
        // QSA full-attention branch: 9 tensors.
        const std::string b = p + "self_attn.";
        const std::size_t idx_qk_out =
            (static_cast<std::size_t>(Qwen38ExpectedInvariants::indexer_n_heads) + 1U)
            * Qwen38ExpectedInvariants::indexer_head_dim;
        const std::size_t q_out =
            static_cast<std::size_t>(Qwen38ExpectedInvariants::num_attention_heads)
            * Qwen38ExpectedInvariants::head_dim * 2U;
        const std::size_t kv_out =
            static_cast<std::size_t>(Qwen38ExpectedInvariants::num_key_value_heads)
            * Qwen38ExpectedInvariants::head_dim;
        {
            auto shard = shard_for(b + "indexer.index_qk_proj.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(),
                b + "indexer.index_qk_proj.weight",
                {static_cast<std::int64_t>(idx_qk_out), Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.qsa.index_qk = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "indexer.q_layernorm.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(),
                b + "indexer.q_layernorm.weight",
                {Qwen38ExpectedInvariants::indexer_head_dim}, telemetry, false);
            if (!v) return v.error();
            w.qsa.q_layernorm = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "indexer.k_layernorm.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(),
                b + "indexer.k_layernorm.weight",
                {Qwen38ExpectedInvariants::indexer_head_dim}, telemetry, false);
            if (!v) return v.error();
            w.qsa.k_layernorm = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "q_proj.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "q_proj.weight",
                {static_cast<std::int64_t>(q_out), Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.qsa.q_proj = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "k_proj.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "k_proj.weight",
                {static_cast<std::int64_t>(kv_out), Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.qsa.k_proj = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "v_proj.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "v_proj.weight",
                {static_cast<std::int64_t>(kv_out), Qwen38ExpectedInvariants::hidden_size},
                telemetry, false);
            if (!v) return v.error();
            w.qsa.v_proj = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "o_proj.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "o_proj.weight",
                {Qwen38ExpectedInvariants::hidden_size,
                 static_cast<std::int64_t>(
                     Qwen38ExpectedInvariants::num_attention_heads
                     * Qwen38ExpectedInvariants::head_dim)},
                telemetry, false);
            if (!v) return v.error();
            w.qsa.o_proj = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "q_norm.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "q_norm.weight",
                {Qwen38ExpectedInvariants::head_dim}, telemetry, false);
            if (!v) return v.error();
            w.qsa.q_norm = std::move(v.value());
        }
        {
            auto shard = shard_for(b + "k_norm.weight");
            if (!shard) return shard.error();
            auto v = load_tensor_f32(checkpoint_root, shard.value(), b + "k_norm.weight",
                {Qwen38ExpectedInvariants::head_dim}, telemetry, false);
            if (!v) return v.error();
            w.qsa.k_norm = std::move(v.value());
        }
    }
    {
        auto shard = shard_for(p + "mlp.gate.weight");
        if (!shard) return shard.error();
        auto v = load_tensor_f32(checkpoint_root, shard.value(), p + "mlp.gate.weight",
            {Qwen38ExpectedInvariants::num_experts, Qwen38ExpectedInvariants::hidden_size},
            telemetry, false);
        if (!v) return v.error();
        w.router = std::move(v.value());
    }
    {
        auto shard = shard_for(p + "mlp.shared_expert.gate_proj.weight");
        if (!shard) return shard.error();
        auto v = load_tensor_f32(checkpoint_root, shard.value(), p + "mlp.shared_expert.gate_proj.weight",
            {Qwen38ExpectedInvariants::shared_expert_intermediate_size,
             Qwen38ExpectedInvariants::hidden_size},
            telemetry, false);
        if (!v) return v.error();
        w.shared_mlp.gate_proj = std::move(v.value());
    }
    {
        auto shard = shard_for(p + "mlp.shared_expert.up_proj.weight");
        if (!shard) return shard.error();
        auto v = load_tensor_f32(checkpoint_root, shard.value(), p + "mlp.shared_expert.up_proj.weight",
            {Qwen38ExpectedInvariants::shared_expert_intermediate_size,
             Qwen38ExpectedInvariants::hidden_size},
            telemetry, false);
        if (!v) return v.error();
        w.shared_mlp.up_proj = std::move(v.value());
    }
    {
        auto shard = shard_for(p + "mlp.shared_expert.down_proj.weight");
        if (!shard) return shard.error();
        auto v = load_tensor_f32(checkpoint_root, shard.value(), p + "mlp.shared_expert.down_proj.weight",
            {Qwen38ExpectedInvariants::hidden_size,
             Qwen38ExpectedInvariants::shared_expert_intermediate_size},
            telemetry, false);
        if (!v) return v.error();
        w.shared_mlp.down_proj = std::move(v.value());
    }
    {
        auto shard = shard_for(p + "mlp.shared_expert_gate.weight");
        if (!shard) return shard.error();
        auto v = load_tensor_f32(checkpoint_root, shard.value(), p + "mlp.shared_expert_gate.weight",
            {1, Qwen38ExpectedInvariants::hidden_size}, telemetry, false);
        if (!v) return v.error();
        w.shared_mlp.shared_gate = std::move(v.value());
    }

    if (telemetry) {
        telemetry->load_ms = std::chrono::steady_clock::now() - t0;
    }
    return w;
}

Result<std::vector<float>> qwen38_load_embed_row(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t token_id,
    Qwen38Layer0Telemetry* telemetry) {
    auto kind = qwen38_detect_checkpoint_kind(checkpoint_root);
    if (!kind) return kind.error();
    if (kind.value() == Qwen38CheckpointKind::gguf) {
        return qwen38_load_gguf_embed_row(checkpoint_root, token_id, telemetry);
    }
    const std::string s130 = "model-00130-of-00131.safetensors";
    const std::string embed_name = "model.language_model.embed_tokens.weight";
    const auto shard_path = checkpoint_root / s130;
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(shard_path, ec);
    if (ec) {
        return make_error(ErrorCode::io_error, "cannot stat embed shard", s130);
    }
    auto header = read_qwen38_safetensors_header(shard_path, file_size, 4U * 1024U * 1024U);
    if (!header) {
        return make_error(header.error().code, header.error().message, "layer0 embed header");
    }
    const Qwen38TensorDescriptor* desc = nullptr;
    for (const auto& t : header.value().tensors) {
        if (t.name == embed_name) { desc = &t; break; }
    }
    if (desc == nullptr) {
        return make_error(ErrorCode::not_found, "embed tensor not found", embed_name);
    }
    if (desc->dtype != "BF16" || desc->shape.size() != 2U
        || desc->shape[1] != Qwen38ExpectedInvariants::hidden_size) {
        return make_error(ErrorCode::invalid_format, "embed tensor drift", embed_name);
    }
    if (token_id >= static_cast<std::uint32_t>(desc->shape[0])) {
        return make_error(ErrorCode::invalid_argument, "token id out of vocab",
                          std::to_string(token_id));
    }
    // Bounded read: fetch exactly one embedding row instead of the whole
    // 1.27 GiB embedding tensor.
    const std::size_t row_bytes = static_cast<std::size_t>(desc->shape[1]) * 2U;
    const std::uint64_t row_begin_abs =
        8U + header.value().json_header_bytes + desc->data_offset_begin
        + static_cast<std::uint64_t>(token_id) * row_bytes;
    if (row_begin_abs + row_bytes > header.value().total_bytes) {
        return make_error(ErrorCode::invalid_format, "embed row out of payload", embed_name);
    }
    std::ifstream in(shard_path, std::ios::binary);
    if (!in) {
        return make_error(ErrorCode::io_error, "cannot open embed shard", s130);
    }
    in.seekg(static_cast<std::streamoff>(row_begin_abs));
    std::vector<std::byte> row_bytes_buf(row_bytes);
    in.read(reinterpret_cast<char*>(row_bytes_buf.data()),
            static_cast<std::streamsize>(row_bytes));
    if (static_cast<std::uint64_t>(in.gcount()) != row_bytes) {
        return make_error(ErrorCode::io_error, "short read of embed row", embed_name);
    }
    std::vector<std::int64_t> row_shape{1, desc->shape[1]};
    auto row = qwen38_decode_bf16(row_bytes_buf, row_shape);
    if (!row) {
        return make_error(row.error().code, row.error().message, "layer0 embed row decode");
    }
    if (telemetry) {
        telemetry->payload_bytes_read += row_bytes;
        telemetry->tensors_read += 1U;
    }
    return row;
}

Result<Qwen38Layer0ForwardResult> qwen38_layer_forward(
    const Qwen38LayerTrunk& trunk,
    std::span<const float> input_stream,
    Qwen38LayerRuntimeState& runtime_state,
    std::uint32_t num_experts,
    std::uint32_t top_k) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint32_t H = Qwen38ExpectedInvariants::hidden_size;
    const std::uint32_t C = Qwen38ExpectedInvariants::hc_count;
    const std::uint32_t R = Qwen38ExpectedInvariants::hc_lowrank;
    const Qwen38LayerTrunk& weights = trunk;

    // Input stream (already hc_count * hidden_size).
    if (input_stream.size() != static_cast<std::size_t>(C) * H) {
        return make_error(ErrorCode::invalid_format, "input stream width drift",
                          std::to_string(input_stream.size()));
    }
    if (auto r = require_finite(input_stream, "input_stream"); !r) return r.error();
    const std::span<const float> stream_in = input_stream;

    Qwen38GatedResidualView attn_view{
        weights.attn_hyper.hc_norm, weights.attn_hyper.mix_down,
        weights.attn_hyper.mix_up, weights.attn_hyper.block_inject};
    // attn gated residual on the input stream
    auto attn_res = qwen38_gated_residual(stream_in, attn_view, C, H, R, 1.0e-6F);
    if (!attn_res) return attn_res.error();
    if (auto r = require_finite(attn_res.value().mixed_input, "attn_gr_mixed"); !r) return r.error();

    std::vector<float> attn_out_vec;
    if (!weights.is_qsa) {
        // Gated DeltaNet: the caller owns the state (zero it for the first
        // token, reuse it for subsequent tokens so conv history and SSM carry).
        Qwen38DeltaNetView dn_view{
            weights.delta_net.in_proj_qkv, weights.delta_net.in_proj_z,
            weights.delta_net.in_proj_a, weights.delta_net.in_proj_b,
            weights.delta_net.conv1d, weights.delta_net.a_log,
            weights.delta_net.dt_bias, weights.delta_net.norm,
            weights.delta_net.out_proj};
        dn_view.hidden_size = H;
        dn_view.num_k_heads = Qwen38ExpectedInvariants::linear_num_key_heads;
        dn_view.num_v_heads = Qwen38ExpectedInvariants::linear_num_value_heads;
        dn_view.key_head_dim = Qwen38ExpectedInvariants::linear_key_head_dim;
        dn_view.value_head_dim = Qwen38ExpectedInvariants::linear_value_head_dim;
        dn_view.conv_kernel = Qwen38ExpectedInvariants::linear_conv_kernel_dim;
        dn_view.rms_eps = 1.0e-6F;
        Qwen38DeltaNetState& delta_net_state = runtime_state.delta_net;
        if (delta_net_state.conv.empty() && delta_net_state.ssm.empty()) {
            delta_net_state.conv.assign(
                (static_cast<std::size_t>(Qwen38ExpectedInvariants::linear_num_key_heads)
                     * Qwen38ExpectedInvariants::linear_key_head_dim * 2U
                 + static_cast<std::size_t>(Qwen38ExpectedInvariants::linear_num_value_heads)
                     * Qwen38ExpectedInvariants::linear_value_head_dim)
                    * (Qwen38ExpectedInvariants::linear_conv_kernel_dim - 1U),
                0.0F);
            delta_net_state.ssm.assign(
                static_cast<std::size_t>(Qwen38ExpectedInvariants::linear_num_value_heads)
                    * Qwen38ExpectedInvariants::linear_key_head_dim
                    * Qwen38ExpectedInvariants::linear_value_head_dim,
                0.0F);
        }
        auto dn_out = qwen38_delta_net_decode(attn_res.value().mixed_input, dn_view, delta_net_state);
        if (!dn_out) return dn_out.error();
        if (auto r = require_finite(dn_out.value(), "deltanet_out"); !r) return r.error();
        attn_out_vec = std::move(dn_out.value());
    } else {
        // QSA full attention: the caller owns the indexer-key and KV caches.
        Qwen38QsaLayerView qsa_view{
            weights.qsa.index_qk, weights.qsa.q_layernorm, weights.qsa.k_layernorm,
            weights.qsa.q_proj, weights.qsa.k_proj, weights.qsa.v_proj,
            weights.qsa.o_proj, weights.qsa.q_norm, weights.qsa.k_norm};
        qsa_view.hidden_size = H;
        qsa_view.indexer_heads = Qwen38ExpectedInvariants::indexer_n_heads;
        qsa_view.indexer_dim = Qwen38ExpectedInvariants::indexer_head_dim;
        qsa_view.budget = Qwen38ExpectedInvariants::indexer_budget;
        qsa_view.compress_ratio = 4U;
        qsa_view.attn_heads = Qwen38ExpectedInvariants::num_attention_heads;
        qsa_view.attn_kv_heads = Qwen38ExpectedInvariants::num_key_value_heads;
        qsa_view.attn_head_dim = Qwen38ExpectedInvariants::head_dim;
        qsa_view.rms_eps = 1.0e-6F;
        auto qsa_out = qwen38_qsa_layer_token(attn_res.value().mixed_input, qsa_view,
                                              runtime_state.qsa);
        if (!qsa_out) return qsa_out.error();
        if (auto r = require_finite(qsa_out.value(), "qsa_out"); !r) return r.error();
        attn_out_vec = std::move(qsa_out.value());
    }

    auto attn_out = qwen38_gated_residual_combine(
        attn_res.value().residual, attn_out_vec, attn_res.value().injection_weights, C, H);
    if (!attn_out) return attn_out.error();

    // mlp gated residual
    Qwen38GatedResidualView mlp_view{
        weights.mlp_hyper.hc_norm, weights.mlp_hyper.mix_down,
        weights.mlp_hyper.mix_up, weights.mlp_hyper.block_inject};
    auto mlp_res = qwen38_gated_residual(attn_out.value(), mlp_view, C, H, R, 1.0e-6F);
    if (!mlp_res) return mlp_res.error();
    if (auto r = require_finite(mlp_res.value().mixed_input, "mlp_gr_mixed"); !r) return r.error();

    // router
    auto route = qwen38_moe_route(mlp_res.value().mixed_input, weights.router,
                                  num_experts, top_k, true);
    if (!route) return route.error();

    // shared expert
    auto shared = qwen38_shared_mlp(mlp_res.value().mixed_input,
                                    weights.shared_mlp.gate_proj,
                                    weights.shared_mlp.up_proj,
                                    weights.shared_mlp.down_proj,
                                    weights.shared_mlp.shared_gate,
                                    H, Qwen38ExpectedInvariants::shared_expert_intermediate_size);
    if (!shared) return shared.error();
    if (auto r = require_finite(shared.value(), "shared_expert"); !r) return r.error();

    Qwen38Layer0ForwardResult out;
    out.route = route.value();
    out.mixed_input = std::move(mlp_res.value().mixed_input);
    out.residual = std::move(mlp_res.value().residual);
    out.injection_weights = std::move(mlp_res.value().injection_weights);
    out.shared_out = std::move(shared.value());
    out.telemetry.selected_experts = route.value().indices.size();
    out.telemetry.forward_ms = std::chrono::steady_clock::now() - t0;
    return out;
}

Result<std::vector<float>> qwen38_layer0_finish(
    const Qwen38Layer0ForwardResult& partial,
    std::span<const float> expert_sum) {
    const std::uint32_t H = Qwen38ExpectedInvariants::hidden_size;
    const std::uint32_t C = Qwen38ExpectedInvariants::hc_count;
    if (expert_sum.size() != H || partial.shared_out.size() != H
        || partial.residual.size() != static_cast<std::size_t>(C) * H
        || partial.injection_weights.size() != C) {
        return make_error(ErrorCode::invalid_argument, "layer0 finish size mismatch",
                          "layer0_finish");
    }
    std::vector<float> moe_total(H, 0.0F);
    for (std::uint32_t d = 0; d < H; ++d) {
        moe_total[d] = partial.shared_out[d] + expert_sum[d];
    }
    return qwen38_gated_residual_combine(partial.residual, moe_total,
                                         partial.injection_weights, C, H);
}

Result<Qwen38OwnedExpert> qwen38_load_layer_expert(
    const Qwen38LayerTrunk& trunk,
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expert_id,
    Qwen38Layer0Telemetry* telemetry) {
    if (trunk.is_gguf) {
        auto checkpoint = qwen38_open_gguf_checkpoint(checkpoint_root);
        if (!checkpoint) return checkpoint.error();
        return qwen38_load_gguf_expert(trunk, checkpoint.value(), expert_id, telemetry);
    }
    if (expert_id >= Qwen38ExpectedInvariants::num_experts) {
        return make_error(ErrorCode::invalid_argument, "expert id out of range",
                          std::to_string(expert_id));
    }
    const auto it = trunk.expert_shards.find(expert_id);
    if (it == trunk.expert_shards.end()) {
        return make_error(ErrorCode::not_found, "expert not in shard map",
                          std::to_string(expert_id));
    }
    const std::string p = "model.language_model.layers." + std::to_string(trunk.layer)
        + ".mlp.experts." + std::to_string(expert_id) + ".";
    const std::string& gate_shard = it->second[0];
    const std::string& up_shard = it->second[1];
    const std::string& down_shard = it->second[2];
    if (gate_shard.empty() || up_shard.empty() || down_shard.empty()) {
        return make_error(ErrorCode::not_found, "expert shards missing from map", p);
    }

    Qwen38OwnedExpert expert;
    const std::vector<std::int64_t> gate_shape{
        static_cast<std::int64_t>(Qwen38ExpectedInvariants::moe_intermediate_size),
        static_cast<std::int64_t>(Qwen38ExpectedInvariants::hidden_size)};
    const std::vector<std::int64_t> down_shape{
        static_cast<std::int64_t>(Qwen38ExpectedInvariants::hidden_size),
        static_cast<std::int64_t>(Qwen38ExpectedInvariants::moe_intermediate_size)};
    auto gate = load_tensor_f32(checkpoint_root, gate_shard, p + "gate_proj.weight",
                                gate_shape, telemetry, true);
    if (!gate) return gate.error();
    if (auto r = require_finite(gate.value(), p + "gate_proj"); !r) return r.error();
    auto up = load_tensor_f32(checkpoint_root, up_shard, p + "up_proj.weight",
                              gate_shape, telemetry, true);
    if (!up) return up.error();
    if (auto r = require_finite(up.value(), p + "up_proj"); !r) return r.error();
    auto down = load_tensor_f32(checkpoint_root, down_shard, p + "down_proj.weight",
                                down_shape, telemetry, true);
    if (!down) return down.error();
    if (auto r = require_finite(down.value(), p + "down_proj"); !r) return r.error();
    expert.gate_proj = std::move(gate.value());
    expert.up_proj = std::move(up.value());
    expert.down_proj = std::move(down.value());
    return expert;
}

}  // namespace vastloom::inference
