// Qwen3.8-Flash-Next GGUF (qwen4exp) checkpoint source.
//
// Reads llama.cpp-format Q8_0 (and friends) GGUF shards through the generic
// model::GgufSplitCatalog and converts the ggml [in, out] layout into the
// row-major [out, in] layout the reference operators use. Routed experts live
// in fused [in, out, 512] tensors; one expert slice is a fixed byte stride.

#include "vastloom/inference/qwen3_8_flash_next.hpp"
#include "vastloom/inference/qwen3_8_flash_next_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/model/gguf.hpp"

namespace vastloom::inference {

namespace {

bool all_finite_local(std::span<const float> v) {
    for (const float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

Result<void> require_finite_local(std::span<const float> v, std::string_view stage) {
    if (!all_finite_local(v)) {
        return make_error(ErrorCode::invalid_format,
                          "gguf stage produced non-finite values", std::string(stage));
    }
    return {};
}

// GGML quant layouts used by this checkpoint (block bytes per element count).
struct GgmlLayout {
    std::uint64_t block_elements;
    std::uint64_t block_bytes;
};

Result<GgmlLayout> ggml_layout(std::uint32_t ggml_type) {
    switch (ggml_type) {
        case 0U: return GgmlLayout{1U, 4U};    // F32
        case 1U: return GgmlLayout{1U, 2U};    // F16
        case 8U: return GgmlLayout{32U, 34U};  // Q8_0
        case 30U: return GgmlLayout{1U, 2U};   // BF16
        default:
            return make_error(ErrorCode::unsupported, "unsupported GGML type for qwen4exp",
                              std::to_string(ggml_type));
    }
}

std::vector<float> transpose_ggml(std::span<const float> flat,
                                  std::uint64_t ne0,
                                  std::uint64_t ne1) {
    // ggml flat index = i0 + i1 * ne0 (ne0 contiguous); row-major [ne1, ne0]
    // out[o * ne0 + i] = flat[i + o * ne0] is a plain copy of that layout, so
    // the ggml [in, out] tensor becomes row-major [out, in] directly.
    std::vector<float> out(flat.size());
    std::memcpy(out.data(), flat.data(), flat.size() * sizeof(float));
    std::ignore = ne0;
    std::ignore = ne1;
    return out;
}

// Cached open checkpoints: the chain opens one trunk/expert source per call
// and re-parsing ~11 MiB headers per shard each time would dominate I/O.
std::mutex& gguf_cache_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, Qwen38GgufCheckpoint>& gguf_cache() {
    static std::map<std::string, Qwen38GgufCheckpoint> cache;
    return cache;
}

}  // namespace

float qwen38_f16_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t sign = (bits >> 15) & 0x1U;
    const std::uint32_t exp = (bits >> 10) & 0x1FU;
    const std::uint32_t mant = bits & 0x3FFU;
    float value = 0.0F;
    if (exp == 0U) {
        // Subnormal half: mant/1024 * 2^-14 == mant * 2^-24.
        value = std::ldexp(static_cast<float>(mant), -24);
    } else if (exp == 31U) {
        value = mant == 0U
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(static_cast<float>(mant) + 1024.0F,
                           static_cast<int>(exp) - 25);
    }
    return sign ? -value : value;
}

Result<std::vector<float>> qwen38_decode_q8_0(
    std::span<const std::byte> payload,
    std::uint64_t element_count) {
    const std::uint64_t blocks = (element_count + 31U) / 32U;
    if (payload.size() < blocks * 34U) {
        return make_error(ErrorCode::invalid_argument, "Q8_0 payload too small",
                          "elements=" + std::to_string(element_count));
    }
    std::vector<float> out(static_cast<std::size_t>(element_count));
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(payload.data());
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = bytes + b * 34U;
        std::uint16_t bits = 0U;
        std::memcpy(&bits, block, 2U);
        const float scale = qwen38_f16_to_f32(bits);
        const std::size_t base = static_cast<std::size_t>(b * 32U);
        const std::size_t count =
            static_cast<std::size_t>((std::min<std::uint64_t>)(32U, element_count - b * 32U));
        for (std::size_t i = 0; i < count; ++i) {
            const auto q = static_cast<std::int8_t>(block[2U + i]);
            out[base + i] = static_cast<float>(q) * scale;
        }
    }
    return out;
}

Result<Qwen38CheckpointKind> qwen38_detect_checkpoint_kind(
    const std::filesystem::path& checkpoint_root) {
    std::error_code ec;
    if (std::filesystem::exists(checkpoint_root / "model.safetensors.index.json", ec)) {
        return Qwen38CheckpointKind::safetensors;
    }
    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_root, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            return Qwen38CheckpointKind::gguf;
        }
        if (entry.is_directory()) {
            std::error_code inner_ec;
            for (const auto& inner : std::filesystem::directory_iterator(entry.path(), inner_ec)) {
                if (inner.is_regular_file() && inner.path().extension() == ".gguf") {
                    return Qwen38CheckpointKind::gguf;
                }
            }
        }
    }
    return make_error(ErrorCode::not_found,
                      "no safetensors index or GGUF shards under checkpoint root",
                      checkpoint_root.string());
}

Result<Qwen38GgufCheckpoint> qwen38_open_gguf_checkpoint(
    const std::filesystem::path& checkpoint_root) {
    {
        std::lock_guard<std::mutex> lock(gguf_cache_mutex());
        const auto it = gguf_cache().find(checkpoint_root.string());
        if (it != gguf_cache().end()) {
            return it->second;
        }
    }
    std::vector<std::filesystem::path> shard_paths;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_root, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            shard_paths.push_back(entry.path());
        } else if (entry.is_directory()) {
            std::error_code inner_ec;
            for (const auto& inner : std::filesystem::directory_iterator(entry.path(), inner_ec)) {
                if (inner.is_regular_file() && inner.path().extension() == ".gguf") {
                    shard_paths.push_back(inner.path());
                }
            }
        }
    }
    if (shard_paths.empty()) {
        return make_error(ErrorCode::not_found, "no GGUF shards found", checkpoint_root.string());
    }
    std::sort(shard_paths.begin(), shard_paths.end());
    std::vector<model::GgufFile> shards;
    shards.reserve(shard_paths.size());
    for (const auto& p : shard_paths) {
        auto file = model::GgufFile::open(p);
        if (!file) {
            return make_error(file.error().code, file.error().message,
                              "gguf shard open: " + p.string());
        }
        shards.push_back(std::move(file.value()));
    }
    auto catalog = model::finalize_gguf_split_catalog(std::move(shards));
    if (!catalog) {
        return make_error(catalog.error().code, catalog.error().message, "gguf split catalog");
    }
    if (catalog.value().architecture != "qwen4exp") {
        return make_error(ErrorCode::invalid_format,
                          "GGUF architecture is not qwen4exp", catalog.value().architecture);
    }
    Qwen38GgufCheckpoint checkpoint;
    checkpoint.root = checkpoint_root;
    checkpoint.catalog = std::move(catalog.value());
    std::lock_guard<std::mutex> lock(gguf_cache_mutex());
    auto [it, inserted] = gguf_cache().emplace(checkpoint_root.string(), checkpoint);
    return it->second;
}

Result<std::vector<float>> qwen38_read_gguf_tensor_f32(
    const Qwen38GgufCheckpoint& checkpoint,
    std::string_view tensor_name,
    const std::vector<std::uint64_t>& expected_dims,
    bool transpose,
    Qwen38Layer0Telemetry* telemetry) {
    const auto loc = checkpoint.catalog.tensors.find(std::string{tensor_name});
    if (loc == checkpoint.catalog.tensors.end()) {
        return make_error(ErrorCode::not_found, "GGUF tensor not in catalog",
                          std::string{tensor_name});
    }
    const model::GgufFile& shard = checkpoint.catalog.shards[loc->second.shard_index];
    const model::GgufTensorDescriptor& desc =
        shard.tensors()[loc->second.tensor_index];
    if (desc.shape.size() != expected_dims.size()) {
        return make_error(ErrorCode::invalid_format, "GGUF tensor rank drift",
                          std::string{tensor_name});
    }
    for (std::size_t i = 0; i < expected_dims.size(); ++i) {
        if (desc.shape[i] != expected_dims[i]) {
            return make_error(ErrorCode::invalid_format, "GGUF tensor shape drift",
                              std::string{tensor_name});
        }
    }
    if (!desc.encoded_bytes.has_value()) {
        return make_error(ErrorCode::invalid_format, "GGUF tensor byte size unknown",
                          std::string{tensor_name});
    }
    const std::uint64_t begin = desc.absolute_offset;
    const std::uint64_t span = desc.encoded_bytes.value();
    if (begin + span > shard.file_bytes()) {
        return make_error(ErrorCode::invalid_format, "GGUF tensor out of file bounds",
                          std::string{tensor_name});
    }
    std::ifstream in(shard.path(), std::ios::binary);
    if (!in) {
        return make_error(ErrorCode::io_error, "cannot open gguf shard", shard.path().string());
    }
    in.seekg(static_cast<std::streamoff>(begin));
    std::vector<std::byte> payload(static_cast<std::size_t>(span));
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(span));
    if (static_cast<std::uint64_t>(in.gcount()) != span) {
        return make_error(ErrorCode::io_error, "short read of gguf tensor",
                          std::string{tensor_name});
    }
    if (telemetry) {
        telemetry->payload_bytes_read += span;
        telemetry->tensors_read += 1U;
    }

    const std::uint64_t elements = desc.element_count;
    std::vector<float> flat;
    switch (desc.ggml_type) {
        case 0U: {  // F32
            flat.resize(static_cast<std::size_t>(elements));
            std::memcpy(flat.data(), payload.data(), payload.size());
            break;
        }
        case 1U: {  // F16
            flat.resize(static_cast<std::size_t>(elements));
            for (std::uint64_t i = 0; i < elements; ++i) {
                std::uint16_t bits = 0U;
                std::memcpy(&bits, payload.data() + i * 2U, 2U);
                flat[static_cast<std::size_t>(i)] = qwen38_f16_to_f32(bits);
            }
            break;
        }
        case 8U: {
            auto decoded = qwen38_decode_q8_0(payload, elements);
            if (!decoded) return decoded.error();
            flat = std::move(decoded.value());
            break;
        }
        case 30U: {  // BF16
            auto decoded = qwen38_decode_bf16(payload, {static_cast<std::int64_t>(elements)});
            if (!decoded) return decoded.error();
            flat = std::move(decoded.value());
            break;
        }
        default:
            return make_error(ErrorCode::unsupported, "unsupported ggml type",
                              std::to_string(desc.ggml_type));
    }
    if (auto r = require_finite_local(flat, tensor_name); !r) return r.error();
    if (transpose && desc.shape.size() == 2U) {
        // ggml [in, out] flat (ne0 contiguous over in) is exactly row-major
        // [out, in]; reinterpret without per-element work.
        return transpose_ggml(flat, desc.shape[0], desc.shape[1]);
    }
    return flat;
}

namespace {

constexpr std::uint32_t kQwen38FullAttentionInterval = 4U;

struct GgufTensorNames {
    std::string hc_attn_norm;
    std::string hc_attn_down;
    std::string hc_attn_up;
    std::string hc_attn_inject;
    std::string hc_ffn_norm;
    std::string hc_ffn_down;
    std::string hc_ffn_up;
    std::string hc_ffn_inject;
    std::string router;
    std::string shared_gate;
    std::string shared_gate_proj;
    std::string shared_up_proj;
    std::string shared_down_proj;
    std::string gate_exps;
    std::string up_exps;
    std::string down_exps;
};

GgufTensorNames gguf_names(std::uint32_t layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    GgufTensorNames n;
    n.hc_attn_norm = p + "hc_attn_norm.weight";
    n.hc_attn_down = p + "hc_attn_down.weight";
    n.hc_attn_up = p + "hc_attn_up.weight";
    n.hc_attn_inject = p + "hc_attn_inject.weight";
    n.hc_ffn_norm = p + "hc_ffn_norm.weight";
    n.hc_ffn_down = p + "hc_ffn_down.weight";
    n.hc_ffn_up = p + "hc_ffn_up.weight";
    n.hc_ffn_inject = p + "hc_ffn_inject.weight";
    n.router = p + "ffn_gate_inp.weight";
    n.shared_gate = p + "ffn_gate_inp_shexp.weight";
    n.shared_gate_proj = p + "ffn_gate_shexp.weight";
    n.shared_up_proj = p + "ffn_up_shexp.weight";
    n.shared_down_proj = p + "ffn_down_shexp.weight";
    n.gate_exps = p + "ffn_gate_exps.weight";
    n.up_exps = p + "ffn_up_exps.weight";
    n.down_exps = p + "ffn_down_exps.weight";
    return n;
}

constexpr std::uint64_t kHidden = Qwen38ExpectedInvariants::hidden_size;
constexpr std::uint64_t kStream =
    static_cast<std::uint64_t>(Qwen38ExpectedInvariants::hc_count) * kHidden;
constexpr std::uint64_t kRank = Qwen38ExpectedInvariants::hc_lowrank;
constexpr std::uint64_t kExperts = Qwen38ExpectedInvariants::num_experts;
constexpr std::uint64_t kInter = Qwen38ExpectedInvariants::moe_intermediate_size;

}  // namespace

Result<Qwen38LayerTrunk> qwen38_load_gguf_layer_trunk(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t layer,
    Qwen38Layer0Telemetry* telemetry) {
    const auto t0 = std::chrono::steady_clock::now();
    auto checkpoint = qwen38_open_gguf_checkpoint(checkpoint_root);
    if (!checkpoint) return checkpoint.error();

    const std::uint32_t layers = Qwen38ExpectedInvariants::num_hidden_layers;
    if (layer >= layers) {
        return make_error(ErrorCode::invalid_argument, "layer id out of range",
                          std::to_string(layer));
    }
    const bool is_qsa = qwen38_is_qsa_layer(layer, kQwen38FullAttentionInterval);
    const GgufTensorNames names = gguf_names(layer);
    const std::string p = "blk." + std::to_string(layer) + ".";

    Qwen38LayerTrunk w;
    w.layer = layer;
    w.is_qsa = is_qsa;
    w.is_gguf = true;

    auto load = [&](const std::string& name,
                    const std::vector<std::uint64_t>& dims,
                    bool transpose) -> Result<std::vector<float>> {
        return qwen38_read_gguf_tensor_f32(checkpoint.value(), name, dims, transpose, telemetry);
    };

    auto load_residual = [&](const std::string& norm, const std::string& down,
                             const std::string& up, const std::string& inject,
                             Qwen38OwnedResidual& out) -> Result<void> {
        auto hc_norm = load(norm, {kStream}, false);
        if (!hc_norm) return hc_norm.error();
        auto mix_down = load(down, {kStream, kRank}, true);
        if (!mix_down) return mix_down.error();
        auto mix_up = load(up, {kRank, kStream}, true);
        if (!mix_up) return mix_up.error();
        auto inject_w = load(inject, {kStream, Qwen38ExpectedInvariants::hc_count}, true);
        if (!inject_w) return inject_w.error();
        out.hc_norm = std::move(hc_norm.value());
        out.mix_down = std::move(mix_down.value());
        out.mix_up = std::move(mix_up.value());
        out.block_inject = std::move(inject_w.value());
        return {};
    };
    if (auto r = load_residual(names.hc_attn_norm, names.hc_attn_down, names.hc_attn_up,
                               names.hc_attn_inject, w.attn_hyper); !r) {
        return r.error();
    }
    if (auto r = load_residual(names.hc_ffn_norm, names.hc_ffn_down, names.hc_ffn_up,
                               names.hc_ffn_inject, w.mlp_hyper); !r) {
        return r.error();
    }

    if (!is_qsa) {
        // Gated DeltaNet branch.
        const std::uint64_t kDim =
            static_cast<std::uint64_t>(Qwen38ExpectedInvariants::linear_num_key_heads)
            * Qwen38ExpectedInvariants::linear_key_head_dim;
        const std::uint64_t vDim =
            static_cast<std::uint64_t>(Qwen38ExpectedInvariants::linear_num_value_heads)
            * Qwen38ExpectedInvariants::linear_value_head_dim;
        const std::string b = p + "ssm_";
        auto qkv = load(p + "attn_qkv.weight", {kHidden, 2U * kDim + vDim}, true);
        if (!qkv) return qkv.error();
        w.delta_net.in_proj_qkv = std::move(qkv.value());
        auto z = load(p + "attn_gate.weight", {kHidden, vDim}, true);
        if (!z) return z.error();
        w.delta_net.in_proj_z = std::move(z.value());
        auto a = load(b + "alpha.weight", {kHidden, Qwen38ExpectedInvariants::linear_num_value_heads}, true);
        if (!a) return a.error();
        w.delta_net.in_proj_a = std::move(a.value());
        auto bb = load(b + "beta.weight", {kHidden, Qwen38ExpectedInvariants::linear_num_value_heads}, true);
        if (!bb) return bb.error();
        w.delta_net.in_proj_b = std::move(bb.value());
        auto conv = load(b + "conv1d.weight",
                         {Qwen38ExpectedInvariants::linear_conv_kernel_dim, 2U * kDim + vDim}, false);
        if (!conv) return conv.error();
        w.delta_net.conv1d = std::move(conv.value());
        auto a_log = load(b + "a", {Qwen38ExpectedInvariants::linear_num_value_heads}, false);
        if (!a_log) return a_log.error();
        w.delta_net.a_log = std::move(a_log.value());
        auto dt = load(b + "dt.bias", {Qwen38ExpectedInvariants::linear_num_value_heads}, false);
        if (!dt) return dt.error();
        w.delta_net.dt_bias = std::move(dt.value());
        auto norm = load(b + "norm.weight", {Qwen38ExpectedInvariants::linear_value_head_dim}, false);
        if (!norm) return norm.error();
        w.delta_net.norm = std::move(norm.value());
        auto out = load(b + "out.weight", {vDim, kHidden}, true);
        if (!out) return out.error();
        w.delta_net.out_proj = std::move(out.value());
    } else {
        // QSA full-attention branch.
        const std::uint64_t qOut =
            static_cast<std::uint64_t>(Qwen38ExpectedInvariants::num_attention_heads)
            * Qwen38ExpectedInvariants::head_dim * 2U;
        const std::uint64_t kvOut =
            static_cast<std::uint64_t>(Qwen38ExpectedInvariants::num_key_value_heads)
            * Qwen38ExpectedInvariants::head_dim;
        const std::uint64_t idxQ =
            static_cast<std::uint64_t>(Qwen38ExpectedInvariants::indexer_n_heads)
            * Qwen38ExpectedInvariants::indexer_head_dim;
        const std::uint64_t idxK = Qwen38ExpectedInvariants::indexer_head_dim;
        auto q = load(p + "attn_q.weight", {kHidden, qOut}, true);
        if (!q) return q.error();
        w.qsa.q_proj = std::move(q.value());
        auto k = load(p + "attn_k.weight", {kHidden, kvOut}, true);
        if (!k) return k.error();
        w.qsa.k_proj = std::move(k.value());
        auto v = load(p + "attn_v.weight", {kHidden, kvOut}, true);
        if (!v) return v.error();
        w.qsa.v_proj = std::move(v.value());
        auto o = load(p + "attn_output.weight", {qOut / 2U, kHidden}, true);
        if (!o) return o.error();
        w.qsa.o_proj = std::move(o.value());
        auto qn = load(p + "attn_q_norm.weight", {Qwen38ExpectedInvariants::head_dim}, false);
        if (!qn) return qn.error();
        w.qsa.q_norm = std::move(qn.value());
        auto kn = load(p + "attn_k_norm.weight", {Qwen38ExpectedInvariants::head_dim}, false);
        if (!kn) return kn.error();
        w.qsa.k_norm = std::move(kn.value());
        auto iq = load(p + "indexer.q_proj.weight", {kHidden, idxQ}, true);
        if (!iq) return iq.error();
        auto ik = load(p + "indexer.k_proj.weight", {kHidden, idxK}, true);
        if (!ik) return ik.error();
        w.qsa.index_qk.reserve(iq.value().size() + ik.value().size());
        w.qsa.index_qk.insert(w.qsa.index_qk.end(), iq.value().begin(), iq.value().end());
        w.qsa.index_qk.insert(w.qsa.index_qk.end(), ik.value().begin(), ik.value().end());
        auto iqn = load(p + "indexer.q_norm.weight", {Qwen38ExpectedInvariants::indexer_head_dim}, false);
        if (!iqn) return iqn.error();
        w.qsa.q_layernorm = std::move(iqn.value());
        auto ikn = load(p + "indexer.k_norm.weight", {Qwen38ExpectedInvariants::indexer_head_dim}, false);
        if (!ikn) return ikn.error();
        w.qsa.k_layernorm = std::move(ikn.value());
    }

    {
        auto router = load(names.router, {kHidden, kExperts}, true);
        if (!router) return router.error();
        w.router = std::move(router.value());
    }
    {
        auto sg = load(names.shared_gate, {kHidden}, false);
        if (!sg) return sg.error();
        w.shared_mlp.shared_gate = std::move(sg.value());
    }
    {
        auto gp = load(names.shared_gate_proj, {kHidden, kInter}, true);
        if (!gp) return gp.error();
        w.shared_mlp.gate_proj = std::move(gp.value());
    }
    {
        auto up = load(names.shared_up_proj, {kHidden, kInter}, true);
        if (!up) return up.error();
        w.shared_mlp.up_proj = std::move(up.value());
    }
    {
        auto dp = load(names.shared_down_proj, {kInter, kHidden}, true);
        if (!dp) return dp.error();
        w.shared_mlp.down_proj = std::move(dp.value());
    }

    // Fused expert tensors: [in, out, 512] ggml; expert k slice at k*stride.
    auto expert_source = [&](const std::string& name) -> Result<Qwen38LayerTrunk::Qwen38GgufExpertSource> {
        const auto loc = checkpoint.value().catalog.tensors.find(name);
        if (loc == checkpoint.value().catalog.tensors.end()) {
            return make_error(ErrorCode::not_found, "fused expert tensor missing", name);
        }
        const model::GgufFile& shard = checkpoint.value().catalog.shards[loc->second.shard_index];
        const model::GgufTensorDescriptor& desc = shard.tensors()[loc->second.tensor_index];
        if (!desc.encoded_bytes.has_value() || desc.shape.size() != 3U
            || desc.shape[2U] != kExperts || desc.element_count % kExperts != 0U) {
            return make_error(ErrorCode::invalid_format, "fused expert tensor shape invalid", name);
        }
        Qwen38LayerTrunk::Qwen38GgufExpertSource src;
        src.shard = shard.path();
        src.data_offset = desc.absolute_offset;
        src.expert_stride_bytes = desc.encoded_bytes.value() / kExperts;
        return src;
    };
    {
        auto g = expert_source(names.gate_exps);
        if (!g) return g.error();
        w.gguf_gate = std::move(g.value());
        auto u = expert_source(names.up_exps);
        if (!u) return u.error();
        w.gguf_up = std::move(u.value());
        auto d = expert_source(names.down_exps);
        if (!d) return d.error();
        w.gguf_down = std::move(d.value());
    }

    if (telemetry) {
        telemetry->load_ms = std::chrono::steady_clock::now() - t0;
    }
    return w;
}

Result<std::vector<float>> qwen38_load_gguf_embed_row(
    const std::filesystem::path& checkpoint_root,
    std::uint32_t token_id,
    Qwen38Layer0Telemetry* telemetry) {
    auto checkpoint = qwen38_open_gguf_checkpoint(checkpoint_root);
    if (!checkpoint) return checkpoint.error();
    // token_embd ggml [2560, 248320]: token k column occupies Q8_0 blocks
    // [k * (2560/32), ...) — contiguous bytes because 2560 % 32 == 0.
    constexpr std::uint64_t kRow = Qwen38ExpectedInvariants::hidden_size;
    const std::string name = "token_embd.weight";
    const auto loc = checkpoint.value().catalog.tensors.find(name);
    if (loc == checkpoint.value().catalog.tensors.end()) {
        return make_error(ErrorCode::not_found, "token_embd missing", name);
    }
    const model::GgufFile& shard = checkpoint.value().catalog.shards[loc->second.shard_index];
    const model::GgufTensorDescriptor& desc = shard.tensors()[loc->second.tensor_index];
    if (desc.ggml_type != 8U || desc.shape.size() != 2U || desc.shape[0] != kRow) {
        return make_error(ErrorCode::invalid_format, "token_embd layout drift", name);
    }
    if (token_id >= desc.shape[1]) {
        return make_error(ErrorCode::invalid_argument, "token id out of vocab",
                          std::to_string(token_id));
    }
    const std::uint64_t blocks_per_row = kRow / 32U;
    const std::uint64_t row_bytes = blocks_per_row * 34U;
    const std::uint64_t begin = desc.absolute_offset + static_cast<std::uint64_t>(token_id) * row_bytes;
    if (begin + row_bytes > shard.file_bytes()) {
        return make_error(ErrorCode::invalid_format, "token row out of bounds", name);
    }
    std::ifstream in(shard.path(), std::ios::binary);
    if (!in) {
        return make_error(ErrorCode::io_error, "cannot open embedding shard", shard.path().string());
    }
    in.seekg(static_cast<std::streamoff>(begin));
    std::vector<std::byte> payload(static_cast<std::size_t>(row_bytes));
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(row_bytes));
    if (static_cast<std::uint64_t>(in.gcount()) != row_bytes) {
        return make_error(ErrorCode::io_error, "short read of embedding row", name);
    }
    auto row = qwen38_decode_q8_0(payload, kRow);
    if (!row) return row.error();
    if (auto r = require_finite_local(row.value(), "gguf_embed_row"); !r) return r.error();
    if (telemetry) {
        telemetry->payload_bytes_read += row_bytes;
        telemetry->tensors_read += 1U;
    }
    return row;
}

Result<Qwen38OwnedExpert> qwen38_load_gguf_expert(
    const Qwen38LayerTrunk& trunk,
    const Qwen38GgufCheckpoint& /*checkpoint*/,
    std::uint32_t expert_id,
    Qwen38Layer0Telemetry* telemetry) {
    if (expert_id >= Qwen38ExpectedInvariants::num_experts) {
        return make_error(ErrorCode::invalid_argument, "expert id out of range",
                          std::to_string(expert_id));
    }
    const std::uint64_t kRow = Qwen38ExpectedInvariants::hidden_size;
    const std::uint64_t inter = Qwen38ExpectedInvariants::moe_intermediate_size;
    auto slice = [&](const Qwen38LayerTrunk::Qwen38GgufExpertSource& src,
                     std::uint64_t in_dim,
                     std::uint64_t out_dim) -> Result<std::vector<float>> {
        const std::uint64_t elements = in_dim * out_dim;
        const std::uint64_t blocks = (elements + 31U) / 32U;
        const std::uint64_t span = blocks * 34U;
        if (src.expert_stride_bytes != span) {
            return make_error(ErrorCode::invalid_format, "expert slice stride drift",
                              src.shard.string());
        }
        const std::uint64_t begin = src.data_offset
            + static_cast<std::uint64_t>(expert_id) * src.expert_stride_bytes;
        if (begin + span > std::filesystem::file_size(src.shard)) {
            return make_error(ErrorCode::invalid_format, "expert slice out of bounds",
                              src.shard.string());
        }
        std::ifstream in(src.shard, std::ios::binary);
        if (!in) {
            return make_error(ErrorCode::io_error, "cannot open expert shard", src.shard.string());
        }
        in.seekg(static_cast<std::streamoff>(begin));
        std::vector<std::byte> payload(static_cast<std::size_t>(span));
        in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(span));
        if (static_cast<std::uint64_t>(in.gcount()) != span) {
            return make_error(ErrorCode::io_error, "short read of expert slice",
                              src.shard.string());
        }
        if (telemetry) {
            telemetry->payload_bytes_read += span;
            telemetry->expert_payload_bytes += span;
        }
        auto flat = qwen38_decode_q8_0(payload, elements);
        if (!flat) return flat.error();
        // ggml per-expert [in, out] flat == row-major [out, in].
        std::vector<float> out(flat.value().size());
        std::memcpy(out.data(), flat.value().data(), flat.value().size() * sizeof(float));
        std::ignore = in_dim;
        std::ignore = out_dim;
        return out;
    };
    Qwen38OwnedExpert expert;
    auto gate = slice(trunk.gguf_gate, kRow, inter);
    if (!gate) return gate.error();
    if (auto r = require_finite_local(gate.value(), "gguf_expert_gate"); !r) return r.error();
    auto up = slice(trunk.gguf_up, kRow, inter);
    if (!up) return up.error();
    if (auto r = require_finite_local(up.value(), "gguf_expert_up"); !r) return r.error();
    auto down = slice(trunk.gguf_down, inter, kRow);
    if (!down) return down.error();
    if (auto r = require_finite_local(down.value(), "gguf_expert_down"); !r) return r.error();
    expert.gate_proj = std::move(gate.value());
    expert.up_proj = std::move(up.value());
    expert.down_proj = std::move(down.value());
    return expert;
}

}  // namespace vastloom::inference
