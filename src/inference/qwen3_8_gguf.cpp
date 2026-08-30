// Qwen3.8-Flash-Next GGUF (qwen4exp) checkpoint source.
//
// Reads llama.cpp-format Q8_0 (and friends) GGUF shards through the generic
// model::GgufSplitCatalog and converts the ggml [in, out] layout into the
// row-major [out, in] layout the reference operators use. Routed experts live
// in fused [in, out, 512] tensors; one expert slice is a fixed byte stride.

#include "vastloom/inference/qwen3_8_flash_next.hpp"
#include "vastloom/inference/qwen3_8_flash_next_reference.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <functional>
#include <string>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/model/gguf.hpp"

namespace vastloom::inference {

// [performance] Small static thread pool for the hot scalar kernels. The
// reference operators stay single-threaded; the GGUF forward path uses this
// fan-out because a 6B-active decode is otherwise compute-bound on one core.
namespace {

unsigned int hw_threads() {
    static const unsigned int n = [] {
        const auto h = std::thread::hardware_concurrency();
        return h == 0U ? 2U : (std::min)(h, 16U);
    }();
    return n;
}

void parallel_for(std::uint64_t begin, std::uint64_t end, std::uint64_t grain,
                  const std::function<void(std::uint64_t, std::uint64_t)>& fn) {
    const std::uint64_t total = end > begin ? end - begin : 0U;
    if (total == 0U) return;
    const unsigned int n = hw_threads();
    if (n == 1U || total <= grain) {
        fn(begin, end);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(n - 1U);
    const std::uint64_t chunk = (total + n - 1U) / n;
    for (unsigned int t = 1U; t < n; ++t) {
        const std::uint64_t lo = begin + t * chunk;
        if (lo >= end) break;
        const std::uint64_t hi = (std::min)(end, lo + chunk);
        threads.emplace_back([&fn, lo, hi] { fn(lo, hi); });
    }
    fn(begin, (std::min)(end, begin + chunk));
    for (auto& th : threads) th.join();
}

}  // namespace

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

// ggml flat index = i0 + i1 * ne0 (ne0 contiguous); that byte order IS the
// row-major [out, in] layout every operator consumes, so this "transpose"
// is an element-preserving copy. Nothing is permuted; the name describes
// the caller's intent, and the copy exists because the read path returns
// an owned buffer either way.
std::vector<float> identity_copy_row_major(std::span<const float> flat) {
    std::vector<float> out(flat.size());
    std::memcpy(out.data(), flat.data(), flat.size() * sizeof(float));
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


// llama.cpp GGUF norm weights are stored already as (1 + w) because the
// reference kernels multiply them plainly; our rms_norm_affine adds 1 back.
// Shifting by -1 at load time keeps the operator untouched.
void shift_norm_weight(std::vector<float>& w) {
    for (float& v : w) {
        v -= 1.0F;
    }
}
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
    if (blocks < 4096U) {
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
    parallel_for(0U, blocks, 8192U, [&](std::uint64_t lo, std::uint64_t hi) {
        for (std::uint64_t b = lo; b < hi; ++b) {
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
    });
    return out;
}


// One persistent read stream per shard file: the forward path opens each
// tensor read on the same file 20+ times per layer, and reopen+close on a
// 46 GiB shard costs real milliseconds each time.
std::mutex& stream_cache_mutex() {
    static std::mutex m;
    return m;
}

std::ifstream& persistent_shard_stream(const std::filesystem::path& path) {
    static std::map<std::string, std::ifstream> streams;
    std::lock_guard<std::mutex> lock(stream_cache_mutex());
    auto it = streams.find(path.string());
    if (it == streams.end()) {
        it = streams.emplace(path.string(), std::ifstream(path, std::ios::binary)).first;
    }
    return it->second;
}

// [performance] Resident raw Q8_0 payload cache: the forward path re-reads
// the same trunk tensors every token. Caching the raw Q8_0 bytes (~7.3 GiB
// for all 48 layer trunks) removes the NVMe re-read while keeping the
// dequantization on the compute path. Bounded by an aggregate cap.
std::mutex& raw_cache_mutex() {
    static std::mutex m;
    return m;
}

struct RawCacheEntry {
    std::vector<std::byte> bytes;
};

std::map<std::string, RawCacheEntry>& raw_cache() {
    static std::map<std::string, RawCacheEntry> cache;
    return cache;
}

std::uint64_t& raw_cache_bytes() {
    static std::uint64_t total = 0U;
    return total;
}

constexpr std::uint64_t kRawCacheCapBytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;

std::atomic<std::uint64_t> raw_cache_fetches{0U};
std::atomic<std::uint64_t> raw_cache_hits{0U};

bool& raw_cache_enabled() {
    static bool enabled = true;
    return enabled;
}

std::pair<std::uint64_t, std::uint64_t> qwen38_raw_cache_stats() {
    return {raw_cache_fetches.load(std::memory_order_relaxed),
            raw_cache_hits.load(std::memory_order_relaxed)};
}

void qwen38_set_raw_cache_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(raw_cache_mutex());
    raw_cache_enabled() = enabled;
}

const std::vector<std::byte>* raw_cache_fetch(const std::string& key) {
    std::lock_guard<std::mutex> lock(raw_cache_mutex());
    if (!raw_cache_enabled()) return nullptr;
    raw_cache_fetches.fetch_add(1U, std::memory_order_relaxed);
    auto it = raw_cache().find(key);
    if (it != raw_cache().end()) {
        raw_cache_hits.fetch_add(1U, std::memory_order_relaxed);
    }
    if (it == raw_cache().end()) return nullptr;
    return &it->second.bytes;
}

// Returns true when the cache accepted the payload (the buffer is consumed);
// false when disabled, full, or out of memory (the buffer is left intact).
bool raw_cache_store(const std::string& key, std::vector<std::byte>&& bytes) {
    std::lock_guard<std::mutex> lock(raw_cache_mutex());
    if (!raw_cache_enabled()) return false;
    const std::size_t size = bytes.size();
    if (raw_cache_bytes() + size > kRawCacheCapBytes) return false;
    try {
        // Move the payload into the map so a store never doubles its byte
        // footprint (the caller's buffer is transferred, not copied).
        raw_cache()[key].bytes = std::move(bytes);
    } catch (const std::bad_alloc&) {
        // Governor discipline: on memory pressure drop the cache instead of
        // failing the request. The byte budget stays conservative; commit
        // pressure (Job limits, other tenants) is discovered right here.
        raw_cache_enabled() = false;
        raw_cache().clear();
        raw_cache_bytes() = 0U;
        return false;
    }
    raw_cache_bytes() += size;
    return true;
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
        if (ec) {
            return make_error(ErrorCode::io_error, "cannot scan checkpoint root",
                              ec.message() + " [" + checkpoint_root.string() + "]");
        }
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
        // Retain the PLE metadata arrays (multipliers/offsets/vocab sizes).
        model::GgufReaderLimits limits;
        limits.retained_integral_array_keys = {
            "qwen4exp.ple.layer_multipliers",
            "qwen4exp.ple.head_offsets",
            "qwen4exp.ple.head_vocab_sizes",
        };
        auto file = model::GgufFile::open(p, limits);
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
    const std::string cache_key = shard.path().string() + "#" + std::to_string(begin)
        + "#" + std::to_string(span);
    const std::vector<std::byte>* cached = raw_cache_fetch(cache_key);
    std::vector<std::byte> owned;
    std::span<const std::byte> payload;
    if (cached != nullptr) {
        // Cache hit: dequantize straight from the resident bytes (no copy).
        payload = *cached;
    } else {
        std::ifstream& in = persistent_shard_stream(shard.path());
        if (!in) {
            return make_error(ErrorCode::io_error, "cannot open gguf shard", shard.path().string());
        }
        in.seekg(static_cast<std::streamoff>(begin));
        owned.resize(static_cast<std::size_t>(span));
        in.read(reinterpret_cast<char*>(owned.data()), static_cast<std::streamsize>(span));
        if (static_cast<std::uint64_t>(in.gcount()) != span) {
            return make_error(ErrorCode::io_error, "short read of gguf tensor",
                              std::string{tensor_name});
        }
        // Move the payload into the cache when it accepts it; otherwise the
        // buffer stays owned here and is used directly below.
        auto candidate = std::move(owned);
        if (raw_cache_store(cache_key, std::move(candidate))) {
            cached = raw_cache_fetch(cache_key);
            payload = cached != nullptr ? std::span<const std::byte>{*cached} : candidate;
        } else {
            payload = candidate;
        }
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
        return identity_copy_row_major(flat);
    }
    return flat;
}

// Matrix reader that keeps Q8_0 tensors in native block form. The ggml flat
// layout already matches the row-major [out, in] convention every operator
// consumes (the "transpose" pass is an identity copy), so a Q8_0 tensor can
// be stored as raw blocks and read by linear_w with results bit-identical to
// the dequantized F32 form at a quarter of the RAM. Any other type (or a
// tensor failing the block-shape checks) falls back to F32 unchanged.
Result<Qwen38Weight> qwen38_read_gguf_weight(
    const Qwen38GgufCheckpoint& checkpoint,
    std::string_view tensor_name,
    const std::vector<std::uint64_t>& expected_dims,
    Qwen38Layer0Telemetry* telemetry) {
    const auto loc = checkpoint.catalog.tensors.find(std::string{tensor_name});
    if (loc != checkpoint.catalog.tensors.end()) {
        const model::GgufFile& shard = checkpoint.catalog.shards[loc->second.shard_index];
        const model::GgufTensorDescriptor& desc = shard.tensors()[loc->second.tensor_index];
        bool shape_ok = desc.shape.size() == expected_dims.size();
        for (std::size_t i = 0; shape_ok && i < expected_dims.size(); ++i) {
            shape_ok = desc.shape[i] == expected_dims[i];
        }
        if (shape_ok && desc.ggml_type == 8U && desc.encoded_bytes.has_value()
            && desc.element_count % 32U == 0U) {
            const std::uint64_t blocks = desc.element_count / 32U;
            const std::uint64_t span = desc.encoded_bytes.value();
            const std::uint64_t begin = desc.absolute_offset;
            if (span == blocks * 34U && begin + span <= shard.file_bytes()) {
                const std::string cache_key = shard.path().string() + "#"
                    + std::to_string(begin) + "#" + std::to_string(span);
                std::vector<std::byte> bytes;
                if (const std::vector<std::byte>* cached = raw_cache_fetch(cache_key)) {
                    bytes = *cached;
                } else {
                    std::ifstream& in = persistent_shard_stream(shard.path());
                    if (!in) {
                        return make_error(ErrorCode::io_error, "cannot open gguf shard",
                                          shard.path().string());
                    }
                    in.seekg(static_cast<std::streamoff>(begin));
                    bytes.resize(static_cast<std::size_t>(span));
                    in.read(reinterpret_cast<char*>(bytes.data()),
                            static_cast<std::streamsize>(span));
                    if (static_cast<std::uint64_t>(in.gcount()) != span) {
                        return make_error(ErrorCode::io_error, "short read of gguf tensor",
                                          std::string{tensor_name});
                    }
                    // The cache keeps its own copy only when it is enabled (resident trunks
                    // disable it); the weight always owns the original bytes.
                    if (raw_cache_enabled() && !bytes.empty()) {
                        raw_cache_store(cache_key, std::vector<std::byte>{bytes});
                    }
                }
                if (telemetry) {
                    telemetry->payload_bytes_read += span;
                    telemetry->tensors_read += 1U;
                }
                Qwen38Weight weight;
                weight.q8 = std::move(bytes);
                return weight;
            }
        }
    }
    auto f32 = qwen38_read_gguf_tensor_f32(checkpoint, tensor_name, expected_dims, false,
                                           telemetry);
    if (!f32) return f32.error();
    Qwen38Weight weight;
    weight.f32 = std::move(f32.value());
    return weight;
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
    w.gguf_root = checkpoint_root;

    auto load = [&](const std::string& name,
                    const std::vector<std::uint64_t>& dims,
                    bool transpose) -> Result<std::vector<float>> {
        return qwen38_read_gguf_tensor_f32(checkpoint.value(), name, dims, transpose, telemetry);
    };
    auto load_w = [&](const std::string& name,
                      const std::vector<std::uint64_t>& dims) -> Result<Qwen38Weight> {
        return qwen38_read_gguf_weight(checkpoint.value(), name, dims, telemetry);
    };

    auto load_residual = [&](const std::string& norm, const std::string& down,
                             const std::string& up, const std::string& inject,
                             Qwen38OwnedResidual& out) -> Result<void> {
        auto hc_norm = load(norm, {kStream}, false);
        if (!hc_norm) return hc_norm.error();
        auto mix_down = load_w(down, {kStream, kRank});
        if (!mix_down) return mix_down.error();
        auto mix_up = load_w(up, {kRank, kStream});
        if (!mix_up) return mix_up.error();
        auto inject_w = load_w(inject, {kStream, Qwen38ExpectedInvariants::hc_count});
        if (!inject_w) return inject_w.error();
        shift_norm_weight(hc_norm.value());
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
        auto qkv = load_w(p + "attn_qkv.weight", {kHidden, 2U * kDim + vDim});
        if (!qkv) return qkv.error();
        w.delta_net.in_proj_qkv = std::move(qkv.value());
        auto z = load_w(p + "attn_gate.weight", {kHidden, vDim});
        if (!z) return z.error();
        w.delta_net.in_proj_z = std::move(z.value());
        auto a = load_w(b + "alpha.weight", {kHidden, Qwen38ExpectedInvariants::linear_num_value_heads});
        if (!a) return a.error();
        w.delta_net.in_proj_a = std::move(a.value());
        auto bb = load_w(b + "beta.weight", {kHidden, Qwen38ExpectedInvariants::linear_num_value_heads});
        if (!bb) return bb.error();
        w.delta_net.in_proj_b = std::move(bb.value());
        auto conv = load(b + "conv1d.weight",
                         {Qwen38ExpectedInvariants::linear_conv_kernel_dim, 2U * kDim + vDim}, false);
        if (!conv) return conv.error();
        w.delta_net.conv1d = std::move(conv.value());
        auto a_log = load(b + "a", {Qwen38ExpectedInvariants::linear_num_value_heads}, false);
        if (!a_log) return a_log.error();
        // The GGUF conversion pre-applies -exp(A_log): ssm_a IS the log-space
        // decay gate (always <= 0), used directly by llama.cpp as
        // gate = softplus(a + dt) * ssm_a. Our operator reconstructs
        // g = -exp(a_log) * softplus(...), so we store log(-ssm_a), which
        // reproduces g = ssm_a * softplus(...) exactly.
        for (std::size_t i = 0; i < a_log.value().size(); ++i) {
            const float v = a_log.value()[i];
            if (v >= 0.0F) {
                return make_error(ErrorCode::invalid_format, "ssm_a must be negative",
                                  "layer " + std::to_string(layer));
            }
            a_log.value()[i] = std::log(-v);
        }
        w.delta_net.a_log = std::move(a_log.value());
        auto dt = load(b + "dt.bias", {Qwen38ExpectedInvariants::linear_num_value_heads}, false);
        if (!dt) return dt.error();
        w.delta_net.dt_bias = std::move(dt.value());
        auto norm = load(b + "norm.weight", {Qwen38ExpectedInvariants::linear_value_head_dim}, false);
        if (!norm) return norm.error();
        w.delta_net.norm = std::move(norm.value());
        auto out = load_w(b + "out.weight", {vDim, kHidden});
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
        auto q = load_w(p + "attn_q.weight", {kHidden, qOut});
        if (!q) return q.error();
        w.qsa.q_proj = std::move(q.value());
        auto k = load_w(p + "attn_k.weight", {kHidden, kvOut});
        if (!k) return k.error();
        w.qsa.k_proj = std::move(k.value());
        auto v = load_w(p + "attn_v.weight", {kHidden, kvOut});
        if (!v) return v.error();
        w.qsa.v_proj = std::move(v.value());
        auto o = load_w(p + "attn_output.weight", {qOut / 2U, kHidden});
        if (!o) return o.error();
        w.qsa.o_proj = std::move(o.value());
        auto qn = load(p + "attn_q_norm.weight", {Qwen38ExpectedInvariants::head_dim}, false);
        if (!qn) return qn.error();
        shift_norm_weight(qn.value());
        w.qsa.q_norm = std::move(qn.value());
        auto kn = load(p + "attn_k_norm.weight", {Qwen38ExpectedInvariants::head_dim}, false);
        if (!kn) return kn.error();
        shift_norm_weight(kn.value());
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
        shift_norm_weight(iqn.value());
        w.qsa.q_layernorm = std::move(iqn.value());
        auto ikn = load(p + "indexer.k_norm.weight", {Qwen38ExpectedInvariants::indexer_head_dim}, false);
        if (!ikn) return ikn.error();
        shift_norm_weight(ikn.value());
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
        auto gp = load_w(names.shared_gate_proj, {kHidden, kInter});
        if (!gp) return gp.error();
        w.shared_mlp.gate_proj = std::move(gp.value());
    }
    {
        auto up = load_w(names.shared_up_proj, {kHidden, kInter});
        if (!up) return up.error();
        w.shared_mlp.up_proj = std::move(up.value());
    }
    {
        auto dp = load_w(names.shared_down_proj, {kInter, kHidden});
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

    // PLE weights (layer 1 only) + metadata arrays.
    if (layer == 1U) {
        auto key = load(p + "ple_key.weight", {kHidden, kStream}, true);
        if (!key) return key.error();
        w.ple.key_proj = std::move(key.value());
        auto value = load(p + "ple_value.weight", {kHidden, kHidden}, true);
        if (!value) return value.error();
        w.ple.value_proj = std::move(value.value());
        auto nk = load(p + "ple_norm_key.weight", {kStream}, false);
        if (!nk) return nk.error();
        shift_norm_weight(nk.value());
        w.ple.norm_key = std::move(nk.value());
        auto nq = load(p + "ple_norm_query.weight", {kStream}, false);
        if (!nq) return nq.error();
        shift_norm_weight(nq.value());
        w.ple.norm_query = std::move(nq.value());
        auto nc = load(p + "ple_norm_conv.weight", {kStream}, false);
        if (!nc) return nc.error();
        shift_norm_weight(nc.value());
        w.ple.norm_conv = std::move(nc.value());
        auto conv = load(p + "ple_conv1d.weight",
                         {4U, kStream}, false);
        if (!conv) return conv.error();
        w.ple.conv1d = std::move(conv.value());
        auto read_meta = [&](const char* key_name, std::uint64_t* out,
                             std::size_t count) -> Result<void> {
            const auto* entry = checkpoint.value().catalog.shards.front().find_metadata(key_name);
            if (entry == nullptr || entry->unsigned_array_values.size() < count) {
                return make_error(ErrorCode::not_found, "PLE metadata array missing", key_name);
            }
            for (std::size_t i = 0; i < count; ++i) {
                out[i] = entry->unsigned_array_values[i];
            }
            return {};
        };
        if (auto r = read_meta("qwen4exp.ple.layer_multipliers", w.ple.multipliers, 3U); !r) {
            return r.error();
        }
        if (auto r = read_meta("qwen4exp.ple.head_offsets", w.ple.head_offsets, 16U); !r) {
            return r.error();
        }
        if (auto r = read_meta("qwen4exp.ple.head_vocab_sizes", w.ple.head_vocab_sizes, 16U); !r) {
            return r.error();
        }
        w.has_ple = true;
    }

    if (telemetry) {
        telemetry->load_ms = std::chrono::steady_clock::now() - t0;
    }
    return w;
}

Result<std::vector<float>> qwen38_ple_forward(
    const Qwen38LayerTrunk& trunk,
    const Qwen38GgufCheckpoint& checkpoint,
    Qwen38PleRuntime& state,
    std::span<const float> input_stream,
    std::uint32_t token_id,
    Qwen38Layer0Telemetry* telemetry) {
    if (!trunk.has_ple) {
        return make_error(ErrorCode::invalid_argument, "PLE forward on layer without PLE",
                          std::to_string(trunk.layer));
    }
    constexpr std::uint32_t kEos = Qwen38ExpectedInvariants::eos_token_id;
    constexpr std::uint64_t kPleHeadDim = 160U;
    constexpr std::uint32_t kHeads = 16U;
    constexpr std::uint32_t kEmbDim = kHeads * kPleHeadDim;  // 2560
    const std::uint32_t H = Qwen38ExpectedInvariants::hidden_size;
    const std::uint32_t C = Qwen38ExpectedInvariants::hc_count;

    if (!state.context_initialized) {
        state.previous_context[0] = kEos;
        state.previous_context[1] = kEos;
        state.conv_state.assign(static_cast<std::size_t>(kStream) * 9U, 0.0F);
        state.context_initialized = true;
    }
    // Token history: [prev(2), current]; shifted[s] pads with EOS.
    std::uint32_t history[3] = {state.previous_context[0], state.previous_context[1], token_id};
    std::uint32_t shifted[3][3];
    for (std::uint32_t s = 0; s < 3U; ++s) {
        for (std::uint32_t i = 0; i < 3U; ++i) {
            shifted[s][i] = (i >= s) ? history[i - s] : kEos;
        }
    }
    // n-gram hash per head (ngram 2 -> heads 0..7, ngram 3 -> heads 8..15).
    std::uint64_t ids[kHeads];
    for (std::uint32_t g = 0; g < 2U; ++g) {
        const std::uint32_t ngram = g + 2U;
        std::uint64_t mixed = static_cast<std::uint64_t>(shifted[0][2]) * trunk.ple.multipliers[0];
        for (std::uint32_t position = 1U; position < ngram; ++position) {
            mixed ^= static_cast<std::uint64_t>(shifted[position][2]) * trunk.ple.multipliers[position];
        }
        for (std::uint32_t h = 0; h < 8U; ++h) {
            const std::uint32_t head = g * 8U + h;
            ids[head] = mixed % trunk.ple.head_vocab_sizes[head] + trunk.ple.head_offsets[head];
        }
    }

    // Embedding lookup: one 160-element column per id (contiguous Q8_0 spans).
    const auto& catalog = checkpoint.catalog;
    const auto loc = catalog.tensors.find("per_layer_token_embd.weight");
    if (loc == catalog.tensors.end()) {
        return make_error(ErrorCode::not_found, "per_layer_token_embd missing", "ple");
    }
    const model::GgufFile& shard = catalog.shards[loc->second.shard_index];
    const model::GgufTensorDescriptor& desc = shard.tensors()[loc->second.tensor_index];
    if (desc.ggml_type != 8U || desc.shape.size() != 2U || desc.shape[0] != kPleHeadDim) {
        return make_error(ErrorCode::invalid_format, "per_layer_token_embd layout drift", "ple");
    }
    constexpr std::uint64_t kBlocksPerColumn = kPleHeadDim / 32U;
    constexpr std::uint64_t kColumnBytes = kBlocksPerColumn * 34U;
    std::vector<float> embeddings(kEmbDim, 0.0F);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
        if (ids[h] >= desc.shape[1]) {
            return make_error(ErrorCode::invalid_format, "ple id out of table",
                              std::to_string(ids[h]));
        }
        const std::uint64_t begin = desc.absolute_offset + ids[h] * kColumnBytes;
        std::ifstream in(shard.path(), std::ios::binary);
        if (!in) {
            return make_error(ErrorCode::io_error, "cannot open ple shard", shard.path().string());
        }
        in.seekg(static_cast<std::streamoff>(begin));
        std::vector<std::byte> payload(static_cast<std::size_t>(kColumnBytes));
        in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(kColumnBytes));
        if (static_cast<std::uint64_t>(in.gcount()) != kColumnBytes) {
            return make_error(ErrorCode::io_error, "short read of ple column", "ple");
        }
        auto column = qwen38_decode_q8_0(payload, kPleHeadDim);
        if (!column) return column.error();
        for (std::uint32_t i = 0; i < kPleHeadDim; ++i) {
            embeddings[static_cast<std::size_t>(h) * kPleHeadDim + i] = column.value()[i];
        }
        if (telemetry) {
            telemetry->payload_bytes_read += kColumnBytes;
            telemetry->tensors_read += 1U;
        }
    }

    // key/value projections + norm/gate (Qwen4ExpTextPLELayer.forward).
    auto key = qwen38_linear(embeddings, trunk.ple.key_proj, kStream, kEmbDim);
    // silu local to this TU (the reference one lives in an anonymous namespace).
    auto ple_silu = [](float v) { return v / (1.0F + std::exp(-v)); };
    if (!key) return key.error();
    auto key_normed = qwen38_rms_norm_affine(key.value(), trunk.ple.norm_key, 1.0e-6F, H);
    if (!key_normed) return key_normed.error();
    auto value = qwen38_linear(embeddings, trunk.ple.value_proj, H, kEmbDim);
    if (!value) return value.error();
    auto query_normed = qwen38_rms_norm_affine(input_stream, trunk.ple.norm_query, 1.0e-6F, H);
    if (!query_normed) return query_normed.error();

    std::vector<float> gated_value(kStream, 0.0F);
    const double gate_scale = 1.0 / std::sqrt(static_cast<double>(H));
    for (std::uint32_t b = 0; b < C; ++b) {
        double gate = 0.0;
        for (std::uint32_t i = 0; i < H; ++i) {
            gate += static_cast<double>(key_normed.value()[b * H + i])
                * static_cast<double>(query_normed.value()[b * H + i]);
        }
        gate *= gate_scale;
        const double mag = std::max(std::fabs(gate), 1.0e-6);
        const double shaped = std::sqrt(mag) * (gate < 0.0 ? -1.0 : (gate > 0.0 ? 1.0 : 0.0));
        const double sig = 1.0 / (1.0 + std::exp(-shaped));
        for (std::uint32_t i = 0; i < H; ++i) {
            gated_value[b * H + i] = static_cast<float>(sig * static_cast<double>(value.value()[i]));
        }
    }
    auto gated_normed = qwen38_rms_norm_affine(gated_value, trunk.ple.norm_conv, 1.0e-6F, H);
    if (!gated_normed) return gated_normed.error();

    // Dilated depthwise short conv (dilation 3, kernel 4, state 9) + silu.
    if (state.conv_state.size() != static_cast<std::size_t>(kStream) * 9U) {
        return make_error(ErrorCode::invalid_format, "ple conv state size drift", "ple");
    }
    std::vector<float> conv_out(kStream, 0.0F);
    std::vector<float> next_state(static_cast<std::size_t>(kStream) * 9U, 0.0F);
    for (std::uint32_t c = 0; c < kStream; ++c) {
        const float* hist = state.conv_state.data() + static_cast<std::size_t>(c) * 9U;
        double acc = 0.0;
        for (std::uint32_t k = 0; k < 3U; ++k) {
            acc += static_cast<double>(trunk.ple.conv1d[c * 4U + k]) * static_cast<double>(hist[k * 3U]);
        }
        acc += static_cast<double>(trunk.ple.conv1d[c * 4U + 3U])
            * static_cast<double>(gated_normed.value()[c]);
        conv_out[c] = ple_silu(static_cast<float>(acc));
        // New state: [old[1..8], current].
        for (std::uint32_t j = 0; j < 8U; ++j) {
            next_state[static_cast<std::size_t>(c) * 9U + j] = hist[j + 1U];
        }
        next_state[static_cast<std::size_t>(c) * 9U + 8U] = gated_normed.value()[c];
    }
    state.conv_state = std::move(next_state);

    // output = gated_value + conv_out, added to the input stream by the caller
    // (kept here as the sum for clarity).
    std::vector<float> out(kStream, 0.0F);
    for (std::uint64_t i = 0; i < kStream; ++i) {
        out[i] = gated_value[i] + conv_out[i];
    }
    return out;
}

Result<std::vector<float>> qwen38_load_gguf_embed_row(    const std::filesystem::path& checkpoint_root,
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

Result<Qwen38OwnedExpertQ8> qwen38_load_gguf_expert_q8(
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
                     std::uint64_t out_dim,
                     std::vector<std::byte>& dest) -> Result<bool> {
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
        // Local stream on purpose: the greedy expert loader issues concurrent
        // slice reads, and the persistent stream mutex would serialize them.
        std::ifstream in(src.shard, std::ios::binary);
        if (!in) {
            return make_error(ErrorCode::io_error, "cannot open expert shard", src.shard.string());
        }
        in.seekg(static_cast<std::streamoff>(begin));
        dest.resize(static_cast<std::size_t>(span));
        in.read(reinterpret_cast<char*>(dest.data()), static_cast<std::streamsize>(span));
        if (static_cast<std::uint64_t>(in.gcount()) != span) {
            return make_error(ErrorCode::io_error, "short read of expert slice",
                              src.shard.string());
        }
        if (telemetry) {
            telemetry->payload_bytes_read += span;
            telemetry->expert_payload_bytes += span;
        }
        return true;
    };
    Qwen38OwnedExpertQ8 expert;
    if (auto r = slice(trunk.gguf_gate, kRow, inter, expert.gate); !r) return r.error();
    if (auto r = slice(trunk.gguf_up, kRow, inter, expert.up); !r) return r.error();
    if (auto r = slice(trunk.gguf_down, inter, kRow, expert.down); !r) return r.error();
    return expert;
}

Result<Qwen38GgufHead> qwen38_load_gguf_head(
    const std::filesystem::path& checkpoint_root,
    Qwen38Layer0Telemetry* telemetry) {
    const auto t0 = std::chrono::steady_clock::now();
    auto checkpoint = qwen38_open_gguf_checkpoint(checkpoint_root);
    if (!checkpoint) return checkpoint.error();
    Qwen38GgufHead head;
    auto norm = qwen38_read_gguf_tensor_f32(checkpoint.value(), "output_hc_norm.weight",
                                            {kStream}, false, telemetry);
    if (!norm) return norm.error();
    auto down = qwen38_read_gguf_tensor_f32(checkpoint.value(), "output_hc_down.weight",
                                            {kStream, kRank}, true, telemetry);
    if (!down) return down.error();
    auto up = qwen38_read_gguf_tensor_f32(checkpoint.value(), "output_hc_up.weight",
                                          {kRank, kStream}, true, telemetry);
    if (!up) return up.error();
    shift_norm_weight(norm.value());
    head.mixer.hc_norm = std::move(norm.value());
    head.mixer.mix_down.f32 = std::move(down.value());
    head.mixer.mix_up.f32 = std::move(up.value());
    auto lm = qwen38_read_gguf_tensor_f32(checkpoint.value(), "output.weight",
                                          {kHidden, Qwen38ExpectedInvariants::vocab_size},
                                          true, telemetry);
    if (!lm) return lm.error();
    head.lm_head = std::move(lm.value());
    if (telemetry) {
        telemetry->load_ms = std::chrono::steady_clock::now() - t0;
    }
    return head;
}

Result<std::vector<float>> qwen38_gguf_logits(
    const Qwen38GgufHead& head,
    std::span<const float> final_stream) {
    const std::uint32_t H = Qwen38ExpectedInvariants::hidden_size;
    const std::uint32_t C = Qwen38ExpectedInvariants::hc_count;
    const std::uint32_t R = Qwen38ExpectedInvariants::hc_lowrank;
    if (final_stream.size() != static_cast<std::size_t>(C) * H) {
        return make_error(ErrorCode::invalid_format, "final stream width drift",
                          std::to_string(final_stream.size()));
    }
    // Global gated-residual mixer with use_combine=false: no block_inject.
    const Qwen38GatedResidualView mixer_view{
        head.mixer.hc_norm, head.mixer.mix_down.ref(), head.mixer.mix_up.ref(), {}};
    auto collapsed = qwen38_gated_residual(final_stream, mixer_view, C, H, R, 1.0e-6F);
    if (!collapsed) return collapsed.error();
    const std::vector<float>& mixed = collapsed.value().mixed_input;
    if (auto r = require_finite_local(mixed, "gguf_head_mixed"); !r) return r.error();

    const std::size_t vocab = Qwen38ExpectedInvariants::vocab_size;
    if (head.lm_head.size() != vocab * H) {
        return make_error(ErrorCode::invalid_format, "lm_head size drift",
                          std::to_string(head.lm_head.size()));
    }
    std::vector<float> logits(vocab, 0.0F);
    parallel_for(0U, vocab, 2048U, [&](std::uint64_t lo, std::uint64_t hi) {
        for (std::uint64_t v = lo; v < hi; ++v) {
            const float* row = head.lm_head.data() + v * H;
            double acc = 0.0;
            for (std::size_t i = 0; i < H; ++i) {
                acc += static_cast<double>(mixed[i]) * static_cast<double>(row[i]);
            }
            logits[v] = static_cast<float>(acc);
        }
    });
    return logits;
}

// --- GGUF vocabulary (tokenizer.ggml.tokens) ---

namespace {

// Minimal GGUF header KV walker: returns the file offset and length of the
// raw value bytes for one metadata key. Works on the metadata-only first
// shard whose header fits in a bounded read.
Result<std::pair<std::uint64_t, std::uint64_t>> locate_gguf_kv(
    const std::filesystem::path& shard_path,
    std::string_view key,
    std::vector<std::byte>& header_buffer) {
    std::ifstream in(shard_path, std::ios::binary);
    if (!in) {
        return make_error(ErrorCode::io_error, "cannot open gguf shard for vocab",
                          shard_path.string());
    }
    in.read(reinterpret_cast<char*>(header_buffer.data()),
            static_cast<std::streamsize>(header_buffer.size()));
    const std::uint64_t loaded = static_cast<std::uint64_t>(in.gcount());
    if (loaded < 24U || header_buffer[0] != std::byte{0x47U}) {
        return make_error(ErrorCode::invalid_format, "not a GGUF header", shard_path.string());
    }
    const auto need = [&](std::uint64_t p, std::uint64_t n) -> Result<std::uint64_t> {
        if (n > loaded || p > loaded - n) {
            return make_error(ErrorCode::invalid_format, "gguf header read out of bounds",
                              shard_path.string());
        }
        return p + n;
    };
    const auto u32_at = [&](std::uint64_t p) -> Result<std::uint32_t> {
        auto end = need(p, 4U);
        if (!end) return end.error();
        std::uint32_t v = 0U;
        std::memcpy(&v, header_buffer.data() + p, 4U);
        return v;
    };
    const auto u64_at = [&](std::uint64_t p) -> Result<std::uint64_t> {
        auto end = need(p, 8U);
        if (!end) return end.error();
        std::uint64_t v = 0U;
        std::memcpy(&v, header_buffer.data() + p, 8U);
        return v;
    };
    const auto str_at = [&](std::uint64_t p, std::string* out) -> Result<std::uint64_t> {
        auto len_result = u64_at(p);
        if (!len_result) return len_result.error();
        const std::uint64_t len = len_result.value();
        auto end = need(p + 8U, len);
        if (!end) return end.error();
        out->assign(reinterpret_cast<const char*>(header_buffer.data()) + p + 8U,
                    static_cast<std::size_t>(len));
        return end.value();
    };
    const auto array_bytes = [&](std::uint64_t et, std::uint64_t n) -> Result<std::uint64_t> {
        std::uint64_t sz = 0U;
        switch (et) {
            case 0U: case 1U: case 7U: sz = 1U; break;
            case 2U: case 3U: sz = 2U; break;
            case 4U: case 5U: case 6U: sz = 4U; break;
            case 10U: case 11U: case 12U: sz = 8U; break;
            default:
                return make_error(ErrorCode::invalid_format,
                                  "gguf array element type unsupported", std::to_string(et));
        }
        if (n > loaded / sz) {
            return make_error(ErrorCode::invalid_format, "gguf array size overflows header",
                              shard_path.string());
        }
        return n * sz;
    };
    std::uint64_t pos = 24U;  // magic + version + tensor_count + kv_count
    auto kv_count_result = u64_at(16U);
    if (!kv_count_result) return kv_count_result.error();
    const std::uint64_t kv_count = kv_count_result.value();
    const std::uint64_t header_end = loaded;
    for (std::uint64_t i = 0; i < kv_count && pos < header_end; ++i) {
        std::string k;
        auto after_key = str_at(pos, &k);
        if (!after_key) return after_key.error();
        pos = after_key.value();
        auto type_result = u32_at(pos);
        if (!type_result) return type_result.error();
        const std::uint32_t type = type_result.value();
        pos += 4U;
        if (k == key) {
            // Value spans until the next KV key start; compute by walking it.
            // Caller only needs scalar/array-of-string values, so walk here.
            std::uint64_t value_begin = pos;
            if (type == 9U) {
                auto et_result = u32_at(pos);
                if (!et_result) return et_result.error();
                const std::uint32_t et = et_result.value();
                pos += 4U;
                auto n_result = u64_at(pos);
                if (!n_result) return n_result.error();
                const std::uint64_t n = n_result.value();
                pos += 8U;
                if (et == 8U) {
                    for (std::uint64_t j = 0; j < n; ++j) {
                        auto next = str_at(pos, &k);
                        if (!next) return next.error();
                        pos = next.value();
                    }
                } else {
                    auto span = array_bytes(et, n);
                    if (!span) return span.error();
                    auto end = need(pos, span.value());
                    if (!end) return end.error();
                    pos = end.value();
                }
            } else if (type == 8U) {
                auto next = str_at(pos, &k);
                if (!next) return next.error();
                pos = next.value();
            } else {
                auto span = array_bytes(type, 1U);
                if (!span) return span.error();
                auto end = need(pos, span.value());
                if (!end) return end.error();
                pos = end.value();
            }
            return std::pair<std::uint64_t, std::uint64_t>{value_begin, pos - value_begin};
        }
        // Skip this KV's value entirely.
        if (type == 9U) {
            auto et_result = u32_at(pos);
            if (!et_result) return et_result.error();
            const std::uint32_t et = et_result.value();
            pos += 4U;
            auto n_result = u64_at(pos);
            if (!n_result) return n_result.error();
            const std::uint64_t n = n_result.value();
            pos += 8U;
            if (et == 8U) {
                for (std::uint64_t j = 0; j < n; ++j) {
                    auto next = str_at(pos, &k);
                    if (!next) return next.error();
                    pos = next.value();
                }
            } else {
                auto span = array_bytes(et, n);
                if (!span) return span.error();
                auto end = need(pos, span.value());
                if (!end) return end.error();
                pos = end.value();
            }
        } else if (type == 8U) {
            auto next = str_at(pos, &k);
            if (!next) return next.error();
            pos = next.value();
        } else {
            auto span = array_bytes(type, 1U);
            if (!span) return span.error();
            auto end = need(pos, span.value());
            if (!end) return end.error();
            pos = end.value();
        }
    }
    return make_error(ErrorCode::not_found, "gguf key not found", std::string{key});
}

}  // namespace

Result<std::vector<std::string>> qwen38_load_gguf_vocab(
    const std::filesystem::path& checkpoint_root) {
    auto checkpoint = qwen38_open_gguf_checkpoint(checkpoint_root);
    if (!checkpoint) return checkpoint.error();
    // tokenizer.ggml.tokens lives in the metadata shard; locate it by walking
    // every shard's header until found (bounded read per shard).
    for (const auto& shard : checkpoint.value().catalog.shards) {
        std::vector<std::byte> header(
            static_cast<std::size_t>((std::min<std::uint64_t>)(shard.file_bytes(), 64U * 1024U * 1024U)));
        auto located = locate_gguf_kv(shard.path(), "tokenizer.ggml.tokens", header);
        if (!located) continue;
        const auto [offset, length] = located.value();
        // Re-walk the value region and collect the strings.
        std::vector<std::byte> value(length);
        std::ifstream in(shard.path(), std::ios::binary);
        if (!in) {
            return make_error(ErrorCode::io_error, "cannot reopen vocab shard", shard.path().string());
        }
        in.seekg(static_cast<std::streamoff>(offset));
        in.read(reinterpret_cast<char*>(value.data()), static_cast<std::streamsize>(length));
        if (static_cast<std::uint64_t>(in.gcount()) != length) {
            return make_error(ErrorCode::io_error, "short read of vocab", shard.path().string());
        }
        std::vector<std::string> vocab;
        vocab.reserve(Qwen38ExpectedInvariants::vocab_size);
        // The value region is [array_type u32][count u64][count strings].
        std::uint64_t pos = 12U;
        while (pos + 8U <= length) {
            std::uint64_t len = 0U;
            std::memcpy(&len, value.data() + pos, 8U);
            pos += 8U;
            if (pos + len > length) {
                return make_error(ErrorCode::invalid_format, "vocab string out of range",
                                  shard.path().string());
            }
            vocab.emplace_back(reinterpret_cast<const char*>(value.data()) + pos,
                               static_cast<std::size_t>(len));
            pos += len;
        }
        if (vocab.size() != Qwen38ExpectedInvariants::vocab_size) {
            return make_error(ErrorCode::invalid_format, "vocab size drift",
                              std::to_string(vocab.size()));
        }
        return vocab;
    }
    return make_error(ErrorCode::not_found, "tokenizer.ggml.tokens not in any shard",
                      checkpoint_root.string());
}

std::string qwen38_gpt2_piece_to_bytes(std::string_view piece) {
    // Inverse of the gpt2 bytes_to_unicode byte-level encoding: printable
    // bytes map to themselves; the remaining 68 bytes map to codepoints 256+n
    // in ascending byte order.
    static const std::map<std::uint32_t, unsigned char> inverse = [] {
        std::map<std::uint32_t, unsigned char> m;
        std::uint32_t n = 0U;
        for (unsigned int b = 0U; b < 256U; ++b) {
            const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1U && b <= 0xACU)
                || (b >= 0xAEU && b <= 0xFFU);
            if (printable) {
                m[b] = static_cast<unsigned char>(b);
            } else {
                m[256U + n] = static_cast<unsigned char>(b);
                ++n;
            }
        }
        return m;
    }();
    std::string out;
    out.reserve(piece.size());
    // The piece is the UTF-8 encoding of mapped codepoints; decode UTF-8 and
    // map each codepoint back to its raw byte.
    std::size_t i = 0U;
    while (i < piece.size()) {
        const auto lead = static_cast<unsigned char>(piece[i]);
        std::uint32_t cp = 0U;
        std::size_t len = 1U;
        if (lead < 0x80U) {
            cp = lead;
        } else if ((lead & 0xE0U) == 0xC0U) {
            cp = lead & 0x1FU;
            len = 2U;
        } else if ((lead & 0xF0U) == 0xE0U) {
            cp = lead & 0x0FU;
            len = 3U;
        } else {
            cp = lead & 0x07U;
            len = 4U;
        }
        bool complete = true;
        for (std::size_t j = 1U; j < len; ++j) {
            if (i + j >= piece.size()) { complete = false; break; }
            cp = (cp << 6U) | (static_cast<unsigned char>(piece[i + j]) & 0x3FU);
        }
        if (complete) {
            auto it = inverse.find(cp);
            if (it != inverse.end()) {
                out.push_back(static_cast<char>(it->second));
            }
        }
        i += len;
    }
    return out;
}

}  // namespace vastloom::inference
