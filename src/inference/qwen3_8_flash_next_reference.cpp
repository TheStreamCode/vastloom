#include "vastloom/inference/qwen3_8_flash_next_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <thread>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

float qwen38_fp8_e4m3_to_f32(std::uint8_t bits) noexcept {
    const std::uint32_t sign = (bits >> 7) & 0x1U;
    const std::uint32_t exp = (bits >> 3) & 0xFU;
    const std::uint32_t mant = bits & 0x7U;

    // E4M3FN special: exp==15 && mant==7 => NaN (no Inf in E4M3FN).
    // exp==15 with mant<7 is finite: (1 + mant/8) * 2^8, up to the 448 max.
    // The real FP8 checkpoints contain these encodings, so treating all of
    // exp==15 as NaN corrupts real weights (found on the official 125B).
    if (exp == 15U && mant == 7U) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    float value = 0.0F;
    if (exp == 0U) {
        // Subnormal: mant/8 * 2^(1-bias) where bias=7 => 2^-6
        // value = (mant / 8.0) * 2^-6
        if (mant == 0U) {
            value = 0.0F;
        } else {
            value = static_cast<float>(mant) / 8.0F * std::exp2(-6.0F);
        }
    } else {
        // Normal: (1 + mant/8) * 2^(exp - bias)
        value = (1.0F + static_cast<float>(mant) / 8.0F) * std::exp2(static_cast<float>(static_cast<int>(exp) - 7));
    }

    return sign ? -value : value;
}

float qwen38_bf16_to_f32(std::uint16_t bits) noexcept {
    std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
    float out = 0.0F;
    std::memcpy(&out, &f32_bits, sizeof(out));
    return out;
}

std::uint16_t qwen38_f32_to_bf16(float value) noexcept {
    std::uint32_t f32_bits = 0U;
    std::memcpy(&f32_bits, &value, sizeof(f32_bits));
    // Round to nearest even: add 0x7FFF + ((f32_bits>>16)&1) before truncation.
    // For the reference we use simple truncation (round toward zero) — bit-exact
    // for the values we pin in tests. Use truncation to keep the reference trivial.
    // Callers that need RNE should implement it separately.
    // Here we do RNE to match the common hardware path.
    std::uint32_t rounding_bias = 0x7FFFU + ((f32_bits >> 16) & 1U);
    f32_bits += rounding_bias;
    return static_cast<std::uint16_t>(f32_bits >> 16);
}

Result<std::vector<float>> qwen38_decode_bf16(
    std::span<const std::byte> payload,
    const std::vector<std::int64_t>& shape) {
    std::size_t elements = 1;
    for (auto d : shape) {
        if (d <= 0) {
            return make_error(ErrorCode::invalid_argument,
                "shape dim must be positive", "bf16_decode shape");
        }
        elements *= static_cast<std::size_t>(d);
    }
    if (payload.size() != elements * 2U) {
        return make_error(ErrorCode::invalid_argument,
            "payload size does not match shape for BF16",
            "expected=" + std::to_string(elements * 2U) + " got=" + std::to_string(payload.size()));
    }
    std::vector<float> out;
    out.reserve(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        std::uint16_t bits = 0U;
        std::memcpy(&bits, payload.data() + i * 2U, 2U);
        // Payload is little-endian BF16 as stored in safetensors (native LE).
        // On LE hosts memcpy gives the correct value directly.
        out.push_back(qwen38_bf16_to_f32(bits));
    }
    return out;
}

Result<std::vector<float>> qwen38_decode_fp8_block_scale(
    std::span<const std::byte> weight_payload,
    std::span<const std::byte> scale_payload,
    const std::vector<std::int64_t>& weight_shape,
    const std::vector<std::int64_t>& scale_shape,
    std::size_t block_rows,
    std::size_t block_cols) {
    if (weight_shape.size() != 2) {
        return make_error(ErrorCode::invalid_argument,
            "FP8 block-scale requires 2D weight shape", "fp8_decode");
    }
    if (scale_shape.size() != 2) {
        return make_error(ErrorCode::invalid_argument,
            "FP8 block-scale requires 2D scale shape", "fp8_decode");
    }
    const std::size_t rows = static_cast<std::size_t>(weight_shape[0]);
    const std::size_t cols = static_cast<std::size_t>(weight_shape[1]);
    const std::size_t scale_rows = static_cast<std::size_t>(scale_shape[0]);
    const std::size_t scale_cols = static_cast<std::size_t>(scale_shape[1]);

    const std::size_t expected_scale_rows = (rows + block_rows - 1) / block_rows;
    const std::size_t expected_scale_cols = (cols + block_cols - 1) / block_cols;
    if (scale_rows != expected_scale_rows || scale_cols != expected_scale_cols) {
        return make_error(ErrorCode::invalid_argument,
            "scale shape does not match block tiling",
            "weight=[" + std::to_string(rows) + "," + std::to_string(cols) + "]"
            + " scale=[" + std::to_string(scale_rows) + "," + std::to_string(scale_cols) + "]"
            + " expected=[" + std::to_string(expected_scale_rows) + "," + std::to_string(expected_scale_cols) + "]");
    }

    const std::size_t weight_elements = rows * cols;
    const std::size_t scale_elements = scale_rows * scale_cols;
    if (weight_payload.size() != weight_elements) {
        return make_error(ErrorCode::invalid_argument,
            "weight payload size mismatch for F8_E4M3",
            "expected=" + std::to_string(weight_elements) + " got=" + std::to_string(weight_payload.size()));
    }
    if (scale_payload.size() != scale_elements * 2U) {
        return make_error(ErrorCode::invalid_argument,
            "scale payload size mismatch for BF16",
            "expected=" + std::to_string(scale_elements * 2U) + " got=" + std::to_string(scale_payload.size()));
    }

    // Decode scales to F32 once.
    std::vector<float> scales_f32;
    scales_f32.reserve(scale_elements);
    for (std::size_t i = 0; i < scale_elements; ++i) {
        std::uint16_t bits = 0U;
        std::memcpy(&bits, scale_payload.data() + i * 2U, 2U);
        scales_f32.push_back(qwen38_bf16_to_f32(bits));
    }

    std::vector<float> out;
    out.reserve(weight_elements);
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t block_row = r / block_rows;
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t block_col = c / block_cols;
            const std::size_t scale_idx = block_row * scale_cols + block_col;
            const float scale = scales_f32[scale_idx];
            const std::size_t weight_idx = r * cols + c;
            std::uint8_t bits = 0U;
            std::memcpy(&bits, weight_payload.data() + weight_idx, 1U);
            const float w = qwen38_fp8_e4m3_to_f32(bits);
            // NaN * scale => NaN, which is the correct OCP behavior (propagate NaN).
            out.push_back(w * scale);
        }
    }
    return out;
}

Result<std::vector<float>> qwen38_decode_tensor_payload(
    std::span<const std::byte> payload,
    std::span<const std::byte> scale_payload,
    const std::string& dtype,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>& scale_shape) {
    if (dtype == "BF16") {
        if (!scale_payload.empty()) {
            return make_error(ErrorCode::invalid_argument,
                "BF16 tensor should not have scale payload", dtype);
        }
        return qwen38_decode_bf16(payload, shape);
    }
    if (dtype == "F8_E4M3") {
        return qwen38_decode_fp8_block_scale(payload, scale_payload, shape, scale_shape, 128, 128);
    }
    if (dtype == "F32") {
        std::size_t elements = 1;
        for (auto d : shape) elements *= static_cast<std::size_t>(d);
        if (payload.size() != elements * 4U) {
            return make_error(ErrorCode::invalid_argument,
                "payload size mismatch for F32", dtype);
        }
        std::vector<float> out(elements);
        std::memcpy(out.data(), payload.data(), payload.size());
        return out;
    }
    return make_error(ErrorCode::unsupported,
        "unsupported dtype for decode", dtype);
}

Result<std::vector<float>> qwen38_rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon) {
    if (input.size() != weight.size()) {
        return make_error(ErrorCode::invalid_argument,
            "rms_norm input and weight must have same size",
            "input=" + std::to_string(input.size()) + " weight=" + std::to_string(weight.size()));
    }
    if (input.empty()) {
        return make_error(ErrorCode::invalid_argument,
            "rms_norm input must be non-empty", "rms_norm");
    }
    double sum_sq = 0.0;
    for (float v : input) sum_sq += static_cast<double>(v) * static_cast<double>(v);
    const double mean_sq = sum_sq / static_cast<double>(input.size());
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(epsilon));
    std::vector<float> out;
    out.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        out.push_back(static_cast<float>(static_cast<double>(input[i]) * inv_rms * static_cast<double>(weight[i])));
    }
    return out;
}

namespace {

constexpr std::uint64_t kSplitmixGamma = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kSplitmixM1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t kSplitmixM2 = 0x94D049BB133111EBULL;
constexpr std::int64_t kNgramPrime1 = 10007;

float sigmoid_f(float x) noexcept {
    if (x >= 0.0F) {
        const float z = std::exp(-x);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(x);
    return z / (1.0F + z);
}

float silu_f(float x) noexcept {
    return x * sigmoid_f(x);
}

float softplus_f(float x) noexcept {
    if (x > 20.0F) {
        return x;
    }
    return std::log1p(std::exp(x));
}

bool is_prime_u32(std::uint32_t value) noexcept {
    if (value < 2U) {
        return false;
    }
    if ((value % 2U) == 0U) {
        return value == 2U;
    }
    for (std::uint32_t d = 3U; d <= value / d; d += 2U) {
        if ((value % d) == 0U) {
            return false;
        }
    }
    return true;
}

// [performance] Thread fan-out for the hot reference kernels. Row-parallel
// only: every output element keeps its single-threaded accumulation order,
// so results are bit-identical to the sequential path.
unsigned int qwen38_hw_threads() {
    static const unsigned int n = [] {
        const auto h = std::thread::hardware_concurrency();
        return h == 0U ? 2U : (std::min)(h, 16U);
    }();
    return n;
}

void qwen38_parallel_for(std::uint64_t begin, std::uint64_t end, std::uint64_t grain,
                         const std::function<void(std::uint64_t, std::uint64_t)>& fn) {
    const std::uint64_t total = end > begin ? end - begin : 0U;
    if (total == 0U) return;
    const unsigned int n = qwen38_hw_threads();
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
Result<std::vector<float>> linear_f(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t out_dim,
    std::size_t in_dim) {
    if (input.size() != in_dim || weight.size() != out_dim * in_dim) {
        return make_error(ErrorCode::invalid_argument, "linear size mismatch", "qwen38_linear");
    }
    std::vector<float> out(out_dim, 0.0F);
    // Row-parallel above a work threshold; every output row keeps its own
    // sequential double accumulation, so results are bit-identical to the
    // single-threaded path.
    if (out_dim * in_dim >= 4U * 1024U * 1024U) {
        qwen38_parallel_for(0U, out_dim, 64U, [&](std::uint64_t lo, std::uint64_t hi) {
            for (std::uint64_t o = lo; o < hi; ++o) {
                double acc = 0.0;
                const std::size_t row = o * in_dim;
                for (std::size_t i = 0; i < in_dim; ++i) {
                    acc += static_cast<double>(weight[row + i]) * static_cast<double>(input[i]);
                }
                out[o] = static_cast<float>(acc);
            }
        });
        return out;
    }
    for (std::size_t o = 0; o < out_dim; ++o) {
        double acc = 0.0;
        const std::size_t row = o * in_dim;
        for (std::size_t i = 0; i < in_dim; ++i) {
            acc += static_cast<double>(weight[row + i]) * static_cast<double>(input[i]);
        }
        out[o] = static_cast<float>(acc);
    }
    return out;
}

std::int64_t wrap_mul_i64(std::int64_t a, std::int64_t b) noexcept {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
}

std::int64_t py_remainder(std::int64_t value, std::int64_t modulus) noexcept {
    std::int64_t rem = value % modulus;
    if (rem < 0) {
        rem += modulus;
    }
    return rem;
}

}  // namespace

Result<std::vector<float>> qwen38_rms_norm_affine(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon,
    std::size_t group_size) {
    if (input.size() != weight.size() || input.empty()) {
        return make_error(ErrorCode::invalid_argument,
            "rms_norm_affine input and weight must have the same non-empty size",
            "input=" + std::to_string(input.size()) + " weight=" + std::to_string(weight.size()));
    }
    const std::size_t gsz = group_size == 0 ? input.size() : group_size;
    if (gsz == 0 || (input.size() % gsz) != 0) {
        return make_error(ErrorCode::invalid_argument,
            "rms_norm_affine group_size must divide input size", "rms_norm_affine");
    }
    std::vector<float> out(input.size(), 0.0F);
    for (std::size_t start = 0; start < input.size(); start += gsz) {
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < gsz; ++i) {
            const double v = static_cast<double>(input[start + i]);
            sum_sq += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum_sq / static_cast<double>(gsz) + static_cast<double>(epsilon));
        for (std::size_t i = 0; i < gsz; ++i) {
            const double affine = 1.0 + static_cast<double>(weight[start + i]);
            out[start + i] = static_cast<float>(static_cast<double>(input[start + i]) * inv * affine);
        }
    }
    return out;
}

Result<Qwen38GatedResidualResult> qwen38_gated_residual(
    std::span<const float> hyper_input,
    const Qwen38GatedResidualView& weights,
    std::uint32_t hc_count,
    std::uint32_t hidden_size,
    std::uint32_t hc_lowrank,
    float epsilon) {
    if (hc_count <= 1U || hidden_size == 0U || hc_lowrank == 0U) {
        return make_error(ErrorCode::invalid_argument, "gated residual dims must be positive", "gated_residual");
    }
    const std::size_t hc_hidden = static_cast<std::size_t>(hc_count) * static_cast<std::size_t>(hidden_size);
    const std::size_t rank = static_cast<std::size_t>(hc_lowrank);
    if (hyper_input.size() != hc_hidden || weights.hc_norm.size() != hc_hidden
        || weights.mix_down.size() != rank * hc_hidden || weights.mix_up.size() != hc_hidden * rank) {
        return make_error(ErrorCode::invalid_argument, "gated residual weight shape mismatch", "gated_residual");
    }
    const bool use_inject = !weights.block_inject.empty();
    if (use_inject && weights.block_inject.size() != static_cast<std::size_t>(hc_count) * hc_hidden) {
        return make_error(ErrorCode::invalid_argument, "block_inject shape mismatch", "gated_residual");
    }

    auto normed = qwen38_rms_norm_affine(hyper_input, weights.hc_norm, epsilon, hidden_size);
    if (!normed) {
        return normed.error();
    }
    auto bottleneck = linear_f(normed.value(), weights.mix_down, rank, hc_hidden);
    if (!bottleneck) {
        return bottleneck.error();
    }
    const float inv_hc = 1.0F / static_cast<float>(hc_count);
    for (float& v : bottleneck.value()) {
        v = silu_f(v * inv_hc);
    }
    auto mix = linear_f(bottleneck.value(), weights.mix_up, hc_hidden, rank);
    if (!mix) {
        return mix.error();
    }
    for (float& v : mix.value()) {
        v = sigmoid_f(v);
    }

    Qwen38GatedResidualResult out;
    out.residual.assign(hyper_input.begin(), hyper_input.end());
    out.mixed_input.assign(hidden_size, 0.0F);
    for (std::uint32_t b = 0; b < hc_count; ++b) {
        for (std::uint32_t i = 0; i < hidden_size; ++i) {
            const std::size_t idx = static_cast<std::size_t>(b) * hidden_size + i;
            out.mixed_input[i] += mix.value()[idx] * normed.value()[idx];
        }
    }
    for (float& v : out.mixed_input) {
        v *= inv_hc;
    }
    if (use_inject) {
        auto logits = linear_f(normed.value(), weights.block_inject, hc_count, hc_hidden);
        if (!logits) {
            return logits.error();
        }
        out.injection_weights.resize(hc_count);
        for (std::uint32_t b = 0; b < hc_count; ++b) {
            out.injection_weights[b] = 2.0F * sigmoid_f(logits.value()[b] * inv_hc);
        }
    }
    return out;
}

Result<std::vector<float>> qwen38_gated_residual_combine(
    std::span<const float> residual,
    std::span<const float> sublayer_out,
    std::span<const float> injection_weights,
    std::uint32_t hc_count,
    std::uint32_t hidden_size) {
    const std::size_t hc_hidden = static_cast<std::size_t>(hc_count) * static_cast<std::size_t>(hidden_size);
    if (hc_count == 0U || hidden_size == 0U || residual.size() != hc_hidden
        || sublayer_out.size() != hidden_size || injection_weights.size() != hc_count) {
        return make_error(ErrorCode::invalid_argument, "gated residual combine size mismatch", "gated_residual_combine");
    }
    std::vector<float> out(residual.begin(), residual.end());
    for (std::uint32_t b = 0; b < hc_count; ++b) {
        for (std::uint32_t i = 0; i < hidden_size; ++i) {
            out[static_cast<std::size_t>(b) * hidden_size + i] += sublayer_out[i] * injection_weights[b];
        }
    }
    return out;
}

Result<std::vector<float>> qwen38_delta_net_decode(
    std::span<const float> hidden,
    const Qwen38DeltaNetView& weights,
    Qwen38DeltaNetState& state) {
    const std::size_t hidden_size = weights.hidden_size;
    const std::size_t k_heads = weights.num_k_heads;
    const std::size_t v_heads = weights.num_v_heads;
    const std::size_t dk = weights.key_head_dim;
    const std::size_t dv = weights.value_head_dim;
    const std::size_t kernel = weights.conv_kernel;
    if (hidden_size == 0 || k_heads == 0 || v_heads == 0 || dk == 0 || dv == 0 || kernel < 2
        || (v_heads % k_heads) != 0 || hidden.size() != hidden_size) {
        return make_error(ErrorCode::invalid_argument, "delta-net dims invalid", "delta_net");
    }
    const std::size_t key_dim = dk * k_heads;
    const std::size_t value_dim = dv * v_heads;
    const std::size_t conv_dim = key_dim * 2U + value_dim;
    const std::size_t conv_hist = kernel - 1U;
    if (weights.in_proj_qkv.size() != conv_dim * hidden_size
        || weights.in_proj_z.size() != value_dim * hidden_size
        || weights.in_proj_a.size() != v_heads * hidden_size
        || weights.in_proj_b.size() != v_heads * hidden_size
        || weights.conv1d.size() != conv_dim * kernel
        || weights.a_log.size() != v_heads || weights.dt_bias.size() != v_heads
        || weights.norm.size() != dv || weights.out_proj.size() != hidden_size * value_dim) {
        return make_error(ErrorCode::invalid_argument, "delta-net weight shape mismatch", "delta_net");
    }
    if (state.conv.size() != conv_dim * conv_hist || state.ssm.size() != v_heads * dk * dv) {
        return make_error(ErrorCode::invalid_argument, "delta-net state shape mismatch", "delta_net");
    }

    auto qkv = linear_f(hidden, weights.in_proj_qkv, conv_dim, hidden_size);
    if (!qkv) {
        return qkv.error();
    }
    auto z = linear_f(hidden, weights.in_proj_z, value_dim, hidden_size);
    if (!z) {
        return z.error();
    }
    auto a = linear_f(hidden, weights.in_proj_a, v_heads, hidden_size);
    if (!a) {
        return a.error();
    }
    auto b = linear_f(hidden, weights.in_proj_b, v_heads, hidden_size);
    if (!b) {
        return b.error();
    }

    std::vector<float> conv_out(conv_dim, 0.0F);
    std::vector<float> next_conv(conv_dim * conv_hist, 0.0F);
    for (std::size_t c = 0; c < conv_dim; ++c) {
        std::vector<float> hist(kernel, 0.0F);
        for (std::size_t k = 0; k < conv_hist; ++k) {
            hist[k] = state.conv[c * conv_hist + k];
        }
        hist[conv_hist] = qkv.value()[c];
        double acc = 0.0;
        for (std::size_t k = 0; k < kernel; ++k) {
            acc += static_cast<double>(weights.conv1d[c * kernel + k]) * static_cast<double>(hist[k]);
        }
        conv_out[c] = silu_f(static_cast<float>(acc));
        for (std::size_t k = 0; k < conv_hist; ++k) {
            next_conv[c * conv_hist + k] = hist[k + 1U];
        }
    }
    state.conv = std::move(next_conv);

    const float* q_ptr = conv_out.data();
    const float* k_ptr = conv_out.data() + key_dim;
    const float* v_ptr = conv_out.data() + 2U * key_dim;
    const std::size_t repeat = v_heads / k_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dk));
    std::vector<float> core(value_dim, 0.0F);

    for (std::size_t vh = 0; vh < v_heads; ++vh) {
        const std::size_t kh = weights.tile_k_heads ? (vh % k_heads) : (vh / repeat);
        std::vector<float> qh(dk, 0.0F);
        std::vector<float> kk(dk, 0.0F);
        double q_sum = 0.0;
        double k_sum = 0.0;
        for (std::size_t i = 0; i < dk; ++i) {
            qh[i] = q_ptr[kh * dk + i];
            kk[i] = k_ptr[kh * dk + i];
            q_sum += static_cast<double>(qh[i]) * static_cast<double>(qh[i]);
            k_sum += static_cast<double>(kk[i]) * static_cast<double>(kk[i]);
        }
        const double q_inv = 1.0 / std::sqrt(q_sum + 1.0e-6);
        const double k_inv = 1.0 / std::sqrt(k_sum + 1.0e-6);
        for (std::size_t i = 0; i < dk; ++i) {
            qh[i] = static_cast<float>(static_cast<double>(qh[i]) * q_inv * scale);
            kk[i] = static_cast<float>(static_cast<double>(kk[i]) * k_inv);
        }

        const float beta = sigmoid_f(b.value()[vh]);
        const float g = -std::exp(weights.a_log[vh]) * softplus_f(a.value()[vh] + weights.dt_bias[vh]);
        const float decay = std::exp(g);
        float* s = state.ssm.data() + vh * dk * dv;
        for (std::size_t i = 0; i < dk * dv; ++i) {
            s[i] *= decay;
        }
        std::vector<float> kv_mem(dv, 0.0F);
        for (std::size_t i = 0; i < dk; ++i) {
            for (std::size_t j = 0; j < dv; ++j) {
                kv_mem[j] += s[i * dv + j] * kk[i];
            }
        }
        std::vector<float> delta(dv, 0.0F);
        for (std::size_t j = 0; j < dv; ++j) {
            delta[j] = (v_ptr[vh * dv + j] - kv_mem[j]) * beta;
        }
        for (std::size_t i = 0; i < dk; ++i) {
            for (std::size_t j = 0; j < dv; ++j) {
                s[i * dv + j] += kk[i] * delta[j];
            }
        }
        std::vector<float> o(dv, 0.0F);
        double o_sum = 0.0;
        for (std::size_t i = 0; i < dk; ++i) {
            for (std::size_t j = 0; j < dv; ++j) {
                o[j] += s[i * dv + j] * qh[i];
            }
        }
        for (std::size_t j = 0; j < dv; ++j) {
            o_sum += static_cast<double>(o[j]) * static_cast<double>(o[j]);
        }
        const double o_inv = 1.0 / std::sqrt(o_sum / static_cast<double>(dv) + static_cast<double>(weights.rms_eps));
        for (std::size_t j = 0; j < dv; ++j) {
            // Qwen4ExpTextRMSNormGated output gate: the converted checkpoint
            // uses a sigmoid (llama.cpp: "the one numerical difference from
            // Qwen3.5's GDN: sigmoid output gate, not silu"), which the HF
            // modeling main does not carry. Validated against llama.cpp.
            const float gated = sigmoid_f(z.value()[vh * dv + j]);
            core[vh * dv + j] = static_cast<float>(
                static_cast<double>(weights.norm[j]) * static_cast<double>(o[j]) * o_inv * static_cast<double>(gated));
        }
    }
    return linear_f(core, weights.out_proj, hidden_size, value_dim);
}

void qwen38_apply_partial_rope(
    std::span<float> head,
    std::uint64_t position,
    std::uint64_t rope_dim,
    float theta) {
    if (rope_dim == 0U || rope_dim > head.size() || rope_dim % 2U != 0U || theta <= 0.0F) {
        return;
    }
    const std::size_t half = static_cast<std::size_t>(rope_dim / 2U);
    // emb = cat(freqs, freqs) with rotate_half pairing (i, i + half).
    for (std::size_t i = 0; i < half; ++i) {
        const double freq = std::pow(static_cast<double>(theta),
            -2.0 * static_cast<double>(i) / static_cast<double>(rope_dim));
        const double angle = freq * static_cast<double>(position);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const float x1 = head[i];
        const float x2 = head[half + i];
        head[i] = static_cast<float>(static_cast<double>(x1) * c
            - static_cast<double>(x2) * s);
        head[half + i] = static_cast<float>(static_cast<double>(x2) * c
            + static_cast<double>(x1) * s);
    }
}

Result<Qwen38QsaSelectResult> qwen38_qsa_select(const Qwen38QsaSelectView& view) {
    const std::size_t n_q = view.n_query_heads;
    const std::size_t d = view.head_dim;
    const std::size_t t = view.token_count;
    const std::size_t compress = view.compress_ratio;
    const std::size_t budget = view.budget;
    if (n_q == 0 || d == 0 || t == 0 || compress == 0 || budget == 0 || (budget % compress) != 0
        || view.query_heads.size() != n_q * d || view.raw_keys.size() != t * d || view.k_norm.size() != d) {
        return make_error(ErrorCode::invalid_argument, "qsa select size mismatch", "qsa_select");
    }
    const std::size_t n_blocks = t / compress;
    const std::size_t block_topk = budget / compress;
    const std::size_t pad = budget + compress - 1U;
    Qwen38QsaSelectResult out;
    out.selected.assign(pad, -1);
    out.block_scores.assign(n_blocks, 0.0F);

    std::vector<std::pair<float, std::size_t>> ranked;
    ranked.reserve(n_blocks);
    std::vector<float> q_rope_storage;
    std::span<const float> q_heads = view.query_heads;
    if (view.rope_theta > 0.0F && view.rope_dim > 0U && view.rope_dim <= d) {
        q_rope_storage.assign(q_heads.begin(), q_heads.end());
        for (std::size_t h = 0; h < n_q; ++h) {
            qwen38_apply_partial_rope(
                std::span<float>(q_rope_storage.data() + h * d, d),
                view.query_position, view.rope_dim, view.rope_theta);
        }
        q_heads = q_rope_storage;
    }
    const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(d));
    for (std::size_t b = 0; b < n_blocks; ++b) {
        std::vector<float> pooled(d, 0.0F);
        for (std::size_t k = 0; k < compress; ++k) {
            const float* row = view.raw_keys.data() + (b * compress + k) * d;
            for (std::size_t i = 0; i < d; ++i) {
                pooled[i] += row[i];
            }
        }
        for (float& v : pooled) {
            v /= static_cast<float>(compress);
        }
        auto kn = qwen38_rms_norm_affine(pooled, view.k_norm, view.rms_eps, 0);
        if (!kn) {
            return kn.error();
        }
        // Official indexer applies partial RoPE to the pooled block keys at
        // the block's starting position, and to the query at the current
        // position (done once below for the query heads).
        if (view.rope_theta > 0.0F && view.rope_dim > 0U) {
            auto pooled_head = kn.value();
            qwen38_apply_partial_rope(pooled_head,
                static_cast<std::uint64_t>(b) * view.compress_ratio,
                view.rope_dim, view.rope_theta);
            double score = 0.0;
            for (std::size_t h = 0; h < n_q; ++h) {
                double dot = 0.0;
                for (std::size_t i = 0; i < d; ++i) {
                    dot += static_cast<double>(q_heads[h * d + i])
                        * static_cast<double>(pooled_head[i]);
                }
                score += (dot > 0.0) ? dot : 0.0;
            }
            const float s = static_cast<float>(score * inv_sqrt);
            out.block_scores[b] = s;
            ranked.push_back({s, b});
            continue;
        }
        double score = 0.0;
        for (std::size_t h = 0; h < n_q; ++h) {
            double dot = 0.0;
            for (std::size_t i = 0; i < d; ++i) {
                dot += static_cast<double>(q_heads[h * d + i]) * static_cast<double>(kn.value()[i]);
            }
            score += (dot > 0.0) ? dot : 0.0;
        }
        const float s = static_cast<float>(score * inv_sqrt);
        out.block_scores[b] = s;
        ranked.push_back({s, b});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    });

    std::size_t nsel = 0;
    const std::size_t take = (std::min)(block_topk, n_blocks);
    for (std::size_t i = 0; i < take; ++i) {
        const std::size_t b = ranked[i].second;
        for (std::size_t k = 0; k < compress; ++k) {
            out.selected[nsel++] = static_cast<std::int32_t>(b * compress + k);
        }
    }
    for (std::size_t i = n_blocks * compress; i < t; ++i) {
        out.selected[nsel++] = static_cast<std::int32_t>(i);
    }
    return out;
}

std::uint64_t qwen38_splitmix64(std::uint64_t value) noexcept {
    value = (value + kSplitmixGamma);
    value = (value ^ (value >> 30U)) * kSplitmixM1;
    value = (value ^ (value >> 27U)) * kSplitmixM2;
    return value ^ (value >> 31U);
}

Result<std::uint32_t> qwen38_nth_prime_after(std::uint32_t start, std::uint32_t count) {
    if (count == 0U) {
        return make_error(ErrorCode::invalid_argument, "nth_prime_after count must be > 0", "ngram");
    }
    std::uint32_t prime = start;
    for (std::uint32_t n = 0; n < count; ++n) {
        if (prime == std::numeric_limits<std::uint32_t>::max()) {
            return make_error(ErrorCode::invalid_argument, "nth_prime_after overflow", "ngram");
        }
        ++prime;
        while (!is_prime_u32(prime)) {
            if (prime == std::numeric_limits<std::uint32_t>::max()) {
                return make_error(ErrorCode::invalid_argument, "nth_prime_after overflow", "ngram");
            }
            ++prime;
        }
    }
    return prime;
}

Result<std::vector<std::int64_t>> qwen38_ngram_multipliers(
    std::uint32_t unigram_vocab,
    std::uint32_t ngram_size,
    std::uint32_t ple_layer_index,
    std::int32_t seed) {
    if (unigram_vocab == 0U || ngram_size < 2U) {
        return make_error(ErrorCode::invalid_argument, "ngram multiplier args invalid", "ngram");
    }
    const std::uint64_t max_long = 0x7FFFFFFFFFFFFFFFULL;
    const std::uint64_t multiplier_max = max_long / static_cast<std::uint64_t>(unigram_vocab);
    const std::uint64_t half_bound = (std::max)(std::uint64_t{1}, multiplier_max / 2U);
    const std::uint64_t base_seed =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(seed) + kNgramPrime1 * static_cast<std::int64_t>(ple_layer_index));
    std::vector<std::int64_t> out;
    out.reserve(ngram_size);
    for (std::uint32_t index = 0; index < ngram_size; ++index) {
        const std::uint64_t value = base_seed + kSplitmixGamma * static_cast<std::uint64_t>(index + 1U);
        const std::uint64_t mixed = qwen38_splitmix64(value) % half_bound;
        out.push_back(static_cast<std::int64_t>(2U * mixed + 1U));
    }
    return out;
}

Result<std::vector<std::int64_t>> qwen38_ngram_hash_ids(
    std::span<const std::int32_t> token_ids,
    std::span<const std::int32_t> previous_context,
    std::span<const std::int64_t> multipliers,
    std::span<const std::int64_t> head_vocab_sizes,
    std::span<const std::int64_t> head_offsets,
    std::uint32_t ngram_size,
    std::uint32_t heads_per_ngram,
    std::int32_t eos_token_id) {
    if (token_ids.empty() || ngram_size < 2U || heads_per_ngram == 0U
        || multipliers.size() != ngram_size) {
        return make_error(ErrorCode::invalid_argument, "ngram hash args invalid", "ngram");
    }
    const std::size_t ngram_heads = static_cast<std::size_t>(ngram_size - 1U) * heads_per_ngram;
    if (head_vocab_sizes.size() != ngram_heads || head_offsets.size() != ngram_heads) {
        return make_error(ErrorCode::invalid_argument, "ngram head table size mismatch", "ngram");
    }
    const std::size_t context_len = ngram_size - 1U;
    std::vector<std::int32_t> history;
    history.reserve(context_len + token_ids.size());
    if (previous_context.empty()) {
        history.insert(history.end(), context_len, eos_token_id);
    } else if (previous_context.size() == context_len) {
        history.insert(history.end(), previous_context.begin(), previous_context.end());
    } else {
        return make_error(ErrorCode::invalid_argument, "previous_context must be ngram_size-1", "ngram");
    }
    history.insert(history.end(), token_ids.begin(), token_ids.end());
    const std::size_t seq = history.size();

    auto shift_right = [&](std::size_t shift) {
        std::vector<std::int32_t> out(seq, eos_token_id);
        if (shift == 0) {
            return history;
        }
        std::vector<std::int32_t> eos_pos(seq, -1);
        for (std::size_t i = 0; i < seq; ++i) {
            if (history[i] == eos_token_id) {
                eos_pos[i] = static_cast<std::int32_t>(i);
            }
        }
        std::vector<std::int32_t> prev_inc(seq, -1);
        std::int32_t cur = -1;
        for (std::size_t i = 0; i < seq; ++i) {
            cur = (std::max)(cur, eos_pos[i]);
            prev_inc[i] = cur;
        }
        for (std::size_t i = 0; i < seq; ++i) {
            const std::int32_t prev_eos = (i == 0) ? -1 : prev_inc[i - 1];
            const std::int32_t segment_start = prev_eos + 1;
            const std::int32_t pos_in_seg = static_cast<std::int32_t>(i) - segment_start;
            const std::int32_t source = static_cast<std::int32_t>(i) - static_cast<std::int32_t>(shift);
            if (pos_in_seg >= static_cast<std::int32_t>(shift) && source >= 0) {
                out[i] = history[static_cast<std::size_t>(source)];
            }
        }
        return out;
    };

    std::vector<std::vector<std::int32_t>> shifted;
    shifted.reserve(ngram_size);
    for (std::uint32_t s = 0; s < ngram_size; ++s) {
        shifted.push_back(shift_right(s));
    }

    std::vector<std::int64_t> ids;
    ids.reserve(token_ids.size() * ngram_heads);
    const std::size_t first = context_len;
    for (std::size_t t = first; t < seq; ++t) {
        for (std::uint32_t ngram = 2; ngram <= ngram_size; ++ngram) {
            const std::size_t start = static_cast<std::size_t>(ngram - 2U) * heads_per_ngram;
            for (std::uint32_t head = 0; head < heads_per_ngram; ++head) {
                std::uint64_t mixed = static_cast<std::uint64_t>(
                    wrap_mul_i64(shifted[0][t], multipliers[0]));
                for (std::uint32_t p = 1; p < ngram; ++p) {
                    mixed ^= static_cast<std::uint64_t>(wrap_mul_i64(shifted[p][t], multipliers[p]));
                }
                const std::int64_t mixed_s = static_cast<std::int64_t>(mixed);
                const std::size_t head_idx = start + head;
                if (head_vocab_sizes[head_idx] <= 0) {
                    return make_error(ErrorCode::invalid_argument, "ngram vocab size must be > 0", "ngram");
                }
                const std::int64_t rem = py_remainder(mixed_s, head_vocab_sizes[head_idx]);
                ids.push_back(rem + head_offsets[head_idx]);
            }
        }
    }
    return ids;
}

Result<std::vector<float>> qwen38_ngram_lookup(
    std::span<const std::int64_t> ngram_ids,
    std::span<const float> table,
    std::uint32_t table_rows,
    std::uint32_t head_dim) {
    if (head_dim == 0U || table_rows == 0U || table.size() != static_cast<std::size_t>(table_rows) * head_dim) {
        return make_error(ErrorCode::invalid_argument, "ngram table shape mismatch", "ngram_lookup");
    }
    std::vector<float> out;
    out.reserve(ngram_ids.size() * head_dim);
    for (std::int64_t id : ngram_ids) {
        if (id < 0 || static_cast<std::uint64_t>(id) >= table_rows) {
            return make_error(ErrorCode::invalid_argument, "ngram id out of range", "ngram_lookup");
        }
        const std::size_t row = static_cast<std::size_t>(id) * head_dim;
        out.insert(out.end(), table.begin() + static_cast<std::ptrdiff_t>(row),
            table.begin() + static_cast<std::ptrdiff_t>(row + head_dim));
    }
    return out;
}


Result<std::vector<float>> qwen38_linear(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t out_dim,
    std::size_t in_dim) {
    return linear_f(input, weight, out_dim, in_dim);
}

Result<std::vector<float>> qwen38_shared_mlp(
    std::span<const float> hidden,
    std::span<const float> gate_proj,
    std::span<const float> up_proj,
    std::span<const float> down_proj,
    std::span<const float> shared_gate,
    std::uint32_t hidden_size,
    std::uint32_t intermediate) {
    const std::size_t h = hidden_size;
    const std::size_t mid = intermediate;
    if (h == 0 || mid == 0 || hidden.size() != h) {
        return make_error(ErrorCode::invalid_argument, "shared mlp dims invalid", "shared_mlp");
    }
    auto gate = linear_f(hidden, gate_proj, mid, h);
    if (!gate) {
        return gate.error();
    }
    auto up = linear_f(hidden, up_proj, mid, h);
    if (!up) {
        return up.error();
    }
    for (std::size_t i = 0; i < mid; ++i) {
        gate.value()[i] = silu_f(gate.value()[i]) * up.value()[i];
    }
    auto down = linear_f(gate.value(), down_proj, h, mid);
    if (!down) {
        return down.error();
    }
    auto sg = linear_f(hidden, shared_gate, 1, h);
    if (!sg) {
        return sg.error();
    }
    const float scale = sigmoid_f(sg.value()[0]);
    for (float& v : down.value()) {
        v *= scale;
    }
    return down;
}

Result<Qwen38MoeRoute> qwen38_moe_route(
    std::span<const float> hidden,
    std::span<const float> router,
    std::uint32_t num_experts,
    std::uint32_t top_k,
    bool norm_topk_prob) {
    const std::size_t e = num_experts;
    const std::size_t k = top_k;
    if (e == 0 || k == 0 || k > e || hidden.empty() || router.size() != e * hidden.size()) {
        return make_error(ErrorCode::invalid_argument, "moe route dims invalid", "moe_route");
    }
    auto logits = linear_f(hidden, router, e, hidden.size());
    if (!logits) {
        return logits.error();
    }
    double max_l = static_cast<double>(logits.value()[0]);
    for (float v : logits.value()) {
        max_l = (std::max)(max_l, static_cast<double>(v));
    }
    std::vector<double> probs(e, 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < e; ++i) {
        probs[i] = std::exp(static_cast<double>(logits.value()[i]) - max_l);
        sum += probs[i];
    }
    const double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
    std::vector<std::pair<double, std::uint32_t>> ranked;
    ranked.reserve(e);
    for (std::size_t i = 0; i < e; ++i) {
        ranked.push_back({probs[i] * inv, static_cast<std::uint32_t>(i)});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });
    Qwen38MoeRoute route;
    route.logits = std::move(logits.value());
    route.indices.resize(k);
    route.weights.resize(k);
    double top_sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        route.indices[i] = ranked[i].second;
        route.weights[i] = static_cast<float>(ranked[i].first);
        top_sum += ranked[i].first;
    }
    if (norm_topk_prob && top_sum > 0.0) {
        const double scale = 1.0 / top_sum;
        for (float& w : route.weights) {
            w = static_cast<float>(static_cast<double>(w) * scale);
        }
    }
    return route;
}

Result<std::vector<float>> qwen38_expert_mlp(
    std::span<const float> hidden,
    std::span<const float> gate_proj,
    std::span<const float> up_proj,
    std::span<const float> down_proj,
    std::uint32_t hidden_size,
    std::uint32_t intermediate) {
    const std::size_t h = hidden_size;
    const std::size_t mid = intermediate;
    if (h == 0 || mid == 0 || hidden.size() != h) {
        return make_error(ErrorCode::invalid_argument, "expert mlp dims invalid", "expert_mlp");
    }
    auto gate = linear_f(hidden, gate_proj, mid, h);
    if (!gate) {
        return gate.error();
    }
    auto up = linear_f(hidden, up_proj, mid, h);
    if (!up) {
        return up.error();
    }
    for (std::size_t i = 0; i < mid; ++i) {
        gate.value()[i] = silu_f(gate.value()[i]) * up.value()[i];
    }
    return linear_f(gate.value(), down_proj, h, mid);
}

Result<const Qwen38OwnedExpert*> qwen38_expert_cache_acquire(
    Qwen38ExpertLru& cache,
    std::uint32_t key,
    const Qwen38OwnedExpert& source) {
    if (cache.capacity == 0) {
        return make_error(ErrorCode::invalid_argument, "expert cache capacity must be > 0", "expert_cache");
    }
    ++cache.telemetry.requests;
    for (std::size_t i = 0; i < cache.keys.size(); ++i) {
        if (cache.keys[i] == key) {
            ++cache.telemetry.hits;
            auto k = cache.keys[i];
            auto v = std::move(cache.values[i]);
            cache.keys.erase(cache.keys.begin() + static_cast<std::ptrdiff_t>(i));
            cache.values.erase(cache.values.begin() + static_cast<std::ptrdiff_t>(i));
            cache.keys.push_back(k);
            cache.values.push_back(std::move(v));
            return &cache.values.back();
        }
    }
    ++cache.telemetry.misses;
    cache.telemetry.logical_bytes += static_cast<std::uint64_t>(
        (source.gate_proj.size() + source.up_proj.size() + source.down_proj.size()) * sizeof(float));
    if (cache.keys.size() >= cache.capacity) {
        cache.keys.erase(cache.keys.begin());
        cache.values.erase(cache.values.begin());
        ++cache.telemetry.evictions;
    }
    cache.keys.push_back(key);
    cache.values.push_back(source);
    return &cache.values.back();
}

Result<std::vector<float>> qwen38_moe_block(
    std::span<const float> hidden,
    const Qwen38OwnedMoe& moe,
    const Qwen38OwnedSharedMlp& shared,
    Qwen38ExpertLru& cache,
    std::uint32_t layer_id,
    std::uint32_t hidden_size,
    std::uint32_t intermediate,
    std::uint32_t top_k,
    bool norm_topk_prob) {
    const auto num_experts = static_cast<std::uint32_t>(moe.experts.size());
    auto route = qwen38_moe_route(hidden, moe.router, num_experts, top_k, norm_topk_prob);
    if (!route) {
        return route.error();
    }
    std::vector<float> out(hidden_size, 0.0F);
    for (std::size_t i = 0; i < route.value().indices.size(); ++i) {
        const std::uint32_t eid = route.value().indices[i];
        if (eid >= num_experts) {
            return make_error(ErrorCode::invalid_argument, "routed expert id out of range", "moe_block");
        }
        const std::uint32_t key = (layer_id << 16) | eid;
        auto acquired = qwen38_expert_cache_acquire(cache, key, moe.experts[eid]);
        if (!acquired) {
            return acquired.error();
        }
        auto y = qwen38_expert_mlp(hidden, acquired.value()->gate_proj, acquired.value()->up_proj,
            acquired.value()->down_proj, hidden_size, intermediate);
        if (!y) {
            return y.error();
        }
        const float w = route.value().weights[i];
        for (std::size_t d = 0; d < out.size(); ++d) {
            out[d] += y.value()[d] * w;
        }
    }
    auto shared_out = qwen38_shared_mlp(hidden, shared.gate_proj, shared.up_proj, shared.down_proj,
        shared.shared_gate, hidden_size, intermediate);
    if (!shared_out) {
        return shared_out.error();
    }
    for (std::size_t d = 0; d < out.size(); ++d) {
        out[d] += shared_out.value()[d];
    }
    return out;
}

std::vector<float> qwen38_expand_stream(std::span<const float> hidden, std::uint32_t hc_count) {
    std::vector<float> out(hidden.size() * hc_count, 0.0F);
    for (std::uint32_t b = 0; b < hc_count; ++b) {
        std::copy(hidden.begin(), hidden.end(), out.begin() + static_cast<std::ptrdiff_t>(b * hidden.size()));
    }
    return out;
}

Result<std::vector<float>> qwen38_qsa_layer_token(
    std::span<const float> hidden,
    const Qwen38QsaLayerView& weights,
    Qwen38QsaCache& cache) {
    const std::size_t h = weights.hidden_size;
    const std::size_t idx_heads = weights.indexer_heads;
    const std::size_t idx_dim = weights.indexer_dim;
    const std::size_t attn_heads = weights.attn_heads;
    const std::size_t attn_kv = weights.attn_kv_heads;
    const std::size_t attn_dim = weights.attn_head_dim;
    if (h == 0 || idx_heads == 0 || idx_dim == 0 || attn_heads == 0 || attn_kv == 0 || attn_dim == 0
        || (attn_heads % attn_kv) != 0 || hidden.size() != h) {
        return make_error(ErrorCode::invalid_argument, "qsa layer dims invalid", "qsa_layer");
    }
    const std::size_t qk_out = (idx_heads + 1U) * idx_dim;
    const std::size_t q_out = attn_heads * attn_dim * 2U;
    const std::size_t kv_out = attn_kv * attn_dim;
    if (weights.index_qk.size() != qk_out * h || weights.q_layernorm.size() != idx_dim
        || weights.k_layernorm.size() != idx_dim || weights.q_proj.size() != q_out * h
        || weights.k_proj.size() != kv_out * h || weights.v_proj.size() != kv_out * h
        || weights.o_proj.size() != h * attn_heads * attn_dim
        || weights.q_norm.size() != attn_dim || weights.k_norm.size() != attn_dim) {
        return make_error(ErrorCode::invalid_argument, "qsa layer weight shape mismatch", "qsa_layer");
    }

    auto qk = linear_f(hidden, weights.index_qk, qk_out, h);
    if (!qk) {
        return qk.error();
    }
    // Absolute position of the current token within this layer's cache.
    const std::uint32_t current_position =
        static_cast<std::uint32_t>(cache.index_keys.size() / idx_dim);
    std::vector<float> q_idx(idx_heads * idx_dim, 0.0F);
    for (std::size_t head = 0; head < idx_heads; ++head) {
        auto head_span = std::span<const float>(qk.value().data() + head * idx_dim, idx_dim);
        auto normed = qwen38_rms_norm_affine(head_span, weights.q_layernorm, weights.rms_eps, 0);
        if (!normed) {
            return normed.error();
        }
        // Indexer query receives partial RoPE at the current position.
        if (weights.rope_theta > 0.0F && weights.rope_dim > 0U
            && weights.rope_dim <= idx_dim) {
            qwen38_apply_partial_rope(normed.value(), current_position,
                weights.rope_dim, weights.rope_theta);
        }
        std::copy(normed.value().begin(), normed.value().end(), q_idx.begin() + static_cast<std::ptrdiff_t>(head * idx_dim));
    }
    cache.index_keys.insert(cache.index_keys.end(),
        qk.value().begin() + static_cast<std::ptrdiff_t>(idx_heads * idx_dim), qk.value().end());
    const std::size_t tokens = cache.index_keys.size() / idx_dim;
    Qwen38QsaSelectView select_view{
        q_idx, cache.index_keys, weights.k_layernorm,
        static_cast<std::uint32_t>(idx_heads), static_cast<std::uint32_t>(idx_dim),
        static_cast<std::uint32_t>(tokens), weights.budget, weights.compress_ratio, weights.rms_eps,
        weights.rope_theta, weights.rope_dim, current_position};
    auto selected = qwen38_qsa_select(select_view);
    if (!selected) {
        return selected.error();
    }

    auto qg = linear_f(hidden, weights.q_proj, q_out, h);
    if (!qg) {
        return qg.error();
    }
    std::vector<float> query(attn_heads * attn_dim, 0.0F);
    std::vector<float> gate(attn_heads * attn_dim, 0.0F);
    for (std::size_t head = 0; head < attn_heads; ++head) {
        auto q_head = std::span<const float>(qg.value().data() + head * attn_dim * 2U, attn_dim);
        auto g_head = std::span<const float>(qg.value().data() + head * attn_dim * 2U + attn_dim, attn_dim);
        auto qn = qwen38_rms_norm_affine(q_head, weights.q_norm, weights.rms_eps, 0);
        if (!qn) {
            return qn.error();
        }
        // Attention query receives partial RoPE at the current position
        // (after q_norm, matching the official attention).
        if (weights.rope_theta > 0.0F && weights.rope_dim > 0U
            && weights.rope_dim <= attn_dim) {
            qwen38_apply_partial_rope(qn.value(), current_position,
                weights.rope_dim, weights.rope_theta);
        }
        std::copy(qn.value().begin(), qn.value().end(), query.begin() + static_cast<std::ptrdiff_t>(head * attn_dim));
        std::copy(g_head.begin(), g_head.end(), gate.begin() + static_cast<std::ptrdiff_t>(head * attn_dim));
    }
    auto k = linear_f(hidden, weights.k_proj, kv_out, h);
    if (!k) {
        return k.error();
    }
    for (std::size_t head = 0; head < attn_kv; ++head) {
        auto k_head = std::span<const float>(k.value().data() + head * attn_dim, attn_dim);
        auto kn = qwen38_rms_norm_affine(k_head, weights.k_norm, weights.rms_eps, 0);
        if (!kn) {
            return kn.error();
        }
        // Attention key receives partial RoPE at the current position before
        // being cached (the official cache stores rotated keys).
        if (weights.rope_theta > 0.0F && weights.rope_dim > 0U
            && weights.rope_dim <= attn_dim) {
            qwen38_apply_partial_rope(kn.value(), current_position,
                weights.rope_dim, weights.rope_theta);
        }
        std::copy(kn.value().begin(), kn.value().end(), k.value().begin() + static_cast<std::ptrdiff_t>(head * attn_dim));
    }
    auto v = linear_f(hidden, weights.v_proj, kv_out, h);
    if (!v) {
        return v.error();
    }
    cache.attn_k.insert(cache.attn_k.end(), k.value().begin(), k.value().end());
    cache.attn_v.insert(cache.attn_v.end(), v.value().begin(), v.value().end());

    std::vector<std::size_t> keep;
    keep.reserve(tokens);
    for (std::int32_t idx : selected.value().selected) {
        if (idx >= 0 && static_cast<std::size_t>(idx) < tokens) {
            keep.push_back(static_cast<std::size_t>(idx));
        }
    }
    if (keep.empty()) {
        for (std::size_t i = 0; i < tokens; ++i) {
            keep.push_back(i);
        }
    }

    const std::size_t rep = attn_heads / attn_kv;
    const double scale = 1.0 / std::sqrt(static_cast<double>(attn_dim));
    std::vector<float> attn_out(attn_heads * attn_dim, 0.0F);
    for (std::size_t head = 0; head < attn_heads; ++head) {
        const std::size_t kvh = head / rep;
        std::vector<double> scores(keep.size(), 0.0);
        double max_s = -1.0e30;
        for (std::size_t n = 0; n < keep.size(); ++n) {
            const float* key = cache.attn_k.data() + keep[n] * kv_out + kvh * attn_dim;
            double dot = 0.0;
            for (std::size_t d = 0; d < attn_dim; ++d) {
                dot += static_cast<double>(query[head * attn_dim + d]) * static_cast<double>(key[d]);
            }
            scores[n] = dot * scale;
            max_s = (std::max)(max_s, scores[n]);
        }
        double sum = 0.0;
        for (double& s : scores) {
            s = std::exp(s - max_s);
            sum += s;
        }
        const double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
        for (std::size_t n = 0; n < keep.size(); ++n) {
            const float* val = cache.attn_v.data() + keep[n] * kv_out + kvh * attn_dim;
            const double a = scores[n] * inv;
            for (std::size_t d = 0; d < attn_dim; ++d) {
                attn_out[head * attn_dim + d] += static_cast<float>(a * static_cast<double>(val[d]));
            }
        }
    }
    for (std::size_t i = 0; i < attn_out.size(); ++i) {
        attn_out[i] *= sigmoid_f(gate[i]);
    }
    return linear_f(attn_out, weights.o_proj, h, attn_heads * attn_dim);
}

Result<std::vector<float>> qwen38_qsa_sparse_attention(
    std::span<const float> query,
    std::span<const float> k_cache,
    std::span<const float> v_cache,
    std::span<const std::int32_t> selected,
    std::span<const float> gate,
    std::uint32_t attn_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    const std::size_t heads = attn_heads;
    const std::size_t kv = kv_heads;
    const std::size_t dim = head_dim;
    if (heads == 0 || kv == 0 || dim == 0 || (heads % kv) != 0 || query.size() != heads * dim
        || gate.size() != heads * dim) {
        return make_error(ErrorCode::invalid_argument, "qsa sparse dims invalid", "qsa_sparse");
    }
    const std::size_t tokens = (kv * dim == 0) ? 0 : k_cache.size() / (kv * dim);
    if (k_cache.size() != tokens * kv * dim || v_cache.size() != tokens * kv * dim) {
        return make_error(ErrorCode::invalid_argument, "qsa cache size mismatch", "qsa_sparse");
    }
    std::vector<std::size_t> keep;
    keep.reserve(selected.size());
    for (std::int32_t idx : selected) {
        if (idx >= 0 && static_cast<std::size_t>(idx) < tokens) {
            keep.push_back(static_cast<std::size_t>(idx));
        }
    }
    if (keep.empty() && tokens > 0) {
        for (std::size_t i = 0; i < tokens; ++i) {
            keep.push_back(i);
        }
    }
    const std::size_t rep = heads / kv;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<float> out(heads * dim, 0.0F);
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t kvh = head / rep;
        std::vector<double> scores(keep.size(), 0.0);
        double max_s = -1.0e30;
        for (std::size_t n = 0; n < keep.size(); ++n) {
            const float* key = k_cache.data() + keep[n] * kv * dim + kvh * dim;
            double dot = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                dot += static_cast<double>(query[head * dim + d]) * static_cast<double>(key[d]);
            }
            scores[n] = dot * scale;
            max_s = (std::max)(max_s, scores[n]);
        }
        double sum = 0.0;
        for (double& s : scores) {
            s = std::exp(s - max_s);
            sum += s;
        }
        const double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
        for (std::size_t n = 0; n < keep.size(); ++n) {
            const float* val = v_cache.data() + keep[n] * kv * dim + kvh * dim;
            const double a = scores[n] * inv;
            for (std::size_t d = 0; d < dim; ++d) {
                out[head * dim + d] += static_cast<float>(a * static_cast<double>(val[d]));
            }
        }
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] *= sigmoid_f(gate[i]);
    }
    return out;
}

namespace {

Qwen38GatedResidualView residual_view(const Qwen38OwnedResidual& w) {
    return {w.hc_norm, w.mix_down, w.mix_up, w.block_inject};
}

Qwen38DeltaNetView delta_view(const Qwen38OwnedDeltaNet& w, const Qwen38ReducedDims& d) {
    return {w.in_proj_qkv, w.in_proj_z, w.in_proj_a, w.in_proj_b, w.conv1d, w.a_log, w.dt_bias, w.norm, w.out_proj,
        d.hidden_size, d.dn_k_heads, d.dn_v_heads, d.dn_key_dim, d.dn_value_dim, d.dn_kernel, d.rms_eps};
}

Qwen38QsaLayerView qsa_view(const Qwen38OwnedQsa& w, const Qwen38ReducedDims& d) {
    return {w.index_qk, w.q_layernorm, w.k_layernorm, w.q_proj, w.k_proj, w.v_proj, w.o_proj, w.q_norm, w.k_norm,
        d.hidden_size, d.indexer_heads, d.indexer_dim, d.budget, d.compress_ratio,
        d.attn_heads, d.attn_kv_heads, d.attn_head_dim, d.rms_eps};
}

Result<std::vector<float>> gather_embed(const Qwen38ReducedModel& model, std::int32_t token) {
    const std::size_t h = model.dims.hidden_size;
    const std::size_t v = model.dims.vocab_size;
    if (token < 0 || static_cast<std::uint32_t>(token) >= v || model.embed.size() != v * h) {
        return make_error(ErrorCode::invalid_argument, "embed token out of range", "reduced");
    }
    const std::size_t row = static_cast<std::size_t>(token) * h;
    return std::vector<float>(model.embed.begin() + static_cast<std::ptrdiff_t>(row),
        model.embed.begin() + static_cast<std::ptrdiff_t>(row + h));
}

Result<std::vector<float>> project_logits(const Qwen38ReducedModel& model, std::span<const float> hidden) {
    return linear_f(hidden, model.lm_head, model.dims.vocab_size, model.dims.hidden_size);
}

Result<std::vector<float>> mix_to_hidden(
    std::span<const float> stream,
    const Qwen38OwnedResidual& mixer,
    const Qwen38ReducedDims& dims) {
    auto mixed = qwen38_gated_residual(
        stream, residual_view(mixer), dims.hc_count, dims.hidden_size, dims.hc_lowrank, dims.rms_eps);
    if (!mixed) {
        return mixed.error();
    }
    return mixed.value().mixed_input;
}

Result<std::vector<float>> reduced_one_token(
    const Qwen38ReducedModel& model,
    std::int32_t token,
    Qwen38ReducedState& state,
    bool run_mtp,
    std::vector<float>* mtp_logits) {
    auto embed = gather_embed(model, token);
    if (!embed) {
        return embed.error();
    }
    std::vector<float> stream = qwen38_expand_stream(embed.value(), model.dims.hc_count);

    auto gr0 = qwen38_gated_residual(stream, residual_view(model.layer0_attn_hc),
        model.dims.hc_count, model.dims.hidden_size, model.dims.hc_lowrank, model.dims.rms_eps);
    if (!gr0) {
        return gr0.error();
    }
    auto dn = qwen38_delta_net_decode(gr0.value().mixed_input, delta_view(model.layer0_delta, model.dims), state.delta);
    if (!dn) {
        return dn.error();
    }
    auto after0 = qwen38_gated_residual_combine(
        gr0.value().residual, dn.value(), gr0.value().injection_weights, model.dims.hc_count, model.dims.hidden_size);
    if (!after0) {
        return after0.error();
    }
    auto gr0m = qwen38_gated_residual(after0.value(), residual_view(model.layer0_mlp_hc),
        model.dims.hc_count, model.dims.hidden_size, model.dims.hc_lowrank, model.dims.rms_eps);
    if (!gr0m) {
        return gr0m.error();
    }
    auto mlp0 = qwen38_moe_block(gr0m.value().mixed_input, model.layer0_moe, model.layer0_mlp, state.experts,
        0, model.dims.hidden_size, model.dims.moe_intermediate, model.dims.experts_per_token,
        model.dims.norm_topk_prob);
    if (!mlp0) {
        return mlp0.error();
    }
    auto layer0 = qwen38_gated_residual_combine(
        gr0m.value().residual, mlp0.value(), gr0m.value().injection_weights, model.dims.hc_count, model.dims.hidden_size);
    if (!layer0) {
        return layer0.error();
    }

    auto gr1 = qwen38_gated_residual(layer0.value(), residual_view(model.layer1_attn_hc),
        model.dims.hc_count, model.dims.hidden_size, model.dims.hc_lowrank, model.dims.rms_eps);
    if (!gr1) {
        return gr1.error();
    }
    auto qsa1 = qwen38_qsa_layer_token(gr1.value().mixed_input, qsa_view(model.layer1_qsa, model.dims), state.layer1);
    if (!qsa1) {
        return qsa1.error();
    }
    auto after1 = qwen38_gated_residual_combine(
        gr1.value().residual, qsa1.value(), gr1.value().injection_weights, model.dims.hc_count, model.dims.hidden_size);
    if (!after1) {
        return after1.error();
    }
    auto gr1m = qwen38_gated_residual(after1.value(), residual_view(model.layer1_mlp_hc),
        model.dims.hc_count, model.dims.hidden_size, model.dims.hc_lowrank, model.dims.rms_eps);
    if (!gr1m) {
        return gr1m.error();
    }
    auto mlp1 = qwen38_moe_block(gr1m.value().mixed_input, model.layer1_moe, model.layer1_mlp, state.experts,
        1, model.dims.hidden_size, model.dims.moe_intermediate, model.dims.experts_per_token,
        model.dims.norm_topk_prob);
    if (!mlp1) {
        return mlp1.error();
    }
    auto layer1 = qwen38_gated_residual_combine(
        gr1m.value().residual, mlp1.value(), gr1m.value().injection_weights, model.dims.hc_count, model.dims.hidden_size);
    if (!layer1) {
        return layer1.error();
    }

    auto hidden = mix_to_hidden(layer1.value(), model.mixer, model.dims);
    if (!hidden) {
        return hidden.error();
    }

    if (run_mtp && mtp_logits != nullptr) {
        auto hn = qwen38_rms_norm_affine(hidden.value(), model.mtp_pre_hidden, model.dims.rms_eps, 0);
        if (!hn) {
            return hn.error();
        }
        auto en = qwen38_rms_norm_affine(embed.value(), model.mtp_pre_embed, model.dims.rms_eps, 0);
        if (!en) {
            return en.error();
        }
        auto fh = linear_f(hn.value(), model.mtp_fc_hidden, model.dims.hidden_size, model.dims.hidden_size);
        if (!fh) {
            return fh.error();
        }
        auto fe = linear_f(en.value(), model.mtp_fc_embed, model.dims.hidden_size, model.dims.hidden_size);
        if (!fe) {
            return fe.error();
        }
        for (std::size_t i = 0; i < fh.value().size(); ++i) {
            fh.value()[i] += fe.value()[i];
        }
        auto mstream = qwen38_expand_stream(fh.value(), model.dims.hc_count);
        auto mgr = qwen38_gated_residual(mstream, residual_view(model.mtp_attn_hc),
            model.dims.hc_count, model.dims.hidden_size, model.dims.hc_lowrank, model.dims.rms_eps);
        if (!mgr) {
            return mgr.error();
        }
        auto mq = qwen38_qsa_layer_token(mgr.value().mixed_input, qsa_view(model.mtp_qsa, model.dims), state.mtp);
        if (!mq) {
            return mq.error();
        }
        auto ma = qwen38_gated_residual_combine(
            mgr.value().residual, mq.value(), mgr.value().injection_weights, model.dims.hc_count, model.dims.hidden_size);
        if (!ma) {
            return ma.error();
        }
        auto mm = qwen38_gated_residual(ma.value(), residual_view(model.mtp_mlp_hc),
            model.dims.hc_count, model.dims.hidden_size, model.dims.hc_lowrank, model.dims.rms_eps);
        if (!mm) {
            return mm.error();
        }
        auto mlp = qwen38_moe_block(mm.value().mixed_input, model.mtp_moe, model.mtp_mlp, state.experts,
            2, model.dims.hidden_size, model.dims.moe_intermediate, model.dims.experts_per_token,
            model.dims.norm_topk_prob);
        if (!mlp) {
            return mlp.error();
        }
        auto mb = qwen38_gated_residual_combine(
            mm.value().residual, mlp.value(), mm.value().injection_weights, model.dims.hc_count, model.dims.hidden_size);
        if (!mb) {
            return mb.error();
        }
        auto mh = mix_to_hidden(mb.value(), model.mtp_mixer, model.dims);
        if (!mh) {
            return mh.error();
        }
        auto ml = project_logits(model, mh.value());
        if (!ml) {
            return ml.error();
        }
        *mtp_logits = std::move(ml.value());
    }
    return hidden;
}

}  // namespace

Result<Qwen38ReducedState> qwen38_reduced_zero_state(const Qwen38ReducedDims& dims) {
    if (dims.hidden_size == 0 || dims.dn_kernel < 2 || dims.dn_k_heads == 0 || dims.dn_v_heads == 0
        || (dims.dn_v_heads % dims.dn_k_heads) != 0) {
        return make_error(ErrorCode::invalid_argument, "reduced dims invalid", "zero_state");
    }
    const std::size_t key_dim = static_cast<std::size_t>(dims.dn_key_dim) * dims.dn_k_heads;
    const std::size_t value_dim = static_cast<std::size_t>(dims.dn_value_dim) * dims.dn_v_heads;
    const std::size_t conv_dim = key_dim * 2U + value_dim;
    Qwen38ReducedState state;
    state.delta.conv.assign(conv_dim * (dims.dn_kernel - 1U), 0.0F);
    state.delta.ssm.assign(static_cast<std::size_t>(dims.dn_v_heads) * dims.dn_key_dim * dims.dn_value_dim, 0.0F);
    state.experts.capacity = dims.expert_cache_slots;
    return state;
}

Result<Qwen38ReducedOutput> qwen38_reduced_prefill(
    const Qwen38ReducedModel& model,
    std::span<const std::int32_t> token_ids,
    Qwen38ReducedState& state) {
    if (token_ids.empty()) {
        return make_error(ErrorCode::invalid_argument, "prefill tokens must be non-empty", "reduced_prefill");
    }
    Qwen38ReducedOutput out;
    out.hidden.reserve(token_ids.size() * model.dims.hidden_size);
    out.logits.reserve(token_ids.size() * model.dims.vocab_size);
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        std::vector<float> mtp;
        auto hidden = reduced_one_token(model, token_ids[i], state, true, &mtp);
        if (!hidden) {
            return hidden.error();
        }
        auto logits = project_logits(model, hidden.value());
        if (!logits) {
            return logits.error();
        }
        out.hidden.insert(out.hidden.end(), hidden.value().begin(), hidden.value().end());
        out.logits.insert(out.logits.end(), logits.value().begin(), logits.value().end());
        if (i + 1U == token_ids.size()) {
            out.mtp_logits = std::move(mtp);
        }
    }
    out.expert_cache = state.experts.telemetry;
    return out;
}

Result<Qwen38ReducedOutput> qwen38_reduced_decode(
    const Qwen38ReducedModel& model,
    std::int32_t token_id,
    Qwen38ReducedState& state) {
    std::vector<float> mtp;
    auto hidden = reduced_one_token(model, token_id, state, true, &mtp);
    if (!hidden) {
        return hidden.error();
    }
    auto logits = project_logits(model, hidden.value());
    if (!logits) {
        return logits.error();
    }
    Qwen38ReducedOutput out;
    out.hidden = std::move(hidden.value());
    out.logits = std::move(logits.value());
    out.mtp_logits = std::move(mtp);
    out.expert_cache = state.experts.telemetry;
    return out;
}

namespace {

std::int32_t sample_token(
    std::span<const float> logits,
    std::size_t top_k,
    float temperature,
    std::mt19937_64& rng) {
    const std::size_t n = logits.size();
    if (n == 0) {
        return 0;
    }
    std::vector<std::pair<double, std::int32_t>> scored;
    scored.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        scored.push_back({static_cast<double>(logits[i]), static_cast<std::int32_t>(i)});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });
    const std::size_t k = (std::min)(top_k == 0 ? n : top_k, n);
    if (temperature <= 0.0F) {
        return scored.front().second;
    }
    std::vector<double> probs(k, 0.0);
    double max_l = scored.front().first;
    double sum = 0.0;
    const double inv_temp = 1.0 / static_cast<double>(temperature);
    for (std::size_t i = 0; i < k; ++i) {
        probs[i] = std::exp((scored[i].first - max_l) * inv_temp);
        sum += probs[i];
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        return scored.front().second;
    }
    std::uniform_real_distribution<double> dist(0.0, sum);
    const double draw = dist(rng);
    double acc = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        acc += probs[i];
        if (draw <= acc) {
            return scored[i].second;
        }
    }
    return scored.back().second;
}

}  // namespace

Result<Qwen38ReducedGenerationResult> qwen38_reduced_generate(
    const Qwen38ReducedModel& model,
    std::span<const std::int32_t> prompt_ids,
    const Qwen38ReducedGenerationOptions& options) {
    if (prompt_ids.empty()) {
        return make_error(ErrorCode::invalid_argument, "generate prompt must be non-empty", "reduced_generate");
    }
    if (options.max_new_tokens == 0U || options.max_new_tokens > kQwen38ReducedMaxTokens) {
        return make_error(ErrorCode::invalid_argument,
            "generate max_new_tokens out of range", std::to_string(options.max_new_tokens));
    }
    for (std::int32_t id : prompt_ids) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= model.dims.vocab_size) {
            return make_error(ErrorCode::invalid_argument, "prompt token out of range", std::to_string(id));
        }
    }
    const auto start = std::chrono::steady_clock::now();
    auto state = qwen38_reduced_zero_state(model.dims);
    if (!state) {
        return state.error();
    }
    auto prefill = qwen38_reduced_prefill(model, prompt_ids, state.value());
    if (!prefill) {
        return prefill.error();
    }
    const auto after_prompt = std::chrono::steady_clock::now();
    const double prompt_eval_ms =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(after_prompt - start).count())
        / 1000.0;

    Qwen38ReducedGenerationResult out;
    out.prompt_ids.assign(prompt_ids.begin(), prompt_ids.end());
    out.stats.prompt_tokens = prompt_ids.size();
    out.stats.requested_max_tokens = options.max_new_tokens;
    out.stats.effective_max_tokens = (std::min)(options.max_new_tokens,
        options.context_limit_tokens > prompt_ids.size() ? options.context_limit_tokens - prompt_ids.size()
                                                         : std::size_t{0});
    std::mt19937_64 rng(options.seed);
    std::int32_t token = static_cast<std::int32_t>(prompt_ids.back());
    bool stop = false;

    while (out.generated_ids.size() < out.stats.effective_max_tokens) {
        ++out.stats.forward_passes;
        if (options.cancelled && options.cancelled()) {
            out.finish_reason = Qwen38ReducedFinishReason::cancelled;
            stop = true;
            break;
        }
        if (options.timeout.count() > 0) {
            const auto now = std::chrono::steady_clock::now();
            if (now - start > options.timeout) {
                out.finish_reason = Qwen38ReducedFinishReason::timeout;
                stop = true;
                break;
            }
        }
        auto decoded = qwen38_reduced_decode(model, token, state.value());
        if (!decoded) {
            return decoded.error();
        }
        token = sample_token(decoded.value().logits, options.top_k, options.temperature, rng);
        out.generated_ids.push_back(token);
        if (options.on_token) {
            options.on_token(token);
        }
        if (std::find(options.stop_token_ids.begin(), options.stop_token_ids.end(), token)
            != options.stop_token_ids.end()) {
            out.finish_reason = Qwen38ReducedFinishReason::stop_token;
            out.stop_token_id = token;
            stop = true;
            break;
        }
        if (out.generated_ids.size() + prompt_ids.size() >= options.context_limit_tokens) {
            out.finish_reason = Qwen38ReducedFinishReason::context_limit;
            stop = true;
            break;
        }
    }
    if (!stop) {
        out.finish_reason = Qwen38ReducedFinishReason::max_tokens;
    }
    const auto end = std::chrono::steady_clock::now();
    const double generation_ms =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - after_prompt).count())
        / 1000.0;
    out.stats.generated_tokens = out.generated_ids.size();
    out.stats.forward_passes = out.generated_ids.size();
    out.stats.expert_requests = state.value().experts.telemetry.requests;
    out.stats.expert_cache_hits = state.value().experts.telemetry.hits;
    out.stats.expert_cache_misses = state.value().experts.telemetry.misses;
    out.stats.logical_expert_bytes_read = state.value().experts.telemetry.logical_bytes;
    out.stats.prompt_eval_ms = prompt_eval_ms;
    out.stats.generation_ms = generation_ms;
    const double total_s = (prompt_eval_ms + generation_ms) / 1000.0;
    out.stats.tokens_per_second =
        total_s > 0.0 ? static_cast<double>(out.generated_ids.size()) / total_s : 0.0;
    return out;
}

std::string_view to_string(Qwen38ReducedFinishReason reason) noexcept {
    using R = Qwen38ReducedFinishReason;
    switch (reason) {
        case R::stop_token: return "stop_token";
        case R::context_limit: return "context_limit";
        case R::max_tokens: return "max_tokens";
        case R::cancelled: return "cancelled";
        case R::timeout: return "timeout";
    }
    return "unknown";
}

}  // namespace vastloom::inference
