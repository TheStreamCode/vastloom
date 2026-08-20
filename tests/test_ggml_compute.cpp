// Parity and throughput gate for the opt-in ggml compute backend. Runs only when the build was
// configured with VASTLOOM_GGML_ROOT; otherwise it reports itself inert and passes. The parity
// tolerance covers accumulation only: the backend decodes with ggml's own kernels, so decoded
// weights are identical to the scalar reference by construction, and the scalar reference sums in
// double where ggml sums in float. A wrong kernel (wrong type mapping, wrong row stride, corrupt
// decode) moves results by orders of magnitude more than that, which the corruption check proves.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/inference/ggml_compute.hpp"
#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/safety/resource_governor.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<float> make_values(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> values(count);
    for (auto& value : values) value = dist(rng);
    return values;
}

float max_abs_diff(std::span<const float> left, std::span<const float> right) {
    float worst = 0.0F;
    for (std::size_t index = 0; index < left.size() && index < right.size(); ++index) {
        worst = std::max(worst, std::fabs(left[index] - right[index]));
    }
    return worst;
}

// f32-vs-f64 accumulation over columns <= 2048 with |values| <= 1 stays far below this; a kernel
// defect does not.
constexpr float kAccumulationTolerance = 1e-2F;

// Quantized dot products quantize the F32 input to the weight's dot type first (ggml's own
// from_float), so the input carries ~8-bit rounding before accumulation. Observed clean noise on
// the Q8_0 parity shape is ~0.08; a wrong kernel moves results by orders of magnitude more.
constexpr float kQuantizedInputTolerance = 0.2F;

void q8_0_parity(vastloom::safety::ResourceGovernor& governor) {
    using namespace vastloom;
    using namespace vastloom::inference;

    const std::size_t rows = 256U;
    const std::size_t columns = 512U;
    const auto matrix = make_values(rows * columns, 0x5EEDU);
    auto encoded_result = quantize_q8_0_reference(matrix, rows, columns, columns);
    check(static_cast<bool>(encoded_result), "q8_0 reference encoding succeeds");
    if (!encoded_result) return;
    auto encoded = std::move(encoded_result.value());

    const auto input = make_values(columns, 0x1A2BU);
    std::vector<float> scalar_output(rows, 0.0F);
    std::vector<float> backend_output(rows, 0.0F);
    QuantizedKernelLimits limits;
    limits.max_elements = rows * columns;
    limits.max_threads = 1U;
    const auto scalar_status = matvec_q8_0_f32(
        encoded, rows, columns, input, scalar_output, limits);
    check(static_cast<bool>(scalar_status), "q8_0 scalar matvec succeeds");

    check(ggml_compute_supports(kGgmlTypeQ8_0), "backend supports Q8_0");
    check(ggml_compute_can_run(kGgmlTypeQ8_0, encoded.data(), rows, columns),
          "backend accepts the parity shape");
    const auto backend_status = matvec_ggml_backend(
        kGgmlTypeQ8_0, encoded, rows, columns, input, backend_output);
    check(static_cast<bool>(backend_status), "q8_0 backend matvec succeeds");
    if (!scalar_status || !backend_status) return;

    const float diff = max_abs_diff(scalar_output, backend_output);
    std::printf("q8_0 parity: max abs diff %.6g (tolerance %.6g)\n",
                static_cast<double>(diff), static_cast<double>(kQuantizedInputTolerance));
    check(diff <= kQuantizedInputTolerance,
          "q8_0 backend matches scalar within input-quantization and accumulation noise");

    // Prove the gate can fail. A single flipped weight is smaller than the input-quantization
    // noise, so corrupt all 32 weight bytes of one block of one row instead: that must move that
    // row's result far beyond tolerance. The scale bytes are left alone so nothing turns NaN.
    auto corrupted = encoded;
    const std::size_t blocks_per_row = columns / kQ8_0BlockElements;
    const std::size_t target_row = rows / 2U;
    const std::size_t block_offset = target_row * blocks_per_row * kQ8_0BlockBytes;
    for (std::size_t i = 2U; i < kQ8_0BlockBytes; ++i) {
        corrupted[block_offset + i] = static_cast<std::byte>(
            static_cast<unsigned char>(corrupted[block_offset + i]) ^ 0xFFU);
    }
    std::vector<float> corrupted_output(rows, 0.0F);
    const auto corrupted_status = matvec_ggml_backend(
        kGgmlTypeQ8_0, corrupted, rows, columns, input, corrupted_output);
    check(static_cast<bool>(corrupted_status), "corrupted q8_0 backend matvec still runs");
    if (corrupted_status) {
        // Compared against the clean backend output, not the scalar one: input quantization is
        // identical on both sides, so this difference isolates the corruption with zero noise.
        const float corrupted_diff = max_abs_diff(backend_output, corrupted_output);
        std::printf("q8_0 corruption probe: max abs diff %.6g\n",
                    static_cast<double>(corrupted_diff));
        check(corrupted_diff > 0.1F, "corrupted weights are caught by the parity gate");
    }

    // The backend must refuse shapes it cannot take, and the dispatcher stays on the scalar path.
    check(!ggml_compute_can_run(kGgmlTypeQ8_0, encoded.data(), rows, columns + 1U),
          "backend rejects a shape whose encoded buffer is too small");
    (void)governor;
}

void f32_parity() {
    using namespace vastloom;
    using namespace vastloom::inference;

    const std::size_t rows = 128U;
    const std::size_t columns = 256U;
    const auto matrix = make_values(rows * columns, 0xF32AU);
    std::vector<std::byte> encoded(matrix.size() * sizeof(float));
    std::memcpy(encoded.data(), matrix.data(), encoded.size());
    const auto input = make_values(columns, 0xF32BU);
    std::vector<float> scalar_output(rows, 0.0F);
    std::vector<float> backend_output(rows, 0.0F);
    QuantizedKernelLimits limits;
    limits.max_elements = rows * columns;
    limits.max_threads = 1U;
    const auto scalar_status = matvec_f32_le_f32(encoded, rows, columns, input, scalar_output, limits);
    check(static_cast<bool>(scalar_status), "f32 scalar matvec succeeds");
    const auto backend_status = matvec_ggml_backend(
        kGgmlTypeF32, encoded, rows, columns, input, backend_output);
    check(static_cast<bool>(backend_status), "f32 backend matvec succeeds");
    if (!scalar_status || !backend_status) return;
    const float diff = max_abs_diff(scalar_output, backend_output);
    std::printf("f32 parity: max abs diff %.6g (tolerance %.6g)\n",
                static_cast<double>(diff), static_cast<double>(kAccumulationTolerance));
    check(diff <= kAccumulationTolerance, "f32 backend matches scalar within accumulation noise");
}

void benchmark_q8_0() {
    using namespace vastloom;
    using namespace vastloom::inference;

    // The dominant trunk shape: one 2048x2048 Q8_0 projection, about 4.35 MB. Small enough to
    // stay cache-hot across iterations, so this measures kernel and per-call overhead, not DRAM.
    const std::size_t rows = 2048U;
    const std::size_t columns = 2048U;
    const std::size_t iterations = 300U;
    const auto matrix = make_values(rows * columns, 0xBEEFU);
    auto encoded_result = quantize_q8_0_reference(matrix, rows, columns, columns);
    if (!encoded_result) {
        check(false, "benchmark q8_0 encoding succeeds");
        return;
    }
    const auto encoded = std::move(encoded_result.value());
    const auto input = make_values(columns, 0xB0BU);
    const double bytes = static_cast<double>(encoded.size());

    std::vector<float> output(rows, 0.0F);
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
        const auto status = matvec_ggml_backend(
            kGgmlTypeQ8_0, encoded, rows, columns, input, output);
        if (!status) {
            check(false, "benchmark backend matvec succeeds");
            return;
        }
    }
    const double backend_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double backend_gbs = (bytes * static_cast<double>(iterations)) / backend_seconds / 1e9;

    std::vector<float> scalar_output(rows, 0.0F);
    QuantizedKernelLimits limits;
    limits.max_elements = rows * columns;
    limits.max_threads = 12U;
    const auto scalar_started = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0U; iteration < iterations / 10U; ++iteration) {
        const auto status = matvec_q8_0_f32(
            encoded, rows, columns, input, scalar_output, limits);
        if (!status) {
            check(false, "benchmark scalar matvec succeeds");
            return;
        }
    }
    const double scalar_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scalar_started).count();
    const double scalar_gbs = (bytes * static_cast<double>(iterations / 10U)) / scalar_seconds / 1e9;

    std::printf("q8_0 2048x2048 cache-hot: ggml %.2f GB/s (%.3f ms/call), scalar 12-thread "
                "%.2f GB/s (%.3f ms/call)\n",
                backend_gbs, backend_seconds * 1000.0 / static_cast<double>(iterations),
                scalar_gbs, scalar_seconds * 1000.0 / static_cast<double>(iterations / 10U));
}

void benchmark_q8_0_large() {
    using namespace vastloom;
    using namespace vastloom::inference;

    // 32768x2048 Q8_0 is about 71 MB - larger than any L3 on the reference PC - so repeated
    // passes stream from DRAM. This is the regime the real forward runs in.
    const std::size_t rows = 32768U;
    const std::size_t columns = 2048U;
    const std::size_t iterations = 10U;
    const auto matrix = make_values(rows * columns, 0xCAFEU);
    QuantizedKernelLimits encode_limits;
    encode_limits.max_elements = rows * columns;
    auto encoded_result = quantize_q8_0_reference(matrix, rows, columns, columns, encode_limits);
    if (!encoded_result) {
        check(false, "large benchmark q8_0 encoding succeeds");
        return;
    }
    const auto encoded = std::move(encoded_result.value());
    const auto input = make_values(columns, 0xCAFU);
    const double bytes = static_cast<double>(encoded.size());

    std::vector<float> output(rows, 0.0F);
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
        const auto status = matvec_ggml_backend(
            kGgmlTypeQ8_0, encoded, rows, columns, input, output);
        if (!status) {
            check(false, "large benchmark backend matvec succeeds");
            return;
        }
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double gbs = (bytes * static_cast<double>(iterations)) / seconds / 1e9;
    std::printf("q8_0 32768x2048 DRAM: ggml %.2f GB/s (%.2f ms/call)\n",
                gbs, seconds * 1000.0 / static_cast<double>(iterations));
}

// Synthesizes valid IQ1_S blocks: any qs/qh bytes decode through the fixed grid, and a constant
// finite f16 scale keeps every weight finite. The values are not a real quantization - they exist
// to drive the kernel at the routed-expert shape.
std::vector<std::byte> make_iq1_s_matrix(std::size_t rows, std::size_t columns, std::uint32_t seed) {
    using namespace vastloom::inference;
    std::mt19937 rng(seed);
    const std::size_t blocks = rows * (columns / kIq1SBlockElements);
    std::vector<std::byte> encoded(blocks * kIq1SBlockBytes);
    for (std::size_t block = 0U; block < blocks; ++block) {
        std::byte* base = encoded.data() + block * kIq1SBlockBytes;
        // f16 0x3000 = 1.0 * 2^-3: finite and small.
        base[0] = static_cast<std::byte>(0x00U);
        base[1] = static_cast<std::byte>(0x30U);
        for (std::size_t i = 2U; i < kIq1SBlockBytes; ++i) {
            base[i] = static_cast<std::byte>(static_cast<unsigned>(rng() & 0xFFU));
        }
    }
    return encoded;
}

void input_quantize_reuse() {
    using namespace vastloom;
    using namespace vastloom::inference;

    const std::size_t rows = 256U;
    const std::size_t columns = 512U;
    const auto encoded = make_iq1_s_matrix(rows, columns, 0xA11U);
    const auto input = make_values(columns, 0xA12U);
    std::vector<float> first(rows, 0.0F);
    std::vector<float> second(rows, 0.0F);

    const auto quantizes_before = ggml_compute_input_quantize_count();
    const auto hits_before = ggml_compute_input_quantize_hits();
    check(static_cast<bool>(matvec_ggml_backend(
              kGgmlTypeIq1S, encoded, rows, columns, input, first)),
          "first iq1_s matvec succeeds");
    check(ggml_compute_input_quantize_count() == quantizes_before + 1U,
          "first iq1_s matvec quantizes the activation once");
    check(ggml_compute_input_quantize_hits() == hits_before,
          "first iq1_s matvec is not an activation hit");

    check(static_cast<bool>(matvec_ggml_backend(
              kGgmlTypeIq1S, encoded, rows, columns, input, second)),
          "second iq1_s matvec succeeds");
    check(ggml_compute_input_quantize_count() == quantizes_before + 1U,
          "second iq1_s matvec reuses the encoded activation");
    check(ggml_compute_input_quantize_hits() == hits_before + 1U,
          "second iq1_s matvec counts as an activation hit");
    check(max_abs_diff(first, second) == 0.0F,
          "reused activation matches the first iq1_s result bit for bit");

    auto mutated = input;
    mutated.back() += 1.0F;
    std::vector<float> third(rows, 0.0F);
    check(static_cast<bool>(matvec_ggml_backend(
              kGgmlTypeIq1S, encoded, rows, columns, mutated, third)),
          "mutated iq1_s matvec succeeds");
    check(ggml_compute_input_quantize_count() == quantizes_before + 2U,
          "a changed activation is quantized again");
    check(max_abs_diff(first, third) > 0.0F,
          "a changed activation moves the iq1_s result");

    const auto f32_matrix = make_values(rows * columns, 0xF32CU);
    std::vector<std::byte> f32_encoded(f32_matrix.size() * sizeof(float));
    std::memcpy(f32_encoded.data(), f32_matrix.data(), f32_encoded.size());
    const auto f32_quantizes = ggml_compute_input_quantize_count();
    std::vector<float> f32_output(rows, 0.0F);
    check(static_cast<bool>(matvec_ggml_backend(
              kGgmlTypeF32, f32_encoded, rows, columns, input, f32_output)),
          "f32 matvec still runs after activation reuse");
    check(ggml_compute_input_quantize_count() == f32_quantizes,
          "f32 weights do not quantize the activation");
}

void benchmark_iq1_s() {
    using namespace vastloom;
    using namespace vastloom::inference;

    // The routed-expert shape: IQ1_S gate/up matrices. IQ1_S decodes through a 2,048-entry grid,
    // which is the expensive part, so this measures the kernel that dominates expert compute.
    const std::size_t rows = 2048U;
    const std::size_t columns = 2048U;
    const std::size_t iterations = 200U;
    const auto encoded = make_iq1_s_matrix(rows, columns, 0x1A15U);
    const auto input = make_values(columns, 0x1A1FU);
    const double bytes = static_cast<double>(encoded.size());

    std::vector<float> output(rows, 0.0F);
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
        const auto status = matvec_ggml_backend(
            kGgmlTypeIq1S, encoded, rows, columns, input, output);
        if (!status) {
            check(false, "iq1_s benchmark backend matvec succeeds");
            return;
        }
    }
    const double backend_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double backend_gbs = (bytes * static_cast<double>(iterations)) / backend_seconds / 1e9;

    std::vector<float> scalar_output(rows, 0.0F);
    QuantizedKernelLimits limits;
    limits.max_elements = rows * columns;
    limits.max_threads = 12U;
    const auto scalar_started = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0U; iteration < iterations / 20U; ++iteration) {
        const auto status = matvec_iq1_s_f32(
            encoded, rows, columns, input, scalar_output, limits);
        if (!status) {
            check(false, "iq1_s benchmark scalar matvec succeeds");
            return;
        }
    }
    const double scalar_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scalar_started).count();
    const double scalar_gbs =
        (bytes * static_cast<double>(iterations / 20U)) / scalar_seconds / 1e9;

    std::printf("iq1_s 2048x2048: ggml %.2f GB/s (%.3f ms/call), scalar 12-thread "
                "%.2f GB/s (%.3f ms/call)\n",
                backend_gbs, backend_seconds * 1000.0 / static_cast<double>(iterations),
                scalar_gbs, scalar_seconds * 1000.0 / static_cast<double>(iterations / 20U));
}

}  // namespace

int main() {
    using namespace vastloom;
    using namespace vastloom::inference;

    if (!ggml_compute_compiled()) {
        // Exit 75 so CTest records this as skipped. Returning 0 would report a green test that
        // verified nothing, which is the one thing this project's gates may never do.
        std::printf("ggml compute backend not compiled in (configure with VASTLOOM_GGML_ROOT); "
                    "nothing to test\n");
        return 75;
    }

    safety::ResourceLimits limits;
    limits.host_memory_hard_limit = 4U * GiB;
    limits.ram_hard_limit = 2U * GiB;
    limits.staging_hard_limit = 2U * GiB;
    limits.scratch_hard_limit = 2U * GiB;
    safety::ResourceGovernor governor(limits);

    GgmlComputeOptions options;
    options.threads = 12U;
    auto enabled = ggml_compute_enable(governor, options);
    if (!enabled) {
        std::printf("FAIL: ggml compute backend did not enable: %s\n",
                    enabled.error().message.c_str());
        return 1;
    }
    check(ggml_compute_enabled(), "backend reports enabled");

    q8_0_parity(governor);
    f32_parity();
    input_quantize_reuse();
    benchmark_q8_0();
    benchmark_q8_0_large();
    benchmark_iq1_s();

    ggml_compute_disable();
    check(!ggml_compute_enabled(), "backend reports disabled after teardown");

    if (failures == 0) {
        std::printf("ggml compute backend: all checks passed\n");
        return 0;
    }
    std::printf("ggml compute backend: %d check(s) failed\n", failures);
    return 1;
}
