#include "vastloom/inference/ggml_compute.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <vector>

#include "vastloom/inference/ggml_quantization.hpp"
#include "vastloom/safety/resource_governor.hpp"

#if defined(VASTLOOM_HAVE_GGML)
#include "ggml-cpu.h"
#include "ggml.h"
#endif

namespace vastloom::inference {
namespace {

std::atomic<std::uint64_t> g_accelerated{0U};
std::atomic<std::uint64_t> g_fallback{0U};
std::atomic<std::uint64_t> g_input_quantizes{0U};
std::atomic<std::uint64_t> g_input_quantize_hits{0U};

// ggml accepts unaligned data on x86 (GGML_MEM_ALIGN is 4 there), but the streamed buffers are
// at least max_align-aligned in practice and requiring it keeps the SIMD paths on the footing
// they were measured on. Anything less stays on the scalar reference path.
constexpr std::size_t kMinimumDataAlignment = alignof(std::max_align_t);

bool aligned_enough(const void* pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) % kMinimumDataAlignment) == 0U;
}

#if defined(VASTLOOM_HAVE_GGML)

struct QuantizedActivation {
    const float* source{nullptr};
    std::size_t columns{0U};
    ggml_type type{GGML_TYPE_COUNT};
    std::uint64_t fingerprint{0U};
    std::vector<std::byte> encoded;
};

struct GgmlBackendState {
    safety::Reservation reservation;
    std::vector<std::byte> pool;
    std::vector<std::uint8_t> work;
    ggml_context* context{nullptr};
    ggml_threadpool_t threadpool{nullptr};
    unsigned threads{0U};
    QuantizedActivation activation;
};

std::uint64_t fingerprint_f32(std::span<const float> input) noexcept {
    std::uint64_t hash = input.size() ^ UINT64_C(14695981039346656037);
    const auto* words = reinterpret_cast<const std::uint32_t*>(input.data());
    for (std::size_t index = 0U; index < input.size(); ++index) {
        hash ^= words[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::mutex& state_mutex() {
    static std::mutex guard;
    return guard;
}

GgmlBackendState& backend_state() {
    static GgmlBackendState instance;
    return instance;
}

std::atomic<bool> g_active{false};

ggml_type to_ggml_type(std::uint32_t ggml_type) noexcept {
    switch (ggml_type) {
        case kGgmlTypeF32: return GGML_TYPE_F32;
        case kGgmlTypeQ8_0: return GGML_TYPE_Q8_0;
        case kGgmlTypeQ4K: return GGML_TYPE_Q4_K;
        case kGgmlTypeQ5K: return GGML_TYPE_Q5_K;
        case kGgmlTypeQ6K: return GGML_TYPE_Q6_K;
        case kGgmlTypeIq2Xxs: return GGML_TYPE_IQ2_XXS;
        case kGgmlTypeIq3Xxs: return GGML_TYPE_IQ3_XXS;
        case kGgmlTypeIq1S: return GGML_TYPE_IQ1_S;
        case kGgmlTypeIq2S: return GGML_TYPE_IQ2_S;
        case kGgmlTypeBf16: return GGML_TYPE_BF16;
        default: return GGML_TYPE_COUNT;
    }
}

#endif

}  // namespace

bool ggml_compute_compiled() noexcept {
#if defined(VASTLOOM_HAVE_GGML)
    return true;
#else
    return false;
#endif
}

bool ggml_compute_enabled() noexcept {
#if defined(VASTLOOM_HAVE_GGML)
    return g_active.load(std::memory_order_acquire);
#else
    return false;
#endif
}

void ggml_compute_note_fallback() noexcept { g_fallback.fetch_add(1U, std::memory_order_relaxed); }

std::uint64_t ggml_compute_accelerated_count() noexcept {
    return g_accelerated.load(std::memory_order_relaxed);
}

std::uint64_t ggml_compute_fallback_count() noexcept {
    return g_fallback.load(std::memory_order_relaxed);
}

std::uint64_t ggml_compute_input_quantize_count() noexcept {
    return g_input_quantizes.load(std::memory_order_relaxed);
}

std::uint64_t ggml_compute_input_quantize_hits() noexcept {
    return g_input_quantize_hits.load(std::memory_order_relaxed);
}

#if defined(VASTLOOM_HAVE_GGML)

Result<void> ggml_compute_enable(safety::ResourceGovernor& governor,
                                 const GgmlComputeOptions& options) {
    if (options.threads == 0U || options.pool_bytes == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "ggml compute backend needs at least one thread and a graph pool");
    }
    std::lock_guard<std::mutex> lock(state_mutex());
    auto& state = backend_state();
    if (g_active.load(std::memory_order_acquire)) {
        return make_error(ErrorCode::invalid_argument, "ggml compute backend is already enabled");
    }
    // Bounded, Governor-charged memory: the per-call graph pool and the plan work buffer live
    // here for the life of the backend rather than being re-allocated on every matvec.
    auto reserved = governor.try_reserve(
        safety::ResourceKind::scratch, options.pool_bytes, "ggml compute pool");
    if (!reserved) return reserved.error();
    state.reservation = std::move(reserved.value());
    try {
        state.pool.resize(options.pool_bytes);
    } catch (const std::bad_alloc&) {
        state.reservation.release();
        return make_error(ErrorCode::resource_exhausted,
                          "Unable to allocate the ggml compute graph pool");
    }
    auto params = ggml_threadpool_params_default(static_cast<int>(options.threads));
    state.threadpool = ggml_threadpool_new(&params);
    if (state.threadpool == nullptr) {
        state.pool.clear();
        state.pool.shrink_to_fit();
        state.reservation.release();
        return make_error(ErrorCode::internal_error, "Unable to create the ggml thread pool");
    }
    ggml_init_params init_params;
    init_params.mem_size = state.pool.size();
    init_params.mem_buffer = state.pool.data();
    init_params.no_alloc = true;
    state.context = ggml_init(init_params);
    if (state.context == nullptr) {
        ggml_threadpool_free(state.threadpool);
        state.threadpool = nullptr;
        state.pool.clear();
        state.pool.shrink_to_fit();
        state.reservation.release();
        return make_error(ErrorCode::internal_error, "Unable to init the ggml compute context");
    }
    state.threads = options.threads;
    g_active.store(true, std::memory_order_release);
    return {};
}

void ggml_compute_disable() noexcept {
    std::lock_guard<std::mutex> lock(state_mutex());
    auto& state = backend_state();
    if (!g_active.load(std::memory_order_acquire)) return;
    g_active.store(false, std::memory_order_release);
    if (state.context != nullptr) {
        ggml_free(state.context);
        state.context = nullptr;
    }
    if (state.threadpool != nullptr) {
        ggml_threadpool_free(state.threadpool);
        state.threadpool = nullptr;
    }
    state.activation = {};
    state.work.clear();
    state.work.shrink_to_fit();
    state.pool.clear();
    state.pool.shrink_to_fit();
    state.reservation.release();
    state.threads = 0U;
}

bool ggml_compute_supports(std::uint32_t ggml_type) noexcept {
    return to_ggml_type(ggml_type) != GGML_TYPE_COUNT;
}

bool ggml_compute_can_run(std::uint32_t ggml_type,
                          const std::byte* encoded_matrix,
                          std::size_t rows,
                          std::size_t columns) noexcept {
    if (!ggml_compute_enabled() || encoded_matrix == nullptr || rows == 0U || columns == 0U) {
        return false;
    }
    const auto type = to_ggml_type(ggml_type);
    if (type == GGML_TYPE_COUNT) return false;
    // ggml_row_size divides ne0 by the block size; a partial block would truncate silently.
    const auto block = static_cast<std::size_t>(ggml_blck_size(type));
    if (block == 0U || columns % block != 0U) return false;
    return aligned_enough(encoded_matrix);
}

Result<void> matvec_ggml_backend(std::uint32_t ggml_type,
                                 std::span<const std::byte> encoded_matrix,
                                 std::size_t rows,
                                 std::size_t columns,
                                 std::span<const float> input,
                                 std::span<float> output) {
    if (!ggml_compute_enabled()) {
        return make_error(ErrorCode::unsupported, "ggml compute backend is not enabled");
    }
    if (!ggml_compute_can_run(ggml_type, encoded_matrix.data(), rows, columns) ||
        input.size() != columns || output.size() != rows) {
        return make_error(ErrorCode::invalid_argument,
                          "ggml compute matvec shape or alignment is not supported");
    }
    const auto type = to_ggml_type(ggml_type);
    const auto expected_bytes =
        (columns / static_cast<std::size_t>(ggml_blck_size(type))) *
        static_cast<std::size_t>(ggml_type_size(type)) * rows;
    if (encoded_matrix.size() < expected_bytes) {
        return make_error(ErrorCode::invalid_argument,
                          "ggml compute matvec buffer is smaller than its shape");
    }

    auto& state = backend_state();
    if (state.context == nullptr) {
        return make_error(ErrorCode::internal_error, "ggml compute context is not initialized");
    }
    ggml_reset(state.context);
    ggml_context* context = state.context;

    // The weights point at the streamed bytes. The activation is quantized once to the weight's
    // vec_dot type and reused for later matvecs of the same vector (gate and up of every expert
    // in a layer share one hidden state). ggml then skips its own from_float.
    enum ggml_type activation_type = GGML_TYPE_F32;
    const void* input_data = input.data();
    const auto* cpu_traits = ggml_get_type_traits_cpu(type);
    if (cpu_traits != nullptr && cpu_traits->vec_dot_type != GGML_TYPE_F32 &&
        cpu_traits->vec_dot_type != GGML_TYPE_COUNT) {
        const auto* quant_traits = ggml_get_type_traits_cpu(cpu_traits->vec_dot_type);
        if (quant_traits != nullptr && quant_traits->from_float != nullptr) {
            const auto fingerprint = fingerprint_f32(input);
            auto& cached = state.activation;
            if (cached.source == input.data() && cached.columns == columns &&
                cached.type == cpu_traits->vec_dot_type && cached.fingerprint == fingerprint &&
                !cached.encoded.empty()) {
                g_input_quantize_hits.fetch_add(1U, std::memory_order_relaxed);
            } else {
                const auto encoded_bytes = ggml_row_size(cpu_traits->vec_dot_type, columns);
                try {
                    cached.encoded.resize(encoded_bytes);
                } catch (const std::bad_alloc&) {
                    return make_error(ErrorCode::resource_exhausted,
                                      "Unable to allocate the ggml activation encoding");
                }
                quant_traits->from_float(
                    input.data(), cached.encoded.data(), static_cast<int64_t>(columns));
                cached.source = input.data();
                cached.columns = columns;
                cached.type = cpu_traits->vec_dot_type;
                cached.fingerprint = fingerprint;
                g_input_quantizes.fetch_add(1U, std::memory_order_relaxed);
            }
            activation_type = cached.type;
            input_data = cached.encoded.data();
        }
    }

    ggml_tensor* weights = ggml_new_tensor_2d(
        context, type, static_cast<int64_t>(columns), static_cast<int64_t>(rows));
    ggml_tensor* input_tensor =
        ggml_new_tensor_1d(context, activation_type, static_cast<int64_t>(columns));
    if (weights == nullptr || input_tensor == nullptr) {
        return make_error(ErrorCode::resource_exhausted,
                          "The ggml compute graph pool is too small for this matrix");
    }
    weights->data = const_cast<void*>(static_cast<const void*>(encoded_matrix.data()));
    input_tensor->data = const_cast<void*>(input_data);
    ggml_tensor* result = ggml_mul_mat(context, weights, input_tensor);
    if (result == nullptr) {
        return make_error(ErrorCode::internal_error, "Unable to build the ggml matvec node");
    }
    result->data = output.data();

    ggml_cgraph* graph = ggml_new_graph(context);
    if (graph == nullptr) {
        return make_error(ErrorCode::resource_exhausted,
                          "The ggml compute graph pool is too small for this graph");
    }
    ggml_build_forward_expand(graph, result);

    ggml_cplan plan = ggml_graph_plan(graph, static_cast<int>(state.threads), state.threadpool);
    if (plan.work_size > state.work.size()) {
        try {
            state.work.resize(plan.work_size);
        } catch (const std::bad_alloc&) {
            return make_error(ErrorCode::resource_exhausted,
                              "Unable to allocate the ggml compute work buffer");
        }
    }
    plan.work_data = state.work.data();
    const auto status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        return make_error(ErrorCode::internal_error, "ggml matvec compute did not succeed");
    }
    g_accelerated.fetch_add(1U, std::memory_order_relaxed);
    return {};
}

#else  // VASTLOOM_HAVE_GGML

Result<void> ggml_compute_enable(safety::ResourceGovernor&, const GgmlComputeOptions&) {
    return make_error(ErrorCode::unsupported,
                      "This build has no ggml checkout; configure with VASTLOOM_GGML_ROOT");
}

void ggml_compute_disable() noexcept {}

bool ggml_compute_supports(std::uint32_t) noexcept { return false; }

bool ggml_compute_can_run(std::uint32_t, const std::byte*, std::size_t, std::size_t) noexcept {
    return false;
}

Result<void> matvec_ggml_backend(std::uint32_t,
                                 std::span<const std::byte>,
                                 std::size_t,
                                 std::size_t,
                                 std::span<const float>,
                                 std::span<float>) {
    return make_error(ErrorCode::unsupported,
                      "This build has no ggml checkout; configure with VASTLOOM_GGML_ROOT");
}

#endif

}  // namespace vastloom::inference
