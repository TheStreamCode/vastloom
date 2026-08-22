#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::inference {

// One E1 capability module: a compact low-rank adapter plus a routing embedding, stored as an
// immutable derived artifact. This is the mechanics half of the falsification ladder - storage,
// retrieval and injection contracts only. Whether such modules carry real teacher capability is
// E2's question and deliberately out of scope here.

// Which frozen linear projection a target rewires. Norms and embeddings are not linear layers,
// so they are not addressable; the LM head is, keyed at the synthetic layer index below.
enum class CapabilityTarget : std::uint8_t {
    q = 0,
    k = 1,
    v = 2,
    o = 3,
    gate = 4,
    up = 5,
    down = 6,
    lm_head = 7,
};

[[nodiscard]] std::string_view to_string(CapabilityTarget target) noexcept;
[[nodiscard]] std::optional<CapabilityTarget> capability_target_from_u32(std::uint32_t raw) noexcept;

inline constexpr std::array<CapabilityTarget, 8> kAllCapabilityTargets{
    CapabilityTarget::q,    CapabilityTarget::k,     CapabilityTarget::v,
    CapabilityTarget::o,    CapabilityTarget::gate,  CapabilityTarget::up,
    CapabilityTarget::down, CapabilityTarget::lm_head};

// Synthetic layer index under which the LM head binding is stored.
inline constexpr std::uint32_t kCapabilityLmHeadLayer = 0xFFFF'FFFEU;

struct ModuleTargetSpec {
    std::uint32_t layer{0};
    CapabilityTarget target{CapabilityTarget::q};
    std::uint32_t rows{0};   // logical output width of the wrapped weight
    std::uint32_t cols{0};   // logical input width of the wrapped weight
};

struct ModuleTarget {
    ModuleTargetSpec spec;
    std::vector<float> a;  // rank x cols, row-major
    std::vector<float> b;  // rows x rank, row-major
};

// Everything the loader guarantees about one parsed module file. Values are finite, dimensions
// are within ceilings, targets are unique and the SHA-256 trailer matched the bytes before it.
struct CapabilityModule {
    std::string name;
    std::uint32_t rank{0};
    float scale{0.0F};
    std::vector<ModuleTarget> targets;
    std::array<float, 64> routing_embedding{};
    ByteCount file_bytes{0};
};

inline constexpr std::size_t kRoutingEmbeddingDim = 64;
inline constexpr std::uint32_t kMaxModuleRank = 64;
inline constexpr std::uint32_t kMaxModuleTargets = 256;
inline constexpr ByteCount kMaxModuleFileBytes = 64U * 1024U * 1024U;

// Serializes one valid module atomically enough for tests and the module-init command: the file
// is written once with its SHA-256 trailer computed over everything preceding it.
[[nodiscard]] Result<void> write_capability_module(const std::filesystem::path& path,
                                                   const CapabilityModule& module);

[[nodiscard]] Result<CapabilityModule> load_capability_module(const std::filesystem::path& path);

// Deterministic request embedding: FNV-1a over lowercased character trigrams accumulated into a
// fixed-dim bag, then L2-normalized. No std::hash anywhere - the value must not vary by platform.
[[nodiscard]] std::array<float, kRoutingEmbeddingDim> embed_request(std::string_view request);

struct RoutedModule {
    std::size_t index{0};       // position in the caller's module vector
    float similarity{0.0F};     // cosine similarity against the request embedding
};

// Orders modules by similarity to the request, strongest first, truncated to top_k. Ties break
// by lower index so the selection is stable run to run.
[[nodiscard]] std::vector<RoutedModule> route_modules(const std::vector<CapabilityModule>& modules,
                                                      std::string_view request,
                                                      std::size_t top_k);

// One resolved injection point: the low-rank delta for exactly one (layer, target) pair.
struct CapabilityOverlayBinding {
    std::uint32_t layer{0};
    CapabilityTarget target{CapabilityTarget::q};
    std::uint32_t rows{0};
    std::uint32_t cols{0};
    std::uint32_t rank{0};
    float scale{0.0F};
    const float* a{nullptr};  // rank x cols, borrowed from the owning module
    const float* b{nullptr};  // rows x rank, borrowed from the owning module
    std::string source_name;
};

// The runtime view handed to a generation call. Bindings borrow module storage, so every module
// a caller wants applied must outlive the overlay and the generation using it. The base model's
// weights are never touched: the delta is added after each wrapped matvec, which is what makes
// "frozen base" a structural fact rather than a promise.
class CapabilityOverlay {
public:
    CapabilityOverlay() = default;

    void add_binding(CapabilityOverlayBinding binding);
    [[nodiscard]] const CapabilityOverlayBinding* find(std::uint32_t layer,
                                                       CapabilityTarget target) const noexcept;

    // Cost accounting: multiply-accumulates the overlays added on top of the base path, plus the
    // resident bytes of the modules behind them. E1's declared failure mode is cost dominance,
    // so these counters are part of the contract, not telemetry garnish.
    void note_macs(std::uint64_t macs) const noexcept;
    [[nodiscard]] std::uint64_t applied_macs() const noexcept { return applied_macs_; }
    void set_module_bytes(ByteCount bytes) noexcept { module_bytes_ = bytes; }
    [[nodiscard]] ByteCount module_bytes() const noexcept { return module_bytes_; }
    void set_selected_name(std::string name) { selected_name_ = std::move(name); }
    [[nodiscard]] const std::string& selected_name() const noexcept { return selected_name_; }
    [[nodiscard]] std::size_t binding_count() const noexcept { return bindings_.size(); }

private:
    std::vector<CapabilityOverlayBinding> bindings_;
    mutable std::uint64_t applied_macs_{0};
    ByteCount module_bytes_{0};
    std::string selected_name_;
};

// The single injection kernel: adds one binding's low-rank contribution into an
// already-computed base output, so the frozen weights themselves are never rewritten.
// Both passes accumulate in double and the scale is applied once, matching the scalar
// reference matvec's accumulation discipline. `hidden_scratch` must hold at least `rank`
// elements; anything less is refused rather than silently truncated.
[[nodiscard]] Result<void> apply_capability_delta(const CapabilityOverlayBinding& binding,
                                                  std::span<const float> input,
                                                  std::span<float> hidden_scratch,
                                                  std::span<float> output);

// One supervised position for LM-head distillation: the frozen base's final-norm output, the
// base logits it produced (the softmax denominator needs them), and the teacher-derived token
// this position should predict.
struct CapabilityFitPair {
    std::vector<float> hidden;
    std::vector<float> base_logits;
    std::uint32_t target_id{0};
};

struct CapabilityFitResult {
    std::vector<float> a;  // rank x hidden, row-major
    std::vector<float> b;  // vocab x rank, row-major
    double initial_loss{0.0};
    double final_loss{0.0};
    std::size_t positions{0};
};
// Closed-form-gradient behavioural cloning of an LM-head low-rank adapter. Because the adapter
// sits at the network's last linear layer, every gradient is exact small-matrix algebra - no
// backpropagation engine exists anywhere in this project, by design. B starts at zero, so the
// very first step's loss equals the frozen base's cross-entropy: parity before capability.
//
// `regularization` is decoupled weight decay, AdamW-style: it never enters the gradient
// (an additive L2 term resonated through u and diverged - measured, not guessed) but ends
// every step with a multiplicative contraction clamped at zero, `max(0, 1 - lr*reg)`. No
// hyperparameter pair can overshoot, and an enormous weight simply collapses the adapter
// back to the frozen base - bounding off-skill interference by construction.
[[nodiscard]] Result<CapabilityFitResult> fit_lm_head_lora(
    std::size_t vocab_size, std::size_t hidden_size, std::size_t rank, std::size_t steps,
    double learning_rate, double regularization, std::uint64_t seed,
    const std::vector<CapabilityFitPair>& pairs);

}  // namespace vastloom::inference
