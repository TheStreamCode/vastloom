#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::model {
class LoadedExpert;
class LoadedTensorTile;
class ResidentByteBuffer;
struct LogicalTensorCatalog;
}  // namespace vastloom::model

namespace vastloom::safety {
class ResourceGovernor;
}  // namespace vastloom::safety

namespace vastloom::safety {
class ProcessSafetyGuard;
}  // namespace vastloom::safety

namespace vastloom::inference {

class DeepseekV4LoadedTrunkTensorSet;
class DeepseekV4LoadedGlobalTensorSet;
struct DeepseekV4GgufCatalogReport;

struct DeepseekV4RouteResult {
    std::vector<float> weights;
    std::vector<std::uint32_t> indices;
    std::size_t token_count{0};
    std::size_t top_k{0};
};

struct DeepseekV4HyperConnectionSplit {
    std::vector<float> pre;
    std::vector<float> post;
    std::vector<float> combination;
    std::size_t token_count{0};
    std::size_t multiplier{0};
};

struct DeepseekV4HyperConnectionPreResult {
    std::vector<float> reduced;
    DeepseekV4HyperConnectionSplit split;
    std::size_t width{0};
};

struct DeepseekV4ReferenceConfig {
    std::size_t vocabulary_size{0};
    std::size_t width{0};
    std::size_t intermediate_width{0};
    std::size_t layer_count{0};
    std::size_t hash_layer_count{0};
    std::size_t mtp_layer_count{0};
    std::size_t head_count{0};
    std::size_t expert_count{0};
    std::size_t experts_per_token{0};
    std::size_t query_lora_rank{0};
    std::size_t head_width{0};
    std::size_t rope_width{0};
    std::size_t output_groups{0};
    std::size_t output_lora_rank{0};
    std::size_t window_size{0};
    std::vector<std::size_t> compression_ratios;
    std::vector<std::size_t> mtp_compression_ratios;
    float compression_rope_theta{0.0F};
    std::size_t original_context{0};
    float rope_theta{0.0F};
    float rope_factor{0.0F};
    float beta_fast{0.0F};
    float beta_slow{0.0F};
    std::size_t index_head_count{0};
    std::size_t index_head_width{0};
    std::size_t index_top_k{0};
    std::size_t hyper_multiplier{0};
    std::size_t sinkhorn_iterations{0};
    float norm_epsilon{0.0F};
    float hyper_epsilon{0.0F};
    float route_scale{0.0F};
    // Empty schedules preserve the explicit scalar clamp used by reduced fixtures. Verified
    // GGUF configurations provide one routed/shared value per main or MTP block.
    std::vector<float> routed_expert_swiglu_limits;
    std::vector<float> shared_expert_swiglu_limits;
    float swiglu_limit{0.0F};
    // Worker threads the packed matvecs may use. Rows are independent and each is computed exactly
    // as it would be serially, so this changes speed and nothing else. One keeps the reference
    // behaviour, which is what every fixture test relies on.
    unsigned cpu_threads{1U};
};

struct DeepseekV4PackedMatrix {
    DeepseekV4PackedMatrix() = default;
    DeepseekV4PackedMatrix(std::uint32_t type,
                           std::size_t output_rows,
                           std::size_t input_columns,
                           std::size_t physical_columns,
                           std::vector<std::byte> bytes)
        : ggml_type(type),
          rows(output_rows),
          logical_columns(input_columns),
          storage_columns(physical_columns),
          encoded(std::move(bytes)) {}

    std::uint32_t ggml_type{0U};
    std::size_t rows{0U};
    std::size_t logical_columns{0U};
    std::size_t storage_columns{0U};
    std::vector<std::byte> encoded;
    std::shared_ptr<const model::ResidentByteBuffer> resident;
    std::size_t resident_offset{0U};
    std::size_t resident_bytes{0U};
    // Stable name for these exact weights, independent of where they currently live in memory.
    //
    // An accelerator that keeps a device-side copy needs to know when two descriptors mean the same
    // matrix. The host address cannot answer that: expert buffers are released as the forward pass
    // advances, and the allocator reuses their addresses, so an address-keyed cache silently serves
    // one expert's weights under another's name. Zero means no identity was supplied, which callers
    // must read as "do not cache" rather than "cache by address".
    std::uint64_t cache_identity{0U};
    // Whether the next forward pass will ask for these exact weights again.
    //
    // Always-active weights will: every token multiplies by every one of them. A routed expert will
    // not, with high probability - top-6 of 256 means consecutive tokens pick different ones, and
    // tripling the expert cache was measured to change the bytes read by nothing. A cache that
    // cannot tell them apart lets the second kind evict the first, which is what kept every layer
    // trunk being re-read on every pass regardless of how much VRAM was available.
    bool recurring{false};
};

// Returns the exact encoded view owned by this descriptor. The matrix must outlive the returned
// span; resident-backed descriptors retain their immutable payload through a shared lease.
[[nodiscard]] Result<std::span<const std::byte>> deepseek_v4_packed_matrix_payload(
    const DeepseekV4PackedMatrix& matrix,
    std::string_view identity);

struct DeepseekV4PackedIntegerTensor {
    DeepseekV4PackedIntegerTensor() = default;
    DeepseekV4PackedIntegerTensor(std::size_t element_count,
                                  std::vector<std::byte> bytes)
        : elements(element_count), encoded(std::move(bytes)) {}

    std::size_t elements{0U};
    std::vector<std::byte> encoded;
    std::shared_ptr<const model::ResidentByteBuffer> resident;
    std::size_t resident_offset{0U};
    std::size_t resident_bytes{0U};
};

struct DeepseekV4PackedFloatTensor {
    std::uint32_t ggml_type{0U};
    std::size_t elements{0U};
    std::vector<std::byte> encoded;
    std::shared_ptr<const model::ResidentByteBuffer> resident;
    std::size_t resident_offset{0U};
    std::size_t resident_bytes{0U};
};

struct DeepseekV4ResolvedExpert {
    std::uint32_t expert_id{0U};
    DeepseekV4PackedMatrix w1;
    DeepseekV4PackedMatrix w2;
    DeepseekV4PackedMatrix w3;
    std::uint64_t resident_bytes{0U};
    std::uint64_t logical_bytes_read{0U};
    bool cache_hit{false};
    bool asynchronous_load{false};
    std::size_t load_request_sequence{0U};
    double load_queue_ms{0.0};
    double load_service_ms{0.0};
    double load_total_ms{0.0};
    std::chrono::steady_clock::time_point load_service_started_at;
    std::chrono::steady_clock::time_point load_completed_at;
};

class DeepseekV4ExpertBatch {
public:
    DeepseekV4ExpertBatch() = default;
    DeepseekV4ExpertBatch(const DeepseekV4ExpertBatch&) = delete;
    DeepseekV4ExpertBatch& operator=(const DeepseekV4ExpertBatch&) = delete;
    virtual ~DeepseekV4ExpertBatch() = default;

    [[nodiscard]] virtual Result<DeepseekV4ResolvedExpert> acquire(
        std::uint32_t expert_id) = 0;
};

using DeepseekV4ExpertProvider = std::function<Result<std::vector<DeepseekV4ResolvedExpert>>(
    std::string_view block_prefix,
    std::span<const std::uint32_t> unique_expert_ids)>;

using DeepseekV4ExpertBatchProvider =
    std::function<Result<std::unique_ptr<DeepseekV4ExpertBatch>>(
        std::string_view block_prefix,
        std::span<const std::uint32_t> unique_expert_ids)>;

struct DeepseekV4ExpertProviderBinding {
    DeepseekV4ExpertProviderBinding() = default;
    DeepseekV4ExpertProviderBinding(std::size_t request_limit,
                                    DeepseekV4ExpertProvider resolver)
        : max_experts_per_request(request_limit), resolve(std::move(resolver)) {}

    std::size_t max_experts_per_request{0U};
    DeepseekV4ExpertProvider resolve;
    DeepseekV4ExpertBatchProvider begin;
};

// Bind exact complete tensors from bounded file loaders without copying their payload. The
// returned descriptor retains the loader's ResidencyUse through a shared immutable buffer.
[[nodiscard]] Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const model::LoadedTensorTile& tile,
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns);

[[nodiscard]] Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const model::LoadedExpert& expert,
    std::size_t catalog_tensor_index,
    std::uint32_t ggml_type,
    std::size_t rows,
    std::size_t logical_columns,
    std::size_t storage_columns,
    // Expert buffers are released and reallocated as the pass advances, so the caller - which knows
    // the layer and expert this bundle came from - must supply the identity. Leaving it zero is a
    // valid choice that costs acceleration, never correctness.
    std::uint64_t cache_identity = 0U);

[[nodiscard]] Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const model::LoadedTensorTile& tile,
    std::size_t elements);

[[nodiscard]] Result<DeepseekV4PackedFloatTensor> deepseek_v4_bind_packed_float(
    const model::LoadedTensorTile& tile,
    std::uint32_t ggml_type,
    std::size_t elements);

[[nodiscard]] Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const model::LoadedExpert& expert,
    std::size_t catalog_tensor_index,
    std::size_t elements);

// Infers the physical matrix shape/type from a verified complete trunk tensor set while the
// caller supplies the unpadded logical input width required by the architecture graph.
[[nodiscard]] Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    std::string_view name,
    std::size_t logical_columns);

[[nodiscard]] Result<DeepseekV4PackedIntegerTensor> deepseek_v4_bind_packed_i32(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    std::string_view name);

[[nodiscard]] Result<DeepseekV4PackedFloatTensor> deepseek_v4_bind_packed_float(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    std::string_view name);

[[nodiscard]] Result<DeepseekV4PackedMatrix> deepseek_v4_bind_packed_matrix(
    const DeepseekV4LoadedGlobalTensorSet& tensors,
    std::string_view name,
    std::size_t logical_columns);

[[nodiscard]] Result<DeepseekV4PackedFloatTensor> deepseek_v4_bind_packed_float(
    const DeepseekV4LoadedGlobalTensorSet& tensors,
    std::string_view name);

// Explicit canonical-name adapter for the 21 always-active tensors in layer 0 of the pinned
// DeepSeek V4 Flash 0731 GGUF graph. It validates every logical consumer shape and returns a
// detached binding set; callers decide when to publish it into a reference graph.
struct DeepseekV4ReferenceLayerTrunk {
    std::size_t layer{0U};
    std::size_t tensor_count{0U};
    std::map<std::string, DeepseekV4PackedMatrix, std::less<>> matrices;
    std::map<std::string, DeepseekV4PackedIntegerTensor, std::less<>> integer_tensors;
    std::map<std::string, DeepseekV4PackedFloatTensor, std::less<>> float_tensors;
};

// Where a layer's always-active tensors come from. One implementation reads a verified trunk tensor
// set held in a single artifact buffer; another streams each tensor straight out of the checkpoint.
// The layout that maps them onto graph consumers is identical either way.
struct DeepseekV4LayerTrunkSource {
    std::function<Result<DeepseekV4PackedMatrix>(std::string_view, std::size_t)> matrix;
    std::function<Result<DeepseekV4PackedFloatTensor>(std::string_view)> vector;
    std::function<Result<DeepseekV4PackedIntegerTensor>(std::string_view, std::size_t)> table;
};

// One always-active tensor: the checkpoint object it comes from, the consumer that binds it and the
// shape that consumer requires.
//
// This used to exist only as control flow inside the binder loop, which meant the sole way to obtain
// one matrix was to bind all of them. A window that streams trunks pays that as a whole-layer read
// for a single missing matrix, so the mapping is data now and both callers walk the same list.
struct DeepseekV4TrunkTensorMapping {
    enum class Kind : std::uint8_t { matrix, vector, table };

    Kind kind{Kind::matrix};
    std::string source;
    std::string consumer;
    // Matrices: output rows and logical input columns. Vectors and tables: elements only.
    std::size_t rows{0U};
    std::size_t elements{0U};
};

// Every always-active tensor of one layer, in binding order, derived from metadata alone.
[[nodiscard]] Result<std::vector<DeepseekV4TrunkTensorMapping>> deepseek_v4_layer_trunk_mappings(
    const DeepseekV4ReferenceConfig& config,
    std::size_t layer);

[[nodiscard]] Result<DeepseekV4ReferenceLayerTrunk> deepseek_v4_build_layer_trunk(
    const DeepseekV4LayerTrunkSource& source,
    const DeepseekV4ReferenceConfig& config,
    std::size_t layer);

[[nodiscard]] Result<DeepseekV4ReferenceLayerTrunk>
deepseek_v4_bind_0731_gguf_layer0_trunk(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    const DeepseekV4ReferenceConfig& config);

// The same adapter for any layer of the pinned checkpoint. Layer 0 has 21 always-active tensors;
// deeper layers add a learned-router bias instead of the hash table, an attention compressor when
// their compression ratio is non-zero, and an indexer as well at ratio 4. Every loaded tensor must
// reach a consumer, so a layer whose set does not match its declared layout is refused rather than
// partially bound.
[[nodiscard]] Result<DeepseekV4ReferenceLayerTrunk>
deepseek_v4_bind_0731_gguf_layer_trunk(
    const DeepseekV4LoadedTrunkTensorSet& tensors,
    const DeepseekV4ReferenceConfig& config,
    std::size_t layer);

// Streams one layer's always-active tensors straight from a checkpoint whose payload is present,
// one complete tile per tensor under its own governed reservation. This is the path a full-depth
// forward pass uses: no intermediate artifact, and only the layers actually reached are resident.
[[nodiscard]] Result<DeepseekV4ReferenceLayerTrunk>
deepseek_v4_load_0731_gguf_layer_trunk(
    const model::LogicalTensorCatalog& catalog,
    const DeepseekV4ReferenceConfig& config,
    std::size_t layer,
    safety::ResourceGovernor& governor);

// One always-active matrix's shape, type and stable identity from metadata alone, without reading a
// byte of it. Enough to index a layer and to recognise weights an accelerator already holds, which
// is what lets a pass start without reading the model to learn its own shape.
[[nodiscard]] Result<DeepseekV4PackedMatrix> deepseek_v4_describe_0731_gguf_trunk_matrix(
    const model::LogicalTensorCatalog& catalog,
    std::string_view source,
    std::size_t layer,
    std::size_t logical_columns);

// A layer's small always-active tensors: normalization vectors, router biases and the hash routing
// table. These stay resident for the whole pass - they are what the graph consults to decide shapes
// and routes, and together they are a rounding error against the matrices.
[[nodiscard]] Result<DeepseekV4PackedFloatTensor> deepseek_v4_load_0731_gguf_trunk_vector(
    const model::LogicalTensorCatalog& catalog,
    std::string_view source,
    std::size_t layer,
    safety::ResourceGovernor& governor);

[[nodiscard]] Result<DeepseekV4PackedIntegerTensor> deepseek_v4_load_0731_gguf_trunk_table(
    const model::LogicalTensorCatalog& catalog,
    std::string_view source,
    std::size_t layer,
    std::size_t elements,
    safety::ResourceGovernor& governor);

struct DeepseekV4ReferenceGlobals {
    std::map<std::string, DeepseekV4PackedMatrix, std::less<>> matrices;
    std::map<std::string, DeepseekV4PackedFloatTensor, std::less<>> float_tensors;
};

// Maps the six canonical 0731 global tensors to embed/head/norm/Hyper-Connection consumers only
// after their complete closed set has reached RAM_READY under one governed reservation.
[[nodiscard]] Result<DeepseekV4ReferenceGlobals>
deepseek_v4_bind_0731_gguf_globals(
    const DeepseekV4LoadedGlobalTensorSet& tensors,
    const DeepseekV4ReferenceConfig& config);

// The same six consumers, streamed straight from a checkpoint whose payload is present.
[[nodiscard]] Result<DeepseekV4ReferenceGlobals>
deepseek_v4_load_0731_gguf_globals(
    const model::LogicalTensorCatalog& catalog,
    const DeepseekV4ReferenceConfig& config,
    safety::ResourceGovernor& governor);

// Converts only a complete verified 0731 architecture snapshot. No tensor payload is read.
[[nodiscard]] Result<DeepseekV4ReferenceConfig>
deepseek_v4_reference_config_from_0731_gguf(
    const DeepseekV4GgufCatalogReport& report);

// Validates the exact 21 canonical layer-0 trunk descriptors against their native consumers
// without allocating or reading any tensor payload.
[[nodiscard]] Result<void> deepseek_v4_validate_0731_gguf_layer0_catalog(
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4ReferenceConfig& config);

struct DeepseekV4ReferenceCatalogValidation {
    std::size_t layers{0U};
    std::size_t tensors{0U};
    std::uint64_t bytes{0U};
};

// Validates every canonical main-layer non-routed descriptor against the native reference
// consumers without allocating or reading tensor payload.
[[nodiscard]] Result<DeepseekV4ReferenceCatalogValidation>
deepseek_v4_validate_0731_gguf_layer_catalogs(
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4ReferenceConfig& config);

struct DeepseekV4ReferenceCheckpointCatalogValidation {
    DeepseekV4ReferenceCatalogValidation layers;
    std::size_t global_tensors{0U};
    std::uint64_t global_bytes{0U};
    std::size_t expert_bundles{0U};
    std::size_t expert_tensor_slices{0U};
    std::size_t physical_expert_tensors{0U};
    std::size_t logical_tensors{0U};
    std::uint64_t tensor_bytes{0U};
};

// Closes the complete pinned GGUF catalog across main-layer trunks, six global consumers and
// every logical routed-expert slice. No tensor payload is allocated or read.
[[nodiscard]] Result<DeepseekV4ReferenceCheckpointCatalogValidation>
deepseek_v4_validate_0731_gguf_checkpoint_catalog(
    const DeepseekV4GgufCatalogReport& report,
    const DeepseekV4ReferenceConfig& config);

// Optional device-side matvec. The graph knows nothing about CUDA; it asks whether a given packed
// matrix can be multiplied elsewhere, and falls back to the CPU kernel whenever the answer is no.
// An accelerator that declines must leave the output untouched.
class DeepseekV4MatvecAccelerator {
public:
    DeepseekV4MatvecAccelerator() = default;
    DeepseekV4MatvecAccelerator(const DeepseekV4MatvecAccelerator&) = delete;
    DeepseekV4MatvecAccelerator& operator=(const DeepseekV4MatvecAccelerator&) = delete;
    virtual ~DeepseekV4MatvecAccelerator() = default;

    [[nodiscard]] virtual Result<bool> matvec(
        const DeepseekV4PackedMatrix& matrix,
        std::span<const float> input,
        std::span<float> output) = 0;
};

struct DeepseekV4ReferenceGraph {
    DeepseekV4ReferenceConfig config;
    DeepseekV4MatvecAccelerator* accelerator{nullptr};
    std::map<std::string, std::vector<float>, std::less<>> float_tensors;
    std::map<std::string, std::vector<std::uint32_t>, std::less<>> integer_tensors;
    std::map<std::string, DeepseekV4PackedMatrix, std::less<>> packed_matrices;
    std::map<std::string, DeepseekV4PackedIntegerTensor, std::less<>> packed_integer_tensors;
    std::map<std::string, DeepseekV4PackedFloatTensor, std::less<>> packed_float_tensors;
    std::map<std::string, DeepseekV4ExpertProviderBinding, std::less<>> expert_providers;

    // Optional source for matrices that are not held in `packed_matrices`.
    //
    // Holding every layer trunk for the whole run is what sets the host memory floor - 6.39 GB of
    // the 7.62 GB a full-depth pass needs - and that floor is what decides whether the model runs
    // on a given machine at all, and how much memory is left for the accelerator. A provider lets
    // the caller keep a bounded window of layers resident instead, the way expert caches already
    // work. Returning a null pointer means "I do not have that name", which is not an error.
    //
    // The returned matrix must stay valid until the next provider call for a different layer.
    std::function<Result<const DeepseekV4PackedMatrix*>(std::string_view)> matrix_provider;
    // Names the provider can supply. The graph chooses architecture paths by asking whether a
    // tensor exists, so that question has to be answerable without paying to load it.
    std::set<std::string, std::less<>> provided_matrix_names;
};

struct DeepseekV4ReferenceCompressorState {
    std::vector<float> key_values;
    std::vector<float> scores;
    std::vector<float> compressed;
};

struct DeepseekV4ReferenceAttentionState {
    std::vector<float> window_key_values;
    DeepseekV4ReferenceCompressorState compressor;
    DeepseekV4ReferenceCompressorState indexer;
};

struct DeepseekV4ReferenceState {
    std::size_t next_position{0};
    std::vector<DeepseekV4ReferenceAttentionState> layers;
    std::vector<DeepseekV4ReferenceAttentionState> mtp_layers;
};

struct DeepseekV4LayerTrace {
    struct ExpertAccess {
        std::uint32_t expert_id{0U};
        std::uint64_t resident_bytes{0U};
        std::uint64_t logical_bytes_read{0U};
        bool cache_hit{false};
        bool asynchronous_load{false};
        std::size_t load_request_sequence{0U};
        double load_queue_ms{0.0};
        double load_service_ms{0.0};
        double load_total_ms{0.0};
        double acquire_wait_ms{0.0};
        double compute_ms{0.0};
        double io_compute_overlap_ms{0.0};
    };

    std::size_t layer{0};
    DeepseekV4RouteResult route;
    std::vector<ExpertAccess> expert_accesses;
    std::vector<std::int32_t> attention_indices;
    std::size_t attention_selected_per_token{0};
};

struct DeepseekV4ReferenceForwardResult {
    std::vector<float> logits;
    std::vector<DeepseekV4LayerTrace> layers;
    std::uint32_t greedy_token_id{0};
    std::vector<float> mtp_logits;
    std::vector<DeepseekV4LayerTrace> mtp_layers;
    std::uint32_t mtp_greedy_token_id{0};
};

inline constexpr std::size_t kDeepseekV4ReferenceMaxTokens = 4'096U;

// Optional transaction checkpoint invoked before expensive graph phases and each main/MTP
// block. A failed checkpoint aborts without publishing the caller's incremental state.
struct DeepseekV4ReferenceExecutionControl {
    std::function<Result<void>()> checkpoint;
};

enum class DeepseekV4ReferenceFinishReason {
    stop_token,
    context_limit,
    max_tokens,
    cancelled,
    timeout,
};

struct DeepseekV4ReferenceGenerationOptions {
    std::size_t max_new_tokens{64U};
    std::size_t context_limit_tokens{kDeepseekV4ReferenceMaxTokens};
    float temperature{0.0F};
    std::size_t top_k{40U};
    std::uint64_t seed{1U};
    std::chrono::milliseconds timeout{std::chrono::minutes(5)};
    std::vector<std::uint32_t> stop_token_ids;
    std::function<bool()> cancelled;
    std::function<void(std::uint32_t)> on_token;
};

struct DeepseekV4ReferenceGenerationStats {
    std::size_t prompt_tokens{0U};
    std::size_t generated_tokens{0U};
    std::size_t requested_max_tokens{0U};
    std::size_t effective_max_tokens{0U};
    std::size_t forward_passes{0U};
    std::size_t expert_requests{0U};
    std::size_t expert_cache_hits{0U};
    std::size_t expert_cache_misses{0U};
    std::uint64_t resident_expert_bytes{0U};
    std::uint64_t logical_expert_bytes_read{0U};
    double prompt_eval_ms{0.0};
    double generation_ms{0.0};
    double tokens_per_second{0.0};
    double expert_load_service_ms{0.0};
    double expert_compute_ms{0.0};
    double io_compute_overlap_ms{0.0};
};

struct DeepseekV4ReferenceGenerationResult {
    std::vector<std::uint32_t> prompt_ids;
    std::vector<std::uint32_t> generated_ids;
    DeepseekV4ReferenceFinishReason finish_reason{
        DeepseekV4ReferenceFinishReason::max_tokens};
    std::optional<std::uint32_t> stop_token_id;
    DeepseekV4ReferenceGenerationStats stats;
};

[[nodiscard]] Result<std::vector<float>> deepseek_v4_rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t row_count,
    std::size_t width,
    float epsilon);

[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_learned(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const float> selection_bias,
    std::size_t token_count,
    std::size_t width,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_hash(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> token_to_experts,
    std::size_t token_count,
    std::size_t width,
    std::size_t vocabulary_size,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

// These entry points accept the raw router projection before sqrt-softplus. They let the
// architecture graph consume a packed GGML matrix without materializing a float weight matrix.
[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_learned_logits(
    std::span<const float> logits,
    std::span<const float> selection_bias,
    std::size_t token_count,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

[[nodiscard]] Result<DeepseekV4RouteResult> deepseek_v4_route_hash_logits(
    std::span<const float> logits,
    std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> token_to_experts,
    std::size_t token_count,
    std::size_t vocabulary_size,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_expert_swiglu(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const float> down_weight,
    std::span<const float> up_weight,
    std::span<const float> routing_weights,
    std::size_t token_count,
    std::size_t width,
    std::size_t intermediate_width,
    float swiglu_limit);

[[nodiscard]] Result<DeepseekV4HyperConnectionSplit> deepseek_v4_hc_split_sinkhorn(
    std::span<const float> mixes,
    std::span<const float> scale,
    std::span<const float> base,
    std::size_t token_count,
    std::size_t multiplier,
    std::size_t sinkhorn_iterations,
    float epsilon);

[[nodiscard]] Result<DeepseekV4HyperConnectionPreResult> deepseek_v4_hc_pre(
    std::span<const float> input,
    std::span<const float> mixing_weight,
    std::span<const float> scale,
    std::span<const float> base,
    std::size_t token_count,
    std::size_t multiplier,
    std::size_t width,
    std::size_t sinkhorn_iterations,
    float norm_epsilon,
    float hc_epsilon);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_hc_post(
    std::span<const float> sublayer,
    std::span<const float> residual,
    const DeepseekV4HyperConnectionSplit& split,
    std::size_t width);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_apply_rope(
    std::span<const float> input,
    std::span<const std::uint32_t> positions,
    std::size_t row_count,
    std::size_t head_count,
    std::size_t head_width,
    float theta,
    bool inverse = false);

[[nodiscard]] Result<std::vector<float>> deepseek_v4_sparse_attention(
    std::span<const float> query,
    std::span<const float> key_value,
    std::span<const float> attention_sink,
    std::span<const std::int32_t> selected_indices,
    std::size_t query_count,
    std::size_t key_value_count,
    std::size_t head_count,
    std::size_t head_width,
    std::size_t selected_per_query,
    float softmax_scale);

// Bounded single-batch oracle for reduced float fixtures. Passing a state captures the
// raw/compressed caches needed by deepseek_v4_reference_decode. Quantized checkpoint
// kernels remain outside this reference path.
[[nodiscard]] Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_prefill(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> token_ids,
    DeepseekV4ReferenceState* state = nullptr,
    const DeepseekV4ReferenceExecutionControl* control = nullptr);

[[nodiscard]] Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_decode(
    const DeepseekV4ReferenceGraph& graph,
    std::uint32_t token_id,
    DeepseekV4ReferenceState& state,
    const DeepseekV4ReferenceExecutionControl* control = nullptr);

// Bounded token-level autoregressive driver for the native reference graph. It uses main-model
// logits for generation; MTP is evaluated only because it remains part of the pinned graph oracle.
// The optional ProcessSafetyGuard is sampled at every execution checkpoint.
[[nodiscard]] Result<DeepseekV4ReferenceGenerationResult>
deepseek_v4_reference_generate(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> prompt_ids,
    const DeepseekV4ReferenceGenerationOptions& options,
    const safety::ProcessSafetyGuard* process_guard = nullptr);

[[nodiscard]] std::string_view to_string(
    DeepseekV4ReferenceFinishReason reason) noexcept;

}  // namespace vastloom::inference
