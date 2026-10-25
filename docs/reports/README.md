# Validation reports

Reports are evidence snapshots tied to a date, host, source revision and project state. They are not
rolling claims: a later code change does not retroactively alter an earlier result.

Releases through `0.4.0-alpha.3` used the former project name, executable and persisted identifiers.
Those historical documents are intentionally preserved verbatim. Beginning with
`0.4.0-alpha.4`, current documentation and commands use **Vastloom**. Compatibility readers keep
the already-downloaded model stores and checkpoint manifests usable without mutation.

Start with the newest report for current claims, then consult milestone-specific reports for exact
hardware, commands, measurements and limitations.

## Current reports

- [`P0_EXPERT_PACK_IO_VALIDATION.md`](P0_EXPERT_PACK_IO_VALIDATION.md) —
  post-conclusion P0 experiment: a derived expert pack with one aligned read per expert and an
  opt-in buffered/bypass I/O mode on the chain and layer probes. Token 0 drops from 54 to 18 range
  reads, layer 0 from 18 to 6, and the five-token prompt from 264 to 88 with byte-identical
  logits/hidden; actual read-path counters and a direct-I/O proxy show no wall-time win, so no
  throughput or cold-cache claim. The chain also emits a bounded cache plan (`--cache-entries`) and
  a recording-only usage profile; the elevated cold-cache helper is not executed here. Three layers
  of 43; no full-model result.
- [`D3C_MULTITOKEN_ORACLE_VALIDATION.md`](D3C_MULTITOKEN_ORACLE_VALIDATION.md) —
  D3c: a five-token causal prompt closes the layer-2 4:1 compression window; the independent
  NumPy oracle reproduces the native 3-layer prefill (logits max abs 1.91e-06, hidden 5.96e-08,
  argmax 20,267, all 15 per-token routes) and reports one compressed row with six selected slots
  per position. Three layers of 43; no full-model claim.
- [`D3B_CHAIN_ORACLE_VALIDATION.md`](D3B_CHAIN_ORACLE_VALIDATION.md) —
  D3b: the independent NumPy oracle executes the 3-layer hash-routed chain (including the layer-2
  4:1 compressor/indexer) and matches the native chain probe at depths 1–3 for tokens 0 and 1;
  logits max abs 1.91e-06 and hidden max abs 3.81e-06, the same tolerances as D2b.2. Boundary:
  one prefill token closes no compression window, so the compressed-slot attention is
  synthetically, not natively, compared; three layers of 43.
- [`D2B_LAYER0_ORACLE_VALIDATION.md`](D2B_LAYER0_ORACLE_VALIDATION.md) —
  D2b.2: an independent NumPy implementation reads the same three range artifacts and reproduces
  the probe's layer-0 logits (max abs diff 9.54e-07, gate 1e-3) and final hidden state
  (4.77e-07), with the same argmax and route. Raw FNV differs by one float32 ulp and is reported
  as a diagnostic; layer 0 only, no full-model or throughput claim.
- [`D3_HASH_CHAIN_VALIDATION.md`](D3_HASH_CHAIN_VALIDATION.md) —
  the D3 executor evidence: `deepseek-chain-probe` executes 2 and 3 hash-routed layers from
  per-layer range artifacts, resolves each layer's route from its own `tid2eid`, loads trunks
  lazily and reports exact trunk/expert bytes. Deterministic; the independent-oracle comparison
  is in [`D3B_CHAIN_ORACLE_VALIDATION.md`](D3B_CHAIN_ORACLE_VALIDATION.md), and both opt-in CMake
  chain tests pass for the single-token scope.
- [`D2B_DEVICE_RESIDENCY_VALIDATION.md`](D2B_DEVICE_RESIDENCY_VALIDATION.md) —
  D2b.1: routed-expert matrices resident in VRAM under a fixed ceiling, with per-matrix CPU
  fallback; CPU and GPU runs produce bit-identical layer-0 logits; counters and the forced
  1 MiB fallback are reported. Not a VRAM payload tier and no oracle yet.
- [`D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md`](D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md) —
  the DeepSeek refocus D1/D2a evidence: pinned `UD-IQ2_XXS` metadata identities, a metadata-only
  store, exact range plans, and one real layer (layer 0 of 43) executed from 780,087,788 B of
  range payload with the checkpoint's own `tid2eid` route and six complete experts. Deterministic
  across runs; not the model's output, not a parity result, and no VRAM placement yet.
- [`0_5_0_ALPHA1_SERVE_SAFETY_VALIDATION.md`](0_5_0_ALPHA1_SERVE_SAFETY_VALIDATION.md) —
  the maintenance update: JSON-object constrained decoding and control-token rejection on the
  dense SmolLM2 path, UTF-8-safe SSE streaming, the two serve defects the new tests exposed
  (producer lifetime and an empty stop filter), Release/Debug gates and the bounded
  non-claims. No large-model record was re-measured.
- [`Q11C_QWEN38_LLAMA_PARITY_VALIDATION.md`](Q11C_QWEN38_LLAMA_PARITY_VALIDATION.md) —
  the comparison track's external validation: the native runtime's greedy generation matches
  llama.cpp token for token on the official Q8_0 `qwen4exp` checkpoint, and the six defects
  only the external oracle could find.
- [`Q11B_QWEN38_REAL_CATALOG_VALIDATION.md`](Q11B_QWEN38_REAL_CATALOG_VALIDATION.md) —
  the real 125B FP8 catalog validated end-to-end in native C++.

## Earlier milestone reports

- [`M9_DEEPSEEK_CUDA_EXPERT_KERNEL_VALIDATION.md`](M9_DEEPSEEK_CUDA_EXPERT_KERNEL_VALIDATION.md) —
  every storage type on the routed path given a device kernel: 5.60× on a 16-layer generation with
  identical text and identical expert traffic, all three matvec gates at zero error, and the two
  defects a passing gate was hiding — a VRAM cache keyed on recycled host addresses, and parity gates
  built from dyadic values that made f32 and f64 indistinguishable.
- [`M9_DEEPSEEK_COMPLETE_FORWARD_VALIDATION.md`](M9_DEEPSEEK_COMPLETE_FORWARD_VALIDATION.md) —
  the complete 43-layer model executed on the reference PC: 258 expert bundles, 1,770,651,648 expert
  bytes matching the metadata prediction exactly, 129,280 logits, and host memory that never
  approached its floor. One forward pass, not generation.
- [`M9_DEEPSEEK_OFFICIAL_LAYER_EXECUTION_VALIDATION.md`](M9_DEEPSEEK_OFFICIAL_LAYER_EXECUTION_VALIDATION.md) —
  first real DeepSeek layer executed from official payload: catalog relocation onto imported
  artifacts, eight quantizations running together, the checkpoint's own router choosing six experts,
  and deterministic layer-0 logits. One layer of 43, not the model's output.
- [`M9_DEEPSEEK_AUTHORITATIVE_ROUTING_VALIDATION.md`](M9_DEEPSEEK_AUTHORITATIVE_ROUTING_VALIDATION.md) —
  first official DeepSeek payload import: the layer-0 trunk, the pinned `tid2eid` table read through
  the governed range path, one token's authoritative expert selection and the six complete bundles
  it names; nothing executed from that payload yet.
- [`M8_CUDA_RMSNORM_VALIDATION.md`](M8_CUDA_RMSNORM_VALIDATION.md) — first real toolkit-free CUDA
  data operation: governed packed VRAM, synchronous H2D/D2H, scalar PTX RMSNorm, CPU parity and
  transactional failure coverage on the RTX 4070 Super through Windows and WSL; still no CUDA
  model graph.
- [`M8_CUDA_DRIVER_VALIDATION.md`](M8_CUDA_DRIVER_VALIDATION.md) — native toolkit-free Windows/Linux
  CUDA capability discovery, Governor-first move-only VRAM ownership, deterministic allocation/
  cleanup faults and real 4 MiB RTX 4070 Super smoke allocations on Windows and WSL; no kernels or
  GPU inference.
- [`M9_DEEPSEEK_REFERENCE_GENERATION_VALIDATION.md`](M9_DEEPSEEK_REFERENCE_GENERATION_VALIDATION.md) —
  exact eight-token greedy parity with a freshly regenerated pinned preview graph, portable seeded
  top-k sampling, transactional per-layer safety checkpoints and exact generation-level
  expert-cache telemetry on the reduced graph; no official GGUF payload was executed.
- [`M9_DEEPSEEK_GLOBAL_TENSOR_SET_VALIDATION.md`](M9_DEEPSEEK_GLOBAL_TENSOR_SET_VALIDATION.md) —
  exact 596,000,788-byte six-global plan from verified official metadata, dedicated immutable
  range-artifact scope, one aggregate governed fixture buffer/lease and embedding/head/norm/
  Hyper-Connection prefill/decode parity; the official payload passed dry-run only.
- [`M9_DEEPSEEK_MXFP4_EXPERT_KERNEL_VALIDATION.md`](M9_DEEPSEEK_MXFP4_EXPERT_KERNEL_VALIDATION.md) —
  pinned-ggml MXFP4 E8M0/E2M1 scalar dequantization/matvec parity, bounded complete-matrix row
  lookup and alternate reduced IQ1_S-gate/up plus MXFP4-down prefill/decode graph parity, with no
  official MXFP4 payload acquisition.
- [`M9_DEEPSEEK_IQ2XXS_EXPERT_KERNEL_VALIDATION.md`](M9_DEEPSEEK_IQ2XXS_EXPERT_KERNEL_VALIDATION.md) —
  pinned-ggml IQ2_XXS scalar dequantization/matvec parity, bounded complete-matrix row lookup and
  alternate reduced IQ2_XXS-gate/up plus IQ3_XXS-down prefill/decode graph parity, with no
  official IQ2_XXS payload acquisition.
- [`M9_DEEPSEEK_Q4K_GLOBAL_KERNEL_VALIDATION.md`](M9_DEEPSEEK_Q4K_GLOBAL_KERNEL_VALIDATION.md) —
  pinned-ggml Q4_K scalar dequantization/matvec parity, bounded complete-matrix single-row lookup
  and reduced packed embedding/output prefill plus incremental graph parity, with no official
  payload acquisition.
- [`M9_DEEPSEEK_CHECKPOINT_CATALOG_VALIDATION.md`](M9_DEEPSEEK_CHECKPOINT_CATALOG_VALIDATION.md) —
  zero-payload closure of all six global consumers, 11,008 expert bundles, 33,024 routed slices,
  1,328 physical tensors, 34,223 logical descriptors and 82,533,892,444 known tensor bytes, with
  explicit layout-only Q4_K/IQ2_XXS/MXFP4 boundaries.
- [`M9_DEEPSEEK_ALL_LAYER_CATALOG_VALIDATION.md`](M9_DEEPSEEK_ALL_LAYER_CATALOG_VALIDATION.md) —
  zero-payload validation of all 43 official main-layer layouts, 1,193 non-routed descriptors and
  6,390,088,008 encoded bytes against native uncompressed/hash, 4:1 compressor/indexer and 128:1
  compressor consumers.
- [`M9_DEEPSEEK_METADATA_TO_GRAPH_VALIDATION.md`](M9_DEEPSEEK_METADATA_TO_GRAPH_VALIDATION.md) —
  exact pinned 0731 GGUF metadata to native reference-config construction, bounded retention of
  the compression and distinct routed/shared SwiGLU schedules, and zero-payload validation of all
  21 official layer-0 non-routed descriptors against their graph consumers.
- [`M9_DEEPSEEK_LAYER0_TRUNK_BINDING_VALIDATION.md`](M9_DEEPSEEK_LAYER0_TRUNK_BINDING_VALIDATION.md) —
  retained F32/BF16 vector consumers, an explicit 21-name canonical 0731 layer-0 adapter, one
  2,844-byte aggregate trunk lease and unchanged reduced mixed-format prefill/decode parity.
  Official metadata was revalidated, but no official trunk payload was acquired or executed.
- [`M9_DEEPSEEK_TRUNK_TENSOR_SET_VALIDATION.md`](M9_DEEPSEEK_TRUNK_TENSOR_SET_VALIDATION.md) —
  transactional aggregate multi-range loading, complete non-routed catalog coverage, one retained
  2,644-byte artifact buffer and full reduced mixed-format prefill/decode parity, with missing,
  duplicate, split, wrong-type, corruption, cancellation, budget and lease-cleanup faults. No
  official trunk byte was acquired or executed.
- [`M9_DEEPSEEK_BOUND_TRUNK_LOAD_VALIDATION.md`](M9_DEEPSEEK_BOUND_TRUNK_LOAD_VALIDATION.md) —
  one exact artifact-bound F32 fixture range loaded through aggregate-budgeted RAM/staging,
  rehashed in memory and retained through a scalar matvec, with allocation, cancellation,
  corruption, truncation and lease-cleanup faults while the complete artifact stays non-executing.
- [`M9_DEEPSEEK_RANGE_ARTIFACT_BINDING_VALIDATION.md`](M9_DEEPSEEK_RANGE_ARTIFACT_BINDING_VALIDATION.md) —
  closed-world native artifact verification, full local range rehashing and exact trunk binding to
  regenerated pinned GGUF catalog identities, validated with synthetic corruption and forgery
  cases while remaining explicitly non-executing.
- [`M9_DEEPSEEK_RANGE_ARTIFACT_IMPORT_VALIDATION.md`](M9_DEEPSEEK_RANGE_ARTIFACT_IMPORT_VALIDATION.md) —
  explicit-authority bounded HTTP range import, strict plan/source validation, per-range hashes,
  crash cleanup and read-only atomic publication, plus a real 138,867,160-byte trunk dry-run with
  no official payload download.
- [`M9_DEEPSEEK_LAYER_PROBE_PLAN_VALIDATION.md`](M9_DEEPSEEK_LAYER_PROBE_PLAN_VALIDATION.md) —
  exact non-executing layer-0 trunk+top-6 range planning from verified real metadata,
  177,795,544 logical bytes across 39 immutable source ranges, strict caller-ID/routing-authority
  disclosure and cross-platform malformed-input/sanitizer evidence without a payload read.
- [`M9_DEEPSEEK_MIXED_FORMAT_LAYER_VALIDATION.md`](M9_DEEPSEEK_MIXED_FORMAT_LAYER_VALIDATION.md) —
  one complete reduced DeepSeek main-layer/token composition with Q8_0/I32 trunk data,
  IQ1_S W1/W3 and IQ3_XXS W2 expert bundles, governed synchronous/asynchronous file-backed
  residency, deterministic FP32 parity and transactional selected-expert failure coverage.
- [`M9_DEEPSEEK_ASYNC_COMPUTE_OVERLAP_VALIDATION.md`](M9_DEEPSEEK_ASYNC_COMPUTE_OVERLAP_VALIDATION.md) —
  bounded selected-miss begin/acquire batches, deterministic reduction order, controlled positive
  read-service/CPU-compute overlap, real-expert async ownership, abandoned-load accounting and
  cross-platform sanitizer/compiler evidence without a physical-NVMe speed claim.
- [`M9_DEEPSEEK_REAL_SELECTED_EXPERT_VALIDATION.md`](M9_DEEPSEEK_REAL_SELECTED_EXPERT_VALIDATION.md) —
  canonical 0731 W1/W2/W3 catalog binding, one complete 6,488,064-byte real mixed-format expert,
  governed cache miss/hit semantics and bit-identical ggml output.
- [`M9_DEEPSEEK_SELECTED_EXPERT_STREAMING_VALIDATION.md`](M9_DEEPSEEK_SELECTED_EXPERT_STREAMING_VALIDATION.md) —
  router-first selected-bundle acquisition, bounded synchronous RAM LRU behavior, exact logical
  byte/hit/eviction telemetry and unchanged reduced packed-layer parity.
- [`M9_DEEPSEEK_GOVERNED_LAYER_VALIDATION.md`](M9_DEEPSEEK_GOVERNED_LAYER_VALIDATION.md) —
  ownership-preserving resident buffers, 12 file-backed trunk tiles, four indivisible expert
  bundles, exact 5,364-byte Governor accounting, eviction interlocks, failure rollback and full
  reduced prefill/incremental parity without payload copies into the graph.
- [`M9_DEEPSEEK_PACKED_LAYER_VALIDATION.md`](M9_DEEPSEEK_PACKED_LAYER_VALIDATION.md) —
  ggml-verified deterministic Q8_0 encoding, all 23 matrices of one reduced DeepSeek main layer,
  I32 hash routing, full prefill/incremental graph parity and transactional malformed-input tests;
  that milestone deliberately left the fixture memory-owned, a boundary closed by the newer
  governed-layer report above.
- [`M9_DEEPSEEK_TRUNK_STORAGE_KERNEL_VALIDATION.md`](M9_DEEPSEEK_TRUNK_STORAGE_KERNEL_VALIDATION.md) —
  pinned-ggml Q8_0/Q5_K/Q6_K parity, bounded BF16/F32 matvec and I32 router-table handling,
  transactional output, and real metadata coverage for all 1,193 layer-trunk tensors while
  full payload and layer/token execution remain absent.
- [`M9_LAYER_TRUNK_STREAMING_VALIDATION.md`](M9_LAYER_TRUNK_STREAMING_VALIDATION.md) — exact
  `blk.N.*` identities, deterministic 64 MiB per-layer trunk plans, a governed single-tile loader,
  fault cleanup and real 43-layer 0731 metadata-plan results without downloading weight payloads.
- [`M7_DEEPSEEK_GGUF_CHECKPOINT_BINDING_VALIDATION.md`](M7_DEEPSEEK_GGUF_CHECKPOINT_BINDING_VALIDATION.md) —
  exact 0731 `UD-IQ1_S` object identities, bounded atomic metadata/full importer, immutable
  checkpoint binding, full-hash and single-tier admission gates, plus real 5.34 MB metadata
  validation without downloading the 82.54 GB payload.
- [`M7_DEEPSEEK_WORKING_SET_VALIDATION.md`](M7_DEEPSEEK_WORKING_SET_VALIDATION.md) —
  bounded compression-schedule retention, metadata-derived resident/streamed trunk floors,
  complete-expert single-tier admission and real 0731 `UD-IQ1_S` header results.
- [`M1_MODEL_ADMISSION_VALIDATION.md`](M1_MODEL_ADMISSION_VALIDATION.md) — native
  minimum-working-set and complete-expert RAM/VRAM capacity gate, aggregate host-memory
  governance, strict/legacy manifest behavior and reference-PC accept/reject evidence.
- [`M7_DEEPSEEK_REAL_EXPERT_VALIDATION.md`](M7_DEEPSEEK_REAL_EXPERT_VALIDATION.md) —
  one complete real DeepSeek 0731 gate/up IQ1_S plus down IQ3_XXS expert, bit-identical ggml
  parity, governed asynchronous RAM loading, fail-closed budget/lease cleanup and explicitly
  warm-buffered process-I/O measurements.
- [`M7_DEEPSEEK_IQ1S_KERNEL_VALIDATION.md`](M7_DEEPSEEK_IQ1S_KERNEL_VALIDATION.md) —
  exact IQ1_S dequantization/matvec parity on one real 1.64 MB DeepSeek 0731 expert-gate slice,
  including the explicit incomplete-expert/non-inference boundary.
- [`M7_DEEPSEEK_0731_GGUF_CATALOG_VALIDATION.md`](M7_DEEPSEEK_0731_GGUF_CATALOG_VALIDATION.md) —
  bounded native GGUF v3/split-header parsing, real `UD-IQ1_S` packed-expert mapping and the
  explicit metadata-only/non-execution boundary.
- [`M6_DEEPSEEK_METADATA_INSPECTION_VALIDATION.md`](M6_DEEPSEEK_METADATA_INSPECTION_VALIDATION.md) —
  exact official DeepSeek V4 Flash revision, bounded header-only import, native
  69,187-tensor inspection and the remaining non-execution boundary.
- [`M6_DEEPSEEK_TOKENIZER_VALIDATION.md`](M6_DEEPSEEK_TOKENIZER_VALIDATION.md) —
  native Unicode byte-BPE and basic chat-template parity against the pinned official
  tokenizer, including malformed-layout rejection and cross-platform sanitizer gates.
- [`M6_DEEPSEEK_REFERENCE_OPERATORS_VALIDATION.md`](M6_DEEPSEEK_REFERENCE_OPERATORS_VALIDATION.md) —
  bounded CPU equations and reproducible vectors for RMSNorm, routing, routed SwiGLU,
  Hyper-Connections, base RoPE and selected-latent sparse attention.
- [`M6_DEEPSEEK_GRAPH_VALIDATION.md`](M6_DEEPSEEK_GRAPH_VALIDATION.md) — official-source
  deterministic two-layer graph with native logit, greedy-token, expert-route and
  raw/compressed attention-index parity.
- [`M7_DEEPSEEK_MTP_VALIDATION.md`](M7_DEEPSEEK_MTP_VALIDATION.md) — checkpoint-shaped
  single-block MTP reference execution with native draft-logit, proposed-token, route and
  sparse-attention-index parity.
- [`M7_DEEPSEEK_INCREMENTAL_VALIDATION.md`](M7_DEEPSEEK_INCREMENTAL_VALIDATION.md) — five
  official main+MTP decode steps with transactional circular raw state, independent learned
  attention/indexer compression and exact cache-boundary parity.
- [`M7_LOGICAL_TENSOR_CATALOG_VALIDATION.md`](M7_LOGICAL_TENSOR_CATALOG_VALIDATION.md) —
  model-independent tensor roles, immutable source ranges and 11,264 pinned DeepSeek expert
  bundles coalesced into exactly 22,528 bounded ranges.
- [`M9_RESIDENCY_STATE_VALIDATION.md`](M9_RESIDENCY_STATE_VALIDATION.md) — legal
  NVMe/RAM/VRAM transitions, atomic Governor reservation ownership, RAII use leases,
  concurrent admission and fail-closed cleanup.
- [`M9_PACKED_EXPERT_LOAD_VALIDATION.md`](M9_PACKED_EXPERT_LOAD_VALIDATION.md) — governed
  multi-range reads into one packed RAM buffer, exact per-tensor slices, ready-state publication
  and failure cleanup.
- [`M9_ASYNC_EXPERT_LOAD_VALIDATION.md`](M9_ASYNC_EXPERT_LOAD_VALIDATION.md) — fixed workers,
  bounded queues, pre-enqueue reservations, exclusive transfers and shutdown cancellation.

M6 is complete for its reduced float correctness gate. M7/M9 now include MTP, reduced incremental
state, a bounded exact-sequence reference-generation lifecycle and one complete real quantized
expert. M8 now has native CUDA capability, governed allocation and one scalar RMSNorm copy/kernel
gate, while full-shard/layer/token official-payload DeepSeek execution and generation, optimized
CUDA kernels, speculative verification and GPU inference remain future work. The preceding
completed storage milestone is M5:

- [`M5_DEVICE_IO_CACHE_BYPASS_VALIDATION.md`](M5_DEVICE_IO_CACHE_BYPASS_VALIDATION.md) —
  device-level Windows/Linux counters, bounded direct/unbuffered reads, repeatable
  Windows physical-byte measurements and the completed M5 gate.

The preceding [`M5_READINESS_STAGED_IO_VALIDATION.md`](M5_READINESS_STAGED_IO_VALIDATION.md)
records readiness replay, output-identical two-stage expert reads, bounded two-worker
safety and paired warm-buffered measurements.

The latest release-level report remains
[`ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md`](ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md),
covering the Vastloom rebrand, compatibility, governance, release packaging, visual
assets and cross-platform validation.

## Earlier milestone reports

These records predate the milestone numbering and several predate the Vastloom name. They remain
the evidence for the paths that later work builds on, so they are listed here rather than left
unreferenced:

- [`ALPHA4_IO_OBSERVABILITY_VALIDATION.md`](ALPHA4_IO_OBSERVABILITY_VALIDATION.md) — process-visible
  I/O deltas separated from physical device traffic, plus exact unique-expert reuse distance in the
  real OLMoE pipeline.
- [`ALPHA4_ASYNC_IO_VALIDATION.md`](ALPHA4_ASYNC_IO_VALIDATION.md) — the OLMoE CPU oracle connected
  to the bounded one-worker asynchronous expert-I/O path with measured read-service/CPU overlap.
- [`ALPHA4_OLMOE_VALIDATION.md`](ALPHA4_OLMOE_VALIDATION.md) — the pinned OLMoE adapter, top-8
  routing parity and NVMe-to-RAM expert streaming under a RAM cap below checkpoint size (M4/M5).
- [`ALPHA3_SMOLLM2_VALIDATION.md`](ALPHA3_SMOLLM2_VALIDATION.md) — the pinned SmolLM2-135M-Instruct
  dense CPU path with tokenizer, logit and greedy-output parity against Transformers (M3).
- [`REAL_MODEL_INFRASTRUCTURE_VALIDATION.md`](REAL_MODEL_INFRASTRUCTURE_VALIDATION.md) — bounded
  JSON/Safetensors parsing, immutable import and model-store foundations shared by every later
  real-model adapter.
- [`ALPHA2_WINDOWS_VALIDATION.md`](ALPHA2_WINDOWS_VALIDATION.md) — the alpha.2 source revalidated on
  the Windows 11 reference PC, including the wide-character and UTF-8 console paths.
- [`ALPHA2_VALIDATION.md`](ALPHA2_VALIDATION.md) — the first complete native CPU inference path:
  verified manifest, TinyTransformer loader, KV cache, decoding and CLI (M2).
- [`ALPHA1_VALIDATION.md`](ALPHA1_VALIDATION.md) — the research harness, safety state machine and
  reservation core (M0/M1).
