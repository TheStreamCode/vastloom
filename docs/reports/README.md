# Validation reports

Reports are evidence snapshots tied to a date, host, source revision and project state. They are not
rolling claims: a later code change does not retroactively alter an earlier result.

Releases through `0.4.0-alpha.3` used the former project name, executable and persisted identifiers.
Those historical documents are intentionally preserved verbatim. Beginning with
`0.4.0-alpha.4`, current documentation and commands use **Vastloom**. Compatibility readers keep
the already-downloaded model stores and checkpoint manifests usable without mutation.

Start with the newest report for current claims, then consult milestone-specific reports for exact
hardware, commands, measurements and limitations.

## Current report

- [`M9_DEEPSEEK_PACKED_LAYER_VALIDATION.md`](M9_DEEPSEEK_PACKED_LAYER_VALIDATION.md) —
  ggml-verified deterministic Q8_0 encoding, all 23 matrices of one reduced DeepSeek main layer,
  I32 hash routing, full prefill/incremental graph parity and transactional malformed-input tests;
  the fixture is still memory-owned rather than bound to governed file-loader leases.
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

M6 is complete for its reduced float correctness gate. M7 now includes MTP, reduced incremental
state and one complete real quantized expert, while full-shard/layer/token DeepSeek execution,
optimized packed kernels, speculative verification and CUDA remain M7/M8 work. The preceding
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
