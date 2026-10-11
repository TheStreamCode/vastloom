# Changelog

## 0.4.0-alpha.5 - Unreleased

- **Robustness pass (October 2026)**:
  - every shell-out in the hardware probe is now bounded by a hard 3 s deadline on
    Windows too (CreateProcess with an explicit pipe and TerminateProcess on
    timeout): a wedged driver can no longer freeze `HardwareProbe::collect` and
    with it every CLI entry point;
  - `vastloom-watchdog` gains an always-on smoke test (argument ranges,
    stale/unreadable heartbeat in observe mode, PID mismatch, and the enforced
    two-sample termination of a live child);
  - the trace simulator's LFU policy keeps its victim ranking in an ordered
    index: evictions are O(log n) instead of a full map scan per victim, which
    made million-record traces unusable;
  - the Qwen importer pins both known Unsloth revisions to immutable commits
    (a moving `main` silently downloaded different bytes under the same claimed
    revision) and requires `--revision` for any other repository;
  - checkpoint downloads retry transient transport failures (resets, timeouts,
    truncated responses) up to three times with backoff and resume, in both the
    full-file and the bounded-range importers; clean protocol refusals are not
    retried;
  - the safety-profile budget tables use designated initializers (a transposed
    pair of literals was a silent policy corruption), the release workflow
    refuses tags that disagree with the CMake project version, the
    `metadata_only_only` field typo is renamed, and `render_paper_pdf.py` uses
    argparse like every other tool.
- **Cross-platform build restored (October 2026)**: the tree now compiles clean with
  GCC/Clang under `-Wall -Wextra -Wconversion -Wsign-conversion -Wdouble-promotion
  -Werror` (the CI matrix and CodeQL build had been failing on Linux since the
  repository went public). Platform-typed socket results, explicit float-to-double
  casts at every double-promotion site, `[[maybe_unused]]` on ggml-only helpers,
  and bounded explicit reads replacing the `istreambuf_iterator` forms that trip
  GCC 13's `-Wnull-dereference` false positive at `-O3`. CI now also runs the
  offline distill test suites, which no runner had executed before.
- **Trace simulator defect**: an expert too large for the VRAM tier now falls back
  to the RAM tier instead of silently becoming a permanent storage miss (LRU and
  LFU, both hit-promotion and storage-miss paths), and PCIe traffic is only
  counted when the VRAM tier actually accepts the payload.
- **Whole-project review (October 2026)**: a full read of every source area
  (model loading and governed I/O, inference kernels, runtime/API/build)
  found no CRITICAL defects; every MAJOR finding was fixed without touching
  a single measured parity claim:
  - Windows Unicode safety restored where it was violated: the heartbeat
    write and the verification-receipt write now stage through
    `std::filesystem::path` (previously narrow-string names could corrupt or
    mismatch the atomic rename), and the receipt write no longer swallows
    its own failure.
  - `BoundedIoPool` survives a worker-start failure (wake, observe
    stopping_, join, rethrow) mirroring `BoundedExpertLoadPool`; the IO pool
    was the only pool that could hang the process under thread exhaustion.
  - The loopback server applies a 30 s receive deadline per connection: a
    client that opens a socket and goes silent can no longer wedge the
    serial server (and `stop()`) forever.
  - The OpenAI protocol's `stop` sequences are now actually honoured in both
    serve handlers (tiny and real model): a holdback-filter truncates
    streaming output at the earliest stop sequence, non-streaming text is
    cut before it, and the finish reason reports `stop`.
  - `qwen38_delta_net_decode` and `qwen38_reduced_zero_state` reject
    conv kernels above the fixed 8-slot history window (previously a stack
    out-of-bounds write with kernel >= 9; the pinned Qwen3.8 invariant is 4).
  - The raw Q8_0 payload cache is bounded by the machine's own safe runtime
    policy (at most a quarter of the effective safe host limit, never above
    the historical 16 GiB) instead of a hardcoded ceiling, and its enable
    flag is `std::atomic` (the unsynchronized read was a formal data race).
  - Expert prefetch converts allocation failures into `Result` errors
    instead of `std::terminate` at `future.get()`; `file_size` uses the
    `error_code` overload on both expert slice paths.
  - `RowWorkPool` no longer deadlocks or terminates when a worker job
    throws: the first failure is rethrown at the caller once every helper
    has finished, and an inline failure releases the generation.
  - The GGUF vocab reader guards the corrupt-length wraparound
    (`pos + len` -> subtraction with bound) that could request a ~2^64
    allocation.
  - The watchdog validates option ranges (no `chrono` UB from unbounded
    `stoull`), and `--enforce` now requires two consecutive stale samples
    before terminating the runtime.
  - The chat protocol refuses `temperature` outside 0..10 (a finite double
    above FLT_MAX previously became `+inf` after the float cast), and
    `qwen38-greedy --json` emits text through the shared `json_escape`
    (control bytes are escaped instead of silently dropped).
  - Documented contracts that were implicit: `ggml_compute_disable` is
    owner-thread-only, the persistent shard stream is single-thread by
    design, the expert LRU pointer is invalidated by the next acquire, and
    the 64-byte GGUF tensor-name bound is the reader's own.
- **Q11i-Q11j performance work**: Q8_0-native expert matvec (bit-identical),
  O(1) expert LRU, concurrent expert prefetch, Q8_0-native trunk residency;
  greedy path 33.75 -> ~3.3-4 s/token on the reference PC while keeping the
  llama.cpp parity checks green (the structured margin-aware battery tooling is a pending Q11l deliverable).
- **P3a `--cuda-experts`**: bit-identical VRAM-resident Q8_0 expert tier;
  measured slower than the 12-thread CPU kernel (FP64 parity accumulation
  cannot use parallel reductions on consumer GPUs) and kept opt-in.
- **Q11k ggml backend**: opt-in `--ggml-backend` SIMD Q8_0 matvecs
  (forward ~2x on compute); piece-aligned with the margin-aware parity
  definition introduced with D1's root-cause analysis.
- **D1**: the oracle quantizes activations per matvec; parity is enforced
  margin-aware and the activation-quantization mirror was considered and
  dropped. Landscape: ik_llama.cpp (Qwen-3.8 support) and KTransformers
  named as the state of the art for hierarchical MoE inference; Vastloom's
  differentiation is the Safety Governor and the capability-transfer path.
- **Project hygiene**: .clang-format policy added; personal RESUME.md
  removed from version control and ignored.

## 0.4.0-alpha.4 - 2026-08-09

- **Strategic alignment - parity redefined, two pillars, landscape checked.**
  D1's root cause (the oracle quantizes activations to Q8_0 on every Q8_0
  matvec; our double path is more accurate) closed the token-parity question:
  parity with llama.cpp is now a margin-aware verification instrument, not a
  product target, and reproducing the oracle's activation quantization was
  considered and dropped. The project declares two pillars: (1) governed
  hierarchical memory + capability transfer research, and (2) competitive
  pursuit only through levers an mmap-only runtime lacks (expert-quant
  re-import, ggml SIMD as default, a routing-aware VRAM expert tier beating
  the page cache). The README now names the checked state of the art:
  ik_llama.cpp (Qwen-3.8-Flash-Next supported, PR 2365, fused delta-net,
  auto-fit VRAM MoE offload) and KTransformers (CPU-GPU expert scheduling,
  3-tier prefix cache) are ahead for plain hierarchical inference on a 12 GB
  card, and honest benchmarking must include them.

- **Q11j - Q8_0-native trunks (P2d) and the measured CUDA expert verdict (P3a).**
  The large trunk projections (hyper-connections, DeltaNet and QSA
  projections, shared expert) now keep native Q8_0 blocks through a
  `Qwen38Weight`/`linear_w` seam that is bit-identical to the F32 path, so
  resident startup went 16.8 s -> 3.1 s, the 8-forward greedy run 72.6 s ->
  29.8 s (~3.3 s/token) and the no-override streaming path 194 s -> 32 s;
  the streaming raw-payload cache also degrades gracefully under commit
  pressure instead of crashing (a pre-existing defect found while A/B-ing).
  `--cuda-experts` was then wired through the existing VRAM-resident Q8_0
  matvec accelerator with device-only cache entries (no NVMe read, no host
  payload for experts the device already holds). It is bit-identical but
  SLOWER than the 12-thread CPU kernel on the reference RTX 4070 Super
  (39.8-43.5 s vs 35.7 s over 8 forwards): the FP64 parity accumulation
  cannot use the device's parallel reductions and the parity-friendly grids
  are tiny, so the tier stays opt-in and the CPU kernel is the product path.
  The next performance lever is a smaller expert quant, not the GPU.

- **Q11i - Q8_0-native experts and concurrent prefetch (4x greedy speedup).**
  Routed experts are no longer dequantized to a 19.7 MB F32 copy per use:
  `qwen38_expert_mlp_q8` reads contiguous Q8_0 blocks directly with the same
  per-row sequential double accumulation as `linear_f`, so results are
  bit-identical to the F32 path and the full oracle parity check was not
  needed to accept it. The greedy CLI caches raw Q8_0 slices in an O(1) LRU,
  dispatches all top-k misses as concurrent per-thread reads, and clamps
  cache capacity so no pointer is used across an eviction. The delta-net
  decode conv is channel-parallel, its recurrence loop value-head parallel,
  and the `linear_f` parallel threshold now covers expert-sized matvecs.
  End to end the greedy CLI went from ~33 s/token (Q11h) to ~4 s/token on the
  reference PC with trunk F32 residency on; llama.cpp on the same machine is
  ~3.2 tok/s, so the CPU gap is now roughly twelve times and it is a
  residency-architecture gap (Q8_0 under the page cache vs 13 GB of F32),
  which is the next milestone alongside CUDA expert execution.

- **Q11h - Greedy parity with llama.cpp achieved.** The native Vastloom
  runtime now generates token-identical output to llama.cpp on the official
  Qwen3.8-Flash-Next Q8_0 checkpoint ("The capital of France is" → " Paris.
  The capital of Germany is Berlin. The capital of Italy is Rome. The" for 16
  greedy tokens). Systematic stage-by-stage comparison against llama.cpp
  (built locally, rev 6fe7498, with an optional intermediate-tensor dump)
  surfaced five real defects: the DeltaNet output gate is sigmoid, not silu
  (llama.cpp: "the one numerical difference from Qwen3.5's GDN"); the GGUF
  ssm_a carries pre-applied -exp(A_log); GGUF norm weights are stored as
  (1+w) except the RMSNormGated ssm_norm; k-head mapping is ggml-repeat tile
  semantics (vh % k_heads); and the greedy CLI replayed a stale trunk across
  tokens. Partial RoPE (64/256, theta 1e7) over attention and indexer, the
  PLE n-gram block (hash → fused 160-per-head table lookup → gate → dilated
  depthwise conv) and a correct multi-token prefill were also required. The
  external oracle is now established for all future milestones.

- **Q11g - Qwen3.8-Flash-Next GGUF (qwen4exp) checkpoint source.** The
  official unsloth Q8_0 GGUF (6 shards, 175.3 GiB, SHA-256 verified through
  the canonical importer, now generalized with `--repo/--gguf-dir`) replaced
  the deleted FP8 safetensors checkpoint. New `src/inference/qwen3_8_gguf.cpp`
  implements Q8_0/F16 decode on top of the generic `GgufSplitCatalog`,
  validates `general.architecture=qwen4exp` (1224 tensors), converts the ggml
  [in, out] layout, reconstructs the fused indexer projections, reads routed
  experts as fixed-stride slices of the fused `[in, out, 512]` tensors and
  embedding rows as contiguous Q8_0 column spans. A safetensors/GGUF
  dispatcher keeps every CLI working on both layouts; the opt-in checkpoint
  test now covers both. Full-depth 0..47 runs on the Q8_0 checkpoint
  (6.58 GB payload, deterministic replay); sums differ from FP8 by design
  (requantization). All fp8/E4M3 decode is now exercised only by historical
  pins.

- **Q11f - Qwen3.8-Flash-Next real layer chain and 48-layer full-depth run.**
  `qwen38_layer_forward` now takes the gated-residual input stream (a layer's
  output feeds the next; only layer 0 starts from the expanded embedding row),
  and the new CLI `vastloom qwen38-chain --layers --tokens` runs consecutive
  real layers with per-layer state and a shared bounded expert LRU. Found and
  fixed the access violation that blocked the full-depth run: the streaming
  mode's final report indexed loaded trunks out of bounds (one trunk resident
  at a time); the attention kind is now derived from the layer id. Streaming
  mode is verified bit-identical to resident mode, and the official
  checkpoint runs all 48 layers (36 DeltaNet + 12 QSA, 480 real FP8 experts,
  9.63 GB payload read) with a bit-identical replay (output sum −28.9797).
  The n-gram PLE block on layer 1 is documented as not yet integrated.

- **Q11e - Qwen3.8-Flash-Next real QSA full-attention layer.** The layer
  loader is generalized to any layer: `qwen38_load_layer_trunk` resolves the
  22 non-routed tensors from the weight map (no hardcoded shard names) and
  loads the Gated DeltaNet or QSA branch according to
  `full_attention_interval`; `qwen38_layer_forward` takes a caller-owned
  `Qwen38LayerRuntimeState` (DeltaNet conv/SSM or QSA indexer-key + KV
  caches). The real layer-3 QSA weights match the pinned operator layout
  exactly (q_proj carries query+gate per head: 24x256x2). On the official
  checkpoint the QSA layer runs a deterministic 5-token prefill (BOS sum
  28.0353, bit-identical replay), and the CLI (`qwen38-layer --layer 3`,
  `qwen38-prefill --layer 3`) agrees with the opt-in test.

- **Q11d - Qwen3.8-Flash-Next real multi-token layer-0 prefill.** Trunk/row
  split (`qwen38_load_layer0_trunk` builds the expert->shard map once; embed
  rows stay bounded 5 KiB reads), caller-owned DeltaNet state so conv history
  and SSM carry across tokens, and new CLI `vastloom qwen38-prefill --tokens`
  with a bounded expert LRU (default cap 16). On the official checkpoint a
  5-token prefill is deterministic end-to-end (bit-identical replay), the BOS
  sum reproduces the Q11c single-token result exactly, the router provably
  depends on the carried state (same token, different route after BOS), and
  the LRU records real recurrence hits (44 misses / 6 hits / 28 evictions).
  Opt-in checkpoint test extended with prefill + replay determinism.

- **Q11c - Qwen3.8-Flash-Next real layer-0 forward.** New CLI
  `vastloom qwen38-layer0 <root> --token N` runs the complete official
  layer-0 on real weights: bounded single-row embedding read (5 KB instead of
  the 1.27 GiB tensor), attn gated residual, zeroed-state Gated DeltaNet,
  combine, mlp gated residual, router (top-10 of 512), shared expert,
  on-demand load of the 10 selected FP8 experts and the final combine.
  Found and fixed a real-weight decoder defect: `qwen38_fp8_e4m3_to_f32`
  treated every `exp==15` byte as NaN, but E4M3FN (the HF format) defines NaN
  only for `exp==15 && mant==7`; `0x78..0x7E` are finite values up to 448 and
  occur in the official checkpoint (3,370 in one gate_proj alone). Stage-named
  finiteness guards were added along the forward. The opt-in checkpoint test
  now also asserts a finite, deterministic layer-0 output with an exact top-k
  route. Self-pinned run: no external oracle exists for `qwen4_exp`.

- **Q11b - Qwen3.8-Flash-Next real checkpoint catalog.** The official
  `unsloth/Qwen3.8-Flash-Next-FP8` 125B checkpoint (172.8 GiB, 131 shards,
  152,089 tensors) is imported through the canonical immutable importer and
  now validates end-to-end in native C++. New CLI `vastloom qwen38-inspect`
  (metadata-only: config + index + optional first-shard header) and opt-in
  serial test `vastloom_qwen38_checkpoint` (`-DVASTLOOM_QWEN38_MODEL=<root>`).
  Fixed fixture-to-real drift: `ExpectedCatalog::ngram_tensors` 128 -> 129 (the
  real checkpoint carries `ple_embedding.ngram_embedding.weight_scale`, which
  the synthetic fixture lacked). New HF-token-aware download path for
  `import_hf_model`/`import_qwen38_fp8` via the `HF_TOKEN` environment
  variable (never logged). No tensor payload has been read for compute yet.

- **Q11 - Qwen3.8-Flash-Next reduced autoregressive generate (infra).** New
  `qwen38_reduced_generate` mirrors the DeepSeek reference driver on the
  reduced graph: bounded prefill then a decode loop with greedy or seeded
  softmax+top-k sampling, `stop_token_ids`, `context_limit_tokens`,
  cancellation and timeout checkpoints, `on_token` streaming and
  prompt/generation/expert telemetry (including tok/s). Always-on tests in
  `test_qwen38_reduced_graph` cover greedy, seeded determinism, stop/cancel/
  timeout/context-limit and empty-prompt refusal. The official 48-layer 125B
  gate remains unmeasured (no checkpoint).

- **Q10 - Qwen3.8-Flash-Next QSA sparse attention.** New
  `qwen38_qsa_sparse_attention` (GQA gather on `selected` only, scaled-dot
  softmax, sigmoid gate) factored from the Q7 layer. Indexer remains
  `qwen38_qsa_select` (Q5). Always-on `test_qwen38_qsa_sparse` verifies
  sparse vs dense divergence and indexer→sparse on 8 synthetic tokens.
  Self-pinned; no ggml oracle, no 4096 live.

- **Q9 - Qwen3.8-Flash-Next DeltaNet persist + paged n-gram cache.** New
  `qwen38_plan_persist_budget` sizes the 36 Gated DeltaNet FP32 states
  (117,669,888 bytes) plus the 4 GiB n-gram scenario-B working set and
  compares the paper total against the 12.3 GiB live-RAM gate. 
  `qwen38_deltanet_persist_create` reserves that state through the Governor.
  `qwen38_ngram_page_cache_open` / `qwen38_ngram_page_gather` keep a bounded
  LRU of table pages with next-page prefetch and hit/miss/eviction/prefetch
  telemetry. Always-on `test_qwen38_persist_and_ngram_pages`. No 125B
  download; official live RAM is still unmeasured.

- **Q8 - Qwen3.8-Flash-Next learned MoE router + expert LRU.** New
  `qwen38_moe_route` (softmax top-k, `norm_topk_prob` renormalization, no
  hash shortcut), `qwen38_expert_mlp`, and `qwen38_moe_block` that adds the
  weighted routed SwiGLU sum to the existing shared expert. Bounded LRU
  `qwen38_expert_cache_acquire` records hits/misses/evictions and logical
  bytes keyed by `(layer, expert)`. The reduced Q7 graph now routes 4
  experts at top-2 on every layer including MTP. Always-on
  `test_qwen38_moe_router` pins logits `[1,0,1]` -> experts `0,2` with
  equal 0.5 weights and a one-slot eviction. No llama.cpp oracle.

- **Q7 - Qwen3.8-Flash-Next reduced prefill/decode graph.** New self-pinned
  two-layer + MTP reference: Gated Residual wraps a Gated DeltaNet block and
  a QSA block (indexer select + masked GQA + sigmoid output gate), each
  followed by the shared SwiGLU expert only. Global mixer collapses the
  widened residual stream; MTP mixes last hidden and last embed through
  `fc_hidden`/`fc_embed` and one QSA+MLP layer. `qwen38_reduced_prefill` and
  `qwen38_reduced_decode` keep DeltaNet/QSA/MTP state; a two-token prefill
  matches two incremental decodes. Also exported `qwen38_linear`,
  `qwen38_shared_mlp`, `qwen38_expand_stream`, `qwen38_qsa_layer_token`.
  Always-on `test_qwen38_reduced_graph`. No routed experts (Q8) and no 125B
  download.

- **Q6 - Qwen3.8-Flash-Next non-routed layer graph.** New
  `qwen38_plan_non_routed_graph` emits the canonical 48-layer + 1 MTP plan
  (36 Gated DeltaNet / 12 QSA from `full_attention_interval=4`) without
  routed experts or vision. Each language layer maps 22 always-active
  tensors (9 attention + 8 Gated Residual + 5 shared/router); layer 1 adds
  the 138 PLE objects; MTP contributes 29 (QSA layer + `fc_*` + mixer).
  Consumers are DeltaNet, QSA indexer/attention, Gated Residual, MoE
  router/shared, n-gram PLE, MTP head and the three global mixer/embed/lm_head
  tensors (1228 text-side non-routed names, pinned against the real Unsloth
  FP8 index). `qwen38_validate_non_routed_plan` enforces those counts;
  `qwen38_bind_non_routed_plan` attaches shard paths and refuses a missing
  name. Always-on `test_qwen38_non_routed_graph_plan`.

- **Q5 - Qwen3.8-Flash-Next self-pinned primitive operators.** Native C++20
  kernels matching Hugging Face `modeling_qwen4_exp.py` (llama.cpp still has no
  `qwen4_exp`). New `qwen38_rms_norm_affine` (grouped RMS, affine `1+w`),
  `qwen38_gated_residual` / `qwen38_gated_residual_combine` (SiLU bottleneck
  scaled by `hc_count`, sigmoid read-gate, mean over residual branches, write
  gate `2*sigmoid(inject/hc_count)`), `qwen38_delta_net_decode` (causal conv1d +
  SiLU, L2-norm Q/K, GQA repeat, gated-delta recurrent update in FP32, sigmoid
  gated RMSNorm, output projection), `qwen38_qsa_select` (mean-pool
  `compress_ratio` keys, ReLU-sum scores, top-k complete blocks plus incomplete
  tail), and n-gram `splitmix64` / prime vocab / hash-XOR / table lookup.
  Always-on `test_qwen38_flash_next_reference_operators` pins reduced golden
  values for all four primitives. No 125B download.

- **Q4 - Qwen3.8-Flash-Next range importer + safety governor.** New
  `qwen38_read_tensor_payload(shard, header, tensor_name)` reads exactly one
  tensor's payload (and its optional `weight_scale_inv` companion) from a
  safetensors shard with strict bounds checks against the header JSON length
  and file size, computing the SHA-256 on the fly. New
  `qwen38_import_tensor_transaction(request, header, governor)` implements the
  three-object gate: source range (tensor + scale) -> expected SHA-256 verified
  before any write -> bounded temp write + atomic rename. The Governor must
  admit the aggregate (weight+scale) as a RAM reservation or the transaction
  fails with `resource_exhausted`. Atomic refusals for out-of-bounds range,
  hash mismatch, destination collision and missing shard/dir, with no `.tmp`
  leftover on rejected paths. Always-on test `test_qwen38_import_transaction`
  on the 3.6 KiB mock shard covers: BF16 `A_log` read (no scale), F8_E4M3
  `up_proj.weight` [6,5] + BF16 `scale_inv` [1,1] read, missing tensor refused,
  successful import producing `al_log_range.bin`, wrong-hash refusal, existing
  destination collision, and governor-full refusal.

- **Q3 - Qwen3.8-Flash-Next tensor & expert catalog.** Extended `Qwen38InspectionResult`
  with `Qwen38LayerSummary` (48 entries, per-layer total 3094, layer 1 3232 with
  138 PLE). New constants `Qwen38ExpectedCatalog` (152089 tensors, 131 shards,
  148650 language, 333 vision, 3101 mtp, 5 global, 128 ngram, 75264 FP8
  `weight_scale_inv`) and function `qwen38_validate_catalog` that enforces them.
  `qwen38_summarize_layers` classifies each `weight_map` entry by layer and type
  (linear_attn 9 / self_attn 9, attn_hyper 4, mlp_hyper 4, gate/shared, 3072
  experts per layer as 1536 weight F8_E4M3 +1536 scale BF16, ple 0/138). Always-on
  test `test_qwen38_catalog_validation` builds a synthetic full catalog of 152089
  entries with the exact distribution and proves `validate` passes, and that the
  mock fixture (25 tensors) is correctly rejected. Discovery documented from the
  real `model.safetensors.index.json` (17 MB, fetched 2026-08-28): every language
  layer has 3094 tensors, the 128 ngram shards live on layer 1, mtp contributes
  3101 tensors.

- **Q2 - Qwen3.8-Flash-Next reference operators (FP8/BF16/RMSNorm).** Upstream
  `llama.cpp` `d0132a6` still has no `qwen4_exp` (only `qwen3next`/`qwen35` with
  DeltaNet + full-attention ibrida; QSA / n-gram / Gated Residual are missing),
  so the original Q2 gate “5 token oracolari pinned-ggml via llama.cpp” is blocked.
  Q2 is redefined as self-pinned reference kernels: new header
  `qwen3_8_flash_next_reference.hpp` with `qwen38_fp8_e4m3_to_f32` (OCP E4M3,
  exp 15 => NaN, subnormal 2^-6), `qwen38_bf16_to_f32`/`f32_to_bf16` (bit-exact),
  `qwen38_decode_bf16`, `qwen38_decode_fp8_block_scale` (F8_E4M3 + BF16 scale per
  block `[128,128]`, verified against real SHARD 2 header: `gate_proj.weight`
  `[640,2560]` F8_E4M3 + `weight_scale_inv` `[5,20]` BF16), and `qwen38_rms_norm`.
  Mock shard `model-00001-of-00131.safetensors` regenerated with BF16 `scale_inv`
  (`1.0` => `[0x80,0x3F]`) and payload-relative `data_offsets` (0-based) plus
  non-zero payload. Always-on test `test_qwen38_flash_next_reference_operators`
  covers FP8 scalar smoke (0x00/0x38/0xB8/0x3C/0x01/0x7F/0xFF), BF16 round-trip,
  BF16 2x2 decode, FP8 block-scale 4x4 with uniform and per-block scales,
  RMSNorm, and real payload read for `A_log` BF16 `[4]` from the fixture.
  Remote header fetch for `model-00002-of-00131` confirms 282800-byte header,
  2048 tensors (1024 BF16 + 1024 F8_E4M3) per shard.

- **Q1 - Qwen3.8-Flash-Next safetensors catalog reader.** Header-only POC of the
  125B FP8 model (`unsloth/Qwen3.8-Flash-Next-FP8`, 131 shards, ~170 GB total)
  downloaded and inspected: confirmed `model_type=qwen4_exp` with 48 layers, 512
  experts, 10 routed + 1 shared, FP8 block-scale (`weight_block_size=[128,128]`)
  on the expert weights. The header walk confirmed the three new primitives are
  present in the file system: `hyper_connection_mixer` global + `attn_hyper_connection`
  per-layer (Gated Residual, 4 branches / rank 320), `linear_attn.*` (Gated DeltaNet
  with `in_proj_qkv` / `in_proj_z` / `conv1d` / `A_log` / `dt_bias` / `norm` /
  `out_proj`), and `layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}`
  (N-gram table, 20M entries × 128 parts, inserted at layer 1). New native
  reader in `inspect_qwen38_flash_next_safetensors()` (parses `model.safetensors.index.json`)
  and `read_qwen38_safetensors_header()` (parses the binary header of one safetensors
  shard, attaches `weight` + `weight_scale_inv` pairs). Always-on test
  `test_qwen38_flash_next_safetensors` runs against a 3.6 KiB mock shard
  in `tests/fixtures/qwen38_flash_next/` (BF16 + FP8 block-scale tensors)
  and verifies dtype counts, scale_inv attachment, and data_offsets
  integrity. The decision to skip the Qwen3.8-27B POC is documented
  in ROADMAP Q1: the 27B is a different model (`qwen3_5`, 64 layer, dense, no
  MoE, no n-gram, no Gated Residual) and would not have validated any of
  the three new primitives.

- **Q0 - Qwen3.8-Flash-Next skeleton, deep-clean DeepSeek artifacts.** The execution track
  pivots from DeepSeek V4 Flash 0731 to Qwen3.8-Flash-Next (Qwen4 architecture preview).
  Skeleton in `include/vastloom/inference/qwen3_8_flash_next.hpp` declares the
  `Qwen38FlashNextConfig`/`Qwen38FlashNextInspection` types, the
  `load_qwen38_flash_next_config` parser (with strict `model_type=qwen4_exp` guard), and
  `inspect_qwen38_flash_next_config` that asserts pinned invariants for the 2026-08-26
  release. Always-on test `test_qwen38_flash_next_config` parses a mock config (no real
  download) and checks the three new primitives (QSA indexer, n-gram PLE, Gated Residual)
  are present in the parsed metadata. Decision matrix and FP8-quota rationale live in
  `docs/planning/QWEN38_BUDGET_ESTIMATE.md`; archive of the previous DeepSeek track lives
  in `docs/archive/deepseek_0731/`. Disk hygiene: removed 94.79 GiB of DeepSeek originals
  from `C:\StargateModels\models\original\` and 13.83 GiB of historical build directories
  from `build/` (50+ stale `m9-*`/`m10-*`/`wsl-iq1s-*`/`vastloom-*-smoke` dirs and
  `vastloom-gguf-test-store` cache). Host now reports 1.34 TiB free on `C:`.

- **M13 prefix-KV session reuse: the server remembers.** `vastloom serve MODEL_DIR` now
  dispatches to the dense SmolLM2 path when given a directory (matching the chat/model-info/
  self-test dispatch pattern), creating a persistent `LlamaSession` whose Key/Value buffers
  are Governor-reserved once at up to 4,096 positions and held across requests. Each turn,
  the framed conversation is diffed against the cached sequence via a new pure
  `common_prefix` function; only the diverging tail is evaluated (`start_position` was
  already supported inside `evaluate_tokens` - this opens it). Divergence or overflow
  degrades safely to the stateless path; the response carries a non-standard
  `vastloom_cached_prefix_tokens` field so clients can observe reuse. The internal
  `InferenceSession` was refactored from owned vectors to spans, letting the same buffers
  serve both per-call and persistent lifetimes without copies. Always-on tests gate the
  prefix function and session contracts; the opt-in SmolLM2 suite proves on the real
  checkpoint that turn two reports nonzero cached tokens while turn one reports zero.
  ROADMAP M13's session/prefix-KV checkbox is now checked (dense path scope).

- **DFlash companion drafter inspected, and CPU-regime speculation measured closed.** New
  `docs/research/DFLASH_DRAFT_FEASIBILITY.md` documents the local
  `deepseek-v4-flash-0731-dspark` companion: a ~19.85-billion-parameter, three-layer
  teacher-shaped drafter (same MLA/MoE/indexer family, 256-expert 6+1 MoE, hyper-
  connections with sinkhorn) stored as 10.15 GiB of mostly MXFP4, conditioning on teacher
  layers [41,42,43] and predicting markov blocks of five tokens - the DFlash paper's design
  realized against this checkpoint. Vastloom's existing inspector refuses it (identity-pinned
  catalog), so facts came from a standalone header walk. The Phase A1 RAM blocker falls:
  both models map comfortably (~22 GiB remained free through a dual-model run). The
  empirical ceiling check via the already-built llama.cpp reference (`--spec-type
  draft-dspark`) then measured the speculative arm at **0.78x the plain baseline**
  (1.4 vs 1.8 tok/s generation, lossless `Roma` continuation in both): three teacher-shaped
  MoE layers forwarding per proposed block cost more CPU than acceptance amortizes in this
  single-stream consumer regime. Speculative decoding is therefore closed for this project's
  regime with data; the pairing stands as live E4 evidence for target-conditioned state
  injection.

- **Final-hidden distillation site instantiated, and the head-boundary question closed with
  measured evidence.** New `--site hidden` on `module-fit` trains an adapter that reshapes
  the final hidden state BEFORE the frozen head consumes it (`logits = W(h + B(A h))`),
  injected through a new `CapabilityTarget::final_hidden` binding; gradients stay
  closed-form through the head transpose, and the fitter carries the same parity start,
  decoupled decay and fail-closed divergence guard as its LM-head sibling, plus global-norm
  gradient clipping. Two real defects were fixed along the way, both found by instrumented
  measurement: an unsigned-underflow in the shared splitmix initialization turned "negative"
  draws into ~+0.33 positives (harmless at the LM-head site, explosive through Wᵀg), and
  letting both adapter factors move resonated through ‖W·B‖ until the projection was frozen
  at its deterministic init. Measured on the grown transcript (29 positions, rank 8, 1200
  steps): stable everywhere, behaviorally flat - best loss 3.34 -> 3.19, zero movement on
  any probe. Combined with the earlier data-lever negative, the pre-declared criterion now
  fires with evidence from both placements: neither induces generalizable skill, so real
  capability transfer requires network-depth backpropagation - an offline torch trainer or
  equivalent becomes the only credible route, versus parking capability-transfer and
  returning to M13 prefix-KV. The decision is recorded in the vision document with the full
  three-way placement table.

- **Data-lever experiment closed with a measured negative.** The distillation probe set grew
  from three to eight train records (five capitals, three translations; 29 supervised
  positions) plus seven eval probes, transcribed through the resume loop - which also caught
  the author, not the teacher: DeepSeek answered "Brasília" with its proper accent and the
  accuracy gate rightly refused the fixture's accent-less expectation until the fixture was
  fixed and the record backfilled from stored text without a new teacher run. Refit sweep at
  rank 8 across `--reg {10,30,100}`: full train coverage appeared (5/5 leads-with at reg 10)
  but held-out same-skill generalization stayed 0/7 at every weight, and chitchat broke again
  exactly when the fit saturated. Conclusion recorded in the vision document: an LM-head-only
  adapter performs retrieval of trained answers rather than induced skill, so growing data is
  closed as a dead end for this placement; adapter placement beyond the LM head becomes the
  blocking lever and a design decision for a future milestone.

- **E2 compiler executed and completed: closed-form LM-head distillation with decoupled
  weight decay, measured across a train/eval split.** New `vastloom module-fit MODEL
  TRANSCRIPT OUT` distils a capability module straight from a teacher-transcript artifact:
  teacher-forced sequences are framed with the pinned SmolLM2 chat template, final-norm
  hidden states come from the new `LlamaModel::hidden_states`, base logits from the new
  `head_dot`, and `inference::fit_lm_head_lora` runs exact small-matrix-gradient
  cross-entropy descent on the adapter (B starts at zero, so step-one loss equals the frozen
  base's - parity before capability; no autograd engine exists in this project by design).
  The first unregularized run memorized all eleven positions (loss 4.26 -> 0.0) but broke an
  off-skill probe the base answered correctly. The recorded lever was then implemented:
  additive L2 diverged at any weight - the spring resonates through u - and was replaced by
  **decoupled weight decay**, AdamW-style, with the contraction clamped at `max(0, 1-lr*reg)`
  so no hyperparameter pair can overshoot; divergence now fails closed as a refused fit.
  Probes gained `expected` answers and a train/eval split, eval probes never cost teacher
  runs, and new `tools/distill_evaluate.py` measures baseline versus adapted arms with
  strict-equality plus leads-with grading over one reproducible command (fifteen transcript
  tests and eight evaluator tests cover both tools with faked runners). Measured sweep on
  the pinned checkpoint: the operating point is **`--reg 30`** (loss 4.26 -> 0.86), which
  preserves the off-skill "Hello" answer verbatim - the recorded interference defect is
  closed - while capital-2 now leads with the exact teacher answer; translate-1 loses its
  exact match at this weight and unseen same-skill probes stay unsolved, so generalization
  is recorded as absent at eleven training positions, with data and placement named as the
  remaining levers rather than mechanics.

- **E2 slice 1: the teacher-transcript pipeline, executed for real.** New
  `tools/distill_transcript.py` drives `deepseek-forward` over a bounded probe set and
  publishes the `vastloom.distill_transcript.v1` artifact - prompt, completion text,
  completion ids and finish reason per probe - which is the vocabulary-free behavioural
  target any offline module compiler will fit against (teacher and student tokenizers do
  not share a vocabulary, so logit matching was never on the table). The runner is
  injectable, publication is atomic with an 8 MiB ceiling, probes are validated fail-closed,
  and resume persists after every successful probe: the first real run needed three attempts
  because the host's available RAM fluctuated (9.42 GB required at window 43 against 8.2 GB
  falling to 4.2 GB), which also surfaced the preflight's accounting - resident trunk is
  charged as `min(total, average x (window+1) x 2)`, so the window only saves RAM below ~20.
  Probes carry an `expected` answer and every record a `teacher_correct` gate, backfilled
  over already-stored records without new teacher runs; `deepseek-forward --json` prints two
  objects (preflight then generation) and the tool keeps whichever carries tokens. The
  executed finding is recorded in the vision document: the IQ2_XXS greedy teacher fails
  arithmetic word problems (1/3), so arithmetic is excluded as a distillation target; the
  chosen skill is factual recall plus translation, where the teacher scores 3/3 against a
  SmolLM2 baseline failing all six probes. Fifteen unit tests cover the pipeline with a
  faked teacher; the E2 fork stands - the compiler stays offline Python tooling in the
  importer's dependency class, and the native runtime only ever consumes published VLCM
  modules.

- **E1 executed: capability-module injection mechanics on the frozen SmolLM2 base.** New
  `inference::CapabilityModule` container (bounded VLCM v1: rank/target ceilings, all-finite
  weights, unique targets, SHA-256 trailer over every preceding byte), a deterministic
  FNV-trigram request router, and a low-rank overlay kernel that adds each binding's delta
  after the base matvec - the frozen weights are never rewritten. `vastloom module-init`
  writes deterministic synthetic modules sized to a pinned checkpoint (splitmix64 values,
  routing embedding derived from the module name), and `chat --capability-module` (repeatable,
  up to eight) routes the prompt to one module, binds it against the real config dimensions
  and reports `capability:{selected,considered,bindings,added_macs,module_bytes}` in JSON.
  Always-on tests cover round-trip, seven corruption refusals, routing determinism, a
  double-precision delta oracle and zero-scale bit parity; the opt-in SmolLM2 contract now
  proves on the real checkpoint that an inert module leaves greedy output bit-identical while
  an active module is deterministic and cost-accounted. Measured on the reference run:
  rank-8/scale-0.05 across all 150 targets costs 78,796,800 added MACs for 8 tokens (~9.9M per
  token, about 7% of the dense model) in an 8.3 MB module - and degenerates greedy output to
  repeated token 0, so delta magnitude/coverage tuning is recorded as E1's first finding.
  Parity held; cost is bounded and counted. E2 (real teacher distillation) remains the
  milestone that decides capability.

- **F1 evidence synthesis for the capability-transfer vision.** New
  `docs/research/CAPABILITY_TRANSFER_FEASIBILITY.md` answers the vision's five evidence
  questions against the pinned local artifacts (ggml `30bf868`, the repository's device and
  inference layers) and dated published work. Verdicts: behaviour distillation into compact
  low-rank modules is standard engineering with an unknown gain magnitude (E2 remains the
  deciding experiment); retrieval and composition over adapter pools is supported by mature
  published systems (LoraRetriever, MOLE, LoRAuter, MoLoRA - the last reporting specialized
  modules beating a 4.7x-larger model); speculative decoding splits into frontman self-MTP
  (shipped with the selected checkpoint), cross-model draft-verify (sound, pre-modelled at
  3.2-3.7x in the streaming paper, blocked by the Phase A1 host envelope, heterogeneous-
  vocabulary algorithms now exist upstream) and teacher-as-draft (rejected: it reinstates
  per-token teacher latency); the pinned ggml carries every tensor primitive a gemma4
  text-only forward needs including partial-dims RoPE and flash attention with logit
  softcap, so the adapter gaps are Vastloom-side graph-builder work plus one genuinely new
  component, a SentencePiece tokenizer; and every storage type a UD-Q4_K_XL frontman
  contains already has a parity-validated device matvec kernel, which makes hybrid
  CPU-graph/device-matvec offload the recommended first device route ahead of any full
  device graph. The vision document's F1 section now records these verdicts inline.

- **North-star vision: a two-model capability-transfer runtime.** New
  `docs/research/CAPABILITY_TRANSFER_VISION.md` records the direction this project is now
  steering toward: a fast, dense, resident **frontman** (default `gemma-4-12B-it` QAT Q4,
  ~6.72 GB, MTP + native tool calling) expanded at runtime by a **dormant brain**
  (`deepseek-v4-flash-0731`, streamed from NVMe), through one of three bounded bridges -
  an offline compiler that distils the teacher into compact low-rank modules, an oracle that
  wakes the teacher for a single hard step, and retrieval of knowledge/exemplars. The
  document states explicitly what is excluded (splicing teacher parameters into a different
  architecture), converts "x100 intelligence" into a task-scoped falsification ladder
  (E1 module injection, E2 capability transfer, E3 oracle wake-up, E4 cross-model state
  transfer), and inherits every Safety-Governor invariant unchanged. A speed-dependencies
  section states the honest arithmetic: the "fast frontman" promise (~30-60 tok/s) requires
  a device model backend that does not exist yet, CPU-resident execution lands in low
  single-digit tok/s, oracle wake-ups cost seconds-to-minutes per call, and F1 carries a
  fifth evidence question on exactly that device-compute gap. This is a north star,
  not a claim: nothing in it is implemented or measured until a dated gate says so.

- **M13 first slice: an OpenAI-compatible loopback endpoint.** New `vastloom serve
  MODEL_OR_MANIFEST [--port N]` binds `127.0.0.1` only (remote access is out of scope by
  construction) and serves `GET /v1/models` and `POST /v1/chat/completions` from the bundled
  TinyChat path, single-connection-at-a-time so its resource envelope stays bounded. Requests
  are parsed by the existing `api::parse_chat_completion_request` (bounded body/messages/tokens);
  `stream:true` is served as a real SSE stream (role frame, one delta frame per generated piece
  through the model's own `on_text` callback, finish frame, `data: [DONE]`) - the emit callback
  returns false once the client is gone, so a disconnect stops feeding the stream;
  `temperature>0` is refused rather than silently served greedy, and Governor refusals surface
  as 503 via `api::http_status_for`. Backed by a new minimal portable HTTP server
  (`api::LoopbackServer`, Winsock/POSIX) whose oversized-body refusals drain the already
  committed request bytes (bounded, no buffering) so the 413 reaches the client instead of a
  connection reset. Covered by an always-on end-to-end test that does real socket round-trips
  against the bundled fixture (models list, the deterministic "2+2 fa 4." completion, the
  SSE stream, 404, 413 over the body ceiling). Prefix-KV session reuse - the load-bearing
  M13 piece - remains the next step; it needs a stateful model, not the stateless TinyChat
  fixture.

- **Persist-decode validated end-to-end on the real IQ2 catalog via the new preflight.** A
  `--dry-run` sweep at the loaded host's ~5.1 GB live budget shows the mechanism engaging
  exactly when the overhead-inclusive resident set fits: `--layers 8/16` report
  `persist_decode:true`, `--layers 24` fits but falls back to the prefill expert set, and
  `--layers 32/43` refuse up front (exit 75) instead of dying mid-pass. Staging (284 MB
  worst-case range) and the ggml pool (8 MB) are now charged. The full window-43 *timed*
  measurement is **infeasible on a 31 GiB host**: it needs a live budget of ~9.1 GB
  (>= ~17.1 GiB available), and at that RAM the trunk window and the persisted experts cannot
  both be afforded - the §6.5 trade-off (persist bought with trunk residency loses) expressed
  at the admission level. Closed as a measured constraint, not attempted by force.

- **The forward preflight now charges what the runtime reserves, so an approved run cannot be
  surprised mid-pass.** Extracted as `inference::deepseek_v4_forward_preflight` (pure, unit-
  tested): it adds the bounded-read staging for the worst-case single source range and, when
  `--ggml-compute` is on, the backend's graph pool, to the trunk/globals/expert resident set,
  and only enables `persist_decode` when that whole overhead-inclusive total fits the live
  budget. Previously the preflight checked weights alone against `ram_hard_limit` while the
  loader and backend reserved staging and scratch on the same host budget afterwards — the
  failure mode that passed preflight and then died in the expert loader with `bad_alloc`.
  `--json` preflight now reports `staging_bytes` and `ggml_pool_bytes`. This is the plan's
  "count the loader's buffers in the preflight" step; it makes refusals arrive up front
  (exit 75) instead of as mid-pass crashes.

- **The per-token bandwidth budget was an addition error, and is corrected everywhere.**
  The catalog says 6,390,088,008 trunk bytes + 596,000,788 globals + 1,770,651,648 expert
  bytes = 8,756,740,444, i.e. **8.76 GB/token**, not 8.00: the original figure wrote
  6.39 + 2.37 as 8.00 and labelled the 2.37 row "experts" when it was experts plus globals.
  Recomputed against 8.76: 1.8 tok/s needs 15.8 GB/s (not 14.4); halving expert traffic
  bounds end-to-end gain at 1.11× (not 1.17×), 1.25× in the limit of free experts; K=7
  speculative verification amortizes the always-active weight to 1.00 GB per accepted token
  for a 3.2–3.7× reduction, not 4×; the IQ1_S engine delivered 2.9–3.4 GB/s effective, and
  llama.cpp's idle 3.06–3.31 tok/s is 27–29 GB/s. Corrected in the streaming paper (with a
  correction note in §4), `RESUME.md`, `docs/ROADMAP.md`, `README.md`, `AGENTS.md` and
  `docs/research/PHASE_A1_SPECULATION_FEASIBILITY.md`. The executed full-depth forward read
  exactly 1,770,651,648 expert bytes, matching the corrected derivation to the byte.

- **The ggml compute gate re-passed on 2026-08-19 with the paper under revision.** Parity
  0.084 against the 0.2 tolerance (F32 at 9.5e-7), corruption probe 0.379, benchmarks
  167 GB/s Q8_0 cache-hot, 95 GB/s streaming, 14 GB/s IQ1_S and 0.18 GB/s scalar IQ1_S —
  the recorded table values reproduce within machine-state variance. A fresh IQ2_XXS
  window-43 generate was refused by the Governor under this session's host load
  (7.69 GB required against a 3.90 GB live budget, exit 75), which is the designed outcome;
  Table 5 of the paper keeps its recorded state-declared figures. The paper is also
  re-rendered to PDF with the new `tools/render_paper_pdf.py` so the shareable artifact
  carries the corrected figures.

- **Docs match the IQ2 execution path.** `AGENTS.md`, `README.md`, `RESUME.md`,
  `docs/ROADMAP.md`, `docs/MODEL_COMPATIBILITY.md`, `docs/SAFETY.md`,
  `docs/VALIDATION_MATRIX.md` and the streaming paper now state that official
  generate is UD-IQ2_XXS only, ~0.3 tok/s on the reference PC, and that persist
  must not steal the trunk window.

- **A quarter of routed experts come back on the next token.** 1,070 / 4,102
  identities recurred. Persist of the last six experts per layer cut expert
  traffic from 31.2 GB to 24.6 GB on the same prompt — and doubling the trunk
  reloads (43 → 383) to make that persist fit made the run **slower** (6.48 vs
  3.27 s/token). Persist stays opt-in when window 43 already fits; do not steal
  the trunk to pay for it.

- **Expert-load workers keep GGUF shard handles open for their lifetime.**
  `BufferedReadSession` lives on the async pool thread, not process-wide, so
  fixture overwrites still take a fresh open. IQ2 remasure on window 43 stayed
  in the same band (3,488 vs 3,198 ms/token) with a colder page cache (42.7 GB
  device reads vs 23.8). Not a throughput claim.

- **Buffered reads hint SEQUENTIAL_SCAN instead of RANDOM_ACCESS.** Expert and
  trunk ranges are large sequential spans; RANDOM_ACCESS was disabling Windows
  read-ahead. A handle cache on the same shards was tried and withdrawn: it
  served stale bytes after an in-place overwrite, and the corrupt-global gate
  caught it. Window-43 remasure did not fit this host's live budget; a window-16
  run under cache pressure is not a baseline.

- **Official path is IQ2_XXS only.** The UD-IQ1_S store was deleted. First IQ2
  generate on this host (43 layers, chat prompt, `--cpu-only --ggml-compute
  --trunk-window 43`): correct `**Roma**`, stop at 10 tokens, **3,317 ms/token**
  at 2 workers then **3,198 ms/token** at 4 workers. Overlap 4.9 → 7.8 s;
  expert load remains most of the 32 s generate. IQ2 experts are larger than
  IQ1; do not compare that 3.2 s/token to the 2.6 s IQ1 figure as a regression.

- **Window 43 fits again, and the official 12-token run is the fastest yet:
  2,587 ms/token.** The previous 3,063 ms/token needed a two-layer expert
  preflight that refused this host by 400 MB (exit 75) even though peak commit
  on the same prompt was 3.0 GB at window 12. Layers now release after they
  finish, hash prefetch of N+1 runs only on single-token steps, and the
  preflight charges one selected set. Same ids (`**Roma**`), 43 trunk reloads,
  7,655 activation-quantize hits, peak commit 7.46 GB inside a 7.83 GB budget.
  Expert load is still 23.3 s of the 31.0 s generate: that is the next fight.

- **Hash-routed layer N+1 now starts loading during layer N.** `begin` no longer
  evicts the previous layer; the graph prefetches the next hash layer before
  attention and releases a layer only after it finishes. Peak expert residency
  stays two layers, which the preflight already budgets. Learned-router layers
  are unchanged: their identities are still unknown until after attention.
  Packed-matrix matvecs reuse thread-local scratch and, with `--ggml-compute`,
  write the ggml result straight into the row buffer.

- **Expert loads no longer bounce through a second host copy, and ggml quantizes
  each activation once.** `load_prepared_expert_bundle` now reads each source range
  straight into the published RAM buffer. The Governor still reserves staging for
  the largest range; the extra `memcpy` of every expert byte is gone. The ggml
  backend keeps one graph context for the life of `--ggml-compute` and encodes
  each F32 activation to the weight's `vec_dot` type (Q8_K for IQ1_S/IQ3_XXS,
  Q8_0 for Q8_0) on first use, then reuses that encoding for later matvecs of
  the same vector. Gate and up of every expert in a layer share one hidden
  state, so they share one encode. `--json` reports `ggml_input_quantizes` and
  `ggml_input_quantize_hits`. The reuse gate can fail: a second IQ1_S matvec
  with the same input must not increment the quantize counter and must match
  the first result bit for bit; a mutated input must increment it and move the
  output. Wall time on the official checkpoint is not yet remeasured.

- **Working paper.** `docs/research/CONSUMER_STREAMING_INFERENCE.md` collects the measured
  findings as an experience / negative-results report: the per-token bandwidth budget
  (corrected to 8.76 GB/token, see the top of this changelog), the
  measurements that cannot fail, the eight closed directions, the ggml handoff, the
  path-conditioned trunk-window result, and the speculative run that this host cannot
  complete. It does not claim a faster engine.

- **Hash-routed expert identities are known from the token id, so their loads now start
  before the gate matvec.** Learned-router layers still cannot: the router reads the
  post-attention hidden state, and prefetching N+1 during N+1's attention would be a
  guess. On the same 12-token prompt the wall time did not move (3,063 to 3,003 ms/token,
  inside run-to-run noise): only the first three of forty-three layers are hash-routed.
  Generated ids unchanged. 29/29 tests.

- **On CPU-only ggml, a resident trunk and overlapped expert loads cut another 36%.** The
  earlier finding that a wider trunk window was slower was a GPU-path fact: extra host RAM
  stole the accelerator's ceiling. With `--cpu-only --ggml-compute` that coupling is gone.
  `--trunk-window 43` keeps every layer trunk after the first pass (43 reloads instead of 494,
  5.94 GB of trunk traffic instead of 68.4). Two asynchronous expert workers then submit the
  selected misses after routing; the graph computes the shared expert - which lives in the
  trunk - while those loads run. Same prompt, 12 tokens, identical ids:

  | | ms/token | trunk reloads | overlap |
  |---|---:|---:|---:|
  | window 2, sync | 4,816 | 494 | 0 |
  | window 8, sync | 4,541 | 428 | 0 |
  | window 43, sync | 3,446 | 43 | 0 |
  | window 43, async | **3,063** | 43 | 5,017 ms |

  Peak commit 7.09 GB, inside the 11 GB budget. The async queue cap moved from 64 to 256 so a
  12-token prefill (up to 72 selected experts) can submit as one batch. `--json` now reports
  `io_compute_overlap_ms`. Still about 10x behind llama.cpp's 302 ms/token on the same machine
  state: the remaining cost is serial expert I/O after the shared-expert overlap is spent, and
  IQ decode.

- **Compute now delegates to ggml's kernels, and the engine is 6.5x faster - but the gap that
  remains is streaming, not compute.** Phase C of the approved plan started with the entry
  ticket: Vastloom's scalar reference kernels were 21x slower than llama.cpp, which buried any
  story the streaming layer could tell. An opt-in backend now hands matrix-vector products to a
  pinned ggml checkout (0.19.0, commit `30bf868` - the same commit the NOTICE and the parity
  oracles already name), built static and linked behind a new `VASTLOOM_GGML_ROOT` CMake option.
  Without the option the build is exactly what it was: the scalar reference kernels stay the
  default and the validated path.

  Enabling is explicit and governed: `deepseek-forward --ggml-compute` reserves the backend's
  graph pool with the Governor before anything allocates, creates one ggml thread pool at the
  Governor's thread budget, and tears both down on exit. Dispatch sits at the two existing choke
  points - the transactional matvec and the fused routed expert - and falls back to the scalar
  kernels call by call for types, shapes and alignments the backend declines. A backend failure is
  an error, never a silent switch.

  On the real checkpoint (43 layers, 12 tokens, `--cpu-only`, same prompt):

  | | scalar | ggml backend |
  |---|---:|---:|
  | ms/token | 27,925 | **4,282** |
  | expert compute | 231.3 s | 9.1 s |
  | generated ids | `7529 113213 283 9 ...` | identical |
  | text | `La capitale d'Italia è **Roma**.` | identical |

  39,181 matvecs accelerated, 276 fell back, peak commit 1.79 GB. The ids match because the
  backend decodes with ggml's own kernels - the bytes are the same GGUF blocks Vastloom streams -
  and ggml quantizes the activation to the dot type exactly as llama.cpp does; greedy decoding is
  stable across that rounding.

  The new `vastloom_ggml_compute` test gates the path: Q8_0 and F32 parity against the scalar
  reference within accumulation-plus-input-quantization noise (0.084 observed, 0.2 allowed, F32
  at 9.5e-7), a corruption probe built so it can fail (clean backend-vs-backend comparison,
  noise-free), and the benchmarks below. 29/29 tests pass.

  The honest remainder, measured the same day on the same machine state: llama.cpp CPU runs the
  same checkpoint at 302 ms/token, so Vastloom is still about 14x behind. The benchmarks say the
  kernels are no longer why: Q8_0 measures 199 GB/s cache-hot and 100 GB/s streaming a 71 MB
  matrix, at ~0.02 ms per-call overhead. What is left is the streaming itself - 494 trunk
  reloads and governed expert loads per 12 tokens, with no overlap between read and compute - and
  IQ decoding, where IQ1_S measures 16 GB/s cache-hot against Q8_0's 199 and the cold experts in
  the real run averaged 3.2 GB/s. The next lever is I/O, not compute.

- **Speculative decoding on the reference PC reached the machine's envelope, not a code wall.**
  Phase A.1 of the approved plan measured llama.cpp `b1-84e908c` on the same checkpoint at
  3.06-3.31 tok/s today (CPU-only, 12 threads, greedy, correct `**Roma**` output) — against the
  1.51 tok/s recorded this morning with a byte-identical binary and flags. Prompt eval barely
  moved between the two (651 vs 647 ms/token); decode halved (662 vs 326 ms/token). Batch-1
  decode is bandwidth-bound, and machine state — page-cache warmth, background contention —
  decides which figure a run gets. The 1.8 tok/s pinned in earlier documents is one such state,
  not a property of the engine.

  The DSpark speculative run could not be measured: initialization faults ~21 GiB of mapped
  pages in ~8 s (76.9 GB target + 10.1 GB draft on 31.1 GiB RAM); Available Bytes reached
  1.1 GiB before the draft began loading, and the standing safety constraint stops there.
  Three attempts, all aborted by design; pre-warming the target's page cache did not change the
  init profile. The attempt established one ecosystem fact regardless: llama.cpp already ships
  first-class `draft-dspark` support (dedicated implementation, Markov head, anchor-first
  blocks, `ctx_other` target-context wiring). Recorded verdict: the limit is the machine; the
  strategic decision — adopt an existing engine on this hardware, or continue research where
  target+draft fit (~48 GiB free RAM) — is open. See
  `docs/research/PHASE_A1_SPECULATION_FEASIBILITY.md`.

- **A device allocation now costs both budgets, because it costs both.** The Governor tracked VRAM
  and host RAM as separate pools while WDDM backs device memory with host commit — measured on this
  machine at one for one. So a VRAM reservation passed a check that never asked whether the host
  could back it, and the failure surfaced wherever host memory was next requested: a preflight said
  yes and the tile loader died on `bad_alloc` mid-pass.

  `CudaDriver::allocate` reserves the same bytes twice on Windows, once against each budget, and the
  allocation holds both leases. The refusal now arrives where it can be handled — the accelerator
  declines a matrix and that matvec runs on the CPU — rather than where it cannot.

  The command that failed yesterday now completes:

  | | before | after |
  |---|---|---|
  | outcome | `bad_alloc` in the tile loader | completes, output correct |
  | ceiling | ~7.4 GB, then crash | 3.34 GB |
  | peak commit | — | 6.52 GB, inside budget |
  | declined matvecs | — | 0 |

  This was not the accounting gap the roadmap predicted. The loader's buffers were already charged;
  what was missing was that the accelerator's own allocations were charged to one pool while drawing
  on two.

  It also settles a question this milestone kept reopening: **trunk residency is not reachable on
  this hardware.** 6.39 GB of resident weights cost 6.39 GB of host commit on top of a working set
  that already needs about 2.8, against a budget near 11 — and now that the ceiling reflects what
  the host can actually back, it self-limits to 3.34 GB rather than overcommitting and failing. The
  step function stands; this machine cannot reach the step.

- **Residency is a step, not a slope, and the step is above what this host can reach.** Raising the
  accelerator ceiling from 4.33 GB to 5.20 GB by freeing host memory changed nothing measurable:
  uploads stayed at 8.54 GB per pass, the reuse factor at 1.03, evictions at about 20,300. Both
  figures sit below the 6.39 GB trunk, and under a cyclic scan a cache smaller than the working set
  evicts everything before it is next needed whatever its size. 5.20 GB buys exactly what 1.05 did.

- **The 3 GiB reserve was doing real work, and the measurement that said otherwise could not have
  seen it.** Earlier this milestone the unmodelled host allocation was measured at 0.23 GB — peak
  commit 5.12 GB against 2.56 charged and 2.33 held by the accelerator — and the reserve was called
  thirteen times larger than what it stood for. Crossing the residency threshold with
  `--gpu-reserve 1024` produced:

  ```
  preflight: fits: true     (required 2.75 GB, budget 11.24 GB)
  runtime:   Unable to allocate the bounded tensor tile buffer
  ```

  The accelerator took roughly 7.4 GB of VRAM, which costs an equal host commit, and the tile
  loader then had nothing left. That 0.23 GB was measured with a small ceiling — in a regime where
  the accelerator consumed little commit and the buffers it competes with never met resistance. It
  did not generalize to the regime it was being used to justify.

  Fifth check this milestone that could not fail, and the subtlest: not a quantity fixed by
  construction, but one measured under conditions where the effect it bounds does not occur.

  The safety machinery behaved: a refusal, with the host at 19.2 GiB free and responsive. The known
  defect it exposes is the same one the roadmap already carries — the preflight does not charge the
  loader's own read buffers, so it approves and the loader fails. Modelling them is the principled
  way to shrink the reserve, and the only one: the reserve cannot simply be cut.

- `deepseek-forward --gpu-reserve MB` overrides the unmodelled-host reserve, so a residency
  threshold can be crossed under measurement without changing what every other run holds back. The
  default is unchanged and the Governor still refuses in advance.

- **VRAM costs host commit one for one, measured rather than assumed — and six gigabytes of it are
  available.** The belief that a device allocation carries an equal host commit charge on Windows
  has capped the accelerator's ceiling since this milestone began. `cuda-info --allocation-smoke`
  now reports host commit either side of the allocation:

  | device allocation | host commit delta | marginal ratio |
  |---:|---:|---:|
  | 1 GB | 1.25 GB | 1.25 |
  | 2 GB | 2.25 GB | 1.13 |
  | 4 GB | 4.26 GB | 1.06 |
  | 6 GB | 6.26 GB | 1.04 |

  A fixed 0.25 GB for the CUDA context, and a marginal cost converging on 1.00. The coupling is
  real.

  The more useful half: **six gigabytes allocate**. Neither Windows nor the driver refuses them, so
  the engine's 1–2.3 GB ceiling is its own arithmetic — `budget − required − 3 GiB` — where that
  3 GiB reserve was measured earlier to stand in for 0.23 GB of unmodelled allocation.

  What follows is a host-memory condition rather than a code change. The 6.39 GB trunk can be
  resident when the machine has roughly 22 GiB free; at 17 GiB the ceiling lands near 3.2 GB and it
  cannot. Residency is worth having: it would take PCIe traffic from 8.54 GB per pass to 2.37 and
  the reuse factor from 1.03 to about four.

  The smoke's argument ceiling went from 64 MiB to 24 GiB to make this measurable. It is a sanity
  bound on the argument, not the safety mechanism — the Governor still refuses in advance anything
  the host cannot afford, and did so at 8 GB.

- **Interleaving the Q8_0 weight layout for coalescing was 21% slower, and the reason ties it to a
  different problem.** One thread per output row against row-major storage puts consecutive threads
  a whole row apart, so a warp issues thirty-two separate memory transactions. Storing block *b* of
  every row together fixes that — a warp then covers 1,088 contiguous bytes — and it preserves the
  accumulation order exactly, because each thread still walks its own blocks and columns in the same
  sequence. Paired at a pinned 1 GiB ceiling:

  | | s/token | kernel ms |
  |---|---:|---:|
  | row-major | **6.81** | **26,165** |
  | block-interleaved | 8.27 | 30,046 |

  The rearrangement costs one pass over the matrix per upload, and with a reuse factor of 1.03
  every matrix is uploaded on every token. So the engine now shuffles 8.5 GB on the CPU per pass to
  save reads on a kernel that is not bandwidth-bound at batch 1 — one thread grinding 4,096 serial
  iterations is latency-bound, and coalescing a machine that is already stalled does not start it.

  Reverted, but the idea is not refuted: it is **coupled to residency**. Repacking amortizes only
  over matrices that stay on the device, and none do. If the VRAM ceiling ever holds the trunk, this
  becomes worth retrying — and at that point the accumulation-order-preserving version is the one to
  reach for first, before anything that needs a tolerance.

- `--json` now reports `device_read_bytes`, `device_read_operations`, `gpu_uploaded_bytes` and
  `gpu_upload_ms`, which is what separated a warm page cache from a saturated disk.

- **Removing the GPU makes a token four times slower, so the accelerator stays.** The advice under
  consideration was to drop the hybrid GPU decode and invest in a CPU path, on the sound argument
  that batch-1 decode with 1-2.3 GB of resident VRAM makes the device a PCIe streaming accelerator
  rather than a place where weights live. Measured on this code:

  | | s/token | effective bandwidth |
  |---|---:|---:|
  | llama.cpp, CPU | **0.55** | **15.9 GB/s** |
  | here, hybrid GPU | 6.88 | 1.27 GB/s |
  | here, `--cpu-only` | 28.5 | 0.31 GB/s |

  (Effective bandwidths recomputed against the corrected 8.76 GB/token budget.)

  Expert compute goes from 15,706 ms to 235,897 when those matvecs fall back — fifteen times — and
  that is the whole difference.

  The useful number is the third row against the first: **our CPU kernels are 51x behind llama.cpp's
  on the same device.** That gap is too large for anything but what the advice itself described —
  SIMD-friendly weight repacking, AVX512 quantized kernels, a fused MoE — so the diagnosis of what
  llama.cpp does right survives even though its conclusion does not.

  Neither the device nor the memory hierarchy is the lever. The kernels are, on both sides: the GPU
  starts 17x off its own bandwidth, the CPU roughly 51x off llama.cpp's. The GPU has the shorter
  road because it is already four times ahead.

  A prediction was written before the run, as usual, and it was wrong: rough parity was expected on
  the grounds that 1.27 GB/s is nowhere near any bandwidth limit, so the device should not matter.
  It matters because our two kernel sets are not equally bad.

- **A token moves 8.76 GB of weights, so 1.8 tokens per second needs 15.8 GB/s. We delivered
  1.27 on this path.** (Originally logged as 8.00 GB / 14.4 GB/s / 1.16 GB/s; the budget was
  an addition error, corrected at the top of this changelog.) That is roughly 3% of what a
  dual-channel desktop memory bus hands over, and it reframes every performance question this
  milestone has asked.

  The target does not require a GPU: 15.8 GB/s sits inside RAM bandwidth, and llama.cpp reaches
  1.8 tokens per second on CPU alone, which puts it near a third of the memory bus — what a
  well-written SIMD kernel achieves.

  It also explains a run of null results. At 1.27 GB/s nothing is near a bandwidth limit — not
  PCIe, not VRAM, not RAM — which is why the accelerator ceiling between 1.05 and 2.34 GB never
  correlated with anything, and why every I/O and cache change measured this milestone hit its
  prediction on bytes and moved the clock by nothing. The gap is in the kernels, not the device and
  not the memory hierarchy.

- **The forward pass was reading the Hyper-Connection mixing matrix transposed, and the model now
  answers `"La capitale d'Italia è **Roma**."`** The stray `R` that this changelog spent a milestone
  attributing to 1.5-bit quantization was one index expression.

  `deepseek_v4_hc_post` summed `combination[destination][source]` where the layout is
  `[source][destination]`. The Sinkhorn normalization ends with a pass over sources, so
  `Σ_src comb[dst][src] = 1` and that direction is a convex combination of the residual streams,
  which preserves magnitude. Summing the transpose sums a direction nothing normalized: slightly
  more than one, every layer, forty-three times.

  Verified against llama.cpp on the same checkpoint and the same tokens. All eight strongest logits
  now match in the same order, with differences of 0.01 to 0.06 — the f64-versus-f32 accumulation
  gap between two independent implementations, not a defect:

  | | here | llama.cpp |
  |---|---|---|
  | top 1–8 | 5, 201, 372, 1897, 7249, 13989, 6201, 271 | 5, 201, 372, 1897, 7249, 13989, 6201, 271 |

  The hidden state at forty-three layers fell from RMS 4,064 to 1,010, and the generated token that
  had been wrong is now 6091 `oma` — the one the reference scored nineteen points above ours.

- **The M6 operator fixture carried the same transposition**, which is why the defect survived a
  milestone. `tools/reference_deepseek_v4_operators.py` describes itself as generating vectors "from
  the pinned equations": it is a re-implementation written by this project, not the model's own code,
  and it indexed `combination[destination][source]` exactly as the C++ did. Two implementations
  agreeing proves nothing when one was written from the other's reading of the equations.

  Both are corrected and the fixture regenerated. The test failing was the correct outcome and the
  reason for changing it is external: llama.cpp, which shares no author with either.

  This is the fourth time this milestone that a check could not fail — after dyadic parity values
  that could not separate f32 from f64, a cost breakdown obtained by subtraction, and a hidden-state
  measurement taken after the normalization that fixes it.

- **The hidden state explodes with depth: RMS 6.8 at one layer, 4,064 at forty-three.**

  | layers | RMS | mean | min | max |
  |---:|---:|---:|---:|---:|
  | 1 | 6.82 | 0.05 | −25.9 | 21.5 |
  | 8 | 13.42 | 0.25 | −79.4 | 53.2 |
  | 24 | 79.81 | −0.62 | −288.9 | 254.0 |
  | 43 | **4,064.32** | −81.91 | −31,054 | 25,746 |

  Six hundred fold, and accelerating: about 1.10 per layer over the first eight, 1.12 through
  twenty-four, 1.23 through forty-three. A compounding per-layer factor slightly above one is what
  that curve is, and the final normalization then hides it — which is why the logits look merely
  wrong rather than obviously broken, and why forty-three layers of it produce a direction dominated
  by whichever components grew fastest.

  The suspect is the Hyper-Connection post step, whose weights are `2 * sigmoid(...)` and therefore
  range up to two. Applied where the reference does something else, that multiplies the residual by
  roughly 1.1 to 1.2 every layer, which is exactly the measured shape.

- **The first version of this measurement could not have found it**, and that is the third time this
  milestone has built a gate that cannot fail. `--dump-hidden` sampled the hidden state *after* the
  final RMS normalization, which fixes the norm by construction: it reported 19.72 at every depth,
  and that number is a fact about normalization rather than about the model. Sampling before it took
  the same four runs and showed a factor of six hundred.

  The pattern to watch for: a quantity that reports the same value however the thing under test
  behaves. Dyadic parity gates could not separate f32 from f64 and passed two wrong kernels; a cost
  breakdown obtained by subtraction charged the trunk for everything unattributed and redirected a
  milestone; and this reported a constant because a normalization had just imposed one.

- **The forward-pass defect reproduces on a single token**, which turns the debugging loop from a
  twelve-token generation into one forward pass. Given nothing but the beginning-of-sequence token,
  the two engines agree on almost nothing:

  | rank | llama.cpp | here |
  |---|---|---|
  | 1 | 5 — 16.67 | **20** — 14.78 |
  | 2 | 201 — 15.40 | **18** — 14.00 |
  | 3 | 372 — 15.36 | **21** — 13.97 |
  | 4 | 1897 — 15.09 | **22** — 13.78 |
  | 5 | 7249 — 14.54 | **26** — 13.70 |

  One token appears in both lists, and the rest do not. So this is not something that accumulates
  over a sequence, and not something the attention window, the indexer or the positional encoding
  can be blamed for: with one token there is no history to attend to and one position to encode.

  The shape of our ranking is the lead: **18, 19, 20, 21, 22, 26** — consecutive vocabulary
  positions. A ranking dominated by adjacent ids is what happens when the hidden state carries no
  direction worth speaking of and the logits fall back on the row statistics of the LM head. That
  points at a hidden state that is near-constant or near-zero by the time it reaches the head,
  rather than at a wrong-but-meaningful one.

  Also measured, and worth keeping straight: at twelve tokens both engines put 7529 first — ours
  more strongly, 35.46 against 33.71 — while positions two through seven are nearly disjoint. The
  argmax agreeing is what let this survive a full milestone unnoticed.

- **Routed experts share nothing, so there is no base to factor out of them.** The idea was that an
  expert might be a shared component plus a small individual part, in which case the shared part
  could stay resident and only the parts would stream — and the saving would grow with precision,
  making a high-precision checkpoint affordable on a machine that cannot hold it. It is a question
  about weights rather than about inference, so `deepseek-expert-redundancy` answers it without a
  correct forward pass and without downloading anything.

  | | storage | mean pairwise cosine | residual after the mean |
  |---|---|---:|---:|
  | layer 0 | IQ1_S | 0.00049 | 0.9371 |
  | layer 0 | IQ2_XXS | 0.00091 | 0.9367 |
  | layer 20 | IQ2_XXS | 0.00172 | 0.9361 |
  | *sixteen independent vectors* | | *0* | *0.9375* |

  Orthogonal, and the arithmetic settles it: the mean of N independent vectors carries 1/N of the
  energy, leaving 1 − 1/N behind. At N=16 that predicts 0.9375 and the measurement says 0.936. The
  6% that looked shared is the statistical residue of averaging sixteen unrelated things, not
  structure. The largest cosine found in any pair was 0.0059.

  The obvious objection — that 1.5-bit quantization might have destroyed a correlation the original
  weights have — is why the second and third rows exist: a different codebook at 2.06 bits and a
  different depth give the same answer.

  This closes expert deduplication, and with it the hope of making an FP8-class checkpoint cheap by
  compressing what it repeats. It repeats nothing. Which is also the healthy result: a mixture whose
  experts had overlapped would have been wasting its capacity. The levers that remain are amortizing
  one expert read across several tokens, and kernels that are currently seventeen times off the
  memory bandwidth.

  Cost of finding out: one afternoon and no download, against the 300 GB and the Safetensors
  execution path the other answer would have justified.

- **An external oracle exists at last, and it disagrees with us.** llama.cpp built from source against
  the same 82.54 GB checkpoint, given the same twelve prompt tokens, greedy, answers
  `"La capitale d'Italia è **Roma**."` This project answers `"La capitale d'Italia è **R Roma**."`

  The stray `R` is ours. It survived a quantization change from 1.5 to 2.06 bits, an exact
  detokenizer round-trip and a prefill-versus-decode comparison — three suspects cleared, and the
  only one left standing was the forward pass, which no internal test could indict because every
  internal test shares it.

  Eight generated tokens match exactly, including the `R` itself; the ninth differs, `oma` (6091)
  where we produce ` Roma` (38710). That looked like a near-tie decided differently — our matvecs
  accumulate in `double` by design and ggml's in `float` — so the logits were measured on both
  sides at that exact position, through a one-shot probe linked against llama.cpp's own library.

  | | llama.cpp | here |
  |---|---:|---:|
  | strongest | **6091 `oma`, 39.51** | 38710 ` Roma`, 24.83 |
  | margin over second | **17.20** | 1.85 |
  | logit of 38710 | 17.71 | 24.83 (chosen) |

  Not a tie, and not arithmetic. The reference is overwhelming — 39.5 with a seventeen-point gap —
  and puts our choice 21.8 below its own. We rank its answer sixth.

  The useful part is the shape rather than the ranking: our strongest logit is 24.8 where theirs is
  39.5, so our distribution is flat where theirs is sharp. A rounding difference cannot flatten a
  distribution. That points at the final hidden state being attenuated or misdirected rather than at
  any single expert or quantization path, which puts the output Hyper-Connection mixing, the final
  normalization and the LM head at the top of the list. Eight tokens agreed because those calls were
  not close either.

  Two things are independently confirmed by the same run: the tokenizer and chat template are exact
  — llama.cpp turns the templated prompt into the identical twelve ids — and llama.cpp supports both
  `deepseek4` and `dflash`, so the DSpark draft has a reference too.

- **llama.cpp is about fourteen times faster than this engine, on CPU, against our GPU path.**
  Same machine, same checkpoint, same twelve threads, no CUDA in their build: 545 ms per token
  against our 7,577. Model load took them ten seconds.

  This is the comparison this project had never made, and it reframes every performance number in
  this changelog. The kernels here are deliberately naive — one thread per row, no coalescing, a
  matvec measured seventeen times off the memory bandwidth — and that gap now has a name and a
  number rather than an assumption that it was small.

- **Trunk reloading is about a tenth of a token, not half of it, and this milestone spent a day
  optimising the wrong thing on the strength of that figure.** The 50% came from subtraction — all
  unaccounted time charged to the trunk — and subtraction cannot tell reading from computing.

  The correction came from a change that worked exactly as predicted and bought nothing. The
  streaming window now retains a fixed set of layers instead of discarding the oldest: under a
  cyclic reference pattern a window smaller than the cycle has a hit rate of zero however it picks
  its victim, but a hit rate of capacity/cycle if it simply stops picking. At eight layers that is
  393 reloads and 54.5 GB against 473 and 65.3 — the predicted 393 and ~54 to the digit — and the
  time per token moved 1.9%, inside the noise.

  Ten point eight fewer gigabytes read for no measurable time bounds what the reading costs: about
  7.7 GB/s, which is page-cache speed rather than NVMe speed. The same page cache that made direct
  I/O three times slower has been serving the trunk all along. Where a token actually goes, of
  75,769 ms:

  | | ms | share |
  |---|---:|---:|
  | GPU kernels | 23,724 | 31% |
  | expert loading | 15,014 | 20% |
  | expert compute | 13,601 | 18% |
  | everything else, trunk reads included | 23,430 | 31% |

  Expert bytes are 26.1 GB and are read once each, so that traffic is real; trunk bytes are the same
  6.39 GB re-read eleven times, and mostly are not. Future work belongs on the kernels and the
  experts.

  The retention change stays: fewer bytes for the same time, and at the default window of two it is
  a strict improvement over discarding the oldest. It carries no speed claim.

  It also shipped broken for one run, and the discipline caught it: the prediction was written down
  first, 5,805 reloads and 810 GB came back instead of 393 and 54, and the missing `transient` check
  was obvious within a second. A vaguer prediction would have filed it as another idea that did not
  work, and left the defect in.

- `deepseek-forward --gpu-limit N` pins the accelerator's ceiling to N mebibytes, so an A/B compares
  two runs rather than two machine states. The derived ceiling moves with whatever else the host is
  doing — it varied between 1.05 and 2.34 GB across one afternoon — and it is the largest single
  influence on the time per token, which invalidated three comparisons here before it was noticed.
  It only ever lowers: the Governor's limit still bounds it.

- **A busy host no longer switches the GPU off entirely.** The accelerator's ceiling is what the RAM
  limit leaves after the pass's own requirement and a 3 GiB reserve for allocations the preflight
  does not model, and on a loaded machine that subtraction reaches zero: a full-depth token cost
  55,788 ms against 8,405 with an idle device beside it. When the full reserve would leave nothing,
  a gigabyte is now held back instead and a gigabyte of ceiling granted.

  The reserve was measurable all along and nobody had measured it. `--json` now reports
  `peak_commit_bytes`, and on a 43-layer pass the process peaked at 5.12 GB against 2.56 charged and
  2.33 held by the accelerator — so the unmodelled part is **0.23 GB**, a thirteenth of the three
  gigabytes held back for it. The arithmetic closing to within a rounding error also confirms the
  1:1 host commit charge for VRAM that the ceiling is premised on.

  A gigabyte and no more, because more buys nothing. On one prompt a 512 MiB ceiling cost 22,781 ms
  per token and a 1.41 GB one 20,682 — a 9% spread, inside the noise — and six earlier runs between
  1.05 and 2.34 GB showed no trend. Only zero is different. This removes a cliff rather than
  climbing a slope, and claims the least commit that does so.

  Exercised rather than reasoned about: the tight path was provoked deliberately by lengthening the
  prompt until the requirement pushed the headroom below zero. At the resulting 1.41 GB ceiling the
  pass peaked at 4.27 GB against a 6.15 GB budget, and the 3.74 GB it charged was itself 0.78 GB
  more than it touched, so the safety margin survives twice over.

  One correction to how this was arrived at: the constant was first raised on a comparison between
  two different prompt lengths, whose prefill costs are not comparable. The same-prompt measurement
  contradicted it, and the constant went back down.

- The accelerator now evicts what will not be asked for again before what will. Recency cannot rank
  these matrices: the forward pass touches every one of them exactly once per token, so
  least-recently-used *is* arrival order and evicts what is needed soonest — the textbook
  pathological case for a cyclic scan. What separates them is recurrence. Always-active weights are
  multiplied by every token; a routed expert is top-6 of 256 and probably will not be picked again,
  which is why tripling the expert cache changed the bytes read by nothing. Roughly 774 expert
  uploads per pass were evicting the trunk, and `trunk_reloads` sat at exactly 473 — 43 layers times
  11 passes, every layer re-read on every pass — whether the ceiling was 1 GB or 6.

  **Measured, and it changes nothing.** At a ceiling pinned to 1 GiB the run returns `trunk_reloads`
  473, `trunk_bytes_read` 65.3 GB and `trunk_device_hits` 7,095 — the same three numbers as every
  other configuration tried, whether the victim is chosen by recency or by recurrence, whether the
  window streams layers or single matrices, and at every ceiling between 1.05 and 2.34 GB.

  The arithmetic says why, and it is not a bad choice of victim. A pass wants about 9 GB — 6.39 of
  trunk plus 2.5 of experts — against a cache this host cannot take past 2.3. At that ratio
  everything is evicted before it is next needed no matter which order it leaves in. 473 is exactly
  43 layers times 11 passes: every layer is re-read on every pass, always. Cache policy is closed as
  a lever until the ceiling approaches the working set, which needs host commit this machine does
  not have.

  The change is kept because it is right by construction — a routed expert genuinely will not be
  asked for again and an always-active matrix genuinely will — and because the pathology it removes
  returns the moment a ceiling can hold the trunk. It is credited with no gain.

  One number is worth suspicion rather than trust: `trunk_device_hits` is 7,095 in every run ever
  measured here. A count that does not move when the cache policy, the cache size and the fetch
  granularity all change is not measuring what the cache retains between passes.

- A GPU with no host budget now says so. The accelerator's ceiling is what the RAM limit leaves
  after the pass's own requirement and a 3 GiB reserve for what the preflight does not model, and
  on a busy machine that reaches zero: the model runs about fourfold slower and looks merely slow.
  The Governor is right to refuse — the host has no commit to spare — but the remedy belongs to the
  user and is host memory, not a faster disk or a better GPU, so the reason is now printed.

- Streaming one trunk matrix at a time instead of one layer at a time was implemented, measured and
  rejected. It is the obvious optimisation — a layer is 138 MB and a matrix about 5, and the window
  was re-reading the whole layer whenever any single matrix of it was absent from the device — and
  it saved **nothing**. Three paired runs on one machine state:

  | | ms/token | load_ms | trunk bytes read | reloads |
  |---|---:|---:|---:|---:|
  | whole layer | 8,337 | 1,555 | 65.3 GB | 473 |
  | one matrix | 9,951 | 83 | **65.3 GB** | 7,029 |
  | layer, metadata index | 8,979 / 8,768 | 79 / 95 | 65.3 GB | 473 |

  Identical bytes, to three significant figures, and 19% slower. The prediction had been 7 GB per
  token falling to 2.2, and it was wrong for a reason worth keeping: the device served 7,095 matrix
  requests without a read in *both* configurations — the same number — because what fits in VRAM is
  whatever is small. Roughly half the matrices are resident and they carry about a fifteenth of the
  bytes, so "read only what is missing" reads essentially the whole trunk anyway, in 7,029 governed
  reads of 10 MB where 473 sequential reads of 138 MB had done. Decline counts do not predict value;
  neither do hit counts. Bytes do, and nothing was counting them, which is why `trunk_bytes_read`
  now exists.

- The startup pass no longer reads 6.39 GB of trunk weights to learn shapes the catalog already
  knows: `load_ms` falls from 1,555 to 79. The mapping from a graph consumer to its checkpoint
  tensor used to exist only as control flow inside the binder loop, so the only way to learn one
  matrix's shape was to load all of them; it is data now, and both the binder and the index walk the
  same list.

  No speed is claimed. At ten tokens the total wall clock is unchanged or marginally worse, because
  that startup read was not waste — it was prefetch, warming the page cache that the first forward
  pass then read from. The same effect that made direct I/O three times slower. The cost moves later
  rather than disappearing, and amortises as a generation lengthens.

- Measured again that the accelerator's residency ceiling is not the binding constraint at these
  sizes: across four runs it varied between 1.05 and 2.06 GB with live host memory, and the time per
  token did not follow it. The ceiling mattered when it was the difference between 90% of matvecs
  declined and 28%; with every storage type now having a kernel, it no longer is.

- **IQ2_S decodes, and the UD-IQ2_XXS checkpoint starts.** One layer of forty-three stored its gate
  and up projections in a format this project had in no form, and two tensors out of 129 were enough
  to stop the model. It is not a variant of IQ2_XXS despite the name: 82-byte blocks rather than 66,
  a 1,024-entry grid rather than 256 — addressed by ten bits, two of which are lifted out of a
  per-group high-bit byte — two four-bit scales per group of 32 rather than one derived from a sign
  word, and explicit sign bytes rather than a seven-bit index whose eighth bit is a parity.

  Checked against ggml's own `dequantize_row_iq2_s`, compiled and run under WSL on the same bytes,
  rather than against this implementation's output: the FNV hash of all 1,536 decoded values matches
  at zero absolute error, as do the matvec results. The gate uses scales that vary in exponent and
  mantissa, including a subnormal, and a non-dyadic input, and it was verified capable of failing by
  corrupting one digit of the expected hash and watching the suite go red.

  The lesson that made this necessary stands: *dynamic* means the mixture is read from the file, not
  inferred from the name. `ffn_gate_exps` is IQ2_XXS on forty-two layers and IQ2_S on layer 26.

  The 90.86 GB checkpoint then ran all 43 layers and answered
  `"La capitale d'Italia è **R Roma**."` on its own stop token, at 7,800 ms per token with a 0.59 GB
  accelerator ceiling. 264 matvecs were declined and every one of them was type 22 — IQ2_S, the only
  format now without a device kernel — which is how we know the new decoder is on the path rather
  than merely compiled.

- **The stray `R` is not a quantization artefact.** This changelog said it was the ~1.5-bit
  quantization showing through, and a 2.06-bit checkpoint of the same model has now produced the
  identical string, stray `R` included. Two quantizations differing by half a bit per weight do not
  agree character for character by accident on a defect that belongs to one of them. The earlier
  explanation was a plausible story rather than a measured one.

  Two suspects have since been cleared. The detokenizer is exact: the generated ids
  `7529, 113213, 283, 9, 75993, 7269, 2619, 52, 38710, 42499` re-encode from the rendered text to
  themselves, and 52 is `R` while 38710 is ` Roma` — the model really did emit both. The incremental
  decode is not feeding the model stale state either: replaying the same prefix through prefill
  alone, via the new `--continue-ids`, predicts 38710 as well. Prefill and decode agree.

  What that leaves is the forward pass itself or the model's own behaviour, and this test cannot
  separate them because both paths share that forward pass. Distinguishing them needs the external
  oracle this project still does not have. A second prompt fails differently —
  `"La capitale della Francia è **Par**."` truncates rather than duplicating — so whatever it is, it
  is not a single fixed off-by-one.

- `deepseek-forward --json` now reports `generated_ids`, and `--continue-ids` appends exact tokens
  after the chat template so a partly generated answer can be replayed as a prompt. A decoded string
  cannot distinguish a model that chose an odd token from a decoder that mangled a reasonable one,
  and that distinction was most of the work above.

- Each quantization now carries its own pinned identity rather than one being privileged. A store
  holding UD-IQ2_XXS is recognised, verified and executed exactly as one holding UD-IQ1_S, and a
  store matching neither is verified against nothing. Two independent gates enforced this — the spec
  comparison in the catalog reader and the reference-config check in the graph — and both were
  extended rather than relaxed: the promise is that a checkpoint cannot quietly become a different
  one, which is per-quantization and stays intact.

- The GGUF reader accepts metadata under both `deepseek4.` and `dflash.` prefixes, so the DSpark
  draft describes itself to the same reader as the main checkpoint. The draft is downloaded
  (10.15 GB): architecture dflash, three blocks, `target_layers [41,42,43]`, 256 experts at top-6,
  routed experts in MXFP4 with everything else Q8_0, BF16 or F32 — every one of which already has a
  device kernel. Its catalog is still rejected because the validator expects a 43-layer tensor and
  expert mapping.

- Added the IQ2_XXS kernel, which the decline breakdown had ranked first. Declines at 43 layers fall
  from 5,089 to 2,749 and a token from about 13,226 to 11,739-11,954 ms, with zero absolute error
  against the reference. The two runs land within 1.8% of each other, tighter than anything else
  measured, so this clears the noise floor.

  On a full answer the gain is larger than the four-token runs showed. The same question that cost
  12,180 ms per token before this kernel now costs 8,736 - 28% - and returns the same
  `"La capitale d'Italia è **R Roma**."` on its own stop token. Longer generations amortise both the
  load and the VRAM warm-up, so short measurements understate what a real response costs; four
  tokens showed 10%.

  The gain is smaller than the decline count implied, and the standalone gate says why: IQ2_XXS
  costs 2.05 ms on twelve CPU threads against IQ1_S's 8.03 for the same shape, so it wins 2.3x on
  the device rather than 9x. Decline counts rank what to write next; they do not predict what each
  is worth.

  Grids now start on an eight-byte boundary rather than immediately after the weights. They are read
  with 64-bit loads, and block strides of 50 and 66 bytes end a matrix wherever they please - latent
  for IQ1_S, and it would have fired on the first shape whose byte count was not a multiple of eight.

- The model answered a question. With the checkpoint's chat template applied,
  `"Qual e la capitale d'Italia?"` produced `"La capitale d'Italia è **R Roma**."` and stopped on its
  own stop token rather than running out of budget — correct answer, correct Italian, assistant-style
  formatting, at 12,180 ms per token over 43 layers.

  Worth recording because the same model looks wrong without the template: `"La capitale d'Italia e"`
  continues as `" una grande citta cosmopol"`, which is a perfectly good completion of a sentence
  fragment and no answer at all. The distinction between completing text and answering a question is
  the template, not the weights. The stray `R` is the ~1.5-bit quantization showing through on a
  token the model was otherwise certain of.

- The accelerator evicts its least recently used matrix instead of refusing the next one, and a
  full-depth token went from 30,280 to 11,639 ms. It had been filling VRAM and then declining
  everything else for the rest of the run — 6,962 of 12,007 matvecs — on the implicit theory that a
  transfer is expensive. The arithmetic says otherwise: a routed expert costs about 0.29 ms to
  upload and 0.88 ms to multiply against roughly 8 ms on twelve CPU threads, so re-uploading it on
  every single use still wins about sevenfold. Refusing work to avoid a transfer that costs a
  seventh of the work is a bad trade. Pinning trunk matrices was the same mistake in reverse and was
  reverted: no matrix deserves permanent space.

- Added dense FP32 and BF16 matvec kernels, chosen by counting declines per storage type rather than
  by assumption. The expected next kernel was Q5_K; the measured ranking put FP32 third and BF16
  fifth, and both are far simpler — FP32 needs no decoding, BF16 is a 16-bit shift. Declines fell
  from 3,432 to 2,375 and both types left the table. No speed is claimed: repeated runs of identical
  code span 11,639 to 14,955 ms per token on this host, and the change does not clear that.

- Layer trunks are streamed through a bounded two-layer window instead of all staying resident, and
  a full-depth pass now needs 2.12 GB of host RAM rather than 7.62 GB. The floor stops scaling with
  depth: a deeper model no longer demands a bigger machine, and a machine with spare memory spends
  it on speed instead of on admission. It pays a second time through the accelerator, whose ceiling
  is derived from what the preflight leaves unspent — 1.80 GB before, 6.06 GB after — which took
  matvecs on the GPU from 1,490 to 4,617 and a token from 42,466 to 30,123 ms, with the same
  `" una grande"` and the same expert traffic. At 16 layers, where VRAM was never scarce, the same
  window is 33% slower: the reloads buy nothing when memory is not the binding constraint.

- Every storage type on the routed expert path now has a CUDA kernel, and a 16-layer generation
  dropped from 13,300 to 2,374 ms per token — 5.60× — producing the same text (`" guaranteedlyahi"`)
  and reading the same 3,963,551,744 bytes of expert payload as the CPU path. IQ3_XXS carried most of
  that: it is the most expensive format to decode on the CPU (9.76 ms against Q8_0's 3.04 for the
  same shape), so it gained the most on a device where the decoding hides behind memory latency. All
  three kernels — Q8_0, IQ1_S, IQ3_XXS — now match the reference at zero error rather than under a
  tolerance, because they accumulate in f64 in the reference's own order. At full depth the same
  kernels take a 43-layer generation from 48,970 to 37,680 ms per token, `complete_model: true`,
  output `" una grande"` on both paths. See the
  [CUDA expert kernel validation](docs/reports/M9_DEEPSEEK_CUDA_EXPERT_KERNEL_VALIDATION.md).

- The accelerator's VRAM residency is now capped by the host memory the preflight left unspent.
  A Windows device allocation carries host commit charge, so resident matrices compete for the same
  budget the weights were checked against: a 43-layer pass with the accelerator at its full 6 GB
  ceiling passed preflight and then died in the expert loader with `bad_alloc`, while the identical
  `--cpu-only` run completed. It now spends only the leftover headroom less a gigabyte for
  allocations the preflight does not model — 3.46 GB at full depth, where it settles for 1.30×
  because 9,457 of 12,007 matvecs are declined for want of VRAM. At 16 layers, where there is room,
  the same kernels give 5.60×. Streaming layer trunks is therefore not only about fitting on smaller
  machines; it is what would free the host memory the GPU needs.

- Fixed two defects that a passing parity gate was hiding, both found because the model's output text
  changed while the gate reported zero error.

  The accelerator indexed its resident VRAM matrices by the host address of the buffer they came
  from. Expert buffers are released as the pass advances, the allocator reuses those addresses, and a
  cache entry then served one expert's weights under another's name. The tell was that the output
  changed *between runs of the same command* — wrong arithmetic is wrong the same way every time.
  Matrices now carry a `cache_identity` derived from what they are (layer, expert and catalog tensor
  for experts; name and byte range otherwise), and no identity means no caching, so a caller that
  omits it loses speed rather than correctness.

  Separately, the parity gates were built from dyadic values — a fixed scale of 0.25, a delta of
  0.125 and a k/256 input — which makes f32 and f64 accumulation agree to the bit and let kernels
  accumulating in `fma.rn.f32` pass while the reference summed in `double`. The gates now vary the
  FP16 exponent per block, fill the mantissa, and use non-dyadic input.

- The GPU is now in the inference path. The graph offers each packed matvec to an optional
  accelerator and falls back to the CPU kernel on refusal, so matrices without a device kernel run
  unchanged. Q8_0 matrices stay resident in VRAM after their first upload — 2.90 GB across a
  24-layer pass — because a decode step revisits the same matrices and copying per launch would cost
  more than the kernel saves. The driver also caches its compiled module per context, holding its
  own context reference so the module cannot outlive it.

  At 24 layers with identical output text: 24,328 ms per token on CPU against 22,246–23,509 ms with
  2,728 matvecs accelerated and 4,899 declined. Far short of the kernel's own 5.4×, and the reason is
  measured: 2.65 ms per accelerated call against 0.558 ms in isolation, because each call still
  pushes and pops the context around three synchronous operations. The round trips, not the kernel,
  are what to fix next.

- Added the first quantized CUDA kernel: a Q8_0 matvec in hand-written PTX, loaded through the same
  toolkit-free driver path as the RMSNorm gate. On the reference RTX 4070 Super a 4096×4096 matvec
  runs in 0.5735 ms against 3.0305 ms for the twelve-threaded CPU kernel — 5.28× — with a maximum
  absolute error of 3.8147e-06 against a 2e-2 gate. Blocks accumulate in the reference's order but in
  FP32 rather than FP64, so this is a tolerance gate, not a parity gate. The kernel is deliberately
  naive: one thread per row, no coalescing, no warp reduction, matrix copied per call. Nothing is
  wired into the graph yet, so the model does not run faster because of it — this establishes the
  kernel and its numerics. `cuda-info --matvec-smoke` reports both timings.

- The complete DeepSeek V4 Flash 0731 model executed a forward pass on the reference PC: all 43
  layers, the checkpoint's own routers, 258 expert bundles and 1,770,651,648 bytes of expert payload
  — exactly the `no_cache_expert_bytes_per_token` figure derived from metadata at an earlier
  milestone. Token 0 produced 129,280 logits and next token 20; weights loaded in 3.97 s and the
  forward took 29.4 s with scalar CPU kernels. The process peaked at 7.17 GB while free host RAM
  never fell below 11.7 GB against the Governor's 8 GiB floor, and everything was released at the
  end. Raising depth in steps exposed two defects: the indexer's head-weight projection was bound
  outside the indexer namespace, which only surfaces at layer 2 where the first 4:1 compressor
  appears, and the reference layer ceiling was still sized for reduced fixtures at 16. See the
  [complete forward validation](docs/reports/M9_DEEPSEEK_COMPLETE_FORWARD_VALIDATION.md). This is one
  forward pass, not generation, and 29.4 s per token is correct rather than usable.

- A completed checkpoint verification now leaves a receipt, so the 82.54 GB SHA-256 pass that gates
  execution is paid once rather than before every prompt. The receipt records each file's size and
  last-write time beside the hash that was confirmed, keyed on the manifest's own hash; if both
  still match, the bytes on disk are the bytes that were verified. This trusts filesystem metadata,
  which is weaker than rehashing — a file edited in place preserving size and timestamp would pass —
  and it is bounded: any mismatch, a missing or malformed receipt, or a changed manifest forces the
  full verification again, and the receipt is written only after every file passes.

- Inference now uses half the machine's logical processors rather than a fixed six, which predated
  any threaded kernel. Combined with the threaded matvecs, generation went from 60,301 ms per token
  serial to 32,730 at six threads and 25,891 at twelve, with byte-identical output throughout.

- Packed matvec rows now split across the Governor's thread budget, taking generation from 60.3 s to
  32.7 s per token on the reference PC with byte-identical output. Rows are independent and each is
  still computed exactly as it would be serially, so the ggml parity fixtures are unaffected. The
  gain falls short of the thread count because only the matvecs are threaded — attention,
  normalization, RoPE and the Hyper-Connection paths remain serial, which is where the next
  speedup lives.

- **The complete model generates coherent text.** `"La capitale d'Italia e"` produced
  `" una grande citta cosmopol"` through all 43 layers of the official checkpoint — six tokens at
  60.3 s each, 16 GB of expert payload read from NVMe, 7.62 GB resident, free host RAM never below
  10.3 GB. The same prompt at 12 layers returns gibberish, which is the clearest evidence that the
  full-depth path is executing the real model rather than a fixture. `--max-tokens` routes through
  the reference decode loop with KV reuse; a minute per token is correct, not practical.

- The model now answers a text prompt. The checkpoint's vocabulary proved identical to the pinned
  tokenizer — 129,280 tokens index for index, all 127,741 merges — so `--prompt` encodes through the
  implementation already validated against Hugging Face, with `--chat` applying the checkpoint's own
  template. `"La capitale d'Italia e"` ran all 43 layers, resolved 1,069 expert bundles across
  7.34 GB of payload and predicted `" una"`. Each layer now releases its experts when the next layer
  asks for its own, so peak expert residency is two layers' worth instead of 43: the same prompt
  would otherwise have needed 13.6 GB of experts alone, and now the whole pass fits in 7.62 GB with
  free host RAM steady at 12 GB.

- Added `deepseek-forward`, which runs the real graph to a chosen depth straight from a complete
  checkpoint, and refuses in advance what will not fit. Its preflight sums the exact resident cost
  from the catalog — layer trunks, globals and the expert-cache ceiling — and compares it against the
  live safe RAM budget before allocating anything. On the reference PC a full 43-layer pass needs
  7,056,343,388 bytes against an 8,053,104,640-byte budget while 8 GiB of host RAM stays
  untouchable. Layer trunks and globals stream one complete tile per tensor under their own
  reservations, each layer keeps its own bounded expert cache so one layer cannot evict another's
  working set, and the process guard is sampled at every layer boundary. `--layers` keeps depth
  explicit, and the output states whether the pass was the complete model or a partial one.

- Executed the first real DeepSeek layer from official payload. `deepseek-layer-probe` relocates the
  verified catalog onto the imported artifacts, loads trunk and globals through their governed
  reservations and runs one complete layer with the checkpoint's own router selecting the experts.
  Token 0 executes experts 254, 222, 245, 200, 53 and 35, reads 38,928,384 expert bytes and produces
  129,280 logits in `[-16.0828, 15.4815]` in 1.07 s; two runs agree exactly. Q8_0, Q5_K, Q6_K, BF16,
  I32, IQ1_S, IQ3_XXS and Q4_K all execute together for the first time. This is one layer of 43 and
  the command says so — `complete_model: false` — so these logits are not the model's output.
  Relocation is partial by design: a tensor no artifact provided keeps an absent source and fails on
  open, which is what stops a token whose experts were never imported from executing against wrong
  bytes.

- Imported the first official DeepSeek V4 Flash 0731 weight payload and made the routing decision
  that justifies it. Hash-routed layers select experts by table lookup, so `blk.0.ffn_gate_tid2eid`
  — part of the trunk — determines the selection for a token without executing anything. The
  bounded importer fetched the 21-range / 138,867,160-byte layer-0 trunk, the new
  `deepseek_v4_resolve_hash_route` read that table through the governed rehashed range path, and
  token 0 resolved to experts 35, 53, 200, 222, 245 and 254. The resulting plan reports
  `official-tid2eid-verified`, which opened the previously refused `--scope selected-experts` and
  imported those six complete bundles (18 ranges, 38,928,384 bytes). Learned-router layers,
  out-of-vocabulary tokens, absent or mis-sized tables and out-of-range expert ids all fail closed.
  82.36 GB of the checkpoint remains undownloaded.

- Proved the imported trunk is consumable by the native graph. `deepseek-gguf-range-bind
  --load-tensor-set` rereads all 21 official ranges into one governed RAM buffer, rehashes every
  packed subspan and maps them onto their declared consumers: 11 matrices, 9 vectors and the routing
  table, 138,867,160 resident bytes in 371 ms on the reference PC. Two opt-in tests
  (`VASTLOOM_DEEPSEEK_V4_TRUNK_ARTIFACT`) pin both the route and this consumer contract. The layer
  is bound, not executed.

- Separated environment refusals from defects across the governed test suite. A Safety Governor
  refusal caused by host memory pressure was reported as a test failure, so an unrelated process
  holding 20 GiB made four tests fail on one run and five different tests fail on the next. Every
  registered test now carries `SKIP_RETURN_CODE 75`, the CLI's own `resource_exhausted` code, and
  CTest reports those contracts as skipped. `tests/host_pressure.py` gives the Python drivers the
  same outcome and recognises only the three refusals raised by `make_safe_runtime_policy` before
  any work starts; VRAM budget, allocation, parity and malformed-output failures still fail. The
  two zero-payload GGUF contracts moved from one positional `PASS_REGULAR_EXPRESSION` each to
  explicit per-key JSON comparison of the same 24 and 7 pinned values, so field reordering can no
  longer weaken them.

- Added `docs/VALIDATION_MATRIX.md`, which maps every documented claim to the test that defends it,
  the CMake variable that enables that test and the artifact it requires. It records what the
  13 always-on tests cover, which 11 tests are opt-in, and that neither real-model parity nor real
  GPU execution is re-checked by CI. `tools/fetch_deepseek_test_slices.py` restores the three
  pinned expert slices (6,488,064 bytes) behind an explicit `--allow-download` flag with per-range
  length and SHA-256 verification, so the M7 real-expert evidence is reproducible on a clean
  machine. The eight earlier alpha-era reports are now listed in `docs/reports/README.md` instead
  of remaining unreferenced.

- Added the first real CUDA math path without a toolkit dependency: one Governor-owned packed
  FP32 buffer, synchronous host/device copies and an embedded scalar PTX RMSNorm. The explicit
  `cuda-info --rmsnorm-smoke` gate matches the DeepSeek CPU reference within `3e-5` on the RTX
  4070 Super through native Windows and WSL, while deterministic faults cover copies, launch,
  release and reservation cleanup. This is not yet a CUDA model backend.

- Added a zero-payload complete-catalog contract for the pinned 0731 GGUF checkpoint. It validates
  six global consumers, all 11,008 routed bundles and 33,024 expert slices, then reconciles 1,328
  physical tensors, 34,223 logical descriptors, 6,986,088,796 always-active bytes,
  75,547,803,648 routed bytes and 82,533,892,444 known tensor bytes. Q4_K, IQ2_XXS and MXFP4
  receive layout-only accounting and remain explicitly non-executable without parity kernels.

- Pinned the official `deepseek-ai/DeepSeek-V4-Flash` checkpoint at immutable revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`; removed the unsupported provisional
  “0731” identity from the active planning manifest.
- Added a 19,404,391-byte metadata/header import that uses exact HTTP ranges, pinned
  SHA-256 values and atomic publication instead of downloading 159,617,149,040 bytes
  of weight shards.
- Added `deepseek-inspect` with exact validation for 46 shards, 69,187 tensors,
  Safetensors shapes/offsets, FP8/FP4 storage dtypes and routed-expert byte layout.
- Added native DeepSeek V4 Unicode pre-tokenization, its 129,280-token byte BPE,
  bounded added-token trie and basic system/user/assistant chat framing. Exact IDs match
  Hugging Face tokenizers 0.22.1 across committed multilingual reference vectors.
- Added bounded native DeepSeek reference equations and deterministic vectors for RMSNorm,
  learned/hash routing, routed SwiGLU, Hyper-Connections, base RoPE and selected-latent
  sparse attention. These are operator tests, not a DeepSeek forward or generation claim.
- Added a deterministic 147,136-byte graph exported from the SHA-256-pinned official
  `model.py`, a bounded native two-layer FP32 prefill and one checkpoint-shaped MTP block.
  Main and MTP logits, proposed tokens, learned/hash expert routes and raw/compressed attention
  indices pass the declared parity gate; full-checkpoint and speculative DeepSeek execution
  remain unsupported.
- Extended the fixture to 159,533 bytes with five official single-token decode steps. Native
  main/MTP logits, routes and raw/compressed attention indices match across circular-window and
  overlapping 4:1 cache updates; state mutation is transactional and aggregate-budgeted before
  allocation.
- Recorded, without downloading weights, the newer official DeepSeek V4 Flash 0731 release,
  its Unsloth GGUF profiles and Laguna S 2.1 as separate compatibility targets. None inherits
  the preview checkpoint's execution claims; the universal tensor/expert residency contract
  and per-adapter acceptance ladder are now explicit.
- Added a bounded native GGUF v3 reader with strict metadata/tensor/alignment/offset limits,
  exact GGML encoded-byte accounting, split-header catalogs and header-prefix mode. The CLI can
  inspect one file or build a metadata-only multi-shard DeepSeek catalog without reading payloads.
- Bound the pinned 82,539,237,792-byte 0731 `UD-IQ1_S` profile's 1,328 physical tensors to
  34,223 logical descriptors and 11,008 packed expert bundles from bounded headers. The derived
  1,770,651,648 no-cache expert bytes/token is a planning value, not measured SSD traffic;
  complete quantized execution remains unsupported.
- Added the exact 0731 `UD-IQ1_S` three-object importer and `deepseek-gguf-checkpoint` source
  gate. Metadata-only mode downloads and verifies 5,343,584 bytes; full source ranges require an
  immutable manifest, every complete-file SHA-256 and live single-tier expert admission.
  `--no-verify` remains metadata-only, the official 82.54 GB payload was not downloaded, and the
  CLI still reports `execution_supported=false` until a verified full-checkpoint layer/token
  consumer exists.
- Added `deepseek-gguf-probe-plan` and a native non-executing range-plan API. It validates one
  exact layer, exactly top-k unique caller-supplied expert IDs, complete W1/W2/W3 bundles,
  immutable source identities, bounded range counts and overflow-safe totals. Real layer-0
  metadata yields 21 trunk plus 18 expert ranges / 177,795,544 logical bytes; routing authority,
  full-payload presence and execution remain explicitly false.
- Corrected metadata-only logical sources to name the immutable complete GGUF shard objects rather
  than local `.header` snapshots. Loaders still reject metadata-only catalogs, so this preserves
  accurate future range identity without widening execution authority.
- Added `tools/import_hf_ranges.py` for explicit-authority partial payload acquisition. It accepts
  only the pinned DeepSeek 0731 profile, strictly validates a hash-verified plan, enforces selected
  byte/range/disk bounds, requires exact HTTP 206/content ranges, hashes every payload and
  atomically publishes a read-only `vastloom.range-artifact.v1` under `models/optimized`.
- Kept partial artifacts fail-closed: their manifest declares the complete checkpoint unverified
  and execution unsupported; caller-supplied expert/all scopes are rejected before network access.
  Offline Windows/Linux tests cover exact ranges, malformed plans, explicit authorization, wrong
  HTTP ranges, crash cleanup and immutable reuse. The real 138,867,160-byte trunk passed dry-run
  only and no official payload was downloaded.
- Preserved exact `blk.N.*` identity for all non-routed GGUF tensors and added deterministic
  per-layer trunk plans with a 64 MiB ceiling. The real 0731 metadata maps all 43 layers into
  1,193 tensor tiles; a new governed fixture loader covers immutable identity, Unicode paths,
  budget refusal, cancellation, short reads, leases and zero-reservation cleanup. No official
  weight payload or complete DeepSeek layer was executed.
- Added bounded scalar Q8_0, Q5_K, Q6_K, BF16 and F32 decode/matvec kernels plus a separate I32
  router-table decoder. Synthetic raw blocks match the pinned ggml oracle, caller output is
  transactional through bounded scratch, and the real 0731 metadata now reports native storage
  handling for all 1,193 layer-trunk tensors / 6,390,088,008 encoded bytes. This is storage/math
  coverage only for the official payload: the 82.54 GB payload remains absent and
  `execution_supported=false`.
- Added deterministic ggml-compatible Q8_0 reference encoding and connected packed matrix/I32
  storage directly to the reduced DeepSeek graph. Its 11 trunk/shared matrices use Q8_0, routed
  W1/W3 now use eight IQ1_S matrices and routed W2 uses four IQ3_XXS matrices. All 23 matrices in
  main layer 0 execute through transactional storage dispatch; prefill and incremental main/MTP
  logits, routes and attention indices match an independently dequantized oracle. Official
  payload execution and generation remain open.
- Added immutable resident byte buffers shared by the bounded trunk and expert loaders. A
  file-backed reduced DeepSeek layer now binds 12 complete trunk tiles and four indivisible expert
  bundles directly into the graph, retains exactly 8,180 governed mixed-format payload bytes
  across prefill and decode, blocks eviction while referenced, and returns all reservations to
  zero on release. Official payload execution remains open.
- Split the reduced DeepSeek MoE graph at its authoritative router boundary and added a reusable
  bounded selected-expert cache. The graph deduplicates routed IDs, requests only complete selected
  bundles, executes their retained packed matrices and records hit/miss/logical-byte telemetry. A
  two-entry synchronous RAM LRU loads experts 2/3 on the first token, reuses them on the next and
  replaces them with 0/1 on the third while matching the quantized oracle. Oversized batches,
  malformed providers, insufficient budgets and eviction of in-flight bundles fail closed. This is
  a reduced fixture path, not official-payload DeepSeek generation or an SSD-speed result.
- Added a bounded scalar IQ1_S dequantization/matvec reference kernel derived from pinned ggml
  equations and licensed codebook data. Synthetic vectors and one exact 1,638,400-byte real
  DeepSeek 0731 expert-gate slice match the upstream oracle bit for bit across all 2,048 rows.
- Added exact scalar IQ3_XXS decode/matvec plus a transactional IQ1_S gate/up, FP32 SwiGLU and
  IQ3_XXS down expert path. Three pinned real ranges totaling 6,488,064 bytes now pass through a
  one-worker governed packed-RAM load and match the ggml oracle bit for bit across all 4,096 final
  outputs; complete official-payload layer/token execution and model generation remain open.
- Bound canonical 0731 GGUF expert identities directly from the executable logical catalog into
  strict W1/W2/W3 cache descriptors. The same 6,488,064-byte real IQ1_S/IQ3_XXS expert now passes
  through the production router-first one-entry RAM cache: the first resolution is one exact miss,
  the second is a zero-read hit, retained leases block premature eviction, malformed mixed-format
  descriptors publish no output and every Governor reservation returns to zero. This connects one
  real selected expert to the cache/kernel boundary, not a complete DeepSeek layer or generation.
- Connected selected DeepSeek misses to a bounded asynchronous batch/acquire scheduler. The
  reduced graph submits every deduplicated complete bundle, waits only at that expert's fixed
  compute turn and preserves numerical reduction order while recording load, wait, compute and
  read-service/compute-intersection timing. A controlled fixture proves positive overlap with
  exact oracle parity; the real 6,488,064-byte expert passes the same future/cache/kernel boundary.
  One active batch per cache, full-request queue capacity, drain-on-abandon and exact successful
  wasted-prefetch accounting keep this correctness slice bounded. This is not a physical-NVMe,
  complete-layer or throughput result.
- Made the controlled overlap regression deterministic by explicitly coordinating the second
  fixture read-service start before its bounded 50 ms service window. This changes test timing
  only, not the production scheduler; the native suite then passed 50/50 repetitions and TSan.
- Extended the complete reduced layer/token gate to the real 0731 expert format combination:
  Q8_0 trunk/shared tensors, I32 hash routing, IQ1_S W1/W3 and IQ3_XXS W2. The 8,180-byte
  file-backed fixture produces four 1,384-byte indivisible expert bundles; synchronous and
  asynchronous cache execution matches the dequantized oracle for prefill/decode, and selected IQ
  truncation plus malformed W2 format fail transactionally. These are deterministic reduced
  bytes, not official checkpoint payload or a performance result.
- Documented the tested platform boundary explicitly: Windows/MSVC and Linux GCC/Clang CPU paths
  are validated, while macOS and Metal remain unvalidated targets.
- Added a public reference-PC admission gate: a model must fit one complete expert plus the
  required active block, KV, scratch and staging inside live post-reserve RAM/VRAM budgets. Kimi K3
  is now capacity-study-only on the 32 GiB RAM / 12 GiB VRAM machine unless an exact profile passes
  that gate.
- Implemented that gate in the native planner: strict MoE manifests now require complete expert,
  layer, top-k, context and minimum-working-set metadata; scratch is an explicit placement; and
  each plan names the real RAM or VRAM tier that can contain one complete expert. The Resource
  Governor now enforces one aggregate ceiling across RAM destinations, staging and scratch, with
  deterministic acceptance, pressure, rejection and overflow tests.
- Replaced repeated GGUF metadata `seekg` calls with verified sequential discards. The same real
  5.26 MB header on Windows-mounted WSL storage improved from 37.42 s to 0.11 s while preserving
  strict bounded/truncation behavior.
- Made capacity-sensitive tests deterministic without weakening production disk reserves and
  gated CodeQL/Dependency Review until the repository is public or GitHub Advanced Security is
  explicitly enabled through the `VASTLOOM_ENABLE_GHAS` repository variable.
- Fixed CLI path construction so Windows wide-character arguments remain Unicode-native through
  `std::filesystem`; an accented GGUF filename is now covered end to end.
- Added the shared logical tensor catalog and bound all 69,187 pinned DeepSeek preview tensors
  to role, shape, dtype and immutable source ranges. Its 67,584 routed tensors form 11,264
  main/MTP expert bundles whose six tensors coalesce into exactly two ranges per expert; no
  payload read or execution support is claimed.
- Added the governed multitier residency state machine. RAM plus bounded staging and later VRAM
  destinations are reserved before transition publication; failed compound reservations roll
  back, while pins and RAII use leases prevent unsafe eviction. The state-only slice allocates no
  tensor buffer and makes no new inference claim.
- Added the first governed packed-expert load transaction: canonical source ranges are read into a
  reserved compact RAM buffer, every expert tensor receives an exact packed slice and an RAII use
  lease prevents eviction until the bytes are destroyed. The synchronous fixture baseline is not
  a DeepSeek or physical-NVMe performance claim.
- Added a shared bounded asynchronous expert-load pool with one to four fixed workers, a queue of
  at most 64 requests, pre-enqueue RAM/staging reservations, exclusive transfer leases, safe
  queue-full rollback, shutdown cancellation and queue/service/total telemetry. This enables a
  future compute/I/O overlap path but does not itself prove DeepSeek throughput.
- Added an explicit `--allow-large-download` gate for checkpoint imports containing at
  least 20 GiB of weight payload. DeepSeek execution remains unsupported and is reported
  as such by the CLI.
- Added path-scoped device-I/O snapshots: Windows maps the checkpoint volume to
  `PhysicalDriveN` and samples `IOCTL_DISK_PERFORMANCE`; Linux accepts non-virtual
  `/sys/dev/block/.../stat` mappings and otherwise reports the counter unavailable.
- Added the fail-closed `direct-unbuffered-v1` cache-bypass protocol for bounded OLMoE
  expert reads. It requires async no-cache execution, a trustworthy device counter and
  measured device bytes at least equal to logical expert bytes; it never flushes the
  global operating-system file cache.
- Validated three repeatable Windows cache-bypass runs with identical token `47111`,
  1,610,612,736 logical bytes and 1,610,743,808 whole-device bytes per run.
- Extended native OLMoE traces from 12 to 19 fields with submission, queue, ready,
  compute-window, route-order and reduction-order telemetry while retaining all earlier
  trace formats.
- Added a bounded trace replay that compares fixed-route execution with theoretical
  ready-expert compute order while preserving the authoritative reduction order.
- Added opt-in two-stage OLMoE reads: contiguous 8 MiB gate/up plus 4 MiB down requests,
  two workers, bounded queue/staging reservations, exact byte accounting and down-read
  versus gate/up-compute overlap telemetry.
- Validated staged reads against the pinned Transformers logits/routes and measured a
  46.0% median warm buffered-pipeline improvement over whole 12 MiB reads on three paired
  WSL runs. This is not a physical-NVMe or DeepSeek throughput claim.

- Qwen3.8 performance groundwork (commit `02fa840`): resident F32 trunks, a bounded raw
  Q8_0 payload cache with graceful degradation under commit pressure, persistent shard
  streams and parallel hot kernels; greedy per-token wall time went from 33.75 s to
  2.7 s/token in the forward phase before the later Q11i Q8_0-native rework.
## 0.4.0-alpha.4 (rebrand) - 2026-08-09

- Renamed the public project, CLI, C++ namespace, headers, CMake targets and current
  documentation to **Vastloom** with the slogan “Run giant models. Keep your machine yours.”
- Preserved read compatibility for legacy manifests, imported-checkpoint metadata and
  model-store markers so existing verified model downloads remain immutable and reusable.
- Added Apache-2.0 licensing, community governance, support and security policies,
  contribution templates, CodeQL, dependency review and release packaging automation.
- Added a purpose-built README cover and two-to-one social preview, plus public brand and
  release documentation.
- Regenerated the bundled synthetic fixture under the Vastloom model identity and retained
  the original native binary and heartbeat wire magic for format compatibility.

Historical entries below retain the project name and command spelling used by those releases.

## 0.4.0-alpha.3 — 2026-08-09

- Added cross-platform process-I/O snapshots around the real OLMoE forward path.
- Exposed Windows read operation/transfer deltas and Linux logical plus storage-layer
  read deltas while preserving their different semantics in text and JSON output.
- Replaced fragmented Windows asynchronous stream reads with Unicode-safe bounded
  `CreateFileW`/`ReadFile` ranges; one expert now requires one 12 MiB operation.
- Added exact unique-expert reuse-distance telemetry with cold/reused access counts,
  reuse ratio, mean, P50, P95 and maximum distance.
- Added Unicode-path, counter-semantics, operation-fragmentation and real repeated-route
  regression tests.

This slice measures process-visible I/O and routing locality without claiming physical
NVMe traffic. Buffered cache state, cold-cache behavior, device counters, OLMoE decoding,
CUDA and DeepSeek remain outside the validated scope.

## 0.4.0-alpha.2 — 2026-08-09

- Connected the real OLMoE expert path to the reservation-backed bounded I/O pool.
- Added opt-in synchronous/asynchronous selection, one-worker prefetch depth one or two,
  12 MiB request bounds, cancellation and per-request timeouts capped at 60 seconds.
- Coalesced each pinned expert's three validated contiguous BF16 tensors into one bounded
  read while preserving the native gate/up/down layout and reference logits.
- Added request queue/service/total timing plus expert prefetch count, foreground wait
  and measured read-service/CPU overlap telemetry.
- Extended atomic OLMoE CSV traces from nine to twelve fields while retaining compatibility
  with four-field baselines and nine-field legacy traces.
- Added deterministic I/O worker, short-read, allocation, cancellation and timeout fault
  injection, including real-checkpoint fail-closed integration coverage.
- Retained synchronous I/O as the default and capped the asynchronous OLMoE adapter to
  one worker, two outstanding prefetch slots and Resource Governor staging reservations.

This slice validates bounded buffered read/CPU overlap without changing model output. It
does not claim physical NVMe bytes, cold-cache disk throughput, direct I/O, OLMoE text
generation, CUDA or DeepSeek support.

## 0.4.0-alpha.1 — 2026-08-09

- Pinned and fully verified `allenai/OLMoE-1B-7B-0924-Instruct` revision
  `7f1c97f440f06ce36705e4f2b843edb5925f4498` with three immutable BF16 shards.
- Added exact OLMoE configuration, tokenizer, chat-template and 3,219-tensor layout
  validation without requiring tensor payloads to reside in RAM.
- Added a native CPU OLMoE reference graph with RMSNorm, Q/K normalization, RoPE,
  causal attention, authoritative softmax top-8 routing and weighted SwiGLU experts.
- Added explicit NVMe-to-RAM expert streaming: the 953,421,824-byte always-active
  trunk remains in reserved RAM while one 12 MiB expert is read at a time.
- Added bounded no-cache, LRU, LFU and static-hot expert-cache policies, with a hard
  two-GiB cache ceiling and Resource Governor reservations before allocation.
- Added `moe-inspect` and `moe-logits`, atomic route telemetry, exact expert byte/I/O
  accounting, timeout/Ctrl+C checks and metadata-only execution refusal.
- Added an independently bounded Transformers 4.45.2 oracle and committed one- and
  two-token fixtures for logits, greedy token, route IDs and routing weights.
- Extended trace simulation to accept native telemetry and compare no-cache, LRU,
  LFU and static-hot policies.

This milestone proves correct real MoE routing and synchronous NVMe-to-RAM expert
streaming on CPU under a host-derived memory cap. It does not yet provide OLMoE text
generation/chat, CUDA, asynchronous I/O/compute overlap or a DeepSeek adapter.

## 0.3.0-alpha.1 — 2026-08-09

- Added a bounded JSON parser and defensive Safetensors catalog/reader.
- Added a 300 GiB-reserve model store, immutable atomic Hugging Face importer,
  and host-derived real-model runtime policy.
- Added Windows process memory Job Object support and live commit/headroom checks
  for real-model execution paths.
- Added `store-info`, `runtime-policy`, and Safetensors-aware `inspect` commands.
- Added the first genuinely pretrained model adapter, pinned to
  `HuggingFaceTB/SmolLM2-135M-Instruct` revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Added the model's GPT-2 byte-level BPE tokenizer, exact chat template, BF16
  Safetensors loader, Llama CPU reference kernels, grouped-query attention,
  incremental KV cache, greedy/seeded generation and persistent chat history.
- Added trusted Transformers parity fixtures for tokenization, logits, greedy output
  and multi-turn chat, plus a native real-model self-test.
- Added bounded process-memory enforcement, six-thread maximum execution,
  cancellation and full-checkpoint verification by default for the real path.

This milestone proves one real published dense checkpoint on the native CPU path. It
does not yet include MoE execution, NVMe expert streaming or CUDA.

## 0.2.0-alpha.2 — 2026-08-09

- Added the first native end-to-end token-generation path.
- Added the bounded `SGTMV001` TinyTransformer model format and strict loader.
- Added byte-level prompt encoding, CPU LayerNorm/attention/MLP kernels and incremental
  KV caching.
- Added greedy decoding and deterministic seeded top-k/temperature sampling.
- Added `stargate chat`, `stargate model-info` and `stargate self-test`.
- Added interactive streaming, one-shot JSON output, generation timeouts, token limits,
  system-prompt override and response statistics.
- Added the 358.91 KiB `stargate/tinychat-v1` validation model, exact manifest and five
  deterministic response vectors.
- Added a reproducible PyTorch training/export tool that atomically publishes the
  model, vectors and SHA-256 manifest only after all probes pass.
- Added end-to-end CLI integration tests, integrity-failure tests, malformed-model
  tests, deterministic-sampling tests and additional Resource Governor checks.
- Added explicit generation finish reasons and valid-Unicode repair for arbitrary sampled
  bytes in JSON output.
- Added manifest-to-binary metadata cross-checks for model id, layer count and context
  length.
- Added concurrent generation coverage and ThreadSanitizer validation of the shared model
  path.
- Added a UTF-8 Windows `wmain` entry point and UTF-8 console configuration for accented
  CLI prompts.
- Added integration coverage for direct `.sgtm` loading and explicit quick manifest mode.
- Added CMake installation of the bundled validation model under `share/stargate/models`.
- Regenerated the committed fixture from the current exporter and verified byte-identical
  output across independent training runs.
- Fixed model/governor destruction order so weight reservations are released before
  the owning governor is destroyed.
- Added UTF-8 MSVC compilation and expanded warning-as-error/sanitizer validation.

This release validates the native CPU inference pipeline. It does not yet include a
DeepSeek/MiniMax/Kimi adapter, GPU kernels or NVMe-backed generation.

## 0.1.0-alpha.1 — 2026-08-08

- Added C++20/CMake project foundation.
- Added hardware profiler and JSON reporting.
- Added conservative hardware-adaptive execution planner.
- Added atomic resource reservations and safety state machine.
- Added bounded asynchronous I/O with reservation-backed buffers.
- Added strict model manifest, SHA-256 and path/symlink confinement.
- Added immutable model-store layout command.
- Added atomic heartbeat and independent watchdog.
- Added MoE trace simulator with LRU/static-hot policies and bytes/token metrics.
- Added unit tests, warning-as-error build, ASan/UBSan and CI definitions.

No model inference kernels are included in this release.
