# Roadmap and acceptance gates

> **Closed as a general programme (October 2026); scoped DeepSeek refocus.**
> The summer-2026 Qwen programme below is closed: Q0–Q11j are its completed
> evidence record, and the Qwen open items (Q11k–Q11n, Q12–Q13) are
> comparison/validation-only, preserved as history rather than scheduled work;
> Q14 (capability transfer) remains a design-only compass. The repository
> remains a working record: every completed milestone links to the report or
> test that defends it.
>
> **October 2026 refocus:** the primary execution target returns to **DeepSeek V4
> Flash 0731** (`UD-IQ2_XXS`) via bounded exact range slices, specified as D0–D4
> at the end of this file. D1, D2a, D2b, D3 (single- and multi-token scope) and the D4 design
> with closed decisions are measured. This concludes the research record: the VRAM *payload*
> tier, ratio-128 and learned-router/full-depth execution are not planned and need new
> subsystems or the full checkpoint. The Qwen track is comparison/validation-only.
>
> **Post-conclusion update `0.5.0-alpha.1` (2026-10-20):** a maintenance and
> serve-safety update on the dense SmolLM2 path (JSON-object constrained
> decoding, control-token rejection, UTF-8-safe streaming) is recorded at the
> end of this file. The maintenance slice itself reopens nothing: it advances no
> large-model milestone.

Vastloom advances only when each milestone has a measurable correctness and safety gate.
A command existing in the CLI is not enough; it must run against a validated model path.

> **Architectural pivot (2026-08-26):** the DeepSeek V4 Flash 0731 track (M0–M13) is frozen
> in `docs/archive/deepseek_0731/`. The new track is **Qwen3.8-Flash-Next** (Qwen4
> architecture preview). Roadmap Q0–Q14 below.
>
> **October 2026 refocus:** the Qwen track is now **comparison/validation-only** — Q0–Q11j
> stay as completed evidence, Q11k–Q13 keep their history but no new Qwen features are
> planned, and Q14 stays design-only. The primary execution target returns to DeepSeek V4
> Flash 0731 in the reduced-payload form specified by D0–D4 at the end of this file. The
> archived M0–M13 record stays frozen; D0–D4 is a new range-slice sequence, not a resumption
> of it.

---

## Q0 - Skeleton and Qwen3.8 budget gate - complete

- [x] Verify llama.cpp mainline support for `qwen4_exp`/DeltaNet/QSA/n-gram
  (commit `84e908c`: no dedicated model; only `qwen3next` and `qwen35` with DeltaNet +
  hybrid full-attention; QSA / n-gram / Gated Residual are still to be implemented
  upstream as well).
- [x] Skeleton `include/vastloom/inference/qwen3_8_flash_next.hpp` with
  `Qwen38FlashNextConfig`, `Qwen38FlashNextInspection`, config.json parser pinned at
  2026-08-26.
- [x] Always-on test `test_qwen38_flash_next_config` validating parse, strict invariants
  and rejection of a wrong model_type, without downloading the real checkpoint.
- [x] Paper budget estimate in `docs/planning/QWEN38_BUDGET_ESTIMATE.md` (Unsloth FP8 as
  the binding choice; no GGUF IQ quant closes both the quality and the RAM gate).

**Gate:** green build with `VASTLOOM_WARNINGS_AS_ERRORS=ON`, `vastloom_tests` ALL PASSED,
ctest green. No real weights downloaded at this point.

## Q1 - Safetensors catalog reader for Qwen3.8-Flash-Next FP8 - complete

- [x] Remote header inspection and `model.safetensors.index.json` for
  `unsloth/Qwen3.8-Flash-Next-FP8`: 131 safetensors shards, 152,089 tensors total,
  range ~840 MB – 1.2 GB per shard, ~170 GB total on disk.
- [x] POC downloaded (`model-00001-of-00131.safetensors` ~1 GB) for real byte-level
  inspection of the safetensors header.
- [x] Architectural findings from the real POC:
  - MoE experts in **FP8 block-scale** (`weight` F8_E4M3 + `weight_scale_inv` BF16,
    `quant_method=fp8`, `weight_block_size=[128,128]`, scale shape `[5,20]` for
    `gate_proj` `[640,2560]`).
  - Global **hyper_connection_mixer** = Qwen's Gated Residual (4 branches,
    `hc_count=4`, `hc_lowrank=320`).
  - Per-layer **attn_hyper_connection** = a local instance of the Gated Residual.
  - **linear_attn** = Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `conv1d`,
    `A_log`, `dt_bias`, `norm`, `out_proj`).
  - **layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}** = PLE n-gram table
    on layer 1 (0-indexed, corresponding to `ple_layer_ids=[2]` 1-indexed).
  - 73,728 non-scale expert tensors + 75,264 scale_inv (FP8) = 147,456 expert
    tensors across 48 layers.
- [x] Native C++20 safetensors header parser in
  `read_qwen38_safetensors_header()` with automatic attachment of the `weight` +
  `weight_scale_inv` pairs.
- [x] `model.safetensors.index.json` parser in
  `inspect_qwen38_flash_next_safetensors()` with FP8 block-scale counting,
  n-gram tensor count, global/per-layer Gated Residual presence.
- [x] Valid safetensors mock shard (3.6 KiB) in `tests/fixtures/qwen38_flash_next/`
  with FP8 block-scale experts + DeltaNet tensors + global hyper_connection_mixer +
  attn_hyper_connection.
- [x] Always-on test `test_qwen38_flash_next_safetensors` exercising the parser on the
  mock and verifying counts, dtypes, scale_inv attachment, data_offsets integrity
  against header_end and total_bytes.

**Gate:** green release build, `vastloom_tests` ALL PASSED, `ctest --preset release`
13/14 (1 intentional skip). No download of the full 125B (170 GB out of scale for a
POC). The "POC on 27B FP8" option was rejected after inspection: the 27B is a
**different** model (`qwen3_5`, 64 layers, dense, no MoE, no n-gram, no Gated Residual)
and would not have validated the three new primitives.

## Q2 - Reference operators for Qwen3.8-Flash-Next FP8 - complete

> **Upstream blocked:** llama.cpp `d0132a6` (2026-08-27) has no `qwen4_exp` yet.
> No `src/models/qwen4*.cpp`, no QSA / n-gram / Gated Residual support upstream.
> The original gate ("5 oracle tokens pinned-ggml via llama.cpp") is therefore
> **not executable** and Q2 was redefined on a self-pinned reference.

- [x] Pure reference operators implemented in
  `include/vastloom/inference/qwen3_8_flash_next_reference.hpp` /
  `src/inference/qwen3_8_flash_next_reference.cpp`:
  - `qwen38_fp8_e4m3_to_f32` (OCP E4M3, exp 15 => NaN, subnormal 2^-6),
  - `qwen38_bf16_to_f32` / `qwen38_f32_to_bf16` (bit-exact),
  - `qwen38_decode_bf16` (payload -> F32 row-major),
  - `qwen38_decode_fp8_block_scale` (F8_E4M3 weight + BF16 per-block scale
    `[128,128]`, scale shape `[ceil(R/128), ceil(C/128)]`),
  - `qwen38_rms_norm` (reference RMSNorm).
- [x] Mock shard regenerated with `weight_scale_inv` in **BF16** (shape `[1,1]`,
  value `1.0` => bytes `[0x80,0x3F]`) and deterministic non-zero payload,
  with correct payload-relative (0-based) `data_offsets`.
- [x] Always-on test `test_qwen38_flash_next_reference_operators` exercising:
  - FP8 E4M3 scalar smoke (0x00=0, 0x38=1.0, 0xB8=-1.0, 0x3C=1.5, 0x01 subnormal
    1/512, 0x7F/0xFF=NaN),
  - BF16 round-trip (0,1,-1,0.5,2,448,-448),
  - BF16 decode 2x2,
  - FP8 block-scale 4x4 with 2x2 blocks (uniform and per-block scale),
  - 3-element RMSNorm,
  - real payload read from the mock shard (`A_log` BF16 [4]).
- [x] Remote header verification of SHARD 2 (`model-00002-of-00131`): 282,800-byte
  header, `gate_proj.weight` F8_E4M3 `[640,2560]` + `gate_proj.weight_scale_inv` BF16
  `[5,20]`, 2,048 tensors per shard (1,024 BF16 + 1,024 F8_E4M3).

**Gate:** `vastloom_tests` ALL PASSED, `ctest --preset release` 13/14 (1 skipped),
`compileall` green. No download of the full 125B beyond POC shard 1.

**Next step (Q4):** range importer + safety governor for the FP8 payload (exact read of
`weight` + `weight_scale_inv` with SHA-256 validation and a bounded transaction, in
preparation for the 125B download).

## Q3 - Tensor & expert catalog for Qwen3.8-Flash-Next FP8 - complete

- [x] Extended `Qwen38InspectionResult` with a per-layer breakdown:
  `Qwen38LayerSummary` (total, linear_attn 9 / self_attn 9, attn_hyper 4,
  mlp_hyper 4, mlp_gate 1, mlp_shared_gate 1, mlp_shared_expert 3,
  mlp_experts 3072 (1536 F8_E4M3 weights + 1536 BF16 scales), ple 0/138).
  The `qwen38_summarize_layers(weight_map)` function classifies each entry by
  layer and type.
- [x] Added `Qwen38ExpectedCatalog` with pinned constants for the 2026-08-26 release:
  `total_tensors=152089`, `total_shards=131`,
  `language_tensors=148650` (47×3094 + 3232), `per_layer 3094/3232`,
  `vision=333`, `mtp=3101`, `global=5`, `ngram=128`, `fp8_scale_inv=75264`.
  Validation in `qwen38_validate_catalog(result)` rejecting any catalog that does
  not close the gates (total, shards, per-layer, ngram, scale).
- [x] Automatic population of `layers`, `vision_tensors`, `mtp_tensors`,
  `global_tensors` inside `inspect_qwen38_flash_next_safetensors` via
  `qwen38_summarize_layers`.
- [x] Findings confirmed by the real POC (already in Q1/Q2): 48 layers with 3,094
  tensors each (layer 1 = 3,232 with PLE), 6 global hyper, 196+196 per-layer hyper,
  333 vision, 3,101 MTP.
- [x] Always-on test `test_qwen38_catalog_validation`: builds a synthetic full catalog
  of 152,089 entries in memory with the exact distribution (9 linear/self, 8 hyper,
  5 gate/shared, 3,072 experts as 1,536 weight + 1,536 scale, 128 n-gram + 10 ple
  proj for layer 1, 333 vision, 3,101 MTP, 5 global) and verifies that `summarize`
  yields 48 layers with `total 3094/3232` and that `validate` passes; it also
  verifies that the mock fixture (25 tensors) is **rejected** by the validator
  (a strictness proof).

**Gate:** `vastloom_tests` ALL PASSED, `ctest --preset release` 13/14 (1 skipped),
`compileall` green. No 17 MB file committed: the full catalog is an in-memory synthetic
for the gate, while the mock fixture stays at 3.6 KiB.

## Q4 - Range importer + safety governor - complete

- [x] `qwen38_read_tensor_payload(shard, header, tensor_name)`: reads a tensor's exact
  payload from the safetensors file with bounds checks against `json_header_bytes` and
  `total_bytes`. If the tensor has a `weight_scale_inv` companion, reads it together
  and returns it separately.
- [x] SHA-256 hash of the payload computed on the fly (`Sha256`), writing nothing.
- [x] `qwen38_import_tensor_transaction(request, header, governor)`: three-object
  transaction — (1) source range from the shard (tensor + scale), (2) expected
  SHA-256 verified **before** any write, (3) bounded write to temp + atomic rename in
  the destination dir. The Governor must admit the aggregate (weight+scale) as a RAM
  reservation.
- [x] Atomic rejections: out-of-bounds range, hash mismatch, destination collision,
  governor refusal (`resource_exhausted`), missing shard/dir. No residual `.tmp` on
  rejected paths.
- [x] Always-on test `test_qwen38_import_transaction` on the mock shard:
  (a) reads `A_log` BF16 [4] without scale, (b) reads `up_proj.weight` F8_E4M3
  [6,5] + scale_inv BF16 [1,1], (c) missing tensor rejected, (d) full transaction
  creates a verified `al_log_range.bin`, (e) wrong hash rejected without residual
  files, (f) destination collision rejected, (g) full governor => `resource_exhausted`.

**Gate:** `vastloom_tests` ALL PASSED, `ctest --preset release` 13/14 (1 skipped),
`compileall` green. The real 125B payload import stays deferred (requires a download
after the kernels and the POC).

## Q5 - Self-pinned native operators for the three primitives - complete

> **Upstream blocked:** llama.cpp `d0132a6` has no `qwen4_exp`. Q5 is self-pinned on the
> equations of `transformers` `modeling_qwen4_exp.py` / `modular_qwen4_exp.py`
> (Hugging Face, 2026-08), with golden values from a reduced Python oracle.

- [x] FP8 → F32 decoder already in Q2 (`qwen38_fp8_e4m3_to_f32`, block-scale).
- [x] `qwen38_rms_norm_affine`: grouped RMS + affine `(1 + weight)` as in
  `Qwen4ExpTextRMSNorm`.
- [x] Gated Residual: `qwen38_gated_residual` + `qwen38_gated_residual_combine`
  (SiLU bottleneck / `hc_count`, sigmoid read-gate, branch mean,
  write-gate `2 * sigmoid(inject / hc_count)`).
- [x] Gated DeltaNet one-token: `qwen38_delta_net_decode` (causal conv1d + SiLU,
  L2-norm Q/K, repeat_interleave, `g = -exp(A_log) * softplus(a + dt_bias)`,
  recurrent FP32 update, gated-sigmoid RMSNorm, `out_proj`).
- [x] QSA indexer: `qwen38_qsa_select` (pool `compress_ratio`, k-RMSNorm,
  `relu(q·k).sum / sqrt(d)`, top-k blocks + incomplete tail).
- [x] N-gram: `qwen38_splitmix64`, `qwen38_nth_prime_after`,
  `qwen38_ngram_multipliers` (seed 1234), `qwen38_ngram_hash_ids`
  (right-shift ignore-EOS + XOR hash), `qwen38_ngram_lookup`.
- [x] Always-on tests extended in `test_qwen38_flash_next_reference_operators`.

**Gate:** `vastloom_tests` ALL PASSED. No 125B download. Async n-gram prefetch and RoPE
indexer stay in Q9/Q10.

## Q6 - Non-routed 48-layer graph for Qwen3.8 - complete

- [x] `qwen38_plan_non_routed_graph(config)`: 36 Gated DeltaNet + 12 QSA
  (`full_attention_interval=4`) + 1 MTP QSA. No routed-expert tensors.
- [x] Canonical names from the real Unsloth FP8 catalog: 22 non-routed per layer
  (9 attn + 8 hyper + 5 shared/router), +138 PLE on layer 1, 29 MTP
  (22 + 4 head + 3 mixer), 5 global (embed, lm_head, mixer).
- [x] Consumer mapping: DeltaNet, QSA indexer/attention, Gated Residual
  attn/mlp, MoE router/shared, n-gram PLE, MTP head, global embed/lm_head/mixer.
- [x] `qwen38_validate_non_routed_plan` closes 48+1, 36/12, 1194+5+29=1228.
- [x] `qwen38_bind_non_routed_plan` attaches the shards; one missing name => refuse.
- [x] Always-on test `test_qwen38_non_routed_graph_plan`.

**Gate:** the non-routed plan covers exactly 48 layers + 1 MTP. No download.

## Q7 - Reduced-layer prefill/decode (2-3 layers + MTP) - complete

> Self-pinned: llama.cpp has no `qwen4_exp`. The reduced graph recomposes the Q5
> kernels; the oracle is the same primitive sequence, not ggml.

- [x] Reduced F32 graph: embed → stream `hc_count` → layer-0 Gated Residual +
  DeltaNet + shared SwiGLU → layer-1 Gated Residual + QSA (indexer + masked GQA +
  sigmoid gate) + shared SwiGLU → global mixer → `lm_head`.
- [x] MTP: `rms(h)`/`rms(embed)` + `fc_hidden`/`fc_embed` → one QSA+MLP layer
  + mixer → extra logits.
- [x] `qwen38_reduced_prefill` / `qwen38_reduced_decode` with DeltaNet state and
  QSA/MTP caches. A 2-token prefill == two incremental decodes.
- [x] Public primitives `qwen38_linear`, `qwen38_shared_mlp`,
  `qwen38_expand_stream`, `qwen38_qsa_layer_token`.
- [x] Always-on test `test_qwen38_reduced_graph`. No routed experts yet (Q8).

**Gate:** `vastloom_tests` ALL PASSED. No download.

## Q8 - MoE routing: 10 routed + 1 shared - complete

> Self-pinned on `Qwen4ExpTextTopKRouter` / `Qwen4ExpTextSparseMoeBlock`
> (`norm_topk_prob=true`). llama.cpp has no `qwen4_exp`.

- [x] `qwen38_moe_route`: logits = xW^T, F64 softmax, top-k, renormalized selected
  weights. No hash routing.
- [x] `qwen38_expert_mlp` + `qwen38_moe_block`: weighted sum of the routed experts +
  shared SwiGLU (sigmoid gate).
- [x] LRU `qwen38_expert_cache_acquire` with hit/miss/eviction/logical-byte
  telemetry. Key `(layer << 16) | expert`.
- [x] Q7 reduced graph: 4 experts, top-2, 2-slot cache on layer0/1/MTP.
- [x] Test `test_qwen38_moe_router` (golden [1,0] → experts 0+2, weights 0.5/0.5)
  and telemetry in the reduced prefill.

**Gate:** self-pinned routing on the test seed. No download. No llama.cpp comparison.

## Q9 - Gated DeltaNet state persistence + resident n-gram lookup - complete (infra)

> No 125B download. The "live RAM < 12.3 GB on the official generate" gate stays
> **unmeasured**. Q9 closes the persistence planner and the paged cache on fixtures.

- [x] `qwen38_plan_persist_budget`: 36 DeltaNet layers, conv+SSM FP32 =
  117,669,888 B (~112.2 MiB) + n-gram working set (default 4 GiB scenario B).
  Paper total 4.11 GiB < 12.3 GiB gate.
- [x] `qwen38_deltanet_persist_create`: Governor reservation and zeroed 36 states
  (conv 10240×3, SSM 48×128×128).
- [x] `qwen38_ngram_page_cache_open` / `qwen38_ngram_page_gather`: bounded pages,
  LRU, next-page prefetch, hit/miss/eviction/prefetch/logical-byte telemetry.
- [x] Always-on test `test_qwen38_persist_and_ngram_pages`.

**Infra gate:** budget and cache green. This is **not** a live measurement on the 125B.

## Q10 - QSA sparse attention (MQA indexer + gather + sparse matmul) - complete (self-pinned)

> llama.cpp has no `qwen4_exp`. The "pinned-ggml 4096" gate is self-pinned.

- [x] `qwen38_qsa_select` already in Q5 (MQA indexer 4q/1k, budget 2048, compress 4).
- [x] `qwen38_qsa_sparse_attention`: GQA gather over the `selected` only, scaled dot,
  softmax, weighted sum, sigmoid gate — extracted from `qwen38_qsa_layer_token`.
- [x] Always-on test `test_qwen38_qsa_sparse`: sparse over 2 tokens vs dense over 8,
  budget enforcement and indexer→sparse pipeline on 8 synthetic tokens.

**Gate:** self-pinned parity on short context. The real 4096 case is unmeasured.

## Q11 - Full-depth generate validation (48 layers + MTP) - complete (infra)

> The real gate (20+ tokens on the 125B FP8, llama.cpp comparison) stays **not
> executable** without the checkpoint. Q11 closes the generate driver on the reduced
> graph, ready for full depth.

- [x] `qwen38_reduced_generate`: prefill + decode loop, greedy/sampled (seeded
  softmax+top-k), `stop_token_ids`, `context_limit_tokens`, cancellation, timeout,
  `on_token`, stats (prompt/generated/forward, expert telemetry, ms, tok/s).
- [x] Always-on tests: 3-token greedy, seeded determinism, stop/cancel/timeout/
  context-limit, empty prompt rejected.

**Infra gate:** driver green on the reduced graph. The 125B is not downloaded.

## Q11b - Real 125B FP8 catalog - complete

> The `unsloth/Qwen3.8-Flash-Next-FP8` checkpoint (172.8 GiB, 131 shards, 140 files)
> was imported through the canonical path
> (`tools/import_qwen38_fp8.py` over `import_hf_model.import_model`: `.vastloom-store`
> store, locking, per-file SHA-256, atomic manifest, read-only) into
> `C:\vastloom-store\models\original\unsloth-Qwen3.8-Flash-Next-FP8--main`.
> The download required an HF token (anonymous throttle ~30 KB/s after the burst;
> authenticated ~78 MB/s, ~35 minutes total).

- [x] `vastloom qwen38-inspect <root> [--load-header] [--json]`: metadata-only
  inspection of config + index (+ first shard header) with pinned validation.
- [x] Full validation of the real catalog: 131 shards, 152,089 tensors, 75,264
  FP8 block-scale experts, 129 n-gram tensors (128 shards + weight_scale), 48
  layers, 333 vision, 3,101 MTP, 5 globals.
- [x] Fixture→real drift corrected: `ExpectedCatalog::ngram_tensors` 128→129 (the
  `ple_embedding.ngram_embedding.weight_scale` tensor exists in the real checkpoint
  and not in the Q1 fixture; already foreseen in the Q6 plan).
- [x] Serial opt-in test `vastloom_qwen38_checkpoint`
  (`-DVASTLOOM_QWEN38_MODEL=<root>`), metadata-only: PASS on the real checkpoint.

**Gate:** real catalog validated end-to-end in native C++. No tensor payload read for
compute yet (next step: binding and a real layer).

## Q11c - Real layer-0 executing - complete

- [x] `vastloom qwen38-layer0 <root> --token N [--json] [--dump-hidden]`:
  full real layer-0 forward — embedding row (bounded read of a single row, 5 KB),
  attn gated residual, Gated DeltaNet (zeroed state), combine, mlp gated residual,
  top-10 router over 512, shared expert, on-demand load of the 10 selected FP8
  experts, final combine.
- [x] **FP8 bug found on the real payload and fixed**: `qwen38_fp8_e4m3_to_f32`
  treated all `exp==15` as NaN; in E4M3FN (the HF format) NaN is only
  `exp==15 && mant==7`, while `0x78–0x7E` are finite up to 448. The official
  checkpoint contains such encodings (3,370 in a single gate_proj); the old decoder
  corrupted any real inference. The Q2 test was extended with the new pins
  (0x78→256, 0x7E→448, 0xFE→−448).
- [x] Per-stage finiteness guards (embed/stream/gated/DeltaNet/shared/experts) with
  errors naming the stage.
- [x] Extended opt-in test: real forward on the BOS token, finiteness, exact top-k
  route, bit-identical determinism on double execution.
- [x] Router verified input-sensitive: token 100 → experts
  {215,226,122,448,483,363,459,40,239,500}; BOS token → {411,334,286,152,291,
  106,421,304,226,331}.

**Gate:** complete layer-0 executed on real weights (204 MB payload read, ~1.2 s load,
~46 ms forward). Deterministic, finite output. This is a self-pinned run: no external
oracle exists for `qwen4_exp`.

## Q11d - Real multi-token layer-0 prefill - complete

- [x] Refactor to trunk/row: `qwen38_load_layer0_trunk` (22 tensors + expert→shard map
  built once, no re-parsing of the 17 MiB index per expert), `qwen38_load_embed_row`
  (bounded 5 KiB read), caller-owned DeltaNet state.
- [x] `vastloom qwen38-prefill <root> --tokens <csv> [--expert-cache N]`:
  multi-token prefill with conv history and SSM carrying across tokens, bounded
  expert LRU (default capacity 16 ≈ 315 MiB decoded) and telemetry.
- [x] Implicit regression: the BOS token in the prefill reproduces the Q11c
  single-token sum (3.0418) bit-identically.
- [x] Verified streaming: token 100 alone and the same token after BOS produce
  different routes ({215,...} vs {283,...}) — the route depends on state, as it
  should.
- [x] Real LRU over 5 tokens: 44 misses / 6 hits / 28 evictions; the hits come from
  real expert recurrence (226 in tokens 0-2, 334 in tokens 0/4).
- [x] Extended opt-in test: 5-token prefill, bit-identical replay, state-carry
  assertion, CLI/test cross-check on the same sums.

**Gate:** deterministic real multi-token prefill (bit-identical replay) with state
carrying across tokens. Self-pinned; QSA/MTP and full depth stay open (Q11e).

## Q11e - Real QSA layer (layer 3) - complete

- [x] Loader generalized to any layer: `qwen38_load_layer_trunk` resolves the 22
  non-routed tensors from the weight_map (no hardcoded shards), loads the DeltaNet
  branch (9 tensors) or the QSA branch (9 tensors) according to
  `full_attention_interval`, and maps the experts of the requested layer.
- [x] Generic forward `qwen38_layer_forward` with caller-owned
  `Qwen38LayerRuntimeState`: conv/SSM for DeltaNet, indexer-keys + KV cache for QSA.
- [x] **Real QSA layout confirmed by the pinned operator**: `q_proj`
  [12288, 2560] = query+gate per head (24×256×2), `index_qk_proj` [640, 2560]
  = (4+1)×128, `k/v_proj` [512, 2560] = 2×256, `o_proj` [2560, 6144].
- [x] CLI: `qwen38-layer --layer N` and `qwen38-prefill --layer N` (the
  `qwen38-layer0` alias stays compatible).
- [x] Real QSA layer 3: BOS sum 28.0353, deterministic 5-token prefill (bit-identical
  replay), state-dependent route, selected FP8 experts {169,418,141,...}; CLI and test
  agree.

**Gate:** both layer types (36 DeltaNet + 12 QSA) execute on real weights with
deterministic multi-token prefill. Self-pinned.

## Q11f - Chain and real full depth (48 layers) - complete

- [x] `qwen38_layer_forward` accepts the **input stream** (hc_count × hidden) instead
  of the embedding row: one layer's gated-residual output feeds the next; only layer
  0 starts from the expanded embedding.
- [x] `vastloom qwen38-chain --layers <csv> --tokens <csv> --expert-cache N
  [--no-trunk-resident] [--trace-layers]`: a chain of real layers with per-layer
  state and a shared expert LRU (key layer<<16|expert).
- [x] **Memory bug found and fixed**: in streaming mode (non-resident trunk) the
  final print indexed `trunks[i]` beyond the single loaded element — an OOB read
  causing the access violation (0xC0000005) of the full-depth run; short chains
  passed only by lucky heap layout. The attention type is now derived from the
  layer id.
- [x] Streaming mode verified identical to resident mode (same sums, chain 0-3):
  the trunk is loaded and executed one layer at a time.
- [x] **Real full-depth run 0→47**: 48/48 layers (36 DeltaNet + 12 QSA), 480 real FP8
  experts, 9,633,094,400 payload bytes read, bit-identical replay (sum −28.9797).
  Documented note: the layer-1 PLE n-gram block is not yet integrated in the forward.

**Gate:** full-depth trunk+MoE executed on real weights, deterministic. Self-pinned;
the external comparison arrives with Q8_0 + llama.cpp.

## Q11g - Q8_0 GGUF checkpoint (qwen4exp) - complete

- [x] Download of the official Q8_0 (`unsloth/Qwen3.8-Flash-Next-GGUF`, 6 shards,
  175.3 GiB, SHA-256 verified, read-only, atomic manifest) through the extended
  driver `tools/import_qwen38_fp8.py --repo --gguf-dir`. The FP8 safetensors
  checkpoint was **deleted** on request: the store now holds the Q8_0 only.
- [x] GGUF metadata read and verified: `general.architecture = qwen4exp`,
  1,224 tensors (811 Q8_0, 389 F32, 24 BF16), all invariants match and the metadata
  also carries rope_sections, per-layer compress_ratios,
  ple.head_offsets/vocab_sizes, `embedding_length_per_layer_input = 160`.
- [x] `src/inference/qwen3_8_gguf.cpp`: Q8_0 decode (F16 scale + 32 int8),
  `qwen38_f16_to_f32`, validated split-catalog opening, tensor read with shape
  check + dequant + conversion from the ggml `[in, out]` layout; `ssm_a` confirmed
  F32 as A_log (all-negative range).
- [x] Complete gguf→operator mapping: `attn_qkv`→in_proj_qkv,
  `attn_gate`→in_proj_z, `ssm_alpha/beta`→in_proj_a/b, `ssm_out`→out_proj,
  `hc_attn/ffn_*`→gated residual, **split** indexer into q/k_proj (the fused
  index_qk is reconstructed by concatenation), experts as **fixed-stride slices**
  of the fused `[in, out, 512]` tensors, column embedding from `token_embd`.
- [x] Automatic safetensors/GGUF dispatch in `qwen38_load_layer_trunk`,
  `qwen38_load_embed_row` and `qwen38_load_layer_expert`: every CLI
  (`qwen38-layer`, `qwen38-prefill`, `qwen38-chain`) runs on both.
- [x] Real full depth on Q8_0: 48/48 layers, 6.58 GB payload, exit 0. Sums differ
  from FP8 due to requantization (expected): layer-0 BOS 6.65287, layer-3 17.7864,
  chain 0-3 bit-identical replay (23.0238).
- [x] Bifurcated safetensors/GGUF opt-in test; `VASTLOOM_QWEN38_MODEL` reconfigured
  to the Q8_0 root.

**Gate:** the native C++ runtime now executes the **llama.cpp-compatible** checkpoint
end-to-end (single layers, prefill, chain, full depth) — the base for the llama.cpp
oracle and for VRAM expert residency.

## Q11h - Greedy parity with llama.cpp - complete

> **External oracle reached**: the native runtime generates the same tokens as
> llama.cpp on the same Q8_0 checkpoint. End of the self-pinned era.

- [x] llama.cpp `6fe7498` cloned and built (CPU Release, MSVC);
  `llama-completion`, `llama-tokenize`, `llama-debug` (+ optional dump patch via
  env, local clone only, not committed).
- [x] Systematic per-stage debugging with intermediate dumps (`hc_norm`,
  `hc_mixed`, `attn_output`, `final_output`) and numpy replication — six defects
  found and fixed:
  1. **DeltaNet z-gate: sigmoid, not silu** — llama: "the one numerical difference
     from Qwen3.5's GDN: sigmoid output gate, not silu"; the HF modeling main does
     not carry it (this was my Q11c fix error).
  2. **ssm_a pre-transformed**: the GGUF already carries −exp(A_log) (verified
     all-negative range); the loader converts it to log space for the operator.
  3. **Norm weights already (1+w)** in the GGUF (verified mean ≈ 1): shift by −1 at
     load, except `ssm_norm` (RMSNormGated, direct weight).
  4. **Tile k-head mapping** (vh % k_heads, ggml repeat semantics), confirmed by the
     ggml kernel `ik1 = iv1 % nek1`.
  5. **Partial RoPE** (64 of 256 dims, theta 1e7) on attention q/k and indexer
     q/pooled-keys + **PLE integrated** (n-gram hash → table lookup → key/value →
     gate → dilated conv), missing in Q7/Q10.
  6. **Trunk misalignment in the greedy CLI**: layer 0 ran with layer 47's trunk of
     the previous token (trunks.size()==1 always).
- [x] Stage-by-stage numerical verification with identical input (llama's hc_mixed):
  qkv/conv ~0.01%, delta rule raw sum 0.8686 vs 0.8687 (0.01%), post-norm core with
  sigmoid ~0.2%.
- [x] **Greedy parity**: "The capital of France is" → " Paris. The capital of
  Germany is Berlin. The capital of Italy is Rome. The" (16 tokens) = identical to
  llama.cpp greedy. Second prompt ("Hello world") plausible and deterministic.

**Gate:** the native Vastloom runtime produces the same tokens as llama.cpp on the
official Q8_0 checkpoint. The external oracle is established for Q12+.

## Q11j - Throughput P2b-P3a - complete

- [x] Q8_0-native expert matvec + O(1) LRU + parallel DeltaNet decode
  (`71b8a5a`): bit-identical, 33.75 → ~7 s/token.
- [x] Concurrent top-k expert prefetch (`8e9c970`): ~4 s/token.
- [x] Q8_0-native trunk residency, P2d (`19b3319`): residency startup
  16.8 → 3.1 s; streaming without the override 194 → 32 s; parity unchanged.
- [x] P3a `--cuda-experts` (`70a8e6b`): bit-identical but MEASURED slower than the
  12-thread CPU kernel (FP64 at 1/64-rate + parity-constrained grids) → stays
  opt-in, the CPU kernel is the product path.

**Gate:** 16/16-token greedy parity kept at every step; honest numbers in
AGENTS.md/CHANGELOG. ~3.3-4 s/token generated, against llama.cpp's 0.32 s/token
(~10x gap, structural: page cache vs our own caches).

## Q11k - Chunked prefill (Qwen track)

> **Track status (October 2026 refocus): comparison/validation-only.** Q11k is no longer the
> priority of the active programme; the design and open items below are kept as the Qwen
> track's contract, not as scheduled work.

> **Design:** [`planning/Q11K_CHUNKED_PREFILL_DESIGN.md`](planning/Q11K_CHUNKED_PREFILL_DESIGN.md)
> — layer-major/trunk-stationary chunk pipeline, batched Q8_0 GEMM, chunked GDN/QSA, MoE
> expert union, the chunk-boundary invariance contract, the Governor budget plan and the
> blocking entry criteria. Design only: no chunked-prefill code exists.

Motivation: the checkpoint declares a 262,144-token context; DeltaNet states are O(1)
and the QSA caches grow ~72 KB/token in F32 (128k ≈ 9.4 GB, halvable with F16 caches —
feasible in RAM), but serial 1-token prefill (~3.5-4 s/token) makes the long context
unreachable in practice. Without fast prefill, Q12 (prefix-KV) and Q13 (serve) stay
short demos.

- [x] Lever found before chunking: OPT-IN ggml SIMD backend for Q8_0 matvecs
      (`4b42627`): forward 6.1→1.7 s, expert 19.6→11.2 s over 8 forwards; 16/16
      parity kept on the existing battery; chunked GDN in the scalar reference would
      be NEGATIVE (O(C²·d) in scalar > serial recurrence O(d) per token) → useful only
      together with SIMD GEMMs.
- [x] DEFECT D1 RESOLVED AS A DIAGNOSIS (`299988a`): stage bisection on "Once" —
      hc_norm equal to 1e-6, hc_gate deviating 3.6e-2; the numpy replica of
      ggml_vec_dot_q8_0_q8_0 (Q8_0-quantized activations) closes the 8.6x gap.
      Cause: llama.cpp quantizes the ACTIVATIONS for every matvec with Q8_0 weights;
      our double path is more accurate than the reference and on near-ties (margin
      0.36 on that prompt) can legitimately flip. It is not a kernel defect.
- [x] Parity decision closed (August 2026): parity is a margin-aware verification
      instrument (identical tokens where the logit margin > threshold; near-ties =
      documented deviation because our double path is more accurate than the
      activation-quantizing oracle). The `--llama-activation-quant` mode was
      considered and DROPPED: the oracle's rounding is not cloned for parity's sake.
- [ ] Q11l - ggml SIMD as default + Q8_0 lm_head (`output.weight` is Q8_0: 660 MB
      resident vs 2.5 GB F32, SIMD matvec) and the margin-aware parity battery
      (tools/parity_battery.py, to be created: ≥6 EN/IT prompts with margin dumps;
      exit-code gate).
      Honest measurement vs llama.cpp AND ik_llama.cpp.
- [ ] Q11m - Q4_K expert re-import (a new profile beside Q8_0, gated by the
      minimum working set; ~1.3 GB/token of expert traffic) with the pinned-ggml
      decoder already in the repo.
- [ ] Q11n - model-aware VRAM tier: expert pinning by routing statistics + layer
      N+1 prefetch during N, under the Governor. Falsifiable goal: beat the OS page
      cache on expert hit-rate at equal RAM — a lever ONLY Vastloom can take;
      benchmark against ik_llama.cpp (`--cpu-moe`/`-ot`) and mainline llama.cpp.
- [ ] Q14 - capability transfer: DESIGN + falsifiable ladder only for now (a fixed
      task suite, a resident frontman, selective N+1 wake-up via the already-resolved
      hash, distilled bridges with the existing `distill_*` tooling).
- [ ] Chunked prefill driven by the ggml backend: batches of T tokens per layer
      (GEMMs reuse the Q8 rows in the L2 cache), conv/PLE aligned.
- [ ] Chunked form of the GDN delta rule only if the profile requires it.
- [ ] Batched QSA prefill: indexer scoring for several queries per pass.
- [ ] Measurement: prefill throughput token/s on long prompts vs the serial
      baseline, bytes/token read, in a dedicated report.

**Gate:** margin-aware parity (identical tokens where the margin exceeds the
threshold, near-ties documented) on a battery of ≥ 6 prompts × 24 tokens; extended
tolerance-declared replay of the multi-token prefill; throughput measured and
documented.

## Q12 - Prefix-KV MoE stream (M13 port)

> **Track status (October 2026 refocus): comparison/validation-only, not scheduled.**

- [ ] Extend `LlamaSession` to the MoE stream path.
- [ ] DeltaNet/QSA KV cache persisted across consecutive requests.
- [ ] OpenAI-compatible endpoint as for the M13 SmolLM2 one.

**Gate:** two consecutive turns with cached_prefix_tokens > 0 on the MoE stream.

> **Pre-validation on the OLMoE stream (October 2026, done):** the M13 session
> pattern is implemented for `OlmoeRuntimeModel` (`OlmoeSession` with a
> Governor-reserved persistent KV workspace, prefix reuse with stateless
> degradation on divergence/overflow) and served through `serve <olmoe-dir>`
> (`/v1/models`, `/v1/chat/completions` streaming + non-stream, greedy-only,
> `vastloom_cached_prefix_tokens` observability). Covered by the opt-in
> `vastloom_olmoe_session` C++ equivalence test and the `vastloom_olmoe_serve`
> Python smoke test (both gated on `VASTLOOM_OLMOE_FULL_MODEL`; harness
> verified without weights, real-checkpoint run pending). Q12 itself — the
> same pattern on the Qwen DeltaNet/QSA stream — stays open.

## Q13 - vastloom serve for Qwen3.8

> **Track status (October 2026 refocus): comparison/validation-only, not scheduled.**

- [ ] Native text→id tokenizer for qwen4exp (gpt2-BPE: `tokenizer.ggml.tokens`
      + `merges` + pre-tokenizer from the GGUF metadata), validated against
      `llama-tokenize.exe` (oracle) on EN/IT fixtures.
- [ ] ChatML framing of the messages (im_start/im_end from the GGUF vocab; the
      template in the file is decorative, the shape is known and pinned).
- [ ] `serve <directory>` dispatch: Q8_0 GGUF qwen4exp detection → Qwen3.8 chat
      engine (trunk residency, expert LRU, `--cuda-experts` option); generation is
      the qwen38-greedy loop extracted from the CLI, with cancel/timeout from the
      reduce_generate pattern.
- [ ] `/v1/models`, `/v1/chat/completions` endpoints (streaming + non-stream).
- [ ] Opt-in Python test via `tests/test_qwen38_cli.py` (to be created).

**Gate:** end-to-end green smoke test. Honesty note: before Q11k the serve session
re-prefills every turn at ~3.5-4 s/token → short-context demos; the realistic long
context depends on Q11k+Q12, not on Q13.

## Post-conclusion update 0.5.0-alpha.1 (2026-10-20)

> A maintenance and safety update after the project conclusion. The maintenance slice
> reopens nothing: it advances no large-model milestone, no throughput or parity measurement
> was repeated, and the Qwen items stay blocked by the shared entry criteria in this section.
> The separate DeepSeek refocus section below records the only continuation: D1/D2a, D2b, D3
> (single- and multi-token scope) and the closed D4 design are measured; the VRAM payload tier,
> ratio-128 and learned-router/full-depth execution are not planned in this concluded record.

### Completed

- [x] **DeepInfer review** (`docs/research/DEEPINFER_REVIEW.md`): a datacenter-only engine
  (SM100/Ascend 950, ~511 GB V4.1-Flash checkpoint, a 64+64-GPU minimum deployment); the
  adoption order P0/P1/P2 and the rejected mechanisms are recorded; no code or dependency
  was adopted.
- [x] **Dense serve safety**: chat message content that encodes a reserved control token is
  refused fail-closed with HTTP 400; every SSE chunk is valid UTF-8 (multi-byte sequences
  are buffered across token boundaries); a streaming-producer lifetime defect was fixed
  (the SSE producer invoked request state that had already gone out of scope).
- [x] **JSON-object constrained decoding** on the dense SmolLM2 path:
  `response_format: {"type": "json_object"}` on `serve` (exactly one root object, no JSON
  Schema or GBNF), `finish_reason` `stop`/`length`, and the `vastloom_constraint_complete`
  extension field; the one-shot CLI exposes `chat --json-object`.
- [x] **Q11k design** (`docs/planning/Q11K_CHUNKED_PREFILL_DESIGN.md`): the chunked-prefill
  contract, memory budget and blocking entry criteria for the next milestone.
- [x] **Documentation and metadata**: README.md, AGENTS.md, SAFETY.md,
  MODEL_COMPATIBILITY.md, CHANGELOG.md and CITATION.cff updated for `0.5.0-alpha.1`.
- [x] **Stale-note correction**: the Q11C report still says the performance levers were
  "designed but not implemented"; Q11j implemented the opt-in ggml SIMD backend, Q8_0 trunk
  residency and concurrent expert prefetch, and Q11k now has the design above. The Q11C
  report stays a dated evidence snapshot and is not rewritten.

### Blocked backlog (Qwen track, comparison/validation-only)

No large checkpoint is present on the reference PC any more — the Qwen3.8-Flash-Next Q8_0
GGUF and the DeepSeek V4 Flash 0731 full profile are gone, and a local llama.cpp oracle is not
available. The items below are **blocked, not retired**. The DeepSeek refocus below does not
depend on this backlog: it is range-slice based and has its own gates.

| Item | Gate when unblocked |
|---|---|
| Q11k — chunked prefill | chunk-boundary invariance and token-serial parity pass; prefill throughput measured in a report |
| Q11l — ggml SIMD default + margin battery | `tools/parity_battery.py` (≥ 6 prompts × 24 tokens) green; honest comparison vs llama.cpp and ik_llama.cpp |
| Q11m — Q4_K expert re-import | new profile passes the minimum-working-set gate; expert bytes/token measured |
| Q11n — model-aware VRAM tier | beats the OS page cache on expert hit-rate at equal RAM |
| Q12 — prefix-KV on the Qwen stream | two consecutive turns with `vastloom_cached_prefix_tokens` > 0 |
| Q13 — serve for Qwen3.8 | native tokenizer parity with `llama-tokenize`; end-to-end serve smoke |
| Q14 — capability transfer | design-only ladder; unchanged |

**Shared entry criteria (all of the above):**

1. The exact large checkpoint revision is legally available again and re-imported through
   the canonical immutable store, with every file's SHA-256 verified and the live capacity
   gate passed.
2. The pinned llama.cpp oracle (`6fe7498`, or a later revision re-pinned explicitly) is
   built locally, and the margin-aware parity battery with margin dumps exists.
3. The Governor preflight passes on the reference PC (32 GiB RAM, 12 GiB VRAM) or on a
   host that passes the public minimum-working-set gate; the plan is re-derived from live
   capacity, never copied from an older machine state.
4. The always-on reduced-graph test for the mechanism exists before real-weight work, and
   every claim waits for a dated validation report under `docs/reports/`.

## DeepSeek refocus (October 2026)

> **Status: DeepSeek refocus measured through D3 (single- and multi-token) and the closed D4
> design; research record concluded, no full-model claim.** The decision is recorded below and
> seven steps are now measured: a metadata-only IQ2_XXS store and range imports (D1), one real
> layer-0 execution (D2a), bit-identical device expert residency with bounded CPU fallback
> (D2b.1), an independent NumPy layer-0 oracle (D2b.2), a deterministic 2/3-layer hash-routed
> chain (D3a) whose independent chain oracle matches tokens 0/1 at depths 1–3 (D3b), a
> five-token prefill that opens the layer-2 4:1 compression window and is matched by the same
> oracle (D3c) and a design-only D4 byte budget with closed decisions. Evidence is the dated
> [`D2 layer-0 range execution`](reports/D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md),
> [`D2b.1 device residency`](reports/D2B_DEVICE_RESIDENCY_VALIDATION.md),
> [`D2b.2 layer-0 oracle`](reports/D2B_LAYER0_ORACLE_VALIDATION.md),
> [`D3 hash chain`](reports/D3_HASH_CHAIN_VALIDATION.md),
> [`D3b chain oracle`](reports/D3B_CHAIN_ORACLE_VALIDATION.md),
> [`D3c multi-token oracle`](reports/D3C_MULTITOKEN_ORACLE_VALIDATION.md) and
> [`D4 on-demand expert ranges`](planning/D4_ON_DEMAND_EXPERT_RANGES.md) documents. Still not
> claimed: a VRAM *payload* tier (expert bundles still enter governed RAM first), throughput,
> ratio-128 or learned-router/full-depth execution and any full-model result. Those remaining
> items are not planned in this concluded research record. Historical full-payload and Qwen
> records stay dated evidence.

**Decision.** The primary execution target returns to **DeepSeek V4 Flash 0731**
(`unsloth/DeepSeek-V4-Flash-0731-GGUF`, revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`) because the DeepInfer comparison and the newer
DeepSeek-V4.1-Flash-1017 model are DeepSeek-centric and the research is meant to be
DeepSeek-first. `UD-IQ2_XXS` is the only execution quant (three GGUF shards, 90.86 GB total);
IQ1 / `UD-IQ1_S` stays non-product and historical. The full checkpoint is **not stored** on the
reference PC: every step below works with bounded exact range slices — trunk tiles, the six
global consumers and the selected complete expert bundles — through the existing pinned
importer, immutable store and Safety Governor. Qwen3.8-Flash-Next becomes
comparison/validation-only: its code, tests and llama.cpp parity evidence stay, and no new
Qwen features are planned. The `0.5.0-alpha.1` dense serve update remains an orthogonal
maintenance slice.

**Memory contract.** Selected experts are placed in VRAM with a bounded fallback to system
RAM; the always-active trunk weights stream from NVMe under the Governor; no full model is
ever resident, and RAM plus VRAM is never summed to admit an indivisible expert. D2b.1 keeps
expert matrices resident in VRAM with a bounded CPU fallback, but the bundles still enter
governed RAM first: this is device residency, not a VRAM *payload* tier
(`ResidencyObject` VRAM transitions remain unused on this path).

**Entry criteria for every D-gate.**

1. The pinned 0731 revision and its metadata identities are verified through the existing
   bounded GGUF reader; header/metadata-only reads are allowed, a full-checkpoint download is
   not.
2. Any payload is fetched only by the pinned range importer with explicit authorization, the
   selected-payload cap and the model-store free-space reserve, then rehashed and published
   atomically under `models/optimized`; header prefixes never become executable sources.
3. The Governor preflight passes from live capacity (32 GiB RAM / 12 GiB VRAM reference PC),
   and every command reports its skipped/refused state instead of degrading silently.
4. Each gate's evidence is a dated validation report under `docs/reports/`; no claim precedes
   its report, and no full-checkpoint execution is claimed from a range-slice result.

**Sequence.** Each item is unchecked until its gate passes.

- [x] **D0 — documentation refocus.** README, AGENTS, this roadmap, MODEL_COMPATIBILITY and
      CHANGELOG state the same target, quant, reduced-payload approach, memory contract and
      D1–D4 gates; no source, test or dated report is changed.
      **Gate met:** the documents agree on the decision and on what is *not* claimed.
- [x] **D1 — IQ2_XXS metadata identities and one-layer range plan/import.** Pinned the
      `UD-IQ2_XXS` identities (complete split 1, 5,257,664 B, plus separately hashed
      2,097,152-byte header prefixes for splits 2/3), imported a 9,451,968-byte metadata-only
      store and verified it with `deepseek-gguf-checkpoint` (pinned spec, reference config and
      layer-0 catalog, hashes verified). The catalog closes at 1,328 physical / 34,223 logical
      tensors, 43 layers, 256 experts top-6 and 11,008 bundles; always-active 6,986,088,796 B,
      routed 83,869,302,784 B, bundles 7,536,640–9,830,400 B. Three range plans were imported
      with every local range hash verified: layer-0 trunk 138,867,160 B / 21 ranges, globals
      596,000,788 B / 6 ranges and the authoritative route 45,219,840 B / 18 ranges, resolved
      from the imported trunk's own `tid2eid` for token 0 (experts 254, 222, 245, 200, 53, 35;
      `routing_authority_verified: true`). Total payload: 780,087,788 B in 45 ranges.
      **Gate met:** pinned identities, exact plan hashes, verified spans and a metadata-only
      checkpoint verification; D1 itself claims no execution.
- [x] **D2a — real layer-0 execution from IQ2_XXS ranges.** `deepseek-layer-probe` ran layer 0
      from the metadata store plus the three verify-on-load range artifacts with `--token 0`:
      1 of 43 layers (`complete_model: false`), 129,280 logits, min −16.1273, max 15.5206,
      argmax 30,528, the same six experts, 45 relocated tensors / six bundles, and 45,219,840
      expert + 138,867,160 trunk + 596,000,788 global bytes; two runs were identical at
      345.325 ms and 347.433 ms. This is one layer of 43 and **not** the model's output.
      **Gate met (execution half only):** deterministic, repeatable execution from verified
      IQ2_XXS payload. No parity at that gate; D2b.2 later supplies the layer-0 logits oracle.
- [x] **D2b.1 — device expert residency with bounded CPU fallback.** `deepseek-layer-probe`
      gained `--cuda-experts`, `--gpu-limit`, `--gpu-reserve`, `--cpu-only` and the
      `--dump-logits/--dump-hidden/--dump-trace` dumps (the CPU default is unchanged). With
      `--cuda-experts --gpu-limit 16`: offered 38, accelerated 18, declined 20 (all recurring
      trunk/global), evicted 12, resident 6 matrices / 15,230,976 B, uploaded 45,219,840 B,
      kernel 138.961 ms, upload 6.2527 ms; all 129,280 logits are bit-identical to the CPU run
      (max abs diff 0.0). With `--gpu-limit 1`: accelerated 0, declined 38, resident 0 and the
      logits are still identical. The opt-in `vastloom_deepseek_v4_gpu_residency` contract
      passes with the same counters and a declared 1e-3 fallback tolerance.
      **Gate met:** bounded device residency and CPU fallback are bit-identical. This is **not**
      a VRAM payload tier: expert bundles are read into governed RAM first and the device holds
      matrix copies; `ResidencyObject` VRAM transitions remain unused.
- [x] **D2b.2 — independent layer-0 logits oracle.** An independent NumPy implementation
      (`tools/reference_deepseek_v4_layer_oracle.py`; no C++ calls, no network) dequantizes the
      same artifact bytes and recomputes layer 0: 129,280 logits agree within
      **9.5367431640625e-07** (gate 1e-3), the 4,096-wide final hidden state within
      **4.76837158203125e-07** (rtol 1e-5 / atol 1e-6), and argmax 30,528 plus route
      254/222/245/200/53/35 are identical. The synthetic decoder section passes 87 pinned
      checks; 66 oracle stage checksums are recorded, of which only logits and the final hidden
      state have a probe counterpart (the FNV difference is a 1-ulp diagnostic, not a gate).
      `VASTLOOM_DEEPSEEK_V4_ORACLE_PYTHON` registers `vastloom_deepseek_v4_layer_oracle`;
      `ctest` PASS at 87.14 s.
      **Gate met for layer 0:** external-reference parity within the declared tolerance.
- **D2b status: closed** — placement (D2b.1) and the layer-0 oracle (D2b.2) cover the gate's
      wording. A true VRAM *payload* tier remains open and is explicitly not claimed.
- [x] **D3a — hash-routed chain executor and measured runs.** New `deepseek-chain-probe`
      executes 2 or 3 layers from per-layer trunk/expert artifacts through a lazy trunk window
      and per-layer expert caches; only the three hash-routed layers are admitted (a
      learned-router layer is refused because its route needs the preceding hidden state).
      Token 0, 3 layers: argmax 86,051, logits [−16.0034, 16.1624], trunk 437,727,112 B,
      expert bytes 135,659,520 B, 133 relocated tensors / 18 bundles, 2,761.23 ms; the repeat
      run is bit-identical (max abs diff 0.0). Token 1, 3 layers: argmax 69,146 with
      per-layer authoritative routes. Token 0, 2 layers: argmax 86,051. Extra D3 download:
      434,539,472 B (trunks 1–2 + token-0 experts) plus 135,659,520 B (token-1 experts); local
      payload is about 1.3 GiB.
      **Executor gate met:** multi-layer range execution is deterministic and byte-accounted.
- [x] **D3b — D3 gate: oracle extension and automated chain tests.** The independent NumPy
      oracle now executes the three-layer chain, including the layer-2 4:1 compressor and
      indexer, and reproduces the native chain probe for tokens 0 and 1 at depths 1, 2 and 3:
      logits max abs diff **9.5367431640625e-07** (token 0) / **1.9073486328125e-06** (token 1),
      hidden max abs diff 4.76837158203125e-07 → 3.814697265625e-06 (token 0) and
      2.384185791015625e-07 (token 1), argmax 30,528 / 86,051 / 86,051 (token 0) and 69,146
      (token 1), identical per-layer routes, unchanged tolerances; the synthetic section passes
      104 pinned checks. Two opt-in CMake tests PASS: `vastloom_deepseek_v4_chain_probe`
      (587.64 s: pinned 2/3-layer results, deterministic repeat, malformed chains refused) and
      `vastloom_deepseek_v4_chain_oracle` (780.83 s: six token/depth comparisons). The GPU chain
      (`--cuda-experts --gpu-limit 64`: offered 116, accelerated 54, declined 62, evicted 28,
      26 resident matrices / 66,350,080 B, uploaded 135,659,520 B) is bit-identical to the CPU
      chain (max abs diff 0.0).
      **Gate met for the measured scope:** single-token prefill, three hash-routed layers,
      two tokens × depths 1–3. A one-token prefill closes no compression window, so
      compressed-slot attention and indexer selections with `top_k > 0` are synthetically, not
      natively, compared.
- **D3 status: met for both the single-token and the multi-token scopes.** No throughput or
      full-model claim.
- [x] **D3c — multi-token prefill and compressed-slot attention.** `deepseek-chain-probe
      --tokens 0,1,2,3,4 --layers 3` ran the five-token causal prompt twice with bit-identical
      dumps: argmax 20,267, logits [−38.0188, 14.7002], trunk 437,727,112 B, expert bytes
      663,224,320 B (88 bundles from 15 artifacts, duplicated bundles merged), 391 oracle
      stages. The independent NumPy oracle matches the last-position logits within
      **1.9073486328125e-06** and the final hidden state within **5.960464477539063e-08**, with
      all 15 per-layer/per-token routes identical and a non-empty compressed window: one
      compressed row, selected by positions 3 and 4, compressed stages 512/128 and six slots
      per position. The chain contract test was extended with the same five-token leg (PASS).
      **Gate met:** multi-token prefill and the real 4:1 compressed-slot path reproduce the
      oracle within the unchanged tolerances; no throughput claim.
- [x] **D4 — on-demand expert range design and byte budget (design-only).** The design is
      `docs/planning/D4_ON_DEMAND_EXPERT_RANGES.md` with the offline estimator
      `tools/estimate_deepseek_token_bytes.py`: pre-download all always-active ranges once
      (6,986,088,796 B in 1,199 ranges plus the 9,451,968-byte metadata store, footprint
      6,995,540,764 B inside the 10 GB budget) and fetch selected complete expert bundles on
      demand under bounded staging/cache and Governor reservations. Worst case is
      1,965,686,784 expert bytes per token; a full-depth no-cache token needs 8,951,775,580 B.
      **Decisions closed (§11 of the design):** SHA-256 first-touch digests for bundles without
      a pre-pinned digest; disk bundle cache 2,000,000,000 B, staging 268,435,456 B and RAM
      expert cache 1,073,741,824 B reserved through the Governor; the design does not depend on
      the separate, unimplemented VRAM *payload* tier; no fetch occurs in this project.
      **Design gate met:** written design plus byte-and-budget analysis with closed decisions,
      and no network, no full-checkpoint download and no execution claim.

**Closing note (October 2026).** Remaining: the VRAM *payload* tier, ratio-128 layers and
learned-router/full-depth execution — not planned in this concluded research record (they need
new subsystems or the full checkpoint); no further milestones. The consolidated, dependency-ordered
backlog for resuming the research — including the Qwen revalidation checklist and the safety entry
criteria — is [`planning/FUTURE_WORK.md`](planning/FUTURE_WORK.md). A full 90.86 GB checkpoint
download, IQ1/`UD-IQ1_S` execution, CUDA dense/attention kernels, speculative decoding and any
Qwen feature work were already out of scope, and a gate that would need a llama.cpp comparison
against the full checkpoint stays blocked rather than changing that rule.

---

## Archived track

The M0–M13 track (DeepSeek V4 Flash 0731) is frozen in `docs/archive/deepseek_0731/`.
The October 2026 DeepSeek refocus above is a separate reduced-payload sequence and does not
unfreeze this archive. The C++ tests defending the pinned-ggml parity for
IQ1_S/IQ2_XXS/Q4_K stay compiled and green — they are historical contracts. For the
original M14+ (multi-backend, Kimi K3, Laguna, MiniMax, Vulkan/ROCm, multi-NVMe) see the
archived roadmap preserved in git history (the previous content of this file, retrievable with
`git log --follow docs/ROADMAP.md`).
