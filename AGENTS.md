# Agent contribution guide

This file applies to the entire repository. Human-facing contribution rules are in
`CONTRIBUTING.md`; automated coding agents must follow both documents.

## Mission and current boundary

Vastloom is a native, hardware-adaptive local inference runtime that treats VRAM, RAM and
NVMe as a model-aware hierarchy.

**North star.** The capability-transfer vision lives in
`docs/research/CAPABILITY_TRANSFER_VISION.md`: a fast, dense, resident **frontman**
(default `gemma-4-12B-it` QAT Q4) expanded at runtime by a **dormant brain**
(`deepseek-v4-flash-0731` on NVMe) through bounded bridges — offline distillation into
compact low-rank modules, a selective oracle wake-up, and retrieval. It is explicitly *not*
a claim: nothing in it is implemented or measured, splicing teacher parameters across
architectures is listed as excluded, and "x100 intelligence" is replaced by a task-scoped
falsification ladder. Treat it as a compass; do not let it rewrite the execution checkpoint
below or any present-tense validation claim.

**Two declared pillars (August 2026 decision).** (1) Research: guaranteed hierarchical
memory under the Safety Governor plus the capability-transfer path - the properties
llama.cpp, ik_llama.cpp and KTransformers do not offer (see "Prior art" in README for the
checked landscape; ik_llama.cpp already ships Qwen-3.8-Flash-Next and is today's fastest
open option for large-MoE-on-12GB-VRAM, so honest benchmarking must measure against it,
not only mainline llama.cpp). (2) Competitive pursuit as a second pillar, pursued only
through the levers an mmap-only runtime lacks: expert-quant re-import, ggml SIMD default,
and a routing-aware VRAM expert tier that beats the OS page cache.

**Parity is a verification instrument, not a product target.** Greedy token parity with
llama.cpp is required where logit margins are large; near-tie continuations may diverge
documented (D1: the oracle quantizes activations to Q8_0 for every Q8_0 matvec, our
double-precision path is more accurate, so near ties may legitimately flip, oracle
self-consistency notwithstanding). Do not reproduce the oracle's activation quantization
for parity's sake; `--llama-activation-quant` was considered and dropped.

**Execution checkpoint.** The primary track is now **Qwen3.8-Flash-Next**
(`qwen4exp`, 125B MoE, 6B active): the native runtime executes the official
unsloth **Q8_0 GGUF** checkpoint (175.3 GiB, canonical immutable import at
`C:\vastloom-store`, FP8 safetensors deleted after the GGUF replaced it)
end-to-end and achieves **greedy token parity with llama.cpp** (`6fe7498`,
CPU, same checkpoint). Do not claim competitive throughput: the reference-PC greedy CLI runs
~4 s/token against llama.cpp's ~3.2 tok/s on the same machine; trunks stream from NVMe per
token unless an explicit RAM-headroom override permits Q8_0-native residency (~3.3 s/token
when resident), and experts resolve through a bounded Q8_0 cache. A VRAM-resident expert tier
is implemented and bit-identical (`--cuda-experts`) but measures slower than the 12-thread
CPU Q8_0 kernel on the reference RTX 4070 Super — FP64 parity accumulation cannot use the
device's parallel reductions — so it stays opt-in; cross-layer prefetch is not implemented.
The DeepSeek V4
Flash 0731 track is historical: `UD-IQ2_XXS` was its only execution quant,
IQ1 / `UD-IQ1_S` was deleted; its pinned decoders and reduced-graph fixtures
remain as historical tests, not as the execution path.

**Current measured position (reference PC).** Qwen3.8-Flash-Next greedy
generate matches llama.cpp token-for-token ("The capital of France is" →
" Paris. The capital of Germany is Berlin. The capital of Italy is Rome.
The" for 16 tokens) after fixing five real defects surfaced only by the
external oracle: sigmoid DeltaNet output gate (not silu), pre-applied
`-exp(A_log)` in GGUF `ssm_a`, GGUF norm weights stored as `(1+w)` except
RMSNormGated `ssm_norm`, ggml-repeat tile k-head mapping (`vh % k_heads`),
and a stale-trunk bug in the greedy CLI. Partial RoPE (64/256, theta 1e7)
and the PLE n-gram block are integrated. The llama.cpp oracle (clone
`6fe7498` + optional intermediate-tensor dump, local only) is the authority
for every future change; prefer it over this project's own reference
whenever the two disagree.

The longer boundary below is a lab log. Where it says official generate is unvalidated or
that the remaining gap is missing kernels, this paragraph wins.

The current validated fixture paths are the synthetic TinyTransformer,
SmolLM2 dense CPU inference, OLMoE CPU expert streaming and a non-executing DeepSeek V4 Flash
metadata/header inspector plus a closed-world non-executing partial-range trunk binder, native
tokenizer/basic chat framing and bounded equation-level reference operators. A deterministic
two-layer DeepSeek float prefill plus one checkpoint-shaped
MTP block now match the pinned preview graph for prefill and five incremental raw/compressed
cache steps. A bounded autoregressive reference driver also matches eight trusted greedy tokens
and a portable seeded sequence, with per-layer cancellation/timeout checkpoints and exact observed
expert-cache telemetry. Verified headers populate a universal tensor/expert catalog, and a buffer-free
governed state machine validates legal NVMe/RAM/VRAM reservation transitions. Synchronous and
bounded asynchronous fixture paths now load canonical ranges into governed packed RAM buffers
with exact tensor slices and exclusive transfer leases. The router-first reduced fixture now
submits all selected misses through a bounded batch, acquires each expert at its deterministic
compute turn and records positive controlled read-service/CPU-compute overlap without changing
the oracle result. Only one batch may own a cache and abandoned loads are drained and counted as
wasted prefetch. Speculative decoding does not exist yet. A native
Windows/Linux CUDA Driver API boundary now probes device capability without retaining a context
and performs explicit Governor-backed allocations with a live VRAM reserve. Its first synchronous
data-plane slice packs FP32 input, weight and output in one governed device allocation, copies
host-to-device, launches a toolkit-free scalar PTX RMSNorm, copies device-to-host and releases the
allocation. The reference RTX 4070 Super passed the 2x4096 CPU-parity smoke on Windows and WSL
with `0.000025` maximum absolute error. This is one reference operator, not a CUDA model backend.
The newer
official 0731 Unsloth `UD-IQ1_S` GGUF profile now has a bounded v3/split-header reader and a
metadata-only packed-expert catalog: 1,328 physical tensors map to 11,008 logical expert bundles.
A bounded scalar IQ1_S dequantization/matvec oracle matches pinned ggml output on one exact real
layer-0/expert-0 gate slice. Scalar IQ3_XXS parity and one complete real gate/up/SwiGLU/down
expert now also match pinned ggml bit for bit after a governed asynchronous RAM load. Its exact
three-object importer and full-hash/single-tier source-binding gate are implemented, but the real
82.54 GB payload has not been imported or executed. A non-executing checkpoint-shaped layer
planner now combines the exact layer-0 trunk and six caller-supplied complete bundles into
39 immutable-source ranges / 177,795,544 logical bytes. Those IDs have no verified routing
authority and none of the planned payload was fetched. A separate explicit-authority range
importer now validates the plan, exact HTTP spans and per-range hashes, then publishes only a
read-only non-executable derived artifact. The real 138,867,160-byte trunk passed dry-run only;
selected-expert import remains blocked. A native closed-world verifier and exact pinned-catalog
trunk binder pass synthetic faults. One complete four-byte F32 fixture range now moves through
an aggregate-budgeted RAM/staging transaction, is rehashed after read, retains a use lease and
reaches the scalar matvec consumer. A closed-set loader now preflights complete catalog coverage,
reserves aggregate RAM plus maximum-range staging, rehashes every packed subspan and publishes
only the full transaction. All 43
layers now expose deterministic
non-routed trunk plans (1,193 tensors/tiles at a 64 MiB ceiling), and a synthetic governed tile
transaction validates payload reads and cleanup. Scalar Q8_0/Q5_K/Q6_K/BF16/F32 references plus
typed I32 router decoding cover all 1,193 layer-trunk storage types in metadata, but no official
trunk tile was read. A deterministic reduced layer now consumes 11 packed Q8_0 trunk/shared
matrices, eight IQ1_S routed W1/W3 matrices, four IQ3_XXS routed W2 matrices and one I32 hash table;
prefill and incremental results match an independently dequantized oracle. A file-backed
repetition loads 12 complete trunk tiles and four indivisible mixed-format expert bundles, then
retains all 8,180 payload bytes through 16 Governor-backed immutable buffers without copying them
into the graph. A second artifact-backed repetition now loads all 21 layer-0 trunk objects under
canonical 0731 `blk.0.*` names into one 2,844-byte buffer/lease: 11 matrices, nine retained F32
vectors and one I32 table bind through an explicit name/shape adapter. Combined with the four
separately governed expert bundles, the complete reduced prefill/decode oracle remains unchanged.
F32/BF16 decode scratch is capped at 1,048,576 elements. Missing, duplicate, split,
wrong-role/type/source, corrupt and cancelled sets fail transactionally. The reduced MoE graph
now routes first and resolves only the
deduplicated selected complete bundles through a bounded synchronous two-entry RAM LRU with exact
hit/miss/eviction/logical-byte telemetry. Canonical 0731 GGUF identities now bind one complete
6,488,064-byte extracted IQ1_S/IQ3_XXS expert into the same one-entry cache/consumer boundary;
first-miss/second-hit telemetry, pinned ggml output and lease cleanup pass across tested compilers.
The same real expert also passes the asynchronous future/cache/kernel boundary, although one
expert alone cannot prove useful overlap. The mixed-format layer uses reduced synthetic IQ bytes;
the real 4,096/2,048 expert is not yet composed with official trunk payload. The exact pinned
0731 GGUF metadata now derives the native 43-layer reference configuration, including separate
43-value routed/shared SwiGLU clamp schedules and the 46-value compression schedule. A
buffer-free validator maps all 1,193 official non-routed layer descriptors across 43 layers to
exact graph-consumer dimensions and encoded storage sizes: 21-object uncompressed/hash layouts,
31-object 4:1 compressor/indexer layouts and 25-object 128:1 compressor layouts total
6,390,088,008 bytes. No official trunk tensor was loaded. A complete zero-payload checkpoint
catalog pass now closes the six global consumers, all 11,008 expert bundles and all 33,024
logical expert slices as well: 1,328 physical tensors partition into 1,193 layer tensors, six
globals and 129 packed expert containers, while 34,223 logical tensors account for exactly
82,533,892,444 encoded bytes. Q4_K global weights now have a pinned-ggml scalar
dequantization/matvec oracle and a bounded single-row decoder; reduced packed embedding and LM
head prefill/decode match an independently dequantized graph. IQ2_XXS gate/up weights now also
have pinned-ggml scalar dequantization/matvec and row-decoder parity; an alternate reduced graph
with every routed W1/W3 matrix in IQ2_XXS matches its independently dequantized prefill/decode
oracle. MXFP4 now has pinned-ggml scalar dequantization/matvec and row-decoder parity as well; a
second alternate reduced graph with every routed W2 matrix in MXFP4 matches independently
dequantized prefill/decode. The six global consumers now also have an exact pinned-metadata plan:
596,000,788 bytes in six ranges with a 297,861,120-byte maximum. A dedicated non-routed artifact
scope replays every immutable source span, and a reduced Q8_0/F32 fixture loads all six through one
aggregate governed RAM buffer/lease into embedding, output, normalization and Hyper-Connection
prefill/decode parity. The official set passed importer dry-run only. No official IQ2_XXS, MXFP4
or global payload was read. One real DeepSeek layer now executes from official payload: the catalog is relocated onto imported
artifacts, and token 0 runs through Q8_0 attention, Q5_K/Q6_K shared expert, BF16 router, I32 table,
IQ1_S/IQ3_XXS experts and Q4_K embedding to produce 129,280 deterministic layer-0 logits. That is
one layer of 43 and every command reports it as such. `deepseek-forward` executes to an explicit
`--layers` depth straight from a complete checkpoint, refusing in advance any depth whose exact
resident cost exceeds the live safe RAM budget. Full-depth IQ2_XXS generate is validated on the
reference PC (see the execution paragraph at the top of this file). Complete CUDA model
inference is still not a claim: the GPU path is reference kernels plus one RMSNorm smoke, not
a full device graph.

The first official payload is local. Hash-routed layers select experts by table lookup, so the
imported layer-0 trunk (21 ranges, 138,867,160 bytes) answers the routing question without
needing a learned gate: `deepseek_v4_resolve_hash_route` reads the pinned
`blk.0.ffn_gate_tid2eid` table through the governed rehashed range path. Token 0 on that
table resolves to experts 35, 53, 200, 222, 245 and 254. Learned-router layers need the
post-attention hidden state; hash N+1 can prefetch during N, learned N+1 cannot. Physical
device-vs-logical byte counters are reported; do not treat logical expert bytes as SSD
traffic.
Windows and Linux CPU paths are tested; macOS/Metal remain unvalidated. Laguna S 2.1 remains a
reviewed compatibility target, not a validated execution path.

Reference-PC execution targets must pass the public minimum-working-set gate: one complete routed
expert, the required always-active block, KV, scratch and staging must fit live post-reserve
budgets derived from 32 GiB RAM and 12 GiB VRAM. Kimi K3 is capacity-study-only on that machine
unless an exact profile passes this gate; do not download or claim execution for an ineligible
profile merely because its complete weights fit on SSD.

Do not restart or generically redesign the project. Inspect the relevant implementation,
documentation, tests and git state before changing code. Preserve working components and do not
replace real paths with placeholders.

An external reference now exists and is the authority: llama.cpp built from source reads the same
checkpoint. For the Qwen3.8-Flash-Next track it validated greedy parity and found defects no
internal test could — the DeltaNet output gate is sigmoid (not silu as the HF modeling main
carries), GGUF `ssm_a` is pre-transformed, GGUF norm weights are stored as `(1+w)`, and the
k-head mapping is ggml-repeat tile. For the DeepSeek track it found the Hyper-Connection mixing
matrix transposition the same way. Prefer it over this project's own reference whenever the two
disagree, and treat bit-exactness as a property of the storage decoders rather than of accumulation
order: this reference sums in `double` where ggml sums in `float`, and their logits differ by
0.01 to 0.06 by construction.

Measured position: a token still moves 8.76 GB of weights (6.99 GB always-active trunk plus
globals, 1.77 GB selected experts). The Qwen3.8 greedy CLI on the reference PC (CPU, trunk
Q8_0-native residency under a RAM-headroom override, Q8_0-native expert matvec, concurrent
expert reads, 512-entry expert cache) runs ~3.3-4 s/token end to end; llama.cpp `6fe7498` on
the same machine and checkpoint reaches ~3.2 tok/s, so the honest ratio is roughly ten to
twelve times, not competitive. The remaining gap is structural: llama.cpp keeps everything
in Q8_0 under the OS page cache, while our ~7.3 GB Q8_0 trunk residency plus cache and
runtime state still leaves less RAM for the page cache than an mmap-only runtime uses.
Trunk residency was cut from 13 GB of F32 to Q8_0 bytes in P2d (bit-identical, startup
16.8 -> 3.1 s); the CUDA expert tier was measured and rejected as a default (P3a: the FP64
parity accumulation cannot use parallel reductions on consumer parts, so the 12-thread CPU
kernel stays ahead). What could still close the gap is a smaller expert quant (Q4_K/IQ4
re-import: halves the ~2.5 GB/token expert traffic), the ggml SIMD backend promoted to
default (measured ~2x on compute, opt-in via `--ggml-backend`), and a model-aware VRAM
expert tier that beats the page cache by pinning routing-hot experts - the one lever an
mmap-only runtime cannot replicate. The checkpoint declares a 262,144-token context and
the state architecture scales well (36 of 48 layers are O(1) DeltaNet; the 12 QSA layers
add ~72 KB/token of F32 caches, ~9.4 GB at 128k — halving to ~4.7 GB if the caches go
F16), but the serial one-token prefill (~3.5-4 s/token) makes
long contexts impractical today: chunked prefill (roadmap Q11k) is the first lever,
ahead of prefix-KV sessions (Q12) and the Qwen3.8 serve path (Q13). Do not describe any
throughput here as competitive.
Do not compare an IQ2_XXS second-per-token to an older IQ1_S figure as if they were the
same experiment.

Before trusting a number, ask under what conditions it was taken. Five checks in this milestone
could not fail: dyadic parity values that cannot separate f32 from f64; a cost breakdown by
subtraction; a hidden-state norm sampled after the normalization that fixes it; PCIe traffic assumed
equal to logical bytes; and a memory reserve measured in the regime where it is not needed, then
used to justify the regime where it is.

## Safety invariants

- The Safety Governor outranks performance scheduling.
- Reserve large allocations before allocating; preserve host RAM, VRAM and disk headroom.
- Keep queues, reads, buffers, caches, speculation and worker counts bounded.
- Treat original model weights as immutable. Publish derived artifacts only after temporary
  creation, validation, checksum and atomic rename.
- Never add generic uncontrolled swap, implicit precision loss or unverified fallback behavior.
- Do not execute or advertise IQ1 / UD-IQ1_S as the product path. Execution is IQ2_XXS or
  higher.
- On pressure, reduce cache/prefetch/speculation first. On emergency, abort generation and release
  resources instead of risking host instability.
- Keep Windows paths Unicode-safe and common code portable to Linux.

Persisted legacy `stargate.*` identifiers are read-only compatibility contracts. New artifacts
must use `vastloom.*`; never rewrite a user's existing checkpoint merely to rebrand it.

## Workflow

1. Read the relevant docs and tests.
2. Establish a correctness baseline and exact model revision.
3. Implement the smallest native vertical slice.
4. Add deterministic parity, malformed-input and fault tests.
5. Build Release and Debug with warnings as errors.
6. Run ASan/UBSan on Linux where supported.
7. Benchmark only after correctness; record hardware, cache state and bytes/token.
8. Update documentation and add a validation report for a milestone.

Do not download multi-gigabyte models unless the task explicitly requires it and the model store,
free-space reserve and immutable import path have been verified. Never commit downloaded weights,
build directories, credentials, private prompts or machine-specific paths.

## Standard gates

```bash
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release

cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug

python -m compileall -q tools tests
python -m unittest tests/test_manifest_tool.py -v
python -m unittest tests/test_distill_transcript.py -v
python -m unittest tests/test_distill_evaluate.py -v
python tests/test_import_hf_model.py
```

External model tests are opt-in through the `VASTLOOM_*_MODEL` CMake paths. For the Qwen3.8
track: `-DVASTLOOM_QWEN38_MODEL=<imported checkpoint root>` enables
`vastloom_qwen38_checkpoint` (serial, metadata catalog + real layer-0/3 forwards + prefill
determinism on the GGUF layout). Run large tests serially. Distinguish simulated traces,
warm-cache measurements, process-visible I/O and physical device traffic in every report.

The Qwen3.8 CLI surface: `qwen38-inspect` (catalog validation), `qwen38-layer --layer N`,
`qwen38-prefill --tokens` (multi-token prefill, bounded expert LRU), `qwen38-chain --layers`
(consecutive real layers), `qwen38-greedy --prompt-ids` (full-depth greedy generation with
detokenize). The safetensors/GGUF dispatcher selects the checkpoint layout automatically.
llama.cpp comparison tooling (clone `6fe7498` + optional intermediate dump) stays outside the
repository.

## Claims and commits

Never say a model, backend, optimization, package or platform works unless it was built and tested.
Report exact passing counts and environmental failures separately. Keep commits coherent, preserve
unrelated work and do not commit build outputs or the local research clone under `build/`.
