# Agent contribution guide

This file applies to the entire repository. Human-facing contribution rules are in
`CONTRIBUTING.md`; automated coding agents must follow both documents.

## Mission and current boundary

Vastloom is a native, hardware-adaptive local inference runtime that treats VRAM, RAM and
NVMe as a model-aware hierarchy. The current validated paths are the synthetic TinyTransformer,
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
wasted prefetch. Speculative decoding and CUDA inference kernels do not exist yet. A native
Windows/Linux CUDA Driver API boundary now probes device capability without retaining a context
and performs only explicit Governor-backed allocations with a live VRAM reserve; the reference
RTX 4070 Super passed a 4 MiB allocate/release smoke on Windows and WSL. The newer
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
or global payload was read. A complete
0731 quantized layer/token
path, physical NVMe overlap,
official-payload generation and CUDA inference remain unvalidated.
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

## Safety invariants

- The Safety Governor outranks performance scheduling.
- Reserve large allocations before allocating; preserve host RAM, VRAM and disk headroom.
- Keep queues, reads, buffers, caches, speculation and worker counts bounded.
- Treat original model weights as immutable. Publish derived artifacts only after temporary
  creation, validation, checksum and atomic rename.
- Never add generic uncontrolled swap, implicit precision loss or unverified fallback behavior.
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
python tests/test_import_hf_model.py
```

External model tests are opt-in through the `VASTLOOM_*_MODEL` CMake paths. Run large tests
serially. Distinguish simulated traces, warm-cache measurements, process-visible I/O and physical
device traffic in every report.

## Claims and commits

Never say a model, backend, optimization, package or platform works unless it was built and tested.
Report exact passing counts and environmental failures separately. Keep commits coherent, preserve
unrelated work and do not commit build outputs or the local research clone under `build/`.
