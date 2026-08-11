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
cache steps. Verified headers populate a universal tensor/expert catalog, and a buffer-free
governed state machine validates legal NVMe/RAM/VRAM reservation transitions. Synchronous and
bounded asynchronous fixture paths now load canonical ranges into governed packed RAM buffers
with exact tensor slices and exclusive transfer leases. The router-first reduced fixture now
submits all selected misses through a bounded batch, acquires each expert at its deterministic
compute turn and records positive controlled read-service/CPU-compute overlap without changing
the oracle result. Only one batch may own a cache and abandoned loads are drained and counted as
wasted prefetch. Speculative generation and CUDA do not exist yet. The newer
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
into the graph. A second artifact-backed repetition loads all 12 non-routed Q8_0/I32 catalog
entries into one 2,644-byte buffer/lease, combines them with the four separately governed expert
bundles and preserves the complete reduced prefill/decode oracle. Missing, duplicate, split,
wrong-role/type/source, corrupt and cancelled sets fail transactionally. The reduced MoE graph
now routes first and resolves only the
deduplicated selected complete bundles through a bounded synchronous two-entry RAM LRU with exact
hit/miss/eviction/logical-byte telemetry. Canonical 0731 GGUF identities now bind one complete
6,488,064-byte extracted IQ1_S/IQ3_XXS expert into the same one-entry cache/consumer boundary;
first-miss/second-hit telemetry, pinned ggml output and lease cleanup pass across tested compilers.
The same real expert also passes the asynchronous future/cache/kernel boundary, although one
expert alone cannot prove useful overlap. The mixed-format layer uses reduced synthetic IQ bytes;
the real 4,096/2,048 expert is not yet composed with official trunk payload. Retained F32/BF16
vector/scalar graph bindings and every official trunk tensor remain absent. A complete 0731
quantized layer/token path, physical NVMe overlap, generation and CUDA remain unvalidated.
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
