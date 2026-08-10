# Agent contribution guide

This file applies to the entire repository. Human-facing contribution rules are in
`CONTRIBUTING.md`; automated coding agents must follow both documents.

## Mission and current boundary

Vastloom is a native, hardware-adaptive local inference runtime that treats VRAM, RAM and
NVMe as a model-aware hierarchy. The current validated paths are the synthetic TinyTransformer,
SmolLM2 dense CPU inference, OLMoE CPU expert streaming and a non-executing DeepSeek V4 Flash
metadata/header inspector plus native tokenizer/basic chat framing and bounded equation-level
reference operators. A deterministic two-layer DeepSeek float prefill plus one checkpoint-shaped
MTP block now match the pinned preview graph for prefill and five incremental raw/compressed
cache steps. Verified headers populate a universal tensor/expert catalog, and a buffer-free
governed state machine validates legal NVMe/RAM/VRAM reservation transitions. Synchronous and
bounded asynchronous fixture paths now load canonical ranges into governed packed RAM buffers
with exact tensor slices and exclusive transfer leases. Full-checkpoint payload binding,
compute-coupled shared streaming, speculative generation and CUDA do not exist yet. The newer
official 0731 Unsloth `UD-IQ1_S` GGUF profile now has a bounded v3/split-header reader and a
metadata-only packed-expert catalog: 1,328 physical tensors map to 11,008 logical expert bundles.
A bounded scalar IQ1_S dequantization/matvec oracle matches pinned ggml output on one exact real
layer-0/expert-0 gate slice. Scalar IQ3_XXS parity and one complete real gate/up/SwiGLU/down
expert now also match pinned ggml bit for bit after a governed asynchronous RAM load. Direct
verified full-shard binding, a complete 0731 layer/token path, generation and CUDA remain
unvalidated. Windows and Linux CPU paths are tested; macOS/Metal remain unvalidated. Laguna S 2.1
remains a reviewed compatibility target, not a validated execution path.

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
