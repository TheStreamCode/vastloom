# A — slice research findings and optimization agenda

Date: 2026-10-26. Scope: run the available validation suite on the reduced DeepSeek slices, record
what the measurements say about inference speed, and rank the next optimization experiments. This
is a research note, not a new claim: nothing here is faster until a dated report says so.

## Suite status

The full debug suite was run with the imported range artifacts (metadata store, three layer trunks,
globals and selected experts) and the release multi-token pack contract was run separately. Results
are recorded in the final section; the environmental caveat is the CLI's hard-coded 300 GiB storage
reserve, which refuses to start real-model work on a volume with less free space. The tests were
therefore pointed at a volume that satisfies the reserve; that is a host condition, not a code or
safety change.

The measured scope remains one real layer and a three-layer hash-routed chain. No full-depth,
full-model, throughput or placement claim is made.

## Thread-scaling experiment (first lever measured)

`deepseek-chain-probe` and `deepseek-layer-probe` now accept `--cpu-threads N` so the scalar
reference matvecs can be swept without rebuilding. The same artifacts and outputs were used
(argmax unchanged in every row):

| Threads | token 0 | five-token |
|---:|---:|---:|
| 1 | 4,270 ms | 13,360 ms |
| 2 | 3,382 ms | 9,275 ms |
| 4 | 2,409 ms | 7,537 ms |
| 8 | 2,186 ms | 5,733 ms |
| 12 | 2,053 ms | 5,463 ms |
| 16 | 4,053 ms | 13,247 ms |

The default policy thread count is **12**, and it is the measured optimum on this host; 16 threads
oversubscribe the physical cores and are dramatically slower. Exposing the override is useful for
experiments, but it does **not** improve the default. The next real compute lever is the SIMD/ggml
backend or a different quantized kernel, not more threads.

### SIMD follow-up (measured)

The pinned ggml backend was then built as an external checkout and compared on the same slices:
layer 0 went from 344.96 ms to **27.50 ms (12.5×)**, token 0 from 2,059.57 ms to **1,291.46 ms
(1.6×)** and the five-token chain from 5,316.70 ms to **1,658.22 ms (3.2×)**, with identical argmax
and routes but logits differing by up to 0.19/0.17/0.46. That is the largest speed lever found, but
it is **not** bit-identical and exceeds the oracle tolerance, so it stays opt-in and the default is
unchanged. See [`SIMD_BACKEND_EXPERIMENT.md`](../reports/SIMD_BACKEND_EXPERIMENT.md); the next step
is a parity/quality gate for the SIMD accumulation, not a default switch.

## What the measurements already say

1. **Compute dominates at this slice scale.** Token 0 moves 135,659,520 expert bytes and takes
   ~2.1 s; the five-token prompt moves 663,224,320 expert bytes and takes ~5.3 s. The pack reduces
   range reads 3× (54→18 and 264→88) but the warm and cold wall times do not improve. A ~1.7 GB
   slice streamed at consumer-SSD speed is a small fraction of the matvec time.
2. **The bypass path is not a win on the reference host.** The direct-I/O proxy and the cold-cache
   run both show bypass slightly slower than buffered for these recurring ranges. That matches the
   earlier in-tree measurement; the right default remains buffered, with bypass reserved for cold
   or much larger records.
3. **The cache plan is a safety bound, not a speed lever.** The plan reports floor, live headroom,
   ceiling and chosen entries, and fails closed outside them; it prevents an inadmissible cache but
   does not make a larger cache faster. The current chain capacity is already below the ceiling.
4. **Usage-profile recording alone cannot move bytes.** It records the heat signal for a future
   placement policy; without pinning/protected entries and a workload that evicts, it has no runtime
   effect.

## Ranked optimization agenda

### 1. Measure and improve the CPU compute path (highest expected value)

The reference chain uses the scalar reference kernels. The repository already has an opt-in ggml
SIMD backend for other paths; the first experiment is to measure whether the DeepSeek graph can use
it or whether the bottleneck is the scalar quantized matvec, thread count or memory layout.

**Experiment:** run the same chain with the existing `--ggml-compute` path where supported, and with
a controlled CPU-thread count; record per-layer time, kernel time and expert-bytes/time. A 2×
compute win would matter far more than the 3× read-call reduction.

**Gate:** identical logits/hidden, a dated report, and a measured per-layer improvement or an
explicit negative result.

### 2. Prefetch the predictable next hash layer in the chain

Layers 0–2 are hash-routed, so layer N+1's experts are knowable before layer N finishes. The
full-depth forward path already has a bounded asynchronous prefetch mechanism; the chain probe is
synchronous. The experiment is to issue next-layer expert reads during the current layer's compute
and measure overlap and wasted bytes.

**Gate:** same outputs; prefetch hit-rate, overlap and wasted bytes reported; net latency win or the
feature stays off.

### 3. Pinned/protected cache entries driven by the usage profile

The recording-only profile is the input for a real placement policy. Under a cache smaller than the
prompt's expert union, the current LRU evicts entries that the next token will need. A protected
set (top heat entries, hysteresis, bounded swap count) could turn the recorded signal into fewer
misses.

**Gate:** same outputs; hit rate and bytes/token improve on a repeated workload; no precision or
routing change.

### 4. Expert quantization/size trade with a quality gate

The checkpoint offers IQ2_XXS, IQ2_M, IQ3_S and larger. A smaller expert representation cuts bytes
per token, but changes quality. The project's correctness-first rule means any such experiment is
opt-in, warned, and measured with a continuation-level quality gate in addition to logits/KL.

**Gate:** a new imported range set, an independent quality comparison, and a report that separates
quality from speed.

### 5. Full-depth and end-to-end token rate (blocked)

The current runtime cannot execute learned-router or full-depth layers; a full checkpoint download
does not change that by itself. This remains blocked until the executor, import path and oracle for
those layers exist. Do not use a partial-depth timing as a model throughput result.

## Immediate next experiment

The thread-scaling half of **1 (CPU compute path)** is now measured: the default 12-thread policy
is the optimum, and 16 threads oversubscribe badly. The next experiment is the **SIMD/ggml backend
or an alternative quantized matvec kernel**; it needs no new weights, only the existing three-layer
artifacts and a controlled A/B. The second is **2 (hash-layer prefetch)**, which is implementable in
the chain executor without new artifacts.

## Final suite result

A full debug CTest run (31 of 32 tests; the slow multi-token pack test was excluded and passed
separately in release) produced **29 passed, 2 failed, 7 skipped**:

- The two failures were environmental: `vastloom_deepseek_v4_gpu_residency` and
  `vastloom_deepseek_v4_expert_pack_cli` both reported `Less than 8 GiB of RAM headroom is currently
  available` while unrelated user processes (ComfyUI and a Rust build) held most of the 32 GiB. The
  same tests passed earlier when the host had headroom (GPU residency 266.57 s; pack CLI 443.37 s in
  debug, and the release multi-token contract passed).
- The seven skips were the optional ggml/CUDA probes and the DeepSeek oracle/chain/import tests,
  which are gated by host pressure or optional external inputs; each had passed individually when
  the host was free (chain probe 587.64 s, chain oracle 780.83 s, layer oracle 87.14 s in earlier
  runs).
- No test failed on a parity, format or logic assertion.

This is a host-capacity result, not a regression. The suite must be run serially on an otherwise
idle machine; the Governor refusing to start is the designed behavior.

## Reproduction

Use the existing range artifacts and the CMake opt-in variables; run the CLI from a volume that
satisfies the 300 GiB storage reserve, or the Governor will (correctly) refuse to start. The pack,
cache-plan and usage-profile commands are documented in
[`docs/USAGE.md`](../USAGE.md) and
[`docs/reports/P0_EXPERT_PACK_IO_VALIDATION.md`](../reports/P0_EXPERT_PACK_IO_VALIDATION.md).
