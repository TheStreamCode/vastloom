# SIMD backend experiment — scalar vs ggml

Date: 2026-10-26. This is a speed experiment on the existing DeepSeek range slices, not a default
change and not a product claim. It answers the first question from
[`A_SLICE_RESEARCH_FINDINGS.md`](../planning/A_SLICE_RESEARCH_FINDINGS.md): how much does the
optional ggml SIMD backend actually buy on this workload, and at what numerical cost?

## Setup

- Pinned external ggml checkout: commit `30bf8685ed4eb0a47f2b06229543327749904150` (the commit
  recorded in `NOTICE`), configured with `-DVASTLOOM_GGML_ROOT=<checkout>`.
- Standard scalar build: `build/release` (no ggml checkout).
- SIMD build: `build/ggml-release`, same sources and flags.
- Artifacts: the same D: store slices used by the P0 experiments (metadata store, three layer
  trunks, globals, selected experts). `--cpu-threads 12` on both builds.
- `--ggml-compute` is now accepted by `deepseek-layer-probe` and `deepseek-chain-probe`; a standard
  build without the checkout refuses it with "This build has no ggml checkout".

## Results

Layer 0, token 0:

| Build | Elapsed | ggml accelerated / fallback | Logits vs scalar |
|---|---:|---|---|
| scalar | 344.96 ms | — | baseline |
| ggml | **27.50 ms** (12.5×) | 36 / 2 | max abs 0.1890, mean 0.0257, argmax 30,528 identical |

Chain, token 0:

| Build | Elapsed | ggml accelerated / fallback | Logits vs scalar |
|---|---:|---|---|
| scalar | 2,059.57 ms | — | baseline |
| ggml | **1,291.46 ms** (1.6×) | 112 / 4 | max abs 0.1686, mean 0.0231, argmax 86,051 identical |

Chain, prompt `[0,1,2,3,4]`:

| Build | Elapsed | Logits vs scalar |
|---|---:|---|
| scalar | 5,316.70 ms | baseline |
| ggml | **1,658.22 ms** (3.2×) | max abs 0.4636, mean 0.0635, argmax 20,267 identical |

The same run repeated with scalar and ggml produced the same argmax and the same routes in every
case; the logits are **not** bit-identical. The maximum deviation grows with depth (0.19 at layer 0,
0.46 at the five-token chain), consistent with float accumulation in the SIMD kernels versus the
double accumulation in the scalar reference, amplified across layers.

## What this means

- The SIMD backend is the **largest measured speed lever** on this path: 12.5× on one layer and
  3.2× on the five-token chain. It is far larger than the I/O changes (read-call reduction was
  neutral) and the thread scaling (2× from 1 to 12 threads, default already optimal).
- It is **not yet a default**: the logits differ beyond the project's oracle tolerance (1e-3) and
  the backend's numerical behavior must be characterized before it can replace the scalar reference.
  The scalar path remains the correctness baseline.
- The next experiment is a **parity/quality gate**: determine whether the ggml kernel can be made
  to accumulate in double (or whether a mixed-precision variant keeps the deviation within the
  oracle tolerance), and compare continuation-level behavior, not only the last-position logits.
  Until that gate exists, `--ggml-compute` stays opt-in and the default is unchanged.

## Boundaries

- One host, one build, one set of range slices; no full-depth or full-model claim.
- The scalar and SIMD builds are the same sources; only the external ggml checkout differs.
- The ggml checkout is a build-time dependency and is not committed; the standard build remains
  dependency-free and refuses the flag.
- No throughput claim for a product: the measured numbers are a dated experiment on three layers of
  43.

## Reproduction

```powershell
git clone https://github.com/ggml-org/ggml.git build\ggml-pinned
git -C build\ggml-pinned checkout 30bf8685ed4eb0a47f2b06229543327749904150
cmake -S . -B build\ggml-release -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DVASTLOOM_BUILD_TESTS=OFF -DVASTLOOM_WARNINGS_AS_ERRORS=ON `
  -DVASTLOOM_GGML_ROOT="$PWD\build\ggml-pinned"
cmake --build build\ggml-release --parallel
build\ggml-release\vastloom.exe deepseek-chain-probe <metadata> <globals> --layers 3 --token 0 `
  --trunk 0=<l0> --trunk 1=<l1> --trunk 2=<l2> `
  --experts 0=<e0> --experts 1=<e1> --experts 2=<e2> `
  --cpu-threads 12 --ggml-compute --dump-logits ggml.f32 --json
```

Compare the dump with the scalar build's dump; the JSON `ggml_compute` object reports accelerated
and fallback matvec counts.
