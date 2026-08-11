# M9 DeepSeek IQ2_XXS expert-kernel validation

Validation date: 2026-08-11

Implementation revision:
`2263f49ddc6bf2e5cdaaf5b89381c2e0c8b1d4e9`

## Scope and result

This milestone closes the native scalar execution boundary for the IQ2_XXS gate/up variant in
the pinned Unsloth DeepSeek V4 Flash 0731 `UD-IQ1_S` profile:

```text
pinned GGML IQ2_XXS blocks
  -> bounded little-endian scalar decode
  -> FP32 matvec / one-row lookup
  -> transactional GGML dispatcher
  -> reduced routed W1/W3 + IQ3_XXS W2 graph
  -> prefill and incremental decode parity
```

The deterministic raw-block decoder and matvec match the independently compiled GGML oracle. An
alternate reduced DeepSeek graph replaces every routed W1/W3 matrix with IQ2_XXS, guaranteeing
that every selected expert crosses the new kernel; prefill and incremental main/MTP results match
the same graph with those exact bytes independently dequantized to FP32.

No official IQ2_XXS tensor payload, complete layer, routed token or checkpoint was downloaded or
executed. `execution_supported=false` remains correct for the official profile.

## Pinned format and independent oracle

The format and immutable 256-entry magnitude codebook are pinned to ggml commit
`30bf8685ed4eb0a47f2b06229543327749904150` under its MIT license. The local codebook matches all
256 upstream entries and has the compile-time little-endian FNV-1a fingerprint
`bb4ee025b5ac6e8e`.

One IQ2_XXS block represents 256 values in 66 bytes:

- one little-endian FP16 block scale;
- 32 packed 16-bit words, interpreted in eight groups of 32 values;
- four 8-bit codebook indices per group;
- four 7-bit sign indices plus a 4-bit local-scale field in the remaining packed word.

An ignored helper was compiled directly against the pinned local ggml library and evaluated six
deterministic blocks before the Vastloom expectations were added. Its 1,536 decoded FP32 values
produced fingerprint `9bca8e03f48dafe9`. Selected exact outputs were:

| Index | FP32 value |
|---:|---:|
| 0 | 9.375 |
| 7 | 3 |
| 8 | -16.125 |
| 31 | -3 |
| 32 | 5 |
| 63 | -5 |
| 127 | -9 |
| 255 | 17 |
| 256 | 2.5 |
| 511 | 8.5 |

The independent 3-row by 512-column FP32 dot products were
`-32.86767578125`, `-284.89828491210938` and `-0.011499286629259586`.

## Native boundary

The implementation adds:

- bounded `dequantize_iq2_xxs` and `matvec_iq2_xxs_f32` entry points;
- exact encoded-size, complete-block, matrix-shape and configured-element-limit checks;
- rejection of non-finite FP16 scales and non-finite matvec inputs;
- IQ2_XXS support in the native storage/matvec capability query;
- one-row decoding from a complete validated packed matrix without materializing other rows;
- transactional GGML dispatch that publishes caller-visible output only after every row passes;
- a local immutable codebook with a compile-time fingerprint and no runtime ggml dependency.

The existing generic packed-matrix consumer then admits IQ2_XXS without a model-specific fallback
or precision change.

## Reduced graph validation

The original reduced 0731-format fixture remains byte-for-byte stable with IQ1_S W1/W3 and
IQ3_XXS W2. A separate graph copy replaces all eight routed W1/W3 matrices across four experts
with deterministic IQ2_XXS bytes while retaining IQ3_XXS down projections. The test:

1. independently dequantizes the exact packed bytes into an FP32 oracle graph;
2. executes packed prefill through routing, selected experts, logits and MTP logits;
3. executes one incremental decode step with retained reference state;
4. compares logits, greedy IDs, routes, routing weights and attention indices;
5. corrupts every IQ2_XXS scale and verifies fail-closed prefill.

Replacing every routed gate/up matrix means the test cannot pass by selecting an unchanged
IQ1_S-only expert.

## Malformed-input and transactional tests

The native suite rejects:

- truncated encoded buffers;
- empty or non-256-aligned output spans;
- columns that are not a complete IQ2_XXS block;
- configured element ceilings below the requested operation;
- out-of-range row IDs;
- non-finite block scales;
- non-finite matvec inputs;
- corrupt selected graph matrices.

Failed full dequantization, row decoding and transactional matvec leave the corresponding
caller-visible destination unchanged.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported VRAM, driver 610.88;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- project warnings treated as errors in every compiler build;
- no CUDA code and no official DeepSeek tensor payload executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release + real metadata opt-ins | 15/15 passed |
| GCC Debug + real metadata opt-ins | 15/15 passed |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed |
| Clang 18 Release + real metadata opt-ins | 15/15 passed |
| GCC Debug + libstdc++ debug iterators/assertions + real metadata | 15/15 passed |
| MSVC Release x64 + real metadata opt-ins | 15/15 passed |
| MSVC Debug x64 + real metadata opt-ins | 15/15 passed |
| Python manifest suite | 4/4 passed, one Windows symlink case skipped |
| Python importer suite and `compileall` | passed |

The two native TSan binaries passed with `TSAN_OPTIONS=halt_on_error=1` under
`setarch x86_64 -R`. This avoids the known WSL address-layout conflict; it is not presented as a
complete TSan CTest matrix.

## Performance statement

No token rate, full-matrix throughput, physical-device bandwidth, bytes/token or speedup is
reported. The scalar implementation is a correctness oracle. The graph fixture is deliberately
small and resident, so it does not measure official expert payload access, NVMe overlap or CUDA.

## Known limitations and next bottleneck

- No official IQ2_XXS expert slice was acquired or executed.
- MXFP4 down-expert variants remain layout-only.
- The six global tensors do not yet share one retained governed binding transaction.
- The official `tid2eid` payload has not produced an authoritative route for a real prompt.
- The reference graph has no complete 43-layer official-payload loop or generation driver.
- Physical NVMe overlap, pinned host memory, VRAM placement and CUDA remain unimplemented.

The next safe no-download slice is pinned-ggml MXFP4 parity, followed by the retained six-global
binding transaction. Only after those correctness gates should an explicitly authorized range
import fetch the layer-0 trunk and authoritative selected experts.
