# M9 DeepSeek MXFP4 expert-kernel validation

Validation date: 2026-08-11

Implementation revision:
`006a99e0ded3474507533d78392644834fafdadd`

## Scope and result

This milestone closes the native scalar execution boundary for the MXFP4 down-projection variant
in the pinned Unsloth DeepSeek V4 Flash 0731 `UD-IQ1_S` profile:

```text
pinned GGML MXFP4 blocks
  -> bounded E8M0/E2M1 scalar decode
  -> FP32 matvec / one-row lookup
  -> transactional GGML dispatcher
  -> reduced routed IQ1_S W1/W3 + MXFP4 W2 graph
  -> prefill and incremental decode parity
```

The deterministic raw-block decoder and matvec match an independently compiled GGML oracle. A
separate DeepSeek graph replaces every routed W2 matrix with MXFP4, guaranteeing that every
selected expert crosses the new kernel; prefill and incremental main/MTP results match the same
graph with those exact bytes decoded to FP32 before execution.

No official MXFP4 tensor payload, complete layer, routed token or checkpoint was downloaded or
executed. `execution_supported=false` remains correct for the official profile.

## Pinned format and independent oracle

The format, E8M0 scale conversion and doubled E2M1 value table are pinned to ggml commit
`30bf8685ed4eb0a47f2b06229543327749904150` under its MIT license.

One MXFP4 block represents 32 values in 17 bytes:

- one unsigned E8M0 exponent byte;
- 16 bytes containing two E2M1 nibbles each;
- low nibbles map to output positions 0-15 and high nibbles to positions 16-31;
- the pinned half-scale conversion compensates for the doubled E2M1 table.

An ignored helper was compiled directly against the pinned local ggml library and evaluated six
deterministic blocks before the Vastloom expectations were added. Its 192 decoded FP32 values
produced fingerprint `53a1edd1080feb82`. Selected exact outputs included:

| Index | FP32 value |
|---:|---:|
| 0 | -3 |
| 7 | -8 |
| 15 | 8 |
| 31 | 2 |
| 64 | 1.7632415262334313e-38 |
| 95 | 3.5264830524668625e-38 |
| 96 | -1.1754943508222875e-38 |
| 127 | -5.8774717541114375e-39 |
| 160 | 1024 |
| 191 | -1536 |

The independent three-row by 64-column FP32 dot products were `13.18359375`,
`3.3634750467864281e-38` and `-213.97833251953125`.

## Native boundary

The implementation adds:

- bounded `dequantize_mxfp4` and `matvec_mxfp4_f32` entry points;
- exact encoded-size, complete-block, matrix-shape and configured-element-limit checks;
- exact handling of the two subnormal E8M0 edge encodings;
- rejection of scaled E2M1 values that overflow FP32 and of non-finite matvec inputs;
- MXFP4 support in the native storage/matvec capability query;
- one-row decoding from a complete validated packed matrix without materializing other rows;
- transactional GGML dispatch that publishes caller-visible output only after every row passes;
- no runtime ggml dependency or precision fallback.

The whole MXFP4 matrix is prevalidated before full dequantization publishes its first value. This
is required because the E8M0 byte is finite by construction while particular high-exponent/E2M1
products may overflow.

## Reduced graph validation

The original reduced 0731-format fixture remains byte-for-byte stable with IQ1_S W1/W3 and
IQ3_XXS W2. A separate graph copy replaces all four routed W2 matrices across four experts with
deterministic MXFP4 bytes while retaining IQ1_S gate/up projections. The test:

1. decodes the exact packed bytes into an FP32 oracle graph;
2. executes packed prefill through routing, selected experts, logits and MTP logits;
3. executes one incremental decode step with retained reference state;
4. compares logits, greedy IDs, routes, routing weights and attention indices;
5. injects overflowing E8M0/E2M1 blocks into every MXFP4 matrix and verifies fail-closed prefill.

Replacing every routed down matrix means the test cannot pass by selecting an unchanged
IQ3_XXS-only expert.

## Malformed-input and transactional tests

The native suite rejects:

- truncated encoded buffers;
- empty or non-32-aligned output spans;
- columns that are not a complete MXFP4 block;
- configured element ceilings below the requested operation;
- out-of-range row IDs;
- E8M0/E2M1 products that overflow FP32;
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
| GCC Release + real metadata opt-ins | 15/15 passed in 12.66 s |
| GCC Debug + real metadata opt-ins | 15/15 passed in 28.49 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 81.11 s |
| Clang 18 Release + real metadata opt-ins | 15/15 passed in 12.87 s |
| GCC Debug + libstdc++ debug iterators/assertions + real metadata | 15/15 passed in 47.23 s |
| MSVC Release x64 + real metadata opt-ins | 15/15 passed in 10.50 s |
| MSVC Debug x64 + real metadata opt-ins | 15/15 passed in 55.43 s |
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

- No official MXFP4 expert slice was acquired or executed.
- The six global tensors do not yet share one retained governed binding transaction.
- The official `tid2eid` payload has not produced an authoritative route for a real prompt.
- The reference graph has no complete 43-layer official-payload loop or generation driver.
- Physical NVMe overlap, pinned host memory, VRAM placement and CUDA remain unimplemented.

The next safe no-download slice is the retained six-global binding transaction. Only after that
correctness gate should an explicitly authorized range import fetch the layer-0 trunk and
authoritative selected experts.
