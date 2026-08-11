# M9 DeepSeek Q4_K global-kernel validation

Validation date: 2026-08-11

Implementation commit: `795d6e2b3f594e3c71b717bbc507ca4006829dab`

## Scope

This milestone closes the first executable storage-format gap outside the per-layer catalog:

```text
pinned GGML Q4_K blocks
  -> bounded scalar dequantization
  -> transactional FP32 matvec
  -> complete-matrix / single-row lookup
  -> packed token embedding and LM head
  -> reduced DeepSeek prefill and incremental parity
```

The official `token_embd.weight` and `output.weight` payloads were not acquired, bound or
executed. This report proves the Q4_K kernel and graph boundary with deterministic raw blocks and
a reduced graph; it does not prove a complete official DeepSeek layer, token or generation.

## Pinned correctness oracle

The Q4_K block contract is pinned to
[`ggml` commit `30bf8685ed4eb0a47f2b06229543327749904150`](https://github.com/ggml-org/ggml/blob/30bf8685ed4eb0a47f2b06229543327749904150/src/ggml-quants.c)
under its MIT license. The independent local oracle invoked ggml's own
`dequantize_row_q4_K` on six deterministic 144-byte blocks. Vastloom consumes the same raw bytes,
not bytes produced by its own code.

| Oracle value | Expected result |
|---|---:|
| FP32-output FNV-1a fingerprint | `84e8e84336ea0ad7` |
| row 0 FP64-accumulated FP32 result | `281.6796875` |
| row 1 FP64-accumulated FP32 result | `-792.12872314453125` |
| row 2 FP64-accumulated FP32 result | `0.0096547976136207581` |

Pinned individual values cover both nibbles, all four 64-value groups, multiple 6-bit scale/min
packing branches, positive/negative FP16 super-scales, subnormal scales and block boundaries.
The upstream license is retained in `third_party/ggml/LICENSE` and attributed by `NOTICE`.

## Native implementation

The scalar reference path implements the little-endian 256-value/144-byte Q4_K block:

- FP16 super-scale and FP16 super-minimum;
- eight packed 6-bit local scales and eight packed 6-bit local minimums in 12 bytes;
- four groups of 32 low-nibble plus 32 high-nibble values;
- `d * local_scale * q - dmin * local_minimum` reconstruction;
- FP64 accumulation followed by checked FP32 publication.

Both FP16 fields in every selected block must be finite. Dimensions, column block alignment,
encoded byte count, input finiteness and the explicit element ceiling are validated before
matvec output publication. Q4_K is exposed through the existing transactional GGML dispatcher
only after this parity gate.

`dequantize_ggml_row_f32` validates the complete matrix descriptor and exact encoded span, checks
the requested row and caller-owned storage-width output, computes its byte offset with overflow
guards and decodes only that row. A corrupt selected row leaves its output unchanged. This avoids
materializing a full FP32 embedding table.

The graph keeps a separate 1,073,741,824-element packed-descriptor ceiling. It admits the pinned
129,280 x 4,096 global matrices (529,530,880 logical elements each) while the ordinary live graph
allocation ceiling remains 64 million elements. Per-call kernel limits are narrowed to the exact
validated packed matrix element count.

## Graph integration

The reference graph now resolves `embed.weight` from either the existing FP32 fixture store or a
`DeepseekV4PackedMatrix`. Packed prefill and incremental decode request only the rows named by the
validated token IDs, copy only the logical width from physical storage and reuse those rows for
the MTP embedding branch. `head.weight` already uses the packed linear dispatcher and now accepts
Q4_K through the parity-tested kernel.

The deterministic integration test replaces both global matrices in the reduced two-layer graph:

- 16 physical embedding rows, logical width 8, storage width 256;
- 16 physical LM-head rows, logical width 8, storage width 256;
- FP32 oracle tensors built from the independently pinned Q4_K decoder output;
- exact main/MTP logits and greedy-token parity in prefill;
- exact main/MTP logit parity on one incremental cache step;
- equal retained attention/compressor/indexer state after prefill.

## Fault coverage

Native tests additionally reject:

- truncated Q4_K blocks and complete matrices;
- a non-finite super-scale or super-minimum;
- output not aligned to 256 values;
- matrix columns not aligned to a complete block;
- configured element ceilings below the requested operation;
- out-of-range row IDs and wrong row-output sizes;
- unsupported I32-to-FP32 row decoding;
- overlapping transactional buffers;
- truncated packed graph embedding payloads;
- packed embedding storage width below its logical width;
- a non-finite scale in a selected packed embedding row.

The public dequantized row, transactional matvec output and caller-visible prefill state remain
unchanged when the corresponding operation fails.

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

No token rate, full-table throughput, physical-device bandwidth, bytes/token or speedup is
reported. The scalar implementation is a correctness oracle. The graph fixture is deliberately
small and resident; it does not measure the 297,861,120-byte official embedding or output object,
SSD access or CUDA execution.

## Known limitations and next bottleneck

- The two official Q4_K global payloads were not acquired or bound through retained model-store
  leases.
- The other four global F32 objects have a validated catalog contract but no complete six-object
  retained global binder.
- IQ2_XXS gate/up and MXFP4 down expert variants remain layout-only.
- The official `tid2eid` payload has not produced an authoritative route for a real prompt.
- The reference graph still has no complete 43-layer official-payload loop or generation driver.
- Physical NVMe overlap, pinned host memory, VRAM placement and CUDA remain unimplemented.

The next safe no-download slice is pinned-ggml IQ2_XXS parity, followed by MXFP4 and a retained
six-global binding transaction. Only after those correctness gates should an explicitly
authorized range import fetch the layer-0 trunk and authoritative selected experts.
