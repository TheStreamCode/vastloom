# M9 bounded layer-trunk streaming validation

Validation date: 2026-08-10

Validated runtime commit: `98e91fcc3c3824c7a851b61e0ab760371beb676d`

## Scope

This slice turns the streamed-trunk capacity profile into a native, model-independent I/O
primitive:

```text
verified logical tensor catalog
  -> exact `blk.N.*` layer identity
  -> deterministic non-routed tensor order
  -> at most 64 MiB immutable source tiles
  -> RAM destination + staging reservation
  -> bounded read-only range load
  -> RAM_READY publication + use lease
  -> eviction and zero reservations
```

Routed experts are deliberately excluded. One complete expert bundle remains indivisible and must
fit wholly in one safe RAM tier or wholly in one safe VRAM tier. RAM and VRAM are never pooled to
admit an oversized expert. This is the reference-PC compatibility boundary: Kimi K3 remains a
metadata/capacity study on 32 GiB RAM and 12 GiB VRAM unless a specific profile passes that rule.

This report does not claim a complete DeepSeek layer, token, generation loop, CUDA backend or
physical-NVMe performance result.

## Implemented

- Every non-routed DeepSeek GGUF tensor named `blk.N.*` now retains its checked layer identity in
  the universal logical catalog.
- `build_layer_streaming_plan` selects one layer's non-routed tensors, excludes routed experts and
  splits every immutable source range into deterministic tiles no larger than a caller-supplied
  bound. Tensor count, tile count, total bytes and offsets use explicit limits and overflow checks.
- Metadata-only catalogs may produce capacity plans but cannot load payload bytes.
- `load_tensor_tile` verifies the tile against its catalog identity, checks its residency object,
  reserves the complete RAM destination and staging amount before allocation, performs one bounded
  Unicode-safe read, handles cancellation before and after I/O, and publishes `RAM_READY` only
  after the entire tile succeeds.
- The returned buffer owns an RAII use lease. Destruction releases physical bytes before the
  reservation becomes evictable.
- The DeepSeek checkpoint JSON/text report exposes the exact per-layer plan envelope.

## Real 0731 metadata result

The already-imported 5,343,584-byte metadata/header checkpoint was inspected at pinned revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`. No large payload object was downloaded or read.

| Property | Result |
|---|---:|
| Planned layers | 43 / 43 |
| Non-routed layer tensors | 1,193 |
| 64 MiB tiles | 1,193 |
| Minimum encoded trunk bytes in one layer | 138,867,160 |
| Maximum encoded trunk bytes in one layer | 161,707,992 |
| Largest individual trunk tensor | 35,651,584 bytes |
| Largest planned tile | 35,651,584 bytes |
| Configured tile ceiling | 67,108,864 bytes |
| Largest complete expert bundle | 8,781,824 bytes |

The fact that tile count equals tensor count is specific to this quantization: every individual
non-routed layer tensor is already below 64 MiB. A complete layer is not loaded at once. Other
models or quantizations may split one tensor into multiple tiles under the same contract.

Five warm MSVC Release metadata commands took 375.12–415.43 ms, median 392.44 ms. That command
parses headers and verifies 5.34 MB of metadata files; it is not a checkpoint-load, SSD-streaming
or inference benchmark.

## Fault and safety coverage

The native deterministic fixture covers:

- exact tensor/tile order, offsets, final-tile flags and byte totals;
- exclusion of routed experts and global non-layer tensors;
- missing layer, zero limit, tensor-count and tile-count rejection;
- successful Unicode-path range loading and exact byte contents;
- metadata-only payload refusal;
- modified source identity and wrong residency-size refusal;
- per-tile limit and aggregate host-memory budget refusal;
- cancellation before admission and after the read;
- short read failure;
- `FAILED` publication only after transfer release; and
- zero RAM/staging reservations after eviction or every injected failure.

A live host-pressure check also failed closed. While an unrelated local compilation reduced
available RAM first to 6.29 GB and then 2.10 GB, `deepseek-gguf-checkpoint` refused to start with
the 8 GiB headroom warning / safe-budget error. After WSL released its completed build memory and
11.50 GB was available, the same MSVC Release checkpoint test passed. No reserve was weakened and
no model payload allocation was attempted during pressure.

## Test matrix

| Configuration | Result |
|---|---:|
| MSVC Release, warnings as errors, real SmolLM2 + OLMoE + 0731 metadata | 13/13 passed |
| MSVC Debug, warnings as errors, real SmolLM2 + metadata adapters | 13/13 passed |
| GCC Release, warnings as errors, all available real-model opt-ins | 14/14 passed |
| Clang Release, warnings as errors | 11/11 passed |
| GCC Debug, ASan + UBSan + leak detection, halt on error | 11/11 passed |

The GCC Release OLMoE reference dominated its 208.78-second suite at 168.12 seconds. The native
unit suite containing the new layer-streaming fault tests took 0.12 seconds in that build. These
times are test-harness observations, not model throughput.

ThreadSanitizer is not listed as passing. Its existing WSL build remains unusable on this host
because the runtime exits before `main` with `unexpected memory mapping`; that environmental
failure is separate from the ASan/UBSan result.

## Known limitations

- The two large official GGUF objects (82.53 GB of weight payload in total) remain undownloaded,
  so no official trunk tile has been loaded.
- Global embedding/output tensors are catalogued but do not yet have a phase-specific streaming
  consumer.
- Q8_0, Q5_K, Q6_K, BF16 and F32 trunk kernels required by a complete 0731 layer are missing.
- The tile loader is synchronous and buffered. Direct I/O, pinned buffers, async double buffering,
  VRAM copies and read/compute overlap remain unimplemented for this path.
- No bytes/token, physical-device bytes, TTFT or token/s claim can be made before a real token path
  exists.

## Next bottleneck

After explicit approval for the 82.54 GB download and a fresh storage/headroom check, import and
fully hash the pinned checkpoint. Then consume one planned official trunk tile plus the already
defined complete expert bundle in a quantized one-layer/token probe. The next kernel work is the
missing Q8_0/Q5_K/Q6_K/BF16/F32 trunk operator set; only after parity may the loader become
asynchronous and overlap NVMe reads with CPU/CUDA compute.
