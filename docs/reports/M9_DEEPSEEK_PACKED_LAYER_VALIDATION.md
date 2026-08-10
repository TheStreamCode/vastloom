# M9 DeepSeek packed-layer reference validation

Validation date: 2026-08-10
Code commit: `1705e932b8b1b818cc60f2e31fd6a0be1340b8f2`

## Scope

This milestone connects the bounded GGML storage kernels to the native DeepSeek graph for the
first time:

```text
pinned official reduced graph topology
  -> deterministic Q8_0 encoding with zero-padded physical rows
  -> packed matrix descriptors (logical shape != storage stride)
  -> transactional scalar matvec dispatch
  -> hash/learned router logits
  -> attention + routed/shared expert SwiGLU + Hyper-Connections
  -> full prefill and incremental state
  -> main and MTP logits/routes/attention indices
```

The committed test replaces every matrix consumed by main layer 0 with packed Q8_0 bytes: 23
matrices, 154 physical rows and 5,236 encoded bytes. The layer-0 hash router table is separately
encoded as 128 bytes of little-endian I32. Logical widths of 4–16 values are padded to a physical
Q8_0 stride of 32 only for this reduced fixture; padding is always zero and is not visible to the
architecture equations.

This is a synthetic, cache-resident correctness fixture. It does not read an official 0731 trunk
tile and does not make DeepSeek generation or SSD-throughput claims.

## Independent reference chain

The deterministic Q8_0 encoder implements the reference ggml row algorithm and is limited by the
same bounded element contract as the decoders. A 3x17 logical matrix padded to 3x32 produces the
encoded FNV-1a fingerprint `083a653165a92c21`, exactly matching a standalone oracle compiled from
ggml commit `30bf8685ed4eb0a47f2b06229543327749904150`.

For graph parity, the float oracle is built from the independently verified dequantization of the
exact encoded matrices. The packed graph never materializes those weight matrices: it decodes at
most one fixed GGML block inside each matvec. Prefill and one incremental decode compare main/MTP
logits, greedy IDs, expert IDs, routing weights and attention indices with an absolute tolerance
of `3e-5`.

The earlier official FP32 fixture remains unchanged and still matches the pinned official graph.
Quantized parity therefore does not replace or weaken the architecture oracle.

## Safety and failure behavior

- Matrix descriptors require exact output rows, exact logical input width, a non-zero physical
  stride and a supported GGML matvec type.
- Matrix elements remain under the 64-million-element reference ceiling.
- Inputs, scales and outputs must be finite.
- The Q8_0 encoder rejects malformed strides, shape mismatch, non-finite source data, FP16 scale
  overflow and configured-limit exhaustion.
- Matvecs write disjoint scratch first and publish a row only after complete validation.
- I32 routing tables reject byte truncation, shape mismatch and negative expert IDs.
- A failed incremental decode leaves the caller's previously published cache state unchanged.

The packed bytes in this fixture are owned by the in-memory reference graph. The next slice must
bind their lifetime to the existing Governor-backed trunk/expert loader leases before this design
can consume large file-backed payloads without unaccounted duplication.

## Validation host

- Windows 11 development host, build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- NVIDIA GeForce RTX 4070 Super present but unused;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2` with 15,912,636 kB exposed RAM;
- no CUDA code was compiled or executed.

## Exact results

| Toolchain/configuration | Result |
|---|---:|
| GCC Release, all configured real-model opt-ins, serial | 14/14 passed in 235.34 s |
| GCC Debug | 9/9 passed in 5.59 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 11/11 passed in 69.42 s |
| Clang Release, focused native/operator/graph gates | 3/3 passed in 0.29 s |
| MSVC Release | 10/10 passed in 7.91 s |
| MSVC Debug, including three real DeepSeek slice tests | 13/13 passed in 9.13 s |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python offline importer | passed |
| Python `compileall` | passed |

The full GCC Release suite included SmolLM2 and OLMoE paths already present in the local model
store. OLMoE took 188.48 seconds and ran alone. No resource limits were relaxed.

## Performance statement

No token rate, SSD bandwidth or bytes/token figure is reported for this milestone. The packed
payload is only 5,364 bytes including I32 and remains in CPU cache; timing it as a model benchmark
would be misleading. Relevant performance work begins after the same consumer operates on
Governor-owned file-backed tiles and exposes exact logical/process/device I/O counters.

## Known limitations

- Only Q8_0 is composed through the complete reduced layer in this fixture. Q5_K, Q6_K, BF16,
  F32, IQ1_S and IQ3_XXS retain their separate pinned kernel/expert parity evidence.
- Packed buffers are not yet sourced from `LoadedTensorTile` or retained `LoadedExpert` leases.
- There is no official 0731 trunk payload, full-checkpoint generation, MTP verification loop,
  CUDA backend, VRAM residency or physical NVMe measurement.
- The reduced graph has two tiny main layers and one tiny MTP layer; it establishes control-flow
  correctness, not production numerical quality or speed.

## Next bottleneck

Introduce an ownership-preserving packed tensor view whose backing object retains the Governor
reservation and residency lease. Use it to load a file-backed trunk fixture and selected expert
bundle, execute one quantized layer without copying into ungoverned vectors, inject budget/read/
cancellation failures, and only then request approval for official 82.54 GB payload access.
