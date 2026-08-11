# M9 DeepSeek layer-0 trunk binding validation

Validation date: 2026-08-11

Implementation commit: `253696953db30439b038717dcba3f621d5b49636`

## Scope

This milestone closes the remaining synthetic tensor-consumer gap in the 21-range layer-0
trunk boundary:

```text
complete verified layer-0 tensor set
  -> retain one immutable Governor-owned packed buffer
  -> bind F32/BF16 vectors plus supported matrices and I32 routing data
  -> map 21 canonical 0731 GGUF blk.0.* names to exact reference-graph consumers
  -> reject incompatible names, types, shapes and storage
  -> execute reduced prefill and incremental decode without copying encoded payload into graph
```

The executable artifact is synthetic and contains 2,844 trunk bytes. The separately verified
real metadata identifies the official 138,867,160-byte layer-0 trunk, but no byte from that trunk
or the complete 82.54 GB checkpoint was downloaded or executed.

## Implemented

The reference graph now has a retained packed-float descriptor for F32 and BF16 vectors/scalars.
It stores either owned test bytes or one exact subrange of a shared immutable resident buffer.
The consumer:

- validates dtype, GGML type, catalog shape, exact encoded byte count and resident bounds;
- decodes explicit little-endian F32 or BF16 only when the reference operator needs the values;
- rejects non-finite decoded values before math or state publication;
- caps this transient reference decode at 1,048,576 elements, or 4 MiB of FP32 scratch;
- leaves matrix payloads encoded and uses the existing block-wise transactional matvec kernels.

An explicit `deepseek_v4_bind_0731_gguf_layer0_trunk` adapter now maps exactly these 21 official
identities into the reference graph:

- 11 matrix consumers covering attention, router, shared experts and hyper-connections;
- nine F32/BF16 normalization, sink, base and scale vectors;
- one typed I32 `tid2eid` hash-routing table.

The adapter requires the closed tensor set to contain exactly 21 entries, validates each expected
row count/vector length against the supplied architecture configuration and returns a detached
binding set. It does not heuristically rewrite arbitrary names and it does not publish partial
graph state if a binding fails.

## Correctness and fault coverage

The reduced layer fixture now republishes all non-routed objects under the canonical official
`blk.0.*` GGUF names. Its aggregate artifact contains:

- 11 Q8_0 matrices;
- nine F32 vectors;
- one I32 routing table;
- 2,844 total trunk bytes in 21 separately hashed ranges.

Four separately governed routed-expert bundles add eight IQ1_S gate/up matrices and four
IQ3_XXS down matrices. After the tensor-set loader wrapper is destroyed, the graph alone retains
one aggregate trunk lease. The complete reduced prefill and one incremental decode step match the
independently dequantized oracle for logits, greedy tokens, routes, routing weights and attention
indices.

Tests additionally cover:

- a configuration whose query LoRA rank disagrees with the canonical tensor rows;
- attempts to bind the I32 table as float and an F32 vector as I32 or matrix;
- wrong packed-float element count and GGML type;
- truncated and ambiguous resident storage;
- an encoded non-finite F32 value, with incremental state unchanged after rejection;
- F32 and BF16 resident vector binding with exact byte counts;
- aggregate lease cleanup, eviction and zero RAM/staging reservations after graph release.

The separately present real 0731 metadata was exercised by the 15-test GCC suites and still
reports the exact 21 canonical layer-0 ranges and 138,867,160 logical trunk bytes. That is metadata
and planning evidence only; the complete official tensor set was not materialized.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM and NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code and no official DeepSeek payload were executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors, existing real-metadata opt-ins | 15/15 passed in 14.91 s |
| GCC Debug, warnings as errors, existing real-metadata opt-ins | 15/15 passed in 36.61 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 75.05 s |
| Clang 18 Release, warnings as errors | 11/11 passed in 5.07 s |
| GCC Debug + libstdc++ debug iterators/assertions | 11/11 passed in 8.69 s |
| MSVC Release x64, warnings as errors | 11/11 passed in 5.63 s |
| MSVC Debug x64, warnings as errors | 11/11 passed in 8.54 s |
| Python manifest suite | 4/4 passed |
| Python importer suite and `compileall` | passed |

ASan used leak detection and halt-on-error; UBSan halted on the first finding. A focused TSan
attempt passed `vastloom_tests` in 2.65 seconds, then `vastloom_range_artifact` aborted before
`main` with `FATAL: ThreadSanitizer: unexpected memory mapping`. TSan is therefore an
environmental non-result and is not counted as a passing gate.

## Performance statement

No tokens/second, physical-device bandwidth, bytes/token or speedup is reported. The executable
fixture totals only 8,380 bytes across trunk plus expert source data and runs from test storage.
It validates identities, ownership, bounded decoding and numerical composition, not official
checkpoint or NVMe performance.

## Known limitations

- The complete official layer-0 payload was not imported, loaded or executed.
- The synthetic trunk uses Q8_0 for all 11 matrices. Separate kernels cover the official
  Q8_0/Q5_K/Q6_K/BF16/F32 types, but their complete real mixed-format composition has not run.
- The canonical adapter covers the 21-object layer-0 layout only. Later compressor/indexer
  layers, global embedding/output tensors, MTP generation and full-model orchestration remain
  unbound.
- The adapter receives a caller-supplied reference configuration. A fail-closed constructor from
  the verified official GGUF metadata is still required before an official payload test.
- Packed vectors use bounded transient FP32 decode scratch in the scalar reference path; fused
  production kernels do not exist.
- Authoritative official routing, selected-expert execution for one real token, physical NVMe
  overlap, CUDA and generation remain unvalidated.

## Next bottleneck

Derive the exact reference configuration and per-consumer dimensions directly from the verified
0731 GGUF metadata, then validate all 21 real catalog descriptors without payload access. Only
after that passes and the user explicitly authorizes the bounded transfer should Vastloom import
the 138,867,160-byte layer-0 trunk, execute its authoritative `tid2eid` route for a pinned input
and acquire only the six selected complete experts.
