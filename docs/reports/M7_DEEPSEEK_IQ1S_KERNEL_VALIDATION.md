# M7 DeepSeek V4 Flash 0731 IQ1_S kernel validation

Validation date: 2026-08-10

Implementation commit: `9dd6870`

## Scope

This milestone adds the first native quantized compute path for the pinned
[`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
`UD-IQ1_S` profile:

```text
bounded IQ1_S bytes
  -> validated FP16 block scales
  -> exact GGML IQ1_S codebook decode
  -> caller-owned FP32 block scratch
  -> scalar CPU matrix-vector result
```

It proves dequantization and matrix-vector parity for one real layer-0/expert-0 gate matrix. It
does not execute the expert's down projection, a complete MoE expert, a DeepSeek layer, a token
or a generation loop.

## Pinned source and real slice

- model revision: `fbbb5b93fb787c21338159b0af3318bb3f4d9768`;
- profile: `DeepSeek-V4-Flash-0731-UD-IQ1_S`;
- source shard: `00002-of-00003`, declared size 49,093,726,624 bytes;
- tensor: `blk.0.ffn_gate_exps.weight`, expert 0;
- absolute HTTP range: `1,537,107,360–1,538,745,759`;
- exact slice size: 1,638,400 bytes;
- slice SHA-256: `4710ef29f496c13acabd1371d1d2a17be3f321f85a0b12287145e7c0b409c4f9`;
- matrix shape: 2,048 rows × 4,096 columns;
- quantization: GGML type 19, `IQ1_S`, 256 values per 50-byte block.

Only this 1.64 MB range was acquired through HTTP `206`; the 82.54 GB profile was not
downloaded. The slice and upstream research checkout remain ignored beneath `build/` and are not
part of the repository.

## Reference implementation

The codebook and equations were checked against exact
[`ggml` commit `30bf8685ed4eb0a47f2b06229543327749904150`](https://github.com/ggml-org/ggml/blob/30bf8685ed4eb0a47f2b06229543327749904150/src/ggml-quants.c).
Vastloom carries only the required immutable IQ1_S codebook and scalar reference implementation;
it has no runtime dependency on ggml. The bundled codebook fingerprint is fixed at
`6703ed863501ae2e`, and the ggml MIT license is installed with the project.

The implementation:

- accepts complete little-endian 50-byte blocks only;
- validates dimensions, encoded sizes, overflow, a 64-million-element default ceiling, finite
  FP16 block scales and finite FP32 inputs before publishing output;
- prevalidates every block scale before dequantization so a late invalid block cannot leave a
  partially updated destination;
- uses caller-owned output and one fixed 256-float row scratch buffer;
- accumulates the reference dot product in FP64 and converts the final row result to FP32;
- provides no implicit fallback or quantization substitution.

The deterministic synthetic vector matches ggml's complete decoded-output FNV-1a fingerprint
`9b66de452b7f3c8d`. The real gate matrix produces FP32 output fingerprint
`069f079ee59c3b13`, first value `0.378450394` and last value `-0.220082492`.

An independent executable linked to the pinned upstream ggml source compared all 2,048 real
matrix rows. Result: zero mismatches and maximum absolute error zero.

## Measurements

Host: Windows 11 Pro build 26200, Ryzen 9 7900X (12 cores/24 threads), 32 GiB RAM, with the
reference comparison running in WSL2 Linux 6.18.33.2. The source slice was warm in the operating
system cache.

Seven sequential 2,048 × 4,096 matvec comparisons produced:

| Path | Range | Median |
|---|---:|---:|
| Vastloom scalar reference | 21.568–22.989 ms | 21.819 ms |
| Pinned ggml dequantize + identical FP64 dot | 8.583–9.206 ms | 8.678 ms |

This is an intentionally unoptimized, warm-cache CPU kernel microbenchmark. It excludes SSD read,
router, activation, down/up projections, layer trunk, attention, KV state and scheduling. It is
not tokens/s, bytes/token or proof that DeepSeek runs.

The real-header parser also exposed a portability bottleneck during validation: repeated
`seekg` calls for discarded tokenizer strings took 37.42 s on Windows-mounted WSL storage.
Replacing them with bounded sequential discard reads reduced the same process wall time to
0.11 s while strengthening short-input detection. This is header parsing performance, not model
payload streaming.

## Validation matrix

All native builds used project warnings as errors. The real GGUF header and IQ1_S slice were
enabled as opt-in external tests.

| Toolchain/configuration | Result |
|---|---:|
| MSVC 19.44.35228 Release | 11/11 passed in 3.65 s |
| MSVC 19.44.35228 Debug | 11/11 passed in 6.72 s |
| GCC 13.3 Release on WSL2 | 11/11 passed in 3.82 s |
| Clang 18.1 Release on WSL2 | 11/11 passed in 3.77 s |
| GCC 13.3 Debug + ASan/UBSan | 11/11 passed in 14.60 s |
| GCC 13.3 Debug + ThreadSanitizer, focused native tests | 2/2 passed in 2.52 s |

Python bytecode compilation passed. Manifest tests passed one test and skipped the Windows
symlink case because symlink creation was unavailable. The offline importer test passed with its
disk-capacity probe mocked to a deterministic 64 GiB; production still enforces the unchanged
20 GiB minimum reserve and fail-closed behavior.

## Known limitations

- The real slice is a single gate projection; it is not read through the production expert
  loader during compute.
- The matching up projection is also IQ1_S, but the expert down projection is IQ3_XXS and has no
  Vastloom reference kernel yet.
- No complete expert SwiGLU, top-6 layer execution or full-checkpoint probe exists for 0731.
- No SIMD, multithreaded, fused, CUDA or VRAM IQ1_S kernel exists.
- The two large GGUF shards have not been fully imported or SHA-256 verified locally.
- No physical NVMe bytes/token or I/O/compute overlap was measured in this milestone.

## Next bottleneck

Implement an exact IQ3_XXS CPU oracle, then load one complete gate/up/down expert bundle through
the governed asynchronous range loader and execute routed SwiGLU with parity. That is the first
point where read/compute overlap and logical versus physical bytes can be measured honestly.
