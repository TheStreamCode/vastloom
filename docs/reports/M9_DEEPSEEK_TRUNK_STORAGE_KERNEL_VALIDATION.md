# M9 DeepSeek trunk storage-kernel validation

Validation date: 2026-08-10

Validated runtime commit: `370f10bc111d73bb070262221286411b47c85953`

## Scope

This slice adds the bounded scalar storage kernels required by every non-routed tensor in the
pinned DeepSeek V4 Flash 0731 `UD-IQ1_S` layer catalog:

```text
verified tensor identity and GGML type
  -> exact encoded-size and dimension checks
  -> finite scale/input validation
  -> fixed-size scalar block decode
  -> FP64-accumulated reference matvec into caller scratch
  -> publish caller output only after complete success
```

The three `ffn_gate_tid2eid` router tables are I32, not floating-point matrices. They use a
separate bounded little-endian integer decoder and are deliberately rejected by the matvec
dispatcher.

This report proves storage-format coverage and deterministic scalar reference math. It does not
claim that the official 82.54 GB payload was downloaded, that a complete DeepSeek layer or token
ran, or that CPU/CUDA/NVMe performance is production-ready. The CLI continues to report
`execution_supported=false`.

## Implemented

- Exact little-endian GGML block decoders and scalar FP32 matvecs for:
  - Q8_0: 32 values in 34 bytes;
  - Q5_K: 256 values in 176 bytes;
  - Q6_K: 256 values in 210 bytes.
- Alignment-independent BF16 and F32 little-endian decode/matvec paths.
- A bounded I32 little-endian decoder for router lookup tables.
- A storage-type capability query distinct from the matvec capability query. I32 is a supported
  storage type but not a supported floating matvec.
- A transactional type dispatcher for GGML types F32, Q8_0, Q5_K, Q6_K, IQ3_XXS, IQ1_S and BF16.
  It writes to caller-owned scratch first and copies to the public output only after success.
- The existing 64,000,000-element ceiling remains the default. Encoded bytes, rows, columns,
  block alignment, finite inputs and non-overlapping live buffers are checked before publication.
- The DeepSeek GGUF JSON/text report now distinguishes covered and unsupported layer-trunk
  storage types and bytes. This field is coverage evidence, not an execution flag.

No decoder allocates a decoded matrix. Quantized matvecs use at most one fixed 256-float block;
all large encoded, input, scratch and output spans remain caller-owned and therefore can be
reserved by the Safety Governor before use.

## Independent ggml oracle

The Q8_0, Q5_K and Q6_K equations were compared against a standalone executable linked to pinned
ggml commit `30bf8685ed4eb0a47f2b06229543327749904150`. The oracle and Vastloom consumed the same
deterministically generated raw block bytes; the committed tests contain only the inputs and
expected results, not a runtime dependency on ggml.

| Format | Decoded FP32 FNV-1a fingerprint | Oracle matvec rows |
|---|---|---:|
| Q8_0 | `25edd1a40129ee99` | 3 × 64 |
| Q5_K | `3ec4a59589931118` | 3 × 512 |
| Q6_K | `cd39c3f1219afbc9` | 3 × 512 |

All fingerprints, selected decoded values and the three FP64-accumulated/FP32-published row
outputs match the pinned oracle exactly. BF16, F32 and I32 use explicit bit-pattern vectors,
including signed I32 extrema.

## Real 0731 metadata coverage

The previously verified 5,343,584-byte header import was inspected at Unsloth revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`. The two large payload objects were not downloaded or
read.

| GGML type | Role in layer trunk | Tensors | Encoded bytes | Bounded native handling |
|---|---|---:|---:|---|
| F32 (0) | norms, sinks, Hyper-Connection state | 658 | 164,788,552 | decode + matvec |
| Q8_0 (8) | attention projections | 318 | 5,214,044,160 | decode + matvec |
| Q5_K (13) | attention/shared-expert projections | 126 | 605,552,640 | decode + matvec |
| Q6_K (14) | shared-expert down projections | 45 | 306,216,960 | decode + matvec |
| I32 (26) | token-id to expert-id router tables | 3 | 9,308,160 | integer decode only |
| BF16 (30) | router weights | 43 | 90,177,536 | decode + matvec |
| **Total** | | **1,193** | **6,390,088,008** | **1,193 / 1,193 covered** |

The report therefore records zero unsupported layer-trunk tensors and zero unsupported bytes.
Routed IQ1_S/IQ3_XXS expert tensors are outside this count and retain their separate complete-
expert parity evidence.

## Fault and safety coverage

The native tests cover:

- truncated and incorrectly sized block payloads;
- columns not divisible by the storage block width;
- element-limit refusal before decode/compute;
- non-finite FP16 Q8_0/Q5_K/Q6_K scales and Q5_K minima;
- non-finite BF16/F32 weights and FP32 inputs;
- exact little-endian I32 signed decoding and size/limit refusal;
- unsupported type refusal, including correct rejection of I32 by the matvec dispatcher;
- scratch/input/output overlap rejection; and
- unchanged caller-visible output when a later encoded block is malformed.

The malformed-scale tests place the invalid value in a later block, proving that validation runs
before public dequantized output is modified. The transactional dispatcher separately proves that
a failed matvec does not publish partial rows.

## Test matrix

| Configuration | Result |
|---|---:|
| GCC 13.3 Release, warnings as errors, all configured real-model opt-ins | 14/14 passed |
| Clang 18.1 Release, warnings as errors | 11/11 passed |
| GCC Debug | 9/9 passed |
| GCC Debug, ASan + UBSan + leak detection, halt on error | 11/11 passed |
| MSVC Release, warnings as errors | build passed; 12/13 tests passed |
| MSVC Debug, warnings as errors | build passed; 11/11 non-large-model tests passed |
| Python compileall + offline importer | passed |
| Python manifest tests | 3 passed, 1 Windows symlink-capability skip |

The one MSVC Release failure and the two separately attempted MSVC Debug real-model tests were
environmental safety refusals, not sanitizer findings or inference mismatches. Concurrent local
work reduced available Windows RAM; OLMoE reported safe caps of 2,715,766,784 and 2,490,298,368
bytes against its 3,201,568,768-byte minimum, while later Debug invocations reported less than the
required 8 GiB headroom. No limit was weakened. The same source passed the complete 14-test GCC
matrix, including SmolLM2, OLMoE and the pinned DeepSeek metadata/checkpoint gates.

ThreadSanitizer was not rerun. Its existing WSL configuration remains unavailable on this host
because the runtime exits before `main` with `unexpected memory mapping`.

## Synthetic hot-cache timing

The validation host was the reference Ryzen 9 7900X (12 cores / 24 threads) under Linux
6.18.33.2 WSL2. A single-threaded GCC Release probe used all-zero valid encoded blocks at the
largest representative layer-0 shapes, excluded one warm-up and recorded three samples:

| Format and real catalog shape | Encoded bytes | Median | Mean |
|---|---:|---:|---:|
| Q8_0, 8,192 × 4,096 | 35,651,584 | 76.466 ms | 76.381 ms |
| Q5_K, 2,048 × 4,096 | 5,767,168 | 19.961 ms | 19.938 ms |
| Q6_K, 4,096 × 2,048 | 6,881,280 | 25.019 ms | 26.046 ms |

These are synthetic hot-cache scalar-kernel timings. They do not include file I/O, routing,
attention, normalization, KV state, layer scheduling or generation and must not be converted into
tokens/second. All-zero scales exercise the complete decode/multiply loops but do not measure
real-weight numerical distribution effects.

## Known limitations

- The official full GGUF source is absent, so no official trunk tensor byte has entered these
  kernels.
- Storage coverage does not connect tensor names to the complete DeepSeek layer equation. Norm,
  attention, Hyper-Connection, shared-expert and router consumers still need an architecture-
  specific packed layer path.
- The kernels are scalar correctness references: no SIMD, fused operator, multithreading or CUDA
  implementation exists for this trunk path.
- The layer tile loader remains synchronous/buffered and is not connected to compute. No direct
  I/O, pinned host memory, double buffering, NVMe/compute overlap or physical bytes/token was
  measured.
- Global embedding/output tensors are not part of the 1,193 layer-trunk coverage count.
- No DeepSeek logits, routed token, generation, TTFT or tokens/second result was produced.

## Next bottleneck

Build an architecture-specific quantized one-layer fixture that binds the existing logical tensor
roles to these storage kernels, the reference operators, incremental state and one governed
complete expert. It must match the pinned official graph before any full-model execution claim.
After explicit approval for the 82.54 GB import and a fresh disk/RAM/thermal check, the same
consumer can be tested against fully hashed official source ranges and then connected to bounded
double-buffered I/O.
