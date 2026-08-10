# M7 DeepSeek GGUF working-set validation

Validation date: 2026-08-10

Implementation commit: `f1dbf635cbc311b6c862b70efbb1dd6a3e970b99`

## Scope

This milestone turns the official DeepSeek V4 Flash 0731 `UD-IQ1_S` GGUF catalog into a bounded,
metadata-derived capacity plan. It does not load the full payload or execute a DeepSeek layer or
token.

## Validation host

The native probe reported Windows, an AMD64 Family 25 Model 97 CPU with 24 logical cores, AVX2,
AVX-512F and FMA, 33,407,475,712 bytes of physical RAM, and an NVIDIA GeForce RTX 4070 SUPER with
12,878,610,432 bytes of VRAM and driver 610.88. Storage capacity was 2,047,435,862,016 bytes. The
GCC, Clang and sanitizer matrices ran in WSL on the same physical PC.

The validated path is:

```text
bounded GGUF v3 header snapshots
  -> selectively retain one integral compression-schedule array
  -> cross-check architecture scalars against the catalog request
  -> derive KV, compressor state, RoPE and bounded kernel/activation scratch
  -> calculate resident-trunk and streamed-trunk floors
  -> require one complete expert bundle to fit one RAM or VRAM tier
  -> report capacity candidates against live Governor budgets
```

## What was implemented

- The generic GGUF reader still skips array payloads by default. A caller may retain only named
  integral arrays, with a separate aggregate element limit. Requested scalar, floating, string,
  nested, duplicated or oversized retention inputs fail closed.
- `deepseek-gguf-inspect` retains only
  `deepseek4.attention.compress_ratios`, accepts explicit bounded `--context` and `--batch-size`
  assumptions, and reads no tensor payload.
- The DeepSeek adapter cross-checks layer, hash-layer, expert, top-k, context, hidden, attention,
  indexer, FFN and Hyper-Connection metadata. Inconsistent metadata/options and incompatible
  compression contexts are rejected.
- All products and sums use checked 64-bit accounting. Architecture dimensions, retained arrays,
  batch size and context remain bounded before working-set arithmetic.
- Persistent main/indexer KV is budgeted as BF16. Official incremental compressor state is
  budgeted as FP32. Scratch includes explicit attention/indexer/router/expert/logit activations and
  a fixed 64 MiB bounded kernel tile.
- Two placement profiles are reported without changing persisted manifest semantics:
  - `resident-trunk` includes all always-active encoded weights;
  - `streamed-trunk` leaves them immutable on SSD and budgets only runtime state plus a bounded
    current trunk tile.
- The complete maximum expert bundle must fit wholly in the remaining RAM tier or wholly in the
  VRAM tier. Tests prove that two individually insufficient tiers are never summed to admit it.

## Real header result

The command used the three previously captured header prefixes. Split 0 is a complete metadata
header; splits 1 and 2 are bounded prefixes with their immutable full file sizes supplied through
`--declared-sizes`.

```powershell
vastloom deepseek-gguf-inspect `
  build/research-gguf-header/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf `
  build/research-gguf-header/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf `
  build/research-gguf-header/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf `
  --declared-sizes 5257664,49093726624,33440253504 `
  --revision fbbb5b93 --context 4096 --batch-size 1 --json
```

Catalog identity remained unchanged:

| Field | Exact result |
|---|---:|
| Declared GGUF bytes | 82,539,237,792 |
| Physical tensors | 1,328 |
| Logical tensors | 34,223 |
| Logical experts | 11,008 |
| Always-active encoded bytes | 6,986,088,796 |
| Routed-expert encoded bytes | 75,547,803,648 |
| Minimum complete expert bundle | 6,488,064 |
| Maximum complete expert bundle | 8,781,824 |
| Top-6 no-cache expert bytes/token | 1,770,651,648 |

The derived batch-1, 4,096-token working set was:

| Component/profile | Exact bytes |
|---|---:|
| Persistent BF16 KV | 33,816,576 |
| FP32 compression state | 12,206,080 |
| Shared RoPE table | 1,048,576 |
| Activation and bounded kernel scratch | 69,052,440 |
| Runtime state / streamed-trunk host floor | 116,123,672 |
| One complete expert tier | 8,781,824 |
| Streamed-trunk CPU floor | 124,905,496 |
| Streamed-trunk CPU double-buffer floor | 133,687,320 |
| Resident-trunk host floor | 7,102,212,468 |
| Resident-trunk CPU floor | 7,110,994,292 |
| Resident-trunk CPU double-buffer floor | 7,119,776,116 |

At the recorded live probe, the safe caps were 6,578,827,264 bytes of host RAM and
8,657,043,456 bytes of VRAM. The streamed-trunk capacity profile passed and one complete expert
fit independently in either tier. The resident-trunk profile failed because its 7,102,212,468-byte
host floor exceeded the live safe RAM cap. These caps are intentionally dynamic and will change
with current system/GPU use.

This is the expected safe behavior: nominal 32 GB RAM and 12 GB VRAM are not treated as freely
allocatable, and the smaller streamed profile does not imply that its execution path exists.

## Tests actually run

| Toolchain/configuration | Result |
|---|---|
| MSVC 19.44 Release, warnings as errors, external SmolLM2 + full OLMoE | 12/12 passed |
| Clean MSVC 19.44 Debug, warnings as errors, standard gates | 9/9 passed |
| GCC 13.3 Release, warnings as errors, SmolLM2 + full OLMoE + DeepSeek metadata | 13/13 passed |
| Clang 18.1 Release, warnings as errors, DeepSeek metadata | 10/10 passed |
| GCC ASan + UBSan, leak detection and halt-on-error, DeepSeek metadata | 10/10 passed; no finding |
| Python `compileall` | passed |
| Manifest tool unit tests | 3 passed, 1 Windows symlink skip |
| Offline Hugging Face importer | passed |

The opt-in MSVC Debug external-model configuration passed 11/11 tests when the full OLMoE runtime
probe was excluded. The isolated OLMoE runtime probe failed safely because Debug `moe-inspect`
did not finish checkpoint verification within its 300-second command timeout. The identical real
OLMoE path passed in MSVC and GCC Release; the Debug external-model result is not reported as a
pass.

ThreadSanitizer was attempted separately. The current WSL/kernel environment terminated every
instrumented Vastloom binary before `main` with `FATAL: ThreadSanitizer: unexpected memory
mapping`; six of nine tests therefore failed at process startup and the three tests that did not
launch an instrumented binary passed. This is an environmental TSan failure, not a clean race
result, and is not listed as passing validation.

Malformed/fault coverage added in this slice includes non-integral requested arrays, scalar
retention requests, duplicate retention keys, mismatched architecture declarations, contexts not
divisible by compression ratios, and the no-RAM-plus-VRAM expert-splitting invariant.

## Performance

No inference throughput is reported. Header parsing and capacity arithmetic are not model
generation benchmarks. The important future performance baselines remain physical NVMe bytes per
accepted token, TTFT, decode tokens/s, cache hit rate, GPU idle time and I/O stall time for an
executable layer/token path.

## Known limitations

- The two large GGUF payload shards were not downloaded, imported or fully SHA-256 verified.
- No DeepSeek trunk tile is read or executed by this path.
- No complete DeepSeek layer, routed token, generation loop or chat has run.
- `mtp_enabled` is explicitly false in the derived baseline.
- The 64 MiB kernel tile is a bounded native planning assumption; the production quantized kernel
  must validate or revise it before execution admission.
- CUDA allocation, pinned host memory and RAM-to-VRAM expert transfer are not connected.
- A streamed-trunk candidate can be physically safe but unacceptably slow. It must never be called
  interactive until measured end-to-end.
- Kimi K3 is excluded from reference-PC execution validation. It remains metadata/capacity-only
  until an exact profile proves that a complete expert bundle fits one real safe tier; no swap or
  cross-tier fiction is allowed.

## Next bottleneck

Bind immutable, fully verified `UD-IQ1_S` payload shards to the catalog and execute one complete
quantized layer/token while keeping all reads, trunk tiles, expert bundles and scratch under the
derived reservations. That produces the first meaningful trunk/expert bytes-per-token baseline.
Only then should resident placement, CUDA kernels and read/transfer/compute overlap be optimized.
