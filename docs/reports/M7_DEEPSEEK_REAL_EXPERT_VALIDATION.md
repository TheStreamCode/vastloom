# M7 DeepSeek V4 Flash 0731 real-expert validation

Validation date: 2026-08-10

Validated implementation commit: `5e17866cba268ff75254be3e26c0c081c3c4b21e`

## Scope

This milestone executes one complete quantized routed expert from the pinned
[`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
`UD-IQ1_S` profile:

```text
three exact, hash-pinned GGUF ranges
  -> one-worker / one-entry bounded load queue
  -> pre-reserved packed RAM residency
  -> IQ1_S gate and up projections
  -> FP32 SiLU(gate) * up
  -> IQ3_XXS down projection
  -> transactional FP32 output publication
  -> lease release and governed RAM eviction
```

All 4,096 final outputs match an independent build of the pinned ggml decoder bit for bit.
This is the first execution of a complete real DeepSeek expert in Vastloom. It is not a full
DeepSeek layer, routed token, generation path or throughput claim. The test uses three small
extracted range files; it does not treat metadata-only header prefixes as executable shards and
does not download the 82.54 GB checkpoint.

## Exact source identity

- repository: `unsloth/DeepSeek-V4-Flash-0731-GGUF`;
- revision: `fbbb5b93fb787c21338159b0af3318bb3f4d9768`;
- profile: `DeepSeek-V4-Flash-0731-UD-IQ1_S`;
- source object: `DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf`;
- declared source size: 49,093,726,624 bytes;
- source Xet identity: `d0a9e82d487241a17011acc4d91fcaabb6d8d219772d393121e350bf8d6cdfe0`.

Only these three HTTP `206` ranges were retained for the expert test:

| Projection | Source range, inclusive | Encoding | Bytes | Slice SHA-256 |
|---|---:|---|---:|---|
| gate | `1537107360-1538745759` | IQ1_S | 1,638,400 | `4710ef29f496c13acabd1371d1d2a17be3f321f85a0b12287145e7c0b409c4f9` |
| up | `1967521184-1969159583` | IQ1_S | 1,638,400 | `7be122f1839feb8fdd84f57b9d20d5b57566227f8ba7d8ffc50123b158bf762f` |
| down | `708142496-711353759` | IQ3_XXS | 3,211,264 | `65b9cdc96f9e753584dc6eac4723616a494a98113bf25836fcda5ec72c400204` |

The download client enforced the exact response length and a maximum-filesize guard equal to
the requested range. It therefore could not silently accept the complete 49.09 GB shard. The
three local slices total exactly 6,488,064 bytes and are ignored by Git.

## Native implementation

The new CPU reference path implements the exact little-endian GGML `IQ3_XXS` block contract:

- 256 values per 98-byte block;
- one FP16 block scale;
- 64 quantization-index bytes;
- eight little-endian scale/sign words;
- the exact 256-entry IQ3 grid from ggml commit
  `30bf8685ed4eb0a47f2b06229543327749904150` under its MIT license;
- even-parity expansion of each seven-bit sign index;
- fixed 256-float decode scratch and FP64 dot-product accumulation.

Compile-time FNV-1a fingerprints guard both embedded immutable grids:

| Grid | Fingerprint |
|---|---|
| IQ1_S | `6703ed863501ae2e` |
| IQ3_XXS | `dfa5dc83d6a155d5` |

The routed-expert function accepts caller-owned encoded spans, input, scratch and final output.
It validates dimensions, encoded byte counts, finite scales, finite input and non-overlapping
live buffers. Gate and up are multiplied first, the SiLU-gated activation is written into scratch,
and IQ3_XXS down output is computed into a separate scratch span. The caller-visible output is
copied only after every stage succeeds.

## Independent parity result

The oracle was compiled against ggml commit
`30bf8685ed4eb0a47f2b06229543327749904150`. It invokes ggml's own `dequantize_row_iq1_s`
and `dequantize_row_iq3_xxs`, performs the same FP64 dot-product accumulation and applies the
FP32 SwiGLU equation independently of the Vastloom decoder.

| Check | Result |
|---|---:|
| Standalone IQ3_XXS down rows compared | 4,096 / 4,096 exact |
| Standalone IQ3_XXS maximum absolute error | 0 |
| Standalone down output fingerprint | `15774ed1fd7eb1a0` |
| Complete expert outputs compared | 4,096 / 4,096 exact |
| Complete expert maximum absolute error | 0 |
| Complete expert output fingerprint | `90388543cbcbab52` |
| First / last complete-expert value | `-0.0709934607` / `0.105151489` |

The committed external-slice test rechecks all three file identities, reconstructs one logical
expert bundle, loads it asynchronously, resolves the three packed tensor identities and requires
the complete-expert fingerprint above on MSVC, GCC and Clang.

## Safety properties exercised

The real-expert test deliberately uses tighter limits than the host:

| Limit | Value |
|---|---:|
| RAM hard limit | 8 MiB |
| staging hard limit | 4 MiB |
| scratch hard limit | 128 KiB |
| workers | 1 |
| queued requests | 1 |
| ranges per expert | 3 |
| maximum range | 4 MiB |
| maximum expert payload | 8 MiB |
| reserved input/output/compute scratch | 65,536 bytes |

The test proves that:

- a RAM limit one byte below the 6,488,064-byte expert rejects submission before worker I/O or
  destination allocation and leaves residency in `NVME`;
- the successful request reserves the full RAM destination and largest read staging range before
  allocation;
- the live `LoadedExpert` lease prevents eviction while kernel spans reference its bytes;
- staging is released before `RAM_READY` publication;
- RAM becomes evictable only after the kernel-use lease ends;
- RAM, staging and scratch reservations all return to zero after eviction;
- malformed/truncated blocks, non-finite scales/input, element ceilings and overlapping scratch
  fail without publishing partial final output.

No model weight is modified. The loader opens each source read-only and the external test hashes
each complete extracted range before submission.

## Validation host

- Windows 11 Pro `10.0.26200`, build 26200;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- AVX2, AVX-512F and FMA detected;
- 33,407,475,712 bytes physical RAM, 11,914,313,728 bytes available at the recorded probe;
- NVIDIA GeForce RTX 4070 Super, 12,878,610,432 bytes VRAM;
- NVIDIA driver 610.88;
- source files on the Windows `C:` volume;
- CUDA was not used.

## Build and test matrix

All native configurations used project warnings as errors and included the optional real GGUF
header plus gate/up/down slice tests.

| Toolchain / configuration | Result |
|---|---:|
| MSVC 19.44 Release | 13/13 passed |
| MSVC 19.44 Debug | 13/13 passed |
| GCC 13.3 Release | 13/13 passed |
| Clang 18.1 Release | 13/13 passed |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 13/13 passed |
| GCC 13.3 Debug + ThreadSanitizer | 13/13 passed |

ASan ran with leak detection and halt/abort on finding; UBSan and TSan were configured to halt on
the first finding. TSan ran with ASLR disabled for the instrumented WSL processes. No sanitizer
finding was reported.

The standard Python gates also passed: `compileall`; the offline Hugging Face importer; and the
manifest-tool suite with one pass plus the expected Windows symlink-creation skip.

## Warm buffered measurement

After one warm-up, the MSVC Release executable ran nine measured iterations. The timing excludes
the pre-test SHA-256 checks but includes pool submission, worker queueing, three buffered reads,
packed-buffer publication and lease acquisition.

| Metric | Median | Minimum | Maximum |
|---|---:|---:|---:|
| governed load | 1.9116 ms | 1.7344 ms | 2.1094 ms |
| scalar expert compute | 42.7731 ms | 42.5938 ms | 44.6640 ms |

Every iteration reported exactly three process read operations and 6,488,064 process transfer
bytes through Windows `GetProcessIoCounters`.

These are deliberately labeled **warm buffered and process-visible** measurements. Windows did
not provide a storage-layer delta for this path, the OS cache was not bypassed, and no global
cache was flushed. The figures are not physical NVMe bandwidth, bytes/token, DeepSeek tokens/s or
evidence of read/compute overlap. The kernel is scalar and only one expert is executed.

## Known limitations

- Only layer 0, expert 0 is executed, using a deterministic synthetic hidden-state vector.
- The three verified ranges are extracted test artifacts, not direct reads from a fully imported
  and SHA-256-verified 49.09 GB shard.
- Router selection/weighting, shared experts, the always-active trunk, attention, KV state and
  MTP verification are not connected to this quantized expert path.
- There is no full layer, token-generation loop, chat path or useful DeepSeek tokens/s result.
- The scalar kernels have no SIMD, threading, fusion, CUDA or VRAM implementation.
- Reads are ordinary buffered I/O and run before compute; there is no SSD-to-compute overlap.
- The quality of the `UD-IQ1_S` quantization profile was not evaluated.

## Next bottleneck

The next correctness slice is an immutable full-shard resolver that binds the existing
header-derived `(shard, offset, bytes)` identities to verified payload files without copying or
rewriting them. It must then drive router-selected experts through this same governed loader and
assemble one complete quantized layer/token probe. Only after output parity should Vastloom add
SIMD/CUDA kernels, VRAM residency and overlapped direct I/O.
