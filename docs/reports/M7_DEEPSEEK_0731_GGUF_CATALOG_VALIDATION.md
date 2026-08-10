# M7 DeepSeek V4 Flash 0731 GGUF catalog validation

Validation date: 2026-08-10

Implementation commit: `8b71ee9`

## Scope

This milestone adds a non-executing native path for the pinned
[`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
revision:

```text
bounded GGUF v3 header or prefix
  -> validated split catalog
  -> packed `*_exps.weight` tensors
  -> per-layer/per-expert immutable source ranges
  -> universal logical tensor and expert catalog
```

It proves header parsing, exact encoded-size accounting and logical expert addressing for the
reviewed `UD-IQ1_S` profile. It does not load the large weight payload, dequantize a tensor, run a
DeepSeek layer, generate a token or allocate CUDA/VRAM memory.

## Pinned artifact and bounded acquisition

- repository revision: `fbbb5b93fb787c21338159b0af3318bb3f4d9768`;
- profile: `DeepSeek-V4-Flash-0731-UD-IQ1_S`;
- declared split sizes: 5,257,664, 49,093,726,624 and 33,440,253,504 bytes;
- declared total: 82,539,237,792 bytes;
- complete split-zero SHA-256:
  `2fa152e25a14500e42e6d98f20d200eec1e945f34dc1cfb981b651b9c86b97be`.

No multi-gigabyte model was downloaded. The first 5,257,664-byte shard contains metadata only.
For the two payload shards, the audit retained exactly the first 16 MiB returned through bounded
HTTP range requests; the native reader stopped at the aligned header boundary after 52,320 and
33,600 bytes respectively. Those prefixes are ignored local research artifacts beneath `build/`,
not committed weights or verified executable sources.

## Implementation

The new reader follows the published
[`ggml` GGUF v3 format contract](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and:

- accepts little-endian GGUF v3 and validates magic, counts, metadata types, tensor dimensions,
  unique names, alignment, relative/absolute offsets and zero header padding;
- calculates exact encoded sizes for the GGML storage types observed in the selected profile,
  including `IQ1_S` and `MXFP4`, and checks every result against the next tensor offset;
- supports full local files and bounded header-prefix snapshots with an explicit declared full
  shard size;
- caps one header at 256 MiB, strings at 16 MiB, metadata and tensors at one million each,
  aggregate array elements at 16 million, dimensions at eight, nesting at eight and alignment at
  one MiB;
- rejects symlinks, truncation, duplicate keys/tensors, malformed UTF-8 scalar metadata,
  overflow, invalid block shapes, overlapping offsets and inconsistent split numbering;
- never permits a header snapshot to become an executable tensor source unless separate immutable
  full-shard paths are supplied; the CLI catalog command is always `metadata_only`.

The DeepSeek adapter recognizes the three packed routed matrices in every layer:

```text
blk.N.ffn_down_exps.weight
blk.N.ffn_gate_exps.weight
blk.N.ffn_up_exps.weight
```

The final GGUF dimension is the 256-expert axis. Each packed tensor is therefore divided into 256
non-overlapping logical source ranges without copying or rewriting the artifact. Every resulting
expert bundle contains exactly one down, gate and up slice. All other tensors remain always-active
logical objects classified by role.

The CLI now provides:

```text
vastloom inspect FILE.gguf [--declared-size SIZE] [--tensor-limit N] [--json]
vastloom deepseek-gguf-inspect SHARD... [--declared-sizes CSV] [--revision ID] [--json]
```

Windows CLI paths are converted from the native wide command line to UTF-8 and then back into a
Unicode-native `std::filesystem::path`; a real accented temporary path is covered by the CLI test.

## Real-header result

The three bounded headers produced this native metadata-only catalog:

| Field | Result |
|---|---:|
| Split shards | 3 |
| Physical tensors | 1,328 |
| Packed routed-expert tensors | 129 |
| Logical tensors after slicing | 34,223 |
| Layers | 43 |
| Experts per layer | 256 |
| Logical expert bundles | 11,008 |
| Known encoded tensor bytes | 82,533,892,444 |
| Always-active encoded bytes | 6,986,088,796 |
| Routed-expert encoded bytes | 75,547,803,648 |
| Smallest expert bundle | 6,488,064 bytes |
| Largest expert bundle | 8,781,824 bytes |
| Top-k | 6 |
| Theoretical no-cache expert bytes/token | 1,770,651,648 bytes |

The last value is a deterministic sum of the encoded expert slices selected by the architecture's
top-6 contract across 43 layers. It is not measured SSD traffic, does not include the always-active
trunk or KV/state traffic and is not a tokens-per-second claim.

Five separate Windows Release invocations built the complete metadata catalog in 305.9–386.5 ms,
with a median process wall time of 336.3 ms. This is a warm local header/catalog micro-measurement;
it reads no weight payload and provides no inference-performance evidence.

## Validation matrix

All builds used project warnings as errors.

| Toolchain/configuration | Result |
|---|---:|
| MSVC 19.44.35228 Release, real split-zero header enabled | 10/10 passed in 4.30 s |
| MSVC 19.44.35228 Debug, clean external-model configuration | 9/9 passed in 5.58 s |
| GCC 13.3 Release on WSL2 | 9/9 passed in 3.70 s |
| Clang 18.1 Release on WSL2 | 9/9 passed in 3.56 s |
| GCC 13.3 Debug + ASan/UBSan, leak detection and halt-on-error | 9/9 passed in 12.45 s |
| GCC 13.3 Debug + ThreadSanitizer, native focused gate | 1/1 passed in 1.96 s |

The deterministic GGUF coverage includes a 43-layer/256-expert packed fixture, exact CLI JSON,
bounded tensor output, a Unicode Windows filename, header snapshots, continuation splits,
quantized byte spans, malformed magic/count/alignment/offset/truncation, array limits, duplicate
and missing split/projection conditions and the non-executable snapshot barrier.

Python bytecode compilation passed. The offline importer passed. Manifest tooling passed one test
and skipped the Windows symlink case because symlink creation was unavailable.

A separate optional MSVC configuration containing the resident SmolLM2 and OLMoE external tests
was attempted while Windows reported only 13,498,531,840 available bytes. Eight of eleven tests
passed; three real-model tests correctly stopped because the Safety Governor calculated no safe
model RAM after preserving host headroom. The clean Release/Debug and Linux matrices above isolate
the GGUF change and pass in full; the resource refusal is reported separately, not converted into
a passing test.

## Known limitations

- The two large shard payloads were not downloaded or SHA-256 verified.
- The revision supplied to `deepseek-gguf-inspect` is a label; the current command does not prove
  remote provenance or publish an immutable full-checkpoint import manifest.
- No IQ/other GGML dequantization or matvec kernel is connected to the catalog.
- No full 0731 tokenizer/chat, DSpark/MTP verification loop or generation path is validated.
- No expert bundle is consumed by CPU compute, copied into pinned RAM/VRAM or executed by CUDA.
- The 1.77 GB/token number is header-derived logical expert traffic, not physical device I/O.
- Other GGUF quantization profiles and architectures require their own encoded-layout and parity
  gates; there is no silent fallback or automatic quality downgrade.

## Next bottleneck

The next correctness gate is an IQ1_S-focused native CPU dequantization/matvec oracle bound to one
real packed expert slice, compared against the authoritative GGML/llama.cpp equations on fixed
vectors. After that passes, the bounded asynchronous loader can feed expert N+1 while CPU compute
consumes expert N, with logical and physical NVMe bytes/token measured separately. CUDA and the
full 0731 generation loop remain later gates.
