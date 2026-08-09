# M6 DeepSeek V4 Flash metadata inspection validation

Validation date: 2026-08-09  
Implementation commit: `de44029`
Tokenizer follow-up: `c815fdb`

## Scope

This report validates the metadata half of M6 against the official
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
repository at immutable commit
`60d8d70770c6776ff598c94bb586a859a38244f1`:

```text
pinned official artifact identities
  -> bounded metadata and HTTP Range header import
  -> full SHA-256 verification
  -> native config/index/Safetensors inspection
  -> exact tensor, shard, dtype and expert-layout report
```

This is not DeepSeek inference. The command explicitly returns
`execution_supported=false`; no model payload, activation, KV cache or GPU allocation is
performed. The architecture-faithful fixture, reference logits/routes and native
DeepSeek graph remain open M6/M7 work. Native tokenizer/basic-chat parity landed later
in `c815fdb` and is recorded in the companion
[tokenizer validation report](M6_DEEPSEEK_TOKENIZER_VALIDATION.md).

The former “DeepSeek V4 Flash 0731” name in the planning profile was provisional. The
official repository exposed no matching tag at validation time, so the active profile now
uses the exact repository name and commit instead of preserving an unsupported identity.

## Checkpoint identity

| Field | Verified value |
|---|---:|
| Architecture | `DeepseekV4ForCausalLM` |
| Main layers | 43 |
| Hidden width | 4,096 |
| Attention heads / KV heads | 64 / 1 |
| Maximum declared context | 1,048,576 tokens |
| Routed / shared experts | 256 / 1 |
| Selected experts per token | 6 |
| Hash-routing layers | 3 |
| MTP layers | 1 |
| Safetensors shards | 46 |
| Tensors | 69,187 |
| Weight-shard file bytes | 159,617,149,040 |
| Tensor payload bytes | 159,609,485,896 |

The native inspector validates every tensor name against its exact shard, parses every
shape, dtype and data range, rejects overlaps/overflow, and compares the aggregate catalog
with the pinned layout.

## Bounded header-only protocol

The importer downloads five normal metadata files plus the exact prefix of each of the 46
Safetensors shards. For every shard it:

1. requests only the pinned `8-byte length + JSON header` range;
2. requires HTTP `206` and an exact `Content-Range` containing the pinned source size;
3. caps every header below 1 MiB in the native inspector;
4. verifies a separately pinned SHA-256 for the prefix;
5. publishes the 51-file metadata artifact atomically into the immutable model store.

The resulting import is 19,404,391 bytes, including 7,663,144 bytes of header snapshots.
No tensor payload was downloaded. A full DeepSeek import is also protected by a separate
`--allow-large-download` opt-in in addition to the existing free-space reserve and atomic
publication rules.

## Exact tensor storage results

| Safetensors dtype | Tensors | Payload bytes |
|---|---:|---:|
| BF16 | 433 | 2,830,518,528 |
| F32 | 417 | 144,672,072 |
| F8_E4M3 | 375 | 6,023,020,544 |
| F8_E8M0 | 34,167 | 8,858,737,664 |
| I8 packed expert storage | 33,792 | 141,733,920,768 |
| I64 | 3 | 18,616,320 |
| **Total** | **69,187** | **159,609,485,896** |

Exact routed-expert accounting derived from the verified headers:

| Metric | Bytes |
|---|---:|
| Routed expert tensors | 150,592,290,816 |
| Always-active/non-routed tensors | 9,017,195,080 |
| One routed expert | 13,369,344 |
| Six active experts in one layer | 80,216,064 |
| Main-model no-cache expert payload per token | 3,449,290,752 |

The last value is a logical lower-bound input for later streaming experiments, not a
measured NVMe bytes/token result. It excludes caching, prefetch waste, filesystem/device
alignment and MTP execution.

## Validation host

- Windows 11 Pro 10.0.26200;
- AMD Ryzen 9 7900X, 24 logical CPUs;
- 32 GB DDR5 RAM;
- NVIDIA RTX 4070 Super, 12,282 MiB reported VRAM;
- Lexar SSD NM790 2 TB, NTFS;
- WSL 2 Linux used for GCC, Clang and sanitizer builds;
- GPU detected but not used.

The external metadata import lives outside the repository. No downloaded model artifact
or machine-specific path is committed.

## Build and test matrix

| Toolchain/configuration | Result |
|---|---:|
| MSVC 19.44 Release, warnings as errors, real metadata import | 6/6 passed in 3.85 s |
| MSVC 19.44 Debug, warnings as errors, real metadata import | 6/6 passed in 12.09 s |
| GCC 13.3 Release, warnings as errors, real metadata import | 6/6 passed in 7.60 s |
| GCC 13.3 Debug, warnings as errors, real metadata import | 6/6 passed in 11.90 s |
| Clang 18.1 Release, warnings as errors, real metadata import | 6/6 passed in 7.69 s |
| GCC Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 6/6 passed in 25.97 s |

ASan ran with leak detection and halt-on-error; UBSan was configured to halt on the
first finding. The real integration test also mutates the pinned config without changing
its length: full mode rejects its SHA-256 and quick mode independently rejects the
architecture drift.

Additional checks passed:

- Python `compileall` for `tools` and `tests`;
- offline importer tests, including resumed ranges and bounded header ranges;
- offline manifest-tool tests (the Windows symlink case was skipped because symlink
  creation was unavailable there; it remains covered on Linux);
- native Windows invocation of `deepseek-inspect --json` with full metadata hashes.

## Performance

No tokens/s, TTFT or physical NVMe bytes/token result is reported because no DeepSeek
inference occurred. The metadata import completed in 35.6 seconds on the recorded host,
but this network-dependent setup time is not a runtime performance metric.

## Known limitations and next bottleneck

- no architecture-faithful reduced DeepSeek fixture;
- no trusted reference logits, expert routes or output tokens;
- no full thinking/tool-call template surface or completion parser;
- no DeepSeek attention, hash router, MTP or packed FP4/FP8 compute kernels;
- no full weight download or full-model probe;
- no DeepSeek CUDA, VRAM tier or NVMe-to-VRAM execution;
- no end-to-end accepted-token I/O measurement.

The next bottleneck is an architecture-faithful reduced fixture generated from the pinned
official inference code, followed by deterministic reference vectors. M6 closes only when
that fixture agrees on tensors, logits, routes and tokens within an explicit tolerance.
