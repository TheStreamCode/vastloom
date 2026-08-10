# M7 universal logical tensor catalog validation

Validation date: 2026-08-10

Implementation commit: `3362adc`

## Scope

This slice turns verified checkpoint metadata into model-independent tensor and routed-expert
objects. The shared catalog records role, dtype, shape, logical layer/expert identity and the
immutable source shard/offset/length. It then groups the routed tensors selected together by the
MoE graph into bounded expert bundles.

The first real binding uses the pinned `deepseek-ai/DeepSeek-V4-Flash` preview revision
`60d8d70770c6776ff598c94bb586a859a38244f1`. Only its verified metadata and original
Safetensors headers were used. No weight payload, GGUF file, CUDA allocation or model generation
was involved.

## Implemented contract

The new shared `model/tensor_catalog` layer validates and finalizes:

- non-empty model/revision and unique tensor names;
- bounded tensor and expert counts;
- non-empty source ranges with checked offset and aggregate-byte arithmetic;
- absence of overlapping ranges within every immutable shard;
- mandatory layer/expert identity for every routed-expert tensor;
- deterministic `(main-or-speculative, layer, expert)` bundle ordering;
- coalescing only when two ranges are exactly adjacent in the same shard;
- graceful `resource_exhausted` failure if bounded catalog allocation fails.

All derived totals and bundle vectors are built locally and published only after validation. The
catalog does not require the payload files to exist during header-only inspection and cannot read
from a metadata snapshot.

## Pinned DeepSeek result

The existing exact name-to-shard validator now classifies all 69,187 tensors and binds their
original Safetensors offsets into the shared contract:

| Identity | Verified value |
|---|---:|
| Total tensor payload | 159,609,485,896 bytes |
| Always-active/non-routed tensors | 9,017,195,080 bytes |
| Routed-expert tensors | 67,584 |
| Routed-expert payload | 150,592,290,816 bytes |
| Main plus MTP expert bundles | 11,264 |
| Tensors per expert bundle | 6 |
| Bytes per expert bundle | 13,369,344 |
| Coalesced source ranges | 22,528 |
| Coalesced ranges per expert | exactly 2 |

The two-range result is derived from the real pinned shard layout: the six tensors of each expert
are not submitted as six independent reads. This is a structural request-count reduction, not an
SSD bandwidth, latency, cache-bypass or tokens-per-second measurement.

`deepseek-inspect --json` now reports `logical_catalog_valid`, `expert_bundle_count` and
`expert_read_range_count` while continuing to report `execution_supported=false`.

## Tests

The generic unit test covers a trunk tensor, an expert spanning adjacent and cross-shard ranges,
exact coalescing, deterministic totals, duplicate names, missing expert identity, overlapping
ranges, offset overflow and configured count limits. The opt-in pinned DeepSeek integration test
requires the exact 11,264-bundle/22,528-range layout and retains its checksum, corruption,
tokenizer and chat-framing checks.

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors | 11/11 passed |
| MSVC 19.44 Debug, warnings as errors | 10/10 passed |
| GCC 13.3 Release, warnings as errors | 8/8 passed |
| Clang 18.1 Release, warnings as errors | 8/8 passed |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 8/8 passed |
| Pinned DeepSeek metadata/header integration | 1/1 passed |

ASan was configured with leak detection and halt-on-error; UBSan halted on its first finding.
Python byte compilation, the offline Hugging Face importer test and `git diff --check` also
passed.

## Known limitations and next bottleneck

- The logical catalog is implemented, but legal residency transitions and ownership are not.
- Source ranges have not yet been issued through a DeepSeek payload reader.
- The pinned preview stores one expert in two coalesced regions; no physical SSD throughput or
  overlap was measured.
- Packed I8/FP8/FP4 interpretation and CPU/CUDA kernels do not exist for the full checkpoint.
- The official 0731 Safetensors release, its Unsloth GGUF derivatives and Laguna S 2.1 still need
  independent pinned adapters and parity gates.

The next bottleneck is a Resource-Governor-backed residency object with explicit
`NVME -> READING -> RAM_READY -> COPYING_TO_VRAM -> VRAM_READY` transitions, followed by bounded
reads of these exact expert ranges and bytes-per-accepted-token telemetry.
