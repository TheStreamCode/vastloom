# M7 DeepSeek V4 Flash incremental-state validation

Validation date: 2026-08-10

Implementation commit: `e242bcb`

## Scope

This report closes the reduced incremental raw/compressed attention-state gate for the pinned
`deepseek-ai/DeepSeek-V4-Flash` preview revision
`60d8d70770c6776ff598c94bb586a859a38244f1`:

```text
8-token official prefill
  -> bounded main + MTP cache state
  -> five single-token decode steps
  -> raw circular window + learned 4:1 overlap compression
  -> main + MTP logits, routes and sparse-attention indices
```

The newer official `DeepSeek-V4-Flash-0731` release is a separate target and was not executed by
this report. No full checkpoint, GGUF, CUDA allocation or NVMe expert stream was used.

## Incremental state

Each main/MTP attention layer owns explicit bounded state:

- circular raw KV window;
- learned attention-compressor projected values and scores;
- committed compressed attention KV;
- independent learned-indexer compressor values, scores and compressed KV.

Prefill populates exactly the layouts used by official decode. Incremental decode writes the
current raw slot, updates compressor state, commits a new compressed entry only at a complete
ratio boundary and builds the official circular/local plus compressed index order.

State updates are transactional. Decode works on a bounded copy and publishes it only after the
entire main and MTP step, logits and post-validation succeed. A missing LM head injected after
cache work leaves the caller's state byte-for-byte unchanged for the checked cache vectors.

## Trusted reference fixture

The 159,533-byte fixture remains generated from the SHA-256-pinned official `model.py` using
deterministic FP32 values and equation-equivalent CPU kernel stubs. Its five decode tokens cover
positions 8–12:

| Position | Input token | Main greedy | MTP proposed | Main selected indices/layer | MTP selected indices |
|---:|---:|---:|---:|---|---:|
| 8 | 10 | 4 | 12 | 3 / 5 | 3 |
| 9 | 12 | 11 | 5 | 3 / 5 | 3 |
| 10 | 13 | 11 | 5 | 3 / 5 | 3 |
| 11 | 14 | 4 | 12 | 3 / 5 | 3 |
| 12 | 15 | 4 | 12 | 3 / 5 | 3 |

Position 11 completes a new overlapping 4:1 compressed entry; position 12 consumes the updated
compressed/indexer caches. This prevents the test from passing with a raw-window-only cache.

For every step, native output matched:

- all 16 main and 16 MTP logits within absolute tolerance `8e-5`;
- exact main greedy and MTP proposed token IDs;
- every main and MTP expert ID exactly;
- every main and MTP routing weight within `8e-5`;
- every circular/compressed sparse-attention index exactly.

The prefill test still removes every one of 123 fixture tensors and requires failure. Additional
tests reject malformed state, out-of-range tokens, non-finite weights, missing tensors and a
precomputed state budget above the 64-million-element ceiling.

## Build and test matrix

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors | 11/11 passed in 84.68 s |
| MSVC 19.44 Debug, warnings as errors | 10/10 passed in 114.33 s |
| GCC 13.3 Release, warnings as errors | 8/8 passed in 3.69 s |
| Clang 18.1 Release, warnings as errors | 8/8 passed in 3.62 s |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 8/8 passed in 9.90 s |

ASan ran with leak detection and halt-on-error; UBSan halted on its first finding. Python byte
compilation, offline importer tests, official fixture byte reproduction, dependency-free fixture
validation and `git diff --check` also passed. The manifest suite reported one pass and one
Windows symlink-capability skip.

## Safety boundary

The runtime computes the complete cache-state element budget before allocation. Individual
shape products and the aggregate state are bounded; malformed dimensions, cache lengths,
non-finite cache values and unexpected compressor contents are rejected. Prefill/decode remain
capped at 4,096 positions, 16 total main/MTP layers, one MTP layer and 64 million state elements.

The fixture is approximately 156 KiB. Test wall time is not a decode-speed measurement, and no
tokens/s, TTFT, device traffic, thermals or large-model memory claim is made.

## Known limitations and next bottleneck

- reduced FP32 fixture only, not packed checkpoint tensors;
- no target-verification/acceptance loop around MTP proposals;
- no cancellation callback inside long reference operators;
- no full reasoning/tool-call parser;
- no CUDA or NVMe-to-VRAM path;
- no full-model probe for the preview or 0731 release.

The next bottleneck is a role-aware packed tensor/expert binding shared by Safetensors and future
GGUF adapters, followed by an explicitly budgeted NVMe→RAM→VRAM expert pipeline whose primary
metric is bytes read per accepted token.
