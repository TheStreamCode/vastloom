# M9 DeepSeek bounded reference-generation validation

Validation date: 2026-08-11

Implementation commit: `3f39697e43f92e9d9b2fb12b8bbff3a137eac8e6`

## Scope and result

This milestone closes a bounded token-level generation loop over the already validated native
DeepSeek reference graph:

```text
bounded prompt token IDs
  -> transactional multi-layer prefill
  -> main-model logits
  -> greedy or portable seeded top-k sampling
  -> stop-token/context/maximum-token decision
  -> transactional incremental raw/compressed state
  -> repeat with per-layer safety checkpoints
  -> exact expert-cache and logical-read telemetry
```

The trusted fixture was regenerated from the hash-pinned preview `model.py`. From the committed
eight-token prompt, its independent FP32 graph produces this eight-token greedy continuation:

```text
4 11 14 4 11 14 4 11
```

The native driver produces the same sequence. A separate temperature-0.7, top-5 sequence with
seed `0x564153544c4f4f4d` is fixed across the tested standard libraries as:

```text
4 11 13 2 13 11
```

This is reduced-fixture token parity. It is not proof that the official 82.54 GB GGUF payload can
generate tokens, and it is not a chat CLI for that checkpoint.

## Implemented

- A public bounded `deepseek_v4_reference_generate` token driver composes prefill and incremental
  decode without exposing partially updated cache state.
- Greedy and seeded top-k/temperature sampling validate finite logits and use raw `mt19937_64`
  bits for cross-library reproducibility. Subtract-before-divide softmax remains defined for the
  smallest positive FP32 temperature.
- Generation is capped at 4,096 reference tokens, one hour, the caller's smaller context window,
  4,096 new tokens and 64 unique in-vocabulary stop tokens.
- Finish reasons are explicit: `stop_token`, `context_limit`, `max_tokens`, `cancelled` and
  `timeout`.
- Execution checkpoints run before graph work, each main layer, the final head, MTP and state
  publication. They can sample the `ProcessSafetyGuard`; callback exceptions fail closed.
- Cancellation or timeout during prefill/decode returns a clean partial generation result while
  the caller's incremental state remains unchanged.
- Generation aggregates only observed expert accesses: request, hit, miss, resident-byte,
  logical-byte, load-service, compute and I/O/compute-overlap totals. It rejects non-finite or
  overflowing telemetry rather than publishing it.
- The trusted graph fixture format is now `vastloom.deepseek_v4.graph_fixture.v3` and its exporter
  recreates generation from a fresh official-reference model so earlier arbitrary decode probes
  cannot contaminate the cache.

MTP logits remain part of each reference forward pass because the pinned graph oracle validates
them. MTP proposals are not accepted speculatively and never select the authoritative generated
token in this driver.

## Correctness and fault coverage

The native tests cover:

- exact eight-token greedy parity with the independently regenerated official-reference graph;
- exact portable seeded-sampling output and repeated-run determinism;
- the smallest positive FP32 temperature;
- stop token before publication, context truncation and maximum-token completion;
- streaming token callback equivalence with the accumulated token vector;
- immediate timeout, cancellation after one published token and throwing callbacks;
- empty/oversized generation, non-finite temperature, invalid top-k/context/timeout and malformed
  stop-token sets;
- a layer checkpoint failure during prefill and decode with the previous cache state unchanged;
- a real bounded two-entry fixture expert cache across three generation forwards, where aggregated
  requests, hits, misses, resident bytes and logical bytes exactly match the cache snapshot;
- zero inferred expert traffic when the graph uses resident fixture tensors without a provider.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported VRAM, driver 610.88;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- warnings treated as errors in every compiler build;
- no CUDA code and no official GGUF payload executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release + real metadata opt-ins | 15/15 passed |
| GCC Debug + real metadata opt-ins | 15/15 passed in 32.30 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 85.71 s |
| Clang 18 Release + real metadata opt-ins | 15/15 passed in 13.83 s |
| GCC Debug + libstdc++ debug iterators/assertions | 15/15 passed in 53.56 s |
| MSVC Release x64 + real metadata opt-ins | 15/15 passed in 10.63 s |
| MSVC Debug x64 + real metadata opt-ins | 15/15 passed in 57.39 s |
| Pinned official-reference fixture regeneration/check | passed byte for byte |
| Python manifest suite | 3 passed, 1 Windows symlink case skipped |
| Python importer suite and `compileall` | passed |

The two native TSan binaries passed with `TSAN_OPTIONS=halt_on_error=1` under
`setarch x86_64 -R`, avoiding the known WSL address-layout conflict. This is focused native
evidence, not a complete TSan CTest matrix.

## Performance statement

No model speed, physical NVMe bandwidth or useful bytes/token result is reported. The generation
fixture has width 8, two main layers, one MTP layer and vocabulary 16; its timing is dominated by
test overhead and cannot represent the 43-layer checkpoint. The milestone measures exact logical
expert bytes from the cache-backed fixture only and makes no hardware-throughput claim.

## Known limitations and next bottleneck

- The driver consumes an already assembled graph; it does not construct the complete official
  graph from streamed artifacts.
- The bounded reference graph remains limited to 4,096 tokens and reduced execution dimensions.
- There is no official 43-layer trunk-plus-expert token path.
- The official globals, layer-0 trunk and authoritative `tid2eid` route remain dry-run/metadata
  only.
- The native DeepSeek tokenizer is not yet connected to this token-level driver as a chat CLI.
- Thinking/tool-call parsing, speculative MTP verification, physical NVMe overlap and CUDA remain
  unimplemented.

The next correctness bottleneck is unchanged at the payload boundary: after explicit user
authorization, acquire the exact 138,867,160-byte layer-0 trunk, execute its real `tid2eid` route
for a pinned token input and load only the six authoritative complete expert bundles. That result
can then replace the reduced layer inside this now-validated generation lifecycle.
