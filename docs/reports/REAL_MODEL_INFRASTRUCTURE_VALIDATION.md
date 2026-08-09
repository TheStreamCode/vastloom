# Real-model infrastructure validation

Validation date: 2026-08-09

Implementation commit: `13443cd1de41d1fee0bc7d5f48b6f376e7bc42e2`

## Implemented

- bounded JSON parsing with byte, depth, node, string, array and object limits;
- defensive Safetensors header parsing, dtype/shape/offset/overflow validation,
  overlap detection, multi-shard tensor cataloging and 64 MiB bounded reads;
- an external model-store layout with immutable originals, reconstructable cache,
  scratch, state and telemetry areas;
- a configurable free-space reserve, set to 300 GiB on the reference PC;
- a pinned Hugging Face importer using bounded buffers, exact sizes, SHA-256,
  temporary artifacts and atomic directory publication;
- a runtime policy capped at six CPU threads, two I/O workers, four queued reads,
  512 MiB staging, 64 MiB requests, 8 GiB system RAM headroom and 300 GiB disk
  headroom;
- Windows Job Object process-memory enforcement plus live commit/headroom probes;
- `inspect`, `store-init`, `store-info` and `runtime-policy` CLI coverage.

## Actually tested

The native suite tests valid and malformed JSON, duplicate keys, nesting/input
limits, valid and invalid Safetensors offsets, bounded range reads, model-store
identity/path checks, disk-preflight rejection and safe runtime-policy rejection.
The Python importer suite is offline and verifies store identity, metadata/full
file selection, dry-run behavior and immutable-target mismatch rejection.

| Gate | Exact result |
|---|---:|
| MSVC 19.44 Release, `/WX` | 5/5 passed |
| MSVC 19.44 Debug, `/WX` | 5/5 passed |
| GCC 13 Release, `-Werror` | 5/5 passed |
| Clang 18 Release, `-Werror` | 5/5 passed |
| GCC 13 Debug + ASan + UBSan | 5/5 passed |
| Python `py_compile` | passed |

ASan used leak detection and halt-on-error. UBSan used halt-on-error and stack
traces.

## Real checkpoint inspection

The external store was initialized at `C:\StargateModels`; it is not part of the
Git repository. The pinned `HuggingFaceTB/SmolLM2-135M-Instruct` revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac` was imported and revalidated.

- checkpoint bytes: 269,060,552;
- tensor payload bytes: 269,030,016;
- tensors: 272;
- dtype: BF16 for all tensors;
- model-weight SHA-256:
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.

Five Release inspections took 18.834 ms cold and 7.785-8.759 ms warm. This is
metadata inspection performance, not inference performance.

## Known limitations and next bottleneck

No real-model logits or tokens are claimed yet. The next bottleneck is the native
GPT-2 byte-level tokenizer plus architecture-specific Llama CPU reference path,
followed by logit and greedy-output comparison against the pinned Transformers
reference. CUDA and large-MoE loading remain outside this milestone.
