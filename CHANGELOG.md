# Changelog

## Unreleased

- Pinned the official `deepseek-ai/DeepSeek-V4-Flash` checkpoint at immutable revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`; removed the unsupported provisional
  “0731” identity from the active planning manifest.
- Added a 19,404,391-byte metadata/header import that uses exact HTTP ranges, pinned
  SHA-256 values and atomic publication instead of downloading 159,617,149,040 bytes
  of weight shards.
- Added `deepseek-inspect` with exact validation for 46 shards, 69,187 tensors,
  Safetensors shapes/offsets, FP8/FP4 storage dtypes and routed-expert byte layout.
- Added native DeepSeek V4 Unicode pre-tokenization, its 129,280-token byte BPE,
  bounded added-token trie and basic system/user/assistant chat framing. Exact IDs match
  Hugging Face tokenizers 0.22.1 across committed multilingual reference vectors.
- Added bounded native DeepSeek reference equations and deterministic vectors for RMSNorm,
  learned/hash routing, routed SwiGLU, Hyper-Connections, base RoPE and selected-latent
  sparse attention. These are operator tests, not a DeepSeek forward or generation claim.
- Added a deterministic 147,136-byte graph exported from the SHA-256-pinned official
  `model.py`, a bounded native two-layer FP32 prefill and one checkpoint-shaped MTP block.
  Main and MTP logits, proposed tokens, learned/hash expert routes and raw/compressed attention
  indices pass the declared parity gate; full-checkpoint and speculative DeepSeek execution
  remain unsupported.
- Extended the fixture to 159,533 bytes with five official single-token decode steps. Native
  main/MTP logits, routes and raw/compressed attention indices match across circular-window and
  overlapping 4:1 cache updates; state mutation is transactional and aggregate-budgeted before
  allocation.
- Recorded, without downloading weights, the newer official DeepSeek V4 Flash 0731 release,
  its Unsloth GGUF profiles and Laguna S 2.1 as separate compatibility targets. None inherits
  the preview checkpoint's execution claims; the universal tensor/expert residency contract
  and per-adapter acceptance ladder are now explicit.
- Added the shared logical tensor catalog and bound all 69,187 pinned DeepSeek preview tensors
  to role, shape, dtype and immutable source ranges. Its 67,584 routed tensors form 11,264
  main/MTP expert bundles whose six tensors coalesce into exactly two ranges per expert; no
  payload read or execution support is claimed.
- Added the governed multitier residency state machine. RAM plus bounded staging and later VRAM
  destinations are reserved before transition publication; failed compound reservations roll
  back, while pins and RAII use leases prevent unsafe eviction. The state-only slice allocates no
  tensor buffer and makes no new inference claim.
- Added the first governed packed-expert load transaction: canonical source ranges are read into a
  reserved compact RAM buffer, every expert tensor receives an exact packed slice and an RAII use
  lease prevents eviction until the bytes are destroyed. The synchronous fixture baseline is not
  a DeepSeek or physical-NVMe performance claim.
- Added a shared bounded asynchronous expert-load pool with one to four fixed workers, a queue of
  at most 64 requests, pre-enqueue RAM/staging reservations, exclusive transfer leases, safe
  queue-full rollback, shutdown cancellation and queue/service/total telemetry. This enables a
  future compute/I/O overlap path but does not itself prove DeepSeek throughput.
- Added an explicit `--allow-large-download` gate for checkpoint imports containing at
  least 20 GiB of weight payload. DeepSeek execution remains unsupported and is reported
  as such by the CLI.
- Added path-scoped device-I/O snapshots: Windows maps the checkpoint volume to
  `PhysicalDriveN` and samples `IOCTL_DISK_PERFORMANCE`; Linux accepts non-virtual
  `/sys/dev/block/.../stat` mappings and otherwise reports the counter unavailable.
- Added the fail-closed `direct-unbuffered-v1` cache-bypass protocol for bounded OLMoE
  expert reads. It requires async no-cache execution, a trustworthy device counter and
  measured device bytes at least equal to logical expert bytes; it never flushes the
  global operating-system file cache.
- Validated three repeatable Windows cache-bypass runs with identical token `47111`,
  1,610,612,736 logical bytes and 1,610,743,808 whole-device bytes per run.
- Extended native OLMoE traces from 12 to 19 fields with submission, queue, ready,
  compute-window, route-order and reduction-order telemetry while retaining all earlier
  trace formats.
- Added a bounded trace replay that compares fixed-route execution with theoretical
  ready-expert compute order while preserving the authoritative reduction order.
- Added opt-in two-stage OLMoE reads: contiguous 8 MiB gate/up plus 4 MiB down requests,
  two workers, bounded queue/staging reservations, exact byte accounting and down-read
  versus gate/up-compute overlap telemetry.
- Validated staged reads against the pinned Transformers logits/routes and measured a
  46.0% median warm buffered-pipeline improvement over whole 12 MiB reads on three paired
  WSL runs. This is not a physical-NVMe or DeepSeek throughput claim.

## 0.4.0-alpha.4 — 2026-08-09

- Renamed the public project, CLI, C++ namespace, headers, CMake targets and current
  documentation to **Vastloom** with the slogan “Run giant models. Keep your machine yours.”
- Preserved read compatibility for legacy manifests, imported-checkpoint metadata and
  model-store markers so existing verified model downloads remain immutable and reusable.
- Added Apache-2.0 licensing, community governance, support and security policies,
  contribution templates, CodeQL, dependency review and release packaging automation.
- Added a purpose-built README cover and two-to-one social preview, plus public brand and
  release documentation.
- Regenerated the bundled synthetic fixture under the Vastloom model identity and retained
  the original native binary and heartbeat wire magic for format compatibility.

Historical entries below retain the project name and command spelling used by those releases.

## 0.4.0-alpha.3 — 2026-08-09

- Added cross-platform process-I/O snapshots around the real OLMoE forward path.
- Exposed Windows read operation/transfer deltas and Linux logical plus storage-layer
  read deltas while preserving their different semantics in text and JSON output.
- Replaced fragmented Windows asynchronous stream reads with Unicode-safe bounded
  `CreateFileW`/`ReadFile` ranges; one expert now requires one 12 MiB operation.
- Added exact unique-expert reuse-distance telemetry with cold/reused access counts,
  reuse ratio, mean, P50, P95 and maximum distance.
- Added Unicode-path, counter-semantics, operation-fragmentation and real repeated-route
  regression tests.

This slice measures process-visible I/O and routing locality without claiming physical
NVMe traffic. Buffered cache state, cold-cache behavior, device counters, OLMoE decoding,
CUDA and DeepSeek remain outside the validated scope.

## 0.4.0-alpha.2 — 2026-08-09

- Connected the real OLMoE expert path to the reservation-backed bounded I/O pool.
- Added opt-in synchronous/asynchronous selection, one-worker prefetch depth one or two,
  12 MiB request bounds, cancellation and per-request timeouts capped at 60 seconds.
- Coalesced each pinned expert's three validated contiguous BF16 tensors into one bounded
  read while preserving the native gate/up/down layout and reference logits.
- Added request queue/service/total timing plus expert prefetch count, foreground wait
  and measured read-service/CPU overlap telemetry.
- Extended atomic OLMoE CSV traces from nine to twelve fields while retaining compatibility
  with four-field baselines and nine-field legacy traces.
- Added deterministic I/O worker, short-read, allocation, cancellation and timeout fault
  injection, including real-checkpoint fail-closed integration coverage.
- Retained synchronous I/O as the default and capped the asynchronous OLMoE adapter to
  one worker, two outstanding prefetch slots and Resource Governor staging reservations.

This slice validates bounded buffered read/CPU overlap without changing model output. It
does not claim physical NVMe bytes, cold-cache disk throughput, direct I/O, OLMoE text
generation, CUDA or DeepSeek support.

## 0.4.0-alpha.1 — 2026-08-09

- Pinned and fully verified `allenai/OLMoE-1B-7B-0924-Instruct` revision
  `7f1c97f440f06ce36705e4f2b843edb5925f4498` with three immutable BF16 shards.
- Added exact OLMoE configuration, tokenizer, chat-template and 3,219-tensor layout
  validation without requiring tensor payloads to reside in RAM.
- Added a native CPU OLMoE reference graph with RMSNorm, Q/K normalization, RoPE,
  causal attention, authoritative softmax top-8 routing and weighted SwiGLU experts.
- Added explicit NVMe-to-RAM expert streaming: the 953,421,824-byte always-active
  trunk remains in reserved RAM while one 12 MiB expert is read at a time.
- Added bounded no-cache, LRU, LFU and static-hot expert-cache policies, with a hard
  two-GiB cache ceiling and Resource Governor reservations before allocation.
- Added `moe-inspect` and `moe-logits`, atomic route telemetry, exact expert byte/I/O
  accounting, timeout/Ctrl+C checks and metadata-only execution refusal.
- Added an independently bounded Transformers 4.45.2 oracle and committed one- and
  two-token fixtures for logits, greedy token, route IDs and routing weights.
- Extended trace simulation to accept native telemetry and compare no-cache, LRU,
  LFU and static-hot policies.

This milestone proves correct real MoE routing and synchronous NVMe-to-RAM expert
streaming on CPU under a host-derived memory cap. It does not yet provide OLMoE text
generation/chat, CUDA, asynchronous I/O/compute overlap or a DeepSeek adapter.

## 0.3.0-alpha.1 — 2026-08-09

- Added a bounded JSON parser and defensive Safetensors catalog/reader.
- Added a 300 GiB-reserve model store, immutable atomic Hugging Face importer,
  and host-derived real-model runtime policy.
- Added Windows process memory Job Object support and live commit/headroom checks
  for real-model execution paths.
- Added `store-info`, `runtime-policy`, and Safetensors-aware `inspect` commands.
- Added the first genuinely pretrained model adapter, pinned to
  `HuggingFaceTB/SmolLM2-135M-Instruct` revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Added the model's GPT-2 byte-level BPE tokenizer, exact chat template, BF16
  Safetensors loader, Llama CPU reference kernels, grouped-query attention,
  incremental KV cache, greedy/seeded generation and persistent chat history.
- Added trusted Transformers parity fixtures for tokenization, logits, greedy output
  and multi-turn chat, plus a native real-model self-test.
- Added bounded process-memory enforcement, six-thread maximum execution,
  cancellation and full-checkpoint verification by default for the real path.

This milestone proves one real published dense checkpoint on the native CPU path. It
does not yet include MoE execution, NVMe expert streaming or CUDA.

## 0.2.0-alpha.2 — 2026-08-09

- Added the first native end-to-end token-generation path.
- Added the bounded `SGTMV001` TinyTransformer model format and strict loader.
- Added byte-level prompt encoding, CPU LayerNorm/attention/MLP kernels and incremental
  KV caching.
- Added greedy decoding and deterministic seeded top-k/temperature sampling.
- Added `stargate chat`, `stargate model-info` and `stargate self-test`.
- Added interactive streaming, one-shot JSON output, generation timeouts, token limits,
  system-prompt override and response statistics.
- Added the 358.91 KiB `stargate/tinychat-v1` validation model, exact manifest and five
  deterministic response vectors.
- Added a reproducible PyTorch training/export tool that atomically publishes the
  model, vectors and SHA-256 manifest only after all probes pass.
- Added end-to-end CLI integration tests, integrity-failure tests, malformed-model
  tests, deterministic-sampling tests and additional Resource Governor checks.
- Added explicit generation finish reasons and valid-Unicode repair for arbitrary sampled
  bytes in JSON output.
- Added manifest-to-binary metadata cross-checks for model id, layer count and context
  length.
- Added concurrent generation coverage and ThreadSanitizer validation of the shared model
  path.
- Added a UTF-8 Windows `wmain` entry point and UTF-8 console configuration for accented
  CLI prompts.
- Added integration coverage for direct `.sgtm` loading and explicit quick manifest mode.
- Added CMake installation of the bundled validation model under `share/stargate/models`.
- Regenerated the committed fixture from the current exporter and verified byte-identical
  output across independent training runs.
- Fixed model/governor destruction order so weight reservations are released before
  the owning governor is destroyed.
- Added UTF-8 MSVC compilation and expanded warning-as-error/sanitizer validation.

This release validates the native CPU inference pipeline. It does not yet include a
DeepSeek/MiniMax/Kimi adapter, GPU kernels or NVMe-backed generation.

## 0.1.0-alpha.1 — 2026-08-08

- Added C++20/CMake project foundation.
- Added hardware profiler and JSON reporting.
- Added conservative hardware-adaptive execution planner.
- Added atomic resource reservations and safety state machine.
- Added bounded asynchronous I/O with reservation-backed buffers.
- Added strict model manifest, SHA-256 and path/symlink confinement.
- Added immutable model-store layout command.
- Added atomic heartbeat and independent watchdog.
- Added MoE trace simulator with LRU/static-hot policies and bytes/token metrics.
- Added unit tests, warning-as-error build, ASan/UBSan and CI definitions.

No model inference kernels are included in this release.
