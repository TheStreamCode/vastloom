# Changelog

## Unreleased

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
