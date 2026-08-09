# Changelog

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
