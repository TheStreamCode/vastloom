# Validation reports

Reports are evidence snapshots tied to a date, host, source revision and project state. They are not
rolling claims: a later code change does not retroactively alter an earlier result.

Releases through `0.4.0-alpha.3` used the former project name, executable and persisted identifiers.
Those historical documents are intentionally preserved verbatim. Beginning with
`0.4.0-alpha.4`, current documentation and commands use **Vastloom**. Compatibility readers keep
the already-downloaded model stores and checkpoint manifests usable without mutation.

Start with the newest report for current claims, then consult milestone-specific reports for exact
hardware, commands, measurements and limitations.

## Current report

- [`M6_DEEPSEEK_METADATA_INSPECTION_VALIDATION.md`](M6_DEEPSEEK_METADATA_INSPECTION_VALIDATION.md) —
  exact official DeepSeek V4 Flash revision, bounded header-only import, native
  69,187-tensor inspection and the remaining non-execution boundary.
- [`M6_DEEPSEEK_TOKENIZER_VALIDATION.md`](M6_DEEPSEEK_TOKENIZER_VALIDATION.md) —
  native Unicode byte-BPE and basic chat-template parity against the pinned official
  tokenizer, including malformed-layout rejection and cross-platform sanitizer gates.
- [`M6_DEEPSEEK_REFERENCE_OPERATORS_VALIDATION.md`](M6_DEEPSEEK_REFERENCE_OPERATORS_VALIDATION.md) —
  bounded CPU equations and reproducible vectors for RMSNorm, routing, routed SwiGLU,
  Hyper-Connections, base RoPE and selected-latent sparse attention.

M6 is still open: these primitives are not a forward graph, and official reference
logits/routes/tokens have not passed their gates. The latest completed milestone remains M5:

- [`M5_DEVICE_IO_CACHE_BYPASS_VALIDATION.md`](M5_DEVICE_IO_CACHE_BYPASS_VALIDATION.md) —
  device-level Windows/Linux counters, bounded direct/unbuffered reads, repeatable
  Windows physical-byte measurements and the completed M5 gate.

The preceding [`M5_READINESS_STAGED_IO_VALIDATION.md`](M5_READINESS_STAGED_IO_VALIDATION.md)
records readiness replay, output-identical two-stage expert reads, bounded two-worker
safety and paired warm-buffered measurements.

The latest release-level report remains
[`ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md`](ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md),
covering the Vastloom rebrand, compatibility, governance, release packaging, visual
assets and cross-platform validation.
