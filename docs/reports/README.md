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

- [`M5_READINESS_STAGED_IO_VALIDATION.md`](M5_READINESS_STAGED_IO_VALIDATION.md) —
  readiness replay, output-identical two-stage expert reads, bounded two-worker safety,
  paired warm-buffered measurements and the remaining physical-I/O gate.

The latest release-level report remains
[`ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md`](ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md),
covering the Vastloom rebrand, compatibility, governance, release packaging, visual
assets and cross-platform validation.
