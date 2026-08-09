# Security policy

## Supported versions

Vastloom is pre-release research software. Security fixes are made on the latest alpha line;
older snapshots are not maintained.

| Version | Supported |
|---|---|
| Latest `0.4.0-alpha.*` | Yes |
| Older alpha releases | No |

## Reporting a vulnerability

Please email `dev@mikesoft.it` with the subject `Vastloom security report`. Do not open a
public issue for path traversal, arbitrary file access, unsafe process termination, model
artifact corruption, malformed-model crashes, resource exhaustion or privacy failures.

Include the affected revision, platform, reproduction steps and impact. Avoid attaching
copyrighted model weights or sensitive prompts. You may use a minimal synthetic fixture.
The maintainer will acknowledge receipt, coordinate triage and agree on disclosure timing.

## Trust boundaries

- The native runtime processes local models, prompts and hardware metadata. It has no HTTP
  inference server and does not transmit prompts or telemetry.
- `tools/import_hf_model.py` is a separate opt-in network tool. It downloads only explicitly
  pinned HTTPS artifacts, checks declared sizes and hashes, preserves free-disk headroom and
  publishes an immutable checkpoint atomically.
- Runnable `.sgm` manifests are fully SHA-256 verified by default. Direct `.sgtm` input and
  `--no-verify` are controlled-development modes and weaken identity verification.
- Original model weights are immutable. Derived data belongs in reconstructable cache or a
  separately validated, atomically published artifact.
- The watchdog observes by default. Process termination requires explicit `--enforce` plus
  matching PID and heartbeat checks.

Vastloom cannot make untrusted model licenses safe, guarantee that a host has adequate
cooling, or prevent all consequences of running experimental native code. Keep backups,
start with the safe profile and review the plan before enabling large-model execution.
