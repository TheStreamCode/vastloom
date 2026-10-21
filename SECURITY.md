# Security policy

## Supported versions

Vastloom is pre-release research software. Security fixes are made on the latest alpha line;
older snapshots are not maintained.

| Version | Supported |
| --- | --- |
| Latest alpha release | Yes |
| Older alpha releases | No |

## Reporting a vulnerability

Do not open a public issue for a security problem. Report it privately through GitHub Security
Advisories on this repository
(<https://github.com/TheStreamCode/vastloom/security/advisories/new>), or by email to
<michael@mikesoft.it> with the subject `Vastloom security report` when GitHub is not available.

Include the affected revision and platform, a description of the issue and its impact, reproduction
steps or a proof of concept, and any mitigation you already know about. Avoid attaching copyrighted
model weights or sensitive prompts; a minimal synthetic fixture is enough. You can expect an initial
acknowledgement within 7 days and a status update within 30 days. Disclosure timing is agreed with
the reporter, and credit is given unless you ask otherwise.

## Scope

In scope:

- path traversal, arbitrary file access, unsafe process termination, model artifact corruption,
  malformed-model crashes, resource exhaustion and privacy failures in this repository;
- credential or secret exposure caused by this project;
- defects in the runtime, build scripts or release artifacts.

Out of scope:

- vulnerabilities in third-party models, weights, libraries or drivers; report those upstream;
- misconfiguration of a user's environment;
- model license questions and the consequences of running untrusted model code.

## Trust boundaries

- The native runtime processes local models, prompts and hardware metadata. Its only HTTP
  surface is the explicit `serve` command, which binds to 127.0.0.1 (loopback only, no
  remote interface); it does not transmit prompts or telemetry anywhere.
- `tools/import_hf_model.py` is a separate opt-in network tool. It downloads only explicitly
  pinned HTTPS artifacts, checks declared sizes and hashes, preserves free-disk headroom and
  publishes an immutable checkpoint atomically.
- Runnable `.sgm` manifests are fully SHA-256 verified by default. Direct `.sgtm` input and
  `--no-verify` are controlled-development modes and weaken identity verification.
- Original model weights are immutable. Derived data belongs in reconstructable cache or a
  separately validated, atomically published artifact.
- The watchdog observes by default. Process termination requires explicit `--enforce` plus
  matching PID and heartbeat checks.

## Handling of credentials

This project never asks for credentials in issues, pull requests or logs. Do not paste API keys,
tokens, cookies or personal data in a report. Redact secrets from logs before attaching them.

Vastloom cannot make untrusted model licenses safe, guarantee that a host has adequate
cooling, or prevent all consequences of running experimental native code. Keep backups,
start with the safe profile and review the plan before enabling large-model execution.
