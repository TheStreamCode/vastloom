# Security policy

Stargate `0.2.0-alpha.2` processes local model files, prompts and hardware metadata. It
has no network download path or HTTP inference server.

Please report security issues privately to `dev@mikesoft.it`. Do not open a public issue
for path traversal, arbitrary file access, unsafe process termination, corruption of
model artifacts, malformed-model crashes or resource-exhaustion vulnerabilities.

## Model trust

Use a `.sgm` manifest for runnable models. `stargate chat`, `model-info` and `self-test`
perform full SHA-256 verification by default before loading the referenced weight file.

A direct `.sgtm` path receives strict structural checks but cannot prove that the file is
the intended model. `--no-verify` also downgrades a manifest to quick path and file-size
validation. Both modes are intended only for controlled local development.

## Local prompt handling

The TinyChat CLI performs inference entirely in the local process. It does not transmit
prompts, outputs or telemetry. Future opt-in network or mesh modes must document their
privacy boundary separately and must not be presented as anonymous without a formal
threat model.

## Watchdog

The watchdog defaults to observe-only mode. It can terminate a process only when
`--enforce` is supplied explicitly and the PID/heartbeat checks match.

Any future inference server must bind to loopback by default, require explicit
configuration before accepting remote traffic and apply request-level resource limits.
