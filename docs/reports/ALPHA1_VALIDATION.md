# Alpha 1 validation report

Validation date: 2026-08-08

The project was named Stargate Local AI when this report was written; the commands below are
recorded as they were run. The binary is now `vastloom`, so `stargate <command>` reads
`vastloom <command>` today. Nothing else in this snapshot has been rewritten.

## Builds

- GCC 14.2, C++20, Debug: passed.
- GCC 14.2, Release with all project warnings promoted to errors: passed.
- GCC 14.2, Debug with AddressSanitizer and UndefinedBehaviorSanitizer: passed.
- Windows build is defined in CI but was not executed inside the Linux development container.

## Automated tests

C++ tests passed for:

- byte parsing and overflow rejection;
- SHA-256 known vector;
- strict manifest parsing;
- quick and full model verification;
- deliberate hash mismatch detection;
- `..` path rejection;
- symlink escape detection where supported;
- resource oversubscription rejection and reservation release;
- emergency safety escalation and hysteretic recovery;
- reference-PC planning and hard headroom;
- bounded asynchronous reads;
- out-of-file reads and oversized requests;
- MoE trace simulation;
- atomic heartbeat write/read.

Python offline-tool tests passed for:

- exact manifest generation and hash publication;
- cleanup/no leaked temporary manifest;
- file-symlink rejection.

## Manual CLI smoke tests

Passed:

```text
stargate doctor
stargate inspect
stargate verify --full
stargate plan --reference-pc
stargate trace-sim --policy lru
stargate trace-sim --policy static-hot
stargate io-smoke
stargate store-init
stargate heartbeat-write
```

JSON output from `doctor`, `inspect`, `plan` and `trace-sim` was parsed successfully by
Python's standard `json` module.

## Explicitly not validated yet

- CUDA compilation or execution on an RTX 4070 Super;
- real DeepSeek V4 Flash checkpoint inspection;
- token/logit correctness;
- direct I/O, pinned memory or GPU copy overlap;
- Windows runtime behavior;
- thermal telemetry/control on the reference PC;
- end-to-end tokens per second.

Those items begin at M2–M5 and must not be inferred from this foundation release.
