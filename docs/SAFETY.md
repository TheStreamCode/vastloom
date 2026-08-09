# Safety model

Stargate must fail by ending or rejecting a generation, never by exhausting the host.

## Threats considered

- process or system OOM;
- VRAM exhaustion and driver instability;
- pagefile/swap storms that make the desktop unresponsive;
- disk-full conditions during download, conversion or cache writes;
- SSD thermal throttling and excessive avoidable writes;
- corrupt, partial or mismatched weight shards;
- malformed model headers, dimensions or tensor layouts;
- path traversal and symlink escape from a model root;
- queue growth and concurrent over-reservation;
- deadlocked runtime or stale process heartbeat;
- arithmetic overflow in byte, shape and offset accounting;
- unbounded prompts, generation lengths or execution time.

## Budget ownership

Only `ResourceGovernor` owns mutable capacity counters. Workers request a move-only RAII
reservation before allocation. The reservation remains alive for as long as the buffer
or model allocation is alive and is released automatically.

This prevents the classic race where multiple workers each observe the same free memory
and all allocate it.

A loaded model keeps a reservation that references its governor. The governor must
therefore outlive the model. The CLI encodes that lifetime in member declaration order
and regression tests verify that the reservation returns to zero when the model is
destroyed.

## Headroom

The planner creates separate reserves for:

- host OS and foreground applications;
- emergency RAM recovery;
- GPU driver/display allocations;
- minimum storage free space;
- bounded I/O staging and scratch.

The safety profile is conservative. Performance mode may use more resources but never
removes hard reserves.

## State machine

```text
NORMAL
  │ immediate escalation
  ▼
PRESSURE  → reduce prefetch/cache/speculation
  ▼
CRITICAL  → reject new work and drain current operations
  ▼
EMERGENCY → abort generation and release recoverable resources
```

Escalation is immediate. Recovery is one level at a time after multiple healthy samples,
preventing oscillation around a threshold.

## TinyTransformer execution bounds

Before allocating weights, the alpha.2 loader validates:

- maximum physical model size;
- fixed magic, version, flags and reserved fields;
- exact header, metadata and tensor byte counts;
- architecture dimensions and arithmetic overflow;
- special-token ranges and collisions;
- tensor-layout completeness;
- finite FP32 payload values;
- absence of trailing or truncated data.

Before generation it validates:

- a non-empty prompt that leaves at least one context slot;
- bounded maximum output tokens;
- finite temperature and legal top-k;
- a timeout between 1 ms and 1 hour;
- a Resource Governor reservation for the complete KV/workspace estimate.

The prompt is byte-tokenized, so its context cost can be checked exactly before building
large intermediate buffers.

## Model-store rules

```text
models/original/   immutable source weights
models/optimized/  separately generated and verified artifacts
cache/             reconstructable placement/cache data
scratch/           temporary bounded files
state/             manifests, receipts and heartbeat
telemetry/         local metrics only
```

Original weights must never be rewritten during inference. A converter or training tool
writes to a temporary path, flushes and verifies the artifact, then atomically publishes
the result. The TinyChat exporter follows this pattern for the model, test vectors and
manifest.

## Verification modes

- **Quick:** safe path resolution plus exact file-size checks.
- **Full:** quick checks plus SHA-256 for every chunk.

`chat`, `model-info` and `self-test` use full verification by default for `.sgm`
manifests. `--no-verify` explicitly downgrades to quick verification for development.
A direct `.sgtm` path receives structural validation but no cryptographic identity check.

A metadata-only research profile cannot verify weights and must never be treated as a
runnable model manifest.

## Real OLMoE streaming bounds

The alpha.4 OLMoE probe is fail-closed around one exact checkpoint revision:

- all three shard sizes and SHA-256 values, model identity, tensor names, shapes and
  BF16 dtypes must match the pinned manifest;
- a metadata-only import can be inspected/tokenized but is rejected before runtime
  weight allocation;
- the 13.84 GB checkpoint is never proposed as resident when it exceeds the live RAM
  cap; only the 953,421,824-byte always-active trunk is reserved and loaded;
- one streamed expert is exactly 12 MiB, and every no-cache or cache buffer is reserved
  before allocation;
- asynchronous mode has exactly one worker, accepts prefetch depth one or two, limits
  each request to 12 MiB and reserves every outstanding buffer against the staging cap;
- each asynchronous request has a positive timeout capped at 60 seconds and checks the
  same cancellation callback before and after its bounded read;
- cache capacity is rounded to complete experts, must hold at least one expert and is
  hard-capped at 2 GiB;
- FP32 K/V state and a 64 MiB activation scratch budget are reserved before forward;
- the Windows Job Object, six-thread maximum and 8 GiB live system-headroom check apply
  to the complete process; cancellation, timeout and live pressure are checked at every
  layer boundary;
- telemetry is written to a process-specific temporary file and atomically renamed;
  existing traces are never overwritten.

Original checkpoint files are read-only and remain outside the repository. Expert
traffic is produced by exact bounded reads, not pagefile/swap. Synchronous mode remains
the default. Opt-in asynchronous mode uses one validated contiguous expert range and
fails closed on worker failure, short read, allocation failure, cancellation or timeout.
It is still a buffered reference pipeline, not production direct I/O.

## Watchdog

`stargate-watchdog` is a separate process. It validates that the heartbeat belongs to the
expected PID and is recent. Observe-only mode is the default. `--enforce` must be
supplied explicitly before it can terminate a stale runtime.

The alpha.2 TinyChat CLI does not yet publish a continuous generation heartbeat; that
integration remains required before long-running large-model inference.

## Tests implemented through alpha.4.2

- allocation oversubscription and reservation release;
- critical/emergency state escalation and hysteretic recovery;
- exact SHA-256 and intentionally corrupted weights;
- path traversal and symlink escape;
- truncated model payload;
- invalid empty model id;
- non-finite tensor payload;
- model load under an insufficient RAM budget;
- bounded prompt, sampling and timeout options;
- bounded asynchronous short/out-of-range reads;
- seeded generation reproducibility;
- valid-Unicode JSON serialization for arbitrary sampled byte sequences;
- end-to-end CLI validation, direct/manifest loading modes and JSON error behavior.
- exact OLMoE tokenizer, top-logit, greedy-token, expert-route and routing-weight parity
  against a bounded Transformers reference;
- metadata-only runtime refusal, missing-shard rejection, one-millisecond timeout and
  invalid expert-cache policy rejection;
- real no-cache byte accounting, real two-token LRU hits and native-telemetry simulation
  for no-cache, LRU, LFU and static-hot baselines;
- real asynchronous OLMoE parity for logits and all 128 routed experts, bounded prefetch
  counts, telemetry parsing and positive read-service/CPU overlap;
- deterministic worker-failure, short-read, allocation-failure, cancellation and
  request-timeout injection through both the bounded I/O unit path and real OLMoE CLI.

## Safety tests still required for production adapters

- live low-RAM, low-VRAM and disk-pressure monitoring during generation;
- process kill during model conversion/publication;
- queue saturation under real multi-GB reads;
- stale heartbeat and wrong-PID enforcement on all supported operating systems;
- CUDA allocation failure, device loss and reset recovery;
- cancellation while transfers and kernels are in flight;
- NVMe thermal/power policy transitions on real hardware;
- fuzzing of Safetensors, tokenizer and model-adapter metadata.
