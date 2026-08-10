# Safety model

Vastloom must fail by ending or rejecting a generation, never by exhausting the host.

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

Logical tensor/expert residency uses the same rule. `begin_read` reserves the full RAM
destination and bounded staging capacity before publishing work; `begin_vram_copy` reserves
VRAM while the RAM source remains valid. A failed second reservation rolls back without changing
state. Pins and counted RAII use leases reject eviction, and failure cleanup refuses to release a
buffer still in active use. A separate exclusive transfer lease protects `READING` and
`COPYING_TO_VRAM` work from a concurrent failure, eviction or second transfer. Residency handles
and leases must never outlive their governor.

The shared packed-expert loader allocates only after those reservations. It caps expert bytes,
individual range bytes and range count; rejects non-canonical ranges and incomplete/overlapping
tensor slices; destroys scratch before releasing staging; and keeps its use lease alive until the
packed bytes are destroyed. Short reads, bounds failures and cancellation publish `FAILED` only
after temporary buffers have been released.

The shared asynchronous expert pool keeps one to four fixed workers and one to 64 queued
requests. Each request reserves its complete destination and bounded staging before enqueue. A
second capacity check rolls those reservations back if another submitter filled the queue or
shutdown began. Shutdown fails queued requests, completes their futures, releases their
reservations and joins any already-running bounded transaction.

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

## DeepSeek metadata and large-download interlocks

The pinned DeepSeek V4 Flash preview inspection path is intentionally non-executing:

- metadata-only import selects five ordinary metadata files and 46 header snapshots,
  totaling exactly 19,404,391 bytes;
- every header request is bounded to its pinned prefix, must return HTTP `206` with the
  exact source-file `Content-Range`, and must match a separately pinned SHA-256;
- a server that ignores or widens the range is rejected before its response can be
  accepted as an artifact;
- downloads retain the existing 1 MiB copy buffer, rate limit, partial-resume validation,
  disk reserve, lock, temporary publication and read-only immutable target;
- `deepseek-inspect` caps one header at 1 MiB and 2,000 tensors, validates all data ranges
  against the declared original shard size and refuses payload reads from snapshots;
- the CLI always reports `execution_supported=false`; it performs no model-weight, KV,
  activation or VRAM allocation;
- selecting a checkpoint with at least 20 GiB of weight files requires the separate
  `--allow-large-download` opt-in in addition to storage headroom checks.

The opt-in authorizes only an immutable download. It does not imply that the runtime can
execute the checkpoint, and it does not relax the 300 GiB default free-space reserve.
The newer official 0731 release, Unsloth GGUF variants and Laguna S 2.1 have separate
identities and no executable import authorization. A user-supplied URL never bypasses revision
pinning, license review, exact size/hash catalogs or the large-download interlock.

The 0731 GGUF reader adds a metadata-only safety boundary:

- one header is capped at 256 MiB, with explicit entry, tensor, string, array, dimension,
  nesting and alignment ceilings;
- split prefixes require the declared complete shard size, and every tensor offset/encoded span
  is checked against that immutable declared boundary;
- symlinks, malformed values, duplicate keys/tensors, invalid block shapes, offset collisions,
  nonzero padding, truncation and inconsistent split numbering fail closed;
- `deepseek-gguf-inspect` never reads tensor payloads and always publishes a `metadata_only`
  catalog;
- a header-prefix path cannot be used as an executable tensor source. A library caller must
  explicitly supply separate full-shard paths, after which normal immutable import and checksum
  gates are still required before execution;
- the scalar IQ1_S oracle accepts caller-supplied bounded spans only, prevalidates dimensions,
  encoded size, all block scales and finite inputs, and has a 64-million-element default ceiling;
- the oracle is not registered as a GGUF/catalog execution backend, so metadata snapshots and
  unverified full shards still cannot reach the token path.

The local validation used one complete 5.26 MB metadata shard and two 16 MiB HTTP-range prefixes;
it did not download the 82.54 GB profile. A separate exact 1.64 MB HTTP range was used for the
real IQ1_S parity test and remains outside the immutable model store. Header-derived logical
bytes/token are planning data, not physical device telemetry and never relax Resource Governor
limits.

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
- asynchronous whole mode has exactly one worker and one 12 MiB range per miss; staged
  mode has exactly two workers and validated 8 MiB gate/up plus 4 MiB down ranges;
- both asynchronous modes accept prefetch depth one or two, cap the queue at four
  requests, limit every request to at most 12 MiB and reserve every outstanding buffer
  against the staging cap before submission;
- cache-bypass mode is opt-in, requires async plus no-cache, and reserves twice the
  logical request plus a 128 KiB alignment allowance before allocating the output and
  direct-I/O envelope;
- unaligned tensor ranges are expanded only to the surrounding bounded sectors; the
  immutable source remains read-only and only the requested bytes reach expert compute;
- Windows asynchronous range reads use a read-only Unicode-native file handle, an exact
  validated offset and a loop bounded by the already reserved request size;
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
the default. Opt-in asynchronous mode uses either one validated contiguous expert range
or the checkpoint's contiguous gate/up region plus separate adjacent down region. Both
fail closed on worker failure, short read, allocation failure, cancellation or timeout.
Buffered mode remains the default. The `direct-unbuffered-v1` measurement protocol uses
`FILE_FLAG_NO_BUFFERING` or `O_DIRECT`; it never changes global cache limits or flushes
the host file cache.

Process-I/O sampling is observability only and cannot relax a reservation or safety
limit. Failure to obtain a counter never fails otherwise-correct inference; the CLI
reports the counter as unavailable. Windows transfer bytes are explicitly not treated
as physical NVMe reads, so a warm file-cache result cannot influence future placement
as though it were a device measurement. Device-level sampling is separate and retains
its exact source, device identity and scope. Cache-bypass mode is stricter: it refuses to
start without a trustworthy mapping and rejects a run whose device bytes are below the
logical expert payload. Whole-device counters can still include unrelated host traffic.

## Watchdog

`vastloom-watchdog` is a separate process. It validates that the heartbeat belongs to the
expected PID and is recent. Observe-only mode is the default. `--enforce` must be
supplied explicitly before it can terminate a stale runtime.

The alpha.2 TinyChat CLI does not yet publish a continuous generation heartbeat; that
integration remains required before long-running large-model inference.

## Tests implemented through alpha.4

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
- staged OLMoE parity, exact 8 MiB/4 MiB accounting, bounded two-worker overlap, LRU
  insertion/reuse and injected gate/up short-read refusal;
- 19-field readiness traces and bounded fixed-route versus ready-order replay with
  contiguous route/reduction-order validation;
- deterministic worker-failure, short-read, allocation-failure, cancellation and
  request-timeout injection through both the bounded I/O unit path and real OLMoE CLI.
- Unicode Windows shard-path reads, monotonic process-I/O deltas and platform-specific
  storage-byte availability semantics;
- Windows `PhysicalDrive` and native-Linux block-device mapping, monotonic device deltas,
  explicit unavailable behavior and rejection of virtual Linux block devices;
- aligned cache-bypass reads for aligned and unaligned ranges, doubled staging
  reservation, full-model output parity and physical-byte coverage enforcement;
- DeepSeek exact-revision identity, bounded HTTP header ranges, 46 verified header
  snapshots, 69,187 tensor/shard mappings, FP8 dtype parsing and large-download refusal;
- regression protection against fragmented Windows expert reads and exact reuse-distance
  results for synthetic and real repeated-route traces.
- legal NVMe/RAM/VRAM residency transitions, atomic multi-resource rollback, pressure refusal,
  concurrent read admission, pin/use-lease eviction guards and final reservation release.
- Unicode buffered multi-range expert loading, exact packed tensor slices, pre-allocation limits,
  out-of-bounds read cleanup, cancellation and zero residual reservations.
- exclusive transfer leases, deterministic asynchronous queue saturation, reservation rollback,
  queued-request shutdown cancellation and ThreadSanitizer coverage.
- bounded GGUF v3 header/split parsing, packed-expert slicing, exact quantized byte spans,
  Unicode Windows paths, malformed-header rejection and non-executable snapshot enforcement.

## Safety tests still required for production adapters

- live low-RAM, low-VRAM and disk-pressure monitoring during generation;
- process kill during model conversion/publication;
- queue saturation under real multi-GB reads;
- stale heartbeat and wrong-PID enforcement on all supported operating systems;
- CUDA allocation failure, device loss and reset recovery;
- cancellation while transfers and kernels are in flight;
- NVMe thermal/power policy transitions on real hardware;
- fuzzing of Safetensors, tokenizer and model-adapter metadata.
