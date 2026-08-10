# M9 bounded asynchronous expert-load validation

Validation date: 2026-08-10

Implementation commit: `2e4592a`

## Scope

This slice turns the shared governed packed loader into a bounded asynchronous transaction:

```text
validated ExpertBundle
  -> reserve complete RAM destination + largest-range staging
  -> acquire one exclusive transfer lease
  -> enqueue within a fixed-capacity pool
  -> read and compact canonical immutable ranges on a worker
  -> RAM_READY + use lease + future result
```

This is the model-independent mechanism needed to submit expert N+1 while a caller computes with
expert N. The test payload is a 32-byte temporary fixture. No DeepSeek or Laguna weight payload
was downloaded, read or executed, and no compute/I/O speedup is claimed.

## Implemented path

`BoundedExpertLoadPool` provides:

- one to four fixed workers and a queue capped between one and 64 requests;
- complete RAM-destination and largest-range staging reservations before a request can enter the
  queue;
- a second queue-capacity check after reservation, with complete rollback if another submitter
  filled the queue or shutdown began;
- one exclusive move-only transfer lease per in-flight residency object, preventing failure,
  eviction or VRAM-copy admission while a worker still owns its buffers;
- the same range, slice, cancellation, short-read and allocation checks as the synchronous oracle;
- monotonic request sequence plus queue, service and total elapsed-time telemetry;
- deterministic shutdown: queued work becomes `FAILED`, promises are completed and all
  reservations are released, while an already running bounded transaction is joined;
- constructor cleanup that wakes and joins already-created workers if later worker construction
  fails.

The queue bound counts waiting requests. In-flight work is additionally bounded by the fixed
worker count, and every admitted request remains independently constrained by the Resource
Governor. A returned `LoadedExpert` keeps its RAM reservation and use lease until its packed bytes
are destroyed.

## Tests

The native tests intentionally block a single worker inside the cancellation boundary, then
verify:

- `READING`, one active transfer and pre-existing RAM/staging reservations while the worker is
  blocked;
- rejection of a second transfer and refusal to fail/release an active transaction;
- exactly one queued request and fail-closed rejection of the next request without residency
  mutation or reservation leakage;
- ordered request IDs and non-negative queue/service/total timings;
- transition to `RAM_READY`, zero transfer leases and one use lease on completion;
- eviction only after the returned packed buffers are destroyed;
- shutdown cancellation of queued work while safely joining the running request;
- final zero RAM and staging reservations;
- Unicode multi-range byte and tensor-slice parity with the synchronous loader.

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors | 11/11 passed in 78.36 s |
| MSVC 19.44 Debug, warnings as errors | 10/10 passed in 103.24 s |
| GCC 13.3 Release, warnings as errors | 8/8 passed in 3.50 s |
| Clang 18.1 Release, warnings as errors | 8/8 passed in 3.45 s |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 8/8 passed in 10.99 s |
| GCC ThreadSanitizer native suite | passed with per-process ASLR disabled |
| Pinned DeepSeek metadata/header integration | 1/1 passed in 3.52 s |

The normal WSL ThreadSanitizer launcher retains the known early-runtime mapping conflict.
`setarch x86_64 -R` changed ASLR only for the TSan process; the native suite returned
`ALL VASTLOOM TESTS PASSED` with no finding. ASan used leak detection and halt-on-error; UBSan and
TSan halted on their first finding.

Python byte compilation, the manifest tests (one pass and one Windows symlink-capability skip),
the offline Hugging Face importer test and `git diff --check` also passed.

## Performance boundary

This slice proves bounded admission and overlap capability, not useful overlap. The fixture is too
small and cache-resident to represent NVMe latency, queue depth, DeepSeek bytes/token or end-to-end
throughput. `queue_ms`, `service_ms` and `total_ms` are transaction telemetry only. Process-wide
I/O deltas may include concurrent requests, and device deltas retain the documented platform and
whole-device scope.

## Known limitations and next bottleneck

- The shared loader still uses buffered synchronous operating-system reads inside each worker.
- Ranges within one expert are read serially and copied through one bounded scratch buffer.
- No model execution loop consumes one future while computing another expert yet.
- The shared loader is not connected to direct/unbuffered I/O, pinned host memory or CUDA copies.
- No cache manager deduplicates or reuses completed expert bundles.
- No DeepSeek 0731 GGUF reader, quantized kernel or full-checkpoint path exists.
- Cancellation is checked between bounded reads, not inside one blocking operating-system read.

The next gate is a deterministic compute/I/O overlap harness on realistically sized immutable
ranges, followed by the official 0731 GGUF catalog and quantization-specific CPU parity path.
