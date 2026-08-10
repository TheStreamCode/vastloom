# M9 DeepSeek asynchronous selected-expert overlap validation

Validation date: 2026-08-10
Implementation commit: `322f863243c03d7b538a5ef3081f5b19cf4cd07b`

## Scope

This milestone connects the existing bounded fixed-worker loader to the router-first DeepSeek
expert cache and reference graph:

```text
authoritative route
  -> deduplicate selected complete experts
  -> reserve RAM destination and bounded staging for every miss
  -> submit all misses to a fixed-worker, fixed-capacity queue
  -> acquire expert N only at its numerical compute turn
  -> load expert N+1 while the CPU computes expert N
  -> reduce expert outputs in the original deterministic order
  -> retain hits and publish exact load/wait/compute/overlap telemetry
  -> drain abandoned work and release every reservation
```

The deterministic reduced graph proves that read service can overlap useful expert computation
without changing logits, routes, attention indices or state. A separate opt-in test sends the
real 6,488,064-byte DeepSeek 0731 layer-0/expert-0 IQ1_S/IQ3_XXS bundle through the same
asynchronous batch/acquire/cache/kernel boundary. Because that real test contains only one expert,
it proves the future and ownership path but is not an overlap or storage-performance benchmark.

The complete 82.54 GB checkpoint was not downloaded or executed. This milestone does not claim a
complete official layer/token, generation, CUDA, direct-I/O or physical-NVMe speed result.

## Implemented vertical slice

- The provider contract retains its synchronous `resolve` baseline and adds an optional bounded
  `begin`/`acquire` batch. Existing adapters therefore do not silently change behavior.
- Asynchronous caches accept one to four fixed workers. Queue capacity must be large enough for
  the complete maximum selected set and remains capped at 64 requests.
- Only one asynchronous batch may own a cache in this correctness slice. Concurrent batches,
  synchronous resolution and cache clearing fail closed until the owner releases it.
- The batch validates block identity, request size, unique expert IDs, LRU/cache invariants and
  telemetry bounds before submission. It evicts only safe cold entries and submits misses in the
  same numeric order used by the reference compute loop.
- Every miss creates an indivisible complete-expert residency, reserves its RAM destination and
  largest bounded staging range before enqueue, and keeps its transfer/use leases alive through
  publication and compute.
- `acquire(expert_id)` waits only for that expert's future, validates its canonical W1/W2/W3
  descriptors and installs it in the RAM cache. Cache hits return immediately with zero read
  timing and zero logical bytes.
- The graph preserves authoritative accumulation order while recording request sequence,
  queue/service/total load time, foreground acquire wait, expert compute time and the exact
  intersection of earlier compute windows with later read-service intervals.
- Batch destruction drains unacquired futures, evicts unpublished RAM, releases reservations and
  records successful unused reads as abandoned loads and wasted-prefetch bytes. Telemetry
  overflow permanently fails the cache closed instead of wrapping counters.
- Provider construction, batch creation and acquisition contain allocation, standard and
  arbitrary exceptions and return typed errors without publishing partial graph output.

## Deterministic overlap result

The reduced packed fixture uses one worker and a queue of two. Its existing cancellation callback
acts as a deterministic barrier: the second load cannot complete until the first miss has been
installed, then its bounded service interval is held for 20 ms while the first expert computes.
The result:

- matches the independently dequantized packed oracle exactly;
- keeps asynchronous request sequences positive and increasing;
- reports non-negative queue, service, total, acquire-wait and compute times;
- reports a strictly positive read-service/previous-expert-compute intersection;
- passes 20 consecutive native executions after the final code change.

The barrier is test-only and uses the loader's existing cancellation hook. No sleep, artificial
delay or benchmark hook was added to production code. This proves scheduler semantics, not useful
overlap on the reference NVMe device.

## Real selected-expert result

The separately hash-pinned real bundle remains:

| Matrix | Format | Shape | Encoded bytes |
|---|---|---:|---:|
| gate / W1 | IQ1_S | 2,048 x 4,096 | 1,638,400 |
| up / W3 | IQ1_S | 2,048 x 4,096 | 1,638,400 |
| down / W2 | IQ3_XXS | 4,096 x 2,048 | 3,211,264 |
| **Complete expert** | mixed | indivisible bundle | **6,488,064** |

A one-worker/one-request asynchronous batch loads the complete expert, returns a positive request
sequence and valid absolute service/completion timestamps, binds the canonical matrices and
matches the pinned ggml output fingerprint `90388543cbcbab52`. Cache clearing is rejected during
the batch; after release, all RAM, staging and scratch reservations return to zero.

## Safety and fault coverage

- zero workers preserve the synchronous path and require zero asynchronous queue capacity;
- invalid worker/queue combinations and a queue smaller than the maximum selected set are
  rejected at construction;
- insufficient aggregate host RAM rejects a batch before allocation or I/O;
- a second batch, synchronous resolution and clear are rejected while one batch owns the cache;
- cancellation, short/failed results, null batches, malformed identities/timing and provider
  exceptions return errors without partial output or reservation leakage;
- abandoning a two-miss batch drains both requests, records two abandoned loads and their exact
  bytes as wasted prefetch, and releases all residency;
- cache hits report no stale asynchronous timing;
- one complete expert remains wholly in RAM; RAM and VRAM are never combined and no generic swap
  or implicit precision downgrade is introduced.

## Validation host

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32,624,488 KiB visible Windows RAM; 11,691,656 KiB free at the recorded probe;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported VRAM and 8,966 MiB free at the probe,
  unused by these CPU tests;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code was compiled or executed.

## Exact results

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors | 10/10 passed in 10.41 s |
| GCC Debug, warnings as errors | 10/10 passed in 23.12 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 10/10 passed in 60.98 s |
| GCC ThreadSanitizer native suite, per-process ASLR disabled | passed |
| GCC libstdc++ debug iterators/assertions | native suite passed |
| Clang 18 Release, warnings as errors | 9/9 passed in 3.97 s |
| MSVC Release x64, warnings as errors | 9/9 passed in 4.54 s |
| MSVC Debug x64, warnings as errors | 9/9 passed in 6.98 s |
| Real-expert opt-in: GCC Release | 1/1 passed in 0.42 s |
| Real-expert opt-in: GCC ASan + UBSan | 1/1 passed in 1.54 s |
| Real-expert opt-in: Clang Release | 1/1 passed in 0.26 s |
| Real-expert opt-in: MSVC Release x64 | 1/1 passed in 0.23 s |
| Real-expert opt-in: MSVC Debug x64 | 1/1 passed in 0.94 s |
| Repeated native overlap regression | 20/20 passed |
| Serial real-model regression subset | 5/5 passed in 199.19 s |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python offline importer, `compileall` and `git diff --check` | passed |

ASan used leak detection and halt-on-error; UBSan and TSan halted on the first finding. TSan used
`setarch x86_64 -R` only for its instrumented process. MSVC host and target architecture were
explicitly x64. The real-expert tests used already-present small extracted ranges; no large model
was downloaded.

The serial real-model regression used already-present verified local artifacts: SmolLM2 passed in
29.92 seconds, OLMoE inspection in 1.22 seconds, OLMoE CPU runtime in 161.71 seconds, DeepSeek
metadata inspection in 5.87 seconds and the synthetic full-checkpoint binding gate in 0.44
seconds.

## Performance statement

No token rate, TTFT, SSD bandwidth, physical bytes/token or end-to-end speedup is reported. The
strictly positive overlap comes from a controlled reduced fixture and establishes event ordering.
The real one-expert path cannot overlap two experts, and its files may be served by the OS cache.
Logical bytes, wasted-prefetch bytes and timestamp intersections are correctness telemetry, not a
substitute for physical-device measurement under a declared cache state.

## Known limitations

- Only one asynchronous batch may use a cache at once; concurrent generations over that cache
  fail closed.
- The batch drains abandoned work rather than cancelling each submitted request independently.
- Failed partial reads do not yet expose successful-byte telemetry, so wasted-prefetch accounting
  covers only loads that completed successfully.
- The shared pool still performs buffered blocking OS reads inside fixed workers.
- The official trunk payload is absent, so the real expert is not composed into a complete
  quantized layer/token.
- Expert residency is RAM-only; pinned host buffers, VRAM placement, CUDA copies and CUDA kernels
  do not exist.
- Scalar mixed-format kernels are correctness references, not SIMD/fused production kernels.

## Next bottleneck

Compose one complete mixed-format DeepSeek layer/token from bounded official-format trunk tiles
and selected real experts while preserving exact logits and routes. Only after that small bounded
probe passes should the project request approval to import and fully hash the official payload.
Physical-device bytes, cold/warm cache state and useful I/O/compute overlap must then be measured
before any throughput claim.
