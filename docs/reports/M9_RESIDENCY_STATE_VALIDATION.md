# M9 governed multitier residency-state validation

Validation date: 2026-08-10

Implementation commit: `eeff002`

## Scope

This slice implements the shared control plane for moving one logical tensor or expert bundle
through the model-aware hierarchy:

```text
NVME -> READING -> RAM_READY -> COPYING_TO_VRAM -> VRAM_READY
  ^          |          |                              |
  |          +----------+----------- FAILED <---------+
  +--------------------- eviction --------------------+
```

It owns Resource Governor reservations and transition state only. It deliberately allocates no
weight buffer, submits no I/O, calls no CUDA API and does not make DeepSeek executable.

## Reservation and lifetime rules

- `begin_read` atomically obtains the complete RAM destination reservation and bounded staging
  reservation before publishing `READING`.
- If either reservation fails, the other is rolled back and the object remains `NVME`.
- `publish_ram_ready` releases staging only after the caller has completed and validated the read.
- `begin_vram_copy` reserves the complete VRAM destination while retaining the valid RAM source.
- A failed VRAM reservation leaves the object and its RAM reservation unchanged.
- `publish_vram_ready` can retain a warm RAM copy or release it explicitly.
- RAM/VRAM eviction is rejected while the object is pinned or has an active use lease.
- `ResidencyUse` is move-only RAII state. It retains the shared control object and prevents its
  reservation from disappearing even if the scheduler's original handle is destroyed.
- `fail` releases staging/RAM/VRAM reservations and records a reason, but refuses to free an
  object still in active use. `retry` is legal only after complete cleanup.
- The governor must outlive all residency handles and leases that reference its reservations,
  matching the existing loaded-model lifetime contract.

The factory can create one compact residency control object per finalized logical expert bundle
and revalidates aggregate expert bytes and source-range counts. It does not instantiate the
11,264-object DeepSeek table during ordinary header inspection.

## Tests

Native tests cover:

- every legal NVMe/RAM/VRAM transition and reservation total;
- illegal double reads, early publish, invalid staging sizes and use-before-ready;
- RAM rejection, staging rejection after successful RAM reservation and exact rollback;
- VRAM rejection with the prior RAM-ready object preserved;
- pin and RAII use-lease eviction guards;
- optional retention or eviction of the RAM copy after VRAM publication;
- failure cleanup and retry;
- governor `EMERGENCY` refusal without a state change;
- concurrent read submission through two shared handles, with exactly one winner;
- lease/object destruction order and final reservation release;
- expert-table construction from the generic logical catalog.

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors | 11/11 passed in 80.46 s |
| MSVC 19.44 Debug, warnings as errors | 10/10 passed in 109.01 s |
| GCC 13.3 Release, warnings as errors | 8/8 passed in 3.45 s |
| Clang 18.1 Release, warnings as errors | 8/8 passed in 3.33 s |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 8/8 passed in 11.43 s |
| GCC ThreadSanitizer native suite | passed with per-process ASLR disabled |
| Pinned DeepSeek metadata/header integration | 1/1 passed in 3.99 s |

The normal WSL ThreadSanitizer launch terminated before `main` with
`FATAL: ThreadSanitizer: unexpected memory mapping`. Re-running the same binary with
`setarch x86_64 -R` changed ASLR only for that process and returned `ALL VASTLOOM TESTS PASSED`.
No TSan finding was reported. ASan used leak detection and halt-on-error; UBSan and TSan halted
on their first finding.

Python byte compilation, the offline Hugging Face importer test and `git diff --check` also
passed.

## Performance and safety boundary

This state-only test does not touch an SSD or GPU and therefore records no bytes/token,
throughput, latency, overlap, temperature or power result. It verifies admission and lifecycle
rules, not data movement. No large model was downloaded or loaded.

## Known limitations and next bottleneck

- No buffer object is yet coupled to the residency reservation.
- The bounded I/O pool does not yet publish its result into a residency object.
- There is no checksum/content validation between `READING` and `RAM_READY` for a packed bundle.
- CUDA, pinned host memory and asynchronous RAM-to-VRAM copies are absent.
- Failure while a kernel holds a lease requires cancellation/worker coordination before cleanup;
  the controller correctly refuses premature release but does not initiate that cancellation.
- No cache manager yet chooses victims or reduces prefetch under `PRESSURE`.

The next bottleneck is a bounded packed-expert load transaction that reads the exact coalesced
catalog ranges into a reserved buffer, validates completion, publishes `RAM_READY` atomically and
records logical/process/device bytes separately.
