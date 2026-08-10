# M9 DeepSeek bound trunk-load validation

Validation date: 2026-08-10

Implementation commit: `3652d709400a54de3484ff291ae175e08c38be04`

## Scope

This milestone connects one exact local range from a verified
`vastloom.range-artifact.v1` trunk binding to the Resource Governor and an existing scalar
storage consumer:

```text
hash-verified partial artifact + exact pinned-catalog binding
  -> validate immutable local range identity and NVME residency
  -> reserve complete RAM destination plus staging under the aggregate host cap
  -> bounded exact read
  -> SHA-256 the in-memory bytes again
  -> publish RAM_READY and retain a use lease
  -> transactional scalar F32 matvec
  -> release lease and evict cleanly
```

This proves one bounded range transaction and one scalar consumer. It does not make the partial
artifact, checkpoint or DeepSeek model executable. No official checkpoint or partial payload was
downloaded.

## Implemented

The model-independent `load_bound_range` primitive now:

- accepts only a non-empty logical identity, regular non-symlink file, valid SHA-256 and non-empty
  in-file range;
- applies a caller-lowerable range bound with an absolute 1 GiB ceiling;
- requires the supplied residency object to match the logical ID, exact byte count and `NVME`
  state;
- reserves the complete RAM destination and equally bounded staging before allocating or reading;
- reads exactly the declared span into one destination vector without an additional payload copy;
- hashes the bytes in memory after the read and before `RAM_READY` publication, detecting a
  same-size mutation after artifact verification;
- retains the RAM reservation through a shared immutable buffer/use lease;
- reports logical bytes, elapsed time and optional process/device I/O samples without converting
  them into a physical-NVMe claim;
- converts cancellation callback failures, allocation failures, short reads, corruption and late
  cancellation into controlled errors with reservation cleanup.

The DeepSeek wrapper accepts only an exact, fully hash-verified, explicitly non-executing trunk
binding. It gives every selected range a deterministic logical identity before invoking the
generic loader.

## Synthetic correctness and fault coverage

The native fixture contains a split ten-byte attention tensor and one complete four-byte F32
normalization tensor across four local artifact ranges. The complete F32 range encodes `1.5`; the
governed loader retains it while the transactional `1 x 1` matvec consumes input `2.0` and returns
exactly `3.0`.

The fixture verifies:

- an eight-byte aggregate host budget covers exactly four bytes of RAM and four bytes of staging;
- the staging reservation is released at `RAM_READY` while the RAM/use lease remains live;
- eviction is rejected while a retained buffer exists and succeeds after release;
- a seven-byte aggregate host budget fails before allocation with zero residual reservations;
- a caller bound above the absolute 1 GiB cap is rejected;
- wrong logical residency identity is rejected;
- early cancellation and a throwing callback leave the object in `NVME` with no reservation;
- late cancellation moves to `FAILED` and releases RAM/staging;
- same-size corruption after binding is caught by the second SHA-256 pass and releases resources;
- a truncated payload is rejected before the residency transition.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM and NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code and no official DeepSeek payload were executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors, real metadata dry-run | 15/15 passed in 13.16 s |
| GCC Debug, warnings as errors, real metadata dry-run | 15/15 passed in 28.70 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 70.28 s |
| GCC Release with existing SmolLM2/OLMoE opt-in regressions | 18/18 passed in 207.00 s |
| Clang 18 Release, warnings as errors | 11/11 passed in 4.98 s |
| GCC Debug + libstdc++ debug iterators/assertions | 11/11 passed in 8.63 s |
| MSVC Release x64, warnings as errors | 11/11 passed in 5.49 s |
| MSVC Debug x64, warnings as errors | 11/11 passed in 8.64 s |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python importer suite and `compileall` | passed |

After adding the throwing-callback fault case, the dedicated native artifact/loader test was
rebuilt and passed again under every compiler/configuration in the table.

ASan used leak detection and halt-on-error; UBSan halted on the first finding. ThreadSanitizer
binaries compiled, but the dedicated executable aborted before `main` with
`FATAL: ThreadSanitizer: unexpected memory mapping` in this WSL session. TSan is therefore an
environmental non-result, not listed as passing and not evidence of a project race.

## Performance statement

No model throughput, latency, bytes/token, physical-device traffic or speedup is reported. The
scalar transaction reads and consumes only four synthetic bytes. The existing 138,867,160-byte
official layer-0 trunk remains no-network dry-run evidence and was not acquired.

## Known limitations

- Only one complete F32 fixture range reaches a scalar consumer; the other bound ranges are not
  assembled into architecture tensor descriptors.
- The local artifact remains a partial, non-executing projection. It does not verify the complete
  source shard or checkpoint.
- The official hash-routing table has not executed, so selected-expert artifact scopes remain
  unauthorized.
- No official trunk payload has been acquired, decoded or executed.
- Full DeepSeek layer/token execution, generation, CUDA, pinned host buffers and useful
  NVMe/compute overlap remain absent.

## Next bottleneck

Build a bounded retained tensor-set assembler that maps every required bound trunk range into the
typed DeepSeek descriptors and executes the complete reduced graph from the artifact boundary.
It must reject missing, duplicate, split/incomplete or wrong-type tensors transactionally. Only
after that synthetic artifact-to-graph gate passes should the user be asked to authorize the real
138,867,160-byte layer-0 trunk transfer.
