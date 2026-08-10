# M9 governed packed-expert load validation

Validation date: 2026-08-10

Implementation commit: `55a8c5e`

## Scope

This slice connects the universal expert bundle and residency control plane to the first physical
load transaction:

```text
validated ExpertBundle
  -> reserve complete RAM destination + largest bounded range scratch
  -> READING
  -> read canonical immutable source ranges
  -> compact in range order
  -> expose tensor_index -> packed offset/length slices
  -> destroy scratch
  -> RAM_READY + RAII use lease
```

The test source is a tiny temporary binary fixture that remains unchanged during the load. The
pinned DeepSeek header catalog validates the same bundle/slice structure but has no weight
payload, so no DeepSeek expert was read or executed.

## Implemented path

The logical catalog now records a packed slice for every expert tensor. Each slice names its
catalog tensor index and exact offset/length inside the bundle's compacted buffer. Finalization
requires every expert tensor to fall wholly within one coalesced source range and requires packed
bytes to equal the bundle identity.

The new synchronous reference loader:

- rejects empty, oversized, overflowing, overlapping, adjacent/uncoalesced or non-canonical
  ranges before starting work;
- requires non-overlapping tensor slices that uniquely cover the complete packed payload;
- reserves the complete destination and the largest individual range before allocating either;
- uses the existing Unicode-native Windows buffered range reader and bounded Linux buffered path;
- checks cancellation before each range and after the final read;
- compacts multiple ranges into one read-only buffer and exposes bounded per-tensor spans;
- destroys the scratch buffer before publishing `RAM_READY` and releasing staging capacity;
- holds a move-only use lease for as long as the packed bytes exist, preventing eviction;
- publishes `FAILED` and releases all reservations on bounds, I/O, cancellation or allocation
  failure;
- records logical bytes, range count, elapsed load time and optional process/device deltas without
  conflating their meanings.

The public loader defaults remain capped at 512 MiB per expert, 128 MiB per range and 64 ranges.
Architecture adapters may set tighter bounds.

## DeepSeek structural result

The pinned preview metadata/header integration still verifies:

- 11,264 main/MTP expert bundles;
- 6 tensors and now 6 packed tensor slices per bundle;
- 13,369,344 logical bytes per bundle;
- exactly 2 coalesced source ranges per bundle.

These are verified addresses and packed layouts only. The payload shards are absent and
`execution_supported` remains `false`.

## Tests

The native fixture reads two non-adjacent ranges from a Unicode path, verifies the ten resulting
bytes in compacted order, resolves both tensor slices and confirms that an unknown tensor is
rejected. It also covers limit refusal before state mutation, adjacent/uncoalesced range rejection,
out-of-file bounds, cancellation, `FAILED` cleanup, use-lease eviction refusal and final zero
Governor reservations.

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors | 11/11 passed in 80.14 s |
| MSVC 19.44 Debug, warnings as errors | 10/10 passed in 107.83 s |
| GCC 13.3 Release, warnings as errors | 8/8 passed in 3.52 s |
| Clang 18.1 Release, warnings as errors | 8/8 passed in 3.51 s |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 8/8 passed in 10.32 s |
| GCC ThreadSanitizer native suite | passed with per-process ASLR disabled |
| Pinned DeepSeek metadata/header integration | 1/1 passed in 4.05 s |

The normal WSL ThreadSanitizer launcher retains the previously recorded early-runtime mapping
conflict. `setarch x86_64 -R` was applied only to the TSan process; the native suite returned
`ALL VASTLOOM TESTS PASSED` with no finding. ASan used leak detection and halt-on-error; UBSan and
TSan halted on their first finding.

Python byte compilation, manifest tests (one pass and one Windows symlink-capability skip), the
offline Hugging Face importer test and `git diff --check` also passed.

## Performance boundary

No performance result is reported. The fixture is 32 bytes and necessarily served from cache; its
elapsed time cannot represent NVMe, DeepSeek or token throughput. Process and device counters are
implemented as optional transaction telemetry but were not used to make a physical-I/O claim.

## Known limitations and next bottleneck

- The loader is synchronous and reads ranges serially.
- Buffered mode can hit the operating-system cache; direct/unbuffered mode is not connected.
- The scratch-to-destination copy is a correctness baseline, not a zero-copy path.
- No per-bundle content checksum exists beyond the already verified immutable checkpoint identity.
- The loader is not connected to OLMoE inference or a DeepSeek payload checkpoint.
- Packed I8/FP8/FP4 decoding and CPU/CUDA kernels remain absent.
- Cancellation is checked between bounded ranges, not inside one operating-system read.

The next bottleneck is an asynchronous multi-range transaction that owns preallocated destination
spans, overlaps the next expert read with current expert compute, supports the existing
direct/unbuffered measurement protocol and reports logical, process and physical device bytes per
accepted token separately.
