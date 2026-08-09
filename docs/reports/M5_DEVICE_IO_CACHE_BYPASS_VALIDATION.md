# M5 device-I/O and cache-bypass validation

Validation date: 2026-08-09  
Implementation commit: `3eeef39`

## Scope

This report closes the final M5 measurement gate for the pinned real OLMoE path:

- map a checkpoint path to a device-level read counter without relabeling process bytes;
- provide a repeatable operating-system cache-bypass protocol without flushing global
  host cache state;
- prove bounded allocation, exact output parity and physical-byte coverage on Windows;
- preserve explicit unavailable behavior on ambiguous or virtual storage mappings.

This is an OLMoE one-token/logit measurement. It is not DeepSeek inference, accepted-token
throughput, SSD firmware-cache control, CUDA execution or a production direct-I/O scheduler.

## Reference host

The final Windows probe reported:

- Microsoft Windows 11 Pro `10.0.26200`, build `26200`;
- AMD Ryzen 9 7900X, 24 logical CPUs;
- 33,407,475,712 bytes of visible RAM;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB total VRAM, driver `610.88`;
- Lexar SSD NM790 2TB, NTFS, exposed as `PhysicalDrive0`;
- 1,203,166,220,288 bytes free on the model volume at the recorded probe.

The GPU was detected but not used by the OLMoE CPU reference path.

## Device-counter semantics

Windows resolves the checkpoint's local drive-letter volume with
`IOCTL_STORAGE_GET_DEVICE_NUMBER`, opens the corresponding `PhysicalDriveN` read-only,
and samples `IOCTL_DISK_PERFORMANCE` immediately before and after the forward pass.
The JSON record retains:

- source: `windows-ioctl-disk-performance`;
- device identity: for this validation, `PhysicalDrive0`;
- scope: `whole-device`;
- read-operation and read-byte deltas.

The counter covers the complete device, not only Vastloom. Concurrent reads from other
processes can increase the delta, so it is an upper bound for the measured workload.

On native Linux, Vastloom maps `stat().st_dev` through `/sys/dev/block` and accepts the
counter only when the canonical sysfs path is non-virtual. Partition and whole-device
scopes remain distinct. WSL `/mnt/c`, overlay filesystems, device-mapper volumes and other
ambiguous mappings remain unavailable rather than being called physical storage.

## `direct-unbuffered-v1`

The cache-bypass protocol is opt-in through:

```powershell
vastloom moe-logits MODEL --text Hello `
  --cache no-cache --io async --io-cache bypass `
  --expert-read staged --prefetch-depth 2 --no-verify --json
```

It requires all of the following:

- asynchronous bounded I/O;
- the no-cache expert policy;
- an available device-level counter before execution;
- read-only `FILE_FLAG_NO_BUFFERING` on Windows or `O_DIRECT` on Linux;
- device read bytes at least equal to exact logical expert bytes;
- the same output parity gate as buffered execution.

Tensor offsets need not be sector-aligned. Each request expands to the smallest bounded
surrounding sector envelope and copies only the requested tensor bytes to expert compute.
Before allocating, the I/O pool reserves twice the logical request plus a 128 KiB alignment
allowance. The existing request-size, worker, queue, timeout and cancellation caps remain
in force.

The protocol never calls `SetSystemFileCacheSize`, never drops Linux global page cache and
never modifies the checkpoint. It bypasses the operating-system file cache; it cannot
guarantee that the SSD controller's internal cache is cold. Microsoft documents both the
no-buffering behavior and its buffer/offset alignment requirements in
[File buffering](https://learn.microsoft.com/windows/win32/fileio/file-buffering).

## Alignment finding

The first Windows prototype attempted each exact tensor offset directly and failed safely
with `ERROR_INVALID_PARAMETER` during the seek. The checkpoint's tensor offsets are not
all aligned to the volume's direct-I/O sector requirement. The accepted implementation
uses the bounded sector envelope described above. Unit tests cover both aligned and
unaligned ranges.

## Repeatability result

Three consecutive `direct-unbuffered-v1` runs used the same binary, checkpoint, one-token
prompt, no-cache policy, staged reader and prefetch depth two:

| Run | Greedy token | Forward ms | Logical expert bytes | Device operations | Whole-device bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | 47111 | 792.978 | 1,610,612,736 | 287 | 1,610,743,808 |
| 2 | 47111 | 780.328 | 1,610,612,736 | 287 | 1,610,743,808 |
| 3 | 47111 | 804.192 | 1,610,612,736 | 287 | 1,610,743,808 |

Median forward time was 792.978 ms. Every device delta exceeded the logical payload by
exactly 131,072 bytes, or 1 KiB per routed expert, due to bounded sector envelopes. The
byte overhead was approximately 0.0081%.

A final evidence run with the committed JSON protocol identifier reported:

| Field | Value |
|---|---:|
| `io_cache_protocol` | `direct-unbuffered-v1` |
| Greedy token | 47111 (`" Byr"`) |
| Forward time | 864.052 ms |
| Logical expert bytes | 1,610,612,736 |
| Process read operations | 256 |
| Process transfer bytes | 1,610,743,808 |
| Device identity/scope | `PhysicalDrive0` / `whole-device` |
| Device read operations | 287 |
| Device read bytes | 1,610,743,808 |

Greedy token, top logits, all 128 routes and routing weights matched the committed
Transformers reference through the Windows integration suite.

## Build and test matrix

| Toolchain/configuration | Result |
|---|---|
| MSVC Release, warnings as errors, external SmolLM2 and OLMoE | 8/8 passed in 75.98 s |
| MSVC Release, final full OLMoE target after protocol identifier | 1/1 passed in 62.57 s |
| MSVC Debug, warnings as errors, external SmolLM2 and OLMoE metadata | 7/7 passed in 108.58 s |
| GCC Release, external SmolLM2 and OLMoE | 8/8 passed in 201.33 s |
| Clang Release, warnings as errors | 5/5 passed in 2.75 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 5/5 passed in 8.13 s |

The GCC ThreadSanitizer build completed, but its test process terminated before `main`
with `FATAL: ThreadSanitizer: unexpected memory mapping`. This is a repeatable WSL/TSan
environmental limitation and is not listed as a passing result.

Tests cover monotonic and incompatible device snapshots, explicit unsupported mappings,
aligned and unaligned cache-bypass ranges, CLI option constraints, device source/scope,
full-model reference parity, and physical bytes not falling below logical bytes.

## Known limitations

- whole-device counters include unrelated host activity;
- SSD firmware/controller cache state is outside user-mode control;
- Linux device-level validation was not possible for the checkpoint stored on WSL `/mnt/c`;
- direct I/O currently copies from an aligned staging envelope into the output buffer;
- OLMoE still exposes final logits/routes rather than generated chat tokens;
- no pinned host memory, VRAM expert tier, CUDA backend or DeepSeek adapter exists yet.

## Next bottleneck

M5 is complete. The next milestone is M6: pin the exact official DeepSeek V4 Flash
checkpoint contract, inspect its metadata and Safetensors layout without loading payloads,
then build an architecture-faithful reduced fixture with trusted logits and routes.
