# Alpha 4.3 I/O observability and reuse-distance validation

Validation date: 2026-08-09

Implementation commit: `1cf1cc3aa22a97dad43cf86dc8e9063cec400bd5`

## Scope

Alpha 4.3 makes the real OLMoE pipeline's I/O behavior inspectable without confusing
model payload, operating-system accounting and physical NVMe traffic:

```text
validated routed expert range
        -> bounded buffered read
        -> process read-operation/byte delta
        -> exact native route trace
        -> unique-expert reuse-distance report
```

It also fixes a Windows implementation artifact found through the new counters. The
asynchronous C++ stream path issued 393,216 process read operations for 128 contiguous
12 MiB experts. The new Unicode-native Win32 range reader issues one bounded operation
per expert in the validated probe.

## Implemented

- portable process-I/O snapshot and monotonic-delta types;
- `GetProcessIoCounters` read operations and transfer bytes on Windows;
- bounded `/proc/self/io` parsing for Linux `syscr`, `rchar` and `read_bytes`;
- explicit source and availability fields in text and JSON CLI output;
- Unicode-safe `CreateFileW`, `SetFilePointerEx` and synchronous `ReadFile` ranges inside
  the existing one-worker asynchronous pool;
- exact unique-expert LRU stack distance in `O(records log records)` bounded by the
  already bounded trace size;
- cold/reused counts, reuse ratio, mean, P50, P95 and maximum distance;
- Unicode-path, counter, fragmentation and real repeated-route regression coverage.

The Windows `ReadTransferCount` value is process transfer accounting. It is not labeled
as a physical-device counter. Linux `read_bytes` is reported separately as a
storage-layer counter. Neither replaces a future per-device measurement.

## Safety envelope

The final Windows probe reported:

- AMD Ryzen 9 7900X, 24 logical CPUs;
- 33,407,475,712 bytes physical RAM and 13,492,285,440 bytes available;
- RTX 4070 Super with 12,878,610,432 bytes VRAM, not used by the CPU forward;
- 1,205,598,654,464 bytes available beneath the workspace volume.

The existing real-model policy retained eight GiB of current RAM availability, used no
more than six CPU threads and kept the process inside the Windows Job Object cap. One
manual attempt made while available RAM was transiently below that headroom was refused
before the model allocation; the same probe passed after availability recovered. This
is the intended fail-closed behavior.

The native file path remains buffered and read-only. It does not enable unbounded direct
I/O, expand the worker count, change the two-request queue cap, remove staging
reservations or rewrite a checkpoint.

## Correctness and observability results

The final real-checkpoint manual command used raw `Hello`, no cache, asynchronous I/O
and prefetch depth two. It returned:

- greedy token `47111` (` Byr`), unchanged from the Transformers oracle;
- 128 routed expert misses and 128 trace events;
- 1,610,612,736 exact logical expert bytes;
- 128 Windows process read operations;
- 1,610,612,736 Windows process transfer bytes;
- `storage_read_bytes_available=false` and a JSON `null` storage-byte value;
- 700.945 ms forward time, 463.558 ms worker service and 92.7624 ms measured
  service/CPU overlap.

Initial checkpoint verification and trunk loading took 44.8991 seconds in that run and
is reported separately from forward time.

The repeated two-token route fixture produced 256 records over 128 unique
`(layer, expert)` identities: 128 cold accesses, 128 reused accesses and P50/P95 exact
unique reuse distance 127. This is the expected access sequence, not a synthetic cache
estimate substituted for native routing.

## Build and test matrix

| Gate | Exact result |
|---|---:|
| MSVC 19.44 Release, `/WX`, complete external SmolLM2 and OLMoE | 8/8 passed |
| MSVC 19.44 Debug, `/WX`, external SmolLM2 and metadata OLMoE | 7/7 passed |
| GCC Release, `-Werror` | 5/5 passed |
| Clang Release, `-Werror` | 5/5 passed |
| GCC Debug, `-Werror` | 5/5 passed |
| GCC Debug + ASan + UBSan, leak/halt enabled | 5/5 passed |
| GCC Debug + libstdc++ debug iterators/assertions | 5/5 passed |
| Python `compileall` for tools, tests and scripts | passed |

The final Windows Release suite passed in 68.64 seconds; its complete-checkpoint OLMoE
test passed in 54.04 seconds. The Windows Debug suite passed in 81.88 seconds.

ThreadSanitizer compiled successfully, but this WSL runtime stopped all instrumented
native executables before `main` with `ThreadSanitizer: unexpected memory mapping`.
TSan is unavailable here, not passing and not reporting a source-code race.

## Buffered warm-cache pipeline measurements

Five interleaved runs per mode used MSVC Release, raw `Hello`, no cache, the same
1,610,612,736 logical expert bytes and the same greedy token `47111`.

| Mode | Forward mean (min-max) | I/O service mean | Foreground wait mean | Actual overlap mean | Process read operations |
|---|---:|---:|---:|---:|---:|
| sync | 1,176.476 ms (1,107.27-1,223.53) | 975.722 ms | 0 ms | 0 ms | 393,216 |
| async depth 1 | 769.089 ms (625.229-823.479) | 374.655 ms | 221.438 ms | 105.211 ms | 128 |
| async depth 2 | 645.752 ms (496.54-686.112) | 413.079 ms | 48.110 ms | 92.352 ms | 128 |

Depth one reduced mean forward time by 34.6% and depth two by 45.1% relative to the
synchronous stream baseline in this warm buffered protocol. The strong change is
consistent with removing 4 KiB read fragmentation, but it is not an SSD throughput or
physical bytes/token result: the files were warm, Windows buffering remained enabled
and no device counter was captured.

## Known limitations

- no Windows physical-device byte counter or repeatable cold-cache protocol;
- the synchronous reference reader still uses the portable C++ stream path;
- buffered reads only; no direct/unbuffered I/O or pinned host memory;
- one worker and fixed depth, not an adaptive storage scheduler;
- no per-device bandwidth, queue-depth, temperature or power telemetry;
- no OLMoE generation loop, CUDA/VRAM expert tier or DeepSeek adapter;
- the complete external OLMoE path was validated on Windows, not under Linux
  sanitizers.

## Next bottleneck

Capture trustworthy physical-device traffic on the reference Windows NVMe and define a
repeatable cold-cache protocol. Once logical, process, storage-layer and physical bytes
can be compared without ambiguity, begin the pinned DeepSeek V4 Flash checkpoint
inspector and architecture-faithful reduced fixture under the same safety boundaries.
