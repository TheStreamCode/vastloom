# M5 readiness and staged-I/O validation

Validation date: 2026-08-09  
Implementation commit: `ed9801cc39f2d1fe928e66b719b4efb1af3f9ec8`

## Scope

This report validates two bounded experiments on the pinned real OLMoE path:

1. readiness telemetry plus fixed-route versus ready-expert trace replay;
2. an opt-in staged expert read that fetches the contiguous 8 MiB gate/up range and
   the adjacent 4 MiB down range independently.

The baseline whole-expert path remains available and reads one contiguous 12 MiB range.
Both modes execute the same CPU reference graph and preserve the original checkpoint.

This is not a DeepSeek, CUDA, physical-NVMe or accepted-token throughput validation.
OLMoE still exposes final logits and routing telemetry rather than a generation loop.

## Pinned model contract

- model: `allenai/OLMoE-1B-7B-0924-Instruct`;
- revision: `7f1c97f440f06ce36705e4f2b843edb5925f4498`;
- tensor payload: 13.84 GB across three BF16 Safetensors shards;
- always-active trunk reservation: 953,421,824 bytes;
- routed expert: three 4 MiB matrices, 12 MiB total;
- per-token probe: 16 layers, top-8 routing, 128 expert executions.

The inspected checkpoint stores each expert's down projection immediately before a
contiguous gate/up region. Staged mode therefore issues an 8 MiB gate/up request and a
4 MiB down request without repacking, rewriting or deriving alternate weights.

## Safety and boundedness

Whole asynchronous mode uses exactly one worker. Staged mode uses exactly two workers,
with prefetch depth restricted to one or two and a maximum queue size of four requests.
Every request owns a Resource Governor staging reservation until its buffer is consumed.
Each request is at most 12 MiB and retains the existing positive timeout, cancellation,
short-read and worker-failure checks.

The first Windows staged attempt was deliberately refused when the host had only about
9.59 GB of available RAM. The runtime retained 8 GiB for Windows and could not safely
reserve the 953,421,824-byte trunk plus execution resources. No safety limit was lowered
and no user process was terminated to force the run.

## Readiness telemetry and replay

The native trace format grew from 12 to 19 fields:

```text
token,layer,expert,size,routing_weight,tier,hit,bytes_read,
io_service_ms,foreground_wait_ms,io_compute_overlap_ms,
reuse_distance,submitted_ms,queue_ms,ready_ms,
compute_started_ms,compute_completed_ms,route_order,reduction_order
```

The parser remains compatible with the committed 4-, 9- and 12-field formats. For every
token/layer group it bounds the record count, validates finite nonnegative timings, and
requires unique contiguous route and reduction orders. The simulator may model compute
in ready order, but it never changes the authoritative reduction order used for output.

A deterministic synthetic fixture measured:

| Replay | Makespan | Foreground wait |
|---|---:|---:|
| Fixed route order | 19 ms | 10 ms |
| Ready-expert order | 14 ms | 5 ms |

The replay reordered three compute slots and saved 5 ms. The current real OLMoE trace
contained 16 token/layer groups but produced no ready-order change and no modeled saving.
Consequently, a native ready-order scheduler was not added without evidence.

## Correctness results

Whole and staged modes returned the same pinned reference result:

- greedy token: `47111` (`" Byr"`);
- top logits: identical to the committed Transformers oracle;
- expert choices and route weights: identical for all 128 executions;
- logical expert payload: 1,610,612,736 bytes in both modes;
- staged accounting: 1,073,741,824 gate/up bytes plus 536,870,912 down bytes;
- staged process read operations: 290, versus 162 in whole mode, matching the expected
  extra range per expert.

The real integration suite also verified staged LRU insertion and reuse, exact byte
accounting, positive down-read/gate-up-compute overlap, and fail-closed handling of an
injected gate/up short read.

## Warm buffered pipeline benchmark

The benchmark ran under WSL 2 on the reference PC:

- Windows 11 host;
- AMD Ryzen 9 7900X, 24 logical CPUs exposed to WSL;
- NVIDIA RTX 4070 Super detected, but not used for inference;
- 16,294,531,072 bytes of WSL memory;
- GCC Release build;
- ordinary buffered file reads against an already-used checkpoint;
- no-cache policy, asynchronous I/O, prefetch depth two;
- alternating whole/staged order, three runs per mode.

| Sequence | Mode | Forward ms | I/O service ms | Foreground wait ms | Service/CPU overlap ms | Down overlap ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | whole | 5,787.77 | 5,706.13 | 5,494.52 | 115.33 | 0.00 |
| 2 | staged | 3,125.64 | 5,545.39 | 2,844.85 | 215.92 | 79.60 |
| 3 | staged | 3,265.18 | 5,712.76 | 2,923.44 | 223.29 | 82.29 |
| 4 | whole | 5,557.08 | 5,476.54 | 5,266.19 | 112.39 | 0.00 |
| 5 | whole | 6,017.80 | 5,933.79 | 5,712.88 | 117.07 | 0.00 |
| 6 | staged | 3,091.04 | 5,487.66 | 2,812.14 | 217.31 | 79.92 |

Median whole time was 5,787.77 ms. Median staged time was 3,125.64 ms, a 46.0%
improvement in this warm buffered pipeline. Every run returned token `47111` and exactly
1,610,612,736 logical expert bytes.

This result measures scheduling and buffered-read overlap. It does not measure cold-cache
device bandwidth or physical NVMe bytes: WSL and the Windows host file cache may have
served some or all data without device traffic.

## Negative experiment

The first staged prototype used one worker. It was slower than whole reads:

- staged, depth two: 6,173.44 ms versus whole 5,821.14 ms, approximately 6% slower;
- staged, depth one: 6,470.20 ms versus whole 5,611.92 ms, approximately 15.3% slower.

The one-worker staged design was therefore rejected. The accepted implementation adds
only the bounded two-worker option that produced a measured improvement while retaining
output parity.

## Build and test matrix

| Toolchain/configuration | Result |
|---|---|
| MSVC Release, warnings as errors | 5/5 passed |
| MSVC Debug, warnings as errors | 5/5 passed |
| Clang 18 Release, warnings as errors | 5/5 passed |
| GCC Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 5/5 passed |
| GCC Release, full external OLMoE integration after staged cache/fault additions | 1/1 passed in 161.02 s |
| GCC Release, complete suite with external SmolLM2 and OLMoE | 8/8 passed in 221.26 s |

The MSVC Release invocation initially omitted `-C Release`, causing CTest to report five
tests as not run because it could not locate configuration-specific executables. The
corrected `ctest -C Release` invocation passed 5/5; this was an invocation error, not a
source failure.

## Known limitations and next gate

- no per-device Windows NVMe byte counter;
- no repeatable cold-cache protocol;
- no direct/unbuffered I/O or pinned host memory;
- no full OLMoE generation/chat loop;
- no DeepSeek V4 adapter, checkpoint validation or architecture fixture;
- no CUDA inference or VRAM expert tier.

The remaining M5 gate is a trustworthy physical-device counter plus a safe repeatable
cold-cache protocol. Only after separating logical, process, storage-layer and physical
traffic should the exact DeepSeek checkpoint work begin.
