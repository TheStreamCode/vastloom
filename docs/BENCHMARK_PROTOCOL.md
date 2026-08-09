# Benchmark protocol

A fast number is not useful unless its memory traffic, cache state and correctness are
known.

## Required record

Every run must store:

- Vastloom commit and build type;
- operating system, driver and backend;
- CPU, cores used and RAM configuration;
- GPU(s), VRAM and PCIe generation/link width;
- SSD model/controller/filesystem and measured temperature;
- exact model revision, manifest hash and quantization;
- prompt/workload identifier without private prompt text when necessary;
- cold/warm cache state;
- context length, generated tokens and sampling settings;
- TTFT, prefill tokens/s and accepted generation tokens/s;
- peak process RAM and VRAM;
- NVMe bytes/token and observed read bandwidth;
- logical requested bytes, process transfer bytes, storage-layer bytes and physical-device
  bytes as separate fields;
- PCIe bytes/token;
- VRAM/RAM expert hit ratios;
- prefetch accuracy and wasted bytes;
- speculative acceptance and verification cost;
- output/correctness gate result;
- CPU/GPU/SSD temperatures and energy when available.

## Primary metrics

1. **Accepted tokens per second.** Draft tokens do not count.
2. **NVMe bytes per accepted token.** This exposes the physical I/O lower bound.
3. **GPU idle percentage caused by missing weights.**
4. **Quality/correctness delta against the exact baseline.**
5. **Peak bounded resources and safety-state transitions.**

## Experimental discipline

- Change one independent variable per A/B run.
- Repeat runs and publish variance, not only the best sample.
- Separate cold-cache, warm-cache and learned-placement results.
- Prefer Vastloom's `direct-unbuffered-v1` protocol for a repeatable operating-system
  cache-bypass measurement. Record it as cache-bypass, not proof that the SSD's internal
  hardware cache was cold.
- Never reduce or flush the global host file cache merely to produce a benchmark.
- Label buffered-file warm-cache probes as pipeline measurements, never as NVMe
  bandwidth or physical bytes/token evidence.
- Include negative results.
- Never compare different model revisions or quantizations as if only the runtime changed.
- Do not claim an optimization from a microbenchmark until end-to-end accepted-token
  throughput improves without failing correctness gates.

## I/O counter vocabulary

- **Logical expert bytes:** exact model payload requested by Vastloom. This is suitable
  for cache-policy accounting, not proof that the SSD served those bytes.
- **Process transfer bytes:** bytes attributed to process reads by the operating system.
  Windows `GetProcessIoCounters.ReadTransferCount` is reported in this category.
- **Storage-layer bytes:** Linux `/proc/self/io` `read_bytes`; this can show file-cache
  effects but is still not a per-device hardware counter.
- **Physical-device bytes:** bytes attributed to the selected block device by a separate
  device-level measurement. Windows uses `IOCTL_DISK_PERFORMANCE` on the mapped
  `PhysicalDriveN`; native Linux uses `/sys/dev/block/.../stat` only when the mapping is
  non-virtual. The reported scope can cover unrelated host I/O and must be retained.

Every table and JSON consumer must retain the source label and availability flag. A
missing physical counter must remain unavailable, never be silently replaced by logical
or process-transfer bytes.

## Cache-bypass protocol

`direct-unbuffered-v1` is valid only when all of the following are true:

- the original checkpoint is opened read-only through direct/unbuffered I/O;
- every output buffer and aligned sector envelope is reserved before allocation;
- expert policy is `no-cache`, I/O is asynchronous and queues/workers retain adapter caps;
- the path maps unambiguously to a device-level counter;
- device read bytes are at least the exact logical expert payload;
- model output passes the same reference parity gate as buffered mode.

This protocol isolates operating-system file-cache effects without mutating global cache
state. It cannot disable SSD controller or firmware caches, and a whole-device delta can
include concurrent reads from other processes. Repeat runs and report variance.
