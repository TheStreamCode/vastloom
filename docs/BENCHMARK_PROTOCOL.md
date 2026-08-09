# Benchmark protocol

A fast number is not useful unless its memory traffic, cache state and correctness are
known.

## Required record

Every run must store:

- Stargate commit and build type;
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
- Label buffered-file warm-cache probes as pipeline measurements, never as NVMe
  bandwidth or physical bytes/token evidence.
- Include negative results.
- Never compare different model revisions or quantizations as if only the runtime changed.
- Do not claim an optimization from a microbenchmark until end-to-end accepted-token
  throughput improves without failing correctness gates.

## I/O counter vocabulary

- **Logical expert bytes:** exact model payload requested by Stargate. This is suitable
  for cache-policy accounting, not proof that the SSD served those bytes.
- **Process transfer bytes:** bytes attributed to process reads by the operating system.
  Windows `GetProcessIoCounters.ReadTransferCount` is reported in this category.
- **Storage-layer bytes:** Linux `/proc/self/io` `read_bytes`; this can show file-cache
  effects but is still not a per-device hardware counter.
- **Physical-device bytes:** bytes attributed to the selected block device by a separate
  device-level measurement. Stargate does not yet capture this on the reference Windows
  host.

Every table and JSON consumer must retain the source label and availability flag. A
missing physical counter must remain unavailable, never be silently replaced by logical
or process-transfer bytes.
