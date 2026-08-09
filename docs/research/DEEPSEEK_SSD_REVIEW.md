# DeepSeekV4SSD review for Vastloom

Reviewed on 2026-08-09 at upstream commit
[`57e440e`](https://github.com/yanun0323/deepseek_ssd/commit/57e440e48b79fb0399beae7e9965481f74b35b25).

## Verdict

[DeepSeekV4SSD](https://github.com/yanun0323/deepseek_ssd) is highly relevant evidence that
model-specific expert layout, bounded residency and execution ordering can make a 284B-class sparse
checkpoint interactive on a high-bandwidth consumer system. It is useful as an experimental baseline,
not as a drop-in backend: it targets Apple Silicon, MLX/Metal and 64 GiB unified memory, while Vastloom's
first target is Windows, CUDA, 12 GiB discrete VRAM and 32 GiB system RAM.

No upstream code is copied by this review. Any future adaptation must be independently validated and
retain attribution required by the upstream MIT license.

## Evidence worth carrying forward

The upstream validation record reports a pinned 43-layer checkpoint with 256 routed experts per layer,
six selected experts per token, about 8.24 GiB of common weights and 156,015,738,880 installed weight
bytes. Its measurements clearly separate direct and cached SSD results, warm-cache effects, bytes read,
peak memory and token throughput. That reporting discipline matches Vastloom's benchmark policy.

The most transferable findings are:

1. **Ready-expert scheduling.** Submit all required reads, compute resident or completed experts as they
   become ready, then preserve the native selected-order reduction. Upstream reports a median 12.9%
   decode-throughput improvement over paired tests without changing token hashes.
2. **Layer-major prefill.** Long prompts can reuse a complete layer's experts before advancing to the next
   layer. Upstream reports that its varied 14K-token test reduced expert reads from 2,290.6 GB to 131.5 GB
   and improved time from 386.65 s to 105.81 s. This is architecture- and workload-specific, not a promise
   for Vastloom.
3. **Prefill-to-decode handoff.** Route histograms already observed during prefill can guide short-lived
   cache admission for early decode without changing the router or adding speculative SSD reads.
4. **Split expert staging.** The gate/up projection region can be consumed before the down projection is
   needed. Reading those regions separately may overlap the down-weight read with first-stage compute and
   makes partial cache hits possible.
5. **Measurements before cache growth.** Upstream found that blindly increasing slots and a layer-order
   eviction policy did not improve total time. Vastloom should keep policy experiments trace-driven.

Sources: upstream [README](https://github.com/yanun0323/deepseek_ssd),
[validation record](https://github.com/yanun0323/deepseek_ssd/blob/master/docs/VALIDATION.md) and
[expert-streaming research](https://github.com/yanun0323/deepseek_ssd/blob/master/docs/EXPERT_STREAMING_RESEARCH_2026-08-09.md).

## Vastloom adoption order

The status below distinguishes implemented OLMoE experiments from later DeepSeek work.

### P0 — readiness telemetry — implemented

The real OLMoE trace now records submission, queue wait, ready time, compute start/end,
route order and reduction order. The parser remains compatible with the earlier 4-, 9-
and 12-field formats.

### P1 — trace-driven ready-expert simulator — implemented

The bounded simulator replays each token/layer group, validates contiguous unique route
and reduction orders, and reports fixed/ready makespan, wait, savings and compute reorder
count. A synthetic out-of-order-ready fixture demonstrates a 19 ms to 14 ms replay; the
current real OLMoE traces produce no ready-order change, so a native scheduler is not yet
justified.

### P2 — OLMoE gate+up / down split prototype — accepted for OLMoE

The pinned checkpoint stores down immediately before the contiguous gate/up region. The
opt-in native path reads gate/up as 8 MiB and down as 4 MiB through two workers and a
four-request maximum queue; every buffer is reserved independently and the original
weights remain untouched. Greedy token, top logits, routes and 1,610,612,736 logical
bytes match whole mode. Three alternating paired WSL runs measured 5,787.77 ms median
for whole reads and 3,125.64 ms for staged reads, a 46.0% warm buffered-pipeline win.
This passes the 5% research gate but is not a physical-NVMe result.

### M5 measurement gate — implemented

Vastloom now maps the Windows checkpoint volume to `PhysicalDriveN`, preserves the
whole-device scope, and provides a bounded `direct-unbuffered-v1` protocol instead of
flushing global cache state. Three repeated Windows runs produced identical reference
output and 1,610,743,808 device bytes for 1,610,612,736 logical bytes. Native Linux
accepts only non-virtual block-device mappings; WSL `/mnt/c` remains unavailable.

### P3 — long-prefill layer-major baseline

Do not transfer the upstream full-layer strategy blindly: OLMoE and DeepSeek have different expert sizes,
counts and compute balance. First simulate memory cost and bytes saved from real traces, then reserve the
entire temporary layer before any read. The Safety Governor may reject the path regardless of speed.

### DeepSeek-specific adapter — metadata contract pinned

Official commit `60d8d70770c6776ff598c94bb586a859a38244f1` is now inspectable through
46 verified header snapshots without downloading tensor payloads. The catalog confirms
three hash-routing layers, 43 main layers, 256 routed experts, top-6 selection and one
MTP block. This closes the metadata prerequisite but does not validate runtime semantics.

After the architecture-faithful fixture and native correctness gates, re-evaluate
hash-layer exact prefetch, prefill-to-decode handoff and region-aware cache admission.
Learned prediction remains after real route traces and required-vs-speculative I/O
priority exist.

## What Vastloom should not copy

- A Python/MLX generation hot path for the Windows/CUDA target.
- Fixed worker, slot or memory defaults derived from a 64 GiB unified-memory Mac.
- Warm page-cache throughput presented as physical SSD performance.
- A model-specific packed format before its conversion, validation, checksum and atomic-publication
  contract exists.
- Any approximation that changes router eligibility in the default exact mode.

The practical conclusion is positive: DeepSeekV4SSD strengthens Vastloom's core thesis and identifies
two low-risk next experiments—ready-expert scheduling and split expert reads—while reinforcing that
**NVMe bytes per accepted token** matters more than raw read bandwidth.
