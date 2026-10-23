# Peer review — SSD-streaming engines (Kimi K3, DS4, Colibri, WARP)

Status: research input, not a milestone. Date: 2026-10-23. This document reviews four
independent projects that solve the same class of problem Vastloom studies — running
Mixture-of-Experts models whose weights do not fit in RAM — and records what Vastloom should adopt,
what it already does better, and what it should not copy.

Method: the public READMEs and design documents were read directly from the repositories below (no
code was executed, no weights downloaded). Repository facts (stars, language, description) are a
snapshot of 2026-10-23. Claims attributed to a peer are that project's published claims, not
measurements reproduced here; claims about Vastloom are checked against the in-tree code.

## The four projects

| Project | URL | What it is | Published position |
|---|---|---|---|
| kimi-k3-ssd-streaming | <https://github.com/jiro-prog/kimi-k3-ssd-streaming> | Windows experiments + llama.cpp patch for a 2.8T-parameter Kimi K3 (`UD-IQ1_S`, 594 GB) on 16 GB RAM / 8 GB VRAM | 45.0 → 4.54 s/token after prefetch, direct reads, re-quantization and DSpark |
| DwarfStar / ds4 | <https://github.com/antirez/ds4> | DeepSeek V4-specific C engine for Metal, CUDA and ROCm, with SSD streaming, compressed KV, multi-GPU and RDMA | 24k stars; 126 tok/s aggregate on 8×L40S, 16 sessions; Q2 Flash on a 128 GB Mac |
| Colibri | <https://github.com/JustVugg/colibri> | Pure-C, dependency-free engine that keeps the dense part in RAM and streams routed experts from disk, with learned placement | 43.5k stars; 0.93 tok/s DeepSeek V4 Flash on a 32 GB Ryzen; 1.5–1.6 tok/s with CUDA |
| WARP | <https://github.com/sqliteai/warp> | Embeddable C runtime with a purpose-built container: trunk resident, expert banks streamed, 3-bit vector-quantized experts | 2.5k stars; one coalesced `pread` per expert; explicit RAM floor/ceiling and paging-cliff analysis |

## What each one actually teaches

### kimi-k3-ssd-streaming — Windows I/O is the whole game

- **Page faults cap out around 1.2 GB/s on Windows**, regardless of thread count; unbuffered reads
  reach ~5 GB/s. The project's fastest configuration reads the SSD at 4.87 GB/s (97% of the drive).
- **A live memory-mapped section serializes unbuffered reads of the same file.** llama.cpp keeps the
  GGUF mapped, so switching to direct reads alone did not parallelize; mapping into placeholders and
  closing the section handle for streamed ranges was required. This is a Windows-specific trap that
  a generic "use O_DIRECT/NO_BUFFERING" change would not find.
- **`VirtualQuery` cost grows with the number of touched pages** (start-up 130 s → 1.64 s after
  avoiding the scan), a reminder to keep metadata probes O(1), not O(pages).
- **Speculative decoding helps even when SSD-bound** (11.6 → 7.32 s/token), and a small draft
  fine-tuned on Japanese gave another ~8% on Japanese prompts.
- **Re-quantizing the non-expert weights was the lever once I/O was saturated**: the final 3-line
  verification pass reads ~48 GB at ~4.8 GB/s and the SSD has 0.1% idle; every 1 GB removed from a
  pass is ~0.2 s. When the disk is the bottleneck, the only remaining lever is bytes per pass.

**Adopt:** direct/unbuffered reads only with the section-handle discipline; measure page-fault vs
direct read paths; treat bytes-per-pass as a first-class budget; use a 10 ms timeline to attribute
disk, CPU and layer time.

### DwarfStar (ds4) — specialize the engine, and keep state

- A single-model family engine (DeepSeek V4 Flash/PRO, GLM, Qwen) rather than a generic runtime;
  quantization formats and kernels are retained from GGML, but the graph and scheduling are
  model-specific.
- SSD streaming for experts with compressed KV caches, **KV-cache persistence** and session files,
  chunked prefill, MTP/DSpark speculation, Engram tables kept on disk, and directional steering.
- Scales beyond one box: tensor parallelism over RDMA, pipeline parallelism over machines,
  multi-user server; documented power-limit control.

**Adopt:** KV persistence and chunked prefill are prerequisites for long-context usefulness; Engram
(or any other auxiliary table) should be disk-backed rather than resident by default; measure at
the same "task" level (prefill + generation) instead of per-layer microbenchmarks.

### Colibri — placement is speed, never precision

- **Learning cache:** stores actual routing counts in a usage file, auto-pins the hottest experts in
  spare RAM at startup, auto-sizes the LRU to the RAM budget, and optionally repins at turn
  boundaries with decay, hysteresis (25%) and a swap limit. The engine literally gets faster with
  use.
- **Router-lookahead prefetch:** applying layer L+1's router to layer L's post-attention state
  recalls **71.6%** of the true top-8, against 41.3% for "same experts as last token"; a dedicated
  I/O thread prefetches during the current layer. On a disk-saturated host, hint-only prefetch can
  be net negative — measure it.
- **I/O knobs with published wins:** O_DIRECT +65% on one host, io_uring batched reads, an async
  expert-load pool (−18% disk service), multi-drive mirroring (+37.5% decode with two NVMe), NUMA
  interleave, and a `plan` command that prints hot/warm/cold tiers, the reason for each placement
  and machine-readable `next_actions`.
- **Explicit lossy mode is opt-in and warns:** top-p expert selection (30–40% less disk) and router
  reduction are outside the default quality policy; the default preserves precision and routing.

**Adopt:** persist routing heat, auto-size and repin the cache; make every lossy knob explicit and
warned; add a plan/introspection command; measure prefetch on the actual disk-saturated host because
it can be negative.

### WARP — the container is the optimization

- **One coalesced read per expert**: gate/up/down are adjacent on disk and loaded with a single
  `pread`; the project calls it "the difference between usable and unusable NVMe throughput".
- **4 KiB-aligned records, 4 KiB sizes**, versioned container, per-expert CRC, frozen manifest;
  O_DIRECT-friendly by construction. The engine checks format version and sizes on the read path.
- **RAM floor and ceiling:** the floor is what must be resident to run at all; the default cache is
  sized from the budget, capped at the expert-bank size, and the engine refuses/plans rather than
  swapping. Larger cache is *not* monotonic: 17.32 GB produced 0.63 tok/s while 23.32 GB collapsed
  to 0.07–0.09 because page faults rose. The engine reads `memory.max`/cgroup limits, not just
  physical RAM.
- **3-bit residual-VQ experts** measured on real Kimi experts (19.4% error; 2-bit at 33% is too
  lossy); **non-uniform per-expert bit allocation was measured and dropped** because the benefit
  spread was ~1.0–1.15×. Shared low-rank is specified but not implemented.
- **Sharding:** experts split round-robin across bank files/drives (expert `e` lives in shard
  `e % N`); the engine also supports optional substitutes and a continuation-based quality gate.

**Adopt:** a derived, aligned, per-expert-record container with maintained provenance; explicit
floor/ceiling cache sizing with a paging-cliff guard; multi-drive expert banks; measure quantization
changes with a continuation gate, not only KL/logit distance.

## Where Vastloom stands today

Verified in the working tree:

- **Provenance is stronger than every peer reviewed here.** Pinned GGUF revision, per-range SHA-256,
  immutable artifacts, metadata-only identities, authoritative `tid2eid` routes, independent NumPy
  oracles and bit-identical CPU/device comparisons. None of the peers documents an equivalent
  end-to-end hash chain with a public, replayable gate.
- **An I/O layer already exists but is not wired to the DeepSeek path.** `src/io/bounded_io_pool.cpp`
  implements Windows `FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS` and Linux `O_DIRECT`, aligned
  buffers, an async pool and device counters (`:70-72`, `:172`, `:245-260`). The DeepSeek range
  loaders still call `read_buffered_range_into` (`src/model/bound_range_loader.cpp:214`,
  `src/model/layer_streaming.cpp:193`, `src/model/expert_loader.cpp:133`). The expert-loader comment
  records a measured reason: unbuffered reads of the same small ranges were three times slower when
  the page cache already held recurrences. So the right change is a **policy and a measurement**,
  not a blanket switch.
- **Range artifacts are many small files, not coalesced bundles.** A six-expert layer-0 artifact
  contains 18 separate range files (three per expert, sizes 3,211,264 / 2,162,688 / 2,162,688 B);
  the three tensors are far apart in the source GGUF. There is no single-read-per-expert property.
- **The DeepSeek chain cache is synchronous and policy-limited.** It has LRU capacity bounds
  (`src/cli/main.cpp:4160-4175`) but no async workers, prefetch or learned heat; the full-depth
  forward path does use four async workers (`src/cli/main.cpp:5212`), and the graph prefetches only
  hash-routed next layers (`src/inference/deepseek_v4_graph.cpp:693-718`). Learned-router layers get
  no prefetch.
- **Cache-policy machinery already exists elsewhere:** OLMoE supports
  `--cache no-cache|lru|lfu|static-hot`, `--io sync|async`, `--io-cache buffered|bypass`,
  `--prefetch-depth` (`src/cli/main.cpp:349-351`; `src/inference/olmoe_runtime.cpp:950-1015`), and
  `trace-sim` can compare policies. The DeepSeek path has not inherited these knobs.
- **No usage memory, no auto budget plan, no cache-cliff guard.** The Governor and preflight are
  conservative and explicit, but there is no persisted routing heat, no repin, no plan output with
  tiers/reasons/next actions.
- **No KV persistence or chunked prefill** on the DeepSeek path; the Qwen backlog holds the design
  (Q11k/Q12) but the engines reviewed here treat them as baseline.

## Prioritized improvements

### P0 — highest value, fits the existing architecture

1. **Derived expert-bundle container with one aligned read per expert.**
   Pack each expert's three tensors into one contiguous, 4 KiB-aligned record in a derived artifact
   (a sibling payload file with a JSON index and per-record CRC), while keeping the pinned GGUF
   range identity and SHA-256 manifest as the provenance layer. Load with one `pread` per expert.
   Expected: fewer syscalls, real NVMe coalescing, direct-I/O eligibility; WARP calls this the
   difference between usable and unusable throughput.
   Gate: the layer/chain logits stay bit-identical to the current artifacts; read counters show one
   read per expert instead of three; cold/warm A/B published in a report.
2. **Adaptive I/O policy (`--io-cache auto`) with measurement.**
   Keep buffered reads for small recurring ranges (the existing measurement), switch to the existing
   bypass pool for cold/large expert streams and coalesced records; expose the mode in telemetry.
   Gate: same logits in every mode; cold/warm throughput on the reference PC reported, including the
   Windows section-handle caveat from the Kimi project.
3. **Learned expert placement.**
   Persist per-expert hit counts across runs in a sidecar usage file; auto-pin the hottest experts
   in RAM/VRAM; add a bounded repin at turn boundaries with hysteresis. Placement must never change
   arithmetic: the cache remains a source of bytes, not a precision or routing decision.
   Gate: bit-identical outputs; hit rate and bytes/token improve on repeated prompts; the sidecar is
   ignored by default and its absence is harmless.
4. **Router-lookahead prefetch for learned layers.**
   Hash layers already prefetch; for learned layers, apply layer L+1's router to layer L's
   post-attention state (Colibri's 71.6% vs 41.3% recall result) and issue bounded readahead on a
   dedicated I/O thread. Keep it opt-in and measure on the actual host: the peer explicitly reports
   it can be negative on a saturated disk.
   Gate: prefetch hit-rate and wasted-prefetch bytes reported; no wrong-byte execution; net decode
   time improvement measured, or the gate fails and the feature stays off.

### P1 — important after the P0 I/O/placement work

5. **Plan and cache-cliff guard.** Port WARP's floor/ceiling reasoning into a `plan`-style output
   (resident floor, chosen cache, expected bottleneck, next actions) and add a guard/test that a
   larger cache cannot silently trigger page-fault collapse. Include cgroup/usable-RAM detection on
   Linux, not just physical RAM.
6. **Multi-drive expert banks.** Shard expert records by layer/expert across two or more NVMe
   devices (round-robin bank files), optionally mirrored on fast RAM; Colibri measured +37.5% with
   two drives. Keep the single-store path as the default and fail closed if a bank is missing.
7. **KV persistence and chunked prefill.** Adopt DS4's session/KV files and prefill chunking before
   claiming long-context usefulness; the Qwen design (Q11k/Q12) is the template.
8. **Quality/speed knobs behind explicit warnings.** Expert top-p selection, expert-count reduction
   and smaller quantization are legitimate research knobs, but they must remain opt-in, warned, and
   measured with a continuation-level quality gate (WARP) in addition to logit/KL metrics. Vastloom's
   default must stay exact.
9. **Speculation, revisited only with a draft.** Kimi and DS4 both show speculation helps in the
   SSD-bound regime; Vastloom's DSpark feasibility work is archived. Any return needs draft weights,
   a pinned kernel family for reproducibility, and a clear acceptance-rate reporting rule.

### P2 — optional, after a full-depth execution exists

10. **Observability.** A trace viewer/dashboard over the existing JSON traces (expert fires, I/O
    timeline, tier residency) would make the hierarchy legible; Colibri's dashboard is the reference.
11. **Multi-device and RDMA.** DS4 demonstrates tensor/pipeline parallelism and multi-user serving.
    This is beyond the reference PC but affects the architecture of the store and the expert bank.

## What not to copy

- **A container without provenance.** Any packed record must remain traceable to the pinned source
  range and hash; a container that cannot answer "which original bytes are these" is out of bounds.
- **Silent precision or routing changes.** Colibri and WARP both keep placement separate from
  arithmetic; Vastloom's fail-closed rule is stronger and should stay.
- **Predicting learned routes as if they were authority.** A predicted route may prefetch; it must
  never decide which experts execute.
- **Blanket O_DIRECT.** The in-tree measurement says it was slower for recurring small ranges;
  policy must depend on the regime.
- **Benchmarks without cold/warm state, device vs logical bytes and a dated report.**

## Experiments that can run on the current slice (no new downloads)

1. **Read-count and layout baseline.** Count the current reads per expert and verify the 3-read
   property on the existing artifacts (18 files / 6 experts); build a packed sidecar prototype under
   `build/` and confirm one read per expert without changing the logits.
2. **I/O mode A/B.** Reuse the existing bypass pool on one layer/chain run; report cold and warm
   runs, Windows file-section behavior and exact wall time. Keep the buffered default until the
   measurement says otherwise.
3. **Cache-policy simulation.** Use `trace-sim` and recorded routing traces to compare LRU,
   static-hot and LFU on the same sequence, including the paging-cliff hypothesis.
4. **Prefetch recall study.** The current three-layer slice has hash routing only, so learned-layer
   lookahead cannot be measured yet; record the experiment protocol and defer it to the first
   learned-layer execution (F2 in `docs/planning/FUTURE_WORK.md`).

## Conclusion

The peers converge on three levers Vastloom has not yet applied to the DeepSeek path: **coalesced
aligned records, adaptive direct I/O, and learned placement with bounded prefetch**. Vastloom's
differentiator is the opposite axis — pinned provenance, independent oracles, bit-identical
placement and honest partial-depth labeling. The right improvement is to adopt the I/O and
placement mechanics inside the existing verification envelope, not to compete on feature breadth
or throughput claims.

## Links

- Peer projects: see the table above.
- Vastloom code references: `src/io/bounded_io_pool.cpp`, `src/model/expert_loader.cpp`,
  `src/model/bound_range_loader.cpp`, `src/model/layer_streaming.cpp`,
  `src/inference/deepseek_v4_graph.cpp`, `src/cli/main.cpp`.
- Resume backlog: [`docs/planning/FUTURE_WORK.md`](../planning/FUTURE_WORK.md).
- Measured evidence: [`docs/reports/`](../reports/README.md).
