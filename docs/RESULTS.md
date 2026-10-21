# Results and positioning

What the project established, what it deliberately did not, and where it sits against the
mmap-based engines it measured against. The numbered evidence for each claim is in
[`reports/`](reports/README.md); the command surface is in [`USAGE.md`](USAGE.md).

## North star

The further north star, a fast resident frontman expanded at runtime by a dormant brain on NVMe
through bounded bridges, is documented separately in
[`archive/deepseek_0731/CAPABILITY_TRANSFER_VISION.md`](archive/deepseek_0731/CAPABILITY_TRANSFER_VISION.md).
It is a compass for future work, not a claim about what runs today.

## Why Vastloom exists

Most runtimes ask whether a model fits. Vastloom asks what each tensor is, when it is
needed and which memory tier can serve it safely. The Safety Governor has authority over
the performance scheduler: memory is reserved before allocation, queues and I/O buffers
are bounded, original weights remain immutable, and emergency pressure aborts generation
instead of sacrificing the host.

```text
verified model -> safe hardware plan -> model-aware placement -> bounded inference -> telemetry
                     VRAM -> RAM -> NVMe
```

This was real systems research, run as vertical slices: native dense inference, real
MoE routing, bounded expert streaming on a smaller published checkpoint, a complete real
DeepSeek expert, and finally full native execution of Qwen3.8-Flash-Next with externally
validated greedy parity — the result the Key results section records.

Every milestone has a dated validation report under [`reports/`](reports/README.md);
the index there lists all of them, current track first. The ones to read first:

- [`Q11C Qwen3.8 llama.cpp parity`](reports/Q11C_QWEN38_LLAMA_PARITY_VALIDATION.md) — the
  primary track's external validation, including the six defects only the oracle could find.
- [`Q11B real 125B catalog validation`](reports/Q11B_QWEN38_REAL_CATALOG_VALIDATION.md) —
  the imported FP8 checkpoint's catalog closed end-to-end in native C++.
- [`Alpha 4 OLMoE validation`](reports/ALPHA4_OLMOE_VALIDATION.md) — real MoE routing and
  bounded expert streaming on a published checkpoint.
- [`M9 complete forward validation`](reports/M9_DEEPSEEK_COMPLETE_FORWARD_VALIDATION.md) —
  the full 43-layer DeepSeek execution on the reference PC (historical track).

Current and candidate model support is kept explicit in the
[`model compatibility matrix`](MODEL_COMPATIBILITY.md).

## What Vastloom adds over an mmap-only runtime

Throughput was never Vastloom's win: on the reference PC it decodes roughly
ten times slower than the mmap-based engines (see Current limitations and the
prior-art landscape below). What it does offer is the scaffolding around the
matvecs — the control plane an mmap-only runtime structurally lacks:

- **Safety Governor with reservation semantics** — every large allocation is
  reserved before it is made through move-only RAII reservations, under
  per-kind plus aggregate host-memory ceilings; workers that race on the
  same free memory cannot jointly oversubscribe it
  ([`SAFETY.md`](SAFETY.md));
- **OS, emergency and storage headroom** — the planner subtracts host, VRAM
  and free-space reserves before budgeting a single weight, and live
  headroom is re-sampled during execution;
- **OS-enforced process caps** — on Windows the process runs under a Job
  Object memory limit with lowered priority; VRAM is a separate tier that is
  never summed with RAM to admit an expert, and device allocations are
  charged against the host budget they also consume;
- **pressure ladder, not hope** — on pressure the runtime sheds cache,
  prefetch and speculation first; on emergency it aborts the generation and
  releases resources instead of risking host stability;
- **bounded execution everywhere** — queues, workers, prefetch depth, request
  bodies, generation lengths, timeouts and cancellation are explicit
  parameters; every shell-out carries a hard deadline and a watchdog guards
  the process heartbeat;
- **no silent behavior** — declined shapes fall back to the scalar reference
  kernels call by call and visibly; a backend failure is an error, never a
  quiet substitution; temperature sampling the engine lacks is refused
  rather than served as secret greedy output;
- **overflow-safe accounting** — byte, shape and offset math is
  checked-add throughout, so a corrupt header cannot turn into an overflowed
  allocation;
- **immutable, verified model imports** — pinned revisions, per-file SHA-256,
  temporary staging with atomic publication, read-only stores; original
  weights are never mutated ([`MANIFEST_V1.md`](MANIFEST_V1.md));
- **exact, sourced telemetry** — per-expert routing traces, logical vs
  process-visible vs physical-device byte counters, and a reuse-distance
  trace simulator; Windows and Linux counters are reported with their scope,
  never conflated;
- **admission planning** — a capacity gate that refuses a model whose working
  set cannot fit the live machine before anything downloads
  ([`REFERENCE_PLAN.md`](REFERENCE_PLAN.md));
- **external-oracle validation method** — greedy parity against independently
  built llama.cpp as a falsification instrument, with per-milestone evidence
  snapshots under [`reports/`](reports/README.md).

That scaffolding is the project's transferable result: the parts a faster
engine would still need before running untrusted hundred-gigabyte models on
someone's desktop.

## Key results

What the project established, each with its evidence:

- **Independent greedy parity on a 125B MoE.** The native runtime generates
  the identical 16-token continuation as stock llama.cpp (`6fe7498`, CPU,
  same official Q8_0 checkpoint) on the pinned prompt — two independent
  implementations agreeing token for token
  ([parity report](reports/Q11C_QWEN38_LLAMA_PARITY_VALIDATION.md)).
- **Full native execution, not a demo slice.** All 48 layers end to end (36
  Gated DeltaNet + 12 QSA), MoE top-10-of-512 routing with shared expert,
  PLE n-gram block, partial RoPE, and multi-token prefill with carried
  conv/SSM/KV state.
- **Five real defects found only by the external oracle** — a sigmoid (not
  silu) DeltaNet output gate, pre-transformed `ssm_a`, `(1+w)` norm storage,
  ggml-repeat k-head tiling, and a stale-trunk bug in the greedy CLI — none
  of them visible to self-pinned tests.
- **Measured negative results kept as findings.** A bit-identical CUDA
  expert tier measured slower than the 12-thread CPU kernel and left opt-in;
  scalar chunked prefill shown to be a negative return; the near-tie parity
  deviation diagnosed (D1) as the oracle's activation quantization and
  deliberately not mirrored.
- **A working governed runtime around it.** Immutable verified imports, a
  reservation-based Safety Governor, exact sourced telemetry, three
  OpenAI-compatible serve endpoints, and a validation report per milestone.

## Current limitations

Alpha.4 proves one real dense path, one real CPU MoE streaming path and the
Qwen3.8-Flash-Next externally-parity-validated path — not the final Vastloom
research thesis. It currently has:

- **Qwen3.8-Flash-Next**: greedy parity validated margin-aware — identical
  tokens wherever the oracle's logit margin is clear; near-tie continuations
  may diverge by documented design (D1: the oracle quantizes activations to
  Q8_0 on every Q8_0 matvec while Vastloom's double reference is more
  accurate, so Vastloom does not mirror llama.cpp's rounding); measured
  ~3.3-4 s/token with trunk residency on the reference PC against
  llama.cpp ~0.3 s/token and ik_llama.cpp ~0.4 s/token on the same
  checkpoint — an honest ~10x gap whose remaining levers are expert-quant
  re-import, a ggml-SIMD default (`--ggml-backend`, measured ~2x on compute
  as an opt-in today) and a routing-aware VRAM expert tier; Q8_0-native
  trunk residency and a device-resident bit-identical expert tier
  (`--cuda-experts`, measured slower than the 12-thread CPU kernel on
  consumer GPUs and kept opt-in) are implemented; the MTP draft head and
  vision tower are not executed; sampling is greedy-only in the parity
  CLI;
- two executable pinned Hugging Face adapters plus a DeepSeek metadata/tokenizer, reduced
  FP32 prefill oracle, file-backed governed Q8_0 layer, official 0731 GGUF metadata catalog and
  verified source-binding boundary; no full-checkpoint DeepSeek, MiniMax or Kimi execution adapter;
- native CUDA driver discovery, governed allocation, synchronous copies, one scalar FP32 RMSNorm
  parity kernel and quantized Q8_0/IQ1_S/IQ3_XXS matvec kernels wired into the DeepSeek graph with
  weights resident in VRAM, but no attention/dense CUDA path, no streams and no complete GPU
  inference - activations still return to the host between operations; no ROCm, Vulkan or Metal
  inference backend;
- scalar IQ1_S/IQ2_XXS/IQ3_XXS/MXFP4 expert, Q4_K global and Q8_0/Q5_K/Q6_K/BF16/F32 trunk
  reference kernels, plus bounded Q4_K/IQ2_XXS/MXFP4 row lookup and I32 router-table decoding, one
  deterministic reduced
  mixed-format layer composition and one
  separately validated real selected-expert cache/kernel bridge; an exact checkpoint-shaped range
  plan exists, but no authoritative official-payload route/layer/token execution, SIMD/fused
  production kernel or CUDA inference path;
- complete metadata-derived trunk plans and storage-type coverage plus a governed
  tile/expert-to-DeepSeek fixture consumer and reduced router-first bounded expert cache with an
  opt-in single-owner asynchronous batch; partial immutable range publication and a native
  closed-world verifier/exact trunk binder now feed a complete reduced packed trunk catalog,
  including retained F32/BF16 vectors, into one aggregate governed tensor set; a parallel
  six-global fixture transaction now feeds every global consumer from one aggregate lease, while
  the official 596,000,788-byte global payload, any official trunk payload read and the full-model
  loop remain absent;
- no default VRAM expert tier (the bit-identical `--cuda-experts` tier exists but stays opt-in after measuring slower than the 12-thread CPU kernel; pinned host buffers are not implemented); direct/unbuffered I/O is currently an
  opt-in OLMoE measurement protocol, not the production path for every adapter;
- OLMoE currently exposes final-logit probes and route telemetry, not decoding/chat;
- the scalar CPU kernels remain the default and the parity reference; production CPU speed
  requires the opt-in ggml backend (`VASTLOOM_GGML_ROOT` at configure time, `--ggml-compute` at
  run time), and the CUDA path covers Q8_0/IQ1_S/IQ3_XXS matvec only. The asynchronous
  selected-expert scheduler uses one to four bounded workers, permits one active batch per cache
  and defaults to ordinary buffered file I/O;
- device counters and a repeatable OS-cache-bypass protocol are available where the
  checkpoint has an unambiguous local-device mapping, but whole-device deltas can include
  unrelated traffic and SSD temperature/power telemetry is not implemented;
- no persistent KV reuse in the chat CLI or on the MoE stream path (roadmap Q12); the dense
  real-model serve keeps a prefix-KV session;
- no remote/networked API: the `serve` endpoint binds to 127.0.0.1 only and
  is the entire HTTP surface.

The TinyChat throughput is not representative of a large model because the complete
fixture is smaller than one megabyte and remains resident in CPU RAM.

## Research path

> Closed with the project's conclusion (October 2026); preserved below as the
> planned sequence that was not executed further.

The first production target is DeepSeek V4 Flash-class memory pressure, followed only by MoE
profiles whose runtime state fits safe host RAM and whose complete expert bundle fits one real RAM
or VRAM tier. RAM and VRAM are never added together to admit an indivisible expert. Kimi K3 is a
future metadata/capacity stress study, not an execution target for the 32 GiB RAM / 12 GiB VRAM
reference machine unless an exact profile passes that public gate. The main research metric is not
merely SSD throughput. It is:

> **How few bytes must be fetched from NVMe for each accepted token?**

The retained six-global binder and reduced reference-generation lifecycle are complete. The next
technical sequence remains incremental:

1. after explicit user authorization, use the completed bounded importer to acquire only the exact
   138,867,160-byte layer-0 trunk ranges while preserving the model-store free-space reserve;
2. bind and execute those 21 official objects through the now-verified metadata-derived reference
   configuration, then decode the official `tid2eid` route for a pinned input;
3. acquire only that route's six complete
   selected expert bundles (38,928,384 bytes for layer 0) and match the official layer output;
4. measure useful physical-device expert-read/compute overlap with declared cache state and exact
   logical, process-visible and device bytes;
5. only after explicit approval and a fresh capacity check, import and fully hash complete pinned
   shards when a later gate actually requires them;
6. connect tokenizer/chat framing and target-verified MTP speculative acceptance to the bounded
   generation lifecycle, then add CUDA dense, attention, router and packed
   expert kernels;
7. explicit VRAM/RAM/NVMe placement with pinned buffers and direct/overlapped I/O;
8. adaptive placement, predictive routing and speculative expert deduplication;
9. OpenAI-compatible local server and coding-agent integration.

Detailed acceptance gates are in [`ROADMAP.md`](ROADMAP.md).
The reference-PC working-set rule is defined in
[`REFERENCE_PLAN.md`](REFERENCE_PLAN.md#reference-pc-model-admission-gate).

## Research paper

The systems experience of this project is written up as a working paper (August
2026, not peer-reviewed):

- [`research/CONSUMER_STREAMING_INFERENCE.md`](research/CONSUMER_STREAMING_INFERENCE.md)
  — consumer streaming inference for large MoE models: governed memory
  hierarchy, measured results and negative results;
- [`research/vastloom-streaming-moe-inference.pdf`](research/vastloom-streaming-moe-inference.pdf)
  — the same paper rendered for reading and sharing.

Related feasibility notes live beside it: the
[capability-transfer analysis](research/CAPABILITY_TRANSFER_FEASIBILITY.md),
the [DeepSeek-on-SSD review](research/DEEPSEEK_SSD_REVIEW.md) and the
[speculation feasibility study](research/PHASE_A1_SPECULATION_FEASIBILITY.md).

## Prior art and research baselines

Vastloom is an independent project and currently contains no copied code from the
following projects. They remain important engineering and benchmarking references:

- [Colibri](https://github.com/JustVugg/colibri)
- [DwarfStar / DS4](https://github.com/antirez/ds4)
- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
- [DeepSeekV4SSD](https://github.com/yanun0323/deepseek_ssd) - reviewed in
  [`research/DEEPSEEK_SSD_REVIEW.md`](research/DEEPSEEK_SSD_REVIEW.md)
- [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)

**Where the state of the art stands (checked August 2026).** For plain
hierarchical CPU+GPU MoE inference this project is not the state of the art:

- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (llama.cpp fork):
  Qwen-3.8-Flash-Next supported (PR 2365), fused delta-net, tensor overrides
  and auto-fit MoE offload to VRAM with the rest streamed from RAM/NVMe under
  the page cache. This is today's fastest open option for exactly the
  "large MoE, 12 GB VRAM, rest on SSD" use case, and it is our oracle.
- [KTransformers / kt-kernel](https://github.com/kvcache-ai/ktransformers)
  (Tsinghua, 19k stars): CPU-GPU expert scheduling with hot-expert placement,
  3-tier GPU-RAM-disk prefix cache, FP8 kernels, SFT on consumer hardware.
  This is the strongest public reference for the heterogeneous-expert-
  placement problem this runtime researches.

Neither provides the Safety Governor's reservation semantics, a double
reference oracle, or the capability-transfer path (resident frontman plus
selective dormant-brain wake-up); that gap is Vastloom's research scope.
For pure competitive inference runs today, these projects are ahead of
Vastloom, and honest benchmarking must measure against them, not just
mainline llama.cpp (currently ~10x in Vastloom's disfavor on the reference
PC). A future public release must preserve all applicable notices for any code that is later
adapted rather than independently implemented.
