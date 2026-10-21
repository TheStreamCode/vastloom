# Validated capabilities

This page records what each track in the source actually does, with the pinned model identity and
the validation evidence behind it. Execution status by model profile is summarized in
[`MODEL_COMPATIBILITY.md`](MODEL_COMPATIBILITY.md); measured results and limitations are in
[`RESULTS.md`](RESULTS.md).

## Qwen3.8-Flash-Next (primary track, llama.cpp-parity validated)

- **Pinned checkpoint:** `unsloth/Qwen3.8-Flash-Next-Q8_0` GGUF (6 shards,
  175.3 GiB, SHA-256 verified) imported through the canonical immutable
  importer into the external model store; the FP8 safetensors release was
  imported first, validated against the pinned 152,089-tensor catalog, then
  removed once the GGUF replaced it.
- **Native `qwen4exp` execution:** full 48-layer forward (36 Gated DeltaNet +
  12 QSA sparse attention), multi-token prefill with carried conv/SSM/KV
  state, MoE routing (top-10 of 512, shared expert), the PLE n-gram block
  (hash -> fused 160-per-head table lookup -> gate -> dilated depthwise conv),
  partial RoPE (64/256, theta 1e7) and the global mixer + LM head.
- **Greedy token parity with llama.cpp** (rev `6fe7498`, CPU, same
  checkpoint): "The capital of France is" -> " Paris. The capital of Germany
  is Berlin. The capital of Italy is Rome. The" for 16 greedy tokens,
  identical on both engines. Stage-by-stage tensor comparison against
  llama.cpp surfaced and fixed five real defects that self-pinned tests
  could not see (sigmoid DeltaNet output gate, pre-applied `-exp(A_log)` in
  `ssm_a`, GGUF norm weights already `(1+w)`, ggml-repeat tile k-head
  mapping, and a stale-trunk bug in the greedy CLI).
- **CLI:** `vastloom qwen38-inspect` (metadata catalog validation),
  `qwen38-layer` (single real layer forward), `qwen38-prefill` (multi-token
  prefill with a bounded expert LRU), `qwen38-chain` (consecutive real
  layers, streaming trunks), `qwen38-greedy` (full-depth greedy generation
  with detokenize). The safetensors/GGUF dispatcher keeps the FP8 path
  working for historical pins.
- **Opt-in real-checkpoint test** (`-DVASTLOOM_QWEN38_MODEL=<root>`): catalog
  validation, per-layer forwards and prefill determinism on both layouts.

## Real pretrained model path

- **Pinned checkpoint:** `HuggingFaceTB/SmolLM2-135M-Instruct` at revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`, imported outside the repository.
- **Immutable import and verification:** bounded downloads, exact file sizes and
  SHA-256, temporary artifacts and atomic publication into the external model store.
- **Native tokenizer and template:** GPT-2 byte-level BPE, Unicode/Italian prompts and
  the model's exact chat framing.
- **CPU reference inference:** BF16 weights, FP32 activations, RMSNorm, RoPE,
  grouped-query causal attention, SwiGLU and tied output embeddings.
- **Generation and chat:** greedy or seeded top-k sampling, incremental KV cache,
  persistent multi-turn history, streaming, JSON, timeout and Ctrl+C cancellation.
- **Reference parity:** exact token IDs and greedy outputs, with final-token logits
  matching the pinned Transformers reference within the declared tolerance.
- **Fail-closed runtime:** full checkpoint verification by default, Resource Governor
  reservations, live memory-headroom checks, Windows Job Object cap and the CPU
  thread pool capped to half the logical cores (max 12; `runtime_policy.cpp`).

## Real MoE and NVMe-to-RAM path

- **Pinned real MoE:** `allenai/OLMoE-1B-7B-0924-Instruct` revision
  `7f1c97f440f06ce36705e4f2b843edb5925f4498`, three immutable BF16 shards and
  3,219 validated tensors.
- **Non-resident execution:** the 953,421,824-byte always-active trunk is reserved in
  RAM; the 12,884,901,888 bytes of routed experts remain on NVMe and are addressed by
  exact shard/offset identity.
- **Authoritative CPU graph:** Q/K RMSNorm, RoPE, causal attention, full 64-way router
  softmax, top-8 selection and routing-weighted SwiGLU experts.
- **Bounded expert memory:** one 12 MiB active expert, an explicitly reserved cache
  between 12 MiB and 2 GiB, and reservation-backed prefetch buffers constrained by the
  staging governor.
- **Opt-in asynchronous pipeline:** whole-expert mode uses one 12 MiB range and one worker;
  staged mode uses two workers with an 8 MiB gate/up range plus a 4 MiB down range. Both
  retain prefetch depth one or two, bounded queues, timeouts and cancellation checks.
- **Exact telemetry:** token, layer, expert, size, routing weight, tier, hit/miss,
  logical bytes read, submit/queue/ready times, compute windows, selected/reduction order
  and measured service/CPU overlap, atomically published without overwrite.
- **I/O observability:** Windows reports process read operations and transfer bytes from
  `GetProcessIoCounters`; Linux additionally reports the storage-layer `read_bytes`
  counter from `/proc/self/io`. Windows device-level telemetry maps the checkpoint volume
  to `PhysicalDriveN` and samples `IOCTL_DISK_PERFORMANCE`; native Linux accepts only a
  non-virtual `/sys/dev/block` mapping. Every source and scope remains explicit.
- **Safe cache-bypass protocol:** opt-in `direct-unbuffered-v1` reads bypass the operating
  system file cache without flushing global state. The protocol requires async no-cache
  execution, bounded aligned envelopes and a working device counter, otherwise it fails
  closed.
- **Reuse telemetry:** native traces report cold/reused accesses plus mean, P50, P95 and
  maximum exact unique-expert reuse distance.
- **Readiness replay:** `trace-sim` compares fixed-route and theoretical ready-expert
  compute order without changing the model's selected reduction order.
- **Independent parity:** one- and two-token native probes match Transformers 4.45.2
  for greedy token, top logits, every selected expert and every recorded route weight.
- **Safe preflight:** metadata-only checkpoints cannot execute, full SHA-256 is the
  default, resident loading is rejected when the checkpoint exceeds the live cap, and
  timeout/Ctrl+C/headroom checks stop safely between layers.
- **Fault injection:** worker failure, short read, allocation failure, cancellation and
  I/O timeout are exercised through the real OLMoE path and release reservations cleanly.

## Native CUDA control boundary

- **Toolkit-free capability probe:** `cuda-info` resolves the Windows/Linux CUDA Driver API at
  runtime and reports driver API version, device name, total VRAM, compute capability, SM count,
  maximum threads per block and unified-addressing support. A capability-only probe does not
  retain a CUDA context or allocate model memory.
- **Governor-first allocation:** the VRAM reservation succeeds before the CUDA backend may retain
  the primary context or call `cuMemAlloc`; live device memory is then rechecked against a
  caller-supplied reserve.
- **Transactional ownership:** move-only allocation handles retain the CUDA driver, device context
  and Governor reservation until `cuMemFree`. Backend allocation faults roll back the reservation;
  an unrecoverable cleanup failure deliberately leaves its bytes charged instead of advertising
  leaked VRAM as free.
- **Bounded real smoke:** the CLI requires an explicit `--allocation-smoke`, caps it at 64 MiB,
  applies the normal process/runtime policy and preserves at least 1 GiB of live VRAM by default.
  The reference RTX 4070 Super passed a 4 MiB allocate/release cycle through both `nvcuda.dll` and
  WSL `libcuda.so.1`.
- **First real operation:** `cuda-info --rmsnorm-smoke` reserves caller scratch and one packed
  81,920-byte VRAM buffer, performs synchronous H2D/D2H copies and runs a scalar PTX RMSNorm over
  two 4,096-wide rows. Both native Windows and WSL match the CPU reference within `3e-5` and
  release the device allocation.
- **Honest boundary:** this is a deliberately scalar correctness kernel. CUDA attention, dense and
  quantized MoE kernels, streams, pinned buffers, asynchronous overlap and GPU model inference do
  not exist yet.

## Synthetic validation path retained

- **TinyTransformer v1 adapter:** strict binary parser, bounded dimensions, exact
  tensor-layout validation and finite-weight checks.
- **Byte-level tokenizer and chat template:** deterministic UTF-8 byte encoding with
  BOS/EOS/end-of-turn tokens.
- **CPU reference kernels:** LayerNorm, causal multi-head attention, GELU MLP, residual
  connections, tied output embeddings and incremental KV caching.
- **Generation:** greedy decoding and seeded top-k/temperature sampling.
- **Streaming chat CLI:** interactive REPL, one-shot prompts, JSON output, timeouts,
  generation limits, system-prompt override and response statistics.
- **Bundled validation model:** a 358.91 KiB, two-layer causal Transformer trained on a
  small synthetic prompt/response corpus.
- **End-to-end self-test:** five deterministic Italian/English response vectors.
- **Manifest integrity by default:** `.sgm` inputs are fully SHA-256 verified before
  weights are loaded. Direct `.sgtm` files remain available for development and are
  clearly reported as unverified.

## Runtime foundation

- **Hardware probe:** CPU ISA, logical cores, RAM availability, NVIDIA VRAM and storage
  capacity.
- **Fail-closed execution planner:** produces VRAM/RAM/NVMe budgets, reserves explicit KV,
  staging and scratch placement, and rejects MoE profiles whose declared minimum working set or
  one complete expert cannot fit a real post-reserve RAM/VRAM tier.
- **Resource Governor:** atomic reservations prevent concurrent workers from
  oversubscribing RAM, VRAM, staging buffers or scratch space. RAM destinations, staging and
  scratch also share one aggregate host-memory ceiling, so their independent limits cannot add
  up beyond the safe process budget.
- **Governed residency:** logical objects move only through legal
  `NVME -> READING -> RAM_READY -> COPYING_TO_VRAM -> VRAM_READY` states. RAM, staging and
  VRAM destinations are reserved before work begins; pins and RAII use leases block eviction.
- **Packed expert baseline:** canonical source ranges can be loaded into one reserved RAM buffer
  with exact per-tensor slices through either the synchronous oracle or a fixed-worker,
  fixed-capacity asynchronous pool. The full gate/up/down path is now exercised on one real
  DeepSeek 0731 expert, including canonical catalog binding and the bounded cache/consumer bridge.
  The reduced DeepSeek layer consumes selected file-backed fixture bundles through the same cache
  abstraction. Its optional async batch overlaps later read service with earlier CPU expert
  compute in a controlled fixture while preserving reduction order; successful abandoned reads
  are drained and counted as wasted prefetch. That reduced layer now uses IQ1_S W1/W3 and
  IQ3_XXS W2 for all four routed bundles. Complete verified 0731 shards and a checkpoint-shaped
  official-payload layer are not connected.
- **Governed trunk tile:** one non-routed source tile is identity-checked against the catalog,
  destination and staging are reserved before allocation, metadata-only catalogs are refused and
  an RAII lease protects the resulting buffer through use/eviction.
- **Safety state machine:** `NORMAL -> PRESSURE -> CRITICAL -> EMERGENCY`, with immediate
  escalation and hysteretic recovery.
- **Bounded asynchronous I/O:** fixed worker, queue and request limits; every buffer
  holds a live reservation until released. Windows range reads use Unicode-safe native
  file handles and either one bounded 12 MiB range or staged 8 MiB plus 4 MiB ranges.
- **Weight integrity:** strict manifest parser, exact offline manifest generator,
  relative-path validation, file-size checks, SHA-256 verification and symlink-escape
  protection.
- **Model-store separation:** immutable originals, derived optimized weights,
  reconstructable cache, scratch, state and telemetry.
- **Independent watchdog:** observes an atomically published heartbeat and can remain
  observe-only or terminate a stale runtime only when explicitly launched with
  `--enforce`.
- **MoE trace simulator:** compares no-cache, LRU, LFU and static hot-set placement
  while reporting VRAM/RAM hit rates, modeled NVMe bytes per token, PCIe traffic and
  fixed-order versus ready-order timing.
- **Linux and Windows CI definitions**, strict compiler warnings and ASan/UBSan jobs.
