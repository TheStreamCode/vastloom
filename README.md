<p align="center">
  <img src="docs/assets/vastloom-hero.png" alt="Vastloom routes giant model tensors through VRAM, system RAM and NVMe while protecting the host computer" width="100%">
</p>

<h1 align="center">Vastloom</h1>

<p align="center"><strong>Run giant models. Keep your machine yours.</strong></p>

<p align="center">
  A safety-first native runtime for open-weight models that outgrow conventional VRAM and RAM limits.
</p>

<p align="center">
  <a href="https://github.com/TheStreamCode/vastloom/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/TheStreamCode/vastloom/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/TheStreamCode/vastloom/security/code-scanning"><img alt="CodeQL" src="https://github.com/TheStreamCode/vastloom/actions/workflows/codeql.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-6ee7ff.svg"></a>
  <img alt="Status: experimental alpha" src="https://img.shields.io/badge/status-experimental_alpha-bd7bff.svg">
</p>

Vastloom treats **VRAM → RAM → NVMe** as one model-aware memory hierarchy. Its long-term
goal is practical local inference for giant sparse Mixture-of-Experts models on ordinary
consumer hardware—without uncontrolled swapping, surprise resource exhaustion or destructive
model conversion.

> [!IMPORTANT]
> **Current source — `0.4.0-alpha.4` plus unreleased DeepSeek and multitier-control slices.**
> Vastloom now runs the routed token path of the pinned 13.84 GB
> `allenai/OLMoE-1B-7B-0924-Instruct` checkpoint on CPU while keeping only the
> always-active trunk and a bounded expert cache in RAM. Native logits, top-8 routes
> and routing weights match a Transformers reference. The bounded reader now exposes
> process-I/O deltas without labeling Windows transfer counters as physical NVMe traffic,
> and trace simulation reports exact unique-expert reuse distance. The official DeepSeek
> V4 Flash revision is now pinned, its 46-shard/69,187-tensor metadata is inspectable
> from verified header snapshots, and its tokenizer/basic chat framing match the official
> reference. Bounded CPU primitives now cover normalization, routing, expert SwiGLU,
> Hyper-Connections, YaRN, compressed attention and its learned indexer. A deterministic
> two-layer FP32 prefill matches the pinned official graph for logits, greedy token, expert
> routes and attention indices. Its 69,187 tensors are now bound to a shared logical catalog:
> 11,264 main/MTP expert bundles collapse from six tensors to exactly two contiguous source
> ranges each. A governed fixed-worker pool can now reserve, queue and asynchronously load those
> generic multi-range bundles into packed RAM buffers. A bounded native GGUF v3
> reader now also maps the pinned 82.54 GB `UD-IQ1_S` 0731 profile from three headers into 11,008
> logical expert bundles without downloading its complete payload. One complete 6.49 MB real
> layer-0/expert-0 bundle now passes bit-for-bit gate/up IQ1_S, SwiGLU and down IQ3_XXS parity
> against pinned ggml after canonical GGUF binding and a governed one-entry router-first RAM cache;
> its first resolution reads exactly 6,488,064 logical bytes and its second is a zero-read hit.
> The exact three-object GGUF distribution now has a bounded immutable importer and a fail-closed
> source-binding command:
> only a fully hashed payload that passes the single-tier expert gate can expose executable source
> ranges. Bounded scalar Q8_0, Q5_K, Q6_K, BF16 and F32 reference kernels plus an I32 router-table
> decoder now cover the storage types of all 1,193 per-layer trunk tensors in the real metadata.
> The reduced DeepSeek graph now consumes 11 Q8_0 trunk/shared matrices, eight IQ1_S routed
> gate/up matrices, four IQ3_XXS routed down matrices and its hash router as I32. Prefill and
> incremental main/MTP logits, routes and attention indices match the independently dequantized
> packed oracle; the original official FP32 gate remains unchanged. A second pass reads the same
> 8,180-byte payload from a Unicode file as 12 bounded trunk tiles and four indivisible
> mixed-format expert bundles. The graph retains their 16 Governor-backed
> immutable buffers directly, blocks eviction while in use and releases every reservation after
> execution. The reduced MoE graph can now stop after authoritative routing, deduplicate selected
> IDs and resolve only those complete file-backed bundles through a two-entry governed RAM LRU.
> Three one-token steps prove a 2-miss load, a 2-hit reuse and a 2-eviction replacement while
> preserving packed-oracle logits, routes, attention indices and transactional state.
> An opt-in bounded batch now submits all selected misses, acquires each expert only at its
> deterministic compute turn and preserves the same oracle while a controlled fixture records a
> positive read-service/CPU-compute intersection. The real 6.49 MB expert passes that asynchronous
> future/cache/kernel boundary too; a single real expert cannot by itself prove useful overlap.
> A new plan-only checkpoint command now combines one exact 0731 layer trunk with exactly six
> structurally validated complete expert bundles: layer 0 requires 177,795,544 logical bytes in
> 39 immutable source ranges, with a 35,651,584-byte maximum range and a 6,488,064-byte
> single-expert tier floor. The supplied expert IDs are not yet an authoritative `tid2eid` route,
> and none of these payload bytes was downloaded or executed.
> A separate bounded range importer can now validate that plan, enforce an explicit download flag,
> request exact revision-pinned HTTP spans, hash each result and atomically publish a read-only
> partial artifact under `models/optimized`. The real 138,867,160-byte layer-0 trunk passed only a
> no-network dry-run; selected-expert imports remain blocked until routing authority is verified.
> The native verifier now treats that artifact as a closed file set, rehashes every range and binds
> it back to the exact regenerated catalog. A synthetic complete four-byte F32 trunk range then
> passes an aggregate-budgeted RAM/staging transaction, a second post-read SHA-256 check, retained
> residency and a scalar matvec. The same boundary now accepts only complete closed tensor sets:
> all ranges are preflighted, aggregate RAM plus maximum-range staging is reserved once, each
> packed subspan is rehashed, and `RAM_READY` is published only after the whole set passes. The
> reduced mixed-format graph now consumes all 21 layer-0 trunk objects from one 2,844-byte
> aggregate artifact buffer: 11 matrices, nine retained F32 vectors and one I32 route table are
> published under canonical `blk.0.*` GGUF identities and mapped explicitly to their graph
> consumers. Four indivisible IQ1_S/IQ3_XXS expert bundles remain separately governed; prefill and
> decode still match the independent oracle. The exact verified 0731 metadata now directly
> constructs the native 43-layer reference configuration, retains the 46-value compression and
> separate 43-value routed/shared SwiGLU schedules, and validates every official layer-0 trunk
> descriptor against its graph consumer without reading payload. That contract now extends over
> all 43 main layers: 1,193 non-routed descriptors and 6,390,088,008 encoded bytes match the native
> uncompressed/hash, 4:1 compressor/indexer and 128:1 compressor consumers. A second zero-payload
> closure now validates the six global consumers and all 11,008 expert bundles / 33,024 logical
> expert slices. The complete partition is exact: 1,328 physical tensors become 34,223 logical
> descriptors accounting for 82,533,892,444 known tensor bytes. Q4_K now has a scalar
> pinned-ggml oracle, transactional matvec and bounded row lookup; reduced packed embedding and
> output-head prefill/decode match their dequantized graph oracle. IQ2_XXS now has the same
> bounded scalar/transactional boundary, and a reduced graph with every routed gate/up matrix in
> IQ2_XXS matches independently dequantized prefill and decode. MXFP4 now has the same bounded
> decoder, matvec and row-lookup boundary; a second reduced graph with every routed down matrix in
> MXFP4 also matches independently dequantized prefill and decode.
> The six official global tensors now have their own deterministic plan and immutable artifact
> scope. Verified metadata closes at 596,000,788 bytes in six ranges, led by two 297,861,120-byte
> Q4_K matrices. A reduced Q8_0/F32 fixture loads the same six logical consumers into one
> aggregate Governor-backed buffer and retains exactly one lease across embedding, output,
> normalization and Hyper-Connection prefill/decode parity. The official range set passed a
> no-network dry-run only; none of its payload was downloaded or executed.
> The reduced graph now also has a bounded native autoregressive lifecycle. Its eight greedy
> tokens match a freshly regenerated hash-pinned official `model.py` oracle exactly; seeded top-k
> sampling is portable across the tested standard libraries. Per-layer checkpoints enforce
> timeout, cancellation and `ProcessSafetyGuard` sampling without publishing partial cache state,
> while observed cache hit/miss and logical expert bytes are aggregated across generation.
> The first CUDA control-plane slice is also native: `cuda-info` loads the Windows/Linux driver
> dynamically without requiring the CUDA toolkit, reports exact device capability without creating
> a context, and can perform an explicitly requested Governor-backed smoke allocation capped at
> 64 MiB. On the reference RTX 4070 Super, Windows and WSL both reported compute capability 8.9
> and completed/released a 4 MiB allocation while preserving 1 GiB of live VRAM headroom.
> A second explicit smoke now reserves one packed 81,920-byte device buffer, synchronously copies
> two 4,096-wide FP32 rows plus their weights to the GPU, launches a toolkit-free scalar PTX
> RMSNorm, copies the result back and compares it with the existing DeepSeek CPU reference.
> Windows and WSL both passed at `0.000025` maximum absolute error and released all VRAM.
> The complete 82.54 GB payload
> has not been downloaded or executed; this synthetic gate is not a complete official DeepSeek
> layer/token path,
> physical-NVMe performance, official-payload generation or complete CUDA inference implementation.

## Why Vastloom exists

Most runtimes ask whether a model fits. Vastloom asks what each tensor is, when it is
needed and which memory tier can serve it safely. The Safety Governor has authority over
the performance scheduler: memory is reserved before allocation, queues and I/O buffers
are bounded, original weights remain immutable, and emergency pressure aborts generation
instead of sacrificing the host.

```text
verified model → safe hardware plan → model-aware placement → bounded inference → telemetry
                     VRAM ↔ RAM ↔ NVMe
```

This is real systems research, not a claim that the full DeepSeek V4 Flash model already runs.
The current vertical slices prove native dense inference, real MoE routing, bounded expert
streaming on a smaller published checkpoint and one complete real DeepSeek expert before the
runtime scales the new selected-expert layer path to verified official payloads and generation.

Measured results and limitations are recorded in the
[`Alpha 4 OLMoE report`](docs/reports/ALPHA4_OLMOE_VALIDATION.md) and the
[`Alpha 4.2 async-I/O report`](docs/reports/ALPHA4_ASYNC_IO_VALIDATION.md). The newest
reader, counter and reuse-distance results are in the
[`Alpha 4.3 I/O-observability report`](docs/reports/ALPHA4_IO_OBSERVABILITY_VALIDATION.md).
The rebrand, compatibility, packaging and open-source readiness checks are recorded in the
[`Alpha 4 open-source validation report`](docs/reports/ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md).
The newest staged-read and readiness results are in the
[`M5 staged-I/O validation report`](docs/reports/M5_READINESS_STAGED_IO_VALIDATION.md).
The completed physical-counter and cache-bypass gate is documented in the
[`M5 device-I/O validation report`](docs/reports/M5_DEVICE_IO_CACHE_BYPASS_VALIDATION.md).
The first DeepSeek-specific slice is documented in the
[`M6 metadata inspection report`](docs/reports/M6_DEEPSEEK_METADATA_INSPECTION_VALIDATION.md).
Native prompt encoding is documented in the
[`M6 tokenizer validation report`](docs/reports/M6_DEEPSEEK_TOKENIZER_VALIDATION.md).
The first bounded mathematical primitives are documented in the
[`M6 reference-operator report`](docs/reports/M6_DEEPSEEK_REFERENCE_OPERATORS_VALIDATION.md).
The completed reduced graph gate is documented in the
[`M6 reduced-graph validation report`](docs/reports/M6_DEEPSEEK_GRAPH_VALIDATION.md).
The checkpoint-shaped MTP reference gate is documented in the
[`M7 MTP validation report`](docs/reports/M7_DEEPSEEK_MTP_VALIDATION.md).
The five-step raw/compressed cache gate is documented in the
[`M7 incremental-state report`](docs/reports/M7_DEEPSEEK_INCREMENTAL_VALIDATION.md).
The model-independent role/range binding is documented in the
[`M7 logical tensor catalog report`](docs/reports/M7_LOGICAL_TENSOR_CATALOG_VALIDATION.md).
The reservation-backed NVMe/RAM/VRAM control plane is documented in the
[`M9 residency-state report`](docs/reports/M9_RESIDENCY_STATE_VALIDATION.md).
The first governed multi-range-to-packed-buffer transaction is documented in the
[`M9 packed expert load report`](docs/reports/M9_PACKED_EXPERT_LOAD_VALIDATION.md).
The bounded shared worker/queue transaction is documented in the
[`M9 asynchronous expert-load report`](docs/reports/M9_ASYNC_EXPERT_LOAD_VALIDATION.md).
The official 0731 GGUF header and packed-expert binding is documented in the
[`M7 0731 GGUF catalog report`](docs/reports/M7_DEEPSEEK_0731_GGUF_CATALOG_VALIDATION.md).
The first real quantized matrix parity gate is documented in the
[`M7 IQ1_S kernel report`](docs/reports/M7_DEEPSEEK_IQ1S_KERNEL_VALIDATION.md).
The first complete real quantized expert and governed-load gate is documented in the
[`M7 real-expert report`](docs/reports/M7_DEEPSEEK_REAL_EXPERT_VALIDATION.md).
The derived resident/streamed trunk capacity gate is documented in the
[`M7 working-set report`](docs/reports/M7_DEEPSEEK_WORKING_SET_VALIDATION.md).
The first model-independent per-layer trunk plan and governed tile transaction are documented in
the [`M9 layer-trunk streaming report`](docs/reports/M9_LAYER_TRUNK_STREAMING_VALIDATION.md).
The complete layer-trunk storage-type kernel gate is documented in the
[`M9 trunk storage-kernel report`](docs/reports/M9_DEEPSEEK_TRUNK_STORAGE_KERNEL_VALIDATION.md).
The first architecture-level packed layer is documented in the
[`M9 packed-layer report`](docs/reports/M9_DEEPSEEK_PACKED_LAYER_VALIDATION.md).
Its file-backed, Governor-owned execution is documented in the
[`M9 governed-layer report`](docs/reports/M9_DEEPSEEK_GOVERNED_LAYER_VALIDATION.md).
Its router-first bounded selected-expert execution is documented in the
[`M9 selected-expert streaming report`](docs/reports/M9_DEEPSEEK_SELECTED_EXPERT_STREAMING_VALIDATION.md).
The canonical real-expert catalog/cache/kernel bridge is documented in the
[`M9 real selected-expert report`](docs/reports/M9_DEEPSEEK_REAL_SELECTED_EXPERT_VALIDATION.md).
The bounded router-to-future-to-compute scheduler and controlled overlap result are documented in
the [`M9 asynchronous overlap report`](docs/reports/M9_DEEPSEEK_ASYNC_COMPUTE_OVERLAP_VALIDATION.md).
The complete reduced Q8_0/IQ1_S/IQ3_XXS layer composition is documented in the
[`M9 mixed-format layer report`](docs/reports/M9_DEEPSEEK_MIXED_FORMAT_LAYER_VALIDATION.md).
The first exact checkpoint-shaped range plan and its explicit non-execution/routing boundary are
documented in the
[`M9 layer probe plan report`](docs/reports/M9_DEEPSEEK_LAYER_PROBE_PLAN_VALIDATION.md).
The bounded immutable range importer and its real-metadata no-network gate are documented in the
[`M9 range artifact import report`](docs/reports/M9_DEEPSEEK_RANGE_ARTIFACT_IMPORT_VALIDATION.md).
The closed-world native verifier and exact pinned-catalog trunk binding are documented in the
[`M9 range artifact binding report`](docs/reports/M9_DEEPSEEK_RANGE_ARTIFACT_BINDING_VALIDATION.md).
The first governed bound-range-to-scalar transaction is documented in the
[`M9 bound trunk-load report`](docs/reports/M9_DEEPSEEK_BOUND_TRUNK_LOAD_VALIDATION.md).
The complete reduced artifact tensor-set transaction and graph parity gate are documented in the
[`M9 trunk tensor-set report`](docs/reports/M9_DEEPSEEK_TRUNK_TENSOR_SET_VALIDATION.md).
The verified GGUF metadata-to-reference-config bridge and 21-descriptor layer-0 gate are documented
in the
[`M9 metadata-to-graph report`](docs/reports/M9_DEEPSEEK_METADATA_TO_GRAPH_VALIDATION.md).
The complete 43-layer/1,193-descriptor metadata-to-consumer contract is documented in the
[`M9 all-layer catalog report`](docs/reports/M9_DEEPSEEK_ALL_LAYER_CATALOG_VALIDATION.md).
The full global/trunk/expert physical and logical closure is documented in the
[`M9 checkpoint catalog report`](docs/reports/M9_DEEPSEEK_CHECKPOINT_CATALOG_VALIDATION.md).
The first packed global-weight execution slice is documented in the
[`M9 Q4_K global-kernel report`](docs/reports/M9_DEEPSEEK_Q4K_GLOBAL_KERNEL_VALIDATION.md).
The alternate packed gate/up execution slice is documented in the
[`M9 IQ2_XXS expert-kernel report`](docs/reports/M9_DEEPSEEK_IQ2XXS_EXPERT_KERNEL_VALIDATION.md).
The alternate packed down-projection execution slice is documented in the
[`M9 MXFP4 expert-kernel report`](docs/reports/M9_DEEPSEEK_MXFP4_EXPERT_KERNEL_VALIDATION.md).
The complete governed six-global plan, artifact and consumer boundary is documented in the
[`M9 global tensor-set report`](docs/reports/M9_DEEPSEEK_GLOBAL_TENSOR_SET_VALIDATION.md).
The bounded token-level prefill/decode loop and trusted greedy/seeded vectors are documented in
the [`M9 reference-generation report`](docs/reports/M9_DEEPSEEK_REFERENCE_GENERATION_VALIDATION.md).
The native driver capability and governed VRAM allocation boundary are documented in the
[`M8 CUDA driver report`](docs/reports/M8_CUDA_DRIVER_VALIDATION.md).
The pinned importer and immutable full-source admission boundary are documented in the
[`M7 checkpoint-binding report`](docs/reports/M7_DEEPSEEK_GGUF_CHECKPOINT_BINDING_VALIDATION.md).
Current and candidate model support is kept explicit in the
[`model compatibility matrix`](docs/MODEL_COMPATIBILITY.md).

## Reference target

The first consumer reference machine is:

- AMD Ryzen 9 7900X, 12 cores / 24 threads
- NVIDIA GeForce RTX 4070 Super, 12 GiB VRAM
- 32 GiB DDR5 RAM
- 2 TiB NVMe SSD
- Windows 11, with Linux supported for development and benchmarking

The engine is not hard-coded for that PC. Hardware is probed at runtime and every
allocation plan is derived from detected capacity and conservative headroom.

### Platform status

| Platform | Current status |
|---|---|
| Windows 11 x86-64 | Native CPU paths plus CUDA driver capability/governed allocation validated with MSVC; primary consumer target |
| Linux x86-64 | Native CPU paths plus CUDA driver capability/governed allocation validated with GCC and Clang; ASan/UBSan and focused TSan binaries pass |
| macOS | Source includes a portable CPU/RAM probe; CPU CI and Metal are explicit unvalidated [M14 roadmap](docs/ROADMAP.md#public-macos-and-metal-track) work |

Vastloom must therefore be described as working on Windows and Linux today. macOS is a planned
portability target, not a validated platform claim.

## What works in the current source

### Real pretrained model path

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
  reservations, live memory-headroom checks, Windows Job Object cap and at most six
  CPU inference threads.

### Real MoE and NVMe-to-RAM path

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

### Native CUDA control boundary

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

### Synthetic validation path retained

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

### DeepSeek V4 Flash inspection and reference foundation

- **Pinned preview identity:** `deepseek-ai/DeepSeek-V4-Flash` revision
  `60d8d70770c6776ff598c94bb586a859a38244f1` remains the oracle for the evidence below.
  The newer official `DeepSeek-V4-Flash-0731` release is tracked separately and has not
  inherited those validation claims.
- **No payload required:** a 19,404,391-byte metadata import contains the config,
  tokenizer, index and 7,663,144 bytes of exact Safetensors header snapshots instead of
  159,617,149,040 bytes of weight shards.
- **Exact native catalog:** `deepseek-inspect` verifies 46 shard identities, 69,187
  tensor names/mappings, shapes, offsets, dtype totals and the 150,592,290,816-byte
  routed-expert layout.
- **Universal expert identity:** the pinned layout now produces 11,264 logical main/MTP
  expert bundles of 13,369,344 bytes. Exact adjacency safely coalesces their 67,584 tensors
  into 22,528 bounded source ranges, two per expert.
- **Native prompt path:** the exact Sequence/Split/ByteLevel tokenizer, all 1,283
  added-token definitions and basic system/user/assistant chat framing match the pinned
  Hugging Face tokenizer across multilingual vectors.
- **Bounded CPU equations:** deterministic native vectors cover RMSNorm, learned and hash
  routing, routed-expert SwiGLU, Hyper-Connection mixing, base RoPE and selected-latent
  sparse attention, with strict shape/finite checks and an element ceiling.
- **Architecture-level prefill:** a two-layer FP32 fixture exported from the pinned official
  `model.py` exercises a hash router, learned router, shared expert, YaRN, overlapping 4:1
  compression and learned attention indexer. Native logits, greedy token, all routes and
  attention indices pass the declared parity tolerance.
- **Checkpoint-shaped MTP reference:** the same oracle now executes one native MTP block,
  including embedding/hidden preprocessing, Hyper-Connections, learned MoE, sparse attention
  and its own HC output head. Its logits, proposed token, routes and indices match the official
  graph, but no speculative verification loop exists yet.
- **Incremental cache parity:** five official decode steps match for main/MTP logits, tokens,
  routes and indices while exercising the circular raw window, independent attention/indexer
  caches and a newly completed overlapping 4:1 compressed entry. State publication is
  transactional and total cache size is rejected before allocation when over budget.
- **Bounded reference generation:** the native driver now composes prefill and incremental state
  into an eight-token greedy continuation that exactly matches a freshly regenerated pinned
  `model.py` oracle. Seeded top-k sampling uses portable `mt19937_64` bits; stop/context/max-token,
  timeout and cancellation finish reasons are explicit. Checkpoints run between layers and before
  state publication, and generation telemetry sums only observed expert-cache accesses.
- **Large-download interlock:** a full import of at least 20 GiB requires the additional
  `--allow-large-download` flag after storage headroom has been reviewed.
- **Honest boundary:** the full-checkpoint inspector still reports
  `execution_supported=false`. The reduced graph now executes 11 Q8_0 trunk/shared matrices,
  eight IQ1_S routed W1/W3 matrices and four IQ3_XXS routed W2 matrices in its first main layer,
  plus an I32 hash table; remaining fixture tensors stay FP32. The packed layer binds file-backed
  trunk/expert loader buffers without payload copies into the graph. Its router-first provider
  loads only deduplicated selected complete mixed-format bundles through a bounded RAM cache and
  emits exact access telemetry. Canonical 0731 GGUF identities can feed one real
  IQ1_S/IQ3_XXS selected expert
  through that same cache/consumer boundary with exact miss/hit accounting and retained leases.
  An optional one-to-four-worker batch submits all selected misses and acquires each result at its
  fixed numerical compute turn. A controlled reduced fixture proves a positive read-service/
  compute intersection without changing the oracle, and the real expert passes the same async
  boundary. This is scheduling correctness, not a physical-device speed result.
  A verified complete official-payload layer/token consumer, official-payload generation,
  target-verified speculative generation, useful official-payload overlap and CUDA inference
  kernels remain future gates.
- **Official 0731 GGUF catalog:** the separate Unsloth revision
  `fbbb5b93fb787c21338159b0af3318bb3f4d9768` now has a bounded GGUF v3 header reader and
  DeepSeek-specific packed-expert mapper. The reviewed `UD-IQ1_S` split exposes 1,328 physical
  tensors, 34,223 logical tensors and 11,008 layer/expert bundles.
- **Verified metadata-to-graph contract:** only the exact pinned manifest/spec may retain the
  bounded 46-value compression array and separate 43-value routed/shared SwiGLU clamp arrays.
  It derives the native 43-layer, width-4,096, 64-head, 256-expert/top-6 configuration and checks
  all 1,193 official non-routed layer descriptors, storage types, consumer dimensions and
  6,390,088,008 encoded bytes. The exact layouts are 21 objects for uncompressed/hash layers,
  31 for 4:1 compressor/indexer layers and 25 for 128:1 compressor layers. This validator performs
  no tensor allocation or payload read.
- **Complete checkpoint catalog closure:** the same verified headers now bind all six global
  consumers and every routed bundle/slice to exact native identities, shapes, roles and encoded
  layouts. The 1,328 physical entries partition into 1,193 non-routed layer tensors, six globals
  and 129 packed expert containers. The logical view contains 34,223 descriptors: 1,193 layer
  tensors, six globals and 33,024 slices across 11,008 bundles. Always-active 6,986,088,796 bytes
  plus routed 75,547,803,648 bytes equals 82,533,892,444 known tensor bytes. This catalog gate
  itself performs layout arithmetic only. Later parity gates now enable scalar Q4_K, IQ2_XXS and
  MXFP4: Q4_K feeds the bounded packed embedding/output path, while alternate reduced MoE graphs
  execute IQ2_XXS routed gate/up matrices or MXFP4 routed down matrices. All three formats have
  bounded row lookup and transactional matvec dispatch.
- **Governed six-global binding:** `deepseek-gguf-global-plan` derives the exact six official
  ranges from the closed catalog. The dedicated `globals` importer scope preserves the same
  explicit network authorization, byte cap, disk reserve and atomic immutable publication rules.
  Native rebinding replays the plan against the pinned source identities, and the loader reserves
  the complete packed destination plus only the largest range as staging before any allocation or
  read. The fixture binds embedding, output head, output norm and all three output
  Hyper-Connection tensors from one shared lease. The official 596,000,788-byte plan passed
  dry-run only.
- **Header-derived traffic floor:** the encoded routed experts total 75,547,803,648 bytes;
  a top-6 no-cache decode would request 1,770,651,648 logical expert bytes per token before
  caching/prefetch gains. This is not measured physical SSD traffic or a speed claim.
- **First complete real quantized expert:** exact gate and up IQ1_S slices plus the IQ3_XXS down
  slice total 6,488,064 bytes. Canonical layer/expert tensor names are converted from the shared
  catalog into strict W1/W2/W3 descriptors. A one-worker loader and the production one-entry
  router-first cache reserve and retain the complete indivisible bundle; the bounded scalar CPU
  path executes SwiGLU and matches pinned ggml for all 4,096 final outputs bit for bit. The first
  cache resolution reports one miss and 6,488,064 logical bytes, while the second reports a hit
  and zero bytes. This is one extracted official expert, not a complete official-payload layer,
  routed token or model path.
- **GGUF safety boundary:** `deepseek-gguf-inspect` reads metadata only. Header prefixes cannot
  become executable sources. The real-expert test uses three separately hash-pinned extracted
  ranges; it does not promote a header prefix or unverified full shard into an executable source.
  `deepseek-gguf-checkpoint` accepts only the pinned immutable import identity. A complete import
  must pass full-file SHA-256 and the live single-tier expert admission gate before it can expose
  executable source ranges; even then, `execution_supported=false` until a verified
  full-checkpoint layer/token consumer exists.
- **Bounded layer-trunk plan:** every non-routed `blk.N.*` tensor now keeps its exact layer
  identity. The real 0731 metadata yields 43/43 deterministic plans, 1,193 tensors/tiles and a
  35,651,584-byte largest tile under a 64 MiB cap. A governed fixture proves one immutable tile
  read and fault cleanup. Exact scalar storage handling now covers all 1,193 tensors /
  6,390,088,008 encoded bytes through Q8_0, Q5_K, Q6_K, BF16, F32 and I32 paths. The reduced
  packed layer validates their architecture boundary, but no complete official-payload layer was
  executed.
- **Checkpoint-shaped layer range plan:** `deepseek-gguf-probe-plan` joins one complete trunk plan
  to exactly top-k unique complete expert bundles without reading payload. On layer 0 the real
  metadata yields 21 trunk plus 18 expert ranges, 177,795,544 logical bytes total, a maximum range
  of 35,651,584 bytes and a 6,488,064-byte single-tier expert floor, all in the immutable second
  shard identity. The current IDs are caller-supplied and explicitly unverified; the official
  hash-route table must execute before these offsets can be treated as an authoritative token
  route.
- **Bounded partial artifact import:** `tools/import_hf_ranges.py` accepts only the pinned plan and
  model identity, defaults to a 256 MiB total cap and 300 GiB free-space reserve, requires explicit
  `--allow-range-download`, validates exact HTTP 206/content ranges and publishes per-range hashes
  atomically under `models/optimized`. Current expert/all scopes fail before network access because
  routing authority is false. The real layer-0 trunk has passed dry-run only; no official payload
  range has been downloaded.
- **Native range-artifact binding:** `range-artifact-inspect` parses the derived artifact under
  fixed manifest/range/payload bounds, rejects symlinks, traversal and unexpected files, and
  rehashes every local range by default. `deepseek-gguf-range-bind` then reproduces the trunk plan
  from hash-verified pinned metadata and matches every local byte span to its exact complete-shard
  path, size and SHA-256 identity. The result exposes offset-zero local tile views but still reports
  `complete_checkpoint_verified=false` and `execution_supported=false`; no official payload was
  acquired or executed for this gate.
- **Governed bound-range load:** one complete F32 range from the synthetic binding now reserves its
  full RAM destination and staging together, performs an exact bounded read, hashes the in-memory
  bytes again, publishes `RAM_READY` and remains protected by a retained use lease while the scalar
  matvec consumes it. Allocation refusal, early/late cancellation, callback failure, same-size
  corruption, truncation and eviction cleanup fail safely. The absolute per-load ceiling is 1 GiB
  and callers can lower it.
- **Governed trunk tensor set:** a closed set of up to 4,096 independently hashed ranges is
  preflighted before one aggregate residency transaction. Defaults cap the set at 512 MiB total
  and 64 MiB per range; absolute ceilings are 4 GiB and 1 GiB. Missing, duplicate, split,
  wrong-role/type/source or corrupt tensors fail without partial publication. The reduced packed
  catalog now binds all 21 canonical layer-0 identities from one 2,844-byte buffer and one retained
  lease: 11 matrices, nine F32 vectors and one I32 table. A fail-closed 0731 name adapter validates
  their consumer shapes, F32/BF16 vector decoding is capped at 4 MiB of transient FP32 scratch,
  and the mixed-format prefill/decode oracle remains unchanged. This remains synthetic; no
  official trunk payload reaches the graph.

### Runtime foundation

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
  `NVME → READING → RAM_READY → COPYING_TO_VRAM → VRAM_READY` states. RAM, staging and
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
- **Safety state machine:** `NORMAL → PRESSURE → CRITICAL → EMERGENCY`, with immediate
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

## Build and validate

### Linux

```bash
./scripts/build.sh
```

Or manually:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVASTLOOM_BUILD_TESTS=ON \
  -DVASTLOOM_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Windows

From PowerShell with CMake and a C++20 compiler:

```powershell
./scripts/build.ps1
```

MSVC builds use UTF-8 source handling. The Windows executable also receives the native
wide-character command line, converts it to UTF-8 and configures UTF-8 console input/output,
so accented prompts are not dependent on the legacy ANSI code page.

A CMake install also copies the validation fixture to `share/vastloom/models` beneath
the selected install prefix.

## Chat with the bundled model

Run the deterministic model self-test first:

```bash
./build/vastloom self-test \
  examples/models/vastloom-tinychat-v1.sgm
```

Start the interactive chat:

```bash
./build/vastloom chat \
  examples/models/vastloom-tinychat-v1.sgm
```

Example session:

```text
Vastloom 0.4.0-alpha.4 — TinyChat CLI
Model: vastloom/tinychat-v1 (128 token context)
Weights: SHA-256 verified
This validation model is stateless: each line is an independent prompt.

you> Quanto fa 2+2?
model> 2+2 fa 4.
```

Run one prompt without the REPL:

```bash
./build/vastloom chat \
  examples/models/vastloom-tinychat-v1.sgm \
  --prompt "Qual è la capitale d'Italia?" \
  --no-stream
```

Machine-readable invocation:

```bash
./build/vastloom chat \
  examples/models/vastloom-tinychat-v1.sgm \
  --prompt "Ciao" \
  --json
```

Interactive commands:

```text
/help
/model
/stats
/system TEXT
/reset
/exit
/quit
```

The TinyChat fixture is intentionally stateless. The CLI keeps command state and the
latest metrics, but it does not concatenate conversation history for this model.

## Chat with the real SmolLM2 checkpoint

Initialize the external store and import the pinned checkpoint. Model files remain
outside Git and are published only after size and SHA-256 validation:

```powershell
vastloom store-init C:\VastloomModels --reserve 300GiB
python tools\import_hf_model.py --model smollm2 --store C:\VastloomModels
```

Stores created before the Vastloom rename remain valid: the runtime and importer accept
the legacy marker and import manifest without renaming, rewriting or rehashing them into a
new format. New stores and imports use only the Vastloom identifiers.

Then validate and chat with the imported directory:

```powershell
$model = 'C:\VastloomModels\models\original\HuggingFaceTB-SmolLM2-135M-Instruct--12fd25f77366fa6b3b4b768ec3050bf629380bac'
vastloom model-info $model --json
vastloom self-test $model --json
vastloom chat $model
```

The real adapter supports conversation history and `/clear`, `/stats`, `/model`,
`/help` and `/exit`. Hash verification is enabled by default; `--no-verify` exists
only for repeated local development tests.

## Probe the real OLMoE checkpoint

Import the pinned MoE into the same external store. The importer requires the
configured free-space reserve, downloads with bounded buffers/rate, resumes verified
partials and publishes only after all files match their pinned SHA-256:

```powershell
python tools\import_hf_model.py --model olmoe --store C:\VastloomModels
$model = 'C:\VastloomModels\models\original\allenai-OLMoE-1B-7B-0924-Instruct--7f1c97f440f06ce36705e4f2b843edb5925f4498'
vastloom moe-inspect $model --json
vastloom moe-logits $model --text 'Hello' --cache no-cache --io async --prefetch-depth 2 --trace C:\VastloomModels\telemetry\hello.csv --json

# Opt-in staged gate/up + down pipeline
vastloom moe-logits $model --text 'Hello' --cache no-cache --io async --expert-read staged --prefetch-depth 2 --json

# Repeatable OS-cache-bypass/device-I/O measurement; never flushes global Windows cache
vastloom moe-logits $model --text 'Hello' --cache no-cache --io async --io-cache bypass --expert-read staged --prefetch-depth 2 --json
```

`moe-logits` is a bounded correctness/telemetry probe, not an OLMoE chat or generation
command. Full verification is enabled by default. After an immediately preceding full
verification, `--no-verify` may be used for repeated local probes without rehashing all
13.84 GB. Expert cache capacity is explicitly reserved and capped at 2 GiB.
Asynchronous I/O is opt-in while the baseline remains under study. Whole mode uses one
worker and one 12 MiB request per expert. `--expert-read staged` requires asynchronous
I/O and uses two bounded workers, an 8 MiB gate/up request and a 4 MiB down request without
repacking the immutable checkpoint. Both modes allow prefetch depth one or two and use a
30-second request timeout by default. `--inject-io-fault` is a test-only development option;
it must not be used for normal inference.

`--io-cache bypass` activates the `direct-unbuffered-v1` measurement protocol. It requires
`--io async --cache no-cache`, reserves both the returned buffer and an aligned direct-I/O
envelope, and refuses to run unless the checkpoint maps to a trustworthy device counter.
It does not alter or globally evict the Windows file cache.

The JSON `process_io` and `device_io` objects identify their operating-system source and
scope. On Windows,
`process_read_bytes` is process transfer accounting and `storage_read_bytes` remains
`null`; it must not be interpreted as physical SSD traffic. `device_io` is a whole-device
delta and can include concurrent activity from other processes. `trace-sim --json` reports
exact cold/reused counts, unique-expert reuse distance and, for current 19-field traces,
a bounded fixed-order versus ready-order replay.

## Inspect DeepSeek V4 Flash without downloading the weights

The M6 metadata path downloads only bounded, independently hashed metadata and header
ranges. It is the safe default for architecture work:

```powershell
python tools\import_hf_model.py --model deepseek-v4-flash --store C:\VastloomModels --metadata-only
$model = 'C:\VastloomModels\models\original\deepseek-ai-DeepSeek-V4-Flash--60d8d70770c6776ff598c94bb586a859a38244f1--metadata'
vastloom deepseek-inspect $model --json
vastloom tokenize $model --text 'Ciao!' --chat --json
```

The inspector allocates metadata structures only and never treats the snapshot as a
runnable checkpoint. Do not add `--allow-large-download` merely to try the command: the
complete checkpoint still has no Vastloom inference path.

The newer 0731 GGUF path can likewise catalog user-supplied complete files or bounded header
prefixes. For three prefixes, their original complete sizes must be supplied in split order:

```powershell
vastloom deepseek-gguf-inspect .\part-00001.gguf .\part-00002.gguf .\part-00003.gguf `
  --declared-sizes 5257664,49093726624,33440253504 `
  --revision fbbb5b93fb787c21338159b0af3318bb3f4d9768 `
  --json
```

This command emits a `metadata_only` logical catalog. It does not download shards, verify a remote
revision, read tensor payloads or authorize inference. Use only exact sizes from an independently
verified artifact manifest; a user-entered revision is reported as a label, not cryptographic
proof.

The preferred 0731 path uses the pinned importer contract. Metadata-only mode transfers exactly
5,343,584 bytes, verifies all three objects and atomically publishes a Windows-safe artifact name:

```powershell
python tools\import_hf_model.py `
  --model deepseek-v4-flash-0731-gguf-iq1s `
  --store C:\VastloomModels `
  --metadata-only

$model = 'C:\VastloomModels\models\original\deepseek-v4-flash-0731-iq1s-fbbb5b93--metadata'
vastloom deepseek-gguf-checkpoint $model --context 4096 --batch-size 1 --json
$plan = Join-Path $env:TEMP 'vastloom-deepseek-layer-0-plan.json'
vastloom deepseek-gguf-probe-plan $model `
  --layer 0 --experts 0,1,2,3,4,5 --context 4096 --batch-size 1 --json |
  Out-File -LiteralPath $plan -Encoding utf8

$globals = Join-Path $env:TEMP 'vastloom-deepseek-globals-plan.json'
vastloom deepseek-gguf-global-plan $model --context 4096 --batch-size 1 --json |
  Out-File -LiteralPath $globals -Encoding utf8

python tools\import_hf_ranges.py `
  --model deepseek-v4-flash-0731-gguf-iq1s `
  --plan $plan --store C:\VastloomModels --scope trunk --dry-run --json

python tools\import_hf_ranges.py `
  --model deepseek-v4-flash-0731-gguf-iq1s `
  --plan $globals --store C:\VastloomModels --scope globals `
  --max-total-mib 569 --dry-run --json
```

The command requires the exact model ID, 40-character revision, paths, sizes and SHA-256 values.
Raw GGUF paths are rejected. A future full import is 82,539,237,792 bytes and additionally requires
`--allow-large-download`; with the default policy it also preserves 300 GiB of free storage. Do not
run that import merely to test the CLI: the real full payload has not yet passed a Vastloom
layer/token execution gate. `--no-verify` is inspection-only and can never create an executable
source binding.

The probe command is also metadata-only. The example IDs `0..5` demonstrate exact layout and
capacity accounting; they are not the route of a real token. Its JSON therefore reports
`plan_only=true`, `execution_supported=false` and `routing_authority_verified=false`. No listed
range is fetched by this command.

The importer example is deliberately a dry-run. A real range transfer also needs
`--allow-range-download`, enforces the selected-byte cap and publishes only a non-executable
derived artifact. The native verifier/binder has passed synthetic corruption and identity tests,
but this is not authorization to add that flag: the real 138,867,160-byte transfer still requires
the user's explicit approval. UTF-8 and UTF-8-BOM plan files are accepted; UTF-16 shell
redirection is not.

The global plan is likewise metadata-only and deterministic: it contains six ranges totaling
596,000,788 bytes, with a 297,861,120-byte maximum. Its dry-run raises only the importer's selected
payload cap to 569 MiB; it still makes no network request. A real transfer requires the same
explicit approval and does not make the artifact executable.

After an explicitly authorized import, the non-executing verification sequence is:

```powershell
vastloom range-artifact-inspect $artifact --json
vastloom deepseek-gguf-range-bind $model $artifact `
  --layer 0 --experts 0,1,2,3,4,5 --context 4096 --batch-size 1 --json

vastloom deepseek-gguf-global-bind $model $globalArtifact `
  --context 4096 --batch-size 1 --json
```

The second command always performs local range hashes and has no `--no-verify` bypass. The expert
IDs must match those used to construct the plan, but remain layout-only until the official router
table runs. Successful output proves exact trunk source binding, not a runnable DeepSeek layer.

## TinyChat CLI options

```text
vastloom chat MODEL_OR_MANIFEST
    [--prompt TEXT]
    [--max-tokens N]
    [--temperature F]
    [--top-k N]
    [--seed N]
    [--timeout-ms N]
    [--system TEXT]
    [--no-stream]
    [--no-stats]
    [--no-verify]
    [--json]
```

`--json` is supported for one-shot prompts. Hash verification is enabled by default
when the input is a Vastloom manifest; `--no-verify` downgrades that to quick path and
size verification and should be used only during local development.

JSON responses include an explicit `finish_reason` (`end_of_turn`, `end_of_sequence`,
`invalid_token`, `context_length`, `max_tokens` or `timeout`). Arbitrary sampled
bytes are repaired to valid Unicode before JSON serialization.

## Reproduce the validation model

The committed model is generated by `tools/train_tiny_chat.py`. PyTorch is needed only
for fixture training, not for native inference:

```bash
python3 tools/train_tiny_chat.py
```

The tool:

1. trains the deterministic byte-level Transformer;
2. refuses to export if any registered response probe fails;
3. writes the model through a temporary file and atomic replacement;
4. writes the test vectors and exact SHA-256 manifest atomically.

The committed model SHA-256 is:

```text
9e3dd63d2df11cbbf0207dae5f87ee6afa3f5b456aacc7e0b984b515e9cff9ef
```

See [`docs/TINY_MODEL_V1.md`](docs/TINY_MODEL_V1.md) for the binary format and tensor
order.

## Other commands

```text
vastloom chat MODEL_OR_MANIFEST
vastloom model-info MODEL_OR_MANIFEST
vastloom self-test MODEL_OR_MANIFEST
vastloom doctor
vastloom cuda-info [--allocation-smoke SIZE|--rmsnorm-smoke]
    [--minimum-free-vram SIZE] [--json]
vastloom inspect MANIFEST_OR_MODEL_HEADERS [--declared-size SIZE] [--tensor-limit N]
vastloom verify MANIFEST [--full]
vastloom plan MANIFEST
vastloom trace-sim TRACE.csv
vastloom io-smoke FILE
vastloom store-init PATH [--reserve 300GiB]
vastloom store-info PATH [--reserve 300GiB]
vastloom runtime-policy [--storage PATH] [--json]
vastloom tokenize MODEL_DIRECTORY --text TEXT [--chat] [--json]
vastloom real-logits MODEL_DIRECTORY --text TEXT [--chat] [--top N] [--json]
vastloom moe-inspect MODEL_DIRECTORY [--no-verify] [--json]
vastloom deepseek-inspect MODEL_DIRECTORY [--no-verify] [--json]
vastloom deepseek-gguf-inspect SHARD... [--declared-sizes CSV] [--revision ID] [--json]
vastloom deepseek-gguf-checkpoint MODEL_DIRECTORY [--context N] [--batch-size N]
    [--no-verify] [--json]
vastloom deepseek-gguf-probe-plan MODEL_DIRECTORY --layer N --experts CSV
    [--tile-size SIZE] [--context N] [--batch-size N] [--no-verify] [--json]
vastloom deepseek-gguf-global-plan MODEL_DIRECTORY
    [--context N] [--batch-size N] [--no-verify] [--json]
vastloom range-artifact-inspect ARTIFACT_DIRECTORY [--no-verify] [--json]
vastloom deepseek-gguf-range-bind MODEL_DIRECTORY ARTIFACT_DIRECTORY --layer N --experts CSV
    [--tile-size SIZE] [--context N] [--batch-size N] [--json]
vastloom deepseek-gguf-global-bind MODEL_DIRECTORY ARTIFACT_DIRECTORY
    [--context N] [--batch-size N] [--json]
vastloom moe-logits MODEL_DIRECTORY --text TEXT [--chat] [--top N]
    [--cache no-cache|lru|lfu|static-hot] [--cache-bytes SIZE]
    [--io sync|async] [--io-cache buffered|bypass] [--expert-read whole|staged]
    [--prefetch-depth 1|2] [--io-timeout-ms N]
    [--timeout-ms N] [--trace PATH] [--no-verify] [--json]
vastloom heartbeat-write PATH
vastloom version
```

Examples:

```bash
# Non-invasive hardware report
./build/vastloom doctor --storage /path/to/model-drive

# Native CUDA capability only; does not retain a context or allocate model memory
./build/vastloom cuda-info --json

# Explicitly reserve, allocate and release a bounded 4 MiB VRAM smoke buffer
./build/vastloom cuda-info --allocation-smoke 4MiB --minimum-free-vram 1GiB --json

# Real synchronous H2D -> scalar PTX RMSNorm -> D2H CPU-parity smoke
./build/vastloom cuda-info --rmsnorm-smoke --minimum-free-vram 1GiB --json

# Inspect and fully verify a model manifest
./build/vastloom inspect examples/models/vastloom-tinychat-v1.sgm
./build/vastloom verify examples/models/vastloom-tinychat-v1.sgm --full

# Generate the verified-metadata research plan for the consumer reference PC
./build/vastloom plan \
  examples/manifests/deepseek-v4-flash.research.sgm \
  --reference-pc \
  --profile balanced

# Compare cache policies on an expert-routing trace
./build/vastloom trace-sim examples/traces/tiny.csv \
  --vram 128MiB \
  --ram 256MiB \
  --policy lru
```

## Safety invariants

Vastloom follows these rules before performance work begins:

1. Never allocate against all installed RAM or VRAM.
2. Never let independent workers infer free capacity independently.
3. Never modify original model weights during inference.
4. Never publish a converted or downloaded artifact before verification completes.
5. Never follow a manifest path outside the declared model root.
6. Never silently change model precision or router semantics.
7. Degrade prefetch, cache and speculation before aborting a generation.
8. Abort a generation before risking an OS freeze, disk exhaustion or corrupt state.

All inference paths bound model dimensions, metadata, file size, context and timeout
before allocating execution buffers. Real-model paths additionally
keep 8 GiB of current RAM availability for the host, cap themselves to half physical RAM,
require 300 GiB free-storage headroom and refuse the load if reservations fail. See
[`docs/SAFETY.md`](docs/SAFETY.md).

## Reference planning result

Using the exact-revision DeepSeek V4 Flash research profile and the target PC, the balanced
planner currently reserves:

| Budget | Planned value |
|---|---:|
| Windows/host RAM reserve | 6.40 GiB |
| Emergency RAM reserve | 2.00 GiB |
| Maximum Vastloom RAM | 23.04 GiB |
| VRAM reserve | 1.56 GiB |
| Maximum Vastloom VRAM | 10.22 GiB |
| Storage free-space reserve | 100 GiB |

These are planning outputs, not performance claims. The DeepSeek manifest is now derived
from an exact checkpoint/header catalog but remains metadata-only because it contains no official
weight payload and has no executable full-checkpoint consumer.

The official 0731 `UD-IQ1_S` GGUF catalog now also derives two batch-1, 4,096-token capacity
profiles from exact metadata. The streamed-trunk floor is 124,905,496 bytes including one complete
8,781,824-byte expert; the resident-trunk floor is 7,110,994,292 bytes. This makes the intended
low-memory contract explicit: trunk weights may remain immutable on SSD and move through bounded
tiles, but one expert is never split across RAM and VRAM. The 43 layer plans use 1,193 tiles with
a 35,651,584-byte maximum. Every layer-trunk storage type now has a bounded scalar decoder and
the floating matrix types have transactional reference matvecs. A reduced first layer now consumes
file-backed Q8_0/I32 storage while retaining the Governor leases for complete trunk tensors and
expert bundles. A router-first two-entry RAM cache now loads only the unique experts selected for
each one-token step, reports hits/misses/evictions/logical bytes and refuses an oversized
multi-token union before I/O. The reduced graph now also has a bounded autoregressive driver with
trusted greedy/seeded token parity and transactional per-layer cancellation. Official-payload
DeepSeek execution and generation are not implemented.

The exact layer-0 range plan adds 138,867,160 trunk bytes and six 6,488,064-byte expert bundles:
177,795,544 logical bytes in 39 ranges. It is a metadata result, not a download or device-traffic
measurement, and its caller-supplied expert IDs are not authoritative routing output.
The bounded range importer accepted the 21-range trunk scope in dry-run mode and confirmed the
138,867,160-byte disk plan without creating an artifact or opening a payload connection. Expert
and complete scopes remain fail-closed while routing authority is unverified.

## Current limitations

Alpha.4 proves one real dense path and one real CPU MoE streaming path with an
opt-in bounded asynchronous reader, not the final Vastloom research thesis.
It currently has:

- two executable pinned Hugging Face adapters plus a DeepSeek metadata/tokenizer, reduced
  FP32 prefill oracle, file-backed governed Q8_0 layer, official 0731 GGUF metadata catalog and
  verified source-binding boundary; no full-checkpoint DeepSeek, MiniMax or Kimi execution adapter;
- native CUDA driver discovery, governed allocation, synchronous copies and one scalar FP32
  RMSNorm parity kernel, but no attention/dense/quantized-MoE CUDA path, streams or complete GPU
  inference; no ROCm, Vulkan or Metal inference backend;
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
- no VRAM expert tier or pinned host buffers; direct/unbuffered I/O is currently an
  opt-in OLMoE measurement protocol, not the production path for every adapter;
- OLMoE currently exposes final-logit probes and route telemetry, not decoding/chat;
- expert kernels remain scalar CPU reference implementations; the asynchronous selected-expert
  scheduler uses one to four bounded workers, permits one active batch per cache and defaults to
  ordinary buffered file I/O;
- device counters and a repeatable OS-cache-bypass protocol are available where the
  checkpoint has an unambiguous local-device mapping, but whole-device deltas can include
  unrelated traffic and SSD temperature/power telemetry is not implemented;
- no persistent KV reuse between chat turns; history is currently recomputed;
- no local HTTP API yet.

The TinyChat throughput is not representative of a large model because the complete
fixture is smaller than one megabyte and remains resident in CPU RAM.

## Research path

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

Detailed acceptance gates are in [`docs/ROADMAP.md`](docs/ROADMAP.md).
The reference-PC working-set rule is defined in
[`docs/REFERENCE_PLAN.md`](docs/REFERENCE_PLAN.md#reference-pc-model-admission-gate).

## Prior art and research baselines

Vastloom is an independent project and currently contains no copied code from the
following projects. They remain important engineering and benchmarking references:

- [Colibrì](https://github.com/JustVugg/colibri)
- [DwarfStar / DS4](https://github.com/antirez/ds4)
- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
- [DeepSeekV4SSD](https://github.com/yanun0323/deepseek_ssd) — reviewed in
  [`docs/research/DEEPSEEK_SSD_REVIEW.md`](docs/research/DEEPSEEK_SSD_REVIEW.md)
- [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)

A future public release must preserve all applicable notices for any code that is later
adapted rather than independently implemented.

## Community and license

Vastloom is available under the [Apache License 2.0](LICENSE). Contributions are welcome;
start with [CONTRIBUTING.md](CONTRIBUTING.md), follow the [Code of Conduct](CODE_OF_CONDUCT.md)
and use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities. Governance,
support scope, release gates and brand assets are documented in [GOVERNANCE.md](GOVERNANCE.md),
[SUPPORT.md](SUPPORT.md), [docs/RELEASING.md](docs/RELEASING.md) and
[docs/BRANDING.md](docs/BRANDING.md).

Third-party model weights are not distributed with the project and remain subject to their
own licenses and terms.
