# Roadmap and acceptance gates

Vastloom advances only when each milestone has a measurable correctness and safety gate.
A command existing in the CLI is not enough; it must run against a validated model path.

## M0 — Research harness — complete in alpha.1

- [x] Native C++20/CMake project.
- [x] Hardware report and JSON output.
- [x] Strict model planning manifest.
- [x] MoE expert trace simulator.
- [x] LRU and static hot-set baselines.
- [x] NVMe bytes/token and PCIe traffic metrics.

**Gate:** reproducible trace results with bounded input size.

## M1 — Safe runtime core — complete in alpha.1

- [x] Atomic RAM/VRAM/staging/scratch reservations.
- [x] Safety state machine with hysteresis.
- [x] Fail-closed capacity planner.
- [x] Minimum-working-set and complete-expert RAM/VRAM admission gate.
- [x] Aggregate host-memory ceiling across RAM destinations, staging and scratch.
- [x] Bounded asynchronous I/O.
- [x] SHA-256 model verification and path confinement.
- [x] Atomic heartbeat and independent watchdog.
- [x] Strict warnings, ASan/UBSan and Linux/Windows CI definitions.

**Gate:** no oversubscription, path escape or unbounded queue under tests; independently legal
host-memory reservations cannot exceed their aggregate post-reserve ceiling.

## M2 — Native validation slice — complete in alpha.2

- [x] Bounded native model format and strict binary loader.
- [x] Real byte-level causal Transformer fixture.
- [x] CPU LayerNorm, causal attention, MLP and tied LM-head kernels.
- [x] Incremental KV cache.
- [x] Greedy and seeded top-k/temperature decoding.
- [x] Interactive streaming chat CLI and one-shot JSON mode.
- [x] Exact manifest verification before model loading.
- [x] Five deterministic end-to-end response probes.
- [x] CLI integration, corruption, malformed-file and resource-limit tests.

**Gate:** the native runtime reproduces every registered Python reference response and
refuses corrupt, truncated, oversized or unsafe inputs without leaking reservations.

## M3 — Real dense reference model — complete in alpha.3

- [x] Pin a published pretrained open-weight checkpoint revision.
- [x] Import immutable files through temporary artifacts, exact sizes and SHA-256.
- [x] Parse configuration, tokenizer metadata and Safetensors defensively.
- [x] Implement the exact tokenizer and chat template.
- [x] Implement CPU reference attention, MLP, normalization, RoPE and KV cache.
- [x] Match trusted tokenizer IDs, final logits and greedy output.
- [x] Exercise multi-turn history, EOS, sampling, context rejection and corruption.

**Gate:** the native pinned SmolLM2 path matches the registered Transformers reference,
passes full-checkpoint verification and refuses unsafe allocations.

## M4 — Small real MoE reference — complete in alpha.4

- [x] Pin `allenai/OLMoE-1B-7B-0924-Instruct` to an exact revision.
- [x] Validate exact shard sizes, hashes, tensor names, dtypes and shapes.
- [x] Implement the checkpoint-compatible tokenizer and chat template.
- [x] Implement authoritative top-k routing, routing weights and routed experts.
- [x] Compare expert selections, routing weights, final logits and tokens with a trusted
  reference.
- [x] Record token/layer/expert/bytes/latency telemetry and repeatable route traces.

**Gate:** every selected expert and registered output agrees with the reference without
requiring the complete checkpoint to reside in RAM.

## M5 — Real NVMe-to-RAM expert streaming — complete

- [x] Separate always-active trunk weights from individually addressable experts.
- [x] Add bounded expert reads, buffers, cancellation and exact byte accounting.
- [x] Implement no-cache, LRU, LFU and static-hot baselines in the real token path.
- [x] Inject short reads, missing/corrupt shards, allocation failure and I/O timeout.
- [x] Measure logical expert file bytes/token, cache hits and synchronous expert I/O time.
- [x] Add a bounded one-worker asynchronous path and measure read-service/CPU overlap.
- [x] Add exact unique-expert reuse-distance reporting.
- [x] Add process-I/O operation/byte counters with explicit platform semantics.
- [x] Record per-expert queue, ready and compute timestamps; compare fixed-route and
  ready-order scheduling in the trace simulator.
- [x] Prototype an output-identical 8 MiB gate/up plus 4 MiB down staged read without
  repacking original OLMoE weights.
- [x] Add physical-device I/O counters and a repeatable OS-cache-bypass protocol.

**Gate:** the real small MoE completes safely under a RAM cap below total checkpoint
size, with no uncontrolled paging or unreserved buffer.

The ordering and acceptance gates for the two new experiments are documented in the
[DeepSeekV4SSD review](research/DEEPSEEK_SSD_REVIEW.md). They remain research tasks,
not DeepSeek runtime claims. The OLMoE staged path passed parity and improved median
warm buffered forward time by 46.0% in three paired WSL runs. Three Windows
`direct-unbuffered-v1` runs then returned identical reference output and exactly
1,610,743,808 whole-device bytes for 1,610,612,736 logical expert bytes per run. M5 is
closed. The exact official DeepSeek checkpoint contract, reduced float main graph and one
checkpoint-shaped MTP block and reduced incremental state are now validated; executable packed
payload kernels and the speculative verification loop remain M7 work. The real 0731 metadata now
also defines a non-executing 177,795,544-byte layer-0 trunk+top-6 range plan; authoritative routing
and official-payload execution remain open.

## M6 — Exact DeepSeek inspector and architecture-faithful fixture — complete

- [x] Select and pin the exact DeepSeek V4 Flash checkpoint revision.
- [x] Parse `config.json`, tokenizer metadata and Safetensors headers without loading
  tensor payloads.
- [x] Generate a full immutable import catalog with per-shard size and SHA-256.
- [x] Map exact tensor names, shapes, dtypes and shard offsets.
- [x] Match the pinned tokenizer IDs and basic system/user/assistant chat framing.
- [x] Add bounded reduced vectors and native reference primitives for RMSNorm, learned/hash
  routing, routed SwiGLU, Hyper-Connections, base RoPE and selected-latent attention.
- [x] Build a reduced but architecture-faithful DeepSeek fixture.
- [x] Export target logits, route selections and token vectors from a trusted reference.
- [x] Record initial routing and raw/compressed attention-index traces.

**Gate:** every fixture tensor and reference token/logit/route agrees within an explicit
numerical tolerance before optimization begins.

The completed metadata half is recorded in the
[M6 DeepSeek inspection report](reports/M6_DEEPSEEK_METADATA_INSPECTION_VALIDATION.md).
Native prompt encoding is recorded separately in the
[M6 DeepSeek tokenizer report](reports/M6_DEEPSEEK_TOKENIZER_VALIDATION.md).
The bounded equation-level foundation is recorded in the
[M6 reference-operator report](reports/M6_DEEPSEEK_REFERENCE_OPERATORS_VALIDATION.md).
The completed graph gate is recorded in the
[M6 reduced-graph report](reports/M6_DEEPSEEK_GRAPH_VALIDATION.md).
It pins official commit `60d8d70770c6776ff598c94bb586a859a38244f1`; the former
“0731” label was provisional for that preview. A newer official 0731 repository now exists and
is tracked as a separate target in the [model compatibility matrix](MODEL_COMPATIBILITY.md). Its
derived `UD-IQ1_S` GGUF headers and one expert are now validated in M7; the official full
Safetensors payload and complete official-payload layer/token remain unvalidated.

## M7 — DeepSeek CPU reference inference

- [x] Bind the pinned preview Safetensors header catalog to universal tensor/expert roles and
  exact coalesced source ranges.
- [x] Parse the official 0731 `UD-IQ1_S` GGUF split headers and bind packed expert subranges to
  the universal catalog without reading payloads.
- [x] Derive bounded KV/compressor/scratch and resident-versus-streamed trunk working sets from
  retained exact GGUF metadata; require one complete expert to fit a single RAM or VRAM tier.
- [x] Pin the three exact 0731 `UD-IQ1_S` object sizes/SHA-256 values, import bounded headers or
  complete immutable shards atomically, and expose full source ranges only after complete hash
  verification plus live single-tier admission. The real 82.54 GB payload remains undownloaded.
- [x] Preserve layer identity for every `blk.N.*` trunk tensor, produce deterministic per-layer
  plans with a 64 MiB tile ceiling and validate a governed single-tile payload transaction on a
  fault-injected fixture. No official weight payload was read.
- [x] Match pinned ggml Q8_0/Q5_K/Q6_K storage math, add bounded BF16/F32 matvec and I32 router
  decoding, and prove storage handling for all 1,193 / 6,390,088,008-byte layer-trunk tensors in
  the real metadata. No official trunk payload was read or layer executed.
- [x] Match pinned ggml IQ1_S dequantization/matvec output on one exact real 0731 expert-gate
  slice with bounded scalar CPU code.
- [x] Match pinned ggml IQ3_XXS output and execute one complete real 0731 gate/up/down expert
  through the governed asynchronous loader with bit-identical final output.
- [ ] Consume the verified full GGUF source binding through packed IQ/FP4/FP8 trunk/expert
  kernels in one complete layer/token probe.
- [x] Bounded operator primitives for normalization, base RoPE, selected-latent attention,
  learned/hash routing, routed SwiGLU and Hyper-Connections.
- [x] Reduced float main graph with YaRN, overlapping compression, learned indexer,
  shared expert, Hyper-Connection head and official logit/route parity.
- [x] Checkpoint-compatible tokenizer and basic system/user/assistant chat template.
- [ ] Thinking/tool-call encoding and completion parsing.
- [x] Transactional raw/compressed incremental KV/state path with five-step official parity.
- [x] Bounded autoregressive reference generation with exact eight-token greedy parity, portable
  seeded top-k sampling, explicit finish reasons, per-layer cancellation/timeout checkpoints and
  observed expert-cache telemetry. This is reduced-fixture execution, not official GGUF generation.
- [x] Native single-block MTP reference path with official logit/route/index parity.
- [x] Reduced quality fixture runner.
- [ ] Full-checkpoint probe and regression corpus.

**Gate:** end-to-end tokens, logits and expert choices match the reference implementation
on the reduced fixture and selected full-model probes.

The completed reduced MTP slice is recorded in the
[M7 MTP validation report](reports/M7_DEEPSEEK_MTP_VALIDATION.md). It validates the
checkpoint-compatible component equations, not speculative decoding or full-checkpoint
execution.
The cache-state slice is recorded in the
[M7 incremental validation report](reports/M7_DEEPSEEK_INCREMENTAL_VALIDATION.md).
The bounded token lifecycle is recorded in the
[M9 reference-generation report](reports/M9_DEEPSEEK_REFERENCE_GENERATION_VALIDATION.md).
The model-independent tensor/expert binding is recorded in the
[M7 logical catalog report](reports/M7_LOGICAL_TENSOR_CATALOG_VALIDATION.md).
The separate official 0731 GGUF header binding is recorded in the
[M7 0731 GGUF catalog report](reports/M7_DEEPSEEK_0731_GGUF_CATALOG_VALIDATION.md).
The immutable import/source gate is recorded in the
[M7 checkpoint-binding report](reports/M7_DEEPSEEK_GGUF_CHECKPOINT_BINDING_VALIDATION.md).
The bounded per-layer trunk plan and tile transaction are recorded in the
[M9 layer-trunk streaming report](reports/M9_LAYER_TRUNK_STREAMING_VALIDATION.md).
The complete scalar layer-trunk storage-type gate is recorded in the
[M9 trunk storage-kernel report](reports/M9_DEEPSEEK_TRUNK_STORAGE_KERNEL_VALIDATION.md).
The first reduced architecture-level packed Q8_0 baseline is recorded in the
[M9 packed-layer report](reports/M9_DEEPSEEK_PACKED_LAYER_VALIDATION.md).
The ownership-preserving file-backed repetition is recorded in the
[M9 governed-layer report](reports/M9_DEEPSEEK_GOVERNED_LAYER_VALIDATION.md).
The router-first bounded cache and exact access telemetry are recorded in the
[M9 selected-expert streaming report](reports/M9_DEEPSEEK_SELECTED_EXPERT_STREAMING_VALIDATION.md).
The canonical 0731 catalog-to-cache mixed-format expert bridge is recorded in the
[M9 real selected-expert report](reports/M9_DEEPSEEK_REAL_SELECTED_EXPERT_VALIDATION.md).
The bounded selected-miss batch and controlled read-service/CPU-compute overlap are recorded in
the [M9 asynchronous overlap report](reports/M9_DEEPSEEK_ASYNC_COMPUTE_OVERLAP_VALIDATION.md).
The complete reduced Q8_0/IQ1_S/IQ3_XXS layer composition is recorded in the
[M9 mixed-format layer report](reports/M9_DEEPSEEK_MIXED_FORMAT_LAYER_VALIDATION.md).
The exact non-executing checkpoint-shaped range boundary is recorded in the
[M9 layer probe plan report](reports/M9_DEEPSEEK_LAYER_PROBE_PLAN_VALIDATION.md).
The explicit-authority bounded downloader and immutable partial-artifact publication gate are
recorded in the
[M9 range artifact import report](reports/M9_DEEPSEEK_RANGE_ARTIFACT_IMPORT_VALIDATION.md).
The native closed-world verification and exact pinned-catalog trunk binding are recorded in the
[M9 range artifact binding report](reports/M9_DEEPSEEK_RANGE_ARTIFACT_BINDING_VALIDATION.md).
The aggregate governed tensor-set loader and reduced artifact-to-graph parity gate are recorded in
the [M9 trunk tensor-set report](reports/M9_DEEPSEEK_TRUNK_TENSOR_SET_VALIDATION.md).
The first real packed-matrix parity gate is recorded in the
[M7 IQ1_S kernel report](reports/M7_DEEPSEEK_IQ1S_KERNEL_VALIDATION.md).
The completed one-expert gate/up/SwiGLU/down path is recorded in the
[M7 real-expert report](reports/M7_DEEPSEEK_REAL_EXPERT_VALIDATION.md); it does not yet execute a
complete official-payload layer or token.
The capacity-only resident/streamed trunk derivation is recorded in the
[M7 working-set report](reports/M7_DEEPSEEK_WORKING_SET_VALIDATION.md).

## M8 — CUDA path for the RTX 4070 Super

- [x] Native Windows/Linux CUDA Driver API capability probe without a build-time CUDA toolkit or
  context allocation; the reference RTX 4070 Super reports compute capability 8.9 on both hosts.
- [x] Move-only bounded CUDA allocation integrated with the Resource Governor, live VRAM reserve
  recheck, transactional rollback and fail-closed cleanup accounting; a 4 MiB reference-PC smoke
  passed with 1 GiB headroom.
- [x] Toolkit-free scalar FP32 RMSNorm with bounded synchronous H2D/D2H copies, one packed
  Governor-owned device buffer and CPU-reference parity on Windows and WSL. This is a correctness
  kernel, not the dense/attention gate.
- [ ] Dense and attention/state kernels.
- [ ] Optimized packed expert kernels for the selected quantization (scalar IQ1_S/IQ3_XXS
  complete-expert oracle exists).
- [ ] Async copy streams, cancellation and error recovery.
- [ ] CPU fallback for unsupported operators.

**Gate:** CUDA output passes the same token/logit/route fixtures as CPU, with no
unbounded allocation and clean recovery from injected CUDA failures.

The completed control-plane portion is recorded in the
[M8 CUDA driver validation report](reports/M8_CUDA_DRIVER_VALIDATION.md). The milestone gate stays
open until the remaining CUDA kernels match the CPU graph fixtures. The first real operation is
recorded in the [M8 CUDA RMSNorm report](reports/M8_CUDA_RMSNORM_VALIDATION.md).

## M9 — Explicit multitier execution

- [x] Universal logical tensor/expert catalog with validated immutable source ranges.
- [x] Legal NVMe/RAM/VRAM state transitions, reservation ownership and use-lease eviction guards.
- [x] Couple a synchronous packed buffer and validated multi-range completion to `RAM_READY`.
- [x] Add bounded asynchronous multi-range transactions with exclusive transfer leases.
- [x] Feed one complete real DeepSeek 0731 expert from governed packed RAM into a scalar kernel.
- [x] Build deterministic non-routed per-layer trunk tile plans and complete a governed,
  cancellation-aware single-tile RAM load with fault cleanup.
- [x] Add transactional scalar handling for every GGML storage type used by those 1,193 trunk
  tensors, including a separate non-matvec I32 router-table path.
- [x] Bind packed tensor identities to one architecture-specific quantized DeepSeek layer
  fixture and match an independently dequantized oracle before reading official trunk payloads.
- [x] Retain those packed views through Governor-owned trunk/expert loader leases and repeat the
  layer test from bounded file-backed ranges without unaccounted payload copies.
- [x] Split execution at the router boundary and load only selected complete expert bundles;
  trace load/hit/eviction and exact logical bytes across multiple tokens.
- [x] Derive strict W1/W2/W3 bindings from canonical 0731 GGUF identities and feed one complete
  real IQ1_S/IQ3_XXS expert through the bounded selected-expert cache and scalar consumer.
- [x] Connect selected-expert futures to deterministic CPU compute and measure controlled
  read-service/compute overlap with exact oracle parity; physical official-payload overlap remains
  a later performance gate.
- [x] Compose a complete reduced layer/token with Q8_0/I32 trunk, IQ1_S W1/W3 and IQ3_XXS W2
  selected bundles through governed file-backed synchronous/asynchronous cache paths.
- [x] Define a bounded checkpoint-shaped layer-0 range plan from verified official metadata:
  21 trunk plus 18 selected-expert ranges, 177,795,544 logical bytes total. The six caller-supplied
  IDs validate layout/capacity only and do not claim authoritative routing or payload execution.
- [x] Add an explicit-authority bounded range importer with exact HTTP range/length validation,
  per-range SHA-256, crash cleanup and read-only atomic publication. The real 138,867,160-byte
  trunk plan passed dry-run only; no official payload was downloaded.
- [x] Bind `vastloom.range-artifact.v1` back to exact catalog tensors in a closed-world native
  reader, rehash all local ranges and expose exact local trunk tile views. Synthetic malformed,
  corruption and forged-binding cases pass without an official payload download.
- [x] Feed one complete bound F32 fixture range through an aggregate-budgeted trunk load and
  transactional scalar matvec. Post-read hash verification, short-read, corruption, callback,
  cancellation, reservation and retained-lease cleanup cases pass across tested toolchains.
- [x] Assemble every required non-routed range in the reduced packed catalog into one retained
  typed tensor set and execute the complete reduced graph from the artifact boundary. Missing,
  duplicate, split, wrong-role/type/source, corrupt and cancelled inputs fail transactionally.
- [x] Add retained typed F32/BF16 vector and scalar graph consumers plus an explicit canonical
  0731 layer-0 name adapter. The 21-object reduced artifact retains one lease and matches the
  prefill/decode oracle; the official payload remains unexecuted.
- [x] Derive the exact reference configuration from verified 0731 GGUF metadata and validate all
  21 official layer-0 catalog descriptors against their graph consumers without reading payload.
  The pinned metadata now supplies the 46-entry compression schedule, distinct 43-entry
  routed/shared SwiGLU schedules and every native consumer dimension; see the
  [metadata-to-graph validation](reports/M9_DEEPSEEK_METADATA_TO_GRAPH_VALIDATION.md).
- [x] Extend the zero-payload consumer contract across all 43 official main layers. The
  uncompressed/hash, 4:1 compressor/indexer and 128:1 compressor layouts validate all 1,193
  non-routed descriptors and 6,390,088,008 encoded bytes against native consumers; see the
  [all-layer catalog validation](reports/M9_DEEPSEEK_ALL_LAYER_CATALOG_VALIDATION.md).
- [x] Close the complete pinned GGUF catalog without payload access. Six global consumers,
  11,008 routed bundles and 33,024 logical expert slices now reconcile with the 1,193 layer
  tensors: 1,328 physical objects, 34,223 logical descriptors and 82,533,892,444 known tensor
  bytes partition exactly. At that zero-payload gate Q4_K/IQ2_XXS/MXFP4 remained layout-only; see the
  [checkpoint catalog validation](reports/M9_DEEPSEEK_CHECKPOINT_CATALOG_VALIDATION.md).
- [x] Add pinned-ggml Q4_K parity, transactional matvec and bounded packed-row lookup; consume
  reduced packed embedding/output matrices in prefill and incremental decode without allocating
  the full FP32 table. See the
  [Q4_K global-kernel validation](reports/M9_DEEPSEEK_Q4K_GLOBAL_KERNEL_VALIDATION.md).
- [x] Add pinned-ggml IQ2_XXS parity, transactional matvec and bounded row lookup; execute an
  alternate reduced MoE graph with IQ2_XXS W1/W3 and IQ3_XXS W2 through prefill and incremental
  decode. See the
  [IQ2_XXS expert-kernel validation](reports/M9_DEEPSEEK_IQ2XXS_EXPERT_KERNEL_VALIDATION.md).
- [x] Add pinned-ggml MXFP4 parity, transactional matvec and bounded row lookup; execute an
  alternate reduced MoE graph with IQ1_S W1/W3 and MXFP4 W2 through prefill and incremental
  decode. See the
  [MXFP4 expert-kernel validation](reports/M9_DEEPSEEK_MXFP4_EXPERT_KERNEL_VALIDATION.md).
- [x] Bind all six retained global consumers through one governed transaction without weakening
  the execution gate. Official metadata produces six exact ranges totaling 596,000,788 bytes
  with a 297,861,120-byte maximum; its importer passed dry-run only. A reduced Q8_0/F32 artifact
  retains one lease across embedding, output, norm and Hyper-Connection prefill/decode parity;
  see the [global tensor-set validation](reports/M9_DEEPSEEK_GLOBAL_TENSOR_SET_VALIDATION.md).
- [x] Acquire the bounded official layer-0 trunk (21 ranges, 138,867,160 bytes) and decode the real
  `tid2eid` route from it. Hash-routed layers select by table lookup, so the imported trunk answers
  the routing question without executing the model: token 0 resolves to experts 35, 53, 200, 222,
  245 and 254 under `official-tid2eid-verified`. That authority opened the selected-expert scope and
  imported those six complete bundles (18 ranges, 38,928,384 bytes). See the
  [authoritative routing validation](reports/M9_DEEPSEEK_AUTHORITATIVE_ROUTING_VALIDATION.md).
- [x] Execute that probe: one complete official layer runs from imported payload with the
  checkpoint's own router selecting the experts. Token 0 executes experts 254, 222, 245, 200, 53 and
  35 through Q8_0/Q5_K/Q6_K/BF16/I32/IQ1_S/IQ3_XXS/Q4_K kernels and produces 129,280 layer-0 logits
  deterministically. See the
  [official layer execution validation](reports/M9_DEEPSEEK_OFFICIAL_LAYER_EXECUTION_VALIDATION.md).
  This is one layer of 43, not the model's output, and it is not yet matched against an independent
  oracle.
- [x] Execute a complete forward pass. The 82.54 GB checkpoint is local, and all 43 layers run from
  it: 258 expert bundles resolved by the model's own routers, 1,770,651,648 expert bytes — exactly
  the predicted no-cache figure — 129,280 logits in 29.4 s with 7.17 GB resident and the host's
  8 GiB floor untouched. See the
  [complete forward validation](reports/M9_DEEPSEEK_COMPLETE_FORWARD_VALIDATION.md).
- [x] Accept text prompts. The checkpoint's vocabulary is identical to the pinned tokenizer —
  129,280 tokens index for index, 127,741 merges — so `--prompt` encodes through the implementation
  already validated against Hugging Face, and `--chat` applies the checkpoint's template.
  `"La capitale d'Italia e"` runs all 43 layers and predicts `" una"`.
- [x] Release each layer's experts when the next layer asks for its own, so expert residency is
  bounded by two layers rather than by prompt length: the same six-token prompt fell from a 13.6 GB
  expert working set to 632 MB.
- [x] Generate more than one token through the reference decode loop, reusing KV state across steps.
- [x] Pay the checkpoint verification once instead of before every prompt: a receipt of each file's
  size, timestamp and confirmed hash takes startup from 357.6 s to 0.4 s, and any mismatch forces
  the full pass again.
- [x] Thread the packed matvecs and give inference half the machine rather than a fixed six workers:
  60.3 s per token to 26 s, byte-identical output, workers drawn from a pool created once.
- [x] First quantized CUDA kernel: Q8_0 matvec in hand-written PTX, 0.5735 ms against 3.0305 ms for
  the twelve-threaded CPU kernel at 4096×4096, max error 3.8147e-06. Naive on purpose — one thread
  per row, no coalescing, matrix copied per call — so the headroom above 5.28× is large.
- [x] IQ1_S and IQ3_XXS kernels for the routed experts, which are the larger share of per-layer
  arithmetic. A 16-layer generation went from 13,300 to 2,374 ms per token — 5.60× — with identical
  text and identical expert traffic. Attention itself is still on the CPU.
- [x] Wire the GPU into the graph with weights resident in VRAM: 2,728 matvecs accelerated across a
  24-layer pass, 2.90 GB resident, identical output. Net gain is small — 24,328 ms per token to
  22,246 — because each accelerated call still costs 2.65 ms against the kernel's own 0.558 ms.
- [x] Collapse the per-matvec round trips into one driver entry. Worth 0.5%, which measured the
  assumption rather than confirming it: the round trips were not the constraint.
- [x] IQ1_S and IQ3_XXS kernels, which were indeed the 89%. IQ3_XXS carried most of the gain because
  it is the most expensive format to decode on the CPU — 9.76 ms against Q8_0's 3.04 for the same
  shape. All three kernels now match the reference at zero error rather than under a tolerance, by
  accumulating in f64 in the reference's own order.
- [x] Make the parity gates capable of failing. Built from dyadic values — a fixed 0.25 scale, a
  0.125 delta, a k/256 input — they could not distinguish f32 from f64 accumulation and passed
  kernels that were summing in the wrong precision.
- [x] Key the VRAM cache on what a matrix is rather than where it happens to live. Host addresses are
  recycled when expert buffers are released, so an address-keyed cache served one expert's weights
  under another's name, and the model's output changed between runs of the same command.
- [ ] Then the rest of the layer — attention, normalization, RoPE, Hyper-Connection — after which
  keeping activations resident on the device stops being an optimisation and becomes the only
  sensible design, because nothing would need to return to the host between operations.
- [x] Load and release layer trunks the way experts are. A bounded two-layer window took a
  full-depth pass from 7.62 GB of host RAM to 2.12 GB, and because the accelerator's ceiling is
  derived from what the preflight leaves unspent, that memory went straight to the device: 1.80 GB
  to 6.06 GB, 1,490 matvecs accelerated to 4,617, 42,466 ms per token to 30,123, identical output.
  The floor no longer scales with depth, which is the project's founding idea made operational.
- [x] Evict rather than refuse. The accelerator filled VRAM and then declined every later matrix for
  the rest of the run, sending 6,962 of 12,007 matvecs to the CPU to avoid transfers that cost a
  seventh of the work they were avoiding. Least-recently-used eviction: 11,639 ms per token from
  30,280, output unchanged.
- [x] Dense FP32 and BF16 kernels, picked by counting declines per storage type. The assumption was
  Q5_K; the measurement put FP32 third and BF16 fifth, both far cheaper to write. Declines 3,432 to
  2,375.
- [ ] Kernels for IQ2_XXS (1,092 declines), Q5_K (882), Q6_K (315) and MXFP4 (84) — all that now
  keeps work off the device.
- [ ] Cut per-token I/O, which is roughly half the remaining cost: 4.54 GB of experts plus 6.39 GB of
  trunk reloads. No expert bundle survives between decode steps, and the window reloads a whole
  149 MB layer whenever any single matrix of it is missing from VRAM.
- [ ] Count the expert loader's own read buffers in the preflight. At 32 layers it approves and the
  loader then fails on `bad_alloc`, which should have been an up-front refusal with a correct figure.
- [ ] Pinned host buffers.
- [ ] Direct/overlapped NVMe reads where supported.
- [ ] Resident-trunk and bounded streamed-trunk placement profiles.
- [ ] VRAM/RAM expert tiers.
- [ ] Bounded staging and cancellation-aware prefetch.

**Gate:** the selected target model starts within hard limits and survives injected
short reads, OOM, cancellation and storage-pressure events.

## M10 — Adaptive expert placement

- [ ] Per-layer LRU baseline in the real token path.
- [ ] Frequency, reuse-distance and transfer/compute cost model.
- [ ] Persisted workload heat with local-only privacy-safe state.
- [ ] Held-out workload evaluation to detect policy overfitting.
- [ ] Explicit wasted-byte and eviction accounting.

**Gate:** lower NVMe bytes/token than LRU on held-out traces and real prompts, without
changing target router decisions.

## M11 — Predictive prefetch

- [ ] One-layer router look-ahead.
- [ ] Transfer deadline model.
- [ ] Prefetch cancellation and stale-request handling.
- [ ] Adaptive depth based on measured SSD, PCIe and kernel timing.
- [ ] Predictor confidence and fallback policy.

**Gate:** lower GPU I/O stall and higher accepted tokens/s end-to-end after counting all
prediction overhead and wasted I/O.

## M12 — Speculative memory and expert superbatching

- [ ] Integrate the checkpoint-compatible draft/DSpark/MTP component.
- [ ] Produce future routing graphs from draft tokens.
- [ ] Batch-union and deduplicate future experts.
- [ ] Schedule one read for expert reuse across speculative tokens.
- [ ] Verify every accepted prefix with the authoritative target model.

**Gate:** benefit remains positive after draft compute, extra I/O, rejected tokens and
verification costs; output remains target-model correct.

## M13 — Local service and agent integration

- [ ] OpenAI-compatible loopback API. The point is not the protocol but the ecosystem: opencode,
  Continue, Aider, Zed and Claude Code all drive an OpenAI endpoint, so one server replaces every
  client this project would otherwise have to write.
- [ ] Streaming, cancellation, structured outputs and tool calls.
- [ ] Session/prefix KV management. This is the load-bearing piece, not the JSON. A coding agent
  resends the same long context every turn and changes only its tail; without prefix reuse each
  request re-runs the whole prompt, and no amount of kernel work compensates for that.
- [ ] Request-level resource budgets and watchdog integration.
- [ ] Coding-agent compatibility tests.

**Gate:** representative coding-agent workloads complete reliably on the reference PC,
and the server binds to loopback unless remote access is explicitly configured.

## M14 — Model and backend expansion

- [ ] Complete and execute the official DeepSeek V4 Flash 0731/DSpark profile.
- [x] Add bounded GGUF v3 metadata/tensor import with explicit GGML quantization identity.
- [ ] Laguna S 2.1 architecture adapter and expert-streaming parity.
- [ ] MiniMax-class MoE adapter.
- [ ] Kimi K3 metadata-only capacity and traffic study; do not download or execute it on the
  reference PC unless an exact profile passes the minimum-working-set admission gate.
- [ ] Vulkan and ROCm backends according to measured demand.
- [ ] Multi-NVMe and optional LAN tier experiments.

Kimi K3 remains an extreme capacity study, not a reference-PC inference target. Practical model
validation is limited to MoE profiles whose runtime state and bounded trunk tile fit safe host RAM
and whose complete indivisible expert bundle fits either one remaining RAM tier or one real VRAM
tier. The gate uses dynamic budgets derived from 32 GiB RAM and 12 GiB VRAM with host headroom
preserved; RAM and VRAM are never summed to admit one oversized expert.

### Public macOS and Metal track

The primary maintainer does not currently have physical Mac hardware. This track therefore stays
public and explicitly unvalidated until evidence is produced on Apple hardware:

- [ ] Add a GitHub-hosted macOS build/test job for the CPU-only runtime.
- [ ] Replace the conservative macOS available-memory estimate with a measured, fail-closed host
  pressure probe.
- [ ] Pass tokenizer, dense, MoE, GGUF, malformed-input and cancellation parity on Apple Silicon.
- [ ] Add a Metal capability probe and Governor-backed allocation limits.
- [ ] Implement Metal kernels incrementally against the same CPU token/logit/route fixtures.
- [ ] Validate memory pressure, thermal behavior and clean cancellation on at least one physical
  Apple Silicon Mac before describing Metal as supported.

**Gate:** macOS may be listed as CPU-supported only after its CI matrix passes. Metal may be listed
as supported only after real-device correctness, resource-safety and performance reports exist.
