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
- [x] Bounded asynchronous I/O.
- [x] SHA-256 model verification and path confinement.
- [x] Atomic heartbeat and independent watchdog.
- [x] Strict warnings, ASan/UBSan and Linux/Windows CI definitions.

**Gate:** no oversubscription, path escape or unbounded queue under tests.

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
checkpoint-shaped MTP block are now validated; the packed checkpoint loader, incremental state
and speculative verification loop remain M7 work.

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
is tracked as a separate, not-yet-validated target in the
[model compatibility matrix](MODEL_COMPATIBILITY.md).

## M7 — DeepSeek CPU reference inference

- [x] Bind the pinned preview Safetensors header catalog to universal tensor/expert roles and
  exact coalesced source ranges.
- [ ] Bind full payload/GGUF catalogs and packed FP4/FP8 weights to executable kernels.
- [x] Bounded operator primitives for normalization, base RoPE, selected-latent attention,
  learned/hash routing, routed SwiGLU and Hyper-Connections.
- [x] Reduced float main graph with YaRN, overlapping compression, learned indexer,
  shared expert, Hyper-Connection head and official logit/route parity.
- [x] Checkpoint-compatible tokenizer and basic system/user/assistant chat template.
- [ ] Thinking/tool-call encoding and completion parsing.
- [x] Transactional raw/compressed incremental KV/state path with five-step official parity.
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
The model-independent tensor/expert binding is recorded in the
[M7 logical catalog report](reports/M7_LOGICAL_TENSOR_CATALOG_VALIDATION.md).

## M8 — CUDA path for the RTX 4070 Super

- [ ] Native CUDA device capability probe.
- [ ] Bounded CUDA allocator integrated with Resource Governor.
- [ ] Dense and attention/state kernels.
- [ ] Packed expert kernels for the selected quantization.
- [ ] Async copy streams, cancellation and error recovery.
- [ ] CPU fallback for unsupported operators.

**Gate:** CUDA output passes the same token/logit/route fixtures as CPU, with no
unbounded allocation and clean recovery from injected CUDA failures.

## M9 — Explicit multitier execution

- [x] Universal logical tensor/expert catalog with validated immutable source ranges.
- [ ] Legal NVMe/RAM/VRAM state transitions and reservation ownership.
- [ ] Pinned host buffers.
- [ ] Direct/overlapped NVMe reads where supported.
- [ ] Dense-trunk-first placement.
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

- [ ] OpenAI-compatible loopback API.
- [ ] Streaming, cancellation, structured outputs and tool calls.
- [ ] Session/prefix KV management.
- [ ] Request-level resource budgets and watchdog integration.
- [ ] Coding-agent compatibility tests.

**Gate:** representative coding-agent workloads complete reliably on the reference PC,
and the server binds to loopback unless remote access is explicitly configured.

## M14 — Model and backend expansion

- [ ] Pin and implement the official DeepSeek V4 Flash 0731/DSpark profile.
- [ ] Add bounded GGUF metadata/tensor import with explicit quantization identity.
- [ ] Laguna S 2.1 architecture adapter and expert-streaming parity.
- [ ] MiniMax-class MoE adapter.
- [ ] Kimi K3 capacity and traffic benchmark.
- [ ] Vulkan, ROCm and Metal backends according to measured demand.
- [ ] Multi-NVMe and optional LAN tier experiments.

Kimi K3 remains an extreme research target, not the first product milestone.
