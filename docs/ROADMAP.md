# Roadmap and acceptance gates

Stargate advances only when each milestone has a measurable correctness and safety gate.
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

## M3 — Exact DeepSeek inspector and architecture-faithful fixture

- [ ] Select and pin the exact DeepSeek V4 Flash checkpoint revision.
- [ ] Parse `config.json`, tokenizer metadata and Safetensors headers without loading
  tensor payloads.
- [ ] Generate a full manifest with per-shard size and SHA-256.
- [ ] Map exact tensor names, shapes, dtypes and shard offsets.
- [ ] Build a reduced but architecture-faithful DeepSeek fixture.
- [ ] Export target logits, route selections and token vectors from a trusted reference.
- [ ] Record initial routing traces.

**Gate:** every fixture tensor and reference token/logit/route agrees within an explicit
numerical tolerance before optimization begins.

## M4 — DeepSeek CPU reference inference

- [ ] Model-specific tensor loader.
- [ ] Dense, attention/state, router and routed-expert reference kernels.
- [ ] Checkpoint-compatible tokenizer and chat template.
- [ ] Incremental KV/state path.
- [ ] Native speculative component reference path where present.
- [ ] Quality fixture runner and regression corpus.

**Gate:** end-to-end tokens, logits and expert choices match the reference implementation
on the reduced fixture and selected full-model probes.

## M5 — CUDA path for the RTX 4070 Super

- [ ] Native CUDA device capability probe.
- [ ] Bounded CUDA allocator integrated with Resource Governor.
- [ ] Dense and attention/state kernels.
- [ ] Packed expert kernels for the selected quantization.
- [ ] Async copy streams, cancellation and error recovery.
- [ ] CPU fallback for unsupported operators.

**Gate:** CUDA output passes the same token/logit/route fixtures as CPU, with no
unbounded allocation and clean recovery from injected CUDA failures.

## M6 — Explicit multitier execution

- [ ] Tensor placement table and legal state transitions.
- [ ] Pinned host buffers.
- [ ] Direct/overlapped NVMe reads where supported.
- [ ] Dense-trunk-first placement.
- [ ] VRAM/RAM expert tiers.
- [ ] Bounded staging and cancellation-aware prefetch.

**Gate:** the selected target model starts within hard limits and survives injected
short reads, OOM, cancellation and storage-pressure events.

## M7 — Adaptive expert placement

- [ ] Per-layer LRU baseline in the real token path.
- [ ] Frequency, reuse-distance and transfer/compute cost model.
- [ ] Persisted workload heat with local-only privacy-safe state.
- [ ] Held-out workload evaluation to detect policy overfitting.
- [ ] Explicit wasted-byte and eviction accounting.

**Gate:** lower NVMe bytes/token than LRU on held-out traces and real prompts, without
changing target router decisions.

## M8 — Predictive prefetch

- [ ] One-layer router look-ahead.
- [ ] Transfer deadline model.
- [ ] Prefetch cancellation and stale-request handling.
- [ ] Adaptive depth based on measured SSD, PCIe and kernel timing.
- [ ] Predictor confidence and fallback policy.

**Gate:** lower GPU I/O stall and higher accepted tokens/s end-to-end after counting all
prediction overhead and wasted I/O.

## M9 — Speculative memory and expert superbatching

- [ ] Integrate the checkpoint-compatible draft/DSpark/MTP component.
- [ ] Produce future routing graphs from draft tokens.
- [ ] Batch-union and deduplicate future experts.
- [ ] Schedule one read for expert reuse across speculative tokens.
- [ ] Verify every accepted prefix with the authoritative target model.

**Gate:** benefit remains positive after draft compute, extra I/O, rejected tokens and
verification costs; output remains target-model correct.

## M10 — Local service and agent integration

- [ ] OpenAI-compatible loopback API.
- [ ] Streaming, cancellation, structured outputs and tool calls.
- [ ] Session/prefix KV management.
- [ ] Request-level resource budgets and watchdog integration.
- [ ] Coding-agent compatibility tests.

**Gate:** representative coding-agent workloads complete reliably on the reference PC,
and the server binds to loopback unless remote access is explicitly configured.

## M11 — Model and backend expansion

- [ ] MiniMax-class MoE adapter.
- [ ] Kimi K3 capacity and traffic benchmark.
- [ ] Vulkan, ROCm and Metal backends according to measured demand.
- [ ] Multi-NVMe and optional LAN tier experiments.

Kimi K3 remains an extreme research target, not the first product milestone.
