# Architecture

## Scope

Vastloom is a native inference runtime, not a model and not a generic swap wrapper. Its
purpose is to execute a validated model graph while controlling where each weight/state
object lives and how it moves. A small CLI is included as a reference client and
validation surface. The primary track is **Qwen3.8-Flash-Next** (`qwen4exp`): the
official unsloth Q8_0 GGUF executes end-to-end through the `qwen38-*` CLI family and
achieves greedy token parity with llama.cpp. The DeepSeek V4 Flash `UD-IQ2_XXS`
path (`deepseek-forward`) is historical; IQ1 is not an execution path. See
`README.md` and `AGENTS.md` for the current measured position and the
`docs/reports/Q11C_QWEN38_LLAMA_PARITY_VALIDATION.md` report for the parity evidence.

```text
                     Model adapter
                          │
                    execution graph
                          │
        ┌─────────────────┴─────────────────┐
        │             control plane         │
        │ profiler · planner · governor     │
        │ telemetry · policy · watchdog     │
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │               data plane          │
        │ tensor I/O · copies · kernels     │
        │ routing · KV · token verification │
        └───────────┬─────────┬─────────────┘
                    │         │
               VRAM/RAM      NVMe
```

## Control plane

The control plane may be feature-rich, but it must remain outside latency-critical
kernels. It owns:

- hardware discovery and future microbenchmarks;
- exact model metadata and compatibility checks;
- resource budgets and safety state;
- placement and prefetch policies;
- telemetry and reproducible benchmark records;
- process heartbeat and shutdown coordination.

## Data plane

The data plane must be small, bounded and predictable:

- model-specific packed tensor readers;
- bounded async I/O;
- pinned staging buffers;
- CUDA/CPU/Vulkan/ROCm/Metal kernels;
- transfer/compute overlap;
- target-model verification of speculative tokens.

No data-plane worker may allocate outside the Resource Governor.

The planner first establishes a host budget after OS and emergency reserves, then deducts bounded
KV, staging, scratch and current trunk-tile placement before assigning the optional expert cache.
A MoE plan is capacity-compatible only if this runtime state fits host RAM and one complete expert
bundle fits either the remaining RAM expert tier or one real GPU-resident tier. A resident-trunk
profile additionally budgets every always-active encoded weight; a streamed-trunk profile leaves
those immutable weights on SSD and reports the recurrent traffic. RAM, staging and scratch share
an aggregate Governor ceiling; VRAM is enforced separately and is never summed with RAM to fit one
expert.

VRAM being enforced separately does not make it free of host memory. On Windows a device allocation
carries commit charge, so matrices the accelerator keeps resident compete for the same host budget
the weights were checked against. A 43-layer pass that ignored this passed its preflight and then
failed in the expert loader with `bad_alloc`. The accelerator's residency ceiling is therefore
capped at what the preflight left unspent, less a gigabyte for the allocations it does not model —
which at full depth is close to nothing, and is one more reason the resident-trunk floor is the
constraint worth removing.

## Native CUDA driver boundary

The first M8 slice uses the CUDA Driver API through a runtime-loaded platform library. Windows
loads only the system `nvcuda.dll`; Linux loads `libcuda.so.1`. The project therefore still builds
without CUDA headers, a CUDA toolkit or `nvcc`. Capability probing calls `cuInit` and device-query
entry points only: it does not retain a primary context or allocate VRAM.

An allocation is a two-authority transaction:

```text
caller requests bytes + live VRAM reserve
  -> Resource Governor reserves VRAM bytes
  -> backend retains device primary context
  -> backend makes context current and samples live free VRAM
  -> reject if requested bytes + reserve do not fit
  -> cuMemAlloc
  -> move-only allocation owns pointer + context + reservation
  -> cuMemFree -> context release -> Governor release
```

The CLI exposes this only as an explicit smoke allocation capped at 64 MiB and applies the normal
runtime/process policy first. The reusable API remains bounded by the caller's Governor hard cap.
Backend allocation failure rolls the reservation back. Explicit release failure preserves the
handle for retry; if destruction cannot free the external allocation, Vastloom detaches the RAII
handle while deliberately leaving its Governor bytes charged and keeps the driver module alive
until process teardown. That fail-closed state prevents a second worker from treating uncertain
VRAM as available.

The M8 slice adds one bounded synchronous data-plane transaction:

```text
accounted caller input + weight
  -> reserve host output scratch
  -> reserve one packed input/weight/output VRAM buffer
  -> synchronous H2D input + weight
  -> JIT-load scalar PTX RMSNorm and synchronize
  -> synchronous D2H output
  -> finite check + CPU-reference comparison at the CLI gate
  -> free device buffer, then release Governor reservations
```

The PTX is embedded and loaded through the Driver API, so the build still needs neither CUDA
headers nor `nvcc`. The kernel deliberately assigns one thread to each row: it is a correctness
oracle, not an optimized reduction. Every byte range is overflow-checked, FP32-aligned and
non-overlapping. The public operation caps itself at 4,194,304 elements; its returned host vector
retains its scratch reservation until destruction. H2D, kernel, D2H and exception faults roll back
the allocation. If the external device release itself remains uncertain, VRAM accounting stays
charged fail-closed.

No CUDA streams, pinned host memory, asynchronous copies, attention, quantized expert kernel or
model graph exists yet. This proves the first real GPU math/copy boundary only.

## Current inference paths

> **Track boundary (October 2026).** The M5-M9 mechanism narrative that previously lived
> in this section - the DeepSeek loading, residency, graph and kernel story - is preserved
> verbatim in
> [`archive/deepseek_0731/ARCHITECTURE_NARRATIVE.md`](archive/deepseek_0731/ARCHITECTURE_NARRATIVE.md),
> where the frozen track keeps its documentation. This section now describes the path the
> runtime actually executes today.

The primary execution path is Qwen3.8-Flash-Next (`qwen4exp`) on the official unsloth Q8_0
GGUF checkpoint (6 shards, 175.3 GiB), dispatched automatically from safetensors or GGUF:

```text
embedding row (bounded single-row read)
  -> input stream expansion (hc_count branches)
  -> 48 layers, alternating 3 Gated DeltaNet + 1 QSA (full_attention_interval = 4):
       attn gated residual
         -> 36x Gated DeltaNet: causal conv1d + SiLU, L2-normalized Q/K,
            recurrent delta-rule SSM update, gated-sigmoid RMSNorm, out_proj
         -> 12x QSA: MQA indexer (pool, k-RMSNorm, top-k block selection)
            -> gather -> masked GQA -> sigmoid gate
       mlp gated residual
         -> router (top-10 of 512, norm_topk_prob) + shared SwiGLU
         -> selected routed experts (SwiGLU each), weighted combine
       -> next layer's input stream
  -> global hyper-connection mixer
  -> lm_head
  -> greedy or sampled token (stop ids, context limit, cancellation)
```

Weight serving on this path: the per-layer trunk streams from NVMe per token, or
resides Q8_0-native in RAM under an explicit RAM-headroom override; the 10 selected
experts resolve through a bounded Q8_0 LRU with concurrent top-k prefetch, optionally a
VRAM-resident tier (`--cuda-experts`); an opt-in ggml SIMD backend serves supported
matvecs (`--ggml-backend`). Partial RoPE (64 of 256 dims, theta 1e7) and the PLE n-gram
block are integrated. The measured position - greedy token parity with llama.cpp on the
same checkpoint - is recorded in the
[Q11C parity report](reports/Q11C_QWEN38_LLAMA_PARITY_VALIDATION.md); the milestone
history is the [roadmap's Q11 series](ROADMAP.md).


## TinyTransformer adapter boundary

The alpha.2 adapter defines:

- exact `SGTMV001` binary header and tensor order;
- byte-level vocabulary and prompt framing;
- pre-LayerNorm causal Transformer semantics;
- tied token/output embeddings;
- model-specific self-test vectors;
- hard format and dimension limits.

It deliberately does not generalize tensor names or quantization. The DeepSeek metadata
adapter is separate and reuses the core integrity/model-store modules; its future
execution graph will also reuse the governor and bounded I/O infrastructure.

## Memory hierarchy

```text
L0  VRAM   current graph, hot experts, hot KV, transfer buffers
L1  RAM    dense trunk, warm experts, KV spill, bounded staging
L2  NVMe   immutable weight backing, cold experts, persistent KV
```

Vastloom does not ask the OS to discover this hierarchy through page faults. Placement
is model-aware and explicit. NVMe is a backing tier, not fake RAM.

Architecture adapters map exact checkpoint tensors into the shared role/tier contract described
in [`MODEL_COMPATIBILITY.md`](MODEL_COMPATIBILITY.md). Live state and the current bounded trunk
tile receive residency before expert cache capacity; the resident-trunk profile also pins the
complete always-active catalog. Selected expert bundles move through explicit bounded
NVMe→RAM→VRAM transitions; scheduling can change readiness order but not authoritative routing
or numerical reduction order.

The shared residency control object now enforces those transitions and owns move-only Governor
reservations. It reserves the complete RAM destination plus bounded read staging before exposing
`READING`, retains RAM while reserving VRAM, and blocks eviction while pinned or protected by an
RAII use lease. A transfer lease additionally gives one worker exclusive ownership of an active
read/copy transaction and prevents concurrent failure cleanup from freeing its buffers. The
synchronous oracle and a fixed-worker, fixed-capacity asynchronous pool both connect canonical
ranges to physical packed RAM buffers, destroy scratch before ready-state publication and keep a
use lease alive until the buffer is destroyed. The router-first reduced execution loop now uses
those futures: it submits the selected misses as one bounded batch, waits at each expert's compute
boundary and records queue, service, total, acquire-wait, compute and overlap timing. A controlled
fixture proves positive overlap without changing its oracle; one real expert proves the same
future/cache/kernel ownership path. Shared direct I/O, physical-device overlap evidence and CUDA
copy events are not connected yet.

The layer-tile loader uses the same residency contract for one non-routed source range. It checks
the planned identity against the immutable catalog, reserves both destination and staging before
allocation, publishes only after a complete read and retains a use lease until the byte buffer is
destroyed. It is currently synchronous/buffered and has no official DeepSeek kernel consumer.

For TinyChat only L1 is active:

```text
RAM
├── verified FP32 weights
├── incremental K/V cache
├── bounded operator workspaces
└── generated output
```

Both weights and generation workspace require Resource Governor reservations.

For the pinned OLMoE probe, L1 and L2 are active explicitly:

```text
RAM
├── immutable always-active BF16 trunk (reserved before load)
├── bounded expert staging/cache reservation (12 MiB to 2 GiB)
├── reservation-backed whole or staged asynchronous read buffers
├── FP32 K/V cache reserved for the requested context
└── bounded activation scratch

NVMe
└── immutable expert tensors addressed by validated shard and byte offset
```

A cache hit never reads the expert again. A miss reports exactly 12 MiB of logical
expert traffic whether it uses one range or 8 MiB plus 4 MiB staged ranges. The current
implementation does not place experts in VRAM and uses
ordinary buffered file reads, so operating-system cache hits may avoid physical NVMe.
Process counters quantify operations and transfers, but do not remove this cache ambiguity.

## Model lifecycle and governor ownership

A loaded model keeps a move-only reservation for resident weights. The owning
`ResourceGovernor` must outlive that model. The CLI enforces this structurally by
storing the governor before the model, so destruction releases model reservations first.

Production adapters should preserve the same lifecycle rule for every resident tensor,
staging buffer and device allocation.

## Model adapters

Each production adapter will define:

- expected tensor names, shapes, dtypes and shard layout;
- tokenizer and chat template;
- attention/state and routed-expert graph;
- router semantics;
- supported quantization contracts;
- optional native speculative components;
- correctness fixtures and acceptance tolerances.

The core owns safety, I/O and scheduling; adapters own architecture semantics.

## Planned large-model token path

```text
current hidden state
       │
       ├── router/look-ahead predicts next working set
       │
       ├── scheduler reserves staging bytes
       │
       ├── NVMe reads run while current compute continues
       │
       ├── verified buffers move to RAM/VRAM
       │
       ├── target kernels execute exact router choices
       │
       └── accepted token updates heat, reuse and cost models
```

A predictor may influence prefetch order. It may never alter authoritative target-model
router decisions or accepted output tokens.

## Current modules

- `hardware/`: non-invasive machine probe.
- `cuda/`: dynamically loaded Windows/Linux CUDA capability probe, Governor-backed move-only VRAM
  allocation, synchronous copies, a bounded scalar FP32 RMSNorm parity kernel and the
  `CudaMatvecAccelerator`: a VRAM-resident quantized Q8_0/IQ matvec tier (used by
  `qwen38-greedy --cuda-experts` and the DeepSeek graph) with LRU eviction that prefers
  recurring weights and device-only fast paths keyed on stable weight identities.
- `planner/`: conservative capacity and placement plan.
- `safety/`: reservations, state machine and heartbeat.
- `model/`: manifest/Safetensors/GGUF parsing, imported-checkpoint and closed-world partial-range
  verification, immutable model storage, the universal logical tensor/expert catalog and governed
  residency state.
- `io/`: bounded asynchronous reads plus platform-labeled process-I/O counters.
- `research/`: native-telemetry-driven no-cache/LRU/LFU/static-hot simulation and exact
  unique-expert reuse distance.
- `inference/`: TinyTransformer, pinned SmolLM2 generation, pinned OLMoE routed CPU
  reference/streaming, the primary **Qwen3.8-Flash-Next (qwen4exp)** track - Safetensors/GGUF
  dispatcher, Q8_0-native trunk weights (`Qwen38Weight` + `linear_w`, bit-identical F32-or-Q8
  matvec seam), routed-expert Q8_0 matvecs with an O(1) LRU and concurrent prefetch, DeltaNet
  parallel decode and the `qwen38-inspect/layer/prefill/chain/greedy` command family - plus the
  DeepSeek Safetensors/GGUF inspection, the reduced router-first selected-expert reference path,
  exact non-executing DeepSeek trunk-artifact binding and complete reduced packed-catalog
  tensor-set consumption.
- `inference/ggml_compute`: opt-in delegation of CPU matrix-vector products to a pinned ggml
  checkout, enabled at configure time by `VASTLOOM_GGML_ROOT` and at run time by
  `deepseek-forward --ggml-compute` or `qwen38-greedy --ggml-backend`. The Governor charges its
  fixed pool before allocation, one thread pool is created at the Governor's thread budget, and
  both are torn down at exit. The backend is single-threaded by contract (the enabling thread
  owns all matvecs; a cross-thread call is a visible error, never a silent race). Dispatch sits
  at the existing choke points - the transactional matvec, the fused routed expert and the
  Qwen3.8 Q8_0 linear - and falls back to the scalar reference kernels call by call. The scalar
  kernels remain the default and the parity reference; a backend error is an error, never a
  silent switch.
- `cli/`: interactive and one-shot model client commands.
- `api/`: OpenAI request/response protocol with bounded parsing, and the loopback-only HTTP
  server behind `vastloom serve` (single connection at a time, SSE streaming, Governor
  refusals surfaced as HTTP 503).
- `tools/`: pinned full/metadata import plus bounded non-executable range-artifact publication.

The data plane now has device-level I/O plus a repeatable operating-system cache-bypass
protocol. DeepSeek inspection must reuse these safety and identity boundaries rather
than bypass them. Pinned host memory and asynchronous GPU transfers remain later layers.
