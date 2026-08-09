# Architecture

## Scope

Stargate is a native inference runtime, not a model and not a generic swap wrapper. Its
purpose is to execute a validated model graph while controlling where each weight/state
object lives and how it moves. A small CLI is included as a reference client and
validation surface.

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

## Current inference paths

Alpha.2 retains the complete synthetic CPU-only reference slice:

```text
.sgm manifest
      │ full SHA-256 verification
      ▼
SGTMV001 TinyTransformer file
      │ strict header/tensor-layout validation
      ▼
Resource Governor weight reservation
      │
      ▼
byte tokenizer + chat framing
      │
      ▼
incremental CPU Transformer
LayerNorm → causal MHA/KV → MLP
      │
      ▼
greedy or seeded top-k sampling
      │
      ▼
streaming CLI / JSON result
```

The entire validation model remains resident in RAM. This path validates the adapter,
model lifecycle, tokenizer, KV cache, sampling and client surface; it does not yet
exercise NVMe paging or MoE routing.

Alpha.3 adds a separate real-checkpoint path:

```text
atomic pinned Hugging Face import
      │ exact sizes + SHA-256
      ▼
bounded Safetensors catalog + BF16 store
      │ Resource Governor + process memory cap
      ▼
GPT-2 byte BPE + exact SmolLM chat template
      │
      ▼
native Llama CPU reference graph
RMSNorm → RoPE/GQA/KV → SwiGLU
      │
      ▼
greedy or seeded sampling
      │
      ▼
streaming / JSON / persistent-history CLI
```

This adapter is intentionally pinned to one revision of SmolLM2. It loads the verified
269 MB BF16 checkpoint into reserved RAM and uses FP32 activations. It is a correctness
oracle for subsequent backends, not a generic Llama compatibility claim.

Alpha.4.2 extends the first real sparse path with an opt-in bounded prefetch pipeline:

```text
pinned OLMoE import (3 BF16 shards, 13.84 GB tensor payload)
      │ full hash and exact 3,219-tensor catalog
      ▼
reserved 953,421,824-byte always-active trunk in RAM
      │ exact expert shard/offset table
      ▼
native CPU attention + authoritative 64-way/top-8 router
      │
      ├── sync bounded read, or one-worker async prefetch (depth 1-2)
      │        └── reservation-backed 12 MiB buffers + request timeout
      └── optional reserved RAM cache (no-cache/LRU/LFU/static-hot)
      │
      ▼
weighted SwiGLU expert accumulation → final logits
      │
      ▼
atomic token/layer/expert/tier/bytes/service/wait/overlap trace
```

The model object gives every routed expert a logical `(layer, expert)` identity that is
independent of its physical shard offset or current RAM residency. The runtime does not
map or allocate the complete checkpoint. Synchronous reads remain the default oracle.
The optional asynchronous mode validates each expert's three BF16 matrices as one
contiguous 12 MiB shard range, reserves staging before submission, and schedules the
next one or two routed experts on a single worker. It measures how much actual read
service completed while foreground CPU work continued. These timings and logical byte
counts do not substitute for physical-device counters or a cold-cache NVMe benchmark.

## TinyTransformer adapter boundary

The alpha.2 adapter defines:

- exact `SGTMV001` binary header and tensor order;
- byte-level vocabulary and prompt framing;
- pre-LayerNorm causal Transformer semantics;
- tied token/output embeddings;
- model-specific self-test vectors;
- hard format and dimension limits.

It deliberately does not generalize tensor names or quantization. The production
DeepSeek adapter will be separate while reusing the core governor, integrity and I/O
modules.

## Memory hierarchy

```text
L0  VRAM   current graph, hot experts, hot KV, transfer buffers
L1  RAM    dense trunk, warm experts, KV spill, bounded staging
L2  NVMe   immutable weight backing, cold experts, persistent KV
```

Stargate does not ask the OS to discover this hierarchy through page faults. Placement
is model-aware and explicit. NVMe is a backing tier, not fake RAM.

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
├── zero to two reservation-backed 12 MiB asynchronous read buffers
├── FP32 K/V cache reserved for the requested context
└── bounded activation scratch

NVMe
└── immutable expert tensors addressed by validated shard and byte offset
```

A cache hit never reads the expert again. A miss reports exactly 12 MiB of logical
expert traffic. The current implementation does not place experts in VRAM and uses
ordinary buffered file reads, so operating-system cache hits may avoid physical NVMe.

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
- `planner/`: conservative capacity and placement plan.
- `safety/`: reservations, state machine and heartbeat.
- `model/`: manifest/Safetensors parsing, imported-checkpoint verification and immutable
  model storage.
- `io/`: bounded asynchronous reads.
- `research/`: native-telemetry-driven no-cache/LRU/LFU/static-hot simulation.
- `inference/`: TinyTransformer, pinned SmolLM2 generation and pinned OLMoE routed
  CPU reference/streaming adapters.
- `cli/`: interactive and one-shot model client commands.

The next data-plane measurement step is reuse distance plus physical-device I/O and a
repeatable cold-cache protocol. DeepSeek inspection must reuse these safety and identity
boundaries rather than bypass them.
