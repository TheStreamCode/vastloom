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

## Current alpha.2 inference path

Alpha.2 adds a complete CPU-only reference slice:

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
- `model/`: manifest parsing and immutable weight verification.
- `io/`: bounded asynchronous reads.
- `research/`: trace-driven cache/traffic simulation.
- `inference/`: native TinyTransformer loader, tokenizer, KV cache and CPU generation.
- `cli/`: interactive and one-shot model client commands.

The next adapter work must extend this architecture rather than bypass its safety and
integrity boundaries.
