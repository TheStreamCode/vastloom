# Architecture

## Scope

Vastloom is a native inference runtime, not a model and not a generic swap wrapper. Its
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

The current M5 development slice extends the first real sparse path with bounded staged
reads, readiness timing and observability:

```text
pinned OLMoE import (3 BF16 shards, 13.84 GB tensor payload)
      │ full hash and exact 3,219-tensor catalog
      ▼
reserved 953,421,824-byte always-active trunk in RAM
      │ exact expert shard/offset table
      ▼
native CPU attention + authoritative 64-way/top-8 router
      │
      ├── sync bounded read, or async prefetch (depth 1-2)
      │        ├── whole: one worker × 12 MiB
      │        └── staged: two workers × (8 MiB gate/up + 4 MiB down)
      │             reservation-backed buffers + request timeout
      └── optional reserved RAM cache (no-cache/LRU/LFU/static-hot)
      │
      ▼
weighted SwiGLU expert accumulation → final logits
      │
      ▼
atomic token/layer/expert/tier/bytes/submit/queue/ready/compute/order trace
      │
      ├── exact unique-expert reuse distance
      ├── fixed-order versus ready-order replay
      ├── process read-operation/byte deltas
      └── scoped block-device read-operation/byte deltas
```

The model object gives every routed expert a logical `(layer, expert)` identity that is
independent of its physical shard offset or current RAM residency. The runtime does not
map or allocate the complete checkpoint. Synchronous reads remain the default oracle.
The optional whole asynchronous mode validates each expert's three BF16 matrices as one
contiguous 12 MiB shard range, reserves staging before submission, and schedules the
next one or two routed experts on one worker. Staged mode separately validates the
physically contiguous 8 MiB gate/up region and adjacent 4 MiB down region. Two bounded
workers let the down range overlap gate/up compute without repacking or modifying the
checkpoint. Every request owns a staging reservation until its buffer is consumed.
These timings and logical byte counts remain separate from device-level counters and
cache-bypass measurements.

On Windows the asynchronous worker opens each immutable shard through `CreateFileW`,
seeks with `SetFilePointerEx` and performs a bounded `ReadFile` range. This keeps Unicode
paths native and prevents the C++ stream implementation from fragmenting one 12 MiB
expert into thousands of 4 KiB operations. Buffered mode remains the default. The
opt-in `direct-unbuffered-v1` protocol uses `FILE_FLAG_NO_BUFFERING` on Windows and
`O_DIRECT` on Linux. It expands unaligned tensor ranges to a bounded sector envelope,
copies only the requested bytes, and reserves the output plus direct buffer before I/O.

The runtime samples process counters immediately around the forward pass. Windows
`GetProcessIoCounters` exposes read-operation and transfer-byte accounting but no
physical-storage byte counter. Linux `/proc/self/io` supplies logical `rchar`/`syscr`
and storage-layer `read_bytes`. A separate Windows sampler maps the checkpoint volume to
`PhysicalDriveN` and reads `IOCTL_DISK_PERFORMANCE`; native Linux uses a non-virtual
`/sys/dev/block/.../stat` mapping. Device deltas retain a whole-device or partition scope
and may include unrelated activity. Ambiguous and virtual mappings remain unavailable.
The trace simulator independently computes exact LRU stack distance in
`O(records log records)` using bounded state. Current 19-field traces additionally
support a bounded fixed-route versus ready-expert replay while preserving the recorded
reduction order.

M6 adds a non-executing DeepSeek V4 Flash adapter boundary:

```text
official commit 60d8d707…
      │ pinned sizes + SHA-256
      ▼
5 metadata files + 46 bounded Safetensors header snapshots (19.4 MB total)
      │ exact HTTP 206/Content-Range + atomic immutable import
      ▼
DeepSeek config + 69,187-entry weight index
      │ exact name → shard map
      ▼
bounded native header parser
shape · dtype · data offset · overlap/overflow checks
      │
      ▼
46-shard / 69,187-tensor / expert-layout inspection report
execution_supported=false
```

The snapshots contain the original eight-byte Safetensors header length and JSON header,
but no payload. The reader receives the pinned original shard size so it can validate
every data range against the real payload boundary while refusing tensor reads from a
header-only object. The adapter recognizes the checkpoint's FP8 formats and packed I8
expert storage, exact 43-layer/256-expert top-6 layout, hash-routing tensors, attention
compressors and MTP block. It does not instantiate a DeepSeek graph or allocate model
state. The fixture and reference-parity gates must pass before that boundary moves.

The pinned tokenizer has a separate executable native path:

```text
verified tokenizer.json
      │ exact Sequence + Split regex contract
      ▼
locale-independent Unicode category tables
      │ 1-3 digit groups · CJK/Japanese · letters/marks · punctuation/whitespace
      ▼
byte-level BPE + bounded added-token trie
      │
      ▼
basic system/user/assistant chat framing → exact reference token IDs
```

The added-token trie keeps literal matching proportional to the traversed prefix instead
of scanning all 1,283 definitions at every byte. Prompt input remains capped at 16 MiB.
This path does not imply that a DeepSeek graph or generation loop exists.

The next M6 slice adds bounded equation-level CPU primitives:

```text
deterministic reduced numeric fixture
      │ exact expert IDs + FP32 values at absolute tolerance 3e-5
      ▼
RMSNorm · learned/hash router · routed SwiGLU
Hyper-Connection pre/Sinkhorn/post · base RoPE
selected-latent sparse attention + attention sink
      │ strict shapes · finite values · overflow checks · 64M-element ceiling
      ▼
operator parity boundary — graph composition described below
```

The current sparse-attention primitive consumes already-selected latent KV positions.
The graph below supplies learned indexer construction, compressed-position scheduling,
YaRN scaling, shared-expert aggregation and the Hyper-Connection LM head. A separate reduced
MTP invocation is required because the official main `Transformer.forward` does not invoke it.

The completed M6 fixture composes those operators into the main prefill graph:

```text
8 token IDs -> embedding -> HC expansion
      │
      ├─ layer 0: sliding MLA + hash top-2 MoE + shared expert
      └─ layer 1: sliding MLA + 4:1 overlap compressors
                  + learned sparse indexer + learned top-2 MoE + shared expert
      │
      ▼
HC head -> RMSNorm -> 16 reference logits -> greedy token
      │
      └─ MTP: normalized embedding + normalized main hidden
              -> learned sparse-attention/MoE block
              -> MTP HC head -> 16 draft logits -> proposed token
```

All 123 deterministic fixture tensors use the official parameter names and are consumed by
the native run. Five subsequent official decode positions now exercise transactional main/MTP
state mutation:

```text
raw KV -> fixed circular window
layer input -> attention compressor state -> committed compressed KV
            -> indexer compressor state   -> committed scoring KV
                                          -> exact raw/compressed selection indices
```

Position 11 completes a new overlapping 4:1 entry and position 12 consumes it. Every complete
state is aggregate-budgeted before allocation and published only after the full step succeeds.
The prefill/MTP oracle is batch-one FP32, capped at 4,096 tokens, 16 total layers, one MTP layer
and 64 million state elements. It intentionally has no target-verification loop or packed
checkpoint kernels, and is not selected by `deepseek-inspect` for the full checkpoint.

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
in [`MODEL_COMPATIBILITY.md`](MODEL_COMPATIBILITY.md). Always-active trunk and live state receive
residency before expert cache capacity. Selected expert bundles move through explicit bounded
NVMe→RAM→VRAM transitions; scheduling can change readiness order but not authoritative routing
or numerical reduction order.

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
- `planner/`: conservative capacity and placement plan.
- `safety/`: reservations, state machine and heartbeat.
- `model/`: manifest/Safetensors parsing, imported-checkpoint verification and immutable
  model storage.
- `io/`: bounded asynchronous reads plus platform-labeled process-I/O counters.
- `research/`: native-telemetry-driven no-cache/LRU/LFU/static-hot simulation and exact
  unique-expert reuse distance.
- `inference/`: TinyTransformer, pinned SmolLM2 generation and pinned OLMoE routed
  CPU reference/streaming adapters.
- `cli/`: interactive and one-shot model client commands.

The data plane now has device-level I/O plus a repeatable operating-system cache-bypass
protocol. DeepSeek inspection must reuse these safety and identity boundaries rather
than bypass them. Pinned host memory and GPU transfers remain later layers.
