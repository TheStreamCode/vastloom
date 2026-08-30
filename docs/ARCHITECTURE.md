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

M7 now binds verified storage metadata into a shared model-independent catalog:

```text
architecture adapter + verified tensor headers
      │ role · dtype · shape · logical layer/expert
      │ immutable shard · absolute offset · byte count
      ▼
universal logical tensor catalog
      │ duplicate/overflow/overlap/count validation
      ▼
deterministic routed-expert bundles
      │ exact same-shard adjacency coalescing
      ▼
bounded source ranges for the future residency scheduler
```

For the pinned DeepSeek preview, 67,584 routed tensors become 11,264 main/MTP expert
bundles. Each bundle is 13,369,344 bytes and its six tensors coalesce into exactly two source
ranges, for 22,528 ranges total. Each bundle now also carries six checked packed slices so a
consumer can recover exact tensors after range compaction. These are validated addresses, not
issued DeepSeek I/O requests; the header-only inspection still has no payload and reports
execution as unsupported.

The official 0731 GGUF path reaches the same catalog through a different physical layout:

```text
bounded GGUF v3 split headers
      │ exact GGML type · block size · aligned relative offset
      ▼
129 packed `blk.N.ffn_{down,gate,up}_exps.weight` tensors
      │ final dimension = 256 experts
      ▼
33,024 non-overlapping projection slices
      │ group by `(layer, expert)`
      ▼
11,008 three-slice expert bundles + 1,199 always-active tensors
```

The reviewed `UD-IQ1_S` headers contain 1,328 physical tensors and become 34,223 logical
descriptors. Header-prefix objects are always metadata-only unless a caller supplies separate
immutable full-shard paths. `deepseek-gguf-checkpoint` is the only CLI promotion boundary: it
requires the pinned importer manifest, exact model/revision/path/size/SHA identities, full-file
hash verification and live single-tier expert admission. Skipping hashes forces the resulting
catalog back to metadata-only. Exact encoded byte accounting is available to planning, but the
real full payload has not been imported and layer/token execution remains outside this slice.

For the exact pinned checkpoint spec, the bounded reader retains only three explicitly named
architecture arrays: the 46-value compression schedule and the independent 43-value routed and
shared SwiGLU clamp schedules. Integral and floating arrays share one aggregate element budget;
wrong types, duplicate requested keys, scalars and incomplete schedules fail closed. The verified
snapshot constructs `DeepseekV4ReferenceConfig` directly from GGUF metadata, including vocabulary,
context, attention/indexer, Hyper-Connection, MoE and YaRN dimensions. Synthetic and unpinned
catalogs cannot claim this bridge merely by presenting compatible tensor names.

The separate layer-0 catalog validator reuses the canonical consumer layout from the resident
trunk binder. It verifies exactly 11 matrices, nine F32/BF16 vectors and one I32 `tid2eid` table,
including logical dimensions, GGML storage type and recomputed encoded byte count. It allocates no
tensor buffer and reads no payload; it is a metadata-to-consumer contract, not execution.

The all-layer validator extends that same contract over every main block. It derives the expected
layout from the verified layer index, hash-layer count and compression ratio:

- layers 0–1 contain 21 common/hash objects;
- 4:1 layers contain 31 objects, adding attention compression and learned-indexer consumers;
- 128:1 layers contain 25 objects, adding attention compression without the indexer;
- hash layers require an exact two-dimensional I32 `top-k × vocabulary` table, while learned
  layers require the expert-probability bias.

Every canonical name, role, layer, matrix row/logical-column contract, float shape, GGML storage
type and recomputed byte count is checked. Unexpected non-routed tensors fail the complete layer.
The real headers validate 43 layers, 1,193 descriptors and 6,390,088,008 encoded bytes, matching
the independently built layer planner totals without payload allocation or reads.

The checkpoint catalog validator then closes the objects outside that layer-only view. Six
global descriptors are matched to token embedding, LM head, output normalization and the three
Hyper-Connection head consumers. Every `(layer, expert)` bundle must contain exactly one canonical
down, gate and up slice, every slice must be claimed once, its packed range must remain in bounds,
and coalesced read bytes must equal the bundle total. The physical partition must be exactly
`1,193 layer + 6 global + 129 packed expert = 1,328`; the logical partition must be
`1,193 + 6 + 33,024 = 34,223`. Checked byte accumulation must close at 82,533,892,444 bytes.
The catalog pass recognizes Q4_K, IQ2_XXS and MXFP4 as GGML block layouts without authorizing
execution. Separate pinned-ggml parity gates now admit all three through
`supports_reference_matvec_ggml_type`. Each supports bounded row decode and transactional matvec;
Q4_K feeds reduced embedding/output consumers, IQ2_XXS feeds an alternate routed gate/up graph
and MXFP4 feeds an alternate routed down graph. The official checkpoint still remains
non-executing: format capability does not replace verified source, layer and token gates.

Every non-routed `blk.N.*` descriptor now retains its checked layer identity. The shared layer
planner walks catalog order, excludes routed-expert bundles and splits each immutable tensor range
at a 64 MiB ceiling. On the real 0731 metadata this produces complete plans for 43/43 layers:
1,193 tensors and 1,193 tiles, with 138,867,160–161,707,992 encoded bytes per layer and a largest
individual tile of 35,651,584 bytes. The aggregate layer is never allocated as one buffer.

```text
logical layer N
  -> ordered non-routed tensors
  -> bounded source tile(s) per tensor
  -> reserve RAM destination + staging
  -> read immutable range
  -> RAM_READY + use lease
  -> compute consumer (not connected for official 0731 payload)
```

The synchronous shared tile loader validates this transaction on a synthetic payload, including
Unicode paths, cancellation, short reads and reservation cleanup. Metadata-only catalogs can plan
but are forbidden to load. The official 82.54 GB payload is absent, so these real layer numbers
remain metadata-derived placement evidence rather than an execution claim.

### Checkpoint-shaped DeepSeek layer probe planning

The non-executing `DeepseekV4LayerProbePlan` composes one layer trunk plan with exactly the
adapter-declared top-k unique complete expert bundles. It retains every tensor slice and immutable
full source-object identity while independently reporting imported header verification, full
payload presence and executable binding. A metadata snapshot can therefore describe real source
offsets, but its local `.header` prefix is never presented as an executable tensor source.

The official first three layers use `tid2eid[input_ids]`, so the eventual token path must remain
two-phase:

```text
verified trunk ranges
  -> execute attention/shared/router state and decode authoritative tid2eid IDs
  -> validate and deduplicate exactly six IDs
  -> acquire only their complete W1/W2/W3 bundles
  -> execute expert reduction in authoritative order
```

On the pinned real layer-0 metadata, the first phase is 138,867,160 bytes in 21 ranges. Six
complete 6,488,064-byte bundles add 38,928,384 bytes in 18 ranges, producing a 177,795,544-byte,
39-range plan with one 35,651,584-byte maximum range. The current command accepts caller-supplied
IDs only to validate layout and capacity, always emits `routing_authority_verified=false`, and
performs no payload read or execution.

### Immutable partial range artifacts

The offline Python range importer consumes that JSON as an untrusted bounded input. It rechecks
the pinned model/revision, verified-metadata provenance, counts, totals, non-overlap and every
complete source-object identity before selecting a scope. The current plan can authorize only its
21-range trunk scope; expert/all scopes stop before network access because routing authority is
false.

```text
validated plan + explicit --allow-range-download
  -> selected payload cap + model-store free-space reserve
  -> exact revision/object HTTP Range request
  -> exact 206 + Content-Range + Content-Length
  -> fsynced range + local SHA-256
  -> target-specific verified cache
  -> same-filesystem .incoming directory
  -> complete manifest/range revalidation
  -> read-only atomic rename into models/optimized
```

`vastloom.range-artifact.v1` maps short local `ranges/NNNN.bin` paths back to the source object,
complete source size/SHA-256 identity, original offset, exact bytes, role and local range hash. It
always declares the full checkpoint unverified and execution unsupported. The source-object hash
is recorded from the pinned catalog; only the partial payload receives a local hash in this path.
This prevents a small derived artifact from masquerading as the complete shard.

Crash recovery is target-scoped under the same exclusive import lock. Process-unique partial
responses are discarded instead of resumed across servers, stale unpublished trees are removed,
and already complete cached ranges are rehashed before reuse. A native artifact reader/catalog
binder closes the identity boundary. One selected complete bound range can reach the shared
governed RAM loader and a scalar storage consumer. A second closed-set path now requires complete
coverage of every non-routed tensor in the bound layer catalog, packs all independently hashed
ranges into one aggregate-governed buffer and exposes typed retained views to the reference graph.
The successful graph path is still a reduced synthetic catalog, not an official checkpoint layer.

```text
vastloom.range-artifact.v1 directory
  -> bounded JSON and closed file-set validation
  -> local size plus SHA-256 verification for every range
  -> reproduce trunk or six-global plan from hash-verified pinned GGUF metadata
  -> exact model/revision/scope/label/source span/full-object identity match
  -> offset-zero immutable local tile views
  -> complete-tensor preflight before residency
  -> aggregate RAM + maximum-range staging reservation
  -> direct packed reads + second in-memory SHA-256 per range
  -> one retained RAM use lease + typed matrix/I32/F32/BF16 graph views
  -> reduced fixture execution only; official execution remains unsupported
```

The generic verifier accepts at most 4,096 ranges, 1 GiB per range and 16 GiB total; callers may
lower those bounds. It rejects root or payload symlinks, relative-path traversal, missing or extra
files, source overlap, inconsistent object identity, size/hash mismatch and any artifact claiming
to be a complete checkpoint or executable. The DeepSeek binder additionally requires full local
range hashes, hash-verified pinned metadata and a plan reproducible from that catalog. It supports
trunk and dedicated global scopes while routing authority remains false. Expert-bearing scopes
still require explicit authority. The manifest's raw plan SHA-256 is retained
as provenance; exact regenerated range identities, rather than a shell-dependent JSON newline or
encoding, are the native binding authority.

The generic bound-range loader verifies the selected identity and source bounds again, requires a
matching `NVME` residency object, and caps any one load at 1 GiB even if a caller requests more.
`begin_read` reserves the complete destination plus equally bounded staging against the aggregate
host-memory ceiling before the byte vector exists. After the exact read, SHA-256 is computed over
the destination bytes before `RAM_READY` publication. The returned immutable buffer owns a
`ResidencyUse`, so eviction remains illegal until the last consumer releases it. The current
fixture consumes one complete four-byte F32 normalization range in a transactional `1 x 1` matvec;
it remains the smallest single-range oracle.

The companion range-set loader accepts at most 4,096 ranges, defaults to 512 MiB total and 64 MiB
per range, and has absolute 4 GiB/1 GiB ceilings. It preflights all paths, bounds, hashes, unique
logical IDs and the exact aggregate total before changing residency. `begin_read` reserves the
entire packed RAM destination plus only the largest single-range staging allowance. Every source
is then read directly into its final subspan and rehashed; no partial set becomes `RAM_READY`.
Missing, corrupt, cancelled or allocation-failed transactions release all reservations and become
`FAILED` only if loading had begun.

The DeepSeek set adapter reproduces the complete non-routed tensor set for the bound layer from
the separately verified catalog. Each range must be one complete final tensor at offset zero and
must preserve exact role, GGML type and original source identity. Matrix and I32 graph binders
retain the one shared buffer, infer physical shape from the catalog and reject unsupported storage
or logical-width mismatches without copying encoded data.

The parallel global adapter first closes the complete checkpoint catalog, then emits exactly six
indivisible ranges in canonical consumer order. On the pinned 0731 metadata they occupy one source
object and total 596,000,788 bytes; each Q4_K embedding/head matrix is 297,861,120 bytes. A global
artifact is rebound to those exact source spans and loaded under one aggregate reservation. Three
packed matrix views map to `embed.weight`, `head.weight` and `hc_head_fn`; three F32/BF16 views map
to `norm.weight`, `hc_head_base` and `hc_head_scale`. All retain the same immutable buffer and one
use lease. Because the official total and maximum range exceed the 512 MiB/64 MiB defaults, a real
caller must explicitly admit both verified plan values without exceeding the unchanged 4 GiB/1 GiB
absolute ceilings. The official plan passed importer dry-run only; the executable parity test uses
a small Q8_0/F32 fixture.

The scalar storage layer now recognizes every GGML type present in those 1,193 layer tensors.
Q8_0, Q5_K and Q6_K follow the exact pinned ggml block layouts; BF16 and F32 are decoded from
explicit little-endian bytes; and the three `ffn_gate_tid2eid` I32 routing tables use a separate
integer decoder instead of being forced through a floating matvec. The real metadata reports
6,390,088,008 / 6,390,088,008 encoded bytes covered and zero unsupported layer tensors.

Floating matrix formats share a transactional dispatcher: the caller reserves disjoint input,
scratch and output spans, the complete operation writes scratch first, and public output is copied
only after every encoded block and row succeeds. Quantized matrices are never expanded in full;
one fixed block of at most 256 floats is decoded at a time.

The reduced DeepSeek graph is now the first architecture consumer of that boundary. Its first
main layer replaces 11 trunk/shared float matrices with deterministic Q8_0 storage, eight routed
W1/W3 matrices with IQ1_S, four routed W2 matrices with IQ3_XXS and its hash table with I32 bytes.
The graph distinguishes logical columns from a block-aligned physical stride, pads activations
with zeros, dispatches transactional matvecs, and feeds raw router projections into the unchanged
sqrt-softplus/top-k equations. Full prefill and incremental steps match the independently
dequantized oracle for main/MTP logits, routes and attention indices.

The same graph now accepts immutable resident byte buffers created only by the bounded trunk and
expert loaders. Each buffer owns its `ResidencyUse`; graph descriptors hold a shared reference plus
an exact checked subrange. The reduced file-backed test loads 12 complete Q8_0/I32 trunk tensors
and four indivisible IQ1_S/IQ3_XXS three-matrix expert bundles, then destroys every loader wrapper
before inference. Its 16 residency objects retain exactly 8,180 bytes and remain non-evictable
until the graph releases its final reference. This eliminates payload copies at the loader/graph
boundary.

The artifact-backed repetition now covers the complete reduced layer-0 trunk shape: 11 matrices,
nine F32 vectors and one I32 table are stored in 21 separately hashed ranges totaling 2,844 bytes.
Those ranges use the canonical official `blk.0.*` names. An explicit 0731 adapter maps them to the
21 exact `layers.0.*` consumers, validates matrix rows, logical input widths, vector lengths and
the hash-table element count, and returns a detached binding set. There is no heuristic suffix
matching and no partial graph publication on failure.

The same canonical layout is now checked independently against the real pinned 0731 catalog
before any range is acquired. The exact metadata-derived configuration supplies all consumer
dimensions, and the validator recomputes each encoded span from shape and GGML type. The official
layer-0 result is 21 descriptors and 138,867,160 logical bytes, with zero payload reads.

Packed F32/BF16 vectors retain exact subranges of the same shared immutable buffer. The scalar
reference path decodes explicit little-endian storage into transient FP32 only at consumption,
rejects non-finite values and caps that scratch at 1,048,576 elements (4 MiB). Matrix payloads
remain encoded and block-decoded by their existing kernels. The four routed bundles retain the
other 5,536 bytes independently. After every loader wrapper is destroyed, complete reduced
prefill and incremental decode still match the independently dequantized oracle. This proves the
closed tensor-set/name-adapter/graph ownership boundary, not official 0731 payload execution.

Routed and shared expert execution now consume distinct per-block SwiGLU limits when verified
schedules are present. Both schedules are validated before graph execution; main blocks index by
main-layer number and MTP blocks use their offset after the main layers. Reduced legacy fixtures
retain their explicit single scalar only when both schedules are absent. This mirrors the pinned
upstream graph's separate `swiglu_clamp_exp` and `swiglu_clamp_shexp` metadata semantics.

The router-first reduced slice stops after authoritative routing, deduplicates expert IDs and
calls a block-scoped provider before expert compute. The reusable synchronous baseline owns a
fixed-capacity RAM LRU of complete three-matrix bundles. Misses execute the existing governed
file loader, hits retain the same immutable backing, and eviction is refused while an inference
result still shares that buffer. An optional asynchronous binding begins one bounded batch for the
selected set, submits every miss to the fixed-worker pool, and acquires expert N only immediately
before its fixed numerical compute turn. Later read service can therefore overlap earlier CPU
expert compute without changing router choice or accumulation order. The graph executes only the
returned packed W1/W2/W3 matrices and emits per-expert resident bytes, logical bytes read, hit
state, request sequence, load/wait/compute timing and the intersection between prior compute
windows and each asynchronous service interval. An oversized selected set is rejected before the
provider runs.

The current cache allows one asynchronous batch owner. A second batch, synchronous resolution or
clear fails closed while it is active. Destruction drains unacquired futures, evicts unpublished
RAM and counts every successfully completed unused load as wasted prefetch. Queue capacity must
hold the entire configured maximum selected set, workers remain capped at four and every request
is independently reserved through the Governor before enqueue.

This closes router-first acquisition for the deterministic mixed-format reduced layer. A second
adapter maps
canonical official 0731 GGUF expert identities to strict W1/W2/W3 descriptors and sends one
extracted real IQ1_S/IQ3_XXS bundle through the same cache/consumer boundary. It does not compose
that checkpoint-shaped expert with the reduced graph or official trunk payload. The real bundle
also passes the asynchronous
future/cache/kernel boundary, while a controlled two-expert reduced fixture proves positive
read-service/CPU-compute overlap with exact oracle parity. Physical-NVMe performance, VRAM
placement and production prefill chunking remain separate gates.

```text
pinned importer manifest + exact object identities
  -> bounded GGUF header/split validation
  -> full SHA-256 (required for executable source ranges)
  -> runtime state fits safe host RAM
  -> one complete expert fits RAM OR VRAM, never their sum
  -> immutable source binding
  -> execution_supported=false until a verified full-checkpoint layer/token consumer exists
```

The primary GGUF metadata shard also supplies the exact compression schedule through an explicit
bounded-retention request; other metadata arrays remain skipped. For batch 1 and 4,096 context
tokens, the native calculator accounts for 33,816,576 bytes of persistent BF16 KV, 12,206,080
bytes of FP32 compressor state, a 1,048,576-byte shared RoPE table and 69,052,440 bytes of bounded
activation/kernel scratch. Together with the largest 8,781,824-byte expert, this yields a
124,905,496-byte streamed-trunk CPU floor. Adding the complete encoded always-active catalog yields
a 7,110,994,292-byte resident-trunk CPU floor. These are capacity profiles; neither executes a
DeepSeek layer/token.

The routed-expert quantized compute oracle accepts complete GGML IQ1_S and IQ3_XXS blocks,
validates every FP16 scale and all dimensions before final output publication, decodes through
fixed 256-float scratch and performs scalar FP64-accumulated matvecs. The trunk storage paths use
the same bounded/transactional rules for Q8_0, Q5_K, Q6_K, BF16 and F32, while I32 remains a
typed lookup-table decoder.
Q4_K follows the same rule for global matrices: its two FP16 scales and complete block layout are
validated before output publication. A row decoder verifies the complete matrix byte contract but
materializes only one caller-selected physical row, allowing token embedding lookup without a
529-million-element FP32 allocation. The reduced graph consumes both packed `embed.weight` and
`head.weight` in prefill and incremental decode; the 1,073,741,824-element descriptor ceiling
admits the pinned 129,280 x 4,096 matrices without permitting unbounded matrix arithmetic.
MXFP4 down projections use one E8M0 scale and 16 packed E2M1 bytes per 32 values. Their bounded
validator evaluates every scaled nibble before output publication and rejects FP32 overflow;
standalone decode/matvec/row lookup and an all-routed-W2 reduced graph match the pinned oracle.
One exact DeepSeek 0731 layer-0/expert-0 gate/up/down bundle is loaded through the shared bounded
worker and `RAM_READY` lease, then resolved through a one-entry router-first RAM cache. The same
bundle also passes the optional asynchronous begin/acquire path. Canonical
GGUF names determine W1/W2/W3 instead of test-defined slice positions. The mixed-format bridge
executes FP32 SwiGLU and matches pinned ggml bit for bit across all 4,096 final outputs; a second
resolution is a zero-read cache hit. The test uses three hash-pinned extracted ranges rather than
an unverified full shard, so this is not registered as a generic GGUF model execution backend or
an overlap benchmark.

```text
verified gate/up/down ranges (6,488,064 bytes)
  -> executable logical catalog -> canonical W1/W2/W3 binding
  -> Governor reserves packed RAM + largest staging range
  -> fixed worker performs three bounded read-only reads
  -> RAM_READY + cache-retained use lease
  -> IQ1_S gate/up -> SiLU multiply -> IQ3_XXS down
  -> transactional FP32 output
  -> second resolution: cache hit + zero logical bytes
  -> lease release -> RAM eviction -> zero reservations
```

## DeepSeek reference generation lifecycle

The reduced DeepSeek graph now has a bounded native autoregressive driver. It composes prompt-token
prefill, authoritative main-model sampling and transactional incremental cache updates without
giving MTP proposals control over the generated token. Greedy output matches eight independently
regenerated tokens from the hash-pinned preview graph; seeded top-k sampling uses raw PRNG bits so
the committed sequence is identical across the tested standard libraries.

```text
bounded token IDs
  -> prefill checkpoint -> each main layer -> head -> MTP -> state commit
  -> greedy or seeded top-k selection
  -> stop/context/maximum-token decision
  -> incremental checkpoint chain
  -> token publication + observed cache telemetry
```

Cancellation, timeout, process-safety and callback failures are checked before graph work, every
main layer, the head, MTP and state publication. A failed checkpoint leaves the caller's previous
incremental state unchanged. Limits cap reference context/new-token counts at 4,096 and stop-token
IDs at 64. Reported expert traffic is summed only from actual provider/cache accesses; a resident
fixture graph cannot invent SSD bytes.

This lifecycle consumes an already assembled reduced graph. It is not wired to the official
43-layer payload, native tokenizer/chat CLI, speculative MTP acceptance, physical NVMe I/O or
CUDA, and therefore is not an official DeepSeek generation path.

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
