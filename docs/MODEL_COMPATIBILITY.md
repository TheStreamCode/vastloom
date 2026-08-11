# Model compatibility and universal streaming contract

Vastloom is model-adaptive, not model-agnostic. The memory scheduler, tensor catalog, bounded
I/O pipeline and placement policy are shared; each architecture still needs a small adapter that
defines exact tensor roles, graph equations, tokenizer and chat semantics. A familiar MoE label
or matching tensor dimensions is never enough to enable execution automatically.

## Support levels

| Model/profile | Current evidence | Execution status |
|---|---|---|
| Synthetic TinyTransformer | Native loader, tokenizer, KV cache, generation and CLI vectors | Validated CPU execution |
| SmolLM2-135M-Instruct | Pinned Safetensors, tokenizer/chat and Transformers parity | Validated dense CPU execution |
| OLMoE-1B-7B-0924-Instruct | Pinned shards, top-k routes, expert streaming and I/O telemetry | Validated CPU expert streaming |
| `deepseek-ai/DeepSeek-V4-Flash` preview at `60d8d707...` | Header catalog, tokenizer, reduced main+MTP prefill and five incremental cache steps | Reduced FP32 reference execution only |
| `deepseek-ai/DeepSeek-V4-Flash-0731` at `7872f01b...` | Public metadata/config reviewed on 2026-08-10 | Not imported, inspected or executed by Vastloom yet |
| `unsloth/DeepSeek-V4-Flash-0731-GGUF` at `fbbb5b93...` | Pinned atomic importer, verified metadata binding, 43 bounded layer-trunk plans, exact layer-0 trunk+top-6 range planning, bounded immutable partial-artifact import with a real trunk dry-run, native closed-world range verification/exact trunk binding, aggregate governed tensor-set loading, all trunk storage types covered by scalar references, one router-first governed reduced Q8_0/IQ1_S/IQ3_XXS layer, and canonical synchronous/asynchronous cache-to-kernel parity on one real expert | The complete reduced packed trunk catalog now reaches the synthetic graph through one artifact buffer; one separate extracted official expert also executes. The checkpoint-shaped official-payload path remains non-executing, its expert IDs are not yet authoritative and no official range payload was downloaded; retained vector/scalar bindings, complete shards, official layer/token and generation remain unvalidated |
| `unsloth/Laguna-S-2.1` at `1a646907...` | Public config, custom-code and shard metadata reviewed on 2026-08-10 | Adapter and execution not implemented |
| Kimi K3-class profiles | Public-roadmap capacity/traffic study only | Not a reference-PC execution target unless one exact complete expert passes the single-tier gate |

Historical DeepSeek reports remain tied to the preview revision they actually tested. The newer
0731 release is a separate compatibility target, not a silent rename of that evidence.

## Newly reviewed targets

### DeepSeek V4 Flash 0731

The [official 0731 repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
advertises the same core 43-layer, 256-expert, top-6 architecture family as the preview, but its
current config includes a longer 47-entry compression schedule and DSpark metadata: block size
5, Markov rank 256 and target layers 40–42. Its 48 Safetensors shards total 166,886,535,336
bytes. Those differences require a new pinned import catalog, source hash, fixture and parser
profile before execution can be claimed.

The [Unsloth GGUF distribution](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
is a derived quantized artifact, not the architecture oracle. At the reviewed revision its main
profiles range from 82,539,237,792 bytes (`UD-IQ1_S`) to 161,869,615,520 bytes
(`UD-Q8_K_XL`), plus separate approximately 10–11 GB DSpark files. Supporting it requires a
bounded GGUF metadata/tensor reader and quantization-specific parity tests; no profile may be
selected through silent quality degradation.

The bounded catalog and immutable source-admission boundary are now implemented for `UD-IQ1_S`.
Native inspection of the
three split headers validates 1,328 physical tensors and slices the 129 packed expert matrices
into 11,008 logical layer/expert bundles. The catalog accounts for 75,547,803,648 routed-expert
bytes and 6,986,088,796 always-active bytes. A top-6 no-cache schedule would request
1,770,651,648 logical expert bytes/token. This is header-derived planning data: the two large
payload shards were not downloaded and no full payload hash was verified locally.

The importer pins all three complete object paths, sizes and SHA-256 values at revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`. Metadata-only mode imports the complete first split
plus exact 52,320- and 33,600-byte prefixes, totaling 5,343,584 bytes. The native checkpoint
command revalidates the immutable manifest and every imported hash. A full import with skipped
hashes remains metadata-only; only complete full-file verification plus live single-tier expert
admission can expose executable source ranges. This source binding is implemented and covered by
a synthetic full-payload test, but the official 82,539,237,792-byte payload has not been imported
or executed.

The inspector retains only the explicitly requested 46-entry integral compression schedule under
a separate array bound, cross-checks the 43-layer/256-expert/top-6 scalar metadata and derives the
batch-1, 4,096-token memory components. Runtime state is 116,123,672 bytes; the largest expert is
8,781,824 bytes. A streamed-trunk CPU floor is therefore 124,905,496 bytes, while keeping all
6,986,088,796 always-active encoded bytes resident raises the CPU floor to 7,110,994,292 bytes.
All 43 layers also produce bounded non-routed plans: 1,193 tensors become 1,193 tiles under a
64 MiB ceiling, with 138,867,160–161,707,992 bytes of trunk per layer and a 35,651,584-byte
largest tile. These are metadata-derived plans; the official tile sources remain unavailable and
no executable official-payload DeepSeek layer exists.

The layer probe planner now composes a complete trunk plan with exactly top-k unique complete
expert bundles while preserving immutable full shard identities. For layer 0, six structurally
valid bundles produce 21 trunk ranges plus 18 expert ranges: 177,795,544 logical bytes total,
with a 35,651,584-byte maximum range and a 6,488,064-byte single-expert floor. The current IDs
`0..5` are a representative layout selection, not a route for an official input. The first three
layers use `tid2eid[input_ids]`; their real table payload must be acquired and decoded before the
six expert offsets become authoritative. The planner reads no payload and always reports
`execution_supported=false`.

The separate range importer now accepts that plan as bounded untrusted input and can atomically
publish exact revision-pinned spans under `models/optimized` after explicit authorization. It
enforces exact HTTP range/length responses, local per-range hashes, a selected-payload cap and the
model-store disk reserve. Its real-metadata test validated only a no-network dry-run for the
138,867,160-byte layer-0 trunk. Expert/all scopes remain blocked by the unverified route. A native
closed-world verifier now rehashes every local range and the DeepSeek binder matches exact local
trunk views to the independently verified catalog, regenerated plan and pinned complete-object
identities. One complete synthetic F32 range now additionally passes an aggregate-budgeted
RAM/staging read, a second in-memory SHA-256, retained residency and scalar matvec. A closed-set
loader now goes further: it rejects incomplete catalog coverage before residency, reserves all
packed RAM plus maximum-range staging once, rehashes every subspan and publishes only the whole
set. The reduced fixture consumes all 21 canonical layer-0 trunk identities from one 2,844-byte
artifact buffer: 11 matrices, nine retained F32 vectors and one I32 table. An explicit 0731
adapter maps the official `blk.0.*` names to exact graph consumers and validates their dimensions;
F32/BF16 vector decode is bounded to 4 MiB of transient FP32 scratch. The complete reduced
mixed-format prefill/decode graph still matches its oracle. No official range was downloaded, the
exact runtime configuration is not yet derived from verified GGUF metadata, and the profile's
execution status is unchanged.

The layer catalog uses F32, Q8_0, Q5_K, Q6_K, I32 and BF16. Bounded scalar handling now covers all
1,193 tensors and all 6,390,088,008 encoded layer-trunk bytes: floating matrix formats have exact
decode/reference matvec paths, while the three I32 `ffn_gate_tid2eid` router tables have a typed
integer decoder. Q8_0/Q5_K/Q6_K synthetic raw blocks match pinned ggml fingerprints. This closes
the real-metadata storage-format gate only. Separately, all 23 matrix sites in the reduced
official-topology layer fixture now consume Q8_0 bytes plus an I32 hash table and match an
independently dequantized prefill/decode oracle. No official trunk payload byte or complete 0731
layer has run, and the reduced fixture does not inherit an official-payload execution claim.

Three bounded ranges from the second shard have now been tested separately: the 1,638,400-byte
gate and up IQ1_S matrices plus the 3,211,264-byte IQ3_XXS down matrix form the complete
layer-0/expert-0 bundle. A fixed one-worker, one-entry loader pre-reserves and packs all 6,488,064
bytes in RAM, then the scalar CPU reference executes SwiGLU and matches pinned ggml for all 4,096
outputs bit for bit. The executable catalog now derives W1/W2/W3 from the canonical GGUF names and
feeds the retained descriptors through the production router-first cache/consumer bridge. The
first resolution is one exact miss, the second is a zero-read hit, and in-flight leases prevent
eviction. A separate one-worker asynchronous begin/acquire batch produces the same output and
ownership behavior. With only one real expert it cannot prove useful overlap. The ranges are
independently SHA-256 pinned extracted test artifacts, not a substitute for importing and
verifying the complete source shard. No complete official-payload layer, routed token or
generation path has executed.

### Laguna S 2.1

The [Laguna S 2.1 model card](https://huggingface.co/unsloth/Laguna-S-2.1) describes a 118B
total, approximately 8B-active-per-token MoE. Its reviewed config declares 48 layers, 256 routed
experts with top-10 selection, one shared expert, 12 full-attention plus 36 sliding-attention
layers, a 512-token sliding window and a 1,048,576-token maximum context. The BF16 distribution
contains 46 Safetensors shards totaling 235,128,558,560 bytes and uses custom `laguna` model
code under OpenMDW-1.1.

From config dimensions alone, one routed BF16 expert is approximately 18,874,368 bytes and ten
experts in one layer are approximately 188,743,680 bytes. Across 47 sparse layers, a no-cache
expert-only decode would move roughly 8.87 GB/token. These are planning estimates, not measured
Vastloom I/O and not a runtime claim. Laguna is valuable as the second large adapter because it
tests whether placement and streaming remain independent from DeepSeek-specific attention,
routing and speculative-decoding details.

## Universal logical tensor contract

Every imported tensor receives a logical identity independent of its physical tier:

```text
model revision
  + architecture adapter
  + role (embedding, norm, attention, router, shared expert, routed expert, head, state)
  + layer / expert / projection identity
  + shape / dtype / quantization / checksum
  + immutable source shard and bounded byte range
```

The shared catalog and Safetensors-backed binding are implemented for the pinned DeepSeek
preview. Its 69,187 tensors produce 11,264 logical expert bundles and 22,528 coalesced immutable
source ranges. The separate GGUF binding now populates the same contract for the official 0731
`UD-IQ1_S` headers by slicing each packed tensor along its expert axis. Its pinned checkpoint gate
can couple fully hashed immutable shards to those ranges after capacity admission. Scalar IQ1_S
and IQ3_XXS oracles execute one complete real extracted bundle through the shared governed loader.
All non-routed `blk.N.*` tensors now retain their layer identity and feed a shared bounded tile
planner/loader. The real 0731 metadata produces 43 complete layer plans, 1,193 tensors/tiles and a
35,651,584-byte maximum tile under the 64 MiB ceiling. Scalar storage handling covers every type
and encoded byte in those layer plans. A reduced architecture consumer now composes Q8_0/I32
trunk bytes with IQ1_S W1/W3 and IQ3_XXS W2 bytes from governed file-backed trunk tiles and
complete expert bundles without copying payloads into the graph. The reduced graph routes first,
deduplicates the selected IDs and loads only those complete mixed-format bundles through a bounded
synchronous RAM LRU with access telemetry.
An opt-in bounded batch now submits the reduced graph's selected misses and acquires each result
at its fixed numerical compute turn; a controlled fixture reports positive read-service/CPU-
compute overlap while preserving the oracle. The real extracted 0731 expert uses the same async
cache boundary after strict canonical catalog binding, but is not composed with official trunk
payload or the reduced graph because its checkpoint dimensions differ. Official full-shard
import, execution of the now-defined checkpoint-shaped 0731 range plan, authoritative hash
routing, useful physical-device overlap, optimized kernels and the Laguna adapter remain open.

The shared runtime may then place a tensor or expert bundle in one of these explicit states:

```text
NVME -> READING -> RAM_READY -> COPYING_TO_VRAM -> VRAM_READY
                    |                              |
                    +---------- EVICTABLE <-------+
```

The shared control object now implements the displayed states, a terminal/retryable `FAILED`
state, a pin flag, counted RAII use leases and an exclusive transfer lease. RAM, VRAM and staging
transitions are budgeted by the Resource Governor. The shared synchronous loader and bounded
worker pool now publish a physical packed RAM buffer only after every canonical range completes.
One real DeepSeek gate/up/down bundle now reaches the quantized CPU consumer while protected by a
use lease. VRAM copies and a model-wide quantized consumer remain unconnected.

## Residency policy

The safe placement policies are:

- prefer a resident trunk when the live plan can preserve host headroom;
- allow an explicit low-memory streamed-trunk profile that keeps immutable trunk weights on SSD
  and reserves only bounded verified tiles, while exposing its recurrent bytes/token cost;
- keep hot expert bundles in VRAM, warm bundles in RAM and cold bundles immutable on NVMe;
- read every selected expert completely into reserved RAM or VRAM staging before its kernel
  consumes that region;
- reserve double/triple buffers, queue entries and destination memory before submission;
- evict prefetch and cold experts before degrading the always-active path;
- fail closed or enter an explicit low-memory profile if the minimum trunk/state working set
  cannot preserve operating-system headroom.

Model admission is based on the minimum live working set, not total SSD capacity or marketing
“active parameter” counts. Router/reduction state, requested KV, activation scratch and one
bounded trunk tile must fit safe host RAM. At least one complete routed-expert bundle must fit
wholly in either the remaining RAM tier or one real VRAM tier. The resident-trunk profile adds all
always-active encoded weights to its destinations; the streamed profile does not. Vastloom does
not split an indivisible expert through generic swap or pretend RAM plus VRAM is one allocation.

Streaming the always-active trunk from SSD on every layer/token is a first-class low-memory plan,
but not a performance result: it multiplies unavoidable traffic. The runtime must reduce bytes per
accepted token through residency, reuse, overlap and model-specific tiling, not merely maximize a
sequential disk benchmark.

## Expert-streaming pipeline

The common pipeline is architecture-independent once an adapter maps its expert tensors:

```text
authoritative router output
  -> deduplicate selected expert bundles
  -> resolve VRAM/RAM hits
  -> submit bounded aligned reads for misses
  -> consume gate/up as soon as its validated region is ready
  -> overlap down read, CPU/GPU compute and next-layer prefetch
  -> reduce outputs in authoritative order
  -> update hit/miss, bytes, queue, transfer and stall telemetry
```

Required metrics are logical and device NVMe bytes per accepted token, cache hit rate, wasted
prefetch bytes, queue/read/copy/compute timestamps, PCIe bytes, GPU idle time and output parity.
Ready-order execution may change scheduling, but never the router choice or numerical reduction
order without an explicit quality-changing mode.

The shared synchronous oracle and bounded asynchronous pool implement range reads, compaction,
per-tensor slices and `RAM_READY` publication. In addition to synthetic fault coverage, a
one-worker/one-entry configuration feeds one real DeepSeek 0731 expert through canonical catalog
binding, both synchronous and asynchronous RAM-cache resolution and the scalar mixed-format
kernel. A first miss and second hit validate exact logical-byte accounting and lease-protected
residency. The reduced mixed-format graph connects router output to a two-entry RAM LRU and can
submit all selected misses before acquiring them at each expert's fixed compute turn. Its four
synthetic bundles use the same IQ1_S gate/up plus IQ3_XXS down format combination as the real
profile. A controlled fixture proves positive service/compute intersection with exact output
parity. The general pool supports
one to four workers, a queue of at most 64 requests, pre-enqueue reservations and safe queued-work
cancellation; the current cache permits one active asynchronous batch and drains/counts successful
unused loads as wasted prefetch. No test yet overlaps real official-payload multi-expert I/O with
compute; direct I/O remains in the OLMoE-specific measurement path.

The separate non-routed tile primitive currently performs one synchronous buffered range at a
time under the same Governor/residency rules. It proves bounded trunk payload movement and fault
cleanup, not compute coupling or SSD speed. A production loop may double-buffer only after both
buffers, queue slots and the target kernel have been budgeted and parity-tested.

## Format policy

- Safetensors is the authoritative path for official revisions and exact tensor identities.
- GGUF is a first-class bounded metadata/catalog format, not a shortcut around architecture
  validation; executable payload import remains gated on exact hashes and quantized parity.
- Derived layouts are written as temporary artifacts, verified and atomically published.
- Original Safetensors/GGUF files remain immutable.
- Quantization profiles are named, reported and parity-tested; there is no implicit precision
  downgrade based only on available memory.

## Acceptance ladder for a new model

1. Pin repository revision, license, config, tokenizer and every artifact identity.
2. Reject the profile unless its minimum irreducible working set fits live host budgets.
3. Parse headers/metadata without payload and produce a role-aware storage plan.
4. Match tokenizer, chat template and completion parsing.
5. Match reduced operator, route, logit and cache-state fixtures.
6. Run selected full-model probes under hard RAM/VRAM/disk limits.
7. Enable CPU and then GPU execution only after output parity.
8. Publish measured bytes/token and overlap; never infer speed from SSD specifications.
