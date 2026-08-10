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
| `unsloth/DeepSeek-V4-Flash-0731-GGUF` at `fbbb5b93...` | Bounded GGUF v3/split headers and `UD-IQ1_S` packed-expert catalog validated | Metadata-only catalog; quantized execution not implemented |
| `unsloth/Laguna-S-2.1` at `1a646907...` | Public config, custom-code and shard metadata reviewed on 2026-08-10 | Adapter and execution not implemented |

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

The first half of that boundary is now implemented for `UD-IQ1_S`. Native inspection of the
three split headers validates 1,328 physical tensors and slices the 129 packed expert matrices
into 11,008 logical layer/expert bundles. The catalog accounts for 75,547,803,648 routed-expert
bytes and 6,986,088,796 always-active bytes. A top-6 no-cache schedule would request
1,770,651,648 logical expert bytes/token. This is header-derived planning data: the two large
payload shards were not downloaded, no full artifact hash was verified and no tensor was
dequantized or executed.

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
`UD-IQ1_S` headers by slicing each packed tensor along its expert axis. Executable full-shard
binding, quantized kernels and the Laguna adapter remain open.

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
VRAM copies and quantized kernel consumption are not connected yet.

## Residency policy

The default safe plan is:

- keep the always-active trunk, routers and current KV/state resident in RAM and/or VRAM;
- keep hot expert bundles in VRAM, warm bundles in RAM and cold bundles immutable on NVMe;
- read every selected expert completely into reserved RAM or VRAM staging before its kernel
  consumes that region;
- reserve double/triple buffers, queue entries and destination memory before submission;
- evict prefetch and cold experts before degrading the always-active path;
- fail closed or enter an explicit low-memory profile if the minimum trunk/state working set
  cannot preserve operating-system headroom.

Streaming the always-active trunk from SSD on every layer/token is technically possible but is
not the normal interactive plan: it multiplies unavoidable traffic. The runtime must reduce
bytes per accepted token, not merely maximize sequential benchmark bandwidth.

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

The shared synchronous oracle and bounded asynchronous pool now implement range reads,
compaction, per-tensor slices and `RAM_READY` publication on a synthetic immutable fixture. The
pool supports one to four workers, a queue of at most 64 requests, pre-enqueue reservations and
safe queued-work cancellation. It makes next-expert overlap possible but does not yet connect a
kernel consumer or measure useful overlap. Cache lookup and direct I/O remain in the existing
OLMoE-specific path.

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
2. Parse headers/metadata without payload and produce a role-aware storage plan.
3. Match tokenizer, chat template and completion parsing.
4. Match reduced operator, route, logit and cache-state fixtures.
5. Run selected full-model probes under hard RAM/VRAM/disk limits.
6. Enable CPU and then GPU execution only after output parity.
7. Publish measured bytes/token and overlap; never infer speed from SSD specifications.
