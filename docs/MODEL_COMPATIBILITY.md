# Model compatibility and universal streaming contract

Vastloom is model-adaptive, not model-agnostic. The memory scheduler, tensor catalog, bounded
I/O pipeline and placement policy are shared; each architecture still needs a small adapter that
defines exact tensor roles, graph equations, tokenizer and chat semantics. A familiar MoE label
or matching tensor dimensions is never enough to enable execution automatically.

## Support levels

| Model/profile | Current evidence | Execution status |
|---|---|---|
| `unsloth/Qwen3.8-Flash-Next-GGUF` `Q8_0` (`qwen4exp`, 125B MoE, 175.3 GiB) | **Comparison/validation track (October 2026 refocus).** Full 48-layer chain, PLE n-gram block and partial RoPE on the official checkpoint; greedy 16/16 token parity with llama.cpp `6fe7498` (CPU, same checkpoint); ~3.3-4 s/token greedy on the reference PC | **Externally validated execution path, no longer primary.** Real checkpoint end-to-end; greedy-only, throughput behind llama.cpp and honestly reported; no new Qwen features planned |
| Synthetic TinyTransformer | Native loader, tokenizer, KV cache, generation and CLI vectors | Validated CPU execution |
| SmolLM2-135M-Instruct | Pinned Safetensors, tokenizer/chat and Transformers parity; dense serve path supports JSON-object mode (`response_format: {"type": "json_object"}`, root object only) and fail-closed control-token rejection | Validated dense CPU execution; constrained JSON output added in `0.5.0-alpha.1` |
| OLMoE-1B-7B-0924-Instruct | Pinned shards, top-k routes, expert streaming and I/O telemetry | Validated CPU expert streaming |
| `deepseek-ai/DeepSeek-V4-Flash` preview at `60d8d707...` | Header catalog, tokenizer, reduced main+MTP prefill, five incremental cache steps and exact eight-token greedy/portable seeded generation | Reduced FP32 reference execution only |
| `deepseek-ai/DeepSeek-V4-Flash-0731` at `7872f01b...` | Public metadata/config reviewed on 2026-08-10 | Not imported, inspected or executed by Vastloom yet |
| `unsloth/DeepSeek-V4-Flash-0731-GGUF` `UD-IQ2_XXS` | D1 metadata identities and catalog (9,451,968 B store, 1,328 physical / 34,223 logical tensors); 780,087,788 B in 45 verified ranges imported; D2a layer-0 execution deterministic twice (1 of 43 layers, 129,280 logits); D2b.1 device-expert residency bit-identical to the CPU path (18 accelerated, 6 resident matrices / 15,230,976 B, max abs diff 0.0); D2b.2 independent NumPy oracle matches the same logits within 9.54e-07; D3 runs 2–3 hash-routed layers with a chain oracle (tokens 0/1, depths 1–3, logits within 1.91e-06) and a five-token prefill (argmax 20,267; logits 1.91e-06 / hidden 5.96e-08; the layer-2 4:1 compressed window is exercised), two opt-in CMake tests and a bit-identical GPU chain; the D4 design gate is closed with fixed decisions | **Active reduced-payload target (October 2026 refocus).** IQ2_XXS is the only execution quant. The full 90.86 GB checkpoint is not stored locally; the track uses bounded exact range slices through the immutable store. One layer of 43 and a 3-layer hash-routed chain (single- and five-token prefill) are validated against independent oracles; the D4 design gate is closed. D2b is closed except the VRAM *payload* tier, which is not claimed. Ratio-128 layers, learned-router/full-depth execution and full-model execution are explicitly out of scope for this concluded record. Historical full-depth records remain dated and are not revalidated. |
| `unsloth/DeepSeek-V4-Flash-0731-GGUF` `UD-IQ1_S` (`fbbb5b93...`) | Catalog, importer, IQ1_S/IQ3_XXS kernels and earlier generate measurements | **Do not execute.** Store deleted on the reference PC. Historical tests and reports only. |
| `unsloth/Laguna-S-2.1` at `1a646907...` | Public config, custom-code and shard metadata reviewed on 2026-08-10 | Adapter and execution not implemented |
| Kimi K3-class profiles | Public-roadmap capacity/traffic study only | Not a reference-PC execution target unless one exact complete expert passes the single-tier gate |

Historical DeepSeek reports remain tied to the preview revision they actually tested. The newer
0731 release is a separate compatibility target, not a silent rename of that evidence. Since the
October 2026 refocus, `UD-IQ2_XXS` on DeepSeek V4 Flash 0731 is the primary execution target,
pursued with bounded exact range slices because the full checkpoint is not stored; D1/D2a now
validate one layer of 43 from 45 verified ranges, D2b.1 validates bit-identical device expert
residency, D2b.2 validates the layer-0 logits against an independent NumPy oracle, D3 validates
a 2/3-layer hash-routed chain for single- and five-token prefill against the independent chain
oracle, and the D4 design gate is closed with fixed decisions; Qwen3.8-Flash-Next is the
comparison/validation track. Ratio-128 layers, learned-router/full-depth execution and the VRAM
payload tier are explicitly out of scope for this concluded record. Rows with a dated-only
status are records:
the large checkpoints are no longer present on the reference PC, and the `0.5.0-alpha.1`
maintenance update did not re-measure or revalidate any large-model path.

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

The bounded catalog and immutable source-admission boundary were first implemented for
`UD-IQ1_S`. Product execution on the reference PC was `UD-IQ2_XXS` only. Under the October 2026
refocus that profile returns as the active reduced-payload target: IQ2_XXS is the only execution
quant, the full 90.86 GB checkpoint is not stored locally, and the resumed work is range-slice
based, so its historical measured records are not revalidated by `0.5.0-alpha.1`. IQ1 / `UD-IQ1_S`
stays non-product and historical.

The October 2026 D1/D2a slice is measured and reported in
[`reports/D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md`](reports/D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md).
D1 pins the `UD-IQ2_XXS` metadata identities (complete split 1, 5,257,664 B, plus separately
hashed 2,097,152-byte header prefixes for splits 2/3), imports a 9,451,968-byte metadata-only
store and verifies it with `deepseek-gguf-checkpoint` (pinned spec, reference config, layer-0
catalog and hashes). Three range plans are imported with every local hash verified: layer-0
trunk 138,867,160 B / 21 ranges, globals 596,000,788 B / 6 ranges and the authoritative route
45,219,840 B / 18 ranges resolved from the imported trunk's own `tid2eid` for token 0 (experts
254, 222, 245, 200, 53, 35), total 780,087,788 B in 45 ranges. D2a executes layer 0 twice with
identical output: 1 of 43 layers, 129,280 logits (min −16.1273, max 15.5206), argmax 30,528,
45 relocated tensors / six bundles, 345.325 ms and 347.433 ms. At that gate there was no
independent logits oracle; D2b.2 supplies one below.

D2b.1, D2b.2 and D3 (single- and multi-token scope) are also measured and reported in
[`reports/D2B_DEVICE_RESIDENCY_VALIDATION.md`](reports/D2B_DEVICE_RESIDENCY_VALIDATION.md),
[`reports/D2B_LAYER0_ORACLE_VALIDATION.md`](reports/D2B_LAYER0_ORACLE_VALIDATION.md),
[`reports/D3_HASH_CHAIN_VALIDATION.md`](reports/D3_HASH_CHAIN_VALIDATION.md) and
[`reports/D3C_MULTITOKEN_ORACLE_VALIDATION.md`](reports/D3C_MULTITOKEN_ORACLE_VALIDATION.md). With
`--cuda-experts --gpu-limit 16` the layer probe reports offered 38, accelerated 18, declined 20
(all recurring trunk/global), evicted 12, 6 resident matrices / 15,230,976 B, uploaded
45,219,840 B, kernel 138.961 ms and upload 6.2527 ms; all 129,280 logits are bit-identical to
the CPU run (max abs diff 0.0). With `--gpu-limit 1`, nothing is accelerated and the logits are
still identical. This is device residency, not a VRAM payload tier: expert bundles still enter
governed RAM first. `deepseek-chain-probe` executes 2–3 hash-routed layers from per-layer trunk
and expert artifacts (lazy trunk window, per-layer expert caches, authoritative routes from each
layer's own `tid2eid`; learned-router layers are refused): token 0 over 3 layers gives argmax
86,051, logits [−16.0034, 16.1624], trunk 437,727,112 B, expert bytes 135,659,520 B, 2,761.23 ms
and a bit-identical repeat; token 1 gives argmax 69,146. Two opt-in CMake tests pass
(`vastloom_deepseek_v4_chain_probe` 587.64 s, `vastloom_deepseek_v4_chain_oracle` 780.83 s), and
the GPU chain (`--cuda-experts --gpu-limit 64`: offered 116, accelerated 54, declined 62,
evicted 28, 26 resident matrices / 66,350,080 B, uploaded 135,659,520 B) is bit-identical to CPU
(max abs diff 0.0). D3 is met for both scopes: the single-token chain above, and the five-token
prefill `[0,1,2,3,4]` (argmax 20,267, logits within 1.91e-06, hidden within 5.96e-08, all 15
per-layer/per-token routes identical; the layer-2 4:1 compressed window is non-empty, with one
compressed row selected by positions 3–4, compressed stages 512/128 and six slots per position).
Ratio-128 layers, learned-router/full-depth execution and the VRAM payload tier remain out of
scope for this concluded record.

D2b.2 is the first external-reference comparison of this path: an independent NumPy
implementation (`tools/reference_deepseek_v4_layer_oracle.py`; no C++ calls, no network) parses
the pinned GGUF headers, maps the same range artifacts to tensors and recomputes layer 0 with
independent Q8_0/Q5_K/Q6_K/Q4_K/BF16/F32/I32/IQ2_XXS/IQ3_XXS decoders. All 129,280 logits agree
within **9.5367431640625e-07** (gate 1e-3), the final hidden state within
**4.76837158203125e-07** (rtol 1e-5 / atol 1e-6), and argmax 30,528 plus route
254/222/245/200/53/35 are identical; the synthetic section passes 87 pinned checks.
`VASTLOOM_DEEPSEEK_V4_ORACLE_PYTHON` registers `vastloom_deepseek_v4_layer_oracle` (`ctest`
PASS at 87.14 s). This is layer-0 parity within the declared tolerance, not bit-identity and
not model parity; 66 oracle stage checksums are recorded but only the logits and the final
hidden state have a probe counterpart.

The chain oracle extends that implementation to three layers (`--layers 3 --depths 1,2,3`),
including the layer-2 4:1 compressor and indexer: for tokens 0/1 at depths 1–3 the logits agree
within **9.5367431640625e-07** (token 0) / **1.9073486328125e-06** (token 1), the final hidden
state within 4.76837158203125e-07 → 3.814697265625e-06 (token 0) and 2.384185791015625e-07
(token 1), with identical argmax (30,528 / 86,051 / 86,051 and 69,146) and per-layer routes and
unchanged tolerances; the synthetic section now has 104 pinned checks. Three layers of 43, no
throughput or full-model claim.

D3c extends the same chain oracle to the five-token causal prompt `[0,1,2,3,4]`: the last-position
logits agree within **1.9073486328125e-06** and the final hidden state within
**5.960464477539063e-08**, argmax 20,267 (logits [−38.0188, 14.7002]) and all 15
per-layer/per-token routes are identical; the native run is deterministic across two runs with
trunk 437,727,112 B and expert bytes 663,224,320 B (88 bundles from 15 artifacts). The layer-2
4:1 window is really exercised: one compressed row is selected by positions 3–4, with compressed
stages 512/128 and six slots per position. The chain contract test includes the five-token leg.
D4's design gate is closed with the §11 decisions (SHA-256 first-touch digests; disk bundle cache
2,000,000,000 B, staging 268,435,456 B, RAM expert cache 1,073,741,824 B; no dependency on the
VRAM payload tier; no fetch in this project). Ratio-128 layers, learned-router/full-depth
execution and the VRAM *payload* tier remain out of scope for this concluded record.
Native inspection of the
three split headers validates 1,328 physical tensors and slices the 129 packed expert matrices
into 11,008 logical layer/expert bundles. For the historical `UD-IQ1_S` catalog the routed-expert
total is 75,547,803,648 bytes (top-6 no-cache: 1,770,651,648 logical expert bytes/token); the
active `UD-IQ2_XXS` catalog keeps the same 6,986,088,796 always-active bytes and closes at
83,869,302,784 routed bytes (top-6 no-cache: 1,965,686,784 logical expert bytes/token), with
expert bundles of 7,536,640–9,830,400 bytes. This is header-derived planning data: no complete
payload shard was downloaded, while the D1 range payload is imported and hash-verified.

The importer pins all three complete object paths, sizes and SHA-256 values at revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`. For the historical `UD-IQ1_S` profile,
metadata-only mode imports the complete first split
plus exact 52,320- and 33,600-byte prefixes, totaling 5,343,584 bytes. The native checkpoint
command revalidates the immutable manifest and every imported hash. A full import with skipped
hashes remains metadata-only; only complete full-file verification plus live single-tier expert
admission can expose executable source ranges. This source binding is implemented and covered by
a synthetic full-payload test, but the official 82,539,237,792-byte IQ1_S payload has not been
imported or executed.

For the historical `UD-IQ1_S` profile, the inspector retains only the explicitly requested
46-entry integral compression schedule under
a separate array bound, cross-checks the 43-layer/256-expert/top-6 scalar metadata and derives the
batch-1, 4,096-token memory components. Runtime state is 116,123,672 bytes; the largest expert is
8,781,824 bytes. A streamed-trunk CPU floor is therefore 124,905,496 bytes, while keeping all
6,986,088,796 always-active encoded bytes resident raises the CPU floor to 7,110,994,292 bytes.
All 43 layers also produce bounded non-routed plans: 1,193 tensors become 1,193 tiles under a
64 MiB ceiling, with 138,867,160–161,707,992 bytes of trunk per layer and a 35,651,584-byte
largest tile. These are metadata-derived plans. For the historical `UD-IQ1_S` profile the
official tile sources remain unavailable as complete shards; for the active `UD-IQ2_XXS`
profile D2a executes layer 0 and the D3 executor runs the three hash-routed layers (0–2) from
imported range artifacts.

The layer probe planner composes a complete trunk plan with exactly top-k unique complete
expert bundles while preserving immutable full shard identities. For the historical
`UD-IQ1_S` profile, layer 0 produced 21 trunk ranges plus 18 caller-supplied expert ranges:
177,795,544 logical bytes, a 35,651,584-byte maximum range and a 6,488,064-byte single-expert
floor; those IDs were a representative selection, not a route for an official input. For the
active `UD-IQ2_XXS` profile D1 resolves the route from the imported trunk's own `tid2eid` for
token 0 (experts 254, 222, 245, 200, 53, 35) with routing authority verified, making those
offsets authoritative for that token. The planner itself reads no payload and reports
`execution_supported=false`; the separate D2a layer probe consumes the three range artifacts.

The separate range importer now accepts that plan as bounded untrusted input and can atomically
publish exact revision-pinned spans under `models/optimized` after explicit authorization. It
enforces exact HTTP range/length responses, local per-range hashes, a selected-payload cap and the
model-store disk reserve. Its real-metadata test began with a no-network dry-run for the
138,867,160-byte layer-0 trunk; D1 then imported the trunk, globals and route scopes with
explicit authorization. A native
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
mixed-format prefill/decode graph still matches its oracle. The exact pinned GGUF metadata now
derives the native 43-layer configuration, including the 46-entry compression schedule and
separate 43-entry routed/shared SwiGLU clamp schedules. A buffer-free gate validates every real
main-layer trunk layout: 1,193 descriptors and 6,390,088,008 encoded bytes across 43 layers match
their native uncompressed/hash, 4:1 compressor/indexer or 128:1 compressor consumers. D1 later
downloaded and hash-verified the 780,087,788-byte range payload; D2a executes one layer from it.

A complete checkpoint-catalog validator now adds the six global consumers and all routed experts
to that contract. It verifies canonical global shapes/roles, every deterministic layer/expert
bundle identity, three one-to-one packed slices and exact coalesced read-byte closure. The final
partition is 1,193 layer tensors + 6 globals + 129 physical packed expert containers = 1,328
physical tensors; its logical expansion is 1,193 + 6 + 33,024 = 34,223 descriptors across 11,008
expert bundles. The historical `UD-IQ1_S` totals close as 6,986,088,796 always-active +
75,547,803,648 routed = 82,533,892,444 known tensor bytes; the active `UD-IQ2_XXS` catalog closes
at 6,986,088,796 + 83,869,302,784 = 90,855,391,580. That catalog closure performs layout
validation only. A later
pinned-ggml test now admits Q4_K global matrices to the scalar kernel and bounded row-lookup path;
another pinned-ggml test admits IQ2_XXS gate/up matrices to scalar dequantization, row lookup and
transactional matvec. An alternate reduced graph executes IQ2_XXS W1/W3 plus IQ3_XXS W2 for
prefill and incremental decode against an independently dequantized oracle. A second pinned-ggml
test admits MXFP4 down matrices to the same bounded scalar/row/transactional boundary; a reduced
graph with IQ1_S W1/W3 plus MXFP4 W2 also matches its dequantized prefill/decode oracle.

The six global consumers now also have a dedicated deterministic range plan and artifact scope.
Pinned official metadata produces six ranges in one immutable source object: two 297,861,120-byte
Q4_K matrices plus four F32 tensors, totaling 596,000,788 bytes. Native binding regenerates that
plan, revalidates every source identity/span and admits only a closed fully hashed non-executing
artifact. A reduced Q8_0/F32 fixture then loads all six consumers into one aggregate
Governor-backed RAM buffer and retains one shared lease across embedding, output, normalization
and Hyper-Connection prefill/decode parity. The official set began as an importer dry-run; D1
imported all 596,000,788 bytes with local hashes verified and D2a executes them as part of the
layer-0 probe. A real transfer must explicitly authorize network access
and raise the importer and loader defaults to the independently verified plan without bypassing
the Governor or the unchanged absolute bounds.

The layer catalog uses F32, Q8_0, Q5_K, Q6_K, I32 and BF16. Bounded scalar handling now covers all
1,193 tensors and all 6,390,088,008 encoded layer-trunk bytes: floating matrix formats have exact
decode/reference matvec paths, while the three I32 `ffn_gate_tid2eid` router tables have a typed
integer decoder. Q8_0/Q5_K/Q6_K synthetic raw blocks match pinned ggml fingerprints. This closes
the real-metadata storage-format gate only. Separately, all 23 matrix sites in the reduced
official-topology layer fixture now consume Q8_0 bytes plus an I32 hash table and match an
independently dequantized prefill/decode oracle. No official trunk payload byte had run at that
gate; the later D1/D2a slice reads the 138,867,160-byte IQ2_XXS trunk and executes one complete
layer, and the reduced fixture does not inherit that claim.

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
verifying the complete source shard. Those extracted IQ1_S ranges validated one expert; the
later D2a run executes a complete official IQ2_XXS layer 0 (1 of 43) and the D3 executor
extends range execution to the three hash-routed layers, while a routed token or generation
path does not exist.

The reduced assembled graph now also runs a bounded token lifecycle: eight greedy tokens match a
fresh hash-pinned preview-model regeneration, portable seeded top-k sampling is deterministic,
per-layer cancellation/timeout is transactional and generation telemetry reports only observed
expert-cache access. That control-flow result does not use the official 0731 payload, tokenizer or
43-layer graph; the later D2a layer-0 execution and the D3 executor's 2–3 hash-routed layers are
the only real-payload results, and they remain a partial depth (at most 3 of 43).

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
sequential disk benchmark. The October 2026 DeepSeek refocus states the intended contract for that
track: selected experts placed in VRAM with bounded fallback to system RAM, always-active trunk
weights streamed from NVMe under the Governor, and no full model resident.

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
one to four workers, a queue of at most 256 requests, pre-enqueue reservations and safe queued-work
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
