# Development log

This page preserves the chronological development record that the README carried before the
project concluded in October 2026. Current status and results are in
[`RESULTS.md`](RESULTS.md), and the README is the index.

## October 2026 — DeepSeek refocus

The project concluded with a scoped refocus that returned the primary execution target to
**DeepSeek V4 Flash 0731** (`UD-IQ2_XXS`), pursued with bounded exact range slices because the full
90.86 GB checkpoint is not stored, and demoted Qwen3.8-Flash-Next to comparison/validation-only.
The measured sequence was D1 (metadata identities, catalog and range plans), D2a (one real
layer-0 execution), D2b.1 (bit-identical device expert residency with bounded CPU fallback), D2b.2
(independent layer-0 oracle), D3/D3b (2/3-layer hash-routed chain and chain oracle), D3c (five-token
prefill with a non-empty layer-2 4:1 compressed window) and a design-only D4 on-demand expert-range
budget with closed decisions. The VRAM *payload* tier, ratio-128 layers and
learned-router/full-depth execution were not implemented and are not planned. **Project concluded;
no further milestones.**

## Current measured position (August 2026)

> Snapshot preserved as history: it predates the October 2026 refocus above. The execution status
> and measured results are in [`RESULTS.md`](RESULTS.md).

The primary track was at that point **Qwen3.8-Flash-Next** (`qwen4exp`, 125B MoE, 6B active,
48 layers = 36 Gated DeltaNet + 12 QSA, 512 routed experts top-10 + shared). The
native runtime executes the official unsloth **Q8_0 GGUF** checkpoint (175.3 GiB,
canonical immutable import) end-to-end — per-layer forwards, multi-token prefill,
the full 48-layer chain, PLE n-gram block and RoPE included — and achieves
**greedy token parity with llama.cpp** (`6fe7498`, CPU, same checkpoint): pinned
prompt "The capital of France is" produces the identical 16-token continuation
on both engines. The DeepSeek V4 Flash track was considered historical in this snapshot; the
October 2026 refocus reversed that. The Qwen
path is externally validated — that is the difference that mattered here. The long note
below is a chronological lab log; where it disagrees with this box, this box wins.

Historical figures in this note (60 s/token, 6.9 s GPU, "kernels are the gap",
"generation unvalidated") are superseded by the box above. They stay as the path that
got here.

## Lab log: unreleased work after 0.4.0-alpha.4

Vastloom now runs the routed token path of the pinned 13.84 GB
`allenai/OLMoE-1B-7B-0924-Instruct` checkpoint on CPU while keeping only the
always-active trunk and a bounded expert cache in RAM. Native logits, top-8 routes
and routing weights match a Transformers reference. The bounded reader now exposes
process-I/O deltas without labeling Windows transfer counters as physical NVMe traffic,
and trace simulation reports exact unique-expert reuse distance. The official DeepSeek
V4 Flash revision is now pinned, its 46-shard/69,187-tensor metadata is inspectable
from verified header snapshots, and its tokenizer/basic chat framing match the official
reference. Bounded CPU primitives now cover normalization, routing, expert SwiGLU,
Hyper-Connections, YaRN, compressed attention and its learned indexer. A deterministic
two-layer FP32 prefill matches the pinned official graph for logits, greedy token, expert
routes and attention indices. Its 69,187 tensors are now bound to a shared logical catalog:
11,264 main/MTP expert bundles collapse from six tensors to exactly two contiguous source
ranges each. A governed fixed-worker pool can now reserve, queue and asynchronously load those
generic multi-range bundles into packed RAM buffers. A bounded native GGUF v3
reader now also maps the pinned 82.54 GB `UD-IQ1_S` 0731 profile from three headers into 11,008
logical expert bundles without downloading its complete payload. One complete 6.49 MB real
layer-0/expert-0 bundle now passes bit-for-bit gate/up IQ1_S, SwiGLU and down IQ3_XXS parity
against pinned ggml after canonical GGUF binding and a governed one-entry router-first RAM cache;
its first resolution reads exactly 6,488,064 logical bytes and its second is a zero-read hit.
The exact three-object GGUF distribution now has a bounded immutable importer and a fail-closed
source-binding command:
only a fully hashed payload that passes the single-tier expert gate can expose executable source
ranges. Bounded scalar Q8_0, Q5_K, Q6_K, BF16 and F32 reference kernels plus an I32 router-table
decoder now cover the storage types of all 1,193 per-layer trunk tensors in the real metadata.
The reduced DeepSeek graph now consumes 11 Q8_0 trunk/shared matrices, eight IQ1_S routed
gate/up matrices, four IQ3_XXS routed down matrices and its hash router as I32. Prefill and
incremental main/MTP logits, routes and attention indices match the independently dequantized
packed oracle; the original official FP32 gate remains unchanged. A second pass reads the same
8,180-byte payload from a Unicode file as 12 bounded trunk tiles and four indivisible
mixed-format expert bundles. The graph retains their 16 Governor-backed
immutable buffers directly, blocks eviction while in use and releases every reservation after
execution. The reduced MoE graph can now stop after authoritative routing, deduplicate selected
IDs and resolve only those complete file-backed bundles through a two-entry governed RAM LRU.
Three one-token steps prove a 2-miss load, a 2-hit reuse and a 2-eviction replacement while
preserving packed-oracle logits, routes, attention indices and transactional state.
An opt-in bounded batch now submits all selected misses, acquires each expert only at its
deterministic compute turn and preserves the same oracle while a controlled fixture records a
positive read-service/CPU-compute intersection. The real 6.49 MB expert passes that asynchronous
future/cache/kernel boundary too; a single real expert cannot by itself prove useful overlap.
A new plan-only checkpoint command now combines one exact 0731 layer trunk with exactly six
structurally validated complete expert bundles: layer 0 requires 177,795,544 logical bytes in
39 immutable source ranges, with a 35,651,584-byte maximum range and a 6,488,064-byte
single-expert tier floor. The supplied expert IDs are not yet an authoritative `tid2eid` route,
and none of these payload bytes was downloaded or executed.
A separate bounded range importer can now validate that plan, enforce an explicit download flag,
request exact revision-pinned HTTP spans, hash each result and atomically publish a read-only
partial artifact under `models/optimized`. The real 138,867,160-byte layer-0 trunk passed only a
no-network dry-run; selected-expert imports remain blocked until routing authority is verified.
The native verifier now treats that artifact as a closed file set, rehashes every range and binds
it back to the exact regenerated catalog. A synthetic complete four-byte F32 trunk range then
passes an aggregate-budgeted RAM/staging transaction, a second post-read SHA-256 check, retained
residency and a scalar matvec. The same boundary now accepts only complete closed tensor sets:
all ranges are preflighted, aggregate RAM plus maximum-range staging is reserved once, each
packed subspan is rehashed, and `RAM_READY` is published only after the whole set passes. The
reduced mixed-format graph now consumes all 21 layer-0 trunk objects from one 2,844-byte
aggregate artifact buffer: 11 matrices, nine retained F32 vectors and one I32 route table are
published under canonical `blk.0.*` GGUF identities and mapped explicitly to their graph
consumers. Four indivisible IQ1_S/IQ3_XXS expert bundles remain separately governed; prefill and
decode still match the independent oracle. The exact verified 0731 metadata now directly
constructs the native 43-layer reference configuration, retains the 46-value compression and
separate 43-value routed/shared SwiGLU schedules, and validates every official layer-0 trunk
descriptor against its graph consumer without reading payload. That contract now extends over
all 43 main layers: 1,193 non-routed descriptors and 6,390,088,008 encoded bytes match the native
uncompressed/hash, 4:1 compressor/indexer and 128:1 compressor consumers. A second zero-payload
closure now validates the six global consumers and all 11,008 expert bundles / 33,024 logical
expert slices. The complete partition is exact: 1,328 physical tensors become 34,223 logical
descriptors accounting for 82,533,892,444 known tensor bytes. Q4_K now has a scalar
pinned-ggml oracle, transactional matvec and bounded row lookup; reduced packed embedding and
output-head prefill/decode match their dequantized graph oracle. IQ2_XXS now has the same
bounded scalar/transactional boundary, and a reduced graph with every routed gate/up matrix in
IQ2_XXS matches independently dequantized prefill and decode. MXFP4 now has the same bounded
decoder, matvec and row-lookup boundary; a second reduced graph with every routed down matrix in
MXFP4 also matches independently dequantized prefill and decode.
The six official global tensors now have their own deterministic plan and immutable artifact
scope. Verified metadata closes at 596,000,788 bytes in six ranges, led by two 297,861,120-byte
Q4_K matrices. A reduced Q8_0/F32 fixture loads the same six logical consumers into one
aggregate Governor-backed buffer and retains exactly one lease across embedding, output,
normalization and Hyper-Connection prefill/decode parity. The official range set passed a
no-network dry-run only; none of its payload was downloaded or executed.
The reduced graph now also has a bounded native autoregressive lifecycle. Its eight greedy
tokens match a freshly regenerated hash-pinned official `model.py` oracle exactly; seeded top-k
sampling is portable across the tested standard libraries. Per-layer checkpoints enforce
timeout, cancellation and `ProcessSafetyGuard` sampling without publishing partial cache state,
while observed cache hit/miss and logical expert bytes are aggregated across generation.
The first CUDA control-plane slice is also native: `cuda-info` loads the Windows/Linux driver
dynamically without requiring the CUDA toolkit, reports exact device capability without creating
a context, and can perform an explicitly requested Governor-backed smoke allocation capped at
64 MiB. On the reference RTX 4070 Super, Windows and WSL both reported compute capability 8.9
and completed/released a 4 MiB allocation while preserving 1 GiB of live VRAM headroom.
A second explicit smoke now reserves one packed 81,920-byte device buffer, synchronously copies
two 4,096-wide FP32 rows plus their weights to the GPU, launches a toolkit-free scalar PTX
RMSNorm, copies the result back and compares it with the existing DeepSeek CPU reference.
Windows and WSL both passed at `0.000025` maximum absolute error and released all VRAM.
**The complete DeepSeek V4 Flash 0731 model generates text on the reference PC.**
`"La capitale d'Italia e"` -> `" una grande citta cosmopol"`, through all 43 layers of the official
82.54 GB checkpoint, with the model's own routers selecting experts. Six tokens at 60.3 s each,
7.62 GB resident, and the host's 8 GiB of free RAM never touched. The same prompt truncated to
12 layers returns gibberish — the depth is doing the work. This is the model running, not the
model being practical: a minute per token needs the CUDA path and resident experts before anyone
would want to use it.

**The forward pass was wrong while every one of those numbers was measured.** An external oracle
— llama.cpp, built from source against the same checkpoint and given the same tokens — disagreed
on the answer, and the cause was one index expression: the Hyper-Connection mixing matrix was read
transposed, so each layer multiplied the residual by slightly more than one and the hidden state
reached RMS 4,064 by the last of forty-three. It is fixed, all eight strongest logits now match
the reference in order, and the model answers `"La capitale d'Italia è **Roma**."` The speed
figures below stand as measurements of the engine's throughput; they were never measurements of a
correct computation until now. The lesson is in the changelog and it is not a small one: the M6
Python reference carried the same transposition, so two implementations written from one reading
of the equations confirmed each other for a milestone.

**Against that oracle the engine is 14× slower on its GPU path than llama.cpp is on CPU alone** —
6.9 s per token against 0.55, same machine, same checkpoint, twelve threads each. A token moves
8.76 GB of weights, so 1.8 tokens per second needs 15.8 GB/s of effective bandwidth and this path
delivered 1.27, a few percent of a desktop memory bus. (Originally logged as 8.00 GB and 14.4 GB/s;
corrected — an addition error, see the streaming paper's Table 1.) The gap is in the kernels, not
the device and not the memory hierarchy: a matvec measured 17× off the bandwidth it could reach.

**A full-depth token went from 48,970 ms to about 8,000 in one day of work** — 6.1×, and 7.5×
against the original serial implementation's 60,301 — with the output byte-identical at every
step and host RAM down from 7.62 GB to 2.12. Every matvec in the model now runs on the device:
nine kernels, `gpu_declined` zero. The two largest gains were not kernels but corrections to
memory policy that had never been checked against their own arithmetic — evicting instead of
refusing work (2.6×), and streaming layer trunks instead of holding all 43 resident (1.4×). Four changes, each measured: quantized CUDA kernels for the routed
path, layer trunks streamed through a bounded window instead of all held resident (host RAM 7.62 GB
to 2.12 GB, which is also what freed the memory the GPU needed), least-recently-used eviction in
place of refusing work when VRAM filled, and dense kernels chosen by counting which storage types
the model was actually being refused on. The two largest gains both came from correcting a policy
that had never been checked against its own arithmetic.

**Every quantization on the routed path now runs on the GPU.** Q8_0 for attention, IQ1_S for the
experts' gate and up, IQ3_XXS for their down projection, all in hand-written PTX through the same
toolkit-free driver. A 16-layer generation went from 13,300 to 2,374 ms per token — 5.60× — and
produced the same text and the same expert traffic as the CPU path, byte for byte. Each kernel
matches the reference at zero absolute error, not under a tolerance, because it accumulates in f64
in the reference's own order. Getting there meant fixing a VRAM cache that indexed matrices by a
host address the allocator recycles, and parity gates built from dyadic values that could not tell
f32 from f64. See the
[`M9 CUDA expert kernel report`](reports/M9_DEEPSEEK_CUDA_EXPERT_KERNEL_VALIDATION.md).

**One real DeepSeek layer now executes from official weights.** The checkpoint's own router picks
the experts, and eight quantizations run together — Q8_0 attention, Q5_K/Q6_K shared expert, BF16
router, I32 routing table, IQ1_S/IQ3_XXS routed experts and the Q4_K embedding — producing 129,280
deterministic logits for the first layer. One layer of 43: every command reports
`executed_layers` against `model_layers` so a partial pass cannot read as the model's answer.
`deepseek-forward` runs to an explicit depth from a complete checkpoint and refuses in advance any
depth that would not fit the live safe RAM budget, which always leaves 8 GiB to the host.

The first official payload is now local, and the checkpoint itself decided which of it to fetch.
DeepSeek's first three layers route by table lookup rather than by a learned gate, so the imported
layer-0 trunk — 21 ranges, 138,867,160 bytes — contains the `tid2eid` table that determines a
token's experts. Reading it through the governed rehashed range path resolves token 0 to experts
35, 53, 200, 222, 245 and 254, which is the authority the importer had been waiting for: the
previously refused selected-expert scope then fetched exactly those six complete bundles
(18 ranges, 38,928,384 bytes). 82.36 GB of the checkpoint is still not downloaded, and nothing
has been executed from the part that is: this is not yet a complete official DeepSeek
layer/token path,
physical-NVMe performance, official-payload generation or complete CUDA inference implementation.
