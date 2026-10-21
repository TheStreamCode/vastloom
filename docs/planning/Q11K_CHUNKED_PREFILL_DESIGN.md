# Q11k - Chunked prefill design

> **Status: DESIGN ONLY, NOT IMPLEMENTED.** No chunked-prefill code exists, no measurement was
> taken and no throughput claim may cite this document. The design is written so the blocked
> milestone has a testable contract when the checkpoints and the oracle are available again; the
> entry criteria at the end are part of the definition, not a formality.

## 1. Objective and why it is the next milestone

The Qwen3.8-Flash-Next checkpoint executes end-to-end with greedy token parity against llama.cpp,
but only one token at a time. Prefill is the same serial loop as decode: the reference PC measured
roughly **3.5–4 s per token**, so a 4,096-token prompt costs about four hours before the first
generated token. The checkpoint declares a **262,144-token context**, and the blocker is not memory
but prompt processing.

Chunked prefill processes a run of prompt tokens per forward pass and amortizes the two costs the
serial path pays per token:

- **trunk traffic and trunk compute.** In streamed mode the trunk is read and executed once per
  token per layer; in resident mode the matrices are executed once per token. A chunk executes one
  layer's trunk once for C tokens.
- **expert traffic.** Per token, every layer selects 10 of 512 experts. Across a chunk, selections
  are deduplicated into a union per layer, so each selected expert's weights are read and applied
  once for all rows that route to it instead of once per token.

This is the first lever ahead of prefix-KV sessions (Q12) and the Qwen serve path (Q13): both are
short-context demos while prefill is serial. It is also the prerequisite for measuring a long
context at all on the reference machine.

## 2. Preserved semantics

The chunked path must reproduce the validated token-serial path, not replace it:

- 48 layers: 36 Gated DeltaNet + 12 QSA (`full_attention_interval = 4`), hidden 2,560,
  hyper-connection stream of 4 × 2,560, 512 routed experts top-10 plus one shared expert,
  intermediate 640, PLE n-gram block on layer 1, partial RoPE 64/256 at theta 1e7, global mixer and
  LM head over a 248,320-token vocabulary.
- DeltaNet state: per layer, conv history (10,240 channels × 3) and the SSM matrix
  (48 × 128 × 128), F32. All 36 layers together are 117,669,888 bytes.
- QSA state: per layer, indexer keys plus K/V caches. The 12 QSA layers add about 72 KB/token of
  F32 caches, roughly 9.4 GB at 128k.
- Greedy decoding and the llama.cpp oracle remain the authority. The chunked path is accepted only
  where tokens match the token-serial reference, and a split-dependent logit is a defect, not a
  tolerance.

## 3. Layer-major, trunk-stationary pipeline

A chunk is processed **layer-major**, one layer at a time over all C token rows:

```text
embedding rows (C bounded reads)
  -> for layer L = 0..47:
       load/retain layer L trunk (resident: pointer; streamed: one read)
       attention branch over C rows, state carried from previous chunk
       MoE: route all C rows, deduplicate the expert union, expert-stationary loop
  -> final mixer + LM head (last chunk only)
  -> state publication at the chunk boundary, then next chunk
```

The serial path is token-major: for a chunk of C tokens it would load layer L's trunk C times. The
chunked path loads it once, which is the whole point when the trunk is streamed from NVMe. A
streamed trunk read is about 137 MB per layer on average (6.58 GB of Q8_0 layer payload over 48
layers); at C = 64 it is amortized over 64 tokens instead of repeated 64 times.

Per-layer state is mutated only after the complete chunk succeeds. A failed or cancelled chunk
leaves every cache at the previous published boundary, so the next chunk can restart from a clean
token position.

## 4. Batched linear algebra

- Every Q8_0 matvec becomes a **GEMM over C rows** for one weight matrix. Row outputs are
  independent, so a correct batched kernel is a pure reuse of the weight rows; the Q8 rows stay hot
  in cache for the whole row batch.
- This requires the **opt-in ggml SIMD backend** (`VASTLOOM_GGML_ROOT` at configure time, the
  `qwen38-*` CLI flag/`--ggml-backend` at run time). The scalar reference kernels stay the
  correctness oracle; a scalar chunked path would be slower than the serial loop and is not the
  performance path (the scalar chunked-Gated-DeltaNet result is already recorded as negative in
  Q11j).
- Reduction order: each output row must accumulate over the K dimension in the same order as the
  scalar kernel, so a row is comparable element by element. If the backend cannot guarantee that,
  the profile declares a per-stage tolerance and the greedy-token gate becomes the acceptance test.
- The LM head runs only on the final chunk, or only on the rows the caller needs (the last prompt
  token for generation).

## 5. Chunked Gated DeltaNet recurrence

The validated semantics are the token-serial recurrence: causal conv1d over a 4-token window, gated
delta rule with F32 state, gated RMSNorm and output projection. The chunked algorithm must be
algebraically equivalent:

- Split the C rows of a layer into intra-chunk sub-blocks; within a sub-block, the delta-rule update
  admits a matrix form (WY/UT-style transform) with the causal structure made explicit.
- The inter-block update carries the same conv/SSM state that the serial path carries token by
  token; chunk boundaries are exactly the token boundaries, so the state after a chunk equals the
  state after the last token of that chunk.
- Cost: the chunked form is more arithmetic than the serial recurrence (O(C·d) per token serial
  against an intra-chunk matrix cost that grows with C), which is why it is only admissible on top
  of the SIMD GEMM path. It must earn its place through trunk and expert amortization, not through
  the recurrence alone.
- Correctness gate: per-layer state equality at every chunk boundary against the serial reference on
  the reduced graph, then greedy-token equality on the real checkpoint.

## 6. Batched QSA indexer and attention

QSA is the expensive long-context branch: indexer heads 4 × 128, budget 2,048, compress ratio 4,
attention 24 heads / 2 KV heads with head dim 256, partial RoPE on attention q/k and indexer
queries/pooled keys, and a sigmoid output gate.

- **Indexer:** score candidate blocks for all C queries of the chunk in one pass, apply RoPE to the
  queries, and select the top 2,048 blocks per query under the causal mask. Blocks that are not yet
  visible to a query (later in the chunk) must be excluded exactly as in the serial path.
- **Sparse attention:** gather the selected keys per query and run the attention over them, plus the
  cached past. Per-head score buffers are bounded by a query tile (see the budget below), never by
  C × context.
- **Cache append:** K/V and indexer keys for the chunk are appended in token order after the
  attention compute, and the cache length is advanced once per chunk. The append is part of the
  chunk transaction, not an incremental side effect.
- Causal mask: a query at absolute position p attends to keys up to p only. The mask is computed
  from absolute positions, not from chunk-relative indices, because that is the invariant the
  boundary tests check.

## 7. MoE expert union per chunk

Routing runs on every row of the chunk with the existing F64-softmax router (top-10 of 512 plus the
shared SwiGLU expert), then:

1. **Union per layer:** collect the selected expert ids of all C rows; deduplicate. At most 512
   distinct experts, in practice fewer.
2. **Expert-stationary loop:** for each unique expert, gather its assigned rows, apply one batched
   Q8_0 expert MLP (gate/up/down, intermediate 640), and scatter the result into a per-row
   accumulation buffer.
3. **Selected-order reduction:** each row's 10 expert contributions are summed in the original
   top-k selection order, so the scalar accumulation order of the serial path is preserved even
   though the compute order is grouped by expert.
4. **Bounded resolution:** misses resolve through the existing bounded expert LRU keyed by
   `(layer << 16) | expert` and the bounded prefetch/submit path. One expert is 5,222,400 Q8_0
   bytes (4,915,200 weights); a cache of 32 slots is about 160 MiB. The union itself is never
   required to be resident: experts stream through the cache one at a time.

A chunk whose union or prefetch queue cannot fit the declared expert budget is split (bisect the
chunk) rather than admitted with an overcommitted reservation.

## 8. Chunk-boundary invariance contract

This is the acceptance contract for Q11k. For one token sequence and one engine configuration:

1. **Split invariance (exact).** For every prefix length p and every chunk split of that prefix, the
   logits produced for position p must be **bit-identical** across splits when the same kernels are
   used. A split is a list of positive chunk lengths that sums to p; splits differ only in chunk
   boundaries, never in tokens.
2. **Reference parity (declared).** Against the token-serial path: exact for the scalar backend;
   otherwise greedy tokens identical where the oracle logit margin is clear, with near-tie
   deviations measured and documented under the existing D1 decision (the oracle quantizes
   activations to Q8_0, Vastloom does not mirror that rounding).
3. **State equivalence.** After each chunk, DeltaNet conv/SSM state, QSA indexer/KV caches and the
   PLE n-gram context equal the state the serial path holds after the same token position. The PLE
   context is the two-token history the n-gram hash consumes, initialised to EOS at sequence start.
4. **No partial publication.** State advances only after a chunk completes; a failure leaves the
   previous boundary intact and a retry re-runs only the failed chunk.

Test procedure (always-on reduced graph, no real weights):

- prefill a deterministic token sequence of length N under splits `[N]`, `[1] × N`, `[2, 3, 5, …]`
  and a randomly drawn split; compare per-position logits, greedy tokens and every state vector;
- sweep the chunk size over 1..C_max and assert the final logits and final state are independent of
  the size;
- run the same sequence twice and assert bit-identical state publication;
- inject a Governor refusal and a cancellation between chunks and assert the published state is
  unchanged and no reservation leaks.

On the opt-in real checkpoint, the same comparisons run against the token-serial path for a small
number of prompts, plus greedy-token equality with the llama.cpp oracle and the margin-aware parity
battery (Q11l).

## 9. Memory and Governor budget plan

The chunk workspace is reserved **once per chunk, before any allocation**, from the same safe host
budget the rest of the runtime uses. The chunk size is derived from the budget, not chosen first.

Per-token workspace at hidden 2,560 and 4 hyper-connection streams:

| Component | Bytes per token | Note |
|---|---:|---|
| Hidden stream | 40,960 | 4 × 2,560 × F32 |
| Router logits | 2,048 | 512 experts × F32 |
| Expert accumulation | 10,240 | 2,560 × F32 |
| PLE gather (layer 1 only) | 40,960 | 10,240 key/value floats |
| QSA query tile | bounded by tile | 32 rows × 2,048 selected × 24 heads × 4 B ≈ 6.3 MiB |
| Indexer scores | bounded by tile | 32 rows × blocks × 4 indexer heads × 4 B |

At 128k context, a 32-row indexer tile is about 16.8 MiB and the QSA score tile about 6.3 MiB, so a
C = 128 chunk needs roughly 35 MiB of activation workspace before the trunk and expert cache. With a
32-slot expert cache (~160 MiB) and one staging buffer (~5 MiB), the whole per-chunk working set is
about 200 MiB, which the reference PC's post-reserve RAM budget can admit.

Rules:

- `C_max = clamp(floor(available_workspace / per_token_workspace), 8, 512)`; the default chunk is 64.
- The trunk is either resident (about 7.3 GB Q8_0) or streamed per layer once per chunk; the chunk
  never forces trunk residency.
- KV/state growth is **not** reduced by chunking: QSA caches are still ~72 KB/token F32, so a long
  context remains a separate admission problem (F16 compression is the P1 item).
- A chunk workspace that cannot be reserved is refused before the first read; the caller can retry
  with a smaller chunk but cannot bypass the Governor.

## 10. Always-on tests versus the opt-in real-checkpoint gate

- **Always-on (CI, no weights):** extend the reduced Qwen38 graph (2 layers + MTP, 4 experts top-2)
  with a chunked-prefill entry point and the invariance suite of section 8. These tests prove
  semantics, determinism, budget refusal and transactional state publication; they prove nothing
  about speed and must not be described as a benchmark.
- **Opt-in (`-DVASTLOOM_QWEN38_MODEL=<Q8_0 root>`, serial):** real chunked prefill against the
  token-serial path on the official checkpoint, greedy-token equality with llama.cpp `6fe7498`, a
  measured bytes/token and prefill tok/s report. This requires the checkpoint to be locally
  available again and is therefore blocked today.

## 11. Failure modes

| Failure | Required behavior |
|---|---|
| Split-dependent logits beyond the declared contract | Refuse the path; it is a defect, not a tolerance. |
| QSA cache/state budget exceeded | Refuse the chunk before allocation; the caller may reduce C. |
| Expert union exceeds cache or queue bounds | Bisect the chunk or refuse; never overcommit the reservation. |
| Governor refusal mid-chunk | Roll back the chunk; no state publication; no leaked reservation. |
| Short read, timeout or cancellation during expert fetch | Transactional failure; previous chunk boundary remains valid. |
| NaN/Inf at any stage | Stage-named error; generation stops; no partial output. |
| Chunk boundary state mismatch against serial | Block the milestone; do not publish a chunked path. |
| Greedy token divergence from the oracle at a clear margin | Treated as a correctness defect, not as near-tie documentation. |

## 12. Entry criteria (blocking)

All of the following must hold before implementation starts; they are shared with Q11l–Q11n, Q12
and Q13 and recorded in [`docs/ROADMAP.md`](../ROADMAP.md):

1. **Large checkpoint locally available again.** The official unsloth Q8_0 GGUF
   (`qwen4exp`, 175.3 GiB, exact revision and SHA-256) is re-imported through the canonical
   immutable store. No large checkpoint remains on the reference PC today, which is why the
   milestone is blocked.
2. **Oracle available.** llama.cpp at the pinned revision `6fe7498` is built locally and the
   margin-aware parity battery (`tools/parity_battery.py`, Q11l) exists, with at least six EN/IT
   prompts × 24 tokens and margin dumps.
3. **Budget preflight.** The chunk workspace, trunk profile, expert cache and state fit the live
   safe RAM budget on the reference PC (32 GiB RAM, 12 GiB VRAM) after the Governor's reserves.
4. **Invariance harness merged first.** The always-on split-invariance tests of section 8 exist and
   fail on a deliberately broken chunk boundary before any real-weight work begins.
5. **No claim without a dated report.** Prefill throughput, bytes/token and cache state are reported
   in a validation report under `docs/reports/`; nothing here is a present-tense performance claim.

## 13. Non-goals

Continuous batching, multi-request scheduling and prefix reuse are Q12/Q13 work, not Q11k.
Speculative decoding, KV cache compression (F16) and schema-constrained decoding are separate
milestones. Chunked prefill changes how a prompt is processed, not what the model computes.
