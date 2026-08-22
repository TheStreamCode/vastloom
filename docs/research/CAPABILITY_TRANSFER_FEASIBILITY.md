# Capability-transfer feasibility matrix (F1 evidence synthesis)

**Status:** research synthesis, dated 2026-08-21. This document answers the five F1
questions posed in `CAPABILITY_TRANSFER_VISION.md` against two evidence sources: the pinned
local artifacts (ggml checkout `30bf868`, ggml 0.19.0; the repository's own device and
inference layers) and published work cited below. It contains no implementation and makes no
performance claim; every number is either measured earlier by this project (linked) or an
external result attributed to its source.

## Summary matrix

| Q | Question | Verdict | One-line consequence |
|---|---|---|---|
| (a) | Cross-model behaviour distillation into compact low-rank modules | **Sound, standard engineering; gain magnitude unknown** | The compiler needs no new science; only E2 can size the payoff |
| (b) | Retrieval and composition over adapter pools | **Supported, repeatedly, in production-shaped systems** | Index/router/injector follow proven patterns; build the bounded subset |
| (c) | Speculative decoding across the two models | **Split**: frontman self-MTP shipped; cross-model draft→verify sound but host-blocked; teacher-as-draft rejected | Keep MTP intra-frontman; park cross-model behind the Phase A1 blocker |
| (d) | Pinned ggml carries `gemma4` text-forward ops | **All tensor primitives present**; gaps are entirely Vastloom-side (graph builder, loader glue, SentencePiece tokenizer) | The adapter is graph-builder work, not kernel work |
| (e) | Device-compute gap to a fast frontman | **Every needed storage type already has a parity-validated device matvec kernel; nothing else about a device graph exists** | Hybrid CPU-graph/device-matvec offload is the pragmatic first route |

## (a) Compiler: behaviour distillation into compact modules

Behaviour distillation - training a small adapter so a student matches a teacher's outputs or
logits on curated queries - rests on standard results: knowledge distillation (Hinton et al.,
2015), low-rank adaptation (Hu et al., 2021), and on-policy variants such as GKD (Agarwal et
al., 2023). Nothing about it is research-uncertain; what is uncertain is **how much of a
stronger teacher survives the compression into rank-r slots**, which is precisely what E2
measures on a narrow task before any broader claim.

Floor cost, stated honestly: producing one module means running the teacher across a query
set and training the adapter against it. On this project's terms the teacher pass reuses the
existing selective streaming path; the adapter training itself is ordinary consumer-GPU
fine-tuning work (rank-16..64 adapters are routinely trained this way in public tooling).
The MoLoRA result below trains four rank-32 reasoning adapters on a 1.7B base and reports the
composite beating a 4.7x-larger model on four benchmarks - the strongest published echo of
this vision's central bet that specialized modules can substitute for scale.

**Consequence:** the compiler is scheduled engineering, not invention. Its acceptance gate
stays E2's pre-declared one: `frontman+module` must beat `frontman` by more than noise, or
the vision shrinks.

## (b) Retrieval and composition over adapters

This is the question the vision most needed answered, and the literature answers it many
times over:

- **LoraRetriever** (Zhao et al., Findings of ACL 2024): retrieve-then-compose, input-aware,
  dynamic adapter pools.
- **RAMoLE** (arXiv 2406.16989): retrieval-augmented mixture of LoRA experts with a router
  decoupled from the adapters, zero-shot over unseen modules.
- **LoRAuter** (arXiv 2601.21795, Jan 2026): training-free task-level routing; matches
  oracle-selected adapters in-domain, demonstrated over pools of 1500+ adapters.
- **MOLE** (ICLR 2024) and **LoraHub** (COLM 2024): composition-weight learning; PEFT ships
  linear/SVD/TIES/DARE merge strategies as library functions.
- **MoLoRA** (arXiv 2603.15965, Mar 2026): per-token adapter routing with learned gating;
  reports specialization beating scale (above).

The pattern the vision needs - embed the request, retrieve top-k modules, compose, apply on
a frozen base - is exactly these systems' shape. Vastloom needs only a small, governed slice
of it: frozen-embedding similarity, fixed k, bounded pool sizes, lease discipline inherited
from the expert cache.

**Consequence:** build the bounded subset; do not invent a new routing scheme.

## (c) Speculative decoding across the two models

Three distinct ideas hide under "MTP/speculation"; they have different verdicts.

1. **Frontman self-MTP: supported, shipped.** The selected frontman ships its own MTP
   drafter that shares the target KV cache and changes no output. Intra-model, same
   vocabulary, no new machinery. This is the default acceleration story.
2. **Cross-model draft→verify (frontman drafts, dormant brain verifies): sound, blocked
   here.** Output distribution equals the verifier's, so quality flows from the teacher while
   the cost model amortizes one trunk read over K drafted tokens - the exact case the
   streaming paper models at factors 3.2-3.7 for K=7 (§9, estimate). Two hard blockers stand
   in the way on this host: Phase A1 measured that speculation cannot run within the safe
   envelope today, and the pair has **heterogeneous vocabularies**, where naive token-ID
   passing corrupts output silently (documented operationally in sglang issue #24051).
   Lossless cross-vocabulary algorithms exist and are merged upstream - SLEM/TLI (Timor et
   al., ICML 2025 oral; up to 2.8x reported) and TokenTiming (ACL 2026; 1.57x typical) - so
   this stays parked behind the RAM blocker rather than deleted.
3. **Teacher-as-draft: rejected permanently.** Drafting runs the drafter per token; making
   the streamed 82.5 GB brain the per-token proposer reinstates exactly the latency the
   architecture exists to remove.

## (d) Pinned ggml 0.19.0 versus a `gemma4` text-only forward

The pinned checkout is pure ggml: tensor operators and backends, **not** model graphs
(verified: the tree contains no llama.cpp model-architecture sources). A Vastloom-side graph
builder is therefore mandatory regardless - the question is whether the required *ops*
exist. They do:

| gemma4 text-forward need | ggml 0.19.0 primitive | Present |
|---|---|---|
| Embedding lookup | `ggml_get_rows` | yes |
| Linear projections (QAT Q4_K/Q5_K/Q6_K/F32/BF16) | `ggml_mul_mat`; types decoded by the repo's own reference kernels | yes |
| RMSNorm | `ggml_rms_norm` | yes |
| p-RoPE (partial rotary dims) | `ggml_rope_ext` `n_dims < head_dim` rotation | yes |
| Sliding-window + global attention | `ggml_flash_attn_ext` with mask (windowing is a graph-level mask/layout concern), incl. `logit_softcap` and attention sinks | yes |
| Unified KV on global layers | graph-level tensor reuse (no op needed) | layout work |
| Gated MLP | `ggml_swiglu_split` / gelu family | yes |
| Softmax, scale, add, views/copies | `ggml_soft_max_ext`, `ggml_scale`, `ggml_add`, `ggml_view_*`, `ggml_cpy` | yes |

(The `gated_delta_net` op also present belongs to qwen35/kimi-linear hybrids; gemma4 does not
need it.)

The genuine gaps are all on Vastloom's side of the boundary:

1. **A `gemma4` graph builder** - the deepseek graph shows this project knows how to do
   this; it is work, not risk.
2. **A SentencePiece tokenizer.** The gemma family uses SentencePiece (Unigram-style)
   vocabularies (~262K entries); `Gpt2Tokenizer` is byte-level BPE only. This is the one
   genuinely new algorithmic component the frontman drags in, and it should be sized before
   the adapter milestone is promised.
3. Loader glue: the GGUF reader and catalog already bind canonical names to shapes/types.

## (e) The device-compute gap

What exists today, verified in-tree:

- CUDA Driver API boundary: capability probe without context retention, Governor-charged
  allocations with live VRAM reserve, synchronous copies (validated smoke on RTX 4070 Super).
- A PTX scalar RMSNorm with CPU-parity smoke (max abs error 0.000025, 2x4096).
- Device matvec kernels for **F32, BF16, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4, IQ1_S, IQ2_XXS,
  IQ3_XXS** - each validated against the CPU-parity oracle - plus a residency-LRU
  accelerator (`CudaMatvecAccelerator`) that keeps hot matrices in VRAM inside the DeepSeek
  streaming path.

That kernel set already covers **every storage type a UD-Q4_K_XL frontman contains**. What
does not exist is everything around them: device-side attention, KV management, softmax,
RoPE, embedding gather, sampling, multi-tensor orchestration, stream pipelines, graph
capture.

Two routes follow, and the pragmatic ordering matters:

1. **Hybrid offload first (recommended):** keep the graph on CPU and delegate frontman
   matvecs to the existing device kernels through the accelerator pattern already proven
   inside the DeepSeek path. This converts the "fast frontman" promise from a full device
   backend into an extension of working machinery, at the price of PCIe transfers per call -
   acceptable only for VRAM-resident weights, which the accelerator's residency logic already
   manages.
2. **Full device graph later:** a much larger programme; treat as its own milestone with its
   own gates, never implied by this vision.

Until one of them lands, the honest description stays as written in the vision document:
*"capable, at CPU speeds"*.

## Consequences fed back into the plan

1. No change to the falsification ladder: E1 mechanics on the frozen SmolLM2 proxy, E2 as
   the deciding milestone, E3 selective wake-up, E4 speculative frontier.
2. New prerequisite surfaced: **size the SentencePiece tokenizer work** before promising any
   `gemma4` adapter milestone (it is the only new-algorithm component).
3. Open decision #4 (device path) now has a recommended answer: **hybrid CPU-graph +
   device-matvec offload first**, full device graph as a separate future milestone.
4. Cross-model speculation stays parked behind the Phase A1 RAM blocker, with the
   heterogeneous-vocabulary algorithms noted as the tool to reach for when it unblocks.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, 2015.
- Agarwal et al., *GKD: Generalized Knowledge Distillation for Auto-regressive Sequence
  Models*, 2023.
- Zhao et al., *LoraRetriever: Input-Aware LoRA Retrieval and Composition*, Findings of ACL
  2024.
- Wu et al., *Mixture of LoRA Experts (MOLE)*, ICLR 2024.
- Huang et al., *LoraHub*, COLM 2024.
- Ostapenko et al., *Towards Modular LLMs by Building and Reusing a Library of LoRAs
  (Arrow)*, ICML 2024.
- *RAMoLE: Retrieval-Augmented Mixture of LoRA Experts*, arXiv 2406.16989.
- *LoRAuter: Effective LoRA Adapter Routing using Task Representations*, arXiv 2601.21795,
  2026.
- *MoLoRA: Composable Specialization via Per-Token Adapter Routing*, arXiv 2603.15965, 2026.
- Timor et al., *Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies*,
  ICML 2025 (arXiv 2502.05202).
- *TokenTiming: Dynamic Alignment for Universal Speculative Decoding*, ACL 2026.
- sgl-project/sglang issue #24051: silent corruption under mismatched draft/target
  vocabularies (operational hazard documentation).
- This repository: `docs/research/CONSUMER_STREAMING_INFERENCE.md` §9 (speculation cost
  model), `docs/research/PHASE_A1_SPECULATION_FEASIBILITY.md` (host-envelope blocker),
  `src/cuda/` (device kernel inventory), `build/research-ggml` @ `30bf868` (primitive
  inventory).
