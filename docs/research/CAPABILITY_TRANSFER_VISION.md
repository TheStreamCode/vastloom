# Vastloom capability-transfer vision: one fast resident model, one dormant brain

**Status:** north star, not a validation claim. Nothing in this document is implemented or
measured until a dated table below says it is. Where the language sounds like a present-tense
fact after one of those tables, read it forwards through the honesty rule in `AGENTS.md`.

## One paragraph

Vastloom is a two-model local-intelligence architecture. A **frontman** — a fast, dense,
resident model that speaks to the user and drives every ordinary inference — is expanded at
runtime by a **dormant brain**: a much larger model kept on NVMe, never fully loaded, that
lends the frontman capabilities it does not have. The bridge between them has one hard rule,
and this entire document is the attempt to uphold it honestly:

> **The teacher pays once, or it pays only for the step that matters. The user's per-token
> latency and memory stay those of the frontman.**

What crosses the bridge is *not* the dormant brain's raw parameters, and it is *not* magic.
It is one of three real, bounded things:

| Mechanism | What crosses the bridge | When the teacher pays | Runtime cost to the frontman |
|---|---|---|---|
| **Compiler (offline distillation → compact adaptive modules)** | a low-rank skill module distilled from the teacher's behaviour | once, offline, per skill | none — the module is already on the frontman's weights |
| **Oracle (selective test-time wake-up)** | the *result* of one teacher forward pass on one difficult step (verdict, draft, judgement) | rarely, for that step only | bounded selective streaming, already staged |
| **Retrieval (knowledge, behaviours, exemplars)** | embeddings and small documents, never parameters | precomputed | a small fixed prefix |

A fourth row — extracting a slice of the teacher's parameters and splicing it into a
different model — is recorded as **excluded**, not because it would be hard, but because it
does not exist: capability in a trained language model is distributed, entangled and lossy
over millions of weights in every layer, and a weight vector from a MoE transformer is not
meaningful inside a dense model with a different architecture. The bridge is built from the
three rows that are real.

The claim the project is eventually allowed to make is never "x100 intelligence". It is
task-scoped and measured: *on task T, frontman + modules beats frontman alone by Y and
approaches teacher-oracle by Z, at frontman latency L*. If the numbers do not show that, the
vision shrinks to what they do show.

## Why this is not "already the MoE streaming"

Vastloom's current executable path streams one 82.5 GB model and activates a 1.77 GB
expert slice per token. That is *one* model with parts that sleep. The vision here is
different and additional: **two independent models, two jobs, one governed bridge.**

- The MoE streaming answers "how does a single too-big model run at all on consumer
  hardware". Its answer is correct and already validated.
- The two-model bridge answers the question the MoE path cannot: "how does a model that is
  *fast* to run also get to be *capable* beyond its own weights, without inheriting the
  latency of the capable model". The MoE frontman is not fast (0.3 tok/s CPU); a dense
  resident frontman is. The bridge exists to give the fast one the capability the slow one
  has, on demand.

The two ideas share the substrate — the Safety Governor, the immutable store, the identity
and hash boundaries, the bounded streaming — but they are different products with different
claims. This document owns the second one.

## The reference instantiation (a selection, not a commitment)

The models below are the current best choices *for the two jobs*, judged by what is applied
rather than what is popular. They are replaceable; the bridge is agnostic to them.

### Frontman — `google/gemma-4-12B-it` (QAT `UD-Q4_K_XL`, ~6.72 GB)

- A 12B dense, encoder-free (12B "Unified") instruct model with a **QAT Q4** checkpoint that
  trades ~3× memory for near-BF16 quality — intelligence preserved rather than quantized
  away, which is the point of the "small brain that surprises".
- **MTP drafter included** (shares the target KV, verifies every token) — speculative
  decoding arrives with the model rather than as a later project.
- **Native function calling and `system` role** — the substrate the orchestrator needs.
- Fits a 12 GiB VRAM reserve comfortably with headroom; a 27B-class dense frontman does not.

### Dormant brain — `deepseek-v4-flash-0731` (IQ2_XXS, 82.5 GB)

- The one streaming path Vastloom has actually executed and hash-verified, with
  authoritative routing. It is the perfect dormant: too slow to be a frontman, excellent as
  a rarely-woken reserve.
- MTP block already catalogued (`num_nextn_predict_layers`), so an oracle-as-draft path has a
  known landing point.

Both choices are revisable in the light of the F1 evidence (see below). The honest
alternatives table: a gemma-31B or a 26B A4B MoE as a second teacher for homogeneous
architecture, and any standard dense 7–8B for a cheaper frontman, are open options, not
decisions taken.

## Speed dependencies (the promise behind the bridge)

The bridge adds **capability**, never speed. Every speed claim in this vision therefore
reduces to one question: *what executes the frontman?* The three scenarios below are
bandwidth-derived estimates for the reference PC (31 GiB RAM, RTX 4070 Super), not
measurements; none of them has been run.

| Frontman execution path | Arithmetic | Honest outcome |
|---|---|---|
| **Resident in VRAM + a real device model backend** | 6.72 GB of weights over ~504 GB/s gives a ~74 tok/s decode ceiling; mature kernels reach a fraction of it | **~30–60 tok/s** — the product promise. But Vastloom's device layer today is reference kernels plus one RMSNorm smoke, not a model backend |
| **Resident in RAM, CPU** | 6.72 GB per token over ~30–50 GB/s of dual-channel memory caps at ~4–7 tok/s before any kernel inefficiency | **low single digits (~1–4 tok/s)** with today's scalar/ggml-delegated kernels — an order above the measured 0.3 tok/s DeepSeek path, but not the snappy frontman experience |
| **Oracle wake-up (either path)** | one full-depth teacher pass moves at least the measured 8.76 GB/token budget | a verdict costs **seconds to minutes** depending on prompt length; acceptable only if rare and declared to the user |

Two modifiers apply to whichever path wins:

- **MTP/speculative decoding** multiplies throughput — the paper estimates factors of
  3.2–3.7 for K=7 on the streaming cost model, but Phase A1 found speculation unmeasurable
  within this host's safe envelope, so treat it as an estimate with no hardware number yet.
- **The capability modules must be latency-free by construction**: an injected adapter adds
  flops proportional to its rank, and that overhead is part of E1's acceptance criteria, not
  an afterthought.

The consequence is stated once and plainly: **the "fast frontman" promise depends on a
device compute path that does not exist yet.** Until it does, the honest description of this
system is "capable, at CPU speeds".

## What "capability module" means, operationally

The compiler's product is the thing that must be cheap to store, cheap to retrieve, and
cheap to apply on a frozen base. The working definition, to be pinned by experiment not by
assertion:

- **One skill module = one low-rank adapter** (the `A`/`B` pair of a LoRA, applied to frozen
  linear layers) **+ one small routing embedding** (what the index matches against a request).
- **Cost per module** is a first-class number: `bytes_on_disk`, `rank`, `flops_added` — with
  a ceiling per module so the store never pretends a module is free.
- The **base stays frozen**. A module changes behaviour through the adapter, never through a
  rewrite of the frontman's weights; the frontman remains the verified, immovable asset, and
  a module is a derived, validated artifact under the same immutable-store rules.

## The four machinery pieces (this is the scaffold)

1. **Compiler (offline).** Turns "teacher behaviour on skill S" into a compact module. The
   only physiologically-grounded form is behaviour distillation — matching outputs/logits on
   curated queries — which is *training on the module*, offline, once. This is the only way
   the phrase "without inferring the teacher" is literally true at runtime.
2. **Index / router (runtime).** Retrieves k modules by embedding proximity to the request,
   bounded, with the same LRU/lease discipline the expert cache already has.
3. **Injector (runtime).** Applies the adapter on the frozen base for the duration of the
   request and tears it down; measured overhead, no silent reshaping of the base.
4. **Oracle dispatcher (runtime).** Wakes the dormant brain rarely: when the frontman's
   self-confidence is low or the task is flagged hard, and the Governor's budget allows it.
   The wake-up does one bounded teacher forward pass on one step and returns a verdict,
   a draft, or a judgement. Nothing about this watcher may outrank the Safety Governor.

The whole system is **model-agnostic**: a LoRA slot sits on any linear layer, the index does
not know what emits the embedding, and the dispatcher does not know what sleeps behind the
streaming path. That is what makes the frontman selection reversible.

## The 100× question (answered with a falsification ladder, not a slogan)

The vision was born as "x100 intelligence from a small model plus a big one on disk". That
number is not measurable in general. The ladder below converts it into four experiments,
cheapest first, each with a pre-declared verdict:

| Id | Experiment | What it would establish | Pre-declared failure |
|---|---|---|---|
| **E1** | inject a LoRA module on a frozen base, retrieve it by embedding, verify parity | the *mechanics*: compact modules are storable, retrievable, injectable | parity breaks, or module cost dominates |
| **E2** | distill one skill module from the DeepSeek teacher (offline, via the existing selective streaming) and measure `frontman+module` vs `frontman` vs `teacher` | whether transferred **capability** is real and bounded | the adapted model regresses, or gains only noise |
| **E3** | oracle wake-up: one bounded dormant pass as verdict/draft on a hard step | whether **selective wake-up** pays (quality vs cost curve) | the wake-up costs more than the capability it adds |
| **E4** | cross-model *state* transfer (the large model pre-computes hidden/KV state once; the frontman conditions on a projection) | whether a **test-time transfer** without per-token teacher cost exists | no transfer is observed |

E2 is the milestone that decides the vision. E4 is the frontier: it is the literal reading
of "give the small model the big model's power without running the big model per token", and
it is also the experiment most likely to be a documented negative result. It is scheduled
last and clearly marked speculative.

### E1 status — executed 2026-08-21

Mechanics shipped and gated: bounded VLCM container with SHA-256 trailer, deterministic
FNV-trigram router, overlay kernel applied after each wrapped matvec on the frozen base,
`vastloom module-init` for deterministic synthetic modules, and
`chat --capability-module` with full JSON cost accounting. Always-on tests cover container
corruption refusals, routing determinism and a double-precision delta oracle; the opt-in
SmolLM2 contract proves on the real pinned checkpoint that an inert (`--zero-a`) module
leaves greedy output bit-identical while an active module stays deterministic.

Measured on the reference run (rank 8, scale 0.05, all 150 targets of SmolLM2-135M):
78,796,800 added MACs across 8 generated tokens (~9.9M per token, roughly 7% of the dense
model's own matvec work) in an 8.3 MB module. Verdict against the pre-declared failures:
**parity held; cost is bounded and counted.** First finding recorded: that delta magnitude
degenerates greedy output to repeated token 0, so magnitude/coverage tuning belongs to E2's
compiler design rather than being an afterthought.

### E2 status — started 2026-08-21

The architecture fork is decided: **the module compiler is offline Python tooling**, in the
same dependency class as the HF importer - never a runtime dependency of the native engine.
Training infrastructure is standard engineering best bought whole; the native side stays
out of gradient computation and consumes only published modules through the validated VLCM
container. Targets are vocabulary-free by construction (prompt -> completion text and ids),
because teacher and student tokenizers do not align.

Slice 1 shipped: `tools/distill_transcript.py` builds the teacher-transcript artifact
(`vastloom.distill_transcript.v1`) by driving `deepseek-forward` over a committed probe set
(`tests/fixtures/distill_probes.json`) - bounded inputs, atomic publication, injectable
runner, per-record runtime provenance, and a resume mode that persists after every
successful probe because the host's RAM budget can refuse the next one at any time.

**First real transcript executed 2026-08-21** (3 records, all `teacher_correct=true`),
and it immediately produced the finding that justifies the whole gate: on arithmetic
word problems the IQ2_XXS greedy teacher is itself wrong (1 of 3 correct), so arithmetic
is **excluded as a distillation target** - distilling from wrong targets would bake the
errors in. The chosen skill is factual recall plus translation, where the measured gap is
maximal: teacher 3/3 correct against a SmolLM2 baseline that fails all six probes by
echoing them. Run conditions recorded honestly: window 43 and 32 were refused under host
memory pressure (9.42 GB required against 8.2 GB falling to 4.2 GB available), which
surfaced the preflight's accounting - resident trunk is charged as
`min(total, average x (window+1) x 2)`, so shrinking the window only helps below ~20 -
and the run landed at window 8 with persist engaged and a three-attempt resume loop.
`deepseek-forward --json` prints two objects (preflight, then generation); the tool
decodes them stream-wise and keeps the one carrying tokens.

**Compiler executed 2026-08-21, completed same day** (`vastloom module-fit`, closed-form
LM-head distillation - no autograd engine exists or is needed when the adapter sits at the
network's last linear layer): cross-entropy on the 11 supervised positions fell 4.26 -> 0.0
unregularized, producing a 1.59 MB VLCM module that led every seen probe with the teacher's
answer - but an off-skill probe the base answered correctly degraded into target-token
debris: catastrophic interference.

The recorded fix lever was implemented and measured. A first attempt (squared-delta spring
on the logits) diverged at any weight - the spring feeds back through u and resonates -
which surfaced as a fail-closed refusal rather than a NaN module. The shipped mechanism is
**decoupled weight decay**, AdamW-style: decay never enters the gradient; each step ends
with a multiplicative contraction clamped at `max(0, 1 - lr*reg)`, so no hyperparameter
pair can overshoot. With a train/eval probe split and a reproducible evaluator
(`tools/distill_evaluate.py`, strict-equality plus leads-with grading), the sweep found the
operating point at **`--reg 30`**:

| Probe | Split | Base | Frontman+module (reg=30) |
|---|---|---|---|
| chitchat-1 ("Hello") | eval | correct | **correct - preserved verbatim** |
| capital-2 (train) | train | echoes prompt | **leads with "Canberra"**, grammatical continuation |
| translate-1 (train) | train | echoes prompt | cross-contaminated, not leading |
| capital-1 (train) | train | echoes prompt | wrong (adjacent city) |
| capital-3/4, translate-2 | eval | wrong | still wrong - no generalization |

Verdict, stated plainly: **off-skill interference is solved** (the recorded defect closed),
transfer is real but partial at this data scale, and generalization to unseen probes of the
same skill does not exist yet. The remaining levers are data and placement, not mechanics:
more transcript records, KL anchors over generic contexts, adapter placement beyond the LM
head, and an explicit train/eval transcript split grown beyond three records.

### Data lever executed 2026-08-21: measured negative for this placement

The first named lever - more transcript records - was tested directly. The probe set grew
to eight train probes (five capitals, three translations) and seven eval probes; the
teacher transcribed all eight correctly under the resume loop (one expected-answer fix:
the teacher wrote "Brasília" with its proper accent and the accuracy gate rightly flagged
the fixture's accent-less expectation - corrected fixture-side, backfilled from the stored
text without a new teacher run). Refit at rank 8 / 600 steps / lr 0.05, swept:

| Config (29 positions) | Train exact | Train leads-with | Eval correct | chitchat-1 |
|---|---|---|---|---|
| `--reg 10` (loss -> 0.00) | 3/8 | 5/5 | **0/7** | broken |
| `--reg 30` (loss -> 0.78) | 2/8 | 1/1 graded | 1/7 | preserved |
| `--reg 100` (loss -> 2.22) | 0/8 | 0/0 | 1/7 | preserved |

Growing the data nearly tripled the supervised positions and full train coverage appeared -
but held-out same-skill generalization stayed at zero for every weight, and the
fit-versus-off-skill tradeoff sharpened rather than softened. The conclusion is now
measured, not speculated: **with an LM-head-only adapter, capability transfer is retrieval
of trained answers, not an induced skill.** Data growth alone is closed as a dead end for
this placement; the blocking lever is adapter placement beyond the LM head (or richer
target spaces), which requires machinery this project deliberately does not have - a
backpropagation path through the network - and therefore becomes a design decision for the
next milestone rather than an incremental step.

### Head-boundary closed 2026-08-21: both placements measured

The second placement lever was then instantiated end-to-end: a **final-hidden adapter**
(`CapabilityTarget::final_hidden`), injected before the frozen head consumes the normalized
state - `logits = W(h + B(A h))` - with gradients still closed-form through the head
transpose. Two real defects surfaced and were fixed by measurement, not guesswork: an
unsigned-underflow in the shared splitmix initialization turned "negative" draws into
~+0.33 positives (harmless at the LM-head site, explosive through Wᵀg here), and the
two-factor adapter resonated through ‖W·B‖ until the projection was frozen at its
deterministic init. Gradient clipping bounds every remaining step.

The measured sweep on the grown transcript (29 positions, rank 8, 1200 steps) was **stable
everywhere and behaviorally flat**: best loss 3.34 -> 3.19, zero movement on any probe,
chitchat preserved only because the adapter barely moves anything. The three-way picture:

| Placement | Fit ability | Generalization | Off-skill safety |
|---|---|---|---|
| LM head (logit-space) | strong (loss -> 0) | none | breaks unless reg ≈ 30 |
| final_hidden (frozen proj., closed-form) | ~none under stable hyperparameters | none | safe |
| mid-layer | requires backprop through the network - machinery this project deliberately lacks | untested | — |

**The pre-declared criterion fires: the head boundary is closed from both sides.** Neither
logit-space adaptation nor closed-form hidden-space adaptation induces generalizable skill;
real capability transfer therefore requires gradients through the network itself - an
optimizer beyond this project's deliberately closed-form scope. The remaining choice
(offline torch trainer versus strategic pause back to M13 prefix-KV) now rests on measured
ground rather than prediction, and every instrument built along the way - container,
router, injector, transcript pipeline, evaluator - remains reusable under either branch.

## The phases

**F0 — this document** plus the honesty rules that make it falsifiable.

**F1 — evidence synthesis.** Five questions, answered against current literature and the
pinned ggml checkout before any code: (a) is cross-model behaviour distillation into compact
adapters sound, and at what floor cost; (b) what retrieval over adapters is supported, not
aspirational; (c) cross-model MTP/speculative decoding; (d) what the pinned ggml 0.19.0
back-end carries for `gemma4` ops (p-RoPE, sliding/global attention, unified KV on global
layers) versus what a text-only adapter would have to implement; (e) what stands between
today's device layer and full GPU model execution for the frontman — extending the CUDA path
beyond its reference kernels versus delegating the whole graph to the pinned ggml checkout —
because every "fast" claim in this document depends entirely on that answer. Output: the
feasibility matrix that converts the frontman/teacher selection from "chosen" to
"defensible". **Completed 2026-08-21:** see
`CAPABILITY_TRANSFER_FEASIBILITY.md` — verdicts: (a) sound engineering, gain unknown until
E2; (b) retrieval/composition supported by mature published systems; (c) frontman self-MTP
shipped, cross-model speculation sound but host-blocked; (d) all ggml primitives present,
gaps are graph-builder plus a SentencePiece tokenizer; (e) every needed storage type already
has a parity-validated device matvec kernel — hybrid CPU-graph/device-matvec offload is the
recommended first device route.

**F2 — the falsification ladder (E1→E4), cheapest first.** E1 on the frozen SmolLM2 base for
speed first, then on the real frontman; E2 against the DeepSeek teacher; E3 reusing the
existing selective-flow machinery; E4 only if budget and appetite remain.

**F3 — the scaffold.** The governed two-model runtime: resident frontman in VRAM, dormant
teacher streamed from NVMe, offline compiler publishing immutable modules into the store,
index router, injector, and the oracle dispatcher under the Safety Governor. This is
Vastloom 2.0, and it is the only place the word "Vastloom" names the *bridge* rather than
the single-model streaming runtime Vastloom already is.

## Safety invariants inherited, unchanged

The bridge adds no new authority to break the existing invariants. The frontman and the
dormant brain are both immutable model weights; a capability module is a derived artifact
published only after temporary creation, validation, checksum and atomic rename. Every
queue, cache, wake-up count, adapter slot and retrieval list is bounded. On pressure the
bridge reduces wake-ups, retrieval and adapters first — it never reduces the reserve. The
Safety Governor outranks the oracle dispatcher, the compiler and the performance goals
alike.

## Decisions still open (do not build past them)

1. Teacher: DeepSeek V4 Flash 0731 is the default (only validated stream path); a
   homogeneous gemma-31B/26B-A4B teacher is an open alternative.
2. `gemma4` adapter ordering: build the text-only slice first (global layers, unified KV,
   p-RoPE), deferring vision/audio.
3. F2 base: prove the bridge on frozen SmolLM2 first, then on the real frontman.
4. Frontman device compute path: extend the CUDA model backend beyond its reference kernels,
   delegate the full graph to the pinned ggml checkout, or accept CPU speeds for the first
   slices. This decision gates the "fast" half of the promise and belongs to F1(e).
