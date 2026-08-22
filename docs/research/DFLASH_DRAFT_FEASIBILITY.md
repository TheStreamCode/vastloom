# DFlash companion drafter feasibility (dspark inspection)

**Status:** O1 feasibility report, dated 2026-08-21. Inspection only - nothing here runs
the drafter yet. Companion to the speculation record in
`CONSUMER_STREAMING_INFERENCE.md` §9 and `docs/reports/M12` planning, and input to the
Phase A1 reversal documented at the end.

## The artifact

`C:\StargateModels\models\original\deepseek-v4-flash-0731-dspark\dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`
- 10,896,057,440 bytes on disk (10.15 GiB), GGUF v3, 81 tensors, 59 metadata keys.
- `general.architecture = dflash`; Vastloom's existing `deepseek-gguf-inspect` refuses it
  because its catalog gate is pinned to the main checkpoint identity
  (`unsloth/DeepSeek-V4-Flash-0731-GGUF`) - reading it required a standalone header walk.

## What the drafter actually is

A **~19.85-billion-parameter miniature of the teacher itself**, not a head-style drafter:

| Property | Value |
|---|---|
| Layers | **3**, teacher-shaped (same MLA family: q_lora_rank 1024, output_lora_rank 1024, output_group_count 8, indexer top_k 512) |
| Hidden width | 4096 (matches teacher) |
| MoE | 256 experts, 6+1 routing, expert FFN 2048 - the same expert geometry as the teacher |
| Hyper-connections | count 4, sinkhorn_iterations 20 |
| Hash-routed layers | 0 |
| Block decoding | `block_size = 5`, with `markov_w1/w2`, `enc`, `fc`, `conf_proj` tensors (markov block prediction, not autoregressive) |
| Conditioning | `target_layers = [41, 42, 43]` - consumes the teacher's last three layers' hidden states |
| Context / RoPE | 1,048,576 / yarn ×16 from 65,536 |
| Storage | ~18 G elements in MXFP4 (≈9-10 GiB packed) plus Q8_0/BF16/F32 remainder |

Tensor inventory: 24 tensors per block ×3 blocks, then `conf_proj`, `enc`, `fc`,
`markov_w1`, `markov_w2`, `output_hc_base/fn/scale`, `output_norm`.

This is the DFlash paper's design realized against this checkpoint: few deep
teacher-shaped layers, conditioned on target hidden features, predicting whole blocks in
parallel. The conditioning targets being layers 41-43 means the drafter consumes states
the verification pass produces anyway.

## Why the Phase A1 blocker falls

Phase A1 assumed three things, each now superseded:

1. **A resident draft alongside the teacher window.** The companion is an autonomous GGUF
   in the same streamable format family as the teacher (MXFP4/Q8/BF16 - all types with
   validated kernels in this repository). It streams like the teacher: three layers per
   generated block, never 10 GiB resident.
2. **Fixed trunk window 43.** The trunk-window knob and persist-decode are validated
   end-to-end; the RAM equation can be re-swept with a reduced window while both models'
   working sets ride the same Governor budgets.
3. **Conditioning unavailable without training.** False: `target_layers = [41,42,43]`
   names states the graph already computes during every verification pass. Capturing them
   is new plumbing but follows the `hidden_states` precedent.

Honest counterweights recorded with equal weight: the verify pass remains one full
streamed sweep of the dormant brain per block, so the net gain is acceptance amortization
(K ≈ 4-5 accepted of 5 proposed ⇒ plausibly ~2.5-3.5× over the measured 0.3 tok/s,
i.e. ~0.75-1.0 tok/s). That does not make the teacher fast; it makes **oracle calls
(E3) practical** and reopens M12 with a concrete, locally-present drafter. Integration is
milestone-scale work regardless: a `dflash` reader (existing inspector is identity-pinned),
a three-layer runtime reusing the teacher's MLA/MoE/indexer components, mid-forward hidden
capture at fixed layers, markov block decode semantics (reverse-engineered against
llama.cpp's `dflash` implementation), and a two-model Governor preflight.

## Ceiling check: measured on the reference PC

The external llama.cpp reference at `build/research-llama-cpp` (commit `84e908c`) already
ships full speculative wiring for four draft families - MTP, **dflash**, **dspark**,
eagle3 - with `--spec-type draft-dspark` selecting this companion ("dspark outranks
dflash, its sidecar carries the extra Markov head"). `llama-cli.exe` was already built;
no new compilation was needed.

Protocol mirroring §3.2 of the streaming paper: the pinned Italian capital-city question,
greedy (`--temp 0 --top-k 1`), context 512, no warmup, 128 new tokens, CPU only,
conversation mode as shipped. Correctness check: the continuation contains `Roma`.

| Arm | Prompt t/s | Generation t/s | Correctness |
|---|---|---|---|
| Baseline (teacher alone) | 2.2 | **1.8** | Roma ✓ |
| Teacher + DSPark draft (`draft-dspark`) | 1.2 | **1.4** | Roma ✓ (lossless) |

**The speculative arm is 0.78x - slower than the baseline.**

Why, stated honestly: the drafter is not free. Three teacher-shaped layers with their own
256-expert MoE (6+1 active, FFN 2048) must forward once per proposed block, reading its
expert weights from disk each use, on the same CPU cores that would otherwise be sweeping
the teacher. With moderate acceptance on this thinking-mode prompt, the per-block draft
cost exceeds the amortized verification savings. The DFlash paper's 6x comes from GPUs
with the drafter resident in VRAM and the target verified in parallel batches - a
different serving regime than this project's single-stream consumer CPU.

Two positives survive: output is lossless (identical `Roma` continuation), which stands
as live evidence for capability-transfer experiment E4's mechanism - the small model
conditioned on big-model states produces the same tokens; and the host survived the dual
model comfortably (~22 GiB still free after the run), so Phase A1's RAM fear specifically
does not return even with both models mapped.

## Verdict

- Speculative decoding via the native companion is **closed for this project's CPU
  regime**: it cannot beat the plain streaming baseline here, independent of the earlier
  RAM question, which measurement has now retired separately.
- As an E4 artifact, the lossless pairing is recorded evidence that target-conditioned
  state injection works end-to-end on exactly this checkpoint family.
- The 10.9 GiB MXFP4 companion remains catalogued and readable; no further Vastloom
  engineering is planned against it unless the serving regime changes fundamentally.
