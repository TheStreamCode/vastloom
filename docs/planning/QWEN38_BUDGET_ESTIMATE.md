# Qwen3.8-Flash-Next — Budget Estimate (Phase 0, revised)

> **Status: REVISED DRAFT, NOT VALIDATED.** Every number below is declared by the model
> publisher or read from remote `config.json`/GGUF headers/Unsloth docs. None of it is yet
> the result of a download + run on the reference PC. This budget gate is "paper, not
> copper".
>
> **Binding user decisions (revised):**
> 1. Full model switch (DeepSeek 0731 → Qwen3.8-Flash-Next)
> 2. Native C++20 operators for all three new primitives (DeltaNet, QSA, n-gram)
> 3. **Quant: Unsloth FP8** (not IQ, not UD-IQ1_S) — lossless-quality constraint
> 4. Roadmap Q1..Q13, archive only the DeepSeek 0731 docs
> 5. **1-2 weeks of infrastructure preparation** before downloading the 125B
> 6. **POC on Qwen3.8-27B FP8** as pre-validation
> 7. Cancel and stop if Phase 0 fails

---

## 1. Why FP8 instead of IQ

### Key Unsloth Dynamic 3.0 result (from `https://unsloth.ai/docs/basics/dynamic-3.0-ggufs`)

> *"There is a sharp drop off from UD-Q2_K_XL to UD-IQ2_S for 32 token prediction from around
> 25% accuracy to under 8-10%. This sharp drop off means tool calling and non thinking modes
> break down."*
>
> *"1-bit should not be used for agentic use-cases"*

For Gemma 3 27B (Unsloth Dynamic 3.0 reference, same calibration techniques):

| Quant | Disk (GB) | KLD vs BF16 | Efficiency (KLD/GB) | Notes |
| --- | --- | --- | --- | --- |
| UD-IQ1_S | 6.06 | 0.972 | 0.16 | failed, ~72% top-1% accuracy |
| UD-IQ2_XXS | 7.31 | 0.521 | 0.071 | failed, tool calling broken |
| UD-Q2_K_XL | 9.95 | 0.221 | 0.022 | **quality sweet spot** |
| UD-Q3_K_XL | 12.76 | 0.081 | 0.0067 | good |
| UD-Q4_K_XL | 15.64 | 0.024 | 0.0015 | lossless-ish |

### RAM gate for Qwen3.8-Flash-Next (extrapolation)

Linear scaling from IQ1_S = 67.55 GB disk, ~10.7 GB live (our earlier estimate):

| Quant | Disk estimate | Live estimate | 12.3 GB gate? | Verdict |
| --- | --- | --- | --- | --- |
| UD-IQ1_S | 67.55 GB | ~10.7 GB | borderline | failed on quality |
| UD-IQ2_XXS | ~79 GB | ~11.7 GB | borderline | failed on quality |
| **UD-Q2_K_XL** | **~108 GB** | **~16 GB** | **fail** | quality sweet spot, **fails RAM** |
| UD-Q3_K_XL | ~149 GB | ~22 GB | fail | fail |
| UD-Q4_K_XL | ~191 GB | ~28 GB | fail | fail |
| **Unsloth FP8** | **~85 GB (to verify)** | **~10-12 GB** | **likely pass** | lossless-ish |

**Conclusion**: no IQ quant reaches lossless quality **and** fits the gate. FP8 is the only
option that combines:
- Lossless-ish (KLD << 0.1 vs BF16, similar to Q4_K_M or better)
- Live RAM comparable to IQ2_XXS (FP8 tensors weigh about as much as IQ2, but the numeric is
  dense)
- Officially supported by llama.cpp (native FP8)
- Immediately available for Qwen3.8

---

## 2. Unsloth FP8 for Qwen3.8-Flash-Next

### HF catalog

- Repo: `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-FP8`
- Base: `Qwen/Qwen3.8-Flash-Next` (BF16, 360 GB, 131 shards)
- Also: `Qwen/Qwen3.8-Flash-Next-FP8` (official Qwen)

### FP8 disk estimate

From 360 GB BF16 → FP8 halves it in theory → ~180 GB theoretical. In practice, with the
n-gram table and metadata offloaded: **estimated ~85 GB** (by analogy with the FP8 of other
100B+ MoE models).

### FP8 live RAM estimate (this is the point to verify in Q2)

- One-layer BF16 trunk ≈ ~6.99 GB on DeepSeek
- Same trunk in FP8 ≈ ~3.5 GB (half)
- KV/DeltaNet state ≈ unchanged
- Selected experts in FP8 ≈ ~0.9 GB (half of the 1.77 IQ figure)
- N-gram offload 4 GB (scenario B)
- **Estimated total: ~9-11 GB live** ← likely pass of the 12.3 GB gate

**To be measured in Q2** with a POC on Qwen3.8-27B FP8.

---

## 3. Revised work plan (1-2 weeks of preparation)

### Phase 0 — Budget estimate (this document)

Status: **complete**.

### Q0 — Environment setup (this week)

**Goal**: prepare the native C++20 test infrastructure for the three new primitives WITHOUT
downloading Qwen3.8-Flash-Next FP8 (85 GB).

**Tasks**:
1. Verify that `llama.cpp` mainline has `qwen4_exp` support with DeltaNet, QSA, n-gram and
   Gated Residual. If yes: clone its pinned commit (e.g. ggml 0.19.0 commit 30bf868 as for
   DeepSeek). If not: plan independent development.
2. Add `include/vastloom/inference/qwen3_8_flash_next.hpp` as an empty skeleton (types,
   config, no implementation).
3. Add an always-on C++ test `test_qwen38_metadata_inspector` that reads a mock metadata file
   (NOT a real checkpoint) to validate the `config.json` parser.
4. Commit "Add Qwen3.8-Flash-Next inspector skeleton".

### Q1 — POC on Qwen3.8-27B FP8 (~15 GB disk)

**Goal**: download and validate a single FP8 layer of Qwen3.8-27B (the smaller "sibling"
model, same architecture) to confirm the native C++20 pipeline holds end-to-end.

**Tasks**:
1. Download `unsloth/Qwen3.8-27B-FP8` (~15 GB, includes the n-gram table + the 27B)
2. Implement the catalog reader for FP8 tensors (new dtype)
3. Implement a pinned-ggml FP8 → F32 decoder
4. Implement **a single reduced layer** (DeltaNet + shared MoE + optional QSA):
   - 1-token forward
   - Comparison against pinned-ggml llama.cpp
   - If bit-exact: record **Phase 0 green**, proceed
   - If not bit-exact: diagnose and fix
5. Live RAM measurement: confirm <12.3 GB in a single-token forward

**Cancel and stop** if FP8 does not close the pinned-ggml parity (exceptional situation).

### Q2 — Pinned-ggml full-model reference

**Goal**: the local llama.cpp runs Qwen3.8-27B FP8 end-to-end, 5 oracle tokens produced.

Only after Q1 is green.

### Q3 — Tensor & expert catalog for Qwen3.8-Flash-Next FP8

**Goal**: enumerate every tensor of the 125B FP8 (layer trunks, shared, routed, n-gram
tables, gated-residual, DeltaNet state, QSA indexer).

Only after Q2 is green.

### Q4 — Range importer + safety governor

Three-object gate as for DeepSeek 0731.

### Q5 — Pinned-ggml native operators

FP8 decoder (new), DeltaNet state, QSA indexer, n-gram tables, Gated Residual.

### Q6 — Non-routed 48-layer graph

Hybrid 3 linear_attn + 1 full_attn × 12 structure.

### Q7 — Reduced-layer prefill/decode (2-3 layers + MTP)

### Q8 — MoE routing 10+1

**No hash shortcut** (unlike DeepSeek).

### Q9 — Gated DeltaNet state persistence + resident n-gram lookup

### Q10 — QSA sparse attention

### Q11 — Full-depth generate validation (48 layers)

### Q12 — Prefix-KV MoE stream (M13 port)

### Q13 — `vastloom serve` for Qwen3.8

---

## 4. Updated binding decision

> **DOWNLOAD `unsloth/Qwen3.8-Flash-Next-FP8` ONLY AFTER the POC on
> `unsloth/Qwen3.8-27B-FP8` has confirmed:**
> 1. Pinned-ggml parity on at least 1 complete layer (DeltaNet + shared MoE)
> 2. Live RAM < 12.3 GB in a single-token forward
> 3. Preparation time: 1-2 weeks
>
> If the POC fails: **cancel and stop** (by user constraint).

---

## 5. Reference files

- HF: `https://huggingface.co/Qwen/Qwen3.8-Flash-Next`
- HF (config): `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`
- GitHub: `https://github.com/QwenLM/Qwen3.8-Flash-Next`
- Tech report PDF: `https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf`
- Unsloth FP8: `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-FP8`
- Unsloth 27B (POC target): `https://huggingface.co/unsloth/Qwen3.8-27B-FP8`
- 27B GGUF (for calibration): `https://huggingface.co/unsloth/Qwen3.8-27B-GGUF`
- Unsloth docs Qwen3.8: `https://unsloth.ai/docs/models/qwen3.8-next`
- Unsloth Dynamic 3.0: `https://unsloth.ai/docs/basics/dynamic-3.0-ggufs`

---

**STOP. Awaiting user approval to proceed with Q0 (environment setup).**
