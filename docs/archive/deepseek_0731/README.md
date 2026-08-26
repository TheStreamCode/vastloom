# Archive: DeepSeek V4 Flash 0731 execution track (frozen 2026-08-26)

This directory contains the planning and validation documents for the
**DeepSeek V4 Flash 0731** execution track that ran from M0 to M13 on the
reference PC.

## Why archived

On 2026-08-26 the project pivoted to a new execution target:
**Qwen3.8-Flash-Next** (preview of the Qwen4 architecture). The new track is
described in the top-level `docs/ROADMAP.md` (Q1..Q13) and budgeted in
`docs/planning/QWEN38_BUDGET_ESTIMATE.md`.

## What is here

- `VALIDATION_MATRIX.md` — the validation matrix that defended every DeepSeek
  claim up to and including M13 (prefix-KV session reuse on the dense
  SmolLM2 serve path).
- `DFLASH_DRAFT_FEASIBILITY.md` — exploratory work on the DFlash/dspark
  companion, including the empirical 0.78× measurement on the CPU regime.
- `CAPABILITY_TRANSFER_VISION.md` — the north-star vision for capability
  transfer between a fast resident frontman and a dormant brain on NVMe.
  Three independent negative measurements closed this avenue for this milestone.

## What is NOT here

- **Code.** All C++ source under `src/` and all tests under `tests/` remain in
  their original location. They are historical parity contracts; deleting them
  would discard pinned-ggml evidence chains. They stay compiled and verified by
  `vastloom_tests`.
- **Checkpoints.** All `C:\StargateModels\models\original\deepseek-*` payloads
  were removed in the Q0 cleanup. They are no longer on the host disk.

## Status

- All gates listed in `VALIDATION_MATRIX.md` were green as of commit `1529309`.
- `CHANGELOG.md` retains the full milestone history; only the new Q0/Q1 work
  is added forward of that commit.
- The original M14+ section of the roadmap (multi-backend, Kimi K3, Laguna,
  MiniMax, Vulkan/ROCm, multi-NVMe) is preserved in git history as the
  previous content of `docs/ROADMAP.md`. Do not reapply it.
