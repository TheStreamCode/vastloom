# D1/D2 — IQ2_XXS metadata identities and real layer-0 execution from range slices

Validation date: 2026-10-21. Source revision: `546af36` plus the uncommitted DeepSeek refocus
changes. Track: DeepSeek V4 Flash 0731 (`unsloth/DeepSeek-V4-Flash-0731-GGUF`, revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`), `UD-IQ2_XXS` only.

## Scope

This report covers the first two steps of the October 2026 DeepSeek refocus:

- **D1** — pin `UD-IQ2_XXS` metadata identities (complete first split plus independently hashed
  header prefixes for splits 2/3), import a metadata-only store, and derive exact range plans for
  one layer and the six globals;
- **D2a** — execute one real layer (layer 0) from those ranges: trunk, the authoritative
  `tid2eid` route, six complete experts and the six global consumers, with no full checkpoint.

**Not established:** this is one layer of 43. It is not the model's output, not a generation, not
a parity result against ggml/llama.cpp, and not a throughput measurement. The VRAM expert tier
with system-RAM offload was **not** exercised; the payload path still uses the CPU artifact
probe. D2's placement half and any deeper execution remain open.

## What changed for D1

- `src/inference/deepseek_v4_gguf.cpp` — the pinned IQ2_XXS identity now carries separate
  metadata identities: split 1 complete (5,257,664 B), splits 2/3 as 2,097,152-byte header
  prefixes with their own SHA-256. Splits 2/3 payload sizes and hashes are unchanged.
- `src/cli/main.cpp` — `deepseek-gguf-checkpoint`, `deepseek-gguf-probe-plan`,
  `deepseek-gguf-global-plan` and `deepseek-gguf-global-bind` now use the **detected** pinned
  identity (`pinned_spec_for`) instead of hard-coding the IQ1_S spec, so an IQ2_XXS metadata
  store is accepted and an IQ1_S store keeps working.
- `tools/import_hf_model.py` — new alias `deepseek-v4-flash-0731-gguf-iq2xxs` with the pinned
  split sizes/hashes and the two header prefixes.
- `tools/import_hf_ranges.py` — accepts both pinned aliases; plan/source validation is per
  quantization.
- No kernel change: layer 0 stores routed gate/up as IQ2_XXS and down as IQ3_XXS, and both
  reference kernels already existed.

## D1 evidence

Metadata-only store (payload, excluding manifests): **9,451,968 B** = 5,257,664 (split 1,
SHA-256 `c58c9d62…`) + 2 × 2,097,152 (header prefixes, SHA-256 `dbbfe94e…` and `16272402…`).

| Check | Result |
|---|---|
| `deepseek-gguf-checkpoint` on the store | `ok`, `pinned_0731_checkpoint_spec_verified: true`, `reference_config_verified: true`, `layer0_reference_catalog_verified: true`, hashes verified |
| Metadata catalog | 1,328 physical tensors, 34,223 logical tensors, 43 layers, 256 experts/top-6, 11,008 expert bundles |
| Always-active bytes | 6,986,088,796 |
| Routed-expert bytes | 83,869,302,784 |
| Expert bundle range | 7,536,640 – 9,830,400 B |
| Layer-0 trunk plan | 138,867,160 B in 21 ranges, plan SHA-256 `e6274609…` |
| Global plan | 596,000,788 B in 6 ranges, plan SHA-256 `06713153…` |
| Authoritative route (token 0) | experts **254, 222, 245, 200, 53, 35** (`official-tid2eid-verified`), 45,219,840 B in 18 ranges, plan SHA-256 `c9b27ecd…` |

The route was resolved from the imported IQ2_XXS trunk's own `blk.0.ffn_gate_tid2eid` table, not
supplied by the caller. It matches the historical IQ1_S record for the same token, which is a
useful cross-quantization consistency check on the non-routed trunk, not a parity claim.

## D2a evidence: one real layer executed

Command shape (model directory = metadata store artifact; three verify-on-load range artifacts):

```powershell
vastloom deepseek-layer-probe <metadata-store-dir> <trunk-artifact> <expert-artifact> <global-artifact> --token 0 --json
```

Two consecutive runs, cold artifact cache per process:

| Field | Run 1 | Run 2 |
|---|---:|---:|
| `ok` | true | true |
| `complete_model` / `executed_layers` / `model_layers` | false / 1 / 43 | false / 1 / 43 |
| logits | 129,280 | 129,280 |
| logit minimum | −16.1273 | −16.1273 |
| logit maximum | 15.5206 | 15.5206 |
| argmax token | 30,528 | 30,528 |
| experts executed | 254, 222, 245, 200, 53, 35 | identical |
| expert bytes read | 45,219,840 | 45,219,840 |
| trunk bytes | 138,867,160 | 138,867,160 |
| globals bytes | 596,000,788 | 596,000,788 |
| relocated tensors / expert bundles | 45 / 6 | 45 / 6 |
| elapsed | 345.325 ms | 347.433 ms |

The range artifacts reported `local_range_hashes_verified: true`; the selected-expert artifact
also carries `routing_authority_verified: true`. The store held **752.99 MiB** on disk after the
import (metadata store, three artifacts and their manifests), against 780,087,788 B of logical
payload in 45 ranges.

### Missing-expert failure mode

Token 1 routes to experts 239, 202, 122, 23, 115, 57, none of which are in the imported
selected-expert artifact. The probe exits with code 74 (`io_error`) and
`"Unable to inspect buffered I/O source"`, and produces no logits: it refuses instead of
executing nine-tenths of the layer against placeholder or wrong bytes. This matches the failure
mode recorded for the historical IQ1_S layer-0 probe and is re-verified here for IQ2_XXS.

Cross-quantization comparison with the historical M9 layer-0 record (IQ1_S/IQ3_XXS, dated
2026-08): identical expert set and identical argmax 30,528; logits ranged
`[-16.0828, 15.4815]` there and `[-16.1273, 15.5206]` here. The difference is the expected
quantization effect; the two runs are **not** the same experiment and this is not parity.

## Gates

| Gate | Result |
|---|---|
| Release build (MSVC, warnings as errors) | PASS |
| Debug build (MSVC, warnings as errors) | PASS |
| `ctest --test-dir build\\debug` | 14 passed, 0 failed, 1 skipped (`vastloom_ggml_compute`, ggml not configured) |
| Python `compileall`, `test_import_hf_model.py`, `ruff check tools tests` | PASS; the importer test now pins the IQ2_XXS metadata/full identity counts, byte totals, SHA-256 presence, alias and large-download authorization |

## Not established and next blockers

- **No logits oracle.** There is no complete IQ2_XXS checkpoint and no local ggml reference for
  this profile, so the layer-0 logits are deterministic but not yet independently verified.
  D2b needs a bounded oracle (for example a Python dequantization/matvec reference built from
  the same artifact bytes, or a regenerated pinned fixture) before any parity language.
- **No VRAM expert tier on this path.** The payload probe reads experts through the artifact
  cache; the VRAM-resident tier with system-RAM fallback exists only in other paths
  (`--cuda-experts` is Qwen3.8-only today). Exercising the refocus memory contract requires new
  placement plumbing (recon: `ResidencyObject` VRAM has no callers in this path).
- **One layer, not a chain.** `deepseek-forward` requires the complete checkpoint; no executor
  chains range artifacts. D3 needs a new bounded multi-layer executor, and D4 a design for
  on-demand expert ranges (one-token expert union is 1,965,686,784 B; the full always-active
  trunk is 6,986,088,796 B).

## Reproduction

```powershell
# metadata-only store (about 9.45 MB)
vastloom store-init build\analysis\deepseek-0731\store --reserve 20GiB
python tools\import_hf_model.py --model deepseek-v4-flash-0731-gguf-iq2xxs `
  --store build\analysis\deepseek-0731\store --metadata-only --reserve-gib 40

# plans, dry run then import (trunk 138,867,160 B; globals 596,000,788 B)
vastloom deepseek-gguf-probe-plan <store-metadata-dir> --layer 0 --experts 0,1,2,3,4,5 --json > layer0-plan.json
python tools\import_hf_ranges.py --model deepseek-v4-flash-0731-gguf-iq2xxs `
  --plan layer0-plan.json --store build\analysis\deepseek-0731\store --scope trunk --allow-range-download
# authoritative experts from the imported trunk, then the six globals
vastloom deepseek-gguf-probe-plan <store-metadata-dir> --layer 0 --route-artifact <trunk-artifact> --token 0 --json > route-plan.json
python tools\import_hf_ranges.py --model deepseek-v4-flash-0731-gguf-iq2xxs `
  --plan route-plan.json --store build\analysis\deepseek-0731\store --scope selected-experts --allow-range-download
python tools\import_hf_ranges.py --model deepseek-v4-flash-0731-gguf-iq2xxs `
  --plan global-plan.json --store build\analysis\deepseek-0731\store --scope globals --allow-range-download
# one layer, not the model
vastloom deepseek-layer-probe <store-metadata-dir> <trunk-artifact> <expert-artifact> <global-artifact> --token 0 --json
```

The imported payload and store are local validation artifacts under the git-ignored `build/`
tree; they are not part of the repository and must never be committed.
