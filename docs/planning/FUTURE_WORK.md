# Future work and how to resume

**Status: research record concluded (October 2026). This file is the memory of what could be
attempted next; it is not a commitment, an active milestone or a claim.** Nothing in the backlog
below has been implemented or measured beyond where a dated report says so. Read it together with
[`ROADMAP.md`](../ROADMAP.md), [`DEEPSEEK_V4_FLASH.md`](../DEEPSEEK_V4_FLASH.md) and the validation
reports under [`reports/`](../reports/README.md).

## 1. Where the record stands

The DeepSeek V4 Flash 0731 reduced-payload track closed D1 through D4 (including D3b/D3c), and the
Qwen3.8-Flash-Next track remains comparison/validation-only. What is measured and defensible:

| Area | Closed gate | Evidence |
|---|---|---|
| Metadata and plans | IQ2_XXS metadata-only store, exact range plans/imports | [`D1/D2a`](../reports/D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md) |
| One real layer | Layer 0 from 780,087,788 B of verified range payload | [`D2a`](../reports/D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md) |
| Device residency | Expert matrices in VRAM with bounded CPU fallback, bit-identical | [`D2b.1`](../reports/D2B_DEVICE_RESIDENCY_VALIDATION.md) |
| Layer-0 oracle | Independent NumPy reproduction within 9.54e-07 on logits | [`D2b.2`](../reports/D2B_LAYER0_ORACLE_VALIDATION.md) |
| Hash-routed chain | 2/3 layers deterministic, GPU chain bit-identical | [`D3`](../reports/D3_HASH_CHAIN_VALIDATION.md) |
| Chain oracle | Tokens 0/1, depths 1–3, within 1.91e-06 on logits | [`D3b`](../reports/D3B_CHAIN_ORACLE_VALIDATION.md) |
| Multi-token prefill | 5-token prompt opens the layer-2 4:1 window, within 1.91e-06 | [`D3c`](../reports/D3C_MULTITOKEN_ORACLE_VALIDATION.md) |
| On-demand design | Byte budget and closed decisions, no fetch | [`D4`](../planning/D4_ON_DEMAND_EXPERT_RANGES.md) |

Key byte facts for planning: always-active weights **6,986,088,796 B**; 43 layer trunks
**6,390,088,008 B**; globals **596,000,788 B**; one-token expert union **1,965,686,784 B**;
a full-depth no-cache token **8,951,775,580 B**; metadata store **9,451,968 B**. The reference PC
has 32 GiB RAM and 12 GiB VRAM.

## 2. Non-negotiables before any future work

These come from the repository's safety invariants and the D-gate entry criteria; they apply to
every backlog item below.

1. The exact pinned revision is verified first; a header prefix never becomes an executable source.
2. Payload enters only through the pinned range importer with explicit authorization, byte cap and
   store free-space reserve; ranges are rehashed and published atomically.
3. The Governor preflight passes from **live** capacity; refusal is a valid outcome, not a bug to
   bypass.
4. A mechanism gets an always-on reduced-graph test before real-weight work, and every claim waits
   for a dated report under `docs/reports/`.
5. No full-checkpoint download, no IQ1/`UD-IQ1_S` execution, no silent fallback, no throughput or
   full-model claim from a partial-depth result.
6. Keep the server/default path local-only: any network fetch is opt-in, bounded and fail-closed.

## 3. Backlog (ordered by dependency, not by attractiveness)

### F1 — VRAM *payload* tier for expert bundles

**Why.** D2b.1 keeps expert **matrix copies** in VRAM, but bundles still enter governed RAM first.
The declared memory contract ("selected experts placed in VRAM with a bounded fallback to system
RAM") is therefore only half implemented.

**Scope.** Use the existing `ResidencyObject` RAM↔VRAM transitions (they have no production caller
on this path) for the per-layer expert payload, with a bounded RAM fallback when the bundle cannot
live on the device. Keep the CPU path as the correctness reference.

**Entry criteria.** No new checkpoint needed: the current layer-0 and chain artifacts are enough.
An always-on fake-backend test for the residency transitions and an opt-in GPU test with a forced
cap are required before real weights.

**Gate.** Bit-identical logits/hidden against the CPU path for layer 0 and the chain; telemetry
reports per-tier resident bytes, transfers, evictions and exact fallback counts; the Governor
refuses an over-commit instead of evicting something else. No speed claim.

**Risks.** Windows commit charge, VRAM fragmentation, and confusing logical bytes with device
traffic. A slower device path is an acceptable measured result.

### F2 — Learned-router layers (layer 3 and deeper)

**Why.** Layers 0–2 route by `tid2eid`; from layer 3 the router uses the post-attention hidden
state, so the expert set is knowable only during execution. This is the hard boundary that keeps
D3 at three layers.

**Scope.** A chain executor that carries the hidden state across a learned-router layer, runs only
that layer's router first, then resolves its six experts. Two admissible data plans: import all 256
expert bundles of the chosen layer (about 1.9 GB for one layer) so the route can be served
locally, or implement the on-demand fetch of F4.

**Entry criteria.** The layer-0..2 chain (D3a/D3c) and the oracle extension that produces the
pre-attention hidden state; a fixed layer-3 trunk artifact and its plan; a clear statement of
which expert-coverage plan is used.

**Gate.** Layer 3 executes from real payload with the route resolved from the actual hidden state;
its logits/hidden match an independent oracle (the NumPy chain can be extended to run the learned
router) within the established tolerances. One layer beyond the hash prefix is the first honest
target, not full depth.

**Risks.** A locally imported expert set can silently cover only some routes; coverage must be
validated, not assumed. Without F4, storage grows by hundreds of MB per token set.

### F3 — Ratio-128 compressed layers

**Why.** The checkpoint's compression schedule is `[0, 0, 4, 128, 4, 128, ...]`; layer 3 uses the
128:1 compressor, whose 25-object layout is catalogued but not modelled by the oracle (it refuses
it precisely). The 4:1 path is exercised by D3c; 128:1 is not.

**Dependency.** Ratio-128 layers are ≥3, so this needs F2 (learned router) first. It also needs
the per-layer indexer/compressor equations for the 128:1 layout, which exist in the C++ reference
graph but must be re-implemented independently for an oracle.

**Gate.** A real ratio-128 layer executes and matches the independent oracle with a non-empty
compressed window; stage counts prove the window is used. No full-depth claim.

### F4 — On-demand expert range fetch (implements the D4 design)

**Why.** This is the only plan that keeps the local footprint near the 10 GB budget while allowing
thousands of experts. The design is complete; the implementation is not.

**Scope.** A bounded, opt-in remote range reader for selected expert bundles: exact HTTP ranges
from the pinned plan, SHA-256 on first touch and on reuse, retries/timeouts, disk bundle cache
2,000,000,000 B, staging 268,435,456 B and RAM expert cache 1,073,741,824 B, all under Governor
reservations and fail-closed on any mismatch. Default remains local-only.

**Entry criteria.** F1/F2 decide where fetched bytes go; a named `--allow-network-expert-fetch`
flag; fault tests for hash mismatch, timeout, insufficient disk and network absence; a report that
separates network, disk and device traffic.

**Gate.** A single-token, three-layer chain executes with at least one expert bundle fetched on
demand and hashed, with the same logits as the local-artifact run; then a bounded full-depth
experiment. It does not claim a full-model result.

### F5 — Full-depth single-token feasibility

**Why.** The byte arithmetic says one no-cache token needs 8,951,775,580 B; the design keeps the
trunk local (6,390,088,008 B) plus globals (596,000,788 B) and fetches the expert union on demand
(1,965,686,784 B worst case). Whether that runs within the Governor budget and how long it takes
are open questions.

**Dependency.** F2, F3 (compressor layouts beyond 4:1), F4, and a way to oracle the result. The
honest boundary: without the complete checkpoint and a local llama.cpp oracle, full-depth parity
against the model cannot be claimed; only stage-wise oracle checks and self-consistency are
available. Historical full-depth records in this repository are dated evidence, not revalidation.

**Gate.** A full 43-layer prefill completes deterministically under the live budget with exact
byte telemetry, and every layer's route is authoritative. Label it partial-verification; no
model-quality or speed claim.

### F6 — Qwen3.8-Flash-Next revalidation (comparison track)

**Why.** Its llama.cpp parity record is the project's only externally validated end-to-end path;
it can be re-opened for comparison work if the exact checkpoint is available again.

**Entry criteria (from `ROADMAP.md`).** The exact Q8_0 GGUF revision is legally available and
re-imported through the canonical immutable store with every SHA-256 verified; the pinned
llama.cpp oracle (`6fe7498`, or an explicitly re-pinned revision) is built locally; the margin-aware
parity battery exists; the Governor preflight passes on the live host. Then the blocked backlog
(Q11k chunked prefill, Q11l SIMD default, Q11m Q4_K expert re-import, Q11n routing-aware VRAM tier,
Q12 prefix-KV, Q13 serve) becomes actionable in the documented order.

**Gate.** The existing Q11C parity standard: greedy continuation token-for-token where margins are
large, with near-tie divergence measured rather than hidden.

### F7 — Optional research directions, not commitments

- Compare against **ik_llama.cpp** and **KTransformers** on the same expert-placement problem
  (`README` prior-art section explains why they are the honest baselines).
- Revisit **chunked prefill** for the DeepSeek path if F5 makes long contexts relevant.
- Revisit **V4.1-Flash** only as metadata/catalog study: its ~511 GB checkpoint, SM100/Ascend
  kernels and 64+64-GPU deployment are out of reach for the reference PC (see the DeepInfer
  review). Do not download it.
- **DSpark / speculative decoding** stays out of scope until an execution loop exists; the
  historical feasibility work is archived.

### F8 — Adopt the SSD-streaming peer-review findings

A separate review of four public engines (Kimi K3 SSD streaming, DwarfStar/ds4, Colibri, WARP)
identified four P0 improvements that fit Vastloom's existing architecture; the full analysis,
evidence and "what not to copy" are in
[`docs/research/SSD_STREAMING_PEER_REVIEW.md`](../research/SSD_STREAMING_PEER_REVIEW.md).

1. **Coalesced, 4 KiB-aligned expert-bundle container**: one `pread` per expert (current artifacts
   use three separate range files per expert), with the pinned GGUF range identity and SHA-256
   manifest retained as provenance.
2. **Adaptive I/O policy**: keep the measured buffered default for recurring small ranges and use
   the existing bypass pool (`src/io/bounded_io_pool.cpp`) for cold/large or coalesced records;
   publish a cold/warm A/B. The peer evidence includes Windows section-handle serialization and
   O_DIRECT wins on some drives.
3. **Learned expert placement**: persist per-expert routing heat across runs, auto-pin the hottest
   experts in spare RAM/VRAM and repin at bounded turn boundaries with hysteresis; placement must
   never change arithmetic.
4. **Router-lookahead prefetch for learned layers**: predict layer L+1's experts from layer L's
   post-attention state (Colibri reports 71.6% recall versus 41.3% for last-token repetition),
   issue bounded readahead on a dedicated I/O thread, and keep it opt-in because hint-only
   prefetch can be negative on a saturated disk.

**Status (2026-10-23):** items 1 and 2 are implemented and validated as an opt-in experiment in
[`docs/reports/P0_EXPERT_PACK_IO_VALIDATION.md`](../reports/P0_EXPERT_PACK_IO_VALIDATION.md):
one aligned read per expert, byte-identical outputs, read calls 54 → 18 (token 0), 18 → 6
(layer 0) and 264 → 88 (five-token prompt), with buffered/bypass modes and actual read-path
counters. The direct-I/O proxy showed no wall-time win and bypass was slightly slower; a cold-cache
reboot comparison remains future work (helper: `scripts/purge-standby.ps1`, elevated only). Item 3
has a recording-only prototype (`--usage-profile`, no pinning yet); item 4 is still future work.
The P1 cache-plan/paging guard is implemented in the chain probe (`cache_plan`, `--cache-entries`);
the remaining P1 items are still future work.

Each item needs the usual gate: bit-identical or oracle-verified outputs, exact read/byte counters,
a dated report, and a measured win (or an explicit negative result) before it becomes a default.
These are research directions, not active milestones.

## 4. What deliberately stays out of scope

- Full 90.86 GB checkpoint download; IQ1/`UD-IQ1_S` execution; CUDA dense/attention kernels;
  speculative decoding; any new Qwen feature work.
- Treating a 1–3-layer or partial-depth result as "the model works".
- Throughput comparisons without recorded hardware, cache state and byte accounting.
- Using a predicted learned-router route as authority (prediction is allowed only for bounded
  prefetch, never to choose which experts execute).

## 5. How to restart from a clean machine

Artifacts live under the git-ignored `build/` tree and are **not** in the repository. A machine
with the repo can reproduce the measured slice from the pinned public sources:

```powershell
# 1. store + metadata-only IQ2_XXS import (about 9.5 MB)
vastloom store-init <store> --reserve 20GiB
python tools/import_hf_model.py --model deepseek-v4-flash-0731-gguf-iq2xxs --store <store> --metadata-only --reserve-gib 40

# 2. plans and range imports (trunk, globals, authoritative experts)
vastloom deepseek-gguf-probe-plan <store-metadata> --layer 0 --experts 0,1,2,3,4,5 --json > layer0.json
python tools/import_hf_ranges.py --model deepseek-v4-flash-0731-gguf-iq2xxs --plan layer0.json --store <store> --scope trunk --allow-range-download
python tools/import_hf_ranges.py --model deepseek-v4-flash-0731-gguf-iq2xxs --plan globals.json --store <store> --scope globals --allow-range-download

# 3. real layer / chain probes
vastloom deepseek-layer-probe <store-metadata> <trunk> <experts> <globals> --token 0 --json
vastloom deepseek-chain-probe <store-metadata> <globals> --layers 3 --tokens 0,1,2,3,4 --trunk 0=<t0> --trunk 1=<t1> --trunk 2=<t2> --experts 0=<e0> ... --json

# 4. independent oracle (NumPy interpreter) and opt-in tests
python tests/test_deepseek_layer_oracle.py --prefill --vastloom build/release/vastloom --model <store-metadata> ...
python tests/test_deepseek_chain_contract.py --vastloom build/release/vastloom --model <store-metadata> ...
```

CMake opt-in variables used by the tests are documented in `README.md` and `CMakeLists.txt`:
`VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT`, `..._TRUNK_ARTIFACT`, `..._EXPERT_ARTIFACT`,
`..._GLOBALS_ARTIFACT`, `..._ORACLE_PYTHON`, `..._CHAIN_*`.

The full measured claims and their tolerances are in the reports; do not copy a number from this
file into a new claim without re-running the corresponding test.

## 6. Decision log pointers

- D4 §11 (closed): first-touch SHA-256, cache sizes, no payload-tier dependency, no fetch in the
  concluded project.
- `AGENTS.md` execution checkpoint: what is measured, what is not, and how to treat a request to
  re-run a large-model test.
- `ROADMAP.md` closing note: the same remaining items, stated as not planned.
