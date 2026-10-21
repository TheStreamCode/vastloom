# D3 — hash-routed multi-layer chain from range artifacts

Validation date: 2026-10-21. Track: DeepSeek V4 Flash 0731, `UD-IQ2_XXS`, hash-routed prefix
(layers 0, 1, 2). This report records the **executor and its measured runs**. The D3 gate also
requires an independent logits comparison; the oracle extension beyond layer 0 is not yet
available, so D3 is **not closed** by this report.

## Scope

Build and run a bounded multi-layer executor over imported range artifacts:

- one trunk artifact and one authoritative expert artifact per layer;
- routes resolved from each layer's own imported `tid2eid` table, not supplied by the caller;
- layers loaded from their artifacts on demand through a bounded window;
- per-layer expert caches, optional device-expert residency, per-layer trace and byte telemetry;
- only the three hash-routed layers are admitted. A learned-router layer (3+) is refused, because
  its expert selection cannot be planned before the preceding layers execute (D4 scope).

## Implementation

- New CLI `deepseek-chain-probe MODEL_DIRECTORY GLOBAL_ARTIFACT --layers N --token ID
  --trunk L=PATH --experts L=PATH [--cuda-experts] [--gpu-limit MIB] [--gpu-reserve MIB]
  [--dump-logits PATH] [--dump-hidden PATH] [--dump-trace PATH] [--json]`.
- The command validates both positional directories, the closed option surface, the pinned
  metadata store (`pinned_spec_for`), `layers ∈ {2,3}`, the globals artifact, and each layer's
  trunk/experts artifacts (scope, layer, hash verification; experts must be
  `routing_authority_verified`).
- It relocates the catalog onto all artifacts, binds one placeholder plan per layer to identify
  the trunk artifact, resolves the token's hash route from that binding, and then builds the
  multi-layer reference graph: per-layer trunk mappings, small tensors and routing tables
  resident, one expert cache per layer (`layers.L.ffn`), and a lazy window that loads a layer's
  whole trunk artifact the first time one of its matrices is requested.
- Counters: `trunk_loads`, `trunk_bytes_read`, expert `bytes_read`/`misses`/`hits`/`evictions`,
  relocated tensors/bundles, optional accelerator counters, per-layer route and executed experts,
  elapsed time. Raw logits/hidden dumps and a JSON trace are available for the oracle.

## Artifacts acquired for this gate

| Object | Bytes | Ranges/objects |
|---|---:|---|
| Layer 0 trunk (reused from D2a) | 138,867,160 | 21 |
| Layer 1 trunk | 138,867,160 | 21 |
| Layer 2 trunk (compressed layout) | 159,992,792 | 31 |
| Experts, token 0, each layer (6 × 7,536,640) | 45,219,840 | 18 |
| Experts, token 1, each layer | 45,219,840 | 18 |
| Globals (reused from D1) | 596,000,788 | 6 |

Additional download for D3: 138,867,160 + 159,992,792 + 3 × 45,219,840 = **434,539,472 B** for
token 0, plus **135,659,520 B** for the token-1 expert sets. Total local payload after D3
(metadata store, D1/D2a/D3 artifacts) is about 1.3 GiB, well inside the stated budget.

## Results

Reference PC, CPU path, one token per run, cold artifact cache per process.

| Run | Layers | Argmax | Logits range | Trunk loads / bytes | Expert bytes / misses | Elapsed |
|---|---:|---:|---|---:|---:|---:|
| Token 0 | 2 | 86,051 | [−16.2274, 15.7631] | 2 / 277,734,320 | 90,439,680 / 12 | 1,404.34 ms |
| Token 0 (repeat) | 2 | 86,051 | identical | 2 / 277,734,320 | 90,439,680 / 12 | 1,927.61 ms |
| Token 0 | 3 | 86,051 | [−16.0034, 16.1624] | 3 / 437,727,112 | 135,659,520 / 18 | 2,761.23 ms |
| Token 0 (repeat) | 3 | 86,051 | identical | 3 / 437,727,112 | 135,659,520 / 18 | 2,203.04 ms |
| Token 1 | 3 | 69,146 | [−28.131, 20.7582] | 3 / 437,727,112 | 135,659,520 / 18 | 2,103.71 ms |
| Prompt `[0,1,2,3,4]` | 3 | 20,267 | [−38.0188, 14.7002] | 3 / 437,727,112 | 663,224,320 / 88 | 5,577.33 ms |
| Token 0, GPU expert tier | 3 | 86,051 | [−16.0034, 16.1624] (bit-identical to CPU) | 3 / 437,727,112 | 135,659,520 / 18 | 1,787.26 ms |

The five-token prompt opens the layer-2 compression window. Its 15 authoritative expert artifacts
(5 tokens × 3 layers) merge into 88 unique bundles / 343 relocated tensors; the union is what the
caches hold and what the 663,224,320 expert bytes account for. The run is byte-identical across
two repetitions; per-token routes for all three layers are in the JSON `routes_per_token` field
and in `chain5-reference.json`.

GPU leg (`--cuda-experts --gpu-limit 64`): offered 116, accelerated 54 (3 layers × 6 experts ×
3 matrices), declined 62 (recurring trunk matrices), evicted 28, resident 26 matrices /
66,350,080 B, uploaded 135,659,520 B, kernel 150.458 ms, upload 20.3482 ms. The 129,280 logits are
byte-identical to the CPU chain (`max abs diff 0.0`). This is device residency of expert matrices
under a cap, not a VRAM payload tier.

Per-layer authoritative routes:

| Layer | Token 0 | Token 1 |
|---|---|---|
| 0 | 35, 53, 200, 222, 245, 254 | 23, 57, 115, 122, 202, 239 |
| 1 | 8, 97, 137, 158, 163, 184 | 21, 63, 126, 150, 170, 219 |
| 2 | 43, 53, 108, 115, 132, 143 | 26, 39, 217, 221, 240, 247 |

Byte identity checks: `trunk_bytes_read` is exactly 2 × 138,867,160 for two layers and
2 × 138,867,160 + 159,992,792 for three; `expert_bytes_read` is exactly 6 × 7,536,640 per layer.
`relocated_tensors`/`relocated_expert_bundles` are 84/12 for two layers and 133/18 for three.
Repeated runs produced byte-identical 129,280-float logits (`max abs diff 0.0`).

## Independent oracle parity (D3b)

`tools/reference_deepseek_v4_layer_oracle.py` now runs the same 3-layer chain with an independent
NumPy implementation (layer-indexed graph, layer-2 4:1 compressor/indexer, per-depth head).
`docs/reports/D3B_CHAIN_ORACLE_VALIDATION.md` records the comparison against these dumps:

| Token | Depths | Logits max abs diff | Final hidden max abs diff | Argmax |
|---|---|---|---|---|
| 0 | 1, 2, 3 | 9.54e-07 | 4.77e-07 → 3.81e-06 | 30,528 / 86,051 / 86,051 |
| 1 | 1, 2, 3 | 1.91e-06 | 2.38e-07 | 69,146 at every depth |

Routes agree per layer; tolerances unchanged (logits 1e-3, hidden rtol 1e-5/atol 1e-6). Boundary:
one prefill token is below the layer-2 compression ratio, so the compressed caches are empty and
compressed-slot attention is synthetically, not natively, compared; multi-token prefill is the
next test of that path.

### Multi-token prefill (D3c)

A five-token prompt `[0,1,2,3,4]` closes one 4:1 compression window and exercises the compressed
slot for real. The independent oracle reproduces the native run: logits max abs diff
**1.9073486328125e-06**, final hidden **5.960464477539063e-08**, argmax 20,267; all fifteen
per-layer per-token routes agree; the oracle reports one compressed row, six compressed slots per
position, and non-empty compressed stages (512/128 values), so the window is not a synthetic
claim. See `docs/reports/D3C_MULTITOKEN_ORACLE_VALIDATION.md`. The multi-token leg is also pinned
in the chain contract test (pinned 5-token result, deterministic repeat, 15 expert artifacts).

## Gate status

- **D3: met for the measured scope.** Deterministic 2/3-layer execution from artifacts, the
  independent oracle at depths 1–3 for tokens 0 and 1, a five-token prefill with a non-empty
  compressed window matched at 1.9e-06 on the logits, the GPU chain logits bit-identical to CPU,
  and the opt-in CTest contracts (`vastloom_deepseek_v4_chain_probe`, 587.64 s;
  `vastloom_deepseek_v4_chain_oracle`, 780.83 s).
- **Documented boundary:** ratio-128 layers, learned-router layers and full depth are not covered
  (they need the complete checkpoint or the D4 network-fetch subsystem); no throughput or
  full-model claim.

## Gates

| Gate | Result |
|---|---|
| Release and Debug builds (MSVC, warnings as errors) | PASS |
| `ctest --test-dir build\debug` | 15 tests: 14 passed, 0 failed, 1 skipped (`vastloom_ggml_compute`) |
| Python `compileall` | PASS |
| Multi-layer execution from artifacts | PASS, 2 and 3 layers, deterministic |
| Opt-in chain contract (`vastloom_deepseek_v4_chain_probe`) | PASS (587.64 s): pinned 3/2-layer results, deterministic repeat, malformed chains refused, optional GPU leg bit-identical |
| Opt-in layer-0 oracle (`vastloom_deepseek_v4_layer_oracle`) | PASS (87.14 s) |

## Not established

- **Multi-token prefill and the non-empty compression window.** One prefill token is below the
  layer-2 compression ratio, so compressed-slot attention and a non-empty indexer selection are
  implemented and synthetically checked but not compared against C++. The single-token chain and
  every layer's route are verified against the independent oracle (§ above).
- No GPU-chain benchmark claim: the GPU leg is faster or slower depending on the cap and is not
  a product path; only its logits were checked (bit-identical to CPU).
- Three of 43 layers; not the model's output; no throughput claim; learned-router layers and
  full-depth execution are refused by design.

## Reproduction

```powershell
$V = "build\release\vastloom.exe"
$M = "<store>\models\original\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata"
$G = "<globals artifact>"
& $V deepseek-chain-probe $M $G --layers 3 --token 0 `
  --trunk "0=<l0-trunk>" --trunk "1=<l1-trunk>" --trunk "2=<l2-trunk>" `
  --experts "0=<l0-experts-t0>" --experts "1=<l1-experts-t0>" --experts "2=<l2-experts-t0>" `
  --json --dump-logits chain3-logits-t0.f32 --dump-hidden chain3-hidden-t0.f32
```

Artifacts and dumps are local validation data under `build/`; they are not committed.
