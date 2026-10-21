# D3b — independent chain oracle parity

Validation date: 2026-10-21. Track: DeepSeek V4
Flash 0731, `UD-IQ2_XXS`, layers 0–2 of 43, single-token prefill. This report closes the oracle
half of D3b: the independent NumPy implementation now executes the three-layer hash-routed chain
and reproduces the native `deepseek-chain-probe` dumps for two tokens and three prefix depths.

## Scope

The native executor runs 2–3 consecutive hash-routed layers, each from its own imported trunk and
selected-expert artifacts, and evaluates the shared Hyper-Connection head plus LM head at the final
layer. The independent oracle runs the same chain from the same bytes and writes one logits/hidden
pair per requested depth (1, 2, 3). The comparison gate is unchanged from D2b.2: 129,280 logits and
4,096 final-hidden values per depth/token, logits maximum absolute difference ≤ `1e-3`, hidden
≤ `1e-3` with `rtol=1e-5`/`atol=1e-6`, argmax equal, and the per-layer route equal.

This is three layers of 43 and never the model's output.

## Method

### Inputs and independence

The oracle reads the pinned metadata store (GGUF v3 headers for the three shards), the globals
artifact, one trunk artifact per layer and one selected-expert artifact per layer and token. It
parses the headers and tensor tables itself, maps every artifact range to a tensor by source-object
name, exact source offset and byte count (packed expert ranges are aligned slices of the 3-D
container tensors), and checks each manifest's scope, layer, model identity, revision, range count
and local file sizes. It never imports or executes C++ code and has no network path. Each
`selected-experts` artifact must carry `routing_authority_verified`; the proxy route is not used.

Two additions to the D2b.2 checks matter here:

- `load_artifact` now takes the expected layer and rejects a trunk or expert artifact whose
  manifest layer is not the chain layer it is bound to.
- `build_layer` consumes **every** trunk range of the layer and fails if one is left unmodelled.
  Layer 0 and 1 consume 21 objects each; layer 2 consumes 31. A silently ignored compressor tensor
  would therefore fail instead of producing plausible logits.

The decoders are unchanged from D2b.2 (Q8_0, Q5_K, Q6_K, Q4_K, BF16, F32, I32, IQ2_XXS, IQ3_XXS)
and the pinned ggml codebook fingerprints are still asserted at startup.

### Chain composition

The C++ chain seeds the hidden state by replicating the embedding row `hyper_multiplier` times and
then, for each layer, passes `ffn_post` (the post-FFN Hyper-Connection output, 4 × 4,096 values) to
the next block. The final head reduces that state to 4,096 values before the final RMS
normalization and the LM head. The oracle reproduces exactly that composition: `execute_layer`
accepts the previous layer's hidden state instead of re-reading the embedding, returns its own
`ffn_post`, and `final_head` is evaluated at each requested prefix depth from the same state. No
per-layer head is applied inside the chain.

Layer 1 is the same graph as layer 0 on its own trunk, clamp schedule entry and `blk.1` hash table;
the layer index parameterizes the tensor names, the routing table and the SwiGLU clamp lookup.
There is no layer-0 special case left in the equations.

The layer-0 CLI is preserved: one plain `--trunk`/`--experts` pair, `logits.f32`, 66 stage
checksums, the same summary format. Re-running it after this change reproduces the previous oracle
run bit-for-bit (identical 66 stage checksums and logits SHA-256) and the same deltas against the
recorded probe dumps (logits `9.5367431640625e-07`, hidden `4.76837158203125e-07`).

### Layer 2: compression ratio 4

The checkpoint's schedule is `[0, 0, 4, 128, 4, 128, ...]`, so layer 2 is the first compressed
layer and uses the overlapping 4:1 compressor **and** the learned indexer
(`deepseek_v4_graph.cpp`: `compress_prefill`, `indexer_indices`, `attention_prefill`). Three
consequences are implemented from the C++ source:

- **RoPE schedule.** A layer with ratio ≠ 0 switches both Q/K RoPE and the inverse output RoPE to
  `deepseek4.attention.compress_rope_freq_base` (160,000) with the YARN original-context correction
  at 65,536 positions. Layers 0 and 1 use θ = 10,000 with the correction disabled. This is the
  only graph difference visible at one token and it is what makes the layer-2 numbers distinct.
- **Compressor and indexer projections.** The attention compressor (`wkv`, `wgate`, `ape`, `norm`)
  and, for ratio 4, the indexer (`wq_b`, `weights_proj`, its compressor and `norm`) are bound,
  decoded and evaluated. The attention compressor projects the normalized attention input; the
  indexer projects the normalized query rank, applies its own compression-schedule RoPE, scores
  each compressed slot per head with the positive-part projection and the head-scale factor, and
  ranks by score then slot. The oracle stages the projections, scores, indexer query/weights and
  compressed rows.
- **Attention indices.** Window slots come first (one prefill token → slot 0), followed by the
  selected compressed slots (`token_count + slot`).

### Comparison harness

`tests/test_deepseek_layer_oracle.py` keeps the always-on synthetic section and the layer-0
integration, and adds an opt-in chain section. For tokens `0,1` and depths `1,2,3` it runs:

- `deepseek-layer-probe` for depth 1 (token-specific layer-0 artifacts);
- `deepseek-chain-probe --layers 2` and `--layers 3` with per-layer trunk/expert artifacts;
- the oracle once per token with `--layers 3 --depths 1,2,3`.

Dumps are staged in a temporary directory under `build/` and removed unless `--keep` is given. The
probe's JSON `route` is the resolved set in ascending order; the trace's `experts` list is the
execution order. The test pins the sorted set per layer and compares the execution order with the
oracle's table order, so a reordering in either field is reported instead of silently accepted.
Tolerances are constants in the test and were not relaxed.

## Layer-2 compressor implementation boundary

The compressor equations are implemented in full, not stubbed: `compressor_windows` reproduces the
overlap rule (the previous window is scored together with the current one, and skipped for the
first window), the double-precision `scores + APE` sum over float32 operands, the double-precision
softmax and the float32 rounded average; `compressor_prefill` then applies RMS normalization and
the compression-schedule RoPE. The synthetic section pins the window output against an independent
loop reference and the full normalized/rotated rows against fixed values (12 tokens, ratio 4,
output width 2), and checks that fewer tokens than the ratio produce no row.

What the real chain exercises is narrower, and the difference is stated rather than hidden:

- A single prefill token is **fewer tokens than the ratio** (1 < 4), so `cutoff = 0`,
  `compressed_count = 0` and both the attention compressor and the indexer compressor emit zero
  rows. Layer 2 therefore attends over its window slot alone, exactly as the C++ path does. The
  oracle stages `attention_compressed` and `indexer_compressed` with count 0, and the test requires
  that count.
- The compressed-slot attention output (softmax over `window + compressed` slots) and an indexer
  selection with `top_k > 0` are implemented but **not exercised numerically** by these artifacts,
  because no compressed slot exists at one token. They are covered only by the synthetic checks
  above. A multi-token prefill comparison would be the first test of that path.
- Compression ratio 128 (layers 3, 5, …) is refused with a precise error: its single-slot
  compressor layout differs and is not modelled here. Only ratios 0 and 4 are admitted.
- Only the hash-routed prefix is admitted (`layers ≤ deepseek4.hash_layer_count = 3`). A learned
  router needs the post-attention state and complete expert coverage, which is D4 scope.

## Results

Both tokens pass at every requested depth. Deltas are maximum absolute differences against the
probe dumps; no tolerance was relaxed.

| Token | Depth | Logits max abs diff | Final hidden max abs diff | Argmax | Compression ratio of the new layer |
|---|---|---|---|---|---|
| 0 | 1 | 9.5367431640625e-07 | 4.76837158203125e-07 | 30,528 | 0 |
| 0 | 2 | 9.5367431640625e-07 | 1.9073486328125e-06 | 86,051 | 0 |
| 0 | 3 | 9.5367431640625e-07 | 3.814697265625e-06 | 86,051 | 4 |
| 1 | 1 | 1.9073486328125e-06 | 2.384185791015625e-07 | 69,146 | 0 |
| 1 | 2 | 1.9073486328125e-06 | 2.384185791015625e-07 | 69,146 | 0 |
| 1 | 3 | 1.9073486328125e-06 | 2.384185791015625e-07 | 69,146 | 4 |

Logit ranges (oracle and probe agree to the deltas above):

| Token | Depth | Minimum | Maximum |
|---|---|---|---|
| 0 | 1 | −16.127300262451172 | 15.52056884765625 |
| 0 | 2 | −16.227359771728516 | 15.763052940368652 |
| 0 | 3 | −16.003379821777344 | 16.162439346313477 |
| 1 | 1 | −27.019575119018555 | 20.86294937133789 |
| 1 | 2 | −27.986600875854492 | 20.946231842041016 |
| 1 | 3 | −28.131023406982422 | 20.758203506469727 |

Routes, execution (table) order; the probe's sorted set is identical:

| Token | Layer 0 | Layer 1 | Layer 2 |
|---|---|---|---|
| 0 | 254, 222, 245, 200, 53, 35 | 163, 137, 158, 97, 184, 8 | 108, 115, 43, 53, 132, 143 |
| 1 | 239, 202, 122, 23, 115, 57 | 63, 126, 219, 21, 170, 150 | 217, 221, 240, 26, 247, 39 |

Sorted sets for reference: token 0 `35/53/200/222/245/254`, `8/97/137/158/163/184`,
`43/53/108/115/132/143`; token 1 `23/57/115/122/202/239`, `21/63/126/150/170/219`,
`26/39/217/221/240/247`.

Test output (abridged):

```
synthetic block decoders PASS (104 pinned checks)
token 0 depth 1 PASS (logits max|delta|=9.53674316e-07, hidden max|delta|=4.76837158e-07, argmax=30528, ...)
token 0 depth 2 PASS (... hidden max|delta|=1.90734863e-06, argmax=86051, ...)
token 0 depth 3 PASS (... hidden max|delta|=3.81469727e-06, argmax=86051, ...)
token 1 depth 1 PASS (logits max|delta|=1.90734863e-06, ...)
token 1 depth 2 PASS (logits max|delta|=1.90734863e-06, ...)
token 1 depth 3 PASS (logits max|delta|=1.90734863e-06, ...)
PASS (6 depth comparisons, max logits |delta|=1.90734863e-06, max hidden |delta|=3.81469727e-06)
```

No divergence was observed. The harness reports the first divergent depth/layer with the offending
index and both values, a route mismatch as a layer-level divergence, and the first non-finite
oracle stage when one exists; none of those paths fired. The growth of the hidden delta with depth
(4.8e-07 → 3.8e-06 for token 0) is the expected float32 accumulation drift, more than two orders of
magnitude inside the gate.

### Stage coverage

The oracle records 205 stage checksums per token run: every layer-0/1/2 intermediate (the D2b.2
set, prefixed `layerL.`), each layer's post-FFN hidden state, and the per-depth head stages
(`depthD.final_hidden`, `depthD.final_norm`, `depthD.logits`). The test requires 33 of those files
per token (counts, SHA-256, FNV-1a and finiteness) and recomputes both checksum maps. The chain
addition adds layer-2 compressor projections/scores, the indexer query/weights, and both compressed
row buffers (required to be empty at one token).

As in D2b.2, the probe exposes only logits and final hidden (plus routes) — so the remaining stage
checksums are recorded but cannot be compared against C++. This is a trace-coverage limitation,
not a tolerance exception. Closing it needs stage dumps from the probe.

### Determinism

- The oracle was run twice for token 0, depth 3: all 205 stage checksums and every depth
  logits/hidden SHA-256 were identical. There is no RNG, no threading and no time-dependent input.
- `deepseek-chain-probe --layers 3 --token 0` was re-run during this validation and its dumps were
  bit-identical (`numpy.array_equal`) to the Lead's earlier `chain3-logits-cpu.f32` /
  `chain3-hidden-cpu.f32`, max difference 0.0.
- Probe and oracle are not bit-identical to each other: raw FNV hashes differ on layer 0 (as
  documented in D2b.2) because the probe's MSVC scalar path and NumPy's float64 accumulation round
  a small fraction of values to the neighbouring float32. Value comparisons are the gate; the FNV
  equality assertion stays opt-in (`--require-checksum-equality`).

## Implementation files

- `tools/reference_deepseek_v4_layer_oracle.py` (117,707 bytes): chain mode
  `--layers N --trunk LAYER=PATH --experts LAYER=PATH --depths D1,D2,...`, layer-indexed layer
  builder, compressor/indexer implementation, per-depth head, 205-stage summary, and the
  `logits-depthD.f32`/`hidden-depthD.f32` dumps plus `summary.json`. The legacy single-layer CLI and
  summary format are unchanged. Peak working set 340.8 MiB and wall time about 30 s for one token at
  depth 3 with stage files on the reference PC.
- `tests/test_deepseek_layer_oracle.py` (65,632 bytes): 104 synthetic pinned checks (the 87 decoder
  checks plus 17 compressor/indexer checks), the unchanged layer-0 integration, and the opt-in
  chain integration with per-token expert maps, route pinning and first-divergence reporting.
- `docs/reports/D3B_CHAIN_ORACLE_VALIDATION.md`: this report.
- CMake: `vastloom_deepseek_v4_chain_oracle` is registered when
  `VASTLOOM_DEEPSEEK_V4_ORACLE_PYTHON`, the three trunk artifacts, the globals artifact and both
  tokens' expert artifacts are supplied; the CTest run passed in 780.83 s. The companion
  `vastloom_deepseek_v4_chain_probe` contract (pinned results, determinism, malformed refusal)
  passed in 587.64 s.

## Reproduction

The PATH Python has no NumPy; use the bundled interpreter (Python 3.12, NumPy 2.3.5).

```powershell
$python = "<NumPy-capable Python>"
$opt = "build\analysis\deepseek-0731\store\models\optimized"
$md  = "build\analysis\deepseek-0731\store\models\original\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata"

& $python tests\test_deepseek_layer_oracle.py `
  --vastloom build\release\vastloom.exe --model $md `
  --artifact "$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-trunk-e62746099f66" `
  --experts "$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-experts-c9b27ecd7c32" `
  --globals "$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-globals-067131537b81" `
  --chain-trunk "0=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-trunk-e62746099f66" `
  --chain-trunk "1=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l1-trunk-6995ba972b9b" `
  --chain-trunk "2=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l2-trunk-a4de2e64fef7" `
  --chain-experts "0:0=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-experts-c9b27ecd7c32" `
  --chain-experts "0:1=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l1-experts-9fc8722240e4" `
  --chain-experts "0:2=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l2-experts-7bbd313e360a" `
  --chain-experts "1:0=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-experts-25a997485fa5" `
  --chain-experts "1:1=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l1-experts-98dc681775ab" `
  --chain-experts "1:2=$opt\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l2-experts-02c6893f873e"
```

The synthetic section also runs without artifacts: `& $python tests\test_deepseek_layer_oracle.py`.
The oracle alone:

```powershell
& $python tools\reference_deepseek_v4_layer_oracle.py --model $md `
  --trunk "0=...l0-trunk..." --trunk "1=...l1-trunk..." --trunk "2=...l2-trunk..." `
  --experts "0=...l0-experts..." --experts "1=...l1-experts..." --experts "2=...l2-experts..." `
  --globals "...globals..." --layers 3 --depths 1,2,3 --token 0 `
  --output-dir build\analysis\deepseek-0731\oracle-work\chain-t0 --stages
```

Artifacts and dumps are local validation data under `build/`; they are not committed.

## Gate status

- **D3b oracle half: met** for layers 0–2, tokens 0 and 1, depths 1–3. Logits and final hidden are
  reproduced well inside the unchanged tolerances; argmax and every per-layer route agree; the
  layer-2 compressed layer is included.
- **D3 gate: met for the measured scope.** Together with
  `docs/reports/D3_HASH_CHAIN_VALIDATION.md` (executor, 2/3-layer deterministic runs, GPU chain
  bit-identical) and the two opt-in CTest contracts, D3's executor-and-oracle requirement is
  satisfied for single-token prefill. The documented boundary remains: one-token prefill closes no
  compression window, so compressed-slot attention and a non-empty indexer selection are
  synthetically checked but not compared against C++; multi-token prefill is the next test of that
  path. D4 (design) and the VRAM payload tier remain open.
- No blocker was hit: layer 2's compressor/indexer was reproducible within the stated boundary, so
  the task's fallback (deliver depth 2 only and leave D3 open) was not needed.

## Not established

- Three layers of 43 and one prefill token: this is not the model's output, not a full-depth run,
  and not a generation result.
- The compressed-slot attention output and an indexer selection with `top_k > 0` are implemented
  and synthetically checked but not compared against C++, because a one-token prefill closes no
  compression window. The layer-2 claim is therefore about the compressed layer's single-token
  path (compression-schedule RoPE, tensor binding, projections, empty compressed caches), not about
  multi-token compressed attention.
- No ratio-128 layer, no learned-router layer, no decode/incremental path, no KV reuse.
- No VRAM/GPU chain, no residency behaviour and no throughput claim.
- Not bit-identical: raw FNV hashes differ for layer 0; the value gate is the claim.
- The oracle trusts the imported artifact payloads; it does not re-verify their SHA-256, which the
  probe does on load.
- The chain test is opt-in and registered as `vastloom_deepseek_v4_chain_oracle` (and the
  executor-only contract as `vastloom_deepseek_v4_chain_probe`); both require the artifact paths
  through CMake cache variables and are not part of the always-on suite.
