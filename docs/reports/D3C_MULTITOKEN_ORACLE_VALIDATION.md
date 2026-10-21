# D3c — multi-token prefill oracle parity

Validation date: 2026-10-21. Track: DeepSeek V4 Flash 0731, `UD-IQ2_XXS`, layers 0–2 of 43,
five-token causal prompt `[0, 1, 2, 3, 4]`. This report closes the oracle half of D3c: the
independent NumPy implementation now executes a multi-token prefill, opens the layer-2 4:1
compression window, selects compressed slots through the learned indexer, and reproduces the
native `deepseek-chain-probe --tokens` dumps.

## Scope

The native executor runs 2–3 consecutive hash-routed layers, each from its own imported trunk
and selected-expert artifacts, and evaluates the shared Hyper-Connection head plus LM head on
the last prompt position. With `--tokens` it admits one to sixteen prompt tokens; the selected
experts of every position come from its own `tid2eid` row and the union of those routes is
resolved once per layer.

The independent oracle runs the same graph from the same bytes and writes the last-position
logits plus the pre-normalization hidden state. The comparison gate is unchanged from D3b:
129,280 logits and 4,096 final-hidden values, logits maximum absolute difference ≤ `1e-3`,
hidden ≤ `1e-3` with `rtol=1e-5`/`atol=1e-6`, argmax equal, and every per-layer per-token route
equal. No tolerance was relaxed.

This is three layers of 43 over one prompt; it is not the model's output.

## Method

### Inputs and independence

The oracle reads the pinned metadata store (GGUF v3 headers for the three shards), the globals
artifact, one trunk artifact per layer and five selected-expert artifacts per layer, one per
prompt-token route. It parses the headers and tensor tables itself, maps every artifact range to
a tensor by source-object name, exact source offset and byte count, and checks each manifest's
scope, layer, model identity, revision, range count and local file sizes. It never imports or
executes C++ code and has no network path. Each `selected-experts` artifact must carry
`routing_authority_verified`.

A layer may now carry several expert artifacts, so `merge_expert_artifacts` merges them and
admits an expert selected by two prompt positions only when both copies name the same immutable
source spans `(source_object, source_offset, source_bytes)`. A disagreement fails instead of
silently taking the first copy. The union is then required to equal the routing union of the
prompt exactly, the same strictness the single-token path applies to one route.

### Multi-token equations

The equations were derived from `src/inference/deepseek_v4_graph.cpp` and implemented
independently:

- **Embedding and Hyper-Connection.** `embedding_rows` reads one embedding row per prompt token;
  each row is replicated `hyper_multiplier` times to form the flat hidden state, exactly as
  `deepseek_v4_reference_prefill` does. `hc_pre` and `hc_post` now run per token: RMS
  normalization over the 16,384-value flattened row, the packed mixing projection, the Sinkhorn
  split, the reduced 4,096-value row, and the residual combination, with each row reduced
  independently in float64.
- **Attention window.** `window_indices` becomes a per-token table: for position `t` and
  `selected = min(tokens, 128)`, the valid entries are `max(0, t - 127) … t` and every entry
  after `t` is `-1`, so positions never see the future. The oracle reproduces that fill exactly.
- **4:1 compressor.** `compress_prefill` with `ratio = 4` uses `coefficient = 2`, so `wkv` and
  `wgate` project to `2 × 512` columns: the first half re-scores the previous window and the
  second half the current one. `cutoff = tokens - tokens % ratio`, `count = cutoff / ratio`;
  the window softmax is computed in double over `scores + APE` with the APE row indexed by the
  position inside the window, the average is rounded once to float32, then RMS-normalized and
  rotated by the compression-schedule RoPE with position stride `ratio`. Five prompt tokens
  therefore close exactly one window; positions 0–3 feed it and position 4 is the tail that
  never enters a window.
- **Indexer.** `indexer_indices` projects the normalized query rank through `wq_b`, rotates it at
  the token positions with the compression-schedule RoPE, and scores every compressed slot per
  head with `max(dot, 0) * head_weight * 1/sqrt(head_width)/sqrt(head_count)`. A position `t`
  may only score slots with `slot < (t + 1) / ratio`; the ranking sorts by score descending and
  slot ascending, selects `min(index_top_k, compressed_count)` slots, and reports them as
  attention-cache indices `tokens + slot`. With five tokens only positions 3 and 4 may select
  slot 0; positions 0–2 leave the compressed slot at `-1`.
- **Attention over window + compressed slots.** The key/value cache is the `tokens` key rows
  followed by the compressed rows; the index row per position is the window slots followed by
  the selected compressed slots. `deepseek_v4_sparse_attention` is reproduced per (position,
  head) with the attention sink added to the denominator, then the inverse RoPE is applied at
  the token positions and the grouped `wo_a`/`wo_b` output projection runs per position.
- **Per-token MoE.** The BF16 router gate is projected per position; each position reads its
  own `tid2eid` row, activates the 256 gate logits with `sqrt(softplus)`, and normalizes only
  its six selected scores times `route_scale`. The shared expert runs for every position. Routed
  experts are executed in the C++ order (expert ascending, then position, then slot), so the
  float32 accumulation order is the same.
- **Head.** The Hyper-Connection head, final RMS normalization and LM head are evaluated on the
  last position only, as `final_logits` does with `token_count` rows.

### Single-token compatibility

`execute_layer` is now a one-token wrapper over `execute_prefill_layer`, so there is a single
implementation of the layer equations. Re-running the D3b chain (token 0, layers 0–2, depths
1–3) after the refactor produced **205 identical stage checksums** and identical per-depth
logits/hidden checksums (`logits-depth1/2/3.f32`, `hidden-depth1/2/3.f32`), and the same deltas
against the recorded probe dumps (logits `9.5367431640625e-07`, hidden `4.76837158203125e-07`).
The single-token CLI (`--token`, `--depths`, `logits-depthD.f32`) and its summary format
`vastloom.deepseek_v4.chain_oracle.v1` are unchanged.

The prompt mode is additive: `--tokens ID,...`, repeated `--experts LAYER=PATH`, a new summary
format `vastloom.deepseek_v4.prefill_oracle.v1`, and `logits.f32` / `hidden.f32`. `--token` and
`--tokens` are mutually exclusive; `--depths` is rejected in prompt mode because the native probe
evaluates the head only at the final layer.

## Direct evidence that compressed slots are non-empty and used

A five-token prompt closes exactly one 4:1 window (`5 // 4 = 1`) and leaves one tail position
that never enters a window. The oracle stages and reports the slot explicitly:

| Evidence | Value |
|---|---|
| `compression.compressed_layers[0].compressed_rows` | 1 |
| `compression.compressed_layers[0].compressed_selected` | 1 (`min(index_top_k=512, 1)`) |
| `compression.compressed_layers[0].window_selected` | 5 (the whole prompt) |
| `compression.compressed_layers[0].selected_per_token` | 6 |
| `layer2.attention_compressed` stage | 512 values = 1 slot × head_width 512 |
| `layer2.indexer_compressed` stage | 128 values = 1 slot × index_head_width 128 |
| `compression.total_compressed_rows` | 1 |

The attention index table of layer 2 is the direct proof that the slot is attended to, not just
computed (`layer2.attention_indices`, one row per position, six slots):

```
position 0: [ 0, -1, -1, -1, -1, -1]
position 1: [ 0,  1, -1, -1, -1, -1]
position 2: [ 0,  1,  2, -1, -1, -1]
position 3: [ 0,  1,  2,  3, -1,  5]
position 4: [ 0,  1,  2,  3,  4,  5]
```

Index `5` is `token_count + compressed_slot_0`: the compressed row appended after the five token
keys. Positions 0–2 leave it at `-1` because their compression window has not closed; positions
3 and 4 select it. The test requires this exact table, the non-zero stage counts, and a finite
non-zero compressed vector; a run whose compressed stage count is zero fails the section.

## Results

The oracle reproduces the recorded native dumps. Deltas are maximum absolute differences; the
tolerances are the D3b gate, unchanged.

| Quantity | Count | Max abs diff | Gate |
|---|---|---|---|
| Last-position logits | 129,280 | 1.9073486328125e-06 | ≤ 1e-3 |
| Final hidden (pre-normalization) | 4,096 | 5.960464477539063e-08 | ≤ 1e-3 and `rtol=1e-5`/`atol=1e-6` (passes both) |

| Quantity | Oracle | Native probe |
|---|---|---|
| Argmax | 20,267 | 20,267 |
| Logit minimum | −38.018795013427734 | −38.018795013427734 |
| Logit maximum | 14.700206756591797 | 14.700206756591797 |

Per-layer, per-position routes (sorted sets; identical in the oracle, the probe JSON and the
probe trace):

| Layer | Token 0 | Token 1 | Token 2 | Token 3 | Token 4 |
|---|---|---|---|---|---|
| 0 | 35, 53, 200, 222, 245, 254 | 23, 57, 115, 122, 202, 239 | 37, 38, 45, 96, 108, 228 | 42, 51, 69, 102, 162, 182 | 7, 67, 81, 154, 239, 246 |
| 1 | 8, 97, 137, 158, 163, 184 | 21, 63, 126, 150, 170, 219 | 13, 16, 17, 46, 73, 237 | 30, 102, 177, 192, 205, 242 | 71, 78, 125, 186, 188, 253 |
| 2 | 43, 53, 108, 115, 132, 143 | 26, 39, 217, 221, 240, 247 | 66, 87, 153, 185, 200, 216 | 71, 83, 85, 102, 122, 216 | 12, 32, 52, 101, 142, 224 |

The route sets also match the pinned `chain5-reference.json` (`routes_per_token`), which the
test treats as an independent expectation: a coordinated drift of the oracle and the probe still
fails against the pinned table.

### Artifact identity

| Quantity | Oracle merge | Native result |
|---|---|---|
| Expert bundles | 29 + 30 + 29 = 88 | `relocated_expert_bundles` 88 |
| Artifact ranges | 73 trunk + 270 expert = 343 | `relocated_tensors` 343 |
| Trunk bytes | 437,727,112 | `trunk_bytes_read` 437,727,112 |
| Expert artifact payloads | 678,297,600 | 15 artifact manifests |
| Unique-bundle bytes | 678,297,600 − 2 × 7,536,640 = 663,224,320 | `expert_bytes_read` 663,224,320 |

The two duplicated bundles (one per layer where two positions route to the same expert) explain
the 15,073,280-byte difference between the raw artifact sum and the unique-bundle bytes that the
native cache reports; the identity is exact, not approximate.

### Determinism

- `deepseek-chain-probe --tokens 0,1,2,3,4` was run twice by the test. The two logits dumps and
  the two hidden dumps are **byte-identical** (`read_bytes` equality, `numpy.array_equal` would
  be weaker) and the per-position routes in the JSON and trace are identical.
- The pre-refactor and post-refactor single-token chain runs share all 205 stage checksums and
  every per-depth dump checksum (see "Single-token compatibility").
- The oracle prefill run was repeated with stage files: all **485 stage checksums**, the
  `logits.f32` checksum and the `hidden.f32` checksum are identical, and both dumps are
  byte-identical (`read_bytes` equality).
- There is no RNG, no threading and no time-dependent input in the oracle.

### Stage coverage and cost

The oracle records 485 stage checksums for the five-token run: every layer-0/1/2 intermediate
(prefixed `layerL.`), each layer's final hidden state, the head stages (`hc_head_mixes`,
`final_hidden`, `final_norm`, `logits`), and per-position expert stages named
`expert_ID_tPOSITION_ROLE`. The test requires 391 of them explicitly (counts, SHA-256, FNV-1a,
finiteness) and recomputes both checksum maps. Three measured runs took 60.1 s, 61.8 s and
62.5 s for one prompt with stage files on the reference PC; the sampled peak working set was
347.8 MiB. The multi-token state is small enough that the peak is dominated by the LM-head
decode buffer, as in the single-token run (340.8 MiB), not by the prompt activations.

### Test harness

`tests/test_deepseek_layer_oracle.py` keeps the always-on synthetic section (104 pinned checks)
and the single-token layer-0 and chain sections, and adds `--prefill`:

- it requires `--chain-tokens 0,1,2,3,4`, `--chain-trunk` for layers 0–2 and fifteen
  `--chain-experts TOKEN:LAYER=PATH` maps;
- it runs `deepseek-chain-probe --tokens 0,1,2,3,4 --layers 3` twice and compares the dumps;
- it runs the oracle with `--tokens 0,1,2,3,4 --layers 3` and the same 15 artifacts;
- it compares logits, hidden, argmax and every per-layer per-token route, requires the compressed
  slot evidence above, and checks the merged artifact identity;
- it reports the first divergent stage with the offending index and both values when a value
  gate fails, and the first non-finite oracle stage when one exists.

Test output (abridged):

```
synthetic block decoders PASS (104 pinned checks)
probe determinism PASS (two runs, bit-identical logits and hidden, identical per-token routes)
compressed-slot evidence PASS (layer 2: 1 compressed row, 6 slots/token, compressed stages 512/128)
artifact identity PASS (88 merged bundles, 343 ranges, 678297600 expert artifact bytes)
per-token routes PASS (all 3 layers, 5 positions)
PASS (prompt [0, 1, 2, 3, 4], 3 layers, 391 stages, logits max|delta|=1.90734863e-06,
      hidden max|delta|=5.96046448e-08, argmax=20267, min=-38.018795, max=14.700207)
```

One harness correction was needed outside the new equations: the multi-token CLI writes the
trace key `tokens` (the pre-token trace wrote `token`). The single-token chain section now
accepts either key, so the D3b checks keep running against the current binary without changing
what they assert.

No divergent stage, layer or position was observed. If one had occurred, the test would name
the first one instead of reporting a pass. One boundary is worth stating precisely: the native
probe does not expose its compressed index table or the compressed key/value rows, so the C++
side of the compressed-slot claim is the value agreement over the whole three-layer chain, not
a native stage dump. The oracle's own table and stage counts are direct; the probe's route
lists and the 1.9e-06 logits delta are what tie that path to the native execution.

## Gate status

- **D3c: met for this prompt** (layers 0–2, prompt `[0, 1, 2, 3, 4]`). Logits and final hidden
  are reproduced well inside the unchanged tolerances; argmax and all fifteen per-layer
  per-token routes agree with the native probe and with the pinned reference; the layer-2
  compressed slot is non-empty and selected by the two positions that may see it; the merged
  artifact identity matches the probe's relocation counters exactly. The native executor side is
  the Lead's `deepseek-chain-probe --tokens` result, deterministic across two runs and recorded
  in `chain5-reference.json`; the multi-token leg is also pinned in the chain contract test.
- The prefill parity test is opt-in and is invoked manually with the bundled NumPy interpreter
  and the imported artifacts (reproduction below); it is not part of the always-on suite.
- No blocker was hit: the compressed-slot path was reproducible, so the task's fallback
  (deliver correct multi-token results without compression) was not needed.

## Reproduction

The PATH Python has no NumPy; use the bundled interpreter (Python 3.12, NumPy 2.3.5).

```powershell
$python = "<NumPy-capable Python>"
$opt = "build\analysis\deepseek-0731\store\models\optimized"
$p   = "deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range"
$md  = "build\analysis\deepseek-0731\store\models\original\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata"

& $python tests\test_deepseek_layer_oracle.py --prefill `
  --vastloom build\release\vastloom.exe --model $md --globals "$opt\$p-globals-067131537b81" `
  --chain-trunk "0=$opt\$p-l0-trunk-e62746099f66" `
  --chain-trunk "1=$opt\$p-l1-trunk-6995ba972b9b" `
  --chain-trunk "2=$opt\$p-l2-trunk-a4de2e64fef7" `
  --chain-experts "0:0=$opt\$p-l0-experts-c9b27ecd7c32" `
  --chain-experts "1:0=$opt\$p-l0-experts-25a997485fa5" `
  --chain-experts "2:0=$opt\$p-l0-experts-fcdef978d641" `
  --chain-experts "3:0=$opt\$p-l0-experts-f583b115c0b8" `
  --chain-experts "4:0=$opt\$p-l0-experts-a62cc5c70505" `
  --chain-experts "0:1=$opt\$p-l1-experts-9fc8722240e4" `
  --chain-experts "1:1=$opt\$p-l1-experts-98dc681775ab" `
  --chain-experts "2:1=$opt\$p-l1-experts-dad449f1f68b" `
  --chain-experts "3:1=$opt\$p-l1-experts-5a1fcb2fe132" `
  --chain-experts "4:1=$opt\$p-l1-experts-243b585ebcf4" `
  --chain-experts "0:2=$opt\$p-l2-experts-7bbd313e360a" `
  --chain-experts "1:2=$opt\$p-l2-experts-02c6893f873e" `
  --chain-experts "2:2=$opt\$p-l2-experts-ee4a0162688a" `
  --chain-experts "3:2=$opt\$p-l2-experts-16a50bc0ab82" `
  --chain-experts "4:2=$opt\$p-l2-experts-bbebf970fc6d" `
  --chain-tokens 0,1,2,3,4
```

The oracle alone (prompt mode; one `--experts LAYER=PATH` per prompt-token route, five per
layer, exactly as the probe accepts them):

```powershell
& $python tools\reference_deepseek_v4_layer_oracle.py --model $md `
  --trunk "0=...l0-trunk..." --trunk "1=...l1-trunk..." --trunk "2=...l2-trunk..." `
  --experts "0=...l0-t0..." --experts "0=...l0-t1..." --experts "0=...l0-t2..." `
  --experts "0=...l0-t3..." --experts "0=...l0-t4..." `
  --experts "1=...l1-t0..." --experts "1=...l1-t1..." --experts "1=...l1-t2..." `
  --experts "1=...l1-t3..." --experts "1=...l1-t4..." `
  --experts "2=...l2-t0..." --experts "2=...l2-t1..." --experts "2=...l2-t2..." `
  --experts "2=...l2-t3..." --experts "2=...l2-t4..." `
  --globals "...globals..." --layers 3 --tokens 0,1,2,3,4 `
  --output-dir build\analysis\deepseek-0731\oracle-work\d3c-prefill5 --stages
```

Artifacts and dumps are local validation data under `build/`; they are not committed.

## Files

- `tools/reference_deepseek_v4_layer_oracle.py` (143,176 bytes): prompt mode (`--tokens`,
  repeated `--experts LAYER=PATH`), `execute_prefill_layer`, per-token Hyper-Connection, window
  table, per-token compressor/indexer selection and sparse attention, expert-artifact merge, new
  summary format and `logits.f32`/`hidden.f32` dumps; the single-token wrapper preserves the
  D3b surface.
- `tests/test_deepseek_layer_oracle.py` (96,660 bytes): the `--prefill` section (probe
  determinism, oracle parity, per-token routes, compressed-slot evidence, artifact identity) on
  top of the unchanged synthetic, layer-0 and chain sections; 104 synthetic checks remain.
- `docs/reports/D3C_MULTITOKEN_ORACLE_VALIDATION.md`: this report.
- Local evidence (not committed): `build/analysis/deepseek-0731/chain5-logits-t0-4.f32`,
  `chain5-hidden-t0-4.f32`, `chain5-reference.json`, `chain5-trace-t0-4.json` and the oracle
  output under `build/analysis/deepseek-0731/oracle-work/d3c-prefill5`.

No file under `src/`, `include/` or `CMakeLists.txt` was modified; the metadata store and the
imported artifacts were read only.

## Not established

- Three layers of 43 over one five-token prompt: not the model's output, not a full-depth run,
  not a generation result.
- Only compression ratio 4 is exercised natively; ratio 128 remains refused. Only hash-routed
  layers 0–2 run; a learned router needs the post-attention state and complete expert coverage.
- The oracle still trusts the imported artifact payloads; it does not re-verify their SHA-256,
  which the probe does on load.
- No incremental/decode path, no KV reuse, no chunked prefill, no GPU chain, no residency or
  throughput claim.
- Not bit-identical: raw FNV hashes differ between the MSVC scalar probe and NumPy, as in D3b;
  the value gate is the claim.
- The prefill test is opt-in and needs the imported artifacts and the release binary; it was run
  serially. Running it concurrently with another model probe can be refused by the Safety
  Governor for RAM headroom, and that refusal is environmental, not a parity result.
