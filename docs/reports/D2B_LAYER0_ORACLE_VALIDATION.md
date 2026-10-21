# D2b.2 — independent layer-0 oracle parity

Validation date: 2026-10-21. Track: DeepSeek V4 Flash 0731, `UD-IQ2_XXS`, layer 0. This report
covers the independent logits oracle required by D2b and records the first external-reference
comparison of the range-slice payload path.

## Scope

The native `deepseek-layer-probe` executed layer 0 from the three imported range artifacts
(trunk 138,867,160 B / 21 ranges, experts 45,219,840 B / 18 ranges, globals 596,000,788 B /
6 ranges) and produced 129,280 logits plus the 4,096-wide pre-normalization hidden state. A
separate Python implementation reads the same artifact bytes, dequantizes them independently and
recomputes the layer. The gate: the two logits vectors agree within a declared tolerance and the
route/argmax agree. This is layer 0 of 43 and never the model's output.

## Method

- **Inputs, no C++ calls.** The oracle takes the pinned metadata store (GGUF v3 headers for the
  three shards) and the three `vastloom.range-artifact.v1` directories. It parses the GGUF
  metadata and tensor tables itself, maps every artifact range to a tensor by source-object
  basename, exact source offset and byte count (packed expert ranges are matched as aligned
  slices of the 3-D container tensors), and checks the manifest scope, layer, revision, range
  count and local file sizes. Payload SHA-256 values are not re-verified by the oracle: the
  probe verifies them on load, and this oracle reads the already-published files. The oracle is
  deterministic, has no network path and never imports or executes C++ code.
- **Independent decoders.** Q8_0, Q5_K, Q6_K, Q4_K, BF16, F32, I32, IQ2_XXS and IQ3_XXS are
  implemented from the block layouts in `src/inference/ggml_quantization.cpp`. The two IQ
  codebooks are the pinned ggml constants; their canonical FNV-1a fingerprints are asserted at
  startup. The decoders were checked against the hand-written blocks and expected values pinned
  in `tests/test_main.cpp` against ggml commit `30bf8685`: 87 checks, all passing.
- **Equations.** Embedding row, RMSNorm, Hyper-Connection split/Sinkhorn/pre/post, MLA attention
  (sliding window, Q/K RoPE, sink), hash routing from `tid2eid`, shared expert SwiGLU with the
  checkpoint's clamp schedule, six routed experts, Hyper-Connection head and the LM head. Every
  graph tensor is float32; matrix-vector products accumulate in float64 in column order with one
  float32 rounding (`numpy.cumsum` reproduces the reference kernel's serial sum). Layer 0's
  compression ratio is the checkpoint's 0, so the compressor/indexer path is not modelled; the
  oracle refuses a non-zero layer-0 ratio rather than guessing.
- **Reused material.** The equation structure and the float32 boundary choices were taken from
  `tools/reference_deepseek_v4_operators.py` (RMSNorm, Hyper-Connection split/post ordering,
  RoPE, sparse attention, expert SwiGLU) and the pinned ggml codebooks; no code was imported from
  the C++ runtime. One implementation defect was found while cross-checking: the Q6_K decoder
  initially used the first half's scales for the second half, which moved argmax to 45,178 and
  the logits by up to 0.4; using the correct second-half scale block restored agreement to 1e-6.
  That defect was in this Python oracle, not in the C++ probe.
- **Comparison harness.** `tests/test_deepseek_layer_oracle.py` runs the probe with
  `--dump-logits`, `--dump-hidden` and `--dump-trace`, runs the oracle with `--stages`, and
  compares logits, final hidden state, route, artifact identities and the oracle's stage files.
  Tolerances are constants in the test: logits `atol = 1e-3`, hidden `rtol = 1e-5`,
  `atol = 1e-6`. A raw FNV-1a equality check is reported as a diagnostic and is not asserted: a
  bitwise hash has no tolerance, and the residual 1-ulp differences described below would make
  that assertion fail for a correct implementation. `--require-checksum-equality` exists for a
  future bit-identical implementation.

## Implementation files

- `tools/reference_deepseek_v4_layer_oracle.py` (90,056 bytes): parser, decoders, layer equations,
  `--model/--trunk/--experts/--globals/--token/--output-dir/--stages` CLI. Writes
  `logits.f32`, `summary.json` (66 stage checksums) and optional `stages/<name>.f32`. Peak
  working set measured at 336 MiB, wall time about 9.4 s for one token on the reference PC.
- `src/cli/main.cpp`: `--dump-logits`, `--dump-hidden`, `--dump-trace` on the probe (raw f32 and
  a small JSON summary), added in parallel by the Lead.
- `tests/test_deepseek_layer_oracle.py`: always-on synthetic section (87 pinned decoder/matvec
  checks) plus the opt-in artifact integration. It checks for `--dump-logits` in the probe help
  and CLI source and reports a blocker instead of falling back to a weaker comparison.
- `CMakeLists.txt`: opt-in variable `VASTLOOM_DEEPSEEK_V4_ORACLE_PYTHON` (a NumPy-capable
  interpreter) and test `vastloom_deepseek_v4_layer_oracle`, registered only when the three
  artifacts and that interpreter are supplied.

## Results

Artifacts and command from the D2b.1 report; the oracle and probe read the same files. Synthetic
section: **PASS, 87 pinned checks**. Integration section:

| Quantity | Probe | Oracle | Comparison |
|---|---|---|---|
| Logits | 129,280 values | 129,280 values | max abs diff **9.5367431640625e-07** (gate `1e-3`); 12,648 values differ bitwise |
| Final hidden | 4,096 values | 4,096 values | max abs diff **4.76837158203125e-07** (gate `rtol 1e-5`, `atol 1e-6`); 47 values differ bitwise |
| Logit minimum / maximum | −16.127300262451172 / 15.52056884765625 | identical | equal |
| Argmax | 30,528 | 30,528 | equal |
| Route | 254, 222, 245, 200, 53, 35 | same | equal |
| Route weights | — | 0.0538878515, 1.0755851269, 0.1847247481, 0.0483085811, 0.0697485581, 0.0677451119 | sum 1.5, as required by the checkpoint scale |
| Oracle stage checksums | probe trace: 2 (logits, hidden) | 66 | **stage-by-stage comparison blocked**: the probe trace has only logits/hidden checksums |
| Logits FNV-1a 64 | 8,508,266,960,150,154,582 | 9,066,088,410,461,933,972 | differs; diagnostic only |
| Hidden FNV-1a 64 | 1,613,290,218,847,626,553 | 848,093,564,231,876,881 | differs; diagnostic only |

Test output: `PASS (logits max|delta|=9.53674316e-07, hidden max|delta|=4.76837158e-07,
argmax=30528, route=[254, 222, 245, 200, 53, 35], stages=17, checksum_matches=0)`.
Here `stages=17` is the number of stage dumps the test requires; the oracle records 66.

The residual is last-bit float32 rounding: the probe's MSVC scalar path and NumPy's float64
accumulation evaluate the same equations in the same order but do not share an instruction
stream, so a small fraction of values round to the neighbouring float32. The measured maximum is
two orders of magnitude inside the 1e-3 gate; the raw checksums are the diagnostic that says a
future implementation could be tightened, not a value failure.

## Stage coverage and the blocker

The oracle writes a checksum for every stage: embedding, both Hyper-Connection pre/split/reduced
blocks, normalized attention input, q-rank/norm/q, kv-norm/k/v, attention context/inverse
RoPE/grouped/output, attention post, ffn mix/normalized, router logits/scores/route/weights,
shared expert gate/up/activated/output, each selected expert's gate/up/activated/output, MoE
output, ffn post, Hyper-Connection head mixes/final hidden/final norm and logits. The test
recomputes the SHA-256 of every required stage file and cross-checks count, SHA-256 and FNV in
both summary maps.

Only two of those stages have a counterpart in the probe trace, because the probe exposes raw
dumps and checksums for the logits and the final hidden state only. The other 64 stage checksums
are recorded but cannot be compared yet. Closing that gap needs `--dump-stages` (or a `stages`
object in `--dump-trace`); the test already compares such an object when present. This is a
coverage limitation of the trace, not a tolerance exception: no stage tolerance was loosened.

## Gate status

- **D2b.2 (independent oracle): met** for layer 0. The probe's logits and hidden state are
  reproduced by an independent implementation within the declared tolerance; argmax and route
  agree; the six decoders pass the pinned ggml vectors.
- **D2b as a whole: the placement half is D2b.1** (`docs/reports/D2B_DEVICE_RESIDENCY_VALIDATION.md`:
  device expert residency with bounded CPU fallback, bit-identical logits). The two halves cover
  the gate's wording. A true VRAM **payload** tier remains open and is explicitly not claimed.
- **D3 gate remains open:** the oracle covers layer 0 only; layers 1–2 (including the layer-2
  compressor/indexer path) and the chain comparison are D3b.

## Tolerances

| Comparison | Tolerance | Measured |
|---|---|---|
| Logits | max abs diff <= 1e-3 | 9.54e-7 |
| Final hidden | rtol 1e-5, atol 1e-6 (and max abs <= 1e-3) | 4.77e-7 max abs, 2.15e-7 max relative |
| Stage files | FNV/SHA fully reproducible from the written bytes | 66/66 |
| Probe stage checksums | exact equality where the trace exposes them | 0/2 (1-ulp difference; diagnostic) |

## Not established

- No stage-by-stage comparison against the probe beyond logits and final hidden; the probe trace
  has no stage dumps. The oracle records all 66 stage checksums for when that interface exists.
- No llama.cpp/ggml end-to-end comparison and no full-checkpoint reference: the checkpoint is
  absent by design. The decoder vectors are pinned to ggml, but the graph path is not compared to
  a ggml graph.
- Not bit-identical: raw FNV differs, and the oracle is not a substitute for the C++ CPU path in
  performance or in bit-exact regression tests.
- Layer 0 only, token 0 only in the recorded run; no multi-layer chain, no VRAM payload tier, no
  throughput claim, no full-model claim.
- The oracle does not re-verify artifact payload SHA-256; it trusts the imported artifacts that
  the probe also verifies on load.

## Reproduction

The PATH Python on the reference PC has no NumPy. Use the bundled DSH interpreter (Python 3.12.14,
NumPy 2.3.5) for every invocation:

```powershell
$python = "<NumPy-capable Python>"
& $python tests\test_deepseek_layer_oracle.py `
  --vastloom build\release\vastloom.exe `
  --model build\analysis\deepseek-0731\store\models\original\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata `
  --artifact build\analysis\deepseek-0731\store\models\optimized\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-trunk-e62746099f66 `
  --experts build\analysis\deepseek-0731\store\models\optimized\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-experts-c9b27ecd7c32 `
  --globals build\analysis\deepseek-0731\store\models\optimized\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-globals-067131537b81
# or, when the three artifacts and the interpreter are configured:
#   ctest -R vastloom_deepseek_v4_layer_oracle
```

The integration test stages its temporary dumps under `build/` (native writes outside the
workspace are denied on the reference host) and removes them unless `--keep` is given. The
synthetic section also runs without artifacts:

```powershell
& $python tests\test_deepseek_layer_oracle.py
```

Artifacts and dumps are local validation data under `build/`; they are not committed.
