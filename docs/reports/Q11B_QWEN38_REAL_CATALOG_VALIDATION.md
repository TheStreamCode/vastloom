# Qwen3.8-Flash-Next real checkpoint catalog validation (Q11b)

Validation date: 2026-08-28

Implementation commit: `c31d62d` (Q5–Q11) + this change

## Scope

First real-payload milestone on the official 125B FP8 checkpoint: canonical
immutable import and complete native catalog validation. Metadata only — no
tensor payload has been read for compute.

## Checkpoint identity

- Repository: `unsloth/Qwen3.8-Flash-Next-FP8`, revision `main`
- Store: `C:\vastloom-store` (`.vastloom-store` marker, models/original + cache)
- Artifact: `models/original/unsloth-Qwen3.8-Flash-Next-FP8--main`
- Files: 140 (131 weight shards + 8 metadata + `model.safetensors.index.json`)
- Weight bytes: 185,523,317,458 (172.8 GiB), every file SHA-256 verified during
  download, all files read-only, manifest `vastloom-import.json` published
  atomically after full verification
- Download path: `tools/import_qwen38_fp8.py` over the canonical
  `import_hf_model.import_model` transaction (bounded reads, per-file hash,
  hardlink staging, atomic rename, cache cleanup)

## Network findings (anonymous vs authenticated)

- Anonymous HuggingFace downloads from this network are throttled by a
  cumulative per-IP token bucket: ~140–170 MB/s burst for roughly 13 GiB,
  then ~30 KB/s sustained. A 172.8 GiB transfer is infeasible anonymously.
- With an authenticated token (`HF_TOKEN` environment variable, read by the
  importer, never logged or passed on the command line) the same transfer
  sustains ~78 MB/s and completes in ~35 minutes.
- `model.safetensors.index.json` was initially missing from the artifact
  because the driver's metadata filter excluded any path containing
  "safetensors"; fixed and the file was fetched and placed read-only.

## Validated catalog (native C++, `vastloom qwen38-inspect --load-header`)

| Property | Expected (pinned) | Real checkpoint |
|---|---:|---:|
| Total tensors | 152,089 | 152,089 |
| Weight shards | 131 | 131 |
| FP8 block-scale expert tensors | 75,264 | 75,264 |
| N-gram PLE tensors | 129 | 129 |
| Language layers | 48 | 48 |
| Vision tensors | 333 | 333 |
| MTP tensors | 3,101 | 3,101 |
| Global tensors | 5 | 5 |
| Gated residual (global + per-layer) | yes | yes |
| First shard header | parses | parses (22 tensors, 3 FP8 + scale_inv) |

One fixture-to-real drift was found and corrected: the real checkpoint carries
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale`,
which the Q1 synthetic fixture lacked. `Qwen38ExpectedCatalog::ngram_tensors`
moved from 128 to 129. The Q6 non-routed graph planner already listed the
tensor, so no planner change was needed.

## Gates

| Gate | Exact result |
|---|---:|
| `vastloom_tests` (always-on, MSVC) | ALL PASSED |
| `ctest --preset release` | 13/14 passed, 1 skipped (`vastloom_ggml_compute`, intentional) |
| `vastloom_qwen38_checkpoint` (opt-in, real checkpoint, serial) | PASS |
| `vastloom qwen38-inspect --load-header` (real root) | validated, exit 0 |
| Python `compileall` tools+tests | OK |
| Python importer offline tests | PASS |

## Not claimed

- No tensor payload has been decoded for compute. The next milestone binds
  real tensors (starting with the layer-0 non-routed set) into the reference
  graph and runs prefill/decode parity.
- No throughput, latency or quality claim is made about the real checkpoint.
- `model.safetensors.index.json` was placed into the read-only artifact after
  import; a future re-import with the fixed driver will include it natively.

---

# Q11c addendum: real layer-0 forward (2026-08-28)

## Implemented

- `vastloom qwen38-layer0 <root> --token N [--json] [--dump-hidden]`: the full
  official layer-0 on real weights — bounded single-row embedding read
  (5 KiB), attn gated residual, Gated DeltaNet from a zeroed state, combine,
  mlp gated residual, router top-10 of 512, shared expert, on-demand load of
  the 10 routed FP8 experts, final gated-residual combine.
- Stage-named finiteness guards (embed stream, attn GR, DeltaNet, mlp GR,
  shared expert, per-expert weights and outputs).
- Opt-in checkpoint test extended with the forward: finite output, exact
  top-k route, bit-identical determinism across two full runs.

## FP8 decoder defect found on real payload

`qwen38_fp8_e4m3_to_f32` returned NaN for every byte with `exp==15`. In the
E4M3FN format actually used by the checkpoint (HF `quantization_config
fmt=e4m3`), NaN is only `exp==15 && mant==7` (0x7F/0xFF); bytes `0x78..0x7E`
are finite values (256..448) and occur in real weights — 3,370 such bytes in
`layers.0.mlp.experts.411.gate_proj.weight` alone. The old decoder turned
every one into NaN, which surfaced as a non-finite layer output only when the
first real FP8 expert was decoded; synthetic Q2 fixtures never contained these
encodings. Fixed, with the Q2 pin extended (0x78→256, 0x7E→448, 0xFE→−448).

## Measured (reference PC, single run, warm cache)

| Metric | Value |
|---|---:|
| Payload bytes read (trunk + 1 embed row + 10 experts) | 203,950,528 |
| Expert payload bytes (10 experts × 3 matrices) | 49,152,000 |
| Load time (config+index+catalog+trunk+row) | ~1.2 s |
| Forward time (full layer-0, scalar F32 reference) | ~46 ms |
| Token BOS (248044) selected experts | 411, 334, 286, 152, 291, 106, 421, 304, 226, 331 |
| Token 100 selected experts | 215, 226, 122, 448, 483, 363, 459, 40, 239, 500 |
| Layer-0 output sum (BOS) | 3.0418 (bit-identical across runs) |

## Gates

| Gate | Exact result |
|---|---:|
| `vastloom_tests` (always-on) | ALL PASSED |
| `ctest --preset release` | 13/14 passed, 1 skipped (`vastloom_ggml_compute`) |
| `vastloom_qwen38_checkpoint` (opt-in, real, serial) | PASS (catalog + layer-0 forward) |
| Python `compileall` | OK |

## Still not claimed

- The layer-0 run is self-pinned: no external oracle for `qwen4_exp` exists
  (llama.cpp lacks the architecture; no torch on this machine). The FP8
  decoder now matches the documented E4M3FN semantics and the composition
  operators were pinned against the Q5 Python equation oracle, but the full
  layer output has no independent reference.
- Prefill (multi-token), QSA layers (3, 7, 11, ...), MTP and the 48-layer
  generate remain future work.

