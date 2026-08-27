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
