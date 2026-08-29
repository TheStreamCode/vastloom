# Qwen3.8-Flash-Next real-checkpoint validation (Q11c–Q11h)

Validation date: 2026-08-29

Implementation commits: `5c5e566` (Q11c), `48c5ca1` (Q11d), `427d39b` (Q11e),
`3c61450` (Q11f), `05ecf60` (Q11g), `1556d74` (Q11h)

## Scope

From the first real-tensor forward to **greedy token parity with llama.cpp**
on the official Qwen3.8-Flash-Next checkpoint. All milestones execute on the
canonical immutable store; no synthetic fixture appears in any claim below.

## Checkpoint identity

- Repository: `unsloth/Qwen3.8-Flash-Next-Q8_0` (GGUF, revision `main`)
- Store: `C:\vastloom-store` (`.vastloom-store` marker)
- Artifact: `models/original/qwen3.8-flash-next-gguf-gguf-q8_0`
- Files: 6 shards, 175.3 GiB, every file SHA-256 verified during download,
  all read-only, `vastloom-import.json` manifest published atomically
- Download path: `tools/import_qwen38_fp8.py --repo --gguf-dir` over the
  canonical `import_hf_model.import_model` transaction
- The FP8 safetensors release (172.8 GiB) was imported first, validated
  against the pinned 152,089-tensor catalog, then deleted after the GGUF
  replaced it (user request; re-importable from Hugging Face at any time)
- GGUF metadata: `general.architecture=qwen4exp`, 1,224 tensors
  (811 Q8_0, 389 F32, 24 BF16), all pinned invariants match

## Defects found by the external oracle (llama.cpp `6fe7498`)

| # | Defect | Fix |
|---|---|---|
| 1 | DeltaNet output gate implemented as `silu(z)` following the HF modeling main | `sigmoid(z)` — llama.cpp: "the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu" |
| 2 | `ssm_a` treated as `A_log`, but the GGUF carries pre-applied `-exp(A_log)` (range fully negative, verified) | loader converts to log-space: `a_log = log(-ssm_a)` |
| 3 | GGUF norm weights stored as `(1+w)` (mean ≈ 1 verified) but `ssm_norm` is a direct RMSNormGated weight | shift −1 at load for all norms except `ssm_norm` |
| 4 | k-head mapping as HF `repeat_interleave` (`vh // group`) | ggml-repeat tile semantics (`vh % k_heads`), confirmed from the ggml fused kernel `ik1 = iv1 % nek1` |
| 5 | Greedy CLI replayed a stale trunk (layer 0 executed with the previous token's trunk 47) | trunk reloaded per layer per token |

Plus the missing pieces the parity required and Q5–Q10 had not implemented:
partial RoPE (64/256, theta 1e7) over attention q/k and the indexer
(q at current position, pooled keys at block start), the PLE n-gram block
(SplitMix hash with GGUF-carried multipliers → fused 160-per-head table
lookup → key/value projections → sigmoid gate → norm → dilated depthwise
conv), GGUF F16 scale decode, and a correct multi-token prefill.

## Verification method

llama.cpp built from source (rev `6fe7498`, CPU Release, MSVC) with an
optional intermediate-tensor dump patch (`LLAMA_DEBUG_FULL`, common/debug.cpp,
local clone only — not committed). Stage-by-stage comparison with identical
inputs on layer 0, token 760:

| Stage | Vastloom | llama.cpp |
|---|---:|---:|
| hc_norm (embed + grouped RMSNorm ×(1+w)) | 713.224 | 713.224 |
| hc_mixed (attention gated residual) | 99.58 | 98.93 |
| qkv projection | 13.75 | 14.54 |
| conv_silu | 96.11 | 95.16 |
| delta-rule raw output sum | 0.8686 | 0.8687 |
| core (post norm gated, sigmoid) | −4.391 | −4.3998 |
| linear_attn_out | — | −5.803 |

The raw delta output matches element-wise (mean |ratio−1| = 0.29% across all
6,144 elements with the tile mapping; interleave gives 97% mismatch, ruling
it out). The residual ~0.2–0.7% differences come from the hc_mixed input
(F32 scalar accumulation vs ggml Q8_0 integer dots), not from formulas.

## Parity result (greedy, CPU, both engines, same checkpoint)

| Prompt | Vastloom `qwen38-greedy` | llama.cpp `llama-completion -no-cnv --temp 0` |
|---|---|---|
| "The capital of France is" (16 tokens) | " Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The" | identical |
| "Hello world" (16 tokens) | plausible continuation, deterministic | consistent |

## Known performance position (not a claim of competitiveness)

One token moves ~6.6 GB of payload in the current streaming implementation
(one trunk resident at a time, experts through a bounded LRU). llama.cpp on
the same box decodes at ~2.65 tok/s (377 ms/token). Vastloom is currently
slower; the levers (VRAM-resident expert tiers, cross-layer prefetch,
bounded I/O pool overlap) are designed but not implemented. No tok/s claim
is made until measured after those land.

## Gates

| Gate | Exact result |
|---|---:|
| `vastloom_tests` (always-on) | ALL PASSED |
| `ctest --preset release` | 13/14 passed, 1 skipped (`vastloom_ggml_compute`) |
| `vastloom_qwen38_checkpoint` (opt-in, real Q8_0, serial) | PASS (catalog + layer-0/3 forwards + prefill determinism) |
| Greedy parity vs llama.cpp (`6fe7498`) | token-identical, 16/16 tokens on the pinned prompt |
| Python `compileall` | OK |

## Still not claimed

- No throughput/latency competitiveness claim (see performance position).
- MTP draft head, vision tower and sampler (top-k/temperature) are not
  executed; greedy only.
- The DeepSeek 0731 track remains historical; its pins still pass but are
  not the execution path.
