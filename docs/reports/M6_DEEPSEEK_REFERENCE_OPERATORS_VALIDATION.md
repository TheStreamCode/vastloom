# M6 DeepSeek V4 Flash reference-operator validation

Validation date: 2026-08-10

Implementation commit: `aca4fc5`

## Scope

This report validates the first bounded native CPU equations for the official
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
repository at immutable commit
`60d8d70770c6776ff598c94bb586a859a38244f1`:

```text
pinned model equations
  -> deterministic reduced numeric vectors
  -> bounded native FP32 reference operators
  -> cross-compiler and sanitizer comparison
```

This is an operator-level foundation, not DeepSeek inference. It does not load model
weights, execute a Transformer layer, produce logits or generate text.

## Implemented reference operators

- FP32 RMSNorm with double-precision accumulation;
- learned `sqrt(softplus)` routing with selection-only correction bias, normalized
  authoritative route weights and route scaling;
- token-ID hash routing through the checkpoint's `tid2eid` contract;
- routed-expert SwiGLU with the model's asymmetric clamp and optional route weighting;
- Hyper-Connection pre-mixing, pre/post coefficients, row softmax and alternating
  Sinkhorn normalization;
- Hyper-Connection residual recombination;
- forward and inverse base RoPE;
- selected-latent sparse attention with the per-head attention sink included in the
  softmax denominator.

All public entry points validate dimensions, exact tensor lengths, finite inputs and
normalization scalars. Shape multiplication is overflow checked and every reference
tensor is capped at 64 million elements before allocation.

## Deterministic fixture

`tools/reference_deepseek_v4_operators.py` is a dependency-free equation-level Python
translation. A fixed xorshift generator produces the committed
`tests/fixtures/deepseek_v4_operators.json` vectors for every operator above. CTest runs
the generator in `--check` mode, so an edited or non-reproducible fixture fails the gate.

The native tests compare values with an explicit absolute tolerance of `3e-5` and compare
learned and hash expert IDs exactly. They also reject invalid shapes, non-finite routing
bias, out-of-range token/expert indices, odd RoPE width and invalid sparse-attention
selection.

The Python translation is intentionally independent of the C++ implementation, but it is
not yet the trusted official end-to-end model oracle. The M6 gate still requires a reduced
official graph with reference logits, routes and tokens.

## Build and test matrix

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors, focused current-source gate | 3/3 passed in 0.38 s |
| GCC 13.3 Release, warnings as errors | 7/7 passed in 3.36 s |
| Clang 18.1 Release, warnings as errors | 7/7 passed in 3.27 s |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 7/7 passed in 10.07 s |

ASan used leak detection and halt-on-error; UBSan halted on the first finding. Fixture
reproduction, Python byte compilation and `git diff --check` also passed.

The separately configured MSVC Release suite passed the nine small/real-model tests but
the large OLMoE runtime test was refused by the existing Safety Governor after WSL and
file-cache activity reduced the safe RAM cap below its 3.2 GB minimum. A retry progressed
farther and then refused the optional 2 GiB expert-cache reservation. This was a bounded
pre-allocation refusal (`returncode=75`), not a crash, OOM or DeepSeek failure; its exact
results were 9/10 followed by 0/1 for the isolated retry. The earlier clean run in the
same development session passed 10/10 before the available-memory pressure developed.

## Performance and safety

No token rate, TTFT or model-memory number is reported because no forward graph ran. The
committed fixture is approximately 12 KiB and no checkpoint payload was read. The tests
therefore cannot predict full-model speed or thermals.

The large-model interlock remains unchanged. DeepSeek's 159.6 GB weight payload is not
downloaded automatically, and none of these operators bypasses the Resource Governor or
authorizes a full-model allocation.

## Known limitations and next bottleneck

- no architecture-faithful reduced layer or full forward graph;
- no YaRN-scaled RoPE, compressed-position/indexer construction or complete MLA path;
- no shared-expert aggregation, MTP block or Hyper-Connection LM head;
- no packed FP4/FP8 dequantization;
- no reference logits, generated tokens or full-model probe;
- no DeepSeek tensor residency, NVMe streaming, CUDA or VRAM allocation.

The next bottleneck is composing these primitives with the missing attention/indexer,
shared-expert and MTP equations into a tiny architecture-faithful graph, then exporting
logits and routes from the pinned official implementation.
