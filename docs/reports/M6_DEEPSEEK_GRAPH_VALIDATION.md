# M6 DeepSeek V4 Flash reduced-graph validation

Validation date: 2026-08-10

Implementation commits: `9a6be5d`, `d413fe8`, `91cf229`

## Scope

This report closes the reduced float graph gate for the official
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
repository at immutable revision
`60d8d70770c6776ff598c94bb586a859a38244f1`:

```text
pinned official model.py
  -> deterministic 102,104-byte float fixture
  -> bounded native two-layer prefill graph
  -> logits + greedy token + routes + sparse-attention indices
```

This is real architecture-level reference execution, but it is not the full checkpoint.
No 159.6 GB weight payload, packed FP4/FP8 tensor, CUDA allocation or NVMe expert stream
is involved.

## Trusted reference construction

The exporter verifies official `inference/model.py` with SHA-256
`ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71` before import.
It instantiates the official classes on CPU with deterministic FP32 parameters. Quantized
GEMM calls are forbidden for this fixture. The two TileLang entry points required by the
official model are replaced by equation-equivalent CPU implementations already covered by
the separate operator vectors. Activation quantization is a no-op because this is the float
oracle; the shared Hadamard transform in the indexer can likewise be omitted without changing
its unquantized query/KV dot products.

`tools/export_deepseek_v4_fixture.py --check` reruns the pinned official graph and requires
byte-identical JSON. The dependency-free `--validate-only` path runs in ordinary CTest and
checks the committed source identity and fixture structure.

## Reduced architecture

| Property | Fixture value |
|---|---:|
| Input tokens | 8 |
| Vocabulary / width | 16 / 8 |
| Main layers | 2 |
| Attention heads / head width / RoPE width | 2 / 6 / 4 |
| Hyper-Connection multiplier | 2 |
| Routed experts / top-k / shared experts | 4 / 2 / 1 |
| Hash-routed layers | 1 |
| Learned-routed layers | 1 |
| Sliding window | 3 |
| Layer-1 compression | 4:1 overlap + learned indexer |
| Fixture tensors | 82 |
| Float / integer elements | 3,293 / 32 |

The native prefill follows the official order:

```text
embedding -> HC expand
  -> HC pre -> RMSNorm -> MLA projection/window/compressor/indexer/sparse attention
  -> inverse RoPE -> grouped output LoRA -> HC post
  -> HC pre -> RMSNorm -> hash or learned router
  -> weighted routed experts + unweighted shared expert -> HC post
  -> HC head -> final RMSNorm -> LM head
```

Base and YaRN-scaled RoPE are both exercised. The compressed layer uses independent
attention and indexer compressors, overlapping gated pooling, causal compressed positions,
learned top-k selection and the attention sink.

## Exact parity gate

Against the pinned official exporter, the native result passed:

- all 16 final logits within absolute tolerance `8e-5`;
- exact greedy token ID `4`;
- all 32 routed expert IDs exactly;
- all 32 routing weights within `8e-5`;
- all 64 raw/compressed sparse-attention indices exactly.

The test removes each of the 82 fixture tensors in turn and requires the native forward to
fail, proving that the declared fixture contains no unexercised parameter. It also rejects a
missing LM head, non-finite embedding, token outside the vocabulary and an overflowing graph
configuration before unsafe allocation.

## Build and test matrix

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors, focused final-source gate | 4/4 passed in 0.41 s |
| MSVC 19.44 Debug, warnings as errors, focused final-source gate | 4/4 passed in 0.86 s |
| GCC 13.3 Release, warnings as errors | 8/8 passed in 3.41 s |
| Clang 18.1 Release, warnings as errors | 8/8 passed in 3.35 s |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 8/8 passed in 9.53 s |

ASan used leak detection and halt-on-error; UBSan halted on the first finding. The official
exporter reproduction, dependency-free fixture validation, Python byte compilation and
`git diff --check` also passed.

The broader MSVC Release suite passed 10/10 tests in 15.56 seconds with the unrelated large
OLMoE runtime test deliberately excluded after its Safety Governor had already refused new
large reservations under host-memory pressure. The MSVC Debug configuration, which does not
include that full OLMoE runtime, passed 10/10 in 107.76 seconds before the final additional
all-tensor-consumption assertions; the final focused Debug gate above includes those assertions.

## Safety and measured performance

The graph validates all configuration scalars, token IDs, tensor shapes and finite values.
Shape products are overflow checked; the prefill is capped at 4,096 tokens, 16 layers and
64 million total graph elements. Malformed state fails before tensor allocation or indexing.

The 102,104-byte fixture runs below meaningful timing resolution, so no tokens/s, TTFT or
thermal claim is made. It neither predicts nor demonstrates full-model performance.

## Known limitations and next bottleneck

- no full-checkpoint tensor loader or FP4/FP8 dequantization/compute;
- no incremental compressed KV/state path;
- no native MTP block or speculative decoding path;
- no full reasoning/tool-call template and completion parser;
- no CUDA kernels, VRAM tier or pinned host transfers;
- no DeepSeek NVMe expert streaming or full-model probe.

M6 is complete for its reduced float correctness gate. M7 now needs to bind the verified
Safetensors catalog to this model-specific graph, implement MTP and incremental state, and
then run carefully selected full-checkpoint probes under the existing resource interlocks.
