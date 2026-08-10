# M7 DeepSeek V4 Flash MTP reference validation

Validation date: 2026-08-10

Implementation commit: `043854f096bcc0880601c7de07c26ac02606f60d`

## Scope

This report closes the reduced, checkpoint-shaped MTP component gate for the official
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
repository at immutable revision
`60d8d70770c6776ff598c94bb586a859a38244f1`:

```text
pinned official model.py
  -> deterministic two-layer main prefill
  -> one official MTP block
  -> main + MTP logits, tokens, routes and sparse-attention indices
  -> bounded native FP32 parity runner
```

This validates the MTP equations and native graph composition on a deliberately tiny fixture.
It is not speculative decoding, full-checkpoint execution or a performance result. The
159,617,149,040-byte checkpoint payload was not downloaded or read.

## Trusted reference and fixture

The exporter verifies the official `inference/model.py` SHA-256
`ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71` before import. It uses
deterministically initialized FP32 official modules and equation-equivalent CPU replacements
for the two TileLang entry points. Quantized GEMM is forbidden in this oracle.

The committed fixture is 147,136 bytes and contains:

| Property | Fixture value |
|---|---:|
| Input tokens | 8 |
| Vocabulary / width | 16 / 8 |
| Main / MTP layers | 2 / 1 |
| Hyper-Connection multiplier | 2 |
| Routed experts / top-k / shared experts | 4 / 2 / 1 |
| Main routing | one hash layer + one learned layer |
| MTP routing | one learned layer |
| Fixture tensors | 123 |
| FP32 / integer elements | 4,750 / 32 |

`tools/export_deepseek_v4_fixture.py --check` reproduced the committed JSON byte for byte.
The dependency-free `--validate-only` path independently checked its source identity, structure
and expected-vector dimensions.

## Native MTP path

After the authoritative two-layer main prefill, the reference runtime now executes the official
MTP ordering:

```text
token embedding -> MTP embedding RMSNorm -> embedding projection
main hidden -> MTP hidden RMSNorm -> hidden projection
projected embedding broadcast-add
  -> Hyper-Connection attention pre/post
  -> MTP sparse selected-latent attention
  -> Hyper-Connection FFN pre/post
  -> learned top-k routed experts + shared expert
  -> MTP HC head -> MTP RMSNorm -> shared LM head
```

The graph helpers were generalized around an explicit tensor prefix, compression ratio and
router mode. This keeps the main and MTP blocks on one checked implementation without treating
them as interchangeable checkpoint layouts.

## Exact parity gate

Against the pinned official exporter, the native result passed:

- all 16 main logits and all 16 MTP logits within absolute tolerance `8e-5`;
- exact main greedy token ID `4` and exact MTP proposed token ID `11`;
- all 32 main and 16 MTP routed expert IDs exactly;
- all 32 main and 16 MTP route weights within `8e-5`;
- all 64 main and 24 MTP sparse-attention indices exactly.

The test removes each of the 123 declared tensors in turn and requires the forward to fail. It
also retains missing-head, non-finite tensor, out-of-vocabulary token and overflowing-shape
rejection coverage.

## Build and test matrix

| Toolchain/configuration | Exact result |
|---|---:|
| MSVC 19.44 Release, warnings as errors | 11/11 passed in 79.76 s |
| MSVC 19.44 Debug, warnings as errors | 10/10 passed in 110.10 s |
| GCC 13.3 Release, warnings as errors | 8/8 passed in 3.48 s |
| Clang 18.1 Release, warnings as errors | 8/8 passed in 3.41 s |
| GCC 13.3 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 8/8 passed in 10.09 s |

ASan used leak detection and halt-on-error; UBSan halted on the first finding. Python byte
compilation, the offline importer, manifest tests, official fixture reproduction and
dependency-free fixture validation also passed. The manifest suite reported one pass and one
Windows symlink-capability skip; the skip is environmental and unrelated to MTP.

The Windows validation host exposed a Ryzen 9 7900X-class 24-thread CPU, 32 GB RAM, RTX 4070
Super with 12 GB VRAM and approximately 1.20 TB free on the project volume at the recorded
probe. The MTP test itself remained CPU-only and did not allocate CUDA memory.

## Safety and performance boundary

The reference graph rejects invalid configuration, unsupported compression ratios, non-finite
values, out-of-range tokens, shape overflow and tensor stores above 64 million elements. It is
capped at 4,096 tokens, 16 total main/MTP layers and one MTP layer. The fixture is only about
144 KiB, so no large-model reservation or storage traffic was required.

CTest wall times are reported only as validation evidence. They include unrelated integration
tests and are not token-throughput measurements. No tokens/s, TTFT, NVMe bytes/token, thermals
or full-model memory claim is made.

## Known limitations and next bottleneck

- no full-checkpoint tensor binding or packed FP4/FP8 compute;
- no incremental compressed KV/state path;
- no speculative target-verification, acceptance or rejection loop;
- no full reasoning/tool-call completion parser;
- no CUDA kernels, VRAM tier or pinned host transfers;
- no DeepSeek generation or full-checkpoint probe.

The next correctness bottleneck is incremental state parity for raw and compressed attention.
After that gate, the verified Safetensors catalog can be bound to packed checkpoint storage under
the existing resource and large-download interlocks.
