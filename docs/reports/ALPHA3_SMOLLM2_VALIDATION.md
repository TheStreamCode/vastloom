# Alpha 3 SmolLM2 validation

Validation date: 2026-08-09

Implementation commit: `ed777bf94ca59ed7630bab4327437fcb04f5c54e`

## Implemented

- pinned `HuggingFaceTB/SmolLM2-135M-Instruct` adapter for revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`;
- GPT-2 byte-level BPE tokenizer, special tokens and exact SmolLM chat template;
- verified BF16 Safetensors loading through bounded 64 MiB reads;
- CPU RMSNorm, Llama RoPE, grouped-query causal attention, incremental KV cache,
  SwiGLU and tied output projection;
- greedy and deterministic seeded top-k sampling;
- one-shot streaming/JSON generation and interactive multi-turn history;
- real-model `model-info`, `tokenize`, `real-logits` and adapter-aware `self-test`;
- full SHA-256 verification by default, Resource Governor reservations, live host
  headroom checks, six-thread limit, timeout and safe Ctrl+C cancellation.

No answers are hardcoded in the inference path. Expected strings exist only in the
test fixture and self-test oracle.

## Reference identity

- model: `HuggingFaceTB/SmolLM2-135M-Instruct`;
- revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`;
- checkpoint bytes: 269,060,552;
- tensor payload bytes: 269,030,016;
- tensors: 272 BF16 tensors;
- `model.safetensors` SHA-256:
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`;
- trusted reference: Python 3, PyTorch 2.4.1, Transformers 4.42.3,
  Tokenizers 0.19.1 and Safetensors 0.4.5.

The model is stored under `C:\StargateModels`, outside the repository.

## Correctness results

Native token IDs exactly matched the reference for ASCII, accented Italian, special
tokens, individual digit splitting and the complete chat template. Raw and templated
final-token top-logit ordering matched exactly; recorded values differed by at most
`1e-3` in the automated gate.

Greedy native output exactly matched Transformers for:

| Prompt | Expected and actual output |
|---|---|
| `Hello` | `Hello! How can I help you today?` |
| `Qual è la capitale d'Italia?` | `Italia.` |
| `Write one short sentence about the Moon.` | `The Moon, a celestial body, is a mysterious and awe-inspiring sight that captivates our imagination and inspires awe in us.` |

The interactive two-turn case also matched exactly, and two runs with temperature
`0.7`, top-k `20` and seed `42` produced identical token IDs.

The full verified native self-test passed 4/4 cases, including multi-turn history, in
5.381 seconds. The full Windows CTest suite passed 6/6 cases in 13.61 seconds.

| Gate | Exact result |
|---|---:|
| MSVC 19.44 Release, `/WX`, including real model | 6/6 passed |
| GCC 13 Release, `-Werror` | 5/5 passed |
| Clang 18 Release, `-Werror` | 5/5 passed |
| GCC 13 Debug + ASan + UBSan | 5/5 passed |
| ASan + UBSan real-model final-logit probe | passed |
| ThreadSanitizer real-model final-logit probe | passed |
| Python `py_compile` | passed |

The Linux CTest entries omit the external-model suite by design; that suite is enabled
only when `STARGATE_SMOLLM2_MODEL` points to the separately imported checkpoint.

## Short performance observation

On the Windows reference PC, observed Release model load was approximately 200-218 ms.
A 31-token `Hello` chat prefill took approximately 634-643 ms, and its short nine-token
decode ran at approximately 47-48 tokens/s. These are brief correctness observations,
not sustained thermal measurements or large-model performance claims.

## Failure and safety coverage

The tested path rejects a corrupt imported file, incompatible model id/revision,
malformed or incomplete configuration, invalid/duplicate CLI options, oversized
context and failed memory reservations. Model weights are immutable, checkpoint hashes
are verified before loading, and the Windows process cap is established before the
large allocation.

## Known limitations and next bottleneck

- CPU reference execution only;
- one exact dense checkpoint, not a generic Llama adapter;
- BF16 weights expand into FP32 compute and are not optimized kernels;
- history is correct but previous KV state is recomputed on each turn;
- no CUDA, quantization, real MoE routing or NVMe expert paging.

The next bottleneck is a pinned small real MoE: exact tokenizer/config/tensor mapping,
authoritative top-k router/expert execution and route telemetry. Its checkpoint cannot
be loaded as one resident allocation under the current safe RAM cap, so expert tensors
must be addressed and streamed explicitly.
