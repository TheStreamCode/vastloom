# DeepInfer review for Vastloom

Reviewed on 2026-10-20 against the public [DeepInfer repository](https://github.com/deepseek-ai/DeepInfer)
and a local clone at `build/deepinfer-clone` (no `.git`: 342 non-vendor files, 270 Python files /
47,754 lines, 33 C++ files / 5,153 lines under `csrc/`). The vendor submodules are empty in the
clone, so kernel behaviour is visible only through call sites. Nothing was built or executed, and no
GPU or checkpoint was available. Paths below are relative to the clone root; external facts are
taken from the official pages linked under [Sources](#sources).

## Verdict

DeepInfer is a **datacenter scale-out serving engine**, not a local runtime. It is a vLLM-derived
engine ("Derived from vLLM", `deepinfer/engine/llm_engine.py:2`) rebuilt around one model family
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`) and two accelerator stacks, with a C++ scheduler in place of
vLLM's Python one. It is Apache-2.0, Linux-only, and targets **NVIDIA SM100 and Huawei Ascend 950**
exclusively.

The release is unambiguous about its scope: benchmarks on 4 SM100 GPUs for prefill and 64 for
decode, a documented minimum efficient deployment unit of **64 + 64 GPUs** at 155 T tokens/day, a KV
cache on a dedicated 3FS cluster, and expert parallelism tuned for 64–128 ranks. Its target
checkpoint reports **763.2B parameters in ~511 GB on disk** and requires 614 GB of VRAM by the vLLM
recipe. No row of that stack is feasible on the reference PC (12 GiB VRAM, 32 GiB RAM); every
vendored kernel library documents SM90/SM100 or Ascend requirements and none documents a CPU or Ada
fallback.

The **ideas** travel; the **code does not**. Three mechanisms were small, separable and testable on
Vastloom's dense SmolLM2 path, and this project adopted them in `0.5.0-alpha.1`: JSON-object
constrained decoding, prompt control-token rejection and UTF-8-safe streaming. The rest is recorded
below with an explicit adoption status so a future maintainer does not re-open a closed question
from the vendor's framing alone.

No upstream code is copied by this review. Any future adaptation must be independently implemented,
validated and attributed under the upstream Apache-2.0 terms.

## Evidence worth carrying forward

| Mechanism | Evidence in the clone | Portability verdict |
|---|---|---|
| Structured outputs: compiled grammar, token mask per step, compilation off the critical path | `structure_output/so_matcher_builder.py:19-95`, `llm_engine.py:606-630`, `structure_output/mask.py:23-33`, `layers/sampler.py:433-437` | Portable in concept and adopted in P0 as a native JSON-object constraint. The vendor uses `xgrammar`; Vastloom keeps its own bounded state machine and takes no dependency. |
| Batch invariance and deterministic sampling | `environ.py:197`, `worker/runtime_config.py:67-69`, `tilelang_ops/random_sampling.py:207-213`, `README.md:27-28` | The sampling half is directly portable (counter RNG, per-sequence seed). The logits half is achieved by *disabling* fused kernels, which Vastloom must not copy; the transferable part is the invariance contract, now written into the Q11k design. |
| Chunked/batched prefill and one-step-stale metadata | `csrc/engine/scheduler.hpp:372-382`, `worker/model_runner.py:148-230`, `worker/worker.py:96-107` | The scheduling concept maps to Q11k. The device-side metadata repair path does not: it needs CUDA graphs and a device pipeline a CPU/GGUF runtime lacks. |
| Bounded prefill for a windowed decoder half (SWA replay / YOCO truncation) | `models/deepseek.py:190-208`, `input_metadata/yoco.py:86-182`, `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:149-168` | Architectural, but model-specific: it needs a persistent prefix state and a sliding-window decoder. Deferred (P1) behind Q11k/Q12; Qwen3.8's DeltaNet + QSA does not expose the same 128-token window. |
| Min-peak redundant-expert placement (EPLB policy) | `model_executor/eplb/policy.py:12-70,99-123` | The greedy objective is separable: "rank" becomes "cache slot", "replica" becomes "pinned copy". Blocked (P1) behind Q11n and the measured CUDA-tier negative; the weight-shuffle machinery is not portable. |
| Confidence-scheduled speculative verification (DSpark) | `models/mtp/dflash.py:46-79`, `csrc/engine/mtp_varlen.hpp:150-267`, `layers/sampler.py:411-457` | Checkpoint-bound: a learned confidence head predicts per-draft acceptance. Vastloom measured speculation at 0.78× on its CPU single-stream track and has no Qwen DSpark drafter. Not adopted. |
| Paged KV and compressed-attention state | `worker/cache_engine.py:540-578,918-966`, `v4_modules/cache_common.py:109-161` | FP4/MXFP8 layouts and 128B alignment are datacenter-specific. The flat slab with carved per-layer views and a per-sequence SWA ring is the transferable part, and it is exactly the shape Q11k/Q12 need. |
| Async "zero-overhead" scheduling (ZOS) | `worker/worker.py:96-107,113-160`, `worker/model_runner.py:321-360` | Device-metadata locked: the CPU is deliberately one step stale and the device repairs lengths. A single-request CPU+NVMe runtime has nothing to overlap at this layer; only the bounded-prefetch principle transfers, and partial overlap already exists. |
| C++ scheduler and deferred metadata ownership | `csrc/engine/scheduler.hpp:19-22,275-341`, `llm_engine.py:67-119` | POSIX-only headers, torch-extension ABI, Python 3.12. The concept (compiled scheduler, pooled metadata, one step of slack) is portable; the code is not reachable from Windows/MSVC. |
| Test and benchmark discipline | `tests/conftest.py:12-26`, `tests/models/test_sampler.py:1-10`, `examples/README.md:35-45`, `doc/inference_deployment_qa.md:249-256` | Portable and valuable: golden baselines, statistical sampler gates, decoupled prefill/decode windows, and negative results stated as negative results. Adopted as requirements for the Q11k invariance harness. |

## Vastloom adoption order

The ordering is by cost and evidence, not by vendor importance: a mechanism is adopted only where it
is bounded, testable without a large checkpoint, and does not change model arithmetic.

### P0 — implemented in `0.5.0-alpha.1` (dense SmolLM2 path)

1. **JSON-object constrained decoding.** `response_format: {"type": "json_object"}` on the serve path
   and `chat --json-object` on the dense CLI apply a native token constraint: exactly one RFC 8259
   JSON object as the root value, bounded depth (8), string bytes (1024) and number bytes (64). The
   constraint masks the sampling distribution per step and never touches the model's arithmetic.
   Root object only: JSON Schema and GBNF compilation are separate future constraints, and an
   unsupported `response_format` is refused with HTTP 400 rather than served as text.
2. **Prompt control-token rejection.** Chat message content that encodes a tokenizer control token
   is refused fail-closed (HTTP 400 on serve) instead of being silently reinterpreted by the chat
   template. The refusal deliberately does not echo the content or the token.
3. **UTF-8-safe streaming.** Token boundaries can split a multi-byte UTF-8 sequence; the SSE
   emitter buffers a well-formed trailing prefix and releases it with the next chunk, so no frame
   ever ends inside a code point.

The same update fixes a streaming producer lifetime defect: the SSE handler captured request-scoped
state by reference and the transport invoked it after the handler had returned. Request state now
lives in a shared block owned by the producer, which is also what makes a stateful decoder
constraint safe to attach to the streaming path.

### P1 — blocked (design or policy ready, execution gated)

- **KV cache compression to F16.** Qwen3.8's 12 QSA layers add about 72 KB/token of F32 caches
  (~9.4 GB at 128k); halving them is a roadmap hypothesis, not a measurement. It needs the real
  checkpoint and a quality harness, and it interacts with the Q11k chunk-boundary state contract.
  Source: `worker/cache_engine.py:540-578` for the ring/slab discipline.
- **Routing-aware expert placement.** Reuse the min-peak objective for a consumer VRAM/RAM expert
  cache (roadmap Q11n). The falsifier is beating the OS page cache at equal RAM; the measured
  negative of the `--cuda-experts` tier must be answered first.
- **Q11k chunked prefill.** The design is written in
  [`docs/planning/Q11K_CHUNKED_PREFILL_DESIGN.md`](../planning/Q11K_CHUNKED_PREFILL_DESIGN.md) and
  draws on DeepInfer's prefill/decode separation and batch metadata. It is blocked by the entry
  criteria recorded in the roadmap: the large checkpoint must be locally available again and the
  llama.cpp oracle parity battery must exist.

### P2 — deferred

- **Tokenizer and API protocol patterns from `deepseek-recipe`.** The V4.1 model card documents no
  Jinja chat template; chat framing is carried in `encoding/encoding.py` and in a Rust toolkit. This
  is relevant only when the Q13 native text tokenizer and ChatML framing start, and it is not needed
  by the dense path.

### Not adopted

- **Vendor kernels.** FlashMLA, DeepGEMM, DeepEP, DeepSelect and TileKernels are MIT but require
  Hopper/Blackwell SM90/SM100, CUDA 13 and NVLink/RDMA (or Ascend 950 + CANN 9.2). None documents a
  CPU path; DeepSelect does not state a minimum architecture, which is not evidence of Ada support.
- **Expert-parallel replication machinery.** EPLB weight shuffles, NCCL/HCCL groups, NVLink
  symmetric-memory exchanges and 64–128-rank all-to-all have no meaning on one device.
- **Disaggregated serving.** Prefill/decode disaggregation and the 3FS KV store are declared out of
  scope by the upstream release itself (hooks are empty virtuals, `llm_engine.py:801-826`); there is
  no consumer analogue of a 100 PiB KV tier.
- **SWA/YOCO bounded replay.** Model-specific and dependent on a persistent prefix state; deferred
  until Q11k/Q12 exist, and the exact scheme does not apply to Qwen3.8.
- **Speculative decoding with a confidence head.** Closed by measurement on Vastloom's historical
  CPU track (0.78× the plain baseline) and checkpoint-bound on the draft side. Revisit only after a
  batched multi-token forward exists and an independent drafter is available.
- **Async device-metadata scheduling and the C++ scheduler.** No device metadata pipeline, no CUDA
  graphs, no batch-serving engine: the hardware floor, not the idea, is the blocker.

## What Vastloom should not copy

- A second inference engine or a serving framework. DeepInfer exists to serve one model family on
  datacenter accelerators; duplicating it would abandon the governed single-machine thesis.
- The hardware floor. Anything whose prerequisites are SM90/SM100, 614 GB VRAM or 64+ GPUs is a
  design reference only; it must never be downloaded, built or advertised as a consumer path.
- The "batch invariance by disabling optimizations" trade. Determinism bought by removing fused
  kernels is not available to a runtime whose scalar reference is the parity instrument. State the
  invariance contract and test it; do not adopt the mechanism.
- Vendor benchmark framing. 155 T tokens/day, `moe_lb` 1.07–1.09 and 4-GPU prefill numbers are
  datacenter measurements; they are not comparable to single-machine CPU+NVMe results and must not
  be cited as such.
- Production-serving assumptions: 3FS KV storage, 7-day TTLs and GPU-hour economics have no
  consumer counterpart.
- Acceptance heuristics that need a model artefact Vastloom does not have. The DSpark confidence
  head lives in the checkpoint; a heuristic fitted to a different model is not that mechanism.

## Limits of this review

- `vendor/` is empty in the clone: kernel-level statements are call-site statements, not kernel
  statements, and the vendor documentation was read from the upstream repositories instead.
- Nothing ran. No build, no GPU, no checkpoint. Vendor benchmark numbers are vendor claims.
- DeepInfer's exact vLLM fork base is undocumented: no `vllm` dependency, no import, no version
  pin. The relationship is lineage acknowledged by the README, not a runtime dependency.
- Model size and storage do not reconcile by simple byte arithmetic because experts are FP4-packed
  and ~23.6B UE8M0 block scales are stored separately; both figures are API-reported.
- The V4.1 technical report PDF could not be fetched; architecture figures come from the model
  card, `config.json` and the vLLM recipe.
- DeepInfer documents no maximum sequence length, memory footprint or per-GPU memory plan for its
  own engine; the 1,048,576 positions belong to the model, not to a verified serving limit.

## Sources

- DeepInfer: [repository](https://github.com/deepseek-ai/DeepInfer) ·
  [deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) ·
  [GitHub API](https://api.github.com/repos/deepseek-ai/DeepInfer)
- Target model: [DeepSeek-V4.1-Flash-1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) ·
  [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) ·
  [HF API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?blobs=true) ·
  [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- Kernels: [FlashMLA](https://github.com/deepseek-ai/FlashMLA) ·
  [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [DeepEP](https://github.com/deepseek-ai/DeepEP) ·
  [DeepSelect](https://github.com/deepseek-ai/DeepSelect) ·
  [TileKernels](https://github.com/deepseek-ai/TileKernels)
- Research and public algorithms: [DSpark, arXiv:2607.05147](https://arxiv.org/abs/2607.05147) ·
  [DeepSpec](https://github.com/deepseek-ai/DeepSpec) · [EPLB](https://github.com/deepseek-ai/EPLB) ·
  [vLLM SWA bounded replay, PR #59039](https://github.com/vllm-project/vllm/pull/59039)
- Consumer references for the deferred mechanisms: [llama.cpp GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) ·
  [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) ·
  [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) ·
  [SGLang paged experts](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/paged_experts.mdx) ·
  [KTransformers](https://github.com/kvcache-ai/ktransformers)
