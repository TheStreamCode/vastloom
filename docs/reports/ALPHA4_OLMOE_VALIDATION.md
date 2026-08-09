# Alpha 4 OLMoE streaming validation

Validation date: 2026-08-09

Implementation commit: `03919d85394cbbaf90e28c9c2c58fbde71215729`

## Implemented

- exact adapter for `allenai/OLMoE-1B-7B-0924-Instruct` revision
  `7f1c97f440f06ce36705e4f2b843edb5925f4498`;
- bounded, resumable and atomic import of three immutable BF16 weight shards;
- strict config, tokenizer, chat-template, tensor-name, dtype, shape, shard and offset
  validation;
- CPU RMSNorm, Q/K normalization, RoPE, causal attention, 64-way router softmax,
  authoritative top-8 selection and routing-weighted SwiGLU expert execution;
- a 953,421,824-byte resident always-active trunk plus exact synchronous reads of one
  12 MiB expert at a time from NVMe;
- no-cache, LRU, LFU and static-hot RAM-cache implementations with Resource Governor
  reservations and a two-GiB ceiling;
- atomic telemetry containing token, layer, expert, expert size, routing weight, tier,
  hit/miss, bytes read and I/O latency;
- `moe-inspect`, `moe-logits` and native-telemetry cache simulation;
- an independent Transformers oracle that keeps a four-GiB Windows process cap, reads
  bounded tensor ranges and materializes only one layer/one expert at a time.

No model answers, expert choices or logits are hardcoded in the runtime. Expected values
exist only in the committed reference fixture.

## Checkpoint identity

- architecture: `OlmoeForCausalLM`;
- layers/width/heads: 16 / 2,048 / 16;
- experts: 64 per layer, top-8 per token;
- expert intermediate width: 1,024;
- context: 4,096;
- vocabulary: 50,304;
- tensors: 3,219 BF16 tensors;
- tensor payload: 13,838,323,712 bytes;
- routed-expert payload: 12,884,901,888 bytes;
- always-active payload: 953,421,824 bytes;
- one expert: 12,582,912 bytes;
- expert traffic without cache: 1,610,612,736 bytes per token.

Pinned shard identities:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 4,997,744,872 | `6c79ac0487c3f23e8ee3d38752197d1e4a1a39d6c1438ec5fd7862874bb19321` |
| `model-00002-of-00003.safetensors` | 4,997,235,176 | `b58ca7a9f28f35d76b56b0ec40fc46c356fd84ccf078083b251a3ad6a2da9a35` |
| `model-00003-of-00003.safetensors` | 3,843,741,912 | `e93653ed509223e63eabae6f72ac8ea5a41115c5c7785574b9b2068ca0961c45` |

The checkpoint is stored read-only beneath `C:\StargateModels`, outside Git. A final
native inspection rehashed all files and reported `hashes_verified=true`.

Official references used for the oracle and tensor contract are the
[pinned Hugging Face checkpoint](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct/tree/7f1c97f440f06ce36705e4f2b843edb5925f4498),
[Transformers 4.45.2 OLMoE implementation](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/olmoe/modeling_olmoe.py)
and the [AllenAI OLMoE repository](https://github.com/allenai/OLMoE).

## Correctness results

The trusted oracle used Python 3.12, PyTorch 2.4.1 CPU and Transformers 4.45.2 with
FP32 activations derived from the pinned BF16 weights. It used official OLMoE operator
components but an independent bounded range reader so a five-GB shard was never mapped
through the four-GiB process cap.

For raw `Hello` (`[12092]`) and `HelloHello` (`[12092, 12092]`):

- native greedy token exactly matched Transformers: token `47111`, text ` Byr`;
- all ten registered top-logit token IDs matched and logit error remained below the
  committed absolute tolerance of `2e-4`;
- all 128 one-token and 256 two-token route events selected the same expert IDs;
- every recorded routing weight matched within `2e-5`;
- the one-token no-cache probe reported exactly 128 misses and 1,610,612,736 bytes;
- the repeated two-token LRU probe reported exactly 128 misses followed by 128 hits,
  reading 1,610,612,736 bytes instead of the 3,221,225,472-byte no-cache baseline.

The duplicate-token case is deliberately simple: it is a deterministic cache-reuse and
incremental-attention fixture, not a representative language-quality prompt.

## Build and test matrix

| Gate | Exact result |
|---|---:|
| MSVC 19.44 Release, `/WX`, external SmolLM2 + metadata OLMoE + full OLMoE | 8/8 passed |
| MSVC 19.44 Debug, `/WX`, external SmolLM2 + metadata OLMoE | 7/7 passed |
| GCC Release, `-Werror` | 5/5 passed |
| Clang Release, `-Werror` | 5/5 passed |
| GCC Debug, `-Werror` | 5/5 passed |
| GCC Debug + ASan + UBSan, halt/leak detection | 5/5 passed |
| GCC Debug + libstdc++ debug iterators/assertions | 5/5 passed |
| Python `py_compile` for tools and integration tests | passed |

The full Windows test verifies all checkpoint hashes, runs the one-token no-cache and
two-token LRU probes, compares logits/routes/weights, validates all four trace policies,
forces a one-millisecond timeout and rejects a missing shard. It passed in 46.50 seconds;
the complete eight-test CTest run passed in 60.38 seconds.

ThreadSanitizer compiled successfully, but this WSL instance terminated every
instrumented native executable before `main` with `ThreadSanitizer: unexpected memory
mapping`. This is reported as unavailable, not passing and not a source-code finding.

## Measured short-run performance and safety

On the Windows 11 reference PC, the final two-token LRU probe measured:

| Metric | Result |
|---|---:|
| Process wall time | 2,054.96 ms |
| Model trunk load | 602.68 ms |
| Native forward | 1,311.79 ms |
| Expert I/O time | 1,016.57 ms |
| Expert cache hits / misses | 128 / 128 |
| Logical expert file bytes | 1,610,612,736 |
| Logical expert bytes per input token | 805,306,368 |
| Peak working set | 2,584,252,416 bytes |
| Peak private bytes | 2,585,677,824 bytes |
| Process CPU time | 3.34 s |

The immediately preceding full inspection calculated a live Stargate RAM cap of
5,070,151,680 bytes while retaining eight GiB for the host. The complete checkpoint was
therefore marked non-resident and the streaming plan accepted. Runtime cache/trunk/KV/
scratch allocations remained below that cap, CPU execution was limited to six threads,
and original checkpoint files remained read-only.

These are short correctness probes, not sustained decode throughput or thermal results.
The runtime currently returns final logits; it does not yet generate OLMoE output tokens,
so no decode tokens/second claim is made. Reads use the operating-system buffered file
path, and the checkpoint had just been hashed; logical file bytes are exact, but physical
SSD bytes and cold-device bandwidth were not measured.

## Known limitations

- CPU synchronous reference kernels only; no CUDA/VRAM expert tier;
- no asynchronous/direct I/O, pinned host memory or read/compute overlap;
- no OLMoE generation loop, persistent chat state or KV reuse across commands;
- one exact OLMoE revision, not a generic MoE adapter;
- BF16 weights with FP32 activations and no production quantized kernels;
- the runtime static-hot baseline pins a deterministic static set; trace simulation can
  rank an offline hot set, but learned online placement is not implemented;
- missing-shard and timeout failures are covered, while injected mid-read failure,
  allocation failure and I/O timeout still require dedicated hooks;
- the external 13.84 GB runtime probe was not executed under Linux sanitizers.

## Next bottleneck

Finish storage fault injection and cancellation at read boundaries, then replace the
synchronous expert reads with bounded asynchronous buffers and measure actual I/O/
compute overlap. Only after that gate should development move to the exact DeepSeek V4
Flash inspector and architecture-faithful fixture. CUDA remains necessary for practical
interactive performance.
