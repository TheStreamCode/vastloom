# Alpha 4.2 bounded asynchronous expert-I/O validation

Validation date: 2026-08-09

Implementation commit: `d15b26a98437038d9a37d1214064149a8ddb67c3`

## Scope

Alpha 4.2 connects the already validated OLMoE CPU oracle to Stargate's bounded I/O
pool. It adds an opt-in asynchronous path without changing router or expert math:

```text
validated (layer, expert) identity
        -> exact contiguous 12 MiB shard range
        -> Resource Governor staging reservation
        -> one bounded worker, depth 1 or 2
        -> native gate/up/down BF16 layout
        -> authoritative CPU expert compute
```

Synchronous reads remain the default. This milestone validates a buffered file-I/O
pipeline; it is not direct I/O, a physical NVMe traffic measurement or a CUDA path.

## Implemented

- `sync` and opt-in `async` expert-I/O modes in `moe-logits`;
- one asynchronous worker and an adapter limit of one or two outstanding prefetches;
- one 12 MiB request per routed expert after proving that its gate, up and down tensors
  form one contiguous bounded range in one validated shard;
- Resource Governor staging reservation before every request;
- per-request timeout, forward cancellation before and after reads and bounded shutdown;
- deterministic worker failure, short read, allocation failure, cancellation and timeout
  injection on a selected request sequence;
- queue, worker-service and total request timing;
- foreground wait and exact intersection between I/O-worker service intervals and CPU
  expert-compute intervals;
- twelve-field atomic trace output, while the simulator retains four- and nine-field
  compatibility;
- cache-aware scheduling that does not submit a prefetch for a resident expert.

The fault-injection CLI switch is a development test surface. It is rejected unless
asynchronous mode is selected.

## Safety envelope

The Windows host probe reported:

- Windows 11 on AMD Ryzen 9 7900X, 24 logical CPUs;
- 33,407,475,712 bytes physical RAM and 13,956,579,328 bytes available at the recorded
  probe;
- RTX 4070 Super with 12,878,610,432 bytes VRAM, not used by this CPU path;
- 1,207,959,883,776 bytes available beneath `C:\StargateModels`.

The runtime policy still retains eight GiB of current RAM availability for the host,
caps inference at six CPU threads and applies the Windows process-memory Job Object.
For the one-token no-cache depth-two probe, explicit reservations comprise:

| Reservation | Bytes |
|---|---:|
| Always-active OLMoE trunk | 953,421,824 |
| One active expert | 12,582,912 |
| Two prefetch staging buffers | 25,165,824 |
| One-token FP32 K/V cache | 262,144 |
| Activation scratch | 67,108,864 |
| **Total explicit reservation** | **1,058,541,568** |

The model shards are opened read-only. No checkpoint or derived artifact is rewritten.
Each future owns its staging reservation until the buffer is consumed or the operation
fails. A failure stops the forward, drains or cancels the bounded queue during pool
destruction and releases all reservations.

## Correctness and failure results

Against the committed Transformers 4.45.2 fixture for raw `Hello`:

- greedy token remained exactly `47111` (` Byr`);
- all registered top logits remained within the `2e-4` absolute tolerance;
- all 128 selected experts and routing weights remained within the existing route gate;
- async no-cache reported exactly 128 prefetches, 128 misses and 1,610,612,736 logical
  expert bytes;
- a two-token async LRU run submitted only 128 prefetches, then served all 128 repeated
  experts from RAM without new reads;
- the atomic trace contained 128 valid twelve-field rows and remained consumable by all
  four cache-policy simulations.

The real-checkpoint integration test separately injected and rejected:

| Fault | Required fail-closed result |
|---|---|
| Worker I/O failure | no expert compute after the failed request |
| Short read | wrong byte count rejected |
| Expert allocation failure | bounded `resource_exhausted` error |
| Cancellation | bounded cancellation error |
| Request timeout | bounded timeout error |

The pre-existing full-hash corruption and missing-shard tests also remain green.

## Build and test matrix

| Gate | Exact result |
|---|---:|
| MSVC 19.44 Release, `/WX`, complete external SmolLM2 and OLMoE | 8/8 passed |
| MSVC 19.44 Debug, `/WX`, external SmolLM2 and metadata OLMoE | 7/7 passed |
| GCC Release, `-Werror` | 5/5 passed |
| Clang Release, `-Werror` | 5/5 passed |
| GCC Debug, `-Werror` | 5/5 passed |
| GCC Debug + ASan + UBSan, leak/halt enabled | 5/5 passed |
| GCC Debug + libstdc++ debug iterators/assertions | 5/5 passed |
| Python `py_compile` for all tools and tests | passed |

The final Windows Release suite passed in 72.14 seconds; its complete-checkpoint OLMoE
test passed in 56.99 seconds. The final Windows Debug suite passed in 111.71 seconds.

ThreadSanitizer compiled successfully. Every instrumented native executable was stopped
by this WSL runtime before `main` with `ThreadSanitizer: unexpected memory mapping`.
TSan is therefore unavailable in this environment, not passing and not reporting a
source-code race.

## Buffered warm-cache pipeline measurements

Five interleaved runs per mode used MSVC Release, raw `Hello`, no-cache, a 12 MiB active
expert budget and the same 1,610,612,736 logical bytes. Every run produced greedy token
`47111`.

| Mode | Forward mean (min-max) | I/O service mean | Foreground wait mean | Actual service/compute overlap mean |
|---|---:|---:|---:|---:|
| sync | 1,178.210 ms (1,157.99-1,225.23) | 988.029 ms | 0 ms | 0 ms |
| async depth 1 | 1,348.468 ms (1,317.37-1,371.25) | 1,034.074 ms | 892.172 ms | 107.310 ms |
| async depth 2 | 1,155.778 ms (1,115.55-1,202.99) | 1,052.448 ms | 691.867 ms | 104.777 ms |

Depth two reduced mean forward time by 1.9% relative to synchronous in this small warm
probe. Depth one regressed by 14.5%. The result supports retaining a bounded depth-two
experiment, but it is not evidence of an SSD throughput improvement.

The files had already been verified and repeatedly read, Windows buffered I/O was used,
and no operating-system or device counter was captured. `expert_bytes_read` is therefore
the exact logical requested payload, not physical NVMe bytes. No tokens/second result is
reported because `moe-logits` performs a final-logit correctness probe, not decoding.

## Known limitations

- ordinary buffered reads; no unbuffered/direct I/O or pinned host memory;
- one worker only and fixed prefetch depth, not an adaptive scheduler;
- no physical-device byte, queue-depth, bandwidth, temperature or power counters;
- no repeatable cold-cache protocol yet;
- no reuse-distance or wasted-prefetch reporting;
- CPU expert kernels remain synchronous FP32-reference compute over BF16 weights;
- no OLMoE generation loop, CUDA/VRAM expert tier or DeepSeek adapter;
- the complete external OLMoE path was validated on Windows, not under Linux sanitizers.

## Next bottleneck

Add reuse-distance telemetry and trustworthy physical-device I/O counters, then define a
repeatable cold-cache benchmark that cannot be confused with Windows file-cache speed.
After that measurement gate, proceed to the exact DeepSeek V4 Flash checkpoint inspector
and architecture-faithful fixture while preserving the same reservation and failure
boundaries.
