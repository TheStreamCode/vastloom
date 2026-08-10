# M9 DeepSeek real selected-expert cache validation

Validation date: 2026-08-10
Code commit: `1a33937b24da4fea8e28f959ee5d972253a6f416`

## Scope

This milestone connects one complete extracted DeepSeek V4 Flash 0731 expert to the production
router-first cache and mixed-format CPU consumer:

```text
executable logical GGUF catalog
  -> canonical layer/expert tensor identity
  -> strict W1 gate / W2 down / W3 up binding
  -> bounded one-entry complete-bundle RAM cache
       miss -> Governor reservation -> three bounded immutable reads -> RAM_READY
       hit  -> retain the same immutable payload -> zero logical bytes read
  -> IQ1_S gate/up -> FP32 SiLU multiply -> IQ3_XXS down
  -> transactional FP32 output
  -> lease release -> cache clear -> zero Governor reservations
```

The three source ranges are separately size- and SHA-256-pinned extractions from the reviewed
Unsloth `UD-IQ1_S` checkpoint. The complete 82.54 GB distribution was not downloaded. This proves
one real selected expert at the cache/kernel boundary; it does not prove a complete official
layer, routed token, generation loop, CUDA path or SSD throughput.

## Implemented vertical slice

- `deepseek_v4_make_gguf_expert_bindings` accepts only an executable, revision-identified catalog
  and the exact requested layer/expert count.
- Canonical names map `ffn_gate_exps`, `ffn_down_exps` and `ffn_up_exps` to W1/W2/W3. Tensor role,
  layer, expert, speculative flag, shape, GGML type, encoded size, bundle slice and contiguous
  expert identity are checked before a binding is published.
- Metadata-only catalogs, incomplete layers, duplicate/missing roles, malformed names, truncated
  shapes, invalid byte counts and caller limits fail closed.
- `deepseek_v4_execute_resolved_iq1_s_iq3_xxs_f32` accepts only IQ1_S W1/W3 and IQ3_XXS W2 with
  exact unpadded 4096/2048 dimensions. It performs no routing, allocation or payload copy.
- Resident matrix views are borrowed from descriptors whose shared lease keeps the complete
  immutable packed buffer alive. The public output remains untouched when format validation fails.

## Deterministic result

The layer-0/expert-0 bundle contains:

| Matrix | Format | Shape | Encoded bytes |
|---|---|---:|---:|
| gate / W1 | IQ1_S | 2,048 x 4,096 | 1,638,400 |
| up / W3 | IQ1_S | 2,048 x 4,096 | 1,638,400 |
| down / W2 | IQ3_XXS | 4,096 x 2,048 | 3,211,264 |
| **Complete expert** | mixed | indivisible bundle | **6,488,064** |

The first cache resolution reports one miss, 6,488,064 resident bytes and 6,488,064 logical bytes
read. The FNV-1a fingerprint over all 4,096 FP32 outputs matches the pinned ggml oracle at
`90388543cbcbab52`; the first value is `-0.0709934607` and the last is `0.105151489`.
The second resolution reports one hit and zero additional logical bytes. Cache clearing is rejected
while either resolved result retains the expert, succeeds after release, and returns RAM, staging
and scratch reservations to zero.

## Safety and fault coverage

- a one-byte RAM shortfall rejects the expert before allocation or worker I/O;
- the destination is capped at 8 MiB, staging at 4 MiB and aggregate host memory at 10 MiB;
- the cache holds exactly one complete expert and never splits it across RAM and VRAM;
- all three extracted source files are regular files with exact size and SHA-256 before use;
- malformed W2 format is rejected before the caller's NaN-filled output changes;
- in-flight leases prevent eviction and explicit clearing;
- every cache, staging and scratch reservation returns to zero after release;
- no model file is modified and no large model is downloaded.

## Validation host

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32,624,488 KiB visible Windows RAM; 11,279,476 KiB free at the recorded probe;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported VRAM and 8,672 MiB free at the probe,
  unused by this CPU test;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022 17.14.36/MSVC;
- no CUDA code was compiled or executed.

## Exact results

| Toolchain/configuration | Result |
|---|---:|
| GCC Release, warnings as errors | 10/10 passed in 10.18 s |
| GCC Debug, warnings as errors | 10/10 passed in 23.93 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 10/10 passed in 61.88 s |
| GCC ThreadSanitizer native suite, per-process ASLR disabled | passed |
| Clang 18 Release, warnings as errors | 9/9 passed in 3.72 s |
| MSVC Release, warnings as errors | 9/9 passed in 3.63 s |
| MSVC Debug, warnings as errors | 9/9 passed in 6.84 s |
| Real-expert opt-in: GCC Release | 1/1 passed |
| Real-expert opt-in: GCC ASan + UBSan | 1/1 passed |
| Real-expert opt-in: Clang Release | 1/1 passed |
| Real-expert opt-in: MSVC Release | 1/1 passed |
| Real-expert opt-in: MSVC Debug | 1/1 passed |
| Serial real-model regression subset | 5/5 passed in 209.61 s |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python offline importer and `compileall` | passed |

The serial regression used already-present verified local artifacts: SmolLM2 passed in 31.91
seconds, OLMoE inspection in 1.29 seconds, OLMoE CPU runtime in 169.51 seconds, DeepSeek metadata
inspection in 6.27 seconds and the synthetic full-checkpoint binding gate in 0.58 seconds. TSan
used `setarch x86_64 -R` only for the instrumented process; no system-wide ASLR setting changed.

## Performance statement

No token rate, TTFT, SSD bandwidth, physical bytes/token or useful overlap is reported. The source
ranges were already present locally, cache state outside the explicit first/second resolution was
not controlled as a device benchmark, and the mixed-format scalar kernel is a correctness oracle.
The 6,488,064-byte count is exact logical payload traffic on the miss, not measured physical NVMe
traffic.

## Known limitations

- The test executes one extracted expert, not a fully imported immutable checkpoint shard.
- No official trunk tensor is combined with the expert, so no complete layer/token runs.
- Acquisition is synchronous at the router-first cache boundary; the fixed-worker pool is tested
  separately but is not compute-overlapped here.
- The cache tier is RAM only; VRAM placement, pinned staging and CUDA kernels are absent.
- The kernels are scalar references, not SIMD/fused production kernels.
- Multi-token prefill chunking, target-verified MTP generation and completion parsing are absent.

## Next bottleneck

Connect the existing bounded fixed-worker loader to canonical selected-expert resolution without
changing router choice or reduction order. Then compose one complete mixed-format layer/token from
bounded trunk tiles and routed experts, requiring exact logit/route parity and timestamped logical
and physical I/O evidence before any throughput claim or full-checkpoint download.
