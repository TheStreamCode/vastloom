# M9 DeepSeek mixed-format layer validation

Validation date: 2026-08-10
Implementation commit: `42c9590bad814b390bf94f5a1d1b9e9886d443d9`

## Scope

This milestone closes the reduced-fixture composition gap between the bounded DeepSeek trunk path
and the mixed-format selected-expert path:

```text
deterministic reduced DeepSeek main layer
  -> Q8_0 attention / Hyper-Connection / router / shared-expert matrices
  -> I32 hash-routing table
  -> authoritative top-2 selected routed experts
       W1 gate: IQ1_S
       W3 up:   IQ1_S
       W2 down: IQ3_XXS
  -> bounded file-backed trunk tiles + complete expert bundles
  -> synchronous and asynchronous governed RAM cache paths
  -> complete prefill and incremental layer/token execution
  -> logits, routes, attention indices and state versus a dequantized FP32 oracle
```

The graph dispatch, loader, cache and kernels already supported these formats independently. The
new gate proves that they compose in one architecture-level forward path. It intentionally uses a
small deterministic fixture so every transition and failure can be exercised quickly across
compilers and sanitizers.

This is not an official-payload DeepSeek result. The fixture uses reduced dimensions and generated
GGML-compatible bytes. The separately pinned real 6,488,064-byte expert remains validated at the
same cache/kernel boundary, but it is not inserted into this reduced graph because its real
4,096/2,048 dimensions do not match the fixture. The 82.54 GB checkpoint was not downloaded.

## Mixed-format fixture

The first reduced main layer consumes 23 packed matrices plus one I32 table:

| Role | Format | Matrix count | Encoded bytes |
|---|---|---:|---:|
| attention, Hyper-Connection, router and shared expert | Q8_0 | 11 | 2,516 |
| routed W1/W3 | IQ1_S | 8 | 2,400 |
| routed W2 | IQ3_XXS | 4 | 3,136 |
| hash route table | I32 | 1 tensor | 128 |
| **Complete fixture file** | mixed | 23 matrices + table | **8,180** |

The logical catalog reports 5,536 routed-expert bytes and 2,644 always-active bytes. Four
indivisible expert bundles contain 1,384 bytes each. The intentionally padded 256-column IQ
storage tests the same logical-versus-physical shape boundary used by the real quantized profile.

## Deterministic parity

- Every packed matrix is independently dequantized into an FP32 oracle graph.
- Full prefill and incremental decode match that oracle within `3e-5` for main and MTP logits and
  route weights; greedy tokens, route indices and attention indices match exactly.
- The file-backed repetition loads every non-routed matrix as a bounded trunk tile and every
  routed W1/W2/W3 group as one complete expert bundle. Graph descriptors retain the Governor-owned
  immutable buffers without copying payloads.
- The router-first two-entry cache loads only the two unique experts selected for a token, reuses
  them on the next token and safely replaces them on a later route.
- The asynchronous one-worker/two-request path submits the selected mixed-format misses, acquires
  them at deterministic compute boundaries and preserves the same output/state oracle.
- The test explicitly verifies that the failure token selects expert 0, truncates that expert's
  IQ1_S W1 payload and observes transactional decode failure with unchanged incremental state.

The standalone IQ1_S/IQ3_XXS tests remain pinned to ggml-derived fingerprints. This layer gate
therefore validates composition of those already-verified storage equations; it is not a new
quantizer-quality claim.

## Safety and malformed-input coverage

- destination, staging, scratch and aggregate host-memory limits are set before any file-backed
  load;
- every complete expert remains one RAM residency object and is never split across RAM and VRAM;
- cancellation, one-byte budget refusal, wrong matrix shape, truncated Q8/IQ payload, malformed
  I32 route data and invalid resident subranges fail without partial state publication;
- a W2 descriptor relabeled as IQ1_S is rejected because its encoded size no longer matches the
  canonical IQ3_XXS slice;
- oversized/duplicate/unknown selected sets, in-flight eviction and async queue/worker violations
  retain the existing fail-closed behavior;
- all fixture files are temporary, small and removed with the test directory; no model artifact is
  mutated or downloaded.

## Validation host

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM and NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code was compiled or executed.

## Exact results

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors | 10/10 passed in 9.44 s |
| GCC Debug, warnings as errors | 10/10 passed in 22.77 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 10/10 passed in 60.02 s |
| GCC ThreadSanitizer native suite, per-process ASLR disabled | passed |
| GCC libstdc++ debug iterators/assertions | native suite passed |
| Clang 18 Release, warnings as errors | 9/9 passed in 3.79 s |
| MSVC Release x64, warnings as errors | 9/9 passed in 3.73 s |
| MSVC Debug x64, warnings as errors | 9/9 passed in 7.03 s |
| Repeated mixed-format native regression | 20/20 passed |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python offline importer, `compileall` and `git diff --check` | passed |

ASan used leak detection and halt-on-error; UBSan and TSan halted on the first finding. TSan used
`setarch x86_64 -R` only for its instrumented process. MSVC host and target architecture were
explicitly x64.

The immediately preceding asynchronous milestone also passed the serial five-test SmolLM2,
OLMoE and DeepSeek real-model regression on the same production runtime. This commit changes only
the deterministic native integration gate, so that 199.19-second subset was not redundantly rerun
and is not counted in the table above.

## Performance statement

No token rate, TTFT, SSD bandwidth, physical bytes/token or speedup is reported. The 8,180-byte
fixture is deliberately cache-sized and its timing is meaningless for large-model performance.
The value of this gate is complete mixed-format control-flow, ownership and numerical parity.

## Known limitations

- The fixture dimensions are reduced; it is architecture-faithful but not checkpoint-shaped.
- Routed bytes are deterministic synthetic IQ blocks, not official checkpoint payload.
- The real 4,096/2,048 expert and the reduced graph are validated separately, not in one forward.
- No official trunk range has been read or executed.
- Expert residency is RAM-only and kernels are scalar CPU references.
- Physical NVMe overlap, pinned buffers, VRAM placement, CUDA and generation remain absent.

## Next bottleneck

Define a checkpoint-shaped, single-token layer probe and derive the smallest exact set of official
trunk and selected-expert ranges it would require. The probe must preserve the same hard RAM,
staging and disk reserves. Any new range download needs explicit approval, immutable identities
and hashes; a full checkpoint import remains unnecessary until this bounded probe passes.
