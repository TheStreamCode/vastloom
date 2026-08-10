# M9 DeepSeek router-first selected-expert validation

Validation date: 2026-08-10
Code commit: `f88072922e40f78c773acc80dd5bb0151a6ff12e`

## Scope

This milestone connects authoritative DeepSeek routing to bounded file-backed expert acquisition
and packed reference compute:

```text
router logits -> exact top-k route
  -> stable deduplication of selected expert IDs
  -> block-scoped request bound
  -> bounded complete-bundle RAM cache
       hit  -> retain immutable resident buffer
       miss -> Governor reservation -> bounded file read -> RAM_READY
       full -> evict least-recently-used bundle only when no inference retains it
  -> packed W1/W3 SwiGLU + W2 projection
  -> authoritative weighted reduction
  -> per-expert hit/miss/logical-byte trace
```

The implementation is a reusable synchronous correctness baseline. The validation fixture has the
official DeepSeek layer topology but reduced deterministic dimensions and Q8_0 storage. It is not
an official DeepSeek 0731 payload run, generation path, asynchronous-overlap result or CUDA result.

## Implemented vertical slice

- `DeepseekV4ReferenceGraph` accepts a provider per exact MoE block identity.
- The graph computes routing first, validates every expert ID and deduplicates the selected set
  before invoking the provider once.
- A mandatory request limit rejects an oversized selected set before any provider I/O.
- `DeepseekV4BoundedExpertCache` validates complete W1/W2/W3 bundle bindings, exact GGML encoded
  sizes and immutable catalog slice identities at creation.
- Every miss uses the existing `load_expert_bundle` transaction, which reserves the complete RAM
  destination and bounded staging before allocation/read.
- The fixed-capacity LRU stores only complete indivisible expert bundles. It never joins RAM and
  VRAM capacity and never splits one expert between tiers.
- Resident packed descriptors retain the loader's immutable buffer and `ResidencyUse`; eviction
  skips/rejects buffers still retained by an inference result.
- The layer trace reports expert ID, resident bytes, logical bytes read and cache-hit state.
- Provider exceptions, allocation failures and malformed result identity/count/telemetry are
  converted to explicit errors. Incremental graph state is published only after the full step
  succeeds.

## Deterministic result

The 5,364-byte Unicode-path fixture contains 2,644 bytes of always-active layer data and four
complete 680-byte routed-expert bundles. The cache is capped at two experts / 1,360 resident
payload bytes for this sequence:

| Step | Authoritative selected experts | Cache result | Logical expert bytes read |
|---|---|---:|---:|
| One-token prefill | `2, 3` | 2 misses | 1,360 |
| Incremental token 10 | `2, 3` | 2 hits | 0 |
| Incremental token 12 | `0, 1` | 2 misses, 2 LRU evictions | 1,360 |

After the three accepted steps the cache reports four misses, two hits, two evictions and exactly
2,720 cumulative logical bytes read. At every step, main/MTP logits, greedy token IDs, route IDs,
route weights and attention indices match the independently dequantized oracle within the existing
absolute tolerance of `3e-5`.

The eight-token fixture selects four unique experts. Its configured request/cache bound is two, so
that prefill is rejected before the provider runs. Production prefill chunking is still required;
the test does not silently expand the cache.

## Safety and fault coverage

- wrong-block, duplicate, unknown and oversized provider requests are rejected;
- cache construction rejects an incomplete/invalid matrix layout, duplicate expert identity and a
  request limit larger than cache capacity;
- a one-byte RAM budget refuses the first expert without a cached entry or residual RAM/staging
  reservation;
- a provider returning the wrong bundle count is rejected and incremental state is unchanged;
- an arbitrary exception thrown by a provider is contained and returned as an error;
- while direct consumers retain experts 0/1, a competing 2/3 request cannot evict them and
  explicit cache clearing also fails safely;
- after those consumers release, cache clear and trunk eviction return RAM, VRAM and staging
  reservations to zero;
- no source model file is modified and no large model was downloaded.

## Validation host

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32,624,488 KiB visible Windows RAM; 11,268,880 KiB free at the recorded probe;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported VRAM, unused by this CPU test;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`, 15,912,636 KiB exposed RAM;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022 17.14.36/MSVC;
- no CUDA code was compiled or executed.

## Exact results

| Toolchain/configuration | Result |
|---|---:|
| GCC Release, warnings as errors | 10/10 passed in 9.65 s |
| GCC Release, serial real-model opt-in subset | 5/5 passed in 196.49 s |
| GCC Debug, warnings as errors | 10/10 passed in 23.08 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 10/10 passed in 57.62 s |
| Clang 18 Release, warnings as errors | 9/9 passed in 3.81 s |
| MSVC Release, warnings as errors | 9/9 passed in 3.75 s |
| MSVC Debug, warnings as errors | 9/9 passed in 6.80 s |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python offline importer | passed |
| Python `compileall` | passed |

The serial opt-in subset used already-present verified local artifacts. SmolLM2 passed in 29.21
seconds, OLMoE inspection in 1.12 seconds, OLMoE CPU runtime in 159.58 seconds, DeepSeek metadata
inspection in 6.11 seconds and the synthetic full-checkpoint source-binding gate in 0.44 seconds.
No resource limit was relaxed and no external model was downloaded.

The ThreadSanitizer build completed, but the instrumented native test exited before project code
with `ThreadSanitizer: unexpected memory mapping` under WSL. TSan is not a passing gate for this
milestone and produced no project race finding.

## Performance statement

No token rate, TTFT, SSD bandwidth or physical NVMe bytes/token is reported. The routed payload is
only 2,720 bytes, reads are synchronous and buffered, process/device I/O counters are disabled in
the deterministic fixture, and the operating-system cache state is not controlled. The exact
logical-byte telemetry validates accounting semantics, not useful large-model performance.

## Known limitations

- The selected-expert consumer is the reduced Q8_0 fixture, not the official mixed-format 0731
  checkpoint.
- The cache tier is RAM only. VRAM placement, pinned staging and CUDA kernels are absent.
- Expert acquisition is synchronous; the existing asynchronous loader pool is not connected to
  this router-first compute path.
- Multi-token prefill is not chunked into bounded selected sets.
- The trunk remains fully resident for the reduced fixture; a production streamed-trunk schedule
  is not connected.
- There is no generation loop, completion parser, target-verified MTP, cancellation spanning
  compute, direct I/O or measured read/compute overlap.

## Next bottleneck

Bind the verified official 0731 GGUF identities to an architecture-specific mixed-format layer
consumer. Begin with one layer and a bounded token/prefill probe, then connect the fixed-worker
loader so expert N+1 can be acquired while useful work continues. The acceptance gate is exact
logit/route parity plus measured timestamps and logical/device bytes; full-model import remains
unnecessary until that layer probe can run correctly and safely from pinned slices.
