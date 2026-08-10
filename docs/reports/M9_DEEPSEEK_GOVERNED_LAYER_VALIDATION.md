# M9 DeepSeek governed file-backed layer validation

Validation date: 2026-08-10
Code commit: `c5c5ebf023b0448aac7ffd953794cf401e89c4d8`

## Scope

This milestone closes the payload-ownership gap between the bounded file loaders and the reduced
native DeepSeek graph:

```text
immutable file ranges
  -> Safety Governor pre-reservation
  -> complete trunk tensor tiles OR complete indivisible expert bundles
  -> immutable resident byte buffers that own their ResidencyUse
  -> checked zero-copy graph descriptors
  -> Q8_0 matrix and I32 router dispatch
  -> prefill + incremental state + main/MTP outputs
  -> graph release -> eviction -> reservation release
```

The same 5,364-byte packed fixture from the preceding architecture test is written to a Unicode
path and reconstructed through the actual model loaders. Eleven non-routed matrices and one I32
router table arrive as 12 complete tensor tiles. Twelve routed matrices arrive as four complete,
indivisible three-matrix expert bundles. The resulting 16 resident buffers back all 23 Q8_0
matrices and the I32 table.

This proves the loader-to-reference-graph ownership contract. It does not execute an official
DeepSeek 0731 layer or implement a production expert scheduler.

## Ownership and accounting result

`ResidentByteBuffer` owns both an immutable byte vector and the `ResidencyUse` that accounts for
it. `LoadedTensorTile` and `LoadedExpert` expose shared retained handles; a graph descriptor stores
that handle plus an exact checked offset and length. The test destroys every loader result wrapper
before inference.

After binding:

- Governor RAM payload reservation: exactly 5,364 bytes;
- Governor staging reservation: zero bytes;
- live residency objects: 16;
- active use leases: exactly one per resident buffer;
- eviction while any graph descriptor references a buffer: rejected;
- payload copy into graph-owned vectors: none on the governed path;
- reservations after clearing descriptors and evicting residents: zero.

The reference test retains a separate in-memory oracle fixture for comparison. That oracle is test
scaffolding and is not part of the governed consumer path; small container/control metadata is not
counted as model payload.

## Correctness result

The governed file-backed graph matches the independently dequantized float oracle within the
existing absolute tolerance of `3e-5` for:

- complete prefill main and MTP logits;
- one incremental decode main and MTP logits;
- greedy token IDs;
- selected expert IDs and routing weights;
- sparse-attention indices;
- incremental cache publication.

The original official FP32 graph fixture and the preceding graph-owned packed fixture remain
unchanged and passing.

## Safety and failure coverage

- A host-memory hard limit one byte below the simultaneous RAM-plus-staging requirement refuses
  the first tile before allocation and leaves all reservations at zero.
- Cancellation after the bounded read marks the residency failed and releases RAM and staging.
- A matrix descriptor with the wrong shape is rejected at bind time from its exact encoded size.
- A missing expert tensor identity is rejected before graph publication.
- A resident subrange outside its immutable backing is rejected during decode; the caller's
  previously published incremental state remains unchanged.
- A descriptor that supplies both graph-owned bytes and a resident backing is rejected as
  ambiguous.
- Complete expert bundles are never divided between RAM and VRAM. This test uses RAM only and
  does not add the two tier capacities together.

## Validation host

- Windows 11 development host, build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- NVIDIA GeForce RTX 4070 Super present but unused;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2` with 15,912,636 kB exposed RAM;
- no CUDA code was compiled or executed.

## Exact results

| Toolchain/configuration | Result |
|---|---:|
| GCC Release with all configured real-model opt-ins, serial | 14/14 passed in 232.43 s |
| GCC Debug | 10/10 passed in 24.67 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 10/10 passed in 65.05 s |
| Clang Release, focused native/operator/graph gates | 3/3 passed in 0.27 s |
| MSVC Release, clean warning-as-error configuration | 9/9 passed in 3.91 s |
| MSVC Debug, clean warning-as-error configuration | 9/9 passed in 6.92 s |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python offline importer | passed |
| Python `compileall` | passed |

The final GCC Release run included SmolLM2 in 35.33 seconds and the OLMoE runtime reference in
184.94 seconds, serially. No resource ceiling was relaxed.

A separate MSVC build directory still contained three opt-in real-model tests. Compilation and its
9 core tests passed, while SmolLM2/OLMoE execution was refused because the live Windows safe RAM
cap was zero. That is a fail-closed environmental result, not a passing real-model run.

The ThreadSanitizer build completed, but every instrumented executable exited before project code
ran with `ThreadSanitizer: unexpected memory mapping` under this WSL environment. TSan is therefore
not listed as passing for this milestone; no race finding was produced.

## Performance statement

No token rate, SSD bandwidth or physical bytes-per-token number is reported. The payload is only
5,364 bytes, process/device I/O measurement is disabled in this deterministic test, and the data is
cache-resident after loading. Treating its timing as a model benchmark would be misleading.

## Known limitations

- The file-backed layer is a reduced official-topology fixture, not an official 0731 payload.
- All four tiny fixture experts are loaded before routing. A real MoE path must route first and
  acquire only the selected complete expert bundles.
- The graph currently keeps the reduced layer's resident descriptors until explicit release; a
  production scheduler must bound residency across layers and tokens.
- Only Q8_0 is composed through the complete reduced layer. Mixed official trunk formats and the
  real IQ1_S/IQ3_XXS expert path have separate kernel evidence but are not yet composed together.
- There is no full-checkpoint generation, target-verified MTP loop, CUDA/VRAM tier, pinned host
  memory, direct I/O or physical NVMe overlap measurement.

## Next bottleneck

Split the MoE layer at the router boundary and provide a bounded selected-expert resolver. It must
load only routed complete bundles, deduplicate repeated expert IDs, retain them for compute, emit
hit/miss and logical-byte telemetry, and evict safely across multiple tokens. Validate that path
with the same deterministic oracle before binding exact mixed-format 0731 identities or requesting
permission to import the 82.54 GB payload.
