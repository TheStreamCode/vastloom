# M9 DeepSeek governed global tensor-set validation

Validation date: 2026-08-11

Implementation commit: `342110e704e38467383b9f49a60b6da6b508844c`

## Scope and result

This milestone closes the six-global metadata-to-resident-consumer boundary for the pinned
Unsloth DeepSeek V4 Flash 0731 `UD-IQ1_S` profile:

```text
verified closed checkpoint catalog
  -> deterministic six-tensor global plan
  -> explicit bounded range import
  -> full local SHA-256 verification
  -> exact pinned source-span rebinding
  -> one aggregate RAM plus maximum-range staging reservation
  -> direct reads into one immutable packed buffer
  -> embedding, output, normalization and Hyper-Connection consumers
  -> reduced prefill and incremental decode parity
```

The official metadata produces exactly six ranges totaling 596,000,788 bytes in one pinned
source object. The largest individual range is 297,861,120 bytes. The real range importer passed
no-network dry-run only: none of those payload bytes was downloaded, allocated or executed.

The executable test uses a small deterministic Q8_0/F32 fixture. Its six ranges are loaded under
one Governor transaction and one retained `ResidencyUse`; the resulting graph matches an
independent graph built directly from the same packed bytes for prefill and one incremental
decode step.

## Implemented

- `vastloom.deepseek_v4.global_tensor_plan.v1` reproduces the complete checkpoint-catalog
  closure before exposing canonical `token_embd`, output, output norm and three output
  Hyper-Connection ranges.
- `deepseek-gguf-global-plan` emits the exact plan in text or JSON without payload access.
- The range importer accepts a dedicated `globals` scope, requires a verified global plan,
  preserves a false routing-authority claim and retains the existing explicit download flag,
  byte cap, disk reserve, exact HTTP range and atomic read-only publication rules.
- `vastloom.range-artifact.v1` admits `globals` only with `global` entries and rejects routing
  authority, mixed roles, overlapping spans, extra files and malformed identities.
- The architecture binding replays the plan from the verified catalog, checks the pinned model,
  revision, object size/hash/path and every original span, then exposes a non-executing exact
  source binding.
- The closed-set loader requires all six complete, unique catalog tensors. It reserves aggregate
  RAM plus only the largest range as staging, reads directly into one destination, rehashes every
  subspan and publishes `RAM_READY` only after the complete transaction succeeds.
- Three packed matrices bind to `embed.weight`, `head.weight` and `hc_head_fn`; three retained
  F32/BF16 tensors bind to `norm.weight`, `hc_head_base` and `hc_head_scale` without copying
  encoded payload.
- Consumer descriptors share the same immutable buffer. Destroying the loader does not release
  its one lease while the graph is live, and eviction succeeds only after all six consumer views
  are removed.

The generic range-set defaults remain 512 MiB total and 64 MiB per range. Because the official
global set and its two Q4_K matrices exceed both defaults, a future real load must explicitly set
both bounds from the verified plan and must still pass the independent Governor budgets. The
absolute 4 GiB total and 1 GiB per-range ceilings were not raised.

The official set also exceeds the range importer's 256 MiB selected-payload default. A future
transfer therefore needs both the explicit `--allow-range-download` authorization and an explicit
cap no lower than the verified 596,000,788-byte plan. Neither condition is an execution grant.

## Official metadata result

Pinned revision: `fbbb5b93fb787c21338159b0af3318bb3f4d9768`.

| Tensor | GGML type | Encoded bytes |
|---|---:|---:|
| `token_embd.weight` | Q4_K | 297,861,120 |
| `output.weight` | Q4_K | 297,861,120 |
| `output_norm.weight` | F32 | 16,384 |
| `output_hc_base.weight` | F32 | 16 |
| `output_hc_scale.weight` | F32 | 4 |
| `output_hc_fn.weight` | F32 | 262,144 |
| **Total** |  | **596,000,788** |

The command reports `metadata_only=true`, `full_payload_present=false`,
`executable_source_binding_verified=false` and `execution_supported=false`.

## Correctness and fault coverage

The native and importer tests cover:

- deterministic canonical plan order, exact totals, largest range and source-object count;
- wrong/misordered global plan identity and cross-scope import rejection;
- generic artifact wrong-role and forged routing-authority rejection;
- exact artifact/catalog/plan/checkpoint rebinding and tampered-plan rejection;
- complete six-range load, exact slice lookup and one aggregate use lease;
- packed Q8_0 embedding/output plus F32 norm/Hyper-Connection consumer binding;
- full reduced prefill and incremental decode parity against the direct packed graph;
- wrong consumer dimensions, missing range and unavailable tensor rejection;
- insufficient aggregate host-memory budget before allocation or I/O;
- cancellation before residency transition;
- same-size post-binding payload corruption detected by the second SHA-256 pass;
- zero RAM/staging reservations after every failure and after final eviction.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported VRAM, driver 610.88;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- warnings treated as errors in every compiler build;
- no CUDA code and no official global payload executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release + real metadata opt-ins | 15/15 passed in 21.26 s |
| GCC Debug + real metadata opt-ins | 15/15 passed in 29.65 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 83.32 s |
| Clang 18 Release + real metadata opt-ins | 15/15 passed in 15.22 s |
| GCC Debug + libstdc++ debug iterators/assertions + real metadata | 15/15 passed in 52.25 s |
| MSVC Release x64 + real metadata opt-ins | 15/15 passed in 11.61 s |
| MSVC Debug x64 + real metadata opt-ins | 15/15 passed in 61.39 s |
| Python manifest suite | 3 passed, 1 Windows symlink case skipped |
| Python importer suite and `compileall` | passed |

The two native TSan binaries passed with `TSAN_OPTIONS=halt_on_error=1` under
`setarch x86_64 -R`, avoiding the known WSL address-layout conflict. This is focused native
evidence, not a complete TSan CTest matrix.

## Performance statement

No token rate, physical NVMe throughput, bytes/token or speedup is reported. The official command
read verified metadata only and the executable payload is a tiny resident fixture. This milestone
proves identity, safety, ownership and numerical integration, not storage performance.

## Known limitations and next bottleneck

- The 596,000,788-byte official global set passed planning and importer dry-run only.
- No official global artifact was downloaded, rebound, loaded or executed.
- The official layer-0 trunk and authoritative `tid2eid` table have not been acquired.
- No complete official trunk-plus-six-expert layer/token path exists yet.
- The reference graph still has no 43-layer official-payload generation driver.
- Physical NVMe overlap, pinned host buffers, VRAM placement and CUDA remain unimplemented.

The next safe correctness slice is to import the exact layer-0 trunk only after explicit user
authorization, execute its official router for pinned token inputs, record the authoritative six
expert IDs and then import/load only those complete bundles. That probe must remain bounded by
the verified plan and single-tier expert admission before any performance work begins.
