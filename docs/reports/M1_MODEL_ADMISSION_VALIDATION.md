# M1 model working-set admission validation

Validation date: 2026-08-10

Validated implementation commit: `7c9c2003cdf2ecc4a46a2f5cb0a02308ab94ac57`

## Scope

This milestone turns the public reference-PC working-set rule into an enforced native capacity
gate:

```text
live hardware snapshot
  -> OS, emergency, driver and storage reserves
  -> explicit KV, staging and scratch placement
  -> adapter-declared minimum RAM working set
  -> one complete routed-expert bundle
  -> one real RAM or VRAM admission tier
  -> aggregate host-memory Governor ceiling
  -> capacity-compatible plan or fail-closed rejection
```

The gate answers whether a declared model profile can fit the irreducible working set on the
sampled host without treating SSD space, installed RAM or RAM+VRAM as fictional executable
capacity. It does not make a metadata-only manifest executable and does not prove that the
adapter's declared minimum is correct for an unimplemented full graph.

## Native implementation

The implementation adds four connected safety boundaries:

1. New `vastloom.manifest.v1` MoE manifests require nonzero layer/context counts, bounded top-k,
   total routed bytes, one complete expert-bundle size and minimum live RAM. Dense, routed and
   active byte declarations cannot exceed the checkpoint total.
2. Persisted `stargate.manifest.v1` artifacts remain readable. Incomplete legacy MoE metadata
   emits a compatibility warning, but the planner rejects it rather than guessing missing values.
3. The planner deducts explicit KV, staging and scratch placement before sizing optional expert
   RAM cache. It compares `minimum_ram_bytes` with the live post-reserve host budget, then requires
   `estimated_expert_bytes` to fit either the remaining RAM expert tier or the GPU-resident tier.
   The selected tier and its actual cap are reported in text and JSON.
4. `ResourceGovernor` enforces one atomic aggregate ceiling across RAM destinations, staging and
   scratch in addition to the existing per-kind limits. VRAM remains a separate tier. Aggregate
   use participates in pressure-state escalation, and combined byte-budget arithmetic is checked
   for overflow.

The offline manifest generator now accepts and validates the same MoE identity fields before
atomic publication. It does not emit a manifest that the native strict reader would immediately
reject.

## Deterministic admission results

The committed DeepSeek research manifest is metadata-only and represents the pinned M6 preview
capacity profile. On the simulated reference PC and balanced profile it produced:

| Field | Result |
|---|---:|
| physical RAM / VRAM | 32 GiB / 12 GiB |
| safe post-reserve RAM | 23.04 GiB |
| safe post-reserve VRAM | 10.22 GiB |
| declared minimum RAM working set | 16.00 GiB |
| declared complete expert bundle | 12.75 MiB |
| selected complete-expert tier | RAM |
| selected RAM expert-tier cap | 9.00 GiB |
| explicit RAM staging | 1.50 GiB |
| explicit RAM scratch | 943.72 MiB |
| result / CLI exit | capacity-compatible / `0` |

The same manifest with simulated physical RAM reduced to 20 GiB produced a 13.00 GiB safe RAM
budget, the blocker `minimum RAM working set exceeds the live post-reserve budget`, and exit code
`78`. No weight allocation or download occurred in either command.

Unit coverage also proves:

- admission to RAM when one complete expert fits the RAM expert tier;
- admission to VRAM when it does not fit the RAM tier but fits the separate GPU-resident tier;
- rejection of that same profile when no qualifying GPU tier exists;
- rejection of an oversized expert, missing expert identity, missing count, low current RAM and
  combined-budget overflow;
- atomic rejection when RAM plus staging plus scratch would exceed their aggregate ceiling even
  though each individual per-kind request is legal;
- VRAM reservations do not consume the host aggregate ceiling;
- aggregate use alone escalates the Governor to `PRESSURE`;
- incomplete new manifests fail parsing, while incomplete legacy manifests remain readable but
  fail planning admission.

## Validation host

- Windows 11, native MSVC validation plus Ubuntu WSL cross-toolchain/sanitizer validation;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- AVX2, AVX-512F and FMA detected;
- 33,407,475,712 bytes physical RAM;
- 12,618,141,696 bytes available at the recorded final probe;
- NVIDIA GeForce RTX 4070 Super, 12,878,610,432 bytes VRAM;
- NVIDIA driver 610.88;
- CUDA was not used by this milestone.

## Build and test matrix

Every native configuration used project warnings as errors.

| Toolchain / configuration | Result |
|---|---:|
| MSVC 19.44 clean Release, no external models | 9/9 passed |
| MSVC 19.44 Debug, no external models | 9/9 passed |
| GCC 13.3 Release, SmolLM2 + full OLMoE + DeepSeek metadata opt-ins | 13/13 passed |
| Clang 18.1 Release, DeepSeek metadata opt-in | 10/10 passed |
| GCC 13.3 Debug + ASan/UBSan, DeepSeek metadata opt-in | 10/10 passed |
| GCC 13.3 Debug + TSan | 9/9 passed |

ASan ran with leak detection and halt/abort on finding. UBSan and TSan halted on the first
finding; TSan ran with ASLR disabled for the instrumented WSL processes. No sanitizer finding was
reported.

Python `compileall` and the offline importer passed. The manifest-tool suite passed three tests;
the Windows symlink test was skipped because the current account could not create a symlink.

A separate Windows opt-in run passed 11/12 tests and deliberately stopped before OLMoE execution:
the verified inspector measured only 2,255,454,208 safe RAM bytes against a 3,201,568,768-byte
minimum streaming estimate while other desktop workloads were active. The test harness expected
admission and therefore reported failure, but the runtime's rejection was the required safety
behavior. No limit was raised and no process was terminated to force the test through. The same
full real OLMoE path passed in the final GCC Release run under sufficient governed headroom.

## Performance evidence

No inference-throughput result is claimed. This milestone changes control-plane admission and
reservation accounting, not expert kernels or storage throughput. The 171.31-second OLMoE test in
the final GCC matrix is an end-to-end correctness/fault suite and must not be interpreted as a
model benchmark. No physical NVMe bytes/token, CUDA throughput or DeepSeek tokens/s measurement
was taken.

## Known limitations

- `minimum_ram_bytes` remains adapter-declared. It is strictly enforced but is not yet derived
  automatically from every tensor shape, requested context and backend workspace.
- The passing DeepSeek research plan is capacity-only. Its manifest is metadata-only, and full
  packed trunk/layer/token execution remains unsupported.
- The VRAM tier is a planner capacity result; CUDA allocation, kernels and transfer execution do
  not exist yet.
- The gate proves one complete expert can be resident. It assumes selected experts can execute
  sequentially; adapters requiring multiple simultaneous experts must declare that larger bundle
  as their indivisible expert working set.
- Requested-context-specific KV planning is not yet exposed by `vastloom plan`; adapters must
  conservatively include the supported context policy in their declared minimum.
- macOS/Metal remain unvalidated and are tracked separately in the public roadmap.
- No exact Kimi K3 execution profile was admitted or downloaded. It remains capacity-study-only
  on the 32 GiB / 12 GiB reference machine.

## Next bottleneck

The next correctness slice is to derive, rather than manually declare, the DeepSeek 0731 minimum
working set from the verified GGUF catalog, chosen context and actual packed kernel workspace.
That derived plan must bind immutable full shards and drive one router-selected complete layer and
token through the existing governed loader. Only after parity should VRAM residency, CUDA kernels
and overlapped direct I/O be connected.
