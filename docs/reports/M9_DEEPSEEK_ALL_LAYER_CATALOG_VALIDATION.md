# M9 DeepSeek all-layer catalog validation

Validation date: 2026-08-11

Implementation commit: `a3b95e08fd9f80a6206454896cef7b7b20fa482f`

## Scope

This milestone extends the verified 0731 metadata-to-consumer contract from layer 0 to every
main layer without acquiring tensor payload:

```text
exact pinned metadata + native reference configuration
  -> derive one canonical consumer layout per layer and compression mode
  -> validate name / layer / role / shape / GGML type / encoded bytes
  -> reject missing or unexpected non-routed objects
  -> reconcile complete layer totals with the independent streaming planner
  -> 43 layers / 1,193 tensors / 6,390,088,008 bytes
```

Only the previously imported 5,343,584-byte metadata/header artifact was opened. No byte from the
declared 82.54 GB tensor payload was downloaded, read or executed.

## Implemented contracts

The new all-layer validator first requires the pinned manifest/spec, exact metadata-derived
`DeepseekV4ReferenceConfig` and the existing layer-0 contract. It then derives each layer layout
from the verified hash-layer count and compression schedule.

### Common/hash layout

Layers 0 and 1 contain 21 non-routed objects each:

- 11 attention, router, shared-expert and Hyper-Connection matrices;
- nine normalization, attention-sink and Hyper-Connection F32/BF16 tensors;
- one exact two-dimensional I32 `experts-per-token × vocabulary` routing table.

Layer 2 retains the same hash table and adds the 4:1 compression/indexer objects below.

### 4:1 compressor/indexer layout

Every ratio-4 layer contains 31 non-routed objects. In addition to the 21-object common/router
set, it requires:

- attention compressor gate/KV matrices, positional coefficients and normalization;
- indexer query and head-weight projections;
- indexer compressor gate/KV matrices, positional coefficients and normalization.

There are 21 such layers, including hash layer 2 and the even learned-router layers through 42.

### 128:1 compressor layout

Every ratio-128 layer contains 25 non-routed objects: the 21-object common/learned-router set plus
the four attention-compressor tensors. No learned indexer tensors are accepted in this layout.
There are 20 such layers, the odd learned-router layers from 3 through 41.

The exact count reconciliation is:

```text
2 × 21 + 21 × 31 + 20 × 25 = 1,193 tensors
```

The validator recomputes every encoded span through the bounded GGML storage rules and totals
6,390,088,008 bytes. That result must agree with the independent layer-streaming planner's layer,
tensor, supported-storage and byte totals. Per-layer trunk sizes remain 138,867,160 through
161,707,992 bytes, with a 35,651,584-byte largest individual tensor/tile.

## Safety and failure behavior

For each expected tensor the validator requires:

- exact canonical `blk.N.*` identity and checked layer number;
- no routed-expert identity in the trunk set;
- the exact catalog role expected by its consumer;
- matrix rows and minimum logical input width, or the exact F32/BF16 shape;
- exact two-dimensional hash-table orientation;
- a GGML type supported by the bounded reference storage path;
- recomputed encoded bytes equal to the immutable source range.

After expected objects pass, the complete non-routed count for that layer must match. This rejects
unknown extras instead of silently leaving them unbound. Counts and bytes use overflow-checked
accumulation. Allocation failure returns `resource_exhausted`; no partial verification state is
published.

The CLI publishes the all-layer gate and its totals only after the complete validation succeeds.
The tested checkpoint still reports `metadata_only=true`, `full_payload_present=false`,
`executable_source_binding_verified=false` and `execution_supported=false`.

## Correctness and fault coverage

Deterministic native tests cover:

- successful complete single-layer hash-layout validation;
- exact returned layer/tensor/byte totals;
- rejection of a flattened rather than two-dimensional `tid2eid` table;
- rejection of an unexpected non-routed layer tensor;
- rejection of a canonical matrix with the wrong catalog role;
- the earlier wrong encoded-byte, missing tensor, wrong config and unsupported-storage failures.

The real metadata opt-in test validates every compressed/indexer layout and requires the CLI JSON
to report `all_layer_reference_catalogs_verified=true`, 43 layers, 1,193 tensors and
6,390,088,008 bytes.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported VRAM, driver 610.88;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- warnings as errors in every compiler build;
- no CUDA code and no official DeepSeek tensor payload executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release + real metadata opt-ins | 15/15 passed in 13.06 s |
| GCC Debug + real metadata opt-ins | 15/15 passed in 28.88 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 74.81 s |
| Clang 18 Release + real metadata opt-ins | 15/15 passed in 12.35 s |
| GCC Debug + libstdc++ debug iterators/assertions + real metadata | 15/15 passed in 38.93 s |
| MSVC Release x64 + real metadata opt-ins | 15/15 passed in 9.87 s |
| MSVC Debug x64 + real metadata opt-ins | 15/15 passed in 48.86 s |
| Python manifest suite | 4/4 passed, one platform-specific symlink case skipped |
| Python importer suite and `compileall` | passed |

The two native TSan binaries passed with `TSAN_OPTIONS=halt_on_error=1` under
`setarch x86_64 -R`, which avoids the known WSL address-layout conflict. Ordinary TSan CTest
launches can still abort before `main` with `unexpected memory mapping`; this remains an
environmental harness limitation rather than a fully passing TSan matrix entry.

Manual `deepseek-gguf-checkpoint --json` inspection passed and reported:

- all pinned identity/config/layer-0/all-layer gates: `true`;
- validated layers / tensors / bytes: 43 / 1,193 / 6,390,088,008;
- streaming-planner layers / tensors / covered bytes: the same 43 / 1,193 / 6,390,088,008;
- full payload present and execution supported: both `false`.

## Performance statement

No tokens/second, physical-device bandwidth, bytes/token or speedup is reported. This work walks
metadata and performs bounded shape/storage arithmetic. It proves that every official main-layer
trunk object has a known native consumer contract; it does not prove that those consumers produce
correct output on official payload.

## Known limitations

- No official trunk tensor was acquired, decoded or executed.
- The 1,193 validated descriptors exclude routed expert slices and global embedding/output/MTP
  objects; those have separate catalogs or remain future binding work.
- The official `tid2eid` payload has not produced an authoritative route for a real prompt.
- No official trunk has been composed with its six selected complete experts.
- Physical NVMe overlap, CUDA, full generation and useful DeepSeek tokens/second remain
  unvalidated.

## Next bottleneck

After explicit user authorization, acquire only the exact 138,867,160-byte layer-0 trunk through
the bounded range importer. Bind its 21 objects using the already validated contract, decode an
authoritative `tid2eid` route for a pinned input, fetch only the resulting six complete expert
bundles and compare the complete layer output with the pinned reference.
