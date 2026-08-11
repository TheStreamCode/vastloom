# M9 DeepSeek metadata-to-graph validation

Validation date: 2026-08-11

Implementation commit: `05e58dce20e7bace2f89aed3ef1db279d26792ef`

## Scope

This milestone closes the configuration-authority gap between the exact pinned
`unsloth/DeepSeek-V4-Flash-0731-GGUF` metadata and the native reference graph:

```text
verified three-object metadata manifest + exact pinned checkpoint spec
  -> bounded GGUF scalar/array retention
  -> complete architecture snapshot
  -> native DeepseekV4ReferenceConfig
  -> exact 21-object layer-0 consumer/storage validation
  -> no tensor allocation and no payload read
```

The inspected artifact contains only 5,343,584 bytes of independently hashed metadata/header
objects. The declared complete checkpoint is 82,539,237,792 bytes, but it was not imported,
opened as payload or executed. This is a metadata-to-graph contract, not DeepSeek inference.

## Exact identity and architecture

- model: `unsloth/DeepSeek-V4-Flash-0731-GGUF`;
- revision: `fbbb5b93fb787c21338159b0af3318bb3f4d9768`;
- quantization profile: `UD-IQ1_S`;
- split objects: 3;
- physical tensors: 1,328;
- logical tensors: 34,223;
- logical routed-expert bundles: 11,008;
- vocabulary: 129,280;
- maximum context: 1,048,576 tokens;
- width: 4,096;
- main/hash layers: 43 / 3;
- attention heads / KV heads / head width / RoPE width: 64 / 1 / 512 / 64;
- routed experts / selected experts / shared experts: 256 / 6 / 1;
- routed-expert intermediate width: 2,048;
- query LoRA rank / output groups / output LoRA rank: 1,024 / 8 / 1,024;
- sliding window: 128;
- learned indexer heads / width / top-k: 64 / 128 / 512;
- Hyper-Connection multiplier / Sinkhorn iterations: 4 / 20;
- compression schedule values: 46, restricted to `0`, `4` or `128`;
- routed/shared SwiGLU clamp schedule values: 43 / 43, with layer-0 values `10` / `10`.

The exact pinned spec comparison includes model/revision, layer/expert/top-k values and every
shard path, complete size, complete SHA-256, metadata path, metadata length and metadata SHA-256.
A merely shape-compatible or renamed catalog cannot cross this gate.

## Implemented

The GGUF reader can now retain explicitly named F32/F64 arrays in addition to integral arrays.
Retention remains opt-in and bounded:

- integral and floating arrays share one aggregate element budget;
- at most 64 retained keys are accepted across both kinds;
- duplicate and cross-kind duplicate keys fail before parsing;
- requested scalars, wrong element kinds, nested values, budget overflow and non-finite schedule
  values fail closed;
- every unrequested array is still skipped instead of materialized.

For the exact pinned checkpoint, the inspector derives a complete architecture snapshot and then
constructs `DeepseekV4ReferenceConfig`. The bridge validates scalar ranges, schedule lengths,
supported compression ratios, YaRN identity, normalized sqrt-softplus routing and all dimensions
needed by the reference consumers.

Routed and shared expert paths no longer assume one common clamp when verified schedules exist.
The native graph validates both schedules before execution, indexes main and MTP blocks without
aliasing their positions and applies the separate values to the correct expert path. Reduced
legacy fixtures keep their explicit scalar behavior only when both schedules are absent. This
matches the separate `swiglu_clamp_exp` and `swiglu_clamp_shexp` loading and graph use reviewed in
the upstream
[`deepseek4.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek4.cpp) and
[`llama-graph.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp) on the
validation date.

The metadata-only layer-0 validator reuses the resident binder's canonical layout and requires
exactly:

- 11 matrices for attention, router, shared experts and Hyper-Connections;
- nine F32/BF16 normalization, sink, base and scale vectors;
- one I32 `tid2eid` routing table with `vocabulary * top-k` elements.

For every descriptor it checks canonical name, layer/role, shape, logical consumer dimensions,
GGML storage type and independently recomputed encoded byte count. The official result is 21
objects and 138,867,160 logical bytes. The validator does not acquire any source range.

The checkpoint CLI now reports separate machine-readable gates for exact pinned-spec identity,
reference-config verification and layer-0 descriptor verification, plus the bounded architecture
summary. It continues to report `metadata_only=true`, `full_payload_present=false`,
`executable_source_binding_verified=false` and `execution_supported=false` for the tested store.

## Correctness and fault coverage

Native tests cover:

- exact F32 array retention and shared integral/floating element-budget accounting;
- requested scalar, wrong-kind, duplicate and cross-kind duplicate rejection;
- legacy scalar-clamp parity against the new routed/shared schedule path;
- malformed schedule-length rejection before graph execution;
- successful pinned-report to native-config construction;
- successful zero-payload validation of the complete reduced 21-object canonical catalog;
- rejection of a mismatched native config;
- rejection of a descriptor whose encoded bytes disagree with shape/storage;
- rejection of an incomplete routed clamp schedule;
- rejection of an unpinned checkpoint identity.

The real metadata opt-in test additionally requires all three new verification flags, vocabulary
129,280, schedule counts 46/43/43 and 43 planned layers in the CLI JSON.

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
| GCC Release + real metadata opt-ins | 15/15 passed in 12.14 s |
| GCC Debug + real metadata opt-ins | 15/15 passed in 26.45 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 71.75 s |
| Clang 18 Release + real metadata opt-ins | 15/15 passed in 12.16 s |
| GCC Debug + libstdc++ debug iterators/assertions + real metadata | 15/15 passed in 32.24 s |
| MSVC Release x64 + real metadata opt-ins | 15/15 passed in 10.04 s |
| MSVC Debug x64 + real metadata opt-ins | 15/15 passed in 47.79 s |
| Python manifest suite | 4/4 passed, one platform-specific symlink case skipped |
| Python importer suite and `compileall` | passed |

The ordinary TSan CTest launch aborted before `main` with
`FATAL: ThreadSanitizer: unexpected memory mapping`. Running the two native test binaries under
`setarch x86_64 -R` avoided that WSL address-layout conflict: `vastloom_tests` and
`vastloom_range_artifact_test` both passed with `TSAN_OPTIONS=halt_on_error=1`. The ordinary CTest
result remains an environmental harness failure and is not reported as a fully passing TSan
matrix entry.

Manual `deepseek-gguf-checkpoint --json` inspection passed and reported:

- pinned spec, reference config and layer-0 catalog gates: all `true`;
- total metadata/header bytes: 5,343,584;
- declared full object bytes: 82,539,237,792;
- declared tensor payload bytes: 82,533,894,208;
- layer-0 trunk bytes: 138,867,160;
- full payload present and execution supported: both `false`.

## Performance statement

No tokens/second, physical-device bandwidth, bytes/token or speedup is reported. The only real
checkpoint data read was the previously imported metadata/header artifact. The validator performs
catalog arithmetic and therefore cannot demonstrate tensor decode, NVMe overlap or generation
performance.

## Known limitations

- No byte from the official 138,867,160-byte layer-0 trunk was acquired or executed.
- The official `tid2eid` table has not produced an authoritative route for a real prompt.
- The exact six selected experts for such a route have not been acquired or composed with the
  official trunk.
- Only the layer-0 non-routed consumer catalog is checked end to end. Later compressor/indexer
  layer payloads, global embeddings/output and MTP generation remain unbound.
- The bridge constructs a scalar CPU reference configuration; it is not a production packed
  whole-model executor.
- Physical NVMe overlap, CUDA, full generation and useful DeepSeek tokens/second remain
  unvalidated.

## Next bottleneck

After explicit user authorization, import only the exact 138,867,160-byte layer-0 trunk through
the existing bounded range pipeline, keeping the model-store free-space reserve intact. Bind its
21 verified objects, execute a pinned input far enough to decode the authoritative `tid2eid`
route, acquire only the resulting six complete expert bundles and compare the complete layer
output with the pinned reference. Only then should Vastloom measure physical-device read/compute
overlap or extend the path toward token generation.
