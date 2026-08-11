# M9 DeepSeek complete checkpoint catalog validation

Validation date: 2026-08-11

Implementation commit: `59c17bc59be7b3846b95b9918f5530ef78e866bb`

## Scope

This milestone closes the complete pinned 0731 `UD-IQ1_S` GGUF catalog without acquiring or
reading tensor payload:

```text
verified 5,343,584-byte header artifact
  -> exact 43-layer native consumer contract
  -> six global consumer descriptors
  -> 11,008 deterministic routed-expert bundles
  -> 33,024 one-to-one down/gate/up logical slices
  -> physical + logical + byte partition closure
```

The declared 82,539,237,792-byte checkpoint was not downloaded. The pass opened only the already
verified metadata/header artifact, allocated no weight buffer and left
`execution_supported=false`.

## Implemented contract

The checkpoint validator first requires the pinned model/revision/manifest identity, the exact
metadata-derived reference configuration and the passing 43-layer contract. It then validates the
six layer-independent objects against their graph consumers:

| GGUF tensor | Consumer | Shape | Storage in pinned profile |
|---|---|---:|---|
| `token_embd.weight` | token embedding | `4096 × 129280` | Q4_K |
| `output.weight` | LM head | `4096 × 129280` | Q4_K |
| `output_norm.weight` | final RMSNorm | `4096` | F32 |
| `output_hc_base.weight` | HC output base | `4` | F32 |
| `output_hc_scale.weight` | HC output scale | `1` | F32 |
| `output_hc_fn.weight` | HC output projection | `16384 × 4` | F32 |

Those six descriptors total 596,000,788 encoded bytes. Names, absent layer/expert identities,
roles, shapes, storage layouts and recomputed byte counts must all match; an unexpected seventh
global object fails the closed set.

For every one of 43 layers and 256 experts, the corresponding bundle must have exactly three
canonical logical descriptors:

- `blk.N.ffn_down_exps.weight.expert.E`;
- `blk.N.ffn_gate_exps.weight.expert.E`;
- `blk.N.ffn_up_exps.weight.expert.E`.

Every tensor index must be unique, every packed slice must map one-to-one to an index, every
slice/range must stay inside the bundle, and coalesced read bytes must equal the complete bundle
size. Down projections use IQ3_XXS or MXFP4 in this pinned mixed profile; gate/up projections use
the same IQ1_S or IQ2_XXS type within a bundle. Shapes and GGML block bytes are recomputed under a
checked element ceiling.

## Exact closure

The independently derived partitions agree exactly:

```text
physical:
  1,193 non-routed layer tensors
      6 global tensors
    129 packed expert projection containers (43 × 3)
  -------------------------------------------------
  1,328 physical tensors

logical:
  1,193 non-routed layer tensors
      6 global tensors
 33,024 expert slices (43 × 256 × 3)
  -------------------------------------------------
 34,223 logical tensors in 11,008 expert bundles

bytes:
  6,390,088,008 layer-trunk bytes
    596,000,788 global bytes
  -------------------------------------------------
  6,986,088,796 always-active bytes
 75,547,803,648 routed-expert bytes
  -------------------------------------------------
 82,533,892,444 known tensor bytes
```

The GGUF payload declaration is 82,533,894,208 bytes; the 1,764-byte difference is file-format
padding and is not presented as tensor data.

## Layout-only execution boundary

The bounded byte-layout calculator now recognizes Q4_K, IQ2_XXS and MXFP4 blocks because they are
present in the pinned catalog. This does not add execution support. All three deliberately remain
false under `supports_reference_matvec_ggml_type` and
`supports_reference_storage_ggml_type`; no dequantization, matvec or embedding lookup is claimed.

Existing parity-tested IQ1_S/IQ3_XXS kernels execute the extracted layer-0/expert-0 bundle only.
The mixed checkpoint contains IQ2_XXS gate/up matrices in multiple layers and MXFP4 down matrices
in layers 26 and 42, while its global embedding and output head are Q4_K. These are explicit
blockers for complete checkpoint execution.

## Correctness and fault coverage

Deterministic native tests cover:

- successful one-layer checkpoint closure with six synthetic global consumers and four complete
  mixed-format expert bundles;
- exact global, bundle, slice, physical, logical and byte totals;
- Q4_K, IQ2_XXS and MXFP4 encoded-byte arithmetic while execution capability stays false;
- wrong global role and unexpected global rejection;
- non-canonical routed expert name rejection;
- duplicate expert tensor-index rejection;
- duplicate packed-slice rejection;
- wrong physical tensor count rejection;
- the inherited malformed layer shape/role/byte/config and unsupported-storage faults.

The opt-in real-metadata CTest requires the CLI JSON to publish the complete gate and every
official total. Manual inspection returned:

- `checkpoint_reference_catalog_verified=true`;
- global tensors / bytes: 6 / 596,000,788;
- expert bundles / logical slices / physical expert tensors: 11,008 / 33,024 / 129;
- physical / logical tensors: 1,328 / 34,223;
- known / always-active / routed bytes: 82,533,892,444 / 6,986,088,796 /
  75,547,803,648;
- `metadata_only=true`, `full_payload_present=false`,
  `executable_source_binding_verified=false` and `execution_supported=false`.

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
| GCC Release + real metadata opt-ins | 15/15 passed |
| GCC Debug + real metadata opt-ins | 15/15 passed |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed |
| Clang 18 Release + real metadata opt-ins | 15/15 passed |
| GCC Debug + libstdc++ debug iterators/assertions + real metadata | 15/15 passed |
| MSVC Release x64 + real metadata opt-ins | 15/15 passed |
| MSVC Debug x64 + real metadata opt-ins | 15/15 passed |
| Python manifest suite | 4/4 passed, one platform-specific symlink case skipped |
| Python importer suite and `compileall` | passed |

The two native TSan binaries passed with `TSAN_OPTIONS=halt_on_error=1` under
`setarch x86_64 -R`, which avoids the known WSL address-layout conflict. Ordinary TSan CTest
launches can still abort before `main` with `unexpected memory mapping`; this remains an
environmental harness limitation rather than a complete TSan CTest matrix result.

## Performance statement

No tokens/second, physical-device bandwidth, bytes/token or speedup is reported. This work walks
34,223 metadata descriptors and performs bounded integer/layout validation. It proves catalog
closure, not useful inference or SSD streaming performance.

## Known limitations and next bottleneck

- No official global, trunk or new expert payload was acquired or executed.
- Q4_K, IQ2_XXS and MXFP4 have layout-only support and no parity-tested native kernels.
- Packed Q4_K token embedding lookup and the six-global retained binder do not exist.
- The official `tid2eid` payload has not produced an authoritative route for a real prompt.
- Physical NVMe overlap, CUDA, complete layer/token execution, generation and tokens/second remain
  unvalidated.

The next safe no-download slice is pinned-ggml parity for Q4_K, IQ2_XXS and MXFP4, followed by a
bounded packed embedding/output/global binder. Only after that path is closed should an explicitly
authorized range import acquire the 138,867,160-byte layer-0 trunk and its authoritative six
experts.
