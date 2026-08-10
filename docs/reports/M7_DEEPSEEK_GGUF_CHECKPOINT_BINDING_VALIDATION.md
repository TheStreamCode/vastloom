# M7 DeepSeek GGUF checkpoint-binding validation

Validation date: 2026-08-10

Implementation commit: `6c27c3ba79d69d22368cf7052ee78d008ebdcb54`

## Scope

This milestone pins the exact
[`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/tree/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-IQ1_S)
`UD-IQ1_S` objects, adds a bounded immutable importer and binds a verified complete import to the
existing logical tensor catalog. It does not download the two large payload objects, execute a
DeepSeek layer/token, generate text or allocate CUDA memory.

The new trust path is:

```text
pinned model ID + revision + path + size + SHA-256
  -> bounded download / disk reserve / atomic immutable import
  -> native manifest and exact object-set validation
  -> full-file SHA-256 (required for complete payload sources)
  -> bounded GGUF split/catalog validation
  -> live runtime-state RAM admission
  -> one complete expert fits RAM OR VRAM, never their sum
  -> immutable executable source ranges
  -> execution_supported=false until a layer/token consumer exists
```

## Exact artifact contract

The pinned revision is `fbbb5b93fb787c21338159b0af3318bb3f4d9768`.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `...00001-of-00003.gguf` | 5,257,664 | `2fa152e25a14500e42e6d98f20d200eec1e945f34dc1cfb981b651b9c86b97be` |
| `...00002-of-00003.gguf` | 49,093,726,624 | `e9fc77544cd48aa0c31fbf32fdf5432f2b560b92a9aeffd9f690d99867cf0e5f` |
| `...00003-of-00003.gguf` | 33,440,253,504 | `fd99d5cd0503d668b6febc2eedcc9db68e264cfb79dd355efc88c15eabe4248d` |

The complete split is 82,539,237,792 bytes. Metadata-only import uses the complete first object
and exact 52,320- and 33,600-byte HTTP range prefixes for the other two objects. The bounded
metadata artifact is therefore exactly 5,343,584 bytes. Each prefix has its own pinned SHA-256
and declared full-object size.

## What was implemented

- `deepseek-v4-flash-0731-gguf-iq1s` is a first-class pinned importer profile.
- Metadata-only and full imports select disjoint exact file sets. A full import has
  82,533,980,128 bytes marked as weight payload and requires `--allow-large-download` in addition
  to the default 300 GiB storage reserve.
- The artifact directory uses the short stable name
  `deepseek-v4-flash-0731-iq1s-fbbb5b93`. The complete model ID and 40-character revision remain
  in the immutable manifest. This avoids legacy Windows `MAX_PATH` failures in nested temporary
  files without weakening identity.
- `deepseek-gguf-checkpoint` accepts only an importer directory. Raw GGUF paths, wrong model IDs,
  revisions, file counts, paths, sizes and declared hashes fail closed.
- Default validation hashes every imported file. Metadata-only mode verifies all three small
  objects. A complete import must verify all 82.54 GB before source ranges can become executable.
- `--no-verify` remains a quick inspection mode: even when complete objects exist, the catalog is
  forced back to `metadata_only` and cannot publish an executable source binding.
- A complete, fully hashed payload is rejected if runtime state does not fit the safe host budget
  or the largest complete expert fits neither the remaining RAM tier nor the independent VRAM
  tier. RAM and VRAM are never pooled for one expert.
- The CLI still reports `execution_supported=false`. Source authorization is not model execution.

## Real metadata validation

No payload weight object was downloaded. A local model store was initialized with its 300 GiB
reserve, then only the bounded pinned metadata profile was imported:

```powershell
python tools\import_hf_model.py `
  --model deepseek-v4-flash-0731-gguf-iq1s `
  --store build\vastloom-gguf-test-store `
  --metadata-only `
  --reserve-gib 300 `
  --max-mib-per-sec 32

vastloom deepseek-gguf-checkpoint `
  build\vastloom-gguf-test-store\models\original\deepseek-v4-flash-0731-iq1s-fbbb5b93--metadata `
  --context 4096 --batch-size 1 --json
```

The final import downloaded and SHA-256 verified 5,257,664 + 52,320 + 33,600 bytes, atomically
published the read-only target and produced:

| Field | Result |
|---|---:|
| Imported manifest verified | `true` |
| Imported file hashes verified | `true` |
| Full payload present | `false` |
| Executable source binding verified | `false` |
| Execution supported | `false` |
| Physical tensors | 1,328 |
| Logical tensors | 34,223 |
| Logical experts | 11,008 |
| Maximum complete expert bundle | 8,781,824 bytes |
| Streamed-trunk CPU floor | 124,905,496 bytes |
| Resident-trunk CPU floor | 7,110,994,292 bytes |

At the recorded final command, the live safe caps were 5,722,259,456 bytes of host RAM and
8,723,103,744 bytes of VRAM. The streamed-trunk capacity profile passed, the complete expert fit
independently in either tier, and the resident-trunk profile failed. These are dynamic capacity
results, not inference or throughput results.

The first real import attempt exposed a Windows path-length bug before publication: the nested
temporary path was 266 characters. No immutable target was created. The short artifact name was
then added, offline tests were extended and the same real import completed successfully.

## Deterministic and failure coverage

The native fixture builds both a complete two-shard GGUF import and a header-only import with
independent manifests and hashes. It verifies:

- complete full-file verification and executable source-range publication;
- verified metadata-only catalog publication without payload authorization;
- forced metadata-only behavior when full-file hashes are skipped;
- exact pinned production identity and total byte accounting;
- full-payload refusal when two insufficient RAM/VRAM tiers would need to be pooled;
- wrong expected SHA-256 rejection;
- missing shard rejection;
- post-manifest payload corruption rejection;
- raw GGUF CLI paths failing to bypass the importer contract;
- explicit `execution_supported=false` in GGUF catalog JSON.

The importer tests additionally cover exact metadata/full file selection, large-download opt-in,
bounded HTTP ranges, resume validation, atomic publication, Windows-safe target naming and all
pinned object/header hashes.

## Tests actually run

All native configurations used project warnings as errors.

| Toolchain/configuration | Exact result |
|---|---|
| MSVC 19.44 Release, SmolLM2 + OLMoE metadata/full + real 0731 metadata import | 13/13 passed |
| MSVC 19.44 Debug, SmolLM2 + OLMoE metadata + real 0731 metadata import | 12/12 passed |
| GCC 13.3 Release, SmolLM2 + OLMoE metadata/full + DeepSeek preview/0731 metadata | 14/14 passed |
| Clang 18.1 Release, DeepSeek preview + real 0731 metadata import | 11/11 passed |
| GCC Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 11/11 passed; no finding |
| Python `compileall` | passed |
| Manifest-tool unit tests | 3 passed, 1 Windows symlink skip |
| Offline Hugging Face importer | passed |

ASan used leak detection and halt-on-error; UBSan used halt-on-error and stack traces. The full
OLMoE runtime was excluded from MSVC Debug because its known debug-build checkpoint verification
exceeds the command timeout; the same path passed in MSVC and GCC Release. ThreadSanitizer remains
unavailable in this WSL/kernel environment because instrumented binaries fail before `main` with
`FATAL: ThreadSanitizer: unexpected memory mapping`; this is not reported as a passing race check.

## Performance

No tokens/s, TTFT, NVMe bytes/token, physical bandwidth or overlap result is reported. The real
checkpoint test hashes only 5.34 MB of metadata/header data. Its sub-second Release CTest time is
not representative of a complete checkpoint or token path.

## Compatibility boundary

The gate is architecture-independent in principle: a supported MoE must prove that irreducible
runtime state fits safe host RAM and that its largest complete expert bundle fits one real safe
RAM tier or one real safe VRAM tier. Remaining trunk and cold experts may stay immutable on NVMe
only when the adapter supplies bounded verified tiles and an actual consumer.

On the 32 GiB RAM / 12 GiB VRAM reference PC, Kimi K3 is therefore excluded from execution
validation unless an exact quantized profile passes this same gate. It remains a metadata and
capacity research target; generic swap, cross-tier expert splitting and fictional RAM+VRAM totals
are not accepted alternatives. DeepSeek 0731 `UD-IQ1_S` passes the expert-size capacity gate, but
that alone does not make the model runnable.

## Known limitations

- The real 49,093,726,624- and 33,440,253,504-byte payload objects were not downloaded, imported
  or fully hashed locally.
- Production full-source binding is implemented but has only been exercised with a synthetic
  complete payload; real official source binding remains unvalidated until the full import exists.
- No always-active trunk tile, routed expert or full DeepSeek layer reads from the bound official
  checkpoint.
- No complete quantized layer/token, generation loop, MTP verification loop or chat path exists.
- CUDA, pinned host memory, VRAM copies and NVMe/read/transfer/compute overlap remain unconnected.
- Capacity admission cannot predict useful speed. The 1,770,651,648-byte logical no-cache expert
  traffic floor still requires measured cache/prefetch reduction.

## Next bottleneck

The next slice is not another metadata feature. After an explicit review of download size and
storage reserve, import and fully hash the real three-object checkpoint, then use the verified
catalog to read one bounded always-active trunk tile and the routed expert bundle for one complete
quantized layer/token. That run must preserve the derived reservations and report logical/process/
physical bytes separately. Only after correctness should CUDA, direct I/O, pinned buffers and
read/transfer/compute overlap be optimized.
