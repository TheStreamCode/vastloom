# M9 DeepSeek native range-artifact binding validation

Validation date: 2026-08-10

Implementation commit: `ced9de9101ed0382e75431634b67d4da4ebe0c61`

## Scope

This milestone adds the first native consumer of the derived
`vastloom.range-artifact.v1` container. It validates the container and binds trunk ranges to the
pinned DeepSeek V4 Flash 0731 GGUF catalog, but deliberately stops before tensor loading or model
execution:

```text
immutable partial artifact
  -> bounded closed-world manifest/file validation
  -> full local range SHA-256 verification
  -> regenerate the trunk plan from verified pinned metadata
  -> exact local-range-to-complete-shard identity binding
  -> immutable offset-zero local tile views
  -> execution unsupported
```

No official checkpoint or partial payload was downloaded for this milestone. The existing real
138,867,160-byte layer-0 trunk result remains a metadata-derived dry-run only.

## Implemented

The model-independent verifier now:

- accepts at most 4,096 rows, 1 GiB per row and 16 GiB total, with caller-lowerable bounds;
- parses at most a 16 MiB bounded JSON manifest with duplicate-key rejection;
- requires a regular non-symlink root, manifest, `ranges` directory and payload files;
- treats the directory as closed: extra, missing or differently named files fail;
- rejects absolute paths, root components, `.`/`..`, escapes and source-span overlap;
- validates counts, totals, per-range size, local hash syntax, original source identity and span;
- requires a consistent size/SHA-256 identity for every repeated source object;
- rehashes every local range by default;
- rejects artifacts that claim complete-checkpoint verification or execution support.

`range-artifact-inspect` exposes full and `--no-verify` inspection. Quick inspection remains
non-executing and cannot be passed to the DeepSeek binder.

The DeepSeek-specific trunk binder additionally:

- requires an importer manifest and metadata files already verified by SHA-256;
- requires a full local range-hash pass, with no CLI bypass;
- rebuilds the deterministic layer plan from the independently verified GGUF catalog;
- matches model, revision, layer, order, tensor label, role, source path/object, complete-object
  size/SHA-256, original offset and byte count;
- exposes each local artifact file as an immutable range beginning at offset zero;
- retains `complete_checkpoint_verified=false` and `execution_supported=false`.

`deepseek-gguf-range-bind` reconstructs the pinned metadata report and plan before invoking this
boundary. Caller-supplied expert IDs still reproduce only the plan context; they are not an
authoritative route.

## Synthetic correctness and fault coverage

The dedicated native fixture uses one metadata-only catalog, a split ten-byte trunk tensor, one
complete three-matrix expert bundle and three local range files. It verifies successful exact
binding and rejects:

- a same-size corrupted local payload;
- a quick/no-hash artifact presented to the binder;
- an extra payload file and path traversal;
- an out-of-plan but otherwise bounded source offset;
- a forged complete-source SHA-256;
- an artifact claiming execution support;
- a forged plan tensor identity;
- a mismatched pinned checkpoint identity;
- unverified catalog metadata;
- invalid caller bounds.

The CLI integration suite checks verified JSON output, quick-mode disclosure, corrupted payload
refusal, closed-layout enforcement, unknown/duplicate/missing options and the absence of a
DeepSeek binding hash bypass.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM and NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code and no official model payload were executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors, real metadata dry-run | 15/15 passed in 12.58 s |
| GCC Debug, warnings as errors, real metadata dry-run | 15/15 passed in 27.69 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 71.47 s |
| Clang 18 Release, warnings as errors | 11/11 passed in 5.15 s |
| GCC Debug + libstdc++ debug iterators/assertions | 11/11 passed in 8.49 s |
| MSVC Release x64, warnings as errors | 11/11 passed in 5.11 s |
| MSVC Debug x64, warnings as errors | 11/11 passed in 8.34 s |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Native/WSL Python importer and range-artifact CLI suites | passed |
| Python `compileall`, Markdown link check and `git diff --check` | passed |

ASan used leak detection and halt-on-error; UBSan halted on the first finding. ThreadSanitizer
binaries compiled, but every instrumented executable aborted before `main` with
`FATAL: ThreadSanitizer: unexpected memory mapping` in this WSL session. TSan is therefore an
environmental non-result, not listed as passing and not evidence of a project race.

## Performance statement

No throughput, latency, bytes/token, physical-device traffic or speedup is reported. The native
fixture hashes and binds only ten trunk bytes. The official 138,867,160-byte value was not read;
it remains dry-run planning evidence.

## Known limitations

- The raw source-plan SHA-256 is preserved as provenance. Native authority comes from reproducing
  and comparing every exact range identity, avoiding dependence on shell-specific trailing newline
  or text encoding.
- Only trunk scope can bind while routing authority is false. Selected-expert and combined
  artifacts remain unavailable.
- The binder exposes exact local tile sources but does not yet feed them through the governed
  tensor loader or DeepSeek scalar graph.
- No official trunk payload has been acquired, decoded or executed.
- Full DeepSeek generation, CUDA, pinned host buffers and NVMe/compute overlap remain absent.

## Next bottleneck

Connect the bound local trunk views to a Governor-owned tensor-load transaction and the reduced
DeepSeek scalar consumer. Prove short-read, cancellation, allocation refusal and lease cleanup on
synthetic partial artifacts. Only after that gate passes should the user be asked to authorize the
real 138,867,160-byte layer-0 trunk acquisition.
