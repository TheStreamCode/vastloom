# Alpha 4 open-source rebrand validation report

Validation date: 2026-08-09
Implementation commit: `b3c2f2cfe6d9a918769b8020efc5ef321bea4711`

## Scope

This report validates the rename of the public project from Stargate Local AI to **Vastloom**,
the compatibility boundary for existing local artifacts, the new open-source repository surface,
release packaging and the unchanged inference paths available in `0.4.0-alpha.4`.

It does not claim that DeepSeek V4 Flash or the CUDA backend works. The real-model gates below
exercise the existing SmolLM2 dense reference and OLMoE CPU expert-streaming paths. The bundled
TinyChat checkpoint remains a deliberately synthetic integration fixture.

## Implemented

- Renamed the public CLI, watchdog, C++ namespace, headers, CMake project and exported target to
  `vastloom` / `Vastloom::Core`.
- Retained read-only compatibility for legacy manifests, imported-checkpoint metadata and model
  store markers. Conflicting legacy/current identities and symlinked control files fail closed.
- Preserved the existing `SGTMV001` model magic and `SGHB1` watchdog magic as stable wire-format
  identifiers rather than rewriting immutable data for branding.
- Added an Apache-2.0 license, notice, contribution and security policies, governance, support,
  code of conduct, citation metadata, issue forms, pull-request template and code ownership.
- Added pinned CI, sanitizers, CodeQL, dependency review, Dependabot and tagged-package workflows.
- Added installable CMake package metadata, `TGZ`/`ZIP` packages and SHA-256 sidecars.
- Regenerated the deterministic bundled fixture under model id `vastloom/tinychat-v1`.
- Added the project identity, README cover, GitHub social preview and documented generation briefs.
- Reviewed DeepSeekV4SSD as research input without copying its code or treating its Apple/MLX
  measurements as Windows/CUDA results.

## Compatibility contract

| Artifact | New writes | Existing reads |
|---|---|---|
| Model manifest | `vastloom.manifest.v1` | also `stargate.manifest.v1` |
| HF import metadata | `vastloom.hf-import.v1` / `vastloom-import.json` | also the legacy schema/name |
| Model store | `.vastloom-store` / `vastloom.store.v1` | also the legacy marker/version |
| CMake options | `VASTLOOM_*` | legacy `STARGATE_*` cache values are imported |
| Native model/heartbeat magic | unchanged | unchanged |

This is deliberately asymmetric: new artifacts use Vastloom identities, while verified existing
downloads remain usable without mutation. If both current and legacy control files are present, or
if a control file is a symlink, the reader rejects the store rather than guessing.

## Validation host

The Windows baseline probe recorded before the final memory-pressure check reported:

- Windows on AMD Ryzen 9 7900X-class CPU, 24 logical processors;
- AVX2, AVX-512F and FMA available;
- 33,407,475,712 bytes total RAM and 12,896,555,008 bytes available at the final probe;
- NVIDIA GeForce RTX 4070 SUPER, driver 610.88;
- 12,878,610,432 bytes total VRAM and 9,053,405,184 bytes free at the final probe;
- GPU temperature 53 C and configured power limit 220 W;
- 2,047,435,862,016-byte storage volume with 1,204,414,742,528 bytes free.

These values are a point-in-time diagnostic, not reservations granted to inference. The Resource
Governor still preserves explicit host headroom and may choose much smaller budgets.

## Build and test matrix

| Toolchain/configuration | Result |
|---|---|
| MSVC 19.44 Release, warnings as errors, full real-model suite | 8/8 passed in 72.67 s |
| MSVC clean Release, warnings as errors, bounded suite | 5/5 passed |
| MSVC Debug, warnings as errors, bounded suite | 5/5 passed |
| GCC 13 Release, warnings as errors, full real-model suite | 8/8 passed in 188.84 s |
| Clang 18 Release, warnings as errors, bounded suite | 5/5 passed in 2.55 s |
| GCC 13 Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 5/5 passed in 8.47 s |
| GCC 13 + libstdc++ debug iterators/assertions | 5/5 passed in 4.85 s |
| GCC 13 ThreadSanitizer | compiled; runtime unavailable in this WSL environment |

The final full MSVC Release run included the native suite, TinyChat chat/self-test/integration, offline HF
importer, SmolLM2 reference, OLMoE inspection and OLMoE runtime reference. The final full GCC run
ran the same eight tests. Python tools and tests passed `compileall`; the offline importer suite
passed. The Windows manifest-tool suite passed its applicable test and skipped the symlink-only
case where the host did not expose the required privilege.

A later real-model Debug run encountered only 7,806,771,200 bytes of available RAM. The Safety
Governor returned exit code 75 and refused SmolLM2 interactive startup and the following OLMoE
execution probe because less than 8 GiB of headroom remained. No limit was reduced and the run is
not counted as a real-model Debug pass. It exposed a pressure-sensitive test assumption: OLMoE
inspection had required a streaming candidate whenever the cap was nonzero, even when the live
safe cap was below the model's estimated minimum. The follow-up test now requires the reported
decision to match that comparison. A fresh Debug build with no real checkpoint configured then
passed the bounded 5/5 suite in 3.44 s.

The TSan binaries terminate before `main` with `FATAL: ThreadSanitizer: unexpected memory mapping`
under the available WSL runtime. This is recorded as an environmental limitation, not a pass.

## Fixture and manual checks

Bundled fixture:

- model id: `vastloom/tinychat-v1`;
- file: `examples/models/vastloom-tinychat-v1.sgtm`;
- size: 367,520 bytes;
- SHA-256: `9e3dd63d2df11cbbf0207dae5f87ee6afa3f5b456aacc7e0b984b515e9cff9ef`.

Two independent exporter processes reproduced the same file byte for byte. Full manifest
verification checked one chunk and all 367,520 bytes with no missing chunks, size mismatches, hash
mismatches or unsafe paths. The final one-shot Italian smoke prompt returned `2+2 fa 4.` with
`finish_reason=end_of_turn`.

An installed-package consumer successfully used `find_package(Vastloom CONFIG REQUIRED)`, linked
`Vastloom::Core` and reported version `0.4.0-alpha.4`. The staged installed CLI passed the bundled
five-vector self-test.

## Release artifacts

Local CPack output was regenerated after the final MSVC test run:

| Artifact | SHA-256 |
|---|---|
| `vastloom-0.4.0-alpha.4-win64.zip` | `faa43a91ada65db3b4df56b00cd1ddde2e2d6a26185923d7c7cedff8f7a9cc1a` |
| `vastloom-0.4.0-alpha.4-win64.tar.gz` | `8e7400cd8745a60274105ef1ec177a94d866e12fac5876dacc9d4d73b72fb60e` |

The archives and generated checksum sidecars are build outputs and remain outside Git.

## Visual assets

| Asset | Dimensions | Size | SHA-256 |
|---|---:|---:|---|
| `vastloom-hero.png` | 1983 x 793 | 2,053,440 bytes | `c4d3dae549d4075d5664b07e5a44fd794d579dbc2629d487269f1ba785d3bb4b` |
| `vastloom-social-preview.png` | 1774 x 887 | 1,907,266 bytes | `6cbface6b8e3ab35e3884771afea9d789b2dd0f8f689c85978dd56f18d2044f1` |
| `vastloom-social-preview-1280x640.jpg` | 1280 x 640 | 226,508 bytes | `20822c7cb2bd004d15ea34f1b0a31fb83ed32a4894cf5590b4c1ea4073bbffa8` |

All three assets were inspected after generation. The exact creative briefs and intended use are
versioned in `docs/BRANDING.md`.

## DeepSeekV4SSD research outcome

The upstream project is useful evidence for Vastloom's thesis, but it is not a compatible backend.
Its validated path targets Apple Silicon, MLX/Metal and 64 GiB unified memory; Vastloom's reference
target is Windows, CUDA, 12 GiB discrete VRAM and 32 GiB RAM.

Two bounded experiments were promoted to the research roadmap:

1. trace-driven ready-expert scheduling that computes completed experts while later reads continue,
   but preserves the model's selected-order reduction;
2. split OLMoE reads using the existing contiguous 8 MiB gate/up region followed by the 4 MiB down
   region, with identical-logit and reservation gates before any native adoption.

Layer-major prefill and prefill-to-decode cache handoff remain later experiments. Fixed Mac-derived
worker counts, cache sizes and warm-page-cache throughput are explicitly not adopted.

## Performance

No new large-model throughput claim was produced by this release. TinyChat runs entirely in CPU
cache and its token rate is not representative. The real-model tests are correctness and bounded-I/O
gates, not a DeepSeek benchmark. No optimization from DeepSeekV4SSD has been implemented yet.

## Known limitations

- DeepSeek V4 Flash, CUDA inference and NVMe-to-VRAM overlap are not implemented.
- The generated GitHub Actions and CodeQL workflows have not run on a public remote yet.
- No Git remote, tag, GitHub release or uploaded repository social preview was created during this
  local validation.
- ThreadSanitizer could not execute in the available WSL runtime.
- The Windows package was generated locally; release-workflow artifacts remain unverified until the
  tag workflow runs on GitHub.
- Physical NVMe device counters, cold-cache controls and useful large-model tokens per second remain
  open measurement work.

## Next bottleneck

The next safe vertical slice is readiness telemetry and a trace replay simulator, followed by the
bounded 8 MiB + 4 MiB OLMoE staged-read experiment. Only a measured improvement with identical
outputs and no weakened resource reservation should graduate into the native runtime. The later
DeepSeek adapter still requires inspection of one exact official checkpoint contract before any
model-specific implementation begins.
