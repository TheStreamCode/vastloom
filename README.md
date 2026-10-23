# Vastloom

Run giant models. Keep your machine yours

[![CI](https://github.com/TheStreamCode/vastloom/actions/workflows/ci.yml/badge.svg)](https://github.com/TheStreamCode/vastloom/actions/workflows/ci.yml)
[![CodeQL](https://github.com/TheStreamCode/vastloom/actions/workflows/codeql.yml/badge.svg)](https://github.com/TheStreamCode/vastloom/actions/workflows/codeql.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status: concluded research](https://img.shields.io/badge/status-concluded_research-lightgrey.svg)](#project-status)

![Vastloom routes giant model tensors through VRAM, system RAM and NVMe while protecting the host computer](docs/assets/vastloom-hero.png)

Vastloom is a native C++ research runtime that treats VRAM, system RAM and NVMe as one model-aware memory hierarchy. A Safety Governor reserves memory before allocation, every queue and buffer is bounded, original weights stay immutable, and pressure aborts work instead of risking the host.

The October 2026 track asked one question: how much of a **90.86 GB DeepSeek V4 Flash** checkpoint can run on a **12 GiB consumer GPU** with 32 GiB of RAM, while keeping every byte hash-verified and every result honest? The record: **780,087,788 B of verified range slices**, one real layer and a deterministic three-layer hash-routed chain, reproduced by independent NumPy oracles within **1e-6**.

**New here? Start with the [two-minute showcase](docs/SHOWCASE.md)**, then [`docs/RESULTS.md`](docs/RESULTS.md). This is concluded research, not a product; full-model parity, throughput and a VRAM payload tier are not claimed.

## Contents

- [Project status](#project-status)
- [Install](#install)
- [Quick start](#quick-start)
- [Capabilities](#capabilities)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Privacy and security](#privacy-and-security)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Documentation](#documentation)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)

## Project status

Vastloom was a summer-2026 research experiment in governed local inference for giant Mixture-of-Experts (MoE) models, first on DeepSeek V4 Flash and then on Qwen3.8-Flash-Next. The project concluded in October 2026 and the repository is preserved as a working record: it builds, its test suites pass, and every measured claim links to the report or test that defends it. No further milestones are planned.

The October 2026 refocus returned the **primary research track** to DeepSeek V4 Flash 0731 (`UD-IQ2_XXS`), pursued with bounded exact range slices because the full 90.86 GB checkpoint is not stored. The measured steps cover a metadata-only IQ2_XXS store and verified range imports (780,087,788 B in 45 ranges); one deterministic layer-0 execution (129,280 logits, argmax 30,528); bit-identical device expert residency with bounded CPU fallback; independent layer-0 and three-layer chain oracles (logits within 9.54e-07 and 1.91e-06), including a five-token prefill that opens the layer-2 4:1 compressed window (logits within 1.91e-06, hidden within 5.96e-08); a deterministic 2/3-layer hash-routed chain with opt-in CMake tests and a bit-identical GPU chain; and a design-only D4 on-demand byte budget with closed decisions. Qwen3.8-Flash-Next is the comparison/validation track.

Not claimed: full-model parity, throughput, a VRAM *payload* tier, ratio-128 layers or learned-router/full-depth execution. Those remaining items are not planned in this concluded record. The gate list is in [`docs/ROADMAP.md`](docs/ROADMAP.md#deepseek-refocus-october-2026).

## Install

Vastloom is not published as a package. It is built from source, and the build registers the tests whose inputs live in the repository.

Requirements: CMake 3.24 or newer, a C++20 compiler, Python 3.10 or newer, and Ninja for the supplied presets.

```bash
git clone https://github.com/TheStreamCode/vastloom.git
cd vastloom
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

The helper scripts `scripts/build.sh` and `scripts/build.ps1` run the same sequence for Linux and Windows. After a build, `scripts/demo.sh` (or `pwsh scripts/demo.ps1`) runs the zero-download demo: version, hardware report and the bundled deterministic fixture. A CMake install also copies the bundled validation fixture to `share/vastloom/models` under the selected install prefix.

## Quick start

The bundled TinyTransformer fixture needs no download and is the shortest path to a first successful run:

```bash
./build/vastloom doctor
./build/vastloom self-test examples/models/vastloom-tinychat-v1.sgm
./build/vastloom chat examples/models/vastloom-tinychat-v1.sgm
```

`doctor` reports the detected hardware without touching a model. `self-test` checks the fixture against its deterministic response vectors, and `chat` starts the interactive REPL. Real checkpoints are opt-in: each one is imported into an external model store outside the repository, verified by size and SHA-256, and published immutably. The [usage guide](docs/USAGE.md) covers the DeepSeek reduced-payload probes (primary track), the Qwen3.8 comparison track, SmolLM2 and OLMoE probes, the local server and the full command list.

The loopback server exposes `GET /v1/models` and `POST /v1/chat/completions` on `127.0.0.1` only:

```bash
./build/vastloom serve examples/models/vastloom-tinychat-v1.sgtm --port 8099
```

## Capabilities

- **Primary track: DeepSeek V4 Flash 0731 reduced payload.** `UD-IQ2_XXS` executes from bounded exact range slices imported through the pinned range importer, because the full 90.86 GB checkpoint is not stored: `deepseek-layer-probe` executes one real layer from a metadata store plus three verified artifacts, `deepseek-chain-probe` executes 2/3 hash-routed layers over single- and five-token prompts, independent NumPy oracles reproduce both paths, and opt-in device expert residency is bit-identical to the CPU path without being a VRAM payload tier. See [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) and the [layer-0](docs/reports/D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md), [residency](docs/reports/D2B_DEVICE_RESIDENCY_VALIDATION.md), [layer-0 oracle](docs/reports/D2B_LAYER0_ORACLE_VALIDATION.md), [chain](docs/reports/D3_HASH_CHAIN_VALIDATION.md), [chain oracle](docs/reports/D3B_CHAIN_ORACLE_VALIDATION.md) and [multi-token oracle](docs/reports/D3C_MULTITOKEN_ORACLE_VALIDATION.md) reports.
- **Comparison/validation track: Qwen3.8-Flash-Next.** The official Q8_0 GGUF checkpoint executes through the full 48-layer graph, and greedy generation matches llama.cpp token for token on the pinned prompt. This evidence and the KV/long-context machinery are kept as the external validation reference; no new Qwen features are planned. See the [parity report](docs/reports/Q11C_QWEN38_LLAMA_PARITY_VALIDATION.md).
- **Safety Governor.** Explicit reservations, per-kind and aggregate memory ceilings, lease-protected residency, a pressure ladder, and operating-system-enforced process caps. See [`docs/SAFETY.md`](docs/SAFETY.md).
- **Governed NVMe streaming.** Routed experts stay immutable on disk and move through bounded, hashed ranges into reserved RAM or VRAM; every transition reserves its destination first.
- **Real-model adapters.** SmolLM2 dense CPU inference, OLMoE CPU expert streaming with exact I/O telemetry, and the DeepSeek reduced-payload probes above; a header snapshot is never treated as a runnable model.
- **Native CUDA control boundary.** Toolkit-free driver discovery, Governor-first allocation, one scalar PTX RMSNorm parity kernel and a bounded VRAM-resident matvec accelerator used by the DeepSeek reduced-payload graphs, without a CUDA toolkit or `nvcc`.
- **External-oracle validation.** llama.cpp, built from source against the same checkpoints, is the falsification authority; self-pinned tests alone are not evidence.
- **Measured, not asserted.** Every milestone has a dated validation report under [`docs/reports/`](docs/reports/README.md) that distinguishes simulated traces, warm-cache measurements, process-visible I/O and physical device traffic.
- **Local-only surface.** No telemetry, no remote API; the only HTTP surface is the explicit loopback server.

## Requirements

The validation reference is one consumer PC: an AMD Ryzen 9 7900X (12 cores, 24 threads), an NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM, 32 GiB DDR5 RAM, a 2 TiB NVMe SSD, and Windows 11. The engine is not hard-coded for that machine: hardware is probed at run time and every allocation plan is derived from detected capacity and conservative headroom.

| Platform | Current status |
| --- | --- |
| Windows 11 x86-64 | Native CPU paths plus CUDA driver capability and governed allocation validated with MSVC; primary consumer target |
| Linux x86-64 | Native CPU paths plus CUDA driver capability and governed allocation validated with GCC and Clang; sanitizer jobs pass |
| macOS | Portable CPU and RAM probe only; CPU CI and Metal remain unvalidated roadmap work |

Vastloom therefore works on Windows and Linux today. macOS is a planned portability target, not a validated platform claim.

Windows builds use MSVC with UTF-8 source handling. The executable receives the native wide-character command line, converts it to UTF-8 and configures UTF-8 console input and output, so accented prompts do not depend on the legacy ANSI code page.

Model weights are not distributed with the repository. Real-model tests run only where their checkpoints, slices or NVIDIA driver are present, and they never run in CI. Real-model execution keeps host headroom: the runtime reserves 8 GiB of current RAM availability for the host, caps itself to half of physical RAM, requires 300 GiB of free storage by default and refuses a load whose reservations fail.

## Configuration

There is no configuration file for the engine. Placement and bounds are derived from detected hardware and the explicit command-line policy, and `vastloom runtime-policy --json` reports the effective process and memory policy. External checkpoints live in the model store; the opt-in tests can also receive their local paths through CMake cache variables, which are unset by default:

| Setting | Purpose | Default |
| --- | --- | --- |
| Model store root | External directory holding immutable imports; never inside the repository | Supplied per command |
| Free-space reserve | Storage headroom preserved before an import or execution plan | 300 GiB for real-model paths |
| `VASTLOOM_QWEN38_MODEL` | Imported Qwen3.8-Flash-Next checkpoint for the comparison/validation tests | Unset; tests report SKIPPED |
| `VASTLOOM_SMOLLM2_MODEL`, `VASTLOOM_OLMOE_MODEL`, `VASTLOOM_OLMOE_FULL_MODEL` | Imported dense and MoE checkpoints for their opt-in tests | Unset; tests report SKIPPED |
| `VASTLOOM_DEEPSEEK_V4_*` | DeepSeek metadata, header, range-artifact, oracle-interpreter and chain-artifact inputs for the reduced-payload tests | Unset; tests report SKIPPED |
| `VASTLOOM_GGML_ROOT` | External ggml checkout that enables the opt-in SIMD backend | Unset; scalar kernels stay the default |

The DeepSeek refocus consumes pinned range artifacts imported through `tools/import_hf_ranges.py`, not a CMake model path: the `VASTLOOM_DEEPSEEK_V4_*` entries only point opt-in tests at already-imported local artifacts. Artifacts and metadata stores are local validation data and are never committed.

## Privacy and security

Vastloom processes local models, prompts and hardware metadata. It writes no telemetry and makes no remote request; the only network surface is the `serve` command, which binds to loopback only and transmits nothing on its own. Model imports are a separate, explicit opt-in tool that downloads only pinned HTTPS artifacts, checks declared sizes and hashes, and preserves free-disk headroom before publishing an immutable checkpoint. Original weights are never mutated, and derived artifacts are published only after temporary creation, validation and atomic rename.

Report vulnerabilities through the private process in [`SECURITY.md`](SECURITY.md), never in a public issue. The trust boundaries and the pressure ladder are documented in [`docs/SAFETY.md`](docs/SAFETY.md).

The runtime follows these invariants before any performance work:

- Never allocate against all installed RAM or VRAM.
- Never let independent workers infer free capacity independently.
- Never modify original model weights during inference.
- Never publish a converted or downloaded artifact before verification completes.
- Never follow a manifest path outside the declared model root.
- Never silently change model precision or router semantics.
- Degrade prefetch, cache and speculation before aborting a generation.
- Abort a generation before risking an OS freeze, disk exhaustion or corrupt state.

## Troubleshooting

- **Real-model tests are skipped.** The opt-in CMake cache paths were not set, or the host is short on memory or disk. Governed commands report `SKIPPED` with the reason; read the run output rather than the exit code alone.
- **An import is refused.** Check the pinned revision, the free-space reserve and the declared SHA-256 values. Imports fail closed and publish nothing on a mismatch; a resumed download continues from verified partials only.
- **`cuda-info` cannot load the driver.** The CUDA Driver API is resolved at run time and no toolkit is required, so this usually means the NVIDIA driver is absent or the device is not visible. CPU paths are unaffected.
- **A checkpoint still uses `stargate.*` identifiers.** Legacy stores remain readable by design. New stores and imports use `vastloom.*` identifiers, and an existing checkpoint is never rewritten to rebrand it.

## Development

```bash
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release

cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug
```

The build uses warnings as errors, and the Linux jobs also run ASan and UBSan. The Python offline tools have their own checks:

```bash
python -m compileall -q tools tests
python -m unittest tests/test_manifest_tool.py -v
python -m unittest tests/test_distill_transcript.py -v
python -m unittest tests/test_distill_evaluate.py -v
python tests/test_import_hf_model.py
```

External model tests are opt-in through the `VASTLOOM_*_MODEL` cache paths; never download a large model only to satisfy a check. Benchmark records must state hardware, model revision, cache state and bytes per token, as described in [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md). Contribution rules, engineering invariants and pull-request expectations are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Documentation

- [`docs/SHOWCASE.md`](docs/SHOWCASE.md) — the two-minute overview: what was built, what actually ran, the numbers that matter and the honest limits.
- [`docs/USAGE.md`](docs/USAGE.md) — commands for every track, CLI options and the local server.
- [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) — what each validated track does, with pinned model identities.
- [`docs/RESULTS.md`](docs/RESULTS.md) — key results, limitations, prior art and the research path.
- [`docs/MODEL_COMPATIBILITY.md`](docs/MODEL_COMPATIBILITY.md) — support levels, the universal tensor contract and the acceptance ladder for a new model.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — control plane, data plane, CUDA driver boundary and memory hierarchy.
- [`docs/SAFETY.md`](docs/SAFETY.md) — threats considered, budget ownership and the safety state machine.
- [`docs/REFERENCE_PLAN.md`](docs/REFERENCE_PLAN.md) — the reference-PC admission gate and current planning output.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — milestone status and the acceptance gates that were not executed.
- [`docs/HISTORY.md`](docs/HISTORY.md) — the chronological development log preserved from the README.
- [`docs/DEEPSEEK_V4_FLASH.md`](docs/DEEPSEEK_V4_FLASH.md) — the DeepSeek V4 Flash 0731 reduced-payload track, its measured gates and the dated metadata/reference foundation.
- [`docs/reports/`](docs/reports/README.md) — dated validation reports for every milestone, including the DeepSeek reduced-payload series: the layer-0 range execution, the device-residency and layer-0-oracle reports, and the chain, chain-oracle and multi-token-oracle reports.
- [`docs/planning/D4_ON_DEMAND_EXPERT_RANGES.md`](docs/planning/D4_ON_DEMAND_EXPERT_RANGES.md) — the design-only on-demand expert-range plan and byte budget.
- [`docs/planning/FUTURE_WORK.md`](docs/planning/FUTURE_WORK.md) — the consolidated, dependency-ordered backlog for resuming the research, with entry criteria and explicit non-goals.
- [`docs/research/CONSUMER_STREAMING_INFERENCE.md`](docs/research/CONSUMER_STREAMING_INFERENCE.md) and the [PDF edition](docs/research/vastloom-streaming-moe-inference.pdf) — the systems working paper.
- [`docs/research/SSD_STREAMING_PEER_REVIEW.md`](docs/research/SSD_STREAMING_PEER_REVIEW.md) — the peer review of Kimi K3 SSD streaming, DwarfStar/ds4, Colibri and WARP, with the prioritized I/O and placement improvements.
- [`docs/MANIFEST_V1.md`](docs/MANIFEST_V1.md), [`docs/TINY_MODEL_V1.md`](docs/TINY_MODEL_V1.md), [`docs/RELEASING.md`](docs/RELEASING.md) and [`docs/BRANDING.md`](docs/BRANDING.md) — format, fixture, release and brand references.

## Support

Vastloom is concluded research software, provided without warranty. Use GitHub Issues for reproducible defects, GitHub Discussions for setup questions and research ideas, and the private process in [`SECURITY.md`](SECURITY.md) for vulnerabilities. The support scope and the diagnostic checklist are in [`SUPPORT.md`](SUPPORT.md).

## Contributing

The project has concluded, so new feature work is not planned. Any future change follows [`CONTRIBUTING.md`](CONTRIBUTING.md), the [Code of Conduct](CODE_OF_CONDUCT.md) and the governance model in [`GOVERNANCE.md`](GOVERNANCE.md).

## License

Vastloom is available under the Apache License 2.0; see [`LICENSE`](LICENSE). Copyright and the ggml-derived notice are recorded in [`NOTICE`](NOTICE). Third-party model weights are not distributed with the project and remain subject to their own licenses and terms.

---

Maintained by [Michael Gasperini (Mikesoft)](https://mikesoft.it).
If this project is useful to you, consider [sponsoring its maintenance](https://github.com/sponsors/TheStreamCode).
