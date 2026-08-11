# Validation matrix

Validation reports are dated snapshots: they record that a claim was true on a given host, at a
given commit, with given inputs. They do not say whether that claim is still checked automatically
today. This matrix answers that second question.

Read it before trusting a present-tense sentence in `README.md`, `AGENTS.md` or `docs/ROADMAP.md`.
Every row states which test enforces a claim, when that test actually runs, and what has to exist on
the machine for it to run at all.

## How coverage is layered

| Layer | What runs | Where |
|---|---|---|
| Always-on | 13 tests with inputs committed to the repository | every local build and every CI job |
| Opt-in model tests | 11 tests gated behind `VASTLOOM_*` CMake paths | only where the operator supplies the inputs |
| Live GPU test | the real CUDA kernel inside `vastloom_cuda_cli` | only on a host with an NVIDIA driver and enough free RAM |

A test that is not configured is silently absent from `ctest`, not reported as skipped. A passing
`ctest` run therefore means "everything configured on this machine passed", never "every claim in
the documentation was re-verified".

## Always-on tests

These need nothing beyond the repository and a C++20 toolchain; the six Python-driven entries also
need a Python 3 interpreter, which CMake detects with `find_package(Python3)`.

| Test | Claim it defends |
|---|---|
| `vastloom_tests` | core, safety, planner, catalog, residency, quantization and graph unit contracts |
| `vastloom_range_artifact` | closed-world range-artifact reader, rehashing and forged-binding refusal |
| `vastloom_cuda_driver` | CUDA driver boundary against the deterministic fake backend, including injected faults |
| `vastloom_cli_tiny_chat` | bundled TinyTransformer answers its registered prompt |
| `vastloom_cli_tiny_self_test` | bundled manifest verification and self-test path |
| `vastloom_cli_tiny_integration` | streaming, JSON and resource-limit CLI behaviour on the tiny model |
| `vastloom_cuda_cli` | `cuda-info` argument contract, structured JSON errors and — when possible — the real RMSNorm kernel |
| `vastloom_hf_importer_offline` | offline import, atomic publication and hash enforcement |
| `vastloom_deepseek_v4_tokenizer` | DeepSeek byte-BPE IDs and chat framing against committed vectors |
| `vastloom_deepseek_v4_operator_fixture` | bounded reference operators against `tests/fixtures/deepseek_v4_operators.json` |
| `vastloom_deepseek_v4_graph_fixture` | reduced prefill/decode/MTP graph against `tests/fixtures/deepseek_v4_graph.json` |
| `vastloom_deepseek_v4_gguf_cli` | GGUF v3 reader limits and malformed-input refusal |
| `vastloom_range_artifact_cli` | range-artifact CLI contract and failure modes |

## Opt-in model tests

Each row lists the CMake variable, the artifact it must point at, and how that artifact is
obtained. Sizes are the bytes that must exist locally, not the size of the published checkpoint.

| Test | CMake variable | Required artifact | How to obtain it |
|---|---|---|---|
| `vastloom_smollm2_reference` | `VASTLOOM_SMOLLM2_MODEL` | pinned SmolLM2-135M-Instruct checkpoint | `tools/import_hf_model.py` |
| `vastloom_olmoe_inspection` | `VASTLOOM_OLMOE_MODEL` | OLMoE metadata-only import | `tools/import_hf_model.py --metadata-only` |
| `vastloom_olmoe_runtime_reference` | `VASTLOOM_OLMOE_FULL_MODEL` | complete 13.84 GB OLMoE checkpoint | `tools/import_hf_model.py` |
| `vastloom_deepseek_v4_inspection` | `VASTLOOM_DEEPSEEK_V4_MODEL` | DeepSeek V4 Flash preview metadata (19,404,391 bytes) | `tools/import_hf_model.py --metadata-only` |
| `vastloom_deepseek_v4_gguf_header` | `VASTLOOM_DEEPSEEK_V4_GGUF_HEADER` | one `UD-IQ1_S` shard header prefix | bounded range download of the shard prefix |
| `vastloom_deepseek_v4_gguf_checkpoint` | `VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT` | metadata-mode 0731 GGUF store (5,343,584 bytes) | `vastloom deepseek-gguf-checkpoint` in metadata mode |
| `vastloom_deepseek_v4_gguf_probe_plan` | `VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT` | same store | same command |
| `vastloom_deepseek_v4_range_import_dry_run` | `VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT` | same store | same command |
| `vastloom_deepseek_v4_iq1s_slice` | `VASTLOOM_DEEPSEEK_V4_IQ1S_SLICE` | 1,638,400-byte gate slice | `tools/fetch_deepseek_test_slices.py` |
| `vastloom_deepseek_v4_iq3xxs_slice` | `VASTLOOM_DEEPSEEK_V4_IQ3XXS_SLICE` | 3,211,264-byte down slice | `tools/fetch_deepseek_test_slices.py` |
| `vastloom_deepseek_v4_real_expert` | the three slice variables together | 6,488,064 bytes total | `tools/fetch_deepseek_test_slices.py` |

The slice helper is explicit-authority: it prints what it would fetch and downloads nothing until
`--allow-download` is passed, then verifies every range against its pinned length and SHA-256.

```bash
python tools/fetch_deepseek_test_slices.py                   # dry run
python tools/fetch_deepseek_test_slices.py --allow-download  # 6,488,064 bytes
```

## What CI does not cover

The GitHub matrix builds and tests six configurations, and all of them run the always-on layer
only. Two categories of claim are therefore never re-checked by CI:

- **Real-model parity.** SmolLM2, OLMoE, DeepSeek metadata, the 0731 GGUF catalog and the real
  quantized expert are only verified where an operator configures their artifacts.
- **Real GPU execution.** No hosted runner exposes an NVIDIA device, so `cuda-info --json` fails
  its probe there and `vastloom_cuda_cli` validates only the argument and error contract. The
  RMSNorm kernel result recorded in `docs/reports/M8_CUDA_RMSNORM_VALIDATION.md` comes exclusively
  from the reference PC.

## Host pressure is not a failure, and not a pass either

Every governed command refuses to start while the host is short on physical RAM, RAM headroom or
disk reserve, and reports `ErrorCode::resource_exhausted` as exit code 75. That refusal depends on
what else the machine happens to be running: during this matrix's own validation an unrelated
20 GiB process made four tests fail on one run and five different tests fail on the next.

Every registered test therefore carries `SKIP_RETURN_CODE 75`, so CTest reports those contracts as
`Skipped`. Python drivers reach the same outcome through `tests/host_pressure.py`, which recognises
only refusals measured against memory that is free at that instant: the three raised by
`make_safe_runtime_policy` before any work begins, and the tiny-model admission gate. VRAM budget
refusals, allocation failures, parity mismatches and malformed output remain failures.

Read the skip count, not just the exit code. `24/24 passed` and `20 passed, 4 skipped` are very
different claims, and only the first one is evidence.

## Last local run of the full matrix

- Date: 2026-08-11.
- Host: Windows 11, Ryzen 9 7900X, RTX 4070 Super, 32 GiB RAM.
- Toolchain: MSVC 19.44, Ninja, Release, warnings as errors.
- Base commit: `30256fe`.
- Configured layers: always-on plus all 11 opt-in model tests.
- Result: **24 registered, 14 passed, 10 skipped, 0 failed.**

The ten skips were not a property of the code. An unrelated application held 12–20 GiB of RAM for
the whole session, leaving 0.3–7.8 GiB free against the Governor's 8 GiB headroom requirement, so
every governed contract refused to start. Earlier runs of the same tree, before the skip mechanism
existed, reported those refusals as four failures on one attempt and five different failures on the
next — the clearest possible demonstration of why the distinction is needed.

What that run does establish, on this host: the always-on layer, the DeepSeek metadata inspection,
the GGUF header contract, and the three pinned real-expert slice tests including the complete
6,488,064-byte IQ1_S/IQ3_XXS expert. What it does not establish: SmolLM2 and OLMoE parity, the
zero-payload GGUF catalog and probe-plan contracts, the range-importer dry run, the TinyChat CLI
paths and the live CUDA kernel. Those need a re-run on an idle host before any of them may be
described as currently verified.
