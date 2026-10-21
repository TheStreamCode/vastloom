# Contributing to Vastloom

Thank you for helping build safer local inference for models that exceed conventional
memory limits. Vastloom is experimental systems software: correctness, reproducibility
and host safety take precedence over benchmark headlines.

Active development concluded in October 2026 and the repository is preserved as a working
record, so new feature work is not planned. Defect reports and focused documentation fixes
remain welcome.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). For usage
questions, see [Support](SUPPORT.md). Report vulnerabilities through the private process
in [Security](SECURITY.md), never through a public issue.

## Before opening a change

- Search existing issues and pull requests.
- For a substantial architecture change, open a design issue first and describe the
  safety boundary, correctness oracle and measurement plan.
- Keep original model weights immutable. Generated or optimized forms must use temporary
  output, validation, checksum and atomic publication.
- Never commit third-party model weights, credentials, private prompts or machine-specific
  model-store paths.

## Local development

Prerequisites are CMake 3.24+, a C++20 compiler, Python 3.10+ and Ninja for the supplied
presets. A standard validation build is:

```bash
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
python -m unittest tests/test_manifest_tool.py -v
```

On Windows, run the same commands from PowerShell with Visual Studio Build Tools or Ninja
available. The project also provides `scripts/build.ps1` and `scripts/build.sh`.

Before submitting C++ runtime changes, run Debug and warning-as-error builds. On Linux,
also run the ASan/UBSan preset when supported:

```bash
cmake --preset debug && cmake --build --preset debug --parallel && ctest --preset debug
cmake --preset asan && cmake --build --preset asan --parallel && ctest --preset asan
```

External real-model tests are opt-in through CMake paths: `VASTLOOM_QWEN38_MODEL` (the
primary Qwen3.8 track), `VASTLOOM_SMOLLM2_MODEL`, `VASTLOOM_OLMOE_MODEL` and
`VASTLOOM_OLMOE_FULL_MODEL`, the `VASTLOOM_DEEPSEEK_V4_*` family, and `VASTLOOM_GGML_ROOT`
(the external ggml checkout for the opt-in SIMD backend). Do not download large
models merely to satisfy a pull-request check.

## Pull requests

Keep changes small and coherent. A pull request should include:

- the problem and the chosen approach;
- tests added or changed, with exact results;
- hardware and model revision for performance or parity claims;
- peak RAM/VRAM, NVMe bytes per generated token and cache policy for placement claims;
- known limitations and negative results;
- documentation updates for observable behavior or architecture decisions.

Every important feature needs deterministic correctness tests and malformed-input or fault
tests proportional to its risk. Do not call simulated traces real-model benchmarks, and do
not claim a backend or model works until it has been built and executed.

## Engineering invariants

1. Preserve exact model semantics unless a quality-affecting mode is explicit and measured.
2. Reserve large allocations through the Resource Governor before allocating.
3. Keep queues, reads, staging buffers, caches and speculation bounded.
4. Preserve operating-system headroom; pressure reduction outranks throughput.
5. Keep the token-generation hot path native. Python is for conversion, validation and analysis.
6. Keep common code portable; Windows Unicode paths and Linux support are both required.
7. Add dependencies only when their cost, license and maintenance burden are justified.

## Style and commits

Follow the surrounding C++ and Python style, keep public interfaces documented, and avoid
unrelated formatting churn. Use Conventional Commits (`type(scope): summary`, English,
imperative mood) such as `docs: clarify the admission gate`, and keep one logical change per
commit. Contributions are accepted under the repository's Apache-2.0 license.
