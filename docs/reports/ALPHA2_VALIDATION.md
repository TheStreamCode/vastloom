# Alpha 2 validation report

Validation date: 2026-08-09

The project was named Stargate Local AI when this report was written; the commands below are
recorded as they were run. The binary is now `vastloom`, so `stargate <command>` reads
`vastloom <command>` today. Nothing else in this snapshot has been rewritten.

## Scope

Alpha 2 validates one complete native CPU inference path:

```text
verified manifest
  -> bounded TinyTransformer loader
  -> byte-level prompt encoding
  -> causal Transformer + incremental KV cache
  -> greedy or seeded sampled decoding
  -> streaming/JSON/interactive chat CLI
```

The bundled model is a deliberately tiny synthetic fixture. Passing this report proves
that the implemented loader, reference kernels, generation loop, CLI and safety gates
work together for that adapter. It does not prove readiness for DeepSeek V4 Flash,
MiniMax M3, Kimi K3, GPU kernels or NVMe-backed generation.

## Validation host

The container reported:

- operating system: Linux 6.18.35;
- CPU: AMD EPYC 9V74 80-Core Processor;
- logical CPUs exposed to the container: 5;
- CPU features: AVX2, AVX-512F and FMA;
- memory visible to the container: 6,368,813,056 bytes;
- available memory at the recorded probe: 5,681,000,448 bytes;
- NVIDIA GPUs detected: none.

The user's Windows 11 reference PC with Ryzen 9 7900X and RTX 4070 Super was not
available in this environment and was not tested by this report.
The Windows-specific wide-character argument conversion and UTF-8 console path are present in
the source/CI definition but could not be executed inside this Linux container.

## Bundled model identity

- model id: `stargate/tinychat-v1`;
- format: `SGTMV001`;
- size: 367,520 bytes;
- SHA-256: `8adfe9bee5964fc6e62ca0bfd5994e620f97ac0c576c1ecb030851780118a33a`;
- architecture: 2 layers, width 64, 4 heads, FFN width 128, context 128;
- tokenizer: 259-token byte/special-token vocabulary;
- manifest verification: full SHA-256 by default.

## Compiler and runtime matrix

The final source and fixture passed all four CTest cases in each matrix entry:

| Toolchain/configuration | Result |
|---|---|
| GCC Release, project warnings as errors | 4/4 passed |
| Clang 17 Release, project warnings as errors | 4/4 passed |
| GCC Debug, project warnings as errors | 4/4 passed |
| GCC Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 4/4 passed |
| GCC Debug + ThreadSanitizer | 4/4 passed |
| GCC Debug + libstdc++ debug iterators/assertions | 4/4 passed |

AddressSanitizer was run with leak detection and halt-on-error. UndefinedBehaviorSanitizer
and ThreadSanitizer were also configured to halt on the first reported error.

A separate clean Release build was configured in an empty build directory, compiled with
warnings as errors and passed all four CTest cases plus the native self-test.

A GCC `-fanalyzer` build was attempted separately but did not complete within the
available execution window. It is therefore not listed as a passing validation result.

## Native and end-to-end tests

The C++ suite covers:

- strict model loading and configuration checks;
- exact greedy output for the registered fixture;
- streaming callback equivalence with the accumulated result;
- explicit generation finish conditions;
- seeded sampling determinism;
- very small positive temperature handling;
- native five-vector self-test;
- eight concurrent generations sharing one immutable model/governor;
- empty and oversized prompt rejection;
- invalid generation-option rejection;
- truncated model rejection;
- empty model-id rejection;
- non-finite weight rejection;
- insufficient Resource Governor budget rejection;
- model reservation release on destruction;
- JSON escaping of valid UTF-8 and arbitrary invalid byte sequences.

The Python CLI integration suite invokes the built native executable and covers:

- command-specific help;
- `model-info --json`;
- direct `.sgtm` execution and manifest `--no-verify` quick mode;
- all five committed prompt/response vectors, including an accented Italian command-line prompt;
- `self-test --json`;
- the interactive REPL and its `/stats` and `/model` commands;
- deterministic seeded sampling;
- valid JSON/Unicode under high-temperature arbitrary-byte output;
- unknown, missing and duplicate option rejection;
- safe oversized-prompt rejection;
- deliberate model corruption blocked by SHA-256 verification;
- case-insensitive `.SGM` handling;
- rejection of manifest/binary model-id mismatch.

The existing offline manifest-tool tests also passed 2/2, and all Python tools/tests
compiled with `py_compile`.

A deterministic 200-case malformed-model mutation smoke test was also run through the
ASan/UBSan executable. It covered bounded header bit flips, truncation, trailing data and
payload mutations; no sanitizer finding, fatal exception or unexpected process exit occurred.
This is additional defensive testing, not a substitute for a dedicated coverage-guided fuzzer.

## Deterministic fixture reproduction

Two independent 600-step exporter processes were run with the committed seed and
configuration. They produced byte-identical `.sgtm` files with the same SHA-256 shown
above. The committed model was regenerated from the current exporter and compared byte
for byte with those independent outputs.

This establishes reproducibility on the recorded environment. It is not a claim of
bit-identical training across arbitrary PyTorch versions, operating systems or hardware.

## Manual smoke tests

The following final commands completed successfully:

```text
stargate version
stargate doctor --json
stargate model-info examples/models/stargate-tinychat-v1.sgm --json
stargate self-test examples/models/stargate-tinychat-v1.sgm --json
stargate chat examples/models/stargate-tinychat-v1.sgm --prompt "Quanto fa 2+2?" --json
```

The self-test returned all five expected Italian/English responses. The one-shot chat
returned `2+2 fa 4.` with `finish_reason=end_of_turn`.

A staged `cmake --install` was also executed. CMake copied the fixture beneath
`share/stargate/models`; the installed `stargate` binary reported `0.2.0-alpha.2` and
passed the same five-case self-test against that installed, verified manifest.

Tiny-model token rates are intentionally omitted as performance evidence: this fixture
is hundreds of kilobytes, runs entirely on CPU cache and is not representative of a
large-model runtime.

## Explicitly not validated yet

- Windows/MSVC execution on the reference PC;
- CUDA compilation or execution on the RTX 4070 Super;
- a real Safetensors/GGUF adapter;
- DeepSeek V4 Flash, MiniMax M3 or Kimi K3 inference;
- MoE routing and expert kernels in the token path;
- NVMe model paging, direct I/O, pinned memory or SSD-to-GPU overlap;
- persistent multi-turn conversation history;
- OpenAI/Anthropic-compatible local server APIs;
- thermal and power guardrails on physical consumer hardware;
- useful large-model tokens per second.
