# Stargate Local AI

**Stargate Local AI** is a safety-first native inference runtime under development for
running open-weight models that are much larger than the host's VRAM and RAM,
especially Mixture-of-Experts models. It treats **VRAM, RAM and NVMe as an explicit
model-memory hierarchy** instead of relying on uncontrolled operating-system swap.

> **Current status — `0.2.0-alpha.2`: native end-to-end validation milestone.**
> Stargate now loads a verified model, tokenizes a prompt, executes a real causal
> Transformer with an incremental KV cache, samples tokens and exposes an interactive
> streaming chat CLI. The bundled TinyChat model is deliberately tiny and synthetic;
> it validates the runtime path but is not a general-purpose LLM. DeepSeek V4 Flash,
> CUDA and NVMe expert streaming are not implemented yet.

The development branch after alpha.2 also contains a bounded Safetensors inspector,
an immutable external model store and a pinned importer for the first real-model
target. These are storage/metadata foundations only; real SmolLM2 generation is not
claimed until its native adapter passes reference parity.

## Reference target

The first consumer reference machine is:

- AMD Ryzen 9 7900X, 12 cores / 24 threads
- NVIDIA GeForce RTX 4070 Super, 12 GiB VRAM
- 32 GiB DDR5 RAM
- 2 TiB NVMe SSD
- Windows 11, with Linux supported for development and benchmarking

The engine is not hard-coded for that PC. Hardware is probed at runtime and every
allocation plan is derived from detected capacity and conservative headroom.

## What works in alpha.2

### Native token-generation path

- **TinyTransformer v1 adapter:** strict binary parser, bounded dimensions, exact
  tensor-layout validation and finite-weight checks.
- **Byte-level tokenizer and chat template:** deterministic UTF-8 byte encoding with
  BOS/EOS/end-of-turn tokens.
- **CPU reference kernels:** LayerNorm, causal multi-head attention, GELU MLP, residual
  connections, tied output embeddings and incremental KV caching.
- **Generation:** greedy decoding and seeded top-k/temperature sampling.
- **Streaming chat CLI:** interactive REPL, one-shot prompts, JSON output, timeouts,
  generation limits, system-prompt override and response statistics.
- **Bundled validation model:** a 358.91 KiB, two-layer causal Transformer trained on a
  small synthetic prompt/response corpus.
- **End-to-end self-test:** five deterministic Italian/English response vectors.
- **Manifest integrity by default:** `.sgm` inputs are fully SHA-256 verified before
  weights are loaded. Direct `.sgtm` files remain available for development and are
  clearly reported as unverified.

### Runtime foundation

- **Hardware probe:** CPU ISA, logical cores, RAM availability, NVIDIA VRAM and storage
  capacity.
- **Fail-closed execution planner:** produces VRAM/RAM/NVMe budgets and rejects plans
  that consume the OS or disk emergency reserve.
- **Resource Governor:** atomic reservations prevent concurrent workers from
  oversubscribing RAM, VRAM, staging buffers or scratch space.
- **Safety state machine:** `NORMAL → PRESSURE → CRITICAL → EMERGENCY`, with immediate
  escalation and hysteretic recovery.
- **Bounded asynchronous I/O:** fixed worker, queue and request limits; every buffer
  holds a live reservation until released.
- **Weight integrity:** strict manifest parser, exact offline manifest generator,
  relative-path validation, file-size checks, SHA-256 verification and symlink-escape
  protection.
- **Model-store separation:** immutable originals, derived optimized weights,
  reconstructable cache, scratch, state and telemetry.
- **Independent watchdog:** observes an atomically published heartbeat and can remain
  observe-only or terminate a stale runtime only when explicitly launched with
  `--enforce`.
- **MoE trace simulator:** compares LRU and static hot-set placement while reporting
  VRAM/RAM hit rates, NVMe bytes per token and PCIe traffic.
- **Linux and Windows CI definitions**, strict compiler warnings and ASan/UBSan jobs.

## Build and validate

### Linux

```bash
./scripts/build.sh
```

Or manually:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTARGATE_BUILD_TESTS=ON \
  -DSTARGATE_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Windows

From PowerShell with CMake and a C++20 compiler:

```powershell
./scripts/build.ps1
```

MSVC builds use UTF-8 source handling. The Windows executable also receives the native
wide-character command line, converts it to UTF-8 and configures UTF-8 console input/output,
so accented prompts are not dependent on the legacy ANSI code page.

A CMake install also copies the validation fixture to `share/stargate/models` beneath
the selected install prefix.

## Chat with the bundled model

Run the deterministic model self-test first:

```bash
./build/stargate self-test \
  examples/models/stargate-tinychat-v1.sgm
```

Start the interactive chat:

```bash
./build/stargate chat \
  examples/models/stargate-tinychat-v1.sgm
```

Example session:

```text
Stargate Local AI 0.2.0-alpha.2 — TinyChat CLI
Model: stargate/tinychat-v1 (128 token context)
Weights: SHA-256 verified
This validation model is stateless: each line is an independent prompt.

you> Quanto fa 2+2?
model> 2+2 fa 4.
```

Run one prompt without the REPL:

```bash
./build/stargate chat \
  examples/models/stargate-tinychat-v1.sgm \
  --prompt "Qual è la capitale d'Italia?" \
  --no-stream
```

Machine-readable invocation:

```bash
./build/stargate chat \
  examples/models/stargate-tinychat-v1.sgm \
  --prompt "Ciao" \
  --json
```

Interactive commands:

```text
/help
/model
/stats
/system TEXT
/reset
/exit
/quit
```

The TinyChat fixture is intentionally stateless. The CLI keeps command state and the
latest metrics, but it does not concatenate conversation history for this model.

## TinyChat CLI options

```text
stargate chat MODEL_OR_MANIFEST
    [--prompt TEXT]
    [--max-tokens N]
    [--temperature F]
    [--top-k N]
    [--seed N]
    [--timeout-ms N]
    [--system TEXT]
    [--no-stream]
    [--no-stats]
    [--no-verify]
    [--json]
```

`--json` is supported for one-shot prompts. Hash verification is enabled by default
when the input is a Stargate manifest; `--no-verify` downgrades that to quick path and
size verification and should be used only during local development.

JSON responses include an explicit `finish_reason` (`end_of_turn`, `end_of_sequence`,
`invalid_token`, `context_length`, `max_tokens` or `timeout`). Arbitrary sampled
bytes are repaired to valid Unicode before JSON serialization.

## Reproduce the validation model

The committed model is generated by `tools/train_tiny_chat.py`. PyTorch is needed only
for fixture training, not for native inference:

```bash
python3 tools/train_tiny_chat.py
```

The tool:

1. trains the deterministic byte-level Transformer;
2. refuses to export if any registered response probe fails;
3. writes the model through a temporary file and atomic replacement;
4. writes the test vectors and exact SHA-256 manifest atomically.

The committed model SHA-256 is:

```text
8adfe9bee5964fc6e62ca0bfd5994e620f97ac0c576c1ecb030851780118a33a
```

See [`docs/TINY_MODEL_V1.md`](docs/TINY_MODEL_V1.md) for the binary format and tensor
order.

## Other commands

```text
stargate chat MODEL_OR_MANIFEST
stargate model-info MODEL_OR_MANIFEST
stargate self-test MODEL_OR_MANIFEST
stargate doctor
stargate inspect MANIFEST_OR_SAFETENSORS
stargate verify MANIFEST [--full]
stargate plan MANIFEST
stargate trace-sim TRACE.csv
stargate io-smoke FILE
stargate store-init PATH [--reserve 300GiB]
stargate store-info PATH [--reserve 300GiB]
stargate runtime-policy [--storage PATH] [--json]
stargate heartbeat-write PATH
stargate version
```

Examples:

```bash
# Non-invasive hardware report
./build/stargate doctor --storage /path/to/model-drive

# Inspect and fully verify a model manifest
./build/stargate inspect examples/models/stargate-tinychat-v1.sgm
./build/stargate verify examples/models/stargate-tinychat-v1.sgm --full

# Generate the provisional plan for the consumer reference PC
./build/stargate plan \
  examples/manifests/deepseek-v4-flash-0731.research.sgm \
  --reference-pc \
  --profile balanced

# Compare cache policies on an expert-routing trace
./build/stargate trace-sim examples/traces/tiny.csv \
  --vram 128MiB \
  --ram 256MiB \
  --policy lru
```

## Safety invariants

Stargate follows these rules before performance work begins:

1. Never allocate against all installed RAM or VRAM.
2. Never let independent workers infer free capacity independently.
3. Never modify original model weights during inference.
4. Never publish a converted or downloaded artifact before verification completes.
5. Never follow a manifest path outside the declared model root.
6. Never silently change model precision or router semantics.
7. Degrade prefetch, cache and speculation before aborting a generation.
8. Abort a generation before risking an OS freeze, disk exhaustion or corrupt state.

The TinyTransformer path also bounds model dimensions, metadata, file size, context,
generation length and timeout before allocating execution buffers. See
[`docs/SAFETY.md`](docs/SAFETY.md).

## Reference planning result

Using the provisional DeepSeek V4 Flash research profile and the target PC, the balanced
planner currently reserves:

| Budget | Planned value |
|---|---:|
| Windows/host RAM reserve | 6.40 GiB |
| Emergency RAM reserve | 2.00 GiB |
| Maximum Stargate RAM | 23.04 GiB |
| VRAM reserve | 1.56 GiB |
| Maximum Stargate VRAM | 10.22 GiB |
| Storage free-space reserve | 100 GiB |

These are planning outputs, not performance claims. The DeepSeek manifest remains a
metadata-only research profile until it is generated from an exact supported
checkpoint.

## Current limitations

Alpha.2 proves the native execution plumbing, not the final Stargate research thesis.
It currently has:

- no DeepSeek, MiniMax, Kimi, GGUF or Hugging Face model adapter;
- no CUDA, ROCm, Vulkan or Metal inference backend;
- no production quantized kernels;
- no VRAM/RAM/NVMe tensor paging during token generation;
- no MoE router or expert execution in the token path;
- no multi-turn history in the bundled fixture;
- no local HTTP API yet.

The TinyChat throughput is not representative of a large model because the complete
fixture is smaller than one megabyte and remains resident in CPU RAM.

## Research path

The first production target is DeepSeek V4 Flash-class memory pressure, followed by a
larger MiniMax-class MoE and eventually Kimi K3 as the extreme benchmark. The main
research metric is not merely SSD throughput. It is:

> **How few bytes must be fetched from NVMe for each accepted token?**

The next technical sequence is:

1. exact DeepSeek configuration and Safetensors inspector;
2. architecture-faithful CPU oracle and tensor fixtures;
3. CUDA dense, attention, router and packed expert kernels;
4. explicit VRAM/RAM/NVMe placement;
5. overlapped storage, PCIe transfer and compute;
6. routing traces and adaptive expert placement;
7. predictive router look-ahead;
8. speculative future-route unions and expert deduplication;
9. OpenAI-compatible local server and coding-agent integration.

Detailed acceptance gates are in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Prior art and research baselines

Stargate is an independent project and currently contains no copied code from the
following projects. They remain important engineering and benchmarking references:

- [Colibrì](https://github.com/JustVugg/colibri)
- [DwarfStar / DS4](https://github.com/antirez/ds4)
- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
- [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)

A future public release must preserve all applicable notices for any code that is later
adapted rather than independently implemented.

## Repository policy

This repository is private research code. The current `LICENSE` grants no redistribution
rights; an explicit open-source license can be chosen before publication.
