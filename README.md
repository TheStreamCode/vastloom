<p align="center">
  <img src="docs/assets/vastloom-hero.png" alt="Vastloom routes giant model tensors through VRAM, system RAM and NVMe while protecting the host computer" width="100%">
</p>

<h1 align="center">Vastloom</h1>

<p align="center"><strong>Run giant models. Keep your machine yours.</strong></p>

<p align="center">
  A safety-first native runtime for open-weight models that outgrow conventional VRAM and RAM limits.
</p>

<p align="center">
  <a href="https://github.com/TheStreamCode/vastloom/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/TheStreamCode/vastloom/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/TheStreamCode/vastloom/security/code-scanning"><img alt="CodeQL" src="https://github.com/TheStreamCode/vastloom/actions/workflows/codeql.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-6ee7ff.svg"></a>
  <img alt="Status: experimental alpha" src="https://img.shields.io/badge/status-experimental_alpha-bd7bff.svg">
</p>

Vastloom treats **VRAM → RAM → NVMe** as one model-aware memory hierarchy. Its long-term
goal is practical local inference for giant sparse Mixture-of-Experts models on ordinary
consumer hardware—without uncontrolled swapping, surprise resource exhaustion or destructive
model conversion.

> [!IMPORTANT]
> **Current source — `0.4.0-alpha.4` plus unreleased M6 metadata and tokenizer slices.**
> Vastloom now runs the routed token path of the pinned 13.84 GB
> `allenai/OLMoE-1B-7B-0924-Instruct` checkpoint on CPU while keeping only the
> always-active trunk and a bounded expert cache in RAM. Native logits, top-8 routes
> and routing weights match a Transformers reference. The bounded reader now exposes
> process-I/O deltas without labeling Windows transfer counters as physical NVMe traffic,
> and trace simulation reports exact unique-expert reuse distance. The official DeepSeek
> V4 Flash revision is now pinned, its 46-shard/69,187-tensor metadata is inspectable
> from verified header snapshots, and its tokenizer/basic chat framing match the official
> reference. DeepSeek logits, generation and CUDA are not implemented.

## Why Vastloom exists

Most runtimes ask whether a model fits. Vastloom asks what each tensor is, when it is
needed and which memory tier can serve it safely. The Safety Governor has authority over
the performance scheduler: memory is reserved before allocation, queues and I/O buffers
are bounded, original weights remain immutable, and emergency pressure aborts generation
instead of sacrificing the host.

```text
verified model → safe hardware plan → model-aware placement → bounded inference → telemetry
                     VRAM ↔ RAM ↔ NVMe
```

This is real systems research, not a claim that DeepSeek V4 Flash already runs. The current
vertical slices prove native dense inference, real MoE routing, bounded expert streaming and
measurement on smaller published checkpoints before the runtime scales to that target.

Measured results and limitations are recorded in the
[`Alpha 4 OLMoE report`](docs/reports/ALPHA4_OLMOE_VALIDATION.md) and the
[`Alpha 4.2 async-I/O report`](docs/reports/ALPHA4_ASYNC_IO_VALIDATION.md). The newest
reader, counter and reuse-distance results are in the
[`Alpha 4.3 I/O-observability report`](docs/reports/ALPHA4_IO_OBSERVABILITY_VALIDATION.md).
The rebrand, compatibility, packaging and open-source readiness checks are recorded in the
[`Alpha 4 open-source validation report`](docs/reports/ALPHA4_OPEN_SOURCE_REBRAND_VALIDATION.md).
The newest staged-read and readiness results are in the
[`M5 staged-I/O validation report`](docs/reports/M5_READINESS_STAGED_IO_VALIDATION.md).
The completed physical-counter and cache-bypass gate is documented in the
[`M5 device-I/O validation report`](docs/reports/M5_DEVICE_IO_CACHE_BYPASS_VALIDATION.md).
The first DeepSeek-specific slice is documented in the
[`M6 metadata inspection report`](docs/reports/M6_DEEPSEEK_METADATA_INSPECTION_VALIDATION.md).
Native prompt encoding is documented in the
[`M6 tokenizer validation report`](docs/reports/M6_DEEPSEEK_TOKENIZER_VALIDATION.md).

## Reference target

The first consumer reference machine is:

- AMD Ryzen 9 7900X, 12 cores / 24 threads
- NVIDIA GeForce RTX 4070 Super, 12 GiB VRAM
- 32 GiB DDR5 RAM
- 2 TiB NVMe SSD
- Windows 11, with Linux supported for development and benchmarking

The engine is not hard-coded for that PC. Hardware is probed at runtime and every
allocation plan is derived from detected capacity and conservative headroom.

## What works in the current source

### Real pretrained model path

- **Pinned checkpoint:** `HuggingFaceTB/SmolLM2-135M-Instruct` at revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`, imported outside the repository.
- **Immutable import and verification:** bounded downloads, exact file sizes and
  SHA-256, temporary artifacts and atomic publication into the external model store.
- **Native tokenizer and template:** GPT-2 byte-level BPE, Unicode/Italian prompts and
  the model's exact chat framing.
- **CPU reference inference:** BF16 weights, FP32 activations, RMSNorm, RoPE,
  grouped-query causal attention, SwiGLU and tied output embeddings.
- **Generation and chat:** greedy or seeded top-k sampling, incremental KV cache,
  persistent multi-turn history, streaming, JSON, timeout and Ctrl+C cancellation.
- **Reference parity:** exact token IDs and greedy outputs, with final-token logits
  matching the pinned Transformers reference within the declared tolerance.
- **Fail-closed runtime:** full checkpoint verification by default, Resource Governor
  reservations, live memory-headroom checks, Windows Job Object cap and at most six
  CPU inference threads.

### Real MoE and NVMe-to-RAM path

- **Pinned real MoE:** `allenai/OLMoE-1B-7B-0924-Instruct` revision
  `7f1c97f440f06ce36705e4f2b843edb5925f4498`, three immutable BF16 shards and
  3,219 validated tensors.
- **Non-resident execution:** the 953,421,824-byte always-active trunk is reserved in
  RAM; the 12,884,901,888 bytes of routed experts remain on NVMe and are addressed by
  exact shard/offset identity.
- **Authoritative CPU graph:** Q/K RMSNorm, RoPE, causal attention, full 64-way router
  softmax, top-8 selection and routing-weighted SwiGLU experts.
- **Bounded expert memory:** one 12 MiB active expert, an explicitly reserved cache
  between 12 MiB and 2 GiB, and reservation-backed prefetch buffers constrained by the
  staging governor.
- **Opt-in asynchronous pipeline:** whole-expert mode uses one 12 MiB range and one worker;
  staged mode uses two workers with an 8 MiB gate/up range plus a 4 MiB down range. Both
  retain prefetch depth one or two, bounded queues, timeouts and cancellation checks.
- **Exact telemetry:** token, layer, expert, size, routing weight, tier, hit/miss,
  logical bytes read, submit/queue/ready times, compute windows, selected/reduction order
  and measured service/CPU overlap, atomically published without overwrite.
- **I/O observability:** Windows reports process read operations and transfer bytes from
  `GetProcessIoCounters`; Linux additionally reports the storage-layer `read_bytes`
  counter from `/proc/self/io`. Windows device-level telemetry maps the checkpoint volume
  to `PhysicalDriveN` and samples `IOCTL_DISK_PERFORMANCE`; native Linux accepts only a
  non-virtual `/sys/dev/block` mapping. Every source and scope remains explicit.
- **Safe cache-bypass protocol:** opt-in `direct-unbuffered-v1` reads bypass the operating
  system file cache without flushing global state. The protocol requires async no-cache
  execution, bounded aligned envelopes and a working device counter, otherwise it fails
  closed.
- **Reuse telemetry:** native traces report cold/reused accesses plus mean, P50, P95 and
  maximum exact unique-expert reuse distance.
- **Readiness replay:** `trace-sim` compares fixed-route and theoretical ready-expert
  compute order without changing the model's selected reduction order.
- **Independent parity:** one- and two-token native probes match Transformers 4.45.2
  for greedy token, top logits, every selected expert and every recorded route weight.
- **Safe preflight:** metadata-only checkpoints cannot execute, full SHA-256 is the
  default, resident loading is rejected when the checkpoint exceeds the live cap, and
  timeout/Ctrl+C/headroom checks stop safely between layers.
- **Fault injection:** worker failure, short read, allocation failure, cancellation and
  I/O timeout are exercised through the real OLMoE path and release reservations cleanly.

### Synthetic validation path retained

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

### DeepSeek V4 Flash inspection path

- **Official pinned identity:** `deepseek-ai/DeepSeek-V4-Flash` revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`; the former “0731” name was a
  provisional project alias, not an official checkpoint tag.
- **No payload required:** a 19,404,391-byte metadata import contains the config,
  tokenizer, index and 7,663,144 bytes of exact Safetensors header snapshots instead of
  159,617,149,040 bytes of weight shards.
- **Exact native catalog:** `deepseek-inspect` verifies 46 shard identities, 69,187
  tensor names/mappings, shapes, offsets, dtype totals and the 150,592,290,816-byte
  routed-expert layout.
- **Native prompt path:** the exact Sequence/Split/ByteLevel tokenizer, all 1,283
  added-token definitions and basic system/user/assistant chat framing match the pinned
  Hugging Face tokenizer across multilingual vectors.
- **Large-download interlock:** a full import of at least 20 GiB requires the additional
  `--allow-large-download` flag after storage headroom has been reviewed.
- **Honest boundary:** the inspector always reports `execution_supported=false`;
  graph-fixture logits/routes, DeepSeek kernels, CUDA and generation remain future gates.

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
  holds a live reservation until released. Windows range reads use Unicode-safe native
  file handles and either one bounded 12 MiB range or staged 8 MiB plus 4 MiB ranges.
- **Weight integrity:** strict manifest parser, exact offline manifest generator,
  relative-path validation, file-size checks, SHA-256 verification and symlink-escape
  protection.
- **Model-store separation:** immutable originals, derived optimized weights,
  reconstructable cache, scratch, state and telemetry.
- **Independent watchdog:** observes an atomically published heartbeat and can remain
  observe-only or terminate a stale runtime only when explicitly launched with
  `--enforce`.
- **MoE trace simulator:** compares no-cache, LRU, LFU and static hot-set placement
  while reporting VRAM/RAM hit rates, modeled NVMe bytes per token, PCIe traffic and
  fixed-order versus ready-order timing.
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
  -DVASTLOOM_BUILD_TESTS=ON \
  -DVASTLOOM_WARNINGS_AS_ERRORS=ON
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

A CMake install also copies the validation fixture to `share/vastloom/models` beneath
the selected install prefix.

## Chat with the bundled model

Run the deterministic model self-test first:

```bash
./build/vastloom self-test \
  examples/models/vastloom-tinychat-v1.sgm
```

Start the interactive chat:

```bash
./build/vastloom chat \
  examples/models/vastloom-tinychat-v1.sgm
```

Example session:

```text
Vastloom 0.4.0-alpha.4 — TinyChat CLI
Model: vastloom/tinychat-v1 (128 token context)
Weights: SHA-256 verified
This validation model is stateless: each line is an independent prompt.

you> Quanto fa 2+2?
model> 2+2 fa 4.
```

Run one prompt without the REPL:

```bash
./build/vastloom chat \
  examples/models/vastloom-tinychat-v1.sgm \
  --prompt "Qual è la capitale d'Italia?" \
  --no-stream
```

Machine-readable invocation:

```bash
./build/vastloom chat \
  examples/models/vastloom-tinychat-v1.sgm \
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

## Chat with the real SmolLM2 checkpoint

Initialize the external store and import the pinned checkpoint. Model files remain
outside Git and are published only after size and SHA-256 validation:

```powershell
vastloom store-init C:\VastloomModels --reserve 300GiB
python tools\import_hf_model.py --model smollm2 --store C:\VastloomModels
```

Stores created before the Vastloom rename remain valid: the runtime and importer accept
the legacy marker and import manifest without renaming, rewriting or rehashing them into a
new format. New stores and imports use only the Vastloom identifiers.

Then validate and chat with the imported directory:

```powershell
$model = 'C:\VastloomModels\models\original\HuggingFaceTB-SmolLM2-135M-Instruct--12fd25f77366fa6b3b4b768ec3050bf629380bac'
vastloom model-info $model --json
vastloom self-test $model --json
vastloom chat $model
```

The real adapter supports conversation history and `/clear`, `/stats`, `/model`,
`/help` and `/exit`. Hash verification is enabled by default; `--no-verify` exists
only for repeated local development tests.

## Probe the real OLMoE checkpoint

Import the pinned MoE into the same external store. The importer requires the
configured free-space reserve, downloads with bounded buffers/rate, resumes verified
partials and publishes only after all files match their pinned SHA-256:

```powershell
python tools\import_hf_model.py --model olmoe --store C:\VastloomModels
$model = 'C:\VastloomModels\models\original\allenai-OLMoE-1B-7B-0924-Instruct--7f1c97f440f06ce36705e4f2b843edb5925f4498'
vastloom moe-inspect $model --json
vastloom moe-logits $model --text 'Hello' --cache no-cache --io async --prefetch-depth 2 --trace C:\VastloomModels\telemetry\hello.csv --json

# Opt-in staged gate/up + down pipeline
vastloom moe-logits $model --text 'Hello' --cache no-cache --io async --expert-read staged --prefetch-depth 2 --json

# Repeatable OS-cache-bypass/device-I/O measurement; never flushes global Windows cache
vastloom moe-logits $model --text 'Hello' --cache no-cache --io async --io-cache bypass --expert-read staged --prefetch-depth 2 --json
```

`moe-logits` is a bounded correctness/telemetry probe, not an OLMoE chat or generation
command. Full verification is enabled by default. After an immediately preceding full
verification, `--no-verify` may be used for repeated local probes without rehashing all
13.84 GB. Expert cache capacity is explicitly reserved and capped at 2 GiB.
Asynchronous I/O is opt-in while the baseline remains under study. Whole mode uses one
worker and one 12 MiB request per expert. `--expert-read staged` requires asynchronous
I/O and uses two bounded workers, an 8 MiB gate/up request and a 4 MiB down request without
repacking the immutable checkpoint. Both modes allow prefetch depth one or two and use a
30-second request timeout by default. `--inject-io-fault` is a test-only development option;
it must not be used for normal inference.

`--io-cache bypass` activates the `direct-unbuffered-v1` measurement protocol. It requires
`--io async --cache no-cache`, reserves both the returned buffer and an aligned direct-I/O
envelope, and refuses to run unless the checkpoint maps to a trustworthy device counter.
It does not alter or globally evict the Windows file cache.

The JSON `process_io` and `device_io` objects identify their operating-system source and
scope. On Windows,
`process_read_bytes` is process transfer accounting and `storage_read_bytes` remains
`null`; it must not be interpreted as physical SSD traffic. `device_io` is a whole-device
delta and can include concurrent activity from other processes. `trace-sim --json` reports
exact cold/reused counts, unique-expert reuse distance and, for current 19-field traces,
a bounded fixed-order versus ready-order replay.

## Inspect DeepSeek V4 Flash without downloading the weights

The M6 metadata path downloads only bounded, independently hashed metadata and header
ranges. It is the safe default for architecture work:

```powershell
python tools\import_hf_model.py --model deepseek-v4-flash --store C:\VastloomModels --metadata-only
$model = 'C:\VastloomModels\models\original\deepseek-ai-DeepSeek-V4-Flash--60d8d70770c6776ff598c94bb586a859a38244f1--metadata'
vastloom deepseek-inspect $model --json
vastloom tokenize $model --text 'Ciao!' --chat --json
```

The inspector allocates metadata structures only and never treats the snapshot as a
runnable checkpoint. Do not add `--allow-large-download` merely to try the command: the
complete checkpoint still has no Vastloom inference path.

## TinyChat CLI options

```text
vastloom chat MODEL_OR_MANIFEST
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
when the input is a Vastloom manifest; `--no-verify` downgrades that to quick path and
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
9e3dd63d2df11cbbf0207dae5f87ee6afa3f5b456aacc7e0b984b515e9cff9ef
```

See [`docs/TINY_MODEL_V1.md`](docs/TINY_MODEL_V1.md) for the binary format and tensor
order.

## Other commands

```text
vastloom chat MODEL_OR_MANIFEST
vastloom model-info MODEL_OR_MANIFEST
vastloom self-test MODEL_OR_MANIFEST
vastloom doctor
vastloom inspect MANIFEST_OR_SAFETENSORS
vastloom verify MANIFEST [--full]
vastloom plan MANIFEST
vastloom trace-sim TRACE.csv
vastloom io-smoke FILE
vastloom store-init PATH [--reserve 300GiB]
vastloom store-info PATH [--reserve 300GiB]
vastloom runtime-policy [--storage PATH] [--json]
vastloom tokenize MODEL_DIRECTORY --text TEXT [--chat] [--json]
vastloom real-logits MODEL_DIRECTORY --text TEXT [--chat] [--top N] [--json]
vastloom moe-inspect MODEL_DIRECTORY [--no-verify] [--json]
vastloom deepseek-inspect MODEL_DIRECTORY [--no-verify] [--json]
vastloom moe-logits MODEL_DIRECTORY --text TEXT [--chat] [--top N]
    [--cache no-cache|lru|lfu|static-hot] [--cache-bytes SIZE]
    [--io sync|async] [--io-cache buffered|bypass] [--expert-read whole|staged]
    [--prefetch-depth 1|2] [--io-timeout-ms N]
    [--timeout-ms N] [--trace PATH] [--no-verify] [--json]
vastloom heartbeat-write PATH
vastloom version
```

Examples:

```bash
# Non-invasive hardware report
./build/vastloom doctor --storage /path/to/model-drive

# Inspect and fully verify a model manifest
./build/vastloom inspect examples/models/vastloom-tinychat-v1.sgm
./build/vastloom verify examples/models/vastloom-tinychat-v1.sgm --full

# Generate the verified-metadata research plan for the consumer reference PC
./build/vastloom plan \
  examples/manifests/deepseek-v4-flash.research.sgm \
  --reference-pc \
  --profile balanced

# Compare cache policies on an expert-routing trace
./build/vastloom trace-sim examples/traces/tiny.csv \
  --vram 128MiB \
  --ram 256MiB \
  --policy lru
```

## Safety invariants

Vastloom follows these rules before performance work begins:

1. Never allocate against all installed RAM or VRAM.
2. Never let independent workers infer free capacity independently.
3. Never modify original model weights during inference.
4. Never publish a converted or downloaded artifact before verification completes.
5. Never follow a manifest path outside the declared model root.
6. Never silently change model precision or router semantics.
7. Degrade prefetch, cache and speculation before aborting a generation.
8. Abort a generation before risking an OS freeze, disk exhaustion or corrupt state.

All inference paths bound model dimensions, metadata, file size, context and timeout
before allocating execution buffers. Real-model paths additionally
keep 8 GiB of current RAM availability for the host, cap themselves to half physical RAM,
require 300 GiB free-storage headroom and refuse the load if reservations fail. See
[`docs/SAFETY.md`](docs/SAFETY.md).

## Reference planning result

Using the exact-revision DeepSeek V4 Flash research profile and the target PC, the balanced
planner currently reserves:

| Budget | Planned value |
|---|---:|
| Windows/host RAM reserve | 6.40 GiB |
| Emergency RAM reserve | 2.00 GiB |
| Maximum Vastloom RAM | 23.04 GiB |
| VRAM reserve | 1.56 GiB |
| Maximum Vastloom VRAM | 10.22 GiB |
| Storage free-space reserve | 100 GiB |

These are planning outputs, not performance claims. The DeepSeek manifest is now derived
from an exact checkpoint/header catalog but remains metadata-only because execution has
not passed the fixture and reference-parity gates.

## Current limitations

Alpha.4 proves one real dense path and one real CPU MoE streaming path with an
opt-in bounded asynchronous reader, not the final Vastloom research thesis.
It currently has:

- two executable pinned Hugging Face adapters plus one DeepSeek metadata/header and
  tokenizer adapter; no DeepSeek inference, MiniMax, Kimi or GGUF execution adapter;
- no CUDA, ROCm, Vulkan or Metal inference backend;
- no production quantized kernels;
- no VRAM expert tier or pinned host buffers; direct/unbuffered I/O is currently an
  opt-in OLMoE measurement protocol, not the production path for every adapter;
- OLMoE currently exposes final-logit probes and route telemetry, not decoding/chat;
- expert compute remains a synchronous CPU reference implementation; the asynchronous
  reader uses one or two bounded workers and defaults to ordinary buffered file I/O;
- device counters and a repeatable OS-cache-bypass protocol are available where the
  checkpoint has an unambiguous local-device mapping, but whole-device deltas can include
  unrelated traffic and SSD temperature/power telemetry is not implemented;
- no persistent KV reuse between chat turns; history is currently recomputed;
- no local HTTP API yet.

The TinyChat throughput is not representative of a large model because the complete
fixture is smaller than one megabyte and remains resident in CPU RAM.

## Research path

The first production target is DeepSeek V4 Flash-class memory pressure, followed by a
larger MiniMax-class MoE and eventually Kimi K3 as the extreme benchmark. The main
research metric is not merely SSD throughput. It is:

> **How few bytes must be fetched from NVMe for each accepted token?**

The next technical sequence remains incremental:

1. architecture-faithful DeepSeek V4 Flash fixture and reference logit/route parity;
2. CUDA dense, attention, router and packed expert kernels;
3. explicit VRAM/RAM/NVMe placement with pinned buffers and direct/overlapped I/O;
4. adaptive placement, predictive routing and speculative expert deduplication;
5. OpenAI-compatible local server and coding-agent integration.

Detailed acceptance gates are in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Prior art and research baselines

Vastloom is an independent project and currently contains no copied code from the
following projects. They remain important engineering and benchmarking references:

- [Colibrì](https://github.com/JustVugg/colibri)
- [DwarfStar / DS4](https://github.com/antirez/ds4)
- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
- [DeepSeekV4SSD](https://github.com/yanun0323/deepseek_ssd) — reviewed in
  [`docs/research/DEEPSEEK_SSD_REVIEW.md`](docs/research/DEEPSEEK_SSD_REVIEW.md)
- [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)

A future public release must preserve all applicable notices for any code that is later
adapted rather than independently implemented.

## Community and license

Vastloom is available under the [Apache License 2.0](LICENSE). Contributions are welcome;
start with [CONTRIBUTING.md](CONTRIBUTING.md), follow the [Code of Conduct](CODE_OF_CONDUCT.md)
and use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities. Governance,
support scope, release gates and brand assets are documented in [GOVERNANCE.md](GOVERNANCE.md),
[SUPPORT.md](SUPPORT.md), [docs/RELEASING.md](docs/RELEASING.md) and
[docs/BRANDING.md](docs/BRANDING.md).

Third-party model weights are not distributed with the project and remain subject to their
own licenses and terms.
