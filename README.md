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
> **Current source — `0.4.0-alpha.4` plus unreleased DeepSeek and multitier-control slices.**
> Vastloom now runs the routed token path of the pinned 13.84 GB
> `allenai/OLMoE-1B-7B-0924-Instruct` checkpoint on CPU while keeping only the
> always-active trunk and a bounded expert cache in RAM. Native logits, top-8 routes
> and routing weights match a Transformers reference. The bounded reader now exposes
> process-I/O deltas without labeling Windows transfer counters as physical NVMe traffic,
> and trace simulation reports exact unique-expert reuse distance. The official DeepSeek
> V4 Flash revision is now pinned, its 46-shard/69,187-tensor metadata is inspectable
> from verified header snapshots, and its tokenizer/basic chat framing match the official
> reference. Bounded CPU primitives now cover normalization, routing, expert SwiGLU,
> Hyper-Connections, YaRN, compressed attention and its learned indexer. A deterministic
> two-layer FP32 prefill matches the pinned official graph for logits, greedy token, expert
> routes and attention indices. Its 69,187 tensors are now bound to a shared logical catalog:
> 11,264 main/MTP expert bundles collapse from six tensors to exactly two contiguous source
> ranges each. A governed fixed-worker pool can now reserve, queue and asynchronously load those
> generic multi-range bundles into packed RAM buffers. A bounded native GGUF v3
> reader now also maps the pinned 82.54 GB `UD-IQ1_S` 0731 profile from three headers into 11,008
> logical expert bundles without downloading its complete payload. One complete 6.49 MB real
> layer-0/expert-0 bundle now passes bit-for-bit gate/up IQ1_S, SwiGLU and down IQ3_XXS parity
> against pinned ggml after a governed one-worker load into RAM. The exact three-object GGUF
> distribution now has a bounded immutable importer and a fail-closed source-binding command:
> only a fully hashed payload that passes the single-tier expert gate can expose executable source
> ranges. Bounded scalar Q8_0, Q5_K, Q6_K, BF16 and F32 reference kernels plus an I32 router-table
> decoder now cover the storage types of all 1,193 per-layer trunk tensors in the real metadata.
> The reduced DeepSeek graph now consumes all 23 matrices of its first main layer as Q8_0 plus
> its hash router as I32. Prefill and incremental main/MTP logits, routes and attention indices
> match the independently dequantized packed oracle; the original official FP32 gate remains
> unchanged. These tiny bytes are fixture-owned, not yet bound to Governor-backed file loaders.
> The real 82.54 GB payload has not been downloaded or executed; storage-format coverage is not a
> complete DeepSeek layer/token path, generation or CUDA implementation.

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

This is real systems research, not a claim that the full DeepSeek V4 Flash model already runs.
The current vertical slices prove native dense inference, real MoE routing, bounded expert
streaming on a smaller published checkpoint and one complete real DeepSeek expert before the
runtime scales to a full layer and token path.

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
The first bounded mathematical primitives are documented in the
[`M6 reference-operator report`](docs/reports/M6_DEEPSEEK_REFERENCE_OPERATORS_VALIDATION.md).
The completed reduced graph gate is documented in the
[`M6 reduced-graph validation report`](docs/reports/M6_DEEPSEEK_GRAPH_VALIDATION.md).
The checkpoint-shaped MTP reference gate is documented in the
[`M7 MTP validation report`](docs/reports/M7_DEEPSEEK_MTP_VALIDATION.md).
The five-step raw/compressed cache gate is documented in the
[`M7 incremental-state report`](docs/reports/M7_DEEPSEEK_INCREMENTAL_VALIDATION.md).
The model-independent role/range binding is documented in the
[`M7 logical tensor catalog report`](docs/reports/M7_LOGICAL_TENSOR_CATALOG_VALIDATION.md).
The reservation-backed NVMe/RAM/VRAM control plane is documented in the
[`M9 residency-state report`](docs/reports/M9_RESIDENCY_STATE_VALIDATION.md).
The first governed multi-range-to-packed-buffer transaction is documented in the
[`M9 packed expert load report`](docs/reports/M9_PACKED_EXPERT_LOAD_VALIDATION.md).
The bounded shared worker/queue transaction is documented in the
[`M9 asynchronous expert-load report`](docs/reports/M9_ASYNC_EXPERT_LOAD_VALIDATION.md).
The official 0731 GGUF header and packed-expert binding is documented in the
[`M7 0731 GGUF catalog report`](docs/reports/M7_DEEPSEEK_0731_GGUF_CATALOG_VALIDATION.md).
The first real quantized matrix parity gate is documented in the
[`M7 IQ1_S kernel report`](docs/reports/M7_DEEPSEEK_IQ1S_KERNEL_VALIDATION.md).
The first complete real quantized expert and governed-load gate is documented in the
[`M7 real-expert report`](docs/reports/M7_DEEPSEEK_REAL_EXPERT_VALIDATION.md).
The derived resident/streamed trunk capacity gate is documented in the
[`M7 working-set report`](docs/reports/M7_DEEPSEEK_WORKING_SET_VALIDATION.md).
The first model-independent per-layer trunk plan and governed tile transaction are documented in
the [`M9 layer-trunk streaming report`](docs/reports/M9_LAYER_TRUNK_STREAMING_VALIDATION.md).
The complete layer-trunk storage-type kernel gate is documented in the
[`M9 trunk storage-kernel report`](docs/reports/M9_DEEPSEEK_TRUNK_STORAGE_KERNEL_VALIDATION.md).
The first architecture-level packed layer is documented in the
[`M9 packed-layer report`](docs/reports/M9_DEEPSEEK_PACKED_LAYER_VALIDATION.md).
The pinned importer and immutable full-source admission boundary are documented in the
[`M7 checkpoint-binding report`](docs/reports/M7_DEEPSEEK_GGUF_CHECKPOINT_BINDING_VALIDATION.md).
Current and candidate model support is kept explicit in the
[`model compatibility matrix`](docs/MODEL_COMPATIBILITY.md).

## Reference target

The first consumer reference machine is:

- AMD Ryzen 9 7900X, 12 cores / 24 threads
- NVIDIA GeForce RTX 4070 Super, 12 GiB VRAM
- 32 GiB DDR5 RAM
- 2 TiB NVMe SSD
- Windows 11, with Linux supported for development and benchmarking

The engine is not hard-coded for that PC. Hardware is probed at runtime and every
allocation plan is derived from detected capacity and conservative headroom.

### Platform status

| Platform | Current status |
|---|---|
| Windows 11 x86-64 | Native CPU paths validated with MSVC; primary consumer target |
| Linux x86-64 | Native CPU paths validated with GCC and Clang; ASan/UBSan and TSan gates pass |
| macOS | Source includes a portable CPU/RAM probe; CPU CI and Metal are explicit unvalidated [M14 roadmap](docs/ROADMAP.md#public-macos-and-metal-track) work |

Vastloom must therefore be described as working on Windows and Linux today. macOS is a planned
portability target, not a validated platform claim.

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

### DeepSeek V4 Flash inspection and reference foundation

- **Pinned preview identity:** `deepseek-ai/DeepSeek-V4-Flash` revision
  `60d8d70770c6776ff598c94bb586a859a38244f1` remains the oracle for the evidence below.
  The newer official `DeepSeek-V4-Flash-0731` release is tracked separately and has not
  inherited those validation claims.
- **No payload required:** a 19,404,391-byte metadata import contains the config,
  tokenizer, index and 7,663,144 bytes of exact Safetensors header snapshots instead of
  159,617,149,040 bytes of weight shards.
- **Exact native catalog:** `deepseek-inspect` verifies 46 shard identities, 69,187
  tensor names/mappings, shapes, offsets, dtype totals and the 150,592,290,816-byte
  routed-expert layout.
- **Universal expert identity:** the pinned layout now produces 11,264 logical main/MTP
  expert bundles of 13,369,344 bytes. Exact adjacency safely coalesces their 67,584 tensors
  into 22,528 bounded source ranges, two per expert.
- **Native prompt path:** the exact Sequence/Split/ByteLevel tokenizer, all 1,283
  added-token definitions and basic system/user/assistant chat framing match the pinned
  Hugging Face tokenizer across multilingual vectors.
- **Bounded CPU equations:** deterministic native vectors cover RMSNorm, learned and hash
  routing, routed-expert SwiGLU, Hyper-Connection mixing, base RoPE and selected-latent
  sparse attention, with strict shape/finite checks and an element ceiling.
- **Architecture-level prefill:** a two-layer FP32 fixture exported from the pinned official
  `model.py` exercises a hash router, learned router, shared expert, YaRN, overlapping 4:1
  compression and learned attention indexer. Native logits, greedy token, all routes and
  attention indices pass the declared parity tolerance.
- **Checkpoint-shaped MTP reference:** the same oracle now executes one native MTP block,
  including embedding/hidden preprocessing, Hyper-Connections, learned MoE, sparse attention
  and its own HC output head. Its logits, proposed token, routes and indices match the official
  graph, but no speculative verification loop exists yet.
- **Incremental cache parity:** five official decode steps match for main/MTP logits, tokens,
  routes and indices while exercising the circular raw window, independent attention/indexer
  caches and a newly completed overlapping 4:1 compressed entry. State publication is
  transactional and total cache size is rejected before allocation when over budget.
- **Large-download interlock:** a full import of at least 20 GiB requires the additional
  `--allow-large-download` flag after storage headroom has been reviewed.
- **Honest boundary:** the full-checkpoint inspector still reports
  `execution_supported=false`. The reduced graph now executes all 23 matrices in its first main
  layer from deterministic Q8_0 bytes and its hash table from I32 bytes, while the remaining
  fixture tensors stay FP32. Those packed bytes are fixture-owned: a verified official-payload
  layer/token consumer, target-verified speculative generation and CUDA remain future gates.
- **Official 0731 GGUF catalog:** the separate Unsloth revision
  `fbbb5b93fb787c21338159b0af3318bb3f4d9768` now has a bounded GGUF v3 header reader and
  DeepSeek-specific packed-expert mapper. The reviewed `UD-IQ1_S` split exposes 1,328 physical
  tensors, 34,223 logical tensors and 11,008 layer/expert bundles.
- **Header-derived traffic floor:** the encoded routed experts total 75,547,803,648 bytes;
  a top-6 no-cache decode would request 1,770,651,648 logical expert bytes per token before
  caching/prefetch gains. This is not measured physical SSD traffic or a speed claim.
- **First complete real quantized expert:** exact gate and up IQ1_S slices plus the IQ3_XXS down
  slice total 6,488,064 bytes. A one-worker/one-entry governed loader reserves and packs all three,
  then a bounded scalar CPU path executes SwiGLU and matches pinned ggml for all 4,096 final
  outputs bit for bit. This is one official-payload expert, not a complete official-payload layer,
  routed token or model path.
- **GGUF safety boundary:** `deepseek-gguf-inspect` reads metadata only. Header prefixes cannot
  become executable sources. The real-expert test uses three separately hash-pinned extracted
  ranges; it does not promote a header prefix or unverified full shard into an executable source.
  `deepseek-gguf-checkpoint` accepts only the pinned immutable import identity. A complete import
  must pass full-file SHA-256 and the live single-tier expert admission gate before it can expose
  executable source ranges; even then, `execution_supported=false` until a verified
  full-checkpoint layer/token consumer exists.
- **Bounded layer-trunk plan:** every non-routed `blk.N.*` tensor now keeps its exact layer
  identity. The real 0731 metadata yields 43/43 deterministic plans, 1,193 tensors/tiles and a
  35,651,584-byte largest tile under a 64 MiB cap. A governed fixture proves one immutable tile
  read and fault cleanup. Exact scalar storage handling now covers all 1,193 tensors /
  6,390,088,008 encoded bytes through Q8_0, Q5_K, Q6_K, BF16, F32 and I32 paths. The reduced
  packed layer validates their architecture boundary, but no complete official-payload layer was
  executed.

### Runtime foundation

- **Hardware probe:** CPU ISA, logical cores, RAM availability, NVIDIA VRAM and storage
  capacity.
- **Fail-closed execution planner:** produces VRAM/RAM/NVMe budgets, reserves explicit KV,
  staging and scratch placement, and rejects MoE profiles whose declared minimum working set or
  one complete expert cannot fit a real post-reserve RAM/VRAM tier.
- **Resource Governor:** atomic reservations prevent concurrent workers from
  oversubscribing RAM, VRAM, staging buffers or scratch space. RAM destinations, staging and
  scratch also share one aggregate host-memory ceiling, so their independent limits cannot add
  up beyond the safe process budget.
- **Governed residency:** logical objects move only through legal
  `NVME → READING → RAM_READY → COPYING_TO_VRAM → VRAM_READY` states. RAM, staging and
  VRAM destinations are reserved before work begins; pins and RAII use leases block eviction.
- **Packed expert baseline:** canonical source ranges can be loaded into one reserved RAM buffer
  with exact per-tensor slices through either the synchronous oracle or a fixed-worker,
  fixed-capacity asynchronous pool. The full gate/up/down path is now exercised on one real
  DeepSeek 0731 expert; no DeepSeek layer or token loop consumes these buffers yet.
- **Governed trunk tile:** one non-routed source tile is identity-checked against the catalog,
  destination and staging are reserved before allocation, metadata-only catalogs are refused and
  an RAII lease protects the resulting buffer through use/eviction.
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

The newer 0731 GGUF path can likewise catalog user-supplied complete files or bounded header
prefixes. For three prefixes, their original complete sizes must be supplied in split order:

```powershell
vastloom deepseek-gguf-inspect .\part-00001.gguf .\part-00002.gguf .\part-00003.gguf `
  --declared-sizes 5257664,49093726624,33440253504 `
  --revision fbbb5b93fb787c21338159b0af3318bb3f4d9768 `
  --json
```

This command emits a `metadata_only` logical catalog. It does not download shards, verify a remote
revision, read tensor payloads or authorize inference. Use only exact sizes from an independently
verified artifact manifest; a user-entered revision is reported as a label, not cryptographic
proof.

The preferred 0731 path uses the pinned importer contract. Metadata-only mode transfers exactly
5,343,584 bytes, verifies all three objects and atomically publishes a Windows-safe artifact name:

```powershell
python tools\import_hf_model.py `
  --model deepseek-v4-flash-0731-gguf-iq1s `
  --store C:\VastloomModels `
  --metadata-only

$model = 'C:\VastloomModels\models\original\deepseek-v4-flash-0731-iq1s-fbbb5b93--metadata'
vastloom deepseek-gguf-checkpoint $model --context 4096 --batch-size 1 --json
```

The command requires the exact model ID, 40-character revision, paths, sizes and SHA-256 values.
Raw GGUF paths are rejected. A future full import is 82,539,237,792 bytes and additionally requires
`--allow-large-download`; with the default policy it also preserves 300 GiB of free storage. Do not
run that import merely to test the CLI: the real full payload has not yet passed a Vastloom
layer/token execution gate. `--no-verify` is inspection-only and can never create an executable
source binding.

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
vastloom inspect MANIFEST_OR_MODEL_HEADERS [--declared-size SIZE] [--tensor-limit N]
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
vastloom deepseek-gguf-inspect SHARD... [--declared-sizes CSV] [--revision ID] [--json]
vastloom deepseek-gguf-checkpoint MODEL_DIRECTORY [--context N] [--batch-size N]
    [--no-verify] [--json]
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

The official 0731 `UD-IQ1_S` GGUF catalog now also derives two batch-1, 4,096-token capacity
profiles from exact metadata. The streamed-trunk floor is 124,905,496 bytes including one complete
8,781,824-byte expert; the resident-trunk floor is 7,110,994,292 bytes. This makes the intended
low-memory contract explicit: trunk weights may remain immutable on SSD and move through bounded
tiles, but one expert is never split across RAM and VRAM. The 43 layer plans use 1,193 tiles with
a 35,651,584-byte maximum. Every layer-trunk storage type now has a bounded scalar decoder and
the floating matrix types have transactional reference matvecs. A reduced first layer consumes
fixture-owned Q8_0/I32 storage, but the official-payload, Governor-owned DeepSeek trunk-streaming
token path is not implemented.

## Current limitations

Alpha.4 proves one real dense path and one real CPU MoE streaming path with an
opt-in bounded asynchronous reader, not the final Vastloom research thesis.
It currently has:

- two executable pinned Hugging Face adapters plus a DeepSeek metadata/tokenizer, reduced
  FP32 prefill oracle, official 0731 GGUF metadata catalog and verified source-binding boundary;
  no full-checkpoint DeepSeek, MiniMax, Kimi or file-backed GGUF layer/token execution adapter;
- no CUDA, ROCm, Vulkan or Metal inference backend;
- scalar IQ1_S/IQ3_XXS expert and Q8_0/Q5_K/Q6_K/BF16/F32 trunk reference kernels, plus bounded
  I32 router-table decoding and one deterministic Q8_0 reduced-layer composition, but no
  official-payload complete quantized layer/token, SIMD/fused production kernel or CUDA path;
- complete metadata-derived trunk plans and storage-type coverage with a fixture-only tile loader,
  but no official trunk payload read or tile-to-DeepSeek-operator consumer;
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

The first production target is DeepSeek V4 Flash-class memory pressure, followed only by MoE
profiles whose runtime state fits safe host RAM and whose complete expert bundle fits one real RAM
or VRAM tier. RAM and VRAM are never added together to admit an indivisible expert. Kimi K3 is a
future metadata/capacity stress study, not an execution target for the 32 GiB RAM / 12 GiB VRAM
reference machine unless an exact profile passes that public gate. The main research metric is not
merely SSD throughput. It is:

> **How few bytes must be fetched from NVMe for each accepted token?**

The next technical sequence remains incremental:

1. retain packed tensor views through the governed trunk/expert loader leases and execute the
   completed Q8_0 one-layer fixture from bounded file-backed ranges without unaccounted copies;
2. only after explicit approval and a fresh capacity check, import and fully hash the real pinned
   shards and run the same bounded layer probe on official source ranges;
3. connect a target-verified MTP decoding loop, then add CUDA dense, attention, router and packed
   expert kernels;
4. explicit VRAM/RAM/NVMe placement with pinned buffers and direct/overlapped I/O;
5. adaptive placement, predictive routing and speculative expert deduplication;
6. OpenAI-compatible local server and coding-agent integration.

Detailed acceptance gates are in [`docs/ROADMAP.md`](docs/ROADMAP.md).
The reference-PC working-set rule is defined in
[`docs/REFERENCE_PLAN.md`](docs/REFERENCE_PLAN.md#reference-pc-model-admission-gate).

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
