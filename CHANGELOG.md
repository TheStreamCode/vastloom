# Changelog

## Unreleased

- The accelerator evicts its least recently used matrix instead of refusing the next one, and a
  full-depth token went from 30,280 to 11,639 ms. It had been filling VRAM and then declining
  everything else for the rest of the run — 6,962 of 12,007 matvecs — on the implicit theory that a
  transfer is expensive. The arithmetic says otherwise: a routed expert costs about 0.29 ms to
  upload and 0.88 ms to multiply against roughly 8 ms on twelve CPU threads, so re-uploading it on
  every single use still wins about sevenfold. Refusing work to avoid a transfer that costs a
  seventh of the work is a bad trade. Pinning trunk matrices was the same mistake in reverse and was
  reverted: no matrix deserves permanent space.

- Added dense FP32 and BF16 matvec kernels, chosen by counting declines per storage type rather than
  by assumption. The expected next kernel was Q5_K; the measured ranking put FP32 third and BF16
  fifth, and both are far simpler — FP32 needs no decoding, BF16 is a 16-bit shift. Declines fell
  from 3,432 to 2,375 and both types left the table. No speed is claimed: repeated runs of identical
  code span 11,639 to 14,955 ms per token on this host, and the change does not clear that.

- Layer trunks are streamed through a bounded two-layer window instead of all staying resident, and
  a full-depth pass now needs 2.12 GB of host RAM rather than 7.62 GB. The floor stops scaling with
  depth: a deeper model no longer demands a bigger machine, and a machine with spare memory spends
  it on speed instead of on admission. It pays a second time through the accelerator, whose ceiling
  is derived from what the preflight leaves unspent — 1.80 GB before, 6.06 GB after — which took
  matvecs on the GPU from 1,490 to 4,617 and a token from 42,466 to 30,123 ms, with the same
  `" una grande"` and the same expert traffic. At 16 layers, where VRAM was never scarce, the same
  window is 33% slower: the reloads buy nothing when memory is not the binding constraint.

- Every storage type on the routed expert path now has a CUDA kernel, and a 16-layer generation
  dropped from 13,300 to 2,374 ms per token — 5.60× — producing the same text (`" guaranteedlyahi"`)
  and reading the same 3,963,551,744 bytes of expert payload as the CPU path. IQ3_XXS carried most of
  that: it is the most expensive format to decode on the CPU (9.76 ms against Q8_0's 3.04 for the
  same shape), so it gained the most on a device where the decoding hides behind memory latency. All
  three kernels — Q8_0, IQ1_S, IQ3_XXS — now match the reference at zero error rather than under a
  tolerance, because they accumulate in f64 in the reference's own order. At full depth the same
  kernels take a 43-layer generation from 48,970 to 37,680 ms per token, `complete_model: true`,
  output `" una grande"` on both paths. See the
  [CUDA expert kernel validation](docs/reports/M9_DEEPSEEK_CUDA_EXPERT_KERNEL_VALIDATION.md).

- The accelerator's VRAM residency is now capped by the host memory the preflight left unspent.
  A Windows device allocation carries host commit charge, so resident matrices compete for the same
  budget the weights were checked against: a 43-layer pass with the accelerator at its full 6 GB
  ceiling passed preflight and then died in the expert loader with `bad_alloc`, while the identical
  `--cpu-only` run completed. It now spends only the leftover headroom less a gigabyte for
  allocations the preflight does not model — 3.46 GB at full depth, where it settles for 1.30×
  because 9,457 of 12,007 matvecs are declined for want of VRAM. At 16 layers, where there is room,
  the same kernels give 5.60×. Streaming layer trunks is therefore not only about fitting on smaller
  machines; it is what would free the host memory the GPU needs.

- Fixed two defects that a passing parity gate was hiding, both found because the model's output text
  changed while the gate reported zero error.

  The accelerator indexed its resident VRAM matrices by the host address of the buffer they came
  from. Expert buffers are released as the pass advances, the allocator reuses those addresses, and a
  cache entry then served one expert's weights under another's name. The tell was that the output
  changed *between runs of the same command* — wrong arithmetic is wrong the same way every time.
  Matrices now carry a `cache_identity` derived from what they are (layer, expert and catalog tensor
  for experts; name and byte range otherwise), and no identity means no caching, so a caller that
  omits it loses speed rather than correctness.

  Separately, the parity gates were built from dyadic values — a fixed scale of 0.25, a delta of
  0.125 and a k/256 input — which makes f32 and f64 accumulation agree to the bit and let kernels
  accumulating in `fma.rn.f32` pass while the reference summed in `double`. The gates now vary the
  FP16 exponent per block, fill the mantissa, and use non-dyadic input.

- The GPU is now in the inference path. The graph offers each packed matvec to an optional
  accelerator and falls back to the CPU kernel on refusal, so matrices without a device kernel run
  unchanged. Q8_0 matrices stay resident in VRAM after their first upload — 2.90 GB across a
  24-layer pass — because a decode step revisits the same matrices and copying per launch would cost
  more than the kernel saves. The driver also caches its compiled module per context, holding its
  own context reference so the module cannot outlive it.

  At 24 layers with identical output text: 24,328 ms per token on CPU against 22,246–23,509 ms with
  2,728 matvecs accelerated and 4,899 declined. Far short of the kernel's own 5.4×, and the reason is
  measured: 2.65 ms per accelerated call against 0.558 ms in isolation, because each call still
  pushes and pops the context around three synchronous operations. The round trips, not the kernel,
  are what to fix next.

- Added the first quantized CUDA kernel: a Q8_0 matvec in hand-written PTX, loaded through the same
  toolkit-free driver path as the RMSNorm gate. On the reference RTX 4070 Super a 4096×4096 matvec
  runs in 0.5735 ms against 3.0305 ms for the twelve-threaded CPU kernel — 5.28× — with a maximum
  absolute error of 3.8147e-06 against a 2e-2 gate. Blocks accumulate in the reference's order but in
  FP32 rather than FP64, so this is a tolerance gate, not a parity gate. The kernel is deliberately
  naive: one thread per row, no coalescing, no warp reduction, matrix copied per call. Nothing is
  wired into the graph yet, so the model does not run faster because of it — this establishes the
  kernel and its numerics. `cuda-info --matvec-smoke` reports both timings.

- The complete DeepSeek V4 Flash 0731 model executed a forward pass on the reference PC: all 43
  layers, the checkpoint's own routers, 258 expert bundles and 1,770,651,648 bytes of expert payload
  — exactly the `no_cache_expert_bytes_per_token` figure derived from metadata at an earlier
  milestone. Token 0 produced 129,280 logits and next token 20; weights loaded in 3.97 s and the
  forward took 29.4 s with scalar CPU kernels. The process peaked at 7.17 GB while free host RAM
  never fell below 11.7 GB against the Governor's 8 GiB floor, and everything was released at the
  end. Raising depth in steps exposed two defects: the indexer's head-weight projection was bound
  outside the indexer namespace, which only surfaces at layer 2 where the first 4:1 compressor
  appears, and the reference layer ceiling was still sized for reduced fixtures at 16. See the
  [complete forward validation](docs/reports/M9_DEEPSEEK_COMPLETE_FORWARD_VALIDATION.md). This is one
  forward pass, not generation, and 29.4 s per token is correct rather than usable.

- A completed checkpoint verification now leaves a receipt, so the 82.54 GB SHA-256 pass that gates
  execution is paid once rather than before every prompt. The receipt records each file's size and
  last-write time beside the hash that was confirmed, keyed on the manifest's own hash; if both
  still match, the bytes on disk are the bytes that were verified. This trusts filesystem metadata,
  which is weaker than rehashing — a file edited in place preserving size and timestamp would pass —
  and it is bounded: any mismatch, a missing or malformed receipt, or a changed manifest forces the
  full verification again, and the receipt is written only after every file passes.

- Inference now uses half the machine's logical processors rather than a fixed six, which predated
  any threaded kernel. Combined with the threaded matvecs, generation went from 60,301 ms per token
  serial to 32,730 at six threads and 25,891 at twelve, with byte-identical output throughout.

- Packed matvec rows now split across the Governor's thread budget, taking generation from 60.3 s to
  32.7 s per token on the reference PC with byte-identical output. Rows are independent and each is
  still computed exactly as it would be serially, so the ggml parity fixtures are unaffected. The
  gain falls short of the thread count because only the matvecs are threaded — attention,
  normalization, RoPE and the Hyper-Connection paths remain serial, which is where the next
  speedup lives.

- **The complete model generates coherent text.** `"La capitale d'Italia e"` produced
  `" una grande citta cosmopol"` through all 43 layers of the official checkpoint — six tokens at
  60.3 s each, 16 GB of expert payload read from NVMe, 7.62 GB resident, free host RAM never below
  10.3 GB. The same prompt at 12 layers returns gibberish, which is the clearest evidence that the
  full-depth path is executing the real model rather than a fixture. `--max-tokens` routes through
  the reference decode loop with KV reuse; a minute per token is correct, not practical.

- The model now answers a text prompt. The checkpoint's vocabulary proved identical to the pinned
  tokenizer — 129,280 tokens index for index, all 127,741 merges — so `--prompt` encodes through the
  implementation already validated against Hugging Face, with `--chat` applying the checkpoint's own
  template. `"La capitale d'Italia e"` ran all 43 layers, resolved 1,069 expert bundles across
  7.34 GB of payload and predicted `" una"`. Each layer now releases its experts when the next layer
  asks for its own, so peak expert residency is two layers' worth instead of 43: the same prompt
  would otherwise have needed 13.6 GB of experts alone, and now the whole pass fits in 7.62 GB with
  free host RAM steady at 12 GB.

- Added `deepseek-forward`, which runs the real graph to a chosen depth straight from a complete
  checkpoint, and refuses in advance what will not fit. Its preflight sums the exact resident cost
  from the catalog — layer trunks, globals and the expert-cache ceiling — and compares it against the
  live safe RAM budget before allocating anything. On the reference PC a full 43-layer pass needs
  7,056,343,388 bytes against an 8,053,104,640-byte budget while 8 GiB of host RAM stays
  untouchable. Layer trunks and globals stream one complete tile per tensor under their own
  reservations, each layer keeps its own bounded expert cache so one layer cannot evict another's
  working set, and the process guard is sampled at every layer boundary. `--layers` keeps depth
  explicit, and the output states whether the pass was the complete model or a partial one.

- Executed the first real DeepSeek layer from official payload. `deepseek-layer-probe` relocates the
  verified catalog onto the imported artifacts, loads trunk and globals through their governed
  reservations and runs one complete layer with the checkpoint's own router selecting the experts.
  Token 0 executes experts 254, 222, 245, 200, 53 and 35, reads 38,928,384 expert bytes and produces
  129,280 logits in `[-16.0828, 15.4815]` in 1.07 s; two runs agree exactly. Q8_0, Q5_K, Q6_K, BF16,
  I32, IQ1_S, IQ3_XXS and Q4_K all execute together for the first time. This is one layer of 43 and
  the command says so — `complete_model: false` — so these logits are not the model's output.
  Relocation is partial by design: a tensor no artifact provided keeps an absent source and fails on
  open, which is what stops a token whose experts were never imported from executing against wrong
  bytes.

- Imported the first official DeepSeek V4 Flash 0731 weight payload and made the routing decision
  that justifies it. Hash-routed layers select experts by table lookup, so `blk.0.ffn_gate_tid2eid`
  — part of the trunk — determines the selection for a token without executing anything. The
  bounded importer fetched the 21-range / 138,867,160-byte layer-0 trunk, the new
  `deepseek_v4_resolve_hash_route` read that table through the governed rehashed range path, and
  token 0 resolved to experts 35, 53, 200, 222, 245 and 254. The resulting plan reports
  `official-tid2eid-verified`, which opened the previously refused `--scope selected-experts` and
  imported those six complete bundles (18 ranges, 38,928,384 bytes). Learned-router layers,
  out-of-vocabulary tokens, absent or mis-sized tables and out-of-range expert ids all fail closed.
  82.36 GB of the checkpoint remains undownloaded.

- Proved the imported trunk is consumable by the native graph. `deepseek-gguf-range-bind
  --load-tensor-set` rereads all 21 official ranges into one governed RAM buffer, rehashes every
  packed subspan and maps them onto their declared consumers: 11 matrices, 9 vectors and the routing
  table, 138,867,160 resident bytes in 371 ms on the reference PC. Two opt-in tests
  (`VASTLOOM_DEEPSEEK_V4_TRUNK_ARTIFACT`) pin both the route and this consumer contract. The layer
  is bound, not executed.

- Separated environment refusals from defects across the governed test suite. A Safety Governor
  refusal caused by host memory pressure was reported as a test failure, so an unrelated process
  holding 20 GiB made four tests fail on one run and five different tests fail on the next. Every
  registered test now carries `SKIP_RETURN_CODE 75`, the CLI's own `resource_exhausted` code, and
  CTest reports those contracts as skipped. `tests/host_pressure.py` gives the Python drivers the
  same outcome and recognises only the three refusals raised by `make_safe_runtime_policy` before
  any work starts; VRAM budget, allocation, parity and malformed-output failures still fail. The
  two zero-payload GGUF contracts moved from one positional `PASS_REGULAR_EXPRESSION` each to
  explicit per-key JSON comparison of the same 24 and 7 pinned values, so field reordering can no
  longer weaken them.

- Added `docs/VALIDATION_MATRIX.md`, which maps every documented claim to the test that defends it,
  the CMake variable that enables that test and the artifact it requires. It records what the
  13 always-on tests cover, which 11 tests are opt-in, and that neither real-model parity nor real
  GPU execution is re-checked by CI. `tools/fetch_deepseek_test_slices.py` restores the three
  pinned expert slices (6,488,064 bytes) behind an explicit `--allow-download` flag with per-range
  length and SHA-256 verification, so the M7 real-expert evidence is reproducible on a clean
  machine. The eight earlier alpha-era reports are now listed in `docs/reports/README.md` instead
  of remaining unreferenced.

- Added the first real CUDA math path without a toolkit dependency: one Governor-owned packed
  FP32 buffer, synchronous host/device copies and an embedded scalar PTX RMSNorm. The explicit
  `cuda-info --rmsnorm-smoke` gate matches the DeepSeek CPU reference within `3e-5` on the RTX
  4070 Super through native Windows and WSL, while deterministic faults cover copies, launch,
  release and reservation cleanup. This is not yet a CUDA model backend.

- Added a zero-payload complete-catalog contract for the pinned 0731 GGUF checkpoint. It validates
  six global consumers, all 11,008 routed bundles and 33,024 expert slices, then reconciles 1,328
  physical tensors, 34,223 logical descriptors, 6,986,088,796 always-active bytes,
  75,547,803,648 routed bytes and 82,533,892,444 known tensor bytes. Q4_K, IQ2_XXS and MXFP4
  receive layout-only accounting and remain explicitly non-executable without parity kernels.

- Pinned the official `deepseek-ai/DeepSeek-V4-Flash` checkpoint at immutable revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`; removed the unsupported provisional
  “0731” identity from the active planning manifest.
- Added a 19,404,391-byte metadata/header import that uses exact HTTP ranges, pinned
  SHA-256 values and atomic publication instead of downloading 159,617,149,040 bytes
  of weight shards.
- Added `deepseek-inspect` with exact validation for 46 shards, 69,187 tensors,
  Safetensors shapes/offsets, FP8/FP4 storage dtypes and routed-expert byte layout.
- Added native DeepSeek V4 Unicode pre-tokenization, its 129,280-token byte BPE,
  bounded added-token trie and basic system/user/assistant chat framing. Exact IDs match
  Hugging Face tokenizers 0.22.1 across committed multilingual reference vectors.
- Added bounded native DeepSeek reference equations and deterministic vectors for RMSNorm,
  learned/hash routing, routed SwiGLU, Hyper-Connections, base RoPE and selected-latent
  sparse attention. These are operator tests, not a DeepSeek forward or generation claim.
- Added a deterministic 147,136-byte graph exported from the SHA-256-pinned official
  `model.py`, a bounded native two-layer FP32 prefill and one checkpoint-shaped MTP block.
  Main and MTP logits, proposed tokens, learned/hash expert routes and raw/compressed attention
  indices pass the declared parity gate; full-checkpoint and speculative DeepSeek execution
  remain unsupported.
- Extended the fixture to 159,533 bytes with five official single-token decode steps. Native
  main/MTP logits, routes and raw/compressed attention indices match across circular-window and
  overlapping 4:1 cache updates; state mutation is transactional and aggregate-budgeted before
  allocation.
- Recorded, without downloading weights, the newer official DeepSeek V4 Flash 0731 release,
  its Unsloth GGUF profiles and Laguna S 2.1 as separate compatibility targets. None inherits
  the preview checkpoint's execution claims; the universal tensor/expert residency contract
  and per-adapter acceptance ladder are now explicit.
- Added a bounded native GGUF v3 reader with strict metadata/tensor/alignment/offset limits,
  exact GGML encoded-byte accounting, split-header catalogs and header-prefix mode. The CLI can
  inspect one file or build a metadata-only multi-shard DeepSeek catalog without reading payloads.
- Bound the pinned 82,539,237,792-byte 0731 `UD-IQ1_S` profile's 1,328 physical tensors to
  34,223 logical descriptors and 11,008 packed expert bundles from bounded headers. The derived
  1,770,651,648 no-cache expert bytes/token is a planning value, not measured SSD traffic;
  complete quantized execution remains unsupported.
- Added the exact 0731 `UD-IQ1_S` three-object importer and `deepseek-gguf-checkpoint` source
  gate. Metadata-only mode downloads and verifies 5,343,584 bytes; full source ranges require an
  immutable manifest, every complete-file SHA-256 and live single-tier expert admission.
  `--no-verify` remains metadata-only, the official 82.54 GB payload was not downloaded, and the
  CLI still reports `execution_supported=false` until a verified full-checkpoint layer/token
  consumer exists.
- Added `deepseek-gguf-probe-plan` and a native non-executing range-plan API. It validates one
  exact layer, exactly top-k unique caller-supplied expert IDs, complete W1/W2/W3 bundles,
  immutable source identities, bounded range counts and overflow-safe totals. Real layer-0
  metadata yields 21 trunk plus 18 expert ranges / 177,795,544 logical bytes; routing authority,
  full-payload presence and execution remain explicitly false.
- Corrected metadata-only logical sources to name the immutable complete GGUF shard objects rather
  than local `.header` snapshots. Loaders still reject metadata-only catalogs, so this preserves
  accurate future range identity without widening execution authority.
- Added `tools/import_hf_ranges.py` for explicit-authority partial payload acquisition. It accepts
  only the pinned DeepSeek 0731 profile, strictly validates a hash-verified plan, enforces selected
  byte/range/disk bounds, requires exact HTTP 206/content ranges, hashes every payload and
  atomically publishes a read-only `vastloom.range-artifact.v1` under `models/optimized`.
- Kept partial artifacts fail-closed: their manifest declares the complete checkpoint unverified
  and execution unsupported; caller-supplied expert/all scopes are rejected before network access.
  Offline Windows/Linux tests cover exact ranges, malformed plans, explicit authorization, wrong
  HTTP ranges, crash cleanup and immutable reuse. The real 138,867,160-byte trunk passed dry-run
  only and no official payload was downloaded.
- Preserved exact `blk.N.*` identity for all non-routed GGUF tensors and added deterministic
  per-layer trunk plans with a 64 MiB ceiling. The real 0731 metadata maps all 43 layers into
  1,193 tensor tiles; a new governed fixture loader covers immutable identity, Unicode paths,
  budget refusal, cancellation, short reads, leases and zero-reservation cleanup. No official
  weight payload or complete DeepSeek layer was executed.
- Added bounded scalar Q8_0, Q5_K, Q6_K, BF16 and F32 decode/matvec kernels plus a separate I32
  router-table decoder. Synthetic raw blocks match the pinned ggml oracle, caller output is
  transactional through bounded scratch, and the real 0731 metadata now reports native storage
  handling for all 1,193 layer-trunk tensors / 6,390,088,008 encoded bytes. This is storage/math
  coverage only for the official payload: the 82.54 GB payload remains absent and
  `execution_supported=false`.
- Added deterministic ggml-compatible Q8_0 reference encoding and connected packed matrix/I32
  storage directly to the reduced DeepSeek graph. Its 11 trunk/shared matrices use Q8_0, routed
  W1/W3 now use eight IQ1_S matrices and routed W2 uses four IQ3_XXS matrices. All 23 matrices in
  main layer 0 execute through transactional storage dispatch; prefill and incremental main/MTP
  logits, routes and attention indices match an independently dequantized oracle. Official
  payload execution and generation remain open.
- Added immutable resident byte buffers shared by the bounded trunk and expert loaders. A
  file-backed reduced DeepSeek layer now binds 12 complete trunk tiles and four indivisible expert
  bundles directly into the graph, retains exactly 8,180 governed mixed-format payload bytes
  across prefill and decode, blocks eviction while referenced, and returns all reservations to
  zero on release. Official payload execution remains open.
- Split the reduced DeepSeek MoE graph at its authoritative router boundary and added a reusable
  bounded selected-expert cache. The graph deduplicates routed IDs, requests only complete selected
  bundles, executes their retained packed matrices and records hit/miss/logical-byte telemetry. A
  two-entry synchronous RAM LRU loads experts 2/3 on the first token, reuses them on the next and
  replaces them with 0/1 on the third while matching the quantized oracle. Oversized batches,
  malformed providers, insufficient budgets and eviction of in-flight bundles fail closed. This is
  a reduced fixture path, not official-payload DeepSeek generation or an SSD-speed result.
- Added a bounded scalar IQ1_S dequantization/matvec reference kernel derived from pinned ggml
  equations and licensed codebook data. Synthetic vectors and one exact 1,638,400-byte real
  DeepSeek 0731 expert-gate slice match the upstream oracle bit for bit across all 2,048 rows.
- Added exact scalar IQ3_XXS decode/matvec plus a transactional IQ1_S gate/up, FP32 SwiGLU and
  IQ3_XXS down expert path. Three pinned real ranges totaling 6,488,064 bytes now pass through a
  one-worker governed packed-RAM load and match the ggml oracle bit for bit across all 4,096 final
  outputs; complete official-payload layer/token execution and model generation remain open.
- Bound canonical 0731 GGUF expert identities directly from the executable logical catalog into
  strict W1/W2/W3 cache descriptors. The same 6,488,064-byte real IQ1_S/IQ3_XXS expert now passes
  through the production router-first one-entry RAM cache: the first resolution is one exact miss,
  the second is a zero-read hit, retained leases block premature eviction, malformed mixed-format
  descriptors publish no output and every Governor reservation returns to zero. This connects one
  real selected expert to the cache/kernel boundary, not a complete DeepSeek layer or generation.
- Connected selected DeepSeek misses to a bounded asynchronous batch/acquire scheduler. The
  reduced graph submits every deduplicated complete bundle, waits only at that expert's fixed
  compute turn and preserves numerical reduction order while recording load, wait, compute and
  read-service/compute-intersection timing. A controlled fixture proves positive overlap with
  exact oracle parity; the real 6,488,064-byte expert passes the same future/cache/kernel boundary.
  One active batch per cache, full-request queue capacity, drain-on-abandon and exact successful
  wasted-prefetch accounting keep this correctness slice bounded. This is not a physical-NVMe,
  complete-layer or throughput result.
- Made the controlled overlap regression deterministic by explicitly coordinating the second
  fixture read-service start before its bounded 50 ms service window. This changes test timing
  only, not the production scheduler; the native suite then passed 50/50 repetitions and TSan.
- Extended the complete reduced layer/token gate to the real 0731 expert format combination:
  Q8_0 trunk/shared tensors, I32 hash routing, IQ1_S W1/W3 and IQ3_XXS W2. The 8,180-byte
  file-backed fixture produces four 1,384-byte indivisible expert bundles; synchronous and
  asynchronous cache execution matches the dequantized oracle for prefill/decode, and selected IQ
  truncation plus malformed W2 format fail transactionally. These are deterministic reduced
  bytes, not official checkpoint payload or a performance result.
- Documented the tested platform boundary explicitly: Windows/MSVC and Linux GCC/Clang CPU paths
  are validated, while macOS and Metal remain unvalidated targets.
- Added a public reference-PC admission gate: a model must fit one complete expert plus the
  required active block, KV, scratch and staging inside live post-reserve RAM/VRAM budgets. Kimi K3
  is now capacity-study-only on the 32 GiB RAM / 12 GiB VRAM machine unless an exact profile passes
  that gate.
- Implemented that gate in the native planner: strict MoE manifests now require complete expert,
  layer, top-k, context and minimum-working-set metadata; scratch is an explicit placement; and
  each plan names the real RAM or VRAM tier that can contain one complete expert. The Resource
  Governor now enforces one aggregate ceiling across RAM destinations, staging and scratch, with
  deterministic acceptance, pressure, rejection and overflow tests.
- Replaced repeated GGUF metadata `seekg` calls with verified sequential discards. The same real
  5.26 MB header on Windows-mounted WSL storage improved from 37.42 s to 0.11 s while preserving
  strict bounded/truncation behavior.
- Made capacity-sensitive tests deterministic without weakening production disk reserves and
  gated CodeQL/Dependency Review until the repository is public or GitHub Advanced Security is
  explicitly enabled through the `VASTLOOM_ENABLE_GHAS` repository variable.
- Fixed CLI path construction so Windows wide-character arguments remain Unicode-native through
  `std::filesystem`; an accented GGUF filename is now covered end to end.
- Added the shared logical tensor catalog and bound all 69,187 pinned DeepSeek preview tensors
  to role, shape, dtype and immutable source ranges. Its 67,584 routed tensors form 11,264
  main/MTP expert bundles whose six tensors coalesce into exactly two ranges per expert; no
  payload read or execution support is claimed.
- Added the governed multitier residency state machine. RAM plus bounded staging and later VRAM
  destinations are reserved before transition publication; failed compound reservations roll
  back, while pins and RAII use leases prevent unsafe eviction. The state-only slice allocates no
  tensor buffer and makes no new inference claim.
- Added the first governed packed-expert load transaction: canonical source ranges are read into a
  reserved compact RAM buffer, every expert tensor receives an exact packed slice and an RAII use
  lease prevents eviction until the bytes are destroyed. The synchronous fixture baseline is not
  a DeepSeek or physical-NVMe performance claim.
- Added a shared bounded asynchronous expert-load pool with one to four fixed workers, a queue of
  at most 64 requests, pre-enqueue RAM/staging reservations, exclusive transfer leases, safe
  queue-full rollback, shutdown cancellation and queue/service/total telemetry. This enables a
  future compute/I/O overlap path but does not itself prove DeepSeek throughput.
- Added an explicit `--allow-large-download` gate for checkpoint imports containing at
  least 20 GiB of weight payload. DeepSeek execution remains unsupported and is reported
  as such by the CLI.
- Added path-scoped device-I/O snapshots: Windows maps the checkpoint volume to
  `PhysicalDriveN` and samples `IOCTL_DISK_PERFORMANCE`; Linux accepts non-virtual
  `/sys/dev/block/.../stat` mappings and otherwise reports the counter unavailable.
- Added the fail-closed `direct-unbuffered-v1` cache-bypass protocol for bounded OLMoE
  expert reads. It requires async no-cache execution, a trustworthy device counter and
  measured device bytes at least equal to logical expert bytes; it never flushes the
  global operating-system file cache.
- Validated three repeatable Windows cache-bypass runs with identical token `47111`,
  1,610,612,736 logical bytes and 1,610,743,808 whole-device bytes per run.
- Extended native OLMoE traces from 12 to 19 fields with submission, queue, ready,
  compute-window, route-order and reduction-order telemetry while retaining all earlier
  trace formats.
- Added a bounded trace replay that compares fixed-route execution with theoretical
  ready-expert compute order while preserving the authoritative reduction order.
- Added opt-in two-stage OLMoE reads: contiguous 8 MiB gate/up plus 4 MiB down requests,
  two workers, bounded queue/staging reservations, exact byte accounting and down-read
  versus gate/up-compute overlap telemetry.
- Validated staged reads against the pinned Transformers logits/routes and measured a
  46.0% median warm buffered-pipeline improvement over whole 12 MiB reads on three paired
  WSL runs. This is not a physical-NVMe or DeepSeek throughput claim.

## 0.4.0-alpha.4 — 2026-08-09

- Renamed the public project, CLI, C++ namespace, headers, CMake targets and current
  documentation to **Vastloom** with the slogan “Run giant models. Keep your machine yours.”
- Preserved read compatibility for legacy manifests, imported-checkpoint metadata and
  model-store markers so existing verified model downloads remain immutable and reusable.
- Added Apache-2.0 licensing, community governance, support and security policies,
  contribution templates, CodeQL, dependency review and release packaging automation.
- Added a purpose-built README cover and two-to-one social preview, plus public brand and
  release documentation.
- Regenerated the bundled synthetic fixture under the Vastloom model identity and retained
  the original native binary and heartbeat wire magic for format compatibility.

Historical entries below retain the project name and command spelling used by those releases.

## 0.4.0-alpha.3 — 2026-08-09

- Added cross-platform process-I/O snapshots around the real OLMoE forward path.
- Exposed Windows read operation/transfer deltas and Linux logical plus storage-layer
  read deltas while preserving their different semantics in text and JSON output.
- Replaced fragmented Windows asynchronous stream reads with Unicode-safe bounded
  `CreateFileW`/`ReadFile` ranges; one expert now requires one 12 MiB operation.
- Added exact unique-expert reuse-distance telemetry with cold/reused access counts,
  reuse ratio, mean, P50, P95 and maximum distance.
- Added Unicode-path, counter-semantics, operation-fragmentation and real repeated-route
  regression tests.

This slice measures process-visible I/O and routing locality without claiming physical
NVMe traffic. Buffered cache state, cold-cache behavior, device counters, OLMoE decoding,
CUDA and DeepSeek remain outside the validated scope.

## 0.4.0-alpha.2 — 2026-08-09

- Connected the real OLMoE expert path to the reservation-backed bounded I/O pool.
- Added opt-in synchronous/asynchronous selection, one-worker prefetch depth one or two,
  12 MiB request bounds, cancellation and per-request timeouts capped at 60 seconds.
- Coalesced each pinned expert's three validated contiguous BF16 tensors into one bounded
  read while preserving the native gate/up/down layout and reference logits.
- Added request queue/service/total timing plus expert prefetch count, foreground wait
  and measured read-service/CPU overlap telemetry.
- Extended atomic OLMoE CSV traces from nine to twelve fields while retaining compatibility
  with four-field baselines and nine-field legacy traces.
- Added deterministic I/O worker, short-read, allocation, cancellation and timeout fault
  injection, including real-checkpoint fail-closed integration coverage.
- Retained synchronous I/O as the default and capped the asynchronous OLMoE adapter to
  one worker, two outstanding prefetch slots and Resource Governor staging reservations.

This slice validates bounded buffered read/CPU overlap without changing model output. It
does not claim physical NVMe bytes, cold-cache disk throughput, direct I/O, OLMoE text
generation, CUDA or DeepSeek support.

## 0.4.0-alpha.1 — 2026-08-09

- Pinned and fully verified `allenai/OLMoE-1B-7B-0924-Instruct` revision
  `7f1c97f440f06ce36705e4f2b843edb5925f4498` with three immutable BF16 shards.
- Added exact OLMoE configuration, tokenizer, chat-template and 3,219-tensor layout
  validation without requiring tensor payloads to reside in RAM.
- Added a native CPU OLMoE reference graph with RMSNorm, Q/K normalization, RoPE,
  causal attention, authoritative softmax top-8 routing and weighted SwiGLU experts.
- Added explicit NVMe-to-RAM expert streaming: the 953,421,824-byte always-active
  trunk remains in reserved RAM while one 12 MiB expert is read at a time.
- Added bounded no-cache, LRU, LFU and static-hot expert-cache policies, with a hard
  two-GiB cache ceiling and Resource Governor reservations before allocation.
- Added `moe-inspect` and `moe-logits`, atomic route telemetry, exact expert byte/I/O
  accounting, timeout/Ctrl+C checks and metadata-only execution refusal.
- Added an independently bounded Transformers 4.45.2 oracle and committed one- and
  two-token fixtures for logits, greedy token, route IDs and routing weights.
- Extended trace simulation to accept native telemetry and compare no-cache, LRU,
  LFU and static-hot policies.

This milestone proves correct real MoE routing and synchronous NVMe-to-RAM expert
streaming on CPU under a host-derived memory cap. It does not yet provide OLMoE text
generation/chat, CUDA, asynchronous I/O/compute overlap or a DeepSeek adapter.

## 0.3.0-alpha.1 — 2026-08-09

- Added a bounded JSON parser and defensive Safetensors catalog/reader.
- Added a 300 GiB-reserve model store, immutable atomic Hugging Face importer,
  and host-derived real-model runtime policy.
- Added Windows process memory Job Object support and live commit/headroom checks
  for real-model execution paths.
- Added `store-info`, `runtime-policy`, and Safetensors-aware `inspect` commands.
- Added the first genuinely pretrained model adapter, pinned to
  `HuggingFaceTB/SmolLM2-135M-Instruct` revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Added the model's GPT-2 byte-level BPE tokenizer, exact chat template, BF16
  Safetensors loader, Llama CPU reference kernels, grouped-query attention,
  incremental KV cache, greedy/seeded generation and persistent chat history.
- Added trusted Transformers parity fixtures for tokenization, logits, greedy output
  and multi-turn chat, plus a native real-model self-test.
- Added bounded process-memory enforcement, six-thread maximum execution,
  cancellation and full-checkpoint verification by default for the real path.

This milestone proves one real published dense checkpoint on the native CPU path. It
does not yet include MoE execution, NVMe expert streaming or CUDA.

## 0.2.0-alpha.2 — 2026-08-09

- Added the first native end-to-end token-generation path.
- Added the bounded `SGTMV001` TinyTransformer model format and strict loader.
- Added byte-level prompt encoding, CPU LayerNorm/attention/MLP kernels and incremental
  KV caching.
- Added greedy decoding and deterministic seeded top-k/temperature sampling.
- Added `stargate chat`, `stargate model-info` and `stargate self-test`.
- Added interactive streaming, one-shot JSON output, generation timeouts, token limits,
  system-prompt override and response statistics.
- Added the 358.91 KiB `stargate/tinychat-v1` validation model, exact manifest and five
  deterministic response vectors.
- Added a reproducible PyTorch training/export tool that atomically publishes the
  model, vectors and SHA-256 manifest only after all probes pass.
- Added end-to-end CLI integration tests, integrity-failure tests, malformed-model
  tests, deterministic-sampling tests and additional Resource Governor checks.
- Added explicit generation finish reasons and valid-Unicode repair for arbitrary sampled
  bytes in JSON output.
- Added manifest-to-binary metadata cross-checks for model id, layer count and context
  length.
- Added concurrent generation coverage and ThreadSanitizer validation of the shared model
  path.
- Added a UTF-8 Windows `wmain` entry point and UTF-8 console configuration for accented
  CLI prompts.
- Added integration coverage for direct `.sgtm` loading and explicit quick manifest mode.
- Added CMake installation of the bundled validation model under `share/stargate/models`.
- Regenerated the committed fixture from the current exporter and verified byte-identical
  output across independent training runs.
- Fixed model/governor destruction order so weight reservations are released before
  the owning governor is destroyed.
- Added UTF-8 MSVC compilation and expanded warning-as-error/sanitizer validation.

This release validates the native CPU inference pipeline. It does not yet include a
DeepSeek/MiniMax/Kimi adapter, GPU kernels or NVMe-backed generation.

## 0.1.0-alpha.1 — 2026-08-08

- Added C++20/CMake project foundation.
- Added hardware profiler and JSON reporting.
- Added conservative hardware-adaptive execution planner.
- Added atomic resource reservations and safety state machine.
- Added bounded asynchronous I/O with reservation-backed buffers.
- Added strict model manifest, SHA-256 and path/symlink confinement.
- Added immutable model-store layout command.
- Added atomic heartbeat and independent watchdog.
- Added MoE trace simulator with LRU/static-hot policies and bytes/token metrics.
- Added unit tests, warning-as-error build, ASan/UBSan and CI definitions.

No model inference kernels are included in this release.
