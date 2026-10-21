# DeepSeek V4 Flash inspection and reference foundation

The DeepSeek V4 Flash track is historical: its execution profile (`UD-IQ2_XXS`) was the only
execution quantization, and the newer metadata and partial-range paths below never reached an
official-payload layer, token or generation. Current execution status by profile is in
[`MODEL_COMPATIBILITY.md`](MODEL_COMPATIBILITY.md); the reports under
[`reports/`](reports/README.md) hold the milestone evidence.

## Preview metadata and reference foundation

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
- **Bounded reference generation:** the native driver now composes prefill and incremental state
  into an eight-token greedy continuation that exactly matches a freshly regenerated pinned
  `model.py` oracle. Seeded top-k sampling uses portable `mt19937_64` bits; stop/context/max-token,
  timeout and cancellation finish reasons are explicit. Checkpoints run between layers and before
  state publication, and generation telemetry sums only observed expert-cache accesses.
- **Large-download interlock:** a full import of at least 20 GiB requires the additional
  `--allow-large-download` flag after storage headroom has been reviewed.
- **Honest boundary:** the full-checkpoint inspector still reports
  `execution_supported=false`. The reduced graph now executes 11 Q8_0 trunk/shared matrices,
  eight IQ1_S routed W1/W3 matrices and four IQ3_XXS routed W2 matrices in its first main layer,
  plus an I32 hash table; remaining fixture tensors stay FP32. The packed layer binds file-backed
  trunk/expert loader buffers without payload copies into the graph. Its router-first provider
  loads only deduplicated selected complete mixed-format bundles through a bounded RAM cache and
  emits exact access telemetry. Canonical 0731 GGUF identities can feed one real
  IQ1_S/IQ3_XXS selected expert
  through that same cache/consumer boundary with exact miss/hit accounting and retained leases.
  An optional one-to-four-worker batch submits all selected misses and acquires each result at its
  fixed numerical compute turn. A controlled reduced fixture proves a positive read-service/
  compute intersection without changing the oracle, and the real expert passes the same async
  boundary. This is scheduling correctness, not a physical-device speed result.
  A verified complete official-payload layer/token consumer, official-payload generation,
  target-verified speculative generation, useful official-payload overlap and CUDA inference
  kernels remain future gates.
- **Official 0731 GGUF catalog:** the separate Unsloth revision
  `fbbb5b93fb787c21338159b0af3318bb3f4d9768` now has a bounded GGUF v3 header reader and
  DeepSeek-specific packed-expert mapper. The reviewed `UD-IQ1_S` split exposes 1,328 physical
  tensors, 34,223 logical tensors and 11,008 layer/expert bundles.
- **Verified metadata-to-graph contract:** only the exact pinned manifest/spec may retain the
  bounded 46-value compression array and separate 43-value routed/shared SwiGLU clamp arrays.
  It derives the native 43-layer, width-4,096, 64-head, 256-expert/top-6 configuration and checks
  all 1,193 official non-routed layer descriptors, storage types, consumer dimensions and
  6,390,088,008 encoded bytes. The exact layouts are 21 objects for uncompressed/hash layers,
  31 for 4:1 compressor/indexer layers and 25 for 128:1 compressor layers. This validator performs
  no tensor allocation or payload read.
- **Complete checkpoint catalog closure:** the same verified headers now bind all six global
  consumers and every routed bundle/slice to exact native identities, shapes, roles and encoded
  layouts. The 1,328 physical entries partition into 1,193 non-routed layer tensors, six globals
  and 129 packed expert containers. The logical view contains 34,223 descriptors: 1,193 layer
  tensors, six globals and 33,024 slices across 11,008 bundles. Always-active 6,986,088,796 bytes
  plus routed 75,547,803,648 bytes equals 82,533,892,444 known tensor bytes. This catalog gate
  itself performs layout arithmetic only. Later parity gates now enable scalar Q4_K, IQ2_XXS and
  MXFP4: Q4_K feeds the bounded packed embedding/output path, while alternate reduced MoE graphs
  execute IQ2_XXS routed gate/up matrices or MXFP4 routed down matrices. All three formats have
  bounded row lookup and transactional matvec dispatch.
- **Governed six-global binding:** `deepseek-gguf-global-plan` derives the exact six official
  ranges from the closed catalog. The dedicated `globals` importer scope preserves the same
  explicit network authorization, byte cap, disk reserve and atomic immutable publication rules.
  Native rebinding replays the plan against the pinned source identities, and the loader reserves
  the complete packed destination plus only the largest range as staging before any allocation or
  read. The fixture binds embedding, output head, output norm and all three output
  Hyper-Connection tensors from one shared lease. The official 596,000,788-byte plan passed
  dry-run only.
- **Header-derived traffic floor:** the encoded routed experts total 75,547,803,648 bytes;
  a top-6 no-cache decode would request 1,770,651,648 logical expert bytes per token before
  caching/prefetch gains. This is not measured physical SSD traffic or a speed claim.
- **First complete real quantized expert:** exact gate and up IQ1_S slices plus the IQ3_XXS down
  slice total 6,488,064 bytes. Canonical layer/expert tensor names are converted from the shared
  catalog into strict W1/W2/W3 descriptors. A one-worker loader and the production one-entry
  router-first cache reserve and retain the complete indivisible bundle; the bounded scalar CPU
  path executes SwiGLU and matches pinned ggml for all 4,096 final outputs bit for bit. The first
  cache resolution reports one miss and 6,488,064 logical bytes, while the second reports a hit
  and zero bytes. This is one extracted official expert, not a complete official-payload layer,
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
- **Checkpoint-shaped layer range plan:** `deepseek-gguf-probe-plan` joins one complete trunk plan
  to exactly top-k unique complete expert bundles without reading payload. On layer 0 the real
  metadata yields 21 trunk plus 18 expert ranges, 177,795,544 logical bytes total, a maximum range
  of 35,651,584 bytes and a 6,488,064-byte single-tier expert floor, all in the immutable second
  shard identity. The current IDs are caller-supplied and explicitly unverified; the official
  hash-route table must execute before these offsets can be treated as an authoritative token
  route.
- **Bounded partial artifact import:** `tools/import_hf_ranges.py` accepts only the pinned plan and
  model identity, defaults to a 256 MiB total cap and 300 GiB free-space reserve, requires explicit
  `--allow-range-download`, validates exact HTTP 206/content ranges and publishes per-range hashes
  atomically under `models/optimized`. Current expert/all scopes fail before network access because
  routing authority is false. The real layer-0 trunk has passed dry-run only; no official payload
  range has been downloaded.
- **Native range-artifact binding:** `range-artifact-inspect` parses the derived artifact under
  fixed manifest/range/payload bounds, rejects symlinks, traversal and unexpected files, and
  rehashes every local range by default. `deepseek-gguf-range-bind` then reproduces the trunk plan
  from hash-verified pinned metadata and matches every local byte span to its exact complete-shard
  path, size and SHA-256 identity. The result exposes offset-zero local tile views but still reports
  `complete_checkpoint_verified=false` and `execution_supported=false`; no official payload was
  acquired or executed for this gate.
- **Governed bound-range load:** one complete F32 range from the synthetic binding now reserves its
  full RAM destination and staging together, performs an exact bounded read, hashes the in-memory
  bytes again, publishes `RAM_READY` and remains protected by a retained use lease while the scalar
  matvec consumes it. Allocation refusal, early/late cancellation, callback failure, same-size
  corruption, truncation and eviction cleanup fail safely. The absolute per-load ceiling is 1 GiB
  and callers can lower it.
- **Governed trunk tensor set:** a closed set of up to 4,096 independently hashed ranges is
  preflighted before one aggregate residency transaction. Defaults cap the set at 512 MiB total
  and 64 MiB per range; absolute ceilings are 4 GiB and 1 GiB. Missing, duplicate, split,
  wrong-role/type/source or corrupt tensors fail without partial publication. The reduced packed
  catalog now binds all 21 canonical layer-0 identities from one 2,844-byte buffer and one retained
  lease: 11 matrices, nine F32 vectors and one I32 table. A fail-closed 0731 name adapter validates
  their consumer shapes, F32/BF16 vector decoding is capped at 4 MiB of transient FP32 scratch,
  and the mixed-format prefill/decode oracle remains unchanged. This remains synthetic; no
  official trunk payload reaches the graph.

## Inspection commands

These commands read metadata only. Payload acquisition requires the explicit authorization
described in the sections above and in [`REFERENCE_PLAN.md`](REFERENCE_PLAN.md).

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
$plan = Join-Path $env:TEMP 'vastloom-deepseek-layer-0-plan.json'
vastloom deepseek-gguf-probe-plan $model `
  --layer 0 --experts 0,1,2,3,4,5 --context 4096 --batch-size 1 --json |
  Out-File -LiteralPath $plan -Encoding utf8

$globals = Join-Path $env:TEMP 'vastloom-deepseek-globals-plan.json'
vastloom deepseek-gguf-global-plan $model --context 4096 --batch-size 1 --json |
  Out-File -LiteralPath $globals -Encoding utf8

python tools\import_hf_ranges.py `
  --model deepseek-v4-flash-0731-gguf-iq1s `
  --plan $plan --store C:\VastloomModels --scope trunk --dry-run --json

python tools\import_hf_ranges.py `
  --model deepseek-v4-flash-0731-gguf-iq1s `
  --plan $globals --store C:\VastloomModels --scope globals `
  --max-total-mib 569 --dry-run --json
```

The command requires the exact model ID, 40-character revision, paths, sizes and SHA-256 values.
Raw GGUF paths are rejected. A future full import is 82,539,237,792 bytes and additionally requires
`--allow-large-download`; with the default policy it also preserves 300 GiB of free storage. Do not
run that import merely to test the CLI: the real full payload has not yet passed a Vastloom
layer/token execution gate. `--no-verify` is inspection-only and can never create an executable
source binding.

The probe command is also metadata-only. The example IDs `0..5` demonstrate exact layout and
capacity accounting; they are not the route of a real token. Its JSON therefore reports
`plan_only=true`, `execution_supported=false` and `routing_authority_verified=false`. No listed
range is fetched by this command.

The importer example is deliberately a dry-run. A real range transfer also needs
`--allow-range-download`, enforces the selected-byte cap and publishes only a non-executable
derived artifact. The native verifier/binder has passed synthetic corruption and identity tests,
but this is not authorization to add that flag: the real 138,867,160-byte transfer still requires
the user's explicit approval. UTF-8 and UTF-8-BOM plan files are accepted; UTF-16 shell
redirection is not.

The global plan is likewise metadata-only and deterministic: it contains six ranges totaling
596,000,788 bytes, with a 297,861,120-byte maximum. Its dry-run raises only the importer's selected
payload cap to 569 MiB; it still makes no network request. A real transfer requires the same
explicit approval and does not make the artifact executable.

After an explicitly authorized import, the non-executing verification sequence is:

```powershell
vastloom range-artifact-inspect $artifact --json
vastloom deepseek-gguf-range-bind $model $artifact `
  --layer 0 --experts 0,1,2,3,4,5 --context 4096 --batch-size 1 --json

vastloom deepseek-gguf-global-bind $model $globalArtifact `
  --context 4096 --batch-size 1 --json
```

The second command always performs local range hashes and has no `--no-verify` bypass. The expert
IDs must match those used to construct the plan, but remain layout-only until the official router
table runs. Successful output proves exact trunk source binding, not a runnable DeepSeek layer.
