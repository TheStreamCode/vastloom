# P0 expert-pack and adaptive I/O validation

Validation date: 2026-10-23. Track: DeepSeek V4 Flash 0731, `UD-IQ2_XXS`, hash-routed layers 0–2
from the existing imported range artifacts. This is a post-conclusion experiment, not a new
milestone: it implements the two P0 items of
[`docs/research/SSD_STREAMING_PEER_REVIEW.md`](../research/SSD_STREAMING_PEER_REVIEW.md) and
records the result, including the parts that did not improve.

## What was built

1. **Derived expert pack** (`vastloom.deepseek-v4-expert-pack.v1`): one contiguous, 4 KiB-aligned
   record per expert, holding the same three tensors in the same order the catalog bundle already
   uses. The pack records the source artifact manifest hashes and the per-range SHA-256 values, so
   it remains a derived artifact rather than a checkpoint. A whole-payload SHA-256 is verified when
   the pack is opened.
2. **C++ verifier/applier** (`src/inference/deepseek_v4_expert_pack.cpp`): validates format,
   model/revision/layer, alignment, no-overlap records, and exact tensor order/offsets/shapes/
   storage types against the pinned catalog; then rewrites only the included experts to one local
   read range. The kernel bindings and packed tensor offsets are unchanged.
3. **Packer** (`tools/pack_deepseek_experts.py`): consumes the existing
   `deepseek-gguf-probe-plan` plans plus `import_hf_ranges` artifacts, verifies their manifests and
   local hashes, merges experts across prompt tokens, pads records to 4 KiB and publishes
   `experts-L{layer}.bin` + `vastloom-expert-pack.json` atomically. No weights are downloaded.
4. **Loader I/O mode and read telemetry**: `ExpertLoadOptions` now carries
   `io::IoCacheMode`; the loader uses the existing `read_range_into` bypass path when requested and
   records `read_calls`, `bypass_read_calls` and `buffered_read_calls`. The read API reports the
   path that actually delivered each read, so a silent buffered fallback is visible instead of being
   assumed. The chain and layer-probe JSON expose the counters plus `expert_source`,
   `expert_pack_records`, `expert_pack_bytes`, `io_cache_mode` and per-layer `io_cache_modes`.
5. **CLI**: `deepseek-chain-probe` accepts `--expert-pack L=PATH` as an alternative to
   `--experts L=PATH`, and `--io-cache buffered|bypass|auto`. `deepseek-layer-probe` accepts the
   same options, with either `MODEL TRUNK EXPERT GLOBALS` or `MODEL TRUNK GLOBALS --expert-pack`.
   `auto` resolves to bypass for packs and to buffered for v1 artifacts; the v1 default is
   unchanged.

## Correctness

Both prompts were run twice from the v1 artifacts and from packs generated from those same
artifacts. The dumps are byte-identical.

| Run | argmax | logits min / max | logits dump SHA-256 | hidden dump SHA-256 |
|---|---|---|---|---|
| token 0, v1 | 86,051 | −16.0034 / 16.1624 | `7F4A7AE5…FE6791` | `DE045CF5…A99AE33` |
| token 0, pack | 86,051 | −16.0034 / 16.1624 | `7F4A7AE5…FE6791` | `DE045CF5…A99AE33` |
| prompt `[0,1,2,3,4]`, v1 | 20,267 | −38.0188 / 14.7002 | `85359E67…B7596B` | `8AE58AB2…E4E40` |
| prompt `[0,1,2,3,4]`, pack | 20,267 | −38.0188 / 14.7002 | `85359E67…B7596B` | `8AE58AB2…E4E40` |

Full hashes are recorded in the run outputs under `build/analysis/deepseek-0731/`; they are not
committed. Routes and per-token routes are identical to the v1 runs.

The packs used here: token 0 has three packs of **45,219,840 B** each (6 records per layer); the
five-token prompt has packs of **218,562,560 B / 226,099,200 B / 218,562,560 B** (29/30/29 records,
88 unique bundles total, 663,224,320 B), matching the v1 expert-byte total.

The single-layer probe was checked the same way: v1 `deepseek-layer-probe` produced logits with
SHA-256 `B201386B…C2BE239` and the pack run produced the same bytes, argmax 30,528, with read calls
**18 → 6** (one per expert) and all six reads delivered through the requested bypass path.

## Read calls

The pack changes the number of range reads, not the bytes read:

| Prompt | Source | Expert misses | Read calls | Reads per miss |
|---|---|---|---|---|
| token 0, 3 layers | v1 artifacts | 18 | 54 | 3 |
| token 0, 3 layers | expert pack | 18 | 18 | 1 |
| prompt `[0,1,2,3,4]`, 3 layers | v1 artifacts | 88 | 264 | 3 |
| prompt `[0,1,2,3,4]`, 3 layers | expert pack | 88 | 88 | 1 |
| layer 0, 1 layer | v1 artifact | 6 | 18 | 3 |
| layer 0, 1 layer | expert pack | 6 | 6 | 1 |

## Warm-cache timing (not a throughput claim)

Three runs for token 0 and two runs for the five-token prompt, sequential, process-warm page cache,
no attempt to drop the Windows cache. Means:

| Prompt | v1 buffered | pack buffered | pack bypass |
|---|---|---|---|
| token 0 | 2,101.0 ms | 2,158.8 ms | 2,201.8 ms |
| `[0,1,2,3,4]` | 5,473.0 ms | 5,272.9 ms | 5,511.8 ms |

Reading is not the bottleneck of these runs; expert bytes are 45–663 MB, most of it paged in after
the first iteration. The pack was neutral-to-slower on token 0 and about 3.7% faster on the
five-token prompt, both inside the observed run-to-run spread. The bypass path was slower on the
warm cache, consistent with the in-tree measurement that small recurring ranges are faster buffered.
**No speed claim is made.** The result that is stable and attributable is the 3× read-call
reduction and the unchanged outputs.

## Direct-I/O proxy (no cache drop)

The reference machine is not elevated, so a system-wide standby-list purge is not available; this
run is a **direct-I/O proxy**, not a reboot-cold measurement. The new counters prove which path
delivered the bytes: in every bypass run below, `bypass_read_calls == read_calls` and
`buffered_read_calls == 0`, so the operating system did not serve those reads from the file cache
and no silent fallback occurred.

| Prompt | Config | Read calls | Actual bypass / buffered | Mean wall time |
|---|---|---|---|---|
| token 0 | v1 buffered | 54 | 0 / 54 | 2,127 ms |
| token 0 | v1 bypass | 54 | 54 / 0 | 2,165 ms |
| token 0 | pack buffered | 18 | 0 / 18 | 2,104 ms |
| token 0 | pack bypass | 18 | 18 / 0 | 2,117 ms |
| `[0,1,2,3,4]` | v1 buffered | 264 | 0 / 264 | 5,233 ms |
| `[0,1,2,3,4]` | v1 bypass | 264 | 264 / 0 | 5,529 ms |
| `[0,1,2,3,4]` | pack buffered | 88 | 0 / 88 | 5,202 ms |
| `[0,1,2,3,4]` | pack bypass | 88 | 88 / 0 | 5,335 ms |

Even with the cache explicitly bypassed, the pack's 3× read-call reduction does not translate into
a wall-time win in this workload: the device delivers the same bytes and compute dominates. The
bypass path is slightly slower on both prompts. This is a negative result for "one read per expert
is a speed lever at this scale", not a failure of the mechanism; the peers operate at model sizes
where tens of thousands of small reads per token dominate.

## Cold-cache measurement

The cold measurement was executed after all: an elevated PowerShell session ran
`scripts/purge-standby.ps1 -Force` immediately before each of the eight runs, so the standby list was
flushed between configurations. The helper enables `SeProfileSingleProcessPrivilege` and requests
`MemoryPurgeStandbyList`; it refuses to run without elevation (verified separately) and is not part
of normal execution. One sample per configuration, sequential, after a successful purge each time.

| Prompt | Config | Read calls | Cold elapsed | Warm mean (earlier) |
|---|---|---|---|---|
| token 0 | v1 buffered | 54 | 2,071 ms | 2,101 ms |
| token 0 | v1 bypass | 54 | 2,256 ms | 2,165 ms |
| token 0 | pack buffered | 18 | 2,328 ms | 2,159 ms |
| token 0 | pack bypass | 18 | 2,226 ms | 2,202 ms |
| `[0,1,2,3,4]` | v1 buffered | 264 | 5,375 ms | 5,473 ms |
| `[0,1,2,3,4]` | v1 bypass | 264 | 5,700 ms | 5,529 ms |
| `[0,1,2,3,4]` | pack buffered | 88 | 5,445 ms | 5,273 ms |
| `[0,1,2,3,4]` | pack bypass | 88 | 5,627 ms | 5,512 ms |

The cold-cache run is still negative for the pack: the 3× read-call reduction does not lower wall
time, and the bypass path remains slower. At this slice size (135 MB token 0, 663 MB five-token
expert bytes) the device read is a small fraction of the ~2–5 s compute; the peers operate at model
sizes where the per-token expert bytes are orders of magnitude larger. The purge helper works
(8/8 successful requests), but the mechanism's speed benefit does not appear at this scale.

## Cache plan and paging guard

The chain probe now computes a cache plan before it creates the per-layer caches. The resident floor
is the range-slice working set (globals + maximum layer trunk + KV/compression/rope/scratch/runtime),
not the full-checkpoint `minimum_host_resident_bytes`; the ceiling is
`(live host hard limit - floor) / maximum expert bundle`, capped at 128 entries. The plan must hold
at least one layer request; `--cache-entries N` can lower the capacity but an explicit value below
one request, or above the ceiling, fails closed. The automatic capacity is clamped with a reported
reason.

For the measured token-0 chain the plan is: resident floor **989,956,124 B**, headroom
**2,991,409,124 B**, ceiling **128 entries / 1,258,291,200 B**, chosen **8 entries / 78,643,200 B**
(`clamped: false`). An explicit `--cache-entries 4` fails with "smaller than one layer request";
`--cache-entries 200` is rejected by the option bound; `--cache-entries 16` is admitted. The plan
does not prove that a larger cache is faster; it prevents a cache that cannot fit the live headroom
from being admitted at all. The unit test also checks that the ceiling never decreases when the RAM
budget increases.

## Usage-profile prototype (recording only)

`--usage-profile PATH` records per-layer/per-expert route counts in a local
`vastloom.deepseek-v4-usage-profile.v1` JSON sidecar after a chain run. The file is merged across
runs, the pinned model/revision must match, malformed files fail closed, and the JSON output lists
the top experts per layer. Two token-0 runs produced observations 3 → 6 with the expected counts
(35, 53, 200, 222, 245, 254 for layer 0, and the matching routes for layers 1–2).

This is deliberately **recording only**: it does not preload, pin or evict experts, and no placement
or speed claim is made. The actual pinning policy remains P0.3 future work; the prototype records
the signal that policy would need.

## Tests

- `vastloom_deepseek_v4_expert_pack` (C++): valid pack verification/application and eight malformed
  cases (wrong identity, wrong layer/format, bad payload hash, missing routing authority, misaligned
  record, wrong tensor bytes, duplicate expert, truncated payload). PASS.
- `vastloom_deepseek_v4_expert_pack_tool` (Python): aligned records, per-record hashes, multi-plan
  merge, mismatched plan/artifact, missing routing authority, existing output/`--force`. PASS.
- `vastloom_deepseek_v4_expert_pack_cli` (opt-in, real artifacts): token 0, three layers, v1 vs pack
  buffered and bypass, plus the single-layer probe; byte-identical dumps and expected read counts.
  PASS.
- `vastloom_deepseek_v4_expert_pack_cli_multi` (opt-in, real artifacts): the five-token union
  (15 artifacts), v1 vs pack buffered and bypass; byte-identical dumps, read calls 264 → 88, and
  actual bypass path on every requested read. PASS.
- `vastloom_deepseek_v4_cache_plan` (C++): automatic/requested plans, minimum-request guard,
  clamping, zero-headroom refusal, monotonic ceiling. PASS.
- `vastloom_deepseek_v4_usage_profile` (C++): missing-file empty profile, round-trip counts, top
  ordering, model mismatch and malformed-file refusal. PASS.
- `scripts/purge-standby.ps1`: non-elevated invocation refuses with the administrator-required
  message; an elevated session executed eight successful purge requests for the cold measurement.
- Existing core tests: release 11/11, debug 11/11 pass. The default v1 chain behavior is unchanged.
- Release and debug builds pass with warnings-as-errors.

## Boundaries

- Only the expert union of the measured prompts was packed; the tool supports any subset, but a
  route that selects an expert not in the pack fails closed with a missing-file error.
- The pack is opt-in; the default remains the v1 artifact path with buffered reads.
- No checkpoint or imported artifact was modified; packs are derived under `build/` and are not
  committed.
- The read counters report the path that actually delivered each read. In the bypass proxy runs,
  every requested read was delivered by the bypass path (`buffered_read_calls == 0`); the loader
  still keeps the documented buffered fallback for platforms that refuse an unbuffered read.
- The cold measurement used one purge-and-run sample per configuration; it is not a statistical
  distribution. The elevated purge helper was executed successfully, but the pack and bypass paths
  showed no cold-cache wall-time win.
- The usage profile is recording only; no pinning or placement behavior is enabled by this report.
- This experiment does not change the project's measured claims: one real layer and a three-layer
  hash-routed chain, no full model, no throughput result, no VRAM payload tier.

## Reproduction

Build packs from the existing plans and artifacts (paths abbreviated):

```powershell
python tools\pack_deepseek_experts.py --layer 0 --output build\analysis\deepseek-0731\packs\L0 `
  --plan build\analysis\deepseek-0731\layer0-route-plan.json `
  --artifact <store>\...--range-l0-experts-c9b27ecd7c32

build\release\vastloom.exe deepseek-chain-probe <metadata-store> <globals> --layers 3 --token 0 `
  --trunk 0=<l0-trunk> --trunk 1=<l1-trunk> --trunk 2=<l2-trunk> `
  --expert-pack 0=<pack-L0> --expert-pack 1=<pack-L1> --expert-pack 2=<pack-L2> `
  --io-cache bypass --dump-logits pack-logits.f32 --dump-hidden pack-hidden.f32 --json

build\release\vastloom.exe deepseek-layer-probe <metadata-store> <l0-trunk> <globals> `
  --expert-pack <pack-L0> --io-cache bypass --token 0 `
  --dump-logits layer-pack-logits.f32 --json
```

Run the same commands once with `--experts L=PATH` and compare the dumps byte for byte, then run the
opt-in CTests keyed by the existing `VASTLOOM_DEEPSEEK_V4_CHAIN_*` variables (the multi-token test
also needs the `..._EXPERTS2/3/4_*` variables).

## Files

- `include/vastloom/inference/deepseek_v4_expert_pack.hpp`,
  `src/inference/deepseek_v4_expert_pack.cpp`
- `tools/pack_deepseek_experts.py`
- `include/vastloom/model/expert_loader.hpp`, `src/model/expert_loader.cpp`
- `include/vastloom/inference/deepseek_v4_expert_provider.hpp`,
  `src/inference/deepseek_v4_expert_provider.cpp`
- `src/cli/main.cpp`, `CMakeLists.txt`
- `include/vastloom/inference/deepseek_v4_cache_plan.hpp`,
  `src/inference/deepseek_v4_cache_plan.cpp`
- `include/vastloom/inference/deepseek_v4_usage_profile.hpp`,
  `src/inference/deepseek_v4_usage_profile.cpp`
- `scripts/purge-standby.ps1`
- `tests/test_deepseek_expert_pack.cpp`, `tests/test_pack_deepseek_experts.py`,
  `tests/test_deepseek_expert_pack_cli.py`, `tests/test_deepseek_cache_plan.cpp`,
  `tests/test_deepseek_usage_profile.cpp`
