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
   records `read_calls`. The chain JSON exposes `expert_read_calls`, `expert_source`,
   `expert_pack_records`, `expert_pack_bytes`, `io_cache_mode` and per-layer `io_cache_modes`.
5. **CLI**: `deepseek-chain-probe` accepts `--expert-pack L=PATH` as an alternative to
   `--experts L=PATH`, and `--io-cache buffered|bypass|auto`. `auto` resolves to bypass for packs
   and to buffered for v1 artifacts; the v1 default is unchanged.

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

## Read calls

The pack changes the number of range reads, not the bytes read:

| Prompt | Source | Expert misses | Read calls | Reads per miss |
|---|---|---|---|---|
| token 0, 3 layers | v1 artifacts | 18 | 54 | 3 |
| token 0, 3 layers | expert pack | 18 | 18 | 1 |
| prompt `[0,1,2,3,4]`, 3 layers | v1 artifacts | 88 | 264 | 3 |
| prompt `[0,1,2,3,4]`, 3 layers | expert pack | 88 | 88 | 1 |

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

A cold-cache or network-range comparison is the next experiment; it needs a controlled cache drop
(or a fresh boot) and, for the on-demand path, the D4 fetch implementation. Neither was done here.

## Tests

- `vastloom_deepseek_v4_expert_pack` (C++): valid pack verification/application and eight malformed
  cases (wrong identity, wrong layer/format, bad payload hash, missing routing authority, misaligned
  record, wrong tensor bytes, duplicate expert, truncated payload). PASS.
- `vastloom_deepseek_v4_expert_pack_tool` (Python): aligned records, per-record hashes, multi-plan
  merge, mismatched plan/artifact, missing routing authority, existing output/`--force`. PASS.
- `vastloom_deepseek_v4_expert_pack_cli` (opt-in, real artifacts): token 0, three layers, v1 vs pack
  buffered and bypass; byte-identical dumps and expected read counts. PASS (449.96 s in debug).
- Existing core tests: release 11/11, debug 11/11 pass. The default v1 chain behavior is unchanged.
- Release and debug builds pass with warnings-as-errors.

## Boundaries

- Only the expert union of the measured prompts was packed; the tool supports any subset, but a
  route that selects an expert not in the pack fails closed with a missing-file error.
- The pack is opt-in; the default remains the v1 artifact path with buffered reads.
- No checkpoint or imported artifact was modified; packs are derived under `build/` and are not
  committed.
- `io_cache_mode` in the JSON is the requested mode; `read_range_into` may fall back to a buffered
  read when the platform rejects an unbuffered request, and this experiment did not add a separate
  per-read fallback counter. The read-call reduction does not depend on that path.
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
```

Run the same command once with `--experts L=PATH` and compare the dumps byte for byte, then run the
opt-in CTest keyed by the existing `VASTLOOM_DEEPSEEK_V4_CHAIN_*` variables.

## Files

- `include/vastloom/inference/deepseek_v4_expert_pack.hpp`,
  `src/inference/deepseek_v4_expert_pack.cpp`
- `tools/pack_deepseek_experts.py`
- `include/vastloom/model/expert_loader.hpp`, `src/model/expert_loader.cpp`
- `include/vastloom/inference/deepseek_v4_expert_provider.hpp`,
  `src/inference/deepseek_v4_expert_provider.cpp`
- `src/cli/main.cpp`, `CMakeLists.txt`
- `tests/test_deepseek_expert_pack.cpp`, `tests/test_pack_deepseek_experts.py`,
  `tests/test_deepseek_expert_pack_cli.py`
