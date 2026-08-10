# M9 DeepSeek bounded range-artifact import validation

Validation date: 2026-08-10
Implementation commit: `58826b4fdf63df4fe2f9a9b24afe5cdb2eda2b91`

## Scope

This milestone turns the non-executing DeepSeek layer plan into a bounded, immutable partial-
payload import contract:

```text
hash-verified pinned metadata import
  -> deterministic layer probe JSON
  -> strict local plan validation
  -> explicit range-download authorization
  -> exact revision-pinned HTTP byte ranges
  -> per-range fsync + SHA-256
  -> verified temporary derived artifact
  -> read-only atomic publication in models/optimized
```

The new `tools/import_hf_ranges.py` does not make a partial artifact executable. Every published
manifest records `complete_checkpoint_verified=false` and `execution_supported=false`. It stores
the pinned complete-source object size/SHA-256 as identity evidence, but it does not claim that the
complete remote shard was downloaded or hashed locally.

No official model payload was downloaded during this milestone. The real 138,867,160-byte
layer-0 trunk was exercised in dry-run mode only; network tests used a 256-byte localhost fixture.

## Admission and download contract

The importer accepts only the pinned
`unsloth/DeepSeek-V4-Flash-0731-GGUF` `UD-IQ1_S` profile and validates:

- a regular non-symlink plan no larger than 16 MiB, strict UTF-8/UTF-8-BOM JSON and no duplicate
  object keys;
- exact plan format, model ID, revision, plan-only/non-execution flags and hash-verified metadata
  provenance;
- top-k count, unique expert IDs, routing-source consistency, range counts, byte totals,
  non-overlap and source-object uniqueness;
- every object against the pinned path, complete size and complete SHA-256 identity;
- every offset/length against that object's immutable boundary;
- at most 4,096 plan ranges, at most 1 GiB per range and a 256 MiB default selected-payload cap;
- a disk reserve of at least 20 GiB (300 GiB by default) before writing.

`--scope trunk` is allowed for the current plan. `--scope selected-experts` and `--scope all`
fail before any network access because the current C++ planner reports
`routing_authority_verified=false` and `selection_source=caller-supplied-unverified`. A future
authoritative plan must name `official-tid2eid-verified`; flipping one flag while retaining the
caller-supplied source is rejected.

Any non-dry import additionally requires `--allow-range-download`. The URL is derived only from
the pinned model/revision/object contract; there is no user-supplied remote URL. Every request
must return HTTP 206, the exact `Content-Range`, the exact complete-object size and the exact
remaining `Content-Length`. Short, widened and overlong responses fail closed.

## Artifact format and crash behavior

Successful ranges are stored in a reconstructable, target-specific cache and rehashed before
reuse. In-progress ranges use process-unique partial names and are never resumed across a failed
response, avoiding a composite artifact from different remote responses. On restart, the locked
import removes only its own stale partial files and target-specific `.incoming-*` tree; verified
completed ranges may be reused.

The final `vastloom.range-artifact.v1` directory contains:

```text
models/optimized/<model>--range-l<layer>-<scope>-<plan-hash>/
  vastloom-range-artifact.json
  ranges/0000.bin
  ranges/0001.bin
  ...
```

Each manifest row binds the local short artifact path and SHA-256 to its role, source object,
source object size/SHA-256, original offset and exact byte count. The importer hard-links verified
cache files into a same-filesystem temporary directory, writes and fsyncs the manifest, revalidates
the complete artifact, makes files read-only and atomically renames the directory. Original model
objects are never created, modified or presented by this tool.

## Real-metadata dry-run

The opt-in CTest generated the real layer-0 plan from the verified 5,343,584-byte metadata import,
then invoked:

```text
python tools/import_hf_ranges.py
  --model deepseek-v4-flash-0731-gguf-iq1s
  --plan <temporary-plan.json>
  --store <verified-store>
  --scope trunk
  --reserve-gib 20
  --max-total-mib 133
  --dry-run --json
```

It confirmed 21 ranges / 138,867,160 bytes, `execution_supported=false`, no target creation and
no network payload access. A second invocation for `selected-experts` was rejected for missing
routing authority.

## Offline payload and fault coverage

Windows and Linux tests used a local HTTP server and two disjoint fixture ranges. They verify:

- exact requests `bytes=4-8` and `bytes=20-26`;
- exact payload bytes and local SHA-256 in a published immutable artifact;
- full artifact revalidation and zero-network reuse;
- explicit authorization and total-byte-cap refusal;
- rejection of caller-supplied expert routing, forged routing-source combinations, inconsistent
  totals and duplicate JSON keys;
- wrong HTTP `Content-Range` rejection and partial-file cleanup;
- cleanup of a simulated crash partial, incomplete cache pair and unpublished incoming tree;
- atomic publication under `models/optimized`, never `models/original`.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM and NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code was compiled or executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors, real metadata dry-run | 13/13 passed in 11.86 s |
| GCC Debug, warnings as errors, real metadata dry-run | 13/13 passed in 27.56 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 13/13 passed in 69.83 s |
| Clang 18 Release, warnings as errors | 12/12 passed in 6.67 s |
| MSVC Release x64, warnings as errors | 12/12 passed in 6.54 s |
| MSVC Debug x64, warnings as errors | 12/12 passed in 16.92 s |
| Python offline importer on native Windows | passed |
| Python offline importer on Linux/WSL | passed |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python `compileall` and `git diff --check` | passed |

ASan used leak detection and halt-on-error; UBSan halted on the first finding. The sanitizer does
not instrument Python internals, but its full CTest run includes the same offline importer and
real-metadata dry-run gates. MSVC host and target architecture were explicitly x64.

## Performance statement

No SSD throughput, token rate, TTFT, bytes/token or speedup is reported. The only network payload
was 12 bytes split across two localhost test requests. The real 138,867,160-byte value is dry-run
planning evidence, not downloaded, process-visible or physical-device traffic.

## Known limitations

- A partial range SHA-256 is computed after exact revision-pinned transport; it is not an
  independently pre-pinned range hash. The complete source SHA-256 is recorded but not locally
  verified unless the full shard is present.
- No native reader or catalog binder consumes `vastloom.range-artifact.v1` yet.
- The official layer-0 trunk payload has not been acquired, decoded or executed.
- No authoritative `tid2eid` route or selected-expert artifact can currently be imported.
- HTTP range acquisition is sequential and buffered; direct I/O and compute overlap are later
  runtime concerns, not importer behavior.
- CUDA, VRAM placement and DeepSeek generation remain absent.

## Next bottleneck

Add a native bounded reader/binder for `vastloom.range-artifact.v1`. It must rehash every local
range, match the model/revision/plan and every source span back to the verified GGUF catalog, then
expose only the exact trunk tile views needed by the existing scalar reference path. This can be
validated entirely with synthetic payloads. Only after that consumer passes should the user be
asked to authorize the real 138,867,160-byte layer-0 trunk acquisition.
