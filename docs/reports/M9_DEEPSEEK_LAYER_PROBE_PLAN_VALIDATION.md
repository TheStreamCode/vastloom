# M9 DeepSeek layer probe range-plan validation

Validation date: 2026-08-10
Implementation commit: `c64536f237fbf75afd525ee66646d4060c0cd10b`

## Scope

This milestone defines the smallest exact range plan currently derivable for one
checkpoint-shaped DeepSeek V4 Flash 0731 layer:

```text
verified pinned GGUF metadata
  -> one complete non-routed layer trunk, split into bounded tiles
  -> exactly top-k unique caller-supplied expert IDs
  -> complete indivisible W1/W2/W3 bundles for those experts
  -> exact immutable source-object offsets, tensor slices and byte totals
  -> plan-only JSON; no payload read and no execution
```

The pinned official implementation makes the first three layers hash-routed through
`tid2eid[input_ids]`. Correct execution therefore has two phases: read and execute the
trunk/router phase first, obtain the authoritative six expert IDs, then acquire only those six
complete expert bundles. The planner implements the range boundary between those phases, but it
does not yet decode the official router payload. Its selected IDs are explicitly reported as
`caller-supplied-unverified` and `routing_authority_verified=false`.

No multi-gigabyte model object was downloaded. The 82.54 GB profile remains absent and
`execution_supported=false`.

## Native contract

`build_deepseek_v4_layer_probe_plan` validates:

- exact model/revision identity, layer bounds and declared top-k;
- exactly top-k unique expert IDs, each within the checkpoint expert count;
- a positive tile ceiling and a bounded total range count;
- complete canonical expert bundles with consistent logical tensor identities and packed slices;
- nonempty, nonoverflowing immutable source ranges and checked aggregate byte arithmetic;
- separation between verified imported header identity, full-payload presence and executable
  source binding.

Metadata-only catalogs retain the immutable complete shard object names in their logical ranges.
They never expose local `.header` snapshot files as if those prefixes contained executable tensor
payload.

The CLI command is:

```text
vastloom deepseek-gguf-probe-plan MODEL_DIRECTORY
    --layer N --experts CSV
    [--tile-size SIZE] [--context N] [--batch-size N]
    [--no-verify] [--json]
```

It verifies the pinned import by default, derives the range plan and exits. It does not fetch,
allocate or execute any model payload.

## Exact real-metadata result

The locally verified 5,343,584-byte metadata import for
`unsloth/DeepSeek-V4-Flash-0731-GGUF` revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768` produced this layer-0 plan with the default
64 MiB tile ceiling and selected IDs `0,1,2,3,4,5`:

| Measurement | Exact result |
|---|---:|
| Non-routed trunk | 138,867,160 bytes |
| Trunk read ranges | 21 |
| Six complete expert bundles | 38,928,384 bytes |
| Expert read ranges | 18 |
| Complete planned payload | **177,795,544 bytes** |
| Total read ranges | **39** |
| Largest individual range | 35,651,584 bytes |
| One-expert single-tier floor | 6,488,064 bytes |
| Unique immutable source objects | 1 |
| Source object | `DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf` |

IDs `0..5` are a deterministic layout/capacity selection, not a verified route for a real token.
Every layer-0 routed expert has the same encoded size and layout, so these totals are representative
of a top-6 layer-0 range set; their offsets are not evidence of the expert IDs that the official
hash table would select.

The JSON output also records `metadata_only=true`, `imported_file_hashes_verified=true`,
`full_payload_present=false` and `executable_source_binding_verified=false`.

## Failure coverage

Native and CLI tests reject:

- missing `--layer` or `--experts`, malformed/empty CSV and unknown options;
- an out-of-range layer or expert ID;
- duplicate expert IDs and a selection whose count differs from top-k;
- a zero tile/range bound or a range plan exceeding its configured ceiling;
- a missing or inconsistent layer/expert catalog entry;
- malformed expert ranges, tensor identities, packed offsets and aggregate-byte overflow.

The external opt-in CTest runs the CLI against the real metadata import and locks the exact
layer-0 totals above without reading tensor payload.

## Validation host

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM and NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code was compiled or executed.

## Exact results

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors, real metadata opt-in | 12/12 passed in 11.04 s |
| GCC Debug, warnings as errors, real metadata opt-in | 12/12 passed in 25.44 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 12/12 passed in 65.03 s |
| GCC ThreadSanitizer native suite, per-process ASLR disabled | passed |
| GCC libstdc++ debug iterators/assertions | native suite passed |
| Clang 18 Release, warnings as errors | 11/11 passed in 4.72 s |
| MSVC Release x64, warnings as errors | 11/11 passed in 4.70 s |
| MSVC Debug x64, warnings as errors | 11/11 passed in 12.76 s |
| Repeated native regression after synchronization fix | 50/50 passed |
| Python manifest suite | 3 passed, 1 Windows symlink-capability skip |
| Python offline importer, `compileall` and `git diff --check` | passed |

ASan used leak detection and halt-on-error; UBSan and TSan halted on the first finding. TSan used
`setarch x86_64 -R` only for its instrumented process. MSVC host and target architecture were
explicitly x64.

During repeated validation, a pre-existing test-only overlap assertion failed once because the
fixture did not guarantee that its second simulated read service had started before the compute
window. The fixture now coordinates that event explicitly and holds the simulated service for a
bounded 50 ms. No production scheduler behavior changed. The native suite then passed 50
consecutive repetitions and TSan.

## Performance statement

No token rate, TTFT, SSD bandwidth, physical bytes/token or speedup is reported. The
177,795,544-byte value is the exact logical sum of required metadata ranges for this selected
layer-0 plan. Those bytes were not downloaded, read from the physical device or timed. Warm-cache,
process-visible and device traffic are therefore all intentionally absent from this report.

## Known limitations

- No official trunk payload range has been acquired or executed.
- The official layer-0 `tid2eid` table has not been decoded for a real input token.
- Caller-supplied expert IDs are validated structurally but have no routing authority.
- No complete official-payload layer output, logit or token has been produced.
- The planner does not itself create immutable partial-range artifacts.
- Kernels remain scalar CPU references; VRAM placement, CUDA and generation remain absent.

## Next bottleneck

Implement a bounded immutable range-artifact importer that publishes only after temporary
creation, exact length/content-range validation, checksum and atomic rename. After explicit user
approval, the first acquisition should contain only the 138,867,160-byte layer-0 trunk plan. That
payload can then decode the authoritative `tid2eid` route for a pinned input, after which Vastloom
can acquire the exact six complete expert bundles (38,928,384 bytes for layer 0) and run the first
official-payload layer parity gate. A complete checkpoint download is not required for that slice.
