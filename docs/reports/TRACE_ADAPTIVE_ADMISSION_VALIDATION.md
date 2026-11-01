# Adaptive expert admission on replay traces

Validation date: 2026-11-01. Host: reference PC (Windows 11, AMD Ryzen 9 7900X, 32 GiB RAM,
RTX 4070 Super). Source: working tree on top of `9ae0c9c`. Scope: **trace replay only**. No model
was loaded, no NVMe read was issued by the simulator, and no throughput or hit-rate claim about a
real Mixture-of-Experts model is made here: the OLMoE and DeepSeek stores are not present on the
reference PC, so no recorded routing trace from those runs is available.

## What was built

- `research::CachePolicy::adaptive` in `trace-sim`: the same two-hit admission rule as the OLMoE
  runtime (`inference/adaptive_admission.hpp`). A first-seen expert is counted as a storage miss
  and not retained; a repeated miss is admitted; eviction among admitted entries prefers the
  lowest frequency and then the oldest use.
- Ghost history sizing: tier capacity divided by the trace's mean expert size, clamped to the
  native 4..1024 slots, then multiplied by four (at most 4,096 keys). The simulator keeps its own
  ghost structure for the same reason it keeps its own LRU/LFU caches: it replays traces without
  runtime cache state.
- Tests: `tests/test_trace_adaptive.cpp` (policy round-trip plus a deterministic scan trace) and
  an adaptive case in the `test_trace` suite. `trace-sim --policy adaptive`, `--json` includes the
  policy name in `policy`.

## Method

```bash
./build/vastloom trace-sim examples/traces/tiny.csv --ram 128MiB --policy adaptive --json
```

The fixture is the bundled `examples/traces/tiny.csv`: 12 records over 4 tokens, 6 unique
experts, every expert 64 MiB. One 64 MiB slot holds exactly one expert, so `--ram` sets the
admission capacity in experts. No VRAM tier was configured.

## Results on the bundled trace

| RAM capacity | Policy | RAM hits | Storage misses | NVMe bytes |
|---|---|---:|---:|---:|
| 128 MiB (2 experts) | no-cache | 0 | 12 | 768 MiB |
| | lru | 0 | 12 | 768 MiB |
| | lfu | 0 | 12 | 768 MiB |
| | adaptive | 2 | 10 | 640 MiB |
| | static-hot | 7 | 5 | 320 MiB |
| 192 MiB (3 experts) | no-cache | 0 | 12 | 768 MiB |
| | lru | 3 | 9 | 576 MiB |
| | lfu | 5 | 7 | 448 MiB |
| | adaptive | 3 | 9 | 576 MiB |
| | static-hot | 9 | 3 | 192 MiB |
| 256 MiB (4 experts) | no-cache | 0 | 12 | 768 MiB |
| | lru | 5 | 7 | 448 MiB |
| | lfu | 6 | 6 | 384 MiB |
| | adaptive | 3 | 9 | 576 MiB |
| | static-hot | 10 | 2 | 128 MiB |

`static-hot` is an offline policy that ranks experts by full-trace frequency, so it is an upper
bound for this fixture, not an online competitor.

The separate scan fixture in `tests/test_trace_adaptive.cpp` has 104 records: a hot expert seen
twice, 100 one-shot experts around it, then two later reuses of the hot expert, all 64 MiB and
one slot. `no-cache` misses 104 times, `lru` 103 times (one hit), `adaptive` 102 times (two hits):
the admitted hot expert survives a hundred cold experts that LRU keeps displacing.

## What the numbers say

- At a two-slot capacity the admission filter is the only online policy that retains anything
  useful: LRU and LFU thrash on the one-shot experts, adaptive keeps the repeated one.
- At three slots adaptive draws level with LRU on this fixture, and at four slots it loses to both
  LRU and LFU. The reason is structural: refusing first-seen retention costs hits once the cache
  can hold the whole working set of a 12-record, 6-expert trace.
- The result is a direction, not a parameter recommendation. Clinical evidence for a real MoE
  would need the recorded 19-field readiness traces from an OLMoE or DeepSeek run, which are not
  available on this host.

## Reproduction

```bash
cmake --preset release && cmake --build --preset release --parallel
ctest --preset release -R vastloom_trace_adaptive
for p in no-cache lru lfu static-hot adaptive; do
  ./build/vastloom trace-sim examples/traces/tiny.csv --ram 128MiB --policy "$p" --json
done
```

## Environment note

The native suite initially crashed in two binaries on this host with `0xC0000409` and no output.
Cause: Windows Defender Controlled Folder Access denies the unsigned test executables write access
to the system temporary directory, so `std::filesystem::create_directories` at test start threw and
terminated the process. The two affected binaries were exactly the ones that created a scratch
directory, which is why the failure looked unrelated to any code change and reproduced identically
on unmodified `HEAD`. `tests/test_temp_dir.hpp` now falls back to the working directory when the
system temporary directory is denied; with that fallback the full suite passes (22 tests, 1
skipped: the ggml backend is not compiled in).

## Non-claims

- No model was executed; no real routing trace was available.
- No throughput, latency or hit-rate claim for OLMoE or any other model.
- `adaptive` remains opt-in; `lru` stays the default in both the runtime and the simulator.
