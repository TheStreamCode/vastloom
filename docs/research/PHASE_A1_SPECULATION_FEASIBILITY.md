# Phase A.1 — Speculative decoding feasibility on the reference PC

Date: 2026-08-13. Host: reference PC, 31.1 GiB RAM, RTX 4070 Super (unused here), Windows,
machine otherwise idle. Engine under test: llama.cpp build `b1-84e908c` (MSVC 19.44, CPU-only,
AVX512, `n_threads = 12 / 24`), binaries in `build/research-llama-cpp/build-msvc/bin`.

This is a research measurement for the approved plan (`lucky-zooming-babbage.md`, Phase A.1):
falsify the thesis that speculative verification is worth much more to a streaming engine by
measuring whether an existing engine already delivers it. No Vastloom code was changed.

## Setup

- Target checkpoint: `deepseek-v4-flash-0731-iq1s-fbbb5b93/UD-IQ1_S`, 3 shards, 76.9 GB, mmap.
- Draft: `deepseek-v4-flash-0731-dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`, 10.1 GB, mmap.
- Prompt: raw templated
  `<|begin of sentence|><|User|>Qual e la capitale d'Italia?<|Assistant|>` + newline + `</think>`
  (special characters U+FF5C / U+2581), 13 tokens after tokenization.
- Generation: greedy (`--temp 0 --top-k 1`), seed 0, `n_predict 12`, ctx 512, `--no-warmup`.
- Safety protocol: no `--no-mmap`, no `--mlock`, one run at a time, Available Bytes monitored
  every 2 s with a pre-declared kill floor, process killed on breach.

## Baseline (no speculation), measured today

| tool | decode | prompt eval | output |
|---|---:|---:|---|
| `llama-completion -no-cnv` | 3589.14 ms / 11 runs = **3.06 tok/s** | 13 tokens, 1.55 tok/s | `La capitale d'Italia è **Roma**. È la` |
| `llama-server` `/completion`, `speculative.types: none` | 3627.889 ms / 12 tokens = **3.31 tok/s** (302.3 ms/token) | 13 tokens, 2.41 tok/s | same text |

Server command: `-m <main> -c 512 -t 12 --host 127.0.0.1 --port 8099 --no-webui --timeout 300
--parallel 1 --no-warmup`, POST `/completion` with the raw prompt, `n_predict 12`.

## The recorded 1.51 tok/s baseline was machine state, not configuration

The earlier pinned baseline (`lc-base.txt`, same morning) measured **1.51 tok/s**
(7282.90 ms / 11 runs) with the identical binary, flags and thread counts; its `system_info`
line is byte-identical to today's. Prompt eval barely moved (650.97 vs 646.55 ms/token) while
decode halved (662.08 vs 326.29 ms/token). Decode at batch 1 is bandwidth-bound; 3.06 tok/s
x 8.76 GB/token = 26.8 GB/s effective, consistent with the memory bus, while 1.51 tok/s =
13.2 GB/s. Page-cache warmth and background contention decide which one a given run gets.
(The original figure used the 8.00 GB/token budget, since corrected to 8.76; see the
streaming paper's Table 1.)
Any comparison against the old 1.8 tok/s figure must state machine state alongside it.

## Speculative run: not measurable within the host's safe envelope

`llama-server` with `--spec-draft-model <dspark> --spec-draft-n-max 7` (rest identical) was
attempted three times:

1. Kill at FreePhysicalMemory 2.8 GiB — wrong metric (free excludes reclaimable standby).
2. Guard switched to Available Bytes, floor 2 GiB: trajectory 22.0 -> 17.5 -> 16.3 -> 6.7 ->
   1.1 GiB in 8 s; killed. The draft model had not begun loading at the kill point
   (`loading draft model` appears at server time +10.7 s).
3. Main-model page cache pre-warmed by a full baseline server run immediately before; same
   trajectory, killed at 1.1 GiB available. Warming did not change the init profile.

Cause: initialization faults roughly 21 GiB of mapped model pages in ~8 s (76.9 GB target plus
10.1 GB draft mapped onto a 31.1 GiB host) and the Windows trimmer does not reclaim fast enough
to keep Available Bytes above the floor while the draft still has to load. The host owner's
standing constraint is that the machine must never approach saturation, so the measurement stops
here. The two logged init errors are benign and expected on this path:
`dflash requires ctx_other to be set (this warning is normal during memory fitting)` and the
resulting `[spec] failed to measure draft model memory`; the server proceeds to load the draft
after them (`common_speculative_init_result`).

## Ecosystem fact established independently of the failed run

llama.cpp `b1-84e908c` ships first-class support for this draft family:
`common/speculative.cpp` contains `common_speculative_impl_draft_dflash` with an explicit
`is_dspark` mode (Markov head, anchor-first block layout, `dflash.block_size` metadata read),
registers both `draft-dflash` and `draft-dspark` spec types, and wires the draft context to the
target context (`cparams.ctx_other = ctx_tgt`, speculative.cpp:2298). Speculative decoding for
this exact model family is implemented upstream; only its speed on adequate hardware is unknown.

## Verdict against the pre-declared table

None of the plan's rows applies exactly, and that is recorded rather than smoothed over:

- Speculation could not be measured (host envelope), so neither ">= 4 tok/s" nor "< 2.5 tok/s"
  was observed.
- The no-speculation baseline is 3.31 tok/s today, above the "everything under 1.8" row.

Closest row: **the limit is the machine** — with the caveat that today's idle-state baseline
already sits at 55-80% of the plan's declared 4-6 tok/s bandwidth ceiling without speculation.
The differentiating thesis was not falsified by measurement on this host, but its remaining
space is narrow: the only lever left (speculation) already exists upstream, and the open
question is its speed on a host that can hold target plus draft (>= 48 GiB free RAM
recommended for the same initialization spike).

## Consequences

1. The original goal — DeepSeek V4 Flash 0731 usable on this PC — is met by llama.cpp alone in
   the idle state (3.3 tok/s, correct output), and degrades to ~1.5 tok/s under contention.
   That number is state-dependent and must always be reported with its conditions.
2. Measuring the speculative path requires either a nearly-empty 31 GiB host (all applications
   closed, ~3.5 GiB more headroom — still marginal) or a host with substantially more RAM.
3. The strategic decision the plan defers to Phase B is now open with new inputs: adopt an
   existing engine for this machine, or continue engine research where the remaining
   differentiator (speculation designed for streaming) can actually be measured.
