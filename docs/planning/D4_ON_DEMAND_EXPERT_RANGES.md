# D4 — On-demand expert ranges for DeepSeek V4 Flash 0731 `UD-IQ2_XXS` (design, byte budget)

**Design-only gate.** 2026-10-21, `docs/ROADMAP.md` D4. Target: `unsloth/DeepSeek-V4-Flash-0731-GGUF`, revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`, quant `UD-IQ2_XXS` only; reference PC 32 GiB RAM, 12 GiB VRAM, Windows 11. Nothing
was fetched, executed or measured for this document: every number is encoded GGUF tensor bytes from the local headers
(Appendix A) or a figure already published by D1/D2a/D2b.1/D3a and the recon. `GB` = 10^9 bytes; figures that are not header
arithmetic are marked **[estimate]**.

## 1. Gate and blockers

D4 (`docs/ROADMAP.md`, "D4 — deeper/full-depth strategy") requires a written design plus byte-and-budget analysis: per-token
and per-layer NVMe bytes, bounded cache/staging reservation, Governor feasibility and failure behavior at the reference-PC
budget, with no full-checkpoint download and no execution claim. This file is that deliverable and closes no execution gate.
The track now measures D2a (one layer), D2b.1 (device matrix residency, explicitly not a payload tier) and D3a (hash layers 0–2
chained from per-layer artifacts, learned routers and full depth refused). D4's remaining gaps are learned-router layers, full
depth, the D2b.2 oracle and the budget itself.

## 2. Verified byte baseline

| Quantity | Bytes | Source |
|---|---:|---|
| Full checkpoint, 3 shards | 90,860,736,928 | 5,257,664 + 49,890,588,800 + 40,964,890,464 |
| Always-active: 43 trunks + 6 globals | 6,986,088,796 | catalog; tool reproduces |
| 43 layer trunks (1,193 tensors) | 6,390,088,008 | catalog; tool reproduces |
| Layer trunk min / max | 138,867,160 / 161,707,992 | layers 0–1 / layer 26 |
| Globals (6 tensors, max range 297,861,120) | 596,000,788 | catalog; tool reproduces |
| Routed experts, 11,008 bundles | 83,869,302,784 | catalog |
| Expert bundle min / max | 7,536,640 / 9,830,400 | layer 0 / layer 26 |
| One-token expert union, 6 × 43 bundles | 1,965,686,784 | catalog `no_cache_expert_bytes_per_token` |
| One token, full depth, no cache | 8,951,775,580 | 6,390,088,008 + 596,000,788 + 1,965,686,784 |
| Metadata-only store (3 local objects) | 9,451,968 | D1; tool reproduces |
| Always-active + metadata store | 6,995,540,764 | tool |

Architecture: 43 layers, 256 experts, top-6, `hash_layer_count = 3` (layers 0–2 route through `tid2eid`; layers 3–42 use
learned routers that need the post-attention hidden state). Layer 26 stores routed gate/up as IQ2_S and down as MXFP4; IQ2_S
has no CUDA kernel. Layers 26 and 42 store down as MXFP4. One bundle is three packed slices (gate, up, down) = three exact
ranges (18 for six experts).

## 3. Hybrid plan

1. **Pre-download all always-active ranges once**: 6,986,088,796 B in 1,199 exact ranges (1,193 trunk tensors, one tile each,
   plus 6 globals) and the 9,451,968-byte metadata store; footprint 6,995,540,764 B, inside the 10 GB budget (10,000,000,000
   B), leaving 3,004,459,236 B for bundles, staging and margin.
2. **Fetch only selected expert bundles on demand**, one token at a time, under §7. No other part of the 90.86 GB checkpoint is
   ever requested.

Experts are 92.3% of the checkpoint but 22.0% of a token's bytes; the always-active half is 7.7% of the checkpoint but 78.0% of
a token's bytes. Paying it once turns the token problem into a bounded expert-fetch problem.

## 4. Per-layer and per-token budget

`bundle` = one complete routed expert (3 ranges); `6 × bundle` = one token's cost there; `cumulative` ends at trunk plus
one-token experts, and adding the 596,000,788-byte globals gives 8,951,775,580 B.

| L | trunk B | bundle B | 6 × bundle B | layer total B | cumulative B |
|---:|---:|---:|---:|---:|---:|
| 0 | 138,867,160 | 7,536,640 | 45,219,840 | 184,087,000 | 184,087,000 |
| 1 | 138,867,160 | 7,536,640 | 45,219,840 | 184,087,000 | 368,174,000 |
| 2 | 159,992,792 | 7,536,640 | 45,219,840 | 205,212,632 | 573,386,632 |
| 3 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 759,092,576 |
| 4 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 961,203,512 |
| 5 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 1,146,909,456 |
| 6 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 1,349,020,392 |
| 7 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 1,534,726,336 |
| 8 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 1,736,837,272 |
| 9 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 1,922,543,216 |
| 10 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 2,124,654,152 |
| 11 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 2,310,360,096 |
| 12 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 2,512,471,032 |
| 13 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 2,698,176,976 |
| 14 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 2,900,287,912 |
| 15 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 3,085,993,856 |
| 16 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 3,288,104,792 |
| 17 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 3,473,810,736 |
| 18 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 3,675,921,672 |
| 19 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 3,861,627,616 |
| 20 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 4,063,738,552 |
| 21 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 4,249,444,496 |
| 22 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 4,451,555,432 |
| 23 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 4,637,261,376 |
| 24 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 4,839,372,312 |
| 25 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 5,025,078,256 |
| 26 | 161,707,992 | 9,830,400 | 58,982,400 | 220,690,392 | 5,245,768,648 |
| 27 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 5,431,474,592 |
| 28 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 5,633,585,528 |
| 29 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 5,819,291,472 |
| 30 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 6,021,402,408 |
| 31 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 6,207,108,352 |
| 32 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 6,409,219,288 |
| 33 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 6,594,925,232 |
| 34 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 6,797,036,168 |
| 35 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 6,982,742,112 |
| 36 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 7,184,853,048 |
| 37 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 7,370,558,992 |
| 38 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 7,572,669,928 |
| 39 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 7,758,375,872 |
| 40 | 156,891,096 | 7,536,640 | 45,219,840 | 202,110,936 | 7,960,486,808 |
| 41 | 140,486,104 | 7,536,640 | 45,219,840 | 185,705,944 | 8,146,192,752 |
| 42 | 156,891,096 | 8,781,824 | 52,690,944 | 209,582,040 | 8,355,774,792 |

## 5. Byte domains and cache bounds

The table counts all always-active bytes as streamed once per token, per the memory contract (trunk weights stream from NVMe
under the Governor); retaining the 596,000,788-byte globals in RAM would remove that term, which this design does not assume.

| Domain | Cold first token | Steady state |
|---|---:|---:|
| Local disk, always-active (pre-downloaded) | 6,986,088,796 | 6,986,088,796 |
| Network, expert bundles | 1,965,686,784 (258 bundles, 774 ranges) | 0 … 1,965,686,784 **[estimate]** |
| Local disk, expert misses from the disk cache | 0 | 0 … 1,965,686,784 **[estimate]** |
| RAM expert cache (resident) | 0 | ≤ 1,000,000,000 |
| One-time pre-download network | 6,995,540,764 (1,199 ranges + store) | 0 |

Capacity bounds are exact even though hit rates are not measured: a 1,000,000,000-byte RAM cache holds 131 average bundles
or 101 maximum ones — 50.9% of one token's 1,965,686,784 expert bytes. A 2,000,000,000-byte disk bundle cache holds 101.7% of
one cold token's expert bytes; staging of 268,435,456 B holds 27 maximum bundles, above the 12-bundle queue bound. Routing
skew is unmeasured, so no hit ratio is claimed.

## 6. Schedule

**Hash layers 0–2.** Routing is a table lookup on the token id (`tid2eid`), so all three routes are known before layer 0's
expert turn. Prefetch layer N+1's six bundles while layer N computes; layers 0–2 may be requested together at token start (18
bundles, 135,659,520 B).

**Learned layers 3–42.** Expert ids exist only after that layer's attention and router. Fetch the six bundles after the router,
overlap the transfer with the same layer's shared-expert compute (`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`), then
wait at the routed-expert turn. Whether that compute hides a 45,219,840–58,982,400-byte fetch is **[estimate]**; it has not
been timed.

**Bounded queue.** At most 12 bundles in flight (6 current + 6 hash prefetch), 774 expert ranges per token, 4 worker threads,
one batch owning one cache, abandoned loads drained and counted. On pressure: stop prefetch, then shrink caches, then fail
closed. Never start a layer whose bundles are unstaged.

## 7. Protocol

- **Spans** come only from the pinned plan: `object`, `offset`, `bytes`, three ranges per bundle. `Content-Range: bytes
  <start>-<end>/<source_size>` must match the pinned source object (5,257,664 / 49,890,588,800 / 40,964,890,464). No span is
  derived at runtime.
- **Integrity.** SHA-256 per range on receipt, compared with the expected digest when the pinned plan or artifact manifest
  carries one (D1/D2 artifacts do); a mismatch discards the range and fails the token. First-touch digests for never-imported
  bundles are an open decision (§11). Each layer still needs its own imported trunk and expert artifacts carrying
  `routing_authority_verified`; an on-demand fetch never substitutes for that gate.
- **Transport.** 60 s per-range timeout (the importer's default), at most 3 retries with backoff, bounded concurrency, no
  partial publish: verified bytes go to a staging file and are renamed into the read-only cache only when complete.
- **Staging and caches.** 268,435,456 B staging (pre-reserved), 1,000,000,000 B RAM LRU keyed `(layer, expert)`, 2,000,000,000
  B disk LRU of immutable verified bundles. Every structure is bounded and every eviction is counted.
- **Governor.** Reserve RAM (cache + staging + runtime state) and disk (staging plus bundle cache inside the 10 GB budget)
  before the first fetch; refuse the token if the reservation fails. Host RAM, VRAM and disk headroom are preserved; the
  Governor outranks scheduling.
- **Fail closed.** Hash mismatch, `Content-Range` mismatch, wrong object identity, short body, disk-reserve breach or exhausted
  retries end the token with a non-zero status and no logits — D2a's missing-expert mode (`io_error` 74, no output) is the
  precedent. No silent fallback to another quant, to bytes of unknown provenance, or to placeholder weights.
- **Opt-in.** Fetching requires the existing explicit gate (`--allow-range-download`-style); the default run prints exact
  ranges and bytes only. Layer 26's IQ2_S gate/up must take the CPU path explicitly under device residency (D2b.1); its
  missing CUDA kernel is a known constraint, not a fallback.

## 8. Budget and feasibility at the reference PC

| Resource | Constraint | Reserved by this design | Headroom |
|---|---:|---:|---:|
| Local disk | 10,000,000,000 (design input) | 9,263,976,220 | 736,023,780 |
| RAM | 32 GiB host; 12,746,850,304 observed safe cap | 1,520,343,600 | 11,226,506,704 **[estimate]** |
| VRAM | 12 GiB; 9,595,518,960 observed safe cap | 1,073,741,824 planned tier, not implemented | 8,521,777,136 |
| Network | no declared cap | 1,965,686,784 cold token | rate-limit dependent |

Disk total: 6,986,088,796 always-active + 9,451,968 store + 268,435,456 staging + 2,000,000,000 bundle cache. RAM total:
116,123,672 runtime state + 135,784,472 double-buffered trunk stream + 1,000,000,000 RAM cache + 268,435,456 staging; graph and
reference scratch are extra. The safe caps are the recon's observed values at context 4096, batch 1, not re-measured here.

Verdict: disk and RAM are feasible as a bounded experiment; a VRAM payload tier is absent (D2b.1 is device residency, not a
tier); network feasibility cannot be proven without a measured fetch, and no throughput is claimed.

## 9. Risks

- **Rate limits and availability.** A cold token needs 774 range requests averaging 2,539,906 B against an endpoint with no
  SLA; throttling, per-request latency and repository removal are real failure modes. The revision is pinned, but a vanished or
  gated repository makes the plan unservable.
- **Latency.** At an illustrative 12.5 MB/s (not measured), 1,965,686,784 B is ~157 s before compute; the design must report
  that wait, not hide it.
- **Thin disk margin.** 736,023,780 B covers manifests, plans and logs; a 2.5 GB bundle cache would leave only 236 MB and is
  rejected.
- **Routing skew unknown.** Hot/cold expert distributions are unmeasured: stable skew makes the caches effective, uniform
  routing re-fetches nearly every token.

## 10. Rejected alternatives

1. **Full 90,860,736,928-byte download.** 9.09× the 10 GB budget, contradicts the reduced-payload checkpoint and adds no design
   value.
2. **All 256 experts of every layer, or caching them over time.** 83,869,302,784 B (11,008 bundles), 8.4× the budget; only a
   bounded LRU working set is admissible. The "≈71 GB" figure does not reconcile with this profile: 6,488,064 × 11,008 =
   71,420,608,512 B is an IQ1_S-era bundle average, whereas the verified IQ2_XXS routed total is 83,869,302,784 B.
3. **Predicting learned-layer routes.** Layers 3–42 route on the post-attention hidden state; a predicted bundle set is
   unverifiable and would fetch wrong bytes. Only hash layers 0–2 are predictable, and the design prefetches exactly those.

## 11. Decisions (closed for this design gate)

1. **First-touch digests.** Every fetched range is SHA-256 verified before use. A bundle that was never imported has no
   pre-pinned digest in the current tooling, so the first fetch records its digest in the local anchor and every reuse is
   re-verified against it. A hash-only plan is not required; a digest mismatch fails closed and discards the fetched bytes.
2. **Cache bounds.** Disk bundle cache 2,000,000,000 B, staging 268,435,456 B and RAM expert cache 1,073,741,824 B, all
   reserved through the Governor before transfer. The 2.5 GB proposal is rejected because it breaks the headroom rule.
3. **VRAM payload tier.** D4 does not depend on a payload tier that does not exist: the first implementation runs on the
   CPU/RAM path with optional matrix residency (the D2b.1 semantics, which is not a payload tier). A true VRAM payload tier
   remains a separate, unimplemented item and is not claimed by this design.
4. **Fetching.** This project ships the design and the offline byte-budget tool only; no D4 gate run fetches payload. Any
   future run is opt-in with an explicit network-fetch flag and the bounded, fail-closed protocol in §7, and its result would
   be a new report rather than an extension of this one.

## 12. Non-claims and links

Design only: no payload fetched or executed, no full-model, throughput, parity or placement claim. Byte tables are
header-derived. Measured: D1/D2a (one layer, 129,280 logits, argmax 30,528), D2b.1 (bit-identical device residency), D2b.2
(layer-0 oracle), D3/D3b (2/3 hash layers and chain oracle, argmax 86,051 / 69,146) and D3c (5-token prefill with a non-empty
compressed window, argmax 20,267); nothing beyond three layers of 43.

- Gate: [`docs/ROADMAP.md`](../ROADMAP.md) D4 and the D2b/D3 entries above it.
- Evidence: [`D2 layer 0`](../reports/D2_IQ2XXS_LAYER0_RANGE_EXECUTION.md),
  [`D2b.1`](../reports/D2B_DEVICE_RESIDENCY_VALIDATION.md), [`D3a`](../reports/D3_HASH_CHAIN_VALIDATION.md).
- Recon: `build/analysis/deepseek-0731-refocus-recon.md` §4–§6 (local, git-ignored).

## Appendix A — offline byte-budget tool

`tools/estimate_deepseek_token_bytes.py` (stdlib-only, Python 3.12) parses the local GGUF v3 headers itself —
magic/version/counts, metadata, tensor infos, alignment — and computes encoded bytes with the same GGML block table as
`src/model/gguf.cpp`. It opens no payload byte, hashes nothing, writes nothing and has no network code.

```powershell
python tools\estimate_deepseek_token_bytes.py [--model <store-artifact-dir>] [--json]
```

Default model path: `build/analysis/deepseek-0731/store/models/original/deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata`. It
prints per-layer trunk bytes, trunk total, globals, expert bundle range, one-token expert union, full-depth streamed total and
the always-active local footprint; `--json` emits the same deterministic report
(`vastloom.deepseek_v4.token_bytes_estimate.v1`). Exit code 2 means an unreadable or inconsistent header — no partial report.
Its output reproduces every baseline number in §2, including 6,986,088,796 / 83,869,302,784 / 1,965,686,784 / 8,951,775,580 B.
