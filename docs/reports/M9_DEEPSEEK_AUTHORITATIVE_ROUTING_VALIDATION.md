# M9 DeepSeek authoritative routing and first official payload import

Validation date: 2026-08-11

## Scope and result

This milestone crosses the boundary the project had deliberately refused to cross until now: it
downloads official DeepSeek V4 Flash 0731 weight payload, and it does so only after the checkpoint
itself has named which weights are needed.

```text
verified 0731 metadata
  -> deterministic layer-0 range plan
  -> bounded trunk import (21 ranges, 138,867,160 bytes)
  -> governed read of the pinned tid2eid table
  -> authoritative expert selection for one token
  -> bounded selected-expert import (18 ranges, 38,928,384 bytes)
```

Every earlier plan carried `selection_source=caller-supplied-unverified`, and the importer refused
`--scope selected-experts` because of it. That refusal was correct: a caller-supplied expert list
proves nothing about what the model would actually route. This milestone removes the refusal by
removing its cause.

No layer has been executed from this payload yet. The 82.54 GB checkpoint remains undownloaded;
177,795,544 bytes of it are now local.

## Why hash-routed layers can be resolved without execution

DeepSeek V4 Flash routes its first `hash_layer_count` layers — three, per the pinned metadata — by
table lookup rather than by a learned gate. The reference implementation is explicit
(`src/inference/deepseek_v4_reference.cpp`):

```cpp
indices[token * top_k + slot] = token_to_experts[token_ids[token] * top_k + slot];
```

The gate logits still weight each selected expert's contribution, but they cannot change *which*
experts are selected. Selection is a pure function of the token id and the pinned
`blk.N.ffn_gate_tid2eid.weight` table — and that table is part of the layer trunk, which is
imported before any expert is.

This is why the trunk had to come first: importing it is what makes the routing question
answerable. Layers with a learned router are refused by `deepseek_v4_resolve_hash_route`, because
for them the selection genuinely depends on activations that only execution can produce.

## Imported payload

| Scope | Ranges | Bytes | Elapsed | Source object |
|---|---:|---:|---:|---|
| trunk | 21 | 138,867,160 | 19.7 s | `DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf` |
| selected-experts | 18 | 38,928,384 | 17.8 s | same |

Both imports ran through the existing explicit-authority importer: exact revision-pinned HTTP
ranges, mandatory `206` responses with exact `Content-Range` and `Content-Length`, per-range
SHA-256, fsync, and read-only atomic publication under `models/optimized`. The published manifests
still record `complete_checkpoint_verified=false` and `execution_supported=false`.

The trunk artifact then rebinds against the regenerated catalog with all 21 local range hashes and
its exact source binding verified.

## The resolved route

The routing table is `blk.0.ffn_gate_tid2eid.weight`: 3,102,720 bytes of `I32`, shaped
129,280 × 6, SHA-256 `da364a46796cbc4a6fb2616272f3f92dcd7fc009c87202a8f25eac22ad5da5c1`. It is read
through the same governed, rehashed bound-range path as any other payload, under a residency
created for exactly that range.

For token 0 the checkpoint selects experts **35, 53, 200, 222, 245, 254**. An independent Python
decode of the same range file returned the same six ids, and the table's full contents are
consistent with the pinned architecture: 775,680 entries, every value inside `[0, 256)`, all 256
experts reachable.

The resulting plan reports `selection_source=official-tid2eid-verified`,
`routing_authority_verified=true` and `routing_token=0`. The importer's existing consistency check —
authority and source must agree, and flipping one without the other is rejected — accepted it and
opened the selected-expert scope.

## Bounds and refusals

`deepseek_v4_resolve_hash_route` fails closed on:

- a layer at or beyond `hash_layer_count`, which has a learned router;
- a catalog without verified reference architecture metadata;
- a token id at or beyond the pinned 129,280-entry vocabulary;
- a trunk artifact that does not contain the routing table;
- a table bound as a partial tile rather than one complete tensor;
- a table whose byte count disagrees with `vocabulary_size × experts_per_token × 4`;
- any table entry naming an expert outside the pinned expert count.

Repeated slots resolve to one bundle, not one read per slot, so a token whose top-k names the same
expert twice imports it once.

The CLI keeps the two routing sources mutually exclusive: `--experts` and `--route-artifact` cannot
be combined, and `--route-artifact` without `--token` is rejected rather than silently defaulted.

## Validation

- Native fixture coverage for all seven refusals above plus three resolutions, including the
  repeated-slot case, in `tests/test_range_artifact.cpp`.
- CLI contract coverage for the mutually exclusive routing sources in `tests/test_gguf_cli.py`.
- Windows MSVC Release, complete matrix with every opt-in model test configured: 24 registered,
  24 passed, 0 failed, 0 skipped.
- The real trunk artifact rebinds and its authoritative plan regenerates deterministically.

## What this does not establish

No official layer has been executed. No token has been generated. The imported experts have not yet
been fed through the quantized kernels together with the trunk, and the global tensors —
embedding, output head, final normalization — have not been imported at all, so there is no way yet
to obtain the layer's real input from a prompt. Throughput, NVMe traffic per token and CUDA
execution remain untouched by this milestone.
