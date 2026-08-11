# M9 first official DeepSeek layer execution

Validation date: 2026-08-11

## Scope and result

Vastloom now executes a real DeepSeek V4 Flash 0731 layer from official payload. One layer of 43,
with the checkpoint's own router choosing the experts:

```text
imported artifacts (trunk + selected experts + globals)
  -> relocated catalog
  -> governed RAM load of trunk and globals
  -> Q4_K embedding lookup for the token
  -> Hyper-Connection pre-mix, RMSNorm
  -> Q8_0/Q5_K attention with Q6_K/Q5_K shared expert
  -> BF16 router + I32 tid2eid table -> six IQ1_S/IQ3_XXS experts
  -> Hyper-Connection post-mix
  -> output normalization + Q4_K head
  -> 129,280 logits
```

Every quantization the layer needs executes together for the first time: Q8_0, Q5_K, Q6_K, BF16,
I32, IQ1_S, IQ3_XXS and Q4_K, on payload whose SHA-256 was verified per range at import and rehashed
again at load.

**These logits are not the model's output.** The real network is 43 layers deep and this run executes
one. The number below is what layer 0 alone produces, and it is reported that way by the command
itself: `complete_model: false`, `executed_layers: 1`, `model_layers: 43`.

## Measured result

| Quantity | Value |
|---|---|
| Token | 0 |
| Experts the router selected and the graph executed | 254, 222, 245, 200, 53, 35 |
| Expert payload read | 38,928,384 bytes (6 misses, cold cache) |
| Trunk resident | 138,867,160 bytes |
| Globals resident | 596,000,788 bytes |
| Relocated catalog tensors / bundles | 45 / 6 |
| Logits | 129,280 values in `[-16.0828, 15.4815]` |
| Layer-0 argmax | 30,528 |
| End-to-end | 1.07 s including both governed loads |

Two consecutive runs produced identical logits, identical argmax and the identical expert order, so
the path is deterministic on this host.

## Catalog relocation

The imported artifacts hold payload; the verified catalog points at a checkpoint object that was
never downloaded. Everything downstream — expert bindings, the bounded cache, the reference graph —
already reads through catalog source ranges, so relocation is the whole bridge:
`deepseek_v4_relocate_catalog_to_artifacts` rewrites the source of each tensor an artifact contains
to that artifact's local file.

The result is a catalog that is partially readable, and that is the point. A tensor no artifact
provided keeps its original source and its file does not exist, so touching it fails on open. That
was confirmed directly: token 1 selects experts 239, 202, 122, 23, 115 and 57, none of which were
imported, and the probe fails with an I/O error instead of executing against wrong bytes.

Relocation refuses artifacts whose model, revision or verification state disagree with the catalog,
ranges that match no tensor, two ranges claiming one tensor, a length that differs from the catalog's,
and coalesced expert bundles, whose positional rewrite cannot be inferred safely. An expert bundle is
relocated only when all of its tensors moved, never half in the artifact and half in an absent
checkpoint.

## Bounds

- The one-layer graph keeps every real dimension, schedule and epsilon from the verified metadata;
  only depth is cut to what has been imported. Layer 0 is hash-routed, so the reduced hash-layer
  count is 1.
- Trunk, globals and experts each pass through their own governed reservation. The expert cache is
  bounded to six complete bundles and never splits one across tiers.
- The `ProcessSafetyGuard` and the runtime policy apply as they do to any real-model command, so a
  host without headroom refuses the probe rather than running it.

## Validation

- `vastloom_deepseek_v4_official_layer_probe` pins the executed expert set, byte totals, logit count,
  layer-0 argmax and the explicit `complete_model: false` reporting.
- Windows MSVC Release with every opt-in test configured, including four on official weights:
  28 registered, 28 passed, 0 failed, 0 skipped.

## What remains for a real token

A token the model would actually emit needs all 43 layers. Two things stand in the way, and only one
is a matter of downloading:

1. **Trunk depth.** 6,390,088,008 bytes across 43 layers, of which 138,867,160 are local.
2. **Expert discovery.** Only the first three layers route by table. From layer 3 the router is
   learned, so which experts a token needs is knowable only while executing that layer — which is
   precisely the streaming problem Vastloom exists to solve, and it wants the complete checkpoint on
   local storage rather than 258 range requests per token.

Throughput, NVMe bytes per token, KV cache behaviour across layers and any CUDA execution of this
path remain unmeasured.
