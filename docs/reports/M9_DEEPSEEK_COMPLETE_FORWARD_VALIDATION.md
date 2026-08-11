# M9 complete DeepSeek forward pass

Validation date: 2026-08-11

## Scope and result

The complete DeepSeek V4 Flash 0731 model executed a forward pass on the reference PC. All 43
layers, the checkpoint's own routers, the official quantized weights:

```text
82.54 GB checkpoint on local NVMe
  -> full-hash gate (every file verified against its pinned SHA-256)
  -> capacity preflight against the live safe RAM budget
  -> 596,000,788 bytes of globals streamed under the Governor
  -> 43 layer trunks streamed, one complete tile per tensor
  -> 43 bounded expert caches, one per layer
  -> 258 expert bundles resolved by the model's own routing
  -> 129,280 logits
```

| Quantity | Value |
|---|---|
| Layers executed | 43 of 43 (`complete_model: true`) |
| Input token | 0 |
| Next token | 20 |
| Expert payload read | 1,770,651,648 bytes across 258 bundles |
| Resident total | 7,056,343,388 bytes |
| Weight load | 3.97 s |
| Forward | 29.4 s |
| Full-hash verification of 82.54 GB | 338 s (~245 MB/s), once per invocation |

The expert traffic matches the project's own prediction exactly: `no_cache_expert_bytes_per_token`
was derived from metadata alone at an earlier milestone as 1,770,651,648 bytes, and the executed
pass read precisely that. 43 layers × 6 experts = 258 bundles, no cache hits, because a single token
never revisits a layer.

## Host safety

The user's requirement was that the machine must not be destabilised. Measured during the run:

| Moment | Free host RAM |
|---|---|
| Before | 18.7 GB |
| During weight load and forward | 11.7 GB (minimum observed) |
| After | 18.9 GB |

The process peaked at 7.17 GB resident. The Governor's floor of 8 GiB of free host memory was never
approached, and every byte was released when the pass ended.

Three gates stood in front of the run, in this order:

1. **Preflight.** The exact resident cost is summed from the catalog and compared against
   `min(half of physical RAM, free RAM − 8 GiB)` before anything is allocated. It refuses in advance
   and reports the shortfall: an earlier attempt at 43 layers with a 64-bundle expert cache was
   refused at 7,548,125,532 required against 7,056,396,288 available.
2. **Governor reservations.** Trunk tiles, globals and expert bundles each reserve before they
   allocate. Nothing in this path can grow without permission.
3. **Process guard**, sampled at every layer boundary, so a host that degrades mid-load stops the
   pass rather than pushing through it.

Depth was raised gradually — 1, 12, then 43 layers — rather than starting at the worst case.

## Two defects the depth exposed

**Indexer projection name.** The head-weight projection was mapped to
`layers.N.attn.weights_proj.weight`, but the graph reads it inside the indexer as
`layers.N.attn.indexer.weights_proj.weight`. Layers 0 and 1 have no indexer, so a one-layer probe
and a two-layer pass both succeeded; the error appeared at layer 2, the first with a 4:1 compressor.
This is why depth was raised in steps.

**Reference layer ceiling.** `kMaximumReferenceLayers` was 16, sized for the reduced fixtures rather
than a real checkpoint. It is now 64 — still bounded, with depth separately gated by the preflight
and the Governor.

## Bounds

- One token, no generation loop: this is a single forward pass, not multi-token decoding.
- 29.4 s per token with scalar CPU kernels. Correct, not usable. No CUDA kernel participates in this
  path, and no throughput claim is made.
- All 43 trunks are resident simultaneously. The project's own design — load a layer, execute it,
  release it — is not yet implemented here, so a host with less free RAM would be refused rather
  than served.
- The result is not compared against an independent implementation of the full model. It is
  internally consistent and deterministic, and its expert traffic matches the metadata prediction,
  but no external oracle has confirmed token 20 for this input.
- Prompt handling is absent: the tokenizer reads `tokenizer.json`, which the GGUF distribution does
  not carry, so input is a raw token id rather than text.
