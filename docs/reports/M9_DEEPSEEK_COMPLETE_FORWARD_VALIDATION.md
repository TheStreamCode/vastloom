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

## Text prompts

The GGUF distribution carries its vocabulary as metadata, not as `tokenizer.json`, so text had no
route into this path. It turned out not to need a second tokenizer: the checkpoint's
`tokenizer.ggml.tokens` matches the pinned DeepSeek tokenizer index for index — 128,000 base tokens
plus 1,283 added tokens, 129,280 in total — and all 127,741 merges are identical. `--prompt`
therefore encodes through the tokenizer already validated against Hugging Face tokenizers 0.22.1,
and `--chat` applies the checkpoint's own template.

Encoding happens before the preflight, because the resident cost depends on it. A synchronous
resolve hands the graph every expert a layer selected at once, so the cache must hold one distinct
bundle per token per slot — six for a single token, twelve for two — and there is one cache per
layer, all live simultaneously. The first version of this preflight charged a single cache and
understated a full-depth pass by gigabytes; it now multiplies by depth.

| Prompt | Experts per layer | Expert cache | Required | Forward | Prediction |
|---|---:|---:|---:|---:|---|
| one token id | 6 | 2.26 GB | 7,056,343,388 | 29.4 s | token 20 |
| `"Roma"` (2 tokens) | 12 | 4.53 GB | 11,517,509,980 | 59.5 s | `" "` |
| `"La capitale d'Italia e"` (6 tokens) | 36 | 0.63 GB | 7,618,380,124 | 167.7 s | `" una"` |

The first two rows held every layer's experts to the end of the pass, so the cost grew with prompt
length and would have needed 13.6 GB of experts alone for the third. The third row does not, because
a layer's experts are dead once the next layer asks for its own: each resolve now releases the
earlier caches first, and peak expert residency is two layers' worth rather than 43. Prompt length
no longer multiplies it — 36 experts per layer cost less resident memory than 12 did before.

The six-token run resolved 1,069 bundles and read 7.34 GB of expert payload while staying at 7.62 GB
resident, with free host RAM steady at 12 GB.

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
- The chat template is applied on request, but there is no conversation: one prompt in, one predicted
  token out. Generation needs the decode loop and KV-cache reuse that the reference driver already
  implements for the reduced graph.
- 167.7 s for a six-token prefill. Every layer is read from NVMe on every invocation, and the full
  82.54 GB hash verification adds 338 s before any of it. Neither is optimised.
