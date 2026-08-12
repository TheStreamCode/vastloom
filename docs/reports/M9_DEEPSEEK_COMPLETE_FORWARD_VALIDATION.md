# M9 complete DeepSeek forward pass and generation

Validation date: 2026-08-11

## Result

The complete DeepSeek V4 Flash 0731 model generated coherent Italian text on the reference PC:

```text
prompt: "La capitale d'Italia e"
output: " una grande citta cosmopol"
```

All 43 layers, the checkpoint's own routers, the official IQ1_S quantized weights, six generated
tokens through the decode loop at 60.3 s per token. 16,015,622,144 bytes of expert payload were read
from NVMe during the run while 7.62 GB stayed resident and free host RAM never fell below 10.3 GB.

This is the model working: prompt in, sensible continuation out. It is not the model being
practical — a minute per token is three orders of magnitude away from usable.

## Scope

All 43 layers, the checkpoint's own routers, the official quantized weights:

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

## Generation

`--max-tokens` routes through the reference decode loop, which reuses KV state instead of re-running
the prompt. Each decode step routes a single token, so its expert working set is six bundles per
layer regardless of prompt length.

| Depth | Prompt | Generated | Output | Per token |
|---:|---|---:|---|---:|
| 12 | `"Ciao"` | 4 | `" komonосновнимMonochromaticte"` | 10.5 s |
| 43 | `"La capitale d'Italia e"` | 6 | `" una grande citta cosmopol"` | 60.3 s |

The twelve-layer output is incoherent because a truncated model is not the model; the full-depth
output is a grammatical Italian continuation. Together they are the clearest evidence that depth is
what carries the meaning, and that the full-depth path is executing the real thing.

Generation read 16,015,622,144 bytes of expert payload for six tokens — about 2.7 GB per token,
close to the 1.77 GB per-token no-cache figure plus the prefill's wider working set. Every byte
comes from NVMe on every token because no expert survives between steps.

## Where the minute went

2.7 GB per token is under a second on this SSD, so the cost was compute: the matvec kernels walked
one row at a time on one core. Rows are independent, so they now split across the Governor's thread
budget, with no reordering inside a row.

| | ms per token | Output |
|---|---:|---|
| serial | 60,301 | `" una grande citta cosmopol"` |
| six workers | 32,730 | `" una grande citta cosmopol"` |
| twelve workers | 25,891 | `" una grande citta cosmopol"` |

Identical text throughout: threading changed speed and nothing else. Twelve threads give 2.33×, and
the step from six to twelve is only 1.26× — roughly 40% of the run is still serial, which is where
the next gain is. Only the packed matvecs are threaded; attention, normalization, RoPE and the
Hyper-Connection mixing are not.

Workers are now drawn from a pool created once rather than spawned per matvec, of which a generation
issues tens of thousands. That halved the test suite, from about 110 s to 50 s. On full-model
generation it mostly steadied the measurement: 25,891 and 28,641 ms per token before, 25,921 and
26,859 after. The remaining spread is the machine, not the model.

The thread count itself came from the runtime policy, which capped inference at six regardless of
the machine. That ceiling predated any threaded kernel; it is now half the logical processors, which
leaves the host as much CPU as the model takes.

## Starting the model

Execution requires every checkpoint file to match its pinned SHA-256 — 357.6 s for this profile.
Paying that before each prompt made the model unusable no matter how fast it generated, so a
completed verification now leaves a receipt recording each file's size and last-write time beside
the confirmed hash. When both still match, the hash is not recomputed.

| | First run | Second run |
|---|---:|---:|
| checkpoint verification | 357.6 s | 0.4 s |

End to end, the same six-token generation went from roughly 500 s to 176.6 s: 0.4 s to verify,
3.4 s to load weights and 171.8 s to generate.

The receipt trusts filesystem metadata, which is weaker than rehashing. Any mismatch, a missing or
malformed receipt, or a changed manifest forces the full pass again, and the receipt is written only
after every file passes.

## Bounds

- 26–29 s per generated token with scalar CPU kernels on twelve threads, and 2.7 GB of NVMe reads
  per token because no expert survives between decode steps. Faster, still not usable. No CUDA
  kernel participates in this path, and roughly 40% of the run — attention, normalization, RoPE,
  Hyper-Connection mixing — is still single-threaded.
- The result is not compared against an independent implementation of the full model. It is
  internally consistent and deterministic, and its expert traffic matches the metadata prediction,
  but no external oracle has confirmed token 20 for this input.
- The chat template is applied on request, but there is no interactive session: one prompt, one
  response, one process.
- Every layer is read from NVMe on every invocation. The hash verification is now cached between
  runs; the weight loading is not.
- All 43 trunks stay resident, so the pass needs 6.39 GB before it starts and a host with less free
  RAM is refused rather than served. Experts are released as the graph advances; trunks are not.

## Where the time actually goes

The Q8_0 kernel is in the graph and the model gained 10%. The accelerator's own telemetry explains
why, and it is not the kernel:

| | 24-layer generation |
|---|---:|
| total | 66.5 s |
| in the Q8_0 kernel, including transfers | 7.2 s |
| everything else, on the CPU | 59.3 s |

Q8_0 covers the attention projections — about 11% of the run. Even at zero cost they cap the speedup
at 1.12×. The 89% is the routed experts in IQ1_S and IQ3_XXS, plus attention, normalization, RoPE
and the Hyper-Connection mixing.

That reorders the remaining work. Optimising the Q8_0 kernel further, or shaving its per-call
round trips, cannot move the model: 2,728 calls at 2.64 ms each against 0.71 ms standalone hides at
most 5 s inside a 66 s run. The kernels that matter are the ones for the expert quantizations, and
after them the rest of the layer — at which point keeping activations resident stops being an
optimisation and becomes the only sensible design, because nothing would need to come back to the
host between operations.

## What would make it interactive

At 26 s per token the model answers but cannot hold a conversation. The measurements point at one
thing: arithmetic. NVMe supplies 2.7 GB per token in under a second, so the memory hierarchy is not
the constraint, and the remaining serial fraction is worth at most another 1.6× even if perfectly
parallelised.

The order of magnitude has to come from the kernels. Two routes, with a constraint worth stating:

- **CUDA.** The driver, the Governor-backed allocator and one scalar RMSNorm kernel already run on
  the reference RTX 4070 Super. What is missing is the quantized matvec family — Q8_0 for the three
  large attention projections, IQ1_S and IQ3_XXS for the routed experts — and attention itself. A
  GPU kernel is allowed a declared numerical tolerance, exactly as the RMSNorm gate carries `3e-5`.
- **CPU SIMD** would give perhaps 4–8× on the same kernels, but it reorders accumulation, and these
  kernels are validated bit-for-bit against pinned ggml output. That parity is what makes the
  results trustworthy at all, so vectorising the reference path would trade the guarantee for the
  speed. A separate, explicitly-toleranced fast path is the honest way to do it.
