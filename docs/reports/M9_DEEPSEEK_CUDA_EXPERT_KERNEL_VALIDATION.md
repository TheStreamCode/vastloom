# M9 CUDA kernels for the routed expert path

Validation date: 2026-08-12

## Result

Every storage type on the routed path now has a device kernel, and the model says exactly what it
said on the CPU:

| 16 layers, `"La capitale d'Italia e"`, 3 tokens | Output | ms per token | Expert payload read |
|---|---|---:|---:|
| CPU only | `" guaranteedlyahi"` | 13,300 | 3,963,551,744 |
| Q8_0 + IQ1_S + IQ3_XXS on the GPU | `" guaranteedlyahi"` | **2,374** | 3,963,551,744 |

5.60×, with identical text and identical expert traffic. The byte count matters as much as the text:
routing is derived from the arithmetic, so a divergence in the kernels moves it, and an unchanged
figure means the same experts were selected in the same order.

At full depth, on a host with 17 GiB free:

| 43 layers, same prompt, 2 tokens | Output | ms per token | GPU matvecs | Declined |
|---|---|---:|---:|---:|
| CPU only | `" una grande"` | 48,970 | 0 | 0 |
| With the three kernels | `" una grande"` | **37,680** | 2,550 | 9,457 |

`complete_model: true` in both, and `" una grande"` is the opening of the `" una grande citta
cosmopol"` this model produced at full depth in the previous milestone. Only 1.30× here, and the
declined count says why: 79% of matvecs never reach the device because there is no VRAM left to hold
their weights. That is a host-memory limit, not a GPU one — see below.

Standalone, against the twelve-threaded CPU reference on an RTX 4070 Super:

| Kernel | Covers | GPU | CPU | Max abs error |
|---|---|---:|---:|---:|
| Q8_0 | attention projections | 0.8394 ms | 3.0364 ms | 0 |
| IQ1_S | expert gate and up | 0.8808 ms | 8.0326 ms | 0 |
| IQ3_XXS | expert down | 0.8753 ms | 9.7560 ms | 0 |

Zero, not a tolerance. Each kernel accumulates in f64 in the reference's own order, so the f32 result
is bit-identical. The earlier Q8_0 gate carried a 2e-2 tolerance because it accumulated in FP32; that
is no longer necessary, and the reason it was a mistake is below.

`cuda-info --matvec-smoke`, `--iq1s-smoke` and `--iq3xxs-smoke` run each gate on its own.

## Why IQ3_XXS carried the speedup

Q8_0 alone had produced 1.10× because attention is roughly 11% of the run. Adding IQ1_S took a
24-layer generation from 22,182 to 15,712 ms per token. IQ3_XXS took a 16-layer generation from
13,300 to 2,374.

The disproportion is in the decoders, not the shapes. IQ3_XXS is the most expensive format to decode
on the CPU — 9.76 ms against Q8_0's 3.04 for the same matrix — because each 98-byte block drives a
grid lookup, a popcount-derived sign pattern and a per-group scale for all 256 weights. On the GPU
that work is per-thread and hidden behind memory latency, so the format that cost the most is the
one that gained the most.

## Two defects a passing parity gate was hiding

Both were found because the model's output text changed while the gate reported zero error. Neither
would have been caught by making the gate stricter in the obvious way.

### A synthetic gate built from dyadic values proves nothing

The IQ1_S gate pinned every block's FP16 scale to 0x3400 (0.25), used a delta of 0.125 and fed it an
input of k/256. Every product is then exactly representable, so f32 and f64 accumulation agree to
the bit — and the gate passed a kernel that was accumulating in `fma.rn.f32` while the reference sums
in `double`. Reporting an error of exactly zero across 2,048 rows should itself have been suspicious:
it is the signature of degenerate arithmetic, not of a correct kernel.

The gates now vary the FP16 exponent per block, fill the mantissa, and use a non-dyadic input. Both
kernels were changed to accumulate in f64 in the reference's order — including materializing
`scale * q` as an f32 weight before widening, because the reference rounds there too. They still pass
at zero error, and IQ1_S is still 9× faster than the CPU despite the 64-bit arithmetic, because these
kernels are bound by decoding and memory, not by floating-point throughput.

### A VRAM cache keyed on a host address is wrong

This was the defect that changed the model's output. The accelerator kept each matrix resident in
VRAM after its first upload and indexed those entries by the host address of the buffer they came
from. Expert buffers are released as the pass advances — that is what keeps the working set inside
the Governor's budget — and the allocator hands the freed address to the next expert. The key then
collides, and the GPU multiplies by another expert's weights under the right name.

The diagnostic was that the output changed *between runs of the same command*: `" _“_ _“_ _“_"`, then
`"。”# ninete nakalista"`. Wrong arithmetic is wrong the same way every time; only an identity
problem is non-deterministic. It stayed latent while the cache limit was 512 matrices, because
attention matrices live for the whole pass and never have their addresses reused, and appeared as
soon as the limit was raised to 8,192 and experts began to fit.

Matrices now carry a `cache_identity`: for experts the layer, expert and catalog tensor index; for
everything else the tensor's name and byte range. Zero means no identity was supplied and the
accelerator declines to cache, so a caller that forgets it loses speed and never correctness.

An intermediate fix — holding a strong lease on the host buffer so its address could not be reused —
was refused by the Governor with exit 75 rather than exhausting the machine. That refusal was useful
twice: it protected the host, and it confirmed the buffers really were being recycled.

## Format details worth recording

**IQ3_XXS blocks are 98 bytes.** The four-byte group field therefore lands on an odd address every
other block, and a 32-bit load there fails with `CUDA_ERROR_MISALIGNED_ADDRESS`. It is assembled from
byte loads. IQ1_S is unaffected: its 50-byte block keeps every 16-bit field on an even address.

**IQ3_XXS writes columns out of order.** The reference decodes a block to f32 and then walks columns
0..255 in sequence, but within a subblock the first grid supplies columns `base+0..3` and the second
`base+4..7`. A kernel that decodes as it goes must make two passes of four lanes rather than handling
both grids in one, or it sums in a different order and gets a different answer.

**Both grids ride in the matrix's own allocation.** IQ1_S needs its 2,048-entry table (16 KiB) and
IQ3_XXS its 256-entry table (1 KiB); placing them at a known offset inside the same allocation avoids
a second pointer to manage and a second lifetime to get wrong. Both are exposed from the reference
(`iq1_s_grid_table`, `iq3_xxs_grid_table`) so the device cannot carry a copy that drifts.

## Host safety

Three refusals occurred during this work, all clean, all exit code 75:

1. The preflight refused 43 layers before allocating anything, reporting `7618380124 required,
   6367707136 available`.
2. The Governor refused the host-lease experiment described above.
3. The expert loader caught `std::bad_alloc` and returned an error rather than letting the
   allocation failure escape.

No crash, no freeze, no swapping. The machine stayed usable throughout.

## VRAM is not free of host memory

The first attempt at 43 layers with these kernels passed its preflight and then died inside the
expert loader with `std::bad_alloc`, while the identical run with `--cpu-only` completed. The
difference is the accelerator, and the mechanism is that a Windows device allocation carries host
commit charge: up to 6 GB of resident matrices were competing for the same budget the preflight had
just checked the weights against, and were invisible to it.

The accelerator's residency ceiling is now capped at what the preflight left unspent, less a
gigabyte for host allocations the preflight does not model. That is what makes the 43-layer GPU run
above possible at all — it settled on 3.46 GB of VRAM rather than asking for 6 — and it is also why
that run only reached 1.30×. At full depth the weights leave almost nothing behind.

This inverts the usual reading of the remaining work. Streaming layer trunks instead of holding all
43 resident is no longer only about fitting on smaller machines; it is what would free the host
memory that the GPU needs in order to keep expert weights resident. The 5.60× at 16 layers is what
this hardware does when the VRAM is there.

### Eager adoption was tried and made it worse

The obvious move — upload each trunk matrix as its layer loads, then release the host copy — was
implemented and measured. It is 25% slower at full depth:

| 43 layers | GPU matvecs | VRAM resident | ms per token |
|---|---:|---:|---:|
| Upload on first use | 2,550 | 3.46 GB | 37,680 |
| Upload as the layer loads | 2,534 | 2.97 GB | 47,191 |

Two reasons, both worth keeping in mind before trying it again.

**It spends VRAM in arrival order rather than on the busiest matrices.** Trunk matrices are uploaded
before the routed experts have asked for anything, and the experts do most of the arithmetic. Lazy
residency lets whoever actually runs claim the space; eager residency gives it to whoever loads
first.

**What it frees in RAM it spends again in commit charge.** The host copy really is released —
`deepseek_v4_load_0731_gguf_layer_trunk` reads each always-active tensor as its own tile, so buffers
are per-tensor and dropping one matrix's reference frees that matrix. But the VRAM it moves into
costs host commit charge of its own, at roughly one to one on this platform: raising the residency
ceiling to include the trunk bytes made a 43-layer pass fail in the expert loader with `bad_alloc`,
where the same run with the conservative ceiling completed. Relocating a byte from RAM to VRAM is
not the same as not needing it.

That is the constraint to design against, and it is why streaming the trunks — loading and releasing
them as the graph walks layers, the way expert caches already work — is the route worth taking. It
reduces the host floor rather than relocating it, and the per-tensor buffers this path already uses
mean nothing structural stands in the way.

### Streaming them works, and pays twice

Two layers resident instead of 43, reloaded as the graph advances:

| 43 layers | All trunks resident | Two-layer window |
|---|---:|---:|
| Host RAM required | 7.62 GB | **2.12 GB** |
| Accelerator ceiling | 1.80 GB | **6.06 GB** |
| Matvecs on the GPU | 1,490 | **4,617** |
| ms per token | 42,466 | **30,123** |

Same `" una grande"`, same `complete_model: true`, same expert traffic. The memory is as much the
result as the speed: the floor no longer scales with depth, so a deeper model does not demand a
bigger machine, and a machine with spare memory spends it on speed instead of on admission.

Three things had to move together, and the middle one is the interesting one.

The preflight charges the window rather than the depth, or it would refuse passes that now fit. The
accelerator's ceiling is derived from what the preflight leaves unspent, so freeing host RAM hands
it directly to the device — that is the second payment, and it is where most of the speed came from.
And the reserve for what the preflight does not model went from 1 GiB to 3: with trunks streamed, a
ceiling computed from the leftover claims nearly all the remaining commit and starves the very
reloads that created the room. The first attempt died in the tile loader for exactly that reason.

The ceiling is saturated again at 6.06 GB, so this is not the end of the gain.

A cost worth stating: at 16 layers, where VRAM was never scarce, streaming is 33% *slower* —
3,148 ms per token against 2,374 — because the reloads buy nothing there. The window helps only when
memory is the binding constraint, which at full depth it is.

## Bounds

- The 5.60× figure is measured at 16 layers, where VRAM residency is not constrained. The same
  kernels give 1.30× at 43 layers, for the reason above. Neither number is the kernels' ceiling.
- The preflight does not count the expert loader's own read buffers. At 32 layers it approves and the
  loader then fails on `bad_alloc`, which should have been an up-front refusal carrying a correct
  figure.
- 1,443 matvecs were declined at 16 layers and 9,457 at 43 — remaining storage types, and above all
  the residency ceiling.
- Attention, normalization, RoPE and the Hyper-Connection mixing remain on the CPU, and activations
  return to the host between operations.
- Still no external oracle. The kernels are verified against this project's own ggml-parity
  reference, which is itself pinned against ggml output, but no independent implementation has
  confirmed the generated text.
