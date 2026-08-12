# Changelog

## Unreleased

- **The forward pass was reading the Hyper-Connection mixing matrix transposed, and the model now
  answers `"La capitale d'Italia è **Roma**."`** The stray `R` that this changelog spent a milestone
  attributing to 1.5-bit quantization was one index expression.

  `deepseek_v4_hc_post` summed `combination[destination][source]` where the layout is
  `[source][destination]`. The Sinkhorn normalization ends with a pass over sources, so
  `Σ_src comb[dst][src] = 1` and that direction is a convex combination of the residual streams,
  which preserves magnitude. Summing the transpose sums a direction nothing normalized: slightly
  more than one, every layer, forty-three times.

  Verified against llama.cpp on the same checkpoint and the same tokens. All eight strongest logits
  now match in the same order, with differences of 0.01 to 0.06 — the f64-versus-f32 accumulation
  gap between two independent implementations, not a defect:

  | | here | llama.cpp |
  |---|---|---|
  | top 1–8 | 5, 201, 372, 1897, 7249, 13989, 6201, 271 | 5, 201, 372, 1897, 7249, 13989, 6201, 271 |

  The hidden state at forty-three layers fell from RMS 4,064 to 1,010, and the generated token that
  had been wrong is now 6091 `oma` — the one the reference scored nineteen points above ours.

- **The M6 operator fixture carried the same transposition**, which is why the defect survived a
  milestone. `tools/reference_deepseek_v4_operators.py` describes itself as generating vectors "from
  the pinned equations": it is a re-implementation written by this project, not the model's own code,
  and it indexed `combination[destination][source]` exactly as the C++ did. Two implementations
  agreeing proves nothing when one was written from the other's reading of the equations.

  Both are corrected and the fixture regenerated. The test failing was the correct outcome and the
  reason for changing it is external: llama.cpp, which shares no author with either.

  This is the fourth time this milestone that a check could not fail — after dyadic parity values
  that could not separate f32 from f64, a cost breakdown obtained by subtraction, and a hidden-state
  measurement taken after the normalization that fixes it.

- **The hidden state explodes with depth: RMS 6.8 at one layer, 4,064 at forty-three.**

  | layers | RMS | mean | min | max |
  |---:|---:|---:|---:|---:|
  | 1 | 6.82 | 0.05 | −25.9 | 21.5 |
  | 8 | 13.42 | 0.25 | −79.4 | 53.2 |
  | 24 | 79.81 | −0.62 | −288.9 | 254.0 |
  | 43 | **4,064.32** | −81.91 | −31,054 | 25,746 |

  Six hundred fold, and accelerating: about 1.10 per layer over the first eight, 1.12 through
  twenty-four, 1.23 through forty-three. A compounding per-layer factor slightly above one is what
  that curve is, and the final normalization then hides it — which is why the logits look merely
  wrong rather than obviously broken, and why forty-three layers of it produce a direction dominated
  by whichever components grew fastest.

  The suspect is the Hyper-Connection post step, whose weights are `2 * sigmoid(...)` and therefore
  range up to two. Applied where the reference does something else, that multiplies the residual by
  roughly 1.1 to 1.2 every layer, which is exactly the measured shape.

- **The first version of this measurement could not have found it**, and that is the third time this
  milestone has built a gate that cannot fail. `--dump-hidden` sampled the hidden state *after* the
  final RMS normalization, which fixes the norm by construction: it reported 19.72 at every depth,
  and that number is a fact about normalization rather than about the model. Sampling before it took
  the same four runs and showed a factor of six hundred.

  The pattern to watch for: a quantity that reports the same value however the thing under test
  behaves. Dyadic parity gates could not separate f32 from f64 and passed two wrong kernels; a cost
  breakdown obtained by subtraction charged the trunk for everything unattributed and redirected a
  milestone; and this reported a constant because a normalization had just imposed one.

- **The forward-pass defect reproduces on a single token**, which turns the debugging loop from a
  twelve-token generation into one forward pass. Given nothing but the beginning-of-sequence token,
  the two engines agree on almost nothing:

  | rank | llama.cpp | here |
  |---|---|---|
  | 1 | 5 — 16.67 | **20** — 14.78 |
  | 2 | 201 — 15.40 | **18** — 14.00 |
  | 3 | 372 — 15.36 | **21** — 13.97 |
  | 4 | 1897 — 15.09 | **22** — 13.78 |
  | 5 | 7249 — 14.54 | **26** — 13.70 |

  One token appears in both lists, and the rest do not. So this is not something that accumulates
  over a sequence, and not something the attention window, the indexer or the positional encoding
  can be blamed for: with one token there is no history to attend to and one position to encode.

  The shape of our ranking is the lead: **18, 19, 20, 21, 22, 26** — consecutive vocabulary
  positions. A ranking dominated by adjacent ids is what happens when the hidden state carries no
  direction worth speaking of and the logits fall back on the row statistics of the LM head. That
  points at a hidden state that is near-constant or near-zero by the time it reaches the head,
  rather than at a wrong-but-meaningful one.

  Also measured, and worth keeping straight: at twelve tokens both engines put 7529 first — ours
  more strongly, 35.46 against 33.71 — while positions two through seven are nearly disjoint. The
  argmax agreeing is what let this survive a full milestone unnoticed.

- **Routed experts share nothing, so there is no base to factor out of them.** The idea was that an
  expert might be a shared component plus a small individual part, in which case the shared part
  could stay resident and only the parts would stream — and the saving would grow with precision,
  making a high-precision checkpoint affordable on a machine that cannot hold it. It is a question
  about weights rather than about inference, so `deepseek-expert-redundancy` answers it without a
  correct forward pass and without downloading anything.

  | | storage | mean pairwise cosine | residual after the mean |
  |---|---|---:|---:|
  | layer 0 | IQ1_S | 0.00049 | 0.9371 |
  | layer 0 | IQ2_XXS | 0.00091 | 0.9367 |
  | layer 20 | IQ2_XXS | 0.00172 | 0.9361 |
  | *sixteen independent vectors* | | *0* | *0.9375* |

  Orthogonal, and the arithmetic settles it: the mean of N independent vectors carries 1/N of the
  energy, leaving 1 − 1/N behind. At N=16 that predicts 0.9375 and the measurement says 0.936. The
  6% that looked shared is the statistical residue of averaging sixteen unrelated things, not
  structure. The largest cosine found in any pair was 0.0059.

  The obvious objection — that 1.5-bit quantization might have destroyed a correlation the original
  weights have — is why the second and third rows exist: a different codebook at 2.06 bits and a
  different depth give the same answer.

  This closes expert deduplication, and with it the hope of making an FP8-class checkpoint cheap by
  compressing what it repeats. It repeats nothing. Which is also the healthy result: a mixture whose
  experts had overlapped would have been wasting its capacity. The levers that remain are amortizing
  one expert read across several tokens, and kernels that are currently seventeen times off the
  memory bandwidth.

  Cost of finding out: one afternoon and no download, against the 300 GB and the Safetensors
  execution path the other answer would have justified.

- **An external oracle exists at last, and it disagrees with us.** llama.cpp built from source against
  the same 82.54 GB checkpoint, given the same twelve prompt tokens, greedy, answers
  `"La capitale d'Italia è **Roma**."` This project answers `"La capitale d'Italia è **R Roma**."`

  The stray `R` is ours. It survived a quantization change from 1.5 to 2.06 bits, an exact
  detokenizer round-trip and a prefill-versus-decode comparison — three suspects cleared, and the
  only one left standing was the forward pass, which no internal test could indict because every
  internal test shares it.

  Eight generated tokens match exactly, including the `R` itself; the ninth differs, `oma` (6091)
  where we produce ` Roma` (38710). That looked like a near-tie decided differently — our matvecs
  accumulate in `double` by design and ggml's in `float` — so the logits were measured on both
  sides at that exact position, through a one-shot probe linked against llama.cpp's own library.

  | | llama.cpp | here |
  |---|---:|---:|
  | strongest | **6091 `oma`, 39.51** | 38710 ` Roma`, 24.83 |
  | margin over second | **17.20** | 1.85 |
  | logit of 38710 | 17.71 | 24.83 (chosen) |

  Not a tie, and not arithmetic. The reference is overwhelming — 39.5 with a seventeen-point gap —
  and puts our choice 21.8 below its own. We rank its answer sixth.

  The useful part is the shape rather than the ranking: our strongest logit is 24.8 where theirs is
  39.5, so our distribution is flat where theirs is sharp. A rounding difference cannot flatten a
  distribution. That points at the final hidden state being attenuated or misdirected rather than at
  any single expert or quantization path, which puts the output Hyper-Connection mixing, the final
  normalization and the LM head at the top of the list. Eight tokens agreed because those calls were
  not close either.

  Two things are independently confirmed by the same run: the tokenizer and chat template are exact
  — llama.cpp turns the templated prompt into the identical twelve ids — and llama.cpp supports both
  `deepseek4` and `dflash`, so the DSpark draft has a reference too.

- **llama.cpp is about fourteen times faster than this engine, on CPU, against our GPU path.**
  Same machine, same checkpoint, same twelve threads, no CUDA in their build: 545 ms per token
  against our 7,577. Model load took them ten seconds.

  This is the comparison this project had never made, and it reframes every performance number in
  this changelog. The kernels here are deliberately naive — one thread per row, no coalescing, a
  matvec measured seventeen times off the memory bandwidth — and that gap now has a name and a
  number rather than an assumption that it was small.

- **Trunk reloading is about a tenth of a token, not half of it, and this milestone spent a day
  optimising the wrong thing on the strength of that figure.** The 50% came from subtraction — all
  unaccounted time charged to the trunk — and subtraction cannot tell reading from computing.

  The correction came from a change that worked exactly as predicted and bought nothing. The
  streaming window now retains a fixed set of layers instead of discarding the oldest: under a
  cyclic reference pattern a window smaller than the cycle has a hit rate of zero however it picks
  its victim, but a hit rate of capacity/cycle if it simply stops picking. At eight layers that is
  393 reloads and 54.5 GB against 473 and 65.3 — the predicted 393 and ~54 to the digit — and the
  time per token moved 1.9%, inside the noise.

  Ten point eight fewer gigabytes read for no measurable time bounds what the reading costs: about
  7.7 GB/s, which is page-cache speed rather than NVMe speed. The same page cache that made direct
  I/O three times slower has been serving the trunk all along. Where a token actually goes, of
  75,769 ms:

  | | ms | share |
  |---|---:|---:|
  | GPU kernels | 23,724 | 31% |
  | expert loading | 15,014 | 20% |
  | expert compute | 13,601 | 18% |
  | everything else, trunk reads included | 23,430 | 31% |

  Expert bytes are 26.1 GB and are read once each, so that traffic is real; trunk bytes are the same
  6.39 GB re-read eleven times, and mostly are not. Future work belongs on the kernels and the
  experts.

  The retention change stays: fewer bytes for the same time, and at the default window of two it is
  a strict improvement over discarding the oldest. It carries no speed claim.

  It also shipped broken for one run, and the discipline caught it: the prediction was written down
  first, 5,805 reloads and 810 GB came back instead of 393 and 54, and the missing `transient` check
  was obvious within a second. A vaguer prediction would have filed it as another idea that did not
  work, and left the defect in.

- `deepseek-forward --gpu-limit N` pins the accelerator's ceiling to N mebibytes, so an A/B compares
  two runs rather than two machine states. The derived ceiling moves with whatever else the host is
  doing — it varied between 1.05 and 2.34 GB across one afternoon — and it is the largest single
  influence on the time per token, which invalidated three comparisons here before it was noticed.
  It only ever lowers: the Governor's limit still bounds it.

- **A busy host no longer switches the GPU off entirely.** The accelerator's ceiling is what the RAM
  limit leaves after the pass's own requirement and a 3 GiB reserve for allocations the preflight
  does not model, and on a loaded machine that subtraction reaches zero: a full-depth token cost
  55,788 ms against 8,405 with an idle device beside it. When the full reserve would leave nothing,
  a gigabyte is now held back instead and a gigabyte of ceiling granted.

  The reserve was measurable all along and nobody had measured it. `--json` now reports
  `peak_commit_bytes`, and on a 43-layer pass the process peaked at 5.12 GB against 2.56 charged and
  2.33 held by the accelerator — so the unmodelled part is **0.23 GB**, a thirteenth of the three
  gigabytes held back for it. The arithmetic closing to within a rounding error also confirms the
  1:1 host commit charge for VRAM that the ceiling is premised on.

  A gigabyte and no more, because more buys nothing. On one prompt a 512 MiB ceiling cost 22,781 ms
  per token and a 1.41 GB one 20,682 — a 9% spread, inside the noise — and six earlier runs between
  1.05 and 2.34 GB showed no trend. Only zero is different. This removes a cliff rather than
  climbing a slope, and claims the least commit that does so.

  Exercised rather than reasoned about: the tight path was provoked deliberately by lengthening the
  prompt until the requirement pushed the headroom below zero. At the resulting 1.41 GB ceiling the
  pass peaked at 4.27 GB against a 6.15 GB budget, and the 3.74 GB it charged was itself 0.78 GB
  more than it touched, so the safety margin survives twice over.

  One correction to how this was arrived at: the constant was first raised on a comparison between
  two different prompt lengths, whose prefill costs are not comparable. The same-prompt measurement
  contradicted it, and the constant went back down.

- The accelerator now evicts what will not be asked for again before what will. Recency cannot rank
  these matrices: the forward pass touches every one of them exactly once per token, so
  least-recently-used *is* arrival order and evicts what is needed soonest — the textbook
  pathological case for a cyclic scan. What separates them is recurrence. Always-active weights are
  multiplied by every token; a routed expert is top-6 of 256 and probably will not be picked again,
  which is why tripling the expert cache changed the bytes read by nothing. Roughly 774 expert
  uploads per pass were evicting the trunk, and `trunk_reloads` sat at exactly 473 — 43 layers times
  11 passes, every layer re-read on every pass — whether the ceiling was 1 GB or 6.

  **Measured, and it changes nothing.** At a ceiling pinned to 1 GiB the run returns `trunk_reloads`
  473, `trunk_bytes_read` 65.3 GB and `trunk_device_hits` 7,095 — the same three numbers as every
  other configuration tried, whether the victim is chosen by recency or by recurrence, whether the
  window streams layers or single matrices, and at every ceiling between 1.05 and 2.34 GB.

  The arithmetic says why, and it is not a bad choice of victim. A pass wants about 9 GB — 6.39 of
  trunk plus 2.5 of experts — against a cache this host cannot take past 2.3. At that ratio
  everything is evicted before it is next needed no matter which order it leaves in. 473 is exactly
  43 layers times 11 passes: every layer is re-read on every pass, always. Cache policy is closed as
  a lever until the ceiling approaches the working set, which needs host commit this machine does
  not have.

  The change is kept because it is right by construction — a routed expert genuinely will not be
  asked for again and an always-active matrix genuinely will — and because the pathology it removes
  returns the moment a ceiling can hold the trunk. It is credited with no gain.

  One number is worth suspicion rather than trust: `trunk_device_hits` is 7,095 in every run ever
  measured here. A count that does not move when the cache policy, the cache size and the fetch
  granularity all change is not measuring what the cache retains between passes.

- A GPU with no host budget now says so. The accelerator's ceiling is what the RAM limit leaves
  after the pass's own requirement and a 3 GiB reserve for what the preflight does not model, and
  on a busy machine that reaches zero: the model runs about fourfold slower and looks merely slow.
  The Governor is right to refuse — the host has no commit to spare — but the remedy belongs to the
  user and is host memory, not a faster disk or a better GPU, so the reason is now printed.

- Streaming one trunk matrix at a time instead of one layer at a time was implemented, measured and
  rejected. It is the obvious optimisation — a layer is 138 MB and a matrix about 5, and the window
  was re-reading the whole layer whenever any single matrix of it was absent from the device — and
  it saved **nothing**. Three paired runs on one machine state:

  | | ms/token | load_ms | trunk bytes read | reloads |
  |---|---:|---:|---:|---:|
  | whole layer | 8,337 | 1,555 | 65.3 GB | 473 |
  | one matrix | 9,951 | 83 | **65.3 GB** | 7,029 |
  | layer, metadata index | 8,979 / 8,768 | 79 / 95 | 65.3 GB | 473 |

  Identical bytes, to three significant figures, and 19% slower. The prediction had been 7 GB per
  token falling to 2.2, and it was wrong for a reason worth keeping: the device served 7,095 matrix
  requests without a read in *both* configurations — the same number — because what fits in VRAM is
  whatever is small. Roughly half the matrices are resident and they carry about a fifteenth of the
  bytes, so "read only what is missing" reads essentially the whole trunk anyway, in 7,029 governed
  reads of 10 MB where 473 sequential reads of 138 MB had done. Decline counts do not predict value;
  neither do hit counts. Bytes do, and nothing was counting them, which is why `trunk_bytes_read`
  now exists.

- The startup pass no longer reads 6.39 GB of trunk weights to learn shapes the catalog already
  knows: `load_ms` falls from 1,555 to 79. The mapping from a graph consumer to its checkpoint
  tensor used to exist only as control flow inside the binder loop, so the only way to learn one
  matrix's shape was to load all of them; it is data now, and both the binder and the index walk the
  same list.

  No speed is claimed. At ten tokens the total wall clock is unchanged or marginally worse, because
  that startup read was not waste — it was prefetch, warming the page cache that the first forward
  pass then read from. The same effect that made direct I/O three times slower. The cost moves later
  rather than disappearing, and amortises as a generation lengthens.

- Measured again that the accelerator's residency ceiling is not the binding constraint at these
  sizes: across four runs it varied between 1.05 and 2.06 GB with live host memory, and the time per
  token did not follow it. The ceiling mattered when it was the difference between 90% of matvecs
  declined and 28%; with every storage type now having a kernel, it no longer is.

- **IQ2_S decodes, and the UD-IQ2_XXS checkpoint starts.** One layer of forty-three stored its gate
  and up projections in a format this project had in no form, and two tensors out of 129 were enough
  to stop the model. It is not a variant of IQ2_XXS despite the name: 82-byte blocks rather than 66,
  a 1,024-entry grid rather than 256 — addressed by ten bits, two of which are lifted out of a
  per-group high-bit byte — two four-bit scales per group of 32 rather than one derived from a sign
  word, and explicit sign bytes rather than a seven-bit index whose eighth bit is a parity.

  Checked against ggml's own `dequantize_row_iq2_s`, compiled and run under WSL on the same bytes,
  rather than against this implementation's output: the FNV hash of all 1,536 decoded values matches
  at zero absolute error, as do the matvec results. The gate uses scales that vary in exponent and
  mantissa, including a subnormal, and a non-dyadic input, and it was verified capable of failing by
  corrupting one digit of the expected hash and watching the suite go red.

  The lesson that made this necessary stands: *dynamic* means the mixture is read from the file, not
  inferred from the name. `ffn_gate_exps` is IQ2_XXS on forty-two layers and IQ2_S on layer 26.

  The 90.86 GB checkpoint then ran all 43 layers and answered
  `"La capitale d'Italia è **R Roma**."` on its own stop token, at 7,800 ms per token with a 0.59 GB
  accelerator ceiling. 264 matvecs were declined and every one of them was type 22 — IQ2_S, the only
  format now without a device kernel — which is how we know the new decoder is on the path rather
  than merely compiled.

- **The stray `R` is not a quantization artefact.** This changelog said it was the ~1.5-bit
  quantization showing through, and a 2.06-bit checkpoint of the same model has now produced the
  identical string, stray `R` included. Two quantizations differing by half a bit per weight do not
  agree character for character by accident on a defect that belongs to one of them. The earlier
  explanation was a plausible story rather than a measured one.

  Two suspects have since been cleared. The detokenizer is exact: the generated ids
  `7529, 113213, 283, 9, 75993, 7269, 2619, 52, 38710, 42499` re-encode from the rendered text to
  themselves, and 52 is `R` while 38710 is ` Roma` — the model really did emit both. The incremental
  decode is not feeding the model stale state either: replaying the same prefix through prefill
  alone, via the new `--continue-ids`, predicts 38710 as well. Prefill and decode agree.

  What that leaves is the forward pass itself or the model's own behaviour, and this test cannot
  separate them because both paths share that forward pass. Distinguishing them needs the external
  oracle this project still does not have. A second prompt fails differently —
  `"La capitale della Francia è **Par**."` truncates rather than duplicating — so whatever it is, it
  is not a single fixed off-by-one.

- `deepseek-forward --json` now reports `generated_ids`, and `--continue-ids` appends exact tokens
  after the chat template so a partly generated answer can be replayed as a prompt. A decoded string
  cannot distinguish a model that chose an odd token from a decoder that mangled a reasonable one,
  and that distinction was most of the work above.

- Each quantization now carries its own pinned identity rather than one being privileged. A store
  holding UD-IQ2_XXS is recognised, verified and executed exactly as one holding UD-IQ1_S, and a
  store matching neither is verified against nothing. Two independent gates enforced this — the spec
  comparison in the catalog reader and the reference-config check in the graph — and both were
  extended rather than relaxed: the promise is that a checkpoint cannot quietly become a different
  one, which is per-quantization and stays intact.

- The GGUF reader accepts metadata under both `deepseek4.` and `dflash.` prefixes, so the DSpark
  draft describes itself to the same reader as the main checkpoint. The draft is downloaded
  (10.15 GB): architecture dflash, three blocks, `target_layers [41,42,43]`, 256 experts at top-6,
  routed experts in MXFP4 with everything else Q8_0, BF16 or F32 — every one of which already has a
  device kernel. Its catalog is still rejected because the validator expects a 43-layer tensor and
  expert mapping.

- Added the IQ2_XXS kernel, which the decline breakdown had ranked first. Declines at 43 layers fall
  from 5,089 to 2,749 and a token from about 13,226 to 11,739-11,954 ms, with zero absolute error
  against the reference. The two runs land within 1.8% of each other, tighter than anything else
  measured, so this clears the noise floor.

  On a full answer the gain is larger than the four-token runs showed. The same question that cost
  12,180 ms per token before this kernel now costs 8,736 - 28% - and returns the same
  `"La capitale d'Italia è **R Roma**."` on its own stop token. Longer generations amortise both the
  load and the VRAM warm-up, so short measurements understate what a real response costs; four
  tokens showed 10%.

  The gain is smaller than the decline count implied, and the standalone gate says why: IQ2_XXS
  costs 2.05 ms on twelve CPU threads against IQ1_S's 8.03 for the same shape, so it wins 2.3x on
  the device rather than 9x. Decline counts rank what to write next; they do not predict what each
  is worth.

  Grids now start on an eight-byte boundary rather than immediately after the weights. They are read
  with 64-bit loads, and block strides of 50 and 66 bytes end a matrix wherever they please - latent
  for IQ1_S, and it would have fired on the first shape whose byte count was not a multiple of eight.

- The model answered a question. With the checkpoint's chat template applied,
  `"Qual e la capitale d'Italia?"` produced `"La capitale d'Italia è **R Roma**."` and stopped on its
  own stop token rather than running out of budget — correct answer, correct Italian, assistant-style
  formatting, at 12,180 ms per token over 43 layers.

  Worth recording because the same model looks wrong without the template: `"La capitale d'Italia e"`
  continues as `" una grande citta cosmopol"`, which is a perfectly good completion of a sentence
  fragment and no answer at all. The distinction between completing text and answering a question is
  the template, not the weights. The stray `R` is the ~1.5-bit quantization showing through on a
  token the model was otherwise certain of.

- The accelerator evicts its least recently used matrix instead of refusing the next one, and a
  full-depth token went from 30,280 to 11,639 ms. It had been filling VRAM and then declining
  everything else for the rest of the run — 6,962 of 12,007 matvecs — on the implicit theory that a
  transfer is expensive. The arithmetic says otherwise: a routed expert costs about 0.29 ms to
  upload and 0.88 ms to multiply against roughly 8 ms on twelve CPU threads, so re-uploading it on
  every single use still wins about sevenfold. Refusing work to avoid a transfer that costs a
  seventh of the work is a bad trade. Pinning trunk matrices was the same mistake in reverse and was
  reverted: no matrix deserves permanent space.

- Added dense FP32 and BF16 matvec kernels, chosen by counting declines per storage type rather than
  by assumption. The expected next kernel was Q5_K; the measured ranking put FP32 third and BF16
  fifth, and both are far simpler — FP32 needs no decoding, BF16 is a 16-bit shift. Declines fell
  from 3,432 to 2,375 and both types left the table. No speed is claimed: repeated runs of identical
  code span 11,639 to 14,955 ms per token on this host, and the change does not clear that.

- Layer trunks are streamed through a bounded two-layer window instead of all staying resident, and
  a full-depth pass now needs 2.12 GB of host RAM rather than 7.62 GB. The floor stops scaling with
  depth: a deeper model no longer demands a bigger machine, and a machine with spare memory spends
  it on speed instead of on admission. It pays a second time through the accelerator, whose ceiling
  is derived from what the preflight leaves unspent — 1.80 GB before, 6.06 GB after — which took
  matvecs on the GPU from 1,490 to 4,617 and a token from 42,466 to 30,123 ms, with the same
  `" una grande"` and the same expert traffic. At 16 layers, where VRAM was never scarce, the same
  window is 33% slower: the reloads buy nothing when memory is not the binding constraint.

- Every storage type on the routed expert path now has a CUDA kernel, and a 16-layer generation
  dropped from 13,300 to 2,374 ms per token — 5.60× — producing the same text (`" guaranteedlyahi"`)
  and reading the same 3,963,551,744 bytes of expert payload as the CPU path. IQ3_XXS carried most of
  that: it is the most expensive format to decode on the CPU (9.76 ms against Q8_0's 3.04 for the
  same shape), so it gained the most on a device where the decoding hides behind memory latency. All
  three kernels — Q8_0, IQ1_S, IQ3_XXS — now match the reference at zero error rather than under a
  tolerance, because they accumulate in f64 in the reference's own order. At full depth the same
  kernels take a 43-layer generation from 48,970 to 37,680 ms per token, `complete_model: true`,
  output `" una grande"` on both paths. See the
  [CUDA expert kernel validation](docs/reports/M9_DEEPSEEK_CUDA_EXPERT_KERNEL_VALIDATION.md).

- The accelerator's VRAM residency is now capped by the host memory the preflight left unspent.
  A Windows device allocation carries host commit charge, so resident matrices compete for the same
  budget the weights were checked against: a 43-layer pass with the accelerator at its full 6 GB
  ceiling passed preflight and then died in the expert loader with `bad_alloc`, while the identical
  `--cpu-only` run completed. It now spends only the leftover headroom less a gigabyte for
  allocations the preflight does not model — 3.46 GB at full depth, where it settles for 1.30×
  because 9,457 of 12,007 matvecs are declined for want of VRAM. At 16 layers, where there is room,
  the same kernels give 5.60×. Streaming layer trunks is therefore not only about fitting on smaller
  machines; it is what would free the host memory the GPU needs.

- Fixed two defects that a passing parity gate was hiding, both found because the model's output text
  changed while the gate reported zero error.

  The accelerator indexed its resident VRAM matrices by the host address of the buffer they came
  from. Expert buffers are released as the pass advances, the allocator reuses those addresses, and a
  cache entry then served one expert's weights under another's name. The tell was that the output
  changed *between runs of the same command* — wrong arithmetic is wrong the same way every time.
  Matrices now carry a `cache_identity` derived from what they are (layer, expert and catalog tensor
  for experts; name and byte range otherwise), and no identity means no caching, so a caller that
  omits it loses speed rather than correctness.

  Separately, the parity gates were built from dyadic values — a fixed scale of 0.25, a delta of
  0.125 and a k/256 input — which makes f32 and f64 accumulation agree to the bit and let kernels
  accumulating in `fma.rn.f32` pass while the reference summed in `double`. The gates now vary the
  FP16 exponent per block, fill the mantissa, and use non-dyadic input.

- The GPU is now in the inference path. The graph offers each packed matvec to an optional
  accelerator and falls back to the CPU kernel on refusal, so matrices without a device kernel run
  unchanged. Q8_0 matrices stay resident in VRAM after their first upload — 2.90 GB across a
  24-layer pass — because a decode step revisits the same matrices and copying per launch would cost
  more than the kernel saves. The driver also caches its compiled module per context, holding its
  own context reference so the module cannot outlive it.

  At 24 layers with identical output text: 24,328 ms per token on CPU against 22,246–23,509 ms with
  2,728 matvecs accelerated and 4,899 declined. Far short of the kernel's own 5.4×, and the reason is
  measured: 2.65 ms per accelerated call against 0.558 ms in isolation, because each call still
  pushes and pops the context around three synchronous operations. The round trips, not the kernel,
  are what to fix next.

- Added the first quantized CUDA kernel: a Q8_0 matvec in hand-written PTX, loaded through the same
  toolkit-free driver path as the RMSNorm gate. On the reference RTX 4070 Super a 4096×4096 matvec
  runs in 0.5735 ms against 3.0305 ms for the twelve-threaded CPU kernel — 5.28× — with a maximum
  absolute error of 3.8147e-06 against a 2e-2 gate. Blocks accumulate in the reference's order but in
  FP32 rather than FP64, so this is a tolerance gate, not a parity gate. The kernel is deliberately
  naive: one thread per row, no coalescing, no warp reduction, matrix copied per call. Nothing is
  wired into the graph yet, so the model does not run faster because of it — this establishes the
  kernel and its numerics. `cuda-info --matvec-smoke` reports both timings.

- The complete DeepSeek V4 Flash 0731 model executed a forward pass on the reference PC: all 43
  layers, the checkpoint's own routers, 258 expert bundles and 1,770,651,648 bytes of expert payload
  — exactly the `no_cache_expert_bytes_per_token` figure derived from metadata at an earlier
  milestone. Token 0 produced 129,280 logits and next token 20; weights loaded in 3.97 s and the
  forward took 29.4 s with scalar CPU kernels. The process peaked at 7.17 GB while free host RAM
  never fell below 11.7 GB against the Governor's 8 GiB floor, and everything was released at the
  end. Raising depth in steps exposed two defects: the indexer's head-weight projection was bound
  outside the indexer namespace, which only surfaces at layer 2 where the first 4:1 compressor
  appears, and the reference layer ceiling was still sized for reduced fixtures at 16. See the
  [complete forward validation](docs/reports/M9_DEEPSEEK_COMPLETE_FORWARD_VALIDATION.md). This is one
  forward pass, not generation, and 29.4 s per token is correct rather than usable.

- A completed checkpoint verification now leaves a receipt, so the 82.54 GB SHA-256 pass that gates
  execution is paid once rather than before every prompt. The receipt records each file's size and
  last-write time beside the hash that was confirmed, keyed on the manifest's own hash; if both
  still match, the bytes on disk are the bytes that were verified. This trusts filesystem metadata,
  which is weaker than rehashing — a file edited in place preserving size and timestamp would pass —
  and it is bounded: any mismatch, a missing or malformed receipt, or a changed manifest forces the
  full verification again, and the receipt is written only after every file passes.

- Inference now uses half the machine's logical processors rather than a fixed six, which predated
  any threaded kernel. Combined with the threaded matvecs, generation went from 60,301 ms per token
  serial to 32,730 at six threads and 25,891 at twelve, with byte-identical output throughout.

- Packed matvec rows now split across the Governor's thread budget, taking generation from 60.3 s to
  32.7 s per token on the reference PC with byte-identical output. Rows are independent and each is
  still computed exactly as it would be serially, so the ggml parity fixtures are unaffected. The
  gain falls short of the thread count because only the matvecs are threaded — attention,
  normalization, RoPE and the Hyper-Connection paths remain serial, which is where the next
  speedup lives.

- **The complete model generates coherent text.** `"La capitale d'Italia e"` produced
  `" una grande citta cosmopol"` through all 43 layers of the official checkpoint — six tokens at
  60.3 s each, 16 GB of expert payload read from NVMe, 7.62 GB resident, free host RAM never below
  10.3 GB. The same prompt at 12 layers returns gibberish, which is the clearest evidence that the
  full-depth path is executing the real model rather than a fixture. `--max-tokens` routes through
  the reference decode loop with KV reuse; a minute per token is correct, not practical.

- The model now answers a text prompt. The checkpoint's vocabulary proved identical to the pinned
  tokenizer — 129,280 tokens index for index, all 127,741 merges — so `--prompt` encodes through the
  implementation already validated against Hugging Face, with `--chat` applying the checkpoint's own
  template. `"La capitale d'Italia e"` ran all 43 layers, resolved 1,069 expert bundles across
  7.34 GB of payload and predicted `" una"`. Each layer now releases its experts when the next layer
  asks for its own, so peak expert residency is two layers' worth instead of 43: the same prompt
  would otherwise have needed 13.6 GB of experts alone, and now the whole pass fits in 7.62 GB with
  free host RAM steady at 12 GB.

- Added `deepseek-forward`, which runs the real graph to a chosen depth straight from a complete
  checkpoint, and refuses in advance what will not fit. Its preflight sums the exact resident cost
  from the catalog — layer trunks, globals and the expert-cache ceiling — and compares it against the
  live safe RAM budget before allocating anything. On the reference PC a full 43-layer pass needs
  7,056,343,388 bytes against an 8,053,104,640-byte budget while 8 GiB of host RAM stays
  untouchable. Layer trunks and globals stream one complete tile per tensor under their own
  reservations, each layer keeps its own bounded expert cache so one layer cannot evict another's
  working set, and the process guard is sampled at every layer boundary. `--layers` keeps depth
  explicit, and the output states whether the pass was the complete model or a partial one.

- Executed the first real DeepSeek layer from official payload. `deepseek-layer-probe` relocates the
  verified catalog onto the imported artifacts, loads trunk and globals through their governed
  reservations and runs one complete layer with the checkpoint's own router selecting the experts.
  Token 0 executes experts 254, 222, 245, 200, 53 and 35, reads 38,928,384 expert bytes and produces
  129,280 logits in `[-16.0828, 15.4815]` in 1.07 s; two runs agree exactly. Q8_0, Q5_K, Q6_K, BF16,
  I32, IQ1_S, IQ3_XXS and Q4_K all execute together for the first time. This is one layer of 43 and
  the command says so — `complete_model: false` — so these logits are not the model's output.
  Relocation is partial by design: a tensor no artifact provided keeps an absent source and fails on
  open, which is what stops a token whose experts were never imported from executing against wrong
  bytes.

- Imported the first official DeepSeek V4 Flash 0731 weight payload and made the routing decision
  that justifies it. Hash-routed layers select experts by table lookup, so `blk.0.ffn_gate_tid2eid`
  — part of the trunk — determines the selection for a token without executing anything. The
  bounded importer fetched the 21-range / 138,867,160-byte layer-0 trunk, the new
  `deepseek_v4_resolve_hash_route` read that table through the governed rehashed range path, and
  token 0 resolved to experts 35, 53, 200, 222, 245 and 254. The resulting plan reports
  `official-tid2eid-verified`, which opened the previously refused `--scope selected-experts` and
  imported those six complete bundles (18 ranges, 38,928,384 bytes). Learned-router layers,
  out-of-vocabulary tokens, absent or mis-sized tables and out-of-range expert ids all fail closed.
  82.36 GB of the checkpoint remains undownloaded.

- Proved the imported trunk is consumable by the native graph. `deepseek-gguf-range-bind
  --load-tensor-set` rereads all 21 official ranges into one governed RAM buffer, rehashes every
  packed subspan and maps them onto their declared consumers: 11 matrices, 9 vectors and the routing
  table, 138,867,160 resident bytes in 371 ms on the reference PC. Two opt-in tests
  (`VASTLOOM_DEEPSEEK_V4_TRUNK_ARTIFACT`) pin both the route and this consumer contract. The layer
  is bound, not executed.

- Separated environment refusals from defects across the governed test suite. A Safety Governor
  refusal caused by host memory pressure was reported as a test failure, so an unrelated process
  holding 20 GiB made four tests fail on one run and five different tests fail on the next. Every
  registered test now carries `SKIP_RETURN_CODE 75`, the CLI's own `resource_exhausted` code, and
  CTest reports those contracts as skipped. `tests/host_pressure.py` gives the Python drivers the
  same outcome and recognises only the three refusals raised by `make_safe_runtime_policy` before
  any work starts; VRAM budget, allocation, parity and malformed-output failures still fail. The
  two zero-payload GGUF contracts moved from one positional `PASS_REGULAR_EXPRESSION` each to
  explicit per-key JSON comparison of the same 24 and 7 pinned values, so field reordering can no
  longer weaken them.

- Added `docs/VALIDATION_MATRIX.md`, which maps every documented claim to the test that defends it,
  the CMake variable that enables that test and the artifact it requires. It records what the
  13 always-on tests cover, which 11 tests are opt-in, and that neither real-model parity nor real
  GPU execution is re-checked by CI. `tools/fetch_deepseek_test_slices.py` restores the three
  pinned expert slices (6,488,064 bytes) behind an explicit `--allow-download` flag with per-range
  length and SHA-256 verification, so the M7 real-expert evidence is reproducible on a clean
  machine. The eight earlier alpha-era reports are now listed in `docs/reports/README.md` instead
  of remaining unreferenced.

- Added the first real CUDA math path without a toolkit dependency: one Governor-owned packed
  FP32 buffer, synchronous host/device copies and an embedded scalar PTX RMSNorm. The explicit
  `cuda-info --rmsnorm-smoke` gate matches the DeepSeek CPU reference within `3e-5` on the RTX
  4070 Super through native Windows and WSL, while deterministic faults cover copies, launch,
  release and reservation cleanup. This is not yet a CUDA model backend.

- Added a zero-payload complete-catalog contract for the pinned 0731 GGUF checkpoint. It validates
  six global consumers, all 11,008 routed bundles and 33,024 expert slices, then reconciles 1,328
  physical tensors, 34,223 logical descriptors, 6,986,088,796 always-active bytes,
  75,547,803,648 routed bytes and 82,533,892,444 known tensor bytes. Q4_K, IQ2_XXS and MXFP4
  receive layout-only accounting and remain explicitly non-executable without parity kernels.

- Pinned the official `deepseek-ai/DeepSeek-V4-Flash` checkpoint at immutable revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`; removed the unsupported provisional
  “0731” identity from the active planning manifest.
- Added a 19,404,391-byte metadata/header import that uses exact HTTP ranges, pinned
  SHA-256 values and atomic publication instead of downloading 159,617,149,040 bytes
  of weight shards.
- Added `deepseek-inspect` with exact validation for 46 shards, 69,187 tensors,
  Safetensors shapes/offsets, FP8/FP4 storage dtypes and routed-expert byte layout.
- Added native DeepSeek V4 Unicode pre-tokenization, its 129,280-token byte BPE,
  bounded added-token trie and basic system/user/assistant chat framing. Exact IDs match
  Hugging Face tokenizers 0.22.1 across committed multilingual reference vectors.
- Added bounded native DeepSeek reference equations and deterministic vectors for RMSNorm,
  learned/hash routing, routed SwiGLU, Hyper-Connections, base RoPE and selected-latent
  sparse attention. These are operator tests, not a DeepSeek forward or generation claim.
- Added a deterministic 147,136-byte graph exported from the SHA-256-pinned official
  `model.py`, a bounded native two-layer FP32 prefill and one checkpoint-shaped MTP block.
  Main and MTP logits, proposed tokens, learned/hash expert routes and raw/compressed attention
  indices pass the declared parity gate; full-checkpoint and speculative DeepSeek execution
  remain unsupported.
- Extended the fixture to 159,533 bytes with five official single-token decode steps. Native
  main/MTP logits, routes and raw/compressed attention indices match across circular-window and
  overlapping 4:1 cache updates; state mutation is transactional and aggregate-budgeted before
  allocation.
- Recorded, without downloading weights, the newer official DeepSeek V4 Flash 0731 release,
  its Unsloth GGUF profiles and Laguna S 2.1 as separate compatibility targets. None inherits
  the preview checkpoint's execution claims; the universal tensor/expert residency contract
  and per-adapter acceptance ladder are now explicit.
- Added a bounded native GGUF v3 reader with strict metadata/tensor/alignment/offset limits,
  exact GGML encoded-byte accounting, split-header catalogs and header-prefix mode. The CLI can
  inspect one file or build a metadata-only multi-shard DeepSeek catalog without reading payloads.
- Bound the pinned 82,539,237,792-byte 0731 `UD-IQ1_S` profile's 1,328 physical tensors to
  34,223 logical descriptors and 11,008 packed expert bundles from bounded headers. The derived
  1,770,651,648 no-cache expert bytes/token is a planning value, not measured SSD traffic;
  complete quantized execution remains unsupported.
- Added the exact 0731 `UD-IQ1_S` three-object importer and `deepseek-gguf-checkpoint` source
  gate. Metadata-only mode downloads and verifies 5,343,584 bytes; full source ranges require an
  immutable manifest, every complete-file SHA-256 and live single-tier expert admission.
  `--no-verify` remains metadata-only, the official 82.54 GB payload was not downloaded, and the
  CLI still reports `execution_supported=false` until a verified full-checkpoint layer/token
  consumer exists.
- Added `deepseek-gguf-probe-plan` and a native non-executing range-plan API. It validates one
  exact layer, exactly top-k unique caller-supplied expert IDs, complete W1/W2/W3 bundles,
  immutable source identities, bounded range counts and overflow-safe totals. Real layer-0
  metadata yields 21 trunk plus 18 expert ranges / 177,795,544 logical bytes; routing authority,
  full-payload presence and execution remain explicitly false.
- Corrected metadata-only logical sources to name the immutable complete GGUF shard objects rather
  than local `.header` snapshots. Loaders still reject metadata-only catalogs, so this preserves
  accurate future range identity without widening execution authority.
- Added `tools/import_hf_ranges.py` for explicit-authority partial payload acquisition. It accepts
  only the pinned DeepSeek 0731 profile, strictly validates a hash-verified plan, enforces selected
  byte/range/disk bounds, requires exact HTTP 206/content ranges, hashes every payload and
  atomically publishes a read-only `vastloom.range-artifact.v1` under `models/optimized`.
- Kept partial artifacts fail-closed: their manifest declares the complete checkpoint unverified
  and execution unsupported; caller-supplied expert/all scopes are rejected before network access.
  Offline Windows/Linux tests cover exact ranges, malformed plans, explicit authorization, wrong
  HTTP ranges, crash cleanup and immutable reuse. The real 138,867,160-byte trunk passed dry-run
  only and no official payload was downloaded.
- Preserved exact `blk.N.*` identity for all non-routed GGUF tensors and added deterministic
  per-layer trunk plans with a 64 MiB ceiling. The real 0731 metadata maps all 43 layers into
  1,193 tensor tiles; a new governed fixture loader covers immutable identity, Unicode paths,
  budget refusal, cancellation, short reads, leases and zero-reservation cleanup. No official
  weight payload or complete DeepSeek layer was executed.
- Added bounded scalar Q8_0, Q5_K, Q6_K, BF16 and F32 decode/matvec kernels plus a separate I32
  router-table decoder. Synthetic raw blocks match the pinned ggml oracle, caller output is
  transactional through bounded scratch, and the real 0731 metadata now reports native storage
  handling for all 1,193 layer-trunk tensors / 6,390,088,008 encoded bytes. This is storage/math
  coverage only for the official payload: the 82.54 GB payload remains absent and
  `execution_supported=false`.
- Added deterministic ggml-compatible Q8_0 reference encoding and connected packed matrix/I32
  storage directly to the reduced DeepSeek graph. Its 11 trunk/shared matrices use Q8_0, routed
  W1/W3 now use eight IQ1_S matrices and routed W2 uses four IQ3_XXS matrices. All 23 matrices in
  main layer 0 execute through transactional storage dispatch; prefill and incremental main/MTP
  logits, routes and attention indices match an independently dequantized oracle. Official
  payload execution and generation remain open.
- Added immutable resident byte buffers shared by the bounded trunk and expert loaders. A
  file-backed reduced DeepSeek layer now binds 12 complete trunk tiles and four indivisible expert
  bundles directly into the graph, retains exactly 8,180 governed mixed-format payload bytes
  across prefill and decode, blocks eviction while referenced, and returns all reservations to
  zero on release. Official payload execution remains open.
- Split the reduced DeepSeek MoE graph at its authoritative router boundary and added a reusable
  bounded selected-expert cache. The graph deduplicates routed IDs, requests only complete selected
  bundles, executes their retained packed matrices and records hit/miss/logical-byte telemetry. A
  two-entry synchronous RAM LRU loads experts 2/3 on the first token, reuses them on the next and
  replaces them with 0/1 on the third while matching the quantized oracle. Oversized batches,
  malformed providers, insufficient budgets and eviction of in-flight bundles fail closed. This is
  a reduced fixture path, not official-payload DeepSeek generation or an SSD-speed result.
- Added a bounded scalar IQ1_S dequantization/matvec reference kernel derived from pinned ggml
  equations and licensed codebook data. Synthetic vectors and one exact 1,638,400-byte real
  DeepSeek 0731 expert-gate slice match the upstream oracle bit for bit across all 2,048 rows.
- Added exact scalar IQ3_XXS decode/matvec plus a transactional IQ1_S gate/up, FP32 SwiGLU and
  IQ3_XXS down expert path. Three pinned real ranges totaling 6,488,064 bytes now pass through a
  one-worker governed packed-RAM load and match the ggml oracle bit for bit across all 4,096 final
  outputs; complete official-payload layer/token execution and model generation remain open.
- Bound canonical 0731 GGUF expert identities directly from the executable logical catalog into
  strict W1/W2/W3 cache descriptors. The same 6,488,064-byte real IQ1_S/IQ3_XXS expert now passes
  through the production router-first one-entry RAM cache: the first resolution is one exact miss,
  the second is a zero-read hit, retained leases block premature eviction, malformed mixed-format
  descriptors publish no output and every Governor reservation returns to zero. This connects one
  real selected expert to the cache/kernel boundary, not a complete DeepSeek layer or generation.
- Connected selected DeepSeek misses to a bounded asynchronous batch/acquire scheduler. The
  reduced graph submits every deduplicated complete bundle, waits only at that expert's fixed
  compute turn and preserves numerical reduction order while recording load, wait, compute and
  read-service/compute-intersection timing. A controlled fixture proves positive overlap with
  exact oracle parity; the real 6,488,064-byte expert passes the same future/cache/kernel boundary.
  One active batch per cache, full-request queue capacity, drain-on-abandon and exact successful
  wasted-prefetch accounting keep this correctness slice bounded. This is not a physical-NVMe,
  complete-layer or throughput result.
- Made the controlled overlap regression deterministic by explicitly coordinating the second
  fixture read-service start before its bounded 50 ms service window. This changes test timing
  only, not the production scheduler; the native suite then passed 50/50 repetitions and TSan.
- Extended the complete reduced layer/token gate to the real 0731 expert format combination:
  Q8_0 trunk/shared tensors, I32 hash routing, IQ1_S W1/W3 and IQ3_XXS W2. The 8,180-byte
  file-backed fixture produces four 1,384-byte indivisible expert bundles; synchronous and
  asynchronous cache execution matches the dequantized oracle for prefill/decode, and selected IQ
  truncation plus malformed W2 format fail transactionally. These are deterministic reduced
  bytes, not official checkpoint payload or a performance result.
- Documented the tested platform boundary explicitly: Windows/MSVC and Linux GCC/Clang CPU paths
  are validated, while macOS and Metal remain unvalidated targets.
- Added a public reference-PC admission gate: a model must fit one complete expert plus the
  required active block, KV, scratch and staging inside live post-reserve RAM/VRAM budgets. Kimi K3
  is now capacity-study-only on the 32 GiB RAM / 12 GiB VRAM machine unless an exact profile passes
  that gate.
- Implemented that gate in the native planner: strict MoE manifests now require complete expert,
  layer, top-k, context and minimum-working-set metadata; scratch is an explicit placement; and
  each plan names the real RAM or VRAM tier that can contain one complete expert. The Resource
  Governor now enforces one aggregate ceiling across RAM destinations, staging and scratch, with
  deterministic acceptance, pressure, rejection and overflow tests.
- Replaced repeated GGUF metadata `seekg` calls with verified sequential discards. The same real
  5.26 MB header on Windows-mounted WSL storage improved from 37.42 s to 0.11 s while preserving
  strict bounded/truncation behavior.
- Made capacity-sensitive tests deterministic without weakening production disk reserves and
  gated CodeQL/Dependency Review until the repository is public or GitHub Advanced Security is
  explicitly enabled through the `VASTLOOM_ENABLE_GHAS` repository variable.
- Fixed CLI path construction so Windows wide-character arguments remain Unicode-native through
  `std::filesystem`; an accented GGUF filename is now covered end to end.
- Added the shared logical tensor catalog and bound all 69,187 pinned DeepSeek preview tensors
  to role, shape, dtype and immutable source ranges. Its 67,584 routed tensors form 11,264
  main/MTP expert bundles whose six tensors coalesce into exactly two ranges per expert; no
  payload read or execution support is claimed.
- Added the governed multitier residency state machine. RAM plus bounded staging and later VRAM
  destinations are reserved before transition publication; failed compound reservations roll
  back, while pins and RAII use leases prevent unsafe eviction. The state-only slice allocates no
  tensor buffer and makes no new inference claim.
- Added the first governed packed-expert load transaction: canonical source ranges are read into a
  reserved compact RAM buffer, every expert tensor receives an exact packed slice and an RAII use
  lease prevents eviction until the bytes are destroyed. The synchronous fixture baseline is not
  a DeepSeek or physical-NVMe performance claim.
- Added a shared bounded asynchronous expert-load pool with one to four fixed workers, a queue of
  at most 64 requests, pre-enqueue RAM/staging reservations, exclusive transfer leases, safe
  queue-full rollback, shutdown cancellation and queue/service/total telemetry. This enables a
  future compute/I/O overlap path but does not itself prove DeepSeek throughput.
- Added an explicit `--allow-large-download` gate for checkpoint imports containing at
  least 20 GiB of weight payload. DeepSeek execution remains unsupported and is reported
  as such by the CLI.
- Added path-scoped device-I/O snapshots: Windows maps the checkpoint volume to
  `PhysicalDriveN` and samples `IOCTL_DISK_PERFORMANCE`; Linux accepts non-virtual
  `/sys/dev/block/.../stat` mappings and otherwise reports the counter unavailable.
- Added the fail-closed `direct-unbuffered-v1` cache-bypass protocol for bounded OLMoE
  expert reads. It requires async no-cache execution, a trustworthy device counter and
  measured device bytes at least equal to logical expert bytes; it never flushes the
  global operating-system file cache.
- Validated three repeatable Windows cache-bypass runs with identical token `47111`,
  1,610,612,736 logical bytes and 1,610,743,808 whole-device bytes per run.
- Extended native OLMoE traces from 12 to 19 fields with submission, queue, ready,
  compute-window, route-order and reduction-order telemetry while retaining all earlier
  trace formats.
- Added a bounded trace replay that compares fixed-route execution with theoretical
  ready-expert compute order while preserving the authoritative reduction order.
- Added opt-in two-stage OLMoE reads: contiguous 8 MiB gate/up plus 4 MiB down requests,
  two workers, bounded queue/staging reservations, exact byte accounting and down-read
  versus gate/up-compute overlap telemetry.
- Validated staged reads against the pinned Transformers logits/routes and measured a
  46.0% median warm buffered-pipeline improvement over whole 12 MiB reads on three paired
  WSL runs. This is not a physical-NVMe or DeepSeek throughput claim.

## 0.4.0-alpha.4 — 2026-08-09

- Renamed the public project, CLI, C++ namespace, headers, CMake targets and current
  documentation to **Vastloom** with the slogan “Run giant models. Keep your machine yours.”
- Preserved read compatibility for legacy manifests, imported-checkpoint metadata and
  model-store markers so existing verified model downloads remain immutable and reusable.
- Added Apache-2.0 licensing, community governance, support and security policies,
  contribution templates, CodeQL, dependency review and release packaging automation.
- Added a purpose-built README cover and two-to-one social preview, plus public brand and
  release documentation.
- Regenerated the bundled synthetic fixture under the Vastloom model identity and retained
  the original native binary and heartbeat wire magic for format compatibility.

Historical entries below retain the project name and command spelling used by those releases.

## 0.4.0-alpha.3 — 2026-08-09

- Added cross-platform process-I/O snapshots around the real OLMoE forward path.
- Exposed Windows read operation/transfer deltas and Linux logical plus storage-layer
  read deltas while preserving their different semantics in text and JSON output.
- Replaced fragmented Windows asynchronous stream reads with Unicode-safe bounded
  `CreateFileW`/`ReadFile` ranges; one expert now requires one 12 MiB operation.
- Added exact unique-expert reuse-distance telemetry with cold/reused access counts,
  reuse ratio, mean, P50, P95 and maximum distance.
- Added Unicode-path, counter-semantics, operation-fragmentation and real repeated-route
  regression tests.

This slice measures process-visible I/O and routing locality without claiming physical
NVMe traffic. Buffered cache state, cold-cache behavior, device counters, OLMoE decoding,
CUDA and DeepSeek remain outside the validated scope.

## 0.4.0-alpha.2 — 2026-08-09

- Connected the real OLMoE expert path to the reservation-backed bounded I/O pool.
- Added opt-in synchronous/asynchronous selection, one-worker prefetch depth one or two,
  12 MiB request bounds, cancellation and per-request timeouts capped at 60 seconds.
- Coalesced each pinned expert's three validated contiguous BF16 tensors into one bounded
  read while preserving the native gate/up/down layout and reference logits.
- Added request queue/service/total timing plus expert prefetch count, foreground wait
  and measured read-service/CPU overlap telemetry.
- Extended atomic OLMoE CSV traces from nine to twelve fields while retaining compatibility
  with four-field baselines and nine-field legacy traces.
- Added deterministic I/O worker, short-read, allocation, cancellation and timeout fault
  injection, including real-checkpoint fail-closed integration coverage.
- Retained synchronous I/O as the default and capped the asynchronous OLMoE adapter to
  one worker, two outstanding prefetch slots and Resource Governor staging reservations.

This slice validates bounded buffered read/CPU overlap without changing model output. It
does not claim physical NVMe bytes, cold-cache disk throughput, direct I/O, OLMoE text
generation, CUDA or DeepSeek support.

## 0.4.0-alpha.1 — 2026-08-09

- Pinned and fully verified `allenai/OLMoE-1B-7B-0924-Instruct` revision
  `7f1c97f440f06ce36705e4f2b843edb5925f4498` with three immutable BF16 shards.
- Added exact OLMoE configuration, tokenizer, chat-template and 3,219-tensor layout
  validation without requiring tensor payloads to reside in RAM.
- Added a native CPU OLMoE reference graph with RMSNorm, Q/K normalization, RoPE,
  causal attention, authoritative softmax top-8 routing and weighted SwiGLU experts.
- Added explicit NVMe-to-RAM expert streaming: the 953,421,824-byte always-active
  trunk remains in reserved RAM while one 12 MiB expert is read at a time.
- Added bounded no-cache, LRU, LFU and static-hot expert-cache policies, with a hard
  two-GiB cache ceiling and Resource Governor reservations before allocation.
- Added `moe-inspect` and `moe-logits`, atomic route telemetry, exact expert byte/I/O
  accounting, timeout/Ctrl+C checks and metadata-only execution refusal.
- Added an independently bounded Transformers 4.45.2 oracle and committed one- and
  two-token fixtures for logits, greedy token, route IDs and routing weights.
- Extended trace simulation to accept native telemetry and compare no-cache, LRU,
  LFU and static-hot policies.

This milestone proves correct real MoE routing and synchronous NVMe-to-RAM expert
streaming on CPU under a host-derived memory cap. It does not yet provide OLMoE text
generation/chat, CUDA, asynchronous I/O/compute overlap or a DeepSeek adapter.

## 0.3.0-alpha.1 — 2026-08-09

- Added a bounded JSON parser and defensive Safetensors catalog/reader.
- Added a 300 GiB-reserve model store, immutable atomic Hugging Face importer,
  and host-derived real-model runtime policy.
- Added Windows process memory Job Object support and live commit/headroom checks
  for real-model execution paths.
- Added `store-info`, `runtime-policy`, and Safetensors-aware `inspect` commands.
- Added the first genuinely pretrained model adapter, pinned to
  `HuggingFaceTB/SmolLM2-135M-Instruct` revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Added the model's GPT-2 byte-level BPE tokenizer, exact chat template, BF16
  Safetensors loader, Llama CPU reference kernels, grouped-query attention,
  incremental KV cache, greedy/seeded generation and persistent chat history.
- Added trusted Transformers parity fixtures for tokenization, logits, greedy output
  and multi-turn chat, plus a native real-model self-test.
- Added bounded process-memory enforcement, six-thread maximum execution,
  cancellation and full-checkpoint verification by default for the real path.

This milestone proves one real published dense checkpoint on the native CPU path. It
does not yet include MoE execution, NVMe expert streaming or CUDA.

## 0.2.0-alpha.2 — 2026-08-09

- Added the first native end-to-end token-generation path.
- Added the bounded `SGTMV001` TinyTransformer model format and strict loader.
- Added byte-level prompt encoding, CPU LayerNorm/attention/MLP kernels and incremental
  KV caching.
- Added greedy decoding and deterministic seeded top-k/temperature sampling.
- Added `stargate chat`, `stargate model-info` and `stargate self-test`.
- Added interactive streaming, one-shot JSON output, generation timeouts, token limits,
  system-prompt override and response statistics.
- Added the 358.91 KiB `stargate/tinychat-v1` validation model, exact manifest and five
  deterministic response vectors.
- Added a reproducible PyTorch training/export tool that atomically publishes the
  model, vectors and SHA-256 manifest only after all probes pass.
- Added end-to-end CLI integration tests, integrity-failure tests, malformed-model
  tests, deterministic-sampling tests and additional Resource Governor checks.
- Added explicit generation finish reasons and valid-Unicode repair for arbitrary sampled
  bytes in JSON output.
- Added manifest-to-binary metadata cross-checks for model id, layer count and context
  length.
- Added concurrent generation coverage and ThreadSanitizer validation of the shared model
  path.
- Added a UTF-8 Windows `wmain` entry point and UTF-8 console configuration for accented
  CLI prompts.
- Added integration coverage for direct `.sgtm` loading and explicit quick manifest mode.
- Added CMake installation of the bundled validation model under `share/stargate/models`.
- Regenerated the committed fixture from the current exporter and verified byte-identical
  output across independent training runs.
- Fixed model/governor destruction order so weight reservations are released before
  the owning governor is destroyed.
- Added UTF-8 MSVC compilation and expanded warning-as-error/sanitizer validation.

This release validates the native CPU inference pipeline. It does not yet include a
DeepSeek/MiniMax/Kimi adapter, GPU kernels or NVMe-backed generation.

## 0.1.0-alpha.1 — 2026-08-08

- Added C++20/CMake project foundation.
- Added hardware profiler and JSON reporting.
- Added conservative hardware-adaptive execution planner.
- Added atomic resource reservations and safety state machine.
- Added bounded asynchronous I/O with reservation-backed buffers.
- Added strict model manifest, SHA-256 and path/symlink confinement.
- Added immutable model-store layout command.
- Added atomic heartbeat and independent watchdog.
- Added MoE trace simulator with LRU/static-hot policies and bytes/token metrics.
- Added unit tests, warning-as-error build, ASan/UBSan and CI definitions.

No model inference kernels are included in this release.
