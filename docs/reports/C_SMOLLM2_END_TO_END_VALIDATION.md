# C — SmolLM2 dense end-to-end validation

Validation date: 2026-10-26. This is the small-model end-to-end path requested after the full
DeepSeek checkpoint download was cancelled: it exercises a complete, locally executable model
without touching the range-slice track.

## Pinned model

- **Model**: `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Revision**: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- **Imported payload**: `model.safetensors` 269,060,552 B plus tokenizer/config files
  (2,104,556 B `tokenizer.json`, 800,662 B `vocab.json`, 466,391 B `merges.txt`, 3,764 B
  `tokenizer_config.json`, 861 B `config.json`, 132 B `generation_config.json`, 655 B
  `special_tokens_map.json`); total under 273 MB.
- The store was created with `vastloom store-init` (50 GiB reserve) and imported with
  `tools/import_hf_model.py --model smollm2 --reserve-gib 20 --max-mib-per-sec 50`. Every file was
  size- and SHA-256-verified by the importer.

## Tests executed

| Test | Result |
|---|---|
| `tests/test_smollm2_cli.py` against `tests/fixtures/smollm2_reference.json` | **PASS** |
| `tests/test_smollm2_serve.py` | **ALL PASSED (8 cases)** |

The CLI contract covers `model-info` full verification (272 tensors), tokenizer parity and
round-trip, reference top-logit cases, greedy generation, seeded sampling determinism, capability
module inert/live behavior, and deterministic active-module generation. The serve contract covers
the local endpoint with eight cases, including streaming and request handling.

The tests were run from a working directory whose volume has more than the hard-coded 300 GiB
storage reserve; the model store itself is a local, git-ignored build artifact. This is an
environmental workaround for the Governor's host reserve, not a code or safety change.

## What this establishes

- The dense SmolLM2 path is **fully executable end-to-end** from a pinned import: tokenizer,
  reference logits, generation, capability modules and the local serve endpoint.
- The import and test tooling can be exercised without downloading a large checkpoint; this is the
  practical counterpart to the reduced DeepSeek range track.
- The measured payload is small enough to fit the earlier “under 10 GB” constraint.

## Limits

- This is a 135M dense model, not a Mixture-of-Experts and not a large-model result.
- No throughput, memory-hierarchy, GPU, or full-model DeepSeek claim follows from it.
- The model store and the temporary working directory used for the storage-reserve workaround are
  local validation state under `build/` or outside the repository; nothing is committed.
- The serve test is a functional contract, not a production or concurrency benchmark.
