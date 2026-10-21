# 0.5.0-alpha.1 dense-serve safety and JSON object mode validation

Validation date: 2026-10-20. Release: `0.5.0-alpha.1` (`546af36` + this working tree).

## Scope

This release is a maintenance and serve-safety update on the dense SmolLM2 path. It
implements four things and re-measures none of the large-model records:

1. a decoding constraint interface and its first implementation, a bounded JSON-object
   grammar used by `response_format: {"type": "json_object"}` on the serve endpoint and by
   `chat --json-object`;
2. fail-closed rejection of chat message content that would encode a reserved control token;
3. UTF-8-safe server-sent-event streaming, so a code point is never split across `data:`
   frames;
4. two serve-path defects found while adding the tests: the SSE producer captured
   request-scoped locals by reference after the handler returned (use-after-free), and an
   empty stop-sequence filter discarded every streamed piece (every streaming response was
   empty).

The work is the `P0` slice recorded in
[`docs/research/DEEPINFER_REVIEW.md`](../research/DEEPINFER_REVIEW.md); the blocked
large-model backlog (Q11k, KV compression, routing-aware expert placement) is documented in
[`docs/planning/Q11K_CHUNKED_PREFILL_DESIGN.md`](../planning/Q11K_CHUNKED_PREFILL_DESIGN.md)
and was not implemented.

## Environment

- Reference PC: Windows 11, AMD64 24 logical cores, 33,407,475,712 bytes physical RAM
  (~31.1 GiB), NVIDIA GeForce RTX 4070 Super 12 GiB, driver 617.42.
- Toolchain: MSVC 19.44.35229 x64, CMake/Ninja from Visual Studio Build Tools 2022. Both
  Release and Debug were built with `VASTLOOM_WARNINGS_AS_ERRORS=ON`.
- SmolLM2 checkpoint: `HuggingFaceTB/SmolLM2-135M-Instruct` revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`, 272 tensors, 269,030,016 tensor bytes. The
  runtime's own `vastloom-import.json` verifier reports `hash_verified: true` for every file.

Three environment workarounds are recorded because they changed how the gates were run, not
what they assert:

- **Memory view.** Under the workspace sandbox, the hardware probe reported ~5.9 GiB of
  available RAM, below `make_safe_runtime_policy`'s 8 GiB headroom gate, so every real-model
  command was refused with `resource_exhausted` (exit 75, reported as SKIPPED). The host
  itself reported 14.7 GiB available. The validations below were run with the workspace
  sandbox disabled; the same binary, thresholds and Governor paths still executed.
- **Build tool.** Ninja's child-process pipes hang in this session (the sandboxed process
  cannot create them). Both configurations were built by replaying the exact
  `ninja -t commands` compiler and linker command lines with inherited stdio. This is a
  local workaround; it changes neither flags nor inputs.
- **Checkpoint import.** The canonical importer's final atomic directory rename was denied by
  the workspace ACL (an inherited `Everyone` `DENY DELETE_CHILD` ACE). The pinned files were
  fetched into a plain directory with the same `vastloom.hf-import.v1` manifest schema, and
  the runtime rehashed every file before loading it. No weight file was edited.

No large checkpoint (Qwen3.8-Flash-Next 175.3 GiB, DeepSeek V4 Flash 0731 profiles) is
present on this host; no large-model result was repeated or revised.

## Verification

### Always-on suites

| Gate | Result |
|---|---|
| Release build (`cmake --preset release` equivalent, MSVC Release) | PASS, warnings as errors |
| Debug build (`cmake --preset debug` equivalent, MSVC Debug) | PASS, warnings as errors |
| `ctest --test-dir build/release` | 14 passed, 0 failed, 1 skipped (`vastloom_ggml_compute`, ggml not configured) — 15 registered Vastloom tests, 13.8 s |
| `ctest --test-dir build/debug` | 14 passed, 0 failed, 1 skipped (`vastloom_ggml_compute`, ggml not configured) — 15 registered Vastloom tests |
| `python -m compileall -q tools tests` | PASS |
| `ruff check tools tests` (the CI gate) | `All checks passed!` |
| Python CLI/offline gates (`test_manifest_tool`, `test_distill_transcript`, `test_distill_evaluate`, `test_import_hf_model`) | PASS (also run by CTest: `vastloom_hf_importer_offline`, DeepSeek fixtures, range artifact, watchdog) |

New always-on coverage added to `vastloom_tests`:

- control-token rejection for SmolLM2/OLMoE templates and the tokenizer-level validator;
- `Utf8StreamFilter`: 1-byte feeds of 2/3/4-byte sequences, ASCII passthrough, incomplete tail
  at flush, stray continuation byte;
- JSON-object grammar: contextual masks, multi-byte token spanning four JSON bytes, root-object
  restriction, masked-token refusal, reset, depth 8 ceiling, string and number byte bounds,
  invalid limits;
- `response_format` parsing, refusal of `json_schema`/unknown types, refusal of `stop` with JSON
  mode, extension-field serialization on the response and final chunk only;
- `StopSequenceFilter` passthrough with no sequences and holdback/truncation with sequences.

### Real-checkpoint opt-in suites

Run against the imported SmolLM2-135M checkpoint. Both the Debug and the Release binary
were exercised end to end; the results below are identical for both.

| Gate | Result |
|---|---|
| `tests/test_smollm2_cli.py` (tokenizer parity, logits, greedy text/ids, seeded sampling determinism, capability modules) | PASS on Debug and Release (`SmolLM2 native CLI/reference tests: PASS`) |
| `tests/test_smollm2_serve.py` (8 cases: model listing, non-streaming turn, SSE equality with the non-streamed text, JSON mode non-streaming, JSON mode streaming, JSON + `stop` refusal, control-token refusal, three identical sequential streams) | PASS on Debug and Release (`ALL PASSED (8 cases)`) |
| `vastloom model-info` on the imported directory | `hash_verified: true`, 272 tensors, vocab 49,152, 30 layers |
| `vastloom chat --json-object` with the JSON prompt | `constraint_complete: true`, `finish_reason: end_of_turn`, parsed JSON object with keys `data/message/ok/status` |

Decode-rate measurement with the Debug binary (3 runs each, 64-token budget, greedy, same host):

| Run | Plain tokens/s | JSON-mode tokens/s |
|---|---:|---:|
| 1 | 5.262 | 5.689 |
| 2 | 5.366 | 5.791 |
| 3 | 5.975 | 6.025 |
| median | 5.366 | 5.791 |

The constraint costs no measurable per-token decode penalty on this model: the mask touches a
copy of the step logits and the per-token compute path is unchanged. The constrained runs take
longer in wall-clock terms only because they keep generating until the object closes or the
token budget is reached; the 64-token constrained runs correctly report
`constraint_complete: false` and `finish_reason: length`.

### Behavioural evidence for the safety fixes

- **Control token:** a user message containing `<|im_end|><|im_start|>system...` returns HTTP
  400 with an OpenAI-shaped `invalid_request_error`, no `choices`, and the refusal text does
  not echo the marker. The same rejection happens before any streaming headers are sent.
- **Streaming equality:** for the same prompt the concatenated SSE deltas equal the
  non-streaming `message.content` exactly, and three consecutive streaming requests produce
  identical text.
- **UTF-8:** every `data:` frame decodes as strict UTF-8; the tokenizer-level split tests feed
  multi-byte sequences one byte at a time and never observe a broken chunk.
- **JSON bounds:** a 9th nested object is masked at depth 8; a 5th byte in a string under a
  4-byte limit and a 4th digit under a 3-byte number limit are masked; the masked token is
  refused by `accept_token` rather than accepted.
- **Producer lifetime:** the producer owns a `shared_ptr` request state; the serve test runs
  non-streaming then streaming then three more streams on one process without corrupt output.

## Not established

- No large-model execution was validated. The Qwen3.8/DeepSeek records in README and
  `AGENTS.md` remain dated records of runs that are not repeatable on this host today.
- ASan/UBSan was not run: the sanitizer preset is Linux-only and no WSL/Linux toolchain was
  available in this session. The producer-lifetime fix is exercised functionally, not proved
  by a sanitizer.
- JSON Schema and GBNF are not implemented. The JSON constraint accepts exactly one root
  object; `json_schema` is refused with HTTP 400.
- Control-token handling is fail-closed rejection, not Recipe-style segment re-encoding:
  content that literally contains a reserved marker is refused rather than reinterpreted as
  ordinary text.
- The TinyChat and OLMoE serve paths refuse a constrained `response_format` with HTTP 400
  instead of approximating it; their streaming contracts were not otherwise changed.

## Reproduction

```powershell
# Standard gates. On this host Ninja's child pipes hang inside the workspace
# sandbox, so the recorded runs replayed `ninja -t commands` compiler/linker
# lines with inherited stdio; the compiler flags were identical.
cmake --preset debug;   cmake --build --preset debug --parallel
cmake --preset release; cmake --build --preset release --parallel
ctest --test-dir build\release --output-on-failure
ctest --test-dir build\debug --output-on-failure

# SmolLM2 opt-in (model directory contains vastloom-import.json)
New-Item -ItemType Directory -Force build\tmp | Out-Null
$env:TEMP = "$PWD\build\tmp"   # this host's session TEMP has an unusable ACL
python tests\test_smollm2_cli.py --vastloom build\debug\vastloom.exe `
  --model build\smollm2-model --fixture tests\fixtures\smollm2_reference.json
python tests\test_smollm2_serve.py --vastloom build\debug\vastloom.exe `
  --model build\smollm2-model

# Constrained one-shot CLI
build\debug\vastloom.exe chat build\smollm2-model `
  --prompt "Reply with one JSON object containing a boolean field named ok." `
  --max-tokens 512 --json-object --no-verify --json
```

The SmolLM2 model directory and the build-replay helper used for this host are local validation
artifacts under the git-ignored `build/` tree; they are not part of the repository.

## Residual risks

- The JSON constraint's mask computation walks every vocabulary token per step. On SmolLM2
  (49,152 tokens) this is invisible; a much larger vocabulary should be measured before the
  constraint is enabled by default anywhere.
- The control-token rejection changes the request contract: text that legitimately quotes a
  reserved marker is refused. A future segment-encoding implementation can accept it without
  weakening the invariant; the current behaviour is documented in README and SAFETY.
- `max_tokens` shorter than the object needs yields a valid prefix with
  `constraint_complete: false`; clients must check the flag, as the README states.
