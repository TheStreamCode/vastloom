# M6 DeepSeek V4 Flash tokenizer validation

Validation date: 2026-08-09

Implementation commit: `c815fdb`

## Scope

This report validates the native tokenizer and basic chat-framing slice for the official
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
repository at immutable commit
`60d8d70770c6776ff598c94bb586a859a38244f1`:

```text
pinned tokenizer.json (SHA-256 verified)
  -> exact Sequence/Split/ByteLevel layout validation
  -> native Unicode pre-tokenization
  -> byte-level BPE + all added tokens
  -> official basic chat framing
  -> token IDs checked against Hugging Face tokenizers 0.22.1
```

This is not DeepSeek model inference. It proves prompt bytes can reach the future graph
with the expected token IDs; no weight tensor, activation, KV state, routed expert or GPU
allocation is involved.

## Pinned tokenizer identity

| Field | Verified value |
|---|---:|
| `tokenizer.json` bytes | 6,367,146 |
| SHA-256 | `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf` |
| Model | byte-level BPE |
| Base BPE vocabulary | 128,000 |
| Final vocabulary | 129,280 |
| BPE merges | 127,741 |
| Added-token definitions | 1,283 |
| BOS / EOS | 0 / 1 |
| Maximum declared context | 1,048,576 |

The loader accepts the DeepSeek layout only when the empty Sequence normalizer, all
three Split regular expressions, their `Isolated` behavior and the final ByteLevel stage
match the pinned contract. A same-length mutation from `\p{N}{1,3}` to `\p{N}{1,4}` is
rejected before tokenization.

## Native implementation

The native path implements the official ordered split stages:

1. Unicode numbers isolated in groups of one to three;
2. Chinese, Hiragana and Katakana runs isolated;
3. the official Unicode letter/mark, punctuation/symbol, newline and whitespace rules;
4. GPT-style byte-to-Unicode mapping and BPE merges.

Unicode general categories are generated into a checked-in interval table from Python
`unicodedata` 15.0.0. This avoids host-locale and Windows/Linux classification drift.
The table is reproducible with `tools/generate_unicode_categories.py`.

All added-token definitions are parsed and validated, including non-special chat tokens.
A sparse byte trie performs longest literal matching, avoiding an unbounded scan of all
1,283 definitions for every input byte. The existing 16 MiB tokenizer-input ceiling and
bounded JSON limits remain in force.

Basic chat mode now matches the official encoder for system, user and assistant turns:

```text
<｜begin▁of▁sentence｜>{optional system}<｜User｜>{prompt}<｜Assistant｜></think>
```

Role ordering, duplicate/misplaced system messages and `--system` without `--chat` fail
closed. Full thinking-mode history manipulation, DSML tool calls, quick-instruction tasks
and completion parsing are not part of this slice.

## Reference vectors

The real pinned tokenizer produced exact native/reference agreement for:

| Input class | Reference result |
|---|---:|
| English punctuation | 4 tokens |
| Italian accents and grouped digits | 16 tokens |
| composed and combining accents | 10 tokens |
| Chinese and Japanese | 5 tokens |
| emoji, symbols, newlines and trailing spaces | 12 tokens |
| Italian basic chat prompt | 7 tokens |
| Italian system + user prompt | 11 tokens |

The exact ID arrays are committed in `tests/test_deepseek_v4_cli.py`. An additional
deterministic 200-case mixed Unicode probe (3,830 characters, 2,503 output tokens) agreed
exactly with Hugging Face tokenizers 0.22.1. That probe covered ASCII, grouped digits,
Latin accents, combining marks, Greek, Cyrillic, Chinese, Japanese, emoji, symbols and
mixed whitespace. It is additional validation, not an exhaustive proof for every Unicode
code point.

An always-on reduced tokenizer test constructs a small compatible byte-BPE fixture. It
checks one-to-three digit isolation, punctuation-plus-ASCII-word grouping, added-token
decoding, chat framing and malformed-regex rejection without requiring downloaded model
metadata.

## Build and test matrix

| Toolchain/configuration | Result |
|---|---:|
| MSVC 19.44 Release, warnings as errors, real metadata | 7/7 passed in 5.63 s |
| MSVC 19.44 Debug, warnings as errors, real metadata | 7/7 passed in 28.67 s |
| GCC 13.3 Release, warnings as errors, real metadata | 7/7 passed in 9.64 s |
| Clang 18.1 Release, warnings as errors, real metadata | 7/7 passed in 9.26 s |
| GCC Debug + AddressSanitizer + UndefinedBehaviorSanitizer | 7/7 passed in 51.92 s |

ASan used leak detection and halt-on-error; UBSan was configured to halt on the first
finding. Python source compilation and generated Unicode-table byte comparison also
passed.

## Performance and safety

No tokens/s or TTFT number is reported because no model forward pass occurred. Loading
and tokenizing the real metadata uses only the 6.1 MiB tokenizer artifact and bounded
native structures. It does not trigger the large-download interlock because it does not
read any of the 159.6 GB weight payload.

## Known limitations and next bottleneck

- no reduced architecture-faithful DeepSeek graph fixture;
- no reference logits, expert routes or generated tokens;
- no full reasoning/tool-call template surface;
- no DeepSeek attention, Hyper-Connection, router, expert or MTP execution;
- no packed FP4/FP8 compute path;
- no full checkpoint download, CUDA or multi-tier DeepSeek execution.

The next bottleneck is the reduced float reference graph and its deterministic logits and
route vectors. The full checkpoint must remain undownloaded until that gate passes and a
bounded execution path exists.
