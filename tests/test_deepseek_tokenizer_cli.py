#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str] | None = None) -> None:
    if condition:
        return
    detail = "" if process is None else f"\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    raise AssertionError(message + detail)


def bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(0xA1, 0xAC + 1))
    byte_values += list(range(0xAE, 0xFF + 1))
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(byte_values, unicode_values, strict=True)}


def byte_symbols(text: str, mapping: dict[int, str]) -> list[str]:
    return [mapping[value] for value in text.encode("utf-8")]


def make_tokenizer(root: Path) -> tuple[dict[str, int], int, int, int]:
    mapping = bytes_to_unicode()
    vocab: dict[str, int] = {
        "<｜begin▁of▁sentence｜>": 0,
        "<｜end▁of▁sentence｜>": 1,
        "<｜▁pad▁｜>": 2,
    }
    for value in range(256):
        vocab[mapping[value]] = len(vocab)

    merges: list[str] = []
    merge_set: set[str] = set()

    def add_merged(text: str) -> int:
        symbols = byte_symbols(text, mapping)
        prefix = symbols[0]
        for symbol in symbols[1:]:
            merged = prefix + symbol
            merge = prefix + " " + symbol
            if merge not in merge_set:
                merges.append(merge)
                merge_set.add(merge)
            if merged not in vocab:
                vocab[merged] = len(vocab)
            prefix = merged
        return vocab[prefix]

    token_123 = add_merged("123")
    token_1234 = add_merged("1234")
    token_bang_hello = add_merged("!Hello")

    user_id = len(vocab)
    assistant_id = user_id + 1
    thinking_end_id = user_id + 2
    added_tokens = [
        (0, "<｜begin▁of▁sentence｜>", True),
        (1, "<｜end▁of▁sentence｜>", True),
        (2, "<｜▁pad▁｜>", True),
        (user_id, "<｜User｜>", False),
        (assistant_id, "<｜Assistant｜>", False),
        (thinking_end_id, "</think>", False),
    ]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": special,
            }
            for token_id, content, special in added_tokens
        ],
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": r"\p{N}{1,3}"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "Split",
                    "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": False, "vocab": vocab, "merges": merges},
    }
    config = {
        "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
        "eos_token": {"content": "<｜end▁of▁sentence｜>"},
        "pad_token": {"content": "<｜end▁of▁sentence｜>"},
    }
    (root / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return vocab, token_123, token_1234, token_bang_hello


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vastloom-deepseek-tokenizer-") as temporary:
        root = Path(temporary)
        vocab, token_123, token_1234, token_bang_hello = make_tokenizer(root)

        numeric = run([args.vastloom, "tokenize", str(root), "--text", "1234", "--json"])
        require(numeric.returncode == 0, "reduced DeepSeek numeric tokenization failed", numeric)
        numeric_payload = json.loads(numeric.stdout)
        require(
            numeric_payload["family"] == "deepseek_v4_byte_bpe"
            and numeric_payload["ids"] == [token_123, vocab["4"]]
            and token_1234 not in numeric_payload["ids"],
            "DeepSeek 1-3 digit isolation was not preserved",
            numeric,
        )

        punctuation = run([args.vastloom, "tokenize", str(root), "--text", "!Hello", "--json"])
        require(punctuation.returncode == 0, "reduced DeepSeek punctuation tokenization failed", punctuation)
        require(
            json.loads(punctuation.stdout)["ids"] == [token_bang_hello],
            "DeepSeek punctuation-plus-ASCII-word isolation was not preserved",
            punctuation,
        )

        chat = run([args.vastloom, "tokenize", str(root), "--text", "A", "--chat", "--json"])
        require(chat.returncode == 0, "reduced DeepSeek chat template failed", chat)
        payload = json.loads(chat.stdout)
        require(payload["decoded"] == "<｜begin▁of▁sentence｜><｜User｜>A<｜Assistant｜></think>",
                "reduced DeepSeek chat framing changed", chat)

        tokenizer_path = root / "tokenizer.json"
        malformed = tokenizer_path.read_text(encoding="utf-8").replace(r"\p{N}{1,3}", r"\p{N}{1,4}", 1)
        tokenizer_path.write_text(malformed, encoding="utf-8")
        rejected = run([args.vastloom, "tokenize", str(root), "--text", "1234", "--json"])
        require(rejected.returncode != 0 and "split regex differs" in rejected.stderr,
                "malformed DeepSeek tokenizer layout was accepted", rejected)

    print("DeepSeek reduced tokenizer/chat-template tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
