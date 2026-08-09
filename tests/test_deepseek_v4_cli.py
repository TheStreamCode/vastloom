#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(command: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str] | None = None) -> None:
    if condition:
        return
    details = ""
    if process is not None:
        details = f"\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    raise AssertionError(message + details)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    command = [args.vastloom, "deepseek-inspect", args.model, "--json"]
    inspected = run(command)
    require(inspected.returncode == 0, "verified DeepSeek metadata/header inspection failed", inspected)
    payload = json.loads(inspected.stdout)
    require(payload["model_id"] == "deepseek-ai/DeepSeek-V4-Flash", "DeepSeek model id mismatch", inspected)
    require(
        payload["revision"] == "60d8d70770c6776ff598c94bb586a859a38244f1",
        "DeepSeek revision mismatch",
        inspected,
    )
    require(payload["metadata_only"] is True, "DeepSeek metadata import was marked complete", inspected)
    require(payload["hashes_verified"] is True, "DeepSeek metadata hashes were not verified", inspected)
    require(payload["header_catalog_complete"] is True, "DeepSeek header catalog is incomplete", inspected)
    require(payload["execution_supported"] is False, "DeepSeek execution was claimed prematurely", inspected)
    require(payload["shard_count"] == 46 and payload["tensor_count"] == 69_187, "DeepSeek tensor identity changed", inspected)
    require(payload["tensor_payload_bytes"] == 159_609_485_896, "DeepSeek payload bytes changed", inspected)
    require(payload["routed_expert_bytes"] == 150_592_290_816, "DeepSeek expert bytes changed", inspected)
    require(payload["always_active_bytes"] == 9_017_195_080, "DeepSeek trunk bytes changed", inspected)
    require(payload["bytes_per_expert"] == 13_369_344, "DeepSeek per-expert bytes changed", inspected)
    require(
        payload["active_expert_bytes_per_layer"] == 80_216_064,
        "DeepSeek active expert bytes changed",
        inspected,
    )
    dtypes = {item["dtype"]: (item["tensor_count"], item["bytes"]) for item in payload["dtypes"]}
    require(
        dtypes
        == {
            "BF16": (433, 2_830_518_528),
            "F32": (417, 144_672_072),
            "F8_E4M3": (375, 6_023_020_544),
            "F8_E8M0": (34_167, 8_858_737_664),
            "I8": (33_792, 141_733_920_768),
            "I64": (3, 18_616_320),
        },
        "DeepSeek dtype catalog changed",
        inspected,
    )

    quick = run(command[:-1] + ["--no-verify", "--json"])
    require(quick.returncode == 0, "DeepSeek quick inspection failed", quick)
    require(json.loads(quick.stdout)["hashes_verified"] is False, "quick inspection claimed full hashes", quick)

    unknown = run([args.vastloom, "deepseek-inspect", args.model, "--unknown"])
    require(unknown.returncode != 0, "unknown DeepSeek option was accepted", unknown)

    tokenizer_vectors = [
        ("Hello, world!", [19923, 14, 2058, 3]),
        (
            "Quanto fa 2+2? È facile: 1234567.",
            [4100, 10829, 5035, 223, 20, 13, 20, 33, 66978, 58155, 28, 223, 6895, 18009, 25, 16],
        ),
        ("Caffè e caffe\u0301 — già!", [37, 3871, 2666, 312, 274, 30087, 17793, 2136, 62199, 3]),
        ("你好世界。日本語テスト", [30594, 3427, 320, 88768, 109288]),
        (
            "emoji 🚀✨ symbols ++=\nnext\n\n  ",
            [18872, 7063, 7351, 251, 225, 57398, 19435, 16571, 24925, 6695, 271, 262],
        ),
    ]
    for text, expected_ids in tokenizer_vectors:
        tokenized = run([args.vastloom, "tokenize", args.model, "--text", text, "--json"])
        require(tokenized.returncode == 0, "native DeepSeek tokenization failed", tokenized)
        token_payload = json.loads(tokenized.stdout)
        require(
            token_payload["family"] == "deepseek_v4_byte_bpe"
            and token_payload["ids"] == expected_ids
            and token_payload["decoded"] == text,
            "native DeepSeek tokens differ from Hugging Face tokenizers 0.22.1",
            tokenized,
        )

    chat = run([args.vastloom, "tokenize", args.model, "--text", "Ciao!", "--chat", "--json"])
    require(chat.returncode == 0, "native DeepSeek chat formatting failed", chat)
    chat_payload = json.loads(chat.stdout)
    require(
        chat_payload["ids"] == [0, 128803, 37, 23144, 3, 128804, 128822]
        and chat_payload["decoded"]
        == "<｜begin▁of▁sentence｜><｜User｜>Ciao!<｜Assistant｜></think>",
        "native DeepSeek chat template differs from the official encoder",
        chat,
    )
    system_chat = run(
        [
            args.vastloom,
            "tokenize",
            args.model,
            "--text",
            "Ciao!",
            "--chat",
            "--system",
            "Sei utile.",
            "--json",
        ]
    )
    require(system_chat.returncode == 0, "native DeepSeek system prompt formatting failed", system_chat)
    require(
        json.loads(system_chat.stdout)["ids"]
        == [0, 4374, 75, 96126, 16, 128803, 37, 23144, 3, 128804, 128822],
        "native DeepSeek system prompt differs from the official encoder",
        system_chat,
    )
    bad_system = run([args.vastloom, "tokenize", args.model, "--text", "x", "--system", "y", "--json"])
    require(bad_system.returncode != 0, "tokenize accepted --system without --chat", bad_system)

    with tempfile.TemporaryDirectory(prefix="vastloom-deepseek-corrupt-") as temporary:
        copy = Path(temporary) / "model"
        shutil.copytree(args.model, copy)
        config_path = copy / "config.json"
        original = config_path.read_bytes()
        corrupted = original.replace(b'"num_hidden_layers": 43', b'"num_hidden_layers": 42')
        require(len(corrupted) == len(original) and corrupted != original, "test could not mutate pinned config")
        config_path.chmod(0o600)
        config_path.write_bytes(corrupted)
        verified_corruption = run([args.vastloom, "deepseek-inspect", str(copy), "--json"])
        require(verified_corruption.returncode != 0, "DeepSeek config corruption passed SHA-256", verified_corruption)
        structural_corruption = run(
            [args.vastloom, "deepseek-inspect", str(copy), "--no-verify", "--json"]
        )
        require(structural_corruption.returncode != 0, "DeepSeek config drift passed structural validation", structural_corruption)

        config_path.write_bytes(original)
        tokenizer_path = copy / "tokenizer.json"
        tokenizer = tokenizer_path.read_bytes()
        malformed_tokenizer = tokenizer.replace(b"\\\\p{N}{1,3}", b"\\\\p{N}{1,4}", 1)
        require(
            len(malformed_tokenizer) == len(tokenizer) and malformed_tokenizer != tokenizer,
            "test could not mutate the tokenizer split regex",
        )
        tokenizer_path.chmod(0o600)
        tokenizer_path.write_bytes(malformed_tokenizer)
        rejected_tokenizer = run(
            [args.vastloom, "tokenize", str(copy), "--text", "1234", "--json"]
        )
        require(
            rejected_tokenizer.returncode != 0 and "split regex differs" in rejected_tokenizer.stderr,
            "DeepSeek tokenizer regex drift was accepted",
            rejected_tokenizer,
        )

    print("DeepSeek V4 Flash metadata/header/tokenizer reference tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
