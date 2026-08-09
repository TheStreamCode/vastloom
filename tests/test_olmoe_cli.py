#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=30)


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str] | None = None) -> None:
    if condition:
        return
    if process is not None:
        message += f"\nreturncode={process.returncode}\nstdout={process.stdout!r}\nstderr={process.stderr!r}"
    raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stargate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    executable = str(args.stargate.resolve())
    model = str(args.model.resolve())

    process = run([executable, "moe-inspect", model, "--json"])
    require(process.returncode == 0, "verified OLMoE inspection failed", process)
    payload = json.loads(process.stdout)
    require(payload["model_id"] == "allenai/OLMoE-1B-7B-0924-Instruct", "model id mismatch", process)
    require(payload["revision"] == "7f1c97f440f06ce36705e4f2b843edb5925f4498", "revision mismatch", process)
    require(payload["hashes_verified"] is True, "full metadata verification was not reported", process)
    require(payload["metadata_only"] is True, "test fixture unexpectedly includes full weights", process)
    require(payload["tensor_count"] == 3219, "tensor count mismatch", process)
    require(payload["tensor_payload_bytes"] == 13_838_323_712, "tensor payload mismatch", process)
    require(payload["routed_expert_bytes"] == 12_884_901_888, "expert byte classification mismatch", process)
    require(payload["always_active_bytes"] == 953_421_824, "always-active byte classification mismatch", process)
    require(payload["no_cache_expert_bytes_per_token"] == 1_610_612_736, "no-cache traffic mismatch", process)
    require(payload["resident_load_supported"] is False, "metadata-only import was marked runnable", process)
    if payload["safe_ram_cap_bytes"]:
        require(payload["streaming_plan_candidate"] is True, "safe streaming preflight unexpectedly failed", process)

    tokenizer_cases = (
        ("Hello", [12092]),
        ("Quanto fa 2+2?", [3864, 11780, 4195, 374, 12, 19, 32]),
        ("Qual è la capitale d’Italia?", [34255, 12187, 826, 43395, 1079, 277, 457, 1147, 25431, 32]),
        (" digits 123", [24321, 15567]),
    )
    for text, expected in tokenizer_cases:
        tokenized = run([executable, "tokenize", model, "--text", text, "--json"])
        require(tokenized.returncode == 0 and json.loads(tokenized.stdout)["ids"] == expected,
                f"OLMoE tokenizer mismatch for {text!r}", tokenized)
    chat = run([executable, "tokenize", model, "--text", "Hello", "--chat", "--json"])
    require(
        chat.returncode == 0
        and json.loads(chat.stdout)["ids"]
        == [50279, 29, 93, 4537, 49651, 187, 12092, 187, 29, 93, 515, 5567, 49651, 187],
        "OLMoE chat-template token IDs differ from Transformers",
        chat,
    )

    quick = run([executable, "moe-inspect", model, "--no-verify", "--json"])
    require(quick.returncode == 0 and json.loads(quick.stdout)["hashes_verified"] is False,
            "explicit quick inspection failed", quick)
    invalid = run([executable, "moe-inspect", model, "--unknown"])
    require(invalid.returncode != 0, "unknown OLMoE option was accepted", invalid)
    metadata_run = run([executable, "moe-logits", model, "--text", "Hello", "--no-verify", "--json"])
    require(metadata_run.returncode != 0 and "complete verified checkpoint" in metadata_run.stderr,
            "metadata-only checkpoint reached OLMoE execution", metadata_run)
    invalid_cache = run([executable, "moe-logits", model, "--text", "Hello", "--cache", "random", "--json"])
    require(invalid_cache.returncode != 0, "unknown real expert-cache policy was accepted", invalid_cache)

    with tempfile.TemporaryDirectory(prefix="stargate-olmoe-corrupt-") as temporary:
        copy = Path(temporary) / "model"
        shutil.copytree(args.model, copy)
        config = copy / "config.json"
        config.chmod(stat.S_IWRITE | stat.S_IREAD)
        config.write_bytes(config.read_bytes().replace(b'"hidden_size": 2048', b'"hidden_size": 2047'))
        corrupted = run([executable, "moe-inspect", str(copy), "--json"])
        require(corrupted.returncode != 0 and "SHA-256" in corrupted.stderr,
                "metadata corruption was not rejected by full verification", corrupted)

    print("OLMoE pinned metadata/inspection tests: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as error:
        print(f"ERROR: command timed out safely: {error}", file=sys.stderr)
        raise SystemExit(1) from error
