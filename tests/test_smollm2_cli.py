#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from host_pressure import guard, run_contract


CONTRACT = "SmolLM2 dense reference contract"


def run(command: list[str], *, input_text: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return guard(subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
        timeout=timeout,
        check=False,
    ))


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str] | None = None) -> None:
    if condition:
        return
    if process is not None:
        message += f"\nreturncode={process.returncode}\nstdout={process.stdout!r}\nstderr={process.stderr!r}"
    raise AssertionError(message)


def json_command(command: list[str], *, success: bool = True) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    process = run(command)
    require((process.returncode == 0) == success, "unexpected command result", process)
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(f"invalid JSON output: {error}\n{process.stdout!r}") from error
    return payload, process


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    executable = str(args.vastloom.resolve())
    model = str(args.model.resolve())
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))

    info, process = json_command([executable, "model-info", model, "--json"], success=True)
    require(info["model_id"] == fixture["model_id"], "model id mismatch", process)
    require(info["revision"] == fixture["revision"], "revision mismatch", process)
    require(info["hash_verified"] is True and info["tensor_count"] == 272, "full verification was not reported", process)

    for case in fixture["tokenizer_cases"]:
        payload, process = json_command([executable, "tokenize", model, "--text", case["text"], "--json"])
        require(payload["ids"] == case["ids"], f"tokenizer parity failed for {case['text']!r}", process)
        require(payload["decoded"] == case["text"], f"tokenizer round trip failed for {case['text']!r}", process)

    for case in fixture["logit_cases"]:
        command = [executable, "real-logits", model, "--text", case["text"], "--top", str(len(case["top"])), "--no-verify", "--json"]
        if case["chat"]:
            command.append("--chat")
        payload, process = json_command(command)
        actual = payload["top_logits"]
        require([item["id"] for item in actual] == [item[0] for item in case["top"]], "top-logit token order mismatch", process)
        for item, expected in zip(actual, case["top"], strict=True):
            require(abs(item["logit"] - expected[1]) <= 1e-3, f"logit mismatch for token {expected[0]}", process)

    for case in fixture["generation_cases"]:
        payload, process = json_command([
            executable, "chat", model, "--prompt", case["prompt"], "--max-tokens", str(case["max_tokens"]),
            "--no-verify", "--json"
        ])
        require(payload["text"] == case["text"], f"greedy text mismatch for {case['prompt']!r}", process)
        require(payload["generated_ids"] == case["ids"], f"greedy token mismatch for {case['prompt']!r}", process)

    sampled_command = [
        executable, "chat", model, "--prompt", "Give me one word.", "--max-tokens", "8", "--temperature", "0.7",
        "--top-k", "20", "--seed", "42", "--no-verify", "--json"
    ]
    sampled_one, _ = json_command(sampled_command)
    sampled_two, _ = json_command(sampled_command)
    require(sampled_one["generated_ids"] == sampled_two["generated_ids"], "seeded sampling is not deterministic")

    conversation = "\n".join(fixture["multi_turn"]["inputs"] + ["/stats", "/model", "/exit", ""])
    interactive = run([
        executable, "chat", model, "--max-tokens", "64", "--no-verify", "--no-stream", "--no-stats"
    ], input_text=conversation)
    require(interactive.returncode == 0, "interactive chat failed", interactive)
    for expected in fixture["multi_turn"]["outputs"]:
        require(expected in interactive.stdout, f"interactive response missing: {expected!r}", interactive)
    require("Prompt tokens:" in interactive.stdout and "HuggingFaceTB/SmolLM2" in interactive.stdout,
            "interactive /stats or /model failed", interactive)

    for invalid in (
        [executable, "chat", model, "--prompt"],
        [executable, "chat", model, "--prompt", "x", "--prompt", "y"],
        [executable, "chat", model, "--prompt", "x", "--unknown"],
    ):
        process = run(invalid)
        require(process.returncode != 0, "invalid real-model option was accepted", process)

    oversized = run([
        executable, "chat", model, "--prompt", "x " * 9000, "--max-tokens", "1", "--no-verify", "--json"
    ])
    require(oversized.returncode != 0 and "context" in oversized.stderr.lower(), "oversized prompt was not rejected safely", oversized)

    print("SmolLM2 native CLI/reference tests: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_contract(CONTRACT, main))
    except subprocess.TimeoutExpired as error:
        print(f"ERROR: command timed out safely: {error}", file=sys.stderr)
        raise SystemExit(1) from error
