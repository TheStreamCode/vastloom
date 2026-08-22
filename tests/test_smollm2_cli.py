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

    # E1 capability-module mechanics on the frozen base: a stored-but-inert module must not move
    # a single greedy token, an active module must be routed, bound and cost-accounted, and the
    # whole thing must be deterministic.
    import tempfile

    base_case = fixture["generation_cases"][0]
    chat_prefix = [
        executable, "chat", model, "--prompt", base_case["prompt"],
        "--max-tokens", str(base_case["max_tokens"]), "--no-verify", "--json",
    ]
    baseline, _ = json_command(chat_prefix)

    with tempfile.TemporaryDirectory() as scratch:
        zero_module = Path(scratch) / "parity-zero.vlcm"
        payload, process = json_command([
            executable, "module-init", model, str(zero_module),
            "--name", "parity-zero", "--rank", "4", "--scale", "0.5", "--seed", "7",
            "--zero-a", "--json",
        ])
        require(payload["ok"] is True and payload["targets"] == 150, "module-init did not report its targets", process)

        inert, _ = json_command(chat_prefix + ["--capability-module", str(zero_module)])
        require(inert["text"] == baseline["text"], "inert module changed the generated text", process)
        require(inert["generated_ids"] == baseline["generated_ids"], "inert module changed generated tokens", process)
        capability = inert.get("capability")
        require(capability is not None, "capability accounting missing from JSON output", process)
        require(capability["selected"] == "parity-zero", "router did not select the only module", process)
        require(capability["added_macs"] > 0, "inert module reported no injection cost", process)
        require(capability["bindings"] > 0, "capability binding count missing", process)

        live_module = Path(scratch) / "live-skill.vlcm"
        payload, process = json_command([
            executable, "module-init", model, str(live_module),
            "--name", "live-skill", "--rank", "8", "--scale", "0.05", "--seed", "9", "--json",
        ])
        require(payload["ok"] is True, "live module was not written", process)
        active_one, _ = json_command(chat_prefix + ["--capability-module", str(live_module)])
        active_two, _ = json_command(chat_prefix + ["--capability-module", str(live_module)])
        require(active_one["capability"]["selected"] == "live-skill", "active module was not selected", process)
        require(active_one["generated_ids"] == active_two["generated_ids"],
                "active module generation is not deterministic", process)
        require(active_one["capability"]["added_macs"] > 0, "active module reported no cost", process)

        both_one, _ = json_command(
            chat_prefix + ["--capability-module", str(zero_module), "--capability-module", str(live_module)])
        both_two, _ = json_command(
            chat_prefix + ["--capability-module", str(zero_module), "--capability-module", str(live_module)])
        require(both_one["capability"]["considered"] == 2, "router did not see both modules", process)
        require(both_one["capability"]["selected"] == both_two["capability"]["selected"],
                "routing is not stable across identical requests", process)

        missing = run(chat_prefix + ["--capability-module", str(Path(scratch) / "absent.vlcm")])
        require(missing.returncode != 0, "a missing capability module was accepted", missing)

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

    # M13: stateful serve with prefix-KV reuse. Turn one fills the cache; turn two resends
    # the same history plus a tail and must report the shared prefix as cache-served.
    import tempfile
    import time
    import urllib.request

    def http_post(port, body):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    serve_port = 18123
    server_proc = subprocess.Popen(
        [executable, "serve", model, "--port", str(serve_port), "--no-verify"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ready = False
        deadline = time.time() + 60
        while time.time() < deadline and not ready:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{serve_port}/v1/models", timeout=2)
                ready = True
            except Exception:
                time.sleep(0.5)
        require(ready, "serve did not become ready within 60 s")

        turn1 = http_post(serve_port, {
            "model": "smollm2", "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say hello."}]})
        require(len(turn1["choices"][0]["message"]["content"]) > 0, "turn 1 produced no text")
        require(turn1["vastloom_cached_prefix_tokens"] == 0, "turn 1 must report zero cached tokens")

        history = [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": turn1["choices"][0]["message"]["content"]},
            {"role": "user", "content": "And goodbye."},
        ]
        turn2 = http_post(serve_port, {
            "model": "smollm2", "max_tokens": 16, "messages": history})
        require(turn2["vastloom_cached_prefix_tokens"] > 0,
                "turn 2 must report prefix reuse")
        require(len(turn2["choices"][0]["message"]["content"]) > 0, "turn 2 produced no text")
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=10)

    print("SmolLM2 native CLI/reference tests: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_contract(CONTRACT, main))
    except subprocess.TimeoutExpired as error:
        print(f"ERROR: command timed out safely: {error}", file=sys.stderr)
        raise SystemExit(1) from error
