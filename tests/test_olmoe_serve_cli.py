#!/usr/bin/env python3
"""Opt-in smoke test for `vastloom serve` on the OLMoE checkpoint.

Spawns the OpenAI-compatible endpoint on a fixed loopback port, waits for
readiness by polling `/v1/models`, runs two short greedy turns, verifies the
greedy-only temperature refusal and the SSE stream terminator, then stops the
server. Enabled by configuring CMake with
`-DVASTLOOM_OLMOE_FULL_MODEL=<complete pinned OLMoE checkpoint>`.

Follows the M13 SmolLM2 serve-test shape: fixed port, HTTP readiness poll,
`--no-verify` (hash verification is covered by the runtime-reference test).
Turn two resends the turn-one history plus a tail and must report the shared
prefix as cache-served — the Q12 gate in endpoint form.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from host_pressure import skipped


CONTRACT = "OLMoE serve endpoint contract"
SERVE_PORT = 18124
READINESS_TIMEOUT = 120
REQUEST_TIMEOUT = 600


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wait_ready(
    base: str,
    timeout: int = READINESS_TIMEOUT,
    process: subprocess.Popen[str] | None = None,
) -> bool:
    """Poll the models endpoint until the server answers or time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def http_json(method: str, url: str, payload: dict[str, object] | None = None) -> tuple[int, str]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    executable = str(args.vastloom.resolve())
    model = str(args.model.resolve())
    model_id = args.model.resolve().name
    base = f"http://127.0.0.1:{SERVE_PORT}"

    with tempfile.TemporaryDirectory(prefix="vastloom-olmoe-serve-") as temporary:
        stderr_path = Path(temporary) / "serve-stderr.log"
        with stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                [executable, "serve", model, "--port", str(SERVE_PORT), "--no-verify"],
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                text=True,
            )
            try:
                if not wait_ready(base, READINESS_TIMEOUT, process):
                    code = process.poll()
                    tail = stderr_path.read_text(encoding="utf-8")[-2000:]
                    if code == 75 and any(
                        marker in tail
                        for marker in (
                            "of physical RAM",
                            "of RAM headroom",
                            "storage reserve",
                            "safe RAM headroom for the model",
                        )
                    ):
                        return skipped(CONTRACT, tail.strip())
                    raise AssertionError(
                        f"serve did not become ready within {READINESS_TIMEOUT} s "
                        f"(returncode={code})\nstderr tail: {tail!r}"
                    )

                status, body = http_json("GET", f"{base}/v1/models")
                require(status == 200, f"GET /v1/models returned {status}: {body!r}")
                listing = json.loads(body)
                require(
                    listing["data"][0]["id"] == model_id,
                    f"model listing mismatch: {body!r}",
                )

                turn_one = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 2,
                }
                status, body = http_json("POST", f"{base}/v1/chat/completions", turn_one)
                require(status == 200, f"turn one returned {status}: {body!r}")
                first = json.loads(body)
                require(first["object"] == "chat.completion", f"bad object: {body!r}")
                require(
                    isinstance(first["choices"][0]["message"]["content"], str),
                    f"missing content: {body!r}",
                )
                require(
                    first.get("vastloom_cached_prefix_tokens") == 0,
                    f"cold turn must report zero cached tokens: {body!r}",
                )
                require(first["usage"]["completion_tokens"] <= 2, f"token cap ignored: {body!r}")

                history = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": first["choices"][0]["message"]["content"]},
                    {"role": "user", "content": "And goodbye?"},
                ]
                turn_two = {"model": model_id, "messages": history, "max_tokens": 2}
                status, body = http_json("POST", f"{base}/v1/chat/completions", turn_two)
                require(status == 200, f"turn two returned {status}: {body!r}")
                second = json.loads(body)
                require(
                    isinstance(second.get("vastloom_cached_prefix_tokens"), int)
                    and second["vastloom_cached_prefix_tokens"] > 0,
                    f"turn two must report prefix reuse: {body!r}",
                )
                require(
                    len(second["choices"][0]["message"]["content"]) > 0,
                    f"turn two produced no text: {body!r}",
                )

                sampled = dict(turn_one)
                sampled["temperature"] = 0.7
                status, body = http_json("POST", f"{base}/v1/chat/completions", sampled)
                require(status == 400, f"temperature refusal returned {status}: {body!r}")
                require("greedy" in body.lower(), f"refusal must name greedy decoding: {body!r}")

                streamed = dict(turn_one)
                streamed["stream"] = True
                status, body = http_json("POST", f"{base}/v1/chat/completions", streamed)
                require(status == 200, f"streaming turn returned {status}: {body!r}")
                require("data: " in body, f"streaming turn has no SSE frames: {body!r}")
                require(
                    body.rstrip().endswith("data: [DONE]"),
                    f"stream missing terminator: {body!r}",
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
    print("OLMoE serve smoke: ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
