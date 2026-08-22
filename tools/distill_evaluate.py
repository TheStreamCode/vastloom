#!/usr/bin/env python3
"""Measures baseline versus adapted behaviour across a probe set.

Runs every probe through `vastloom chat` twice - once bare and once with an injected
capability module - and grades both answers against the probe's expected answer with a
case- and punctuation-insensitive comparison. Two signals are recorded per side: strict
normalized equality (`*_correct`) and whether the answer merely *starts* with the expected
one (`*_leads`), because a rank-limited head adapter tends to lead with the right token and
then drift. This is E2's measurement instrument, so its numbers come from one reproducible
command instead of hand-typed terminal sessions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

EVALUATE_FORMAT = "vastloom.distill_evaluate.v1"
MAX_EVALUATION_PROBES = 64

_TOOL_PATH = Path(__file__).resolve().parent / "distill_transcript.py"
_spec = importlib.util.spec_from_file_location("_distill_transcript_for_evaluate", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
transcript_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transcript_tool)


class EvaluateError(Exception):
    """Raised for any refused input or failed generation; the caller exits non-zero."""


def normalize_answer(text: str) -> str:
    return transcript_tool.normalize_answer(text)


def load_eval_probes(path: Path) -> list[dict[str, Any]]:
    probes = transcript_tool.load_probes(path)
    if not probes or len(probes) > MAX_EVALUATION_PROBES:
        raise EvaluateError(f"probe count must be between 1 and {MAX_EVALUATION_PROBES}")
    missing = [probe["id"] for probe in probes if "expected" not in probe]
    if missing:
        raise EvaluateError(f"probes without an expected answer cannot be graded: {missing}")
    return probes


def run_case(runner: Callable[[list[str]], Any], executable: Path, model: Path,
             prompt: str, max_tokens: int, module: Path | None) -> str:
    command = [str(executable), "chat", str(model), "--prompt", prompt,
               "--max-tokens", str(max_tokens), "--temperature", "0",
               "--no-verify", "--json"]
    if module is not None:
        command += ["--capability-module", str(module)]
    process = runner(command)
    if getattr(process, "returncode", 1) != 0:
        tail = (getattr(process, "stderr", "") or "")[-300:]
        raise EvaluateError(f"chat failed: {tail}")
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise EvaluateError(f"chat output is not JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("ok") is not True \
            or not isinstance(payload.get("text"), str):
        raise EvaluateError("chat did not report ok")
    return payload["text"]


def evaluate(probes: list[dict[str, Any]], runner: Callable[[list[str]], Any],
             executable: Path, model: Path,
             module: Path | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for probe in probes:
        expected = probe["expected"]
        base_text = run_case(runner, executable, model, probe["prompt"],
                             probe["max_tokens"], module=None)
        record: dict[str, Any] = {
            "id": probe["id"],
            "split": probe["split"],
            "expected": expected,
            "base_text": base_text,
        }
        if normalize_answer(base_text) == normalize_answer(expected):
            record["base_correct"] = True
        elif normalize_answer(base_text).startswith(normalize_answer(expected)):
            record["base_correct"] = False
            record["base_leads"] = True
        else:
            record["base_correct"] = False
        if module is not None:
            adapted_text = run_case(runner, executable, model, probe["prompt"],
                                    probe["max_tokens"], module=module)
            record["adapted_text"] = adapted_text
            if normalize_answer(adapted_text) == normalize_answer(expected):
                record["adapted_correct"] = True
            elif normalize_answer(adapted_text).startswith(normalize_answer(expected)):
                record["adapted_correct"] = False
                record["adapted_leads"] = True
            else:
                record["adapted_correct"] = False
        records.append(record)
    return records


def write_summary(path: Path, records: list[dict[str, Any]], module_name: str | None) -> None:
    envelope = {"format": EVALUATE_FORMAT,
                "module": module_name,
                "records": records}
    encoded = (json.dumps(envelope, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def main(argv: list[str] | None = None,
         runner: Callable[[list[str]], Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--module", type=Path,
                        help="inject this capability module as the adapted arm")
    arguments = parser.parse_args(argv)

    if runner is None:
        def runner(command: list[str]) -> Any:
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            return subprocess.run(command, capture_output=True, text=True,
                                  encoding="utf-8", errors="strict",
                                  env=environment, timeout=1_800, check=False)

    try:
        probes = load_eval_probes(arguments.probes)
        records = evaluate(probes, runner, arguments.vastloom, arguments.model,
                           arguments.module)
        write_summary(arguments.out, records,
                      arguments.module.name if arguments.module is not None else None)
    except (EvaluateError, transcript_tool.TranscriptError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    train = [r for r in records if r["split"] == "train"]
    held = [r for r in records if r["split"] == "eval"]
    def rate(rows: list[dict[str, Any]], key: str) -> str:
        graded = [r for r in rows if key in r]
        hits = sum(1 for r in graded if r[key])
        return f"{hits}/{len(graded)}"
    print(f"baseline correct:      train {rate(train, 'base_correct')}  "
          f"eval {rate(held, 'base_correct')}")
    if arguments.module is not None:
        print(f"adapted correct:       train {rate(train, 'adapted_correct')}  "
              f"eval {rate(held, 'adapted_correct')}")
        print(f"adapted leads-with:    train {rate(train, 'adapted_leads')}  "
              f"eval {rate(held, 'adapted_leads')}")
    print(f"Wrote {len(records)} evaluation records to {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
