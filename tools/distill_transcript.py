#!/usr/bin/env python3
"""Builds the E2 teacher-transcript artifact for capability distillation.

Runs the pinned DeepSeek teacher through `vastloom deepseek-forward` over a bounded probe
set and records what the teacher answered, token by token, into one JSON artifact. The
targets are deliberately vocabulary-free - prompt to completion text plus completion ids -
because the teacher and the student do not share a tokenizer, so logit-space matching is
not available to the compiler.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

TRANSCRIPT_FORMAT = "vastloom.distill_transcript.v1"
MAX_PROBES = 64
MAX_PROMPT_CHARS = 4_000
MAX_COMPLETION_TOKENS = 512
MAX_TRANSCRIPT_BYTES = 8 * 1024 * 1024


class TranscriptError(Exception):
    """Raised for any refused input or failed teacher run; the caller exits non-zero."""


def load_probes(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TranscriptError(f"unreadable probe set {path}: {error}") from error
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_PROBES:
        raise TranscriptError(f"probe set must be a list of 1..{MAX_PROBES} probes")
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise TranscriptError("every probe must be an object")
        identifier = item.get("id")
        prompt = item.get("prompt")
        max_tokens = item.get("max_tokens", 16)
        expected = item.get("expected")
        split = item.get("split", "train")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise TranscriptError("probe ids must be non-empty and unique")
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= MAX_PROMPT_CHARS:
            raise TranscriptError(
                f"probe {identifier}: prompt must be 1..{MAX_PROMPT_CHARS} characters")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_COMPLETION_TOKENS:
            raise TranscriptError(
                f"probe {identifier}: max_tokens must be 1..{MAX_COMPLETION_TOKENS}")
        if expected is not None and (not isinstance(expected, str) or not expected.strip()
                                     or len(expected) > 200):
            raise TranscriptError(
                f"probe {identifier}: expected must be a non-empty string up to 200 chars")
        if split not in ("train", "eval"):
            raise TranscriptError(f"probe {identifier}: split must be 'train' or 'eval'")
        seen.add(identifier)
        probe: dict[str, Any] = {"id": identifier, "prompt": prompt,
                                 "max_tokens": max_tokens, "split": split}
        if expected is not None:
            probe["expected"] = expected
        probes.append(probe)
    return probes


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    kept = "".join(character if character.isalnum() or character.isspace() else " "
                   for character in lowered)
    return " ".join(kept.split())


def build_command(vastloom: Path, checkpoint: Path, tokenizer: Path,
                  probe: dict[str, Any], chat: bool, layers: int,
                  trunk_window: int) -> list[str]:
    # The validated execution checkpoint from AGENTS.md: full-depth greedy generate on
    # UD-IQ2_XXS is only claimed with the CPU path, the ggml backend and the 43-matrix trunk
    # window. A shallower forward would silently produce weaker targets.
    command = [
        str(vastloom), "deepseek-forward", str(checkpoint),
        "--prompt", probe["prompt"],
        "--tokenizer", str(tokenizer),
        "--max-tokens", str(probe["max_tokens"]),
        "--layers", str(layers),
        "--trunk-window", str(trunk_window),
        "--cpu-only",
        "--ggml-compute",
        "--json",
    ]
    if chat:
        command.append("--chat")
    return command


def parse_forward_output(stdout: str) -> tuple[str, list[int], str]:
    # deepseek-forward prints two JSON objects on success: the preflight report first, the
    # generation result second. Decode them one at a time and keep the one that carries the
    # generated tokens, so extra lines - now or added later - cannot break the transcript.
    decoder = json.JSONDecoder()
    result: dict[str, Any] | None = None
    cursor = 0
    while cursor < len(stdout):
        while cursor < len(stdout) and stdout[cursor].isspace():
            cursor += 1
        if cursor >= len(stdout):
            break
        try:
            payload, end = decoder.raw_decode(stdout, cursor)
        except json.JSONDecodeError as error:
            raise TranscriptError(f"teacher output is not JSON: {error}") from error
        cursor = end
        if isinstance(payload, dict) and "generated_ids" in payload:
            result = payload
    if result is None:
        raise TranscriptError("teacher output carried no generation result")
    if result.get("ok") is not True:
        raise TranscriptError("teacher forward did not report ok")
    identifiers = result.get("generated_ids")
    finish = result.get("finish_reason")
    text = result.get("text", "")
    if not isinstance(identifiers, list) or not all(
        isinstance(value, int) and 0 <= value <= 0xFFFFFFFF for value in identifiers
    ):
        raise TranscriptError("teacher generated_ids are malformed")
    if len(identifiers) > MAX_COMPLETION_TOKENS:
        raise TranscriptError("teacher produced more tokens than the probe budget allows")
    if not isinstance(text, str) or not isinstance(finish, str) or not finish:
        raise TranscriptError("teacher text or finish_reason is missing")
    return text, [int(value) for value in identifiers], finish


def transcribe(probes: list[dict[str, Any]], runner: Callable[[list[str]], Any],
               vastloom: Path, checkpoint: Path, tokenizer: Path,
               chat: bool = True, layers: int = 43,
               trunk_window: int = 43,
               runtime: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for probe in probes:
        process = runner(build_command(vastloom, checkpoint, tokenizer, probe, chat,
                                       layers, trunk_window))
        if getattr(process, "returncode", 1) != 0:
            tail = (getattr(process, "stderr", "") or "")[-400:]
            raise TranscriptError(
                f"teacher forward failed for probe {probe['id']}: {tail}")
        text, identifiers, finish = parse_forward_output(process.stdout)
        record: dict[str, Any] = {
            "id": probe["id"],
            "prompt": probe["prompt"],
            "completion_text": text,
            "completion_ids": identifiers,
            "finish_reason": finish,
        }
        if "expected" in probe:
            record["expected"] = probe["expected"]
            # The teacher-accuracy gate: distilling from a wrong target would bake the error
            # into the student, so every record carries whether the teacher itself was right.
            record["teacher_correct"] = (
                normalize_answer(text) == normalize_answer(probe["expected"]))
        if runtime:
            record["runtime"] = dict(runtime)
        records.append(record)
    return records


def write_transcript(path: Path, records: list[dict[str, Any]], checkpoint_name: str) -> None:
    envelope: dict[str, Any] = {"format": TRANSCRIPT_FORMAT, "teacher": checkpoint_name}
    envelope["records"] = records
    encoded = (json.dumps(envelope, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if len(encoded) > MAX_TRANSCRIPT_BYTES:
        raise TranscriptError("transcript exceeds the artifact size ceiling")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def read_transcript(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TranscriptError(f"unreadable transcript {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != TRANSCRIPT_FORMAT:
        raise TranscriptError("transcript format is unrecognized")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise TranscriptError("transcript has no records")
    return payload


def main(argv: list[str] | None = None,
         runner: Callable[[list[str]], Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-chat", action="store_true",
                        help="send raw prompts without the chat template")
    parser.add_argument("--layers", type=int, default=43,
                        help="teacher forward depth (43 = full pinned checkpoint)")
    parser.add_argument("--trunk-window", type=int, default=43,
                        help="validated trunk window for the CPU+ggml path")
    arguments = parser.parse_args(argv)

    if runner is None:
        def runner(command: list[str]) -> Any:
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            return subprocess.run(command, capture_output=True, text=True,
                                  encoding="utf-8", errors="strict",
                                  env=environment, timeout=3_600, check=False)

    try:
        probes = load_probes(arguments.probes)

        # Eval probes are graded against their known expected answers and never cost a
        # teacher run: only the train split is transcribed.
        probes = [probe for probe in probes if probe["split"] == "train"]
        if not probes:
            raise TranscriptError("the probe set contains no train-split probes")

        # Resume: a Governor refusal on probe five must not throw away the transcripts already
        # paid for. Existing records are kept verbatim; only missing ids run, and the merged
        # artifact always follows the probe set's order.
        runtime_meta = {"forward_layers": arguments.layers,
                        "trunk_window": arguments.trunk_window,
                        "chat_template": not arguments.no_chat}
        existing: dict[str, dict[str, Any]] = {}
        if arguments.out.exists():
            try:
                for record in read_transcript(arguments.out)["records"]:
                    existing[record["id"]] = record
            except TranscriptError:
                existing = {}

        pending = [probe for probe in probes if probe["id"] not in existing]
        fresh_count = 0
        # Persist after every single success: the teacher costs minutes per probe and the
        # host's RAM budget can refuse the next one at any time, so nothing already paid for
        # may depend on the remaining probes also succeeding.
        for probe in pending:
            one = transcribe([probe], runner, arguments.vastloom, arguments.checkpoint,
                             arguments.tokenizer, chat=not arguments.no_chat,
                             layers=arguments.layers, trunk_window=arguments.trunk_window,
                             runtime=runtime_meta)
            existing[probe["id"]] = one[0]
            fresh_count += 1
            ordered_now = [existing[p["id"]] for p in probes if p["id"] in existing]
            write_transcript(arguments.out, ordered_now,
                             arguments.checkpoint.resolve().name)
        # Records written before a probe gained an "expected" field are backfilled here: the
        # accuracy gate is a pure function of the stored completion text and the expectation,
        # so deriving it needs no new teacher run.
        for probe in probes:
            record = existing.get(probe["id"])
            if record is not None and "expected" in probe:
                record["expected"] = probe["expected"]
                record["teacher_correct"] = (
                    normalize_answer(record.get("completion_text", "")) ==
                    normalize_answer(probe["expected"]))
        records = [existing[probe["id"]] for probe in probes]
        # Always publish the final merged state, even when nothing new ran: this is what
        # prunes records whose probes left the set, and it keeps the file truthful.
        write_transcript(arguments.out, records, arguments.checkpoint.resolve().name)
    except TranscriptError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {len(records)} teacher transcripts to {arguments.out} "
          f"({fresh_count} new)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
