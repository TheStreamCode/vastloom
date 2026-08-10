#!/usr/bin/env python3
"""Real-metadata, no-network integration test for the bounded range importer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, check=False, capture_output=True, text=True, encoding="utf-8")


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str]) -> None:
    if not condition:
        raise AssertionError(
            f"{message}\nexit={process.returncode}\nstdout={process.stdout}\nstderr={process.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--importer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve(strict=True)
    store = model.parents[2]
    if not (store / ".vastloom-store").is_file():
        raise AssertionError(f"model is not inside a Vastloom store: {model}")
    with tempfile.TemporaryDirectory(prefix="vastloom-range-cli-") as temporary:
        plan_path = Path(temporary) / "layer-0-plan.json"
        planned = run(
            str(args.vastloom),
            "deepseek-gguf-probe-plan",
            str(model),
            "--layer",
            "0",
            "--experts",
            "0,1,2,3,4,5",
            "--json",
        )
        require(planned.returncode == 0, "real metadata range planning failed", planned)
        plan = json.loads(planned.stdout)
        require(
            plan["trunk_bytes"] == 138_867_160
            and plan["trunk_range_count"] == 21
            and plan["routing_authority_verified"] is False,
            "real metadata range-plan contract changed",
            planned,
        )
        plan_path.write_text(planned.stdout, encoding="utf-8", newline="\n")

        dry_run = run(
            sys.executable,
            str(args.importer),
            "--model",
            "deepseek-v4-flash-0731-gguf-iq1s",
            "--plan",
            str(plan_path),
            "--store",
            str(store),
            "--scope",
            "trunk",
            "--reserve-gib",
            "20",
            "--max-total-mib",
            "133",
            "--dry-run",
            "--json",
        )
        require(dry_run.returncode == 0, "real metadata range-import dry-run failed", dry_run)
        result = json.loads(dry_run.stdout)
        target = Path(result["target"])
        require(
            result["dry_run"] is True
            and result["execution_supported"] is False
            and result["payload_bytes"] == 138_867_160
            and result["range_count"] == 21
            and not target.exists(),
            "real metadata dry-run mutated storage or changed its accounting",
            dry_run,
        )

        unauthorized = run(
            sys.executable,
            str(args.importer),
            "--model",
            "deepseek-v4-flash-0731-gguf-iq1s",
            "--plan",
            str(plan_path),
            "--store",
            str(store),
            "--scope",
            "selected-experts",
            "--reserve-gib",
            "20",
            "--max-total-mib",
            "64",
            "--dry-run",
            "--json",
        )
        require(
            unauthorized.returncode != 0
            and "routing authority" in unauthorized.stdout
            and not target.exists(),
            "caller-supplied expert IDs reached the range importer",
            unauthorized,
        )

    print("DeepSeek range importer CLI tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
