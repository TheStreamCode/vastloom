#!/usr/bin/env python3
"""Real-metadata, no-network integration test for the bounded range importer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from host_pressure import host_pressure_reason, skipped


CONTRACT = "bounded range importer real-metadata contract"


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
    # Name the store's pinned quantization: IQ2_XXS is the active profile, IQ1_S is the historical
    # test profile. Both use the same architecture and layer-0 trunk, so the plan assertions below
    # stay identical; only the source-object identity in the importer changes.
    model_alias = (
        "deepseek-v4-flash-0731-gguf-iq2xxs"
        if (model / "UD-IQ2_XXS").exists()
        else "deepseek-v4-flash-0731-gguf-iq1s"
    )
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
        pressure = host_pressure_reason(planned)
        if pressure is not None:
            return skipped(CONTRACT, pressure)
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
            model_alias,
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
            model_alias,
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

        global_plan_path = Path(temporary) / "global-plan.json"
        global_planned = run(
            str(args.vastloom),
            "deepseek-gguf-global-plan",
            str(model),
            "--json",
        )
        require(global_planned.returncode == 0, "real metadata global planning failed", global_planned)
        global_plan = json.loads(global_planned.stdout)
        require(
            global_plan["format"] == "vastloom.deepseek_v4.global_tensor_plan.v1"
            and global_plan["tensor_count"] == 6
            and global_plan["payload_bytes"] == 596_000_788
            and global_plan["routing_authority_verified"] is False,
            "real metadata global plan contract changed",
            global_planned,
        )
        global_plan_path.write_text(global_planned.stdout, encoding="utf-8", newline="\n")
        global_dry_run = run(
            sys.executable,
            str(args.importer),
            "--model",
            model_alias,
            "--plan",
            str(global_plan_path),
            "--store",
            str(store),
            "--scope",
            "globals",
            "--reserve-gib",
            "20",
            "--max-total-mib",
            "569",
            "--dry-run",
            "--json",
        )
        require(
            global_dry_run.returncode == 0,
            "real metadata global range-import dry-run failed",
            global_dry_run,
        )
        global_result = json.loads(global_dry_run.stdout)
        require(
            global_result["dry_run"] is True
            and global_result["execution_supported"] is False
            and global_result["payload_bytes"] == 596_000_788
            and global_result["range_count"] == 6
            and not Path(global_result["target"]).exists(),
            "real metadata global dry-run mutated storage or changed accounting",
            global_dry_run,
        )

    print("DeepSeek range importer CLI tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
