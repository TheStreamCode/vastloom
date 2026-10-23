#!/usr/bin/env python3
"""Opt-in contract: packed experts load in one read and reproduce the v1 chain bit for bit.

The test builds one derived pack per chain layer from the already-imported expert artifacts, then
runs ``deepseek-chain-probe`` twice: once with the v1 expert artifacts (three reads per expert) and
once with the packs (one read per expert). Logits and hidden dumps must be byte-identical, and the
JSON read counters must show the expected reduction.

Requires real local artifacts; it is skipped in a plain checkout when the CMake opt-in variables are
absent. It never downloads weights.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LABEL = re.compile(r"^expert\[\d+:(\d+)\]\.range\[\d+\]$")


def run(command: list[str], expect: int = 0) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != expect:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def artifact_experts(artifact: Path) -> list[int]:
    manifest = json.loads((artifact / "vastloom-range-artifact.json").read_text())
    experts = set()
    for entry in manifest["ranges"]:
        match = LABEL.match(entry["label"])
        if match is None:
            raise RuntimeError(f"unexpected expert artifact label {entry['label']!r}")
        experts.add(int(match.group(1)))
    return sorted(experts)


def make_plan(vastloom: Path, model: Path, layer: int, artifact: Path, work: Path) -> Path:
    experts = artifact_experts(artifact)
    result = run([str(vastloom), "deepseek-gguf-probe-plan", str(model), "--layer", str(layer),
                  "--experts", ",".join(str(expert) for expert in experts), "--json"])
    plan = json.loads(result.stdout.strip().splitlines()[-1])
    if not plan.get("ok"):
        raise RuntimeError(f"probe plan failed: {plan}")
    path = work / f"layer{layer}-plan.json"
    path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    return path


def chain_json(vastloom: Path, model: Path, globals_path: Path, trunks: dict[int, Path],
               experts: dict[int, Path] | None, packs: dict[int, Path], layers: int,
               io_cache: str, logits: Path, hidden: Path) -> dict:
    command = [str(vastloom), "deepseek-chain-probe", str(model), str(globals_path),
               "--layers", str(layers), "--token", "0"]
    for layer in range(layers):
        command += ["--trunk", f"{layer}={trunks[layer]}"]
        if experts is not None:
            command += ["--experts", f"{layer}={experts[layer]}"]
        else:
            command += ["--expert-pack", f"{layer}={packs[layer]}"]
    command += ["--io-cache", io_cache, "--dump-logits", str(logits),
                "--dump-hidden", str(hidden), "--json"]
    result = run(command)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--globals", type=Path, required=True)
    parser.add_argument("--trunk", action="append", required=True)
    parser.add_argument("--experts", action="append", required=True)
    parser.add_argument("--tool", type=Path, default=Path("tools/pack_deepseek_experts.py"))
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    trunks = {int(raw.split("=", 1)[0]): Path(raw.split("=", 1)[1]) for raw in args.trunk}
    experts = {int(raw.split("=", 1)[0]): Path(raw.split("=", 1)[1]) for raw in args.experts}
    missing = [layer for layer in range(args.layers)
               if layer not in trunks or layer not in experts]
    if missing:
        print(f"missing artifacts for layers {missing}", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="vastloom-expert-pack-cli-", dir=Path.cwd()))
    try:
        packs: dict[int, Path] = {}
        for layer in range(args.layers):
            plan = make_plan(args.vastloom.resolve(), args.model.resolve(), layer,
                             experts[layer].resolve(), work)
            output = work / f"pack-L{layer}"
            run([sys.executable, str(args.tool.resolve()), "--layer", str(layer),
                 "--output", str(output), "--plan", str(plan),
                 "--artifact", str(experts[layer].resolve())])
            packs[layer] = output

        v1_logits = work / "v1-logits.f32"
        v1_hidden = work / "v1-hidden.f32"
        v1 = chain_json(args.vastloom, args.model, args.globals, trunks, experts, {},
                        args.layers, "buffered", v1_logits, v1_hidden)

        packed_results = []
        for mode in ("buffered", "bypass"):
            logits = work / f"pack-{mode}-logits.f32"
            hidden = work / f"pack-{mode}-hidden.f32"
            packed = chain_json(args.vastloom, args.model, args.globals, trunks, None, packs,
                                args.layers, mode, logits, hidden)
            packed_results.append((mode, packed, logits, hidden))

        failures = []
        for mode, packed, logits, hidden in packed_results:
            if v1["argmax_token"] != packed["argmax_token"]:
                failures.append(f"{mode}: argmax differs")
            if v1["logit_minimum"] != packed["logit_minimum"] or \
                    v1["logit_maximum"] != packed["logit_maximum"]:
                failures.append(f"{mode}: logit extrema differ")
            if v1_logits.read_bytes() != logits.read_bytes():
                failures.append(f"{mode}: logits dump differs")
            if v1_hidden.read_bytes() != hidden.read_bytes():
                failures.append(f"{mode}: hidden dump differs")
            if packed["expert_source"] != "expert-pack":
                failures.append(f"{mode}: expert source is {packed['expert_source']}")
            if packed["expert_read_calls"] != packed["expert_misses"]:
                failures.append(f"{mode}: expected one read per expert, got "
                                f"{packed['expert_read_calls']} for {packed['expert_misses']}")
            if mode == "bypass" and any(value != "bypass" for value in packed["io_cache_modes"]):
                failures.append("bypass: io cache modes did not resolve to bypass")

        expected_v1_reads = sum(3 for _ in range(v1["expert_misses"]))
        if v1["expert_read_calls"] != expected_v1_reads:
            failures.append(f"v1: expected {expected_v1_reads} reads, "
                            f"got {v1['expert_read_calls']}")
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        print(f"PASS: v1 {v1['expert_read_calls']} reads -> pack "
              f"{packed_results[0][1]['expert_read_calls']} reads; logits/hidden identical")
        return 0
    finally:
        if args.keep:
            print(f"work directory kept at {work}")
        else:
            import shutil
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
