#!/usr/bin/env python3
"""Opt-in contract: packed experts load in one read and reproduce the v1 chain bit for bit.

The test builds derived packs from already-imported expert artifacts, then runs
``deepseek-chain-probe`` with the v1 artifacts (three reads per expert) and with the packs (one
read per expert). Logits and hidden dumps must be byte-identical, and the JSON read counters must
show the expected reduction. A multi-token run covers the union/merge path; a layer-0 run covers the
single-layer probe. It never downloads weights.
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


def parse_layered(raw_values: list[str], option: str) -> dict[int, Path]:
    values = {}
    for raw in raw_values:
        if "=" not in raw:
            raise RuntimeError(f"{option} expects LAYER=PATH, got {raw!r}")
        layer_text, path = raw.split("=", 1)
        layer = int(layer_text)
        if layer in values:
            raise RuntimeError(f"duplicate {option} layer {layer}")
        values[layer] = Path(path)
    return values


def parse_token_layered(raw_values: list[str]) -> dict[tuple[int, int], Path]:
    values = {}
    for raw in raw_values:
        if "=" not in raw or ":" not in raw.split("=", 1)[0]:
            raise RuntimeError(f"--experts-token expects TOKEN:LAYER=PATH, got {raw!r}")
        head, path = raw.split("=", 1)
        token_text, layer_text = head.split(":", 1)
        key = (int(token_text), int(layer_text))
        if key in values:
            raise RuntimeError(f"duplicate expert set for token {key[0]} layer {key[1]}")
        values[key] = Path(path)
    return values


def artifact_experts(artifact: Path) -> list[int]:
    manifest = json.loads((artifact / "vastloom-range-artifact.json").read_text())
    experts = set()
    for entry in manifest["ranges"]:
        match = LABEL.match(entry["label"])
        if match is None:
            raise RuntimeError(f"unexpected expert artifact label {entry['label']!r}")
        experts.add(int(match.group(1)))
    return sorted(experts)


def make_plan(vastloom: Path, model: Path, layer: int, artifact: Path, work: Path,
              tag: str) -> Path:
    experts = artifact_experts(artifact)
    result = run([str(vastloom), "deepseek-gguf-probe-plan", str(model), "--layer", str(layer),
                  "--experts", ",".join(str(expert) for expert in experts), "--json"])
    plan = json.loads(result.stdout.strip().splitlines()[-1])
    if not plan.get("ok"):
        raise RuntimeError(f"probe plan failed: {plan}")
    path = work / f"layer{layer}-{tag}-plan.json"
    path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    return path


def chain_json(vastloom: Path, model: Path, globals_path: Path, trunks: dict[int, Path],
               expert_sets: dict[tuple[int, int], Path] | None, packs: dict[int, Path],
               layers: int, tokens: list[int], io_cache: str, logits: Path, hidden: Path) -> dict:
    command = [str(vastloom), "deepseek-chain-probe", str(model), str(globals_path),
               "--layers", str(layers), "--tokens", ",".join(str(token) for token in tokens)]
    for layer in range(layers):
        command += ["--trunk", f"{layer}={trunks[layer]}"]
        if expert_sets is not None:
            for token in tokens:
                command += ["--experts", f"{layer}={expert_sets[(token, layer)]}"]
        else:
            command += ["--expert-pack", f"{layer}={packs[layer]}"]
    command += ["--io-cache", io_cache, "--dump-logits", str(logits),
                "--dump-hidden", str(hidden), "--json"]
    result = run(command)
    return json.loads(result.stdout.strip().splitlines()[-1])


def layer_probe(vastloom: Path, model: Path, trunk: Path, globals_path: Path,
                expert_artifact: Path | None, pack: Path | None, token: int,
                io_cache: str, logits: Path) -> dict:
    command = [str(vastloom), "deepseek-layer-probe", str(model), str(trunk)]
    if expert_artifact is not None:
        command += [str(expert_artifact), str(globals_path), "--io-cache", io_cache]
    else:
        command += [str(globals_path), "--expert-pack", str(pack), "--io-cache", io_cache]
    command += ["--token", str(token), "--dump-logits", str(logits), "--json"]
    result = run(command)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--globals", type=Path, required=True)
    parser.add_argument("--trunk", action="append", required=True)
    parser.add_argument("--experts", action="append", default=[])
    parser.add_argument("--experts-token", action="append", default=[])
    parser.add_argument("--tool", type=Path, default=Path("tools/pack_deepseek_experts.py"))
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    trunks = parse_layered(args.trunk, "--trunk")
    expert_sets = parse_token_layered(args.experts_token)
    if args.experts:
        if expert_sets:
            parser.error("use either --experts (token 0) or --experts-token, not both")
        for layer, path in parse_layered(args.experts, "--experts").items():
            expert_sets[(0, layer)] = path
    if not expert_sets:
        parser.error("at least one --experts or --experts-token is required")
    tokens = sorted({token for token, _ in expert_sets})
    for token in tokens:
        for layer in range(args.layers):
            if (token, layer) not in expert_sets:
                raise RuntimeError(f"missing expert artifact for token {token} layer {layer}")
    for layer in range(args.layers):
        if layer not in trunks:
            raise RuntimeError(f"missing trunk artifact for layer {layer}")

    work = Path(tempfile.mkdtemp(prefix="vastloom-expert-pack-cli-", dir=Path.cwd()))
    try:
        packs: dict[int, Path] = {}
        for layer in range(args.layers):
            command = [sys.executable, str(args.tool.resolve()), "--layer", str(layer),
                       "--output", str(work / f"pack-L{layer}")]
            for token in tokens:
                artifact = expert_sets[(token, layer)].resolve()
                plan = make_plan(args.vastloom.resolve(), args.model.resolve(), layer,
                                 artifact, work, f"t{token}")
                command += ["--plan", str(plan), "--artifact", str(artifact)]
            run(command)
            packs[layer] = work / f"pack-L{layer}"

        v1_logits = work / "v1-logits.f32"
        v1_hidden = work / "v1-hidden.f32"
        v1 = chain_json(args.vastloom, args.model, args.globals, trunks, expert_sets, {},
                        args.layers, tokens, "buffered", v1_logits, v1_hidden)

        packed_results = []
        for mode in ("buffered", "bypass"):
            logits = work / f"pack-{mode}-logits.f32"
            hidden = work / f"pack-{mode}-hidden.f32"
            packed = chain_json(args.vastloom, args.model, args.globals, trunks, None, packs,
                                args.layers, tokens, mode, logits, hidden)
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
            if packed["expert_bypass_read_calls"] + packed["expert_buffered_read_calls"] != \
                    packed["expert_read_calls"]:
                failures.append(f"{mode}: read-path counters do not add up")
            if mode == "bypass" and packed["expert_bypass_read_calls"] != \
                    packed["expert_read_calls"]:
                failures.append("bypass: requested bypass did not deliver every read")

        expected_v1_reads = 3 * v1["expert_misses"]
        if v1["expert_read_calls"] != expected_v1_reads:
            failures.append(f"v1: expected {expected_v1_reads} reads, "
                            f"got {v1['expert_read_calls']}")

        if (args.token, 0) in expert_sets:
            probe_logits_v1 = work / "layer-v1-logits.f32"
            probe_logits_pack = work / "layer-pack-logits.f32"
            probe_v1 = layer_probe(args.vastloom, args.model, trunks[0], args.globals,
                                   expert_sets[(args.token, 0)], None, args.token, "buffered",
                                   probe_logits_v1)
            probe_pack = layer_probe(args.vastloom, args.model, trunks[0], args.globals,
                                     None, packs[0], args.token, "bypass", probe_logits_pack)
            if probe_v1["argmax_token"] != probe_pack["argmax_token"] or \
                    probe_logits_v1.read_bytes() != probe_logits_pack.read_bytes():
                failures.append("layer-probe: pack logits differ")
            if probe_pack["expert_read_calls"] != probe_pack["expert_misses"]:
                failures.append("layer-probe: expected one read per expert")
            if probe_v1["expert_read_calls"] != 3 * probe_v1["expert_misses"]:
                failures.append("layer-probe: v1 baseline is not three reads per expert")

        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        print(f"PASS: tokens {tokens}; v1 {v1['expert_read_calls']} reads -> pack "
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
