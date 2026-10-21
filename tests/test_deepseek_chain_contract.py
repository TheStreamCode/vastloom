#!/usr/bin/env python3
"""Opt-in contract for the DeepSeek 0731 hash-routed chain probe.

The chain probe executes 2 or 3 hash-routed layers from one trunk and one authoritative expert
artifact per layer. This driver pins the measured 2/3-layer results, checks that a repeated run is
byte-identical, and checks that malformed chains are refused. It never skips the comparison: a
missing artifact is an error, and only host pressure or a missing CUDA driver (for the optional
GPU leg) produces the documented skip.
"""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from host_pressure import host_pressure_reason, skipped


CONTRACT = "0731 hash-routed chain contract"
LOGIT_ATOL = 1.0e-3

DEPTH3_CONTRACT = {
    "complete_model": False,
    "executed_layers": 3,
    "model_layers": 43,
    "token": 0,
    "logits": 129_280,
    "argmax_token": 86_051,
    "trunk_loads": 3,
    "trunk_bytes_read": 437_727_112,
    "expert_bytes_read": 135_659_520,
    "expert_misses": 18,
    "relocated_tensors": 133,
    "relocated_expert_bundles": 18,
}

DEPTH3_ROUTES = [
    [35, 53, 200, 222, 245, 254],
    [8, 97, 137, 158, 163, 184],
    [43, 53, 108, 115, 132, 143],
]

DEPTH3_LOGITS = {"logit_minimum": -16.0034, "logit_maximum": 16.1624}

# A 5-token prompt crosses the layer-2 compression ratio (4), so the compressed window is non-empty.
MULTI_TOKENS = [0, 1, 2, 3, 4]
MULTI_LAYER_ROUTES = [
    [[35, 53, 200, 222, 245, 254], [23, 57, 115, 122, 202, 239], [37, 38, 45, 96, 108, 228],
     [42, 51, 69, 102, 162, 182], [7, 67, 81, 154, 239, 246]],
    [[8, 97, 137, 158, 163, 184], [21, 63, 126, 150, 170, 219], [13, 16, 17, 46, 73, 237],
     [30, 102, 177, 192, 205, 242], [71, 78, 125, 186, 188, 253]],
    [[43, 53, 108, 115, 132, 143], [26, 39, 217, 221, 240, 247], [66, 87, 153, 185, 200, 216],
     [71, 83, 85, 102, 122, 216], [12, 32, 52, 101, 142, 224]],
]
MULTI_CONTRACT = {
    "executed_layers": 3,
    "prompt_tokens": 5,
    "logits": 129_280,
    "argmax_token": 20_267,
    "expert_bytes_read": 663_224_320,
    "relocated_tensors": 343,
    "relocated_expert_bundles": 88,
}
MULTI_LOGITS = {"logit_minimum": -38.0188, "logit_maximum": 14.7002}
DEPTH2_CONTRACT = {
    "executed_layers": 2,
    "trunk_bytes_read": 277_734_320,
    "expert_bytes_read": 90_439_680,
    "argmax_token": 86_051,
}
DEPTH2_LOGITS = {"logit_minimum": -16.2274, "logit_maximum": 15.7631}


def run(vastloom: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(vastloom), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=3_600,
    )


def require(
    condition: bool,
    message: str,
    process: subprocess.CompletedProcess[str] | None = None,
) -> None:
    if condition:
        return
    detail = ""
    if process is not None:
        detail = f"\nexit={process.returncode}\nstdout={process.stdout}\nstderr={process.stderr}"
    raise AssertionError(message + detail)


def read_f32(path: Path) -> array.array[float]:
    values: array.array[float] = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def chain_arguments(args: argparse.Namespace, layers: int, *, dump: Path | None = None,
                    gpu_limit: int | None = None) -> list[str]:
    values = [
        "deepseek-chain-probe",
        str(args.model),
        str(args.globals_),
        "--layers",
        str(layers),
        "--token",
        str(args.token),
    ]
    for layer, trunk in enumerate((args.trunk0, args.trunk1, args.trunk2)):
        if layer < layers:
            values.extend(("--trunk", f"{layer}={trunk}"))
    for layer, experts in enumerate((args.experts0, args.experts1, args.experts2)):
        if layer < layers:
            values.extend(("--experts", f"{layer}={experts}"))
    if gpu_limit is not None:
        values.extend(("--cuda-experts", "--gpu-limit", str(gpu_limit)))
    if dump is not None:
        values.extend(("--dump-logits", str(dump)))
    values.append("--json")
    return values


def check_contract(payload: dict[str, object], expected: dict[str, object]) -> None:
    mismatches = [
        f"  {key}: expected {value!r}, got {payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise AssertionError(f"{CONTRACT} changed:\n" + "\n".join(mismatches))


def check_logits(payload: dict[str, object], expected: dict[str, float]) -> None:
    for key, value in expected.items():
        actual = payload.get(key)
        require(
            isinstance(actual, (int, float)) and abs(float(actual) - value) <= LOGIT_ATOL,
            f"{CONTRACT}: {key} is {actual!r}, expected {value} within {LOGIT_ATOL}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--globals", dest="globals_", type=Path, required=True)
    for index in range(3):
        parser.add_argument(f"--trunk{index}", type=Path, required=True)
        parser.add_argument(f"--experts{index}", type=Path, required=True)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument(
        "--multi-expert",
        action="append",
        default=[],
        metavar="TOKEN:LAYER=PATH",
        help="expert artifact for one prompt token and layer; 15 maps enable the 5-token leg",
    )
    args = parser.parse_args()
    for name in ("model", "globals_", "trunk0", "trunk1", "trunk2", "experts0", "experts1",
                 "experts2"):
        setattr(args, name, getattr(args, name).resolve(strict=True))

    scratch_parent = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="vastloom-chain-", dir=scratch_parent) as directory:
        root = Path(directory)
        depth3 = root / "depth3.f32"
        depth3_repeat = root / "depth3-repeat.f32"
        depth2 = root / "depth2.f32"

        first = run(args.vastloom, chain_arguments(args, 3, dump=depth3))
        pressure = host_pressure_reason(first)
        if pressure is not None:
            return skipped(CONTRACT, pressure)
        require(first.returncode == 0, "3-layer chain failed", first)
        payload = json.loads(first.stdout)
        check_contract(payload, DEPTH3_CONTRACT)
        check_logits(payload, DEPTH3_LOGITS)
        actual_layers = payload.get("layers")
        require(
            isinstance(actual_layers, list) and len(actual_layers) == len(DEPTH3_ROUTES),
            "3-layer chain reported the wrong number of layer records",
            first,
        )
        for index, expected_route in enumerate(DEPTH3_ROUTES):
            layer = actual_layers[index]
            require(
                layer.get("layer") == index and layer.get("route") == expected_route,
                f"3-layer chain layer {index} route changed: {layer!r}",
                first,
            )
            require(
                layer.get("routes_per_token") == [expected_route],
                f"3-layer chain layer {index} per-token routes changed: {layer!r}",
                first,
            )

        repeat = run(args.vastloom, chain_arguments(args, 3, dump=depth3_repeat))
        require(repeat.returncode == 0, "3-layer chain repeat failed", repeat)
        require(
            read_f32(depth3) == read_f32(depth3_repeat),
            "3-layer chain is not deterministic across runs",
            repeat,
        )

        second = run(args.vastloom, chain_arguments(args, 2, dump=depth2))
        require(second.returncode == 0, "2-layer chain failed", second)
        payload = json.loads(second.stdout)
        check_contract(payload, DEPTH2_CONTRACT)
        check_logits(payload, DEPTH2_LOGITS)

        malformed = run(
            args.vastloom,
            [
                "deepseek-chain-probe",
                str(args.model),
                str(args.globals_),
                "--layers",
                "4",
                "--token",
                "0",
                "--trunk",
                f"0={args.trunk0}",
                "--experts",
                f"0={args.experts0}",
                "--json",
            ],
        )
        require(malformed.returncode != 0, "a 4-layer chain was not refused", malformed)
        duplicate = run(
            args.vastloom,
            [
                "deepseek-chain-probe",
                str(args.model),
                str(args.globals_),
                "--layers",
                "2",
                "--token",
                "0",
                "--trunk",
                f"0={args.trunk0}",
                "--trunk",
                f"0={args.trunk0}",
                "--experts",
                f"0={args.experts0}",
                "--experts",
                f"1={args.experts1}",
                "--json",
            ],
        )
        require(duplicate.returncode != 0, "a duplicate layer argument was not refused", duplicate)
        missing = run(
            args.vastloom,
            [
                "deepseek-chain-probe",
                str(args.model),
                str(args.globals_),
                "--layers",
                "2",
                "--token",
                "0",
                "--trunk",
                f"0={args.trunk0}",
                "--trunk",
                f"1={args.trunk1}",
                "--experts",
                f"0={args.experts0}",
                "--json",
            ],
        )
        require(missing.returncode != 0, "a chain without layer-1 experts was not refused", missing)

        info = run(args.vastloom, ["cuda-info", "--json"])
        if info.returncode == 0:
            gpu_path = root / "depth3-gpu.f32"
            gpu = run(args.vastloom, chain_arguments(args, 3, dump=gpu_path, gpu_limit=64))
            require(gpu.returncode == 0, "3-layer GPU chain failed", gpu)
            gpu_payload = json.loads(gpu.stdout)
            require(
                gpu_payload["argmax_token"] == DEPTH3_CONTRACT["argmax_token"],
                "GPU chain argmax changed",
                gpu,
            )
            reference = read_f32(depth3)
            candidate = read_f32(gpu_path)
            require(len(reference) == len(candidate), "GPU chain logits length changed", gpu)
            difference = max(abs(left - right) for left, right in zip(reference, candidate))
            require(
                difference <= LOGIT_ATOL,
                f"GPU chain logits differ from the CPU chain by {difference}",
                gpu,
            )
            require(
                gpu_payload.get("accelerator", {}).get("accelerated", 0) > 0,
                "GPU chain reported no accelerated matrix",
                gpu,
            )
            print(f"{CONTRACT}: GPU leg max|delta|={difference:.3e}")
        else:
            print(f"{CONTRACT}: GPU leg skipped, no usable CUDA driver", file=sys.stderr)

        if args.multi_expert:
            experts_by_token_layer: dict[tuple[int, int], Path] = {}
            for raw in args.multi_expert:
                try:
                    token_text, rest = raw.split(":", 1)
                    layer_text, path_text = rest.split("=", 1)
                    token = int(token_text)
                    layer = int(layer_text)
                except ValueError as error:
                    raise SystemExit(f"{CONTRACT}: malformed --multi-expert value {raw!r}") from error
                require(
                    0 <= token < len(MULTI_TOKENS) and 0 <= layer < 3,
                    f"{CONTRACT}: --multi-expert token/layer outside 0..4/0..2: {raw!r}",
                )
                experts_by_token_layer[(token, layer)] = Path(path_text).resolve(strict=True)
            require(
                len(experts_by_token_layer) == len(MULTI_TOKENS) * 3,
                f"{CONTRACT}: the 5-token leg needs one expert artifact per token and layer",
            )
            multi_path = root / "multi.f32"
            multi_command = [
                "deepseek-chain-probe",
                str(args.model),
                str(args.globals_),
                "--layers",
                "3",
                "--tokens",
                ",".join(str(token) for token in MULTI_TOKENS),
                "--json",
                "--dump-logits",
                str(multi_path),
            ]
            for layer, trunk in enumerate((args.trunk0, args.trunk1, args.trunk2)):
                multi_command.extend(("--trunk", f"{layer}={trunk}"))
            for token in MULTI_TOKENS:
                for layer in range(3):
                    multi_command.extend(
                        ("--experts", f"{layer}={experts_by_token_layer[(token, layer)]}")
                    )
            multi = run(args.vastloom, multi_command)
            require(multi.returncode == 0, "5-token chain failed", multi)
            multi_payload = json.loads(multi.stdout)
            check_contract(multi_payload, MULTI_CONTRACT)
            check_logits(multi_payload, MULTI_LOGITS)
            require(
                [layer.get("routes_per_token") for layer in multi_payload.get("layers", [])]
                == MULTI_LAYER_ROUTES,
                "5-token chain per-token routes changed",
                multi,
            )
            repeat_path = root / "multi-repeat.f32"
            repeat_command = [part for part in multi_command]
            repeat_command[repeat_command.index(str(multi_path))] = str(repeat_path)
            multi_repeat = run(args.vastloom, repeat_command)
            require(multi_repeat.returncode == 0, "5-token chain repeat failed", multi_repeat)
            require(
                read_f32(multi_path) == read_f32(repeat_path),
                "5-token chain is not deterministic across runs",
                multi_repeat,
            )
            print(f"{CONTRACT}: 5-token leg PASS (compression window open)")

    print(f"{CONTRACT}: PASS (3 and 2 layers, deterministic, malformed chains refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
