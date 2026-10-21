#!/usr/bin/env python3
"""Pinned JSON contracts for the zero-payload DeepSeek 0731 GGUF commands.

These two commands used to be checked with one long `PASS_REGULAR_EXPRESSION` per test. That
matched fields positionally, so reordering the JSON could silently weaken the check, and any
Governor refusal was reported as a contract violation. Comparing parsed keys removes both problems
and keeps the pinned values visible in one place.

Neither command reads payload: both work from verified metadata only.
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


# Keys are dotted paths into the JSON document. The previous regular expression matched anywhere
# in the serialised text, so it could not tell a top-level field from a nested one; these paths can.
CHECKPOINT_CONTRACT = {
    "checkpoint.imported_manifest_verified": True,
    "checkpoint.pinned_0731_checkpoint_spec_verified": True,
    "checkpoint.reference_config_verified": True,
    "checkpoint.layer0_reference_catalog_verified": True,
    "checkpoint.all_layer_reference_catalogs_verified": True,
    "checkpoint.checkpoint_reference_catalog_verified": True,
    "checkpoint.reference_catalog_validated_layers": 43,
    "checkpoint.reference_catalog_validated_tensors": 1_193,
    "checkpoint.reference_catalog_validated_bytes": 6_390_088_008,
    "checkpoint.reference_catalog_global_tensors": 6,
    "checkpoint.reference_catalog_global_bytes": 596_000_788,
    "checkpoint.reference_catalog_expert_bundles": 11_008,
    "checkpoint.reference_catalog_expert_tensor_slices": 33_024,
    "checkpoint.reference_catalog_physical_expert_tensors": 129,
    "reference_architecture.vocabulary_size": 129_280,
    "reference_architecture.compression_schedule_values": 46,
    "reference_architecture.routed_swiglu_schedule_values": 43,
    "reference_architecture.shared_swiglu_schedule_values": 43,
    "physical_tensors": 1_328,
    "logical_tensors": 34_223,
    "tensor_bytes": 90_855_391_580,
    "always_active_bytes": 6_986_088_796,
    "routed_expert_bytes": 83_869_302_784,
    "layer_streaming.planned_layers": 43,
}

PROBE_PLAN_CONTRACT = {
    "plan_only": True,
    "routing_authority_verified": False,
    "trunk_bytes": 138_867_160,
    "selected_expert_bytes": 45_219_840,
    "total_payload_bytes": 184_087_000,
    "total_range_count": 39,
    "unique_source_objects": 1,
}

# The imported layer-0 trunk, reread and rehashed into one governed RAM buffer, then mapped onto the
# graph's declared consumers: 11 matrices, 9 vectors and the routing table.
TRUNK_LOAD_CONTRACT = {
    "layer": 0,
    "resident_bytes": 138_867_160,
    "tensors": 21,
    "graph_matrices": 11,
    "graph_float_tensors": 9,
    "graph_integer_tensors": 1,
    "ranges_read": 21,
    "execution_supported": False,
}

# The six globals, dominated by two 297,861,120-byte Q4_K matrices, reaching the embedding, output
# head, normalization and Hyper-Connection consumers through one governed reservation.
GLOBALS_LOAD_CONTRACT = {
    "resident_bytes": 596_000_788,
    "tensors": 6,
    "graph_matrices": 3,
    "graph_float_tensors": 3,
    "ranges_read": 6,
    "execution_supported": False,
}

# One real layer of 43, executed from official IQ2_XXS payload: Q8_0 attention, Q5_K/Q6_K shared
# expert, BF16 router, I32 routing table, IQ2_XXS/IQ3_XXS routed experts and the Q4_K embedding,
# with the checkpoint's own router choosing the six experts. These logits are the first layer's,
# not the model's.
LAYER_PROBE_CONTRACT = {
    "complete_model": False,
    "executed_layers": 1,
    "model_layers": 43,
    "token": 0,
    "logits": 129_280,
    "argmax_token": 30_528,
    "experts_executed": [254, 222, 245, 200, 53, 35],
    "expert_bytes_read": 45_219_840,
    "expert_misses": 6,
    "trunk_bytes": 138_867_160,
    "global_bytes": 596_000_788,
    "relocated_tensors": 45,
    "relocated_expert_bundles": 6,
}

ROUTE_CONTRACT = {
    "layer": 0,
    "expected_top_k": 6,
    "selection_source": "official-tid2eid-verified",
    "routing_authority_verified": True,
    "routing_token": 0,
    "selected_experts": [35, 53, 200, 222, 245, 254],
    "trunk_bytes": 138_867_160,
    "selected_expert_bytes": 45_219_840,
}

COMMANDS = {
    "checkpoint": (
        "0731 GGUF checkpoint catalog contract",
        ("deepseek-gguf-checkpoint", "{model}", "--json"),
        CHECKPOINT_CONTRACT,
    ),
    "probe-plan": (
        "0731 GGUF layer-0 probe-plan contract",
        ("deepseek-gguf-probe-plan", "{model}", "--layer", "0", "--experts", "0,1,2,3,4,5", "--json"),
        PROBE_PLAN_CONTRACT,
    ),
    "route": (
        "0731 authoritative tid2eid route contract",
        ("deepseek-gguf-probe-plan", "{model}", "--layer", "0",
         "--route-artifact", "{artifact}", "--token", "0", "--json"),
        ROUTE_CONTRACT,
    ),
    "trunk-load": (
        "0731 layer-0 trunk payload consumer contract",
        ("deepseek-gguf-range-bind", "{model}", "{artifact}", "--layer", "0",
         "--experts", "35,53,200,222,245,254", "--load-tensor-set", "--json"),
        TRUNK_LOAD_CONTRACT,
    ),
    "globals-load": (
        "0731 global payload consumer contract",
        ("deepseek-gguf-global-bind", "{model}", "{artifact}", "--load-tensor-set", "--json"),
        GLOBALS_LOAD_CONTRACT,
    ),
    "layer-probe": (
        "0731 official layer-0 execution contract",
        ("deepseek-layer-probe", "{model}", "{artifact}", "{experts}", "{globals}",
         "--token", "0", "--json"),
        LAYER_PROBE_CONTRACT,
    ),
}


MISSING = "<missing>"


def _probe(
    vastloom: Path,
    model: Path,
    trunk: Path,
    experts: Path,
    globals_: Path,
    *,
    extra: tuple[str, ...] = (),
    logits_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(vastloom),
        "deepseek-layer-probe",
        str(model),
        str(trunk),
        str(experts),
        str(globals_),
        "--token",
        "0",
        "--json",
        *extra,
    ]
    if logits_path is not None:
        command.extend(("--dump-logits", str(logits_path)))
    return subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8", timeout=3_600
    )


def _read_logits(path: Path) -> array.array[float]:
    values: array.array[float] = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def probe_gpu_residency(args: argparse.Namespace) -> int:
    """Device residency with a bounded CPU fallback, compared against the CPU reference."""
    contract = "0731 GPU expert residency contract"
    if sys.byteorder != "little":
        raise AssertionError(f"{contract} requires a little-endian host for raw float32 dumps")
    required = {"artifact": args.artifact, "experts": args.experts, "globals": args.globals}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"{contract} requires --{', --'.join(missing)}")
    model = args.model.resolve(strict=True)
    trunk = args.artifact.resolve(strict=True)
    experts = args.experts.resolve(strict=True)
    globals_ = args.globals.resolve(strict=True)

    info = subprocess.run(
        [str(args.vastloom), "cuda-info", "--json"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    if info.returncode != 0:
        print(f"{contract}: skipped, no usable CUDA driver on this host", file=sys.stderr)
        return 75

    with tempfile.TemporaryDirectory(prefix="vastloom-gpu-residency-") as directory:
        root = Path(directory)
        cpu_logits = root / "cpu.f32"
        gpu_logits = root / "gpu.f32"
        fallback_logits = root / "fallback.f32"

        cpu = _probe(args.vastloom, model, trunk, experts, globals_, logits_path=cpu_logits)
        if cpu.returncode != 0:
            pressure = host_pressure_reason(cpu)
            if pressure is not None:
                return skipped(contract, pressure)
            raise AssertionError(f"{contract} CPU reference failed\nstderr={cpu.stderr}")
        gpu = _probe(
            args.vastloom,
            model,
            trunk,
            experts,
            globals_,
            extra=("--cuda-experts", "--gpu-limit", "16"),
            logits_path=gpu_logits,
        )
        if gpu.returncode != 0:
            raise AssertionError(f"{contract} GPU probe failed\nstderr={gpu.stderr}")
        fallback = _probe(
            args.vastloom,
            model,
            trunk,
            experts,
            globals_,
            extra=("--cuda-experts", "--gpu-limit", "1"),
            logits_path=fallback_logits,
        )
        if fallback.returncode != 0:
            raise AssertionError(f"{contract} fallback probe failed\nstderr={fallback.stderr}")

        cpu_payload = json.loads(cpu.stdout)
        gpu_payload = json.loads(gpu.stdout)
        fallback_payload = json.loads(fallback.stdout)
        if "accelerator" in cpu_payload:
            raise AssertionError(f"{contract}: the default CPU path must not report an accelerator")

        reference = _read_logits(cpu_logits)
        if not reference:
            raise AssertionError(f"{contract}: the CPU reference produced no logits")
        for name, payload, path in (
            ("GPU", gpu_payload, gpu_logits),
            ("fallback", fallback_payload, fallback_logits),
        ):
            values = _read_logits(path)
            if len(values) != len(reference):
                raise AssertionError(f"{contract}: {name} logits length changed")
            if payload["argmax_token"] != cpu_payload["argmax_token"]:
                raise AssertionError(f"{contract}: {name} argmax changed")
            difference = max(abs(left - right) for left, right in zip(values, reference))
            if difference > 1.0e-3:
                raise AssertionError(
                    f"{contract}: {name} logits differ from the CPU reference by {difference}"
                )

        gpu_stats = gpu_payload.get("accelerator")
        if not isinstance(gpu_stats, dict):
            raise AssertionError(f"{contract}: GPU run did not report accelerator statistics")
        if gpu_stats["accelerated"] <= 0:
            raise AssertionError(f"{contract}: no matrix ran on the device")
        if gpu_stats["declined_recurring"] <= 0:
            raise AssertionError(f"{contract}: --cuda-experts did not keep the trunk on the CPU")
        if gpu_stats["resident_bytes"] <= 0 or gpu_stats["evicted"] <= 0:
            raise AssertionError(f"{contract}: the capped VRAM ceiling did not evict and re-fill")
        if gpu_stats["kernel_ms"] <= 0.0 or gpu_stats["uploaded_bytes"] <= 0:
            raise AssertionError(f"{contract}: the device tier reported no kernel or upload")

        fallback_stats = fallback_payload.get("accelerator")
        if not isinstance(fallback_stats, dict):
            raise AssertionError(f"{contract}: fallback run did not report accelerator statistics")
        if fallback_stats["accelerated"] != 0 or fallback_stats["resident_matrices"] != 0:
            raise AssertionError(f"{contract}: a 1 MiB ceiling still admitted device matrices")
        if fallback_stats["declined_recurring"] <= 0:
            raise AssertionError(f"{contract}: the fallback run did not see the recurring matrices")

        print(
            f"{contract}: PASS (accelerated={gpu_stats['accelerated']}, "
            f"evicted={gpu_stats['evicted']}, resident={gpu_stats['resident_bytes']} B, "
            f"fallback logits identical within 1e-3)"
        )
    return 0



def resolve(payload: object, path: str) -> object:
    """Follow a dotted path into the parsed document."""
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        help="imported trunk range artifact; required by the route, trunk-load and layer contracts",
    )
    parser.add_argument("--experts", type=Path, help="imported selected-expert range artifact")
    parser.add_argument("--globals", type=Path, help="imported global range artifact")
    parser.add_argument(
        "--command", choices=[*sorted(COMMANDS), "layer-probe-gpu"], required=True
    )
    args = parser.parse_args()

    if args.command == "layer-probe-gpu":
        return probe_gpu_residency(args)

    contract, template, expected = COMMANDS[args.command]
    substitutions = {"model": str(args.model.resolve(strict=True))}
    for name, value in (
        ("artifact", args.artifact),
        ("experts", args.experts),
        ("globals", args.globals),
    ):
        if not any("{" + name + "}" in part for part in template):
            continue
        if value is None:
            raise SystemExit(f"{contract} requires --{name}")
        substitutions[name] = str(value.resolve(strict=True))
    arguments = [
        str(args.vastloom),
        *(part.format(**{key: substitutions.get(key, "") for key in
                         ("model", "artifact", "experts", "globals")}) for part in template),
    ]
    process = subprocess.run(
        arguments, check=False, capture_output=True, text=True, encoding="utf-8", timeout=3_600
    )

    pressure = host_pressure_reason(process)
    if pressure is not None:
        return skipped(contract, pressure)
    if process.returncode != 0:
        raise AssertionError(
            f"{contract} command failed\nexit={process.returncode}\nstderr={process.stderr}"
        )

    payload = json.loads(process.stdout)
    mismatches = [
        f"  {path}: expected {value!r}, got {resolve(payload, path)!r}"
        for path, value in expected.items()
        if resolve(payload, path) != value
    ]
    if mismatches:
        raise AssertionError(f"{contract} changed:\n" + "\n".join(mismatches))

    print(f"{contract}: PASS ({len(expected)} pinned values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
