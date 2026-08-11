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
import json
from pathlib import Path
import subprocess

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
    "tensor_bytes": 82_533_892_444,
    "always_active_bytes": 6_986_088_796,
    "routed_expert_bytes": 75_547_803_648,
    "layer_streaming.planned_layers": 43,
}

PROBE_PLAN_CONTRACT = {
    "plan_only": True,
    "routing_authority_verified": False,
    "trunk_bytes": 138_867_160,
    "selected_expert_bytes": 38_928_384,
    "total_payload_bytes": 177_795_544,
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

ROUTE_CONTRACT = {
    "layer": 0,
    "expected_top_k": 6,
    "selection_source": "official-tid2eid-verified",
    "routing_authority_verified": True,
    "routing_token": 0,
    "selected_experts": [35, 53, 200, 222, 245, 254],
    "trunk_bytes": 138_867_160,
    "selected_expert_bytes": 38_928_384,
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
}


MISSING = "<missing>"


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
        help="imported trunk range artifact; required by the route and trunk-load contracts",
    )
    parser.add_argument("--command", choices=sorted(COMMANDS), required=True)
    args = parser.parse_args()

    contract, template, expected = COMMANDS[args.command]
    model = str(args.model.resolve(strict=True))
    artifact = ""
    if any("{artifact}" in part for part in template):
        if args.artifact is None:
            raise SystemExit(f"{contract} requires --artifact")
        artifact = str(args.artifact.resolve(strict=True))
    arguments = [
        str(args.vastloom),
        *(part.format(model=model, artifact=artifact) for part in template),
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
