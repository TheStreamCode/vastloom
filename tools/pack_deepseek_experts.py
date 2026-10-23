#!/usr/bin/env python3
"""Pack verified DeepSeek selected-expert range artifacts into aligned expert records.

The existing range artifacts store one file per source range (three per expert). This tool builds a
derived pack whose payload has exactly one contiguous, 4 KiB-aligned record per expert, in the same
tensor order the catalog bundle already uses. The pack keeps the original source plan and artifact
manifest identities, so it stays a derived artifact rather than a new checkpoint.

Inputs are the outputs already produced by ``deepseek-gguf-probe-plan`` and
``import_hf_ranges.py``: one or more plan/artifact pairs, for example the six experts selected by
several prompt tokens. Pass ``--plan`` and ``--artifact`` the same number of times, in order.

Usage:
    python tools/pack_deepseek_experts.py --layer 0 --output build/packs/L0 \
        --plan layer0-token0-plan.json --artifact build/store/...--l0-experts-... \
        --plan layer0-token1-plan.json --artifact build/store/...--l0-experts-...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

FORMAT = "vastloom.deepseek-v4-expert-pack.v1"
MANIFEST_NAME = "vastloom-expert-pack.json"
PAYLOAD_NAME = "experts-L{layer}.bin"
ALIGNMENT = 4096
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_RECORDS = 256
RANGE_LABEL = re.compile(r"^expert\[(\d+):(\d+)\]\.range\[(\d+)\]$")


class PackError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackError(f"cannot read {path}: {error}") from error


def safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise PackError(f"artifact range escapes its root: {relative}") from error
    return candidate


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackError(message)


def load_artifact(path: Path, layer: int) -> tuple[dict, dict[int, list[dict]]]:
    manifest_path = path / "vastloom-range-artifact.json"
    manifest = load_json(manifest_path)
    require(manifest.get("format") == "vastloom.range-artifact.v1",
            f"{path}: not a range artifact v1 manifest")
    require(manifest.get("scope") == "selected-experts",
            f"{path}: scope is {manifest.get('scope')!r}, expected selected-experts")
    require(manifest.get("layer") == layer,
            f"{path}: layer {manifest.get('layer')} does not match --layer {layer}")
    require(manifest.get("routing_authority_verified") is True,
            f"{path}: artifact does not carry verified routing authority")
    ranges = manifest.get("ranges")
    require(isinstance(ranges, list) and ranges, f"{path}: artifact has no ranges")
    by_expert: dict[int, list[dict]] = {}
    for entry in ranges:
        label = entry.get("label")
        match = RANGE_LABEL.match(label or "")
        require(match is not None, f"{path}: unexpected range label {label!r}")
        expert = int(match.group(2))
        range_index = int(match.group(3))
        local = safe_child(path, str(entry.get("artifact", "")))
        expected_bytes = entry.get("bytes")
        require(isinstance(expected_bytes, int) and expected_bytes > 0,
                f"{path}: range {label} has no byte count")
        require(local.is_file(), f"{path}: missing payload {local}")
        require(local.stat().st_size == expected_bytes,
                f"{path}: payload size mismatch for {local}")
        actual = sha256_file(local)
        require(actual == entry.get("sha256"),
                f"{path}: payload hash mismatch for {label}")
        by_expert.setdefault(expert, []).append({
            "range_index": range_index,
            "path": local,
            "bytes": expected_bytes,
            "sha256": actual,
            "source_object": entry.get("source_object"),
            "source_offset": entry.get("source_offset"),
            "source_bytes": entry.get("source_bytes"),
        })
    for expert, entries in by_expert.items():
        entries.sort(key=lambda item: item["range_index"])
        require([item["range_index"] for item in entries] == list(range(len(entries))),
                f"{path}: expert {expert} ranges are not contiguous")
    return manifest, by_expert


def build_records(
    plans: list[Path], artifacts: list[Path], layer: int
) -> tuple[list[dict], str, str, list[dict]]:
    require(len(plans) == len(artifacts) and plans,
            "--plan and --artifact must be supplied in matching positive counts")
    model_id = None
    revision = None
    sources: list[dict] = []
    experts: dict[int, dict] = {}

    for plan_path, artifact_path in zip(plans, artifacts):
        plan = load_json(plan_path)
        plan_layer = plan.get("layer")
        require(plan_layer == layer, f"{plan_path}: layer does not match --layer {layer}")
        require(plan.get("format") == "vastloom.deepseek_v4.layer_probe_plan.v1",
                f"{plan_path}: unsupported plan format {plan.get('format')!r}")
        plan_experts = plan.get("experts")
        require(isinstance(plan_experts, list) and plan_experts,
                f"{plan_path}: plan has no experts")

        manifest, artifact_experts = load_artifact(artifact_path, layer)
        if model_id is None:
            model_id = manifest.get("model_id")
            revision = manifest.get("revision")
        require(manifest.get("model_id") == model_id,
                f"{artifact_path}: model identity differs from the first artifact")
        require(manifest.get("revision") == revision,
                f"{artifact_path}: revision differs from the first artifact")
        sources.append({
            "path": str(artifact_path),
            "manifest_sha256": sha256_file(artifact_path / "vastloom-range-artifact.json"),
            "source_plan_sha256": manifest.get("source_plan_sha256"),
            "scope": manifest.get("scope"),
        })

        for plan_expert in plan_experts:
            expert = plan_expert.get("expert")
            tensors = plan_expert.get("tensors")
            require(isinstance(expert, int) and isinstance(tensors, list) and tensors,
                    f"{plan_path}: malformed expert record")
            ranges = artifact_experts.get(expert)
            require(ranges is not None and len(ranges) == len(tensors),
                    f"{artifact_path}: expert {expert} has "
                    f"{0 if ranges is None else len(ranges)} ranges, expected {len(tensors)}")
            expected = []
            for tensor, source_range in zip(tensors, ranges):
                require(tensor.get("bytes") == source_range["bytes"],
                        f"{plan_path}: expert {expert} tensor bytes do not match the artifact")
                expected.append({
                    "name": tensor.get("tensor"),
                    "packed_offset": tensor.get("packed_offset"),
                    "bytes": tensor.get("bytes"),
                    "ggml_type": tensor.get("ggml_type"),
                    "shape": tensor.get("shape"),
                })
            packed_bytes = sum(item["bytes"] for item in expected)
            require(packed_bytes == plan_expert.get("bytes"),
                    f"{plan_path}: expert {expert} byte total does not match the plan")
            record = {
                "expert": expert,
                "bytes": packed_bytes,
                "tensors": expected,
                "ranges": ranges,
                "source_sha256": [item["sha256"] for item in ranges],
            }
            previous = experts.get(expert)
            if previous is not None:
                require(previous["source_sha256"] == record["source_sha256"] and
                        previous["tensors"] == record["tensors"],
                        f"{artifact_path}: expert {expert} appears with different bytes")
                continue
            experts[expert] = record

    require(experts, "no experts were selected")
    ordered = [experts[key] for key in sorted(experts)]
    require(len(ordered) <= MAX_RECORDS, "too many expert records")
    records = []
    for record in ordered:
        records.append({
            "expert": record["expert"],
            "bytes": record["bytes"],
            "tensors": record["tensors"],
            "ranges": record["ranges"],
            "source_range_sha256": record["source_sha256"],
        })
    return records, model_id or "", revision or "", sources


def write_pack(output: Path, layer: int, records: list[dict], model_id: str,
               revision: str, sources: list[dict], force: bool) -> dict:
    if output.exists() and not force:
        raise PackError(f"{output} already exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".expert-pack-", dir=str(output.parent)))
    try:
        payload_path = temp_root / PAYLOAD_NAME.format(layer=layer)
        payload_bytes = 0
        packed_records = []
        with payload_path.open("wb") as payload:
            for record in records:
                offset = payload.tell()
                require(offset % ALIGNMENT == 0, "internal alignment error")
                digest = hashlib.sha256()
                for source_range in record["ranges"]:
                    with source_range["path"].open("rb") as source:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                            payload.write(block)
                            digest.update(block)
                require(payload.tell() - offset == record["bytes"],
                        f"expert {record['expert']}: record byte count changed while packing")
                padding = (-payload.tell()) % ALIGNMENT
                if padding:
                    payload.write(b"\0" * padding)
                packed_records.append({
                    "expert": record["expert"],
                    "offset": offset,
                    "bytes": record["bytes"],
                    "sha256": digest.hexdigest(),
                    "source_range_sha256": record["source_range_sha256"],
                    "tensors": record["tensors"],
                })
                payload_bytes = payload.tell()
        require(payload_bytes <= MAX_PAYLOAD_BYTES, "pack payload exceeds its safety bound")
        payload_hash = sha256_file(payload_path)
        manifest = {
            "format": FORMAT,
            "model_id": model_id,
            "revision": revision,
            "layer": layer,
            "routing_authority_verified": True,
            "alignment": ALIGNMENT,
            "payload": {
                "path": payload_path.name,
                "bytes": payload_bytes,
                "sha256": payload_hash,
            },
            "source_artifacts": sources,
            "experts": packed_records,
        }
        manifest_path = temp_root / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        os.replace(temp_root, output)
        return manifest
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="append", type=Path, default=[])
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        require(args.layer >= 0, "layer must be non-negative")
        records, model_id, revision, sources = build_records(
            [path.resolve() for path in args.plan],
            [path.resolve() for path in args.artifact],
            args.layer,
        )
        manifest = write_pack(args.output.resolve(), args.layer, records, model_id,
                              revision, sources, args.force)
    except PackError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: filesystem failure: {error}", file=sys.stderr)
        return 2

    summary = {
        "ok": True,
        "format": manifest["format"],
        "layer": manifest["layer"],
        "records": len(manifest["experts"]),
        "payload_bytes": manifest["payload"]["bytes"],
        "payload_sha256": manifest["payload"]["sha256"],
        "output": str(args.output.resolve()),
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"Expert pack layer {manifest['layer']}: {summary['records']} records, "
              f"{summary['payload_bytes']} bytes -> {summary['output']}")
        print(f"payload sha256 {summary['payload_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
