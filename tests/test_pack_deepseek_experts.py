#!/usr/bin/env python3
"""Tests for tools/pack_deepseek_experts.py (stdlib-only, no model weights)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "pack_deepseek_experts.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_expert(expert: int) -> dict:
    tensors = []
    packed_offset = 0
    for index, suffix in enumerate(("down", "gate", "up")):
        tensors.append({
            "tensor": f"blk.0.ffn_{suffix}_exps.weight.expert.{expert}",
            "shape": [1, 4],
            "packed_offset": packed_offset,
            "bytes": 4,
            "ggml_type": 18,
            "source": {"object": "shard.gguf", "offset": expert * 100 + index * 4, "bytes": 4},
        })
        packed_offset += 4
    return {"expert": expert, "bytes": packed_offset, "tensors": tensors,
            "read_ranges": [{"object": "shard.gguf", "offset": expert * 100 + index * 4,
                             "bytes": 4} for index in range(3)]}


def make_plan(layer: int, experts: list[int]) -> dict:
    return {
        "format": "vastloom.deepseek_v4.layer_probe_plan.v1",
        "layer": layer,
        "experts": [make_expert(expert) for expert in experts],
    }


def make_artifact(root: Path, layer: int, experts: list[int],
                  name: str = "artifact") -> Path:
    artifact = root / name
    ranges_dir = artifact / "ranges"
    ranges_dir.mkdir(parents=True)
    ranges = []
    payload_bytes = 0
    index = 0
    for expert in experts:
        for range_index in range(3):
            data = bytes([(expert + range_index + index) % 256] * 4)
            name = f"ranges/{index:04d}.bin"
            (artifact / name).write_bytes(data)
            ranges.append({
                "role": "selected-expert",
                "label": f"expert[{index}:{expert}].range[{range_index}]",
                "artifact": name,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "source_object": "shard.gguf",
                "source_path": "shard.gguf",
                "source_object_bytes": 1_000_000,
                "source_object_sha256": "0" * 64,
                "source_offset": expert * 100 + range_index * 4,
                "source_bytes": len(data),
            })
            payload_bytes += len(data)
            index += 1
    write_json(artifact / "vastloom-range-artifact.json", {
        "format": "vastloom.range-artifact.v1",
        "model_id": "test/model",
        "revision": "rev1",
        "scope": "selected-experts",
        "layer": layer,
        "routing_authority_verified": True,
        "payload_bytes": payload_bytes,
        "range_count": len(ranges),
        "source_plan_sha256": "1" * 64,
        "source_plan_format": "vastloom.deepseek_v4.layer_probe_plan.v1",
        "verification": "pinned-revision-content-range-plus-local-sha256",
        "complete_checkpoint_verified": False,
        "execution_supported": False,
        "ranges": ranges,
    })
    return artifact


class PackToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vastloom-pack-test-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_tool(self, layer: int, output: Path, plan: Path, artifact: Path,
                 *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL), "--layer", str(layer), "--output", str(output),
             "--plan", str(plan), "--artifact", str(artifact), *extra],
            capture_output=True, text=True, check=False,
        )

    def test_pack_records_are_aligned_and_hashed(self) -> None:
        plan_path = self.root / "plan.json"
        write_json(plan_path, make_plan(0, [5, 9]))
        artifact = make_artifact(self.root, 0, [5, 9])
        output = self.root / "pack"
        result = self.run_tool(0, output, plan_path, artifact)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "vastloom-expert-pack.json").read_text())
        self.assertEqual(manifest["format"], "vastloom.deepseek-v4-expert-pack.v1")
        self.assertEqual(manifest["layer"], 0)
        self.assertEqual(len(manifest["experts"]), 2)
        self.assertEqual([entry["expert"] for entry in manifest["experts"]], [5, 9])
        payload = (output / "experts-L0.bin").read_bytes()
        self.assertEqual(len(payload), manifest["payload"]["bytes"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["payload"]["sha256"])
        for entry in manifest["experts"]:
            self.assertEqual(entry["offset"] % 4096, 0)
            record = payload[entry["offset"]:entry["offset"] + entry["bytes"]]
            self.assertEqual(hashlib.sha256(record).hexdigest(), entry["sha256"])
            self.assertEqual([tensor["name"] for tensor in entry["tensors"]],
                             [tensor["tensor"] for tensor in
                              make_expert(entry["expert"])["tensors"]])

    def test_duplicate_expert_across_plans_is_merged(self) -> None:
        plan_a = self.root / "plan-a.json"
        plan_b = self.root / "plan-b.json"
        write_json(plan_a, make_plan(0, [7]))
        write_json(plan_b, make_plan(0, [7]))
        artifact_a = make_artifact(self.root, 0, [7])
        artifact_b = make_artifact(self.root, 0, [7], name="artifact-b")
        output = self.root / "pack"
        result = self.run_tool(0, output, plan_a, artifact_a,
                               "--plan", str(plan_b), "--artifact", str(artifact_b))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "vastloom-expert-pack.json").read_text())
        self.assertEqual(len(manifest["experts"]), 1)

    def test_mismatched_plan_and_artifact_are_refused(self) -> None:
        plan_path = self.root / "plan.json"
        write_json(plan_path, make_plan(0, [5]))
        artifact = make_artifact(self.root, 0, [6])
        result = self.run_tool(0, self.root / "pack", plan_path, artifact)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_routing_authority_is_refused(self) -> None:
        plan_path = self.root / "plan.json"
        write_json(plan_path, make_plan(0, [5]))
        artifact = make_artifact(self.root, 0, [5])
        manifest_path = artifact / "vastloom-range-artifact.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["routing_authority_verified"] = False
        write_json(manifest_path, manifest)
        result = self.run_tool(0, self.root / "pack", plan_path, artifact)
        self.assertNotEqual(result.returncode, 0)

    def test_existing_output_requires_force(self) -> None:
        plan_path = self.root / "plan.json"
        write_json(plan_path, make_plan(0, [5]))
        artifact = make_artifact(self.root, 0, [5])
        output = self.root / "pack"
        output.mkdir()
        (output / "sentinel").write_text("keep")
        result = self.run_tool(0, output, plan_path, artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((output / "sentinel").exists())
        forced = self.run_tool(0, output, plan_path, artifact, "--force")
        self.assertEqual(forced.returncode, 0, forced.stderr)
        self.assertFalse((output / "sentinel").exists())


if __name__ == "__main__":
    unittest.main()
