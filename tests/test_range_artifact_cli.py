#!/usr/bin/env python3
"""Offline CLI checks for the bounded native range-artifact verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


VASTLOOM: Path


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(VASTLOOM), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


class RangeArtifactCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="vastloom-range-cli-")
        self.root = Path(self.temporary.name) / "artifact"
        (self.root / "ranges").mkdir(parents=True)
        self.payload = b"bounded-range"
        (self.root / "ranges" / "0000.bin").write_bytes(self.payload)
        self.write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self) -> None:
        manifest = {
            "complete_checkpoint_verified": False,
            "execution_supported": False,
            "format": "vastloom.range-artifact.v1",
            "layer": 7,
            "model_id": "test/range-cli",
            "payload_bytes": len(self.payload),
            "range_count": 1,
            "ranges": [
                {
                    "artifact": "ranges/0000.bin",
                    "bytes": len(self.payload),
                    "label": "trunk[0] blk.7.attn.weight",
                    "role": "trunk",
                    "sha256": hashlib.sha256(self.payload).hexdigest(),
                    "source_bytes": len(self.payload),
                    "source_object": "model.gguf",
                    "source_object_bytes": 1024,
                    "source_object_sha256": "1" * 64,
                    "source_offset": 128,
                    "source_path": "weights/model.gguf",
                }
            ],
            "revision": "fixture",
            "routing_authority_verified": False,
            "scope": "trunk",
            "source_plan_format": "vastloom.deepseek_v4.layer_probe_plan.v1",
            "source_plan_sha256": "2" * 64,
            "verification": "pinned-revision-content-range-plus-local-sha256",
        }
        (self.root / "vastloom-range-artifact.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

    def test_verified_json_and_quick_mode(self) -> None:
        result = run_cli("range-artifact-inspect", str(self.root), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertTrue(document["local_range_hashes_verified"])
        self.assertFalse(document["complete_checkpoint_verified"])
        self.assertFalse(document["execution_supported"])
        self.assertEqual(document["range_count"], 1)

        (self.root / "ranges" / "0000.bin").write_bytes(b"corrupt-range")
        refused = run_cli("range-artifact-inspect", str(self.root), "--json")
        self.assertEqual(refused.returncode, 65)
        self.assertFalse(json.loads(refused.stderr)["ok"])
        quick = run_cli("range-artifact-inspect", str(self.root), "--no-verify", "--json")
        self.assertEqual(quick.returncode, 0, quick.stderr)
        self.assertFalse(json.loads(quick.stdout)["local_range_hashes_verified"])

    def test_closed_layout_and_option_rejection(self) -> None:
        (self.root / "ranges" / "extra.bin").write_bytes(b"x")
        result = run_cli("range-artifact-inspect", str(self.root), "--json")
        self.assertEqual(result.returncode, 65)
        (self.root / "ranges" / "extra.bin").unlink()

        unknown = run_cli("range-artifact-inspect", str(self.root), "--unknown")
        self.assertEqual(unknown.returncode, 64)
        duplicate = run_cli(
            "range-artifact-inspect", str(self.root), "--json", "--json"
        )
        self.assertEqual(duplicate.returncode, 64)
        missing = run_cli("range-artifact-inspect")
        self.assertEqual(missing.returncode, 64)

    def test_deepseek_bind_has_no_hash_bypass(self) -> None:
        missing = run_cli("deepseek-gguf-range-bind")
        self.assertEqual(missing.returncode, 64)
        bypass = run_cli(
            "deepseek-gguf-range-bind",
            str(self.root),
            str(self.root),
            "--layer",
            "0",
            "--experts",
            "0",
            "--no-verify",
            "--json",
        )
        self.assertEqual(bypass.returncode, 64)


def main() -> int:
    global VASTLOOM
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    arguments, unittest_arguments = parser.parse_known_args()
    VASTLOOM = arguments.vastloom.resolve()
    unittest.main(argv=[__file__, *unittest_arguments])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
