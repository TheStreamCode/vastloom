from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "generate_manifest.py"


class ManifestToolTests(unittest.TestCase):
    def test_generates_atomic_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            payload = b"vastloom"
            (model / "weights.bin").write_bytes(payload)
            output = model / "model.sgm"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    str(model),
                    "--output",
                    str(output),
                    "--model-id",
                    "test/model",
                    "--architecture",
                    "test_moe",
                    "--quiet",
                ],
                check=True,
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("format=vastloom.manifest.v1", text)
            self.assertIn("model_total_bytes=8", text)
            self.assertIn(hashlib.sha256(payload).hexdigest(), text)
            self.assertFalse(any(output.parent.glob(f".{output.name}.*.tmp")))

    def test_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            target = root / "outside.bin"
            target.write_bytes(b"outside")
            link = model / "link.bin"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("Symlink creation is unavailable on this platform")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    str(model),
                    "--output",
                    str(model / "model.sgm"),
                    "--model-id",
                    "test/model",
                    "--architecture",
                    "test_moe",
                    "--quiet",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Refusing file symlink", completed.stderr)


if __name__ == "__main__":
    unittest.main()
