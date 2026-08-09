#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_hf_model", ROOT / "tools" / "import_hf_model.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stargate-import-test-") as temporary:
        root = Path(temporary)
        (root / "models" / "original").mkdir(parents=True)
        (root / ".stargate-store").write_text(MODULE.MARKER, encoding="utf-8")
        require(MODULE.validate_store(root) == root.resolve(), "valid store rejected")
        target = MODULE.import_model(
            root,
            MODULE.SMOLLM2,
            metadata_only=True,
            reserve_bytes=20 * MODULE.GIB,
            timeout=1.0,
            dry_run=True,
        )
        require(target.name.endswith("--metadata"), "metadata target is not isolated")
        require(not target.exists(), "dry-run created an artifact")
        require(len(MODULE.selected_files(MODULE.SMOLLM2, True)) == 7, "weight was not excluded")
        require(len(MODULE.selected_files(MODULE.SMOLLM2, False)) == 8, "full file set is incomplete")

        invalid = root / "invalid"
        invalid.mkdir()
        try:
            MODULE.validate_store(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid store marker was accepted")

        existing = root / "models" / "original" / "existing"
        existing.mkdir()
        (existing / "stargate-import.json").write_text(json.dumps({}), encoding="utf-8")
        try:
            MODULE.validate_existing(existing, MODULE.SMOLLM2, (), True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("mismatched immutable artifact identity was accepted")

    print("Hugging Face importer offline tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
