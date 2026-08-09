#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading


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
    with tempfile.TemporaryDirectory(prefix="vastloom-import-test-") as temporary:
        root = Path(temporary)
        (root / "models" / "original").mkdir(parents=True)
        (root / "cache").mkdir()
        (root / ".vastloom-store").write_text(MODULE.MARKER, encoding="utf-8")
        require(MODULE.validate_store(root) == root.resolve(), "valid store rejected")

        legacy_root = root / "legacy-store"
        (legacy_root / "models" / "original").mkdir(parents=True)
        (legacy_root / "cache").mkdir()
        (legacy_root / ".stargate-store").write_text(MODULE.LEGACY_MARKER, encoding="utf-8")
        require(MODULE.validate_store(legacy_root) == legacy_root.resolve(), "legacy store rejected")
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
        require(len(MODULE.selected_files(MODULE.OLMOE, True)) == 6, "OLMoE metadata set is incomplete")
        require(len(MODULE.selected_files(MODULE.OLMOE, False)) == 9, "OLMoE full file set is incomplete")
        require(
            sum(item.size for item in MODULE.selected_files(MODULE.OLMOE, False) if item.weight)
            == 13_838_721_960,
            "OLMoE shard byte count changed",
        )
        require(
            all(item.sha256 for item in MODULE.OLMOE.files),
            "OLMoE files must all have pinned SHA-256 values",
        )
        deepseek_metadata = MODULE.selected_files(MODULE.DEEPSEEK_V4_FLASH, True)
        deepseek_full = MODULE.selected_files(MODULE.DEEPSEEK_V4_FLASH, False)
        require(len(deepseek_metadata) == 51, "DeepSeek metadata/header set is incomplete")
        require(len(deepseek_full) == 51, "DeepSeek full file set is incomplete")
        require(
            sum(item.size for item in deepseek_full if item.weight) == 159_617_149_040,
            "DeepSeek shard byte count changed",
        )
        require(
            sum(item.size for item in deepseek_metadata if item.range_prefix) == 7_663_144,
            "DeepSeek bounded header byte count changed",
        )
        require(
            all(item.sha256 for item in MODULE.DEEPSEEK_V4_FLASH.files),
            "DeepSeek files and header snapshots must all have pinned SHA-256 values",
        )
        require(
            all(
                item.metadata_only_only and item.source_name and item.source_size
                for item in deepseek_metadata
                if item.range_prefix
            ),
            "DeepSeek header snapshot identity is incomplete",
        )
        try:
            MODULE.import_model(
                root,
                MODULE.DEEPSEEK_V4_FLASH,
                metadata_only=False,
                reserve_bytes=20 * MODULE.GIB,
                timeout=1.0,
                dry_run=True,
            )
        except RuntimeError as error:
            require("--allow-large-download" in str(error), "large-import refusal is not actionable")
        else:
            raise AssertionError("DeepSeek full import did not require explicit large-download authorization")

        invalid = root / "invalid"
        invalid.mkdir()
        try:
            MODULE.validate_store(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid store marker was accepted")

        symlink_store = root / "symlink-store"
        (symlink_store / "models" / "original").mkdir(parents=True)
        (symlink_store / "cache").mkdir()
        try:
            (symlink_store / ".vastloom-store").symlink_to(root / ".vastloom-store")
        except OSError:
            pass
        else:
            try:
                MODULE.validate_store(symlink_store)
            except RuntimeError:
                pass
            else:
                raise AssertionError("symlinked store marker was accepted")

        existing = root / "models" / "original" / "existing"
        existing.mkdir()
        (existing / "vastloom-import.json").write_text(json.dumps({}), encoding="utf-8")
        try:
            MODULE.validate_existing(existing, MODULE.SMOLLM2, (), True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("mismatched immutable artifact identity was accepted")

        legacy_existing = root / "models" / "original" / "legacy-existing"
        legacy_existing.mkdir()
        legacy_payload = b"legacy"
        legacy_spec = MODULE.ModelSpec(
            "legacy",
            "test/legacy",
            "revision",
            (MODULE.FileSpec("data.bin", len(legacy_payload), hashlib.sha256(legacy_payload).hexdigest()),),
        )
        (legacy_existing / "data.bin").write_bytes(legacy_payload)
        (legacy_existing / "stargate-import.json").write_text(
            json.dumps(
                {
                    "format": MODULE.LEGACY_IMPORT_FORMAT,
                    "model_id": legacy_spec.model_id,
                    "revision": legacy_spec.revision,
                    "metadata_only": False,
                    "files": [
                        {
                            "name": "data.bin",
                            "size": len(legacy_payload),
                            "sha256": hashlib.sha256(legacy_payload).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        MODULE.validate_existing(legacy_existing, legacy_spec, legacy_spec.files, False)

        payload = bytes(range(256)) * 8

        class RangeHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - standard-library handler API
                start = 0
                requested = self.headers.get("Range")
                if requested:
                    require(requested.startswith("bytes=") and requested.endswith("-"), "invalid test Range header")
                    start = int(requested[6:-1])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(payload) - start))
                self.end_headers()
                self.wfile.write(payload[start:])

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            partial = root / "resume.partial"
            partial.write_bytes(payload[:333])
            expected = hashlib.sha256(payload).hexdigest()
            actual = MODULE.download_file(
                f"http://127.0.0.1:{server.server_port}/weights",
                partial,
                MODULE.FileSpec("weights", len(payload), expected, True),
                5.0,
                1024 * 1024,
            )
            require(actual == expected and partial.read_bytes() == payload, "bounded Range resume failed")
        finally:
            server.shutdown()
            worker.join(timeout=5)

        class PrefixRangeHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - standard-library handler API
                requested = self.headers.get("Range")
                require(requested is not None and requested.startswith("bytes="), "missing prefix Range header")
                start_text, end_text = requested[6:].split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                require(0 <= start <= end < len(payload), "invalid bounded prefix range")
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                self.wfile.write(payload[start : end + 1])

            def log_message(self, _format: str, *_args: object) -> None:
                return

        prefix_server = ThreadingHTTPServer(("127.0.0.1", 0), PrefixRangeHandler)
        prefix_worker = threading.Thread(target=prefix_server.serve_forever, daemon=True)
        prefix_worker.start()
        try:
            prefix = payload[:257]
            partial = root / "header.partial"
            partial.write_bytes(prefix[:73])
            item = MODULE.FileSpec(
                "headers/test.header",
                len(prefix),
                hashlib.sha256(prefix).hexdigest(),
                source_name="weights",
                metadata_only_only=True,
                range_prefix=True,
                source_size=len(payload),
            )
            actual = MODULE.download_file(
                f"http://127.0.0.1:{prefix_server.server_port}/weights",
                partial,
                item,
                5.0,
                1024 * 1024,
            )
            require(actual == item.sha256 and partial.read_bytes() == prefix, "bounded header snapshot failed")
        finally:
            prefix_server.shutdown()
            prefix_worker.join(timeout=5)

    print("Hugging Face importer offline tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
