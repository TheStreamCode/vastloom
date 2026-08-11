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
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_hf_model", ROOT / "tools" / "import_hf_model.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RANGE_SPEC = importlib.util.spec_from_file_location(
    "import_hf_ranges", ROOT / "tools" / "import_hf_ranges.py"
)
assert RANGE_SPEC and RANGE_SPEC.loader
RANGE_MODULE = importlib.util.module_from_spec(RANGE_SPEC)
sys.modules[RANGE_SPEC.name] = RANGE_MODULE
RANGE_SPEC.loader.exec_module(RANGE_MODULE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def range_plan_document(
    model_id: str,
    revision: str,
    *,
    routing_authority_verified: bool = False,
) -> dict[str, object]:
    source = "test.gguf"
    return {
        "ok": True,
        "format": RANGE_MODULE.PLAN_FORMAT,
        "model_id": model_id,
        "revision": revision,
        "plan_only": True,
        "execution_supported": False,
        "layer": 0,
        "expected_top_k": 2,
        "selection_source": (
            "official-tid2eid-verified"
            if routing_authority_verified
            else "caller-supplied-unverified"
        ),
        "routing_authority_verified": routing_authority_verified,
        "payload_access": {
            "metadata_only": True,
            "imported_file_hashes_verified": True,
            "full_payload_present": False,
            "executable_source_binding_verified": False,
        },
        "tile_limit_bytes": 64 * 1024 * 1024,
        "trunk_bytes": 12,
        "selected_expert_bytes": 7,
        "total_payload_bytes": 19,
        "maximum_read_range_bytes": 7,
        "minimum_expert_tier_bytes": 4,
        "trunk_range_count": 2,
        "selected_expert_range_count": 2,
        "total_range_count": 4,
        "unique_source_objects": 1,
        "selected_experts": [0, 1],
        "trunk_ranges": [
            {
                "tensor": "blk.0.attn_q.weight",
                "role": "attention",
                "tensor_offset": 0,
                "final_tile": True,
                "ggml_type": 8,
                "source": {"object": source, "offset": 4, "bytes": 5},
            },
            {
                "tensor": "blk.0.attn_k.weight",
                "role": "attention",
                "tensor_offset": 0,
                "final_tile": True,
                "ggml_type": 8,
                "source": {"object": source, "offset": 20, "bytes": 7},
            },
        ],
        "experts": [
            {
                "expert": 0,
                "bytes": 3,
                "read_ranges": [{"object": source, "offset": 40, "bytes": 3}],
                "tensors": [],
            },
            {
                "expert": 1,
                "bytes": 4,
                "read_ranges": [{"object": source, "offset": 50, "bytes": 4}],
                "tensors": [],
            },
        ],
    }


def global_plan_document(model_id: str, revision: str) -> dict[str, object]:
    names = (
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "output_hc_base.weight",
        "output_hc_scale.weight",
        "output_hc_fn.weight",
    )
    return {
        "ok": True,
        "format": RANGE_MODULE.GLOBAL_PLAN_FORMAT,
        "model_id": model_id,
        "revision": revision,
        "plan_only": True,
        "execution_supported": False,
        "routing_authority_verified": False,
        "payload_access": {
            "metadata_only": True,
            "imported_file_hashes_verified": True,
            "full_payload_present": False,
            "executable_source_binding_verified": False,
        },
        "tensor_count": 6,
        "payload_bytes": 6,
        "maximum_read_range_bytes": 1,
        "unique_source_objects": 1,
        "tensors": [
            {
                "tensor": name,
                "role": "global",
                "shape": [1],
                "ggml_type": 0,
                "source": {"object": "test.gguf", "offset": 100 + index, "bytes": 1},
            }
            for index, name in enumerate(names)
        ],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vastloom-import-test-") as temporary:
        root = Path(temporary)
        (root / "models" / "original").mkdir(parents=True)
        (root / "models" / "optimized").mkdir()
        (root / "cache").mkdir()
        (root / ".vastloom-store").write_text(MODULE.MARKER, encoding="utf-8")
        require(MODULE.validate_store(root) == root.resolve(), "valid store rejected")

        legacy_root = root / "legacy-store"
        (legacy_root / "models" / "original").mkdir(parents=True)
        (legacy_root / "cache").mkdir()
        (legacy_root / ".stargate-store").write_text(MODULE.LEGACY_MARKER, encoding="utf-8")
        require(MODULE.validate_store(legacy_root) == legacy_root.resolve(), "legacy store rejected")
        with mock.patch.object(
            MODULE.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=64 * MODULE.GIB),
        ):
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
        gguf_metadata = MODULE.selected_files(MODULE.DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S, True)
        gguf_full = MODULE.selected_files(MODULE.DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S, False)
        require(len(gguf_metadata) == 3, "DeepSeek 0731 GGUF metadata set is incomplete")
        require(len(gguf_full) == 3, "DeepSeek 0731 GGUF full set is incomplete")
        require(
            sum(item.size for item in gguf_metadata) == 5_343_584,
            "DeepSeek 0731 bounded GGUF metadata byte count changed",
        )
        require(
            sum(item.size for item in gguf_full) == 82_539_237_792,
            "DeepSeek 0731 GGUF split byte count changed",
        )
        require(
            sum(item.size for item in gguf_full if item.weight) == 82_533_980_128,
            "DeepSeek 0731 GGUF payload authorization byte count changed",
        )
        require(
            all(item.sha256 for item in MODULE.DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S.files),
            "DeepSeek 0731 GGUF objects and headers must have pinned SHA-256 values",
        )
        require(
            {item.name for item in gguf_metadata if item.range_prefix}
            == {
                "headers/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf.header",
                "headers/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf.header",
            },
            "DeepSeek 0731 GGUF metadata import did not select the exact bounded headers",
        )
        require(
            MODULE.target_name(MODULE.DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S, True)
            == "deepseek-v4-flash-0731-iq1s-fbbb5b93--metadata",
            "DeepSeek 0731 Windows-safe artifact name changed",
        )
        try:
            MODULE.import_model(
                root,
                MODULE.DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S,
                metadata_only=False,
                reserve_bytes=20 * MODULE.GIB,
                timeout=1.0,
                dry_run=True,
            )
        except RuntimeError as error:
            require("--allow-large-download" in str(error), "GGUF large-import refusal is not actionable")
        else:
            raise AssertionError("DeepSeek 0731 full GGUF import lacked explicit authorization")
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

        range_payload = bytes(range(256))
        range_spec = MODULE.ModelSpec(
            "range-test",
            "test/range-model",
            "0123456789abcdef0123456789abcdef01234567",
            (
                MODULE.FileSpec(
                    "UD/test.gguf",
                    len(range_payload),
                    hashlib.sha256(range_payload).hexdigest(),
                    True,
                ),
            ),
            artifact_name="range-test",
        )
        plan_path = root / "range-plan.json"
        plan_path.write_text(
            json.dumps(range_plan_document(range_spec.model_id, range_spec.revision)),
            encoding="utf-8",
        )
        plan = RANGE_MODULE.load_plan(plan_path, range_spec)
        require(len(plan.trunk_ranges) == 2, "range plan trunk was not parsed")
        global_plan_path = root / "global-plan.json"
        global_plan_path.write_text(
            json.dumps(global_plan_document(range_spec.model_id, range_spec.revision)),
            encoding="utf-8",
        )
        global_plan = RANGE_MODULE.load_plan(global_plan_path, range_spec)
        require(
            global_plan.plan_format == RANGE_MODULE.GLOBAL_PLAN_FORMAT
            and len(global_plan.selected("globals")) == 6
            and not global_plan.trunk_ranges
            and not global_plan.expert_ranges,
            "global tensor plan was not parsed as a closed six-range set",
        )
        try:
            global_plan.selected("trunk")
        except RuntimeError as error:
            require("layer probe" in str(error), "global-plan scope refusal is not actionable")
        else:
            raise AssertionError("global tensor plan was accepted for trunk import")
        with mock.patch.object(
            MODULE.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=64 * MODULE.GIB),
        ):
            global_target = RANGE_MODULE.import_range_artifact(
                root,
                range_spec,
                global_plan,
                scope="globals",
                reserve_bytes=20 * MODULE.GIB,
                timeout=1.0,
                dry_run=True,
                allow_range_download=False,
                max_total_bytes=6,
            )
        require(
            "range-globals" in global_target.name and not global_target.exists(),
            "global range dry-run changed storage or artifact identity",
        )
        malformed_global = global_plan_document(range_spec.model_id, range_spec.revision)
        malformed_global["tensors"][0]["tensor"] = "output.weight"  # type: ignore[index]
        malformed_global_path = root / "malformed-global-plan.json"
        malformed_global_path.write_text(json.dumps(malformed_global), encoding="utf-8")
        try:
            RANGE_MODULE.load_plan(malformed_global_path, range_spec)
        except RuntimeError:
            pass
        else:
            raise AssertionError("misordered global tensor plan was accepted")
        bom_plan_path = root / "range-plan-bom.json"
        bom_plan_path.write_bytes(b"\xef\xbb\xbf" + plan_path.read_bytes())
        require(
            len(RANGE_MODULE.load_plan(bom_plan_path, range_spec).trunk_ranges) == 2,
            "UTF-8 BOM range plan was not parsed",
        )
        try:
            plan.selected("selected-experts")
        except RuntimeError as error:
            require("routing authority" in str(error), "unverified route refusal is not actionable")
        else:
            raise AssertionError("unverified selected-expert range import was accepted")
        forged_route = range_plan_document(range_spec.model_id, range_spec.revision)
        forged_route["routing_authority_verified"] = True
        forged_route_path = root / "forged-route-plan.json"
        forged_route_path.write_text(json.dumps(forged_route), encoding="utf-8")
        try:
            RANGE_MODULE.load_plan(forged_route_path, range_spec)
        except RuntimeError:
            pass
        else:
            raise AssertionError("inconsistent routing-authority source was accepted")

        with mock.patch.object(
            MODULE.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=64 * MODULE.GIB),
        ):
            dry_target = RANGE_MODULE.import_range_artifact(
                root,
                range_spec,
                plan,
                scope="trunk",
                reserve_bytes=20 * MODULE.GIB,
                timeout=1.0,
                dry_run=True,
                allow_range_download=False,
                max_total_bytes=12,
            )
        require(not dry_target.exists(), "range dry-run created an artifact")
        try:
            RANGE_MODULE.import_range_artifact(
                root,
                range_spec,
                plan,
                scope="trunk",
                reserve_bytes=20 * MODULE.GIB,
                timeout=1.0,
                dry_run=False,
                allow_range_download=False,
                max_total_bytes=12,
            )
        except RuntimeError as error:
            require("--allow-range-download" in str(error), "range authorization refusal is not actionable")
        else:
            raise AssertionError("range import lacked explicit download authorization")
        try:
            RANGE_MODULE.import_range_artifact(
                root,
                range_spec,
                plan,
                scope="trunk",
                reserve_bytes=20 * MODULE.GIB,
                timeout=1.0,
                dry_run=True,
                allow_range_download=False,
                max_total_bytes=11,
            )
        except RuntimeError as error:
            require("exceed" in str(error), "range byte-cap refusal is not actionable")
        else:
            raise AssertionError("range import exceeded its explicit payload cap")

        requested_ranges: list[str] = []

        stale_cache = root / "cache" / "range-imports" / dry_target.name
        stale_cache.mkdir(parents=True)
        (stale_cache / ".0000.partial-crash").write_bytes(b"partial")
        (stale_cache / "0000.bin").write_bytes(b"unpublished")
        stale_incoming = root / "models" / "optimized" / f".incoming-{dry_target.name}-crash"
        stale_incoming.mkdir()
        (stale_incoming / "unpublished.bin").write_bytes(b"partial")

        class ExactRangeHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - standard-library handler API
                requested = self.headers.get("Range", "")
                start_text, end_text = requested.removeprefix("bytes=").split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                require(0 <= start <= end < len(range_payload), "invalid exact range request")
                requested_ranges.append(requested)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(range_payload)}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                self.wfile.write(range_payload[start : end + 1])

            def log_message(self, _format: str, *_args: object) -> None:
                return

        range_server = ThreadingHTTPServer(("127.0.0.1", 0), ExactRangeHandler)
        range_worker = threading.Thread(target=range_server.serve_forever, daemon=True)
        range_worker.start()
        try:
            with (
                mock.patch.object(
                    MODULE.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=64 * MODULE.GIB),
                ),
                mock.patch.object(
                    RANGE_MODULE.base,
                    "file_url",
                    return_value=f"http://127.0.0.1:{range_server.server_port}/test.gguf",
                ),
            ):
                imported = RANGE_MODULE.import_range_artifact(
                    root,
                    range_spec,
                    plan,
                    scope="trunk",
                    reserve_bytes=20 * MODULE.GIB,
                    timeout=5.0,
                    dry_run=False,
                    allow_range_download=True,
                    max_total_bytes=12,
                    max_bytes_per_second=1024 * 1024,
                )
        finally:
            range_server.shutdown()
            range_worker.join(timeout=5)
        require(requested_ranges == ["bytes=4-8", "bytes=20-26"], "exact source ranges changed")
        require(not stale_incoming.exists(), "stale range staging directory was not cleaned")
        require(not (stale_cache / ".0000.partial-crash").exists(),
                "stale range partial was not cleaned")
        manifest = json.loads(
            (imported / RANGE_MODULE.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        require(manifest["format"] == RANGE_MODULE.ARTIFACT_FORMAT, "range artifact format changed")
        require(manifest["execution_supported"] is False, "partial artifact claimed execution")
        require(manifest["complete_checkpoint_verified"] is False, "partial artifact claimed completeness")
        require(manifest["payload_bytes"] == 12 and manifest["range_count"] == 2,
                "range artifact accounting changed")
        require((imported / "ranges" / "0000.bin").read_bytes() == range_payload[4:9],
                "first immutable range payload changed")
        require((imported / "ranges" / "0001.bin").read_bytes() == range_payload[20:27],
                "second immutable range payload changed")
        reused = RANGE_MODULE.import_range_artifact(
            root,
            range_spec,
            plan,
            scope="trunk",
            reserve_bytes=20 * MODULE.GIB,
            timeout=1.0,
            dry_run=False,
            allow_range_download=False,
            max_total_bytes=12,
        )
        require(reused == imported, "verified immutable range artifact was not reused")

        malformed = range_plan_document(range_spec.model_id, range_spec.revision)
        malformed["total_payload_bytes"] = 20
        malformed_path = root / "malformed-range-plan.json"
        malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
        try:
            RANGE_MODULE.load_plan(malformed_path, range_spec)
        except RuntimeError:
            pass
        else:
            raise AssertionError("inconsistent range plan totals were accepted")

        duplicate_path = root / "duplicate-range-plan.json"
        duplicate_path.write_text('{"format":"a","format":"b"}', encoding="utf-8")
        try:
            RANGE_MODULE.load_plan(duplicate_path, range_spec)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate range-plan JSON keys were accepted")

        class WrongRangeHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - standard-library handler API
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-4/{len(range_payload)}")
                self.send_header("Content-Length", "5")
                self.end_headers()
                self.wfile.write(range_payload[:5])

            def log_message(self, _format: str, *_args: object) -> None:
                return

        wrong_server = ThreadingHTTPServer(("127.0.0.1", 0), WrongRangeHandler)
        wrong_worker = threading.Thread(target=wrong_server.serve_forever, daemon=True)
        wrong_worker.start()
        wrong_partial = root / "wrong-range.partial"
        try:
            try:
                RANGE_MODULE.download_range(
                    f"http://127.0.0.1:{wrong_server.server_port}/test.gguf",
                    wrong_partial,
                    plan.trunk_ranges[0],
                    5.0,
                    1024 * 1024,
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("wrong HTTP Content-Range was accepted")
        finally:
            wrong_server.shutdown()
            wrong_worker.join(timeout=5)
        require(not wrong_partial.exists(), "failed range download left a partial artifact")

    print("Hugging Face importer offline tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
