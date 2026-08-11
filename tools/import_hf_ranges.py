#!/usr/bin/env python3
"""Import bounded ranges from a verified Vastloom plan into an immutable derived artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import Any, Iterable
from urllib.request import Request, urlopen

import import_hf_model as base


MIB = 1024**2
LAYER_PLAN_FORMAT = "vastloom.deepseek_v4.layer_probe_plan.v1"
GLOBAL_PLAN_FORMAT = "vastloom.deepseek_v4.global_tensor_plan.v1"
# Backward-compatible test/tool constant for the original layer plan.
PLAN_FORMAT = LAYER_PLAN_FORMAT
ARTIFACT_FORMAT = "vastloom.range-artifact.v1"
ARTIFACT_MANIFEST_NAME = "vastloom-range-artifact.json"
MAX_PLAN_BYTES = 16 * MIB
MAX_PLAN_RANGES = 4_096
MAX_RANGE_BYTES = 1024**3
DEFAULT_MAX_TOTAL_BYTES = 256 * MIB


@dataclass(frozen=True)
class SourceRange:
    role: str
    label: str
    source_object: str
    source_path: str
    source_size: int
    source_sha256: str
    offset: int
    size: int


@dataclass(frozen=True)
class ValidatedPlan:
    raw_sha256: str
    plan_format: str
    model_id: str
    revision: str
    layer: int
    routing_authority_verified: bool
    trunk_ranges: tuple[SourceRange, ...]
    expert_ranges: tuple[SourceRange, ...]
    global_ranges: tuple[SourceRange, ...]

    def selected(self, scope: str) -> tuple[SourceRange, ...]:
        if scope == "globals":
            if self.plan_format != GLOBAL_PLAN_FORMAT or not self.global_ranges:
                raise RuntimeError("globals import requires a verified global tensor plan")
            return self.global_ranges
        if self.plan_format != LAYER_PLAN_FORMAT:
            raise RuntimeError(f"scope {scope} requires a verified layer probe plan")
        if scope == "trunk":
            return self.trunk_ranges
        if not self.routing_authority_verified:
            raise RuntimeError(
                "selected-expert range import refused: the plan has no verified routing authority"
            )
        if scope == "selected-experts":
            return self.expert_ranges
        if scope == "all":
            return self.trunk_ranges + self.expert_ranges
        raise RuntimeError(f"unsupported range-import scope: {scope}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _integer(mapping: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"range plan field {key} is invalid")
    return value


def _boolean(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise RuntimeError(f"range plan field {key} is invalid")
    return value


def _string(mapping: dict[str, Any], key: str, *, maximum: int = 4_096) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise RuntimeError(f"range plan field {key} is invalid")
    return value


def _source_specs(spec: base.ModelSpec) -> dict[str, base.FileSpec]:
    result: dict[str, base.FileSpec] = {}
    for item in base.selected_files(spec, False):
        object_name = Path(item.name).name
        if object_name in result:
            raise RuntimeError(f"pinned model has an ambiguous source object: {object_name}")
        if not item.sha256 or len(item.sha256) != 64:
            raise RuntimeError(f"pinned source object lacks SHA-256: {item.name}")
        result[object_name] = item
    return result


def _parse_source_range(
    value: Any,
    *,
    role: str,
    label: str,
    sources: dict[str, base.FileSpec],
) -> SourceRange:
    if not isinstance(value, dict):
        raise RuntimeError(f"range plan source is invalid: {label}")
    source_object = _string(value, "object", maximum=512)
    source = sources.get(source_object)
    if source is None or not source.sha256:
        raise RuntimeError(f"range plan names an unpinned source object: {source_object}")
    offset = _integer(value, "offset")
    size = _integer(value, "bytes", minimum=1)
    if size > MAX_RANGE_BYTES or offset > source.size or size > source.size - offset:
        raise RuntimeError(f"range plan source span is outside its pinned object: {label}")
    return SourceRange(
        role=role,
        label=label,
        source_object=source_object,
        source_path=source.name,
        source_size=source.size,
        source_sha256=source.sha256,
        offset=offset,
        size=size,
    )


def _validate_nonoverlap(ranges: Iterable[SourceRange]) -> None:
    ordered = sorted(ranges, key=lambda item: (item.source_object, item.offset, item.size))
    previous: SourceRange | None = None
    for current in ordered:
        if (
            previous is not None
            and previous.source_object == current.source_object
            and current.offset < previous.offset + previous.size
        ):
            raise RuntimeError(
                f"range plan has overlapping source spans: {previous.label} and {current.label}"
            )
        previous = current


def load_plan(path: Path, spec: base.ModelSpec) -> ValidatedPlan:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"range plan must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_PLAN_BYTES:
        raise RuntimeError(f"range plan size is outside the {MAX_PLAN_BYTES}-byte limit: {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"range plan JSON is invalid: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError("range plan root must be an object")
    if (
        document.get("format") not in (LAYER_PLAN_FORMAT, GLOBAL_PLAN_FORMAT)
        or document.get("ok") is not True
        or document.get("plan_only") is not True
        or document.get("execution_supported") is not False
        or document.get("model_id") != spec.model_id
        or document.get("revision") != spec.revision
    ):
        raise RuntimeError("range plan identity or non-execution contract does not match")
    payload_access = document.get("payload_access")
    if not isinstance(payload_access, dict) or payload_access.get("imported_file_hashes_verified") is not True:
        raise RuntimeError("range plan was not derived from a hash-verified metadata import")

    plan_format = str(document["format"])
    sources = _source_specs(spec)
    if plan_format == GLOBAL_PLAN_FORMAT:
        if document.get("routing_authority_verified") is not False:
            raise RuntimeError("global tensor plan must not claim routing authority")
        tensor_values = document.get("tensors")
        canonical = (
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            "output_hc_base.weight",
            "output_hc_scale.weight",
            "output_hc_fn.weight",
        )
        if not isinstance(tensor_values, list) or len(tensor_values) != len(canonical):
            raise RuntimeError("global tensor plan must contain exactly six tensors")
        globals_: list[SourceRange] = []
        for index, (item, expected_name) in enumerate(
            zip(tensor_values, canonical, strict=True)
        ):
            if not isinstance(item, dict) or _string(item, "tensor") != expected_name:
                raise RuntimeError("global tensor plan order or identity is invalid")
            globals_.append(
                _parse_source_range(
                    item.get("source"),
                    role="global",
                    label=f"global[{index}] {expected_name}",
                    sources=sources,
                )
            )
        global_ranges = tuple(globals_)
        _validate_nonoverlap(global_ranges)
        total = sum(item.size for item in global_ranges)
        if (
            _integer(document, "tensor_count") != len(global_ranges)
            or _integer(document, "payload_bytes") != total
            or _integer(document, "maximum_read_range_bytes")
            != max(item.size for item in global_ranges)
            or _integer(document, "unique_source_objects")
            != len({item.source_object for item in global_ranges})
        ):
            raise RuntimeError("global tensor plan counts or byte totals are inconsistent")
        return ValidatedPlan(
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            plan_format=plan_format,
            model_id=spec.model_id,
            revision=spec.revision,
            layer=0,
            routing_authority_verified=False,
            trunk_ranges=(),
            expert_ranges=(),
            global_ranges=global_ranges,
        )

    layer = _integer(document, "layer")
    expected_top_k = _integer(document, "expected_top_k", minimum=1)
    if expected_top_k > 64:
        raise RuntimeError("range plan top-k exceeds the 64-ID safety limit")
    routing_authority = _boolean(document, "routing_authority_verified")
    selection_source = _string(document, "selection_source", maximum=128)
    if (
        (not routing_authority and selection_source != "caller-supplied-unverified")
        or (routing_authority and selection_source != "official-tid2eid-verified")
    ):
        raise RuntimeError("range plan routing-authority source is inconsistent")
    selected_experts = document.get("selected_experts")
    if (
        not isinstance(selected_experts, list)
        or len(selected_experts) != expected_top_k
        or any(type(item) is not int or item < 0 for item in selected_experts)
        or len(set(selected_experts)) != len(selected_experts)
    ):
        raise RuntimeError("range plan selected expert IDs are invalid")

    trunk_values = document.get("trunk_ranges")
    if not isinstance(trunk_values, list) or not trunk_values:
        raise RuntimeError("range plan has no trunk ranges")
    trunk: list[SourceRange] = []
    for index, item in enumerate(trunk_values):
        if not isinstance(item, dict):
            raise RuntimeError("range plan trunk entry is invalid")
        tensor = _string(item, "tensor")
        trunk.append(
            _parse_source_range(
                item.get("source"), role="trunk", label=f"trunk[{index}] {tensor}", sources=sources
            )
        )

    expert_values = document.get("experts")
    if not isinstance(expert_values, list) or len(expert_values) != expected_top_k:
        raise RuntimeError("range plan expert bundles do not match top-k")
    experts: list[SourceRange] = []
    expert_bytes = 0
    expert_ids: list[int] = []
    minimum_expert_tier = 0
    for expert_index, item in enumerate(expert_values):
        if not isinstance(item, dict):
            raise RuntimeError("range plan expert entry is invalid")
        expert_id = _integer(item, "expert")
        bundle_bytes = _integer(item, "bytes", minimum=1)
        ranges = item.get("read_ranges")
        if not isinstance(ranges, list) or not ranges:
            raise RuntimeError(f"range plan expert {expert_id} has no read ranges")
        parsed = [
            _parse_source_range(
                source,
                role="selected-expert",
                label=f"expert[{expert_index}:{expert_id}].range[{range_index}]",
                sources=sources,
            )
            for range_index, source in enumerate(ranges)
        ]
        if sum(entry.size for entry in parsed) != bundle_bytes:
            raise RuntimeError(f"range plan expert {expert_id} byte total does not match its ranges")
        expert_ids.append(expert_id)
        expert_bytes += bundle_bytes
        minimum_expert_tier = max(minimum_expert_tier, bundle_bytes)
        experts.extend(parsed)
    if expert_ids != selected_experts:
        raise RuntimeError("range plan expert bundle order differs from selected expert IDs")

    all_ranges = tuple(trunk + experts)
    if len(all_ranges) > MAX_PLAN_RANGES:
        raise RuntimeError("range plan exceeds the 4,096-range safety limit")
    _validate_nonoverlap(all_ranges)
    trunk_bytes = sum(item.size for item in trunk)
    if (
        _integer(document, "trunk_range_count") != len(trunk)
        or _integer(document, "selected_expert_range_count") != len(experts)
        or _integer(document, "total_range_count") != len(all_ranges)
        or _integer(document, "trunk_bytes") != trunk_bytes
        or _integer(document, "selected_expert_bytes") != expert_bytes
        or _integer(document, "total_payload_bytes") != trunk_bytes + expert_bytes
        or _integer(document, "minimum_expert_tier_bytes") != minimum_expert_tier
        or _integer(document, "unique_source_objects")
        != len({item.source_object for item in all_ranges})
    ):
        raise RuntimeError("range plan counts or byte totals are inconsistent")

    return ValidatedPlan(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        plan_format=plan_format,
        model_id=spec.model_id,
        revision=spec.revision,
        layer=layer,
        routing_authority_verified=routing_authority,
        trunk_ranges=tuple(trunk),
        expert_ranges=tuple(experts),
        global_ranges=(),
    )


def _artifact_name(spec: base.ModelSpec, plan: ValidatedPlan, scope: str) -> str:
    model = spec.artifact_name or spec.model_id.replace("/", "-")
    safe_scope = scope.replace("selected-experts", "experts")
    if scope == "globals":
        return f"{model}--range-globals-{plan.raw_sha256[:12]}"
    return f"{model}--range-l{plan.layer}-{safe_scope}-{plan.raw_sha256[:12]}"


def _range_record(entry: SourceRange, index: int, sha256: str) -> dict[str, object]:
    return {
        "artifact": f"ranges/{index:04d}.bin",
        "bytes": entry.size,
        "label": entry.label,
        "role": entry.role,
        "sha256": sha256,
        "source_bytes": entry.size,
        "source_object": entry.source_object,
        "source_object_bytes": entry.source_size,
        "source_object_sha256": entry.source_sha256,
        "source_offset": entry.offset,
        "source_path": entry.source_path,
    }


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    partial = path.with_name(path.name + f".partial-{os.getpid()}-{time.time_ns()}")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(document, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def download_range(
    url: str,
    partial: Path,
    entry: SourceRange,
    timeout: float,
    max_bytes_per_second: int,
) -> str:
    headers = {
        "Range": f"bytes={entry.offset}-{entry.offset + entry.size - 1}",
        "User-Agent": "vastloom-local-ai/0.4-range-importer",
    }
    request = Request(url, headers=headers)
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = getattr(response, "status", response.getcode())
            expected_content_range = (
                f"bytes {entry.offset}-{entry.offset + entry.size - 1}/{entry.source_size}"
            )
            if status_code != 206 or response.headers.get("Content-Range", "") != expected_content_range:
                raise RuntimeError(f"server returned an invalid bounded range for {entry.label}")
            content_length = response.headers.get("Content-Length", "")
            if content_length != str(entry.size):
                raise RuntimeError(f"server returned an invalid content length for {entry.label}")
            with partial.open("xb") as output:
                while True:
                    chunk = response.read(base.COPY_BUFFER_BYTES)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > entry.size:
                        raise RuntimeError(f"server sent more bytes than planned for {entry.label}")
                    output.write(chunk)
                    digest.update(chunk)
                    if max_bytes_per_second > 0:
                        target_elapsed = received / max_bytes_per_second
                        delay = target_elapsed - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(min(delay, 1.0))
                output.flush()
                os.fsync(output.fileno())
        if received != entry.size:
            raise RuntimeError(
                f"short range download for {entry.label}: expected {entry.size}, got {received}"
            )
        actual_hash = digest.hexdigest()
        if base.hash_file(partial) != actual_hash:
            raise RuntimeError(f"post-write SHA-256 verification failed for {entry.label}")
        return actual_hash
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _load_json_file(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PLAN_BYTES:
        raise RuntimeError(f"unsafe or oversized {description}: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"invalid {description}: {path}")
    return result


def _validate_artifact(target: Path, plan: ValidatedPlan, scope: str) -> None:
    manifest = _load_json_file(target / ARTIFACT_MANIFEST_NAME, "range artifact manifest")
    selected = plan.selected(scope)
    records = manifest.get("ranges")
    if (
        manifest.get("format") != ARTIFACT_FORMAT
        or manifest.get("model_id") != plan.model_id
        or manifest.get("revision") != plan.revision
        or manifest.get("source_plan_format") != plan.plan_format
        or manifest.get("source_plan_sha256") != plan.raw_sha256
        or manifest.get("scope") != scope
        or manifest.get("layer") != plan.layer
        or manifest.get("execution_supported") is not False
        or manifest.get("complete_checkpoint_verified") is not False
        or manifest.get("routing_authority_verified") is not plan.routing_authority_verified
        or manifest.get("range_count") != len(selected)
        or manifest.get("payload_bytes") != sum(item.size for item in selected)
        or not isinstance(records, list)
        or len(records) != len(selected)
    ):
        raise RuntimeError(f"range artifact identity does not match: {target}")
    for index, (entry, record) in enumerate(zip(selected, records, strict=True)):
        if not isinstance(record, dict):
            raise RuntimeError(f"range artifact record is invalid: {index}")
        artifact = f"ranges/{index:04d}.bin"
        path = target / artifact
        expected = {
            "artifact": artifact,
            "bytes": entry.size,
            "label": entry.label,
            "role": entry.role,
            "source_bytes": entry.size,
            "source_object": entry.source_object,
            "source_object_bytes": entry.source_size,
            "source_object_sha256": entry.source_sha256,
            "source_offset": entry.offset,
            "source_path": entry.source_path,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"range artifact record identity does not match: {index}")
        expected_hash = record.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.size
            or base.hash_file(path) != expected_hash
        ):
            raise RuntimeError(f"range artifact payload failed verification: {path}")


def _cache_record_matches(path: Path, entry: SourceRange, index: int, payload: Path) -> str | None:
    if not path.exists() and not path.is_symlink() and not payload.exists() and not payload.is_symlink():
        return None
    if path.is_symlink() or payload.is_symlink() or not path.is_file() or not payload.is_file():
        raise RuntimeError(f"range cache is incomplete or unsafe: {payload}")
    record = _load_json_file(path, "range cache receipt")
    expected = _range_record(entry, index, str(record.get("sha256", "")))
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"range cache receipt identity does not match: {payload}")
    expected_hash = record.get("sha256")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or payload.stat().st_size != entry.size
        or base.hash_file(payload) != expected_hash
    ):
        raise RuntimeError(f"range cache payload failed verification: {payload}")
    return expected_hash


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_generated_tree(path: Path, expected_parent: Path) -> None:
    if path.parent.resolve(strict=True) != expected_parent or path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"unsafe stale range-import staging path: {path}")
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            child.chmod(stat.S_IWRITE | stat.S_IREAD)
    shutil.rmtree(path)


def _cleanup_stale_state(cache_root: Path, optimized: Path, target_name: str) -> None:
    for partial in cache_root.glob(".*.partial-*"):
        if partial.parent.resolve(strict=True) != cache_root.resolve(strict=True) or partial.is_symlink():
            raise RuntimeError(f"unsafe stale range-import partial: {partial}")
        if not partial.is_file():
            raise RuntimeError(f"stale range-import partial is not a file: {partial}")
        partial.chmod(stat.S_IWRITE | stat.S_IREAD)
        partial.unlink()
    for stale in optimized.glob(f".incoming-{target_name}-*"):
        _remove_generated_tree(stale, optimized)


def import_range_artifact(
    store: Path,
    spec: base.ModelSpec,
    plan: ValidatedPlan,
    *,
    scope: str,
    reserve_bytes: int,
    timeout: float,
    dry_run: bool,
    allow_range_download: bool,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_bytes_per_second: int = 128 * MIB,
) -> Path:
    root = base.validate_store(store)
    optimized = (root / "models" / "optimized").resolve(strict=True)
    if optimized.parent.parent != root or optimized.is_symlink():
        raise RuntimeError(f"unsafe Vastloom optimized-model directory: {optimized}")
    selected = plan.selected(scope)
    payload_bytes = sum(item.size for item in selected)
    if not selected or max_total_bytes <= 0 or payload_bytes > max_total_bytes:
        raise RuntimeError(
            f"range import refused: {payload_bytes} bytes exceed the {max_total_bytes}-byte cap"
        )
    target = optimized / _artifact_name(spec, plan, scope)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise RuntimeError(f"unsafe existing range artifact: {target}")
        _validate_artifact(target, plan, scope)
        return target
    if not dry_run and not allow_range_download:
        raise RuntimeError(
            "range payload import requires --allow-range-download after reviewing the exact plan"
        )
    if dry_run:
        base.preflight_disk(root, payload_bytes, reserve_bytes)
        return target

    cache_root = root / "cache" / "range-imports" / target.name
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.resolve(strict=True).parent.parent != (root / "cache").resolve(strict=True):
        raise RuntimeError("unsafe range-import cache path")
    lock_path = cache_root / ".import.lock"
    incoming = optimized / f".incoming-{target.name}-{os.getpid()}-{time.time_ns()}"
    if incoming.parent.resolve(strict=True) != optimized:
        raise RuntimeError("unsafe range-import staging path")

    try:
        with base.import_lock(lock_path):
            _cleanup_stale_state(cache_root, optimized, target.name)
            hashes: list[str] = []
            cached_files: list[Path] = []
            cached_bytes = 0
            for index, entry in enumerate(selected):
                cached = cache_root / f"{index:04d}.bin"
                receipt = cache_root / f"{index:04d}.json"
                if (cached.exists() or cached.is_symlink()) != (
                    receipt.exists() or receipt.is_symlink()
                ):
                    for incomplete in (cached, receipt):
                        if incomplete.is_symlink() or (incomplete.exists() and not incomplete.is_file()):
                            raise RuntimeError(f"unsafe incomplete range cache entry: {incomplete}")
                        if incomplete.exists():
                            incomplete.chmod(stat.S_IWRITE | stat.S_IREAD)
                            incomplete.unlink()
                cached_hash = _cache_record_matches(receipt, entry, index, cached)
                if cached_hash is not None:
                    cached_bytes += entry.size
                    hashes.append(cached_hash)
                    cached_files.append(cached)
                    continue
                base.preflight_disk(root, payload_bytes - cached_bytes, reserve_bytes)
                partial = cache_root / f".{index:04d}.partial-{os.getpid()}-{time.time_ns()}"
                source = base.FileSpec(
                    entry.source_path,
                    entry.source_size,
                    entry.source_sha256,
                    weight=True,
                )
                base.progress(
                    f"download range {index + 1}/{len(selected)} {entry.source_object} "
                    f"[{entry.offset},{entry.offset + entry.size})"
                )
                actual_hash = download_range(
                    base.file_url(spec, source), partial, entry, timeout, max_bytes_per_second
                )
                os.replace(partial, cached)
                _write_json_atomic(receipt, _range_record(entry, index, actual_hash))
                cached_bytes += entry.size
                hashes.append(actual_hash)
                cached_files.append(cached)

            incoming.mkdir(mode=0o700)
            ranges_directory = incoming / "ranges"
            ranges_directory.mkdir()
            records: list[dict[str, object]] = []
            for index, (entry, cached, actual_hash) in enumerate(
                zip(selected, cached_files, hashes, strict=True)
            ):
                destination = ranges_directory / f"{index:04d}.bin"
                os.link(cached, destination)
                records.append(_range_record(entry, index, actual_hash))
            manifest = {
                "complete_checkpoint_verified": False,
                "execution_supported": False,
                "format": ARTIFACT_FORMAT,
                "layer": plan.layer,
                "model_id": plan.model_id,
                "payload_bytes": payload_bytes,
                "range_count": len(selected),
                "ranges": records,
                "revision": plan.revision,
                "routing_authority_verified": plan.routing_authority_verified,
                "scope": scope,
                "source_plan_format": plan.plan_format,
                "source_plan_sha256": plan.raw_sha256,
                "verification": "pinned-revision-content-range-plus-local-sha256",
            }
            _write_json_atomic(incoming / ARTIFACT_MANIFEST_NAME, manifest)
            _validate_artifact(incoming, plan, scope)
            for index, cached in enumerate(cached_files):
                cached.unlink()
                (cache_root / f"{index:04d}.json").unlink(missing_ok=True)
            for path in incoming.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IREAD)
            _fsync_directory(ranges_directory)
            _fsync_directory(incoming)
            if target.exists() or target.is_symlink():
                raise RuntimeError(f"range artifact target appeared during publication: {target}")
            incoming.rename(target)
            _fsync_directory(optimized)
            return target
    except BaseException:
        if incoming.exists() and incoming.parent.resolve(strict=True) == optimized:
            for path in incoming.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IWRITE | stat.S_IREAD)
            shutil.rmtree(incoming)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(base.DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S.alias,), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("trunk", "selected-experts", "all", "globals"), required=True
    )
    parser.add_argument("--reserve-gib", type=int, default=300)
    parser.add_argument("--max-total-mib", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-mib-per-sec", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-range-download", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.reserve_gib < 20
        or args.max_total_mib < 1
        or args.max_total_mib > 16 * 1024
        or args.timeout <= 0
        or args.max_mib_per_sec < 1
        or args.max_mib_per_sec > 1024
    ):
        print("ERROR: reserve, range cap, timeout or bandwidth limit is outside safe bounds", file=sys.stderr)
        return 64
    base.lower_process_priority()
    try:
        spec = base.MODELS[args.model]
        plan = load_plan(args.plan, spec)
        selected = plan.selected(args.scope)
        target = import_range_artifact(
            args.store,
            spec,
            plan,
            scope=args.scope,
            reserve_bytes=args.reserve_gib * base.GIB,
            timeout=args.timeout,
            dry_run=args.dry_run,
            allow_range_download=args.allow_range_download,
            max_total_bytes=args.max_total_mib * MIB,
            max_bytes_per_second=args.max_mib_per_sec * MIB,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        else:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "execution_supported": False,
                    "ok": True,
                    "payload_bytes": sum(item.size for item in selected),
                    "plan_sha256": plan.raw_sha256,
                    "range_count": len(selected),
                    "routing_authority_verified": plan.routing_authority_verified,
                    "scope": args.scope,
                    "target": str(target),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
