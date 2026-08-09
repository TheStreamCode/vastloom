#!/usr/bin/env python3
"""Pinned, bounded and atomic Hugging Face checkpoint importer for Stargate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen


GIB = 1024**3
DEFAULT_DISK_RESERVE = 300 * GIB
COPY_BUFFER_BYTES = 1024 * 1024
MARKER = "format=stargate.store.v1\noriginal_weights=immutable\ncache=reconstructable\n"


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    sha256: str | None = None
    weight: bool = False


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    revision: str
    files: tuple[FileSpec, ...]


SMOLLM2 = ModelSpec(
    alias="smollm2",
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
    files=(
        FileSpec("config.json", 861),
        FileSpec("generation_config.json", 132),
        FileSpec("merges.txt", 466_391),
        FileSpec(
            "model.safetensors",
            269_060_552,
            "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
            True,
        ),
        FileSpec("special_tokens_map.json", 655),
        FileSpec("tokenizer.json", 2_104_556),
        FileSpec("tokenizer_config.json", 3_764),
        FileSpec("vocab.json", 800_662),
    ),
)

MODELS = {SMOLLM2.alias: SMOLLM2}


def validate_store(store: Path) -> Path:
    root = store.resolve(strict=True)
    marker = root / ".stargate-store"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != MARKER:
        raise RuntimeError(f"invalid Stargate model store: {root}")
    original = (root / "models" / "original").resolve(strict=True)
    if original.parent.parent != root or original.is_symlink():
        raise RuntimeError(f"unsafe Stargate original-model directory: {original}")
    return root


def target_name(spec: ModelSpec, metadata_only: bool) -> str:
    model = spec.model_id.replace("/", "-")
    suffix = "--metadata" if metadata_only else ""
    return f"{model}--{spec.revision}{suffix}"


def selected_files(spec: ModelSpec, metadata_only: bool) -> tuple[FileSpec, ...]:
    return tuple(item for item in spec.files if not (metadata_only and item.weight))


def preflight_disk(root: Path, required_bytes: int, reserve_bytes: int) -> None:
    if required_bytes < 0 or reserve_bytes < 20 * GIB:
        raise RuntimeError("invalid download size or disk reserve")
    available = shutil.disk_usage(root).free
    if available < required_bytes + reserve_bytes:
        raise RuntimeError(
            f"download refused: {available} bytes free, {required_bytes} required, "
            f"and {reserve_bytes} reserved"
        )


def file_url(spec: ModelSpec, item: FileSpec) -> str:
    repository = quote(spec.model_id, safe="/")
    revision = quote(spec.revision, safe="")
    filename = quote(item.name, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{filename}?download=true"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing(target: Path, spec: ModelSpec, files: Iterable[FileSpec], metadata_only: bool) -> None:
    manifest_path = target / "stargate-import.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"existing immutable target has no import manifest: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("model_id") != spec.model_id
        or manifest.get("revision") != spec.revision
        or bool(manifest.get("metadata_only")) != metadata_only
    ):
        raise RuntimeError(f"existing immutable target identity does not match: {target}")
    recorded = {item["name"]: item for item in manifest.get("files", [])}
    for item in files:
        path = target / item.name
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"existing immutable artifact is missing or has the wrong size: {path}")
        expected_hash = item.sha256 or recorded.get(item.name, {}).get("sha256")
        if not expected_hash or hash_file(path) != expected_hash:
            raise RuntimeError(f"existing immutable artifact failed SHA-256 validation: {path}")


def download_file(url: str, partial: Path, item: FileSpec, timeout: float) -> str:
    digest = hashlib.sha256()
    received = 0
    request = Request(url, headers={"User-Agent": "stargate-local-ai/0.3-importer"})
    with urlopen(request, timeout=timeout) as response, partial.open("xb") as output:
        while True:
            chunk = response.read(COPY_BUFFER_BYTES)
            if not chunk:
                break
            received += len(chunk)
            if received > item.size:
                raise RuntimeError(f"server sent more bytes than declared for {item.name}")
            output.write(chunk)
            digest.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    if received != item.size:
        raise RuntimeError(f"short download for {item.name}: expected {item.size}, got {received}")
    actual_hash = digest.hexdigest()
    if item.sha256 and actual_hash != item.sha256:
        raise RuntimeError(f"SHA-256 mismatch for {item.name}")
    return actual_hash


def import_model(
    store: Path,
    spec: ModelSpec,
    *,
    metadata_only: bool,
    reserve_bytes: int,
    timeout: float,
    dry_run: bool,
) -> Path:
    root = validate_store(store)
    files = selected_files(spec, metadata_only)
    required_bytes = sum(item.size for item in files)
    preflight_disk(root, required_bytes, reserve_bytes)
    original = root / "models" / "original"
    target = original / target_name(spec, metadata_only)
    if target.exists():
        validate_existing(target, spec, files, metadata_only)
        return target
    if dry_run:
        return target

    incoming = original / f".incoming-{target.name}-{os.getpid()}-{time.time_ns()}"
    if incoming.parent.resolve(strict=True) != original.resolve(strict=True):
        raise RuntimeError("unsafe importer staging path")
    incoming.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    try:
        for item in files:
            preflight_disk(root, item.size, reserve_bytes)
            destination = incoming / item.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            actual_hash = download_file(file_url(spec, item), partial, item, timeout)
            os.replace(partial, destination)
            records.append({"name": item.name, "size": item.size, "sha256": actual_hash})

        manifest = {
            "format": "stargate.hf-import.v1",
            "model_id": spec.model_id,
            "revision": spec.revision,
            "metadata_only": metadata_only,
            "files": records,
        }
        manifest_partial = incoming / "stargate-import.json.partial"
        with manifest_partial.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(manifest_partial, incoming / "stargate-import.json")
        for path in incoming.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        incoming.rename(target)
        return target
    except BaseException:
        if incoming.exists() and incoming.parent.resolve(strict=True) == original.resolve(strict=True):
            for path in incoming.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IWRITE | stat.S_IREAD)
            shutil.rmtree(incoming)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--reserve-gib", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reserve_gib < 20 or args.timeout <= 0:
        print("ERROR: reserve must be at least 20 GiB and timeout must be positive", file=sys.stderr)
        return 64
    try:
        target = import_model(
            args.store,
            MODELS[args.model],
            metadata_only=args.metadata_only,
            reserve_bytes=args.reserve_gib * GIB,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
