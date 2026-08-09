#!/usr/bin/env python3
"""Pinned, bounded and atomic Hugging Face checkpoint importer for Stargate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
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

if os.name == "nt":
    import msvcrt
else:
    import fcntl


GIB = 1024**3
DEFAULT_DISK_RESERVE = 300 * GIB
COPY_BUFFER_BYTES = 1024 * 1024
MARKER = "format=stargate.store.v1\noriginal_weights=immutable\ncache=reconstructable\n"


def progress(message: str) -> None:
    try:
        print(message, file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass


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

OLMOE = ModelSpec(
    alias="olmoe",
    model_id="allenai/OLMoE-1B-7B-0924-Instruct",
    revision="7f1c97f440f06ce36705e4f2b843edb5925f4498",
    files=(
        FileSpec("config.json", 759, "42ea6f5fe31e9243290ba3c348a4ad615b88a6fbc6e10544e21a87b6dff5da6c"),
        FileSpec("generation_config.json", 120, "77cf3f0caaa5e6d6972d910c70ca03f64fc47eccb4374082bdc8fc7b2c524a45"),
        FileSpec(
            "model-00001-of-00003.safetensors",
            4_997_744_872,
            "6c79ac0487c3f23e8ee3d38752197d1e4a1a39d6c1438ec5fd7862874bb19321",
            True,
        ),
        FileSpec(
            "model-00002-of-00003.safetensors",
            4_997_235_176,
            "b58ca7a9f28f35d76b56b0ec40fc46c356fd84ccf078083b251a3ad6a2da9a35",
            True,
        ),
        FileSpec(
            "model-00003-of-00003.safetensors",
            3_843_741_912,
            "e93653ed509223e63eabae6f72ac8ea5a41115c5c7785574b9b2068ca0961c45",
            True,
        ),
        FileSpec(
            "model.safetensors.index.json",
            287_214,
            "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
        ),
        FileSpec(
            "special_tokens_map.json",
            433,
            "cf22141a130514760f6280166b8d1c9a56610578188aad0790eb6eaf6ea1eb08",
        ),
        FileSpec(
            "tokenizer.json",
            2_115_598,
            "b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167",
        ),
        FileSpec(
            "tokenizer_config.json",
            5_986,
            "d3079f9c26a02fd28ee8d9bdfea4e96da7f5e803e06707ec8d1511acaaac7864",
        ),
    ),
)

MODELS = {item.alias: item for item in (SMOLLM2, OLMOE)}


def validate_store(store: Path) -> Path:
    root = store.resolve(strict=True)
    marker = root / ".stargate-store"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != MARKER:
        raise RuntimeError(f"invalid Stargate model store: {root}")
    original = (root / "models" / "original").resolve(strict=True)
    if original.parent.parent != root or original.is_symlink():
        raise RuntimeError(f"unsafe Stargate original-model directory: {original}")
    cache = (root / "cache").resolve(strict=True)
    if cache.parent != root or cache.is_symlink():
        raise RuntimeError(f"unsafe Stargate cache directory: {cache}")
    return root


@contextmanager
def import_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    locked = False
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError(f"another import owns the checkpoint lock: {path}") from error
        yield
    finally:
        try:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


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


def lower_process_priority() -> None:
    """Best-effort priority reduction; safety must not depend on it succeeding."""
    try:
        if os.name == "nt":
            below_normal_priority_class = 0x00004000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            kernel32.SetPriorityClass.restype = ctypes.c_int
            if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal_priority_class):
                raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
        else:
            os.nice(5)
    except OSError:
        pass


def validate_existing(target: Path, spec: ModelSpec, files: Iterable[FileSpec], metadata_only: bool) -> None:
    expected_files = tuple(files)
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
    if len(recorded) != len(manifest.get("files", [])) or len(recorded) != len(expected_files):
        raise RuntimeError(f"existing immutable artifact has an unexpected file set: {target}")
    for item in expected_files:
        path = target / item.name
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"existing immutable artifact is missing or has the wrong size: {path}")
        expected_hash = item.sha256 or recorded.get(item.name, {}).get("sha256")
        if not expected_hash or hash_file(path) != expected_hash:
            raise RuntimeError(f"existing immutable artifact failed SHA-256 validation: {path}")


def download_file(
    url: str,
    partial: Path,
    item: FileSpec,
    timeout: float,
    max_bytes_per_second: int,
) -> str:
    digest = hashlib.sha256()
    received = partial.stat().st_size if partial.exists() else 0
    if received > item.size:
        raise RuntimeError(f"partial download exceeds declared size for {item.name}")
    if received:
        progress(f"rehash partial {item.name}: {received}/{item.size} bytes")
        with partial.open("rb") as existing:
            while chunk := existing.read(COPY_BUFFER_BYTES):
                digest.update(chunk)
    initial_bytes = received
    started = time.monotonic()
    next_report = ((received // GIB) + 1) * GIB
    progress(f"download {item.name}: {received}/{item.size} bytes")
    headers = {"User-Agent": "stargate-local-ai/0.3-importer"}
    if received:
        headers["Range"] = f"bytes={received}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        if received and status != 206:
            raise RuntimeError(f"server refused safe resume for {item.name}; partial file was preserved")
        if received:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {received}-"):
                raise RuntimeError(f"server returned an invalid resume range for {item.name}; partial file was preserved")
        mode = "ab" if received else "xb"
        with partial.open(mode) as output:
            while True:
                chunk = response.read(COPY_BUFFER_BYTES)
                if not chunk:
                    break
                received += len(chunk)
                if received > item.size:
                    raise RuntimeError(f"server sent more bytes than declared for {item.name}")
                output.write(chunk)
                digest.update(chunk)
                if received >= next_report:
                    progress(f"download {item.name}: {received}/{item.size} bytes")
                    next_report += GIB
                if max_bytes_per_second > 0:
                    transferred = received - initial_bytes
                    target_elapsed = transferred / max_bytes_per_second
                    delay = target_elapsed - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
            output.flush()
            os.fsync(output.fileno())
    if received != item.size:
        raise RuntimeError(f"short download for {item.name}: expected {item.size}, got {received}")
    actual_hash = digest.hexdigest()
    if item.sha256 and actual_hash != item.sha256:
        raise RuntimeError(f"SHA-256 mismatch for {item.name}")
    progress(f"verified {item.name}: {received} bytes")
    return actual_hash


def import_model(
    store: Path,
    spec: ModelSpec,
    *,
    metadata_only: bool,
    reserve_bytes: int,
    timeout: float,
    dry_run: bool,
    max_bytes_per_second: int = 128 * 1024 * 1024,
) -> Path:
    root = validate_store(store)
    files = selected_files(spec, metadata_only)
    original = root / "models" / "original"
    target = original / target_name(spec, metadata_only)
    if target.exists():
        validate_existing(target, spec, files, metadata_only)
        return target
    if dry_run:
        preflight_disk(root, sum(item.size for item in files), reserve_bytes)
        return target

    download_root = root / "cache" / "imports" / target.name
    download_root.mkdir(parents=True, exist_ok=True)
    if download_root.resolve(strict=True).parent.parent != (root / "cache").resolve(strict=True):
        raise RuntimeError("unsafe importer download-cache path")
    preserved_bytes = 0
    for item in files:
        cached = download_root / item.name
        partial = cached.with_name(cached.name + ".partial")
        if cached.exists():
            preserved_bytes += min(cached.stat().st_size, item.size)
        elif partial.exists():
            preserved_bytes += min(partial.stat().st_size, item.size)
    preflight_disk(root, sum(item.size for item in files) - preserved_bytes, reserve_bytes)
    lock_path = download_root / ".import.lock"
    incoming = original / f".incoming-{target.name}-{os.getpid()}-{time.time_ns()}"
    if incoming.parent.resolve(strict=True) != original.resolve(strict=True):
        raise RuntimeError("unsafe importer staging path")
    try:
        with import_lock(lock_path):
            records: list[dict[str, object]] = []
            cached_files: list[Path] = []
            for item in files:
                cached = download_root / item.name
                partial = cached.with_name(cached.name + ".partial")
                sidecar = cached.with_name(cached.name + ".sha256")
                cached.parent.mkdir(parents=True, exist_ok=True)
                if cached.exists():
                    expected_hash = item.sha256
                    if expected_hash is None and sidecar.is_file():
                        expected_hash = sidecar.read_text(encoding="ascii").strip()
                    if cached.stat().st_size != item.size or not expected_hash or hash_file(cached) != expected_hash:
                        raise RuntimeError(f"cached artifact failed validation; preserved for inspection: {cached}")
                    actual_hash = expected_hash
                    progress(f"reused verified cache {item.name}: {item.size} bytes")
                else:
                    already_received = partial.stat().st_size if partial.exists() else 0
                    preflight_disk(root, item.size - already_received, reserve_bytes)
                    actual_hash = download_file(
                        file_url(spec, item), partial, item, timeout, max_bytes_per_second
                    )
                    os.replace(partial, cached)
                    sidecar_partial = sidecar.with_name(sidecar.name + ".partial")
                    sidecar_partial.write_text(actual_hash + "\n", encoding="ascii", newline="\n")
                    os.replace(sidecar_partial, sidecar)
                records.append({"name": item.name, "size": item.size, "sha256": actual_hash})
                cached_files.append(cached)

            incoming.mkdir(mode=0o700)
            for item, cached in zip(files, cached_files, strict=True):
                destination = incoming / item.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(cached, destination)

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
            incoming.rename(target)

            for cached in cached_files:
                cached.unlink()
                cached.with_name(cached.name + ".sha256").unlink(missing_ok=True)
            for path in target.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IREAD)
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
    parser.add_argument("--max-mib-per-sec", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reserve_gib < 20 or args.timeout <= 0 or args.max_mib_per_sec < 1 or args.max_mib_per_sec > 1024:
        print("ERROR: reserve, timeout or bandwidth limit is outside safe bounds", file=sys.stderr)
        return 64
    lower_process_priority()
    try:
        target = import_model(
            args.store,
            MODELS[args.model],
            metadata_only=args.metadata_only,
            reserve_bytes=args.reserve_gib * GIB,
            timeout=args.timeout,
            dry_run=args.dry_run,
            max_bytes_per_second=args.max_mib_per_sec * 1024 * 1024,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
