#!/usr/bin/env python3
"""Generate a strict Vastloom v1 chunk manifest from an existing model directory.

The tool is offline, standard-library only, rejects symlinks by default, hashes every
file in bounded chunks, and atomically publishes the completed manifest.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_FILES = 100_000
HASH_BUFFER_BYTES = 8 * 1024 * 1024
MAX_U64 = (1 << 64) - 1
BYTE_VALUE = re.compile(
    r"^([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z]*)$"
)
BYTE_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "tb": 1_000_000_000_000,
    "k": 1 << 10,
    "kib": 1 << 10,
    "m": 1 << 20,
    "mib": 1 << 20,
    "g": 1 << 30,
    "gib": 1 << 30,
    "t": 1 << 40,
    "tib": 1 << 40,
}


@dataclass(frozen=True)
class Chunk:
    relative_path: str
    size: int
    sha256: str
    role: str
    dtype: str = "opaque"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--revision", default="local")
    parser.add_argument("--tokenizer-family", default="")
    parser.add_argument("--minimum-ram", default="0")
    parser.add_argument("--recommended-ram", default="0")
    parser.add_argument("--minimum-vram", default="0")
    parser.add_argument("--recommended-vram", default="0")
    parser.add_argument("--dense-bytes", default="0")
    parser.add_argument("--routed-expert-bytes", default="0")
    parser.add_argument("--active-bytes-per-token", default="0")
    parser.add_argument("--estimated-expert-bytes", default="0")
    parser.add_argument("--layer-count", type=int, default=0)
    parser.add_argument("--expert-count", type=int, default=0)
    parser.add_argument("--selected-experts-per-token", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--requires-avx2", action="store_true")
    parser.add_argument("--supports-speculation", action="store_true")
    parser.add_argument("--allow-symlinks", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_byte_value(value: str, field: str) -> int:
    if "\n" in value or "\r" in value or "\t" in value:
        raise ValueError(f"{field} must be a single-line byte value")
    normalized = value.strip().replace("_", "")
    match = BYTE_VALUE.fullmatch(normalized)
    if match is None:
        raise ValueError(f"{field} is not a valid byte value: {value}")
    suffix = match.group(2).lower()
    if suffix not in BYTE_MULTIPLIERS:
        raise ValueError(f"{field} uses an unsupported byte suffix: {suffix}")
    try:
        number = Decimal(match.group(1))
    except InvalidOperation as error:
        raise ValueError(f"{field} is not a valid byte value: {value}") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    result = int((number * BYTE_MULTIPLIERS[suffix]).to_integral_value(rounding=ROUND_HALF_UP))
    if result > MAX_U64:
        raise ValueError(f"{field} exceeds the uint64 byte range")
    return result


def validate_model_fields(args: argparse.Namespace, total: int) -> None:
    byte_fields = {
        "dense-bytes": args.dense_bytes,
        "routed-expert-bytes": args.routed_expert_bytes,
        "active-bytes-per-token": args.active_bytes_per_token,
        "estimated-expert-bytes": args.estimated_expert_bytes,
        "minimum-ram": args.minimum_ram,
        "recommended-ram": args.recommended_ram,
        "minimum-vram": args.minimum_vram,
        "recommended-vram": args.recommended_vram,
    }
    parsed = {name: parse_byte_value(value, name) for name, value in byte_fields.items()}
    for name in ("dense-bytes", "routed-expert-bytes", "active-bytes-per-token"):
        if parsed[name] > total:
            raise ValueError(f"{name} exceeds the checkpoint byte total")
    if parsed["recommended-ram"] and parsed["recommended-ram"] < parsed["minimum-ram"]:
        raise ValueError("recommended-ram is below minimum-ram")
    if parsed["recommended-vram"] and parsed["recommended-vram"] < parsed["minimum-vram"]:
        raise ValueError("recommended-vram is below minimum-vram")

    counts = {
        "layer-count": args.layer_count,
        "expert-count": args.expert_count,
        "selected-experts-per-token": args.selected_experts_per_token,
        "max-context-tokens": args.max_context_tokens,
    }
    if any(value < 0 or value > 0xFFFFFFFF for value in counts.values()):
        raise ValueError("Layer, expert and context counts must fit uint32 and be non-negative")
    has_expert_data = (
        parsed["routed-expert-bytes"] != 0
        or parsed["estimated-expert-bytes"] != 0
        or args.selected_experts_per_token != 0
    )
    if args.expert_count == 0:
        if has_expert_data:
            raise ValueError("Expert data requires --expert-count")
    elif (
        args.layer_count == 0
        or args.max_context_tokens == 0
        or args.selected_experts_per_token == 0
        or args.selected_experts_per_token > args.expert_count
        or parsed["routed-expert-bytes"] == 0
        or parsed["estimated-expert-bytes"] == 0
        or parsed["estimated-expert-bytes"] > parsed["routed-expert-bytes"]
        or parsed["minimum-ram"] == 0
    ):
        raise ValueError(
            "MoE manifests require layers, context, bounded top-k, expert bytes and minimum RAM"
        )


def classify_role(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in {"config.json", "generation_config.json", "params.json"}:
        return "config"
    if "tokenizer" in name or name in {"vocab.json", "merges.txt", "special_tokens_map.json"}:
        return "tokenizer"
    if "router" in name:
        return "router"
    if "embedding" in name or "embed_tokens" in name:
        return "embedding"
    if suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}:
        return "other"
    return "support"


def iter_files(root: Path, output: Path, allow_symlinks: bool) -> Iterable[Path]:
    output_resolved = output.resolve(strict=False)
    count = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        safe_dirnames: list[str] = []
        for dirname in sorted(dirnames):
            candidate = current / dirname
            if candidate.is_symlink():
                if allow_symlinks:
                    raise RuntimeError(
                        f"Directory symlink is unsupported even with --allow-symlinks: {candidate}"
                    )
                raise RuntimeError(f"Refusing directory symlink: {candidate}")
            safe_dirnames.append(dirname)
        dirnames[:] = safe_dirnames

        for filename in sorted(filenames):
            candidate = current / filename
            if candidate.resolve(strict=False) == output_resolved:
                continue
            if candidate.name.endswith((".tmp", ".partial", ".part")):
                continue
            if candidate.is_symlink() and not allow_symlinks:
                raise RuntimeError(f"Refusing file symlink: {candidate}")
            if not candidate.is_file():
                continue
            count += 1
            if count > MAX_FILES:
                raise RuntimeError(f"Model directory exceeds {MAX_FILES:,} files")
            yield candidate


def hash_file(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb", buffering=0) as source:
        while True:
            block = source.read(HASH_BUFFER_BYTES)
            if not block:
                break
            size += len(block)
            hasher.update(block)
    stat_size = path.stat().st_size
    if size != stat_size:
        raise RuntimeError(f"File changed while hashing: {path} ({size} != {stat_size})")
    return size, hasher.hexdigest()


def sanitize_scalar(value: str, field: str) -> str:
    if not value or "\n" in value or "\r" in value or "\t" in value:
        raise ValueError(f"{field} must be a non-empty single-line value without tabs")
    return value


def build_chunks(root: Path, files: Iterable[Path], quiet: bool) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("../") or relative.startswith("/"):
            raise RuntimeError(f"Unsafe relative path: {relative}")
        size, digest = hash_file(path)
        if size <= 0:
            raise RuntimeError(f"Zero-byte files are not accepted as model chunks: {relative}")
        chunks.append(Chunk(relative, size, digest, classify_role(path)))
        if not quiet:
            print(f"[{index}] {relative}: {size:,} bytes {digest[:12]}…", file=sys.stderr)
    chunks.sort(key=lambda chunk: chunk.relative_path)
    return chunks


def render_manifest(args: argparse.Namespace, chunks: list[Chunk]) -> str:
    total = sum(chunk.size for chunk in chunks)
    validate_model_fields(args, total)
    tokenizer_family = (
        sanitize_scalar(args.tokenizer_family, "tokenizer-family") if args.tokenizer_family else ""
    )
    lines = [
        "# Generated by tools/generate_manifest.py; do not edit chunk metadata manually.",
        "format=vastloom.manifest.v1",
        f"model_id={sanitize_scalar(args.model_id, 'model-id')}",
        f"revision={sanitize_scalar(args.revision, 'revision')}",
        f"architecture={sanitize_scalar(args.architecture, 'architecture')}",
        f"tokenizer_family={tokenizer_family}",
        "metadata_only=false",
        f"requires_avx2={'true' if args.requires_avx2 else 'false'}",
        f"supports_speculation={'true' if args.supports_speculation else 'false'}",
        f"model_total_bytes={total}",
        f"required_storage_bytes={total}",
        f"dense_bytes={args.dense_bytes}",
        f"routed_expert_bytes={args.routed_expert_bytes}",
        f"active_bytes_per_token={args.active_bytes_per_token}",
        f"estimated_expert_bytes={args.estimated_expert_bytes}",
        f"minimum_ram_bytes={args.minimum_ram}",
        f"recommended_ram_bytes={args.recommended_ram}",
        f"minimum_vram_bytes={args.minimum_vram}",
        f"recommended_vram_bytes={args.recommended_vram}",
        f"layer_count={args.layer_count}",
        f"expert_count={args.expert_count}",
        f"selected_experts_per_token={args.selected_experts_per_token}",
        f"max_context_tokens={args.max_context_tokens}",
        f"chunk_count={len(chunks)}",
    ]
    for chunk in chunks:
        lines.append(
            "\t".join(
                [
                    "chunk",
                    chunk.relative_path,
                    str(chunk.size),
                    chunk.sha256,
                    chunk.role,
                    chunk.dtype,
                    "-",
                    "-",
                    "true",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def atomic_write(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    root = args.model_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Model root is not a directory: {root}")
    output = args.output.resolve(strict=False)
    chunks = build_chunks(root, iter_files(root, output, args.allow_symlinks), args.quiet)
    if not chunks:
        raise RuntimeError("Model directory contains no eligible files")
    atomic_write(output, render_manifest(args, chunks))
    if not args.quiet:
        print(f"Published {len(chunks)} chunks to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
