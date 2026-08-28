#!/usr/bin/env python3
"""Drive the canonical Vastloom HF importer for Qwen3.8-Flash-Next-FP8.

This is a thin wrapper around ``import_hf_model.import_model``: it discovers the
exact file set from the Hugging Face repo tree (so the pinned spec never drifts
from the repository) and downloads the full 125B FP8 checkpoint through the
project's verified, bounded, atomic import path.

Usage:
    python tools/import_qwen38_fp8.py --store <vastloom-store> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_hf_model import (  # noqa: E402
    GIB,
    FileSpec,
    ModelSpec,
    import_model,
)

MODEL_ID = "unsloth/Qwen3.8-Flash-Next-FP8"
REVISION = "main"
TREE_URL = "https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
META_SUFFIXES = (".json", ".txt")


def fetch_tree(repo: str) -> list[dict]:
    url = TREE_URL.format(repo=repo, rev=REVISION)
    headers = {"User-Agent": "vastloom-local-ai/0.4-importer"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def build_spec(repo: str, gguf_dir: str | None) -> ModelSpec:
    tree = fetch_tree(repo)
    files: list[FileSpec] = []
    for entry in sorted(tree, key=lambda item: item["path"]):
        path = entry["path"]
        size = int(entry["size"])
        if gguf_dir is not None:
            # GGUF mode: every shard under the quant directory is a weight file.
            if path.startswith(gguf_dir) and path.endswith(".gguf"):
                files.append(FileSpec(path, size, None, True))
            continue
        if path.startswith("model-") and path.endswith(".safetensors"):
            files.append(FileSpec(path, size, None, True))
        elif path == "model.safetensors.index.json":
            files.append(FileSpec(path, size, None, False))
        elif any(path.endswith(suffix) for suffix in META_SUFFIXES) and "safetensors" not in path:
            files.append(FileSpec(path, size, None, False))
    if not files:
        raise RuntimeError("empty repository selection")
    alias = (repo.split("/")[-1] + ("-gguf" if gguf_dir else "-fp8")).lower()
    artifact = alias
    if gguf_dir:
        artifact += "-" + gguf_dir.strip("/").lower().replace("/", "-")
    return ModelSpec(alias=alias, model_id=repo, revision=REVISION, files=tuple(files),
                     artifact_name=artifact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--repo", default=MODEL_ID)
    parser.add_argument("--gguf-dir", default=None,
                        help="include only *.gguf under this repo directory (e.g. Q8_0/)")
    parser.add_argument("--reserve-gib", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-mib-per-sec", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = build_spec(args.repo, args.gguf_dir)
    weight_bytes = sum(item.size for item in spec.files if item.weight)
    print(f"spec: {len(spec.files)} files, {weight_bytes / GIB:.1f} GiB weights", file=sys.stderr, flush=True)
    try:
        target = import_model(
            args.store,
            spec,
            metadata_only=False,
            reserve_bytes=args.reserve_gib * GIB,
            timeout=args.timeout,
            dry_run=args.dry_run,
            max_bytes_per_second=args.max_mib_per_sec * 1024 * 1024,
            allow_large_download=True,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
