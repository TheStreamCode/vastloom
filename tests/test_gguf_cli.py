#!/usr/bin/env python3
"""Deterministic CLI coverage for bounded GGUF and DeepSeek packed-expert catalogs."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


def append_string(buffer: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buffer.extend(struct.pack("<Q", len(encoded)))
    buffer.extend(encoded)


def build_fixture(path: Path) -> int:
    tensor_names = [
        f"blk.{layer}.ffn_{projection}_exps.weight"
        for layer in range(43)
        for projection in ("down", "gate", "up")
    ]
    tensor_names.append("output.weight")

    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensor_names), 2))
    append_string(header, "general.architecture")
    header.extend(struct.pack("<I", 8))  # STRING
    append_string(header, "deepseek4")
    append_string(header, "general.alignment")
    header.extend(struct.pack("<II", 4, 32))  # UINT32

    offset = 0
    for name in tensor_names:
        append_string(header, name)
        if name == "output.weight":
            header.extend(struct.pack("<IQIQ", 1, 1, 0, offset))  # F32
            offset += 4
        else:
            header.extend(struct.pack("<IQQQIQ", 3, 32, 1, 256, 8, offset))  # Q8_0
            offset += 8_704
    while len(header) % 32:
        header.append(0)
    path.write_bytes(header + bytes(offset))
    return len(header) + offset


def run(executable: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *arguments],
        text=True,
        encoding="utf-8",
        errors="strict",
        capture_output=True,
        check=False,
        timeout=30,
    )


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str]) -> None:
    if not condition:
        raise AssertionError(
            f"{message}\nreturncode={process.returncode}\nstdout={process.stdout!r}\nstderr={process.stderr!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="vastloom-gguf-cli-") as temporary:
        model = Path(temporary) / "DeepSeek-à.gguf"
        declared_size = build_fixture(model)

        inspected = run(args.vastloom, "inspect", str(model), "--tensor-limit", "1", "--json")
        require(inspected.returncode == 0, "bounded GGUF inspection failed", inspected)
        inspection = json.loads(inspected.stdout)
        require(inspection["architecture"] == "deepseek4", "architecture mismatch", inspected)
        require(inspection["tensor_count"] == 130, "physical tensor count mismatch", inspected)
        require(len(inspection["tensors"]) == 1, "tensor output bound mismatch", inspected)

        catalogued = run(
            args.vastloom,
            "deepseek-gguf-inspect",
            str(model),
            "--declared-sizes",
            str(declared_size),
            "--revision",
            "fixture",
            "--json",
        )
        require(catalogued.returncode == 0, "DeepSeek GGUF catalog failed", catalogued)
        catalog = json.loads(catalogued.stdout)
        require(catalog["metadata_only"] is True, "catalog was not metadata-only", catalogued)
        require(catalog["physical_tensors"] == 130, "catalog tensor count mismatch", catalogued)
        require(catalog["logical_tensors"] == 33_025, "logical tensor count mismatch", catalogued)
        require(catalog["logical_experts"] == 11_008, "logical expert count mismatch", catalogued)
        require(catalog["routed_expert_bytes"] == 1_122_816, "routed bytes mismatch", catalogued)
        require(catalog["always_active_bytes"] == 4, "always-active bytes mismatch", catalogued)
        require(catalog["no_cache_expert_bytes_per_token"] == 26_316,
                "no-cache bytes/token mismatch", catalogued)

        rejected = run(
            args.vastloom,
            "deepseek-gguf-inspect",
            str(model),
            "--declared-sizes",
            f"{declared_size},{declared_size}",
            "--json",
        )
        require(rejected.returncode != 0 and "differs from shard count" in rejected.stderr,
                "mismatched declared sizes were accepted", rejected)

    print("DeepSeek GGUF CLI tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
