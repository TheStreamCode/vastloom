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

    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensor_names), 18))
    append_string(header, "general.architecture")
    header.extend(struct.pack("<I", 8))  # STRING
    append_string(header, "deepseek4")
    append_string(header, "general.alignment")
    header.extend(struct.pack("<II", 4, 32))  # UINT32

    scalar_metadata = {
        "deepseek4.block_count": 43,
        "deepseek4.hash_layer_count": 0,
        "deepseek4.context_length": 1_048_576,
        "deepseek4.embedding_length": 4_096,
        "deepseek4.attention.head_count": 64,
        "deepseek4.attention.key_length": 512,
        "deepseek4.attention.value_length": 512,
        "deepseek4.rope.dimension_count": 64,
        "deepseek4.attention.sliding_window": 128,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_used_count": 6,
        "deepseek4.expert_feed_forward_length": 2_048,
        "deepseek4.attention.indexer.head_count": 64,
        "deepseek4.attention.indexer.key_length": 128,
        "deepseek4.hyper_connection.count": 4,
    }
    for key, value in scalar_metadata.items():
        append_string(header, key)
        header.extend(struct.pack("<II", 4, value))  # UINT32
    append_string(header, "deepseek4.attention.compress_ratios")
    ratios = [0, 0] + [4 if layer % 2 == 0 else 128 for layer in range(2, 43)]
    header.extend(struct.pack("<IIQ", 9, 5, len(ratios)))  # ARRAY of INT32
    header.extend(struct.pack(f"<{len(ratios)}i", *ratios))

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
        require(catalog["execution_supported"] is False,
                "GGUF inspection made an execution claim", catalogued)
        require(catalog["physical_tensors"] == 130, "catalog tensor count mismatch", catalogued)
        require(catalog["logical_tensors"] == 33_025, "logical tensor count mismatch", catalogued)
        require(catalog["logical_experts"] == 11_008, "logical expert count mismatch", catalogued)
        require(catalog["routed_expert_bytes"] == 1_122_816, "routed bytes mismatch", catalogued)
        require(catalog["always_active_bytes"] == 4, "always-active bytes mismatch", catalogued)
        require(catalog["no_cache_expert_bytes_per_token"] == 26_316,
                "no-cache bytes/token mismatch", catalogued)
        working_set = catalog["working_set"]
        require(working_set["context_tokens"] == 4_096, "context assumption mismatch", catalogued)
        require(working_set["persistent_kv_bytes"] == 33_816_576,
                "persistent KV accounting mismatch", catalogued)
        require(working_set["compression_state_bytes"] == 12_206_080,
                "compressor state accounting mismatch", catalogued)
        require(working_set["minimum_expert_tier_bytes"] == 102,
                "expert tier accounting mismatch", catalogued)
        require(working_set["runtime_state_bytes"] == 115_622_936,
                "streamed-trunk runtime state mismatch", catalogued)
        require(working_set["minimum_streamed_trunk_cpu_ram_bytes"] == 115_623_038,
                "streamed-trunk CPU floor mismatch", catalogued)
        require(working_set["minimum_cpu_streaming_ram_bytes"] == 115_623_042,
                "resident-trunk CPU floor mismatch", catalogued)
        admission = catalog["admission"]
        if admission["evaluated"]:
            require(admission["capacity_plan_candidate"] is True,
                    "small fixture should pass a live admission gate", catalogued)

        bad_context = run(
            args.vastloom,
            "deepseek-gguf-inspect",
            str(model),
            "--declared-sizes",
            str(declared_size),
            "--context",
            "130",
            "--json",
        )
        require(bad_context.returncode != 0 and "divisible" in bad_context.stderr,
                "incompatible compressed context was accepted", bad_context)

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

        unimported = run(
            args.vastloom,
            "deepseek-gguf-checkpoint",
            str(model),
            "--json",
        )
        require(unimported.returncode != 0,
                "raw GGUF path bypassed the immutable import contract", unimported)

        missing_probe_layer = run(
            args.vastloom,
            "deepseek-gguf-probe-plan",
            str(model),
            "--experts",
            "0,1,2,3,4,5",
        )
        require(missing_probe_layer.returncode != 0 and "requires --layer" in missing_probe_layer.stderr,
                "probe planner accepted an implicit layer", missing_probe_layer)

        missing_probe_experts = run(
            args.vastloom,
            "deepseek-gguf-probe-plan",
            str(model),
            "--layer",
            "0",
        )
        require(
            missing_probe_experts.returncode != 0
            and "either --experts or --route-artifact" in missing_probe_experts.stderr,
            "probe planner accepted a missing selected-expert set", missing_probe_experts)

        # Routing authority comes from one source or the other, never both and never half of one.
        both_sources = run(
            args.vastloom,
            "deepseek-gguf-probe-plan",
            str(model),
            "--layer",
            "0",
            "--experts",
            "0,1,2,3,4,5",
            "--route-artifact",
            str(model),
            "--token",
            "0",
        )
        require(
            both_sources.returncode != 0
            and "either --experts or --route-artifact" in both_sources.stderr,
            "probe planner accepted two competing routing sources", both_sources)

        route_without_token = run(
            args.vastloom,
            "deepseek-gguf-probe-plan",
            str(model),
            "--layer",
            "0",
            "--route-artifact",
            str(model),
        )
        require(
            route_without_token.returncode != 0
            and "must be supplied together" in route_without_token.stderr,
            "probe planner accepted a route artifact without a token", route_without_token)

        token_without_route = run(
            args.vastloom,
            "deepseek-gguf-probe-plan",
            str(model),
            "--layer",
            "0",
            "--token",
            "0",
        )
        require(
            token_without_route.returncode != 0
            and "must be supplied together" in token_without_route.stderr,
            "probe planner accepted a token without a route artifact", token_without_route)

        malformed_probe_experts = run(
            args.vastloom,
            "deepseek-gguf-probe-plan",
            str(model),
            "--layer",
            "0",
            "--experts",
            "0,,2",
        )
        require(malformed_probe_experts.returncode != 0 and "unsigned integers" in malformed_probe_experts.stderr,
                "probe planner accepted a malformed selected-expert set", malformed_probe_experts)

    print("DeepSeek GGUF CLI tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
