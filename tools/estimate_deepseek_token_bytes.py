#!/usr/bin/env python3
"""Offline per-token byte budget for the DeepSeek V4 Flash 0731 `UD-IQ2_XXS` reduced-payload track.

The tool reads the local GGUF headers (a complete first split plus 2 MiB header prefixes, or the
three complete splits), parses the metadata and tensor table itself, and derives the encoded-byte
split between always-active tensors, routed-expert bundles and the six global consumers. It never
opens a payload byte, never hashes, never writes and never touches the network, so the output is a
header-derived plan, not a measurement. Deterministic by construction: tensors are ordered by layer
and name, files by shard index, and no timestamp enters the report.

The GGUF v3 layout is parsed as the native reader does: magic/version/counts, metadata key/value
pairs, tensor infos, then the alignment-padded data offset. Encoded sizes use the same GGML block
table as `src/model/gguf.cpp`, so the totals are expected to equal the pinned catalog arithmetic
(always-active 6,986,088,796 B, routed experts 83,869,302,784 B, one-token expert union
1,965,686,784 B, full-depth streamed 8,951,775,580 B for this profile).

Usage:
    python tools/estimate_deepseek_token_bytes.py
    python tools/estimate_deepseek_token_bytes.py --model <store-artifact-dir> --json

Exit codes: 0 on success, 2 on an unreadable or inconsistent header (fail closed, no partial
report).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import sys

REPORT_FORMAT = "vastloom.deepseek_v4.token_bytes_estimate.v1"
DEFAULT_MODEL = (
    "build/analysis/deepseek-0731/store/models/original/"
    "deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata"
)

MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TENSORS = 4_096
MAX_METADATA_ENTRIES = 4_096
MAX_STRING_BYTES = 1024 * 1024
MAX_RETAINED_STRING_BYTES = 256
MAX_ARRAY_ELEMENTS = 1 << 20
MAX_ARRAY_DEPTH = 8
MAX_DIMENSIONS = 8
MAX_TENSOR_NAME_BYTES = 1_024
MAX_ALIGNMENT = 4_096
DEFAULT_ALIGNMENT = 32

# GGML storage layouts: ggml type id -> (elements per block, bytes per block).
# Kept in step with `ggml_storage_layout` in `src/model/gguf.cpp`; an unknown id is an error
# rather than a guess, because every byte in this report comes from this table.
GGML_STORAGE_LAYOUT: dict[int, tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 40),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
    39: (32, 17),    # MXFP4
}

# GGUF metadata value types; only scalars are retained, arrays are validated and skipped.
_SCALAR_TYPES: dict[int, tuple[str, int]] = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_STRING_TYPE = 8
_ARRAY_TYPE = 9

_EXPERT_SUFFIXES = {
    ".ffn_gate_exps.weight": "gate",
    ".ffn_up_exps.weight": "up",
    ".ffn_down_exps.weight": "down",
}
_BLOCK_PREFIX = "blk."
_SHARD_PATTERN = re.compile(r"-(\d{5})-of-(\d{5})\.gguf(\.header)?$", re.IGNORECASE)

ARCHITECTURE_KEYS = (
    "deepseek4.block_count",
    "deepseek4.hash_layer_count",
    "deepseek4.expert_count",
    "deepseek4.expert_used_count",
    "deepseek4.embedding_length",
    "deepseek4.expert_feed_forward_length",
)


class GgufError(RuntimeError):
    """Raised when a local GGUF header cannot be read or does not close as expected."""


class _Cursor:
    """Little-endian bounded reader over an in-memory header slice."""

    __slots__ = ("data", "position")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.position + count > len(self.data):
            raise GgufError(
                f"truncated GGUF header at byte {self.position}: "
                f"{count} more byte(s) requested, {len(self.data) - self.position} available"
            )
        start = self.position
        self.position += count
        return self.data[start:start + count]

    def unpack(self, code: str, size: int) -> int | float | bool:
        return struct.unpack(code, self.take(size))[0]

    def u32(self) -> int:
        return int(self.unpack("<I", 4))

    def u64(self) -> int:
        return int(self.unpack("<Q", 8))

    def string(self, maximum: int) -> bytes:
        length = self.u64()
        if length > maximum:
            raise GgufError(f"GGUF string of {length} bytes exceeds the {maximum}-byte bound")
        return self.take(length)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    layer: int | None
    dimensions: tuple[int, ...]
    ggml_type: int
    offset: int
    element_count: int
    encoded_bytes: int


@dataclass(frozen=True)
class GgufHeader:
    path: Path
    local_bytes: int
    version: int
    header_bytes: int
    alignment: int
    tensor_count: int
    metadata_count: int
    metadata: dict[str, object]
    tensors: tuple[TensorInfo, ...]


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def _element_count(dimensions: tuple[int, ...], name: str) -> int:
    count = 1
    for dimension in dimensions:
        if dimension == 0:
            raise GgufError(f"tensor {name} has an empty dimension")
        count *= dimension
    return count


def _drop_value(cursor: _Cursor, value_type: int, depth: int) -> None:
    if value_type in _SCALAR_TYPES:
        cursor.take(_SCALAR_TYPES[value_type][1])
        return
    if value_type == _STRING_TYPE:
        cursor.string(MAX_STRING_BYTES)
        return
    if value_type != _ARRAY_TYPE:
        raise GgufError(f"unsupported GGUF metadata value type {value_type}")
    if depth >= MAX_ARRAY_DEPTH:
        raise GgufError("GGUF metadata array nesting exceeds the configured depth")
    element_type = cursor.u32()
    count = cursor.u64()
    if count > MAX_ARRAY_ELEMENTS:
        raise GgufError(f"GGUF metadata array of {count} elements exceeds the configured bound")
    for _ in range(count):
        _drop_value(cursor, element_type, depth + 1)


def _read_value(cursor: _Cursor, value_type: int) -> tuple[object, bool]:
    """Return (value, retained). Non-scalar values are dropped and reported as retained=False."""
    if value_type in _SCALAR_TYPES:
        code, size = _SCALAR_TYPES[value_type]
        return cursor.unpack(code, size), True
    if value_type == _STRING_TYPE:
        raw = cursor.string(MAX_STRING_BYTES)
        if len(raw) > MAX_RETAINED_STRING_BYTES:
            return None, False
        return raw.decode("utf-8", errors="replace"), True
    if value_type == _ARRAY_TYPE:
        _drop_value(cursor, value_type, 0)
        return None, False
    raise GgufError(f"unsupported GGUF metadata value type {value_type}")


def parse_gguf_header(path: Path) -> GgufHeader:
    try:
        local_bytes = path.stat().st_size
    except OSError as error:
        raise GgufError(f"unable to stat {path}: {error}") from error
    if local_bytes < 24:
        raise GgufError(f"{path} is too small to carry a GGUF header")
    try:
        with path.open("rb") as handle:
            data = handle.read(min(local_bytes, MAX_HEADER_BYTES))
    except OSError as error:
        raise GgufError(f"unable to read {path}: {error}") from error

    cursor = _Cursor(data)
    if cursor.take(4) != b"GGUF":
        raise GgufError(f"{path} does not start with the GGUF magic")
    version = cursor.u32()
    if version != 3:
        raise GgufError(f"{path} has GGUF version {version}; only little-endian v3 is supported")
    tensor_count = cursor.u64()
    metadata_count = cursor.u64()
    if tensor_count > MAX_TENSORS:
        raise GgufError(f"{path} declares {tensor_count} tensors, above the {MAX_TENSORS} bound")
    if metadata_count > MAX_METADATA_ENTRIES:
        raise GgufError(
            f"{path} declares {metadata_count} metadata entries, above the "
            f"{MAX_METADATA_ENTRIES} bound"
        )

    metadata: dict[str, object] = {}
    for _ in range(metadata_count):
        key = cursor.string(MAX_STRING_BYTES).decode("ascii", errors="replace")
        if not key or key in metadata:
            raise GgufError(f"{path} has an empty or duplicated metadata key {key!r}")
        value_type = cursor.u32()
        value, retained = _read_value(cursor, value_type)
        if retained:
            metadata[key] = value

    alignment_value = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    if not isinstance(alignment_value, int) or isinstance(alignment_value, bool):
        raise GgufError(f"{path} has a non-integral general.alignment")
    alignment = int(alignment_value)
    if alignment < 8 or alignment > MAX_ALIGNMENT or alignment % 8 != 0:
        raise GgufError(f"{path} has general.alignment {alignment} outside the safe bounds")

    tensors: list[TensorInfo] = []
    names: set[str] = set()
    for _ in range(tensor_count):
        name = cursor.string(MAX_TENSOR_NAME_BYTES).decode("utf-8", errors="replace")
        dimensions_count = cursor.u32()
        if not name or name in names or dimensions_count == 0 or dimensions_count > MAX_DIMENSIONS:
            raise GgufError(f"{path} has an invalid or duplicated tensor name {name!r}")
        names.add(name)
        dimensions = tuple(cursor.u64() for _ in range(dimensions_count))
        ggml_type = cursor.u32()
        offset = cursor.u64()
        if ggml_type not in GGML_STORAGE_LAYOUT:
            raise GgufError(f"{path} tensor {name} uses unsupported GGML type {ggml_type}")
        if offset % alignment != 0:
            raise GgufError(f"{path} tensor {name} offset {offset} is not aligned")
        layout = GGML_STORAGE_LAYOUT[ggml_type]
        elements = _element_count(dimensions, name)
        if dimensions[0] % layout[0] != 0 or elements % layout[0] != 0:
            raise GgufError(f"{path} tensor {name} shape is incompatible with its GGML block type")
        tensors.append(
            TensorInfo(
                name=name,
                layer=_tensor_layer(name),
                dimensions=dimensions,
                ggml_type=ggml_type,
                offset=offset,
                element_count=elements,
                encoded_bytes=elements // layout[0] * layout[1],
            )
        )

    header_bytes = _align_up(cursor.position, alignment)
    if header_bytes > local_bytes:
        raise GgufError(
            f"{path} declares a {header_bytes}-byte header but only {local_bytes} bytes are local; "
            "the metadata store does not cover this GGUF header"
        )
    return GgufHeader(
        path=path,
        local_bytes=local_bytes,
        version=version,
        header_bytes=header_bytes,
        alignment=alignment,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        metadata=metadata,
        tensors=tuple(tensors),
    )


def _tensor_layer(name: str) -> int | None:
    if not name.startswith(_BLOCK_PREFIX):
        return None
    separator = name.find(".", len(_BLOCK_PREFIX))
    if separator <= len(_BLOCK_PREFIX):
        return None
    block = name[len(_BLOCK_PREFIX):separator]
    if not block.isdigit():
        return None
    return int(block)


def _expert_projection(name: str) -> str | None:
    for suffix, projection in _EXPERT_SUFFIXES.items():
        if name.endswith(suffix):
            return projection
    return None


@dataclass(frozen=True)
class Shard:
    index: int
    header: GgufHeader
    kind: str


def _discover_shards(model: Path) -> list[Shard]:
    if not model.is_dir():
        raise GgufError(f"model path {model} is not a directory")
    candidates = sorted(model.rglob("*.gguf")) + sorted(model.rglob("*.gguf.header"))
    by_index: dict[int, list[Path]] = {}
    declared_counts: dict[int, int] = {}
    for candidate in candidates:
        match = _SHARD_PATTERN.search(candidate.name)
        if match is None:
            continue
        index = int(match.group(1))
        count = int(match.group(2))
        if declared_counts.setdefault(index, count) != count:
            raise GgufError(f"shard {index} declares conflicting shard counts {count} and "
                            f"{declared_counts[index]}")
        by_index.setdefault(index, []).append(candidate)

    if not by_index:
        raise GgufError(f"model path {model} contains no GGUF shard or header-prefix objects")
    indices = sorted(by_index)
    expected = list(range(1, len(indices) + 1))
    if indices != expected:
        raise GgufError(f"model path {model} has non-contiguous shard indices {indices}")

    shards: list[Shard] = []
    for index in indices:
        paths = by_index[index]
        complete = [path for path in paths if not path.name.endswith(".header")]
        if len(complete) > 1:
            raise GgufError(f"shard {index} has several complete GGUF objects: {paths}")
        if not complete and len(paths) > 1:
            raise GgufError(f"shard {index} has several header prefixes: {paths}")
        chosen = complete[0] if complete else paths[0]
        header = parse_gguf_header(chosen)
        kind = "complete" if complete else "header-prefix"
        shards.append(Shard(index, header, kind))
    return shards


def _metadata_owner(shards: list[Shard]) -> GgufHeader:
    candidates = [(shard.header.metadata_count, -shard.index, shard.header) for shard in shards]
    candidates = [candidate for candidate in candidates if candidate[0]]
    if not candidates:
        raise GgufError("no shard carries GGUF metadata")
    owner = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]
    for shard in shards:
        for key in ("general.architecture",):
            value = shard.header.metadata.get(key)
            if value is not None and value != owner.metadata.get(key):
                raise GgufError(f"shard {shard.index} disagrees with the metadata shard on {key}")
        split_count = shard.header.metadata.get("split.count")
        if split_count is not None and split_count != len(shards):
            raise GgufError(f"shard {shard.index} declares split.count {split_count}, "
                            f"but {len(shards)} shard object(s) are local")
    return owner


def _required_unsigned(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise GgufError(f"required unsigned metadata {key} is missing")
    return value


@dataclass(frozen=True)
class LayerBudget:
    layer: int
    trunk_bytes: int
    trunk_tensors: int
    routed_bytes: int
    bundle_bytes: int
    bundle_tensors: int


def _collect_budget(shards: list[Shard]) -> dict[str, object]:
    owner = _metadata_owner(shards)
    layers = _required_unsigned(owner.metadata, "deepseek4.block_count")
    experts = _required_unsigned(owner.metadata, "deepseek4.expert_count")
    top_k = _required_unsigned(owner.metadata, "deepseek4.expert_used_count")
    hash_layers = _required_unsigned(owner.metadata, "deepseek4.hash_layer_count")
    if layers == 0 or experts == 0 or top_k == 0 or top_k > experts or hash_layers > layers:
        raise GgufError("DeepSeek architecture metadata is unsupported or inconsistent")

    trunk: dict[int, int] = {}
    trunk_tensors: dict[int, int] = {}
    routed: dict[int, int] = {}
    routed_tensors: dict[int, int] = {}
    globals_named: list[TensorInfo] = []
    total_tensors = 0

    for shard in shards:
        for tensor in shard.header.tensors:
            total_tensors += 1
            if tensor.layer is None:
                globals_named.append(tensor)
                continue
            projection = _expert_projection(tensor.name)
            if projection is None:
                trunk[tensor.layer] = trunk.get(tensor.layer, 0) + tensor.encoded_bytes
                trunk_tensors[tensor.layer] = trunk_tensors.get(tensor.layer, 0) + 1
            else:
                routed[tensor.layer] = routed.get(tensor.layer, 0) + tensor.encoded_bytes
                routed_tensors[tensor.layer] = routed_tensors.get(tensor.layer, 0) + 1

    layer_ids = sorted(trunk)
    if layer_ids != list(range(layers)):
        raise GgufError(f"layer trunk tensors cover {layer_ids}, not layers 0..{layers - 1}")
    if sorted(routed) != list(range(layers)):
        raise GgufError(f"routed-expert tensors cover {sorted(routed)}, not layers 0..{layers - 1}")
    for layer in range(layers):
        if routed_tensors[layer] != len(_EXPERT_SUFFIXES):
            raise GgufError(f"layer {layer} has {routed_tensors[layer]} routed-expert containers, "
                            f"expected {len(_EXPERT_SUFFIXES)}")

    budgets: list[LayerBudget] = []
    for layer in range(layers):
        if routed[layer] % experts != 0:
            raise GgufError(f"layer {layer} routed-expert bytes {routed[layer]} do not split into "
                            f"{experts} equal bundles")
        budgets.append(
            LayerBudget(
                layer=layer,
                trunk_bytes=trunk[layer],
                trunk_tensors=trunk_tensors[layer],
                routed_bytes=routed[layer],
                bundle_bytes=routed[layer] // experts,
                bundle_tensors=routed_tensors[layer],
            )
        )

    return {
        "layers": layers,
        "experts": experts,
        "top_k": top_k,
        "hash_layers": hash_layers,
        "budgets": budgets,
        "globals": sorted(globals_named, key=lambda tensor: tensor.name),
        "total_tensors": total_tensors,
        "metadata_owner": owner,
    }


def _build_report(model: Path) -> dict[str, object]:
    shards = _discover_shards(model)
    budget = _collect_budget(shards)
    budgets: list[LayerBudget] = budget["budgets"]  # type: ignore[assignment]
    globals_named: list[TensorInfo] = budget["globals"]  # type: ignore[assignment]
    owner: GgufHeader = budget["metadata_owner"]  # type: ignore[assignment]
    top_k = int(budget["top_k"])

    trunk_total = sum(entry.trunk_bytes for entry in budgets)
    trunk_tensors_total = sum(entry.trunk_tensors for entry in budgets)
    routed_total = sum(entry.routed_bytes for entry in budgets)
    globals_bytes = sum(tensor.encoded_bytes for tensor in globals_named)
    always_active = trunk_total + globals_bytes
    metadata_bytes = sum(shard.header.local_bytes for shard in shards)
    union_bytes = sum(top_k * entry.bundle_bytes for entry in budgets)
    full_depth = trunk_total + globals_bytes + union_bytes

    layer_min = min(budgets, key=lambda entry: (entry.trunk_bytes, entry.layer))
    layer_max = max(budgets, key=lambda entry: (entry.trunk_bytes, -entry.layer))
    bundle_min = min(budgets, key=lambda entry: (entry.bundle_bytes, entry.layer))
    bundle_max = max(budgets, key=lambda entry: (entry.bundle_bytes, -entry.layer))

    per_layer = []
    cumulative = 0
    for entry in budgets:
        token_layer_bytes = entry.trunk_bytes + top_k * entry.bundle_bytes
        cumulative += token_layer_bytes
        per_layer.append(
            {
                "layer": entry.layer,
                "trunk_bytes": entry.trunk_bytes,
                "trunk_tensors": entry.trunk_tensors,
                "routed_expert_bytes": entry.routed_bytes,
                "bundle_bytes": entry.bundle_bytes,
                "bundle_tensors": entry.bundle_tensors,
                "token_expert_bytes": top_k * entry.bundle_bytes,
                "token_layer_bytes": token_layer_bytes,
                "token_cumulative_bytes": cumulative,
            }
        )

    return {
        "format": REPORT_FORMAT,
        "read_only": True,
        "design_only": True,
        "model": str(model.resolve()),
        "sources": [
            {
                "object": shard.header.path.name,
                "kind": shard.kind,
                "local_bytes": shard.header.local_bytes,
                "header_bytes": shard.header.header_bytes,
                "gguf_version": shard.header.version,
                "tensor_count": shard.header.tensor_count,
                "metadata_count": shard.header.metadata_count,
            }
            for shard in shards
        ],
        "architecture": {
            "layer_count": int(budget["layers"]),
            "hash_layer_count": int(budget["hash_layers"]),
            "expert_count": int(budget["experts"]),
            "top_k": top_k,
            "embedding_length": owner.metadata.get("deepseek4.embedding_length"),
            "expert_feed_forward_length": owner.metadata.get("deepseek4.expert_feed_forward_length"),
            "physical_tensors": int(budget["total_tensors"]),
        },
        "always_active": {
            "trunk_bytes": trunk_total,
            "trunk_tensors": trunk_tensors_total,
            "layer_trunk_min": {"layer": layer_min.layer, "bytes": layer_min.trunk_bytes},
            "layer_trunk_max": {"layer": layer_max.layer, "bytes": layer_max.trunk_bytes},
            "global_bytes": globals_bytes,
            "global_tensors": len(globals_named),
            "global_max_tensor_bytes": max(
                (tensor.encoded_bytes for tensor in globals_named), default=0
            ),
            "globals": [
                {
                    "name": tensor.name,
                    "ggml_type": tensor.ggml_type,
                    "dimensions": list(tensor.dimensions),
                    "encoded_bytes": tensor.encoded_bytes,
                }
                for tensor in globals_named
            ],
            "payload_bytes": always_active,
            "metadata_store_bytes": metadata_bytes,
            "local_footprint_bytes": always_active + metadata_bytes,
        },
        "routed_experts": {
            "expert_count": int(budget["experts"]),
            "bundle_count": int(budget["experts"]) * len(budgets),
            "routed_total_bytes": routed_total,
            "bundle_min": {"layer": bundle_min.layer, "bytes": bundle_min.bundle_bytes},
            "bundle_max": {"layer": bundle_max.layer, "bytes": bundle_max.bundle_bytes},
        },
        "token": {
            "top_k": top_k,
            "expert_union_bytes": union_bytes,
            "full_depth_streamed_bytes": full_depth,
        },
        "per_layer": per_layer,
    }


def _format_bytes(value: int) -> str:
    return f"{value:,}"


def _text_report(report: dict[str, object]) -> str:
    architecture = report["architecture"]
    always = report["always_active"]
    routed = report["routed_experts"]
    token = report["token"]
    assert isinstance(architecture, dict) and isinstance(always, dict)
    assert isinstance(routed, dict) and isinstance(token, dict)

    lines = [
        "DeepSeek V4 Flash 0731 UD-IQ2_XXS - offline token byte budget (header-derived)",
        f"format: {report['format']}",
        f"model:  {report['model']}",
        "",
        "local header objects:",
    ]
    for source in report["sources"]:  # type: ignore[union-attr]
        assert isinstance(source, dict)
        lines.append(
            f"  {source['object']}  {source['kind']}  local={_format_bytes(source['local_bytes'])} B"
            f"  header={_format_bytes(source['header_bytes'])} B"
            f"  tensors={source['tensor_count']}"
        )
    lines.extend(
        [
            "",
            f"architecture: layers={architecture['layer_count']}"
            f" experts={architecture['expert_count']} top_k={architecture['top_k']}"
            f" hash_layers={architecture['hash_layer_count']}"
            f" tensors={_format_bytes(architecture['physical_tensors'])}",
            f"embedding_length={architecture['embedding_length']}"
            f" expert_feed_forward_length={architecture['expert_feed_forward_length']}",
            "",
            "always-active payload (pre-downloaded, local disk):",
            f"  trunk total            {_format_bytes(always['trunk_bytes']):>18} B"
            f"  ({architecture['layer_count']} layers,"
            f" {_format_bytes(always['trunk_tensors'])} tensors)",
            f"  globals                {_format_bytes(always['global_bytes']):>18} B"
            f"  ({always['global_tensors']} tensors, largest"
            f" {_format_bytes(always['global_max_tensor_bytes'])} B)",
            f"  always-active total    {_format_bytes(always['payload_bytes']):>18} B",
            f"  metadata store         {_format_bytes(always['metadata_store_bytes']):>18} B",
            f"  local footprint        {_format_bytes(always['local_footprint_bytes']):>18} B"
            "  (always-active + metadata store)",
            "",
            "routed experts (fetched on demand, never pre-downloaded):",
            f"  routed total           {_format_bytes(routed['routed_total_bytes']):>18} B"
            f"  ({_format_bytes(routed['bundle_count'])} bundles)",
            f"  bundle min             {_format_bytes(routed['bundle_min']['bytes']):>18} B"
            f"  (layer {routed['bundle_min']['layer']})",
            f"  bundle max             {_format_bytes(routed['bundle_max']['bytes']):>18} B"
            f"  (layer {routed['bundle_max']['layer']})",
            f"  one-token expert union {_format_bytes(token['expert_union_bytes']):>18} B"
            f"  ({token['top_k']} bundles x {architecture['layer_count']} layers)",
            f"  full-depth streamed    {_format_bytes(token['full_depth_streamed_bytes']):>18} B"
            "  (trunk + globals + one-token experts)",
            "",
            "per-layer budget (bytes):",
            "  layer  trunk        tensors  bundle      6x bundle   token layer  cumulative",
        ]
    )
    for entry in report["per_layer"]:  # type: ignore[union-attr]
        assert isinstance(entry, dict)
        lines.append(
            f"  {entry['layer']:>5}  {_format_bytes(entry['trunk_bytes']):>11}"
            f"  {entry['trunk_tensors']:>7}  {_format_bytes(entry['bundle_bytes']):>11}"
            f"  {_format_bytes(entry['token_expert_bytes']):>11}"
            f"  {_format_bytes(entry['token_layer_bytes']):>11}"
            f"  {_format_bytes(entry['token_cumulative_bytes']):>12}"
        )
    lines.extend(
        [
            "",
            "Notes: all bytes are encoded GGUF tensor bytes from the local headers; no payload",
            "byte, hash, network request or write is involved. `bundle` is one complete routed",
            "expert (gate + up + down packed slices) and `6x bundle` is one token's cost for that",
            "layer before any cache. The cumulative column ends at the trunk plus one-token expert",
            "total; the full-depth streamed figure adds the globals to it.",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(DEFAULT_MODEL),
        help="metadata store artifact directory, or any directory holding the GGUF shard headers",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = _build_report(args.model)
    except GgufError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
