#!/usr/bin/env python3
"""Independent NumPy oracle for DeepSeek V4 Flash 0731 layers 0-2 (UD-IQ2_XXS).

The native ``deepseek-layer-probe`` executes layer 0 from three imported range
artifacts; ``deepseek-chain-probe`` executes the first two or three hash-routed
layers, each from its own trunk and selected-expert artifacts. This tool reads
the same bytes through its own GGUF tensor-table parser and its own
quantized-block decoders, runs the documented layer equations in Python, and
writes the logits plus per-stage checksums. It shares no code with the C++
runtime: the only inputs are the metadata store, the artifact manifests and the
artifact payload files.

Three modes preserve one implementation of the layer:

* one layer (the historical CLI, unchanged): layer 0 from a single trunk and
  expert artifact, one logits vector and 66 stage checksums;
* chain (``--layers N --trunk L=PATH --experts L=PATH``): layers 0..N-1 run in
  sequence, the pre-normalization hidden state of each layer feeds the next,
  and the shared Hyper-Connection head plus LM head are evaluated at every
  requested prefix depth;
* prompt prefill (``--tokens ID,ID,...``): the same chain over a causal prompt of
  one to sixteen tokens. Embedding rows are read for every prompt token, the
  attention window and the compression windows advance with the position, the
  learned indexer selects compressed slots per token, each layer routes every
  token through its own hash-table row, and the shared head plus LM head are
  evaluated on the last position. ``--experts L=PATH`` may then repeat per
  layer: one artifact per prompt-token route, merged into one bundle set.

Layer 1 is structurally the layer-0 graph on its own weights and routing table.
Layer 2 carries compression ratio 4: the attention compressor and the learned
indexer are implemented. With a single prefill token fewer tokens than the ratio
arrive, so both produce zero compressed slots and the layer attends over its
window alone (as the C++ path does); a prompt of five tokens closes one window,
appends one compressed slot to the key/value cache of each token that may see it
(positions 3 and 4) and selects it per token through the indexer.

Scope: three layers of 43 at most, one to sixteen prefill tokens, compression
ratios 0 and 4. It is a verification instrument, not a model.

Scalar semantics follow the CPU reference path: dequantized weights and every
graph tensor are float32; matrix-vector products accumulate in float64 in
column order and round once to float32, matching ``matvec_blocked`` and
``linear`` in the C++ reference. That ordering is reproduced with
``numpy.cumsum`` so the resulting float32 values, and therefore the FNV-1a
checksums, are comparable with the probe trace.

Quantized codebooks (IQ2_XXS, IQ3_XXS) are the pinned ggml tables copied as
constants; their FNV-1a fingerprints are asserted at startup, so a transcription
error fails loudly instead of producing plausible weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment failure
    raise SystemExit(
        "the layer oracle requires NumPy; this interpreter has no numpy module. "
        "Install it or run with an interpreter that provides it "
        f"(import error: {error})"
    ) from error


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32
ARTIFACT_FORMAT = "vastloom.range-artifact.v1"
SUMMARY_FORMAT = "vastloom.deepseek_v4.layer0_oracle.v1"
CHAIN_SUMMARY_FORMAT = "vastloom.deepseek_v4.chain_oracle.v1"
PREFILL_SUMMARY_FORMAT = "vastloom.deepseek_v4.prefill_oracle.v1"

# GGML storage types used by the imported layer-0 ranges.
F32 = 0
Q8_0 = 8
Q4_K = 12
Q5_K = 13
Q6_K = 14
IQ2_XXS = 16
IQ3_XXS = 18
I32 = 26
BF16 = 30
MXFP4 = 39  # present in other layers' headers, mapped but never decoded by this oracle
IQ2_S = 22  # present in other layers' headers, mapped but never decoded by this oracle

TYPE_NAMES = {
    F32: "F32",
    Q8_0: "Q8_0",
    Q4_K: "Q4_K",
    Q5_K: "Q5_K",
    Q6_K: "Q6_K",
    IQ2_XXS: "IQ2_XXS",
    IQ3_XXS: "IQ3_XXS",
    IQ2_S: "IQ2_S",
    I32: "I32",
    BF16: "BF16",
    MXFP4: "MXFP4",
}
BLOCK_LAYOUT = {
    Q8_0: (32, 34),
    Q4_K: (256, 144),
    Q5_K: (256, 176),
    Q6_K: (256, 210),
    IQ2_XXS: (256, 66),
    IQ3_XXS: (256, 98),
    IQ2_S: (256, 82),
    MXFP4: (32, 17),
}
DENSE_LAYOUT = {F32: 4, BF16: 2, I32: 4}

# Pinned ggml IQ2_XXS and IQ3_XXS codebooks (ggml commit 30bf8685, MIT).
IQ2_XXS_GRID_VALUES = (
    0x0808080808080808, 0x080808080808082B, 0x0808080808081919, 0x0808080808082B08, 0x0808080808082B2B, 0x0808080808190819, 0x0808080808191908, 0x08080808082B0808,
    0x08080808082B082B, 0x08080808082B2B08, 0x08080808082B2B2B, 0x0808080819080819, 0x0808080819081908, 0x0808080819190808, 0x0808080819192B08, 0x08080808192B0819,
    0x08080808192B1908, 0x080808082B080808, 0x080808082B08082B, 0x080808082B082B2B, 0x080808082B2B082B, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192B081908, 0x080808192B192B08, 0x0808082B08080808, 0x0808082B0808082B, 0x0808082B082B082B, 0x0808082B2B08082B,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082B0819, 0x08081908082B1908, 0x0808190819080808, 0x080819081908082B, 0x0808190819082B08,
    0x08081908192B0808, 0x080819082B080819, 0x080819082B081908, 0x080819082B190808, 0x080819082B2B1908, 0x0808191908080808, 0x080819190808082B, 0x0808191908082B08,
    0x08081919082B0808, 0x080819191908192B, 0x08081919192B2B19, 0x080819192B080808, 0x080819192B190819, 0x0808192B08082B19, 0x0808192B08190808, 0x0808192B19080808,
    0x0808192B2B081908, 0x0808192B2B2B1908, 0x08082B0808080808, 0x08082B0808081919, 0x08082B0808082B08, 0x08082B0808191908, 0x08082B08082B2B08, 0x08082B0819080819,
    0x08082B0819081908, 0x08082B0819190808, 0x08082B081919082B, 0x08082B082B082B08, 0x08082B1908081908, 0x08082B1919080808, 0x08082B2B0808082B, 0x08082B2B08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082B0819, 0x0819080819080808, 0x08190808192B0808, 0x081908082B081908, 0x081908082B190808,
    0x081908082B191919, 0x0819081908080808, 0x0819081908082B08, 0x08190819082B0808, 0x0819081919190808, 0x0819081919192B2B, 0x081908192B080808, 0x0819082B082B1908,
    0x0819082B19081919, 0x0819190808080808, 0x0819190808082B08, 0x08191908082B0808, 0x08191908082B1919, 0x0819190819082B19, 0x081919082B080808, 0x0819191908192B08,
    0x08191919192B082B, 0x0819192B08080808, 0x0819192B0819192B, 0x08192B0808080819, 0x08192B0808081908, 0x08192B0808190808, 0x08192B0819080808, 0x08192B082B080819,
    0x08192B1908080808, 0x08192B1908081919, 0x08192B192B2B0808, 0x08192B2B19190819, 0x082B080808080808, 0x082B08080808082B, 0x082B080808082B2B, 0x082B080819081908,
    0x082B0808192B0819, 0x082B08082B080808, 0x082B08082B08082B, 0x082B0819082B2B19, 0x082B081919082B08, 0x082B082B08080808, 0x082B082B0808082B, 0x082B190808080819,
    0x082B190808081908, 0x082B190808190808, 0x082B190819080808, 0x082B19081919192B, 0x082B191908080808, 0x082B191919080819, 0x082B1919192B1908, 0x082B192B2B190808,
    0x082B2B0808082B08, 0x082B2B08082B0808, 0x082B2B082B191908, 0x082B2B2B19081908, 0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192B08,
    0x19080808082B0819, 0x19080808082B1908, 0x1908080819080808, 0x1908080819082B08, 0x190808081919192B, 0x19080808192B0808, 0x190808082B080819, 0x190808082B081908,
    0x190808082B190808, 0x1908081908080808, 0x19080819082B0808, 0x19080819192B0819, 0x190808192B080808, 0x190808192B081919, 0x1908082B08080819, 0x1908082B08190808,
    0x1908082B19082B08, 0x1908082B1919192B, 0x1908082B192B2B08, 0x1908190808080808, 0x1908190808082B08, 0x19081908082B0808, 0x190819082B080808, 0x190819082B192B19,
    0x190819190819082B, 0x19081919082B1908, 0x1908192B08080808, 0x19082B0808080819, 0x19082B0808081908, 0x19082B0808190808, 0x19082B0819080808, 0x19082B0819081919,
    0x19082B1908080808, 0x19082B1919192B08, 0x19082B19192B0819, 0x19082B192B08082B, 0x19082B2B19081919, 0x19082B2B2B190808, 0x1919080808080808, 0x1919080808082B08,
    0x1919080808190819, 0x1919080808192B19, 0x19190808082B0808, 0x191908082B080808, 0x191908082B082B08, 0x1919081908081908, 0x191908191908082B, 0x191908192B2B1908,
    0x1919082B2B190819, 0x191919082B190808, 0x191919082B19082B, 0x1919191908082B2B, 0x1919192B08080819, 0x1919192B19191908, 0x19192B0808080808, 0x19192B0808190819,
    0x19192B0808192B19, 0x19192B08192B1908, 0x19192B1919080808, 0x19192B2B08082B08, 0x192B080808081908, 0x192B080808190808, 0x192B080819080808, 0x192B0808192B2B08,
    0x192B081908080808, 0x192B081919191919, 0x192B082B08192B08, 0x192B082B192B0808, 0x192B190808080808, 0x192B190808081919, 0x192B191908190808, 0x192B19190819082B,
    0x192B19192B081908, 0x192B2B081908082B, 0x2B08080808080808, 0x2B0808080808082B, 0x2B08080808082B2B, 0x2B08080819080819, 0x2B0808082B08082B, 0x2B08081908081908,
    0x2B08081908192B08, 0x2B08081919080808, 0x2B08082B08190819, 0x2B08190808080819, 0x2B08190808081908, 0x2B08190808190808, 0x2B08190808191919, 0x2B08190819080808,
    0x2B081908192B0808, 0x2B08191908080808, 0x2B0819191908192B, 0x2B0819192B191908, 0x2B08192B08082B19, 0x2B08192B19080808, 0x2B08192B192B0808, 0x2B082B080808082B,
    0x2B082B1908081908, 0x2B082B2B08190819, 0x2B19080808081908, 0x2B19080808190808, 0x2B190808082B1908, 0x2B19080819080808, 0x2B1908082B2B0819, 0x2B1908190819192B,
    0x2B1908192B080808, 0x2B19082B19081919, 0x2B19190808080808, 0x2B191908082B082B, 0x2B19190819081908, 0x2B19191919190819, 0x2B192B082B080819, 0x2B192B19082B0808,
    0x2B2B08080808082B, 0x2B2B080819190808, 0x2B2B08082B081919, 0x2B2B081908082B19, 0x2B2B082B08080808, 0x2B2B190808192B08, 0x2B2B2B0819190808, 0x2B2B2B1908081908,
)

IQ3_XXS_GRID_VALUES = (
    0x04040404, 0x04040414, 0x04040424, 0x04040C0C, 0x04040C1C, 0x04040C3E, 0x04041404, 0x04041414, 0x04041C0C, 0x04042414, 0x04043E1C, 0x04043E2C, 0x040C040C, 0x040C041C, 0x040C0C04, 0x040C0C14,
    0x040C140C, 0x040C142C, 0x040C1C04, 0x040C1C14, 0x040C240C, 0x040C2C24, 0x040C3E04, 0x04140404, 0x04140414, 0x04140424, 0x04140C0C, 0x04141404, 0x04141414, 0x04141C0C, 0x04141C1C, 0x04141C3E,
    0x04142C0C, 0x04142C3E, 0x04143E2C, 0x041C040C, 0x041C043E, 0x041C0C04, 0x041C0C14, 0x041C142C, 0x041C3E04, 0x04240C1C, 0x04241C3E, 0x04242424, 0x04242C3E, 0x04243E1C, 0x04243E2C, 0x042C040C,
    0x042C043E, 0x042C1C14, 0x042C2C14, 0x04341C2C, 0x04343424, 0x043E0C04, 0x043E0C24, 0x043E0C34, 0x043E241C, 0x043E340C, 0x0C04040C, 0x0C04041C, 0x0C040C04, 0x0C040C14, 0x0C04140C, 0x0C04141C,
    0x0C041C04, 0x0C041C14, 0x0C041C24, 0x0C04243E, 0x0C042C04, 0x0C0C0404, 0x0C0C0414, 0x0C0C0C0C, 0x0C0C1404, 0x0C0C1414, 0x0C14040C, 0x0C14041C, 0x0C140C04, 0x0C140C14, 0x0C14140C, 0x0C141C04,
    0x0C143E14, 0x0C1C0404, 0x0C1C0414, 0x0C1C1404, 0x0C1C1C0C, 0x0C1C2434, 0x0C1C3434, 0x0C24040C, 0x0C24042C, 0x0C242C04, 0x0C2C1404, 0x0C2C1424, 0x0C2C2434, 0x0C2C3E0C, 0x0C34042C, 0x0C3E1414,
    0x0C3E2404, 0x14040404, 0x14040414, 0x14040C0C, 0x14040C1C, 0x14041404, 0x14041414, 0x14041434, 0x14041C0C, 0x14042414, 0x140C040C, 0x140C041C, 0x140C042C, 0x140C0C04, 0x140C0C14, 0x140C140C,
    0x140C1C04, 0x140C341C, 0x140C343E, 0x140C3E04, 0x14140404, 0x14140414, 0x14140C0C, 0x14140C3E, 0x14141404, 0x14141414, 0x14141C3E, 0x14142404, 0x14142C2C, 0x141C040C, 0x141C0C04, 0x141C0C24,
    0x141C3E04, 0x141C3E24, 0x14241C2C, 0x14242C1C, 0x142C041C, 0x142C143E, 0x142C240C, 0x142C3E24, 0x143E040C, 0x143E041C, 0x143E0C34, 0x143E242C, 0x1C04040C, 0x1C040C04, 0x1C040C14, 0x1C04140C,
    0x1C04141C, 0x1C042C04, 0x1C04342C, 0x1C043E14, 0x1C0C0404, 0x1C0C0414, 0x1C0C1404, 0x1C0C1C0C, 0x1C0C2424, 0x1C0C2434, 0x1C14040C, 0x1C14041C, 0x1C140C04, 0x1C14142C, 0x1C142C14, 0x1C143E14,
    0x1C1C0C0C, 0x1C1C1C1C, 0x1C241C04, 0x1C24243E, 0x1C243E14, 0x1C2C0404, 0x1C2C0434, 0x1C2C1414, 0x1C2C2C2C, 0x1C340C24, 0x1C341C34, 0x1C34341C, 0x1C3E1C1C, 0x1C3E3404, 0x24040424, 0x24040C3E,
    0x24041C2C, 0x24041C3E, 0x24042C1C, 0x24042C3E, 0x240C3E24, 0x24141404, 0x24141C3E, 0x24142404, 0x24143404, 0x24143434, 0x241C043E, 0x241C242C, 0x24240424, 0x24242C0C, 0x24243424, 0x242C142C,
    0x242C241C, 0x242C3E04, 0x243E042C, 0x243E0C04, 0x243E0C14, 0x243E1C04, 0x2C040C14, 0x2C04240C, 0x2C043E04, 0x2C0C0404, 0x2C0C0434, 0x2C0C1434, 0x2C0C2C2C, 0x2C140C24, 0x2C141C14, 0x2C143E14,
    0x2C1C0414, 0x2C1C2C1C, 0x2C240C04, 0x2C24141C, 0x2C24143E, 0x2C243E14, 0x2C2C0414, 0x2C2C1C0C, 0x2C342C04, 0x2C3E1424, 0x2C3E2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340C140C,
    0x340C340C, 0x34140C3E, 0x34143424, 0x341C1C04, 0x341C1C34, 0x34242424, 0x342C042C, 0x342C2C14, 0x34341C1C, 0x343E041C, 0x343E140C, 0x3E04041C, 0x3E04042C, 0x3E04043E, 0x3E040C04, 0x3E041C14,
    0x3E042C14, 0x3E0C1434, 0x3E0C2404, 0x3E140C14, 0x3E14242C, 0x3E142C14, 0x3E1C0404, 0x3E1C0C2C, 0x3E1C1C1C, 0x3E1C3404, 0x3E24140C, 0x3E24240C, 0x3E2C0404, 0x3E2C0414, 0x3E2C1424, 0x3E341C04,
)

IQ2_XXS_GRID = np.array(IQ2_XXS_GRID_VALUES, dtype=np.uint64)
IQ3_XXS_GRID = np.array(IQ3_XXS_GRID_VALUES, dtype=np.uint32)
POPCOUNT_7BIT = np.array([bin(value).count("1") & 1 for value in range(128)], dtype=np.uint32)


def verify_codebooks() -> None:
    """Fail loudly if a constant table was mistranscribed."""
    if grid_fnv1a(IQ2_XXS_GRID) != 0xBB4EE025B5AC6E8E:
        raise OracleError("the embedded IQ2_XXS codebook does not match its pinned fingerprint")
    if grid_fnv1a(IQ3_XXS_GRID) != 0xDFA5DC83D6A155D5:
        raise OracleError("the embedded IQ3_XXS codebook does not match its pinned fingerprint")


class OracleError(RuntimeError):
    """A verified input, equation or tolerance precondition failed."""


def fnv1a64(payload: bytes) -> int:
    """FNV-1a 64 with the same offset basis as the C++ probe trace."""
    value = 1469598103934665603
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def fnv1a64_standard(payload: bytes) -> int:
    """Canonical FNV-1a 64, used by the pinned ggml codebook fingerprints."""
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def grid_fnv1a(values: np.ndarray) -> int:
    dtype = "<u8" if values.dtype == np.uint64 else "<u4"
    return fnv1a64_standard(values.astype(dtype).tobytes())


def f32_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def array_checksums(values: np.ndarray) -> dict[str, object]:
    payload = f32_bytes(values)
    return {
        "count": int(values.size),
        "fnv1a64": fnv1a64(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def storage_bytes(ggml_type: int, rows: int, columns: int) -> int:
    if ggml_type in BLOCK_LAYOUT:
        block_elements, block_bytes = BLOCK_LAYOUT[ggml_type]
        if columns % block_elements != 0:
            raise OracleError(
                f"columns {columns} are not a multiple of the "
                f"{TYPE_NAMES[ggml_type]} block size {block_elements}"
            )
        return rows * (columns // block_elements) * block_bytes
    if ggml_type in DENSE_LAYOUT:
        return rows * columns * DENSE_LAYOUT[ggml_type]
    raise OracleError(f"storage type {ggml_type} is not used by the layer-0 oracle")


def row_bytes(ggml_type: int, columns: int) -> int:
    return storage_bytes(ggml_type, 1, columns)


# ---------------------------------------------------------------------------
# Quantized block decoders
# ---------------------------------------------------------------------------

_Q8_0_DTYPE = np.dtype({"names": ["d", "q"], "formats": ["<f2", ("i1", 32)],
                        "offsets": [0, 2], "itemsize": 34})
_Q4_K_DTYPE = np.dtype({"names": ["d", "dmin", "scales", "qs"],
                        "formats": ["<f2", "<f2", ("u1", 12), ("u1", 128)],
                        "offsets": [0, 2, 4, 16], "itemsize": 144})
_Q5_K_DTYPE = np.dtype({"names": ["d", "dmin", "scales", "qh", "ql"],
                        "formats": ["<f2", "<f2", ("u1", 12), ("u1", 32), ("u1", 128)],
                        "offsets": [0, 2, 4, 16, 48], "itemsize": 176})
_Q6_K_DTYPE = np.dtype({"names": ["ql", "qh", "scales", "d"],
                        "formats": [("u1", 128), ("u1", 64), ("i1", 16), "<f2"],
                        "offsets": [0, 128, 192, 208], "itemsize": 210})
_IQ2_DTYPE = np.dtype({"names": ["d", "qs"], "formats": ["<f2", ("u1", 64)],
                       "offsets": [0, 2], "itemsize": 66})
_IQ3_DTYPE = np.dtype({"names": ["d", "qs", "scales"],
                       "formats": ["<f2", ("u1", 64), ("u1", 32)],
                       "offsets": [0, 2, 66], "itemsize": 98})


def _as_blocks(raw: bytes, dtype: np.dtype) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype)


def _scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the eight K-quant scale/minimum pairs exactly as ``qk_scale_min``."""
    count = scales.shape[0]
    scale = np.empty((count, 8), dtype=np.uint8)
    minimum = np.empty((count, 8), dtype=np.uint8)
    for index in range(8):
        if index < 4:
            scale[:, index] = scales[:, index] & 63
            minimum[:, index] = scales[:, index + 4] & 63
        else:
            scale[:, index] = (scales[:, index + 4] & 15) | ((scales[:, index - 4] >> 6) << 4)
            minimum[:, index] = (scales[:, index + 4] >> 4) | ((scales[:, index] >> 6) << 4)
    return scale, minimum


def _decode_q8_0(raw: bytes, blocks: int) -> np.ndarray:
    values = _as_blocks(raw, _Q8_0_DTYPE)
    if values.size != blocks:
        raise OracleError("Q8_0 block count mismatch")
    scale = values["d"].astype(np.float32)
    quants = values["q"].astype(np.float32)
    return (scale[:, None] * quants).reshape(-1)


def _decode_q4_k(raw: bytes, blocks: int) -> np.ndarray:
    values = _as_blocks(raw, _Q4_K_DTYPE)
    if values.size != blocks:
        raise OracleError("Q4_K block count mismatch")
    scale = values["d"].astype(np.float32)
    minimum = values["dmin"].astype(np.float32)
    sc, mn = _scale_min(values["scales"])
    quantized = values["qs"]
    output = np.empty((blocks, 256), dtype=np.float32)
    for group in range(4):
        for half in range(2):
            index = group * 2 + half
            d = scale * sc[:, index].astype(np.float32)
            m = minimum * mn[:, index].astype(np.float32)
            if half == 0:
                q = (quantized[:, group * 32:group * 32 + 32] & 15).astype(np.float32)
            else:
                q = (quantized[:, group * 32:group * 32 + 32] >> 4).astype(np.float32)
            output[:, group * 64 + half * 32:group * 64 + half * 32 + 32] = (
                d[:, None] * q - m[:, None]
            )
    return output.reshape(-1)


def _decode_q5_k(raw: bytes, blocks: int) -> np.ndarray:
    values = _as_blocks(raw, _Q5_K_DTYPE)
    if values.size != blocks:
        raise OracleError("Q5_K block count mismatch")
    scale = values["d"].astype(np.float32)
    minimum = values["dmin"].astype(np.float32)
    sc, mn = _scale_min(values["scales"])
    quantized = values["ql"]
    high = values["qh"]
    output = np.empty((blocks, 256), dtype=np.float32)
    for group in range(4):
        first_mask = 1 << (2 * group)
        second_mask = 2 << (2 * group)
        low = quantized[:, group * 32:group * 32 + 32]
        high_bits = high
        first = ((low & 15).astype(np.float32)
                 + np.where((high_bits & first_mask) != 0, np.float32(16.0), np.float32(0.0)))
        second = ((low >> 4).astype(np.float32)
                  + np.where((high_bits & second_mask) != 0, np.float32(16.0), np.float32(0.0)))
        for half, q in ((0, first), (1, second)):
            index = group * 2 + half
            d = scale * sc[:, index].astype(np.float32)
            m = minimum * mn[:, index].astype(np.float32)
            output[:, group * 64 + half * 32:group * 64 + half * 32 + 32] = (
                d[:, None] * q - m[:, None]
            )
    return output.reshape(-1)


def _decode_q6_k(raw: bytes, blocks: int) -> np.ndarray:
    values = _as_blocks(raw, _Q6_K_DTYPE)
    if values.size != blocks:
        raise OracleError("Q6_K block count mismatch")
    scale = values["d"].astype(np.float32)
    scales = values["scales"].astype(np.float32)
    quantized = values["ql"]
    high = values["qh"]
    output = np.empty((blocks, 256), dtype=np.float32)
    for half in range(2):
        base = half * 64
        high_base = half * 32
        for lane in range(32):
            scale_index = lane // 16
            low_first = quantized[:, base + lane]
            low_second = quantized[:, base + 32 + lane]
            high_byte = high[:, high_base + lane]
            q1 = ((low_first & 15) | (((high_byte >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((low_second & 15) | (((high_byte >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((low_first >> 4) | (((high_byte >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((low_second >> 4) | (((high_byte >> 6) & 3) << 4)).astype(np.int32) - 32
            offset = half * 128
            for column, q, scale_offset in (
                (0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)
            ):
                weight = scale * scales[:, half * 8 + scale_index + scale_offset]
                output[:, offset + column + lane] = (
                    weight * q.astype(np.float32)
                )
    return output.reshape(-1)


def _decode_iq2_xxs(raw: bytes, blocks: int) -> np.ndarray:
    values = _as_blocks(raw, _IQ2_DTYPE)
    if values.size != blocks:
        raise OracleError("IQ2_XXS block count mismatch")
    scale = values["d"].astype(np.float32)
    quantized = values["qs"]
    output = np.empty((blocks, 256), dtype=np.float32)
    lanes = np.arange(8, dtype=np.uint64) * np.uint64(8)
    for group in range(8):
        packed = quantized[:, group * 8:group * 8 + 8]
        auxiliary = (packed[:, 4].astype(np.uint32)
                     | (packed[:, 5].astype(np.uint32) << 8)
                     | (packed[:, 6].astype(np.uint32) << 16)
                     | (packed[:, 7].astype(np.uint32) << 24))
        local_scale = ((scale * (np.float32(0.5) + (auxiliary >> 28).astype(np.float32)))
                       * np.float32(0.25))
        for subblock in range(4):
            grid_index = packed[:, subblock].astype(np.intp)
            grid = IQ2_XXS_GRID[grid_index]
            magnitudes = ((grid[:, None] >> lanes) & np.uint64(0xFF)).astype(np.float32)
            sign_index = ((auxiliary >> (7 * subblock)) & 127).astype(np.uint32)
            signs = sign_index | (POPCOUNT_7BIT[sign_index] << 7)
            sign = np.where((signs[:, None] & (1 << np.arange(8, dtype=np.uint32))) != 0,
                            np.float32(-1.0), np.float32(1.0))
            base = group * 32 + subblock * 8
            output[:, base:base + 8] = (local_scale[:, None] * magnitudes) * sign
    return output.reshape(-1)


def _decode_iq3_xxs(raw: bytes, blocks: int) -> np.ndarray:
    values = _as_blocks(raw, _IQ3_DTYPE)
    if values.size != blocks:
        raise OracleError("IQ3_XXS block count mismatch")
    scale = values["d"].astype(np.float32)
    quantized = values["qs"]
    scales = values["scales"]
    output = np.empty((blocks, 256), dtype=np.float32)
    lanes = np.arange(4, dtype=np.uint32) * np.uint32(8)
    for group in range(8):
        auxiliary = (scales[:, group * 4].astype(np.uint32)
                     | (scales[:, group * 4 + 1].astype(np.uint32) << 8)
                     | (scales[:, group * 4 + 2].astype(np.uint32) << 16)
                     | (scales[:, group * 4 + 3].astype(np.uint32) << 24))
        local_scale = ((scale * (np.float32(0.5) + (auxiliary >> 28).astype(np.float32)))
                       * np.float32(0.5))
        for subblock in range(4):
            first_index = quantized[:, group * 8 + subblock * 2].astype(np.intp)
            second_index = quantized[:, group * 8 + subblock * 2 + 1].astype(np.intp)
            first_grid = IQ3_XXS_GRID[first_index]
            second_grid = IQ3_XXS_GRID[second_index]
            first = ((first_grid[:, None] >> lanes) & np.uint32(0xFF)).astype(np.float32)
            second = ((second_grid[:, None] >> lanes) & np.uint32(0xFF)).astype(np.float32)
            sign_index = ((auxiliary >> (7 * subblock)) & 127).astype(np.uint32)
            signs = sign_index | (POPCOUNT_7BIT[sign_index] << 7)
            first_sign = np.where((signs[:, None] & (1 << np.arange(4, dtype=np.uint32))) != 0,
                                  np.float32(-1.0), np.float32(1.0))
            second_sign = np.where(
                (signs[:, None] & (1 << (np.arange(4, dtype=np.uint32) + 4))) != 0,
                np.float32(-1.0), np.float32(1.0))
            base = group * 32 + subblock * 8
            output[:, base:base + 4] = (local_scale[:, None] * first) * first_sign
            output[:, base + 4:base + 8] = (local_scale[:, None] * second) * second_sign
    return output.reshape(-1)


def decode_matrix(raw: bytes, ggml_type: int, rows: int, columns: int) -> np.ndarray:
    """Decode ``rows`` complete rows of ``columns`` values into float32."""
    if ggml_type in DENSE_LAYOUT:
        required = rows * columns * DENSE_LAYOUT[ggml_type]
        if len(raw) != required:
            raise OracleError(
                f"{TYPE_NAMES[ggml_type]} payload has {len(raw)} bytes, expected {required}"
            )
        if ggml_type == F32:
            return np.frombuffer(raw, dtype="<f4").reshape(rows, columns).copy()
        if ggml_type == BF16:
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            return bits.view(np.float32).reshape(rows, columns)
        return np.frombuffer(raw, dtype="<i4").reshape(rows, columns).copy()
    block_elements, block_bytes = BLOCK_LAYOUT[ggml_type]
    required = storage_bytes(ggml_type, rows, columns)
    if len(raw) != required:
        raise OracleError(
            f"{TYPE_NAMES[ggml_type]} payload has {len(raw)} bytes, expected {required}"
        )
    blocks = rows * (columns // block_elements)
    decoder = {
        Q8_0: _decode_q8_0,
        Q4_K: _decode_q4_k,
        Q5_K: _decode_q5_k,
        Q6_K: _decode_q6_k,
        IQ2_XXS: _decode_iq2_xxs,
        IQ3_XXS: _decode_iq3_xxs,
    }[ggml_type]
    return decoder(raw, blocks).reshape(rows, columns)


# ---------------------------------------------------------------------------
# GGUF parsing
# ---------------------------------------------------------------------------

@dataclass
class GGUFTensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    shard: str
    offset: int
    encoded_bytes: int

    @property
    def rows(self) -> int:
        if len(self.dims) < 2:
            raise OracleError(f"tensor {self.name} is not two-dimensional")
        return int(self.dims[1])

    @property
    def columns(self) -> int:
        if not self.dims:
            raise OracleError(f"tensor {self.name} has no dimensions")
        return int(self.dims[0])

    @property
    def slices(self) -> int:
        return 1 if len(self.dims) < 3 else int(self.dims[2])


class GGUFReader:
    """Minimal little-endian GGUF v3 reader for headers and tensor tables."""

    def __init__(self, payload: bytes, path: Path) -> None:
        self.payload = payload
        self.path = path
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.payload):
            raise OracleError(f"GGUF header is truncated in {self.path}")
        value = self.payload[self.offset:self.offset + count]
        self.offset += count
        return value

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > 1 << 20:
            raise OracleError(f"GGUF string length {length} is implausible in {self.path}")
        return self.take(length).decode("utf-8", "strict")

    def value(self, value_type: int, depth: int = 0) -> object:
        if depth > 2:
            raise OracleError("GGUF metadata nesting is too deep")
        if value_type == 0:
            return struct.unpack("<B", self.take(1))[0]
        if value_type == 1:
            return struct.unpack("<b", self.take(1))[0]
        if value_type == 2:
            return struct.unpack("<H", self.take(2))[0]
        if value_type == 3:
            return struct.unpack("<h", self.take(2))[0]
        if value_type == 4:
            return self.u32()
        if value_type == 5:
            return struct.unpack("<i", self.take(4))[0]
        if value_type == 6:
            return struct.unpack("<f", self.take(4))[0]
        if value_type == 7:
            return struct.unpack("<B", self.take(1))[0] != 0
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = self.u32()
            count = self.u64()
            if count > 32_000_000:
                raise OracleError(f"GGUF array count {count} is implausible in {self.path}")
            if element_type == 8:
                # The tokenizer token/merge arrays are large and unused; consume the
                # element payload so the stream stays aligned, but retain only the count.
                for _ in range(count):
                    self.string()
                return {"array": "string", "count": count}
            return [self.value(element_type, depth + 1) for _ in range(count)]
        if value_type == 10:
            return self.u64()
        if value_type == 11:
            return struct.unpack("<q", self.take(8))[0]
        if value_type == 12:
            return struct.unpack("<d", self.take(8))[0]
        raise OracleError(f"GGUF value type {value_type} is not supported")

    def header(self) -> tuple[dict[str, object], list[GGUFTensor], int]:
        if self.u32() != GGUF_MAGIC:
            raise OracleError(f"GGUF magic is invalid in {self.path}")
        version = self.u32()
        if version != GGUF_VERSION:
            raise OracleError(f"GGUF version {version} is not the pinned v3 format")
        tensor_count = self.u64()
        metadata_count = self.u64()
        if tensor_count > 1_000_000 or metadata_count > 100_000:
            raise OracleError(f"GGUF header counts are implausible in {self.path}")
        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = self.string()
            value_type = self.u32()
            if key in metadata:
                raise OracleError(f"GGUF metadata key {key} is duplicated")
            metadata[key] = self.value(value_type)
        tensors: list[GGUFTensor] = []
        for _ in range(tensor_count):
            name = self.string()
            dimensions = self.u32()
            if dimensions == 0 or dimensions > 8:
                raise OracleError(f"GGUF tensor {name} has an invalid rank")
            dims = tuple(self.u64() for _ in range(dimensions))
            ggml_type = self.u32()
            relative = self.u64()
            if ggml_type not in TYPE_NAMES:
                raise OracleError(
                    f"GGUF tensor {name} uses unsupported storage type {ggml_type}"
                )
            rows = int(dims[1]) if dimensions >= 2 else 1
            columns = int(dims[0])
            if dimensions >= 3:
                encoded = storage_bytes(ggml_type, rows, columns) * int(dims[2])
            else:
                encoded = storage_bytes(ggml_type, rows, columns)
            tensors.append(GGUFTensor(name, dims, ggml_type, self.path.name, relative, encoded))
        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT) or DEFAULT_ALIGNMENT)
        if alignment <= 0 or alignment > 1 << 20:
            raise OracleError(f"GGUF alignment {alignment} is invalid")
        data_start = (self.offset + alignment - 1) // alignment * alignment
        for tensor in tensors:
            tensor.offset += data_start
        if tensors and tensors[-1].offset + tensors[-1].encoded_bytes > 1 << 20 and len(self.payload) < 1 << 20:
            raise OracleError("GGUF header prefix does not cover the tensor table")
        return metadata, tensors, data_start


def read_header(path: Path) -> tuple[dict[str, object], list[GGUFTensor], int]:
    payload = path.read_bytes()
    return GGUFReader(payload, path).header()


# ---------------------------------------------------------------------------
# Metadata store and artifact parsing
# ---------------------------------------------------------------------------

@dataclass
class ModelStore:
    root: Path
    model_id: str
    revision: str
    metadata: dict[str, object]
    tensors: dict[str, list[GGUFTensor]]
    shard_files: dict[str, Path]  # artifact source_object basename -> local GGUF path
    vocabulary_size: int
    config: "Config"


@dataclass
class Config:
    vocabulary_size: int
    width: int
    intermediate_width: int
    layer_count: int
    hash_layer_count: int
    head_count: int
    expert_count: int
    experts_per_token: int
    query_lora_rank: int
    head_width: int
    rope_width: int
    output_groups: int
    output_lora_rank: int
    window_size: int
    compression_ratios: list[int]
    compression_rope_theta: float
    original_context: int
    rope_theta: float
    rope_factor: float
    beta_fast: float
    beta_slow: float
    index_head_count: int
    index_head_width: int
    index_top_k: int
    hyper_multiplier: int
    sinkhorn_iterations: int
    norm_epsilon: float
    hyper_epsilon: float
    route_scale: float
    routed_swiglu_limits: list[float]
    shared_swiglu_limits: list[float]

    @property
    def mix_width(self) -> int:
        return (2 + self.hyper_multiplier) * self.hyper_multiplier

    @property
    def flattened_width(self) -> int:
        return self.hyper_multiplier * self.width


def _required_u32(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 0xFFFFFFFF:
        raise OracleError(f"GGUF metadata {key} is missing or not an unsigned 32-bit value")
    return value


def _required_float(metadata: dict[str, object], key: str) -> float:
    value = metadata.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise OracleError(f"GGUF metadata {key} is missing or not finite")
    return float(value)


def _required_float_array(metadata: dict[str, object], key: str, count: int) -> list[float]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise OracleError(f"GGUF metadata {key} is missing or has the wrong length")
    result = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
            raise OracleError(f"GGUF metadata {key} contains a non-finite value")
        result.append(float(item))
    return result


def _required_u32_array(metadata: dict[str, object], key: str, count: int) -> list[int]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise OracleError(f"GGUF metadata {key} is missing or has the wrong length")
    result = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise OracleError(f"GGUF metadata {key} contains a non-integer value")
        result.append(int(item))
    return result


def load_model_store(root: Path) -> ModelStore:
    root = root.resolve()
    manifest_path = root / "vastloom-import.json"
    if not manifest_path.is_file():
        raise OracleError(f"metadata store manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "vastloom.hf-import.v1":
        raise OracleError("metadata store manifest format is not vastloom.hf-import.v1")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise OracleError("metadata store manifest lists no files")

    shard_headers: dict[int, tuple[str, Path]] = {}
    metadata: dict[str, object] | None = None
    shard_tensors: dict[str, list[GGUFTensor]] = {}
    shard_files: dict[str, Path] = {}
    for entry in files:
        relative = entry.get("name")
        if not isinstance(relative, str):
            raise OracleError("metadata store manifest has a malformed file entry")
        path = root / relative
        if not path.is_file():
            raise OracleError(f"metadata store file is missing: {path}")
        size = int(entry.get("size", -1))
        if path.stat().st_size != size:
            raise OracleError(f"metadata store file size mismatch: {path}")
        header_metadata, tensors, _ = read_header(path)
        source_name = path.name[:-len(".header")] if path.name.endswith(".header") else path.name
        shard_files[source_name] = path
        split = header_metadata.get("split.no")
        if not isinstance(split, int):
            split = 0 if "general.architecture" in header_metadata else None
        if split is None:
            raise OracleError(f"GGUF shard identity is missing in {path}")
        if split in shard_headers:
            raise OracleError(f"duplicate GGUF shard {split}")
        shard_headers[split] = (source_name, path)
        shard_tensors[source_name] = tensors
        if "general.architecture" in header_metadata:
            if metadata is not None:
                raise OracleError("multiple shards carry the full architecture metadata")
            metadata = header_metadata
    if metadata is None:
        raise OracleError("no shard carries the DeepSeek architecture metadata")
    if metadata.get("general.architecture") != "deepseek4":
        raise OracleError("GGUF architecture is not deepseek4")
    model_id = manifest.get("model_id")
    revision = manifest.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise OracleError("metadata store manifest has no pinned model identity")

    token_embd = None
    for tensors in shard_tensors.values():
        for tensor in tensors:
            if tensor.name == "token_embd.weight":
                token_embd = tensor
    if token_embd is None:
        raise OracleError("GGUF tensor table has no token_embd.weight")
    config = derive_config(metadata, token_embd)
    tokens = metadata.get("tokenizer.ggml.tokens")
    if isinstance(tokens, dict) and tokens.get("count") != config.vocabulary_size:
        raise OracleError(
            "tokenizer vocabulary length does not match the token embedding row count"
        )
    return ModelStore(root, model_id, revision, metadata, shard_tensors, shard_files,
                      config.vocabulary_size, config)


def derive_config(metadata: dict[str, object], token_embd: GGUFTensor) -> Config:
    layers = _required_u32(metadata, "deepseek4.block_count")
    hash_layers = _required_u32(metadata, "deepseek4.hash_layer_count")
    width = _required_u32(metadata, "deepseek4.embedding_length")
    heads = _required_u32(metadata, "deepseek4.attention.head_count")
    kv_heads = _required_u32(metadata, "deepseek4.attention.head_count_kv")
    experts = _required_u32(metadata, "deepseek4.expert_count")
    top_k = _required_u32(metadata, "deepseek4.expert_used_count")
    gating = _required_u32(metadata, "deepseek4.expert_gating_func")
    shared = _required_u32(metadata, "deepseek4.expert_shared_count")
    intermediate = _required_u32(metadata, "deepseek4.expert_feed_forward_length")
    query_lora = _required_u32(metadata, "deepseek4.attention.q_lora_rank")
    head_width = _required_u32(metadata, "deepseek4.attention.key_length")
    value_width = _required_u32(metadata, "deepseek4.attention.value_length")
    rope_width = _required_u32(metadata, "deepseek4.rope.dimension_count")
    output_groups = _required_u32(metadata, "deepseek4.attention.output_group_count")
    output_lora = _required_u32(metadata, "deepseek4.attention.output_lora_rank")
    window = _required_u32(metadata, "deepseek4.attention.sliding_window")
    original_context = _required_u32(metadata, "deepseek4.rope.scaling.original_context_length")
    index_heads = _required_u32(metadata, "deepseek4.attention.indexer.head_count")
    index_width = _required_u32(metadata, "deepseek4.attention.indexer.key_length")
    index_top_k = _required_u32(metadata, "deepseek4.attention.indexer.top_k")
    hyper = _required_u32(metadata, "deepseek4.hyper_connection.count")
    sinkhorn = _required_u32(metadata, "deepseek4.hyper_connection.sinkhorn_iterations")
    compression_rope = _required_float(metadata, "deepseek4.attention.compress_rope_freq_base")
    rope_theta = _required_float(metadata, "deepseek4.rope.freq_base")
    rope_factor = _required_float(metadata, "deepseek4.rope.scaling.factor")
    beta_fast = _required_float(metadata, "deepseek4.rope.scaling.yarn_beta_fast")
    beta_slow = _required_float(metadata, "deepseek4.rope.scaling.yarn_beta_slow")
    norm_epsilon = _required_float(metadata, "deepseek4.attention.layer_norm_rms_epsilon")
    hyper_epsilon = _required_float(metadata, "deepseek4.hyper_connection.epsilon")
    route_scale = _required_float(metadata, "deepseek4.expert_weights_scale")
    normalized = metadata.get("deepseek4.expert_weights_norm")
    scaling = metadata.get("deepseek4.rope.scaling.type")
    if normalized is not True or scaling != "yarn":
        raise OracleError("DeepSeek reference architecture requires normalized weights and yarn RoPE")
    if (layers == 0 or hash_layers > layers or width == 0 or heads == 0 or kv_heads != 1
            or experts == 0 or top_k == 0 or top_k > experts or gating != 4 or shared != 1
            or intermediate == 0 or query_lora == 0 or head_width == 0
            or value_width != head_width or rope_width == 0 or rope_width > head_width
            or rope_width % 2 != 0 or output_groups == 0 or output_lora == 0 or window == 0
            or original_context == 0 or index_heads == 0 or index_width < rope_width
            or index_top_k == 0 or hyper == 0 or hyper > 16 or sinkhorn == 0
            or compression_rope <= 0 or rope_theta <= 0 or rope_factor <= 0
            or beta_fast <= 0 or beta_slow <= 0 or norm_epsilon <= 0 or hyper_epsilon <= 0
            or route_scale <= 0):
        raise OracleError("DeepSeek GGUF reference architecture metadata is inconsistent")
    if (heads * head_width) % output_groups != 0:
        raise OracleError("attention width is not divisible by the output group count")
    ratios = _required_u32_array(
        metadata, "deepseek4.attention.compress_ratios", layers + hash_layers
    )
    if any(value not in (0, 4, 128) for value in ratios):
        raise OracleError("compression schedule contains an unsupported ratio")
    routed_limits = _required_float_array(metadata, "deepseek4.swiglu_clamp_exp", layers)
    shared_limits = _required_float_array(metadata, "deepseek4.swiglu_clamp_shexp", layers)
    if any(value < 0 for value in routed_limits + shared_limits):
        raise OracleError("SwiGLU clamp schedule contains a negative limit")
    vocabulary = int(token_embd.dims[1])
    if int(token_embd.dims[0]) != width or vocabulary == 0:
        raise OracleError("token embedding shape does not match the architecture width")
    return Config(
        vocabulary_size=vocabulary,
        width=width,
        intermediate_width=intermediate,
        layer_count=layers,
        hash_layer_count=hash_layers,
        head_count=heads,
        expert_count=experts,
        experts_per_token=top_k,
        query_lora_rank=query_lora,
        head_width=head_width,
        rope_width=rope_width,
        output_groups=output_groups,
        output_lora_rank=output_lora,
        window_size=window,
        compression_ratios=ratios[:layers],
        compression_rope_theta=compression_rope,
        original_context=original_context,
        rope_theta=rope_theta,
        rope_factor=rope_factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        index_head_count=index_heads,
        index_head_width=index_width,
        index_top_k=index_top_k,
        hyper_multiplier=hyper,
        sinkhorn_iterations=sinkhorn,
        norm_epsilon=norm_epsilon,
        hyper_epsilon=hyper_epsilon,
        route_scale=route_scale,
        routed_swiglu_limits=routed_limits,
        shared_swiglu_limits=shared_limits,
    )


@dataclass
class Range:
    label: str
    role: str
    source_object: str
    source_offset: int
    source_bytes: int
    path: Path
    tensor: GGUFTensor
    slice_index: int | None = None


@dataclass
class Artifact:
    root: Path
    scope: str
    layer: int
    model_id: str
    revision: str
    payload_bytes: int
    ranges: list[Range]
    source_plan_sha256: str
    routing_authority_verified: bool


def load_artifact(path: Path, expected_scope: str, store: ModelStore,
                  expected_layer: int = 0) -> Artifact:
    root = path.resolve()
    manifest_path = root if root.is_file() else root / "vastloom-range-artifact.json"
    if not manifest_path.is_file():
        raise OracleError(f"range artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise OracleError(f"range artifact format is not {ARTIFACT_FORMAT}: {manifest_path}")
    if manifest.get("model_id") != store.model_id or manifest.get("revision") != store.revision:
        raise OracleError("range artifact model identity does not match the metadata store")
    scope = manifest.get("scope")
    if scope != expected_scope:
        raise OracleError(
            f"range artifact scope {scope!r} is not the expected {expected_scope!r}: {manifest_path}"
        )
    layer = manifest.get("layer")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer != expected_layer:
        raise OracleError(
            f"range artifact layer {layer!r} is not the expected layer {expected_layer}: "
            f"{manifest_path}"
        )
    entries = manifest.get("ranges")
    if not isinstance(entries, list) or not entries:
        raise OracleError("range artifact lists no ranges")
    if int(manifest.get("range_count", -1)) != len(entries):
        raise OracleError("range artifact range_count does not match its range list")
    all_tensors: dict[str, list[GGUFTensor]] = store.tensors
    ranges: list[Range] = []
    total = 0
    for entry in entries:
        try:
            label = str(entry["label"])
            role = str(entry["role"])
            source_object = str(entry["source_object"])
            source_offset = int(entry["source_offset"])
            source_bytes = int(entry["source_bytes"])
            local = str(entry["artifact"])
            declared = int(entry["bytes"])
            sha256 = entry.get("sha256")
        except (KeyError, TypeError, ValueError) as error:
            raise OracleError(f"range artifact entry is malformed: {entry!r}") from error
        if declared != source_bytes:
            raise OracleError(f"range {label} local and source byte counts differ")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise OracleError(f"range {label} has no SHA-256 digest")
        local_path = (manifest_path.parent / local).resolve()
        if not local_path.is_file():
            raise OracleError(f"range payload is missing: {local_path}")
        if local_path.stat().st_size != declared:
            raise OracleError(f"range payload size mismatch: {local_path}")
        candidates = all_tensors.get(source_object)
        if candidates is None:
            raise OracleError(f"range {label} names an unknown source object {source_object}")
        exact = [tensor for tensor in candidates
                 if tensor.offset == source_offset and tensor.encoded_bytes == source_bytes]
        slice_index = None
        if exact:
            tensor = exact[0]
        else:
            containing = [tensor for tensor in candidates
                          if tensor.offset <= source_offset
                          and source_offset + source_bytes <= tensor.offset + tensor.encoded_bytes]
            if len(containing) != 1:
                raise OracleError(
                    f"range {label} does not map to exactly one GGUF tensor "
                    f"({len(containing)} candidates)"
                )
            tensor = containing[0]
            if tensor.slices <= 1:
                raise OracleError(f"range {label} is a partial slice of a non-packed tensor")
            slice_bytes = tensor.encoded_bytes // tensor.slices
            delta = source_offset - tensor.offset
            if source_bytes != slice_bytes or delta % slice_bytes != 0:
                raise OracleError(f"range {label} is not an aligned packed-tensor slice")
            slice_index = delta // slice_bytes
        local_tensor = GGUFTensor(
            name=tensor.name,
            dims=tensor.dims[:2] if slice_index is not None else tensor.dims,
            ggml_type=tensor.ggml_type,
            shard=local_path.name,
            offset=0,
            encoded_bytes=source_bytes,
        )
        local_tensor.offset = 0
        ranges.append(Range(label, role, source_object, source_offset, source_bytes,
                            local_path, local_tensor, slice_index))
        total += declared
    if total != int(manifest.get("payload_bytes", -1)):
        raise OracleError("range artifact payload_bytes does not match the range total")
    return Artifact(
        root=manifest_path.parent,
        scope=scope,
        layer=layer,
        model_id=str(manifest["model_id"]),
        revision=str(manifest["revision"]),
        payload_bytes=total,
        ranges=ranges,
        source_plan_sha256=str(manifest.get("source_plan_sha256", "")),
        routing_authority_verified=bool(manifest.get("routing_authority_verified", False)),
    )


# ---------------------------------------------------------------------------
# Layer tensor layout
# ---------------------------------------------------------------------------

@dataclass
class MatrixRef:
    name: str
    tensor: GGUFTensor
    rows: int
    columns: int
    ggml_type: int
    path: Path

    def read(self, offset: int, count: int) -> bytes:
        with self.path.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(count)
        if len(payload) != count:
            raise OracleError(f"short read from {self.path} at {offset} ({count} bytes)")
        return payload

    def matvec(self, vector: np.ndarray, *, row_start: int = 0,
               row_count: int | None = None, chunk_rows: int = 1024) -> np.ndarray:
        if vector.size != self.columns:
            raise OracleError(
                f"{self.name}: input has {vector.size} values, expected {self.columns}"
            )
        if row_count is None:
            row_count = self.rows
        if row_start < 0 or row_count <= 0 or row_start + row_count > self.rows:
            raise OracleError(f"{self.name}: row range is outside the matrix")
        if self.ggml_type not in TYPE_NAMES:
            raise OracleError(f"{self.name}: unsupported storage type {self.ggml_type}")
        bytes_per_row = row_bytes(self.ggml_type, self.columns)
        required = storage_bytes(self.ggml_type, self.rows, self.columns)
        if required != self.tensor.encoded_bytes:
            raise OracleError(
                f"{self.name}: tensor payload is {self.tensor.encoded_bytes} bytes, "
                f"storage layout requires {required}"
            )
        source = np.asarray(vector, dtype=np.float64)
        output = np.empty(row_count, dtype=np.float32)
        for start in range(0, row_count, chunk_rows):
            count = min(chunk_rows, row_count - start)
            offset = (row_start + start) * bytes_per_row
            raw = self.read(offset, count * bytes_per_row)
            weights = decode_matrix(raw, self.ggml_type, count, self.columns)
            products = weights.astype(np.float64) * source[None, :]
            sums = np.cumsum(products, axis=1)[:, -1]
            output[start:start + count] = sums.astype(np.float32)
        return output

    def row(self, row: int) -> np.ndarray:
        if row < 0 or row >= self.rows:
            raise OracleError(f"{self.name}: row {row} is outside the matrix")
        bytes_per_row = row_bytes(self.ggml_type, self.columns)
        raw = self.read(row * bytes_per_row, bytes_per_row)
        return decode_matrix(raw, self.ggml_type, 1, self.columns)[0]


@dataclass
class VectorRef:
    name: str
    tensor: GGUFTensor
    elements: int
    ggml_type: int
    path: Path

    def read(self) -> np.ndarray:
        required = storage_bytes(self.ggml_type, 1, self.elements)
        if required != self.tensor.encoded_bytes:
            raise OracleError(
                f"{self.name}: vector payload is {self.tensor.encoded_bytes} bytes, "
                f"storage layout requires {required}"
            )
        with self.path.open("rb") as stream:
            raw = stream.read(required)
        if len(raw) != required:
            raise OracleError(f"short read from {self.path}")
        return decode_matrix(raw, self.ggml_type, 1, self.elements)[0]


@dataclass
class IntegerRef:
    name: str
    tensor: GGUFTensor
    elements: int
    path: Path

    def read(self, rows: int, columns: int) -> np.ndarray:
        required = rows * columns * 4
        if required != self.tensor.encoded_bytes:
            raise OracleError(f"{self.name}: integer table byte count does not match its shape")
        with self.path.open("rb") as stream:
            raw = stream.read(required)
        return np.frombuffer(raw, dtype="<i4").reshape(rows, columns).copy()


TRUNK_MATRIX_LAYOUT = [
    ("attn_kv.weight", "attn.wkv.weight", lambda c: (c.head_width, c.width), Q8_0),
    ("attn_output_a.weight", "attn.wo_a.weight",
     lambda c: (c.output_groups * c.output_lora_rank, c.head_count * c.head_width // c.output_groups), Q8_0),
    ("attn_output_b.weight", "attn.wo_b.weight",
     lambda c: (c.width, c.output_groups * c.output_lora_rank), Q8_0),
    ("attn_q_a.weight", "attn.wq_a.weight", lambda c: (c.query_lora_rank, c.width), Q5_K),
    ("attn_q_b.weight", "attn.wq_b.weight",
     lambda c: (c.head_count * c.head_width, c.query_lora_rank), Q8_0),
    ("ffn_down_shexp.weight", "ffn.shared_experts.w2.weight",
     lambda c: (c.width, c.intermediate_width), Q6_K),
    ("ffn_gate_inp.weight", "ffn.gate.weight",
     lambda c: (c.expert_count, c.width), BF16),
    ("ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight",
     lambda c: (c.intermediate_width, c.width), Q5_K),
    ("ffn_up_shexp.weight", "ffn.shared_experts.w3.weight",
     lambda c: (c.intermediate_width, c.width), Q5_K),
    ("hc_attn_fn.weight", "hc_attn_fn", lambda c: (c.mix_width, c.flattened_width), F32),
    ("hc_ffn_fn.weight", "hc_ffn_fn", lambda c: (c.mix_width, c.flattened_width), F32),
]

TRUNK_FLOAT_LAYOUT = [
    ("attn_kv_a_norm.weight", "attn.kv_norm.weight", lambda c: c.head_width, F32),
    ("attn_norm.weight", "attn_norm.weight", lambda c: c.width, F32),
    ("attn_q_a_norm.weight", "attn.q_norm.weight", lambda c: c.query_lora_rank, F32),
    ("attn_sinks.weight", "attn.attn_sink", lambda c: c.head_count, F32),
    ("ffn_norm.weight", "ffn_norm.weight", lambda c: c.width, F32),
    ("hc_attn_base.weight", "hc_attn_base", lambda c: c.mix_width, F32),
    ("hc_attn_scale.weight", "hc_attn_scale", lambda c: 3, F32),
    ("hc_ffn_base.weight", "hc_ffn_base", lambda c: c.mix_width, F32),
    ("hc_ffn_scale.weight", "hc_ffn_scale", lambda c: 3, F32),
]

# Compression-ratio-4 layers add an overlapping attention compressor (two ratio-sized
# windows are scored together) and a learned indexer with its own 4:1 compressor.
COMPRESSOR_MATRIX_LAYOUT = [
    ("attn_compressor_kv.weight", "attn.compressor.wkv.weight",
     lambda c, r: (compressor_coefficient(r) * c.head_width, c.width), Q8_0),
    ("attn_compressor_gate.weight", "attn.compressor.wgate.weight",
     lambda c, r: (compressor_coefficient(r) * c.head_width, c.width), Q8_0),
]

COMPRESSOR_FLOAT_LAYOUT = [
    ("attn_compressor_ape.weight", "attn.compressor.ape",
     lambda c, r: (r, compressor_coefficient(r) * c.head_width), F32),
    ("attn_compressor_norm.weight", "attn.compressor.norm.weight", lambda c, r: (1, c.head_width), F32),
]

INDEXER_MATRIX_LAYOUT = [
    ("indexer.attn_q_b.weight", "attn.indexer.wq_b.weight",
     lambda c, r: (c.index_head_count * c.index_head_width, c.query_lora_rank), Q8_0),
    ("indexer.proj.weight", "attn.indexer.weights_proj.weight",
     lambda c, r: (c.index_head_count, c.width), F32),
    ("indexer_compressor_kv.weight", "attn.indexer.compressor.wkv.weight",
     lambda c, r: (compressor_coefficient(r) * c.index_head_width, c.width), Q8_0),
    ("indexer_compressor_gate.weight", "attn.indexer.compressor.wgate.weight",
     lambda c, r: (compressor_coefficient(r) * c.index_head_width, c.width), Q8_0),
]

INDEXER_FLOAT_LAYOUT = [
    ("indexer_compressor_ape.weight", "attn.indexer.compressor.ape",
     lambda c, r: (r, compressor_coefficient(r) * c.index_head_width), F32),
    ("indexer_compressor_norm.weight", "attn.indexer.compressor.norm.weight",
     lambda c, r: (1, c.index_head_width), F32),
]


def compressor_coefficient(ratio: int) -> int:
    """The ratio-4 compressor scores the previous and current windows together."""
    return 2 if ratio == 4 else 1

GLOBAL_MATRIX_LAYOUT = [
    ("token_embd.weight", "embed.weight", lambda c: (c.vocabulary_size, c.width), Q4_K),
    ("output.weight", "head.weight", lambda c: (c.vocabulary_size, c.width), Q4_K),
    ("output_hc_fn.weight", "hc_head_fn", lambda c: (c.hyper_multiplier, c.flattened_width), F32),
]

GLOBAL_FLOAT_LAYOUT = [
    ("output_norm.weight", "norm.weight", lambda c: c.width, F32),
    ("output_hc_base.weight", "hc_head_base", lambda c: c.hyper_multiplier, F32),
    ("output_hc_scale.weight", "hc_head_scale", lambda c: 1, F32),
]


def _tensor_map(artifact: Artifact) -> dict[str, Range]:
    result: dict[str, Range] = {}
    for entry in artifact.ranges:
        name = entry.tensor.name
        if entry.slice_index is not None:
            continue
        if name in result:
            raise OracleError(f"artifact {artifact.scope} maps tensor {name} twice")
        result[name] = entry
    return result


def _make_matrix(entry: Range, rows: int, columns: int, ggml_type: int,
                 expected_label_prefix: str) -> MatrixRef:
    if entry.tensor.ggml_type != ggml_type:
        raise OracleError(
            f"{expected_label_prefix}: {entry.tensor.name} is {TYPE_NAMES.get(entry.tensor.ggml_type)}"
            f", expected {TYPE_NAMES[ggml_type]}"
        )
    if entry.tensor.rows != rows or entry.tensor.columns != columns:
        raise OracleError(
            f"{entry.tensor.name}: shape is {entry.tensor.rows}x{entry.tensor.columns}, "
            f"expected {rows}x{columns}"
        )
    if storage_bytes(ggml_type, rows, columns) != entry.tensor.encoded_bytes:
        raise OracleError(f"{entry.tensor.name}: encoded byte count does not match its shape")
    return MatrixRef(entry.tensor.name, entry.tensor, rows, columns, ggml_type, entry.path)


def _make_vector(entry: Range, elements: int, ggml_type: int) -> VectorRef:
    if entry.tensor.ggml_type != ggml_type:
        raise OracleError(f"{entry.tensor.name}: unexpected vector storage type")
    product = 1
    for dimension in entry.tensor.dims:
        product *= int(dimension)
    if product != elements:
        raise OracleError(f"{entry.tensor.name}: vector length does not match the graph")
    return VectorRef(entry.tensor.name, entry.tensor, elements, ggml_type, entry.path)


def build_layer(store: ModelStore, trunk: Artifact, globals_: Artifact,
                layer_index: int) -> dict[str, object]:
    config = store.config
    if layer_index < 0 or layer_index >= config.layer_count:
        raise OracleError(f"layer {layer_index} is outside the checkpoint")
    ratio = config.compression_ratios[layer_index]
    if ratio not in (0, 4):
        raise OracleError(
            f"layer {layer_index} compression ratio {ratio} is not modelled by this oracle "
            f"(only 0 and 4)"
        )
    prefix = f"blk.{layer_index}."
    consumer_prefix = f"layers.{layer_index}."
    trunk_by_name = _tensor_map(trunk)
    global_by_name = _tensor_map(globals_)
    layer: dict[str, object] = {"matrices": {}, "vectors": {}, "compression_ratio": ratio}
    matrices: dict[str, MatrixRef] = layer["matrices"]  # type: ignore[assignment]
    vectors: dict[str, VectorRef] = layer["vectors"]  # type: ignore[assignment]
    consumed: set[str] = set()

    def add_trunk_matrix(source: str, consumer: str, rows: int, columns: int,
                         ggml_type: int) -> None:
        entry = trunk_by_name.get(source)
        if entry is None:
            raise OracleError(f"trunk artifact is missing {source}")
        matrices[consumer] = _make_matrix(entry, rows, columns, ggml_type, source)
        consumed.add(source)

    def add_trunk_vector(source: str, consumer: str, elements: int,
                         ggml_type: int) -> None:
        entry = trunk_by_name.get(source)
        if entry is None:
            raise OracleError(f"trunk artifact is missing {source}")
        vectors[consumer] = _make_vector(entry, elements, ggml_type)
        consumed.add(source)

    for suffix, consumer_suffix, shape, ggml_type in TRUNK_MATRIX_LAYOUT:
        rows, columns = shape(config)
        add_trunk_matrix(prefix + suffix, consumer_prefix + consumer_suffix,
                         rows, columns, ggml_type)
    for suffix, consumer_suffix, elements, ggml_type in TRUNK_FLOAT_LAYOUT:
        add_trunk_vector(prefix + suffix, consumer_prefix + consumer_suffix,
                         elements(config), ggml_type)
    if ratio != 0:
        for suffix, consumer_suffix, shape, ggml_type in COMPRESSOR_MATRIX_LAYOUT:
            rows, columns = shape(config, ratio)
            add_trunk_matrix(prefix + suffix, consumer_prefix + consumer_suffix,
                             rows, columns, ggml_type)
        for suffix, consumer_suffix, shape, ggml_type in COMPRESSOR_FLOAT_LAYOUT:
            rows, columns = shape(config, ratio)
            if suffix.startswith("attn_compressor_ape") and rows != ratio:
                raise OracleError(f"{prefix}{suffix}: APE row count is not the compression ratio")
            add_trunk_vector(prefix + suffix, consumer_prefix + consumer_suffix,
                             rows * columns, ggml_type)
    if ratio == 4:
        for suffix, consumer_suffix, shape, ggml_type in INDEXER_MATRIX_LAYOUT:
            rows, columns = shape(config, ratio)
            add_trunk_matrix(prefix + suffix, consumer_prefix + consumer_suffix,
                             rows, columns, ggml_type)
        for suffix, consumer_suffix, shape, ggml_type in INDEXER_FLOAT_LAYOUT:
            rows, columns = shape(config, ratio)
            add_trunk_vector(prefix + suffix, consumer_prefix + consumer_suffix,
                             rows * columns, ggml_type)
    for source, consumer, shape, ggml_type in GLOBAL_MATRIX_LAYOUT:
        entry = global_by_name.get(source)
        if entry is None:
            raise OracleError(f"global artifact is missing {source}")
        rows, columns = shape(config)
        matrices[consumer] = _make_matrix(entry, rows, columns, ggml_type, source)
    for source, consumer, elements, ggml_type in GLOBAL_FLOAT_LAYOUT:
        entry = global_by_name.get(source)
        if entry is None:
            raise OracleError(f"global artifact is missing {source}")
        vectors[consumer] = _make_vector(entry, elements(config), ggml_type)
    table_name = prefix + "ffn_gate_tid2eid.weight"
    table_entry = trunk_by_name.get(table_name)
    if table_entry is None:
        raise OracleError(f"trunk artifact is missing {table_name}")
    if table_entry.tensor.ggml_type != I32:
        raise OracleError("hash routing table is not I32")
    if table_entry.tensor.dims != (config.experts_per_token, config.vocabulary_size):
        raise OracleError("hash routing table shape does not match the checkpoint")
    layer["table"] = IntegerRef(
        table_entry.tensor.name, table_entry.tensor,
        config.vocabulary_size * config.experts_per_token, table_entry.path
    )
    consumed.add(table_name)
    unmodelled = sorted(set(trunk_by_name) - consumed)
    if unmodelled:
        raise OracleError(
            f"trunk artifact for layer {layer_index} carries tensors this oracle does not "
            f"model: {', '.join(unmodelled)}"
        )
    return layer


@dataclass
class ExpertBundle:
    expert_id: int
    matrices: dict[str, MatrixRef]
    # (source_object, source_offset, source_bytes) per role. Selected experts shared by two
    # prompt-token routes must come from the same immutable source span.
    sources: dict[str, tuple[str, int, int]]


def build_experts(store: ModelStore, experts: Artifact,
                  layer_index: int) -> dict[int, ExpertBundle]:
    config = store.config
    prefix = f"blk.{layer_index}."
    bundles: dict[int, ExpertBundle] = {}
    for entry in experts.ranges:
        if entry.slice_index is None:
            raise OracleError(f"selected-expert range {entry.label} is not a packed slice")
        label = entry.label
        if not (label.startswith("expert[") and "].range[" in label):
            raise OracleError(f"selected-expert range label is not canonical: {label}")
        identifier = label[len("expert["):label.index("]")]
        try:
            _slot, expert_text = identifier.split(":")
            expert_id = int(expert_text)
            range_index = int(label.rsplit(".range[", 1)[1].rstrip("]"))
        except ValueError as error:
            raise OracleError(f"selected-expert range label is malformed: {label}") from error
        if expert_id != entry.slice_index:
            raise OracleError(f"selected-expert range expert id disagrees with its offset: {label}")
        if expert_id >= config.expert_count or range_index not in (0, 1, 2):
            raise OracleError(f"selected-expert range identity is out of range: {label}")
        roles = {0: ("ffn_down_exps", "w2", (config.width, config.intermediate_width), IQ3_XXS),
                 1: ("ffn_gate_exps", "w1", (config.intermediate_width, config.width), IQ2_XXS),
                 2: ("ffn_up_exps", "w3", (config.intermediate_width, config.width), IQ2_XXS)}
        suffix, role, (rows, columns), ggml_type = roles[range_index]
        expected_name = f"{prefix}{suffix}.weight"
        if entry.tensor.name != expected_name:
            raise OracleError(
                f"{label}: mapped tensor {entry.tensor.name} is not {expected_name}"
            )
        if entry.tensor.ggml_type != ggml_type:
            raise OracleError(f"{label}: expert tensor storage type is wrong")
        matrix = MatrixRef(entry.tensor.name, entry.tensor, rows, columns, ggml_type, entry.path)
        bundle = bundles.setdefault(expert_id, ExpertBundle(expert_id, {}, {}))
        if role in bundle.matrices:
            raise OracleError(f"expert {expert_id} has duplicate {role} matrices")
        bundle.matrices[role] = matrix
        bundle.sources[role] = (entry.source_object, entry.source_offset, entry.source_bytes)
    for expert_id, bundle in bundles.items():
        if set(bundle.matrices) != {"w1", "w2", "w3"}:
            raise OracleError(f"expert {expert_id} bundle is incomplete")
    return bundles


def merge_expert_artifacts(store: ModelStore, artifacts: list[Artifact],
                           layer_index: int) -> dict[int, ExpertBundle]:
    """Merge the per-route selected-expert artifacts of one layer.

    A prompt gives each token its own authoritative route, so a layer may carry
    several artifact sets. An expert selected by two tokens appears in both; the
    merged bundle is admitted only when every copy names the same source spans.
    """
    if not artifacts:
        raise OracleError(f"layer {layer_index} has no selected-expert artifact")
    merged: dict[int, ExpertBundle] = {}
    for artifact in artifacts:
        if artifact.routing_authority_verified is not True:
            raise OracleError(
                f"selected-expert artifact {artifact.root} does not carry verified "
                f"routing authority"
            )
        for expert_id, bundle in build_experts(store, artifact, layer_index).items():
            existing = merged.get(expert_id)
            if existing is None:
                merged[expert_id] = bundle
                continue
            if existing.sources != bundle.sources:
                raise OracleError(
                    f"expert {expert_id} of layer {layer_index} is bound to two different "
                    f"source spans by the merged artifacts"
                )
    return merged


# ---------------------------------------------------------------------------
# Layer equations
# ---------------------------------------------------------------------------

def rms_norm(values: np.ndarray, weight: np.ndarray, rows: int, width: int,
             epsilon: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32).reshape(rows, width)
    weight32 = np.asarray(weight, dtype=np.float32)
    output = np.empty_like(source)
    epsilon64 = float(epsilon)
    weight64 = weight32.astype(np.float64)
    for row in range(rows):
        row_values = source[row].astype(np.float64)
        squares = float(np.cumsum(row_values * row_values)[-1])
        inverse = 1.0 / math.sqrt(squares / float(width) + epsilon64)
        output[row] = (row_values * inverse * weight64).astype(np.float32)
    return output.reshape(-1)


def sqrt_softplus(value: float) -> np.float32:
    promoted = float(value)
    softplus = max(promoted, 0.0) + math.log1p(math.exp(-abs(promoted)))
    return np.float32(math.sqrt(softplus))


def sigmoid32(value: float) -> np.float32:
    promoted = float(value)
    if promoted >= 0.0:
        return np.float32(1.0 / (1.0 + math.exp(-promoted)))
    exponential = math.exp(promoted)
    return np.float32(exponential / (1.0 + exponential))


def linear_serial(matrix: MatrixRef, vector: np.ndarray) -> np.ndarray:
    return matrix.matvec(vector)


def rope_tail(values: np.ndarray, rows: int, heads: int, head_width: int,
              rope_width: int, theta: float, original_context: int, factor: float,
              beta_fast: float, beta_slow: float, position_stride: int,
              inverse: bool, position_offset: int) -> np.ndarray:
    output = np.array(values, dtype=np.float32, copy=True)
    rope_offset = head_width - rope_width
    for pair in range(rope_width // 2):
        frequency = rope_frequency(pair, rope_width, theta, original_context, factor,
                                   beta_fast, beta_slow)
        for row in range(rows):
            position = float(position_offset + row * position_stride)
            angle = position * frequency
            if inverse:
                angle = -angle
            cosine = math.cos(angle)
            sine = math.sin(angle)
            for head in range(heads):
                offset = (row * heads + head) * head_width + rope_offset + pair * 2
                real = float(output[offset])
                imaginary = float(output[offset + 1])
                output[offset] = np.float32(real * cosine - imaginary * sine)
                output[offset + 1] = np.float32(imaginary * cosine + real * sine)
    return output


def rope_frequency(pair: int, rope_width: int, theta: float, original_context: int,
                   factor: float, beta_fast: float, beta_slow: float) -> float:
    frequency = math.pow(theta, -2.0 * float(pair) / float(rope_width))
    if original_context == 0:
        return frequency
    def correction(rotations: float) -> float:
        return (float(rope_width) * math.log(float(original_context) / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(theta)))
    low = max(0.0, math.floor(correction(beta_fast)))
    high = min(float(rope_width - 1), math.ceil(correction(beta_slow)))
    denominator = high - low
    if denominator == 0.0:
        denominator = 0.001
    ramp = min(max((float(pair) - low) / denominator, 0.0), 1.0)
    smooth = 1.0 - ramp
    return frequency / factor * (1.0 - smooth) + frequency * smooth


def normalize_queries(query: np.ndarray, tokens: int, heads: int, width: int,
                      epsilon: float) -> np.ndarray:
    output = np.array(query, dtype=np.float32, copy=True)
    for token in range(tokens):
        for head in range(heads):
            offset = (token * heads + head) * width
            values = output[offset:offset + width].astype(np.float64)
            squares = float(np.cumsum(values * values)[-1])
            inverse = 1.0 / math.sqrt(squares / float(width) + float(epsilon))
            output[offset:offset + width] = (values * inverse).astype(np.float32)
    return output


def hc_split_sinkhorn(mixes: np.ndarray, scale: np.ndarray, base: np.ndarray,
                      multiplier: int, iterations: int, epsilon: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilon32 = np.float32(epsilon)
    pre = np.empty(multiplier, dtype=np.float32)
    post = np.empty(multiplier, dtype=np.float32)
    combination = np.empty(multiplier * multiplier, dtype=np.float32)
    for index in range(multiplier):
        value = np.float32(np.float32(mixes[index] * scale[0]) + np.float32(base[index]))
        pre[index] = np.float32(sigmoid32(float(value)) + epsilon32)
        value = np.float32(np.float32(mixes[multiplier + index] * scale[1]) + np.float32(base[multiplier + index]))
        post[index] = np.float32(np.float32(2.0) * sigmoid32(float(value)))
    for row in range(multiplier):
        values = np.empty(multiplier, dtype=np.float32)
        for column in range(multiplier):
            index = 2 * multiplier + row * multiplier + column
            values[column] = np.float32(np.float32(mixes[index] * scale[2]) + np.float32(base[index]))
        maximum = float(np.max(values))
        exponentials = np.empty(multiplier, dtype=np.float32)
        for column in range(multiplier):
            exponentials[column] = np.float32(math.exp(float(values[column]) - maximum))
        denominator = float(np.cumsum(exponentials.astype(np.float64))[-1])
        for column in range(multiplier):
            combination[row * multiplier + column] = np.float32(
                np.float32(float(exponentials[column]) / denominator) + epsilon32
            )
    def normalize_columns() -> None:
        for column in range(multiplier):
            total = 0.0
            for row in range(multiplier):
                total += float(combination[row * multiplier + column])
            for row in range(multiplier):
                combination[row * multiplier + column] = np.float32(
                    float(combination[row * multiplier + column]) / (total + float(epsilon))
                )
    normalize_columns()
    for _ in range(1, iterations):
        for row in range(multiplier):
            total = 0.0
            for column in range(multiplier):
                total += float(combination[row * multiplier + column])
            for column in range(multiplier):
                combination[row * multiplier + column] = np.float32(
                    float(combination[row * multiplier + column]) / (total + float(epsilon))
                )
        normalize_columns()
    return pre, post, combination


@dataclass
class HCPre:
    rows: int
    reduced: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    combination: np.ndarray
    mixes: np.ndarray
    normalized: np.ndarray


def hc_pre(input_values: np.ndarray, config: Config, mixing: MatrixRef,
           scale: np.ndarray, base: np.ndarray) -> HCPre:
    """Per-token Hyper-Connection pre-mix; ``input_values`` is (rows, flattened)."""
    flattened = config.flattened_width
    source = np.asarray(input_values, dtype=np.float32)
    if source.size == 0 or source.size % flattened:
        raise OracleError(
            f"Hyper-Connection input has {source.size} values, expected a multiple of "
            f"{flattened}"
        )
    rows = source.size // flattened
    source = source.reshape(rows, flattened)
    multiplier = config.hyper_multiplier
    normalized = np.empty_like(source)
    mixes_rows = np.empty((rows, config.mix_width), dtype=np.float32)
    pre = np.empty((rows, multiplier), dtype=np.float32)
    post = np.empty((rows, multiplier), dtype=np.float32)
    combination = np.empty((rows, multiplier * multiplier), dtype=np.float32)
    reduced = np.empty((rows, config.width), dtype=np.float32)
    for row in range(rows):
        values = source[row].astype(np.float64)
        squares = float(np.cumsum(values * values)[-1])
        inverse = 1.0 / math.sqrt(squares / float(flattened) + float(config.norm_epsilon))
        normalized[row] = (values * inverse).astype(np.float32)
        mixes_rows[row] = linear_serial(mixing, normalized[row])
        row_pre, row_post, row_combination = hc_split_sinkhorn(
            mixes_rows[row], scale, base, multiplier, config.sinkhorn_iterations,
            config.hyper_epsilon
        )
        pre[row] = row_pre
        post[row] = row_post
        combination[row] = row_combination
        for column in range(config.width):
            value = 0.0
            for copy in range(multiplier):
                value += float(row_pre[copy]) * float(source[row, copy * config.width + column])
            reduced[row, column] = np.float32(value)
    return HCPre(rows, reduced, pre, post, combination, mixes_rows, normalized)


def hc_post(sublayer: np.ndarray, residual: np.ndarray, split: HCPre,
            multiplier: int, width: int) -> np.ndarray:
    """Per-token Hyper-Connection post-mix; the residual is (rows, multiplier, width)."""
    rows = split.rows
    sublayer32 = np.asarray(sublayer, dtype=np.float32)
    residual32 = np.asarray(residual, dtype=np.float32)
    if sublayer32.size != rows * width or residual32.size != rows * multiplier * width:
        raise OracleError("Hyper-Connection post input shape does not match its split")
    sublayer32 = sublayer32.reshape(rows, width)
    residual32 = residual32.reshape(rows, multiplier, width)
    output = np.empty((rows, multiplier, width), dtype=np.float32)
    for row in range(rows):
        for destination in range(multiplier):
            post_value = float(split.post[row, destination])
            for column in range(width):
                value = post_value * float(sublayer32[row, column])
                for source in range(multiplier):
                    value += float(
                        split.combination[row, source * multiplier + destination]
                    ) * float(residual32[row, source, column])
                output[row, destination, column] = np.float32(value)
    return output.reshape(-1)


def sparse_attention(query: np.ndarray, key_value: np.ndarray, sink: np.ndarray,
                     indices: np.ndarray, tokens: int, heads: int, head_width: int,
                     softmax_scale: np.float32) -> np.ndarray:
    """Causal window plus selected compressed slots, per query token and head."""
    index_table = np.asarray(indices, dtype=np.int32)
    if index_table.size % tokens:
        raise OracleError("sparse-attention indices do not match the token count")
    index_table = index_table.reshape(tokens, -1)
    selected_per_token = index_table.shape[1]
    query32 = np.asarray(query, dtype=np.float32)
    key_value32 = np.asarray(key_value, dtype=np.float32)
    if query32.size != tokens * heads * head_width:
        raise OracleError("sparse-attention query shape does not match the token count")
    if key_value32.size % head_width:
        raise OracleError("sparse-attention key/value cache is not a whole number of rows")
    query32 = query32.reshape(tokens, heads, head_width)
    key_value32 = key_value32.reshape(-1, head_width)
    output = np.zeros((tokens, heads, head_width), dtype=np.float32)
    scale64 = float(softmax_scale)
    for token in range(tokens):
        any_valid = False
        for head in range(heads):
            scores = []
            for slot in range(selected_per_token):
                selected = int(index_table[token, slot])
                if selected < 0:
                    scores.append(-math.inf)
                    continue
                if selected >= key_value32.shape[0]:
                    raise OracleError("sparse attention selects an invalid KV index")
                any_valid = True
                q = query32[token, head].astype(np.float64)
                k = key_value32[selected].astype(np.float64)
                dot = float(np.cumsum(q * k)[-1])
                scores.append(dot * scale64)
            maximum = max(scores)
            denominator = math.exp(float(sink[head]) - maximum)
            for score in scores:
                if math.isfinite(score):
                    denominator += math.exp(score - maximum)
            for column in range(head_width):
                accumulated = 0.0
                for slot in range(selected_per_token):
                    selected = int(index_table[token, slot])
                    if selected < 0:
                        continue
                    accumulated += math.exp(scores[slot] - maximum) * float(
                        key_value32[selected, column]
                    )
                output[token, head, column] = np.float32(accumulated / denominator)
        if not any_valid:
            raise OracleError("sparse attention query has no valid selected index")
    return output.reshape(-1)


def compressor_windows(projected: np.ndarray, scores: np.ndarray, ape: np.ndarray,
                       ratio: int, output_width: int) -> np.ndarray:
    """The gated, overlapping window average of ``compress_prefill``, before norm/RoPE.

    ``projected`` and ``scores`` are ``(tokens, coefficient * output_width)``
    float32; ``ape`` is ``(ratio, coefficient * output_width)``. Returns
    ``(tokens // ratio, output_width)`` float32 rows. Fewer tokens than the ratio
    produce no row at all, which is the single-token prefill case of layers 2.
    """
    if projected.ndim != 2 or projected.shape[1] != ape.shape[1]:
        raise OracleError("compressor projections do not match the APE width")
    if scores.shape != projected.shape:
        raise OracleError("compressor scores do not match the projections")
    overlap = ratio == 4
    coefficient = compressor_coefficient(ratio)
    projected_width = projected.shape[1]
    if projected_width != coefficient * output_width or ape.shape[0] != ratio:
        raise OracleError("compressor layout does not match its ratio and output width")
    token_count = projected.shape[0]
    cutoff = token_count - token_count % ratio
    count = cutoff // ratio
    if count == 0:
        return np.empty((0, output_width), dtype=np.float32)
    values = np.empty((count, output_width), dtype=np.float32)
    for window in range(count):
        for dimension in range(output_width):
            maximum = -math.inf
            for candidate in range(coefficient * ratio):
                previous = overlap and candidate < ratio
                if previous and window == 0:
                    continue
                position = candidate % ratio
                source_token = (window - 1 if previous else window) * ratio + position
                column = (0 if previous else output_width if overlap else 0) + dimension
                score = float(scores[source_token, column]) + float(ape[position, column])
                maximum = max(maximum, score)
            denominator = 0.0
            accumulated = 0.0
            for candidate in range(coefficient * ratio):
                previous = overlap and candidate < ratio
                if previous and window == 0:
                    continue
                position = candidate % ratio
                source_token = (window - 1 if previous else window) * ratio + position
                column = (0 if previous else output_width if overlap else 0) + dimension
                score = float(scores[source_token, column]) + float(ape[position, column])
                exponential = math.exp(score - maximum)
                denominator += exponential
                accumulated += exponential * float(projected[source_token, column])
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise OracleError("compressor normalization is invalid")
            values[window, dimension] = np.float32(accumulated / denominator)
    return values


def compressor_prefill(projected: np.ndarray, scores: np.ndarray, ape: np.ndarray,
                       norm_weight: np.ndarray, ratio: int, output_width: int,
                       rope_width: int, config: Config) -> np.ndarray:
    """``compress_prefill``: window average, RMS normalization and compression RoPE."""
    values = compressor_windows(projected, scores, ape, ratio, output_width)
    count = values.shape[0]
    if count == 0:
        return values
    normalized = rms_norm(values.reshape(-1), norm_weight, count, output_width,
                          config.norm_epsilon)
    rotated = rope_tail(normalized, count, 1, output_width, rope_width,
                        config.compression_rope_theta, config.original_context,
                        config.rope_factor, config.beta_fast, config.beta_slow,
                        ratio, False, 0)
    return rotated.reshape(count, output_width)


def indexer_selection(projected_query: np.ndarray, compressed: np.ndarray,
                      head_weights: np.ndarray, config: Config, ratio: int,
                      token_count: int) -> tuple[list[list[int]], np.ndarray]:
    """Rank the compressed slots of every prefill token as ``indexer_indices`` does.

    The returned values are attention-cache indices (``token_count + slot``) per
    token; a token that has not yet completed a compression window selects
    nothing. The second result is the per-token score table (diagnostic only).
    """
    compressed = np.asarray(compressed, dtype=np.float32)
    if compressed.ndim == 2:
        compressed_count = compressed.shape[0]
        compressed = compressed.reshape(-1)
    elif compressed.ndim == 1:
        if compressed.size % config.index_head_width != 0:
            raise OracleError("indexer compressed cache is not a whole number of slots")
        compressed_count = compressed.size // config.index_head_width
    else:
        raise OracleError("indexer compressed cache has an invalid rank")
    selected = min(config.index_top_k, compressed_count)
    if projected_query.size != token_count * config.index_head_count * config.index_head_width:
        raise OracleError("indexer query does not match the index head geometry")
    if head_weights.size != token_count * config.index_head_count:
        raise OracleError("indexer head weights do not match the index head count")
    query = np.asarray(projected_query, dtype=np.float32).reshape(
        token_count, config.index_head_count, config.index_head_width)
    weights = np.asarray(head_weights, dtype=np.float32).reshape(
        token_count, config.index_head_count)
    head_scale = (1.0 / math.sqrt(float(config.index_head_width))
                  / math.sqrt(float(config.index_head_count)))
    scores_table = np.full((token_count, compressed_count), -math.inf, dtype=np.float32)
    result: list[list[int]] = []
    compressed32 = compressed.reshape(compressed_count, config.index_head_width)
    for token in range(token_count):
        # A token may reference only the compression windows that closed at or
        # before its own position.
        allowed = min((token + 1) // ratio, compressed_count)
        ranking: list[tuple[float, int]] = []
        for compressed_index in range(compressed_count):
            score = -math.inf
            if compressed_index < allowed:
                score = 0.0
                for head in range(config.index_head_count):
                    dot = float(np.cumsum(
                        query[token, head].astype(np.float64)
                        * compressed32[compressed_index].astype(np.float64))[-1])
                    score += max(dot, 0.0) * float(weights[token, head]) * head_scale
                scores_table[token, compressed_index] = np.float32(score)
            ranking.append((score, compressed_index))
        ranking.sort(key=lambda entry: (-entry[0], entry[1]))
        result.append([token_count + index for score, index in ranking[:selected]
                       if math.isfinite(score)])
    return result, scores_table


def _linear_rows(matrix: MatrixRef, values: np.ndarray, rows: int, *,
                 row_start: int = 0, row_count: int | None = None) -> np.ndarray:
    """Apply a matrix to every row; one matvec per row keeps the C++ accumulation order."""
    source = np.asarray(values, dtype=np.float32)
    if source.size != rows * matrix.columns:
        raise OracleError(
            f"{matrix.name}: input has {source.size} values, expected {rows * matrix.columns}"
        )
    source = source.reshape(rows, matrix.columns)
    output_rows = matrix.rows if row_count is None else row_count
    output = np.empty((rows, output_rows), dtype=np.float32)
    for row in range(rows):
        output[row] = matrix.matvec(source[row], row_start=row_start, row_count=row_count)
    return output


def execute_prefill_layer(store: ModelStore, layer: dict[str, object],
                          experts: dict[int, ExpertBundle], token_ids: list[int],
                          layer_index: int, input_hidden: np.ndarray | None,
                          stage_sink: dict[str, np.ndarray]) -> dict[str, object]:
    """Run one hash-routed block over a causal prompt of ``len(token_ids)`` tokens."""
    config = store.config
    matrices: dict[str, MatrixRef] = layer["matrices"]  # type: ignore[assignment]
    vectors: dict[str, VectorRef] = layer["vectors"]  # type: ignore[assignment]
    table: IntegerRef = layer["table"]  # type: ignore[assignment]
    ratio = int(layer["compression_ratio"])  # type: ignore[arg-type]
    name = f"layers.{layer_index}."
    tokens = len(token_ids)
    if tokens == 0:
        raise OracleError("a prefill needs at least one token")
    if tokens > 16:
        raise OracleError("this oracle admits at most sixteen prefill tokens")
    for token in token_ids:
        if token < 0 or token >= config.vocabulary_size:
            raise OracleError(f"token {token} is outside the vocabulary")
    flattened_width = config.flattened_width

    # The embedding seeds layer 0; every later layer consumes the previous
    # layer's post-FFN hyper-connection hidden state, one row per prompt token.
    if input_hidden is None:
        if layer_index != 0:
            raise OracleError(f"layer {layer_index} requires the previous layer hidden state")
        embedding = np.empty((tokens, config.width), dtype=np.float32)
        for row, token in enumerate(token_ids):
            embedding[row] = matrices["embed.weight"].row(token)
        stage_sink["embedding"] = embedding.reshape(-1)
        hidden = np.tile(embedding, (1, config.hyper_multiplier)).astype(np.float32)
    else:
        source = np.asarray(input_hidden, dtype=np.float32)
        if source.size != tokens * flattened_width:
            raise OracleError(
                f"layer {layer_index} input has {source.size} values, "
                f"expected {tokens * flattened_width}"
            )
        hidden = source.reshape(tokens, flattened_width)

    attn_norm_weight = vectors[name + "attn_norm.weight"].read()
    q_norm_weight = vectors[name + "attn.q_norm.weight"].read()
    kv_norm_weight = vectors[name + "attn.kv_norm.weight"].read()
    ffn_norm_weight = vectors[name + "ffn_norm.weight"].read()
    sinks = vectors[name + "attn.attn_sink"].read()
    hc_attn_scale = vectors[name + "hc_attn_scale"].read()
    hc_attn_base = vectors[name + "hc_attn_base"].read()
    hc_ffn_scale = vectors[name + "hc_ffn_scale"].read()
    hc_ffn_base = vectors[name + "hc_ffn_base"].read()

    residual_attention = hidden.copy()
    attention_mix = hc_pre(hidden, config, matrices[name + "hc_attn_fn"],
                           hc_attn_scale, hc_attn_base)
    stage_sink["hc_attn_mixes"] = attention_mix.mixes.reshape(-1)
    stage_sink["hc_attn_normalized"] = attention_mix.normalized.reshape(-1)
    stage_sink["hc_attn_reduced"] = attention_mix.reduced.reshape(-1)
    stage_sink["hc_attn_pre"] = attention_mix.pre.reshape(-1)
    stage_sink["hc_attn_post"] = attention_mix.post.reshape(-1)
    stage_sink["hc_attn_combination"] = attention_mix.combination.reshape(-1)

    normalized_attention = rms_norm(
        attention_mix.reduced, attn_norm_weight, tokens, config.width, config.norm_epsilon
    )
    stage_sink["attn_norm"] = normalized_attention

    # A compressed layer switches both directions of RoPE to the compression
    # schedule; on an uncompressed layer the original-context correction is off.
    theta = config.rope_theta if ratio == 0 else config.compression_rope_theta
    original_context = 0 if ratio == 0 else config.original_context

    # MLA query path. Every projection is applied per token: the C++ reference
    # reduces each row independently, so batching rows must not reorder sums.
    query_rank = _linear_rows(matrices[name + "attn.wq_a.weight"], normalized_attention,
                              tokens)
    stage_sink["q_rank"] = query_rank.reshape(-1)
    normalized_query_rank = rms_norm(
        query_rank, q_norm_weight, tokens, config.query_lora_rank, config.norm_epsilon
    )
    stage_sink["q_rank_norm"] = normalized_query_rank
    query = _linear_rows(matrices[name + "attn.wq_b.weight"], normalized_query_rank, tokens)
    query = normalize_queries(query.reshape(-1), tokens, config.head_count,
                              config.head_width, config.norm_epsilon)
    stage_sink["q_normalized"] = query
    query = rope_tail(query, tokens, config.head_count, config.head_width, config.rope_width,
                      theta, original_context, config.rope_factor, config.beta_fast,
                      config.beta_slow, 1, False, 0)
    stage_sink["q"] = query

    key_value = _linear_rows(matrices[name + "attn.wkv.weight"], normalized_attention,
                             tokens)
    key_value = rms_norm(key_value, kv_norm_weight, tokens, config.head_width,
                         config.norm_epsilon)
    stage_sink["kv_norm"] = key_value
    key_value = rope_tail(key_value, tokens, 1, config.head_width, config.rope_width,
                          theta, original_context, config.rope_factor, config.beta_fast,
                          config.beta_slow, 1, False, 0)
    stage_sink["k"] = key_value
    stage_sink["v"] = key_value.copy()

    # Causal window: token t sees min(t + 1, window_size) positions, and slots
    # for positions after t stay -1 so sparse attention excludes them.
    window_selected = min(tokens, config.window_size)
    window_table = np.full((tokens, window_selected), -1, dtype=np.int32)
    for token in range(tokens):
        window_base = token - config.window_size + 1 if token >= config.window_size else 0
        for slot in range(window_selected):
            candidate = window_base + slot
            if candidate <= token:
                window_table[token, slot] = candidate

    compressed_selected = 0
    compressed_indices: list[list[int]] = [[] for _ in range(tokens)]
    compressed_rows = np.empty((0, config.head_width), dtype=np.float32)
    if ratio != 0:
        coefficient = compressor_coefficient(ratio)
        projected_width = coefficient * config.head_width
        compressor_projected = _linear_rows(
            matrices[name + "attn.compressor.wkv.weight"], normalized_attention, tokens)
        compressor_scores = _linear_rows(
            matrices[name + "attn.compressor.wgate.weight"], normalized_attention, tokens)
        compressor_ape = vectors[name + "attn.compressor.ape"].read().reshape(
            ratio, projected_width)
        compressor_norm = vectors[name + "attn.compressor.norm.weight"].read()
        stage_sink["attention_compressor_projected"] = compressor_projected.reshape(-1)
        stage_sink["attention_compressor_scores"] = compressor_scores.reshape(-1)
        compressed_rows = compressor_prefill(
            compressor_projected, compressor_scores, compressor_ape, compressor_norm,
            ratio, config.head_width, config.rope_width, config)
        stage_sink["attention_compressed"] = compressed_rows.reshape(-1)
        if ratio == 4:
            index_projected_width = coefficient * config.index_head_width
            indexer_query = _linear_rows(
                matrices[name + "attn.indexer.wq_b.weight"], normalized_query_rank,
                tokens)
            indexer_query = rope_tail(
                indexer_query.reshape(-1), tokens, config.index_head_count,
                config.index_head_width, config.rope_width,
                config.compression_rope_theta, config.original_context,
                config.rope_factor, config.beta_fast, config.beta_slow, 1, False, 0)
            indexer_weights = _linear_rows(
                matrices[name + "attn.indexer.weights_proj.weight"], normalized_attention,
                tokens)
            indexer_projected = _linear_rows(
                matrices[name + "attn.indexer.compressor.wkv.weight"],
                normalized_attention, tokens)
            indexer_scores = _linear_rows(
                matrices[name + "attn.indexer.compressor.wgate.weight"],
                normalized_attention, tokens)
            indexer_ape = vectors[name + "attn.indexer.compressor.ape"].read().reshape(
                ratio, index_projected_width)
            indexer_norm = vectors[name + "attn.indexer.compressor.norm.weight"].read()
            indexer_compressed = compressor_prefill(
                indexer_projected, indexer_scores, indexer_ape, indexer_norm, ratio,
                config.index_head_width, config.rope_width, config)
            compressed_indices, _ranking = indexer_selection(
                indexer_query, indexer_compressed, indexer_weights, config, ratio,
                token_count=tokens)
            compressed_selected = min(config.index_top_k,
                                      compressed_rows.shape[0])
            stage_sink["indexer_query"] = indexer_query.reshape(-1)
            stage_sink["indexer_weights"] = indexer_weights.reshape(-1)
            stage_sink["indexer_compressed"] = indexer_compressed.reshape(-1)
        else:  # pragma: no cover - build_layer admits only ratios 0 and 4
            raise OracleError(
                f"layer {layer_index} compression ratio {ratio} is not modelled by this oracle"
            )
        if compressed_rows.shape[0] != 0:
            key_value = np.concatenate([key_value, compressed_rows.reshape(-1)])
    selected_per_token = window_selected + compressed_selected
    attention_indices = np.full((tokens, selected_per_token), -1, dtype=np.int32)
    for token in range(tokens):
        attention_indices[token, :window_selected] = window_table[token]
        for slot, value in enumerate(compressed_indices[token]):
            if slot >= compressed_selected:
                raise OracleError("the indexer selected more slots than were admitted")
            attention_indices[token, window_selected + slot] = value
    stage_sink["attention_indices"] = attention_indices.reshape(-1)
    softmax_scale = np.float32(1.0 / math.sqrt(float(config.head_width)))
    attended = sparse_attention(query, key_value, sinks, attention_indices, tokens,
                                config.head_count, config.head_width, softmax_scale)
    stage_sink["attention_context"] = attended
    attended = rope_tail(attended, tokens, config.head_count, config.head_width,
                         config.rope_width, theta, original_context, config.rope_factor,
                         config.beta_fast, config.beta_slow, 1, True, 0)
    stage_sink["attention_inverse_rope"] = attended

    group_input = config.head_count * config.head_width // config.output_groups
    grouped = np.empty((tokens, config.output_groups * config.output_lora_rank),
                       dtype=np.float32)
    for token in range(tokens):
        for group in range(config.output_groups):
            group_start = (token * config.head_count * config.head_width
                           + group * group_input)
            group_values = attended[group_start:group_start + group_input]
            projected = matrices[name + "attn.wo_a.weight"].matvec(
                group_values,
                row_start=group * config.output_lora_rank,
                row_count=config.output_lora_rank,
            )
            grouped[token, group * config.output_lora_rank:
                    (group + 1) * config.output_lora_rank] = projected
    stage_sink["attention_grouped"] = grouped.reshape(-1)
    attention_output = _linear_rows(matrices[name + "attn.wo_b.weight"], grouped, tokens)
    stage_sink["attention_output"] = attention_output.reshape(-1)

    attention_post = hc_post(attention_output, residual_attention, attention_mix,
                             config.hyper_multiplier, config.width)
    stage_sink["attention_post"] = attention_post

    residual_ffn = attention_post.copy()
    ffn_mix = hc_pre(attention_post, config, matrices[name + "hc_ffn_fn"],
                     hc_ffn_scale, hc_ffn_base)
    stage_sink["hc_ffn_mixes"] = ffn_mix.mixes.reshape(-1)
    stage_sink["hc_ffn_normalized"] = ffn_mix.normalized.reshape(-1)
    stage_sink["hc_ffn_reduced"] = ffn_mix.reduced.reshape(-1)
    stage_sink["hc_ffn_pre"] = ffn_mix.pre.reshape(-1)
    stage_sink["hc_ffn_post"] = ffn_mix.post.reshape(-1)
    stage_sink["hc_ffn_combination"] = ffn_mix.combination.reshape(-1)
    normalized_ffn = rms_norm(ffn_mix.reduced, ffn_norm_weight, tokens, config.width,
                              config.norm_epsilon)
    stage_sink["ffn_norm"] = normalized_ffn

    # Router: BF16 gate projection, sqrt-softplus activation, hash table route.
    # Each prompt token reads its own table row and normalizes its own weights.
    router_logits = _linear_rows(matrices[name + "ffn.gate.weight"], normalized_ffn, tokens)
    stage_sink["router_logits"] = router_logits.reshape(-1)
    router_scores = np.empty_like(router_logits)
    for token in range(tokens):
        for column in range(config.expert_count):
            router_scores[token, column] = sqrt_softplus(float(router_logits[token, column]))
    stage_sink["router_scores"] = router_scores.reshape(-1)
    hash_table = table.read(config.vocabulary_size, config.experts_per_token)
    per_token_routes: list[list[int]] = []
    per_token_weights: list[np.ndarray] = []
    for token, token_id in enumerate(token_ids):
        selected = [int(value) for value in hash_table[token_id]]
        if len(set(selected)) == 0 or any(
                value < 0 or value >= config.expert_count for value in selected):
            raise OracleError("hash routing table selected an invalid expert")
        weights = np.empty(len(selected), dtype=np.float32)
        total = 0.0
        for value in selected:
            total += float(router_scores[token, value])
        if not math.isfinite(total) or total <= 0.0:
            raise OracleError("selected route weights cannot be normalized")
        for slot, expert in enumerate(selected):
            weights[slot] = np.float32(
                float(router_scores[token, expert]) / total * float(config.route_scale)
            )
        per_token_routes.append(selected)
        per_token_weights.append(weights)
    stage_sink["route_indices"] = np.array(per_token_routes, dtype=np.int32).reshape(-1)
    stage_sink["route_weights"] = np.array(per_token_weights,
                                           dtype=np.float32).reshape(-1)

    # Shared expert (SwiGLU with the layer's shared clamp schedule).
    shared_limit = np.float32(config.shared_swiglu_limits[layer_index])
    shared_gate = _linear_rows(matrices[name + "ffn.shared_experts.w1.weight"],
                               normalized_ffn, tokens)
    shared_up = _linear_rows(matrices[name + "ffn.shared_experts.w3.weight"],
                             normalized_ffn, tokens)
    stage_sink["shared_gate"] = shared_gate.reshape(-1)
    stage_sink["shared_up"] = shared_up.reshape(-1)
    shared_activated = np.empty((tokens, config.intermediate_width), dtype=np.float32)
    for token in range(tokens):
        for column in range(config.intermediate_width):
            gate_value = float(shared_gate[token, column])
            up_value = float(shared_up[token, column])
            if float(shared_limit) > 0.0:
                gate_value = min(gate_value, float(shared_limit))
                up_value = max(-float(shared_limit), min(up_value, float(shared_limit)))
            shared_activated[token, column] = np.float32(
                gate_value / (1.0 + math.exp(-gate_value)) * up_value
            )
    stage_sink["shared_activated"] = shared_activated.reshape(-1)
    shared_output = _linear_rows(matrices[name + "ffn.shared_experts.w2.weight"],
                                 shared_activated, tokens)
    stage_sink["shared_output"] = shared_output.reshape(-1)

    # Routed experts: the artifacts must contain exactly the union of the prompt
    # tokens' routes. The contribution order follows the C++ executor (experts
    # ascending, then tokens, then slots), which fixes the float32 sum.
    selected_union = {expert for route in per_token_routes for expert in route}
    if set(experts) != selected_union:
        raise OracleError(
            f"selected-expert artifacts hold {sorted(experts)}, the routers selected "
            f"{sorted(selected_union)}"
        )
    routed_limit = np.float32(config.routed_swiglu_limits[layer_index])
    moe = np.zeros((tokens, config.width), dtype=np.float32)
    for expert in range(config.expert_count):
        if expert not in selected_union:
            continue
        bundle = experts[expert]
        for token in range(tokens):
            for slot, routed_expert in enumerate(per_token_routes[token]):
                if routed_expert != expert:
                    continue
                routing = float(per_token_weights[token][slot])
                gate = bundle.matrices["w1"].matvec(normalized_ffn[
                    token * config.width:(token + 1) * config.width])
                up = bundle.matrices["w3"].matvec(normalized_ffn[
                    token * config.width:(token + 1) * config.width])
                prefix = (f"expert_{expert}" if tokens == 1
                          else f"expert_{expert}_t{token}")
                stage_sink[f"{prefix}_gate"] = gate
                stage_sink[f"{prefix}_up"] = up
                activated = np.empty(config.intermediate_width, dtype=np.float32)
                for column in range(config.intermediate_width):
                    gate_value = float(gate[column])
                    up_value = float(up[column])
                    if float(routed_limit) > 0.0:
                        gate_value = min(gate_value, float(routed_limit))
                        up_value = max(-float(routed_limit),
                                       min(up_value, float(routed_limit)))
                    value = gate_value / (1.0 + math.exp(-gate_value)) * up_value
                    value *= routing
                    activated[column] = np.float32(value)
                stage_sink[f"{prefix}_activated"] = activated
                expert_output = bundle.matrices["w2"].matvec(activated)
                stage_sink[f"{prefix}_output"] = expert_output
                moe[token] += expert_output
    moe += shared_output
    stage_sink["moe_output"] = moe.reshape(-1)

    ffn_post = hc_post(moe, residual_ffn, ffn_mix, config.hyper_multiplier, config.width)
    stage_sink["ffn_post"] = ffn_post
    return {
        "hidden": ffn_post,
        "routes": per_token_routes,
        "route_weights": per_token_weights,
        "attention_indices": attention_indices,
        "window_selected": window_selected,
        "compressed_selected": compressed_selected,
        "compressed_rows": int(compressed_rows.shape[0]),
    }


def execute_layer(store: ModelStore, layer: dict[str, object],
                  experts: dict[int, ExpertBundle], token: int, layer_index: int,
                  input_hidden: np.ndarray | None,
                  stage_sink: dict[str, np.ndarray]) -> dict[str, object]:
    """Single-token wrapper over ``execute_prefill_layer`` (the historical CLI path)."""
    flat_input = None
    if input_hidden is not None:
        source = np.asarray(input_hidden, dtype=np.float32)
        if source.size != store.config.flattened_width:
            raise OracleError(
                f"layer {layer_index} input has {source.size} values, "
                f"expected {store.config.flattened_width}"
            )
        flat_input = source.reshape(-1)
    result = execute_prefill_layer(store, layer, experts, [token], layer_index,
                                   flat_input, stage_sink)
    return {
        "hidden": np.asarray(result["hidden"], dtype=np.float32).reshape(-1),
        "route": result["routes"][0],
        "route_weights": [float(value) for value in result["route_weights"][0]],
        "attention_indices": [int(value) for value in result["attention_indices"][0]],
    }


def final_head(store: ModelStore, layer: dict[str, object], hidden: np.ndarray,
               stage_sink: dict[str, np.ndarray], prefix: str = "") -> dict[str, object]:
    """Hyper-Connection head, final RMS normalization and the LM head."""
    config = store.config
    matrices: dict[str, MatrixRef] = layer["matrices"]  # type: ignore[assignment]
    vectors: dict[str, VectorRef] = layer["vectors"]  # type: ignore[assignment]
    head_mixing = matrices["hc_head_fn"]
    head_scale = vectors["hc_head_scale"].read()
    head_base = vectors["hc_head_base"].read()
    flattened = hidden.astype(np.float64)
    squares = float(np.cumsum(flattened * flattened)[-1])
    inverse = 1.0 / math.sqrt(squares / float(config.flattened_width)
                              + float(config.norm_epsilon))
    mixes = head_mixing.matvec(hidden)
    stage_sink[prefix + "hc_head_mixes"] = mixes
    reduced = np.empty(config.width, dtype=np.float32)
    for column in range(config.width):
        value = 0.0
        for copy in range(config.hyper_multiplier):
            mixed = float(mixes[copy]) * inverse
            activation = 1.0 / (1.0 + math.exp(-(
                mixed * float(head_scale[0]) + float(head_base[copy])
            )))
            value += (activation + float(config.hyper_epsilon)) * float(
                hidden[copy * config.width + column]
            )
        reduced[column] = np.float32(value)
    stage_sink[prefix + "final_hidden"] = reduced
    final_norm = rms_norm(reduced, vectors["norm.weight"].read(), 1, config.width,
                          config.norm_epsilon)
    stage_sink[prefix + "final_norm"] = final_norm
    logits = matrices["head.weight"].matvec(final_norm, chunk_rows=2048)
    stage_sink[prefix + "logits"] = logits
    return {
        "logits": logits,
        "final_hidden": reduced,
        "argmax": int(np.argmax(logits)),
        "minimum": float(np.min(logits)),
        "maximum": float(np.max(logits)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_stage_files(directory: Path, stages: dict[str, np.ndarray]) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, object] = {}
    for name in sorted(stages):
        values = np.ascontiguousarray(stages[name], dtype=np.float32)
        path = directory / f"{name}.f32"
        path.write_bytes(values.tobytes())
        checksums = array_checksums(values)
        checksums["dtype"] = "float32-le"
        checksums["file"] = str(path)
        entries[name] = checksums
    return entries


def stage_summary(stages: dict[str, np.ndarray]) -> dict[str, object]:
    entries: dict[str, object] = {}
    for name in sorted(stages):
        values = np.ascontiguousarray(stages[name], dtype=np.float32)
        entries[name] = array_checksums(values)
    return entries


def run_oracle(model: Path, trunk: Path, experts_path: Path, globals_path: Path,
               token: int, output_dir: Path, write_stages: bool) -> dict[str, object]:
    if sys.byteorder != "little":
        raise OracleError("the oracle requires a little-endian host for raw float32 output")
    verify_codebooks()
    store = load_model_store(model)
    trunk_artifact = load_artifact(trunk, "trunk", store, 0)
    experts_artifact = load_artifact(experts_path, "selected-experts", store, 0)
    globals_artifact = load_artifact(globals_path, "globals", store, 0)
    if experts_artifact.routing_authority_verified is not True:
        raise OracleError("selected-expert artifact does not carry verified routing authority")
    layer = build_layer(store, trunk_artifact, globals_artifact, 0)
    bundles = build_experts(store, experts_artifact, 0)
    stages: dict[str, np.ndarray] = {}
    result = execute_layer(store, layer, bundles, token, 0, None, stages)
    head = final_head(store, layer, result["hidden"], stages)
    output_dir.mkdir(parents=True, exist_ok=True)
    logits = np.ascontiguousarray(head["logits"], dtype=np.float32)
    logits_path = output_dir / "logits.f32"
    logits_path.write_bytes(logits.tobytes())
    summary: dict[str, object] = {
        "format": SUMMARY_FORMAT,
        "ok": True,
        "model_id": store.model_id,
        "revision": store.revision,
        "token": token,
        "layer": 0,
        "model_layers": store.config.layer_count,
        "note": "one layer of a 43-layer model; not the model's output",
        "accumulation": "float64 column-order products, single float32 rounding",
        "logits": {
            "count": int(logits.size),
            "argmax": int(head["argmax"]),
            "minimum": float(head["minimum"]),
            "maximum": float(head["maximum"]),
            **array_checksums(logits),
            "file": str(logits_path),
        },
        "route": {
            "indices": result["route"],
            "weights": result["route_weights"],
            "source": "trunk tensor blk.0.ffn_gate_tid2eid.weight, row token",
        },
        "stages": stage_summary(stages),
        "artifacts": {
            "trunk_bytes": trunk_artifact.payload_bytes,
            "trunk_ranges": len(trunk_artifact.ranges),
            "trunk_plan_sha256": trunk_artifact.source_plan_sha256,
            "expert_bytes": experts_artifact.payload_bytes,
            "expert_ranges": len(experts_artifact.ranges),
            "expert_plan_sha256": experts_artifact.source_plan_sha256,
            "global_bytes": globals_artifact.payload_bytes,
            "global_ranges": len(globals_artifact.ranges),
            "global_plan_sha256": globals_artifact.source_plan_sha256,
        },
        "configuration": {
            "width": store.config.width,
            "intermediate_width": store.config.intermediate_width,
            "head_count": store.config.head_count,
            "head_width": store.config.head_width,
            "rope_width": store.config.rope_width,
            "experts_per_token": store.config.experts_per_token,
            "hyper_multiplier": store.config.hyper_multiplier,
            "sinkhorn_iterations": store.config.sinkhorn_iterations,
            "window_size": store.config.window_size,
            "layer0_compression_ratio": store.config.compression_ratios[0],
        },
    }
    if write_stages:
        summary["stage_files"] = write_stage_files(output_dir / "stages", stages)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_chain_oracle(model: Path, trunks: dict[int, Path], experts: dict[int, Path],
                     globals_path: Path, token: int, layers: int, depths: list[int],
                     output_dir: Path, write_stages: bool) -> dict[str, object]:
    """Run layers 0..``layers``-1 and evaluate the shared head at each depth."""
    if sys.byteorder != "little":
        raise OracleError("the oracle requires a little-endian host for raw float32 output")
    verify_codebooks()
    store = load_model_store(model)
    config = store.config
    if layers < 1 or layers > config.hash_layer_count:
        raise OracleError(
            f"the chain admits 1..{config.hash_layer_count} hash-routed layers, got {layers}"
        )
    if sorted(trunks) != list(range(layers)) or sorted(experts) != list(range(layers)):
        raise OracleError("the chain needs exactly one trunk and one expert artifact per layer")
    if not depths or sorted(set(depths)) != depths or depths[-1] > layers or depths[0] < 1:
        raise OracleError("requested depths must be a sorted non-empty subset of 1..layers")
    globals_artifact = load_artifact(globals_path, "globals", store, 0)
    trunk_artifacts: dict[int, Artifact] = {}
    expert_artifacts: dict[int, Artifact] = {}
    built_layers: dict[int, dict[str, object]] = {}
    built_experts: dict[int, dict[int, ExpertBundle]] = {}
    for layer_index in range(layers):
        trunk_artifact = load_artifact(trunks[layer_index], "trunk", store, layer_index)
        expert_artifact = load_artifact(experts[layer_index], "selected-experts", store,
                                        layer_index)
        if expert_artifact.routing_authority_verified is not True:
            raise OracleError(
                f"selected-expert artifact for layer {layer_index} does not carry verified "
                f"routing authority"
            )
        trunk_artifacts[layer_index] = trunk_artifact
        expert_artifacts[layer_index] = expert_artifact
        built_layers[layer_index] = build_layer(store, trunk_artifact, globals_artifact,
                                                layer_index)
        built_experts[layer_index] = build_experts(store, expert_artifact, layer_index)

    stages: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    depth_results: dict[int, dict[str, object]] = {}
    hidden: np.ndarray | None = None
    for layer_index in range(layers):
        layer_stages: dict[str, np.ndarray] = {}
        result = execute_layer(store, built_layers[layer_index], built_experts[layer_index],
                               token, layer_index, hidden, layer_stages)
        hidden = np.ascontiguousarray(result["hidden"], dtype=np.float32)
        layer_stages["hidden"] = hidden
        for stage_name, values in layer_stages.items():
            stages[f"layer{layer_index}.{stage_name}"] = values
        records.append({
            "layer": layer_index,
            "compression_ratio": int(built_layers[layer_index]["compression_ratio"]),
            "route": {
                "indices": result["route"],
                "weights": result["route_weights"],
                "source": f"trunk tensor blk.{layer_index}.ffn_gate_tid2eid.weight, row token",
            },
            "attention_indices": result["attention_indices"],
        })
        depth = layer_index + 1
        if depth in depths:
            head_stages: dict[str, np.ndarray] = {}
            head = final_head(store, built_layers[layer_index], hidden, head_stages,
                              prefix=f"depth{depth}.")
            stages.update(head_stages)
            depth_results[depth] = head
    if hidden is None:  # pragma: no cover - layers >= 1 is enforced above
        raise OracleError("the chain produced no hidden state")

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_summaries: dict[str, object] = {}
    for depth, head in depth_results.items():
        logits = np.ascontiguousarray(head["logits"], dtype=np.float32)
        logits_path = output_dir / f"logits-depth{depth}.f32"
        logits_path.write_bytes(logits.tobytes())
        hidden_path = output_dir / f"hidden-depth{depth}.f32"
        hidden_path.write_bytes(np.ascontiguousarray(head["final_hidden"],
                                                      dtype=np.float32).tobytes())
        depth_summaries[str(depth)] = {
            "logits": {
                "count": int(logits.size),
                "argmax": int(head["argmax"]),
                "minimum": float(head["minimum"]),
                "maximum": float(head["maximum"]),
                **array_checksums(logits),
                "file": str(logits_path),
            },
            "final_hidden": {
                "count": int(np.asarray(head["final_hidden"]).size),
                **array_checksums(np.asarray(head["final_hidden"], dtype=np.float32)),
                "file": str(hidden_path),
            },
        }
    summary: dict[str, object] = {
        "format": CHAIN_SUMMARY_FORMAT,
        "ok": True,
        "model_id": store.model_id,
        "revision": store.revision,
        "token": token,
        "executed_layers": layers,
        "model_layers": config.layer_count,
        "note": f"{layers} of {config.layer_count} layers; not the model's output",
        "accumulation": "float64 column-order products, single float32 rounding",
        "depths": depth_summaries,
        "layers": records,
        "stages": stage_summary(stages),
        "artifacts": {
            "globals": {
                "bytes": globals_artifact.payload_bytes,
                "ranges": len(globals_artifact.ranges),
                "plan_sha256": globals_artifact.source_plan_sha256,
            },
            "layers": {
                str(layer_index): {
                    "trunk_bytes": trunk_artifacts[layer_index].payload_bytes,
                    "trunk_ranges": len(trunk_artifacts[layer_index].ranges),
                    "trunk_plan_sha256": trunk_artifacts[layer_index].source_plan_sha256,
                    "expert_bytes": expert_artifacts[layer_index].payload_bytes,
                    "expert_ranges": len(expert_artifacts[layer_index].ranges),
                    "expert_plan_sha256": expert_artifacts[layer_index].source_plan_sha256,
                }
                for layer_index in range(layers)
            },
        },
        "configuration": {
            "width": config.width,
            "intermediate_width": config.intermediate_width,
            "head_count": config.head_count,
            "head_width": config.head_width,
            "rope_width": config.rope_width,
            "experts_per_token": config.experts_per_token,
            "hyper_multiplier": config.hyper_multiplier,
            "sinkhorn_iterations": config.sinkhorn_iterations,
            "window_size": config.window_size,
            "index_head_count": config.index_head_count,
            "index_head_width": config.index_head_width,
            "index_top_k": config.index_top_k,
            "compression_ratios": [config.compression_ratios[index]
                                   for index in range(layers)],
        },
    }
    if write_stages:
        summary["stage_files"] = write_stage_files(output_dir / "stages", stages)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_prefill_oracle(model: Path, trunks: dict[int, Path],
                       expert_paths: dict[int, list[Path]],
                       globals_path: Path, tokens: list[int], layers: int,
                       output_dir: Path, write_stages: bool) -> dict[str, object]:
    """Run layers 0..``layers``-1 over one causal prompt, head on the last token."""
    if sys.byteorder != "little":
        raise OracleError("the oracle requires a little-endian host for raw float32 output")
    if not tokens or len(tokens) > 16:
        raise OracleError(f"a prefill prompt needs 1..16 tokens, got {len(tokens)}")
    verify_codebooks()
    store = load_model_store(model)
    config = store.config
    if layers < 1 or layers > config.hash_layer_count:
        raise OracleError(
            f"the prompt prefill admits 1..{config.hash_layer_count} hash-routed layers, "
            f"got {layers}"
        )
    if sorted(trunks) != list(range(layers)):
        raise OracleError("the prompt prefill needs exactly one trunk artifact per layer")
    if sorted(expert_paths) != list(range(layers)) or any(
            not expert_paths[layer] for layer in range(layers)):
        raise OracleError("the prompt prefill needs at least one expert artifact per layer")
    globals_artifact = load_artifact(globals_path, "globals", store, 0)
    trunk_artifacts: dict[int, Artifact] = {}
    expert_artifacts: dict[int, list[Artifact]] = {}
    built_layers: dict[int, dict[str, object]] = {}
    built_experts: dict[int, dict[int, ExpertBundle]] = {}
    for layer_index in range(layers):
        trunk_artifact = load_artifact(trunks[layer_index], "trunk", store, layer_index)
        artifacts = [load_artifact(path, "selected-experts", store, layer_index)
                     for path in expert_paths[layer_index]]
        trunk_artifacts[layer_index] = trunk_artifact
        expert_artifacts[layer_index] = artifacts
        built_layers[layer_index] = build_layer(store, trunk_artifact, globals_artifact,
                                                layer_index)
        built_experts[layer_index] = merge_expert_artifacts(store, artifacts, layer_index)

    stages: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    hidden: np.ndarray | None = None
    for layer_index in range(layers):
        layer_stages: dict[str, np.ndarray] = {}
        result = execute_prefill_layer(store, built_layers[layer_index],
                                       built_experts[layer_index], tokens, layer_index,
                                       hidden, layer_stages)
        hidden = np.ascontiguousarray(result["hidden"], dtype=np.float32).reshape(
            len(tokens), config.flattened_width)
        layer_stages["hidden"] = hidden.reshape(-1)
        for stage_name, values in layer_stages.items():
            stages[f"layer{layer_index}.{stage_name}"] = values
        indices = np.asarray(result["attention_indices"], dtype=np.int32)
        records.append({
            "layer": layer_index,
            "compression_ratio": int(built_layers[layer_index]["compression_ratio"]),
            "window_selected": int(result["window_selected"]),
            "compressed_rows": int(result["compressed_rows"]),
            "compressed_selected": int(result["compressed_selected"]),
            "selected_per_token": int(indices.shape[1]),
            "compression_exercised": bool(int(result["compressed_rows"]) > 0
                                          and int(result["compressed_selected"]) > 0),
            "attention_indices_per_token": [[int(value) for value in row]
                                            for row in indices],
            "routes_per_token": [[int(value) for value in route]
                                 for route in result["routes"]],
            "route_sets_per_token": [sorted({int(value) for value in route})
                                     for route in result["routes"]],
            "route_weights_per_token": [[float(value) for value in weights]
                                        for weights in result["route_weights"]],
            "route_source": (f"trunk tensor blk.{layer_index}.ffn_gate_tid2eid.weight, "
                             f"row per prompt token"),
        })
    if hidden is None:  # pragma: no cover - layers >= 1 is enforced above
        raise OracleError("the prefill produced no hidden state")
    head_stages: dict[str, np.ndarray] = {}
    head = final_head(store, built_layers[layers - 1], hidden[-1].reshape(-1), head_stages)
    stages.update(head_stages)

    output_dir.mkdir(parents=True, exist_ok=True)
    logits = np.ascontiguousarray(head["logits"], dtype=np.float32)
    logits_path = output_dir / "logits.f32"
    logits_path.write_bytes(logits.tobytes())
    hidden_path = output_dir / "hidden.f32"
    hidden_path.write_bytes(np.ascontiguousarray(head["final_hidden"],
                                                  dtype=np.float32).tobytes())
    compressed_layers = [
        {
            "layer": record["layer"],
            "ratio": record["compression_ratio"],
            "compressed_rows": record["compressed_rows"],
            "compressed_selected": record["compressed_selected"],
            "window_selected": record["window_selected"],
            "selected_per_token": record["selected_per_token"],
            "compression_exercised": record["compression_exercised"],
        }
        for record in records if int(record["compression_ratio"]) != 0
    ]
    compressed_stage_counts: dict[str, int] = {}
    for layer_index in range(layers):
        for suffix in ("attention_compressed", "indexer_compressed"):
            key = f"layer{layer_index}.{suffix}"
            if key in stages:
                compressed_stage_counts[key] = int(np.asarray(stages[key]).size)
    summary: dict[str, object] = {
        "format": PREFILL_SUMMARY_FORMAT,
        "ok": True,
        "model_id": store.model_id,
        "revision": store.revision,
        "tokens": [int(token) for token in tokens],
        "prompt_tokens": len(tokens),
        "head_position": len(tokens) - 1,
        "executed_layers": layers,
        "model_layers": config.layer_count,
        "note": (f"{layers} of {config.layer_count} layers over a {len(tokens)}-token "
                 f"prefill; not the model's output"),
        "accumulation": "float64 column-order products, single float32 rounding",
        "logits": {
            "count": int(logits.size),
            "argmax": int(head["argmax"]),
            "minimum": float(head["minimum"]),
            "maximum": float(head["maximum"]),
            **array_checksums(logits),
            "file": str(logits_path),
        },
        "final_hidden": {
            "count": int(np.asarray(head["final_hidden"]).size),
            **array_checksums(np.asarray(head["final_hidden"], dtype=np.float32)),
            "file": str(hidden_path),
        },
        "layers": records,
        "compression": {
            "compressed_layers": compressed_layers,
            "compressed_stage_counts": compressed_stage_counts,
            "total_compressed_rows": sum(int(entry["compressed_rows"])
                                         for entry in compressed_layers),
        },
        "stages": stage_summary(stages),
        "artifacts": {
            "globals": {
                "bytes": globals_artifact.payload_bytes,
                "ranges": len(globals_artifact.ranges),
                "plan_sha256": globals_artifact.source_plan_sha256,
            },
            "layers": {
                str(layer_index): {
                    "trunk_bytes": trunk_artifacts[layer_index].payload_bytes,
                    "trunk_ranges": len(trunk_artifacts[layer_index].ranges),
                    "trunk_plan_sha256": trunk_artifacts[layer_index].source_plan_sha256,
                    "expert_artifacts": [
                        {
                            "root": str(artifact.root),
                            "bytes": artifact.payload_bytes,
                            "ranges": len(artifact.ranges),
                            "plan_sha256": artifact.source_plan_sha256,
                            "routing_authority_verified":
                                artifact.routing_authority_verified,
                        }
                        for artifact in expert_artifacts[layer_index]
                    ],
                    "expert_bytes": sum(artifact.payload_bytes
                                        for artifact in expert_artifacts[layer_index]),
                    "expert_ranges": sum(len(artifact.ranges)
                                         for artifact in expert_artifacts[layer_index]),
                    "expert_bundles": len(built_experts[layer_index]),
                }
                for layer_index in range(layers)
            },
        },
        "configuration": {
            "width": config.width,
            "intermediate_width": config.intermediate_width,
            "head_count": config.head_count,
            "head_width": config.head_width,
            "rope_width": config.rope_width,
            "experts_per_token": config.experts_per_token,
            "hyper_multiplier": config.hyper_multiplier,
            "sinkhorn_iterations": config.sinkhorn_iterations,
            "window_size": config.window_size,
            "index_head_count": config.index_head_count,
            "index_head_width": config.index_head_width,
            "index_top_k": config.index_top_k,
            "compression_ratios": [config.compression_ratios[index]
                                   for index in range(layers)],
        },
    }
    if write_stages:
        summary["stage_files"] = write_stage_files(output_dir / "stages", stages)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _parse_layer_paths(values: list[str], option: str) -> dict[int, Path]:
    """Parse ``--trunk PATH`` or repeated ``--trunk LAYER=PATH`` values."""
    if len(values) == 1 and "=" not in values[0]:
        return {0: Path(values[0])}
    result: dict[int, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise OracleError(f"{option} must be LAYER=PATH when the chain has several layers")
        layer_text, path_text = raw.split("=", 1)
        try:
            layer_index = int(layer_text)
        except ValueError as error:
            raise OracleError(f"{option} layer index is not an integer: {raw}") from error
        if layer_index < 0 or not path_text:
            raise OracleError(f"{option} value is malformed: {raw}")
        if layer_index in result:
            raise OracleError(f"{option} repeats layer {layer_index}")
        result[layer_index] = Path(path_text)
    return result


def _parse_layer_path_lists(values: list[str], option: str) -> dict[int, list[Path]]:
    """Parse repeated ``--experts LAYER=PATH`` values; one layer may carry several."""
    result: dict[int, list[Path]] = {}
    for raw in values:
        if "=" not in raw:
            raise OracleError(f"{option} must be LAYER=PATH for a prompt prefill")
        layer_text, path_text = raw.split("=", 1)
        try:
            layer_index = int(layer_text)
        except ValueError as error:
            raise OracleError(f"{option} layer index is not an integer: {raw}") from error
        if layer_index < 0 or not path_text:
            raise OracleError(f"{option} value is malformed: {raw}")
        result.setdefault(layer_index, []).append(Path(path_text))
    return result


def _parse_tokens(text: str) -> list[int]:
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise OracleError("--tokens contains an empty token id")
        try:
            value = int(item)
        except ValueError as error:
            raise OracleError(f"--tokens contains a non-integer value: {item}") from error
        if value < 0:
            raise OracleError(f"--tokens contains a negative token id: {item}")
        result.append(value)
    if not result:
        raise OracleError("--tokens is empty")
    return result


def _parse_depths(text: str | None, layers: int) -> list[int]:
    if text is None:
        return [layers]
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as error:
            raise OracleError(f"--depths contains a non-integer value: {item}") from error
        result.append(value)
    if not result:
        raise OracleError("--depths is empty")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independent NumPy oracle for DeepSeek V4 Flash 0731 layers 0-2."
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="metadata store directory (vastloom-import.json and GGUF headers)")
    parser.add_argument("--trunk", action="append", required=True, metavar="[LAYER=]PATH",
                        help="trunk artifact directory; repeat with LAYER=PATH for a chain")
    parser.add_argument("--experts", action="append", required=True, metavar="[LAYER=]PATH",
                        help="selected-expert artifact directory; repeat with LAYER=PATH")
    parser.add_argument("--globals", type=Path, required=True,
                        help="global range artifact directory")
    parser.add_argument("--layers", type=int, default=1,
                        help="number of consecutive layers to execute (default 1)")
    parser.add_argument("--depths", type=str, default=None,
                        help="comma-separated prefix depths for a chain (default: the depth)")
    parser.add_argument("--token", type=int, default=None,
                        help="single prefill token id (default 0; mutually exclusive with --tokens)")
    parser.add_argument("--tokens", type=str, default=None, metavar="ID,ID,...",
                        help="causal prompt for multi-token prefill, 1..16 tokens; "
                             "--experts LAYER=PATH may then repeat per layer")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="destination for logits, hidden states, summary.json and stages")
    parser.add_argument("--stages", action="store_true",
                        help="also write every stage vector as raw little-endian float32")
    args = parser.parse_args(argv)
    prefill_mode = args.tokens is not None
    chain_mode = (args.layers > 1 or args.depths is not None
                  or any("=" in value for value in args.trunk + args.experts))
    try:
        if args.layers < 1:
            raise OracleError("--layers must be at least 1")
        if prefill_mode:
            if args.token is not None:
                raise OracleError("--token and --tokens are mutually exclusive")
            if args.depths is not None:
                raise OracleError(
                    "--depths is single-token chain only; a prompt prefill evaluates the "
                    "shared head on the last position of the final layer"
                )
            tokens = _parse_tokens(args.tokens)
            trunks = _parse_layer_paths(args.trunk, "--trunk")
            expert_lists = _parse_layer_path_lists(args.experts, "--experts")
            summary = run_prefill_oracle(args.model, trunks, expert_lists, args.globals,
                                         tokens, args.layers, args.output_dir, args.stages)
        else:
            token = 0 if args.token is None else args.token
            trunks = _parse_layer_paths(args.trunk, "--trunk")
            experts = _parse_layer_paths(args.experts, "--experts")
            if chain_mode:
                depths = _parse_depths(args.depths, args.layers)
                summary = run_chain_oracle(args.model, trunks, experts, args.globals, token,
                                           args.layers, depths, args.output_dir, args.stages)
            else:
                if args.layers != 1 or set(trunks) != {0} or set(experts) != {0}:
                    raise OracleError(
                        "single-layer mode takes one plain --trunk and --experts path"
                    )
                summary = run_oracle(args.model, trunks[0], experts[0], args.globals,
                                     token, args.output_dir, args.stages)
    except (OracleError, OSError, ValueError) as error:
        print(f"layer oracle: {error}", file=sys.stderr)
        return 1
    if prefill_mode:
        print(json.dumps({
            "ok": True,
            "prompt_tokens": summary["prompt_tokens"],
            "executed_layers": summary["executed_layers"],
            "logits": summary["logits"]["count"],
            "argmax": summary["logits"]["argmax"],
            "minimum": summary["logits"]["minimum"],
            "maximum": summary["logits"]["maximum"],
            "compression": summary["compression"],
            "summary": summary["summary_path"],
        }, sort_keys=True))
    elif chain_mode:
        print(json.dumps({
            "ok": True,
            "executed_layers": summary["executed_layers"],
            "depths": {
                depth: {
                    "argmax": entry["logits"]["argmax"],
                    "minimum": entry["logits"]["minimum"],
                    "maximum": entry["logits"]["maximum"],
                }
                for depth, entry in summary["depths"].items()
            },
            "summary": summary["summary_path"],
        }, sort_keys=True))
    else:
        print(json.dumps({
            "ok": True,
            "logits": summary["logits"]["count"],
            "argmax": summary["logits"]["argmax"],
            "minimum": summary["logits"]["minimum"],
            "maximum": summary["logits"]["maximum"],
            "route": summary["route"]["indices"],
            "summary": summary["summary_path"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
