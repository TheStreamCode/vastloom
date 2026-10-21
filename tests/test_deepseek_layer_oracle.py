#!/usr/bin/env python3
"""Opt-in parity test for the independent DeepSeek V4 Flash 0731 oracle.

The native probes execute real layers from imported range artifacts and can dump the 129,280
logits and the 4,096-wide pre-normalization hidden state as headerless little-endian float32.
This test runs the probes and the independent NumPy oracle on the same bytes and compares both
dumps.

Four sections:

* **Synthetic, always on.** Hand-written quantized blocks with decoded values, FNV fingerprints
  and matvec outputs pinned in ``tests/test_main.cpp`` against ggml commit ``30bf8685``. No
  artifacts and no C++ call: this checks the six decoders and the serial double-accumulation
  matvec on their own. It validates ``tools/reference_deepseek_v4_layer_oracle.py`` directly.
* **Layer-0 artifact integration, opt-in.** With ``--model``, ``--artifact``, ``--experts``,
  ``--globals`` and ``--vastloom`` the full layer is compared: logits (maximum absolute
  difference <= 1e-3), final hidden state (<= 1e-3 and ``rtol=1e-5``/``atol=1e-6``), route,
  artifact identities, and the oracle's per-stage checksums recomputed from its own files.
* **Chain artifact integration, opt-in.** With repeated ``--chain-trunk LAYER=PATH`` and
  ``--chain-experts TOKEN:LAYER=PATH`` (plus ``--model``/``--globals``/``--vastloom``) both
  tokens run through ``deepseek-chain-probe`` at depths 2 and 3 and through
  ``deepseek-layer-probe`` at depth 1; the oracle runs the same chain once per token and emits
  one logits/hidden pair per depth. Every depth is compared with the same tolerances, the
  per-layer route must match, and the layer-2 compressor/indexer stages are required and
  checked finite. A divergence is reported at the first depth (and therefore the first layer)
  that fails, with the offending values. Selected-expert artifacts are imported per token, so
  the expert map is per token and layer; a shared map would compare token 1 against token 0's
  experts.
* **Multi-token prefill integration, opt-in.** With ``--prefill``, ``--chain-tokens 0,1,2,3,4``
  and fifteen ``--chain-experts TOKEN:LAYER=PATH`` maps, the probe runs
  ``deepseek-chain-probe --tokens 0,1,2,3,4`` twice (determinism) and the oracle runs in prompt
  mode. Last-position logits and final hidden are compared with the same tolerances, every
  per-layer per-token route is checked against the probe JSON and an independent pinned table,
  and layer 2 must report a non-zero compressed-slot stage count with the compressed slot
  selected by positions 3 and 4 only.

Raw FNV equality with the probe trace is reported as a diagnostic, not asserted: a bitwise
hash has no tolerance and the probe's MSVC scalar path can differ from NumPy by one float32
ulp even when every equation matches. ``--require-checksum-equality`` turns that diagnostic
into a failure for a future bit-identical implementation.
Raw FNV equality with the probe trace is reported as a diagnostic, not asserted: a bitwise
hash has no tolerance and the probe's MSVC scalar path can differ from NumPy by one float32
ulp even when every equation matches. ``--require-checksum-equality`` turns that diagnostic
into a failure for a future bit-identical implementation.

The probe must advertise ``--dump-logits`` (checked through ``--help`` and the CLI source). If it
does not, the integration test reports a blocker and exits 2; it never skips the comparison or
falls back to a weaker check. Missing ``--dump-hidden``/``--dump-trace`` are reported as blockers.

The test needs the imported metadata store and artifacts, a built probe and about a gigabyte of
transient memory. Dumps are staged under build/ and removed unless --keep is given. Run it with
an interpreter that provides NumPy:

    & "C:\\path\\to\\python-with-numpy\\python.exe" `
      tests\\test_deepseek_layer_oracle.py `
      --vastloom build\\release\\vastloom.exe `
      --model build\\analysis\\deepseek-0731\\store\\models\\original\\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata `
      --artifact build\\analysis\\deepseek-0731\\store\\models\\optimized\\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-trunk-e62746099f66 `
      --experts build\\analysis\\deepseek-0731\\store\\models\\optimized\\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-l0-experts-c9b27ecd7c32 `
      --globals build\\analysis\\deepseek-0731\\store\\models\\optimized\\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--range-globals-067131537b81 `
      --chain-trunk 0=...l0-trunk... --chain-trunk 1=...l1-trunk... --chain-trunk 2=...l2-trunk... `
      --chain-experts 0:0=...l0-experts-t0... --chain-experts 0:1=...l1-experts-t0... `
      --chain-experts 0:2=...l2-experts-t0... --chain-experts 1:0=...l0-experts-t1... `
      --chain-experts 1:1=...l1-experts-t1... --chain-experts 1:2=...l2-experts-t1...

For the multi-token prefill section (the D3c gate) add `--prefill --chain-tokens 0,1,2,3,4`
and fifteen `--chain-experts TOKEN:LAYER=PATH` maps instead of the six single-token maps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment failure
    raise SystemExit(
        "this test requires NumPy; this interpreter has no numpy module. "
        "Run it with an interpreter that provides NumPy "
        f"(import error: {error})"
    ) from error

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import reference_deepseek_v4_layer_oracle as oracle  # noqa: E402
from host_pressure import host_pressure_reason, skipped  # noqa: E402


CONTRACT = "0731 independent layer-0 oracle parity"
CHAIN_CONTRACT = "0731 independent chain oracle parity (layers 0-2)"
PREFILL_CONTRACT = "0731 independent 5-token prefill oracle parity (layers 0-2)"

LOGITS = 129_280
WIDTH = 4_096
ARGMAX_TOKEN = 30_528
ROUTE_EXPERTS = [254, 222, 245, 200, 53, 35]
MODEL_LAYERS = 43

# Sorted route sets the native probes must keep resolving, per token and layer. These are
# independent of the oracle: a coordinated drift of both implementations still fails here.
CHAIN_ROUTES = {
    0: ([35, 53, 200, 222, 245, 254], [8, 97, 137, 158, 163, 184],
        [43, 53, 108, 115, 132, 143]),
    1: ([23, 57, 115, 122, 202, 239], [21, 63, 126, 150, 170, 219],
        [26, 39, 217, 221, 240, 247]),
}
CHAIN_ARGMAX = {0: {1: 30_528, 2: 86_051, 3: 86_051},
                1: {1: 69_146, 2: 69_146, 3: 69_146}}

# Multi-token prefill: the pinned causal prompt, its native result and the per-layer route of
# every position. These are independent of the oracle; ``chain5-reference.json`` is the source.
PREFILL_TOKENS = (0, 1, 2, 3, 4)
PREFILL_LAYERS = 3
PREFILL_ARGMAX = 20_267
PREFILL_ROUTES = {
    0: ((35, 53, 200, 222, 245, 254), (23, 57, 115, 122, 202, 239),
        (37, 38, 45, 96, 108, 228), (42, 51, 69, 102, 162, 182),
        (7, 67, 81, 154, 239, 246)),
    1: ((8, 97, 137, 158, 163, 184), (21, 63, 126, 150, 170, 219),
        (13, 16, 17, 46, 73, 237), (30, 102, 177, 192, 205, 242),
        (71, 78, 125, 186, 188, 253)),
    2: ((43, 53, 108, 115, 132, 143), (26, 39, 217, 221, 240, 247),
        (66, 87, 153, 185, 200, 216), (71, 83, 85, 102, 122, 216),
        (12, 32, 52, 101, 142, 224)),
}
# The layer-2 compression window closes after four tokens: one compressed key/value slot
# exists for the five-token prompt, and only positions 3 and 4 may attend to it.
PREFILL_COMPRESSED_ROWS = 1
PREFILL_SELECTED_PER_TOKEN = 6

# Declared tolerances. Do not relax these to make a run pass.
TOLERANCE = 1.0e-3          # maximum absolute difference for logits and hidden state
STAGE_RTOL = 1.0e-5         # relative tolerance for the dumped stage vectors
STAGE_ATOL = 1.0e-6         # absolute tolerance for values near zero

# Stage dumps the oracle must expose. The names are part of the diagnostic contract: dropping
# one would remove a localization boundary between two equations.
REQUIRED_STAGES = {
    "embedding": WIDTH,
    "hc_attn_reduced": WIDTH,
    "q": 64 * 512,
    "k": 512,
    "v": 512,
    "attention_output": WIDTH,
    "router_scores": 256,
    "route_indices": 6,
    "route_weights": 6,
    "shared_output": WIDTH,
    "final_hidden": WIDTH,
}

_SCALES = (0x3C00, 0xB800, 0x3555, 0x4000, 0x0400, 0x0001)
_MINIMUM_SCALES = (0x3800, 0xBC00, 0x3000, 0x3C00, 0x0001, 0x0400)


# ---------------------------------------------------------------------------
# Synthetic quantized-block checks (no artifacts)
# ---------------------------------------------------------------------------

def float_fingerprint(values: np.ndarray) -> int:
    """Canonical FNV-1a 64 over little-endian float32 bits, as in test_main.cpp."""
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    return oracle.fnv1a64_standard(payload)


def _make_input(columns: int) -> np.ndarray:
    values = np.empty(columns, dtype=np.float32)
    for column in range(columns):
        centered = (column % 29) - 14
        values[column] = np.float32(
            np.float32(centered) * np.float32(0.03125)
            + np.float32(column % 3) * np.float32(0.0078125)
        )
    return values


def _write_and_bind(directory: Path, name: str, raw: bytes, ggml_type: int,
                    rows: int, columns: int) -> "oracle.MatrixRef":
    path = directory / f"{name}.bin"
    path.write_bytes(raw)
    tensor = oracle.GGUFTensor(
        name=name,
        dims=(columns, rows),
        ggml_type=ggml_type,
        shard=path.name,
        offset=0,
        encoded_bytes=len(raw),
    )
    return oracle.MatrixRef(name, tensor, rows, columns, ggml_type, path)


def _decode_and_bind(directory: Path, name: str, raw: bytes, ggml_type: int,
                     rows: int, columns: int):
    decoded = oracle.decode_matrix(raw, ggml_type, rows, columns)
    return decoded, _write_and_bind(directory, name, raw, ggml_type, rows, columns)


def _assert_fingerprint(name: str, values: np.ndarray, expected: int) -> None:
    actual = float_fingerprint(values)
    if actual != expected:
        raise AssertionError(
            f"{CONTRACT}: {name} decoded float32 fingerprint is {actual:#018x}, "
            f"expected {expected:#018x}"
        )


def _assert_spots(name: str, values: np.ndarray, spots: dict[int, float]) -> None:
    flat = values.reshape(-1)
    for index, expected in spots.items():
        actual = float(flat[index])
        if actual != expected:
            raise AssertionError(
                f"{CONTRACT}: {name}[{index}] decoded {actual!r}, expected {expected!r}"
            )


def _assert_matvec(name: str, matrix: "oracle.MatrixRef", inputs: np.ndarray,
                   expected: list[float], tolerances: list[float]) -> None:
    actual = matrix.matvec(inputs)
    if actual.size != len(expected):
        raise AssertionError(
            f"{CONTRACT}: {name} matvec produced {actual.size} rows, expected {len(expected)}"
        )
    for index, (value, want, tolerance) in enumerate(zip(actual, expected, tolerances)):
        if abs(float(value) - want) > tolerance:
            raise AssertionError(
                f"{CONTRACT}: {name} matvec row {index}: got {float(value)!r}, "
                f"expected {want!r} (tolerance {tolerance})"
            )


def run_synthetic() -> int:
    """Decode the pinned reference blocks; return the number of checks performed."""
    checks = 0
    with tempfile.TemporaryDirectory(prefix="vastloom-oracle-blocks-") as directory:
        root = Path(directory)

        raw = bytearray(6 * 34)
        for block in range(6):
            struct.pack_into("<H", raw, block * 34, _SCALES[block])
            for index in range(32):
                raw[block * 34 + 2 + index] = (block * 37 + index * 53 + 11) & 0xFF
        decoded, _ = _decode_and_bind(root, "q8_0", bytes(raw), oracle.Q8_0, 6, 32)
        _assert_fingerprint("Q8_0", decoded, 0x25EDD1A40129EE99)
        checks += 1
        _assert_spots("Q8_0", decoded, {
            0: 11.0, 7: 126.0, 8: -77.0, 31: 118.0, 32: -24.0, 63: 50.5, 127: -54.0,
        })
        checks += 7
        _assert_matvec("Q8_0", _write_and_bind(root, "q8_0_mv", bytes(raw), oracle.Q8_0, 3, 64),
                       _make_input(64),
                       [-168.55859375, -2.8284435272216797, 0.0058367089368402958],
                       [0.0, 0.0, 0.0])
        checks += 3

        def k_block_bytes(block_bytes: int, fill) -> bytes:
            payload = bytearray(6 * block_bytes)
            for block in range(6):
                base = block * block_bytes
                struct.pack_into("<H", payload, base, _SCALES[block])
                struct.pack_into("<H", payload, base + 2, _MINIMUM_SCALES[block])
                for index in range(12):
                    payload[base + 4 + index] = (block * 29 + index * 47 + 7) & 0xFF
                fill(payload, base, block)
            return bytes(payload)

        def q4_fill(payload: bytearray, base: int, block: int) -> None:
            for index in range(128):
                payload[base + 16 + index] = (block * 37 + index * 53 + 11) & 0xFF

        def q5_fill(payload: bytearray, base: int, block: int) -> None:
            for index in range(32):
                payload[base + 16 + index] = (block * 41 + index * 31 + 13) & 0xFF
            for index in range(128):
                payload[base + 48 + index] = (block * 37 + index * 53 + 11) & 0xFF

        def q6_fill(payload: bytearray, base: int, block: int) -> None:
            for index in range(128):
                payload[base + index] = (block * 37 + index * 53 + 11) & 0xFF
            for index in range(64):
                payload[base + 128 + index] = (block * 41 + index * 31 + 13) & 0xFF
            for index in range(16):
                payload[base + 192 + index] = (block * 59 + index * 43 + 17) & 0xFF
            struct.pack_into("<H", payload, base + 208, _SCALES[block])

        q4 = k_block_bytes(144, q4_fill)
        decoded, _ = _decode_and_bind(root, "q4_k", q4, oracle.Q4_K, 6, 256)
        _assert_fingerprint("Q4_K", decoded, 0x84E8E84336EA0AD7)
        checks += 1
        _assert_spots("Q4_K", decoded, {
            0: 75.5, 7: 96.5, 8: 19.5, 31: 40.5, 32: -25.0, 63: 353.0,
            127: 12.0, 255: 212.0, 256: 32.0, 511: -125.5,
        })
        checks += 10
        _assert_matvec("Q4_K", _write_and_bind(root, "q4_k_mv", q4, oracle.Q4_K, 3, 512),
                       _make_input(512),
                       [281.6796875, -792.12872314453125, 0.0096547976136207581],
                       [0.0, 0.0, 0.0])
        checks += 3

        q5 = k_block_bytes(176, q5_fill)
        decoded, _ = _decode_and_bind(root, "q5_k", q5, oracle.Q5_K, 6, 256)
        _assert_fingerprint("Q5_K", decoded, 0x3EC4A59589931118)
        checks += 1
        _assert_spots("Q5_K", decoded, {
            0: 187.5, 7: 96.5, 8: 131.5, 31: 40.5, 32: -25.0, 63: 1217.0,
            127: 332.0, 255: 916.0, 256: 32.0, 511: -453.5,
        })
        checks += 10
        _assert_matvec("Q5_K", _write_and_bind(root, "q5_k_mv", q5, oracle.Q5_K, 3, 512),
                       _make_input(512),
                       [668.2421875, 1631.9100341796875, -0.046117834746837616],
                       [0.0, 0.0, 0.0])
        checks += 3

        q6 = k_block_bytes(210, q6_fill)
        decoded, _ = _decode_and_bind(root, "q6_k", q6, oracle.Q6_K, 6, 256)
        _assert_fingerprint("Q6_K", decoded, 0xCD39C3F1219AFBC9)
        checks += 1
        _assert_spots("Q6_K", decoded, {
            0: -85.0, 7: 238.0, 8: -221.0, 31: 360.0, 32: 2781.0, 63: -2420.0,
            127: 1054.0, 255: -530.0, 511: 540.5,
        })
        checks += 9
        _assert_matvec("Q6_K", _write_and_bind(root, "q6_k_mv", q6, oracle.Q6_K, 3, 512),
                       _make_input(512),
                       [-4684.79296875, 7494.287109375, 0.5950310230255127],
                       [0.0, 0.0, 0.0])
        checks += 3

        raw = bytearray(6 * 66)
        for block in range(6):
            struct.pack_into("<H", raw, block * 66, _SCALES[block])
            for index in range(32):
                packed = (block * 733 + index * 977 + 0x1234) & 0xFFFF
                raw[block * 66 + 2 + index * 2] = packed & 0xFF
                raw[block * 66 + 3 + index * 2] = packed >> 8
        decoded, _ = _decode_and_bind(root, "iq2_xxs", bytes(raw), oracle.IQ2_XXS, 6, 256)
        _assert_fingerprint("IQ2_XXS", decoded, 0x9BCA8E03F48DAFE9)
        checks += 1
        _assert_spots("IQ2_XXS", decoded, {
            0: 9.375, 7: 3.0, 8: -16.125, 31: -3.0, 32: 5.0, 63: -5.0,
            127: -9.0, 255: 17.0, 256: 2.5, 511: 8.5,
        })
        checks += 10
        _assert_matvec("IQ2_XXS", _write_and_bind(root, "iq2_xxs_mv", bytes(raw),
                                                  oracle.IQ2_XXS, 3, 512),
                       _make_input(512),
                       [-32.86767578125, -284.89828491210938, -0.011499286629259586],
                       [0.0, 0.0, 1.0e-9])
        checks += 3

        raw = bytearray(6 * 98)
        for block in range(6):
            struct.pack_into("<H", raw, block * 98, _SCALES[block])
            for index in range(64):
                raw[block * 98 + 2 + index] = (block * 37 + index * 53 + 11) & 0xFF
            for group in range(8):
                auxiliary = ((block * 3 + group * 5) & 15) << 28
                for subblock in range(4):
                    sign_index = (block * 19 + group * 13 + subblock * 23 + 7) & 127
                    auxiliary |= sign_index << (7 * subblock)
                for byte in range(4):
                    raw[block * 98 + 66 + group * 4 + byte] = (auxiliary >> (8 * byte)) & 0xFF
        decoded, _ = _decode_and_bind(root, "iq3_xxs", bytes(raw), oracle.IQ3_XXS, 6, 256)
        _assert_fingerprint("IQ3_XXS", decoded, 0xF36989DCC58B1664)
        checks += 1
        _assert_spots("IQ3_XXS", decoded, {
            0: -11.0, 7: -3.0, 8: 5.0, 31: -5.0, 32: 170.5, 255: 7.0,
            256: -54.25, 511: -19.5,
        })
        checks += 8
        _assert_matvec("IQ3_XXS", _write_and_bind(root, "iq3_xxs_mv", bytes(raw),
                                                  oracle.IQ3_XXS, 3, 512),
                       _make_input(512),
                       [-347.998046875, 2887.319580078125, 0.064146831631660461],
                       [0.0, 1.0e-4, 1.0e-8])
        checks += 3

        dense = np.array([1.0, -2.0, 0.5, 4.0, -1.0, 3.0, -0.25, 2.0], dtype=np.float32)
        bf16_bits = (0x3F80, 0xC000, 0x3F00, 0x4080, 0xBF80, 0x4040, 0xBE80, 0x4000)
        bf16_raw = b"".join(struct.pack("<H", value) for value in bf16_bits)
        f32_raw = dense.astype("<f4").tobytes()
        bf16_values, bf16_matrix = _decode_and_bind(root, "bf16", bf16_raw, oracle.BF16, 2, 4)
        f32_values, f32_matrix = _decode_and_bind(root, "f32", f32_raw, oracle.F32, 2, 4)
        if not np.array_equal(bf16_values.reshape(-1), dense):
            raise AssertionError(f"{CONTRACT}: BF16 decode does not reproduce the pinned values")
        if not np.array_equal(f32_values.reshape(-1), dense):
            raise AssertionError(f"{CONTRACT}: F32 decode does not reproduce the pinned values")
        dense_input = np.array([0.25, -0.5, 2.0, 1.0], dtype=np.float32)
        _assert_matvec("BF16", bf16_matrix, dense_input, [6.25, -0.25], [0.0, 0.0])
        _assert_matvec("F32", f32_matrix, dense_input, [6.25, -0.25], [0.0, 0.0])
        checks += 6

        try:
            oracle.decode_matrix(q4[:-1], oracle.Q4_K, 6, 256)
        except oracle.OracleError:
            checks += 1
        else:
            raise AssertionError(f"{CONTRACT}: Q4_K decode accepted a truncated block")

        # Layer-2 compressor: the ratio-4 path scores the previous and current windows
        # together, adds the APE and softmaxes in double before one float32 rounding.
        ratio, output_width, token_count = 4, 2, 12
        coefficient = oracle.compressor_coefficient(ratio)
        projected_width = coefficient * output_width
        projected = np.empty((token_count, projected_width), dtype=np.float32)
        scores = np.empty_like(projected)
        ape = np.empty((ratio, projected_width), dtype=np.float32)
        for token in range(token_count):
            for column in range(projected_width):
                projected[token, column] = np.float32(0.375 * token - 0.25 * column + 0.5)
                scores[token, column] = np.float32(
                    0.125 * ((token * 7 + column * 3) % 11) - 0.5)
        for position in range(ratio):
            for column in range(projected_width):
                ape[position, column] = np.float32(0.0625 * (position + 1) * (column - 1))
        windows = oracle.compressor_windows(projected, scores, ape, ratio, output_width)
        expected_windows = np.empty((token_count // ratio, output_width), dtype=np.float32)
        for window in range(token_count // ratio):
            for dimension in range(output_width):
                candidates = []
                for candidate in range(coefficient * ratio):
                    previous = coefficient == 2 and candidate < ratio
                    if previous and window == 0:
                        continue
                    position = candidate % ratio
                    source = (window - 1 if previous else window) * ratio + position
                    column = (0 if previous else output_width if coefficient == 2 else 0) + dimension
                    score = float(scores[source, column]) + float(ape[position, column])
                    candidates.append((score, float(projected[source, column])))
                maximum = max(score for score, _value in candidates)
                denominator = sum(math.exp(score - maximum) for score, _value in candidates)
                accumulated = sum(math.exp(score - maximum) * value
                                  for score, value in candidates)
                expected_windows[window, dimension] = np.float32(accumulated / denominator)
        if windows.shape != expected_windows.shape or not np.array_equal(windows,
                                                                        expected_windows):
            raise AssertionError(
                f"{CONTRACT}: compressor windows do not match the independent reference: "
                f"{windows.tolist()} vs {expected_windows.tolist()}")
        checks += 1 + windows.size
        if oracle.compressor_windows(projected[:1], scores[:1], ape, ratio,
                                     output_width).shape != (0, output_width):
            raise AssertionError(f"{CONTRACT}: the compressor emitted a partial window")
        checks += 1
        compressor_config = types.SimpleNamespace(
            compression_rope_theta=160_000.0, original_context=65_536,
            rope_factor=16.0, beta_fast=32.0, beta_slow=1.0, norm_epsilon=1.0e-6)
        compressed = oracle.compressor_prefill(
            projected, scores, ape, np.array([1.0, -0.5], dtype=np.float32), ratio,
            output_width, 2, compressor_config)
        pinned_compressed = [1.23101806640625, -0.3480612635612488, -1.0529793500900269,
                             -0.5277884006500244, 0.32005029916763306, 1.1022359132766724]
        if [float(value) for value in compressed.reshape(-1)] != pinned_compressed:
            raise AssertionError(
                f"{CONTRACT}: compressed rows changed: {compressed.reshape(-1).tolist()}")
        checks += 1 + compressed.size

        # Layer-2 indexer: scores every compressed slot with a per-head positive projection.
        # The API is per prompt position; the last position of a twelve-token prompt may see
        # all three compressed slots, while the first three positions may see none.
        indexer_config = types.SimpleNamespace(index_top_k=2, index_head_count=2,
                                               index_head_width=2)
        one_query = np.array([1.0, 0.5, -1.0, 2.0], dtype=np.float32)
        one_weights = np.array([0.5, 1.5], dtype=np.float32)
        index_query = np.tile(one_query, token_count).reshape(-1)
        index_weights = np.tile(one_weights, token_count).reshape(-1)
        index_compressed = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        selected, _scores = oracle.indexer_selection(
            index_query, index_compressed, index_weights, indexer_config, ratio, token_count)
        if selected[-1] != [14, 13] or any(selected[position] for position in range(3)):
            raise AssertionError(
                f"{CONTRACT}: indexer selection is {selected}, expected [] at closed "
                f"positions and [14, 13] at the last one"
            )
        checks += 1
        empty, _empty_scores = oracle.indexer_selection(
            one_query, index_compressed, one_weights, indexer_config, ratio, 1)
        if empty != [[]]:
            raise AssertionError(f"{CONTRACT}: the indexer selected a window for one token")
        checks += 1

    oracle.verify_codebooks()
    checks += 2
    print(f"{CONTRACT}: synthetic block decoders PASS ({checks} pinned checks)")
    return checks


# ---------------------------------------------------------------------------
# Probe/oracle integration
# ---------------------------------------------------------------------------

def _read_f32(path: Path, expected: int | None = None) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise AssertionError(f"{CONTRACT}: {path} is not a whole float32 dump ({len(raw)} bytes)")
    values = np.frombuffer(raw, dtype="<f4").copy()
    if expected is not None and values.size != expected:
        raise AssertionError(
            f"{CONTRACT}: {path} has {values.size} values, expected {expected}"
        )
    return values


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        raise AssertionError(
            f"{CONTRACT}: vector lengths differ ({left.size} vs {right.size})"
        )
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max())


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )


def _help_text(vastloom: Path) -> str:
    process = _run([str(vastloom), "--help"], 120)
    return (process.stdout or "") + (process.stderr or "")


def _source_has_flag(flag: str) -> bool:
    source = REPO_ROOT / "src" / "cli" / "main.cpp"
    if not source.is_file():
        return False
    return flag in source.read_text(encoding="utf-8", errors="replace")


def _probe_flags(vastloom: Path) -> tuple[bool, bool, bool, str]:
    """Return (logits, hidden, trace, blocker)."""
    help_text = _help_text(vastloom)
    logits = "--dump-logits" in help_text
    hidden = "--dump-hidden" in help_text
    trace = "--dump-trace" in help_text
    if logits:
        return logits, hidden, trace, ""
    if _source_has_flag("--dump-logits"):
        return False, hidden, trace, (
            "the CLI source advertises --dump-logits but the built probe does not list it; "
            "rebuild the release binary"
        )
    return False, hidden, trace, (
        "the probe has no --dump-logits (checked --help and src/cli/main.cpp); "
        "the independent oracle cannot be compared against it"
    )


def _probe(args: argparse.Namespace, logits_path: Path, hidden_path: Path,
           trace_path: Path, token: int | None = None,
           trunk: Path | None = None,
           experts: Path | None = None) -> tuple[dict, Path | None, Path | None]:
    token = args.token if token is None else token
    trunk = args.artifact if trunk is None else trunk
    experts = args.experts if experts is None else experts
    command = [
        str(args.vastloom),
        "deepseek-layer-probe",
        str(args.model),
        str(trunk),
        str(experts),
        str(args.globals),
        "--token",
        str(token),
        "--json",
        "--dump-logits",
        str(logits_path),
    ]
    if args.dump_hidden:
        command.extend(("--dump-hidden", str(hidden_path)))
    if args.dump_trace:
        command.extend(("--dump-trace", str(trace_path)))
    process = _run(command, args.timeout)
    pressure = host_pressure_reason(process)
    if pressure is not None:
        raise SystemExit(skipped(CONTRACT, pressure))
    if process.returncode != 0:
        raise AssertionError(
            f"{CONTRACT}: the native probe failed with exit {process.returncode}\n"
            f"stderr={process.stderr}\nstdout={process.stdout}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{CONTRACT}: the native probe did not print JSON: {error}\n{process.stdout}"
        ) from error
    if payload.get("executed_layers") != 1 or payload.get("model_layers") != MODEL_LAYERS:
        raise AssertionError(f"{CONTRACT}: the probe did not execute exactly layer 0 of 43")
    if payload.get("logits") != LOGITS or payload.get("token") != token:
        raise AssertionError(f"{CONTRACT}: the probe reported an unexpected token or shape")
    if not logits_path.is_file() or logits_path.stat().st_size != LOGITS * 4:
        raise AssertionError(f"{CONTRACT}: the probe did not write {LOGITS} logits")
    hidden = hidden_path if args.dump_hidden else None
    if hidden is not None and (not hidden.is_file() or hidden.stat().st_size != WIDTH * 4):
        raise AssertionError(f"{CONTRACT}: the probe did not write the final hidden state")
    trace = trace_path if args.dump_trace else None
    if trace is not None and not trace.is_file():
        raise AssertionError(f"{CONTRACT}: the probe did not write the trace file")
    return payload, hidden, trace


def _oracle(args: argparse.Namespace, root: Path) -> tuple[dict, Path]:
    output = root / "oracle"
    command = [
        str(args.python),
        str(args.oracle),
        "--model",
        str(args.model),
        "--trunk",
        str(args.artifact),
        "--experts",
        str(args.experts),
        "--globals",
        str(args.globals),
        "--token",
        str(args.token),
        "--output-dir",
        str(output),
        "--stages",
    ]
    process = _run(command, args.timeout)
    if process.returncode != 0:
        raise AssertionError(
            f"{CONTRACT}: the independent oracle failed with exit {process.returncode}\n"
            f"stderr={process.stderr}\nstdout={process.stdout}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{CONTRACT}: the oracle did not print JSON: {error}\n{process.stdout}"
        ) from error
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise AssertionError(f"{CONTRACT}: the oracle did not write summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("ok") is not True or summary.get("ok") is not True:
        raise AssertionError(f"{CONTRACT}: the oracle did not report success")
    return summary, output


def _check_stages(summary: dict, output: Path, route: list[int]) -> int:
    stage_files = summary.get("stage_files")
    if not isinstance(stage_files, dict) or not stage_files:
        raise AssertionError(f"{CONTRACT}: the oracle did not report stage files")
    required = dict(REQUIRED_STAGES)
    for expert in route:
        required[f"expert_{expert}_output"] = WIDTH
    for name, expected in required.items():
        entry = stage_files.get(name)
        if not isinstance(entry, dict):
            raise AssertionError(f"{CONTRACT}: stage {name} is missing from the summary")
        path = output / "stages" / f"{name}.f32"
        if not path.is_file():
            raise AssertionError(f"{CONTRACT}: stage file {path} is missing")
        values = _read_f32(path, expected)
        if entry.get("count") != expected:
            raise AssertionError(f"{CONTRACT}: stage {name} reports the wrong count")
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{CONTRACT}: stage {name} contains a non-finite value")
        if entry.get("sha256") != hashlib.sha256(values.tobytes()).hexdigest():
            raise AssertionError(f"{CONTRACT}: stage {name} checksum does not match its file")
    summary_stages = summary.get("stages")
    if not isinstance(summary_stages, dict) or set(summary_stages) != set(stage_files):
        raise AssertionError(f"{CONTRACT}: stage checksums and stage files disagree")
    for name, entry in stage_files.items():
        checksums = summary_stages[name]
        if (checksums.get("count") != entry.get("count")
                or checksums.get("sha256") != entry.get("sha256")
                or checksums.get("fnv1a64") != entry.get("fnv1a64")):
            raise AssertionError(f"{CONTRACT}: stage checksums disagree for {name}")
    return len(required)


def _compare_trace_checksums(trace: dict, summary: dict, oracle_logits: np.ndarray,
                             args: argparse.Namespace) -> int:
    """Compare the probe's raw FNV hashes; return the number of matches."""
    expected = {
        "logits": (trace.get("logits_fnv1a64"), summary["logits"]["fnv1a64"], oracle_logits),
        "final_hidden": (trace.get("final_hidden_fnv1a64"),
                         summary["stages"]["final_hidden"]["fnv1a64"],
                         _read_f32(Path(summary["stage_files"]["final_hidden"]["file"]), WIDTH)),
    }
    mismatches = 0
    matches = 0
    for name, (probe_hash, oracle_hash, values) in expected.items():
        if probe_hash is None:
            print(f"{CONTRACT}: BLOCKED - the probe trace has no {name} checksum",
                  file=sys.stderr)
            continue
        if probe_hash == oracle_hash:
            matches += 1
            print(f"{CONTRACT}: {name} FNV checksum matches the probe")
            continue
        mismatches += 1
        print(
            f"{CONTRACT}: {name} raw FNV differs (probe {probe_hash}, oracle {oracle_hash}); "
            f"a bitwise hash has no tolerance, and the probe's MSVC scalar path differs by "
            f"one float32 ulp on some values. Value comparisons above are the gate."
        )
    if mismatches and args.require_checksum_equality:
        raise AssertionError(
            f"{CONTRACT}: {mismatches} raw checksum(s) differ and "
            f"--require-checksum-equality was requested"
        )
    stage_trace = trace.get("stages")
    if isinstance(stage_trace, dict):
        compared = 0
        for name, entry in stage_trace.items():
            if name not in summary["stages"]:
                raise AssertionError(f"{CONTRACT}: trace stage {name} is unknown")
            expected_hash = entry.get("fnv1a64") if isinstance(entry, dict) else None
            if expected_hash == summary["stages"][name]["fnv1a64"]:
                compared += 1
            else:
                print(f"{CONTRACT}: trace stage {name} checksum differs")
        print(f"{CONTRACT}: trace stage checksums compared ({compared} matched)")
    else:
        print(
            f"{CONTRACT}: BLOCKED - the probe trace exposes only logits/final_hidden "
            f"checksums; the other stage checksums are recorded by the oracle but cannot be "
            f"compared yet",
            file=sys.stderr,
        )
    return matches


def run_integration(args: argparse.Namespace) -> int:
    for name, value in (
        ("vastloom", args.vastloom), ("model", args.model), ("artifact", args.artifact),
        ("experts", args.experts), ("globals", args.globals), ("oracle", args.oracle),
        ("python", args.python),
    ):
        if not value.exists():
            raise SystemExit(f"{CONTRACT} requires an existing --{name}: {value}")
    if args.token < 0:
        raise SystemExit(f"{CONTRACT} requires a non-negative token id")
    if sys.byteorder != "little":
        raise SystemExit(f"{CONTRACT} requires a little-endian host for raw float32 dumps")

    available, has_hidden, has_trace, blocker = _probe_flags(args.vastloom)
    if not available:
        print(f"{CONTRACT}: BLOCKED - {blocker}", file=sys.stderr)
        return 2
    args.dump_hidden = has_hidden
    args.dump_trace = has_trace

    build_root = REPO_ROOT / "build"
    if not build_root.is_dir():
        raise SystemExit(f"{CONTRACT} requires the build directory for staged dumps: {build_root}")
    root = Path(tempfile.mkdtemp(prefix="vastloom-deepseek-layer0-oracle-", dir=build_root))
    try:
        probe_logits_path = root / "probe-logits.f32"
        probe_hidden_path = root / "probe-hidden.f32"
        probe_trace_path = root / "probe-trace.json"
        probe, probe_hidden, probe_trace = _probe(
            args, probe_logits_path, probe_hidden_path, probe_trace_path
        )
        summary, output = _oracle(args, root)

        oracle_logits = _read_f32(output / "logits.f32", LOGITS)
        reference_logits = _read_f32(probe_logits_path, LOGITS)
        logits_delta = _max_abs_diff(oracle_logits, reference_logits)
        if logits_delta > TOLERANCE:
            raise AssertionError(
                f"{CONTRACT}: logits differ by {logits_delta:.9g} > {TOLERANCE}"
            )

        if summary.get("format") != "vastloom.deepseek_v4.layer0_oracle.v1":
            raise AssertionError(f"{CONTRACT}: unexpected oracle summary format")
        if summary.get("layer") != 0 or summary.get("model_layers") != MODEL_LAYERS:
            raise AssertionError(f"{CONTRACT}: the oracle did not describe layer 0 of 43")
        if summary.get("token") != args.token:
            raise AssertionError(f"{CONTRACT}: the oracle summarized the wrong token")
        logits_summary = summary.get("logits", {})
        if logits_summary.get("count") != LOGITS:
            raise AssertionError(f"{CONTRACT}: the oracle reported the wrong logits count")
        if logits_summary.get("argmax") != probe.get("argmax_token"):
            raise AssertionError(f"{CONTRACT}: the oracle argmax differs from the probe")
        if logits_summary.get("sha256") != hashlib.sha256(oracle_logits.tobytes()).hexdigest():
            raise AssertionError(f"{CONTRACT}: the oracle logits checksum is wrong")
        if int(logits_summary.get("argmax", -1)) != int(np.argmax(oracle_logits)):
            raise AssertionError(f"{CONTRACT}: the oracle logits argmax is inconsistent")

        oracle_route = summary.get("route", {}).get("indices")
        if oracle_route != probe.get("experts_executed"):
            raise AssertionError(
                f"{CONTRACT}: route mismatch (oracle {oracle_route}, "
                f"probe {probe.get('experts_executed')})"
            )
        if len(summary.get("route", {}).get("weights", [])) != 6:
            raise AssertionError(f"{CONTRACT}: the oracle route weights are incomplete")

        artifacts = summary.get("artifacts", {})
        for key, expected in (
            ("trunk_ranges", 21), ("expert_ranges", 18), ("global_ranges", 6),
            ("trunk_bytes", 138_867_160), ("expert_bytes", 45_219_840),
            ("global_bytes", 596_000_788),
        ):
            if artifacts.get(key) != expected:
                raise AssertionError(
                    f"{CONTRACT}: artifact summary {key} is {artifacts.get(key)!r}, "
                    f"expected {expected!r}"
                )

        if args.token == 0:
            if probe.get("argmax_token") != ARGMAX_TOKEN:
                raise AssertionError(f"{CONTRACT}: the pinned probe argmax changed")
            if probe.get("experts_executed") != ROUTE_EXPERTS:
                raise AssertionError(f"{CONTRACT}: the pinned probe route changed")

        stage_count = _check_stages(summary, output, [int(value) for value in oracle_route])
        print(f"{CONTRACT}: logits PASS ({LOGITS} values, max abs diff {logits_delta:.9g})")

        if probe_hidden is not None:
            oracle_hidden = _read_f32(output / "stages" / "final_hidden.f32", WIDTH)
            reference_hidden = _read_f32(probe_hidden, WIDTH)
            hidden_delta = _max_abs_diff(oracle_hidden, reference_hidden)
            if hidden_delta > TOLERANCE:
                raise AssertionError(
                    f"{CONTRACT}: final hidden differs by {hidden_delta:.9g} > {TOLERANCE}"
                )
            if not np.allclose(reference_hidden, oracle_hidden, rtol=STAGE_RTOL,
                               atol=STAGE_ATOL):
                raise AssertionError(
                    f"{CONTRACT}: final hidden differs beyond rtol={STAGE_RTOL}, "
                    f"atol={STAGE_ATOL} (max abs diff {hidden_delta:.9g})"
                )
            print(
                f"{CONTRACT}: final hidden PASS ({WIDTH} values, "
                f"max abs diff {hidden_delta:.9g}, rtol {STAGE_RTOL}, atol {STAGE_ATOL})"
            )
        else:
            hidden_delta = float("nan")
            print(f"{CONTRACT}: BLOCKED - no --dump-hidden; final hidden not compared",
                  file=sys.stderr)

        checksum_matches = 0
        if probe_trace is not None:
            trace = json.loads(probe_trace.read_text(encoding="utf-8"))
            checksum_matches = _compare_trace_checksums(trace, summary, oracle_logits, args)
        else:
            print(f"{CONTRACT}: BLOCKED - no --dump-trace; checksums not compared",
                  file=sys.stderr)

        print(
            f"{CONTRACT}: PASS "
            f"(logits max|delta|={logits_delta:.9g}, "
            f"hidden max|delta|={hidden_delta:.9g}, "
            f"argmax={logits_summary['argmax']}, route={oracle_route}, "
            f"stages={stage_count}, checksum_matches={checksum_matches})"
        )
        if args.keep:
            print(f"{CONTRACT}: kept {root}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


def _parse_layer_map(values: list[str] | None, option: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"{CHAIN_CONTRACT}: {option} needs LAYER=PATH, got {raw!r}")
        layer_text, path_text = raw.split("=", 1)
        try:
            layer = int(layer_text)
        except ValueError:
            raise SystemExit(f"{CHAIN_CONTRACT}: {option} layer is not an integer: {raw!r}")
        if layer < 0 or not path_text:
            raise SystemExit(f"{CHAIN_CONTRACT}: {option} value is malformed: {raw!r}")
        if layer in result:
            raise SystemExit(f"{CHAIN_CONTRACT}: {option} repeats layer {layer}")
        result[layer] = Path(path_text)
    return result


def _parse_expert_maps(values: list[str] | None) -> dict[int, dict[int, Path]]:
    """Parse ``--chain-experts TOKEN:LAYER=PATH`` values.

    Selected-expert artifacts are imported for one token's route, so the chain test needs a
    different set per token; a single map would compare token 1 against token 0's experts.
    """
    result: dict[int, dict[int, Path]] = {}
    for raw in values or []:
        try:
            token_text, rest = raw.split(":", 1)
            layer_text, path_text = rest.split("=", 1)
            token = int(token_text)
            layer = int(layer_text)
        except ValueError:
            raise SystemExit(
                f"{CHAIN_CONTRACT}: --chain-experts needs TOKEN:LAYER=PATH, got {raw!r}")
        if token < 0 or layer < 0 or not path_text:
            raise SystemExit(f"{CHAIN_CONTRACT}: --chain-experts value is malformed: {raw!r}")
        token_map = result.setdefault(token, {})
        if layer in token_map:
            raise SystemExit(
                f"{CHAIN_CONTRACT}: --chain-experts repeats token {token} layer {layer}")
        token_map[layer] = Path(path_text)
    return result


def _parse_csv_ints(text: str, option: str) -> list[int]:
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            raise SystemExit(f"{CHAIN_CONTRACT}: {option} contains {item!r}, not an integer")
    if not result:
        raise SystemExit(f"{CHAIN_CONTRACT}: {option} is empty")
    return result


def _chain_probe(args: argparse.Namespace, token: int, layers: int,
                 trunks: dict[int, Path], experts: dict[int, Path],
                 logits_path: Path, hidden_path: Path,
                 trace_path: Path) -> tuple[dict, dict]:
    command = [
        str(args.vastloom),
        "deepseek-chain-probe",
        str(args.model),
        str(args.globals),
        "--layers",
        str(layers),
        "--token",
        str(token),
        "--json",
    ]
    for layer in range(layers):
        command.extend(("--trunk", f"{layer}={trunks[layer]}"))
    for layer in range(layers):
        command.extend(("--experts", f"{layer}={experts[layer]}"))
    command.extend(("--dump-logits", str(logits_path),
                    "--dump-hidden", str(hidden_path),
                    "--dump-trace", str(trace_path)))
    process = _run(command, args.timeout)
    pressure = host_pressure_reason(process)
    if pressure is not None:
        raise SystemExit(skipped(CHAIN_CONTRACT, pressure))
    if process.returncode != 0:
        raise AssertionError(
            f"{CHAIN_CONTRACT}: the native chain probe failed with exit {process.returncode}\n"
            f"stderr={process.stderr}\nstdout={process.stdout}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{CHAIN_CONTRACT}: the chain probe did not print JSON: {error}\n{process.stdout}"
        ) from error
    if payload.get("ok") is not True or payload.get("complete_model") is not False:
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain probe did not report a partial chain")
    if (payload.get("executed_layers") != layers
            or payload.get("model_layers") != MODEL_LAYERS
            or payload.get("token") != token
            or payload.get("logits") != LOGITS):
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain probe reported an unexpected shape")
    entries = payload.get("layers")
    if not isinstance(entries, list) or len(entries) != layers:
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain probe reported the wrong layer count")
    for index, entry in enumerate(entries):
        if entry.get("layer") != index or not isinstance(entry.get("route"), list):
            raise AssertionError(f"{CHAIN_CONTRACT}: chain probe layer {index} is malformed")
    if not logits_path.is_file() or logits_path.stat().st_size != LOGITS * 4:
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain probe did not write the logits dump")
    if not hidden_path.is_file() or hidden_path.stat().st_size != WIDTH * 4:
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain probe did not write the hidden dump")
    if not trace_path.is_file():
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain probe did not write the trace")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    # The pre-token CLI wrote ``token``; the multi-token trace writes ``tokens``. Both are
    # accepted so one binary lineage can run either section.
    trace_tokens = trace.get("tokens")
    if trace_tokens is None and trace.get("token") is not None:
        trace_tokens = [trace.get("token")]
    if (trace_tokens != [token] or trace.get("executed_layers") != layers
            or trace.get("model_layers") != MODEL_LAYERS or trace.get("logits") != LOGITS):
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain trace disagrees with its summary")
    trace_layers = trace.get("layers")
    if not isinstance(trace_layers, list) or len(trace_layers) != layers:
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain trace has the wrong layer count")
    for index, entry in enumerate(trace_layers):
        # The JSON field is the resolved unique set in ascending order; the trace keeps
        # the table order the layer actually executed. Both must describe the same set.
        if entry.get("layer") != index:
            raise AssertionError(f"{CHAIN_CONTRACT}: the chain trace layer order is malformed")
        if sorted(entry.get("route", [])) != entries[index].get("route"):
            raise AssertionError(f"{CHAIN_CONTRACT}: the chain trace route set differs from the JSON")
    return payload, trace


def _chain_oracle(args: argparse.Namespace, token: int, layers: int, depths: list[int],
                  trunks: dict[int, Path], experts: dict[int, Path],
                  root: Path) -> tuple[dict, Path]:
    output = root / f"oracle-token{token}"
    command = [
        str(args.python),
        str(args.oracle),
        "--model",
        str(args.model),
    ]
    for layer in range(layers):
        command.extend(("--trunk", f"{layer}={trunks[layer]}"))
    for layer in range(layers):
        command.extend(("--experts", f"{layer}={experts[layer]}"))
    command.extend((
        "--globals", str(args.globals),
        "--layers", str(layers),
        "--depths", ",".join(str(depth) for depth in depths),
        "--token", str(token),
        "--output-dir", str(output),
        "--stages",
    ))
    process = _run(command, args.timeout)
    if process.returncode != 0:
        raise AssertionError(
            f"{CHAIN_CONTRACT}: the independent chain oracle failed with exit "
            f"{process.returncode}\nstderr={process.stderr}\nstdout={process.stdout}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{CHAIN_CONTRACT}: the chain oracle did not print JSON: {error}\n{process.stdout}"
        ) from error
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain oracle did not write summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (payload.get("ok") is not True or summary.get("ok") is not True
            or summary.get("format") != "vastloom.deepseek_v4.chain_oracle.v1"):
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain oracle did not report success")
    if (summary.get("executed_layers") != layers
            or summary.get("model_layers") != MODEL_LAYERS
            or summary.get("token") != token):
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain oracle described the wrong run")
    if sorted(int(key) for key in summary.get("depths", {})) != sorted(depths):
        raise AssertionError(f"{CHAIN_CONTRACT}: the chain oracle emitted the wrong depths")
    return summary, output


def _check_chain_stages(summary: dict, output: Path, layers: int, depths: list[int]) -> int:
    stage_files = summary.get("stage_files")
    if not isinstance(stage_files, dict) or not stage_files:
        raise AssertionError(f"{CHAIN_CONTRACT}: the oracle did not report stage files")
    required: dict[str, int] = {}
    for layer in range(layers):
        required[f"layer{layer}.hidden"] = 4 * WIDTH
        required[f"layer{layer}.attn_norm"] = WIDTH
        required[f"layer{layer}.router_scores"] = 256
        required[f"layer{layer}.route_indices"] = 6
        required[f"layer{layer}.attention_output"] = WIDTH
        required[f"layer{layer}.shared_output"] = WIDTH
    for depth in depths:
        required[f"depth{depth}.final_hidden"] = WIDTH
        required[f"depth{depth}.final_norm"] = WIDTH
        required[f"depth{depth}.logits"] = LOGITS
    if layers >= 3:
        # Layer 2 is the compressed layer of the chain: its compressor and indexer
        # projections are computed even though one prefill token closes no window.
        required["layer2.attention_compressor_projected"] = 2 * 512
        required["layer2.attention_compressor_scores"] = 2 * 512
        required["layer2.attention_compressed"] = 0
        required["layer2.indexer_query"] = 64 * 128
        required["layer2.indexer_weights"] = 64
        required["layer2.indexer_compressed"] = 0
    for name, expected in required.items():
        entry = stage_files.get(name)
        if not isinstance(entry, dict):
            raise AssertionError(f"{CHAIN_CONTRACT}: stage {name} is missing from the summary")
        path = output / "stages" / f"{name}.f32"
        if not path.is_file():
            raise AssertionError(f"{CHAIN_CONTRACT}: stage file {path} is missing")
        values = _read_f32(path, expected)
        if entry.get("count") != expected:
            raise AssertionError(f"{CHAIN_CONTRACT}: stage {name} reports the wrong count")
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{CHAIN_CONTRACT}: stage {name} contains a non-finite value")
        if entry.get("sha256") != hashlib.sha256(values.tobytes()).hexdigest():
            raise AssertionError(f"{CHAIN_CONTRACT}: stage {name} checksum does not match its file")
    summary_stages = summary.get("stages")
    if not isinstance(summary_stages, dict) or set(summary_stages) != set(stage_files):
        raise AssertionError(f"{CHAIN_CONTRACT}: stage checksums and stage files disagree")
    for name, entry in stage_files.items():
        checksums = summary_stages[name]
        if (checksums.get("count") != entry.get("count")
                or checksums.get("sha256") != entry.get("sha256")
                or checksums.get("fnv1a64") != entry.get("fnv1a64")):
            raise AssertionError(f"{CHAIN_CONTRACT}: stage checksums disagree for {name}")
    return len(required)


def _first_nonfinite_stage(summary: dict, output: Path) -> str | None:
    for name in sorted(summary.get("stage_files", {})):
        path = output / "stages" / f"{name}.f32"
        if not path.is_file():
            continue
        values = _read_f32(path)
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            return f"{name}[{int(bad[0])}]={float(values[bad[0]])}"
    return None


def _compare_depth(contract: str, quantity: str, oracle_values: np.ndarray,
                   probe_values: np.ndarray, token: int, depth: int,
                   *, hidden: bool = False) -> float:
    delta = _max_abs_diff(oracle_values, probe_values)
    if delta > TOLERANCE:
        index = int(np.argmax(np.abs(oracle_values.astype(np.float64)
                                     - probe_values.astype(np.float64))))
        raise AssertionError(
            f"{contract}: first divergent stage is depth {depth} (layer {depth - 1}) "
            f"{quantity}: token {token}, max abs diff {delta:.9g} > {TOLERANCE} at index "
            f"{index} (probe {float(probe_values[index])!r}, "
            f"oracle {float(oracle_values[index])!r})"
        )
    if hidden and not np.allclose(probe_values, oracle_values, rtol=STAGE_RTOL,
                                  atol=STAGE_ATOL):
        relative = float(np.max(np.abs(probe_values.astype(np.float64)
                                       - oracle_values.astype(np.float64))
                                / np.maximum(np.abs(probe_values.astype(np.float64)),
                                             1.0e-30)))
        raise AssertionError(
            f"{contract}: final hidden at depth {depth} exceeds rtol={STAGE_RTOL}, "
            f"atol={STAGE_ATOL} (max abs diff {delta:.9g}, max relative {relative:.9g})"
        )
    return delta


def run_chain_integration(args: argparse.Namespace) -> int:
    for name, value in (
        ("vastloom", args.vastloom), ("model", args.model), ("globals", args.globals),
        ("oracle", args.oracle), ("python", args.python),
    ):
        if not value.exists():
            raise SystemExit(f"{CHAIN_CONTRACT} requires an existing --{name}: {value}")
    help_text = _help_text(args.vastloom)
    if "deepseek-chain-probe" not in help_text:
        print(f"{CHAIN_CONTRACT}: BLOCKED - the built CLI has no deepseek-chain-probe",
              file=sys.stderr)
        return 2
    if "--dump-logits" not in help_text:
        print(f"{CHAIN_CONTRACT}: BLOCKED - the built CLI has no --dump-logits",
              file=sys.stderr)
        return 2
    if "--dump-hidden" not in help_text:
        print(f"{CHAIN_CONTRACT}: BLOCKED - the built CLI has no --dump-hidden, so the "
              f"final hidden state cannot be compared", file=sys.stderr)
        return 2
    args.dump_hidden = True
    args.dump_trace = False
    trunks = _parse_layer_map(args.chain_trunk, "--chain-trunk")
    expert_maps = _parse_expert_maps(args.chain_experts)
    for path in trunks.values():
        if not path.exists():
            raise SystemExit(f"{CHAIN_CONTRACT}: --chain-trunk path does not exist: {path}")
    for token_map in expert_maps.values():
        for path in token_map.values():
            if not path.exists():
                raise SystemExit(f"{CHAIN_CONTRACT}: --chain-experts path does not exist: {path}")
    tokens = _parse_csv_ints(args.chain_tokens, "--chain-tokens")
    depths = sorted(set(_parse_csv_ints(args.chain_depths, "--chain-depths")))
    layers = depths[-1]
    if layers < 1 or layers > 3:
        raise SystemExit(f"{CHAIN_CONTRACT}: depths outside 1..3 are not admitted")
    for token in tokens:
        for layer in range(layers):
            if layer not in trunks:
                raise SystemExit(
                    f"{CHAIN_CONTRACT}: the chain needs one --chain-trunk for layer {layer}")
            if token not in expert_maps or layer not in expert_maps[token]:
                raise SystemExit(
                    f"{CHAIN_CONTRACT}: the chain needs one --chain-experts {token}:{layer}=PATH "
                    f"for token {token} layer {layer}")
    build_root = REPO_ROOT / "build"
    if not build_root.is_dir():
        raise SystemExit(f"{CHAIN_CONTRACT} requires the build directory: {build_root}")
    root = Path(tempfile.mkdtemp(prefix="vastloom-deepseek-chain-oracle-", dir=build_root))
    try:
        deltas: dict[tuple[int, int], tuple[float, float]] = {}
        comparisons = 0
        for token in tokens:
            if token < 0:
                raise SystemExit(f"{CHAIN_CONTRACT}: token ids must be non-negative")
            token_experts = expert_maps[token]
            summary, output = _chain_oracle(args, token, layers, depths, trunks,
                                            token_experts, root)
            stage_count = _check_chain_stages(summary, output, layers, depths)
            oracle_layers = summary.get("layers")
            if not isinstance(oracle_layers, list) or len(oracle_layers) != layers:
                raise AssertionError(f"{CHAIN_CONTRACT}: the oracle layer records are missing")
            for depth in depths:
                tag = f"token{token}-depth{depth}"
                logits_path = root / f"probe-{tag}-logits.f32"
                hidden_path = root / f"probe-{tag}-hidden.f32"
                trace_path = root / f"probe-{tag}-trace.json"
                if depth == 1:
                    payload, _probe_hidden, _trace = _probe(
                        args, logits_path, hidden_path, trace_path,
                        token=token, trunk=trunks[0], experts=token_experts[0])
                    ordered_routes = [payload.get("experts_executed")]
                    resolved_routes = [sorted(payload.get("experts_executed", []))]
                else:
                    payload, trace = _chain_probe(
                        args, token, depth, trunks, token_experts, logits_path, hidden_path,
                        trace_path)
                    resolved_routes = [entry.get("route") for entry in payload["layers"]]
                    # The trace keeps the expert access order the layer executed, which is
                    # the table order the oracle also reads; `route` is the resolved set.
                    ordered_routes = [entry.get("experts") for entry in trace["layers"]]
                expected = CHAIN_ROUTES[token]
                for layer_index in range(depth):
                    if resolved_routes[layer_index] != list(expected[layer_index]):
                        raise AssertionError(
                            f"{CHAIN_CONTRACT}: the native probe route changed at token "
                            f"{token} layer {layer_index}: {resolved_routes[layer_index]}, "
                            f"expected {list(expected[layer_index])}"
                        )
                    oracle_route = oracle_layers[layer_index]["route"]["indices"]
                    # The set is the routing decision; the order is an execution/trace detail and
                    # must not turn a correct route set into a divergence.
                    probe_route_set = sorted(int(value) for value in ordered_routes[layer_index])
                    oracle_route_set = sorted(int(value) for value in oracle_route)
                    if probe_route_set != oracle_route_set:
                        nonfinite = _first_nonfinite_stage(summary, output)
                        detail = (f"; first non-finite oracle stage: {nonfinite}"
                                  if nonfinite else "")
                        raise AssertionError(
                            f"{CHAIN_CONTRACT}: first divergent stage is layer {layer_index} "
                            f"route (token {token}, depth {depth}): probe "
                            f"{ordered_routes[layer_index]}, oracle {oracle_route}{detail}"
                        )
                probe_argmax = int(payload.get("argmax_token", -1))
                if probe_argmax != CHAIN_ARGMAX[token][depth]:
                    raise AssertionError(
                        f"{CHAIN_CONTRACT}: the native probe argmax changed at token {token} "
                        f"depth {depth}: {probe_argmax}, expected "
                        f"{CHAIN_ARGMAX[token][depth]}"
                    )
                oracle_logits = _read_f32(output / f"logits-depth{depth}.f32", LOGITS)
                probe_logits = _read_f32(logits_path, LOGITS)
                logits_delta = _compare_depth(CHAIN_CONTRACT, "logits", oracle_logits,
                                              probe_logits, token, depth)
                oracle_hidden = _read_f32(output / f"hidden-depth{depth}.f32", WIDTH)
                reference_hidden = _read_f32(hidden_path, WIDTH)
                hidden_delta = _compare_depth(CHAIN_CONTRACT, "final hidden", oracle_hidden,
                                              reference_hidden, token, depth, hidden=True)
                oracle_argmax = int(np.argmax(oracle_logits))
                if oracle_argmax != probe_argmax:
                    raise AssertionError(
                        f"{CHAIN_CONTRACT}: argmax differs at token {token} depth {depth}: "
                        f"probe {probe_argmax}, oracle {oracle_argmax}"
                    )
                depth_summary = summary["depths"][str(depth)]
                if depth_summary["logits"]["sha256"] != hashlib.sha256(
                        oracle_logits.tobytes()).hexdigest():
                    raise AssertionError(
                        f"{CHAIN_CONTRACT}: the oracle logits checksum is wrong at depth {depth}")
                if depth_summary["final_hidden"]["sha256"] != hashlib.sha256(
                        oracle_hidden.tobytes()).hexdigest():
                    raise AssertionError(
                        f"{CHAIN_CONTRACT}: the oracle hidden checksum is wrong at depth {depth}")
                deltas[(token, depth)] = (logits_delta, hidden_delta)
                comparisons += 1
                print(
                    f"{CHAIN_CONTRACT}: token {token} depth {depth} PASS "
                    f"(logits max|delta|={logits_delta:.9g}, "
                    f"hidden max|delta|={hidden_delta:.9g}, argmax={probe_argmax}, "
                    f"routes={resolved_routes})"
                )
            print(f"{CHAIN_CONTRACT}: token {token} stage files PASS ({stage_count} checked)")
        worst_logits = max(value[0] for value in deltas.values())
        worst_hidden = max(value[1] for value in deltas.values())
        print(
            f"{CHAIN_CONTRACT}: PASS ({comparisons} depth comparisons, "
            f"max logits |delta|={worst_logits:.9g}, "
            f"max hidden |delta|={worst_hidden:.9g})"
        )
        if args.keep:
            print(f"{CHAIN_CONTRACT}: kept {root}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


def _prefill_probe(args: argparse.Namespace, tokens: list[int], layers: int,
                   trunks: dict[int, Path], expert_maps: dict[int, dict[int, Path]],
                   logits_path: Path, hidden_path: Path,
                   trace_path: Path) -> tuple[dict, dict]:
    """Run ``deepseek-chain-probe --tokens`` over the whole prompt."""
    command = [
        str(args.vastloom),
        "deepseek-chain-probe",
        str(args.model),
        str(args.globals),
        "--layers",
        str(layers),
        "--tokens",
        ",".join(str(token) for token in tokens),
        "--json",
    ]
    for layer in range(layers):
        command.extend(("--trunk", f"{layer}={trunks[layer]}"))
    for layer in range(layers):
        for token in tokens:
            command.extend(("--experts", f"{layer}={expert_maps[token][layer]}"))
    command.extend(("--dump-logits", str(logits_path),
                    "--dump-hidden", str(hidden_path),
                    "--dump-trace", str(trace_path)))
    process = _run(command, args.timeout)
    pressure = host_pressure_reason(process)
    if pressure is not None:
        raise SystemExit(skipped(PREFILL_CONTRACT, pressure))
    if process.returncode != 0:
        raise AssertionError(
            f"{PREFILL_CONTRACT}: the native prefill probe failed with exit "
            f"{process.returncode}\nstderr={process.stderr}\nstdout={process.stdout}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{PREFILL_CONTRACT}: the prefill probe did not print JSON: {error}\n"
            f"{process.stdout}"
        ) from error
    if payload.get("ok") is not True or payload.get("complete_model") is not False:
        raise AssertionError(f"{PREFILL_CONTRACT}: the prefill probe did not report a partial "
                             f"chain")
    if (payload.get("executed_layers") != layers
            or payload.get("model_layers") != MODEL_LAYERS
            or payload.get("prompt_tokens") != len(tokens)
            or payload.get("tokens") != list(tokens)
            or payload.get("logits") != LOGITS
            or payload.get("argmax_token") != PREFILL_ARGMAX):
        raise AssertionError(
            f"{PREFILL_CONTRACT}: the prefill probe reported an unexpected shape: "
            f"{ {key: payload.get(key) for key in ('executed_layers', 'prompt_tokens', 'tokens', 'logits', 'argmax_token')} }"
        )
    entries = payload.get("layers")
    if not isinstance(entries, list) or len(entries) != layers:
        raise AssertionError(f"{PREFILL_CONTRACT}: the prefill probe reported the wrong layer "
                             f"count")
    for index, entry in enumerate(entries):
        if entry.get("layer") != index:
            raise AssertionError(f"{PREFILL_CONTRACT}: prefill layer order is malformed")
        routes = entry.get("routes_per_token")
        if not isinstance(routes, list) or len(routes) != len(tokens):
            raise AssertionError(
                f"{PREFILL_CONTRACT}: layer {index} reports {routes!r}, expected one route "
                f"per prompt token"
            )
        expected = [sorted(PREFILL_ROUTES[index][token_index])
                    for token_index in range(len(tokens))]
        if routes != expected:
            raise AssertionError(
                f"{PREFILL_CONTRACT}: the native per-token route changed at layer {index}: "
                f"{routes}, expected {expected}"
            )
    if not logits_path.is_file() or logits_path.stat().st_size != LOGITS * 4:
        raise AssertionError(f"{PREFILL_CONTRACT}: the probe did not write the logits dump")
    if not hidden_path.is_file() or hidden_path.stat().st_size != WIDTH * 4:
        raise AssertionError(f"{PREFILL_CONTRACT}: the probe did not write the hidden dump")
    if not trace_path.is_file():
        raise AssertionError(f"{PREFILL_CONTRACT}: the probe did not write the trace")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if (trace.get("tokens") != list(tokens) or trace.get("executed_layers") != layers
            or trace.get("model_layers") != MODEL_LAYERS or trace.get("logits") != LOGITS):
        raise AssertionError(f"{PREFILL_CONTRACT}: the prefill trace disagrees with its JSON")
    trace_layers = trace.get("layers")
    if not isinstance(trace_layers, list) or len(trace_layers) != layers:
        raise AssertionError(f"{PREFILL_CONTRACT}: the prefill trace has the wrong layer count")
    for index, entry in enumerate(trace_layers):
        if entry.get("layer") != index:
            raise AssertionError(f"{PREFILL_CONTRACT}: the prefill trace layer order is "
                                 f"malformed")
        if entry.get("routes_per_token") != entries[index].get("routes_per_token"):
            raise AssertionError(
                f"{PREFILL_CONTRACT}: the trace and JSON per-token routes differ at layer "
                f"{index}"
            )
        if entry.get("route") != entries[index].get("route"):
            raise AssertionError(
                f"{PREFILL_CONTRACT}: the trace and JSON route differ at layer {index}"
            )
        if len(entry.get("experts", [])) == 0:
            raise AssertionError(
                f"{PREFILL_CONTRACT}: the trace reports no executed expert at layer {index}"
            )
    return payload, trace


def _prefill_oracle(args: argparse.Namespace, tokens: list[int], layers: int,
                    trunks: dict[int, Path], expert_maps: dict[int, dict[int, Path]],
                    root: Path) -> tuple[dict, Path]:
    output = root / "oracle-prefill"
    command = [str(args.python), str(args.oracle), "--model", str(args.model)]
    for layer in range(layers):
        command.extend(("--trunk", f"{layer}={trunks[layer]}"))
    for layer in range(layers):
        for token in tokens:
            command.extend(("--experts", f"{layer}={expert_maps[token][layer]}"))
    command.extend((
        "--globals", str(args.globals),
        "--tokens", ",".join(str(token) for token in tokens),
        "--layers", str(layers),
        "--output-dir", str(output),
        "--stages",
    ))
    process = _run(command, args.timeout)
    if process.returncode != 0:
        raise AssertionError(
            f"{PREFILL_CONTRACT}: the independent prefill oracle failed with exit "
            f"{process.returncode}\nstderr={process.stderr}\nstdout={process.stdout}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{PREFILL_CONTRACT}: the prefill oracle did not print JSON: {error}\n"
            f"{process.stdout}"
        ) from error
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise AssertionError(f"{PREFILL_CONTRACT}: the oracle did not write summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (payload.get("ok") is not True or summary.get("ok") is not True
            or summary.get("format") != "vastloom.deepseek_v4.prefill_oracle.v1"):
        raise AssertionError(f"{PREFILL_CONTRACT}: the prefill oracle did not report success")
    if (summary.get("executed_layers") != layers
            or summary.get("model_layers") != MODEL_LAYERS
            or summary.get("prompt_tokens") != len(tokens)
            or summary.get("tokens") != list(tokens)
            or summary.get("head_position") != len(tokens) - 1):
        raise AssertionError(f"{PREFILL_CONTRACT}: the prefill oracle described the wrong run")
    return summary, output


def _check_prefill_stages(summary: dict, output: Path, tokens: list[int],
                          layers: int) -> int:
    """Require every prefill stage, including the per-token expert stages."""
    stage_files = summary.get("stage_files")
    if not isinstance(stage_files, dict) or not stage_files:
        raise AssertionError(f"{PREFILL_CONTRACT}: the oracle did not report stage files")
    token_count = len(tokens)
    window_size = int(summary["configuration"]["window_size"])
    index_top_k = int(summary["configuration"]["index_top_k"])
    index_head_count = int(summary["configuration"]["index_head_count"])
    index_head_width = int(summary["configuration"]["index_head_width"])
    head_width = int(summary["configuration"]["head_width"])
    required: dict[str, int] = {}
    for layer in range(layers):
        required[f"layer{layer}.hidden"] = token_count * 4 * WIDTH
        required[f"layer{layer}.attn_norm"] = token_count * WIDTH
        required[f"layer{layer}.router_scores"] = token_count * 256
        required[f"layer{layer}.route_indices"] = token_count * 6
        required[f"layer{layer}.attention_output"] = token_count * WIDTH
        required[f"layer{layer}.shared_output"] = token_count * WIDTH
        ratio = int(summary["layers"][layer]["compression_ratio"])
        window = min(token_count, window_size)
        if ratio == 0:
            required[f"layer{layer}.attention_indices"] = token_count * window
        else:
            compressed = token_count // ratio
            required[f"layer{layer}.attention_indices"] = (
                token_count * (window + min(index_top_k, compressed)))
            required[f"layer{layer}.attention_compressor_projected"] = (
                token_count * 2 * head_width)
            required[f"layer{layer}.attention_compressor_scores"] = (
                token_count * 2 * head_width)
            required[f"layer{layer}.attention_compressed"] = compressed * head_width
            required[f"layer{layer}.indexer_query"] = (
                token_count * index_head_count * index_head_width)
            required[f"layer{layer}.indexer_weights"] = token_count * index_head_count
            required[f"layer{layer}.indexer_compressed"] = compressed * index_head_width
    required["final_hidden"] = WIDTH
    required["final_norm"] = WIDTH
    required["hc_head_mixes"] = 4
    required["logits"] = LOGITS
    intermediate_width = int(summary["configuration"]["intermediate_width"])
    for layer, record in enumerate(summary["layers"]):
        for token_index, route in enumerate(record["routes_per_token"]):
            for expert in route:
                for role, size in (("gate", intermediate_width),
                                   ("up", intermediate_width),
                                   ("activated", intermediate_width),
                                   ("output", WIDTH)):
                    required[f"layer{layer}.expert_{expert}_t{token_index}_{role}"] = size
    for name, expected in required.items():
        entry = stage_files.get(name)
        if not isinstance(entry, dict):
            raise AssertionError(f"{PREFILL_CONTRACT}: stage {name} is missing from the "
                                 f"summary")
        path = output / "stages" / f"{name}.f32"
        if not path.is_file():
            raise AssertionError(f"{PREFILL_CONTRACT}: stage file {path} is missing")
        values = _read_f32(path, expected)
        if entry.get("count") != expected:
            raise AssertionError(f"{PREFILL_CONTRACT}: stage {name} reports the wrong count")
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{PREFILL_CONTRACT}: stage {name} contains a non-finite value")
        if entry.get("sha256") != hashlib.sha256(values.tobytes()).hexdigest():
            raise AssertionError(f"{PREFILL_CONTRACT}: stage {name} checksum does not match "
                                 f"its file")
    summary_stages = summary.get("stages")
    if not isinstance(summary_stages, dict) or set(summary_stages) != set(stage_files):
        raise AssertionError(f"{PREFILL_CONTRACT}: stage checksums and stage files disagree")
    for name, entry in stage_files.items():
        checksums = summary_stages[name]
        if (checksums.get("count") != entry.get("count")
                or checksums.get("sha256") != entry.get("sha256")
                or checksums.get("fnv1a64") != entry.get("fnv1a64")):
            raise AssertionError(f"{PREFILL_CONTRACT}: stage checksums disagree for {name}")
    return len(required)


def _check_compressed_evidence(summary: dict, output: Path, tokens: list[int]) -> str:
    """Prove that layer 2 closed a window and attended over a compressed slot."""
    compression = summary.get("compression")
    if not isinstance(compression, dict):
        raise AssertionError(f"{PREFILL_CONTRACT}: the summary has no compression block")
    compressed = [entry for entry in compression.get("compressed_layers", [])
                  if entry.get("compression_exercised") is True]
    if not compressed:
        raise AssertionError(
            f"{PREFILL_CONTRACT}: no compressed layer was exercised; the prompt closed no "
            f"compression window"
        )
    entry = compressed[0]
    if entry.get("layer") != 2 or entry.get("ratio") != 4:
        raise AssertionError(f"{PREFILL_CONTRACT}: the exercised compressed layer is not "
                             f"layer 2 at ratio 4: {entry!r}")
    if entry.get("compressed_rows") != PREFILL_COMPRESSED_ROWS:
        raise AssertionError(
            f"{PREFILL_CONTRACT}: layer 2 reports {entry.get('compressed_rows')} compressed "
            f"rows, expected {PREFILL_COMPRESSED_ROWS}"
        )
    if entry.get("compressed_selected") != min(512, PREFILL_COMPRESSED_ROWS):
        raise AssertionError(f"{PREFILL_CONTRACT}: layer 2 selected the wrong number of "
                             f"compressed slots")
    if entry.get("selected_per_token") != PREFILL_SELECTED_PER_TOKEN:
        raise AssertionError(
            f"{PREFILL_CONTRACT}: layer 2 attends over {entry.get('selected_per_token')} "
            f"slots per token, expected {PREFILL_SELECTED_PER_TOKEN}"
        )
    if int(compression.get("total_compressed_rows", 0)) < 1:
        raise AssertionError(f"{PREFILL_CONTRACT}: the compressed-slot total is not positive")
    counts = compression.get("compressed_stage_counts")
    if not isinstance(counts, dict):
        raise AssertionError(f"{PREFILL_CONTRACT}: the compressed stage counts are missing")
    for name, expected in (("layer2.attention_compressed", 512),
                           ("layer2.indexer_compressed", 128)):
        if int(counts.get(name, 0)) != expected or expected <= 0:
            raise AssertionError(
                f"{PREFILL_CONTRACT}: stage {name} holds {counts.get(name)!r} values, "
                f"expected {expected}; the compressed-slot path was not exercised"
            )
    compressed_values = _read_f32(output / "stages" / "layer2.attention_compressed.f32", 512)
    if not np.all(np.isfinite(compressed_values)) or float(np.abs(compressed_values).max()) == 0.0:
        raise AssertionError(f"{PREFILL_CONTRACT}: the attention compressed slot is not a "
                             f"finite non-zero vector")
    indexer_values = _read_f32(output / "stages" / "layer2.indexer_compressed.f32", 128)
    if not np.all(np.isfinite(indexer_values)) or float(np.abs(indexer_values).max()) == 0.0:
        raise AssertionError(f"{PREFILL_CONTRACT}: the indexer compressed slot is not a "
                             f"finite non-zero vector")
    # The index map is the direct evidence that the slot is used, not merely computed:
    # positions 3 and 4 select cache index 5 (the compressed row) as their sixth slot,
    # while positions 0-2 leave it at -1.
    token_count = len(tokens)
    table = _read_f32(output / "stages" / "layer2.attention_indices.f32",
                      token_count * PREFILL_SELECTED_PER_TOKEN)
    table = table.astype(np.int64).reshape(token_count, PREFILL_SELECTED_PER_TOKEN)
    for token in range(token_count):
        window = min(token + 1, token_count)
        expected = ([*range(window), *([-1] * (token_count - window)),
                     token_count if token + 1 >= 4 else -1])
        actual = [int(value) for value in table[token]]
        if actual != expected:
            raise AssertionError(
                f"{PREFILL_CONTRACT}: layer-2 attention indices at position {token} are "
                f"{actual}, expected {expected}"
            )
    return (f"layer 2: {PREFILL_COMPRESSED_ROWS} compressed row, "
            f"{PREFILL_SELECTED_PER_TOKEN} slots/token, compressed stages "
            f"{counts['layer2.attention_compressed']}/{counts['layer2.indexer_compressed']}")


def run_prefill_integration(args: argparse.Namespace) -> int:
    for name, value in (
        ("vastloom", args.vastloom), ("model", args.model), ("globals", args.globals),
        ("oracle", args.oracle), ("python", args.python),
    ):
        if not value.exists():
            raise SystemExit(f"{PREFILL_CONTRACT} requires an existing --{name}: {value}")
    help_text = _help_text(args.vastloom)
    if "deepseek-chain-probe" not in help_text:
        print(f"{PREFILL_CONTRACT}: BLOCKED - the built CLI has no deepseek-chain-probe",
              file=sys.stderr)
        return 2
    if "--tokens" not in help_text:
        print(f"{PREFILL_CONTRACT}: BLOCKED - the built chain probe has no --tokens; rebuild "
              f"the release binary with multi-token prefill support", file=sys.stderr)
        return 2
    if "--dump-logits" not in help_text or "--dump-hidden" not in help_text:
        print(f"{PREFILL_CONTRACT}: BLOCKED - the chain probe does not expose both "
              f"--dump-logits and --dump-hidden", file=sys.stderr)
        return 2
    tokens = _parse_csv_ints(args.chain_tokens, "--chain-tokens")
    if tuple(tokens) != PREFILL_TOKENS:
        raise SystemExit(
            f"{PREFILL_CONTRACT}: the prefill section is pinned to the prompt "
            f"{list(PREFILL_TOKENS)}, got {tokens}"
        )
    layers = args.prefill_layers
    if layers != PREFILL_LAYERS:
        raise SystemExit(
            f"{PREFILL_CONTRACT}: the prefill section is pinned to {PREFILL_LAYERS} layers, "
            f"got {layers}"
        )
    trunks = _parse_layer_map(args.chain_trunk, "--chain-trunk")
    expert_maps = _parse_expert_maps(args.chain_experts)
    for layer in range(layers):
        if layer not in trunks:
            raise SystemExit(
                f"{PREFILL_CONTRACT}: the prefill needs one --chain-trunk for layer {layer}")
        if not trunks[layer].exists():
            raise SystemExit(f"{PREFILL_CONTRACT}: --chain-trunk path does not exist: "
                             f"{trunks[layer]}")
    for token in tokens:
        for layer in range(layers):
            if token not in expert_maps or layer not in expert_maps[token]:
                raise SystemExit(
                    f"{PREFILL_CONTRACT}: the prefill needs --chain-experts "
                    f"{token}:{layer}=PATH for prompt token {token} layer {layer}"
                )
            if not expert_maps[token][layer].exists():
                raise SystemExit(f"{PREFILL_CONTRACT}: --chain-experts path does not exist: "
                                 f"{expert_maps[token][layer]}")
    build_root = REPO_ROOT / "build"
    if not build_root.is_dir():
        raise SystemExit(f"{PREFILL_CONTRACT} requires the build directory: {build_root}")
    root = Path(tempfile.mkdtemp(prefix="vastloom-deepseek-prefill-oracle-", dir=build_root))
    try:
        logits_path = root / "probe-logits.f32"
        hidden_path = root / "probe-hidden.f32"
        trace_path = root / "probe-trace.json"
        payload, trace = _prefill_probe(args, tokens, layers, trunks, expert_maps,
                                        logits_path, hidden_path, trace_path)
        repeat_logits = root / "probe-logits-repeat.f32"
        repeat_hidden = root / "probe-hidden-repeat.f32"
        repeat_trace_path = root / "probe-trace-repeat.json"
        repeat_payload, repeat_trace = _prefill_probe(
            args, tokens, layers, trunks, expert_maps, repeat_logits, repeat_hidden,
            repeat_trace_path)
        if logits_path.read_bytes() != repeat_logits.read_bytes():
            raise AssertionError(
                f"{PREFILL_CONTRACT}: a repeated prefill probe produced different logits")
        if hidden_path.read_bytes() != repeat_hidden.read_bytes():
            raise AssertionError(
                f"{PREFILL_CONTRACT}: a repeated prefill probe produced a different hidden "
                f"state")
        if (payload.get("layers") != repeat_payload.get("layers")
                or trace.get("layers") != repeat_trace.get("layers")):
            raise AssertionError(
                f"{PREFILL_CONTRACT}: a repeated prefill probe produced different routes")
        print(f"{PREFILL_CONTRACT}: probe determinism PASS (two runs, bit-identical logits and "
              f"hidden, identical per-token routes)")

        summary, output = _prefill_oracle(args, tokens, layers, trunks, expert_maps, root)
        stage_count = _check_prefill_stages(summary, output, tokens, layers)
        evidence = _check_compressed_evidence(summary, output, tokens)
        print(f"{PREFILL_CONTRACT}: compressed-slot evidence PASS ({evidence})")

        # Artifact identity: five route artifacts per layer overlap on two bundles, so the raw
        # artifact sum holds 90 bundles and the union 88. The probe reported 88 relocated
        # bundles and 343 relocated tensors (73 trunk ranges + 270 expert ranges); both must
        # agree with the oracle's own merge.
        artifacts = summary.get("artifacts", {})
        layer_artifacts = artifacts.get("layers", {})
        bundles = sum(int(layer_artifacts[str(layer)]["expert_bundles"])
                      for layer in range(layers))
        trunk_ranges = sum(int(layer_artifacts[str(layer)]["trunk_ranges"])
                           for layer in range(layers))
        expert_ranges = sum(int(layer_artifacts[str(layer)]["expert_ranges"])
                            for layer in range(layers))
        expert_bytes = sum(int(layer_artifacts[str(layer)]["expert_bytes"])
                           for layer in range(layers))
        if (bundles, trunk_ranges + expert_ranges, expert_bytes) != (88, 343, 678_297_600):
            raise AssertionError(
                f"{PREFILL_CONTRACT}: merged artifact identity is "
                f"({bundles} bundles, {trunk_ranges + expert_ranges} ranges, "
                f"{expert_bytes} expert bytes), expected (88, 343, 678297600)"
            )
        print(f"{PREFILL_CONTRACT}: artifact identity PASS ({bundles} merged bundles, "
              f"{trunk_ranges + expert_ranges} ranges, {expert_bytes} expert artifact bytes)")

        oracle_layers = summary.get("layers")
        if not isinstance(oracle_layers, list) or len(oracle_layers) != layers:
            raise AssertionError(f"{PREFILL_CONTRACT}: the oracle layer records are missing")
        for layer in range(layers):
            oracle_sets = oracle_layers[layer].get("route_sets_per_token")
            expected = [sorted(PREFILL_ROUTES[layer][index]) for index in range(len(tokens))]
            probe_sets = payload["layers"][layer].get("routes_per_token")
            if oracle_sets != expected:
                raise AssertionError(
                    f"{PREFILL_CONTRACT}: first divergent stage is layer {layer} route: "
                    f"oracle {oracle_sets}, expected {expected}"
                )
            if probe_sets != expected:
                raise AssertionError(
                    f"{PREFILL_CONTRACT}: the probe per-token routes changed at layer "
                    f"{layer}: {probe_sets}, expected {expected}")
        print(f"{PREFILL_CONTRACT}: per-token routes PASS (all {layers} layers, "
              f"{len(tokens)} positions)")

        oracle_logits = _read_f32(output / "logits.f32", LOGITS)
        probe_logits = _read_f32(logits_path, LOGITS)
        logits_delta = _max_abs_diff(oracle_logits, probe_logits)
        if logits_delta > TOLERANCE:
            index = int(np.argmax(np.abs(oracle_logits.astype(np.float64)
                                         - probe_logits.astype(np.float64))))
            nonfinite = _first_nonfinite_stage(summary, output)
            detail = f"; first non-finite oracle stage: {nonfinite}" if nonfinite else ""
            raise AssertionError(
                f"{PREFILL_CONTRACT}: first divergent stage is the last-position logits: "
                f"max abs diff {logits_delta:.9g} > {TOLERANCE} at index {index} "
                f"(probe {float(probe_logits[index])!r}, "
                f"oracle {float(oracle_logits[index])!r}){detail}"
            )
        oracle_argmax = int(np.argmax(oracle_logits))
        probe_argmax = int(payload.get("argmax_token", -1))
        if oracle_argmax != PREFILL_ARGMAX or probe_argmax != PREFILL_ARGMAX:
            raise AssertionError(
                f"{PREFILL_CONTRACT}: argmax is not the pinned {PREFILL_ARGMAX} "
                f"(probe {probe_argmax}, oracle {oracle_argmax})"
            )
        oracle_hidden = _read_f32(output / "hidden.f32", WIDTH)
        probe_hidden = _read_f32(hidden_path, WIDTH)
        hidden_delta = _max_abs_diff(oracle_hidden, probe_hidden)
        if hidden_delta > TOLERANCE:
            raise AssertionError(
                f"{PREFILL_CONTRACT}: first divergent stage is the final hidden state: max "
                f"abs diff {hidden_delta:.9g} > {TOLERANCE}"
            )
        if not np.allclose(probe_hidden, oracle_hidden, rtol=STAGE_RTOL, atol=STAGE_ATOL):
            raise AssertionError(
                f"{PREFILL_CONTRACT}: the final hidden state exceeds rtol={STAGE_RTOL}, "
                f"atol={STAGE_ATOL} (max abs diff {hidden_delta:.9g})"
            )
        if summary["logits"]["sha256"] != hashlib.sha256(oracle_logits.tobytes()).hexdigest():
            raise AssertionError(f"{PREFILL_CONTRACT}: the oracle logits checksum is wrong")
        if summary["final_hidden"]["sha256"] != hashlib.sha256(
                oracle_hidden.tobytes()).hexdigest():
            raise AssertionError(f"{PREFILL_CONTRACT}: the oracle hidden checksum is wrong")
        if summary["logits"]["minimum"] != float(np.min(oracle_logits)) or \
                summary["logits"]["maximum"] != float(np.max(oracle_logits)):
            raise AssertionError(f"{PREFILL_CONTRACT}: the oracle logits range is inconsistent")
        print(
            f"{PREFILL_CONTRACT}: PASS (prompt {list(tokens)}, {layers} layers, "
            f"{stage_count} stages, logits max|delta|={logits_delta:.9g}, "
            f"hidden max|delta|={hidden_delta:.9g}, argmax={PREFILL_ARGMAX}, "
            f"min={summary['logits']['minimum']:.6f}, "
            f"max={summary['logits']['maximum']:.6f})"
        )
        if args.keep:
            print(f"{PREFILL_CONTRACT}: kept {root}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastloom", type=Path, help="built vastloom executable")
    parser.add_argument("--model", type=Path, help="metadata-only GGUF store directory")
    parser.add_argument("--artifact", type=Path, help="imported layer-0 trunk range artifact")
    parser.add_argument("--experts", type=Path, help="imported selected-expert range artifact")
    parser.add_argument("--globals", type=Path, help="imported global range artifact")
    parser.add_argument("--token", type=int, default=0,
                        help="single prefill token id for the layer-0 section (default: 0)")
    parser.add_argument("--chain-trunk", action="append", metavar="LAYER=PATH",
                        help="chain trunk artifact per layer (repeatable)")
    parser.add_argument("--chain-experts", action="append", metavar="TOKEN:LAYER=PATH",
                        help="chain selected-expert artifact per token and layer (repeatable)")
    parser.add_argument("--chain-tokens", type=str, default="0,1",
                        help="comma-separated token ids for the chain section (default: 0,1); "
                             "with --prefill this is the whole prompt")
    parser.add_argument("--chain-depths", type=str, default="1,2,3",
                        help="comma-separated prefix depths for the chain section (default: 1,2,3)")
    parser.add_argument("--prefill", action="store_true",
                        help="run the multi-token prefill section instead of the single-token "
                             "chain section (requires 2..16 --chain-tokens and per-token "
                             "--chain-experts maps)")
    parser.add_argument("--prefill-layers", type=int, default=3,
                        help="layers executed by the prefill section (default 3)")
    parser.add_argument("--oracle", type=Path,
                        default=REPO_ROOT / "tools" / "reference_deepseek_v4_layer_oracle.py",
                        help="independent oracle script")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="interpreter used to run the oracle (must provide NumPy)")
    parser.add_argument("--timeout", type=int, default=3_600,
                        help="per-command timeout in seconds (default: 3600)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary dumps and print their directory")
    parser.add_argument("--require-checksum-equality", action="store_true",
                        help="fail on raw FNV differences (only for a bit-identical implementation)")
    args = parser.parse_args(argv)

    run_synthetic()

    layer_inputs = (args.model, args.artifact, args.experts, args.globals)
    chain_requested = bool(args.chain_trunk or args.chain_experts)
    if args.prefill:
        missing = [name for name, value in (("--vastloom", args.vastloom),
                                            ("--model", args.model),
                                            ("--globals", args.globals))
                   if value is None]
        if missing or not chain_requested:
            parser.error(
                "the prefill section requires --vastloom, --model, --globals and "
                "--chain-trunk/--chain-experts (missing: "
                f"{missing + ([] if chain_requested else ['--chain-trunk/--chain-experts'])})"
            )
        return run_prefill_integration(args)
    if not any(layer_inputs) and args.vastloom is None and not chain_requested:
        print(f"{CONTRACT}: no artifacts supplied; synthetic section only")
        return 0
    if not all(layer_inputs) or args.vastloom is None:
        if not chain_requested:
            parser.error(
                "artifact integration requires --vastloom, --model, --artifact, --experts "
                "and --globals"
            )
        if not (args.model and args.globals and args.vastloom):
            parser.error(
                "the chain section requires --vastloom, --model and --globals "
                "in addition to --chain-trunk/--chain-experts"
            )
    status = 0
    if all(layer_inputs) and args.vastloom is not None:
        status = run_integration(args)
    if status == 0 and chain_requested:
        status = run_chain_integration(args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
