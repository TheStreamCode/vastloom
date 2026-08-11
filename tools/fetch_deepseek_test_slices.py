#!/usr/bin/env python3
"""Regenerate the three pinned DeepSeek expert slices used by the opt-in native parity tests.

The IQ1_S/IQ3_XXS kernel and complete-expert tests consume raw byte ranges extracted from one
official `UD-IQ1_S` shard. Those files are deliberately ignored by Git, so a clean machine cannot
re-run the tests that
`docs/reports/M7_DEEPSEEK_IQ1S_KERNEL_VALIDATION.md` and
`docs/reports/M7_DEEPSEEK_REAL_EXPERT_VALIDATION.md` describe. This tool restores exactly those
6,488,064 bytes and nothing else.

Downloading stays explicit and fail-closed: without `--allow-download` the tool only reports what
it would fetch. Every range is validated against its pinned offset, length and SHA-256 before it is
published read-only, so a wrong revision or a truncated response can never reach a test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import import_hf_model as base  # noqa: E402
import import_hf_ranges as ranges  # noqa: E402


MODEL_ID = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
REVISION = "fbbb5b93fb787c21338159b0af3318bb3f4d9768"
SOURCE_OBJECT = "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf"
SOURCE_SIZE = 49_093_726_624

# Layer 0 / expert 0, exactly as recorded in the M7 real-expert validation report.
SLICES = (
    ranges.SourceRange(
        role="expert-gate",
        label="layer0-expert0-gate.iq1s",
        source_object=SOURCE_OBJECT,
        source_path=SOURCE_OBJECT,
        source_size=SOURCE_SIZE,
        source_sha256="",
        offset=1_537_107_360,
        size=1_638_400,
    ),
    ranges.SourceRange(
        role="expert-up",
        label="layer0-expert0-up.iq1s",
        source_object=SOURCE_OBJECT,
        source_path=SOURCE_OBJECT,
        source_size=SOURCE_SIZE,
        source_sha256="",
        offset=1_967_521_184,
        size=1_638_400,
    ),
    ranges.SourceRange(
        role="expert-down",
        label="layer0-expert0-down.iq3xxs",
        source_object=SOURCE_OBJECT,
        source_path=SOURCE_OBJECT,
        source_size=SOURCE_SIZE,
        source_sha256="",
        offset=708_142_496,
        size=3_211_264,
    ),
)

EXPECTED_SHA256 = {
    "layer0-expert0-gate.iq1s":
        "4710ef29f496c13acabd1371d1d2a17be3f321f85a0b12287145e7c0b409c4f9",
    "layer0-expert0-up.iq1s":
        "7be122f1839feb8fdd84f57b9d20d5b57566227f8ba7d8ffc50123b158bf762f",
    "layer0-expert0-down.iq3xxs":
        "65b9cdc96f9e753584dc6eac4723616a494a98113bf25836fcda5ec72c400204",
}

CMAKE_VARIABLES = {
    "layer0-expert0-gate.iq1s": "VASTLOOM_DEEPSEEK_V4_IQ1S_SLICE",
    "layer0-expert0-up.iq1s": "VASTLOOM_DEEPSEEK_V4_IQ1S_UP_SLICE",
    "layer0-expert0-down.iq3xxs": "VASTLOOM_DEEPSEEK_V4_IQ3XXS_SLICE",
}


def source_url() -> str:
    return f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{SOURCE_OBJECT}"


def existing_is_valid(path: Path, entry: ranges.SourceRange) -> bool:
    if not path.is_file() or path.stat().st_size != entry.size:
        return False
    return base.hash_file(path) == EXPECTED_SHA256[entry.label]


def publish(partial: Path, target: Path) -> None:
    partial.replace(target)
    target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def fetch(entry: ranges.SourceRange, directory: Path, timeout: float, rate: int) -> None:
    target = directory / entry.label
    partial = directory / f"{entry.label}.partial"
    partial.unlink(missing_ok=True)
    actual = ranges.download_range(source_url(), partial, entry, timeout, rate)
    expected = EXPECTED_SHA256[entry.label]
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{entry.label}: pinned SHA-256 {expected} does not match downloaded {actual}"
        )
    target.unlink(missing_ok=True)
    publish(partial, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("build/research-gguf-slices"),
        help="destination directory for the extracted slices",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="required to perform any network request",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--max-bytes-per-second",
        type=int,
        default=0,
        help="optional download rate limit; 0 disables throttling",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory: Path = args.directory
    total = sum(entry.size for entry in SLICES)

    print(f"source:   {source_url()}")
    print(f"slices:   {len(SLICES)} ranges, {total:,} bytes total")
    print(f"target:   {directory.resolve()}")

    directory.mkdir(parents=True, exist_ok=True)
    missing = [entry for entry in SLICES if not existing_is_valid(directory / entry.label, entry)]
    for entry in SLICES:
        state = "present and verified" if entry not in missing else "missing or invalid"
        print(f"  {entry.label:<32} {entry.size:>10,} bytes  {state}")

    if not missing:
        print("all slices already match their pinned hashes; nothing to download")
    elif not args.allow_download:
        pending = sum(entry.size for entry in missing)
        print(
            f"dry run: {len(missing)} range(s) / {pending:,} bytes would be downloaded. "
            "Re-run with --allow-download to fetch them."
        )
        return 0
    else:
        for entry in missing:
            print(f"downloading {entry.label} ...")
            fetch(entry, directory, args.timeout, args.max_bytes_per_second)
            print(f"  verified {EXPECTED_SHA256[entry.label]}")

    print("\nConfigure the opt-in tests with:")
    for entry in SLICES:
        path = (directory / entry.label).resolve()
        print(f"  -D{CMAKE_VARIABLES[entry.label]}={path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
