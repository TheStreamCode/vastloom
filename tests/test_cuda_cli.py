#!/usr/bin/env python3
"""Cross-platform CLI contract for the optional native CUDA driver boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from host_pressure import host_pressure_reason, skipped


CONTRACT = "CUDA CLI capability/unavailable contract"


def run(executable: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [executable, *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str]) -> None:
    if not condition:
        raise AssertionError(
            f"{message}\nexit={process.returncode}\nstdout={process.stdout}\nstderr={process.stderr}"
        )


def error_payload(process: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return json.loads(process.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", required=True)
    args = parser.parse_args()

    help_result = run(args.vastloom, "cuda-info", "--help")
    require(help_result.returncode == 0 and "64 MiB" in help_result.stdout
            and "native PTX" in help_result.stdout,
            "cuda-info help is missing its bounded allocation/kernel contract", help_result)

    for arguments in (
        ("--unknown", "--json"),
        ("--json", "--json"),
        ("--allocation-smoke", "--json"),
        ("--allocation-smoke", "0", "--json"),
        ("--allocation-smoke", "65MiB", "--json"),
        ("--rmsnorm-smoke", "--rmsnorm-smoke", "--json"),
        ("--allocation-smoke", "1MiB", "--rmsnorm-smoke", "--json"),
        ("--minimum-free-vram", "1GiB", "--json"),
    ):
        rejected = run(args.vastloom, "cuda-info", *arguments)
        require(rejected.returncode != 0, f"cuda-info accepted invalid arguments {arguments}", rejected)
        payload = error_payload(rejected)
        require(payload.get("ok") is False and isinstance(payload.get("error"), str),
                "cuda-info did not return a structured JSON error", rejected)

    # Closed-source CUDA drivers are not sanitizer-instrumented and some WSL builds leak their
    # cuInit bookkeeping or reject ASan's address layout. The fake-backend native test remains
    # fully sanitized; exercise the real optional driver only in non-sanitizer configurations.
    sanitizer_active = any(
        os.environ.get(name)
        for name in ("ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS")
    )
    if sanitizer_active:
        print("CUDA CLI capability/unavailable contract: PASS (live probe skipped under sanitizer)")
        return 0

    probe = run(args.vastloom, "cuda-info", "--json")
    if probe.returncode == 0:
        payload = json.loads(probe.stdout)
        require(payload.get("ok") is True and isinstance(payload.get("devices"), list),
                "successful cuda-info output is malformed", probe)
        for device in payload["devices"]:
            require(
                isinstance(device, dict)
                and isinstance(device.get("name"), str)
                and isinstance(device.get("total_vram_bytes"), int)
                and isinstance(device.get("compute_capability_major"), int)
                and isinstance(device.get("compute_capability_minor"), int),
                "cuda-info returned an incomplete device record",
                probe,
            )
        rmsnorm = run(args.vastloom, "cuda-info", "--rmsnorm-smoke", "--json")
        pressure = host_pressure_reason(rmsnorm)
        if pressure is not None:
            return skipped(f"{CONTRACT} live RMSNorm", pressure)
        require(rmsnorm.returncode == 0, "CUDA RMSNorm smoke failed", rmsnorm)
        rmsnorm_payload = json.loads(rmsnorm.stdout)
        smoke = rmsnorm_payload.get("rmsnorm_smoke")
        require(
            isinstance(smoke, dict)
            and smoke.get("passed") is True
            and smoke.get("rows") == 2
            and smoke.get("width") == 4096
            and isinstance(smoke.get("max_abs_error"), (int, float))
            and smoke["max_abs_error"] <= 3.0e-5
            and smoke.get("device_buffer_bytes") == 81920
            and smoke.get("host_to_device_bytes") == 49152
            and smoke.get("device_to_host_bytes") == 32768,
            "successful CUDA RMSNorm smoke output is malformed",
            rmsnorm,
        )
    else:
        payload = error_payload(probe)
        require(payload.get("ok") is False and isinstance(payload.get("error"), str),
                "unavailable CUDA driver did not fail with structured JSON", probe)

    print("CUDA CLI capability/unavailable contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
