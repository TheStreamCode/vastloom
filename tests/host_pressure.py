#!/usr/bin/env python3
"""Shared detection of Safety Governor refusals caused by host pressure.

Vastloom refuses real-model work when the machine it runs on is short of memory or disk. That
refusal is the Governor doing its job, and it depends on whatever else the operator happens to be
running. A test suite cannot tell that apart from a defect unless the distinction is made
explicitly, so every Python test that drives a governed command routes its failures through here.

The rule is deliberately narrow. A refusal counts as host pressure only when it carries the CLI's
`resource_exhausted` exit code and names a limit measured against memory that is free at that
instant. VRAM budget refusals, allocation failures, parity mismatches and malformed output are
defects and must keep failing.
"""

from __future__ import annotations

import json
import subprocess
from typing import Callable


# `ErrorCode::resource_exhausted` maps to this process exit code in the CLI.
RESOURCE_EXHAUSTED_EXIT = 75

# The exact host-side conditions checked by `make_safe_runtime_policy`, plus the tiny-model
# admission gate, which likewise measures memory that is free at this instant.
HOST_PRESSURE_MARKERS = (
    "of physical RAM",
    "of RAM headroom",
    "storage reserve",
    "safe RAM headroom for the model",
)


def _message(process: subprocess.CompletedProcess[str]) -> str:
    """Return the refusal text, whether the command reported JSON or plain stderr."""
    try:
        payload = json.loads(process.stderr)
    except (ValueError, TypeError):
        return process.stderr or ""
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return process.stderr or ""


def host_pressure_reason(process: subprocess.CompletedProcess[str]) -> str | None:
    """Return the refusal message when the host was too loaded to run the command."""
    if process.returncode != RESOURCE_EXHAUSTED_EXIT:
        return None
    message = _message(process)
    return message.strip() if any(marker in message for marker in HOST_PRESSURE_MARKERS) else None


class HostPressure(RuntimeError):
    """A governed command refused to start because the host was too loaded."""


def guard(process: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    """Pass a completed process through, raising `HostPressure` on a Governor host refusal.

    Tests that assert a specific refusal — a metadata-only checkpoint rejected for lacking payload,
    for instance — otherwise report the wrong reason as a contract violation. Wrap the test's own
    `run` helper with this, then wrap its entry point with `run_contract`.
    """
    reason = host_pressure_reason(process)
    if reason is not None:
        raise HostPressure(reason)
    return process


def run_contract(contract: str, entry: Callable[[], int]) -> int:
    """Run a test entry point, converting a host refusal into a skip."""
    try:
        return entry()
    except HostPressure as pressure:
        return skipped(contract, str(pressure))


def skipped(contract: str, reason: str) -> int:
    """Report a contract as unverified and exit with the code CTest maps to `Skipped`.

    Returning the Governor's own `resource_exhausted` code keeps the outcome honest: CMake marks
    these tests `SKIP_RETURN_CODE 75`, so the run shows the contract as skipped rather than passed.
    """
    print(f"{contract}: SKIPPED (host under pressure: {reason})")
    return RESOURCE_EXHAUSTED_EXIT
