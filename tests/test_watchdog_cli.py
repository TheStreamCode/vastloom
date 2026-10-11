#!/usr/bin/env python3
"""Offline smoke checks for the vastloom-watchdog binary."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import unittest


WATCHDOG: Path

HEARTBEAT_TEMPLATE = (
    "SGHB1\n"
    "pid={pid}\n"
    "sequence=1\n"
    "unix_time_ms={unix_time_ms}\n"
    "state=normal\n"
    "ram_reserved=0\n"
    "vram_reserved=0\n"
)


def run_watchdog(*arguments: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(WATCHDOG), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )


class WatchdogCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="vastloom-watchdog-")
        self.root = Path(self.temporary.name)
        self.heartbeat = self.root / "heartbeat.txt"
        self.watch_pid = str(_spawn_sleeper().pid)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_heartbeat(self, pid: str, unix_time_ms: int) -> None:
        self.heartbeat.write_text(HEARTBEAT_TEMPLATE.format(pid=pid, unix_time_ms=unix_time_ms), encoding="ascii")

    def test_missing_arguments_exit_64(self) -> None:
        outcome = run_watchdog("--heartbeat", str(self.heartbeat))
        self.assertEqual(outcome.returncode, 64)

    def test_argument_ranges_exit_64(self) -> None:
        self.write_heartbeat(self.watch_pid, _now_ms())
        for invalid in ("--timeout-ms 100", "--interval-ms 10", "--grace-ms 999999999"):
            outcome = run_watchdog("--pid", self.watch_pid, "--heartbeat", str(self.heartbeat), invalid)
            self.assertEqual(outcome.returncode, 64, invalid)

    def test_unreadable_heartbeat_observe_only_exit_2(self) -> None:
        self.heartbeat.write_text("not a heartbeat\n", encoding="ascii")
        outcome = run_watchdog("--pid", self.watch_pid, "--heartbeat", str(self.heartbeat))
        self.assertEqual(outcome.returncode, 2)

    def test_stale_heartbeat_observe_only_exit_2(self) -> None:
        self.write_heartbeat(self.watch_pid, _now_ms() - 60_000)
        outcome = run_watchdog("--pid", self.watch_pid, "--heartbeat", str(self.heartbeat))
        self.assertEqual(outcome.returncode, 2)

    def test_pid_mismatch_exit_65(self) -> None:
        self.write_heartbeat("4194304", _now_ms())
        outcome = run_watchdog("--pid", self.watch_pid, "--heartbeat", str(self.heartbeat))
        self.assertEqual(outcome.returncode, 65)

    def test_stale_heartbeat_enforce_terminates_child_exit_3(self) -> None:
        self.write_heartbeat(self.watch_pid, _now_ms() - 60_000)
        outcome = run_watchdog(
            "--pid", self.watch_pid,
            "--heartbeat", str(self.heartbeat),
            "--timeout-ms", "250",
            "--interval-ms", "50",
            "--grace-ms", "500",
            "--enforce",
            timeout=30.0,
        )
        self.assertEqual(outcome.returncode, 3)
        time.sleep(0.2)
        outcome_poll = subprocess.run(
            ["tasklist", "/FI", f"PID eq {self.watch_pid}"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ) if sys.platform == "win32" else subprocess.run(
            ["ps", "-p", self.watch_pid],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertNotIn(self.watch_pid, outcome_poll.stdout)


def _now_ms() -> int:
    return int(time.time() * 1000.0)


def _spawn_sleeper() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def main() -> int:
    global WATCHDOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchdog", type=Path, required=True)
    arguments, unittest_arguments = parser.parse_known_args()
    WATCHDOG = arguments.watchdog.resolve()
    unittest.main(argv=[__file__, *unittest_arguments])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
