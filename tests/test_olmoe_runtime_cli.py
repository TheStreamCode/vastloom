#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


EXPERT_BYTES = 12 * 1024 * 1024
ROUTES_PER_TOKEN = 16 * 8


def run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=timeout,
    )


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str] | None = None) -> None:
    if condition:
        return
    if process is not None:
        message += f"\nreturncode={process.returncode}\nstdout={process.stdout!r}\nstderr={process.stderr!r}"
    raise AssertionError(message)


def trace_sim(executable: str, trace: Path, policy: str) -> dict[str, object]:
    process = run([
        executable, "trace-sim", str(trace), "--vram", "0", "--ram", "2GiB",
        "--policy", policy, "--json",
    ], timeout=30)
    require(process.returncode == 0, f"trace simulation failed for {policy}", process)
    return json.loads(process.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastloom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    executable = str(args.vastloom.resolve())
    model = str(args.model.resolve())
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))

    inspection = run([executable, "moe-inspect", model, "--json"], timeout=300)
    require(inspection.returncode == 0, "full verified OLMoE inspection failed", inspection)
    info = json.loads(inspection.stdout)
    require(info["metadata_only"] is False and info["hashes_verified"] is True,
            "full checkpoint was not hash-verified", inspection)
    require(info["streaming_plan_candidate"] is True, "safe streaming preflight rejected the checkpoint", inspection)

    with tempfile.TemporaryDirectory(prefix="vastloom-olmoe-runtime-") as temporary:
        trace = Path(temporary) / "native-trace.csv"
        process = run([
            executable, "moe-logits", model, "--text", fixture["text"], "--top", "10",
            "--cache", "no-cache", "--cache-bytes", str(EXPERT_BYTES),
            "--io", "async", "--prefetch-depth", "2", "--io-timeout-ms", "30000",
            "--timeout-ms", "600000", "--trace", str(trace), "--no-verify", "--json",
        ])
        require(process.returncode == 0, "native streamed OLMoE forward failed", process)
        payload = json.loads(process.stdout)
        require(payload["greedy_id"] == fixture["greedy_token_id"], "greedy token differs from Transformers", process)
        require(payload["expert_cache_hits"] == 0, "no-cache execution reported a cache hit", process)
        require(payload["expert_cache_misses"] == ROUTES_PER_TOKEN, "expert miss count mismatch", process)
        require(payload["io_mode"] == "async" and payload["prefetch_depth"] == 2,
                "asynchronous I/O configuration was not reported", process)
        require(payload["expert_prefetches"] == ROUTES_PER_TOKEN,
                "bounded prefetch count differs from routed misses", process)
        require(payload["expert_prefetch_overlap_ms"] > 0.0,
                "asynchronous expert reads did not overlap any CPU compute", process)
        require(payload["expert_bytes_read"] == ROUTES_PER_TOKEN * EXPERT_BYTES,
                "measured expert bytes differ from the no-cache baseline", process)
        require(payload["process_io"]["available"] is True
                and payload["process_io"]["process_read_bytes"] >= payload["expert_bytes_read"],
                "process I/O counters did not observe the bounded expert reads", process)
        if payload["process_io"]["source"] == "windows-get-process-io-counters":
            require(payload["process_io"]["storage_read_bytes_available"] is False
                    and payload["process_io"]["storage_read_bytes"] is None,
                    "Windows process transfer bytes were mislabeled as physical storage bytes", process)
            require(ROUTES_PER_TOKEN <= payload["process_io"]["read_operations"] <= 2 * ROUTES_PER_TOKEN,
                    "Windows bounded expert reads regressed to fragmented operations", process)
        require(payload["trace_events"] == ROUTES_PER_TOKEN, "native trace event count mismatch", process)

        native_logits = {int(item["id"]): float(item["logit"]) for item in payload["top_logits"]}
        tolerance = float(fixture["logit_absolute_tolerance"])
        for item in fixture["top_logits"]:
            token_id = int(item["token_id"])
            require(token_id in native_logits, f"reference top-logit token {token_id} missing natively", process)
            require(math.isclose(native_logits[token_id], float(item["logit"]), abs_tol=tolerance, rel_tol=0.0),
                    f"logit mismatch for token {token_id}", process)

        with trace.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == ROUTES_PER_TOKEN, "telemetry CSV row count mismatch")
        reference_routes = {(int(item["token"]), int(item["layer"])): item for item in fixture["routes"]}
        grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
        for row in rows:
            key = (int(row["token"]), int(row["layer"]))
            grouped.setdefault(key, []).append(row)
            require(int(row["size_bytes"]) == EXPERT_BYTES, "telemetry expert size mismatch")
            require(row["tier"] == "nvme" and row["hit"] == "0", "no-cache telemetry tier mismatch")
            require(int(row["bytes_read"]) == EXPERT_BYTES, "telemetry read count mismatch")
            require(row["prefetched"] == "1", "asynchronous expert read was not marked prefetched")
            require(float(row["prefetch_wait_ms"]) >= 0.0 and float(row["prefetch_overlap_ms"]) >= 0.0,
                    "invalid asynchronous prefetch timing")
            require(float(row["submitted_ms"]) <= float(row["ready_ms"])
                    and float(row["queue_ms"]) >= 0.0
                    and float(row["compute_started_ms"]) <= float(row["compute_completed_ms"]),
                    "invalid expert readiness/compute timing")
            require(row["route_order"] == row["reduction_order"],
                    "native selected reduction order changed")
        require(grouped.keys() == reference_routes.keys(), "native/reference token-layer keys differ")
        for key, native in grouped.items():
            reference = reference_routes[key]
            require([int(row["route_order"]) for row in native] == list(range(8)),
                    f"route-order telemetry is not contiguous at token/layer {key}")
            require([int(row["expert"]) for row in native] == reference["experts"],
                    f"expert routing differs at token/layer {key}")
            for row, expected in zip(native, reference["weights"], strict=True):
                require(math.isclose(float(row["routing_weight"]), float(expected), abs_tol=2e-5, rel_tol=0.0),
                        f"routing weight differs at token/layer {key}")

        staged_trace = Path(temporary) / "native-staged-trace.csv"
        staged_process = run([
            executable, "moe-logits", model, "--text", fixture["text"], "--top", "10",
            "--cache", "no-cache", "--cache-bytes", str(EXPERT_BYTES),
            "--io", "async", "--expert-read", "staged", "--prefetch-depth", "2",
            "--io-timeout-ms", "30000", "--timeout-ms", "600000", "--trace", str(staged_trace),
            "--no-verify", "--json",
        ])
        require(staged_process.returncode == 0, "staged OLMoE forward failed", staged_process)
        staged_payload = json.loads(staged_process.stdout)
        require(staged_payload["greedy_id"] == fixture["greedy_token_id"],
                "staged greedy token differs from Transformers", staged_process)
        require(staged_payload["expert_read_mode"] == "staged"
                and staged_payload["staged_expert_reads"] == ROUTES_PER_TOKEN,
                "staged mode/count was not reported", staged_process)
        require(staged_payload["expert_bytes_read"] == ROUTES_PER_TOKEN * EXPERT_BYTES
                and staged_payload["staged_gate_up_bytes_read"] == ROUTES_PER_TOKEN * 8 * 1024 * 1024
                and staged_payload["staged_down_bytes_read"] == ROUTES_PER_TOKEN * 4 * 1024 * 1024,
                "staged byte accounting differs from the whole-expert baseline", staged_process)
        require(staged_payload["staged_down_compute_overlap_ms"] > 0.0,
                "down-stage reads did not overlap gate/up compute", staged_process)
        staged_logits = {int(item["id"]): float(item["logit"]) for item in staged_payload["top_logits"]}
        for item in fixture["top_logits"]:
            token_id = int(item["token_id"])
            require(token_id in staged_logits
                    and math.isclose(staged_logits[token_id], float(item["logit"]),
                                     abs_tol=tolerance, rel_tol=0.0),
                    f"staged logit mismatch for token {token_id}", staged_process)
        with staged_trace.open(newline="", encoding="utf-8") as stream:
            staged_rows = list(csv.DictReader(stream))
        require(len(staged_rows) == ROUTES_PER_TOKEN, "staged telemetry CSV row count mismatch")
        staged_grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
        for row in staged_rows:
            staged_grouped.setdefault((int(row["token"]), int(row["layer"])), []).append(row)
        for key, native in staged_grouped.items():
            require([int(row["expert"]) for row in native] == reference_routes[key]["experts"],
                    f"staged routing differs at token/layer {key}")
        staged_replay = trace_sim(executable, staged_trace, "no-cache")
        require(staged_replay["ready_order_available"] is True
                and staged_replay["ready_order_groups"] == 16,
                "staged readiness replay did not cover all layers")

        multi_case = fixture["multi_token_case"]
        multi_trace = Path(temporary) / "native-multi-token-trace.csv"
        multi_process = run([
            executable, "moe-logits", model, "--text", multi_case["text"], "--top", "10",
            "--cache", "lru", "--cache-bytes", "2GiB", "--timeout-ms", "600000",
            "--trace", str(multi_trace), "--no-verify", "--json",
        ])
        require(multi_process.returncode == 0, "native two-token cached OLMoE forward failed", multi_process)
        multi_payload = json.loads(multi_process.stdout)
        require(multi_payload["input_tokens"] == 2 and multi_payload["greedy_id"] == multi_case["greedy_token_id"],
                "two-token output differs from Transformers", multi_process)
        require(multi_payload["expert_cache_hits"] == ROUTES_PER_TOKEN
                and multi_payload["expert_cache_misses"] == ROUTES_PER_TOKEN,
                "real LRU cache did not reuse every repeated routed expert", multi_process)
        require(multi_payload["expert_bytes_read"] == ROUTES_PER_TOKEN * EXPERT_BYTES,
                "real LRU cache byte count mismatch", multi_process)
        multi_logits = {int(item["id"]): float(item["logit"]) for item in multi_payload["top_logits"]}
        for item in multi_case["top_logits"]:
            token_id = int(item["token_id"])
            require(token_id in multi_logits
                    and math.isclose(multi_logits[token_id], float(item["logit"]), abs_tol=tolerance, rel_tol=0.0),
                    f"two-token logit mismatch for token {token_id}", multi_process)
        with multi_trace.open(newline="", encoding="utf-8") as stream:
            multi_rows = list(csv.DictReader(stream))
        require(len(multi_rows) == 2 * ROUTES_PER_TOKEN, "two-token telemetry CSV row count mismatch")
        multi_reference = {(int(item["token"]), int(item["layer"])): item for item in multi_case["routes"]}
        multi_grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
        for row in multi_rows:
            key = (int(row["token"]), int(row["layer"]))
            multi_grouped.setdefault(key, []).append(row)
            expected_hit = key[0] == 1
            require(row["tier"] == ("ram" if expected_hit else "nvme"), "two-token telemetry tier mismatch")
            require(row["hit"] == ("1" if expected_hit else "0"), "two-token telemetry hit mismatch")
            require(int(row["bytes_read"]) == (0 if expected_hit else EXPERT_BYTES),
                    "two-token telemetry byte count mismatch")
        require(multi_grouped.keys() == multi_reference.keys(), "two-token native/reference route keys differ")
        for key, native in multi_grouped.items():
            reference = multi_reference[key]
            require([int(row["expert"]) for row in native] == reference["experts"],
                    f"two-token expert routing differs at {key}")
            for row, expected in zip(native, reference["weights"], strict=True):
                require(math.isclose(float(row["routing_weight"]), float(expected), abs_tol=2e-5, rel_tol=0.0),
                        f"two-token routing weight differs at {key}")

        async_cached = run([
            executable, "moe-logits", model, "--text", multi_case["text"], "--top", "1",
            "--cache", "lru", "--cache-bytes", "2GiB", "--io", "async", "--prefetch-depth", "2",
            "--timeout-ms", "600000", "--no-verify", "--json",
        ])
        require(async_cached.returncode == 0, "asynchronous cached OLMoE forward failed", async_cached)
        async_cached_payload = json.loads(async_cached.stdout)
        require(async_cached_payload["greedy_id"] == multi_case["greedy_token_id"]
                and async_cached_payload["expert_prefetches"] == ROUTES_PER_TOKEN
                and async_cached_payload["expert_cache_hits"] == ROUTES_PER_TOKEN
                and async_cached_payload["expert_cache_misses"] == ROUTES_PER_TOKEN,
                "asynchronous prefetch did not stop at real cache hits", async_cached)

        staged_cached = run([
            executable, "moe-logits", model, "--text", multi_case["text"], "--top", "1",
            "--cache", "lru", "--cache-bytes", "2GiB", "--io", "async",
            "--expert-read", "staged", "--prefetch-depth", "2",
            "--timeout-ms", "600000", "--no-verify", "--json",
        ])
        require(staged_cached.returncode == 0, "staged cached OLMoE forward failed", staged_cached)
        staged_cached_payload = json.loads(staged_cached.stdout)
        require(staged_cached_payload["greedy_id"] == multi_case["greedy_token_id"]
                and staged_cached_payload["staged_expert_reads"] == ROUTES_PER_TOKEN
                and staged_cached_payload["expert_cache_hits"] == ROUTES_PER_TOKEN
                and staged_cached_payload["expert_cache_misses"] == ROUTES_PER_TOKEN
                and staged_cached_payload["expert_bytes_read"] == ROUTES_PER_TOKEN * EXPERT_BYTES,
                "staged cache insertion/reuse differs from the whole-expert path", staged_cached)

        no_cache = trace_sim(executable, multi_trace, "no-cache")
        lru = trace_sim(executable, multi_trace, "lru")
        lfu = trace_sim(executable, multi_trace, "lfu")
        static_hot = trace_sim(executable, multi_trace, "static-hot")
        require(no_cache["storage_misses"] == 2 * ROUTES_PER_TOKEN, "no-cache simulation mismatch")
        for name, report in (("LRU", lru), ("LFU", lfu)):
            require(report["storage_misses"] == ROUTES_PER_TOKEN and report["ram_hits"] == ROUTES_PER_TOKEN,
                    f"{name} simulation did not reuse the actual routed experts")
            require(report["cold_expert_accesses"] == ROUTES_PER_TOKEN
                    and report["reused_expert_accesses"] == ROUTES_PER_TOKEN
                    and report["p50_unique_reuse_distance"] == ROUTES_PER_TOKEN - 1
                    and report["p95_unique_reuse_distance"] == ROUTES_PER_TOKEN - 1,
                    f"{name} reuse-distance report differs from the repeated-token fixture")
            require(report["ready_order_available"] is True and report["ready_order_groups"] == 32,
                    f"{name} readiness replay did not cover every token/layer group")
        require(static_hot["ram_hits"] == 2 * ROUTES_PER_TOKEN and static_hot["storage_misses"] == 0,
                "static-hot simulation mismatch")

        timed_out = run([
            executable, "moe-logits", model, "--text", fixture["text"], "--cache", "no-cache",
            "--timeout-ms", "1", "--no-verify", "--json",
        ])
        require(timed_out.returncode != 0 and "cancelled or timed out safely" in timed_out.stderr,
                "bounded OLMoE timeout did not stop before expert execution", timed_out)

        injected_failures = {
            "fail": "Injected I/O worker failure",
            "short-read": "Short read",
            "allocation": "Injected I/O allocation failure",
            "cancelled": "I/O request cancelled safely",
            "timeout": "I/O request timed out safely",
        }
        for fault, expected_error in injected_failures.items():
            failed = run([
                executable, "moe-logits", model, "--text", fixture["text"], "--cache", "no-cache",
                "--io", "async", "--prefetch-depth", "1", "--inject-io-fault", fault,
                "--timeout-ms", "600000", "--no-verify", "--json",
            ])
            require(failed.returncode != 0 and expected_error in failed.stderr,
                    f"injected {fault} did not fail closed", failed)

        staged_short_read = run([
            executable, "moe-logits", model, "--text", fixture["text"], "--cache", "no-cache",
            "--io", "async", "--expert-read", "staged", "--prefetch-depth", "2",
            "--inject-io-fault", "short-read", "--timeout-ms", "600000", "--no-verify", "--json",
        ])
        require(staged_short_read.returncode != 0 and "Short read" in staged_short_read.stderr,
                "staged gate/up short-read injection did not fail closed", staged_short_read)

        incomplete = Path(temporary) / "incomplete-model"
        incomplete.mkdir()
        for source in args.model.iterdir():
            if source.suffix != ".safetensors":
                shutil.copyfile(source, incomplete / source.name)
        missing = run([executable, "moe-inspect", str(incomplete), "--no-verify", "--json"], timeout=30)
        missing_error = missing.stderr.lower()
        require(missing.returncode != 0 and "model-00001-of-00003.safetensors" in missing_error
                and ("missing" in missing_error or "size does not match" in missing_error),
                "missing full-model shards were not rejected", missing)

    print("OLMoE native streaming/reference/route-policy tests: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as error:
        print(f"ERROR: command timed out safely: {error}", file=sys.stderr)
        raise SystemExit(1) from error
