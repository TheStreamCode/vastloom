#!/usr/bin/env python3
"""End-to-end tests for Stargate's native TinyTransformer chat CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def invoke(
    executable: Path,
    *arguments: str,
    stdin: str | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *arguments],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="strict",
        timeout=timeout,
        check=False,
    )


def require(condition: bool, message: str, process: subprocess.CompletedProcess[str] | None = None) -> None:
    if condition:
        return
    details = message
    if process is not None:
        details += (
            f"\nreturncode={process.returncode}"
            f"\nstdout={process.stdout!r}"
            f"\nstderr={process.stderr!r}"
        )
    raise AssertionError(details)


def parse_json_output(process: subprocess.CompletedProcess[str]) -> dict[str, object]:
    require(process.stdout.strip() != "", "expected JSON on stdout", process)
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON output: {exc}\nstdout={process.stdout!r}") from exc
    require(isinstance(value, dict), "expected a JSON object", process)
    return value


def test_command_help(executable: Path) -> None:
    for command in ("chat", "model-info", "self-test"):
        process = invoke(executable, command, "--help")
        require(process.returncode == 0, f"{command} --help failed", process)
        require("Usage:" in process.stdout, f"{command} --help omitted usage", process)


def test_model_info(executable: Path, manifest: Path) -> None:
    process = invoke(executable, "model-info", str(manifest), "--json")
    require(process.returncode == 0, "model-info failed", process)
    payload = parse_json_output(process)
    require(payload.get("ok") is True, "model-info did not report success", process)
    model = payload.get("model")
    require(isinstance(model, dict), "model-info omitted model metadata", process)
    require(model.get("model_id") == "stargate/tinychat-v1", "unexpected model id", process)
    require(model.get("context_length") == 128, "unexpected context length", process)
    require(payload.get("manifest_hash_verified") is True, "manifest hash was not verified", process)


def test_direct_model_and_quick_manifest_mode(executable: Path, manifest: Path) -> None:
    model_path = manifest.parent / "stargate-tinychat-v1.sgtm"

    direct_info = invoke(executable, "model-info", str(model_path), "--json")
    require(direct_info.returncode == 0, "direct model-info failed", direct_info)
    direct_payload = parse_json_output(direct_info)
    require(direct_payload.get("manifest_hash_verified") is False,
            "direct model was incorrectly reported as hash verified", direct_info)

    direct_chat = invoke(
        executable, "chat", str(model_path), "--prompt", "Ciao", "--no-stream", "--json"
    )
    require(direct_chat.returncode == 0, "direct model chat failed", direct_chat)
    require(parse_json_output(direct_chat).get("text") == "Ciao! Sono Stargate TinyChat.",
            "direct model produced the wrong response", direct_chat)

    quick_info = invoke(executable, "model-info", str(manifest), "--no-verify", "--json")
    require(quick_info.returncode == 0, "quick manifest model-info failed", quick_info)
    quick_payload = parse_json_output(quick_info)
    require(quick_payload.get("manifest_hash_verified") is False,
            "--no-verify was incorrectly reported as full verification", quick_info)


def test_registered_vectors(executable: Path, manifest: Path, vectors_path: Path) -> None:
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))["vectors"]
    for vector in vectors:
        process = invoke(
            executable,
            "chat",
            str(manifest),
            "--prompt",
            vector["prompt"],
            "--max-tokens",
            "96",
            "--temperature",
            "0",
            "--top-k",
            "1",
            "--json",
        )
        require(process.returncode == 0, f"chat failed for {vector['prompt']!r}", process)
        payload = parse_json_output(process)
        require(payload.get("ok") is True, "chat JSON did not report success", process)
        require(payload.get("text") == vector["expected"], f"wrong response for {vector['prompt']!r}", process)
        stats = payload.get("stats")
        require(isinstance(stats, dict), "chat JSON omitted generation stats", process)
        require(stats.get("generated_tokens", 0) > 0, "chat generated no tokens", process)
        require(stats.get("stopped_on_timeout") is False, "chat timed out", process)
        require(payload.get("finish_reason") in {"end_of_turn", "end_of_sequence"},
                "unexpected deterministic finish reason", process)


def test_native_self_test(executable: Path, manifest: Path) -> None:
    process = invoke(executable, "self-test", str(manifest), "--json")
    require(process.returncode == 0, "native self-test failed", process)
    payload = parse_json_output(process)
    require(payload.get("ok") is True, "native self-test reported failure", process)
    cases = payload.get("cases")
    require(isinstance(cases, list) and len(cases) == 5, "unexpected native self-test case count", process)
    require(all(case.get("passed") is True for case in cases), "a native self-test vector failed", process)


def test_interactive_repl(executable: Path, manifest: Path) -> None:
    process = invoke(
        executable,
        "chat",
        str(manifest),
        "--no-stream",
        "--no-stats",
        stdin="Ciao\n/stats\n/model\n/exit\n",
    )
    require(process.returncode == 0, "interactive chat failed", process)
    require("model> Ciao! Sono Stargate TinyChat." in process.stdout, "interactive response missing", process)
    require("No response statistics" not in process.stdout, "interactive stats were not retained", process)
    require("Model:           stargate/tinychat-v1" in process.stdout, "/model output missing", process)


def test_json_remains_utf8_for_arbitrary_sampled_bytes(executable: Path, manifest: Path) -> None:
    process = invoke(
        executable,
        "chat",
        str(manifest),
        "--prompt",
        "unknown random prompt",
        "--temperature",
        "2",
        "--top-k",
        "100",
        "--seed",
        "1",
        "--max-tokens",
        "64",
        "--json",
    )
    require(process.returncode == 0, "high-temperature JSON generation failed", process)
    payload = parse_json_output(process)
    require(isinstance(payload.get("text"), str), "sampled JSON text is not valid Unicode", process)


def test_sampling_is_seeded(executable: Path, manifest: Path) -> None:
    command = (
        "chat",
        str(manifest),
        "--prompt",
        "Ciao",
        "--temperature",
        "0.7",
        "--top-k",
        "8",
        "--seed",
        "42",
        "--max-tokens",
        "32",
        "--json",
    )
    first = invoke(executable, *command)
    second = invoke(executable, *command)
    require(first.returncode == 0 and second.returncode == 0, "seeded sampling command failed", first)
    first_payload = parse_json_output(first)
    second_payload = parse_json_output(second)
    require(first_payload.get("text") == second_payload.get("text"), "same seed produced different text")


def test_cli_validation(executable: Path, manifest: Path) -> None:
    unknown = invoke(executable, "chat", str(manifest), "--unknown", "--json")
    require(unknown.returncode == 64, "unknown option did not return EX_USAGE", unknown)
    unknown_payload = json.loads(unknown.stderr)
    require(unknown_payload.get("ok") is False, "unknown option error was not JSON", unknown)

    missing = invoke(executable, "chat", str(manifest), "--prompt", "--json")
    require(missing.returncode == 64, "missing option value did not return EX_USAGE", missing)
    missing_payload = json.loads(missing.stderr)
    require("Missing value" in str(missing_payload.get("error")), "wrong missing-value error", missing)

    duplicate = invoke(executable, "chat", str(manifest), "--prompt", "Ciao", "--prompt", "Hello", "--json")
    require(duplicate.returncode == 64, "duplicate option did not return EX_USAGE", duplicate)

    oversized = invoke(executable, "chat", str(manifest), "--prompt", "x" * 256, "--json")
    require(oversized.returncode == 75, "oversized prompt did not fail safely", oversized)
    oversized_payload = json.loads(oversized.stderr)
    require(oversized_payload.get("ok") is False, "oversized prompt error was not JSON", oversized)


def test_manifest_integrity_gate(executable: Path, manifest: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="stargate-chat-integrity-") as temporary_directory:
        root = Path(temporary_directory)
        copied_manifest = root / manifest.name
        copied_model = root / "stargate-tinychat-v1.sgtm"
        shutil.copy2(manifest, copied_manifest)
        shutil.copy2(manifest.parent / copied_model.name, copied_model)

        with copied_model.open("r+b") as model_file:
            model_file.seek(-16, 2)
            original = model_file.read(1)
            require(len(original) == 1, "unable to select corruption byte")
            model_file.seek(-1, 1)
            model_file.write(bytes([original[0] ^ 0x01]))

        process = invoke(executable, "chat", str(copied_manifest), "--prompt", "Ciao", "--json")
        require(process.returncode == 65, "corrupt model bypassed manifest integrity gate", process)
        payload = json.loads(process.stderr)
        require(payload.get("ok") is False, "corrupt model error was not JSON", process)
        require("verification failed" in str(payload.get("error", "")).lower(), "wrong integrity error", process)

    with tempfile.TemporaryDirectory(prefix="stargate-chat-metadata-") as temporary_directory:
        root = Path(temporary_directory)
        copied_manifest = root / "MODEL.SGM"
        copied_model = root / "stargate-tinychat-v1.sgtm"
        shutil.copy2(manifest.parent / copied_model.name, copied_model)
        manifest_text = manifest.read_text(encoding="utf-8").replace(
            "model_id=stargate/tinychat-v1", "model_id=stargate/tampered-id"
        )
        copied_manifest.write_text(manifest_text, encoding="utf-8", newline="\n")

        process = invoke(executable, "chat", str(copied_manifest), "--prompt", "Ciao", "--json")
        require(process.returncode == 65, "manifest/model metadata mismatch was accepted", process)
        payload = json.loads(process.stderr)
        require("model id" in str(payload.get("error", "")).lower(), "wrong metadata mismatch error", process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stargate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    args = parser.parse_args()

    executable = args.stargate.resolve()
    manifest = args.manifest.resolve()
    vectors = args.vectors.resolve()
    require(executable.is_file(), f"stargate executable does not exist: {executable}")
    require(manifest.is_file(), f"manifest does not exist: {manifest}")
    require(vectors.is_file(), f"test vectors do not exist: {vectors}")

    test_command_help(executable)
    test_model_info(executable, manifest)
    test_direct_model_and_quick_manifest_mode(executable, manifest)
    test_registered_vectors(executable, manifest, vectors)
    test_native_self_test(executable, manifest)
    test_interactive_repl(executable, manifest)
    test_sampling_is_seeded(executable, manifest)
    test_json_remains_utf8_for_arbitrary_sampled_bytes(executable, manifest)
    test_cli_validation(executable, manifest)
    test_manifest_integrity_gate(executable, manifest)
    print("TinyChat CLI integration tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
