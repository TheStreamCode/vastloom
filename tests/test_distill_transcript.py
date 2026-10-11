from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "distill_transcript.py"
PROBES = ROOT / "tests" / "fixtures" / "distill_probes.json"

_spec = importlib.util.spec_from_file_location("distill_transcript", TOOL)
assert _spec is not None and _spec.loader is not None
distill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distill)


def fake_process(returncode: int = 0, payload: dict | None = None,
                 stderr: str = "") -> types.SimpleNamespace:
    stdout = json.dumps(payload if payload is not None else {
        "ok": True, "generated_ids": [7, 9], "text": "risposta",
        "finish_reason": "end_of_turn", "prompt_tokens": 5,
    })
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class ProbeValidationTests(unittest.TestCase):
    def test_loads_the_committed_probe_set(self) -> None:
        probes = distill.load_probes(PROBES)
        self.assertEqual(len(probes), 15)
        self.assertEqual([p["id"] for p in probes if p["split"] == "train"],
                         ["capital-1", "capital-2", "translate-1",
                          "capital-5", "capital-6", "capital-7",
                          "translate-3", "translate-4"])
        self.assertEqual(len([p for p in probes if p["split"] == "eval"]), 7)
        self.assertTrue(all(1 <= p["max_tokens"] <= 512 for p in probes))
        self.assertTrue(all("expected" in p for p in probes))

    def test_rejects_unknown_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probes.json"
            path.write_text(json.dumps([{"id": "a", "prompt": "x", "split": "holdout"}]),
                            encoding="utf-8")
            with self.assertRaises(distill.TranscriptError):
                distill.load_probes(path)

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probes.json"
            path.write_text(json.dumps([
                {"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"},
            ]), encoding="utf-8")
            with self.assertRaises(distill.TranscriptError):
                distill.load_probes(path)

    def test_rejects_out_of_range_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probes.json"
            path.write_text(json.dumps([{"id": "a", "prompt": "x", "max_tokens": 0}]),
                            encoding="utf-8")
            with self.assertRaises(distill.TranscriptError):
                distill.load_probes(path)

    def test_rejects_oversized_probe_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probes.json"
            path.write_text(json.dumps(
                [{"id": f"p{i}", "prompt": "x"} for i in range(distill.MAX_PROBES + 1)]),
                encoding="utf-8")
            with self.assertRaises(distill.TranscriptError):
                distill.load_probes(path)


class TranscriptionTests(unittest.TestCase):
    def test_records_teacher_output_through_a_fake_runner(self) -> None:
        probes = [{"id": "one", "prompt": "p1", "max_tokens": 4},
                  {"id": "two", "prompt": "p2", "max_tokens": 4}]
        seen_commands: list[list[str]] = []

        def runner(command: list[str]) -> types.SimpleNamespace:
            seen_commands.append(command)
            return fake_process()

        records = distill.transcribe(probes, runner, Path("vastloom"),
                                     Path("checkpoint"), Path("tokenizer"))
        self.assertEqual(len(seen_commands), 2)
        self.assertIn("deepseek-forward", seen_commands[0])
        self.assertIn("--chat", seen_commands[0])
        # The validated full-depth invocation is part of the contract: shallower targets
        # would be silently weaker.
        for flag, value in (("--layers", "43"), ("--trunk-window", "43")):
            index = seen_commands[0].index(flag)
            self.assertEqual(seen_commands[0][index + 1], value)
        self.assertIn("--cpu-only", seen_commands[0])
        self.assertIn("--ggml-compute", seen_commands[0])
        self.assertEqual(records[0]["completion_text"], "risposta")
        self.assertEqual(records[0]["completion_ids"], [7, 9])
        self.assertEqual(records[0]["finish_reason"], "end_of_turn")

    def test_raw_mode_omits_the_chat_flag(self) -> None:
        seen: list[list[str]] = []

        def runner(command: list[str]) -> types.SimpleNamespace:
            seen.append(command)
            return fake_process()

        distill.transcribe([{"id": "a", "prompt": "p", "max_tokens": 2}], runner,
                           Path("v"), Path("c"), Path("t"), chat=False)
        self.assertNotIn("--chat", seen[0])

    def test_stamps_teacher_accuracy_against_expected(self) -> None:
        probes = [{"id": "good", "prompt": "p", "max_tokens": 4, "expected": "Lisbona!"},
                  {"id": "bad", "prompt": "p", "max_tokens": 4, "expected": "Roma"}]
        payloads = iter([
            fake_process(payload={"ok": True, "generated_ids": [1], "text": "Lisbona",
                                  "finish_reason": "stop_token"}),
            fake_process(payload={"ok": True, "generated_ids": [2], "text": "Milano",
                                  "finish_reason": "stop_token"}),
        ])

        records = distill.transcribe(probes, lambda command: next(payloads),
                                     Path("v"), Path("c"), Path("t"))
        self.assertIs(records[0]["teacher_correct"], True)
        self.assertIs(records[1]["teacher_correct"], False)
        self.assertEqual(records[0]["expected"], "Lisbona!")

    def test_refuses_a_failed_teacher_run(self) -> None:
        def runner(command: list[str]) -> types.SimpleNamespace:
            return fake_process(returncode=3, stderr="boom")

        with self.assertRaises(distill.TranscriptError):
            distill.transcribe([{"id": "a", "prompt": "p", "max_tokens": 2}], runner,
                               Path("v"), Path("c"), Path("t"))

    def test_refuses_malformed_teacher_json(self) -> None:
        def runner(command: list[str]) -> types.SimpleNamespace:
            return fake_process(payload={"ok": False})

        with self.assertRaises(distill.TranscriptError):
            distill.transcribe([{"id": "a", "prompt": "p", "max_tokens": 2}], runner,
                               Path("v"), Path("c"), Path("t"))


class TranscriptArtifactTests(unittest.TestCase):
    def test_round_trip_and_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "transcript.json"
            records = [{"id": "a", "prompt": "p", "completion_text": "t",
                        "completion_ids": [1], "finish_reason": "end_of_turn"}]
            distill.write_transcript(out, records, "teacher-checkpoint")
            self.assertFalse((Path(temporary) / "transcript.json.tmp").exists())
            payload = distill.read_transcript(out)
            self.assertEqual(payload["format"], distill.TRANSCRIPT_FORMAT)
            self.assertEqual(payload["teacher"], "teacher-checkpoint")
            self.assertEqual(payload["records"][0]["completion_ids"], [1])

    def test_rejects_unrecognized_format_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "bad.json"
            out.write_text("{\"format\":\"something.else\"}", encoding="utf-8")
            with self.assertRaises(distill.TranscriptError):
                distill.read_transcript(out)


class CommandLineTests(unittest.TestCase):
    def test_main_writes_the_artifact_through_an_injected_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out.json"
            seen: list[list[str]] = []

            def runner(command: list[str]) -> types.SimpleNamespace:
                seen.append(command)
                return fake_process()

            code = distill.main(
                ["--vastloom", "vastloom.exe", "--checkpoint", "some-checkpoint",
                 "--tokenizer", "tok", "--probes", str(PROBES), "--out", str(out)],
                runner=runner,
            )
            self.assertEqual(code, 0)
            # Only the train split reaches the teacher; eval probes cost nothing.
            self.assertEqual(len(seen), 8)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 8)
            self.assertEqual(payload["records"][0]["completion_text"], "risposta")
            self.assertFalse((Path(temporary) / "out.json.tmp").exists())

    def test_resume_keeps_paid_for_records_and_refills_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out.json"
            state = {"calls": 0}

            def flaky(command: list[str]) -> types.SimpleNamespace:
                state["calls"] += 1
                if state["calls"] == 2:
                    return fake_process(returncode=75, stderr="budget refused")
                return fake_process(payload={
                    "ok": True, "generated_ids": [state["calls"]],
                    "text": f"t{state['calls']}", "finish_reason": "end_of_turn",
                })

            arguments = ["--vastloom", "vastloom.exe", "--checkpoint", "ck",
                         "--tokenizer", "tok", "--probes", str(PROBES), "--out", str(out)]
            self.assertNotEqual(distill.main(arguments, runner=flaky), 0)
            partial = distill.read_transcript(out)
            self.assertEqual([r["id"] for r in partial["records"]], ["capital-1"])

            self.assertEqual(distill.main(arguments, runner=lambda c: fake_process()), 0)
            payload = distill.read_transcript(out)
            self.assertEqual(len(payload["records"]), 8)
            self.assertEqual([r["id"] for r in payload["records"]],
                             ["capital-1", "capital-2", "translate-1",
                              "capital-5", "capital-6", "capital-7",
                              "translate-3", "translate-4"])
            self.assertEqual(payload["records"][0]["completion_ids"], [1],
                             "the resumed run must keep the earlier record verbatim")
            self.assertEqual(payload["records"][1]["runtime"]["trunk_window"], 43)

    def test_resume_backfills_the_accuracy_gate_without_rerunning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out.json"
            stale = {"format": distill.TRANSCRIPT_FORMAT, "teacher": "ck", "records": [
                {"id": "capital-1", "prompt": "p", "completion_text": "Lisbona",
                 "completion_ids": [5], "finish_reason": "stop_token"},
            ]}
            out.write_text(json.dumps(stale), encoding="utf-8")
            code = distill.main(
                ["--vastloom", "vastloom.exe", "--checkpoint", "ck",
                 "--tokenizer", "tok", "--probes", str(PROBES), "--out", str(out)],
                runner=lambda command: fake_process())
            self.assertEqual(code, 0)
            payload = distill.read_transcript(out)
            first = payload["records"][0]
            self.assertIs(first["teacher_correct"], True)
            self.assertEqual(first["completion_ids"], [5],
                             "backfill must not touch the stored teacher output")
            self.assertIn("runtime", payload["records"][1])

    def test_failing_teacher_makes_main_exit_non_zero_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out.json"
            code = distill.main(
                ["--vastloom", "vastloom.exe", "--checkpoint", "ck",
                 "--tokenizer", "tok", "--probes", str(PROBES), "--out", str(out)],
                runner=lambda command: fake_process(returncode=3, stderr="boom"),
            )
            self.assertNotEqual(code, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
