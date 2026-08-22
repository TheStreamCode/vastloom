from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "distill_evaluate.py"
PROBES = ROOT / "tests" / "fixtures" / "distill_probes.json"

_spec = importlib.util.spec_from_file_location("distill_evaluate", TOOL)
assert _spec is not None and _spec.loader is not None
evaluate_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_tool)


def fake_process(text: str, returncode: int = 0) -> types.SimpleNamespace:
    payload = {"ok": True, "text": text} if returncode == 0 else {"ok": False}
    return types.SimpleNamespace(returncode=returncode,
                                 stdout=json.dumps(payload), stderr="boom")


class GradingTests(unittest.TestCase):
    def test_exact_match_ignores_case_and_punctuation(self) -> None:
        probes = [{"id": "a", "prompt": "p", "max_tokens": 4, "split": "train",
                   "expected": "The house is red"}]
        answers = iter(["The house is red.", "  the HOUSE is RED  "])

        records = evaluate_tool.evaluate(
            probes, lambda command: fake_process(next(answers)),
            Path("vastloom.exe"), Path("model"), module=Path("m.vlcm"))
        self.assertIs(records[0]["base_correct"], True)
        self.assertIs(records[0]["adapted_correct"], True)

    def test_leads_with_flag_catches_prefix_degeneration(self) -> None:
        probes = [{"id": "a", "prompt": "p", "max_tokens": 4, "split": "train",
                   "expected": "Lisbona"}]
        answers = iter(["Qual è la capitale", "Lisbonaa.berraaaaisis"])

        records = evaluate_tool.evaluate(
            probes, lambda command: fake_process(next(answers)),
            Path("vastloom.exe"), Path("model"), module=Path("m.vlcm"))
        self.assertIs(records[0]["base_correct"], False)
        self.assertNotIn("base_leads", records[0])
        self.assertIs(records[0]["adapted_correct"], False)
        self.assertIs(records[0]["adapted_leads"], True)


class RunnerContractTests(unittest.TestCase):
    def test_bare_arm_makes_one_call_and_adapted_two(self) -> None:
        probes = [{"id": "a", "prompt": "p", "max_tokens": 4, "split": "eval",
                   "expected": "x"}]

        bare_calls: list[list[str]] = []
        evaluate_tool.evaluate(probes, lambda c: (bare_calls.append(c), fake_process("x"))[1],
                               Path("vastloom.exe"), Path("model"), module=None)
        self.assertEqual(len(bare_calls), 1)
        self.assertNotIn("--capability-module", bare_calls[0])

        both_calls: list[list[str]] = []
        evaluate_tool.evaluate(probes, lambda c: (both_calls.append(c), fake_process("x"))[1],
                               Path("vastloom.exe"), Path("model"), module=Path("m"))
        self.assertEqual(len(both_calls), 2)
        self.assertIn("--capability-module", both_calls[1])

    def test_failing_generation_raises(self) -> None:
        probes = [{"id": "a", "prompt": "p", "max_tokens": 4, "split": "train",
                   "expected": "x"}]
        with self.assertRaises(evaluate_tool.EvaluateError):
            evaluate_tool.evaluate(probes, lambda c: fake_process("", returncode=3),
                                   Path("vastloom.exe"), Path("model"), module=None)


class CommandLineTests(unittest.TestCase):
    def test_summary_is_written_with_both_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "eval.json"
            expected_by_prompt = {
                probe["prompt"]: probe["expected"]
                for probe in json.loads(PROBES.read_text(encoding="utf-8"))
            }

            def runner(command: list[str]) -> types.SimpleNamespace:
                prompt = command[command.index("--prompt") + 1]
                return fake_process(expected_by_prompt[prompt])

            code = evaluate_tool.main(
                ["--vastloom", "vastloom.exe",
                 "--model", "C:/model",
                 "--probes", str(PROBES),
                 "--module", "fitted.vlcm",
                 "--out", str(out)],
                runner=runner,
            )
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], evaluate_tool.EVALUATE_FORMAT)
            self.assertEqual(payload["module"], "fitted.vlcm")
            self.assertEqual(len(payload["records"]), 7)
            self.assertTrue(all(r["base_correct"] for r in payload["records"]))
            self.assertTrue(all(r["adapted_correct"] for r in payload["records"]))

    def test_missing_expected_field_is_refused_before_any_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            probes_path = Path(temporary) / "probes.json"
            probes_path.write_text(json.dumps([{"id": "a", "prompt": "p"}]),
                                   encoding="utf-8")
            out = Path(temporary) / "out.json"
            calls: list[list[str]] = []

            code = evaluate_tool.main(
                ["--vastloom", "vastloom.exe", "--model", "m",
                 "--probes", str(probes_path), "--out", str(out)],
                runner=lambda c: (calls.append(c), fake_process("x"))[1],
            )
            self.assertNotEqual(code, 0)
            self.assertEqual(calls, [])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
