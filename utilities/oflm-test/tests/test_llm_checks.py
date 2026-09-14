"""
Unit tests for LLMTask — the chat suite's per-round checks and its context
retention probe. All network/server calls are mocked so no OFLM server is
required.

This suite used to assert nothing at all: it sent two prompts and wrote the
answers to CSV, so an empty answer, a missing finish_reason, usage counts that
do not add up and an answer from a different model all "passed". Every test
here feeds a check the response one of those defects produces and asserts the
verdict is hard.

Run with:
    python3 -m pytest tests/test_llm_checks.py -v
or:
    python3 tests/test_llm_checks.py
"""
from __future__ import annotations

import sys
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oflm_test.tasks import LLMTask

MODEL = "gemma4:4b"


def _make_task():
    """Instantiate an LLMTask with all server I/O patched away."""
    with patch.object(LLMTask, "_get_oflm_version", return_value="0.9.99"), \
         patch.object(LLMTask, "_fetch_all_models", return_value=[]), \
         patch("os.makedirs"):
        return LLMTask(base_url="http://127.0.0.1:52625/v1", backend_os="linux")


def _usage(prompt=12, completion=34, total=None):
    """The usage object the SDK hands back, as far as these checks read it."""
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                           total_tokens=prompt + completion if total is None else total)


def _round(content, finish_reason="stop", usage=None, model=MODEL, reasoning=""):
    """One _one_round return value: (reasoning, content, finish_reason, usage, model)."""
    return (reasoning, content, finish_reason, usage, model)


class FakeWriter:
    """Collects the rows the suite would have written to CSV."""

    def __init__(self):
        self.rows = []

    def writerow(self, values):
        self.rows.append(list(values))


class TestCheckRound(unittest.TestCase):
    """What the server owes a chat client, whatever the weights answered."""

    def test_good_response_passes(self):
        verdict, detail = LLMTask._check_round(MODEL, MODEL, "Maxwell's equations are...",
                                               "stop", _usage())
        self.assertEqual(verdict, "PASS")
        self.assertIn("stop", detail)

    def test_empty_content_fails(self):
        verdict, detail = LLMTask._check_round(MODEL, MODEL, "", "stop", _usage())
        self.assertEqual(verdict, "FAIL")
        self.assertIn("empty", detail)

    def test_whitespace_only_content_fails(self):
        # An answer of blanks is an empty answer with extra steps, and the old
        # suite wrote it to CSV as a result like any other.
        verdict, _ = LLMTask._check_round(MODEL, MODEL, "   \n\t ", "stop", _usage())
        self.assertEqual(verdict, "FAIL")

    def test_missing_finish_reason_fails(self):
        verdict, detail = LLMTask._check_round(MODEL, MODEL, "an answer", None, _usage())
        self.assertEqual(verdict, "FAIL")
        self.assertIn("finish_reason", detail)
        self.assertIn("truncated", detail)

    def test_usage_that_does_not_add_up_fails(self):
        verdict, detail = LLMTask._check_round(MODEL, MODEL, "an answer", "stop",
                                               _usage(prompt=12, completion=34, total=99))
        self.assertEqual(verdict, "FAIL")
        self.assertIn("does not add up", detail)
        self.assertIn("99", detail)

    def test_zero_token_counts_fail(self):
        verdict, detail = LLMTask._check_round(MODEL, MODEL, "an answer", "stop",
                                               _usage(prompt=0, completion=34, total=34))
        self.assertEqual(verdict, "FAIL")
        self.assertIn("0 prompt", detail)

    def test_negative_token_counts_fail(self):
        verdict, _ = LLMTask._check_round(MODEL, MODEL, "an answer", "stop",
                                          _usage(prompt=12, completion=-3, total=9))
        self.assertEqual(verdict, "FAIL")

    def test_incomplete_usage_soft_fails(self):
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=34)   # no total
        verdict, detail = LLMTask._check_round(MODEL, MODEL, "an answer", "stop", usage)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("incomplete", detail)

    def test_absent_usage_is_fine(self):
        # The streaming path does not always carry usage, so its absence is not
        # a finding; only counts that contradict themselves are.
        verdict, _ = LLMTask._check_round(MODEL, MODEL, "an answer", "stop", None)
        self.assertEqual(verdict, "PASS")

    def test_a_response_naming_a_different_model_soft_fails(self):
        verdict, detail = LLMTask._check_round(MODEL, "qwen3:8b", "an answer", "stop", _usage())
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("qwen3:8b", detail)
        self.assertIn(MODEL, detail)

    def test_a_response_naming_no_model_soft_fails(self):
        verdict, detail = LLMTask._check_round(MODEL, None, "an answer", "stop", _usage())
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("names no model", detail)


class TestMemoryProbe(unittest.TestCase):
    """The probe that asks whether round one ever reached the model.

    Nothing but the earlier turn can supply the reference code, so a server
    that drops the conversation history cannot guess it.
    """

    def setUp(self):
        self.task = _make_task()
        self.writer = FakeWriter()

    def _run(self, first, second):
        with patch.object(self.task, "_one_round", side_effect=[first, second]), \
             patch("oflm_test.tasks.time.sleep"):
            self.task._run_memory_probe(self.writer, MODEL, -1, 0.3, None)

    def test_round_two_repeating_the_code_passes(self):
        code = LLMTask.MEMORY_CODE
        self._run(_round(code), _round(f"The code was {code}."))
        self.assertEqual(self.task.result.verdicts["PASS"], 1)
        self.assertEqual(self.task.result.hard_failures, 0)
        self.assertEqual(len(self.writer.rows), 1)
        row = self.writer.rows[0]
        self.assertEqual(row[0], MODEL)
        self.assertEqual(row[2], LLMTask.MEMORY_FOLLOWUP)
        self.assertTrue(row[6].startswith("PASS"))

    def test_forgetting_a_code_the_model_did_echo_fails(self):
        # The defect this is here for: the history never reached the model, so
        # round two has nothing to repeat.
        code = LLMTask.MEMORY_CODE
        self._run(_round(code), _round("I do not have any reference code."))
        self.assertEqual(self.task.result.hard_failures, 1)
        self.assertEqual(self.task.result.verdicts["FAIL"], 1)
        row = self.writer.rows[0]
        self.assertTrue(row[6].startswith("FAIL"))
        self.assertIn(code, row[6])
        self.assertIn("I do not have any reference code.", row[6])

    def test_a_code_the_model_never_echoed_is_inconclusive(self):
        # Round two failing says nothing about the history when round one never
        # produced the code in the first place, so this is soft, not red.
        self._run(_round("Sure, I will remember it."), _round("I do not have any code."))
        self.assertEqual(self.task.result.verdicts["SOFT-FAIL"], 1)
        self.assertEqual(self.task.result.hard_failures, 0)
        self.assertIn("inconclusive", self.writer.rows[0][6])

    def test_an_empty_round_two_is_reported_as_an_empty_response(self):
        # Not as a model that forgot. The probe runs the same per-round checks
        # the rest of the suite does, so the row names the actual defect.
        self._run(_round(LLMTask.MEMORY_CODE), _round(""))
        self.assertEqual(self.task.result.verdicts["FAIL"], 1)
        self.assertIn("empty response", self.writer.rows[0][6])
        self.assertNotIn("expected", self.writer.rows[0][6])

    def test_a_round_two_with_no_finish_reason_is_reported_as_that(self):
        self._run(_round(LLMTask.MEMORY_CODE),
                  _round(f"The code was {LLMTask.MEMORY_CODE}.", finish_reason=None))
        self.assertEqual(self.task.result.verdicts["FAIL"], 1)
        self.assertIn("finish_reason", self.writer.rows[0][6])

    def test_a_failed_call_is_recorded_as_an_error_row(self):
        with patch.object(self.task, "_one_round", side_effect=RuntimeError("connection reset")), \
             patch("oflm_test.tasks.time.sleep"):
            self.task._run_memory_probe(self.writer, MODEL, -1, 0.3, None)
        self.assertEqual(self.task.result.verdicts["ERROR"], 1)
        self.assertEqual(self.task.result.hard_failures, 1)
        self.assertIn("connection reset", self.writer.rows[0][6])


class TestTwoRounds(unittest.TestCase):
    """The two-round conversation both suites' rows come from."""

    def setUp(self):
        self.task = _make_task()
        self.writer = FakeWriter()

    def test_a_good_run_records_two_verdicts_and_two_rows(self):
        rounds = [_round("Maxwell's equations are...", usage=_usage()),
                  _round("In short, four equations.", usage=_usage())]
        with patch.object(self.task, "_one_round", side_effect=rounds), \
             patch("oflm_test.tasks.time.sleep"):
            self.task._run_two_rounds(self.writer, MODEL, LLMTask.PROMPT,
                                      LLMTask.FOLLOWUP_NON_STREAM, stream=False,
                                      max_completion_tokens=-1, temperature=0.3)
        self.assertEqual(self.task.result.verdicts["PASS"], 2)
        self.assertEqual(self.task.result.hard_failures, 0)
        self.assertEqual(len(self.writer.rows), 2)
        self.assertEqual([row[2] for row in self.writer.rows],
                         [LLMTask.PROMPT, LLMTask.FOLLOWUP_NON_STREAM])
        self.assertEqual([row[5] for row in self.writer.rows], ["stop", "stop"])
        self.assertTrue(all(row[1] == "Non-Stream" for row in self.writer.rows))

    def test_an_exception_in_round_one_stops_the_conversation(self):
        # Round two would be asking about an answer that never arrived, so the
        # run gives up rather than recording a second verdict on nothing.
        with patch.object(self.task, "_one_round",
                          side_effect=RuntimeError("connection reset")) as one_round, \
             patch("oflm_test.tasks.time.sleep"):
            self.task._run_two_rounds(self.writer, MODEL, LLMTask.PROMPT,
                                      LLMTask.FOLLOWUP_STREAM, stream=True,
                                      max_completion_tokens=-1, temperature=0.3)
        self.assertEqual(one_round.call_count, 1)
        self.assertEqual(self.task.result.verdicts["ERROR"], 1)
        self.assertEqual(self.task.result.total, 1)
        self.assertEqual(len(self.writer.rows), 1)
        self.assertIn("connection reset", self.writer.rows[0][6])

    def test_an_empty_answer_in_round_one_is_a_hard_failure(self):
        # The whole point of the suite gaining checks: this used to be a CSV
        # row with an empty cell and a green run.
        rounds = [_round("", usage=_usage()), _round("In short, four equations.", usage=_usage())]
        with patch.object(self.task, "_one_round", side_effect=rounds), \
             patch("oflm_test.tasks.time.sleep"):
            self.task._run_two_rounds(self.writer, MODEL, LLMTask.PROMPT,
                                      LLMTask.FOLLOWUP_NON_STREAM, stream=False,
                                      max_completion_tokens=-1, temperature=0.3)
        self.assertEqual(self.task.result.verdicts["FAIL"], 1)
        self.assertEqual(self.task.result.hard_failures, 1)
        self.assertTrue(self.writer.rows[0][6].startswith("FAIL"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
