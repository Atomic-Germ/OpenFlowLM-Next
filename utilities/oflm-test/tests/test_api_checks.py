"""
Unit tests for ApiConformanceTask — the server conformance suite A1-A5, plus
the verdict bookkeeping in BaseTestTask.record. All network/server calls are
mocked so no OFLM server is required.

Each check here passes against today's server, because the defects it was
written for are already fixed. A check that has never been red is the thing
issue #76 is about, so the central tests feed every check a synthetic pre-fix
response and assert it comes back a hard FAIL.

Run with:
    python3 -m pytest tests/test_api_checks.py -v
or:
    python3 tests/test_api_checks.py
"""
from __future__ import annotations

import sys
import os
import json
import unittest
from unittest.mock import patch

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oflm_test.tasks import ApiConformanceTask, BaseTestTask, SuiteResult, NON_CHAT_MODELS

IMPOSSIBLE = ApiConformanceTask.IMPOSSIBLE_MODEL
MODEL = "gemma3:4b"


def _make_task(server_models=None, model_filter=None):
    """Instantiate an ApiConformanceTask with all server I/O patched away."""
    with patch.object(ApiConformanceTask, "_get_oflm_version", return_value="0.9.99"), \
         patch.object(ApiConformanceTask, "_fetch_all_models",
                      return_value=list(server_models if server_models is not None else [MODEL])), \
         patch("os.makedirs"):
        return ApiConformanceTask(base_url="http://127.0.0.1:52625/v1", backend_os="linux",
                                  model_filter=model_filter)


def _error_body(code, error_type="invalid_request_error", message="messages must be an array"):
    return {"error": {"code": code, "type": error_type, "message": message}}


def _completion(content, model=MODEL, finish_reason="stop"):
    return {"id": "chatcmpl-1", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content}}]}


def _raw(status, body):
    """The (status, parsed, text) triple _post_json hands back."""
    text = "" if body is None else json.dumps(body)
    return status, body, text


class TestJudgeRefusal(unittest.TestCase):
    """A1 — the shared verdict for every request the server must refuse.

    The defect this was written for: the server's own error handler threw while
    building the error body, the outer handler swallowed it, and what went out
    was HTTP 200 with the exception text in an `error` field. Every client SDK
    reads that as a success.
    """

    def setUp(self):
        self.task = _make_task()

    def test_error_text_returned_with_http_200_fails(self):
        # The literal pre-fix body, verbatim: a 200 whose payload is the C++
        # JSON library's own type error.
        body = {"error": "[json.exception.type_error.302] type must be number, but is string"}
        verdict, detail = self.task._judge_refusal(*_raw(200, body))
        self.assertEqual(verdict, "FAIL")
        self.assertIn("never reaches a client as an error", detail)
        self.assertIn("type_error.302", detail)

    def test_ordinary_success_body_with_http_200_fails(self):
        # No error anywhere: the server just answered a request it should have
        # refused, which is the same defect wearing a nicer suit.
        status, body, text = _raw(200, _completion("ok"))
        verdict, detail = self.task._judge_refusal(status, body, text)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("instead of refusing", detail)

    def test_well_formed_openai_error_passes(self):
        status, body, text = _raw(400, _error_body("invalid_request_error"))
        verdict, detail = self.task._judge_refusal(status, body, text)
        self.assertEqual(verdict, "PASS")
        self.assertIn("400", detail)

    def test_a_numeric_error_code_is_not_a_failure(self):
        # The defect was in the reader, not in the field. The server's outer
        # catch builds its 500 with a numeric code and status_for() accepts a
        # number in 400-599 as the status, so both spellings are legitimate and
        # the only thing a client can see of the old bug is the status itself.
        status, body, text = _raw(500, {"error": {"message": "Handler exception: ...",
                                                  "type": "server_error", "code": 500}})
        verdict, detail = self.task._judge_refusal(status, body, text)
        self.assertEqual(verdict, "PASS")
        self.assertIn("500", detail)

    def test_non_json_body_soft_fails(self):
        # Refused, but by the wrong mechanism - the client gets a status and
        # nothing it can parse.
        verdict, detail = self.task._judge_refusal(400, None, "Internal Server Error")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("no error object", detail)

    def test_bare_string_error_soft_fails(self):
        verdict, detail = self.task._judge_refusal(*_raw(400, {"error": "bad request"}))
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("bare string", detail)

    def test_error_without_a_type_soft_fails(self):
        status, body, text = _raw(400, {"error": {"code": "invalid_request_error",
                                                  "message": "nope"}})
        verdict, detail = self.task._judge_refusal(status, body, text)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("names no type", detail)

    def test_unexpected_string_code_soft_fails_not_fails(self):
        # The refusal worked and the body is well formed; only the code differs
        # from the one the OpenAI API would use. Worth saying, not worth failing.
        status, body, text = _raw(400, _error_body("invalid_request_error"))
        verdict, detail = self.task._judge_refusal(status, body, text,
                                                   expected_code="model_not_found")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("model_not_found", detail)


class TestUnknownModelProbe(unittest.TestCase):
    """A2 — a tag the server cannot have loaded must be refused, not answered.

    PR #52: the server ignored the `model` field, let whichever model was
    resident answer, and echoed the requested tag back in the response.
    """

    def setUp(self):
        self.task = _make_task()

    def test_impossible_model_answered_by_another_model_fails(self):
        body = _completion("Sure, here you go.", model=IMPOSSIBLE)
        with patch.object(ApiConformanceTask, "_chat_raw", return_value=_raw(200, body)):
            (verdict, detail), status, note = self.task._probe_unknown_model(MODEL)
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(status, 200)
        # Both halves of the substitution have to be in the line someone reads:
        # what was asked for, and what came back claiming to be it.
        self.assertIn(IMPOSSIBLE, detail)
        self.assertIn("Sure, here you go.", detail)

    def test_model_not_found_refusal_passes(self):
        body = _error_body("model_not_found", "invalid_request_error", "no such model")
        with patch.object(ApiConformanceTask, "_chat_raw", return_value=_raw(400, body)):
            (verdict, detail), status, note = self.task._probe_unknown_model(MODEL)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(status, 400)

    def test_refusal_with_another_code_soft_fails(self):
        body = _error_body("invalid_request_error")
        with patch.object(ApiConformanceTask, "_chat_raw", return_value=_raw(400, body)):
            (verdict, detail), _, _ = self.task._probe_unknown_model(MODEL)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("model_not_found", detail)


class TestModelEchoProbe(unittest.TestCase):
    """A2, second half — an accepted request names the model that answered."""

    def setUp(self):
        self.task = _make_task()

    def test_reporting_the_requested_model_passes(self):
        with patch.object(ApiConformanceTask, "_chat_raw",
                          return_value=_raw(200, _completion("ok", model=MODEL))):
            (verdict, detail), _, _ = self.task._probe_model_echo(MODEL)
        self.assertEqual(verdict, "PASS")
        self.assertIn(MODEL, detail)

    def test_reporting_a_different_model_soft_fails(self):
        with patch.object(ApiConformanceTask, "_chat_raw",
                          return_value=_raw(200, _completion("ok", model="qwen3vl-it:4b"))):
            (verdict, detail), _, _ = self.task._probe_model_echo(MODEL)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("qwen3vl-it:4b", detail)

    def test_reporting_no_model_soft_fails(self):
        body = _completion("ok")
        body.pop("model")
        with patch.object(ApiConformanceTask, "_chat_raw", return_value=_raw(200, body)):
            (verdict, detail), _, _ = self.task._probe_model_echo(MODEL)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("names no model", detail)


class TestTruncationProbe(unittest.TestCase):
    """A3 — an answer cut at max_tokens must not claim it finished.

    PR #52: generation stopped at the token limit and the response still said
    finish_reason 'stop', so a client resuming or retrying had nothing to go on.
    """

    def setUp(self):
        self.task = _make_task()

    def _chat_returns(self, content, finish_reason):
        return patch.object(ApiConformanceTask, "_chat",
                            return_value=("", content, finish_reason, None, MODEL))

    def test_truncated_answer_reporting_stop_fails(self):
        with self._chat_returns("The transistor was inv", "stop"):
            (verdict, detail), _, note = self.task._probe_truncated(MODEL, stream=False)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("invisible to the caller", detail)
        self.assertIn("'stop'", note)

    def test_truncated_answer_reporting_length_passes(self):
        with self._chat_returns("The transistor was inv", "length"):
            (verdict, detail), _, _ = self.task._probe_truncated(MODEL, stream=True)
        self.assertEqual(verdict, "PASS")
        self.assertIn("length", detail)

    def test_no_finish_reason_at_all_fails(self):
        with self._chat_returns("The transistor was inv", None):
            (verdict, detail), _, _ = self.task._probe_truncated(MODEL, stream=True)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("no finish_reason", detail)

    def test_other_finish_reason_soft_fails(self):
        with self._chat_returns("The transistor was inv", "content_filter"):
            (verdict, detail), _, _ = self.task._probe_truncated(MODEL, stream=False)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("content_filter", detail)


class TestCompleteProbe(unittest.TestCase):
    """A3, the other half — an unbounded answer that ended on its own says 'stop'."""

    def setUp(self):
        self.task = _make_task()

    def _chat_returns(self, content, finish_reason):
        return patch.object(ApiConformanceTask, "_chat",
                            return_value=("", content, finish_reason, None, MODEL))

    def test_complete_answer_reporting_stop_passes(self):
        with self._chat_returns("ok", "stop"):
            (verdict, detail), _, _ = self.task._probe_complete(MODEL, stream=False)
        self.assertEqual(verdict, "PASS")

    def test_complete_answer_reporting_length_soft_fails(self):
        with self._chat_returns("ok", "length"):
            (verdict, detail), _, _ = self.task._probe_complete(MODEL, stream=False)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("context limit", detail)

    def test_no_finish_reason_fails(self):
        with self._chat_returns("ok", None):
            (verdict, detail), _, _ = self.task._probe_complete(MODEL, stream=True)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("no finish_reason", detail)


class TestIsolationProbe(unittest.TestCase):
    """A4 — what one request sets must not survive into the next.

    PR #59: one client asking for reasoning_effort turned thinking on for every
    later request, including ones that never mentioned it. The answers stayed
    plausible the whole time, which is why nothing else caught it.
    """

    def setUp(self):
        self.task = _make_task()

    @staticmethod
    def _turn(reasoning, content):
        return (reasoning, content, "stop", None, MODEL)

    # Four calls, in order: the probe, a second probe establishing whether the
    # model is reproducible at all, the poisoning request, then the probe again.
    def test_leaked_reasoning_effort_fails(self):
        calls = [
            self._turn("", "An NPU runs neural networks."),
            self._turn("", "An NPU runs neural networks."),
            self._turn("Let me consider.", "An NPU is an accelerator."),
            self._turn("Let me consider.", "An NPU runs neural networks."),
        ]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, note = self.task._probe_isolation(MODEL)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("no reasoning_effort", detail)
        self.assertIn("changed", detail)
        self.assertIn("before=False after=True", note)

    def test_unchanged_state_and_identical_text_passes(self):
        calls = [
            self._turn("", "An NPU runs neural networks."),
            self._turn("", "An NPU runs neural networks."),
            self._turn("Let me consider.", "An NPU is an accelerator."),
            self._turn("", "An NPU runs neural networks."),
        ]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, note = self.task._probe_isolation(MODEL)
        self.assertEqual(verdict, "PASS")
        self.assertIn("deterministic=True", note)

    def test_text_that_changes_only_after_the_poisoning_request_fails(self):
        # The two baseline probes agree, so the model reproduces its own answer;
        # a different one after a request that set temperature 1.0 is that
        # setting surviving, not run-to-run drift.
        calls = [
            self._turn("", "An NPU runs neural networks."),
            self._turn("", "An NPU runs neural networks."),
            self._turn("Let me consider.", "An NPU is an accelerator."),
            self._turn("", "An NPU speeds up inference."),
        ]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, _ = self.task._probe_isolation(MODEL)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("survived into the next one", detail)

    def test_a_model_that_does_not_reproduce_itself_is_judged_on_thinking_alone(self):
        # Decoding on the NPU is not always reproducible. When the two baseline
        # probes already differ, a later difference says nothing, so the text
        # comparison is dropped rather than reported as a finding.
        calls = [
            self._turn("", "An NPU runs neural networks."),
            self._turn("", "An NPU accelerates inference."),
            self._turn("Let me consider.", "An NPU is an accelerator."),
            self._turn("", "An NPU speeds up inference."),
        ]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, note = self.task._probe_isolation(MODEL)
        self.assertEqual(verdict, "PASS")
        self.assertIn("not usable here", detail)
        self.assertIn("deterministic=False", note)

    def test_the_poisoning_request_carries_reasoning_effort_high(self):
        # If the third call stopped asking for reasoning, the check would still
        # go green on a server that leaks - it would simply never poison it.
        calls = [self._turn("", "a"), self._turn("", "a"),
                 self._turn("x", "b"), self._turn("", "a")]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls) as chat:
            self.task._probe_isolation(MODEL)
        self.assertEqual(chat.call_args_list[2].kwargs.get("reasoning"), "high")
        for index in (0, 1, 3):
            self.assertIsNone(chat.call_args_list[index].kwargs.get("reasoning"))


class TestParityProbe(unittest.TestCase):
    """A5 — both response modes must describe the same request the same way."""

    def setUp(self):
        self.task = _make_task()

    @staticmethod
    def _turn(content, finish_reason, model=MODEL):
        return ("", content, finish_reason, None, model)

    def test_differing_finish_reason_fails(self):
        calls = [self._turn("The transistor", "length"),
                 self._turn("The transistor", "stop")]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, _ = self.task._probe_parity(MODEL)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("'length'", detail)
        self.assertIn("'stop'", detail)

    def test_stream_that_never_reports_a_finish_reason_fails(self):
        calls = [self._turn("The transistor", "length"),
                 self._turn("The transistor", None)]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, _ = self.task._probe_parity(MODEL)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("finish_reason", detail)

    def test_modes_naming_different_models_soft_fails(self):
        calls = [self._turn("The transistor", "length", model=MODEL),
                 self._turn("The transistor", "length", model="qwen3vl-it:4b")]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, _ = self.task._probe_parity(MODEL)
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("qwen3vl-it:4b", detail)

    def test_empty_stream_fails(self):
        calls = [self._turn("The transistor", "length"), self._turn("   ", "length")]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, _ = self.task._probe_parity(MODEL)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("no content", detail)

    def test_agreement_passes(self):
        calls = [self._turn("The transistor", "length"), self._turn("The transistor", "length")]
        with patch.object(ApiConformanceTask, "_chat", side_effect=calls):
            (verdict, detail), _, _ = self.task._probe_parity(MODEL)
        self.assertEqual(verdict, "PASS")
        self.assertIn("length", detail)


class TestErrorFields(unittest.TestCase):
    """The (code, type, message) reader every A1/A2 verdict is built on."""

    def test_openai_error_object(self):
        body = _error_body("model_not_found", "invalid_request_error", "no such model")
        self.assertEqual(BaseTestTask._error_fields(body),
                         ("model_not_found", "invalid_request_error", "no such model"))

    def test_bare_string_error_reports_only_the_message(self):
        # The shape the server produced when its own error handling threw; it
        # is reported as-is rather than smoothed over.
        self.assertEqual(BaseTestTask._error_fields({"error": "boom"}), (None, None, "boom"))

    def test_body_without_an_error_is_all_none(self):
        self.assertEqual(BaseTestTask._error_fields(_completion("ok")), (None, None, None))

    def test_non_dict_body_is_all_none(self):
        self.assertEqual(BaseTestTask._error_fields(None), (None, None, None))
        self.assertEqual(BaseTestTask._error_fields("not json"), (None, None, None))


class TestRecord(unittest.TestCase):
    """Verdict bookkeeping — the part that decides whether a run fails.

    Before this existed a FAIL landed in a CSV cell and the command exited 0,
    which is most of why the suite never caught anything.
    """

    def setUp(self):
        self.task = _make_task()

    def test_bare_string_verdict_is_counted(self):
        self.task.record("PASS", "m / A1 / malformed JSON")
        self.assertEqual(self.task.result.verdicts["PASS"], 1)
        self.assertEqual(self.task.result.failures, [])

    def test_tuple_verdict_is_counted(self):
        self.task.record(("PASS", "refused with HTTP 400"), "m / A1 / no messages")
        self.assertEqual(self.task.result.verdicts["PASS"], 1)

    def test_fail_lands_in_failures_with_where_and_detail(self):
        self.task.record(("FAIL", "answered HTTP 200"), "gemma3:4b / A1 / malformed JSON")
        self.assertEqual(self.task.result.failures,
                         ["gemma3:4b / A1 / malformed JSON: FAIL (answered HTTP 200)"])

    def test_inline_error_detail_is_split_off_the_string(self):
        self.task.record("ERROR: connection refused", "gemma3:4b / A3 / truncated")
        self.assertEqual(self.task.result.verdicts["ERROR"], 1)
        self.assertEqual(self.task.result.failures,
                         ["gemma3:4b / A3 / truncated: ERROR (connection refused)"])

    def test_pass_and_soft_fail_do_not_land_in_failures(self):
        self.task.record(("PASS", "fine"), "m / A2 / reported model")
        self.task.record(("SOFT-FAIL", "code is 'invalid_request_error'"), "m / A2 / unknown")
        self.assertEqual(self.task.result.failures, [])
        self.assertEqual(self.task.result.verdicts["SOFT-FAIL"], 1)

    def test_skip_verdicts_are_not_counted_at_all(self):
        for verdict in ("SKIP", "SKIPPED", "N/A"):
            self.task.record(verdict, "m / A4 / reasoning_effort")
        self.assertEqual(self.task.result.total, 0)
        self.assertEqual(self.task.result.failures, [])

    def test_hard_failures_counts_fail_and_error_only(self):
        self.task.record(("FAIL", "a"), "m / A3 / truncated")
        self.task.record(("ERROR", "b"), "m / A5 / parity")
        self.task.record(("SOFT-FAIL", "c"), "m / A2 / reported model")
        self.task.record(("PASS", "d"), "m / A1 / no messages")
        self.assertEqual(self.task.result.hard_failures, 2)
        self.assertEqual(self.task.result.total, 4)

    def test_soft_fail_alone_never_fails_a_run(self):
        self.task.record(("SOFT-FAIL", "the code is 'invalid_request_error'"), "m / A2 / unknown")
        self.assertEqual(self.task.result.hard_failures, 0)

    def test_record_returns_its_argument_unchanged(self):
        verdict = ("FAIL", "answered HTTP 200")
        self.assertIs(self.task.record(verdict, "m / A1 / malformed JSON"), verdict)
        self.assertEqual(self.task.record("PASS", "m / A1 / no messages"), "PASS")


class TestApiModelSelection(unittest.TestCase):
    """Which models the suite runs against.

    These checks are about the server, not the weights, so one chat model
    answers all of them; swapping models on the NPU buys no coverage.
    """

    def test_no_filter_tests_only_the_first_chat_model(self):
        task = _make_task(server_models=[MODEL, "qwen3vl-it:4b", "some-llm:7b"])
        self.assertEqual(task.models, [MODEL])

    def test_non_chat_models_are_excluded_before_the_first_is_taken(self):
        # Without the exclusion the whole suite would run against a specialised
        # model that is not a general chat model at all.
        task = _make_task(server_models=[NON_CHAT_MODELS[0], MODEL, "qwen3vl-it:4b"])
        self.assertEqual(task.models, [MODEL])

    def test_explicit_filter_is_honoured_in_full(self):
        task = _make_task(server_models=[MODEL, "qwen3vl-it:4b", "some-llm:7b"],
                          model_filter=[MODEL, "qwen3vl-it:4b"])
        self.assertCountEqual(task.models, [MODEL, "qwen3vl-it:4b"])

    def test_non_chat_model_is_excluded_even_when_asked_for(self):
        task = _make_task(server_models=[NON_CHAT_MODELS[0], MODEL],
                          model_filter=[NON_CHAT_MODELS[0]])
        self.assertEqual(task.models, [])

    def test_no_chat_model_leaves_nothing_to_test(self):
        task = _make_task(server_models=list(NON_CHAT_MODELS))
        self.assertEqual(task.models, [])


class TestSuiteResultShape(unittest.TestCase):
    """SuiteResult is what the runner turns into an exit code."""

    def test_empty_result_has_no_failures(self):
        result = SuiteResult("api")
        self.assertEqual(result.hard_failures, 0)
        self.assertEqual(result.total, 0)

    def test_suite_name_comes_from_the_task(self):
        self.assertEqual(_make_task().result.name, "api")
        self.assertEqual(ApiConformanceTask.SUITE_NAME, "api")


if __name__ == "__main__":
    unittest.main(verbosity=2)
