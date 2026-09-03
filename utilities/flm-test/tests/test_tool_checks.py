"""
Unit tests for ToolCallingTask — tool-calling checks at every complexity
level (L1-L5). All network/server calls are mocked so no FLM server is
required.

Run with:
    python3 -m pytest tests/test_tool_checks.py -v
or:
    python3 tests/test_tool_checks.py
"""
from __future__ import annotations

import sys
import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flm_test.tasks import ToolCallingTask


def _make_task():
    """Instantiate a ToolCallingTask with all server I/O patched away."""
    with patch.object(ToolCallingTask, "_get_flm_version", return_value="0.9.99"), \
         patch.object(ToolCallingTask, "_fetch_all_models", return_value=[]), \
         patch("os.makedirs"):
        return ToolCallingTask(base_url="http://127.0.0.1:52625/v1", backend_os="linux")


def _tool_call(name, arguments, call_id="call_001"):
    return {"id": call_id, "name": name, "arguments": arguments}


class TestToolSpec(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_expected_tools_advertised(self):
        names = [t["function"]["name"] for t in self.task.TOOLS]
        for expected in ("get_current_weather", "get_weather_forecast", "lookup_item_price"):
            self.assertIn(expected, names)

    def test_tool_names_unique(self):
        names = [t["function"]["name"] for t in self.task.TOOLS]
        self.assertEqual(len(names), len(set(names)))


class TestArgumentParsing(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_valid_json_string_parses(self):
        self.assertEqual(self.task._parse_arguments('{"location": "Paris"}'),
                         {"location": "Paris"})

    def test_dict_passthrough(self):
        self.assertEqual(self.task._parse_arguments({"location": "Paris"}),
                         {"location": "Paris"})

    def test_invalid_json_returns_none(self):
        self.assertIsNone(self.task._parse_arguments('{"location": '))

    def test_empty_returns_none(self):
        self.assertIsNone(self.task._parse_arguments(""))
        self.assertIsNone(self.task._parse_arguments(None))

    def test_non_object_json_returns_none(self):
        self.assertIsNone(self.task._parse_arguments("[1, 2, 3]"))


class TestLevel1BasicCall(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_correct_call_passes(self):
        verdict = self.task._verify_weather_call(
            [_tool_call("get_current_weather", '{"location": "Paris", "unit": "celsius"}')],
            "Paris")
        self.assertEqual(verdict[0], "PASS")

    def test_missing_call_fails(self):
        self.assertEqual(self.task._verify_weather_call([], "Paris")[0], "FAIL")

    def test_wrong_tool_fails(self):
        verdict = self.task._verify_weather_call(
            [_tool_call("get_weather_forecast", '{"location": "Paris"}')], "Paris")
        self.assertEqual(verdict[0], "FAIL")

    def test_wrong_city_fails(self):
        verdict = self.task._verify_weather_call(
            [_tool_call("get_current_weather", '{"location": "London"}')], "Paris")
        self.assertEqual(verdict[0], "FAIL")

    def test_malformed_arguments_fail(self):
        verdict = self.task._verify_weather_call(
            [_tool_call("get_current_weather", "{not json")], "Paris")
        self.assertEqual(verdict[0], "FAIL")


class TestLevel2ArgumentExtraction(unittest.TestCase):
    """Same mechanical check as L1 but the prompt requires inferring Paris from
    an indirect reference while resisting the decoy forecast tool."""

    def setUp(self):
        self.task = _make_task()

    def test_inferred_city_with_correct_tool_passes(self):
        verdict = self.task._verify_weather_call(
            [_tool_call("get_current_weather", '{"location": "Paris"}')], "Paris")
        self.assertEqual(verdict[0], "PASS")

    def test_decoy_forecast_tool_rejected(self):
        verdict = self.task._verify_weather_call(
            [_tool_call("get_weather_forecast",
                        '{"location": "Paris", "days": 1}')], "Paris")
        self.assertEqual(verdict[0], "FAIL")


class TestLevel3Restraint(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_direct_answer_passes(self):
        verdict = self.task._check_restraint("The capital of France is Paris.", [])
        self.assertEqual(verdict[0], "PASS")

    def test_any_tool_call_fails(self):
        verdict = self.task._check_restraint(
            "", [_tool_call("get_current_weather", '{"location": "France"}')])
        self.assertEqual(verdict[0], "FAIL")

    def test_empty_answer_fails(self):
        verdict = self.task._check_restraint("", [])
        self.assertEqual(verdict[0], "FAIL")

    def test_answer_without_paris_soft_fails(self):
        verdict = self.task._check_restraint("It is the capital of France.", [])
        self.assertEqual(verdict[0], "SOFT-FAIL")


class TestLevel4ParallelCalls(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def _two_city_calls(self):
        return [
            _tool_call("get_current_weather", '{"location": "Paris"}', call_id="call_a"),
            _tool_call("get_current_weather", '{"location": "Tokyo"}', call_id="call_b"),
        ]

    def test_both_cities_covered_passes(self):
        verdict = self.task._check_parallel("", self._two_city_calls())
        self.assertEqual(verdict[0], "PASS")

    def test_single_call_soft_fails(self):
        verdict = self.task._check_parallel(
            "", [_tool_call("get_current_weather", '{"location": "Paris"}')])
        self.assertEqual(verdict[0], "SOFT-FAIL")

    def test_no_calls_fail(self):
        verdict = self.task._check_parallel("", [])
        self.assertEqual(verdict[0], "FAIL")

    def test_wrong_cities_fail(self):
        calls = [
            _tool_call("get_current_weather", '{"location": "Paris"}', call_id="call_a"),
            _tool_call("get_current_weather", '{"location": "London"}', call_id="call_b"),
        ]
        verdict = self.task._check_parallel("", calls)
        self.assertEqual(verdict[0], "FAIL")


class TestLevel5ToolLoop(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_lookup_check_passes_for_widget(self):
        verdict = self.task._check_lookup_call(
            [_tool_call("lookup_item_price", '{"item": "widget"}')])
        self.assertEqual(verdict[0], "PASS")

    def test_lookup_check_fails_without_call(self):
        self.assertEqual(self.task._check_lookup_call([])[0], "FAIL")

    def test_lookup_check_fails_for_wrong_item(self):
        verdict = self.task._check_lookup_call(
            [_tool_call("lookup_item_price", '{"item": "gadget"}')])
        self.assertEqual(verdict[0], "FAIL")

    def test_final_answer_dollar_format_passes(self):
        self.assertEqual(self.task._check_final_answer("The total is $54.00.")[0], "PASS")

    def test_final_answer_plain_number_passes(self):
        self.assertEqual(
            self.task._check_final_answer("3 widgets at 10% off would cost about 54 dollars.")[0],
            "PASS")

    def test_final_answer_wrong_total_fails(self):
        self.assertEqual(self.task._check_final_answer("The total is $53.99.")[0], "FAIL")

    def test_final_answer_embedded_digits_do_not_match(self):
        self.assertFalse(ToolCallingTask.FINAL_ANSWER_PATTERN.search("about 545 items"))


class TestExecuteTool(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_known_location_weather(self):
        result = json.loads(self.task._execute_tool("get_current_weather", {"location": "Paris"}))
        self.assertEqual(result["temperature_celsius"], 18)
        self.assertEqual(result["conditions"], "partly cloudy")

    def test_unknown_location_gets_default(self):
        result = json.loads(
            self.task._execute_tool("get_current_weather", {"location": "Atlantis"}))
        self.assertEqual(result["temperature_celsius"],
                         ToolCallingTask.DEFAULT_TEMPERATURE_CELSIUS)

    def test_fahrenheit_conversion(self):
        result = json.loads(self.task._execute_tool(
            "get_current_weather", {"location": "Paris", "unit": "fahrenheit"}))
        self.assertEqual(result["temperature_fahrenheit"], 64.4)

    def test_price_lookup_widget_is_twenty(self):
        result = json.loads(self.task._execute_tool("lookup_item_price", {"item": "widget"}))
        self.assertEqual(result["unit_price_usd"], 20.0)

    def test_unknown_item_gets_default_price(self):
        result = json.loads(self.task._execute_tool("lookup_item_price", {"item": "fluxcapacitor"}))
        self.assertEqual(result["unit_price_usd"], ToolCallingTask.DEFAULT_PRICE_USD)

    def test_unknown_tool_returns_error_json(self):
        result = json.loads(self.task._execute_tool("delete_database", {}))
        self.assertIn("error", result)


def _delta(content=None, reasoning=None, tool_call_deltas=None):
    return SimpleNamespace(content=content, reasoning_content=reasoning,
                           tool_calls=tool_call_deltas or [])


def _tc_delta(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=call_id,
                           function=SimpleNamespace(name=name, arguments=arguments))


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class TestStreamAssembly(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_fragmented_stream_reassembles_one_call(self):
        chunks = [
            _chunk(_delta(tool_call_deltas=[_tc_delta(0, call_id="call_abc",
                                                      name="get_current_weather")])),
            _chunk(_delta(tool_call_deltas=[_tc_delta(0, arguments='{"location":')])),
            _chunk(_delta(tool_call_deltas=[_tc_delta(0, arguments=' "Paris"}')]),
                   ),
            _chunk(_delta(content="Let me check that for you.")),
        ]
        reasoning, content, tool_calls = self.task._collect_stream_with_tools(chunks)
        self.assertEqual(content, "Let me check that for you.")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["id"], "call_abc")
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")
        self.assertEqual(json.loads(tool_calls[0]["arguments"]), {"location": "Paris"})

    def test_interleaved_indexes_keep_order(self):
        chunks = [
            _chunk(_delta(tool_call_deltas=[
                _tc_delta(1, call_id="b", name="lookup_item_price"),
                _tc_delta(0, call_id="a", name="get_current_weather"),
            ])),
            _chunk(_delta(tool_call_deltas=[
                _tc_delta(1, arguments='{"item": "widget"}'),
                _tc_delta(0, arguments='{"location": "Paris"}'),
            ])),
        ]
        _, _, tool_calls = self.task._collect_stream_with_tools(chunks)
        self.assertEqual([tc["id"] for tc in tool_calls], ["a", "b"])
        self.assertEqual(tool_calls[1]["name"], "lookup_item_price")

    def test_plain_content_stream_has_no_tools(self):
        chunks = [_chunk(_delta(content="Hello")), _chunk(_delta(content=" world"))]
        reasoning, content, tool_calls = self.task._collect_stream_with_tools(chunks)
        self.assertEqual(content, "Hello world")
        self.assertEqual(tool_calls, [])


class TestCallModelNonStream(unittest.TestCase):

    def test_message_normalized_to_dict_form(self):
        task = _make_task()
        message = SimpleNamespace(
            content="Checking.",
            tool_calls=[SimpleNamespace(id="call_x", function=SimpleNamespace(
                name="get_current_weather", arguments='{"location": "Paris"}'))])
        fake_response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        task.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: fake_response)))
        reasoning, content, tool_calls = task._call_model(
            "test-model", [{"role": "user", "content": "hi"}], stream=False,
            max_completion_tokens=-1, temperature=None)
        self.assertEqual(content, "Checking.")
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")
        self.assertEqual(json.loads(tool_calls[0]["arguments"]), {"location": "Paris"})


class TestCallModelReasoningForwarding(unittest.TestCase):
    """_call_model must forward the requested reasoning effort to the API."""

    def _capturing_task(self):
        task = _make_task()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=[]))])

        task.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=fake_create)))
        return task, captured

    def _call(self, task, **overrides):
        kwargs = dict(model_id="test-model",
                      messages=[{"role": "user", "content": "hi"}],
                      stream=False, max_completion_tokens=-1,
                      temperature=None, reasoning=None)
        kwargs.update(overrides)
        return task._call_model(**kwargs)

    def test_reasoning_effort_forwarded(self):
        task, captured = self._capturing_task()
        self._call(task, reasoning="high")
        self.assertEqual(captured["reasoning_effort"], "high")

    def test_no_reasoning_sends_nothing(self):
        """Without --reasoning no reasoning_effort key reaches the request body."""
        task, captured = self._capturing_task()
        self._call(task)
        self.assertNotIn("reasoning_effort", captured)


class TestFormatToolCalls(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_empty_list_formats_as_none(self):
        self.assertEqual(self.task._format_tool_calls([]), "None")

    def test_arguments_parsed_into_dicts(self):
        formatted = json.loads(self.task._format_tool_calls(
            [_tool_call("get_current_weather", '{"location": "Paris"}')]))
        self.assertEqual(formatted[0]["arguments"], {"location": "Paris"})

    def test_invalid_arguments_kept_raw(self):
        formatted = json.loads(self.task._format_tool_calls(
            [_tool_call("get_current_weather", "{broken")]))
        self.assertEqual(formatted[0]["arguments"], "{broken")


if __name__ == "__main__":
    unittest.main(verbosity=2)
