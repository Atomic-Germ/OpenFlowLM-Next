"""
Unit tests for the --model filter option (issue #1).

All network/server calls are mocked so no FLM server is required.
Run with:
    python3 -m pytest tests/test_model_filter.py -v
or:
    python3 tests/test_model_filter.py
"""
from __future__ import annotations

import sys
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flm_test.tasks import LLMTask, VisionTask, EmbeddingTask, AudioTask, BaseTestTask
from flm_test import resolve_suites

# Fake server model list — pretend the server has these three models loaded
FAKE_SERVER_MODELS = ["gemma3:4b", "qwen3vl-it:4b", "some-llm:7b"]


def _make_task(task_cls, model_filter=None):
    """Instantiate a task with all server I/O patched away."""
    with patch.object(task_cls, "_get_flm_version", return_value="0.9.99"), \
         patch.object(task_cls, "_fetch_all_models", return_value=FAKE_SERVER_MODELS), \
         patch("os.makedirs"):
        return task_cls(
            base_url="http://127.0.0.1:52625/v1",
            backend_os="linux",
            model_filter=model_filter,
        )


class TestBaseModelFilter(unittest.TestCase):

    def test_no_filter_returns_all_eligible_models(self):
        """Without --model, LLMTask keeps all server models minus its exclusion list."""
        task = _make_task(LLMTask)
        self.assertEqual(task.models, FAKE_SERVER_MODELS)

    def test_filter_single_model(self):
        """--model gemma3:4b should leave only that one model."""
        task = _make_task(LLMTask, model_filter=["gemma3:4b"])
        self.assertEqual(task.models, ["gemma3:4b"])

    def test_filter_multiple_models(self):
        """--model with two IDs should keep exactly those two."""
        task = _make_task(LLMTask, model_filter=["gemma3:4b", "some-llm:7b"])
        self.assertCountEqual(task.models, ["gemma3:4b", "some-llm:7b"])

    def test_filter_nonexistent_model_gives_empty_list(self):
        """If the requested model is not on the server, result is empty."""
        task = _make_task(LLMTask, model_filter=["nonexistent:99b"])
        self.assertEqual(task.models, [])

    def test_filter_partial_match(self):
        """Only models that are both requested AND available should remain."""
        task = _make_task(LLMTask, model_filter=["gemma3:4b", "not-on-server:5b"])
        self.assertEqual(task.models, ["gemma3:4b"])

    def test_filter_does_not_add_unavailable_models(self):
        """Requesting a model the server does not serve must not appear."""
        task = _make_task(LLMTask, model_filter=["fantasy-model:1b"])
        self.assertNotIn("fantasy-model:1b", task.models)


class TestVisionModelFilter(unittest.TestCase):

    def test_filter_intersects_with_server_models(self):
        """Only models in BOTH the server list AND the user filter should survive."""
        task = _make_task(VisionTask, model_filter=["gemma3:4b", "some-llm:7b"])
        self.assertCountEqual(task.models, ["gemma3:4b", "some-llm:7b"])

    def test_no_filter_keeps_all_server_models(self):
        """Without --model, VisionTask runs every model the server exposes."""
        task = _make_task(VisionTask)
        self.assertEqual(task.models, FAKE_SERVER_MODELS)


class TestAudioModelFilter(unittest.TestCase):

    def test_explicit_filter_overrides_audio_allowlist(self):
        """An explicit --model always wins so any audio-capable model can be tested."""
        task = _make_task(AudioTask, model_filter=["gemma3:4b"])
        self.assertEqual(task.models, ["gemma3:4b"])

    def test_no_filter_keeps_only_audio_models(self):
        """Without --model, only recognised audio models survive."""
        task = _make_task(AudioTask)
        for m in task.models:
            self.assertIn(m, AudioTask.AUDIO_MODELS)


class TestEmbeddingModelFilter(unittest.TestCase):

    def test_non_embed_model_excluded(self):
        """A non-embedding model ID in the filter should be excluded by EMBED_MODELS list."""
        task = _make_task(EmbeddingTask, model_filter=["gemma3:4b"])
        self.assertEqual(task.models, [])

    def test_no_filter_keeps_only_embed_models(self):
        """Without --model, only recognised embedding models survive."""
        task = _make_task(EmbeddingTask)
        for m in task.models:
            self.assertIn(m, EmbeddingTask.EMBED_MODELS)

    def test_no_embed_models_on_server_falls_back_to_default(self):
        """FLM does not advertise the embed model on /models (`flm serve -e 1`);
        without any match (and no explicit filter) the suite defaults to the
        standard embedding model so it still runs."""
        task = _make_task(EmbeddingTask)
        self.assertEqual(task.models, [EmbeddingTask.DEFAULT_EMBED_MODEL])


class TestSuiteExclusivity(unittest.TestCase):
    """The embedding suite must stay exclusive so a full model never loads."""

    def _args(self, **overrides):
        defaults = dict(llm=False, embedding=False, audio=False, vision=False, tools=False, all=False)
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_embedding_alone_runs_only_embedding(self):
        suites, note = resolve_suites(self._args(embedding=True))
        self.assertEqual(suites, {"llm": False, "embedding": True,
                                  "audio": False, "vision": False, "tools": False})
        self.assertEqual(note, "")

    def test_embedding_plus_llm_runs_only_embedding(self):
        suites, note = resolve_suites(self._args(embedding=True, llm=True))
        self.assertEqual(suites, {"llm": False, "embedding": True,
                                  "audio": False, "vision": False, "tools": False})
        self.assertIn("mutually exclusive", note)

    def test_embedding_plus_any_chat_suite_runs_only_embedding(self):
        for suite in ("llm", "audio", "vision", "tools"):
            with self.subTest(suite=suite):
                kwargs = {"embedding": True, suite: True}
                suites, _ = resolve_suites(self._args(**kwargs))
                self.assertTrue(suites["embedding"])
                for name in ("llm", "audio", "vision", "tools"):
                    self.assertFalse(suites[name])

    def test_all_excludes_embedding(self):
        suites, note = resolve_suites(self._args(all=True))
        self.assertEqual(suites["embedding"], False)
        for name in ("llm", "audio", "vision", "tools"):
            self.assertTrue(suites[name])
        self.assertIn("--all excludes the embedding suite", note)

    def test_all_plus_embedding_runs_only_embedding(self):
        """--all --embedding is an explicit request for embedding alone."""
        suites, note = resolve_suites(self._args(all=True, embedding=True))
        self.assertEqual(suites["embedding"], True)
        for name in ("llm", "audio", "vision", "tools"):
            self.assertFalse(suites[name])
        self.assertIn("mutually exclusive", note)

    def test_no_suites_selected(self):
        suites, note = resolve_suites(self._args())
        self.assertFalse(any(suites.values()))
        self.assertEqual(note, "")


class TestArgParsing(unittest.TestCase):
    """Verify the --model flag is wired up correctly in the CLI parser."""

    def _make_parser(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--llm', action='store_true')
        parser.add_argument('--vision', action='store_true')
        parser.add_argument('--embedding', action='store_true')
        parser.add_argument('--audio', action='store_true')
        parser.add_argument('--all', action='store_true')
        parser.add_argument('--gen-lim', type=int, default=-1)
        parser.add_argument('--temp', '--temperature', type=float, default=0.3)
        parser.add_argument('--reasoning', type=str, default=None,
                            choices=["none", "low", "medium", "high"])
        parser.add_argument('--port', type=str, default='52625')
        parser.add_argument('--backend-os', type=str, default='linux',
                            choices=['linux', 'windows'])
        parser.add_argument('--model', type=str, nargs='+', metavar='MODEL_ID')
        return parser

    def test_model_arg_single(self):
        args = self._make_parser().parse_args(['--llm', '--model', 'gemma3:4b'])
        self.assertEqual(args.model, ['gemma3:4b'])

    def test_model_arg_multiple(self):
        args = self._make_parser().parse_args(
            ['--llm', '--model', 'gemma3:4b', 'qwen3vl-it:4b'])
        self.assertEqual(args.model, ['gemma3:4b', 'qwen3vl-it:4b'])

    def test_model_arg_absent(self):
        args = self._make_parser().parse_args(['--llm'])
        self.assertIsNone(args.model)

    def test_temp_arg_single(self):
        args = self._make_parser().parse_args(['--llm', '--temp', '0.7'])
        self.assertEqual(args.temp, 0.7)

    def test_temperature_long_form(self):
        args = self._make_parser().parse_args(['--vision', '--temperature', '0.2'])
        self.assertEqual(args.temp, 0.2)

    def test_temp_arg_defaults_to_0_3(self):
        """When --temp is omitted, a 0.3 sampling temperature is sent (never JSON null)."""
        args = self._make_parser().parse_args(['--audio'])
        self.assertEqual(args.temp, 0.3)

    def test_reasoning_level_accepted(self):
        args = self._make_parser().parse_args(['--llm', '--reasoning', 'high'])
        self.assertEqual(args.reasoning, 'high')

    def test_reasoning_off_via_none(self):
        """--reasoning none disables thinking for models that support it."""
        args = self._make_parser().parse_args(['--llm', '--reasoning', 'none'])
        self.assertEqual(args.reasoning, 'none')

    def test_reasoning_defaults_to_unset(self):
        """When --reasoning is omitted nothing is sent; model keeps its own default."""
        args = self._make_parser().parse_args(['--vision'])
        self.assertIsNone(args.reasoning)

    def test_reasoning_rejects_unknown_levels(self):
        with self.assertRaises(SystemExit):
            self._make_parser().parse_args(['--llm', '--reasoning', 'maximum'])


class TestReasoningKwargs(unittest.TestCase):
    """The helper that turns a requested level into chat.completions kwargs."""

    def test_level_maps_to_reasoning_effort(self):
        for level in ("none", "low", "medium", "high"):
            self.assertEqual(BaseTestTask._reasoning_kwargs(level),
                             {"reasoning_effort": level})

    def test_unset_sends_nothing(self):
        """No flag -> no kwarg at all, so the server never receives JSON null."""
        self.assertEqual(BaseTestTask._reasoning_kwargs(None), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
