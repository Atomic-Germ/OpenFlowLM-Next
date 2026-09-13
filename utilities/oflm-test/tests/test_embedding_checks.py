"""
Unit tests for EmbeddingTask — embedding checks E1-E11. All embedding API calls
are mocked so no OFLM server is required.

Run with:
    python3 -m pytest tests/test_embedding_checks.py -v
or:
    python3 tests/test_embedding_checks.py
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

from oflm_test.tasks import EmbeddingTask


def _make_task():
    """Instantiate an EmbeddingTask with all server I/O patched away."""
    with patch.object(EmbeddingTask, "_get_oflm_version", return_value="0.9.99"), \
         patch.object(EmbeddingTask, "_fetch_all_models", return_value=[]), \
         patch("os.makedirs"):
        return EmbeddingTask(base_url="http://127.0.0.1:52625/v1", backend_os="linux")


def _embedding_entry(vector, index=0, object_type="embedding"):
    return SimpleNamespace(embedding=vector, index=index, object=object_type)


def _embed_response(data, object_type="list"):
    return SimpleNamespace(data=data, object=object_type)


def _vector_body(vector, model="embed-gemma:300m"):
    """A 200 carrying one embedding, in the shape /v1/embeddings really sends."""
    body = {"object": "list", "model": model,
            "data": [{"object": "embedding", "embedding": vector, "index": 0}]}
    return 200, body, json.dumps(body)


def _empty_body(status=200, model="embed-gemma:300m"):
    """A success envelope with no embedding in it."""
    body = {"object": "list", "model": model, "data": []}
    return status, body, json.dumps(body)


def _error(status, message, code="invalid_value", error_type="invalid_request_error",
           param="prompt_name"):
    """An OpenAI-shaped refusal, matching rest_handler.cpp's embeddings errors."""
    body = {"error": {"message": message, "type": error_type, "param": param, "code": code}}
    return status, body, json.dumps(body)


# The refusal a model with no task concept really sends - see the
# TaskPolicy::NotSupported branch in src/server/rest_handler.cpp.
NO_PROMPTS_MESSAGE = ("model 'bge-base:en-v1.5' has no task prompts; remove 'prompt_name'. "
                      "Passing one would be ignored, and the vector would come back "
                      "correctly shaped and unprefixed with nothing to show it.")

# The other refusal the engine makes on purpose: the model HAS prompts, but none
# of them serves query or document. src/open_npue_adapter/npue_embedding.cpp:85.
NO_MATCHING_PROMPT_MESSAGE = ("this model declares task prompts [Clustering, Classification] "
                              "and none of them matches the requested task. Refusing to pick "
                              "one: the prefix changes the vector.")


def _fake_embed_raw(by_prompt):
    """Stands in for _embed_raw, answering per prompt name in call order.

    Each value is a list of (status, body, text) triples; the last one repeats,
    so a prompt that is only ever drawn once needs a single entry.
    """
    def fake(model_id, input_text, **extra):
        queue = by_prompt[extra.get("prompt_name")]
        return queue.pop(0) if len(queue) > 1 else queue[0]
    return fake


class TestCheckNames(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_eleven_checks_defined(self):
        # run() indexes CHECK_NAMES positionally, so a name out of order puts
        # the wrong label on a verdict in the CSV.
        self.assertEqual(len(self.task.CHECK_NAMES), 11)
        for number, name in enumerate(self.task.CHECK_NAMES, 1):
            self.assertTrue(name.startswith(f"E{number} "), name)

    def test_embed_allowlist_present(self):
        self.assertIn("embed-gemma:300m", EmbeddingTask.EMBED_MODELS)

    def test_open_npue_tags_in_allowlist(self):
        # Without these the suite silently declines to test the models the
        # open_npue backend serves: the allowlist filter drops them and the
        # no-filter fallback substitutes DEFAULT_EMBED_MODEL.
        for tag in ("bge-base:en-v1.5", "all-minilm:l6-v2",
                    "gte-multilingual:base"):
            self.assertIn(tag, EmbeddingTask.EMBED_MODELS)

    def test_reference_models_is_a_subset_of_the_allowlist(self):
        # E8 compares against vectors for ONE model; every tag it claims to
        # cover must be a tag the suite can actually be pointed at.
        for tag in EmbeddingTask.REFERENCE_MODELS:
            self.assertIn(tag, EmbeddingTask.EMBED_MODELS)


class TestCosineSimilarity(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_identical_vectors_similarity_one(self):
        self.assertAlmostEqual(self.task._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 1.0)

    def test_orthogonal_vectors_similarity_zero(self):
        self.assertAlmostEqual(self.task._cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_similarity_minus_one(self):
        self.assertAlmostEqual(self.task._cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_parallel_vectors_scaled(self):
        self.assertAlmostEqual(self.task._cosine_similarity([2.0, 4.0], [1.0, 2.0]), 1.0)

    def test_mismatched_lengths_zero(self):
        self.assertEqual(self.task._cosine_similarity([1.0, 0.0], [1.0]), 0.0)

    def test_empty_vectors_zero(self):
        self.assertEqual(self.task._cosine_similarity([], []), 0.0)

    def test_zero_vector_zero(self):
        self.assertEqual(self.task._cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestVectorPreview(unittest.TestCase):

    def test_preview_truncates_and_reports_dim(self):
        self.assertEqual(
            EmbeddingTask._vector_preview([1.0, 2.0, 3.0, 4.0, 5.0]),
            "[1.000000, 2.000000, 3.000000, 4.000000, ...] (5 dims)")

    def test_empty_vector_gives_empty_string(self):
        self.assertEqual(EmbeddingTask._vector_preview([]), "")
        self.assertEqual(EmbeddingTask._vector_preview(None), "")


class TestResponseStructure(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_valid_response_passes(self):
        verdict, vector = self.task._check_response_structure(
            _embed_response([_embedding_entry([0.1, 0.2, 0.3])]))
        self.assertEqual(verdict[0], "PASS")
        self.assertEqual(vector, [0.1, 0.2, 0.3])

    def test_wrong_response_object_fails(self):
        verdict, _ = self.task._check_response_structure(
            SimpleNamespace(data=[_embedding_entry([0.1])], object="embedding"))
        self.assertEqual(verdict[0], "FAIL")

    def test_empty_data_fails(self):
        verdict, _ = self.task._check_response_structure(_embed_response([]))
        self.assertEqual(verdict[0], "FAIL")

    def test_missing_data_fails(self):
        verdict, _ = self.task._check_response_structure(SimpleNamespace(object="list"))
        self.assertEqual(verdict[0], "FAIL")

    def test_wrong_entry_object_fails(self):
        verdict, _ = self.task._check_response_structure(
            _embed_response([_embedding_entry([0.1], object_type="text")]))
        self.assertEqual(verdict[0], "FAIL")

    def test_empty_vector_fails(self):
        verdict, _ = self.task._check_response_structure(
            _embed_response([_embedding_entry([])]))
        self.assertEqual(verdict[0], "FAIL")

    def test_non_numeric_vector_fails(self):
        verdict, _ = self.task._check_response_structure(
            _embed_response([_embedding_entry(["a", "b"])]))
        self.assertEqual(verdict[0], "FAIL")


class TestRepeatability(unittest.TestCase):
    """E2 verdicts keyed to the anomalous-draw rate, not a single comparison."""

    def setUp(self):
        self.task = _make_task()
        self.n = self.task.REPEAT_COUNT
        self.all_stable = [[0.5, 0.5, 0.5]] * self.n

    def test_all_stable_draws_pass(self):
        with patch.object(self.task, "_embed",
                          side_effect=lambda _m, _t: list(self.all_stable.pop(0))):
            verdict, _ = self.task._check_repeatability("embed-gemma:300m")
        self.assertEqual(verdict[0], "PASS")

    def test_single_outlier_soft_fails(self):
        draws = [[0.5, 0.5, 0.5]] * (self.n - 1) + [[1.0, 0.0, 0.0]]

        def side_effect(_m, _t):
            return list(draws.pop(0))

        with patch.object(self.task, "_embed", side_effect=side_effect):
            (verdict, detail), _ = self.task._check_repeatability("embed-gemma:300m")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("flicker", detail)
        # cosine(0.5,0.5,0.5 ; 1,0,0) = 0.577… below the 0.999 stability line
        self.assertIn("0.577", detail)

    def test_repeated_outliers_fail(self):
        draws = [[1.0, 0.0, 0.0]] * 2 + [[0.5, 0.5, 0.5]] * (self.n - 2)

        def side_effect(_m, _t):
            return list(draws.pop(0))

        with patch.object(self.task, "_embed", side_effect=side_effect):
            (verdict, detail), _ = self.task._check_repeatability("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn("inconsistent", detail)


class TestCrossPathConsistency(unittest.TestCase):
    """E6: the same weights through two delivery paths must agree in one run."""

    def setUp(self):
        self.task = _make_task()

    def test_single_and_batch_paths_agree_pass(self):
        with patch.object(self.task, "_embed", return_value=[0.5, 0.5, 0.5]), \
             patch.object(self.task, "_embed_response",
                          return_value=_embed_response([_embedding_entry([0.5, 0.5, 0.5])])):
            verdict, _ = self.task._check_cross_path_consistency("embed-gemma:300m")
        self.assertEqual(verdict[0], "PASS")

    def test_disagreeing_paths_fail(self):
        with patch.object(self.task, "_embed", return_value=[0.5, 0.5, 0.5]), \
             patch.object(self.task, "_embed_response",
                          return_value=_embed_response([_embedding_entry([1.0, 0.0, 0.0])])):
            verdict, _ = self.task._check_cross_path_consistency("embed-gemma:300m")
        self.assertEqual(verdict[0], "FAIL")

    def test_empty_batch_data_fails(self):
        with patch.object(self.task, "_embed", return_value=[0.5, 0.5, 0.5]), \
             patch.object(self.task, "_embed_response",
                          return_value=_embed_response([])):
            (verdict, detail), vec = self.task._check_cross_path_consistency("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn("no data", detail)
        self.assertEqual(vec, [0.5, 0.5, 0.5])


class TestBatchReferenceConsistency(unittest.TestCase):
    """E7: per-draw cosine to the batch-path reference across a larger N."""

    def setUp(self):
        self.task = _make_task()

    def test_all_draws_agree_pass(self):
        with patch.object(self.task, "_embed", return_value=[0.5, 0.5, 0.5]), \
             patch.object(self.task, "_embed_response",
                          return_value=_embed_response([_embedding_entry([0.5, 0.5, 0.5])])):
            (verdict, _), vec = self.task._check_batch_reference_consistency("embed-gemma:300m")
        self.assertEqual(verdict, "PASS")
        self.assertEqual(vec, [0.5, 0.5, 0.5])
        self.assertEqual(len(self.task._reference_draws), self.task.REFERENCE_DRAW_COUNT)

    def test_sparse_outlier_soft_fails(self):
        n = self.task.REFERENCE_DRAW_COUNT
        draws = [[1.0, 0.0, 0.0]] + [[0.5, 0.5, 0.5]] * (n - 1)

        def side_effect(_m, _t):
            return list(draws.pop(0))

        with patch.object(self.task, "_embed", side_effect=side_effect), \
             patch.object(self.task, "_embed_response",
                          return_value=_embed_response([_embedding_entry([0.5, 0.5, 0.5])])):
            (verdict, detail), _ = self.task._check_batch_reference_consistency("embed-gemma:300m")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("1/30", detail)

    def test_majority_outliers_fail(self):
        n = self.task.REFERENCE_DRAW_COUNT
        draws = [[1.0, 0.0, 0.0]] * 8 + [[0.5, 0.5, 0.5]] * (n - 8)

        def side_effect(_m, _t):
            return list(draws.pop(0))

        with patch.object(self.task, "_embed", side_effect=side_effect), \
             patch.object(self.task, "_embed_response",
                          return_value=_embed_response([_embedding_entry([0.5, 0.5, 0.5])])):
            (verdict, detail), _ = self.task._check_batch_reference_consistency("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn("8/30", detail)

    def test_unavailable_reference_fails(self):
        with patch.object(self.task, "_embed", return_value=[0.5, 0.5, 0.5]), \
             patch.object(self.task, "_embed_response", return_value=_embed_response([])):
            (verdict, detail), _ = self.task._check_batch_reference_consistency("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn("reference", detail)

    def test_draw_rows_dump_full_raw_vectors(self):
        with patch.object(self.task, "_embed", return_value=[0.5, 0.5, 0.5]), \
             patch.object(self.task, "_embed_response",
                          return_value=_embed_response([_embedding_entry([0.5, 0.5, 0.5])])):
            self.task._check_batch_reference_consistency("embed-gemma:300m")
        rows = self.task._reference_draw_rows("embed-gemma:300m",
                                              "E7 Batch Reference Consistency")
        self.assertEqual(len(rows), self.task.REFERENCE_DRAW_COUNT)
        first = rows[0]
        self.assertEqual(first[1], "E7 Batch Reference Consistency (draw 1/30)")
        self.assertEqual(first[3], 3)
        self.assertEqual(json.loads(first[4]), [0.5, 0.5, 0.5])
        self.assertIn("cosine vs batch reference: 1.000000", first[5])

    def test_draw_rows_empty_before_any_check(self):
        rows = self.task._reference_draw_rows("embed-gemma:300m", "E7 Batch Reference Consistency")
        self.assertEqual(rows, [])


class TestBatchIntegrity(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()
        self.n = len(self.task.BATCH_INPUTS)
        self.vecs = [[float(i + 1)] * 4 for i in range(self.n)]

    def test_in_order_batch_passes(self):
        with patch.object(self.task, "_embed_response",
                          return_value=_embed_response(
                              [_embedding_entry(v, i) for i, v in enumerate(self.vecs)])):
            verdict, _ = self.task._check_batch_integrity("embed-gemma:300m")
        self.assertEqual(verdict[0], "PASS")

    def test_wrong_count_fails(self):
        fewer = self.vecs[:-1]
        with patch.object(self.task, "_embed_response",
                          return_value=_embed_response(
                              [_embedding_entry(v, i) for i, v in enumerate(fewer)])):
            verdict, _ = self.task._check_batch_integrity("embed-gemma:300m")
        self.assertEqual(verdict[0], "FAIL")

    def test_out_of_order_indexes_fail(self):
        entries = [_embedding_entry(v, (i + 1) % self.n) for i, v in enumerate(self.vecs)]
        with patch.object(self.task, "_embed_response", return_value=_embed_response(entries)):
            verdict, _ = self.task._check_batch_integrity("embed-gemma:300m")
        self.assertEqual(verdict[0], "FAIL")

    def test_empty_response_fails(self):
        with patch.object(self.task, "_embed_response", return_value=_embed_response([])):
            verdict, _ = self.task._check_batch_integrity("embed-gemma:300m")
        self.assertEqual(verdict[0], "FAIL")


class TestDimensionality(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_consistent_dimensions_pass(self):
        vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        with patch.object(self.task, "_embed_response",
                          return_value=_embed_response(
                              [_embedding_entry(v, i) for i, v in enumerate(vecs)])):
            verdict, _ = self.task._check_dimensionality("embed-gemma:300m")
        self.assertEqual(verdict[0], "PASS")

    def test_inconsistent_dimensions_fail(self):
        vecs = [[0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7]]
        with patch.object(self.task, "_embed_response",
                          return_value=_embed_response(
                              [_embedding_entry(v, i) for i, v in enumerate(vecs)])):
            verdict, _ = self.task._check_dimensionality("embed-gemma:300m")
        self.assertEqual(verdict[0], "FAIL")

    def test_empty_batch_fails(self):
        with patch.object(self.task, "_embed_response", return_value=_embed_response([])):
            verdict, _ = self.task._check_dimensionality("embed-gemma:300m")
        self.assertEqual(verdict[0], "FAIL")


class TestSemanticOrdering(unittest.TestCase):
    """Drives _check_semantic_ordering with a deterministic fake vector space."""

    def setUp(self):
        self.task = _make_task()

    VALID = {
        "cat": [1.0, 0.0, 0.0],
        "kitten": [0.9, 0.1, 0.0],
        "ocean": [0.0, 1.0, 0.0],
        "sea": [0.1, 0.9, 0.0],
        "car": [0.0, 0.0, 1.0],
        "desert": [0.0, 0.0, 0.9],
    }

    BROKEN = {
        "cat": [1.0, 0.0, 0.0],
        "kitten": [-1.0, 0.0, 0.0],
        "ocean": [0.0, 1.0, 0.0],
        "sea": [0.0, -1.0, 0.0],
        "car": [0.0, 0.0, 1.0],
        "desert": [0.0, 0.0, -1.0],
    }

    def test_related_closer_than_unrelated_passes(self):
        with patch.object(self.task, "_embed",
                          side_effect=lambda _m, _t: list(self.VALID[_t])):
            verdict, _ = self.task._check_semantic_ordering("embed-gemma:300m")
        self.assertEqual(verdict[0], "PASS")

    def test_related_farther_than_unrelated_fails(self):
        with patch.object(self.task, "_embed",
                          side_effect=lambda _m, _t: list(self.BROKEN[_t])):
            verdict, _ = self.task._check_semantic_ordering("embed-gemma:300m")
        self.assertEqual(verdict[0], "FAIL")


class TestRunCheckErrorHandling(unittest.TestCase):
    """A check that raises must be captured as an ERROR row, not propagate."""

    def test_exception_logged_as_error(self):
        task = _make_task()
        rows = []

        class FakeWriter:
            def writerow(self, values):
                rows.append(values)

        def exploding_check():
            raise RuntimeError("boom")

        with patch("oflm_test.tasks.time.sleep"):
            task._run_check(FakeWriter(), "embed-gemma:300m", "E1 Response Structure",
                            "input text", exploding_check)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "embed-gemma:300m")
        self.assertIn("ERROR: boom", rows[0][5])

    def test_verdict_tuple_written_with_detail(self):
        task = _make_task()
        rows = []

        class FakeWriter:
            def writerow(self, values):
                rows.append(values)

        with patch("oflm_test.tasks.time.sleep"):
            task._run_check(FakeWriter(), "embed-gemma:300m", "E3 Batch & Index Integrity",
                            json.dumps(task.BATCH_INPUTS),
                            lambda: (("PASS", "three embeddings in order"), [0.1, 0.2]))
        self.assertIn("PASS: three embeddings in order", rows[0][5])
        self.assertEqual(rows[0][3], 2)



class TestModelIdentity(unittest.TestCase):
    """E9 — the check that a substituted model cannot hide from.

    Every other check in this suite passes on an embedding for the wrong
    model: it is correctly shaped, correctly normed, deterministic, and
    semantically sensible. E9 is the only one that asks whether the server
    answered the question that was put to it.
    """

    def setUp(self):
        self.task = _make_task()

    def test_answering_an_impossible_model_fails(self):
        # The regression this check was written for: the server ignored the
        # `model` field and served whatever it had loaded, echoing the
        # requested tag back so the response looked correct.
        response = SimpleNamespace(
            data=[_embedding_entry([0.1, 0.2, 0.3])],
            object="list",
            model=EmbeddingTask.IMPOSSIBLE_MODEL,
        )
        with patch.object(EmbeddingTask, "_embed_response", return_value=response):
            (verdict, detail), vector = self.task._check_model_identity("bge-base:en-v1.5")
        self.assertEqual(verdict, "FAIL")
        self.assertIn(EmbeddingTask.IMPOSSIBLE_MODEL, detail)
        self.assertEqual(vector, [0.1, 0.2, 0.3])

    def test_refusing_then_serving_passes(self):
        calls = []

        def fake(self_, model_id, input_text):
            calls.append(model_id)
            if model_id == EmbeddingTask.IMPOSSIBLE_MODEL:
                raise RuntimeError("404 model not found")
            return SimpleNamespace(
                data=[_embedding_entry([1.0, 0.0])], object="list", model=model_id)

        with patch.object(EmbeddingTask, "_embed_response", fake):
            (verdict, detail), vector = self.task._check_model_identity("bge-base:en-v1.5")
        self.assertEqual(verdict, "PASS")
        self.assertEqual(calls, [EmbeddingTask.IMPOSSIBLE_MODEL, "bge-base:en-v1.5"])
        self.assertEqual(vector, [1.0, 0.0])

    def test_answering_with_an_empty_envelope_is_not_called_a_substitution(self):
        # A 2xx carrying an error instead of data is a contract problem of its
        # own, but it is not evidence of a substitution -- say the smaller
        # thing that is actually supported.
        response = SimpleNamespace(data=[], object="list", model=None)
        with patch.object(EmbeddingTask, "_embed_response", return_value=response):
            (verdict, detail), vector = self.task._check_model_identity("bge-base:en-v1.5")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("success envelope", detail)
        self.assertIsNone(vector)

    def test_response_naming_no_model_soft_fails(self):
        def fake(self_, model_id, input_text):
            if model_id == EmbeddingTask.IMPOSSIBLE_MODEL:
                raise RuntimeError("refused")
            return SimpleNamespace(data=[_embedding_entry([1.0])], object="list")

        with patch.object(EmbeddingTask, "_embed_response", fake):
            (verdict, _), _ = self.task._check_model_identity("bge-base:en-v1.5")
        self.assertEqual(verdict, "SOFT-FAIL")

    def test_response_naming_a_different_model_soft_fails(self):
        def fake(self_, model_id, input_text):
            if model_id == EmbeddingTask.IMPOSSIBLE_MODEL:
                raise RuntimeError("refused")
            return SimpleNamespace(data=[_embedding_entry([1.0])], object="list",
                                   model="embed-gemma:300m")

        with patch.object(EmbeddingTask, "_embed_response", fake):
            (verdict, detail), _ = self.task._check_model_identity("bge-base:en-v1.5")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("embed-gemma:300m", detail)


class TestReferenceAgreementScope(unittest.TestCase):
    """E8 knows which model its bundled vectors belong to."""

    def setUp(self):
        self.task = _make_task()

    def test_skips_for_a_model_with_no_bundled_reference(self):
        # Two models embed the same text into different spaces by design, so a
        # cosine between them says nothing about either. Reporting FAIL here
        # would be reporting a conclusion the data does not support.
        (verdict, detail), vector = self.task._check_reference_agreement("bge-base:en-v1.5")
        self.assertEqual(verdict, "SKIP")
        self.assertIn("bge-base:en-v1.5", detail)
        self.assertIsNone(vector)

    def test_still_runs_for_the_model_it_was_made_from(self):
        entries = self.task._reference_entries
        self.assertTrue(entries, "the bundled reference should not be empty")
        with patch.object(EmbeddingTask, "_embed",
                          side_effect=lambda m, t: dict(entries)[t]):
            (verdict, _), _ = self.task._check_reference_agreement("embed-gemma:300m")
        self.assertEqual(verdict, "PASS")


class TestRepeatabilityExactness(unittest.TestCase):
    """E2 reports whether a passing backend was bit-identical or merely close."""

    def setUp(self):
        self.task = _make_task()

    def test_identical_draws_are_reported_as_bit_identical(self):
        with patch.object(EmbeddingTask, "_embed",
                          side_effect=lambda m, t: [1.0, 2.0, 3.0]):
            (verdict, detail), _ = self.task._check_repeatability("m")
        self.assertEqual(verdict, "PASS")
        self.assertIn("bit-identical", detail)
        self.assertNotIn("not bit-identical", detail)

    def test_close_but_unequal_draws_still_pass_and_say_so(self):
        seq = iter(range(1000))

        def fake(self_, model_id, text):
            # A tiny perturbation: well inside STABILITY_THRESHOLD, not equal.
            return [1.0, 2.0, 3.0 + next(seq) * 1e-9]

        with patch.object(EmbeddingTask, "_embed", fake):
            (verdict, detail), _ = self.task._check_repeatability("m")
        self.assertEqual(verdict, "PASS")
        self.assertIn("not bit-identical", detail)


class TestTaskPromptHonoured(unittest.TestCase):
    """E10 - the check that would have caught the dropped task prompt.

    The handler passed task_query whatever the request said, so a document and
    a query embedded the same text to the same vector. That vector is correctly
    shaped, correctly normed and deterministic, so E1 through E9 pass on it.
    """

    def setUp(self):
        self.task = _make_task()

    def test_identical_vectors_under_both_prompts_fail_as_a_dropped_prompt(self):
        # The #52 defect itself: the same text under 'search_query' and under
        # 'search_document' comes back byte for byte the same.
        same = [0.1, 0.2, 0.3]
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_vector_body(same)],
            EmbeddingTask.DOCUMENT_PROMPT: [_vector_body(same)],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), vector = self.task._check_task_prompt_honoured("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn("task prompt is being dropped", detail)
        self.assertIn("1.000000", detail)
        self.assertEqual(vector, same)

    def test_different_vectors_with_stable_repeats_pass_and_report_both_cosines(self):
        query, document = [1.0, 0.0, 0.0], [0.9, 0.1, 0.0]
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_vector_body(query), _vector_body(query)],
            EmbeddingTask.DOCUMENT_PROMPT: [_vector_body(document)],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), vector = self.task._check_task_prompt_honoured("embed-gemma:300m")
        self.assertEqual(verdict, "PASS")
        across = self.task._cosine_similarity(query, document)
        self.assertIn(f"{across:.6f}", detail)      # the two prompts apart
        self.assertIn("1.000000", detail)           # two draws of the same prompt
        self.assertEqual(vector, query)

    def test_a_model_with_no_task_prompts_skips_rather_than_fails(self):
        # Refusing a prompt name is what a BERT-family model should do, and the
        # fix kept that on purpose, so it is not this suite's business to fail it.
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_error(400, NO_PROMPTS_MESSAGE)],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            verdict, vector = self.task._check_task_prompt_honoured("bge-base:en-v1.5")
        self.assertEqual(verdict[0], "SKIP")
        self.assertIn("bge-base:en-v1.5", verdict[1])
        self.assertIsNone(vector)

        # And a SKIP must not land in the run's failure count.
        self.task.record(verdict, "bge-base:en-v1.5 / E10 Task Prompt Honoured")
        self.assertEqual(self.task.result.hard_failures, 0)
        self.assertEqual(self.task.result.total, 0)

    def test_a_model_whose_prompts_serve_neither_task_skips_rather_than_fails(self):
        # The engine refuses to pick a prompt when none matches, and
        # src/open_npue_adapter/README.md calls that the correct behaviour. It is
        # the same deliberate refusal as the BERT one, so it gets the same verdict.
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_error(400, NO_MATCHING_PROMPT_MESSAGE)],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            verdict, vector = self.task._check_task_prompt_honoured("nomic-embed-text:v1.5")
        self.assertEqual(verdict[0], "SKIP")
        self.assertIn("none serves", verdict[1])
        self.assertIsNone(vector)

        self.task.record(verdict, "nomic-embed-text:v1.5 / E10 Task Prompt Honoured")
        self.assertEqual(self.task.result.hard_failures, 0)

    def test_a_repeat_draw_that_fails_is_reported_as_a_failed_draw(self):
        # Without this the cosine against a missing vector comes out 0.0 and the
        # run blames noise for a request that never came back.
        query, document = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_vector_body(query),
                                         _error(500, "Handler exception: engine reset")],
            EmbeddingTask.DOCUMENT_PROMPT: [_vector_body(document)],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), vector = self.task._check_task_prompt_honoured("embed-gemma:300m")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("repeat draw", detail)
        self.assertIn("HTTP 500", detail)
        self.assertNotIn("apart from noise", detail)
        self.assertEqual(vector, query)

    def test_a_refusal_for_any_other_reason_fails(self):
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_error(400, "model 'embed-gemma:300m' is not loaded",
                                                code="model_not_found", param="model")],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), _ = self.task._check_task_prompt_honoured("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn("HTTP 400", detail)
        self.assertIn("is not loaded", detail)

    def test_a_success_carrying_no_embedding_fails(self):
        fake = _fake_embed_raw({EmbeddingTask.QUERY_PROMPT: [_empty_body()]})
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), _ = self.task._check_task_prompt_honoured("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn("no embedding", detail)

    def test_the_document_prompt_being_refused_fails(self):
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_vector_body([1.0, 0.0, 0.0])],
            EmbeddingTask.DOCUMENT_PROMPT: [_error(400, "unknown prompt_name 'search_document'")],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), vector = self.task._check_task_prompt_honoured("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn(EmbeddingTask.DOCUMENT_PROMPT, detail)
        self.assertEqual(vector, [1.0, 0.0, 0.0])

    def test_vectors_that_also_move_between_draws_soft_fail_as_noise(self):
        # The prompts differ, but so do two draws of the same prompt, so the
        # difference is not evidence that the field was read at all.
        query, document = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        fake = _fake_embed_raw({
            EmbeddingTask.QUERY_PROMPT: [_vector_body(query), _vector_body([0.9, 0.1, 0.0])],
            EmbeddingTask.DOCUMENT_PROMPT: [_vector_body(document)],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), _ = self.task._check_task_prompt_honoured("embed-gemma:300m")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("apart from noise", detail)


class TestUnknownTaskPrompt(unittest.TestCase):
    """E11 - a task prompt the server cannot resolve must be refused.

    Quietly falling back to the default is the same defect one step quieter:
    the caller asked for one task, got another, and nothing says so.
    """

    def setUp(self):
        self.task = _make_task()

    def test_an_accepted_unknown_prompt_fails_as_a_silent_substitution(self):
        fake = _fake_embed_raw({
            EmbeddingTask.IMPOSSIBLE_PROMPT: [_vector_body([0.1, 0.2, 0.3])],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), vector = self.task._check_unknown_task_prompt("embed-gemma:300m")
        self.assertEqual(verdict, "FAIL")
        self.assertIn(EmbeddingTask.IMPOSSIBLE_PROMPT, detail)
        self.assertIn("nothing downstream can tell", detail)
        self.assertEqual(vector, [0.1, 0.2, 0.3])

    def test_a_well_formed_refusal_passes(self):
        fake = _fake_embed_raw({
            EmbeddingTask.IMPOSSIBLE_PROMPT: [
                _error(400, "unknown prompt_name 'not_a_task'. Known: [query, search_query, ...]"),
            ],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), vector = self.task._check_unknown_task_prompt("embed-gemma:300m")
        self.assertEqual(verdict, "PASS")
        self.assertIn("400", detail)
        self.assertIn("invalid_value", detail)
        self.assertIsNone(vector)

    def test_a_refusal_inside_a_success_envelope_soft_fails(self):
        fake = _fake_embed_raw({EmbeddingTask.IMPOSSIBLE_PROMPT: [_empty_body()]})
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), vector = self.task._check_unknown_task_prompt("embed-gemma:300m")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("success envelope", detail)
        self.assertIsNone(vector)

    def test_a_refusal_that_is_not_openai_shaped_soft_fails(self):
        # {"error": <string>} is the shape this server produced when its own
        # error handling threw; there is no code or type to report.
        bare = {"error": "Max length reached"}
        fake = _fake_embed_raw({
            EmbeddingTask.IMPOSSIBLE_PROMPT: [(400, bare, json.dumps(bare))],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), _ = self.task._check_unknown_task_prompt("embed-gemma:300m")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("not an", detail)
        self.assertIn("Max length reached", detail)

    def test_a_refusal_with_no_json_body_at_all_soft_fails(self):
        fake = _fake_embed_raw({
            EmbeddingTask.IMPOSSIBLE_PROMPT: [(400, None, "Bad Request")],
        })
        with patch.object(self.task, "_embed_raw", side_effect=fake):
            (verdict, detail), _ = self.task._check_unknown_task_prompt("embed-gemma:300m")
        self.assertEqual(verdict, "SOFT-FAIL")
        self.assertIn("Bad Request", detail)

if __name__ == "__main__":
    unittest.main(verbosity=2)