"""
Unit tests for EmbeddingTask — embedding checks E1-E7. All embedding API calls
are mocked so no FLM server is required.

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

from flm_test.tasks import EmbeddingTask


def _make_task():
    """Instantiate an EmbeddingTask with all server I/O patched away."""
    with patch.object(EmbeddingTask, "_get_flm_version", return_value="0.9.99"), \
         patch.object(EmbeddingTask, "_fetch_all_models", return_value=[]), \
         patch("os.makedirs"):
        return EmbeddingTask(base_url="http://127.0.0.1:52625/v1", backend_os="linux")


def _embedding_entry(vector, index=0, object_type="embedding"):
    return SimpleNamespace(embedding=vector, index=index, object=object_type)


def _embed_response(data, object_type="list"):
    return SimpleNamespace(data=data, object=object_type)


class TestCheckNames(unittest.TestCase):

    def setUp(self):
        self.task = _make_task()

    def test_eight_checks_defined(self):
        self.assertEqual(len(self.task.CHECK_NAMES), 8)
        for name in self.task.CHECK_NAMES:
            self.assertIn(name[0], "E12345678")

    def test_embed_allowlist_present(self):
        self.assertIn("embed-gemma:300m", EmbeddingTask.EMBED_MODELS)


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

        with patch("flm_test.tasks.time.sleep"):
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

        with patch("flm_test.tasks.time.sleep"):
            task._run_check(FakeWriter(), "embed-gemma:300m", "E3 Batch & Index Integrity",
                            json.dumps(task.BATCH_INPUTS),
                            lambda: (("PASS", "three embeddings in order"), [0.1, 0.2]))
        self.assertIn("PASS: three embeddings in order", rows[0][5])
        self.assertEqual(rows[0][3], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)