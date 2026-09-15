"""Tests for the GGUF metadata inspector (Phase 0 baseline).

Run with:
    python3 utilities/gguf-inspect/tests/test_inspect.py
    pytest utilities/gguf-inspect/tests/
"""

from __future__ import annotations

import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gguf_inspect import inspect, GGUFReport  # noqa: E402

# HF snapshots are pinned so the baseline cannot silently drift.
QWEN = ("/home/atomic-germ/.cache/huggingface/hub/models--Atomic-Germ--"
        "Qwen3.8-9B-Distill-GGUF/snapshots/"
        "c43a63ce052c5a4906a20335ebf519907ea01ec5/"
        "Qwen3.8-9B-Q6_K.gguf")
LLAMA = ("/home/atomic-germ/.cache/huggingface/hub/models--Atomic-Germ--"
         "Llama-3.2-1B-Instruct-GGUF/snapshots/"
         "2baaed77eb97af7ec61f344afa096b90643e922c/"
         "Llama-3.2-1B-Instruct-Q8_0.gguf")
MMPROJ = ("/home/atomic-germ/.cache/huggingface/hub/models--Atomic-Germ--"
          "Qwen3.8-9B-Distill-GGUF/snapshots/"
          "c43a63ce052c5a4906a20335ebf519907ea01ec5/"
          "mmproj-F16.gguf")


def _exists(p: str) -> bool:
    return os.path.isfile(p) and os.path.getsize(p) > 0


class GGUFInspectorTests(unittest.TestCase):
    def test_qwen_trunk_and_mtp(self):
        if not _exists(QWEN):
            self.skipTest("Qwen Q6_K snapshot not cached")
        rep = inspect(QWEN)
        self.assertEqual(rep.architecture, "qwen35")
        self.assertEqual(rep.blocks.block_count, 33)
        self.assertEqual(rep.blocks.trunk_layers, 32)
        self.assertTrue(rep.blocks.has_mtp)
        self.assertIn(1, (rep.blocks.nextn_predict_layers,))

    def test_qwen_compatible_projector(self):
        if not _exists(MMPROJ):
            self.skipTest("mmproj snapshot not cached")
        rep = inspect(MMPROJ)
        self.assertTrue(rep.projector.is_clip)
        self.assertIn("Qwen3.5", rep.projector.base_model)

    def test_llama_tensor_rope(self):
        if not _exists(LLAMA):
            self.skipTest("Llama Q8_0 snapshot not cached")
        rep = inspect(LLAMA)
        self.assertEqual(rep.architecture, "llama")
        self.assertTrue(rep.rope.carried)
        self.assertIn("rope_freqs.weight", rep.rope.tensor_names)
        self.assertFalse(rep.blocks.has_mtp)

    def test_tokenizer_template_present(self):
        if not _exists(LLAMA):
            self.skipTest("Llama Q8_0 snapshot not cached")
        rep = inspect(LLAMA)
        self.assertTrue(rep.tokenizer.has_chat_template)
        self.assertEqual(rep.tokenizer.pre, "llama-bpe")

    def test_unsupported_requirements(self):
        if not _exists(QWEN):
            self.skipTest("Qwen Q6_K snapshot not cached")
        rep = inspect(QWEN)
        joined = " ".join(rep.unsupported.requirements)
        self.assertIn("speculative", joined.lower())

    def test_provenance(self):
        rep = inspect(QWEN)
        self.assertEqual(rep.snapshot, "c43a63ce052c5a4906a20335ebf519907ea01ec5")

    def test_no_npu_no_payload_decode(self):
        # Metadata-only: inspecting a large file must not decode any tensor
        # payload (which would take many seconds to minutes). Warm the reader up
        # first so cold-import variance does not make the assertion flaky.
        if not _exists(QWEN):
            self.skipTest("Qwen Q6_K snapshot not cached")
        inspect(QWEN)  # warm
        start = time.perf_counter()
        inspect(QWEN)
        self.assertLess(time.perf_counter() - start, 15.0)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
