"""Tests for the GGUF metadata inspector (Phase 0 baseline).

Run with:
    python3 utilities/gguf-utils/gguf-inspect/tests/test_inspect.py
    pytest utilities/gguf-utils/gguf-inspect/tests/

These exercise real, multi-gigabyte model snapshots, which are not in the repo.
Their locations are therefore resolved from the local Hugging Face cache and can
be overridden per file with an environment variable:

    GGUF_TEST_QWEN     -- Qwen3.8-9B-Distill-GGUF Q6_K trunk model
    GGUF_TEST_MMPROJ   -- the same repo's mmproj-F16.gguf
    GGUF_TEST_LLAMA    -- Llama-3.2-1B-Instruct-GGUF Q8_0 model

Every test that needs a snapshot skips when it is absent, so the suite is green
on CI, where no model is cached. (It previously hard-coded one developer's home
directory, so on CI every functional case silently skipped while
test_provenance -- the one test with no guard -- failed on the missing path.)
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gguf_inspect import inspect, GGUFReport  # noqa: E402

# Revisions are pinned so the baseline cannot silently drift.
_QWEN_REPO = "Atomic-Germ/Qwen3.8-9B-Distill-GGUF"
_QWEN_REV = "c43a63ce052c5a4906a20335ebf519907ea01ec5"
_LLAMA_REPO = "Atomic-Germ/Llama-3.2-1B-Instruct-GGUF"
_LLAMA_REV = "2baaed77eb97af7ec61f344afa096b90643e922c"


def _hf_cache_roots() -> list[Path]:
    roots = []
    if e := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(e))
    if e := os.environ.get("HF_HOME"):
        roots.append(Path(e) / "hub")
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        roots.append(Path(home) / ".cache" / "huggingface" / "hub")
    return roots


def _cached(repo: str, rev: str, filename: str) -> Path | None:
    dirname = "models--" + repo.replace("/", "--")
    for root in _hf_cache_roots():
        p = root / dirname / "snapshots" / rev / filename
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _snapshot(env_var: str, repo: str, rev: str, filename: str):
    """A pinned snapshot path: the env override, else the local HF cache."""
    if override := os.environ.get(env_var):
        p = Path(override)
        return str(p) if p.is_file() and p.stat().st_size > 0 else None
    found = _cached(repo, rev, filename)
    return str(found) if found else None


QWEN = _snapshot("GGUF_TEST_QWEN", _QWEN_REPO, _QWEN_REV, "Qwen3.8-9B-Q6_K.gguf")
LLAMA = _snapshot("GGUF_TEST_LLAMA", _LLAMA_REPO, _LLAMA_REV,
                  "Llama-3.2-1B-Instruct-Q8_0.gguf")
MMPROJ = _snapshot("GGUF_TEST_MMPROJ", _QWEN_REPO, _QWEN_REV, "mmproj-F16.gguf")


def _exists(p: str | None) -> bool:
    return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0


class GGUFInspectorTests(unittest.TestCase):
    def _require(self, path, what):
        # The single guard every snapshot test uses, so a missing model skips
        # rather than failing.
        if not _exists(path):
            self.skipTest(f"{what} snapshot not available")
        return path

    def test_qwen_trunk_and_mtp(self):
        rep = inspect(self._require(QWEN, "Qwen Q6_K"))
        self.assertEqual(rep.architecture, "qwen35")
        self.assertEqual(rep.blocks.block_count, 33)
        self.assertEqual(rep.blocks.trunk_layers, 32)
        self.assertTrue(rep.blocks.has_mtp)
        self.assertIn(1, (rep.blocks.nextn_predict_layers,))

    def test_qwen_compatible_projector(self):
        rep = inspect(self._require(MMPROJ, "Qwen mmproj"))
        self.assertTrue(rep.projector.is_clip)
        self.assertIn("Qwen3.5", rep.projector.base_model)

    def test_llama_tensor_rope(self):
        rep = inspect(self._require(LLAMA, "Llama Q8_0"))
        self.assertEqual(rep.architecture, "llama")
        self.assertTrue(rep.rope.carried)
        self.assertIn("rope_freqs.weight", rep.rope.tensor_names)
        self.assertFalse(rep.blocks.has_mtp)

    def test_tokenizer_template_present(self):
        rep = inspect(self._require(LLAMA, "Llama Q8_0"))
        self.assertTrue(rep.tokenizer.has_chat_template)
        self.assertEqual(rep.tokenizer.pre, "llama-bpe")

    def test_unsupported_requirements(self):
        rep = inspect(self._require(QWEN, "Qwen Q6_K"))
        joined = " ".join(rep.unsupported.requirements)
        self.assertIn("speculative", joined.lower())

    def test_provenance(self):
        # The one test that used to run unguarded against a hard-coded path.
        rep = inspect(self._require(QWEN, "Qwen Q6_K"))
        self.assertEqual(rep.snapshot, "c43a63ce052c5a4906a20335ebf519907ea01ec5")

    def test_no_npu_no_payload_decode(self):
        # Metadata-only: inspecting a large file must not decode any tensor
        # payload (which would take many seconds to minutes). Warm the reader up
        # first so cold-import variance does not make the assertion flaky.
        path = self._require(QWEN, "Qwen Q6_K")
        inspect(path)  # warm
        start = time.perf_counter()
        inspect(path)
        self.assertLess(time.perf_counter() - start, 15.0)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
