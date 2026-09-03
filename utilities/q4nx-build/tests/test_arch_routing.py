"""Tests for -f override routing (family pins, nearest-dim variant inference)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.arch_detect import (  # noqa: E402
    override_resolves_exactly,
    resolve_override_candidates,
)
from q4nx.constants import ModelArch, nearest_qwen35_variant  # noqa: E402
from q4nx.model_converter import get_model_arch_from_gguf  # noqa: E402


def _field(value=None, raw: str = None):
    """GGUFReader field stand-in: raw string access + optional .contents()."""
    return SimpleNamespace(
        parts=[(raw or "").encode()],
        data=[0] if raw is not None else [],
        contents=(lambda: value) if value is not None else (lambda: None),
        name="",
    )


class FakeReader:
    def __init__(self, architecture=None, embedding_length=None):
        self.fields = {}
        if architecture:
            f = _field(raw=architecture)
            f.name = "general.architecture"
            self.fields[f.name] = f
        if embedding_length is not None:
            f = _field(value=embedding_length)
            f.name = "qwen35.embedding_length"
            self.fields[f.name] = f

    def get(self, key):
        return self.fields.get(key)


class OverrideExactnessTest(unittest.TestCase):
    def test_exact_names(self):
        for s in ("qwen3", "qwen3.5-2b", "qwen35-4B", "gemma3", "llama", "lfm2"):
            with self.subTest(s=s):
                self.assertTrue(override_resolves_exactly(s))

    def test_size_suffix_boundary(self):
        self.assertTrue(override_resolves_exactly("gemma4-e2b"))
        self.assertTrue(override_resolves_exactly("qwen3.5-9b"))

    def test_family_prefix_is_not_exact(self):
        # 'qwen3' is a prefix of 'qwen3.5': must NOT collapse to plain qwen3.
        self.assertFalse(override_resolves_exactly("qwen3.5"))
        self.assertFalse(override_resolves_exactly("qwen3.8"))

    def test_unknown_strings(self):
        self.assertFalse(override_resolves_exactly("phi4"))  # names list says phi3
        self.assertFalse(override_resolves_exactly(""))


class ResolveOverrideCandidatesTest(unittest.TestCase):
    def test_qwen35_pins_all_variants_but_not_qwen3(self):
        cands = resolve_override_candidates("qwen3.5")
        self.assertIsNotNone(cands)
        self.assertIn(ModelArch.QWEN35_2B, cands)
        self.assertIn(ModelArch.QWEN35MOE, cands)
        self.assertNotIn(ModelArch.QWEN3, cands)

    def test_dotless_spelling(self):
        self.assertIn(ModelArch.QWEN35_4B, resolve_override_candidates("qwen35"))

    def test_exact_or_unknown_gives_none(self):
        self.assertIsNone(resolve_override_candidates("qwen3.5-2b"))
        self.assertIsNone(resolve_override_candidates(""))
        self.assertIsNone(resolve_override_candidates("nosucharch"))


class NearestQwen35VariantTest(unittest.TestCase):
    def test_known_dims_map_exactly(self):
        for dim, expected in ((1024, ModelArch.QWEN35_08B), (2048, ModelArch.QWEN35_2B),
                              (2560, ModelArch.QWEN35_4B), (4096, ModelArch.QWEN35_9B)):
            with self.subTest(dim=dim):
                arch, delta = nearest_qwen35_variant(dim)
                self.assertEqual(arch, expected)
                self.assertEqual(delta, 0)

    def test_community_width_snaps_to_closest(self):
        arch, delta = nearest_qwen35_variant(2176)
        self.assertEqual(arch, ModelArch.QWEN35_2B)
        self.assertEqual(delta, 128)


class GetModelArchFromGgufTest(unittest.TestCase):
    def test_qwen35_unknown_dim_nearest_variant(self):
        reader = FakeReader(architecture="qwen35", embedding_length=2048)
        self.assertEqual(
            get_model_arch_from_gguf(reader), ModelArch.QWEN35_2B
        )

    def test_family_override_with_unknown_dim(self):
        # The failing user command shape: -f qwen3.5 on a qwen35 GGUF.
        reader = FakeReader(architecture="qwen35", embedding_length=2048)
        self.assertEqual(
            get_model_arch_from_gguf(reader, "qwen3.5"), ModelArch.QWEN35_2B
        )

    def test_family_override_with_known_dim(self):
        reader = FakeReader(architecture="qwen35", embedding_length=2560)
        self.assertEqual(
            get_model_arch_from_gguf(reader, "qwen3.5"), ModelArch.QWEN35_4B
        )

    def test_exact_override_still_wins(self):
        reader = FakeReader(architecture="qwen35", embedding_length=2560)
        self.assertEqual(
            get_model_arch_from_gguf(reader, "qwen3.5-9b"), ModelArch.QWEN35_9B
        )

    def test_explicit_plain_qwen3_still_respects_user(self):
        reader = FakeReader(architecture="qwen3", embedding_length=2560)
        self.assertEqual(get_model_arch_from_gguf(reader, "qwen3"), ModelArch.QWEN3)

    def test_known_dim_without_override(self):
        reader = FakeReader(architecture="qwen35", embedding_length=4096)
        self.assertEqual(get_model_arch_from_gguf(reader), ModelArch.QWEN35_9B)


if __name__ == "__main__":
    unittest.main()
