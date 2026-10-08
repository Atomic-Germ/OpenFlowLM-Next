"""K2-Horizon converter plumbing (Stage 1.8): the arch resolution, the config and the
rope-dimension hook -- no weights, the dry-run half of the converter gate. The Q4_1
target is an audited fact, not an analogy: every dense family config in this tree
defaults to it, Q4_K is what OFLM 1.0.3+ requires for the 35B MoE projections
(cli.py --quant), and the open kernels' pool transcodes a q4_k container to q4_1
anyway (recipes/pack.py q4k_to_q4_1). Source GGUF quant, converter target, Q4NX
storage and kernel compute stay separate: whatever the source holds, the llama-family
path unpacks to the config's target before packing.
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.constants import ModelArch, ModelArchConfigs, ModelArchNames  # noqa: E402
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
    def __init__(self, architecture=None, **fields):
        self.fields = {}
        if architecture:
            f = _field(raw=architecture)
            f.name = "general.architecture"
            self.fields[f.name] = f
        for key, value in fields.items():
            f = _field(value=value)
            f.name = key.replace("__", ".")
            self.fields[f.name] = f

    def get(self, key):
        return self.fields.get(key)


class TestK2ArchResolution(unittest.TestCase):
    def test_the_enum_names_and_config_are_registered(self):
        self.assertIn(ModelArch.K2, ModelArchNames)
        self.assertEqual(set(ModelArchNames[ModelArch.K2]), {"k2_horizon", "k2-horizon", "k2"})
        self.assertEqual(ModelArchConfigs[ModelArch.K2], "k2.json")

    def test_a_k2_horizon_gguf_resolves_without_the_override(self):
        self.assertIs(get_model_arch_from_gguf(FakeReader("k2_horizon")), ModelArch.K2)
        self.assertIs(get_model_arch_from_gguf(FakeReader("K2-Horizon")), ModelArch.K2)

    def test_f_llama_still_short_circuits_the_arch_string(self):
        # the explicit mapping is the supported mechanism; -f ignores the GGUF's own tag
        self.assertIs(get_model_arch_from_gguf(FakeReader("k2_horizon"), "llama"), ModelArch.LLAMA)
        self.assertIs(get_model_arch_from_gguf(FakeReader("llama"), "k2"), ModelArch.K2)

    def test_k2_json_is_llamas_tiling_at_q4_1(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "k2.json").read_text())
        self.assertEqual(cfg["q4nx_config"],
                         {"row_block_size": 32, "col_block_size": 256,
                          "parallel_size": 16, "keep_block_in_2D": False})
        self.assertEqual(cfg["default_tensor_type"], "Q4_1")
        llama = json.loads((Path(__file__).resolve().parents[1] / "configs" / "llama.json").read_text())
        self.assertEqual(cfg["name_map"], llama["name_map"], "k2's tensor names are the llama ones")
        self.assertIn("lm_head", cfg["name_map"])               # untied: output.weight maps to it
        self.assertIn("rope_freqs", cfg["name_map"])           # GGUF-only; the converter skips it


class TestK2QkRowOrder(unittest.TestCase):
    """A k2-horizon GGUF keeps HF's split-half q/k rows; only a llama-arch one is
    interleaved and needs Llama's reorder (Traces: OPEN-FAMILY-K2)."""

    def test_k2_horizon_gguf_rows_are_kept(self):
        import q4nx.models as M
        k2 = M.K2.__new__(M.K2)
        k2.gguf_reader = FakeReader("k2-horizon")
        self.assertFalse(k2._gguf_qk_interleaved())

    def test_llama_arch_rows_are_reordered(self):
        import q4nx.models as M
        k2 = M.K2.__new__(M.K2)
        k2.gguf_reader = FakeReader("llama")
        self.assertTrue(k2._gguf_qk_interleaved())
        llama = M.Llama.__new__(M.Llama)
        llama.gguf_reader = FakeReader("k2-horizon")
        self.assertTrue(llama._gguf_qk_interleaved())     # -f llama keeps llama's rule


class TestK2ConverterClass(unittest.TestCase):
    def test_the_class_registers_and_the_rope_hook_reads_the_own_prefix(self):
        import q4nx.models as M                                    # registers every class
        from q4nx.model_converter import _MODEL_REGISTRY
        self.assertIs(_MODEL_REGISTRY[ModelArch.K2], M.K2)
        self.assertTrue(issubclass(M.K2, M.Llama))

        k2 = M.K2.__new__(M.K2)                                    # no reader/source needed
        k2.gguf_reader = FakeReader("k2_horizon",
                                    **{"k2_horizon__rope__dimension_count": 128})
        self.assertEqual(k2._rope_dim_count(), 128)

        # a llama-arch GGUF (what -f llama converts from) keeps working through the hook
        llama_arch = M.Llama.__new__(M.Llama)
        llama_arch.gguf_reader = FakeReader("llama", **{"llama__rope__dimension_count": 128})
        self.assertEqual(llama_arch._rope_dim_count(), 128)

        # and the K2 hook finds it under whichever prefix the file actually uses
        k2b = M.K2.__new__(M.K2)
        k2b.gguf_reader = FakeReader("llama", **{"llama__rope__dimension_count": 128})
        self.assertEqual(k2b._rope_dim_count(), 128)

        with self.assertRaises(KeyError):
            M.K2.__new__(M.K2).__init__ if False else None or _no_rope(k2)


def _no_rope(k2):
    k2.gguf_reader = FakeReader("k2_horizon")
    return k2._rope_dim_count()


if __name__ == "__main__":
    unittest.main()
