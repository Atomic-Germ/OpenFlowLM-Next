"""Granite's four multipliers: which tensors they land on, and that the fold is exact.

The claim the whole open-kernels Granite path rests on is that all four fold
losslessly into an already-quantized tensor. `open_kernels/recipes/spec.py`
refuses any container whose `attention_multiplier` is not `head_dim ** -0.5`,
so if the fold were approximate the refusal would be guarding nothing.
"""
import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.constants import ModelArch, ModelArchConfigs, ModelArchNames  # noqa: E402
from q4nx.models.granite import (  # noqa: E402
    fold_factor_for,
    fold_factors,
    scale_unpacked,
)

HD = 64                     # granite-4.2-3b
ATTN_MULT = 0.015625        # its only non-unit multiplier


class FoldFactorsTest(unittest.TestCase):
    def test_the_3b_folds_one_tensor_and_lands_on_the_hard_coded_scale(self):
        folds = fold_factors(ATTN_MULT, 1.0, 1.0, 1.0, HD)
        self.assertEqual(folds, {"self_attn.q_proj.weight": 0.125})
        # 0.125 is head_dim ** -0.5 exactly, which is what attn.h applies -- and
        # a power of two, so scaling d/m by it is an exponent shift.
        self.assertEqual(folds["self_attn.q_proj.weight"], HD ** -0.5)

    def test_all_unit_multipliers_touch_nothing(self):
        self.assertEqual(fold_factors(HD ** -0.5, 1.0, 1.0, 1.0, HD), {})

    def test_each_multiplier_reaches_the_tensors_it_belongs_to(self):
        folds = fold_factors(HD ** -0.5, 2.0, 3.0, 4.0, HD)
        self.assertEqual(folds, {
            "model.embed_tokens.weight": 2.0,
            "self_attn.o_proj.weight": 3.0,     # residual_multiplier: BOTH blocks
            "mlp.down_proj.weight": 3.0,
            "lm_head.weight": 0.25,             # logits_scaling divides
        })

    def test_the_factor_is_matched_on_a_suffix_so_it_holds_across_layers(self):
        folds = fold_factors(ATTN_MULT, 1.0, 1.0, 1.0, HD)
        for layer in (0, 17, 39):
            name = f"model.layers.{layer}.self_attn.q_proj.weight"
            self.assertEqual(fold_factor_for(name, folds), 0.125)
        self.assertIsNone(
            fold_factor_for("model.layers.0.self_attn.k_proj.weight", folds))


class ScaleUnpackedTest(unittest.TestCase):
    """The lossless claim: scaling (d, m) equals quantizing c*W, code for code."""

    def _block(self):
        torch.manual_seed(0)
        d = torch.rand(4, 2) + 0.5          # per-block scale
        m = torch.rand(4, 2) - 0.5          # per-block minimum
        qs = torch.randint(0, 16, (4, 64))  # the 4-bit codes
        return d, m, qs

    @staticmethod
    def _dequant(d, m, qs):
        # w = code * d + m, with one (d, m) per 32-wide group
        return qs.float() * d.repeat_interleave(32, dim=1) + m.repeat_interleave(32, dim=1)

    def test_the_codes_do_not_move(self):
        d, m, qs = self._block()
        d2, m2, qs2 = scale_unpacked((d, m, qs), 0.125)
        self.assertTrue(torch.equal(qs, qs2))

    def test_dequantizes_to_exactly_c_times_the_original(self):
        d, m, qs = self._block()
        before = self._dequant(d, m, qs)
        after = self._dequant(*scale_unpacked((d, m, qs), 0.125))
        # 0.125 is a power of two, so this is bit-exact, not close.
        self.assertTrue(torch.equal(after, before * 0.125))

    def test_a_non_power_of_two_is_still_exact_in_the_codes(self):
        d, m, qs = self._block()
        d2, m2, qs2 = scale_unpacked((d, m, qs), 0.3)
        self.assertTrue(torch.equal(qs, qs2))
        self.assertTrue(torch.allclose(self._dequant(d2, m2, qs2),
                                       self._dequant(d, m, qs) * 0.3, atol=0, rtol=1e-6))

    def test_a_float_passthrough_scales_its_values(self):
        w = torch.arange(8.0)
        (out,) = scale_unpacked((w,), 0.5)
        self.assertTrue(torch.equal(out, w * 0.5))

    def test_an_unexpected_arity_is_refused(self):
        with self.assertRaises(ValueError):
            scale_unpacked((torch.zeros(2), torch.zeros(2)), 2.0)


class RegistrationTest(unittest.TestCase):
    def test_the_arch_is_routed_and_has_a_name_map(self):
        self.assertIn("granite", ModelArchNames[ModelArch.GRANITE])
        cfg = Path(__file__).resolve().parents[1] / "configs" / ModelArchConfigs[ModelArch.GRANITE]
        self.assertTrue(cfg.is_file(), f"{cfg} is missing")
        name_map = json.load(open(cfg, encoding="utf-8"))["name_map"]
        # Granite's GGUF tensor set is Llama's dense subset; every tensor the
        # folds name has to be reachable through the map.
        q4nx_names = {v["q4nx_name"] for v in name_map.values()}
        for suffix in ("self_attn.q_proj.weight", "self_attn.o_proj.weight",
                       "mlp.down_proj.weight", "model.embed_tokens.weight",
                       "lm_head.weight"):
            self.assertTrue(any(n.endswith(suffix) for n in q4nx_names),
                            f"no tensor in granite.json ends with {suffix}")


class ConfigRewriteTest(unittest.TestCase):
    """The deployed config.json must state the multipliers AFTER the fold."""

    class _Field:
        def __init__(self, v):
            self._v = v

        def contents(self):
            return self._v

    class _Reader:
        def __init__(self, fields):
            self.fields = fields

    def test_it_writes_the_scale_the_kernels_hard_code(self):
        from q4nx.model_assets import apply_granite_fold_to_config

        reader = self._Reader({
            "granite.attention.head_count": self._Field(40),
            "granite.embedding_length": self._Field(2560),
            "granite.rope.dimension_count": self._Field(64),
            "granite.attention.scale": self._Field(ATTN_MULT),
        })
        cfg = apply_granite_fold_to_config({}, reader)
        self.assertEqual(cfg["attention_multiplier"], HD ** -0.5)
        self.assertEqual(cfg["embedding_multiplier"], 1.0)
        self.assertEqual(cfg["residual_multiplier"], 1.0)
        self.assertEqual(cfg["logits_scaling"], 1.0)
        self.assertEqual(cfg["head_dim"], 64)
        # and it still says what it came from
        self.assertEqual(cfg["q4nx_folded_multipliers"]["attention_multiplier"], ATTN_MULT)

    def test_head_dim_falls_back_to_embedding_over_heads(self):
        from q4nx.model_assets import apply_granite_fold_to_config

        reader = self._Reader({
            "granite.attention.head_count": self._Field(40),
            "granite.embedding_length": self._Field(2560),
        })
        cfg = apply_granite_fold_to_config({}, reader)
        self.assertEqual(cfg["head_dim"], 64)

    def test_it_refuses_rather_than_guess_a_head_dim(self):
        from q4nx.model_assets import apply_granite_fold_to_config

        with self.assertRaises(ValueError):
            apply_granite_fold_to_config({}, self._Reader({}))


if __name__ == "__main__":
    unittest.main()
