"""q4nx/uno.py's layout (Traces: OPEN-UNO-LORA): a misplaced tensor only drafts worse, and greedy Uno still matches decode."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q4nx.uno import TARGETS, derived, metadata, projection_folds  # noqa: E402

R, HID, QW, KVW, FF = 4, 512, 512, 128, 1024


def adapter(seed=0):
    g = torch.Generator().manual_seed(seed)
    sd, dims = {}, {"q_proj": (QW, HID), "k_proj": (KVW, HID), "v_proj": (KVW, HID), "o_proj": (HID, QW),
                    "gate_proj": (FF, HID), "up_proj": (FF, HID), "down_proj": (HID, FF)}
    for p in TARGETS:
        mod = "self_attn" if p.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
        out, inp = dims[p]
        sd[f"model.layers.0.{mod}.{p}.lora_A.weight"] = torch.randn(R, inp, generator=g)
        sd[f"model.layers.0.{mod}.{p}.lora_B.weight"] = torch.randn(out, R, generator=g)
    return sd


def ab(sd, p):
    mod = "self_attn" if p in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    pre = f"model.layers.0.{mod}.{p}"
    return sd[pre + ".lora_A.weight"].numpy(), sd[pre + ".lora_B.weight"].numpy()


class TestUnoLayout(unittest.TestCase):
    def setUp(self):
        self.sd = adapter()
        self.d = derived(self.sd, 0, scale=64.0, n_cores=8)

    def test_a_is_stacked_per_input_and_padded_to_a_band_per_core(self):
        rows = 8 * 64
        aq, ak, av = (ab(self.sd, p)[0] for p in ("q_proj", "k_proj", "v_proj"))
        want = np.zeros((rows, HID))
        want[:R], want[R:2 * R], want[256:256 + R] = aq, ak, av      # v's window starts at 256
        np.testing.assert_array_equal(self.d["a_qkv"], want)
        ag, au = ab(self.sd, "gate_proj")[0], ab(self.sd, "up_proj")[0]
        np.testing.assert_array_equal(self.d["a_gu"][:2 * R], np.concatenate([ag, au]))
        for n, cols in (("a_o", QW), ("a_d", FF)):
            self.assertEqual(self.d[n].shape, (rows, cols))
            self.assertFalse(self.d[n][R:].any())

    def test_b_reads_its_window_of_z_scaled(self):
        # q, k read z[0:256] = [z_q | z_k]; v reads z[256:512] = [z_v | 0]; gate, up read [z_g | z_u]
        for name, p, lo in (("b_q", "q_proj", 0), ("b_k", "k_proj", R), ("b_v", "v_proj", 0), ("b_o", "o_proj", 0),
                            ("b_g", "gate_proj", 0), ("b_u", "up_proj", R), ("b_d", "down_proj", 0)):
            b = ab(self.sd, p)[1]
            t = self.d[name]
            self.assertEqual(t.shape, (b.shape[0], 256), name)
            np.testing.assert_allclose(t[:, lo:lo + R], 64.0 * b, rtol=1e-6, err_msg=name)
            rest = np.delete(t, np.s_[lo:lo + R], axis=1)
            self.assertFalse(rest.any(), name)

    def test_the_layout_reproduces_the_adapter(self):
        # y += s * B (A x) for every target, through the padded tensors exactly as the GEMV reads them
        x = np.random.default_rng(1).standard_normal(HID)
        z = self.d["a_qkv"] @ x
        for name, p, window in (("b_q", "q_proj", z[:256]), ("b_k", "k_proj", z[:256]), ("b_v", "v_proj", z[256:512])):
            a, b = ab(self.sd, p)
            np.testing.assert_allclose(self.d[name] @ window, 64.0 * b @ (a @ x), rtol=1e-4, atol=1e-3, err_msg=name)

    def test_a_rank_that_does_not_fit_is_refused(self):
        sd = {k: torch.cat([v] * 50, 0 if "lora_A" in k else 1) for k, v in self.sd.items()}   # rank 200
        with self.assertRaises(ValueError):
            derived(sd, 0, 64.0)


class TestUnoFolds(unittest.TestCase):
    """A base whose builder folded multipliers into q / o / down needs the same factor on that projection's s B."""

    def test_granite_42_folds_only_q(self):
        cfg = {"head_dim": 64, "q4nx_folded_multipliers": {"attention_multiplier": 0.015625, "embedding_multiplier": 1.0,
                                                            "residual_multiplier": 1.0, "logits_scaling": 1.0}}
        self.assertEqual(projection_folds(cfg), {"q_proj": 0.125})

    def test_every_fold_lands_on_its_projection(self):
        cfg = {"head_dim": 64, "q4nx_folded_multipliers": {"attention_multiplier": 0.0078125, "embedding_multiplier": 12.0,
                                                            "residual_multiplier": 0.22, "logits_scaling": 8.0}}
        folds = projection_folds(cfg)
        self.assertEqual(folds, {"q_proj": 0.0625, "o_proj": 0.22, "down_proj": 0.22})
        sd = adapter()
        d = derived(sd, 0, scale=64.0, n_cores=8, folds=folds)
        for name, p, lo in (("b_q", "q_proj", 0), ("b_k", "k_proj", R), ("b_o", "o_proj", 0), ("b_d", "down_proj", 0),
                            ("b_u", "up_proj", R)):
            np.testing.assert_allclose(d[name][:, lo:lo + R], 64.0 * folds.get(p, 1.0) * ab(sd, p)[1], rtol=1e-6,
                                       err_msg=name)

    def test_a_base_without_recorded_folds_is_unchanged(self):
        self.assertEqual(projection_folds({"num_hidden_layers": 36}), {})
        sd = adapter()
        plain, folded = derived(sd, 0, 64.0), derived(sd, 0, 64.0, folds=projection_folds({}))
        for name in plain:
            np.testing.assert_array_equal(plain[name], folded[name], err_msg=name)

    def test_a_missing_attention_scale_means_no_q_fold(self):
        cfg = {"head_dim": 128, "q4nx_folded_multipliers": {"attention_multiplier": None, "residual_multiplier": None}}
        self.assertEqual(projection_folds(cfg), {})


class TestUnoMetadata(unittest.TestCase):
    ACFG = {"r": 128, "lora_alpha": 8192, "base_model_name_or_path": "ibm-granite/granite-4.2-3b"}

    def test_the_noise_bound_is_recorded_only_when_given(self):
        self.assertEqual(metadata(self.ACFG, 64.0, {}, 100256)["uno_noise_high"], "100256")
        self.assertNotIn("uno_noise_high", metadata(self.ACFG, 64.0, {}, None))

    def test_an_empty_noise_range_is_refused(self):
        with self.assertRaises(ValueError):
            metadata(self.ACFG, 64.0, {}, 1)

    def test_the_folds_are_recorded(self):
        self.assertEqual(metadata(self.ACFG, 64.0, {"q_proj": 0.125}, None)["uno_folds"], '{"q_proj": 0.125}')


if __name__ == "__main__":
    unittest.main()
