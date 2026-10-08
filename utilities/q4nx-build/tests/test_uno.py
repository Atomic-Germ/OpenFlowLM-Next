"""K2-Horizon-7B-Uno's adapter shaped for the L-row GEMV (q4nx/uno.py; Traces: OPEN-UNO-LORA).

A wrong placement does not crash: the draft pass would simply draft worse, and greedy Uno
would still match plain decode -- slower. So the layout is asserted here, on a synthetic
adapter small enough to check by hand."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q4nx.uno import TARGETS, derived  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
