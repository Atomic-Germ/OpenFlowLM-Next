"""Tests for --pad-to-fit hidden-axis padding (qwen35 GGUF path)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.gguf_tensor import GGUFTensor  # noqa: E402
from q4nx.models.qwen35 import (  # noqa: E402
    PAD_HIDDEN_INPUT_SUFFIXES,
    PAD_HIDDEN_NORM_SUFFIXES,
    PAD_HIDDEN_OUTPUT_SUFFIXES,
    pad_hidden_axis,
    pad_hidden_axis_t,
)


class PadHiddenAxisTest(unittest.TestCase):
    def test_pads_columns(self):
        w = torch.ones((4, 8))
        padded = pad_hidden_axis(w, actual=8, target=12)
        self.assertEqual(padded.shape, (4, 12))
        self.assertTrue(torch.equal(padded[:, :8], w))
        self.assertTrue(torch.equal(padded[:, 8:], torch.zeros((4, 4))))

    def test_pads_norm_length(self):
        w = torch.ones(8)
        padded = pad_hidden_axis(w, actual=8, target=12)
        self.assertEqual(padded.shape, (12,))
        self.assertTrue(torch.equal(padded[:8], w))
        self.assertTrue(torch.equal(padded[8:], torch.zeros(4)))

    def test_rows_variant(self):
        w = torch.ones((8, 5))
        padded = pad_hidden_axis_t(w, actual=8, target=12)
        self.assertEqual(padded.shape, (12, 5))
        self.assertTrue(torch.equal(padded[:8], w))
        self.assertTrue(torch.equal(padded[8:], torch.zeros((4, 5))))

    def test_noop_cases(self):
        w2 = torch.ones((4, 8))
        w1 = torch.ones(6)
        wr = torch.ones((6, 3))
        self.assertIs(pad_hidden_axis(w2, actual=8, target=4), w2)   # cannot shrink
        self.assertIs(pad_hidden_axis(w2, actual=8, target=8), w2)   # already fits
        self.assertIs(pad_hidden_axis(w1, actual=8, target=12), w1)  # axis mismatch
        self.assertIs(pad_hidden_axis_t(wr, actual=4, target=8), wr)  # rows != hidden


class PadSuffixRoutingTest(unittest.TestCase):
    def test_input_axis_names(self):
        for name in (
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.3.attn_qkv.weight",
            "blk.4.attn_gate.weight",
            "blk.7.ffn_up.weight",
            "blk.11.ffn_gate.weight",
        ):
            with self.subTest(name=name):
                self.assertTrue(name.endswith(PAD_HIDDEN_INPUT_SUFFIXES))

    def test_output_axis_names(self):
        for name in (
            "blk.3.attn_output.weight",
            "blk.0.ssm_out.weight",
            "blk.7.ffn_down.weight",
        ):
            with self.subTest(name=name):
                self.assertTrue(name.endswith(PAD_HIDDEN_OUTPUT_SUFFIXES))

    def test_norm_names(self):
        for name in ("blk.0.attn_norm.weight", "blk.24.post_attention_norm.weight"):
            with self.subTest(name=name):
                self.assertTrue(name.endswith(PAD_HIDDEN_NORM_SUFFIXES))

    def test_unrelated_names_excluded(self):
        for name in ("blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "output_norm.weight"):
            with self.subTest(name=name):
                self.assertFalse(name.endswith(PAD_HIDDEN_OUTPUT_SUFFIXES))
                self.assertFalse(name.endswith(PAD_HIDDEN_INPUT_SUFFIXES))


class RequantizeRoundTripTest(unittest.TestCase):
    """Padded weights must survive quantize -> unpack with the wider shape."""

    def _roundtrip(self, dtype, cols_actual=64, cols_target=96, rows=32):
        rng = np.random.default_rng(0)
        np_w = rng.standard_normal((rows, cols_target)).astype(np.float32)
        np_w[:, cols_actual:] = 0.0  # zero-padded region
        q = np.ascontiguousarray(
            __import__("gguf").quantize(np_w, dtype).copy()
        )
        if dtype == GGMLQuantizationType.Q4_1:
            d, m, qw = GGUFTensor.unpack_q4_1(q, np_w.shape[1])
            return d, m, qw
        d, m, qw = GGUFTensor.unpack_q8_0(q, np_w.shape[1])
        return d, m, qw

    def test_q4_1_shape(self):
        d, m, qs = self._roundtrip(GGMLQuantizationType.Q4_1)
        self.assertEqual(d.shape, (32, 3))  # rows x (cols / 32-block)
        self.assertEqual(qs.shape, (32, 96))

    def test_q8_0_shape(self):
        d, m, qs = self._roundtrip(GGMLQuantizationType.Q8_0)
        self.assertEqual(d.shape, (32, 3))
        self.assertEqual(qs.shape, (32, 96))

    def test_padded_columns_stay_zero_after_roundtrip(self):
        d, m, qs = self._roundtrip(GGMLQuantizationType.Q8_0)
        # zero-padded region dequantizes to ~0 regardless of per-block scale
        self.assertTrue(torch.all(qs[:, 64:] == 0))


if __name__ == "__main__":
    unittest.main()
