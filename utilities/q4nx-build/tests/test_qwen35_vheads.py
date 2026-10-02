"""The gated DeltaNet's value heads, llama.cpp's tiled GGUF order -> the engine's grouped one.

llama.cpp's convert_hf_to_gguf.py stores Qwen3.5's value heads TILED so ggml can broadcast the
key heads: GGUF position r * num_k + kh holds HF head kh * grp + r, grp = num_v // num_k. The
Q4NX converter must undo exactly that for every value-indexed tensor. It used to hard-code the
9B's geometry -- grp 2, and "v is the second half of qkv" -- which scrambled the 27B (48 value
heads over 16 key heads) in all 48 of its linear-attention layers while every slice compare
still passed, because the kernels and the reference read the same scrambled bytes.

`tile` below is written from llama.cpp's definition, independently of the converter.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.models.qwen35 import untile_qkv, v_untile  # noqa: E402

# (num_k, num_v): the 4B / 9B (grp 2) and the 27B (grp 3)
GEOMETRIES = ((16, 32), (16, 48))


def tile(t: torch.Tensor, num_k: int, num_v: int, axis: int = 0) -> torch.Tensor:
    """HF grouped -> llama.cpp tiled along `axis`: out head r * num_k + kh = in head kh * grp + r."""
    grp = num_v // num_k
    unit = t.shape[axis] // num_v
    heads = t.split(unit, dim=axis)
    order = [kh * grp + r for r in range(grp) for kh in range(num_k)]
    return torch.cat([heads[i] for i in order], dim=axis)


def labelled(num_v: int, unit: int, cols: int = 3) -> torch.Tensor:
    """Rows labelled head * 1000 + row-within-head, so any misplacement shows."""
    h = torch.arange(num_v).repeat_interleave(unit) * 1000 + torch.arange(unit).repeat(num_v)
    return h[:, None].float().repeat(1, cols)


class VHeadUntileTest(unittest.TestCase):
    def test_rows_columns_and_vectors_come_back_grouped(self):
        for num_k, num_v in GEOMETRIES:
            grp = num_v // num_k
            with self.subTest(num_v=num_v):
                rows = labelled(num_v, 128)                          # z, a weight's rows
                self.assertTrue(torch.equal(v_untile(tile(rows, num_k, num_v), grp, 128), rows))
                cols = labelled(num_v, 4).T.contiguous()             # ssm_out's d / m block columns
                self.assertTrue(torch.equal(v_untile(tile(cols, num_k, num_v, axis=1), grp, 4, axis=1), cols))
                vec = torch.arange(num_v).float()                    # A, dt_bias, alpha / beta rows
                self.assertTrue(torch.equal(v_untile(tile(vec, num_k, num_v), grp, 1), vec))

    def test_qkv_untiles_only_the_value_rows_after_q_and_k(self):
        hk = hv = 128
        for num_k, num_v in GEOMETRIES:
            grp, qk_rows = num_v // num_k, 2 * num_k * hk
            with self.subTest(num_v=num_v):
                qk = torch.arange(qk_rows).float()[:, None].repeat(1, 3) + 10 ** 6
                v = labelled(num_v, hv)
                gguf = torch.cat([qk, tile(v, num_k, num_v)])
                self.assertTrue(torch.equal(untile_qkv(gguf, qk_rows, grp, hv), torch.cat([qk, v])))

    def test_the_old_half_split_is_wrong_past_32_heads(self):
        """What the converter did: v = the second half of qkv, untiled two by two. Right at the
        9B, a scramble at the 27B -- the case the tests above pin."""
        hk = hv = 128
        for (num_k, num_v), ok in zip(GEOMETRIES, (True, False)):
            qk_rows = 2 * num_k * hk
            v = labelled(num_v, hv)
            want = torch.cat([torch.zeros(qk_rows, 3), v])
            gguf = torch.cat([torch.zeros(qk_rows, 3), tile(v, num_k, num_v)])
            half = gguf.shape[0] // 2
            old = torch.cat([gguf[:half], v_untile(gguf[half:], 2, hv)])
            self.assertEqual(torch.equal(old, want), ok, num_v)


if __name__ == "__main__":
    unittest.main()
