"""OPEN-CONVERT-DFLASH2: the drafter's ternary matrices land as the engine's exact-ternary q4_1 chunks."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx import model_converter as mc  # noqa: E402
from q4nx.dflash2 import HB, QK, hadamard, pack_q4_1, recover_ternary  # noqa: E402


def bf16(x):
    return torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16).float().numpy()


def ternary(rows, cols, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, (rows, cols)).astype(np.uint8)
    s = bf16(rng.uniform(0.002, 0.05, (rows, cols // QK)))
    return codes, s


class PackQ41Test(unittest.TestCase):
    def test_matches_the_converter_packer(self):
        codes, s = ternary(64, 512)
        got = pack_q4_1(codes, s)
        d = torch.from_numpy(np.repeat(s, QK // 32, axis=1))
        fake = SimpleNamespace(row_block_size=32, col_block_size=256, parallel_size=16, keep_block_in_2D=True)
        want = mc.__dict__["__Q4NX_Converter"]._pack_q4nx(fake, d, -d, torch.from_numpy(codes.astype(np.float32)))
        self.assertEqual(tuple(got.shape), (2, 2, 5120))
        self.assertTrue(torch.equal(got, want))

    def test_decodes_by_the_pool_law(self):
        codes, s = ternary(32, 256, seed=1)
        ch = pack_q4_1(codes, s).numpy().view(np.uint8)[0, 0]
        dd = ch[:512].view("<u2")
        mm = ch[512:1024].view("<u2")
        for r in (0, 5, 17, 31):
            for k in (0, 31, 32, 200, 255):
                idx = (r >> 4) * 4096 + k * 16 + (r & 15)
                q = (ch[1024 + (idx >> 1)] >> (4 * (idx & 1))) & 15
                self.assertEqual(q, codes[r, k])
                g = k // 32
                self.assertEqual(dd[g * 32 + r], bf16([s[r, k // QK]]).view(np.uint32)[0] >> 16)
                self.assertEqual(mm[g * 32 + r], dd[g * 32 + r] ^ 0x8000)


class RecoverTest(unittest.TestCase):
    def test_bf16_gguf_weights_give_back_codes_and_scales(self):
        codes, s = ternary(16, 2048, seed=2)
        w = ((codes.astype(np.float64) - 1).reshape(16, -1, QK) * s[..., None]).reshape(16, -1)
        w = bf16((w.reshape(16, -1, HB) @ hadamard()).reshape(16, -1))
        c2, s2 = recover_ternary(w, "t")
        self.assertTrue(np.array_equal(c2, codes))
        self.assertTrue(np.array_equal(s2, s))

    def test_refuses_a_dense_matrix(self):
        w = np.random.default_rng(3).standard_normal((4, 1024)).astype(np.float32)
        with self.assertRaises(ValueError):
            recover_ternary(w, "dense")


if __name__ == "__main__":
    unittest.main()
