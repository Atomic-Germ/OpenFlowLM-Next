"""SafetensorsStream writes the same bytes as safetensors' save_file, or refuses.

The Bonsai conversion streams its 19 GB container through it instead of holding every tensor
for save_file; a layout that differed would still load in a lenient reader and go unnoticed.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.safetensors_stream import SafetensorsStream  # noqa: E402


class Tiny(SafetensorsStream):
    GAP = 4096        # small enough that the move at close() runs over several chunks
    IMMEDIATE = 64
    COPY = 100


def tensors():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    return {
        "model.layers.1.a": r(48).float(),                       # F32, pending
        "model.layers.0.norm": r(20).to(torch.bfloat16),          # BF16, pending
        "model.embed": r(40, 6).to(torch.bfloat16),               # BF16, rows
        "lm_head": (r(64, 4) * 50).to(torch.int8),                # I8, rows
        "model.layers.0.q": (r(16, 8) * 50).to(torch.int8),       # I8, immediate
        "model.layers.0.alpha": (r(3, 4) * 50).to(torch.int8),    # I8, pending, flushed between
        "model.layers.1.k": (r(10, 9) * 50).to(torch.int8),       # I8, immediate
        "model.layers.10.v": r(7, 3).half(),                      # F16, pending, before the I8
    }


class StreamTest(unittest.TestCase):
    def test_same_bytes_as_save_file(self):
        t = tensors()
        with tempfile.TemporaryDirectory() as d:
            save_file(t, f"{d}/ref.st")
            s = Tiny(f"{d}/s.st")
            for k in ("model.layers.1.a", "model.layers.0.norm", "model.layers.0.alpha", "model.layers.10.v"):
                s[k] = t[k]
            s.write_rows("model.embed", t["model.embed"].split(16))
            s.write_rows("lm_head", t["lm_head"].split(32))
            s["model.layers.0.q"] = t["model.layers.0.q"]
            s["model.layers.1.k"] = t["model.layers.1.k"]
            s.close()
            self.assertEqual(Path(f"{d}/s.st").read_bytes(), Path(f"{d}/ref.st").read_bytes())

    def test_a_tensor_past_its_place_is_refused(self):
        t = tensors()
        with tempfile.TemporaryDirectory() as d:
            s = Tiny(f"{d}/s.st")
            s["model.layers.1.k"] = t["model.layers.1.k"]
            with self.assertRaises(ValueError):
                s["model.layers.0.q"] = t["model.layers.0.q"]
            with self.assertRaises(ValueError):
                s["model.layers.0.alpha"] = t["model.layers.0.alpha"]
            s.f.close()


if __name__ == "__main__":
    unittest.main()
