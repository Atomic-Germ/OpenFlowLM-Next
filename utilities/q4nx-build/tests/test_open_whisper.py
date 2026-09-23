"""Tests for --open-whisper (issue #72): the open Whisper engine's container.

A tiny synthetic checkpoint with Whisper's tensor names stands in for the real 1.6 GB one,
so this runs anywhere. Weights are drawn from values bf16 represents exactly, which makes
every layout check an equality rather than a tolerance.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.open_causal import safetensors_header  # noqa: E402
from q4nx.open_whisper import (  # noqa: E402
    GEOMETRY,
    WEIGHTS_NAME,
    bf16_bits,
    build_open_whisper_repo,
)

D, FFN, MELS, NENC, NDEC, VOCAB, SRC, TGT = 16, 32, 4, 2, 2, 40, 6, 5
TINY = {"d_model": D, "encoder_layers": NENC, "decoder_layers": NDEC,
        "encoder_ffn_dim": FFN, "decoder_ffn_dim": FFN, "num_mel_bins": MELS,
        "max_source_positions": SRC, "max_target_positions": TGT, "vocab_size": VOCAB}


def _exact(rng, *shape):
    """fp16 values with the low 3 mantissa bits clear: exact in bf16 and in fp16."""
    return (rng.integers(-64, 64, size=shape) / 32.0).astype(np.float16)


def _checkpoint(root: Path, seed: int = 0) -> dict:
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    t = {}
    e, d = "model.encoder.", "model.decoder."
    t[e + "conv1.weight"], t[e + "conv1.bias"] = _exact(rng, D, MELS, 3), _exact(rng, D)
    t[e + "conv2.weight"], t[e + "conv2.bias"] = _exact(rng, D, D, 3), _exact(rng, D)
    t[e + "embed_positions.weight"] = _exact(rng, SRC, D)
    t[e + "layer_norm.weight"], t[e + "layer_norm.bias"] = _exact(rng, D), _exact(rng, D)
    for i in range(NENC):
        p = f"{e}layers.{i}."
        for n in "qkv":
            t[p + f"self_attn.{n}_proj.weight"] = _exact(rng, D, D)
        t[p + "self_attn.q_proj.bias"] = _exact(rng, D)
        t[p + "self_attn.v_proj.bias"] = _exact(rng, D)
        t[p + "self_attn.out_proj.weight"], t[p + "self_attn.out_proj.bias"] = _exact(rng, D, D), _exact(rng, D)
        t[p + "fc1.weight"], t[p + "fc1.bias"] = _exact(rng, FFN, D), _exact(rng, FFN)
        t[p + "fc2.weight"], t[p + "fc2.bias"] = _exact(rng, D, FFN), _exact(rng, D)
        for n in ("self_attn_layer_norm", "final_layer_norm"):
            t[p + n + ".weight"], t[p + n + ".bias"] = _exact(rng, D), _exact(rng, D)
    t[d + "embed_tokens.weight"] = _exact(rng, VOCAB, D)
    t[d + "embed_positions.weight"] = _exact(rng, TGT, D)
    t[d + "layer_norm.weight"], t[d + "layer_norm.bias"] = _exact(rng, D), _exact(rng, D)
    for l in range(NDEC):
        p = f"{d}layers.{l}."
        for a in ("self_attn", "encoder_attn"):
            for n in "qkvo":
                name = "out_proj" if n == "o" else f"{n}_proj"
                t[p + f"{a}.{name}.weight"] = _exact(rng, D, D)
                if n != "k":
                    t[p + f"{a}.{name}.bias"] = _exact(rng, D)
        t[p + "fc1.weight"], t[p + "fc1.bias"] = _exact(rng, FFN, D), _exact(rng, FFN)
        t[p + "fc2.weight"], t[p + "fc2.bias"] = _exact(rng, D, FFN), _exact(rng, D)
        for n in ("self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"):
            t[p + n + ".weight"], t[p + n + ".bias"] = _exact(rng, D), _exact(rng, D)
    save_file(t, str(root / "model.safetensors"))
    cfg = dict(TINY, model_type="whisper", bos_token_id=50257, eos_token_id=50257,
               encoder_attention_heads=2, decoder_attention_heads=2)
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|endoftext|>"}))
    return {k: v.astype(np.float32) for k, v in t.items()}


def _read(out: Path) -> dict:
    """The container back as fp32, decoding bf16 by hand (numpy has no bf16)."""
    path = out / WEIGHTS_NAME
    res = {}
    for name, m in safetensors_header(path).items():
        n = int(np.prod(m["shape"]))
        if m["dtype"] == "BF16":
            raw = np.fromfile(path, np.uint16, n, offset=m["offset"])
            a = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.fromfile(path, "<f4", n, offset=m["offset"])
        res[name] = (m["dtype"], a.reshape(m["shape"]))
    return res


class OpenWhisperContainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "src").mkdir()
        cls.src = _checkpoint(root / "src")
        cls.out = root / "out"
        build_open_whisper_repo(str(root / "src"), str(cls.out), geometry=TINY)
        cls.c = _read(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fused_qkv_is_wT_with_zero_k_bias(self):
        s = self.src
        for i in range(NENC):
            p = f"model.encoder.layers.{i}.self_attn."
            dt, b = self.c[f"enc.{i}.qkv.B"]
            self.assertEqual(dt, "BF16")
            want = np.concatenate([s[p + f"{n}_proj.weight"] for n in "qkv"]).T
            np.testing.assert_array_equal(b, want)
            _, bias = self.c[f"enc.{i}.qkv.bias"]
            np.testing.assert_array_equal(bias[:D], s[p + "q_proj.bias"])
            np.testing.assert_array_equal(bias[D:2 * D], 0)
            np.testing.assert_array_equal(bias[2 * D:], s[p + "v_proj.bias"])

    def test_cross_kv_order_and_bias(self):
        _, b = self.c["dec.xkv.B"]
        _, bias = self.c["dec.xkv.bias"]
        self.assertEqual(b.shape, (D, 2 * NDEC * D))
        for l in range(NDEC):
            p = f"model.decoder.layers.{l}.encoder_attn."
            np.testing.assert_array_equal(b[:, 2 * l * D:(2 * l + 1) * D], self.src[p + "k_proj.weight"].T)
            np.testing.assert_array_equal(b[:, (2 * l + 1) * D:(2 * l + 2) * D], self.src[p + "v_proj.weight"].T)
            np.testing.assert_array_equal(bias[2 * l * D:(2 * l + 1) * D], 0)
            np.testing.assert_array_equal(bias[(2 * l + 1) * D:(2 * l + 2) * D], self.src[p + "v_proj.bias"])
        self.assertFalse(any(".encoder_attn.k_proj." in n or ".encoder_attn.v_proj." in n for n in self.c),
                         "cross K/V must live only in dec.xkv")

    def test_conv_im2col_operand_computes_the_convolution(self):
        """im2col GEMM with the stored operand == a direct conv1d (pad 1), both convs."""
        rng = np.random.default_rng(1)
        for name, cin, stride in (("conv1", MELS, 1), ("conv2", D, 2)):
            w = self.src[f"model.encoder.{name}.weight"]          # [D, cin, 3]
            _, B = self.c[f"enc.{name}.B"]
            T = 8
            x = rng.standard_normal((T, cin)).astype(np.float32)   # time-major
            xp = np.pad(x, ((1, 1), (0, 0)))
            t_out = (T - 1) // stride + 1
            direct = np.stack([sum(w[:, :, k] @ xp[t * stride + k] for k in range(3))
                               for t in range(t_out)])
            idx = np.arange(t_out) * stride
            a = np.concatenate([xp[idx], xp[idx + 1], xp[idx + 2]], axis=1)
            np.testing.assert_allclose(a @ B, direct, rtol=1e-5, atol=1e-5)

    def test_dtypes(self):
        for name, (dt, a) in self.c.items():
            vector = a.ndim == 1 or name.endswith(("embed_positions.weight", "enc.pos"))
            self.assertEqual(dt, "F32" if vector else "BF16", name)

    def test_tokenizer_config_gets_the_ids_the_host_reads(self):
        cfg = json.loads((self.out / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["bos_token_id"], 50257)
        self.assertEqual(cfg["eos_token_id"], [50257])
        self.assertEqual(cfg["bos_token"], "<|endoftext|>")

    def test_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as t2:
            build_open_whisper_repo(str(Path(self.tmp.name) / "src"), t2, geometry=TINY)
            for f in sorted(p.name for p in self.out.iterdir()):
                self.assertEqual((self.out / f).read_bytes(), (Path(t2) / f).read_bytes(), f)

    def test_refuses_another_geometry(self):
        with tempfile.TemporaryDirectory() as t2, self.assertRaises(ValueError):
            build_open_whisper_repo(str(Path(self.tmp.name) / "src"), t2)   # the real GEOMETRY
        self.assertEqual(GEOMETRY["d_model"], 1280)

    def test_bf16_rounds_to_nearest_even(self):
        x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5], np.float32)
        # 1+2^-8 is a tie between 1 and 1+2^-7: even mantissa wins (1.0).
        # 1+3*2^-8 ties between 1+2^-7 and 1+2^-6: even is 1+2^-6.
        np.testing.assert_array_equal(bf16_bits(x), [0x3F80, 0x3F80, 0x3F82, 0xC020])


if __name__ == "__main__":
    unittest.main()
