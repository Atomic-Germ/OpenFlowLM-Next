r"""encoder_study: what the NPU's arithmetic does to FLUX.2 [klein]'s VAE encoder (Phase 8;
specs/open-diffusion/plans/edits.md), the encoder's counterpart of vae_study.py.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\encoder_study.py C:\dev\ditref-out\klein_edit_512_s4 [--npu-dump C:\dev\edit-work\enc_dump]

On capture_edit_goldens.py's prepared references, against the fp32 encoder (its taps and
reflat32_<i>.npy): the stage outputs of edit 0 (conv_in, down0..3, mid) and every
reference's packed, normalised tokens, rel_fro, for
    pipe-bf16     diffusers' bf16 encode (what the bf16 pipeline feeds its DiT)
    npu           vae_study's NPU arithmetic (bfp16 conv operands, bf16 between ops,
                  GroupNorm statistics of bf16 inputs) with the exact d = 512 attention
    npu-r128      the same with the attention's score factored at rank 128 (the schedule's)
--npu-dump: chain_test_vae_enc.py --dump's stage outputs of the NPU itself, compared too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import klein_quant_study as kqs  # noqa: E402
import vae_study as vs  # noqa: E402
from capture_edit_goldens import lowrank_encoder_attention  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("goldens")
    ap.add_argument("--refs", type=int, default=12)
    ap.add_argument("--npu-dump")
    a = ap.parse_args()
    from diffusers import AutoencoderKLFlux2

    g = Path(a.goldens)
    names = ["conv_in", "down0", "down1", "down2", "down3", "mid"]

    def load(dtype, mode=None, r128=False):
        vae = AutoencoderKLFlux2.from_pretrained(kqs.MODEL, subfolder="vae", torch_dtype=dtype).eval()
        if r128:                                # from the unquantised weights, as the packer does
            lowrank_encoder_attention(vae)
        if mode:
            vs.swap(vae.encoder, mode)
            vae.quant_conv = vs.EmuConv(vae.quant_conv, mode)
        return vae

    def tokens(vae, x):
        taps = {}
        enc = vae.encoder
        mods = {"conv_in": enc.conv_in, "mid": enc.mid_block}
        mods |= {f"down{b}": blk for b, blk in enumerate(enc.down_blocks)}
        hs = [m.register_forward_hook(lambda _m, _i, o, n=n: taps.__setitem__(
            n, o[0].permute(1, 2, 0).float().numpy())) for n, m in mods.items()]
        with torch.inference_mode():
            mean = vae.encode(x.to(vae.dtype)).latent_dist.mode()[0].double()
        for h in hs:
            h.remove()
        bn = vae.bn
        s = torch.sqrt(bn.running_var.double() + vae.config.batch_norm_eps)
        h2 = mean.shape[1] // 2
        p = mean.reshape(32, h2, 2, h2, 2).permute(0, 2, 4, 1, 3).reshape(128, h2, h2)
        p = (p - bn.running_mean.double().view(-1, 1, 1)) / s.view(-1, 1, 1)
        return p.reshape(128, -1).T.numpy(), taps

    variants = {"pipe-bf16": dict(dtype=torch.bfloat16), "npu": dict(dtype=torch.float32, mode="npu-gn"),
                "npu-r128": dict(dtype=torch.float32, mode="npu-gn", r128=True)}
    rel = lambda u, v: float(np.linalg.norm(u - v) / np.linalg.norm(v))  # noqa: E731
    stage_ref = {n: np.load(g / f"tap_{n}.npy").astype(np.float64) for n in names}
    if a.npu_dump:
        d = Path(a.npu_dump)
        print("NPU (chain test)  " + "  ".join(
            f"{n} {rel(np.load(d / f'tap_{n}.npy').astype(np.float64), stage_ref[n]):.3e}" for n in names))
    for v, kw in variants.items():
        vae = load(kw["dtype"], kw.get("mode"), kw.get("r128", False))
        rows = []
        for i in range(a.refs):
            rgb = np.load(g / f"ref_{i}.npy")
            x = (2 * (torch.from_numpy(rgb).float() / 255) - 1).permute(2, 0, 1)[None]
            tok, taps = tokens(vae, x)
            rows.append(rel(tok, np.load(g / f"reflat32_{i}.npy").astype(np.float64)))
            if i == 0:
                print(f"{v:16s}  " + "  ".join(f"{n} {rel(taps[n], stage_ref[n]):.3e}" for n in names))
        print(f"{v:16s}  tokens rel_fro per reference {np.round(rows, 4).tolist()}  mean {np.mean(rows):.4f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
