r"""capture_vae_goldens: the VAE chain test's inputs and references.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_vae_goldens.py --size 512

From vae_study.py's cached 4-step latents (<out>\klein_<size>_s4\latents\<i>.pt, the
pipeline's decode input: BN de-normalized, unpatchified [1, 32, H/8, W/8]):

    lat_<i>.npy    the DiT's packed latents [(H/16)^2, 128] as bf16 bits (uint16): the
                   pipeline's patchify and BN normalization inverted, rounded to bf16 --
                   what the NPU's VAE reads
    img_<i>.npy    diffusers' fp32 decode of exactly that input, [H, W, 3] float32 in [0, 1]
    img_<i>.png    the same, rounded to uint8 as the pipeline's postprocess does
    tap_<name>.npy prompt 0's intermediate outputs, NHWC float32: conv_in, mid, up0..up3,
                   conv_out (each module's output; up<b> after its upsampler)

into <out>\goldens_vae_<size>\.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    from diffusers import AutoencoderKLFlux2
    from PIL import Image

    src = Path(a.out) / f"klein_{a.size}_s4" / "latents"
    dst = Path(a.out) / f"goldens_vae_{a.size}"
    dst.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLFlux2.from_pretrained(kqs.MODEL, subfolder="vae",
                                             torch_dtype=torch.float32).eval()
    bn = vae.bn
    s = torch.sqrt(bn.running_var.double() + vae.config.batch_norm_eps)
    m = bn.running_mean.double()
    taps = {}

    def hook(name):
        def f(_mod, _inp, out):
            taps[name] = out[0].permute(1, 2, 0).float().numpy()
        return f

    d = vae.decoder
    mods = {"conv_in": d.conv_in, "mid": d.mid_block, "conv_out": d.conv_out}
    mods |= {f"up{b}": blk for b, blk in enumerate(d.up_blocks)}
    handles = [mod.register_forward_hook(hook(n)) for n, mod in mods.items()]
    for i in range(a.prompts):
        lat = torch.load(src / f"{i:02d}.pt").double()           # [1, 32, 2h, 2w]
        _, c, H2, W2 = lat.shape
        p = lat.reshape(1, c, H2 // 2, 2, W2 // 2, 2).permute(0, 1, 3, 5, 2, 4)
        p = p.reshape(1, 4 * c, H2 // 2, W2 // 2)                # channel 4c' + 2dy + dx
        packed = ((p - m.view(1, -1, 1, 1)) / s.view(1, -1, 1, 1)).to(torch.bfloat16)
        tokens = packed[0].permute(1, 2, 0).reshape(-1, 4 * c)   # [(h w), 128]
        np.save(dst / f"lat_{i}.npy", tokens.view(torch.int16).numpy().view(np.uint16))
        # the decode input the NPU's latent_in computes from those bf16 values
        q = packed.double() * s.view(1, -1, 1, 1) + m.view(1, -1, 1, 1)
        q = q.reshape(1, c, 2, 2, H2 // 2, W2 // 2).permute(0, 1, 4, 2, 5, 3)
        z = q.reshape(1, c, H2, W2).float()
        taps.clear()
        with torch.inference_mode():
            y = vae.decode(z, return_dict=False)[0]
        img = (y[0].permute(1, 2, 0) / 2 + 0.5).clamp(0, 1).numpy()
        np.save(dst / f"img_{i}.npy", img)
        Image.fromarray((img * 255).round().astype(np.uint8)).save(dst / f"img_{i}.png")
        if i == 0:
            for n, t in taps.items():
                np.save(dst / f"tap_{n}.npy", t)
        print(f"  {i}: latents {tuple(tokens.shape)}, image {img.shape}", flush=True)
    for h in handles:
        h.remove()
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
