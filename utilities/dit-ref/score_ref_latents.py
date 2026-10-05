r"""score_ref_latents: the NPU encoder's reference tokens, judged through the decoder
(OPEN-DIFFUSION-ENCODER's gate; Phase 8, specs/open-diffusion/plans/edits.md).

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_ref_latents.py C:\dev\ditref-out\klein_edit_512_s4

For every reflat_npu_<i>.npy chain_test_vae_enc.py wrote: undo the BN normalisation and
the 2x2 patchify, decode in fp32, and compare with the fp32 decode of diffusers' fp32
encoder mean (enc_mean_<i>.npy) -- LPIPS and PSNR -- plus the mean's rel_fro. Diffusers'
own bf16 encode (reflat_<i>.npy) is scored the same way for scale.
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
    ap.add_argument("goldens")
    ap.add_argument("--test", default="reflat_npu_{}.npy")
    a = ap.parse_args()
    import lpips
    from diffusers import AutoencoderKLFlux2
    from ml_dtypes import bfloat16

    g = Path(a.goldens)
    vae = AutoencoderKLFlux2.from_pretrained(kqs.MODEL, subfolder="vae",
                                             torch_dtype=torch.float32).eval()
    s = torch.sqrt(vae.bn.running_var.double() + vae.config.batch_norm_eps)
    m = vae.bn.running_mean.double()
    loss = lpips.LPIPS(net="alex", verbose=False)

    def mean_of(tokens: np.ndarray) -> torch.Tensor:
        """[(h w), 128] normalised packed tokens -> the encoder mean [1, 32, 2h, 2w]."""
        T = tokens.shape[0]
        h = int(round(T ** 0.5))
        p = torch.from_numpy(tokens.astype(np.float64)).T.reshape(128, h, h)
        p = p * s.view(-1, 1, 1) + m.view(-1, 1, 1)
        return p.reshape(32, 2, 2, h, h).permute(0, 3, 1, 4, 2).reshape(1, 32, 2 * h, 2 * h).float()

    def decode(z):
        with torch.inference_mode():
            return (vae.decode(z, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)

    rows = {"npu": [], "bf16": []}
    i = 0
    while (g / a.test.format(i)).exists():
        ref_mean = torch.from_numpy(np.load(g / f"enc_mean_{i}.npy"))[None]
        ref = decode(ref_mean)
        cands = {"npu": np.load(g / a.test.format(i)).view(bfloat16)}
        if (g / f"reflat_{i}.npy").exists():
            cands["bf16"] = np.load(g / f"reflat_{i}.npy").view(bfloat16)
        line = f"  ref {i}:"
        for name, tok in cands.items():
            z = mean_of(tok)
            rel = float((z - ref_mean).norm() / ref_mean.norm())
            img = decode(z)
            with torch.no_grad():
                lp = float(loss(ref * 2 - 1, img * 2 - 1))
            psnr = 10 * np.log10(1 / max(float(((ref - img) ** 2).mean()), 1e-12))
            rows[name].append((rel, lp, psnr))
            line += f"  {name}: mean rel_fro {rel:.3e} LPIPS {lp:.4f} PSNR {psnr:.1f} dB"
        print(line, flush=True)
        i += 1
    for name, r in rows.items():
        if r:
            r = np.array(r)
            print(f"{name}: mean rel_fro {r[:, 0].mean():.3e}  LPIPS mean {r[:, 1].mean():.4f} "
                  f"max {r[:, 1].max():.4f}  PSNR {r[:, 2].mean():.1f} dB  ({len(r)} references)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
