r"""score_images: LPIPS and PSNR of one set of PNGs against another, pairwise by name.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py <dir> --test npu_{}.png --ref img_{}.png [--n 8]

For the chain tests' NPU images against captured references (capture_vae_goldens.py
writes img_<i>.png, chain_test_vae.py npu_<i>.png into the same directory).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--test", default="npu_{}.png")
    ap.add_argument("--ref", default="img_{}.png")
    ap.add_argument("--ref-dir", help="references elsewhere (default: dir)")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    import lpips
    from PIL import Image

    d = Path(a.dir)
    rd = Path(a.ref_dir) if a.ref_dir else d
    loss = lpips.LPIPS(net="alex", verbose=False)

    def load(p):
        x = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
        return torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1

    lp, ps = [], []
    for i in range(a.n):
        t, r = d / a.test.format(i), rd / a.ref.format(i)
        if not (t.exists() and r.exists()):
            continue
        x, y = load(t), load(r)
        with torch.no_grad():
            lp.append(float(loss(x, y)))
        mse = float(((x - y) ** 2).mean()) / 4
        ps.append(10 * np.log10(1 / max(mse, 1e-12)))
        print(f"  {i}: LPIPS {lp[-1]:.4f}  PSNR {ps[-1]:.2f} dB")
    print(f"{len(lp)} images: LPIPS mean {np.mean(lp):.4f} max {np.max(lp):.4f}, "
          f"PSNR mean {np.mean(ps):.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
