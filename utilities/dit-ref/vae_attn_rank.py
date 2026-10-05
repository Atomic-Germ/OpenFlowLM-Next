r"""vae_attn_rank: can the VAE mid-block attention (one head, d = 512) run as head dim 128?

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\vae_attn_rank.py --size 512

dit_fa is built for head dim 128. The decoder's scores are
    S_ij = (x_i Wq^T + bq)(x_j Wk^T + bk)^T = [x_i, 1] Ma x_j^T + (terms constant along j)
with Ma = [Wq^T Wk ; bq Wk] (513 x 512); the dropped terms shift a row of S by a constant,
which softmax ignores. A rank-r factorization Ma ~ U V^T gives q' = [x, 1] U, k' = x V of
width r, so r = 128 is one dit_fa head, and O = softmax(q'k'^T / sqrt(512)) V runs as four
heads (V's four 128-column slices) sharing q', k'.

Two factorizations, applied to the fp32 decode of vae_study's cached latents and scored
against the exact fp32 decode (LPIPS, PSNR):
    wsvd-r    plain SVD of Ma
    data-r    SVD of Cq^1/2 Ma Ck^1/2 with Cq, Ck the second moments of the attention's
              normalized input over calibration prompts (--calib, default 0-3): minimizes
              the mean squared score error on data like it
Prompts outside the calibration set are reported separately.

--encoder asks the same of the encoder's mid-block attention (Phase 8 edits): its inputs
are the edit study's prepared references (capture_edit_goldens.py, klein_edit_<size>_s4),
the score is decode(mean with the factored attention) against decode(exact mean), both
decoded exactly, plus the latent mean's rel_fro.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import klein_quant_study as kqs  # noqa: E402


def sqrt_and_inv(c: torch.Tensor, eps: float = 1e-6):
    w, v = torch.linalg.eigh(c)
    w = w.clamp_min(eps * w.max())
    return (v * w.sqrt()) @ v.T, (v * w.rsqrt()) @ v.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--ranks", default="128,192,256")
    ap.add_argument("--calib", default="0,1,2,3")
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    ap.add_argument("--encoder", action="store_true")
    a = ap.parse_args()
    import lpips
    from diffusers import AutoencoderKLFlux2
    import diffusers.models.attention_processor as dap

    vae = AutoencoderKLFlux2.from_pretrained(kqs.MODEL, subfolder="vae",
                                             torch_dtype=torch.float32).eval()
    if a.encoder:
        from PIL import Image
        refs = sorted((Path(a.out) / f"klein_edit_{a.size}_s4").glob("ref_*.png"),
                      key=lambda f: int(f.stem.split("_")[1]))
        lats = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), np.float32) / 127.5 - 1)
                .permute(2, 0, 1)[None] for f in refs]
    else:
        out = Path(a.out) / f"klein_{a.size}_s4"
        lats = [torch.load(f) for f in sorted((out / "latents").glob("*.pt"))]
    coder = vae.encoder if a.encoder else vae.decoder
    attn = next(m for m in coder.modules() if isinstance(m, dap.Attention))
    Wq, bq = attn.to_q.weight.double(), attn.to_q.bias.double()
    Wk = attn.to_k.weight.double()
    Ma = torch.cat([Wq.T @ Wk, (bq @ Wk)[None]], 0)             # [513, 512]

    state = {"mode": "exact", "U": None, "V": None, "grab": None}

    class Proc:
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, *args, **kwargs):
            res = hidden_states
            b, c, h, w = hidden_states.shape
            x = attn.group_norm(hidden_states).view(b, c, h * w).transpose(1, 2)
            if state["grab"] is not None:
                state["grab"].append(x[0].double())
            v = attn.to_v(x)
            if state["mode"] == "exact":
                q, k = attn.to_q(x), attn.to_k(x)
            else:
                xd = x[0].double()
                q = (torch.cat([xd, torch.ones_like(xd[:, :1])], 1) @ state["U"]).float()[None]
                k = (xd @ state["V"]).float()[None]
            o = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None],
                                               scale=attn.scale)[:, 0]
            o = attn.to_out[0](o).transpose(1, 2).reshape(b, c, h, w)
            return o + res

    attn.set_processor(Proc())

    means = {}

    def decode(lt):
        with torch.inference_mode():
            if a.encoder:                       # lt is an image: encode, keep the mean
                lt = vae.encode(lt).latent_dist.mode()
                means.setdefault("exact" if state["mode"] == "exact" else "test", []).append(lt)
            y = vae.decode(lt.float(), return_dict=False)[0]
        return (y / 2 + 0.5).clamp(0, 1)

    calib = [int(i) for i in a.calib.split(",")]
    state["grab"] = []
    ref = [decode(lt) for lt in lats]
    xs = state["grab"]
    state["grab"] = None
    X = torch.cat([xs[i] for i in calib])                        # [n, 512]
    X1 = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    Cq_h, Cq_ih = sqrt_and_inv(X1.T @ X1 / len(X1))
    Ck_h, Ck_ih = sqrt_and_inv(X.T @ X / len(X))

    loss = lpips.LPIPS(net="alex", verbose=False)
    for r in [int(x) for x in a.ranks.split(",")]:
        P, s, Qt = torch.linalg.svd(Ma, full_matrices=False)
        facts = {"wsvd": (P[:, :r] * s[:r].sqrt(), Qt[:r].T * s[:r].sqrt())}
        P, s, Qt = torch.linalg.svd(Cq_h @ Ma @ Ck_h, full_matrices=False)
        facts["data"] = (Cq_ih @ (P[:, :r] * s[:r].sqrt()), Ck_ih @ (Qt[:r].T * s[:r].sqrt()))
        for name, (U, V) in facts.items():
            state.update(mode="lowrank", U=U, V=V)
            lp, ps = [], []
            means["test"] = []
            for i, lt in enumerate(lats):
                img = decode(lt)
                with torch.no_grad():
                    lp.append(float(loss(ref[i] * 2 - 1, img * 2 - 1)))
                ps.append(10 * np.log10(1 / max(float(((ref[i] - img) ** 2).mean()), 1e-12)))
            held = [lp[i] for i in range(len(lats)) if i not in calib]
            rel = ""
            if a.encoder:
                rf = [float((t - e).norm() / e.norm()) for t, e in zip(means["test"], means["exact"])]
                rel = f"  latent rel_fro mean {np.mean(rf):.2e} max {np.max(rf):.2e}"
            print(f"{name}-{r:3d}: LPIPS mean {np.mean(lp):.4f} max {np.max(lp):.4f}"
                  f"  held-out mean {np.mean(held) if held else float('nan'):.4f}"
                  f"  PSNR {np.mean(ps):.2f} dB{rel}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
