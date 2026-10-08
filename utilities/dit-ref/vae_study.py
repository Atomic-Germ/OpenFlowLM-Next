r"""vae_study: what the NPU's arithmetic does to FLUX.2 [klein]'s VAE decoder.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\vae_study.py --size 512

diffusers decodes klein's latents with the VAE upcast to fp32 (force_upcast). On the NPU
every conv is an implicit GEMM on dit_gemm (bf16 activations x bfp16ebs8 weights, the
activations converted to bfp16 in-core along K) and every tensor between ops is bf16.
For a conv whose K runs (ky, kx, channel) with channels innermost, a bfp16 block is 8
consecutive channels of one input pixel, so the emulation is exact and cheap: quantize the
input per (pixel, 8-channel block), the weight per (out-channel, ky, kx, 8-channel block),
and run an ordinary fp32 conv (the accumulator's bf16 re-rounding every 64 of K is left
out; it cost nothing visible in the DiT). The mid-block attention goes through fa_emul.

The checkpoint's VAE weights are bf16 and the pipeline decodes in bf16 (no upcast
despite the config), so `pipe-bf16` is what every image of klein_quant_study shows.

Variants, scored against the fp32-math decode of the same latents (LPIPS, PSNR):
    pipe-bf16   diffusers' own bf16 VAE (torch bf16 kernels)
    vae-bf16    activations rounded to bf16 between ops, weights bf16, fp32 math
    vae-npu     + bfp16 conv operands + dit_fa's attention arithmetic
    vae-npu-gn  vae-npu with GroupNorm statistics in bf16 inputs (fp32 math, as on the NPU)

The bf16 DiT latents per prompt are generated once and cached in
<out>\klein_<size>_s4\latents\<i>.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fa_emul  # noqa: E402
import klein_quant_study as kqs  # noqa: E402


def bf(x):
    return x.to(torch.bfloat16).float()


def bfp16_channels(x: torch.Tensor) -> torch.Tensor:
    """NCHW: bfp16 over blocks of 8 channels at each pixel."""
    n, c, h, w = x.shape
    return fa_emul.bfp16(x.reshape(n, c // 8, 8, h, w), dim=2).reshape(n, c, h, w) \
        if c % 8 == 0 else x


def bfp16_weight(w: torch.Tensor) -> torch.Tensor:
    """[Cout, Cin, kh, kw]: bfp16 over blocks of 8 input channels at each (ky, kx)."""
    co, ci, kh, kw = w.shape
    if ci % 8:
        return bf(w)
    return fa_emul.bfp16(bf(w).reshape(co, ci // 8, 8, kh, kw), dim=2).reshape(co, ci, kh, kw)


class EmuConv(nn.Module):
    def __init__(self, conv: nn.Conv2d, mode: str):
        super().__init__()
        self.conv, self.mode = conv, mode
        w = conv.weight.detach().float()
        self.w = bfp16_weight(w) if mode != "bf16" else bf(w)
        self.b = None if conv.bias is None else bf(conv.bias.detach().float())

    def forward(self, x):
        xq = bf(x.float())
        if self.mode != "bf16":
            xq = bfp16_channels(xq)
        y = F.conv2d(xq, self.w, self.b, self.conv.stride, self.conv.padding)
        return bf(y)


class EmuGroupNorm(nn.Module):
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.gn = gn

    def forward(self, x):
        return bf(F.group_norm(bf(x.float()), self.gn.num_groups, bf(self.gn.weight.float()),
                               bf(self.gn.bias.float()), self.gn.eps))


def swap(module: nn.Module, mode: str) -> None:
    for name, m in list(module.named_modules()):
        parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(m, nn.Conv2d):
            setattr(parent, leaf, EmuConv(m, mode))
        elif isinstance(m, nn.GroupNorm) and mode == "npu-gn":
            setattr(parent, leaf, EmuGroupNorm(m))
        elif isinstance(m, nn.Linear) and mode != "bf16":
            setattr(parent, leaf, kqs.EmuLinear(m, "bfp16-emul"))


def attn_emul(mode):
    """The decoder's single-head d=512 attention through fa_emul (NPU modes) or bf16."""
    import diffusers.models.attention_processor as ap

    class Proc:
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                     temb=None, *args, **kwargs):
            residual = hidden_states
            b, c, h, w = hidden_states.shape
            x = attn.group_norm(hidden_states).view(b, c, h * w).transpose(1, 2)
            q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
            q, k, v = (bf(t.float())[:, None] for t in (q, k, v))       # [b, 1, L, 512]
            if mode == "bf16":
                o = F.scaled_dot_product_attention(q, k, v)
            else:
                o = torch.cat([fa_emul.dit_fa_attention(q[:, :, i:i + 1024], k, v)
                               for i in range(0, q.shape[2], 1024)], dim=2)
            o = bf(o[:, 0]).to(x.dtype)
            o = attn.to_out[1](attn.to_out[0](o))
            o = o.transpose(1, 2).reshape(b, c, h, w)
            return bf((o + residual).float()) if mode != "fp32" else o + residual
    return Proc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompts", type=int, default=len(kqs.PROMPTS))
    ap.add_argument("--variants", default="pipe-bf16,vae-bf16,vae-npu,vae-npu-gn")
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    import lpips
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    out = Path(a.out) / f"klein_{a.size}_s4"
    lat_dir = out / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)
    embeds = torch.load(out / "prompt_embeds.pt")
    pipe = Flux2KleinPipeline.from_pretrained(kqs.MODEL, torch_dtype=torch.bfloat16)
    pipe.text_encoder = None
    lats = []
    for i in range(a.prompts):
        f = lat_dir / f"{i:02d}.pt"
        if not f.exists():
            g = torch.Generator("cpu").manual_seed(1234 + i)
            with torch.inference_mode():
                lt = pipe(prompt_embeds=embeds[i].to(torch.bfloat16), height=a.size, width=a.size,
                          num_inference_steps=4, generator=g, output_type="latent").images
            torch.save(lt, f)
            print(f"  latents {i}", flush=True)
        lats.append(torch.load(f))

    # output_type="latent" returns the latents already unpacked, BN de-normalized and
    # unpatchified ([1, 32, H/8, W/8]), i.e. vae.decode's input.
    def to_image(vae, lt):
        with torch.inference_mode():
            y = vae.decode(lt.to(next(vae.parameters()).dtype), return_dict=False)[0]
            return (y.float() / 2 + 0.5).clamp(0, 1)

    ref_vae = pipe.vae.float()
    ref_imgs = [to_image(ref_vae, lt) for lt in lats]
    loss = lpips.LPIPS(net="alex", verbose=False)
    report = {}
    import copy
    import diffusers.models.attention_processor as dap
    for v in a.variants.split(","):
        vae = copy.deepcopy(ref_vae)
        if v == "pipe-bf16":
            vae = vae.to(torch.bfloat16)
        mode = {"vae-bf16": "bf16", "vae-npu": "npu", "vae-npu-gn": "npu-gn"}.get(v)
        if mode:
            swap(vae.decoder, mode)
            vae.post_quant_conv = EmuConv(vae.post_quant_conv, mode)
            for m in vae.decoder.modules():
                if isinstance(m, dap.Attention):
                    m.set_processor(attn_emul(mode))
        lp, ps = [], []
        vdir = out / v
        vdir.mkdir(exist_ok=True)
        for i, lt in enumerate(lats):
            img = to_image(vae, lt)
            r = ref_imgs[i]
            with torch.no_grad():
                lp.append(float(loss(r * 2 - 1, img * 2 - 1)))
            mse = float(((r - img) ** 2).mean())
            ps.append(10 * np.log10(1 / max(mse, 1e-12)))
            arr = (img[0].float().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(vdir / f"{i:02d}.png")
            print(f"  {v} [{i}] LPIPS {lp[-1]:.4f}", flush=True)
        report[v] = dict(lpips_vs_fp32_vae=float(np.mean(lp)), lpips_max=float(np.max(lp)),
                         psnr=float(np.mean(ps)), per_prompt=lp)
        print(f"{v:12s} LPIPS vs fp32 VAE mean {np.mean(lp):.4f} max {np.max(lp):.4f}  "
              f"PSNR {np.mean(ps):.2f} dB", flush=True)
    (out / "vae_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
