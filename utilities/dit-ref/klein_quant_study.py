r"""klein_quant_study: what each NPU GEMM route does to FLUX.2 [klein] 4B images.

Phase 1 step 2 of .claude/plans/image-diffusion-phase1-plan.md. Runs the diffusers
pipeline on the CPU. Every nn.Linear inside the transformer's double- and
single-stream blocks is swapped for a numerics-faithful emulation of one NPU
datapath; everything else stays bf16. Then it regenerates a fixed prompt set with
fixed noise and scores each variant against the bf16 run.

    bf16            the reference: plain bf16 pipeline
    fp32            the same pipeline in fp32 -- the noise floor of the metrics
    bfp16-emul      gemm_pretiled --emulate-bfp16: A and B each rounded to bfp16
                    (8 values along K share an exponent, round-half-even), fp32
                    accumulate over all of K, C rounded to bf16
    bfp16-bf16acc   mlir-aie whole_array_mixed / ATB (with OFLM's rounding fix):
                    as above, but the accumulator is stored as bf16 after every
                    64 of K, so C is re-rounded K/64 times
    w8a8            gemm_pretiled --int8: activations int8 per token (absmax),
                    weights int8 per output channel (absmax), int32 accumulate,
                    int32 -> bf16, then the two scales
    attn-fa         the transformer's attention through dit_fa's arithmetic
                    (fa_emul.dit_fa_attention: bfp16 Q/K/P/V, bf16 scores and
                    accumulator, the hardware's linear-mantissa exp2)
    attn-fa-exact   the same with an exactly rounded bf16 exp2

A "+" joins a linear mode and an attention mode, e.g. bfp16-bf16acc+attn-fa is the
whole NPU DiT arithmetic (dit_gemm and dit_fa) at once.

A variant starting "te-" generates with prompt embeddings from prompt_embeds_<variant>.pt
(written by te_npu_embeds.py: the text encoder on the NPU's arithmetic), the DiT in bf16;
"+" can add the DiT modes after it, e.g. te-npu+bfp16-bf16acc+attn-fa is the whole NPU
arithmetic.

Needs the venv at C:\dev\ditref-venv (torch CPU, diffusers >= 0.37, lpips) and the
model in the Hugging Face cache (hf download black-forest-labs/FLUX.2-klein-4B
--exclude flux-2-klein-4b.safetensors).

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\klein_quant_study.py --size 512
    ... --variants bf16,bfp16-bf16acc --prompts 2      # quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL = "black-forest-labs/FLUX.2-klein-4B"

PROMPTS = [
    'A storefront sign that reads "OPEN LATE", neon letters, rainy street at night',
    "Close-up portrait of an elderly fisherman, weathered skin, soft window light",
    "A red fox in fresh snow, detailed fur, morning light, telephoto",
    "Isometric illustration of a tiny cozy library with warm lamps, clean vector style",
    "Product photo of a glass perfume bottle on black marble, studio lighting, reflections",
    "Aerial photograph of terraced rice fields at sunrise, mist in the valleys",
    'A chalkboard menu with the words "SOUP OF THE DAY: TOMATO" written in chalk',
    "Gothic cathedral interior, stained glass light beams, high detail stonework",
]

VARIANTS = ["bf16", "fp32", "bfp16-emul", "bfp16-bf16acc", "w8a8"]
ATTN_MODES = ("attn-fa", "attn-fa-exact")


# ------------------------------------------------------------ number formats

def bfp16(x: torch.Tensor) -> torch.Tensor:
    """Round to bfp16ebs8 along the last dim: blocks of 8 share the largest
    element's exponent, each keeps an int8 mantissa (64..127 for the largest),
    round-half-even. Returns the dequantized values in x's dtype (fp32)."""
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // 8, 8)
    _, e = torch.frexp(blocks)                       # |x| in [2^(e-1), 2^e)
    emax = e.amax(dim=-1, keepdim=True)
    scale = torch.ldexp(torch.ones_like(blocks[..., :1]), emax - 7)
    q = torch.round(blocks / scale).clamp_(-128, 127)
    return (q * scale).reshape(shape)


def int8_rows(x: torch.Tensor):
    """Symmetric int8 per row (last dim reduced). Returns (q as float, scale)."""
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    return torch.round(x / s).clamp_(-127, 127), s


class EmuLinear(nn.Module):
    """y = x W^T + b with one NPU datapath's arithmetic. W is [out, in]; the GEMM
    reduces over `in`, which is the dimension bfp16 blocks and int8 scales run along."""

    def __init__(self, lin: nn.Linear, mode: str):
        super().__init__()
        self.mode = mode
        w = lin.weight.detach().float()
        self.bias = None if lin.bias is None else lin.bias.detach().float()
        if mode in ("bfp16-emul", "bfp16-bf16acc"):
            self.w = bfp16(w)
        elif mode == "w8a8":
            self.w, self.ws = int8_rows(w)               # per output channel
        else:
            raise ValueError(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        lead = xf.shape[:-1]
        xf = xf.reshape(-1, xf.shape[-1])
        if self.mode == "bfp16-emul":
            y = (bfp16(xf) @ self.w.T).to(torch.bfloat16).float()
        elif self.mode == "bfp16-bf16acc":
            xq = bfp16(xf)
            y = torch.zeros(xq.shape[0], self.w.shape[0])
            for c in range(0, xq.shape[1], 64):
                y = (y + xq[:, c:c + 64] @ self.w[:, c:c + 64].T).to(torch.bfloat16).float()
        else:  # w8a8
            xq, xs = int8_rows(xf)                          # per token
            y = (xq @ self.w.T).to(torch.bfloat16).float() * xs * self.ws.T
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead, -1).to(dt)


def swap_linears_in(block: nn.Module, mode: str) -> tuple[int, int]:
    """Replace every nn.Linear in one block. Returns (layers swapped, parameters covered)."""
    n = p = 0
    for name, mod in list(block.named_modules()):
        if isinstance(mod, nn.Linear):
            parent = block.get_submodule(name.rsplit(".", 1)[0]) if "." in name else block
            setattr(parent, name.rsplit(".", 1)[-1], EmuLinear(mod, mode))
            n += 1
            p += mod.weight.numel()
    return n, p


def swap_linears(transformer: nn.Module, mode: str) -> tuple[int, int]:
    """Replace every nn.Linear under the double/single-stream blocks. Returns
    (layers swapped, parameters covered)."""
    n = p = 0
    for blocks_name in ("transformer_blocks", "single_transformer_blocks"):
        for block in getattr(transformer, blocks_name):
            dn, dp = swap_linears_in(block, mode)
            n += dn
            p += dp
    return n, p


_ORIG_DISPATCH: dict = {}


def swap_attention(mode: str) -> None:
    """Route the Flux2 processors' attention through fa_emul's dit_fa model."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fa_emul
    import diffusers.models.transformers.transformer_flux2 as tf
    stages = fa_emul.FULL if mode == "attn-fa" else fa_emul.FULL - {"exp_hw"}
    orig = _ORIG_DISPATCH.setdefault("fn", tf.dispatch_attention_fn)

    def fa(query, key, value, attn_mask=None, **kw):
        if attn_mask is not None:
            return orig(query, key, value, attn_mask=attn_mask, **kw)
        q, k, v = (t.float().transpose(1, 2) for t in (query, key, value))   # [B, H, S, D]
        return fa_emul.dit_fa_attention(q, k, v, stages).transpose(1, 2).to(query.dtype)

    tf.dispatch_attention_fn = fa


def restore_attention() -> None:
    import diffusers.models.transformers.transformer_flux2 as tf
    if "fn" in _ORIG_DISPATCH:
        tf.dispatch_attention_fn = _ORIG_DISPATCH["fn"]


# ------------------------------------------------------------------ driver

def load_pipe(dtype):
    from diffusers import Flux2KleinPipeline
    return Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.out) / f"klein_{args.size}_s{args.steps}"
    out.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[:args.prompts]
    variants = args.variants.split(",")

    # Text embeddings once, in bf16, shared by every variant (the text encoder is
    # not what is being studied).
    emb_path = out / "prompt_embeds.pt"
    if emb_path.exists():
        embeds = torch.load(emb_path)
    else:
        pipe = load_pipe(torch.bfloat16)
        t0 = time.time()
        with torch.inference_mode():
            embeds = [pipe.encode_prompt(p, device="cpu")[0] for p in PROMPTS]
        torch.save(embeds, emb_path)
        print(f"text embeddings: {len(PROMPTS)} prompts in {time.time() - t0:.1f}s", flush=True)
        del pipe

    report = {}
    rpath = out / "report.json"
    if rpath.exists():
        report = json.loads(rpath.read_text())
    for v in variants:
        vdir = out / v
        vdir.mkdir(exist_ok=True)
        dtype = torch.float32 if v == "fp32" else torch.bfloat16
        import diffusers.models.transformers.transformer_flux2 as tf
        tf.dispatch_attention_fn = _ORIG_DISPATCH.setdefault("fn", tf.dispatch_attention_fn)
        pipe = load_pipe(dtype)
        pipe.text_encoder = None
        info = {}
        te_tag = next((m for m in v.split("+") if m.startswith("te-")), None)
        v_embeds = embeds
        if te_tag:
            v_embeds = torch.load(out / f"prompt_embeds_{te_tag}.pt")
            info["text_encoder"] = te_tag
        lin_mode = next((m for m in v.split("+") if m not in ATTN_MODES
                         and not m.startswith("te-")), None)
        attn_mode = next((m for m in v.split("+") if m in ATTN_MODES), None)
        if attn_mode:
            swap_attention(attn_mode)
            info["attention"] = attn_mode
        if lin_mode not in (None, "bf16", "fp32"):
            n, p = swap_linears(pipe.transformer, lin_mode)
            total = sum(q.numel() for q in pipe.transformer.parameters()) + p
            info = dict(layers=n, params=p, share=p / total)
            print(f"{v}: emulating {n} linears, {p / 1e9:.2f}B params", flush=True)
        if attn_mode:
            print(f"{v}: attention through {attn_mode}", flush=True)
        times = []
        for i, pr in enumerate(prompts):
            png = vdir / f"{i:02d}.png"
            if png.exists():
                continue
            g = torch.Generator("cpu").manual_seed(args.seed + i)
            t0 = time.time()
            with torch.inference_mode():
                img = pipe(prompt_embeds=v_embeds[i].to(dtype), height=args.size, width=args.size,
                           num_inference_steps=args.steps, generator=g).images[0]
            times.append(time.time() - t0)
            img.save(png)
            print(f"  {v} [{i}] {times[-1]:.1f}s  {pr[:50]}", flush=True)
        info["sec_per_image"] = float(np.mean(times)) if times else None
        report.setdefault(v, {}).update(info)
        rpath.write_text(json.dumps(report, indent=2))
        del pipe

    score(out, prompts, variants, report)
    rpath.write_text(json.dumps(report, indent=2))
    grid(out, prompts, variants)
    return 0


def score(out: Path, prompts, variants, report) -> None:
    import lpips
    from PIL import Image
    loss = lpips.LPIPS(net="alex", verbose=False)

    def load(p):
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        return a, torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1

    for v in variants:
        if v == "bf16":
            continue
        lp, ps = [], []
        for i in range(len(prompts)):
            ra, rt = load(out / "bf16" / f"{i:02d}.png")
            va, vt = load(out / v / f"{i:02d}.png")
            with torch.no_grad():
                lp.append(float(loss(rt, vt)))
            mse = float(((ra - va) ** 2).mean())
            ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        report[v].update(lpips_vs_bf16=float(np.mean(lp)), lpips_max=float(np.max(lp)),
                         psnr_vs_bf16=float(np.mean(ps)), lpips_per_prompt=lp)
        print(f"{v:<14} LPIPS vs bf16 mean {np.mean(lp):.4f} max {np.max(lp):.4f}   "
              f"PSNR {np.mean(ps):.2f} dB", flush=True)


def grid(out: Path, prompts, variants, cell: int = 256) -> None:
    from PIL import Image, ImageDraw
    W, H = cell * len(variants), cell * len(prompts) + 24
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    for j, v in enumerate(variants):
        d.text((j * cell + 6, 6), v, fill="black")
        for i in range(len(prompts)):
            p = out / v / f"{i:02d}.png"
            if p.exists():
                g.paste(Image.open(p).convert("RGB").resize((cell, cell)), (j * cell, 24 + i * cell))
    g.save(out / "grid.png")
    print(f"grid: {out / 'grid.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
