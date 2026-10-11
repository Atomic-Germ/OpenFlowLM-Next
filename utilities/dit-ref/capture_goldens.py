r"""capture_goldens: FLUX.2 [klein] 4B block inputs/outputs and weights, for the NPU chain tests.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_goldens.py --size 512

Runs the diffusers pipeline on the CPU (bf16) for one prompt up to the first transformer
call, and saves, under C:\dev\ditref-out\goldens_<size>\:

    step0/          the transformer's inputs and conditioning at step 0: latents (packed),
                    prompt embeddings, timestep, temb, the three modulation outputs, the
                    concatenated RoPE cos/sin, and the step's velocity
    dbl0/, sgl0/, sgl10/   one block each: its inputs (hidden_states, encoder_hidden_states,
                    modulation), diffusers' output ("out_*"), the same block re-run with
                    the NPU arithmetic ("npu_*": dit_gemm's bfp16 x bf16 GEMM and dit_fa's
                    attention, emulated by klein_quant_study.EmuLinear and
                    fa_emul.dit_fa_attention), and every parameter of the block
    manifest.json   names, shapes, dtypes

Tensors are .npy; bf16 values are stored as float32 (exact).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fa_emul  # noqa: E402
import klein_quant_study as kqs  # noqa: E402

BLOCKS = {"dbl0": ("transformer_blocks", 0), "sgl0": ("single_transformer_blocks", 0),
          "sgl10": ("single_transformer_blocks", 10)}


class Stop(Exception):
    pass


def to_np(t):
    return t.detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(a.out) / f"goldens_{a.size}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"size": a.size, "prompt": kqs.PROMPTS[a.prompt], "seed": a.seed + a.prompt,
                "tensors": {}}

    def save(rel, t):
        arr = to_np(t) if torch.is_tensor(t) else np.asarray(t)
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p.parent / (p.name + ".npy"), arr)   # not with_suffix: "x.weight" is a name
        manifest["tensors"][rel] = {"shape": list(arr.shape),
                                    "dtype": str(t.dtype) if torch.is_tensor(t) else str(arr.dtype)}

    emb_path = Path(a.out) / f"klein_{a.size}_s4" / "prompt_embeds.pt"
    if not emb_path.exists():
        emb_path = Path(a.out) / "klein_512_s4" / "prompt_embeds.pt"
    embeds = torch.load(emb_path)[a.prompt]

    pipe = kqs.load_pipe(torch.bfloat16)
    pipe.text_encoder = None
    tr = pipe.transformer
    captured = {}

    def pre_hook(name):
        def f(mod, args, kwargs):
            if name not in captured:
                captured[name] = {"kwargs": {k: v for k, v in kwargs.items()}}
        return f

    def post_hook(name):
        def f(mod, args, kwargs, output):
            if "out" not in captured[name]:
                captured[name]["out"] = output
        return f

    for name, (attr, i) in BLOCKS.items():
        blk = getattr(tr, attr)[i]
        blk.register_forward_pre_hook(pre_hook(name), with_kwargs=True)
        blk.register_forward_hook(post_hook(name), with_kwargs=True)
    step0 = {}

    def tr_pre(mod, args, kwargs):
        if not step0:
            step0.update(kwargs)

    def tr_post(mod, args, kwargs, output):
        step0["velocity"] = output[0] if isinstance(output, tuple) else output.sample
        raise Stop

    def keep(name):
        def f(mod, args, output):
            step0.setdefault(name, output)
        return f

    tr.register_forward_pre_hook(tr_pre, with_kwargs=True)
    tr.register_forward_hook(tr_post, with_kwargs=True)
    for m in ("time_guidance_embed", "double_stream_modulation_img",
              "double_stream_modulation_txt", "single_stream_modulation"):
        getattr(tr, m).register_forward_hook(keep(m))
    tr.norm_out.linear.register_forward_hook(keep("norm_out_linear"))

    g = torch.Generator("cpu").manual_seed(a.seed + a.prompt)
    try:
        with torch.inference_mode():
            pipe(prompt_embeds=embeds, height=a.size, width=a.size, num_inference_steps=4,
                 generator=g)
    except Stop:
        pass

    save("step0/latents", step0["hidden_states"])
    save("step0/prompt_embeds", step0["encoder_hidden_states"])
    save("step0/timestep", step0["timestep"])
    save("step0/img_ids", step0["img_ids"])
    save("step0/txt_ids", step0["txt_ids"])
    save("step0/temb", step0["time_guidance_embed"])
    for m in ("double_stream_modulation_img", "double_stream_modulation_txt",
              "single_stream_modulation", "norm_out_linear"):
        save(f"step0/{m}", step0[m])
    save("step0/velocity", step0["velocity"])

    for name, (attr, i) in BLOCKS.items():
        blk = getattr(tr, attr)[i]
        kw = captured[name]["kwargs"]
        cos, sin = kw["image_rotary_emb"]
        save(f"{name}/rope_cos", cos)
        save(f"{name}/rope_sin", sin)
        if attr == "transformer_blocks":
            save(f"{name}/in_hidden", kw["hidden_states"])
            save(f"{name}/in_encoder", kw["encoder_hidden_states"])
            save(f"{name}/mod_img", kw["temb_mod_img"])
            save(f"{name}/mod_txt", kw["temb_mod_txt"])
            enc, hid = captured[name]["out"]
            save(f"{name}/out_encoder", enc)
            save(f"{name}/out_hidden", hid)
        else:
            save(f"{name}/in_hidden", kw["hidden_states"])
            save(f"{name}/mod", kw["temb_mod"])
            save(f"{name}/out_hidden", captured[name]["out"])
        for pname, prm in blk.named_parameters():
            save(f"{name}/w/{pname.removesuffix('.weight')}", prm)

        # The same block with the NPU arithmetic.
        kqs.swap_linears_in(blk, "bfp16-bf16acc")
        kqs.swap_attention("attn-fa")
        with torch.inference_mode():
            o = blk(**kw)
        kqs.restore_attention()
        if attr == "transformer_blocks":
            save(f"{name}/npu_encoder", o[0])
            save(f"{name}/npu_hidden", o[1])
        else:
            save(f"{name}/npu_hidden", o)
        print(f"{name}: captured", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"goldens -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
