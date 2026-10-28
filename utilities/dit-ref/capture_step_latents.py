r"""capture_step_latents: diffusers' packed latents after every denoising step (bf16 CPU run).

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_step_latents.py --size 1024 --prompts 1

The study's prompts, embeddings and seeds (klein_quant_study.py). Writes
<out>\goldens_pipe_<size>\step<s>_<i>.npy: [(size/16)^2, 128] bf16 bits, what the NPU runner's
LAT holds after step s -- a per-step check that localizes drift or a bug to a step.
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
    ap.add_argument("--prompts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    dst = Path(a.out) / f"goldens_pipe_{a.size}"
    dst.mkdir(parents=True, exist_ok=True)
    embeds = torch.load(Path(a.out) / f"klein_{a.size}_s4" / "prompt_embeds.pt")
    pipe = kqs.load_pipe(torch.bfloat16)
    pipe.text_encoder = None
    for i in range(a.prompts):
        def cb(p, s, t, kw, i=i):
            lat = kw["latents"][0].to(torch.bfloat16).contiguous().view(torch.int16).numpy()
            np.save(dst / f"step{s}_{i}.npy", lat.view(np.uint16))
            return {}
        g = torch.Generator("cpu").manual_seed(a.seed + i)
        with torch.inference_mode():
            pipe(prompt_embeds=embeds[i].to(torch.bfloat16), height=a.size, width=a.size,
                 num_inference_steps=4, generator=g, output_type="latent",
                 callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        print(f"  prompt {i} -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
