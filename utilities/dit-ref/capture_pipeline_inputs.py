r"""capture_pipeline_inputs: the whole-image NPU runner's host inputs, from diffusers.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_pipeline_inputs.py --size 512

For klein_quant_study.py's prompts and fixed noise (seed + i), what the diffusers
pipeline computes on the host side or before the first block, so the NPU runner
(utilities/dit-chain/generate.py) can check its own host setup exactly and inject the
study's noise:

    prompts.json     the prompts
    ids_<i>.npy      the tokenized prompt (chat template, right-padded to 512), int64
    noise_<i>.npy    the initial packed latents [(H/16)^2, 128], bf16 bits (uint16)
    ctx_<i>.npy      the bf16 text encoder's prompt embeddings [512, 7680] as bf16 bits
                     (klein_<size>_s4/prompt_embeds.pt, what the study's bf16 run used)
    sched.json       sigmas (with the terminal 0) and timesteps of the 4 steps
    temb.npy         the timestep embedding per step [4, 3072], float32 (from bf16)
    mod.npz          per step: double img/txt [4, 18432], single [4, 9216], norm_out
                     [4, 6144] (diffusers' chunk orders), float32 (from bf16)

into <out>\goldens_pipe_<size>\.
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
import klein_quant_study as kqs  # noqa: E402


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    from diffusers.utils.torch_utils import randn_tensor
    from transformers import AutoTokenizer

    dst = Path(a.out) / f"goldens_pipe_{a.size}"
    dst.mkdir(parents=True, exist_ok=True)
    h = a.size // 16
    T = h * h

    tok = AutoTokenizer.from_pretrained(kqs.MODEL, subfolder="tokenizer")
    embeds = torch.load(Path(a.out) / f"klein_{a.size}_s4" / "prompt_embeds.pt")
    (dst / "prompts.json").write_text(json.dumps(kqs.PROMPTS[:a.prompts], indent=2), encoding="utf-8")
    for i, p in enumerate(kqs.PROMPTS[:a.prompts]):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
        np.save(dst / f"ids_{i}.npy", enc["input_ids"][0].astype(np.int64))
        g = torch.Generator("cpu").manual_seed(a.seed + i)
        noise = randn_tensor((1, 128, h, h), generator=g, device=torch.device("cpu"),
                             dtype=torch.bfloat16)
        np.save(dst / f"noise_{i}.npy", bits(noise.reshape(1, 128, T).permute(0, 2, 1)[0]))
        np.save(dst / f"ctx_{i}.npy", bits(embeds[i][0]))
        print(f"  prompt {i}: {int(enc['attention_mask'][0].sum())} real tokens")

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(kqs.MODEL, subfolder="scheduler")
    sig = np.linspace(1.0, 1 / a.steps, a.steps)
    sched.set_timesteps(sigmas=sig, mu=compute_empirical_mu(image_seq_len=T, num_steps=a.steps))
    ts = sched.timesteps.float()
    (dst / "sched.json").write_text(json.dumps({
        "sigmas": [float(s) for s in sched.sigmas], "timesteps": [float(t) for t in ts],
        "image_seq_len": T}, indent=2))

    tr = Flux2Transformer2DModel.from_pretrained(kqs.MODEL, subfolder="transformer",
                                                 torch_dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        # the pipeline passes t / 1000 in the latents' dtype; forward() multiplies back
        timestep = ((ts.to(torch.bfloat16) / 1000).to(torch.bfloat16)) * 1000
        temb = tr.time_guidance_embed(timestep, None)
        mods = {"dbl_img": tr.double_stream_modulation_img(temb),
                "dbl_txt": tr.double_stream_modulation_txt(temb),
                "sgl": tr.single_stream_modulation(temb),
                "norm_out": tr.norm_out.linear(tr.norm_out.silu(temb).to(temb.dtype))}
    np.save(dst / "temb.npy", temb.float().numpy())
    np.savez(dst / "mod.npz", **{k: v.float().numpy() for k, v in mods.items()})
    print(f"-> {dst}  sigmas {[round(float(s), 4) for s in sched.sigmas]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
