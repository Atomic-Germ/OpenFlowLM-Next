r"""igpu_bench: FLUX.2 [klein] 4B on the Radeon 890M, timed stage by stage like the NPU engine.

The same diffusers pipeline as klein_quant_study.py's bf16 reference, moved to the iGPU
through PyTorch-ROCm. Same prompts, same CPU-generated noise (seed + i), so the PNGs it
writes score against the CPU bf16 run (score_images.py) and prove the timed path draws
the same pictures.

Timed per image, with a device sync at every boundary:
    text encoder   encode_prompt (Qwen3-4B, 512 rows)
    one step       transformer steps 1..3 (step 0 also carries the latent prep)
    VAE            last step end -> PIL image (BN de-norm, unpatchify, decode, to host)
    image          text encoder + the whole pipeline call
    host CPU       process CPU time per image (a spinning sync shows up here)

Needs its own venv (the ROCm torch must not meet ditref-venv's CPU torch):

    C:\Python311\python.exe -m venv C:\dev\igpu-venv
    C:\dev\igpu-venv\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ --extra-index-url https://pypi.org/simple "torch[device-gfx1150]==2.12.0+rocm7.14.1"
    C:\dev\igpu-venv\Scripts\python.exe -m pip install diffusers==0.40.0 transformers==5.14.1 accelerate==1.14.0

    C:\dev\igpu-venv\Scripts\python.exe utilities\dit-ref\igpu_bench.py --sizes 512,1024
    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py C:\dev\ditref-out\igpu --test igpu_512_{}.png --ref {:02d}.png --ref-dir C:\dev\ditref-out\klein_512_s4\bf16 --n 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from klein_quant_study import MODEL, PROMPTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="512,1024")
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3, help="warm runs per prompt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--offload", action="store_true", help="enable_model_cpu_offload instead of .to(device)")
    ap.add_argument("--out", default=r"C:\dev\ditref-out\igpu")
    a = ap.parse_args()

    from diffusers import Flux2KleinPipeline

    dev = torch.device("cuda")
    sync = torch.cuda.synchronize
    print(f"torch {torch.__version__}, hip {torch.version.hip}, {torch.cuda.get_device_name(0)}", flush=True)
    print(f"sdpa flash={torch.backends.cuda.flash_sdp_enabled()} mem_efficient="
          f"{torch.backends.cuda.mem_efficient_sdp_enabled()} math={torch.backends.cuda.math_sdp_enabled()}"
          f"  AOTRITON_EXPERIMENTAL={os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '')}", flush=True)

    t0 = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    if a.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)
    sync()
    load = time.perf_counter() - t0
    print(f"load {load:.1f}s", flush=True)
    pipe.set_progress_bar_config(disable=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "hip": torch.version.hip, "device": torch.cuda.get_device_name(0),
              "offload": a.offload, "load_s": load, "sizes": {}}

    def one(size, i):
        stamps = []

        def on_step(p, step, t, kw):
            sync()
            stamps.append(time.perf_counter())
            return kw

        c0 = time.process_time()
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            embeds, _ = pipe.encode_prompt(prompt=PROMPTS[i], device=dev)
            sync()
            t1 = time.perf_counter()
            g = torch.Generator("cpu").manual_seed(a.seed + i)
            img = pipe(prompt_embeds=embeds, height=size, width=size, num_inference_steps=4,
                       generator=g, callback_on_step_end=on_step).images[0]
        sync()
        t2 = time.perf_counter()
        assert not pipe.do_classifier_free_guidance
        steps = np.diff([t1] + stamps)
        return img, dict(image=t2 - t0, text_encoder=t1 - t0, step0=steps[0], step=float(np.mean(steps[1:])),
                         vae=t2 - stamps[-1], host_cpu=time.process_time() - c0)

    for size in [int(s) for s in a.sizes.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        _, first = one(size, 0)
        print(f"{size}: first image {first['image']:.2f}s", flush=True)
        rows = []
        for i in range(a.prompts):
            for r in range(a.runs):
                img, t = one(size, i)
                rows.append(t)
                print(f"  {size} [{i}] run {r}: image {t['image']:.2f}s  te {t['text_encoder']:.2f}  "
                      f"step {t['step']:.2f}  vae {t['vae']:.2f}  host {t['host_cpu']:.2f}", flush=True)
            img.save(out / f"igpu_{size}_{i}.png")
        keys = rows[0].keys()
        s = {k: dict(min=min(r[k] for r in rows), max=max(r[k] for r in rows),
                     mean=float(np.mean([r[k] for r in rows]))) for k in keys}
        s["first_image"] = first["image"]
        s["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["sizes"][size] = s
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"{size}: " + "  ".join(f"{k} {v['min']:.2f}-{v['max']:.2f} (mean {v['mean']:.2f})"
                                      for k, v in s.items() if isinstance(v, dict)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
