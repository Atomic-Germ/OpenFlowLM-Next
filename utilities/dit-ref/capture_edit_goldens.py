r"""capture_edit_goldens: FLUX.2 [klein] edits on the CPU, the goldens of Phase 8.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_edit_goldens.py --size 512
    ... --size 1024 --prompts 2
    ... --variants npu-emul         # the whole NPU DiT arithmetic (klein_quant_study's modes)

An edit is klein's own pipeline with one reference image (specs/open-diffusion/plans/edits.md).
The references are the study's bf16 images (klein_<size>_s4\bf16, or goldens_vae_1024's
decodes at 1024) and four public-domain / CC0 photos from scikit-image's data:
    astronaut  NASA, public domain (512² RGB: the portrait)
    coffee     CC0 (R. Michetti); written as a JPEG carrying EXIF orientation 6, its pixels
               stored rotated, as a phone writes them
    chelsea    CC0 (S. van der Walt): 451 x 300, smaller than 512, so it is upscaled
    text       public domain: grayscale handwriting, 448 x 172
Each reference is loaded the way diffusers users load one (load_image: EXIF orientation,
then RGB), centre-cropped to square and LANCZOS-resized to R -- the NPU path's preparation
(edits.md decision 2). At R x R, diffusers' own preprocessing is then the identity, so
both pipelines see the same pixels.

Into <out>\klein_edit_<size>_s4\:
    prompts.json     [{ref, prompt}] per edit
    refs\            the reference files as given (the host REFERENCE test reads these)
    ref_<i>.png      the prepared R x R RGB8 reference (the gates' input); ref_<i>.npy the
                     same pixels, uint8 [R, R, 3] (the IRON env has no PIL)
    enc_mean_<i>.npy the fp32 VAE encoder's mean [32, R/8, R/8] (the ENCODER gate)
    reflat_<i>.npy   the bf16 pipeline's reference tokens [(R/16)^2, 128] as bf16 bits
                     (patchify + BN normalise of its bf16 encode)
    reflat32_<i>.npy the same from the fp32 mean, float32
    ids_<i>.npy, noise_<i>.npy, ctx_<i>.npy   as capture_pipeline_inputs.py writes them
    img_ids.npy      the reference tokens' position ids [(R/16)^2, 4] (t = 10)
    tap_<name>.npy   edit 0's fp32 encoder intermediates, NHWC: conv_in, down0..3, mid,
                     conv_out, quant_conv
    <variant>\<i>.png   the edits (bf16: plain diffusers; npu-emul: bfp16-bf16acc linears and
                     attn-fa attention in the transformer's blocks, everything else bf16;
                     enc-wsvd128: bf16, the encoder's mid attention factored at rank 128
                     as the NPU runs it, vae_attn_rank.py; ref-npu: bf16, the reference
                     tokens the NPU encoder made, reflat_npu_<i>.npy from
                     utilities/dit-chain/chain_test_vae_enc.py)

Finished files are skipped, so a run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import klein_quant_study as kqs  # noqa: E402

STUDY_EDITS = [
    'Change the sign to read "CLOSED"',
    "Make it a black and white photograph",
    "Turn the red fox into a white arctic fox",
    "Make it nighttime, the windows glowing warmly",
    "Replace the black marble with white sand",
    "Turn the scene into autumn, the fields golden and the trees red",
    'Change the chalk text to "SOUP OF THE DAY: PUMPKIN"',
    "Render it as a detailed pencil sketch",
]
PHOTO_EDITS = [
    ("astronaut", "Change the background to a beach at sunset"),
    ("coffee", "Replace the espresso with a cappuccino with latte art"),
    ("chelsea", "Give the cat a tiny purple wizard hat"),
    ("text", "Rewrite it in blue ink on yellow lined paper"),
]


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def prepare(img, R: int):
    """Centre crop to square, then LANCZOS to R x R (identity when already R x R)."""
    from PIL import Image
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img if s == R else img.resize((R, R), Image.Resampling.LANCZOS)


def lowrank_encoder_attention(vae, rank: int = 128) -> None:
    """The encoder's d = 512 attention with its score factored at `rank` (plain SVD of
    [Wq^T Wk; bq Wk], vae_attn_rank.py's wsvd), in the VAE's own dtype."""
    import diffusers.models.attention_processor as dap
    import torch.nn.functional as F
    attn = next(m for m in vae.encoder.modules() if isinstance(m, dap.Attention))
    Wq, bq = attn.to_q.weight.double(), attn.to_q.bias.double()
    Wk = attn.to_k.weight.double()
    P, s, Qt = torch.linalg.svd(torch.cat([Wq.T @ Wk, (bq @ Wk)[None]], 0), full_matrices=False)
    U, V = P[:, :rank] * s[:rank].sqrt(), Qt[:rank].T * s[:rank].sqrt()

    class Proc:
        def __call__(self, attn, hidden_states, *args, **kwargs):
            b, c, h, w = hidden_states.shape
            x = attn.group_norm(hidden_states).view(b, c, h * w).transpose(1, 2)
            xd = x[0].double()
            q = (torch.cat([xd, torch.ones_like(xd[:, :1])], 1) @ U).to(x.dtype)[None]
            k = (xd @ V).to(x.dtype)[None]
            o = F.scaled_dot_product_attention(q[:, None], k[:, None], attn.to_v(x)[:, None],
                                               scale=attn.scale)[:, 0]
            return attn.to_out[0](o).transpose(1, 2).reshape(b, c, h, w) + hidden_states

    attn.set_processor(Proc())


def write_sources(refdir: Path, size: int, out: Path) -> list[tuple[str, Path]]:
    """The reference files, as a user would hand them over: [(name, path)]."""
    from PIL import Image
    refdir.mkdir(parents=True, exist_ok=True)
    srcs = []
    if size == 512:
        study = [out / "klein_512_s4" / "bf16" / f"{i:02d}.png" for i in range(len(STUDY_EDITS))]
    else:
        study = [out / f"goldens_vae_{size}" / f"img_{i}.png" for i in range(len(STUDY_EDITS))]
    for i, p in enumerate(study):
        if p.exists():
            srcs.append((f"study{i}", p))
    import skimage
    data = Path(skimage.__file__).parent / "data"
    for name, _ in PHOTO_EDITS:
        if name == "coffee":
            dst = refdir / "coffee_exif6.jpg"
            if not dst.exists():
                im = Image.open(data / "coffee.png").convert("RGB")
                exif = Image.Exif()
                exif[0x0112] = 6                      # display = rotate the stored pixels 90 CW
                im.transpose(Image.Transpose.ROTATE_90).save(dst, quality=92, exif=exif.tobytes())
        else:
            dst = refdir / f"{name}.png"
            if not dst.exists():
                Image.open(data / f"{name}.png").save(dst)
        srcs.append((name, dst))
    return srcs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompts", type=int, default=0, help="first N edits (0: all)")
    ap.add_argument("--variants", default="bf16")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    from diffusers import AutoencoderKLFlux2
    from diffusers.utils import load_image
    from diffusers.utils.torch_utils import randn_tensor
    from PIL import Image
    from transformers import AutoTokenizer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    R, h = a.size, a.size // 16
    T = h * h
    out = Path(a.out)
    dst = out / f"klein_edit_{R}_s{a.steps}"
    dst.mkdir(parents=True, exist_ok=True)
    srcs = write_sources(dst / "refs", R, out)
    prompts = {f"study{i}": p for i, p in enumerate(STUDY_EDITS)} | dict(PHOTO_EDITS)
    edits = [{"ref": n, "src": str(p), "prompt": prompts[n]} for n, p in srcs]
    if a.prompts:
        edits = edits[:a.prompts]
    (dst / "prompts.json").write_text(json.dumps(edits, indent=2), encoding="utf-8")

    # -- references, encoder goldens, host inputs
    refs = []
    vae = None
    for i, e in enumerate(edits):
        rp = dst / f"ref_{i}.png"
        if not rp.exists():
            prepare(load_image(e["src"]), R).save(rp)
        refs.append(Image.open(rp).convert("RGB"))
        np.save(dst / f"ref_{i}.npy", np.asarray(refs[i], np.uint8))
        if not (dst / f"enc_mean_{i}.npy").exists():
            if vae is None:
                vae = AutoencoderKLFlux2.from_pretrained(kqs.MODEL, subfolder="vae",
                                                         torch_dtype=torch.float32).eval()
            taps = {}
            hooks = []
            if i == 0:
                enc = vae.encoder
                mods = {"conv_in": enc.conv_in, "mid": enc.mid_block, "conv_out": enc.conv_out,
                        "quant_conv": vae.quant_conv}
                mods |= {f"down{b}": blk for b, blk in enumerate(enc.down_blocks)}

                def hook(name):
                    def f(_m, _i, o):
                        taps[name] = o[0].permute(1, 2, 0).float().numpy()
                    return f
                hooks = [m.register_forward_hook(hook(n)) for n, m in mods.items()]
            x = torch.from_numpy(np.asarray(refs[i], np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
            with torch.inference_mode():
                mean = vae.encode(x).latent_dist.mode()[0]           # [32, R/8, R/8]
            for hk in hooks:
                hk.remove()
            for n, t in taps.items():
                np.save(dst / f"tap_{n}.npy", t)
            np.save(dst / f"enc_mean_{i}.npy", mean.numpy())
            bn = vae.bn
            s = torch.sqrt(bn.running_var + vae.config.batch_norm_eps)
            p = mean.reshape(32, h, 2, h, 2).permute(0, 2, 4, 1, 3).reshape(128, h, h)
            p = (p - bn.running_mean.view(-1, 1, 1)) / s.view(-1, 1, 1)
            np.save(dst / f"reflat32_{i}.npy", p.reshape(128, T).T.numpy())
        g = torch.Generator("cpu").manual_seed(a.seed + i)
        noise = randn_tensor((1, 128, h, h), generator=g, device=torch.device("cpu"),
                             dtype=torch.bfloat16)
        np.save(dst / f"noise_{i}.npy", bits(noise.reshape(1, 128, T).permute(0, 2, 1)[0]))
    del vae
    ids = torch.cartesian_prod(torch.tensor([10]), torch.arange(h), torch.arange(h),
                               torch.arange(1))
    np.save(dst / "img_ids.npy", ids.numpy())

    tok = AutoTokenizer.from_pretrained(kqs.MODEL, subfolder="tokenizer")
    for i, e in enumerate(edits):
        text = tok.apply_chat_template([{"role": "user", "content": e["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
        np.save(dst / f"ids_{i}.npy", enc["input_ids"][0].astype(np.int64))
        e["tokens"] = int(enc["attention_mask"][0].sum())
    (dst / "prompts.json").write_text(json.dumps(edits, indent=2), encoding="utf-8")

    emb_path = dst / "prompt_embeds.pt"
    embeds = torch.load(emb_path) if emb_path.exists() else None
    if embeds is None or len(embeds) != len(edits):      # absent, or cached by a run with another --prompts
        pipe = kqs.load_pipe(torch.bfloat16)
        with torch.inference_mode():
            embeds = [pipe.encode_prompt(e["prompt"], device="cpu")[0] for e in edits]
        torch.save(embeds, emb_path)
        del pipe
    for i in range(len(edits)):
        np.save(dst / f"ctx_{i}.npy", bits(embeds[i][0]))

    # -- the edits
    for v in a.variants.split(","):
        vdir = dst / v
        vdir.mkdir(exist_ok=True)
        todo = [i for i in range(len(edits)) if not (vdir / f"{i:02d}.png").exists()]
        if not todo:
            continue
        kqs.restore_attention()
        pipe = kqs.load_pipe(torch.bfloat16)
        pipe.text_encoder = None
        if v == "npu-emul":
            kqs.swap_linears(pipe.transformer, "bfp16-bf16acc")
            kqs.swap_attention("attn-fa")
        elif v == "enc-wsvd128":
            lowrank_encoder_attention(pipe.vae)
        elif v == "ref-npu":
            npu_tok = {}

            def from_npu(image, generator, _i=[0]):
                t = torch.from_numpy(np.load(dst / f"reflat_npu_{npu_tok['i']}.npy").view(np.int16))
                return t.view(torch.bfloat16).T.reshape(1, 128, h, h)
            pipe._encode_vae_image = from_npu
        elif v != "bf16":
            raise SystemExit(f"unknown variant {v}")
        if v == "bf16":
            # the reference tokens the bf16 pipeline feeds its transformer
            for i in todo:
                im = pipe.image_processor.preprocess(refs[i], height=R, width=R,
                                                     resize_mode="crop").to(torch.bfloat16)
                with torch.inference_mode():
                    lat = pipe._encode_vae_image(im, generator=None)   # [1, 128, h, h]
                np.save(dst / f"reflat_{i}.npy", bits(lat[0].reshape(128, T).T))
        for i in todo:
            if v == "ref-npu":
                npu_tok["i"] = i
            g = torch.Generator("cpu").manual_seed(a.seed + i)
            t0 = time.time()
            with torch.inference_mode():
                img = pipe(image=refs[i], prompt_embeds=embeds[i], height=R, width=R,
                           num_inference_steps=a.steps, generator=g).images[0]
            img.save(vdir / f"{i:02d}.png")
            print(f"  {v} [{i}] {time.time() - t0:.1f}s  {edits[i]['ref']}: {edits[i]['prompt'][:50]}",
                  flush=True)
        del pipe
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
