r"""export_bundle: klein_pipeline's schedule as files a native engine replays (no Python at
run time): src/open_diffusion reads this directory.

    python utilities\dit-chain\export_bundle.py --kernels C:\dev\klein-kernels --out C:\dev\klein-bundle

    bundle.json          family, kernel sets (dirs), resolutions -> schedule files, weights
                         (name -> file, bytes), the tokenizer and prompt template, te_attn's
                         valid_len words, the embedding table
    schedule_<R>.json    buffers {name: bytes}, init {buffer: file} (loaded once), ops
                         [[set, stream, [arg...], phase]] with arg = [buffer, offset, bytes]
                         (bytes 0 = the whole buffer), and the per-image inputs/outputs
    params.bin           PARAMS (qk norm weights + RoPE tables, the text encoder's norms)
    vae_W.bin, vae_S.bin the VAE's packed weights and GroupNorm blocks
    tf_<R>.bin, dt_<R>.bin, qin_<R>.bin   per-resolution constant buffers
    embed.bin            model.embed_tokens as raw bf16 [151936, 2560]
    tokenizer.json       copied from the model

Weights are referenced where generate.py packed them (<kernels>\packed), not copied.
Every weight is its own buffer ("w:<name>"), so an op argument is always a buffer view.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate as G  # noqa: E402

kp = G.kp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--resolutions", default="512,1024")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import chain_test_vae as ctv

    kdir, out = Path(a.kernels).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    md = G.model_dir()
    dit, te = G.SafeTensors(md / "transformer"), G.SafeTensors(md / "text_encoder")
    packed = G.ensure_packed(kdir / "packed", 4)

    kp.fill_params(dit.get, te.get).tofile(out / "params.bin")
    wbytes, vtable, blocks, vindex = ctv.packed_weights(kdir / "vae_packed.npz")
    wbytes.tofile(out / "vae_W.bin")
    blocks.view(np.uint16).tofile(out / "vae_S.bin")
    blk = ctv.vd.BLOCK * ctv.vd.EL * 2

    f, meta = te._where["model.embed_tokens.weight"]
    mm, base = te._maps[f]
    lo, hi = meta["data_offsets"]
    np.asarray(mm[base + lo:base + hi]).tofile(out / "embed.bin")
    shutil.copyfile(md / "tokenizer" / "tokenizer.json", out / "tokenizer.json")
    fa_meta = json.loads((kdir / "fa" / "dit_fa.json").read_text(encoding="utf-8"))

    schedules = {}
    for R in [int(r) for r in a.resolutions.split(",")]:
        pl = kp.plan(R)
        sig = kp.sigmas(R, pl.steps)
        kp.timestep_features(sig, pl.steps).tofile(out / f"tf_{R}.bin")
        kp.dt_params(sig, pl.steps).tofile(out / f"dt_{R}.bin")
        qin = pl.vae.buffers["QIN"]
        q = np.zeros((qin.H * qin.W, qin.C), bfloat16)
        q[:, 512] = 1
        q.tofile(out / f"qin_{R}.bin")
        buffers = dict(pl.buffers) | {"vae_W": wbytes.size, "vae_S": blocks.size * 2}
        init = {"PARAMS": "params.bin", "vae_W": "vae_W.bin", "vae_S": "vae_S.bin",
                "TF": f"tf_{R}.bin", "DT": f"dt_{R}.bin", "v_QIN": f"qin_{R}.bin"}

        def arg(ref):
            kind = ref[0]
            if kind == "buf":
                _, n, off, nb = ref
                return [n, off, 0 if (off == 0 and nb == buffers[n]) else nb]
            if kind == "w":
                return [f"w:{ref[1]}", 0, 0]
            if kind == "vae_w":
                off, n = vtable[ref[1]]
                return ["vae_W", off, n]
            if kind == "vae_gn":
                return ["vae_S", vindex[ref[1]] * blk, blk]
            raise ValueError(ref)

        ops = [[o["set"], o["stream"], [arg(r) for r in o["args"]], o["phase"]] for o in pl.ops]
        sched = {"R": R, "steps": pl.steps, "image_tokens": kp.image_tokens(R),
                 "latent_channels": kp.LAT_CH, "buffers": buffers, "init": init,
                 "inputs": {"tokens": "XT", "token_row_elems": kp.TE_PAD, "latents": "LAT",
                            "ctx": "CTX", "ctx_ld": kp.CTX_LD},
                 "outputs": {"rgba": "v_RGBA", "rgba_row_bytes": 8192, "rgba_used_bytes": 4096},
                 "ops": ops}
        name = f"schedule_{R}.json"
        (out / name).write_text(json.dumps(sched), encoding="utf-8")
        schedules[str(R)] = name
        print(f"{name}: {len(ops)} ops, {len(buffers)} buffers", flush=True)

    bundle = {
        "family": "FLUX.2-klein-4B-NPU2",
        "kernels": str(kdir),
        "kernel_sets": G.SETS,
        "resolutions": schedules,
        "weights": {f"w:{n}": {"file": str(p), "bytes": nb} for n, (p, nb) in packed.items()},
        "tokenizer": "tokenizer.json",
        "prompt_template": kp.chat_text("{prompt}"),
        "max_tokens": kp.L_TXT, "pad_id": kp.PAD_ID,
        "embed": {"file": "embed.bin", "rows": int(meta["shape"][0]), "dim": int(meta["shape"][1])},
        "patch": fa_meta["patch"],
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
