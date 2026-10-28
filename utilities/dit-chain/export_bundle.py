r"""export_bundle: FLUX.2 [klein] 4B as a model directory src/open_diffusion loads -- the
klein_pipeline schedule as files a native engine replays (no Python at run time).

    python utilities\dit-chain\export_bundle.py --checkpoint <FLUX.2-klein-4B snapshot> --out <dir>

`q4nx-build --open-diffusion` runs this (it resolves an HF repo id to the checkpoint and
writes the registry entry). The directory is flat: oflm's remove_model deletes only
top-level files.

    bundle.json          family, the layout hash, resolutions -> schedule files, edits ->
                         schedule files (the sizes an edit configuration exists for), weights
                         (name -> {offset, bytes} in weights.bin), the prompt template, the
                         embedding table
    config.json          model_type flux2-klein, family, resolutions, the layout hash
    schedule_<R>.json    buffers {name: bytes}, init {buffer: file} (loaded once), ops
                         [[set, stream, [arg...], phase]] with arg = [buffer, offset, bytes]
                         (bytes 0 = the whole buffer), and the per-image inputs/outputs
    schedule_<R>e<R>.json  an edit's (klein_pipeline.plan(R, edit=True)); its inputs add the
                         reference buffer's geometry
    weights.bin          every DiT and text-encoder GEMM weight in dit_gemm's bfp16 packing,
                         concatenated, each 4 KiB aligned
    params.bin           PARAMS (qk norm weights + RoPE tables, the text encoder's norms)
    vae_W.bin, vae_S.bin the VAE's packed weights and GroupNorm blocks
    vae_enc_W.bin, vae_enc_S.bin   the encoder's (vae_encoder.pack_weights), with edits
    tf_<R>.bin, dt_<R>.bin, qin_<R>.bin   per-resolution constant buffers
    embed.bin            model.embed_tokens as raw bf16 [151936, 2560]
    tokenizer.json       the checkpoint's
    LICENSE.md           the checkpoint's (Apache 2.0)

The packed weights and the schedules are only valid against kernels built from the same
stream specs: bundle.json and config.json carry export_dit_kernels.layout_hash, and the
engine refuses a kernel set whose manifest has another. Packing needs only the checkpoint
and the layout code, not a kernel build.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate as G  # noqa: E402

kp = G.kp
FAMILY = "FLUX.2-klein-4B-NPU2"
ALIGN = 4096


def _gguf_get(path: Path):
    """`SafeTensors.get`-compatible reader over a klein DiT GGUF (see gguf_weights)."""
    sys.path.insert(0, str(HERE))
    from gguf_weights import GgufWeights

    return GgufWeights(Path(path)).get


def _pack_dit_from_gguf(cache: Path, get, jobs: int = 4) -> dict[str, tuple[Path, int]]:
    """{name: (file, bytes)} for the DiT's GEMM weights, packed from a GGUF getter.

    The same five lines as `G.ensure_packed`'s worker (`build(get) -> pack_b_cols`), without
    the process pool: there is one file to read and it stays mapped, so threads would only
    contend for it. The provenance stamp is the file's identity, so a different GGUF (or a
    repacked node) is repacked rather than reused.
    """
    import hashlib

    specs = kp.dit_weight_specs()
    want = {n: K * N * 9 // 8 for n, (K, N, _) in specs.items()}
    src = get.__self__.gguf.path if hasattr(get, "__self__") else Path("?")
    try:
        st = src.stat()
        fp = hashlib.sha256(f"{src}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()
    except OSError:
        fp = "?"
    cache.mkdir(parents=True, exist_ok=True)
    stamp = cache / "provenance.txt"
    if stamp.exists() and stamp.read_text().strip() != fp:
        print(f"{cache} was packed from another DiT; repacking", flush=True)
        for p in cache.glob("*.bin"):
            p.unlink()
    stamp.write_text(fp + "\n")
    todo = [n for n, nb in want.items()
            if not (cache / f"{n}.bin").exists() or (cache / f"{n}.bin").stat().st_size != nb]
    if todo:
        print(f"packing {len(todo)} DiT weights from {src} -> {cache}", flush=True)
        for n in todo:
            K, N, build = specs[n]
            b = build(get)
            assert b.shape == (K, N), (n, b.shape, (K, N))
            tmp = cache / f"{n}.tmp"
            kp.pack_b_cols(b).tofile(tmp)
            tmp.replace(cache / f"{n}.bin")
        print(f"packed {len(todo)} DiT weights", flush=True)
    return {n: (cache / f"{n}.bin", nb) for n, nb in want.items()}


def build(ckpt: Path, out: Path, resolutions: list[int], jobs: int = 4,
          pack_cache: Path | None = None, edits: list[int] = (),
          transformer_gguf: Path | None = None) -> list[str]:
    """Write the model directory; returns the files written. pack_cache: keep the per-weight
    packed files there (reused by the next build); default a temporary directory. edits: the
    resolutions that also get an edit configuration.

    `transformer_gguf`: take the DiT from a GGUF instead of `ckpt/transformer`. The bundle
    packs the DiT's weights to dit_gemm's bfp16 tiles either way, so the source only has to
    present them by HF name in HF layout -- `gguf_weights.GgufWeights` does (BFL names,
    fused qkv, ggml orientation). The text encoder and the VAE still come from the
    checkpoint: no GGUF of either exists. A GGUF-only source is therefore NOT possible, and
    the caller names the checkpoint that supplies the other two.
    """
    import export_dit_kernels as xk
    import vae_decoder as vd
    import vae_encoder as ve

    out.mkdir(parents=True, exist_ok=True)
    te = G.SafeTensors(ckpt / "text_encoder")
    files = []

    tmp = None
    if pack_cache is None:
        tmp = tempfile.TemporaryDirectory(prefix="klein-pack-", dir=out)
        pack_cache = Path(tmp.name)
    if transformer_gguf is not None:
        # the DiT from the GGUF, the text encoder from the checkpoint (no GGUF of one exists)
        dit_get = _gguf_get(transformer_gguf)
        packed = _pack_dit_from_gguf(pack_cache, dit_get)
        # its own subdirectory: ensure_packed stamps the cache with the CHECKPOINT's identity
        # and clears every *.bin that does not match, which would drop the DiT packs above
        packed |= G.ensure_packed(pack_cache / "text_encoder", jobs, ckpt, only=("text_encoder",))
    else:
        dit = G.SafeTensors(ckpt / "transformer")
        dit_get = dit.get
        packed = G.ensure_packed(pack_cache, jobs, ckpt)
    weights, off = {}, 0
    with open(out / "weights.bin", "wb") as f:
        for n, (p, nb) in packed.items():
            f.write(b"\0" * (-off % ALIGN))
            off += -off % ALIGN
            with open(p, "rb") as src:
                shutil.copyfileobj(src, f, 16 << 20)
            weights[f"w:{n}"] = {"offset": off, "bytes": nb}
            off += nb
    if tmp:
        tmp.cleanup()
    files.append("weights.bin")
    print(f"weights.bin: {len(weights)} weights, {off / 2**30:.2f} GiB", flush=True)

    kp.fill_params(dit_get, te.get).tofile(out / "params.bin")
    vst = G.SafeTensors(ckpt / "vae")
    wbytes, vtable, blocks, vindex = vd.pack_weights(
        {k: vst.get(k) for k in vst._where if not k.endswith("num_batches_tracked")})
    wbytes.tofile(out / "vae_W.bin")
    blocks.view(np.uint16).tofile(out / "vae_S.bin")
    blk = vd.BLOCK * vd.EL * 2
    files += ["params.bin", "vae_W.bin", "vae_S.bin"]
    if edits:
        ewbytes, etable, eblocks, eindex = ve.pack_weights(
            {k: vst.get(k) for k in vst._where if not k.endswith("num_batches_tracked")})
        ewbytes.tofile(out / "vae_enc_W.bin")
        eblocks.view(np.uint16).tofile(out / "vae_enc_S.bin")
        files += ["vae_enc_W.bin", "vae_enc_S.bin"]

    f_, meta = te._where["model.embed_tokens.weight"]
    mm, base = te._maps[f_]
    lo, hi = meta["data_offsets"]
    raw = np.asarray(mm[base + lo:base + hi])
    dt = meta["dtype"]
    if dt == "BF16":
        raw.tofile(out / "embed.bin")
    elif dt in ("F32", "F16"):  # Engine::set_tokens reads bf16
        raw.view(np.float32 if dt == "F32" else np.float16).astype(bfloat16).tofile(out / "embed.bin")
    else:
        raise SystemExit(f"embed_tokens dtype {dt} is not supported (BF16, F16 or F32)")
    shutil.copyfile(ckpt / "tokenizer" / "tokenizer.json", out / "tokenizer.json")
    shutil.copyfile(ckpt / "LICENSE.md", out / "LICENSE.md")
    files += ["embed.bin", "tokenizer.json", "LICENSE.md"]

    schedules, edit_schedules = {}, {}
    for R, edit in [(R, False) for R in resolutions] + [(R, True) for R in edits]:
        pl = kp.plan(R, edit=edit)
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
        if edit:
            buffers |= {"venc_W": ewbytes.size, "venc_S": eblocks.size * 2}
            init |= {"venc_W": "vae_enc_W.bin", "venc_S": "vae_enc_S.bin"}

        def arg(ref):
            kind = ref[0]
            if kind == "buf":
                _, n, o, nb = ref
                return [n, o, 0 if (o == 0 and nb == buffers[n]) else nb]
            if kind == "w":
                return [f"w:{ref[1]}", 0, 0]
            if kind == "vae_w":
                o, n = vtable[ref[1]]
                return ["vae_W", o, n]
            if kind == "vae_gn":
                return ["vae_S", vindex[ref[1]] * blk, blk]
            if kind == "venc_w":
                o, n = etable[ref[1]]
                return ["venc_W", o, n]
            if kind == "venc_gn":
                return ["venc_S", eindex[ref[1]] * blk, blk]
            raise ValueError(ref)

        ops = [[o["set"], o["stream"], [arg(r) for r in o["args"]], o["phase"]] for o in pl.ops]
        inputs = {"tokens": "XT", "token_row_elems": kp.TE_PAD, "latents": "LAT",
                  "ctx": "CTX", "ctx_ld": kp.CTX_LD}
        if edit:
            ib = pl.enc.buffers["IN"]
            inputs |= {"reference": pl.enc_buffers["IN"], "reference_pitch": ib.pitch,
                       "reference_channels": ib.C, "reference_tokens": "REFLAT"}
        sched = {"R": R, "edit": edit, "steps": pl.steps, "image_tokens": kp.image_tokens(R),
                 "latent_channels": kp.LAT_CH, "buffers": buffers, "init": init,
                 "inputs": inputs,
                 "outputs": {"rgba": "v_RGBA", "rgba_row_bytes": 8192, "rgba_used_bytes": 4096},
                 "ops": ops}
        name = f"schedule_{pl.key}.json"
        (out / name).write_text(json.dumps(sched), encoding="utf-8")
        (edit_schedules if edit else schedules)[str(R)] = name
        files += [name, f"tf_{R}.bin", f"dt_{R}.bin", f"qin_{R}.bin"]
        print(f"{name}: {len(ops)} ops, {len(buffers)} buffers", flush=True)

    layout = xk.layout_hash(xk.set_streams(FAMILY, resolutions, edits=edits))
    bundle = {
        "family": FAMILY,
        "layout": layout,
        "resolutions": schedules,
        "edits": edit_schedules,
        "weights_file": "weights.bin",
        "weights": weights,
        "tokenizer": "tokenizer.json",
        "prompt_template": kp.chat_text("{prompt}"),
        "max_tokens": kp.L_TXT, "pad_id": kp.PAD_ID,
        "embed": {"file": "embed.bin", "rows": int(meta["shape"][0]), "dim": int(meta["shape"][1])},
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    # LM_Config::from_pretrained only needs config.json to parse; model_type names the engine
    config = {"model_type": "flux2-klein", "family": FAMILY, "resolutions": resolutions,
              "edits": list(edits), "steps": kp.STEPS, "layout": layout}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    files += ["bundle.json", "config.json"]
    print(f"-> {out} (layout {layout})")
    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default=None, help="default: the HF cache's snapshot")
    ap.add_argument("--resolutions", default="512,1024")
    ap.add_argument("--edits", default="", help="resolutions that also get an edit configuration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4, help="packing processes")
    ap.add_argument("--pack-cache", default=None,
                    help="keep the per-weight packed files here (reused next build)")
    ap.add_argument("--transformer-gguf", default=None,
                    help="take the DiT from this GGUF instead of --checkpoint's transformer/ "
                         "(the text encoder and VAE still come from the checkpoint)")
    a = ap.parse_args()
    if not a.checkpoint and a.transformer_gguf:
        ap.error("--transformer-gguf still needs --checkpoint: the text encoder and the VAE "
                 "have no GGUF")
    build(Path(a.checkpoint) if a.checkpoint else G.model_dir(), Path(a.out).resolve(),
          [int(r) for r in a.resolutions.split(",")], a.jobs,
          Path(a.pack_cache) if a.pack_cache else None,
          [int(r) for r in a.edits.split(",") if r],
          Path(a.transformer_gguf) if a.transformer_gguf else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
