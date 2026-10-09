r"""chain_test_vae_enc: FLUX.2 [klein]'s VAE encoder on the NPU, an edit's reference -> its
packed, normalised reference tokens (open_kernels/vae_encoder.py; Phase 8, edits.md 8.2).

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_edit_goldens.py --size 512
    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\dit-chain\chain_test_vae_enc.py --kernels C:\dev\edit-work\enc --build --goldens C:\dev\ditref-out\klein_edit_512_s4
    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_ref_latents.py C:\dev\ditref-out\klein_edit_512_s4

Every op of the encoder's schedule runs on the NPU in five kernel sets (conv, conv1, vew,
gemm, fa); the host writes the prepared reference (capture_edit_goldens.py's ref_<i>.npy,
as bf16 2 (x / 255) - 1 into channels 0-2, diffusers' arithmetic) and reads the tokens. --build builds the
encoder's streams into --kernels first (each set's own directory, as the exporter lays
them out). Checks, against diffusers' fp32 encoder on the same pixels:
  - edit 0's block outputs (conv_in, down0..3, mid): rel_fro
  - every reference's tokens against the fp32 mean, patchified and BN-normalised
    (reflat32_<i>): rel_fro; diffusers' own bf16 encode (reflat_<i>) is printed beside it
The tokens go to <out>\reflat_npu_<i>.npy for score_ref_latents.py (the decoded LPIPS).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae_encoder as ve  # noqa: E402
from chain_test_vae import VAE_DIR, vae_state_dict  # noqa: E402

SET_DIRS = {"conv": "conv", "conv1": "conv1", "vew": "vew", "gemm": ".", "fa": "fa"}
# The ops whose output buffer is compared with a golden tap (edit 0): what -> tap name
TAPS = {"encoder.conv_in": "conv_in",
        "encoder.down_blocks.0.downsamplers.0.conv": "down0",
        "encoder.down_blocks.1.downsamplers.0.conv": "down1",
        "encoder.down_blocks.2.downsamplers.0.conv": "down2",
        "encoder.down_blocks.3.resnets.1 residual": "down3",
        "encoder.mid_block.resnets.1 residual": "mid"}


def build(kdir: Path, R: int, jobs: int) -> None:
    import export_dit_kernels as ek
    for kset, streams in ve.plan(R).streams.items():
        d = (kdir / SET_DIRS[kset]).resolve()
        d.mkdir(parents=True, exist_ok=True)
        dirs = ek.build_many({n: ek.stream_job(kset, s, d) for n, s in streams.items()},
                             False, jobs)
        ek.assemble(d, dirs, d / f"venc_{kset}.json", {"kernel": kset, "streams": streams})


def packed_weights(cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["w"], z["table"].item(), z["blocks"].view(bfloat16), z["index"].item()
    t0 = time.time()
    w, table, blocks, index = ve.pack_weights(vae_state_dict(VAE_DIR))
    np.savez(cache, w=w, table=np.array(table, dtype=object), blocks=blocks.view(np.uint16),
             index=np.array(index, dtype=object))
    print(f"packed the encoder weights in {time.time() - t0:.0f} s -> {cache}")
    return w, table, blocks, index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--refs", type=int, default=0, help="first N references (0: all)")
    ap.add_argument("--out", help="where to write reflat_npu_<i>.npy (default: the goldens dir)")
    ap.add_argument("--runs", type=int, default=1, help="encodes of reference 0 to time")
    ap.add_argument("--dump", help="write edit 0's stage outputs (tap_<name>.npy, NHWC) here")
    a = ap.parse_args()
    kdir, gdir = Path(a.kernels), Path(a.goldens)
    out = Path(a.out) if a.out else gdir
    n_refs = a.refs or len(json.loads((gdir / "prompts.json").read_text(encoding="utf-8")))
    R = int(np.load(gdir / "ref_0.npy", mmap_mode="r").shape[0])
    if a.build:
        build(kdir, R, a.jobs)
    from npu_host import Npu
    pl = ve.plan(R)

    npu = Npu()
    ks = {n: npu.kernel_set(n, kdir / d) for n, d in SET_DIRS.items()}
    wbytes, table, blocks, index = packed_weights(kdir / "venc_packed.npz")
    W = npu.buf("W", wbytes.size)
    W.write(wbytes)
    S = npu.buf("S", blocks.size * 2)
    S.write(blocks)
    bufs = {n: npu.buf(n, b.nelems * 2) for n, b in pl.buffers.items()}
    for b in bufs.values():
        b.zero()                                   # the zero borders, ZERO
    q = np.zeros((pl.buffers["QIN"].H * pl.buffers["QIN"].W, ve.vd.QIN_W), bfloat16)
    q[:, 512] = 1                                  # the attention GEMM's bias row
    bufs["QIN"].write(q)
    streams = {(op["set"], op["stream"]): ks[op["set"]].stream(op["stream"]) for op in pl.ops}
    blk_bytes = ve.BLOCK * ve.EL * 2

    def arg(ref):
        kind, key = ref
        if kind == "buf":
            return bufs[key].bo
        if kind == "w":
            off, n = table[key]
            return W.view(off, n)
        return S.view(index[key] * blk_bytes, blk_bytes)

    args = [[arg(r) for r in op["args"]] for op in pl.ops]
    inb = pl.buffers["IN"]

    def buf_image(name, C):
        b = pl.buffers[name]
        raw = bufs[name].read(np.uint16).view(bfloat16)
        t = raw[:(b.H + 2 * b.border) * b.pitch * C].reshape(b.H + 2 * b.border, b.pitch, C)
        return t[b.border:b.border + b.H, b.border:b.border + b.W].astype(np.float64)

    def encode(i, check_taps):
        rgb = np.load(gdir / f"ref_{i}.npy")
        x = np.zeros((inb.H + 2, inb.pitch, ve.IN_C), bfloat16)
        x[1:R + 1, 1:R + 1, :3] = (np.float32(2) * (rgb.astype(np.float32) / np.float32(255))
                                   - np.float32(1)).astype(bfloat16)
        bufs["IN"].write(x)
        per_set = defaultdict(float)
        stage = []
        t0 = time.perf_counter()
        for op, bo in zip(pl.ops, args):
            per_set[op["set"]] += streams[(op["set"], op["stream"])].run(*bo)
            if check_taps and op["what"] in TAPS:
                name = TAPS[op["what"]]
                ref = np.load(gdir / f"tap_{name}.npy").astype(np.float64)
                conv = op["set"] in ("conv", "conv1")
                dst = op["args"][2 if conv else 3][1]
                spec = pl.streams[op["set"]][op["stream"]]
                got = buf_image(dst, spec["Cout"] if conv else spec["C"])[..., :ref.shape[2]]
                rf = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                stage.append((name, rf, bool(np.isfinite(got).all())))
                if a.dump:
                    Path(a.dump).mkdir(parents=True, exist_ok=True)
                    np.save(Path(a.dump) / f"tap_{name}.npy", got.astype(np.float32))
        total = (time.perf_counter() - t0) * 1e3
        tok = bufs["REF"].read(np.uint16).view(bfloat16).reshape(-1, 4 * ve.LATENT_CH)
        return tok, total, per_set, stage

    ok = True
    rels = []
    for i in range(n_refs):
        tok, total, per_set, stage = encode(i, i == 0)
        np.save(out / f"reflat_npu_{i}.npy", tok.view(np.uint16))
        t = tok.astype(np.float64)
        f32 = np.load(gdir / f"reflat32_{i}.npy").astype(np.float64)
        rel = float(np.linalg.norm(t - f32) / np.linalg.norm(f32))
        line = f"  ref {i}: tokens rel_fro {rel:.3e} vs the fp32 mean"
        if (gdir / f"reflat_{i}.npy").exists():
            b16 = np.load(gdir / f"reflat_{i}.npy").view(bfloat16).astype(np.float64)
            line += f" (diffusers bf16: {np.linalg.norm(b16 - f32) / np.linalg.norm(f32):.3e})"
        rels.append(rel)
        ok &= bool(np.isfinite(t).all())
        if i == 0:
            for name, rf, fin in stage:
                ok &= fin
                print(f"  stage {name:8s} rel_fro {rf:.3e}{'' if fin else '  NON-FINITE'}")
            sets_ms = ", ".join(f"{k} {v:.0f}" for k, v in sorted(per_set.items()))
            print(f"  {len(pl.ops)} dispatches; {total:.0f} ms (first encode) = {sets_ms} ms")
        print(line)
    for _ in range(a.runs - 1):
        _, total, per_set, _ = encode(0, False)
        print(f"  encode {total:.0f} ms = " +
              ", ".join(f"{k} {v:.0f}" for k, v in sorted(per_set.items())) + " ms")
    print(f"encoder {R}x{R}: tokens rel_fro mean {np.mean(rels):.3e} max {np.max(rels):.3e}  "
          f"{'finite' if ok else 'NON-FINITE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
