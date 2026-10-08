r"""chain_test_vae: FLUX.2 [klein]'s VAE decoder on the NPU, packed latents -> RGBA8.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_vae_goldens.py --size 512
    . C:\dev\mlir-aie\iron_env.ps1
    python open_kernels\export_dit_kernels.py --resolutions 512,1024 --out C:\dev\klein-kernels
    python utilities\dit-chain\chain_test_vae.py --kernels C:\dev\klein-kernels --goldens C:\dev\ditref-out\goldens_vae_512

Every op of open_kernels/vae_decoder.py's schedule runs on the NPU, in five kernel sets
(conv, conv1, vew, gemm, fa); the host only writes the latents and reads the RGBA.
Checks, against diffusers' fp32 decode of the same latents (capture_vae_goldens.py):
  - prompt 0's stage outputs (conv_in, mid, up0..up3, conv_out): rel_fro
  - every prompt's image: PSNR, max |error| in uint8 levels. Gate: PSNR >= 35 dB (the
    CPU emulation of this arithmetic measured 54.5 dB, LPIPS 0.0002; utilities/dit-ref
    vae_study.py). The PNGs are written for LPIPS (utilities/dit-ref/score_images.py).
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
import vae_decoder as vd  # noqa: E402
from npu_host import Npu  # noqa: E402
from safetensors_np import SafeTensors  # noqa: E402

VAE_DIR = Path(r"C:\Users\josha\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B")
# The ops whose output buffer is compared with a golden tap (prompt 0).
TAPS = {"decoder.conv_in": "conv_in", "decoder.mid_block.resnets.1 residual": "mid",
        "decoder.up_blocks.0.upsamplers.0.conv": "up0",
        "decoder.up_blocks.1.upsamplers.0.conv": "up1",
        "decoder.up_blocks.2.upsamplers.0.conv": "up2",
        "decoder.up_blocks.3.resnets.2 residual": "up3", "decoder.conv_out": "conv_out"}


def write_png(path: Path, rgb: np.ndarray) -> None:
    """8-bit RGB PNG with zlib only (the IRON env has no PIL)."""
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    h, w, _ = rgb.shape
    raw = b"".join(bytes([0]) + rgb[y].tobytes() for y in range(h))
    sig = bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10])
    path.write_bytes(sig + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def vae_state_dict(model_dir: Path) -> dict[str, np.ndarray]:
    snap = next((model_dir / "snapshots").iterdir()) / "vae"
    st = SafeTensors(snap)
    return {k: st.get(k) for k in st._where if not k.endswith("num_batches_tracked")}


def packed_weights(cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["w"], z["table"].item(), z["blocks"].view(bfloat16), z["index"].item()
    t0 = time.time()
    w, table, blocks, index = vd.pack_weights(vae_state_dict(VAE_DIR))
    np.savez(cache, w=w, table=np.array(table, dtype=object), blocks=blocks.view(np.uint16),
             index=np.array(index, dtype=object))
    print(f"packed the VAE weights in {time.time() - t0:.0f} s -> {cache}")
    return w, table, blocks, index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--out", help="where to write npu_<i>.png (default: the goldens dir)")
    ap.add_argument("--runs", type=int, default=1, help="decodes of prompt 0 to time")
    ap.add_argument("--profile", action="store_true",
                    help="after the timed decodes: the slowest ops, and time in the first op "
                         "after each kernel-set switch (which carries the switch)")
    a = ap.parse_args()
    kdir, gdir = Path(a.kernels), Path(a.goldens)
    out = Path(a.out) if a.out else gdir
    R = int(np.load(gdir / "img_0.npy", mmap_mode="r").shape[0])
    pl = vd.plan(R)

    npu = Npu()
    sets = {"conv": kdir / "conv", "conv1": kdir / "conv1", "vew": kdir / "vew",
            "gemm": kdir, "fa": kdir / "fa"}
    ks = {n: npu.kernel_set(n, d) for n, d in sets.items()}
    wbytes, table, blocks, index = packed_weights(kdir / "vae_packed.npz")
    W = npu.buf("W", wbytes.size)
    W.write(wbytes)
    S = npu.buf("S", blocks.size * 2)
    S.write(blocks)
    bufs = {n: npu.buf(n, b.nelems * 2) for n, b in pl.buffers.items()}
    for b in bufs.values():
        b.zero()                                   # the zero borders
    q = np.zeros((pl.buffers["QIN"].H * pl.buffers["QIN"].W, vd.QIN_W), bfloat16)
    q[:, 512] = 1                                  # the attention GEMM's bias row
    bufs["QIN"].write(q)
    streams = {(op["set"], op["stream"]): ks[op["set"]].stream(op["stream"]) for op in pl.ops}
    blk_bytes = vd.BLOCK * vd.EL * 2

    def arg(ref):
        kind, key = ref
        if kind == "buf":
            return bufs[key].bo
        if kind == "w":
            off, n = table[key]
            return W.view(off, n)
        return S.view(index[key] * blk_bytes, blk_bytes)

    args = [[arg(r) for r in op["args"]] for op in pl.ops]

    def buf_image(name, C):
        b = pl.buffers[name]
        raw = bufs[name].read(np.uint16).view(bfloat16)
        t = raw[:(b.H + 2 * b.border) * b.pitch * C].reshape(b.H + 2 * b.border, b.pitch, C)
        return t[b.border:b.border + b.H, b.border:b.border + b.W].astype(np.float64)

    times: list[float] = []

    def decode(i, check_taps):
        lat = np.load(gdir / f"lat_{i}.npy")
        bufs["LAT"].write(lat)
        per_set = defaultdict(float)
        stage = []
        times.clear()
        t0 = time.perf_counter()
        for op, bo in zip(pl.ops, args):
            ms = streams[(op["set"], op["stream"])].run(*bo)
            per_set[op["set"]] += ms
            times.append(ms)
            if check_taps and op["what"] in TAPS:
                name = TAPS[op["what"]]
                ref = np.load(gdir / f"tap_{name}.npy").astype(np.float64)
                conv = op["set"] in ("conv", "conv1")
                dst = op["args"][2 if conv else 3][1]
                spec = pl.streams[op["set"]][op["stream"]]
                got = buf_image(dst, spec["Cout"] if conv else spec["C"])
                got = got[..., :ref.shape[2]]
                rf = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                stage.append((name, rf, bool(np.isfinite(got).all())))
        total = (time.perf_counter() - t0) * 1e3
        rgba = bufs["RGBA"].read(np.uint8).reshape(-1, 8192)[:, :4096].reshape(R, R, 4)
        return rgba[..., :3], total, per_set, stage

    ok = True
    psnrs = []
    for i in range(a.prompts):
        img, total, per_set, stage = decode(i, i == 0)
        ref = (np.load(gdir / f"img_{i}.npy") * 255).round()
        err = img.astype(np.float64) - ref
        mse = float((err ** 2).mean())
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
        psnrs.append(psnr)
        ok &= psnr >= 35
        write_png(out / f"npu_{i}.png", np.ascontiguousarray(img))
        if i == 0:
            for name, rf, fin in stage:
                ok &= fin
                print(f"  stage {name:9s} rel_fro {rf:.3e}{'' if fin else '  NON-FINITE'}")
            sets_ms = ", ".join(f"{k} {v:.0f}" for k, v in sorted(per_set.items()))
            sw = sum(1 for x, y in zip(pl.ops, pl.ops[1:]) if x["set"] != y["set"])
            print(f"  {len(pl.ops)} dispatches, {sw} kernel-set switches; {total:.0f} ms "
                  f"(first decode) = {sets_ms} ms")
        print(f"  prompt {i}: PSNR {psnr:.2f} dB, max |err| {np.abs(err).max():.0f} levels")
    for _ in range(a.runs - 1):
        _, total, per_set, _ = decode(0, False)
        sets_ms = ", ".join(f"{k} {v:.0f}" for k, v in sorted(per_set.items()))
        print(f"  decode {total:.0f} ms = {sets_ms} ms")
    if a.profile:
        after = [t for k, t in enumerate(times) if k and pl.ops[k]["set"] != pl.ops[k - 1]["set"]]
        same = [t for k, t in enumerate(times) if k and pl.ops[k]["set"] == pl.ops[k - 1]["set"]]
        print(f"  {len(after)} ops right after a switch: {sum(after):.0f} ms; "
              f"{len(same)} after the same set: {sum(same):.0f} ms")
        for k in sorted(range(len(times)), key=lambda k: -times[k])[:15]:
            op = pl.ops[k]
            sw = "*" if k and pl.ops[k - 1]["set"] != op["set"] else " "
            print(f"   {times[k]:7.2f} ms {sw} {op['set']:5s} {op['stream']:16s} {op['what']}")
    print(f"VAE {R}x{R}: mean PSNR {np.mean(psnrs):.2f} dB (min {np.min(psnrs):.2f})  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
