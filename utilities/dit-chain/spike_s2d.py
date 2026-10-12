r"""spike_s2d: the VAE encoder's stride-2 downsample on the NPU without a new kernel
(Phase 8, spike 8.1.1; specs/open-diffusion/plans/edits.md).

diffusers' Downsample2D: y = conv3x3(pad(x, right 1, bottom 1), stride 2). Here:
  1. the residual add that produces x (x = a + b) is 4 vae_ew dispatches, one per phase
     (p, q): each reads rows 2i + p, columns 2j + q of a and b (px_stride 2C, border 0)
     and writes channels (2p + q)C.. of D, a zero-bordered (H/2) x (W/2) grid of 4C
     channels (px_stride 4C): space-to-depth, D[i, j, (2p+q)C + c] = x[2i+p, 2j+q, c];
  2. dit_conv's 3x3 on D with s2d weights: tap (1 + dy, 1 + dx) channel (2p+q)C + c is
     W[2dy + p, 2dx + q] when that tap exists, everything else (the -1 taps) zero. The
     bottom/right pad lands on D's zero border.

    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\dit-chain\spike_s2d.py --C 128 --H 128 --out C:\dev\edit-work\s2d_128
    (builds the 5 streams into <out>, then runs and checks them against float64)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
OK = ROOT / "open_kernels"
sys.path.insert(0, str(OK))
sys.path.insert(0, str(OK / "designs" / "dit_conv"))
sys.path.insert(0, str(OK / "harness"))

EL, BLOCK = 4096, 19


def parts(W: int) -> int:
    return max(1, 128 // W)


def s2d_weights(w: np.ndarray) -> np.ndarray:
    """[Cout, C, 3, 3] stride-2 weights -> [Cout, 4C, 3, 3] on the s2d grid."""
    co, c = w.shape[:2]
    out = np.zeros((co, 4 * c, 3, 3), w.dtype)
    for p in range(2):
        for q in range(2):
            for dy in range(2):
                for dx in range(2):
                    ky, kx = 2 * dy + p, 2 * dx + q
                    if ky <= 2 and kx <= 2:
                        out[:, (2 * p + q) * c:(2 * p + q + 1) * c, 1 + dy, 1 + dx] = w[:, :, ky, kx]
    return out


def specs(C: int, H: int) -> tuple[dict, dict, dict]:
    """(vew specs, conv spec, layout): full-res grid H x H with C channels."""
    h = H // 2
    px = parts(H) * (H + 2)                       # the full-res buffers' pitch
    pd = parts(h) * (h + 2)                       # D's pitch: the conv reads 130 px of it
    n_full = (H + 2 + 1) * px * C                 # + one extra row, as vae_decoder's buffers
    n_d = (h + 2 + parts(h)) * pd * 4 * C
    vew = {}
    for p in range(2):
        for q in range(2):
            src = {"off": ((p + 1) * px + q + 1) * C, "pitch": px, "border": 0, "px_stride": 2 * C}
            vew[f"s2d_add_{p}{q}"] = {
                "op": "add", "C": C, "H": h, "W": h, "a": src, "b": src,
                "y": {"off": (2 * p + q) * C, "pitch": pd, "border": 1, "px_stride": 4 * C},
                "stats_off": 0, "sizes": {"X": n_full, "B": n_full, "S": BLOCK * EL, "Y": n_d}}
    conv = {"H": h, "W": h, "Cin": 4 * C, "Cout": C, "up": False,
            "x": {"off": 0, "pitch": pd, "border": 1}}
    return vew, conv, {"px": px, "pd": pd, "n_full": n_full, "n_d": n_d}


def build(out: Path, C: int, H: int, jobs: int) -> None:
    import export_dit_kernels as ek
    vew, conv, _ = specs(C, H)
    for kset, design, env_key, extra, sp in (
            ("vew", ek.VEW_DESIGN, "VE_SPEC", {}, vew),
            ("conv", ek.CONV_DESIGN, "DC_SPEC", {"DC_TAPS": 9}, {"s2d_conv": conv})):
        kout = out / kset
        kout.mkdir(parents=True, exist_ok=True)
        dirs = ek.build_many({n: (s | extra, {env_key: json.dumps(s), **extra}, design, kout)
                              for n, s in sp.items()}, False, jobs)
        ek.assemble(kout, dirs, kout / "set.json", {"streams": sp})


def run(out: Path, C: int, H: int, seed: int) -> bool:
    from conv_pack import pack_conv, unpack_conv
    from dit_conv import resolve_spec
    from npu_host import Npu
    vew, conv, lay = specs(C, H)
    h, px, pd = H // 2, lay["px"], lay["pd"]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((H, H, C), dtype=np.float32).astype(bfloat16)
    b = rng.standard_normal((H, H, C), dtype=np.float32).astype(bfloat16)
    w = (rng.standard_normal((C, C, 3, 3)) / np.sqrt(9 * C)).astype(np.float32)
    bias = (0.1 * rng.standard_normal(C)).astype(np.float32)
    packed = pack_conv(s2d_weights(w), bias, 9, False, 4 * C)

    def full(t):
        z = np.zeros((lay["n_full"],), bfloat16)
        z.reshape(-1, px, C)[1:H + 1, 1:H + 1] = t
        return z

    npu = Npu()
    vs = npu.kernel_set("vew", out / "vew")
    cs = npu.kernel_set("conv", out / "conv")
    A, B = npu.buf("A", lay["n_full"] * 2), npu.buf("B", lay["n_full"] * 2)
    A.write(full(a))
    B.write(full(b))
    S = npu.buf("S", BLOCK * EL * 2)
    S.zero()
    D = npu.buf("D", lay["n_d"] * 2)
    D.zero()
    Wb = npu.buf("W", packed.nbytes)
    Wb.write(packed)
    sc = resolve_spec(conv, 9)
    Y = npu.buf("Y", sc["sizes"]["Y"] * 2)
    Y.zero()
    ms = {}
    for n in vew:
        ms[n] = vs.stream(n).run(A.bo, B.bo, S.bo, D.bo)
    ms["conv"] = cs.stream("s2d_conv").run(D.bo, Wb.bo, Y.bo)

    # 1. D is exactly s2d(bf16(a + b)), border zero
    x = (a.astype(np.float32) + b.astype(np.float32)).astype(bfloat16)
    d = D.read(np.uint16).view(bfloat16).reshape(-1, pd, 4 * C)
    want = np.zeros_like(d[:h + 2, :h + 2])
    want[1:h + 1, 1:h + 1] = x.reshape(h, 2, h, 2, C).transpose(0, 2, 1, 3, 4).reshape(h, h, 4 * C)
    n_bad = int((d[:h + 2, :h + 2].view(np.uint16) != want.view(np.uint16)).sum())
    border = np.concatenate([d[0].ravel(), d[h + 1].ravel(), d[:h + 2, 0].ravel(),
                             d[:h + 2, h + 1].ravel()]).astype(np.float32)

    # 2. the conv against diffusers' formula (fp64, the kernel's own bfp16 weights mapped back)
    ws, bk = unpack_conv(packed, 4 * C, C, 9, False)
    wk = np.zeros((C, C, 3, 3))                      # the stride-2 weights the kernel holds
    for p in range(2):
        for q in range(2):
            for dy in range(2):
                for dx in range(2):
                    ky, kx = 2 * dy + p, 2 * dx + q
                    if ky <= 2 and kx <= 2:
                        wk[:, :, ky, kx] = ws[0, :, (2 * p + q) * C:(2 * p + q + 1) * C, 1 + dy, 1 + dx]
    zero_taps = float(np.abs(ws[0][:, :, 0, :]).max() + np.abs(ws[0][:, :, :, 0]).max())
    xp = np.zeros((H + 1, H + 1, C))
    xp[:H, :H] = x.astype(np.float64)
    ref = np.zeros((h, h, C)) + bk[0][:C]
    for ky in range(3):
        for kx in range(3):
            ref += xp[ky:ky + H - 1:2, kx:kx + H - 1:2][:h, :h] @ wk[:, :, ky, kx].T
    y = Y.read(np.uint16).view(bfloat16).reshape(-1, sc["y"]["pitch"], C)[1:h + 1, 1:h + 1]
    rel = float(np.linalg.norm(y.astype(np.float64) - ref) / np.linalg.norm(ref))
    ok = n_bad == 0 and not border.any() and zero_taps == 0 and rel < 3e-2
    t = ", ".join(f"{k} {v:.2f}" for k, v in ms.items())
    print(f"s2d C={C} {H}x{H} -> {h}x{h}: D {n_bad} values wrong, border "
          f"{'zero' if not border.any() else 'DIRTY'}; conv rel_fro {rel:.3e}  [{t} ms]  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--H", type=int, default=128, help="the full-resolution grid (square)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not a.no_build:
        build(out, a.C, a.H, a.jobs)
    return 0 if run(out, a.C, a.H, a.seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
