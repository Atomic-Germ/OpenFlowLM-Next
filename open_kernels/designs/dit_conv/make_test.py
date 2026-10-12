r"""make_test: dit_conv at one shape -- the build's spec, inputs, and (with --build) a run
on the NPU checked against float64.

    python make_test.py --H 32 --W 128 --Cin 64 --Cout 128 [--up] [--taps 1] --out <dir>
        writes <dir>/spec.json; then (IRON env)
    DC_TAPS=9 DC_SPEC=<spec.json contents> python build_design.py designs/dit_conv/dit_conv.py <build>
    python make_test.py <same shape> --out <dir> --build <build> [--runs 3]

--cin-real / --cout-real test channel padding (the VAE's conv_in reads 32 channels padded
to 64; conv_out writes 3 of 128). --phases (with --taps 1 --up) gives each of the 4
output phases its own random 1x1 map (the VAE's unpatchify), --x-plain reads X as a
plain [H*W, Cin] tensor.

Reference: float64 from bf16 X with the weights and biases the kernel multiplies by
(conv_pack.unpack_conv) -- "kernel" -- and with the unquantized weights -- "exact". Gate:
rel_fro against "kernel" < 3e-2 (dit_gemm's; the rest is X's in-core bfp16 conversion and
C's bf16 re-rounding every 64 of K). Also: Y's zero border must stay zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from conv_pack import pack_conv, pack_conv_phases, unpack_conv  # noqa: E402

GATE = 3e-2


def conv_ref(xpad: np.ndarray, w: np.ndarray, rows=None) -> np.ndarray:
    """xpad [H+2, W+2, Cin] (zero border), w [Cout, Cin, kh, kw] -> [len(rows), W, Cout]
    (rows: output rows, default all)."""
    H, W = xpad.shape[0] - 2, xpad.shape[1] - 2
    rows = np.arange(H) if rows is None else np.asarray(rows)
    kh = w.shape[2]
    o = 1 - kh // 2
    out = np.zeros((len(rows), W, w.shape[0]))
    for ky in range(kh):
        for kx in range(kh):
            out += xpad[rows + o + ky, o + kx:o + kx + W] @ w[:, :, ky, kx].T
    return out


def sample_rows(n: int, k: int = 12) -> np.ndarray:
    """All rows of small outputs; else the first and last 4 and 4 from the middle."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.r_[0:4, n // 2 - 2:n // 2 + 2, n - 4:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    for d in ("H", "W", "Cin", "Cout"):
        ap.add_argument(f"--{d}", type=int, required=True)
    ap.add_argument("--cin-real", type=int)
    ap.add_argument("--cout-real", type=int)
    ap.add_argument("--up", action="store_true")
    ap.add_argument("--taps", type=int, default=9)
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--x-plain", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--build")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = {"H": a.H, "W": a.W, "Cin": a.Cin, "Cout": a.Cout, "up": a.up}
    if a.x_plain:
        spec["x"] = {"off": 0, "pitch": a.W, "border": 0}
    (out / "spec.json").write_text(json.dumps(spec))
    if not a.build:
        print(f"wrote {out / 'spec.json'}")
        return 0

    from dit_conv import resolve_spec
    sys.path.insert(0, str(HERE.parents[1] / "harness"))
    from npu_host import KernelSet, Npu, Stream

    s = resolve_spec(spec, a.taps)
    H, W, Ho, Wo = s["H"], s["W"], s["Ho"], s["Wo"]
    ci, co = a.cin_real or a.Cin, a.cout_real or a.Cout
    kh = 3 if a.taps == 9 else 1
    rng = np.random.default_rng(a.seed)
    x = rng.standard_normal((H, W, ci), dtype=np.float32).astype(bfloat16)
    w = (rng.standard_normal((co, ci, kh, kh)) / np.sqrt(ci * kh * kh)).astype(bfloat16)
    b = (0.1 * rng.standard_normal(co)).astype(bfloat16)
    if a.phases:
        wp = (rng.standard_normal((4, co, ci, kh, kh)) / np.sqrt(ci)).astype(bfloat16)
        bph = (0.1 * rng.standard_normal((4, co))).astype(bfloat16)
        packed = pack_conv_phases(wp.astype(np.float32), bph.astype(np.float32), a.Cin)
    else:
        packed = pack_conv(w.astype(np.float32), b.astype(np.float32), a.taps, a.up, a.Cin)
    ws, bias = unpack_conv(packed, a.Cin, a.Cout, a.taps, a.up)

    xpad = np.zeros((H + 2, W + 2, a.Cin))
    xpad[1:-1, 1:-1, :ci] = x.astype(np.float64)
    rows = sample_rows(Ho)                                   # output rows checked
    n_ph = len(ws)

    def assemble(fn, width):
        """Output rows `rows` from fn(phase, source rows) -> [n, W, width] (+ its bias)."""
        o = np.zeros((len(rows), Wo, width))
        for p in range(n_ph):
            py, px = divmod(p, 2) if a.up else (0, 0)
            sel = rows % 2 == py if a.up else np.ones(len(rows), bool)
            src = rows[sel] // 2 if a.up else rows[sel]
            o[sel, px::2 if a.up else 1] = fn(p, src)
        return o

    # "kernel": the weights and biases the kernel multiplies by
    kern = assemble(lambda p, r: conv_ref(xpad, ws[p], r) + bias[p], a.Cout)[..., :co]
    # "exact": the unquantized weights
    if a.phases:
        exact = assemble(lambda p, r: conv_ref(xpad[..., :ci], wp[p].astype(np.float64), r)
                         + bph[p].astype(np.float64), co)
    elif a.up:
        xu = np.zeros((Ho + 2, Wo + 2, ci))
        xu[1:-1, 1:-1] = xpad[1:-1, 1:-1, :ci].repeat(2, 0).repeat(2, 1)
        exact = conv_ref(xu, w.astype(np.float64), rows) + b.astype(np.float64)
    else:
        exact = conv_ref(xpad[..., :ci], w.astype(np.float64), rows) + b.astype(np.float64)
    # every pixel, projected on a random output-channel vector r: a 1-channel conv
    r = rng.standard_normal(a.Cout)
    r[co:] = 0
    proj = np.zeros((Ho, Wo))
    for p in range(n_ph):
        wr = np.einsum("o,oikl->ikl", r, ws[p])[None]
        py, px = divmod(p, 2) if a.up else (0, 0)
        st = 2 if a.up else 1
        proj[py::st, px::st] = conv_ref(xpad, wr)[..., 0] + r @ bias[p]

    xs = np.zeros(s["sizes"]["X"], bfloat16)
    bd = s["x"]["border"]
    xv = xs[s["x"]["off"]:s["x"]["off"] + (H + 2 * bd) * s["x"]["pitch"] * a.Cin]
    xv = xv.reshape(H + 2 * bd, s["x"]["pitch"], a.Cin)
    xv[bd:H + bd, bd:W + bd, :ci] = x

    npu = Npu()
    ks = KernelSet(npu, "conv", Path(a.build))
    npu.sets["conv"] = ks
    st = Stream(ks, "t", Path(a.build) / "insts.bin")
    X = npu.buf("X", xs.nbytes)
    X.write(xs)
    wbuf = np.zeros(s["sizes"]["W"] * 9, np.uint8)
    wbuf[s["w_off"] * 9:s["w_off"] * 9 + packed.size] = packed
    Wb = npu.buf("W", wbuf.nbytes)
    Wb.write(wbuf)
    Y = npu.buf("Y", s["sizes"]["Y"] * 2)
    times = []
    for _ in range(a.runs):
        Y.zero()
        t0 = time.perf_counter()
        st.run(X.bo, Wb.bo, Y.bo)
        times.append((time.perf_counter() - t0) * 1e3)
    y = Y.read(np.uint16).view(bfloat16)[s["y"]["off"]:]
    y = y[:(Ho + 2) * s["y"]["pitch"] * a.Cout].reshape(Ho + 2, s["y"]["pitch"], a.Cout)
    got = y[1 + rows, 1:Wo + 1, :co].astype(np.float64)
    gproj = y[1:Ho + 1, 1:Wo + 1, :co].astype(np.float64) @ r[:co]

    def rel(u, v):
        return float(np.linalg.norm(u - v) / np.linalg.norm(v))

    border = np.concatenate([y[0, :Wo + 2].ravel(), y[Ho + 1, :Wo + 2].ravel(),
                             y[:, 0].ravel(), y[:, Wo + 1].ravel()]).astype(np.float32)
    r_k, r_e, r_p = rel(got, kern), rel(got, exact), rel(gproj, proj)
    ok = bool(np.isfinite(got).all()) and max(r_k, r_p) < GATE and not border.any()
    flops = 2 * Ho * Wo * ci * co * kh * kh
    best = min(times)
    print(f"{H}x{W}x{a.Cin}->{a.Cout} taps={a.taps}{' up' if a.up else ''}: rel_fro vs kernel "
          f"{r_k:.3e}, vs exact {r_e:.3e}, all pixels (projected) {r_p:.3e}; border max {np.abs(border).max():.3g}; "
          f"{best:.2f} ms ({flops / best / 1e9:.2f} TFLOPS)  {'PASS' if ok else 'FAIL'}")
    if not ok:
        perr = np.abs(gproj - proj)
        print(f"  projected: worst pixels {np.argwhere(perr > 0.1 * np.abs(proj).max())[:8].tolist()}")
        err = np.abs(got - kern).max(axis=2)
        bad = np.argwhere(err > 0.1 * np.abs(kern).max())
        print(f"  {len(bad)} bad pixels (row index into {rows.tolist()}); first {bad[:8].tolist()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
