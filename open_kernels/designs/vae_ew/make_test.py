r"""make_test: vae_ew's ops at one shape, against float64.

    python make_test.py --test gn --C 128 --H 64 --W 128 [--silu] [--plain-out] --out <dir>
        writes <dir>/spec_<step>.json for each dispatch of the test; build each (IRON env)
    VE_SPEC=<spec_<step>.json> python build_design.py designs/vae_ew/vae_ew.py <build>_<step>
    python make_test.py <same args> --out <dir> --build <build>        (runs <build>_<step>)

Tests (dispatches in order):
    gn       gn_stats, gn_apply        Y = GroupNorm(X) (* SiLU), 32 groups, eps 1e-6
    addgn    add (stats), gn_apply     Y = X + B, then GroupNorm of it from add's sums
    rgba     rgba                      RGBA8 of channels 0-2

Reference: float64 from the bf16 inputs; GroupNorm's output rounded to bf16 before the
SiLU, as diffusers' bf16 decode rounds it. Gates: rel_fro < 1e-2 (the SiLU is tanh-based,
dit_ew's); add bit-exact (bf16(a + b)); rgba within 1 of round((x/2 + 1/2) 255).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
EL, BLOCK = 4096, 19
STEPS = {"gn": ["stats", "apply"], "addgn": ["add", "apply"], "rgba": ["rgba"]}


def bf(x):
    return np.asarray(x, np.float64).astype(bfloat16).astype(np.float64)


def group_norm(x, gamma, beta, groups=32, eps=1e-6):
    H, W, C = x.shape
    g = x.reshape(H * W, groups, C // groups)
    mean = g.mean(axis=(0, 2), keepdims=True)
    var = g.var(axis=(0, 2), keepdims=True)
    return ((g - mean) / np.sqrt(var + eps)).reshape(H, W, C) * gamma + beta


def silu(x):
    return x / (1 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=list(STEPS))
    for d in ("C", "H", "W"):
        ap.add_argument(f"--{d}", type=int, required=True)
    ap.add_argument("--silu", action="store_true")
    ap.add_argument("--plain-out", action="store_true", help="Y as a plain [H*W, C] tensor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--build")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3, help="runs per dispatch (time = best)")
    ap.add_argument("--mean", type=float, default=0.5, help="input offset (GroupNorm cancellation)")
    a = ap.parse_args()
    C, H, W = a.C, a.H, a.W
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = W + 2
    pad_view = {"off": 0, "pitch": P}
    n_pad = (H + 2) * P * C
    y_view = {"off": 0, "pitch": W, "border": 0} if a.plain_out else pad_view
    n_y = H * W * C if a.plain_out else n_pad
    sizes = {"X": n_pad, "B": n_pad, "S": BLOCK * EL, "Y": n_y}
    specs = {
        "stats": {"op": "gn_stats", "C": C, "H": H, "W": W, "a": pad_view, "stats_off": 0},
        "apply": {"op": "gn_apply", "C": C, "H": H, "W": W, "a": pad_view, "y": y_view,
                  "stats_off": 0, "silu": a.silu},
        "add": {"op": "add", "C": C, "H": H, "W": W, "a": pad_view, "b": pad_view,
                "y": pad_view, "stats": True, "stats_off": 0},
        "rgba": {"op": "rgba", "C": C, "H": H, "W": W, "a": pad_view, "y": {"off": 0}},
    }
    if a.test == "rgba":
        sizes["Y"] = H * W * 8
    for step in STEPS[a.test]:
        specs[step]["sizes"] = sizes
        (out / f"spec_{step}.json").write_text(json.dumps(specs[step]))
    if not a.build:
        print(f"wrote specs {STEPS[a.test]} to {out}")
        return 0

    sys.path.insert(0, str(HERE.parents[1] / "harness"))
    from npu_host import KernelSet, Npu, Stream

    rng = np.random.default_rng(a.seed)
    x = (rng.standard_normal((H, W, C)) + a.mean * rng.standard_normal(C)).astype(bfloat16)
    bx = rng.standard_normal((H, W, C)).astype(bfloat16)
    gamma = (1 + 0.2 * rng.standard_normal(C)).astype(bfloat16)
    beta = (0.2 * rng.standard_normal(C)).astype(bfloat16)

    def padded(t):
        z = np.zeros((H + 2, P, C), bfloat16)
        z[1:H + 1, 1:W + 1] = t
        return z

    npu = Npu()
    runs = {}
    for step in STEPS[a.test]:
        d = Path(f"{a.build}_{step}")
        ks = KernelSet(npu, step, d)
        npu.sets[step] = ks
        runs[step] = Stream(ks, step, d / "insts.bin")
    X = npu.buf("X", sizes["X"] * 2)
    X.write(padded(x))
    Bb = npu.buf("B", sizes["B"] * 2)
    Bb.write(padded(bx))
    S = npu.buf("S", sizes["S"] * 2)
    blk = np.zeros(BLOCK * EL, bfloat16)
    blk[EL:EL + C] = gamma
    blk[EL + 512:EL + 512 + C] = beta
    S.write(blk)
    Y = npu.buf("Y", sizes["Y"] * 2)
    Y.zero()
    ms, ok, add_note = {}, True, ""
    for step in STEPS[a.test]:
        if step == "apply" and a.test == "addgn":
            ysum = Y.read(np.uint16)
            ref_sum = padded(bf(x.astype(np.float64) + bx.astype(np.float64)).astype(bfloat16))
            n_bad = int((ysum != ref_sum.reshape(-1).view(np.uint16)).sum())
            ok &= n_bad == 0
            add_note = f"add: {n_bad} values differ from bf16(a + b); "
            X.write(ysum)                       # GroupNorm of the add's output
            Y.zero()
        ms[step] = min(runs[step].run(X.bo, Bb.bo, S.bo, Y.bo) for _ in range(a.runs))
    raw = Y.read(np.uint16)

    def rel(u, v):
        return float(np.linalg.norm(u - v) / np.linalg.norm(v))

    x64 = x.astype(np.float64)
    if a.test == "rgba":
        got = raw.view(np.uint8)[:H * W * 8].reshape(-1, 8192)[:, :4096].reshape(H, W, 4).astype(int)
        ref = np.clip(np.round((x64[..., :3] / 2 + 0.5) * 255), 0, 255)
        err = int(np.abs(got[..., :3] - ref).max())
        ok = err <= 1 and (got[..., 3] == 255).all()
        print(f"rgba {H}x{W}: max |err| {err}  {ms['rgba']:.2f} ms  {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    src = x64 if a.test == "gn" else bf(x64 + bx.astype(np.float64))
    ref = bf(group_norm(src, gamma.astype(np.float64), beta.astype(np.float64)))
    if a.silu:
        ref = silu(ref)
    y = raw.view(bfloat16)
    got = (y.reshape(H, W, C) if a.plain_out else y.reshape(H + 2, P, C)[1:H + 1, 1:W + 1]) \
        .astype(np.float64)
    r = rel(got, ref)
    ok &= bool(np.isfinite(got).all()) and r < (2e-2 if a.silu else 1e-2)
    if not a.plain_out:
        yp = y.reshape(H + 2, P, C).astype(np.float32)
        border = np.concatenate([yp[0].ravel(), yp[-1].ravel(), yp[:, 0].ravel(), yp[:, W + 1].ravel()])
        ok &= not border.any()
    t = " + ".join(f"{k} {v:.2f} ms" for k, v in ms.items())
    print(f"{a.test} {H}x{W}x{C}{' silu' if a.silu else ''}{' plain' if a.plain_out else ''}: "
          f"{add_note}rel_fro {r:.3e}  {t}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        bad = np.argwhere(np.abs(got - ref).max(axis=2) > 0.1)
        print(f"  {len(bad)} bad pixels, first {bad[:8].tolist()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
