r"""Test vectors for gemv_t2n: random ternary weights as a t2 pool (std_perm band order, 2176 B chunks),
eight random rows, the host's tables (t2n_host.py) and the fp64 reference of W @ x for x as the table
holds it, so the comparison isolates the kernel.

    python make_test.py [--n 6144] [--k 5120] [--out DIR] [--runs 20]
    run_kernel.exe DIR/run.cfg   (or run_null.cfg: the T2N_NULL=1 build, DMA only)
    python make_test.py --compare [--n ..] [--k ..] [--out DIR]

The cfg names designs/gemv_t2n/build_n<n>_k<k> (and build_n<n>_k<k>_null).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from t2n_host import effective_x, reduce, tables  # noqa: E402
from t2n_plan import TOKENS, plan  # noqa: E402
from t2_pack import QK, random_pq2_codes  # noqa: E402


def _lm_head_t2_make_test():
    import importlib.util
    sp = importlib.util.spec_from_file_location("lm_head_t2_make_test", HERE.parent / "lm_head_t2" / "make_test.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


pack_head = _lm_head_t2_make_test().pack_head      # the same band order and chunk as the lm head pool


def reference(codes, s, x, rows=8192):
    out = np.empty((x.shape[0], codes.shape[0]))
    for r0 in range(0, codes.shape[0], rows):
        sc = np.repeat(s[r0:r0 + rows].astype(np.float32).astype(bfloat16).astype(np.float64), QK, axis=1)
        out[:, r0:r0 + rows] = x @ (codes[r0:r0 + rows].astype(np.float64) * sc - sc).T
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--out", type=Path, default=HERE / "test")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = a.out.resolve()
    p = plan(a.n, a.k)
    if a.compare:
        ref = np.fromfile(out / "ref.bin", np.float64).reshape(TOKENS, a.n)
        got = reduce(p, np.fromfile(out / f"y{a.tag}.bin", np.float32).astype(np.float64))
        if a.tag == "_null":
            print(f"null: zeros={not got.any()}")
            return 0
        rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
        cos = [float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30)) for g, r in zip(got, ref)]
        ok = min(cos) > 0.99999 and rel < 1e-3 and np.isfinite(got).all()
        print(f"{'PASS' if ok else 'FAIL'} N={a.n} K={a.k} cos min {min(cos):.9f} maxrel {rel:.2e} "
              f"argmax {[int(np.argmax(g)) for g in got]} vs {[int(np.argmax(r)) for r in ref]}")
        return 0 if ok else 1
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    codes, s = random_pq2_codes(a.n, a.k, rng)
    x = rng.standard_normal((TOKENS, a.k)) * np.exp(rng.uniform(-3, 3, (TOKENS, 1)))   # rows of different scale
    w = pack_head(codes, s)
    tab = tables(p, x)
    ref = reference(codes, s, effective_x(x))
    (out / "w.bin").write_bytes(w.tobytes())
    (out / "t.bin").write_bytes(tab.tobytes())
    (out / "ref.bin").write_bytes(ref.tobytes())
    yb = p["y_floats"] * 4
    for tag in ("", "_null"):
        b = (HERE / f"build_n{a.n}_k{a.k}{tag}").as_posix()
        cfg = ["device", f"xclbin G {b}/final.xclbin", f"kernelx k G {b}/insts.bin",
               f"buf w {w.nbytes} {(out / 'w.bin').as_posix()}", f"buf t {tab.nbytes} {(out / 't.bin').as_posix()}",
               f"buf y {yb}"] + ["run k w t y"] * a.runs + [f"dump y {(out / f'y{tag}.bin').as_posix()} {yb}", ""]
        (out / f"run{tag}.cfg").write_text("\n".join(cfg), newline="\n")
    items = sum(len(i) for i in p["items"])
    print(f"N={a.n} K={a.k}: w {w.nbytes} B, {items} items over {len(p['slices'])} K slices, "
          f"tables {tab.nbytes} B, y {yb} B -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
