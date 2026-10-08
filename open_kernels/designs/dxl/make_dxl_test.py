r"""One layer of the L-row pass against L decode steps, on a make_decode fixture.

    python model/make_decode.py --model-dir <K2 dir> --layers 1 --tokens 4 --out model/out_k2l
    run_kernel model/out_k2l/run_decode.cfg            # dx, one token at a time -> y_res0_t<j>.bin
    python designs/dxl/make_dxl_test.py --fixture model/out_k2l --build designs/dxl/build_k2_l4 --l 4
    run_kernel model/out_k2l/run_dxl.cfg               # dxl, the L rows in one dispatch
    python designs/dxl/make_dxl_test.py --fixture model/out_k2l --l 4 --compare

Rows j = 0 .. L-1 at positions pos0 + j (pos0 = 0: the fixture's tokens). Row j must equal
dx's layer output for token j bit for bit, and match the fp64 replica's ref_res0_t<j>.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from recipes import dxl as DXR  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes.load import load_spec  # noqa: E402


def tok_file(fx: Path, stem: str, j: int) -> Path:
    return fx / (f"{stem}.bin" if j == 0 else f"{stem}_t{j}.bin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--build", default=None)
    ap.add_argument("--l", type=int, default=4)
    ap.add_argument("--spec", default=str(HERE.parents[1] / "recipes" / "specs" / "k2-horizon-7b.json"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    fx = Path(a.fixture).resolve()
    spec = load_spec(Path(a.spec))
    R = for_spec(spec).recipe(spec)
    X = DXR.layout(spec, a.l)
    hid = spec.hidden

    if a.compare:
        y = np.fromfile(fx / "y_dxl_res.bin", np.float32).reshape(a.l, hid)
        ok = True
        for j in range(a.l):
            dx = np.fromfile(tok_file(fx, "y_res0", j), np.float32)
            ref = np.fromfile(tok_file(fx, "ref_res0", j), np.float32).astype(np.float64)
            diff = int((dx.view(np.uint32) != y[j].view(np.uint32)).sum())
            g = y[j].astype(np.float64)
            corr = float(np.corrcoef(g, ref)[0, 1])
            rel = float(np.abs(g - ref).max() / (np.abs(ref).max() + 1e-30))
            good = diff == 0
            ok &= good
            print(f"row {j}: {'PASS' if good else 'FAIL'} vs dx: {diff} of {hid} differ; "
                  f"vs fp64: corr {corr:.6f} maxrel {rel:.2e}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    xres = np.concatenate([np.fromfile(fx / f"xres{j}.bin", np.float32)[:hid] for j in range(a.l)])
    (fx / "xres_l.bin").write_bytes(xres.astype(np.float32).tobytes())
    b = Path(a.build).resolve().as_posix()
    L0 = R.layout
    cfg = ["device", f"xclbin dxl {b}/final.xclbin", f"kernelx dxl dxl {b}/insts.bin",
           f"buf ptab {L0.PTAB_BYTES} {(fx / 'ptab.bin').as_posix()}",
           f"buf pool0 {L0.POOL_BYTES} {(fx / 'pools' / 'pool_L0.bin').as_posix()}",
           f"buf consts0 {L0.CD_BYTES} {(fx / 'consts_0.bin').as_posix()}",
           f"buf xresl {xres.nbytes} {(fx / 'xres_l.bin').as_posix()}",
           f"buf actl {X.AD_BYTES}", f"buf state0 {L0.KV_BYTES}",
           f"attngeom {L0.KV_ROW} {L0.PTAB_ROW} 0", f"attnrows dxl 0 {a.l}"]
    for r in range(a.runs):
        if r:
            cfg.append(f"load xresl {(fx / 'xres_l.bin').as_posix()}")
        cfg.append("run dxl pool0 xresl consts0 state0 actl ptab")
    cfg += [f"dump xresl {(fx / 'y_dxl_res.bin').as_posix()} {xres.nbytes}",
            f"dump actl {(fx / 'y_dxl_act.bin').as_posix()} {X.AD_BYTES}", ""]
    (fx / "run_dxl.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"wrote {fx / 'run_dxl.cfg'}: L={a.l}, act {X.AD_BYTES} B, {a.runs} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
