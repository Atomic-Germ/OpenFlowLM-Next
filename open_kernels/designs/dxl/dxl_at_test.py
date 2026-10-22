"""dxl's L rows at pos0 against dx decoding those positions on the same cache, bit for bit: python dxl_at_test.py --pos0 P [--compare]."""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from recipes import dxl as DXR  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes.load import load_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(HERE.parents[1] / "model" / "out_k2l"))
    ap.add_argument("--spec", default=str(HERE.parents[1] / "recipes" / "specs" / "k2-horizon-7b.json"))
    ap.add_argument("--dx", default=str(HERE.parent / "dense" / "build_k2_h4096"))
    ap.add_argument("--build", default=str(HERE / "build_k2_l4"))
    ap.add_argument("--pos0", type=int, required=True)
    ap.add_argument("--l", type=int, default=4)
    ap.add_argument("--inputs", type=int, default=4, help="cycle the fixture's xres0..N-1 as the token inputs")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    fx = Path(a.fixture).resolve()
    spec = load_spec(Path(a.spec))
    L0 = for_spec(spec).recipe(spec).layout
    X = DXR.layout(spec, a.l)
    hid = spec.hidden
    tag = f"at{a.pos0}"
    if a.compare:
        y = np.fromfile(fx / f"y_dxl_{tag}.bin", np.float32).reshape(a.l, hid)
        ok = True
        for j in range(a.l):
            dx = np.fromfile(fx / f"y_dx_{tag}_{j}.bin", np.float32)[:hid]
            diff = int((dx.view(np.uint32) != y[j].view(np.uint32)).sum())
            ok &= diff == 0
            print(f"row {j} (position {a.pos0 + j}): {'PASS' if diff == 0 else 'FAIL'}, {diff} of {hid} differ")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    xin = [fx / f"xres{t % a.inputs}.bin" for t in range(a.pos0 + a.l)]
    xl = np.concatenate([np.fromfile(xin[a.pos0 + j], np.float32)[:hid] for j in range(a.l)])
    (fx / f"xres_l_{tag}.bin").write_bytes(xl.astype(np.float32).tobytes())
    d, b = Path(a.dx).resolve().as_posix(), Path(a.build).resolve().as_posix()
    cfg = ["device", f"xclbin dx {d}/final.xclbin", f"kernelx dx dx {d}/insts.bin",
           f"xclbin dxl {b}/final.xclbin", f"kernelx dxl dxl {b}/insts.bin",
           f"buf ptab {L0.PTAB_BYTES} {(fx / 'ptab.bin').as_posix()}",
           f"buf pool0 {L0.POOL_BYTES} {(fx / 'pools' / 'pool_L0.bin').as_posix()}",
           f"buf consts0 {L0.CD_BYTES} {(fx / 'consts_0.bin').as_posix()}",
           f"buf xres {hid * 4}", "buf act0 188416", f"buf state0 {L0.KV_BYTES}",
           f"buf xresl {xl.nbytes} {(fx / f'xres_l_{tag}.bin').as_posix()}",
           f"buf actl {X.AD_BYTES}", f"buf lora0 {X.LORA_BYTES}", f"attngeom {L0.KV_ROW} {L0.PTAB_ROW} 0"]
    for t in range(a.pos0 + a.l):
        cfg += [f"load xres {xin[t].as_posix()}", f"attnpos dx {t}", "run dx pool0 xres consts0 state0 act0 ptab"]
        if t >= a.pos0:
            cfg.append(f"dump xres {(fx / f'y_dx_{tag}_{t - a.pos0}.bin').as_posix()} {hid * 4}")
    cfg += [f"attnrows dxl {a.pos0} {a.l}", "run dxl pool0 xresl consts0 state0 actl ptab lora0",
            f"dump xresl {(fx / f'y_dxl_{tag}.bin').as_posix()} {xl.nbytes}", ""]
    (fx / f"run_dxl_{tag}.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"wrote {fx / f'run_dxl_{tag}.cfg'}: {a.pos0 + a.l} decode steps, then {a.l} rows at {a.pos0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
