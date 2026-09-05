"""Test vectors for moe_axpy (x8) + moe_fin: random y_e/xres/shared/xm, real
shared_expert_gate weight from the captured L0 pack (C:/caps/m0d/000118.bo @8192,
bf16[2048]), router weights from designs/router/ref_out.bin if present; fp64 reference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
PACK = Path("/mnt/c/caps/m0d/000118.bo")
ROUT = HERE.parent / "router" / "ref_out.bin"
N, NE = 2048, 8


def main() -> int:
    rng = np.random.default_rng(0)
    sgw = np.fromfile(PACK, np.uint8)[8192:8192 + N * 2].view(bfloat16).copy()
    if ROUT.is_file():
        rout = np.fromfile(ROUT, np.float32)
    else:
        rout = np.zeros(1024, np.float32)
        w = rng.uniform(0.02, 0.4, NE); rout[264:272] = w / w.sum()
    y = (rng.standard_normal((NE, N)) * 0.1).astype(np.float32)
    xres = (rng.standard_normal(N) * 0.05).astype(np.float32)
    shared = (rng.standard_normal(N) * 0.1).astype(np.float32)
    xm = (rng.standard_normal(N) * 0.5).astype(np.float32).astype(bfloat16)
    w8 = rout[264:272].astype(np.float64)
    dot = xm.astype(np.float64) @ sgw.astype(np.float64)
    sg = 1 / (1 + np.exp(-dot))
    out = xres.astype(np.float64) + (w8[:, None] * y.astype(np.float64)).sum(0) + sg * shared.astype(np.float64)
    for name, arr in (("rout", rout), ("xres", xres), ("shared", shared), ("xm", xm), ("sgw", sgw)):
        (HERE / f"{name}.bin").write_bytes(arr.tobytes())
    for e in range(NE):
        (HERE / f"y{e}.bin").write_bytes(y[e].tobytes())
        eb = np.zeros(1024, np.int32); eb[0] = e
        (HERE / f"e{e}.bin").write_bytes(eb.tobytes())
    (HERE / "ref_out.bin").write_bytes(out.astype(np.float32).tobytes())
    d = "C:/code/phlegm/tools/open-kernels/designs/moe_combine"
    cfg = [
        "device",
        f"xclbin A {d}/build_axpy/final.xclbin", f"kernelx ax A {d}/build_axpy/insts.bin",
        f"xclbin F {d}/build_fin/final.xclbin", f"kernelx fin F {d}/build_fin/insts.bin",
        f"buf rout 4096 {d}/rout.bin",
        *[f"buf y{e} 8192 {d}/y{e}.bin" for e in range(NE)],
        *[f"buf e{e} 4096 {d}/e{e}.bin" for e in range(NE)],
        "buf accA 8192", "buf accB 8192",
        f"buf xres {xres.nbytes} {d}/xres.bin",
        f"buf shared {shared.nbytes} {d}/shared.bin",
        f"buf xm {xm.nbytes} {d}/xm.bin",
        f"buf sgw {sgw.nbytes} {d}/sgw.bin",
        f"buf out {xres.nbytes}",
    ]
    # ping-pong the accumulator between two buffers (in-place in/out on one BO is not guaranteed safe)
    for e in range(NE):
        src, dst = ("accA", "accB") if e % 2 == 0 else ("accB", "accA")
        cfg.append(f"run ax rout y{e} {src} e{e} {dst}")
    last = "accB" if (NE - 1) % 2 == 0 else "accA"
    cfg += [f"run fin {last} xres shared xm sgw out", f"dump out {d}/y_out.bin {xres.nbytes}", ""]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"sg={sg:.5f} out[:4]={out[:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
