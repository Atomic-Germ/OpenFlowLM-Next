r"""Test vectors for moe_experts from designs/moe_chain's layer-0 inputs (no model
needed; runs on Windows):

  wexp.bin  = per routed expert k: w_up{k} | w_gate{k} | w_down{k}   (8 x 1,966,080 B)
              then the shared expert w_su | w_sg | w_sd (1,966,080 B)
  hdr.bin   = 20480 B: [moe_chain/y_xm.bin (the ln kernel's bf16 xm) | moe_chain/y_rout.bin
              (the router kernel's f32[1024], w[8] at 264) | sgw.bin | xres.bin]
  ref_acc.bin = the routed sum alone, fp64 from the same pool bytes
                (gemv_q4/make_test.reference), bf16 h as the kernel (diagnostic)
  ref_out.bin = moe_chain/ref_out.bin: xres + acc + sigmoid(xm.sgw) * shared -- the
                block output the kernel emits

    python make_test.py ; run_kernel run.cfg ; python compare.py

No captured buffers are read directly (moe_chain's outputs are); paths in
run.cfg are relative to this directory.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
MC = HERE.parent / "moe_chain"
NE = 8

spec = importlib.util.spec_from_file_location("gemv_make_test", HERE.parent / "gemv_q4" / "make_test.py")
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)


def silu(x):
    return x / (1 + np.exp(-x))


def main() -> int:
    xm_b = np.fromfile(MC / "y_xm.bin", np.uint8)[:4096].view(bfloat16)
    rout = np.fromfile(MC / "y_rout.bin", np.float32)
    w8 = rout[264:272].astype(np.float64)
    acc = np.zeros(2048, np.float64)
    parts = []
    for k in range(NE):
        up = np.fromfile(MC / f"w_up{k}.bin", np.uint8)
        gt = np.fromfile(MC / f"w_gate{k}.bin", np.uint8)
        dn = np.fromfile(MC / f"w_down{k}.bin", np.uint8)
        assert len(up) == len(gt) == len(dn) == 655_360
        u = gm.reference(up, xm_b, 512, 2048, 4).astype(np.float64)
        g = gm.reference(gt, xm_b, 512, 2048, 4).astype(np.float64)
        h = (silu(g) * u).astype(np.float32).astype(bfloat16)
        y = gm.reference(dn, h, 2048, 512, 4).astype(np.float64)
        acc += w8[k] * y
        parts += [up.tobytes(), gt.tobytes(), dn.tobytes()]
        print(f"expert slot {k}: w={w8[k]:.4f} |y|max={np.abs(y).max():.4g}", flush=True)
    parts += [(MC / n).read_bytes() for n in ("w_su.bin", "w_sg.bin", "w_sd.bin")]
    (HERE / "wexp.bin").write_bytes(b"".join(parts))
    hdr = np.zeros(20480, np.uint8)
    hdr[:4096] = xm_b.view(np.uint8)
    hdr[4096:8192] = rout.view(np.uint8)
    hdr[8192:12288] = np.fromfile(MC / "sgw.bin", np.uint8)[:4096]
    hdr[12288:20480] = np.fromfile(MC / "xres.bin", np.uint8)[:8192]
    (HERE / "hdr.bin").write_bytes(hdr.tobytes())
    (HERE / "ref_acc.bin").write_bytes(acc.astype(np.float32).tobytes())
    (HERE / "ref_out.bin").write_bytes((MC / "ref_out.bin").read_bytes())
    cfg = "\n".join([
        "device",
        "xclbin E build/final.xclbin",
        "kernelx me E build/insts.bin",
        f"buf wexp {(NE + 1) * 1_966_080} wexp.bin",
        "buf hdr 20480 hdr.bin",
        "buf out 8192",
        "run me wexp hdr out",
        "run me wexp hdr out",
        "run me wexp hdr out",
        "dump out y_out.bin 8192",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"ref_acc[:4]={acc[:4]} absmax={np.abs(acc).max():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
