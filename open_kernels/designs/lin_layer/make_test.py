r"""Test vectors for lin_a / lin_c from designs/layer_chain's layer-0 inputs and
references (no model needed; runs on Windows). The chain under test is

    run la  pool xres consts state act vec      (ln -> gemv qkv|z -> glue)
    run dn  sin vec sout o                      (designs/deltanet, unchanged)
    run lc  wout o consts act xres hdr          (post -> gemv out -> ln+residual)

against the fp64 replica references layer_chain/make_chain.py wrote
(ref_xn, ref_xres, ref_xm, ref_S, ref_cs). The weight arg of lin_a is the
captured layer-0 pool itself (qkv/z at their pool offsets).

    python make_test.py ; run_kernel run.cfg ; python compare.py

The pool is the captured layer-0 pool ($OPEN_KERNELS_CAPS/m0d/blob_536870912_*.bin);
paths in run.cfg are relative to this directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LC = HERE.parent / "layer_chain"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402
from layout import A_BYTES, C_BYTES, C_LNW, C_NW, C_POSTLN, C_WA, H_BYTES, POOL_BYTES  # noqa: E402

D = ".."                                   # designs/, relative to this cfg
POOL_CAP = "m0d/blob_536870912_836fd8e49f35a0b6.bin"


def main() -> int:
    consts = np.zeros(C_BYTES, np.uint8)
    consts[C_LNW:C_LNW + 4096] = np.fromfile(LC / "lnw.bin", np.uint8)
    consts[C_WA:C_NW] = np.fromfile(LC / "side_glue.bin", np.uint8)[4096:]        # Wa Wb small convw
    consts[C_NW:C_NW + 4096] = np.fromfile(LC / "nw.bin", np.uint8)
    consts[C_POSTLN:C_POSTLN + 4096] = np.fromfile(LC / "postln.bin", np.uint8)
    (HERE / "consts.bin").write_bytes(consts.tobytes())
    (HERE / "state.bin").write_bytes((LC / "state.bin").read_bytes())
    for n in ("ref_xn", "ref_xres", "ref_xm", "ref_S", "ref_cs"):
        (HERE / f"{n}.bin").write_bytes((LC / f"{n}.bin").read_bytes())
    wout = (LC / "w_out.bin").stat().st_size
    cfg = [
        "device",
        "xclbin A build_a/final.xclbin", "kernelx la A build_a/insts.bin",
        f"xclbin N {D}/deltanet/build/final.xclbin", f"kernelx dn N {D}/deltanet/build/insts.bin",
        "xclbin C build_c/final.xclbin", "kernelx lc C build_c/insts.bin",
        f"buf pool {POOL_BYTES} {FX.caps_cfg(POOL_CAP)}",
        f"buf xres 8192 {D}/layer_chain/x_res.bin",
        f"buf consts {C_BYTES} consts.bin",
        "buf state 49152 state.bin",
        f"buf act {A_BYTES}", "buf vec 65536",
        f"buf sin 2097152 {D}/layer_chain/s_in.bin", "buf sout 2097152", "buf o 16384",
        f"buf wout {wout} {D}/layer_chain/w_out.bin",
        f"buf hdr {H_BYTES}",
        "run la pool xres consts state act vec",
        "run dn sin vec sout o",
        "run lc wout o consts act xres hdr",
        f"dump act y_act.bin {A_BYTES}",
        "dump state y_state.bin 49152",
        "dump vec y_vec.bin 65536",
        "dump sout y_S.bin 2097152",
        f"dump hdr y_hdr.bin {H_BYTES}",
        # warm timing (the conv state was updated in place: reload it first)
        "load state state.bin",
        "run la pool xres consts state act vec",
        "run dn sin vec sout o",
        "run lc wout o consts act xres hdr",
        "load state state.bin",
        "run la pool xres consts state act vec",
        "run dn sin vec sout o",
        "run lc wout o consts act xres hdr",
        "",
    ]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print("wrote consts.bin state.bin run.cfg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
