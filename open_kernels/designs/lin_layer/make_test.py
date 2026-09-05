r"""Test vectors for lin_a / lin_c from designs/layer_chain's layer-0 inputs and
references (no model needed; runs on Windows). The chain under test is

    run la  pool xres consts state act vec      (ln -> gemv qkv|z -> glue)
    run dn  sin vec sout o                      (designs/deltanet, unchanged)
    run lc  wout o consts act xres hdr          (post -> gemv out -> ln+residual)

against the fp64 replica references layer_chain/make_chain.py wrote
(ref_xn, ref_xres, ref_xm, ref_S, ref_cs). The weight arg of lin_a is the
captured layer-0 pool itself (qkv/z at their pool offsets).

    python make_test.py ; open-qwen-npu npu designs/lin_layer/run.cfg ; python compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LC = HERE.parent / "layer_chain"
sys.path.insert(0, str(HERE))
from layout import A_BYTES, C_BYTES, C_LNW, C_NW, C_POSTLN, C_WA, H_BYTES, POOL_BYTES  # noqa: E402

D = "C:/code/phlegm/tools/open-kernels/designs"
POOL = "C:/caps/m0d/blob_536870912_836fd8e49f35a0b6.bin"


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
    d = f"{D}/lin_layer"
    cfg = [
        "device",
        f"xclbin A {d}/build_a/final.xclbin", f"kernelx la A {d}/build_a/insts.bin",
        f"xclbin N {D}/deltanet/build/final.xclbin", f"kernelx dn N {D}/deltanet/build/insts.bin",
        f"xclbin C {d}/build_c/final.xclbin", f"kernelx lc C {d}/build_c/insts.bin",
        f"buf pool {POOL_BYTES} {POOL}",
        f"buf xres 8192 {D}/layer_chain/x_res.bin",
        f"buf consts {C_BYTES} {d}/consts.bin",
        f"buf state 49152 {d}/state.bin",
        f"buf act {A_BYTES}", "buf vec 65536",
        f"buf sin 2097152 {D}/layer_chain/s_in.bin", "buf sout 2097152", "buf o 16384",
        f"buf wout {wout} {D}/layer_chain/w_out.bin",
        f"buf hdr {H_BYTES}",
        "run la pool xres consts state act vec",
        "run dn sin vec sout o",
        "run lc wout o consts act xres hdr",
        f"dump act {d}/y_act.bin {A_BYTES}",
        f"dump state {d}/y_state.bin 49152",
        f"dump vec {d}/y_vec.bin 65536",
        f"dump sout {d}/y_S.bin 2097152",
        f"dump hdr {d}/y_hdr.bin {H_BYTES}",
        # warm timing (the conv state was updated in place: reload it first)
        f"load state {d}/state.bin",
        "run la pool xres consts state act vec",
        "run dn sin vec sout o",
        "run lc wout o consts act xres hdr",
        f"load state {d}/state.bin",
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
