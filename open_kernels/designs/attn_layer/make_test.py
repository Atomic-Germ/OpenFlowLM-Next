r"""Test vectors for attn_l from designs/attn_chain's layer-2 inputs and references
(position 11 of the captured 3LiF decode step; no model needed, runs on Windows).
The weight arg is the captured layer-2 pool itself ($OPEN_KERNELS_CAPS/m0d/000123.bo).
Paths in run.cfg are relative to this directory.

    python make_test.py ; run_kernel run.cfg ; python compare.py
(build: ATTN_POS=11 python build_design.py designs/attn_layer/attn_l.py designs/attn_layer/build_pos11)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AC = HERE.parent / "attn_chain"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402
from layout import AA_BYTES, CA_BYTES, CA_LNW, CA_META, CA_POSTLN, KV_BYTES  # noqa: E402

D = ".."                                   # designs/, relative to this cfg


def main() -> int:
    consts = np.zeros(CA_BYTES, np.uint8)
    consts[CA_LNW:CA_LNW + 4096] = np.fromfile(AC / "lnw.bin", np.uint8)
    consts[CA_POSTLN:CA_POSTLN + 4096] = np.fromfile(AC / "postln.bin", np.uint8)
    consts[CA_META:CA_META + 2048] = np.fromfile(AC / "meta.bin", np.uint8)
    (HERE / "consts.bin").write_bytes(consts.tobytes())
    for n in ("ref_knew", "ref_vnew", "ref_og", "ref_xres", "ref_xm", "ref_xres_replica"):
        (HERE / f"{n}.bin").write_bytes((AC / f"{n}.bin").read_bytes())
    cfg = [
        "device",
        "xclbin A build_pos11/final.xclbin", "kernelx al A build_pos11/insts.bin",
        f"buf pool 536870912 {FX.caps_cfg('m0d/000123.bo')}",
        f"buf xres 8192 {D}/attn_chain/xres.bin",
        f"buf consts {CA_BYTES} consts.bin",
        f"buf kv {KV_BYTES} {D}/attn_chain/kv.bin",
        f"buf act {AA_BYTES}", "buf hdr 20480",
        "run al pool xres consts kv act hdr",
        f"dump act y_act.bin {AA_BYTES}",
        "dump hdr y_hdr.bin 20480",
        "run al pool xres consts kv act hdr",
        "run al pool xres consts kv act hdr",
        "",
    ]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print("wrote consts.bin run.cfg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
