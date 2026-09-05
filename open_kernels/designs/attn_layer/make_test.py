r"""Test vectors for attn_l from designs/attn_chain's layer-2 inputs and references
(position 11 of the captured 3LiF decode step; no model needed, runs on Windows).
The weight arg is the captured layer-2 pool itself (C:/caps/m0d/000123.bo).

    python make_test.py ; open-qwen-npu npu designs/attn_layer/run.cfg ; python compare.py
(build: ATTN_POS=11 python build_design.py designs/attn_layer/attn_l.py designs/attn_layer/build_pos11)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
AC = HERE.parent / "attn_chain"
sys.path.insert(0, str(HERE))
from layout import AA_BYTES, CA_BYTES, CA_LNW, CA_META, CA_POSTLN, KV_BYTES  # noqa: E402

D = "C:/code/phlegm/tools/open-kernels/designs"
POOL = "C:/caps/m0d/000123.bo"


def main() -> int:
    consts = np.zeros(CA_BYTES, np.uint8)
    consts[CA_LNW:CA_LNW + 4096] = np.fromfile(AC / "lnw.bin", np.uint8)
    consts[CA_POSTLN:CA_POSTLN + 4096] = np.fromfile(AC / "postln.bin", np.uint8)
    consts[CA_META:CA_META + 2048] = np.fromfile(AC / "meta.bin", np.uint8)
    (HERE / "consts.bin").write_bytes(consts.tobytes())
    for n in ("ref_knew", "ref_vnew", "ref_og", "ref_xres", "ref_xm", "ref_xres_replica"):
        (HERE / f"{n}.bin").write_bytes((AC / f"{n}.bin").read_bytes())
    d = f"{D}/attn_layer"
    cfg = [
        "device",
        f"xclbin A {d}/build_pos11/final.xclbin", f"kernelx al A {d}/build_pos11/insts.bin",
        f"buf pool 536870912 {POOL}",
        f"buf xres 8192 {D}/attn_chain/xres.bin",
        f"buf consts {CA_BYTES} {d}/consts.bin",
        f"buf kv {KV_BYTES} {D}/attn_chain/kv.bin",
        f"buf act {AA_BYTES}", "buf hdr 20480",
        "run al pool xres consts kv act hdr",
        f"dump act {d}/y_act.bin {AA_BYTES}",
        f"dump hdr {d}/y_hdr.bin 20480",
        "run al pool xres consts kv act hdr",
        "run al pool xres consts kv act hdr",
        "",
    ]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print("wrote consts.bin run.cfg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
