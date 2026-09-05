r"""Unit test for ax (the whole full-attention layer + MoE) at position 11, from
attn_chain's layer-2 inputs and references (the captured 3LiF decode step; no
model needed, runs on Windows). The captured 11-row cache is re-laid as the
interleaved rows of layout.py, the position record table comes from
layout.ptab(), and the driver's `attnpos` patches the (position-1 placeholder)
ax0 stream to 11:

    attnpos ax0 11 ; run ax0 pool xres consts kv act ptab ; moeroute2 ax1 act <idx> ; run ax1 ...

References: attn_chain's bf16-faithful og / new cache row / xres / xm. The MoE
part has no layer-2 reference (moe_chain's is layer 0): it runs on the captured
pool's experts with the captured pack's router W / sgw, and its routing is only
checked for sanity. The layer runs three times (xres and kv reloaded) to check
that the patched stream replays cleanly; the outputs must be identical.

    python make_test_ax.py ; run_kernel run_ax.cfg ; python compare_ax.py
(build: for p in 0 1: AX_PART=$p python build_design.py designs/layer_x/ax.py designs/layer_x/build_ax$p)

Captured layer-2 buffers come from $OPEN_KERNELS_CAPS; paths in run_ax.cfg are
relative to this directory.
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
from layout import (AA_BYTES, AA_ROUT, CA_BYTES, CA_LNW, CA_META, CA_POSTLN, CA_RW, CA_SGW, KV_BYTES, KV_ROW,  # noqa: E402
                    POOL_BYTES, PTAB_BYTES, ptab)

D = ".."                                # designs/, relative to this cfg
POOL = "m0d/000123.bo"                  # the captured layer-2 pool (q/k/v/gate/o, experts)
PACK = "m0d/000124.bo"                  # [lnw | postln | sgw | router W ...]
SIDE = "m0d/000125.bo"                  # q_norm / k_norm (effective) at 128 / 640
POS = 11
CAP_V_OFF = 1_073_152                   # FLM's 3 MB pack: K rows @0, V rows here, 1 KB per row


def main() -> int:
    pack = np.fromfile(FX.caps(PACK), np.uint8)
    side = np.fromfile(FX.caps(SIDE), np.uint8)
    consts = np.zeros(CA_BYTES, np.uint8)
    consts[CA_LNW:CA_LNW + 4096] = np.fromfile(AC / "lnw.bin", np.uint8)
    consts[CA_POSTLN:CA_POSTLN + 4096] = np.fromfile(AC / "postln.bin", np.uint8)
    consts[CA_META:CA_META + 512] = side[128:640]
    consts[CA_META + 512:CA_META + 1024] = side[640:1152]
    consts[CA_RW:CA_RW + 1048576] = pack[12288:12288 + 1048576]
    consts[CA_SGW:CA_SGW + 4096] = pack[8192:12288]
    (HERE / "consts_ax.bin").write_bytes(consts.tobytes())
    cap = np.fromfile(AC / "kv.bin", np.uint8)
    kv = np.zeros((KV_BYTES // KV_ROW, KV_ROW), np.uint8)
    for t in range(POS):
        kv[t, :1024] = cap[t * 1024:(t + 1) * 1024]
        kv[t, 1024:] = cap[CAP_V_OFF + t * 1024:CAP_V_OFF + (t + 1) * 1024]
    (HERE / "kv_ax.bin").write_bytes(kv.tobytes())
    (HERE / "ptab.bin").write_bytes(ptab().tobytes())
    runs = [f"attnpos ax0 {POS}",
            "run ax0 pool xres consts kv act ptab",
            f"moeroute2 ax1 act {AA_ROUT + 1024}",
            "run ax1 pool xres consts kv act ptab"]
    reload = [f"load xres {D}/attn_chain/xres.bin", "load kv kv_ax.bin"]

    def dumps(sfx):
        return [f"dump act y_ax_act{sfx}.bin {AA_BYTES}",
                f"dump kv y_ax_kvnew{sfx}.bin {KV_ROW} {POS * KV_ROW}",
                f"dump xres y_ax_xres{sfx}.bin 8192"]

    cfg = [
        "device",
        "xclbin Y build_ax0/final.xclbin",
        "kernelx ax0 Y build_ax0/insts.bin", "kernelx ax1 Y build_ax1/insts.bin",
        f"buf pool {POOL_BYTES} {FX.caps_cfg(POOL)}",
        f"buf xres 8192 {D}/attn_chain/xres.bin",
        f"buf consts {CA_BYTES} consts_ax.bin",
        f"buf kv {KV_BYTES} kv_ax.bin",
        f"buf act {AA_BYTES}",
        f"buf ptab {PTAB_BYTES} ptab.bin",
        *runs, *dumps(""),
        *reload, *runs,
        *reload, *runs, *dumps("_r3"),
        "",
    ]
    (HERE / "run_ax.cfg").write_text("\n".join(cfg), newline="\n")
    print("wrote consts_ax.bin kv_ax.bin ptab.bin run_ax.cfg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
