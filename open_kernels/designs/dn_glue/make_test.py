r"""Test vectors for dn_glue from captured buffers: layer-0 side pool
($OPEN_KERNELS_CAPS/m0d/000119.bo: convw @0, ssm_a @65792, dt_bias @65920,
Wa @66048, Wb @197120) and the layer-0 decode conv state (m0c/000898.bo rows
[3][8192] bf16). xn and qkv are random (N(0,1)); fp64 reference of the glue math.

DNGLUE_NHEAD=16 builds the Qwen3.5 2B / 0.8B geometry out of the same capture: the first
16 of the 32 value heads, so 6144 conv channels instead of 8192, ONE value head per key
head instead of two, and the alpha / beta projection zero-padded from 16 columns to the
accumulator's 32 lanes -- which is exactly what recipes/qwen35.py's `transpose` op with
`dst_rows` writes. Nothing about the capture is 16-head; the point of the case is that the
kernels index a 16-head record set correctly, and the fp64 reference is recomputed for it.

Writes side.bin (our packed layout), qkv.bin, state.bin, ref_nstate.bin,
ref_vec.bin, run.cfg (paths relative to this directory).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

NHEAD = int(os.environ.get("DNGLUE_NHEAD", 32))       # dn_glue.py reads the same knob
HID, HD, KEY_WIDTH = 2048, 128, 2048
NCH = 2 * KEY_WIDTH + NHEAD * HD                     # 8192 at 32 heads, 6144 at 16
KEY_HEADS = KEY_WIDTH // HD                          # 16
GRP = NHEAD // KEY_HEADS                             # value heads per key head: 2 or 1
AB_LANES = 32                                        # dn_glue.h's kV; a 16-head W is padded to it
NT = NCH // 1024
SIDE_BYTES = (1 + 2 * (HID * AB_LANES * 2 // 4096) + 1 + 2 * NT) * 4096


def silu(x):
    return x / (1 + np.exp(-x))


def main() -> int:
    SIDE = FX.caps("m0d/000119.bo")
    STATE = FX.caps("m0c/000898.bo")
    raw = np.fromfile(SIDE, np.uint8)
    # the capture is the 27B's 32-head layer: 8192 conv channels, [HID, 32] projections.
    # At NHEAD 16 keep q, k and the FIRST 16 value heads of each.
    convw = raw[0:65536].view(bfloat16).reshape(4, 8192)[:, :NCH]
    A = raw[65792:65792 + 128].view(np.float32)[:NHEAD].copy()
    dtb = raw[65920:65920 + 128].view(np.float32)[:NHEAD].copy()
    Wa = raw[66048:66048 + 131072].view(bfloat16).reshape(HID, 32)[:, :NHEAD]
    Wb = raw[197120:197120 + 131072].view(bfloat16).reshape(HID, 32)[:, :NHEAD]
    st = np.fromfile(STATE, np.uint8)[:3 * 8192 * 2].view(bfloat16).reshape(3, 8192)[:, :NCH].copy()

    rng = np.random.default_rng(0)
    xn = rng.standard_normal(HID).astype(np.float32).astype(bfloat16)
    qkv = rng.standard_normal(NCH).astype(np.float32)

    # ---- reference (fp64)
    x64 = xn.astype(np.float64)
    alpha = x64 @ Wa.astype(np.float64)
    betal = x64 @ Wb.astype(np.float64)
    decay = np.exp(A.astype(np.float64) * np.log1p(np.exp(alpha + dtb)))
    beta = 1 / (1 + np.exp(-betal))
    seq = np.vstack([st.astype(np.float64), qkv.astype(np.float64)[None, :]])
    c = silu((convw.astype(np.float64) * seq).sum(0))
    def l2n(a):
        return a / np.sqrt((a ** 2).sum(-1, keepdims=True) + 1e-6)
    q = l2n(c[:KEY_WIDTH].reshape(KEY_HEADS, HD))
    k = l2n(c[KEY_WIDTH:2 * KEY_WIDTH].reshape(KEY_HEADS, HD))
    v = c[2 * KEY_WIDTH:].reshape(NHEAD, HD)
    vec = np.zeros((NHEAD, 512), np.float32)
    for h in range(NHEAD):
        vec[h, :HD] = k[h // GRP]
        vec[h, HD:2 * HD] = q[h // GRP]
        vec[h, 2 * HD:3 * HD] = v[h]
        vec[h, 384] = decay[h]
        vec[h, 385] = beta[h]
    nstate = np.vstack([st[1:], qkv.astype(bfloat16)[None, :]])

    # ---- our packed side blob
    # 4 KB elements: [xn][Wa 32][Wb 32][small][convw 2 per conv tile]. The projections are
    # ALWAYS [HID, 32]: at NHEAD 16 columns 16..31 are zero (recipes/pack.py `dst_rows`).
    ab = HID * AB_LANES * 2
    side = np.zeros(SIDE_BYTES, np.uint8)
    side[0:4096] = xn.view(np.uint8)
    for off, W in ((4096, Wa), (4096 + ab, Wb)):
        pad = np.zeros((HID, AB_LANES), bfloat16)
        pad[:, :NHEAD] = W
        side[off:off + ab] = pad.reshape(-1).view(np.uint8)
    o_small = 4096 + 2 * ab
    small = np.zeros(1024, np.float32)
    small[:NHEAD] = A
    small[NHEAD:2 * NHEAD] = dtb
    side[o_small:o_small + 4096] = small.view(np.uint8)
    o_conv = o_small + 4096
    cw = convw.reshape(4, NT, 1024).transpose(1, 0, 2).reshape(-1)     # [tile][4][1024]
    side[o_conv:o_conv + cw.nbytes] = cw.view(np.uint8)

    (HERE / "side.bin").write_bytes(side.tobytes())
    (HERE / "qkv.bin").write_bytes(qkv.tobytes())
    (HERE / "state.bin").write_bytes(st.tobytes())
    (HERE / "ref_nstate.bin").write_bytes(nstate.tobytes())
    (HERE / "ref_vec.bin").write_bytes(vec.tobytes())
    cfg = "\n".join([
        "device",
        "xclbin G build/final.xclbin",
        "kernelx k G build/insts.bin",
        f"buf side {side.nbytes} side.bin",
        f"buf qkv {qkv.nbytes} qkv.bin",
        f"buf state {st.nbytes} state.bin",
        f"buf nstate {st.nbytes}",
        f"buf vec {vec.nbytes}",
        "run k side qkv state nstate vec",
        "run k side qkv state nstate vec",
        f"dump nstate y_nstate.bin {st.nbytes}",
        f"dump vec y_vec.bin {vec.nbytes}",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"NHEAD={NHEAD} NCH={NCH} side={SIDE_BYTES} B  decay[:4]={decay[:4]} beta[:4]={beta[:4]} "
          f"|q|={np.linalg.norm(q, axis=1)[:2]} v absmax={np.abs(v).max():.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
