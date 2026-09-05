r"""Test vectors for dn_glue from captured buffers: layer-0 side pool
($OPEN_KERNELS_CAPS/m0d/000119.bo: convw @0, ssm_a @65792, dt_bias @65920,
Wa @66048, Wb @197120) and the layer-0 decode conv state (m0c/000898.bo rows
[3][8192] bf16). xn and qkv are random (N(0,1)); fp64 reference of the glue math.

Writes side.bin (our packed layout), qkv.bin, state.bin, ref_nstate.bin,
ref_vec.bin, run.cfg (paths relative to this directory).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

NCH, HID, NHEAD, HD = 8192, 2048, 32, 128


def silu(x):
    return x / (1 + np.exp(-x))


def main() -> int:
    SIDE = FX.caps("m0d/000119.bo")
    STATE = FX.caps("m0c/000898.bo")
    raw = np.fromfile(SIDE, np.uint8)
    convw = raw[0:65536].view(bfloat16).reshape(4, NCH)
    A = raw[65792:65792 + 128].view(np.float32).copy()
    dtb = raw[65920:65920 + 128].view(np.float32).copy()
    Wa = raw[66048:66048 + 131072].view(bfloat16).reshape(HID, 32)
    Wb = raw[197120:197120 + 131072].view(bfloat16).reshape(HID, 32)
    st = np.fromfile(STATE, np.uint8)[:3 * NCH * 2].view(bfloat16).reshape(3, NCH)

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
    q = l2n(c[:2048].reshape(16, HD))
    k = l2n(c[2048:4096].reshape(16, HD))
    v = c[4096:].reshape(NHEAD, HD)
    vec = np.zeros((NHEAD, 512), np.float32)
    for h in range(NHEAD):
        vec[h, :HD] = k[h // 2]
        vec[h, HD:2 * HD] = q[h // 2]
        vec[h, 2 * HD:3 * HD] = v[h]
        vec[h, 384] = decay[h]
        vec[h, 385] = beta[h]
    nstate = np.vstack([st[1:], qkv.astype(bfloat16)[None, :]])

    # ---- our packed side blob
    # 4 KB elements: [xn][Wa 32][Wb 32][small][convw 16 (2 per tile)]
    side = np.zeros(335872, np.uint8)
    side[0:4096] = xn.view(np.uint8)
    side[4096:4096 + 131072] = Wa.reshape(-1).view(np.uint8)
    side[135168:135168 + 131072] = Wb.reshape(-1).view(np.uint8)
    small = np.zeros(1024, np.float32)
    small[:32] = A
    small[32:64] = dtb
    side[266240:266240 + 4096] = small.view(np.uint8)
    cw = convw.reshape(4, 8, 1024).transpose(1, 0, 2).reshape(-1)     # [tile][4][1024]
    side[270336:270336 + 65536] = cw.view(np.uint8)

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
    print(f"decay[:4]={decay[:4]} beta[:4]={beta[:4]} |q|={np.linalg.norm(q, axis=1)[:2]} v absmax={np.abs(v).max():.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
