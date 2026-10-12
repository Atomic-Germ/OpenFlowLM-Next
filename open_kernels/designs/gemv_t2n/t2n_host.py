"""The host half of gemv_t2n in NumPy: tables in, partial y out. dflash2.cpp is the same in C++."""
from __future__ import annotations

import numpy as np

from t2n_plan import HDR, REC, TOKENS, Y_BAND


def records(xr: np.ndarray) -> np.ndarray:
    """xr [8, K] (the rotated activation) -> uint8 [K/256, 2 group, 2 quad, REC]."""
    t, k = xr.shape
    assert t == TOKENS and k % 256 == 0
    v = xr.astype(np.float64).reshape(2, 4, k // 128, 128).transpose(2, 0, 1, 3)    # [G, quad, tok, 128]
    m = np.abs(v).max(axis=(2, 3))
    with np.errstate(divide="ignore"):
        sh = np.where(m > 0, 14 - np.floor(np.log2(np.where(m > 0, m, 1.0))), 0).astype(np.int64)
    sh = np.clip(sh, -100, 126)
    xi = np.clip(np.rint(v * np.exp2(sh)[..., None, None]), -32768, 32767).astype(np.int16)
    out = np.zeros((k // 128, 2, REC), np.uint8)
    x8 = xi.reshape(k // 128, 2, 4, 16, 8).transpose(0, 1, 3, 2, 4)                  # [G, quad, o, tok, 8]
    out[..., :1024] = np.ascontiguousarray(x8).view(np.uint8).reshape(k // 128, 2, 1024)
    sums = xi.astype(np.int32).sum(axis=3)                                            # [G, quad, tok]
    out[..., 1024:1152] = np.repeat(sums[..., None], 8, axis=3).astype("<i4").view(np.uint8).reshape(k // 128, 2, 128)
    out[..., 1152:1156] = sh.astype("<i4")[..., None].view(np.uint8)
    return out.reshape(k // 256, 2, 2, REC)


def tables(p: dict, xr: np.ndarray) -> np.ndarray:
    """One table element per item: header, then the slice's records in (kt, group, quad) order."""
    rec = records(xr)
    tb = p["tab_bytes"]
    out = np.zeros((p["n_tabs"], tb), np.uint8)
    for core in p["items"]:
        for it in core:
            e = out[it["tab"]]
            e[:8] = np.array([it["nb"], it["nkt"]], "<i4").view(np.uint8)
            body = rec[it["kt0"]:it["kt0"] + it["nkt"]].reshape(-1)
            e[HDR:HDR + body.size] = body
    return out.reshape(-1)


def reduce(p: dict, y: np.ndarray) -> np.ndarray:
    """The kernel's partial y -> [8, N]: every item's bands added into place."""
    out = np.zeros((TOKENS, p["n"]), np.float64)
    for core in p["items"]:
        for it in core:
            yb = y[it["y"]:it["y"] + it["nb"] * Y_BAND].reshape(it["nb"], 2, 2, 4, 4, 8)   # [b, half, quad, j, tok, p]
            yb = yb.transpose(2, 4, 0, 1, 3, 5).reshape(TOKENS, it["nb"] * 64)               # [quad tok, b half j p]
            out[:, it["b0"] * 64:(it["b0"] + it["nb"]) * 64] += yb
    return out


def effective_x(xr: np.ndarray) -> np.ndarray:
    """What the table says x is: the int16 values back at their shifts (the kernel's exact input)."""
    rec = records(xr).reshape(-1, 2, REC)                       # [G, quad, REC]
    G = rec.shape[0]
    xi = np.ascontiguousarray(rec[..., :1024]).view(np.int16).reshape(G, 2, 16, 4, 8)
    sh = np.ascontiguousarray(rec[..., 1152:1156]).view("<i4").reshape(G, 2)
    v = xi.transpose(1, 3, 0, 2, 4).reshape(2, 4, G, 128).astype(np.float64) * np.exp2(-sh.T[:, None, :, None])
    return v.reshape(TOKENS, G * 128)
