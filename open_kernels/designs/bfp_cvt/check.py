r"""Inputs for bfp_cvt and the analysis of its output.

    python check.py gen      writes x_bfp_cvt.bin, run_bfp_cvt.cfg (and the input bits for check)
    <run_kernel.exe> run_bfp_cvt.cfg      (npu lock)
    python check.py check    decodes y_bfp_cvt.bin: finds the byte layout of a 72-byte block
                             vector, then tests host rounding models value by value

A model that reproduces every byte is what block_host.cpp's bfp16 activation tiler must implement
for the GEMM's numerics to stay as they are (OPEN-GEMM-T2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VECS, TILES = 64, 16
N = VECS * 64 * TILES                       # values
DRIVER = r"C:\code\openflowlm-next\open_kernels\harness\out\run_kernel.exe"


def gen() -> None:
    rng = np.random.default_rng(11)
    nb = N // 8
    sign = rng.integers(0, 2, (nb, 8), dtype=np.uint16) << 15
    base = rng.integers(100, 150, (nb, 1), dtype=np.int32)
    drop = rng.integers(0, 12, (nb, 8), dtype=np.int32)
    drop[:, 0] = 0                                            # lane 0 carries the block max exponent
    e = np.clip(base - drop, 1, 254).astype(np.uint16)
    frac = rng.integers(0, 128, (nb, 8), dtype=np.uint16)
    bits = sign | (e << 7) | frac
    # a share of blocks with all-equal exponents (no shift: is the max element itself rounded?),
    # a share holding zeros, and a share with an exact power-of-two max
    bits[::7] = (bits[::7] & 0x807F) | (bits[::7, :1] & 0x7F80)
    bits[3::11, 2:5] = 0
    bits[5::13, 0] = (bits[5::13, 0] & 0xFF80)
    bits.reshape(-1).tofile(HERE / "x_bits.bin")
    (HERE / "x_bfp_cvt.bin").write_bytes(bits.reshape(-1).astype(np.uint16).tobytes())
    ybytes = N // 64 * 72
    cfg = ["device", "xclbin G build/final.xclbin", "kernelx k G build/insts.bin",
           f"buf x {N * 2} x_bfp_cvt.bin", f"buf y {ybytes}", "run k x y", f"dump y y_bfp_cvt.bin {ybytes}", ""]
    (HERE / "run_bfp_cvt.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"{N} values -> run_bfp_cvt.cfg")


def bf16(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def decode(y: np.ndarray, layout: str) -> tuple[np.ndarray, np.ndarray]:
    """mantissas [nb, 8] int8 and exponents [nb] uint8 under a layout hypothesis"""
    v = y.reshape(-1, 72)
    if layout == "exp_first":
        b = v.reshape(-1, 8, 9)
        return b[..., 1:].astype(np.int8).reshape(-1, 8), b[..., 0].reshape(-1)
    if layout == "exp_last":
        b = v.reshape(-1, 8, 9)
        return b[..., :8].astype(np.int8).reshape(-1, 8), b[..., 8].reshape(-1)
    if layout == "mant_then_exp":
        return v[:, :64].astype(np.int8).reshape(-1, 8), v[:, 64:].reshape(-1)
    if layout == "exp_then_mant":
        return v[:, 8:].astype(np.int8).reshape(-1, 8), v[:, :8].reshape(-1)
    raise ValueError(layout)


def check() -> None:
    x = np.fromfile(HERE / "x_bits.bin", np.uint16).reshape(-1, 8)
    y = np.fromfile(HERE / "y_bfp_cvt.bin", np.uint8)
    xv = bf16(x)
    best = None
    for layout in ("exp_first", "exp_last", "mant_then_exp", "exp_then_mant"):
        m, e = decode(y, layout)
        emax = ((x >> 7) & 0xFF).max(axis=1).astype(np.int64)
        # fit the exponent field against the block max: field = emax + c for one c?
        c = np.bincount((e.astype(np.int64) - emax + 300).clip(0, 600)).argmax() - 300
        frac_c = np.mean(e.astype(np.int64) - emax == c)
        # the value scale: v = m * 2^(field - bias); find bias making |m * 2^(field-bias) - x| small
        errs = {}
        for bias in range(120, 140):
            val = m * np.ldexp(1.0, (e.astype(np.int64) - bias)[:, None])
            scale = np.ldexp(1.0, emax - 127)[:, None]
            errs[bias] = np.mean(np.abs(val - xv) <= scale * 2.0 ** -5)
        b = max(errs, key=errs.get)
        print(f"{layout:14s}: field - emax = {c} for {frac_c * 100:.1f} % of blocks; best bias {b}: "
              f"{errs[b] * 100:.2f} % of values within 2^-5 of the block max")
        if best is None or errs[b] > best[2]:
            best = (layout, b, errs[b], m, e)
    layout, bias, _, m, e = best
    val = m * np.ldexp(1.0, (e.astype(np.int64) - bias)[:, None])
    print(f"\nlayout {layout}, value = mantissa (int8) * 2^(exp - {bias})")
    # host models: shared E = block max exponent (+bump), mantissa = round(x / 2^(E - 127 - 6 + bump))
    emax = ((x >> 7) & 0xFF).max(axis=1).astype(np.int64)
    for bump in (0, 1):
        lsb = np.ldexp(1.0, emax - 127 - 6 + bump)[:, None]
        q = xv / lsb
        for mode in ("rne", "rna", "trunc", "floor"):
            r = {"rne": np.round(q), "rna": np.sign(q) * np.floor(np.abs(q) + 0.5), "trunc": np.trunc(q),
                 "floor": np.floor(q)}[mode]
            r = np.clip(r, -128, 127)
            hv = r * lsb
            ok = hv == val
            print(f"host E = block max {'+1' if bump else '  '}, {mode:5s}: {ok.mean() * 100:8.4f} % identical"
                  f" ({np.sum(~ok)} differ)")
            if bump == 0 and mode == "rne" and np.sum(~ok):
                for i in np.argwhere(~ok)[:6]:
                    bi, li = i
                    print(f"    block {bi} lane {li}: x {xv[bi, li]:.8g} dev {val[bi, li]:.8g} host {hv[bi, li]:.8g}"
                          f"  exps {list(((x[bi] >> 7) & 0xFF).astype(int))} field {int(e[bi])}")


def check_bytes() -> None:
    """bfp16_host.py's conversion under each candidate model against the device bytes, exactly"""
    from bfp16_host import bf16_to_bfp16_v64
    x = np.fromfile(HERE / "x_bits.bin", np.uint16).reshape(-1, 64)
    y = np.fromfile(HERE / "y_bfp_cvt.bin", np.uint8).reshape(-1, 72)
    for layout in ("exp_first", "exp_last", "mant_then_exp", "exp_then_mant"):
        for off in range(-8, 9):
            for ov in ("saturate", "bump", "bump_sym"):
                h = bf16_to_bfp16_v64(x, {"layout": layout, "exp_offset": off, "overflow": ov})
                ok = (h == y).all(axis=1)
                if ok.mean() > 0.5:
                    print(f"model layout={layout} exp_offset={off} overflow={ov}: {ok.mean() * 100:.4f} % of "
                          f"block vectors byte-identical ({np.sum(~ok)} differ)")
                    if np.sum(~ok):
                        i = int(np.argwhere(~ok)[0, 0])
                        print("   first differing vector", i)
                        print("   dev ", y[i].tolist())
                        print("   host", h[i].tolist())


if __name__ == "__main__":
    {"gen": gen, "check": check, "bytes": check_bytes}[sys.argv[1]]()
