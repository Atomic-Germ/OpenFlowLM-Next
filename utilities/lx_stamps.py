"""Decode the LX_STAMP event logs of the merged Bonsai layer image (dux.py; timing-only build).

The log rides in the last band of the second down piece: act[A_OUT2B | AA_OUT2B] + (10 c + 9) * 256
for core c (gen_kernels.py STAMP). Dump the whole 8 x 10 x 256 B region with the CLI:

    open_qwen36_cli ... --dump-act 2:356352:20480:l2.bin --dump-act 3:225280:20480:l3.bin

usage:
    python lx_stamps.py <dump.bin> <lx|ax> [--mhz 1800]
    python lx_stamps.py --span <first.bin> <first kind> <last.bin> <last kind>
        raw core-0 timer ticks from the first dump's xn0 entry to the last dump's end (two layers
        of the SAME step, so one tile timer), for calibrating the clock against the step trace.
"""
from __future__ import annotations

import argparse
import struct


def events(kind: str) -> list[str]:
    seq: list[str] = []

    def prep(p: str) -> None:
        seq.append(f"G<{p}0")
        for i in range(3):
            seq.extend([f"{p}{i}", f"{p}{i}w"])

    prep("xn")
    if kind == "lx":
        for h in range(6):
            seq += [f"r{h}", f"d{h}"]
    prep("og")
    prep("xm")
    for p in ("p0", "p1"):
        seq += [f"G<{p}", p]
    return seq


def load(path: str) -> list[tuple]:
    b = open(path, "rb").read()
    return [struct.unpack_from("<64I", b, (10 * c + 9) * 256) for c in range(8)]


def rows(kind: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """(label, [(from, to), ...]) -- a row is the sum of its spans. The last element of each
    prep carries no exit stamp (program memory), so its table runs into the next phase's row."""
    r: list[tuple[str, list[tuple[str, str]]]] = []

    def prep(p: str, label: str) -> None:
        r.append((f"{label} prep: FWHT (+signs), 3 elems", [(f"{p}{i}", f"{p}{i}w") for i in range(3)]))
        r.append((f"{label} prep: table (+x wait), elems 0-1", [(f"{p}{i}w", f"{p}{i + 1}") for i in range(2)]))

    prep("xn", "xn")
    if kind == "lx":
        r.append(("xn2 table + qkv|z GEMV (32 bands)", [("xn2w", "G<og0")]))
        r.append(("glue wait -> head 0 record", [("G<og0", "r0")]))
        r.append(("DN pass 1 (32 slices), 6 heads", [(f"r{h}", f"d{h}") for h in range(6)]))
        r.append(("DN delta + pass 2 + o + next rec, 5", [(f"d{h}", f"r{h + 1}") for h in range(5)]))
        r.append(("DN head 5 delta + pass 2 + post wait", [("d5", "og0")]))
    else:
        r.append(("xn2 table + q|g|k|v GEMV (28 bands)", [("xn2w", "G<og0")]))
        r.append(("attention wait (-> og0)", [("G<og0", "og0")]))
    prep("og", "og")
    r.append(("og2 table + out/o GEMV (10 bands)", [("og2w", "G<xm0")]))
    r.append(("residual + norm wait", [("G<xm0", "xm0")]))
    prep("xm", "xm")
    r.append(("xm2 table + up|gate GEMV (34 bands)", [("xm2w", "G<p0")]))
    r.append(("h wait", [("G<p0", "p0")]))
    r.append(("down p0: 8 h preps + 10 bands K=8192", [("p0", "G<p1")]))
    r.append(("wait -> p1", [("G<p1", "p1")]))
    r.append(("down p1: 9 h preps + 10 bands K=9216", [("p1", "end")]))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="+")
    ap.add_argument("--mhz", type=float, default=1800.0)
    ap.add_argument("--span", action="store_true")
    a = ap.parse_args()
    if a.span:
        f0, k0, f1, k1 = a.args
        s0, s1 = load(f0)[0], load(f1)[0]
        i0 = events(k0).index("xn0")
        ticks = (s1[62] - s0[i0]) & 0xFFFFFFFF
        print(f"core 0: {ticks} ticks from {f0} xn0 to {f1} end ({ticks / a.mhz / 1e3:.3f} ms at {a.mhz:g} MHz)")
        return 0
    path, kind = a.args
    seq = events(kind)
    idx = {n: i for i, n in enumerate(seq)}
    idx["end"] = 62
    cores = load(path)
    for c, s in enumerate(cores):
        if s[63] != len(seq):
            print(f"core {c}: {s[63]} events logged, expected {len(seq)}")

    def dt(s, x, y):
        return ((s[idx[y]] - s[idx[x]]) & 0xFFFFFFFF) / a.mhz

    print(f"{'phase (us at %g MHz)' % a.mhz:40s}" + "".join(f"{'c' + str(c):>7s}" for c in range(8)) + f"{'mean':>9s}")
    tot = 0.0
    for label, spans in rows(kind):
        v = [sum(dt(s, x, y) for x, y in spans) for s in cores]
        m = sum(v) / 8
        tot += m
        print(f"{label:40s}" + "".join(f"{x:7.1f}" for x in v) + f"{m:9.1f}")
    print(f"{'sum xn0 -> end':40s}{'':56s}{tot:9.1f}")
    s = cores[0]
    if kind == "lx":
        print("core 0 per head (us): pass 1 / delta + pass 2 + o + next record:")
        for h in range(6):
            nxt = f"r{h + 1}" if h < 5 else "og0"
            print(f"  head {h}: {dt(s, f'r{h}', f'd{h}'):6.1f} {dt(s, f'd{h}', nxt):6.1f}")
    for p in ("xn", "og", "xm"):
        print(f"core 0 {p} per element (FWHT, table + wait):",
              [(round(dt(s, f'{p}{i}', f'{p}{i}w'), 1), round(dt(s, f'{p}{i}w', f'{p}{i + 1}'), 1) if i < 2 else None)
               for i in range(3)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
