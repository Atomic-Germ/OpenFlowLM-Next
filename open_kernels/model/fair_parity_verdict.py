"""The routing-aligned verdict (goal mug03nrk-zekgbf, cont.76/78).

The reference run in /tmp/route16 adopts the NPU's top-8 wherever a y_rout dump exists, so it is the
first comparison in this campaign that is fair on routing; it also prints every case where the two
choices differed.  This script reports, in one pass:

  1. the flip list from ref.log, checked against the cont.78 prediction (margin < 2e-5);
  2. the routing-aligned drift |d_L| at L0 and L39, against the NPU's own streams;
  3. the per-token logits correlation and argmax under the fair protocol, next to the end-to-end
     numbers measured earlier;
  4. the L0/L13 attention-half splits (does the "95.4% of the seed is in the DeltaNet half" hold
     for tokens beyond 0, which the standalone probe could not do because their state is carried).
"""
import re
from pathlib import Path

import numpy as np

NPU = Path("/tmp/stage13")      # the NPU dumps (baseline run)
REF = Path("/tmp/route16")      # the routing-aligned reference
CORR_END = [0.999998, 0.999993, 0.999998, 0.999992, 0.999995, 0.999993, 0.999997, 0.999829, 0.999932,
            0.999994, 0.999994, 0.998143, 0.990380, 0.997209, 0.999693, 0.996276]


def sfx(t):
    return "" if t == 0 else f"_t{t}"


def rd(p):
    return np.fromfile(p, np.float32).astype(np.float64)


def margins(t):
    """The NPU's 8th/9th router-logit margin per layer, from the routing dumps."""
    out = {}
    for L in range(40):
        p = REF / f"y_rout{L}{sfx(t)}.bin"
        if not p.is_file():
            continue
        lg = np.fromfile(p, np.float32)[:256].astype(np.float64)
        s = np.sort(lg)[::-1]
        out[L] = s[7] - s[8]
    return out


def main():
    log = (REF / "ref.log").read_text(errors="replace")
    flips = []
    for ln in log.splitlines():
        m = re.match(r"\s*token (\d+) layer (\d+): NPU routed \[([\d, ]+)\], reference \[([\d, ]+)\]", ln)
        if m:
            flips.append((int(m.group(1)), int(m.group(2)),
                          sorted(int(x) for x in m.group(3).split(",")),
                          sorted(int(x) for x in m.group(4).split(","))))
    print(f"1. FLIPS reported by the routing-aligned run: {len(flips)}")
    for t, L, npu, ref in flips:
        mg = margins(t).get(L)
        print(f"   token {t} layer {L}: NPU {npu} vs reference {ref}"
              + (f"   (margin {mg:.2e})" if mg is not None else ""))

    print("\n2/3. routing-aligned drift and logits parity per token (end-to-end corr in brackets):")
    print(f"{'tok':>3} {'|d0|':>9} {'|d39|':>9} {'corr(fair)':>11} {'(end2end)':>10} {'argmax':>7} "
          f"{'passes':>6} {'#flips':>7}")
    fair_pass = 0
    for t in range(16):
        s = sfx(t)
        p0 = REF / f"ref_res0{s}.bin"
        if not p0.is_file():                 # the run writes per layer, so a token is only
            print(f"{t:>3}  (reference has not reached this token yet)")   # done at layer 39
            continue
        d0 = np.linalg.norm(rd(p0) - rd(NPU / f"y_res0{s}.bin"))
        p39 = REF / f"ref_res39{s}.bin"
        d39 = (np.linalg.norm(rd(p39) - rd(NPU / f"y_res39{s}.bin"))
               if p39.is_file() else float("nan"))
        try:
            a, b = rd(NPU / f"y_logits{s}.bin"), rd(REF / f"ref_logits{s}.bin")
            n = min(len(a), len(b))
            c = float(np.corrcoef(a[:n], b[:n])[0, 1])
            am = int(a[:248070].argmax() == b[:248070].argmax())
        except FileNotFoundError:
            print(f"{t:>3} {d0:>9.2e} {d39:>9.2e}   (logits not there yet)")
            continue
        nf = sum(1 for ft, _, _, _ in flips if ft == t)
        fair_pass += int(c > 0.9999)
        print(f"{t:>3} {d0:>9.2e} {d39:>9.2e} {c:>11.6f} {CORR_END[t]:>10.6f} {am:>7} "
              f"{'yes' if c > 0.9999 else 'NO':>6} {nf:>7}")
    print(f"   tokens above the 0.9999 bar under the fair protocol: {fair_pass}/16")

    print("\n4. L0 attention-half split (share of the half's own drift created in the attention half):")
    for t in range(16):
        s = sfx(t)
        p0 = REF / f"ref_attres0{s}.bin"
        if not p0.is_file():
            continue
        att = np.linalg.norm(rd(p0) - rd(NPU / f"act0_att{s}.bin"))
        lay = np.linalg.norm(rd(REF / f"ref_res0{s}.bin") - rd(NPU / f"y_res0{s}.bin"))
        print(f"   token {t:>2}: attention half {att:.3e} of whole layer {lay:.3e} "
              f"({100 * att / max(lay, 1e-30):.1f}%)")


if __name__ == "__main__":
    main()
