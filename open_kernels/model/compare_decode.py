"""Score an open-kernel decode run against the fp64 reference make_decode.py wrote.

    python open_kernels/model/compare_decode.py [--tokens N] [--out DIR]

Per token: the full-vocab logits (correlation, argmax, top-5) and every layer's
residual. PASS needs finite logits, correlation > 0.9999 and the same argmax --
the same bar phlegm's 27B runs were held to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sfx(t):
    return "" if t == 0 else f"_t{t}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    w = Path(a.out)

    allok = True
    for t in range(a.tokens):
        s = sfx(t)
        if not (w / f"y_logits{s}.bin").is_file():
            print(f"token {t}: no y_logits{s}.bin -- the run did not get this far")
            return 1
        ours = np.fromfile(w / f"y_logits{s}.bin", np.float32).astype(np.float64)
        ref = np.fromfile(w / f"ref_logits{s}.bin", np.float32).astype(np.float64)
        n = min(len(ours), len(ref))
        ours, ref = ours[:n], ref[:n]
        corr = float(np.corrcoef(ours, ref)[0, 1])
        print(f"token {t} (position {t}): logits corr {corr:.6f}  argmax ours {int(ours.argmax())} "
              f"ref {int(ref.argmax())}  top5 ours {np.argsort(-ours)[:5].tolist()} "
              f"ref {np.argsort(-ref)[:5].tolist()}")
        l = 0
        while (w / f"y_res{l}{s}.bin").is_file() and (w / f"ref_res{l}{s}.bin").is_file():
            g = np.fromfile(w / f"y_res{l}{s}.bin", np.float32).astype(np.float64)
            r = np.fromfile(w / f"ref_res{l}{s}.bin", np.float32).astype(np.float64)
            rp = w / f"y_rout{l}{s}.bin"
            gi = np.fromfile(rp, np.float32)[256:264].view(np.int32).tolist() if rp.is_file() else None
            print(f"  layer {l:2}: residual corr {np.corrcoef(g, r)[0, 1]:.6f} "
                  f"maxrel {np.abs(g - r).max() / np.abs(r).max():.2e}  routing {gi}")
            l += 1
        ok = bool(np.isfinite(ours).all()) and corr > 0.9999 and int(ours.argmax()) == int(ref.argmax())
        print(f"token {t}: {'PASS' if ok else 'FAIL'}")
        allok &= ok
    print("PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
