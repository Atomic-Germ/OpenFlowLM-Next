"""Compare the 27B open-kernel decode with the CPU replica, per token: full-vocab logits (corr,
argmax, top-5) and per-layer residuals. --tokens N covers positions 0..N-1 of `make_27b.py
--tokens N` (token 0's files are unsuffixed, token t's carry _t{t})."""
import argparse
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).parent / "w27"
ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=1)
a = ap.parse_args()
sfx = lambda t: "" if t == 0 else f"_t{t}"
allok = True
for t in range(a.tokens):
    s = sfx(t)
    ours = np.fromfile(W / f"y_logits{s}.bin", np.float32).astype(np.float64)
    ref = np.fromfile(W / f"ref_logits{s}.bin", np.float32).astype(np.float64)
    n = min(len(ours), len(ref)); ours, ref = ours[:n], ref[:n]
    corr = float(np.corrcoef(ours, ref)[0, 1])
    print(f"token {t} (position {t}): logits corr {corr:.6f}  argmax ours {int(ours.argmax())} ref {int(ref.argmax())}"
          f"  top5 ours {np.argsort(-ours)[:5].tolist()} ref {np.argsort(-ref)[:5].tolist()}")
    l = 0
    while (W / f"y_res{l}{s}.bin").is_file() and (W / f"ref_res{l}{s}.bin").is_file():
        g = np.fromfile(W / f"y_res{l}{s}.bin", np.float32).astype(np.float64)
        r = np.fromfile(W / f"ref_res{l}{s}.bin", np.float32).astype(np.float64)
        gi = np.fromfile(W / f"y_rout{l}{s}.bin", np.float32)[256:264].view(np.int32).tolist() if (W / f"y_rout{l}{s}.bin").is_file() else None
        print(f"  layer {l:2}: residual corr {np.corrcoef(g, r)[0,1]:.6f} maxrel {np.abs(g-r).max()/np.abs(r).max():.2e}  routing {gi}")
        l += 1
    ok = bool(np.isfinite(ours).all()) and corr > 0.9999 and int(ours.argmax()) == int(ref.argmax())
    print(f"token {t}: {'PASS' if ok else 'FAIL'}")
    allok &= ok
print("PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
