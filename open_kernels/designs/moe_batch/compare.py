"""Compare y_<tag>.bin (and h_<tag>.bin) against make_test.py's references.

    python compare.py s16

Gate: rel_fro <= 5e-3 on y, the same bar as gemm_q4_prefill/compare.py (the same
bf16 x bf16 -> f32 datapath, plus a bf16 h between the two products); per-slot
cosine printed so a wrong slot or token column shows up by itself.
"""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from layout import h_from_dev, y_from_dev  # noqa: E402

HID, FF, NT = 2048, 512, 8
REL_FRO_GATE = 5e-3


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "s16"
    got = y_from_dev(np.fromfile(HERE / f"y_{tag}.bin", np.float32), HID).reshape(-1).astype(np.float64)
    ref = np.fromfile(HERE / f"ref_{tag}.bin", np.float32).astype(np.float64)
    n = min(len(got), len(ref))
    got, ref = got[:n], ref[:n]
    S = n // (HID * NT)
    rel_fro = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    ok = rel_fro <= REL_FRO_GATE and bool(np.isfinite(got).all())
    print(f"{'PASS' if ok else 'FAIL'} y: slots={S} rel_fro={rel_fro:.3e} (gate {REL_FRO_GATE:.0e}) cos={cos:.9f} "
          f"finite={np.isfinite(got).all()}")
    G, R = got.reshape(S, NT, HID), ref.reshape(S, NT, HID)
    per_slot = np.einsum("sti,sti->s", G, R) / (np.linalg.norm(G, axis=(1, 2)) * np.linalg.norm(R, axis=(1, 2)) + 1e-30)
    per_tok = np.einsum("sti,sti->st", G, R) / (np.linalg.norm(G, axis=2) * np.linalg.norm(R, axis=2) + 1e-30)
    print(f"per-slot cosine: min={per_slot.min():.9f} (slot {int(np.argmin(per_slot))}) mean={per_slot.mean():.9f}")
    print(f"per-token cosine: min={per_tok.min():.9f} (slot {int(np.argmin(per_tok) // NT)}, "
          f"token {int(np.argmin(per_tok) % NT)})")
    hp = HERE / f"h_{tag}.bin"
    if hp.is_file():
        h = h_from_dev(np.fromfile(hp, bfloat16), FF).reshape(-1).astype(np.float64)
        hr = np.fromfile(HERE / f"refh_{tag}.bin", bfloat16).astype(np.float64)
        m = min(len(h), len(hr))
        hf = float(np.linalg.norm(h[:m] - hr[:m]) / (np.linalg.norm(hr[:m]) + 1e-30))
        print(f"h: rel_fro={hf:.3e} cos={float(h[:m] @ hr[:m] / (np.linalg.norm(h[:m]) * np.linalg.norm(hr[:m]) + 1e-30)):.9f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
