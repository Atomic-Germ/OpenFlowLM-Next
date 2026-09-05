"""Compare y_<tag>.bin (NPU f32 C) against ref_<tag>.bin (fp64 -> f32)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "m512_768x768"
got = np.fromfile(HERE / f"y_{tag}.bin", np.float32).astype(np.float64)
ref = np.fromfile(HERE / f"ref_{tag}.bin", np.float32).astype(np.float64)
n = min(len(got), len(ref))
got, ref = got[:n], ref[:n]
maxabs = np.abs(got - ref).max()
rel = maxabs / (np.abs(ref).max() + 1e-30)
cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
# bf16 inputs, fp32 accumulation on the array vs fp64: the shipped README
# quotes max abs 0.003 at K=768 on 0.1-scale inputs.
ok = cos > 0.99999 and rel < 5e-3 and np.isfinite(got).all()
print(f"{'PASS' if ok else 'FAIL'} {tag} n={n} cos={cos:.8f} maxabs={maxabs:.3e} maxrel={rel:.3e}")
sys.exit(0 if ok else 1)
