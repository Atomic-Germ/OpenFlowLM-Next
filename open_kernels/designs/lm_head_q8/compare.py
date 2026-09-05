"""Compare y_npu.bin against ref.bin (float64 metrics, as in LLMNpuTest)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
region = sys.argv[1] if len(sys.argv) > 1 else "full"
got = np.fromfile(HERE / f"y_{region}.bin", np.float32).astype(np.float64)
ref = np.fromfile(HERE / f"ref_{region}.bin", np.float32).astype(np.float64)
n = min(len(got), len(ref))
got, ref = got[:n], ref[:n]
rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
bad = np.flatnonzero(np.abs(got - ref) > 1e-3 * (np.abs(ref).max() + 1e-30))
ok = cos > 0.9999999 and rel < 1e-4
print(f"{'PASS' if ok else 'FAIL'} n={n} cos={cos:.9f} maxrel={rel:.3e} "
      f"finite={np.isfinite(got).all()} nbad={len(bad)}")
if len(bad):
    print("first bad idx:", bad[:16])
    print("got:", got[bad[:8]])
    print("ref:", ref[bad[:8]])
print("got[:8]", got[:8])
print("ref[:8]", ref[:8])
sys.exit(0 if ok else 1)
