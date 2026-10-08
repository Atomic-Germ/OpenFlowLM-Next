r"""Check dit_gemm's C against make_test.py's float64 references.

    python compare.py <testdir> <stream>

Gate: finite everywhere; rel_fro against the exact product (bf16 A x bf16 B) <= 3e-2; and
per-row cosine > 0.999. The measured error is 1.25e-2 at K = 3072 and 1.9e-2 at
K = 12288 -- bfp16 operands plus the accumulator's re-rounding to bf16 every 64 of K --
so the gate sits above that and far below any layout or dataflow bug (those land at
rel_fro ~1 or produce non-finite values). "kernel" is also printed: the error against the
B the kernel really multiplies by, i.e. the arithmetic without the weight quantization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GATE_REL_FRO = 3e-2
GATE_ROW_COS = 0.999


def main() -> int:
    out, name = Path(sys.argv[1]), sys.argv[2]
    ref = np.load(out / f"ref_{name}.npz")
    M, N, rows = int(ref["M"]), int(ref["N"]), ref["rows"]
    if bool(ref.get("epi", False)):   # SwiGLU epilogue: the first 64 of every 128 columns
        c = np.fromfile(out / f"c_{name}.bin", dtype=bfloat16).reshape(M, N // 64, 2, 64)[:, :, 0]
        c = c.reshape(M, N)
    else:
        c = np.fromfile(out / f"c_{name}.bin", dtype=bfloat16).reshape(M, N)
    finite = bool(np.isfinite(c.astype(np.float32)).all())
    got = c[rows].astype(np.float64)
    fro = {k: float(np.linalg.norm(got - ref[k]) / np.linalg.norm(ref[k])) for k in ("exact", "kernel")}
    cos = (got * ref["exact"]).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref["exact"], axis=1))
    ok = finite and fro["exact"] <= GATE_REL_FRO and float(cos.min()) > GATE_ROW_COS
    print(f"{'PASS' if ok else 'FAIL'} {name} {M}x{N} rows={rows.size} finite={finite} "
          f"rel_fro exact={fro['exact']:.3e} (gate {GATE_REL_FRO:.0e}) kernel={fro['kernel']:.3e} "
          f"row-cos min={cos.min():.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
