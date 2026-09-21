"""Compare c_<name>.bin against ref_<name>.bin for a whisper_gemm stream (float64 metrics).

    python compare.py <test dir> <name> [<name> ...]

Gate: rel_fro <= 5e-3, the bf16 x bf16 -> fp32 GEMM's bar (attn_block uses the same one).
Per-row cosine is reported too: a row is one frame, and a C written transposed shows up as
a collapse there with a good overall cosine.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import whisper_gemm as wg  # noqa: E402

REL_FRO_GATE = 5e-3


def check(d: Path, name: str) -> bool:
    M, _, N = wg.STREAMS[name]
    got = np.fromfile(d / f"c_{name}.bin", np.float32).astype(np.float64)
    ref = np.fromfile(d / f"ref_{name}.bin", np.float32).astype(np.float64)
    if got.size != M * N or ref.size != M * N:
        print(f"FAIL {name}: sizes {got.size} / {ref.size}, want {M * N}")
        return False
    rel_fro = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    G, R = got.reshape(M, N), ref.reshape(M, N)
    c = np.einsum("ij,ij->i", G, R) / (np.linalg.norm(G, axis=1) * np.linalg.norm(R, axis=1) + 1e-30)
    ok = rel_fro <= REL_FRO_GATE and bool(np.isfinite(got).all()) and c.min() > 0.999
    print(f"{'PASS' if ok else 'FAIL'} {name:6s} {M}x{N} rel_fro={rel_fro:.3e} (gate {REL_FRO_GATE:.0e}) "
          f"row cos min={c.min():.9f} (row {int(np.argmin(c))}) finite={np.isfinite(got).all()}")
    return ok


def main() -> int:
    d = Path(sys.argv[1])
    names = sys.argv[2:] or list(wg.STREAMS)
    return 0 if all([check(d, n) for n in names]) else 1


if __name__ == "__main__":
    sys.exit(main())
