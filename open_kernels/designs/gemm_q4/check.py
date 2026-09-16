"""Every token in y_m<M>.bin must equal gemv_q4's verified ref_qkv.bin.

Both dataflows drain each stream contiguously, so the raw buffer needs
de-interleaving. make_test.py leaves "<cores> <naive|blocked> <N>" in
layout_m<M>.txt.

  naive:    one y element per band, so [core][band][token][row] direct and
            [col][band][row][token][64] fan-out (core (c, r) owns bands
            (c*n_rows + r)*bpc..).
  blocked:  one y element per OUTPUT TILE, so [core][tile][band-in-tile][token]
            [row] direct and [col][tile][row][band-in-tile][token][64] fan-out.
            With one tile (bpt = bpc) that collapses to the direct reshape for
            both; with one band per tile it is the naive fan-out order again.

    python check.py <M> [fast|i8|q4k]

The bar matches the kernel: `fast` for GEMM_FAST_EPILOGUE builds (~7e-3 maxrel,
the same order as ggml's CPU kernel), `i8` for GEMM_INT8 on top of that (~1.3e-2,
the price of a 7-bit activation), `q4k` for GEMM_Q4K (~1.6e-3, the price of one
activation binary point per 256 superblock instead of per 32-block). `q4k` is
picked automatically from layout_m<M>.txt; a GEMM_Q4K build with
GEMM_FAST_EPILOGUE on top lands at ~1.4e-2 and needs `i8`'s bar.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"

m = int(sys.argv[1]) if len(sys.argv) > 1 else 4
layout = HERE / f"layout_m{m}.txt"
fields = layout.read_text().split() if layout.exists() else ["8"]
q4k = len(fields) > 4 and fields[4] == "q4k"
bar = next((b for b in ("i8", "fast", "q4k") if b in sys.argv[2:]),
           "q4k" if q4k else "full")
# full: exact epilogue. fast: GEMM_FAST_EPILOGUE drops the low bf16 half (~7e-3,
# ggml's CPU kernel is the same order). i8: int8 activations keep 7 bits of
# block-relative precision instead of 15, which doubles that again. q4k: the
# activation gets one binary point per 256 superblock rather than per 32-block,
# because the int accumulation runs across the whole superblock.
cos_bar, rel_bar = {"full": (0.9999999, 1e-4),
                    "q4k": (0.999999, 5e-3),
                    "fast": (0.99998, 1e-2),
                    "i8": (0.9999, 2e-2)}[bar]
cores = int(fields[0])
blocked = len(fields) > 1 and fields[1] == "blocked"
N = int(fields[2]) if len(fields) > 2 else 8192
BAND_ROWS = 64
bands = N // BAND_ROWS
bpc = bands // cores
bpt = int(fields[3]) if len(fields) > 3 else bpc      # bands per output tile
n_tiles = bpc // bpt

raw = np.fromfile(HERE / f"y_m{m}.bin", np.float32).astype(np.float64)
assert raw.size == m * N, f"expected {m*N} floats, got {raw.size}"

if not blocked:
    if cores <= 8:
        # [core][band][token][row] -> [token][core][band][row]
        got = raw.reshape(cores, bpc, m, BAND_ROWS).transpose(2, 0, 1, 3).reshape(m, N)
    else:
        n_cols, n_rows = 8, cores // 8
        # [col][band][row][token][64] -> [token][col][row][band][64]
        got = (raw.reshape(n_cols, bpc, n_rows, m, BAND_ROWS)
                  .transpose(3, 0, 2, 1, 4)
                  .reshape(m, N))
elif cores <= 8:
    # [core][tile][band][token][row] -> [token][core][tile][band][row]
    got = (raw.reshape(cores, n_tiles, bpt, m, BAND_ROWS)
              .transpose(3, 0, 1, 2, 4)
              .reshape(m, N))
else:
    n_cols, n_rows = 8, cores // 8
    # [col][tile][row][band][token][64] -> [token][col][row][tile][band][64]
    got = (raw.reshape(n_cols, n_tiles, n_rows, bpt, m, BAND_ROWS)
              .transpose(4, 0, 2, 1, 3, 5)
              .reshape(m, N))

# GEMM_Q4K re-expresses the same weights with 6-bit sub-scales, so its
# reference is make_q4k.py's exact product, not gemv_q4's.
ref_file = HERE / "ref_q4k.bin" if q4k else GEMV / "ref_qkv.bin"
ref = np.fromfile(ref_file, np.float32).astype(np.float64)[:N]

ok = True
for i in range(m):
    g = got[i]
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-30)
    cos = float(g @ ref / (np.linalg.norm(g) * np.linalg.norm(ref) + 1e-30))
    row_ok = cos > cos_bar and rel < rel_bar and np.isfinite(g).all()
    ok &= row_ok
    print(f"{'PASS' if row_ok else 'FAIL'} token {i}: cos={cos:.9f} maxrel={rel:.3e}")

print(f"{'ALL PASS' if ok else 'FAILED'}  ({bar} bar vs {ref_file.name}: "
      f"cos > {cos_bar}, maxrel < {rel_bar:g})")
sys.exit(0 if ok else 1)
