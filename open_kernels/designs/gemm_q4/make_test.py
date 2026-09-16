"""Test vectors for gemm_q4: the verified gemv_q4 qkv case, M tokens wide.

Every token gets the SAME activation, so every one of the M output rows must
equal gemv_q4's verified ref_qkv.bin. That checks the batching without needing
a second reference - if the M dimension is wired up wrongly (wrong table slice,
wrong accumulator slice, wrong output scatter) the rows stop matching.

    python make_test.py --m 4 [--build build_pc2_m4] [--cores 32] [--blocked]

`--n` below 8192 takes the leading N rows of the same weights: the pool order is
band-major, so the first N/64 bands are output rows 0..N-1 and the reference is
just ref_qkv.bin truncated. That is how the output-tiled shapes are tested - a
core owning fewer bands is what lets M grow.

Writes x_m<M>.bin and run_m<M>.cfg next to this file; check with check.py <M>.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"

K = 2048
RS, KT = 2, 8                 # row split, k-tiles per band (K = 2048)

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=4)
ap.add_argument("--n", type=int, default=8192, help="output rows (GEMM_N)")
ap.add_argument("--build", default=None)
ap.add_argument("--cores", type=int, default=8)
ap.add_argument("--blocked", action="store_true",
                help="weights k-tile-major per core, for GEMM_BLOCKED=1 builds")
ap.add_argument("--runs", type=int, default=2, help="submits per config; the box drifts")
ap.add_argument("--per-call", type=int, default=1, help="chunks per DMA element (GEMM_PER_CALL)")
ap.add_argument("--bpt", type=int, default=0, help="bands per output tile (GEMM_BPT)")
ap.add_argument("--q4k", action="store_true",
                help="Q4_K chunks from make_q4k.py (4736 B), for GEMM_Q4K=1 builds")
ap.add_argument("--preq", action="store_true",
                help="stream prep_host.py's already-quantised IL table, for GEMM_PREQ=1 "
                     "builds (implies --q4k --blocked; no on-core activation prep)")
a = ap.parse_args()
if a.preq:
    a.q4k = True
    a.blocked = True

N = a.n
TILE = 4736 if a.q4k else 5120          # bytes per 32x256 chunk
W_BYTES = N * K * TILE // 8192
build = a.build or f"build_pc2_m{a.m}"

if a.preq:
    import prep_host
    x_path = prep_host.build(a.m)
    x_bytes = x_path.stat().st_size
    x_name = x_path.name
else:
    x1 = np.fromfile(GEMV / "x_qkv.bin", np.uint8)
    assert x1.size == 2 * K, f"expected {2*K} bytes of bf16 activation, got {x1.size}"
    np.tile(x1, a.m).tofile(HERE / f"x_m{a.m}.bin")
    x_bytes = 2 * K * a.m
    x_name = f"x_m{a.m}.bin"

# Every path in the cfg is RELATIVE to this design directory - run_kernel
# resolves them against the cfg's own location, so one cfg works from any
# checkout and on either side of a WSL/Windows split. See fixture_paths.py.
x_file = x_name

# Two host repacks, either of which a backend would fold into one load-time
# repack. --blocked reorders each core's chunks (band, kt, part) -> (kt, band,
# part) so a band's accumulator is revisited once per k-tile; > 8 cores
# interleaves one DMA element from each of a column's cores, because the
# column's L2 object is exactly that.
src = HERE / "w_q4k_pool.bin" if a.q4k else GEMV / "w_qkv.bin"
w_file = "w_q4k_pool.bin" if a.q4k else "../gemv_q4/w_qkv.bin"
if a.blocked or a.cores > 8 or N != 8192:
    bpc = (N // 64) // a.cores                      # bands per core
    w = np.fromfile(src, np.uint8)[:W_BYTES]
    assert w.size == W_BYTES, (w.size, W_BYTES)
    bpt = a.bpt or bpc                              # bands per output tile
    assert bpc % bpt == 0, (bpc, bpt)
    g = w.reshape(a.cores, bpc // bpt, bpt, KT, RS, TILE)  # [core][tile][band][kt][part]
    tag = "q4k" if a.q4k else ""
    if a.blocked:
        g = g.transpose(0, 1, 3, 2, 4, 5)           # [core][tile][kt][band][part]
        tag += "blk"
    if bpt != bpc:
        tag += f"t{bpt}"
    call = a.per_call * TILE                        # bytes per DMA element
    g = np.ascontiguousarray(g).reshape(a.cores, -1, call)      # [core][element][bytes]
    if a.cores > 8:
        g = (g.reshape(8, a.cores // 8, -1, call)
              .transpose(0, 2, 1, 3))               # [col][element][row][bytes]
        tag += f"c{a.cores}"
    if a.per_call != 1:
        tag += f"p{a.per_call}"
    if N != 8192:
        tag += f"n{N}"
    name = f"w_{tag}.bin"
    np.ascontiguousarray(g).reshape(-1).tofile(HERE / name)
    w_file = name

cfg = f"""device
xclbin G {build}/final.xclbin
kernelx k G {build}/insts.bin
buf w {W_BYTES} {w_file}
buf x {x_bytes} {x_file}
buf y {4 * N * a.m}
{chr(10).join(['run k w x y'] * a.runs)}
dump y y_m{a.m}.bin {4 * N * a.m}
"""
mode = "blocked" if a.blocked else "naive"
# a hung run (ERT state 8) leaves the previous dump in place, which reads as a
# pass; drop it so check.py fails loudly instead
(HERE / f"y_m{a.m}.bin").unlink(missing_ok=True)
(HERE / f"run_m{a.m}.cfg").write_text(cfg, encoding="utf-8", newline="\n")
(HERE / f"layout_m{a.m}.txt").write_text(
    f"{a.cores} {mode} {N} {a.bpt or (N // 64) // a.cores} "
    f"{'q4k' if a.q4k else 'q41'}", encoding="utf-8")
print(f"wrote {x_file} ({x_bytes} B) and run_m{a.m}.cfg -> {build}"
      f"  [N={N}, {a.cores} cores, {mode}{', preq' if a.preq else ''}]")
print(f"run:   (cd designs/gemm_q4 && run_kernel run_m{a.m}.cfg "
      f"&& python check.py {a.m})")
