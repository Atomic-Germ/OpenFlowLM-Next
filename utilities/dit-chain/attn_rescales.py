r"""attn_rescales: how often dit_fa's lazy rescale fires on klein's real attention inputs.

dit_fa's time depends on the data: a 32-row query tile runs the O / l rescale for a
64-key chunk whenever any of its rows' chunk max passes the tile's reference max by more
than TAU (fa_dit.cc; `utilities/dit-ref/fa_emul.py` models it). This runs the real
pipeline (pyxrt, `generate.py`'s Runner) on a study prompt up to each sampled attention
op, reads that op's Q and K from the device, and counts the (tile, chunk) pairs that
rescale for each TAU. No clock is involved, so a busy machine doesn't matter.

For calibration it counts the same on `make_test.py`-style data (Q, K ~ N(0, s^2)), whose
times are in open_kernels/designs/dit_fa/README.md.

    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\dit-chain\attn_rescales.py --kernels C:\dev\klein-kernels --size 1024 \
        --study C:\dev\ditref-out\goldens_pipe_1024 [--prompt 0] [--taus 8,16,32] \
        [--sample step0:dbl0,step0:sgl0,step0:sgl10,step0:sgl19,step3:sgl0,step3:sgl19] [--time 10]

--time alternates real and synthetic inputs run by run in one process, so a busy machine
shifts both alike: it answers "is it the data?", not "how fast is it?".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate as gen  # noqa: E402

TQ, LKP, D = 32, 64, 128
C = math.log2(math.e) / math.sqrt(D)


def rescale_rate(q: np.ndarray, k: np.ndarray, taus: list[float]) -> dict[float, float]:
    """q, k [heads, L, D] float32. The fraction of (tile, chunk) pairs past the first
    chunk that rescale, per TAU (fa_emul's rule without its bf16 roundings)."""
    H, L, _ = q.shape
    hits = {t: 0 for t in taus}
    for h in range(H):
        s = q[h] @ k[h].T                                        # [L, L]
        cm = s.reshape(L // TQ, TQ, L // LKP, LKP).max(-1)       # [tiles, rows, chunks]
        for t in taus:
            thr = t / C
            m = cm[..., 0].copy()                                # the first chunk sets m
            n = 0
            for j in range(1, cm.shape[-1]):
                need = (cm[..., j] > m + thr).any(-1)            # per tile
                n += int(need.sum())
                m = np.where(need[:, None], np.maximum(m, cm[..., j]), m)
            hits[t] += n
    pairs = H * (L // TQ) * (L // LKP - 1)
    return {t: hits[t] / pairs for t in taus}


def heads_of(x: np.ndarray, col: int, heads: int) -> np.ndarray:
    blk = x[:, col:col + heads * D].astype(np.float32)
    return np.ascontiguousarray(blk.reshape(x.shape[0], heads, D).transpose(1, 0, 2))


def time_real_vs_synthetic(r, name, x, ld, st, bos, n, completed) -> None:
    """The op's Q/K/V are columns [0, 3 * 24 * D) of its buffer (outputs lie past them)."""
    import statistics
    import time
    L, w = x.shape[0], 3 * 24 * D
    real = x.view(np.uint16).copy()
    f = x[:, :w].astype(np.float32)
    syn = real.copy()
    rng = np.random.default_rng(0)
    for c in range(0, w, 24 * D):                      # Q, K, V each at their own std
        sd = float(f[:, c:c + 24 * D].std())
        syn[:, c:c + 24 * D] = (rng.standard_normal((L, 24 * D), np.float32) * sd)             .astype(bfloat16).view(np.uint16)
    t = {"real": [], "synthetic": []}
    for _ in range(n):
        for tag, arr in (("real", real), ("synthetic", syn)):
            r.bufs[name].write(arr)
            t0 = time.perf_counter()
            if st.start(*bos).wait() != completed:
                raise RuntimeError(st.name)
            t[tag].append((time.perf_counter() - t0) * 1e3)
    r.bufs[name].write(real)
    print("    time (median of %d, ms): " % n
          + "  ".join(f"{k} {statistics.median(v):.2f}" for k, v in t.items())
          + f"  (std Q/K/V {', '.join(f'{float(f[:, c:c + 24 * D].std()):.2f}' for c in range(0, w, 24 * D))})",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--study", required=True, help="capture_pipeline_inputs.py output dir")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--taus", default="8,16,32")
    ap.add_argument("--sample", default="step0:dbl0,step0:sgl0,step0:sgl10,step0:sgl19,"
                                        "step3:sgl0,step3:sgl19")
    ap.add_argument("--time", type=int, default=0,
                    help="also time each sampled op N times on its real inputs, alternating "
                         "with the same op on N(0, std) inputs of the real std")
    a = ap.parse_args()
    taus = [float(t) for t in a.taus.split(",")]
    R = a.size

    print("calibration, Q/K ~ N(0, s^2), 24 heads (make_test.py --qk-scale s):")
    L = gen.kp.image_tokens(R) + gen.kp.L_TXT
    g = np.random.default_rng(0)
    for s in (1, 2, 3):
        q = g.standard_normal((24, L, D), np.float32) * s
        k = g.standard_normal((24, L, D), np.float32) * s
        rates = rescale_rate(q, k, taus)
        print(f"  s={s}: " + "  ".join(f"TAU {t:g}: {r:.4f}" for t, r in rates.items()), flush=True)

    want = [tuple(x.split(":")) for x in a.sample.split(",")]
    sd = Path(a.study)
    prompts = json.loads((sd / "prompts.json").read_text(encoding="utf-8"))
    r = gen.Runner(Path(a.kernels), R, 4)
    r.set_prompt(prompts[a.prompt], np.load(sd / f"ids_{a.prompt}.npy"))
    r.set_noise(np.load(sd / f"noise_{a.prompt}.npy"))
    from npu_host import COMPLETED
    print(f"prompt {a.prompt} at {R}x{R}:")
    for kset, st, bos, phase, o in r.ops:
        what = o["what"].split()[0]
        if o["stream"].endswith(("_attn_sgl", "_attn_dbl")) and (phase, what) in want:
            _, name, off, nb = o["args"][0]
            ld = r.pl.buffers[name] // 2 // L
            x = r.bufs[name].read(np.uint16, 0, L * ld).reshape(L, ld).view(bfloat16)
            rates = rescale_rate(heads_of(x, 0, 24), heads_of(x, 24 * D, 24), taus)
            print(f"  {phase} {what}: " + "  ".join(f"TAU {t:g}: {v:.4f}" for t, v in rates.items()),
                  flush=True)
            if a.time:
                time_real_vs_synthetic(r, name, x, ld, st, bos, a.time, COMPLETED)
        if st.start(*bos).wait() != COMPLETED:
            raise RuntimeError(f"{kset}/{st.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
