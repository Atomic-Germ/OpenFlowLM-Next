r"""cold_probe: does an op run slower right after its set's configure?

In the one-context engine, attention measured ~36 ms per call against 32.6 ms for the same
op on the same buffers run back to back in its own xclbin context (switch_probe.py). This
times real ops of an installed v2 ELF, one run at a time (start, wait), in three
positions:

    configure   the cfg_<set> run alone (the other set configured before it)
    first       the op right after its set's configure
    again       the same op run again, no configure in between

The configures alternate between the two empty-reset variants, as the engine's do. Buffer
contents are random: only time is measured. Run it on a quiet machine, in turbo, with no
other NPU client.

    . C:\dev\mlir-aie\iron_env.ps1
    xrt-smi configure --pmode turbo
    python utilities\reconfig-probe\cold_probe.py --kernels src\xclbins\FLUX.2-klein-4B-NPU2\open_kernels --size 1024
        [--ops r1024_attn_sgl,r1024_sgl_out,r1024_qk_sgl,r1024_res_all] [--reps 20] [--scale S]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyxrt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
import klein_pipeline as kp  # noqa: E402

TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True, help="an installed v2 kernel set")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--ops", default=None, help="comma list of streams (default: a single block's)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=0,
                    help="fill activations with bf16 N(0, scale) instead of random bytes")
    a = ap.parse_args()
    R = a.size
    names = (a.ops or f"r{R}_attn_sgl,r{R}_sgl_out,r{R}_qk_sgl,r{R}_res_all").split(",")
    kdir = Path(a.kernels)
    man = json.loads((kdir / "diffusion_kernels.json").read_text())
    pl = kp.plan(R)
    dev = pyxrt.device(0)
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(kdir / man["elf"][str(R)])))
    wsize = {n: K * N * 9 // 8 for n, (K, N, _) in (kp.dit_weight_specs() | kp.te_weight_specs()).items()}
    rng = np.random.default_rng(0)
    bufs, views = {}, {}

    def bo(name, nbytes):
        if name not in bufs:
            b = pyxrt.ext.bo(dev, nbytes)
            h = np.frombuffer(b.map(), np.uint8)
            if a.scale and not name.startswith("w:"):
                x = (rng.standard_normal(nbytes // 2, dtype=np.float32) * a.scale).view(np.uint32)
                h.view(np.uint16)[:] = (x >> 16).astype(np.uint16)
            else:
                h[:] = rng.integers(0, 256, h.size, dtype=np.uint8) & 0x3F   # finite bf16s
            b.sync(TO)
            bufs[name] = b
        return bufs[name]

    def arg(ref):
        if ref[0] == "buf":
            _, n, off, nb = ref
            parent = bo(n, pl.buffers[n])
            if off == 0 and nb == pl.buffers[n]:
                return parent
            if (n, off, nb) not in views:
                views[(n, off, nb)] = pyxrt.bo(parent, nb, off)
            return views[(n, off, nb)]
        if ref[0] == "w":
            return bo("w:" + ref[1], wsize[ref[1]])
        raise SystemExit(f"argument kind {ref[0]} not supported here")

    cfg_runs = {s: [pyxrt.run(pyxrt.ext.kernel(ctx, k)) for k in ks] for s, ks in man["cfg"].items()}
    flip = {s: 0 for s in cfg_runs}

    def configure(s):
        r = cfg_runs[s][flip[s]]
        flip[s] ^= 1
        return timed(r)

    def timed(r):
        t0 = time.perf_counter()
        r.start()
        st = r.wait()
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"{st}")
        return (time.perf_counter() - t0) * 1e3

    ops = []
    for want in names:
        o = next(o for o in pl.ops if o["stream"] == want)
        r = pyxrt.run(pyxrt.ext.kernel(ctx, f"{o['set']}:{want}"))
        for i, ref in enumerate(o["args"]):
            r.set_arg(i, arg(ref))
        ops.append((o["set"], want, r))

    print(f"== {R}x{R}, {a.reps} reps, median ms (start to wait, host round trip included)")
    print(f"  {'op':28s} {'configure':>9s} {'first':>8s} {'again':>8s} {'first-again':>11s}")
    for s, name, r in ops:
        other = next(x for x in cfg_runs if x != s)
        c, f, g = [], [], []
        for _ in range(a.reps + 2):
            configure(other)                     # leave the array on another set
            c.append(configure(s))
            f.append(timed(r))
            g.append(timed(r))
        c, f, g = c[2:], f[2:], g[2:]            # the first two warm the context
        m = statistics.median
        print(f"  {s + ':' + name:28s} {m(c):9.3f} {m(f):8.3f} {m(g):8.3f} {m(f) - m(g):11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
