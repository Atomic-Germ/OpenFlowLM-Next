r"""switch_probe: what a kernel-set switch costs, split into the host round trip and the
array's reconfiguration. Runs real ops of klein_pipeline's schedule on their real kernel
sets and buffer shapes (the contents are whatever the BOs hold: only time is measured).

    . C:\dev\mlir-aie\iron_env.ps1
    xrt-smi configure --pmode turbo
    python utilities\dit-chain\switch_probe.py --kernels C:\dev\klein-kernels [--size 512] [--reps 40]

Per op X, with n = --reps:
    queued     n runs of X started back to back on one context, one wait at the end
    waited     start X, wait, n times (the host round trip on every run)
and per pair (X in set A, Y in set B):
    alternate  start X, wait, start Y, wait, n times
    switch     (alternate - waited X - waited Y) / 2: what one context change adds
A pair inside one set measures what changing instruction streams costs (expected ~0).
--timer1 repeats everything with the Windows timer at 1 ms (timeBeginPeriod), in case
XRT's wait sleeps on the system tick.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
import klein_pipeline as kp  # noqa: E402
from npu_host import COMPLETED, Npu  # noqa: E402

SETS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}


class Probe:
    def __init__(self, kdir: Path, R: int):
        self.pl = kp.plan(R)
        self.npu = Npu()
        self.sets = {n: self.npu.kernel_set(n, kdir / d) for n, d in SETS.items()}
        specs = {n: K * N * 9 // 8 for n, (K, N, _) in kp.dit_weight_specs().items()}
        specs |= {n: K * N * 9 // 8 for n, (K, N, _) in kp.te_weight_specs().items()}
        self.wsize = specs
        self.bufs, self.w = {}, {}

    def _arg(self, ref):
        if ref[0] == "buf":
            _, n, off, nb = ref
            if n not in self.bufs:
                self.bufs[n] = self.npu.buf(n, self.pl.buffers[n])
            return self.bufs[n].view(off, nb)
        if ref[0] == "w":
            n = ref[1]
            if n not in self.w:
                self.w[n] = self.npu.buf(f"w:{n}", self.wsize[n])
            return self.w[n].bo
        raise ValueError(f"op argument {ref[0]} is not supported by the probe")

    def op(self, stream: str, phase: str | None = None):
        """The schedule's first op running `stream` (optionally in `phase`)."""
        for o in self.pl.ops:
            if o["stream"] == stream and (phase is None or o["phase"] == phase):
                st = self.sets[o["set"]].stream(stream)
                return o["set"], st, [self._arg(a) for a in o["args"]]
        raise KeyError(stream)


def check(h, what):
    st = h.wait()
    if st != COMPLETED:
        raise RuntimeError(f"{what}: {st}")


def queued(op, n):
    _, st, args = op
    t0 = time.perf_counter()
    hs = [st.start(*args) for _ in range(n)]
    for h in hs:
        check(h, st.name)
    return (time.perf_counter() - t0) * 1e3 / n


def waited(op, n):
    _, st, args = op
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        check(st.start(*args), st.name)
        t.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(t)


def alternate(x, y, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        check(x[1].start(*x[2]), x[1].name)
        check(y[1].start(*y[2]), y[1].name)
        t.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--timer1", action="store_true")
    a = ap.parse_args()
    R = a.size
    p = Probe(Path(a.kernels), R)
    ops = {
        "gemm:t_emb1": p.op("t_emb1"),                 # the smallest ops of each set
        "ew:t_silu": p.op("t_silu"),
        "fa:te_attn": p.op("te_attn"),
        "gemm:te_o": p.op("te_o"),
        f"gemm:r{R}_sgl_out": p.op(f"r{R}_sgl_out"),
        f"ew:r{R}_res_all": p.op(f"r{R}_res_all"),
        f"ew:r{R}_qk_sgl": p.op(f"r{R}_qk_sgl"),
        f"fa:r{R}_attn_sgl": p.op(f"r{R}_attn_sgl"),
    }
    pairs = [
        ("gemm:t_emb1", "gemm:te_o"),                 # same context, two streams
        ("gemm:t_emb1", "ew:t_silu"),
        ("ew:t_silu", "fa:te_attn"),
        ("fa:te_attn", "gemm:t_emb1"),
        (f"gemm:r{R}_sgl_out", f"ew:r{R}_res_all"),
        (f"ew:r{R}_qk_sgl", f"fa:r{R}_attn_sgl"),
        (f"fa:r{R}_attn_sgl", f"gemm:r{R}_sgl_out"),
    ]
    for o in ops.values():                            # warm every context and stream
        queued(o, 3)

    def one_pass(tag):
        print(f"== {tag} ({R}x{R}, {a.reps} reps, median ms)")
        print(f"  {'op':24s} {'queued':>8s} {'waited':>8s} {'round trip':>10s}")
        w = {}
        for k, o in ops.items():
            q = queued(o, a.reps)
            w[k] = waited(o, a.reps)
            print(f"  {k:24s} {q:8.3f} {w[k]:8.3f} {w[k] - q:10.3f}")
        print(f"  {'pair':44s} {'alternate':>9s} {'per switch':>10s}")
        for x, y in pairs:
            alt = alternate(ops[x], ops[y], a.reps)
            print(f"  {x + ' <-> ' + y:44s} {alt:9.3f} {(alt - w[x] - w[y]) / 2:10.3f}")

    one_pass("default timer")
    if a.timer1:
        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod(1)
        try:
            one_pass("timeBeginPeriod(1)")
        finally:
            winmm.timeEndPeriod(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
