r"""preempt_probe: does the one-context replay survive another process using the NPU?

A full-ELF context reconfigured by register writes (compose_elf.py's cfg_<set> kernels)
holds a configuration the firmware does not know about. If another context runs between
two of our commands, ours comes back without it and the next op hangs
(ERT_CMD_STATE_TIMEOUT, 2026-09-29). This loops real ops of the resolution's ELF on
random inputs, in same-set stretches `cfg_<set>, op, ...`, and checks every output
against the first iteration's, in one of two modes:

    queued    every run started on its own, all queued on the context
    runlist   each stretch submitted as one xrt::runlist ("executed atomically")

Run it alone, then with contention from another process, e.g.
    python utilities\dit-chain\switch_probe.py --kernels C:\dev\klein-kernels --reps 400

    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\reconfig-probe\preempt_probe.py --kernels C:\dev\klein-kernels-elf --mode runlist [--seconds 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyxrt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
import klein_pipeline as kp  # noqa: E402

OPS = ["t_emb1", "t_silu", "r512_attn_dbl", "r512_sgl_out", "r512_res_all", "r512_qk_sgl"]
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True, help="an installed v2 kernel set")
    ap.add_argument("--mode", choices=("queued", "runlist"), required=True)
    ap.add_argument("--seconds", type=float, default=30)
    a = ap.parse_args()
    kdir = Path(a.kernels)
    man = json.loads((kdir / "diffusion_kernels.json").read_text())
    pl = kp.plan(512)
    dev = pyxrt.device(0)
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(kdir / man["elf"]["512"])))
    wsize = {n: K * N * 9 // 8 for n, (K, N, _) in (kp.dit_weight_specs() | kp.te_weight_specs()).items()}
    rng = np.random.default_rng(0)
    bufs, views = {}, {}

    def bo(name, nbytes):
        if name not in bufs:
            b = pyxrt.ext.bo(dev, nbytes)
            h = np.frombuffer(b.map(), np.uint8)
            h[:] = rng.integers(0, 256, h.size, dtype=np.uint8) & 0x3F   # finite bf16s
            b.sync(TO)
            bufs[name] = (b, h)
        return bufs[name][0]

    def arg(ref):
        if ref[0] == "buf":
            _, n, off, nb = ref
            parent = bo(n, pl.buffers[n])
            if off == 0 and nb == pl.buffers[n]:
                return parent, (n, 0, nb)
            key = (n, off, nb)
            if key not in views:
                views[key] = pyxrt.bo(parent, nb, off)
            return views[key], key
        if ref[0] == "w":
            return bo("w:" + ref[1], wsize[ref[1]]), ("w:" + ref[1], 0, wsize[ref[1]])
        raise SystemExit(f"argument kind {ref[0]} not supported here")

    chosen = []
    for want in OPS:
        o = next(o for o in pl.ops if o["stream"] == want)
        k = pyxrt.ext.kernel(ctx, f"{o['set']}:{want}")
        args = [arg(r) for r in o["args"]]
        chosen.append((o["set"], want, k, args))
    # v2 manifest: two configure kernels per set (_a/_b) that differ only in the empty device
    # they reset the array with; the engine alternates them across stretches
    cfg = {s: [pyxrt.ext.kernel(ctx, n) for n in man["cfg"][s]] for s in {c[0] for c in chosen}}

    def run_of(k, args):
        r = pyxrt.run(k)
        for i, (b, _) in enumerate(args):
            r.set_arg(i, b)
        return r

    # the ops as the engine issues them: a stretch per run of same-set ops, each stretch
    # starting with its set's configuration
    stretches = []                                   # (set, cfg run, [(name, run, args)])
    for s, name, k, args in chosen:
        if not stretches or stretches[-1][0] != s:
            stretches.append((s, pyxrt.run(cfg[s][len(stretches) % 2]), []))
        stretches[-1][2].append((name, run_of(k, args), args))
    ops = [op for st in stretches for op in st[2]]
    lists = []
    if a.mode == "runlist":
        for s, c, rs in stretches:
            rl = pyxrt.runlist(ctx)
            rl.add(c)
            for _, r, _ in rs:
                rl.add(r)
            lists.append(rl)

    def outputs():
        # every op's last argument range, as written (the op's output, for these streams)
        out = []
        for name, _, args in ops:
            b, (n, off, nb) = args[-1]
            parent, host = bufs[n]
            parent.sync(FROM, nb, off)
            out.append(host[off:off + nb].copy())
        return out

    # every iteration starts from the same inputs (several of these ops write in place)
    initial = {n: h.copy() for n, (_, h) in bufs.items()}

    def restore():
        for n, (b, h) in bufs.items():
            h[:] = initial[n]
            b.sync(TO)

    def one_iteration():
        restore()
        if a.mode == "runlist":
            for rl in lists:
                rl.execute()
            for rl in lists:
                rl.wait()
        else:
            for _, c, rs in stretches:
                c.start()
                for _, r, _ in rs:
                    r.start()
            for _, c, rs in stretches:
                c.wait2()
                for _, r, _ in rs:
                    r.wait2()

    one_iteration()
    ref = outputs()
    n, bad, t0 = 0, 0, time.time()
    while time.time() - t0 < a.seconds:
        try:
            one_iteration()
        except Exception as e:
            print(f"iteration {n}: {str(e).splitlines()[0]}", flush=True)
            return 1
        got = outputs()
        wrong = [ops[i][0] for i, (x, y) in enumerate(zip(ref, got)) if not np.array_equal(x, y)]
        if wrong:
            bad += 1
            print(f"iteration {n}: outputs differ in {wrong}", flush=True)
        n += 1
    print(f"{a.mode}: {n} iterations of {len(ops)} ops in {len(stretches)} stretches in {a.seconds:.0f} s, "
          f"{bad} with a wrong output", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
