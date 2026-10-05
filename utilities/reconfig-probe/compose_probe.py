r"""compose_probe: klein's real kernel sets in ONE hardware context. It composes built
streams (their final.prj/aie.mlir, from export_dit_kernels.py) into a full ELF: each set
becomes a named device, and every op a `main` sequence `configure @<set> { run }`, so an
op is the kernel `main:<set>_<stream>` and the firmware reconfigures the array between
ops of different sets (skipping a reload of the loaded one). Then it times the ops alone
and alternating, queued and waited, like switch_probe.py does for the xclbin contexts.

    . C:\dev\mlir-aie\iron_env.ps1
    xrt-smi configure --pmode turbo
    python utilities\reconfig-probe\compose_probe.py --kernels C:\dev\klein-kernels --work C:\dev\switch-work\compose
        [--ops gemm:r512_sgl_out,ew:r512_res_all,ew:r512_qk_sgl,fa:r512_attn_sgl] [--mode loadpdi|write32s|ctrlpkt]

Buffer contents are whatever the BOs hold (zeros): only time is measured.
"""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pyxrt

SET_DIRS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}
MODES = {
    "loadpdi": ["--get-full-elf"],
    "write32s": ["--get-full-elf", "--expand-load-pdis"],
    "ctrlpkt": ["--get-full-elf", "--load-pdi-to-ctrl-pkt", "--get-ctrlpkt"],
}
SEQ_RE = re.compile(r"aie\.runtime_sequence\((.*?)\) \{")
ELEM_BYTES = {"bf16": 2, "i32": 4, "f32": 4, "i8": 1, 'v8bfp16ebs8">>': 9 / 8}


def arg_types(sig: str) -> list[str]:
    return [t.strip() for t in re.findall(r"%arg\d+: (memref<[^>]*(?:>>|>))", sig)]


def nbytes(t: str) -> int:
    m = re.match(r"memref<(\d+)x(.*)", t)
    n, elem = int(m.group(1)), m.group(2)
    for k, b in ELEM_BYTES.items():
        if elem.startswith(k) or elem.endswith(k):
            return int(n * b)
    raise ValueError(t)


def compose(kdir: Path, ops: list[tuple[str, str]], work: Path) -> tuple[str, dict]:
    """The module text and {op: its main sequence's argument types}."""
    devices, types = {}, {}
    for kset, stream in ops:
        prj = kdir / SET_DIRS[kset] / "build" / stream / "final.prj"
        text = (prj / "aie.mlir").read_text()
        body = text.strip()
        assert body.startswith("module {") and body.endswith("}"), prj
        body = body[len("module {"):-1].strip()
        sig = SEQ_RE.search(body).group(1)
        types[(kset, stream)] = arg_types(sig)
        if kset in devices:                      # the same static configuration: add a sequence
            seq = body[body.index("aie.runtime_sequence("):body.rindex("}")].rstrip()
            devices[kset] = devices[kset].rstrip()[:-1].rstrip() + "\n    " + \
                seq.replace("aie.runtime_sequence(", f"aie.runtime_sequence @{stream}(", 1) + "\n  }"
        else:
            body = body.replace("aie.device(npu2) {", f"aie.device(npu2) @{kset} {{", 1)
            devices[kset] = body.replace("aie.runtime_sequence(", f"aie.runtime_sequence @{stream}(", 1)
        for o in prj.glob("*.o"):
            shutil.copy2(o, work / o.name)
    mains = []
    for (kset, stream), ts in types.items():
        params = ", ".join(f"%a{i}: {t}" for i, t in enumerate(ts))
        args = ", ".join(f"%a{i}" for i in range(len(ts)))
        mains.append(f"    aie.runtime_sequence @{kset}_{stream}({params}) {{\n"
                     f"      aiex.configure @{kset} {{\n"
                     f"        aiex.run @{stream} ({args}) : ({', '.join(ts)})\n"
                     f"      }}\n    }}")
    text = "module {\n  aie.device(npu2) @main {\n" + "\n".join(mains) + "\n  }\n  " + \
        "\n  ".join(devices.values()) + "\n}\n"
    return text, types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ops", default="gemm:r512_sgl_out,ew:r512_res_all,ew:r512_qk_sgl,fa:r512_attn_sgl")
    ap.add_argument("--mode", default="loadpdi", choices=list(MODES))
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    ops = [tuple(o.split(":")) for o in a.ops.split(",")]
    work = Path(a.work) / a.mode
    work.mkdir(parents=True, exist_ok=True)
    text, types = compose(Path(a.kernels), ops, work)
    mlir = work / "aie.mlir"
    if not (work / "aie.elf").exists() or not mlir.exists() or mlir.read_text() != text:
        (work / "aie.mlir").write_text(text)
        t0 = time.time()
        r = subprocess.run(["aiecc", *MODES[a.mode], "aie.mlir"], cwd=work, capture_output=True, text=True)
        if r.returncode or not (work / "aie.elf").exists():
            err = [ln for ln in (r.stdout + r.stderr).splitlines() if "error" in ln.lower()]
            raise SystemExit("aiecc failed:\n" + "\n".join(err[:20] or [r.stdout[-3000:], r.stderr[-3000:]]))
        print(f"built {work / 'aie.elf'} in {time.time() - t0:.0f} s")

    dev = pyxrt.device(0)
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(work / "aie.elf")))
    bos, kern = {}, {}
    for op, ts in types.items():
        kern[op] = pyxrt.ext.kernel(ctx, f"main:{op[0]}_{op[1]}")
        # one BO per (op, argument): sized from the sequence's memref types
        bos[op] = [pyxrt.ext.bo(dev, nbytes(t)) for t in ts]

    # an op's own device sequence: no configure, valid only while its set is the one loaded
    raw = {op: pyxrt.ext.kernel(ctx, f"{op[0]}:{op[1]}") for op in types}

    def start(op, bare=False):
        r = pyxrt.run(raw[op] if bare else kern[op])
        for i, b in enumerate(bos[op]):
            r.set_arg(i, b)
        r.start()
        return r

    def timed(seq, n, queued):
        runs = []
        t0 = time.perf_counter()
        for _ in range(n):
            for op in seq:
                r = start(op)
                if queued:
                    runs.append(r)
                else:
                    r.wait2()
        for r in runs:
            r.wait2()
        return (time.perf_counter() - t0) * 1e3 / n

    for op in types:                                   # warm-up
        timed([op], 2, True)
    print(f"== {a.mode}: median ms over {a.reps} reps")
    alone = {}
    for op in types:
        q = statistics.median(timed([op], 5, True) for _ in range(a.reps // 5 or 1))
        w = statistics.median(timed([op], 1, False) for _ in range(a.reps))
        alone[op] = (q, w)
        print(f"  {op[0] + ':' + op[1]:24s} queued {q:8.3f}  waited {w:8.3f}")
    names = list(types)
    cycles = [names[i:i + 2] for i in range(len(names) - 1)] + ([names] if len(names) > 2 else [])
    for cyc in cycles:
        if len({o[0] for o in cyc}) < 2:
            continue
        for queued in (True, False):
            t = statistics.median(timed(cyc, 5, queued) for _ in range(a.reps // 5 or 1))
            base = sum(alone[o][0 if queued else 1] for o in cyc)
            nsw = sum(1 for x, y in zip(cyc, cyc[1:] + cyc[:1]) if x[0] != y[0])
            print(f"  {' -> '.join(o[0] + ':' + o[1] for o in cyc):64s} {'queued' if queued else 'waited'} "
                  f"{t:8.3f}  per switch {(t - base) / nsw:7.3f}")

    # configure only on a set change: each op once with its configure, then k bare runs
    k = 3

    def bare_alone(op, n):
        runs = [start(op)]
        for r in runs:
            r.wait2()
        t0 = time.perf_counter()
        runs = [start(op, bare=True) for _ in range(n)]
        for r in runs:
            r.wait2()
        return (time.perf_counter() - t0) * 1e3 / n

    def pattern(cyc, n):
        runs = []
        t0 = time.perf_counter()
        for _ in range(n):
            for op in cyc:
                runs.append(start(op))
                runs += [start(op, bare=True) for _ in range(k)]
        for r in runs:
            r.wait2()
        return (time.perf_counter() - t0) * 1e3 / n

    print(f"  configure on a set change only ({k} bare runs after each configured one), queued:")
    bare = {op: statistics.median(bare_alone(op, 5) for _ in range(a.reps // 5 or 1)) for op in types}
    for op in types:
        print(f"    {op[0] + ':' + op[1]:24s} bare {bare[op]:8.3f}")
    for cyc in cycles:
        if len({o[0] for o in cyc}) < 2:
            continue
        t = statistics.median(pattern(cyc, 5) for _ in range(a.reps // 5 or 1))
        base = sum((k + 1) * bare[o] for o in cyc)
        nsw = sum(1 for x, y in zip(cyc, cyc[1:] + cyc[:1]) if x[0] != y[0])
        print(f"    {' -> '.join(o[0] + ':' + o[1] for o in cyc):62s} {t:8.3f}  per switch {(t - base) / nsw:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
