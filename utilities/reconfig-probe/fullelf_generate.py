r"""fullelf_generate: klein_pipeline's whole image in ONE hardware context. Every kernel
set becomes a device of one full ELF (aiecc --get-full-elf --expand-load-pdis): each
stream is its device's own runtime sequence (kernel `<set>:<stream>`), and each set gets
a configure-only kernel `main:cfg_<set>` that writes that set's configuration into the
array. The replay issues cfg_<set> only where the schedule changes set, and queues
everything on the one context -- no drain, no host round trip, no context switch.

Compare its pixels and time with utilities/dit-chain/generate.py on the same inputs:

    . C:\dev\mlir-aie\iron_env.ps1
    xrt-smi configure --pmode turbo
    python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels --size 512 --study C:\dev\ditref-out\goldens_pipe_512 --prompts 1 --ctx-ref --runs 3 --out <ref>
    python utilities\reconfig-probe\fullelf_generate.py --kernels C:\dev\klein-kernels --size 512 --study C:\dev\ditref-out\goldens_pipe_512 --ctx-ref --runs 3 --work C:\dev\switch-work\full --out <out> --ref <ref>

--ctx-ref only for now: te_attn's valid_len is patched into instruction words, which live
inside the ELF here (a scratchpad parameter would carry it).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyxrt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
sys.path.insert(0, str(ROOT / "utilities" / "dit-chain"))
import npu_host  # noqa: E402

SET_DIRS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}
FLAGS = ["--get-full-elf", "--expand-load-pdis"]


# --------------------------------------------------------------------------- composing

def device_text(prj: Path, kset: str, stream: str) -> str:
    body = (prj / "aie.mlir").read_text().strip()
    assert body.startswith("module {") and body.endswith("}"), prj
    body = body[len("module {"):-1].strip()
    return body.replace("aie.device(npu2) {", f"aie.device(npu2) @{kset} {{", 1) \
               .replace("aie.runtime_sequence(", f"aie.runtime_sequence @{stream}(", 1)


def sequence_text(prj: Path, stream: str) -> str:
    body = (prj / "aie.mlir").read_text()
    seq = body[body.index("aie.runtime_sequence("):]
    seq = seq[:seq.rindex("}", 0, seq.rindex("}", 0, seq.rindex("}")))].rstrip()  # drop device + module closers
    return seq.replace("aie.runtime_sequence(", f"aie.runtime_sequence @{stream}(", 1) + "\n    }"


def compose(kdir: Path, streams: dict[str, list[str]], work: Path) -> str:
    devices = []
    for kset, names in streams.items():
        prjs = [kdir / SET_DIRS[kset] / "build" / n / "final.prj" for n in names]
        dev = device_text(prjs[0], kset, names[0]).rstrip()
        assert dev.endswith("}"), kset
        extra = "\n    ".join(sequence_text(p, n) for p, n in zip(prjs[1:], names[1:]))
        devices.append(dev[:-1].rstrip() + ("\n    " + extra if extra else "") + "\n  }")
        for p in prjs:
            for o in p.glob("*.o"):
                dst = work / o.name
                if dst.exists() and dst.read_bytes() != o.read_bytes():
                    raise SystemExit(f"{o.name} differs between kernel sets")
                shutil.copy2(o, dst)
    cfgs = "\n".join(f"    aie.runtime_sequence @cfg_{s}() {{\n      aiex.configure @{s} {{\n      }}\n    }}"
                     for s in streams)
    return "module {\n  aie.device(npu2) @main {\n" + cfgs + "\n  }\n  " + "\n  ".join(devices) + "\n}\n"


def build(kdir: Path, streams: dict[str, list[str]], work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    text = compose(kdir, streams, work)
    elf = work / "aie.elf"
    if elf.exists() and (work / "aie.mlir").exists() and (work / "aie.mlir").read_text() == text:
        return elf
    (work / "aie.mlir").write_text(text)
    t0 = time.time()
    r = subprocess.run(["aiecc", *FLAGS, "aie.mlir"], cwd=work, capture_output=True, text=True)
    if r.returncode or not elf.exists():
        err = [ln for ln in (r.stdout + r.stderr).splitlines() if "error" in ln.lower()]
        raise SystemExit("aiecc failed:\n" + "\n".join(err[:20] or [r.stdout[-3000:], r.stderr[-3000:]]))
    print(f"built {elf} ({elf.stat().st_size / 2**20:.1f} MiB) in {time.time() - t0:.0f} s", flush=True)
    return elf


# --------------------------------------------------------------------------- the host

class _Stream:
    """generate.Runner's view of a stream; here only a name (the ELF holds the code)."""

    def __init__(self, kset, name):
        self.kset, self.name = kset, name

    def patch(self, words, value):
        raise SystemExit("instruction patching (te_attn valid_len) is not implemented on the "
                         "full-ELF path yet: use --ctx-ref")


class _KernelSet:
    def __init__(self, npu, name, directory):
        self.npu, self.name, self.dir = npu, name, Path(directory)
        self._streams = {}

    def stream(self, name):
        return self._streams.setdefault(name, _Stream(self, name))


def _buf_init(self, npu, name, nbytes):
    self.name, self.nbytes = name, nbytes
    self.bo = pyxrt.ext.bo(npu.dev, nbytes)
    self.host = np.frombuffer(self.bo.map(), dtype=np.uint8)
    self._views = {}


class Replay:
    def __init__(self, runner, elf: Path, skip=()):
        self.r = runner
        dev = runner.npu.dev
        self.ctx = pyxrt.hw_context(dev, pyxrt.elf(str(elf)))
        kern = {}
        cfg_k = {}
        self.runs = []                               # (is_cfg, xrt run) in dispatch order
        cur = None
        for kset, st, args, phase, _ in runner.ops:
            if phase in skip:
                continue
            if kset != cur:
                if kset not in cfg_k:
                    cfg_k[kset] = pyxrt.ext.kernel(self.ctx, f"main:cfg_{kset}")
                self.runs.append((True, pyxrt.run(cfg_k[kset])))
                cur = kset
            key = (kset, st.name)
            if key not in kern:
                kern[key] = pyxrt.ext.kernel(self.ctx, f"{kset}:{st.name}")
            run = pyxrt.run(kern[key])
            for i, bo in enumerate(args):
                run.set_arg(i, bo)
            self.runs.append((False, run))
        self.n_cfg = sum(1 for c, _ in self.runs if c)

    def run(self, window: int = 64) -> float:
        t0 = time.perf_counter()
        pending = []
        for _, run in self.runs:
            if len(pending) >= window:
                pending.pop(0).wait2()
            run.start()
            pending.append(run)
        for run in pending:
            run.wait2()
        return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--study", required=True)
    ap.add_argument("--prompts", type=int, default=1)
    ap.add_argument("--ctx-ref", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="generate.py's --out on the same inputs: compare")
    ap.add_argument("--window", type=int, default=64, help="runs in flight")
    a = ap.parse_args()
    if not a.ctx_ref:
        raise SystemExit("only --ctx-ref for now (see the module docstring)")
    skip = ("text",)

    npu_host.KernelSet = _KernelSet
    npu_host.Buf.__init__ = _buf_init
    import chain_test_vae as ctv
    import generate as gen
    import klein_pipeline as kp

    kdir = Path(a.kernels)
    pl = kp.plan(a.size)
    streams: dict[str, list[str]] = {}
    for o in pl.ops:
        if o["phase"] not in skip and o["stream"] not in streams.setdefault(o["set"], []):
            streams[o["set"]].append(o["stream"])
    print(f"{sum(len(v) for v in streams.values())} streams in {len(streams)} sets", flush=True)
    elf = build(kdir, streams, Path(a.work) / f"r{a.size}")

    gen.Npu = npu_host.Npu
    r = gen.Runner(kdir, a.size, 4)
    rp = Replay(r, elf, skip)
    print(f"{len(rp.runs) - rp.n_cfg} dispatches, {rp.n_cfg} configurations", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = Path(a.study)
    report = []
    for i in range(a.prompts):
        r.set_ctx(np.load(sd / f"ctx_{i}.npy"))
        noise = np.load(sd / f"noise_{i}.npy")
        for run in range(a.runs):
            r.set_noise(noise)
            s = rp.run(a.window)
            print(f"[{i:02d} run {run}] {s:.3f} s on the NPU (text skipped)", flush=True)
            report.append({"prompt": i, "run": run, "npu_s": s})
        img = r.image()
        ctv.write_png(out / f"{i:02d}.png", np.ascontiguousarray(img))
        lat = r.latents()
        np.save(out / f"lat_{i:02d}.npy", lat)
        if a.ref:
            ref = Path(a.ref) / f"lat_{i:02d}.npy"
            want = np.load(ref)
            same = np.array_equal(lat, want)
            diff = int(np.count_nonzero(lat != want))
            print(f"  latents vs {ref}: {'bit-identical' if same else f'{diff} of {lat.size} differ'}")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
