r"""loadpdi_probe: what an in-firmware reconfiguration costs (one hardware context, one
full ELF holding several designs' PDIs), for comparison with a context switch between two
xclbins (~2.2 ms, utilities/dit-chain/switch_probe.py).

The designs are mlir-aie's test/npu-xrt/reconfigure_loadpdi pair (one core adding 2 or 3
to four i32s). The main runtime sequence either alternates them N times (`alt`: 2N
reconfigurations) or configures and runs one design 2N times (`same`: the baseline -- the
firmware skips a load_pdi of the PDI already loaded). Per reconfiguration = (alt - same) / 2N. Each of aiecc's three ways to
reconfigure is built and timed:

    loadpdi    load_pdi: the firmware loads the design's PDI
    write32s   --expand-load-pdis: load_pdi expanded into register writes in the stream
    ctrlpkt    --load-pdi-to-ctrl-pkt: the configuration streamed as control packets

    . C:\dev\mlir-aie\iron_env.ps1
    xrt-smi configure --pmode turbo
    python utilities\reconfig-probe\loadpdi_probe.py --work C:\dev\switch-work [--n 32]
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pyxrt

MODES = {
    "loadpdi": ["--get-full-elf"],
    "write32s": ["--get-full-elf", "--expand-load-pdis"],
    "ctrlpkt": ["--get-full-elf", "--load-pdi-to-ctrl-pkt", "--get-ctrlpkt"],
}


def design(name: str, add: int) -> str:
    return f"""
    aie.device(npu2) @{name} {{
        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)
        aie.objectfifo @objfifo_in (%t00, {{%t02}}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @objfifo_out(%t02, {{%t00}}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.core(%t02) {{
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            %cadd = arith.constant {add} : i32
            %c_intmax = arith.constant 0xFFFFFE : index
            scf.for %niter = %c0 to %c_intmax step %c1 {{
                %sin  = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<4xi32>>
                %sout = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<4xi32>>
                %ein  = aie.objectfifo.subview.access %sin [0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
                %eout = aie.objectfifo.subview.access %sout[0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
                scf.for %i = %c0 to %c4 step %c1 {{
                    %0 = memref.load %ein[%i] : memref<4xi32>
                    %1 = arith.addi %0, %cadd : i32
                    memref.store %1, %eout[%i] : memref<4xi32>
                }}
                aie.objectfifo.release @objfifo_in (Consume, 1)
                aie.objectfifo.release @objfifo_out(Produce, 1)
            }}
            aie.end
        }}
        aie.runtime_sequence @{name}_sequence(%a : memref<4xi32>) {{
            %t_in = aiex.dma_configure_task_for @objfifo_in {{
                aie.dma_bd(%a : memref<4xi32> offset = 0 len = 4)
                aie.end
            }}
            %t_out = aiex.dma_configure_task_for @objfifo_out {{
                aie.dma_bd(%a: memref<4xi32> offset = 0 len = 4)
                aie.end
            }} {{issue_token = true}}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
            aiex.dma_free_task(%t_in)
        }}
    }}"""


RUN = """
                %v{i} = memref.subview %arg[0] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 0>>
                %a{i} = memref.reinterpret_cast %v{i} to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 0>> to memref<4xi32>
                aiex.run @{d}_sequence (%a{i}) : (memref<4xi32>)"""


def module(kind: str, n: int) -> str:
    # one run per configure either way (a configure's BDs are freed at its end); `same`
    # names one design every time, and the firmware skips a load_pdi of the loaded PDI
    ds = ("add_two", "add_three") if kind == "alt" else ("add_two", "add_two")
    body = "".join(f"\n            aiex.configure @{d} {{{RUN.format(d=d, i=2 * j + k)}\n            }}"
                   for j in range(n) for k, d in enumerate(ds))
    return (f"module {{\n    aie.device(npu2) @main {{\n"
            f"        aie.runtime_sequence @sequence(%arg : memref<512xi32>) {{{body}\n"
            f"        }}\n    }}\n{design('add_two', 2)}\n{design('add_three', 3)}\n}}\n")


def build(work: Path, mode: str, kind: str, n: int) -> Path:
    d = work / f"{mode}_{kind}_{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "aie.mlir").write_text(module(kind, n))
    if not (d / "aie.elf").exists():
        r = subprocess.run(["aiecc", *MODES[mode], "aie.mlir"], cwd=d,
                           capture_output=True, text=True)
        if r.returncode or not (d / "aie.elf").exists():
            raise SystemExit(f"aiecc {mode} {kind} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return d / "aie.elf"


def time_elf(dev, elf_path: Path, reps: int, expect_add: int) -> float:
    elf = pyxrt.elf(str(elf_path))
    ctx = pyxrt.hw_context(dev, elf)
    kernel = pyxrt.ext.kernel(ctx, "main:sequence")
    bo = pyxrt.ext.bo(dev, 512 * 4)
    host = np.frombuffer(bo.map(), dtype=np.int32)
    t = []
    for i in range(reps + 1):
        host[:] = np.arange(512, dtype=np.int32)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = pyxrt.run(kernel)
        run.set_arg(0, bo)
        t0 = time.perf_counter()
        run.start()
        run.wait2()
        dt = (time.perf_counter() - t0) * 1e3
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        got = host[:4] - np.arange(4)
        if not (got == expect_add).all():
            raise SystemExit(f"{elf_path}: added {got.tolist()}, expected {expect_add}")
        if i:                                   # the first run is the warm-up
            t.append(dt)
    return statistics.median(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--modes", default=",".join(MODES))
    a = ap.parse_args()
    work = Path(a.work)
    dev = pyxrt.device(0)
    print(f"{2 * a.n} runs per sequence; median of {a.reps} (ms)")
    print(f"  {'mode':9s} {'same':>8s} {'alt':>8s} {'per reconfiguration':>20s}")
    for mode in a.modes.split(","):
        try:
            same = time_elf(dev, build(work, mode, "same", a.n), a.reps, 2 * 2 * a.n)
            alt = time_elf(dev, build(work, mode, "alt", a.n), a.reps, 5 * a.n)
        except (SystemExit, RuntimeError) as e:
            print(f"  {mode:9s} FAILED: {str(e)[:2000]}")
            continue
        print(f"  {mode:9s} {same:8.3f} {alt:8.3f} {(alt - same) / (2 * a.n):20.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
