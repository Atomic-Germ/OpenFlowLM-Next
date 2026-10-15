r"""compose_prefill_elf: a dense family's bfp16 prefill streams (the GEMMs and the attention tiers
plus the prep stream) as ONE full ELF, so a prompt's layers run in one hardware context and a
switch between the GEMM array and the attention array is a register-write reconfiguration
(0.3 / 0.7 ms) instead of an xclbin context switch (1.2-2.4 ms). Skill: full-elf-context.

Sets are the manifest contexts the streams run in (`gemmbfp`, `af`); a stream is a manifest
kernel. Builds are assembled as compose_elf.py does (one configuration build, one build per
stream, aiebu-asm last). Writes <kernels>/prefill.elf and <kernels>/prefill_elf.json.

    . C:\dev\mlir-aie\iron_env.ps1
    python open_kernels\compose_prefill_elf.py --kernels <kernel dir> [--jobs 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import compose_elf as ce

HERE = Path(__file__).resolve().parent
DESIGNS = HERE / "designs"
SET_CONTEXTS = ("gemmbfp", "af")
ELF = "prefill.elf"
META = "prefill_elf.json"


def streams_of(man: dict) -> dict[str, list[str]]:
    out = {c: [] for c in SET_CONTEXTS}
    for name, k in man["kernels"].items():
        if k["context"] in out:
            out[k["context"]].append(name)
    for c, ns in out.items():
        if not ns:
            raise SystemExit(f"the manifest has no kernel in context {c}")
    return out


def arg_bytes(prj: Path) -> list[int]:
    sig = ce.SEQ_RE.search((prj / "aie.mlir").read_text()).group(1)
    out = []
    for n, t in re.findall(r"memref<(\d+)x(.*?)>(?:,|$)", sig.replace("), ", "),")):
        out.append(int(n) * (9 if "bfp" in t else 4 if t.startswith(("f32", "i32")) else 1 if t.startswith("i8") else 2))
    return out


def build_elf(kdir: Path, designs: Path = DESIGNS, jobs: int = 4, log=print) -> Path:
    t0 = time.time()
    man = json.loads((kdir / "manifest.json").read_text(encoding="utf-8"))
    streams = streams_of(man)
    prjs = {}
    for s, ns in streams.items():
        for n in ns:
            p = designs / man["builds"][n]["build_dir"] / "final.prj"
            if not (p / "aie.mlir").is_file():
                raise SystemExit(f"{p / 'aie.mlir'} is missing: rebuild kernel {n}")
            prjs[(s, n)] = p
    work = kdir / "elf_build"
    sets = list(streams)

    first = {s: streams[s][0] for s in sets}
    cfg_text = ce.module_text([ce.device_text(prjs[(s, first[s])], s, first[s]) for s in sets], sets)
    cfg_objs = list({o.name: o for s in sets for o in ce._objs(prjs[(s, first[s])])}.values())
    for s in sets:
        for o in ce._objs(prjs[(s, first[s])]):
            twin = next(x for x in cfg_objs if x.name == o.name)
            if hashlib.sha256(twin.read_bytes()).hexdigest() != hashlib.sha256(o.read_bytes()).hexdigest():
                raise SystemExit(f"kernel object {o.name} differs between sets")

    def one(key):
        s, n = key
        return key, ce._aiecc(work / s / n, ce.module_text([ce.device_text(prjs[key], s, n)], [s]),
                              ce._objs(prjs[key]))

    log(f"compose_prefill_elf: {len(prjs)} streams in {len(sets)} sets, {jobs} builds at a time")
    with ThreadPoolExecutor(jobs) as ex:
        cfg_f = ex.submit(ce._aiecc, work / "_config", cfg_text, cfg_objs)
        built = dict(ex.map(one, prjs))
        cfg_prj = cfg_f.result()
    log(f"  built in {time.time() - t0:.0f} s")

    cfg = json.loads((cfg_prj / "full_elf_config.json").read_text())
    kernels = {k["name"]: k for k in cfg["xrt-kernels"]}
    pdis = kernels["main"]["PDIs"]
    empties = tuple(sorted(p["id"] for p in pdis if Path(p["PDI_file"]).stem.startswith("empty")))
    if len(empties) != 2:
        raise SystemExit(f"the configuration build has {len(empties)} empty devices, expected 2")
    cfg_dir = work / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_inst = []
    for s in sets:
        a, b = ce.cfg_variants((cfg_prj / f"npu_insts_full_elf_main_cfg_{s}.bin").read_bytes(), empties)
        for name, code in ((f"cfg_{s}_a", a), (f"cfg_{s}_b", b)):
            f = cfg_dir / f"{name}.bin"
            if not f.is_file() or f.read_bytes() != code:
                f.write_bytes(code)
            cfg_inst.append({"TXN_ctrl_code_file": str(f.resolve()), "id": name})

    def args(n):
        return [{"name": f"arg_{i}", "offset": hex(8 * i), "type": "char *"} for i in range(n)]

    out_kernels = [{"name": "main", "PDIs": pdis, "arguments": kernels["main"]["arguments"], "instance": cfg_inst}]
    nargs, sizes = {}, {}
    for s in sets:
        inst = []
        for n in streams[s]:
            f = built[(s, n)] / f"npu_insts_full_elf_{s}_{n}.bin"
            if not f.is_file():
                raise SystemExit(f"{f} is missing")
            inst.append({"TXN_ctrl_code_file": str(f.resolve()), "id": n})
            nargs[n] = ce.arg_count(prjs[(s, n)])
            sizes[n] = arg_bytes(prjs[(s, n)])
        out_kernels.append({"name": s, "PDIs": pdis, "arguments": args(max(nargs[n] for n in streams[s])),
                            "instance": inst})
    cfg_json = work / "full_elf_config_prefill.json"
    cfg_json.write_text(json.dumps({"xrt-kernels": out_kernels}, indent=2))
    tmp = kdir / (ELF + ".tmp")
    r = subprocess.run(["aiebu-asm", "-t", "aie2_config", "-j", str(cfg_json), "-o", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode or not tmp.is_file():
        raise SystemExit(f"aiebu-asm failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    tmp.replace(kdir / ELF)
    meta = {"elf": ELF, "sets": sets, "streams": streams, "nargs": nargs, "arg_bytes": sizes,
            "cfg": {s: [f"main:cfg_{s}_a", f"main:cfg_{s}_b"] for s in sets}}
    (kdir / META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"  {ELF} {(kdir / ELF).stat().st_size / 2**20:.2f} MiB, done in {time.time() - t0:.0f} s")
    return kdir / META


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--designs", default=str(DESIGNS))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    build_elf(Path(a.kernels).resolve(), Path(a.designs).resolve(), a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
