r"""compose_elf: a diffusion kernel directory's six sets as one full ELF per resolution
(diffusion_r<R>.elf), so a whole image runs in one hardware context. Plan:
specs/open-diffusion/archive/one-context.md; measurements: utilities/reconfig-probe/README.md.

Each ELF holds only the streams that resolution's schedule (klein_pipeline.plan) runs:
every kernel an XRT context creates walks all of the ELF's control code (~2 ms per MB),
and the 1024^2 streams alone are ~9 MB.

The ELF holds, per kernel set, a device whose kernels are its streams (`<set>:<stream>`,
each the stream's own runtime sequence, which does not configure the array), and a
`main` device with one configure-only kernel per set (`main:cfg_<set>`). Built with
aiecc's --expand-load-pdis, a cfg kernel writes its set's configuration as register
writes in the instruction stream (0.3-0.7 ms; a context switch costs ~2.1 ms). A runner
issues cfg_<set> before an op only when the set changes.

te_attn's valid_len (the prompt's length) is baked into its control code: one RTP
write per core, found by diffing the exporter's valid_len probe build (as the xclbin
runners' patch table is). The ELF splits te_attn in two: `fa:te_attn_vl<n>`, those 32
writes with value n (n = 1..512, 784 bytes each), and `fa:te_attn`, everything else. A
runner issues the head for the prompt's length, then te_attn. Whole per-length copies
would add 21 MB.

The ELF is assembled rather than compiled as one module. aiecc's buffer-address pass
walks the whole device once per tile, and its per-core compile clones the whole module
once per core, so one module with every stream costs cores x sequences (67 streams:
671 s, ~10 GB). Here every piece is a small aiecc build:
  - one configuration build: `main` with every cfg_<set>, and each set's device with
    one stream -- all PDIs and cfg control code under ONE PDI numbering;
  - one build per stream (its device alone, plus main's cfg_<set>): that stream's
    control code. It is byte-identical to the same stream built inside the full module;
    the control code of a device's own sequence names no PDI;
then our full_elf_config.json and `aiebu-asm -t aie2_config`, aiecc's own last step.

    . C:\dev\mlir-aie\iron_env.ps1
    python open_kernels\compose_elf.py --kernels <built kernel dir> [--jobs 6]

Every build is kept under <kernels>/elf_build/ and reused while its input is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SET_DIRS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}
FLAGS = ["--get-full-elf", "--expand-load-pdis", "--no-progress"]
ELF_META = "diffusion_elf.json"


def elf_name(R: int) -> str:
    return f"diffusion_r{R}.elf"


VL_STREAM, VL_PROBE_STREAM, VL_BASE, VL_PROBE, VL_MAX = "te_attn", "te_attn_vlprobe", 512, 77, 512
SEQ_RE = re.compile(r"aie\.runtime_sequence\((.*?)\) \{")


def vl_kernel(n: int) -> str:
    return f"te_attn_vl{n}"


# ----------------------------------------------------------------------------- MLIR text

def _module_body(prj: Path) -> str:
    body = (prj / "aie.mlir").read_text().strip()
    if not (body.startswith("module {") and body.endswith("}")):
        raise SystemExit(f"{prj / 'aie.mlir'}: not a single-module design")
    return body[len("module {"):-1].strip()


def device_text(prj: Path, kset: str, stream: str) -> str:
    """The stream's device, named after its set, its runtime sequence after the stream."""
    body = _module_body(prj)
    if body.count("aie.device(npu2) {") != 1 or body.count("aie.runtime_sequence(") != 1:
        raise SystemExit(f"{prj}: expected one anonymous device with one runtime sequence")
    return body.replace("aie.device(npu2) {", f"aie.device(npu2) @{kset} {{", 1) \
               .replace("aie.runtime_sequence(", f"aie.runtime_sequence @{stream}(", 1)


def arg_count(prj: Path) -> int:
    sig = SEQ_RE.search((prj / "aie.mlir").read_text())
    return len(re.findall(r"%arg\d+:", sig.group(1)))


def module_text(devices: list[str], cfg_sets: list[str]) -> str:
    cfgs = "\n".join(f"    aie.runtime_sequence @cfg_{s}() {{\n      aiex.configure @{s} {{\n      }}\n    }}"
                     for s in cfg_sets)
    return "module {\n  aie.device(npu2) @main {\n" + cfgs + "\n  }\n  " + "\n  ".join(devices) + "\n}\n"


# ----------------------------------------------------------------------------- builds

def _aiecc(work: Path, text: str, objs: list[Path]) -> Path:
    """aiecc FLAGS on `text` in `work`, unless the kept build has the same input. Returns
    the project directory holding the PDIs, control code and full_elf_config.json."""
    work.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha256((text + "\0" + " ".join(FLAGS) + "\0" +
                            "".join(hashlib.sha256(o.read_bytes()).hexdigest() for o in objs))
                           .encode()).hexdigest()
    prj = work / "prj"
    if (work / "stamp").is_file() and (work / "stamp").read_text() == stamp and \
            (prj / "full_elf_config.json").is_file():
        return prj
    (work / "stamp").unlink(missing_ok=True)
    shutil.rmtree(prj, ignore_errors=True)
    for o in objs:
        shutil.copy2(o, work / o.name)
    (work / "aie.mlir").write_text(text)
    r = subprocess.run(["aiecc", *FLAGS, "--tmpdir", "prj", "aie.mlir"], cwd=work,
                       capture_output=True, text=True)
    if r.returncode or not (prj / "full_elf_config.json").is_file():
        err = [ln for ln in (r.stdout + r.stderr).splitlines() if "error" in ln.lower()]
        raise SystemExit(f"aiecc failed in {work}:\n" +
                         "\n".join(err[:20] or [r.stdout[-3000:], r.stderr[-3000:]]))
    (work / "stamp").write_text(stamp)
    return prj


def _objs(prj: Path) -> list[Path]:
    return sorted(prj.glob("*.o"))


def set_streams(kdir: Path) -> dict[str, list[str]]:
    """{set: [stream]} for every stream the exporter built (its marker's `streams`)."""
    markers = {"gemm": "dit_kernels.json", "fa": "dit_fa.json", "ew": "dit_ew.json",
               "conv": "dit_conv.json", "conv1": "dit_conv.json", "vew": "vae_ew.json"}
    out = {}
    for s, sub in SET_DIRS.items():
        m = json.loads((kdir / sub / markers[s]).read_text(encoding="utf-8"))
        if not m.get("complete"):
            raise SystemExit(f"{kdir / sub / markers[s]}: incomplete; build the set first")
        out[s] = list(m["streams"])
    return out


def valid_len_words(base: bytes, probe: bytes) -> list[int]:
    a, b = np.frombuffer(base, np.uint32), np.frombuffer(probe, np.uint32)
    if a.size != b.size:
        raise SystemExit(f"te_attn control code: the valid_len probe differs in length ({a.size} vs {b.size})")
    idx = np.nonzero(a != b)[0]
    if idx.size == 0 or (a[idx] != VL_BASE).any() or (b[idx] != VL_PROBE).any():
        raise SystemExit(f"te_attn control code: {idx.size} differing words are not all "
                         f"{VL_BASE} -> {VL_PROBE}; valid_len cannot be patched")
    return [int(i) for i in idx]


# The TXN control code (aie-rt's transaction format): a 16-byte header -- 6 bytes of
# device geometry, 2 spare, u32 op count, u32 total bytes -- then the ops back to back.
TXN_WRITE32, TXN_BLOCKWRITE, TXN_MASKWRITE, TXN_MASKPOLL, TXN_MASKPOLL_BUSY = 0, 1, 3, 4, 7
TXN_LOADPDI, LOADPDI_BYTES, LOADPDI_ID = 8, 16, 2   # op, pad; u16 pdi id; u32 size; u64 address
WRITE32_BYTES, WRITE32_VALUE = 24, 16          # op, col, row, pad; u64 reg; u32 value; u32 size


def txn_ops(code: bytes) -> list[tuple[int, int, int]]:
    """(opcode, offset, bytes) of every op; refused unless the ops fill the stream exactly."""
    n_ops, size = struct.unpack_from("<II", code, 8)
    if size != len(code):
        raise SystemExit(f"TXN header says {size} bytes, the stream has {len(code)}")
    ops, off = [], 16
    for _ in range(n_ops):
        op = code[off]
        if op == TXN_WRITE32:
            n = WRITE32_BYTES
        elif op in (TXN_MASKWRITE, TXN_MASKPOLL, TXN_MASKPOLL_BUSY):
            n = 28
        elif op == TXN_LOADPDI:
            n = LOADPDI_BYTES
        elif op == TXN_BLOCKWRITE:
            n = struct.unpack_from("<I", code, off + 12)[0]
        else:                                   # custom ops (TCT sync, DDR patch, ...) and the rest
            n = struct.unpack_from("<I", code, off + 4)[0]
        if n <= 0 or off + n > len(code):
            raise SystemExit(f"TXN op {op} at byte {off} has a bad length {n}")
        ops.append((op, off, n))
        off += n
    if off != len(code):
        raise SystemExit(f"TXN ops end at byte {off}, the stream at {len(code)}")
    return ops


def txn(code: bytes, ops: list[tuple[int, int, int]]) -> bytes:
    body = b"".join(code[o:o + n] for _, o, n in ops)
    return code[:8] + struct.pack("<II", len(ops), 16 + len(body)) + body


def split_valid_len(base: bytes, words: list[int]) -> tuple[bytes, bytes]:
    """te_attn's control code as (head, tail). The head is the write32 ops that write
    valid_len (one RTP word per core), the tail everything else in order. They are plain
    register writes into the cores' RTP buffers, with no argument patch, and all come
    before the first DMA op, so head-then-tail writes what te_attn writes. A per-length
    head is 784 bytes where a whole te_attn is 41 KB -- and every kernel an XRT context
    creates walks all of the ELF's control code (~1.2 ms per MB)."""
    ops = txn_ops(base)
    at = {w * 4 for w in words}
    head = [op for op in ops if any(op[1] <= a < op[1] + op[2] for a in at)]
    if len(head) != len(words) or any(op[0] != TXN_WRITE32 or (w * 4 - op[1]) != WRITE32_VALUE
                                      for op, w in zip(head, sorted(words))):
        raise SystemExit("te_attn control code: valid_len is not the value of one write32 per word")
    first_dma = next(i for i, op in enumerate(ops) if op[0] != TXN_WRITE32)
    if ops.index(head[-1]) >= first_dma:
        raise SystemExit("te_attn control code: a valid_len write comes after its first DMA op")
    return txn(base, head), txn(base, [op for op in ops if op not in head])


def cfg_variants(code: bytes, empties: tuple[int, int]) -> tuple[bytes, bytes]:
    """A cfg_<set> kernel as two variants that differ only in which empty device's PDI
    its first op loads. That load resets the array before the register writes, but the
    firmware skips a load_pdi of the PDI it last loaded: two cfgs naming the same empty
    back to back write over whatever the array holds -- including another process's
    design after a context switch (the next op then hangs). A runner alternates the
    variants, so every configuration starts from a real reset."""
    ops = txn_ops(code)
    op, off, _ = ops[0]
    pdi = struct.unpack_from("<H", code, off + LOADPDI_ID)[0]
    if op != TXN_LOADPDI or pdi not in empties:
        raise SystemExit(f"a cfg kernel does not start by loading an empty device (op {op}, PDI {pdi})")
    out = []
    for e in empties:
        b = bytearray(code)
        struct.pack_into("<H", b, off + LOADPDI_ID, e)
        out.append(bytes(b))
    return out[0], out[1]


def build_elf(kdir: Path, jobs: int = 6, log=print) -> Path:
    t0 = time.time()
    streams = set_streams(kdir)
    prjs = {(s, n): kdir / SET_DIRS[s] / "build" / n / "final.prj" for s, ns in streams.items() for n in ns}
    has_vl = VL_STREAM in streams.get("fa", [])
    if has_vl:
        prjs[("fa", VL_PROBE_STREAM)] = kdir / "fa" / "build" / VL_PROBE_STREAM / "final.prj"
    for k, p in prjs.items():
        if not (p / "aie.mlir").is_file():
            raise SystemExit(f"{p / 'aie.mlir'} is missing: rebuild stream {k[1]} of set {k[0]}")
    work = kdir / "elf_build"
    sets = list(streams)

    # the configuration build: every set's PDI and cfg code under one numbering
    first = {s: streams[s][0] for s in sets}
    cfg_text = module_text([device_text(prjs[(s, first[s])], s, first[s]) for s in sets], sets)
    cfg_objs = sorted({o for s in sets for o in _objs(prjs[(s, first[s])])}, key=lambda o: o.name)
    names = [o.name for o in cfg_objs]
    if len(names) != len(set(names)):
        dup = {n for n in names if names.count(n) > 1}
        for n in dup:
            if len({hashlib.sha256(o.read_bytes()).hexdigest() for o in cfg_objs if o.name == n}) > 1:
                raise SystemExit(f"kernel object {n} differs between sets")
        cfg_objs = list({o.name: o for o in cfg_objs}.values())

    def one(key):
        s, n = key
        return key, _aiecc(work / s / n, module_text([device_text(prjs[key], s, n)], [s]), _objs(prjs[key]))

    log(f"compose_elf: {sum(len(v) for v in streams.values())} streams in {len(sets)} sets, "
        f"{jobs} builds at a time -> {work}")
    with ThreadPoolExecutor(jobs) as ex:
        cfg_f = ex.submit(_aiecc, work / "_config", cfg_text, cfg_objs)
        built = dict(ex.map(one, prjs))
        cfg_prj = cfg_f.result()
    log(f"  built in {time.time() - t0:.0f} s")

    cfg = json.loads((cfg_prj / "full_elf_config.json").read_text())
    kernels = {k["name"]: k for k in cfg["xrt-kernels"]}
    if set(kernels) != {"main", *sets}:
        raise SystemExit(f"configuration build has kernels {sorted(kernels)}, expected main + {sets}")
    pdis = kernels["main"]["PDIs"]
    empties = tuple(sorted(p["id"] for p in pdis if Path(p["PDI_file"]).stem.startswith("empty")))
    if len(empties) != 2:
        raise SystemExit(f"the configuration build has {len(empties)} empty devices, expected 2")
    cfg_dir = work / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_inst = []
    for s in sets:
        a, b = cfg_variants((cfg_prj / f"npu_insts_full_elf_main_cfg_{s}.bin").read_bytes(), empties)
        for name, code in ((f"cfg_{s}_a", a), (f"cfg_{s}_b", b)):
            f = cfg_dir / f"{name}.bin"
            if not f.is_file() or f.read_bytes() != code:
                f.write_bytes(code)
            cfg_inst.append({"TXN_ctrl_code_file": str(f.resolve()), "id": name})

    def args(n):
        return [{"name": f"arg_{i}", "offset": hex(8 * i), "type": "char *"} for i in range(n)]

    def ctrl(key):
        s, n = key
        f = built[key] / f"npu_insts_full_elf_{s}_{n}.bin"
        if not f.is_file():
            raise SystemExit(f"{f} is missing")
        return f

    vl_dir = work / "vl"

    def write(f: Path, data: bytes) -> Path:
        if not f.is_file() or f.read_bytes() != data:
            f.write_bytes(data)
        return f

    vl_inst = []
    if has_vl:
        # te_attn = its valid_len head for the prompt's length, then the shared tail
        base = ctrl(("fa", VL_STREAM)).read_bytes()
        words = valid_len_words(base, ctrl(("fa", VL_PROBE_STREAM)).read_bytes())
        head, tail = split_valid_len(base, words)
        vl_dir.mkdir(parents=True, exist_ok=True)
        vl_inst.append({"TXN_ctrl_code_file": str(write(vl_dir / f"{VL_STREAM}.bin", tail).resolve()),
                        "id": VL_STREAM})
        hw = np.frombuffer(head, np.uint32).copy()
        at = [(op[1] + WRITE32_VALUE) // 4 for op in txn_ops(head)]
        for n in range(1, VL_MAX + 1):
            hw[at] = n
            vl_inst.append({"TXN_ctrl_code_file": str(write(vl_dir / f"{vl_kernel(n)}.bin", hw.tobytes()).resolve()),
                            "id": vl_kernel(n)})

    # One ELF per resolution, holding only the streams its schedule runs: XRT's kernel
    # creation walks all of an ELF's control code (~2 ms per MB per kernel), and the
    # 1024^2 streams alone are ~9 MB.
    import klein_pipeline as kp  # noqa: E402
    resolutions = json.loads((kdir / "dit_kernels.json").read_text(encoding="utf-8"))["resolutions"]
    elfs, used_by = {}, {}
    for R in resolutions:
        used = {s: [] for s in sets}
        for o in kp.plan(R).ops:
            if o["stream"] not in used[o["set"]]:
                used[o["set"]].append(o["stream"])
        missing = [f"{s}:{n}" for s, ns in used.items() for n in ns if n not in streams[s]]
        if missing:
            raise SystemExit(f"the {R}^2 schedule runs streams this kernel directory lacks: {missing}")
        out_kernels = [{"name": "main", "PDIs": pdis, "arguments": kernels["main"]["arguments"],
                        "instance": cfg_inst}]
        for s in sets:
            inst = [{"TXN_ctrl_code_file": str(ctrl((s, n)).resolve()), "id": n}
                    for n in used[s] if not (s == "fa" and n == VL_STREAM)]
            if s == "fa" and VL_STREAM in used[s]:
                inst += vl_inst
            if not inst:
                continue
            n_args = max(arg_count(prjs[(s, n)]) for n in used[s])
            out_kernels.append({"name": s, "PDIs": pdis, "arguments": args(n_args), "instance": inst})
        cfg_json = work / f"full_elf_config_r{R}.json"
        cfg_json.write_text(json.dumps({"xrt-kernels": out_kernels}, indent=2))
        elf = kdir / elf_name(R)
        tmp = kdir / (elf_name(R) + ".tmp")
        r = subprocess.run(["aiebu-asm", "-t", "aie2_config", "-j", str(cfg_json), "-o", str(tmp)],
                           capture_output=True, text=True)
        if r.returncode or not tmp.is_file():
            raise SystemExit(f"aiebu-asm failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        tmp.replace(elf)
        elfs[str(R)] = elf.name
        used_by[str(R)] = used
        log(f"  {elf} ({elf.stat().st_size / 2**20:.1f} MiB)")
    meta = {"elf": elfs, "sets": sets, "kernels": used_by,
            # alternate the two: see cfg_variants
            "cfg": {s: [f"main:cfg_{s}_a", f"main:cfg_{s}_b"] for s in sets},
            # an op of `stream` runs `head` (with n the prompt's length) first, then itself
            "valid_len": {"stream": VL_STREAM, "head": "te_attn_vl{n}", "max": VL_MAX} if has_vl else None}
    (kdir / ELF_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"  done in {time.time() - t0:.0f} s")
    return kdir / ELF_META


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernels", required=True, help="a kernel directory export_dit_kernels.py built")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    build_elf(Path(a.kernels).resolve(), a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
