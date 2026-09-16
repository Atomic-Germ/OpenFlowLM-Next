#!/usr/bin/env python3
"""What Peano did with a kernel TU: II bounds, bundle counts, slot warnings.

Compiles one .cc the way IRON does, then asks llc for its backend remarks.
No xclbin, no hardware - the numbers come out of the compiler, so a design
change can be judged in seconds instead of a build-and-run cycle.

    python open_kernels/kernel_remarks.py designs/gemv_q4/gemv_q4_p4b32r4_k0.cc

Read the output as: a loop costs `bundles` cycles per iteration, and MII =
max(ResMII, RecMII) is the floor any schedule has to respect. ResMII is set by
whichever VLIW slot the body uses most; RecMII by the longest dependence cycle
carried around the loop. Close to MII means the body itself has to change -
hints won't help. Well above it means the scheduler left something on the table.

The bounds come from the pipeliner's debug channel, which does not say which
loop each one belongs to. With one loop in the file that is unambiguous; with
several, treat them as a set and split the file before drawing conclusions.

Run under WSL with the ironenv venv active (see designs/*/README or AGENTS.md).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

# llc prints these to stderr under -debug-only; they are the only numbers here
# that don't come from a remark.
_MII = re.compile(r"(Res|Rec)MII=(\d+)")
_ARG = re.compile(r"-\s+([A-Za-z][\w-]*):\s+'?([^\n']*?)'?\s*$", re.M)


def _paths() -> tuple[Path, Path]:
    from aie.utils import config

    return Path(config.peano_install_dir()), Path(config.cxx_header_path())


def _arch() -> str:
    """aie2p, which is what build_design.py pins the device to.

    IRON's own _detect_arch() reads the NPU that is present, and under WSL there is
    none, so it falls back to aie2 - a different slot layout and different II bounds
    for the same loop. Pass --arch to look at anything else.
    """
    return "aie2p"


def _clang_cmd(src: Path, arch: str, extra: list[str]) -> list[str]:
    peano, hdr = _paths()
    return [
        str(peano / "bin" / "clang++"), str(src),
        f"-I{hdr}", f"-I{hdr / 'aie_kernels'}", f"-I{hdr / 'aie_kernels' / arch}",
        f"-I{HERE / 'include'}",
        "-D__AIE_API_AIE_ADF_HPP__", f"--target={arch}-none-unknown-elf",
        "-std=c++20", "-O2", "-DNDEBUG",
        "-Wno-deprecated-declarations", "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body", *extra,
    ]


def emit_ir(src: Path, out: Path, arch: str, extra: list[str]) -> None:
    r = subprocess.run(
        _clang_cmd(src, arch, extra) + ["-S", "-emit-llvm", "-o", str(out)],
        cwd=src.parent, capture_output=True, text=True,
    )
    if r.returncode:
        errs = [l for l in r.stderr.splitlines() if ": error: " in l]
        sys.exit("\n  ".join(["compile failed:"] + (errs[:10] or [r.stderr[-800:]])))


def count_bankless_loads(src: Path, arch: str, extra: list[str], obj: Path) -> int:
    """Loads the compiler had to slot without knowing which memory bank they hit.

    Only the driver's own codegen reports these; running llc over pre-emitted IR
    does not.
    """
    out = subprocess.run(
        _clang_cmd(src, arch, extra)
        + ["-c", "-o", str(obj), "-Rpass-missed=aie-multi-slot-pseudo"],
        cwd=src.parent, capture_output=True, text=True,
    )
    return out.stderr.count("[-Rpass-missed=aie-multi-slot-pseudo]")


def run_llc(ir: Path, arch: str, work: Path) -> tuple[list[dict], str]:
    peano, _ = _paths()
    llc = str(peano / "bin" / "llc")
    base = [llc, "-O2", f"-mtriple={arch}"]

    yaml = work / "remarks.yaml"
    subprocess.run(
        base + [str(ir), "-o", str(work / "out.s"), f"-pass-remarks-output={yaml}",
                "-pass-remarks-filter=postpipeliner|aie-hardware-loops|aie-asm-printer"],
        check=True, stderr=subprocess.DEVNULL,
    )
    records = [
        dict(_ARG.findall(r), _pass=(re.search(r"Pass:\s+(\S+)", r) or [None, "?"])[1])
        for r in yaml.read_text().split("--- !") if r.strip()
    ]

    # ResMII/RecMII only surface through the pipeliner's debug channel.
    mii = subprocess.run(
        base + [str(ir), "-o", "/dev/null", "-debug-only=postpipeliner"],
        capture_output=True, text=True,
    ).stderr
    return records, mii


def report(records: list[dict], mii_log: str, slots: int) -> None:
    res = sorted({int(v) for k, v in _MII.findall(mii_log) if k == "Res"})
    rec = sorted({int(v) for k, v in _MII.findall(mii_log) if k == "Rec"})
    if res or rec:
        print(f"II bounds seen: ResMII={res or '-'}  RecMII={rec or '-'}")
        if len(res) <= 1 and len(rec) <= 1:
            print(f"  -> MII = {max(res + rec)} (the floor any schedule must respect)")
        else:
            print("  -> several loops in this TU; the bounds above are not attributed "
                  "to any one of them")

    zol = {r["BasicBlock"]: r["Zero-Overhead-Loop"]
           for r in records if r["_pass"] == "aie-hardware-loops"}
    pipelined = [r for r in records if r["_pass"] == "postpipeliner"]

    print("\nblock                                     bundles  bytes  zero-overhead")
    for r in records:
        if r["_pass"] != "aie-asm-printer":
            continue
        bb = r.get("BasicBlock") or "(unnamed)"
        z = zol.get(bb, "")
        print(f"{bb[:40]:<40}  {r['BundleCount']:>7}  {r['ByteCount']:>5}  {z}")

    print()
    if pipelined:
        for r in pipelined:
            print(f"pipelined: {r.get('Loop','')} II={r.get('II')} NS={r.get('NS')} "
                  f"prologue={r.get('PrologueBundles')} epilogue={r.get('EpilogueBundles')}")
    else:
        print("pipelined: none - no loop got a software-pipelined schedule")
    print(f"loads slotted without a bank annotation: {slots}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path, help="kernel .cc to compile")
    ap.add_argument("--arch", help="aie2 / aie2p (default: aie2p, what build_design.py targets)")
    ap.add_argument("--keep", type=Path, help="keep the .ll/.s/.yaml under this directory")
    ap.add_argument("cflags", nargs="*",
                    help="extra flags for the clang++ step (one source file only)")
    args = ap.parse_args()
    stray = [c for c in args.cflags if not c.startswith("-")]
    if stray:
        ap.error(f"one source at a time; these look like sources, not flags: {stray}")

    arch = args.arch or _arch()
    with tempfile.TemporaryDirectory() as tmp:
        work = args.keep or Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        src = args.source.resolve()
        ir = work / (args.source.stem + ".ll")
        emit_ir(src, ir, arch, list(args.cflags))
        slots = count_bankless_loads(src, arch, list(args.cflags), work / "out.o")
        print(f"{args.source.name}  ({arch})")
        report(*run_llc(ir, arch, work), slots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
