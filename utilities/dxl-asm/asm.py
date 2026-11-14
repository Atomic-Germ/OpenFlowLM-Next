"""Compile a dxl kernel TU to Peano assembly and print its loops' VLIW bundle counts (the cycle floor of each basic block)."""
import os, re, subprocess, sys
from pathlib import Path

SP = Path(r"C:\dev\mlir-aie\ironenv\Lib\site-packages")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SP / "mlir_aie" / "python"), str(REPO / "open_kernels")]
from aie.utils import config  # noqa: E402
from ironutil import include_dirs  # noqa: E402


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    extra = sys.argv[3:]
    cmd = [config.peano_cxx_path(), str(src), "-S", "-o", str(out), f"-I{config.cxx_header_path()}", "-std=c++20",
           "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-Wno-empty-body", "-O2", "-DNDEBUG",
           "-D__AIE_API_AIE_ADF_HPP__", "--target=aie2p-none-unknown-elf", "-Os", "-DDXL_L=4", "-DDXL_JMAX=32"]
    for d in include_dirs() + [str(src.parent), str(REPO / "open_kernels/designs/gemv_q4"), str(REPO / "open_kernels/designs/dxl")]:
        cmd += ["-I", d]
    cmd += extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    print("ok", out)
    lines = out.read_text().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if "Loop Header" in ln:
            start = i - 1
        m = re.match(r"\.L_LEnd\d+:", ln)
        if m and start is not None:
            body = [l for l in lines[start:i + 2] if l.strip() and not l.lstrip().startswith(("//", ".")) and not re.match(r"^\S+:", l)]
            print(f"loop at line {start}: {len(body)} bundles")
            start = None


main()
