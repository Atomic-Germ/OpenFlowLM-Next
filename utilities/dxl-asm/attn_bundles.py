"""Compile attn_stepb.cc for Granite's dx point to Peano assembly and print attn_rowb_impl's bundle count per basic block (loop bodies repeat per block: score loop kRB or kNHL times, value loop kNHL times)."""
import re, subprocess, sys
from pathlib import Path

SP = Path(r"C:\dev\mlir-aie\ironenv\Lib\site-packages")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SP / "mlir_aie" / "python"), str(REPO / "open_kernels")]
from aie.utils import config  # noqa: E402
from ironutil import include_dirs  # noqa: E402

BASE = ["-DATTN_NH=40", "-DATTN_KVH=8", "-DATTN_HD=64", "-DATTN_ROT=64", "-DATTN_GATE=0", "-DATTN_QKNORM=0", "-DATTN_QKNORM_POST=0",
        "-DATTN_EPS=1e-05f", "-DATTN_VEXP=1", "-DATTN_NHL=8", "-DATTN_RB=4"]


def main():
    out = Path(sys.argv[1])
    src = REPO / "open_kernels/designs/attn/attn_stepb.cc"
    cmd = [config.peano_cxx_path(), str(src), "-S", "-o", str(out), f"-I{config.cxx_header_path()}", "-std=c++20",
           "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-Wno-empty-body", "-O2", "-DNDEBUG",
           "-D__AIE_API_AIE_ADF_HPP__", "--target=aie2p-none-unknown-elf", "-Os", *BASE, *sys.argv[2:]]
    for d in include_dirs() + [str(src.parent)]:
        cmd += ["-I", d]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    lines = out.read_text().splitlines()
    st = [i for i, ln in enumerate(lines) if ln.startswith("_Z14attn_rowb_impl")][-1]
    en = next(i for i in range(st, len(lines)) if lines[i].startswith(".Lfunc_end"))
    cur, cnt, blocks = "entry", 0, []
    for ln in lines[st + 1:en]:
        m = re.match(r"^(\.LBB\S+):", ln)
        if m:
            blocks.append((cur, cnt))
            cur, cnt = m.group(1), 0
        elif ln.strip() and not ln.lstrip().startswith(("//", ".")):
            cnt += 1
    blocks.append((cur, cnt))
    print(blocks, "total", sum(c for _, c in blocks))


main()
