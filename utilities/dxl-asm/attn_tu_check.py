"""Compile attention TUs for Granite's dx point to Peano assembly (a fast syntax / codegen check before a design build) and
print each function's bundle count: python attn_tu_check.py OUTDIR TU.cc [TU.cc ...] -- [extra -D flags]."""
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
    out.mkdir(parents=True, exist_ok=True)
    args = sys.argv[2:]
    tus, extra = (args[:args.index("--")], args[args.index("--") + 1:]) if "--" in args else (args, [])
    for tu in tus:
        src = REPO / "open_kernels/designs/attn" / tu
        asm = out / (Path(tu).stem + ".s")
        cmd = [config.peano_cxx_path(), str(src), "-S", "-o", str(asm), f"-I{config.cxx_header_path()}", "-std=c++20",
               "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-Wno-empty-body", "-O2", "-DNDEBUG",
               "-D__AIE_API_AIE_ADF_HPP__", "--target=aie2p-none-unknown-elf", "-Os", *BASE, *extra]
        for d in include_dirs() + [str(src.parent)]:
            cmd += ["-I", d]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{tu}:\n{r.stderr[-4000:]}")
        lines = asm.read_text().splitlines()
        funcs, cur, n = [], None, 0
        for ln in lines:
            m = re.match(r"^([A-Za-z_][\w.$]*):", ln)
            if m and not m.group(1).startswith(".L"):
                if cur:
                    funcs.append((cur, n))
                cur, n = m.group(1), 0
            elif cur and ln.strip() and not ln.lstrip().startswith(("//", ".")) and not re.match(r"^\S+:", ln):
                n += 1
        if cur:
            funcs.append((cur, n))
        print(tu, [(f, c) for f, c in funcs if c > 2])


main()
