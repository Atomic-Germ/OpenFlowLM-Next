"""Run build_design.py while capturing every external-kernel compile invocation.

Keeps copies of the compiled objects in <capture_dir> for post-mortem symbol
checks (the aiecc pipeline deletes final.prj on failure, which hid why the
f32-scale gemv wrappers linked as the wrong symbol).
"""
import os
import shutil
import sys
from pathlib import Path

import aie.utils.compile.utils as cu

CAPTURE = Path(os.environ.get("EF_CAPTURE_DIR", "/tmp/opencode/ef_capture"))
CAPTURE.mkdir(parents=True, exist_ok=True)

_orig = cu.compile_cxx_core_function


def spy(*args, **kwargs):
    src = args[0] if args else kwargs.get("source_path")
    out = kwargs.get("output_path", args[1] if len(args) > 1 else "?")
    print(f"[ef] src={src} out={out} args={kwargs.get('compile_args', args[3] if len(args) > 3 else '?')}",
          flush=True)
    if src and Path(src).exists():
        shutil.copy2(src, CAPTURE / (Path(src).name + ".txt"))
    r = _orig(*args, **kwargs)
    if out and Path(out).exists():
        dst = CAPTURE / Path(out).name
        shutil.copy2(out, dst)
        print(f"[ef] captured {dst}", flush=True)
    return r


cu.compile_cxx_core_function = spy
sys.argv = sys.argv[:1] + sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("bd", str(Path(__file__).resolve().parent.parent / "open_kernels" / "build_design.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.exit(mod.main())
