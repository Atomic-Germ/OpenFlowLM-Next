"""Region census of a layer_x act dump, against the design's layout module.

    python utilities/act-region-report.py ACT.bin [SPEC.json]

A --dump-act <layer>:0:<bytes>:ACT.bin file holds one layer's act buffer; the
regions (A_XN / A_QKV / A_Z / A_VEC / A_O / A_OG / A_OUT / A_RES / A_XM gold
default the GGUF-direct f32 law; GEMV_SCALES_F32=0 reads the q4_1 one.
"""
import os
import sys
import dataclasses
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OK = HERE.parent / "open_kernels"
sys.path.insert(0, str(OK / "designs/layer_x"))
sys.path.insert(0, str(OK))

from ml_dtypes import bfloat16  # noqa: E402


def main() -> int:
    raw = np.fromfile(sys.argv[1], np.uint8)
    spec_path = len(sys.argv) > 2 and sys.argv[2]
    if spec_path:
        os.environ["OPEN_KERNELS_SPEC"] = str(Path(spec_path).resolve())
    if os.environ.get("GEMV_SCALES_F32", "1") in ("0", "false"):
        os.environ["GEMV_SCALES_F32"] = "0"
    import recipes.load
    from recipes.families import for_spec
    from recipes.load import load_spec, current_recipe

    spec = load_spec(Path(os.environ["OPEN_KERNELS_SPEC"]).resolve()) if spec_path else current_recipe()
    if os.environ.get("GEMV_SCALES_F32") == "1":
        spec = dataclasses.replace(spec, quant="q4_1_f32")
    L = for_spec(spec).layout(spec)

    n = int(L.A_BYTES)
    regions = [("XN", L.A_XN, L.A_QKV, bfloat16), ("QKV", L.A_QKV, L.A_Z, np.float32),
               ("Z", L.A_Z, L.A_VEC, np.float32), ("VEC", L.A_VEC, L.A_O, np.float32),
               ("O", L.A_O, L.A_OG, np.float32), ("OG", L.A_OG, L.A_OUT, bfloat16),
               ("OUT", L.A_OUT, L.A_RES, np.float32), ("RES", L.A_RES, L.A_XM, np.float32),
               ("XM", L.A_XM, max(L.A_XM, n), bfloat16)]
    fields = [f for f in ("A_ROUT", "A_HP") if int(getattr(L, f, 0)) > 0]
    if int(getattr(L, "A_H", 0)):
        regions.append(("H", L.A_H, min(int(L.A_H) + 65536, n), np.float32))
    print(f"act {len(raw)} B vs layout A_BYTES {n} ({spec.quant})")
    for name, s, e, dt in regions:
        s, e = int(s), min(int(e), len(raw))
        if e <= s:
            continue
        x = raw[s:e].view(dt).astype(np.float64)
        nf = int((~np.isfinite(x)).sum())
        fin = x[np.isfinite(x)]
        mx = np.abs(fin).max() if fin.size else float("nan")
        print(f"  {name:4} [{s:>7}:{e:>8}) {dt.__name__:8} nan {nf:5d}/{len(x):6d} maxabs {mx:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
