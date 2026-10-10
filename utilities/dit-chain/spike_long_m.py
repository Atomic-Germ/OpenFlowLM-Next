r"""spike_long_m: do dit_gemm streams at an edit's joint length compute what the validated
text-to-image streams compute? (Phase 8, spike 8.1.5; specs/open-diffusion/plans/edits.md)

A stream's rows are independent, so a stream at M = 8704 (1024e1024: 512 text + 2 x 4096
image tokens) must give, bit for bit, the validated 1024 stream's output (M = 4608) on its
first 4608 rows, and on rows 4096..8703 the 1024 stream's output for those rows. The 1024
streams are checked end to end by the chain tests, so this checks the long ones with no
reference arithmetic of their own -- and covers their layouts (ldc, the SwiGLU epilogue,
gathered A), which make_test.py's reference does not model.

    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\dit-chain\spike_long_m.py --long C:\dev\edit-work\gemm8704 --pairs e_sgl_in=r1024_sgl_in,e_sgl_out=r1024_sgl_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "dit_gemm"))


def a_shape(s: dict) -> tuple[int, int]:
    lay = s.get("layout", {})
    lda = lay.get("lda", 2 * s["K"] if lay.get("a_gather") else s["K"])
    return s["M"], lda


def c_shape(s: dict) -> tuple[int, int]:
    return s["M"], s.get("layout", {}).get("ldc", s["N"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, help="kernel dir of the long streams")
    ap.add_argument("--short", default=r"C:\dev\klein-kernels", help="the validated set")
    ap.add_argument("--pairs", required=True, help="long=short,...")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from npu_host import Npu
    from pack import pack_b

    longd, shortd = Path(a.long), Path(a.short)
    ls = json.loads((longd / "dit_kernels.json").read_text())["streams"]
    ss = json.loads((shortd / "dit_kernels.json").read_text())["streams"]
    npu = Npu()
    kl = npu.kernel_set("long", longd)
    rng = np.random.default_rng(a.seed)
    ok_all = True
    results = []
    for pair in a.pairs.split(","):
        ln, sn = pair.split("=")
        sl, sh = ls[ln], ss[sn]
        assert (sl["K"], sl["N"]) == (sh["K"], sh["N"]), (sl, sh)
        assert {k: v for k, v in sl.get("layout", {}).items() if k != "a_size"} == \
            {k: v for k, v in sh.get("layout", {}).items() if k != "a_size"}, (sl, sh)
        M, Ms = sl["M"], sh["M"]
        (_, lda), (_, ldc) = a_shape(sl), c_shape(sl)
        A = rng.standard_normal((M, lda), dtype=np.float32).astype(bfloat16)
        W = pack_b((rng.standard_normal((sl["K"], sl["N"])) / np.sqrt(sl["K"])).astype(np.float32))
        Ab, Wb = npu.buf("A", A.nbytes + 4096), npu.buf("W", W.nbytes)
        Ab.write(A)
        Wb.write(W)
        Cl = npu.buf("Cl", M * ldc * 2)
        Cl.zero()
        kl.stream(ln).run(Ab.bo, Wb.bo, Cl.bo)
        got = Cl.read(np.uint16).reshape(M, ldc)
        results.append((ln, sn, M, Ms, lda, ldc, A, W, got))
    # the short streams in their own context, after the long ones are done
    ks = npu.kernel_set("short", shortd)
    for ln, sn, M, Ms, lda, ldc, A, W, got in results:
        Wb = npu.buf("W2", W.nbytes)
        Wb.write(W)
        bad = []
        for r0 in sorted({0, M - Ms}):
            Ab = npu.buf("As", Ms * lda * 2 + 4096)
            Ab.write(A[r0:r0 + Ms])
            Cs = npu.buf("Cs", Ms * ldc * 2)
            Cs.zero()
            ks.stream(sn).run(Ab.bo, Wb.bo, Cs.bo)
            ref = Cs.read(np.uint16).reshape(Ms, ldc)
            n = int((got[r0:r0 + Ms] != ref).sum())
            bad.append((r0, n, int((ref != 0).sum())))
        ok = all(n == 0 for _, n, _ in bad)
        ok_all &= ok
        det = ", ".join(f"rows {r0}+{Ms}: {n} differ ({nz} nonzero)" for r0, n, nz in bad)
        print(f"{ln} (M {M}) vs {sn} (M {Ms}): {det}  {'PASS' if ok else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
