r"""check_small_gemms: klein's step-boundary GEMMs, zero-padded to dit_gemm's shape rules.

    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\dit-chain\check_small_gemms.py --kernels C:\dev\klein-kernels [--size 512]

x_embedder reads the latents [T, 128] with row stride 128 as K = 512 against zero weight
rows 128..511 (the reads overlap the next rows; they multiply by zero). proj_out's N = 128
is padded to 1024 with zero weight columns. Random bf16 data; reference: float64 with
the bfp16 weights the kernel multiplies by.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "dit_gemm"))
from npu_host import Npu  # noqa: E402
from pack import pack_b, unpack_b  # noqa: E402

H = 3072


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    T = (a.size // 16) ** 2
    rng = np.random.default_rng(0)
    npu = Npu()
    gm = npu.kernel_set("gemm", Path(a.kernels))
    ok = True

    def check(name, got, ref):
        nonlocal ok
        rf = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        passed = bool(np.isfinite(got).all()) and rf < 3e-2
        ok &= passed
        print(f"  {name}: rel_fro {rf:.3e}  {'PASS' if passed else 'FAIL'}")

    # x_embedder: [T, 128] @ [128, 3072]
    lat = rng.standard_normal((T, 128)).astype(bfloat16)
    w = (rng.standard_normal((128, H)) / np.sqrt(128)).astype(np.float32)
    wp = np.zeros((512, H), np.float32)
    wp[:128] = w
    pk = pack_b(wp)
    A = npu.buf("A", (T * 128 + 512) * 2)
    A.write(np.concatenate([lat.reshape(-1), np.ones(512, bfloat16)]))
    B = npu.buf("B", pk.nbytes)
    B.write(pk)
    C = npu.buf("C", T * H * 2)
    gm.stream(f"r{a.size}_x_emb").run(A.bo, B.bo, C.bo)
    got = C.read(np.uint16).view(bfloat16).astype(np.float64).reshape(T, H)
    check("x_emb (K 128 read as 512)", got, lat.astype(np.float64) @ unpack_b(pk, 512, H)[:128])

    # proj_out: [T, 3072] @ [3072, 128], N padded to 1024
    x = rng.standard_normal((T, H)).astype(bfloat16)
    w = (rng.standard_normal((H, 128)) / np.sqrt(H)).astype(np.float32)
    wp = np.zeros((H, 1024), np.float32)
    wp[:, :128] = w
    pk = pack_b(wp)
    A2 = npu.buf("A2", T * H * 2)
    A2.write(x)
    B2 = npu.buf("B2", pk.nbytes)
    B2.write(pk)
    C2 = npu.buf("C2", T * 1024 * 2)
    gm.stream(f"r{a.size}_proj_out").run(A2.bo, B2.bo, C2.bo)
    got = C2.read(np.uint16).view(bfloat16).astype(np.float64).reshape(T, 1024)
    check("proj_out (N 128 of 1024)", got[:, :128],
          x.astype(np.float64) @ unpack_b(pk, H, 1024)[:, :128])
    pad = float(np.abs(got[:, 128:]).max())
    ok &= pad == 0.0
    print(f"  proj_out padding columns max |value| {pad}  {'PASS' if pad == 0 else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
