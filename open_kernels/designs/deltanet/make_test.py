r"""Test vectors for dn_step: S_in from a real captured GDN state (prefill->decode
boundary of layer 0, C:/caps/pf_t11_full, state buffer layout: conv bf16[3,8192]
@0, S fp32[32,128,128] @49152), random unit k/q, random v, decay/beta in (0,1);
fp64 reference of decode_step.py's per-head recurrence.

    python make_test.py [--state FILE] [--seed 0]
Writes s_in.bin, vec.bin, ref_s.bin, ref_o.bin, run.cfg.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CAP = Path("/mnt/c/caps/pf_t11_full")
D, HEADS, VEC = 128, 32, 512


def load_state(path: Path) -> np.ndarray:
    raw = np.fromfile(path, np.uint8)
    return raw[49152:49152 + HEADS * D * D * 4].view(np.float32).reshape(HEADS, D, D).copy()


def reference(S: np.ndarray, k, q, v, decay, beta):
    S = S.astype(np.float64).copy()
    o = np.zeros((HEADS, D))
    for h in range(HEADS):
        S[h] *= decay[h]
        delta = beta[h] * (v[h] - S[h].T @ k[h])
        S[h] += np.outer(k[h], delta)
        o[h] = (S[h].T @ q[h]) / np.sqrt(D)
    return S.astype(np.float32), o.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.state:
        S = load_state(Path(a.state))
    else:
        man = json.loads((CAP / "boundary_manifest.json").read_text())
        sync = man["boundary_state_syncs"][0]["sync"]
        S = load_state(CAP / f"{sync}.bo")
    rng = np.random.default_rng(a.seed)
    k = rng.standard_normal((HEADS, D)); k /= np.linalg.norm(k, axis=1, keepdims=True)
    q = rng.standard_normal((HEADS, D)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.standard_normal((HEADS, D))
    decay = rng.uniform(0.85, 0.999, HEADS)
    beta = rng.uniform(0.05, 0.95, HEADS)
    # bf16-exact-free: keep everything fp32 as the kernel sees it
    k, q, v = k.astype(np.float32), q.astype(np.float32), v.astype(np.float32)
    decay, beta = decay.astype(np.float32), beta.astype(np.float32)

    vec = np.zeros((HEADS, VEC), np.float32)
    vec[:, 0:D] = k
    vec[:, D:2 * D] = q
    vec[:, 2 * D:3 * D] = v
    vec[:, 384] = decay
    vec[:, 385] = beta

    S_ref, o_ref = reference(S, k.astype(np.float64), q.astype(np.float64), v.astype(np.float64),
                             decay.astype(np.float64), beta.astype(np.float64))
    (HERE / "s_in.bin").write_bytes(S.tobytes())
    (HERE / "vec.bin").write_bytes(vec.tobytes())
    (HERE / "ref_s.bin").write_bytes(S_ref.tobytes())
    (HERE / "ref_o.bin").write_bytes(o_ref.tobytes())
    d = "C:/code/phlegm/tools/open-kernels/designs/deltanet"
    cfg = "\n".join([
        "device",
        f"xclbin G {d}/build/final.xclbin",
        f"kernelx k G {d}/build/insts.bin",
        f"buf s {S.nbytes} {d}/s_in.bin",
        f"buf v {vec.nbytes} {d}/vec.bin",
        f"buf so {S.nbytes}",
        f"buf o {o_ref.nbytes}",
        "run k s v so o",
        "run k s v so o",
        f"dump so {d}/y_s.bin {S.nbytes}",
        f"dump o {d}/y_o.bin {o_ref.nbytes}",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"S absmax={np.abs(S).max():.4g} S_ref absmax={np.abs(S_ref).max():.4g} "
          f"o_ref absmax={np.abs(o_ref).max():.4g} o_ref[0,:4]={o_ref[0, :4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
