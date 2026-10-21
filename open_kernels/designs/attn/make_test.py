r"""Prepare the attention gate's fixture: synthetic inputs, the harness program
and the fp64 reference the device is scored against.

    python make_test.py --spec open_kernels/recipes/specs/qwen3-0.6b.json \
                         --rows 17 --out <build dir>

The build (designs/attn/probe_dense.py) fixes the geometry and the window
length; this script fixes the numbers. It writes, into --out:

    cache<i>.bin   the window: [K_t | V_t] per row, bf16
    meta<i>.bin    [qn | kn] and the position record
    qg<i>.bin      q (fp32)
    kvn<i>.bin     [k | v] (fp32)
    ref-new<i>.bin the new row the kernel must produce (bf16)
    ref-og<i>.bin  the attention output the kernel must produce (fp32)
    poison-*.bin   NaN + a canary, loaded into the outputs before every run so a
                   run that does not write them is caught
    attention.cfg  the program open_kernels/harness/run_kernel executes
    attention-fixture.json  what was prepared, and the hashes of every file

Nothing here needs a model, a container or the engine: the gate is what makes a
new model SIZE checkable on a bare NPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from recipes import dense as D  # noqa: E402
from recipes.load import load_spec  # noqa: E402
from designs.attn import attn_reference as REF  # noqa: E402

GUARD = bytes([0xA5]) * 64


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="a kernel spec (recipes/specs/*.json)")
    ap.add_argument("--rows", type=int, required=True, help="the window's row count (the build's ATTN_PROBE_ROWS)")
    ap.add_argument("--out", required=True, help="the build directory (the xclbin and insts.bin live there)")
    ap.add_argument("--seed", type=int, default=2964)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(Path(a.spec))
    G = D.geometry(spec)
    nh, kvh, hd, rot = G.NH, G.KVH, G.HD, G.ROT
    e_a, nptab = G.KVW * 2, G.PTAB_ELEMS
    if a.rows < 1:
        raise SystemExit("make_test: --rows must be at least 1")
    if G.QKVB:
        raise SystemExit("make_test: a family with a q/k/v bias is not covered by this gate")

    rng = np.random.default_rng(a.seed)
    qn = rng.uniform(0.8, 1.2, (hd,)).astype(bfloat16)      # the q/k norm weights
    kn = rng.uniform(0.8, 1.2, (hd,)).astype(bfloat16)
    cs = REF.cos_sin(a.rows, rot, spec.rope_theta)            # the position record's cos / sin

    meta = np.zeros((1 + nptab) * e_a, np.uint8)
    meta[:hd * 2] = qn.view(np.uint8)
    meta[hd * 2:hd * 4] = kn.view(np.uint8)
    rec = e_a                                          # the record's first element
    meta[rec:rec + 8] = np.array([a.rows, a.rows], np.int32).view(np.uint8)
    meta[rec + 512:rec + 512 + cs.nbytes] = cs.view(np.uint8)

    new_bytes = 2 * G.KVW * 2                           # bf16 bytes
    og_bytes = nh * hd * 2
    (out / "poison-new.bin").write_bytes(np.full(2 * G.KVW, np.nan, bfloat16).tobytes() + GUARD)
    (out / "poison-og.bin").write_bytes(np.full(nh * hd, np.nan, bfloat16).tobytes() + GUARD)

    cfg = ["device", "xclbin a final.xclbin", "kernelx a a insts.bin",
           f"buf meta {len(meta)}", f"buf qg {nh * hd * 4}", f"buf kvn {2 * G.KVW * 4}",
           f"buf cache {a.rows * 2 * G.KVW * 2}", f"buf new {new_bytes + len(GUARD)}",
           f"buf og {og_bytes + len(GUARD)}"]
    files = ["final.xclbin", "insts.bin", "poison-new.bin", "poison-og.bin"]
    cases = []

    def case(name: str, q, k, v, cache_k, cache_v):
        i = len(cases)
        qg = np.ascontiguousarray(q, np.float32)
        kvn = np.ascontiguousarray(np.concatenate([k, v], 0), np.float32)
        ck = np.ascontiguousarray(cache_k, bfloat16)
        cv = np.ascontiguousarray(cache_v, bfloat16)
        new, og = REF.decode(q, k, v, qn, kn, cs, ck, cv, nh=nh, kvh=kvh, hd=hd,
                             eps=spec.norm_eps, qk_norm=spec.qk_norm,
                             post=bool(G.QKNORM_POST))
        cache = np.concatenate([ck, cv], 1)               # [rows, 2, KVH, HD]
        for prefix, data in (("meta", meta), ("qg", qg), ("kvn", kvn),
                             ("ref-new", new), ("ref-og", og.astype(np.float32))):
            f = out / f"{prefix}{i}.bin"
            data.tofile(f)
            files.append(f.name)
        f = out / f"cache{i}.bin"
        f.write_bytes(cache.tobytes() + GUARD)
        files.append(f.name)
        cfg.extend([f"load cache cache{i}.bin", f"load meta meta{i}.bin", f"load qg qg{i}.bin",
                    f"load kvn kvn{i}.bin", "load new poison-new.bin", "load og poison-og.bin",
                    "run a meta qg kvn cache new og",
                    f"dump new new{i}.bin {new_bytes + len(GUARD)}",
                    f"dump og og{i}.bin {og_bytes + len(GUARD)}"])
        cases.append({"name": name})

    def randn(*shape):
        return rng.normal(0, 0.6, shape).astype(np.float32)

    # 1. the ordinary case: random q / k / v and a random window
    q, k, v = randn(nh, hd), randn(kvh, hd), randn(kvh, hd)
    case("random", q, k, v, randn(a.rows, kvh, hd), randn(a.rows, kvh, hd))
    # 2. the GQA mapping: every kv group gets its own value, so a head that
    #    reads the wrong group is wrong by a visible amount
    q = np.zeros((nh, hd), np.float32)
    k = np.zeros((kvh, hd), np.float32)
    v = np.repeat(np.arange(1, kvh + 1, dtype=np.float32)[:, None], hd, 1)
    case("groups", q, k, v,
         np.zeros((a.rows, kvh, hd), np.float32),
         np.repeat(np.arange(1, kvh + 1, dtype=np.float32)[None, :, None], a.rows, 0).repeat(hd, 2))
    # 3. a saturated softmax: one row wins, so the running max must be right
    q, k, v = randn(nh, hd) * 40, randn(kvh, hd) * 40, randn(kvh, hd)
    case("saturated", q, k, v, randn(a.rows, kvh, hd) * 40, randn(a.rows, kvh, hd))
    # 4. zero values: the output must be zero, not noise
    q, k, v = randn(nh, hd), randn(kvh, hd), np.zeros((kvh, hd), np.float32)
    case("zeros", q, k, v, randn(a.rows, kvh, hd), np.zeros((a.rows, kvh, hd), np.float32))

    (out / "attention.cfg").write_text("\n".join(cfg) + "\n")
    files.append("attention.cfg")
    meta_json = {"spec": str(a.spec), "rows": a.rows, "nh": nh, "kvh": kvh, "hd": hd, "rot": rot,
                 "nptab": nptab, "e_a": e_a, "cores": G.ACORES, "nhl": G.NHL, "rb": G.RB,
                 "vexp": G.VEXP, "qk_norm": bool(spec.qk_norm), "seed": a.seed, "cases": cases,
                 "sha256": {f: digest(out / f) for f in files}}
    (out / "attention-fixture.json").write_text(json.dumps(meta_json, indent=2) + "\n")
    print(f"prepared {len(cases)} cases: Q{nh}/KV{kvh}/HD{hd}/ROT{rot}, "
          f"cores {G.ACORES}x{G.NHL} heads, RB {G.RB}, rows {a.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
