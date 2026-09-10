r"""0167 Stage 19 / issue #32: does the GEMM route's per-layer accuracy
deficit (Stage 18: e_block ~2.36x e_seq, one layer) compound across the full
40-layer model into a DIFFERENT generated token, or stay a constant-factor
accuracy cost? The decisive test, per the coordinator: argmax / top-5
agreement on real tokens through all 40 layers, against the REAL sequential
hardware path (production `dx`, t=1) -- not against the fp64 replica (that
question, "is either route close to the idealized reference", was already
answered in Stage 18; this one is "would a user see a different token").

T_REAL=16 real, distinct tokens (embeddings), padded to T=256 for the GEMM
route so the ALREADY-BUILT `build_qkv3_t256` / `build_qkv_t256` /
`build_up_gate_t256` / `build_down_proj_t256` designs can be reused unchanged
-- Stage 18 already proved padding is bit-exact on hardware for the real
columns, independent of what the padding columns hold, so reusing that
result here (rather than building new T=16-sized designs) is not a new
assumption.

Both routes use LAYER-MAJOR order (process all 16 tokens through layer 0,
then layer 1, ...), not production's own TOKEN-MAJOR order (one token
through all 40 layers, then the next) -- the two are different topological
orderings of the SAME dependency graph (token i's layer-l input is layer
l-1's output for token i; layer l's causal KV needs tokens 0..i-1 already
processed AT layer l) and give numerically IDENTICAL results for a real
prefill (no token depends on a LATER token, only on itself one layer back and
on earlier tokens at the SAME layer). Layer-major is what both this task's
prior probes and the GEMM route's own natural batching already use; noted
explicitly so it is not mistaken for a hidden shortcut.

    python open_kernels/model/layer40_argmax_probe.py [--real 16]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "npu_offload" / "gemm_rtp"))

from recipes import pack as PK  # noqa: E402
from recipes import dense as QR  # noqa: E402
from recipes.qwen36moe import role_bytes  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from q4nx import Q4NX  # noqa: E402
import replica_dense as RD  # noqa: E402
from npue import tile_b  # noqa: E402

DEFAULT_MODEL_DIR = Path.home() / ".flm" / "models" / "Granite-4.2-3B-NPU2"
RUN_KERNEL = HERE.parent / "harness" / "out" / "run_kernel.exe"
GQP = DESIGNS / "gemm_q4_prefill"
DENSE = DESIGNS / "dense"
DX_BUILD = DESIGNS.parent.parent / "src" / "xclbins" / "Granite-4.2-3B-NPU2" / "open_kernels" / "dx"
K_TILE, MAC_S, MAC_T, TILE_N = 64, 8, 8, 32
T_COMPILED = 256
RUN_RE = re.compile(r"^run (\S+) \[\d+ bufs\] -> state (\d+) \((\d+\.\d+) ms\)")


def write(path, arr):
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def rms64(x, eps):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu64(x):
    return x / (1 + np.exp(-x))


def tile_x(x_tf32):
    x_bf16 = x_tf32.astype(bfloat16)
    x_kt = np.ascontiguousarray(x_bf16.T)
    return tile_b(x_kt.view(np.uint16), K_TILE, TILE_N, MAC_S, MAC_T, order="k,n").view(bfloat16)


def run_gemm(out_dir: Path, tag: str, build_dir: Path, w_list, x, t, warm=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    o = out_dir.as_posix()
    cfg = ["device", f"xclbin G {build_dir}/final.xclbin", f"kernelx k G {build_dir}/insts.bin"]
    write(out_dir / f"{tag}_x.bin", x)
    cfg.append(f"buf x {x.nbytes} {o}/{tag}_x.bin")
    for label, w_bytes, nw in w_list:
        write(out_dir / f"{tag}_w_{label}.bin", w_bytes)
        cfg.append(f"buf w_{label} {len(w_bytes)} {o}/{tag}_w_{label}.bin")
        cfg.append(f"buf y_{label} {nw * t * 4}")
    for label, _, _ in w_list:
        cfg += [f"run k w_{label} x y_{label}"] * warm
    for label, _, nw in w_list:
        cfg.append(f"dump y_{label} {o}/{tag}_y_{label}.bin {nw * t * 4}")
    cfg_path = out_dir / f"run_{tag}.cfg"
    cfg_path.write_text("\n".join(cfg) + "\n", newline="\n")
    r = subprocess.run([str(RUN_KERNEL), str(cfg_path)], cwd=str(out_dir), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError(f"run_kernel.exe exited {r.returncode} for {tag}")
    result = {}
    for label, _, nw in w_list:
        result[label] = np.fromfile(out_dir / f"{tag}_y_{label}.bin", np.float32).reshape(nw, t)
    return result


def gemm_layer(out_dir, q, spec, L, G, l, xres_real, warm=1):
    """One layer of the GEMM route on T_REAL real tokens (padded to 256).
    xres_real: [T_REAL, hid] f64. Returns [T_REAL, hid] f64 (layer output)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hid, nh, kvh, hd, ff = spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.intermediate
    qw, kvw = G.QW, G.KVW
    T_REAL = xres_real.shape[0]
    lt = spec.layer_types[l]
    plan = QR.pack_plan(spec)
    pool = PK.build_layer_pool(plan, lt, q, l)
    consts = PK.build_consts(plan, lt, q, l, L.CD_BYTES)
    max_ctx = 4096
    ptab = PK.ptab(max_ctx, spec.rotary_dim, spec.rope_theta, L.PTAB_ROW, spec.rope_inv_freq(), 0)
    write(out_dir / "pool_L.bin", pool)
    write(out_dir / "consts_L.bin", consts)
    write(out_dir / "ptab.bin", ptab)

    QB = role_bytes(spec, "attn", qw, hid)
    KB = role_bytes(spec, "attn", kvw, hid)
    OB = role_bytes(spec, "attn", hid, qw)
    UB = role_bytes(spec, "ffn", ff, hid)
    DB = role_bytes(spec, "ffn", hid, ff)
    w_qkv3 = np.concatenate([pool[L.POOL_Q:L.POOL_Q + QB], pool[L.POOL_K:L.POOL_K + KB],
                             pool[L.POOL_V:L.POOL_V + KB]])
    w_o = pool[L.POOL_O:L.POOL_O + OB]
    w_up = pool[L.POOL_UP:L.POOL_UP + UB]
    w_gate = pool[L.POOL_GATE:L.POOL_GATE + UB]
    w_down = pool[L.POOL_DOWN:L.POOL_DOWN + DB]
    n_qkv3 = qw + 2 * kvw

    def pad(x2d):
        p = np.zeros((T_COMPILED, x2d.shape[1]), np.float32)
        p[:T_REAL] = x2d.astype(np.float32)
        return p

    ln_w = q.bf16(f"model.layers.{l}.input_layernorm.weight")
    xnorm = pad(rms64(xres_real, spec.norm_eps) * ln_w)  # [256, hid]
    xa = tile_x(xnorm)
    ya = run_gemm(out_dir, "A", Path(GQP / "build_qkv3_t256"), [("qkv3", w_qkv3, n_qkv3)], xa, T_COMPILED, warm)["qkv3"]
    q_out = ya[0:qw, :T_REAL].T.astype(np.float64)
    k_out = ya[qw:qw + kvw, :T_REAL].T.astype(np.float64)
    v_out = ya[qw + kvw:, :T_REAL].T.astype(np.float64)

    # ---- attention: T_REAL real dxB dispatches, LOCAL positions 0..T_REAL-1 for THIS layer
    AD = L.AD_BYTES
    act0 = np.zeros(T_REAL * AD, np.uint8)
    for i in range(T_REAL):
        base = i * AD
        act0[base + L.AD_Q: base + L.AD_Q + qw * 4] = np.frombuffer(q_out[i].astype(np.float32).tobytes(), np.uint8)
        act0[base + L.AD_KVN: base + L.AD_KVN + kvw * 4] = np.frombuffer(k_out[i].astype(np.float32).tobytes(), np.uint8)
        act0[base + L.AD_KVN + kvw * 4: base + L.AD_KVN + 2 * kvw * 4] = np.frombuffer(
            v_out[i].astype(np.float32).tobytes(), np.uint8)
    write(out_dir / "act0.bin", act0)
    write(out_dir / "xres_dummy.bin", np.zeros(T_REAL * hid, np.float32))
    o = out_dir.as_posix()
    dxb_build = DENSE / "build_granite_h2560_dxB"
    cfgB = [
        "device", f"xclbin dxB {dxb_build}/final.xclbin", f"kernelx dxB dxB {dxb_build}/insts.bin",
        f"attngeom {L.KV_ROW} {L.PTAB_ROW} 0",
        f"buf pool {L.POOL_BYTES} {o}/pool_L.bin",
        f"buf xres {T_REAL * hid * 4} {o}/xres_dummy.bin",
        f"buf consts {L.CD_BYTES} {o}/consts_L.bin",
        f"buf kv {L.KV_BYTES}",
        f"buf act {T_REAL * AD} {o}/act0.bin",
        f"buf actb {AD}",
        f"buf ptab {L.PTAB_BYTES} {o}/ptab.bin",
    ]
    for i in range(T_REAL):
        off = i * AD
        cfgB += [f"attnpos dxB {i}", f"copy actb 0 act {off} {AD}", "run dxB pool xres consts kv actb ptab",
                 f"copy act {off} actb 0 {AD}"]
    cfgB.append(f"dump act {o}/act_final.bin {T_REAL * AD}")
    (out_dir / "run_B.cfg").write_text("\n".join(cfgB) + "\n", newline="\n")
    r = subprocess.run([str(RUN_KERNEL), str(out_dir / "run_B.cfg")], cwd=str(out_dir), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError("dxB run failed")
    act_final = np.fromfile(out_dir / "act_final.bin", np.uint8)

    def region(buf, tk, off, n, dt):
        base = tk * AD + off
        return np.frombuffer(buf[base:base + n * np.dtype(dt).itemsize], dtype=dt).astype(np.float32)

    og = np.stack([region(act_final, i, L.AD_OG, qw, bfloat16) for i in range(T_REAL)])  # [T_REAL, qw]

    xo = tile_x(pad(og))
    yo = run_gemm(out_dir, "O", Path(GQP / "build_qkv_t256"), [("o", w_o, hid)], xo, T_COMPILED, warm)["o"]
    attn_out = yo[:, :T_REAL].T.astype(np.float64)

    res1 = xres_real + attn_out
    post_ln_w = q.bf16(f"model.layers.{l}.post_attention_layernorm.weight")
    xm = pad(rms64(res1, spec.norm_eps) * post_ln_w)

    xg = tile_x(xm)
    ygu = run_gemm(out_dir, "GU", Path(GQP / "build_up_gate_t256"),
                   [("gate", w_gate, ff), ("up", w_up, ff)], xg, T_COMPILED, warm)
    gate_out = ygu["gate"][:, :T_REAL].T.astype(np.float64)
    up_out = ygu["up"][:, :T_REAL].T.astype(np.float64)
    h = pad((silu64(gate_out) * up_out).astype(np.float32))

    xd = tile_x(h)
    yd = run_gemm(out_dir, "D", Path(GQP / "build_down_proj_t256"), [("down", w_down, hid)], xd, T_COMPILED, warm)["down"]
    down_out = yd[:, :T_REAL].T.astype(np.float64)

    return res1 + down_out


def gevm_layer(out_dir, q, spec, L, l, xres_real):
    """One layer of the GEVM route (production `dx`, t=1) on T_REAL real
    tokens, own subprocess per layer (fresh xclbin context load each time --
    correctness-neutral; only adds harmless overhead vs production's single
    persistent context)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hid = spec.hidden
    T_REAL = xres_real.shape[0]
    lt = spec.layer_types[l]
    plan = QR.pack_plan(spec)
    pool = PK.build_layer_pool(plan, lt, q, l)
    consts = PK.build_consts(plan, lt, q, l, L.CD_BYTES)
    max_ctx = 4096
    ptab = PK.ptab(max_ctx, spec.rotary_dim, spec.rope_theta, L.PTAB_ROW, spec.rope_inv_freq(), 0)
    write(out_dir / "pool_L.bin", pool)
    write(out_dir / "consts_L.bin", consts)
    write(out_dir / "ptab.bin", ptab)
    write(out_dir / "xres_full.bin", xres_real.astype(np.float32))
    o = out_dir.as_posix()
    AD = L.AD_BYTES
    cfg = [
        "device", f"xclbin dx {DX_BUILD}/final.xclbin", f"kernelx dx dx {DX_BUILD}/insts.bin",
        f"attngeom {L.KV_ROW} {L.PTAB_ROW} 0",
        f"buf pool {L.POOL_BYTES} {o}/pool_L.bin",
        f"buf xres_full {T_REAL * hid * 4} {o}/xres_full.bin",
        f"buf xres_one {hid * 4}",
        f"buf consts {L.CD_BYTES} {o}/consts_L.bin",
        f"buf kv {L.KV_BYTES}",
        f"buf act {AD}",
        f"buf ptab {L.PTAB_BYTES} {o}/ptab.bin",
    ]
    for i in range(T_REAL):
        off = i * hid * 4
        cfg += [f"attnpos dx {i}", f"copy xres_one 0 xres_full {off} {hid * 4}",
                "run dx pool xres_one consts kv act ptab", f"copy xres_full {off} xres_one 0 {hid * 4}"]
    cfg.append(f"dump xres_full {o}/xres_out.bin {T_REAL * hid * 4}")
    (out_dir / "run_seq.cfg").write_text("\n".join(cfg) + "\n", newline="\n")
    r = subprocess.run([str(RUN_KERNEL), str(out_dir / "run_seq.cfg")], cwd=str(out_dir), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError("dx (GEVM) run failed")
    return np.fromfile(out_dir / "xres_out.bin", np.float32).reshape(T_REAL, hid).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--real", type=int, default=16, help="T_REAL real tokens")
    ap.add_argument("--layers", type=int, default=None, help="default: all of spec.num_layers")
    ap.add_argument("--start-layer", type=int, default=0, help="resume from this layer, loading checkpointed xres")
    ap.add_argument("--end-layer", type=int, default=None, help="stop BEFORE this layer (exclusive); default: --layers")
    ap.add_argument("--out", default=str(HERE / "out_layer40"))
    ap.add_argument("--final-only", action="store_true", help="skip the layer loop, just score the checkpointed final xres")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    md = Path(a.model_dir)
    spec = spec_from_model_dir(md)
    R = QR.recipe(spec)
    L, G = R.layout, R.geo
    hid = spec.hidden
    q = Q4NX(md / "model.q4nx")
    q.hidden = hid
    nlayers = a.layers or len(spec.layer_types)
    end_layer = a.end_layer if a.end_layer is not None else nlayers
    T_REAL = a.real
    ckpt_g = out / "xres_gemm_ckpt.npy"
    ckpt_s = out / "xres_gevm_ckpt.npy"
    ids_path = out / "ids.npy"

    VOCAB = 100352
    stride = max(1, (VOCAB - 2000) // T_REAL)

    if a.final_only:
        xres_gemm = np.load(ckpt_g)
        xres_gevm = np.load(ckpt_s)
        ids = np.load(ids_path).tolist()
    elif a.start_layer > 0:
        assert ckpt_g.is_file() and ckpt_s.is_file(), "no checkpoint to resume from -- run --start-layer 0 first"
        xres_gemm = np.load(ckpt_g)
        xres_gevm = np.load(ckpt_s)
        ids = np.load(ids_path).tolist()
        assert len(ids) == T_REAL, "T_REAL mismatch against checkpointed run"
        print(f"resumed from checkpoint at layer {a.start_layer}")
    else:
        ids = [1000 + stride * i for i in range(T_REAL)]
        assert len(set(ids)) == T_REAL
        xres_gemm = np.stack([q.embed(tok, hid) for tok in ids]).astype(np.float64)
        xres_gevm = xres_gemm.copy()
        np.save(ids_path, np.array(ids))

    if not a.final_only:
        print(f"running layers [{a.start_layer}, {end_layer}) of {nlayers} x {T_REAL} real tokens, "
              f"BOTH routes, layer-major ...")
        for l in range(a.start_layer, end_layer):
            print(f"  layer {l}: GEMM route ...")
            xres_gemm = gemm_layer(out / f"L{l}_gemm", q, spec, L, G, l, xres_gemm)
            print(f"  layer {l}: GEVM route (production dx, t=1) ...")
            xres_gevm = gevm_layer(out / f"L{l}_gevm", q, spec, L, l, xres_gevm)
            d = float(np.linalg.norm(xres_gemm - xres_gevm) / np.linalg.norm(xres_gevm))
            print(f"    hidden-state rel diff after layer {l}: {d:.6e}")
            np.save(ckpt_g, xres_gemm)
            np.save(ckpt_s, xres_gevm)

        if end_layer < nlayers:
            print(f"\ncheckpointed after layer {end_layer - 1}; resume with --start-layer {end_layer}")
            return 0

    np.save(out / "xres_gemm_final.npy", xres_gemm)
    np.save(out / "xres_gevm_final.npy", xres_gevm)

    print("\ncomputing real lm_head logits (SAME function, both routes -- isolates layer-stack divergence only) ...")
    agree_top1, margins = [], []
    for i, tok in enumerate(ids):
        hn_g, logits_g = RD.final_logits(q, spec, xres_gemm[i])
        hn_s, logits_s = RD.final_logits(q, spec, xres_gevm[i])
        top1_g = int(np.argmax(logits_g))
        top1_s = int(np.argmax(logits_s))
        top5_g = set(np.argsort(logits_g)[-5:].tolist())
        top5_s = set(np.argsort(logits_s)[-5:].tolist())
        srt_g = np.sort(logits_g)[::-1]
        margin_g = float(srt_g[0] - srt_g[1])
        agree_top1.append(top1_g == top1_s)
        margins.append(margin_g)
        ok = top1_g == top1_s
        print(f"  token {i} (id {tok}): GEMM top1={top1_g} GEVM top1={top1_s} "
              f"{'MATCH' if ok else 'DIVERGE'}  top5 overlap {len(top5_g & top5_s)}/5  "
              f"GEMM top1-top2 margin={margin_g:.4f}")
        if not ok:
            rank_of_gevm_top1_in_gemm = int(np.sum(logits_g > logits_g[top1_s]))
            print(f"    GEVM's top1 (id {top1_s}) ranks #{rank_of_gevm_top1_in_gemm + 1} in GEMM's logits, "
                  f"logit gap = {logits_g[top1_g] - logits_g[top1_s]:.4f}")

    n_match = sum(agree_top1)
    print(f"\ntop-1 agreement: {n_match}/{T_REAL} tokens ({100.0 * n_match / T_REAL:.1f}%)")
    print(f"mean GEMM top1-top2 logit margin: {np.mean(margins):.4f}")
    print("PASS (all tokens agree)" if n_match == T_REAL else "DIVERGENCE FOUND")
    return 0 if n_match == T_REAL else 1


if __name__ == "__main__":
    sys.exit(main())
