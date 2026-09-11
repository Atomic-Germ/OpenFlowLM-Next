r"""Build everything needed to run N decode steps of a stock model on the open
kernels, and the fp64 reference to judge them by.

    python open_kernels/model/make_decode.py --layers 8 --tokens 1 [--model-dir DIR]
    open_kernels/harness/out/run_kernel.exe open_kernels/model/out/run_decode.cfg
    python open_kernels/model/compare_decode.py --tokens 1

Manifest-driven: the model's config.json -> ModelSpec -> the family recipe's
manifest (recipes/manifest.py) says which kernels exist (and which build dir
each comes from), how the per-layer buffers are sized and packed
(recipes/pack.py, the same plan src/open_qwen36/pools.cpp interprets), and
the verb sequence per layer type; the `.cfg` written here is that program
spelled out in the harness's directives. The fp64 reference is replica.py
(Qwen3.6-MoE) or replica_dense.py (Qwen3 dense).

Emitted per layer: the weight pool, the `consts` blob, and a zeroed state / KV
buffer. Emitted once: the lm_head pool, the RoPE position table, the reference
residuals + logits.

Decode starts at position 0 from zeroed state, which is exact prefill for
these architectures (each layer's state update sees one token at a time), and
each next token is the reference's greedy pick -- so the NPU and the reference
run the same token sequence and every position is comparable.

For the MoE family the routed experts are chosen ON DEVICE: the reference
re-uses the NPU's choice when a previous run's `y_rout` file disagrees with
its own (the 8th slot is often a near-tie), so a routing difference doesn't
masquerade as a numerical one -- it is printed when it happens.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from recipes import pack as PK  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from recipes.manifest import manifest  # noqa: E402
from recipes.spec import FULL  # noqa: E402
from q4nx import Q4NX, f32_to_bf16  # noqa: E402

DEFAULT_MODEL_DIR = os.environ.get(
    "FLM_MODEL_DIR", str(Path.home() / ".flm" / "models" / "Qwen3.6-35B-A3B-NPU2"))   # FLM's default model store


def sfx(t):
    return "" if t == 0 else f"_t{t}"


def write(path: Path, arr):
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def npu_routing(out: Path, layer, t):
    """The 8 experts the NPU's router picked in a previous run, if there is one."""
    p = out / f"y_rout{layer}{sfx(t)}.bin"
    if not p.is_file():
        return None
    return np.fromfile(p, np.float32)[256:264].view(np.int32).astype(np.int64)


def build_cfg(m: dict, nl: int, tokens: int, out: Path, pool_dir: Path, max_ctx: int) -> str:
    """The manifest's programs as harness directives."""
    d = DESIGNS.as_posix()
    o = out.as_posix()
    lay = m["layout"]
    types = m["layers"][:nl]
    used = {s["kernel"] for lt in set(types) for s in m["layer_types"][lt]["program"]} | {s["kernel"] for s in m["tail"]}
    cfg = ["device", f"attngeom {lay['kv_row']} {lay['ptab_row']}"]
    if "moe" in lay:
        mo = lay["moe"]
        cfg.append(f"moegeom {mo['experts']} {mo['topk']} {mo['stripe']} {mo['up_bytes']} {mo['down_core']} "
                   f"{mo['pool_down']} {mo['share_up']} {mo['share_gate']} {mo['share_down']}")
    ctx_done = set()
    for kn, kd in m["kernels"].items():
        if kn not in used:
            continue
        bdir = f"{d}/{m['builds'][kd['build']]['build_dir']}"
        if kd["context"] not in ctx_done:
            cfg.append(f"xclbin {kd['context']} {bdir}/final.xclbin")
            ctx_done.add(kd["context"])
        cfg.append(f"kernelx {kn} {kd['context']} {bdir}/insts.bin")
    hid = lay["hidden"]
    for name, size in m["globals"].items():
        if isinstance(size, dict):
            cfg.append(f"buf {name} {size['per_row'] * max_ctx} {o}/{name}.bin")   # a position table (written below)
        elif name == "lmpool":
            cfg.append(f"buf lmpool {size} {pool_dir.as_posix()}/pool_lmhead.bin")
        elif name in ("xres", "normw", "zero"):
            cfg.append(f"buf {name} {size} {o}/{name}0.bin" if name == "xres" else f"buf {name} {size} {o}/{name}.bin")
        else:
            cfg.append(f"buf {name} {size}")
    for l, lt in enumerate(types):
        b = m["layer_types"][lt]["buffers"]
        cfg.append(f"buf pool{l} {lay['pool_bytes']} {pool_dir.as_posix()}/pool_L{l}.bin")
        cfg.append(f"buf consts{l} {b['consts']} {o}/consts_{l}.bin")
        cfg.append(f"buf act{l} {b['act']}")
        st = b["state"]
        if st["kind"] == "kv":
            cfg.append(f"buf state{l} {st['row'] * max_ctx}")
        else:
            cfg.append(f"buf state{l} {st['bytes']} {o}/zstate_{lt}.bin")
    attnpos = [kn for kn, kd in m["kernels"].items() if kn in used and kd.get("patch") == "attnpos"]
    windows = {kn: m["kernels"][kn].get("window", 0) for kn in attnpos}
    runs = []
    for t in range(tokens):
        s = sfx(t)
        if t:
            runs.append(f"load xres {o}/xres{t}.bin")
        for kn in attnpos:
            runs.append(f"attngeom {lay['kv_row']} {lay['ptab_row']} {windows[kn]}")
            runs.append(f"attnpos {kn} {t}")
        for l, lt in enumerate(types):
            for step in m["layer_types"][lt]["program"]:
                if step["op"] == "run":
                    args = " ".join(a + (str(l) if a in ("pool", "consts", "act", "state") else "") for a in step["args"])
                    runs.append(f"run {step['kernel']} {args}")
                elif step["op"] == "moeroute2":
                    runs.append(f"moeroute2 {step['kernel']} act{l} {step['act_off'] + lay['rout_idx_off']}")
                    runs.append(f"dump act{l} {o}/y_rout{l}{s}.bin 4096 {step['act_off']}")
            runs.append(f"dump xres {o}/y_res{l}{s}.bin {hid * 4}")
            runs.append(f"dump act{l} {o}/y_act{l}{s}.bin {m['layer_types'][lt]['buffers']['act']}")
        for step in m["tail"]:
            runs.append(f"run {step['kernel']} " + " ".join(step["args"]))
        runs.append(f"dump logits {o}/y_logits{s}.bin {m['globals']['logits']}")
    return "\n".join(cfg + runs) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, default=8, help="run the first N layers (each holds its pool on the device)")
    ap.add_argument("--tokens", type=int, default=1, help="decode N tokens, positions 0..N-1")
    ap.add_argument("--token", type=int, default=None, help="the first token id (default: the model's bos, else 248045)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--pool-dir", default=None, help="where the big pools live (default <out>/pools)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--reuse-pools", action="store_true", help="keep pool files that already exist")
    ap.add_argument("--cfg-only", action="store_true", help="only rewrite the .cfg")
    ap.add_argument("--requant", action="store_true",
                    help="force the whole run -- spec, packing plan, pools and reference -- onto the "
                         "re-quantising fallback: every q8 projection becomes q4_1 on the way into the "
                         "pool, as it did before OPEN-QUANT-Q8. Run it beside the default (q8 projections "
                         "streamed at q8) for the A/B that says what the q8 GEMV buys. The KERNELS must "
                         "have been exported the same way (OPEN_KERNELS_FORCE_Q4_1=1)")
    ap.add_argument("--strict-routing", action="store_true",
                    help="MoE: keep the reference's own top-8 even where a previous run shows the NPU picked differently")
    a = ap.parse_args()
    if a.requant:
        os.environ["OPEN_KERNELS_FORCE_Q4_1"] = "1"      # read before the spec is derived

    out = Path(a.out).resolve()
    pool_dir = Path(a.pool_dir).resolve() if a.pool_dir else out / "pools"
    out.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    md = Path(a.model_dir)
    spec = spec_from_model_dir(md)
    m = manifest(spec, a.max_ctx)
    lay = m["layout"]
    plan = {"pool_bytes": m["pack"]["pool_bytes"], "chunk_bytes": m["pack"]["chunk_bytes"],
            "layer_types": {lt: d["pack"] for lt, d in m["layer_types"].items()},
            "lm_head": m["pack"]["lm_head"], "embed": m["pack"]["embed"], "norm": m["pack"]["norm"]}
    nl = min(a.layers, spec.num_layers)
    types = list(spec.layer_types[:nl])
    q = Q4NX(md / "model.q4nx")
    # The reference reads what the NPU holds, tensor by tensor: a projection the plan
    # streams at q8 (`q8_perm`) as the container's own q8, every other q8 tensor through
    # the packer's q4_1 (OPEN-QUANT-Q8). So the comparison always measures the kernels.
    q8_pat = [re.compile(re.escape(op["tensor"]).replace(r"\{l\}", r"\d+"))
              for d in plan["layer_types"].values() for op in d["pool"] + d["consts"]
              if op.get("op") == "q8_perm"]
    q.native_q8 = lambda n: any(p.fullmatch(n) for p in q8_pat)
    if q8_pat:
        print(f"  {len(q8_pat)} projection(s) streamed at q8: {sorted(spec.q8_roles)}")
    q.hidden = spec.hidden
    # granite: <|start_of_role|>, the token every turn of its template opens
    # with -- NOT config.json's bos_token_id, which is 100283 = '</documents>'
    # and disagrees with tokenizer_config.json's own bos (<|end_of_text|>).
    tok0 = a.token if a.token is not None else {"qwen36moe": 248045, "qwen35": 248045, "qwen3": 151644,
                                                "llama3": 128000, "gemma3": 2, "hunyuan": 127958,
                                                "granite": 100264, "phi3": 200021}[spec.family]
    print(f"{md.name} ({spec.family}): {spec.num_layers} layers -> running {nl}: {types}")

    if not a.cfg_only:
        write(out / "zero.bin", np.zeros(spec.hidden, np.float32))
        for lt, d in m["layer_types"].items():
            st = d["buffers"]["state"]
            if st["kind"] == "linear":
                write(out / f"zstate_{lt}.bin", np.zeros(st["bytes"], np.uint8))
        for name, g in m["globals"].items():
            if isinstance(g, dict):
                write(out / f"{name}.bin", PK.ptab(a.max_ctx, spec.rotary_dim, spec.rope_theta, g["per_row"],
                                                   g.get("inv_freq", spec.rope_inv_freq()),
                                                   g.get("window", 0), g.get("scale", 1.0),
                                                   g.get("long_inv_freq"), g.get("switch_row")))
        write(out / "normw.bin", f32_to_bf16(q.bf16(plan["norm"]["tensor"])))
        for l, lt in enumerate(types):
            pf = pool_dir / f"pool_L{l}.bin"
            if not (a.reuse_pools and pf.is_file() and pf.stat().st_size == lay["pool_bytes"]):
                PK.build_layer_pool(plan, lt, q, l).tofile(pf)
            write(out / f"consts_{l}.bin", PK.build_consts(plan, lt, q, l, m["layer_types"][lt]["buffers"]["consts"]))
            print(f"  layer {l} {lt}: pool + consts", flush=True)
        lmf = pool_dir / "pool_lmhead.bin"
        if not (a.reuse_pools and lmf.is_file() and lmf.stat().st_size == plan["lm_head"]["pool_bytes"]):
            PK.build_lmhead_pool(plan, q).tofile(lmf)
            print("  lm_head pool", flush=True)

        # ---- the reference: the same token sequence, in fp64, on CPU
        if spec.family == "qwen36moe":
            import replica as R
        elif spec.family == "qwen35":
            import replica_qwen35 as R35
        else:
            import replica_dense as RD
        if spec.family in ("qwen36moe", "qwen35"):
            conv = {l: np.zeros((spec.conv_kernel - 1, spec.lin_qkv_dim)) for l in range(nl)}
            S = {l: np.zeros((spec.lin_value_heads, spec.lin_value_dim, spec.lin_value_dim)) for l in range(nl)}
        K = {l: np.zeros((0, spec.num_kv_heads, spec.head_dim)) for l in range(nl)}
        V = {l: np.zeros((0, spec.num_kv_heads, spec.head_dim)) for l in range(nl)}
        tok = tok0
        for t in range(a.tokens):
            x = q.embed(tok, spec.hidden)
            write(out / f"xres{t}.bin", x.astype(np.float32))
            xr = x.copy()
            for l, lt in enumerate(types):
                if spec.family == "qwen36moe":
                    if lt == FULL:
                        xa, K[l], V[l] = R.attn_decode(q, l, xr.copy(), K[l], V[l], t)
                    else:
                        xa, conv[l], S[l] = R.linear_decode(q, l, xr.copy(), conv[l], S[l])
                    _, _, mine = R.route(q, l, xa)
                    got = None if a.strict_routing else npu_routing(out, l, t)
                    if got is not None and sorted(got.tolist()) != sorted(mine.tolist()):
                        print(f"    token {t} layer {l}: NPU routed {sorted(got.tolist())}, "
                              f"reference {sorted(mine.tolist())} -- using the NPU's")
                        mine = got
                    xr = R.moe_decode(q, l, xa, top=mine)
                    print(f"  token {t} layer {l} {lt} top8={mine.tolist()}", flush=True)
                elif spec.family == "qwen35":
                    xr, conv[l], S[l], K[l], V[l] = R35.layer_decode(q, spec, l, xr.copy(), conv[l], S[l],
                                                                    K[l], V[l], t)
                    print(f"  token {t} layer {l} {lt}: |res| {np.abs(xr).max():.3f}", flush=True)
                else:
                    xr, K[l], V[l] = RD.dense_decode(q, spec, l, xr.copy(), K[l], V[l], t, max_ctx=a.max_ctx)
                    print(f"  token {t} layer {l} {lt}: |res| {np.abs(xr).max():.3f}", flush=True)
                write(out / f"ref_res{l}{sfx(t)}.bin", xr.astype(np.float32))
            if spec.family == "qwen36moe":
                _, logits = R.final_logits(q, xr)
            elif spec.family == "qwen35":
                _, logits = R35.final_logits(q, spec, xr)
            else:
                _, logits = RD.final_logits(q, spec, xr)
            write(out / f"ref_logits{sfx(t)}.bin", logits.astype(np.float32))
            tok = int(logits[:spec.real_vocab].argmax())
            print(f"  token {t} (position {t}): reference argmax {tok}", flush=True)

    cfg = build_cfg(m, nl, a.tokens, out, pool_dir, a.max_ctx)
    (out / "run_decode.cfg").write_text(cfg, newline="\n")
    nruns = len([r for r in cfg.splitlines() if r.startswith("run ")])
    print(f"wrote {out / 'run_decode.cfg'}: {nl} layers, {a.tokens} token(s), {nruns} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
