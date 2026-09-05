r"""Build everything needed to run N decode steps of a stock Qwen3.6-MoE model
on the open kernels, and the fp64 reference to judge them by.

    python open_kernels/model/make_decode.py --layers 8 --tokens 1
    open_kernels/harness/out/run_kernel.exe open_kernels/model/out/run_decode.cfg
    python open_kernels/model/compare_decode.py --tokens 1

Emitted per layer: the 512 MB weight pool (pools.py), a `consts` blob holding
the layer's small weights in the byte layout designs/layer_x/layout.py defines,
and a zeroed state / KV buffer. Emitted once: the lm_head pool, the RoPE
position table, and the reference residuals + logits from replica.py.

Decode starts at position 0 from zeroed state, which is exact prefill for this
architecture (each layer's state update sees one token at a time), and each next
token is the reference's greedy pick -- so the NPU and the reference run the same
token sequence and every position is comparable.

The routed experts are chosen ON DEVICE: the layer kernel's first dispatch ends
with the router, the driver's `moeroute2` points the expert fills at what it
picked, and the second dispatch runs the MoE block. The reference re-uses the
NPU's choice when a previous run's `y_rout` file disagrees with its own (the 8th
slot is often a near-tie), so a routing difference doesn't masquerade as a
numerical one -- it is printed when it happens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(DESIGNS / "layer_x"))
sys.path.insert(0, str(HERE))

import layout as XL  # noqa: E402  (designs/layer_x/layout.py)
import pools as PP  # noqa: E402
import replica as R  # noqa: E402
from q4nx import Q4NX, f32_to_bf16  # noqa: E402

DEFAULT_MODEL_DIR = os.environ.get(
    "FLM_MODEL_DIR", str(Path.home() / ".flm" / "models" / "Qwen3.6-35B-A3B-NPU2"))   # FLM's default model store


def sfx(t):
    return "" if t == 0 else f"_t{t}"


def write(path: Path, arr):
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def layer_consts(m, layer, full_attn, out: Path):
    """The layer's small weights, packed into the `consts` BO layer_x reads."""
    pack = np.frombuffer(PP.build_pack(m, layer), np.uint8)
    side = np.frombuffer(PP.build_side(m, layer, full_attn), np.uint8)
    lnw, postln = pack[0:4096], pack[4096:8192]
    sgw, rw = pack[8192:12288], pack[12288:12288 + 1048576]
    if full_attn:
        c = np.zeros(XL.CA_BYTES, np.uint8)
        c[XL.CA_LNW:XL.CA_LNW + 4096] = lnw
        c[XL.CA_POSTLN:XL.CA_POSTLN + 4096] = postln
        c[XL.CA_META:XL.CA_META + 512] = side[128:640]           # q_norm (effective)
        c[XL.CA_META + 512:XL.CA_META + 1024] = side[640:1152]   # k_norm
        c[XL.CA_RW:XL.CA_RW + 1048576] = rw
        c[XL.CA_SGW:XL.CA_SGW + 4096] = sgw
        write(out / f"constsa_{layer}.bin", c)
    else:
        # The glue's side blob: alpha/beta projections, the [a | dt_bias] record,
        # and conv1d transposed to [8 groups][4 taps][1024]. Its first 4 KB is
        # the xn slot the kernel fills from `act`, so it starts at offset 4096.
        sb = np.zeros(4096 + XL.GLUE_SIDE_BYTES, np.uint8)
        sb[4096:4096 + 131072] = side[66048:66048 + 131072]
        sb[135168:135168 + 131072] = side[197120:197120 + 131072]
        small = np.zeros(1024, np.float32)
        small[:32] = side[65792:65792 + 128].view(np.float32)     # ssm_a
        small[32:64] = side[65920:65920 + 128].view(np.float32)   # ssm_dt.bias
        sb[266240:266240 + 4096] = small.view(np.uint8)
        convw = side[0:65536].view(np.uint16).reshape(4, 8192)
        sb[270336:270336 + 65536] = convw.reshape(4, 8, 1024).transpose(1, 0, 2).reshape(-1).view(np.uint8)
        nw = np.zeros(2048, np.uint16)
        nw[:128] = side[65536:65536 + 256].view(np.uint16)        # ssm_norm, one 4 KB element
        c = np.zeros(XL.C_BYTES, np.uint8)
        c[XL.C_LNW:XL.C_LNW + 4096] = lnw
        c[XL.C_SIDE:XL.C_SIDE + XL.GLUE_SIDE_BYTES] = sb[4096:]
        c[XL.C_NW:XL.C_NW + 4096] = nw.view(np.uint8)
        c[XL.C_POSTLN:XL.C_POSTLN + 4096] = postln
        c[XL.C_RW:XL.C_RW + 1048576] = rw
        c[XL.C_SGW:XL.C_SGW + 4096] = sgw
        c[XL.C_WOUT:XL.C_WOUT + 5242880] = side[328192:328192 + 5242880]   # out_proj, pool order
        write(out / f"consts_{layer}.bin", c)


def npu_routing(out: Path, layer, t):
    """The 8 experts the NPU's router picked in a previous run, if there is one."""
    p = out / f"y_rout{layer}{sfx(t)}.bin"
    if not p.is_file():
        return None
    return np.fromfile(p, np.float32)[256:264].view(np.int32).astype(np.int64)


def build_cfg(args, nl, full, out: Path, pool_dir: Path) -> str:
    d = DESIGNS.as_posix()
    o = out.as_posix()
    cfg = ["device"]
    # Only declare the design a slice actually uses: a short slice can be all
    # linear layers, and registering an xclbin costs a hw_context either way.
    if any(not full[l] for l in range(nl)):
        cfg += [f"xclbin X {d}/layer_x/build_lx0/final.xclbin",
                f"kernelx lx0 X {d}/layer_x/build_lx0/insts.bin",
                f"kernelx lx1 X {d}/layer_x/build_lx1/insts.bin"]
    if any(full[l] for l in range(nl)):
        cfg += [f"xclbin Y {d}/layer_x/build_ax0/final.xclbin",
                f"kernelx ax0 Y {d}/layer_x/build_ax0/insts.bin",
                f"kernelx ax1 Y {d}/layer_x/build_ax1/insts.bin"]
    cfg += [f"xclbin L {d}/ln/build/final.xclbin",
            f"kernelx ln L {d}/ln/build/insts.bin",
            f"xclbin K {d}/lm_head_q8/build_full/final.xclbin",
            f"kernelx lm K {d}/lm_head_q8/build_full/insts.bin",
            f"buf xres 8192 {o}/xres0.bin",
            f"buf zero 8192 {o}/zero.bin",
            f"buf normw 4096 {o}/normw.bin",
            f"buf lmpool {PP.LMHEAD_POOL_BYTES} {pool_dir.as_posix()}/pool_lmhead.bin",
            "buf xresf 8192", "buf hn 4096", "buf logits 993280",
            f"buf ptab {XL.PTAB_BYTES} {o}/ptab.bin"]
    for l in range(nl):
        cfg.append(f"buf pool{l} {XL.POOL_BYTES} {pool_dir.as_posix()}/pool_L{l}.bin")
        if full[l]:
            cfg += [f"buf constsa{l} {XL.CA_BYTES} {o}/constsa_{l}.bin",
                    f"buf acta{l} {XL.AA_BYTES}", f"buf kv{l} {XL.KV_BYTES}"]
        else:
            cfg += [f"buf consts{l} {XL.C_BYTES} {o}/consts_{l}.bin",
                    f"buf state{l} {XL.STATE_BYTES} {o}/zstate.bin",
                    f"buf act{l} {XL.A_BYTES}"]
    runs = []
    for t in range(args.tokens):
        s = sfx(t)
        if t:
            runs.append(f"load xres {o}/xres{t}.bin")
        if any(full[l] for l in range(nl)):
            runs.append(f"attnpos ax0 {t}")   # the attention layers share one stream
        for l in range(nl):
            if full[l]:
                runs += [f"run ax0 pool{l} xres constsa{l} kv{l} acta{l} ptab",
                         f"moeroute2 ax1 acta{l} {XL.AA_ROUT + 1024}",
                         f"run ax1 pool{l} xres constsa{l} kv{l} acta{l} ptab",
                         f"dump acta{l} {o}/y_rout{l}{s}.bin 4096 {XL.AA_ROUT}"]
            else:
                runs += [f"run lx0 pool{l} xres consts{l} state{l} act{l}",
                         f"moeroute2 lx1 act{l} {XL.A_ROUT + 1024}",
                         f"run lx1 pool{l} xres consts{l} state{l} act{l}",
                         f"dump act{l} {o}/y_rout{l}{s}.bin 4096 {XL.A_ROUT}"]
            runs.append(f"dump xres {o}/y_res{l}{s}.bin 8192")
        runs += ["run ln xres zero normw xresf hn",
                 "run lm lmpool hn logits",
                 f"dump logits {o}/y_logits{s}.bin 993280"]
    return "\n".join(cfg + runs) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, default=8,
                    help="run the first N layers (each holds a 512 MB pool on the device)")
    ap.add_argument("--tokens", type=int, default=1, help="decode N tokens, positions 0..N-1")
    ap.add_argument("--token", type=int, default=248045, help="the first token id")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--pool-dir", default=None, help="where the big pools live (default <out>/pools)")
    ap.add_argument("--reuse-pools", action="store_true", help="keep pool files that already exist")
    ap.add_argument("--cfg-only", action="store_true", help="only rewrite the .cfg")
    ap.add_argument("--strict-routing", action="store_true",
                    help="keep the reference's own top-8 even where a previous run shows the NPU "
                         "picked differently (by default the NPU's choice is adopted, so an 8th-slot "
                         "near-tie doesn't read as a numerical error)")
    a = ap.parse_args()

    # Absolute: the .cfg resolves relative paths against its own directory.
    out = Path(a.out).resolve()
    pool_dir = Path(a.pool_dir).resolve() if a.pool_dir else out / "pools"
    out.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    md = Path(a.model_dir)
    cfgj = json.load(open(md / "config.json"))
    interval = cfgj["full_attention_interval"]
    nl = min(a.layers, cfgj["num_hidden_layers"])
    full = {l: ((l + 1) % interval == 0) for l in range(nl)}
    m = Q4NX(md / "model.q4nx")
    print(f"{md.name}: {cfgj['num_hidden_layers']} layers, interval {interval} -> running {nl}, "
          f"attention at {[l for l in range(nl) if full[l]]}")

    if not a.cfg_only:
        write(out / "zero.bin", np.zeros(2048, np.float32))
        write(out / "zstate.bin", np.zeros(XL.STATE_BYTES, np.uint8))
        write(out / "ptab.bin", XL.ptab())
        write(out / "normw.bin", f32_to_bf16(m.bf16("model.norm.weight")))

        for l in range(nl):
            pf = pool_dir / f"pool_L{l}.bin"
            if not (a.reuse_pools and pf.is_file() and pf.stat().st_size == XL.POOL_BYTES):
                PP.build_layer_pool(m, l, full[l]).tofile(pf)
            layer_consts(m, l, full[l], out)
            print(f"  layer {l} {'FULL' if full[l] else 'lin '}: pool + consts", flush=True)
        lmf = pool_dir / "pool_lmhead.bin"
        if not (a.reuse_pools and lmf.is_file() and lmf.stat().st_size == PP.LMHEAD_POOL_BYTES):
            PP.build_lmhead_pool(m).tofile(lmf)
            print("  lm_head pool", flush=True)

        # ---- the reference: the same token sequence, in fp64, on CPU
        conv = {l: np.zeros((3, 8192)) for l in range(nl)}
        S = {l: np.zeros((32, 128, 128)) for l in range(nl)}
        K = {l: np.zeros((0, 2, 256)) for l in range(nl)}
        V = {l: np.zeros((0, 2, 256)) for l in range(nl)}
        tok = a.token
        for t in range(a.tokens):
            x = m.embed(tok)
            write(out / f"xres{t}.bin", x.astype(np.float32))
            xr = x.copy()
            for l in range(nl):
                if full[l]:
                    xa, K[l], V[l] = R.attn_decode(m, l, xr.copy(), K[l], V[l], t)
                else:
                    xa, conv[l], S[l] = R.linear_decode(m, l, xr.copy(), conv[l], S[l])
                _, _, mine = R.route(m, l, xa)
                got = None if a.strict_routing else npu_routing(out, l, t)
                if got is not None and sorted(got.tolist()) != sorted(mine.tolist()):
                    print(f"    token {t} layer {l}: NPU routed {sorted(got.tolist())}, "
                          f"reference {sorted(mine.tolist())} -- using the NPU's")
                    mine = got
                xr = R.moe_decode(m, l, xa, top=mine)
                write(out / f"ref_res{l}{sfx(t)}.bin", xr.astype(np.float32))
                print(f"  token {t} layer {l} {'FULL' if full[l] else 'lin '} top8={mine.tolist()}", flush=True)
            _, logits = R.final_logits(m, xr)
            write(out / f"ref_logits{sfx(t)}.bin", logits.astype(np.float32))
            tok = int(logits.argmax())
            print(f"  token {t} (position {t}): reference argmax {tok}", flush=True)

    cfg = build_cfg(a, nl, full, out, pool_dir)
    (out / "run_decode.cfg").write_text(cfg, newline="\n")
    nruns = len([r for r in cfg.splitlines() if r.startswith("run ")])
    print(f"wrote {out / 'run_decode.cfg'}: {nl} layers, {a.tokens} token(s), {nruns} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
