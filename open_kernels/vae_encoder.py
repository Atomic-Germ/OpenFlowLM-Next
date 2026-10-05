r"""vae_encoder: FLUX.2 [klein]'s VAE encoder on the NPU -- an edit's reference image to the
DiT's reference tokens. Weight packing, buffer layout and dispatch schedule, shared by the
kernel exporter and whatever runs them (utilities/dit-chain/chain_test_vae_enc.py, the
pipeline). It uses the decoder's kernels only (vae_decoder.py): conv/ (dit_conv 3x3),
conv1/ (1x1), vew/ (vae_ew), and one stream each in the DiT's gemm and fa sets.
Plan: specs/open-diffusion/plans/edits.md (8.2).

What runs, from IN (the reference, written by the host as bf16 2 (x / 255) - 1 into channels
0-2 of a zero-bordered R x R x 64 buffer):
  conv_in     conv 3x3, Cin 3 padded to 64 -> 128
  resnet      as the decoder's: [gn_stats] gn_apply+SiLU, conv, gn_stats, gn_apply+SiLU,
              conv, [1x1 shortcut], add (+ the next GroupNorm's sums); the add is in place
  downsample  diffusers' conv3x3(pad(x, right 1, bottom 1), stride 2) as space-to-depth: the
              residual add before it is 4 dispatches, phase (p, q) reading rows 2i + p and
              columns 2j + q (px_stride 2C, border 0) and writing channels (2p + q)C.. of
              D, a zero-bordered (H/2) x (W/2) grid of 4C channels; then a 3x3 conv on D
              whose weights (s2d_weights) are zero on the -1 taps -- the right/bottom pad
              lands on D's zero border. 4x the real FLOPs on three convs (spike 8.1.1,
              utilities/dit-chain/spike_s2d.py).
  attention   the decoder's: gn_apply -> QIN, the rank-128 factored score as one dit_gemm
              [q' x4 | k' x4 | v''] and dit_fa as 4 heads of 128, add
  latent_out  norm_out (gn_apply+SiLU) written as space-to-depth like a downsample's add
              (4 phase dispatches) into D3, a (R/16)^2 grid of 2048 channels; then ONE 3x3
              conv on D3 computes the packed reference tokens: conv_out, quant_conv's 32 mean
              channels, the pipeline's 2x2 patchify (output channel 4c + 2dy + dx) and its
              BatchNorm normalisation folded into the weights (latent_out_weights). Its
              output is zero-bordered (dit_conv writes no other kind), so a vae_ew add of a
              zero buffer copies it into REF, the plain [(R/16)^2, 128] tokens the DiT's
              x_emb reads.

Buffers are NHWC bf16, zero-bordered, one channel count each (vae_decoder.py's rules).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import vae_decoder as vd

HERE = Path(__file__).resolve().parent

IN_C = 64                       # conv_in's input: RGB padded to dit_conv's 64-channel chunk
BLOCK_OUT = vd.BLOCK_OUT        # (128, 256, 512, 512)
LAYERS = 2                      # resnets per down block
LATENT_CH = vd.LATENT_CH
EL, BLOCK = vd.EL, vd.BLOCK
ALIGN = vd.ALIGN


# ---------------------------------------------------------------------------- packing

def s2d_weights(w: np.ndarray) -> np.ndarray:
    """A stride-2 3x3 conv's weights [Cout, C, 3, 3] -> the 3x3 conv on its input's
    space-to-depth grid [Cout, 4C, 3, 3]: tap (1 + dy, 1 + dx), channel (2p + q)C + c holds
    W[2dy + p, 2dx + q] when that tap exists; the -1 taps are zero."""
    co, c = w.shape[:2]
    out = np.zeros((co, 4 * c, 3, 3), w.dtype)
    for p in range(2):
        for q in range(2):
            for dy in range(2):
                for dx in range(2):
                    ky, kx = 2 * dy + p, 2 * dx + q
                    if ky <= 2 and kx <= 2:
                        out[:, (2 * p + q) * c:(2 * p + q + 1) * c, 1 + dy, 1 + dx] = w[:, :, ky, kx]
    return out


def latent_out_weights(sd: dict) -> tuple[np.ndarray, np.ndarray]:
    """conv_out, quant_conv's mean half, the 2x2 patchify and the BatchNorm normalisation as
    one 3x3 conv on norm_out's space-to-depth grid: [128, 4 * 512, 3, 3] and [128]
    (float64). Output channel o = 4c + 2dy + dx is the normalised mean channel c at pixel
    (2i + dy, 2j + dx) of the R/8 grid, which reads that grid's rows 2i + dy - 1 + ky:
    space-to-depth row i + a, phase p with 2a + p = dy - 1 + ky (a in -1..1)."""
    g = lambda k: np.asarray(sd[k], np.float64)  # noqa: E731
    Wc, bc = g("encoder.conv_out.weight"), g("encoder.conv_out.bias")      # [64, 512, 3, 3]
    Wq, bq = g("quant_conv.weight")[:LATENT_CH, :, 0, 0], g("quant_conv.bias")[:LATENT_CH]
    Wm = np.einsum("ck,kiyx->ciyx", Wq, Wc)                                # [32, 512, 3, 3]
    bm = Wq @ bc + bq
    s = np.sqrt(g("bn.running_var") + 1e-4)                                # batch_norm_eps
    m = g("bn.running_mean")
    C = Wc.shape[1]
    w = np.zeros((4 * LATENT_CH, 4 * C, 3, 3))
    b = np.zeros(4 * LATENT_CH)
    for c in range(LATENT_CH):
        for dy in range(2):
            for dx in range(2):
                o = 4 * c + 2 * dy + dx
                b[o] = (bm[c] - m[o]) / s[o]
                for ky in range(3):
                    for kx in range(3):
                        a, p = divmod(dy - 1 + ky, 2)
                        bb, q = divmod(dx - 1 + kx, 2)
                        w[o, (2 * p + q) * C:(2 * p + q + 1) * C, 1 + a, 1 + bb] += Wm[c, :, ky, kx] / s[o]
    return w, b


def pack_weights(sd: dict) -> tuple[np.ndarray, dict[str, tuple[int, int]], np.ndarray,
                                    dict[str, int]]:
    """diffusers AutoencoderKLFlux2 state dict (numpy) -> the encoder's (weights bytes,
    {key: (offset, nbytes)}, GroupNorm blocks (bf16 [n * BLOCK * EL]), {gn key: block
    index}), the layout vae_decoder.pack_weights gives the decoder."""
    conv_pack, pack_b = vd._dit_conv_pack()
    parts, table, off = [], {}, 0

    def add(key, arr):
        nonlocal off
        arr = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        pad = -arr.size % ALIGN
        table[key] = (off, arr.size)
        parts.append(np.concatenate([arr, np.zeros(pad, np.uint8)]))
        off += arr.size + pad

    f = lambda k: np.asarray(sd[k], np.float32)  # noqa: E731
    add("encoder.conv_in", conv_pack.pack_conv(f("encoder.conv_in.weight"),
                                               f("encoder.conv_in.bias"), cin=IN_C))
    gns = []
    for k in sorted(sd):
        if not k.startswith("encoder.") or not k.endswith(".weight"):
            continue
        base = k[:-len(".weight")]
        if sd[k].ndim == 4 and base not in ("encoder.conv_in", "encoder.conv_out"):
            w, taps = f(k), sd[k].shape[2] * sd[k].shape[3]
            if ".downsamplers." in base:
                w = s2d_weights(w)
            add(base, conv_pack.pack_conv(w, f(base + ".bias"), taps=taps))
        if sd[k].ndim == 1 and "norm" in k.split(".")[-2]:
            gns.append(base)
    w, b = latent_out_weights(sd)
    add("latent_out", conv_pack.pack_conv(w.astype(np.float32), b.astype(np.float32)))
    add("attn_qkv", pack_b(vd.attention_weights(sd, "encoder.mid_block.attentions.0.")
                           .astype(np.float32)))
    blocks = np.zeros(len(gns) * BLOCK * EL, bfloat16)
    index = {}
    for i, key in enumerate(gns):
        gamma, beta = f(key + ".weight"), f(key + ".bias")
        e = (i * BLOCK + 1) * EL
        blocks[e:e + gamma.size] = gamma.astype(bfloat16)
        blocks[e + 512:e + 512 + beta.size] = beta.astype(bfloat16)
        index[key] = i
    return np.concatenate(parts), table, blocks, index


# ---------------------------------------------------------------------------- schedule

Buf = vd.Buf
_parts = vd._parts


class Plan(vd.Plan):
    def stream(self, kset: str, spec: dict) -> str:
        import json
        key = (kset, json.dumps(spec, sort_keys=True))
        if key not in self._by_spec:
            name = f"venc{self.R}_{kset}{len(self.streams.setdefault(kset, {})):02d}"
            self.streams[kset][name] = spec
            self._by_spec[key] = name
        return self._by_spec[key]


def plan(R: int) -> Plan:
    """The whole encode at R x R: buffers, ops (in order), and the streams they use."""
    pl = Plan(R)
    B = pl.buffers
    res = [R >> i for i in range(4)]                       # each down block's grid
    L, h = res[3], R // 16
    B["IN"] = Buf(R, R, IN_C, R + 2)
    for i, r in enumerate(res):
        cin = BLOCK_OUT[max(i - 1, 0)]                     # the block's input channels
        B[f"X{i}"] = Buf(r, r, cin, _parts(r) * (r + 2), extra_rows=1)
        for n in ("G", "T", "S"):
            B[f"{n}{i}"] = Buf(r, r, BLOCK_OUT[i], _parts(r) * (r + 2), extra_rows=1)
        if cin != BLOCK_OUT[i]:
            B[f"GI{i}"] = Buf(r, r, cin, _parts(r) * (r + 2), extra_rows=1)
    for i in range(4):                                     # the downsample / latent_out inputs
        r = res[i] // 2 if i < 3 else h
        c = BLOCK_OUT[i]
        B[f"D{i}"] = Buf(r, r, 4 * c, _parts(r) * (r + 2), extra_rows=max(1, _parts(r) - 1))
    B["QIN"] = Buf(L, L, vd.QIN_W, L, border=0)
    B["QKV"] = Buf(L, L, vd.QKV_N, L, border=0)
    B["O"] = Buf(L, L, 512, L, border=0)
    B["OUTL"] = Buf(h, h, 4 * LATENT_CH, _parts(h) * (h + 2), extra_rows=1)
    B["ZERO"] = Buf(h, h, 4 * LATENT_CH, h, border=0)
    B["REF"] = Buf(h, h, 4 * LATENT_CH, h, border=0)
    B["DUMMY"] = Buf(1, 1, BLOCK * EL, 1, border=0)

    def conv(taps, src, dst, H, W, cin, cout, wkey, what=""):
        spec = {"H": H, "W": W, "Cin": cin, "Cout": cout, "up": False,
                "x": B[src].view(), "y": B[dst].view()}
        pl.op("conv" if taps == 9 else "conv1", spec,
              [("buf", src), ("w", wkey), ("buf", dst)], what or wkey)

    def vew(op, C, H, W, a=None, b=None, y=None, gn=None, what="", **kw):
        spec = {"op": op, "C": C, "H": H, "W": W, **kw, "stats_off": 0}
        sizes = {"X": EL, "B": EL, "S": BLOCK * EL, "Y": EL}
        for role, ref in (("a", a), ("b", b), ("y", y)):
            if ref is None:
                continue
            name, extra = ref
            spec[role] = B[name].view(**extra)
            sizes[{"a": "X", "b": "B", "y": "Y"}[role]] = B[name].nelems
        spec["sizes"] = sizes
        args = [("buf", a[0]) if a else ("buf", "DUMMY"),
                ("buf", b[0]) if b else ("buf", "DUMMY"),
                ("gn", gn) if gn else ("buf", "DUMMY"),
                ("buf", y[0]) if y else ("buf", "DUMMY")]
        pl.op("vew", spec, args, what or op)

    def phases(src: str, C: int):
        """The 4 space-to-depth phases: ((p, q), the parity view of src, D's phase slot)."""
        pitch = B[src].pitch
        for p in range(2):
            for q in range(2):
                yield ((p, q), {"off": ((p + 1) * pitch + q + 1) * C, "border": 0,
                                "px_stride": 2 * C},
                       {"off": (2 * p + q) * C, "px_stride": 4 * C})

    state = {"x": "X0", "fresh": None}

    def resnet(key, i, ci, co, next_gn, s2d=None):
        r, x = res[i], state["x"]
        G, T, S = f"G{i}", f"T{i}", f"S{i}"
        GI = G if ci == co else f"GI{i}"
        if state["fresh"] != key + ".norm1":
            vew("gn_stats", ci, r, r, a=(x, {}), gn=key + ".norm1", what=key + ".norm1 stats")
        vew("gn_apply", ci, r, r, a=(x, {}), y=(GI, {}), gn=key + ".norm1", silu=True,
            what=key + ".norm1")
        conv(9, GI, T, r, r, ci, co, key + ".conv1")
        vew("gn_stats", co, r, r, a=(T, {}), gn=key + ".norm2", what=key + ".norm2 stats")
        vew("gn_apply", co, r, r, a=(T, {}), y=(G, {}), gn=key + ".norm2", silu=True,
            what=key + ".norm2")
        conv(9, G, T, r, r, co, co, key + ".conv2")
        if ci != co:
            conv(1, x, S, r, r, ci, co, key + ".conv_shortcut")
            x = state["x"] = S
        if s2d:                                     # the downsample's input, space-to-depth
            for (p, q), src, dst in phases(x, co):
                vew("add", co, r // 2, r // 2, a=(x, src), b=(T, src), y=(s2d, dst),
                    what=f"{key} residual -> {s2d} phase {p}{q}")
            state["fresh"] = None
            return
        vew("add", co, r, r, a=(x, {}), b=(T, {}), y=(x, {}),
            gn=next_gn, stats=next_gn is not None, what=key + " residual")
        state["fresh"] = next_gn

    def attention(key, next_gn):
        r, x = res[3], state["x"]
        if state["fresh"] != key + ".group_norm":
            vew("gn_stats", 512, r, r, a=(x, {}), gn=key + ".group_norm")
        vew("gn_apply", 512, r, r, a=(x, {}), y=("QIN", {"px_stride": vd.QIN_W}),
            gn=key + ".group_norm", what=key + ".group_norm")
        T = r * r
        pl.op("gemm", {"M": T, "K": vd.QIN_W, "N": vd.QKV_N},
              [("buf", "QIN"), ("w", "attn_qkv"), ("buf", "QKV")], key + " q'|k'|v''")
        pl.op("fa", {"L": T, "heads": 4, "kv_heads": 4, "causal": 0, "valid_len": 0,
                     "layout": {"qkv_ld": vd.QKV_N, "k_col": 512, "v_col": 1024, "o_ld": 512}},
              [("buf", "QKV"), ("buf", "QKV"), ("buf", "QKV"), ("buf", "O")], key + " attention")
        vew("add", 512, r, r, a=(x, {}), b=("O", {}), y=(x, {}), gn=next_gn,
            stats=next_gn is not None, what=key + " residual")
        state["fresh"] = next_gn

    # the units in order, each with the GroupNorm it starts with
    units = []
    ci = BLOCK_OUT[0]
    for b, co in enumerate(BLOCK_OUT):
        for j in range(LAYERS):
            last = j == LAYERS - 1 and b < 3
            units.append(("resnet", f"encoder.down_blocks.{b}.resnets.{j}", b, ci, co,
                          f"D{b}" if last else None))
            ci = co
        if b < 3:
            units.append(("down", f"encoder.down_blocks.{b}.downsamplers.0.conv", b, co))
    units += [("resnet", "encoder.mid_block.resnets.0", 3, 512, 512, None),
              ("attn", "encoder.mid_block.attentions.0"),
              ("resnet", "encoder.mid_block.resnets.1", 3, 512, 512, None),
              ("out", "encoder.conv_norm_out")]

    def first_gn(u):
        return {"resnet": lambda: u[1] + ".norm1", "attn": lambda: u[1] + ".group_norm",
                "out": lambda: u[1], "down": lambda: None}[u[0]]()

    conv(9, "IN", "X0", R, R, IN_C, BLOCK_OUT[0], "encoder.conv_in")
    for n, u in enumerate(units):
        nxt = first_gn(units[n + 1]) if n + 1 < len(units) else None
        if u[0] == "resnet":
            resnet(u[1], u[2], u[3], u[4], nxt, s2d=u[5])
        elif u[0] == "attn":
            attention(u[1], nxt)
        elif u[0] == "down":
            i, co = u[2], u[3]
            r = res[i + 1]
            conv(9, f"D{i}", f"X{i + 1}", r, r, 4 * co, co, u[1])
            state.update(x=f"X{i + 1}", fresh=None)
        else:
            x = state["x"]
            if state["fresh"] != u[1]:
                vew("gn_stats", 512, L, L, a=(x, {}), gn=u[1], what=u[1] + " stats")
            for (p, q), src, dst in phases(x, 512):
                vew("gn_apply", 512, h, h, a=(x, src), y=("D3", dst), gn=u[1], silu=True,
                    npix=L * L, what=f"{u[1]} -> D3 phase {p}{q}")
            conv(9, "D3", "OUTL", h, h, 4 * 512, 4 * LATENT_CH, "latent_out",
                 what="latent_out (conv_out, quant_conv, patchify, BN)")
            vew("add", 4 * LATENT_CH, h, h, a=("OUTL", {}), b=("ZERO", {}), y=("REF", {}),
                what="reference tokens -> REF")
    return pl


def stream_specs(resolutions: list[int]) -> dict[str, dict[str, dict]]:
    """{kernel set: {stream: spec}} over the resolutions (the exporter's view)."""
    out: dict[str, dict[str, dict]] = {}
    for R in resolutions:
        for kset, streams in plan(R).streams.items():
            out.setdefault(kset, {}).update(streams)
    return out


if __name__ == "__main__":
    import sys
    for R in [int(r) for r in (sys.argv[1:] or ["512"])]:
        pl = plan(R)
        n = {k: len(v) for k, v in pl.streams.items()}
        mb = sum(b.nelems for b in pl.buffers.values()) * 2 / 2 ** 20
        print(f"{R}: {len(pl.ops)} ops, streams {n}, buffers {mb:.0f} MiB")
        sw = sum(1 for a, b in zip(pl.ops, pl.ops[1:]) if a["set"] != b["set"])
        print(f"   kernel-set switches: {sw}")
