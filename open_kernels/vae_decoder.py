r"""vae_decoder: FLUX.2 [klein]'s VAE decoder on the NPU -- weight packing, buffer layout and
dispatch schedule, shared by the kernel exporter (export_dit_kernels.py builds the streams)
and whatever runs them (utilities/dit-chain/chain_test_vae.py, later the engine).
Design: .claude/plans/image-diffusion-phase5-vae.md.

Kernel sets (export_dit_kernels.py): conv/ (dit_conv, 3x3), conv1/ (dit_conv, 1x1), vew/
(vae_ew), and one stream each in the DiT's gemm and fa sets.

What runs, from the DiT's final latents LAT ([T, 128] packed, T = (R/16)^2):
  latent_in   conv1, 4 output phases: the pipeline's BN de-normalization, unpatchify and
              post_quant_conv folded into one 1x1 map per (dy, dx) -> Z0 (128 channels, 32
              real; zero-bordered)
  conv_in     conv 3x3 -> the mid block's input
  resnet      [gn_stats] gn_apply+SiLU, conv, gn_stats, gn_apply+SiLU, conv, [1x1
              shortcut], add (+ the next GroupNorm's sums); the add is in place
  attention   gn_apply -> QIN [T, 1024] (column 512 is a constant 1: the biases' row),
              one dit_gemm [q' x4 | k' x4 | v''] (N 2048), dit_fa (4 heads of 128), add.
              The one d = 512 head runs as 4 heads of 128 because its score form
              [x,1] Ma x^T (Ma = [Wq^T Wk; bq Wk]) is factored at rank 128 (plain SVD, s^1/2
              on each side; utilities/dit-ref/vae_attn_rank.py: LPIPS 0.0004); q' carries
              1/2 (dit_fa scales by 1/sqrt(128), the VAE by 1/sqrt(512)); Wo and both
              value/output biases fold into v'' (P's rows sum to 1)
  upsample    conv 3x3 on the source grid, 4 output phases (nearest 2x folded in)
  norm_out    gn_apply+SiLU, conv_out (3 of 128 channels), rgba -> RGBA8

Buffers are NHWC bf16, zero-bordered (dit_conv's layout); a conv whose source grid is
narrower than 128 (W = 32, 64 at 512 px) writes 128/W - 1 garbage parts past each output
row's right border, so its output's pitch is (128/W) (Wout + 2).

Bindings: every op's arguments are whole buffers or views -- a conv's packed weights
(a slice of one weights buffer) or a GroupNorm's parameter block (a slice of one blocks
buffer) -- so every stream addresses from offset 0 and same-shaped ops share a stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent

LATENT_CH = 32
BLOCK_OUT = (128, 256, 512, 512)
LAYERS = 2                      # resnets per up block = LAYERS + 1
EL = 4096                       # vae_ew element
BLOCK = 19                      # GroupNorm parameter block, elements (vae_ew.py)
QIN_W = 1024                    # attention GEMM K: 512 channels, the ones column, zeros
QKV_N = 2048                    # q' x4 | k' x4 | v'' | zeros
RANK = 128
ALIGN = 4096                    # bytes; weight slices start aligned (XRT sub-buffers)


# ---------------------------------------------------------------------------- packing

def _dit_conv_pack():
    import sys
    sys.path.insert(0, str(HERE / "designs" / "dit_conv"))
    sys.path.insert(0, str(HERE / "designs" / "dit_gemm"))
    import conv_pack
    from pack import pack_b
    return conv_pack, pack_b


def attention_weights(sd: dict, prefix: str, rank: int = RANK) -> np.ndarray:
    """The attention GEMM's B [QIN_W, QKV_N] (float64), see the module docstring."""
    g = lambda k: np.asarray(sd[prefix + k], np.float64)  # noqa: E731
    Wq, bq, Wk = g("to_q.weight"), g("to_q.bias"), g("to_k.weight")
    Wv, bv, Wo, bo = g("to_v.weight"), g("to_v.bias"), g("to_out.0.weight"), g("to_out.0.bias")
    Ma = np.concatenate([Wq.T @ Wk, (bq @ Wk)[None]], 0)          # [513, 512]
    P, s, Qt = np.linalg.svd(Ma, full_matrices=False)
    U = P[:, :rank] * np.sqrt(s[:rank]) * 0.5                       # q' = [x, 1] U
    V = Qt[:rank].T * np.sqrt(s[:rank])                             # k' = x V
    B = np.zeros((QIN_W, QKV_N))
    for h in range(4):
        B[:513, 128 * h:128 * (h + 1)] = U
        B[:512, 512 + 128 * h:512 + 128 * (h + 1)] = V
    B[:512, 1024:1536] = Wv.T @ Wo.T
    B[512, 1024:1536] = bv @ Wo.T + bo
    return B


def latent_in_weights(sd: dict):
    """Per-phase (dy, dx) 1x1 maps from the packed latent (128 = 32 x (dy, dx)) to
    post_quant_conv's output, with the pipeline's BN de-normalization folded in."""
    s = np.sqrt(np.asarray(sd["bn.running_var"], np.float64) + 1e-4)   # batch_norm_eps
    m = np.asarray(sd["bn.running_mean"], np.float64)
    Wp = np.asarray(sd["post_quant_conv.weight"], np.float64)[:, :, 0, 0]  # [32, 32]
    bp = np.asarray(sd["post_quant_conv.bias"], np.float64)
    ws = np.zeros((4, LATENT_CH, 4 * LATENT_CH, 1, 1))
    bs = np.zeros((4, LATENT_CH))
    for p in range(4):                          # p = 2 dy + dx = packed channel's sub-index
        k = 4 * np.arange(LATENT_CH) + p
        ws[p, :, k, 0, 0] = (Wp * s[k][None, :]).T
        bs[p] = bp + Wp @ m[k]
    return ws, bs


def pack_weights(sd: dict) -> tuple[np.ndarray, dict[str, tuple[int, int]], np.ndarray,
                                    dict[str, int]]:
    """diffusers AutoencoderKLFlux2 state dict (numpy) -> (weights bytes, {key: (offset,
    nbytes)}, GroupNorm blocks (bf16 [n * BLOCK * EL]), {gn key: block index})."""
    conv_pack, pack_b = _dit_conv_pack()
    parts, table, off = [], {}, 0

    def add(key, arr):
        nonlocal off
        arr = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        pad = -arr.size % ALIGN
        table[key] = (off, arr.size)
        parts.append(np.concatenate([arr, np.zeros(pad, np.uint8)]))
        off += arr.size + pad

    f = lambda k: np.asarray(sd[k], np.float32)  # noqa: E731
    ws, bs = latent_in_weights(sd)
    add("latent_in", conv_pack.pack_conv_phases(ws.astype(np.float32), bs.astype(np.float32)))
    add("decoder.conv_in", conv_pack.pack_conv(f("decoder.conv_in.weight"),
                                               f("decoder.conv_in.bias"), cin=128))
    gns = []
    for k in sorted(sd):
        if k.endswith(".weight") and k.startswith("decoder.") and sd[k].ndim == 4 \
                and k not in ("decoder.conv_in.weight",):
            base = k[:-len(".weight")]
            taps = sd[k].shape[2] * sd[k].shape[3]
            add(base, conv_pack.pack_conv(f(k), f(base + ".bias"), taps=taps,
                                          up=".upsamplers." in base))
        if k.startswith("decoder.") and k.endswith(".weight") and sd[k].ndim == 1 \
                and "norm" in k.split(".")[-2]:
            gns.append(k[:-len(".weight")])
    attn = "decoder.mid_block.attentions.0."
    add("attn_qkv", pack_b(attention_weights(sd, attn).astype(np.float32)))
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

@dataclass
class Buf:
    H: int
    W: int
    C: int                      # capacity: channels per pixel (px_stride)
    pitch: int
    border: int = 1
    extra_rows: int = 0

    @property
    def nelems(self) -> int:
        return (self.H + 2 * self.border + self.extra_rows) * self.pitch * self.C

    def view(self, **kw) -> dict:
        v = {"off": 0, "pitch": self.pitch, "border": self.border}
        return v | kw


@dataclass
class Plan:
    R: int
    buffers: dict[str, Buf] = field(default_factory=dict)
    ops: list[dict] = field(default_factory=list)
    streams: dict[str, dict[str, dict]] = field(default_factory=dict)   # set -> name -> spec
    _by_spec: dict = field(default_factory=dict)

    def stream(self, kset: str, spec: dict) -> str:
        key = (kset, json.dumps(spec, sort_keys=True))
        if key not in self._by_spec:
            name = f"vae{self.R}_{kset}{len(self.streams.setdefault(kset, {})):02d}"
            self.streams[kset][name] = spec
            self._by_spec[key] = name
        return self._by_spec[key]

    def op(self, kset: str, spec: dict, args: list, what: str) -> None:
        self.ops.append({"set": kset, "stream": self.stream(kset, spec), "args": args,
                         "what": what})


def _parts(W: int) -> int:
    return max(1, 128 // W)


def plan(R: int) -> Plan:
    """The whole decode at R x R: buffers, ops (in order), and the streams they use."""
    pl = Plan(R)
    B = pl.buffers
    L, h = R // 8, R // 16
    rev = list(reversed(BLOCK_OUT))                     # up block output channels
    res = [L * 2 ** i for i in range(4)]
    B["LAT"] = Buf(h, h, 4 * LATENT_CH, h, border=0, extra_rows=4)
    B["Z0"] = Buf(L, L, 128, _parts(h) * (L + 2))
    B["QIN"] = Buf(L, L, QIN_W, L, border=0)
    B["QKV"] = Buf(L, L, QKV_N, L, border=0)
    B["O"] = Buf(L, L, 512, L, border=0)
    B["OUT"] = Buf(R, R, 128, R + 2)
    B["RGBA"] = Buf(R * R // 1024, 1, EL, 1, border=0)
    B["DUMMY"] = Buf(1, 1, BLOCK * EL, 1, border=0)      # >= any unused argument's declared size
    # A zero-bordered buffer holds ONE channel count: producers write the interior only, so
    # a narrower layout's interior lands on a wider layout's border bytes (and the other
    # way round) -- the conv then reads the previous layout's data as its zero padding.
    # So the stage whose first resnet narrows (512 -> 256, 256 -> 128) normalizes its
    # input into GI<i>, and G/T/S hold the stage's output channels only.
    for i, r in enumerate(res):
        cin = rev[max(i - 1, 0)]                        # the stage's input channels
        src = L if i == 0 else res[i - 1]               # the producer's source grid
        B[f"X{i}"] = Buf(r, r, cin, _parts(src) * (r + 2), extra_rows=1)
        for n in ("G", "T", "S"):
            B[f"{n}{i}"] = Buf(r, r, rev[i], _parts(r) * (r + 2), extra_rows=1)
        if cin != rev[i]:
            B[f"GI{i}"] = Buf(r, r, cin, _parts(r) * (r + 2), extra_rows=1)

    def conv(taps, src, dst, H, W, cin, cout, wkey, up=False, what=""):
        spec = {"H": H, "W": W, "Cin": cin, "Cout": cout, "up": up,
                "x": B[src].view(), "y": B[dst].view()}
        pl.op("conv" if taps == 9 else "conv1", spec,
              [("buf", src), ("w", wkey), ("buf", dst)], what or wkey)

    def vew(op, C, H, W, a=None, b=None, y=None, gn=None, what="", **kw):
        spec = {"op": op, "C": C, "H": H, "W": W, **kw, "stats_off": 0}
        sizes = {"X": EL, "B": EL, "S": BLOCK * EL, "Y": EL}
        for role, (name, extra) in (("a", a or (None, {})), ("b", b or (None, {})),
                                    ("y", y or (None, {}))):
            if name is None:
                continue
            v = B[name].view(**extra)
            spec[role] = v
            ps = v.get("px_stride", C)
            sizes[{"a": "X", "b": "B", "y": "Y"}[role]] = \
                (H + 2 * v["border"]) * v["pitch"] * ps
        if op == "rgba":                                # [pixels / 1024][8 KB], not a view
            spec["y"] = {"off": 0}
            sizes["Y"] = B["RGBA"].nelems
        spec["sizes"] = sizes
        args = [("buf", a[0]) if a else ("buf", "DUMMY"),
                ("buf", b[0]) if b else ("buf", "DUMMY"),
                ("gn", gn) if gn else ("buf", "DUMMY"),
                ("buf", y[0]) if y else ("buf", "DUMMY")]
        pl.op("vew", spec, args, what or op)

    state = {"x": "X0", "fresh": None}

    def resnet(key, i, ci, co, next_gn):
        r, x = res[i], state["x"]
        G, T, S = f"G{i}", f"T{i}", f"S{i}"
        GI = G if ci == co else f"GI{i}"                # one channel count per buffer
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
        vew("add", co, r, r, a=(x, {}), b=(T, {}), y=(x, {}),
            gn=next_gn, stats=next_gn is not None, what=key + " residual")
        state["fresh"] = next_gn

    def attention(key, next_gn):
        r, x = res[0], state["x"]
        if state["fresh"] != key + ".group_norm":
            vew("gn_stats", 512, r, r, a=(x, {}), gn=key + ".group_norm")
        vew("gn_apply", 512, r, r, a=(x, {}), y=("QIN", {"px_stride": QIN_W}),
            gn=key + ".group_norm", what=key + ".group_norm")
        T = r * r
        pl.op("gemm", {"M": T, "K": QIN_W, "N": QKV_N},
              [("buf", "QIN"), ("w", "attn_qkv"), ("buf", "QKV")], key + " q'|k'|v''")
        pl.op("fa", {"L": T, "heads": 4, "kv_heads": 4, "causal": 0, "valid_len": 0,
                     "layout": {"qkv_ld": QKV_N, "k_col": 512, "v_col": 1024, "o_ld": 512}},
              [("buf", "QKV"), ("buf", "QKV"), ("buf", "QKV"), ("buf", "O")], key + " attention")
        vew("add", 512, r, r, a=(x, {}), b=("O", {}), y=(x, {}), gn=next_gn,
            stats=next_gn is not None, what=key + " residual")
        state["fresh"] = next_gn

    # the units in order, each with the GroupNorm it starts with (None: an upsample conv)
    units = [("resnet", "decoder.mid_block.resnets.0", 0, 512, 512),
             ("attn", "decoder.mid_block.attentions.0"),
             ("resnet", "decoder.mid_block.resnets.1", 0, 512, 512)]
    ci = 512
    for b, co in enumerate(rev):
        for j in range(LAYERS + 1):
            units.append(("resnet", f"decoder.up_blocks.{b}.resnets.{j}", b, ci, co))
            ci = co
        if b < 3:
            units.append(("up", f"decoder.up_blocks.{b}.upsamplers.0.conv", b, co))
    units.append(("out", "decoder.conv_norm_out"))

    def first_gn(u):
        return {"resnet": lambda: u[1] + ".norm1", "attn": lambda: u[1] + ".group_norm",
                "out": lambda: u[1], "up": lambda: None}[u[0]]()

    conv(1, "LAT", "Z0", h, h, 4 * LATENT_CH, 128, "latent_in", up=True,
         what="latent_in (BN, unpatchify, post_quant_conv)")
    conv(9, "Z0", "X0", L, L, 128, 512, "decoder.conv_in")
    for n, u in enumerate(units):
        nxt = first_gn(units[n + 1]) if n + 1 < len(units) else None
        if u[0] == "resnet":
            resnet(u[1], u[2], u[3], u[4], nxt)
        elif u[0] == "attn":
            attention(u[1], nxt)
        elif u[0] == "up":
            i, co = u[2], u[3]
            conv(9, state["x"], f"X{i + 1}", res[i], res[i], co, co, u[1], up=True)
            state.update(x=f"X{i + 1}", fresh=None)
        else:
            r = res[3]
            vew("gn_apply", 128, r, r, a=(state["x"], {}), y=("G3", {}), gn=u[1], silu=True,
                what=u[1])
            conv(9, "G3", "OUT", r, r, 128, 128, "decoder.conv_out")
            vew("rgba", 128, r, r, a=("OUT", {}), y=("RGBA", {}), what="rgba")
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
