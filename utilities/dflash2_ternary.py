"""Fake-quantized ternary DFlash 2 drafters (bf16 GGUF copies with chosen tensors replaced in place); rules in bonsai-2-dflash2.md D1."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dflash2_ref import GGML_BF16, Gguf, hadamard  # noqa: E402

G = 128
HB = 1024


def ternary_names(g: Gguf):
    """Projections, FFNs and conv_proj of every layer, plus fc; norms, conv_base and the selector stay bf16/f32."""
    kinds = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
             "attn_conv_proj", "ffn_conv_proj")
    out = ["fc.weight"]
    n = int(g.kv["dflash.block_count"])
    for i in range(n):
        out += [f"blk.{i}.{k}.weight" for k in kinds]
    for nm in out:
        assert g.tensors[nm][1] == GGML_BF16, nm
    return out


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def wht(w: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    o, i = w.shape
    return (w.reshape(o, i // HB, HB) @ H).reshape(o, i)


def ternarize(w: torch.Tensor, rule: str):
    """w [out, in] f32 -> (q int8 [out, in] in {-1, 0, 1}, s f32 [out, in // 128] bf16-exact)."""
    o, i = w.shape
    x = w.reshape(o, i // G, G)
    a = x.abs()
    if rule == "absmean":
        s = bf16_round(a.mean(-1))
        q = torch.clamp(torch.round(x / s.clamp_min(1e-30)[..., None]), -1, 1)
    elif rule == "l2":
        srt, idx = torch.sort(a, dim=-1, descending=True)
        cs = torch.cumsum(srt, -1)
        m = torch.arange(1, G + 1, dtype=torch.float32)
        best = torch.argmax(cs * cs / m, dim=-1)                       # support size - 1
        s = bf16_round(torch.gather(cs, -1, best[..., None])[..., 0] / (best + 1).float())
        rank = torch.empty_like(idx)
        rank.scatter_(-1, idx, torch.arange(G).expand_as(idx))
        q = torch.sign(x) * (rank <= best[..., None])
    else:
        raise ValueError(rule)
    return q.to(torch.int8).reshape(o, i), s


def fake_quant(w: torch.Tensor, variant: str, H: torch.Tensor):
    """-> (w_fq f32 before bf16 rounding, q, s, the rotated w the codes approximate)."""
    if variant == "absmean":
        q, s = ternarize(w, "absmean")
        wq = (q.reshape(*s.shape, G).float() * s[..., None]).reshape(w.shape)
        return wq, q, s, w
    if variant in ("wht_l2", "wht_absmean"):
        wr = wht(w, H)
        q, s = ternarize(wr, "l2" if variant == "wht_l2" else "absmean")
        wq_r = (q.reshape(*s.shape, G).float() * s[..., None]).reshape(w.shape)
        return wht(wq_r, H), q, s, wr
    if variant == "l2":
        q, s = ternarize(w, "l2")
        wq = (q.reshape(*s.shape, G).float() * s[..., None]).reshape(w.shape)
        return wq, q, s, w
    raise ValueError(variant)


def tensor_view(mm, g: Gguf, name):
    dims, t, off = g.tensors[name]
    n = int(np.prod(dims))
    shape = tuple(int(d) for d in reversed(dims))
    return mm[g.data0 + off:g.data0 + off + 2 * n].view(np.uint16).reshape(shape)


def to_bf16_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def from_bf16_bits(b: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((b.astype(np.uint32) << 16).view(np.float32))


def make(a):
    torch.set_num_threads(a.threads)
    g = Gguf(a.src)
    H = hadamard(HB)
    shutil.copyfile(a.src, a.dst)
    mm = np.memmap(a.dst, dtype=np.uint8, mode="r+")
    stats, codes = {}, {}
    tot = dict(num=0.0, den=0.0, numr=0.0, n=0, zeros=0)
    for nm in ternary_names(g):
        v = tensor_view(mm, g, nm)
        w = from_bf16_bits(np.array(v))
        wq, q, s, wr = fake_quant(w, a.variant, H)
        bits = to_bf16_bits(wq)
        v[...] = bits
        wb = from_bf16_bits(bits)
        err = float(((w - wq) ** 2).sum())
        errb = float(((w - wb) ** 2).sum())
        den = float((w ** 2).sum())
        zr = float((q == 0).float().mean())
        stats[nm] = dict(shape=list(w.shape), rel_err=(err / den) ** 0.5, rel_err_bf16_written=(errb / den) ** 0.5,
                         bf16_rounding_vs_fq=(float(((wb - wq) ** 2).sum()) / max(err, 1e-30)) ** 0.5,
                         zero_frac=zr, s_min=float(s.min()), s_max=float(s.max()),
                         s_fp16_subnormal_frac=float((s < 6.104e-5).float().mean()))
        tot["num"] += err; tot["den"] += den; tot["numr"] += errb; tot["n"] += w.numel(); tot["zeros"] += zr * w.numel()
        if a.codes:
            codes[nm + ".q"] = (q + 1).to(torch.uint8).numpy()
            codes[nm + ".s"] = s.numpy().astype(np.float32)
        print(f"{nm:28s} {str(list(w.shape)):16s} rel_err {stats[nm]['rel_err']:.4f} zeros {zr:.3f} "
              f"s [{stats[nm]['s_min']:.2e}, {stats[nm]['s_max']:.2e}]", flush=True)
    mm.flush()
    del mm
    summary = dict(variant=a.variant, params=tot["n"], rel_err=(tot["num"] / tot["den"]) ** 0.5,
                   rel_err_bf16_written=(tot["numr"] / tot["den"]) ** 0.5, zero_frac=tot["zeros"] / tot["n"])
    print(json.dumps(summary))
    Path(a.dst + ".stats.json").write_text(json.dumps(dict(summary=summary, tensors=stats), indent=1))
    if a.codes:
        np.savez(a.codes, **codes)


def check(a):
    """Round trip of a written GGUF: codes and scales recovered, untouched tensors byte-equal."""
    torch.set_num_threads(a.threads)
    gs, gd = Gguf(a.src), Gguf(a.dst)
    H = hadamard(HB)
    names = set(ternary_names(gs))
    for nm in gs.tensors:
        if nm in names:
            continue
        assert np.array_equal(np.asarray(gs.raw(nm)[0]), np.asarray(gd.raw(nm)[0])), nm
    assert gs.kv.keys() == gd.kv.keys()
    worst = 0.0
    for nm in sorted(names):
        w = from_bf16_bits(np.array(tensor_view(gs.mm, gs, nm)))
        wd = from_bf16_bits(np.array(tensor_view(gd.mm, gd, nm)))
        _, q, s, _ = fake_quant(w, a.variant, H)
        if a.variant == "absmean":
            want = (q.reshape(*s.shape, G).float() * s[..., None]).reshape(w.shape)
            assert torch.equal(wd, want), nm
            print(f"{nm:28s} exact s*q", flush=True)
            continue
        r = wht(wd, H).reshape(*s.shape, G)
        q2 = torch.round(r / s[..., None].clamp_min(1e-30)).clamp(-1, 1)
        agree = float((q2 == q.reshape(*s.shape, G).float()).float().mean())
        resid = r - q2 * s[..., None]
        rel = float(resid.norm() / (q2 * s[..., None]).norm())
        worst = max(worst, 1 - agree)
        print(f"{nm:28s} codes recovered {agree:.6f}  |W_gguf Hn - s q| / |s q| = {rel:.2e}", flush=True)
    print(f"untouched tensors byte-equal; worst code disagreement {worst:.2e}")


def drafter_bytes(a):
    """Bytes streamed per drafter pass (weights read once for the whole block) by storage."""
    g = Gguf(a.src)
    tern = set(ternary_names(g))
    n_t = sum(int(np.prod(g.tensors[n][0])) for n in tern)
    other = {n: int(np.prod(g.tensors[n][0])) for n in g.tensors if n not in tern}
    gathered = {n for n in other if n.startswith(("selector_predecessor", "selector_successor"))}
    streamed_other = sum(v for n, v in other.items() if n not in gathered)
    head = 248320 * 5120
    rows = {}
    for st, bpw in (("bf16", 16.0), ("q4_1", 5.0), ("t2", 34 * 8 / 128)):
        wbytes = n_t * bpw / 8
        rows[st] = dict(ternarizable_params=n_t, weight_bytes=wbytes, other_bytes=streamed_other * 2,
                        head_t2_bytes=head * 34 / 128, total=wbytes + streamed_other * 2 + head * 34 / 128)
    print(json.dumps(dict(gathered_codebooks=sorted(gathered), other_streamed_params=streamed_other, rows=rows), indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--src", required=True)
    m.add_argument("--dst", required=True)
    m.add_argument("--variant", required=True, choices=["absmean", "wht_l2", "wht_absmean", "l2"])
    m.add_argument("--codes")
    m.add_argument("--threads", type=int, default=6)
    c = sp.add_parser("check")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    c.add_argument("--variant", required=True, choices=["absmean", "wht_l2", "wht_absmean", "l2"])
    c.add_argument("--threads", type=int, default=6)
    b = sp.add_parser("bytes")
    b.add_argument("--src", required=True)
    a = ap.parse_args()
    {"make": make, "check": check, "bytes": drafter_bytes}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
