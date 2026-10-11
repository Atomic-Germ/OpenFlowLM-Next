r"""fa_emul: a CPU model of dit_fa's attention arithmetic, and a check of it against hardware.

The model (`dit_fa_attention`) follows open_kernels/designs/dit_fa/fa_dit.cc step by step,
per 32-row query tile and 64-key chunk (c = log2(e)/sqrt(d)):

    S  = bf16( bf16(Qb[:, :64] Kb[:, :64]^T) + Qb[:, 64:] Kb[:, 64:]^T )   Qb, Kb bfp16 along d
    if any row of the tile has rowmax S > bf16(m + bf16(TAU/c)):          (lazy rescale)
        m' = max(m, rowmax S);  r = bf16(exp2((m - m')c));  O = bf16(O r);  l = l r;  m = m'
    P  = exp2_hw(S c - m c)  [* exp_fix]      in fp32 up to the exp, bf16 out
    l += rowsum(Pb)  (fp32);   O = bf16(O + Pb Vb)                          Pb, Vb bfp16 along keys
    out = bf16(O * bf16(1 / bf16(l)))

`stages` switches pieces off to attribute the error. q must hold whole 32-row tiles in
order (the rescale decision is per tile), so pass full sequences, not row samples.

exp2 of the scores is the aie2p hardware instruction (aie::exp2<bfloat16>), modelled
bit-exactly by `hw_exp2` from a dump of it (utilities/aie-probes/exp2_probe.py): it
returns 2^n * (1 + f) with f the input's fraction truncated to 7 bits -- a linear
mantissa, +3.8% mean / +6.1% worst at half-integers, zero at integers. Without "exp_hw"
the exp is exactly rounded to bf16; "exp_fix" models fa_dit.cc's FA_EXP_FIX correction.
`--check` compares the model with a real run.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\fa_emul.py --check <make_test dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

LKP, TQ = 64, 32
TAU = 32.0                     # fa_dit.cc FA_TAU (log2 units)
LOWEST = -3.3895313892515355e38
EXP_FIX = (1.4406570, -0.671875, 0.2275390625)   # fa_dit.cc FA_EXP_C0..C2
FULL = frozenset({"qk_bfp", "s_bf16", "pv_bfp", "o_bf16", "fin_bf16", "exp_hw", "lazy"})
FIXED = FULL | {"exp_fix"}


def hw_exp2(x: torch.Tensor) -> torch.Tensor:
    """aie2p's bf16 exp2, bit-exact on [-40, 10] against the probe dump. The fp32 input
    becomes fixed point with 7 fraction bits: floored to 1/128 while the bf16 ulp is finer;
    past that (|x| >= 2) floored to the bf16 grid with the missing low bits read as ones.
    The result is 2^floor(q) * (1 + frac(q)), exact in bf16."""
    x = x.double().clamp(-300.0, 127.9)
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0 ** -126)))
    ulp = torch.exp2(e - 7)
    q = torch.where(ulp > 2.0 ** -7, torch.floor(x / ulp) * ulp + ulp - 2.0 ** -7,
                    torch.floor(x * 128) / 128)
    n = torch.floor(q)
    return torch.ldexp(1.0 + (q - n), n.to(torch.int32)).float()


def exp_fix(y: torch.Tensor) -> torch.Tensor:
    """fa_dit.cc exp_fix: y * h(mantissa(y)), h a quadratic, bf16 arithmetic."""
    c0, c1, c2 = EXP_FIX
    mant, _ = torch.frexp(y)                       # y = mant * 2^e, mant in [0.5, 1)
    m = torch.where(y > 0, mant * 2, torch.ones_like(y))
    h = bf(c0 + c1 * m + c2 * bf(m * m))
    return bf(y * h)


def bfp16(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """bfp16ebs8 along `dim`: blocks of 8 share the largest exponent, int8 mantissas,
    round-half-even (the kernels run conv_even). Same rule as klein_quant_study.bfp16."""
    x = x.movedim(dim, -1)
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // 8, 8)
    _, e = torch.frexp(blocks)
    emax = e.amax(dim=-1, keepdim=True)
    scale = torch.ldexp(torch.ones_like(blocks[..., :1]), emax - 7)
    q = torch.round(blocks / scale).clamp_(-128, 127)
    return (q * scale).reshape(shape).movedim(-1, dim)


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def dit_fa_attention(q, k, v, stages=FULL, mask=None):
    """q [..., Lq, d], k/v [..., Lk, d] (float32; heads in the leading dims), Lq % 32 == 0.
    mask: optional bool [Lq, Lk], True = masked. Returns float32 [..., Lq, d]."""
    d = q.shape[-1]
    c = math.log2(math.e) / math.sqrt(d)
    ident = lambda t: t  # noqa: E731
    qb = bfp16(q) if "qk_bfp" in stages else q
    kb = bfp16(k) if "qk_bfp" in stages else k
    vb = bfp16(v, dim=-2) if "pv_bfp" in stages else v
    pq = (lambda t: bfp16(t)) if "pv_bfp" in stages else ident
    ob = bf if "o_bf16" in stages else ident
    lead = q.shape[:-2]
    Lq, Lk = q.shape[-2], k.shape[-2]
    T = Lq // TQ
    m = torch.full((*lead, T, TQ, 1), LOWEST)
    l = torch.zeros((*lead, T, TQ, 1))
    o = torch.zeros((*lead, T, TQ, v.shape[-1]))
    half = d // 2
    thr_step = float(bf(torch.tensor(TAU / c)))
    for c0 in range(0, Lk, LKP):
        kc, vc = kb[..., c0:c0 + LKP, :], vb[..., c0:c0 + LKP, :]
        if "s_bf16" in stages:
            s = bf(bf(qb[..., :half] @ kc[..., :half].transpose(-1, -2))
                   + qb[..., half:] @ kc[..., half:].transpose(-1, -2))
        else:
            s = qb @ kc.transpose(-1, -2)
        if mask is not None:
            s = s.masked_fill(mask[:, c0:c0 + LKP], LOWEST)
        s = s.reshape(*lead, T, TQ, s.shape[-1])
        nm = s.amax(-1, keepdim=True)
        if "lazy" in stages:
            need = (nm > bf(m + thr_step)).any(dim=-2, keepdim=True)
        else:
            need = torch.ones_like(nm[..., :1, :], dtype=torch.bool)
        m_new = torch.where(need, torch.maximum(m, nm), m)
        r = torch.exp2(((m - m_new) * c).clamp_min(-200.0))
        r = bf(r) if "o_bf16" in stages else r
        o = torch.where(need, ob(o * r), o)
        l = torch.where(need, l * r, l)
        arg = s * c - m_new * c
        if "exp_hw" in stages:
            p = hw_exp2(arg)
            if "exp_fix" in stages:
                p = exp_fix(p)
        else:
            p = torch.exp2(arg.clamp_min(-300.0))
            p = bf(p) if "o_bf16" in stages else p
        pb = pq(p)
        l = l + pb.sum(-1, keepdim=True)
        o = ob(o + pb @ vc.unsqueeze(-3))
        m = m_new
    out = bf(o * bf(1.0 / bf(l))) if "fin_bf16" in stages else o / l
    return out.reshape(*lead, Lq, -1)


def _load_bf16(a):
    return torch.from_numpy((np.asarray(a).astype(np.uint16).astype(np.uint32) << 16).view(np.float32))


def check(testdir: Path, heads: int, rows: int):
    r = np.load(testdir / "ref.npz")
    L, H, KVH, D = int(r["L"]), int(r["heads"]), int(r["kv_heads"]), 128
    q, k, v = (_load_bf16(r[x]) for x in "qkv")
    hw = _load_bf16(np.fromfile(testdir / "o.bin", dtype=np.uint16)).reshape(L, H * D)
    ix = torch.from_numpy(np.unique(np.linspace(0, L - 1, rows).astype(int)))
    cols = torch.arange(L)[None, :]
    mask = (cols >= int(r["valid_len"])).expand(L, L)
    if bool(r["causal"]):
        mask = mask | (cols > torch.arange(L)[:, None])
    use_mask = bool(r["causal"]) or int(r["valid_len"]) < L
    sel = np.unique(np.linspace(0, H - 1, heads).astype(int))
    exact = {}
    for h in sel:
        kh = h // (H // KVH)
        qh = q[ix, h * D:(h + 1) * D].double()
        s = (qh @ k[:, kh * D:(kh + 1) * D].double().T / math.sqrt(D))
        s = s.masked_fill(mask[ix], float("-inf"))
        exact[h] = torch.softmax(s, -1) @ v[:, kh * D:(kh + 1) * D].double()
    variants = {"model": FULL, "model+exp_fix": FIXED}
    variants |= {f"model-{x}": FULL - {x} for x in sorted(FULL)}
    variants["fp32"] = frozenset()
    print(f"rel_fro over heads {list(sel)}, {len(ix)} rows")
    print(f"{'variant':18s} {'vs fp64':>10s} {'vs hw':>10s}")
    n = sum(float((exact[h] ** 2).sum()) for h in sel)
    e_hw = sum(float(((hw[ix, h * D:(h + 1) * D].double() - exact[h]) ** 2).sum()) for h in sel)
    print(f"{'hardware':18s} {math.sqrt(e_hw / n):10.3e} {0:10.3e}")
    for name, st in variants.items():
        e_ref = e_hw = 0.0
        for h in sel:
            kh = h // (H // KVH)
            out = dit_fa_attention(q[:, h * D:(h + 1) * D], k[:, kh * D:(kh + 1) * D],
                                   v[:, kh * D:(kh + 1) * D], st,
                                   mask if use_mask else None)[ix].double()
            e_ref += float(((out - exact[h]) ** 2).sum())
            e_hw += float(((out - hw[ix, h * D:(h + 1) * D].double()) ** 2).sum())
        print(f"{name:18s} {math.sqrt(e_ref / n):10.3e} {math.sqrt(e_hw / n):10.3e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="a dit_fa make_test.py dir with o.bin")
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--rows", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(8)
    check(Path(a.check), a.heads, a.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
