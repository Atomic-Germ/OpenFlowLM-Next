"""DFlash 2 drafter fine-tune under the t2 + OPEN-HADAMARD constraint (straight-through ternary, rotated latents)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

G = 128
HB = 1024
BLOCK = 8
TOP_K = 16
TERNARY_KINDS = ("wq", "wk", "wv", "wo", "gate", "up", "down", "attn_cp", "ffn_cp")
GGUF_NAME = {"wq": "attn_q.weight", "wk": "attn_k.weight", "wv": "attn_v.weight", "wo": "attn_output.weight",
             "gate": "ffn_gate.weight", "up": "ffn_up.weight", "down": "ffn_down.weight",
             "attn_cp": "attn_conv_proj.weight", "ffn_cp": "ffn_conv_proj.weight",
             "attn_norm": "attn_norm.weight", "ffn_norm": "ffn_norm.weight", "q_norm": "attn_q_norm.weight",
             "k_norm": "attn_k_norm.weight", "attn_cb": "attn_conv_base", "ffn_cb": "ffn_conv_base"}


def hadamard(n=HB, dtype=torch.float32):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(dtype)


def bf16_round(x):
    return x.to(torch.bfloat16).float()


@torch.no_grad()
def ternary_l2(w):
    """Per-128 L2-optimal ternary of w [out, in]: (q in {-1,0,1} as w's dtype, s [out, in/128] bf16-exact)."""
    o, i = w.shape
    x = w.reshape(o, i // G, G)
    a = x.abs()
    srt = torch.sort(a, dim=-1, descending=True).values
    cs = torch.cumsum(srt, -1)
    m = torch.arange(1, G + 1, device=w.device, dtype=w.dtype)
    best = torch.argmax(cs * cs / m, dim=-1, keepdim=True)
    s = bf16_round(torch.gather(cs, -1, best) / (best + 1).to(w.dtype))
    thr = torch.gather(srt, -1, best)
    q = torch.sign(x) * (a >= thr)
    return q.reshape(o, i), s[..., 0]


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wr, wq):
        return wq

    @staticmethod
    def backward(ctx, g):
        return g.float(), None


class TernaryLinear(torch.nn.Module):
    """y = Q(Wr) (Hn x): the latent Wr lives in the rotated basis; Q is re-derived once per optimizer step."""

    def __init__(self, w_rot, quant=True, dtype=torch.float32):
        super().__init__()
        self.wr = torch.nn.Parameter(w_rot.to(dtype).contiguous())
        self.quant = quant
        self._cache = None

    def invalidate(self):
        self._cache = None

    def forward(self, x_rot):
        if not self.quant:
            return x_rot @ self.wr.to(x_rot.dtype).T
        if self._cache is None or self._cache.dtype != x_rot.dtype:
            q, s = ternary_l2(self.wr.detach().float())
            # s is bf16-exact and q in {-1,0,1}, so s*q is exact in bf16
            self._cache = (q.reshape(*s.shape, G) * s[..., None]).reshape(self.wr.shape).to(x_rot.dtype)
        return x_rot @ _STE.apply(self.wr, self._cache).T


class Drafter(torch.nn.Module):
    def __init__(self, sd, cfg, quant=True, train_fc=False, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.eps = cfg["eps"]
        self.register_buffer("H", hadamard(HB).to(device), persistent=False)
        rot = lambda w: (w.to(device).float().reshape(w.shape[0], -1, HB) @ self.H).reshape(w.shape)
        self.layers = torch.nn.ModuleList()
        for ly in sd["layers"]:
            m = torch.nn.Module()
            for k in TERNARY_KINDS:
                setattr(m, k, TernaryLinear(rot(ly[k].float()), quant, dtype))
            for k in ("attn_norm", "ffn_norm", "q_norm", "k_norm", "attn_cb", "ffn_cb"):
                setattr(m, k, torch.nn.Parameter(ly[k].float().clone()))
            self.layers.append(m)
        self.out_norm = torch.nn.Parameter(sd["out_norm"].float().clone())
        self.sel_hidden = torch.nn.Parameter(sd["sel_hidden"].float().clone())
        self.train_fc = train_fc
        if train_fc:
            self.fc = TernaryLinear(rot(sd["fc"].float()), quant, dtype)
            self.enc_norm = torch.nn.Parameter(sd["enc_norm"].float().clone())
        self.register_buffer("sel_prev", sd["sel_prev"].to(torch.bfloat16), persistent=False)
        self.register_buffer("sel_next", sd["sel_next"].to(torch.bfloat16), persistent=False)

    def ternary_modules(self):
        mods = [(f"blk.{i}.{GGUF_NAME[k]}", getattr(ly, k)) for i, ly in enumerate(self.layers) for k in TERNARY_KINDS]
        if self.train_fc:
            mods.append(("fc.weight", self.fc))
        return mods

    def invalidate(self):
        for _, m in self.ternary_modules():
            m.invalidate()

    def wht(self, x):
        sh = x.shape
        return (x.reshape(-1, HB) @ self.H.to(x.dtype)).reshape(sh)

    def rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(x.dtype)

    def rope(self, x, pos):
        hd = x.shape[-1]
        half = hd // 2
        inv = self.cfg["rope_base"] ** (-torch.arange(0, half, device=x.device, dtype=torch.float64) * 2.0 / hd)
        ang = pos.double()[:, None] * inv[None]
        c, s = torch.cos(ang).to(x.dtype)[:, None], torch.sin(ang).to(x.dtype)[:, None]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    def conv(self, x, dyn, base, side):
        A, B, n = x.shape
        c = dyn.view(A, B, 2, 2, n // self.cfg["group"])[:, :, side]
        w = c.repeat_interleave(self.cfg["group"], dim=-1) + base[side].to(x.dtype)
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
        return w[:, :, 0] * x + w[:, :, 1] * prev

    def features(self, feat):
        if not self.train_fc:
            return feat
        return self.rms(self.fc(self.wht(feat)), self.enc_norm)

    def forward(self, g, ctx_pos, emb, anchors):
        """Context features g at ctx_pos, block inputs emb [A, 8, n] anchored at anchors -> final-normed rows."""
        cfg = self.cfg
        nh, nkv, hd = cfg["n_head"], cfg["n_kv"], cfg["hd"]
        A = anchors.shape[0]
        bpos = (anchors[:, None] + torch.arange(BLOCK, device=g.device)[None]).reshape(-1)
        T = g.shape[0]
        # context keys: before the anchor and inside the sliding window of each block row; block keys: own block
        dq = bpos[:, None] - ctx_pos[None, :]
        m_ctx = (ctx_pos[None, :] < anchors.repeat_interleave(BLOCK)[:, None]) & (dq < cfg["n_swa"])
        blk = torch.arange(A, device=g.device).repeat_interleave(BLOCK)
        m_blk = blk[:, None] == blk[None, :]
        mask = torch.cat([m_ctx, m_blk], 1)[None, None]
        g_r = self.wht(g)
        x = emb
        for ly in self.layers:
            kc = self.rope(self.rms(ly.wk(g_r).view(T, nkv, hd), ly.k_norm), ctx_pos)
            vc = ly.wv(g_r).view(T, nkv, hd)
            h = self.rms(x, ly.attn_norm)
            dyn = ly.attn_cp(self.wht(h))
            hc_r = self.wht(self.conv(h, dyn, ly.attn_cb, 0))
            q = self.rope(self.rms(ly.wq(hc_r).view(A * BLOCK, nh, hd), ly.q_norm), bpos)
            k = self.rope(self.rms(ly.wk(hc_r).view(A * BLOCK, nkv, hd), ly.k_norm), bpos)
            v = ly.wv(hc_r).view(A * BLOCK, nkv, hd)
            K = torch.cat([kc, k], 0).repeat_interleave(nh // nkv, dim=1)
            V = torch.cat([vc, v], 0).repeat_interleave(nh // nkv, dim=1)
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                                               attn_mask=mask, scale=1.0 / math.sqrt(hd))
            o = o[0].transpose(0, 1).reshape(A, BLOCK, nh * hd)
            o = self.conv(ly.wo(self.wht(o)), dyn, ly.attn_cb, 1)
            x2 = o + x
            h2 = self.rms(x2, ly.ffn_norm)
            dyn2 = ly.ffn_cp(self.wht(h2))
            h2c_r = self.wht(self.conv(h2, dyn2, ly.ffn_cb, 0))
            f = ly.down(self.wht(F.silu(ly.gate(h2c_r)) * ly.up(h2c_r)))
            x = self.conv(f, dyn2, ly.ffn_cb, 1) + x2
        return self.rms(x, self.out_norm)


class Target:
    """Bonsai 2's borrowed embedding (primal basis) and head folded with its activation transform."""

    def __init__(self, path, device, dtype=torch.bfloat16):
        z = np.load(path)
        signs = torch.from_numpy(z["signs"]).float()
        H = hadamard(HB)
        n = z["emb_s"].shape[1] * G
        self.emb = torch.empty(z["emb_s"].shape[0], n, dtype=dtype)
        self.head = torch.empty(z["out_s"].shape[0], n, dtype=dtype)
        for name, dst, kind in (("emb", self.emb, "inv"), ("out", self.head, "fwd")):
            codes, s = z[name + "_codes"], z[name + "_s"]
            for r0 in range(0, codes.shape[0], 8192):
                w = torch.from_numpy(unpack2(codes[r0:r0 + 8192], n)).float() - 1
                w = (w.view(w.shape[0], -1, G) * torch.from_numpy(s[r0:r0 + 8192]).float()[..., None]).view(w.shape[0], n)
                w = (w.view(-1, HB) @ H).view(w.shape[0], n)
                dst[r0:r0 + w.shape[0]] = (w * signs).to(dtype)    # emb: s*(H z); head: (W H) diag(s)
        self.emb = self.emb.to(device)
        self.head = self.head.to(device)


def unpack2(packed, n):
    p = np.asarray(packed, np.uint8)
    out = np.stack([(p >> (2 * j)) & 3 for j in range(4)], -1)
    return out.reshape(p.shape[0], -1)[:, :n]


def pack2(codes):
    c = np.asarray(codes, np.uint8).reshape(codes.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)


class Data:
    def __init__(self, d):
        d = Path(d)
        self.idx = json.loads((d / "index.json").read_text())
        self.ids = np.memmap(d / "ids.bin", np.int32, "r")
        self.feat_dim = self.idx["feat_dim"]
        self.feat = np.memmap(d / "feat.bin", np.float16, "r")
        self.seqs = self.idx["seqs"]

    def get(self, i):
        s = self.seqs[i]
        ids = np.asarray(self.ids[s["ids_off"]:s["ids_off"] + s["n_ids"]])
        f = np.asarray(self.feat[s["feat_off"]:s["feat_off"] + s["n_rows"] * self.feat_dim]).reshape(s["n_rows"], self.feat_dim)
        return s, ids, f


def anchors_for(s, stride, rng):
    """Anchors in the generated part with all 7 labels present: position a predicts ids[a+1 .. a+7]."""
    lo = max(s["n_prompt"] - 1, s["first_pos"] + 1)
    hi = s["n_ids"] - BLOCK
    if hi < lo:
        return []
    off = rng.randrange(stride)
    return list(range(lo + off, hi + 1, stride))


def make_batch(dr, tgt, s, ids, f, anchors, device, mask_id, dtype):
    a = torch.tensor(anchors, device=device)
    lo = max(s["first_pos"], min(anchors) - dr.cfg["n_swa"] + 1)
    hi = max(anchors)
    r0, r1 = lo - s["first_pos"], hi - s["first_pos"]
    g = torch.from_numpy(f[r0:r1].astype(np.float32)).to(device, dtype)
    ctx_pos = torch.arange(lo, hi, device=device)
    toks = torch.full((len(anchors), BLOCK), mask_id, dtype=torch.long)
    toks[:, 0] = torch.from_numpy(ids[anchors].astype(np.int64))
    emb = tgt.emb[toks.to(device)].to(dtype)
    labels = torch.from_numpy(np.stack([ids[x + 1:x + BLOCK] for x in anchors]).astype(np.int64)).to(device)
    return g, ctx_pos, emb, a, labels, toks[:, 0].to(device)


def losses(dr, tgt, hn, labels, anchor_tok, wpos, lat_w):
    A = hn.shape[0]
    rows = hn[:, 1:].reshape(-1, hn.shape[-1])
    logits = (rows @ tgt.head.T.to(rows.dtype)).float()
    ce = F.cross_entropy(logits, labels.reshape(-1), reduction="none").view(A, BLOCK - 1)
    l_ce = (ce * wpos).sum(1).mean()
    stats = dict(ce=float(ce.detach().mean()), top1=float((logits.detach().argmax(-1) == labels.reshape(-1)).float().mean()))
    if lat_w <= 0:
        return l_ce, stats
    lg = logits.view(A, BLOCK - 1, -1)
    cand = torch.topk(lg.detach(), TOP_K, dim=-1).indices                      # [A, 7, 16]
    unary = torch.gather(lg, 2, cand)
    gate = (hn[:, 1:].float() @ dr.sel_hidden.T)                               # [A, 7, rank]
    succ = dr.sel_next[cand].float()                                           # [A, 7, 16, r]
    pred_tok = torch.cat([anchor_tok[:, None], labels[:, :-1]], 1)             # true predecessor token per row
    pv = dr.sel_prev[pred_tok].float()                                         # [A, 7, r]
    score = torch.einsum("ajsr,ajr->ajs", succ, pv * gate) + unary             # column of the true predecessor
    hit = cand == labels[..., None]
    has = hit.any(-1)
    prev_in = torch.cat([torch.ones(A, 1, dtype=torch.bool, device=hn.device), has[:, :-1]], 1)
    ok = has & prev_in
    tgt_slot = hit.float().argmax(-1)
    lce = F.cross_entropy(score.view(-1, TOP_K), tgt_slot.view(-1), reduction="none").view(A, BLOCK - 1)
    l_lat = (lce * ok * wpos).sum(1).mean()
    stats["lat_ok"] = float(ok.float().mean())
    return l_ce + lat_w * l_lat, stats


@torch.no_grad()
def walk_accept(dr, tgt, hn, labels, anchor_tok, ks=(1, 3, 7)):
    """Greedy selector walk per block (rows 1..7) against the labels: accepted drafts for each k."""
    A = hn.shape[0]
    rows = hn[:, 1:]
    lg = (rows.reshape(-1, rows.shape[-1]) @ tgt.head.T.to(rows.dtype)).float().view(A, BLOCK - 1, -1)
    cand = torch.topk(lg, TOP_K, dim=-1).indices
    unary = torch.gather(lg, 2, cand)
    gate = rows.float() @ dr.sel_hidden.T
    succ = dr.sel_next[cand].float()
    out = torch.zeros(A, BLOCK - 1, dtype=torch.long, device=hn.device)
    slot = None
    for j in range(BLOCK - 1):
        if j == 0:
            pv = dr.sel_prev[anchor_tok].float()[:, None]                     # [A, 1, r]
        else:
            pv = dr.sel_prev[cand[:, j - 1]].float()                           # [A, 16, r]
        score = torch.einsum("asr,apr->asp", succ[:, j], pv * gate[:, j][:, None]) + unary[:, j][..., None]
        col = score[:, :, 0] if j == 0 else torch.gather(score, 2, slot[:, None, None].expand(A, TOP_K, 1))[..., 0]
        slot = col.argmax(-1)
        out[:, j] = cand[:, j].gather(1, slot[:, None])[:, 0]
    match = (out == labels).long().cumprod(1)
    return {k: match[:, :k].sum(1).float() for k in ks}, out


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def evaluate(dr, tgt, data, seq_ids, device, mask_id, dtype, max_anchors=4000):
    dr.eval()
    acc = {1: [], 3: [], 7: []}
    n = 0
    rng = random.Random(7)
    with torch.no_grad():
        for i in seq_ids:
            s, ids, f = data.get(i)
            an = anchors_for(s, 4, rng)
            for c in range(0, len(an), 64):
                ch = an[c:c + 64]
                g, ctx_pos, emb, a, labels, atok = make_batch(dr, tgt, s, ids, f, ch, device, mask_id, dtype)
                with torch.autocast(device.type, dtype=dtype, enabled=True):
                    hn = dr(dr.features(g), ctx_pos, emb, a)
                r, _ = walk_accept(dr, tgt, hn, labels, atok)
                for k in acc:
                    acc[k].append(r[k].cpu())
                n += len(ch)
            if n >= max_anchors:
                break
    dr.train()
    return {k: float(torch.cat(v).mean()) for k, v in acc.items()}, n


def export(dr, path, extra):
    out = dict(extra=extra, ternary={}, dense={})
    for name, m in dr.ternary_modules():
        q, s = ternary_l2(m.wr.detach().float())
        out["ternary"][name] = dict(codes=torch.from_numpy(pack2((q + 1).to(torch.uint8).cpu().numpy())),
                                    shape=list(q.shape), s=s.cpu().float())
    for i, ly in enumerate(dr.layers):
        for k in ("attn_norm", "ffn_norm", "q_norm", "k_norm", "attn_cb", "ffn_cb"):
            out["dense"][f"blk.{i}.{GGUF_NAME[k]}"] = getattr(ly, k).detach().cpu().float()
    out["dense"]["output_norm.weight"] = dr.out_norm.detach().cpu().float()
    out["dense"]["selector_hidden.weight"] = dr.sel_hidden.detach().cpu().float()
    if dr.train_fc:
        out["dense"]["enc.output_norm.weight"] = dr.enc_norm.detach().cpu().float()
    torch.save(out, path)


def train(a):
    torch.manual_seed(0)
    rng = random.Random(0)
    device = torch.device(a.device)
    dtype = torch.bfloat16
    sd = load_state(a.init)
    cfg = sd["cfg"]
    data = Data(a.data)
    tgt = Target(a.target, device)
    dr = Drafter(sd, cfg, quant=not a.no_quant, train_fc=data.feat_dim != cfg["n_embd"], device=device).to(device)
    if a.eval_only:
        dr.requires_grad_(False)
    del sd
    n_seq = len(data.seqs)
    order = list(range(n_seq))
    rng.shuffle(order)
    n_val = max(1, int(n_seq * a.val_frac))
    val, trn = order[:n_val], order[n_val:]
    params = [p for p in dr.parameters() if p.requires_grad]
    opt = None if a.eval_only else torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8,
                                                     fused=device.type == "cuda")
    wpos = torch.tensor([a.pos_decay ** j for j in range(BLOCK - 1)], device=device)
    wpos = wpos / wpos.sum() * (BLOCK - 1)
    log = open(Path(a.out) / "train_log.jsonl", "a")
    t0 = time.time()
    best = -1.0
    if a.eval_init or a.eval_only:
        r, n = evaluate(dr, tgt, data, val, device, cfg["mask_id"], dtype, a.eval_anchors)
        rec = dict(step=0, quant=not a.no_quant, val=r, n=n, t=time.time() - t0)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        best = r[3]
        if a.eval_only:
            return
    work = []
    for ep in range(a.epochs):
        ep_items = []
        for i in trn:
            s = data.seqs[i]
            an = anchors_for(s, a.stride, rng)
            for c in range(0, len(an), a.anchors):
                ep_items.append((i, an[c:c + a.anchors]))
        rng.shuffle(ep_items)
        work += ep_items
    total_steps = len(work) // a.accum
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    print(f"{n_seq} seqs ({len(trn)} train, {len(val)} val), {len(work)} micro-batches, {total_steps} steps", flush=True)
    step = 0
    acc_stats = []
    for mb in range(0, total_steps * a.accum):
        i, an = work[mb]
        s, ids, f = data.get(i)
        g, ctx_pos, emb, aa, labels, atok = make_batch(dr, tgt, s, ids, f, an, device, cfg["mask_id"], dtype)
        with torch.autocast(device.type, dtype=dtype, enabled=True):
            hn = dr(dr.features(g), ctx_pos, emb, aa)
        loss, st = losses(dr, tgt, hn, labels, atok, wpos, a.lat_w)
        (loss / a.accum).backward()
        acc_stats.append(st)
        if (mb + 1) % a.accum:
            continue
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * step / total_steps)) * 0.9 + 0.1)
        for pg in opt.param_groups:
            pg["lr"] = lr
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        dr.invalidate()
        step += 1
        if step % a.log_every == 0:
            m = {k: float(np.mean([x[k] for x in acc_stats if k in x])) for k in acc_stats[0]}
            rec = dict(step=step, lr=lr, gn=gn, **m, t=time.time() - t0)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            acc_stats = []
        if step % a.eval_every == 0 or step == total_steps:
            r, n = evaluate(dr, tgt, data, val, device, cfg["mask_id"], dtype, a.eval_anchors)
            rec = dict(step=step, val=r, n=n, t=time.time() - t0)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if r[3] > best:
                best = r[3]
                export(dr, Path(a.out) / "drafter_t2_best.pt", dict(step=step, val=r, args=vars(a)))
    export(dr, Path(a.out) / "drafter_t2_last.pt", dict(step=step, args=vars(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="init state (.pt from dflash2_ftprep.py init)")
    ap.add_argument("--target", required=True, help="bonsai borrowed pieces (.npz from dflash2_ftprep.py target)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--anchors", type=int, default=48)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pos-decay", type=float, default=0.85)
    ap.add_argument("--lat-w", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-anchors", type=int, default=3000)
    ap.add_argument("--eval-init", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    train(a)


if __name__ == "__main__":
    sys.exit(main())
