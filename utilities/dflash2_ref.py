"""DFlash 2 drafter reference (CPU, torch fp32) for Ternary Bonsai 2; `gate` checks it against tf3.exe dumps."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

GGML_F32, GGML_F16, GGML_BF16, GGML_PQ2_0 = 0, 1, 30, 142
PQ2_BYTES, QK = 34, 128
TOP_K = 16
LAT_W = TOP_K + TOP_K * TOP_K
N_SWA = 2048


class Gguf:
    """Minimal GGUF reader: metadata and memory-mapped tensor data."""

    _SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def __init__(self, path):
        self.path = str(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.pos = 0
        assert self._read(4) == b"GGUF"
        self._u32()
        n_t, n_kv = self._u64(), self._u64()
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            self.kv[k] = self._val(self._u32())
        self.tensors = {}
        for _ in range(n_t):
            name = self._str()
            dims = [self._u64() for _ in range(self._u32())]
            self.tensors[name] = (dims, self._u32(), self._u64())
        align = int(self.kv.get("general.alignment", 32))
        self.data0 = (self.pos + align - 1) // align * align

    def _read(self, n):
        b = bytes(self.mm[self.pos:self.pos + n])
        self.pos += n
        return b

    def _u32(self):
        return struct.unpack("<I", self._read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._read(8))[0]

    def _str(self):
        return self._read(self._u64()).decode("utf-8", "replace")

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._u32(), self._u64()
            if et in self._SCALAR:
                fmt = self._SCALAR[et]
                sz = struct.calcsize(fmt)
                return list(np.frombuffer(self._read(sz * n), dtype=fmt))
            return [self._val(et) for _ in range(n)]
        fmt = self._SCALAR[t]
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]

    def raw(self, name):
        dims, t, off = self.tensors[name]
        n = int(np.prod(dims))
        nbytes = {GGML_F32: 4 * n, GGML_F16: 2 * n, GGML_BF16: 2 * n, GGML_PQ2_0: n // QK * PQ2_BYTES}[t]
        return self.mm[self.data0 + off:self.data0 + off + nbytes], dims, t

    def f32(self, name):
        """numpy f32 array, shape = reversed GGUF dims ([out, in] for a matmul weight)."""
        b, dims, t = self.raw(name)
        shape = tuple(int(d) for d in reversed(dims))
        if t == GGML_F32:
            return np.frombuffer(b, np.float32).reshape(shape)
        if t == GGML_F16:
            return np.frombuffer(b, np.float16).astype(np.float32).reshape(shape)
        if t == GGML_BF16:
            return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if t == GGML_PQ2_0:
            return pq2_rows(b, dims, None)
        raise ValueError(f"{name}: unsupported type {t}")


def pq2_rows(b, dims, rows):
    """PQ2_0 [k, n] -> f32 [len(rows) or n, k]: value = s*code - s, code lane i of a u32 at bits 2i."""
    k, n = int(dims[0]), int(dims[1])
    blk = np.frombuffer(b, np.uint8).reshape(n, k // QK, PQ2_BYTES)
    if rows is not None:
        blk = blk[np.asarray(rows)]
    s = np.ascontiguousarray(blk[..., :2]).view(np.float16).astype(np.float32)          # [r, nb, 1]
    w = np.ascontiguousarray(blk[..., 2:]).view(np.uint32)                               # [r, nb, 8]
    codes = (w[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3                   # [r, nb, 8, 16]
    v = codes.reshape(blk.shape[0], k // QK, QK).astype(np.float32) * s - s
    return v.reshape(blk.shape[0], k)


def hadamard(n=1024):
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return torch.from_numpy(h / math.sqrt(n))


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_neox(x, pos, base, n_rot=128):
    """NeoX pairs (i, i + n_rot/2); MROPE with sections [n_rot/2, 0, 0, 0] is the same rotation."""
    half = n_rot // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float64, device=x.device) * 2.0 / n_rot)
    ang = pos.to(torch.float64)[:, None] * inv[None, :]
    c, s = torch.cos(ang).float()[:, None, :], torch.sin(ang).float()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:n_rot]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., n_rot:]], dim=-1)


def f16_round(x):
    return x.half().float()


class Target:
    """Ternary Bonsai 2's borrowed pieces: the latent token embedding and the rotated lm head."""

    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        g = Gguf(path)
        self.g = g
        bs = int(g.kv["prism.hadamard.block_size"])
        widths = [int(w) for w in g.kv["prism.hadamard.sign_widths"]]
        vals = np.asarray(g.kv["prism.hadamard.sign_values"], np.float32)
        off = {w: sum(widths[:i]) for i, w in enumerate(widths)}
        self.n_embd = int(g.kv["qwen35.embedding_length"])
        self.signs = torch.from_numpy(vals[off[self.n_embd]:off[self.n_embd] + self.n_embd].copy()).to(self.device)
        self.H = hadamard(bs).to(self.device)
        self.bs = bs
        self.emb_raw, self.emb_dims, _ = g.raw("token_embd.weight")
        self.head = None

    def wht(self, x):
        sh = x.shape
        return (x.reshape(-1, self.bs) @ self.H).reshape(sh)

    def embed(self, ids):
        """h = s * (H z): the inverse transform applied right after the lookup."""
        z = torch.from_numpy(pq2_rows(self.emb_raw, self.emb_dims, ids)).to(self.device)
        return self.wht(z) * self.signs

    def load_head(self):
        if self.head is None:
            b, dims, _ = self.g.raw("output.weight")
            n = int(dims[1])
            self.head = torch.empty(n, self.n_embd, device=self.device)
            for r0 in range(0, n, 16384):
                r = np.arange(r0, min(n, r0 + 16384))
                self.head[r0:r0 + len(r)] = torch.from_numpy(pq2_rows(b, dims, r)).to(self.device)
        return self.head

    def logits(self, x):
        """logits = W (H (s * x)): build_lora_mm's activation transform for a rotated head."""
        return self.wht(x * self.signs) @ self.load_head().T


class Drafter:
    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        g = Gguf(path)
        self.g = g
        kv = g.kv
        self.eps = float(kv["dflash.attention.layer_norm_rms_epsilon"])
        self.base = float(kv["dflash.rope.freq_base"])
        self.n_layer = int(kv["dflash.block_count"])
        self.n_head = int(kv["dflash.attention.head_count"])
        self.n_kv = int(kv["dflash.attention.head_count_kv"])
        self.hd = int(kv["dflash.attention.key_length"])
        self.taps = [int(x) for x in kv["dflash.target_layers"]]
        self.kernel = int(kv["dflash.conv_kernel_size"])
        self.group = int(kv["dflash.conv_group_size"])
        self.rank = int(kv["dflash.selector_rank"])
        self.top_k = int(kv["dflash.selector_top_k"])
        self.mask_id = int(kv["tokenizer.ggml.mask_token_id"])
        self.n_swa = int(kv.get("dflash.attention.sliding_window", 0)) or None
        assert self.kernel == 2 and self.top_k == TOP_K
        t = lambda n: torch.from_numpy(np.array(g.f32(n))).to(self.device)
        self.fc = t("fc.weight")
        self.enc_norm = t("enc.output_norm.weight")
        self.out_norm = t("output_norm.weight")
        self.sel_hidden = t("selector_hidden.weight")          # [rank, n_embd]
        self.sel_prev = t("selector_predecessor.weight")       # [vocab, rank]
        self.sel_next = t("selector_successor.weight")
        self.L = []
        for i in range(self.n_layer):
            p = f"blk.{i}."
            self.L.append({k: t(p + n) for k, n in [
                ("attn_norm", "attn_norm.weight"), ("wq", "attn_q.weight"), ("wk", "attn_k.weight"),
                ("wv", "attn_v.weight"), ("wo", "attn_output.weight"), ("q_norm", "attn_q_norm.weight"),
                ("k_norm", "attn_k_norm.weight"), ("ffn_norm", "ffn_norm.weight"), ("gate", "ffn_gate.weight"),
                ("up", "ffn_up.weight"), ("down", "ffn_down.weight"),
                ("attn_cb", "attn_conv_base"), ("attn_cp", "attn_conv_proj.weight"),
                ("ffn_cb", "ffn_conv_base"), ("ffn_cp", "ffn_conv_proj.weight")]})

    def encode(self, taps, pos, f16_cache=True):
        """taps [T, n_tap * n_embd] at target positions pos -> per layer (K [T, kv, hd] roped, V)."""
        g = rms(taps @ self.fc.T, self.enc_norm, self.eps)
        out = []
        for ly in self.L:
            k = rms((g @ ly["wk"].T).view(-1, self.n_kv, self.hd), ly["k_norm"], self.eps)
            k = rope_neox(k, pos, self.base)
            v = (g @ ly["wv"].T).view(-1, self.n_kv, self.hd)
            out.append((f16_round(k), f16_round(v)) if f16_cache else (k, v))
        return out

    def conv(self, x, dyn, base, side):
        """DFlash 2 dynamic 2-tap grouped conv over block rows: y[t] = w0[t] x[t] + w1[t] x[t-1]."""
        B, n = x.shape
        ng = n // self.group
        c = dyn.view(B, 2, self.kernel, ng)[:, side]                       # [B, tap, g]
        w = c.repeat_interleave(self.group, dim=-1) + base[side][None]      # [B, tap, n]
        prev = torch.cat([torch.zeros(1, n, device=x.device), x[:-1]], dim=0)
        return w[:, 0] * x + w[:, 1] * prev

    def block(self, tgt, anchor, p0, n_draft, cache, cache_pos, f16_cache=True):
        """[anchor, MASK x n_draft] at positions p0.. -> (final-normed rows [B, n], logits [B, vocab])."""
        B = n_draft + 1
        x = tgt.embed([anchor] + [self.mask_id] * n_draft)
        pos = torch.arange(p0, p0 + B, device=self.device)
        rep = self.n_head // self.n_kv
        for il, ly in enumerate(self.L):
            h = rms(x, ly["attn_norm"], self.eps)
            dyn = h @ ly["attn_cp"].T
            h = self.conv(h, dyn, ly["attn_cb"], 0)
            q = rms((h @ ly["wq"].T).view(B, self.n_head, self.hd), ly["q_norm"], self.eps)
            k = rms((h @ ly["wk"].T).view(B, self.n_kv, self.hd), ly["k_norm"], self.eps)
            v = (h @ ly["wv"].T).view(B, self.n_kv, self.hd)
            q, k = rope_neox(q, pos, self.base), rope_neox(k, pos, self.base)
            if f16_cache:
                k, v = f16_round(k), f16_round(v)
            ck, cv = cache[il]
            K = torch.cat([ck, k], 0)
            V = torch.cat([cv, v], 0)
            kp = torch.cat([cache_pos, pos])
            allowed = torch.ones(B, len(kp), dtype=torch.bool, device=self.device)
            if self.n_swa:
                allowed = (pos[:, None] - kp[None, :]) < self.n_swa
            Kr = K.repeat_interleave(rep, dim=1)                              # head h reads kv head h // rep
            Vr = V.repeat_interleave(rep, dim=1)
            s = torch.einsum("bhd,khd->hbk", q, Kr) / math.sqrt(self.hd)
            s = s.masked_fill(~allowed[None], float("-inf"))
            o = torch.einsum("hbk,khd->bhd", torch.softmax(s, -1), Vr).reshape(B, -1)
            o = self.conv(o @ ly["wo"].T, dyn, ly["attn_cb"], 1)
            x2 = o + x
            h2 = rms(x2, ly["ffn_norm"], self.eps)
            dyn2 = h2 @ ly["ffn_cp"].T
            h2 = self.conv(h2, dyn2, ly["ffn_cb"], 0)
            f = (torch.nn.functional.silu(h2 @ ly["gate"].T) * (h2 @ ly["up"].T)) @ ly["down"].T
            x = self.conv(f, dyn2, ly["ffn_cb"], 1) + x2
        hn = rms(x, self.out_norm, self.eps)
        return hn, tgt.logits(hn)

    def lattice(self, anchor, hn, logits):
        """build_dflash2_selector's packed rows 1..B-1: [16 ids | scores at s + 16 p], p = predecessor slot."""
        lg = logits[1:]
        cand = torch.topk(lg, self.top_k, dim=-1).indices                   # [k, 16], descending
        unary = torch.gather(lg, 1, cand)
        gate = hn[1:] @ self.sel_hidden.T                                    # [k, rank]
        succ = self.sel_next[cand]                                           # [k, 16, rank]
        rows = []
        for i in range(lg.shape[0]):
            pred = self.sel_prev[[anchor]] if i == 0 else self.sel_prev[cand[i - 1]]   # [n_pred, rank]
            score = succ[i] @ (pred * gate[i]).T + unary[i][:, None]                     # [s, p]
            if i == 0:
                score = score.expand(self.top_k, self.top_k)
            rows.append(torch.cat([cand[i].float(), score.T.reshape(-1)]))  # p-major: s fastest
        return torch.stack(rows)

    def draft(self, tgt, anchor, p0, n_draft, cache, cache_pos, f16_cache=True):
        hn, logits = self.block(tgt, anchor, p0, n_draft, cache, cache_pos, f16_cache)
        lat = self.lattice(anchor, hn, logits)
        return walk(lat.cpu().numpy(), self.top_k), lat.cpu()


def walk(lat, top_k=TOP_K):
    """common/speculative.cpp's greedy walk (beam 1, p_min 0); position 1 reads predecessor column 0."""
    out, pred = [], 0
    for row in lat:
        scores = row[top_k + pred * top_k: top_k + (pred + 1) * top_k]
        pred = int(np.argmax(scores))
        out.append(int(row[pred]))
    return out


def load_run(run, pid, key):
    lines = (Path(run) / f"{pid}.jsonl").read_text().splitlines()
    hdr = json.loads(lines[0])
    rows = [json.loads(x) for x in lines[1:]]
    return hdr, rows


def margin(lat_row, pred, top_k=TOP_K):
    s = np.sort(lat_row[top_k + pred * top_k: top_k + (pred + 1) * top_k])
    return float(s[-1] - s[-2])


def gate(a):
    torch.set_num_threads(a.threads)
    tgt = Target(a.target, a.device)
    dr = Drafter(a.drafter, a.device)
    k = int(a.key.rsplit("_k", 1)[1])
    nt = len(dr.taps) * tgt.n_embd
    report = {}
    for pid in a.pids:
        hdr, rows = load_run(a.run, pid, a.key)
        if a.taps_file_suffix == ".taps":
            assert hdr["taps"] == dr.taps, (hdr["taps"], dr.taps)
        else:
            assert len(hdr["taps_x"]) == len(dr.taps)
        n0 = hdr["prompt_n"] - 1
        taps = np.fromfile(Path(a.run) / f"{pid}{a.taps_file_suffix}", np.float32).reshape(-1, nt)
        lat_f = Path(a.run) / f"{pid}.{a.key}.lat"
        lat_ll = np.fromfile(lat_f, np.float32).reshape(-1, k, LAT_W) if lat_f.exists() else None
        steps = list(range(0, len(rows), a.stride))
        if a.max_steps:
            steps = steps[:a.max_steps]
        last = n0 + steps[-1]
        with torch.no_grad():
            cache_all = dr.encode(torch.from_numpy(taps[:last]).to(dr.device), torch.arange(last, device=dr.device))
        res = dict(steps=0, same=0, first_tok_same=0, mism=[], cand_overlap=0.0, score_err=[])
        for s in steps:
            p0 = n0 + s
            anchor = hdr["prompt_ids"][-1] if s == 0 else rows[s - 1]["g"]
            lo = max(0, p0 - dr.n_swa + 1) if dr.n_swa else 0
            cache = [(K[lo:p0], V[lo:p0]) for K, V in cache_all]
            with torch.no_grad():
                d, lat = dr.draft(tgt, anchor, p0, k, cache, torch.arange(lo, p0, device=dr.device))
            ll = rows[s]["d"][a.key]
            res["steps"] += 1
            res["same"] += d == ll
            res["first_tok_same"] += d[0] == ll[0]
            lat = lat.numpy()
            if lat_ll is not None:
                ov = [len(set(lat[i, :TOP_K].astype(int)) & set(lat_ll[s, i, :TOP_K].astype(int))) / TOP_K for i in range(k)]
                res["cand_overlap"] += float(np.mean(ov))
                # row-1 scores of candidates both sides hold: the unary logit plus the anchor pair term
                m_ref = {int(c): lat[0, TOP_K + j] for j, c in enumerate(lat[0, :TOP_K])}
                m_ll = {int(c): lat_ll[s, 0, TOP_K + j] for j, c in enumerate(lat_ll[s, 0, :TOP_K])}
                common = set(m_ref) & set(m_ll)
                res["score_err"].append(float(np.max([abs(m_ref[c] - m_ll[c]) for c in common])) if common else None)
            if d != ll:
                i = next(j for j in range(k) if d[j] != ll[j])
                pr = 0 if i == 0 else int(np.where(lat[i - 1, :TOP_K].astype(int) == d[i - 1])[0][0])
                m_r = margin(lat[i], pr)
                m_l = None
                if lat_ll is not None:
                    prl = 0 if i == 0 else int(np.where(lat_ll[s, i - 1, :TOP_K].astype(int) == ll[i - 1])[0][0])
                    m_l = margin(lat_ll[s, i], prl)
                res["mism"].append(dict(step=s, row=i + 1, ref=d[i], ll=ll[i], margin_ref=m_r, margin_ll=m_l))
        se = [x for x in res["score_err"] if x is not None]
        res["score_err_max"] = max(se) if se else None
        res["score_err_med"] = float(np.median(se)) if se else None
        res["cand_overlap"] /= max(1, res["steps"])
        del res["score_err"]
        report[pid] = res
        nt_ = sum(1 for m in res["mism"] if min(m["margin_ref"], m["margin_ll"] if m["margin_ll"] is not None else 9) < a.tie)
        print(f"{pid}: {res['same']}/{res['steps']} drafts identical, first token {res['first_tok_same']}/{res['steps']}, "
              f"mismatches {len(res['mism'])} ({nt_} near-tie < {a.tie}), cand overlap {res['cand_overlap']:.3f}, "
              f"row-1 score |err| median {res['score_err_med']} max {res['score_err_max']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    tot = sum(r["steps"] for r in report.values())
    same = sum(r["same"] for r in report.values())
    mism = [m for r in report.values() for m in r["mism"]]
    tie = sum(1 for m in mism if min(m["margin_ref"], m["margin_ll"] if m["margin_ll"] is not None else 9) < a.tie)
    print(f"TOTAL {same}/{tot} drafts identical ({same / max(1, tot):.4f}); {len(mism)} mismatches, {tie} near-tie (< {a.tie})")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate")
    g.add_argument("--run", required=True)
    g.add_argument("--key", required=True)
    g.add_argument("--drafter", required=True)
    g.add_argument("--target", required=True)
    g.add_argument("--pids", nargs="+", required=True)
    g.add_argument("--stride", type=int, default=1)
    g.add_argument("--max-steps", type=int, default=0)
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--tie", type=float, default=0.1)
    g.add_argument("--taps-file-suffix", default=".taps")
    g.add_argument("--out")
    g.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "gate":
        gate(a)


if __name__ == "__main__":
    sys.exit(main())
