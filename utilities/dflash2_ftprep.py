"""Inputs and outputs of dflash2_train.py: Bonsai's borrowed pieces, the init state, packed teacher data, GGUF export."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dflash2_ref import GGML_BF16, GGML_F32, Gguf, pq2_rows  # noqa: E402

HF = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight", "wv": "self_attn.v_proj.weight",
      "wo": "self_attn.o_proj.weight", "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
      "down": "mlp.down_proj.weight", "attn_cp": "attention_conv.kernel_projection.weight",
      "ffn_cp": "mlp_conv.kernel_projection.weight", "attn_norm": "input_layernorm.weight",
      "ffn_norm": "post_attention_layernorm.weight", "q_norm": "self_attn.q_norm.weight",
      "k_norm": "self_attn.k_norm.weight", "attn_cb": "attention_conv.base_kernel", "ffn_cb": "mlp_conv.base_kernel"}
GG = {"wq": "attn_q.weight", "wk": "attn_k.weight", "wv": "attn_v.weight", "wo": "attn_output.weight",
      "gate": "ffn_gate.weight", "up": "ffn_up.weight", "down": "ffn_down.weight", "attn_cp": "attn_conv_proj.weight",
      "ffn_cp": "ffn_conv_proj.weight", "attn_norm": "attn_norm.weight", "ffn_norm": "ffn_norm.weight",
      "q_norm": "attn_q_norm.weight", "k_norm": "attn_k_norm.weight", "attn_cb": "attn_conv_base", "ffn_cb": "ffn_conv_base"}
CFG = dict(n_embd=5120, n_head=32, n_kv=8, hd=128, group=16, n_swa=2048, rope_base=1e7, eps=1e-6, mask_id=248070)


def target(a):
    """Bonsai 2 token_embd and output as packed 2-bit codes + fp16 scales, and the 5120 signs."""
    g = Gguf(a.target)
    widths = [int(w) for w in g.kv["prism.hadamard.sign_widths"]]
    vals = np.asarray(g.kv["prism.hadamard.sign_values"], np.int8)
    off = {w: sum(widths[:i]) for i, w in enumerate(widths)}
    out = dict(signs=vals[off[5120]:off[5120] + 5120].copy())
    for key, name in (("emb", "token_embd.weight"), ("out", "output.weight")):
        b, dims, _ = g.raw(name)
        n, k = int(dims[1]), int(dims[0])
        blk = np.frombuffer(b, np.uint8).reshape(n, k // 128, 34)
        out[key + "_s"] = np.ascontiguousarray(blk[..., :2]).view(np.float16).reshape(n, k // 128).copy()
        w = np.ascontiguousarray(blk[..., 2:]).view(np.uint32).reshape(n, -1)
        codes = ((w[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).astype(np.uint8).reshape(n, k)
        c = codes.reshape(n, -1, 4)
        out[key + "_codes"] = (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
        # spot check against the reference decoder
        r = np.array([0, 1234, n - 1])
        ref = pq2_rows(b, dims, r)
        cc = codes[r].astype(np.float32).reshape(3, -1, 128)
        s = out[key + "_s"][r].astype(np.float32)[..., None]
        assert np.array_equal((cc * s - s).reshape(3, k), ref)
    np.savez(a.out, **out)
    print("wrote", a.out, {k: v.shape for k, v in out.items()})


def init(a):
    """Init state for the trainer from naklitechie's safetensors or the converted bf16 GGUF (identical values)."""
    sd = dict(cfg=CFG, layers=[])
    if a.src.endswith(".safetensors"):
        from safetensors import safe_open
        f = safe_open(a.src, "pt")
        t = lambda n: f.get_tensor(n).to(torch.bfloat16)
        for i in range(5):
            sd["layers"].append({k: t(f"layers.{i}.{v}") for k, v in HF.items()})
        sd.update(out_norm=t("norm.weight"), sel_hidden=t("candidate_selector.hidden_projection.weight"),
                  fc=t("fc.weight"), enc_norm=t("hidden_norm.weight"),
                  sel_prev=t("candidate_selector.predecessor_codebook.weight"),
                  sel_next=t("candidate_selector.successor_codebook.weight"))
    else:
        g = Gguf(a.src)
        t = lambda n: torch.from_numpy(np.array(g.f32(n))).to(torch.bfloat16)
        for i in range(5):
            sd["layers"].append({k: t(f"blk.{i}.{v}") for k, v in GG.items()})
        sd.update(out_norm=t("output_norm.weight"), sel_hidden=t("selector_hidden.weight"), fc=t("fc.weight"),
                  enc_norm=t("enc.output_norm.weight"), sel_prev=t("selector_predecessor.weight"),
                  sel_next=t("selector_successor.weight"))
    torch.save(sd, a.out)
    print("wrote", a.out)


def read_bin(p):
    b = np.fromfile(p, np.uint8)
    n_prompt, n_ids, first_pos, n_rows = np.frombuffer(b[:16], np.int32)
    ids = np.frombuffer(b[16:16 + 4 * n_ids], np.int32)
    taps = np.frombuffer(b[16 + 4 * n_ids:], np.float16).reshape(n_rows, -1)
    return int(n_prompt), int(n_ids), int(first_pos), ids, taps


def pack(a):
    """tfgen .bin -> index.json + ids.bin + feat.bin; feat g = hidden_norm(fc x) with --fc-gguf's fc, or raw taps."""
    torch.set_num_threads(a.threads)
    files = sorted(Path(a.src).glob("*.bin"))
    if a.limit:
        files = files[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fc = norm = None
    if a.feat == "g":
        g = Gguf(a.fc_gguf)
        fc = torch.from_numpy(np.array(g.f32("fc.weight"))).to(torch.bfloat16)
        norm = torch.from_numpy(np.array(g.f32("enc.output_norm.weight")))
        eps = float(g.kv["dflash.attention.layer_norm_rms_epsilon"])
    cls = {}
    if a.prompts:
        for line in open(a.prompts, encoding="utf-8"):
            r = json.loads(line)
            cls[r["id"]] = r["cls"]
    fi = open(out / "ids.bin", "wb")
    ff = open(out / "feat.bin", "wb")
    fl = open(out / "lab.bin", "wb")
    seqs, ids_off, feat_off, lab_off, dim = [], 0, 0, 0, None
    for n, p in enumerate(files):
        n_prompt, n_ids, first_pos, ids, taps = read_bin(p)
        if n_ids - n_prompt < 9:
            continue
        if a.feat == "g":
            x = torch.from_numpy(taps.astype(np.float32))
            feats = []
            for r0 in range(0, x.shape[0], 1024):
                y = (x[r0:r0 + 1024].to(torch.bfloat16) @ fc.T).float()
                y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * norm
                feats.append(y.half().numpy())
            f = np.concatenate(feats)
        else:
            f = taps
        dim = f.shape[1]
        fi.write(ids.astype(np.int32).tobytes())
        ff.write(f.astype(np.float16).tobytes())
        seq = dict(id=p.stem, cls=cls.get(p.stem, "?"), n_prompt=n_prompt, n_ids=n_ids, first_pos=first_pos,
                   n_rows=int(f.shape[0]), ids_off=ids_off, feat_off=feat_off)
        labf = p.with_suffix(".lab")
        if labf.exists():   # teacher-forced text: the target's argmax after positions n_prompt-1 .. n_ids-2
            lab = np.fromfile(labf, np.int32)
            assert len(lab) == n_ids - n_prompt, p
            fl.write(lab.tobytes())
            seq.update(lab_off=lab_off, n_lab=int(len(lab)))
            lab_off += len(lab)
        seqs.append(seq)
        ids_off += n_ids
        feat_off += f.size
        if n % 50 == 0:
            print(f"{n}/{len(files)} {p.stem} rows {f.shape[0]}", flush=True)
    fi.close(); ff.close(); fl.close()
    gen = sum(s["n_ids"] - s["n_prompt"] for s in seqs)
    (out / "index.json").write_text(json.dumps(dict(feat_dim=dim, feat=a.feat, seqs=seqs)))
    print(f"packed {len(seqs)} sequences, {gen} generated tokens, {feat_off // max(1, dim)} feature rows, dim {dim}")


def export(a):
    """Trained state -> bf16 GGUF on a copy of --template ((s q) Hn per ternary tensor), codes re-checked from the file."""
    from dflash2_train import G, HB, hadamard, unpack2, ternary_l2
    st = torch.load(a.state, map_location="cpu", weights_only=False)
    H = hadamard(HB)
    shutil.copyfile(a.template, a.dst)
    g = Gguf(a.dst)
    mm = np.memmap(a.dst, dtype=np.uint8, mode="r+")

    def view(name):
        dims, t, off = g.tensors[name]
        n = int(np.prod(dims))
        shape = tuple(int(d) for d in reversed(dims))
        if t == GGML_BF16:
            return mm[g.data0 + off:g.data0 + off + 2 * n].view(np.uint16).reshape(shape), t
        assert t == GGML_F32, name
        return mm[g.data0 + off:g.data0 + off + 4 * n].view(np.float32).reshape(shape), t

    worst = 0.0
    for name, e in st["ternary"].items():
        o, i = e["shape"]
        q = torch.from_numpy(unpack2(e["codes"].numpy(), i).astype(np.float32)) - 1
        s = e["s"].float()
        wq_r = (q.view(o, i // G, G) * s[..., None]).view(o, i)
        w = (wq_r.view(o, -1, HB) @ H).view(o, i)
        v, t = view(name)
        v[...] = w.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        back = torch.from_numpy((v.astype(np.uint32) << 16).view(np.float32))
        r = (back.view(o, -1, HB) @ H).view(o, i // G, G)
        q2 = torch.round(r / s[..., None].clamp_min(1e-30)).clamp(-1, 1)
        worst = max(worst, float((q2 != q.view(o, i // G, G)).float().mean()))
    for name, w in st["dense"].items():
        v, t = view(name)
        x = w.float().reshape(v.shape)
        v[...] = x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16) if t == GGML_BF16 else x.numpy()
    mm.flush()
    print(f"wrote {a.dst}: {len(st['ternary'])} ternary + {len(st['dense'])} dense tensors; "
          f"codes recovered from the file, worst disagreement {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("target"); p.add_argument("--target", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("init"); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("pack")
    p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--feat", choices=["g", "taps"], default="g"); p.add_argument("--fc-gguf")
    p.add_argument("--prompts"); p.add_argument("--limit", type=int, default=0); p.add_argument("--threads", type=int, default=6)
    p = sp.add_parser("export")
    p.add_argument("--state", required=True); p.add_argument("--template", required=True); p.add_argument("--dst", required=True)
    a = ap.parse_args()
    {"target": target, "init": init, "pack": pack, "export": export}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
