"""A DFlash 2 block drafter (z-lab's design, the Ternary Bonsai 2 tune) into a q4nx container.

The drafter (llama.cpp arch `dflash`) reads its target's hidden states at five layers, runs five
Qwen3-style layers over a block [last token, MASK x k] and borrows the target's embedding and lm
head. The engine's t2 path (OPEN-QUANT-T2, OPEN-HADAMARD) needs every big matrix as exact-ternary
q4_1 in the 1024-block Walsh-Hadamard basis -- the format the ternary drafter is trained in
(utilities/dflash2_train.py): W = (s q) Hn, q in {-1, 0, 1}, one s per 128 inputs.

Sources for the ternary codes, in order of preference:
  * the trainer's checkpoint (`--drafter-codes x.pt`: codes + scales, exact);
  * the GGUF itself: its bf16 weights are bf16((s q) Hn), and W Hn recovers q and s (the
    recovery is checked; a GGUF that is not ternary in that basis is refused).
Norms, conv_base and the selector tables are copied. Two folds use the TARGET's 5120-wide signs
(its GGUF's prism.hadamard.*): the head reads H(s * hn), so s goes into output_norm, and the
selector gate reads hn unrotated, so its columns take s too (exact sign flips).

Writes <out>/model.q4nx and <out>/config.json (model_type "dflash2"). OPEN-CONVERT-DFLASH2.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

QK = 128
HB = 1024
TERNARY_KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
                 "attn_conv_proj", "ffn_conv_proj")
DENSE_KINDS = ("attn_norm.weight", "ffn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
               "attn_conv_base", "ffn_conv_base")
GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30


def ternary_names(n_layer: int) -> list[str]:
    return ["fc.weight"] + [f"blk.{i}.{k}.weight" for i in range(n_layer) for k in TERNARY_KINDS]


def hadamard(n: int = HB) -> np.ndarray:
    h = np.array([[1.0]], np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _f32(t) -> np.ndarray:
    """A GGUF tensor (F32 / F16 / BF16) as f32 [rows, cols] (reversed GGUF dims)."""
    shape = tuple(int(d) for d in reversed(list(t.shape)))
    d = np.asarray(t.data)
    if t.tensor_type == GGML_F32:
        return d.astype(np.float32).reshape(shape)
    if t.tensor_type == GGML_F16:
        return d.astype(np.float32).reshape(shape)
    if t.tensor_type == GGML_BF16:
        u = d.view("<u2") if d.dtype == np.uint8 else d.astype(np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    raise ValueError(f"{t.name}: tensor type {int(t.tensor_type)} is not F32/F16/BF16")


def recover_ternary(w: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """bf16((s q) Hn) [out, in] -> (codes uint8 = q + 1, s f32 bf16-exact [out, in/128])."""
    o, i = w.shape
    if i % HB:
        raise ValueError(f"{name}: input width {i} is not whole {HB}-blocks")
    r = (w.astype(np.float64).reshape(o, i // HB, HB) @ hadamard()).reshape(o, i // QK, QK)
    a = np.abs(r)
    mx = a.max(-1, keepdims=True)
    q = np.where(a > 0.5 * mx, np.sign(r), 0.0)
    nz = np.maximum((q != 0).sum(-1), 1)
    s = (a * (q != 0)).sum(-1) / nz
    s = torch.from_numpy(s.astype(np.float32)).to(torch.bfloat16).float().numpy()
    resid = np.abs(r - q * s[..., None]).max() / max(float(s.max()), 1e-30)
    if resid > 0.05:
        raise ValueError(f"{name}: not ternary in the Hadamard basis (worst residual {resid:.3f} of s); "
                         "pass the trainer's checkpoint with --drafter-codes")
    return (q + 1).astype(np.uint8).reshape(o, i), s


def codes_from_checkpoint(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for name, e in ck["ternary"].items():
        o, i = (int(v) for v in e["shape"])
        p = e["codes"].numpy().astype(np.uint8)
        c = np.stack([(p >> (2 * j)) & 3 for j in range(4)], -1).reshape(o, -1)[:, :i]
        out[name] = (c.astype(np.uint8), e["s"].float().numpy())
    return out


def pack_q4_1(codes: np.ndarray, s: np.ndarray) -> torch.Tensor:
    """Exact-ternary q4_1 chunks, [rows/32, cols/256, 5120] int8: d = s per 32-block (bf16), m = -d,
    q = code, the q4nx raster and nibble order (model_converter._pack_q4nx, keep_block_in_2D)."""
    rows, cols = codes.shape
    if rows % 32 or cols % 256:
        raise ValueError(f"a {rows} x {cols} matrix is not whole 32 x 256 chunks")
    if codes.max() > 2:
        raise ValueError("code 3 is not ternary")
    d = np.repeat(s.astype(np.float32), QK // 32, axis=1)                       # [rows, cols/32]
    p, q = rows // 32, cols // 256
    dd = d.reshape(p, 32, q, 8).transpose(0, 2, 3, 1).reshape(p, q, 256)         # (c r): g * 32 + r
    db = torch.from_numpy(np.ascontiguousarray(dd)).to(torch.bfloat16)
    mb = (-db.float()).to(torch.bfloat16)
    qw = codes.reshape(p, 32, q, 256).transpose(0, 2, 1, 3)                      # [p, q, r, c]
    qw = qw.reshape(p, q, 2, 16, 256).transpose(0, 1, 2, 4, 3)                   # [p, q, g, c, r16]
    qw = qw.reshape(p, q, 2, 256, 8, 2)
    nib = (qw[..., 0] | (qw[..., 1] << 4)).astype(np.uint8).reshape(p, q, 4096)
    out = np.concatenate([db.view(torch.int16).numpy().view(np.uint8).reshape(p, q, 512),
                          mb.view(torch.int16).numpy().view(np.uint8).reshape(p, q, 512), nib], axis=-1)
    return torch.from_numpy(out.view(np.int8))


def target_signs(target_gguf: str) -> np.ndarray:
    from gguf import GGUFReader
    from .prism import PrismRotation, register_pq2_0
    register_pq2_0()
    rot = PrismRotation(GGUFReader(target_gguf))
    return rot.s_hidden.astype(np.float32)


def build_dflash2_drafter(gguf_path: str, out_dir: str, target_gguf: str, codes_pt: str | None = None) -> dict:
    from gguf import GGUFReader
    from safetensors.torch import save_file

    rd = GGUFReader(gguf_path)
    kv = {k: f.contents() for k, f in rd.fields.items() if not k.startswith("tokenizer.ggml.tokens")
          and not k.startswith("tokenizer.ggml.merges") and not k.startswith("tokenizer.ggml.token_type")}
    if kv.get("general.architecture") != "dflash":
        raise ValueError(f"{gguf_path}: general.architecture is {kv.get('general.architecture')!r}, not 'dflash'")

    def need(key):
        if key not in kv:
            raise ValueError(f"{gguf_path} lacks {key}")
        return kv[key]

    n_layer = int(need("dflash.block_count"))
    hidden = int(need("dflash.embedding_length"))
    tensors = {t.name: t for t in rd.tensors}
    s_hidden = target_signs(target_gguf)
    if len(s_hidden) != hidden:
        raise ValueError(f"the target's hidden signs are {len(s_hidden)} wide, the drafter is {hidden}")
    ck = codes_from_checkpoint(codes_pt) if codes_pt else {}

    out: dict[str, torch.Tensor] = {}
    worst = 0.0
    for name in ternary_names(n_layer):
        w = _f32(tensors[name])
        if name in ck:
            codes, s = ck[name]
            # the checkpoint and the GGUF must be the same drafter (a row sample is proof enough)
            r = slice(0, min(256, w.shape[0]))
            ref = (codes[r].astype(np.float32) - 1).reshape(-1, s.shape[1], QK) * s[r, :, None]
            ref = (ref.reshape(ref.shape[0], -1, HB) @ hadamard().astype(np.float32)).reshape(ref.shape[0], -1)
            err = np.abs(ref - w[r]).max() / max(np.abs(w[r]).max(), 1e-30)
            if err > 1e-2:
                raise ValueError(f"{name}: --drafter-codes does not match the GGUF (max rel diff {err:.3g})")
        else:
            codes, s = recover_ternary(w, name)
        worst = max(worst, float(np.abs(s).max()))
        out[name] = pack_q4_1(codes, s)
    for i in range(n_layer):
        for k in DENSE_KINDS:
            nm = f"blk.{i}.{k}"
            out[nm] = torch.from_numpy(np.ascontiguousarray(_f32(tensors[nm])))
    out["enc.output_norm.weight"] = torch.from_numpy(_f32(tensors["enc.output_norm.weight"]).copy())
    out["output_norm.weight"] = torch.from_numpy(_f32(tensors["output_norm.weight"]) * s_hidden)
    out["selector_hidden.weight"] = torch.from_numpy(_f32(tensors["selector_hidden.weight"]) * s_hidden[None, :])
    for nm in ("selector_predecessor.weight", "selector_successor.weight"):
        out[nm] = torch.from_numpy(_f32(tensors[nm])).to(torch.bfloat16)

    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    save_file(out, str(od / "model.q4nx"))
    taps = [int(x) for x in need("dflash.target_layers")]
    cfg = {
        "model_type": "dflash2",
        "architectures": ["DFlash2Drafter"],
        "hidden_size": hidden,
        "num_hidden_layers": n_layer,
        "num_attention_heads": int(need("dflash.attention.head_count")),
        "num_key_value_heads": int(need("dflash.attention.head_count_kv")),
        "head_dim": int(need("dflash.attention.key_length")),
        "intermediate_size": int(need("dflash.feed_forward_length")),
        "rms_norm_eps": float(need("dflash.attention.layer_norm_rms_epsilon")),
        "rope_theta": float(need("dflash.rope.freq_base")),
        "sliding_window": int(kv.get("dflash.attention.sliding_window", 0)),
        # llama.cpp's GGUF names the layer whose INPUT is the tap; the tap is the output of the one before
        "target_layer_ids": [t - 1 for t in taps],
        "conv_kernel_size": int(need("dflash.conv_kernel_size")),
        "conv_group_size": int(need("dflash.conv_group_size")),
        "selector_rank": int(need("dflash.selector_rank")),
        "selector_top_k": int(need("dflash.selector_top_k")),
        "mask_token_id": int(need("tokenizer.ggml.mask_token_id")),
        "vocab_size": int(tensors["selector_successor.weight"].shape[1]),
        "hadamard": {"block_size": HB, "basis": "unsigned"},
        "borrowed": {"embedding": "target", "lm_head": "target", "signs_folded": "output_norm,selector_hidden_cols"},
        "ternary_source": "checkpoint" if ck else "gguf",
    }
    (od / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(od), "tensor_count": len(out), "files": ["model.q4nx", "config.json"],
            "ternary_source": cfg["ternary_source"], "source": os.path.abspath(gguf_path)}
