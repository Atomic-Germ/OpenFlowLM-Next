r"""The Qwen3.6 vision tower, twice: transformers' Qwen3VLVisionModel loaded with the
shipped `vision_weight.q4nx` (the oracle), and a numpy fp32 forward written the way the
C++ port will be (issue #16 item 3, phase A). They must agree; the port is then checked
against the numpy one on the same fixture.

    python open_kernels/model/replica_vit.py [--model-dir DIR] [--grid 16 16] [--fixture OUT]

What the container holds (333 bf16 tensors, no deepstack mergers -- the 35B does not
use deepstack): patch_embed.proj [1152, 3, 2, 16, 16] + bias, pos_embed [2304, 1152],
27 blocks of {norm1, attn.qkv, attn.proj, norm2, mlp.linear_fc1, mlp.linear_fc2}, and
merger.{norm, linear_fc1, linear_fc2}. Every linear weight is stored PRE-TILED for the
closed engine's vision_mm kernel (config.json: VISION_MM_TILE_M 128 / K 256 / N 64):
[n_out/64][k_in/256][64][256] bf16, zero-padded up to whole tiles -- so qkv's [3456, 1152]
ships as [56, 5, 16384]. `untile()` undoes that; the orientation is confirmed by where the
padding zeros sit (rows >= 3456 are whole zero tiles, k >= 1152 is the zero tail of every
kt = 4 tile).

Geometry (config.json vision_config): 27 layers, dim 1152, 16 heads x 72, MLP 4304
(gelu tanh), merge 2x2 -> 4608 -> 4608 (exact GELU) -> 2048, patch 16, temporal 2,
2304 = 48 x 48 learned positions bilinearly interpolated to the grid, 2-D RoPE over
(h, w) with 18 frequencies each (theta 1e4), LayerNorm eps 1e-6.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q4nx  # noqa: E402

DEFAULT_MODEL_DIR = Path.home() / ".flm" / "models" / "Qwen3.6-35B-A3B-NPU2"
TILE_N, TILE_K = 64, 256


def vision_config(model_dir: Path) -> dict:
    v = json.loads((model_dir / "config.json").read_text())["vision_config"]
    g = lambda k: v[f"QWEN3_6_MOE_{k}"]  # noqa: E731
    return dict(depth=g("VISION_NUM_LAYERS"), hidden=g("VISION_EMBED_DIM"), heads=g("VISION_NUM_HEADS"),
                head_dim=g("VISION_HEAD_DIM"), inter=g("VISION_MLP_INTERMEDIATE_SIZE"), out=g("VISION_OUT_HIDDEN_SIZE"),
                patch=g("PATCH_SIZE"), temporal=g("TEMPORAL_PATCH_SIZE"), merge=g("SPATIAL_MERGE_SIZE"),
                npos=g("VISION_NUM_POSITION_EMBEDDINGS"), eps=g("VISION_LAYER_NORM_EPSILON"), channels=3)


def untile(t: np.ndarray, n_out: int, k_in: int) -> np.ndarray:
    """[nt, kt, 64 * 256] (or [nt, kt, 64, 256]) -> [n_out, k_in], the padding dropped."""
    nt, kt = t.shape[0], t.shape[1]
    w = t.reshape(nt, kt, TILE_N, TILE_K).transpose(0, 2, 1, 3).reshape(nt * TILE_N, kt * TILE_K)
    assert nt * TILE_N >= n_out and kt * TILE_K >= k_in, (t.shape, n_out, k_in)
    return np.ascontiguousarray(w[:n_out, :k_in])


def load_weights(model_dir: Path, cfg: dict) -> dict:
    f = q4nx.Q4NX(str(model_dir / "vision_weight.q4nx"))
    p = "model.visual."
    H, I, O, M = cfg["hidden"], cfg["inter"], cfg["out"], cfg["merge"] ** 2
    w = {"patch_w": f.bf16(p + "patch_embed.proj.weight").reshape(H, -1), "patch_b": f.bf16(p + "patch_embed.proj.bias"),
         "pos": f.bf16(p + "pos_embed.weight"),
         "merger_ln_w": f.bf16(p + "merger.norm.weight"), "merger_ln_b": f.bf16(p + "merger.norm.bias"),
         "merger_fc1_w": untile(f.bf16(p + "merger.linear_fc1.weight"), H * M, H * M), "merger_fc1_b": f.bf16(p + "merger.linear_fc1.bias"),
         "merger_fc2_w": untile(f.bf16(p + "merger.linear_fc2.weight"), O, H * M), "merger_fc2_b": f.bf16(p + "merger.linear_fc2.bias"),
         "blocks": []}
    for i in range(cfg["depth"]):
        b = f"{p}blocks.{i}."
        w["blocks"].append({
            "ln1_w": f.bf16(b + "norm1.weight"), "ln1_b": f.bf16(b + "norm1.bias"),
            "ln2_w": f.bf16(b + "norm2.weight"), "ln2_b": f.bf16(b + "norm2.bias"),
            "qkv_w": untile(f.bf16(b + "attn.qkv.weight"), 3 * H, H), "qkv_b": f.bf16(b + "attn.qkv.bias"),
            "proj_w": untile(f.bf16(b + "attn.proj.weight"), H, H), "proj_b": f.bf16(b + "attn.proj.bias"),
            "fc1_w": untile(f.bf16(b + "mlp.linear_fc1.weight"), I, H), "fc1_b": f.bf16(b + "mlp.linear_fc1.bias"),
            "fc2_w": untile(f.bf16(b + "mlp.linear_fc2.weight"), H, I), "fc2_b": f.bf16(b + "mlp.linear_fc2.bias"),
        })
    return w


# ---- the numpy forward (fp32), written the way the C++ port will be

def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    from scipy.special import erf  # the merger's nn.GELU() is the exact one
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def position_ids(h: int, w: int, merge: int) -> np.ndarray:
    """(h, w) per patch in the merge-block-major order the patches arrive in: [n, 2]."""
    hp, wp = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    return np.stack([hp.reshape(shape).transpose(0, 2, 1, 3).ravel(), wp.reshape(shape).transpose(0, 2, 1, 3).ravel()], -1)


def bilinear_pos_embed(pos_table: np.ndarray, h: int, w: int, merge: int, side: int) -> np.ndarray:
    """The 48 x 48 table interpolated to an h x w grid, rows in merge-block-major order: [n, H]."""
    hg, wg = np.linspace(0, side - 1, h), np.linspace(0, side - 1, w)
    hf, wf = hg.astype(np.int32), wg.astype(np.int32)
    hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
    hfr, wfr = (hg - hf).astype(np.float32), (wg - wf).astype(np.float32)
    idx = [(hf[:, None] * side + wf[None, :]).ravel(), (hf[:, None] * side + wc[None, :]).ravel(),
           (hc[:, None] * side + wf[None, :]).ravel(), (hc[:, None] * side + wc[None, :]).ravel()]
    wt = [((1 - hfr)[:, None] * (1 - wfr)[None, :]).ravel(), ((1 - hfr)[:, None] * wfr[None, :]).ravel(),
          (hfr[:, None] * (1 - wfr)[None, :]).ravel(), (hfr[:, None] * wfr[None, :]).ravel()]
    hi = np.arange(h).reshape(h // merge, merge)
    wi = np.arange(w).reshape(w // merge, merge)
    reorder = (hi[:, :, None, None] * w + wi[None, None, :, :]).transpose(0, 2, 1, 3).ravel()
    out = np.zeros((h * w, pos_table.shape[1]), np.float32)
    for i in range(4):
        out += pos_table[idx[i][reorder]] * wt[i][reorder][:, None]
    return out


def rope_tables(pos: np.ndarray, head_dim: int, theta: float = 1e4):
    """cos / sin [n, head_dim] for the 2-D rotary: (h, w) x 18 frequencies, then duplicated."""
    dim = head_dim // 2                                     # 36: the rotary embedding's dim
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))    # 18
    fr = (pos[:, :, None].astype(np.float32) * inv[None, None, :]).reshape(pos.shape[0], -1)   # [n, 36]
    emb = np.concatenate([fr, fr], -1)                      # [n, 72]
    return np.cos(emb), np.sin(emb)


def apply_rope(x, cos, sin):
    """x [n, heads, hd]; rotate_half over the head dim."""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def vit_forward_np(w: dict, cfg: dict, pixels: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """pixels [n, C*T*P*P] (HF's patch order) -> [n / merge^2, out]."""
    H, NH, HD, eps, M = cfg["hidden"], cfg["heads"], cfg["head_dim"], cfg["eps"], cfg["merge"]
    n = pixels.shape[0]
    assert n == grid_h * grid_w
    x = pixels.astype(np.float32) @ w["patch_w"].T + w["patch_b"]
    x = x + bilinear_pos_embed(w["pos"], grid_h, grid_w, M, int(math.isqrt(cfg["npos"])))
    cos, sin = rope_tables(position_ids(grid_h, grid_w, M), HD)
    scale = HD ** -0.5
    for b in w["blocks"]:
        hN = layer_norm(x, b["ln1_w"], b["ln1_b"], eps)
        qkv = (hN @ b["qkv_w"].T + b["qkv_b"]).reshape(n, 3, NH, HD)
        q, k, v = apply_rope(qkv[:, 0], cos, sin), apply_rope(qkv[:, 1], cos, sin), qkv[:, 2]
        s = np.einsum("nhd,mhd->hnm", q, k) * scale             # bidirectional, one image
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        o = np.einsum("hnm,mhd->nhd", p, v).reshape(n, H)
        x = x + (o @ b["proj_w"].T + b["proj_b"])
        hN = layer_norm(x, b["ln2_w"], b["ln2_b"], eps)
        x = x + (gelu_tanh(hN @ b["fc1_w"].T + b["fc1_b"]) @ b["fc2_w"].T + b["fc2_b"])
    y = layer_norm(x, w["merger_ln_w"], w["merger_ln_b"], eps).reshape(n // (M * M), H * M * M)
    y = gelu_erf(y @ w["merger_fc1_w"].T + w["merger_fc1_b"]) @ w["merger_fc2_w"].T + w["merger_fc2_b"]
    return y


# ---- the oracle: transformers' module with the same weights

def hf_model(w: dict, cfg: dict):
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    c = Qwen3VLVisionConfig(depth=cfg["depth"], hidden_size=cfg["hidden"], num_heads=cfg["heads"], patch_size=cfg["patch"],
                            intermediate_size=cfg["inter"], out_hidden_size=cfg["out"], spatial_merge_size=cfg["merge"],
                            temporal_patch_size=cfg["temporal"], num_position_embeddings=cfg["npos"], in_channels=cfg["channels"],
                            deepstack_visual_indexes=[], hidden_act="gelu_pytorch_tanh")
    c._attn_implementation = "eager"
    m = Qwen3VLVisionModel(c).float().eval()
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    sd = {"patch_embed.proj.weight": t(w["patch_w"]).reshape(cfg["hidden"], cfg["channels"], cfg["temporal"], cfg["patch"], cfg["patch"]),
          "patch_embed.proj.bias": t(w["patch_b"]), "pos_embed.weight": t(w["pos"]),
          "merger.norm.weight": t(w["merger_ln_w"]), "merger.norm.bias": t(w["merger_ln_b"]),
          "merger.linear_fc1.weight": t(w["merger_fc1_w"]), "merger.linear_fc1.bias": t(w["merger_fc1_b"]),
          "merger.linear_fc2.weight": t(w["merger_fc2_w"]), "merger.linear_fc2.bias": t(w["merger_fc2_b"])}
    for i, b in enumerate(w["blocks"]):
        for k, v in (("norm1.weight", "ln1_w"), ("norm1.bias", "ln1_b"), ("norm2.weight", "ln2_w"), ("norm2.bias", "ln2_b"),
                     ("attn.qkv.weight", "qkv_w"), ("attn.qkv.bias", "qkv_b"), ("attn.proj.weight", "proj_w"), ("attn.proj.bias", "proj_b"),
                     ("mlp.linear_fc1.weight", "fc1_w"), ("mlp.linear_fc1.bias", "fc1_b"), ("mlp.linear_fc2.weight", "fc2_w"), ("mlp.linear_fc2.bias", "fc2_b")):
            sd[f"blocks.{i}.{k}"] = t(b[v])
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("rotary_pos_emb") for k in missing), (missing, unexpected)
    return m


def hf_forward(m, pixels: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    import torch
    with torch.no_grad():
        out = m(torch.from_numpy(pixels.astype(np.float32)), torch.tensor([[1, grid_h, grid_w]]))
    return out.pooler_output.numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--grid", type=int, nargs=2, default=(16, 16), help="patch grid h w (each a multiple of 2)")
    ap.add_argument("--fixture", default=None, help="write pixels.bin / grid.json / ref.bin here for the C++ port")
    ap.add_argument("--no-hf", action="store_true")
    a = ap.parse_args()
    md = Path(a.model_dir)
    cfg = vision_config(md)
    t0 = time.time()
    w = load_weights(md, cfg)
    print(f"weights: {cfg['depth']} blocks, un-tiled in {time.time() - t0:.1f} s")
    gh, gw = a.grid
    rng = np.random.default_rng(1)
    pixels = rng.standard_normal((gh * gw, cfg["channels"] * cfg["temporal"] * cfg["patch"] ** 2), dtype=np.float32)
    t0 = time.time()
    y_np = vit_forward_np(w, cfg, pixels, gh, gw)
    print(f"numpy forward: {gh}x{gw} patches -> {y_np.shape} in {time.time() - t0:.2f} s")
    if a.fixture:
        out = Path(a.fixture)
        out.mkdir(parents=True, exist_ok=True)
        pixels.tofile(out / "pixels.bin")
        y_np.astype(np.float32).tofile(out / "ref.bin")
        (out / "grid.json").write_text(json.dumps({"h": gh, "w": gw, "n": gh * gw, "out": int(y_np.shape[1])}))
        print(f"fixture -> {out}")
    if a.no_hf:
        return 0
    t0 = time.time()
    m = hf_model(w, cfg)
    y_hf = hf_forward(m, pixels, gh, gw)
    print(f"transformers forward in {time.time() - t0:.2f} s")
    corr = np.corrcoef(y_np.ravel().astype(np.float64), y_hf.ravel().astype(np.float64))[0, 1]
    err = np.abs(y_np - y_hf)
    print(f"numpy vs transformers: corr {corr:.8f}  max|err| {err.max():.3e}  max|ref| {np.abs(y_hf).max():.3e}  "
          f"rel {err.max() / np.abs(y_hf).max():.2e}")
    return 0 if corr > 0.99999 else 1


if __name__ == "__main__":
    sys.exit(main())
