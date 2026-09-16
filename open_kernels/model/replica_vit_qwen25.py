r"""Qwen2.5-VL's vision tower in numpy fp32 -- the windowed one.

`replica_vit.py` is the other tower: Qwen3-VL's, where every block attends over the whole
image. Qwen2.5-VL splits the image into square windows of merge units and most blocks
attend only inside their own window; four of the thirty-two attend over everything. The
tower also reorders its tokens so each window is contiguous, runs the whole stack in that
order, and puts the merged rows back at the end. Everything else differs too -- RMSNorm
instead of LayerNorm, a SwiGLU MLP instead of one GELU, split q/k/v instead of a fused
projection, no learned position table. The full list is in
`.claude/plans/qwen25vl-windowed-vit.md`.

The oracle here is transformers' own `Qwen2_5_VisionTransformerPretrainedModel`, built with
random weights: transformers makes the tensors, this file reads them, and both run the same
grid. That needs no container and no download, so it is what
`specs/open-engine/tests/test_vision_vit_windowed.py` checks against.

    python open_kernels/model/replica_vit_qwen25.py [--grid 12 10] [--fixture OUT]
    python open_kernels/model/replica_vit_qwen25.py --model-dir DIR [--grid 12 10]
    python open_kernels/model/replica_vit_qwen25.py --container-size

`--fixture` also writes a synthetic `vision_weights.q4nx` in the shipped layout, which is
what `vit_test --windowed` runs the C++ port against.

`load_weights` reads a shipped `vision_weights.q4nx`. It has never been run against a real
container: the names are confirmed by the strings in `src/lib/hrx/libqwen2vl_npu.so`, but
the published container is 50 MiB larger than this tower needs and must not be loaded until
its header has been read. `.claude/plans/qwen25vl-container-size.md`.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TILE_N, TILE_K = 64, 256

# The 3B's tower, for reference. `vision_config(model_dir)` reads the real numbers.
QWEN25VL_3B = dict(depth=32, hidden=1280, heads=16, head_dim=80, inter=3420, out=2048,
                   patch=14, temporal=2, merge=2, window=112, fullatt=(7, 15, 23, 31),
                   eps=1e-6, channels=3)


# ---- geometry: the patch order, the windows, the rotary tables

def position_ids(h: int, w: int, merge: int) -> np.ndarray:
    """(h, w) per patch in the order the processor emits them: merge blocks, row-major."""
    hp, wp = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    return np.stack([hp.reshape(shape).transpose(0, 2, 1, 3).ravel(),
                     wp.reshape(shape).transpose(0, 2, 1, 3).ravel()], -1)


def window_side(window: int, merge: int, patch: int) -> int:
    """A window's side in merge units: 112 px / 2 / 14 px = 4 on every shipped config."""
    return window // merge // patch


def window_index(grid_h: int, grid_w: int, merge: int, window: int, patch: int):
    """Which merge unit goes where, and where the attention segments start.

    Returns (index, cu) with `index` a permutation of the grid's merge units, windows
    contiguous and row-major inside each, and `cu` the segment boundaries **in patches**
    (a merge unit is merge^2 patches, and every token of a unit shares its window).

    The padding is the part worth reading twice: transformers pads to a whole number of
    windows with `side - n % side`, which is `side` and not 0 when n already divides, so a
    grid that fits the windows exactly still grows a whole empty row and column of windows.
    They contribute nothing to `index` and a zero-length segment to `cu`, which the
    de-duplication at the end drops.
    """
    side = window_side(window, merge, patch)
    lh, lw = grid_h // merge, grid_w // merge
    pad_h, pad_w = side - lh % side, side - lw % side
    nwh, nww = (lh + pad_h) // side, (lw + pad_w) // side
    padded = np.full((lh + pad_h, lw + pad_w), -100, np.int64)
    padded[:lh, :lw] = np.arange(lh * lw).reshape(lh, lw)
    blocks = padded.reshape(nwh, side, nww, side).transpose(0, 2, 1, 3).reshape(nwh * nww, side * side)
    keep = blocks != -100
    index = blocks[keep]
    cu = np.concatenate([[0], keep.sum(1).cumsum() * merge * merge]).astype(np.int64)
    return index, _unique_consecutive(cu)


def _unique_consecutive(a: np.ndarray) -> np.ndarray:
    return a[np.concatenate([[True], a[1:] != a[:-1]])]


def rope_tables(pos: np.ndarray, head_dim: int, theta: float = 1e4):
    """cos / sin [n, head_dim]: (h, w) each over head_dim/4 frequencies, then duplicated.

    Qwen2.5-VL builds its rotary embedding with dim = head_dim / 2 and steps the inverse
    frequencies by two, so head_dim / 4 of them per axis -- 20 on the 3B's head_dim 80.
    Same shape of table as the Qwen3-VL tower, a different head_dim.
    """
    dim = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    fr = (pos[:, :, None].astype(np.float32) * inv[None, None, :]).reshape(pos.shape[0], -1)
    emb = np.concatenate([fr, fr], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin):
    """x [n, heads, head_dim]; rotate_half over the head dim."""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


# ---- the pieces

def rms_norm(x, w, eps):
    return x / np.sqrt((x.astype(np.float32) ** 2).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_erf(x):
    """nn.GELU(), the exact one -- the merger's activation."""
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def segment_attention(q, k, v, cu, scale):
    """Softmax attention inside each [cu[i], cu[i+1]) segment. q/k/v [n, heads, head_dim]."""
    out = np.empty_like(v)
    for a, b in zip(cu[:-1], cu[1:]):
        s = np.einsum("nhd,mhd->hnm", q[a:b], k[a:b]) * scale
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[a:b] = np.einsum("hnm,mhd->nhd", p, v[a:b])
    return out


def vit_forward_np(w: dict, cfg: dict, pixels: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """pixels [n, C*T*P*P] in the processor's patch order -> [n / merge^2, out], same order."""
    H, NH, HD, eps, M = cfg["hidden"], cfg["heads"], cfg["head_dim"], cfg["eps"], cfg["merge"]
    unit, n = M * M, pixels.shape[0]
    assert n == grid_h * grid_w, (n, grid_h, grid_w)
    fullatt = set(cfg["fullatt"])

    idx, cu_win = window_index(grid_h, grid_w, M, cfg["window"], cfg["patch"])
    cu_full = np.array([0, n], np.int64)
    # the permutation acts on whole merge units, so widen it to the tokens inside them
    tok = (idx[:, None] * unit + np.arange(unit)[None, :]).ravel()

    x = pixels.astype(np.float32) @ w["patch_w"].T          # no bias: the Conv3d has none
    x = x[tok]
    cos, sin = rope_tables(position_ids(grid_h, grid_w, M)[tok], HD)
    scale = HD ** -0.5
    for i, b in enumerate(w["blocks"]):
        hN = rms_norm(x, b["ln1_w"], eps)
        q = (hN @ b["q_w"].T + b["q_b"]).reshape(n, NH, HD)
        k = (hN @ b["k_w"].T + b["k_b"]).reshape(n, NH, HD)
        v = (hN @ b["v_w"].T + b["v_b"]).reshape(n, NH, HD)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = segment_attention(q, k, v, cu_full if i in fullatt else cu_win, scale).reshape(n, H)
        x = x + (o @ b["o_w"].T + b["o_b"])
        hN = rms_norm(x, b["ln2_w"], eps)
        g = silu(hN @ b["gate_w"].T + b["gate_b"]) * (hN @ b["up_w"].T + b["up_b"])
        x = x + (g @ b["down_w"].T + b["down_b"])
    y = rms_norm(x, w["merger_ln_w"], eps).reshape(n // unit, H * unit)
    y = gelu_erf(y @ w["merger_fc1_w"].T + w["merger_fc1_b"]) @ w["merger_fc2_w"].T + w["merger_fc2_b"]
    return y[np.argsort(idx)]                               # back to the processor's order


# ---- the oracle: transformers' own module

def _hf_config(cfg: dict):
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    c = Qwen2_5_VLVisionConfig(depth=cfg["depth"], hidden_size=cfg["hidden"], num_heads=cfg["heads"],
                              intermediate_size=cfg["inter"], out_hidden_size=cfg["out"],
                              patch_size=cfg["patch"], spatial_merge_size=cfg["merge"],
                              temporal_patch_size=cfg["temporal"], in_channels=cfg["channels"],
                              window_size=cfg["window"], fullatt_block_indexes=list(cfg["fullatt"]),
                              hidden_act="silu")
    c._attn_implementation = "eager"
    # HF's default 0.02 makes every block's contribution tiny next to the residual stream,
    # and then even a tower with the windows wrong agrees with the oracle to 2e-4. At 0.4
    # the blocks carry the output: right gives 6e-6, all-full gives corr 0.65.
    c.initializer_range = 0.4
    return c


def weights_of_hf(m, cfg: dict) -> dict:
    """The oracle's own tensors, split the way the container ships them (separate q/k/v)."""
    H = cfg["hidden"]
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    w = {"patch_w": sd["patch_embed.proj.weight"].reshape(H, -1),
         "merger_ln_w": sd["merger.ln_q.weight"],
         "merger_fc1_w": sd["merger.mlp.0.weight"], "merger_fc1_b": sd["merger.mlp.0.bias"],
         "merger_fc2_w": sd["merger.mlp.2.weight"], "merger_fc2_b": sd["merger.mlp.2.bias"],
         "blocks": []}
    for i in range(cfg["depth"]):
        p = f"blocks.{i}."
        qkv_w, qkv_b = sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]
        w["blocks"].append({
            "ln1_w": sd[p + "norm1.weight"], "ln2_w": sd[p + "norm2.weight"],
            "q_w": qkv_w[:H], "k_w": qkv_w[H:2 * H], "v_w": qkv_w[2 * H:],
            "q_b": qkv_b[:H], "k_b": qkv_b[H:2 * H], "v_b": qkv_b[2 * H:],
            "o_w": sd[p + "attn.proj.weight"], "o_b": sd[p + "attn.proj.bias"],
            "gate_w": sd[p + "mlp.gate_proj.weight"], "gate_b": sd[p + "mlp.gate_proj.bias"],
            "up_w": sd[p + "mlp.up_proj.weight"], "up_b": sd[p + "mlp.up_proj.bias"],
            "down_w": sd[p + "mlp.down_proj.weight"], "down_b": sd[p + "mlp.down_proj.bias"],
        })
    return w


def random_tower(cfg: dict, seed: int = 0):
    """A small random tower and its weights as this file wants them: (module, weights)."""
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    torch.manual_seed(seed)
    m = Qwen2_5_VisionTransformerPretrainedModel(_hf_config(cfg)).float().eval()
    with torch.no_grad():
        for p in m.state_dict().values():
            if p.dim() == 1:            # norms init to exactly 1 and biases to 0, both of
                p.normal_(1.0, 0.1)     # which hide a term that was dropped entirely
    return m, weights_of_hf(m, cfg)


def hf_forward(m, pixels: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    import torch
    with torch.no_grad():
        out = m(torch.from_numpy(pixels.astype(np.float32)), torch.tensor([[1, grid_h, grid_w]]))
    return out.pooler_output.numpy()


# ---- reading a shipped container (UNVERIFIED: no Qwen2.5-VL container has been on this box)

def vision_config(model_dir: Path) -> dict:
    """`config.json`'s vision_config, in the plain transformers keys.

    The shipped 3B keeps its source block verbatim -- `depth`, `hidden_size`, `num_heads`,
    `intermediate_size`, `out_hidden_size`, `patch_size`, `temporal_patch_size`,
    `spatial_merge_size`, `window_size`, `fullatt_block_indexes` -- unlike the Qwen3.5 and
    3.6 towers, whose containers carry OFLM's prefixed names. This read the prefixed form
    until the container arrived on 2026-09-13 and had none of those keys.
    `src/open_qwen36/vision/vit.cpp: qwen25_from_config_text` reads the same set."""
    v = json.loads((Path(model_dir) / "config.json").read_text())["vision_config"]
    missing = [k for k in ("depth", "hidden_size", "num_heads", "intermediate_size", "out_hidden_size",
                           "patch_size", "temporal_patch_size", "spatial_merge_size", "window_size",
                           "fullatt_block_indexes") if k not in v]
    if missing:
        raise RuntimeError(f"vision_config is missing {missing}; it holds {sorted(v)}")
    cfg = dict(depth=v["depth"], hidden=v["hidden_size"], heads=v["num_heads"],
               inter=v["intermediate_size"], out=v["out_hidden_size"], patch=v["patch_size"],
               temporal=v["temporal_patch_size"], merge=v["spatial_merge_size"],
               window=v["window_size"], eps=v.get("rms_norm_eps", 1e-6),
               channels=v.get("in_channels", v.get("in_chans", 3)))
    cfg["head_dim"] = cfg["hidden"] // cfg["heads"]
    cfg["fullatt"] = tuple(v["fullatt_block_indexes"])
    return cfg


def untile(t: np.ndarray, n_out: int, k_in: int) -> np.ndarray:
    """[nt, kt, 64 * 256] -> [n_out, k_in], the zero padding dropped. Same layout the
    Qwen3-VL tower ships in: `vision_mm_weight_rearrange` pads both dims to 256."""
    nt, kt = t.shape[0], t.shape[1]
    w = t.reshape(nt, kt, TILE_N, TILE_K).transpose(0, 2, 1, 3).reshape(nt * TILE_N, kt * TILE_K)
    assert nt * TILE_N >= n_out and kt * TILE_K >= k_in, (t.shape, n_out, k_in)
    return np.ascontiguousarray(w[:n_out, :k_in])


def tile(w: np.ndarray) -> np.ndarray:
    """The inverse: [n_out, k_in] -> [n/64, k/256, 64 * 256], both dims padded up to 256
    (`_multi_modal_mm_weight_rearrange` pads to max(MM_N, MM_K), not to each separately)."""
    pad = max(TILE_N, TILE_K)
    n, k = w.shape
    np_, kp = -(-n // pad) * pad, -(-k // pad) * pad
    q = np.zeros((np_, kp), w.dtype)
    q[:n, :k] = w
    return q.reshape(np_ // TILE_N, TILE_N, kp // TILE_K, TILE_K).transpose(0, 2, 1, 3).reshape(
        np_ // TILE_N, kp // TILE_K, TILE_N * TILE_K)


def tiled_shape(n_out: int, k_in: int) -> list:
    pad = max(TILE_N, TILE_K)
    return [-(-n_out // pad) * pad // TILE_N, -(-k_in // pad) * pad // TILE_K, TILE_N * TILE_K]


# ---- what the shipped container should weigh

ELEMENT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1}


def data_bytes(tensors) -> int:
    """The tensor data alone, which is all the geometry decides -- safetensors packs it with
    no gaps and no alignment."""
    return sum(ELEMENT_BYTES[d] * math.prod(s) for _, d, s in tensors)


def safetensors_bytes(tensors) -> int:
    """The size of the file safetensors would write for `tensors`, to the byte.

    8-byte header length, then the JSON header (no whitespace, space-padded so the data
    starts 8-byte aligned), then the data. The entries are written in the order the writer
    inserted them, which every vision container on disk has sorted by name -- and the order
    only shifts how many digits the offsets take, a few hundred bytes over a whole file.
    This reproduces Gemma3-4B's, Qwen3.5-0.8B/9B's and the 35B's vision_weight.q4nx exactly.
    """
    off, parts = 0, []
    for name, dtype, shape in sorted(tensors):
        nb = ELEMENT_BYTES[dtype] * math.prod(shape)
        parts.append(f'{json.dumps(name)}:{{"dtype":"{dtype}","shape":{json.dumps(shape, separators=(",", ":"))},'
                     f'"data_offsets":[{off},{off + nb}]}}')
        off += nb
    head = len(("{" + ",".join(parts) + "}").encode())
    return 8 + head + (-head % 8) + off


def container_tensors(cfg: dict) -> list:
    """Every tensor `vision_weights.q4nx` holds, as (name, dtype, shape).

    The names are the converter's (`utilities/q4nx-build/configs/qwen2vl.json`) and are
    confirmed independently by the strings in the closed `src/lib/hrx/libqwen2vl_npu.so`,
    which reads exactly this set and nothing else. The shapes follow transformers'
    `Qwen2_5_VisionTransformerPretrainedModel` with the vision_mm tiling applied to the
    nine matrices that go through that kernel.

    Plus `identity`, which is not the tower's. The shipped container holds 519 tensors: the
    518 above and one 5120 x 5120 bf16 identity matrix under that name, tiled like any
    other weight. It is what made the file 50 MiB larger than the tower accounts for, and
    the closed engine presumably multiplies by it to move data through `vision_mm` where
    the arithmetic is a copy. This reference never reads it; it is here so the size model
    reconciles and so `load_weights` does not refuse the real file over a tensor it does
    not need. Header read 2026-09-13, fixture in
    `specs/open-engine/tests/fixtures/qwen25vl_vision_header.json`.
    """
    H, I, O, U = cfg["hidden"], cfg["inter"], cfg["out"], cfg["merge"] ** 2
    p = "model.visual."
    t = [(p + "patch_embed.proj.weight", "BF16",
          [H, cfg["channels"], cfg["temporal"], cfg["patch"], cfg["patch"]]),
         (p + "merger.ln_q.weight", "BF16", [H]),        # normalises before the 2x2 concat
         (p + "merger.mlp.0.weight", "BF16", tiled_shape(H * U, H * U)),
         (p + "merger.mlp.0.bias", "BF16", [H * U]),
         (p + "merger.mlp.2.weight", "BF16", tiled_shape(O, H * U)),
         (p + "merger.mlp.2.bias", "BF16", [O]),
         ("identity", "BF16", tiled_shape(H * U, H * U))]   # not the tower's; see the docstring
    for i in range(cfg["depth"]):
        b = f"{p}{i}."
        for n in ("q", "k", "v", "o"):
            t += [(f"{b}attn.{n}_proj.weight", "BF16", tiled_shape(H, H)),
                  (f"{b}attn.{n}_proj.bias", "BF16", [H])]
        t += [(b + "mlp.gate_proj.weight", "BF16", tiled_shape(I, H)), (b + "mlp.gate_proj.bias", "BF16", [I]),
              (b + "mlp.up_proj.weight", "BF16", tiled_shape(I, H)), (b + "mlp.up_proj.bias", "BF16", [I]),
              (b + "mlp.down_proj.weight", "BF16", tiled_shape(H, I)), (b + "mlp.down_proj.bias", "BF16", [H]),
              (b + "rmsnorm1.weight", "BF16", [H]), (b + "rmsnorm2.weight", "BF16", [H])]
    return t


def qwen3vl_container_tensors(cfg: dict) -> list:
    """The same for the full-attention tower (`replica_vit.py`'s), which is only here as
    the calibration case: its containers are on disk and in the registry, so it is what
    says whether the size model above is right before it is pointed at a file nobody has."""
    H, I, O, U = cfg["hidden"], cfg["inter"], cfg["out"], cfg["merge"] ** 2
    p = "model.visual."
    t = [(p + "patch_embed.proj.weight", "BF16",
          [H, cfg["channels"], cfg["temporal"], cfg["patch"], cfg["patch"]]),
         (p + "patch_embed.proj.bias", "BF16", [H]),
         (p + "pos_embed.weight", "BF16", [cfg["npos"], H]),
         (p + "merger.norm.weight", "BF16", [H]), (p + "merger.norm.bias", "BF16", [H]),
         (p + "merger.linear_fc1.weight", "BF16", tiled_shape(H * U, H * U)),
         (p + "merger.linear_fc1.bias", "BF16", [H * U]),
         (p + "merger.linear_fc2.weight", "BF16", tiled_shape(O, H * U)),
         (p + "merger.linear_fc2.bias", "BF16", [O])]
    for i in range(cfg["depth"]):
        b = f"{p}blocks.{i}."
        t += [(b + "attn.qkv.weight", "BF16", tiled_shape(3 * H, H)), (b + "attn.qkv.bias", "BF16", [3 * H]),
              (b + "attn.proj.weight", "BF16", tiled_shape(H, H)), (b + "attn.proj.bias", "BF16", [H]),
              (b + "mlp.linear_fc1.weight", "BF16", tiled_shape(I, H)), (b + "mlp.linear_fc1.bias", "BF16", [I]),
              (b + "mlp.linear_fc2.weight", "BF16", tiled_shape(H, I)), (b + "mlp.linear_fc2.bias", "BF16", [H])]
        for n in ("norm1", "norm2"):
            t += [(f"{b}{n}.weight", "BF16", [H]), (f"{b}{n}.bias", "BF16", [H])]
    return t


def load_weights(model_dir: Path, cfg: dict) -> dict:
    import q4nx
    path = Path(model_dir) / "vision_weights.q4nx"
    want, got = safetensors_bytes(container_tensors(cfg)), path.stat().st_size
    if got != want:
        raise RuntimeError(
            f"{path} is {got} bytes where this tower accounts for {want} ({got - want:+d}). The shipped 3B "
            f"container reconciles exactly at {want}; a different size means a different tower or a "
            f"different tiling, so list the file's tensor names, dtypes and shapes before trusting this "
            f"loader. .claude/plans/qwen25vl-container-size.md")
    f = q4nx.Q4NX(str(path))
    p = "model.visual."
    H, I, O, U = cfg["hidden"], cfg["inter"], cfg["out"], cfg["merge"] ** 2
    patch = f.bf16(p + "patch_embed.proj.weight")
    assert patch.size == H * cfg["channels"] * cfg["temporal"] * cfg["patch"] ** 2, patch.shape
    w = {"patch_w": patch.reshape(H, -1),
         "merger_ln_w": f.bf16(p + "merger.ln_q.weight"),
         "merger_fc1_w": untile(f.bf16(p + "merger.mlp.0.weight"), H * U, H * U),
         "merger_fc1_b": f.bf16(p + "merger.mlp.0.bias"),
         "merger_fc2_w": untile(f.bf16(p + "merger.mlp.2.weight"), O, H * U),
         "merger_fc2_b": f.bf16(p + "merger.mlp.2.bias"), "blocks": []}
    for i in range(cfg["depth"]):
        b = f"{p}{i}."                       # the converter's q4nx names have no "blocks." segment
        lin = lambda n, o, k: untile(f.bf16(b + n + ".weight"), o, k)  # noqa: E731
        w["blocks"].append({
            "ln1_w": f.bf16(b + "rmsnorm1.weight"), "ln2_w": f.bf16(b + "rmsnorm2.weight"),
            "q_w": lin("attn.q_proj", H, H), "q_b": f.bf16(b + "attn.q_proj.bias"),
            "k_w": lin("attn.k_proj", H, H), "k_b": f.bf16(b + "attn.k_proj.bias"),
            "v_w": lin("attn.v_proj", H, H), "v_b": f.bf16(b + "attn.v_proj.bias"),
            "o_w": lin("attn.o_proj", H, H), "o_b": f.bf16(b + "attn.o_proj.bias"),
            "gate_w": lin("mlp.gate_proj", I, H), "gate_b": f.bf16(b + "mlp.gate_proj.bias"),
            "up_w": lin("mlp.up_proj", I, H), "up_b": f.bf16(b + "mlp.up_proj.bias"),
            "down_w": lin("mlp.down_proj", H, I), "down_b": f.bf16(b + "mlp.down_proj.bias"),
        })
    return w


# ---- the fixture the C++ port is checked against

def round_to_bf16(w: dict) -> dict:
    """Every weight through bf16 and back. The container stores bf16, so without this the
    C++ port would be compared against a reference that saw more precision than it did."""
    import torch
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(torch.bfloat16).float().numpy()  # noqa: E731
    out = {k: f(v) for k, v in w.items() if k != "blocks"}
    out["blocks"] = [{k: f(v) for k, v in b.items()} for b in w["blocks"]]
    return out


def write_container(out: Path, cfg: dict, w: dict) -> None:
    """A synthetic `vision_weights.q4nx` and config.json in the shipped layout: the tensor
    names the converter writes, both dims of every vision_mm matrix padded up to 256 and
    tiled. That lets `vit_test --windowed` exercise the real loader with no container on the
    box -- it does NOT check those names against a shipped file, which only reading one can.
    """
    import torch
    from safetensors.torch import save_file
    bf = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(torch.bfloat16)  # noqa: E731
    H, O = cfg["hidden"], cfg["out"]
    p = "model.visual."
    t = {p + "patch_embed.proj.weight": bf(w["patch_w"]).reshape(
             H, cfg["channels"], cfg["temporal"], cfg["patch"], cfg["patch"]),
         p + "merger.ln_q.weight": bf(w["merger_ln_w"]),
         p + "merger.mlp.0.weight": bf(tile(w["merger_fc1_w"])),
         p + "merger.mlp.0.bias": bf(w["merger_fc1_b"]),
         p + "merger.mlp.2.weight": bf(tile(w["merger_fc2_w"])),
         p + "merger.mlp.2.bias": bf(w["merger_fc2_b"])}
    for i, b in enumerate(w["blocks"]):
        q = f"{p}{i}."
        for n, k in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")):
            t[f"{q}attn.{n}_proj.weight"] = bf(tile(b[k + "_w"]))
            t[f"{q}attn.{n}_proj.bias"] = bf(b[k + "_b"])
        for n in ("gate", "up", "down"):
            t[f"{q}mlp.{n}_proj.weight"] = bf(tile(b[n + "_w"]))
            t[f"{q}mlp.{n}_proj.bias"] = bf(b[n + "_b"])
        t[q + "rmsnorm1.weight"] = bf(b["ln1_w"])
        t[q + "rmsnorm2.weight"] = bf(b["ln2_w"])
    save_file(t, str(out / "vision_weights.q4nx"))
    (out / "config.json").write_text(json.dumps({
        "vision_model_weight": "vision_weights.q4nx",
        "vision_config": {"depth": cfg["depth"], "hidden_size": H, "num_heads": cfg["heads"],
                          "intermediate_size": cfg["inter"], "out_hidden_size": O, "patch_size": cfg["patch"],
                          "temporal_patch_size": cfg["temporal"], "spatial_merge_size": cfg["merge"],
                          "in_chans": cfg["channels"], "hidden_act": "silu",
                          "window_size": cfg["window"], "fullatt_block_indexes": list(cfg["fullatt"])},
    }, indent=1))
    assert (out / "vision_weights.q4nx").stat().st_size == safetensors_bytes(container_tensors(cfg)), \
        "the size model and the writer disagree about this container"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="a shipped container; omit for a random small tower")
    ap.add_argument("--grid", type=int, nargs=2, default=(12, 10), help="patch grid h w (multiples of the merge size)")
    ap.add_argument("--fixture", default=None, help="write pixels.bin / grid.json / ref.bin here for the C++ port")
    ap.add_argument("--container-size", action="store_true",
                    help="what vision_weights.q4nx should weigh for the 3B, against the registry")
    a = ap.parse_args()
    gh, gw = a.grid
    if a.container_size:
        want = 1430158096          # src/model_info.json, qwen2.5vl-it:3b -- and the shipped file
        got = safetensors_bytes(container_tensors(QWEN25VL_3B))
        print(f"modelled {got}  registry {want}  difference {want - got}")
        return 0
    if a.model_dir:
        cfg = vision_config(Path(a.model_dir))
        t0 = time.time()
        w = load_weights(Path(a.model_dir), cfg)
        print(f"weights: {cfg['depth']} blocks, un-tiled in {time.time() - t0:.1f} s")
        m = None
    else:
        cfg = dict(QWEN25VL_3B, depth=4, hidden=64, heads=4, head_dim=16, inter=128, out=32, fullatt=(2,))
        m, w = random_tower(cfg, seed=7)
        print(f"random tower: {cfg}")
    idx, cu = window_index(gh, gw, cfg["merge"], cfg["window"], cfg["patch"])
    print(f"{gh}x{gw} patches -> {len(idx)} merge units, {len(cu) - 1} windows, boundaries {cu.tolist()}")
    pixels = np.random.default_rng(11).standard_normal(
        (gh * gw, cfg["channels"] * cfg["temporal"] * cfg["patch"] ** 2), dtype=np.float32)
    t0 = time.time()
    y = vit_forward_np(w, cfg, pixels, gh, gw)
    print(f"numpy forward -> {y.shape} in {time.time() - t0:.2f} s")
    if a.fixture:
        out = Path(a.fixture)
        out.mkdir(parents=True, exist_ok=True)
        # the container stores bf16, so the reference the port is compared against has to
        # run the rounded weights; the oracle comparison below keeps the fp32 ones
        wq = round_to_bf16(w)
        yq = vit_forward_np(wq, cfg, pixels, gh, gw)
        pixels.tofile(out / "pixels.bin")
        yq.astype(np.float32).tofile(out / "ref.bin")
        (out / "grid.json").write_text(json.dumps({"h": gh, "w": gw, "n": gh * gw, "out": int(yq.shape[1])}))
        if not a.model_dir:
            write_container(out, cfg, wq)
        print(f"fixture -> {out}  (vit_test --windowed {out} {out})")
    if m is None:
        print("no oracle for a container run: build one with transformers and the same weights")
        return 0
    y_hf = hf_forward(m, pixels, gh, gw)
    corr = np.corrcoef(y.ravel().astype(np.float64), y_hf.ravel().astype(np.float64))[0, 1]
    err = np.abs(y - y_hf).max()
    print(f"numpy vs transformers: corr {corr:.8f}  max|err| {err:.3e}  rel {err / np.abs(y_hf).max():.2e}")
    return 0 if corr > 0.99999 else 1


if __name__ == "__main__":
    sys.exit(main())
