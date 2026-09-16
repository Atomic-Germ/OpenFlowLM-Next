r"""Qwen3-VL's deepstack: the vision tower's extra mergers, and where their output lands.

`replica_vit.py` is the tower itself -- Qwen3-VL's full-attention encoder, validated at
corr 1.00000000 against transformers with `deepstack_visual_indexes=[]`, because the 35B
it was written for does not use deepstack. Qwen3-VL does. Three extra mergers hang off
vision blocks 5, 11 and 17 on the 4B, and their three outputs are added back into the
decoder's residual stream, at the image tokens' rows only.

Two things differ from the tower already implemented, and both are easy to get wrong:

1. **The deepstack merger normalises after the shuffle, the ordinary merger before it.**
   `Qwen3VLVisionPatchMerger(use_postshuffle_norm=True)` reshapes `[n, hidden]` to
   `[n / merge^2, hidden * merge^2]` and *then* runs one LayerNorm across the whole merged
   row; the tower's own merger normalises each patch across `hidden` and reshapes
   afterwards. Same weights count, same output shape, different numbers. `merger()` takes
   the flag; `test_vision_deepstack.py` pins that swapping it breaks the comparison, so
   the sensitivity cannot regress unnoticed.

2. **The features are added after a decoder layer runs, not before it.** transformers'
   `Qwen3VLTextModel.forward` computes `hidden_states = decoder_layer(...)` and only then,
   for `layer_idx` in `range(len(deepstack_visual_embeds))`, adds feature `layer_idx` at
   the image rows. So feature 0 lands after layer 0, feature 1 after layer 1, feature 2
   after layer 2. Folding feature 0 into the input embedding instead is a different
   computation -- it would pass through layer 0's attention and MLP first.

The forward here is fp64 throughout. The tower body in `replica_vit.vit_forward_np` is
fp32 and stays that way: it is the reference the C++ port's fixture is cut from, and
moving its numerics would invalidate that comparison. This file recomputes the body so
the deepstack taps are exact, and the two agree with each other to fp32's precision.

The oracle is transformers' own `Qwen3VLVisionModel` with random weights: transformers
builds the tensors, this file reads them, and both run the same grid. Nothing in this repo
sits on both sides, and it needs no container and no download.

    python open_kernels/model/replica_deepstack.py [--grid 8 12] [--depth 6]
    python open_kernels/model/replica_deepstack.py --model-dir DIR --indexes 5 11 17

The `--model-dir` path reads a shipped `vision_weight.q4nx`. No Qwen3-VL container has
been on this box, so it has never been run; see `.claude/plans/qwen3vl-deepstack.md`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import replica_vit as V  # noqa: E402  (position_ids and the tiling are shared)

# Qwen3-VL-4B-Instruct's tower, for reference. A container that carries a vision_config is
# read with `replica_vit.vision_config`; this one does not -- see the plan.
QWEN3VL_4B = dict(depth=24, hidden=1024, heads=16, head_dim=64, inter=4096, out=2560,
                  patch=16, temporal=2, merge=2, npos=2304, eps=1e-6, channels=3,
                  deepstack=(5, 11, 17))


# ---- fp64 primitives


def layer_norm(x, w, b, eps):
    x = np.asarray(x, np.float64)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * np.asarray(w, np.float64) + np.asarray(b, np.float64)


def gelu_tanh(x):
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    """nn.GELU()'s exact form. The mergers use it; the block MLPs use the tanh one."""
    from scipy.special import erf
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def linear(x, w, b):
    return np.asarray(x, np.float64) @ np.asarray(w, np.float64).T + np.asarray(b, np.float64)


def rope_tables(pos: np.ndarray, head_dim: int, theta: float = 1e4):
    """cos / sin [n, head_dim] in fp64: (h, w) x head_dim/4 frequencies, then duplicated."""
    dim = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    fr = (pos[:, :, None].astype(np.float64) * inv[None, None, :]).reshape(pos.shape[0], -1)
    emb = np.concatenate([fr, fr], -1)
    return np.cos(emb), np.sin(emb)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


# ---- the merger, both flavours


def merger(x: np.ndarray, m: dict, cfg: dict, postshuffle: bool) -> np.ndarray:
    """[n, hidden] -> [n / merge^2, out].

    `postshuffle` is `use_postshuffle_norm`: True for the deepstack mergers, which reshape
    into merge groups and then normalise across the whole `hidden * merge^2` row; False for
    the tower's own merger, which normalises each patch across `hidden` first. That single
    reordering is the whole difference between the two module instances.
    """
    width = cfg["hidden"] * cfg["merge"] ** 2
    x = np.asarray(x, np.float64)
    if postshuffle:
        y = layer_norm(x.reshape(-1, width), m["ln_w"], m["ln_b"], cfg["eps"])
    else:
        y = layer_norm(x, m["ln_w"], m["ln_b"], cfg["eps"]).reshape(-1, width)
    return linear(gelu_erf(linear(y, m["fc1_w"], m["fc1_b"])), m["fc2_w"], m["fc2_b"])


# ---- the tower, with the deepstack taps


def vit_forward_deepstack(w: dict, cfg: dict, pixels: np.ndarray, grid_h: int, grid_w: int) -> dict:
    """pixels [n, C*T*P*P] -> {"merged": [n/merge^2, out], "deepstack": [ ... ]}.

    The deepstack feature for index `i` is merger `j` applied to the hidden state *after*
    block `i` has run, in the order `cfg["deepstack"]` lists them.
    """
    H, NH, HD, eps, M = cfg["hidden"], cfg["heads"], cfg["head_dim"], cfg["eps"], cfg["merge"]
    n = pixels.shape[0]
    assert n == grid_h * grid_w, (n, grid_h, grid_w)
    assert grid_h % M == 0 and grid_w % M == 0, "the grid must divide into whole merge blocks"
    idx = list(cfg.get("deepstack", ()))
    assert len(idx) == len(w.get("deepstack", [])), (idx, len(w.get("deepstack", [])))

    x = linear(pixels, w["patch_w"], w["patch_b"])
    x = x + V.bilinear_pos_embed(np.asarray(w["pos"], np.float64), grid_h, grid_w, M,
                                 int(math.isqrt(cfg["npos"])))
    cos, sin = rope_tables(V.position_ids(grid_h, grid_w, M), HD)
    scale = HD ** -0.5
    feats = []
    for i, b in enumerate(w["blocks"]):
        hN = layer_norm(x, b["ln1_w"], b["ln1_b"], eps)
        qkv = linear(hN, b["qkv_w"], b["qkv_b"]).reshape(n, 3, NH, HD)
        q, k, v = apply_rope(qkv[:, 0], cos, sin), apply_rope(qkv[:, 1], cos, sin), qkv[:, 2]
        s = np.einsum("nhd,mhd->hnm", q, k) * scale        # bidirectional, one image
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        o = np.einsum("hnm,mhd->nhd", p, v).reshape(n, H)
        x = x + linear(o, b["proj_w"], b["proj_b"])
        hN = layer_norm(x, b["ln2_w"], b["ln2_b"], eps)
        x = x + linear(gelu_tanh(linear(hN, b["fc1_w"], b["fc1_b"])), b["fc2_w"], b["fc2_b"])
        if i in idx:
            feats.append(merger(x, w["deepstack"][idx.index(i)], cfg, postshuffle=True))
    return {"merged": merger(x, w["merger"], cfg, postshuffle=False), "deepstack": feats,
            "last_hidden": x}


# ---- the decoder side: where the features land


def deepstack_layer_map(num_features: int) -> list[int]:
    """Feature j is added after decoder layer j. transformers gates this as
    `layer_idx in range(len(deepstack_visual_embeds))`, so three features cover layers
    0, 1 and 2 -- not "the input embedding plus layers 1 and 2"."""
    return list(range(num_features))


def inject_deepstack(hidden: np.ndarray, visual_pos_mask: np.ndarray, feature: np.ndarray) -> np.ndarray:
    """One decoder layer's injection: add the feature to the image tokens' rows only.

    `hidden` is [seq, text_hidden] for one sequence, `visual_pos_mask` a [seq] bool with as
    many True entries as `feature` has rows, in prompt order. Text rows are untouched --
    that is what makes a text-only request after an image request unchanged.
    """
    hidden = np.asarray(hidden, np.float64).copy()
    feature = np.asarray(feature, np.float64)
    rows = np.flatnonzero(visual_pos_mask)
    if rows.size != feature.shape[0]:
        raise ValueError(f"visual_pos_mask selects {rows.size} rows but the deepstack feature "
                         f"has {feature.shape[0]} -- the merged patch count and the image "
                         "token count must be the same number")
    if hidden.shape[1] != feature.shape[1]:
        raise ValueError(f"deepstack feature width {feature.shape[1]} != decoder hidden "
                         f"{hidden.shape[1]}; the merger's out_hidden_size is the text width")
    hidden[rows] += feature
    return hidden


# ---- the oracle: transformers builds the weights, this file reads them


def hf_tower(cfg: dict, seed: int = 7):
    """A small random Qwen3VLVisionModel plus the same weights as numpy arrays.

    `initializer_range` is raised well above HF's default for the same reason
    `replica_vit_qwen25.py` does it: at 0.02 the residual stream swamps everything and a
    forward with the merger's norm placement wrong still agrees to a few parts in 1e4.
    """
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(seed)
    c = Qwen3VLVisionConfig(
        depth=cfg["depth"], hidden_size=cfg["hidden"], num_heads=cfg["heads"], patch_size=cfg["patch"],
        intermediate_size=cfg["inter"], out_hidden_size=cfg["out"], spatial_merge_size=cfg["merge"],
        temporal_patch_size=cfg["temporal"], num_position_embeddings=cfg["npos"], in_channels=cfg["channels"],
        deepstack_visual_indexes=list(cfg.get("deepstack", ())), hidden_act="gelu_pytorch_tanh",
        initializer_range=0.4)
    c._attn_implementation = "eager"
    m = Qwen3VLVisionModel(c).float().eval()
    # the LayerNorms all initialise to (1, 0), which hides a wrong norm placement
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("norm.weight") or ".norm1." in name or ".norm2." in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif name.endswith("norm.bias"):
                p.copy_(0.2 * torch.randn_like(p))
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in m.state_dict().items()}

    def lin(pre):
        return {"w": sd[pre + ".weight"], "b": sd[pre + ".bias"]}

    def mrg(pre):
        return {"ln_w": sd[pre + ".norm.weight"], "ln_b": sd[pre + ".norm.bias"],
                "fc1_w": sd[pre + ".linear_fc1.weight"], "fc1_b": sd[pre + ".linear_fc1.bias"],
                "fc2_w": sd[pre + ".linear_fc2.weight"], "fc2_b": sd[pre + ".linear_fc2.bias"]}

    w = {"patch_w": sd["patch_embed.proj.weight"].reshape(cfg["hidden"], -1),
         "patch_b": sd["patch_embed.proj.bias"], "pos": sd["pos_embed.weight"],
         "merger": mrg("merger"),
         "deepstack": [mrg(f"deepstack_merger_list.{j}") for j in range(len(cfg.get("deepstack", ())))],
         "blocks": []}
    for i in range(cfg["depth"]):
        b = f"blocks.{i}."
        w["blocks"].append({
            "ln1_w": sd[b + "norm1.weight"], "ln1_b": sd[b + "norm1.bias"],
            "ln2_w": sd[b + "norm2.weight"], "ln2_b": sd[b + "norm2.bias"],
            "qkv_w": sd[b + "attn.qkv.weight"], "qkv_b": sd[b + "attn.qkv.bias"],
            "proj_w": sd[b + "attn.proj.weight"], "proj_b": sd[b + "attn.proj.bias"],
            "fc1_w": sd[b + "mlp.linear_fc1.weight"], "fc1_b": sd[b + "mlp.linear_fc1.bias"],
            "fc2_w": sd[b + "mlp.linear_fc2.weight"], "fc2_b": sd[b + "mlp.linear_fc2.bias"],
        })
    return m, w


def hf_forward(m, pixels: np.ndarray, grid_h: int, grid_w: int) -> dict:
    import torch
    with torch.no_grad():
        out = m(torch.from_numpy(pixels.astype(np.float32)), torch.tensor([[1, grid_h, grid_w]]))
    return {"merged": out.pooler_output.numpy().astype(np.float64),
            "deepstack": [f.numpy().astype(np.float64) for f in out.deepstack_features],
            "last_hidden": out.last_hidden_state.numpy().astype(np.float64)}


# ---- what the weight file can and cannot say about the tower's shape


UNKNOWN = ("heads", "deepstack")


def geometry_from_tensors(shapes: dict, patch: int = 16, temporal: int = 2) -> dict:
    """Recover what a `vision_weight.q4nx` manifest determines about the tower's geometry.

    Qwen3-VL-4B-Instruct-NPU2 ships no `vision_config`, so the only description of its
    tower on disk is the weight file. Most of the geometry is in the tensor shapes;
    `UNKNOWN` is the part that is not, and has to come from somewhere else:

    * `heads` -- the qkv projection is `[3 * hidden, hidden]` for any head count, so the
      split is invisible. `head_dim` follows once `heads` is known.
    * `deepstack` -- the merger names give how MANY extra mergers there are, never which
      blocks they hang off.

    `patch` and `temporal` are arguments for the same reason: `patch_embed.proj.weight` is
    `[hidden, channels * temporal * patch^2]` and that product has many factorisations.
    The closed engine hardcodes 16 and 2 (`qwen3vl_npu.hpp`), which is where the defaults
    come from; `channels` is then derived and checked against 3.

    `shapes` maps tensor name -> shape, with or without the `model.visual.` prefix, and
    with linears either as `[out, in]` or in the closed engine's `[out/64][in/256][64][256]`
    tiling (which only ever rounds a dimension up, so `hidden` is read from the patch
    embed and the merger widths are checked as bounds, not equalities).
    """
    s = {k.split("model.visual.")[-1]: tuple(v) for k, v in shapes.items()}

    def need(name):
        if name not in s:
            raise ValueError(f"vision_weight: no {name}; cannot derive the tower's geometry")
        return s[name]

    rows = lambda shp: shp[0] * 64 if len(shp) > 2 else shp[0]      # noqa: E731  un-tiled row count
    cols = lambda shp: shp[1] * 256 if len(shp) > 2 else shp[1]     # noqa: E731

    hidden = need("patch_embed.proj.weight")[0]
    patch_dim = int(np.prod(need("patch_embed.proj.weight")[1:]))
    channels, rem = divmod(patch_dim, temporal * patch * patch)
    if rem or channels != 3:
        raise ValueError(f"vision_weight: patch_embed row of {patch_dim} is not "
                         f"channels x {temporal} x {patch}^2 for 3 channels")
    depth = 1 + max((int(k.split(".")[1]) for k in s if k.startswith("blocks.")), default=-1)
    if not depth:
        raise ValueError("vision_weight: no blocks.N.* tensors")
    npos = need("pos_embed.weight")[0]
    inter = rows(need("blocks.0.mlp.linear_fc1.weight"))
    merged = cols(need("merger.linear_fc1.weight"))
    merge_sq, rem = divmod(merged, hidden)
    if rem or int(math.isqrt(merge_sq)) ** 2 != merge_sq:
        raise ValueError(f"vision_weight: merger input {merged} is not hidden {hidden} "
                         "times a square merge factor")
    out = rows(need("merger.linear_fc2.weight"))
    n_deepstack = 1 + max((int(k.split(".")[1]) for k in s
                           if k.startswith("deepstack_merger_list.")), default=-1)
    cfg = dict(depth=depth, hidden=hidden, inter=inter, out=out, npos=npos,
               merge=int(math.isqrt(merge_sq)), patch=patch, temporal=temporal,
               channels=channels, eps=1e-6, n_deepstack=n_deepstack)
    # `inter` and `out` come off tiled tensors as a rounded-up bound; say so rather than
    # letting a caller treat a padded 4352 as the real 4304.
    cfg["exact"] = len(need("blocks.0.mlp.linear_fc1.weight")) == 2
    return cfg


# ---- reading a shipped container

# Qwen3-VL-4B-Instruct-NPU2 does NOT use the 35B's [nt, kt, 64, 256] tiling. Its linears
# are declared two-dimensional as [elements / 32768, 32768], and the order inside is tiles
# of 64 output rows by 512 input columns, row-major within a tile and row-major over the
# tiles. Nothing is padded: every tensor's element count is exactly out * in.
#
# That was checked, not inferred. All 315 tensors of the shipped vision_weight.q4nx were
# compared element for element against Qwen/Qwen3-VL-4B-Instruct's own safetensors: 314
# match exactly under the rule below, and the 315th (pos_embed.weight) is stored at its
# natural shape and matches directly. The bf16 values are identical, so the container is
# upstream's weights reordered, not requantised.
FLAT_TILE_N, FLAT_TILE_K = 64, 512
FLAT_ROW = FLAT_TILE_N * FLAT_TILE_K       # 32768, the container's declared row width


def is_flat_tiled(shape) -> bool:
    """Is this tensor in the [elements / 32768, 32768] form rather than its own shape?"""
    return len(shape) == 2 and shape[1] == FLAT_ROW


def untile_flat(t: np.ndarray, n_out: int, k_in: int) -> np.ndarray:
    """[elements / 32768, 32768] -> [n_out, k_in]."""
    if n_out % FLAT_TILE_N or k_in % FLAT_TILE_K:
        raise ValueError(f"vision_weight: [{n_out}, {k_in}] is not a whole number of "
                         f"{FLAT_TILE_N}x{FLAT_TILE_K} tiles")
    if t.size != n_out * k_in:
        raise ValueError(f"vision_weight: {t.size} elements for [{n_out}, {k_in}]")
    nt, kt = n_out // FLAT_TILE_N, k_in // FLAT_TILE_K
    w = t.reshape(nt, kt, FLAT_TILE_N, FLAT_TILE_K).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(w.reshape(n_out, k_in))


def geometry_from_flat(shapes: dict, patch: int = 16, temporal: int = 2) -> dict:
    """Geometry for a container in the flat-tiled form.

    A [R, 32768] shape gives only an element count, so a width has to come from somewhere.
    `hidden` does: patch_embed.proj.weight keeps its natural five-dimensional shape.
    Everything else divides out of that exactly, because nothing is padded.

    Two numbers are still not here, for the reasons geometry_from_tensors gives: the head
    count (qkv is [3 * hidden, hidden] at any split) and which blocks the deepstack mergers
    hang off (the names say how many, never which).
    """
    s = {k.split("model.visual.")[-1]: tuple(v) for k, v in shapes.items()}

    def elems(name):
        if name not in s:
            raise ValueError(f"vision_weight: no {name}; cannot derive the tower's geometry")
        return int(np.prod(s[name]))

    pe = s.get("patch_embed.proj.weight")
    if pe is None:
        raise ValueError("vision_weight: no patch_embed.proj.weight")
    hidden = pe[0]
    patch_dim = int(np.prod(pe[1:]))
    channels, rem = divmod(patch_dim, temporal * patch * patch)
    if rem or channels != 3:
        raise ValueError(f"vision_weight: patch_embed row of {patch_dim} is not "
                         f"channels x {temporal} x {patch}^2 for 3 channels")
    depth = 1 + max((int(k.split(".")[1]) for k in s if k.startswith("blocks.")), default=-1)
    if not depth:
        raise ValueError("vision_weight: no blocks.N.* tensors")

    inter, rem = divmod(elems("blocks.0.mlp.linear_fc1.weight"), hidden)
    if rem:
        raise ValueError("vision_weight: mlp.linear_fc1 is not a multiple of hidden")
    sq = elems("merger.linear_fc1.weight")
    width = int(math.isqrt(sq))
    if width * width != sq:
        raise ValueError("vision_weight: merger.linear_fc1 is not square")
    merge_sq, rem = divmod(width, hidden)
    if rem or int(math.isqrt(merge_sq)) ** 2 != merge_sq:
        raise ValueError(f"vision_weight: merger width {width} is not hidden {hidden} "
                         "times a square merge factor")
    out, rem = divmod(elems("merger.linear_fc2.weight"), width)
    if rem:
        raise ValueError("vision_weight: merger.linear_fc2 is not a multiple of the merged width")
    n_deepstack = 1 + max((int(k.split(".")[1]) for k in s
                           if k.startswith("deepstack_merger_list.")), default=-1)
    return dict(depth=depth, hidden=hidden, inter=inter, out=out,
                npos=s["pos_embed.weight"][0], merge=int(math.isqrt(merge_sq)),
                patch=patch, temporal=temporal, channels=channels, eps=1e-6,
                n_deepstack=n_deepstack, exact=True)


def read_header(path) -> dict:
    """The safetensors header of a .q4nx, without reading a weight byte."""
    import json
    import struct

    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


def load_container(model_dir, cfg: dict, file_name: str = "vision_weight.q4nx") -> dict:
    """The whole tower plus its deepstack mergers, as vit_forward_deepstack wants them.

    `cfg` needs `heads` and `deepstack` on top of what the file determines; neither is
    recoverable from the weights, see geometry_from_flat.
    """
    import json
    import struct

    path = Path(model_dir) / file_name
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    blob = np.memmap(path, dtype=np.uint8, mode="r")

    def get(name, out=None, inn=None):
        m = hdr[name]
        a, b = m["data_offsets"]
        v = blob[base + a: base + b].view(np.uint16)
        v = (v.astype(np.uint32) << 16).view(np.float32)
        if out is None:
            return np.asarray(v.reshape(m["shape"]), np.float64)
        if is_flat_tiled(m["shape"]):
            return np.asarray(untile_flat(v, out, inn), np.float64)
        return np.asarray(v.reshape(out, inn), np.float64)

    p = "model.visual."
    H, I, O = cfg["hidden"], cfg["inter"], cfg["out"]
    W = H * cfg["merge"] ** 2
    w = {
        "patch_w": get(p + "patch_embed.proj.weight").reshape(H, -1),
        "patch_b": get(p + "patch_embed.proj.bias"),
        "pos": get(p + "pos_embed.weight"),
        "merger": {
            "ln_w": get(p + "merger.norm.weight"), "ln_b": get(p + "merger.norm.bias"),
            "fc1_w": get(p + "merger.linear_fc1.weight", W, W),
            "fc1_b": get(p + "merger.linear_fc1.bias"),
            "fc2_w": get(p + "merger.linear_fc2.weight", O, W),
            "fc2_b": get(p + "merger.linear_fc2.bias"),
        },
        "blocks": [], "deepstack": [],
    }
    for i in range(cfg["depth"]):
        b = f"{p}blocks.{i}."
        w["blocks"].append({
            "ln1_w": get(b + "norm1.weight"), "ln1_b": get(b + "norm1.bias"),
            "ln2_w": get(b + "norm2.weight"), "ln2_b": get(b + "norm2.bias"),
            "qkv_w": get(b + "attn.qkv.weight", 3 * H, H), "qkv_b": get(b + "attn.qkv.bias"),
            "proj_w": get(b + "attn.proj.weight", H, H), "proj_b": get(b + "attn.proj.bias"),
            "fc1_w": get(b + "mlp.linear_fc1.weight", I, H), "fc1_b": get(b + "mlp.linear_fc1.bias"),
            "fc2_w": get(b + "mlp.linear_fc2.weight", H, I), "fc2_b": get(b + "mlp.linear_fc2.bias"),
        })
    for j in range(len(cfg["deepstack"])):
        d = f"{p}deepstack_merger_list.{j}."
        w["deepstack"].append({
            "ln_w": get(d + "norm.weight"), "ln_b": get(d + "norm.bias"),
            "fc1_w": get(d + "linear_fc1.weight", W, W), "fc1_b": get(d + "linear_fc1.bias"),
            "fc2_w": get(d + "linear_fc2.weight", O, W), "fc2_b": get(d + "linear_fc2.bias"),
        })
    return w


# ---- reading a shipped container (UNVERIFIED: no Qwen3-VL container has been on this box)


def load_deepstack_weights(model_dir: Path, cfg: dict) -> list[dict]:
    """The three extra mergers out of `vision_weight.q4nx`.

    The tensor names are the ones the closed `qwen3vl_npu.dll` carries
    (`model.visual.deepstack_merger_list.`); every linear is pre-tiled for the closed
    engine's vision_mm the same way the tower's own merger is, so `untile` applies.
    """
    import q4nx
    f = q4nx.Q4NX(str(Path(model_dir) / "vision_weight.q4nx"))
    width = cfg["hidden"] * cfg["merge"] ** 2
    out = []
    for j in range(len(cfg["deepstack"])):
        p = f"model.visual.deepstack_merger_list.{j}."
        out.append({
            "ln_w": f.bf16(p + "norm.weight").astype(np.float64),
            "ln_b": f.bf16(p + "norm.bias").astype(np.float64),
            "fc1_w": V.untile(f.bf16(p + "linear_fc1.weight"), width, width).astype(np.float64),
            "fc1_b": f.bf16(p + "linear_fc1.bias").astype(np.float64),
            "fc2_w": V.untile(f.bf16(p + "linear_fc2.weight"), cfg["out"], width).astype(np.float64),
            "fc2_b": f.bf16(p + "linear_fc2.bias").astype(np.float64),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="a shipped container; omit for a random small tower")
    ap.add_argument("--grid", type=int, nargs=2, default=(8, 12), help="patch grid h w (multiples of the merge size)")
    ap.add_argument("--depth", type=int, default=6, help="blocks in the random tower")
    ap.add_argument("--indexes", type=int, nargs="*", default=None, help="deepstack layer indexes")
    a = ap.parse_args()
    gh, gw = a.grid

    if a.model_dir:
        md = Path(a.model_dir)
        cfg = dict(QWEN3VL_4B)
        if a.indexes is not None:
            cfg["deepstack"] = tuple(a.indexes)
        w = V.load_weights(md, cfg)
        w["merger"] = {"ln_w": w.pop("merger_ln_w"), "ln_b": w.pop("merger_ln_b"),
                       "fc1_w": w.pop("merger_fc1_w"), "fc1_b": w.pop("merger_fc1_b"),
                       "fc2_w": w.pop("merger_fc2_w"), "fc2_b": w.pop("merger_fc2_b")}
        w["deepstack"] = load_deepstack_weights(md, cfg)
        m = None
    else:
        cfg = dict(QWEN3VL_4B, depth=a.depth, hidden=64, heads=4, head_dim=16, inter=128,
                   out=32, npos=64,
                   deepstack=tuple(a.indexes if a.indexes is not None else (1, a.depth // 2, a.depth - 1)))
        m, w = hf_tower(cfg)
        print(f"random tower: depth {cfg['depth']}, hidden {cfg['hidden']}, deepstack {cfg['deepstack']}")

    pixels = np.random.default_rng(11).standard_normal(
        (gh * gw, cfg["channels"] * cfg["temporal"] * cfg["patch"] ** 2), dtype=np.float32)
    got = vit_forward_deepstack(w, cfg, pixels, gh, gw)
    print(f"{gh}x{gw} patches -> merged {got['merged'].shape}, "
          f"{len(got['deepstack'])} deepstack features of {got['deepstack'][0].shape}")
    print(f"feature j is added after decoder layer j: {deepstack_layer_map(len(got['deepstack']))}")
    if m is None:
        print("no oracle for a container run: build one with transformers and the same weights")
        return 0

    want = hf_forward(m, pixels, gh, gw)
    rc = 0
    for name, g, h in [("merged", got["merged"], want["merged"])] + [
            (f"deepstack[{j}]", got["deepstack"][j], want["deepstack"][j]) for j in range(len(got["deepstack"]))]:
        corr = np.corrcoef(g.ravel(), h.ravel())[0, 1]
        err = np.abs(g - h).max()
        print(f"{name:14s} corr {corr:.8f}  max|err| {err:.3e}  rel {err / np.abs(h).max():.2e}")
        rc |= 0 if corr > 0.99999 else 1
    # The control. Norm placement cannot be got wrong silently -- a deepstack merger's
    # LayerNorm is sized for the merged row and the tower merger's for one patch, so
    # running either through the other's path raises on the shape. What can be got wrong
    # silently is the tap: whether index i means before or after block i. Taking each
    # feature one block early must visibly disagree.
    early = tuple(max(0, i - 1) for i in cfg["deepstack"])
    off = vit_forward_deepstack(w, dict(cfg, deepstack=early), pixels, gh, gw)
    worst = max(np.corrcoef(off["deepstack"][j].ravel(), want["deepstack"][j].ravel())[0, 1]
                for j in range(len(want["deepstack"])))
    print(f"tap-one-block-early control: worst corr {worst:.4f} (must be well under 0.99999)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
