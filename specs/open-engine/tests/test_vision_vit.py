# Traces: OPEN-VISION-VIT-REF (canonical spec: specs/open-engine/spec.md)
"""The vision tower's reference: the shipped `vision_weight.q4nx` un-tiled and run through
a numpy fp32 forward (open_kernels/model/replica_vit.py) must agree with transformers'
Qwen3VLVisionModel loaded with the same weights. The C++ port (src/open_qwen36/vision)
is checked against the same numpy forward by vit_test.exe (procedure in the spec).

Needs the model directory (1 GB vision container) and torch + transformers; skipped
where either is absent, so a plain `pytest` run stays green on a box without them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path.home() / ".flm" / "models" / "Qwen3.6-35B-A3B-NPU2"


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        import transformers.models.qwen3_vl  # noqa: F401
        return True
    except Exception:
        return False


needs_model = pytest.mark.skipif(not (MODEL_DIR / "vision_weight.q4nx").exists() or not _have_torch(),
                                 reason="needs the Qwen3.6-35B vision container and torch + transformers")


@needs_model
def test_untiled_weights_match_transformers_on_a_small_grid():
    import replica_vit as V
    cfg = V.vision_config(MODEL_DIR)
    assert (cfg["depth"], cfg["hidden"], cfg["heads"], cfg["head_dim"], cfg["out"]) == (27, 1152, 16, 72, 2048)
    w = V.load_weights(MODEL_DIR, cfg)
    # the un-tiling is right if the padding rows / columns land where the zeros are
    q = w["blocks"][0]["qkv_w"]
    assert q.shape == (3456, 1152) and np.count_nonzero(q[-1]) > 0 and np.count_nonzero(q[:, -1]) > 0
    gh, gw = 8, 8
    rng = np.random.default_rng(3)
    pixels = rng.standard_normal((gh * gw, cfg["channels"] * cfg["temporal"] * cfg["patch"] ** 2), dtype=np.float32)
    y_np = V.vit_forward_np(w, cfg, pixels, gh, gw)
    y_hf = V.hf_forward(V.hf_model(w, cfg), pixels, gh, gw)
    assert y_np.shape == (gh * gw // 4, cfg["out"])
    corr = np.corrcoef(y_np.ravel().astype(np.float64), y_hf.ravel().astype(np.float64))[0, 1]
    assert corr > 0.99999
    assert np.abs(y_np - y_hf).max() < 1e-3 * np.abs(y_hf).max()


def test_position_ids_are_merge_block_major():
    """The patch order the processor emits: 2x2 merge blocks, row-major over blocks."""
    import replica_vit as V
    p = V.position_ids(4, 4, 2)
    assert p.tolist()[:8] == [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]


def test_bilinear_pos_embed_is_exact_on_the_table_corners():
    """A 48 x 48 grid samples the table itself (weights 1 / 0); the identity, reordered."""
    import replica_vit as V
    side = 4
    table = np.arange(side * side, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32)
    out = V.bilinear_pos_embed(table, side, side, 2, side)
    p = V.position_ids(side, side, 2)
    assert np.allclose(out[:, 0], p[:, 0] * side + p[:, 1])
