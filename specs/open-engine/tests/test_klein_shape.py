# Traces: OPEN-DIFFUSION-SHAPE (the diffusion Shape record).
"""A diffusion family's shape, DERIVED from the checkpoint's own configs.

`klein_pipeline.Shape` is the diffusion equivalent of a text ModelSpec, and
`shape_from_config` is the equivalent of deriving a spec from config.json. The two facts
that are NOT in the configs are the VAE/transformer patch law (`patch_px`) and the distilled
step count (`steps`); the text encoder's tap layers are the pipeline's own default
(`text_encoder_out_layers=(9, 18, 27)` in diffusers' Flux2KleinPipeline, the same class for
every size -- which is why a 36-layer encoder still runs 27 layers here).
"""
from __future__ import annotations

from dataclasses import replace

import klein_pipeline as kp

KLEIN_4B_CFG = {
    "transformer": {"attention_head_dim": 128, "num_attention_heads": 24, "num_layers": 5,
                    "num_single_layers": 20, "in_channels": 128, "mlp_ratio": 3.0},
    "text_encoder": {"hidden_size": 2560, "num_hidden_layers": 36, "intermediate_size": 9728,
                     "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128},
}
KLEIN_9B_CFG = {
    "transformer": {"attention_head_dim": 128, "num_attention_heads": 32, "num_layers": 8,
                    "num_single_layers": 24, "in_channels": 128, "mlp_ratio": 3.0},
    "text_encoder": {"hidden_size": 4096, "num_hidden_layers": 36, "intermediate_size": 12288,
                     "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128},
}


def test_the_4b_shape_is_exactly_what_the_config_says():
    got = kp.shape_from_config(KLEIN_4B_CFG["transformer"], KLEIN_4B_CFG["text_encoder"],
                               family=kp.KLEIN_4B.family)
    assert got == kp.KLEIN_4B
    # the numbers the schedule is made of, named once so a change is visible here
    assert (got.H, got.MLP, got.N_DBL, got.N_SGL) == (3072, 9216, 5, 20)
    assert (got.TE_HID, got.TE_PAD, got.CTX_LD) == (2560, 3072, 8192)


def test_the_9b_shape_is_exactly_what_the_config_says():
    got = kp.shape_from_config(KLEIN_9B_CFG["transformer"], KLEIN_9B_CFG["text_encoder"],
                               family=kp.KLEIN_9B.family)
    assert got == kp.KLEIN_9B
    assert (got.H, got.MLP, got.N_DBL, got.N_SGL) == (4096, 12288, 8, 24)
    # 36-layer encoder, the SAME three taps: 27 layers run, 3*4096 rounds to 12288
    assert (got.TE_LAYERS, got.TE_PAD, got.CTX_LD) == (27, 4096, 12288)


def test_a_shape_finds_its_kernels_by_geometry():
    assert kp.family_for_shape(kp.KLEIN_4B) == "FLUX.2-klein-4B-NPU2"
    assert kp.family_for_shape(kp.KLEIN_9B) == "FLUX.2-klein-9B-NPU2"
    # a different width is a different family, and nothing serves it
    assert kp.family_for_shape(replace(kp.KLEIN_4B, H=2560, MLP=7680)) is None


def test_set_shape_switches_and_restores():
    before = (kp.H, kp.MLP, kp.N_DBL, kp.N_SGL, kp.TE_PAD, kp.MOD_N)
    previous = kp.set_shape(kp.KLEIN_9B)
    try:
        assert (kp.H, kp.MLP, kp.N_DBL, kp.N_SGL, kp.TE_PAD, kp.MOD_N) == (4096, 12288, 8, 24, 4096, 245760)
    finally:
        kp.set_shape(previous)
    assert (kp.H, kp.MLP, kp.N_DBL, kp.N_SGL, kp.TE_PAD, kp.MOD_N) == before
