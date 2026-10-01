r"""Build a diffusion transformer's kernel sets: ONE dit_gemm xclbin carrying an
instruction stream per GEMM shape the model runs, at each supported resolution, and ONE
dit_fa (attention) xclbin and ONE dit_ew (elementwise) xclbin the same way, in fa/ and ew/.

The streams encode klein's buffer layout (the chain tests, utilities/dit-chain/, run it):

    X   [T, 3072]    residual stream, text rows 0..511 then image rows (T = 512 + image)
    Zn  [T, 3072]    LayerNorm+modulate output, the block GEMMs' input
    QKV [T, 9216]    double block: text rows from txt_qkv, image rows from img_qkv
    O   [T, 3072]    double block attention output; AO [T, 3072] out-projection output
    FF  [T, 18432]   double block ff_in output with dit_gemm's SwiGLU epilogue: the
                     SwiGLU is the first 64 of every 128 columns; ff_out reads it gathered
    FU  [T, 33792]   single block: q|k|v (0..9215), then a 6144-column slot the attention
                     writes its output into 64-of-128, then the MLP tiles with the
                     epilogue (15360..); sgl_out reads [attention | SwiGLU] gathered
                     from column 9216 -- no SwiGLU pass, no concat buffer
    P   parameter vectors: a run of 2-3 modulation vectors with one vector of slack on
        each side (dit_ew.py), or the qk block (norm weights + RoPE tables)

GEMM weights are packed with pack.pack_b; ff_in and sgl_in's MLP columns first go
through pack.interleave_swiglu (sgl_in's from column 9216).

The text encoder (Qwen3-4B layers 1-27, 512 tokens) runs on the same three sets, its
hidden size 2560 zero-padded to dit_ew's 3072 row element (norms run over W = 2560):

    XT  [512, 3072]  residual stream; ZT [512, 3072] the RMSNorm output
    QT  [512, 6144]  q (32 heads) | k (8) | v (8), one GEMM: exactly two dit_ew elements
    OT  [512, 4096]  attention output; AT [512, 3072] o_proj / down_proj output (N padded)
    GT  [512, 19456] gate|up with the SwiGLU epilogue; down_proj reads it gathered

The GEMMs read XT/ZT with row stride 3072 and K = 2560 (no padding in K); o_proj and
down_proj's weights are zero-padded to N = 3072 so the padding columns stay zero.

A stream for a row range (text or image rows) is bound to XRT sub-buffers starting at
that row; every view inside a stream starts at row 0 of its binding.

    . C:\dev\mlir-aie\iron_env.ps1
    python open_kernels\export_dit_kernels.py                       # klein 4B, 512 and 1024
    python open_kernels\export_dit_kernels.py --resolutions 1024 --out <dir>

dit_gemm's core program takes its loop bounds as runtime parameters, so every stream's
final.xclbin must be the SAME static configuration -- only the instruction streams
differ, and a model's whole GEMM workload runs in one hardware context. The export
refuses the set otherwise (npu_offload/gemm_rtp's xclbin_identical_mod_uuid, the check
the Whisper and embedding sets use).

The default destination, src/xclbins/<family>/open_kernels/, is gitignored: kernel sets
are built, not checked in.

    python open_kernels\export_dit_kernels.py --out <built dir> --install src\xclbins\<family>\open_kernels

copies a built directory's runtime files only -- diffusion_r<R>.elf, the six sets as one
full ELF per resolution (compose_elf.py; the installer ships xclbins\ recursively) -- and
writes diffusion_kernels.json last: format, family, resolutions, the ELFs, their sets and
cfg kernels, the te_attn valid_len heads, and layout_hash -- the stream specs plus
WEIGHT_FORMAT, which a model directory (utilities/dit-chain/export_bundle.py) must match.
src/open_diffusion finds the set by that manifest. The sets' xclbins are not installed:
only the pyxrt runners (utilities/dit-chain/) use them, from the build directory.

Output:
    final.xclbin, insts_<stream>.bin
    dit_kernels.json   family, streams {name: {M, K, N, role}}, the packing it expects
    toolchain.json     which mlir-aie / Peano / git HEAD built it
    build/<stream>/    per-stream builds (kept; a stream whose shape.json matches is reused)
    fa/                the same for dit_fa: final.xclbin, insts_<stream>.bin, dit_fa.json
                       (streams {name: {L, heads, kv_heads, causal, valid_len, layout, role}});
                       --no-fa skips it
    ew/                dit_ew: final.xclbin, insts_<stream>.bin, dit_ew.json (streams {name:
                       spec}, dit_ew.py's stream spec); --no-ew skips it
    conv/, conv1/      the VAE decoder's convs (dit_conv, 3x3 and 1x1) and
    vew/               its elementwise ops (vae_ew): final.xclbin, insts_<stream>.bin, and
                       dit_conv.json / vae_ew.json (streams {name: spec}). The VAE's
                       attention GEMM and attention are streams of the gemm and fa sets.
                       vae_decoder.py is the schedule these streams come from; --no-vae
                       skips all of it
    diffusion_r<R>.elf every set as one full ELF per resolution, built last once all six
                       sets are (compose_elf.py; --no-elf skips it); diffusion_elf.json
                       names their kernels; elf_build/ keeps the per-set and per-stream builds

Streams for FLUX.2 [klein] 4B (hidden 3072, SwiGLU 9216, 5 double- + 20 single-stream
blocks, 512 text tokens), per resolution R with T = (R/16)^2 image tokens:
    r<R>_img_qkv  T x 3072 x 9216    double block, image stream: q|k|v fused
    r<R>_img_out  T x 3072 x 3072    attention output
    r<R>_img_ffin T x 3072 x 18432   SwiGLU gate|up
    r<R>_img_ffout T x 9216 x 3072
    r<R>_sgl_in   (T+512) x 3072 x 27648   single block: q|k|v|mlp-in fused
    r<R>_sgl_out  (T+512) x 12288 x 3072
and, independent of resolution, the text stream of the double blocks (M = 512):
    txt_qkv, txt_out, txt_ffin, txt_ffout
An edit configuration (--edits R, an R x R output from one R x R reference; klein_pipeline's
plan(R, edit=True)) has 2T image rows: its r<R>e<R>_img_* and r<R>e<R>_sgl_* streams (and
attention and dit_ew streams likewise) run at 2T and 2T + 512 rows; the per-resolution
streams (x_emb, proj_out, euler, the decoder's, the text part's) are R's own. Its VAE
encoder's streams (vae_encoder.py) join the conv, conv1, vew, gemm and fa sets.

Attention streams (dit_fa, head dim 128):
    r<R>_attn     joint attention, Q/K/V/O each token-major [T, 3072] (the standalone test)
    r<R>_attn_dbl double block: Q/K/V read in place from QKV, O -> O
    r<R>_attn_sgl single block: Q/K/V read in place from FU, O -> FU's slot, 64-of-128
    te_attn       the Qwen3-4B text encoder: 512 tokens, 32/8 heads (GQA), causal, keys
                  past the prompt masked -- valid_len is written into the stream as 512
                  and is the prompt length at run time: dit_fa.json's patch.te_attn.valid_len
                  lists the instruction words a runner sets to it (found by diffing a
                  probe build with another value)

klein_pipeline.py (the whole-image schedule) adds: the modulation GEMM for all steps
(`mod`), ctx_emb reading CTX with row stride 8192, and in ew/ `t_silu`, `r<R>_euler`
and the text-encoder taps `te_tap9/18/27`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE / "designs" / "dit_gemm" / "dit_gemm.py"
FA_DESIGN = HERE / "designs" / "dit_fa" / "dit_fa.py"
FA_TAU = 32          # dit_fa.py DF_TAU: the lazy-rescale threshold (phase7-speed.md step 0b)
EW_DESIGN = HERE / "designs" / "dit_ew" / "dit_ew.py"
CONV_DESIGN = HERE / "designs" / "dit_conv" / "dit_conv.py"
VEW_DESIGN = HERE / "designs" / "vae_ew" / "vae_ew.py"
sys.path.insert(0, str(HERE.parent / "npu_offload" / "gemm_rtp"))
sys.path.insert(0, str(HERE / "designs" / "dit_gemm"))
sys.path.insert(0, str(HERE))
import klein_pipeline  # noqa: E402  (the whole-image schedule; no IRON import)

FAMILIES = {
    "FLUX.2-klein-4B-NPU2": dict(hidden=3072, mlp=9216, text_tokens=512, patch_px=16),
}
FA_FAMILIES = {
    "FLUX.2-klein-4B-NPU2": dict(heads=24, text_tokens=512, patch_px=16, mlp=9216,
                                 te_heads=32, te_kv_heads=8),
}


def fu_width(hidden: int, mlp: int) -> int:
    """Single block FU: q|k|v, the attention-output slot (64-of-128), the MLP tiles."""
    return 3 * hidden + 2 * hidden + 2 * mlp


def configs(resolutions: list[int], edits: list[int], patch_px: int) -> list[tuple[str, int]]:
    """(configuration key, image rows of the joint sequence): each resolution's T, then each
    edit's 2T (klein_pipeline.config_key)."""
    T = lambda R: (R // patch_px) ** 2  # noqa: E731
    return [(klein_pipeline.config_key(R), T(R)) for R in resolutions] + \
        [(klein_pipeline.config_key(R, True), 2 * T(R)) for R in edits]


def klein_streams(resolutions: list[int], hidden: int, mlp: int, text_tokens: int,
                  patch_px: int, edits: list[int] = ()) -> dict[str, dict]:
    h, f, L = hidden, mlp, text_tokens
    fu = fu_width(h, f)
    ffin = {"epi": {"first_cb": 0}}                       # SwiGLU epilogue everywhere
    ffout = {"a_gather": True}                            # K = f read out of [., 2f]
    streams = {
        "txt_qkv": dict(M=L, K=h, N=3 * h, role="double.text.qkv"),
        "txt_out": dict(M=L, K=h, N=h, role="double.text.out"),
        "txt_ffin": dict(M=L, K=h, N=2 * f, layout=ffin, role="double.text.ff_in"),
        "txt_ffout": dict(M=L, K=f, N=h, layout=ffout, role="double.text.ff_out"),
    }
    for key, T in configs(resolutions, edits, patch_px):
        M = T + L
        streams.update({
            f"r{key}_img_qkv": dict(M=T, K=h, N=3 * h, role="double.image.qkv"),
            f"r{key}_img_out": dict(M=T, K=h, N=h, role="double.image.out"),
            f"r{key}_img_ffin": dict(M=T, K=h, N=2 * f, layout=ffin, role="double.image.ff_in"),
            f"r{key}_img_ffout": dict(M=T, K=f, N=h, layout=ffout, role="double.image.ff_out"),
            f"r{key}_sgl_in": dict(M=M, K=h, N=3 * h + 2 * f, role="single.qkv_mlp_in",
                                   layout={"ldc": fu, "epi": {"first_cb": 3 * h // 1024,
                                                              "gap": 2 * h}}),
            f"r{key}_sgl_out": dict(M=M, K=h + f, N=h, role="single.out",
                                    layout={"a_gather": True, "a_col": 3 * h, "lda": fu,
                                            "a_size": M * fu}),
        })
    # Step entry / exit and conditioning, zero-padded to the shape rules by packing only:
    # a K below 512 is read with row stride K as K = 512 against zero weight rows (the
    # overlapping reads multiply by zero; the A buffer needs 512 - K values of slack),
    # an N below 1024 gets zero weight columns.
    for R in resolutions:
        T = (R // patch_px) ** 2
        streams[f"r{R}_x_emb"] = dict(M=T, K=512, N=h, layout={"lda": 128, "a_size": T * 128 + 512},
                                      role="x_embedder (K 128 -> 512)")
        streams[f"r{R}_proj_out"] = dict(M=T, K=h, N=1024, role="proj_out (N 128 -> 1024)")
    streams["t_emb1"] = dict(M=512, K=512, N=h, layout={"lda": 256, "a_size": 512 * 256 + 512},
                             role="timestep_embedder.linear_1 (K 256 -> 512; M = steps, padded)")
    streams["t_emb2"] = dict(M=512, K=h, N=h, role="timestep_embedder.linear_2")
    streams.update(klein_pipeline.gemm_streams())         # ctx_emb, the modulation GEMM
    streams.update(qwen3_te_streams(L))
    return streams


# FLUX.2 [klein] 4B's text encoder: Qwen3-4B (hidden 2560, 32/8 heads of 128, MLP 9728).
TE = dict(hidden=2560, heads=32, kv_heads=8, mlp=9728, pad_hidden=3072)


def qwen3_te_streams(L: int, hidden=TE["hidden"], heads=TE["heads"], kv_heads=TE["kv_heads"],
                     mlp=TE["mlp"], pad=TE["pad_hidden"]) -> dict[str, dict]:
    q, kv = heads * 128, kv_heads * 128
    return {
        "te_qkv": dict(M=L, K=hidden, N=q + 2 * kv, layout={"lda": pad, "a_size": L * pad},
                       role="text_encoder.qkv"),
        "te_o": dict(M=L, K=q, N=pad, role="text_encoder.o"),
        "te_gu": dict(M=L, K=hidden, N=2 * mlp,
                      layout={"lda": pad, "a_size": L * pad, "epi": {"first_cb": 0}},
                      role="text_encoder.gate_up"),
        "te_down": dict(M=L, K=mlp, N=pad, layout={"a_gather": True},
                        role="text_encoder.down"),
    }


def klein_fa_streams(resolutions: list[int], heads: int, text_tokens: int, patch_px: int,
                     mlp: int, te_heads: int, te_kv_heads: int,
                     edits: list[int] = ()) -> dict[str, dict]:
    streams = {}
    h3 = 3 * heads * 128
    for key, T_img in configs(resolutions, edits, patch_px):
        T = T_img + text_tokens
        base = dict(L=T, heads=heads, kv_heads=heads, causal=0, valid_len=0)
        if "e" not in key:                                # the standalone test's layout
            streams[f"r{key}_attn"] = base | dict(layout={}, role="joint.attention")
        streams[f"r{key}_attn_dbl"] = base | dict(
            layout=dict(qkv_ld=h3, k_col=h3 // 3, v_col=2 * h3 // 3, o_ld=h3 // 3),
            role="double.attention")
        fu = h3 + 2 * (h3 // 3) + 2 * mlp
        streams[f"r{key}_attn_sgl"] = base | dict(
            layout=dict(qkv_ld=fu, k_col=h3 // 3, v_col=2 * h3 // 3, o_ld=fu, o_col=h3,
                        o_interleave=1),
            role="single.attention")
    q, kv = te_heads * 128, te_kv_heads * 128
    streams["te_attn"] = dict(L=text_tokens, heads=te_heads, kv_heads=te_kv_heads, causal=1,
                              valid_len=text_tokens,
                              layout=dict(qkv_ld=q + 2 * kv, k_col=q, v_col=q + kv, o_ld=q),
                              role="text_encoder.attention")
    return streams


EW_EL = 3072


def klein_ew_streams(resolutions: list[int], hidden: int, mlp: int, text_tokens: int,
                     patch_px: int, edits: list[int] = ()) -> dict[str, dict]:
    """dit_ew specs over the layout in the module docstring."""
    assert hidden == EW_EL, "dit_ew's row element is FLUX.2 [klein]'s hidden size"
    h, L = hidden, text_tokens
    E_mlp = mlp // EW_EL                                  # SwiGLU row = 3 elements

    def view(T, ld=h, off=0, E=1):
        return {"off": off, "ld": ld, "T": T, "E": E}

    streams = {}
    for key, T_img in configs(resolutions, edits, patch_px):
        R = klein_pipeline.parse_config(key)[0]
        rows = {"txt": (L, 0), "img": (T_img, L), "all": (T_img + L, 0)}
        if "e" in key:                                    # the text part's are R's own
            del rows["txt"]
        for part, (T, tok0) in rows.items():
            streams[f"r{key}_ln_{part}"] = {
                "op": "ln_mod", "a": view(T), "y": view(T), "p_off": EW_EL, "n_par": 2,
                "idx": {"shift": 0, "scale": 1},
                "sizes": {"X": T * h, "B": EW_EL, "P": 4 * EW_EL, "Y": T * h, "Z": EW_EL}}
            streams[f"r{key}_res_{part}"] = {
                "op": "res_ln_mod", "a": view(T), "b": view(T), "y": view(T), "z": view(T),
                "p_off": EW_EL, "n_par": 3, "idx": {"gate": 0, "shift": 1, "scale": 2},
                "sizes": {"X": T * h, "B": T * h, "P": 5 * EW_EL, "Y": T * h, "Z": T * h}}
        for part, ld in (("txt", 3 * h), ("img", 3 * h), ("sgl", fu_width(h, mlp))):
            if part not in rows and part != "sgl":
                continue
            T, tok0 = rows["all" if part == "sgl" else part]
            streams[f"r{key}_qk_{part}"] = {
                "op": "qk", "a": view(T, ld), "b": view(T, ld, h), "y": view(T, ld),
                "z": view(T, ld, h), "p_off": EW_EL, "n_par": 3, "tok0": tok0, "n_txt": L,
                "grid_w": R // patch_px, "heads": h // 128,
                "sizes": {k: T * ld for k in ("X", "B", "Y", "Z")} | {"P": 5 * EW_EL}}
    # text encoder: RMSNorm x weight over W = 2560 of the 3072 element, unit-gate residual,
    # Qwen3 q/k (fused q|k|v row = 2 elements; rotate-half RoPE)
    Lt, W, qkv = L, TE["hidden"], TE["heads"] * 128 + 2 * TE["kv_heads"] * 128
    streams["te_rms"] = {
        "op": "ln_mod", "norm": "rms", "W": W, "a": view(Lt), "y": view(Lt), "p_off": EW_EL,
        "n_par": 1, "idx": {"scale": 0},
        "sizes": {"X": Lt * h, "B": EW_EL, "P": 3 * EW_EL, "Y": Lt * h, "Z": EW_EL}}
    streams["te_res_rms"] = {
        "op": "res_ln_mod", "norm": "rms", "unit_gate": 1, "W": W, "a": view(Lt),
        "b": view(Lt), "y": view(Lt), "z": view(Lt), "p_off": EW_EL, "n_par": 1,
        "idx": {"scale": 0},
        "sizes": {"X": Lt * h, "B": Lt * h, "P": 3 * EW_EL, "Y": Lt * h, "Z": Lt * h}}
    streams["te_qk"] = {
        "op": "qk", "rope": "qwen", "a": view(Lt, qkv), "b": view(Lt, qkv, EW_EL),
        "y": view(Lt, qkv), "z": view(Lt, qkv, EW_EL), "p_off": EW_EL, "n_par": 3,
        "heads": 24, "b_q_heads": TE["heads"] - 24, "b_k_heads": TE["kv_heads"],
        "sizes": {k: Lt * qkv for k in ("X", "B", "Y", "Z")} | {"P": 5 * EW_EL}}
    streams.update(klein_pipeline.ew_streams(resolutions))   # silu, euler, text-encoder taps
    return streams


def set_streams(family: str, resolutions: list[int], vae: bool = True,
                edits: list[int] = ()) -> dict[str, dict]:
    """{set: {stream: spec}}: what main() builds, and each set's marker lists as `streams`.
    edits: the resolutions that also get an edit configuration (each must be a resolution)."""
    assert set(edits) <= set(resolutions), (edits, resolutions)
    sets = {"gemm": klein_streams(resolutions, **FAMILIES[family], edits=edits)}
    if family in FA_FAMILIES:
        sets["fa"] = klein_fa_streams(resolutions, **FA_FAMILIES[family], edits=edits)
    sets["ew"] = klein_ew_streams(resolutions, **FAMILIES[family], edits=edits)
    if vae:
        import vae_decoder  # noqa: E402
        import vae_encoder  # noqa: E402
        for coder, rs, what in ((vae_decoder, resolutions, "vae"), (vae_encoder, edits, "vae_enc")):
            if not rs:
                continue
            v = coder.stream_specs(list(rs))
            for n, sp in v.get("gemm", {}).items():
                sets["gemm"][n] = sp | {"role": f"{what}.attention.qkv"}
            for n, sp in v.get("fa", {}).items():
                sets.setdefault("fa", {})[n] = sp | {"role": f"{what}.attention"}
            for k in ("conv", "conv1", "vew"):
                if v.get(k):
                    sets.setdefault(k, {}).update(v[k])
    return sets


# Where each set lives in a kernel directory, and its marker (the last file assemble writes)
SET_DIRS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}
SET_MARKERS = {"gemm": "dit_kernels.json", "fa": "dit_fa.json", "ew": "dit_ew.json",
               "conv": "dit_conv.json", "conv1": "dit_conv.json", "vew": "vae_ew.json"}
# The installed set's manifest (src/open_diffusion finds kernels by it)
MANIFEST, MANIFEST_FORMAT = "diffusion_kernels.json", "oflm-open-diffusion-kernels-v2"
# dit_gemm's weight packing: bump when pack.pack_b's tile order changes (the packed
# weights in a model directory are only valid against it)
WEIGHT_FORMAT = "bfp16ebs8 pack_b v1"


def layout_hash(sets: dict[str, dict]) -> str:
    """What ties a model directory's schedule and packed weights to a kernel set: every
    stream's spec and the weight packing. q4nx-build writes it into the model's config.json
    and bundle.json, --install into the manifest; the engine refuses a mismatch."""
    import hashlib
    norm = json.loads(json.dumps(sets))                    # tuples -> lists, as a marker reads
    blob = json.dumps({"weights": WEIGHT_FORMAT, "sets": norm}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def install(src: Path, dst: Path, family: str) -> None:
    """Copy a built kernel directory's runtime files -- the full ELFs compose_elf.py
    assembled from the six sets (one per resolution), their description and the toolchain
    record -- to dst, then write the manifest last. The sets' xclbins stay in the build
    directory, for the pyxrt runners (utilities/dit-chain/)."""
    import compose_elf  # noqa: E402

    markers = {}
    for s, sub in SET_DIRS.items():
        m = _read_json(src / sub / SET_MARKERS[s])
        if m is None or not m.get("complete"):
            raise SystemExit(f"{src / sub / SET_MARKERS[s]}: missing or incomplete; build the set first")
        markers[s] = m
    resolutions = markers["gemm"]["resolutions"]
    edits = markers["gemm"].get("edits", [])
    keys = [klein_pipeline.config_key(R) for R in resolutions] + \
        [klein_pipeline.config_key(R, True) for R in edits]
    sets = {s: m["streams"] for s, m in markers.items()}
    want = set_streams(family, resolutions, edits=edits)
    if json.loads(json.dumps(want)) != sets:
        raise SystemExit(f"{src} was built from other stream specs than this tree's; rebuild it")
    elf_meta = _read_json(src / compose_elf.ELF_META)
    newest_set = max((src / sub / SET_MARKERS[s]).stat().st_mtime for s, sub in SET_DIRS.items())
    elfs = [src / n for n in (elf_meta or {}).get("elf", {}).values()]
    if elf_meta is None or sorted(elf_meta["elf"]) != sorted(keys) or \
            any(not e.is_file() or e.stat().st_mtime < newest_set for e in elfs):
        raise SystemExit(f"{src}: the resolutions' ELFs are missing or older than the sets; "
                         f"run compose_elf.py (or the export) again")
    (dst / MANIFEST).unlink(missing_ok=True)
    dst.mkdir(parents=True, exist_ok=True)
    # what an earlier install left (the installer ships dst recursively): v1's six set
    # directories and xclbins, and ELFs of another layout
    stale = [dst / sub for s, sub in SET_DIRS.items() if sub != "." and (dst / sub / SET_MARKERS[s]).is_file()]
    keep = {e.name for e in elfs}
    stale += [p for p in dst.iterdir() if p.is_file() and p.name not in keep and (
        p.name in ("final.xclbin", SET_MARKERS["gemm"]) or re.fullmatch(r"insts_.*\.bin|diffusion.*\.elf", p.name))]
    for p in stale:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    if stale:
        print(f"removed {len(stale)} stale kernel-set entries from {dst}")
    for f in [e.name for e in elfs] + [compose_elf.ELF_META, "toolchain.json"]:
        shutil.copyfile(src / f, dst / f)
    (dst / MANIFEST).write_text(json.dumps({
        "format": MANIFEST_FORMAT, "family": family, "resolutions": resolutions,
        "edits": edits, "layout": layout_hash(sets), "elf": elf_meta["elf"], "sets": elf_meta["sets"],
        "cfg": elf_meta["cfg"], "valid_len": elf_meta["valid_len"], "complete": True},
        indent=2) + "\n", encoding="utf-8")
    mib = sum(e.stat().st_size for e in elfs) / 2**20
    print(f"installed {len(elfs)} ELFs ({mib:.1f} MiB) -> {dst} (layout {layout_hash(sets)})")


VL_PROBE, VL_PROBE_STREAM = 77, "te_attn_vlprobe"


def valid_len_words(base: Path, probe: Path, base_value: int, probe_value: int) -> list[int]:
    """The uint32 word indices of an instruction stream that hold its valid_len: the words
    where a build with probe_value differs from the base build -- all of them, and only
    them, holding the two values (refused otherwise)."""
    import numpy as np
    a, b = np.fromfile(base, np.uint32), np.fromfile(probe, np.uint32)
    if a.size != b.size:
        raise SystemExit(f"valid_len probe: streams differ in length ({a.size} vs {b.size})")
    idx = np.nonzero(a != b)[0]
    if idx.size == 0 or (a[idx] != base_value).any() or (b[idx] != probe_value).any():
        raise SystemExit(f"valid_len probe: {idx.size} differing words are not all "
                         f"{base_value} -> {probe_value}; the stream cannot be patched")
    return [int(i) for i in idx]


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_stream(name: str, stamp: dict, env_vars: dict, design: Path, out: Path,
                 force: bool) -> Path:
    bdir = out / "build" / name
    what = " ".join(f"{k}={v}" for k, v in stamp.items())
    if not force and (bdir / "final.xclbin").is_file() and _read_json(bdir / "shape.json") == stamp:
        print(f"  {name:16s} {what}  (kept)")
        return bdir
    env = dict(os.environ, **{k: str(v) for k, v in env_vars.items()})
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "build_design.py"), str(design), str(bdir)],
                       env=env, cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0 or "BUILD_OK" not in r.stdout:
        sys.stdout.write(r.stdout[-4000:])
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"build of stream {name} failed (exit {r.returncode})")
    (bdir / "shape.json").write_text(json.dumps(stamp))
    print(f"  {name:16s} {what}  built in {time.time() - t0:.0f} s")
    return bdir


def build_many(jobs: dict[str, tuple], force: bool, workers: int = 4) -> dict[str, Path]:
    """build_stream over {name: (stamp, env_vars, design, out)}, `workers` at a time."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        futs = {n: ex.submit(build_stream, n, *j, force) for n, j in jobs.items()}
        return {n: f.result() for n, f in futs.items()}


def stream_job(kset: str, s: dict, out: Path, fa_exp_fix: int = 0) -> tuple:
    """build_many's (stamp, env, design, out) for one stream spec of a kernel set."""
    if kset == "gemm":
        return ({"M": s["M"], "K": s["K"], "N": s["N"],
                 **({"layout": s["layout"]} if "layout" in s else {})},
                {"DG_M": s["M"], "DG_K": s["K"], "DG_N": s["N"],
                 "DG_LAYOUT": json.dumps(s.get("layout", {}))}, DESIGN, out)
    if kset == "fa":
        keys = ("L", "heads", "kv_heads", "causal", "valid_len", "layout")
        lay = s.get("layout", {})
        return ({**{k: s[k] for k in keys}, "exp_fix": fa_exp_fix, "tau": FA_TAU},
                {"DF_L": s["L"], "DF_HEADS": s["heads"], "DF_KV_HEADS": s["kv_heads"],
                 "DF_CAUSAL": s["causal"], "DF_VALID_LEN": s["valid_len"],
                 "DF_EXP_FIX": fa_exp_fix, "DF_TAU": FA_TAU,
                 "DF_QKV_LD": lay.get("qkv_ld", 0), "DF_K_COL": lay.get("k_col", 0),
                 "DF_V_COL": lay.get("v_col", 0), "DF_O_LD": lay.get("o_ld", 0),
                 "DF_O_COL": lay.get("o_col", 0), "DF_O_INTERLEAVE": lay.get("o_interleave", 0)},
                FA_DESIGN, out)
    if kset == "ew":
        return (s, {"DE_SPEC": json.dumps(s)}, EW_DESIGN, out)
    if kset in ("conv", "conv1"):
        taps = {"DC_TAPS": 9 if kset == "conv" else 1}
        return (s | taps, {"DC_SPEC": json.dumps(s), **taps}, CONV_DESIGN, out)
    if kset == "vew":
        return (s, {"VE_SPEC": json.dumps(s)}, VEW_DESIGN, out)
    raise ValueError(kset)


def assemble(out: Path, dirs: dict[str, Path], marker: Path, meta: dict) -> None:
    """One xclbin for every stream (refused otherwise), their instruction streams, and the
    marker json last -- a half-written set must not look complete."""
    from export_gemm_rtp import xclbin_identical_mod_uuid  # noqa: E402  (imports IRON)
    from toolchain_provenance import write_toolchain_json  # noqa: E402

    ref_name = next(iter(dirs))
    ref = (dirs[ref_name] / "final.xclbin").read_bytes()
    for n, d in dirs.items():
        if n == ref_name:
            continue
        ok, detail = xclbin_identical_mod_uuid(ref, (d / "final.xclbin").read_bytes())
        if not ok:
            raise SystemExit(f"stream {n}'s xclbin is a different static configuration from "
                             f"{ref_name}'s ({detail}); the set would need two contexts")
    shutil.copyfile(dirs[ref_name] / "final.xclbin", out / "final.xclbin")
    for n, d in dirs.items():
        shutil.copyfile(d / "insts.bin", out / f"insts_{n}.bin")
    write_toolchain_json(out)
    marker.write_text(json.dumps(meta | {"complete": True}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", default="FLUX.2-klein-4B-NPU2", choices=sorted(FAMILIES))
    ap.add_argument("--resolutions", default="512,1024",
                    help="square output sizes; (R/16)^2 image tokens must be a multiple of 512")
    ap.add_argument("--edits", default="",
                    help="resolutions that also get an edit configuration (R x R from an "
                         "R x R reference), e.g. 512")
    ap.add_argument("--out", default=None, help="default src/xclbins/<family>/open_kernels")
    ap.add_argument("--force", action="store_true", help="rebuild streams even if kept")
    ap.add_argument("--no-fa", action="store_true", help="skip the dit_fa (attention) set")
    ap.add_argument("--no-gemm", action="store_true", help="skip the dit_gemm set")
    ap.add_argument("--no-ew", action="store_true", help="skip the dit_ew set")
    ap.add_argument("--no-vae", action="store_true", help="skip the VAE decoder's streams and sets")
    ap.add_argument("--no-elf", action="store_true",
                    help="skip assembling the sets into full ELFs (compose_elf.py)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel stream builds")
    ap.add_argument("--fa-exp-fix", action="store_true",
                    help="build dit_fa with FA_EXP_FIX (see designs/dit_fa/README.md)")
    ap.add_argument("--install", metavar="DIR", default=None,
                    help="copy --out's runtime files (no build/) to DIR with its manifest, "
                         "diffusion_kernels.json, and build nothing; e.g. "
                         "src/xclbins/<family>/open_kernels, which the installer ships")
    args = ap.parse_args()

    resolutions = [int(r) for r in args.resolutions.split(",")]
    edits = [int(r) for r in args.edits.split(",") if r]
    bad = [why for R in edits if (why := klein_pipeline.check_edit(R, R))]
    if bad or not set(edits) <= set(resolutions):
        raise SystemExit(f"--edits {args.edits}: {bad or 'each must be one of --resolutions'}")
    if args.install:
        if not args.out:
            raise SystemExit("--install needs --out: the built kernel directory")
        install(Path(args.out).resolve(), Path(args.install).resolve(), args.family)
        return 0

    from dit_gemm import check_shape  # noqa: E402  (imports IRON)

    sets = set_streams(args.family, resolutions, vae=not args.no_vae, edits=edits)
    streams = sets["gemm"]
    vae = {k: sets[k] for k in ("conv", "conv1", "vew") if k in sets}
    bad = {n: why for n, s in streams.items() if (why := check_shape(s["M"], s["K"], s["N"]))}
    if bad:
        for n, why in bad.items():
            print(f"  {n}: {why}")
        raise SystemExit("unsupported resolution(s): pad the image-token count to a multiple of 512")

    out = Path(args.out).resolve() if args.out else \
        (HERE.parent / "src" / "xclbins" / args.family / "open_kernels").resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.no_gemm:
        marker = out / "dit_kernels.json"
        marker.unlink(missing_ok=True)
        print(f"dit_gemm set for {args.family}, resolutions {resolutions}: "
              f"{len(streams)} streams -> {out}")
        dirs = build_many({n: ({"M": s["M"], "K": s["K"], "N": s["N"],
                                **({"layout": s["layout"]} if "layout" in s else {})},
                               {"DG_M": s["M"], "DG_K": s["K"], "DG_N": s["N"],
                                "DG_LAYOUT": json.dumps(s.get("layout", {}))},
                               DESIGN, out) for n, s in streams.items()}, args.force, args.jobs)
        assemble(out, dirs, marker, {
            "family": args.family,
            "kernel": "dit_gemm",
            "a": "bf16 [M, K] row-major",
            "b": "bfp16ebs8, open_kernels/designs/dit_gemm/pack.py pack_b",
            "c": "bf16 [M, N] row-major",
            "resolutions": resolutions, "edits": edits,
            "streams": streams,
        })
        print(f"OK: {len(streams)} dit_gemm streams over one xclbin -> {out}")

    if not args.no_fa and args.family in FA_FAMILIES:
        sys.path.insert(0, str(HERE / "designs" / "dit_fa"))
        from dit_fa import check_shape as fa_check  # noqa: E402

        fa_streams = sets["fa"]
        bad ={n: why for n, s in fa_streams.items()
               if (why := fa_check(s["L"], s["heads"], s["kv_heads"]))}
        if bad:
            raise SystemExit(f"unsupported attention shape(s): {bad}")
        fa_out = out / "fa"
        fa_out.mkdir(exist_ok=True)
        marker = fa_out / "dit_fa.json"
        marker.unlink(missing_ok=True)
        fix = int(args.fa_exp_fix)
        print(f"dit_fa set for {args.family}: {len(fa_streams)} streams -> {fa_out}")
        keys = ("L", "heads", "kv_heads", "causal", "valid_len", "layout")

        def fa_job(s):
            return ({**{k: s[k] for k in keys}, "exp_fix": fix, "tau": FA_TAU},
                    {"DF_L": s["L"], "DF_HEADS": s["heads"],
                     "DF_KV_HEADS": s["kv_heads"], "DF_CAUSAL": s["causal"],
                     "DF_VALID_LEN": s["valid_len"], "DF_EXP_FIX": fix, "DF_TAU": FA_TAU,
                     "DF_QKV_LD": s["layout"].get("qkv_ld", 0),
                     "DF_K_COL": s["layout"].get("k_col", 0),
                     "DF_V_COL": s["layout"].get("v_col", 0),
                     "DF_O_LD": s["layout"].get("o_ld", 0),
                     "DF_O_COL": s["layout"].get("o_col", 0),
                     "DF_O_INTERLEAVE": s["layout"].get("o_interleave", 0)},
                    FA_DESIGN, fa_out)

        jobs = {n: fa_job(s) for n, s in fa_streams.items()}
        # te_attn's valid_len is the prompt's length: an RTP write in the instruction stream.
        # A probe build with another value finds the words a runner patches per prompt.
        patch = {}
        if "te_attn" in fa_streams:
            jobs[VL_PROBE_STREAM] = fa_job(fa_streams["te_attn"] | {"valid_len": VL_PROBE})
        dirs = build_many(jobs, args.force, args.jobs)
        if VL_PROBE_STREAM in dirs:
            probe = dirs.pop(VL_PROBE_STREAM)
            patch["te_attn"] = {"valid_len": valid_len_words(
                dirs["te_attn"] / "insts.bin", probe / "insts.bin",
                fa_streams["te_attn"]["valid_len"], VL_PROBE)}
        assemble(fa_out, dirs, marker, {
            "family": args.family,
            "kernel": "dit_fa",
            "head_dim": 128,
            "q_k_v_o": "bf16 [tokens, heads*128] token-major (K/V: kv_heads*128)",
            "exp_fix": fix,
            "tau": FA_TAU,
            "resolutions": resolutions, "edits": edits,
            "streams": fa_streams,
            "patch": patch,
        })
        print(f"OK: {len(fa_streams)} dit_fa streams over one xclbin -> {fa_out}")

    if not args.no_ew:
        sys.path.insert(0, str(HERE / "designs" / "dit_ew"))
        from dit_ew import check_spec  # noqa: E402

        ew_streams = sets["ew"]
        bad = {n: why for n, s in ew_streams.items() if (why := check_spec(s))}
        if bad:
            raise SystemExit(f"unsupported dit_ew stream(s): {bad}")
        ew_out = out / "ew"
        ew_out.mkdir(exist_ok=True)
        marker = ew_out / "dit_ew.json"
        marker.unlink(missing_ok=True)
        print(f"dit_ew set for {args.family}: {len(ew_streams)} streams -> {ew_out}")
        dirs = build_many({n: (s, {"DE_SPEC": json.dumps(s)}, EW_DESIGN, ew_out)
                           for n, s in ew_streams.items()}, args.force, args.jobs)
        assemble(ew_out, dirs, marker, {
            "family": args.family,
            "kernel": "dit_ew",
            "row_element": EW_EL,
            "resolutions": resolutions, "edits": edits,
            "streams": ew_streams,
        })
        print(f"OK: {len(ew_streams)} dit_ew streams over one xclbin -> {ew_out}")

    for kset, design, env_key, extra, marker_name, kernel in (
            ("conv", CONV_DESIGN, "DC_SPEC", {"DC_TAPS": 9}, "dit_conv.json", "dit_conv"),
            ("conv1", CONV_DESIGN, "DC_SPEC", {"DC_TAPS": 1}, "dit_conv.json", "dit_conv"),
            ("vew", VEW_DESIGN, "VE_SPEC", {}, "vae_ew.json", "vae_ew")):
        specs = vae.get(kset)
        if not specs:
            continue
        kout = out / kset
        kout.mkdir(exist_ok=True)
        marker = kout / marker_name
        marker.unlink(missing_ok=True)
        print(f"{kernel} set ({kset}) for {args.family}: {len(specs)} streams -> {kout}")
        dirs = build_many({n: (sp | extra, {env_key: json.dumps(sp), **extra}, design, kout)
                           for n, sp in specs.items()}, args.force, args.jobs)
        assemble(kout, dirs, marker, {"family": args.family, "kernel": kernel, **extra,
                                      "resolutions": resolutions, "edits": edits, "streams": specs})
        print(f"OK: {len(specs)} {kernel} streams over one xclbin -> {kout}")

    # the six sets as one full ELF per resolution (one hardware context): what --install ships
    if args.no_elf:
        return 0
    missing = [s for s, sub in SET_DIRS.items()
               if not (_read_json(out / sub / SET_MARKERS[s]) or {}).get("complete")]
    if missing:
        print(f"not building the ELF: set(s) {', '.join(missing)} not built in {out}")
        return 0
    import compose_elf  # noqa: E402
    compose_elf.build_elf(out, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
