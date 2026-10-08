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

copies a built directory's runtime files only (no build/: the installer ships xclbins\
recursively) and writes diffusion_kernels.json last: format, family, resolutions, the
set directories, and layout_hash -- the stream specs plus WEIGHT_FORMAT, which a model
directory (utilities/dit-chain/export_bundle.py) must match. src/open_diffusion finds the
set by that manifest.

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
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE / "designs" / "dit_gemm" / "dit_gemm.py"
FA_DESIGN = HERE / "designs" / "dit_fa" / "dit_fa.py"
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


def klein_streams(resolutions: list[int], hidden: int, mlp: int, text_tokens: int,
                  patch_px: int) -> dict[str, dict]:
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
    for R in resolutions:
        T = (R // patch_px) ** 2
        M = T + L
        streams.update({
            f"r{R}_img_qkv": dict(M=T, K=h, N=3 * h, role="double.image.qkv"),
            f"r{R}_img_out": dict(M=T, K=h, N=h, role="double.image.out"),
            f"r{R}_img_ffin": dict(M=T, K=h, N=2 * f, layout=ffin, role="double.image.ff_in"),
            f"r{R}_img_ffout": dict(M=T, K=f, N=h, layout=ffout, role="double.image.ff_out"),
            f"r{R}_sgl_in": dict(M=M, K=h, N=3 * h + 2 * f, role="single.qkv_mlp_in",
                                 layout={"ldc": fu, "epi": {"first_cb": 3 * h // 1024,
                                                            "gap": 2 * h}}),
            f"r{R}_sgl_out": dict(M=M, K=h + f, N=h, role="single.out",
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
                     mlp: int, te_heads: int, te_kv_heads: int) -> dict[str, dict]:
    streams = {}
    h3 = 3 * heads * 128
    for R in resolutions:
        T = (R // patch_px) ** 2 + text_tokens
        base = dict(L=T, heads=heads, kv_heads=heads, causal=0, valid_len=0)
        streams[f"r{R}_attn"] = base | dict(layout={}, role="joint.attention")
        streams[f"r{R}_attn_dbl"] = base | dict(
            layout=dict(qkv_ld=h3, k_col=h3 // 3, v_col=2 * h3 // 3, o_ld=h3 // 3),
            role="double.attention")
        fu = h3 + 2 * (h3 // 3) + 2 * mlp
        streams[f"r{R}_attn_sgl"] = base | dict(
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
                     patch_px: int) -> dict[str, dict]:
    """dit_ew specs over the layout in the module docstring."""
    assert hidden == EW_EL, "dit_ew's row element is FLUX.2 [klein]'s hidden size"
    h, L = hidden, text_tokens
    E_mlp = mlp // EW_EL                                  # SwiGLU row = 3 elements

    def view(T, ld=h, off=0, E=1):
        return {"off": off, "ld": ld, "T": T, "E": E}

    streams = {}
    for R in resolutions:
        T_img = (R // patch_px) ** 2
        rows = {"txt": (L, 0), "img": (T_img, L), "all": (T_img + L, 0)}
        for part, (T, tok0) in rows.items():
            streams[f"r{R}_ln_{part}"] = {
                "op": "ln_mod", "a": view(T), "y": view(T), "p_off": EW_EL, "n_par": 2,
                "idx": {"shift": 0, "scale": 1},
                "sizes": {"X": T * h, "B": EW_EL, "P": 4 * EW_EL, "Y": T * h, "Z": EW_EL}}
            streams[f"r{R}_res_{part}"] = {
                "op": "res_ln_mod", "a": view(T), "b": view(T), "y": view(T), "z": view(T),
                "p_off": EW_EL, "n_par": 3, "idx": {"gate": 0, "shift": 1, "scale": 2},
                "sizes": {"X": T * h, "B": T * h, "P": 5 * EW_EL, "Y": T * h, "Z": T * h}}
        for part, ld in (("txt", 3 * h), ("img", 3 * h), ("sgl", fu_width(h, mlp))):
            T, tok0 = rows["all" if part == "sgl" else part]
            streams[f"r{R}_qk_{part}"] = {
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


def set_streams(family: str, resolutions: list[int], vae: bool = True) -> dict[str, dict]:
    """{set: {stream: spec}}: what main() builds, and each set's marker lists as `streams`."""
    sets = {"gemm": klein_streams(resolutions, **FAMILIES[family])}
    if family in FA_FAMILIES:
        sets["fa"] = klein_fa_streams(resolutions, **FA_FAMILIES[family])
    sets["ew"] = klein_ew_streams(resolutions, **FAMILIES[family])
    if vae:
        import vae_decoder  # noqa: E402
        v = vae_decoder.stream_specs(resolutions)
        for n, sp in v.get("gemm", {}).items():
            sets["gemm"][n] = sp | {"role": "vae.attention.qkv"}
        for n, sp in v.get("fa", {}).items():
            sets.setdefault("fa", {})[n] = sp | {"role": "vae.attention"}
        for k in ("conv", "conv1", "vew"):
            if v.get(k):
                sets[k] = v[k]
    return sets


# Where each set lives in a kernel directory, and its marker (the last file assemble writes)
SET_DIRS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}
SET_MARKERS = {"gemm": "dit_kernels.json", "fa": "dit_fa.json", "ew": "dit_ew.json",
               "conv": "dit_conv.json", "conv1": "dit_conv.json", "vew": "vae_ew.json"}
# The installed set's manifest (src/open_diffusion finds kernels by it)
MANIFEST, MANIFEST_FORMAT = "diffusion_kernels.json", "oflm-open-diffusion-kernels-v1"
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
    """Copy a built kernel directory's runtime files (each set's final.xclbin, instruction
    streams and marker -- not build/) to dst, then write the manifest last."""
    markers = {}
    for s, sub in SET_DIRS.items():
        m = _read_json(src / sub / SET_MARKERS[s])
        if m is None or not m.get("complete"):
            raise SystemExit(f"{src / sub / SET_MARKERS[s]}: missing or incomplete; build the set first")
        markers[s] = m
    resolutions = markers["gemm"]["resolutions"]
    sets = {s: m["streams"] for s, m in markers.items()}
    want = set_streams(family, resolutions)
    if json.loads(json.dumps(want)) != sets:
        raise SystemExit(f"{src} was built from other stream specs than this tree's; rebuild it")
    (dst / MANIFEST).unlink(missing_ok=True)
    n = 0
    for s, sub in SET_DIRS.items():
        d = dst / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in ["final.xclbin", SET_MARKERS[s], "toolchain.json"] + \
                 [f"insts_{st}.bin" for st in sets[s]]:
            shutil.copyfile(src / sub / f, d / f)
            n += 1
    (dst / MANIFEST).write_text(json.dumps({
        "format": MANIFEST_FORMAT, "family": family, "resolutions": resolutions,
        "layout": layout_hash(sets), "sets": SET_DIRS, "complete": True}, indent=2) + "\n",
        encoding="utf-8")
    print(f"installed {n} files -> {dst} (layout {layout_hash(sets)})")


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
    ap.add_argument("--out", default=None, help="default src/xclbins/<family>/open_kernels")
    ap.add_argument("--force", action="store_true", help="rebuild streams even if kept")
    ap.add_argument("--no-fa", action="store_true", help="skip the dit_fa (attention) set")
    ap.add_argument("--no-gemm", action="store_true", help="skip the dit_gemm set")
    ap.add_argument("--no-ew", action="store_true", help="skip the dit_ew set")
    ap.add_argument("--no-vae", action="store_true", help="skip the VAE decoder's streams and sets")
    ap.add_argument("--jobs", type=int, default=4, help="parallel stream builds")
    ap.add_argument("--fa-exp-fix", action="store_true",
                    help="build dit_fa with FA_EXP_FIX (see designs/dit_fa/README.md)")
    ap.add_argument("--install", metavar="DIR", default=None,
                    help="copy --out's runtime files (no build/) to DIR with its manifest, "
                         "diffusion_kernels.json, and build nothing; e.g. "
                         "src/xclbins/<family>/open_kernels, which the installer ships")
    args = ap.parse_args()

    resolutions = [int(r) for r in args.resolutions.split(",")]
    if args.install:
        if not args.out:
            raise SystemExit("--install needs --out: the built kernel directory")
        install(Path(args.out).resolve(), Path(args.install).resolve(), args.family)
        return 0

    from dit_gemm import check_shape  # noqa: E402  (imports IRON)

    sets = set_streams(args.family, resolutions, vae=not args.no_vae)
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
            "resolutions": resolutions,
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
            return ({**{k: s[k] for k in keys}, "exp_fix": fix},
                    {"DF_L": s["L"], "DF_HEADS": s["heads"],
                     "DF_KV_HEADS": s["kv_heads"], "DF_CAUSAL": s["causal"],
                     "DF_VALID_LEN": s["valid_len"], "DF_EXP_FIX": fix,
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
            "resolutions": resolutions,
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
            "resolutions": resolutions,
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
                                      "resolutions": resolutions, "streams": specs})
        print(f"OK: {len(specs)} {kernel} streams over one xclbin -> {kout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
