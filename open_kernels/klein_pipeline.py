r"""klein_pipeline: FLUX.2 [klein] 4B text-to-image on the NPU -- the whole schedule (text
encoder, conditioning, the denoising steps, the VAE decoder), its buffer layout, weight
packing and the host setup that does not grow with the data. Shared by the kernel exporter
(export_dit_kernels.py builds the streams named here) and whatever runs it
(utilities/dit-chain/generate.py; the C++ engine replays the same ops). Design:
.claude/plans/image-diffusion-phase6-engine.md.

Per image, in dispatch order (every op on the NPU; the host only writes the inputs below,
patches one runtime parameter and reads the RGBA back):

    cond      t_emb1 (K 256 -> 512), silu, t_emb2, silu, then ONE modulation GEMM for all
              steps at once (M = steps padded to 512) whose output order is this module's
              MOD_RUNS: every run of 2-3 vectors a dit_ew op reads sits in its own 6-vector
              slot (slack, vectors, slack, pad -- 36864 bytes, 4096-aligned), so a step's
              run is a sub-buffer of MOD and no op ever needs a shuffle
    text      Qwen3-4B layers 1-27 (export_dit_kernels.py's text-encoder layout). te_attn's
              valid_len is patched to the prompt's length; the three taps (hidden states
              9/18/27) are written by a second res+RMSNorm pass (te_tap<k>) whose residual
              output goes to CTX [512, 8192] at column 2560 k instead of XT -- the 512 zero
              padding columns land in the next tap's slot (overwritten) or past 7680
    encode    an edit only: vae_encoder.plan(R), the reference in e_IN -> REFLAT, its packed
              and normalised tokens [T, 128]
    step s    x_emb (latents -> X image rows), ctx_emb (CTX -> X text rows), the first
              LayerNorm+modulate, 5 double blocks, 20 single blocks (the last one's residual
              op applies norm_out), proj_out -> V, euler LAT += dt_s V
    vae       vae_decoder.plan(R), reading LAT

An edit (plan(R, edit=True), configuration "<R>e<R>"; specs/open-diffusion/plans/edits.md)
is klein's [text | generated | reference] sequence: the reference's T tokens follow the
generated ones in X, run through every block as image rows with the generated tokens'
modulation (the double blocks' "img" part is 2T rows), and are never read back -- proj_out
and euler touch the generated rows only. Each step's x_emb writes the reference rows from
REFLAT again: the blocks overwrite them, so the result can't be kept from the last step.
dit_ew's qk op rotates an image token k >= T as reference token k - T (t = 10). Streams
whose shape depends on the sequence are named r<config>_ (r512e512_attn_sgl); the rest
are the resolution's.

Host inputs per image: XT (the 512 prompt tokens' embedding rows), LAT (the seeded noise),
te_attn's valid_len; an edit's e_IN (the reference as bf16 2 (x / 255) - 1, channels 0-2).
Per resolution: TF (the steps' sinusoidal timestep features) and DT (the Euler dt's).
Everything else is written once at load.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent

H, MLP, L_TXT, HEADS, HD = 3072, 9216, 512, 24, 128
N_DBL, N_SGL = 5, 20
EL = 3072                                    # dit_ew row element
FU = 3 * H + 2 * H + 2 * MLP                 # single block q|k|v, attention slot, MLP tiles
PATCH_PX, LAT_CH = 16, 128
STEPS = 4
ALIGN = 4096                                 # sub-buffer offsets (bytes)

TE_LAYERS, TE_HID, TE_PAD = 27, 2560, 3072
TE_Q, TE_KV, TE_MLP = 4096, 1024, 9728
TAPS = (9, 18, 27)
CTX_LD = 8192                                # CTX row stride: 3 taps x 2560 + one element's spill
PAD_ID = 151643                              # <|endoftext|>

# The modulation GEMM's output, per step row: diffusers' chunk orders are double
# (shift, scale, gate) x (msa, mlp), single (shift, scale, gate), norm_out (scale, shift).
MOD_SRC = {"img": ("double_stream_modulation_img.linear.weight",
                   ("sh_msa", "sc_msa", "ga_msa", "sh_mlp", "sc_mlp", "ga_mlp")),
           "txt": ("double_stream_modulation_txt.linear.weight",
                   ("sh_msa", "sc_msa", "ga_msa", "sh_mlp", "sc_mlp", "ga_mlp")),
           "sgl": ("single_stream_modulation.linear.weight", ("sh", "sc", "ga")),
           "out": ("norm_out.linear.weight", ("sc", "sh"))}
# (run, its vectors) -- ln runs are (shift, scale), res runs (gate, shift, scale): the
# dit_ew streams' idx maps
MOD_RUNS = [
    ("ln_txt", [("txt", "sh_msa"), ("txt", "sc_msa")]),
    ("ln_img", [("img", "sh_msa"), ("img", "sc_msa")]),
    ("res1_txt", [("txt", "ga_msa"), ("txt", "sh_mlp"), ("txt", "sc_mlp")]),
    ("res1_img", [("img", "ga_msa"), ("img", "sh_mlp"), ("img", "sc_mlp")]),
    ("res2_txt", [("txt", "ga_mlp"), ("txt", "sh_msa"), ("txt", "sc_msa")]),
    ("res2_img", [("img", "ga_mlp"), ("img", "sh_msa"), ("img", "sc_msa")]),
    ("res2l_txt", [("txt", "ga_mlp"), ("sgl", "sh"), ("sgl", "sc")]),   # into the single blocks
    ("res2l_img", [("img", "ga_mlp"), ("sgl", "sh"), ("sgl", "sc")]),
    ("sgl", [("sgl", "ga"), ("sgl", "sh"), ("sgl", "sc")]),
    ("sgl_last", [("sgl", "ga"), ("out", "sh"), ("out", "sc")]),         # norm_out
]
MOD_SLOT = 6
MOD_N = len(MOD_RUNS) * MOD_SLOT * EL        # 184320
RUN_INDEX = {name: i for i, (name, _) in enumerate(MOD_RUNS)}


def image_tokens(R: int) -> int:
    return (R // PATCH_PX) ** 2


def euler_elems(T_img: int) -> int:
    """dit_ew euler's tile24 elements (24 tokens x 128 channels), whole per core."""
    n = -(-T_img // 24)
    return n + (-n) % 16


def check_resolution(R: int) -> str | None:
    """None if R x R is supported: square, a multiple of 16 px, and (R/16)^2 image tokens a
    multiple of 512 (dit_gemm's M tile)."""
    if R <= 0 or R % PATCH_PX:
        return f"{R}: the size must be a positive multiple of {PATCH_PX} px"
    if image_tokens(R) % 512:
        return f"{R}: (R/16)^2 = {image_tokens(R)} image tokens is not a multiple of 512"
    return None


def check_edit(R: int, R_ref: int) -> str | None:
    """None if an edit of an R x R output from one R_ref x R_ref reference is supported: the
    reference is the output's size (dit_ew's qk op takes reference tokens as the image
    tokens past grid_w^2, on the same grid) and R is a supported resolution."""
    if R != R_ref:
        return (f"{R} from a {R_ref} reference: edits need the reference at the output's size "
                f"(R x R from R x R)")
    return check_resolution(R)


def config_key(R: int, edit: bool = False) -> str:
    """A configuration's name (its ELF, its streams' prefix): "512", or "512e512" for an
    edit of a 512 x 512 output from a 512 x 512 reference."""
    return f"{R}e{R}" if edit else str(R)


def parse_config(key: str) -> tuple[int, bool]:
    """config_key's inverse: (R, edit)."""
    if "e" in key:
        r, ref = key.split("e")
        why = check_edit(int(r), int(ref))
        assert why is None, why
        return int(r), True
    return int(key), False


# ---------------------------------------------------------------------------- host setup

def chat_text(prompt: str) -> str:
    """Qwen3's chat template, one user turn, generation prompt, enable_thinking=False."""
    return (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


def token_ids(tokenizer, prompt: str) -> tuple[np.ndarray, int]:
    """(ids [512] right-padded, real length); tokenizer: a `tokenizers.Tokenizer`."""
    ids = tokenizer.encode(chat_text(prompt), add_special_tokens=False).ids[:L_TXT]
    out = np.full(L_TXT, PAD_ID, np.int64)
    out[:len(ids)] = ids
    return out, len(ids)


def empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """diffusers' compute_empirical_mu (FLUX.2)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    return float(a * num_steps + (m_200 - 200.0 * a))


def sigmas(R: int, steps: int = STEPS) -> np.ndarray:
    """FlowMatchEulerDiscreteScheduler's sigmas for the klein pipeline, terminal 0
    appended (float32, exponential time shift with the empirical mu)."""
    s = np.linspace(1.0, 1 / steps, steps).astype(np.float32)
    em = math.exp(empirical_mu(image_tokens(R), steps))
    shifted = (em / (em + (1 / s - 1) ** 1.0)).astype(np.float32)
    return np.append(shifted, np.float32(0)).astype(np.float32)


def timestep_features(sig: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """TF: [512, 256] bf16 + 512 slack; row s = Timesteps(256, flip_sin_to_cos) of step s's
    timestep as the transformer sees it (bf16(bf16(bf16(1000 sigma) / 1000) * 1000))."""
    t = np.asarray(sig[:steps], np.float32) * np.float32(1000)
    t = t.astype(bfloat16).astype(np.float32)
    t = (t / np.float32(1000)).astype(bfloat16).astype(np.float32)
    t = (t * np.float32(1000)).astype(bfloat16).astype(np.float32)
    half = 128
    freqs = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half).astype(np.float32)
    ang = t[:, None] * freqs[None, :]
    feat = np.concatenate([np.cos(ang), np.sin(ang)], axis=1)
    tf = np.zeros(512 * 256 + 512, bfloat16)
    tf[:steps * 256] = feat.astype(bfloat16).reshape(-1)
    return tf


DT_SLOT = 4                                  # vectors: slack, dt, slack, pad (24576 bytes)


def dt_params(sig: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """DT: per step a dit_ew parameter run holding dt = sigma_{s+1} - sigma_s as fp32."""
    p = np.zeros(steps * DT_SLOT * EL, bfloat16)
    for s in range(steps):
        dt = np.array([sig[s + 1] - sig[s]], np.float32)
        o = (s * DT_SLOT + 1) * EL
        p[o:o + 2] = dt.view(np.uint16).view(bfloat16)
    return p


# ---------------------------------------------------------------------------- weights

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pack():
    return _load("dit_gemm_pack", HERE / "designs" / "dit_gemm" / "pack.py")


def _ew_params():
    return _load("dit_ew_make_test", HERE / "designs" / "dit_ew" / "make_test.py")


def mod_weight(get) -> np.ndarray:
    """The modulation GEMM's B [3072, MOD_N] (float32); get(name) -> [out, in] weight."""
    B = np.zeros((H, MOD_N), np.float32)
    src = {k: get(w) for k, (w, _) in MOD_SRC.items()}
    for r, (_, vecs) in enumerate(MOD_RUNS):
        for j, (s, v) in enumerate(vecs):
            i = MOD_SRC[s][1].index(v)
            c = (r * MOD_SLOT + 1 + j) * EL
            B[:, c:c + EL] = src[s][i * EL:(i + 1) * EL].T
    return B


def dit_weight_specs() -> dict[str, tuple]:
    """{packed name: (K, N, builder(get) -> float32 [K, N])}; get reads the transformer."""
    pk = _pack()
    sw = pk.interleave_swiglu

    def pad_rows(w, K):
        out = np.zeros((K, w.shape[1]), np.float32)
        out[:w.shape[0]] = w
        return out

    def pad_cols(w, N):
        out = np.zeros((w.shape[0], N), np.float32)
        out[:, :w.shape[1]] = w
        return out

    specs = {
        "t_emb1": (512, H, lambda g: pad_rows(g("time_guidance_embed.timestep_embedder.linear_1.weight").T, 512)),
        "t_emb2": (H, H, lambda g: g("time_guidance_embed.timestep_embedder.linear_2.weight").T),
        "mod": (H, MOD_N, mod_weight),
        "ctx_emb": (7680, H, lambda g: g("context_embedder.weight").T),
        "x_emb": (512, H, lambda g: pad_rows(g("x_embedder.weight").T, 512)),
        "proj_out": (H, 1024, lambda g: pad_cols(g("proj_out.weight").T, 1024)),
    }
    for b in range(N_DBL):
        p = f"transformer_blocks.{b}."
        specs |= {
            f"dbl{b}.txt_qkv": (H, 3 * H, lambda g, p=p: np.concatenate(
                [g(p + f"attn.add_{c}_proj.weight") for c in "qkv"]).T),
            f"dbl{b}.img_qkv": (H, 3 * H, lambda g, p=p: np.concatenate(
                [g(p + f"attn.to_{c}.weight") for c in "qkv"]).T),
            f"dbl{b}.txt_out": (H, H, lambda g, p=p: g(p + "attn.to_add_out.weight").T),
            f"dbl{b}.img_out": (H, H, lambda g, p=p: g(p + "attn.to_out.0.weight").T),
            f"dbl{b}.txt_ffin": (H, 2 * MLP, lambda g, p=p: sw(g(p + "ff_context.linear_in.weight").T, 0)),
            f"dbl{b}.img_ffin": (H, 2 * MLP, lambda g, p=p: sw(g(p + "ff.linear_in.weight").T, 0)),
            f"dbl{b}.txt_ffout": (MLP, H, lambda g, p=p: g(p + "ff_context.linear_out.weight").T),
            f"dbl{b}.img_ffout": (MLP, H, lambda g, p=p: g(p + "ff.linear_out.weight").T),
        }
    for j in range(N_SGL):
        p = f"single_transformer_blocks.{j}.attn."
        specs |= {
            f"sgl{j}.in": (H, 3 * H + 2 * MLP, lambda g, p=p: sw(g(p + "to_qkv_mlp_proj.weight").T, 3 * H)),
            f"sgl{j}.out": (H + MLP, H, lambda g, p=p: g(p + "to_out.weight").T),
        }
    return specs


def te_weight_specs() -> dict[str, tuple]:
    """The text encoder's GEMM weights (Qwen3-4B layers 0-26), get reads text_encoder/."""
    sw = _pack().interleave_swiglu
    specs = {}
    for l in range(TE_LAYERS):
        p = f"model.layers.{l}."

        def padded(w, K, N=TE_PAD):
            out = np.zeros((K, N), np.float32)
            out[:, :w.shape[1]] = w
            return out
        specs |= {
            f"te{l}.qkv": (TE_HID, TE_Q + 2 * TE_KV, lambda g, p=p: np.concatenate(
                [g(p + f"self_attn.{c}_proj.weight") for c in "qkv"]).T),
            f"te{l}.o": (TE_Q, TE_PAD, lambda g, p=p: padded(g(p + "self_attn.o_proj.weight").T, TE_Q)),
            f"te{l}.gu": (TE_HID, 2 * TE_MLP, lambda g, p=p: sw(np.concatenate(
                [g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]).T)),
            f"te{l}.down": (TE_MLP, TE_PAD, lambda g, p=p: padded(g(p + "mlp.down_proj.weight").T, TE_MLP)),
        }
    return specs


def pack_b_cols(b: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """pack.pack_b over column chunks (the packed order is outermost by 128-column tile, so
    chunks concatenate to the whole): bounds the int64 temporaries of a wide matrix."""
    pk = _pack()
    N = b.shape[1]
    return np.concatenate([pk.pack_b(b[:, c:c + chunk]) for c in range(0, N, chunk)])


# the per-block parameter runs dit_ew reads (a vector of slack on each side)
def param_slots() -> list[tuple[str, int]]:
    slots = []
    for b in range(N_DBL):
        slots += [(f"qk_dbl{b}_txt", 5), (f"qk_dbl{b}_img", 5)]
    slots += [(f"qk_sgl{j}", 5) for j in range(N_SGL)]
    for l in range(TE_LAYERS):
        slots += [(f"te{l}_qk", 5), (f"te{l}_post", 3)]
    slots += [(f"te{l}_in", 3) for l in range(TE_LAYERS + 1)]
    return slots


def param_table() -> tuple[dict[str, tuple[int, int]], int]:
    """{slot: (byte offset, bytes)}, total bytes."""
    table, off = {}, 0
    for name, n in param_slots():
        nb = n * EL * 2
        table[name] = (off, nb)
        off += nb + (-nb) % ALIGN
    return table, off


def fill_params(get_dit, get_te) -> np.ndarray:
    """PARAMS contents (bf16): qk norm weights + RoPE tables, RMSNorm weights."""
    ewp = _ew_params()
    table, total = param_table()
    out = np.zeros(total // 2, bfloat16)
    z = np.zeros(EL, bfloat16)

    def put(name, vecs):
        off, nb = table[name]
        arr = np.concatenate([z, vecs, z])
        assert arr.size * 2 == nb, (name, arr.size)
        out[off // 2:off // 2 + arr.size] = arr

    def norm_vec(w):
        v = np.zeros(EL, bfloat16)
        v[:w.size] = np.asarray(w, np.float32).astype(bfloat16)
        return v

    for b in range(N_DBL):
        p = f"transformer_blocks.{b}.attn."
        put(f"qk_dbl{b}_txt", ewp.qk_params(get_dit(p + "norm_added_q.weight"),
                                            get_dit(p + "norm_added_k.weight")))
        put(f"qk_dbl{b}_img", ewp.qk_params(get_dit(p + "norm_q.weight"), get_dit(p + "norm_k.weight")))
    for j in range(N_SGL):
        p = f"single_transformer_blocks.{j}.attn."
        put(f"qk_sgl{j}", ewp.qk_params(get_dit(p + "norm_q.weight"), get_dit(p + "norm_k.weight")))
    for l in range(TE_LAYERS):
        p = f"model.layers.{l}."
        put(f"te{l}_qk", ewp.qwen_qk_params(get_te(p + "self_attn.q_norm.weight"),
                                            get_te(p + "self_attn.k_norm.weight")))
        put(f"te{l}_post", norm_vec(get_te(p + "post_attention_layernorm.weight")))
    for l in range(TE_LAYERS + 1):
        put(f"te{l}_in", norm_vec(get_te(f"model.layers.{l}.input_layernorm.weight")))
    return out


# ---------------------------------------------------------------------------- schedule

@dataclass
class Plan:
    R: int
    steps: int
    buffers: dict[str, int] = field(default_factory=dict)       # name -> bytes
    ops: list[dict] = field(default_factory=list)
    params: dict[str, tuple[int, int]] = field(default_factory=dict)
    vae: object = None                                          # vae_decoder.Plan
    edit: bool = False
    enc: object = None                                          # vae_encoder.Plan (edits)
    enc_buffers: dict[str, str] = field(default_factory=dict)   # encoder buffer -> ours

    @property
    def key(self) -> str:
        return config_key(self.R, self.edit)


def vae_buffer(name: str) -> str:
    """The pipeline's name for a vae_decoder buffer (the latents are the DiT's LAT)."""
    return "LAT" if name == "LAT" else f"v_{name}"


def encoder_buffers(vp, ep) -> dict[str, str]:
    """The pipeline's name for each vae_encoder buffer: REF is REFLAT; a buffer the same
    shape as a decoder buffer is that buffer (the encoder runs before the steps and the
    decoder after them, each writing every buffer before reading it, borders aside, which
    both keep zero) -- that includes the attention's QIN with its constant ones column;
    the rest are e_<name>."""
    out, used = {}, set()
    for n, b in ep.buffers.items():
        if n == "REF":
            out[n] = "REFLAT"
            continue
        if n not in ("IN", "ZERO"):
            twin = next((d for d, db in vp.buffers.items()
                         if d != "LAT" and d not in used and db == b), None)
            if twin is not None:
                used.add(twin)
                out[n] = vae_buffer(twin)
                continue
        out[n] = f"e_{n}"
    return out


def plan(R: int, steps: int = STEPS, edit: bool = False) -> Plan:
    why = check_resolution(R)
    assert why is None, why
    import sys
    sys.path.insert(0, str(HERE))
    import vae_decoder as vd

    T_img = image_tokens(R)
    T_ref = T_img if edit else 0
    T = T_img + T_ref + L_TXT
    Tp = 24 * euler_elems(T_img)
    pl = Plan(R, steps, edit=edit)
    key = pl.key
    B = pl.buffers
    for n, nb in {
        "XT": L_TXT * TE_PAD, "ZT": L_TXT * TE_PAD, "QT": L_TXT * (TE_Q + 2 * TE_KV),
        "OT": L_TXT * TE_Q, "AT": L_TXT * TE_PAD, "GT": L_TXT * 2 * TE_MLP, "CTX": L_TXT * CTX_LD,
        "TF": 512 * 256 + 512, "TE1": 512 * H, "TE1S": 512 * H, "TEMB": 512 * H,
        "TEMBS": 512 * H, "MOD": 512 * MOD_N, "DT": steps * DT_SLOT * EL,
        "X": T * H, "Zn": T * H, "AO": T * H, "O": T * H, "QKV": T * 3 * H,
        "FF": T * 2 * MLP, "FU": T * FU, "V": Tp * 1024, "DUMMY": EL,
    }.items():
        B[n] = nb * 2
    pl.vae = vp = vd.plan(R)
    for n, b in vp.buffers.items():
        B[vae_buffer(n)] = b.nelems * 2
    B["LAT"] = max(B["LAT"], (Tp * LAT_CH + 512) * 2)
    if edit:
        import vae_encoder as ve
        pl.enc = ep = ve.plan(R)
        pl.enc_buffers = encoder_buffers(vp, ep)
        for n, b in ep.buffers.items():
            mine = pl.enc_buffers[n]
            B[mine] = max(B.get(mine, 0), b.nelems * 2)
        B["REFLAT"] = max(B["REFLAT"], (T_ref * LAT_CH + 512) * 2)   # x_emb's K = 512 reads
    pl.params, B["PARAMS"] = param_table()

    ops = pl.ops
    phase = [""]

    def whole(n):
        return ("buf", n, 0, B[n])

    def rows(n, cols, r0, count):
        return ("buf", n, r0 * cols * 2, count * cols * 2)

    def par(name):
        off, nb = pl.params[name]
        return ("buf", "PARAMS", off, nb)

    def mod(s, run):
        r = RUN_INDEX[run]
        n = len(MOD_RUNS[r][1])
        return ("buf", "MOD", (s * MOD_N + r * MOD_SLOT * EL) * 2, (n + 2) * EL * 2)

    def dt(s):
        return ("buf", "DT", s * DT_SLOT * EL * 2, 3 * EL * 2)

    D = whole("DUMMY")

    def op(kset, stream, args, what):
        ops.append({"set": kset, "stream": stream, "args": list(args), "what": what,
                    "phase": phase[0]})

    def gemm(stream, a, w, c, what=None):
        op("gemm", stream, [a, ("w", w), c], what or w)

    def ew(stream, a, b, p, y, z, what):
        op("ew", stream, [a, b or D, p or D, y, z or D], what)

    # -- conditioning: all steps' modulation in one batch
    phase[0] = "cond"
    gemm("t_emb1", whole("TF"), "t_emb1", whole("TE1"))
    ew("t_silu", whole("TE1"), None, None, whole("TE1S"), None, "silu")
    gemm("t_emb2", whole("TE1S"), "t_emb2", whole("TEMB"))
    ew("t_silu", whole("TEMB"), None, None, whole("TEMBS"), None, "silu")
    gemm("mod", whole("TEMBS"), "mod", whole("MOD"), "modulation (all steps)")

    # -- text encoder
    phase[0] = "text"
    XT, ZT, QT, OT, AT, GT = (whole(n) for n in ("XT", "ZT", "QT", "OT", "AT", "GT"))
    ew("te_rms", XT, None, par("te0_in"), ZT, None, "te0 input_layernorm")
    for l in range(TE_LAYERS):
        gemm("te_qkv", ZT, f"te{l}.qkv", QT)
        ew("te_qk", QT, QT, par(f"te{l}_qk"), QT, QT, f"te{l} q/k norm + rope")
        op("fa", "te_attn", [QT, QT, QT, OT], f"te{l} attention")
        gemm("te_o", OT, f"te{l}.o", AT)
        ew("te_res_rms", XT, AT, par(f"te{l}_post"), ZT, XT, f"te{l} residual + post norm")
        gemm("te_gu", ZT, f"te{l}.gu", GT)
        gemm("te_down", GT, f"te{l}.down", AT)
        if l + 1 in TAPS:
            ew(f"te_tap{l + 1}", XT, AT, par(f"te{l + 1}_in"), ZT, whole("CTX"),
               f"hidden state {l + 1} -> CTX")
        if l + 1 < TE_LAYERS:
            ew("te_res_rms", XT, AT, par(f"te{l + 1}_in"), ZT, XT, f"te{l} residual + next norm")

    # -- an edit's reference: the VAE encoder
    if edit:
        phase[0] = "encode"
        for o in pl.enc.ops:
            args = []
            for kind, k in o["args"]:
                args.append(whole(pl.enc_buffers[k]) if kind == "buf" else (f"venc_{kind}", k))
            op(o["set"], o["stream"], args, o["what"])

    # -- denoising steps
    parts = {"txt": (0, L_TXT), "img": (L_TXT, T_img + T_ref)}

    def gs(part, name):
        return f"{'txt' if part == 'txt' else f'r{key}_img'}_{name}"

    def es(part, name):                     # dit_ew streams: the text part's are per R
        return f"r{R if part == 'txt' else key}_{name}_{part}"

    for s in range(steps):
        phase[0] = f"step{s}"
        gemm(f"r{R}_x_emb", whole("LAT"), "x_emb", rows("X", H, L_TXT, T_img))
        if edit:
            gemm(f"r{R}_x_emb", whole("REFLAT"), "x_emb", rows("X", H, L_TXT + T_img, T_ref),
                 "x_emb (reference)")
        gemm("ctx_emb", whole("CTX"), "ctx_emb", rows("X", H, 0, L_TXT))
        for part, (r0, n) in parts.items():
            ew(es(part, "ln"), rows("X", H, r0, n), None, mod(s, f"ln_{part}"),
               rows("Zn", H, r0, n), None, f"ln {part}")
        for b in range(N_DBL):
            last = b == N_DBL - 1
            for part, (r0, n) in parts.items():
                gemm(gs(part, "qkv"), rows("Zn", H, r0, n), f"dbl{b}.{part}_qkv",
                     rows("QKV", 3 * H, r0, n))
            for part, (r0, n) in parts.items():
                q = rows("QKV", 3 * H, r0, n)
                ew(es(part, "qk"), q, q, par(f"qk_dbl{b}_{part}"), q, q, f"dbl{b} qk {part}")
            op("fa", f"r{key}_attn_dbl", [whole("QKV")] * 3 + [whole("O")], f"dbl{b} attention")
            for part, (r0, n) in parts.items():
                gemm(gs(part, "out"), rows("O", H, r0, n), f"dbl{b}.{part}_out", rows("AO", H, r0, n))
            for part, (r0, n) in parts.items():
                x = rows("X", H, r0, n)
                ew(es(part, "res"), x, rows("AO", H, r0, n), mod(s, f"res1_{part}"),
                   rows("Zn", H, r0, n), x, f"dbl{b} res1 {part}")
            for part, (r0, n) in parts.items():
                gemm(gs(part, "ffin"), rows("Zn", H, r0, n), f"dbl{b}.{part}_ffin",
                     rows("FF", 2 * MLP, r0, n))
            for part, (r0, n) in parts.items():
                gemm(gs(part, "ffout"), rows("FF", 2 * MLP, r0, n), f"dbl{b}.{part}_ffout",
                     rows("AO", H, r0, n))
            for part, (r0, n) in parts.items():
                x = rows("X", H, r0, n)
                ew(es(part, "res"), x, rows("AO", H, r0, n),
                   mod(s, f"res2l_{part}" if last else f"res2_{part}"), rows("Zn", H, r0, n), x,
                   f"dbl{b} res2 {part}")
        for j in range(N_SGL):
            gemm(f"r{key}_sgl_in", whole("Zn"), f"sgl{j}.in", whole("FU"))
            ew(f"r{key}_qk_sgl", whole("FU"), whole("FU"), par(f"qk_sgl{j}"), whole("FU"),
               whole("FU"), f"sgl{j} qk")
            op("fa", f"r{key}_attn_sgl", [whole("FU")] * 4, f"sgl{j} attention")
            gemm(f"r{key}_sgl_out", whole("FU"), f"sgl{j}.out", whole("AO"))
            ew(f"r{key}_res_all", whole("X"), whole("AO"),
               mod(s, "sgl_last" if j == N_SGL - 1 else "sgl"), whole("Zn"), whole("X"),
               f"sgl{j} res" + (" + norm_out" if j == N_SGL - 1 else ""))
        gemm(f"r{R}_proj_out", rows("Zn", H, L_TXT, T_img), "proj_out", rows("V", 1024, 0, T_img))
        ew(f"r{R}_euler", whole("LAT"), whole("V"), dt(s), whole("LAT"), None, f"euler {s}")

    # -- VAE decoder
    phase[0] = "vae"
    for o in vp.ops:
        args = []
        for kind, key in o["args"]:
            if kind == "buf":
                args.append(whole(vae_buffer(key)))
            else:
                args.append((f"vae_{kind}", key))      # vae_w: a weight slice, vae_gn: a GN block
        op(o["set"], o["stream"], args, o["what"])
    return pl


# ---------------------------------------------------------------------------- streams

def gemm_streams() -> dict[str, dict]:
    """dit_gemm streams this pipeline adds to export_dit_kernels.klein_streams."""
    return {"mod": dict(M=512, K=H, N=MOD_N, role="modulation, all steps (M = steps padded)"),
            "ctx_emb": dict(M=L_TXT, K=7680, N=H, layout={"lda": CTX_LD, "a_size": L_TXT * CTX_LD},
                            role="context_embedder (CTX row stride 8192)")}


def ew_streams(resolutions: list[int]) -> dict[str, dict]:
    """dit_ew streams this pipeline adds to export_dit_kernels.klein_ew_streams."""
    view = lambda T, ld=EL, off=0: {"off": off, "ld": ld, "T": T, "E": 1}  # noqa: E731
    out = {"t_silu": {"op": "silu", "a": view(512), "y": view(512),
                      "sizes": {"X": 512 * EL, "B": EL, "P": EL, "Y": 512 * EL, "Z": EL}}}
    for R in resolutions:
        n = euler_elems(image_tokens(R))
        Tp = 24 * n
        tile = lambda ld: {"kind": "tile24", "off": 0, "ld": ld, "n": n}  # noqa: E731
        out[f"r{R}_euler"] = {"op": "euler", "a": tile(LAT_CH), "b": tile(1024), "y": tile(LAT_CH),
                              "p_off": EL, "n_par": 1, "real_T": image_tokens(R),
                              "sizes": {"X": Tp * LAT_CH, "B": Tp * 1024, "P": 3 * EL,
                                        "Y": Tp * LAT_CH, "Z": EL}}
    for i, k in enumerate(TAPS):
        out[f"te_tap{k}"] = {
            "op": "res_ln_mod", "norm": "rms", "unit_gate": 1, "W": TE_HID, "a": view(L_TXT),
            "b": view(L_TXT), "y": view(L_TXT), "z": view(L_TXT, CTX_LD, TE_HID * i),
            "p_off": EL, "n_par": 1, "idx": {"scale": 0},
            "sizes": {"X": L_TXT * EL, "B": L_TXT * EL, "P": 3 * EL, "Y": L_TXT * EL,
                      "Z": L_TXT * CTX_LD}}
    return out


if __name__ == "__main__":
    import sys
    for key in sys.argv[1:] or ["512", "1024", "512e512", "1024e1024"]:
        pl = plan(*parse_config(key)[:1], edit=parse_config(key)[1])
        by = {}
        for o in pl.ops:
            by.setdefault(o["phase"], []).append(o)
        gb = sum(pl.buffers.values()) / 2 ** 30
        print(f"{key}: {len(pl.ops)} dispatches, activations {gb:.2f} GiB")
        for ph, ops in by.items():
            sw = sum(1 for a, b in zip(ops, ops[1:]) if a["set"] != b["set"])
            print(f"   {ph:6s} {len(ops):4d} dispatches, {sw:3d} kernel-set switches")
