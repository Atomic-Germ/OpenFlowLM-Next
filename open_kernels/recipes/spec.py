"""ModelSpec: the hyperparameter tuple a recipe composes kernels from.

One plain dataclass, JSON on disk, built from either of the two places a model
already publishes its shape: the HF-style `config.json` OFLM ships beside the
`.q4nx` container, or a GGUF's metadata (`general.architecture`,
`<arch>.embedding_length`, ...). Anything a recipe needs that is not here is
a recipe constant (a family property), not a model property.

Traces: OPEN-SPEC-DERIVE (specs/open-engine/spec.md).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

LINEAR, FULL, DENSE, DENSE_LOCAL = "linear_attention", "full_attention", "dense", "dense_local"
SHORT_CONV = "short_conv"                             # LFM2: a short depthwise causal conv INSTEAD of attention
LAYER_TYPES = (LINEAR, FULL, DENSE, DENSE_LOCAL, SHORT_CONV)   # dense_local: a dense layer with sliding-window attention
# Adding a member here is free: `layer_types` is one field and no shipped spec uses the new
# value, so no hash moves. Adding a dataclass FIELD is not - spec_hash() covers every field.

# ---- the per-role weight format (OPEN-QUANT-Q8). A container stores each tensor at
# q4_1 (5120-byte chunks) or q8 (8704), and which of the two a projection is decides
# whether the kernels stream it at q8 or the packer re-quantizes it. Roles, not tensor
# names, so the recipes stay readable; the deriver maps the names once.
#
# `lm_head` is deliberately NOT a role: the family already fixes the head's format (the
# MoE / qwen35 recipes pack it with `lmhead_q8`, the dense recipes with `std_perm`), so
# putting it in the map would move every shipped model's spec_hash for no kernel change.
QUANT_ROLES = ("attn", "linear", "linear_out", "shared", "ffn", "experts")
QUANT_FORMATS = ("q4_1", "q8", "mxfp4")
DEFAULT_QUANT = "q4_1"
CHUNK_FORMAT = {5120: "q4_1", 8704: "q8"}
# 2560 is deliberately NOT in that table: GPT-OSS ships both its q4_1 projections and its
# MXFP4 experts at that size, so the byte count alone does not name the format and only the
# dtype separates them. A caller that has the dtypes gets the right answer; one that does
# not gets a refusal rather than a guess.
AMBIGUOUS_CHUNK = {2560: {"I8": "q4_1", "U8": "mxfp4"}}


class SpecError(ValueError):
    """The metadata cannot be turned into a ModelSpec; the message names the key."""


@dataclass(frozen=True)
class ModelSpec:
    family: str                       # the recipe: "qwen36moe" | "qwen35" | "qwen3" | "llama3" | "gemma3" | "hunyuan" | "granite" | "phi3" | "qwen2" | "gptoss" (no recipe yet)
    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]      # per layer: LINEAR | FULL | DENSE
    vocab: int                        # lm_head rows (padded)
    real_vocab: int                   # the tokenizer's ids; logits above it are undefined
    # full attention
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int                   # partial RoPE: rotated dims per head
    rope_theta: float
    rope_scaling: dict | None = None  # Llama 3: {"factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"};
                                      # Gemma 3: {"rope_type": "linear", "factor"} (the global layers)
    rope_local_theta: float = 0.0     # Gemma 3: the sliding-window layers' theta (unscaled); 0 = one RoPE for every layer
    sliding_window: int = 0           # rows a dense_local layer attends to (the newest W, itself included)
    qk_norm: bool = True
    attn_gate: bool = True            # sigmoid output gate (a second q-width projection)
    # linear attention (Gated DeltaNet); zeros for a family without it
    lin_key_heads: int = 0
    lin_value_heads: int = 0
    lin_key_dim: int = 0
    lin_value_dim: int = 0
    conv_kernel: int = 0
    # dense FFN (act(gate) * up @ down); 0 for a MoE-only family
    intermediate: int = 0
    activation: str = "silu"          # "silu" | "gelu_tanh"
    sandwich_norms: bool = False      # Gemma: post-attention and post-FFN norms on the block outputs, pre-FFN norm on the residual
    # MoE; num_experts == 0 means dense
    num_experts: int = 0
    experts_per_tok: int = 0
    moe_intermediate: int = 0
    shared_expert_intermediate: int = 0   # 0 = no shared expert
    norm_eps: float = 1e-6
    # the weight format: the string every role is at ("q4_1"), or a role -> format map
    # carrying only the roles that differ from it (`{"attn": "q8"}`). QUANT_ROLES above.
    quant: str | dict = DEFAULT_QUANT
    extra: dict = field(default_factory=dict)   # informational (model name, source)

    # ---- derived
    @property
    def lin_qkv_dim(self) -> int:
        """Rows of the fused q|k|v projection of a linear layer: 2 key groups + value."""
        return 2 * self.lin_key_heads * self.lin_key_dim + self.lin_value_heads * self.lin_value_dim

    @property
    def lin_value_width(self) -> int:
        return self.lin_value_heads * self.lin_value_dim

    @property
    def attn_q_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def attn_kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def has_linear(self) -> bool:
        return LINEAR in self.layer_types

    @property
    def has_full(self) -> bool:
        return FULL in self.layer_types

    @property
    def has_dense(self) -> bool:
        return DENSE in self.layer_types or DENSE_LOCAL in self.layer_types

    @property
    def has_local(self) -> bool:
        return DENSE_LOCAL in self.layer_types

    @property
    def has_short_conv(self) -> bool:
        return SHORT_CONV in self.layer_types

    # ---- the weight format, per role
    @property
    def quant_map(self) -> dict[str, str]:
        """role -> format, every role present."""
        if isinstance(self.quant, str):
            return {r: self.quant for r in QUANT_ROLES}
        return {r: self.quant.get(r, DEFAULT_QUANT) for r in QUANT_ROLES}

    def quant_of(self, role: str) -> str:
        if role not in QUANT_ROLES:
            raise SpecError(f"no quant role {role!r} (have {list(QUANT_ROLES)})")
        return self.quant_map[role]

    @property
    def q8_roles(self) -> frozenset:
        return frozenset(r for r, f in self.quant_map.items() if f == "q8")

    def canonical_quant(self):
        """The form that goes on disk and into the hashes: the bare string when every role
        is at the default, else only the roles that differ. This is what keeps a model with
        no q8 projection hashing, and serialising, exactly as it did before roles existed."""
        m = self.quant_map
        if all(f == DEFAULT_QUANT for f in m.values()):
            return DEFAULT_QUANT
        if isinstance(self.quant, str):
            return self.quant
        return {r: m[r] for r in QUANT_ROLES if m[r] != DEFAULT_QUANT}

    def quant_hash(self) -> str:
        """A short stable hash of the quant map; "" when nothing is at q8. Build directory
        names carry it so a q8 variant is a different kernel set without renaming the
        directories every shipped model already builds into."""
        if not self.q8_roles:
            return ""
        q = self.canonical_quant()
        return hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()[:8]

    def rope_inv_freq(self, local: bool = False, ctx: int | None = None) -> list[float]:
        """The inverse frequency of each rotary pair i < rotary_dim/2: theta^(-2i/rot), with Llama 3's
        wavelength-dependent scaling, a linear factor, or Phi-3's longrope factor lists when
        `rope_scaling` says so (HF's _compute_llama3_parameters / _compute_linear_scaling_rope_parameters
        / _compute_longrope_parameters). `local`: the sliding-window layers' table (Gemma:
        rope_local_theta, unscaled). `ctx`: the context the table serves -- longrope takes its
        long factors above original_max_position_embeddings and its short ones otherwise."""
        import math
        half = self.rotary_dim // 2
        if local:
            theta = self.rope_local_theta or self.rope_theta
            return [theta ** (-i / half) for i in range(half)]
        inv = [self.rope_theta ** (-i / half) for i in range(half)]
        sc = self.rope_scaling
        if sc and sc.get("rope_type") == "linear":
            return [f / float(sc["factor"]) for f in inv]
        if sc and sc.get("rope_type") == "longrope":
            long = ctx is not None and ctx > float(sc["original_max_position_embeddings"])
            fac = sc["long_factor"] if long else sc["short_factor"]
            return [f / float(x) for f, x in zip(inv, fac)]
        if sc and sc.get("rope_type") == "yarn":
            return _yarn_inv_freq(inv, self.rotary_dim, self.rope_theta, sc)
        if sc:
            factor = float(sc["factor"])
            lo, hi = float(sc["low_freq_factor"]), float(sc["high_freq_factor"])
            old = float(sc["original_max_position_embeddings"])
            low_wl, high_wl = old / lo, old / hi
            out = []
            for f in inv:
                wl = 2 * math.pi / f
                if wl < high_wl:
                    out.append(f)
                elif wl > low_wl:
                    out.append(f / factor)
                else:
                    smooth = (old / wl - lo) / (hi - lo)
                    out.append((1 - smooth) * f / factor + smooth * f)
            inv = out
        return inv

    def rope_scale(self) -> float:
        """What cos and sin are multiplied by: longrope's attention factor
        sqrt(1 + ln(factor) / ln(original_max_position_embeddings)), yarn's
        0.1 * ln(factor) + 1, 1.0 for everyone else."""
        import math
        sc = self.rope_scaling
        if sc and sc.get("rope_type") == "longrope":
            return math.sqrt(1.0 + math.log(float(sc["factor"])) / math.log(float(sc["original_max_position_embeddings"])))
        if sc and sc.get("rope_type") == "yarn":
            if sc.get("attention_factor") is not None:
                return float(sc["attention_factor"])
            factor = float(sc["factor"])
            return 1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0
        return 1.0

    # ---- serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_types"] = list(self.layer_types)
        d["quant"] = self.canonical_quant()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSpec":
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise SpecError(f"unknown ModelSpec field(s): {unknown}")
        kw = dict(d)
        kw["layer_types"] = tuple(kw["layer_types"])
        q = kw.get("quant", DEFAULT_QUANT)
        if isinstance(q, Mapping):
            bad = sorted(set(q) - set(QUANT_ROLES))
            if bad:
                raise SpecError(f"quant: unknown role(s) {bad} (have {list(QUANT_ROLES)})")
            kw["quant"] = {r: v for r, v in q.items() if v != DEFAULT_QUANT}
        elif not isinstance(q, str):
            raise SpecError("quant must be a format name or a role -> format map")
        for t in kw["layer_types"]:
            if t not in LAYER_TYPES:
                raise SpecError(f"layer_types: unknown layer type {t!r}")
        if len(kw["layer_types"]) != kw["num_layers"]:
            raise SpecError(f"layer_types has {len(kw['layer_types'])} entries, num_layers is {kw['num_layers']}")
        return cls(**kw)

    @classmethod
    def from_json(cls, s: str) -> "ModelSpec":
        return cls.from_dict(json.loads(s))

    def spec_hash(self) -> str:
        """Stable hash of the hyperparameters (not of `extra`)."""
        d = self.to_dict()
        d.pop("extra", None)
        return "sha256:" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    # ---- sources
    @classmethod
    def from_hf_config(cls, cfg: Mapping[str, Any], real_vocab: int | None = None) -> "ModelSpec":
        """From the HF-style config.json OFLM ships with a model.

        `real_vocab` is the tokenizer's id count (tokenizer.json); it defaults
        to vocab_size, which for OFLM's Qwen3.6 containers is the padded lm_head
        row count -- pass the real one when the tokenizer is at hand."""
        mt = cfg.get("model_type")
        if mt not in HF_FAMILIES:
            raise SpecError(f"model_type {mt!r} has no recipe (known: {sorted(HF_FAMILIES)})")
        return HF_FAMILIES[mt](cfg, real_vocab)

    @classmethod
    def from_gguf_metadata(cls, md: Mapping[str, Any]) -> "ModelSpec":
        """From a GGUF's key/value metadata (llama.cpp's key names)."""
        arch = md.get("general.architecture")
        if arch not in GGUF_FAMILIES:
            raise SpecError(f"general.architecture {arch!r} has no recipe (known: {sorted(GGUF_FAMILIES)})")
        return GGUF_FAMILIES[arch](md)


def _need(d: Mapping[str, Any], key: str, what: str = "config.json"):
    if key not in d:
        raise SpecError(f"{what} lacks {key!r}")
    return d[key]


def _yarn_inv_freq(inv: list[float], rot: int, theta: float, sc: Mapping[str, Any]) -> list[float]:
    """YaRN's inverse frequencies (HF's _compute_yarn_parameters). Each rotary pair is either
    left alone -- it completes few enough turns inside the original context that the model has
    seen its whole range -- or divided by `factor`, which is plain position interpolation. The
    pairs in between take a linear blend of the two, over the dim range where beta_fast and
    beta_slow rotations fit in `original_max_position_embeddings`.

    HF's `linear_ramp_factor` indexes its rot/2 pairs against bounds computed on the rot
    scale; that asymmetry is reproduced here rather than corrected, because the position
    table has to be the one the weights were trained against."""
    import math
    factor = float(sc["factor"])
    orig = float(sc["original_max_position_embeddings"])
    beta_fast = float(sc.get("beta_fast") or 32)
    beta_slow = float(sc.get("beta_slow") or 1)

    def corr(rotations: float) -> float:
        return rot * math.log(orig / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low, high = corr(beta_fast), corr(beta_slow)
    if sc.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0.0), min(high, rot - 1.0)
    if low == high:
        high += 0.001                      # HF prevents the singularity the same way
    out = []
    for i, f in enumerate(inv):
        ramp = min(max((i - low) / (high - low), 0.0), 1.0)
        extrap = 1.0 - ramp
        out.append(f / factor * (1.0 - extrap) + f * extrap)
    return out


def _layer_types_hf(cfg: Mapping[str, Any], n: int) -> tuple[str, ...]:
    if "layer_types" in cfg:
        lt = tuple(cfg["layer_types"])
        if len(lt) != n:
            raise SpecError(f"layer_types has {len(lt)} entries, num_hidden_layers is {n}")
        bad = sorted({t for t in lt if t not in (LINEAR, FULL)})
        if bad:
            raise SpecError(f"layer_types: unknown layer type(s) {bad}")
        return lt
    interval = _need(cfg, "full_attention_interval")
    if interval <= 0:
        raise SpecError("full_attention_interval must be positive")
    return tuple(FULL if (l + 1) % interval == 0 else LINEAR for l in range(n))


def _qwen36moe_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    n = _need(cfg, "num_hidden_layers")
    rope = cfg.get("rope_parameters", {})
    theta = rope.get("rope_theta", cfg.get("rope_theta"))
    if theta is None:
        raise SpecError("config.json lacks 'rope_theta' (top level or rope_parameters)")
    prf = rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
    hd = _need(cfg, "head_dim")
    vocab = _need(cfg, "vocab_size")
    return ModelSpec(
        family="qwen36moe",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=_layer_types_hf(cfg, n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=_need(cfg, "num_attention_heads"),
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=int(round(hd * prf)),
        rope_theta=float(theta),
        qk_norm=True,
        attn_gate=bool(cfg.get("attn_output_gate", True)),
        lin_key_heads=_need(cfg, "linear_num_key_heads"),
        lin_value_heads=_need(cfg, "linear_num_value_heads"),
        lin_key_dim=_need(cfg, "linear_key_head_dim"),
        lin_value_dim=_need(cfg, "linear_value_head_dim"),
        conv_kernel=_need(cfg, "linear_conv_kernel_dim"),
        num_experts=_need(cfg, "num_experts"),
        experts_per_tok=_need(cfg, "num_experts_per_tok"),
        moe_intermediate=_need(cfg, "moe_intermediate_size"),
        shared_expert_intermediate=cfg.get("shared_expert_intermediate_size", 0),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _qwen36moe_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    hd = k("attention.key_length")
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    interval = k("full_attention_interval")
    if interval <= 0:
        raise SpecError(f"{a}.full_attention_interval must be positive")
    conv = k("ssm.conv_kernel")
    key_heads, val_heads = k("ssm.group_count"), k("ssm.time_step_rank")
    key_dim = k("ssm.state_size")
    inner = k("ssm.inner_size")
    if inner % val_heads:
        raise SpecError(f"{a}.ssm.inner_size {inner} is not a multiple of ssm.time_step_rank {val_heads}")
    return ModelSpec(
        family="qwen36moe",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple(FULL if (l + 1) % interval == 0 else LINEAR for l in range(n)),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=k("attention.head_count"),
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        qk_norm=True,
        attn_gate=True,
        lin_key_heads=key_heads,
        lin_value_heads=val_heads,
        lin_key_dim=key_dim,
        lin_value_dim=inner // val_heads,
        conv_kernel=conv,
        num_experts=k("expert_count"),
        experts_per_tok=k("expert_used_count"),
        moe_intermediate=k("expert_feed_forward_length"),
        shared_expert_intermediate=md.get(f"{a}.expert_shared_feed_forward_length", 0),
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )


def _qwen35_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Qwen3.5 dense: the Qwen3.6-MoE layer with the MoE block replaced by a silu-gated
    dense FFN. Every other field is the MoE's -- `layer_types` (3 linear : 1 full),
    `attn_output_gate`, head_dim 256, partial RoPE 0.25, 16 linear key heads of 128,
    conv kernel 4 -- so the field reads are `_qwen36moe_hf`'s.

    Qwen publishes the tower inside a `text_config` (`model_type: qwen3_5_text`) under a
    `qwen3_5` VLM wrapper; OFLM's containers flatten it. Both are read here.
    Images always route to the closed engine (the open one has no vision path)."""
    tc = cfg.get("text_config", cfg)
    if tc.get("num_experts") or tc.get("moe_intermediate_size"):
        raise SpecError("qwen3_5: num_experts / moe_intermediate_size say this is the MoE "
                        "variant (model_type qwen3_5_moe), not the dense one")
    s = _qwen36moe_hf(dict(tc, num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
                           shared_expert_intermediate_size=0, model_type=tc.get("model_type", "qwen3_5")),
                      real_vocab)
    d = s.to_dict()
    d.update(family="qwen35", intermediate=_need(tc, "intermediate_size"),
             activation="silu" if tc.get("hidden_act", "silu") == "silu" else tc["hidden_act"])
    d["extra"] = {"model_type": cfg.get("model_type", tc.get("model_type")), "source": "hf_config"}
    spec = ModelSpec.from_dict(d)
    if spec.activation != "silu":
        raise SpecError(f"qwen3_5: hidden_act {spec.activation!r} is not silu")
    return spec


def _qwen35_gguf(md: Mapping[str, Any]) -> ModelSpec:
    """arch `qwen35` (the dense line; the MoE is `qwen35moe`). llama.cpp writes the same
    ssm.* / full_attention_interval keys as the MoE plus `feed_forward_length`, and no
    expert keys -- read off Qwen3.8-9B-Distill-Q8_0.gguf, 2026-09-06."""
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    for bad in ("expert_count", "expert_feed_forward_length"):
        if md.get(f"{a}.{bad}"):
            raise SpecError(f"{a}: {bad} says this is the MoE variant (architecture qwen35moe)")
    base = _qwen36moe_gguf({**md, f"{a}.expert_count": 0, f"{a}.expert_used_count": 0,
                            f"{a}.expert_feed_forward_length": 0})
    d = base.to_dict()
    d.update(family="qwen35", intermediate=k("feed_forward_length"))
    return ModelSpec.from_dict(d)


def _qwen3_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Qwen3 dense: GQA with q/k RMSNorm, full RoPE, no attention gate, silu-gated FFN."""
    n = _need(cfg, "num_hidden_layers")
    hd = _need(cfg, "head_dim")
    vocab = _need(cfg, "vocab_size")
    if cfg.get("use_sliding_window"):
        raise SpecError("qwen3: use_sliding_window is not supported by the dense recipe")
    return ModelSpec(
        family="qwen3",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=_need(cfg, "num_attention_heads"),
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(cfg, "rope_theta")),
        qk_norm=True,
        attn_gate=False,
        intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _qwen3vl_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Qwen3-VL: the decoder is Qwen3 dense, so it derives as one and links to a Qwen3
    bundle of the same geometry. M-RoPE only changes the position table the engine hands
    the kernels, so it does not reach the spec, and neither does the tower -- the vision
    side is VitConfig's, read from `vision_config`.

    Raw HF nests the decoder under `text_config`; the container OFLM ships flattens it."""
    inner = cfg.get("text_config")
    text = {**inner, "model_type": cfg["model_type"]} if isinstance(inner, Mapping) else cfg
    return _qwen3_hf(text, real_vocab)


def _qwen25vl_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Qwen2.5-VL: the decoder is Qwen2.5 dense, so it derives as one and links to a Qwen2
    bundle of the same geometry -- the 3B's text half hashes to exactly what the shipped
    Qwen2.5-3B-Instruct container does. M-RoPE only changes the position records the engine
    builds, never the kernels, so it does not reach the spec; the tower is VitConfig's, read
    from `vision_config`, which this family's container carries in the plain transformers
    keys.

    OFLM's container flattens the decoder to the top level; raw HF nests it under
    `text_config`."""
    inner = cfg.get("text_config")
    text = {**inner, "model_type": cfg["model_type"]} if isinstance(inner, Mapping) else cfg
    return _qwen2_hf(text, real_vocab)


def _qwen2_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Qwen2.5 dense: GQA without q/k norms, full RoPE, silu-gated FFN. The one thing that
    sets it apart -- a per-channel bias on q/k/v -- is a family property the dense recipe
    carries (`dense.QKV_BIAS_FAMILIES`, attn.h's ATTN_QKV_BIAS), not a field here."""
    n = _need(cfg, "num_hidden_layers")
    heads = _need(cfg, "num_attention_heads")
    hidden = _need(cfg, "hidden_size")
    # Qwen2.5's configs predate the key; Qwen3's always carry it.
    if "head_dim" in cfg:
        hd = cfg["head_dim"]
    elif hidden % heads:
        raise SpecError(f"qwen2: no head_dim in config.json and hidden_size {hidden} is not "
                        f"a multiple of num_attention_heads {heads}")
    else:
        hd = hidden // heads
    vocab = _need(cfg, "vocab_size")
    return ModelSpec(
        family="qwen2",
        hidden=hidden,
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=heads,
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(cfg, "rope_theta")),
        qk_norm=False,
        attn_gate=False,
        intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


_LFM2_LAYER_NAMES = {"conv": SHORT_CONV, SHORT_CONV: SHORT_CONV, FULL: FULL}


def _lfm2_layer_types(cfg: Mapping[str, Any], n: int) -> tuple[str, ...]:
    """LFM2 says which layers keep attention one of two ways: `layer_types`, the list
    `Lfm2Config` builds ("conv" / "full_attention"), or `full_attn_idxs`, the index list the
    container ships. Either way the rest are short-conv layers."""
    if "layer_types" in cfg:
        lt = list(cfg["layer_types"])
        if len(lt) != n:
            raise SpecError(f"layer_types has {len(lt)} entries, num_hidden_layers is {n}")
        bad = sorted({t for t in lt if t not in _LFM2_LAYER_NAMES})
        if bad:
            raise SpecError(f"lfm2: layer_types: {bad} is not a layer type this family has "
                            f"(have {sorted(_LFM2_LAYER_NAMES)})")
        return tuple(_LFM2_LAYER_NAMES[t] for t in lt)
    idxs = set(_need(cfg, "full_attn_idxs"))
    return tuple(FULL if l in idxs else SHORT_CONV for l in range(n))


def _lfm2_ff_dim(cfg: Mapping[str, Any]) -> int:
    """The FFN width, by transformers' own rule: `block_ff_dim` overrides `intermediate_size`
    (`Lfm2Config.__post_init__`), then `Lfm2MLP` takes two thirds of it, applies the
    multiplier and rounds up to `block_multiple_of`. The 1.2B's 12288 lands on 8192, which is
    what its gate / up projections hold -- reading `intermediate_size` raw would have worked
    here by luck and given 5632 on a container that ships only the unadjusted width."""
    ff = int(cfg["block_ff_dim"] if "block_ff_dim" in cfg else _need(cfg, "intermediate_size"))
    if not cfg.get("block_auto_adjust_ff_dim", True):
        return ff
    ff = int(2 * ff / 3)
    mult = cfg.get("block_ffn_dim_multiplier")
    if mult is not None:
        ff = int(mult * ff)
    m = int(cfg.get("block_multiple_of", 256))
    return m * ((ff + m - 1) // m)


def _lfm2_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """LFM2: a hybrid where the layers that are not attention run a short depthwise causal
    convolution instead. Attention is GQA with q/k RMSNorm over the head, full RoPE and no
    gate -- the dense recipe's shape -- and every layer carries the same silu-gated FFN.

    The conv is `conv_L_cache` taps wide over `conv_dim` channels, and `conv_dim` is the
    hidden size on every LFM2 that ships, so it rides on `hidden` and `conv_kernel` (the
    field DeltaNet already has) rather than on a new field that would move every shipped
    model's hash. A config where that stops being true is refused by name."""
    n = _need(cfg, "num_hidden_layers")
    hidden = _need(cfg, "hidden_size")
    heads = _need(cfg, "num_attention_heads")
    hd = cfg.get("head_dim") or (hidden // heads if hidden % heads == 0 else None)
    if hd is None:
        raise SpecError(f"lfm2: no head_dim in config.json and hidden_size {hidden} is not a "
                        f"multiple of num_attention_heads {heads}")
    for key in ("conv_dim", "conv_dim_out"):
        w = cfg.get(key, hidden)
        if w != hidden:
            raise SpecError(f"lfm2: {key} {w} is not hidden_size {hidden}; the short conv is "
                            f"hidden-wide on every LFM2 that ships and ModelSpec has no "
                            f"separate conv width")
    if cfg.get("conv_bias"):
        raise SpecError("lfm2: conv_bias is set; the short-conv block this recipe implements "
                        "has no bias (no LFM2 container ships one)")
    vocab = _need(cfg, "vocab_size")
    return ModelSpec(
        family="lfm2",
        hidden=hidden,
        num_layers=n,
        layer_types=_lfm2_layer_types(cfg, n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=heads,
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(cfg, "rope_theta")),
        qk_norm=True,
        attn_gate=False,
        conv_kernel=int(cfg.get("conv_L_cache", 3)),
        intermediate=_lfm2_ff_dim(cfg),
        norm_eps=float(cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-5))),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )



# GPT-OSS's own layer-type names. `full_attention` here means a plain dense layer, NOT the
# spec's FULL -- that one is the full-attention half of Qwen3.6's linear/full alternation.
_GPTOSS_LAYER_TYPES = {"sliding_attention": DENSE_LOCAL, "full_attention": DENSE}


def _gptoss_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """GPT-OSS: GQA attention over an MoE FFN, a 128-row sliding window on every other layer,
    YaRN RoPE, and a learned per-head attention sink (`self_attn.sinks`, one scalar per head
    per layer) that joins the softmax denominator with no value vector behind it.

    Two config keys mean something other than what they say. `intermediate_size` is the
    EXPERT width -- there is no dense FFN, so `intermediate` stays 0 and `moe_intermediate`
    takes it. And `hidden_act` says silu while the experts compute a clamped SwiGLU:
    `(up + 1) * gate * sigmoid(1.702 * gate)` with gate clipped above at 7 and up clipped
    both ways, gate and up interleaved down the expert's rows rather than split in half. The
    activation records what they do; alpha and the limit are family constants, not fields.

    The sink and the biases are family properties for the same reason the Qwen2 bias is:
    every GPT-OSS has them and `spec_hash()` covers every field. There is no recipe for this
    family yet -- `families.family_module("gptoss")` says what is missing."""
    n = _need(cfg, "num_hidden_layers")
    heads = _need(cfg, "num_attention_heads")
    hidden = _need(cfg, "hidden_size")
    rope = cfg.get("rope_parameters") or {}
    theta = rope.get("rope_theta", cfg.get("rope_theta"))
    if theta is None:
        raise SpecError("gptoss: config.json lacks 'rope_theta' (top level or rope_parameters)")
    sc = cfg.get("rope_scaling") or {k: v for k, v in rope.items() if k != "rope_theta"} or None
    if sc:
        kind = sc.get("rope_type", sc.get("type"))
        if kind != "yarn":
            raise SpecError(f"gptoss: rope_scaling type {kind!r} is not supported (yarn only)")
        if sc.get("mscale") or sc.get("mscale_all_dim"):
            raise SpecError("gptoss: yarn mscale / mscale_all_dim is not supported "
                            "(the attention factor comes from 'factor' alone)")
        for key in ("factor", "original_max_position_embeddings"):
            if sc.get(key) is None:
                raise SpecError(f"gptoss: yarn rope_scaling lacks {key!r}")
        canon = {"rope_type": "yarn", "factor": float(sc["factor"]),
                 "beta_fast": float(sc.get("beta_fast") or 32), "beta_slow": float(sc.get("beta_slow") or 1),
                 "truncate": bool(sc.get("truncate", True)),
                 "original_max_position_embeddings": int(sc["original_max_position_embeddings"])}
        if sc.get("attention_factor") is not None:
            canon["attention_factor"] = float(sc["attention_factor"])
        sc = canon
    if "head_dim" in cfg and cfg["head_dim"]:
        hd = cfg["head_dim"]
    elif hidden % heads:
        raise SpecError(f"gptoss: no head_dim in config.json and hidden_size {hidden} is not "
                        f"a multiple of num_attention_heads {heads}")
    else:
        hd = hidden // heads
    if "layer_types" in cfg:
        raw = tuple(cfg["layer_types"])
        if len(raw) != n:
            raise SpecError(f"gptoss: layer_types has {len(raw)} entries, num_hidden_layers is {n}")
        bad = sorted({t for t in raw if t not in _GPTOSS_LAYER_TYPES})
        if bad:
            raise SpecError(f"gptoss: layer_types: unknown layer type(s) {bad} "
                            f"(have {sorted(_GPTOSS_LAYER_TYPES)})")
        layers = tuple(_GPTOSS_LAYER_TYPES[t] for t in raw)
    else:
        # HF's own default: the sliding layer comes first (configuration_gpt_oss.py)
        layers = tuple(DENSE_LOCAL if l % 2 == 0 else DENSE for l in range(n))
    window = int(cfg.get("sliding_window") or 0)
    if DENSE_LOCAL in layers and window <= 0:
        raise SpecError("gptoss: the sliding_attention layers need a positive 'sliding_window'")
    vocab = _need(cfg, "vocab_size")
    return ModelSpec(
        family="gptoss",
        hidden=hidden,
        num_layers=n,
        layer_types=layers,
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=heads,
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(theta),
        rope_scaling=sc,
        sliding_window=window,
        qk_norm=False,
        attn_gate=False,
        intermediate=0,
        activation="clamped_swiglu",
        num_experts=_need(cfg, "num_local_experts"),
        experts_per_tok=_need(cfg, "num_experts_per_tok"),
        moe_intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _phi3_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Phi-3 / Phi-4-mini: GQA without q/k norms, a PARTIAL rotation (`partial_rotary_factor`
    of the head, 96 of 128 on Phi-4-mini), longrope scaling (a short and a long factor list
    over the same theta, chosen by the context, plus one attention scale on cos / sin), silu
    FFN, tied head. The container materialises `lm_head.weight` and splits the fused
    `qkv_proj` / `gate_up_proj` into the plain names, so the dense recipe reads it as it
    reads Llama. Only the HF derivation exists: a Phi-3 GGUF carries the factor lists as
    tensors, not metadata."""
    n = _need(cfg, "num_hidden_layers")
    heads = _need(cfg, "num_attention_heads")
    hd = cfg.get("head_dim") or _need(cfg, "hidden_size") // heads
    rot = int(round(hd * float(cfg.get("partial_rotary_factor", 1.0))))
    vocab = _need(cfg, "vocab_size")
    if cfg.get("hidden_act", "silu") != "silu":
        raise SpecError(f"phi3: hidden_act {cfg.get('hidden_act')!r} is not silu (the dense kernel's FFN)")
    sc = cfg.get("rope_scaling")
    scaling = None
    raw_scaling = None            # the container's own rope_scaling sub-object, verbatim
    if sc:
        kind = sc.get("rope_type", sc.get("type"))
        if kind != "longrope":
            raise SpecError(f"phi3: rope_scaling type {kind!r} is not supported (longrope only)")
        orig = sc.get("original_max_position_embeddings", cfg.get("original_max_position_embeddings"))
        if orig is None:
            raise SpecError("phi3: longrope needs original_max_position_embeddings")
        short, long = sc.get("short_factor"), sc.get("long_factor")
        if not short or not long or len(short) != rot // 2 or len(long) != rot // 2:
            raise SpecError(f"phi3: longrope wants {rot // 2} short and long factors (one per rotary pair)")
        scaling = {"rope_type": "longrope", "short_factor": [float(x) for x in short],
                   "long_factor": [float(x) for x in long],
                   "factor": float(sc.get("factor") or _need(cfg, "max_position_embeddings") / orig),
                   "original_max_position_embeddings": int(orig)}
        raw_scaling = dict(sc)     # what a real container's config.json literally holds at this key
    return ModelSpec(
        family="phi3",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=heads,
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=rot,
        rope_theta=float(_need(cfg, "rope_theta")),
        rope_scaling=scaling,
        qk_norm=False,
        attn_gate=False,
        intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config",
               # verbatim, for OPEN-FAMILY-PHI3's load-time compatibility check: the
               # canonical dict above renames keys and adds the derived factor, so it is
               # not what a container's config.json literally holds at this key
               "rope_scaling_raw": raw_scaling},
    )


def _qwen3_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    hd = k("attention.key_length")
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    return ModelSpec(
        family="qwen3",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=k("attention.head_count"),
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        qk_norm=True,
        attn_gate=False,
        intermediate=k("feed_forward_length"),
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )


def _llama3_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Llama 3: GQA without q/k norms, full RoPE with the llama3 frequency scaling, no gate, silu FFN.

    `tie_word_embeddings` is NOT a refusal. Llama 3.2 (1B / 3B) ties the head to the
    embedding table, but every container the recipe packs from materialises
    `lm_head.weight` as its own q4 tensor -- OFLM's `.q4nx` does it for
    `Llama-3.2-{1,3}B-NPU2` (I8 [32064, 5120] / [48096, 5120], the whole 128256-row
    head), and utilities/q4nx-build does it for a tied GGUF (the HunYuan converter's
    zero-padded head). The derivation sees only config.json, never the container, so
    the invariant is enforced where it is observable: `pack.apply_op` refuses a
    container that lacks the tensor, by name.
    """
    n = _need(cfg, "num_hidden_layers")
    heads = _need(cfg, "num_attention_heads")
    hd = cfg.get("head_dim") or _need(cfg, "hidden_size") // heads
    vocab = _need(cfg, "vocab_size")
    sc = cfg.get("rope_scaling")
    scaling = None
    if sc:
        if sc.get("rope_type", sc.get("type")) != "llama3":
            raise SpecError(f"llama: rope_scaling type {sc.get('rope_type', sc.get('type'))!r} is not supported (llama3 only)")
        scaling = {k: sc[k] for k in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")}
    return ModelSpec(
        family="llama3",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=heads,
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(cfg, "rope_theta")),
        rope_scaling=scaling,
        qk_norm=False,
        attn_gate=False,
        intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _llama3_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    heads = k("attention.head_count")
    hd = md.get(f"{a}.attention.key_length", k("embedding_length") // heads)
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    scaling = None
    if md.get(f"{a}.rope.scaling.type") == "llama3" or f"{a}.rope.scaling.factor" in md:
        scaling = {"factor": md[f"{a}.rope.scaling.factor"],
                   "low_freq_factor": md.get(f"{a}.rope.scaling.low_freq_factor", 1.0),
                   "high_freq_factor": md.get(f"{a}.rope.scaling.high_freq_factor", 4.0),
                   "original_max_position_embeddings": md.get(f"{a}.rope.scaling.original_context_length", 8192)}
    return ModelSpec(
        family="llama3",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=heads,
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        rope_scaling=scaling,
        qk_norm=False,
        attn_gate=False,
        intermediate=k("feed_forward_length"),
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-5)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )


def _ntk_alpha_base(base: float, alpha: float, head_dim: int) -> float:
    """HunYuan's NTK-aware alpha scaling, applied once at load: the RoPE base is stretched
    to `base * alpha^(d/(d-2))` and the frequencies are otherwise the plain ones. Same
    formula as transformers' HunYuanDenseV1 rotary embedding and llama.cpp's converter
    (conversion/hunyuan.py `scaled_base`), so a GGUF's `rope.freq_base` is already this."""
    return base * (alpha ** (head_dim / (head_dim - 2)))


def _hunyuan_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """HunYuan V1 dense (Hy-MT2-7B, Hunyuan-{1.8,4,7}B): Llama 3's GQA shape with q/k
    RMSNorm applied AFTER RoPE, one static NTK-alpha RoPE base, silu FFN, a tied head.
    The post-rope norm order is a family property of the recipe, not a spec field."""
    n = _need(cfg, "num_hidden_layers")
    heads = _need(cfg, "num_attention_heads")
    hd = cfg.get("head_dim") or cfg.get("attention_head_dim") or _need(cfg, "hidden_size") // heads
    vocab = _need(cfg, "vocab_size")
    if cfg.get("norm_type", "rms") != "rms":
        raise SpecError(f"hunyuan: norm_type {cfg.get('norm_type')!r} is not 'rms'")
    if cfg.get("use_cla"):
        raise SpecError("hunyuan: use_cla (cross-layer attention shares KV between layers) is not supported")
    if cfg.get("num_experts") or cfg.get("moe_topk"):
        raise SpecError("hunyuan: the MoE variants are not supported by the dense recipe")
    if cfg.get("attention_bias") or cfg.get("mlp_bias"):
        raise SpecError("hunyuan: attention_bias / mlp_bias are not supported (the GEMVs have no bias)")
    theta = float(_need(cfg, "rope_theta"))
    sc = cfg.get("rope_scaling")
    if sc:
        kind = sc.get("rope_type", sc.get("type"))
        if kind != "dynamic":
            raise SpecError(f"hunyuan: rope_scaling type {kind!r} is not supported (dynamic/alpha only)")
        if float(sc.get("factor", 1.0)) != 1.0:
            raise SpecError(f"hunyuan: rope_scaling factor {sc.get('factor')} is not 1 (only the NTK alpha is folded in)")
        for k in ("mscale", "mscale_all_dim"):
            if float(sc.get(k, 1.0)) != 1.0:
                raise SpecError(f"hunyuan: rope_scaling {k} {sc[k]} is not 1 (attn.h has no logit rescale)")
        theta = _ntk_alpha_base(theta, float(sc.get("alpha", 1000.0)), hd)
    return ModelSpec(
        family="hunyuan",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=heads,
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=theta,
        qk_norm=bool(cfg.get("use_qk_norm", True)),
        attn_gate=False,
        intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _hunyuan_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    heads = k("attention.head_count")
    hd = md.get(f"{a}.attention.key_length", k("embedding_length") // heads)
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    # llama.cpp folds the NTK alpha into rope.freq_base and writes scaling type NONE, so the
    # base is already the stretched one -- a scaling factor other than 1 would be something new.
    factor = float(md.get(f"{a}.rope.scaling.factor", 1.0))
    if factor != 1.0:
        raise SpecError(f"hunyuan: {a}.rope.scaling.factor {factor} is not 1 (the alpha is folded into freq_base)")
    return ModelSpec(
        family="hunyuan",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=heads,
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        qk_norm=True,
        attn_gate=False,
        intermediate=k("feed_forward_length"),
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-5)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )


def _gemma3_layer_types(n: int, cfg: Mapping[str, Any]) -> tuple[str, ...]:
    if "layer_types" in cfg:
        m = {"sliding_attention": DENSE_LOCAL, "full_attention": DENSE}
        bad = sorted({t for t in cfg["layer_types"] if t not in m})
        if bad:
            raise SpecError(f"gemma3: unknown layer type(s) {bad}")
        lt = tuple(m[t] for t in cfg["layer_types"])
    else:
        pat = int(cfg.get("sliding_window_pattern", 6))
        lt = tuple(DENSE if (l + 1) % pat == 0 else DENSE_LOCAL for l in range(n))
    if len(lt) != n:
        raise SpecError(f"gemma3: layer_types has {len(lt)} entries, num_hidden_layers is {n}")
    return lt


def _gemma3_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Gemma 3 (text): GQA with q/k RMSNorm, GeGLU-tanh FFN, sandwich norms, 5:1 sliding / global layers,
    a local RoPE theta and a linearly scaled global one. The container stores the norms' 1 + w and the
    sqrt(hidden)-scaled embeddings, so neither is a recipe transform."""
    tc = cfg.get("text_config", cfg)
    n = _need(tc, "num_hidden_layers")
    hd = _need(tc, "head_dim")
    vocab = _need(tc, "vocab_size")
    if tc.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise SpecError(f"gemma3: hidden_activation {tc.get('hidden_activation')!r} is not gelu_pytorch_tanh")
    if tc.get("final_logit_softcapping") or tc.get("attn_logit_softcapping"):
        raise SpecError("gemma3: logit softcapping is not supported (Gemma 3 has none)")
    qps = tc.get("query_pre_attn_scalar", hd)
    if abs(float(qps) - hd) > 1e-6:
        raise SpecError(f"gemma3: query_pre_attn_scalar {qps} != head_dim {hd} (attn.h scales by 1/sqrt(HD))")
    sc = tc.get("rope_scaling")
    scaling = None
    if sc:
        if sc.get("rope_type", sc.get("type")) != "linear":
            raise SpecError(f"gemma3: rope_scaling type {sc.get('rope_type')!r} is not supported (linear only)")
        scaling = {"rope_type": "linear", "factor": float(sc["factor"])}
    return ModelSpec(
        family="gemma3",
        hidden=_need(tc, "hidden_size"),
        num_layers=n,
        layer_types=_gemma3_layer_types(n, tc),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=_need(tc, "num_attention_heads"),
        num_kv_heads=_need(tc, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(tc, "rope_theta")),
        rope_scaling=scaling,
        rope_local_theta=float(tc.get("rope_local_base_freq", 10000.0)),
        sliding_window=int(_need(tc, "sliding_window")),
        qk_norm=True,
        attn_gate=False,
        intermediate=_need(tc, "intermediate_size"),
        activation="gelu_tanh",
        sandwich_norms=True,
        norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        quant="q4_1",
        extra={"model_type": cfg.get("model_type", tc.get("model_type")), "source": "hf_config"},
    )


def _gemma3_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    hd = k("attention.key_length")
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    pat = int(md.get(f"{a}.attention.sliding_window_pattern", 6))
    factor = md.get(f"{a}.rope.scaling.factor")
    return ModelSpec(
        family="gemma3",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple(DENSE if (l + 1) % pat == 0 else DENSE_LOCAL for l in range(n)),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=k("attention.head_count"),
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        rope_scaling={"rope_type": "linear", "factor": float(factor)} if factor else None,
        rope_local_theta=float(md.get(f"{a}.rope.local_freq_base", 10000.0)),
        sliding_window=int(k("attention.sliding_window")),
        qk_norm=True,
        attn_gate=False,
        intermediate=k("feed_forward_length"),
        activation="gelu_tanh",
        sandwich_norms=True,
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )



# Granite's four scalar multipliers, and why the recipe can ignore them.
#
# IBM Granite is Llama plus four scalars that transformers applies at run time:
#
#     attention_multiplier   replaces llama's implicit hd**-0.5 attention scale
#     embedding_multiplier   inputs_embeds = embed(ids) * embedding_multiplier
#     residual_multiplier    h = residual + h * residual_multiplier, both blocks
#     logits_scaling         logits = lm_head(h) / logits_scaling
#
# ModelSpec cannot express any of them and designs/attn/attn.h hard-codes
# kScale = 1/sqrt(HD), so a model that needs them cannot run on this recipe --
# the same wall Gemma 3 hits with query_pre_attn_scalar, and handled the same
# way: refuse rather than approximate.
#
# They do not need expressing, because all four FOLD LOSSLESSLY into the
# quantised weights at conversion time (q4nx-build's granite builder):
# q_proj *= attention_multiplier * sqrt(hd); embed_tokens *= embedding_
# multiplier; o_proj and down_proj *= residual_multiplier; lm_head /=
# logits_scaling. The container's config.json then describes the FOLDED model
# and keeps the originals under "q4nx_folded_multipliers".
#
# For granite-4.2-3b the only non-unit factor is attention_multiplier =
# 0.015625 at hd 64, so the fold is q_proj *= 0.125 and the folded config reads
# attention_multiplier = 0.125 = 64**-0.5 exactly. 0.125 is a power of two, so
# it is exact in bf16.
#
# Hence the checks below: this recipe accepts a folded Granite and refuses an
# unfolded one by name, rather than running it and returning plausible garbage.
def _granite_scale_check(where: str, attn_mult: float | None, hd: int,
                         others: Mapping[str, Any]) -> None:
    want = hd ** -0.5
    # The attention multiplier is REQUIRED, and the other three are not, which
    # looks inconsistent until you ask what an absent key means. All four default
    # to 1.0 in transformers' GraniteConfig. For the other three that default IS
    # what the recipe needs, so absence is the good case. For this one it is not:
    # 1.0 against attn.h's 1/sqrt(HD) is a factor of 8 at head_dim 64, applied
    # silently to every score. And an absent key cannot be told apart from a
    # folded container that simply failed to record the fold -- so it is refused
    # here rather than passed on to fail at load, where the manifest check
    # (dense.py's hf_config_check, which requires this key) can only report that
    # a field is missing.
    if attn_mult is None:
        raise SpecError(
            f"granite: {where} does not state the attention multiplier, and it is not "
            f"optional. attn.h hard-codes 1/sqrt(HD) = {want}; transformers defaults "
            f"GraniteConfig.attention_multiplier to 1.0, so a config that omits it "
            f"describes a model scaling attention by 1.0 -- a silent factor of "
            f"{1.0 / want:g} on every score at head_dim {hd}. A container converted by "
            f"q4nx-build records the post-fold value ({want}) explicitly; convert it, "
            f"or state the multiplier if you know it.")
    if abs(float(attn_mult) - want) > 1e-9:
        raise SpecError(
            f"granite: attention_multiplier {attn_mult} != head_dim**-0.5 {want} "
            f"(attn.h scales by 1/sqrt(HD)). This container looks UNFOLDED; convert it "
            f"with q4nx-build, which folds the multiplier into q_proj -- see {where}.")
    for name, v in others.items():
        if v is not None and abs(float(v) - 1.0) > 1e-9:
            raise SpecError(
                f"granite: {name} {v} != 1.0, so it was not folded into the weights "
                f"(the recipe has nowhere to apply it) -- convert with q4nx-build.")


def _granite_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """IBM Granite dense: Llama geometry with the four multipliers folded into the weights."""
    n = _need(cfg, "num_hidden_layers")
    heads = _need(cfg, "num_attention_heads")
    hd = cfg.get("head_dim") or _need(cfg, "hidden_size") // heads
    vocab = _need(cfg, "vocab_size")
    if cfg.get("rope_scaling"):
        raise SpecError("granite: rope_scaling is not supported (only the unscaled RoPE is validated)")
    if cfg.get("tie_word_embeddings"):
        raise SpecError("granite: tied embeddings are not supported (the head must be its own q4 tensor)")
    _granite_scale_check(
        "config.json's q4nx_folded_multipliers", cfg.get("attention_multiplier"), hd,
        {"embedding_multiplier": cfg.get("embedding_multiplier"),
         "residual_multiplier": cfg.get("residual_multiplier"),
         "logits_scaling": cfg.get("logits_scaling")})
    return ModelSpec(
        family="granite",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=heads,
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(cfg, "rope_theta")),
        rope_scaling=None,
        qk_norm=False,
        attn_gate=False,
        intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _granite_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    heads = k("attention.head_count")
    hd = md.get(f"{a}.attention.key_length", k("embedding_length") // heads)
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    if f"{a}.rope.scaling.factor" in md:
        raise SpecError("granite: rope_scaling is not supported (only the unscaled RoPE is validated)")
    # A raw Granite GGUF is UNFOLDED, so these are the real multipliers and this
    # is where that is caught -- the q4nx container is the folded artefact.
    _granite_scale_check(
        f"'{a}.attention.scale' in the GGUF", md.get(f"{a}.attention.scale"), hd,
        {"embedding_scale": md.get(f"{a}.embedding_scale"),
         "residual_scale": md.get(f"{a}.residual_scale"),
         "logit_scale": md.get(f"{a}.logit_scale")})
    return ModelSpec(
        family="granite",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=heads,
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        rope_scaling=None,
        qk_norm=False,
        attn_gate=False,
        intermediate=k("feed_forward_length"),
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-5)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )


def _gemma3_layer_types(n: int, cfg: Mapping[str, Any]) -> tuple[str, ...]:
    if "layer_types" in cfg:
        m = {"sliding_attention": DENSE_LOCAL, "full_attention": DENSE}
        bad = sorted({t for t in cfg["layer_types"] if t not in m})
        if bad:
            raise SpecError(f"gemma3: unknown layer type(s) {bad}")
        lt = tuple(m[t] for t in cfg["layer_types"])
    else:
        pat = int(cfg.get("sliding_window_pattern", 6))
        lt = tuple(DENSE if (l + 1) % pat == 0 else DENSE_LOCAL for l in range(n))
    if len(lt) != n:
        raise SpecError(f"gemma3: layer_types has {len(lt)} entries, num_hidden_layers is {n}")
    return lt


def _gemma3_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Gemma 3 (text): GQA with q/k RMSNorm, GeGLU-tanh FFN, sandwich norms, 5:1 sliding / global layers,
    a local RoPE theta and a linearly scaled global one. The container stores the norms' 1 + w and the
    sqrt(hidden)-scaled embeddings, so neither is a recipe transform."""
    tc = cfg.get("text_config", cfg)
    n = _need(tc, "num_hidden_layers")
    hd = _need(tc, "head_dim")
    vocab = _need(tc, "vocab_size")
    if tc.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise SpecError(f"gemma3: hidden_activation {tc.get('hidden_activation')!r} is not gelu_pytorch_tanh")
    if tc.get("final_logit_softcapping") or tc.get("attn_logit_softcapping"):
        raise SpecError("gemma3: logit softcapping is not supported (Gemma 3 has none)")
    qps = tc.get("query_pre_attn_scalar", hd)
    if abs(float(qps) - hd) > 1e-6:
        raise SpecError(f"gemma3: query_pre_attn_scalar {qps} != head_dim {hd} (attn.h scales by 1/sqrt(HD))")
    sc = tc.get("rope_scaling")
    scaling = None
    if sc:
        if sc.get("rope_type", sc.get("type")) != "linear":
            raise SpecError(f"gemma3: rope_scaling type {sc.get('rope_type')!r} is not supported (linear only)")
        scaling = {"rope_type": "linear", "factor": float(sc["factor"])}
    return ModelSpec(
        family="gemma3",
        hidden=_need(tc, "hidden_size"),
        num_layers=n,
        layer_types=_gemma3_layer_types(n, tc),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=_need(tc, "num_attention_heads"),
        num_kv_heads=_need(tc, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(tc, "rope_theta")),
        rope_scaling=scaling,
        rope_local_theta=float(tc.get("rope_local_base_freq", 10000.0)),
        sliding_window=int(_need(tc, "sliding_window")),
        qk_norm=True,
        attn_gate=False,
        intermediate=_need(tc, "intermediate_size"),
        activation="gelu_tanh",
        sandwich_norms=True,
        norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        quant="q4_1",
        extra={"model_type": cfg.get("model_type", tc.get("model_type")), "source": "hf_config"},
    )


def _gemma3_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    hd = k("attention.key_length")
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    pat = int(md.get(f"{a}.attention.sliding_window_pattern", 6))
    factor = md.get(f"{a}.rope.scaling.factor")
    return ModelSpec(
        family="gemma3",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple(DENSE if (l + 1) % pat == 0 else DENSE_LOCAL for l in range(n)),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=k("attention.head_count"),
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        rope_scaling={"rope_type": "linear", "factor": float(factor)} if factor else None,
        rope_local_theta=float(md.get(f"{a}.rope.local_freq_base", 10000.0)),
        sliding_window=int(k("attention.sliding_window")),
        qk_norm=True,
        attn_gate=False,
        intermediate=k("feed_forward_length"),
        activation="gelu_tanh",
        sandwich_norms=True,
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )

# `qwen3_5_moe_text` is the text-only derivative of the VLM `qwen3_5_moe` (Ornith-1.0-35B-A3B
# and friends: `Qwen3_5MoeForCausalLM`, no vision_config). Every field the recipe and the
# config check read is identical to the VLM's, so it derives through the same builder --
# exactly as `gemma3_text` and `qwen3_5_text` do for their towers.
HF_FAMILIES = {"qwen3_5_moe": _qwen36moe_hf, "qwen3_5_moe_text": _qwen36moe_hf,
               "qwen3_next": _qwen36moe_hf, "qwen3_5": _qwen35_hf,
               "qwen3_5_text": _qwen35_hf, "qwen3": _qwen3_hf, "qwen3_vl": _qwen3vl_hf,
               "qwen3_vl_text": _qwen3vl_hf, "qwen2": _qwen2_hf, "llama": _llama3_hf,
               "qwen2_5_vl": _qwen25vl_hf, "qwen2_5_vl_text": _qwen25vl_hf,
               "gemma3_text": _gemma3_hf, "gemma3": _gemma3_hf, "hunyuan_v1_dense": _hunyuan_hf,
               "granite": _granite_hf, "phi3": _phi3_hf, "lfm2": _lfm2_hf,
               "gpt_oss": _gptoss_hf}
GGUF_FAMILIES = {"qwen35moe": _qwen36moe_gguf, "qwen3next": _qwen36moe_gguf, "qwen35": _qwen35_gguf, "qwen3": _qwen3_gguf, "llama": _llama3_gguf,
                 "gemma3": _gemma3_gguf, "hunyuan-dense": _hunyuan_gguf, "granite": _granite_gguf}
_FAMILY_OF = {_qwen36moe_hf: "qwen36moe", _qwen36moe_gguf: "qwen36moe", _qwen35_hf: "qwen35",
              _qwen35_gguf: "qwen35",
              _qwen3_hf: "qwen3", _qwen3_gguf: "qwen3", _qwen3vl_hf: "qwen3", _qwen2_hf: "qwen2",
              _qwen25vl_hf: "qwen2",
              _llama3_hf: "llama3", _llama3_gguf: "llama3", _gemma3_hf: "gemma3", _gemma3_gguf: "gemma3",
              _hunyuan_hf: "hunyuan", _hunyuan_gguf: "hunyuan",
              _granite_hf: "granite", _granite_gguf: "granite", _phi3_hf: "phi3",
              _lfm2_hf: "lfm2", _gptoss_hf: "gptoss"}


def hf_model_types(family: str) -> list[str]:
    """The config.json model_type values a family's recipe accepts."""
    return sorted(k for k, f in HF_FAMILIES.items() if _FAMILY_OF[f] == family)


def gguf_architectures(family: str) -> list[str]:
    return sorted(k for k, f in GGUF_FAMILIES.items() if _FAMILY_OF[f] == family)


# ---- the per-role weight format, derived from what the model file actually holds
# (OPEN-QUANT-Q8). `config.json` does not say; the container's tensor shapes do, and so
# do a GGUF's tensor types. The tables below are the one place tensor names meet roles.
_LAYER_PREFIX = re.compile(r"^model\.layers?\.\d+\.")
_ATTN_HF = {"self_attn.q_proj.weight": "attn", "self_attn.k_proj.weight": "attn",
            "self_attn.v_proj.weight": "attn", "self_attn.o_proj.weight": "attn"}
_FFN_HF = {"mlp.up_proj.weight": "ffn", "mlp.gate_proj.weight": "ffn", "mlp.down_proj.weight": "ffn"}
# `self_attn.gate_proj` is the LINEAR layer's z projection, not an attention tensor: a
# full-attention layer's gate is the second half of the fused `q_proj`.
_LIN_HF = {"linear_attn.qkv_proj.weight": "linear", "self_attn.gate_proj.weight": "linear",
           "linear_attn.ssm_out_proj.weight": "linear_out"}
_MOE_HF = {"mlp.up_exps_proj.weight": "experts", "mlp.gate_exps_proj.weight": "experts",
           "mlp.down_exps_proj.weight": "experts",
           "mlp.share_up_exps_proj.weight": "shared", "mlp.share_gate_exps_proj.weight": "shared",
           "mlp.share_down_exps_proj.weight": "shared"}
ROLE_TENSORS: dict[str, dict[str, str]] = {
    "qwen36moe": {**_ATTN_HF, **_LIN_HF, **_MOE_HF},
    "qwen35": {**_ATTN_HF, **_LIN_HF, **_FFN_HF},
}
# LFM2's short-conv block has the same two roles a DeltaNet layer does -- a fused input
# projection and an output projection -- so it reuses them rather than adding roles that
# would appear in every family's quant map.
ROLE_TENSORS["lfm2"] = {**_ATTN_HF, **_FFN_HF,
                        "shortconv.in_proj.weight": "linear",
                        "shortconv.out_proj.weight": "linear_out"}
for _f in ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2"):
    ROLE_TENSORS[_f] = {**_ATTN_HF, **_FFN_HF}
# GPT-OSS's routed experts, under the names q4nx-build writes them
# (utilities/q4nx-build/configs/gpt-oss.json). No shared expert and no dense FFN.
# The split names are what a GGUF source carries; q4nx-build's post_gpt_oss_process deletes
# all three and writes the fused tensor, so a real container only ever shows the last one.
ROLE_TENSORS["gptoss"] = {**_ATTN_HF, "ffn_up_exps.weight": "experts",
                          "ffn_gate_exps.weight": "experts", "ffn_down_exps.weight": "experts",
                          "ffn_gate_up_down_exps.weight": "experts"}

_GGUF_BLOCK = re.compile(r"^blk\.\d+\.")
_GGUF_ROLE = {"attn_q.weight": "attn", "attn_k.weight": "attn", "attn_v.weight": "attn",
              "attn_output.weight": "attn",
              "ffn_up.weight": "ffn", "ffn_gate.weight": "ffn", "ffn_down.weight": "ffn",
              "ffn_up_exps.weight": "experts", "ffn_gate_exps.weight": "experts",
              "ffn_down_exps.weight": "experts",
              "ffn_up_shexp.weight": "shared", "ffn_gate_shexp.weight": "shared",
              "ffn_down_shexp.weight": "shared",
              "ssm_in.weight": "linear", "attn_gate.weight": "linear", "ssm_out.weight": "linear_out"}
GGUF_TYPE_FORMAT = {"Q8_0": "q8", "Q4_1": "q4_1", "Q4_0": "q4_1"}


def _collapse(found: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Only the roles that are not at the default; sorted, so the map is canonical."""
    return {r: f for r, (f, _) in sorted(found.items()) if f != DEFAULT_QUANT}


def quant_map_from_chunk_sizes(family: str, chunk_bytes: Mapping[str, int],
                               dtypes: Mapping[str, str] | None = None) -> dict[str, str]:
    """tensor name -> quantized chunk size -> the role map.

    Only the roles that are NOT q4_1 come back, so a stock container derives `{}` and the
    spec keeps hashing as the bare string. A role whose tensors disagree is refused naming
    the tensor that broke it -- half a projection at q8 is not something to guess about.

    `dtypes` separates the two formats that share the 2560-byte chunk: GPT-OSS stores its
    attention projections and its head as q4_1 and its fused expert tensor as MXFP4, both at
    2560, and only the dtype tells them apart. Without it a 2560 chunk reads as q4_1, which
    is right for every caller that only has integer tensors.

    A tensor whose role the packer PLACES, at a chunk size this reader does not know, is
    refused by name. It used to be skipped, which left the role sitting at the q4_1 default
    and handed back a map describing a container that does not exist."""
    table = ROLE_TENSORS.get(family)
    if table is None:
        raise SpecError(f"no tensor-role table for family {family!r} (have {sorted(ROLE_TENSORS)})")
    found: dict[str, tuple[str, str]] = {}
    for name, ch in chunk_bytes.items():
        role = table.get(_LAYER_PREFIX.sub("", name))
        if role is None:
            continue                      # not a projection we place
        ch = int(ch)
        fmt = CHUNK_FORMAT.get(ch)
        if fmt is None and ch in AMBIGUOUS_CHUNK:
            by_dtype = AMBIGUOUS_CHUNK[ch]
            dt = dtypes.get(name) if dtypes is not None else None
            fmt = by_dtype.get(dt)
            if fmt is None:
                raise SpecError(
                    f"{name} fills the {role!r} role at {ch}-byte quant chunks, where the byte "
                    f"count does not name the format: {', '.join(f'{k} is {v}' for k, v in by_dtype.items())}. "
                    f"Got dtype {dt!r}; pass the container's dtypes to tell them apart")
        if fmt is None:
            raise SpecError(f"{name} fills the {role!r} role at {ch}-byte quant chunks, which "
                            f"this reader does not know (it reads {sorted(CHUNK_FORMAT)} and "
                            f"{sorted(AMBIGUOUS_CHUNK)}); refusing rather than assuming {DEFAULT_QUANT}")
        prev = found.get(role)
        if prev is None:
            found[role] = (fmt, name)
        elif prev[0] != fmt:
            raise SpecError(f"{name} is {fmt} but {prev[1]} is {prev[0]}: the {role!r} role must be "
                            f"one format across the model")
    return _collapse(found)


def quant_map_from_gguf_types(family: str, tensor_types: Mapping[str, str]) -> dict[str, str]:
    """GGUF tensor name -> ggml type name -> the role map (the same rules)."""
    table = ROLE_TENSORS.get(family)
    if table is None:
        raise SpecError(f"no tensor-role table for family {family!r} (have {sorted(ROLE_TENSORS)})")
    roles = set(table.values())
    found: dict[str, tuple[str, str]] = {}
    for name, ty in tensor_types.items():
        role = _GGUF_ROLE.get(_GGUF_BLOCK.sub("", name))
        fmt = GGUF_TYPE_FORMAT.get(str(ty).upper())
        if role is None or role not in roles or fmt is None:
            continue
        prev = found.get(role)
        if prev is None:
            found[role] = (fmt, name)
        elif prev[0] != fmt:
            raise SpecError(f"{name} is {fmt} but {prev[1]} is {prev[0]}: the {role!r} role must be "
                            f"one format across the model")
    return _collapse(found)
