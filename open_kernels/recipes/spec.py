"""ModelSpec: the hyperparameter tuple a recipe composes kernels from.

One plain dataclass, JSON on disk, built from either of the two places a model
already publishes its shape: the HF-style `config.json` FLM ships beside the
`.q4nx` container, or a GGUF's metadata (`general.architecture`,
`<arch>.embedding_length`, ...). Anything a recipe needs that is not here is
a recipe constant (a family property), not a model property.

Traces: OPEN-SPEC-DERIVE (specs/open-engine/spec.md).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

LINEAR, FULL, DENSE, DENSE_LOCAL = "linear_attention", "full_attention", "dense", "dense_local"
LAYER_TYPES = (LINEAR, FULL, DENSE, DENSE_LOCAL)      # dense_local: a dense layer with sliding-window attention


class SpecError(ValueError):
    """The metadata cannot be turned into a ModelSpec; the message names the key."""


@dataclass(frozen=True)
class ModelSpec:
    family: str                       # the recipe: "qwen36moe" | "qwen3" | "llama3" | "gemma3" | "hunyuan"
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
    quant: str = "q4_1"
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

    def rope_inv_freq(self, local: bool = False) -> list[float]:
        """The inverse frequency of each rotary pair i < rotary_dim/2: theta^(-2i/rot), with Llama 3's
        wavelength-dependent scaling or a linear factor when `rope_scaling` says so (HF's
        _compute_llama3_parameters / _compute_linear_scaling_rope_parameters). `local`: the
        sliding-window layers' table (Gemma: rope_local_theta, unscaled)."""
        import math
        half = self.rotary_dim // 2
        if local:
            theta = self.rope_local_theta or self.rope_theta
            return [theta ** (-i / half) for i in range(half)]
        inv = [self.rope_theta ** (-i / half) for i in range(half)]
        sc = self.rope_scaling
        if sc and sc.get("rope_type") == "linear":
            return [f / float(sc["factor"]) for f in inv]
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

    def rope_inv_freq(self, local: bool = False) -> list[float]:
        """The inverse frequency of each rotary pair i < rotary_dim/2: theta^(-2i/rot), with Llama 3's
        wavelength-dependent scaling or a linear factor when `rope_scaling` says so (HF's
        _compute_llama3_parameters / _compute_linear_scaling_rope_parameters). `local`: the
        sliding-window layers' table (Gemma: rope_local_theta, unscaled)."""
        import math
        half = self.rotary_dim // 2
        if local:
            theta = self.rope_local_theta or self.rope_theta
            return [theta ** (-i / half) for i in range(half)]
        inv = [self.rope_theta ** (-i / half) for i in range(half)]
        sc = self.rope_scaling
        if sc and sc.get("rope_type") == "linear":
            return [f / float(sc["factor"]) for f in inv]
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

    # ---- serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_types"] = list(self.layer_types)
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
        """From the HF-style config.json FLM ships with a model.

        `real_vocab` is the tokenizer's id count (tokenizer.json); it defaults
        to vocab_size, which for FLM's Qwen3.6 containers is the padded lm_head
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
    """Llama 3: GQA without q/k norms, full RoPE with the llama3 frequency scaling, no gate, silu FFN."""
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
    if cfg.get("tie_word_embeddings"):
        raise SpecError("llama: tied embeddings are not supported (the head must be its own q4 tensor)")
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


HF_FAMILIES = {"qwen3_5_moe": _qwen36moe_hf, "qwen3_next": _qwen36moe_hf, "qwen3": _qwen3_hf, "llama": _llama3_hf,
               "gemma3_text": _gemma3_hf, "gemma3": _gemma3_hf, "hunyuan_v1_dense": _hunyuan_hf}
GGUF_FAMILIES = {"qwen35moe": _qwen36moe_gguf, "qwen3next": _qwen36moe_gguf, "qwen3": _qwen3_gguf, "llama": _llama3_gguf,
                 "gemma3": _gemma3_gguf, "hunyuan-dense": _hunyuan_gguf}
_FAMILY_OF = {_qwen36moe_hf: "qwen36moe", _qwen36moe_gguf: "qwen36moe", _qwen3_hf: "qwen3", _qwen3_gguf: "qwen3",
              _llama3_hf: "llama3", _llama3_gguf: "llama3", _gemma3_hf: "gemma3", _gemma3_gguf: "gemma3",
              _hunyuan_hf: "hunyuan", _hunyuan_gguf: "hunyuan"}


def hf_model_types(family: str) -> list[str]:
    """The config.json model_type values a family's recipe accepts."""
    return sorted(k for k, f in HF_FAMILIES.items() if _FAMILY_OF[f] == family)


def gguf_architectures(family: str) -> list[str]:
    return sorted(k for k, f in GGUF_FAMILIES.items() if _FAMILY_OF[f] == family)
