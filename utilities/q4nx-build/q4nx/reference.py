"""Dependency-light reference oracle for open causal-LM engines.

This is the Phase 0 acceptance gate for the open Gemma3 text engine
(``docs/plans/open_gemma3_text_plan.md``). It produces prompt -> logit/token
fixtures that the C++ engine must reproduce.

Why NumPy instead of ``transformers``:

* The ROCm torch build in this workspace segfaults on ``from_pretrained``, and
  NumPy has no ``bfloat16`` dtype, so ``safetensors.numpy`` refuses the file.
  Neither is a reason to compromise the oracle.
* bf16 is decoded by bit-shifting into fp32 (``uint16 << 16`` viewed as
  ``float32``) — the same operation the C++ engine performs. The oracle is
  therefore aligned with the engine at the bit level.
* The result has no torch/transformers dependency, so the community can run it
  anywhere.

The oracle is deliberately an *independent* implementation. Its correctness is
established two ways: (1) it produces coherent completions, and (2) it is
cross-checked against the existing closed ``gemma_text_npu`` engine, which is
the current production reference.

Numerics: fp32 throughout. The checkpoint is natively bf16, so an fp32
reference isolates implementation error from unavoidable bf16 storage error.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REFERENCE_FORMAT = "flm-open-causal-reference-v1"

# Short prompts plus one deliberately long prompt (>512 tokens) to exercise
# Gemma3's sliding-window attention path.
DEFAULT_PROMPTS: Sequence[str] = (
    "The capital of France is",
    "In a distant galaxy, a small robot discovered",
    "def fibonacci(n):",
    "The three laws of robotics are",
    "Water boils at 100 degrees Celsius because",
)
LONG_PROMPT_REPEATS = 60
LONG_PROMPT_BASE = (
    "The history of computing is a long sequence of incremental advances. "
    "Each generation of machines built upon the ideas of the previous one, "
    "and each new idea opened possibilities that had not been imagined before. "
)


# ---------------------------------------------------------------- io helpers


def bf16_to_f32(raw: bytes) -> "object":
    """Decode bf16 bytes to a float32 array via the uint16<<16 bit trick."""
    import numpy as np

    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _dtype_size(dtype: str) -> int:
    return {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "U32": 4,
        "I16": 2,
        "U16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }[dtype]


def _numpy_dtype(dtype: str):
    import numpy as np

    return {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
        "U32": np.uint32,
        "I16": np.int16,
        "U16": np.uint16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }[dtype]


def read_safetensors(path: Path, as_float32: bool = True) -> Dict[str, "object"]:
    """Read a safetensors file into NumPy arrays, with bf16 support.

    Layout: [u64 header_len][header JSON][tensor data]. bf16 tensors are
    widened to fp32; every other supported dtype is returned natively (or
    cast to fp32 when ``as_float32`` is set and the dtype is floating point).
    """
    import numpy as np

    data = path.read_bytes()
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    base = 8 + header_len
    out: Dict[str, "object"] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = data[base + start : base + end]
        dtype = meta["dtype"]
        if dtype == "BF16":
            arr = bf16_to_f32(raw)
        else:
            arr = np.frombuffer(raw, dtype=_numpy_dtype(dtype))
            if as_float32 and arr.dtype in (np.float16, np.float64):
                arr = arr.astype(np.float32)
        out[name] = arr.reshape(meta["shape"])
    return out


# ------------------------------------------------------------------ reference


class Gemma3Reference:
    """NumPy reference for Gemma3 text models (prefill forward pass).

    Mirrors HF ``Gemma3ForCausalLM``: embedding scaling by ``sqrt(hidden)``,
    RMSNorm with ``(1 + w)``, dual-base RoPE, hybrid sliding/global attention,
    GQA, ``gelu_pytorch_tanh`` MLP, and a final norm into a tied LM head.
    """

    def __init__(self, model_dir: str):
        import numpy as np

        self.np = np
        self.dir = Path(model_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.w = read_safetensors(self.dir / "model.safetensors")

        c = self.cfg
        self.hidden = c["hidden_size"]
        self.inter = c["intermediate_size"]
        self.n_layers = c["num_hidden_layers"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c.get("head_dim", self.hidden // self.n_heads)
        self.vocab = c["vocab_size"]
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.sliding = c.get("sliding_window", 512)
        self.pattern = c.get("sliding_window_pattern", 6)
        self.rope_theta = c.get("rope_theta", 1e6)
        self.rope_local = c.get("rope_local_base_freq", 1e4)
        self.q_scalar = c.get("query_pre_attn_scalar", self.head_dim)

        self.embed_scale = float(self.hidden) ** 0.5
        self.attn_scale = 1.0 / float(self.q_scalar) ** 0.5
        self.groups = self.n_heads // self.n_kv

        # Gemma3 ships no `layer_types` key; derive global layers. In HF
        # Gemma3 every `sliding_window_pattern`-th layer is global.
        self.layer_types = [
            "full_attention" if (i + 1) % self.pattern == 0 else "sliding_attention"
            for i in range(self.n_layers)
        ]

    # -- primitives ------------------------------------------------------

    def rmsnorm(self, x, weight):
        np = self.np
        var = (x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True)
        xhat = x / np.sqrt(var + self.eps)
        return (xhat * (1.0 + weight)).astype(np.float32)

    def rope_tables(self, T: int, theta: float):
        """Return (cos, sin) of shape [T, head_dim] using the HF convention."""
        np = self.np
        half = self.head_dim // 2
        inv = 1.0 / (theta ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim))
        freqs = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def apply_rope(self, x, cos, sin):
        """x: [T, heads, head_dim] -> rotated, HF rotate_half convention."""
        np = self.np
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rot = np.concatenate([-x2, x1], axis=-1)
        return (x * cos[:, None, :] + rot * sin[:, None, :]).astype(np.float32)

    def gelu_tanh(self, x):
        np = self.np
        k = 0.7978845608028654
        return (0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x**3)))).astype(np.float32)

    # -- forward ---------------------------------------------------------

    def prefill(self, token_ids: Sequence[int]):
        """Full-sequence forward; returns logits for the last position."""
        np = self.np
        T = len(token_ids)
        h = self.w["model.embed_tokens.weight"][list(token_ids)].astype(np.float32)
        h = h * self.embed_scale

        cos_g, sin_g = self.rope_tables(T, self.rope_theta)
        cos_l, sin_l = self.rope_tables(T, self.rope_local)

        # Causal mask and sliding-window band mask.
        idx = np.arange(T)
        causal = idx[:, None] >= idx[None, :]
        band = np.abs(idx[:, None] - idx[None, :]) < self.sliding

        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            is_global = self.layer_types[i] == "full_attention"
            cos, sin = (cos_g, sin_g) if is_global else (cos_l, sin_l)
            allowed = causal if is_global else (causal & band)

            x = self.rmsnorm(h, self.w[p + "input_layernorm.weight"])

            q = (x @ self.w[p + "self_attn.q_proj.weight"].T).reshape(T, self.n_heads, self.head_dim)
            k = (x @ self.w[p + "self_attn.k_proj.weight"].T).reshape(T, self.n_kv, self.head_dim)
            v = (x @ self.w[p + "self_attn.v_proj.weight"].T).reshape(T, self.n_kv, self.head_dim)

            q = self.rmsnorm(q, self.w[p + "self_attn.q_norm.weight"])
            k = self.rmsnorm(k, self.w[p + "self_attn.k_norm.weight"])

            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)

            # Attention: [heads, T, head_dim]
            qh = q.transpose(1, 0, 2)
            kh = k.transpose(1, 0, 2)
            vh = v.transpose(1, 0, 2)

            # GQA: broadcast each kv head to its group of query heads.
            kh_e = np.repeat(kh, self.groups, axis=0)
            vh_e = np.repeat(vh, self.groups, axis=0)

            scores = (qh @ kh_e.transpose(0, 2, 1)) * self.attn_scale
            mask = allowed[None, :, :]
            scores = np.where(mask, scores, np.float32(-np.inf))

            scores = scores - scores.max(axis=-1, keepdims=True)
            exp = np.exp(scores)
            # Rows fully masked would be nan; the causal diagonal guarantees at
            # least one allowed position, but guard anyway.
            denom = exp.sum(axis=-1, keepdims=True)
            denom = np.where(denom == 0, np.float32(1.0), denom)
            att = exp / denom

            o = (att @ vh_e).transpose(1, 0, 2).reshape(T, self.n_heads * self.head_dim)
            o = o @ self.w[p + "self_attn.o_proj.weight"].T

            h = h + self.rmsnorm(o, self.w[p + "post_attention_layernorm.weight"])

            x = self.rmsnorm(h, self.w[p + "pre_feedforward_layernorm.weight"])
            gate = self.gelu_tanh(x @ self.w[p + "mlp.gate_proj.weight"].T)
            up = x @ self.w[p + "mlp.up_proj.weight"].T
            h = h + self.rmsnorm((gate * up) @ self.w[p + "mlp.down_proj.weight"].T,
                                 self.w[p + "post_feedforward_layernorm.weight"])

        h_last = self.rmsnorm(h[-1], self.w["model.norm.weight"])
        logits = self.w["model.embed_tokens.weight"] @ h_last  # tied LM head
        return logits.astype(np.float32)

    def greedy_continue(self, token_ids: Sequence[int], steps: int) -> List[int]:
        """Naive re-forward decode; slow but obviously correct as a reference."""
        ids = list(token_ids)
        out = []
        for _ in range(steps):
            logits = self.prefill(ids)
            nxt = int(self.np.argmax(logits))
            out.append(nxt)
            ids.append(nxt)
        return out


# ------------------------------------------------------------------- driver


def _logit_stats(np, v) -> Dict[str, float]:
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "l2_norm": float(np.sqrt((v.astype(np.float64) ** 2).sum())),
        "logsumexp": float(np.log(np.exp(v.astype(np.float64)).sum())),
    }


def _encode(tokenizer_json: Path, text: str) -> List[int]:
    """Encode with the HF `tokenizers` package if available, else empty."""
    try:
        from tokenizers import Tokenizer
    except Exception:
        return []
    tok = Tokenizer.from_file(str(tokenizer_json))
    return tok.encode(text).ids


def generate_reference(
    model_dir: str,
    output_path: str,
    prompts: Optional[Sequence[str]] = None,
    top_k: int = 100,
    continuation_tokens: int = 16,
) -> dict:
    """Run the NumPy reference and write a fixture JSON."""
    import numpy as np

    ref = Gemma3Reference(model_dir)
    prompt_list = list(DEFAULT_PROMPTS) + [LONG_PROMPT_BASE * LONG_PROMPT_REPEATS]
    if prompts:
        prompt_list = list(prompts)

    tokenizer_json = Path(model_dir) / "tokenizer.json"
    entries = []
    for text in prompt_list:
        ids = _encode(tokenizer_json, text)
        if not ids:
            raise RuntimeError(
                "the `tokenizers` package is required to record prompt token ids"
            )
        logits = ref.prefill(ids)
        top_idx = np.argsort(logits)[::-1][:top_k]

        # Greedy continuation re-forwards the whole sequence each step, which is
        # quadratic. Long prompts exist to exercise sliding-window attention and
        # are already covered by their prefill logits, so skip the decode there.
        if len(ids) > 1000:
            continuation: List[int] = []
        else:
            continuation = ref.greedy_continue(ids, continuation_tokens)

        entries.append(
            {
                "text": text,
                "token_ids": ids,
                "token_count": len(ids),
                "argmax_token_id": int(np.argmax(logits)),
                "top_logits": [[int(i), float(logits[i])] for i in top_idx],
                "logit_stats": _logit_stats(np, logits),
                "greedy_continuation": {"ids": continuation},
            }
        )

    payload = {
        "format": REFERENCE_FORMAT,
        "model_dir": str(model_dir),
        "reference_dtype": "float32",
        "engine": "numpy-independent-reference",
        "config": {
            k: ref.cfg.get(k)
            for k in (
                "model_type",
                "architectures",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "sliding_window",
                "sliding_window_pattern",
                "rope_theta",
                "rope_local_base_freq",
                "rms_norm_eps",
                "query_pre_attn_scalar",
            )
            if k in ref.cfg
        },
        "derived": {
            "layer_types": ref.layer_types,
            "global_layer_indices": [i for i, t in enumerate(ref.layer_types) if t == "full_attention"],
            "embed_scale": ref.embed_scale,
            "attn_scale": ref.attn_scale,
            "gqa_groups": ref.groups,
        },
        "prompts": entries,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return {
        "output": str(out),
        "prompt_count": len(entries),
        "token_counts": [e["token_count"] for e in entries],
    }
