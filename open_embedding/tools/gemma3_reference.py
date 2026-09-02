"""Dependency-free numpy reference implementation of google/embeddinggemma-300m.

Reproduces the official sentence-transformers pipeline in float32 using only
numpy + the `tokenizers` package:

    Gemma3 transformer
        -> mean pooling over all token hidden states (include_prompt=True)
        -> Dense 768 -> 3072 (Identity)
        -> Dense 3072 -> 768 (Identity)
        -> L2 normalize

Weights come from the local HF cache / oracle dir. The forward pass mirrors
`transformers` Gemma3TextModel exactly:

  * embeddings scaled by sqrt(hidden_size)
  * full layers  (index 5, 11, 17, 23): unmasked bidirectional attention
  * sliding layers: attention restricted to |q - k| < sliding_window
    (ExMasks use_bidirectional_attention => no causal masking anywhere)
  * q/k RMSNormed, rotary per layer_type (theta 1e6 full / 1e4 local),
    scaling = query_pre_attn_scalar**-0.5 = 1/16 on every layer
  * RMSNorm weights applied as (1 + weight); every op stays in float32

Validation (anchor from the model card): with prompt "query" for the query and
prompt "document" for the four documents, official cosine similarities are

    [0.3011, 0.6359, 0.4930, 0.4889]

`--validate-anchor` reproduces that exact table.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent.parent / "oracle"
HF_SNAP = Path(
    "/home/atomic-germ/.cache/huggingface/hub/models--google--embeddinggemma-300m/"
    "snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
)

DTYPE_MAP = {"F32": np.float32, "F16": np.float16, "BF16": np.float32, "I64": np.int64}

ANCHOR_QUERY = "Which planet is known as the Red Planet?"
ANCHOR_DOCS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
ANCHOR_EXPECTED = [0.3011, 0.6359, 0.4930, 0.4889]

PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
}


# ---------------------------------------------------------------- weights

def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Minimal safetensors reader (header JSON + raw tensor blobs)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
        blob = f.read()
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        dtype = DTYPE_MAP[meta["dtype"]]
        shape = tuple(meta["shape"])
        off = meta["data_offsets"]
        arr = np.frombuffer(blob[off[0]:off[1]], dtype=dtype).reshape(shape)
        if dtype != np.float32:
            arr = arr.astype(np.float32)
        out[name] = arr
    return out


class ReferenceModel:
    def __init__(self):
        self.cfg = json.loads(Path(
            "/home/atomic-germ/.config/flm/models/Embedding-Gemma-300M-NPU2/config.json").read_text())
        self.w = read_safetensors(HF_SNAP / "model.safetensors")
        self.w2 = read_safetensors(ORACLE_DIR / "weights" / "2_Dense.safetensors")["linear.weight"]
        self.w3 = read_safetensors(ORACLE_DIR / "weights" / "3_Dense.safetensors")["linear.weight"]
        self.embed_scale = math.sqrt(self.cfg["hidden_size"])
        self.scaling = self.cfg.get("query_pre_attn_scalar", 1.0) ** -0.5
        self.eps = self.cfg["rms_norm_eps"]
        self.head_dim = self.cfg["head_dim"]
        self.n_heads = self.cfg["num_attention_heads"]
        self.n_kv = self.cfg["num_key_value_heads"]
        self.sliding_window = self.cfg["sliding_window"]
        self.layer_types = self.cfg["layer_types"]
        self.rope_bases = {}
        for lt in sorted(set(self.layer_types)):
            theta = self.cfg["rope_local_base_freq"] if lt == "sliding_attention" else self.cfg["rope_theta"]
            dims = np.arange(0, self.head_dim, 2, dtype=np.float64)
            self.rope_bases[lt] = (1.0 / np.power(np.float64(theta), dims / self.head_dim)).astype(np.float32)

    # ----------------------------------------------------------- pieces

    @staticmethod
    def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        mean = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        norm = x * np.sqrt(1.0 / (mean + float(eps))).astype(np.float32)
        return norm * (1.0 + weight.astype(np.float32))

    def rope(self, positions: np.ndarray, layer_type: str) -> tuple[np.ndarray, np.ndarray]:
        inv = self.rope_bases[layer_type]  # [128]
        freqs = positions[:, None] * inv[None, :]  # [T,128]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [T,256]
        return np.cos(emb), np.sin(emb)

    def attention_mask(self, seq_len: int, layer_type: str) -> np.ndarray:
        if layer_type == "full_attention":
            return np.zeros((seq_len, seq_len), dtype=np.float32)
        dist = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
        allowed = dist < self.sliding_window
        mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
        mask[allowed] = 0.0
        return mask

    def iter_h(self, token_ids: np.ndarray):
        T = token_ids.shape[0]
        h = self.w["embed_tokens.weight"][token_ids] * self.embed_scale  # [T,768] fp32
        rotary = {}
        for lt in self.rope_bases:
            rotary[lt] = self.rope(np.arange(T, dtype=np.float32), lt)
        masks = {}
        for lt in set(self.layer_types):
            masks[lt] = self.attention_mask(T, lt)
        yield "embed", h.copy()

        for i in range(self.cfg["num_hidden_layers"]):
            lt = self.layer_types[i]
            ln = f"layers.{i}."

            residual = h
            x = self.rmsnorm(h, self.w[ln + "input_layernorm.weight"], self.eps)

            q = x @ self.w[ln + "self_attn.q_proj.weight"].T  # [T,768]
            k = x @ self.w[ln + "self_attn.k_proj.weight"].T  # [T,256]
            v = x @ self.w[ln + "self_attn.v_proj.weight"].T  # [T,256]

            q = q.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)  # [H,T,D]
            k = k.reshape(T, self.n_kv, self.head_dim).transpose(1, 0, 2)    # [KV,T,D]
            v = v.reshape(T, self.n_kv, self.head_dim).transpose(1, 0, 2)

            q = self.rmsnorm(q, self.w[ln + "self_attn.q_norm.weight"], self.eps)
            k = self.rmsnorm(k, self.w[ln + "self_attn.k_norm.weight"], self.eps)

            cos, sin = rotary[lt]
            cos, sin = cos[None, :, :], sin[None, :, :]
            # rotate_half
            half = self.head_dim // 2
            q_half = np.concatenate([-q[:, :, half:], q[:, :, :half]], axis=-1)
            k_half = np.concatenate([-k[:, :, half:], k[:, :, :half]], axis=-1)
            q = q * cos + q_half * sin
            k = k * cos + k_half * sin

            scores = np.matmul(q, k.transpose(0, 2, 1)) * self.scaling  # [H,T,T]
            scores = scores + masks[lt][None, :, :]
            att = self._softmax(scores)  # fp32
            out = np.matmul(att, v)  # [H,T,D]
            out = out.transpose(1, 0, 2).reshape(T, self.n_heads * self.head_dim)
            out = out @ self.w[ln + "self_attn.o_proj.weight"].T

            x = self.rmsnorm(out, self.w[ln + "post_attention_layernorm.weight"], self.eps)
            h = residual + x

            residual = h
            x = self.rmsnorm(h, self.w[ln + "pre_feedforward_layernorm.weight"], self.eps)
            gate = self._gelu_tanh(x @ self.w[ln + "mlp.gate_proj.weight"].T)
            up = x @ self.w[ln + "mlp.up_proj.weight"].T
            out = (gate * up) @ self.w[ln + "mlp.down_proj.weight"].T
            x = self.rmsnorm(out, self.w[ln + "post_feedforward_layernorm.weight"], self.eps)
            h = residual + x
            yield i, h.copy()

        yield "final", self.rmsnorm(h, self.w["norm.weight"], self.eps)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x.astype(np.float64))
        return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)

    @staticmethod
    def _gelu_tanh(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    def head(self, last_hidden: np.ndarray) -> np.ndarray:
        pooled = np.mean(last_hidden, axis=0)  # [768]
        y = pooled @ self.w2.T  # [3072]
        y = y @ self.w3.T      # [768]
        n = np.linalg.norm(y)
        return (y / n).astype(np.float32)

    def embed(self, text: str, prompt: str = "query") -> np.ndarray:
        return self.stages(text, prompt)[3]

    def final_hidden(self, token_ids: np.ndarray) -> np.ndarray:
        for name, h in self.iter_h(token_ids):
            pass
        return h

    def stages(self, text: str, prompt: str = "query") -> list[np.ndarray]:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(HF_SNAP / "tokenizer.json"))
        full = PROMPTS[prompt] + text
        ids = np.asarray(tok.encode(full).ids, dtype=np.int64)
        last_hidden = self.final_hidden(ids)
        pooled = np.mean(last_hidden, axis=0)
        d1 = pooled @ self.w2.T
        d2 = d1 @ self.w3.T
        n = float(np.linalg.norm(d2))
        return [pooled.astype(np.float32), d1.astype(np.float32), d2.astype(np.float32), (d2 / n).astype(np.float32)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="query")
    parser.add_argument("--validate-anchor", action="store_true")
    args = parser.parse_args()

    model = ReferenceModel()
    if args.validate_anchor:
        qv = model.embed(ANCHOR_QUERY, "query")
        dvs = [model.embed(d, "document") for d in ANCHOR_DOCS]
        sims = [float(qv @ dv) for dv in dvs]
        for doc, sim, exp in zip(ANCHOR_DOCS[:3], sims, ANCHOR_EXPECTED):
            pass
        print(f"query: {ANCHOR_QUERY}")
        for doc, sim, exp in zip(ANCHOR_DOCS, sims, ANCHOR_EXPECTED):
            flag = "  OK" if abs(sim - exp) < 0.002 else "  MISMATCH"
            print(f"  {sim:7.4f} (official {exp}){flag}  {doc[:55]}")
        return
    if args.text is None:
        parser.error("--text required unless --validate-anchor")
    v = model.embed(args.text, args.prompt)
    print(f"[{args.prompt}] {args.text}")
    print(json.dumps([round(float(x), 6) for x in v]))


if __name__ == "__main__":
    sys.exit(main())