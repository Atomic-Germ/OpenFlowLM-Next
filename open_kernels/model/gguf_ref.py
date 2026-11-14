"""Read a GGUF's TENSORS (not just its metadata), dequantized, in f32.

The vendored GGUF reader (utilities/gguf-utils) is metadata-only, so every GGUF
gate so far stopped at "the pool packed and the layers ran". This is the other
half: the file's own values, in the layout the reference math wants, so
`replica_qwen35` / `replica_prompt` can be pointed at a GGUF model directory and
compare against the engine the way they compare against a `.q4nx` container.

Two deliberate choices:

  * a weight comes back as the FILE holds it (dequantized), not as the pool will
    hold it. The pool requantizes to q4_1 / q8, which costs ~2% and is what the
    kernels run on -- so a reference built from the file differs from the engine
    by that much BY DESIGN. A difference of orders of magnitude, or a
    correlation near zero, is a bug; a 1e-2 relative error is the quantiser.
    `Q4NX.matmul_w` emulates the pool instead; that is right for a container
    (whose bytes ARE the pool) and impossible for a GGUF without duplicating the
    packer -- which is exactly what a gate must not do.

  * names are llama.cpp's (`blk.3.attn_q.weight`), and the map to the HF-style
    names the reference asks for is `HF_TO_GGUF` below. It is written from
    llama.cpp's conversion, not from `pools.cpp`, so a packer that mis-maps a
    tensor cannot hide behind the same mistake here.

Dims in a GGUF are FASTEST-FIRST: `attn_q.weight` [1024, 4096] is 4096 rows of
1024 (row-major [out][in]), which is the reference's `[out_dim, in_dim]`.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# ggml type ids (ggml.h). The ones this tree's kernels ingest; anything else is
# refused by name rather than guessed.
F32, F16 = 0, 1
Q4_0, Q4_1, Q8_0 = 2, 3, 8
Q4_K, Q5_K, Q6_K = 12, 13, 14
TYPE_NAMES = {F32: "F32", F16: "F16", Q4_0: "Q4_0", Q4_1: "Q4_1", Q8_0: "Q8_0",
              Q4_K: "Q4_K", Q5_K: "Q5_K", Q6_K: "Q6_K"}
# bytes per 256 values (QK_K) or per 32 values, and the block's value count
BLOCK = {F32: (1, 4), F16: (1, 2), Q8_0: (32, 34), Q4_0: (32, 18), Q4_1: (32, 20),
         Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210)}


class GgufError(RuntimeError):
    pass


class Gguf:
    """A GGUF file's header and tensors, read on demand."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self._read_header()

    # ---- header ---------------------------------------------------------
    def _rd(self, fmt):
        n = struct.calcsize(fmt)
        b = self._f.read(n)
        if len(b) != n:
            raise GgufError(f"{self.path.name}: truncated header")
        return struct.unpack(fmt, b)[0]

    def _str(self):
        n = self._rd("<Q")
        return self._f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        if t in (0, 7):
            return self._f.read(1)[0]
        if t == 1:
            return self._rd("<b")
        if t == 2:
            return self._rd("<H")
        if t == 3:
            return self._rd("<h")
        if t == 4:
            return self._rd("<I")
        if t == 5:
            return self._rd("<i")
        if t == 6:
            return self._rd("<f")
        if t == 8:
            return self._str()
        if t == 9:
            et = self._rd("<I")
            n = self._rd("<Q")
            return [self._val(et) for _ in range(n)]
        if t == 10:
            return self._rd("<Q")
        if t == 11:
            return self._rd("<q")
        if t == 12:
            return self._rd("<d")
        raise GgufError(f"{self.path.name}: metadata value type {t} is not known")

    def _read_header(self):
        if self._f.read(4) != b"GGUF":
            raise GgufError(f"{self.path.name}: not a GGUF")
        version = self._rd("<I")
        if version < 2:
            raise GgufError(f"{self.path.name}: GGUF v{version} is not supported")
        n_tensors = self._rd("<Q")
        n_kv = self._rd("<Q")
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            t = self._rd("<I")
            self.kv[k] = self._val(t)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = self._rd("<I")
            dims = [self._rd("<Q") for _ in range(nd)]
            ty = self._rd("<I")
            off = self._rd("<Q")
            self.tensors[name] = (ty, dims, off)
        pos = self._f.tell()
        self.base = (pos + 31) // 32 * 32

    # ---- tensors --------------------------------------------------------
    def tensor(self, name):
        if name not in self.tensors:
            raise GgufError(f"{self.path.name}: no tensor {name!r}")
        return self.tensors[name]

    def raw(self, name, row0=0, rows=1):
        ty, dims, off = self.tensor(name)
        if ty not in BLOCK:
            raise GgufError(f"{name} is {TYPE_NAMES.get(ty, ty)}; this reader knows "
                            f"{sorted(TYPE_NAMES[t] for t in BLOCK)}")
        per_block, block_bytes = BLOCK[ty]
        ne0 = int(dims[0])                     # fastest-first: the row width
        nblocks = ne0 // per_block
        stride = nblocks * block_bytes
        self._f.seek(self.base + off + row0 * stride)
        return ty, ne0, self._f.read(rows * stride), stride

    def dequant(self, name, row0=0, rows=1):
        """Rows [row0, row0+rows) as f32 [rows, ne0]."""
        ty, ne0, buf, stride = self.raw(name, row0, rows)
        if ty == F32:
            return np.frombuffer(buf, np.float32).reshape(rows, ne0).copy()
        if ty == F16:
            return np.frombuffer(buf, np.float16).reshape(rows, ne0).astype(np.float32)
        if ty == Q8_0:
            b = np.frombuffer(buf, np.uint8).reshape(rows, ne0 // 32, 34)
            d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
            q = b[:, :, 2:].view(np.int8).astype(np.float32)
            return (q * d).reshape(rows, ne0)
        if ty == Q4_1:
            b = np.frombuffer(buf, np.uint8).reshape(rows, ne0 // 32, 20)
            d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
            m = b[:, :, 2:4].copy().view(np.float16).astype(np.float32)
            q = b[:, :, 4:20]
            lo = (q & 0xF).astype(np.float32)
            hi = (q >> 4).astype(np.float32)
            out = np.empty((rows, ne0 // 32, 32), np.float32)
            out[:, :, 0:16] = lo * d + m
            out[:, :, 16:32] = hi * d + m
            return out.reshape(rows, ne0)
        if ty == Q4_0:
            b = np.frombuffer(buf, np.uint8).reshape(rows, ne0 // 32, 18)
            d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
            q = b[:, :, 2:18]
            lo = (q & 0xF).astype(np.float32) - 8.0
            hi = (q >> 4).astype(np.float32) - 8.0
            out = np.empty((rows, ne0 // 32, 32), np.float32)
            out[:, :, 0:16] = lo * d
            out[:, :, 16:32] = hi * d
            return out.reshape(rows, ne0)
        if ty == Q6_K:
            return _q6k(buf, rows, ne0)
        if ty == Q4_K:
            return _q4k(buf, rows, ne0)
        if ty == Q5_K:
            return _q5k(buf, rows, ne0)
        raise GgufError(f"{name}: {TYPE_NAMES.get(ty, ty)} is not implemented here")

    def rows(self, name):
        return int(self.tensor(name)[1][1]) if len(self.tensor(name)[1]) > 1 else 1

    def matmul(self, name, x, block=8192):
        """W @ x for a [rows, ne0] tensor, streamed so a 248320-row head is fine."""
        x = np.asarray(x, np.float32)
        n = self.rows(name)
        out = np.empty(n, np.float32)
        for r0 in range(0, n, block):
            r1 = min(n, r0 + block)
            out[r0:r1] = self.dequant(name, r0, r1 - r0) @ x
        return out


# ---- the block dequantizers (llama.cpp's laws, vectorised over blocks) ----

def _q6k(buf, rows, ne0):
    b = np.frombuffer(buf, np.uint8).reshape(rows, ne0 // 256, 210)
    d = b[:, :, 208:210].copy().view(np.float16).astype(np.float32)          # [r, nb, 1]
    ql = b[:, :, 0:128]
    qh = b[:, :, 128:192]
    sc = b[:, :, 192:208].view(np.int8).astype(np.float32)                   # [r, nb, 16]
    out = np.empty((rows, ne0 // 256, 256), np.float32)
    for half in range(2):
        q0 = ql[:, :, half * 64:(half + 1) * 64].astype(np.int32)
        h0 = qh[:, :, half * 32:(half + 1) * 32].astype(np.int32)
        s0 = sc[:, :, half * 8:(half + 1) * 8]
        lo = np.arange(32)
        is_ = lo // 16
        q1 = (q0[:, :, lo] & 0xF) | (((h0[:, :, lo] >> 0) & 3) << 4)
        q2 = (q0[:, :, lo + 32] & 0xF) | (((h0[:, :, lo] >> 2) & 3) << 4)
        q3 = (q0[:, :, lo] >> 4) | (((h0[:, :, lo] >> 4) & 3) << 4)
        q4 = (q0[:, :, lo + 32] >> 4) | (((h0[:, :, lo] >> 6) & 3) << 4)
        base = half * 128
        out[:, :, base + lo + 0] = d * s0[:, :, is_ + 0] * (q1 - 32)
        out[:, :, base + lo + 32] = d * s0[:, :, is_ + 2] * (q2 - 32)
        out[:, :, base + lo + 64] = d * s0[:, :, is_ + 4] * (q3 - 32)
        out[:, :, base + lo + 96] = d * s0[:, :, is_ + 6] * (q4 - 32)
    return out.reshape(rows, ne0)


def _k_scale_min(scales, j):
    """Q4_K/Q5_K's packed 6-bit scale and min for sub-block j (llama.cpp's law)."""
    s = scales.astype(np.int32)
    if j < 4:
        d = s[:, :, j] & 63
        m = s[:, :, j + 4] & 63
    else:
        d = (s[:, :, j + 4] & 0xF) | ((s[:, :, j - 4] >> 6) << 4)
        m = (s[:, :, j + 4] >> 4) | ((s[:, :, j] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)


def _q4k(buf, rows, ne0):
    b = np.frombuffer(buf, np.uint8).reshape(rows, ne0 // 256, 144)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, :, 2:4].copy().view(np.float16).astype(np.float32)
    scales = b[:, :, 4:16]
    qs = b[:, :, 16:144].astype(np.int32)
    out = np.empty((rows, ne0 // 256, 256), np.float32)
    lo = np.arange(32)
    for j in range(4):
        sc1, m1 = _k_scale_min(scales, 2 * j)
        sc2, m2 = _k_scale_min(scales, 2 * j + 1)
        q = qs[:, :, 32 * j:32 * (j + 1)]
        out[:, :, 64 * j + lo] = d * sc1[:, :, None] * (q & 0xF) - dmin * m1[:, :, None]
        out[:, :, 64 * j + 32 + lo] = d * sc2[:, :, None] * (q >> 4) - dmin * m2[:, :, None]
    return out.reshape(rows, ne0)


def _q5k(buf, rows, ne0):
    b = np.frombuffer(buf, np.uint8).reshape(rows, ne0 // 256, 176)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, :, 2:4].copy().view(np.float16).astype(np.float32)
    scales = b[:, :, 4:16]
    qh = b[:, :, 16:48].astype(np.int32)
    qs = b[:, :, 48:176].astype(np.int32)
    out = np.empty((rows, ne0 // 256, 256), np.float32)
    lo = np.arange(32)
    for j in range(4):
        sc1, m1 = _k_scale_min(scales, 2 * j)
        sc2, m2 = _k_scale_min(scales, 2 * j + 1)
        q = qs[:, :, 32 * j:32 * (j + 1)]
        hi1 = (qh[:, :, lo] & (1 << (2 * j))) != 0
        hi2 = (qh[:, :, lo] & (2 << (2 * j))) != 0
        v1 = (q & 0xF) + hi1.astype(np.int32) * 16
        v2 = (q >> 4) + hi2.astype(np.int32) * 16
        out[:, :, 64 * j + lo] = d * sc1[:, :, None] * v1 - dmin * m1[:, :, None]
        out[:, :, 64 * j + 32 + lo] = d * sc2[:, :, None] * v2 - dmin * m2[:, :, None]
    return out.reshape(rows, ne0)


# ---- llama.cpp's names -> the names the reference math asks for --------------
#
# The reference (replica_qwen35) is written against the HF container's names, so
# this is the map a GGUF needs to run through it. Fused tensors are sliced the
# way llama.cpp lays them out (q, then gate where the family fuses them).
HF_TO_GGUF = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "model.layers.{l}.self_attn.q_proj.weight": ("blk.{l}.attn_qkv.weight", (0, None)),
    "model.layers.{l}.self_attn.o_proj.weight": "blk.{l}.attn_output.weight",
    "model.layers.{l}.mlp.up_proj.weight": "blk.{l}.ffn_up.weight",
    "model.layers.{l}.mlp.gate_proj.weight": "blk.{l}.ffn_gate.weight",
    "model.layers.{l}.mlp.down_proj.weight": "blk.{l}.ffn_down.weight",
    "model.layers.{l}.linear_attn.qkv_proj.weight": "blk.{l}.attn_qkv.weight",
    "model.layers.{l}.linear_attn.ssm_out_proj.weight": "blk.{l}.ssm_out.weight",
    "model.layers.{l}.self_attn.gate_proj.weight": "blk.{l}.attn_gate.weight",
}


# ---- the reference's view of a GGUF ----------------------------------------

_LAYER_NORM = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.ssm_conv1d.weight": ("ssm_conv1d.weight", True),   # [channels][taps] in the file
    "linear_attn.ssm_alpha_proj.bf16.weight": "ssm_alpha.weight",
    "linear_attn.ssm_beta_proj.bf16.weight": "ssm_beta.weight",
    "linear_attn.ssm_a": "ssm_a",
    "linear_attn.ssm_dt.bias": "ssm_dt.bias",
    "linear_attn.ssm_norm.weight": "ssm_norm.weight",
    "linear_attn.qkv_proj.weight": "attn_qkv.weight",
    "linear_attn.ssm_out_proj.weight": "ssm_out.weight",
    "self_attn.gate_proj.weight": "attn_gate.weight",
    "self_attn.q_proj.weight": ("attn_q.weight", "qgate"),
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
_TOP = {"model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight"}


def _deinterleave_qgate(w, heads, head_rows):
    """[q_head|gate_head] per head (llama.cpp) -> [all q | all gate] (the pool's order)."""
    order = np.empty(2 * heads * head_rows, int)
    for i in range(heads):
        order[i * head_rows:(i + 1) * head_rows] = i * 2 * head_rows + np.arange(head_rows)
        base = heads * head_rows + i * head_rows
        order[base:base + head_rows] = i * 2 * head_rows + head_rows + np.arange(head_rows)
    return np.ascontiguousarray(w[order])


class GgufWeights:
    """The fp64 reference's weight source for a GGUF model directory.

    Same interface as `Q4NX` for the names `replica_qwen35` asks for, but the
    values are the FILE's (dequantized), not the pool's (requantized): the pool
    law is what the kernels run on, so emulating it here would hide a pack bug
    behind the same law. Expect a ~1e-2 relative difference from the engine; a
    near-zero correlation is the finding.
    """

    def __init__(self, path, spec):
        self.g = Gguf(path)
        self.spec = spec
        self.hidden = spec.hidden
        self._cache = {}

    # names ------------------------------------------------------------------
    def _map(self, hf):
        if hf in _TOP:
            return _TOP[hf], False
        if not hf.startswith("model.layers."):
            raise GgufError(f"no GGUF name for {hf!r}")
        rest = hf[len("model.layers."):]
        layer, _, tail = rest.partition(".")
        if tail not in _LAYER_NORM:
            raise GgufError(f"no GGUF name for {hf!r}")
        m = _LAYER_NORM[tail]
        if isinstance(m, tuple):
            return f"blk.{layer}.{m[0]}", m[1]
        return f"blk.{layer}.{m}", False

    def _get(self, hf):
        if hf in self._cache:
            return self._cache[hf]
        name, transform = self._map(hf)
        w = self.g.dequant(name, 0, self.g.rows(name))
        if transform == "qgate":
            # llama.cpp writes a FUSED q|gate per head ([q_h|gate_h|...]); the pool (and
            # the reference's slicing) wants [all q | all gate]. Same values, cos 0.14 if
            # left alone -- this is the order the container is built in.
            w = _deinterleave_qgate(w, self.spec.num_heads, self.spec.head_dim)
        elif transform:
            w = np.ascontiguousarray(w.T)
        w = w.reshape(-1) if w.shape[0] == 1 else w
        self._cache[hf] = w
        return w

    # the reference's calls ---------------------------------------------------
    def bf16(self, hf):
        return self._get(hf)

    def f32(self, hf):
        return self._get(hf)

    def matmul_w(self, hf, out_dim, in_dim):
        w = self._get(hf)
        if w.shape != (out_dim, in_dim):
            raise GgufError(f"{hf}: the file gives {w.shape}, the reference wants "
                            f"{(out_dim, in_dim)}")
        return w

    def embed(self, token, hidden):
        return self.g.dequant("token_embd.weight", int(token), 1)[0]

    def lmhead_logits(self, hn):
        # tied head: the file has no output.weight, so the head IS token_embd
        name = "output.weight" if "output.weight" in self.g.tensors else "token_embd.weight"
        return self.g.matmul(name, hn)
