"""Read FLM's `.q4nx` weight container (format 1.0.2: q4_1 chunks).

The container is a safetensors file: an 8-byte header length, a JSON header of
tensor name -> {dtype, shape, data_offsets}, then the data. BF16/F32 tensors are
plain row-major. Quantized tensors are packed chunks of 8192 values:

  q4 chunk, 5120 B: 256 bf16 scales `d`, 256 bf16 mins `m`, then 4096 B of
  nibbles, block size 32 along the input dim, 16-lane interleaved --
    nibble[(r//16)*4096 + bc*512 + i*16 + (r%16)] = element (row r, col bc*32+i)
  q8 chunk, 8704 B (lm_head only): 256 bf16 scales, then 8192 int8.

In the file, chunk f of a [out, in] tensor covers rows 32*(f//ncol) and cols
256*(f%ncol) -- plain raster. (The NPU weight pools reorder those chunks; the
recipe's packing plan says how -- recipes/qwen36moe.py, applied by recipes/pack.py
over the raw bytes read here.)

The chunk format is PER TENSOR. The stock Qwen3.6-35B keeps only its lm_head at
q8; its fine-tunes (Darwin, Grug, BigBang, Aquila-mini, Ornith 1.5, and
Atomic-Germ's own NPU2 mirror) pack attention, linear-attention and shared-expert
projections at q8 and only the routed experts at q4_1; Qwen3.5 dense containers
store `ssm_out_proj` and alpha / beta at q8. So nothing is refused at open: each
tensor is classified from its own shape, and `dq_tile` reads either format. A
chunk size that is neither 5120 nor 8704 is refused when that tensor is read,
naming it (Q4_K is 4736; 1280 / 2560 are a smaller chunk geometry).

Which reading a q8 tensor gets is the packer's decision, mirrored here so a slice
comparison always measures the KERNELS and never the weights:

  * `native_q8(name)` -- true for the projections this kernel set streams at q8
    (the plan's `q8_perm` ops, OPEN-QUANT-Q8). Those read as the container's own
    q8 values, because that is exactly what the pool holds.
  * `requant_q8` (default True) -- every other q8 tensor reads as the q4_1 the
    packer writes for it (the re-quantising fallback, OPEN-PACK-PLAN).

`make_decode.py --requant` forces the whole run -- spec, plan, pools and this
reader -- onto the fallback, which is the A/B against the q8 path.

Adapted from phlegm's tools/kernel-interp/q4nx.py.
"""
from __future__ import annotations

import json
import mmap
import struct

import numpy as np

CHUNK_Q4 = 5120
CHUNK_Q8 = 8704


def _requant_q4_1(chunks):
    """recipes.pack's q8 -> q4_1 (the packer's own arithmetic; one implementation, not two)."""
    try:
        from recipes.pack import requant_q4_1
    except ImportError:
        import pathlib
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from recipes.pack import requant_q4_1
    return requant_q4_1(chunks)


def bf16_to_f32(u16):
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    """Round-to-nearest-even bf16 encode -> uint16."""
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def dq_chunks_q4_1(chunks):
    """[n, 5120] raw chunk bytes -> [n, 32, 8, 32] f32 (row, block, lane)."""
    nch = chunks.shape[0]
    meta = bf16_to_f32(np.ascontiguousarray(chunks[:, :1024]).view(np.uint16))
    d, mn = meta[:, :256], meta[:, 256:]
    q = chunks[:, 1024:]
    n = np.empty((nch, 8192), dtype=np.float32)
    n[:, 0::2] = q & 0xF
    n[:, 1::2] = q >> 4
    r = np.arange(32)[:, None, None]
    bc = np.arange(8)[None, :, None]
    i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
    j = bc * 32 + r + 0 * i
    vals = n[:, p.reshape(-1)].reshape(nch, 32, 8, 32)
    return vals * d[:, j.reshape(-1)].reshape(nch, 32, 8, 32) + mn[:, j.reshape(-1)].reshape(nch, 32, 8, 32)


def dq_chunks_q8(chunks):
    """[n, 8704] raw q8 chunk bytes -> [n, 32, 8, 32] f32 (row, block, lane).

    Same 32-row x 256-K tile as a q4_1 chunk, and the same code raster; what differs is
    int8 codes with one bf16 scale per (block, row) and no min. Used by the lm_head and,
    on a Qwen3.5 container, by `linear_attn.ssm_{out,alpha,beta}_proj`."""
    chunks = np.asarray(chunks, np.uint8).reshape(-1, CHUNK_Q8)
    nch = chunks.shape[0]
    d = bf16_to_f32(np.ascontiguousarray(chunks[:, :512]).view(np.uint16))
    q = np.ascontiguousarray(chunks[:, 512:]).view(np.int8)
    r = np.arange(32)[:, None, None]
    bc = np.arange(8)[None, :, None]
    i = np.arange(32)[None, None, :]
    p = ((r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)).reshape(-1)
    j = (bc * 32 + r + 0 * i).reshape(-1)
    return q[:, p].reshape(nch, 32, 8, 32).astype(np.float32) * d[:, j].reshape(nch, 32, 8, 32)


class Q4NX:
    def __init__(self, path):
        self.path = str(path)
        self.f = open(self.path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.data_base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors = {k: v for k, v in self.header.items() if k != "__metadata__"}
        self.chunk_bytes = CHUNK_Q4        # the POOL's q4_1 chunk, what the plan's offsets are in
        self.requant_q8 = True             # read a q8 tensor as the q4_1 the packer puts on the NPU
        self.native_q8 = lambda name: False  # ... except these, which the pool holds at q8

    def requant_of(self, name):
        """Does the pool hold this tensor re-quantized to q4_1, or at its own q8?"""
        return self.requant_q8 and not self.native_q8(name)

    def chunk_bytes_of(self, name):
        """The quantized chunk size `name` is stored in (5120 = q4_1, 8704 = q8), 0 when the
        tensor is not quantized. recipes/pack.py probes for this method."""
        t = self.tensors[name]
        return t["shape"][-1] if t.get("dtype") == "I8" and t.get("shape") else 0

    def _refuse(self, name, cb):
        guess = ("Q4_K (FLM 1.0.3), which needs a different dequant" if cb == 4736 else
                 f"a smaller chunk geometry ({cb * 8192 // CHUNK_Q4} values per chunk instead of 8192)"
                 if cb in (1280, 2560) else "not a chunk format this reader knows")
        raise RuntimeError(f"{self.path}: {name} has {cb}-byte quant chunks; this reader handles "
                           f"{CHUNK_Q4} (q4_1) and {CHUNK_Q8} (q8) only -- {cb} is {guess}")

    def raw(self, name):
        o0, o1 = self.tensors[name]["data_offsets"]
        return self.mm[self.data_base + o0: self.data_base + o1]

    def bf16(self, name):
        t = self.tensors[name]
        assert t["dtype"] == "BF16", t
        return bf16_to_f32(np.frombuffer(self.raw(name), dtype=np.uint16).reshape(t["shape"]))

    def f32(self, name):
        t = self.tensors[name]
        assert t["dtype"] == "F32", t
        return np.frombuffer(self.raw(name), dtype=np.float32).reshape(t["shape"])

    def embed(self, token, hidden=2048):
        o0 = self.tensors["model.embed_tokens.weight"]["data_offsets"][0]
        b = self.data_base + o0 + token * hidden * 2
        return bf16_to_f32(np.frombuffer(self.mm[b: b + hidden * 2], dtype=np.uint16)).astype(np.float64)

    def dq_tile(self, raw_bytes, out_dim, in_dim, chunk=CHUNK_Q4, requant=None):
        """Raw chunk bytes -> [out, in] f32, in the file's raster order.

        `chunk` is the format those bytes are in. A q8 tensor reads as the packer's q4_1
        when `requant` (the default: the values the NPU holds), else as its own q8."""
        if requant is None:
            requant = self.requant_q8
        b = np.frombuffer(raw_bytes, dtype=np.uint8)
        if chunk == CHUNK_Q4:
            w = dq_chunks_q4_1(b.reshape(-1, CHUNK_Q4))
        elif chunk == CHUNK_Q8 and requant:
            w = dq_chunks_q4_1(_requant_q4_1(b.reshape(-1, CHUNK_Q8)))
        elif chunk == CHUNK_Q8:
            w = dq_chunks_q8(b)
        else:
            self._refuse("<raw bytes>", chunk)
        w = w.reshape(-1, 32, 256)
        ncol = in_dim // 256
        W = np.empty((out_dim, in_dim), np.float32)
        for f in range(w.shape[0]):
            W[32 * (f // ncol): 32 * (f // ncol) + 32, 256 * (f % ncol): 256 * (f % ncol) + 256] = w[f]
        return W

    def matmul_w(self, name, out_dim, in_dim):
        cb = self.chunk_bytes_of(name)
        if cb not in (CHUNK_Q4, CHUNK_Q8):
            self._refuse(name, cb)
        return self.dq_tile(self.raw(name), out_dim, in_dim, cb, self.requant_of(name))

    def expert_w(self, layer, kind, e):
        """One expert's `kind` ('up' | 'gate' | 'down') matrix, dequantized."""
        name = f"model.layer.{layer}.mlp.{kind}_exps_proj.weight"
        cb = self.chunk_bytes_of(name)
        if cb not in (CHUNK_Q4, CHUNK_Q8):
            self._refuse(name, cb)
        stride = 128 * cb
        b = np.frombuffer(self.raw(name), dtype=np.uint8)
        out_dim, in_dim = (2048, 512) if kind == "down" else (512, 2048)
        return self.dq_tile(b[e * stride:(e + 1) * stride], out_dim, in_dim, cb, self.requant_of(name))

    def shared_w(self, layer, kind):
        out_dim, in_dim = (2048, 512) if kind == "down" else (512, 2048)
        return self.matmul_w(f"model.layer.{layer}.mlp.share_{kind}_exps_proj.weight", out_dim, in_dim)

    def lmhead_logits(self, hn, block=4096):
        """Stream the q8 lm_head against hidden hn[hidden] -> logits[vocab]. `self.hidden`
        (set by the caller; 2048 by default, the 27B's) says how many 256-wide k-tiles a
        chunk row spans -- 16 of them for a Qwen3.5 dense model at HID 4096."""
        lmb = np.frombuffer(self.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, CHUNK_Q8)
        nch = lmb.shape[0]
        hn = np.asarray(hn, np.float32)
        rows = nch // (getattr(self, "hidden", 2048) // 256) * 32
        logits = np.zeros(rows, np.float32)
        r = np.arange(32)[:, None, None]
        bc = np.arange(8)[None, :, None]
        i = np.arange(32)[None, None, :]
        p = ((r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)).reshape(-1)
        j = (bc * 32 + r + 0 * i).reshape(-1)
        ncol = getattr(self, "hidden", 2048) // 256
        for c0 in range(0, nch, block):
            ce = min(c0 + block, nch)
            d = bf16_to_f32(np.ascontiguousarray(lmb[c0:ce, :512]).view(np.uint16))
            qq = np.ascontiguousarray(lmb[c0:ce, 512:]).view(np.int8)
            w = (qq[:, p].reshape(ce - c0, 32, 8, 32).astype(np.float32)
                 * d[:, j].reshape(ce - c0, 32, 8, 32)).reshape(ce - c0, 32, 256)
            for c in range(c0, ce):
                r0, k0 = 32 * (c // ncol), 256 * (c % ncol)
                logits[r0:r0 + 32] += w[c - c0] @ hn[k0:k0 + 256]
        return logits
