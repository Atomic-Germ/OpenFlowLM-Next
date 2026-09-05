"""Read FLM's `.q4nx` weight container (format 1.0.2: q4_1 chunks).

The container is a safetensors file: an 8-byte header length, a JSON header of
tensor name -> {dtype, shape, data_offsets}, then the data. BF16/F32 tensors are
plain row-major. Quantized tensors are packed chunks of 8192 values:

  q4 chunk, 5120 B: 256 bf16 scales `d`, 256 bf16 mins `m`, then 4096 B of
  nibbles, block size 32 along the input dim, 16-lane interleaved --
    nibble[(r//16)*4096 + bc*512 + i*16 + (r%16)] = element (row r, col bc*32+i)
  q8 chunk, 8704 B (lm_head only): 256 bf16 scales, then 8192 int8.

In the file, chunk f of a [out, in] tensor covers rows 32*(f//ncol) and cols
256*(f%ncol) -- plain raster. (The NPU weight pools reorder those chunks; that
permutation lives in pools.py, which reads the raw bytes from here.)

Adapted from phlegm's tools/kernel-interp/q4nx.py, trimmed to the 1.0.2 path:
the 1.0.3 container packs Q4_K superblocks and needs a different dequant, so
this refuses it rather than reading it wrong.
"""
from __future__ import annotations

import json
import mmap
import struct

import numpy as np

CHUNK_Q4 = 5120
CHUNK_Q8 = 8704


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


class Q4NX:
    def __init__(self, path):
        self.path = str(path)
        self.f = open(self.path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.data_base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors = {k: v for k, v in self.header.items() if k != "__metadata__"}
        self.chunk_bytes = self._chunk_bytes()

    def _chunk_bytes(self):
        """1.0.2 packs q4_1 in 5120 B chunks; 1.0.3 packs Q4_K in 4736 B ones.
        Nothing in the header records the version, so read it off a tensor."""
        for k, v in self.tensors.items():
            if v.get("dtype") == "I8" and k != "lm_head.weight":
                cb = v["shape"][-1]
                if cb != CHUNK_Q4:
                    raise RuntimeError(
                        f"{self.path}: {cb}-byte quant chunks (FLM 1.0.3 / Q4_K?); this reader "
                        f"handles the 1.0.2 q4_1 container only")
                return cb
        raise RuntimeError(f"{self.path}: no quantized tensors in the header")

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

    def dq_tile(self, raw_bytes, out_dim, in_dim):
        """Raw q4 chunk bytes -> [out, in] f32, in the file's raster order."""
        chunks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, CHUNK_Q4)
        w = dq_chunks_q4_1(chunks).reshape(-1, 32, 256)
        ncol = in_dim // 256
        W = np.empty((out_dim, in_dim), np.float32)
        for f in range(w.shape[0]):
            W[32 * (f // ncol): 32 * (f // ncol) + 32, 256 * (f % ncol): 256 * (f % ncol) + 256] = w[f]
        return W

    def matmul_w(self, name, out_dim, in_dim):
        return self.dq_tile(self.raw(name), out_dim, in_dim)

    def expert_w(self, layer, kind, e):
        """One expert's `kind` ('up' | 'gate' | 'down') matrix, dequantized."""
        stride = 128 * self.chunk_bytes
        b = np.frombuffer(self.raw(f"model.layer.{layer}.mlp.{kind}_exps_proj.weight"), dtype=np.uint8)
        out_dim, in_dim = (2048, 512) if kind == "down" else (512, 2048)
        return self.dq_tile(b[e * stride:(e + 1) * stride], out_dim, in_dim)

    def shared_w(self, layer, kind):
        out_dim, in_dim = (2048, 512) if kind == "down" else (512, 2048)
        return self.matmul_w(f"model.layer.{layer}.mlp.share_{kind}_exps_proj.weight", out_dim, in_dim)

    def lmhead_logits(self, hn, block=4096):
        """Stream the q8 lm_head against hidden hn[2048] -> logits[vocab]."""
        lmb = np.frombuffer(self.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, CHUNK_Q8)
        nch = lmb.shape[0]
        hn = np.asarray(hn, np.float32)
        logits = np.zeros(self.tensors["lm_head.weight"]["shape"][0] * 32, np.float32)
        r = np.arange(32)[:, None, None]
        bc = np.arange(8)[None, :, None]
        i = np.arange(32)[None, None, :]
        p = ((r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)).reshape(-1)
        j = (bc * 32 + r + 0 * i).reshape(-1)
        for c0 in range(0, nch, block):
            ce = min(c0 + block, nch)
            d = bf16_to_f32(np.ascontiguousarray(lmb[c0:ce, :512]).view(np.uint16))
            qq = np.ascontiguousarray(lmb[c0:ce, 512:]).view(np.int8)
            w = (qq[:, p].reshape(ce - c0, 32, 8, 32).astype(np.float32)
                 * d[:, j].reshape(ce - c0, 32, 8, 32)).reshape(ce - c0, 32, 256)
            for c in range(c0, ce):
                logits[32 * (c // 8): 32 * (c // 8) + 32] += w[c - c0] @ hn[256 * (c % 8): 256 * (c % 8) + 256]
        return logits
