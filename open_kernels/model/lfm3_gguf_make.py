"""Derive the NPU's model.gguf from LiquidAI's LFM3-1.7B-QAD-Q4_0.gguf (OPEN-FAMILY-LFM3).

Two tensors change, every other byte is copied:

  * `token_embd.weight` (the tied head and the embedding) is Q6_K in the QAD file. The open runtime reads the
    embedding rows on the host and the head through lm_head_q8, neither of which has a Q6_K reader, so the table
    is re-encoded as Q8_0 (llama.cpp's quantize_row_q8_0 reference: d = amax / 127 as f16, q = round(x / d)). The
    8-bit codes keep the Q6_K values to <= 1/254 of a block's max.
  * `blk.N.attn_gate.weight` ([2048, 16] Q4_0: one sigmoid gate value a head) becomes [2048, 2048]: each head's row
    repeated head_dim times, so the gate runs as an ordinary q-width projection and attn.h's per-element gate
    (ATTN_GATE) applies it. Q4_0 rows are independent, so the repetition is exact.

The result is NOT loadable by llama.cpp (the gate's shape differs); the llama.cpp references run on the original.

    python lfm3_gguf_make.py LFM3-1.7B-QAD-Q4_0.gguf model.gguf
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lfm3_gguf_reference import decode_q6k  # noqa: E402

F32, F16, Q4_0, Q8_0, Q6_K = 0, 1, 2, 8, 14
ALIGN = 32


def _skip_value(f, t: int) -> None:
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in fixed:
        f.seek(fixed[t], 1)
    elif t == 8:
        n, = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
    elif t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        for _ in range(n):
            _skip_value(f, et)
    else:
        raise ValueError(f"gguf value type {t}")


def read_infos(path):
    f = open(path, "rb")
    magic, ver, nt, nk = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF" and ver == 3, (magic, ver)
    for _ in range(nk):
        n, = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
        t, = struct.unpack("<I", f.read(4))
        _skip_value(f, t)
    kv_end = f.tell()
    infos = []
    for _ in range(nt):
        n, = struct.unpack("<Q", f.read(8))
        name = f.read(n).decode()
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        infos.append({"name": name, "dims": list(dims), "type": ty, "off": off})
    info_end = f.tell()
    base = (info_end + ALIGN - 1) // ALIGN * ALIGN
    return infos, kv_end, base


def nbytes(ty: int, n: int) -> int:
    if ty == F32:
        return 4 * n
    if ty == F16:
        return 2 * n
    if ty == Q4_0:
        return n // 32 * 18
    if ty == Q8_0:
        return n // 32 * 34
    if ty == Q6_K:
        return n // 256 * 210
    raise ValueError(f"type {ty}")


def quant_q8_0(x: np.ndarray) -> bytes:
    """llama.cpp quantize_row_q8_0_ref over rows of 32: d = amax / 127 (f16), q = round(x / d)."""
    b = x.reshape(-1, 32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
    q = np.rint(b * inv[:, None]).astype(np.int8)
    out = np.empty((b.shape[0], 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def make(src: str, dst: str, head_dim: int = 128) -> None:
    infos, kv_end, base = read_infos(src)
    mm = np.memmap(src, np.uint8, "r")
    head = mm[:kv_end].tobytes()
    new = []
    for t in infos:
        n = int(np.prod(t["dims"]))
        raw = mm[base + t["off"]:base + t["off"] + nbytes(t["type"], n)]
        dims, ty = t["dims"], t["type"]
        if t["name"] == "token_embd.weight":
            assert ty == Q6_K, ty
            x = decode_q6k(raw).reshape(dims[1], dims[0])
            data, ty = quant_q8_0(x), Q8_0
        elif t["name"].endswith(".attn_gate.weight"):
            assert ty == Q4_0 and dims[1] * head_dim > dims[1], (ty, dims)
            rows = np.asarray(raw).reshape(dims[1], dims[0] // 32 * 18)
            data = np.repeat(rows, head_dim, axis=0).tobytes()
            dims = [dims[0], dims[1] * head_dim]
        else:
            data = raw.tobytes()
        new.append((t["name"], dims, ty, data))
    # tensor infos with the new offsets, each tensor's data on a 32-byte boundary
    off, infos_b, offs = 0, bytearray(), []
    for name, dims, ty, data in new:
        offs.append(off)
        nm = name.encode()
        infos_b += struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos_b += struct.pack("<IQ", ty, off)
        off += (len(data) + ALIGN - 1) // ALIGN * ALIGN
    with open(dst, "wb") as f:
        f.write(head)
        f.write(infos_b)
        f.write(b"\0" * (-f.tell() % ALIGN))
        for (_, _, _, data), o in zip(new, offs):
            assert f.tell() % ALIGN == 0
            f.write(data)
            f.write(b"\0" * (-len(data) % ALIGN))
    print(f"wrote {dst}: {len(new)} tensors, {Path(dst).stat().st_size} bytes")


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
