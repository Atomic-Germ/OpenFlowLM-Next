"""Write a safetensors file one tensor at a time, byte-identical to safetensors.torch.save_file.

save_file needs every tensor in memory at once -- 19 GB for a 27B container. This writer takes
them as they are produced and keeps only small ones (under IMMEDIATE bytes) until their place
in the file comes up. A big tensor can also arrive as row bands (write_rows), each band going
straight to disk.

save_file's layout, which this reproduces: the tensors in the order (dtype, descending in
safetensors' dtype order; then name), packed back to back with no padding; an 8-byte
little-endian header length; the header as compact JSON in that same order, padded with spaces
to a multiple of 8. A big tensor is written when it arrives, so everything that sorts before it
must have arrived already; a tensor that arrives after its place was passed raises instead of
producing a different layout. The header's size is known only at close(), so the data is
written after a reserved gap and moved down onto the header's end there, in place.
"""
from __future__ import annotations

import atexit
import json
import os

import torch

# save_file's dtype order, highest first (measured against safetensors 0.8.0 for these dtypes).
_ORDER = ("I64", "F64", "F32", "I32", "BF16", "F16", "I16", "I8", "U8", "BOOL")
_NAMES = {torch.int64: "I64", torch.float64: "F64", torch.float32: "F32", torch.int32: "I32",
          torch.bfloat16: "BF16", torch.float16: "F16", torch.int16: "I16", torch.int8: "I8",
          torch.uint8: "U8", torch.bool: "BOOL"}


def _dtype(t: torch.Tensor) -> str:
    if t.dtype not in _NAMES:
        raise TypeError(f"SafetensorsStream: no verified safetensors order for {t.dtype}")
    return _NAMES[t.dtype]


def _key(name: str, dtype: str):
    return _ORDER.index(dtype), name


class SafetensorsStream:
    GAP = 4 << 20          # room reserved for the header (a 27B container's is ~113 KB)
    IMMEDIATE = 1 << 20    # tensors at least this big are written as soon as they arrive
    COPY = 64 << 20        # chunk for moving the data down at close()

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w+b")
        self.f.seek(self.GAP)
        self.pending: dict[str, torch.Tensor] = {}
        self.entries = []          # (name, dtype, shape, begin, end), in file order
        self.end = 0               # data bytes written
        atexit.register(self._discard)   # a run that dies leaves no partial file behind

    def __setitem__(self, name: str, tensor: torch.Tensor):
        if tensor.nbytes < self.IMMEDIATE:
            if self.entries and _key(name, _dtype(tensor)) <= self._last():
                raise ValueError(f"{name} arrived after its place in the file was passed")
            self.pending[name] = tensor
        else:
            self.write_rows(name, [tensor])

    def write_rows(self, name: str, parts):
        """One tensor from consecutive row bands (dim-0 slices, in order), written as they arrive."""
        dtype = shape = begin = None
        rows = 0
        for p in parts:
            p = p.contiguous()
            if dtype is None:
                dtype, shape = _dtype(p), tuple(p.shape[1:])
                self._flush_before(_key(name, dtype))
                begin = self.end
            elif _dtype(p) != dtype or tuple(p.shape[1:]) != shape:
                raise ValueError(f"{name}: row band {tuple(p.shape)} {p.dtype} does not continue the tensor")
            self._write_bytes(p)
            rows += p.shape[0]
        if dtype is None:
            raise ValueError(f"{name}: no row bands")
        self.entries.append((name, dtype, [rows, *shape], begin, self.end))

    def close(self):
        self._flush_before(None)
        header = {name: {"dtype": d, "shape": list(s), "data_offsets": [b, e]}
                  for name, d, s, b, e in self.entries}
        hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
        hb += b" " * (-len(hb) % 8)
        base = 8 + len(hb)
        if base > self.GAP:
            raise ValueError(f"header of {len(hb)} B exceeds the reserved {self.GAP} B")
        self._finish(hb, base)
        atexit.unregister(self._discard)

    # ------------------------------------------------------------------ internals
    def _last(self):
        name, dtype = self.entries[-1][:2]
        return _key(name, dtype)

    def _flush_before(self, key):
        """Write every pending tensor that sorts before `key` (all of them for None), in order."""
        for k, name in sorted((_key(n, _dtype(t)), n) for n, t in self.pending.items()):
            if key is not None and k >= key:
                break
            t = self.pending.pop(name)
            if self.entries and k <= self._last():
                raise ValueError(f"{name} arrived after its place in the file was passed")
            begin = self.end
            self._write_bytes(t.contiguous())
            self.entries.append((name, _dtype(t), list(t.shape), begin, self.end))
        if key is not None and self.entries and key <= self._last():
            raise ValueError(f"{key[1]} arrived after its place in the file was passed")

    def _discard(self):
        if not self.f.closed:
            self.f.close()
            os.remove(self.path)

    def _write_bytes(self, t: torch.Tensor):
        self.f.write(t.reshape(-1).view(torch.uint8).numpy())
        self.end += t.nbytes

    def _finish(self, hb: bytes, base: int):
        """Move the data from GAP down to `base` (front to back, so nothing unread is
        overwritten), then write the header in front of it."""
        f, done = self.f, 0
        while done < self.end:
            n = min(self.COPY, self.end - done)
            f.seek(self.GAP + done)
            chunk = f.read(n)
            f.seek(base + done)
            f.write(chunk)
            done += n
        f.truncate(base + self.end)
        f.seek(0)
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        f.close()
