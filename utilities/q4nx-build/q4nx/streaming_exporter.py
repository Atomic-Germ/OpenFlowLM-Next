"""Streaming safetensors writer for low-RAM pack builds.

The classic export path accumulates every converted tensor into
``conv.q4nx_tensors`` and hands the full dict to ``torch.save_file`` at the
end. For an 80 GB-class GGUF that means the entire packed model must live in
RAM. This writer emits a well-formed safetensors file incrementally: each
packed tensor is written to disk the moment it is produced, so the build's
peak RSS is one tensor plus bookkeeping rather than the whole model.

The format constraint is the header: safetensors wants ``{8-byte length,
JSON header, raw bytes}`` up front, but the JSON needs the offsets of every
tensor, which are only known after all tensors are written. This writer
therefore reserves a fixed, generously-sized header region up front
(``length`` + space-padded JSON) and patches the exact JSON in place when
the file is closed. The padding is JSON whitespace, so any conforming
reader (C++ or Python) accepts it.
"""
from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Tuple

import torch

_DTYPE_MAP = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.bool: "BOOL",
}


class StreamingSafetensorsWriter:
    """Append-only safetensors writer.

    Layout: ``u64(header_reserved) + space-padded JSON (header_reserved bytes)
    + raw tensor bytes``. On close, the reserved region is overwritten with the
    true JSON (plus spaces to ``header_reserved``); ``data_offsets`` inside the
    JSON are relative to the raw-bytes region, matching every conforming
    safetensors reader. ``__metadata__`` is not emitted, mirroring
    ``torch.save_file(..., metadata=None)``.
    """

    def __init__(self, path: str, header_reserved: int = 4 * 1024 * 1024) -> None:
        self.path = path
        self.header_reserved = header_reserved
        self._fh = open(path, "w+b")
        self._fh.write(struct.pack("<Q", header_reserved))
        self._fh.write(b" " * header_reserved)
        self._pos = 8 + header_reserved
        self._entries: List[Tuple[str, str, Tuple[int, ...], int, int]] = []

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().contiguous().cpu()
        try:
            dt = _DTYPE_MAP[t.dtype]
        except KeyError:
            raise ValueError(f"unsupported torch dtype for safetensors: {t.dtype}")
        if t.dtype == torch.bfloat16:
            raw = t.view(torch.uint16).numpy().tobytes()
        else:
            raw = t.numpy().tobytes()
        start = self._pos
        self._fh.write(raw)
        self._pos += len(raw)
        self._entries.append((name, dt, tuple(t.shape), start, self._pos))

    def _data_start(self) -> int:
        return 8 + self.header_reserved

    def close(self) -> None:
        if self._fh is None:
            return
        entries: Dict[str, dict] = {}
        data_start = self._data_start()
        for name, dt, shape, start, end in self._entries:
            entries[name] = {
                "dtype": dt,
                "shape": list(shape),
                "data_offsets": [start - data_start, end - data_start],
            }
        json_bytes = json.dumps(entries).encode("utf-8")
        if len(json_bytes) > self.header_reserved:
            self._fh.close()
            raise RuntimeError(
                f"safetensors header for {self.path} needs {len(json_bytes)} bytes; "
                f"reserved {self.header_reserved}. Re-run with a larger reserved "
                f"header."
            )
        self._fh.seek(8)
        self._fh.write(json_bytes + b" " * (self.header_reserved - len(json_bytes)))
        self._fh.close()
        self._fh = None

    def __enter__(self) -> "StreamingSafetensorsWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class StreamingPack(dict):
    """dict-shaped sink that writes each incoming tensor to disk immediately.

    ``conv.q4nx_tensors[name] = tensor`` becomes ``writer.add_tensor`` plus a
    key-count bookkeeping entry (value placeholder is the name itself), so
    ``len(q4nx_tensors)`` and ``in`` keep working and the packed tensor bytes
    are not retained in the process.
    """

    def __init__(self, writer: StreamingSafetensorsWriter) -> None:
        super().__init__()
        self._writer = writer

    def __setitem__(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            self._writer.add_tensor(name, tensor)
            super().__setitem__(name, name)
        else:
            super().__setitem__(name, tensor)
