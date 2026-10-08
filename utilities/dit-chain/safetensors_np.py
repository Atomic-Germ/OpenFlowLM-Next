r"""Read tensors out of a (sharded) safetensors checkpoint with numpy only, bf16 included.

    st = SafeTensors(r"<model dir>\text_encoder")      # follows model.safetensors.index.json
    w = st.get("model.layers.0.self_attn.q_proj.weight")   # float32 ndarray

The format is an 8-byte little-endian header length, a JSON header {name: {dtype, shape,
data_offsets}}, then the raw data; files are memory-mapped, one tensor copied per get.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64,
       "I32": np.int32, "U8": np.uint8}


class SafeTensors:
    def __init__(self, directory):
        d = Path(directory)
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
            files = sorted(set(wm.values()))
        else:
            files = [p.name for p in d.glob("*.safetensors")]
        self._where = {}
        self._maps = {}
        for f in files:
            with open(d / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            self._maps[f] = (np.memmap(d / f, dtype=np.uint8, mode="r"), 8 + n)
            for name, meta in hdr.items():
                if name != "__metadata__":
                    self._where[name] = (f, meta)

    def names(self):
        return list(self._where)

    def get(self, name: str) -> np.ndarray:
        f, meta = self._where[name]
        mm, base = self._maps[f]
        a, b = meta["data_offsets"]
        raw = np.asarray(mm[base + a:base + b]).view(_DT[meta["dtype"]]).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32)
