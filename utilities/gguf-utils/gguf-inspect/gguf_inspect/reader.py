"""Thin, dependency-light wrapper over the pinned GGUF python reader.

Only metadata and tensor inventory are read. No tensor payload is decoded and
the NPU is never opened. The pinned reader is vendored under third_party and is
located relative to this file so the tool never depends on a developer checkout
of ../llama.cpp being present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional


class GgufReaderError(RuntimeError):
    """Raised when a file cannot be opened as a GGUF or fails validation."""


class GgufTensor:
    def __init__(self, name: str, dtype: str, shape: List[int],
                 n_elements: int, n_bytes: int) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.n_elements = n_elements
        self.n_bytes = n_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "n_elements": self.n_elements,
            "n_bytes": self.n_bytes,
        }


class GgufFile:
    def __init__(self, path: str, version: int, endian: str,
                 byte_order: str, fields: dict, tensors: List[GgufTensor]) -> None:
        self.path = path
        self.version = version
        self.endian = endian
        self.byte_order = byte_order
        self._fields = fields
        self.tensors = tensors
        self.tensor_count = len(tensors)

    @property
    def field_names(self) -> List[str]:
        return list(self._fields.keys())

    def get_field(self, key: str) -> Any:
        if key not in self._fields:
            return None
        return self._fields[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "endian": self.endian,
            "byte_order": self.byte_order,
            "tensor_count": self.tensor_count,
            "field_names": sorted(self._fields.keys()),
            "tensors": [t.to_dict() for t in self.tensors],
        }


def _locate_reader() -> Any:
    cached = _locate_reader.cache
    if cached is not None:
        return cached

    def _usable(mod) -> bool:
        return getattr(mod, "GGUFReader", None) is not None

    # 1. Already importable -- a plain `pip install gguf` (or
    #    `pip install git+https://github.com/ggml-org/llama.cpp`, which is how CI
    #    gets it, since the submodule is not checked out there). This is the
    #    cheapest path and needs no path munging.
    try:
        import gguf as _installed  # type: ignore
        if _usable(_installed):
            _locate_reader.cache = _installed
            return _installed
    except Exception:  # noqa: BLE001
        pass

    # 2. Vendored in-tree. This file is
    # utilities/gguf-utils/gguf-inspect/gguf_inspect/reader.py, so five parents
    # up is the REPOSITORY root. Four stopped at utilities/, which made the first
    # candidate utilities/third_party/... -- a path that exists in no checkout,
    # so every inspection raised before the GGUF was even opened. Search upward
    # for the ancestors that could carry third_party/, covering both the nested
    # and the flat layout.
    here = Path(__file__).resolve().parent
    candidates = [
        r / "third_party" / "Guanaco" / "llama.cpp" / "gguf-py"
        for r in list(here.parents)[:6]
    ]
    candidates.append(Path("/tmp/opencode/guanaco-llama-clean/gguf-py"))
    for cand in candidates:
        if not cand.is_dir():
            continue
        sys.path.insert(0, str(cand))
        try:
            import gguf  # type: ignore
            if _usable(gguf):
                _locate_reader.cache = gguf
                return gguf
        except Exception:  # noqa: BLE001
            continue
    raise GgufReaderError(
        "could not locate the GGUF reader: neither an importable `gguf` "
        "(pip install git+https://github.com/ggml-org/llama.cpp) nor "
        "third_party/Guanaco/llama.cpp/gguf-py"
    )


_locate_reader.cache: Any = None


def _open(path: str) -> GgufFile:
    if not os.path.isfile(path):
        raise GgufReaderError(f"not a file: {path}")
    gguf = _locate_reader()
    try:
        raw = gguf.GGUFReader(path)
    except Exception as exc:  # noqa: BLE001
        raise GgufReaderError(f"{path}: {exc}") from exc

    version = int(raw.get_field("GGUF.version").contents()) \
        if raw.get_field("GGUF.version") else None
    byte_order = raw.byte_order
    if byte_order in ("I", "i", "N", "n", None):
        endian = "native"
    elif byte_order in ("S", "B", ">"):
        endian = "big"
    else:
        endian = "little"

    fields: dict = {}
    for key, field in raw.fields.items():
        try:
            val = field.contents()
        except Exception:  # noqa: BLE001
            val = None
        fields[key] = val

    tensors: List[GgufTensor] = []
    for t in raw.tensors:
        try:
            shape = [int(d) for d in list(t.shape)]
        except Exception:  # noqa: BLE001
            shape = []
        tensors.append(GgufTensor(
            name=str(t.name),
            dtype=str(t.tensor_type.name),
            shape=shape,
            n_elements=int(t.n_elements),
            n_bytes=int(t.n_bytes),
        ))
    return GgufFile(path, version, endian, byte_order, fields, tensors)


def open_gguf(path: str) -> GgufFile:
    return _open(path)
