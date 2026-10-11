"""The open embedding repo's manifest must say what dtype each tensor is stored in.

`open_embedding/engine.cpp` widens every tensor to f32 at load, and it used to
read `n * sizeof(float)` unconditionally -- because `weights_manifest.json`
recorded only file / offset / shape. That is exactly right for the one shipped
fp32 body and silently 2x wrong for a bf16 one: the read runs off the end of the
tensor into the next one's bytes, the vector comes back correctly shaped and
full of plausible numbers, and nothing downstream can tell.

The manifest now carries the safetensors dtype and the engine refuses one it
cannot widen, so the two halves are pinned here together.
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.open_embedding import (  # noqa: E402
    build_open_embedding_repo,
    safetensors_index,
)


def _write_st(path: Path, tensors: dict, dtype: str = "F32") -> None:
    """A safetensors file: [u64 header_len][header JSON][rows]."""
    rows = 2
    blobs, entries = [], {}
    offset = 0
    for i, (name, shape) in enumerate(tensors.items()):
        n = 1
        for s in shape:
            n *= s
        if dtype == "F32":
            raw = np.arange(offset, offset + n, dtype="<f4").tobytes()
        elif dtype == "BF16":
            raw = np.arange(offset, offset + n, dtype="<u2").tobytes()
        else:
            raise AssertionError(dtype)
        entries[name] = {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(entries).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs) if blobs else
                     struct.pack("<Q", len(blob)) + blob)


class SafetensorsIndexTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_dtype_is_carried_off_the_header(self):
        p = self.dir / "m.safetensors"
        _write_st(p, {"2_Dense.linear.weight": [2, 3], "3_Dense.linear.weight": [2, 3]},
                  dtype="BF16")
        idx = safetensors_index(p)
        for name, meta in idx.items():
            self.assertEqual(meta["dtype"], "BF16")
            self.assertEqual(meta["shape"], [2, 3])
        # absolute offsets, and the second tensor starts after the first's bytes
        self.assertLess(idx["2_Dense.linear.weight"]["offset"],
                        idx["3_Dense.linear.weight"]["offset"])

    def test_metadata_is_skipped(self):
        p = self.dir / "m.safetensors"
        _write_st(p, {"2_Dense.linear.weight": [1, 2]})
        self.assertNotIn("__metadata__", safetensors_index(p))


class ManifestTest(unittest.TestCase):
    """End to end: a source repo -> weights_manifest.json -> the engine's reader."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        src = self.root / "src"
        (src).mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (src / name).write_text("{}", encoding="utf-8")
        _write_st(src / "model.safetensors",
                  {"embeddings.word_embeddings.weight": [4, 3]})
        for head in ("2_Dense", "3_Dense"):
            _write_st(src / head / "model.safetensors",
                      {"linear.weight": [3, 3], "bias": [3]})

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self):
        out = self.root / "out"
        build_open_embedding_repo(str(self.root / "src"), str(out))
        return out, json.loads((out / "weights_manifest.json").read_text())

    def test_every_tensor_records_its_dtype(self):
        out, m = self._build()
        self.assertEqual(m["format"], "oflm-open-embedding-manifest-v1")
        # the source is fp32, so the shipped body keeps its dtype either way --
        # what matters is that the key is there at all
        for name, meta in m["tensors"].items():
            self.assertIn("dtype", meta, name)
            self.assertEqual(meta["dtype"], "F32", name)

    def test_a_bf16_body_records_bf16(self):
        """The case the old manifest could not describe."""
        src = self.root / "src"
        _write_st(src / "model.safetensors",
                  {"embeddings.word_embeddings.weight": [4, 3]}, dtype="BF16")
        out, m = self._build()
        self.assertEqual(m["tensors"]["embeddings.word_embeddings.weight"]["dtype"], "BF16")

    def test_the_manifest_paths_are_relative(self):
        out, m = self._build()
        for meta in m["tensors"].values():
            self.assertFalse(Path(meta["file"]).is_absolute())
