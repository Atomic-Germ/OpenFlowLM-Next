"""Cut the GPT-OSS byte fixtures out of the shipped container.

    fixtures/gptoss_mxfp4_chunks.npz   eight whole MXFP4 chunks (OPEN-QUANT-MXFP4)
    fixtures/gptoss_expert_slabs.npz   the bias every column-block-0 chunk carries in its
                                       padding, for all 69 slabs of two experts, plus the
                                       named bias tensors it is checked against
                                       (OPEN-PACK-EXPERT-ORDER)

    python specs/open-engine/tests/make_gptoss_fixtures.py [model_dir]

Default model dir is %FLM_MODEL_PATH%\\models\\GPT-OSS-20B-NPU2. The container is 14.4 GB
and only a few hundred KB of it is read, by seek -- but it is also not in CI, which is why
these fixtures exist. Re-running against the same container must reproduce them byte for
byte; the mxfp4 one predates this script and is checked for exactly that.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
MXFP4 = FIXTURES / "gptoss_mxfp4_chunks.npz"
SLABS = FIXTURES / "gptoss_expert_slabs.npz"

LAYER = 0
# expert 0 and the last one: the stride between experts is 31 tensors wide at that end, so a
# wrong stride cannot land on the right bias by accident the way a neighbour might.
EXPERTS = (0, 31)
# (slab, column block, quarter) of the eight whole chunks, spanning gate, up and down.
PICKS = [(0, 0, 0), (1, 0, 0), (46, 0, 0), (0, 7, 2),
         (23, 11, 1), (60, 22, 3), (2, 3, 0), (47, 15, 2)]


class Container:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.base = 8 + n

    def shape(self, name):
        return self.hdr[name]["shape"]

    def raw(self, name):
        a, b = self.hdr[name]["data_offsets"]
        self.f.seek(self.base + a)
        return self.f.read(b - a)

    def slice(self, name, off, n):
        a, _ = self.hdr[name]["data_offsets"]
        self.f.seek(self.base + a + off)
        return self.f.read(n)


def default_model_dir() -> Path:
    base = os.environ.get("FLM_MODEL_PATH")
    if not base:
        raise SystemExit("set FLM_MODEL_PATH or pass the model directory")
    return Path(base) / "models" / "GPT-OSS-20B-NPU2"


def main(argv):
    model_dir = Path(argv[1]) if len(argv) > 1 else default_model_dir()
    c = Container(model_dir / "model.q4nx")
    sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))
    from model import mxfp4 as M

    W = f"model.layers.{LAYER}.ffn_gate_up_down_exps.weight"
    E, NP, NQ, RG, CH = c.shape(W)
    print(f"{W}: {[E, NP, NQ, RG, CH]}")

    def chunk_at(e, p, q, f, n=CH, off=0):
        idx = ((e * NP + p) * NQ + q) * RG + f
        return c.slice(W, idx * CH + off, n)

    # --- the MXFP4 chunk fixture -------------------------------------------------
    chunks = np.stack([np.frombuffer(chunk_at(0, p, q, f), np.uint8) for p, q, f in PICKS])
    expect = np.stack([M.decode_chunk(ch) for ch in chunks]).astype(np.float32)
    picks = np.array(PICKS, np.int32)
    note = np.array([f"layer {LAYER} expert 0 of {model_dir.name}; "
                     f"(slab, col_block, f) in picks"], dtype="<U67")
    _write(MXFP4, dict(chunks=chunks, expect=expect, picks=picks, note=note))

    # --- the slab-order fixture --------------------------------------------------
    # Only bytes 128..191 of each chunk: that is the whole of what carries the bias, and
    # all 69 slabs x 4 quarters of a whole chunk would be 700 KB for the same assertion.
    bias_bytes = np.stack([
        np.stack([
            np.stack([np.frombuffer(chunk_at(e, p, 0, f, n=64, off=128), np.uint8)
                      for f in range(RG)])
            for p in range(NP)])
        for e in EXPERTS])
    named = np.stack([
        np.stack([np.frombuffer(c.raw(f"model.layers.{LAYER}.mlp.experts.{r}_proj_bias"),
                                "<u2").reshape(E, -1)[e]
                  for r in ("gate", "up", "down")])
        for e in EXPERTS])
    _write(SLABS, dict(
        bias_bytes=bias_bytes, named=named,
        experts=np.array(EXPERTS, np.int32),
        shape=np.array([E, NP, NQ, RG, CH], np.int32),
        note=np.array([f"layer {LAYER} of {model_dir.name}: chunk bytes 128..191 of every "
                       f"column-block-0 chunk, and the named *_proj_bias rows (bf16) they "
                       f"are checked against"], dtype=object)))


def _write(path: Path, d: dict) -> None:
    if path.exists():
        old = np.load(path, allow_pickle=True)
        same = set(old.files) == set(d) and all(np.array_equal(old[k], d[k]) for k in d)
        print(f"{path.name}: {'unchanged' if same else 'REGENERATED (contents differ)'}")
        if same:
            return
    np.savez_compressed(path, **d)
    print(f"{path.name}: {path.stat().st_size} B")


if __name__ == "__main__":
    main(sys.argv)
