# Traces: OPEN-QUANT-Q4K, OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""The converter's q4_1 and q8 writers against our readers, byte for byte.

test_quant_q4k.py already holds `pack_q4k` to `dq_chunks_q4_k`. This does the same
for the two writers that had no reader-side check at all: `_pack_q4nx` (q4_1, the
default for every dense family) and `_pack_q8nx` (q8, used by the lm_head and by
Qwen3.5's ssm_{out,alpha,beta}_proj).

Why it matters: these two are what almost every shipped container is made of. A
transposed plane, a misaligned nibble pair or a wrong stride lands in the pool as
a well-formed chunk of the right size -- so the container loads, every kernel slice
compare passes, and the model decodes garbage. The failure is invisible to any
test that does not read the bytes back with the reader the NPU uses.

No container and no model file: synthetic (scale, min, quant) triples go in and
come back out of the reader, which is what makes this cheap enough for CI.

Scales are chosen bf16-representable so the writer's fp16->bf16 scale cast is
lossless and any residual left is pure LAYOUT. An ordinary f32 scale leaves a
small residual (fp16 has 11 significand bits, bf16 has 8), so the unconstrained
case is bounded separately rather than asserted exact.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
CHUNK_Q4, CHUNK_Q8 = 5120, 8704

torch = pytest.importorskip("torch")


@contextlib.contextmanager
def _converter_as_q4nx():
    """Temporarily make `q4nx` mean the converter package, then put it back.

    The collision is unavoidable in one direction: model_converter.py:27 does an
    ABSOLUTE `from q4nx.gguf_tensor import GGUFTensor`, so the converter cannot be
    loaded under any other name and cannot be loaded file-by-file (its other
    imports are relative). That is exactly why test_quant_q4k.py pins
    gguf_tensor.py instead -- it has no absolute package imports.

    So the converter takes the name for the duration and sys.modules is restored
    in a finally. This is safe because pytest is single-threaded and any test that
    wanted the kernel reader already bound it at ITS import time, before the swap.

    Yields the converter's model_converter module.
    """
    import importlib
    import sys

    pkg_dir = REPO / "utilities" / "q4nx-build" / "q4nx"
    saved = sys.modules.get("q4nx")
    saved_path = list(sys.path)
    try:
        for m in [k for k in sys.modules if k == "q4nx" or k.startswith("q4nx.")]:
            del sys.modules[m]
        sys.path.insert(0, str(pkg_dir.parent))
        import q4nx as converter_pkg          # noqa: F401  (the swap itself)
        yield importlib.import_module("q4nx.model_converter")
    finally:
        for m in [k for k in list(sys.modules) if k == "q4nx" or k.startswith("q4nx.")]:
            del sys.modules[m]
        if saved is not None:
            sys.modules["q4nx"] = saved
        sys.path[:] = saved_path


def _reader():
    """The kernel-side reader, loaded by path under a private name.

    Cached, so it never consults sys.modules["q4nx"] -- which _converter_as_q4nx
    is free to repoint.
    """
    import sys

    name = "_open_kernels_q4nx_reader_for_writer_test"
    if name in sys.modules:
        return sys.modules[name]
    p = REPO / "open_kernels" / "model" / "q4nx.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pack(fn_name, *args, **kw):
    """Call a packer on `self` with the four block-geometry attributes set."""
    with _converter_as_q4nx() as mc:
        ModelArch = importlib.import_module("q4nx.constants").ModelArch

        class _W(mc.__Q4NX_Converter, model_arch=ModelArch.QWEN3):
            def convert(self, *a, **k):
                raise NotImplementedError

            def __init__(self):
                self.row_block_size = 32
                self.col_block_size = 256
                self.parallel_size = 16
                self.keep_block_in_2D = True

        w = _W()
        out = getattr(w, fn_name)(*args, **kw)
        # Keep the writer module alive past the swap; it only holds code objects.
        import torch as _t
        return out.clone() if isinstance(out, _t.Tensor) else out


def _bf16_scales(shape, rng):
    """Uniform scales that survive fp16 -> bf16 unchanged.

    bf16 is a strict subset of fp16 for these magnitudes, so converting bf16 ->
    fp16 here is lossless and the writer's cast back to bf16 is exact.
    """
    bf = torch.from_numpy(rng.uniform(0.01, 0.05, shape).astype(np.float32)).to(torch.bfloat16)
    f16 = bf.to(torch.float16)
    assert torch.equal(f16.to(torch.bfloat16).float(), bf.float())
    return bf, f16


# ------------------------------------------------------------------ q4_1

class TestQ41WriterAgainstReader:
    def test_a_q4_1_chunk_reads_back_exactly(self):
        """value(r, k) = scale(r, k//32) * q(r, k) + min(r, k//32)."""
        K = _reader()

        rng = np.random.default_rng(0)
        R, C, G = 32, 256, 8
        s, _ = _bf16_scales((R, G), rng)
        mn, _ = _bf16_scales((R, G), rng)
        q = rng.integers(0, 16, (R, C)).astype(np.float32)

        packed = _pack("_pack_q4nx", 
            torch.from_numpy(s.float().numpy()),
            torch.from_numpy(mn.float().numpy()),
            torch.from_numpy(q),
        )
        assert packed.numel() == CHUNK_Q4, "q4_1 must emit one 5120-byte chunk per tile"
        chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, CHUNK_Q4)

        got = K.dq_chunks_q4_1(chunks).reshape(R, G, 32)
        want = (s.float().numpy()[:, :, None] * q.reshape(R, G, 32)
                + mn.float().numpy()[:, :, None])
        assert np.array_equal(got, want)

    def test_two_row_blocks_keep_their_order(self):
        """Chunk N is row-block N -- the ordering the pool's plan assumes."""
        K = _reader()

        rng = np.random.default_rng(1)
        R, C, G = 64, 256, 8
        s, _ = _bf16_scales((R, G), rng)
        mn, _ = _bf16_scales((R, G), rng)
        q = rng.integers(0, 16, (R, C)).astype(np.float32)

        packed = _pack("_pack_q4nx", 
            torch.from_numpy(s.float().numpy()),
            torch.from_numpy(mn.float().numpy()),
            torch.from_numpy(q),
        )
        chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, CHUNK_Q4)
        assert chunks.shape[0] == 2

        got = K.dq_chunks_q4_1(chunks).reshape(2, 32, G, 32)
        want = (s.float().numpy()[:, :, None] * q.reshape(R, G, 32)
                + mn.float().numpy()[:, :, None]).reshape(2, 32, G, 32)
        assert np.array_equal(got, want), "row blocks must not be transposed or swapped"

    def test_an_f32_scale_only_costs_the_bf16_rounding(self):
        """The residual with ordinary f32 scales is the scale cast, not layout."""
        K = _reader()

        rng = np.random.default_rng(2)
        R, C, G = 32, 256, 8
        s = rng.uniform(0.01, 0.05, (R, G)).astype(np.float32)
        mn = rng.uniform(0.0, 0.2, (R, G)).astype(np.float32)
        q = rng.integers(0, 16, (R, C)).astype(np.float32)
        packed = _pack("_pack_q4nx", torch.from_numpy(s), torch.from_numpy(mn), torch.from_numpy(q))
        chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, CHUNK_Q4)
        got = K.dq_chunks_q4_1(chunks).reshape(R, G, 32)
        # The writer stores the scale as bf16, so compare against the rounded one.
        s_bf = torch.from_numpy(s).to(torch.bfloat16).float().numpy()
        mn_bf = torch.from_numpy(mn).to(torch.bfloat16).float().numpy()
        want = s_bf[:, :, None] * q.reshape(R, G, 32) + mn_bf[:, :, None]
        assert np.abs(got - want).max() < 1e-5


# ------------------------------------------------------------------ q8

class TestQ8WriterAgainstReader:
    def test_a_q8_chunk_reads_back_exactly(self):
        """value(r, k) = scale(r, k//32) * i8(r, k), with no min plane."""
        K = _reader()

        rng = np.random.default_rng(3)
        R, C, G = 32, 256, 8
        s_bf, s16 = _bf16_scales((R, G), rng)
        q = torch.from_numpy(rng.integers(-127, 127, (R, C)).astype(np.int8))

        packed = _pack("_pack_q8nx", q, s16, None)
        assert packed.numel() == CHUNK_Q8, "q8 must emit one 8704-byte chunk per tile"
        chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, CHUNK_Q8)

        got = K.dq_chunks_q8(chunks).reshape(R, G, 32)
        want = s_bf.float().numpy()[:, :, None] * q.numpy().astype(np.float32).reshape(R, G, 32)
        assert np.array_equal(got, want)

    def test_two_row_blocks_keep_their_order(self):
        K = _reader()

        rng = np.random.default_rng(4)
        R, C, G = 64, 256, 8
        s_bf, s16 = _bf16_scales((R, G), rng)
        q = torch.from_numpy(rng.integers(-127, 127, (R, C)).astype(np.int8))

        packed = _pack("_pack_q8nx", q, s16, None)
        chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, CHUNK_Q8)
        assert chunks.shape[0] == 2

        got = K.dq_chunks_q8(chunks).reshape(2, 32, G, 32)
        want = (s_bf.float().numpy()[:, :, None]
                * q.numpy().astype(np.float32).reshape(R, G, 32)).reshape(2, 32, G, 32)
        assert np.array_equal(got, want)

    def test_negative_codes_survive(self):
        """q8 is signed; a uint8 round trip would silently fold the negatives."""
        K = _reader()

        rng = np.random.default_rng(5)
        R, C, G = 32, 256, 8
        s_bf, s16 = _bf16_scales((R, G), rng)
        q = torch.from_numpy(rng.integers(-128, 127, (R, C)).astype(np.int8))
        assert (q < 0).any(), "fixture must actually contain negative codes"

        packed = _pack("_pack_q8nx", q, s16, None)
        chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, CHUNK_Q8)
        got = K.dq_chunks_q8(chunks).reshape(R, G, 32)
        want = s_bf.float().numpy()[:, :, None] * q.numpy().astype(np.float32).reshape(R, G, 32)
        assert np.array_equal(got, want)


# ------------------------------------------------------- the two must not collide

def test_q4_1_and_q8_chunks_are_different_sizes():
    """A writer emitting the wrong chunk size is the cheapest bug to catch."""
    assert CHUNK_Q4 != CHUNK_Q8
    rng = np.random.default_rng(6)
    R, C, G = 32, 256, 8
    _, s16 = _bf16_scales((R, G), rng)
    q4 = _pack("_pack_q4nx", torch.from_numpy(np.zeros((R, G), np.float32)),
                      torch.from_numpy(np.zeros((R, G), np.float32)),
                      torch.zeros(R, C))
    q8 = _pack("_pack_q8nx", torch.zeros(R, C, dtype=torch.int8), s16, None)
    assert q4.numel() == CHUNK_Q4
    assert q8.numel() == CHUNK_Q8
