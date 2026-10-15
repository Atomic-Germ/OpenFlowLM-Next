r"""Is the f32 scale plane doing anything the bf16 one does not?

The GGUF-direct path gives a model its own pool layout (``spec.quant ==
"q4_1_f32"``): the same 32x256 tiles as a ``.q4nx`` container, but the ``d`` and
``m`` planes hold the GGUF fp16 block scales widened EXACTLY to f32. The q4
chunk is 6144 B against 5120 B, the q8 lm_head chunk 9216 B against 8704 B, and
every design is exported twice so a kernel exists that can read the wider one
(``dx_f32``, ``lm_head_q4_f32``, the ``gemv_q4s32_*`` TUs).

The reason given is fidelity: the file loses nothing, so a GGUF model is
bit-identical to its converted twin.

But the kernel does not read f32 scales. It narrows them to bf16 on the way in
-- ``load_scale`` in ``designs/gemv_q4/gemv_q4.h``, whose own comment says "the
same result a q4nx chunk's pre-narrowed bf16 scales give". And fp16 -> f32 is
exact. So the kernel's RNE(f32 -> bf16) and a host RNE(fp16 -> bf16) round the
same real value into the same format, which makes them the same bits, which
makes the f32 plane a deferral rather than a saving.

This file checks that, three ways:

  1. **The narrowing is one rounding, not two.** Exhaustively over all 65536
     fp16 bit patterns: the kernel's integer RNE applied to the widened f32
     equals a host ``astype(bfloat16)`` of the fp16.
  2. **The two pools carry the same information.** For the same GGUF tensor,
     ``q4_1_pack.pack_q4_1_pool`` (5120 B, bf16) and ``gguf_pool.pack_q4_pool``
     (6144 B, f32) hold byte-identical codes, and the bf16 planes are exactly
     the kernel's narrowing of the f32 planes -- so both pools produce the same
     numbers under the kernel's own arithmetic.
  3. **The q8 lm_head chunk too** (9216 B f32 vs 8704 B bf16), which is where
     ``open-gguf-direct/SKILL.md`` records the f32 build's open numerics bug
     (maxrel ~5e-3 against the q4nx build's 4.5e-6).

It does not test the hardware. ``ggml-xdna`` (github.com/Cyronius/ggml-xdna)
already did: ``hybrid/q4_pack.{h,cpp}`` is ``q4_1_pack.py`` in C++, held
byte-identical to it on real GGUF tensors by ``hybrid/test-q4-pack.cpp``, and
Qwen3-1.7B packed that way ran through the stock kernels correctly at 1.44
TOPS.

``python gguf_scale_width.py``
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import gguf_pool
import q4_1_pack

Q8_CH_BF16 = 8704       # q4nx q8 chunk: 256 bf16 scales, then 8192 int8 codes
Q8_CH_F32 = gguf_pool.CH_Q8


def kernel_rne(f32: np.ndarray) -> np.ndarray:
    """The kernel's f32 -> bf16 convert, as bf16 bit patterns.

    ``load_scale`` uses the AIE native RNE convert; the integer form below is
    the same rounding and is what ``f32_to_bf16`` in ``q4nx_file.hpp`` writes:
    add half an ulp plus the low bit of the kept mantissa, then truncate.
    """
    u = np.ascontiguousarray(f32, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def narrow_plane(f32_bytes: np.ndarray) -> np.ndarray:
    """An f32 scale plane -> the bf16 plane the kernel effectively reads."""
    return kernel_rne(f32_bytes.view(np.float32)).view(np.uint8)


def widen_plane(bf16_bytes: np.ndarray) -> np.ndarray:
    """A bf16 scale plane -> f32 bytes (exact), so an f32-plane dequant sees it."""
    return bf16_bytes.view(bfloat16).astype(np.float32).view(np.uint8)


def q8_chunk_bf16(f32_chunk: np.ndarray) -> np.ndarray:
    """One 9216 B f32-scale q8 chunk -> the 8704 B bf16-scale chunk.

    The only difference between the two layouts: 256 scales at 4 bytes become
    256 at 2, and the 8192 int8 codes move down by 512 bytes unchanged.
    """
    out = np.empty(Q8_CH_BF16, np.uint8)
    out[0:512] = narrow_plane(f32_chunk[0:gguf_pool.SCALE_BYTES])
    out[512:] = f32_chunk[gguf_pool.SCALE_BYTES:]
    return out


def _check_rounding() -> bool:
    """Every fp16 the kernel can meet, both ways round."""
    h = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    finite = np.isfinite(h)
    wide = h[finite].astype(np.float32)                 # exact: fp16 -> f32
    via_kernel = kernel_rne(wide)                       # narrowed in the kernel
    via_host = wide.astype(bfloat16).view(np.uint16)    # narrowed on the host
    same = int(np.count_nonzero(via_kernel == via_host))
    total = int(finite.sum())
    ok = same == total
    print(f"{'PASS' if ok else 'FAIL'} fp16 -> bf16: kernel RNE == host RNE for "
          f"{same}/{total} finite fp16 values "
          f"({int((~finite).sum())} NaN/Inf patterns skipped; scales are finite)")
    return ok


def _check_q4(n: int, k: int, rs: int, gtype: str, rng: np.random.Generator) -> bool:
    """The 5120 B bf16 pool and the 6144 B f32 pool, on the same tensor."""
    raw = gguf_pool.random_q4_blocks(n, k, rng, gguf_type=gtype)
    f32_pool = gguf_pool.pack_q4_pool(raw, rs, gtype).reshape(-1, gguf_pool.CH_Q4)
    bf16_pool = q4_1_pack.pack_q4_1_pool(raw, rs).reshape(-1, q4_1_pack.CH)

    codes = np.array_equal(bf16_pool[:, 1024:], f32_pool[:, 2048:])
    planes = all(
        np.array_equal(bf16_pool[:, lo:hi], narrow_plane(f32_pool[:, flo:fhi].reshape(-1)).reshape(len(f32_pool), -1))
        for lo, hi, flo, fhi in ((0, 512, 0, 1024), (512, 1024, 1024, 2048))
    )

    # ...and therefore the same numbers under the kernel's own arithmetic: take
    # the f32 pool, narrow its planes the way the kernel does, dequantize.
    seen = f32_pool.copy()
    for lo, hi in ((0, 1024), (1024, 2048)):
        seen[:, lo:hi] = widen_plane(narrow_plane(seen[:, lo:hi].reshape(-1))).reshape(len(seen), -1)
    as_kernel_sees = gguf_pool.dequant_pool(seen.reshape(-1), n, k, rs, gtype)
    from_bf16_pool = q4_1_pack.dequant_pool(bf16_pool.reshape(-1), n, k, rs)
    values = np.array_equal(as_kernel_sees, from_bf16_pool)

    ok = codes and planes and values
    saved = (len(f32_pool) * gguf_pool.CH_Q4 - bf16_pool.size) / (len(f32_pool) * gguf_pool.CH_Q4)
    print(f"{'PASS' if ok else 'FAIL'} q4 {gtype} n={n} k={k} rs={rs}: "
          f"codes identical={codes} planes={planes} values={values} "
          f"({bf16_pool.size} B vs {len(f32_pool) * gguf_pool.CH_Q4} B, {saved:.0%} smaller)")
    return ok


def _check_q8(n: int, k: int, rng: np.random.Generator) -> bool:
    """The q8 lm_head chunk: 9216 B f32 vs 8704 B bf16, same values."""
    raw = gguf_pool.random_q8_0_blocks(n, k, rng)
    f32_pool = gguf_pool.pack_q8_pool_lmhead(raw).reshape(-1, Q8_CH_F32)
    bf16_pool = np.stack([q8_chunk_bf16(c) for c in f32_pool])

    codes = np.array_equal(bf16_pool[:, 512:], f32_pool[:, gguf_pool.SCALE_BYTES:])
    seen = f32_pool.copy()
    seen[:, 0:gguf_pool.SCALE_BYTES] = widen_plane(
        narrow_plane(seen[:, 0:gguf_pool.SCALE_BYTES].reshape(-1))).reshape(len(seen), -1)
    as_kernel_sees = gguf_pool.dequant_pool_lmhead(seen.reshape(-1), n, k)
    direct = gguf_pool.dequant_pool_lmhead(f32_pool.reshape(-1), n, k, scale_dtype=bfloat16)
    values = np.array_equal(as_kernel_sees, direct)

    ok = codes and values
    saved = (f32_pool.size - bf16_pool.size) / f32_pool.size
    print(f"{'PASS' if ok else 'FAIL'} q8 lm_head n={n} k={k}: "
          f"codes identical={codes} values={values} "
          f"({bf16_pool.size} B vs {f32_pool.size} B, {saved:.0%} smaller)")
    return ok


def _selftest() -> int:
    rng = np.random.default_rng(0)
    ok = _check_rounding()
    for n, k, rs in [(512, 2048, 2), (2048, 512, 2), (512, 2048, 4)]:
        for gtype in ("Q4_1", "Q4_0"):
            ok &= _check_q4(n, k, rs, gtype, rng)
    for n, k in [(256, 2048), (512, 2048)]:
        ok &= _check_q8(n, k, rng)
    print("\nThe f32 plane holds bits the kernel throws away before its first MAC."
          if ok else "\nSomething here does not hold — the f32 plane may be earning its bytes.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
