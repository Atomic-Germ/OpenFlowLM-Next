from __future__ import annotations
from gguf.constants import GGMLQuantizationType
from gguf import dequantize, quantize
from gguf.constants import GGML_QUANT_SIZES

from typing import Tuple

from dataclasses import dataclass
from mpmath.libmp import int_types
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def _round_up(n: int, multiple: int) -> int:
    """utils.round_up_to_multiple, inlined: this module is also loaded on its own, by
    file path, so it must not reach into the package (specs/open-engine/tests)."""
    return n if multiple == 0 else ((n + multiple - 1) // multiple) * multiple

def _to_bf16_up(x: np.ndarray) -> np.ndarray:
    """Round the magnitude of a non-negative float32 UP to a BF16 value."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = np.where(u & np.uint32(0xFFFF), np.uint32(0x10000), np.uint32(0))
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def _bf16_next_up(x: np.ndarray, n: int) -> np.ndarray:
    """The n-th BF16 value above a BF16-representable, non-negative x (n=0 is x)."""
    if n == 0:
        return np.asarray(x, dtype=np.float32)
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (u + np.uint32(n) * np.uint32(0x10000)).view(np.float32)


def _refit_one_side(t: np.ndarray, search: int = 3,
                    cap: float = 255.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one metadata side of a super-block: exact per-group t_j -> (BF16 P', uint8 p'_j).

    Direct port of the pseudocode in quant.md, "Re-fitting (uint6, FP16) into
    (uint8, BF16)".  The kernel only ever uses the product t_j = P * p_j, so the
    caller hands over exactly that -- Q4_K's (FP16 P, uint6 p_j) collapsed into a
    single exact float32 -- and this preserves it directly.

    The two extra bits of p'_j are spent *absorbing* P's rounding error rather
    than as a plain x4 (which would be a no-op, since P/4 has P's significand):
    P' is fixed first, rounded away from zero so p'_j can never overflow the cap,
    then each p'_j is re-derived against the rounded P'.  Since t_j / P' runs up
    to 255, the quotient can absorb up to 255 * 2^-8 ~ 1 integer step, which the
    uint8 grid -- 4x finer than the uint6 grid it came from -- can represent.

    `search` also tries that many BF16 values above the base P' and keeps
    whichever minimizes sum_j (P' p'_j - t_j)^2.  A slightly larger P' often
    aligns better with several t_j at once than the smallest admissible one, and
    it is what removes the re-fit's sensitivity to how spread the t_j are; 3 is
    the knee of the measured sweep.

    Parameters
    ----------
    t : np.ndarray
        Effective per-group scale (or min), shape (..., groups_per_super_block),
        exact in float32.
    search : int
        Number of extra BF16 candidates above the base P' to score.
    cap : float
        Largest representable p'_j (255 for uint8).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        P' (BF16-representable float32, shape t.shape[:-1]) and p'_j (float32
        integers in [0, cap], shape of t).  Effective value: P' p'_j ~ t_j.
    """
    t = np.ascontiguousarray(t, dtype=np.float32)
    tmax = np.abs(t).max(axis=-1)
    # p'_j is unsigned, so the sign rides on P'; take it from the largest entry
    # (Q4_K's d/dmin are non-negative, so every t_j in a group shares one sign).
    lead = np.take_along_axis(t, np.argmax(np.abs(t), axis=-1)[..., None], axis=-1)[..., 0]
    sigma = np.where(lead < 0, np.float32(-1.0), np.float32(1.0))
    t = sigma[..., None] * t
    live = tmax > 0                                              # dead super-block -> all zero
    base = _to_bf16_up(tmax / np.float32(cap))

    best_P = np.zeros_like(base)
    best_p = np.zeros_like(t)
    best_e = np.full(base.shape, np.inf, dtype=np.float64)
    for c in range(search + 1):
        Pc = np.where(live, _bf16_next_up(base, c), np.float32(0.0)).astype(np.float32)
        inv = np.where(live, 1.0 / np.where(live, Pc, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
        pc = np.clip(np.rint(t * inv[..., None]), 0.0, cap).astype(np.float32)
        err = np.sum((Pc[..., None] * pc - t).astype(np.float64) ** 2, axis=-1)
        take = err < best_e
        best_e = np.where(take, err, best_e)
        best_P = np.where(take, Pc, best_P)
        best_p = np.where(take[..., None], pc, best_p)

    return (sigma * best_P).astype(np.float32), best_p


class GGUFTensor:
    name: str
    shape: Tuple[int, ...]
    data: np.ndarray
    tensor_type: GGMLQuantizationType

    # Q4_K shares one uint6 scale/min per 32 weights, 8 of them per 256-weight super-block.
    Q4_K_GROUP_SIZE = 32

    def __init__(self, name: str, shape: Tuple[int, ...], data: np.ndarray, tensor_type: GGMLQuantizationType):
        self.name = name
        self.shape = shape
        self.data = data
        self.tensor_type = tensor_type

    @staticmethod
    def unpack_q4_0(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]
        data = tensor.view(np.uint8)
        shape = data.shape
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))
        
        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        d = torch.from_numpy(d)
        m = torch.zeros_like(d)
        qs = torch.from_numpy(qs)
        d = d.view(-1, columns // block_size)
        m = m.view(-1, columns // block_size)
        qs = qs.view(-1, columns)

        return d, m, qs

    @staticmethod
    def unpack_q4_1(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_1]
        data = tensor.view(np.uint8)
        shape = data.shape
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))
        
        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)

        d = torch.from_numpy(d).contiguous()
        m = torch.from_numpy(m).contiguous()
        qs = torch.from_numpy(qs).contiguous()
        assert columns % block_size == 0, "Columns must be divisible by block size"

        d = d.view(-1, int(columns // block_size))
        m = m.view(-1, int(columns // block_size))
        qs = qs.view(-1, int(columns))

        return d, m, qs
    
    @staticmethod
    def unpack_q4_k(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack GGML Q4_K into per-group effective scale / min plus the raw uint4 quants.

        A Q4_K block is 144 bytes covering 256 weights = 8 groups of 32:

            2 B   d      FP16 super-block scale S
            2 B   dmin   FP16 super-block min   M
            12 B  scales 8 uint6 s_j + 8 uint6 m_j, bit-packed (get_scale_min_k4)
            128 B qs     256 uint4 q, nibble-packed

        and dequantizes as w^j_i = (S s_j) q^j_i - (M m_j): the min is an unsigned
        magnitude that is *subtracted*, unlike Q4_1's signed +m.

        What comes back here is the *factored-out* form t_j = S s_j and u_j = M m_j,
        one pair per group of 32, held exactly in float32 (11-bit FP16 significand
        times a 6-bit integer needs 17 bits, so the product is exact).  Nothing is
        re-quantized and no super-block structure survives, which means the result
        has the same shape and the same 32-column granularity as unpack_q4_1's
        (d, m, qw) and can go through the model-specific row/column reorders
        untouched.  The (BF16 S', uint8 s'_j) re-fit of quant.md happens later, in
        _pack_q4k, over whichever 8 groups actually end up sharing a super-block
        after those reorders.

        Parameters
        ----------
        tensor : np.ndarray
            Raw Q4_K tensor bytes.
        columns : int
            Row length K; must be a multiple of 256.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            t (rows, K/32), u (rows, K/32), q (rows, K), all float32.  u is the
            unsigned magnitude that is *subtracted*: w = t_j q - u_j.
        """
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K]
        assert columns % block_size == 0, "Columns must be divisible by the Q4_K super-block size"
        data = tensor.view(np.uint8)
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [12])

        S = d.view(np.float16).astype(np.float32).reshape(n_blocks)
        M = dmin.view(np.float16).astype(np.float32).reshape(n_blocks)

        # get_scale_min_k4: groups 0..3 take a plain 6-bit field, groups 4..7 take
        # a low nibble from scales[j+4] plus the two high bits of scales[j-4].
        lo = scales[:, 0:4]      # j = 0..3
        hi = scales[:, 4:8]      # j = 4..7 for the min side, high bits for j = 4..7
        top = scales[:, 8:12]
        s6 = np.concatenate([lo & np.uint8(0x3F),
                             (top & np.uint8(0x0F)) | ((lo >> np.uint8(6)) << np.uint8(4))], axis=1)
        m6 = np.concatenate([hi & np.uint8(0x3F),
                             (top >> np.uint8(4)) | ((hi >> np.uint8(6)) << np.uint8(4))], axis=1)
        s6 = s6.astype(np.float32)
        m6 = m6.astype(np.float32)

        # qs holds four runs of 32 bytes; within a run the low nibbles are the
        # first group of 32 and the high nibbles the next one.
        qb = qs.reshape((n_blocks, 4, 32))
        q = np.stack([qb & np.uint8(0x0F), qb >> np.uint8(4)], axis=2)  # (nb, 4, 2, 32)
        q = q.reshape((n_blocks, block_size)).astype(np.float32)

        # Exact in float32: 11-bit significand x 6-bit integer fits in 24 bits.
        t = (S[:, None] * s6).astype(np.float32)
        u = (M[:, None] * m6).astype(np.float32)

        n_groups = int(columns // GGUFTensor.Q4_K_GROUP_SIZE)
        t = torch.from_numpy(t).contiguous().view(-1, n_groups)
        u = torch.from_numpy(u).contiguous().view(-1, n_groups)
        q = torch.from_numpy(q).contiguous().view(-1, int(columns))

        return t, u, q

    @staticmethod
    def unpack_q8_0(tensors:np.ndarray, columns:int):
        """Split GGML Q8_0 data into scales and quantized values

        Parameters
        ----------
        tensors : np.ndarray
            Q8_0 tensor data
            Format per block (34 bytes):
                - 2 bytes: scale (float16)
                - 32 bytes: 32 x 8-bit quantized values

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            scales, data

        Raises
        ------
        ValueError
            _description_
        """
        byte_per_blocks = 34
        assert tensors.dtype == np.uint8 or tensors.dtype == np.int8, "Input must be np.uint8 or np.int8"
        original_shape = tensors.shape
        assert original_shape[-1] % byte_per_blocks == 0, "The last dimension must be a multiple of 34"
        
        
        blocks = tensors.reshape(*original_shape[:-1], -1, byte_per_blocks)
        
        # Use view() to reinterpret the bytes as float16 bits, not convert the values
        scales = blocks[..., 0:2].view(np.float16) 
        
        # Use view() to reinterpret bytes as signed int8
        data = blocks[..., 2:].view(np.int8)
        
        Q8_block_size = 32
        # Reshape to (rows, cols_per_type) using columns, matching unpack_q4_1 output shape
        scales = np.ascontiguousarray(scales).reshape(-1, columns // Q8_block_size)
        data = np.ascontiguousarray(data).reshape(-1, columns)
        
        return torch.from_numpy(scales.copy()), torch.from_numpy(scales.copy()), torch.from_numpy(data.copy())
        
        
    
    
    @staticmethod
    def e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
        bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
        return bits.view(np.float32)

    @staticmethod
    def reverse_transform_nibble_layout( tensor: torch.Tensor) -> torch.Tensor:
        """Reverses the custom nibble layout transformation."""
        assert tensor.dtype == torch.uint8
        assert tensor.shape[-1] == 16

        # 1. Reverse the final nibble swap
        t_lo = tensor & 0x0F
        t_hi = tensor & 0xF0
        interleaved = (t_lo << 4) | (t_hi >> 4)

        # 2. De-interleave the nibbles from abababab... back to aaaa...bbbb...
        # The high nibbles of 'interleaved' contain the nibbles for the first half (blk_a)
        nibbles_a_parts = interleaved & 0xF0
        # The low nibbles of 'interleaved' contain the nibbles for the second half (blk_b)
        nibbles_b_parts = interleaved & 0x0F

        # Reconstruct blk_a by packing the high nibbles back together
        # Pair up nibbles: (1st high nibble) | (2nd high nibble >> 4)
        blk_a = nibbles_a_parts[..., 0::2] | (nibbles_a_parts[..., 1::2] >> 4)

        # Reconstruct blk_b by packing the low nibbles back together
        # Pair up nibbles: (1st low nibble << 4) | (2nd low nibble)
        blk_b = (nibbles_b_parts[..., 0::2] << 4) | nibbles_b_parts[..., 1::2]

        deinterleaved = torch.cat((blk_a, blk_b), dim=-1)

        # 3. Reverse the initial nibble swap
        t_lo = deinterleaved & 0x0F
        t_hi = deinterleaved & 0xF0
        original_tensor = (t_lo << 4) | (t_hi >> 4)

        return original_tensor       
     
    @staticmethod
    def split_ggml_mxfpx_to_scale_blocks(structured_data: np.ndarray):
        """Split GGML MXFP4 data into scales and data blocks

        Format per block (17 bytes):
            - 1 byte: scale (uint8)
            - 16 bytes: 32 x 4-bit float values (2 exponent bits + 1 mantissa bit each)
        """
        
        assert (structured_data.dtype == np.uint8 or structured_data.dtype == np.int8), "Input must be np.uint8 or np.int8"

        original_shape = structured_data.shape
        assert original_shape[-1] % 17 == 0, "The last dimension must be a multiple of 17"
        
        # Reshape the last dimension into blocks of 17 bytes
        blocks = structured_data.reshape(*original_shape[:-1], -1, 17)
        
        # Extract scales (first byte of each block)
        scales = blocks[..., 0].astype(np.uint8)
        
        # Extract data (remaining 16 bytes, keep as uint8 for 4-bit unpacking)
        data = GGUFTensor.reverse_transform_nibble_layout( torch.from_numpy( blocks[..., 1:].astype(np.uint8))).numpy()     
        return scales, data
    
    @staticmethod
    def unpack_mxfp4(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, data = GGUFTensor.split_ggml_mxfpx_to_scale_blocks(tensor)
        return torch.from_numpy(scale), torch.from_numpy(data)

    
    def dequantize(self) -> torch.Tensor:
        w = dequantize(self.data, self.tensor_type)
        w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
        return w

    def get_used_quantization_type(self, default_tensor_type: GGMLQuantizationType) -> GGMLQuantizationType:
        if self.tensor_type in [GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16, GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.MXFP4]:
            return self.tensor_type
        else:
            # For unsupported types (including Q8_0, which is not a native
            # Q4NX packing), we will dequantize and then quantize to
            # default_tensor_type. This makes Q8_0-source GGUFs honor the
            # config target (e.g. Q4_1 main matmuls) instead of keeping every
            # weight 8-bit.
            return default_tensor_type

    def unpack(self, default_tensor_type: GGMLQuantizationType) -> np.ndarray:
        # Native floating point tensor types (F32/F16/BF16) can only be
        # returned as-is when the caller does NOT require a quantized
        # target (e.g. norm weights, whose config entry sets
        # default_tensor_type to a float type). If the caller requests a
        # quantized target (Q4_0/Q4_1/Q8_0) -- which happens for ordinary
        # matmul weights when the source GGUF stores every tensor in full
        # precision (no per-tensor quantization at all) -- we must still
        # quantize+pack the tensor into that format below. Otherwise the
        # weight is left oversized and in the wrong (unpacked) layout,
        # which the NPU runtime cannot read and crashes on.
        is_native_float = self.tensor_type in (
            GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        )
        wants_quantized_target = default_tensor_type in (
            GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q8_0
        )
        # The Q4_0/Q4_1/Q8_0 block-quantized formats only make sense for 2D
        # matmul weight matrices; 1D tensors (e.g. rope_freqs, biases) are
        # never packed this way even when a config entry is missing and
        # falls back to the global default_tensor_type. Treat those as
        # native float passthrough regardless of the requested target.
        if len(self.shape) < 2:
            wants_quantized_target = False

        if is_native_float and not wants_quantized_target:
            if self.tensor_type == GGMLQuantizationType.F32:
                return [torch.Tensor(np.array(self.data.view(np.float32)))]
            elif self.tensor_type == GGMLQuantizationType.F16:
                return [torch.Tensor(np.array(self.data.view(np.float16).astype(np.float32)))]
            else:  # BF16
                return [torch.from_numpy(self.data.copy()).view(torch.bfloat16)]

        elif self.tensor_type == GGMLQuantizationType.Q4_0:
            return self.unpack_q4_0(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q4_1:
            return self.unpack_q4_1(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q8_0:
            if default_tensor_type == GGMLQuantizationType.Q8_0:
                return self.unpack_q8_0(self.data, self.shape[0])
            # Q8_0 is not the runtime's native packing when the config asks
            # for a different target (e.g. Q4_1 main matmuls): dequantize and
            # re-quantize into the requested format before packing.
            return self._requantize_to(default_tensor_type)
        elif self.tensor_type == GGMLQuantizationType.MXFP4:
            return self.unpack_mxfp4(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q4_K:
            # Read natively. Stacking a dequantize onto a re-quantize costs 0.240 bits of
            # ENOB and 0.375 bpw against this path, and damages the tail far more than the
            # mean; q5/q6 still take the fallback below.
            return self.unpack_q4_k(self.data, self.shape[0])
        else:
            """
                If the tensor type is not natively packable as-is (either a
                truly unsupported GGUF quant type such as Q4_K/Q5_K/Q6_K, or
                a native F32/F16/BF16 tensor that the caller wants quantized
                to a Q4_0/Q4_1/Q8_0 target), dequantize it (gguf.dequantize
                already handles F32/F16/BF16 as a passthrough/cast) and then
                quantize it to default_tensor_type before packing.
            """
            # Block-quantized targets (Q4_0/Q4_1/Q8_0) require the last
            # dimension to be a multiple of 32. Tensors that don't fit -- e.g.
            # small F32 auxiliary tensors with a 3-wide axis, common in LFM2
            # (shortconv / ssm-style weights) -- cannot be packed that way, so
            # keep them as a float passthrough rather than crashing in
            # gguf.quantize.
            if wants_quantized_target and self.shape and self.shape[-1] % 32 != 0:
                w = dequantize(self.data, self.tensor_type)
                w = torch.from_numpy(w).contiguous()
                if self.tensor_type == GGMLQuantizationType.BF16:
                    w = w.view(torch.bfloat16)
                return [w]
            return self._requantize_to(default_tensor_type)

    def _requantize_to(self, default_tensor_type: GGMLQuantizationType) -> np.ndarray:
        """Dequantize the source tensor and re-quantize it into the requested
        target format, returning a (d, m, qw) tuple ready for packing."""
        # ggml has no Q4_K encoder, so a source that has to go through a re-quantize
        # (q5/q6, or a float tensor) lands on Q4_1's grid. A real Q4_K source never
        # reaches here -- `unpack` reads it natively. The caller still packs Q4_K, and
        # the two disagree on the sign of the min: Q4_1 stores an added `m` (<= 0), Q4_K
        # a subtracted magnitude `u` (>= 0). Negating it below is exact and keeps the
        # tuple in the convention the packer the caller will use expects; without it the
        # weights come out silently mirrored about the block minimum.
        want_q4_k = default_tensor_type == GGMLQuantizationType.Q4_K
        if want_q4_k:
            default_tensor_type = GGMLQuantizationType.Q4_1
        try:
            w = dequantize(self.data, self.tensor_type)
            w = torch.from_numpy(w).contiguous().to(torch.bfloat16)

            if default_tensor_type == GGMLQuantizationType.BF16:
                # BF16 is not a quantized format, so no requantization is
                # needed: just return the dequantized tensor directly,
                # consistent with the native BF16 branch above.
                return [w]

            w = w.to(torch.float32).numpy()
            data_quantized = quantize(w, default_tensor_type).copy()
            if default_tensor_type == GGMLQuantizationType.Q4_1:
                d, m, qw = self.unpack_q4_1(data_quantized, self.shape[0])
            elif default_tensor_type == GGMLQuantizationType.Q4_0:
                d, m, qw = self.unpack_q4_0(data_quantized, self.shape[0])
            elif default_tensor_type == GGMLQuantizationType.Q8_0:
                d, m, qw = self.unpack_q8_0(data_quantized, self.shape[0])
            else:
                raise ValueError(f"Unsupported tensor type: {default_tensor_type.name}")
            return (d, -m, qw) if want_q4_k else (d, m, qw)
        except Exception as e:
            print(
                f"[WARN] Could not quantize {self.tensor_type.name} tensor "
                f"(shape {list(self.shape)}) to {default_tensor_type.name}: {e}; "
                "keeping it as a float passthrough instead of crashing."
            )
            try:
                w = dequantize(self.data, self.tensor_type)
                w = torch.from_numpy(np.array(w)).contiguous()
                if self.tensor_type == GGMLQuantizationType.BF16:
                    w = w.view(torch.bfloat16)
                return [w]
            except Exception:
                return None, None, None


def pack_q4k(t: torch.Tensor, u: torch.Tensor, q: torch.Tensor, row_block_size: int,
             col_block_size: int, keep_block_in_2D: bool, search: int = 3) -> torch.Tensor:
    """Pack the FLM q4_k format (uint8 s'/m' per group + bf16 S'/M' per super-block).

    Input is what GGUFTensor.unpack_q4_k returns, *after* any model-specific
    reorder: the effective per-group scale t_j and subtracted min u_j in exact
    float32, shape (rows, cols // 32), plus the uint4 quants (rows, cols).  The
    (BF16 P', uint8 p'_j) re-fit of quant.md is done here, per 256-column
    super-block, so it always fits the 8 groups that really share a super-block
    in the packed output -- a reorder that shuffles columns at a granularity
    finer than 256 (ssm_out_proj moves 128-column chunks) would otherwise leave
    the super-block metadata unrepresentable.

    Upstream (ROCm FLM_Q4NX_Converter, `_Q4NX_Converter._pack_q4k`) reads the block
    geometry off the converter; here it is three arguments, so the packer can be loaded
    and checked on its own -- specs/open-engine/tests/test_quant_q4k.py holds it against
    the engine's reader (OPEN-QUANT-Q4K).

    Mirrors `q4k_block_t` in the decoding kernels' model_spec.h, one struct
    per row_block_size x col_block_size chunk, everything column major over
    the chunk (the whole row span is resident at once, so there is no
    parallel-strided re-order):

        uint8 scales[col_block_size // Q4_group_size][row_block_size]
        uint8 mins  [col_block_size // Q4_group_size][row_block_size]
        uint4 qs    [col_block_size][row_block_size // NUM_int4_in_byte]
        bf16  S     [col_block_size // Q4K_super_block_size][row_block_size]
        bf16  M     [col_block_size // Q4K_super_block_size][row_block_size]

    With the shipping 32x256 chunk that is 256 + 256 + 4096 + 64 + 64 = 4736 B,
    i.e. 4.625 bits per weight, matching get_quantization_byte_size().

    GGUF's Q4_K subtracts the min (w = S' s'_j q - M' m'_j) while the kernel
    adds both accumulators, so M' is stored negated here.
    """
    Q4_group_size = 32
    Q4K_super_block_size = 256
    NUM_int4_in_byte = 2

    t = t.to(torch.float32).contiguous()
    u = u.to(torch.float32).contiguous()
    q = q.contiguous()

    cols = q.shape[-1]
    assert cols % Q4K_super_block_size == 0, "Q4_K needs a multiple of 256 columns"
    assert col_block_size % Q4K_super_block_size == 0

    if cols % col_block_size != 0:
        cols_padded = _round_up(cols, col_block_size)
        t = F.pad(t, (0, (cols_padded - cols) // Q4_group_size), "constant", 0)
        u = F.pad(u, (0, (cols_padded - cols) // Q4_group_size), "constant", 0)
        q = F.pad(q, (0, cols_padded - cols), "constant", 0)

    # Re-fit each side onto (BF16 super-block value, uint8 per-group value).
    groups_per_super = Q4K_super_block_size // Q4_group_size
    S, s8 = _refit_one_side(t.numpy().reshape(-1, groups_per_super), search=search)
    M, m8 = _refit_one_side(u.numpy().reshape(-1, groups_per_super), search=search)
    S = torch.from_numpy(S).view(t.shape[0], -1)
    M = torch.from_numpy(M).view(u.shape[0], -1)
    s8 = torch.from_numpy(s8).view_as(t)
    m8 = torch.from_numpy(m8).view_as(u)

    # scales / mins: one group of 32 columns contributes row_block_size uint8,
    # groups ascending -- the same block layout the bf16 scales use in q4nx.
    s8 = rearrange(s8, '(p r) (u c) -> p u (c r)', r=row_block_size,
                   c=col_block_size // Q4_group_size).contiguous()
    m8 = rearrange(m8, '(p r) (u c) -> p u (c r)', r=row_block_size,
                   c=col_block_size // Q4_group_size).contiguous()
    # S / M: one entry per row per super-block; a 256-column chunk holds exactly one.
    S = rearrange(S, '(p r) (u c) -> p u (c r)', r=row_block_size,
                  c=col_block_size // Q4K_super_block_size).contiguous()
    M = rearrange(M, '(p r) (u c) -> p u (c r)', r=row_block_size,
                  c=col_block_size // Q4K_super_block_size).contiguous()

    # quants: one column = row_block_size nibbles, rows paired into a byte with
    # the even row in the low nibble, columns ascending.
    q = rearrange(q, '(p r) (u c) -> p u r c', r=row_block_size, c=col_block_size)
    q = rearrange(q, 'p u (r b) c -> p u c r b', b=NUM_int4_in_byte).contiguous().to(torch.int8)
    q[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(q[..., 1], 4), 0xF0)
    q[..., 0] = torch.bitwise_or(torch.bitwise_and(q[..., 0], 0x0F), q[..., 1])
    q = rearrange(q[..., 0].contiguous(), 'p u c r -> p u (c r)').contiguous()

    s8 = s8.to(torch.uint8).view(torch.int8).numpy()
    m8 = m8.to(torch.uint8).view(torch.int8).numpy()
    S = S.to(torch.bfloat16).view(torch.int8).numpy()
    M = (-M).to(torch.bfloat16).view(torch.int8).numpy()   # kernel adds the min path
    q = q.view(torch.int8).numpy()

    merged = torch.from_numpy(np.concatenate([s8, m8, q, S, M], axis=-1).copy())
    if not keep_block_in_2D:
        merged = merged.reshape(-1, merged.shape[-1])
    return merged
