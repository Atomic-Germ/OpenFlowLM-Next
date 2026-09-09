/// \file pools.hpp
/// \brief Pack a layer's weights out of the `.q4nx` container into the byte
///        layouts the open kernels stream, following the manifest's packing
///        plan (open_kernels/recipes/qwen36moe.py `pack_plan`).
///
/// A q4_1 source is not dequantized or requantized: the 5120-byte chunks are
/// copied verbatim, only their ORDER changes, because the AIE array streams a
/// matrix band by band rather than in the file's raster order. The laws are the
/// ones phlegm verified byte-for-byte against pools captured from FLM's own
/// engine; open_kernels/recipes/pack.py is the same interpreter in NumPy, and
/// specs/open-engine/tests/test_pack_plan.py holds it to the frozen originals.
///
/// Ops: std_perm / std_perm_gguf (a standard [out, in] matmul tensor into 64-row band
/// order; the _gguf forms read GGUF blocks and widen the fp16 scales EXACTLY to f32),
/// A q8 source (8704-byte chunks) is accepted transparently by every q4 op and
/// re-quantized to q4_1 chunk by chunk on the way into the pool
/// (`requant_q4_1_chunks`). The two formats hold the SAME 32-row x 256-column
/// tile, so no chunk index law changes and no plan, manifest or kernel knows the
/// difference -- which is what lets the Qwen3.6-35B fine-tunes (q8 attention,
/// linear-attention and shared experts; q4_1 routed experts) and Qwen3.5's q8
/// `ssm_out_proj` run on kernels that only have a q4_1 GEMV (OPEN-PACK-PLAN).
/// Any other chunk size is refused, naming the tensor.
///
/// A projection the kernel set streams AT q8 (the manifest carries `q8_perm` instead of
/// `std_perm` for it) is not re-quantized at all: each 8704-byte container chunk is split
/// into two 16-row half-tiles of 5120 bytes -- a byte permutation, no arithmetic -- and
/// those are placed by the q8 band law (four 16-row parts per k-tile instead of two 32-row
/// halves). A container whose tensor is not q8 where the manifest says q8 is refused by
/// name: that is the check that the container agrees with the kernel set (OPEN-QUANT-Q8).
///
/// Ops: std_perm (a standard [out, in] matmul tensor into 64-row band order),
/// expert_stripes (routed up/gate as interleaved transposed stripes),
/// expert_down (the routed down slices), put (small weights verbatim),
/// q8_perm (the same tensor kept at q8: 16-row half-tiles in the q8 band order),
/// conv_transpose (conv1d [taps, NCH] -> [groups][taps][width]),
/// lmhead_q8 (the q8 head's 128-row supertiles) and transpose (a small
/// [rows, cols] tensor -> [cols, rows]).
#pragma once

#include <cstddef>
#include <cstdint>

#include "open_qwen36/manifest.hpp"
#include "open_qwen36/weight_file.hpp"

namespace open_qwen36 {
namespace pools {

/// One op of a plan into `dst` (a buffer of `dst_bytes`).
void apply(const PackOp& op, const WeightFile& m, int layer, uint8_t* dst, size_t dst_bytes, size_t chunk_bytes);

/// `nch` q8 chunks (8704 B each) -> `nch` q4_1 chunks (5120 B each), block for block, in
/// the SAME chunk order (both formats are 32 rows x 256 K, so no permutation happens here).
/// This is the one arithmetic that changes values, and every q4 pack op runs it when the
/// source tensor is q8.
/// Per 32-value block: m = the minimum rounded toward -inf in bf16, d = (max - m)/15
/// rounded toward +inf, nibble = (int)((v - m)/d + 0.5) clipped to [0, 15]. The directed
/// rounding is what makes [m, m + 15d] cover [min, max] and every value land within d/2 of
/// its reading. `recipes/pack.py requant_q4_1` is the same arithmetic in NumPy; pools_test
/// and tests/test_qwen35.py check the two on the same vectors.
void requant_q4_1_chunks(const uint8_t* src, size_t nch, uint8_t* dst);
/// One container q8 chunk (8704 B: scales[256] bf16 then codes[8192] int8, 32 rows x 256 K)
/// -> its 16-row half-tile `half` (5120 B: scales[128] bf16 at [0, 256), codes[4096] int8 at
/// [256, 4352), zero pad). Rows 16*half .. 16*half+15. The container's row-block stride is
/// exactly 4096 codes, so the codes are a verbatim slice and only the scales are gathered;
/// `recipes/pack.py q8_half_tiles` is the same permutation in NumPy.
void q8_half_tile(const uint8_t* chunk, unsigned half, uint8_t* dst);
/// [rows, cols] of `elem`-byte values -> [cols, dst_rows], the columns past `rows` zeroed.
/// dst_rows == rows is the plain transpose; a wider one pads a 16-head DeltaNet's alpha /
/// beta out to dn_glue's 32-lane accumulator.
void transpose_bytes(const uint8_t* src, uint64_t rows, uint64_t cols, uint64_t elem, uint64_t dst_rows,
                     uint8_t* dst);
/// The layer's weight pool (m.pool_bytes, fully written).
void pack_pool(const Manifest& m, const LayerType& lt, const WeightFile& f, int layer, uint8_t* dst);
/// The layer's small-weight blob (lt.consts_bytes, fully written).
void pack_consts(const Manifest& m, const LayerType& lt, const WeightFile& f, int layer, uint8_t* dst);
/// The lm_head pool (m.lmhead_pool_bytes): the manifest's pack.lm_head ops.
void pack_lmhead(const Manifest& m, const WeightFile& f, uint8_t* dst);
/// A small weight (a layernorm) as bf16: the q4nx container stores it bf16, a
/// GGUF may store f32/f16 — converted exactly as the container does.
void pack_norm(const WeightFile& f, const std::string& name, size_t bytes, uint8_t* dst);
/// A position record table: row p = [valid | nf | cos | sin] for the window's row counts
/// (stream_patch::attn_window) and these RoPE frequencies, `rows` rows of m.ptab_row.
void build_ptab(const Manifest& m, const RowGlobal& g, size_t rows, uint8_t* dst);

}  // namespace pools
}  // namespace open_qwen36
