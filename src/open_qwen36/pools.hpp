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
/// A q8 source (8704-byte chunks) is accepted transparently by every q4 op and
/// re-quantized to q4_1 chunk by chunk on the way into the pool
/// (`requant_q4_1_chunks`). The two formats hold the SAME 32-row x 256-column
/// tile, so no chunk index law changes and no plan, manifest or kernel knows the
/// difference -- which is what lets the Qwen3.6-35B fine-tunes (q8 attention,
/// linear-attention and shared experts; q4_1 routed experts) and Qwen3.5's q8
/// `ssm_out_proj` run on kernels that only have a q4_1 GEMV (OPEN-PACK-PLAN).
///
/// A Q4_K source (4736-byte chunks, what FLM 1.0.3+ writes) is accepted the same way and
/// transcoded to q4_1 (`q4k_to_q4_1_chunks`). That one is nearly free: Q4_K's scale and
/// min already have the pool's granularity and index, so only the two bf16 products
/// round (OPEN-QUANT-Q4K). Any other chunk size is refused, naming the tensor.
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
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36 {
namespace pools {

/// One op of a plan into `dst` (a buffer of `dst_bytes`).
void apply(const PackOp& op, const Q4nxFile& m, int layer, uint8_t* dst, size_t dst_bytes, size_t chunk_bytes);

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
/// `nch` Q4_K chunks (4736 B each) -> `nch` q4_1 chunks (5120 B each), in the SAME chunk
/// order. Both formats hold a 32-row x 256-column tile with one (scale, min) pair per
/// (row, 32-column group) at the SAME meta index `g*32 + r`, so nothing is re-quantized:
///   scales[8][32] uint8 @ [0, 256)     mins[8][32] uint8 @ [256, 512)
///   qs[256][16]         @ [512, 4608)  byte k*16 + r/2, even row in the low nibble
///   S[32] bf16          @ [4608, 4672) M[32] bf16 @ [4672, 4736), M already negated
///   value = S[r]*scales[g][r]*nib + M[r]*mins[g][r], which is the pool's nib*d + m
/// The transcode is the two products rounded to bf16 -- the one place values move, by at
/// most a half-ulp, 2^-8 relative -- plus a byte de-interleave of the nibbles, since the
/// pool splits rows 0-15 and 16-31 into two 2048-byte planes. `recipes/pack.py
/// q4k_to_q4_1` is the same in NumPy; pools_test and tests/test_quant_q4k.py hash the
/// same vectors (OPEN-QUANT-Q4K).
void q4k_to_q4_1_chunks(const uint8_t* src, size_t nch, uint8_t* dst);
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
void pack_pool(const Manifest& m, const LayerType& lt, const Q4nxFile& f, int layer, uint8_t* dst);
/// The layer's small-weight blob (lt.consts_bytes, fully written).
void pack_consts(const Manifest& m, const LayerType& lt, const Q4nxFile& f, int layer, uint8_t* dst);
/// The lm_head pool (m.lmhead_pool_bytes): the manifest's pack.lm_head ops.
void pack_lmhead(const Manifest& m, const Q4nxFile& f, uint8_t* dst);
/// A position record table: row p = [valid | nf | cos | sin] for the window's row counts
/// (stream_patch::attn_window) and these RoPE frequencies, `rows` rows of m.ptab_row.
void build_ptab(const Manifest& m, const RowGlobal& g, size_t rows, uint8_t* dst);
/// One position record at KV row `row`: [i32 valid | i32 nf | cos f32[rot/2] @512 | sin ...],
/// the rotary angle of pair i taken at pos[axis(i)]. A text token passes the same position
/// three times. `section` is Qwen3-VL's mrope_section (three counts summing to rot/2) with
/// `interleaved` (pair i takes axis i % 3 within its section); empty: every pair takes pos[0],
/// which is what build_ptab writes for row p with pos = (p, p, p). `row >= g.switch_row`
/// (Phi-3's longrope only; RowGlobal::kSwitchNever for every other family) reads
/// `g.long_inv_freq` in place of `g.inv_freq` for the whole record.
void build_ptab_record(const Manifest& m, const RowGlobal& g, size_t row, const double pos[3],
                       const std::vector<int>& section, bool interleaved, uint8_t* r);


}  // namespace pools
}  // namespace open_qwen36
