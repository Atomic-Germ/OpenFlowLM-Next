/// \file pools.hpp
/// \brief Pack a layer's weights out of the `.q4nx` container into the byte
///        layouts the open kernels stream, following the manifest's packing
///        plan (open_kernels/recipes/qwen36moe.py `pack_plan`).
///
/// Nothing is dequantized or requantized: the 5120-byte q4_1 chunks are copied
/// verbatim, only their ORDER changes, because the AIE array streams a matrix
/// band by band rather than in the file's raster order. The laws are the ones
/// phlegm verified byte-for-byte against pools captured from FLM's own engine;
/// open_kernels/recipes/pack.py is the same interpreter in NumPy, and
/// specs/open-engine/tests/test_pack_plan.py holds it to the frozen originals.
///
/// Ops: std_perm / std_perm_gguf (a standard [out, in] matmul tensor into 64-row band
/// order; the _gguf forms read GGUF blocks and widen the fp16 scales EXACTLY to f32),
/// expert_stripes (routed up/gate as interleaved transposed stripes),
/// expert_down (the routed down slices), put (small weights verbatim),
/// conv_transpose (conv1d [taps, NCH] -> [groups][taps][width]).
#pragma once

#include <cstddef>
#include <cstdint>

#include "open_qwen36/manifest.hpp"
#include "open_qwen36/weight_file.hpp"

namespace open_qwen36 {
namespace pools {

/// One op of a plan into `dst` (a buffer of `dst_bytes`).
void apply(const PackOp& op, const WeightFile& m, int layer, uint8_t* dst, size_t dst_bytes, size_t chunk_bytes);
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
