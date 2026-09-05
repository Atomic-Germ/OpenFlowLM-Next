/// \file pools.hpp
/// \brief Pack a layer's weights out of the `.q4nx` container into the byte
///        layouts the open kernels stream (open_kernels/designs/layer_x).
///
/// Nothing is dequantized or requantized: the 5120-byte q4_1 chunks are copied
/// verbatim, only their ORDER changes, because the AIE array streams a matrix
/// band by band rather than in the file's raster order. The laws are the ones
/// phlegm verified byte-for-byte against pools captured from FLM's own engine
/// and that open_kernels/model/pools.py implements in NumPy; the two builders
/// are byte-compared in this engine's tests.
///
/// Layer pool (512 MB), byte offsets:
///   0           routed experts' up/gate stripes (160 KB each, [up_k | gate_k] x4 per expert)
///   335544320   routed experts' down slices (640 KB each)
///   503316480   share_up   503971840  share_gate   504627200  share_down
///   505282560   qkv (linear layer) / q (attention layer)
///   510525440   k        511180800  v        511836160  gate
///   515768320   z-gate (linear layer)        517079040  o (attention layer)
#pragma once

#include <cstddef>
#include <cstdint>

#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36 {
namespace layout {

// open_kernels/designs/layer_x/layout.py — keep in step with it.
constexpr size_t kPoolBytes = 536870912;
constexpr size_t kLmheadPoolBytes = 542113792;
constexpr size_t kChunk = 5120;
constexpr size_t kStripe = 163840;

// consts, linear layer: [lnw | glue side minus xn | nw | postln | router W | sgw | out_proj]
constexpr size_t C_LNW = 0, C_SIDE = 4096, C_NW = 335872, C_POSTLN = 339968, C_RW = 344064,
                 C_SGW = 1392640, C_WOUT = 1396736, C_BYTES = C_WOUT + 10485760;
constexpr size_t GLUE_SIDE_BYTES = 331776;
// consts, attention layer: [lnw | postln | meta: qn | kn | router W | sgw]
constexpr size_t CA_LNW = 0, CA_POSTLN = 4096, CA_META = 8192, CA_RW = 10240, CA_SGW = 1058816,
                 CA_BYTES = CA_SGW + 4096;
// act (the DDR bounce between the stages), and where the router leaves its record
constexpr size_t A_ROUT = 176128, A_BYTES = 190464;
constexpr size_t AA_ROUT = 83968, AA_BYTES = 98304;
constexpr size_t ROUT_IDX_OFF = 1024;  // int32 idx[8] inside the router record
// state BO (linear layers): [conv state bf16 3x8192][S: 32 heads x 140 rows x 512 B]
constexpr size_t STATE_BYTES = 49152 + 32 * 140 * 512;
// KV cache row [K_t | V_t] (bf16 512 each) and the position record row
constexpr size_t KV_ROW = 2048, PTAB_ROW = 1024;
constexpr size_t HIDDEN = 2048, VOCAB = 248320;
// tokenizer.json's real vocab; lm_head rows above it are padding with undefined content
constexpr size_t REAL_VOCAB = 248070;
constexpr size_t LOGITS_BYTES = VOCAB * 4;  // f32

}  // namespace layout

namespace pools {

/// The 512 MB weight pool of layer `layer` into `dst` (kPoolBytes, fully written).
void build_layer_pool(const Q4nxFile& m, int layer, bool full_attn, uint8_t* dst);
/// The q8 lm_head pool into `dst` (kLmheadPoolBytes).
void build_lmhead_pool(const Q4nxFile& m, uint8_t* dst);
/// The per-layer small-weight blob the layer_x designs read (C_BYTES / CA_BYTES).
void build_consts(const Q4nxFile& m, int layer, bool full_attn, uint8_t* dst);
/// The position record table: row p = [pos | nf = max(p,1) | cos | sin], `rows` rows.
void build_ptab(size_t rows, uint8_t* dst);

}  // namespace pools
}  // namespace open_qwen36
