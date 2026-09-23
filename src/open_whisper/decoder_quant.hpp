//===- decoder_quant.hpp -------------------------------------*- C++ -*-===//
//
// open_whisper -- OPTIONAL, env-selected precision variants for the host
// decoder (task 0180 Part 8): a per-tensor lever to move fewer bytes without
// changing the decoder's default arithmetic. Every variant is OFF unless its
// env var is set, and the unset (default) path in decoder.cpp is untouched --
// this file adds NEW code paths, it does not edit the old ones.
//
// Deliberately its own translation unit pulling in no XRT and no device (the
// same discipline guards.cpp documents for itself): decoder_quant_test.cpp
// links this file plus open_qwen36/q4nx_file.cpp alone, so every kernel here
// is reachable, and testable against a double-precision reference, without
// hardware or the 1.6 GB container's decoder ever running a device.
//
// Three independent variants:
//
//   OW_DEC_XKV = fp32 (default) | bf16
//     The gathered CROSS K/V (decoder.hpp's xkv_gathered_, set once per 30 s
//     window by Decoder::set_encoder_output()) stored as bf16 instead of
//     fp32. Halves 61.4 MB. Self-attention's own KV cache is untouched --
//     this variant only ever touches the cross side.
//
//   OW_DEC_W = bf16 (default) | int8
//     The eight per-layer linears (self q/k/v/out, cross q/out, fc1, fc2)
//     stored as symmetric int8 with one fp32 scale per OUTPUT row, quantized
//     once at load time from the bf16 weights the container ships. NOT the
//     tied head -- that is OW_DEC_HEAD, below, because the head is read two
//     different ways (a lookup on the input side, a sweep on the output
//     side) and conflating the two would lose the lookup's exactness for
//     free.
//
//   OW_DEC_HEAD = bf16 (default) | int8 | int8x
//     The tied head (embed_tokens read as an output projection: logits =
//     h . embed_tokens^T). The EMBEDDING LOOKUP (token_id -> row) always
//     reads the bf16 table -- only the 51866 x 1280 output sweep is
//     affected.
//       int8:  every logit from the int8 approximation.
//       int8x: every logit from int8, then the K=64 rows with the largest
//              int8-approximate logit are recomputed EXACTLY (the same bf16
//              dot decoder.cpp's default path uses) and overwritten in
//              place. See recompute_top_k_exact()'s own comment for why this
//              keeps greedy argmax exact except in one named failure mode.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ow {

// ---------------------------------------------------------------------
// Env parsing -- read ONCE (by the Decoder constructor), strict: anything
// other than the recognised strings throws, matching encoder.cpp's
// fa_attn_requested() (task 0180 Part 6 fixed that parser from failing open
// on a misspelling to refusing; these three start refusing from day one).
// An unset / empty env var is the default and does not throw.
// ---------------------------------------------------------------------

enum class XkvPrecision { FP32, BF16 };
enum class WeightPrecision { BF16, INT8 };
enum class HeadPrecision { BF16, INT8, INT8X };

XkvPrecision parse_xkv_precision();       // OW_DEC_XKV
WeightPrecision parse_weight_precision(); // OW_DEC_W
HeadPrecision parse_head_precision();     // OW_DEC_HEAD

const char *to_string(XkvPrecision p);
const char *to_string(WeightPrecision p);
const char *to_string(HeadPrecision p);

// ---------------------------------------------------------------------
// Per-output-row symmetric int8 quantization (OW_DEC_W / OW_DEC_HEAD).
// ---------------------------------------------------------------------

// W_int8[o, i] ~= W_bf16[o, i] / scale[o], scale[o] = max(|row o|) / 127.
// A row that is exactly zero gets scale 0 and an all-zero int8 row -- the
// division by amax is skipped for that row (never 0/0), and
// linear_int8()'s result for it is exactly 0*0 + bias = bias, matching what
// an exact zero-weight row must produce; no NaN, no branch needed at
// evaluation time (0 * anything_finite == 0 in IEEE 754, and W.scale is
// finite by construction).
struct QLinear {
  std::vector<int8_t> w;    // [out, in], row-major
  std::vector<float> scale; // [out]
  std::vector<float> b;     // [out]
  int64_t out = 0, in = 0;
};

// Quantizes an [out, in] weight given as raw bf16 bits (the container's own
// layout -- decoder.hpp's Linear::w). `bias` must have `out` entries (an
// explicit zero vector for a tensor the container has no bias for, same
// convention as decoder.cpp's load_linear_nobias()).
QLinear quantize_int8_rows_bf16(const uint16_t *w_bf16, const std::vector<float> &bias,
                                int64_t out, int64_t in);

// The same quantization from an already-widened fp32 source -- exposed for
// the test, which builds adversarial rows directly in fp32 rather than
// round-tripping them through bf16 first.
QLinear quantize_int8_rows_f32(const float *w_f32, const std::vector<float> &bias, int64_t out,
                               int64_t in);

// y[o] = scale[o] * (x . w_row_int8[o]) + b[o], AVX2 (falls back to scalar
// without __AVX2__): each int8 row widens to i32 (_mm256_cvtepi8_epi32) then
// f32 (_mm256_cvtepi32_ps), FMA against the fp32 activation `x`. Output rows
// are processed FOUR AT A TIME so the four independent FMA chains share the
// same load of `x` and can run with independent accumulators (nothing about
// `x` differs between rows, only the weight row does) -- row-parallel with
// OpenMP across groups of 4.
void linear_int8(const float *x, const QLinear &W, float *y);

// Double-precision reference of EXACTLY linear_int8()'s arithmetic against
// the SAME already-quantized row `w_row` (length `n`) and its `scale`/`bias`
// -- i.e. "does the kernel correctly implement the dot product against this
// quantized data", not "how much did quantizing cost". Used by
// decoder_quant_test.cpp; not on any hot path.
double linear_int8_row_ref_f64(const float *x, const int8_t *w_row, float scale, float bias,
                               int64_t n);

// ---------------------------------------------------------------------
// bf16-widening dot / axpy for the gathered cross K/V (OW_DEC_XKV=bf16).
// Same shape of kernel as decoder.cpp's own (private, unexported) dot_bf16 /
// axpy8 -- duplicated rather than shared, the same choice decoder.cpp's own
// header already makes for dot8/axpy8 against host_ops.cpp's copies ("so the
// two are directly comparable" / testable without pulling in the other's
// translation unit).
// ---------------------------------------------------------------------

// x (fp32) . w (bf16, widened lane by lane).
float dot_bf16_kernel(const float *x, const uint16_t *w, int64_t n);
// y (fp32) += alpha * v (bf16, widened lane by lane).
void axpy_bf16_kernel(float *y, const uint16_t *v, float alpha, int64_t n);

// Double-precision reference of dot_bf16_kernel against the SAME bf16 bits
// (bf16 -> f64 is exact, so this isolates "does the AVX2 widening dot agree
// with the value those bits actually represent" from bf16's own rounding).
double dot_bf16_ref_f64(const float *x, const uint16_t *w, int64_t n);

// One query row of cross-attention against `len` bf16-gathered K/V rows, per
// head -- the OW_DEC_XKV=bf16 twin of decoder.cpp's private attend_one(),
// generalised the same way (two strides per side) but over uint16_t K/V
// instead of float. Arithmetic order matches attend_one() exactly (same
// max-then-exp-then-normalize, same t-then-head_dim loop nesting) so the
// ONLY difference from the fp32 path is which bits K and V are read from.
void attend_one_xkv_bf16(const float *q, const uint16_t *k_base, int64_t k_row_stride,
                         int64_t k_head_stride, const uint16_t *v_base, int64_t v_row_stride,
                         int64_t v_head_stride, int64_t len, int64_t heads, int64_t head_dim,
                         float scale, float *out, float *scores_scratch, int64_t scores_stride);

// ---------------------------------------------------------------------
// int8x: exact top-K logit recompute (OW_DEC_HEAD=int8x).
// ---------------------------------------------------------------------

// `logits[0, out)` already holds the int8-approximate value for every row.
// Finds the K largest BY THAT APPROXIMATION, recomputes exactly those K rows
// with dot_bf16_kernel against `w_bf16` (the container's own bf16 bits --
// the same tensor linear()'s default bf16 path reads), and overwrites
// logits[idx] with the exact value + bias[idx] (bias may be null for an
// explicit zero vector, matching the tied head's own no-bias convention).
//
// Why this keeps greedy argmax exact: the true (bf16-exact) maximum logit is
// somewhere in [0, out). If it is among the K rows the int8 pass ranked
// highest, this function computes it exactly and overwrites it in place, so
// the corrected array's max IS the true max -- the argmax is exact whether
// or not any of the other out-K rows' int8 approximations are close. The
// only failure mode is the true maximum landing OUTSIDE the int8-ranked
// top-K: that requires int8 quantization error to have pushed the true
// winner below rank K-1 other rows, i.e. an error larger than the gap
// between the true max and the K-th largest int8-approximate logit. K=64 is
// chosen because that gap is, empirically (see decoder_quant_test's
// real-weight measurement), far larger than the per-row int8 error this
// quantization produces -- but it is not a proof, which is why this is
// documented as the named risk rather than claimed away.
void recompute_top_k_exact(const float *x, int64_t out, int64_t in, int64_t k,
                           const uint16_t *w_bf16, const float *bias, float *logits);

}  // namespace ow
