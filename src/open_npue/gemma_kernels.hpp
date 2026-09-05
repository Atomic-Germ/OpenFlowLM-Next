//===- gemma_kernels.hpp -------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- EmbeddingGemma-300M host eltwise kernels: RMSNorm, RoPE,
// GeGLU.
// SPDX-License-Identifier: MIT
//
// STANDALONE ON PURPOSE, same discipline as tokenizer_gemma.hpp: this header
// has no XRT dependency and is not wired into main.cpp's Encoder::run() or
// hub.cpp's model catalogue. That integration (arch=1, the MQA-aware
// packer, the runtime layer loop) is separate, larger, and out of scope --
// see tasks/0063-m12-embeddinggemma-kernels/TASK.md. These functions exist
// so the *numerics* can be built and verified against
// reference/encoder_gemma.py (tasks/0055) independently of that wiring
// decision, exactly how tokenizer_gemma.cpp/.hpp were verified in isolation
// in tasks/0061 before any integration call was made.
//
// Every formula here is read from reference/encoder_gemma.py, not derived
// from memory of "how Gemma usually works" -- see that file's own header for
// the primary-source citations (Gemma3RMSNorm.forward,
// Gemma3RotaryEmbedding.forward, Gemma3MLP.forward in the installed
// transformers 5.15.0). Two load-bearing details, easy to get wrong by
// analogy with other model families:
//
//  * RMSNorm is `x/rms(x) * (1 + weight)`, NOT `x/rms(x) * weight`
//    (Llama-style). The weight is stored zero-centred. Missing the `1 +`
//    compiles, runs, and produces a plausible-looking but completely wrong
//    answer -- CLAUDE.md flags this exact gotcha.
//  * RoPE's base frequency (theta) is PER LAYER, not a single model-wide
//    constant: layers where (i+1) % sliding_window_pattern == 0 (pattern=6,
//    so 0-indexed layers 5, 11, 17, 23) are "full_attention" and use
//    rope_theta (1e6); every other layer is "sliding_attention" and uses
//    rope_local_base_freq (1e4). Getting this backwards desyncs Q/K's
//    rotary phase on 20 of 24 layers while still producing unit-norm
//    rotations, i.e. it looks numerically plausible while being wrong.

#pragma once

#include <cstddef>
#include <cstdint>

namespace npue {

// Gemma3RMSNorm: out[r,k] = x[r,k] * rsqrt(mean_k(x[r,:]^2) + eps) *
// (1 + weight[k]), reduction over the last `dim` elements of each row.
//
// The SAME function serves both RMSNorm uses in the model: the four
// per-layer/per-token norms over the full hidden size (dim = 768, rows =
// batch*seq), and the q_norm/k_norm applied PER ATTENTION HEAD over
// head_dim (dim = 256, rows = batch*heads*seq) -- the formula and the
// reduction axis are identical, only which axis is contiguous in memory
// differs, and that is the caller's layout choice, not this function's.
//
// Reduction and the final multiply are both done in double precision,
// matching reference/encoder_gemma.py's rms_norm() (which upcasts to
// float64 explicitly, "cheap insurance... over 256-768 elements") --
// this is not a speed/accuracy trade, it is matching the reference's own
// design choice so a byte-for-byte agreement is possible at all.
void rms_norm_cpu(const float *x, const float *weight, float *out,
                   int64_t rows, int64_t dim, float eps);

// True iff layer `layer_idx` (0-indexed) is a "full_attention" layer, per
// config.json's own construction (confirmed in
// reference/encoder_gemma.py's GemmaEmbeddingReference.is_full_attention_layer):
// (layer_idx + 1) % sliding_window_pattern == 0. For this checkpoint
// (sliding_window_pattern = 6, 24 layers) that is exactly layers
// {5, 11, 17, 23}.
bool gemma_is_full_attention_layer(int64_t layer_idx,
                                    int64_t sliding_window_pattern = 6);

// RoPE cos/sin tables, matching Gemma3RotaryEmbedding.forward /
// reference/encoder_gemma.py's rope_cos_sin():
//   inv_freq[j] = base^(-2j/head_dim),  j in [0, head_dim/2)
//   freqs[s,j]  = s * inv_freq[j]
//   emb[s,:]    = concat(freqs[s,:], freqs[s,:])   -- NOT interleaved; this
//                 is the "rotate_half" convention apply_rope_cpu below
//                 assumes.
// Computed in double precision, rounded to float32 only at the end
// (matching the reference exactly: pos/inv_freq/freqs/emb are all float64,
// only cos()/sin()'s result is cast down). `cos_out`/`sin_out` must each
// hold seq_len*head_dim floats.
void gemma_rope_tables(int64_t seq_len, int64_t head_dim, double base,
                        float *cos_out, float *sin_out);

// Apply RoPE to `rows` contiguous head_dim-wide vectors -- one row per
// (batch, head, sequence-position) triple, Q or K. `seq_len` gives the
// period for recovering each row's sequence position as `row_index %
// seq_len`, which holds for the natural [B,H,S,D] row-major layout (S is
// the fastest-varying axis before D) that reference/encoder_gemma.py's taps
// use -- a caller with a different row order must pre-arrange rows to match
// or pass its own per-row cos/sin pointers instead of using this
// convenience wrapper.
//
// cos/sin are the [seq_len, head_dim] tables from gemma_rope_tables()
// (selected for the LAYER's base frequency -- see
// gemma_is_full_attention_layer() above -- before calling this). Computed
// in float32 throughout, matching the reference exactly: apply_rope() in
// encoder_gemma.py does NOT upcast to float64 (q*cos_b + rotate_half(q)*
// sin_b is plain float32 numpy arithmetic).
//
// `x` and `out` may be the same pointer (in place); an internal per-row
// scratch copy makes this safe.
void apply_rope_cpu(const float *x, const float *cos, const float *sin,
                     float *out, int64_t rows, int64_t seq_len,
                     int64_t head_dim);

// GeGLU FFN elementwise stage: out = gelu_pytorch_tanh(gate) * up.
//
// NOT the exact-erf GELU main.cpp's gelu_cpu() implements for BERT/MiniLM
// -- a different activation formula, and this is a deliberately separate
// function (gelu_cpu is untouched):
//   gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
//   out          = gelu_tanh(gate) * up
//
// Matches reference/encoder_gemma.py's gelu_tanh()+mlp() exactly, INCLUDING
// its two-stage rounding: gelu_tanh() itself upcasts to float64, evaluates
// the formula, and rounds DOWN to float32 before returning (that rounded
// value is what the reference calls `act`); only then is `act` promoted
// back to float64 to multiply against `up` (also promoted), with the
// product rounded to float32 once more. Collapsing this into one
// float64 computation without the intermediate float32 round would not
// match the reference bit-for-bit.
void geglu_cpu(const float *gate, const float *up, float *out, size_t n);

} // namespace npue
