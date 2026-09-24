//===- host_ops.hpp ------------------------------------------*- C++ -*-===//
//
// open_whisper -- everything that is NOT a GEMM: LayerNorm, GELU, bf16
// rounding, im2col and bidirectional attention. All fp32, all on the host,
// row-parallel with OpenMP; the hot inner loops (attention's dot products
// and its weighted sum over V) are AVX2 when available.
// SPDX-License-Identifier: MIT
//
// Mirrors open_kernels/model/replica_whisper.py function for function, so the
// two are diffable: gemm() there is exactly kernels.hpp's KernelSet::run(),
// and every other function here has the same name and the same job.
//
#pragma once

#include <cstddef>
#include <cstdint>

namespace ow {

// fp32 -> bf16 bits, round-to-nearest-even, and back. Bit-identical to
// tools/npue.py's rounding and to weights.cpp's bf16_rne -- the same rounding
// on both sides of the NPU boundary is what makes A and B agree with what the
// container's writer meant by "bf16".
uint16_t to_bf16(float x);
float from_bf16(uint16_t h);

// Vectorised forms, AVX2 when available (bit-identical to the scalar ones --
// ported with attribution from NpuEmbeddings' src/open_npue/npue_encoder.hpp
// `bf16_fill`/`bf16_read`, whose header there records why the integer-only
// path is exact). Falls back to the scalar loop otherwise.
void bf16_fill(uint16_t *dst, const float *src, size_t n);
void bf16_read(float *dst, const uint16_t *src, size_t n);

// out[r, :] = 0 for r in [real_rows, total_rows). `cols` floats per row.
// Called after every host op that writes a [total_rows, cols] buffer destined
// for the NPU or for the residual stream, so garbage in the padded rows a
// design's fixed M requires never has a chance to grow across 32 layers (a
// GEMM computes each output row from its own input row alone, so padding is a
// per-row invariant the kernel itself cannot enforce).
void zero_pad_rows(float *buf, int64_t real_rows, int64_t total_rows, int64_t cols);

// LayerNorm over the last axis, eps = 1e-5, biased variance (mean/N, not
// mean/(N-1)) -- exactly replica_whisper.py's layer_norm. Row-parallel.
void layer_norm(const float *x, const float *w, const float *b, int64_t rows,
                int64_t cols, float *out);

// GELU, exact erf: 0.5*x*(1+erf(x/sqrt(2))), erf in double, one rounding at
// the end (matches NpuEmbeddings' gelu_erf_exact / replica_whisper.py's
// erf-based gelu to within double-vs-A&S-approximation, both far inside the
// bf16 datapath's own noise floor). In place or out of place; row-parallel.
void gelu(const float *x, int64_t rows, int64_t cols, float *out);

// out[r,:] = GELU(x[r,:] + bias[:]), reading `x` STRICTLY read-only.
//
// This exists because `x` is a GEMM's C buffer, which is MAPPED FROM THE
// DEVICE. Adding the bias in place there dirties CPU cache lines on a mapping
// the NPU also writes into, and a later write-back of those lines lands on top
// of what a later dispatch DMA'd into the same buffer. Measured: two dispatches
// of identical input returning C values that differ in whole 64-byte-aligned
// runs, at an unpredictable layer, in a design that is otherwise
// bit-deterministic (512 dispatches cycling four streams and 32 weight slots
// agree byte for byte). Never write into a device-mapped buffer the device
// also writes.
void gelu_bias(const float *x, int64_t rows, int64_t cols, const float *bias, float *out);

// y[r,:] += bias[:], row-parallel. `rows` may include padded rows -- the
// caller zero-pads afterwards if it matters.
void add_bias(float *y, const float *bias, int64_t rows, int64_t cols);

// out[r,:] = a[r,:] + b[r,:] (residual add), row-parallel.
void add_rows(const float *a, const float *b, int64_t rows, int64_t cols, float *out);

// The conv stem's im2col, tap-major (K = tap*C + channel), taps
// (stride*t-1, stride*t, stride*t+1), zero outside [0, t_in). `x` is
// [t_in, c] time-major; `out` is [m_padded, 3*c], zeroed first so rows
// [t_out, m_padded) -- t_out = (t_in-1)/stride + 1 -- come out zero. Matches
// replica_whisper.py's im2col()/conv_b() K-index convention exactly.
void im2col(const float *x, int64_t t_in, int64_t c, int64_t stride,
           int64_t m_padded, float *out);

// Bidirectional multi-head attention over the first `t` rows of `qkv`
// ([m_padded, 3*d] row-major, columns [0,d)=Q [d,2d)=K [2d,3d)=V, each split
// into `heads` groups of `head_dim` = d/heads), scale 1/sqrt(head_dim),
// softmax with max-subtraction in fp32. Writes rows [0,t) of `out`
// ([m_padded, d]); rows [t, m_padded) of `out` are zeroed (they are never a
// query here, but the buffer feeds a fixed-M GEMM next).
//
// `phases`, when given, accumulates the three parts separately in seconds: the
// scores GEMM (Q.K^T), the row softmax, and the value GEMM (P.V). Off unless a
// pointer is passed -- it costs four clock reads per (head, block of 8 query
// rows) -- and it exists to price moving the two GEMMs onto the array, because
// whatever the softmax costs stays on the host either way and is the Amdahl term.
// The softmax phase ends at 1/sum; the multiply by it is fused into P.V's
// scalar (t multiplies per row against P.V's t*head_dim MACs), which is also
// where an array P.V would carry it -- as one scale of each output row.
struct AttnPhases {
  double scores = 0, softmax = 0, values = 0;
};

// `scratch` must hold 3 * t * d floats. The kernel gathers each head's Q, K and
// V into it contiguously before computing, because in `qkv` consecutive K rows
// are 3*d floats apart -- 15 KB at d = 1280 -- so every dot product of a 64-wide
// head row touched a fresh cache line and the whole 23 MB tensor was re-streamed
// once per query row. Gathered, one head's K is 384 KB and stays in L2 while
// a block of query rows is scored against it.
//
// The arithmetic is UNCHANGED: each output element accumulates over t2 in the
// same increasing order as before, so the result is bit-identical to the
// row-at-a-time version. Only the order in which memory is touched differs.
void attention(const float *qkv, int64_t m_padded, int64_t t, int64_t d,
              int64_t heads, int64_t head_dim, float *out, float *scratch,
              AttnPhases *phases = nullptr);

}  // namespace ow
