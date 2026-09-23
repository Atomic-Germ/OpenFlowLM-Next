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

// ---------------------------------------------------------------------------
// Fast-path host ops (task 0180: speed up the encoder's host-side elementwise
// work). These are NOT bit-identical to the exact ops above in general --
// characterised against a double reference in host_ops_fast_test.cpp (max
// abs/rel error, in ulps where meaningful) and gated by the product-level WER
// harness (tools/wer/), not by golden-token bit-identity (trap 29 says that
// gate is a coin flip below ~1.3x cosine, which every one of these lands
// inside). The exact ops above are UNCHANGED and stay the default; every
// caller selects between the two behind host_fast_enabled().
//
// host_fast_enabled(): OW_HOST_FAST, strict -- unset or "0" is false, "1" is
// true, anything else throws (same discipline as encoder.cpp's OW_ATTN
// parser: a misspelling must not silently fall back to "off" while a caller
// believes it is running the fast path).
bool host_fast_enabled();

// omp_threads(): the team size every OpenMP region in this engine uses, via a
// num_threads() clause rather than omp_set_num_threads() -- the latter is
// process-wide and would also resize every other engine inside oflm.exe.
//
// Why not the default (all 24 logical CPUs): after each region MSVC's OpenMP
// workers spin, and with a thread on every logical CPU the thread blocked in
// xrt::run::wait() is woken late. Measured (task 0180 Part 17, nvidia, fast
// config, 3 runs): npu dispatch 1038 ms at 24 threads, 939 at 12, 819 with
// OMP_WAIT_POLICY=PASSIVE -- which in turn costs the decoder 59% on its ~40
// small regions per step. One thread per physical core keeps the spin (the
// decoder wants it) and leaves the SMT siblings free for the waiter.
//
// Default: std::thread::hardware_concurrency() / 2, at least 1.
// OW_OMP_THREADS=<n> overrides; anything that is not a positive integer throws.
// Every region here writes disjoint outputs (the one reduction sums timers), so
// the team size changes scheduling only, never a value.
int omp_threads();

// Vectorised float32 erf (AVX2 bulk, scalar double-erf tail for n % 8 != 0).
// See host_ops.cpp for the algorithm and its attribution. `erf_scalar_ref` is
// the double-precision reference (std::erf, rounded once) used by the test to
// score erf_avx2's ulp error -- NOT a fast op itself.
void erf_avx2(const float *x, int64_t n, float *out);
float erf_scalar_ref(float x);

// out_bf[r,:] = bf16(GELU(c[r,:] + bias[:])), fused: the fp32 fc1_h
// intermediate (31.5 MB at Whisper's FFN width) is never written or re-read,
// and the round to bf16 for fc2's A happens in the same pass instead of a
// separate bf16_fill call. `c` is a device C buffer: READ ONLY (trap 27; see
// gelu_bias() above -- the same contract applies here unchanged).
void gelu_bias_bf16_fast(const float *c, int64_t rows, int64_t cols, const float *bias,
                        uint16_t *out_bf);

// out_bf[r,:] = bf16(LayerNorm(x)[r,:]). The mean/variance reduction is the
// SAME double-precision accumulation, in the SAME order, as layer_norm()
// above -- only the final per-element combine-and-round differs (vectorised
// float32 instead of promote-to-double-then-cast, then a separate bf16_fill
// pass). No fp32 `h` buffer is written to memory; a small per-thread register
// footprint carries each row's output before it is rounded and stored.
void layer_norm_bf16_fast(const float *x, const float *w, const float *b, int64_t rows,
                          int64_t cols, uint16_t *out_bf);

// x[r,:] += c[r,:] + bias[:], fusing what the exact path runs as THREE passes
// (memcpy C -> scratch, scratch += bias, x += scratch) into one that reads
// `c` ONCE (read-only, trap 27) and reads/writes `x` once. The floating-point
// OPERATION ORDER is preserved exactly (c+bias computed first, then added to
// x, per element) -- so unlike the other fast ops this one is expected to be,
// and is tested for, BIT-IDENTICAL to the exact three-pass path; it is fast
// only because it removes the intermediate buffer's own memory traffic, not
// because it changes any arithmetic. `x` is never device-mapped (run_layer()
// only ever hands GEMM C buffers to `c`), so read-modify-write on it is safe.
void add_bias_residual_fast(const float *c, const float *bias, int64_t rows, int64_t cols,
                            float *x);

// Fused bias-add + head-major gather for attention()'s QKV repack: reads a
// qkv GEMM's device C buffer ONCE (read-only, trap 27), adds the qkv bias
// column-wise, and writes straight into the same head-major scratch layout
// attention() itself fills -- skipping the standalone host `qkv` copy
// (memcpy) and its separate add_bias() pass entirely. Like
// add_bias_residual_fast, the per-element OPERATION (c + bias[col]) is
// unchanged from what add_bias() would have computed, just deferred to when
// the value is read -- expected and tested BIT-IDENTICAL to
// add_bias(qkv)+attention()'s own gather.
void attention_gather_bias_fast(const float *qkv_c, int64_t m_padded, int64_t t, int64_t d,
                                int64_t heads, int64_t head_dim, const float *bias,
                                float *scratch);

// The scoring/softmax/P.V loop attention() already runs, factored out
// UNCHANGED so both attention() (exact gather, above) and the fused fast
// gather (attention_gather_bias_fast, above) can share it byte for byte.
// `scratch` must already hold the head-major [h][q|k|v][t][head_dim] layout;
// `out`'s rows [0,t) are written, rows [t, m_padded) are left to the caller
// (attention() zero-pads before calling; the fast path does the same).
void attention_core(int64_t t, int64_t d, int64_t heads, int64_t head_dim, float *out,
                    float *scratch, AttnPhases *phases = nullptr);

// Row-parallel bf16_fill: bit-identical to bf16_fill (the RNE rounding is
// element-independent, so splitting the array across threads changes no
// result), just spread across OpenMP threads instead of running single-
// threaded. bf16_fill() itself is left untouched so the exact path's
// instruction sequence -- and its timing -- is exactly what it always was.
void bf16_fill_parallel(uint16_t *dst, const float *src, size_t n);

}  // namespace ow
