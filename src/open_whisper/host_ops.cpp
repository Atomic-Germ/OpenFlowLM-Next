//===- host_ops.cpp ------------------------------------------*- C++ -*-===//
// open_whisper -- see host_ops.hpp. SPDX-License-Identifier: MIT
#include "host_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ow {

// OW_HOST_FAST: default changed 2026-09-23 (task 0180 Part 15) -- the fused
// ops are bit-identical or WER-indistinguishable from the exact path
// (host_ops_fast_test.cpp; H3/H4 vs H0 on 1200 utterances) and took
// ENCODE 1990 -> 1621 ms on the nvidia clip, so unset now behaves as "1". "0" restores every exact op
// above. Strict either way, like encoder.cpp's OW_ATTN parser -- a typo
// must throw, not silently measure a different path while the operator
// believes they picked one.
bool host_fast_enabled() {
  const char *e = std::getenv("OW_HOST_FAST");
  if (!e || !*e || std::string(e) == "1") return true;
  if (std::string(e) == "0") return false;
  throw std::runtime_error("OW_HOST_FAST is '" + std::string(e) + "': expected '1'/unset or '0'");
}

int omp_threads() {
  static const int n = [] {
    const char *e = std::getenv("OW_OMP_THREADS");
    if (!e || !*e) {
      const unsigned hc = std::thread::hardware_concurrency();
      return hc >= 2 ? static_cast<int>(hc / 2) : 1;
    }
    const std::string v(e);
    size_t used = 0;
    long k = 0;
    try { k = std::stol(v, &used); } catch (...) { used = 0; }
    if (used != v.size() || k < 1 || k > 1024)
      throw std::runtime_error("OW_OMP_THREADS is '" + v + "': expected a positive integer");
    return static_cast<int>(k);
  }();
  return n;
}

uint16_t to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

float from_bf16(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// Ported with attribution from NpuEmbeddings' src/open_npue/npue_encoder.hpp
// (`bf16_fill`/`bf16_read`): bit-identical to the scalar to_bf16/from_bf16
// above -- every integer op used has the same semantics on uint32 as on the
// __m256i lanes, and after the >> 16 shift the values are in [0, 65535] so
// packus never actually saturates.
#if defined(__AVX2__)
void bf16_fill(uint16_t *dst, const float *src, size_t n) {
  const __m256i k7fff = _mm256_set1_epi32(0x7FFF);
  const __m256i kone = _mm256_set1_epi32(1);
  auto rne = [&](__m256i u) {
    __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), kone);
    return _mm256_srli_epi32(_mm256_add_epi32(u, _mm256_add_epi32(k7fff, odd)), 16);
  };
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i a = rne(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i)));
    __m256i b = rne(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i + 8)));
    __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), p);
  }
  for (; i < n; ++i) dst[i] = to_bf16(src[i]);
}

void bf16_read(float *dst, const uint16_t *src, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(u));
  }
  for (; i < n; ++i) dst[i] = from_bf16(src[i]);
}
#else
void bf16_fill(uint16_t *dst, const float *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = to_bf16(src[i]);
}
void bf16_read(float *dst, const uint16_t *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = from_bf16(src[i]);
}
#endif

// Same RNE rounding as bf16_fill, just chunked across OpenMP threads by row
// of ~4096 elements (arbitrary; large enough that thread launch overhead is
// noise against the AVX2 loop, small enough to give the scheduler enough
// chunks to balance). Bit-identical to bf16_fill for the same input -- each
// output element depends only on its own input element.
void bf16_fill_parallel(uint16_t *dst, const float *src, size_t n) {
  constexpr size_t kChunk = 4096;
  const size_t n_chunks = (n + kChunk - 1) / kChunk;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t c = 0; c < static_cast<int64_t>(n_chunks); ++c) {
    const size_t off = static_cast<size_t>(c) * kChunk;
    const size_t len = std::min(kChunk, n - off);
    bf16_fill(dst + off, src + off, len);
  }
}

void zero_pad_rows(float *buf, int64_t real_rows, int64_t total_rows, int64_t cols) {
  if (real_rows >= total_rows) return;
  std::memset(buf + real_rows * cols, 0,
             static_cast<size_t>(total_rows - real_rows) * static_cast<size_t>(cols) * sizeof(float));
}

// THREADED, and the reason an earlier version of this file was not is worth
// keeping. Two runs of the identical binary gave different per-layer cosines,
// and serialising every host loop made it go away, so it read as a race in
// MSVC's OpenMP runtime. It was not one. The cause was `add_bias` writing into
// the GEMM's C buffer, which is MAPPED FROM THE DEVICE: the dirty CPU cache
// lines that creates are written back on top of whatever a later dispatch
// DMA'd into the same buffer, in whole 64-byte runs, at an unpredictable
// moment. Serialising the host only changed the timing of the write-back.
// The encoder now treats every C buffer as read-only (see gelu_bias() in
// host_ops.hpp) and two runs agree to every digit with all of this threaded.
void layer_norm(const float *x, const float *w, const float *b, int64_t rows,
                int64_t cols, float *out) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t r = 0; r < rows; ++r) {
    const float *xr = x + r * cols;
    double mu = 0.0;
    for (int64_t c = 0; c < cols; ++c) mu += xr[c];
    mu /= static_cast<double>(cols);
    double var = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const double d = xr[c] - mu;
      var += d * d;
    }
    var /= static_cast<double>(cols);
    const double inv_std = 1.0 / std::sqrt(var + 1e-5);
    float *orow = out + r * cols;
    for (int64_t c = 0; c < cols; ++c)
      orow[c] = static_cast<float>((xr[c] - mu) * inv_std) * w[c] + b[c];
  }
}

namespace {
// Abramowitz & Stegun-free erf: use the C++ standard library's, in double,
// then round once -- matches NpuEmbeddings' gelu_erf_exact exactly. Both this
// and replica_whisper.py's own float64 A&S approximation are inside 1.5e-7 of
// each other, three decades below the bf16 datapath's own ~2e-3 noise floor.
inline float gelu_scalar(float x) {
  const double xd = static_cast<double>(x);
  return static_cast<float>(0.5 * xd * (1.0 + std::erf(xd * 0.70710678118654752440)));
}
}  // namespace

void gelu(const float *x, int64_t rows, int64_t cols, float *out) {
  const int64_t n = rows * cols;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t i = 0; i < n; ++i) out[i] = gelu_scalar(x[i]);
}

void gelu_bias(const float *x, int64_t rows, int64_t cols, const float *bias, float *out) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t r = 0; r < rows; ++r) {
    const float *xr = x + r * cols;
    float *orow = out + r * cols;
    for (int64_t c = 0; c < cols; ++c) orow[c] = gelu_scalar(xr[c] + bias[c]);
  }
}

float erf_scalar_ref(float x) {
  return static_cast<float>(std::erf(static_cast<double>(x)));
}

namespace {
#if defined(__AVX2__)
// exp(x) = 2^(x*log2(e)) = 2^n * 2^f, n by round-to-nearest, 2^f by a
// degree-5 minimax polynomial, 2^n by writing the exponent field directly.
// Measured (tasks/0179 Part 17, NpuEmbeddings repo) at 1.192e-07 max relative
// error against expf over the softmax's input range -- one fp32 ulp. Reused
// here VERBATIM WITH ATTRIBUTION (that task's rejected/vector-exp.patch) as
// the exp building block for erf8 below. It was REJECTED there for softmax,
// where one ulp chains through 32 decoder layers and flips a token (trap 29)
// -- a property of softmax's row-normalisation, not of exp8 itself. GELU
// consumes an exp only inside erf's own approximation, once per element, with
// no such chain; host_ops_fast_test.cpp characterises the result directly
// rather than assuming the softmax verdict carries over.
//
// Known deviation from IEEE NaN propagation, found while characterising
// erf8's own NaN input (host_ops_fast_test.cpp): x86's VMAXPS returns its
// SECOND operand when the FIRST is NaN (Intel SDM), and the "clamp to the
// underflow floor" step here is exactly `_mm256_max_ps(x, lo)` with x
// possibly NaN as the first operand -- so a NaN INPUT TO exp8 ITSELF comes
// out as exp8(lo), a small finite number, not NaN. erf8 below still ends up
// NaN for NaN input, because the separate `t` factor multiplying exp8's
// result is independently NaN -- verified empirically, not just reasoned
// about (the whole point of the test is not to trust the reasoning alone).
inline __m256 exp8(__m256 x) {
  const __m256 lo = _mm256_set1_ps(-87.3365f);
  const __m256 too_small = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);
  x = _mm256_max_ps(x, lo);
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  p = _mm256_fmadd_ps(_mm256_mul_ps(p, _mm256_mul_ps(r, r)), _mm256_set1_ps(1.0f),
                     _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
  const __m256i ni = _mm256_cvtps_epi32(n);
  const __m256 pow2n = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
  return _mm256_andnot_ps(too_small, _mm256_mul_ps(p, pow2n));
}

// erfc(|x|) via Numerical Recipes' rational-Chebyshev fit (Press, Teukolsky,
// Vetterling, Flannery, "Numerical Recipes in C", 2nd ed., section 6.2, the
// `erfcc` routine): one polynomial in t = 1/(1+0.5|x|), claimed fractional
// error < 1.2e-7 everywhere -- about one fp32 ulp (2^-23 = 1.19e-7) on its
// own. Evaluated here entirely in float32 (not the double precision the
// formula was fitted in), using exp8 above for the transcendental step; the
// combined error is characterised empirically in host_ops_fast_test.cpp, not
// assumed to still be 1.2e-7.
inline __m256 erfc_avx2_ax(__m256 ax) {   // ax = |x|, caller applies the sign
  const __m256 one = _mm256_set1_ps(1.0f), half = _mm256_set1_ps(0.5f);
  const __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(half, ax, one));
  __m256 poly = _mm256_set1_ps(0.17087277f);
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(-0.82215223f));
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(1.48851587f));
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(-1.13520398f));
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(0.27886807f));
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(-0.18628806f));
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(0.09678418f));
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(0.37409196f));
  poly = _mm256_fmadd_ps(t, poly, _mm256_set1_ps(1.00002368f));
  __m256 arg = _mm256_fnmadd_ps(ax, ax, _mm256_set1_ps(-1.26551223f));
  arg = _mm256_fmadd_ps(t, poly, arg);
  return _mm256_mul_ps(t, exp8(arg));
}

// erf(x) = copysign(1 - erfc(|x|), x): for x >= 0 this is 1 - erfc(x)
// directly (erfcc's ans IS erfc(|x|) there); for x < 0, erfc(x) = 2-ans, so
// erf(x) = 1-(2-ans) = ans-1 = -(1-ans) -- the same magnitude, sign flipped.
// Avoids a branch/blend per lane.
inline __m256 erf8(__m256 x) {
  const __m256 ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
  const __m256 mag = _mm256_sub_ps(_mm256_set1_ps(1.0f), erfc_avx2_ax(ax));
  const __m256 signbit = _mm256_set1_ps(-0.0f);
  return _mm256_or_ps(_mm256_andnot_ps(signbit, mag), _mm256_and_ps(x, signbit));
}
#endif
}  // namespace

void erf_avx2(const float *x, int64_t n, float *out) {
#if defined(__AVX2__)
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, erf8(_mm256_loadu_ps(x + i)));
  for (; i < n; ++i) out[i] = erf_scalar_ref(x[i]);
#else
  for (int64_t i = 0; i < n; ++i) out[i] = erf_scalar_ref(x[i]);
#endif
}

void gelu_bias_bf16_fast(const float *c, int64_t rows, int64_t cols, const float *bias,
                        uint16_t *out_bf) {
#if defined(__AVX2__)
  const __m256 invsqrt2 = _mm256_set1_ps(0.70710678118654752440f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256i k7fff = _mm256_set1_epi32(0x7FFF);
  const __m256i kone_i = _mm256_set1_epi32(1);
  auto rne = [&](__m256i u) {
    __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), kone_i);
    return _mm256_srli_epi32(_mm256_add_epi32(u, _mm256_add_epi32(k7fff, odd)), 16);
  };
  auto gelu8 = [&](__m256 xv) {
    return _mm256_mul_ps(half, _mm256_mul_ps(xv,
             _mm256_add_ps(one, erf8(_mm256_mul_ps(xv, invsqrt2)))));
  };
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t r = 0; r < rows; ++r) {
    const float *cr = c + r * cols;
    uint16_t *obr = out_bf + r * cols;
    int64_t i = 0;
    for (; i + 16 <= cols; i += 16) {
      __m256 x0 = _mm256_add_ps(_mm256_loadu_ps(cr + i), _mm256_loadu_ps(bias + i));
      __m256 x1 = _mm256_add_ps(_mm256_loadu_ps(cr + i + 8), _mm256_loadu_ps(bias + i + 8));
      __m256i u0 = rne(_mm256_castps_si256(gelu8(x0)));
      __m256i u1 = rne(_mm256_castps_si256(gelu8(x1)));
      __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(u0, u1), 0xD8);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(obr + i), p);
    }
    for (; i < cols; ++i) obr[i] = to_bf16(gelu_scalar(cr[i] + bias[i]));
  }
#else
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t r = 0; r < rows; ++r) {
    const float *cr = c + r * cols;
    uint16_t *obr = out_bf + r * cols;
    for (int64_t i = 0; i < cols; ++i) obr[i] = to_bf16(gelu_scalar(cr[i] + bias[i]));
  }
#endif
}

void layer_norm_bf16_fast(const float *x, const float *w, const float *b, int64_t rows,
                          int64_t cols, uint16_t *out_bf) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t r = 0; r < rows; ++r) {
    const float *xr = x + r * cols;
    // Mean/variance: SAME double accumulation, SAME order, as layer_norm()
    // above -- only the final combine+round differs.
    double mu = 0.0;
    for (int64_t c = 0; c < cols; ++c) mu += xr[c];
    mu /= static_cast<double>(cols);
    double var = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const double d = xr[c] - mu;
      var += d * d;
    }
    var /= static_cast<double>(cols);
    const double inv_std = 1.0 / std::sqrt(var + 1e-5);
    uint16_t *obr = out_bf + r * cols;
#if defined(__AVX2__)
    const __m256 muv = _mm256_set1_ps(static_cast<float>(mu));
    const __m256 invv = _mm256_set1_ps(static_cast<float>(inv_std));
    const __m256i k7fff = _mm256_set1_epi32(0x7FFF);
    const __m256i kone_i = _mm256_set1_epi32(1);
    auto rne = [&](__m256i u) {
      __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), kone_i);
      return _mm256_srli_epi32(_mm256_add_epi32(u, _mm256_add_epi32(k7fff, odd)), 16);
    };
    int64_t c = 0;
    for (; c + 16 <= cols; c += 16) {
      __m256 xv0 = _mm256_loadu_ps(xr + c), xv1 = _mm256_loadu_ps(xr + c + 8);
      __m256 n0 = _mm256_mul_ps(_mm256_sub_ps(xv0, muv), invv);
      __m256 n1 = _mm256_mul_ps(_mm256_sub_ps(xv1, muv), invv);
      __m256 o0 = _mm256_fmadd_ps(n0, _mm256_loadu_ps(w + c), _mm256_loadu_ps(b + c));
      __m256 o1 = _mm256_fmadd_ps(n1, _mm256_loadu_ps(w + c + 8), _mm256_loadu_ps(b + c + 8));
      __m256i u0 = rne(_mm256_castps_si256(o0));
      __m256i u1 = rne(_mm256_castps_si256(o1));
      __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(u0, u1), 0xD8);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(obr + c), p);
    }
    for (; c < cols; ++c)
      obr[c] = to_bf16(static_cast<float>((xr[c] - mu) * inv_std) * w[c] + b[c]);
#else
    for (int64_t c = 0; c < cols; ++c)
      obr[c] = to_bf16(static_cast<float>((xr[c] - mu) * inv_std) * w[c] + b[c]);
#endif
  }
}

void add_bias_residual_fast(const float *c, const float *bias, int64_t rows, int64_t cols,
                            float *x) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t r = 0; r < rows; ++r) {
    const float *cr = c + r * cols;
    float *xr = x + r * cols;
    int64_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= cols; i += 8) {
      __m256 t = _mm256_add_ps(_mm256_loadu_ps(cr + i), _mm256_loadu_ps(bias + i));
      _mm256_storeu_ps(xr + i, _mm256_add_ps(_mm256_loadu_ps(xr + i), t));
    }
#endif
    for (; i < cols; ++i) xr[i] += cr[i] + bias[i];
  }
}

void add_bias(float *y, const float *bias, int64_t rows, int64_t cols) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t r = 0; r < rows; ++r) {
    float *yr = y + r * cols;
    for (int64_t c = 0; c < cols; ++c) yr[c] += bias[c];
  }
}

void add_rows(const float *a, const float *b, int64_t rows, int64_t cols, float *out) {
  const int64_t n = rows * cols;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void im2col(const float *x, int64_t t_in, int64_t c, int64_t stride, int64_t m_padded,
           float *out) {
  const int64_t k = 3 * c;
  std::memset(out, 0, static_cast<size_t>(m_padded) * static_cast<size_t>(k) * sizeof(float));
  const int64_t t_out = (t_in - 1) / stride + 1;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t t = 0; t < t_out; ++t) {
    float *row = out + t * k;
    const int64_t centre = t * stride;   // tap 0 sits at x[centre]
    for (int64_t tap = 0; tap < 3; ++tap) {
      const int64_t src = centre + tap - 1;
      if (src < 0 || src >= t_in) continue;   // zero padding, already memset
      std::memcpy(row + tap * c, x + src * c, static_cast<size_t>(c) * sizeof(float));
    }
  }
}

namespace {
#if defined(__AVX2__)
inline float dot8(const float *a, const float *b, int64_t n) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  float s = _mm_cvtss_f32(h);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
inline void axpy8(float *y, const float *x, float alpha, int64_t n) {
  const __m256 av = _mm256_set1_ps(alpha);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] += alpha * x[i];
}
#else
inline float dot8(const float *a, const float *b, int64_t n) {
  float s = 0.f;
  for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
inline void axpy8(float *y, const float *x, float alpha, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}
#endif
}  // namespace

// Fused bias-add + head-major gather: the per-element operation (row value +
// bias[col]) is exactly what add_bias() would have computed on the full qkv
// buffer before attention()'s own gather reads it -- just performed at gather
// time instead of precomputed into a standalone buffer, so it is expected to
// be, and is tested for, bit-identical to add_bias(qkv) + the exact gather
// below. `qkv_c` is a device C buffer: READ ONLY (trap 27).
void attention_gather_bias_fast(const float *qkv_c, int64_t m_padded, int64_t t, int64_t d,
                                int64_t heads, int64_t head_dim, const float *bias,
                                float *scratch) {
  (void)m_padded;
  const int64_t stride = 3 * d;
  const int64_t hd = head_dim;
  const int64_t slice = t * hd;
  auto Q = [&](int64_t h) { return scratch + (h * 3 + 0) * slice; };
  auto K = [&](int64_t h) { return scratch + (h * 3 + 1) * slice; };
  auto V = [&](int64_t h) { return scratch + (h * 3 + 2) * slice; };

  const int64_t gather_work = heads * t;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t idx = 0; idx < gather_work; ++idx) {
    const int64_t h = idx / t, t1 = idx % t;
    const float *row = qkv_c + t1 * stride + h * hd;
    const float *bq = bias + h * hd, *bk = bias + d + h * hd, *bv = bias + 2 * d + h * hd;
    float *qd = Q(h) + t1 * hd, *kd = K(h) + t1 * hd, *vd = V(h) + t1 * hd;
    int64_t c = 0;
#if defined(__AVX2__)
    for (; c + 8 <= hd; c += 8) {
      _mm256_storeu_ps(qd + c, _mm256_add_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(bq + c)));
      _mm256_storeu_ps(kd + c, _mm256_add_ps(_mm256_loadu_ps(row + d + c), _mm256_loadu_ps(bk + c)));
      _mm256_storeu_ps(vd + c, _mm256_add_ps(_mm256_loadu_ps(row + 2 * d + c), _mm256_loadu_ps(bv + c)));
    }
#endif
    for (; c < hd; ++c) {
      qd[c] = row[c] + bq[c];
      kd[c] = row[d + c] + bk[c];
      vd[c] = row[2 * d + c] + bv[c];
    }
  }
}

void attention_core(int64_t t, int64_t d, int64_t heads, int64_t head_dim, float *out,
                    float *scratch, AttnPhases *phases) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t hd = head_dim;
  const int64_t slice = t * hd;
  auto Q = [&](int64_t h) { return scratch + (h * 3 + 0) * slice; };
  auto K = [&](int64_t h) { return scratch + (h * 3 + 1) * slice; };
  auto V = [&](int64_t h) { return scratch + (h * 3 + 2) * slice; };

  // A block of query rows is scored against each K row in turn, so K (and then
  // V) is read once per block rather than once per query row.
  constexpr int64_t QB = 8;
  const int64_t n_blocks = (t + QB - 1) / QB;
  const int64_t work = heads * n_blocks;
  const bool timed = phases != nullptr;
  double acc_s = 0, acc_m = 0, acc_v = 0;
  auto tick = [timed]() {
    return timed ? std::chrono::duration<double>(
                       std::chrono::steady_clock::now().time_since_epoch()).count()
                 : 0.0;
  };
#pragma omp parallel reduction(+ : acc_s, acc_m, acc_v) num_threads(::ow::omp_threads())
  {
    std::vector<float> s(static_cast<size_t>(QB) * static_cast<size_t>(t));
    std::vector<float> acc(static_cast<size_t>(QB) * static_cast<size_t>(head_dim));
    // DYNAMIC, and the obvious-looking alternative was measured and is worse.
    // `b` is head-major, so schedule(static) gives each thread a contiguous run
    // inside ONE head and keeps that head's K and V (384 KB each) in its private
    // cache -- which is the wrong optimisation, because it then has one thread
    // per head and 24 different heads live at once. Under dynamic the threads
    // move through the same head together and share one copy. Measured on the
    // nvidia golden: attention 1548.0 ms dynamic, 1892.9 ms static. The shared
    // cache beats the private one here.
    //
    // A vectorised exp for the softmax below was also built and REJECTED: 1 ulp
    // against expf (measured 1.192e-07), 3.0x on the softmax phase and 1.18x on
    // attention -- and it cost one of the twelve golden token paths, because
    // this encoder amplifies one ulp into a token flip. See
    // tasks/0179 Part 17 and its rejected/vector-exp.patch.
#pragma omp for schedule(dynamic, 1)
    for (int64_t b = 0; b < work; ++b) {
      const int64_t h = b / n_blocks;
      const int64_t q0 = (b % n_blocks) * QB;
      const int64_t nq = std::min<int64_t>(QB, t - q0);
      const float *qh = Q(h), *kh = K(h), *vh = V(h);

      const double c0 = tick();
      for (int64_t t2 = 0; t2 < t; ++t2) {
        const float *krow = kh + t2 * hd;
        for (int64_t qi = 0; qi < nq; ++qi)
          s[static_cast<size_t>(qi) * t + t2] = dot8(qh + (q0 + qi) * hd, krow, hd) * scale;
      }
      const double c1 = tick();
      acc_s += c1 - c0;

      // Row softmax, unchanged: max, exp, normalise, in that order.
      float inv[QB];
      for (int64_t qi = 0; qi < nq; ++qi) {
        float *sr = &s[static_cast<size_t>(qi) * t];
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t t2 = 0; t2 < t; ++t2)
          if (sr[t2] > mx) mx = sr[t2];
        float sum = 0.f;
        for (int64_t t2 = 0; t2 < t; ++t2) {
          const float e = std::exp(sr[t2] - mx);
          sr[t2] = e;
          sum += e;
        }
        inv[qi] = 1.0f / sum;
      }
      const double c2 = tick();
      acc_m += c2 - c1;

      // P.V. Each output element still accumulates over t2 in increasing order,
      // which is what keeps this bit-identical to the row-at-a-time version.
      // The `* inv` normalisation rides in here and is timed as P.V: it is
      // 1/head_dim of this loop's work, and an array P.V would carry it too.
      std::memset(acc.data(), 0, static_cast<size_t>(nq) * static_cast<size_t>(hd) * sizeof(float));
      for (int64_t t2 = 0; t2 < t; ++t2) {
        const float *vrow = vh + t2 * hd;
        for (int64_t qi = 0; qi < nq; ++qi)
          axpy8(&acc[static_cast<size_t>(qi) * hd], vrow,
                s[static_cast<size_t>(qi) * t + t2] * inv[qi], hd);
      }
      for (int64_t qi = 0; qi < nq; ++qi)
        std::memcpy(out + (q0 + qi) * d + h * hd, &acc[static_cast<size_t>(qi) * hd],
                    static_cast<size_t>(hd) * sizeof(float));
      acc_v += tick() - c2;
    }
  }
  if (phases) {
    phases->scores += acc_s;
    phases->softmax += acc_m;
    phases->values += acc_v;
  }
}

void attention(const float *qkv, int64_t m_padded, int64_t t, int64_t d, int64_t heads,
              int64_t head_dim, float *out, float *scratch, AttnPhases *phases) {
  zero_pad_rows(out, t, m_padded, d);
  const int64_t stride = 3 * d;
  const int64_t hd = head_dim;

  // scratch layout: head-major [h][q|k|v][t][head_dim], each slice contiguous.
  // This gather is UNCHANGED from before attention_core() was factored out
  // (bias is already applied to `qkv` by the caller) -- moving it into its
  // own function changed no arithmetic and no order.
  const int64_t slice = t * hd;
  auto Q = [&](int64_t h) { return scratch + (h * 3 + 0) * slice; };
  auto K = [&](int64_t h) { return scratch + (h * 3 + 1) * slice; };
  auto V = [&](int64_t h) { return scratch + (h * 3 + 2) * slice; };

  const int64_t gather_work = heads * t;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t idx = 0; idx < gather_work; ++idx) {
    const int64_t h = idx / t, t1 = idx % t;
    const float *row = qkv + t1 * stride + h * hd;
    std::memcpy(Q(h) + t1 * hd, row, static_cast<size_t>(hd) * sizeof(float));
    std::memcpy(K(h) + t1 * hd, row + d, static_cast<size_t>(hd) * sizeof(float));
    std::memcpy(V(h) + t1 * hd, row + 2 * d, static_cast<size_t>(hd) * sizeof(float));
  }

  attention_core(t, d, heads, head_dim, out, scratch, phases);
}

}  // namespace ow
