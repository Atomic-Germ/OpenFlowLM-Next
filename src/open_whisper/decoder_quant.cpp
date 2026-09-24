//===- decoder_quant.cpp --------------------------------------*- C++ -*-===//
// open_whisper -- see decoder_quant.hpp. SPDX-License-Identifier: MIT
#include "decoder_quant.hpp"
#include "host_ops.hpp"   // ow::omp_threads()

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ow {
namespace {

// Bit-identical to host_ops.cpp's from_bf16 / decoder.cpp's own copy --
// duplicated locally so this translation unit needs nothing beyond
// <cstring>, matching guards.cpp's "pulls in no XRT and no device" goal.
inline float from_bf16_local(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

#if defined(__AVX2__)
inline float hsum8(__m256 acc) {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
#endif

}  // namespace

// ---------------------------------------------------------------------
// Env parsing.
// ---------------------------------------------------------------------

// Defaults changed 2026-09-23 (task 0180 Parts 11-15): the WER gate on 1200
// utterances (LibriSpeech + FLEURS, 9 languages) found bf16 xkv / int8
// weights / int8x head statistically indistinguishable from the exact
// fp32/bf16/bf16 path under the hf protocol (H2Q vs H0: sign test p = 1.0,
// ΔWER -0.005pp [-0.04, +0.05]), while together saving ~158 MB/token
// (~1.6x on the decoder's DRAM traffic). "fp32"/"bf16" still restore the
// exact path exactly -- strict parsing is unchanged, only which value is
// unset's default.
XkvPrecision parse_xkv_precision() {
  const char *e = std::getenv("OW_DEC_XKV");
  if (!e || !*e) return XkvPrecision::BF16;
  const std::string v(e);
  if (v == "fp32") return XkvPrecision::FP32;
  if (v == "bf16") return XkvPrecision::BF16;
  throw std::runtime_error("OW_DEC_XKV is '" + v + "': expected 'fp32' or 'bf16'");
}

WeightPrecision parse_weight_precision() {
  const char *e = std::getenv("OW_DEC_W");
  if (!e || !*e) return WeightPrecision::INT8;
  const std::string v(e);
  if (v == "bf16") return WeightPrecision::BF16;
  if (v == "int8") return WeightPrecision::INT8;
  throw std::runtime_error("OW_DEC_W is '" + v + "': expected 'bf16' or 'int8'");
}

HeadPrecision parse_head_precision() {
  const char *e = std::getenv("OW_DEC_HEAD");
  if (!e || !*e) return HeadPrecision::INT8X;
  const std::string v(e);
  if (v == "bf16") return HeadPrecision::BF16;
  if (v == "int8") return HeadPrecision::INT8;
  if (v == "int8x") return HeadPrecision::INT8X;
  throw std::runtime_error("OW_DEC_HEAD is '" + v + "': expected 'bf16', 'int8' or 'int8x'");
}

const char *to_string(XkvPrecision p) {
  switch (p) {
    case XkvPrecision::FP32: return "fp32";
    case XkvPrecision::BF16: return "bf16";
  }
  return "?";
}
const char *to_string(WeightPrecision p) {
  switch (p) {
    case WeightPrecision::BF16: return "bf16";
    case WeightPrecision::INT8: return "int8";
  }
  return "?";
}
const char *to_string(HeadPrecision p) {
  switch (p) {
    case HeadPrecision::BF16: return "bf16";
    case HeadPrecision::INT8: return "int8";
    case HeadPrecision::INT8X: return "int8x";
  }
  return "?";
}

// ---------------------------------------------------------------------
// int8 quantization.
// ---------------------------------------------------------------------

QLinear quantize_int8_rows_f32(const float *w_f32, const std::vector<float> &bias, int64_t out,
                               int64_t in) {
  if (static_cast<int64_t>(bias.size()) != out)
    throw std::runtime_error("quantize_int8_rows_f32: bias has " + std::to_string(bias.size()) +
                             " entries, expected " + std::to_string(out));
  QLinear Q;
  Q.out = out;
  Q.in = in;
  Q.w.assign(static_cast<size_t>(out) * static_cast<size_t>(in), 0);
  Q.scale.assign(static_cast<size_t>(out), 0.f);
  Q.b = bias;

  for (int64_t o = 0; o < out; ++o) {
    const float *row = w_f32 + static_cast<size_t>(o) * static_cast<size_t>(in);
    float amax = 0.f;
    for (int64_t i = 0; i < in; ++i) amax = std::max(amax, std::fabs(row[i]));
    // An all-zero row: scale stays 0 (already assigned), weights stay 0
    // (already assigned) -- the row/amax division below is never reached,
    // so there is no 0/0 to produce a NaN from.
    if (amax == 0.f) continue;
    const float s = amax / 127.0f;
    Q.scale[static_cast<size_t>(o)] = s;
    int8_t *dst = Q.w.data() + static_cast<size_t>(o) * static_cast<size_t>(in);
    for (int64_t i = 0; i < in; ++i) {
      // Round to nearest, ties away from zero (std::lround); clamp to
      // [-127, 127] -- NOT [-128, 127] -- so the range stays symmetric and
      // scale = amax/127 is exact for the element that hit amax (it always
      // rounds to exactly +-127, never overflowing into -128's asymmetric
      // slot).
      long qi = std::lround(static_cast<double>(row[i]) / static_cast<double>(s));
      if (qi > 127) qi = 127;
      if (qi < -127) qi = -127;
      dst[i] = static_cast<int8_t>(qi);
    }
  }
  return Q;
}

QLinear quantize_int8_rows_bf16(const uint16_t *w_bf16, const std::vector<float> &bias,
                                int64_t out, int64_t in) {
  // Widen once, then reuse the fp32 quantizer -- quantization runs once per
  // tensor at load time, not on any hot path, so the extra pass costs
  // nothing that matters against the GEMV work that follows every step.
  std::vector<float> w_f32(static_cast<size_t>(out) * static_cast<size_t>(in));
  for (size_t i = 0; i < w_f32.size(); ++i) w_f32[i] = from_bf16_local(w_bf16[i]);
  return quantize_int8_rows_f32(w_f32.data(), bias, out, in);
}

#if defined(__AVX2__)
inline float dot_int8_row_avx2(const float *x, const int8_t *w, int64_t n) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(w + i));
    const __m256i w32 = _mm256_cvtepi8_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(w32);
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), wf, acc);
  }
  float s = hsum8(acc);
  for (; i < n; ++i) s += x[i] * static_cast<float>(w[i]);
  return s;
}
#else
inline float dot_int8_row_avx2(const float *x, const int8_t *w, int64_t n) {
  float s = 0.f;
  for (int64_t i = 0; i < n; ++i) s += x[i] * static_cast<float>(w[i]);
  return s;
}
#endif

void linear_int8(const float *x, const QLinear &W, float *y) {
  const int64_t out = W.out, in = W.in;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t ob = 0; ob < out; ob += 4) {
    const int64_t rows = std::min<int64_t>(4, out - ob);
#if defined(__AVX2__)
    if (rows == 4) {
      // Four output rows per iteration: one load of each 8-wide chunk of the
      // (row-independent) activation `x` feeds four independent FMA chains,
      // one per row's own int8 weights -- fewer loads of `x` than four
      // separate dot_int8_row_avx2() calls would issue, and four
      // independent accumulators give the pipeline four in-flight chains
      // instead of one.
      const int8_t *w0 = W.w.data() + static_cast<size_t>(ob + 0) * static_cast<size_t>(in);
      const int8_t *w1 = W.w.data() + static_cast<size_t>(ob + 1) * static_cast<size_t>(in);
      const int8_t *w2 = W.w.data() + static_cast<size_t>(ob + 2) * static_cast<size_t>(in);
      const int8_t *w3 = W.w.data() + static_cast<size_t>(ob + 3) * static_cast<size_t>(in);
      __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
      int64_t i = 0;
      for (; i + 8 <= in; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        acc0 = _mm256_fmadd_ps(
            xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(w0 + i)))), acc0);
        acc1 = _mm256_fmadd_ps(
            xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(w1 + i)))), acc1);
        acc2 = _mm256_fmadd_ps(
            xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(w2 + i)))), acc2);
        acc3 = _mm256_fmadd_ps(
            xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(w3 + i)))), acc3);
      }
      float s0 = hsum8(acc0), s1 = hsum8(acc1), s2 = hsum8(acc2), s3 = hsum8(acc3);
      for (; i < in; ++i) {
        s0 += x[i] * static_cast<float>(w0[i]);
        s1 += x[i] * static_cast<float>(w1[i]);
        s2 += x[i] * static_cast<float>(w2[i]);
        s3 += x[i] * static_cast<float>(w3[i]);
      }
      y[static_cast<size_t>(ob + 0)] = s0 * W.scale[static_cast<size_t>(ob + 0)] + W.b[static_cast<size_t>(ob + 0)];
      y[static_cast<size_t>(ob + 1)] = s1 * W.scale[static_cast<size_t>(ob + 1)] + W.b[static_cast<size_t>(ob + 1)];
      y[static_cast<size_t>(ob + 2)] = s2 * W.scale[static_cast<size_t>(ob + 2)] + W.b[static_cast<size_t>(ob + 2)];
      y[static_cast<size_t>(ob + 3)] = s3 * W.scale[static_cast<size_t>(ob + 3)] + W.b[static_cast<size_t>(ob + 3)];
      continue;
    }
#endif
    for (int64_t j = 0; j < rows; ++j) {
      const int64_t o = ob + j;
      const float s = dot_int8_row_avx2(x, W.w.data() + static_cast<size_t>(o) * static_cast<size_t>(in), in);
      y[static_cast<size_t>(o)] = s * W.scale[static_cast<size_t>(o)] + W.b[static_cast<size_t>(o)];
    }
  }
}

double linear_int8_row_ref_f64(const float *x, const int8_t *w_row, float scale, float bias,
                               int64_t n) {
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i)
    s += static_cast<double>(x[i]) * static_cast<double>(w_row[i]);
  return s * static_cast<double>(scale) + static_cast<double>(bias);
}

// ---------------------------------------------------------------------
// bf16-widening dot / axpy.
// ---------------------------------------------------------------------

#if defined(__AVX2__)
float dot_bf16_kernel(const float *x, const uint16_t *w, int64_t n) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i));
    const __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), wf, acc);
  }
  float s = hsum8(acc);
  for (; i < n; ++i) s += x[i] * from_bf16_local(w[i]);
  return s;
}
void axpy_bf16_kernel(float *y, const uint16_t *v, float alpha, int64_t n) {
  const __m256 av = _mm256_set1_ps(alpha);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(v + i));
    const __m256 vf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, vf, _mm256_loadu_ps(y + i)));
  }
  for (; i < n; ++i) y[i] += alpha * from_bf16_local(v[i]);
}
#else
float dot_bf16_kernel(const float *x, const uint16_t *w, int64_t n) {
  float s = 0.f;
  for (int64_t i = 0; i < n; ++i) s += x[i] * from_bf16_local(w[i]);
  return s;
}
void axpy_bf16_kernel(float *y, const uint16_t *v, float alpha, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += alpha * from_bf16_local(v[i]);
}
#endif

double dot_bf16_ref_f64(const float *x, const uint16_t *w, int64_t n) {
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i)
    s += static_cast<double>(x[i]) * static_cast<double>(from_bf16_local(w[i]));
  return s;
}

// attend_one_xkv_bf16: same shape/order as decoder.cpp's private attend_one,
// over bf16 K/V. See decoder_quant.hpp's comment.
void attend_one_xkv_bf16(const float *q, const uint16_t *k_base, int64_t k_row_stride,
                         int64_t k_head_stride, const uint16_t *v_base, int64_t v_row_stride,
                         int64_t v_head_stride, int64_t len, int64_t heads, int64_t head_dim,
                         float scale, float *out, float *scores_scratch, int64_t scores_stride) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t h = 0; h < heads; ++h) {
    float *scores = scores_scratch + h * scores_stride;
    const float *qh = q + h * head_dim;
    const uint16_t *kh = k_base + h * k_head_stride;
    const uint16_t *vh = v_base + h * v_head_stride;
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t t = 0; t < len; ++t) {
      const uint16_t *krow = kh + t * k_row_stride;
      const float s = dot_bf16_kernel(qh, krow, head_dim) * scale;
      scores[t] = s;
      if (s > mx) mx = s;
    }
    float sum = 0.f;
    for (int64_t t = 0; t < len; ++t) {
      const float e = std::exp(scores[t] - mx);
      scores[t] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    float *oh = out + h * head_dim;
    std::memset(oh, 0, static_cast<size_t>(head_dim) * sizeof(float));
    for (int64_t t = 0; t < len; ++t)
      axpy_bf16_kernel(oh, vh + t * v_row_stride, scores[t] * inv, head_dim);
  }
}

// ---------------------------------------------------------------------
// int8x top-K exact recompute.
// ---------------------------------------------------------------------

void recompute_top_k_exact(const float *x, int64_t out, int64_t in, int64_t k,
                           const uint16_t *w_bf16, const float *bias, float *logits) {
  const int64_t kk = std::min<int64_t>(k, out);
  std::vector<int64_t> idx(static_cast<size_t>(out));
  std::iota(idx.begin(), idx.end(), int64_t{0});
  std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                    [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
  // The header's own reasoning ("the argmax is exact ... whether or not any of the
  // other out-K rows' int8 approximations are close") was too strong (PR #111 review,
  // finding 9): it only checked that the TRUE max, if ranked within the top K by its
  // OWN (possibly noisy) approximation, gets recomputed -- it did not check that some
  // OTHER, un-recomputed row's approximation could still OVERESTIMATE past the now-exact
  // corrected max. That happens precisely when the true-max row's own int8 approximation
  // overestimates ITS true value (pushing its pre-recompute rank threshold above the
  // true max), leaving room in between for an excluded row's inflated approximation to
  // beat the exact winner post-recompute. Fixed by capping every NON-recomputed logit at
  // min(the kk exact values): a capped row can never win the argmax against a correctly
  // recomputed one, and since the kk rows are themselves only approximations of "which
  // rows might be the max" anyway, lowering an excluded row's value costs nothing real --
  // it was never going to be trusted as exact either way.
  float min_exact = std::numeric_limits<float>::infinity();
  for (int64_t j = 0; j < kk; ++j) {
    const int64_t o = idx[static_cast<size_t>(j)];
    const float exact = dot_bf16_kernel(x, w_bf16 + static_cast<size_t>(o) * static_cast<size_t>(in), in) +
                        (bias ? bias[o] : 0.f);
    logits[o] = exact;
    if (exact < min_exact) min_exact = exact;
  }
  for (int64_t j = kk; j < out; ++j) {
    const int64_t o = idx[static_cast<size_t>(j)];
    if (logits[o] > min_exact) logits[o] = min_exact;
  }
}

}  // namespace ow
