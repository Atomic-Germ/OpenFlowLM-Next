//===- gemma_kernels.cpp -------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- EmbeddingGemma-300M host eltwise kernels. See
// gemma_kernels.hpp for the contract and the primary-source citations.
// SPDX-License-Identifier: MIT
//
// AVX2 is used only where it does not change which precision the reference
// computes in -- CLAUDE.md rule 2 / this project's own eltwise precedent
// (main.cpp's layer_norm_cpu, gelu_cpu) is that host elementwise ops are
// cheap relative to GEMM time (F1/F2/F3), so there is no reason to trade
// numerical fidelity for speed here:
//   * rms_norm_cpu vectorizes the double-precision reduction and the final
//     double-precision multiply (4-wide __m256d) -- same precision as the
//     scalar path, just parallel lanes of it.
//   * apply_rope_cpu vectorizes in float32, which is what the reference
//     itself computes in (no upcast) -- so this is not a precision trade
//     either.
//   * geglu_cpu is left scalar. Its double-precision formula calls
//     std::tanh() per element, which has no AVX2 intrinsic; vectorizing the
//     cheap arithmetic around a scalar tanh() call buys little and risks
//     breaking the reference's exact two-stage-rounding contract (see the
//     header) for a function whose whole cost is dominated by the
//     unavoidable scalar transcendental anyway.

#include "gemma_kernels.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace npue {

namespace {

// sqrt(2/pi) to double precision, matching math.sqrt(2.0/math.pi) in
// reference/encoder_gemma.py's gelu_tanh().
constexpr double kGeluTanhC = 0.79788456080286535588;

#if defined(__AVX2__)
inline double hsum_pd256(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  lo = _mm_add_pd(lo, hi);
  __m128d hi2 = _mm_unpackhi_pd(lo, lo);
  lo = _mm_add_sd(lo, hi2);
  return _mm_cvtsd_f64(lo);
}
#endif

} // namespace

void rms_norm_cpu(const float *x, const float *weight, float *out,
                   int64_t rows, int64_t dim, float eps) {
  const double epsd = static_cast<double>(eps);
  for (int64_t r = 0; r < rows; ++r) {
    const float *row = x + r * dim;
    float *orow = out + r * dim;

    double sumsq = 0.0;
    int64_t j = 0;
#if defined(__AVX2__)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    for (; j + 8 <= dim; j += 8) {
      __m256 v = _mm256_loadu_ps(row + j);
      __m128 vlo = _mm256_castps256_ps128(v);
      __m128 vhi = _mm256_extractf128_ps(v, 1);
      __m256d dlo = _mm256_cvtps_pd(vlo);
      __m256d dhi = _mm256_cvtps_pd(vhi);
      acc0 = _mm256_fmadd_pd(dlo, dlo, acc0);
      acc1 = _mm256_fmadd_pd(dhi, dhi, acc1);
    }
    sumsq = hsum_pd256(acc0) + hsum_pd256(acc1);
#endif
    for (; j < dim; ++j) {
      const double v = static_cast<double>(row[j]);
      sumsq += v * v;
    }

    const double var = sumsq / static_cast<double>(dim);
    const double inv_rms = 1.0 / std::sqrt(var + epsd);

    int64_t k = 0;
#if defined(__AVX2__)
    const __m256d inv_rms_v = _mm256_set1_pd(inv_rms);
    const __m256d one = _mm256_set1_pd(1.0);
    for (; k + 4 <= dim; k += 4) {
      __m128 vf = _mm_loadu_ps(row + k);
      __m128 wf = _mm_loadu_ps(weight + k);
      __m256d vd = _mm256_cvtps_pd(vf);
      __m256d wd = _mm256_cvtps_pd(wf);
      __m256d res =
          _mm256_mul_pd(_mm256_mul_pd(vd, inv_rms_v), _mm256_add_pd(wd, one));
      __m128 resf = _mm256_cvtpd_ps(res);
      _mm_storeu_ps(orow + k, resf);
    }
#endif
    for (; k < dim; ++k) {
      const double v = static_cast<double>(row[k]) * inv_rms *
                        (1.0 + static_cast<double>(weight[k]));
      orow[k] = static_cast<float>(v);
    }
  }
}

bool gemma_is_full_attention_layer(int64_t layer_idx,
                                    int64_t sliding_window_pattern) {
  return ((layer_idx + 1) % sliding_window_pattern) == 0;
}

void gemma_rope_tables(int64_t seq_len, int64_t head_dim, double base,
                        float *cos_out, float *sin_out) {
  const int64_t half = head_dim / 2;
  std::vector<double> inv_freq(static_cast<size_t>(half));
  for (int64_t j = 0; j < half; ++j) {
    const double exponent = (2.0 * static_cast<double>(j)) /
                             static_cast<double>(head_dim);
    inv_freq[static_cast<size_t>(j)] = 1.0 / std::pow(base, exponent);
  }
  for (int64_t s = 0; s < seq_len; ++s) {
    float *cs = cos_out + s * head_dim;
    float *sn = sin_out + s * head_dim;
    for (int64_t j = 0; j < half; ++j) {
      const double ang = static_cast<double>(s) * inv_freq[static_cast<size_t>(j)];
      const float c = static_cast<float>(std::cos(ang));
      const float si = static_cast<float>(std::sin(ang));
      cs[j] = c;
      cs[half + j] = c;
      sn[j] = si;
      sn[half + j] = si;
    }
  }
}

void apply_rope_cpu(const float *x, const float *cos, const float *sin,
                     float *out, int64_t rows, int64_t seq_len,
                     int64_t head_dim) {
  const int64_t half = head_dim / 2;
  std::vector<float> tmp(static_cast<size_t>(head_dim));
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t s = r % seq_len;
    const float *row = x + r * head_dim;
    const float *c = cos + s * head_dim;
    const float *sn = sin + s * head_dim;
    // Copy first: out[j] reads row[j] AND row[half+j] (rotate_half), so an
    // in-place call (x == out) must not let the first half's write clobber
    // the second half's read.
    std::memcpy(tmp.data(), row, static_cast<size_t>(head_dim) * sizeof(float));
    float *orow = out + r * head_dim;

    int64_t j = 0;
#if defined(__AVX2__)
    for (; j + 8 <= half; j += 8) {
      __m256 x1 = _mm256_loadu_ps(tmp.data() + j);
      __m256 x2 = _mm256_loadu_ps(tmp.data() + half + j);
      __m256 c1 = _mm256_loadu_ps(c + j);
      __m256 c2 = _mm256_loadu_ps(c + half + j);
      __m256 s1 = _mm256_loadu_ps(sn + j);
      __m256 s2 = _mm256_loadu_ps(sn + half + j);
      // out[j]      = x1*c1 - x2*s1
      // out[half+j] = x2*c2 + x1*s2
      __m256 negx2 = _mm256_sub_ps(_mm256_setzero_ps(), x2);
      __m256 o1 = _mm256_fmadd_ps(x1, c1, _mm256_mul_ps(negx2, s1));
      __m256 o2 = _mm256_fmadd_ps(x2, c2, _mm256_mul_ps(x1, s2));
      _mm256_storeu_ps(orow + j, o1);
      _mm256_storeu_ps(orow + half + j, o2);
    }
#endif
    for (; j < half; ++j) {
      const float x1 = tmp[static_cast<size_t>(j)];
      const float x2 = tmp[static_cast<size_t>(half + j)];
      orow[j] = x1 * c[j] - x2 * sn[j];
      orow[half + j] = x2 * c[half + j] + x1 * sn[half + j];
    }
  }
}

void geglu_cpu(const float *gate, const float *up, float *out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const double x = static_cast<double>(gate[i]);
    const double inner = kGeluTanhC * (x + 0.044715 * x * x * x);
    const double t = std::tanh(inner);
    // gelu_tanh() rounds to float32 HERE in the reference before the
    // caller multiplies by `up` -- matching that intermediate rounding
    // (not fusing everything into one double expression) is required for
    // bit-for-bit agreement, not just close agreement.
    const float act = static_cast<float>(0.5 * x * (1.0 + t));
    const double prod =
        static_cast<double>(act) * static_cast<double>(up[i]);
    out[i] = static_cast<float>(prod);
  }
}

} // namespace npue
