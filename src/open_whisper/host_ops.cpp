//===- host_ops.cpp ------------------------------------------*- C++ -*-===//
// open_whisper -- see host_ops.hpp. SPDX-License-Identifier: MIT
#include "host_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ow {

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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) out[i] = gelu_scalar(x[i]);
}

void gelu_bias(const float *x, int64_t rows, int64_t cols, const float *bias, float *out) {
#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < rows; ++r) {
    const float *xr = x + r * cols;
    float *orow = out + r * cols;
    for (int64_t c = 0; c < cols; ++c) orow[c] = gelu_scalar(xr[c] + bias[c]);
  }
}

void add_bias(float *y, const float *bias, int64_t rows, int64_t cols) {
#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < rows; ++r) {
    float *yr = y + r * cols;
    for (int64_t c = 0; c < cols; ++c) yr[c] += bias[c];
  }
}

void add_rows(const float *a, const float *b, int64_t rows, int64_t cols, float *out) {
  const int64_t n = rows * cols;
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void im2col(const float *x, int64_t t_in, int64_t c, int64_t stride, int64_t m_padded,
           float *out) {
  const int64_t k = 3 * c;
  std::memset(out, 0, static_cast<size_t>(m_padded) * static_cast<size_t>(k) * sizeof(float));
  const int64_t t_out = (t_in - 1) / stride + 1;
#pragma omp parallel for schedule(static)
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

void attention(const float *qkv, int64_t m_padded, int64_t t, int64_t d, int64_t heads,
              int64_t head_dim, float *out, float *scratch, AttnPhases *phases) {
  zero_pad_rows(out, t, m_padded, d);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t stride = 3 * d;
  const int64_t hd = head_dim;

  // scratch layout: head-major [h][q|k|v][t][head_dim], each slice contiguous.
  const int64_t slice = t * hd;
  auto Q = [&](int64_t h) { return scratch + (h * 3 + 0) * slice; };
  auto K = [&](int64_t h) { return scratch + (h * 3 + 1) * slice; };
  auto V = [&](int64_t h) { return scratch + (h * 3 + 2) * slice; };

  const int64_t gather_work = heads * t;
#pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < gather_work; ++idx) {
    const int64_t h = idx / t, t1 = idx % t;
    const float *row = qkv + t1 * stride + h * hd;
    std::memcpy(Q(h) + t1 * hd, row, static_cast<size_t>(hd) * sizeof(float));
    std::memcpy(K(h) + t1 * hd, row + d, static_cast<size_t>(hd) * sizeof(float));
    std::memcpy(V(h) + t1 * hd, row + 2 * d, static_cast<size_t>(hd) * sizeof(float));
  }

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
#pragma omp parallel reduction(+ : acc_s, acc_m, acc_v)
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

}  // namespace ow
