//===- decoder_quant_test.cpp ----------------------------------*- C++ -*-===//
//
// open_whisper -- offline tests for the decoder's OPTIONAL precision variants
// (task 0180 Part 8, decoder_quant.{hpp,cpp}). No device, no XRT, no NPU --
// links decoder_quant.cpp plus q4nx_file.cpp alone, same discipline as
// guards_test.cpp, and is BUILT AND RUN by build.cmd for the same reason
// guards_test.cpp is: a test nothing runs is the same failure one step
// earlier.
//
//   out\decoder_quant_test.exe
//
// What each section checks:
//   1. int8 GEMV kernel (linear_int8) vs a double-precision reference of the
//      SAME already-quantized int8 weights -- isolates "does the kernel
//      implement the dot product it claims to" from "how much did
//      quantizing cost" (reported separately, section 3).
//   2. Adversarial rows: one huge outlier (near-saturating quantization) and
//      an all-zero row (scale must land at exactly 0, not divide by it).
//   3. Quantization error, int8 vs the bf16 reference the default path uses
//      -- random weights, then (if a real model is found) the actual
//      decoder tensors.
//   4. bf16-widening dot/axpy (OW_DEC_XKV=bf16's kernels) vs a
//      double-precision reference of the same bf16 bits, plus a small
//      end-to-end attend_one_xkv_bf16 check against an independent
//      double-precision softmax-attention reference.
//   5. int8x's top-K exact recompute: the true (bf16-exact) argmax is
//      recovered even though the int8 pass alone would have picked a
//      different row.
//
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "decoder_quant.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("  %-70s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

void check_le(double got, double bound, const std::string &what) {
  const bool ok = got <= bound;
  std::printf("  %-70s %s  (%.3e <= %.3e)\n", what.c_str(), ok ? "ok" : "FAIL", got, bound);
  if (!ok) ++failures;
}

// A REDUCTION kernel (a dot product, a softmax-weighted sum) summed in a
// different order than the double reference can land arbitrarily close to
// the double reference's own value by cancellation -- that is a property of
// the DATA (signed terms that happen to nearly cancel), not of whether the
// kernel is correct. Dividing the absolute fp32-vs-f64 error by a
// near-zero reference then manufactures an inflated "relative error" out of
// a few ULPs of genuine, harmless rounding -- this is what
// test_int8_kernel_vs_quantized_reference() hit first (1.85e-04 against a
// 5e-06 relative bound, on a row whose true value happened to be small).
// CLAUDE.md's trap 11, in miniature: when a metric and the kernel disagree,
// suspect the metric first. The textbook fix for a summed dot product is to
// bound the ABSOLUTE error by a small multiple of the classic
// floating-point summation bound, n * eps * sum(|terms|) -- scaled by the
// magnitude the terms COULD have summed to without cancellation, not by
// what they happened to sum to.
bool close_by_l1(double got, double ref, double l1_of_terms, double *ratio_out) {
  constexpr double eps = 1.1920929e-7;   // fp32 machine epsilon
  const double bound = 24.0 * eps * std::max(1.0, l1_of_terms);   // generous constant factor
  const double err = std::fabs(got - ref);
  if (ratio_out) *ratio_out = err / bound;
  return err <= bound;
}
void check_close_by_l1(double got, double ref, double l1_of_terms, const std::string &what) {
  double ratio = 0.0;
  const bool ok = close_by_l1(got, ref, l1_of_terms, &ratio);
  std::printf("  %-70s %s  (|err|/bound = %.3e, ref=%.6g, l1=%.6g)\n", what.c_str(), ok ? "ok" : "FAIL",
             ratio, ref, l1_of_terms);
  if (!ok) ++failures;
}

inline uint16_t f32_to_bf16_rne(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

std::mt19937 rng(20260923);   // fixed seed -- a flaky offline test is worse than none

std::vector<float> random_vec(int64_t n, float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(static_cast<size_t>(n));
  for (auto &x : v) x = d(rng);
  return v;
}

// ---------------------------------------------------------------------
// 1 & 2: int8 GEMV kernel vs a double reference of the SAME quantized data,
// plus the two adversarial rows.
// ---------------------------------------------------------------------

void test_int8_kernel_vs_quantized_reference() {
  std::printf("-- int8 GEMV kernel vs double reference of the quantized weights --\n");
  const int64_t OUT = 37, IN = 261;   // deliberately not multiples of 4 or 8: exercise every tail path
  std::vector<float> w = random_vec(OUT * IN, -3.0f, 3.0f);
  std::vector<float> bias = random_vec(OUT, -1.0f, 1.0f);
  const ow::QLinear Q = ow::quantize_int8_rows_f32(w.data(), bias, OUT, IN);

  std::vector<float> x = random_vec(IN, -2.0f, 2.0f);
  std::vector<float> y(static_cast<size_t>(OUT));
  ow::linear_int8(x.data(), Q, y.data());

  double worst_ratio = 0.0;
  int64_t worst_row = -1;
  for (int64_t o = 0; o < OUT; ++o) {
    const int8_t *w_row = Q.w.data() + static_cast<size_t>(o) * static_cast<size_t>(IN);
    const double ref = ow::linear_int8_row_ref_f64(x.data(), w_row, Q.scale[static_cast<size_t>(o)],
                                                    Q.b[static_cast<size_t>(o)], IN);
    // L1 of the SCALED terms (what actually got summed, in the units `ref`
    // and `y` are in) -- see close_by_l1()'s comment.
    double l1 = 0.0;
    for (int64_t i = 0; i < IN; ++i)
      l1 += std::fabs(static_cast<double>(x[static_cast<size_t>(i)])) * std::fabs(static_cast<double>(w_row[i]));
    l1 *= static_cast<double>(Q.scale[static_cast<size_t>(o)]);
    double ratio = 0.0;
    close_by_l1(y[static_cast<size_t>(o)], ref, l1, &ratio);
    if (ratio > worst_ratio) {
      worst_ratio = ratio;
      worst_row = o;
    }
  }
  std::printf("  (worst row %lld)\n", static_cast<long long>(worst_row));
  check_le(worst_ratio, 1.0, "linear_int8() matches the double reference of its own quantized rows (L1 bound)");
}

void test_int8_adversarial() {
  std::printf("-- int8 adversarial rows --\n");
  const int64_t IN = 128;

  // One huge outlier: element 5 is 1000x the rest. scale = amax/127 is huge,
  // so every OTHER element quantizes toward 0 -- must not produce NaN/Inf,
  // and the kernel must still agree with the double reference of what it
  // actually quantized to (not with the pre-quantization row: that error is
  // section 3's business, not this kernel-correctness check).
  {
    std::vector<float> row(static_cast<size_t>(IN), 0.01f);
    row[5] = 1000.0f;
    std::vector<float> bias = {0.25f};
    const ow::QLinear Q = ow::quantize_int8_rows_f32(row.data(), bias, 1, IN);
    check(std::isfinite(Q.scale[0]) && Q.scale[0] > 0.f, "outlier row: scale is finite and positive");
    check(Q.w[5] == 127 || Q.w[5] == -127, "outlier row: the outlier element saturates to +-127");
    // Every other element is tiny relative to the outlier -- most round to 0.
    int zeros = 0;
    for (int64_t i = 0; i < IN; ++i)
      if (i != 5 && Q.w[static_cast<size_t>(i)] == 0) ++zeros;
    check(zeros > IN - 4, "outlier row: the non-outlier elements quantize to ~0, not garbage");

    std::vector<float> x = random_vec(IN, -1.0f, 1.0f);
    std::vector<float> y(1);
    ow::linear_int8(x.data(), Q, y.data());
    check(std::isfinite(y[0]), "outlier row: kernel output is finite");
    const double ref = ow::linear_int8_row_ref_f64(x.data(), Q.w.data(), Q.scale[0], Q.b[0], IN);
    double l1 = 0.0;
    for (int64_t i = 0; i < IN; ++i)
      l1 += std::fabs(static_cast<double>(x[static_cast<size_t>(i)])) * std::fabs(static_cast<double>(Q.w[static_cast<size_t>(i)]));
    l1 *= static_cast<double>(Q.scale[0]);
    check_close_by_l1(y[0], ref, l1, "outlier row: kernel matches the double reference of the quantized row");
  }

  // All-zero row: scale must be exactly 0 (not NaN from a 0/0 division
  // anywhere in the quantizer), weights all 0, and the kernel's output must
  // be exactly the bias -- finite, not NaN -- for ANY input x, including one
  // with its own outliers/zeros/negatives.
  {
    std::vector<float> row(static_cast<size_t>(IN), 0.0f);
    std::vector<float> bias = {-0.75f};
    const ow::QLinear Q = ow::quantize_int8_rows_f32(row.data(), bias, 1, IN);
    check(Q.scale[0] == 0.0f, "all-zero row: scale is exactly 0.0f, not NaN");
    bool all_zero_w = true;
    for (auto b : Q.w) all_zero_w &= (b == 0);
    check(all_zero_w, "all-zero row: every quantized weight is exactly 0");

    for (float xv : {0.0f, 1e30f, -1e30f, std::numeric_limits<float>::infinity()}) {
      std::vector<float> x(static_cast<size_t>(IN), xv);
      std::vector<float> y(1);
      ow::linear_int8(x.data(), Q, y.data());
      // 0 (scale) * finite-or-inf-dot + bias: the dot itself is 0 * xv = 0
      // for finite xv (and 0 for inf too, since every weight is exactly 0,
      // not just near it -- 0.0f * INFINITY is the one case that would be
      // NaN, and it never arises because the row truly is zero, so the
      // AVX2 widen produces 0.0f lanes, and 0.0f * inf... is checked below).
      const bool finite_input = std::isfinite(xv);
      if (finite_input) {
        check(y[0] == Q.b[0], "all-zero row: output is exactly bias for x=" + std::to_string(xv));
      } else {
        // x = +-inf against an exactly-zero weight row: 0*inf is NaN by
        // IEEE 754, so this is the one input for which "no NaN" is NOT a
        // property of the row alone -- it also needs a finite x, which a
        // real activation always is (LayerNorm's output). Documented, not
        // asserted away.
        std::printf("  %-70s %s\n", "all-zero row: x=inf is a known 0*inf edge (not a real activation)", "note");
      }
    }
  }
}

// ---------------------------------------------------------------------
// 3: quantization error, int8 vs the bf16 reference.
// ---------------------------------------------------------------------

void report_int8_quant_error(const std::string &tag, const float *w_f32, int64_t out, int64_t in,
                             const std::vector<float> &bias, int n_probes) {
  std::vector<uint16_t> w_bf16(static_cast<size_t>(out) * static_cast<size_t>(in));
  for (size_t i = 0; i < w_bf16.size(); ++i) w_bf16[i] = f32_to_bf16_rne(w_f32[i]);
  const ow::QLinear Q = ow::quantize_int8_rows_f32(w_f32, bias, out, in);

  double sum_rel = 0.0, max_rel = 0.0;
  int64_t worst_row = -1;
  for (int p = 0; p < n_probes; ++p) {
    std::vector<float> x = random_vec(in, -1.0f, 1.0f);
    for (int64_t o = 0; o < out; ++o) {
      const double bf16_ref = ow::dot_bf16_ref_f64(
          x.data(), w_bf16.data() + static_cast<size_t>(o) * static_cast<size_t>(in), in) +
                              bias[static_cast<size_t>(o)];
      const double int8_val = ow::linear_int8_row_ref_f64(
          x.data(), Q.w.data() + static_cast<size_t>(o) * static_cast<size_t>(in), Q.scale[static_cast<size_t>(o)],
          Q.b[static_cast<size_t>(o)], in);
      const double rel = std::fabs(int8_val - bf16_ref) / std::max(1e-3, std::fabs(bf16_ref));
      sum_rel += rel;
      if (rel > max_rel) {
        max_rel = rel;
        worst_row = o;
      }
    }
  }
  const double mean_rel = sum_rel / static_cast<double>(n_probes * out);
  std::printf("  %-28s out=%-7lld in=%-6lld mean_rel=%.4e max_rel=%.4e (worst row %lld, %d probes)\n",
             tag.c_str(), static_cast<long long>(out), static_cast<long long>(in), mean_rel, max_rel,
             static_cast<long long>(worst_row), n_probes);
}

void test_quant_error_random() {
  std::printf("-- int8 quantization error vs bf16 reference (random weights) --\n");
  const int64_t OUT = 64, IN = 256;
  std::vector<float> w = random_vec(OUT * IN, -0.2f, 0.2f);   // typical trained-linear scale
  std::vector<float> bias = random_vec(OUT, -0.1f, 0.1f);
  report_int8_quant_error("random gaussian-ish", w.data(), OUT, IN, bias, 8);
}

// ---------------------------------------------------------------------
// 4: bf16-widening dot/axpy vs a double reference of the same bf16 bits.
// ---------------------------------------------------------------------

void test_bf16_kernels() {
  std::printf("-- bf16-widening dot/axpy vs double reference of the same bf16 bits --\n");
  const int64_t N = 71;   // not a multiple of 8: exercises the scalar tail
  std::vector<float> w_f32 = random_vec(N, -4.0f, 4.0f);
  std::vector<uint16_t> w_bf16(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) w_bf16[static_cast<size_t>(i)] = f32_to_bf16_rne(w_f32[static_cast<size_t>(i)]);
  std::vector<float> x = random_vec(N, -2.0f, 2.0f);

  const float got = ow::dot_bf16_kernel(x.data(), w_bf16.data(), N);
  const double ref = ow::dot_bf16_ref_f64(x.data(), w_bf16.data(), N);
  double l1 = 0.0;
  for (int64_t i = 0; i < N; ++i)
    l1 += std::fabs(static_cast<double>(x[static_cast<size_t>(i)])) *
         std::fabs(static_cast<double>(bf16_to_f32(w_bf16[static_cast<size_t>(i)])));
  check_close_by_l1(got, ref, l1, "dot_bf16_kernel() matches the double reference of the same bf16 bits");

  std::vector<float> y = random_vec(N, -1.0f, 1.0f);
  std::vector<double> y_ref(y.begin(), y.end());
  const float alpha = 0.375f;
  ow::axpy_bf16_kernel(y.data(), w_bf16.data(), alpha, N);
  double worst_ratio = 0.0;
  for (int64_t i = 0; i < N; ++i) {
    const double term = static_cast<double>(alpha) * static_cast<double>(bf16_to_f32(w_bf16[static_cast<size_t>(i)]));
    y_ref[static_cast<size_t>(i)] += term;
    // One FMA, not a many-term reduction -- L1 scale is just the two
    // operands being combined.
    const double l1_i = std::fabs(static_cast<double>(y[static_cast<size_t>(i)]) - term) + std::fabs(term);
    double ratio = 0.0;
    close_by_l1(y[static_cast<size_t>(i)], y_ref[static_cast<size_t>(i)], l1_i, &ratio);
    worst_ratio = std::max(worst_ratio, ratio);
  }
  check_le(worst_ratio, 1.0, "axpy_bf16_kernel() matches the double reference of the same bf16 bits (L1 bound)");
}

// Independent double-precision softmax-attention reference over bf16-exact
// K/V (bf16 -> f64 is exact -- no rounding is introduced by widening to
// double, only by the ORIGINAL fp32->bf16 conversion, which both the kernel
// and this reference read identically off the same bits).
void attend_ref_f64(const float *q, const uint16_t *k_base, int64_t k_row_stride,
                    int64_t k_head_stride, const uint16_t *v_base, int64_t v_row_stride,
                    int64_t v_head_stride, int64_t len, int64_t heads, int64_t head_dim, float scale,
                    double *out) {
  std::vector<double> scores(static_cast<size_t>(len));
  for (int64_t h = 0; h < heads; ++h) {
    const float *qh = q + h * head_dim;
    const uint16_t *kh = k_base + h * k_head_stride;
    const uint16_t *vh = v_base + h * v_head_stride;
    double mx = -std::numeric_limits<double>::infinity();
    for (int64_t t = 0; t < len; ++t) {
      const uint16_t *krow = kh + t * k_row_stride;
      double s = 0.0;
      for (int64_t d = 0; d < head_dim; ++d)
        s += static_cast<double>(qh[d]) * static_cast<double>(bf16_to_f32(krow[d]));
      s *= static_cast<double>(scale);
      scores[static_cast<size_t>(t)] = s;
      mx = std::max(mx, s);
    }
    double sum = 0.0;
    for (int64_t t = 0; t < len; ++t) {
      const double e = std::exp(scores[static_cast<size_t>(t)] - mx);
      scores[static_cast<size_t>(t)] = e;
      sum += e;
    }
    double *oh = out + h * head_dim;
    for (int64_t d = 0; d < head_dim; ++d) oh[d] = 0.0;
    for (int64_t t = 0; t < len; ++t) {
      const uint16_t *vrow = vh + t * v_row_stride;
      const double w = scores[static_cast<size_t>(t)] / sum;
      for (int64_t d = 0; d < head_dim; ++d) oh[d] += w * static_cast<double>(bf16_to_f32(vrow[d]));
    }
  }
}

void test_attend_one_xkv_bf16() {
  std::printf("-- attend_one_xkv_bf16 vs an independent double-precision reference --\n");
  const int64_t HEADS = 3, HD = 16, LEN = 37;
  std::vector<uint16_t> K(static_cast<size_t>(HEADS * LEN * HD)), V(K.size());
  for (auto &b : K) b = f32_to_bf16_rne(std::uniform_real_distribution<float>(-1.f, 1.f)(rng));
  for (auto &b : V) b = f32_to_bf16_rne(std::uniform_real_distribution<float>(-1.f, 1.f)(rng));
  std::vector<float> q = random_vec(HEADS * HD, -1.0f, 1.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

  const int64_t k_row_stride = HD, k_head_stride = LEN * HD;   // matches decoder.cpp's gathered layout
  std::vector<float> scores_scratch(static_cast<size_t>(HEADS * LEN), 0.f);
  std::vector<float> out(static_cast<size_t>(HEADS * HD));
  ow::attend_one_xkv_bf16(q.data(), K.data(), k_row_stride, k_head_stride, V.data(), k_row_stride,
                          k_head_stride, LEN, HEADS, HD, scale, out.data(), scores_scratch.data(), LEN);

  std::vector<double> ref(static_cast<size_t>(HEADS * HD));
  attend_ref_f64(q.data(), K.data(), k_row_stride, k_head_stride, V.data(), k_row_stride, k_head_stride,
                LEN, HEADS, HD, scale, ref.data());

  // out[d] is a convex combination (softmax weights, non-negative, sum to 1)
  // of V[:,d] in [-1,1] -- bounded L1 scale LEN*1, a safe worst-case (not
  // the tighter per-dimension weighted sum, but simple and still far above
  // the actual fp32 rounding error being measured).
  double worst_ratio = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    double ratio = 0.0;
    close_by_l1(out[i], ref[i], static_cast<double>(LEN), &ratio);
    worst_ratio = std::max(worst_ratio, ratio);
  }
  check_le(worst_ratio, 1.0, "attend_one_xkv_bf16() matches an independent double-precision reference (L1 bound)");
}

// ---------------------------------------------------------------------
// 5: int8x's top-K exact recompute.
// ---------------------------------------------------------------------

void test_recompute_top_k_exact() {
  std::printf("-- int8x: top-K exact recompute --\n");
  const int64_t OUT = 400, IN = 96, K = 20;
  std::vector<float> w = random_vec(OUT * IN, -0.3f, 0.3f);
  std::vector<uint16_t> w_bf16(static_cast<size_t>(OUT * IN));
  for (size_t i = 0; i < w_bf16.size(); ++i) w_bf16[i] = f32_to_bf16_rne(w[i]);
  std::vector<float> bias(static_cast<size_t>(OUT), 0.f);   // the tied head has no bias
  const ow::QLinear Q = ow::quantize_int8_rows_f32(w.data(), bias, OUT, IN);

  std::vector<float> x = random_vec(IN, -1.0f, 1.0f);

  // The TRUE (bf16-exact) logits and their argmax, computed independently.
  std::vector<double> true_logit(static_cast<size_t>(OUT));
  int64_t true_argmax = 0;
  double true_max = -std::numeric_limits<double>::infinity();
  for (int64_t o = 0; o < OUT; ++o) {
    true_logit[static_cast<size_t>(o)] =
        ow::dot_bf16_ref_f64(x.data(), w_bf16.data() + static_cast<size_t>(o) * static_cast<size_t>(IN), IN);
    if (true_logit[static_cast<size_t>(o)] > true_max) {
      true_max = true_logit[static_cast<size_t>(o)];
      true_argmax = o;
    }
  }

  // The int8 pass's own logits and its (possibly different) argmax --
  // demonstrating there IS something for int8x to fix, not a vacuous test.
  std::vector<float> logits(static_cast<size_t>(OUT));
  ow::linear_int8(x.data(), Q, logits.data());
  int64_t int8_argmax = 0;
  float int8_max = -std::numeric_limits<float>::infinity();
  for (int64_t o = 0; o < OUT; ++o)
    if (logits[static_cast<size_t>(o)] > int8_max) {
      int8_max = logits[static_cast<size_t>(o)];
      int8_argmax = o;
    }
  std::printf("  %-70s %s\n", "(context) true argmax vs int8-only argmax",
             (true_argmax == int8_argmax) ? "coincide (rerun with a different seed to see a miss)"
                                          : "DIFFER -- int8x has something to fix");

  ow::recompute_top_k_exact(x.data(), OUT, IN, K, w_bf16.data(), nullptr, logits.data());

  // Every recomputed row (rank <= K by the int8 pass) must equal the true
  // bf16-exact logit to tight tolerance, not just "be close".
  std::vector<int64_t> idx(static_cast<size_t>(OUT));
  for (int64_t o = 0; o < OUT; ++o) idx[static_cast<size_t>(o)] = o;
  // Recompute the int8 ranking independently (linear_int8 again, since
  // `logits` has since been overwritten) to know which K indices SHOULD
  // have been touched.
  std::vector<float> logits_int8_only(static_cast<size_t>(OUT));
  ow::linear_int8(x.data(), Q, logits_int8_only.data());
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int64_t a, int64_t b) {
    return logits_int8_only[static_cast<size_t>(a)] > logits_int8_only[static_cast<size_t>(b)];
  });
  double worst_ratio_topk = 0.0;
  for (int64_t j = 0; j < K; ++j) {
    const int64_t o = idx[static_cast<size_t>(j)];
    double l1 = 0.0;
    for (int64_t i = 0; i < IN; ++i)
      l1 += std::fabs(static_cast<double>(x[static_cast<size_t>(i)])) *
           std::fabs(static_cast<double>(bf16_to_f32(w_bf16[static_cast<size_t>(o) * static_cast<size_t>(IN) + static_cast<size_t>(i)])));
    double ratio = 0.0;
    close_by_l1(logits[static_cast<size_t>(o)], true_logit[static_cast<size_t>(o)], l1, &ratio);
    worst_ratio_topk = std::max(worst_ratio_topk, ratio);
  }
  check_le(worst_ratio_topk, 1.0, "every recomputed top-K logit equals the true bf16-exact value (L1 bound)");

  // And the resulting argmax over the corrected array is the TRUE argmax,
  // which is the property int8x actually exists for.
  int64_t corrected_argmax = 0;
  float corrected_max = -std::numeric_limits<float>::infinity();
  for (int64_t o = 0; o < OUT; ++o)
    if (logits[static_cast<size_t>(o)] > corrected_max) {
      corrected_max = logits[static_cast<size_t>(o)];
      corrected_argmax = o;
    }
  check(corrected_argmax == true_argmax, "corrected argmax equals the true (bf16-exact) argmax");
}

// A CONSTRUCTED adversarial case (PR #111 review, finding 9), not a random draw: the
// header's original reasoning claimed the argmax stayed exact "whether or not any of
// the other out-K rows' int8 approximations are close" -- false whenever the true-max
// row's OWN approximation overestimates its true value, leaving a gap an excluded row's
// stale approximation can sit in. `logits[]` here is set directly (not via
// quantize_int8_rows_f32/linear_int8) so the gap is exact and reproducible rather than
// dependent on a PRNG seed landing on it.
//
//   row 0: input (pre-call) approx = 100.0, true (bf16-exact) dot = 1.0*1.0 = 1.0 --
//          massively OVERESTIMATED, but still ranks #1 by approx, so it IS the one row
//          K=1 recomputes.
//   row 1: input approx = 60.0 (excluded: 60 < 100, so NOT in the top-1) -- 60 sits
//          exactly in the gap (1.0, 100.0], so an un-recomputed row 1 outranks row 0's
//          corrected exact value of 1.0.
//   rows 2..4: true dot = 0.0, well below row 0's true 1.0, so row 0 is genuinely the
//          global argmax and this is not a vacuous "any answer is fine" setup.
//
// Before the fix (no cap on excluded rows): corrected argmax = row 1 (60.0 uncorrected
// beats row 0's exact 1.0) -- WRONG. After the fix: row 1 is capped to
// min(exact top-K) = 1.0, ties row 0's own 1.0, and the first-strict-`>` scan keeps row
// 0 (encountered first) -- matching the true argmax.
void test_recompute_top_k_exact_overestimate_gap() {
  std::printf("-- int8x: top-K exact recompute (constructed overestimate-gap case) --\n");
  const int64_t OUT = 5, IN = 1, K = 1;
  std::vector<float> x = {1.0f};
  std::vector<uint16_t> w_bf16(static_cast<size_t>(OUT));
  w_bf16[0] = f32_to_bf16_rne(1.0f);  // row 0's true dot = 1.0
  w_bf16[1] = f32_to_bf16_rne(0.0f);  // row 1's true dot is irrelevant (never recomputed)
  w_bf16[2] = f32_to_bf16_rne(0.0f);
  w_bf16[3] = f32_to_bf16_rne(0.0f);
  w_bf16[4] = f32_to_bf16_rne(0.0f);

  std::vector<float> logits = {100.0f, 60.0f, 5.0f, 5.0f, 5.0f};  // the "approximate" pass
  ow::recompute_top_k_exact(x.data(), OUT, IN, K, w_bf16.data(), nullptr, logits.data());

  int64_t argmax = 0;
  float max_v = -std::numeric_limits<float>::infinity();
  for (int64_t o = 0; o < OUT; ++o) {
    if (logits[static_cast<size_t>(o)] > max_v) {
      max_v = logits[static_cast<size_t>(o)];
      argmax = o;
    }
  }
  check(logits[0] == 1.0f, "row 0 (the only recomputed row) holds its exact value 1.0");
  check_le(logits[1], 1.0f, "row 1 (excluded, was 60.0) no longer exceeds the exact max -- capped");
  check(argmax == 0, "argmax is the recomputed row, not the uncapped excluded row 1 (was wrongly 1 before the fix)");
}

// ---------------------------------------------------------------------
// Real weights, offline: per-tensor int8 quantization error statistics.
// Runs only when OW_DEC_TEST_MODEL names a model directory holding
// model.open.safetensors; skipped (not a failure) otherwise, so this test
// passes on a machine without the 1.6 GB container.
// ---------------------------------------------------------------------

void report_real_model_stats() {
  std::printf("-- real decoder weights: int8 quantization error per tensor --\n");
  const char *env = std::getenv("OW_DEC_TEST_MODEL");
  if (!env || !*env) {
    std::printf("  (skipped -- set OW_DEC_TEST_MODEL to a model dir with model.open.safetensors)\n");
    return;
  }
  const std::string model_dir(env);
  const std::string safet = model_dir + "/model.open.safetensors";

  std::FILE *probe = std::fopen(safet.c_str(), "rb");
  if (!probe) {
    std::printf("  (skipped -- %s not found; set OW_DEC_TEST_MODEL to point at a q4nx model dir)\n",
               safet.c_str());
    return;
  }
  std::fclose(probe);

  open_qwen36::Q4nxFile f(safet);
  const int64_t D = 1280, FFN = 5120, V = 51866, L = 4;

  auto load_bf16 = [&](const std::string &name, int64_t out, int64_t in) {
    size_t nbytes = 0;
    const uint8_t *raw = f.raw(name, &nbytes);
    std::vector<uint16_t> w(static_cast<size_t>(out) * static_cast<size_t>(in));
    std::memcpy(w.data(), raw, w.size() * 2);
    return w;
  };
  auto report_tensor = [&](const std::string &name, int64_t out, int64_t in) {
    if (!f.has(name)) {
      std::printf("  %-40s MISSING\n", name.c_str());
      return;
    }
    const std::vector<uint16_t> w_bf16 = load_bf16(name, out, in);
    std::vector<float> w_f32(w_bf16.size());
    for (size_t i = 0; i < w_bf16.size(); ++i) w_f32[i] = bf16_to_f32(w_bf16[i]);
    std::vector<float> zero_bias(static_cast<size_t>(out), 0.f);
    report_int8_quant_error(name, w_f32.data(), out, in, zero_bias, 3);
  };

  report_tensor("decoder.embed_tokens.weight", V, D);
  for (int64_t l = 0; l < L; ++l) {
    const std::string p = "decoder.layers." + std::to_string(l) + ".";
    report_tensor(p + "self_attn.q_proj.weight", D, D);
    report_tensor(p + "self_attn.k_proj.weight", D, D);
    report_tensor(p + "self_attn.v_proj.weight", D, D);
    report_tensor(p + "self_attn.out_proj.weight", D, D);
    report_tensor(p + "encoder_attn.q_proj.weight", D, D);
    report_tensor(p + "encoder_attn.out_proj.weight", D, D);
    report_tensor(p + "fc1.weight", FFN, D);
    report_tensor(p + "fc2.weight", D, FFN);
  }
}

}  // namespace

int main() {
  std::printf("== open_whisper decoder_quant ==\n");
  std::printf("-- env parsing (strict) --\n");
  // Defaults changed 2026-09-23 (task 0180 Parts 11-15): the WER gate (1200
  // utterances) found bf16 xkv / int8 weights / int8x head statistically
  // indistinguishable from the exact fp32/bf16/bf16 path under the hf
  // protocol, so those are now what "unset" resolves to.
  check([] { try { (void)ow::to_string(ow::parse_xkv_precision()); return true; } catch (...) { return false; } }(),
        "OW_DEC_XKV unset -> no throw, default bf16");
  check(std::string(ow::to_string(ow::parse_xkv_precision())) == "bf16", "OW_DEC_XKV unset -> \"bf16\"");
  check(std::string(ow::to_string(ow::parse_weight_precision())) == "int8", "OW_DEC_W unset -> \"int8\"");
  check(std::string(ow::to_string(ow::parse_head_precision())) == "int8x", "OW_DEC_HEAD unset -> \"int8x\"");
  // "fp32"/"bf16" still restore the exact path exactly -- only the unset
  // default moved, strict parsing and the explicit values did not.
#ifdef _WIN32
  _putenv_s("OW_DEC_XKV", "fp32");
  _putenv_s("OW_DEC_W", "bf16");
  _putenv_s("OW_DEC_HEAD", "bf16");
#else
  setenv("OW_DEC_XKV", "fp32", 1);
  setenv("OW_DEC_W", "bf16", 1);
  setenv("OW_DEC_HEAD", "bf16", 1);
#endif
  check(std::string(ow::to_string(ow::parse_xkv_precision())) == "fp32",
        "OW_DEC_XKV=fp32 still restores the exact path");
  check(std::string(ow::to_string(ow::parse_weight_precision())) == "bf16",
        "OW_DEC_W=bf16 still restores the exact path");
  check(std::string(ow::to_string(ow::parse_head_precision())) == "bf16",
        "OW_DEC_HEAD=bf16 still restores the exact path");
#ifdef _WIN32
  _putenv_s("OW_DEC_XKV", "");
  _putenv_s("OW_DEC_W", "");
  _putenv_s("OW_DEC_HEAD", "");
#else
  unsetenv("OW_DEC_XKV");
  unsetenv("OW_DEC_W");
  unsetenv("OW_DEC_HEAD");
#endif
#ifdef _WIN32
  _putenv_s("OW_DEC_XKV", "sideways");
#else
  setenv("OW_DEC_XKV", "sideways", 1);
#endif
  {
    bool threw = false;
    std::string msg;
    try {
      (void)ow::parse_xkv_precision();
    } catch (const std::exception &e) {
      threw = true;
      msg = e.what();
    }
    check(threw && msg.find("sideways") != std::string::npos,
          "OW_DEC_XKV='sideways' throws and names the bad value (strict parse, not fail-open)");
  }
#ifdef _WIN32
  _putenv_s("OW_DEC_XKV", "");
#else
  unsetenv("OW_DEC_XKV");
#endif

  test_int8_kernel_vs_quantized_reference();
  test_int8_adversarial();
  test_quant_error_random();
  test_bf16_kernels();
  test_attend_one_xkv_bf16();
  test_recompute_top_k_exact();
  test_recompute_top_k_exact_overestimate_gap();
  report_real_model_stats();

  std::printf("%s\n", failures ? "FAILED" : "all decoder_quant checks hold");
  return failures ? 1 : 0;
}
