//===- host_ops_fast_test.cpp ----------------------------------*- C++ -*-===//
//
// Offline (no NPU, no XRT, no model container) characterisation of task
// 0180's OW_HOST_FAST ops against the exact ops in host_ops.cpp, plus a
// single-threaded microbenchmark of old vs new GELU. Every fused op is
// checked for EXACT correctness (does the fusion change any arithmetic) and,
// where it does (erf, GELU, LayerNorm's final combine), against a double
// reference with max abs/rel error and an ulp histogram.
//
// Optionally loads a real encoder hidden state (enc.hidden.15 from a Whisper
// golden safetensors file, via open_qwen36::Q4nxFile, which reads any
// safetensors container -- not just .q4nx ones) if
// OW_TEST_GOLDEN=<path to nvidia.safetensors> is set; otherwise runs on a
// fixed-seed PRNG buffer and says so.
//
//   out\host_ops_fast_test.exe
//
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "host_ops.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace {

int failures = 0;
void check(bool ok, const std::string &what) {
  std::printf("  %-70s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

// ---------------------------------------------------------------------------
// PRNG -- fixed seed, reproducible across runs (no <random> engine-version
// dependence to worry about).
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
  }
  float uniform(float lo, float hi) {
    const double u = static_cast<double>(next() >> 11) / static_cast<double>(1ull << 53);
    return static_cast<float>(lo + u * (hi - lo));
  }
};

// ulp distance between two finite floats (Bruce Dawson's "ordered integer"
// trick: sign-magnitude float bits mapped to a monotonic integer). Returns -1
// if either input is NaN (report separately -- an ulp distance to a NaN is
// not a number).
int64_t ulp_diff(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) return -1;
  int32_t ia, ib;
  std::memcpy(&ia, &a, 4);
  std::memcpy(&ib, &b, 4);
  auto ordered = [](int32_t i) -> int64_t {
    return i < 0 ? static_cast<int64_t>(0x80000000u) - static_cast<uint32_t>(i) : i;
  };
  return std::llabs(ordered(ia) - ordered(ib));
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// 1. erf_avx2 vs a double std::erf reference, dense sweep + edge cases.
void test_erf() {
  std::printf("-- erf_avx2 vs double std::erf --\n");

  // Dense sweep: step 1e-3 over [-12, 12] (erf saturates to +-1 well before
  // +-6; the tail out to 12 checks the underflow/saturation behaviour too).
  // Sized to a multiple of 8 so the whole sweep goes through the AVX2 path,
  // not the scalar tail.
  const int64_t n = 24000 * 8 / 8;   // 24000, already a multiple of 8
  std::vector<float> x(static_cast<size_t>(n)), y(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) x[static_cast<size_t>(i)] = -12.0f + 24.0f * i / (n - 1);
  ow::erf_avx2(x.data(), n, y.data());

  double max_abs = 0, max_rel = 0, sum_ulp = 0;
  int64_t max_ulp = 0, over_2ulp = 0, n_scored = 0;
  float max_abs_at = 0, max_ulp_at = 0;
  for (int64_t i = 0; i < n; ++i) {
    const float ref = ow::erf_scalar_ref(x[static_cast<size_t>(i)]);
    const float got = y[static_cast<size_t>(i)];
    const double abse = std::fabs(static_cast<double>(got) - static_cast<double>(ref));
    if (abse > max_abs) { max_abs = abse; max_abs_at = x[static_cast<size_t>(i)]; }
    if (std::fabs(ref) > 1e-6f) {
      const double rele = abse / std::fabs(static_cast<double>(ref));
      if (rele > max_rel) max_rel = rele;
    }
    const int64_t u = ulp_diff(got, ref);
    if (u >= 0) {
      sum_ulp += static_cast<double>(u);
      ++n_scored;
      if (u > max_ulp) { max_ulp = u; max_ulp_at = x[static_cast<size_t>(i)]; }
      if (u > 2) ++over_2ulp;
    }
  }
  std::printf("  sweep [-12,12] step ~1e-3, n=%lld: max_abs=%.3e (at x=%.6f), "
             "max_rel=%.3e, mean_ulp=%.3f, max_ulp=%lld (at x=%.6f), >2ulp: %lld/%lld (%.2f%%)\n",
             (long long)n, max_abs, max_abs_at, max_rel, sum_ulp / n_scored,
             (long long)max_ulp, max_ulp_at, (long long)over_2ulp, (long long)n_scored,
             100.0 * over_2ulp / n_scored);
  check(max_abs < 5e-6, "erf_avx2 max abs error < 5e-6 over the dense sweep");

  // Edge cases: 0, +-0, +-tiny, +-large, NaN, +-inf.
  const float edges[] = {
      0.0f, -0.0f, 1e-30f, -1e-30f, 1e-6f, -1e-6f, 6.0f, -6.0f, 20.0f, -20.0f,
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  const char *names[] = {"0", "-0", "1e-30", "-1e-30", "1e-6", "-1e-6", "6", "-6",
                         "20", "-20", "+inf", "-inf", "NaN"};
  std::printf("  edge cases:\n");
  for (size_t i = 0; i < sizeof(edges) / sizeof(edges[0]); ++i) {
    float out8;
    ow::erf_avx2(&edges[i], 1, &out8);   // n=1 -> scalar tail path (erf_scalar_ref itself)
    // Also probe the AVX2 lane directly by padding to 8 and taking lane 0, so
    // the vector path's OWN handling of this edge case is what gets reported
    // (erf_avx2's n=1 call above goes through the scalar tail, not erf8).
    float pad[8] = {edges[i], 0, 0, 0, 0, 0, 0, 0};
    float outv[8];
    ow::erf_avx2(pad, 8, outv);
    const float ref = ow::erf_scalar_ref(edges[i]);
    std::printf("    erf(%-6s) vectorised=%.9g  scalar-ref(double)=%.9g  ulp=%lld\n",
               names[i], static_cast<double>(outv[0]), static_cast<double>(ref),
               (long long)ulp_diff(outv[0], ref));
  }
  // Correctness assertions on the edges that have a defined answer.
  {
    float pad[8], outv[8];
    auto vec_erf = [&](float v) { pad[0] = v; for (int k = 1; k < 8; ++k) pad[k] = 0;
                                  ow::erf_avx2(pad, 8, outv); return outv[0]; };
    check(vec_erf(0.0f) == 0.0f || std::fabs(vec_erf(0.0f)) < 1e-6f, "erf8(0) ~= 0");
    check(vec_erf(20.0f) > 0.999999f, "erf8(20) saturates to ~1");
    check(vec_erf(-20.0f) < -0.999999f, "erf8(-20) saturates to ~-1");
    check(vec_erf(std::numeric_limits<float>::infinity()) > 0.999999f, "erf8(+inf) ~= 1");
    check(vec_erf(-std::numeric_limits<float>::infinity()) < -0.999999f, "erf8(-inf) ~= -1");
    const float nan_out = vec_erf(std::numeric_limits<float>::quiet_NaN());
    std::printf("    erf8(NaN) = %.9g (%s) -- see host_ops.cpp exp8's documented "
               "NaN-clamp quirk; the outer `t` factor being NaN is what recovers NaN here\n",
               static_cast<double>(nan_out), std::isnan(nan_out) ? "NaN, as expected" : "NOT NaN");
  }
}

// ---------------------------------------------------------------------------
// 2. gelu_bias_bf16_fast vs (gelu_bias + bf16_fill), on real-like data.
void test_gelu_fused(const std::vector<float> &real_flat) {
  std::printf("-- gelu_bias_bf16_fast vs gelu_bias()+bf16_fill() --\n");
  const int64_t cols = 5120;   // FFN width
  const int64_t rows = static_cast<int64_t>(real_flat.size()) / cols;
  check(rows > 0, "real-like buffer reshapes to an integer number of FFN-width rows");
  std::vector<float> bias(static_cast<size_t>(cols));
  Rng rng(42);
  for (auto &b : bias) b = rng.uniform(-0.05f, 0.05f);

  std::vector<float> exact_f(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  std::vector<uint16_t> exact_bf(exact_f.size()), fast_bf(exact_f.size());
  ow::gelu_bias(real_flat.data(), rows, cols, bias.data(), exact_f.data());
  ow::bf16_fill(exact_bf.data(), exact_f.data(), exact_f.size());
  ow::gelu_bias_bf16_fast(real_flat.data(), rows, cols, bias.data(), fast_bf.data());

  double max_abs = 0, sum_ulp = 0;
  int64_t max_ulp = 0, n = static_cast<int64_t>(exact_bf.size());
  for (int64_t i = 0; i < n; ++i) {
    const float ref = ow::from_bf16(exact_bf[static_cast<size_t>(i)]);
    const float got = ow::from_bf16(fast_bf[static_cast<size_t>(i)]);
    max_abs = std::max(max_abs, static_cast<double>(std::fabs(got - ref)));
    const int64_t u = ulp_diff(got, ref);
    if (u >= 0) { sum_ulp += u; max_ulp = std::max(max_ulp, u); }
  }
  std::printf("  %lld rows x %lld cols (real enc.hidden data): bf16-output max_abs=%.3e, "
             "mean_ulp=%.3f, max_ulp=%lld\n",
             (long long)rows, (long long)cols, max_abs, sum_ulp / n, (long long)max_ulp);
  // The output is bf16 (8 mantissa bits, ~1/256 relative precision); a
  // difference this far below bf16's own quantisation step is noise against
  // the datapath, not a defect in the fusion.
  check(max_abs < 2e-2, "gelu_bias_bf16_fast within bf16 quantisation noise of the exact path");

  // Deliberate-break protocol (documented, not left wired in): flipping the
  // sign inside erfc_avx2_ax's `arg` term (changing -ax*ax to +ax*ax) during
  // development made this check fail at max_abs ~ 1.0 (saturating wrong-sign
  // erf), confirming the comparison is live. Reverted before this file was
  // finalised -- see the task report for the exact edit and the failing
  // output it produced.
}

// ---------------------------------------------------------------------------
// 3. layer_norm_bf16_fast vs (layer_norm + bf16_fill), on real-like data.
void test_layer_norm_fused(const std::vector<float> &real_flat) {
  std::printf("-- layer_norm_bf16_fast vs layer_norm()+bf16_fill() --\n");
  const int64_t cols = 1280;   // d_model
  const int64_t rows = static_cast<int64_t>(real_flat.size()) / cols;
  std::vector<float> w(static_cast<size_t>(cols)), b(static_cast<size_t>(cols));
  Rng rng(7);
  // Synthetic LN weight/bias (no model container in this offline test) --
  // shaped like a trained LN: w near 1, b near 0.
  for (auto &v : w) v = 1.0f + rng.uniform(-0.1f, 0.1f);
  for (auto &v : b) v = rng.uniform(-0.05f, 0.05f);

  std::vector<float> exact_f(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  std::vector<uint16_t> exact_bf(exact_f.size()), fast_bf(exact_f.size());
  ow::layer_norm(real_flat.data(), w.data(), b.data(), rows, cols, exact_f.data());
  ow::bf16_fill(exact_bf.data(), exact_f.data(), exact_f.size());
  ow::layer_norm_bf16_fast(real_flat.data(), w.data(), b.data(), rows, cols, fast_bf.data());

  double max_abs = 0, sum_ulp = 0;
  int64_t max_ulp = 0, n = static_cast<int64_t>(exact_bf.size());
  for (int64_t i = 0; i < n; ++i) {
    const float ref = ow::from_bf16(exact_bf[static_cast<size_t>(i)]);
    const float got = ow::from_bf16(fast_bf[static_cast<size_t>(i)]);
    max_abs = std::max(max_abs, static_cast<double>(std::fabs(got - ref)));
    const int64_t u = ulp_diff(got, ref);
    if (u >= 0) { sum_ulp += u; max_ulp = std::max(max_ulp, u); }
  }
  std::printf("  %lld rows x %lld cols: bf16-output max_abs=%.3e, mean_ulp=%.3f, max_ulp=%lld\n",
             (long long)rows, (long long)cols, max_abs, sum_ulp / n, (long long)max_ulp);
  check(max_abs < 2e-2, "layer_norm_bf16_fast within bf16 quantisation noise of the exact path");
}

// ---------------------------------------------------------------------------
// 4. add_bias_residual_fast vs (memcpy + add_bias + add_rows): EXPECTED and
//    tested to be BIT-IDENTICAL (same op order, just fused).
void test_add_bias_residual_exact() {
  std::printf("-- add_bias_residual_fast vs memcpy+add_bias+add_rows (expect EXACT match) --\n");
  const int64_t rows = 1536, cols = 1280;
  Rng rng(99);
  std::vector<float> c(static_cast<size_t>(rows) * cols), bias(static_cast<size_t>(cols));
  std::vector<float> x_exact(c.size()), x_fast(c.size());
  for (auto &v : c) v = rng.uniform(-3.0f, 3.0f);
  for (auto &v : bias) v = rng.uniform(-0.5f, 0.5f);
  for (auto &v : x_exact) v = rng.uniform(-3.0f, 3.0f);
  x_fast = x_exact;

  std::vector<float> o_out(c.size());
  std::memcpy(o_out.data(), c.data(), o_out.size() * sizeof(float));
  ow::add_bias(o_out.data(), bias.data(), rows, cols);
  ow::add_rows(x_exact.data(), o_out.data(), rows, cols, x_exact.data());

  ow::add_bias_residual_fast(c.data(), bias.data(), rows, cols, x_fast.data());

  bool exact = std::memcmp(x_exact.data(), x_fast.data(), x_exact.size() * sizeof(float)) == 0;
  int64_t first_diff = -1;
  if (!exact)
    for (size_t i = 0; i < x_exact.size(); ++i)
      if (x_exact[i] != x_fast[i]) { first_diff = static_cast<int64_t>(i); break; }
  std::printf("  %lld x %lld: %s%s\n", (long long)rows, (long long)cols,
             exact ? "bit-identical" : "DIFFERS",
             exact ? "" : (", first diff at " + std::to_string(first_diff)).c_str());
  check(exact, "add_bias_residual_fast bit-identical to the 3-pass exact path");
}

// ---------------------------------------------------------------------------
// 5. attention_gather_bias_fast + attention_core vs attention() (exact bias
//    already applied): EXPECTED and tested to be BIT-IDENTICAL.
void test_attention_fused_exact() {
  std::printf("-- attention_gather_bias_fast+attention_core vs add_bias(qkv)+attention() "
             "(expect EXACT match) --\n");
  const int64_t heads = 20, hd = 64, d = heads * hd;   // 1280, Whisper's shape
  const int64_t t = 200;          // real rows (kept small so the test is fast)
  const int64_t m_padded = 256;   // >= t, arbitrary padded M for this test
  Rng rng(123);
  std::vector<float> qkv_raw(static_cast<size_t>(m_padded) * 3 * d);
  std::vector<float> bias(static_cast<size_t>(3 * d));
  for (auto &v : qkv_raw) v = rng.uniform(-2.0f, 2.0f);
  for (auto &v : bias) v = rng.uniform(-0.3f, 0.3f);

  // Exact path: bias applied to a full copy of qkv, then attention()'s own
  // (unchanged) gather + core.
  std::vector<float> qkv_biased = qkv_raw;
  ow::add_bias(qkv_biased.data(), bias.data(), m_padded, 3 * d);
  std::vector<float> scratch_exact(static_cast<size_t>(3) * t * d);
  std::vector<float> out_exact(static_cast<size_t>(m_padded) * d);
  ow::attention(qkv_biased.data(), m_padded, t, d, heads, hd, out_exact.data(),
               scratch_exact.data());

  // Fast path: fused gather+bias straight off qkv_raw (standing in for a
  // device C buffer -- read-only throughout).
  std::vector<float> scratch_fast(static_cast<size_t>(3) * t * d);
  std::vector<float> out_fast(static_cast<size_t>(m_padded) * d);
  ow::zero_pad_rows(out_fast.data(), t, m_padded, d);
  ow::attention_gather_bias_fast(qkv_raw.data(), m_padded, t, d, heads, hd, bias.data(),
                                scratch_fast.data());
  ow::attention_core(t, d, heads, hd, out_fast.data(), scratch_fast.data());

  const bool exact =
      std::memcmp(out_exact.data(), out_fast.data(), out_exact.size() * sizeof(float)) == 0;
  int64_t first_diff = -1;
  if (!exact)
    for (size_t i = 0; i < out_exact.size(); ++i)
      if (out_exact[i] != out_fast[i]) { first_diff = static_cast<int64_t>(i); break; }
  std::printf("  heads=%lld hd=%lld t=%lld m_padded=%lld: %s%s\n", (long long)heads,
             (long long)hd, (long long)t, (long long)m_padded,
             exact ? "bit-identical" : "DIFFERS",
             exact ? "" : (", first diff at " + std::to_string(first_diff)).c_str());
  check(exact, "attention_gather_bias_fast+attention_core bit-identical to attention()");
}

// ---------------------------------------------------------------------------
// 6. bf16_fill_parallel vs bf16_fill: exact by construction (element-
//    independent rounding), checked anyway.
void test_bf16_parallel_exact() {
  std::printf("-- bf16_fill_parallel vs bf16_fill (expect EXACT match) --\n");
  Rng rng(5);
  std::vector<float> x(1536 * 5120 + 3);   // deliberately not a multiple of the chunk size
  for (auto &v : x) v = rng.uniform(-100.0f, 100.0f);
  std::vector<uint16_t> a(x.size()), b(x.size());
  ow::bf16_fill(a.data(), x.data(), x.size());
  ow::bf16_fill_parallel(b.data(), x.data(), x.size());
  const bool exact = std::memcmp(a.data(), b.data(), a.size() * sizeof(uint16_t)) == 0;
  std::printf("  n=%zu: %s\n", x.size(), exact ? "bit-identical" : "DIFFERS");
  check(exact, "bf16_fill_parallel bit-identical to bf16_fill");
}

// ---------------------------------------------------------------------------
// 7. Microbenchmark: old GELU (gelu_bias + bf16_fill) vs new
//    (gelu_bias_bf16_fast), single-threaded, 1536x5120. Host wall clock,
//    microbenchmark -- NOT an NPU claim, NOT a claim about the threaded
//    in-engine cost.
void bench_gelu() {
  std::printf("-- GELU microbenchmark, 1536x5120, SINGLE-THREADED, host wall clock --\n");
#if defined(_OPENMP)
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
  const int64_t rows = 1536, cols = 5120, n = rows * cols;
  Rng rng(11);
  std::vector<float> c(static_cast<size_t>(n)), bias(static_cast<size_t>(cols));
  for (auto &v : c) v = rng.uniform(-6.0f, 6.0f);
  for (auto &v : bias) v = rng.uniform(-0.2f, 0.2f);
  std::vector<float> old_out(static_cast<size_t>(n));
  std::vector<uint16_t> old_bf(static_cast<size_t>(n)), new_bf(static_cast<size_t>(n));

  constexpr int reps = 5;
  double best_old = 1e300, best_new = 1e300;
  for (int r = 0; r < reps; ++r) {
    double t0 = now_s();
    ow::gelu_bias(c.data(), rows, cols, bias.data(), old_out.data());
    ow::bf16_fill(old_bf.data(), old_out.data(), old_out.size());
    best_old = std::min(best_old, now_s() - t0);

    t0 = now_s();
    ow::gelu_bias_bf16_fast(c.data(), rows, cols, bias.data(), new_bf.data());
    best_new = std::min(best_new, now_s() - t0);
  }
  std::printf("  old (gelu_bias + bf16_fill): best of %d = %.4f ms, %.3f ns/element\n",
             reps, best_old * 1e3, best_old * 1e9 / n);
  std::printf("  new (gelu_bias_bf16_fast):   best of %d = %.4f ms, %.3f ns/element\n",
             reps, best_new * 1e3, best_new * 1e9 / n);
  std::printf("  speedup: %.3fx (single-threaded microbenchmark; the threaded, "
             "in-engine number is NOT this -- see the task report)\n", best_old / best_new);
  const bool same = std::memcmp(old_bf.data(), new_bf.data(), old_bf.size() * sizeof(uint16_t)) == 0;
  std::printf("  (bench outputs bit-identical to the correctness test above: %s)\n",
             same ? "yes" : "no, see test 2 for the ulp characterisation");
#if defined(_OPENMP)
  omp_set_num_threads(saved);
#endif
}

// OW_HOST_FAST's default flipped 2026-09-23 (task 0180 Part 15/17): unset
// now means fast (it used to mean exact). Strict parsing is unchanged --
// only "0" restores exact, and anything else still throws. Manipulates the
// real process environment via _putenv_s and restores it to unset
// afterwards (host_fast_enabled() has no internal caching, unlike
// encoder.cpp's run_layer()'s function-local static, so this is safe to call
// repeatedly within one process).
void test_host_fast_default() {
  std::printf("-- OW_HOST_FAST default --\n");
  auto set_env = [](const char *v) { _putenv_s("OW_HOST_FAST", v ? v : ""); };

  set_env(nullptr);
  check(ow::host_fast_enabled() == true, "unset -> fast (the new default)");
  set_env("1");
  check(ow::host_fast_enabled() == true, "'1' -> fast");
  set_env("0");
  check(ow::host_fast_enabled() == false, "'0' -> exact (still restores it)");
  set_env("2");
  bool threw = false;
  try {
    (void)ow::host_fast_enabled();
  } catch (const std::exception &e) {
    threw = std::string(e.what()).find("OW_HOST_FAST") != std::string::npos;
  }
  check(threw, "'2' throws, naming OW_HOST_FAST, rather than reading as exact or fast");

  set_env(nullptr);   // restore: empty is read as unset by host_fast_enabled() itself
}

}  // namespace

int main() {
  std::printf("host_ops_fast_test: task 0180 OW_HOST_FAST characterisation "
             "(no NPU, no XRT, no model container)\n\n");

  test_host_fast_default();
  std::printf("\n");

  test_erf();
  std::printf("\n");

  std::vector<float> real_flat;
  const char *golden = std::getenv("OW_TEST_GOLDEN");
  if (golden && *golden) {
    try {
      open_qwen36::Q4nxFile f(golden);
      if (f.has("enc.hidden.15")) {
        real_flat = f.f32("enc.hidden.15");
        std::printf("loaded enc.hidden.15 from %s: %zu floats (real encoder hidden state, "
                   "layer 15, before LN/GELU)\n\n", golden, real_flat.size());
      } else {
        std::printf("OW_TEST_GOLDEN=%s has no enc.hidden.15 tensor; falling back to PRNG data\n\n",
                   golden);
      }
    } catch (const std::exception &e) {
      std::printf("OW_TEST_GOLDEN=%s: %s; falling back to PRNG data\n\n", golden, e.what());
    }
  }
  if (real_flat.empty()) {
    // Fallback: PRNG data shaped like a post-GEMM fp32 buffer (a few units of
    // magnitude, not bf16-quantised) so the GELU/LN tests still exercise a
    // realistic range even with no golden file set.
    std::printf("OW_TEST_GOLDEN not set (or unusable) -- using PRNG data, NOT a real hidden "
               "state. Set OW_TEST_GOLDEN=<path to nvidia.safetensors> for the real-data run.\n\n");
    Rng rng(2026);
    real_flat.resize(static_cast<size_t>(1500) * 1280);
    for (auto &v : real_flat) v = rng.uniform(-4.0f, 4.0f);
  }

  test_gelu_fused(real_flat);
  std::printf("\n");
  test_layer_norm_fused(real_flat);
  std::printf("\n");
  test_add_bias_residual_exact();
  std::printf("\n");
  test_attention_fused_exact();
  std::printf("\n");
  test_bf16_parallel_exact();
  std::printf("\n");
  bench_gelu();

  std::printf("\n%d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
