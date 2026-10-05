// Host check of open_kernels/include/scalar_fp.h against the host FPU (IEEE fp32,
// round-to-nearest-even). Build with any C++ compiler that has __builtin_memcpy:
//   clang++ -O2 -std=c++17 -I open_kernels/include utilities/scalar_fp_test.cpp -o scalar_fp_test
// Prints "PASS <n> products, <m> reciprocals" and exits 0, or the first mismatch and exits 1.
// Traces: OPEN-ATTN-CONTEXT (canonical spec: specs/open-engine/spec.md)
#include "scalar_fp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static uint64_t g_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() {
  g_state ^= g_state << 13;
  g_state ^= g_state >> 7;
  g_state ^= g_state << 17;
  return (uint32_t)(g_state >> 16);
}
// a random normal float with exponent field in [lo, hi]
static float rnd_normal(uint32_t lo, uint32_t hi) {
  const uint32_t e = lo + rnd() % (hi - lo + 1);
  return sfp_float((rnd() & 0x80000000u) | (e << 23) | (rnd() & 0x7FFFFFu));
}

static bool product_is_normal(float a, float b) {
  const double p = std::fabs((double)a * (double)b);
  return p == 0.0 || (p >= 1.1754943508222875e-38 && p < 3.4028234663852886e38);
}

int main(int argc, char **argv) {
  const long n = argc > 1 ? std::atol(argv[1]) : 20000000L;
  long np = 0, nr = 0;
  for (long i = 0; i < n; ++i) {
    // products: the whole exponent range, then a band near 1 (the Newton terms) and
    // around 2^-8 (the RMS scale), where the operands at the call sites live
    const int band = i % 3;
    float a = band == 0 ? rnd_normal(1, 254) : rnd_normal(100, 150);
    float b = band == 0 ? rnd_normal(1, 254) : band == 1 ? rnd_normal(120, 134) : sfp_float(0x3B800000u);
    if (i % 1000 == 7) a = 0.0f;
    if (!product_is_normal(a, b)) continue;
    volatile float va = a, vb = b;
    const float want = va * vb, got = smul_rn(a, b);
    if (sfp_bits(want) != sfp_bits(got)) {
      std::printf("FAIL smul_rn(%a, %a) = %a, want %a\n", a, b, got, want);
      return 1;
    }
    ++np;
    // reciprocals: every exponent whose reciprocal is normal
    const float d = rnd_normal(3, 252);
    volatile float vd = d;
    const float rw = 1.0f / vd, rg = srecip_rn(d);
    if (sfp_bits(rw) != sfp_bits(rg)) {
      std::printf("FAIL srecip_rn(%a) = %a, want %a\n", d, rg, rw);
      return 1;
    }
    ++nr;
  }
  // every mantissa of the reciprocal at one exponent: the division's whole domain
  for (uint32_t m = 0; m < 0x800000u; ++m) {
    const float d = sfp_float((127u << 23) | m);
    volatile float vd = d;
    if (sfp_bits(1.0f / vd) != sfp_bits(srecip_rn(d))) {
      std::printf("FAIL srecip_rn(%a)\n", d);
      return 1;
    }
    ++nr;
  }
  std::printf("PASS %ld products, %ld reciprocals\n", np, nr);
  return 0;
}
