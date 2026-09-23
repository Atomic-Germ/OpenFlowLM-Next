// Execute the production indexing loop with standard host scalar math.
// This verifies addressing, not the accuracy of AIE vecmath's approximations.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
static float sexp(float x) { return std::exp(x); }
static float ssoftplus(float x) { return std::log1p(std::exp(x)); }
static float ssigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
#include "glue_small_bank.h"

int main() {
  float small[96], a[32], b[32], decay[50], beta[50];
  for (unsigned h = 0; h < 48; ++h) {
    small[h] = -0.01f * (h + 1);
    small[48 + h] = 0.013f * (h + 1);
  }
  for (unsigned h = 0; h < 50; ++h) decay[h] = beta[h] = -123.f;
  for (unsigned bank = 0; bank < 2; ++bank) {
    unsigned base = bank * 32, active = bank ? 16 : 32;
    for (unsigned h = 0; h < 32; ++h) {
      a[h] = h < active ? 0.021f * (base + h) : std::numeric_limits<float>::quiet_NaN();
      b[h] = h < active ? -0.017f * (base + h) : std::numeric_limits<float>::quiet_NaN();
    }
    glue_small_bank<48>(small, a, b, decay + 1, beta + 1, base, active);
    for (unsigned h = 0; h < 48; ++h) {
      if (h >= base + active) {
        if (decay[h + 1] != -123.f || beta[h + 1] != -123.f) return 1;
        continue;
      }
      float expected = std::exp(small[h] * std::log1p(std::exp(0.021f * h + small[48 + h])));
      float eb = 1.f / (1.f + std::exp(0.017f * h));
      if (std::abs(decay[h + 1] - expected) > 1e-7f || std::abs(beta[h + 1] - eb) > 1e-7f) return 2;
    }
  }
  if (decay[0] != -123.f || decay[49] != -123.f || beta[0] != -123.f || beta[49] != -123.f) return 3;
  std::puts("PASS: all 48 heads, bank boundary 31/32, tail 47, A/dt_bias and output canaries");
}
