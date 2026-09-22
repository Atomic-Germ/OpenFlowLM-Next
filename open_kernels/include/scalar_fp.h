#pragma once
//===- scalar_fp.h -----------------------------------------*- C++ -*-===//
//
// Correctly-rounded scalar fp32 multiply and reciprocal in integer arithmetic, for a
// core whose program memory cannot carry compiler-rt's soft-float.
//
// The AIE2P scalar unit has no float multiplier or divider, so `a * b` and `1.0f / d` on
// scalars become calls to __mulsf3 / __divsf3 (+ __muldi3 under both): 2,112 bytes on
// the 35B's attention core for the handful of scalar products norm_rope's RMS needs
// and the one reciprocal attn_fin_impl takes per head. Both routines here give the
// IEEE round-to-nearest-even result BIT FOR BIT for the operands they accept -- normal,
// finite, a normal result (and a zero multiplicand) -- which is every operand those two
// call sites can produce (a sum of squares times 2^-8 plus eps; Newton terms near 1;
// the softmax denominator l >= 1). What they drop from compiler-rt is exactly the
// denormal / infinity / NaN handling, so swapping them in changes no result.
//
// No aie_api here: the header is plain C++ so the host test
// (utilities/scalar_fp_test.cpp, specs/open-engine/tests/test_scalar_fp.py) compiles
// the same code and checks it against the host FPU.

#include <stdint.h>

static inline uint32_t sfp_bits(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  return u;
}
static inline float sfp_float(uint32_t u) {
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// a * b, round-to-nearest-even. a, b normal or zero; the product normal (no overflow,
// no underflow). The 24 x 24-bit mantissa product is built from 16-bit halves in two
// 32-bit words -- the scalar unit's multiply is 32 x 32 -> 32.
__attribute__((noinline)) inline float smul_rn(float a, float b) {
  const uint32_t ua = sfp_bits(a), ub = sfp_bits(b);
  const uint32_t s = (ua ^ ub) & 0x80000000u;
  const uint32_t ea = (ua >> 23) & 0xFFu, eb = (ub >> 23) & 0xFFu;
  if (ea == 0 || eb == 0) return sfp_float(s);                // a zero operand
  const uint32_t ma = (ua & 0x7FFFFFu) | 0x800000u, mb = (ub & 0x7FFFFFu) | 0x800000u;
  const uint32_t ah = ma >> 16, al = ma & 0xFFFFu, bh = mb >> 16, bl = mb & 0xFFFFu;
  const uint32_t lo = al * bl;                                // < 2^32
  const uint32_t mid = ah * bl + al * bh;                     // < 2^25
  const uint32_t lo2 = lo + (mid << 16);
  const uint32_t hi = ah * bh + (mid >> 16) + (lo2 < lo ? 1u : 0u);   // P = hi:lo2 in [2^46, 2^48)
  uint32_t e = ea + eb - 127u, m, rest, half;
  if (hi & 0x8000u) {                                         // P >= 2^47: 24 bits below the mantissa
    m = (hi << 8) | (lo2 >> 24);
    rest = lo2 & 0xFFFFFFu;
    half = 0x800000u;
    e += 1;
  } else {                                                    // 23 bits below it
    m = (hi << 9) | (lo2 >> 23);
    rest = lo2 & 0x7FFFFFu;
    half = 0x400000u;
  }
  if (rest > half || (rest == half && (m & 1u))) {
    m += 1;
    if (m == 0x1000000u) { m >>= 1; e += 1; }
  }
  return sfp_float(s | (e << 23) | (m & 0x7FFFFFu));
}

// 1 / d, round-to-nearest-even. d normal, 1/d normal. Restoring division of 2^48 by the
// 24-bit mantissa: 25 quotient bits (24 of mantissa and the round bit), the remainder
// is the sticky bit. A mantissa of exactly 2^23 (d a power of two) is an exact result.
__attribute__((noinline)) inline float srecip_rn(float d) {
  const uint32_t u = sfp_bits(d);
  const uint32_t s = u & 0x80000000u;
  const uint32_t e = (u >> 23) & 0xFFu;
  if ((u & 0x7FFFFFu) == 0) return sfp_float(s | ((254u - e) << 23));
  const uint32_t m = (u & 0x7FFFFFu) | 0x800000u;           // (2^23, 2^24)
  uint32_t rem = 0x1000000u, q = 0;                          // 2^24: m <= rem < 2m
#pragma clang loop unroll(disable)                           // a loop: program memory is the point
  for (unsigned i = 0; i < 25; ++i) {
    const uint32_t bit = rem >= m ? 1u : 0u;
    if (bit) rem -= m;
    q = (q << 1) | bit;
    rem <<= 1;
  }                                                          // q = floor(2^48 / m) in (2^24, 2^25)
  uint32_t mant = q >> 1;
  if ((q & 1u) && (rem != 0 || (mant & 1u))) mant += 1;      // never carries: q <= 2^25 - 4
  return sfp_float(s | ((253u - e) << 23) | (mant & 0x7FFFFFu));
}
