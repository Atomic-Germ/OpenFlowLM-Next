#pragma once
#include <stdint.h>
// Private IEEE binary32 addition in integer arithmetic. No scalar soft-float
// runtime or AIE accumulator rounding. Three guard/round/sticky bits retain
// cancellation and ties-to-even; gradual underflow and signed zero are kept.
static inline uint32_t shift_jam(uint32_t x, unsigned n) {
  if (!n) return x;
  if (n >= 32) return x != 0;
  return (x >> n) | ((x << (32 - n)) != 0);
}
static inline uint32_t add_rne(uint32_t a, uint32_t b) {
  const uint32_t sign_mask = 0x80000000u;
  uint32_t aa = a & ~sign_mask, bb = b & ~sign_mask;
  if (aa > 0x7f800000u) return a | 0x00400000u;
  if (bb > 0x7f800000u) return b | 0x00400000u;
  if (aa == 0x7f800000u || bb == 0x7f800000u) {
    if (aa == bb && ((a ^ b) & sign_mask)) return 0x7fc00000u;
    return aa == 0x7f800000u ? a : b;
  }
  if (!(aa | bb)) return (a & b) & sign_mask;
  if (aa < bb) { uint32_t t = a; a = b; b = t; t = aa; aa = bb; bb = t; }
  if (!bb) return a;
  const uint32_t sign = a & sign_mask;
  unsigned ea = aa >> 23, eb = bb >> 23;
  uint32_t ma = aa & 0x7fffffu, mb = bb & 0x7fffffu;
  if (ea) ma |= 0x800000u; else ea = 1;
  if (eb) mb |= 0x800000u; else eb = 1;
  ma <<= 3; mb = shift_jam(mb << 3, ea - eb);
  uint32_t m;
  if ((a ^ b) & sign_mask) {
    m = ma - mb;
    if (!m) return 0;
    while (m < 0x4000000u && ea > 1) { m <<= 1; --ea; }
  } else {
    m = ma + mb;
    if (m & 0x8000000u) { m = shift_jam(m, 1); ++ea; }
  }
  const uint32_t rem = m & 7u;
  m >>= 3;
  if (rem > 4u || (rem == 4u && (m & 1u))) ++m;
  if (m >= 0x1000000u) { m >>= 1; ++ea; }
  if (ea >= 255) return sign | 0x7f800000u;
  if (m < 0x800000u) ea = 0;
  return sign | (ea << 23) | (m & 0x7fffffu);
}
