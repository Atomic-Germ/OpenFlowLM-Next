#pragma once
#include <stdint.h>

// IEEE binary32 addition using integer lanes, guard/round/sticky bits and
// round-to-nearest, ties-to-even. O supplies a lane type, broadcast and select;
// the same algorithm runs as AIE vectors and in the independent host tests.
// No floating-point adder, scalar neural fallback or ambient rounding mode.
template <class O>
static inline typename O::V fp32_add_rne(typename O::V a, typename O::V b) {
  const auto zero = O::v(0), one = O::v(1);
  const auto signbit = O::v(0x80000000u), absbits = O::v(0x7fffffffu);
  const auto frac = O::v(0x7fffffu), hidden = O::v(0x800000u);
  const auto aa = a & absbits, bb = b & absbits;
  const auto swap = aa < bb;
  const auto big = O::choose(swap, b, a), small = O::choose(swap, a, b);
  const auto same = ((big ^ small) & signbit) == zero;
  auto ea = (big >> 23) & O::v(255), eb = (small >> 23) & O::v(255);
  auto ma = ((big & frac) | O::choose(ea == zero, zero, hidden)) << 3;
  auto mb = ((small & frac) | O::choose(eb == zero, zero, hidden)) << 3;
  ea = O::choose(ea == zero, one, ea);
  eb = O::choose(eb == zero, one, eb);
  auto gap = ea - eb;
  gap = O::choose(gap > O::v(31), O::v(31), gap);
  // Per-lane variable shift with sticky, expressed as fixed SIMD shifts.
#pragma clang loop unroll(full)
  for (unsigned s = 16; s; s >>= 1) {
    auto jam = (mb >> s) | O::choose((mb & O::v((1u << s) - 1)) != zero, one, zero);
    mb = O::choose((gap & O::v(s)) != zero, jam, mb);
  }
  auto sum = O::choose(same, ma + mb, ma - mb);
  const auto overflow = sum >= O::v(1u << 27);
  sum = O::choose(overflow, (sum >> 1) | (sum & one), sum);
  ea = ea + O::choose(overflow, one, zero);
#pragma clang loop unroll(full)
  for (unsigned s = 16; s; s >>= 1) {
    const auto shift = (sum < O::v(1u << (27 - s))) & (ea > O::v(s));
    sum = O::choose(shift, sum << s, sum);
    ea = ea - O::choose(shift, O::v(s), zero);
  }
  const auto low = sum & O::v(7);
  auto mantissa = (sum >> 3) + O::choose((low > O::v(4)) |
      ((low == O::v(4)) & (((sum >> 3) & one) != zero)), one, zero);
  const auto carry = mantissa >= O::v(1u << 24);
  mantissa = O::choose(carry, mantissa >> 1, mantissa);
  ea = ea + O::choose(carry, one, zero);
  ea = O::choose(mantissa < hidden, zero, ea);
  auto result = (big & signbit) | (ea << 23) | (mantissa & frac);
  result = O::choose(ea >= O::v(255), (big & signbit) | O::v(0x7f800000), result);
  // Exact cancellation is +0; -0 + -0 is -0. Propagate quiet NaNs and infinities.
  result = O::choose(mantissa == zero, (a & b) & signbit, result);
  const auto inf = O::v(0x7f800000), quiet = O::v(0x00400000);
  result = O::choose((aa == inf) | (bb == inf), big, result);
  result = O::choose((aa == inf) & (bb == inf) & (((big ^ small) & signbit) != zero), O::v(0x7fc00000), result);
  result = O::choose(bb > inf, b | quiet, result);
  return O::choose(aa > inf, a | quiet, result);
}
