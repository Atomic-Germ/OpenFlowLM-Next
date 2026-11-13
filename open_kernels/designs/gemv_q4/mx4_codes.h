#pragma once
//===- mx4_codes.h -----------------------------------------*- C++ -*-===//
//
// The mx4 nibble -> int8 code decode shared by gemv_mx4.h (GPT-OSS's decode GEMV) and
// gemm_bfp16's MXFP4 B operand (mx4_b.cc, the block prefill's expert GEMM).
//
// The nibble is the GGUF's e2m1 code (sign bit s, magnitude index p = bits 0..2) re-signed by
// the pack into a two's-complement int4: v = p, or -p for a negative (-0 becomes 0) but -8 for
// -12, int4's otherwise unused value. The code -- twice the e2m1 level, an integer -- is
//   v  -8  -6  -5  -4 .. 4   5   6   7        (-7 unused)
//   c -12  -8  -6  -4 .. 4   6   8  12
// This core has no vector table lookup, so it is straight lines through the levels:
//   t    = v + max(v - 4, 0) - max(-4 - v, 0)   v -8 .. 7 -> -12 . -8 -6 -4 .. 4 6 8 10
//   code = t + max(t - 8, 0)                     10 -> 12, nothing else moves
// Six vector ops for 64 weights after the load sign-extends them (vldb.unpack), no mask: the -12
// level sits on int4's spare -8, where the negative side's slope-2 line already reaches, so only
// the +12 needs the third line. (A sign mask costs two more ops, and abs / maxdiff all write their
// compare to one register pair, which Peano spilled to the stack around them.)

#include <aie_api/aie.hpp>
#include <stdint.h>

// 32 B of nibbles (64 values, byte j / 2 holding value j, the low nibble first) -> 64 int8 codes
static inline aie::vector<int8_t, 64> mx4_codes(const uint8_t *__restrict p) {
  const aie::vector<int8_t, 64> v = aie::load_v<32>(p).template cast_to<int4>().unpack();
  const aie::vector<int8_t, 64> t =
      aie::sub(aie::add(v, aie::maxdiff(v, (int8_t)4)), aie::maxdiff((int8_t)-4, v));
  return aie::add(t, aie::maxdiff(t, (int8_t)8));
}
