#pragma once
//===- wht.h -----------------------------------------------*- C++ -*-===//
//
// The activation side of PrismML's rotated-basis ternary models (Ternary Bonsai 2;
// OPEN-HADAMARD). Their weights are stored after a blockwise Walsh-Hadamard rotation,
// so every projection input must go through the same rotation before its GEMV:
//
//   x[1024 b .. 1024 b + 1023]  ->  H(x block) / 32        (normalized Sylvester H)
//
// The model's +-1 signs are folded into weights by q4nx-build wherever that is exact;
// the attention output's are not (a silu / shared kv heads sit in the way), so
// xh_signs_bf16 applies them from a bit table first.
//
// Done in fp32 in a scratch block and written back as bf16, the precision every
// producer feeding these preps already emits:
//   stride >= 16: six stages of vector pairs through the scratch block;
//   stride 1..8:  four in-register stages per 16-lane vector (rotates + selects),
//                 then the 1/32 -- a power of two, so exact -- and the bf16 rounding.
// Measured on its own (designs/gemv_t2, probe 0a): ~5 us per 1024 block per core in the
// one-stage-per-pass form, 864 B of program memory, 4 KB of data memory for the scratch block.

#include <aie_api/aie.hpp>
#include <stdint.h>

static float xh_wht_scratch[1024] __attribute__((aligned(64)));

// Passes t = 64 and 256, in place on the scratch block (one copy for both input types).
__attribute__((noinline)) inline void xh_wht_wide_rest(float *__restrict v) {
#pragma clang loop unroll(disable)
  for (unsigned t = 64; t < 1024; t <<= 2) {
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < 1024; i += 4 * t) {
#pragma clang loop unroll(disable)
      for (unsigned j = i; j < i + t; j += 16) {
        aie::accum<accfloat, 16> a, b, c, d;
        a.from_vector(aie::load_v<16>(v + j));
        b.from_vector(aie::load_v<16>(v + j + t));
        c.from_vector(aie::load_v<16>(v + j + 2 * t));
        d.from_vector(aie::load_v<16>(v + j + 3 * t));
        aie::accum<accfloat, 16> s1, d1, s2, d2;
        s1.from_vector(aie::add(a, b).template to_vector<float>());
        d1.from_vector(aie::sub(a, b).template to_vector<float>());
        s2.from_vector(aie::add(c, d).template to_vector<float>());
        d2.from_vector(aie::sub(c, d).template to_vector<float>());
        aie::store_v(v + j, aie::add(s1, s2).template to_vector<float>());
        aie::store_v(v + j + 2 * t, aie::sub(s1, s2).template to_vector<float>());
        aie::store_v(v + j + t, aie::add(d1, d2).template to_vector<float>());
        aie::store_v(v + j + 3 * t, aie::sub(d1, d2).template to_vector<float>());
      }
    }
  }
}

// The six wide stages into the scratch block, two per pass (strides t and 2t, t = 16, 64, 256):
// four vectors in, both butterfly levels in registers, four out. The same fp32 adds and
// subtracts on the same operands as one stage per pass -- stage t makes a+b, a-b, c+d, c-d and
// stage 2t pairs (j, j+2t) and (j+t, j+3t) -- so the same bits, in half the passes and a third
// of the loop iterations (the single-stage form was loop- and latency-bound, ~5 us a block).
// The first pass reads the input block itself (bf16 widened exactly, or fp32) and writes the
// scratch, where a separate loop used to copy the block into the scratch first: the same
// values reach the same butterflies.
template <typename T>
__attribute__((noinline)) inline void xh_wht_wide(const T *__restrict x, float *__restrict v) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 1024; j += 64) {                  // t = 16: one butterfly per 64 values
    aie::accum<accfloat, 16> a, b, c, d;
    a.from_vector(aie::load_v<16>(x + j));
    b.from_vector(aie::load_v<16>(x + j + 16));
    c.from_vector(aie::load_v<16>(x + j + 32));
    d.from_vector(aie::load_v<16>(x + j + 48));
    aie::accum<accfloat, 16> s1, d1, s2, d2;
    s1.from_vector(aie::add(a, b).template to_vector<float>());
    d1.from_vector(aie::sub(a, b).template to_vector<float>());
    s2.from_vector(aie::add(c, d).template to_vector<float>());
    d2.from_vector(aie::sub(c, d).template to_vector<float>());
    aie::store_v(v + j, aie::add(s1, s2).template to_vector<float>());
    aie::store_v(v + j + 32, aie::sub(s1, s2).template to_vector<float>());
    aie::store_v(v + j + 16, aie::add(d1, d2).template to_vector<float>());
    aie::store_v(v + j + 48, aie::sub(d1, d2).template to_vector<float>());
  }
  xh_wht_wide_rest(v);
}

// The four in-register stages, the 1/32 and the bf16 rounding, scratch -> out (never the scratch).
// A stage pairs lane l (bit t clear) with lane l + t: the even and odd groups of t lanes are split
// out, added and subtracted at half width and zipped back -- y[l] + y[l + t] and y[l] - y[l + t] on
// exactly the lanes the full-width add / subtract and select used to keep, so the same bits for
// half the adds (43 -> 36 bundles a vector by kernel_remarks).
__attribute__((noinline)) inline void xh_wht_narrow_out(const float *__restrict v, bfloat16 *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, 16> inv32 = aie::broadcast<bfloat16, 16>((bfloat16)0.03125f);
  // Two vectors per iteration: each one's four stages are a dependent chain, and two chains
  // side by side fill the bundles one leaves empty.
#pragma clang loop unroll_count(2)
  for (unsigned j = 0; j < 1024; j += 16) {
    aie::vector<float, 16> y = aie::load_v<16>(v + j);
#pragma clang loop unroll(full)
    for (unsigned t = 1; t < 16; t <<= 1) {
      const aie::vector<float, 8> a = aie::filter_even(y, t), b = aie::filter_odd(y, t);
      const auto z = aie::interleave_zip(aie::add(a, b), aie::sub(a, b), t);
      y = aie::concat(z.first, z.second);
    }
    aie::accum<accfloat, 16> ya;
    ya.from_vector(y);
    aie::store_v(out + j, aie::mul(ya.template to_vector<bfloat16>(), inv32).template to_vector<bfloat16>());
  }
}

// One bf16 block in place.
__attribute__((noinline)) inline void xh_wht_bf16(bfloat16 *x) {
  xh_wht_wide<bfloat16>(x, xh_wht_scratch);
  xh_wht_narrow_out(xh_wht_scratch, x);
}

// One fp32 block -> bf16 at `out` (which may be the block's own first half: the input is read
// only by the first wide pass, before anything is written there).
__attribute__((noinline)) inline void xh_wht_f32(const float *x, bfloat16 *out) {
  xh_wht_wide<float>(x, xh_wht_scratch);
  xh_wht_narrow_out(xh_wht_scratch, out);
}

// Flip the sign of x[j] where bit j of `bits` is set (bit b of word w <-> value 32 w + b), n a
// multiple of 32. On the bf16 bit pattern: this backend has no bf16 negate (G_FNEG).
__attribute__((noinline)) inline void xh_signs_bf16(bfloat16 *x, const uint32_t *__restrict bits, unsigned n) {
  int16_t *__restrict xi = (int16_t *)x;
  const aie::vector<int16_t, 32> sb = aie::broadcast<int16_t, 32>((int16_t)0x8000);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < n; j += 32) {
    const aie::vector<int16_t, 32> v = aie::load_v<32>(xi + j);
    aie::store_v(xi + j, aie::select(v, aie::bit_xor(v, sb), aie::mask<32>::from_uint32(bits[j >> 5])));
  }
}
