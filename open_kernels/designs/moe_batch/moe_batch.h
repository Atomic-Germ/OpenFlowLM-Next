#pragma once
// moe_batch's core ops (moe_batch.py). The product is taken transposed, Y^T [8 tokens, rows] =
// X^T [8, K] @ W^T [K, rows], so the activation is the mmul's A operand and the weight its B:
// an aie::mmul<8, 8, 8> B block is 8 k x 8 rows row-major, and a q4_1 chunk's nibbles are
// stored k-major with 16 rows per k (nibble k*16 + r16, gemv_q4.h's law), so one 64-byte load
// is 8 k x 16 rows = two B blocks (the even rows in the low nibbles, the odd rows in the high),
// needing only a mask and a uint8 -> bf16 convert. The row scales d, m are contiguous per
// (32-k block, 16 rows) and ride along as vectors. Nothing is gathered or re-laid.
//
// Layouts, all per 64-row band and 8 tokens:
//   A (x or h) tile for one 64-k tile: [8 k-blocks][8 tokens][8 k] bf16 (1 KB)
//   C tile: [4 row groups of 16][2 (even rows, odd rows)][8 tokens][8 rows] f32 (2 KB);
//     row 16 g + 2 j + p sits at block (g, p) lane t * 8 + j. The odd rows' nibbles arrive
//     x16 (unshifted), folded into their d as d / 16.
#include "vecmath.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned MB_CHUNK = 5120;
static constexpr unsigned MB_NIB = 1024;      // nibbles start; d bf16[256] at 0, m at 512

// eight copies of an 8-lane pattern: the scale for a B block's 64 lanes (k-major, row fast)
static inline aie::vector<bfloat16, 64> mb_rep8(const aie::vector<bfloat16, 8> &v) {
  const aie::vector<bfloat16, 32> q = aie::concat(v, v, v, v);
  return aie::concat(q, q);
}

// C[band's 64 rows x 8 tokens] += the k-tile ky of W[band]'s product with the A tile `xa`
static inline void mb_step_tile(const uint8_t *__restrict band, unsigned ky, const bfloat16 *__restrict xa,
                                float *__restrict c) {
#ifdef MB_NULL_MM
  return;   // timing ablation: the streams without any core work
#endif
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
  const bfloat16 inv16 = 0.0625f;
  for (unsigned g = 0; g < 4; ++g) {
    const uint8_t *__restrict chunk = band + (g >> 1) * MB_CHUNK;
    const unsigned half = g & 1;
    const uint8_t *__restrict nibp = chunk + MB_NIB + half * 2048 + ky * 512;   // byte k_local * 8 + r16 / 2
    const bfloat16 *__restrict dp = reinterpret_cast<const bfloat16 *>(chunk) + half * 16;
    const bfloat16 *__restrict mp = reinterpret_cast<const bfloat16 *>(chunk + 512) + half * 16;
    float *__restrict ce = c + (g * 2) * 64;
    float *__restrict co = ce + 64;
    MMUL acc_e(aie::load_v<64>(ce));
    MMUL acc_o(aie::load_v<64>(co));
    for (unsigned kb = 0; kb < 2; ++kb) {
      const unsigned kb_abs = ky * 2 + kb;
#ifndef MB_NULL_DQ
      const aie::vector<bfloat16, 16> d16 = aie::load_v<16>(dp + kb_abs * 32);
      const aie::vector<bfloat16, 16> m16 = aie::load_v<16>(mp + kb_abs * 32);
      const auto d_eo = aie::interleave_unzip(d16.extract<8>(0), d16.extract<8>(1), 1);
      const auto m_eo = aie::interleave_unzip(m16.extract<8>(0), m16.extract<8>(1), 1);
      const aie::vector<bfloat16, 64> d_e = mb_rep8(d_eo.first);
      const aie::vector<bfloat16, 64> d_o = mb_rep8(aie::mul(d_eo.second, inv16).template to_vector<bfloat16>());
      const aie::vector<bfloat16, 64> m_e = mb_rep8(m_eo.first);
      const aie::vector<bfloat16, 64> m_o = mb_rep8(m_eo.second);
#endif
      for (unsigned il = 0; il < 4; ++il) {
        const unsigned i = kb * 4 + il;                                   // k-block inside the tile
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(nibp + i * 64);
        const aie::vector<bfloat16, 64> ne = aie::to_float<bfloat16>(aie::bit_and((uint8_t)0x0F, q), 0);
        const aie::vector<bfloat16, 64> no = aie::to_float<bfloat16>(aie::bit_and((uint8_t)0xF0, q), 0);
#ifdef MB_NULL_DQ
        const aie::vector<bfloat16, 64> we = ne, wo = no;   // timing ablation: no scales
#else
        accN<64> se, so;
        se.from_vector(m_e);
        so.from_vector(m_o);
        se = aie::mac(se, ne, d_e);
        so = aie::mac(so, no, d_o);
        const aie::vector<bfloat16, 64> we = se.template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> wo = so.template to_vector<bfloat16>();
#endif
        const aie::vector<bfloat16, 64> a = aie::load_v<64>(xa + i * 64);
        acc_e.mac(a, we);
        acc_o.mac(a, wo);
      }
    }
    aie::store_v(ce, acc_e.template to_vector<float>());
    aie::store_v(co, acc_o.template to_vector<float>());
  }
}

static inline void mb_zero_n(float *__restrict c, unsigned n) {
  const aie::vector<float, 32> z = aie::zeros<float, 32>();
  for (unsigned i = 0; i < n; i += 32) aie::store_v(c + i, z);
}

// h = silu(g) * u over the core's two C tiles (128 rows), written as the down projection's A
// tiles: rows back in natural order, [16 k-blocks][8 tokens][8 k] bf16 (2 KB)
static inline void mb_silu_tile(const float *__restrict u, const float *__restrict g, bfloat16 *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned blk = 0; blk < 16; blk += 2) {        // one 16-row group: its even and odd C blocks
    const float *__restrict ue = u + blk * 64;
    const float *__restrict ge = g + blk * 64;
    aie::vector<bfloat16, 32> he[2], ho[2];
    for (unsigned hh = 0; hh < 2; ++hh) {              // tokens 0-3, then 4-7
      accf32 a;
      a.from_vector(fmulN<32>(vsiluN<32>(aie::load_v<32>(ge + hh * 32)), aie::load_v<32>(ue + hh * 32)));
      he[hh] = a.template to_vector<bfloat16>();
      a.from_vector(fmulN<32>(vsiluN<32>(aie::load_v<32>(ge + 64 + hh * 32)), aie::load_v<32>(ue + 64 + hh * 32)));
      ho[hh] = a.template to_vector<bfloat16>();
    }
    // [t][even j], [t][odd j] -> [t][r0..r15] -> the two k-blocks' [t][8]
    const auto z = aie::interleave_zip(aie::concat(he[0], he[1]), aie::concat(ho[0], ho[1]), 1);
    const auto kbs = aie::interleave_unzip(z.first, z.second, 8);
    aie::store_v(h + blk * 64, kbs.first);
    aie::store_v(h + blk * 64 + 64, kbs.second);
  }
}
