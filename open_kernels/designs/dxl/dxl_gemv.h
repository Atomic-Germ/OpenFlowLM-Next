#pragma once
// MACs in gemv_q4_tile's order, not gemm_q4_tile4's (it swaps the last two), so each row is bit-identical to decode.

#define GEMV_PER_CALL 1
#include "../gemv_q4/gemv_q4.h"

#ifndef DXL_L
#define DXL_L 4
#endif
static constexpr unsigned kL = DXL_L;
static_assert(kL % 4 == 0, "DXL_L must be a multiple of 4 (tokens ride the mmul's four rows)");
static constexpr unsigned kBandFloats = kRows * 2;      // 64: one band's accumulator per token

// The app's greedy compares logits rounded to bf16 (nearest even, subnormals to zero); this is that order as an int.
static inline int32_t greedy_key(int32_t v) {
  const uint32_t u = (uint32_t)v;
  const int32_t s = (int32_t)((u & 0x7f800000u) ? (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u : 0u);
  return s < 0 ? s ^ 0x7fffffff : s;
}

__attribute__((noinline)) inline void dxl_tile4(const uint8_t *__restrict tile,
                                                const uint8_t *__restrict tab0, unsigned tabStride,
                                                unsigned K, unsigned kt, bool first, bool last,
                                                float *__restrict y, unsigned yStride) {
  event0();
#ifdef GEMV_NULL
  if (first) {
#pragma clang loop unroll(full)
    for (unsigned t = 0; t < 4; ++t)
      aie::store_v(y + t * yStride, aie::zeros<float, kRows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NB = K / 32;

  const bfloat16 *__restrict dp = (const bfloat16 *)tile;
  const bfloat16 *__restrict mp = (const bfloat16 *)(tile + kDBytes);
  const uint8_t *__restrict nib0 = tile + kMetaBytes;
  const uint8_t *__restrict nib1 = tile + kMetaBytes + 2048;

  const int16_t *__restrict xi[4];
  const int32_t *__restrict sh[4];
  const bfloat16 *__restrict xsh[4];
  const bfloat16 *__restrict xsl[4];
#pragma clang loop unroll(full)
  for (unsigned t = 0; t < 4; ++t) {
    const uint8_t *__restrict tb = tab0 + t * tabStride;
    xi[t] = (const int16_t *)tb + kt * kTileK;
    sh[t] = (const int32_t *)(tb + 2 * K) + kt * kKBlocks;
    xsh[t] = (const bfloat16 *)(tb + 2 * K + 4 * NB) + kt * kKBlocks;
    xsl[t] = xsh[t] + NB;
  }

  aie::accum<accfloat, kRows> acc[4];
#pragma clang loop unroll(full)
  for (unsigned t = 0; t < 4; ++t) {
    if (first)
      acc[t] = aie::zeros<accfloat, kRows>();
    else
      acc[t].from_vector(aie::load_v<kRows>(y + t * yStride));
  }

  const aie::vector<bfloat16, kRows> dsc =
      aie::concat(aie::broadcast<int16_t, 16>((int16_t)0x3F80),
                  aie::broadcast<int16_t, 16>((int16_t)0x3D80)).template cast_to<bfloat16>();

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int32_t, 8> ve[4][2], vo[4][2];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? nib0 : nib1) + kb * 256;
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const unsigned off = kb * kKInBlock + oc * 8;
        const aie::vector<int16_t, 32> A =
            aie::concat(aie::load_v<8>(xi[0] + off), aie::load_v<8>(xi[1] + off),
                        aie::load_v<8>(xi[2] + off), aie::load_v<8>(xi[3] + off));
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(src + oc * 64);
        const aie::vector<uint8_t, 64> e = aie::bit_and((uint8_t)0x0F, q);
        const aie::vector<uint8_t, 64> o = aie::bit_and((uint8_t)0xF0, q);
        if (oc == 0) {
          Ce.mul(A, e);
          Co.mul(A, o);
        } else {
          Ce.mac(A, e);
          Co.mac(A, o);
        }
      }
      const aie::vector<int32_t, 32> Cev = Ce.template to_vector<int32_t>();
      const aie::vector<int32_t, 32> Cov = Co.template to_vector<int32_t>();
#pragma clang loop unroll(full)
      for (unsigned t = 0; t < 4; ++t) {
        ve[t][rb] = Cev.template extract<8>(t);
        vo[t][rb] = Cov.template extract<8>(t);
      }
    }

    const aie::vector<bfloat16, kRows> d32 = aie::load_v<kRows>(dp + kb * kRows);
    const aie::vector<bfloat16, kRows> m32 = aie::load_v<kRows>(mp + kb * kRows);
    auto [de, dod] = aie::interleave_unzip(d32, d32, 1);
    auto [me, mo] = aie::interleave_unzip(m32, m32, 1);
    const aie::vector<bfloat16, kRows> dperm =
        aie::concat(de.template extract<16>(0), dod.template extract<16>(0));
    const aie::vector<bfloat16, kRows> mperm =
        aie::concat(me.template extract<16>(0), mo.template extract<16>(0));
    const aie::vector<bfloat16, kRows> ds = aie::mul(dperm, dsc).template to_vector<bfloat16>();

#pragma clang loop unroll(full)
    for (unsigned t = 0; t < 4; ++t) {
      const aie::vector<int32_t, kRows> vi = aie::concat(ve[t][0], ve[t][1], vo[t][0], vo[t][1]);
      aie::accum<accfloat, kRows> part;
      part.from_vector(aie::to_float<float>(vi, sh[t][kb]));
      const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], hi, ds);
      acc[t] = aie::mac(acc[t], lo, ds);
      acc[t] = aie::mac(acc[t], mperm, xsh[t][kb]);
      acc[t] = aie::mac(acc[t], mperm, xsl[t][kb]);
    }
  }

#pragma clang loop unroll(full)
  for (unsigned t = 0; t < 4; ++t) {
    const aie::vector<float, kRows> yv = acc[t].template to_vector<float>();
    float *__restrict yt = y + t * yStride;
    if (last) {
      auto [r0, r1] = aie::interleave_zip(yv.template extract<16>(0), yv.template extract<16>(1), 1);
      aie::store_v(yt, r0);
      aie::store_v(yt + 16, r1);
    } else {
      aie::store_v(yt, yv);
    }
  }
  event1();
#endif
}

// kt_total counts every k-tile the band will see, LoRA tiles included, so it can exceed K/256.
static inline void dxl_gemv_chunk(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                  float *__restrict y, unsigned c, unsigned s, unsigned KS,
                                  unsigned kt_total) {
  const unsigned part = c & 1u, kt = c >> 1;
  const unsigned ktg = s * (KS / kTileK) + kt;
  const unsigned tb = gemv_q4_tab_bytes(KS);
#pragma clang loop unroll(disable)
  for (unsigned m = 0; m < kL; m += 4) {
    dxl_tile4(tile, tab + m * tb, tb, KS, kt, ktg == 0, ktg == kt_total - 1,
              y + m * kBandFloats + part * kRows, kBandFloats);
  }
}
