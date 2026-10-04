#pragma once
//===- gemv_t2.h -------------------------------------------*- C++ -*-===//
//
// Ternary (2-bit) GEMV on the AIE core: y[N] = W[N, K] @ x[K] with
//   W[r, k] = s[r, k/128] * code - s[r, k/128],   code in {0, 1, 2}
// -- PrismML's PQ2_0 (Ternary Bonsai 2) repacked into pool-order chunks by
// ../../t2_pack.py. Probe 0a of .claude/plans/bonsai-2-ternary.md: does the
// core keep up with the DMA when every byte carries four weights, not two?
//
//   chunk = 32 output rows x 256 K, 2176 B:
//     s    [2][32] bf16 at [0 : 128]      group g = k / 128 of the chunk, row r
//     code [8 kb][4 oc][8 kk][8 p] B at [128 : 2176]
//          byte (kb, oc, kk, p), bits 2j..2j+1 = code(row 8j + p, k = 32 kb + 8 oc + kk)
//
// Band law and activation table are gemv_q4's (64-row bands, half = c % 2,
// k-tile = c / 2; gemv_tab.h's int16 blocks with per-32 power-of-two scales).
//
// The inner product is gemv_q4's too: mmul<4, 8, 8, int16, uint8> with B = one
// 64 B block straight from the chunk, masked to ONE 2-bit field. Field j selects
// rows 8j..8j+7 with value code * 4^j (at most 128, so uint8 holds it); 4^-j
// folds into the scale exactly as the odd rows' 16 does in gemv_q4. Two fields
// per pass (0x03 / 0x0C, then 0x30 / 0xC0) keep two mmul accumulators live, the
// same pressure as gemv_q4's even / odd pair. Per 32-wide K block both kernels
// issue 16 mmuls and 16 masks for 1024 weights; this one reads 2.35x fewer
// bytes. The four products land in row order, so neither the scales nor y need
// gemv_q4's unzip / zip.
//
// The min term is -s times the block sum, so each 32-wide K block's epilogue is
// gemv_q4's four macs with m = -s:
//   y[r] += s[g][r] * 4^-j * 2^-sh * part[r]  (bf16 hi/lo)  -  s[g][r] * xs[kb]

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gemv_tab.h"   // the activation's int16 block table (gemv_q4_prep fills it)

static constexpr unsigned kT2Rows = 32;        // output rows per chunk
static constexpr unsigned kT2BandRows = 64;    // output rows per pool band
static constexpr unsigned kT2KBlocks = 8;      // 32-wide K blocks per chunk
static constexpr unsigned kT2TileK = 256;      // K per chunk
static constexpr unsigned kT2ScaleBytes = 128; // [2 groups][32 rows] bf16
static constexpr unsigned kT2TileBytes = 2176;

#ifndef GEMV_PER_CALL
#define GEMV_PER_CALL 4
#endif
#ifndef GEMV_PER_BAND
#define GEMV_PER_BAND 16
#endif
#ifndef GEMV_ROWSPLIT
#define GEMV_ROWSPLIT 2
#endif
static constexpr unsigned kT2PerCall = GEMV_PER_CALL;
static constexpr unsigned kT2PerBand = GEMV_PER_BAND;
static constexpr unsigned kT2RowSplit = GEMV_ROWSPLIT;

// [1 x8 | 1/4 x8 | 1/16 x8 | 1/64 x8] as bf16: field j's mmul value is 4^j * code.
// Built from 32-lane broadcasts and selects (no 128-bit vector ops on this backend).
static inline aie::vector<bfloat16, kT2Rows> gemv_t2_field_scale() {
  aie::vector<int16_t, kT2Rows> v = aie::broadcast<int16_t, kT2Rows>((int16_t)0x3F80);   // 1
  v = aie::select(v, aie::broadcast<int16_t, kT2Rows>((int16_t)0x3E80), aie::mask<kT2Rows>::from_uint32(0x0000FF00u));
  v = aie::select(v, aie::broadcast<int16_t, kT2Rows>((int16_t)0x3D80), aie::mask<kT2Rows>::from_uint32(0x00FF0000u));
  v = aie::select(v, aie::broadcast<int16_t, kT2Rows>((int16_t)0x3C80), aie::mask<kT2Rows>::from_uint32(0xFF000000u));
  return v.template cast_to<bfloat16>();
}

// One chunk against the k-tile `kt` of the table. `first` starts the band
// accumulator; the rows are already in order, so `last` changes nothing.
// GEMV_NULL: skip the arithmetic (DMA-only probe of the design's dataflow).
__attribute__((noinline)) inline void gemv_t2_tile(const uint8_t *__restrict tile,
                                                   const uint8_t *__restrict tab, unsigned K,
                                                   unsigned kt, bool first, bool last,
                                                   float *__restrict y) {
  (void)last;
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y, aie::zeros<float, kT2Rows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NB = K / 32;

  const bfloat16 *__restrict sp = (const bfloat16 *)tile;
  const uint8_t *__restrict code = tile + kT2ScaleBytes;
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kT2TileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * K) + kt * kT2KBlocks;
  const bfloat16 *__restrict xsh = (const bfloat16 *)(tab + 2 * K + 4 * NB) + kt * kT2KBlocks;
  const bfloat16 *__restrict xsl = xsh + NB;

  aie::accum<accfloat, kT2Rows> acc;
  if (first)
    acc = aie::zeros<accfloat, kT2Rows>();
  else
    acc.from_vector(aie::load_v<kT2Rows>(y));

  const aie::vector<bfloat16, kT2Rows> fsc = gemv_t2_field_scale();

#pragma clang loop unroll(disable)
  for (unsigned g = 0; g < 2; ++g) {
    const aie::vector<bfloat16, kT2Rows> s32 = aie::load_v<kT2Rows>(sp + g * kT2Rows);
    const aie::vector<bfloat16, kT2Rows> ds = aie::mul(s32, fsc).template to_vector<bfloat16>();   // exact

#pragma clang loop unroll(disable)
    for (unsigned b = 0; b < 4; ++b) {
      const unsigned kb = g * 4 + b;
      const uint8_t *__restrict src = code + kb * 256;
      aie::vector<int32_t, 8> ve[2], vo[2];
#pragma clang loop unroll(full)
      for (unsigned jp = 0; jp < 2; ++jp) {
        const uint8_t me = jp == 0 ? (uint8_t)0x03 : (uint8_t)0x30;   // rows 0..7  / 16..23
        const uint8_t mo = jp == 0 ? (uint8_t)0x0C : (uint8_t)0xC0;   // rows 8..15 / 24..31
        aie::mmul<4, 8, 8, int16_t, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
        for (unsigned oc = 0; oc < 4; ++oc) {
          const aie::vector<int16_t, 32> A =
              aie::load_v<8>(xi + kb * 32 + oc * 8).template grow_replicate<32>();
          const aie::vector<uint8_t, 64> q = aie::load_v<64>(src + oc * 64);
          const aie::vector<uint8_t, 64> e = aie::bit_and(me, q);
          const aie::vector<uint8_t, 64> o = aie::bit_and(mo, q);
          if (oc == 0) {
            Ce.mul(A, e);
            Co.mul(A, o);
          } else {
            Ce.mac(A, e);
            Co.mac(A, o);
          }
        }
        ve[jp] = Ce.template to_vector<int32_t>().template extract<8>(0);
        vo[jp] = Co.template to_vector<int32_t>().template extract<8>(0);
      }
      // rows [0..7 | 8..15 | 16..23 | 24..31]
      const aie::vector<int32_t, kT2Rows> vi = aie::concat(ve[0], vo[0], ve[1], vo[1]);
      aie::accum<accfloat, kT2Rows> part;
      part.from_vector(aie::to_float<float>(vi, sh[kb]));
      const aie::vector<bfloat16, kT2Rows> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kT2Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();

      acc = aie::mac(acc, hi, ds);
      acc = aie::mac(acc, lo, ds);
      acc = aie::msc(acc, s32, xsh[kb]);   // m = -s (no bf16 negate on this backend: G_FNEG)
      acc = aie::msc(acc, s32, xsl[kb]);
    }
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
#endif
}

// ---------------------------------------------------------------------------
// GEMV_T2_G128: one activation exponent per 128-wide K block -- the weight
// scale's own group -- so the int32 products of a whole group accumulate in the
// mmul registers and the float epilogue runs once per 128 K instead of per 32.
// Peano's remarks on the per-32 form: 82 bundles per 32-wide block, unpipelined,
// RecMII 61 from the four dependent fp macs on the band accumulator; per-128 the
// chain is paid twice a chunk, not eight times.
//
// Range: |xi| < 2^15, field value <= 128, 128 terms -> |sum| < 2^29, inside int32.
// Precision: x is exact to int16 for elements within 2^7 of the 128-block max
// (gemv_tab.h's per-32 rule, over 4x the span).
//
// Table: int16 xi[K] | int32 s[K/128] | bf16 xs_hi[K/128] | bf16 xs_lo[K/128]
// ---------------------------------------------------------------------------
static constexpr unsigned gemv_t2_tab_bytes(unsigned K) { return 2 * K + K / 32 + K / 32; }

// 32 fp32 lanes -> their sum as bf16 hi / lo (hi + lo == sum to ~2^-16).
static inline void gemv_t2_sum32(const aie::vector<float, 32> &v32, bfloat16 &hi, bfloat16 &lo) {
  aie::accum<accfloat, 16> a16, b16;
  a16.from_vector(v32.template extract<16>(0));
  b16.from_vector(v32.template extract<16>(1));
  const aie::vector<float, 16> v16 = aie::add(a16, b16).template to_vector<float>();
  aie::accum<accfloat, 8> a8, b8;
  a8.from_vector(v16.template extract<8>(0));
  b8.from_vector(v16.template extract<8>(1));
  aie::vector<float, 16> t = aie::concat(aie::add(a8, b8).template to_vector<float>(), aie::zeros<float, 8>());
#pragma clang loop unroll(full)
  for (unsigned st = 4; st >= 1; st >>= 1) {
    aie::accum<accfloat, 16> p, q;
    p.from_vector(t);
    q.from_vector(aie::shuffle_down_rotate(t, st));
    t = aie::add(p, q).template to_vector<float>();
  }
  aie::accum<accfloat, 32> bc;
  bc.from_vector(aie::broadcast<float, 32>(t[0]));
  hi = bc.template to_vector<bfloat16>()[0];
  lo = aie::sub(bc, bc.template to_vector<bfloat16>()).template to_vector<bfloat16>()[0];
}

// Block-quantise x[K] into the per-128 table (once per x per core).
__attribute__((noinline)) inline void gemv_t2_prep(const bfloat16 *__restrict x, uint8_t *__restrict tab,
                                                   unsigned K) {
  const unsigned NG = K / 128;
  aie::set_rounding(aie::rounding_mode::conv_even);
  int16_t *__restrict xi = (int16_t *)tab;
  int32_t *__restrict sh = (int32_t *)(tab + 2 * K);
  bfloat16 *__restrict xsh = (bfloat16 *)(tab + 2 * K + 4 * NG);
  bfloat16 *__restrict xsl = xsh + NG;
  const aie::vector<uint8_t, 64> absmask = gemv_q4_absmask();
#pragma clang loop unroll(disable)
  for (unsigned g = 0; g < NG; ++g) {
    const bfloat16 *__restrict xb = x + g * 128;
    aie::vector<int16_t, 32> mx = aie::zeros<int16_t, 32>();
#pragma clang loop unroll(full)
    for (unsigned i = 0; i < 4; ++i)
      mx = aie::max(mx, aie::bit_and(aie::load_v<32>(xb + 32 * i).template cast_to<uint8_t>(), absmask)
                            .template cast_to<int16_t>());
    int s = 141 - (aie::reduce_max(mx) >> 7);    // 14 - (e - 127), positive floats order as their bits
    if (s > 126) s = 126;                        // zero / tiny block: any scale works
    sh[g] = s;
    const aie::vector<bfloat16, 32> sc =
        aie::broadcast<int16_t, 32>((int16_t)((127 + s) << 7)).template cast_to<bfloat16>();
    aie::accum<accfloat, 32> sum = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(full)
    for (unsigned i = 0; i < 4; ++i) {
      const aie::vector<bfloat16, 32> xv = aie::load_v<32>(xb + 32 * i);
      aie::store_v(xi + g * 128 + 32 * i, aie::to_fixed<int16_t>(aie::mul(xv, sc), 0));
      aie::accum<accfloat, 32> xa;
      xa.from_vector(xv);
      sum = aie::add(sum, xa);
    }
    gemv_t2_sum32(sum.template to_vector<float>(), xsh[g], xsl[g]);
  }
}

__attribute__((noinline)) inline void gemv_t2_tile_g128(const uint8_t *__restrict tile,
                                                        const uint8_t *__restrict tab, unsigned K,
                                                        unsigned kt, bool first, bool last,
                                                        float *__restrict y) {
  (void)last;
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y, aie::zeros<float, kT2Rows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NG = K / 128;
  const bfloat16 *__restrict sp = (const bfloat16 *)tile;
  const uint8_t *__restrict code = tile + kT2ScaleBytes;
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kT2TileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * K) + kt * 2;
  const bfloat16 *__restrict xsh = (const bfloat16 *)(tab + 2 * K + 4 * NG) + kt * 2;
  const bfloat16 *__restrict xsl = xsh + NG;

  aie::accum<accfloat, kT2Rows> acc;
  if (first)
    acc = aie::zeros<accfloat, kT2Rows>();
  else
    acc.from_vector(aie::load_v<kT2Rows>(y));
  const aie::vector<bfloat16, kT2Rows> fsc = gemv_t2_field_scale();

#pragma clang loop unroll(disable)
  for (unsigned g = 0; g < 2; ++g) {
    const aie::vector<bfloat16, kT2Rows> s32 = aie::load_v<kT2Rows>(sp + g * kT2Rows);
    const aie::vector<bfloat16, kT2Rows> ds = aie::mul(s32, fsc).template to_vector<bfloat16>();   // exact
    const int16_t *__restrict xg = xi + g * 128;
    const uint8_t *__restrict cg = code + g * 1024;
    aie::vector<int32_t, 8> ve[2], vo[2];
#pragma clang loop unroll(full)
    for (unsigned jp = 0; jp < 2; ++jp) {
      const uint8_t me = jp == 0 ? (uint8_t)0x03 : (uint8_t)0x30;   // rows 0..7  / 16..23
      const uint8_t mo = jp == 0 ? (uint8_t)0x0C : (uint8_t)0xC0;   // rows 8..15 / 24..31
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned o = 0; o < 16; ++o) {                            // 16 octets of K = one group
        const aie::vector<int16_t, 32> A = aie::load_v<8>(xg + o * 8).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(cg + o * 64);
        const aie::vector<uint8_t, 64> e = aie::bit_and(me, q);
        const aie::vector<uint8_t, 64> od = aie::bit_and(mo, q);
        if (o == 0) {
          Ce.mul(A, e);
          Co.mul(A, od);
        } else {
          Ce.mac(A, e);
          Co.mac(A, od);
        }
      }
      ve[jp] = Ce.template to_vector<int32_t>().template extract<8>(0);
      vo[jp] = Co.template to_vector<int32_t>().template extract<8>(0);
    }
    const aie::vector<int32_t, kT2Rows> vi = aie::concat(ve[0], vo[0], ve[1], vo[1]);
    aie::accum<accfloat, kT2Rows> part;
    part.from_vector(aie::to_float<float>(vi, sh[g]));
    const aie::vector<bfloat16, kT2Rows> hi = part.template to_vector<bfloat16>();
    const aie::vector<bfloat16, kT2Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    acc = aie::mac(acc, hi, ds);
    acc = aie::mac(acc, lo, ds);
    acc = aie::msc(acc, s32, xsh[g]);
    acc = aie::msc(acc, s32, xsl[g]);
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
#endif
}

// ---------------------------------------------------------------------------
// GEMV_T2_FWHT: Bonsai 2's activation-side rotation, applied before the table.
// Every 1024-wide block of x becomes H(x)/32 (normalized Sylvester Walsh-Hadamard;
// the GGUF's +-1 signs are assumed folded into the producer, see the plan). Done
// in fp32 in a scratch block, written back over x as bf16 -- the activation
// precision every producer in the engine already emits.
//   stride >= 16: six stages of vector pairs, through the scratch block;
//   stride 1..8:  four in-register stages per 16-lane vector, then the 1/32
//                 (a power of two, so exact) and the bf16 rounding.
// ---------------------------------------------------------------------------
static float gemv_t2_wht_scratch[1024] __attribute__((aligned(64)));

__attribute__((noinline)) inline void gemv_t2_fwht1024(bfloat16 *__restrict x) {
  float *__restrict v = gemv_t2_wht_scratch;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 1024; j += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(x + j));
    aie::store_v(v + j, a.template to_vector<float>());
  }
#pragma clang loop unroll(disable)
  for (unsigned t = 16; t < 1024; t <<= 1) {
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < 1024; i += 2 * t) {
#pragma clang loop unroll(disable)
      for (unsigned j = i; j < i + t; j += 16) {
        aie::accum<accfloat, 16> a, b;
        a.from_vector(aie::load_v<16>(v + j));
        b.from_vector(aie::load_v<16>(v + j + t));
        aie::store_v(v + j, aie::add(a, b).template to_vector<float>());
        aie::store_v(v + j + t, aie::sub(a, b).template to_vector<float>());
      }
    }
  }
  const aie::vector<bfloat16, 16> inv32 = aie::broadcast<bfloat16, 16>((bfloat16)0.03125f);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 1024; j += 16) {
    aie::vector<float, 16> y = aie::load_v<16>(v + j);
#pragma clang loop unroll(full)
    for (unsigned t = 1; t < 16; t <<= 1) {
      aie::accum<accfloat, 16> ya, dn, up;
      ya.from_vector(y);
      dn.from_vector(aie::shuffle_down_rotate(y, t));    // lane l <- l + t (right for bit t clear)
      up.from_vector(aie::shuffle_up_rotate(y, t));      // lane l <- l - t (right for bit t set)
      const aie::vector<float, 16> s = aie::add(ya, dn).template to_vector<float>();
      const aie::vector<float, 16> d = aie::sub(up, ya).template to_vector<float>();
      const uint32_t bits = t == 1 ? 0xAAAAu : t == 2 ? 0xCCCCu : t == 4 ? 0xF0F0u : 0xFF00u;
      y = aie::select(s, d, aie::mask<16>::from_uint32(bits));
    }
    aie::accum<accfloat, 16> ya;
    ya.from_vector(y);
    aie::store_v(x + j, aie::mul(ya.template to_vector<bfloat16>(), inv32).template to_vector<bfloat16>());
  }
}

#ifdef GEMV_T2_FWHT
#define GEMV_T2_PREP_ENTRY_(K)                                              \
  void gemv_t2_prep_k##K(bfloat16 *__restrict x, uint8_t *__restrict tab) { \
    for (unsigned b = 0; b < (K) / 1024; ++b) gemv_t2_fwht1024(x + b * 1024); \
    gemv_t2_prep(x, tab, K);                                                \
  }
#else
#define GEMV_T2_PREP_ENTRY_(K)                                              \
  void gemv_t2_prep_k##K(const bfloat16 *__restrict x, uint8_t *__restrict tab) { \
    gemv_t2_prep(x, tab, K);                                                \
  }
#endif
#define GEMV_T2_PREP_ENTRY(K) GEMV_T2_PREP_ENTRY_(K)

#ifdef GEMV_T2_G128
#define GEMV_T2_TILE gemv_t2_tile_g128
#else
#define GEMV_T2_TILE gemv_t2_tile
#endif

// One call = kT2PerCall consecutive pool-order chunks of one band, group `group`
// (chunks group*kT2PerCall .. +kT2PerCall-1). y is the band's 64-float
// accumulator; it is complete after the band's last group.
static constexpr unsigned kT2K = kT2TileK * kT2PerBand / kT2RowSplit;
static inline void gemv_t2_pool_group(const uint8_t *__restrict chunks,
                                      const uint8_t *__restrict tab,
                                      unsigned group, float *__restrict y) {
  constexpr unsigned kKt = kT2PerBand / kT2RowSplit;   // k-tiles per band
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kT2PerCall; ++i) {
    const unsigned c = group * kT2PerCall + i;
    const unsigned part = c % kT2RowSplit;
    const unsigned kt = c / kT2RowSplit;
    GEMV_T2_TILE(chunks + i * kT2TileBytes, tab, kT2K, kt, kt == 0, kt == kKt - 1, y + part * kT2Rows);
  }
}

// gemv_t2_p<P>b<B>r<R>_g(chunks, tab, y, group, band): the group and the band as
// RUNTIME arguments (from range_ loops), as gemv_q4's GROUP_ENTRY.
#define GEMV_T2_GROUP_ENTRY__(P, B, R)                                       \
  void gemv_t2_p##P##b##B##r##R##_g(const uint8_t *__restrict t,            \
                                    const uint8_t *__restrict tab,          \
                                    float *__restrict y, int32_t group,     \
                                    int32_t band) {                         \
    gemv_t2_pool_group(t, tab, (unsigned)group, y + band * kT2BandRows * kT2RowSplit / 2); \
  }
#define GEMV_T2_GROUP_ENTRY_(P, B, R) GEMV_T2_GROUP_ENTRY__(P, B, R)
#define GEMV_T2_GROUP_ENTRY() GEMV_T2_GROUP_ENTRY_(GEMV_PER_CALL, GEMV_PER_BAND, GEMV_ROWSPLIT)
