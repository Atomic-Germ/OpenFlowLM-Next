#pragma once
//===- gemv_t2.h -------------------------------------------*- C++ -*-===//
//
// Ternary (2-bit) GEMV on the AIE core (OPEN-QUANT-T2): y[N] = W[N, K] @ x[K] with
//   W[r, k] = s[r, k/128] * code - s[r, k/128],   code in {0, 1, 2}
// -- PrismML's PQ2_0 (Ternary Bonsai 2) as the engine packs it from the container's exact
// q4_1 copy (src/open_qwen36/pools.cpp t2_perm). Probe 0a (designs/gemv_t2) measured this
// body at its DMA floor: 2.18x the q4_1 GEMV on the 27B's gate, 2.07x on its down half.
//
//   chunk = 32 output rows x 256 K, GEMV_T2_CHUNK (2176) B, unpadded:
//     s    [2][32] bf16 at [0 : 128]      group g = k / 128 of the chunk, row r
//     code [8 kb][4 oc][8 kk][8 p] B at [128 : 2176]
//          byte (kb, oc, kk, p), bits 2j..2j+1 = code(row 8j + p, k = 32 kb + 8 oc + kk)
//   (padded to 2560 until 2026-10-03; the stride only matters past the first chunk of an element)
//
// The band law is gemv_q4's (64-row bands, half = c % rs, k-tile = c / rs). The inner
// product is gemv_q4's mmul<4, 8, 8, int16, uint8> with B = one 64 B block masked to one
// 2-bit field: field j is rows 8j..8j+7 at value code * 4^j (<= 128), 4^-j folded into the
// scale. Two fields per pass keep two mmul accumulators live, as gemv_q4's even / odd pair.
//
// The activation table is per 128 K -- the weight scale's own group -- so a group's int32
// products accumulate in the mmul registers and the float epilogue runs twice a chunk
// (per 32 K it was eight times and the kernel was core-bound: Peano RecMII 61 from the four
// dependent fp macs on the band accumulator).
//   table: int16 xi[K] | int32 s[K/128] | bf16 xs_hi[K/128] | bf16 xs_lo[K/128]
// Range: |xi| < 2^15, field value <= 128, 128 terms -> |sum| < 2^29.
// Precision: x is exact to int16 within 2^7 of its 128-block max (gemv_tab.h's rule at 4x
// the span); the min term is -s times the exact block sum.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gemv_tab.h"   // gemv_q4_absmask

#ifndef GEMV_T2_CHUNK
#define GEMV_T2_CHUNK 2176
#endif
#ifndef GEMV_PER_CALL
#define GEMV_PER_CALL 2
#endif
// Octets of a group's K the inner loop is unrolled by (after the first, which starts the
// accumulators). 15 = fully unrolled -- the fastest form, at its DMA floor in probe 0a;
// smaller counts trade program memory for loop overhead (the 27B's lx main core is full).
#ifndef GEMV_T2_OUNROLL
#define GEMV_T2_OUNROLL 15
#endif
#define GEMV_T2_STR_(x) #x
#define GEMV_T2_PRAGMA_(x) _Pragma(GEMV_T2_STR_(x))
#define GEMV_T2_UNROLL_(n) GEMV_T2_PRAGMA_(clang loop unroll_count(n))
static constexpr unsigned kT2Rows = 32;
static constexpr unsigned kT2TileK = 256;
static constexpr unsigned kT2ScaleBytes = 128;
static constexpr unsigned kT2Chunk = GEMV_T2_CHUNK;
static constexpr unsigned kT2PerCall = GEMV_PER_CALL;

static constexpr unsigned gemv_t2_tab_bytes(unsigned K) { return 2 * K + K / 32 + K / 32; }

// [1 x8 | 1/4 x8 | 1/16 x8 | 1/64 x8] as bf16 (32-lane broadcasts + selects: no 128-bit ops).
static inline aie::vector<bfloat16, kT2Rows> gemv_t2_field_scale() {
  aie::vector<int16_t, kT2Rows> v = aie::broadcast<int16_t, kT2Rows>((int16_t)0x3F80);
  v = aie::select(v, aie::broadcast<int16_t, kT2Rows>((int16_t)0x3E80), aie::mask<kT2Rows>::from_uint32(0x0000FF00u));
  v = aie::select(v, aie::broadcast<int16_t, kT2Rows>((int16_t)0x3D80), aie::mask<kT2Rows>::from_uint32(0x00FF0000u));
  v = aie::select(v, aie::broadcast<int16_t, kT2Rows>((int16_t)0x3C80), aie::mask<kT2Rows>::from_uint32(0xFF000000u));
  return v.template cast_to<bfloat16>();
}

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

// Groups g0 .. g0+ng-1 of a K-long bf16 activation into the per-128 table (x points at
// group g0). Integer scalar work only; the float work is vector ops.
__attribute__((noinline)) inline void gemv_t2_prep_groups(const bfloat16 *__restrict x, uint8_t *__restrict tab,
                                                          unsigned K, unsigned g0, unsigned ng) {
  const unsigned NG = K / 128;
  aie::set_rounding(aie::rounding_mode::conv_even);
  int16_t *__restrict xi = (int16_t *)tab;
  int32_t *__restrict sh = (int32_t *)(tab + 2 * K);
  bfloat16 *__restrict xsh = (bfloat16 *)(tab + 2 * K + 4 * NG);
  bfloat16 *__restrict xsl = xsh + NG;
  const aie::vector<uint8_t, 64> absmask = gemv_q4_absmask();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < ng; ++j) {
    const unsigned g = g0 + j;
    const bfloat16 *__restrict xb = x + j * 128;
    aie::vector<int16_t, 32> mx = aie::zeros<int16_t, 32>();
#pragma clang loop unroll(disable)   // program memory: the lx main core is full
    for (unsigned i = 0; i < 4; ++i)
      mx = aie::max(mx, aie::bit_and(aie::load_v<32>(xb + 32 * i).template cast_to<uint8_t>(), absmask)
                            .template cast_to<int16_t>());
    int s = 141 - (aie::reduce_max(mx) >> 7);    // 14 - (e - 127): positive floats order as their bits
    if (s > 126) s = 126;                        // zero / tiny block: any scale works
    sh[g] = s;
    const aie::vector<bfloat16, 32> sc =
        aie::broadcast<int16_t, 32>((int16_t)((127 + s) << 7)).template cast_to<bfloat16>();
    aie::accum<accfloat, 32> sum = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)   // program memory: the lx main core is full
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

// One chunk against k-tile `kt` of the table; `first` starts the band accumulator. The
// rows land in order, so `last` needs no zip. GEMV_NULL: DMA-only probe.
__attribute__((noinline)) inline void gemv_t2_tile(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                   unsigned K, unsigned kt, bool first, float *__restrict y) {
  event0();
#ifdef GEMV_NULL
  if (first)
    aie::store_v(y, aie::zeros<float, kT2Rows>());
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
#if GEMV_T2_OUNROLL >= 15
#pragma clang loop unroll(full)
      for (unsigned o = 0; o < 16; ++o) {                            // the group's 16 octets of K
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
#else
      {                                                              // octet 0 starts the accumulators
        const aie::vector<int16_t, 32> A = aie::load_v<8>(xg).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(cg);
        Ce.mul(A, aie::bit_and(me, q));
        Co.mul(A, aie::bit_and(mo, q));
      }
      GEMV_T2_UNROLL_(GEMV_T2_OUNROLL)
      for (unsigned o = 1; o < 16; ++o) {                            // the group's other 15 octets
        const aie::vector<int16_t, 32> A = aie::load_v<8>(xg + o * 8).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(cg + o * 64);
        Ce.mac(A, aie::bit_and(me, q));
        Co.mac(A, aie::bit_and(mo, q));
      }
#endif
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
    acc = aie::msc(acc, s32, xsh[g]);   // m = -s (no bf16 negate on this backend: G_FNEG)
    acc = aie::msc(acc, s32, xsl[g]);
  }
  aie::store_v(y, acc.template to_vector<float>());
  event1();
#endif
}

// ONE chunk of a w element (two per element), the band law at runtime (rs = 2: half = c % 2,
// k-tile = c / 2). `pbs` = per_band | sub << 16, sub the chunk within the element: the core
// calls the entry twice per element. One chunk per call keeps the core program's GEMV loops at
// the q4_1 27B's size -- with both chunks in one call their trip counts halved (40 -> 20, ...),
// LLVM fully unrolled them, and the lx main core's control program grew 6352 -> 9248 B, over
// its 16 KB. y is the band's 64-float accumulator.
__attribute__((noinline)) inline void gemv_t2_chunk_rt(const uint8_t *__restrict elem, const uint8_t *__restrict tab,
                                                       unsigned group, unsigned pbs, float *__restrict y, unsigned rs) {
  const unsigned sub = pbs >> 16, per_band = pbs & 0xFFFFu;
  const unsigned sh = (rs == 4) ? 2u : 1u;
  const unsigned K = (kT2TileK * per_band) >> sh;
  const unsigned c = group * kT2PerCall + sub;
  const unsigned kt = c >> sh;
  gemv_t2_tile(elem + sub * kT2Chunk, tab, K, kt, kt == 0, y + (c & (rs - 1)) * kT2Rows);
}
