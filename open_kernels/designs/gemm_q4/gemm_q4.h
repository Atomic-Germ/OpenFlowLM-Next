#pragma once
//===- gemm_q4.h -------------------------------------------*- C++ -*-===//
//
// Batched form of gemv_q4: Y[M, N] = W[N, K] @ X[M, K], same pool-order chunks,
// same verified tile body. Each of the M activations gets its own table and its
// own 32-float slice of the band accumulator; the weight chunk is loaded once
// and reused across all M.
//
// The point is to find where the core stops keeping up with the weight stream.
// GEMV is DMA-bound (qkv: 0.50 ms against a 0.45 ms DMA floor), so there is
// compute headroom under the stream; every token M reuses the same bytes, so
// prefill throughput is M x decode throughput right up until the core saturates.
// Sweeping M says where that is.
//
// Two dataflows share the tile body. The naive one finishes a band before
// starting the next, so it needs the whole K table per token (4608 B) and M
// caps out at 8. The blocked one at the bottom of this file keeps every band's
// accumulator resident and streams k-tiles through, so the table is one 256-wide
// slice per token (576 B) and M reaches 24.

#include "../gemv_q4/gemv_q4.h"

#ifndef GEMM_BATCH
#define GEMM_BATCH 1
#endif

static constexpr unsigned kBatch = GEMM_BATCH;

#ifdef GEMM_Q4K
// GEMM_Q4K narrows each 32-block's int32 partial to int16 so the 6-bit
// sub-scale can be applied with an int16 x int16 mac (aie2p has no int32 x
// int32 elementwise mac, and the int32 x int16 form only lands in acc64, which
// at four live 32-lane accumulators is twice what the register file holds).
// |partial| <= 32 x 15 (nibble) x 2^15 (|xi|) < 2^23.91, so a 9-bit SRS is the
// smallest that cannot saturate int16; the odd rows carry the nibble's 16x and
// take 13, which is where their 1/16 goes. Fifteen bits of the partial survive.
static constexpr unsigned kQ4KSrs = 9;
#endif

// int16 xi[K] | int32 s[K/32] | bf16 xs_hi[K/32] | bf16 xs_lo[K/32]
static constexpr unsigned gemm_q4_tab_bytes(unsigned K) { return 2 * K + K / 4; }

// Four tokens through one mmul. The matrix unit is <4, 8, 8>, and the GEMV form
// replicates a single activation across all four A rows, so three quarters of
// every issue is thrown away. Here the four rows carry four different tokens:
// same weight loads, same nibble masks, same mmul count, four times the work.
// The weight-side epilogue (d/m unpack, the permute, the scale) is also shared;
// only the int32 -> float conversion and the four MACs are per token.
__attribute__((noinline)) inline void gemm_q4_tile4(const uint8_t *__restrict tile,
                                                    const uint8_t *__restrict tab0,
                                                    unsigned tabStride, unsigned K,
                                                    unsigned kt, bool first, bool last,
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

    // weight side: computed once, used by all four tokens
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
      acc[t] = aie::mac(acc[t], hi, ds);
#ifndef GEMM_FAST_EPILOGUE
      // fp32 needs two bf16 halves; dropping the low one costs ~8 bits on each
      // block's contribution and saves four of the nine epilogue ops per token
      const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], lo, ds);
      acc[t] = aie::mac(acc[t], mperm, xsl[t][kb]);
#endif
      acc[t] = aie::mac(acc[t], mperm, xsh[t][kb]);
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

// One call = kPerCall consecutive pool-order chunks of one band, for all M
// activations. y holds M consecutive band accumulators.
static inline void gemm_q4_pool_group(const uint8_t *__restrict chunks,
                                      const uint8_t *__restrict tab,
                                      unsigned group, float *__restrict y) {
  constexpr unsigned kKt = kPerBand / kRowSplit;          // k-tiles per band
  constexpr unsigned kBandAcc = kRows * kRowSplit;        // floats per band per token
  constexpr unsigned kTabBytes = gemm_q4_tab_bytes(kK);

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kPerCall; ++i) {
    const unsigned c = group * kPerCall + i;
    const unsigned part = c % kRowSplit;
    const unsigned kt = c / kRowSplit;
    const uint8_t *__restrict tile = chunks + i * kTileBytes;

#if GEMM_BATCH % 4 == 0
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; m += 4) {
      gemm_q4_tile4(tile, tab + m * kTabBytes, kTabBytes, kK, kt, kt == 0, kt == kKt - 1,
                    y + m * kBandAcc + part * kRows, kBandAcc);
    }
#else
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; ++m) {
      gemv_q4_tile(tile, tab + m * kTabBytes, kK, kt, kt == 0, kt == kKt - 1,
                   y + m * kBandAcc + part * kRows);
    }
#endif
  }
}

#define GEMM_Q4_ENTRY__(P, B, R, M, N)                                       \
  void gemm_q4_p##P##b##B##r##R##m##M##_k##N(const uint8_t *__restrict t,    \
                                             const uint8_t *__restrict tab,  \
                                             float *__restrict y) {          \
    gemm_q4_pool_group(t, tab, N, y);                                        \
  }
#define GEMM_Q4_ENTRY_(P, B, R, M, N) GEMM_Q4_ENTRY__(P, B, R, M, N)
#define GEMM_Q4_ENTRY(N) GEMM_Q4_ENTRY_(GEMV_PER_CALL, GEMV_PER_BAND, GEMV_ROWSPLIT, GEMM_BATCH, N)

// Block-quantise all M activations into M consecutive tables.
// One token per call: the host streams the M activations through a single
// K-wide fifo buffer rather than parking all M in L1, which is what caps M.
#define GEMM_Q4_PREP_ENTRY__(K, M)                                     \
  void gemm_q4_prep_k##K##m##M(const bfloat16 *__restrict x,           \
                               uint8_t *__restrict tab, int32_t m) {   \
    gemv_q4_prep(x, tab + (unsigned)m * gemm_q4_tab_bytes(K), K);      \
  }
#define GEMM_Q4_PREP_ENTRY_(K, M) GEMM_Q4_PREP_ENTRY__(K, M)
#define GEMM_Q4_PREP_ENTRY(K) GEMM_Q4_PREP_ENTRY_(K, GEMM_BATCH)

// ---------------------------------------------------------------------------
// Token-interleaved activation table (GEMM_TAB_IL).
//
// The mmul's A operand is 4 tokens x 8 k, row-major, so with one xi[K] per
// token it costs four 8-wide loads and a concat to assemble - 19% of the
// per-token op count. Storing the k-octets interleaved across the group of
// four, xi[octet][token][8], makes it a single 32-wide load. Same bytes, same
// arithmetic, bit-identical results.
//
// Group layout, K = 256 slice, 4 tokens (2304 B, the same 576 B per token):
//   int16 xi[32][4][8]   at 0      (octet = k/8)
//   int32 s[4][8]        at 2048
//   bf16  xs_hi[4][8]    at 2176
//   bf16  xs_lo[4][8]    at 2240
// ---------------------------------------------------------------------------

static constexpr unsigned kILGroup = 4;                              // tokens per mmul
static constexpr unsigned kILOctets = kTileK / 8;                    // 32
static constexpr unsigned kILSOff = kILOctets * kILGroup * 8 * 2;    // 2048
static constexpr unsigned kILXSHOff = kILSOff + kILGroup * kKBlocks * 4;
static constexpr unsigned kILXSLOff = kILXSHOff + kILGroup * kKBlocks * 2;
static constexpr unsigned kILGroupBytes = kILGroup * gemm_q4_tab_bytes(kTileK);

// ---------------------------------------------------------------------------
// GEMM_PREQ: stream an already-quantised IL group table from the host instead
// of raw bf16 activations, so the core spends nothing on gemm_q4_prep_il's
// arithmetic (reduce_max, the int16 round-and-scale, the block-sum tree) - the
// bytes it produces are identical to what the host now sends, so this is a
// straight DMA-into-L1 copy. gemm_q4.py's host-side prep (make_q4k.py's
// gemm_q4_prep_il.py, or wherever the Python mirror lives) must reproduce
// gemm_q4_prep_il's math exactly, including GEMM_Q4K's one-shift-per-256-slice
// variant if q4k is also set.
// ---------------------------------------------------------------------------

#ifdef GEMM_PREQ
__attribute__((noinline)) inline void gemm_q4_copy_group(const uint8_t *__restrict xe,
                                                         uint8_t *__restrict tab,
                                                         unsigned g) {
  uint8_t *__restrict dst = tab + g * kILGroupBytes;
#ifdef GEMM_COPY_NULL
  // TIMING PROBE: keep the acquire/release and the x DMA, drop the copy, so
  // the tile body runs on stale `tab` (results are garbage). Separates "the
  // copy costs" from "waiting on the x stream costs".
  (void)dst; (void)xe;
  return;
#endif
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kILGroupBytes; i += 64)
    aie::store_v(dst + i, aie::load_v<64>(xe + i));
}
#define GEMM_Q4_COPY_GROUP_ENTRY__(M)                                          \
  void gemm_q4_copy_group_m##M(const uint8_t *__restrict xe,                   \
                               uint8_t *__restrict tab, int32_t g) {           \
    gemm_q4_copy_group(xe, tab, (unsigned)g);                                  \
  }
#define GEMM_Q4_COPY_GROUP_ENTRY_(M) GEMM_Q4_COPY_GROUP_ENTRY__(M)
#define GEMM_Q4_COPY_GROUP_ENTRY() GEMM_Q4_COPY_GROUP_ENTRY_(GEMM_BATCH)
#endif

// One token's 256-wide slice into slot t of the group table.
__attribute__((noinline)) inline void gemm_q4_prep_il(const bfloat16 *__restrict x,
                                                      uint8_t *__restrict g, unsigned t) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  int16_t *__restrict xi = (int16_t *)g;
  int32_t *__restrict sh = (int32_t *)(g + kILSOff) + t * kKBlocks;
  bfloat16 *__restrict xsh = (bfloat16 *)(g + kILXSHOff) + t * kKBlocks;
  bfloat16 *__restrict xsl = (bfloat16 *)(g + kILXSLOff) + t * kKBlocks;
  const aie::vector<uint8_t, 64> absmask = gemv_q4_absmask();

#ifdef GEMM_Q4K
  // The kernel accumulates across the whole 256-wide superblock, so every
  // 32-block has to share one binary point: take the max over the slice once.
  aie::vector<int16_t, 32> amax = aie::zeros<int16_t, 32>();
#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    const aie::vector<bfloat16, 32> xv = aie::load_v<32>(x + kb * kKInBlock);
    amax = aie::max(amax, aie::bit_and(xv.template cast_to<uint8_t>(), absmask)
                              .template cast_to<int16_t>());
  }
  int sSlice = 141 - (aie::reduce_max(amax) >> 7);
  if (sSlice > 126) sSlice = 126;
  // the epilogue converts with (s - kQ4KSrs), which must not go negative;
  // clamping here costs precision only for |x| > 2^5, which a normalised
  // activation does not reach
  if (sSlice < (int)kQ4KSrs) sSlice = kQ4KSrs;
#endif

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    const aie::vector<bfloat16, 32> xv = aie::load_v<32>(x + kb * kKInBlock);
#ifdef GEMM_Q4K
    const int s = sSlice;
#else
    const aie::vector<int16_t, 32> ab =
        aie::bit_and(xv.template cast_to<uint8_t>(), absmask).template cast_to<int16_t>();
    const int mx = aie::reduce_max(ab);            // positive floats order as their bits
    int s = 141 - (mx >> 7);                       // 14 - (e - 127)
    if (s > 126) s = 126;
#endif
    sh[kb] = s;
    const aie::vector<bfloat16, 32> sc =
        aie::broadcast<int16_t, 32>((int16_t)((127 + s) << 7)).template cast_to<bfloat16>();
    const aie::vector<int16_t, 32> q = aie::to_fixed<int16_t>(aie::mul(xv, sc), 0);
    // the block's four k-octets land in four different octet slots
#pragma clang loop unroll(full)
    for (unsigned j = 0; j < 4; ++j)
      aie::store_v(xi + (kb * 4 + j) * kILGroup * 8 + t * 8, q.template extract<8>(j));

    aie::vector<bfloat16, 32> hi, lo;
    block_sum_split(x + kb * kKInBlock, hi, lo);
    xsh[kb] = hi[0];
    xsl[kb] = lo[0];
  }
}

// gemm_q4_tile4 against an interleaved group table: one A load per (kb, oc).
__attribute__((noinline)) inline void gemm_q4_tile4_il(const uint8_t *__restrict tile,
                                                       const uint8_t *__restrict g,
                                                       bool first, bool last,
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

  const bfloat16 *__restrict dp = (const bfloat16 *)tile;
  const bfloat16 *__restrict mp = (const bfloat16 *)(tile + kDBytes);
  const uint8_t *__restrict nib0 = tile + kMetaBytes;
  const uint8_t *__restrict nib1 = tile + kMetaBytes + 2048;

  const int16_t *__restrict xi = (const int16_t *)g;
  const int32_t *__restrict sh = (const int32_t *)(g + kILSOff);
  const bfloat16 *__restrict xsh = (const bfloat16 *)(g + kILXSHOff);
  const bfloat16 *__restrict xsl = (const bfloat16 *)(g + kILXSLOff);

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
        const aie::vector<int16_t, 32> A =
            aie::load_v<32>(xi + (kb * 4 + oc) * kILGroup * 8);
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
      part.from_vector(aie::to_float<float>(vi, sh[t * kKBlocks + kb]));
      const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], hi, ds);
#ifndef GEMM_FAST_EPILOGUE
      const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], lo, ds);
      acc[t] = aie::mac(acc[t], mperm, xsl[t * kKBlocks + kb]);
#endif
      acc[t] = aie::mac(acc[t], mperm, xsh[t * kKBlocks + kb]);
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

// ---------------------------------------------------------------------------
// GEMM_Q4K: Q4_K's 6-bit sub-scales folded into the int32 accumulation.
//
// A chunk is 32 rows x 256 K, which is EXACTLY one Q4_K superblock per row, so
// Q4_K's own structure is already the chunk's:
//
//   value(r, k) = d[r] * sc[kb][r] * nib(r, k)  +  mcoef[r] * mn[kb][r]
//
// with sc and mn 6-bit integers and d / mcoef one bf16 pair per row per
// superblock (mcoef = -dmin, so both terms are added).
//
// Q4_1's epilogue - four extracts of the mmul's row groups, a concat, to_float
// with the block's binary point, a bf16 narrowing, two macs - runs once per
// 32-block because every block carries its own bf16 scale. Here the scale is an
// INTEGER, so it can be applied to the partial and the accumulation stays
// integer all the way across the superblock:
//
//   y[r] += d[r] * 2^(kQ4KSrs - s) * SUM_kb { sc[kb][r] * partial[kb][r] }
//
// One float epilogue per 256 K instead of eight, and the d/m unpack with it.
// Per 32-block the whole cost is two SRS and two macs, against nine vector ops.
// The matrix unit is untouched (still mmul<4, 8, 8, int16, uint8> at 256 MACs
// per issue). The plan's other reading of "fold into the nibbles" - expand them
// to int16 on the core and let the mmul eat sc*nib - would instead need the
// dense int16 x int16 shape <4, 4, 8> at half that rate (the 256-MAC <4, 8, 8>
// int16 x int16 is sparse-B only) plus a 16 KB expanded chunk resident in L1.
//
// TWO CONSEQUENCES, both paid for in the activation prep:
//
//  - the partial is narrowed to int16 by a kQ4KSrs-bit SRS before the scale is
//    applied, keeping 15 of its ~22 bits. That, not the activation, is this
//    kernel's dominant rounding.
//  - one binary point per superblock: the eight 32-blocks can no longer carry
//    their own shift (the mmul's C block is shared by four tokens, so the SRS
//    cannot be per token), so the activation is block-quantised at 256 - one
//    reduce_max over the slice - rather than at 32.
//
// The odd-row product carries the nibble's 16x (B = q & 0xF0); it is exactly
// divisible by 16, so its SRS is four bits deeper and d needs no 1/16 half
// (no dsc, unlike every other variant here).
//
// CHUNK LAYOUT (4736 B, against q4_1's 5120 - Q4_K's metadata is smaller):
//   bf16  d    [32]     at    0   permuted rows [evens 0,2..30 | odds 1,3..31]
//   bf16  mcoef[32]     at   64   ditto, = -dmin
//   uint8 sc   [8][32]  at  128   per k-block, same row permutation
//   uint8 mn   [8][32]  at  384   ditto
//   uint8 nib  [4096]   at  640   byte-identical to the q4_1 chunk's nibbles
//
// The row permutation is the one the mmul's lane order already wants (see the
// lane-order note in gemv_q4.h), so the core loads d, sc and mn straight - no
// interleave_unzip, no concat. A backend's load-time repack does it once.
// ---------------------------------------------------------------------------

#ifdef GEMM_Q4K
static constexpr unsigned kQ4KDOff = 0;
static constexpr unsigned kQ4KMcOff = 64;
static constexpr unsigned kQ4KScOff = 128;
static constexpr unsigned kQ4KMnOff = 384;
static constexpr unsigned kQ4KNibOff = 640;
static constexpr unsigned kQ4KTileBytes = kQ4KNibOff + 4096;      // 4736

// Per-chunk scratch, built once and reused by every token group:
//   int16 scv[8][4][32]  at 0     2048 B  sub-scale, 8-row group replicated x4
//   bf16  mnv[8][32]     at 2048   512 B  mn[kb][r] as bf16 (an integer 0..63,
//                                         so exact - see the min note below)
// scv's four groups per k-block are (rb 0 even, rb 1 even, rb 0 odd, rb 1 odd),
// the order the 32 permuted rows already lie in; replicated x4 because the
// mmul's C block is [token][8 rows] and the scale depends only on the row.
static constexpr unsigned kQ4KScvBytes = kKBlocks * 4 * kRows * 2;   // 2048
static constexpr unsigned kQ4KScrBytes = kQ4KScvBytes + kKBlocks * kRows * 2;

__attribute__((noinline)) inline void gemm_q4k_prep_chunk(const uint8_t *__restrict tile,
                                                          uint8_t *__restrict scr) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  int16_t *__restrict scv = (int16_t *)scr;
  bfloat16 *__restrict mnv = (bfloat16 *)(scr + kQ4KScvBytes);

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    const aie::vector<int16_t, kRows> s16 =
        aie::load_v<kRows>(tile + kQ4KScOff + kb * kRows)
            .unpack().template cast_to<int16_t>();
#pragma clang loop unroll(full)
    for (unsigned g = 0; g < 4; ++g)
      aie::store_v(scv + (kb * 4 + g) * kRows,
                   s16.template extract<8>(g).template grow_replicate<kRows>());

    const aie::vector<int32_t, kRows> n32 =
        aie::load_v<kRows>(tile + kQ4KMnOff + kb * kRows)
            .unpack().unpack().template cast_to<int32_t>();
    aie::accum<accfloat, kRows> na;
    na.from_vector(aie::to_float<float>(n32, 0));
    aie::store_v(mnv + kb * kRows, na.template to_vector<bfloat16>());
  }
}

// Four tokens, one chunk, against the scratch prep_chunk left. The mmul loop is
// gemm_q4_tile4_il's unchanged; what differs is that nothing leaves int32 until
// the superblock is finished.
__attribute__((noinline)) inline void gemm_q4k_tile4(const uint8_t *__restrict tile,
                                                     const uint8_t *__restrict scr,
                                                     const uint8_t *__restrict g,
                                                     bool first, bool last,
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

  const uint8_t *__restrict nib0 = tile + kQ4KNibOff;
  const uint8_t *__restrict nib1 = tile + kQ4KNibOff + 2048;
  const int16_t *__restrict scv = (const int16_t *)scr;
  const bfloat16 *__restrict mnv = (const bfloat16 *)(scr + kQ4KScvBytes);

  const int16_t *__restrict xi = (const int16_t *)g;
  const int32_t *__restrict sh = (const int32_t *)(g + kILSOff);
  const bfloat16 *__restrict xsh = (const bfloat16 *)(g + kILXSHOff);
  const bfloat16 *__restrict xsl = (const bfloat16 *)(g + kILXSLOff);

  // the only live state across the k loop: 4 x 32 lanes of int32, one group per
  // (row half, row parity), each holding 4 tokens x 8 rows
  aie::accum<acc32, kRows> ai[4];
#pragma clang loop unroll(full)
  for (unsigned i = 0; i < 4; ++i) ai[i] = aie::zeros<acc32, kRows>();

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? nib0 : nib1) + kb * 256;
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int16_t, 32> A =
            aie::load_v<32>(xi + (kb * 4 + oc) * kILGroup * 8);
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
      // narrow to int16 (the odd rows four bits deeper: their product is
      // exactly 16x the nibble's value), then apply the 6-bit sub-scale
      const aie::vector<int16_t, kRows> Cev = Ce.template to_vector<int16_t>(kQ4KSrs);
      const aie::vector<int16_t, kRows> Cov = Co.template to_vector<int16_t>(kQ4KSrs + 4);
      ai[rb] = aie::mac(ai[rb], Cev, aie::load_v<kRows>(scv + (kb * 4 + rb) * kRows));
      ai[2 + rb] =
          aie::mac(ai[2 + rb], Cov, aie::load_v<kRows>(scv + (kb * 4 + 2 + rb) * kRows));
    }
  }

  const aie::vector<bfloat16, kRows> ds =
      aie::load_v<kRows>((const bfloat16 *)(tile + kQ4KDOff));
  const aie::vector<bfloat16, kRows> mc =
      aie::load_v<kRows>((const bfloat16 *)(tile + kQ4KMcOff));
  const aie::vector<int32_t, kRows> a0 = ai[0].template to_vector<int32_t>();
  const aie::vector<int32_t, kRows> a1 = ai[1].template to_vector<int32_t>();
  const aie::vector<int32_t, kRows> a2 = ai[2].template to_vector<int32_t>();
  const aie::vector<int32_t, kRows> a3 = ai[3].template to_vector<int32_t>();

#pragma clang loop unroll(full)
  for (unsigned t = 0; t < 4; ++t) {
    aie::accum<accfloat, kRows> acc;
    if (first)
      acc = aie::zeros<accfloat, kRows>();
    else
      acc.from_vector(aie::load_v<kRows>(y + t * yStride));

    const aie::vector<int32_t, kRows> vi =
        aie::concat(a0.template extract<8>(t), a1.template extract<8>(t),
                    a2.template extract<8>(t), a3.template extract<8>(t));
    aie::accum<accfloat, kRows> part;
    part.from_vector(aie::to_float<float>(vi, sh[t * kKBlocks] - (int)kQ4KSrs));
    const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
    acc = aie::mac(acc, hi, ds);
#ifndef GEMM_FAST_EPILOGUE
    const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    acc = aie::mac(acc, lo, ds);
#endif
    // The min side still touches every 32-block - mn is an integer but the block
    // sum it multiplies is not. It is summed in fp32 FIRST and scaled by mcoef
    // once, because bf16(mcoef * mn) per block loses 2^-9 on a term that is a
    // large part of the value: measured, that one rounding was 3.7e-3 of 5.0e-3
    // total maxrel. mn as bf16 is exact (an integer below 256), so the sum is.
    aie::accum<accfloat, kRows> tacc = aie::zeros<accfloat, kRows>();
#pragma clang loop unroll(full)
    for (unsigned kb = 0; kb < kKBlocks; ++kb) {
      const aie::vector<bfloat16, kRows> mv = aie::load_v<kRows>(mnv + kb * kRows);
      tacc = aie::mac(tacc, mv, xsh[t * kKBlocks + kb]);
#ifndef GEMM_FAST_EPILOGUE
      tacc = aie::mac(tacc, mv, xsl[t * kKBlocks + kb]);
#endif
    }
    const aie::vector<bfloat16, kRows> th = tacc.template to_vector<bfloat16>();
    acc = aie::mac(acc, th, mc);
#ifndef GEMM_FAST_EPILOGUE
    acc = aie::mac(acc, aie::sub(tacc, th).template to_vector<bfloat16>(), mc);
#endif

    const aie::vector<float, kRows> yv = acc.template to_vector<float>();
    float *__restrict yt = y + t * yStride;
    if (last) {
      auto [r0, r1] = aie::interleave_zip(yv.template extract<16>(0),
                                          yv.template extract<16>(1), 1);
      aie::store_v(yt, r0);
      aie::store_v(yt + 16, r1);
    } else {
      aie::store_v(yt, yv);
    }
  }
  event1();
#endif
}
#endif  // GEMM_Q4K

// ---------------------------------------------------------------------------
// GEMM_G128: what a group-128 weight format would cost (TIMING PROBE ONLY).
//
// Q4_0, Q4_1 and Q4_K all carry a scale every 32 weights, so the epilogue -
// read the mmul's int32 C block, convert with the block's binary point, narrow
// to bf16, two macs - runs eight times per 256-wide chunk. AWQ and GPTQ at
// group 128 would run it twice: the matrix unit can accumulate int32 across
// 128 K (worst case 32768 x 240 x 128 = 1.0e9, inside int32) before anything
// touches float.
//
// This variant does exactly that, reusing sub-block 4*kg's d/m for all four
// sub-blocks. The ARITHMETIC IS WRONG - it is here to measure the shape, not
// to compute anything.
// ---------------------------------------------------------------------------

#ifdef GEMM_G128
#define GEMM_KG 4                               // 32-blocks per scale group
static constexpr unsigned kKGroups = kKBlocks / GEMM_KG;

template <unsigned TOK, typename ATYPE>
__attribute__((noinline)) inline void gemm_q4_g128_body(const uint8_t *__restrict tile,
                                                        const ATYPE *__restrict xi,
                                                        const int32_t *__restrict sh,
                                                        const bfloat16 *__restrict xsh,
                                                        unsigned stride, bool first, bool last,
                                                        float *__restrict y, unsigned yStride) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict dp = (const bfloat16 *)tile;
  const bfloat16 *__restrict mp = (const bfloat16 *)(tile + kDBytes);
  const uint8_t *__restrict nib0 = tile + kMetaBytes;
  const uint8_t *__restrict nib1 = tile + kMetaBytes + 2048;

  aie::accum<accfloat, kRows> acc[TOK];
#pragma clang loop unroll(full)
  for (unsigned t = 0; t < TOK; ++t) {
    if (first)
      acc[t] = aie::zeros<accfloat, kRows>();
    else
      acc[t].from_vector(aie::load_v<kRows>(y + t * yStride));
  }

  const aie::vector<bfloat16, kRows> dsc =
      aie::concat(aie::broadcast<int16_t, 16>((int16_t)0x3F80),
                  aie::broadcast<int16_t, 16>((int16_t)0x3D80)).template cast_to<bfloat16>();

#pragma clang loop unroll(disable)
  for (unsigned kg = 0; kg < kKGroups; ++kg) {
    const unsigned kb0 = kg * GEMM_KG;
    aie::vector<int32_t, 8> ve[TOK][2], vo[TOK][2];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict base = (rb == 0 ? nib0 : nib1) + kb0 * 256;
      const ATYPE *__restrict xa = xi + kb0 * 4 * TOK * 8;
      aie::mmul<TOK, 8, 8, ATYPE, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned j = 0; j < GEMM_KG; ++j) {
#pragma clang loop unroll(full)
        for (unsigned oc = 0; oc < 4; ++oc) {
          const aie::vector<ATYPE, TOK * 8> A =
              aie::load_v<TOK * 8>(xa + (j * 4 + oc) * TOK * 8);
          const aie::vector<uint8_t, 64> q = aie::load_v<64>(base + j * 256 + oc * 64);
          const aie::vector<uint8_t, 64> e = aie::bit_and((uint8_t)0x0F, q);
          const aie::vector<uint8_t, 64> o = aie::bit_and((uint8_t)0xF0, q);
          if (j == 0 && oc == 0) {
            Ce.mul(A, e);
            Co.mul(A, o);
          } else {
            Ce.mac(A, e);
            Co.mac(A, o);
          }
        }
      }
      const aie::vector<int32_t, TOK * 8> Cev = Ce.template to_vector<int32_t>();
      const aie::vector<int32_t, TOK * 8> Cov = Co.template to_vector<int32_t>();
#pragma clang loop unroll(full)
      for (unsigned t = 0; t < TOK; ++t) {
        ve[t][rb] = Cev.template extract<8>(t);
        vo[t][rb] = Cov.template extract<8>(t);
      }
    }

    // one scale group's weight side, from the group's first 32-block
    const aie::vector<bfloat16, kRows> d32 = aie::load_v<kRows>(dp + kb0 * kRows);
    const aie::vector<bfloat16, kRows> m32 = aie::load_v<kRows>(mp + kb0 * kRows);
    auto [de, dod] = aie::interleave_unzip(d32, d32, 1);
    auto [me, mo] = aie::interleave_unzip(m32, m32, 1);
    const aie::vector<bfloat16, kRows> dperm =
        aie::concat(de.template extract<16>(0), dod.template extract<16>(0));
    const aie::vector<bfloat16, kRows> mperm =
        aie::concat(me.template extract<16>(0), mo.template extract<16>(0));
    const aie::vector<bfloat16, kRows> ds = aie::mul(dperm, dsc).template to_vector<bfloat16>();

#pragma clang loop unroll(full)
    for (unsigned t = 0; t < TOK; ++t) {
      const aie::vector<int32_t, kRows> vi = aie::concat(ve[t][0], ve[t][1], vo[t][0], vo[t][1]);
      aie::accum<accfloat, kRows> part;
      part.from_vector(aie::to_float<float>(vi, sh[t * stride + kb0]));
      acc[t] = aie::mac(acc[t], part.template to_vector<bfloat16>(), ds);
      acc[t] = aie::mac(acc[t], mperm, xsh[t * stride + kb0]);
    }
  }

#pragma clang loop unroll(full)
  for (unsigned t = 0; t < TOK; ++t) {
    const aie::vector<float, kRows> yv = acc[t].template to_vector<float>();
    float *__restrict yt = y + t * yStride;
    if (last) {
      auto [r0, r1] = aie::interleave_zip(yv.template extract<16>(0),
                                          yv.template extract<16>(1), 1);
      aie::store_v(yt, r0);
      aie::store_v(yt + 16, r1);
    } else {
      aie::store_v(yt, yv);
    }
  }
  event1();
}

static inline void gemm_q4_g128_il(const uint8_t *__restrict tile, const uint8_t *__restrict g,
                                   bool first, bool last, float *__restrict y, unsigned yStride) {
  gemm_q4_g128_body<4, int16_t>(tile, (const int16_t *)g, (const int32_t *)(g + kILSOff),
                                (const bfloat16 *)(g + kILXSHOff), kKBlocks,
                                first, last, y, yStride);
}
#endif

// ---------------------------------------------------------------------------
// int8 activations (GEMM_INT8).
//
// mmul<8, 8, 8, int8, uint8> is dense and does 512 MACs per issue against the
// int16 form's 256, so eight tokens ride one issue instead of four: the mmul,
// the nibble masks, the A loads and the weight-side epilogue are all halved per
// token. The nibbles stay uint8 exactly as they lie in the chunk. The cost is
// the activation: 7 bits of block-relative precision instead of 15.
//
// Group layout, K = 256 slice, 8 tokens (2560 B, 320 B per token against the
// int16 form's 576):
//   int8  xi[32][8][8]   at 0      (octet = k/8, then token, then k)
//   int32 s[8][8]        at 2048
//   bf16  xs_hi[8][8]    at 2304
//   bf16  xs_lo[8][8]    at 2432
// ---------------------------------------------------------------------------

static constexpr unsigned k8Group = 8;                               // tokens per mmul
static constexpr unsigned k8SOff = kILOctets * k8Group * 8;          // 2048
static constexpr unsigned k8XSHOff = k8SOff + k8Group * kKBlocks * 4;
static constexpr unsigned k8XSLOff = k8XSHOff + k8Group * kKBlocks * 2;
static constexpr unsigned k8GroupBytes = k8XSLOff + k8Group * kKBlocks * 2;
static constexpr unsigned k8TabBytes = k8GroupBytes / k8Group;       // 320

// Two tokens' 256-wide slices into slots 2p and 2p+1 of the group table. Same
// block scheme as the int16 prep with the binary point 8 bits lower
// (xi_max = 2^6 x mantissa), and a pair at a time because the smallest int8
// vector the hardware stores is 16 lanes - one token's k-octet is only 8.
// interleave_zip at granularity 8 lays the pair out as the table wants it:
// concat(lo, hi) = [a0:8 b0:8 a8:16 b8:16 a16:24 b16:24 a24:32 b24:32], i.e.
// the pair's four k-octets back to back.
__attribute__((noinline)) inline void gemm_q4_prep_i8(const bfloat16 *__restrict x2,
                                                      uint8_t *__restrict g, unsigned p) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  int8_t *__restrict xi = (int8_t *)g + p * 16;
  int32_t *__restrict sh = (int32_t *)(g + k8SOff) + 2 * p * kKBlocks;
  bfloat16 *__restrict xsh = (bfloat16 *)(g + k8XSHOff) + 2 * p * kKBlocks;
  bfloat16 *__restrict xsl = (bfloat16 *)(g + k8XSLOff) + 2 * p * kKBlocks;
  const aie::vector<uint8_t, 64> absmask = gemv_q4_absmask();

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int8_t, 32> q[2];
#pragma clang loop unroll(full)
    for (unsigned t = 0; t < 2; ++t) {
      const bfloat16 *__restrict xb = x2 + t * kTileK + kb * kKInBlock;
      const aie::vector<bfloat16, 32> xv = aie::load_v<32>(xb);
      const aie::vector<int16_t, 32> ab =
          aie::bit_and(xv.template cast_to<uint8_t>(), absmask).template cast_to<int16_t>();
      const int mx = aie::reduce_max(ab);
      int s = 133 - (mx >> 7);                     // 6 - (e - 127)
      if (s > 126) s = 126;
      sh[t * kKBlocks + kb] = s;
      const aie::vector<bfloat16, 32> sc =
          aie::broadcast<int16_t, 32>((int16_t)((127 + s) << 7)).template cast_to<bfloat16>();
      q[t] = aie::to_fixed<int8_t>(aie::mul(xv, sc), 0);

      aie::vector<bfloat16, 32> hi, lo;
      block_sum_split(xb, hi, lo);
      xsh[t * kKBlocks + kb] = hi[0];
      xsl[t * kKBlocks + kb] = lo[0];
    }
    auto [lo2, hi2] = aie::interleave_zip(q[0], q[1], 8);
    aie::store_v(xi + (kb * 4 + 0) * k8Group * 8, lo2.template extract<16>(0));
    aie::store_v(xi + (kb * 4 + 1) * k8Group * 8, lo2.template extract<16>(1));
    aie::store_v(xi + (kb * 4 + 2) * k8Group * 8, hi2.template extract<16>(0));
    aie::store_v(xi + (kb * 4 + 3) * k8Group * 8, hi2.template extract<16>(1));
  }
}

// Eight tokens, one chunk. C is 8 tokens x 8 rows per product, so the epilogue
// shape is unchanged - only the mmul count halves.
__attribute__((noinline)) inline void gemm_q4_tile8_i8(const uint8_t *__restrict tile,
                                                       const uint8_t *__restrict g,
                                                       bool first, bool last,
                                                       float *__restrict y, unsigned yStride) {
  event0();
#ifdef GEMV_NULL
  if (first) {
#pragma clang loop unroll(full)
    for (unsigned t = 0; t < 8; ++t)
      aie::store_v(y + t * yStride, aie::zeros<float, kRows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict dp = (const bfloat16 *)tile;
  const bfloat16 *__restrict mp = (const bfloat16 *)(tile + kDBytes);
  const uint8_t *__restrict nib0 = tile + kMetaBytes;
  const uint8_t *__restrict nib1 = tile + kMetaBytes + 2048;

  const int8_t *__restrict xi = (const int8_t *)g;
  const int32_t *__restrict sh = (const int32_t *)(g + k8SOff);
  const bfloat16 *__restrict xsh = (const bfloat16 *)(g + k8XSHOff);
  const bfloat16 *__restrict xsl = (const bfloat16 *)(g + k8XSLOff);

  aie::accum<accfloat, kRows> acc[8];
#pragma clang loop unroll(full)
  for (unsigned t = 0; t < 8; ++t) {
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
    aie::vector<int32_t, 8> ve[8][2], vo[8][2];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? nib0 : nib1) + kb * 256;
      aie::mmul<8, 8, 8, int8_t, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int8_t, 64> A =
            aie::load_v<64>(xi + (kb * 4 + oc) * k8Group * 8);
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
      const aie::vector<int32_t, 64> Cev = Ce.template to_vector<int32_t>();
      const aie::vector<int32_t, 64> Cov = Co.template to_vector<int32_t>();
#pragma clang loop unroll(full)
      for (unsigned t = 0; t < 8; ++t) {
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
    for (unsigned t = 0; t < 8; ++t) {
      const aie::vector<int32_t, kRows> vi = aie::concat(ve[t][0], ve[t][1], vo[t][0], vo[t][1]);
      aie::accum<accfloat, kRows> part;
      part.from_vector(aie::to_float<float>(vi, sh[t * kKBlocks + kb]));
      const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], hi, ds);
#ifndef GEMM_FAST_EPILOGUE
      const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], lo, ds);
      acc[t] = aie::mac(acc[t], mperm, xsl[t * kKBlocks + kb]);
#endif
      acc[t] = aie::mac(acc[t], mperm, xsh[t * kKBlocks + kb]);
    }
  }

#pragma clang loop unroll(full)
  for (unsigned t = 0; t < 8; ++t) {
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

#ifdef GEMM_G128
static inline void gemm_q4_g128_i8(const uint8_t *__restrict tile, const uint8_t *__restrict g,
                                   bool first, bool last, float *__restrict y, unsigned yStride) {
  gemm_q4_g128_body<8, int8_t>(tile, (const int8_t *)g, (const int32_t *)(g + k8SOff),
                               (const bfloat16 *)(g + k8XSHOff), kKBlocks,
                               first, last, y, yStride);
}
#endif

// ---------------------------------------------------------------------------
// Blocked dataflow: the core keeps the accumulators for all its bands resident
// and the weights arrive k-tile-major (for kt: for band: for part), so a band's
// accumulator is revisited once per k-tile instead of being finished before the
// next band starts. The table is one 256-wide k-tile slice per token (576 B)
// instead of the whole K (4608 B), which is what lets M reach 24 in 64 KB of
// L1 - the naive form caps out at 8. The tile body is unchanged: it sees a
// K = 256 table at kt = 0.
// ---------------------------------------------------------------------------

static constexpr unsigned kSliceBytes = gemm_q4_tab_bytes(kTileK);

// One DMA element = kPerCall consecutive chunks of the core's (band, part) grid
// for the current k-tile; `idx` numbers the groups. Every token's accumulator
// for a chunk's band gets its contribution before the element is released.
static inline void gemm_q4_blk_group(const uint8_t *__restrict chunks,
                                     const uint8_t *__restrict tab,
                                     float *__restrict y, unsigned idx, unsigned kt) {
  constexpr unsigned kKt = kPerBand / kRowSplit;          // k-tiles per band
  constexpr unsigned kBandAcc = kRows * kRowSplit;        // floats per band per token
  const bool first = (kt == 0), last = (kt == kKt - 1);
#ifdef GEMM_Q4K
  constexpr unsigned kChunkBytes = kQ4KTileBytes;
  // The unpacked sub-scales are the same for every token, so they are built once
  // per chunk here rather than per four-token group inside the tile body. .bss
  // is not cleared on the core, but every byte is written before it is read.
  alignas(64) static uint8_t scr[kQ4KScrBytes];
#else
  constexpr unsigned kChunkBytes = kTileBytes;
#endif

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kPerCall; ++i) {
    const unsigned c = idx * kPerCall + i;
    const uint8_t *__restrict tile = chunks + i * kChunkBytes;
    float *__restrict yb =
        y + (c / kRowSplit) * kBatch * kBandAcc + (c % kRowSplit) * kRows;
#if defined(GEMM_Q4K)
    gemm_q4k_prep_chunk(tile, scr);
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; m += 4) {
      gemm_q4k_tile4(tile, scr, tab + (m / 4) * kILGroupBytes, first, last,
                     yb + m * kBandAcc, kBandAcc);
    }
#elif defined(GEMM_G128) && defined(GEMM_INT8)
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; m += 8) {
      gemm_q4_g128_i8(tile, tab + (m / 8) * k8GroupBytes, first, last,
                      yb + m * kBandAcc, kBandAcc);
    }
#elif defined(GEMM_G128)
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; m += 4) {
      gemm_q4_g128_il(tile, tab + (m / 4) * kILGroupBytes, first, last,
                      yb + m * kBandAcc, kBandAcc);
    }
#elif defined(GEMM_INT8)
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; m += 8) {
      gemm_q4_tile8_i8(tile, tab + (m / 8) * k8GroupBytes, first, last,
                       yb + m * kBandAcc, kBandAcc);
    }
#elif defined(GEMM_TAB_IL)
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; m += 4) {
      gemm_q4_tile4_il(tile, tab + (m / 4) * kILGroupBytes, first, last,
                       yb + m * kBandAcc, kBandAcc);
    }
#elif GEMM_BATCH % 4 == 0
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; m += 4) {
      gemm_q4_tile4(tile, tab + m * kSliceBytes, kSliceBytes, kTileK, 0, first, last,
                    yb + m * kBandAcc, kBandAcc);
    }
#else
#pragma clang loop unroll(disable)
    for (unsigned m = 0; m < kBatch; ++m) {
      gemv_q4_tile(tile, tab + m * kSliceBytes, kTileK, 0, first, last, yb + m * kBandAcc);
    }
#endif
  }
}

#define GEMM_Q4_BLK_ENTRY__(P, B, R, M)                                        \
  void gemm_q4_blk_p##P##b##B##r##R##m##M(const uint8_t *__restrict t,         \
                                          const uint8_t *__restrict tab,       \
                                          float *__restrict y, int32_t idx,    \
                                          int32_t kt) {                        \
    gemm_q4_blk_group(t, tab, y, (unsigned)idx, (unsigned)kt);                 \
  }
#define GEMM_Q4_BLK_ENTRY_(P, B, R, M) GEMM_Q4_BLK_ENTRY__(P, B, R, M)
#define GEMM_Q4_BLK_ENTRY()                                                    \
  GEMM_Q4_BLK_ENTRY_(GEMV_PER_CALL, GEMV_PER_BAND, GEMV_ROWSPLIT, GEMM_BATCH)

// Block-quantise one token's k-tile slice into slot m of the slice table.
#ifdef GEMM_INT8
// m counts PAIRS of tokens: four pairs fill one eight-token group.
#define GEMM_Q4_PREP_SLICE_ENTRY__(M)                                          \
  void gemm_q4_prep_slice_m##M(const bfloat16 *__restrict x,                   \
                               uint8_t *__restrict tab, int32_t m) {           \
    gemm_q4_prep_i8(x, tab + ((unsigned)m / 4) * k8GroupBytes,                 \
                    (unsigned)m % 4);                                          \
  }
#elif defined(GEMM_TAB_IL)
#define GEMM_Q4_PREP_SLICE_ENTRY__(M)                                          \
  void gemm_q4_prep_slice_m##M(const bfloat16 *__restrict x,                   \
                               uint8_t *__restrict tab, int32_t m) {           \
    gemm_q4_prep_il(x, tab + ((unsigned)m / 4) * kILGroupBytes,                \
                    (unsigned)m % 4);                                          \
  }
#else
#define GEMM_Q4_PREP_SLICE_ENTRY__(M)                                          \
  void gemm_q4_prep_slice_m##M(const bfloat16 *__restrict x,                   \
                               uint8_t *__restrict tab, int32_t m) {           \
    gemv_q4_prep(x, tab + (unsigned)m * kSliceBytes, kTileK);                  \
  }
#endif
#define GEMM_Q4_PREP_SLICE_ENTRY_(M) GEMM_Q4_PREP_SLICE_ENTRY__(M)
#define GEMM_Q4_PREP_SLICE_ENTRY() GEMM_Q4_PREP_SLICE_ENTRY_(GEMM_BATCH)
