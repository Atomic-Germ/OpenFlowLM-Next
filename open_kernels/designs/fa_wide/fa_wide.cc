//===- fa_wide.cc ---------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Flash-attention compute kernels for fa_wide (fa_wide.py): head dims 256 and
// 512. The arithmetic is designs/dit_fa/fa_dit.cc's (itself attn_npu2.cc's, as
// vendored in ../whisper_fa/attn_npu2.cc: Xilinx/mlir-air e91630a,
// programming_examples/flash_attention/kernel_fusion_based, MIT; see
// open_kernels/PROVENANCE.md): the 8x8x8 mmul loop, the lazily rescaled online
// softmax, exp2 on the hardware, the fp32 running sum on the MAC array.
//
// What is different from fa_dit.cc, and why:
//
//   - 32-key chunks and 32 (head dim 256) or 16 (512) query rows a core, so
//     Q, O and the key fifos fit a core's 64 KB at these head dims.
//   - Row loops cover TQ rows of 16 or 32 (fa_dit.cc's assumed 32).
//   - The mask takes the stream's cache position (rtp), a sliding window
//     (rtp) and the pass's first key chunk (rtp), so one stream serves the
//     sliding and the global layers of a head dim at any prompt position.
//   - The score scale is a build constant (FW_SCALE): Gemma 4's scores are
//     unscaled.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef FW_TQ
#define FW_TQ 32 // query rows per core
#endif
#ifndef FW_LKP
#define FW_LKP 32 // keys per chunk
#endif
#ifndef FW_DC
#define FW_DC 64 // head-dim chunk: the K/V/Q piece one mmul call consumes
#endif
#ifndef FW_DFULL
#define FW_DFULL 256 // head dimension
#endif
#ifndef FW_QB
#define FW_QB 1 // cores whose query rows make one fifo block (QB * TQ == LKP)
#endif
#ifndef FW_PASS
#define FW_PASS 256 // query rows a group covers per pass
#endif
#ifndef FW_SCALE
#define FW_SCALE 0.0f // 0: 1 / sqrt(head dim)
#endif
#ifndef FW_TAU
#define FW_TAU 32
#endif
constexpr int TQ = FW_TQ;
constexpr int LKP = FW_LKP;
constexpr int DC = FW_DC;
constexpr int DFULL = FW_DFULL;
constexpr int QB = FW_QB;
constexpr int PASS = FW_PASS;

constexpr int NDCH = DFULL / DC;
static_assert(DFULL % DC == 0, "head dim must be whole chunks");
static_assert(TQ == 16 || TQ == 32, "row loops take 16 or 32 rows");
static_assert(LKP % 16 == 0 && DC % 16 == 0, "mmul 2x2 blocking");
static_assert(QB * TQ == LKP, "a Q block is the size of a K block: they share a fifo");

constexpr int G_ELEMS = TQ * LKP;
constexpr int GP_ELEMS = TQ * DFULL;
constexpr int BLOCK_STRIDE = TQ * 8; // between 8-column blocks, all tiles here
constexpr int REP = TQ * 8;          // a per-row value replicated over 8 lanes

constexpr double sqrt_d = (DFULL == 256) ? 16.0 : (DFULL == 512) ? 22.627416997969522 : 0.0;
static_assert(sqrt_d != 0.0, "head dim 256 or 512");
constexpr double fw_scale = (FW_SCALE > 0.0f) ? double(FW_SCALE) : 1.0 / sqrt_d;
#define log2e (1.44269504089 * fw_scale)

static inline bfloat16 bf16_lowest() {
  uint16_t u = 0xff7f;
  return *(bfloat16 *)&u;
}

// attn_npu2.cc's matmul_vectorized_2x2_mmul, unchanged. A and C are 8x8-block
// column-major ([col block][row block][8][8]); B is K (transpose_b, blocks
// [k block][n block]) or V (blocks [n block][k block]).
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t, bool transpose_b>
static inline void matmul_2x2(const T_in *__restrict pA,
                              const T_in *__restrict pB, T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;
      for (unsigned j = 0; j < colB; j += 2) {
        const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
        const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;
        aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
        aie::vector<T_out, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);
        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        for (unsigned i = 0; i < colA; ++i) {
          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0, B1;
          if constexpr (transpose_b) {
            const T_in *__restrict pBk0 = pB + (i * colB + j) * MMUL::size_B;
            const T_in *__restrict pBk1 = pB + (i * colB + (j + 1)) * MMUL::size_B;
            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pBk0), t, s);
            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pBk1), t, s);
          } else {
            B0 = aie::load_v<MMUL::size_B>(pB1);
            B1 = aie::load_v<MMUL::size_B>(pB2);
          }
          pB1 += MMUL::size_B;
          pB2 += MMUL::size_B;
          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);
        }
        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C * rowA;
      }
    }
}

// attn_npu2.cc's exp2f_poly16 (from mlir-aie aie_kernels/aie2p/exp2f_vec.cc,
// Apache-2.0 WITH LLVM-exception): aie::exp2<float> does not exist on aie2p,
// and the bf16 LUT is 6-49% off on the rescale factor's domain.
static __attribute__((noinline)) aie::vector<float, 16>
exp2f_poly16(aie::vector<float, 16> x) {
  constexpr int N = 16;
  x = aie::max(x, aie::broadcast<float, N>(-111.0f));
  aie::mask<N> overflow = aie::ge(x, aie::broadcast<float, N>(128.0f));
  x = aie::min(x, aie::broadcast<float, N>(127.999f));
  aie::vector<int32_t, N> ki = aie::to_fixed<int32_t>(x);
  aie::vector<float, N> kf = aie::to_float<float>(ki);
  aie::vector<int32_t, N> one = aie::broadcast<int32_t, N>(1);
  aie::vector<int32_t, N> zero = aie::broadcast<int32_t, N>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, N> f = aie::sub(x, aie::to_float<float>(ki));
  aie::vector<float, N> p = aie::broadcast<float, N>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(1.0f));
  aie::vector<int32_t, N> ebits =
      aie::upshift(aie::add(ki, aie::broadcast<int32_t, N>(127)), 23);
  aie::vector<float, N> p2k = ebits.template cast_to<float>();
  aie::vector<float, N> result = aie::mul(p, p2k).to_vector<float>();
  aie::vector<int32_t, N> pos_inf_bits = aie::broadcast<int32_t, N>(0x7f800000);
  aie::vector<float, N> pos_inf = pos_inf_bits.template cast_to<float>();
  return aie::select(result, pos_inf, overflow);
}

// Row max of G ([TQ, LKP], 8x8-block column-major) into out[TQ].
static inline void row_max(const bfloat16 *__restrict in, bfloat16 *__restrict out) {
  constexpr int col_blocks = LKP / 8;
  constexpr int row_blocks = TQ / 8;
  aie::vector<bfloat16, 64> lowest_vec = aie::broadcast<bfloat16, 64>(bf16_lowest());
  for (int rb = 0; rb < row_blocks; rb++) {
    const bfloat16 *__restrict p = in + rb * 64;
    aie::vector<bfloat16, 64> m = lowest_vec;
    for (int cb = 0; cb < col_blocks; cb++)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        m = aie::max(m, aie::load_v<64>(p + cb * BLOCK_STRIDE));
      }
    aie::vector<bfloat16, 64> t = aie::transpose(m, 8, 8);
    aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
    aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
    aie::vector<bfloat16, 8> c = aie::max(b.extract<8>(0), b.extract<8>(1));
    aie::store_v(out + rb * 8, c);
  }
}

// Replicate each of TQ per-row values 8 times, in the 4-row x 8-lane groups a
// 32-lane load of an 8x8 block covers: rep[g*32 + row_in_group*8 + lane]. The
// source is read as 32 lanes (the per-row buffers are 32 long at TQ 16).
static inline void replicate_rows(const bfloat16 *__restrict v32, bfloat16 *__restrict rep) {
  using V = aie::vector<bfloat16, 32>;
  // two 32-byte loads: a 64-byte load ignores the address's low bits, and `up` may sit only 32-byte aligned
  V uv = aie::concat(aie::load_v<16>(v32), aie::load_v<16>(v32 + 16));
  auto z1 = aie::interleave_zip(uv, uv, 1);
  auto z2a = aie::interleave_zip(z1.first, z1.first, 2);
  auto z3a = aie::interleave_zip(z2a.first, z2a.first, 4);
  auto z3b = aie::interleave_zip(z2a.second, z2a.second, 4);
  aie::store_v(rep + 0, z3a.first);
  aie::store_v(rep + 32, z3a.second);
  aie::store_v(rep + 64, z3b.first);
  aie::store_v(rep + 96, z3b.second);
  if constexpr (TQ == 32) {
    auto z2b = aie::interleave_zip(z1.second, z1.second, 2);
    auto z3c = aie::interleave_zip(z2b.first, z2b.first, 4);
    auto z3d = aie::interleave_zip(z2b.second, z2b.second, 4);
    aie::store_v(rep + 128, z3c.first);
    aie::store_v(rep + 160, z3c.second);
    aie::store_v(rep + 192, z3d.first);
    aie::store_v(rep + 224, z3d.second);
  }
}

// G = exp2(G*c - m*c) in place for one 4-row group (n 8-column blocks); nuc =
// the group's -m*c, replicated per lane (32 floats: rows x 8). The argument is
// formed in fp32 (c split into bf16 hi + lo).
template <int n>
static inline __attribute__((always_inline)) void
exp_cols(bfloat16 *__restrict p, const float *__restrict nuc,
         aie::vector<bfloat16, 16> sc_hi, aie::vector<bfloat16, 16> sc_lo) {
  using V = aie::vector<bfloat16, 32>;
  const aie::vector<float, 16> nu0 = aie::load_v<16>(nuc);
  const aie::vector<float, 16> nu1 = aie::load_v<16>(nuc + 16);
  V v[n];
#pragma clang loop unroll(full)
  for (int cb = 0; cb < n; cb++)
    v[cb] = aie::load_v<32>(p + cb * BLOCK_STRIDE);
#pragma clang loop unroll(full)
  for (int cb = 0; cb < n; cb++) {
    aie::vector<bfloat16, 16> lo = v[cb].template extract<16>(0);
    aie::vector<bfloat16, 16> hi = v[cb].template extract<16>(1);
    aie::accum<accfloat, 16> a0, a1;
    a0.from_vector(nu0);
    a1.from_vector(nu1);
    a0 = aie::mac(aie::mac(a0, lo, sc_hi), lo, sc_lo);
    a1 = aie::mac(aie::mac(a1, hi, sc_hi), hi, sc_lo);
    V d;
    d.insert(0, aie::exp2<bfloat16>(a0.to_vector<float>()));
    d.insert(1, aie::exp2<bfloat16>(a1.to_vector<float>()));
    v[cb] = d;
  }
#pragma clang loop unroll(full)
  for (int cb = 0; cb < n; cb++)
    aie::store_v(p + cb * BLOCK_STRIDE, v[cb]);
}

// Multiply the TQ rows of a block-column-major tile with ncols columns by rep's
// per-row factors (replicate_rows).
template <int ncols>
static inline void scale_rows(bfloat16 *__restrict x, const bfloat16 *__restrict rep) {
  using V = aie::vector<bfloat16, 32>;
  constexpr int col_blocks = ncols / 8;
  constexpr int chunk = 8;
  static_assert(col_blocks % chunk == 0);
  for (int gi = 0; gi < TQ / 4; gi++) {
    V rv = aie::load_v<32>(rep + gi * 32);
    bfloat16 *__restrict p = x + gi * 32;
#pragma clang loop unroll(disable)
    for (int c0 = 0; c0 < col_blocks; c0 += chunk) {
      V v[chunk];
#pragma clang loop unroll(full)
      for (int cb = 0; cb < chunk; cb++)
        v[cb] = aie::load_v<32>(p + (c0 + cb) * BLOCK_STRIDE);
#pragma clang loop unroll(full)
      for (int cb = 0; cb < chunk; cb++)
        v[cb] = aie::mul(v[cb], rv).template to_vector<bfloat16>();
#pragma clang loop unroll(full)
      for (int cb = 0; cb < chunk; cb++)
        aie::store_v(p + (c0 + cb) * BLOCK_STRIDE, v[cb]);
    }
  }
}

alignas(64) static bfloat16 s_rep[REP];
alignas(64) static float s_nuc[REP];
alignas(64) static bfloat16 s_newmax[32];
alignas(64) static bfloat16 s_resc[32];

constexpr float TAU = float(FW_TAU);

// l += rowsum(P) on the MAC array: P [TQ, LKP] @ ones [LKP, 8], fp32 out, into the replicated l.
static inline void rowsum_mmul(const bfloat16 *__restrict g, float *__restrict l) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  const aie::vector<bfloat16, 64> ones = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  for (int z = 0; z < TQ / 8; z++) {
    MMUL C;
    C.mul(aie::load_v<64>(g + z * 64), ones);
    for (int i = 1; i < LKP / 8; i++)
      C.mac(aie::load_v<64>(g + (z + i * (TQ / 8)) * 64), ones);
    aie::vector<float, 64> sv = C.template to_vector<float>();
    float *__restrict pl = l + z * 64;
    for (int j = 0; j < 64; j += 16)
      aie::store_v(pl + j, aie::add(aie::load_v<16>(pl + j), sv.extract<16>(j / 16)));
  }
}

// dst[i] (fp32) = max(bf16 src[i], -1e30) * k, n values.
static inline void widen_scale(const bfloat16 *__restrict src, float *__restrict dst, float k, int n) {
  // a row with no key yet has max = lowest, and -lowest*c + lowest*c cancels to +-1e30: the floor keeps its exp2 at 0
  const aie::vector<float, 16> floor = aie::broadcast<float, 16>(-1e30f);
  for (int i = 0; i < n; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(src + i));
    aie::store_v(dst + i, aie::mul(aie::max(a.template to_vector<float>(), floor), k).template to_vector<float>());
  }
}

extern "C" {

#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)

// Q capture. A Q block is QB cores' TQ rows x DFULL; the memtile emits it as
// NDCH [QB*TQ, DC] chunks in mmul order, [DC/8][QB*TQ/8][8][8], so chunk `dch`
// holds this core's rows (sub of the QB) as TQ/8-block runs, one per 8-column
// block. The matmul wants [dch][cb][rb][8][8].
void fw_capture_q(bfloat16 *__restrict src, bfloat16 *__restrict dst, int32_t dch, int32_t sub) {
  SET_ROUNDING();
  constexpr int RUN = TQ * 8;
  for (int cb = 0; cb < DC / 8; cb++) {
    const bfloat16 *__restrict ps = src + (cb * QB + sub) * RUN;
    bfloat16 *__restrict pd = dst + dch * (TQ * DC) + cb * RUN;
    for (int j = 0; j < RUN; j += 32)
      aie::store_v(pd + j, aie::load_v<32>(ps + j));
  }
}

// Start of a pass: zero O, reset the running max (32 long) and sum.
void fw_pass_init(bfloat16 *__restrict gp, bfloat16 *__restrict up, float *__restrict sp) {
  SET_ROUNDING();
  aie::vector<bfloat16, 32> z = aie::zeros<bfloat16, 32>();
  for (int i = 0; i < GP_ELEMS; i += 32)
    aie::store_v(gp + i, z);
  aie::store_v(up, aie::broadcast<bfloat16, 32>(bf16_lowest()));
  aie::vector<float, 16> zf = aie::zeros<float, 16>();
  for (int i = 0; i < REP; i += 16)
    aie::store_v(sp + i, zf);
}

void fw_zero_g(bfloat16 *g) {
  SET_ROUNDING();
  aie::vector<bfloat16, 32> z = aie::zeros<bfloat16, 32>();
  for (int i = 0; i < G_ELEMS; i += 32)
    aie::store_v(g + i, z);
}

// G += Q[:, chunk c] @ K_piece^T  (K piece [LKP, DC])
void fw_qk(bfloat16 *q, bfloat16 *k, bfloat16 *g, int32_t c) {
  SET_ROUNDING();
  matmul_2x2<bfloat16, bfloat16, TQ / 8, DC / 8, LKP / 8, 8, 8, 8, true>(q + c * (TQ * DC), k, g);
}

// O[:, chunk c] += P @ V_piece  (V piece [LKP, DC])
void fw_pv(bfloat16 *g, bfloat16 *v, bfloat16 *gp, int32_t c) {
  SET_ROUNDING();
  matmul_2x2<bfloat16, bfloat16, TQ / 8, LKP / 8, DC / 8, 8, 8, 8, false>(g, v, gp + c * (TQ * DC));
}

// Key mask for chunk `chunk` of pass `pass` (fa_wide.py RTP_*): rtp[2] the core's
// first row in a pass, rtp[3] the stream's cache position, rtp[4] the window (0:
// none), rtp[8 + pass] the pass's first key chunk. Column j is kept for query row r
// when j <= r and, with a window W, j > r - W.
void fw_mask(bfloat16 *__restrict g, int32_t *__restrict rtp, int32_t pass, int32_t chunk) {
  SET_ROUNDING();
  const int32_t row0 = rtp[3] + pass * PASS + rtp[2];
  const int32_t win = rtp[4];
  const int32_t col0 = (rtp[8 + pass] + chunk) * LKP;
  const int32_t col1 = col0 + LKP - 1;
  // every row keeps every column of the chunk?
  const bool causal_all = col1 <= row0;
  const bool window_all = win == 0 || col0 > row0 + TQ - 1 - win;
  if (causal_all && window_all)
    return;
  const bfloat16 ninf = bf16_lowest();
  // no row keeps any column?
  if (col0 > row0 + TQ - 1 || (win && col1 <= row0 - win)) {
    aie::vector<bfloat16, 32> v = aie::broadcast<bfloat16, 32>(ninf);
    for (int i = 0; i < G_ELEMS; i += 32)
      aie::store_v(g + i, v);
    return;
  }
  for (int row = 0; row < TQ; row++) {
    const int32_t r = row0 + row;
    bfloat16 *__restrict p = g + (row / 8) * 64 + (row % 8) * 8;
    for (int j = 0; j < LKP; j++) {
      const int32_t col = col0 + j;
      if (col > r || (win && col <= r - win))
        p[(j / 8) * BLOCK_STRIDE + (j % 8)] = ninf;
    }
  }
}

// One key chunk of the online softmax, after G holds this chunk's scores (fa_dit.cc's softmax_step over TQ rows).
void fw_softmax_step(bfloat16 *__restrict g, bfloat16 *__restrict up, float *__restrict sp, bfloat16 *__restrict gp) {
  SET_ROUNDING();
  using V = aie::vector<bfloat16, 16>;
  const float c = float(log2e);
  row_max(g, s_newmax);
  bool resc = false;
  const V tau = aie::broadcast<bfloat16, 16>((bfloat16)(TAU / c));
  for (int i = 0; i < TQ; i += 16)
    resc |= !aie::gt(aie::load_v<16>(s_newmax + i), aie::add(aie::load_v<16>(up + i), tau)).empty();
  if (resc) {
    for (int i = 0; i < TQ; i += 16) {
      aie::vector<bfloat16, 16> u = aie::load_v<16>(up + i);
      aie::vector<bfloat16, 16> mn = aie::max(u, aie::load_v<16>(s_newmax + i));
      aie::accum<accfloat, 16> au, am;
      au.from_vector(u);
      am.from_vector(mn);
      aie::vector<float, 16> d = aie::sub(au.to_vector<float>(), am.to_vector<float>());
      d = aie::max(d, aie::broadcast<float, 16>(-1e30f));
      aie::vector<float, 16> e = exp2f_poly16(aie::mul(d, c).template to_vector<float>());
      aie::accum<accfloat, 16> ea;
      ea.from_vector(e);
      aie::store_v(s_resc + i, ea.template to_vector<bfloat16>());
      aie::store_v(up + i, mn);
    }
    // O *= r and l *= r, with the same bf16 r
    replicate_rows(s_resc, s_rep);
    scale_rows<DFULL>(gp, s_rep);
    for (int j = 0; j < REP; j += 16) {
      aie::accum<accfloat, 16> ra;
      ra.from_vector(aie::load_v<16>(s_rep + j));
      aie::store_v(sp + j, aie::mul(aie::load_v<16>(sp + j), ra.to_vector<float>()).template to_vector<float>());
    }
    // -m*c for the exponentials
    replicate_rows(up, s_rep);
    widen_scale(s_rep, s_nuc, -c, REP);
  }
  const bfloat16 sc_hi_s = (bfloat16)log2e;
  const bfloat16 sc_lo_s = (bfloat16)(float(log2e) - float(sc_hi_s));
  const aie::vector<bfloat16, 16> sc_hi = aie::broadcast<bfloat16, 16>(sc_hi_s);
  const aie::vector<bfloat16, 16> sc_lo = aie::broadcast<bfloat16, 16>(sc_lo_s);
  for (int gi = 0; gi < TQ / 4; gi++)
    exp_cols<LKP / 8>(g + gi * 32, s_nuc + gi * 32, sc_hi, sc_lo);
  rowsum_mmul(g, sp);
}

// End of a pass: O /= l (l replicated, so its bf16 inverse is scale_rows' table).
void fw_finalize(float *__restrict sp, bfloat16 *__restrict gp) {
  SET_ROUNDING();
  for (int j = 0; j < REP; j += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(sp + j));
    aie::store_v(s_rep + j, aie::inv(a.template to_vector<bfloat16>()));
  }
  scale_rows<DFULL>(gp, s_rep);
}

} // extern "C"
