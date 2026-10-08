//===- fa_dit.cc ----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Flash-attention compute kernels for dit_fa (dit_fa.py). The arithmetic --
// the 8x8x8 mmul loop, the row max, exp(G - u), the rescale exponential and
// the fp32 running sum -- is attn_npu2.cc's, as vendored in
// ../whisper_fa/attn_npu2.cc (Xilinx/mlir-air e91630a,
// programming_examples/flash_attention/kernel_fusion_based, MIT; see
// open_kernels/PROVENANCE.md), with that file's Whisper changes (fp32 running
// sum, software exp2 for the rescale factor, round-to-nearest-even).
//
// What is different here, and why:
//
//   - The head dimension is split into DC-wide chunks inside the core.
//     Q is captured once per pass as NDCH chunks, K and V arrive as
//     [LKP, DC] pieces and each piece is one mmul call; the output
//     accumulator is the whole [TQ, DFULL] slab. At d = 128 the unchunked
//     tiles need ~80 KB of a 64 KB L1 (whisper_fa's own formula).
//   - No cascade. Each core walks every key for its own TQ query rows and
//     finishes its own softmax, so there is no merge; see dit_fa.py.
//   - One call per key chunk for the softmax update (softmax_step), with its
//     scratch in this file's .bss, instead of seven IRON-level calls.
//   - The mask takes the key length and the causal flag at run time
//     (apply_mask), so one xclbin serves the DiT and the text encoder.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

// Compile-time shape (dit_fa.py passes FA_*; bare names like DC collide with
// aie_api identifiers when given as -D macros).
#ifndef FA_TQ
#define FA_TQ 32 // query rows per core
#endif
#ifndef FA_LKP
#define FA_LKP 64 // keys per chunk
#endif
#ifndef FA_DC
#define FA_DC 64 // head-dim chunk: the K/V/Q piece one mmul call consumes
#endif
#ifndef FA_DFULL
#define FA_DFULL 128 // head dimension
#endif
#ifndef FA_QROWS_PASS
#define FA_QROWS_PASS 512 // query rows a group covers per pass (cores x TQ)
#endif
constexpr int TQ = FA_TQ;
constexpr int LKP = FA_LKP;
constexpr int DC = FA_DC;
constexpr int DFULL = FA_DFULL;
constexpr int QROWS_PASS = FA_QROWS_PASS;

constexpr int NDCH = DFULL / DC;
static_assert(DFULL % DC == 0, "head dim must be whole chunks");
static_assert(TQ % 32 == 0, "row loops load 32 rows at a time");
static_assert(LKP % 16 == 0 && DC % 16 == 0, "mmul 2x2 blocking");
// A Q block (two cores' rows) is the size of a K block -- they share a fifo.
static_assert(2 * TQ == LKP, "Q block must be the size of a K block");

constexpr int G_ELEMS = TQ * LKP;
constexpr int GP_ELEMS = TQ * DFULL;
constexpr int BLOCK_STRIDE = TQ * 8; // between 8-column blocks, all tiles here

// scale: log2(e) / sqrt(DFULL), applied inside the exponentials.
constexpr double sqrt_d = (DFULL == 64)    ? 8.0
                          : (DFULL == 128) ? 11.313708498984761
                          : (DFULL == 256) ? 16.0
                          : (DFULL == 512) ? 22.627416997969522
                                           : 0.0;
static_assert(sqrt_d != 0.0, "add sqrt(DFULL) for this head dim");
// exp_fix's quadratic (see there)
#define FA_EXP_C0 1.4406570f
#define FA_EXP_C1 -0.671875f
#define FA_EXP_C2 0.2275390625f
#define log2e (1.44269504089 / sqrt_d)

static inline bfloat16 bf16_lowest() {
  uint16_t u = 0xff7f;
  return *(bfloat16 *)&u;
}
static inline bfloat16 bf16_neg_inf() {
  uint16_t u = 0xff80;
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
    chess_prepare_for_pipelining chess_loop_range(2, ) {
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
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        m = aie::max(m, aie::load_v<64>(p + cb * BLOCK_STRIDE));
      }
    aie::vector<bfloat16, 64> t = aie::transpose(m, 8, 8);
    aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
    aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
    aie::vector<bfloat16, 8> c = aie::max(b.extract<8>(0), b.extract<8>(1));
    aie::store_v(out + rb * 8, c);
  }
}

// Replicate each of 32 per-row values 8 times, in the 4-row x 8-lane groups a
// 32-lane load of an 8x8 block covers: rep[g*32 + row_in_group*8 + lane].
static inline void replicate_rows32(const bfloat16 *__restrict v32,
                                    bfloat16 *__restrict rep) {
  using V = aie::vector<bfloat16, 32>;
  V uv = aie::load_v<32>(v32);
  auto z1 = aie::interleave_zip(uv, uv, 1);
  auto z2a = aie::interleave_zip(z1.first, z1.first, 2);
  auto z2b = aie::interleave_zip(z1.second, z1.second, 2);
  auto z3a = aie::interleave_zip(z2a.first, z2a.first, 4);
  auto z3b = aie::interleave_zip(z2a.second, z2a.second, 4);
  auto z3c = aie::interleave_zip(z2b.first, z2b.first, 4);
  auto z3d = aie::interleave_zip(z2b.second, z2b.second, 4);
  aie::store_v(rep + 0, z3a.first);
  aie::store_v(rep + 32, z3a.second);
  aie::store_v(rep + 64, z3b.first);
  aie::store_v(rep + 96, z3b.second);
  aie::store_v(rep + 128, z3c.first);
  aie::store_v(rep + 160, z3c.second);
  aie::store_v(rep + 192, z3d.first);
  aie::store_v(rep + 224, z3d.second);
}

#ifndef FA_EXP_FIX
#define FA_EXP_FIX 0
#endif

// The hardware exp2 (aie::exp2<bfloat16>) returns 2^n * (1 + f), f the input's
// fraction truncated to 7 bits: a linear mantissa, +3.8% mean / +6.1% worst
// (utilities/aie-probes/exp2_probe.py). FA_EXP_FIX multiplies by h(1 + f) ~
// 2^(f + 1/256) / (1 + f), a quadratic in the returned mantissa: rms error 3e-3
// on [-2, 0]. Coefficients: fa_emul.py's EXP_FIX (bf16 c1/c2, fp32 c0).
static inline __attribute__((always_inline)) aie::vector<bfloat16, 32>
exp_fix(aie::vector<bfloat16, 32> y) {
  aie::vector<int16_t, 32> yi = y.template cast_to<int16_t>();
  aie::vector<int16_t, 32> mi =
      aie::bit_or(aie::bit_and(yi, aie::broadcast<int16_t, 32>(0x007F)),
                  aie::broadcast<int16_t, 32>(0x3F80));
  aie::vector<bfloat16, 32> m = mi.template cast_to<bfloat16>();
  aie::vector<bfloat16, 32> m2 = aie::mul(m, m).template to_vector<bfloat16>();
  aie::accum<accfloat, 32> h;
  h.from_vector(aie::broadcast<float, 32>(FA_EXP_C0));
  h = aie::mac(h, m, aie::broadcast<bfloat16, 32>((bfloat16)FA_EXP_C1));
  h = aie::mac(h, m2, aie::broadcast<bfloat16, 32>((bfloat16)FA_EXP_C2));
  return aie::mul(y, h.template to_vector<bfloat16>()).template to_vector<bfloat16>();
}

// G = exp2(G*c - m*c) in place for one 4-row group (n 8-column blocks); nuc =
// the group's -m*c, replicated per lane (32 floats: rows x 8). The argument is
// formed in fp32 (c split into bf16 hi + lo), so neither c nor G - m is rounded
// to bf16.
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
#if FA_EXP_FIX
    d = exp_fix(d);
#endif
    v[cb] = d;
  }
#pragma clang loop unroll(full)
  for (int cb = 0; cb < n; cb++)
    aie::store_v(p + cb * BLOCK_STRIDE, v[cb]);
}

// Multiply 32 rows of a block-column-major tile with TQ rows and ncols columns
// by rep's per-row factors (replicate_rows32).
template <int ncols>
static inline void scale_rows(bfloat16 *__restrict x, const bfloat16 *__restrict rep) {
  using V = aie::vector<bfloat16, 32>;
  constexpr int col_blocks = ncols / 8;
  constexpr int chunk = col_blocks < 8 ? col_blocks : 8;
  static_assert(col_blocks % chunk == 0);
  for (int gi = 0; gi < 8; gi++) {
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

// Per-row values are kept "replicated": row r's value in lanes r*8 .. r*8+7, the
// layout of an 8-column mmul tile and of replicate_rows32's table.
alignas(64) static bfloat16 s_rep[TQ * 8];   // scratch: a per-row bf16 value, replicated
alignas(64) static float s_nuc[TQ * 8];      // -m*c, replicated (exp argument offset)
alignas(64) static bfloat16 s_newmax[TQ];
alignas(64) static bfloat16 s_resc[TQ];

// Lazy rescale: the reference max m moves only when a row's chunk max exceeds it
// by more than TAU (log2 units), so P = exp2((s - m)c) <= 2^TAU (bf16 has the
// range) and the O / l rescale runs on a handful of chunks per pass, not all.
constexpr float TAU = 8.0f;

// l += rowsum(P) on the MAC array: P [TQ, LKP] @ ones [LKP, 8], fp32 out, into the
// replicated l. The mmul sees P in bfp16 exactly as pv's does, so l normalises the
// weights O was actually built from.
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

// dst[i] (fp32) = bf16 src[i] * k, n values.
static inline void widen_scale(const bfloat16 *__restrict src, float *__restrict dst,
                               float k, int n) {
  for (int i = 0; i < n; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(src + i));
    aie::store_v(dst + i,
                 aie::mul(a.template to_vector<float>(), k).template to_vector<float>());
  }
}

extern "C" {

#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)

// Q capture. A Q block is two cores' TQ rows x DFULL; the memtile emits it as
// NDCH [2*TQ, DC] chunks in mmul order, [DC/8][2*TQ/8][8][8], so chunk `dch`
// holds this core's rows (sub = 0 or 1 of the pair) as TQ/8-block runs, one
// per 8-column block. The matmul wants [dch][cb][rb][8][8].
void capture_q(bfloat16 *__restrict src, bfloat16 *__restrict dst, int32_t dch,
               int32_t sub) {
  SET_ROUNDING();
  constexpr int RUN = TQ * 8;
  for (int cb = 0; cb < DC / 8; cb++) {
    const bfloat16 *__restrict ps = src + (cb * 2 + sub) * RUN;
    bfloat16 *__restrict pd = dst + dch * (TQ * DC) + cb * RUN;
    for (int j = 0; j < RUN; j += 32)
      aie::store_v(pd + j, aie::load_v<32>(ps + j));
  }
}

// Start of a pass: zero the output accumulator, reset the running max/sum.
// sp = l, replicated ([TQ * 8] fp32).
void pass_init(bfloat16 *__restrict gp, bfloat16 *__restrict up, float *__restrict sp) {
  SET_ROUNDING();
  aie::vector<bfloat16, 32> z = aie::zeros<bfloat16, 32>();
  for (int i = 0; i < GP_ELEMS; i += 32)
    aie::store_v(gp + i, z);
  aie::vector<bfloat16, 32> lo = aie::broadcast<bfloat16, 32>(bf16_lowest());
  for (int i = 0; i < TQ; i += 32)
    aie::store_v(up + i, lo);
  aie::vector<float, 16> zf = aie::zeros<float, 16>();
  for (int i = 0; i < TQ * 8; i += 16)
    aie::store_v(sp + i, zf);
}

void zero_g(bfloat16 *__restrict g) {
  SET_ROUNDING();
  aie::vector<bfloat16, 32> z = aie::zeros<bfloat16, 32>();
  for (int i = 0; i < G_ELEMS; i += 32)
    aie::store_v(g + i, z);
}

// G += Q[:, chunk c] @ K_piece^T  (K piece [LKP, DC])
void qk(bfloat16 *q, bfloat16 *k, bfloat16 *g, int32_t c) {
  SET_ROUNDING();
  matmul_2x2<bfloat16, bfloat16, TQ / 8, DC / 8, LKP / 8, 8, 8, 8, true>(
      q + c * (TQ * DC), k, g);
}

// O[:, chunk c] += P @ V_piece  (V piece [LKP, DC])
void pv(bfloat16 *g, bfloat16 *v, bfloat16 *gp, int32_t c) {
  SET_ROUNDING();
  matmul_2x2<bfloat16, bfloat16, TQ / 8, LKP / 8, DC / 8, 8, 8, 8, false>(
      g, v, gp + c * (TQ * DC));
}

// Key mask for one chunk: column j of chunk `chunk` is masked for query row r
// of pass `pass` when j >= valid_len, or (causal) when j > r. rtp = the core's
// runtime parameters (dit_fa.py RTP_*): [.., valid_len @3, causal @4, the
// core's first row within a pass @5].
void apply_mask(bfloat16 *__restrict g, int32_t *__restrict rtp, int32_t pass,
                int32_t chunk) {
  SET_ROUNDING();
  const int32_t valid_len = rtp[3];
  const int32_t causal = rtp[4];
  const int32_t row0 = pass * QROWS_PASS + rtp[5];
  const int32_t col0 = chunk * LKP;
  // Last column any row may keep; all rows keep everything up to it?
  int32_t keep_all = valid_len; // columns < keep_all kept by every row
  if (causal && row0 + 1 < keep_all)
    keep_all = row0 + 1;
  if (col0 + LKP <= keep_all)
    return;
  int32_t keep_any = valid_len; // columns >= keep_any masked for every row
  if (causal && row0 + TQ < keep_any)
    keep_any = row0 + TQ;
  // bf16 lowest, not -inf: the exp argument is formed as G*c - m*c, and
  // -inf*c - m*c must stay finite (it underflows to exactly 0 in exp2).
  const bfloat16 ninf = bf16_lowest();
  if (col0 >= keep_any) {
    aie::vector<bfloat16, 32> v = aie::broadcast<bfloat16, 32>(ninf);
    for (int i = 0; i < G_ELEMS; i += 32)
      aie::store_v(g + i, v);
    return;
  }
  // Ragged: per element.
  for (int row = 0; row < TQ; row++) {
    int32_t bound = valid_len;
    if (causal && row0 + row + 1 < bound)
      bound = row0 + row + 1;
    bfloat16 *__restrict p = g + (row / 8) * 64 + (row % 8) * 8;
    for (int j = 0; j < LKP; j++)
      if (col0 + j >= bound)
        p[(j / 8) * BLOCK_STRIDE + (j % 8)] = ninf;
  }
}

// One key chunk of the online softmax, after G holds this chunk's scores
// (log2(e)/sqrt(d) = c is applied here, not to Q):
//   if any row's chunk max > m + TAU/c:   m' = max(m, rowmax G), r = exp2((m - m')c),
//                                         O *= r, l *= r, m = m'
//   G = exp2(G c - m c);  l += rowsum(G)
// up = m (bf16), sp = l (fp32, replicated), gp = O (bf16, [TQ, DFULL]).
void softmax_step(bfloat16 *__restrict g, bfloat16 *__restrict up,
                  float *__restrict sp, bfloat16 *__restrict gp) {
  SET_ROUNDING();
  using V = aie::vector<bfloat16, 32>;
  const float c = float(log2e);
  row_max(g, s_newmax);
  bool resc = false;
  const V tau = aie::broadcast<bfloat16, 32>((bfloat16)(TAU / c));
  for (int i = 0; i < TQ; i += 32)
    resc |= !aie::gt(aie::load_v<32>(s_newmax + i),
                     aie::add(aie::load_v<32>(up + i), tau)).empty();
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
    for (int rh = 0; rh < TQ / 32; rh++) {
      // O *= r and l *= r, with the same bf16 r
      replicate_rows32(s_resc + rh * 32, s_rep);
      scale_rows<DFULL>(gp + rh * 256, s_rep);
      float *__restrict pl = sp + rh * 256;
      for (int j = 0; j < 256; j += 16) {
        aie::accum<accfloat, 16> ra;
        ra.from_vector(aie::load_v<16>(s_rep + j));
        aie::store_v(pl + j, aie::mul(aie::load_v<16>(pl + j), ra.to_vector<float>())
                                 .template to_vector<float>());
      }
      // -m*c for the exponentials
      replicate_rows32(up + rh * 32, s_rep);
      widen_scale(s_rep, s_nuc + rh * 256, -c, 256);
    }
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
void finalize(float *__restrict sp, bfloat16 *__restrict gp) {
  SET_ROUNDING();
  for (int rh = 0; rh < TQ / 32; rh++) {
    for (int j = 0; j < 256; j += 16) {
      aie::accum<accfloat, 16> a;
      a.from_vector(aie::load_v<16>(sp + rh * 256 + j));
      aie::store_v(s_rep + j, aie::inv(a.template to_vector<bfloat16>()));
    }
    scale_rows<DFULL>(gp + rh * 256, s_rep);
  }
}

} // extern "C"
