//===- mm_bfp_acc.cc - bfp16 x bfp16 GEMM, fp32 partial sums ----------------===//
//
// Derived from ggml-xdna's kernels/bfp16_gemm/aie_kernels/aie2p/mm_bfp_acc.cc
// (Cyrus Attoun, https://github.com/Cyronius/ggml-xdna), itself derived from
// mlir-aie v1.4.2's aie_kernels/aie2p/mm_bfp.cc (Apache-2.0 WITH LLVM-exception,
// Copyright (C) Advanced Micro Devices, Inc.). Kept: the pair-interleaved block
// streams and the 2 x 2 mac_8x8_8x8T expansion. Dropped: the bf16 partial-sum
// entry points. gemm_epilogue's SwiGLU is npu-prefill-engine's out_epilogue mode 2.
//
//   mm_f32_first(A, B, C)   C  = A x B^T over one K chunk
//   mm_f32_mid(A, B, C)     C += A x B^T
//   gemm_epilogue(C, A, m)  m 1: silu(gate) * up of C's alternating 8-column blocks, as the next GEMM's A
//
// A and B are bfp16ebs8 tiles (eight values sharing an exponent byte, nine bytes a
// block) in the order recipes/pack.py's pack_bfp16 writes: the 8x8 sub-tiles go row
// pair by row pair (16 rows), block column by block column, the pair's first eight
// rows then its second. C is fp32, 8x8 blocks, block (z, j) at element
// (z * n/8 + j) * 64, each block row-major.
//
//===----------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

namespace {

constexpr unsigned rowA = DIM_M / 8, colA = DIM_K / 8, colB = DIM_N / 8;
static_assert(DIM_M % 16 == 0 && DIM_N % 16 == 0 && DIM_K % 16 == 0);

using acc_t = aie::accum<accfloat, 64>;

template <bool FIRST>
__attribute__((always_inline)) inline void mm_paired(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB,
                                                     float *pC) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);

  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (unsigned z = 0; z < rowA; z += 2) {
    float *c1 = pC + (z * colB) * 64;
    float *c2 = pC + ((z + 1) * colB) * 64;

    for (unsigned j = 0; j < colB; j += 2) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sA(pA);
      sA.seek(z * colA);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sB(pB);
      sB.seek(j * colA);

      acc_t c00, c01, c10, c11;
      if constexpr (FIRST) {
        c00 = aie::zeros<accfloat, 64>();
        c01 = aie::zeros<accfloat, 64>();
        c10 = aie::zeros<accfloat, 64>();
        c11 = aie::zeros<accfloat, 64>();
      } else {
        c00 = acc_t(aie::load_v<64>(c1));
        c01 = acc_t(aie::load_v<64>(c1 + 64));
        c10 = acc_t(aie::load_v<64>(c2));
        c11 = acc_t(aie::load_v<64>(c2 + 64));
      }

      aie::block_vector<bfp16ebs8, 64> A0, A1, B0, B1;
      AIE_LOOP_MIN_ITERATION_COUNT(2)
      AIE_LOOP_UNROLL(2)
      for (unsigned i = 0; i < colA; ++i) {
        A0 = sA.pop();
        A1 = sA.pop();
        B0 = sB.pop();
        B1 = sB.pop();
        c00 = mac_8x8_8x8T(A0, B0, c00);
        c01 = mac_8x8_8x8T(A0, B1, c01);
        c10 = mac_8x8_8x8T(A1, B0, c10);
        c11 = mac_8x8_8x8T(A1, B1, c11);
      }

      aie::store_v(c1, c00.template to_vector<float>());
      aie::store_v(c1 + 64, c01.template to_vector<float>());
      aie::store_v(c2, c10.template to_vector<float>());
      aie::store_v(c2 + 64, c11.template to_vector<float>());
      c1 += 128;
      c2 += 128;
    }
  }
}

} // namespace

extern "C" {

#ifdef F32_FIRST
void mm_f32_first(bfp16ebs8 *pA, bfp16ebs8 *pB, float *pC) { mm_paired<true>(pA, pB, pC); }
#endif

#ifdef F32_MID
void mm_f32_mid(bfp16ebs8 *pA, bfp16ebs8 *pB, float *pC) { mm_paired<false>(pA, pB, pC); }
#endif
}

#ifdef GLU_EPILOGUE
// core memory is full, so h is encoded into the held A buffer (one 64 x 64 A tile of the next GEMM) and copied into C
namespace {

using vf = aie::vector<float, 32>;
using vb = aie::vector<bfloat16, 32>;
using vi = aie::vector<int32_t, 32>;
using va = aie::accum<accfloat, 32>;

struct hl {
  vb h, l;
};

__attribute__((always_inline)) inline va acc(vf v) {
  va a;
  a.from_vector(v);
  return a;
}
__attribute__((always_inline)) inline va accb(vb v) {
  va a;
  a.from_vector(v);
  return a;
}
__attribute__((always_inline)) inline va accc(float c) { return acc(aie::broadcast<float, 32>(c)); }
__attribute__((always_inline)) inline vb bc(float c) { return aie::broadcast<bfloat16, 32>(c); }
// fp32 multiplies are emulated here, so values needing more than bf16's 8 bits ride as hi + lo
__attribute__((always_inline)) inline hl split(va a) {
  vb h = a.template to_vector<bfloat16>();
  return {h, aie::msc(a, h, bc(1.0f)).template to_vector<bfloat16>()};
}
__attribute__((always_inline)) inline va mac3(va c, hl a, hl b) {
  return aie::mac(aie::mac(aie::mac(c, a.h, b.h), a.h, b.l), a.l, b.h);
}
__attribute__((always_inline)) inline va mul3(hl a, hl b) {
  return aie::mac(aie::mac(aie::mul(a.h, b.h), a.h, b.l), a.l, b.h);
}

// g * u / (1 + e^-g) to about 2^-15, with e = e^-|g| so s = 1 / (1 + e) or e / (1 + e) never cancels
__attribute__((always_inline)) inline va swiglu32(vf g, vf u) {
  const hl gs = split(acc(g)), us = split(acc(u));
  const aie::mask<32> neg = aie::lt(gs.h, bc(0.0f));
  const vb ch = aie::max(aie::min(gs.h, bc(44.0f)), bc(-44.0f));
  const vf tp = aie::mac(aie::mac(aie::mul(ch, bc(1.4453125f)), ch, bc(-0.0026174591f)), gs.l, bc(1.4453125f))
                    .template to_vector<float>();
  const vf tm = aie::msc(aie::msc(aie::negmul(ch, bc(1.4453125f)), ch, bc(-0.0026174591f)), gs.l, bc(1.4453125f))
                    .template to_vector<float>();
  const vf tn = aie::select(tm, tp, neg);
  const vi k = aie::to_fixed<int32_t>(tn);
  const hl f = split(aie::sub(acc(tn), acc(aie::to_float<float>(k))));
  const vb r3 = aie::mac(accc(0.05591704f), f.h, bc(0.00956051f)).template to_vector<bfloat16>();
  const hl r2 = split(aie::mac(accc(0.24024981f), r3, f.h));
  const hl r1 = split(mac3(accc(0.69312197f), r2, f));
  const va p = mac3(accc(0.99999919f), r1, f);
  const vf e = aie::add(p.template to_vector<float>().template cast_to<int32_t>(), aie::upshift(k, 23))
                   .template cast_to<float>();
  const hl es = split(acc(e));
  const hl d = split(aie::add(acc(e), accc(1.0f)));
  const vb y0 = aie::msc(accc(1.4115f), d.h, bc(0.4705f)).template to_vector<bfloat16>();
  const vb e1 = aie::msc(aie::msc(accc(1.0f), d.h, y0), d.l, y0).template to_vector<bfloat16>();
  const va y1 = aie::mac(accb(y0), y0, e1);
  const hl y1s = split(y1);
  const vb e2 = aie::sub(accc(1.0f), mul3(d, y1s)).template to_vector<bfloat16>();
  const hl y = split(aie::mac(y1, y1s.h, e2));
  const hl num = {aie::select(bc(1.0f), es.h, neg), aie::select(bc(0.0f), es.l, neg)};
  const hl s = split(mul3(num, y));
  return mul3(split(mul3(gs, us)), s);
}

constexpr unsigned kOutWords = (DIM_M * DIM_N / 2) / 8 * 9 / 4;   // 1152: one bfp16 A tile, 64 x 64
static_assert(DIM_M == 64 && DIM_N == 128 && DIM_K == 64, "the glu epilogue is written for the 64 x 64 x 128 tile");

} // namespace

extern "C" void gemm_epilogue(float *__restrict pC, bfp16ebs8 *__restrict pA, int32_t mode) {
  if (mode != 1)
    return;
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
#ifdef EPI_SATURATE
  const auto sat = ::aie::swap_saturation(::aie::saturation_mode::EPI_SATURATE);
#endif
  // two 32-lane chains a block: one alone is latency-bound (257 bundles), four spill (180 a chain); this is 188
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(pA);
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned rg = 0; rg < rowA / 2; ++rg)
    for (unsigned cb = 0; cb < colB / 2; ++cb)
      for (unsigned zz = 0; zz < 2; ++zz) {
        const float *g = pC + ((2 * rg + zz) * colB + 2 * cb) * 64;
        const va lo = swiglu32(aie::load_v<32>(g), aie::load_v<32>(g + 64));
        const va hi = swiglu32(aie::load_v<32>(g + 32), aie::load_v<32>(g + 96));
        out << aie::concat(lo, hi).template to_vector<bfp16ebs8>();
      }
  // stream word s is C's (row s / 128, column s % 128), and C is 8x8 blocks
  const float *src = reinterpret_cast<const float *>(pA);
  AIE_LOOP_MIN_ITERATION_COUNT(kOutWords / 8)
  for (unsigned q = 0; q < kOutWords / 8; ++q) {
    const unsigned r = q / colB;
    aie::store_v(pC + ((r / 8) * colB + q % colB) * 64 + (r % 8) * 8, aie::load_unaligned_v<8>(src + q * 8));
  }
#ifdef EPI_SATURATE
  ::aie::set_saturation(sat);
#endif
}
#endif
