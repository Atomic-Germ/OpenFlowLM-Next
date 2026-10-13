//===- mm_bfp_acc.cc - bfp16 x bfp16 GEMM, fp32 partial sums ----------------===//
//
// Derived from ggml-xdna's kernels/bfp16_gemm/aie_kernels/aie2p/mm_bfp_acc.cc
// (Cyrus Attoun, https://github.com/Cyronius/ggml-xdna), itself derived from
// mlir-aie v1.4.2's aie_kernels/aie2p/mm_bfp.cc (Apache-2.0 WITH LLVM-exception,
// Copyright (C) Advanced Micro Devices, Inc.). Kept: the pair-interleaved block
// streams and the 2 x 2 mac_8x8_8x8T expansion. Dropped: the bf16 partial-sum
// entry points and the output epilogue.
//
//   mm_f32_first(A, B, C)   C  = A x B^T over one K chunk
//   mm_f32_mid(A, B, C)     C += A x B^T
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
