//===- mm_atb_stock.cc ------------------------------------------*- C++ -*-===//
//
// The asymmetric-tile-buffering dataflow (n32_core_atb.py) driven by mlir-aie's
// STOCK bf16 x bfp16 microkernel instead of config1's hand-scheduled one.
//
// config1's kernel is tuned for chess (chess_storage register pinning). Built
// with Peano it returns NaN/inf in ~2/3 of C, in a pattern that repeats every
// quarter-tile call -- a codegen problem, not a dataflow one. The two templates
// below are copied verbatim from mlir-aie aie_kernels/aie2p/mm_bfp_mixed.cc
// (mlir-aie 1.4.2, Apache-2.0 WITH LLVM-exception), which mlir-aie CI runs on
// npu2 with Peano. The ATB dataflow already delivers A to the core in that
// kernel's layout (8x8 row-major blocks, row-block major) and uses the same
// blocked C layout; only B's host packing differs (see bfp_gemm_bench.py
// atbs_layout).
//
// Each call multiplies one (DIM_M/4 x DIM_K) A sub-tile into one quarter of the
// (DIM_M x DIM_N) C tile; the quarter index is a counter the zero kernel resets
// at the start of every output tile. Round-to-nearest-even is set for the call
// (the stock kernel leaves floor in force -- see ../wam/mm_bfp_mixed.cc).
//
//===----------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8);
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

// This kernel is a variation of the conventional matrix multiplications in the
// repo that uses different datatypes for the A and B and performs a conversion
// for the A matrix. This kernel should be followed along with the equivalent on
// in bfp16 only on mm.cc
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t>
void matmul_vectorized_2x2_bfp16_bf16(const bfloat16 *__restrict pA,
                                      const bfp16ebs8 *__restrict pB,
                                      bfloat16 *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned z = 0; z < rowA; z += 2) {
    bfloat16 *__restrict pC1 = pC + (z * colB + 0) * sizeC;
    bfloat16 *__restrict pC2 = pC + ((z + 1) * colB + 0) * sizeC;

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        const bfloat16 *__restrict pA1 = pA + (z * colA + 0) * sizeA;
        const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA + 0) * sizeA;

        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
        // For non transposed matrix
        // pB1bfp16.seek(j);
        // pB2bfp16.seek(j + 1);
        pB1bfp16.seek(j * colA);
        pB2bfp16.seek((j + 1) * colA);

        aie::vector<bfloat16, sizeA> A0;
        aie::vector<bfloat16, sizeA> A1;
        aie::block_vector<bfp16ebs8, sizeB> B0;
        aie::block_vector<bfp16ebs8, sizeB> B1;

        aie::accum<accfloat, sizeC> accC00(aie::load_v<sizeC>(pC1));
        aie::accum<accfloat, sizeC> accC01(aie::load_v<sizeC>(pC1 + sizeC));
        aie::accum<accfloat, sizeC> accC10(aie::load_v<sizeC>(pC2));
        aie::accum<accfloat, sizeC> accC11(aie::load_v<sizeC>(pC2 + sizeC));

        aie::accum<accfloat, 64> accA0;
        aie::accum<accfloat, 64> accA1;

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<sizeA>(pA1);
            pA1 += sizeA;
            A1 = aie::load_v<sizeA>(pA2);
            pA2 += sizeA;

            // Convert A0 into bfp16
            accA0 = A0;
            // Convert A1 into bfp16 through a different path (see bfp
            // conversion example)
            accA1 = mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(),
                                           broadcast_one_to_v32bfloat16()));

            // For non transposed matrix
            // B0 = pB1bfp16.pop_seek(colB - 1);
            // B1 = pB2bfp16.pop_seek(colB - 1);
            B0 = pB1bfp16.pop();
            B1 = pB2bfp16.pop();

            accC00 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B0, accC00);
            accC01 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B1, accC01);
            accC10 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B0, accC10);
            accC11 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B1, accC11);
          }

        aie::store_v(pC1, accC00.template to_vector<bfloat16>());
        pC1 += sizeC;
        aie::store_v(pC1, accC01.template to_vector<bfloat16>());
        pC1 += sizeC;
        aie::store_v(pC2, accC10.template to_vector<bfloat16>());
        pC2 += sizeC;
        aie::store_v(pC2, accC11.template to_vector<bfloat16>());
        pC2 += sizeC;
      }
  }
}

extern "C" {

extern int oflm_atbs_quarter;

#ifdef MATMUL_ONLY
__attribute__((section(".data"))) int oflm_atbs_quarter = 0;

void matmul_atb_stock(bfloat16 *__restrict pA, bfp16ebs8 *__restrict pB,
                      bfloat16 *__restrict pC) {
  constexpr int r = 8, s = 8, t = 8;
  constexpr int m = DIM_M / 4, k = DIM_K, n = DIM_N;
  static_assert(m % (2 * r) == 0 && k % s == 0 && n % (2 * t) == 0);
  bfloat16 *pCq = pC + oflm_atbs_quarter * m * n;
  oflm_atbs_quarter = (oflm_atbs_quarter + 1) & 3;
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  matmul_vectorized_2x2_bfp16_bf16<m / r, k / s, n / t, r, s, t>(pA, pB, pCq);
  aie::set_rounding(saved);
}
#endif

#ifdef ZERO_ONLY
void zero_kernel_bf16(bfloat16 *__restrict cOut) {
  oflm_atbs_quarter = 0;
  zero_vectorized<bfloat16, DIM_M, DIM_N>(cOut);
}
#endif
}
