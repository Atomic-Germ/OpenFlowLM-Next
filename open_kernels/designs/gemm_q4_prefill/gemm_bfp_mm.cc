//===- gemm_bfp_mm.cc -------------------------------------------*- C++ -*-===//
//
// GQP_XBFP (OPEN-GEMM-T2, workstream B2 of the Bonsai 2 round-2 plan): the GEMM's matmul on
// operands that are already bfp16ebs8, so the core does no bf16 -> bfp16 conversion.
//
// The bf16 path (mm.cc built with AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16) converts every A and
// B sub-tile to bfp16 inside the k loop -- B after an 8x8 transpose and a multiply by one -- and
// then issues mac_8x8_8x8T. Here the dequant writes A in bfp16 (gemm_q4_dequant.h, GQD_BFPA) and
// the host writes the activations in bfp16 (block_host.cpp), in the layouts that mac reads:
//
//   A [64 rows x KC k]: 64-value block vectors, each 8 rows x 8 k (one 9-byte block -- 8 k
//     sharing an exponent -- per row), ordered [row block pair z/2][k block i][z % 2].
//   B [KC k x 32 tokens], transposed: block vectors of 8 tokens x 8 k (one block per token),
//     ordered [token block pair j/2][k block i][j % 2].
//   C [64 x 32] fp32: (8 x 8) blocks [z][j], row-major inside -- mm.cc's C tile, so gemm_c_tok.cc
//     reads it unchanged.
//
// The pair order is what lets the 2 x 2 loop read its two A and two B block vectors from two
// streams: a 576-bit block vector is only loadable through a FIFO stream, the core has two FIFO
// registers, and with four streams (one per operand, the natural order) Peano spilled the stream
// state to the stack every iteration -- 14 bundles per four macs.
//
// The mac, the 2 x 2 register blocking and the k order are mm.cc's, and the conversions it no
// longer does are the ones the producers now do with the same instruction and rounding mode, so
// C is the bf16 path's to the bit when the host's activation conversion matches the hardware's.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"

#ifndef GQP_KC
#define GQP_KC 128
#endif

static constexpr unsigned MM_M = 64, MM_N = 32, MM_RS = 8;
static constexpr unsigned ROW_B = MM_M / MM_RS;   // 8 row blocks
static constexpr unsigned COL_B = MM_N / MM_RS;   // 4 token blocks
static constexpr unsigned K_B = GQP_KC / MM_RS;   // 16 k blocks at KC 128

extern "C" {

// one source, two objects in the design (GQP_MM_ONLY / GQP_ZERO_ONLY), so each symbol is defined once
#ifndef GQP_MM_ONLY
void gqp_zero_c(float *__restrict c) {
  const aie::vector<float, 32> z = aie::zeros<float, 32>();
  AIE_LOOP_RANGE(64, 64)
  for (unsigned i = 0; i < MM_M * MM_N; i += 32) aie::store_v(c + i, z);
}
#endif

#ifndef GQP_ZERO_ONLY
void gqp_mm_bfp(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *__restrict pC) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned z = 0; z < ROW_B; z += 2) {
    float *__restrict pC1 = pC + (z * COL_B) * 64;
    float *__restrict pC2 = pC + ((z + 1) * COL_B) * 64;
    for (unsigned j = 0; j < COL_B; j += 2) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sa(pA), sb(pB);
      sa.seek(z * K_B);          // pair z / 2 starts at block (z / 2) * 2 K_B = z * K_B (z even)
      sb.seek(j * K_B);
      aie::accum<accfloat, 64> c00(aie::load_v<64>(pC1));
      aie::accum<accfloat, 64> c01(aie::load_v<64>(pC1 + 64));
      aie::accum<accfloat, 64> c10(aie::load_v<64>(pC2));
      aie::accum<accfloat, 64> c11(aie::load_v<64>(pC2 + 64));
      AIE_LOOP_RANGE(K_B, K_B)
      AIE_LOOP_UNROLL(2)
      for (unsigned i = 0; i < K_B; ++i) {
        const aie::block_vector<bfp16ebs8, 64> A0 = sa.pop();
        const aie::block_vector<bfp16ebs8, 64> A1 = sa.pop();
        const aie::block_vector<bfp16ebs8, 64> B0 = sb.pop();
        const aie::block_vector<bfp16ebs8, 64> B1 = sb.pop();
        c00 = mac_8x8_8x8T(A0, B0, c00);
        c01 = mac_8x8_8x8T(A0, B1, c01);
        c10 = mac_8x8_8x8T(A1, B0, c10);
        c11 = mac_8x8_8x8T(A1, B1, c11);
      }
      aie::store_v(pC1, c00.template to_vector<float>());
      aie::store_v(pC1 + 64, c01.template to_vector<float>());
      aie::store_v(pC2, c10.template to_vector<float>());
      aie::store_v(pC2 + 64, c11.template to_vector<float>());
      pC1 += 128;
      pC2 += 128;
    }
  }
}
#endif

}  // extern "C"
