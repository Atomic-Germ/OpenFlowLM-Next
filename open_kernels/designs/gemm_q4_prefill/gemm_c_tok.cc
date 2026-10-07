//===- gemm_c_tok.cc --------------------------------------------*- C++ -*-===//
//
// GQP_YT (OPEN-GEMM-T2): the core's finished C tile, token-major, so the GEMM's output lands in
// DRAM as y[T, N] and the host needs no [N, T] -> [T, N] transpose.
//
// `acc` is mm.cc's C tile for the bfp16 mmul (r = t = 8): 64 weight rows x 32 tokens in (8 x 8)
// blocks, block (z, j) at (z * 4 + j) * 64, element [rr][tt] = (row 8z + rr, token 8j + tt).
// `out` is [32 tokens][64 rows]. Once per 256-row block group, against ~20 bands of work.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"

extern "C" {
void gqp_c_tok_major(const float *__restrict acc, float *__restrict out) {
  AIE_LOOP_RANGE(8, 8)
  for (unsigned z = 0; z < 8; ++z) {
    AIE_LOOP_UNROLL_FULL
    for (unsigned j = 0; j < 4; ++j) {
      // aie::transpose has no 64-lane 32-bit form: transpose the two 4-row halves (as int32
      // bits), each [4 rr][8 tt] -> [8 tt][4 rr], then zip them 4 at a time so that every
      // token's eight rows come out contiguous: tt 0..3 in the first vector, 4..7 in the second
      const float *__restrict blk = acc + (z * 4 + j) * 64;
      const aie::vector<int32_t, 32> t0 = aie::transpose(aie::vector_cast<int32_t>(aie::load_v<32>(blk)), 4, 8);
      const aie::vector<int32_t, 32> t1 = aie::transpose(aie::vector_cast<int32_t>(aie::load_v<32>(blk + 32)), 4, 8);
      const auto zz = aie::interleave_zip(t0, t1, 4);
      const aie::vector<float, 32> lo = aie::vector_cast<float>(zz.first), hi = aie::vector_cast<float>(zz.second);
      AIE_LOOP_UNROLL_FULL
      for (unsigned tt = 0; tt < 4; ++tt) {
        aie::store_v(out + (j * 8 + tt) * 64 + z * 8, lo.template extract<8>(tt));
        aie::store_v(out + (j * 8 + 4 + tt) * 64 + z * 8, hi.template extract<8>(tt));
      }
    }
  }
}
}
