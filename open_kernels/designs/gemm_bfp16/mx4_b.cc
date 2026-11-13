//===- mx4_b.cc - an MXFP4 B tile -> the bfp16 B tile mm_bfp_acc.cc multiplies ---------===//
//
// GB_B=mx4 (gemm_bfp16.py): the weights stay MXFP4 in DDR -- 4352 B a 128 x 64 tile instead of
// bfp16's 9216 -- and each core turns its tile into the bfp16 one in L1 before the multiply. The
// conversion is EXACT: an MXFP4 value is c * 2^(e - 128) with c twice the e2m1 level, an integer
// in [-12, 12] (mx4_codes.h), and a bfp16ebs8 block decodes q * 2^(x - 133) (pack.py), so q = c,
// x = e + 5 for all eight values of a row's 8-K block, which shares the 32-K block's e.
//
//   MXFP4 tile, 4352 B (128 n rows x 64 k; pack.py pack_b_mx4):
//     [0   : 256 )  x = e + 5 per (32-K block kb32, row): [kb32 (2)][sp (8)][half (2)][r (8)],
//                   row = 16 sp + 8 half + r
//     [256 : 4352)  for sp, kb (8-K block, 8), half: 32 B of nibbles, value j (byte j / 2, the low
//                   nibble first) = row 16 sp + 8 half + j / 8, k = 8 kb + j % 8
//   -- each 32-B run is one 8 x 8 bfp16 sub-tile of mm_paired's B, in the order it pops them
//   (row pair, block column, the pair's first eight rows then its second).
//
//===----------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "mx4_codes.h"

#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 128
#endif
static_assert(DIM_K == 64 && DIM_N == 128, "the MXFP4 B tile is written for the 64 x 128 tile");

extern "C" void gb_mx4_b(const uint8_t *__restrict src, bfp16ebs8 *__restrict dst) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(dst);
  const uint8_t *__restrict nib = src + 256;
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (unsigned sp = 0; sp < DIM_N / 16; ++sp)
    for (unsigned kb = 0; kb < DIM_K / 8; ++kb)
#pragma clang loop unroll(full)
      for (unsigned half = 0; half < 2; ++half) {
        v64bfp16ebs8 b;
        b.mantissa = mx4_codes(nib);
        b.exponent = *(const v8int8 *)(src + (kb / 4) * 128 + sp * 16 + half * 8);
        out << aie::block_vector<bfp16ebs8, 64>(b);
        nib += 32;
      }
}
