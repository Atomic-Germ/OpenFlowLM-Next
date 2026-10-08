//===- conv.cc --------------------------------------------------*- C++ -*-===//
//
// dit_conv's tile prologue. The matmul is dit_gemm's (mm_dit.cc, built with
// -DMATMUL_ONLY into the same core); this file starts every output tile.
//
// dit_conv_bias: the first B object of every tile carries the tile's 128 output
// channel biases as raw bf16 in its first 256 bytes (conv_pack.py pack_conv). C starts
// as that row broadcast down all 128 pixels instead of zero, so the conv's bias
// costs no pass of its own. Also resets mm_dit.cc's quarter counter, which
// dit_gemm's zero kernel does there.
//
// C tile layout (mm_dit.cc): [quarter 4][row block 4][column block 16][8 rows][8
// columns], so every 8x8 block of column block j is bias[8j .. 8j+7] on each row.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

extern "C" {

extern int oflm_dit_quarter;

void dit_conv_bias(bfloat16 *__restrict b, bfloat16 *__restrict c) {
  oflm_dit_quarter = 0;
  for (int j = 0; j < DIM_N / 8; j++) {
    aie::vector<bfloat16, 8> v8 = aie::load_v<8>(b + 8 * j);
    aie::vector<bfloat16, 16> v16 = aie::concat(v8, v8);
    aie::vector<bfloat16, 32> v32 = aie::concat(v16, v16);
    aie::vector<bfloat16, 64> blk = aie::concat(v32, v32);
    for (int rb = 0; rb < DIM_M / 8; rb++)
      aie::store_v(c + (rb * (DIM_N / 8) + j) * 64, blk);
  }
}

}
