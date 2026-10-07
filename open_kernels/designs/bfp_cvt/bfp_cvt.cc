// bfp_cvt: the hardware's fp32-accumulator -> bfp16ebs8 conversion, as the bfp16 GEMM applies it to
// its bf16 operands (mm.cc's AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 path: accum<accfloat, 64>
// from the bf16 vector, then to_v64bfp16ebs8, under rounding_mode::conv_even). One 64-value vector
// in, its 72 bytes out, raw -- so the host can read the block layout and the rounding off the bytes.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef BFP_VECS
#define BFP_VECS 64
#endif

extern "C" {

void bfp_cvt(const bfloat16 *__restrict x, bfp16ebs8 *__restrict y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(y);
  for (unsigned i = 0; i < BFP_VECS; ++i) {
    aie::accum<accfloat, 64> acc(aie::load_v<64>(x + i * 64));
    out << acc.to_vector<bfp16ebs8>();
  }
}

}  // extern "C"
