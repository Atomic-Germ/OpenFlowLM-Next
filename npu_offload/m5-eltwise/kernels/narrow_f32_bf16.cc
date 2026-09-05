//===- narrow_f32_bf16.cc -----------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the GEMM's bf16 output epilogue.
// SPDX-License-Identifier: MIT
//
// WHY
// ---
// tasks/0044 measured `read out + bias` -- the host streaming the fp32 GEMM
// result out of write-combined XRT memory -- at 18.8% of a MiniLM encode:
// 679 MB per encode at 19.6 GB/s, the single largest host cost. The bytes
// exist because C leaves the array as fp32 and the host narrows it later.
//
// This narrows it on the core instead, so the DMA moves half as much.
//
// TRAP 2 IS NOT VIOLATED, and the distinction is the whole design.
// `CLAUDE.md` trap 2 forbids `output_dtype=bf16` on the matmul kernel because
// that re-rounds the accumulator at EVERY K step: 7.4e-3 against 1.21e-07.
// Here the matmul still accumulates into an fp32 buffer for the full K
// reduction, and this runs ONCE at the end. It is the same single rounding the
// host does today, moved upstream of the elementwise ops.
//
// ROUNDING MODE
// -------------
// This is the first kernel in the project written with the mode set. The AIE
// default is `aie::rounding_mode::floor` -- "always round towards negative
// infinity" -- which tasks/0044 Part 3 measured as a systematic downward bias
// worth 1.73x on GELU, 1.29x on softmax and 1.62x on LayerNorm, and which took
// LayerNorm's error against a numpy model of the same formula from 3.659e-03
// to 3.967e-05. A GEMM result narrowed under `floor` would carry the same
// one-sided bias into every downstream op.
//
// set_rounding writes a core control register, so it is once per call, not
// once per element.
//
// WHY A SEPARATE OUTPUT BUFFER RATHER THAN IN PLACE
// -------------------------------------------------
// The accumulator is a core-local fp32 Buffer and the destination is the bf16
// ObjectFifo object that the DMA drains. They are different memories and must
// be, because narrowing in place would leave the DMA reading fp32-sized
// strides over half-filled data. The pair costs no extra L1 overall: the
// accumulator needs no double buffering (filled and drained inside one
// iteration) while the C fifo halves, and those cancel exactly --
// 53,248 B at MiniLM's (64,64,48) and 40,960 B at bge-large's (64,64,32),
// both identical to the fp32 design they replace.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <unsigned N>
static inline void narrow_impl(const float *restrict acc,
                               bfloat16 *restrict out) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  // 16 lanes at a time: one 512-bit fp32 vector in, one 256-bit bf16 vector
  // out. Two independent chains so the store unit -- of which there is one,
  // against two load units (guide section 4c) -- is not waiting on a single
  // dependency chain.
  //
  // USE `accum::from_vector`, NOT the multiply-by-1.0f idiom.
  //
  // `aie::to_vector<T>()` is defined on ACCUMULATORS, not on vectors: Peano
  // rejects `aie::to_vector<bfloat16>(vec)` with "constraints not satisfied
  // [with TR = bfloat16, T = aie::vector<float, 16>]". gelu_poly.cc gets its
  // accumulator by multiplying by 1.0f and documents that as the way. **That
  // is expensive here and it was measured, not assumed** -- fp32 multiply on
  // aie2p is EMULATED, so `aie::mul(v, 1.0f)` lowers to real work:
  //
  //   mul-by-1.0f form   34x vmul.f / vadd.f for 64 elements
  //   from_vector form    0x vmul.f / vadd.f, 17 instructions total,
  //                       4x `vst.conv.bf16.fp32` -- the narrowing rides
  //                       along in the STORE and costs nothing
  //
  // (llvm-objdump on both, aie2p, -O2. The multiply by 1.0f is exact either
  // way; it is the emulation that is not free.)
  auto it_in = aie::begin_restrict_vector<16>((float *)acc);
  auto it_out = aie::begin_restrict_vector<16>(out);

  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned i = 0; i < N; i += 32) {
    aie::accum<accfloat, 16> a0, a1;
    a0.from_vector(*it_in++);
    a1.from_vector(*it_in++);
    *it_out++ = a0.to_vector<bfloat16>();
    *it_out++ = a1.to_vector<bfloat16>();
  }

  event1();
}

extern "C" {

// One entry point per production tile size. m*n for the two shipped
// geometries: MiniLM/bge-small tile_n=48 -> 64*48 = 3072; bge-large tile_n=32
// -> 64*32 = 2048; and tile_n=16 for the attention-capable geometry 0043
// enumerated -> 64*16 = 1024. All are multiples of 32.
void narrow_3072_f32_bf16(const float *restrict acc, bfloat16 *restrict out) {
  narrow_impl<3072>(acc, out);
}

void narrow_2048_f32_bf16(const float *restrict acc, bfloat16 *restrict out) {
  narrow_impl<2048>(acc, out);
}

void narrow_1024_f32_bf16(const float *restrict acc, bfloat16 *restrict out) {
  narrow_impl<1024>(acc, out);
}

}  // extern "C"
