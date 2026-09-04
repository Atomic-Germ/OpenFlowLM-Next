//===- narrow_i32_bf16.cc -----------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the int8 GEMM's bf16 output epilogue.
// SPDX-License-Identifier: MIT
//
// WHY THIS EXISTS, AND WHY IT DID NOT BEFORE
// ------------------------------------------
// tasks/0048 established that the bf16 GEMM is bound by the NUMBER of tile
// iterations rather than by bytes, and retired the byte levers on that basis:
// `ffn_up` and `ffn_down` have identical MACs, differ 1.5x in bytes, and
// measured 1.8% apart.
//
// tasks/0080 re-ran exactly that discriminating pair on the int8 datapath and
// it DIVERGES. Fitting both cost models over the four production shapes:
//
//   datapath   traffic model (0010)      iteration model (0048)
//   bf16       R2 0.709, worst 20.1%     R2 0.994, worst  2.6%
//   int8       R2 0.987, worst  5.3%     R2 0.568, worst 36.4%
//
// The two models swap places with the datapath. int8 made the arithmetic cheap
// enough that 0010's `t = 627 us + traffic / ~28 GB/s` governs again -- so
// bytes stopped being free, and C is 61% of the traffic on three of the four
// production shapes.
//
// WHY int32 -> bf16 NEEDS NO EXTRA CORE INPUT
// -------------------------------------------
// This is the reason the lever is reachable at all. Applying the per-column
// `wscale[j]` on the core would need a THIRD input stream, and CLAUDE.md trap
// 3b is explicit that every core tile is already 2/2 in -- the wall that forced
// tasks/0020 to pack gamma+beta into one buffer. But narrowing int32 to bf16 is
// a pure FORMAT conversion with no operand: the host still applies the rank-1
// `sa[i] * wscale[j]` exactly as it does today, reading half the bytes.
//
// WHY IT COSTS NO ACCURACY, MEASURED BEFORE THIS FILE WAS WRITTEN
// ---------------------------------------------------------------
// `npuembed --sim-c-bf16` rounds the accumulator in the host dequantiser
// exactly as this kernel does, which prices the design without building it:
//
//   MiniLM-L6   int32 C 1.178e-03  ->  bf16 C 1.161e-03
//   bge-large   int32 C 4.475e-03  ->  bf16 C 4.436e-03
//
// Both marginally BETTER, i.e. indistinguishable. The mechanism is that the
// int32 accumulator's low bits sit below the int8 quantisation noise already
// present in the operands, so 8 mantissa bits discard nothing that was signal.
// Contrast the bf16 datapath, where narrowing C cost a real 1.38-1.52x on
// 1-cos (tasks/0045) because there the operands were not already quantised.
//
// TRAP 2 IS NOT VIOLATED
// ----------------------
// Same argument as narrow_f32_bf16.cc, and stronger here: the matmul
// accumulates in int32 for the whole K reduction, which is not merely accurate
// but EXACT and order-independent -- integer addition associates. This runs
// once, after the reduction has finished.
//
// ROUNDING MODE
// -------------
// `conv_even` for the reason tasks/0044 Part 3 measured: the AIE default is
// `floor`, a systematic downward bias rather than symmetric noise. A C tensor
// narrowed under `floor` would carry that bias into every downstream op.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <unsigned N>
static inline void narrow_impl(const int32_t *restrict acc,
                               bfloat16 *restrict out) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  // `aie::to_float` is the fixed-point-to-float converter, and on Gen2 it
  // accepts int32 with a free choice of result type -- so int32 -> bf16 is one
  // operation, not a widen followed by a narrow. shift = 0: the accumulator
  // holds integers, the point is at the bottom.
  //
  // Two independent chains, for the reason narrow_f32_bf16.cc documents: there
  // is one store unit against two load units, so a single dependency chain
  // leaves it waiting.
  auto it_in = aie::begin_restrict_vector<16>((int32_t *)acc);
  auto it_out = aie::begin_restrict_vector<16>(out);

  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned i = 0; i < N; i += 32) {
    aie::vector<int32_t, 16> v0 = *it_in++;
    aie::vector<int32_t, 16> v1 = *it_in++;
    *it_out++ = aie::to_float<bfloat16>(v0, 0);
    *it_out++ = aie::to_float<bfloat16>(v1, 0);
  }

  event1();
}

extern "C" {

// One entry point per production tile size, mirroring narrow_f32_bf16.cc:
// m*n for tile_n=48 (MiniLM, bge-small, bge-base, nomic) -> 64*48 = 3072;
// tile_n=32 (bge-large) -> 64*32 = 2048; tile_n=16 (the attention-capable
// geometry tasks/0043 enumerated) -> 64*16 = 1024.
//
// AND 4096, WHICH ONLY THE int8 DATAPATH CAN REACH. tile_n=64 needs
// 2*(64*64*in + 64*64*in + 64*64*4) bytes of L1: 65,536 at bf16's 2-byte
// operands, which is over the 63 KB budget and is exactly why CLAUDE.md's
// geometry section records 64 as the width bge-large cannot have. At int8's
// 1-byte operands it is 49,152. bge-large's N in {1024, 3072, 4096} are all
// multiples of 64*8 = 512, so it needs no padding either -- it has simply been
// running at half the tile width because the geometry was inherited from a
// budget that no longer applies (tasks/0081).
void narrow_4096_i32_bf16(const int32_t *restrict acc, bfloat16 *restrict out) {
  narrow_impl<4096>(acc, out);
}

void narrow_3072_i32_bf16(const int32_t *restrict acc, bfloat16 *restrict out) {
  narrow_impl<3072>(acc, out);
}

void narrow_2048_i32_bf16(const int32_t *restrict acc, bfloat16 *restrict out) {
  narrow_impl<2048>(acc, out);
}

void narrow_1024_i32_bf16(const int32_t *restrict acc, bfloat16 *restrict out) {
  narrow_impl<1024>(acc, out);
}

}  // extern "C"
