//===- mm_dit.cc ------------------------------------------------*- C++ -*-===//
//
// The microkernel of dit_gemm.py: one (DIM_M/4 x DIM_K) bf16 A sub-tile times a
// (DIM_K x DIM_N) bfp16ebs8 B tile, accumulated into one quarter of the
// (DIM_M x DIM_N) bf16 C tile that stays resident in L1 across the whole K walk.
//
// zero_vectorized and matmul_vectorized_2x2_bfp16_bf16 are copied verbatim from
// mlir-aie aie_kernels/aie2p/mm_bfp_mixed.cc (mlir-aie 1.4.2, Apache-2.0 WITH
// LLVM-exception) -- the stock kernel mlir-aie CI runs on npu2 with Peano. Two
// deliberate departures from upstream, both measured
// (utilities/dit-gemm-bench/README.md):
//
//   - Round-to-nearest-even for the call. Upstream leaves the core's default,
//     floor, in force; both in-kernel conversions (A bf16 -> bfp16, and the fp32
//     accumulator -> bf16 after every DIM_K of K) then round toward -inf and the
//     bias accumulates over K: rel_fro 8.3e-02 with a -7% mean offset at
//     K = 3072, against 1.06e-02 with this fix.
//   - The quarter-tile counter lives in .data with external linkage and the zero
//     kernel resets it at the start of every output tile. A static in .bss is
//     only as initialised as the loader makes it.
//
// Why not the asymmetric-tile-buffering example's own microkernel: it is
// scheduled for chess (chess_storage register pinning, closed-source compiler);
// Peano spills its twelve 9-byte block vectors and C comes back NaN/inf in ~2/3
// of every quarter tile.
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

extern int oflm_dit_quarter;

#ifdef MATMUL_ONLY
__attribute__((section(".data"))) int oflm_dit_quarter = 0;

void dit_matmul_quarter(bfloat16 *__restrict pA, bfp16ebs8 *__restrict pB,
                      bfloat16 *__restrict pC) {
  constexpr int r = 8, s = 8, t = 8;
  constexpr int m = DIM_M / 4, k = DIM_K, n = DIM_N;
  static_assert(m % (2 * r) == 0 && k % s == 0 && n % (2 * t) == 0);
  bfloat16 *pCq = pC + oflm_dit_quarter * m * n;
  oflm_dit_quarter = (oflm_dit_quarter + 1) & 3;
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  matmul_vectorized_2x2_bfp16_bf16<m / r, k / s, n / t, r, s, t>(pA, pB, pCq);
  aie::set_rounding(saved);
}
#endif

#ifdef EPI_ONLY
// SwiGLU epilogue, in place, on a finished C tile whose 128 columns are 64 gate + the
// matching 64 up columns (pack.py interleave_swiglu): columns 0..63 become
// silu(gate) * up, 64..127 are left as they are (the drain sends them to scratch).
// The tile is 4 quarters x 4 row blocks, each a contiguous run of 16 8x8 column blocks,
// so gate is the first 512 values of every 1024-value run and up the last 512, at
// matching positions. Runs only on tiles in the epilogue range: column group
// t % rtp[2] >= rtp[3] (dit_gemm.py RTP layout). silu(g) = h (1 + tanh h), h = g / 2;
// aie2p's tanh is approximate (1.4-3.5% rms below |x| = 2) and the leading 1 keeps
// its effect under ~0.5% (utilities/aie-probes/vecmath_probe.py).
void dit_swiglu_epi(bfloat16 *__restrict c, int32_t *__restrict rtp, int32_t t) {
  if (t % rtp[2] < rtp[3])
    return;
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  using VB = aie::vector<bfloat16, 32>;
  for (int run = 0; run < DIM_M * DIM_N / 1024; run++) {
    bfloat16 *__restrict g = c + run * 1024;
    const bfloat16 *__restrict u = g + 512;
    for (int i = 0; i < 512; i += 32) {
      VB h = aie::mul(aie::load_v<32>(g + i), (bfloat16)0.5f).to_vector<bfloat16>();
      aie::accum<accfloat, 32> ha;
      ha.from_vector(h);
      aie::vector<float, 32> hf = ha.to_vector<float>();
      VB th;
      th.insert(0, aie::tanh<bfloat16>(hf.extract<16>(0)));
      th.insert(1, aie::tanh<bfloat16>(hf.extract<16>(1)));
      VB sg = aie::mac(ha, h, th).to_vector<bfloat16>();
      aie::store_v(g + i, aie::mul(sg, aie::load_v<32>(u + i)).to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved);
}
#endif

#ifdef ZERO_ONLY
void dit_zero_c(bfloat16 *__restrict cOut) {
  oflm_dit_quarter = 0;
  zero_vectorized<bfloat16, DIM_M, DIM_N>(cOut);
}
#endif
}
