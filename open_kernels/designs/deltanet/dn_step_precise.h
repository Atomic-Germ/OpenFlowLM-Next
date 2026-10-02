#pragma once
// Standalone diagnostic recurrence with three-component FP32 products.
// Keep the legacy BO/scratch ABI: each 256-BF16 scratch holds 128 floats.
// k_hl/q_hl carry compensated-sum errors; delta_hl carries the FP32 delta.
#include "vecmath_precise.h"

static constexpr unsigned kD = 128, kV = 16, kSliceRows = 16, kNBlk = 8;

static inline void dn_compensated_add(v16f &sum, v16f &error, const v16f &value) {
  const auto corrected = fsubN<16>(value, error);
  const auto next = faddN<16>(sum, corrected);
  error = fsubN<16>(fsubN<16>(next, sum), corrected);
  sum = next;
}

static inline void dn_pass1_slice(const float *__restrict S, const float *__restrict vec,
                                  float *__restrict t, bfloat16 *__restrict k_hl,
                                  bfloat16 *__restrict q_hl, unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *error = reinterpret_cast<float *>(k_hl);
  (void)q_hl;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    auto sum = blk == 0 ? aie::zeros<float, kV>() : aie::load_v<kV>(t + j);
    auto err = blk == 0 ? aie::zeros<float, kV>() : aie::load_v<kV>(error + j);
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kSliceRows; ++i) {
      const auto k = aie::broadcast<float, kV>(vec[blk * kSliceRows + i]);
      dn_compensated_add(sum, err, precise_mulN<16>(aie::load_v<kV>(S + i * kD + j), k));
    }
    aie::store_v(t + j, sum);
    aie::store_v(error + j, err);
  }
}

static inline void dn_pass2_slice(const float *__restrict S, float *__restrict Sout,
                                  const float *__restrict vec, const float *__restrict t,
                                  float *__restrict o, const bfloat16 *__restrict k_hl,
                                  const bfloat16 *__restrict q_hl,
                                  bfloat16 *__restrict delta_hl, unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  (void)k_hl;
  // The public legacy signature is const; this diagnostic owns this scratch.
  float *error = reinterpret_cast<float *>(const_cast<bfloat16 *>(q_hl));
  float *delta = reinterpret_cast<float *>(delta_hl);
  const auto decay = aie::broadcast<float, kV>(vec[384]);
  const auto beta = aie::broadcast<float, kV>(vec[385]);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    if (blk == 0) {
      const auto u = fsubN<16>(aie::load_v<kV>(vec + 2 * kD + j), precise_mulN<16>(decay, aie::load_v<kV>(t + j)));
      aie::store_v(delta + j, precise_mulN<16>(beta, u));
    }
    const auto d = aie::load_v<kV>(delta + j);
    auto sum = blk == 0 ? aie::zeros<float, kV>() : aie::load_v<kV>(o + j);
    auto err = blk == 0 ? aie::zeros<float, kV>() : aie::load_v<kV>(error + j);
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kSliceRows; ++i) {
      const auto k = aie::broadcast<float, kV>(vec[blk * kSliceRows + i]);
      const auto q = aie::broadcast<float, kV>(vec[kD + blk * kSliceRows + i]);
      const auto sn = faddN<16>(precise_mulN<16>(decay, aie::load_v<kV>(S + i * kD + j)), precise_mulN<16>(k, d));
      aie::store_v(Sout + i * kD + j, sn);
      dn_compensated_add(sum, err, precise_mulN<16>(sn, q));
    }
    aie::store_v(error + j, err);
    if (blk == kNBlk - 1)
      sum = precise_mulN<16>(sum, aie::broadcast<float, kV>(0.088388347648318f));
    aie::store_v(o + j, sum);
  }
}
