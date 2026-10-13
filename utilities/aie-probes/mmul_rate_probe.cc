// mmul_rate_probe.cc -- cycles per mmul by operand type with the operands resident in L1, for mmul_rate_probe.py.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VARIANT
#define VARIANT 0
#endif

namespace {
// the index comes from memory the compiler cannot see through, so the operand loads stay in the loop
volatile int32_t g_sel = 0;
template <typename T> __attribute__((always_inline)) inline T *hide(T *p) { return p + g_sel; }
} // namespace

extern "C" {
void mmul_rate(int32_t *__restrict out, int32_t iters) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
#if VARIANT == 0 || VARIANT == 4
  // 0: native bf16 4x8x8; 4: the 8x8x8 shape, which the toolchain runs through bfp16 when the emulation macro is set
#if VARIANT == 0
  constexpr unsigned R = 4;
#else
  constexpr unsigned R = 8;
#endif
  using MM = aie::mmul<R, 8, 8, bfloat16, bfloat16, accfloat>;
  alignas(64) static bfloat16 A[8][2][R * 8];
  alignas(64) static bfloat16 B[8][2][64];
  MM c00, c01, c10, c11;
  {
    auto a0 = aie::load_v<R * 8>(&A[0][0][0]), a1 = aie::load_v<R * 8>(&A[0][1][0]);
    auto b0 = aie::load_v<64>(&B[0][0][0]), b1 = aie::load_v<64>(&B[0][1][0]);
    c00.mul(a0, b0); c01.mul(a0, b1); c10.mul(a1, b0); c11.mul(a1, b1);
  }
  for (int32_t it = 0; it < iters; ++it) {
    const bfloat16 *a = hide(&A[it & 7][0][0]);
    const bfloat16 *b = hide(&B[it & 7][0][0]);
    auto a0 = aie::load_v<R * 8>(a), a1 = aie::load_v<R * 8>(a + R * 8);
    auto b0 = aie::load_v<64>(b), b1 = aie::load_v<64>(b + 64);
    c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
  }
  aie::store_v((float *)out, c00.template to_vector<float>());
  aie::store_v((float *)out + R * 8, c01.template to_vector<float>());
  aie::store_v((float *)out + 2 * R * 8, c10.template to_vector<float>());
  aie::store_v((float *)out + 3 * R * 8, c11.template to_vector<float>());
#elif VARIANT == 1 || VARIANT == 3
  // 1: int8 x int8 8x8x8; 3: int8 x uint4 8x8x8 (B stays packed)
#if VARIANT == 1
  using TB = int8;
#else
  using TB = uint4;
#endif
  using MM = aie::mmul<8, 8, 8, int8, TB, acc32>;
  alignas(64) static int8 A[8][2][64];
  alignas(64) static uint8_t B[8][2][64];
  MM c00, c01, c10, c11;
  {
    auto a0 = aie::load_v<64>(&A[0][0][0]), a1 = aie::load_v<64>(&A[0][1][0]);
    auto b0 = aie::load_v<64>((const TB *)&B[0][0][0]), b1 = aie::load_v<64>((const TB *)&B[0][1][0]);
    c00.mul(a0, b0); c01.mul(a0, b1); c10.mul(a1, b0); c11.mul(a1, b1);
  }
  for (int32_t it = 0; it < iters; ++it) {
    const int8 *a = hide(&A[it & 7][0][0]);
    const TB *b = (const TB *)hide(&B[it & 7][0][0]);
    auto a0 = aie::load_v<64>(a), a1 = aie::load_v<64>(a + 64);
    auto b0 = aie::load_v<64>(b), b1 = aie::load_v<64>(b + 64);
    c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
  }
  aie::store_v((int32_t *)out, c00.template to_vector<int32_t>());
  aie::store_v((int32_t *)out + 64, c01.template to_vector<int32_t>());
  aie::store_v((int32_t *)out + 128, c10.template to_vector<int32_t>());
  aie::store_v((int32_t *)out + 192, c11.template to_vector<int32_t>());
#elif VARIANT == 2
  // bfp16ebs8 x bfp16ebs8 8x8x8, the call whole_array_bfp makes
  alignas(64) static uint8_t A[8][2][72];
  alignas(64) static uint8_t B[8][2][72];
  aie::accum<accfloat, 64> c00 = aie::zeros<accfloat, 64>(), c01 = c00, c10 = c00, c11 = c00;
  for (int32_t it = 0; it < iters; ++it) {
    const bfp16ebs8 *a = (const bfp16ebs8 *)hide(&A[it & 7][0][0]);
    const bfp16ebs8 *b = (const bfp16ebs8 *)hide(&B[it & 7][0][0]);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sA(a);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sB(b);
    aie::block_vector<bfp16ebs8, 64> a0 = sA.pop(), a1 = sA.pop(), b0 = sB.pop(), b1 = sB.pop();
    c00 = mac_8x8_8x8T(a0, b0, c00);
    c01 = mac_8x8_8x8T(a0, b1, c01);
    c10 = mac_8x8_8x8T(a1, b0, c10);
    c11 = mac_8x8_8x8T(a1, b1, c11);
  }
  aie::store_v((float *)out, c00.template to_vector<float>());
  aie::store_v((float *)out + 64, c01.template to_vector<float>());
  aie::store_v((float *)out + 128, c10.template to_vector<float>());
  aie::store_v((float *)out + 192, c11.template to_vector<float>());
#endif
}
}
