// Private 5120-wide Q/gate/K/V preparation. Both passes use production sums.
#define GEMV_Q4_CONSISTENT_SUM 1
#include "gemv_tab.h"
#include "add_rne.h"
extern "C" void refine_prep(const bfloat16 *__restrict e,
                             uint8_t *__restrict tab, int32_t i) {
  const unsigned K = 5120, b0 = 32u * (unsigned)i;
  const float *x = (const float *)e;
  gemv_q4_prep_f32_direct_blocks(x, tab, K, b0, 32);
  const int16_t *xi = (const int16_t *)tab;
  const int32_t *sh = (const int32_t *)(tab + 2 * K);
  alignas(32) float low[32];
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 32; ++j) {
    const unsigned kb = b0 + j;
    alignas(32) float reconstructed[32];
    aie::store_v(reconstructed, aie::to_float<float>(aie::load_v<32>(xi + kb * 32), sh[kb]));
#pragma clang loop unroll(disable)
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint32_t original_bits, reconstructed_bits;
      __builtin_memcpy(&original_bits, x + j * 32 + lane, 4);
      __builtin_memcpy(&reconstructed_bits, reconstructed + lane, 4);
      const uint32_t difference = add_rne(original_bits, reconstructed_bits ^ 0x80000000u);
      __builtin_memcpy(low + lane, &difference, 4);
    }
    gemv_q4_prep_f32_direct_blocks(low, tab + 11520, K, kb, 1);
  }
}
