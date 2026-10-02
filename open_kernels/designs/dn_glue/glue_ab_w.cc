// cache-bust: 1791100001 (IRON hashes only this file, not included headers)
#include "dn_glue.h"

// The alpha / beta projection for MORE than 32 value heads (the 27B's 48): the packer writes
// it [hid, 64] with columns heads..63 zero (`transpose` with dst_rows 64), so a 4 KB side
// element is 32 rows x 64 bf16 and the accumulator is two 32-lane halves. Everything else is
// glue_ab_e.cc's contract: `xn` is ONE 4 KB half of the layer-entry norm output, `tile`
// restarts at 0 in every half, and `first` (the host loop's, set for the first half only)
// resets the accumulator on that half's first tile.
//
// A separate TU rather than a lane knob on glue_ab_tile: kV is also every other glue
// kernel's vector width, and the 32-lane path is in every shipped dense xclbin.
static constexpr unsigned kAbLanesW = 2 * kV;
static constexpr unsigned kAbRowsW = 4096 / (2 * kAbLanesW);     // 32 rows per 4 KB element

extern "C" {
void glue_ab_w(const bfloat16 *__restrict W, const bfloat16 *__restrict xn, float *__restrict acc,
               int tile, int first) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 lo, hi;
  if (first != 0 && tile == 0) {
    lo = aie::zeros<accfloat, kV>();
    hi = aie::zeros<accfloat, kV>();
  } else {
    lo.from_vector(aie::load_v<kV>(acc));
    hi.from_vector(aie::load_v<kV>(acc + kV));
  }
  const bfloat16 *x = xn + (unsigned)tile * kAbRowsW;
#pragma clang loop unroll(disable)
  for (unsigned r = 0; r < kAbRowsW; ++r) {
    lo = aie::mac(lo, aie::load_v<kV>(W + r * kAbLanesW), x[r]);
    hi = aie::mac(hi, aie::load_v<kV>(W + r * kAbLanesW + kV), x[r]);
  }
  aie::store_v(acc, lo.template to_vector<float>());
  aie::store_v(acc + kV, hi.template to_vector<float>());
}
}
