// Gemma 4's per-layer input on the norm helper (gx.py, OPEN-FAMILY-GEMMA4). One entry per TU (IRON compiles
// a source once per ExternalFunction): PLE_OUT selects ple_out, PLE_SCALE ple_scale, else ple_act.
#include "ln.h"
#include "vecmath.h"

#ifndef PLE_D
#define PLE_D 256
#endif

extern "C" {
#if defined(PLE_SCALE)
// out half i = y_i * s: the layer's output scale alone (the 12B has no per-layer input)
void ple_scale(const float *__restrict y0, const float *__restrict y1, const float *__restrict s,
               float *__restrict o, int32_t i) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *y = i ? y1 : y0;
  const float sc = s[0];
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) aie::store_v(o + j, fscaleN<32>(aie::load_v<kV>(y + j), sc));
}
#elif !defined(PLE_OUT)
// e = [g f32[PLE_D] | in f32[PLE_D] | ...] -> o = [gelu_tanh(g) * in, f32[PLE_D] | junk]
void ple_act(const float *__restrict e, float *__restrict o) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict g = e;
  const float *__restrict in = e + PLE_D;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < PLE_D; j += kV) {
    const v32f x = aie::load_v<kV>(g + j);
    const v32f x3 = fmul32(fmul32(x, x), x);
    const v32f z2 = fscaleN<32>(fadd32(x, fscaleN<32>(x3, 0.044715f)), 2.0f * 0.7978845608028654f);
    aie::store_v(o + j, fmul32(fmul32(x, vsigmoidN<32>(z2)), aie::load_v<kV>(in + j)));
  }
}
#else
// out half i = (y_i + r_i) * s, s the f32 at the start of its element (the layer's output scale)
void ple_out(const float *__restrict y0, const float *__restrict y1, const float *__restrict s,
             const float *__restrict r0, const float *__restrict r1, float *__restrict o, int32_t i) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *y = i ? y1 : y0;
  const float *r = i ? r1 : r0;
  const float sc = s[0];
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV)
    aie::store_v(o + j, fscaleN<32>(fadd32(aie::load_v<kV>(y + j), aie::load_v<kV>(r + j)), sc));
}
#endif
}
