//===- vew.cc ----------------------------------------------------*- C++ -*-===//
//
// vae_ew's kernels (vae_ew.py): FLUX.2's VAE decoder's elementwise ops on NHWC
// activations, one element (VEW_EL bf16 = whole pixels x C channels) per call,
// round-to-nearest-even.
//
//   vew_begin      set round-to-nearest-even, zero the channel sums (every dispatch)
//   vew_stats      accumulate per-channel sum and sum of squares of an element
//   vew_stats_out  per-group (32 groups of C/32 channels) partial sums -> an element:
//                  64 floats [g0 sum, g0 sumsq, g1 sum, ...] as raw bits, then zeros
//   vew_param      the GroupNorm parameter block, one element per call: j = 0 gamma[C]
//                  at 0 and beta[C] at 512; j = 1..16 the 16 cores' partial sums; after
//                  j = 16, a = gamma * rstd and b = beta - mean * a per channel
//   vew_apply      y = x * a + b, then SiLU (R_SILU): GroupNorm(+SiLU), rounded to bf16
//                  once before the SiLU as diffusers' bf16 decode does
//   vew_add        y = a + b (the residual); with R_ADD_STATS, accumulates y's stats
//   vew_rgba       x: 1024 pixels x 4 channels (the decoder's output, 3 real) ->
//                  y: 1024 RGBA8 pixels (4096 bytes; the element's second half unused)
//
// Arithmetic as dit_ew's: bf16 x bf16 products into fp32 accumulators, a float that
// scales a vector split into bf16 hi + lo (aie2p has no native fp32 vector multiply).
// x*1 and x*x are exact products, so the statistics are fp32 sums of exact terms.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef VEW_EL
#define VEW_EL 4096
#endif
constexpr int EL = VEW_EL;
constexpr int CMAX = 512;
constexpr int GROUPS = 32;
constexpr float EPS = 1e-6f;

// Runtime parameter slots (vae_ew.py RTP_*).
enum {
  R_NPAR, R_CNT_STATS, R_STATS_OUT, R_CNT_APPLY, R_CNT_ADD, R_ADD_STATS_OUT, R_CNT_RGBA,
  R_C, R_SILU, R_ADD_STATS, R_NPIX
};

// Per-core state (vae_ew.py `par`, floats): gamma, beta (bf16 bits in the first
// half of their float slots), a_hi, a_lo (bf16), b (float), group sums, channel sums.
constexpr int P_GAMMA = 0;                 // bf16[CMAX]   (in float slots 0 .. CMAX/2)
constexpr int P_BETA = CMAX / 2;           // bf16[CMAX]
constexpr int P_AHI = CMAX;                // bf16[CMAX]
constexpr int P_ALO = CMAX + CMAX / 2;     // bf16[CMAX]
constexpr int P_B = 2 * CMAX;              // float[CMAX]
constexpr int P_GS = 3 * CMAX;             // float[2 * GROUPS]
constexpr int P_ACC = 3 * CMAX + 2 * GROUPS; // float[2 * CMAX]: channel sums, sums of squares
constexpr int P_LEN = 5 * CMAX + 2 * GROUPS;

using VB = aie::vector<bfloat16, 32>;
using AC = aie::accum<accfloat, 32>;
using VF = aie::vector<float, 32>;

static inline AC acc_of(VB v) {
  AC a;
  a.from_vector(v);
  return a;
}
static inline AC acc_of(VF v) {
  AC a;
  a.from_vector(v);
  return a;
}

// silu(x) = h (1 + tanh h), h = x / 2 (exact in bf16), in fp32, rounded once (dit_ew's).
static inline VB silu32(VB x) {
  VB h = aie::mul(x, (bfloat16)0.5f).to_vector<bfloat16>();
  AC ha = acc_of(h);
  VF hf = ha.to_vector<float>();
  VB th;
  th.insert(0, aie::tanh<bfloat16>(hf.extract<16>(0)));
  th.insert(1, aie::tanh<bfloat16>(hf.extract<16>(1)));
  return aie::mac(ha, h, th).to_vector<bfloat16>();
}

// acc[c] += sum over the element's pixels of x[., c]; acc[CMAX + c] += x^2.
static inline void stats_acc(const bfloat16 *__restrict x, float *__restrict acc, int C) {
  const int P = EL / C;
  for (int cv = 0; cv < C; cv += 32) {
    AC s = acc_of(aie::load_v<32>(acc + cv));
    AC q = acc_of(aie::load_v<32>(acc + CMAX + cv));
    for (int p = 0; p < P; p++) {
      VB v = aie::load_v<32>(x + p * C + cv);
      s = aie::add(s, v);
      q = aie::mac(q, v, v);
    }
    aie::store_v(acc + cv, s.to_vector<float>());
    aie::store_v(acc + CMAX + cv, q.to_vector<float>());
  }
}

extern "C" {

// Start of every dispatch: round to nearest even from here on (the core resets to
// floor, and vew_param converts to bf16 before any data kernel runs -- the first
// dispatch after a context load came out 2x less accurate), and zero the channel sums
// the stats ops accumulate into.
void vew_begin(float *__restrict par, int32_t *__restrict rtp) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < 2 * CMAX; i++)
    par[P_ACC + i] = 0.0f;
}

void vew_stats(bfloat16 *__restrict x, float *__restrict par, int32_t *__restrict rtp) {
  stats_acc(x, par + P_ACC, rtp[R_C]);
}

void vew_stats_out(bfloat16 *__restrict y, float *__restrict par, int32_t *__restrict rtp) {
  const int C = rtp[R_C], per = C / GROUPS;
  float *__restrict acc = par + P_ACC;
  float *__restrict out = (float *)y;
  for (int g = 0; g < GROUPS; g++) {
    float s = 0.0f, q = 0.0f;
    for (int i = 0; i < per; i++) {
      s += acc[g * per + i];
      q += acc[CMAX + g * per + i];
    }
    out[2 * g] = s;
    out[2 * g + 1] = q;
  }
  for (int i = 2 * GROUPS; i < EL / 2; i++)
    out[i] = 0.0f;
  for (int i = 0; i < 2 * CMAX; i++)
    acc[i] = 0.0f;
}

// Parameter element j = i + shift (vae_ew.py: each core of a pair sees the block one
// element apart; j outside 0..16 is the padding and is dropped).
void vew_param(bfloat16 *__restrict e, float *__restrict par, int32_t i, int32_t shift,
               int32_t *__restrict rtp) {
  const int j = i + shift, C = rtp[R_C];
  bfloat16 *__restrict gamma = (bfloat16 *)(par + P_GAMMA);
  bfloat16 *__restrict beta = (bfloat16 *)(par + P_BETA);
  float *__restrict gs = par + P_GS;
  if (j == 0) {
    for (int c = 0; c < C; c++) {
      gamma[c] = e[c];
      beta[c] = e[CMAX + c];
    }
    for (int g = 0; g < 2 * GROUPS; g++)
      gs[g] = 0.0f;
    return;
  }
  if (j < 1 || j > 16)
    return;
  const float *__restrict part = (const float *)e;
  for (int g = 0; g < 2 * GROUPS; g++)
    gs[g] += part[g];
  if (j < 16)
    return;
  const int per = C / GROUPS;
  const float inv_n = 1.0f / (float(rtp[R_NPIX]) * float(per));
  bfloat16 *__restrict ahi = (bfloat16 *)(par + P_AHI);
  bfloat16 *__restrict alo = (bfloat16 *)(par + P_ALO);
  float *__restrict bb = par + P_B;
  for (int g = 0; g < GROUPS; g++) {
    const float mean = gs[2 * g] * inv_n;
    float var = gs[2 * g + 1] * inv_n - mean * mean;
    const float rstd = aie::invsqrt((var > 0.0f ? var : 0.0f) + EPS);
    for (int i2 = 0; i2 < per; i2++) {
      const int c = g * per + i2;
      const float a = float(gamma[c]) * rstd;
      const bfloat16 hi = (bfloat16)a;
      ahi[c] = hi;
      alo[c] = (bfloat16)(a - float(hi));
      bb[c] = float(beta[c]) - mean * a;
    }
  }
}

void vew_apply(bfloat16 *__restrict x, bfloat16 *__restrict y, float *__restrict par,
               int32_t *__restrict rtp) {
  const int C = rtp[R_C], P = EL / C;
  const bool silu = rtp[R_SILU] != 0;
  const bfloat16 *__restrict ahi = (const bfloat16 *)(par + P_AHI);
  const bfloat16 *__restrict alo = (const bfloat16 *)(par + P_ALO);
  const float *__restrict bb = par + P_B;
  for (int cv = 0; cv < C; cv += 32) {
    const VB h = aie::load_v<32>(ahi + cv), l = aie::load_v<32>(alo + cv);
    const AC b0 = acc_of(aie::load_v<32>(bb + cv));
    for (int p = 0; p < P; p++) {
      VB v = aie::load_v<32>(x + p * C + cv);
      VB t = aie::mac(aie::mac(b0, v, h), v, l).to_vector<bfloat16>();
      aie::store_v(y + p * C + cv, silu ? silu32(t) : t);
    }
  }
}

void vew_add(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict y,
             float *__restrict par, int32_t *__restrict rtp) {
  for (int i = 0; i < EL; i += 32)
    aie::store_v(y + i, aie::add(acc_of(aie::load_v<32>(a + i)), aie::load_v<32>(b + i))
                            .to_vector<bfloat16>());
  if (rtp[R_ADD_STATS])
    stats_acc(y, par + P_ACC, rtp[R_C]);
}

void vew_rgba(bfloat16 *__restrict x, bfloat16 *__restrict y, int32_t *__restrict rtp) {
  uint8_t *__restrict o = (uint8_t *)y;
  for (int p = 0; p < EL / 4; p++) {
    for (int c = 0; c < 3; c++) {
      float f = float(x[4 * p + c]) * 127.5f + 127.5f;
      int v = int(f + 0.5f);
      o[4 * p + c] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    o[4 * p + 3] = 255;
  }
}

} // extern "C"
