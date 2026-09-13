// LFM2's short depthwise causal convolution -- the stage that REPLACES attention in a
// layer, so this header sits where attn.h sits in designs/dense and runs on the same
// helper core.
//
// Per token, after the fused input projection has produced [B | C | u]:
//
//     Bx      = B * u
//     conv[c] = sum_k window[k][c] * w[c][k]      window = [state0, state1, Bx], newest LAST
//     out[c]  = C[c] * conv[c]
//     state  <- [state1, Bx]
//
// `model/replica_lfm2.py: short_conv_step` is the fp64 reference and these must agree. The
// orientation is the part worth stating twice: this token's value pairs with the LAST tap,
// and a window built the other way round still produces plausible numbers.
//
// The weight arrives TAP-MAJOR, [taps][width], not the [width][taps] the container stores.
// The pack plan transposes it (recipes/lfm2.py, a `transpose` op) because a core loading
// tap k for 32 consecutive channels wants those 32 values contiguous; at [width][taps]
// they are a stride-3 gather.
//
// There is no arithmetic here worth spreading over the array: about 10k f32 ops a token
// against the 12.6M MACs of the projection that feeds it. It is an elementwise stage on
// one core.
#pragma once
#include "vecmath.h"

#ifndef SC_TAPS
#define SC_TAPS 3
#endif
#ifndef SC_W                    // channels in one fifo element
#define SC_W 1024
#endif

static_assert(SC_TAPS == 3, "sc.h implements a 3-tap window; another tap count is another "
                            "design (the core holds taps - 1 rows of state)");
static_assert(SC_W % 32 == 0, "SC_W must be a whole number of 32-lane vectors");

constexpr unsigned kSCW = SC_W;
constexpr unsigned kSCV = 32;

extern "C" {

/// One element's channels through the block's elementwise middle.
///
/// `s0` / `s1` are the two earlier tokens still inside the window, oldest first; `n0` / `n1`
/// are the same two slots after this token (n0 = s1, n1 = B * u). Writing both rather than
/// rotating a ring keeps the host out of the state's parity.
///
/// `w0` / `w1` / `w2` are this element's channels of taps 0, 1 and 2 -- contiguous, because
/// the weight is tap-major.
__attribute__((noinline)) inline void
short_conv_elem(const float *__restrict Bv, const float *__restrict Cv, const float *__restrict uv,
                const bfloat16 *__restrict w0, const bfloat16 *__restrict w1,
                const bfloat16 *__restrict w2, const float *__restrict s0,
                const float *__restrict s1, float *__restrict y, float *__restrict n0,
                float *__restrict n1) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kSCW; j += kSCV) {
    const vfN<kSCV> bx = fmulN<kSCV>(aie::load_v<kSCV>(Bv + j), aie::load_v<kSCV>(uv + j));
    // conv over the window, newest last: state0 * w0 + state1 * w1 + bx * w2
    accN<kSCV> acc = aie::zeros<accfloat, kSCV>();
    acc = mac_vv(acc, aie::load_v<kSCV>(s0 + j), aie::load_v<kSCV>(w0 + j));
    acc = mac_vv(acc, aie::load_v<kSCV>(s1 + j), aie::load_v<kSCV>(w1 + j));
    acc = mac_vv(acc, bx, aie::load_v<kSCV>(w2 + j));
    const vfN<kSCV> conv = acc.template to_vector<float>();
    aie::store_v(y + j, fmulN<kSCV>(aie::load_v<kSCV>(Cv + j), conv));
    // the window slides: what was state1 becomes state0, this token becomes state1
    aie::store_v(n0 + j, aie::load_v<kSCV>(s1 + j));
    aie::store_v(n1 + j, bx);
  }
}

}  // extern "C"
