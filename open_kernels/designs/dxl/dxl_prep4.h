#pragma once
// gemv_q4_prep_block's arithmetic, bit-identical, on PB blocks at once so their serial chains overlap.
// Table layout of 4 tokens (a group, 4 * gemv_q4_tab_bytes(K) bytes), which dxl_tile4 reads:
//   int16 xi[K/32 blocks][4 oc][4 tokens][8]  (A for an mmul is one 64 B load) | int32 s[4][K/32] | bf16 xs_hi[4][K/32], xs_lo[4][K/32]
#include "../gemv_q4/gemv_tab.h"

#ifndef DXL_PB
#define DXL_PB 4
#endif

template <unsigned PB>
static inline void dxl_prep_group(const aie::vector<bfloat16, 32> (&xv)[PB], uint8_t *__restrict grp, unsigned K,
                                  unsigned t, unsigned kb, const aie::vector<uint8_t, 64> &absmask) {
  const unsigned NB = K / 32;
  int16_t *__restrict xg = (int16_t *)grp;
  int32_t *__restrict sh = (int32_t *)(grp + 8 * K + t * 4 * NB);
  bfloat16 *__restrict xsh = (bfloat16 *)(grp + 8 * K + 16 * NB + t * 4 * NB);
  bfloat16 *__restrict xsl = xsh + NB;
  int s[PB];
#pragma clang loop unroll(full)
  for (unsigned u = 0; u < PB; ++u) {
    const aie::vector<int16_t, 32> ab =
        aie::bit_and(xv[u].template cast_to<uint8_t>(), absmask).template cast_to<int16_t>();
    const int mx = aie::reduce_max(ab);
    s[u] = 141 - (mx >> 7);
    if (s[u] > 126) s[u] = 126;
    sh[kb + u] = s[u];
  }
#pragma clang loop unroll(full)
  for (unsigned u = 0; u < PB; ++u) {
    const aie::vector<bfloat16, 32> sc =
        aie::broadcast<int16_t, 32>((int16_t)((127 + s[u]) << 7)).template cast_to<bfloat16>();
    const aie::accum<accfloat, 32> a = aie::mul(xv[u], sc);
    const aie::vector<int16_t, 32> r = aie::to_fixed<int16_t>(a, 0);
#pragma clang loop unroll(full)
    for (unsigned oc = 0; oc < 4; ++oc)
      aie::store_v(xg + ((kb + u) * 4 + oc) * 32 + t * 8, r.template extract<8>(oc));
  }
  aie::vector<float, 16> tr[PB];
#pragma clang loop unroll(full)
  for (unsigned u = 0; u < PB; ++u) {
    aie::accum<accfloat, 32> xa;
    xa.from_vector(xv[u]);
    const aie::vector<float, 32> v32 = xa.template to_vector<float>();
    aie::accum<accfloat, 16> a16, b16;
    a16.from_vector(v32.template extract<16>(0));
    b16.from_vector(v32.template extract<16>(1));
    const aie::vector<float, 16> v16 = aie::add(a16, b16).template to_vector<float>();
    aie::accum<accfloat, 8> a8, b8;
    a8.from_vector(v16.template extract<8>(0));
    b8.from_vector(v16.template extract<8>(1));
    tr[u] = aie::concat(aie::add(a8, b8).template to_vector<float>(), aie::zeros<float, 8>());
  }
#pragma clang loop unroll(full)
  for (unsigned st = 4; st >= 1; st >>= 1) {
#pragma clang loop unroll(full)
    for (unsigned u = 0; u < PB; ++u) {
      aie::accum<accfloat, 16> p, q;
      p.from_vector(tr[u]);
      q.from_vector(aie::shuffle_down_rotate(tr[u], st));
      tr[u] = aie::add(p, q).template to_vector<float>();
    }
  }
#pragma clang loop unroll(full)
  for (unsigned u = 0; u < PB; ++u) {
    aie::accum<accfloat, 32> bc;
    bc.from_vector(aie::broadcast<float, 32>(tr[u][0]));
    const aie::vector<bfloat16, 32> hi = bc.template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> lo = aie::sub(bc, hi).template to_vector<bfloat16>();
    xsh[kb + u] = hi[0];
    xsl[kb + u] = lo[0];
  }
}

// Blocks b0 .. b0+nb-1 of token `tok`'s K-wide activation (x points at block b0) into its group's table.
// F32: the activation is fp32, rounded to bf16 first (conv_even), as gemv_q4_prep_f32_blocks does.
template <bool F32>
__attribute__((noinline)) inline void dxl_prep_tok(const void *__restrict xp, uint8_t *__restrict tab, unsigned K,
                                                   unsigned tok, unsigned b0, unsigned nb) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned tb = gemv_q4_tab_bytes(K);
  uint8_t *grp = tab + (tok & ~3u) * tb;
  const unsigned t = tok & 3u;
  const aie::vector<uint8_t, 64> absmask = gemv_q4_absmask();
  auto load = [&](unsigned j) {
    if constexpr (F32) {
      aie::accum<accfloat, 32> a;
      a.from_vector(aie::load_v<32>((const float *)xp + j * 32));
      return a.template to_vector<bfloat16>();
    } else {
      return aie::load_v<32>((const bfloat16 *)xp + j * 32);
    }
  };
  unsigned j = 0;
#pragma clang loop unroll(disable)
  for (; j + DXL_PB <= nb; j += DXL_PB) {
    aie::vector<bfloat16, 32> xv[DXL_PB];
#pragma clang loop unroll(full)
    for (unsigned u = 0; u < DXL_PB; ++u) xv[u] = load(j + u);
    dxl_prep_group<DXL_PB>(xv, grp, K, t, b0 + j, absmask);
  }
#pragma clang loop unroll(disable)
  for (; j < nb; ++j) {
    aie::vector<bfloat16, 32> xv[1] = {load(j)};
    dxl_prep_group<1>(xv, grp, K, t, b0 + j, absmask);
  }
}
