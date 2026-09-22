// PROBE ONLY -- not wired into any design, not built by any recipe. It exists to price
// the code size of the attention score phase written as an `aie::mmul` block product
// against the per-(row, head) `reduce_add` dot products attn_rowb_impl uses today,
// because program memory on the attention core is what decides whether that rewrite is
// possible at all (16,384 bytes, ~1 KB free at RB 2). Compile it with Peano and read
// the symbol size:
//
//   clang++ -c --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG \
//     -D__AIE_API_AIE_ADF_HPP__ <the attn flags> probe_score_mmul.cc
//   python utilities/core_sizes.py --symbols <the .o>    (it reads plain ELF objects too)
//
// The shape: AIE2P's native bf16 mac_dims is (4, 8, 8), so the tile is
// mmul<4, 8, 8, bfloat16, bfloat16, accfloat> -- A 4x8, B 8x8, C 4x8. Put the cached
// ROWS on M and the HEADS on N:
//
//   A = K rows   [RB=4 rows x 8 dims ]  four 8-element loads, one per fifo row, concatenated
//   B = q^T      [8 dims    x 8 heads]  one contiguous 64-element load, q PRE-PACKED per token
//   C = scores   [4 rows    x 8 heads]  which is exactly sv's [row][kNL] layout at kNL 8
//
// so C is sv, M is RB 4, and K_t is read once per block instead of once per (head, row).
// q's hi/lo bf16 split makes it two macs per k-step, the same arithmetic as today.
#include "attn.h"

#ifndef PROBE_RB
#define PROBE_RB 4
#endif

extern "C" {
// qp: the pre-packed q for this core's heads, [j / 8][8 dims][8 heads] bf16, hi half then
// lo half -- what attn_q.cc would write instead of the [head][dim] pair it writes now.
void probe_score_mmul(const bfloat16 *const *__restrict Kb, const bfloat16 *__restrict qp,
                      float *__restrict sv, int kvh) {
  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16, accfloat>;
  MMUL C = MMUL(aie::zeros<float, 32>());
  const unsigned qhalf = (kHD / 8) * 64;          // the hi block's element count
  ATTN_UNROLL_HD      // the same hint the head-dim loops carry today (unroll 4 at HD 256)
  for (unsigned j = 0; j < kHD; j += 8) {
    const auto a = aie::concat(aie::load_v<8>(Kb[0] + (unsigned)kvh * kHD + j),
                               aie::load_v<8>(Kb[1] + (unsigned)kvh * kHD + j),
                               aie::load_v<8>(Kb[2] + (unsigned)kvh * kHD + j),
                               aie::load_v<8>(Kb[3] + (unsigned)kvh * kHD + j));
    const unsigned b = (j / 8) * 64;
    C.mac(a, aie::load_v<64>(qp + b));
    C.mac(a, aie::load_v<64>(qp + qhalf + b));
  }
  aie::store_v(sv, C.template to_vector<float>());
}
}

// The SAME phase as attn_rowb_impl writes it today, lifted out so the two can be sized
// against each other at identical flags: per (row, head) two accumulators over the hi and
// lo halves of q, eight 32-lane macs each, then one reduce_add.
extern "C" {
void probe_score_now(const bfloat16 *const *__restrict Kb, const bfloat16 *__restrict qs,
                     float *__restrict sv, int kvh, int h0) {
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned h = (unsigned)h0 + hl;
    const bfloat16 *q = qs + h * kHD;
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < PROBE_RB; ++r) {
      const bfloat16 *k = Kb[r] + (unsigned)kvh * kHD;
      accf32 d0 = aie::zeros<accfloat, kV>(), d1 = aie::zeros<accfloat, kV>();
      ATTN_UNROLL_HD
      for (unsigned j = 0; j < kHD; j += kV) {
        const vbN<kV> kj = aie::load_v<kV>(k + j);
        d0 = aie::mac(d0, aie::load_v<kV>(q + j), kj);
        d1 = aie::mac(d1, aie::load_v<kV>(q + kQW + j), kj);
      }
      sv[r * kNL + hl] = aie::reduce_add(aie::add(d0, d1).template to_vector<float>());
    }
  }
}
}
