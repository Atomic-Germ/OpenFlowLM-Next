#pragma once
#include "attn.h"
// Experimental RB=1 Qwen attention only. The online softmax phases below are
// derived from production attn_row_impl; K and V add their BF16 low planes.
// Keep the HD loop rolled to leave program space for the diagnostic captures.
static_assert(ATTN_VEXP == 1 && ATTN_RB == 1 && ATTN_NH == 24 &&
              ATTN_KVH == 4 && ATTN_HD == 256 && ATTN_NHL == 4);
#if ATTN_KV_PAIR
__attribute__((noinline)) inline void attn_pair_row(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const bfloat16 *__restrict Kl, const bfloat16 *__restrict Vl,
                                  const ATTN_QT *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml ATTN_H0_PARM) {
  ATTN_H0_DECL
  // No set_rounding here: it is core-wide state, attn_init_impl sets it once per
  // token, and nothing between that and the last row touches it. It was being
  // written once per position per layer -- 82k times a token, for one value.
  alignas(128) float sv[kMLS];
  // The rescale factors leave phase 2 already split into the bf16 (hi, lo) pairs the
  // macs want, and with an INTEGER flag saying which of the two is one. Doing that
  // per head in phase 3 meant a scalar float subtract and a scalar float compare per
  // head per position, and scalar float on this core is a software call.
  alignas(128) bfloat16 ah[kMLS], al[kMLS], bh[kMLS], bl[kMLS];
  alignas(128) int32_t grew_i[kMLS];

  // Unrolled only when this is the hot path. With ATTN_RB > 1 it runs for the
  // handful of rows that do not fill a block, and the 8x of code it costs is
  // 8x of the core's 16 KB of program memory that the block kernel needs.
#if ATTN_RB == 1 && ATTN_HD <= 128
  AIE_LOOP_UNROLL_FULL      // at HD 256 four heads unrolled over eight head-dim steps overflow
#endif                      // program memory (the 35B's ax, 2026-09-08); the loop stays a loop there
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned h = kSplit ? ((unsigned)h0 + hl) : hl;   // folds away when this core owns them all                 // global head: q and the kv mapping
    const unsigned kvh = h / (kNH / kKVH);
    const bfloat16 *q = qs + h * kHD;
    const bfloat16 *k = Kt + kvh * kHD;
    // Two accumulators, not one: the hi and lo terms are independent, so this is
    // twice the distance between dependent macs for the scheduler to fill.
    // (aie::reduce_add_v reduces four of these in one call, which is strictly less
    // arithmetic and measured 116.6 -> 141.2 ms: holding four heads' vectors live
    // to feed it spills, and the spill costs more than the reduction saves.)
    accf32 d0 = aie::zeros<accfloat, kV>(), d1 = aie::zeros<accfloat, kV>();
    #pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kHD; j += kV) {
      const vbN<kV> kj = aie::load_v<kV>(k + j);
      const vbN<kV> klj = aie::load_v<kV>(Kl + kvh*kHD + j);
      d0 = aie::mac(d0, aie::load_v<kV>(q + j), kj);          // hi
      d1 = aie::mac(d1, aie::load_v<kV>(q + kQW + j), kj);    // lo
      d0 = aie::mac(d0, aie::load_v<kV>(q + j), klj);
      d1 = aie::mac(d1, aie::load_v<kV>(q + kQW + j), klj);
    }
    sv[hl] = aie::reduce_add(aie::add(d0, d1).template to_vector<float>());   // 1/sqrt(HD) is in q
  }
  for (unsigned h = kNHL; h < kMLU; ++h) sv[h] = -1e30f;
#if ATTN_ABL < 2
  for (unsigned h = 0; h < kNHL; ++h) ml[h] = sv[h];   // sink, so the scores are not dead code
  return;
#endif

  for (unsigned v = 0; v < kMLU; v += kVE) {
#if ATTN_SCALE_IN_Q
    const vfN<kVE> s = aie::load_v<kVE>(sv + v);
#else
    // q carries no 1/sqrt(HD) at this head dim (ATTN_SCALE_IN_Q): the scores take it here
    const vfN<kVE> s = fmulN<kVE>(aie::load_v<kVE>(sv + v), aie::broadcast<float, kVE>(kScale));
#endif
    const vfN<kVE> m = aie::load_v<kVE>(ml + v);
    const vfN<kVE> mn = aie::max(m, s);
    // mn is m or s, so exactly one of (m - mn), (s - mn) is zero and its exp is
    // exactly 1. One exponential of their SUM therefore carries both: the sum is
    // m - s when s wins and s - m when m does, and the loser's 1.0 is a select
    // rather than a second polynomial.
    const vfN<kVE> e = vexpN<kVE>(faddN<kVE>(fsubN<kVE>(m, mn), fsubN<kVE>(s, mn)));
    const vfN<kVE> one = aie::broadcast<float, kVE>(1.0f);
    const auto grew = aie::gt(s, m);
    const vfN<kVE> a = aie::select(one, e, grew);
    const vfN<kVE> b = aie::select(e, one, grew);
    const vfN<kVE> l = aie::load_v<kVE>(ml + kMLS + v);
    aie::store_v(ml + v, mn);
    aie::store_v(ml + kMLS + v, faddN<kVE>(fmulN<kVE>(l, a), b));
    vbN<kVE> t0, t1;
    splitN<kVE>(a, t0, t1);
    aie::store_v(ah + v, t0);
    aie::store_v(al + v, t1);
    splitN<kVE>(b, t0, t1);
    aie::store_v(bh + v, t0);
    aie::store_v(bl + v, t1);
    aie::store_v(grew_i + v, aie::select(aie::zeros<int32_t, kVE>(),
                                         aie::broadcast<int32_t, kVE>(1), grew));
  }

#if ATTN_ABL < 3
  return;
#endif
  // Unrolled only when this is the hot path. With ATTN_RB > 1 it runs for the
  // handful of rows that do not fill a block, and the 8x of code it costs is
  // 8x of the core's 16 KB of program memory that the block kernel needs.
#if ATTN_RB == 1 && ATTN_HD <= 128
  AIE_LOOP_UNROLL_FULL      // at HD 256 four heads unrolled over eight head-dim steps overflow
#endif                      // program memory (the 35B's ax, 2026-09-08); the loop stays a loop there
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned kvh = (kSplit ? ((unsigned)h0 + hl) : hl) / (kNH / kKVH);
    const bfloat16 *v = Vt + kvh * kHD;
    const bfloat16 bhh = bh[hl], bll = bl[hl];
    float *o = oacc + hl * kHD;
    // a is exp(m_old - m_new), so it is exactly 1 unless THIS position raised the
    // head's running max -- which over a long context happens O(log P) times, not
    // P. The rescale is not merely a multiply by one either: fscaleN splits o into
    // a bf16 pair and remultiplies, so skipping it is both faster and closer to the
    // fp32 accumulator it is meant to leave alone.
    if (grew_i[hl] == 0) {
      // a is exactly 1: this position did not raise the head's running max, which
      // over a long context is the overwhelming majority. Skipping the rescale is
      // not just a multiply by one saved -- it also leaves the fp32 accumulator
      // alone instead of round-tripping it through a bf16 pair.
      #pragma clang loop unroll(disable)
      for (unsigned j = 0; j < kHD; j += kV) {
        accf32 acc;
        acc.from_vector(aie::load_v<kV>(o + j));
        const vbN<kV> vj = aie::load_v<kV>(v + j);
        acc = aie::mac(acc, vj, bhh);
        acc = aie::mac(acc, vj, bll);
        const vbN<kV> vlj = aie::load_v<kV>(Vl + kvh*kHD + j);
        acc = aie::mac(acc, vlj, bhh);
        acc = aie::mac(acc, vlj, bll);
        aie::store_v(o + j, acc.template to_vector<float>());
      }
    } else {
      const bfloat16 ahh = ah[hl], all = al[hl];
      #pragma clang loop unroll(disable)
      for (unsigned j = 0; j < kHD; j += kV) {
        accf32 acc = aie::zeros<accfloat, kV>();
        acc = mac_vs<kV>(acc, aie::load_v<kV>(o + j), ahh, all);
        const vbN<kV> vj = aie::load_v<kV>(v + j);
        acc = aie::mac(acc, vj, bhh);
        acc = aie::mac(acc, vj, bll);
        const vbN<kV> vlj = aie::load_v<kV>(Vl + kvh*kHD + j);
        acc = aie::mac(acc, vlj, bhh);
        acc = aie::mac(acc, vlj, bll);
        aie::store_v(o + j, acc.template to_vector<float>());
      }
    }
  }
}
#else
inline void attn_pair_row(const bfloat16 *K, const bfloat16 *V,
                   const bfloat16 *Kl, const bfloat16 *Vl,
                   const ATTN_QT *q, float *o, float *ml ATTN_H0_PARM) {
  attn_row_impl(K, V, q, o, ml ATTN_H0_ARG);
}
#endif
