#pragma once
// moe_batch's core ops (moe_batch.py). The product is taken transposed, Y^T [tokens, rows] =
// X^T [tokens, K] @ W^T [K, rows], so the activation is the mmul's A operand and the weight its
// B: an aie::mmul<8, 8, 8> B block is 8 k x 8 rows row-major, and a q4_1 chunk's nibbles are
// stored k-major with 16 rows per k (nibble k*16 + r16, gemv_q4.h's law), so one 64-byte load
// is 8 k x 16 rows = two B blocks (the even rows in the low nibbles, the odd rows in the high),
// needing only a mask and a uint8 -> bf16 convert. The row scales d, m are contiguous per
// (32-k block, 16 rows) and ride along as vectors. Nothing is gathered or re-laid.
//
// MB_NT is the token slot width, a multiple of the mmul's 8 A rows; MB_NG = MB_NT / 8 is how
// many of those A blocks a slot holds. Widening it halves the DDR traffic per token
// (.claude/plans/prefill-parity.md, B(1)) and doubles the core work per slot, so it only pays
// when the core is comfortably under the stream. It is not, even now: see mb_step_parity.
//
// Layouts, all per 64-row band and MB_NT tokens:
//   A (x or h) tile for one 64-k tile: [8 k-blocks][MB_NG][8 tokens][8 k] bf16
//   C tile: [4 row groups of 16][2 (even rows, odd rows)][MB_NG][8 tokens][8 rows] f32;
//     row 16 g + 2 j + p of token 8 s + t sits at block (g, p, s) lane t * 8 + j. The odd
//     rows' nibbles arrive x16 (unshifted), folded into their d as d / 16.
#include "vecmath.h"
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <stdint.h>

#ifndef MB_NT
#define MB_NT 8
#endif

static constexpr unsigned MB_CHUNK = 5120;
static constexpr unsigned MB_NIB = 1024;      // nibbles start; d bf16[256] at 0, m at 512
static constexpr unsigned MB_NG = MB_NT / 8;  // mmul A blocks (of 8 tokens) per slot
static_assert(MB_NT % 8 == 0 && MB_NT >= 8, "MB_NT is a whole number of 8-token mmul tiles");

// One parity's row scales from the 16 of a (32-k block, 16-row) group, laid out as a B
// operand: lane k * 8 + r is row 2r's scale (2r + 1 for ODD). Built by halving a 512-bit
// register and doubling back up - concatenating 8-lane pieces instead lowers to one scalar
// extract and push per lane, which used to be most of the inner loop. D16 folds the odd
// rows' x16 (their nibbles arrive unshifted) into the scale.
template <bool ODD, bool D16>
static inline aie::vector<bfloat16, 64> mb_scale(const bfloat16 *__restrict p, unsigned kb_abs) {
  const aie::vector<bfloat16, 16> v16 = aie::load_v<16>(p + kb_abs * 32);
  const aie::vector<bfloat16, 32> v32 = aie::concat(v16, v16);
  aie::vector<bfloat16, 16> h = ODD ? aie::filter_odd(v32, 1) : aie::filter_even(v32, 1);
  if constexpr (ODD && D16) h = aie::mul(h, (bfloat16)0.0625f).to_vector<bfloat16>();
  const aie::vector<bfloat16, 32> h32 = aie::concat(h, h);
  return aie::concat(h32, h32);
}

// One PARITY of a band's 64 rows x MB_NT tokens: C += the k-tile `ky` of W's product with the
// A tile `xa`. The even rows live in the chunk's low nibbles and the odd in the high, and the
// two are taken as separate passes over the same nibble bytes rather than interleaved in one.
//
// That is the whole shape of this kernel, and it is a register-file decision. A 64-lane f32
// mmul accumulator is 256 B; the accumulator file holds about two of them. Interleaving the
// parities keeps two output accumulators live, and the dequantisation of each k-block --
// `se.from_vector(m); mac(se, n, d)` -- needs one more, so the register allocator spilled a
// whole accumulator per k-block. Measured in the emitted code (`kernel_remarks.py`, and the
// spill/reload count in the disassembly): the interleaved inner loop was 88 bundles of which
// 40 carried a spill or a reload, MII 54. One parity at a time leaves one output accumulator
// live, the dequant temporary fits beside it, and the same loop is 41 bundles with ZERO
// spills, MII 39. The second pass re-reads the nibbles and the activations out of L1, which
// is much cheaper than the spill traffic it replaces.
//
// The k-block loop is then fully unrolled, which lets the scheduler pack across what were
// iteration boundaries. Unrolling the k-tile loop around it as well is faster again and does
// not build -- see the note on that loop.
//
// Measured end to end, 16 slots (31.5 MB of expert weights), minima of three runs:
//
//   interleaved parities, no hints (as shipped before)   0.979 ms
//   one parity per pass                                  0.903
//   + the k-block loop unrolled                          0.895   <- this
//   (+ the k-tile loop unrolled too: 0.868, but see below)
//   the weight stream alone (MB_NULL_MM)                 0.779
//
// So the exposed core time fell 0.190 -> 0.116 ms, 39 %, and the dispatch is within 15 % of
// what its DMA costs. What is left of the core is the mmul itself: aie2p's native bf16
// mac_dims is (4, 8, 8), so `mmul<8, 8, 8>` is emulated and each mac carries eight
// `vextbcst` operand builds. Going to the native shape is the next real lever, and it only
// pays alongside a wider MB_NT -- which in turn only pays once the core is cheap. Reading the
// pool contiguously instead of by band stride (MB_CONTIG) is NOT a lever: it moves the stream
// floor 0.779 -> 0.768, 1.4 %.
// The number of 8-token sub-tiles one pass carries, and so how many mmul accumulators are
// live at once. 1 is the shape B(3) settled on and the only one that fits beside the dequant
// temporary; MB_SHARE_NG builds the alternative, where a pass carries the whole slot and one
// dequantised weight feeds every sub-tile, for the A-B that says which way the register file
// falls. Sharing saves the dequant and costs a spill; at MB_NT 16 the spill is dearer.
#ifdef MB_SHARE_NG
#define MB_PASS_NG MB_NG
#else
#define MB_PASS_NG 1
#endif

// One PARITY of a band's 64 rows x MB_PASS_NG sub-tiles of tokens, starting at sub-tile `s0`:
// C += the k-tile `ky` of W's product with the A tile `xa`. The even rows live in the chunk's
// low nibbles and the odd in the high, and the two are taken as separate passes over the same
// nibble bytes rather than interleaved in one.
template <bool ODD>
static inline void mb_step_parity(const uint8_t *__restrict nibp, const bfloat16 *__restrict dp,
                                  const bfloat16 *__restrict mp, unsigned ky, const bfloat16 *__restrict xa,
                                  float *__restrict cp, unsigned s0) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
  // At MB_PASS_NG = 1 this is the single live accumulator the shape is built around: the
  // dequantisation of each k-block needs one more, and the file holds about two.
  MMUL acc[MB_PASS_NG];
  for (unsigned s = 0; s < MB_PASS_NG; ++s) acc[s] = MMUL(aie::load_v<64>(cp + s * 64));
  // NOT unrolled: fully unrolling this as well was 3 % faster at 16 slots (0.868 vs 0.895)
  // and then crashed Peano at the real slot count -- "Register not in mBMs", the aie2p code
  // emitter refusing a register the allocator had picked for the bigger body. A compiler
  // crash that only appears at one MB_SLOTS is not worth 3 %.
  AIE_LOOP_RANGE(2, 2)
  for (unsigned kb = 0; kb < 2; ++kb) {
    const unsigned kb_abs = ky * 2 + kb;
#ifndef MB_NULL_DQ
    const aie::vector<bfloat16, 64> d = mb_scale<ODD, true>(dp, kb_abs);
    const aie::vector<bfloat16, 64> m = mb_scale<ODD, false>(mp, kb_abs);
#endif
    AIE_LOOP_UNROLL_FULL
    for (unsigned il = 0; il < 4; ++il) {
      const unsigned i = kb * 4 + il;
      const aie::vector<uint8_t, 64> q = aie::load_v<64>(nibp + i * 64);
      const aie::vector<bfloat16, 64> n =
          aie::to_float<bfloat16>(aie::bit_and((uint8_t)(ODD ? 0xF0 : 0x0F), q), 0);
#ifdef MB_NULL_DQ
      const aie::vector<bfloat16, 64> w = n;
#else
      accN<64> se;
      se.from_vector(m);
      se = aie::mac(se, n, d);
      const aie::vector<bfloat16, 64> w = se.template to_vector<bfloat16>();
#endif
      for (unsigned s = 0; s < MB_PASS_NG; ++s)
        acc[s].mac(aie::load_v<64>(xa + (i * MB_NG + s0 + s) * 64), w);
    }
  }
  for (unsigned s = 0; s < MB_PASS_NG; ++s) aie::store_v(cp + s * 64, acc[s].template to_vector<float>());
}

// C[band's 64 rows x MB_NT tokens] += k-tile ky of W[band] times the A tile `xa`: the band's
// four 16-row groups, each a pair of parity passes over the same 64 bytes of nibbles, and
// -- above MB_NT 8 -- one such pair per 8-token sub-tile of the slot.
//
// Repeating the pair per sub-tile repeats the dequantisation with it, which is the trade a
// wider slot makes: the weight STREAM is what halves per token, and the dequant work per
// token is unchanged. Sharing one dequantised weight across the sub-tiles instead (the
// arithmetically tidier answer, MB_SHARE_NG) needs MB_NG accumulators live and spills; see
// the note above mb_step_parity.
static inline void mb_step_tile(const uint8_t *__restrict band, unsigned ky, const bfloat16 *__restrict xa,
                                float *__restrict c) {
#ifdef MB_NULL_MM
  return;   // timing ablation: the streams without any core work
#endif
  // The sub-tile pass is the OUTER loop, and is not unrolled: unrolling it, or nesting it
  // inside the row-group loop, emits MB_NG copies of the pair and a 16-row group's pair is
  // most of the core's 16 KB of program memory already ("Overflow of program memory" at
  // MB_NT 16, from the loader, not the compiler). Outside and rolled, the code is the same
  // size at every width and runs the same body MB_NG times.
  AIE_LOOP_RANGE(1, MB_NG / MB_PASS_NG)
  for (unsigned s0 = 0; s0 < MB_NG; s0 += MB_PASS_NG) {
    AIE_LOOP_RANGE(4, 4)
    for (unsigned g = 0; g < 4; ++g) {
      const uint8_t *__restrict chunk = band + (g >> 1) * MB_CHUNK;
      const unsigned half = g & 1;
      const uint8_t *__restrict nibp = chunk + MB_NIB + half * 2048 + ky * 512;
      const bfloat16 *__restrict dp = reinterpret_cast<const bfloat16 *>(chunk) + half * 16;
      const bfloat16 *__restrict mp = reinterpret_cast<const bfloat16 *>(chunk + 512) + half * 16;
      float *__restrict cb = c + (g * 2 * MB_NG) * 64;
      mb_step_parity<false>(nibp, dp, mp, ky, xa, cb + s0 * 64, s0);
      mb_step_parity<true>(nibp, dp, mp, ky, xa, cb + (MB_NG + s0) * 64, s0);
    }
  }
}

static inline void mb_zero_n(float *__restrict c, unsigned n) {
  const aie::vector<float, 32> z = aie::zeros<float, 32>();
  for (unsigned i = 0; i < n; i += 32) aie::store_v(c + i, z);
}

// h = silu(g) * u over the core's two C tiles (128 rows), written as the down projection's A
// tiles: rows back in natural order, [16 k-blocks][MB_NG][8 tokens][8 k] bf16
static inline void mb_silu_tile(const float *__restrict u, const float *__restrict g, bfloat16 *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned blk = 0; blk < 16; blk += 2)          // one 16-row group: its even and odd C blocks
    for (unsigned s = 0; s < MB_NG; ++s) {            // that group's eight tokens at a time
      const float *__restrict ue = u + (blk * MB_NG + s) * 64;
      const float *__restrict ge = g + (blk * MB_NG + s) * 64;
      aie::vector<bfloat16, 32> he[2], ho[2];
      for (unsigned hh = 0; hh < 2; ++hh) {           // tokens 0-3 of the sub-tile, then 4-7
        accf32 a;
        a.from_vector(fmulN<32>(vsiluN<32>(aie::load_v<32>(ge + hh * 32)), aie::load_v<32>(ue + hh * 32)));
        he[hh] = a.template to_vector<bfloat16>();
        a.from_vector(fmulN<32>(vsiluN<32>(aie::load_v<32>(ge + MB_NG * 64 + hh * 32)),
                                aie::load_v<32>(ue + MB_NG * 64 + hh * 32)));
        ho[hh] = a.template to_vector<bfloat16>();
      }
      // [t][even j], [t][odd j] -> [t][r0..r15] -> the two k-blocks' [t][8]
      const auto z = aie::interleave_zip(aie::concat(he[0], he[1]), aie::concat(ho[0], ho[1]), 1);
      const auto kbs = aie::interleave_unzip(z.first, z.second, 8);
      aie::store_v(h + (blk * MB_NG + s) * 64, kbs.first);
      aie::store_v(h + ((blk + 1) * MB_NG + s) * 64, kbs.second);
    }
}
