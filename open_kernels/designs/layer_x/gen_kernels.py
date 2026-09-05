"""Generate the small kernel TUs of the whole-layer designs (one extern "C" entry per file).

Scratch layouts (floats; xcommon.MS_FLOATS / DS_FLOATS):
  ms (MoE, 928):  rw[32] @0 | xr[256] @32 | acc[256] @288 | u[64] @544 | g[64] @608 | yd[256] @672
  ds (DeltaNet, 1280): vec[512] @0 | t[128] @512 | o[128] @640 | k_hl bf16[320] @768 | q_hl @928
                       | delta_hl bf16[256] @1088 | dd bf16[16] @1216
"""
from pathlib import Path

HERE = Path(__file__).parent

GEMV_HDR = '''#define GEMV_PER_CALL 2
#include "gemv_q4.h"
'''

FILES = {
    # ---- q4 GEMV entry points: runtime group / band law, 2 chunks per element (10240 B)
    "gemv_q4_gy.cc": GEMV_HDR + '''// A projection band into its y element: (per_band, rs) = (16, 2) K=2048, (32, 2) K=4096.
extern "C" {
void gemv_q4_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
    "gemv_q4_gup.cc": GEMV_HDR + '''// MoE up (band 0) / gate (band 1): 64-row K=2048 bands (the routed stripe halves through the
// strided tap, the shared expert's band as it lies) into ms[544 + 64 band ..].
extern "C" {
void gemv_q4_gup(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                 int32_t group, int32_t band) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + 544 + 64 * band, 16, 2);
}
}
''',
    "gemv_q4_gdown.cc": GEMV_HDR + '''// MoE down, element j of 8, into ms[672 ..] (the core's 256 rows) against h's table at tab + 4608:
// routed slots: two 128-row RS=4 bands of 4 elements; the shared expert: four 64-row RS=2 bands of 2.
extern "C" {
void gemv_q4_gdown(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                   int32_t j, int32_t slot) {
  if (slot < 8)
    gemv_q4_pool_group_rt(t, tab + 4608, (unsigned)(j % 4), ms + 672 + 128 * (j / 4), 8, 4);
  else
    gemv_q4_pool_group_rt(t, tab + 4608, (unsigned)(j % 2), ms + 672 + 64 * (j / 2), 4, 2);
}
}
''',
    "gemv_q4_prep_k4096_b0n64.cc": '''// og (bf16[4096]) arrives as two 4 KB act elements: blocks 0..63 from the first.
#include "gemv_q4.h"

extern "C" {
GEMV_Q4_PREP_BLOCKS_ENTRY(4096, 0, 64)
}
''',
    "gemv_q4_prep_k4096_b64n64.cc": '''// og (bf16[4096]) arrives as two 4 KB act elements: blocks 64..127 from the second.
#include "gemv_q4.h"

extern "C" {
GEMV_Q4_PREP_BLOCKS_ENTRY(4096, 64, 64)
}
''',
    "gemv_q4_prep_h.cc": '''// the expert hidden h (f32[512], assembled in DDR from the cores' parts) -> bf16 -> its table
// at tab + 4608 (past xm's K=2048 table).
#include "gemv_q4.h"

extern "C" {
void gemv_q4_prep_h(const float *__restrict hf, uint8_t *__restrict tab) {
  gemv_q4_prep_f32(hf, tab + 4608, 512);
}
}
''',
    # ---- MoE
    "moe_hdr2.cc": '''// The MoE header, three 10 KB w-stream elements per core (mode 0, 1, 2):
//   0: [router output f32[1024] | junk]  -> rw = floats 256..287 (w[e] at 8 + e)
//   1: [sgw bf16[2048] | junk]           -> rw[0] = sigmoid(xm . sgw), xm = the act element
//   2: [xres slice f32[256] | junk]      -> xr (this core's 256 residual rows)
#include "vecmath.h"

extern "C" {
void moe_hdr2(const uint8_t *__restrict e, const bfloat16 *__restrict xm, float *__restrict ms, int32_t mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *__restrict rw = ms;
  float *__restrict xr = ms + 32;
  if (mode == 0) {
    aie::store_v(rw, aie::load_v<32>((const float *)(e + 1024)));
  } else if (mode == 1) {
    const bfloat16 *__restrict sgw = (const bfloat16 *)e;
    accf32 d = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 2048; j += 32)
      d = aie::mac(d, aie::load_v<32>(xm + j), aie::load_v<32>(sgw + j));
    // sigmoid on a vector lane: no scalar float ops (they pull in the soft-float library)
    const v32f u = aie::broadcast<float, 32>(aie::reduce_add(d.template to_vector<float>()));
    rw[0] = vsigmoidN<32>(u)[0];
  } else {
    const float *__restrict xrs = (const float *)e;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 256; j += 32)
      aie::store_v(xr + j, aie::load_v<32>(xrs + j));
  }
}
}
''',
    "moe_silu32.cc": '''// h part = silu(g) * u for this core's 64 rows (ms: u @544, g @608), emitted as f32 (one 256 B y
// element); the bf16 rounding happens in the consumer's table prep.
#include "vecmath.h"

extern "C" {
void moe_silu32(const float *__restrict ms, float *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 544;
  const float *__restrict g = ms + 608;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32)
    aie::store_v(h + j, fmul32(vsiluN<32>(aie::load_v<32>(g + j)), aie::load_v<32>(u + j)));
}
}
''',
    "moe_accfin.cc": '''// slot < 8:  acc = (slot == 0 ? 0 : acc) + w[slot] * yd     (the routed weight, rw[8 + slot])
// slot == 8: acc = xres + acc + sigmoid(xm . sgw) * yd        (the shared expert, rw[0]) = the block output
// ms: rw @0, xr @32, acc @288, yd @672. No scalar float ops (soft-float library).
#include "vecmath.h"

extern "C" {
void moe_accfin(float *__restrict ms, int32_t slot) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict rw = ms;
  const float *__restrict xr = ms + 32;
  float *__restrict acc = ms + 288;
  const float *__restrict y = ms + 672;
  const bool shared = slot >= 8;
  v32b wh, wl;
  split32(aie::broadcast<float, 32>(shared ? rw[0] : rw[8 + slot]), wh, wl);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 256; j += 32) {
    accf32 a;
    if (shared)
      a.from_vector(fadd32(aie::load_v<32>(xr + j), aie::load_v<32>(acc + j)));
    else if (slot == 0)
      a = aie::zeros<accfloat, 32>();
    else
      a.from_vector(aie::load_v<32>(acc + j));
    v32b yh, yl;
    split32(aie::load_v<32>(y + j), yh, yl);
    a = aie::mac(a, yh, wh);
    a = aie::mac(a, yh, wl);
    a = aie::mac(a, yl, wh);
    aie::store_v(acc + j, a.template to_vector<float>());
  }
}
}
''',
    "moe_out.cc": '''// y element j (64 floats) = rows 64j..64j+63 of this core's 256-row block output (ms: acc @288).
#include "vecmath.h"

extern "C" {
void moe_out(const float *__restrict ms, float *__restrict y, int32_t j) {
  const float *__restrict acc = ms + 288;
  aie::store_v(y, aie::load_v<32>(acc + 64 * j));
  aie::store_v(y + 32, aie::load_v<32>(acc + 64 * j + 32));
}
}
''',
    # ---- DeltaNet (dnx.h; ds layout in its header)
    "dnx_vcopy.cc": '''#include "dnx.h"
extern "C" {
void dnx_vcopy(const float *__restrict e, float *__restrict ds) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 512; j += kV)
    aie::store_v(ds + DS_VEC + j, aie::load_v<kV>(e + j));
}
}
''',
    "dnx_pass1.cc": '''#include "dnx.h"
extern "C" {
void dnx_pass1(const float *__restrict S, float *__restrict ds, int32_t blk) {
  dnx_pass1_slice(S, ds, (unsigned)blk);
}
}
''',
    "dnx_delta.cc": '''#include "dnx.h"
extern "C" {
void dnx_delta(float *__restrict ds) {
  dnx_delta_head(ds);
}
}
''',
    "dnx_row.cc": '''#include "dnx.h"
extern "C" {
void dnx_row(const float *__restrict S, float *__restrict ds, float *__restrict ye, int32_t blk, int32_t j) {
  dnx_row_half(S, ds, ye, (unsigned)blk, (unsigned)(j >> 1), (unsigned)(j & 1));
}
}
''',
    "dnx_ofin.cc": '''#include "dnx.h"
extern "C" {
void dnx_ofin(const float *__restrict ds, float *__restrict ye, int32_t hf) {
  dnx_ofin_half(ds, ye, (unsigned)hf);
}
}
''',
}

STALE = ["gemv_q4_p2b16r2_g.cc", "gemv_q4_p2b16r2_gu.cc", "gemv_q4_p2b32r2_g.cc", "gemv_q4_p2b8r4_g.cc",
         "gemv_q4_r2h2.cc", "gemv_q4_prep_f32_k512.cc", "moe_acc2.cc", "moe_fin2.cc"]

for name, src in FILES.items():
    p = HERE / name
    if not p.is_file() or p.read_text(encoding="utf-8") != src:
        p.write_text(src, encoding="utf-8", newline="\n")
for name in STALE:
    p = HERE / name
    if p.is_file():
        p.unlink()
print(f"{len(FILES)} kernel files")
