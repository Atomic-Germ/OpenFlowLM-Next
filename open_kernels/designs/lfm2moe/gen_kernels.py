"""Generate lm.py's kernel TUs (one extern "C" entry per file) for OPEN_KERNELS_SPEC (lfm2moe).

    python gen_kernels.py            # into this directory

Every GEMV is gemv_q4.h's q4_1 band walk (the dense dx's, std_perm's 64-row bands, row split 2), exactly
Flint's TUs (designs/flint/gen_kernels.py) under this family's names: lm_gy (a band into its y element: the
mixers' projections), lm_gms (a slot's gate / up band into the scratch at ms + dst), lm_gacc (a down band
into the scratch at ms + 128, then onto the core's slot accumulator, fresh for slot 0), lm_act (w[e] *
silu(g) * u, the routing weight from the 16-float record hy_route writes), lm_act1 (the same unweighted: the
dense layers' MLP slots), lm_wgrab (the record off the x element that carries xm), lm_copy (an accumulated
band to the y fifo), lm_prep / lm_prepf (the activation table from bf16 / f32 elements by element index).

lm_sc is the conv core: sc.h's arithmetic (designs/short_conv, LFM2's) for one 512-channel element, with the
taps arriving as f32 (the GGUF's F32 conv weight, transposed tap-major by the `transpose` pack op, which moves
bytes and does not convert). They are bf16 values (a BF16 source), so the in-core bf16 conversion is exact and
the arithmetic is sc.h's mac_vv form bit for bit.

Scratch `ms` (floats): u[64] @MS_U | g[64] @MS_G | a down band [64] @128.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # open_kernels/
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402


def files(R) -> dict[str, str]:
    G, L = R.geo, R.layout
    from recipes import lfm2moe as FR
    q4 = f'''#define GEMV_PER_CALL {G.PER_CALL}
#include "gemv_q4.h"
'''
    w_off = L.AD_W - L.AD_XM - 4096 * (G.XM_ELEMS - 1)       # the weights' byte offset in xm's last element
    assert w_off == G.W_OFF and 0 <= w_off and w_off + 4 * FR.WV <= 4096
    act = lambda name, weighted: f'''// h band = {'w[e] * ' if weighted else ''}silu(g) * u for one 64-row band (ms: u @{G.MS_U}, g @{G.MS_G}) -> one f32 y
// element. silu(x) = x sigmoid(x) as the dense dx computes it (vsiluN){', then the slot' + "'" + 's routing weight' if weighted else ''}.
#include "vecmath.h"

extern "C" {{
void {name}(const float *__restrict ms, float *__restrict h, const float *__restrict wv, int32_t e) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + {G.MS_U};
  const float *__restrict g = ms + {G.MS_G};
{'  const v32f w = aie::broadcast<float, 32>(wv[e]);' if weighted else '  (void)wv;' + chr(10) + '  (void)e;'}
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {{
    const v32f a = vsiluN<32>(aie::load_v<32>(g + j));
    {'aie::store_v(h + j, fmul32(fmul32(a, aie::load_v<32>(u + j)), w));' if weighted else
     'aie::store_v(h + j, fmul32(a, aie::load_v<32>(u + j)));'}
  }}
}}
}}
'''
    return {
        "lm_gy.cc": q4 + '''// A q4_1 band into its y element (runtime band law: per_band chunks, row split rs).
extern "C" {
void lm_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
           int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
        "lm_gms.cc": q4 + f'''// A q4_1 64-row band of a slot's gate or up into the scratch at ms + dst (up at {G.MS_U}, gate at {G.MS_G}).
extern "C" {{
void lm_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
            int32_t group, int32_t per_band, int32_t dst) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}}
}}
''',
        "lm_gacc.cc": q4 + '''#include "vecmath.h"
// A q4_1 down band of slot e into the scratch at ms + 128 and, after the band's last group, onto
// band `band` of the core's slot accumulator (fresh for e == 0).
extern "C" {
void lm_gacc(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
             float *__restrict acc, int32_t group, int32_t per_band, int32_t band, int32_t e,
             int32_t ngroups) {
  float *__restrict y = ms + 128;
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, 2);
  if (group != ngroups - 1)
    return;
  float *__restrict a = acc + band * 64;
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 16) {
    const vfN<16> v = aie::load_v<16>(y + j);
    if (e == 0)
      aie::store_v(a + j, v);
    else
      aie::store_v(a + j, faddN<16>(aie::load_v<16>(a + j), v));
  }
}
}
''',
        "lm_act.cc": act("lm_act", True),
        "lm_act1.cc": act("lm_act1", False),
        "lm_wgrab.cc": f'''// The routing-weight record (f32[{FR.WV}], the host's hy_route) from xm's last x element, byte {w_off}.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {{
void lm_wgrab(const bfloat16 *__restrict xe, float *__restrict wv) {{
  aie::store_v(wv, aie::load_v<{FR.WV}>((const float *)((const uint8_t *)xe + {w_off})));
}}
}}
''',
        "lm_copy.cc": '''// One accumulated down band (64 floats of acc, band `band`) into its y element.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void lm_copy(const float *__restrict acc, float *__restrict y, int32_t band) {
  const float *__restrict a = acc + band * 64;
#pragma clang loop unroll(full)
  for (unsigned j = 0; j < 64; j += 16) aie::store_v(y + j, aie::load_v<16>(a + j));
}
}
''',
        "lm_prep.cc": '''// Element i of a bf16 activation of K values (2048 per 4 KB element) into the table: blocks
// [64 i, min(64 i + 64, K/32)).
#include "gemv_q4.h"

extern "C" {
void lm_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
''',
        "lm_prepf.cc": '''// Element i of an fp32 activation of K values (1024 per 4 KB element; the fifo types it as bf16)
// into the table: blocks [32 i, min(32 i + 32, K/32)).
#include "gemv_q4.h"

extern "C" {
void lm_prepf(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 32u * (unsigned)i;
  const unsigned nb = (b0 + 32u <= total) ? 32u : total - b0;
  gemv_q4_prep_f32_blocks((const float *)e, tab, (unsigned)K, b0, nb);
}
}
''',
        "lm_sc.cc": f'''// The conv core: LFM2's short depthwise causal conv for one {G.SCW}-channel element (designs/short_conv/sc.h's
// arithmetic), the taps as f32 tap-major rows (the GGUF's F32 weight; bf16 values, so the conversion below is
// exact and the MACs are sc.h's mac_vv bit for bit):
//     bx = B * u;  conv = s0 * w0 + s1 * w1 + bx * w2  (window [s0, s1, bx], newest LAST);  y = bf16(C * conv)
//     n0 = s1;  n1 = bx                                 (the window slides; the state stays f32)
#include "vecmath.h"

extern "C" {{
void lm_sc(const float *__restrict Bv, const float *__restrict Cv, const float *__restrict uv,
           const float *__restrict w0, const float *__restrict w1, const float *__restrict w2,
           const float *__restrict s0, const float *__restrict s1, bfloat16 *__restrict y,
           float *__restrict n0, float *__restrict n1) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < {G.SCW}; j += V) {{
    accN<V> t0, t1, t2;
    t0.from_vector(aie::load_v<V>(w0 + j));
    t1.from_vector(aie::load_v<V>(w1 + j));
    t2.from_vector(aie::load_v<V>(w2 + j));
    const vfN<V> bx = fmulN<V>(aie::load_v<V>(Bv + j), aie::load_v<V>(uv + j));
    accN<V> acc = aie::zeros<accfloat, V>();
    acc = mac_vv(acc, aie::load_v<V>(s0 + j), t0.template to_vector<bfloat16>());
    acc = mac_vv(acc, aie::load_v<V>(s1 + j), t1.template to_vector<bfloat16>());
    acc = mac_vv(acc, bx, t2.template to_vector<bfloat16>());
    const vfN<V> conv = acc.template to_vector<float>();
    accN<V> out;
    out.from_vector(fmulN<V>(aie::load_v<V>(Cv + j), conv));
    aie::store_v(y + j, out.template to_vector<bfloat16>());
    aie::store_v(n0 + j, aie::load_v<V>(s1 + j));
    aie::store_v(n1 + j, bx);
  }}
}}
}}
''',
    }


def generate(R, out: Path = HERE) -> int:
    fs = files(R)
    for name, text in fs.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return len(fs)


if __name__ == "__main__":
    spec = current_spec()
    n = generate(for_spec(spec).recipe(spec))
    print(f"wrote {n} TUs into {HERE}")
