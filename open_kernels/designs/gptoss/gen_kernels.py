"""Generate gx.py's kernel TUs (one extern "C" entry per file) for OPEN_KERNELS_SPEC (gptoss).

    python gen_kernels.py            # into this directory

Every GEMV keeps the bias in the k-tile-0 pad (GEMV_Q8_PAD_BIAS): gx_gy (a q8 band into its y
element: the attention), gx_gms (an expert band into the activation scratch at ms + dst) and
gx_gacc (a down band into the core's expert accumulator, fresh only for the first expert). The
two expert entries read mx4 half-tiles (gemv_mx4.h) when the recipe's experts are MXFP4, else
q8 half-tiles (gemv_q8.h).
gx_act is the expert's clamped SwiGLU scaled by its routing weight (gptoss_gate's arithmetic,
designs/gptoss_gate, NPU-checked, in f32); gx_wgrab lifts the weights off the x element that
carries xm; gx_copy hands an accumulated band to the y fifo; gx_prep / gx_prepf build the
activation table from bf16 / f32 elements by element index (dense_prep's arithmetic).

Scratch `ms` (floats): u[64] @MS_U | g[64] @MS_G.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # open_kernels/
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402


def files(R) -> dict[str, str]:
    G, L = R.geo, R.layout
    q8 = f'''#define GEMV_PER_CALL {G.PER_CALL}
#define GEMV_Q8_PAD_BIAS 1
#include "gemv_q8.h"
'''
    mx4 = G.EXPERT_CH != G.CALL_BYTES // G.PER_CALL
    ex = f'''#define GEMV_PER_CALL {G.EXPERT_PER_CALL}
#define GEMV_Q8_PAD_BIAS 1
#include "{'gemv_mx4.h' if mx4 else 'gemv_q8.h'}"
'''
    fmt = "mx4" if mx4 else "q8"
    walk = "gemv_mx4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band)" if mx4 else \
        "gemv_q8_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 4)"
    acc = "gemv_mx4_pool_group_acc" if mx4 else "gemv_q8_pool_group_acc"
    w_off = L.AD_W - L.AD_XM - 4096 * (G.XM_ELEMS - 1)       # the weights' byte offset in xm's last element
    assert 0 <= w_off and w_off + 4 * 8 <= 4096
    return {
        "gx_gy.cc": q8 + '''// A q8 band into its y element (runtime band law: per_band half-tiles, rs = 4); bias from the pad.
extern "C" {
void gx_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
           int32_t group, int32_t per_band, int32_t rs) {
  gemv_q8_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
        "gx_gms.cc": ex + f'''// An {fmt} 64-row band into the act scratch at ms + dst (up at {G.MS_U}, gate at {G.MS_G}); bias from the pad.
extern "C" {{
void gx_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
            int32_t group, int32_t per_band, int32_t dst) {{
  {walk};
}}
}}
''',
        "gx_gacc.cc": ex + f'''// An {fmt} down band into the core's expert accumulator: band `band` of acc, fresh for expert 0.
extern "C" {{
void gx_gacc(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict acc,
             int32_t group, int32_t per_band, int32_t band, int32_t e) {{
  {acc}(t, tab, (unsigned)group, acc + band * 64, (unsigned)per_band, e == 0);
}}
}}
''',
        "gx_act.cc": f'''// h band = w[e] * (clip(u, -7, 7) + 1) * g' * sigmoid(1.702 g'),  g' = min(g, 7),
// for one 64-row band (ms: u @{G.MS_U}, g @{G.MS_G}, biases already in) -> one f32 y element.
// gptoss_gate's arithmetic (designs/gptoss_gate), kept in f32 and scaled by the routing weight.
#include "vecmath.h"

extern "C" {{
void gx_act(const float *__restrict ms, float *__restrict h, const float *__restrict wv, int32_t e) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const v32f seven = aie::broadcast<float, 32>(7.0f);
  const v32f minus_seven = aie::broadcast<float, 32>(-7.0f);
  const v32f one = aie::broadcast<float, 32>(1.0f);
  const v32f w = aie::broadcast<float, 32>(wv[e]);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {{
    const v32f g = aie::min(aie::load_v<32>(ms + {G.MS_G} + j), seven);
    const v32f u = aie::max(aie::min(aie::load_v<32>(ms + {G.MS_U} + j), seven), minus_seven);
    const v32f sigmoid = vsigmoidN<32>(fscaleN<32>(g, 1.702f));
    const v32f y = fmulN<32>(faddN<32>(u, one), fmulN<32>(g, sigmoid));
    aie::store_v(h + j, fmulN<32>(y, w));
  }}
}}
}}
''',
        "gx_wgrab.cc": f'''// The routing weights (f32[8], the host's route step) from xm's last x element, byte {w_off}.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {{
void gx_wgrab(const bfloat16 *__restrict xe, float *__restrict wv) {{
  aie::store_v(wv, aie::load_v<8>((const float *)((const uint8_t *)xe + {w_off})));
}}
}}
''',
        "gx_copy.cc": '''// One accumulated down band (64 floats of acc, band `band`) into its y element.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void gx_copy(const float *__restrict acc, float *__restrict y, int32_t band) {
  const float *__restrict a = acc + band * 64;
#pragma clang loop unroll(full)
  for (unsigned j = 0; j < 64; j += 16) aie::store_v(y + j, aie::load_v<16>(a + j));
}
}
''',
        "gx_prep.cc": '''// Element i of a bf16 activation of K values (2048 per 4 KB element) into the table: blocks
// [64 i, min(64 i + 64, K/32)).
#include "gemv_q4.h"

extern "C" {
void gx_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
''',
        "gx_prepf.cc": '''// Element i of an fp32 activation of K values (1024 per 4 KB element; the fifo types it as bf16)
// into the table: blocks [32 i, min(32 i + 32, K/32)).
#include "gemv_q4.h"

extern "C" {
void gx_prepf(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 32u * (unsigned)i;
  const unsigned nb = (b0 + 32u <= total) ? 32u : total - b0;
  gemv_q4_prep_f32_blocks((const float *)e, tab, (unsigned)K, b0, nb);
}
}
''',
    }


def generate(R, out: Path = HERE) -> int:
    fs = files(R)
    for name, src in fs.items():
        p = out / name
        if not p.is_file() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8", newline="\n")
    return len(fs)


if __name__ == "__main__":
    spec = current_spec()
    print(f"{generate(for_spec(spec).recipe(spec))} kernel files")
