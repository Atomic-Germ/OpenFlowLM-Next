"""Generate lg.py's kernel TUs (one extern "C" entry per file) for OPEN_KERNELS_SPEC (lfm3).

    python gen_kernels.py            # into this directory

Every GEMV is gemv_q4.h's q4_1 band walk (std_perm's 64-row bands, row split 2), the LFM2-MoE / Flint TUs
under this family's names: lg_gy (a band into its y element: the mixers' projections), lg_gms (a slot's gate /
up band into the scratch at ms + dst), lg_gacc (a down band into the scratch at ms + 128, then onto the core's
slot accumulator, fresh for slot 0), lg_act1 (silu(g) * u, unweighted: the MLP's slots), lg_copy (an
accumulated band to the y fifo), lg_prep / lg_prepf (the activation table from bf16 / f32 elements by element
index).

lg_cp / lg_scmac0 / lg_scmac / lg_scfin are the conv core: LFM2's short depthwise causal conv
(designs/short_conv/sc.h) at LFM3's 7 taps, for one SCW-channel element, a row at a time (the 7 tap rows, 6
state rows, B, C, u and the outputs do not fit one fifo's 16 buffer descriptors at once). The taps arrive as
f32 tap-major rows (the GGUF's F32 weight, transposed by the `transpose` pack op, which moves bytes and does not
convert) and are bf16 values, so the in-core bf16 conversion is exact and the MACs are sc.h's mac_vv form:
    bx = B * u;  conv = sum_{k<6} s_k * w_k + bx * w_6   (window [s_0 .. s_5, bx], newest LAST);  y = bf16(C * conv)
    n_k = s_{k+1} (k < 5);  n_5 = bx                      (the window slides; the state stays f32)

Scratch `ms` (floats): u[64] @MS_U | g[64] @MS_G | a down band [64] @128.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # open_kernels/
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402


def files(R) -> dict[str, str]:
    G = R.geo
    from recipes import lfm3 as FR
    q4 = f'''#define GEMV_PER_CALL {G.PER_CALL}
''' + ('''#define GEMV_Q4_0_NATIVE 1      // LFM3_Q40: 4608-B chunks, m = -8 d
''' if FR.q40_on() else '') + '''#include "gemv_q4.h"
'''
    return {
        "lg_gy.cc": q4 + '''// A q4_1 band into its y element (runtime band law: per_band chunks, row split rs).
extern "C" {
void lg_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
           int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
        "lg_gms.cc": q4 + f'''// A q4_1 64-row band of a slot's gate or up into the scratch at ms + dst (up at {G.MS_U}, gate at {G.MS_G}).
extern "C" {{
void lg_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
            int32_t group, int32_t per_band, int32_t dst) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}}
}}
''',
        "lg_gacc.cc": q4 + '''#include "vecmath.h"
// A q4_1 down band of slot e into the scratch at ms + 128 and, after the band's last group, onto
// band `band` of the core's slot accumulator (fresh for e == 0).
extern "C" {
void lg_gacc(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
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
        "lg_act1.cc": f'''// h band = silu(g) * u for one 64-row band (ms: u @{G.MS_U}, g @{G.MS_G}) -> one f32 y element.
// silu(x) = x sigmoid(x) as the dense dx computes it (vsiluN).
#include "vecmath.h"

extern "C" {{
void lg_act1(const float *__restrict ms, float *__restrict h) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + {G.MS_U};
  const float *__restrict g = ms + {G.MS_G};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {{
    const v32f a = vsiluN<32>(aie::load_v<32>(g + j));
    aie::store_v(h + j, fmul32(a, aie::load_v<32>(u + j)));
  }}
}}
}}
''',
        "lg_copy.cc": '''// One accumulated down band (64 floats of acc, band `band`) into its y element.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void lg_copy(const float *__restrict acc, float *__restrict y, int32_t band) {
  const float *__restrict a = acc + band * 64;
#pragma clang loop unroll(full)
  for (unsigned j = 0; j < 64; j += 16) aie::store_v(y + j, aie::load_v<16>(a + j));
}
}
''',
        "lg_prep.cc": '''// Element i of a bf16 activation of K values (2048 per 4 KB element) into the table: blocks
// [64 i, min(64 i + 64, K/32)).
#include "gemv_q4.h"

extern "C" {
void lg_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
''',
        "lg_prepf.cc": '''// Element i of an fp32 activation of K values (1024 per 4 KB element; the fifo types it as bf16)
// into the table: blocks [32 i, min(32 i + 32, K/32)).
#include "gemv_q4.h"

extern "C" {
void lg_prepf(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 32u * (unsigned)i;
  const unsigned nb = (b0 + 32u <= total) ? 32u : total - b0;
  gemv_q4_prep_f32_blocks((const float *)e, tab, (unsigned)K, b0, nb);
}
}
''',
        "lg_cp.cc": f'''// Row k of a core-local [rows][{G.SCW}] f32 buffer from a fifo element (the conv taps: they are held while the
// state rows and B / C / u stream, so the fifo never has to hold a whole element's inputs at once).
#include "vecmath.h"

extern "C" {{
void lg_cp(const float *__restrict src, float *__restrict dst, int32_t k) {{
  float *__restrict d = dst + k * {G.SCW};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < {G.SCW}; j += 32) aie::store_v(d + j, aie::load_v<32>(src + j));
}}
}}
''',
        "lg_scmac0.cc": f'''// The conv core, state row 0: acc = s_0 * w_0 (w the core-local [{G.TAPS}][{G.SCW}] taps; bf16 values).
#include "vecmath.h"

extern "C" {{
void lg_scmac0(const float *__restrict s, const float *__restrict w, float *__restrict acc) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < {G.SCW}; j += V) {{
    accN<V> t;
    t.from_vector(aie::load_v<V>(w + j));
    accN<V> a = aie::zeros<accfloat, V>();
    a = mac_vv(a, aie::load_v<V>(s + j), t.template to_vector<bfloat16>());
    aie::store_v(acc + j, a.template to_vector<float>());
  }}
}}
}}
''',
        "lg_scmac.cc": f'''// The conv core, state row k >= 1: acc += s_k * w_k, and the new state row k - 1 is this one (n = s_k).
#include "vecmath.h"

extern "C" {{
void lg_scmac(const float *__restrict s, const float *__restrict w, float *__restrict acc,
              float *__restrict n, int32_t k) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
  const float *__restrict wk = w + k * {G.SCW};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < {G.SCW}; j += V) {{
    accN<V> t;
    t.from_vector(aie::load_v<V>(wk + j));
    accN<V> a;
    a.from_vector(aie::load_v<V>(acc + j));
    a = mac_vv(a, aie::load_v<V>(s + j), t.template to_vector<bfloat16>());
    aie::store_v(acc + j, a.template to_vector<float>());
    aie::store_v(n + j, aie::load_v<V>(s + j));
  }}
}}
}}
''',
        "lg_scfin.cc": f'''// The conv core, the new token: bx = B * u; conv = acc + bx * w_{G.TAPS - 1}; y = bf16(C * conv); the newest state row is bx.
// (designs/short_conv/sc.h's arithmetic, the window [s_0 .. s_{G.TAPS - 2}, bx] with the LAST tap on this token.)
#include "vecmath.h"

extern "C" {{
void lg_scfin(const float *__restrict Bv, const float *__restrict Cv, const float *__restrict uv,
              const float *__restrict w, const float *__restrict acc, bfloat16 *__restrict y,
              float *__restrict n) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
  const float *__restrict wk = w + {G.TAPS - 1} * {G.SCW};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < {G.SCW}; j += V) {{
    const vfN<V> bx = fmulN<V>(aie::load_v<V>(Bv + j), aie::load_v<V>(uv + j));
    accN<V> t;
    t.from_vector(aie::load_v<V>(wk + j));
    accN<V> a;
    a.from_vector(aie::load_v<V>(acc + j));
    a = mac_vv(a, bx, t.template to_vector<bfloat16>());
    const vfN<V> conv = a.template to_vector<float>();
    accN<V> out;
    out.from_vector(fmulN<V>(aie::load_v<V>(Cv + j), conv));
    aie::store_v(y + j, out.template to_vector<bfloat16>());
    aie::store_v(n + j, bx);
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
