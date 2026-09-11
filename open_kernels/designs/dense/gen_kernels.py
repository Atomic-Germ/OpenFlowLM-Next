"""Generate the small kernel TUs of the dense layer design (one extern "C" entry per file).

    python gen_kernels.py            # for OPEN_KERNELS_SPEC (a qwen3 spec)

The GEMV band entry (gemv_q4_gy) is generated here with the recipe's chunks-per-element. Also: the up /
gate band into the silu scratch, silu(gate) * up for one 64-row band, and the
activation-table preps that take the ELEMENT index (the core loops over 4 KB
elements; the kernel derives the block range, so no arithmetic on loop indices
in the IRON body).

Scratch `ms` (floats): u[64] @MS_U | g[64] @MS_G.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # open_kernels/
from recipes.load import current_spec  # noqa: E402
from recipes import dense as QR  # noqa: E402



def q8_gy(pc: int) -> str:
    """The q8 projection entry: the q4 `gemv_q4_gy` with the half-tile band law
    (OPEN-QUANT-Q8). One symbol, one TU, generated only when a role is q8."""
    return f'''#define GEMV_PER_CALL {pc}
#include "gemv_q8.h"
// A q8 projection band into its y element: runtime band law (per_band half-tiles, rs = 4).
extern "C" {{
void gemv_q8_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {{
  gemv_q8_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}}
}}
'''


def q8_gms(pc: int, ms_u: int, ms_g: int) -> str:
    """The q8 twin of `gemv_q4_gms`: a 64-row band into the act scratch at ms + dst."""
    return f'''#define GEMV_PER_CALL {pc}
#include "gemv_q8.h"
// A q8 64-row band into the act scratch at ms + dst (the up band at {ms_u}, the gate band at {ms_g}).
extern "C" {{
void gemv_q8_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                 int32_t group, int32_t per_band, int32_t dst) {{
  gemv_q8_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 4);
}}
}}
'''


def q4_gyms(pc: int, ms_u: int, ms_g: int) -> str:
    """`gemv_q4_gy` and `gemv_q4_gms` folded into ONE entry point, for a main core that
    also carries the q8 GEMV (OPEN-QUANT-Q8). The destination is a runtime argument:
    dst < 0 writes the band into its y element, dst >= 0 into the silu scratch at ms + dst.
    The row split is the literal 2 `gemv_q4_gms` already hard-codes: every q4_1 band in a
    dense tail is a 64-row std_perm band, so the band walk's index arithmetic folds away.

    Why fold: `gemv_q4_pool_group_rt` is `static inline`, so each entry point carries its
    own copy of the band walk -- two entries are two bodies, and a container that MIXES
    formats needs the q8 body on the same 16 KB core (the Qwen3.5 4B's `lx` overflowed
    program memory, .claude/plans/q8-hw-results.md section 2). Generated ONLY for such a
    spec: an all-q4_1 or an all-q8 spec keeps today's entries and does not move."""
    return f'''#define GEMV_PER_CALL {pc}
#include "gemv_q4.h"
// The folded q4_1 band entry (mixed-format cores only): dst < 0 -> the band's y element,
// dst >= 0 -> the silu scratch at ms + dst (the up band at {ms_u}, the gate band at {ms_g}).
extern "C" {{
void gemv_q4_gyms(const uint8_t *__restrict t, const uint8_t *__restrict tab,
                  float *__restrict y, float *__restrict ms,
                  int32_t group, int32_t per_band, int32_t dst) {{
  float *__restrict d = (dst < 0) ? y : ms + dst;
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, d, (unsigned)per_band, 2);
}}
}}
'''


# The two projection roles a dense layer has, and the fold condition dx.py reads off them.
PROJ_ROLES = ("attn", "ffn")


def mixed(R) -> bool:
    """The spec's roles mix formats on one main core: something is q8, something is still
    q4_1, and the q4_1 side is the `gy` + `gms` pair the fold replaces."""
    q8 = R.spec.q8_roles
    return bool(q8) and "ffn" not in q8 and any(r not in q8 for r in PROJ_ROLES)


# Generated only for a spec that needs them; removed again when it does not, so a family's
# translation-unit set (and its build key) never gains a file it does not compile.
Q8_FILES = ("gemv_q8_gy.cc", "gemv_q8_gms.cc")
FOLD_FILES = ("gemv_q4_gyms.cc",)


def files(R) -> dict[str, str]:
    G = R.geo
    q8 = R.spec.q8_roles
    hdr = f'''#define GEMV_PER_CALL {G.PER_CALL}
#include "gemv_q4.h"
'''
    out = {
        "gemv_q4_gy.cc": hdr + '''// A band into its y element: runtime band law (per_band chunks, row split rs).
// The entry name carries GEMV_Q4_PREFIX: the f32-scale build (GEMV_SCALES_F32=1,
// -DGEMV_Q4_PREFIX=gemv_q4s32) references the suffixed symbol from its MLIR.
#define GEMV_Q4_WRAP__(PFX, NAME) PFX##_g##NAME
#define GEMV_Q4_WRAP(PFX, NAME) GEMV_Q4_WRAP__(PFX, NAME)
extern "C" {
void GEMV_Q4_WRAP(GEMV_Q4_PREFIX, y)(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                                     int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
        "gemv_q4_gms.cc": hdr + f'''// A 64-row band into the silu scratch at ms + dst (the up band at {G.MS_U}, the gate band at {G.MS_G}).
#define GEMV_Q4_WRAP__(PFX, NAME) PFX##_g##NAME
#define GEMV_Q4_WRAP(PFX, NAME) GEMV_Q4_WRAP__(PFX, NAME)
extern "C" {{
void GEMV_Q4_WRAP(GEMV_Q4_PREFIX, ms)(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                                      int32_t group, int32_t per_band, int32_t dst) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}}
}}
''',
        "dense_act.cc": f'''// h band = act(g) * u for one 64-row band (ms: u @{G.MS_U}, g @{G.MS_G}) -> one f32 y element.
// act = {G.ACT}: silu(x) = x sigmoid(x); gelu_tanh(x) = 0.5 x (1 + tanh(z)) = x sigmoid(2z),
// z = sqrt(2/pi) (x + 0.044715 x^3). Vector ops only (no scalar float on this core).
#include "vecmath.h"

extern "C" {{
void dense_act(const float *__restrict ms, float *__restrict h) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + {G.MS_U};
  const float *__restrict g = ms + {G.MS_G};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {{
    const v32f x = aie::load_v<32>(g + j);
#if {1 if G.ACT == "gelu_tanh" else 0}
    const v32f x3 = fmul32(fmul32(x, x), x);
    const v32f z2 = fscaleN<32>(fadd32(x, fscaleN<32>(x3, 0.044715f)), 2.0f * 0.7978845608028654f);
    const v32f a = fmul32(x, vsigmoidN<32>(z2));
#else
    const v32f a = vsiluN<32>(x);
#endif
    aie::store_v(h + j, fmul32(a, aie::load_v<32>(u + j)));
  }}
}}
}}
''',
        "dense_prep.cc": '''// Element i of a bf16 activation of K values (2048 per 4 KB element) into the table: blocks
// [64 i, min(64 i + 64, K/32)).
#include "gemv_q4.h"

extern "C" {
void dense_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
''',
        "dense_prep_f32.cc": '''// Element i of an fp32 activation of K values (1024 per 4 KB element; the fifo types it as bf16)
// into the table: blocks [32 i, min(32 i + 32, K/32)).
#include "gemv_q4.h"

extern "C" {
void dense_prep_f32(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 32u * (unsigned)i;
  const unsigned nb = (b0 + 32u <= total) ? 32u : total - b0;
  gemv_q4_prep_f32_blocks((const float *)e, tab, (unsigned)K, b0, nb);
}
}
''',
    }
    if mixed(R):
        # The mixed-format core cannot hold both q4_1 entries beside the q8 body, so the
        # pair becomes one folded entry with a runtime destination. Nothing else moves.
        out = {"gemv_q4_gyms.cc": q4_gyms(G.PER_CALL, G.MS_U, G.MS_G),
               **{k: v for k, v in out.items() if k not in ("gemv_q4_gy.cc", "gemv_q4_gms.cc")}}
    if q8:
        out["gemv_q8_gy.cc"] = q8_gy(G.PER_CALL)
    if "ffn" in q8:
        out["gemv_q8_gms.cc"] = q8_gms(G.PER_CALL, G.MS_U, G.MS_G)
    return out


STALE = ["dense_silu.cc"]


def generate(R, out: Path = HERE) -> int:
    fs = files(R)
    gone = list(STALE) + [n for n in Q8_FILES + FOLD_FILES if n not in fs]
    if mixed(R):                      # the folded entry replaces the pair on disk too
        gone += [n for n in ("gemv_q4_gy.cc", "gemv_q4_gms.cc") if n not in fs]
    for name in gone:
        if (out / name).is_file():
            (out / name).unlink()
    for name, src in fs.items():
        p = out / name
        if not p.is_file() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8", newline="\n")
    return len(fs)


if __name__ == "__main__":
    n = generate(QR.recipe(current_spec()))
    print(f"{n} kernel files")
