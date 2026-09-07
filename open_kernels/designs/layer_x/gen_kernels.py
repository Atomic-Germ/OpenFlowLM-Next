"""Generate the small kernel TUs of the whole-layer designs (one extern "C" entry per file),
with the recipe's geometry baked in (open_kernels/recipes/qwen36moe.py, `Common`): the ms
scratch offsets, the rows / hidden per core, the top-k, the widest K and the h table offset.

    python gen_kernels.py            # for OPEN_KERNELS_SPEC, else the checked-in 27B

Scratch layouts (floats; xcommon.MS_FLOATS / DS_FLOATS), for the 27B:
  ms (MoE, 928):  rw[32] @0 | xr[256] @32 | acc[256] @288 | u[64] @544 | g[64] @608 | yd[256] @672
  ds (DeltaNet, 1280): vec[512] @0 | t[128] @512 | o[128] @640 | k_hl bf16[320] @768 | q_hl @928
                       | delta_hl bf16[256] @1088 | dd bf16[16] @1216
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # open_kernels/
from recipes.load import current_recipe  # noqa: E402
from recipes import qwen36moe as Q  # noqa: E402



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
    dst < 0 writes the band into its y element, dst >= 0 into the act scratch at ms + dst.
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
// dst >= 0 -> the act scratch at ms + dst (the up band at {ms_u}, the gate band at {ms_g}).
extern "C" {{
void gemv_q4_gyms(const uint8_t *__restrict t, const uint8_t *__restrict tab,
                  float *__restrict y, float *__restrict ms,
                  int32_t group, int32_t per_band, int32_t dst) {{
  float *__restrict d = (dst < 0) ? y : ms + dst;
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, d, (unsigned)per_band, 2);
}}
}}
'''


# The projection roles the dense tail's main core runs a GEMV for, and the two conditions
# xcommon.py reads off them: a q4_1 entry is instantiated only while some role still needs
# it, and the fold applies only when BOTH formats are on the core.
PROJ_ROLES = ("attn", "linear", "linear_out", "ffn")


def mixed(R) -> bool:
    """The spec's roles mix formats on one main core: something is q8, something is still
    q4_1, and the q4_1 side is the `gy` + `gms` pair the fold replaces."""
    q8 = R.q8
    return bool(q8) and "ffn" not in q8 and any(r not in q8 for r in PROJ_ROLES)


# Generated only for a spec that needs them; removed again when it does not, so a family's
# translation-unit set (and its build key) never gains a file it does not compile.
Q8_FILES = ("gemv_q8_gy.cc", "gemv_q8_gms.cc")
FOLD_FILES = ("gemv_q4_gyms.cc",)


DNX = {
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


def q8_files(R) -> dict[str, str]:
    """The q8 GEMV TUs this recipe needs: `gy` wherever a projection is q8, plus `gms`
    when the dense FFN's up | gate bands are (they go into the act scratch, not a y
    element). Empty for every recipe whose roles are all q4_1."""
    C, q8 = R.common, R.q8
    out: dict[str, str] = {}
    if q8:
        out["gemv_q8_gy.cc"] = q8_gy(C.PER_CALL)
    if "ffn" in q8 and R.ffn is not None:
        out["gemv_q8_gms.cc"] = q8_gms(C.PER_CALL, R.ffn.MS_U, R.ffn.MS_G)
    return out


def dense_files(R) -> dict[str, str]:
    """The FFN tail's TUs for the layer_x fabric (ffn="dense", recipes/qwen35.py).

    They are designs/dense's kernels, generated HERE rather than included from there:
    the two designs both define `gemv_q4_gy`, so compiling one against the other's
    directory is a duplicate-symbol trap. One extern "C" entry per file, as everywhere.
    """
    C, F = R.common, R.ffn
    hdr = f'''#define GEMV_PER_CALL {C.PER_CALL}
#include "gemv_q4.h"
'''
    out = {
        "gemv_q4_gy.cc": hdr + '''// A band into its y element: runtime band law (per_band chunks, row split rs).
extern "C" {
void gemv_q4_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
        "gemv_q4_gms.cc": hdr + f'''// A 64-row band into the act scratch at ms + dst (the up band at {F.MS_U}, the gate band at {F.MS_G}).
extern "C" {{
void gemv_q4_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                 int32_t group, int32_t per_band, int32_t dst) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}}
}}
''',
        "dense_act.cc": f'''// h band = silu(g) * u for one 64-row band (ms: u @{F.MS_U}, g @{F.MS_G}) -> one f32 y element.
// silu(x) = x sigmoid(x). Vector ops only (no scalar float on this core).
#include "vecmath.h"

extern "C" {{
void dense_act(const float *__restrict ms, float *__restrict h) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + {F.MS_U};
  const float *__restrict g = ms + {F.MS_G};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32)
    aie::store_v(h + j, fmul32(vsiluN<32>(aie::load_v<32>(g + j)), aie::load_v<32>(u + j)));
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
        out = {"gemv_q4_gyms.cc": q4_gyms(C.PER_CALL, F.MS_U, F.MS_G),
               **{k: v for k, v in out.items() if k not in ("gemv_q4_gy.cc", "gemv_q4_gms.cc")}}
    out.update(DNX)
    out.update(q8_files(R))
    return out


def files(R) -> dict[str, str]:
    if getattr(R, "kind", "moe") == "dense":
        return dense_files(R)
    C, L = R.common, R.layout
    hid, ff, ne = C.HID, C.FF, C.NE
    kw = C.KWIDE
    nb = Q.ELEM // 2 // 32                        # bf16 blocks of 32 in one 4 KB element
    pb_hid = Q.band_bytes(hid) // C.TILE          # chunks per 64-row band, K = HID
    pb_wide = Q.band_bytes(kw) // C.TILE
    pb_down = Q.q4_bytes(128, ff) // C.TILE       # the routed down band: 128 rows x FF, RS=4
    pb_sdown = Q.band_bytes(ff) // C.TILE         # the shared down band: 64 rows x FF, RS=2
    g_down = C.DOWN_BAND // C.CALL_BYTES          # elements per routed down band
    g_sdown = Q.band_bytes(ff) // C.CALL_BYTES    # elements per shared down band
    gemv_hdr = f'''#define GEMV_PER_CALL {C.PER_CALL}
#include "gemv_q4.h"
'''
    out = {
        # ---- q4 GEMV entry points: runtime group / band law, PER_CALL chunks per element
        "gemv_q4_gy.cc": gemv_hdr + f'''// A projection band into its y element: (per_band, rs) = ({pb_hid}, 2) K={hid}, ({pb_wide}, 2) K={kw}.
extern "C" {{
void gemv_q4_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}}
}}
''',
        "gemv_q4_gup.cc": gemv_hdr + f'''// MoE up (band 0) / gate (band 1): 64-row K={hid} bands (the routed stripe halves through the
// strided tap, the shared expert's band as it lies) into ms[{C.MS_U} + {C.HID_PC} band ..].
extern "C" {{
void gemv_q4_gup(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                 int32_t group, int32_t band) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + {C.MS_U} + {C.HID_PC} * band, {pb_hid}, 2);
}}
}}
''',
        "gemv_q4_gdown.cc": gemv_hdr + f'''// MoE down, element j of {C.DOWN_ELEMS}, into ms[{C.MS_YD} ..] (the core's {C.ROWS_PC} rows) against h's table at tab + {C.H_TAB_OFF}:
// routed slots: {C.DOWN_PER_CORE} 128-row RS=4 bands of {g_down} elements; the shared expert: {C.ROWS_PC // 64} 64-row RS=2 bands of {g_sdown}.
extern "C" {{
void gemv_q4_gdown(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                   int32_t j, int32_t slot) {{
  if (slot < {ne})
    gemv_q4_pool_group_rt(t, tab + {C.H_TAB_OFF}, (unsigned)(j % {g_down}), ms + {C.MS_YD} + 128 * (j / {g_down}), {pb_down}, 4);
  else
    gemv_q4_pool_group_rt(t, tab + {C.H_TAB_OFF}, (unsigned)(j % {g_sdown}), ms + {C.MS_YD} + 64 * (j / {g_sdown}), {pb_sdown}, 2);
}}
}}
''',
        f"gemv_q4_prep_k{kw}_b0n{nb}.cc": f'''// og (bf16[{kw}]) arrives as two 4 KB act elements: blocks 0..{nb - 1} from the first.
#include "gemv_q4.h"

extern "C" {{
GEMV_Q4_PREP_BLOCKS_ENTRY({kw}, 0, {nb})
}}
''',
        f"gemv_q4_prep_k{kw}_b{nb}n{nb}.cc": f'''// og (bf16[{kw}]) arrives as two 4 KB act elements: blocks {nb}..{2 * nb - 1} from the second.
#include "gemv_q4.h"

extern "C" {{
GEMV_Q4_PREP_BLOCKS_ENTRY({kw}, {nb}, {nb})
}}
''',
        "gemv_q4_prep_h.cc": f'''// the expert hidden h (f32[{ff}], assembled in DDR from the cores' parts) -> bf16 -> its table
// at tab + {C.H_TAB_OFF} (past xm's K={hid} table).
#include "gemv_q4.h"

extern "C" {{
void gemv_q4_prep_h(const float *__restrict hf, uint8_t *__restrict tab) {{
  gemv_q4_prep_f32(hf, tab + {C.H_TAB_OFF}, {ff});
}}
}}
''',
        # ---- MoE
        "moe_hdr2.cc": f'''// The MoE header, three 10 KB w-stream elements per core (mode 0, 1, 2):
//   0: [router output f32[{Q.ELEM // 4}] | junk]  -> rw = floats {Q.ROUT_IDX_OFF // 4}..{Q.ROUT_IDX_OFF // 4 + 31} (w[e] at 8 + e)
//   1: [sgw bf16[{hid}] | junk]           -> rw[0] = sigmoid(xm . sgw), xm = the act element
//   2: [xres slice f32[{C.ROWS_PC}] | junk]      -> xr (this core's {C.ROWS_PC} residual rows)
#include "vecmath.h"

extern "C" {{
void moe_hdr2(const uint8_t *__restrict e, const bfloat16 *__restrict xm, float *__restrict ms, int32_t mode) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *__restrict rw = ms;
  float *__restrict xr = ms + {C.MS_XR};
  if (mode == 0) {{
    aie::store_v(rw, aie::load_v<32>((const float *)(e + {Q.ROUT_IDX_OFF})));
  }} else if (mode == 1) {{
    const bfloat16 *__restrict sgw = (const bfloat16 *)e;
    accf32 d = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < {hid}; j += 32)
      d = aie::mac(d, aie::load_v<32>(xm + j), aie::load_v<32>(sgw + j));
    // sigmoid on a vector lane: no scalar float ops (they pull in the soft-float library)
    const v32f u = aie::broadcast<float, 32>(aie::reduce_add(d.template to_vector<float>()));
    rw[0] = vsigmoidN<32>(u)[0];
  }} else {{
    const float *__restrict xrs = (const float *)e;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < {C.ROWS_PC}; j += 32)
      aie::store_v(xr + j, aie::load_v<32>(xrs + j));
  }}
}}
}}
''',
        "moe_silu32.cc": f'''// h part = silu(g) * u for this core's {C.HID_PC} rows (ms: u @{C.MS_U}, g @{C.MS_G}), emitted as f32 (one 256 B y
// element); the bf16 rounding happens in the consumer's table prep.
#include "vecmath.h"

extern "C" {{
void moe_silu32(const float *__restrict ms, float *__restrict h) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + {C.MS_U};
  const float *__restrict g = ms + {C.MS_G};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < {C.HID_PC}; j += 32)
    aie::store_v(h + j, fmul32(vsiluN<32>(aie::load_v<32>(g + j)), aie::load_v<32>(u + j)));
}}
}}
''',
        "moe_accfin.cc": f'''// slot < {ne}:  acc = (slot == 0 ? 0 : acc) + w[slot] * yd     (the routed weight, rw[8 + slot])
// slot == {ne}: acc = xres + acc + sigmoid(xm . sgw) * yd        (the shared expert, rw[0]) = the block output
// ms: rw @0, xr @{C.MS_XR}, acc @{C.MS_ACC}, yd @{C.MS_YD}. No scalar float ops (soft-float library).
#include "vecmath.h"

extern "C" {{
void moe_accfin(float *__restrict ms, int32_t slot) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict rw = ms;
  const float *__restrict xr = ms + {C.MS_XR};
  float *__restrict acc = ms + {C.MS_ACC};
  const float *__restrict y = ms + {C.MS_YD};
  const bool shared = slot >= {ne};
  v32b wh, wl;
  split32(aie::broadcast<float, 32>(shared ? rw[0] : rw[8 + slot]), wh, wl);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < {C.ROWS_PC}; j += 32) {{
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
  }}
}}
}}
''',
        "moe_out.cc": f'''// y element j (64 floats) = rows 64j..64j+63 of this core's {C.ROWS_PC}-row block output (ms: acc @{C.MS_ACC}).
#include "vecmath.h"

extern "C" {{
void moe_out(const float *__restrict ms, float *__restrict y, int32_t j) {{
  const float *__restrict acc = ms + {C.MS_ACC};
  aie::store_v(y, aie::load_v<32>(acc + 64 * j));
  aie::store_v(y + 32, aie::load_v<32>(acc + 64 * j + 32));
}}
}}
''',
    }
    out.update(DNX)
    out.update(q8_files(R))
    return out


STALE = ["gemv_q4_p2b16r2_g.cc", "gemv_q4_p2b16r2_gu.cc", "gemv_q4_p2b32r2_g.cc", "gemv_q4_p2b8r4_g.cc",
         "gemv_q4_r2h2.cc", "gemv_q4_prep_f32_k512.cc", "moe_acc2.cc", "moe_fin2.cc"]


def generate(R, out: Path = HERE) -> int:
    """Write the TUs for recipe R into `out` (only files whose text changed); returns the count."""
    fs = files(R)
    for name, src in fs.items():
        p = out / name
        if not p.is_file() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8", newline="\n")
    gone = list(STALE) + [n for n in Q8_FILES + FOLD_FILES if n not in fs]
    if mixed(R):                      # the folded entry replaces the pair on disk too
        gone += [n for n in ("gemv_q4_gy.cc", "gemv_q4_gms.cc") if n not in fs]
    for name in gone:
        p = out / name
        if p.is_file():
            p.unlink()
    return len(fs)


if __name__ == "__main__":
    n = generate(current_recipe())
    print(f"{n} kernel files")
