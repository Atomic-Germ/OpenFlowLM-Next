r"""vux: dux.py's twin for two positions per weight pass, bit-identical per position (OPEN-DECODE-SPECULATIVE-VERIFY).
VUX_PART=<0|1|2> OPEN_KERNELS_SPEC=<spec.json> OFLM_TERNARY_FORMAT=t2 python build_design.py designs/verify/vux.py designs/verify/build_vux<part>
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
LX = HERE.parent / "layer_x"
GEMV = HERE.parent / "gemv_q4"
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(LX))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_H, A_O, A_OG, A_OUT, A_OUT2, A_OUT2B, A_QKV, A_RES,  # noqa: E402
                    A_VEC, A_XM, A_XN, A_Z,
                    AA_BYTES, AA_H, AA_KVN, AA_OG, AA_OUT, AA_OUT2, AA_OUT2B, AA_QG, AA_RES, AA_XM, AA_XN,
                    C_BYTES, C_LNW, C_NW, C_POSTLN, C_SIDE, C_WOUT, CA_BYTES, CA_LNW, CA_META, CA_POSTLN,
                    ELN, GLUE_SIDE_BYTES, KV_BYTES, KV_ROW, POOL_BYTES, POOL_FFN_DOWN, POOL_FFN_GATE,
                    POOL_FFN_UP, POOL_GATE, POOL_K, POOL_O, POOL_Q, POOL_QKV, POOL_V, POOL_Z, PTAB_BYTES,
                    PTAB_ROW, S_HEAD_BYTES, SIDE_ALPHA, SIDE_BETA, SIDE_CONV, SIDE_SMALL, STATE_BYTES,
                    STATE_S_OFF, R, SPEC)
import xcommon as X  # noqa: E402
import gen_kernels as GK  # noqa: E402

VW = int(os.environ.get("VUX_W", "2"))
if VW != 2:
    sys.exit("vux.py: the verify image is two positions wide (the main core holds two tables, no more)")

D = R.linear
A = R.attn
if D is None or A is None or X.KIND != "dense" or not X.T2 or X.Q8 or X.MIXED or not X.LN_SPLIT:
    sys.exit("vux.py: the verify image is the all-t2 merged dense image's twin (dux.py)")

HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
QKV_PC, Z_PC, OUT_PC, OUT_K = D.QKV_PC, D.Z_PC, D.OUT_PC, D.OUT_K
NH, KVH, HD = A.NH, A.KVH, A.HD
Q_PC, KV_PC, O_PC, O_K = A.Q_PC, A.KV_PC, A.O_PC, A.O_K
QW, KVW = A.QW, A.KVW
assert (OUT_PC, OUT_K, D.OG_ELEMS) == (O_PC, O_K, A.OG_ELEMS)
XN_ELEMS = D.XN_SIDE_ELEMS
OG_ELEMS = D.OG_ELEMS
NBANDS_LIN = QKV_PC + Z_PC
NBANDS_FULL = 2 * Q_PC + 2 * KV_PC
DN_HEADS_PC = X.DN_HEADS_PC
assert len(X.DOWN_SPLIT) == 2 and X.PER_CALL == 1, (X.DOWN_SPLIT, X.PER_CALL)
assert OG_ELEMS == XN_ELEMS == X.FFN.XM_ELEMS
assert all(X.n_groups(k) == X.per_band(k) for k in (HID, OUT_K, *X.DOWN_SPLIT))
KC = [VW * XN_ELEMS, X.n_groups(HID), X.n_groups(OUT_K), len(X.DOWN_SPLIT), VW]
H1_K = X.DOWN_SPLIT[1]
H1_BLOCKS = H1_K // 1024
H1_X = -(-H1_K * 2 // ELEM)
assert H1_K % 1024 == 0 and 2 * H1_K + H1_K // 16 <= H1_X * ELEM <= X.TAB_BYTES
KH = {"hk": [X.DOWN_SPLIT[0], 0], "hn": [VW * X.DOWN_SPLIT[0] * 4 // ELEM, VW * H1_X],
      "hg": [X.n_groups(k) for k in X.DOWN_SPLIT],
      "dk": [2, 16, 64]}                    # vdn_body's halves, S elements per half, row pairs: run-time, so not unrolled
HPREP_TILE = Tile(7, 4)

# 15 = fully unrolled, which compiles smaller than a partial unroll for the two-token tile
VX_OUNROLL = int(os.environ.get("VUX_OUNROLL", "15"))
TAB = X.TAB_BYTES
MSF = X.MS_FLOATS
VDN_BLK, VDN_RS = 2 * 128 * 32 * 4, 1600       # vdn.h's half-head block and record slot
VDN_BYTES = VDN_BLK + (VW + 1) * VDN_RS + 3 * 256 + VW * 512
assert VDN_BYTES <= VW * TAB, (VDN_BYTES, VW * TAB)      # DeltaNet's block, records and sums inside the tables
assert SPEC.lin_value_dim == 128 and S_HEAD_BYTES == 128 * 512

# ---- the linear layer's helpers (lx.py)
NCH, NHEAD = D.NCH, D.NHEAD
TILE, NT = D.TILE, D.NT
G, NG = D.G, D.NG
CONV_ROWS = SPEC.conv_kernel - 1
KEY_TILES = D.VALUE_TILE0
VALUE_TILES = NT - KEY_TILES
CONVW_ELEMS = SPEC.conv_kernel * TILE * 2 // ELEM
AB_TILES = [min(ELEM // 2, HID - h * (ELEM // 2)) // D.AB_ROWS for h in range(XN_ELEMS)]
assert sum(AB_TILES) == D.AB_ELEMS
AB_WIDE = D.AB_LANES > 32
HALF_OUTER = D.GLUE_HALF_OUTER
GLUE_FLAGS = {} if NHEAD == 32 else {"compile_flags": [f"-DDNGLUE_NHEAD={NHEAD}"]}

# ---- the attention cores (ax.py / dux.py, unchanged)
from recipes.attnknobs import probe_env  # noqa: E402
ATTN_FLAGS = [f"-DATTN_NH={NH}", f"-DATTN_KVH={KVH}", f"-DATTN_HD={HD}", f"-DATTN_ROT={A.ROT}", "-DATTN_GATE=1",
              f"-DATTN_VEXP={A.VEXP}", f"-DATTN_NHL={A.NHL}"]
if A.RB > 1:
    ATTN_FLAGS.append(f"-DATTN_RB={A.RB}")
    ATTN_FLAGS += ["-DATTN_BLOCK_WIN=1", "-DATTN_INTFP=1", "-DATTN_TREE=1"]
for _k, _v in probe_env().items():
    if _k not in ("ATTN_RB", "ATTN_FAST"):
        ATTN_FLAGS.append(f"-D{_k}={_v}")
ACORES, NHL, RB = A.ACORES, A.NHL, A.RB
N_OG = NHL // A.HPO
assert ACORES >= 2 and N_OG == 1 and RB > 1
OG_BYTES = NHL * HD * 2
OGJ_TILE = Tile(3, 1)

PART = int(os.environ.get("VUX_PART", 0))
if PART not in (0, 1, 2):
    sys.exit(f"vux.py: VUX_PART={PART} (0 = vlx, 1 = vax, 2 = vlm)")

# ---- the tail (dux.py part 2)
LM_BANDS = SPEC.vocab // X.BAND_ROWS
LM_BANDS_PC = LM_BANDS // N_CORES
LM_BAND_BYTES = X.role_band_bytes("linear", HID)
LM_POOL = LM_BANDS * LM_BAND_BYTES
LM_ACT = 64 * 1024
LM_JUNK = 32 * 1024
from recipes import qwen35 as _Q35  # noqa: E402
if not _Q35.tail_in_layer(SPEC) or LM_POOL != _Q35.lm_t2_pool_bytes(SPEC):
    sys.exit("vux.py: the verify image needs the merged image's in-layer ternary head")

# ---- buffer-argument shapes, one set for every part
SLOT = -(-STATE_BYTES // 4096) * 4096
CONSTS_T = max(C_BYTES, CA_BYTES)
ACT1 = max(A_BYTES, AA_BYTES)
ACT_T = VW * ACT1
STATE_T = max((VW + 1) * SLOT, KV_BYTES, VW * SPEC.vocab * 4)
XRES_T = VW * HID
GLUE_TILE = Tile(2, 4)
YB = X.BAND_ROWS * 4


# ---- the main cores' kernels: the 1-wide sources with a position offset (generated here)
def _sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def gen_sources() -> dict[str, Path]:
    fs = GK.files(R)
    src = {
        # both positions' activations arrive back to back, so element i belongs to position i / ne
        "vx_prep.cc": _sub(fs["dense_prep.cc"],
                           "void dense_prep(bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {\n",
                           "void vx_prep(bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {\n"
                           "  {\n    const unsigned ne = (2u * (unsigned)K + 4095u) / 4096u, t = (unsigned)i / ne;\n"
                           f"    i -= (int32_t)(t * ne);\n    tab += {TAB}u * t;\n  }}\n"),
        "vx_prepf.cc": _sub(fs["dense_prep_f32.cc"],
                            "void dense_prep_f32(bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {\n",
                            "void vx_prepf(bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {\n"
                            f"  {{\n    const unsigned ne = K == 0 ? {H1_X}u : (unsigned)K * 4u / 4096u, t = (unsigned)i / ne;\n"
                            f"    i -= (int32_t)(t * ne);\n    tab += {TAB}u * t;\n  }}\n"),
        # dense_act's own text, so the SwiGLU cannot drift from the decode image's
        "vx_act.cc": _sub(fs["dense_act.cc"],
                          "void dense_act(const float *__restrict ms, float *__restrict h) {\n",
                          "__attribute__((noinline)) static void vx_act1(const float *__restrict ms, float *__restrict h) {\n")
                     + 'extern "C" {\nvoid vx_act(const float *__restrict ms, float *__restrict h0, float *__restrict h1) {\n'
                       f"  vx_act1(ms, h0);\n  vx_act1(ms + {MSF}u, h1);\n}}\n}}\n",
        "vx_gy.cc": f'''// GENERATED by vux.py
#define GEMV_PER_CALL {X.PER_CALL}
#define GEMV_T2_CHUNK {X.TILE}
#define GEMV_T2_OUNROLL {VX_OUNROLL}
#define GEMV_X2_HOIST 1
#include "gemv_t2x2.h"
extern "C" {{
void vx_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y0, float *__restrict y1,
           int32_t group, int32_t per_band, int32_t rs) {{
  gemv_t2x2_chunk_rt(t, tab, {TAB}u, (unsigned)group, (unsigned)per_band, y0, y1, (unsigned)rs);
}}
}}
''',
        "vx_gms.cc": f'''// GENERATED by vux.py
#define GEMV_PER_CALL {X.PER_CALL}
#define GEMV_T2_CHUNK {X.TILE}
#define GEMV_T2_OUNROLL {VX_OUNROLL}
#define GEMV_X2_HOIST 1
#include "gemv_t2x2.h"
extern "C" {{
void vx_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
            int32_t group, int32_t per_band, int32_t dst) {{
  gemv_t2x2_chunk_rt(t, tab, {TAB}u, (unsigned)group, (unsigned)per_band, ms + dst, ms + {MSF} + dst, 2);
}}
}}
''',
    }
    # records leave as their k|q and v|decay|beta halves, so the glue core never holds a position's q / k across tiles
    ab = open(GLUE / "glue_ab_w.cc", encoding="utf-8").read().replace("\r\n", "\n")
    head, body = ab.split('extern "C" {\n', 1)
    body = body.rstrip()
    body = body[: body.rindex("}")].rstrip() + "\n"            # drop the extern block's brace
    body = _sub(body, "void glue_ab_w(", "__attribute__((noinline)) static void vx_ab1(")
    gpre = "// GENERATED by vux.py\n"
    src["vg_ab.cc"] = (gpre + head + body + 'extern "C" {\n'
                       "void vg_ab(const bfloat16 *__restrict W, const bfloat16 *__restrict xn, float *__restrict acc, int tile, int h) {\n"
                       "  const int first = h == 0 ? 1 : 0;      // the host loop's: set for the first xn half only\n"
                       f"  for (unsigned p = 0; p < {VW}u; ++p) vx_ab1(W, xn + 2048u * p, acc + {D.AB_LANES}u * p, tile, first);\n"
                       "}\n}\n")
    src["vg_copy.cc"] = (gpre + '#include "dn_glue.h"\nextern "C" {\n'
                         "void vg_copy(const bfloat16 *__restrict s, bfloat16 *__restrict xn, int p) {\n"
                         "  bfloat16 *__restrict d = xn + 2048u * (unsigned)p;\n"
                         "  for (unsigned j = 0; j < 2048u; j += kV) aie::store_v(d + j, aie::load_v<kV>(s + j));\n}\n}\n")
    src["vg_small.cc"] = (gpre + '#include "dn_glue.h"\nextern "C" {\n'
                          "void vg_small(const float *__restrict small, const float *__restrict a, const float *__restrict b,\n"
                          "              float *__restrict decay, float *__restrict beta) {\n"
                          f"  for (unsigned p = 0; p < {VW}u; ++p)\n"
                          f"    glue_small(small, a + {D.AB_LANES}u * p, b + {D.AB_LANES}u * p, decay + kNHead * p, beta + kNHead * p);\n"
                          "}\n}\n")
    gh = open(GLUE / "dn_glue.h", encoding="utf-8").read().replace("\r\n", "\n")
    i0 = gh.index("static inline void glue_conv_tile(")
    fn = gh[i0: gh.index("// ---- emit record", i0)]
    fn = _sub(fn, "static inline void glue_conv_tile(", "static inline void vg_conv_tile(")
    fn = _sub(fn, "                                  bfloat16 *__restrict ns0, bfloat16 *__restrict ns1,\n"
                  "                                  bfloat16 *__restrict ns2, float *__restrict qk,\n"
                  "                                  float *__restrict vt, unsigned t) {",
              "                                  bfloat16 *__restrict ns2, float *__restrict qk, unsigned t) {")
    fn = _sub(fn, "  float *__restrict dst = (t < kKeyTiles) ? (qk + t * kTile) : vt;", "  float *__restrict dst = qk;")
    fn = _sub(fn, "    aie::store_v(ns0 + j, s1v);\n    aie::store_v(ns1 + j, s2v);\n", "")
    src["vg_conv.cc"] = (gpre + '#include "dn_glue.h"\n' + fn + 'extern "C" {\n'
                         "// position p of tile t: seq = [state rows 0..2 | bf16(x) of positions 0..], its rows are seq[p .. p + 2]\n"
                         "void vg_conv(const float *__restrict q0, const float *__restrict q1, bfloat16 *__restrict seq,\n"
                         "             const bfloat16 *__restrict w01, const bfloat16 *__restrict w23, float *__restrict T,\n"
                         "             int t, int base, int p) {\n"
                         "  const unsigned u = (unsigned)p;\n"
                         "  vg_conv_tile(q0, q1, seq + kTile * u, seq + kTile * (u + 1), seq + kTile * (u + 2), w01, w23,\n"
                         "               seq + kTile * (u + 3), T + kTile * u, (unsigned)(t + base));\n}\n}\n")
    src["vg_rows.cc"] = (gpre + '#include "dn_glue.h"\nextern "C" {\n'
                         "void vg_rows(const bfloat16 *__restrict r0, const bfloat16 *__restrict r1, const bfloat16 *__restrict r2,\n"
                         "             bfloat16 *__restrict seq) {\n"
                         "  for (unsigned j = 0; j < kTile; j += kV) {\n"
                         "    aie::store_v(seq + j, aie::load_v<kV>(r0 + j));\n"
                         "    aie::store_v(seq + kTile + j, aie::load_v<kV>(r1 + j));\n"
                         "    aie::store_v(seq + 2 * kTile + j, aie::load_v<kV>(r2 + j));\n  }\n}\n}\n")
    src["vg_row.cc"] = (gpre + '#include "dn_glue.h"\nextern "C" {\n'
                         "// row k of slot p + 1's conv state: seq[p + 1 + k]\n"
                         "void vg_row(const bfloat16 *__restrict seq, bfloat16 *__restrict o, int p, int k) {\n"
                         "  const bfloat16 *__restrict r = seq + kTile * (unsigned)(p + 1 + k);\n"
                         "  for (unsigned j = 0; j < kTile; j += kV) aie::store_v(o + j, aie::load_v<kV>(r + j));\n}\n}\n")
    # key tile, element e of position p: four of its eight heads' 128 values, each once per value head that reads it
    src["vg_kq.cc"] = (gpre + '#include "dn_glue.h"\nextern "C" {\n'
                         "void vg_kq(const float *__restrict T, float *__restrict o, int e, int p) {\n"
                         "  const float *__restrict t = T + kTile * (unsigned)p;\n"
                         "  for (unsigned k = 0; k < 4; ++k) {\n"
                         "    const unsigned kh = (4u * (unsigned)e + k) / kGrp;\n"
                         "    for (unsigned j = 0; j < kHD; j += kV) aie::store_v(o + kHD * k + j, aie::load_v<kV>(t + kHD * kh + j));\n"
                         "  }\n}\n}\n")
    src["vg_v.cc"] = (gpre + '#include "dn_glue.h"\nextern "C" {\n'
                         "// value tile tv, element e of position p: heads 2e, 2e + 1 of the tile as [v | decay | beta | -] (1 KB each)\n"
                         "void vg_v(const float *__restrict T, const float *__restrict decay, const float *__restrict beta,\n"
                         "          float *__restrict o, int e, int p, int tv) {\n"
                         "  const float *__restrict t = T + kTile * (unsigned)p;\n"
                         "  for (unsigned k = 0; k < 2; ++k) {\n"
                         "    const unsigned hl = 2u * (unsigned)e + k, h = 8u * (unsigned)tv + hl;\n"
                         "    float *__restrict d = o + 256u * k;\n"
                         "    for (unsigned j = 0; j < kHD; j += kV) aie::store_v(d + j, aie::load_v<kV>(t + kHD * hl + j));\n"
                         "    d[kHD] = decay[kNHead * (unsigned)p + h];\n"
                         "    d[kHD + 1] = beta[kNHead * (unsigned)p + h];\n"
                         "  }\n}\n}\n")
    vdn = {"vdn_rec": ("const float *__restrict e, uint8_t *__restrict tab, int32_t p", "e, tab, (unsigned)p"),
           "vdn_ld": ("const float *__restrict e, uint8_t *__restrict tab, int32_t ei", "e, tab, (unsigned)ei"),
           "vdn_row": ("uint8_t *__restrict tab, float *__restrict y0, float *__restrict y1, int32_t p, int32_t r2, "
                       "int32_t h", "tab, y0, y1, (unsigned)p, (unsigned)r2, (unsigned)h"),
           "vdn_ofin": ("const uint8_t *__restrict tab, float *__restrict ye, int32_t p, int32_t h",
                        "tab, ye, (unsigned)(2 * p + h)")}
    vdn_h = hashlib.sha1((HERE / "vdn.h").read_bytes()).hexdigest()[:12]     # IRON's object cache keys on the TU text
    for sym, (params, args) in vdn.items():
        src[f"{sym}.cc"] = (f'// GENERATED by vux.py (vdn.h {vdn_h})\n#include "vdn.h"\nextern "C" {{\n'
                            f"void {sym}({params}) {{ {sym}_impl({args}); }}\n}}\n")
    # attention: attn.h with each core keeping only its own heads' q, and wrappers taking a query index p
    ah = open(ATTN / "attn.h", encoding="utf-8").read().replace("\r\n", "\n")
    ah = _sub(ah, "static constexpr unsigned kQW = kNH * kHD;   // q, stored PRE-SPLIT: [hi bf16[QW] | lo bf16[QW]]\n",
              "static constexpr unsigned kQW = kNH * kHD;   // q, stored PRE-SPLIT: [hi bf16[QW] | lo bf16[QW]]\n"
              "static constexpr unsigned kQL = kNHL * kHD;  // vux: a core stores only its own heads' q [hi | lo]\n")
    for old in ("qh + kQW + j", "qa + kQW + j", "qb + kQW + j"):
        ah = _sub(ah, old, old.replace("kQW", "kQL"))
    assert ah.count("q + kQW + j") == 2
    ah = ah.replace("q + kQW + j", "q + kQL + j")
    assert ah.count("const bfloat16 *q = qs + h * kHD;") == 2
    ah = ah.replace("const bfloat16 *q = qs + h * kHD;", "const bfloat16 *q = qs + hl * kHD;")
    ah = _sub(ah, "const float *q = qs + h * kHD;", "const float *q = qs + hl * kHD;")
    ah = _sub(ah, "const bfloat16 *q0 = qs + hb * kHD;", "const bfloat16 *q0 = qs;")
    assert "kQW + j" not in ah
    src["vattn.h"] = "// GENERATED by vux.py from designs/attn/attn.h: q kept per core (kQL)\n" + ah
    va = (f'// vattn.h {hashlib.sha1(ah.encode()).hexdigest()[:12]}\n#include "vattn.h"\n#if ATTN_QKV_BIAS || ATTN_SINK || ATTN_PTAB_SPLIT || !ATTN_BLOCK_WIN || !ATTN_VEXP\n'
          '#error "vux attention wrappers"\n#endif\n')
    QS, OA, ML = "(2u * kQL)", "(kNHL * kHD)", "(2u * kMLS)"
    src["vattn_meta.cc"] = va + (
        'extern "C" {\nvoid vattn_meta(const uint8_t *__restrict m0, const uint8_t *__restrict m1, bfloat16 *__restrict qn,\n'
        '                bfloat16 *__restrict kn, float *__restrict cs, int32_t *__restrict pb, int p) {\n'
        '  attn_meta_impl(m0, m1, qn, kn, cs + (unsigned)p * kRot, pb + 8u * (unsigned)p);\n}\n}\n')
    src["vattn_q.cc"] = va + (
        'extern "C" {\nvoid vattn_q(const float *__restrict qe, const bfloat16 *__restrict qn, const float *__restrict cs,\n'
        '             bfloat16 *__restrict qs, int h, int p) {\n'
        f'  attn_q_impl(qe, qn, cs + (unsigned)p * kRot, qs + (unsigned)p * {QS}, h);\n}}\n}}\n')
    src["vattn_k.cc"] = va + (
        'extern "C" {\nvoid vattn_k(const float *__restrict ke, const bfloat16 *__restrict kn, const float *__restrict cs,\n'
        '             float *__restrict tmp, bfloat16 *__restrict kout, int h, int p) {\n'
        '  attn_k_impl(ke, kn, cs + (unsigned)p * kRot, tmp, kout, h);\n}\n}\n')
    src["vattn_v.cc"] = va + (
        'extern "C" {\nvoid vattn_v(const float *__restrict ve, bfloat16 *__restrict vout, int h) { attn_v_impl(ve, vout, h); }\n}\n')
    src["vattn_init.cc"] = va + (
        'extern "C" {\nvoid vattn_init(float *__restrict oacc, float *__restrict ml, int p) {\n'
        f'  attn_init_impl(oacc + (unsigned)p * {OA}, ml + (unsigned)p * {ML});\n}}\n}}\n')
    sb = open(ATTN / "attn_stepb.cc", encoding="utf-8").read().replace("\r\n", "\n")
    sb = _sub(sb, '#include "attn.h"\n', va)
    assert sb.count("void attn_stepb(") == 2 and sb.count("int32_t *__restrict pb ATTN_H0_PARM) {\n") == 2
    sb = sb.replace("void attn_stepb(", "void vattn_stepb(")
    sb = sb.replace("int32_t *__restrict pb ATTN_H0_PARM) {\n",
                    "int32_t *__restrict pb ATTN_H0_PARM, int p) {\n"
                    f"  qs += (unsigned)p * {QS};\n  oacc += (unsigned)p * {OA};\n  ml += (unsigned)p * {ML};\n"
                    "  pb += 8u * (unsigned)p;\n")
    src["vattn_stepb.cc"] = sb
    # the position's own row read back from the cache: the bytes attn_stepb_nr takes from kout / vout
    src["vattn_stepb_nre.cc"] = va + (
        'extern "C" {\nvoid vattn_stepb_nre(const bfloat16 *__restrict Kn, const bfloat16 *__restrict Vn,\n'
        '                     const bfloat16 *__restrict qs, float *__restrict oacc, float *__restrict ml, int h0, int p) {\n'
        '  const bfloat16 *K[kRB];\n  const bfloat16 *V[kRB];\n'
        '  for (unsigned r = 0; r < kRB; ++r) {\n    K[r] = Kn;\n    V[r] = Vn;\n  }\n'
        f'  attn_rowb_impl(K, V, qs + (unsigned)p * {QS}, oacc + (unsigned)p * {OA}, ml + (unsigned)p * {ML}, 0u, kRB - 1u, h0);\n'
        '}\n}\n')
    src["vattn_fin.cc"] = va + (
        'extern "C" {\nvoid vattn_fin(const float *__restrict oacc, const float *__restrict ml, const float *__restrict g0,\n'
        '               const float *__restrict g1, bfloat16 *__restrict og, int hp, int p) {\n'
        f'  attn_fin_impl(oacc + (unsigned)p * {OA}, ml + (unsigned)p * {ML}, g0, g1, og, hp);\n}}\n}}\n')
    # copied here because a worktree that never exported has none of them in layer_x
    for name in ("dense_hblk.cc", "dense_hout.cc", "xh_signs.h"):
        src[name] = fs[name]
    out = {}
    for name, text in src.items():
        p = HERE / name
        if not p.is_file() or p.read_text(encoding="utf-8") != text:
            p.write_text(text, encoding="utf-8", newline="\n")
        out[name] = p
    return out


SRC = gen_sources()


def rows3(t: int, slot: int):
    """Tile t of each conv-state row in state slot `slot`."""
    return TensorAccessPattern((1, STATE_T), slot * SLOT + t * TILE * 2, [1, 1, CONV_ROWS, TILE * 2], [0, 0, NCH * 2, 1])


def bt(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def y2(total: int, off: int, nbands: int, pstride: int) -> TensorAccessPattern:
    """Each band drains as [position 0 | position 1], position j landing at off + j * pstride."""
    return TensorAccessPattern((1, total), off, [1, nbands, VW, YB], [0, YB, pstride, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def vux(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
        srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    u8_4k = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    u8_2k = np.ndarray[(2048,), np.dtype[np.uint8]]
    u8_ln = tl["u8_ln"]
    pool_ty = np.ndarray[(LM_POOL if part == 2 else POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(XRES_T,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CONSTS_T,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_T,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(ACT_T,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]
    rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]
    tab2 = np.ndarray[(VW * TAB,), np.dtype[np.uint8]]
    ms2 = np.ndarray[(VW * MSF,), np.dtype[np.float32]]
    e_ty, x_ty, y_ty = t["elem"], t["x"], t["y"]
    i32 = np.int32
    nw_ty = np.ndarray[(SPEC.lin_value_dim,), np.dtype[bfloat16]]
    f32 = np.ndarray[(NHEAD,), np.dtype[np.float32]]
    facc = np.ndarray[(D.AB_LANES,), np.dtype[np.float32]] if AB_WIDE else f32
    fqk = np.ndarray[(2 * D.KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(TILE,), np.dtype[np.float32]]
    fxn = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    u8_1k = np.ndarray[(A.E_A,), np.dtype[np.uint8]]
    b256 = np.ndarray[(HD,), np.dtype[bfloat16]]
    b512 = np.ndarray[(KVW,), np.dtype[bfloat16]]
    f64 = np.ndarray[(A.ROT,), np.dtype[np.float32]]
    f256 = np.ndarray[(HD,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]] if A.VEXP else np.ndarray[(QW,), np.dtype[np.float32]]
    fml = np.ndarray[(2 * A.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    pb_ty = np.ndarray[(8,), np.dtype[np.int32]]
    ogj_ty = np.ndarray[((ACORES - 1) * KVW,), np.dtype[bfloat16]]

    inc = [str(HERE)] + include_dirs() + [str(LX), str(GEMV), str(GLUE), str(POST), str(ATTN), str(X.LN),
                                          str(X.LINL), str(X.RT), str(HERE.parent / "moe_experts")]

    def vf(sym, args, flags):
        return ExternalFunction(sym, source_file=str(SRC[f"{sym}.cc"]), arg_types=args, include_dirs=inc,
                                compile_flags=flags)

    def dnf(sym, args):                    # the tables: no stage reads them while DeltaNet runs
        return ExternalFunction(sym, source_file=str(SRC[f"{sym}.cc"]), arg_types=args, include_dirs=inc,
                                compile_flags=X.OS + [f"-DVDN_NPOS={VW}"])

    K = {
        "gy": vf("vx_gy", [e_ty, tab2, y_ty, y_ty, i32, i32, i32], X.OS),
        "gms": vf("vx_gms", [e_ty, tab2, ms2, i32, i32, i32], X.OS),
        "act": vf("vx_act", [ms2, y_ty, y_ty], X.OS + ["-DVECMATH_COMPACT"]),
        "prep": vf("vx_prep", [x_ty, tab2, i32, i32], X.OZ),
        "prepf": vf("vx_prepf", [x_ty, tab2, i32, i32], X.OZ),
        "vrec": dnf("vdn_rec", [e_ty, tab2, i32]),
        "vld": dnf("vdn_ld", [e_ty, tab2, i32]),
        "vrow": dnf("vdn_row", [tab2, y_ty, y_ty, i32, i32, i32]),
        "vofin": dnf("vdn_ofin", [tab2, y_ty, i32, i32]),
    }
    KN = tuple(K)
    L = X.ln_kernels(inc, tl)
    assert AB_WIDE and HALF_OUTER, "the glue wraps glue_ab_w in its half-outer order"
    assert D.HEADS_PER_TILE == 8 and CONV_ROWS == 3 and SPEC.conv_kernel == 4
    xt_ty = np.ndarray[(VW * TILE,), np.dtype[np.float32]]          # every position's xn half, then its conv tile
    seq_ty = np.ndarray[((CONV_ROWS + VW) * TILE,), np.dtype[bfloat16]]
    accn_ty = np.ndarray[(VW * D.AB_LANES,), np.dtype[np.float32]]
    dbn_ty = np.ndarray[(VW * NHEAD,), np.dtype[np.float32]]

    def gf(sym, src_name, args):
        return ExternalFunction(sym, source_file=str(SRC[src_name]), arg_types=args, include_dirs=inc, **GLUE_FLAGS)

    f_gcopy = gf("vg_copy", "vg_copy.cc", [u8_4k, xt_ty, i32])
    f_gab = gf("vg_ab", "vg_ab.cc", [u8_4k, xt_ty, accn_ty, i32, i32])
    f_gsmall = gf("vg_small", "vg_small.cc", [u8_4k, accn_ty, accn_ty, dbn_ty, dbn_ty])
    f_gconv = gf("vg_conv", "vg_conv.cc", [u8_2k, u8_2k, seq_ty, u8_4k, u8_4k, xt_ty, i32, i32, i32])
    f_grows = gf("vg_rows", "vg_rows.cc", [u8_2k, u8_2k, u8_2k, seq_ty])
    f_grow = gf("vg_row", "vg_row.cc", [seq_ty, u8_2k, i32, i32])
    f_gkq = gf("vg_kq", "vg_kq.cc", [xt_ty, u8_2k, i32, i32])
    f_gv = gf("vg_v", "vg_v.cc", [xt_ty, dbn_ty, dbn_ty, u8_2k, i32, i32, i32])
    post_fn = ExternalFunction("post_fn", source_file=str(POST / "post.cc"), arg_types=[u8_4k, u8_4k, nw_ty, u8_2k],
                               include_dirs=inc)
    post_copy = ExternalFunction("post_copy_nw", source_file=str(POST / "post_copy.cc"), arg_types=[u8_4k, nw_ty],
                                 include_dirs=inc)

    def af(sym, args):
        return ExternalFunction(sym, source_file=str(SRC[f"{sym}.cc"]), arg_types=args, include_dirs=inc,
                                compile_flags=ATTN_FLAGS)

    # per query, side by side: rope cos / sin, block counters, this core's q, o, (m, l)
    csv_ty = np.ndarray[(VW * A.ROT,), np.dtype[np.float32]]
    pbv_ty = np.ndarray[(VW * 8,), np.dtype[np.int32]]
    qsv_ty = np.ndarray[(VW * 2 * NHL * HD,), np.dtype[bfloat16]]
    oav_ty = np.ndarray[(VW * NHL * HD,), np.dtype[np.float32]]
    mlv_ty = np.ndarray[(VW * 2 * A.MLS,), np.dtype[np.float32]]
    f_meta = af("vattn_meta", [u8_1k, u8_1k, b256, b256, csv_ty, pbv_ty, i32])
    f_q = af("vattn_q", [u8_1k, b256, csv_ty, qsv_ty, i32, i32])
    f_k = af("vattn_k", [u8_1k, b256, csv_ty, f256, b512, i32, i32])
    f_v = af("vattn_v", [u8_1k, b512, i32])
    f_init = af("vattn_init", [oav_ty, mlv_ty, i32])
    f_stepn = af("vattn_stepb_nre", [u8_1k, u8_1k, qsv_ty, oav_ty, mlv_ty, i32, i32])
    f_stepb = af("vattn_stepb", [u8_1k] * (2 * RB) + [qsv_ty, oav_ty, mlv_ty, pbv_ty, i32, i32])
    f_fin = af("vattn_fin", [oav_ty, mlv_ty, u8_1k, u8_1k, b512, i32, i32])

    # ---- fifos: dux.py's, but x one deep (two tables fill the core) and y four deep (a band acquires two)
    of_w = [ObjectFifo(e_ty, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=4) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=1)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=X.LNI_DEPTH)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_side = ObjectFifo(u8_4k, name="side", depth=2)
    of_gact = ObjectFifo(u8_2k, name="gact", depth=5)
    of_gout = ObjectFifo(u8_2k, name="gout", depth=3)
    of_pin = ObjectFifo(u8_4k, name="pin", depth=2)
    of_pout = ObjectFifo(u8_2k, name="pout", depth=2)
    of_ain = ObjectFifo(u8_1k, name="ain", depth=max(4, 2 * RB + 2))
    of_aout = ObjectFifo(b512, name="aout", depth=2)
    of_ogj = ObjectFifo(ogj_ty, name="ogj", depth=2)
    og_subs = of_ogj.prod().join([c * KVW for c in range(ACORES - 1)], tile=OGJ_TILE,
                                 depths=[2] * (ACORES - 1), obj_types=[b512] * (ACORES - 1),
                                 names=[f"og{c}" for c in range(1, ACORES)])

    rtp = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(4, dtype=np.int32), use_write_rtp=True)
           for c in range(N_CORES)]

    def count_bufs(c):
        i32n = lambda v: np.array(v, dtype=np.int32)
        return ([Buffer(np.ndarray[(len(KC),), np.dtype[np.int32]], name=f"kc{c}", initial_value=i32n(KC))]
                + [Buffer(np.ndarray[(len(v),), np.dtype[np.int32]], name=f"{n}{c}", initial_value=i32n(v))
                   for n, v in KH.items()])

    # ---- the main cores
    def prep2(xin, prep, tab, kk, nx):
        for i in range_(nx):
            xe = xin.acquire(1)
            prep(xe, tab, kk, i)
            xin.release(1)

    def gemv2(win, yout, tab, gy, nbands, ng):
        for _ in range_(nbands):
            ye = yout.acquire(2)
            for g in range_(ng):
                we = win.acquire(1)
                gy(we, tab, ye[0], ye[1], g, ng, 2)
                win.release(1)
            yout.release(2)

    def vdn_body(win, yout, tab, Kc, nheads, npos, dk):
        for _ in range_(nheads):
            for p in range_(npos):
                e = win.acquire(1)
                Kc["vrec"](e, tab, p)
                win.release(1)
            for h in range_(dk[0]):
                for ei in range_(dk[1]):
                    e = win.acquire(1)
                    Kc["vld"](e, tab, ei)
                    win.release(1)
                for p in range_(npos):
                    for r2 in range_(dk[2]):
                        ye = yout.acquire(2)
                        Kc["vrow"](tab, ye[0], ye[1], p, r2, h)
                        yout.release(2)
            for p in range_(npos):
                for h in range_(dk[0]):
                    ye = yout.acquire(1)
                    Kc["vofin"](tab, ye, p, h)
                    yout.release(1)

    def main_body(win, xin, yout, my_rtp, kc, hk, hn, hg, dk, tab, ms, *ks):
        Kc = dict(zip(KN, ks))
        nx, ng_hid, ng_out, npieces = kc[0], kc[1], kc[2], kc[3]
        prep2(xin, Kc["prep"], tab, HID, nx)
        nbands, nheads, npos = my_rtp[0], my_rtp[1], my_rtp[2]
        gemv2(win, yout, tab, Kc["gy"], nbands, ng_hid)

        def layer_rest():
            vdn_body(win, yout, tab, Kc, nheads, npos, dk)
            prep2(xin, Kc["prep"], tab, OUT_K, nx)
            gemv2(win, yout, tab, Kc["gy"], OUT_PC, ng_out)
            # the dense FFN (xcommon.ffn_body, two positions)
            prep2(xin, Kc["prep"], tab, HID, nx)
            for _ in range_(X.FFN.UP_PC):
                for g in range_(ng_hid):
                    we = win.acquire(1)
                    Kc["gms"](we, tab, ms, g, ng_hid, R.ffn.MS_U)
                    win.release(1)
                for g in range_(ng_hid):
                    we = win.acquire(1)
                    Kc["gms"](we, tab, ms, g, ng_hid, R.ffn.MS_G)
                    win.release(1)
                ye = yout.acquire(2)
                Kc["act"](ms, ye[0], ye[1])
                yout.release(2)
            for pc in range_(npieces):
                prep2(xin, Kc["prepf"], tab, hk[pc], hn[pc])
                gemv2(win, yout, tab, Kc["gy"], X.FFN.DOWN_PC, hg[pc])

        for _ in range_(my_rtp[3]):                            # 1 | 0: the tail stops after the head
            layer_rest()

    # ---- the norm helper: each of its three stages once per position
    def vln_body(ain, aout, f_nr, f_add2, f_add3):
        for _ in range_(VW):                                  # 1. entry norm  [x0 x1 lnw] -> [xn]
            e = ain.acquire(3)
            o = aout.acquire(1)
            f_nr(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)
        for _ in range_(VW):                                  # 2. res = x + out; xm = norm(res)
            for _ in range(2):
                e = ain.acquire(2)
                o = aout.acquire(1)
                f_add2(e[0], e[1], o)
                aout.release(1)
                ain.release(2)
            e = ain.acquire(3)
            o = aout.acquire(1)
            f_nr(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)
        for _ in range_(VW):                                  # 3. xres = res + (out2 + out2b)
            for _ in range(2):
                e = ain.acquire(3)
                o = aout.acquire(1)
                f_add3(e[0], e[1], e[2], o)
                aout.release(1)
                ain.release(3)

    # ---- glue, post, attention, h prep: dux.py's bodies, once per position
    def glue_body(sin, ain, oout, abt, gk, acc_a, acc_b, decay, beta, xt, seq, fcopy, fab, fsmall, fconv, frows, frow,
                  fkq, fv):
        """The state rows are copied out first: release() frees a fifo's oldest elements, so none can be held."""
        npos = gk[0]
        for h in range_(len(AB_TILES)):                         # run-time loop: one copy of the body, not one per half
            for p in range_(npos):
                e = sin.acquire(1)
                fcopy(e, xt, p)
                sin.release(1)
            for acc in (acc_a, acc_b):
                for tile in range_(abt[h]):
                    ww = sin.acquire(1)
                    fab(ww, xt, acc, tile, h)
                    sin.release(1)
        sm = sin.acquire(1)
        fsmall(sm, acc_a, acc_b, decay, beta)
        sin.release(1)
        for base, ntiles, nel in ((0, KEY_TILES, gk[1]), (KEY_TILES, VALUE_TILES, gk[2])):
            for tt in range_(ntiles):
                ww = sin.acquire(CONVW_ELEMS)
                r = ain.acquire(CONV_ROWS)
                frows(r[0], r[1], r[2], seq)
                ain.release(CONV_ROWS)
                for p in range_(npos):
                    q = ain.acquire(2)
                    fconv(q[0], q[1], seq, ww[0], ww[1], xt, tt, base, p)
                    ain.release(2)
                sin.release(CONVW_ELEMS)
                for p in range_(npos):
                    for e in range_(nel):
                        o = oout.acquire(1)
                        if base == 0:
                            fkq(xt, o, e, p)
                        else:
                            fv(xt, decay, beta, o, e, p, tt)
                        oout.release(1)
                for p in range_(npos):
                    for k in range_(gk[3]):
                        o = oout.acquire(1)
                        frow(seq, o, p, k)
                        oout.release(1)

    def post_body(ain, aout, nwb, f, fc):
        """Group-major: group g is DeltaNet round g, done for every position at once."""
        e = ain.acquire(1)
        fc(e, nwb)
        ain.release(1)
        for _ in range_(NG):
            for _ in range_(VW):
                e = ain.acquire(2)
                r = aout.acquire(1)
                f(e[0], e[1], nwb, r)
                aout.release(1)
                ain.release(2)

    HPE = NH // A.Q_AIN_ELEMS                 # heads per q element
    assert NHL % HPE == 0

    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, an, fm, fq_, fk, fv, fi, fsn, ff, fsb, c):
        """One window walk serves every query: blocks past a query's position are all padding to it."""
        h0, e0, ne = c * NHL, c * NHL // HPE, NHL // HPE
        npos = an[0]

        def skip(k, w=1):
            if k:
                for _ in range_(k):
                    ain.acquire(w)
                    ain.release(w)

        for p in range_(npos):
            e = ain.acquire(2)
            fm(e[0], e[1], qn, kn, cs, pb, p)
            ain.release(2)
            skip(e0)
            for h in range_(ne):
                e = ain.acquire(1)
                fq_(e, qn, cs, qs, h, p)
                ain.release(1)
            skip(A.Q_AIN_ELEMS - e0 - ne)
        for p in range_(npos):
            if aout is not None:
                for h in range_(A.K_AIN_ELEMS):
                    e = ain.acquire(1)
                    fk(e, kn, cs, tmp, kout, h, p)
                    ain.release(1)
                for h in range_(A.K_AIN_ELEMS):
                    e = ain.acquire(1)
                    fv(e, vout, h)
                    ain.release(1)
                o = aout.acquire(1)
                for j in range_(KVW):
                    o[j] = kout[j]
                aout.release(1)
                o = aout.acquire(1)
                for j in range_(KVW):
                    o[j] = vout[j]
                aout.release(1)
            else:
                skip(2 * A.K_AIN_ELEMS)
        for p in range_(npos):
            fi(oacc, ml, p)
        for _ in range_(pb[(VW - 1) * 8 + 4]):                  # the last query's window covers every query's
            e = ain.acquire(2 * RB)
            for p in range_(npos):
                fsb(*([e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb, h0, p]))
            ain.release(2 * RB)
        for p in range_(npos):
            e = ain.acquire(2)
            fsn(e[0], e[1], qs, oacc, ml, h0, p)
            ain.release(2)
            skip(c * N_OG, 2)
            for hp in range_(N_OG):
                g = ain.acquire(2)
                o = ogout.acquire(1)
                ff(oacc, ml, g[0], g[1], o, hp, p)
                ogout.release(1)
                ain.release(2)
            skip((ACORES - 1 - c) * N_OG, 2)

    def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, an, fm, fq_, fk, fv, fi, fsn, ff, fsb):
        _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, an, fm, fq_, fk, fv, fi, fsn, ff, fsb, 0)

    def make_attn_body(c):
        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, an, fm, fq_, fk, fv, fi, fsn, ff, fsb):
            _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, an, fm, fq_, fk, fv, fi, fsn, ff, fsb, c)
        return body

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(b256, name=f"qn{s}"), Buffer(b256, name=f"kn{s}"), Buffer(csv_ty, name=f"cs{s}"),
                Buffer(qsv_ty, name=f"qs{s}"), Buffer(f256, name=f"tmp{s}"), Buffer(b512, name=f"kout{s}"),
                Buffer(b512, name=f"vout{s}"), Buffer(oav_ty, name=f"oacc{s}"), Buffer(mlv_ty, name=f"ml{s}"),
                Buffer(pbv_ty, name=f"pb{s}"),
                Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"an{s}", initial_value=np.array([VW, 0], dtype=np.int32))]

    workers = [Worker(vln_body, fn_args=[of_lni.cons(), of_lno.prod(), L["ln_nr"], L["ln_add2"], L["ln_add3"]],
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body,
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), rtp[c], *count_bufs(c),
                                       Buffer(tab2, name=f"tab{c}"), Buffer(ms2, name=f"ms{c}"), *[K[n] for n in KN]],
                              tile=Tile(c, 2), stack_size=0x1800))
    f32_1k = np.ndarray[(1024,), np.dtype[np.float32]]
    of_hpi = ObjectFifo(f32_1k, name="hpi", depth=2)
    u8_tab = np.ndarray[(H1_X * ELEM,), np.dtype[np.uint8]]
    of_hpo = ObjectFifo(u8_4k, name="hpo", depth=2)
    f_hblk = ExternalFunction("dense_hblk", source_file=str(SRC["dense_hblk.cc"]),
                              arg_types=[f32_1k, u8_tab, np.int32, np.int32], include_dirs=inc, compile_flags=X.OZ)
    f_hout = ExternalFunction("dense_hout", source_file=str(SRC["dense_hout.cc"]),
                              arg_types=[u8_tab, u8_4k, np.int32], include_dirs=inc, compile_flags=X.OZ)

    def hprep_body(hin, hout, htab, fb, fo):
        for b in range_(H1_BLOCKS):
            e = hin.acquire(1)
            fb(e, htab, H1_K, b)
            hin.release(1)
        for g in range_(H1_X):
            o = hout.acquire(1)
            fo(htab, o, g)
            hout.release(1)

    workers.append(Worker(hprep_body, fn_args=[of_hpi.cons(), of_hpo.prod(), Buffer(u8_tab, name="hptab"),
                                               f_hblk, f_hout], tile=HPREP_TILE, stack_size=0x800))
    workers.append(Worker(post_body, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(nw_ty, name="nwb"), post_fn, post_copy],
                          tile=Tile(1, 3), stack_size=0x1800))
    workers.append(Worker(glue_body,
                          fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                   Buffer(np.ndarray[(len(AB_TILES),), np.dtype[np.int32]], name="abt",
                                          initial_value=np.array(AB_TILES, dtype=np.int32)),
                                   Buffer(np.ndarray[(4,), np.dtype[np.int32]], name="gk",
                                          initial_value=np.array([VW, 6, 4, CONV_ROWS], dtype=np.int32)),
                                   Buffer(accn_ty, name="acc_a"), Buffer(accn_ty, name="acc_b"),
                                   Buffer(dbn_ty, name="decay"), Buffer(dbn_ty, name="beta"),
                                   Buffer(xt_ty, name="xt"), Buffer(seq_ty, name="seq"),
                                   f_gcopy, f_gab, f_gsmall, f_gconv, f_grows, f_grow, f_gkq, f_gv],
                          tile=GLUE_TILE, stack_size=0x1800))
    afns = [f_meta, f_q, f_k, f_v, f_init, f_stepn, f_fin, f_stepb]
    workers.append(Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=0x1800))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), og_subs[c - 1].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=0x1800))

    def set_rtp(nbands: int, nheads: int, rest: int = 1):
        for c in range(N_CORES):
            rtp[c][0] = nbands
            rtp[c][1] = nheads
            rtp[c][2] = VW
            rtp[c][3] = rest

    def entry_norm(pl, lni, lno, c_xres, a_consts, a_act, lnw_off, xn_off):
        for p in range(VW):
            pl.fill(lni, c_xres, bt(XRES_T, p * HID, HID))
            pl.fill(lni, a_consts, bt(CONSTS_T, lnw_off, ELN))
            pl.drain(lno, a_act, bt(ACT_T, p * ACT1 + xn_off, ELN))

    def residual_norm(lni, lno, c_xres, a_act, a_consts, a_out, a_res, a_xm, c_postln, after=None):
        """xcommon.ln_split_residual_norm once per position; after(p) once position p's xm is in DDR."""
        hh = HID // 2
        for p in range(VW):
            o = p * ACT1
            pl = Pipeline(3)
            for i in range(2):
                pl.fill(lni, c_xres, bt(XRES_T, p * HID + i * hh, hh))
                pl.fill(lni, a_act, bt(ACT_T, o + a_out + i * ELN, ELN))
            pl.drain(lno, a_act, bt(ACT_T, o + a_res, HID * 4))
            pl.finish()
            pl.fill(lni, a_act, bt(ACT_T, o + a_res, HID * 4))
            pl.fill(lni, a_consts, bt(CONSTS_T, c_postln, ELN))
            pl.drain(lno, a_act, bt(ACT_T, o + a_xm, ELN))
            pl.finish()
            if after is not None:
                after(p)

    def close(lni, lno, c_xres, a_act, a_res, a_out2, a_out2b):
        """xcommon.ln_split_close once per position."""
        for p in range(VW):
            o = p * ACT1
            pl = Pipeline(3)
            for i in range(2):
                pl.fill(lni, a_act, bt(ACT_T, o + a_res + i * ELN, ELN))
                pl.fill(lni, a_act, bt(ACT_T, o + a_out2 + i * ELN, ELN))
                pl.fill(lni, a_act, bt(ACT_T, o + a_out2b + i * ELN, ELN))
            pl.drain(lno, c_xres, bt(XRES_T, p * HID, HID))
            pl.finish()

    def ffn(pw, px, py, a_pool, a_act, w_prods, x_prod, y_conss, a_xm, a_h, a_out2, a_out2b, hpi, hpo, a_ht):
        """xcommon.ffn_sequence for two positions: one weight pass, both positions' x and y."""
        bb_h, bb_f = X.role_band_bytes("ffn", HID), X.role_band_bytes("ffn", X.FF)
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, y2(ACT_T, a_h + c * X.FFN.UP_PC * YB, X.FFN.UP_PC, ACT1))
        for j in range(X.FFN.UP_PC):
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_FFN_UP + (c * X.FFN.UP_PC + j) * bb_h, bb_h))
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_FFN_GATE + (c * X.FFN.UP_PC + j) * bb_h, bb_h))
        py.finish(*y_conss)                                   # h is in DDR, both positions
        k0, k1 = X.DOWN_SPLIT
        for p in range(VW):
            px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + a_h, k0 * 4))
        ph = Pipeline(3)
        for p in range(VW):
            ph.fill(hpi, a_act, bt(ACT_T, p * ACT1 + a_h + k0 * 4, k1 * 4))
            ph.drain(hpo, a_act, bt(ACT_T, p * ACT1 + a_ht, H1_X * ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, X.piece_tap(POOL_FFN_DOWN + c * X.FFN.DOWN_PC * bb_f, bb_f,
                                                    X.role_band_bytes("ffn", k0), X.FFN.DOWN_PC))
            py.drain(y_conss[c], a_act, y2(ACT_T, a_out2 + c * X.FFN.DOWN_PC * YB, X.FFN.DOWN_PC, ACT1))
        ph.finish()                                           # both positions' piece-1 tables are in DDR
        for p in range(VW):
            px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + a_ht, H1_X * ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, X.piece_tap(POOL_FFN_DOWN + c * X.FFN.DOWN_PC * bb_f + X.role_band_bytes("ffn", k0),
                                                    bb_f, X.role_band_bytes("ffn", k1), X.FFN.DOWN_PC))
            py.drain(y_conss[c], a_act, y2(ACT_T, a_out2b + c * X.FFN.DOWN_PC * YB, X.FFN.DOWN_PC, ACT1))

    def dn_post(pw, py, a_state, a_act, a_consts, w_prods, y_conss, pin_p, pout_c, glue_wait):
        """Round h is head 8 h + c on core c for every position; post group h follows round h."""
        rec, ohb = D.RECORD_BYTES, D.O_HEAD_BYTES
        pp = Pipeline(3)
        pp.fill(pin_p, a_consts, bt(CONSTS_T, C_NW, ELEM))

        def post_group(g):
            for p in range(VW):
                o = p * ACT1
                pp.drain(pout_c, a_act, bt(ACT_T, o + A_OG + g * G * 2, G * 2))
                pp.fill(pin_p, a_act, bt(ACT_T, o + A_O + g * G * 4, G * 4))
                pp.fill(pin_p, a_act, bt(ACT_T, o + A_Z + g * G * 4, G * 4))

        def drains(hd):
            s_off = STATE_S_OFF + hd * S_HEAD_BYTES
            out = [TensorAccessPattern((1, STATE_T), (p + 1) * SLOT + s_off + h2 * 256, [1, 1, 128, 256], [0, 0, 512, 1])
                   for h2 in range(2) for p in range(VW)]
            return [(a_state, t) for t in out] + [
                (a_act, TensorAccessPattern((1, ACT_T), A_O + hd * ohb, [1, 1, VW, ohb], [0, 0, ACT1, 1]))]

        for h in range(DN_HEADS_PC):
            glue_wait(h)
            for c in range(N_CORES):
                hd = h * N_CORES + c
                pw.fill(w_prods[c], a_act, TensorAccessPattern((1, ACT_T), A_VEC + hd * rec, [1, 1, VW, X.CALL_BYTES],
                                                               [0, 0, ACT1, 1]))
                # [half][quarter plane][rows 0..135][128 B]: 17 rows an element; rows past 127 are never used
                pw.fill(w_prods[c], a_state, TensorAccessPattern((1, STATE_T), STATE_S_OFF + hd * S_HEAD_BYTES,
                                                                 [2, 2, 136, 128], [256, 128, 512, 1]))
            dr = [drains(h * N_CORES + c) for c in range(N_CORES)]
            for k in range(len(dr[0])):
                if k == 3 and h:
                    post_group(h - 1)                       # the throttle has retired round h - 1's o drains
                for c in range(N_CORES):
                    py.drain(y_conss[c], *dr[c][k])
        py.finish(*y_conss)
        post_group(DN_HEADS_PC - 1)
        pp.finish()                                         # every position's og is in DDR

    # ---- part 0: the linear layer, two positions
    def linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                        side_p, gact_p, gout_c, pin_p, pout_c, hpi, hpo):
        set_rtp(NBANDS_LIN, DN_HEADS_PC)
        BB_HID, BB_OUT = X.role_band_bytes("linear", HID), X.role_band_bytes("linear_out", OUT_K)
        pl = Pipeline(3)
        entry_norm(pl, lni, lno, c_xres, a_consts, a_act, C_LNW, A_XN)
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
        for p in range(VW):                                   # position p's xn as soon as it is in DDR
            pl.finish_oldest(lno)
            px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + A_XN, XN_ELEMS * ELEM))
        pl.finish()
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
            py.drain(y_conss[c], a_act, y2(ACT_T, A_QKV + c * QKV_PC * YB, QKV_PC, ACT1))
            py.drain(y_conss[c], a_act, y2(ACT_T, A_Z + c * Z_PC * YB, Z_PC, ACT1))

        def side_ab(ps):
            """Fill order must match vg_ab's half-outer loop."""
            off = 0
            for h, ntiles in enumerate(AB_TILES):
                for p in range(VW):
                    ps.fill(side_p, a_act, bt(ACT_T, p * ACT1 + A_XN + h * ELEM, ELEM))
                for reg in (SIDE_ALPHA, SIDE_BETA):
                    ps.fill(side_p, a_consts, bt(CONSTS_T, C_SIDE + reg + off, ntiles * ELEM))
                off += ntiles * ELEM

        G3 = D.RECORD_BYTES * 3                               # the three value heads that read one key head
        assert NHEAD // (D.KEY_WIDTH // 128) == 3

        def conv_tile(pipe, tt):
            pipe.fill(gact_p, a_state, rows3(tt, 0))
            pipe.fill(gact_p, a_act, TensorAccessPattern((1, ACT_T), A_QKV + tt * TILE * 4, [1, 1, VW, TILE * 4],
                                                         [0, 0, ACT1, 1]))
            if tt < KEY_TILES:                                # q (tiles 0, 1) at record + 512, k (2, 3) at + 0
                off = A_VEC + 8 * (tt % 2) * G3 + (512 if tt < 2 else 0)
                tap = TensorAccessPattern((1, ACT_T), off, [VW, 8, 3, 512], [ACT1, G3, D.RECORD_BYTES, 1])
            else:
                off = A_VEC + 8 * (tt - KEY_TILES) * D.RECORD_BYTES + 1024
                tap = TensorAccessPattern((1, ACT_T), off, [1, VW, 8, 1024], [0, ACT1, D.RECORD_BYTES, 1])
            pipe.drain(gout_c, a_act, tap)
            rec = pipe._q(gout_c)[-1]
            for p in range(VW):
                pipe.drain(gout_c, a_state, rows3(tt, p + 1))
            return rec

        ps = Pipeline(3)
        side_ab(ps)
        ps.fill(side_p, a_consts, bt(CONSTS_T, C_SIDE + SIDE_SMALL, ELEM))
        ps.fill(side_p, a_consts, bt(CONSTS_T, C_SIDE + SIDE_CONV, GLUE_SIDE_BYTES - SIDE_CONV))
        py.finish_oldest(*y_conss)                            # qkv of every position is in DDR (z is still running)
        pipe = Pipeline(3)
        recs = [conv_tile(pipe, tt) for tt in range(KEY_TILES + 1)]

        def glue_wait(h):
            """Round h needs value tile h (heads 8h .. 8h + 7); tile h + 1 is issued first so the glue runs ahead."""
            while len(recs) < min(KEY_TILES + h + 2, NT):
                recs.append(conv_tile(pipe, len(recs)))
            q = pipe._q(gout_c)
            while any(x is recs[KEY_TILES + h] for x in q):
                q.popleft().finish()

        dn_post(pw, py, a_state, a_act, a_consts, w_prods, y_conss, pin_p, pout_c, glue_wait)
        pipe.finish()
        ps.finish()
        for p in range(VW):
            px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + A_OG, OG_ELEMS * ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_consts, bt(CONSTS_T, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
            py.drain(y_conss[c], a_act, y2(ACT_T, A_OUT + c * OUT_PC * YB, OUT_PC, ACT1))
        py.finish()
        residual_norm(lni, lno, c_xres, a_act, a_consts, A_OUT, A_RES, A_XM, C_POSTLN,
                      lambda p: px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + A_XM, XN_ELEMS * ELEM)))
        ffn(pw, px, py, a_pool, a_act, w_prods, x_prod, y_conss, A_XM, A_H, A_OUT2, A_OUT2B, hpi, hpo, A_QKV)
        py.finish()
        close(lni, lno, c_xres, a_act, A_RES, A_OUT2, A_OUT2B)
        pw.finish()
        px.finish()

    # ---- part 1: the full layer, two positions; attention once per position
    def full_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                      ain_p, aout_c, ogj_c, hpi, hpo):
        set_rtp(NBANDS_FULL, 0)
        BB_HID, BB_O = X.role_band_bytes("attn", HID), X.role_band_bytes("attn", O_K)

        def w_regions(c):
            return [(POOL_Q + c * Q_PC * BB_HID, Q_PC * BB_HID), (POOL_GATE + c * Q_PC * BB_HID, Q_PC * BB_HID),
                    (POOL_K + c * KV_PC * BB_HID, KV_PC * BB_HID), (POOL_V + c * KV_PC * BB_HID, KV_PC * BB_HID),
                    (POOL_O + c * O_PC * BB_O, O_PC * BB_O)]

        def y_regions(c):
            qb, kb = Q_PC * YB, KV_PC * YB
            return [(AA_QG + c * qb, Q_PC), (AA_QG + QW * 4 + c * qb, Q_PC),
                    (AA_KVN + c * kb, KV_PC), (AA_KVN + KVW * 4 + c * kb, KV_PC),
                    (AA_OUT + c * O_PC * YB, O_PC)]

        pl = Pipeline(3)
        entry_norm(pl, lni, lno, c_xres, a_consts, a_act, CA_LNW, AA_XN)
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[0]))
        for p in range(VW):                                   # position p's xn as soon as it is in DDR
            pl.finish_oldest(lno)
            px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + AA_XN, XN_ELEMS * ELEM))
        pl.finish()
        # gate last, so the new rows reach the cache and the window walk starts while it runs
        def await_task(pipe, ep, tg):
            q = pipe._q(ep)
            while any(x is tg for x in q):
                q.popleft().finish()

        last = {}
        for c in range(N_CORES):
            for i in (2, 3):
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[i]))
            for i in (0, 2, 3):
                off, nb = y_regions(c)[i]
                py.drain(y_conss[c], a_act, y2(ACT_T, off, nb, ACT1))
                last[c, i] = py._q(y_conss[c])[-1]
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[1]))
            off, nb = y_regions(c)[1]
            py.drain(y_conss[c], a_act, y2(ACT_T, off, nb, ACT1))
            last[c, 1] = py._q(y_conss[c])[-1]
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            await_task(py, y_conss[c], last[c, 0])                  # q of every position is in DDR
        for p in range(VW):
            o = p * ACT1
            pa_in.fill(ain_p, a_consts, bt(CONSTS_T, CA_META, A.E_A))
            pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, (1 + p) * PTAB_ROW, PTAB_ROW))     # the record (attnposv)
            pa_in.fill(ain_p, a_act, bt(ACT_T, o + AA_QG, QW * 4))
        for c in range(N_CORES):
            await_task(py, y_conss[c], last[c, 3])                  # k and v
        for p in range(VW):
            o = p * ACT1
            pa_in.fill(ain_p, a_act, bt(ACT_T, o + AA_KVN, KVW * 4))
            pa_in.fill(ain_p, a_act, bt(ACT_T, o + AA_KVN + KVW * 4, KVW * 4))
        for p in range(VW):
            pa_out.drain(aout_c, a_kv, bt(STATE_T, (1 + p) * KV_ROW, KV_ROW))   # the new rows (attnposv)
        pa_out.finish()                                             # ... are in the cache
        pa_in.fill(ain_p, a_kv, bt(STATE_T, 0, KV_ROW))             # the window, once (attnposv: the last query's)
        for c in range(N_CORES):
            await_task(py, y_conss[c], last[c, 1])                  # gate
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[4]))
            off, nb = y_regions(c)[4]
            py.drain(y_conss[c], a_act, y2(ACT_T, off, nb, ACT1))
        for p in range(VW):
            o = p * ACT1
            pa_in.fill(ain_p, a_kv, bt(STATE_T, (1 + p) * KV_ROW, KV_ROW))     # its own row, back (attnposv)
            pa_in.fill(ain_p, a_act, bt(ACT_T, o + AA_QG + QW * 4, QW * 4))
        for p in range(VW):
            o = p * ACT1
            pa_out.drain(aout_c, a_act, bt(ACT_T, o + AA_OG, OG_BYTES))
            pa_out.drain(ogj_c, a_act, bt(ACT_T, o + AA_OG + OG_BYTES, (ACORES - 1) * OG_BYTES))
        pa_out.finish()
        pa_in.finish()
        for p in range(VW):
            px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + AA_OG, OG_ELEMS * ELEM))
        py.finish()
        residual_norm(lni, lno, c_xres, a_act, a_consts, AA_OUT, AA_RES, AA_XM, CA_POSTLN,
                      lambda p: px.fill(x_prod, a_act, bt(ACT_T, p * ACT1 + AA_XM, XN_ELEMS * ELEM)))
        ffn(pw, px, py, a_pool, a_act, w_prods, x_prod, y_conss, AA_XM, AA_H, AA_OUT2, AA_OUT2B, hpi, hpo, AA_QG)
        py.finish()
        close(lni, lno, c_xres, a_act, AA_RES, AA_OUT2, AA_OUT2B)
        pw.finish()
        px.finish()

    # ---- part 2: the tail. Args: lmpool, xres, normw (as consts), logits (as state), lmact (as act).
    def lm_sequence(a_pool, c_xres, a_norm, a_logits, a_act, lni, lno, w_prods, x_prod, y_conss):
        set_rtp(LM_BANDS_PC, 0, 0)
        pl = Pipeline(3)
        for p in range(VW):                                  # hn_p = final_norm(xres_p): stage 1
            pl.fill(lni, c_xres, bt(XRES_T, p * HID, HID))
            pl.fill(lni, a_norm, bt(ELN, 0, ELN))
            pl.drain(lno, a_act, bt(ACT_T, p * LM_ACT, ELN))
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        lstride = SPEC.vocab * 4
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(LM_POOL, c * LM_BANDS_PC * LM_BAND_BYTES, LM_BANDS_PC * LM_BAND_BYTES))
            py.drain(y_conss[c], a_logits, y2(STATE_T, c * LM_BANDS_PC * YB, LM_BANDS_PC, lstride))
        for p in range(VW):
            pl.finish_oldest(lno)
            px.fill(x_prod, a_act, bt(ACT_T, p * LM_ACT, XN_ELEMS * ELEM))
        pl.finish()
        # the norm helper's stages 2 and 3 on junk, once per position, so it is back on stage 1
        pj = Pipeline(3)
        for n_in in (2, 2, 3) * VW + (3, 3) * VW:
            for _ in range(n_in):
                pj.fill(lni, a_act, bt(ACT_T, 0, ELN))
            pj.drain(lno, a_act, bt(ACT_T, LM_JUNK, ELN))
        pj.finish()
        py.finish()
        pw.finish()
        px.finish()

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                 side_p, gact_p, gout_c, pin_p, pout_c, ain_p, aout_c, ogj_c, hpi, hpo):
        if part == 2:
            lm_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss)
        elif part == 0:
            linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                            side_p, gact_p, gout_c, pin_p, pout_c, hpi, hpo)
        else:
            full_sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                          ain_p, aout_c, ogj_c, hpi, hpo)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                            of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0)),
                            of_ain.prod(tile=Tile(5, 0)), of_aout.cons(tile=Tile(4, 0)), of_ogj.cons(tile=Tile(3, 0)),
                            of_hpi.prod(tile=Tile(7, 0)), of_hpo.cons(tile=Tile(7, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = vux
_src = b"".join([p.read_bytes() for p in sorted(SRC.values())] + [(HERE / "gemv_t2x2.h").read_bytes(),
                (HERE / "vdn.h").read_bytes(),
                (HERE / "vux.py").read_bytes()]
                + sorted(f.read_bytes() for f in LX.glob("*.cc")) + sorted(f.read_bytes() for f in LX.glob("*.h"))
                + [(LX / "xcommon.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.cc"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.h")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (GEMV / "wht.h").read_bytes(), (GEMV / "wht_int.h").read_bytes(),
                   (GEMV / "gemv_t2.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   (HERE.parent.parent / "include" / "scalar_fp.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
