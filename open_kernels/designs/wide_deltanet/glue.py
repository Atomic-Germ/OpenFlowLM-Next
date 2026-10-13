"""48-head conv/record dispatch consuming the separate WideDeltaNet AB result.

side = AB result f32[4,48], padded to 4096 B, then conv bf16[10,4,1024].
Other BOs: qkv f32[10240], old/new conv state bf16[3,10240], records f32[48,512].
The same design serves both hidden widths: xn is consumed by the AB dispatch.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
GLUE = HERE.parent / "dn_glue"
sys.path.insert(0, str(ROOT))
from ironutil import Pipeline, include_dirs

NHEAD, KEY_HEADS, HD, TILE = 48, 16, 128, 1024
KEY_WIDTH = KEY_HEADS * HD
NCH = 2 * KEY_WIDTH + NHEAD * HD
NT = NCH // TILE
KEY_TILES = 2 * KEY_WIDTH // TILE
VALUE_TILES = NT - KEY_TILES
HEADS_PER_TILE = TILE // HD
SIDE_BYTES = 4096 + NT * 4 * TILE * 2


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def wide_glue(side: In, qkv: In, state: In, nstate: Out, vec: Out,
              *, dummy: CompileTime[int] = 0):
    u8s = np.ndarray[(4096,), np.dtype[np.uint8]]
    u8a = np.ndarray[(2048,), np.dtype[np.uint8]]
    heads = np.ndarray[(NHEAD,), np.dtype[np.float32]]
    fqk = np.ndarray[(2 * KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(TILE,), np.dtype[np.float32]]
    side_ty = np.ndarray[(SIDE_BYTES,), np.dtype[np.uint8]]
    qkv_ty = np.ndarray[(NCH,), np.dtype[np.float32]]
    st_ty = np.ndarray[(3 * NCH,), np.dtype[bfloat16]]
    vec_ty = np.ndarray[(NHEAD * 512,), np.dtype[np.float32]]
    inc = include_dirs() + [str(GLUE)]
    flags = [f"-DDNGLUE_NHEAD={NHEAD}"]
    fload = ExternalFunction("wide_glue_load_ab", source_file=str(HERE / "load_ab.cc"),
                             arg_types=[u8s, heads, heads], include_dirs=inc, compile_flags=flags)
    fconv = ExternalFunction("glue_conv", source_file=str(GLUE / "glue_conv.cc"),
                             arg_types=[u8a, u8a, u8a, u8a, u8a, u8s, u8s, u8a, u8a, u8a,
                                        fqk, fvt, np.int32, np.int32],
                             include_dirs=inc, compile_flags=flags)
    femit = ExternalFunction("glue_emit_fn", source_file=str(GLUE / "glue_emit.cc"),
                             arg_types=[fqk, fvt, heads, heads, u8a, np.int32, np.int32],
                             include_dirs=inc, compile_flags=flags)
    side_fifo = ObjectFifo(u8s, name="conv_side", depth=2)
    act_fifo = ObjectFifo(u8a, name="conv_act", depth=5)
    out_fifo = ObjectFifo(u8a, name="conv_out", depth=3)
    decay, beta = Buffer(heads, name="decay"), Buffer(heads, name="beta")
    qk, vt = Buffer(fqk, name="qk"), Buffer(fvt, name="vt")

    def core_body(sin, ain, oout, decay, beta, qk, vt, fload, fconv, femit):
        ab = sin.acquire(1)
        fload(ab, decay, beta)
        sin.release(1)
        for base, ntiles in ((0, KEY_TILES), (KEY_TILES, VALUE_TILES)):
            for t in range_(ntiles):
                w = sin.acquire(2)
                e = ain.acquire(5)
                o = oout.acquire(3)
                fconv(e[0], e[1], e[2], e[3], e[4], w[0], w[1], o[0], o[1], o[2], qk, vt, t, base)
                oout.release(3)
                ain.release(5)
                sin.release(2)
                if base == KEY_TILES:
                    for i in range_(HEADS_PER_TILE):
                        r = oout.acquire(1)
                        femit(qk, vt, decay, beta, r, t, i)
                        oout.release(1)

    worker = Worker(core_body, fn_args=[side_fifo.cons(), act_fifo.cons(), out_fifo.prod(),
                                       decay, beta, qk, vt, fload, fconv, femit],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_side, a_qkv, a_state, c_nstate, c_vec, side_p, act_p, out_c):
        pipe = Pipeline(3)
        pipe.fill(side_p, a_side, TensorAccessPattern((1, SIDE_BYTES), 0,
                                                     [1, 1, 1, SIDE_BYTES], [0, 0, 0, 1]))
        for t in range(NT):
            state_tap = TensorAccessPattern((3, NCH), t * TILE, [1, 1, 3, TILE], [0, 0, NCH, 1])
            pipe.drain(out_c, c_nstate, state_tap)
            if t >= KEY_TILES:
                pipe.drain(out_c, c_vec, TensorAccessPattern((1, NHEAD * 512),
                           (t - KEY_TILES) * HEADS_PER_TILE * 512,
                           [1, 1, 1, HEADS_PER_TILE * 512], [0, 0, 0, 1]))
            pipe.fill(act_p, a_qkv, TensorAccessPattern((1, NCH), t * TILE,
                                                       [1, 1, 1, TILE], [0, 0, 0, 1]))
            pipe.fill(act_p, a_state, state_tap)
        pipe.finish()

    rt = Runtime(sequence, [side_ty, qkv_ty, st_ty, st_ty, vec_ty,
                            side_fifo.prod(), act_fifo.prod(), out_fifo.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = wide_glue
_sources = [Path(__file__), HERE / "load_ab.cc", ROOT / "ironutil.py", ROOT / "include/vecmath.h",
            GLUE / "dn_glue.h", GLUE / "glue_conv.cc", GLUE / "glue_emit.cc"]
SPECIALIZE = {"dummy": int(hashlib.sha256(b"".join(p.read_bytes() for p in _sources)).hexdigest()[:8], 16)}
