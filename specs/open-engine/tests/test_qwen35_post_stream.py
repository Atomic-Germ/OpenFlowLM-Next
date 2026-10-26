"""The shared post producer, FIFO and projection must agree on the output ABI."""
import ast
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from recipes import qwen35
from recipes.load import load_spec
from recipes.qwen36moe import linear
from test_qwen35_wide_glue import spec27, xlayer_module, lx_function, XLAYER, LX


@pytest.mark.parametrize("name", ["qwen35-0.8b", "qwen35-2b", "qwen35-4b", "qwen35-9b",
                                 "qwen35-27b", "qwen36-35b-a3b"])
def test_published_recipes_keep_the_existing_post_abi(name):
    spec = load_spec(XLAYER.parents[2] / "recipes/specs" / (name + ".json"))
    d = linear(spec)
    assert d.POST_F32 is False
    assert d.OG_ELEMS * 4096 == ((d.VW * 2 + 4095) // 4096) * 4096


@pytest.mark.parametrize("fp32", [False, True])
def test_shared_post_kernel_and_fifo_have_the_same_element_type(fp32):
    spec = spec27()
    d = replace(qwen35.recipe(spec).linear, POST_F32=fp32)
    fifos = {}

    def fifo(ty, name, depth):
        fifos[name] = SimpleNamespace(ty=ty, depth=depth)
        return fifos[name]

    ns = dict(np=np, bfloat16=np.uint16, R=SimpleNamespace(linear=d), SPEC=spec,
              X=SimpleNamespace(), ELEM=4096, HID=5120, N_CORES=8, DENSE=True,
              ObjectFifo=fifo, ExternalFunction=lambda name, **kw: SimpleNamespace(**kw),
              GLUE=Path("glue"), POST=Path("post"))
    tree = xlayer_module({"glue_post_types", "glue_post_kernels", "glue_post_fifos"})
    exec(compile(tree, str(XLAYER), "exec"), ns)
    g = ns["glue_post_types"]()
    kernels = ns["glue_post_kernels"]([], g)
    ns["glue_post_fifos"](g)
    expected = g["u8_4k" if fp32 else "u8_2k"]
    assert kernels["post"].arg_types[-1] == fifos["pout"].ty == expected
    assert fifos["pout"].depth == 2
    assert getattr(kernels["post"], "compile_flags", []) == (["-DDN_POST_FP32=1"] if fp32 else [])
    assert fifos["gout"].ty == g["u8_2k"]


@pytest.mark.parametrize("fp32", [False, True])
def test_shared_dense_projection_consumes_post_dtype(fp32):
    calls = []
    d = replace(qwen35.recipe(spec27()).linear, POST_F32=fp32)
    x = SimpleNamespace(unpack_args=lambda args: ({"tab": None}, {}),
                        prep_bands=lambda *args, **kw: calls.append((args, kw)),
                        dn_body=lambda *args: None, ffn_body=lambda *args: None)
    ns = dict(X=x, D=d, DENSE=True, HID=5120, XN_ELEMS=5,
              QKV_PC=d.QKV_PC, Z_PC=d.Z_PC, OUT_PC=d.OUT_PC,
              OUT_K=d.OUT_K, OG_ELEMS=6 if fp32 else 3)
    node = lx_function("main_body")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LX), "exec"), ns)
    ns["main_body"](None, None, None)
    assert len(calls) == 2
    assert calls[1][0][-1] == "linear_out"
    assert calls[1][1].get("fp32", False) is fp32


@pytest.mark.parametrize("fp32", [False, True])
def test_post_dma_covers_each_group_once_without_overlapping_scratch(fp32):
    d = replace(qwen35.recipe(spec27()).linear, POST_F32=fp32)
    drains = []
    loop = next(n for n in ast.walk(lx_function("dense_sequence"))
                if isinstance(n, ast.For) and "A_OG +" in ast.unparse(n))
    width = 4 if fp32 else 2
    start = 393216
    ns = dict(D=d, NG=6, G=1024, ELEM=4096, C_BYTES=8192, C_NW=0,
              A_BYTES=start + 6144 * width, A_OG=start, A_O=40960, A_Z=81920,
              a_act=None, a_consts=None, pin_p=None, pout_c=None,
              bt=lambda total, offset, size: (total, offset, size),
              pipe=SimpleNamespace(drain=lambda _, __, tap: drains.append(tap),
                                   fill=lambda *args: None))
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(LX), "exec"), ns)
    assert drains == [(start + 6144 * width, start + i * 1024 * width, 1024 * width)
                      for i in range(6)]


@pytest.mark.parametrize("fp32", [False, True])
def test_shared_post_store_specialization(tmp_path, fp32):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ preprocessor unavailable")
    (tmp_path / "vecmath.h").write_text("")
    source = XLAYER.parent.parent / "dn_post/post.cc"
    cmd = [compiler, "-E", "-P", "-I", str(tmp_path), str(source)]
    if fp32:
        cmd.insert(1, "-DDN_POST_FP32=1")
    text = subprocess.check_output(cmd, text=True)
    assert f"using post_output_t = {'float' if fp32 else 'bfloat16'};" in text
    assert ("aie::store_v(gh + j, r);" in text) is fp32
    assert ("rr.template to_vector<bfloat16>()" in text) is not fp32
