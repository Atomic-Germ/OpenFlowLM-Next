"""The PRODUCTION attention worker, lifted out of the layer design.

The attention gate (designs/attn/probe_dense.py) must measure the kernel the
engine ships, not a copy of it: a second implementation of the attention would
pass while the shipped one broke. So the worker body is taken from
designs/dense/dx.py -- the design the dense recipe builds -- and only its
surroundings (the geometry, the core count, the loop primitive) are supplied
here.

The extraction is AST-level on purpose: importing dx.py would build the whole
layer design (pools, GEMVs, the norm) and need a model-free build of all of it.
The worker body itself is the production source, verbatim.
"""
from __future__ import annotations

import ast
from pathlib import Path

DX = Path(__file__).resolve().parents[1] / "dense" / "dx.py"


def worker(g, nptab: int, cse: int, rb: int, acores: int, nhl: int, n_og: int, kvw: int, loop):
    """The `_attn` worker of dx.py, bound to this geometry.

    `loop` is the IRON control-flow primitive (aie.iron.controlflow.range_), so
    the body keeps its `for h in range_(...)` loops -- they are part of the
    production code and the gate must not compile them differently.
    """
    tree = ast.parse(DX.read_text())
    dx = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dx")
    body = next(n for n in dx.body if isinstance(n, ast.FunctionDef) and n.name == "_attn")
    ns = {"G": g, "NPTAB": nptab, "CSE": cse, "RB": rb, "ACORES": acores, "NHL": nhl,
          "N_OG": n_og, "KVW": kvw, "range_": loop}
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(DX), "exec"), ns)
    return ns["_attn"]
