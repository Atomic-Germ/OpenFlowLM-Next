"""The PRODUCTION attention worker, lifted out of the layer design.

The attention gate (designs/attn/probe_dense.py) must measure the kernel the
engine ships, not a copy of it: a second implementation of the attention would
pass while the shipped one broke. So the worker body is taken from the design
the family actually builds -- designs/dense/dx.py's `_attn` for the dense
recipe, designs/lfm3/lg.py's `attn_core` for a gated hybrid, and whatever a
family declares in its module's `ATTN_WORKER` otherwise. Only its surroundings
(the geometry, the core count, the loop primitive) are supplied here.

The extraction is AST-level on purpose: importing the design would build the
whole layer (pools, GEMVs, the norm) and need a model-free build of all of it.
The worker body itself is the production source, verbatim.

The worker is presented to the probe by keyword, filtered down to the names the
lifted function actually takes. That is what lets one probe bind both bodies
(dx's `_attn` and lg's `attn_core`) without a per-family copy of the wrapper:
the parameters are the loose signature.
"""
from __future__ import annotations

import ast
from pathlib import Path

DX = Path(__file__).resolve().parents[1] / "dense" / "dx.py"

# Every name a worker body of either shape may take. A family that declares
# `ATTN_WORKER` only has to use these names for the arguments it wants bound.
ARGS = ("ain", "aout", "ogout", "qn", "kn", "cs", "qs", "tmp", "kout", "vout",
        "oacc", "ml", "pb", "pbuf", "h0", "c", "f_meta", "f_q", "f_k", "f_v",
        "f_init", "f_step", "f_stepn", "f_fin", "f_stepb", "bias_in", "kw")


def source_path(source=None) -> Path:
    """The file the worker is lifted from (default: the dense recipe's design)."""
    if not source:
        return DX
    p = Path(source[0])
    return p if p.is_absolute() else DX.parents[2] / p


def worker(g, nptab: int, cse: int, rb: int, acores: int, nhl: int, n_og: int, kvw: int, loop,
           source=None, gate: bool = False):
    """The production attention worker body, bound to this geometry.

    `source` is a family module's (path, function-name) pair -- the design that
    family ships. Without one the dense recipe's `_attn` is lifted, which is
    what every ungated model has always gated.

    Returns (body, params): `params` is the lifted function's own argument
    names, so the probe can bind the body by keyword and stay independent of
    the order a given design happens to use.

    `loop` is the IRON control-flow primitive (aie.iron.controlflow.range_), so
    the body keeps its `for h in range_(...)` loops -- they are part of the
    production code and the gate must not compile them differently.
    """
    path = source_path(source)
    fname = source[1] if source else "_attn"
    tree = ast.parse(path.read_text())
    body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            body = node
            break
    if body is None:
        raise SystemExit(f"probe_support: {path} has no function {fname!r} for this family's ATTN_WORKER")
    params = tuple(a.arg for a in body.args.args)
    ns = {"G": g, "NPTAB": nptab, "CSE": cse, "RB": rb, "ACORES": acores, "NHL": nhl,
          "N_OG": n_og, "KVW": kvw, "range_": loop, "QLOCAL": bool(getattr(g, "QLOCAL", 0)),
          "NOGATE": not gate}
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(path), "exec"), ns)
    return ns[fname], params
