"""The GGUF-direct twin: the f32-scale kernel set and the manifest section.

A GGUF-direct model (one that ships a model.gguf instead of a q4nx container)
is read by the engine through the manifest's "gguf" section: the same program
against the f32-scale pool layout (the q4_1 codes with f32 block scales, see
open_kernels/gguf_pool.py). The section names the kernels it needs, and the
export builds them as the *_f32 twins.

The two must agree. A recipe that emits the section without building the
kernels it names hands the engine a manifest that points at kernels that were
never built -- it fails at load, far from the recipe that caused it. So the
section is emitted only when the twins exist, and this module is where that
decision is made (one implementation, shared by the recipes).

Nothing here is per-family: a recipe that can build the twin gets it, and a
recipe that cannot (the f32 build does not fit a core) says so and ships no
section.
"""
from __future__ import annotations

from dataclasses import replace

# The twin's quant name: the pool layout the f32-scale kernels compile for.
F32_QUANT = "q4_1_f32"
# The base quant a GGUF-direct model is loaded as (the q4nx container's).
BASE_QUANT = "q4_1"
# The env flag that tells a design to compile the f32-scale variant.
SCALES_ENV = "GEMV_SCALES_F32"


def section(spec, max_ctx: int, programs, twins_built: bool = True) -> dict | None:
    """The manifest's "gguf" section: the f32-scale program's own view.

    None when the model is not a GGUF-direct candidate (its quant is not
    q4_1) or when the recipe builds no f32-scale twin for it (`twins_built`
    False) -- the engine swaps this section in and then loads the kernels it
    names, so a section without those kernels is a trap, not a feature.
    """
    if spec.quant != BASE_QUANT or not twins_built:
        return None
    f32 = programs(replace(spec, quant=F32_QUANT), max_ctx)
    return {"contexts": f32["contexts"], "kernels": f32["kernels"],
            "layer_types": f32["layer_types"], "tail": f32["tail"], "globals": f32["globals"]}


def twin_builds(builds: dict, skip: tuple[str, ...] = ()) -> dict:
    """The f32-scale twins of the weight-consuming builds.

    The same design, compiled for the f32-scale pool layout: the build dir
    gains the _f32 suffix and the env says which variant to build. `skip`
    names the builds with no twin (the norm reads no pool; a head with its
    own scale format gets its own flag).
    """
    out: dict = {}
    for name, b in builds.items():
        if name in skip:
            continue
        out[name + "_f32"] = {**b, "build_dir": b["build_dir"] + "_f32",
                              "env": {**b.get("env", {}), SCALES_ENV: "1"}}
    return out
