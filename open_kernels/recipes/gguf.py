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
# The quant name a GGUF-direct model's manifest section is built under: the q4_1
# chunk law (the one every container uses, and the one the q4_1 twins run), with
# the pack ops in their GGUF-reading form. See section().
GGUF_QUANT = "q4_1_gguf"
# The env flag that tells a design to compile the f32-scale variant.
SCALES_ENV = "GEMV_SCALES_F32"


def section(spec, max_ctx: int, programs, twins_built: bool | None = None) -> str | None:
    """The manifest's "gguf" section: the q4_1 program's own view, GGUF-reading.

    A GGUF model now runs on the q4_1 twin -- the chunk law and the kernels every
    shipped container uses -- so the section names the BASE builds and the base
    chunk law, with each projection op in its GGUF-reading form (`std_perm_gguf`).
    Numerically that is the model a q4nx container of the same checkpoint serves.

    The parameter is now "is this family GGUF-ready at all" rather than "was an
    f32-scale twin built": the twins are no longer what a GGUF runs on, so a model
    that cannot build one (a split FF: Qwen3-14B) still gets a section -- that was the
    bug that made `oflm add` refuse Qwen/Qwen3-14B-GGUF while the set existed. The MoE
    passes False while GGUF_EXPERT_OPS_READY is off.
    """
    if spec.quant != BASE_QUANT or twins_built is False:
        # `twins_built=False` is now an explicit "this family is not GGUF-ready yet"
        # (the MoE, whose expert pack ops are not written). Omitted/None = enabled.
        return None
    # The QUANT NAME the nested manifest is built under, not the manifest itself:
    # recipes/manifest.py recurses with it (programs(quant=GGUF_QUANT) -> the q4_1
    # chunk law with GGUF-reading pack ops). Returning a dict here made that
    # recursion take a dict as a quant and silently fall back to q4_1's plain
    # std_perm ops -- which fail on a GGUF source.
    return GGUF_QUANT


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
