"""The weight format, from the tool that WRITES the container.

`spec_hash` covers the quant map, so the kernel build and `oflm add` have to agree
on it byte for byte -- and they used to have to agree by reading the same
`model.q4nx` safetensors header, which the kernel build cannot do: it derives a
spec from a model's `config.json` alone, long before anyone has a container to
look at.

The container is not the authority, though. It is an OUTPUT of `q4nx-build`, and
`utilities/q4nx-build/configs/<family>.json` already states, per tensor, what
`q4nx-build` will write: a `default_tensor_type` for the file and a per-tensor
override in `name_map`. That is the decision, written down, before anything is
quantized -- so it is what both sides should read. Two readers of one authority
agree by construction; two readers of a container that may not exist yet cannot.

So this module answers one question: for a family (and size, where the family is
built per size), which of the spec's quant ROLES does `q4nx-build` write at q8
rather than q4_1? Everything else stays at the q4_1 default.

The configs install alongside the recipes (`share/oflm/utilities/q4nx-build/
configs` next to `share/oflm/open_kernels/recipes`), so an installed oflm can
answer this with no source checkout -- which is what lets `oflm add` link a
kernel set on a machine that only has the package.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .spec import ROLE_TENSORS, SpecError

# A per-tensor `default_tensor_type` at 8 bits, however the config spells it.
_Q8 = {"Q8_0", "Q8", "q8_0", "q8", "Q8_1", "q8_1"}
_Q4_1 = {"Q4_1", "Q4_0", "q4_1", "q4_0", "MXFP4", "Q4_K", "q4_k", "FP4"}

# The `configs/` name for a family, keyed by the family the RECIPES use. The
# catalogue's own family string is the other input (see config_candidates), and
# this is the fallback for a caller that only has a recipe family.
_BY_RECIPE_FAMILY = {
    "qwen36moe": "qwen35moe",
    "qwen35": "qwen3.5",
    "qwen3": "qwen3",
    "qwen2": "qwen2",
    "llama3": "llama",
    "gemma3": "gemma3",
    "hunyuan": "hunyuan",
    "granite": "granite",
    "phi3": "phi4",
    "lfm2": "lfm2",
    "gptoss": "gpt-oss",
}

# Families built one config per size, because their format genuinely differs by
# size and a single family-wide file would be a lie for the others. This is the
# user's "one per family and size" in the one place it is true of the FORMAT
# rather than of the kernels.
_PER_SIZE_FAMILIES = {"qwen3.5"}


def configs_dir() -> Path | None:
    """The `utilities/q4nx-build/configs/` tree, or None when it is not there.

    Searched in the same places `oflm add` looks for recipes: the source tree
    (a dev build), then the installed data tree beside the recipes. The install
    prefix is compiled into the binary, so the second is reachable from an
    installed oflm with no environment variable set.
    """
    env = os.environ.get("Q4NX_CONFIGS_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    here = Path(__file__).resolve()
    # <root>/open_kernels/recipes/q4nx_quant.py -> <root>/utilities/q4nx-build/configs
    src = here.parents[2] / "utilities" / "q4nx-build" / "configs"
    if src.is_dir():
        return src
    # Installed: <prefix>/share/oflm/open_kernels/recipes -> <prefix>/share/oflm
    share = here.parents[2].parent          # .../share/oflm
    inst = share / "utilities" / "q4nx-build" / "configs"
    if inst.is_dir():
        return inst
    return None


def config_candidates(recipe_family: str, cat_family: str | None = None,
                      size: str | None = None) -> list[str]:
    """Config stems to try, most specific first.

    A per-size family is tried at its size first (`qwen3.5_4b`), because that is
    the file that states THIS model's format; the family-wide name follows as the
    fallback for a size the configs have not split out yet.
    """
    base = _BY_RECIPE_FAMILY.get(recipe_family)
    if base is None and cat_family:
        base = cat_family
    if base is None:
        return []
    out: list[str] = []
    if base in _PER_SIZE_FAMILIES and size:
        out.append(f"{base}_{size.lower()}")
    out.append(base)
    return out


def load_config(recipe_family: str, cat_family: str | None = None,
                size: str | None = None) -> tuple[dict, Path] | None:
    """(parsed config, path) for the first candidate that exists, else None."""
    d = configs_dir()
    if d is None:
        return None
    for stem in config_candidates(recipe_family, cat_family, size):
        p = d / f"{stem}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8")), p
            except ValueError:
                continue
    return None


def _type_of(entry: Mapping[str, Any], file_default: str | None) -> str | None:
    """The tensor's type: its own override, else the file default. None if neither."""
    t = entry.get("default_tensor_type") or entry.get("tensor_type") or file_default
    if not isinstance(t, str):
        return None
    if t in _Q8:
        return "q8"
    if t in _Q4_1:
        return "q4_1"
    return None


def quant_map_for(family: str, cat_family: str | None = None,
                  size: str | None = None) -> dict[str, str] | None:
    """The roles `q4nx-build` writes at q8 for this family/size, or None.

    None means "no config found" -- NOT "everything is q4_1". The caller must
    decide what an absent authority means; guessing q4_1 would put a wrong
    format in a spec_hash, and a wrong hash links kernels whose weight layout
    does not match the container. That fails as silent corruption, not an error.

    Roles the config never mentions are omitted, so a stock container derives
    `{}` and the spec keeps hashing as the bare "q4_1" string, exactly as it
    does when a container header is read.
    """
    table = ROLE_TENSORS.get(family)
    if table is None:
        return None
    got = load_config(family, cat_family, size)
    if got is None:
        return None
    cfg, _path = got
    file_default = cfg.get("default_tensor_type")
    out: dict[str, str] = {}
    for _role_name, entry in (cfg.get("name_map") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        q4nx_name = entry.get("q4nx_name")
        if not isinstance(q4nx_name, str):
            continue
        role = table.get(_layer_prefix_sub(q4nx_name))
        if role is None:
            continue
        fmt = _type_of(entry, file_default)
        if fmt == "q8":
            out[role] = "q8"
    return out or None


def _layer_prefix_sub(name: str) -> str:
    """`model.layers.0.mlp.down_proj.weight` -> `mlp.down_proj.weight`.

    The same rewrite `quant_map_from_chunk_sizes` does, so a tensor named in a
    q4nx config resolves to the same role as the same tensor read out of a
    container header. The third component is the layer index -- a number in a
    header, the literal `{bid}` in a config's name_map -- and both go.
    """
    parts = name.split(".")
    while (len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers"
           and (parts[2].isdigit() or parts[2] == "{bid}")):
        parts = parts[3:]
    return ".".join(parts)


def find_config(recipe_family: str, cat_family: str | None = None,
                size: str | None = None) -> Path | None:
    """The q4nx-build config that states this family's format, or None.

    None means "no such config exists", which is what `require_quant_map` turns
    into a refusal. It is NOT the same as a config that declares no q8 tensor --
    that is a found config whose every role is q4_1, and it is the common case.
    """
    got = load_config(recipe_family, cat_family, size)
    return got[1] if got else None


def require_quant_map(family: str, cat_family: str | None = None,
                      size: str | None = None) -> dict[str, str]:
    """The roles q4nx-build writes at q8, refusing when there is no authority.

    An EMPTY dict is a real answer -- a config exists and says every role is
    q4_1 -- and is returned as such. None from `quant_map_for` is not, and is
    what this turns into a refusal: a quant map that cannot be stated cannot be
    hashed, and a wrong hash links kernels whose weight layout does not match
    the container, which fails as silent corruption rather than an error.
    """
    if find_config(family, cat_family, size) is None:
        raise SpecError(
            f"no q4nx-build config for family {family!r}"
            + (f" (catalogue family {cat_family!r}, size {size!r})" if cat_family or size else "")
            + f"; looked for {config_candidates(family, cat_family, size)} under "
            + f"{configs_dir()}. The container format is an OUTPUT of q4nx-build and "
            "this is its input, so without it the quant map -- which is part of "
            "spec_hash -- cannot be stated. Refusing rather than assuming q4_1, "
            "which would silently link kernels that do not match the container.")
    return quant_map_for(family, cat_family, size) or {}
