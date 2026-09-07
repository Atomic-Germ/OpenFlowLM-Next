"""Which ModelSpec a design build / packer run uses.

    OPEN_KERNELS_SPEC=<file.json>   an explicit spec (export_qwen36_kernels.py sets it)
    otherwise                        recipes/specs/qwen36-35b-a3b.json, the checked-in 27B

`spec_from_model_dir` derives one from a model directory's config.json, with
the real vocab from tokenizer.json when it is there -- and the per-role weight
format from `model.q4nx`'s safetensors header (OPEN-QUANT-Q8), which is the
only thing that says whether a projection is stored at q8. The header alone is
read; no weight byte is touched.

    OPEN_KERNELS_FORCE_Q4_1=1        every role back to q4_1 (the re-quantising
                                     fallback, for an A/B against the q8 path)

A derived map that MIXES formats is narrowed to what the hardware has been shown
to build: `narrow_to_buildable` below, against `catalogue.MIXED_CORE_FITS`.
"""
from __future__ import annotations

import dataclasses
import json
import os
import struct
import sys
from pathlib import Path

from .spec import ModelSpec, quant_map_from_chunk_sizes

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "specs" / "qwen36-35b-a3b.json"


def load_spec(path: Path) -> ModelSpec:
    return ModelSpec.from_json(Path(path).read_text(encoding="utf-8"))


def default_spec() -> ModelSpec:
    return load_spec(DEFAULT_SPEC)


def current_spec() -> ModelSpec:
    p = os.environ.get("OPEN_KERNELS_SPEC")
    return load_spec(Path(p)) if p else default_spec()


def tokenizer_vocab(tokenizer_json: Path) -> int | None:
    """The tokenizer's id count: max id over model.vocab and added_tokens, + 1."""
    try:
        t = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ids = list(t.get("model", {}).get("vocab", {}).values()) + [a["id"] for a in t.get("added_tokens", [])]
    return max(ids) + 1 if ids else None


def container_chunk_bytes(model_dir: Path) -> dict[str, int]:
    """Every quantized tensor's chunk size, from `model.q4nx`'s safetensors header only
    (an 8-byte length then the JSON). {} when there is no container to look at."""
    p = Path(model_dir) / "model.q4nx"
    if not p.is_file():
        return {}
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: int(v["shape"][-1]) for k, v in hdr.items()
            if k != "__metadata__" and isinstance(v, dict) and v.get("dtype") == "I8" and v.get("shape")}


def narrow_to_buildable(spec: ModelSpec, m: dict[str, str], can: frozenset) -> dict[str, str]:
    """A MIXED map -- some role at q8 while another the same core runs stays q4_1 -- makes
    the main core carry both weight formats' GEMV bodies, and that core has 16 KB of program
    memory. `catalogue.MIXED_CORE_FITS` holds the (family, hidden) widths where that has been
    built; at any other width `linear_out` goes back to q4_1 and the packer re-quantizes the
    container's q8 out projection, exactly as it did before OPEN-QUANT-Q8. Encoding the
    hardware fact here is the difference between a slightly worse model and an export that
    dies 60 s into aiecc with `Overflow of program memory`."""
    from .catalogue import mixed_core_fits
    if m.get("linear_out") != "q8":
        return m
    if not (set(can) - set(m)):
        return m                        # every role is q8: one format on the core, no fold
    if mixed_core_fits(spec.family, spec.hidden):
        return m
    print(f"open_kernels: {spec.family} hidden {spec.hidden}: linear_out narrowed to q4_1 -- "
          f"no mixed-format main core has been built at this width (16 KB of program memory; "
          f"see .claude/plans/q8m-hw-results.md), so the packer re-quantizes this container's "
          f"q8 out projection at a measured logits corr of 0.999682 against its own q8 values.",
          file=sys.stderr)
    return {r: f for r, f in m.items() if r != "linear_out"}


def quant_for(spec: ModelSpec, model_dir: Path):
    """The spec's weight format for this container: the container's own per-role map, with
    every role the family's designs cannot stream at q8 put back to q4_1 (the packer then
    re-quantizes those, as it always did), then narrowed to a mixed core the hardware can
    actually build. OPEN_KERNELS_FORCE_Q4_1 forces the fallback."""
    if os.environ.get("OPEN_KERNELS_FORCE_Q4_1"):
        return "q4_1"
    sizes = container_chunk_bytes(model_dir)
    if not sizes:
        return spec.quant
    from .families import for_spec
    can = getattr(for_spec(spec), "Q8_ROLES", frozenset())
    m = {r: f for r, f in quant_map_from_chunk_sizes(spec.family, sizes).items() if r in can}
    m = narrow_to_buildable(spec, m, can)
    return m or "q4_1"


def spec_from_model_dir(model_dir: Path) -> ModelSpec:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    rv = tokenizer_vocab(model_dir / "tokenizer.json")
    spec = ModelSpec.from_hf_config(cfg, real_vocab=rv)
    spec = dataclasses.replace(spec, quant=quant_for(spec, Path(model_dir)))
    spec.extra["model"] = Path(model_dir).name
    return spec


def current_recipe(max_ctx: int = 4096):
    """The family recipe for OPEN_KERNELS_SPEC. The whole-layer designs read `R.kind`
    off it to pick their tail: "moe" (qwen36moe) or "dense" (qwen35)."""
    from .families import for_spec
    spec = current_spec()
    return for_spec(spec).recipe(spec, max_ctx)
