"""manifest.json: everything src/open_qwen36 reads about a kernel set.

Written by export_qwen36_kernels.py beside the xclbin / insts pairs. The
driver derives every layout constant, context, kernel, per-layer program and
packing law from it -- no HID, no POOL_*, no lx0 / ax0 names in C++. A model
whose config.json disagrees with `hf_config_check` is refused at startup
(OPEN-MANIFEST). The family recipe (families.py) supplies every block.

    python -m recipes.manifest [--spec FILE | --model-dir DIR] [--out manifest.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cache import build_key
from .families import for_spec
from .load import spec_from_model_dir
from .spec import ModelSpec, hf_model_types

MANIFEST_VERSION = 1


def manifest(spec: ModelSpec, max_ctx: int = 4096, key: str | None = None) -> dict:
    F = for_spec(spec)
    F.recipe(spec, max_ctx)                 # refuses a spec outside the validated points
    prog = F.programs(spec, max_ctx)
    check = {"model_type": hf_model_types(spec.family)}
    check.update(F.hf_config_check(spec))
    m = {
        "manifest_version": MANIFEST_VERSION,
        "family": spec.family,
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash(),
        "build_key": key if key is not None else build_key(spec),
        "max_ctx_default": max_ctx,
        "hf_config_check": check,
        "layout": F.manifest_layout(spec, max_ctx),
        "layers": list(spec.layer_types),
        "contexts": prog["contexts"],
        "kernels": prog["kernels"],
        "layer_types": prog["layer_types"],
        "tail": prog["tail"],
        "globals": prog["globals"],
        "builds": F.builds(spec),
    }
    plan = F.pack_plan(spec)
    for lt, d in plan.pop("layer_types").items():
        m["layer_types"][lt]["pack"] = d
    m["pack"] = plan          # pool_bytes, chunk_bytes, lm_head {pool_bytes, ops}, embed, norm
    return m


def dumps(m: dict) -> str:
    return json.dumps(m, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--spec", help="a ModelSpec JSON (default: the checked-in 27B spec)")
    g.add_argument("--model-dir", help="a model directory with config.json (+ tokenizer.json)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out", help="write here (default: stdout)")
    a = ap.parse_args(argv)
    if a.model_dir:
        spec = spec_from_model_dir(Path(a.model_dir))
    else:
        from .load import default_spec, load_spec
        spec = load_spec(Path(a.spec)) if a.spec else default_spec()
    s = dumps(manifest(spec, a.max_ctx))
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8", newline="\n")
        print(f"wrote {a.out} ({spec.family}, {spec.num_layers} layers, {spec.spec_hash()[:19]})")
    else:
        sys.stdout.write(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
