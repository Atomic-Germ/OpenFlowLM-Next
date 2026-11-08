"""LFM2-MoE gen: layer_x (MoE) + dense (leading dense) TUs.

export_qwen36_kernels.py runs the family GEN (single path into staged tree),
so this wrapper generates both design trees. Layer_x gets the lfm2moe recipe;
dense gets a dense-recipe view of the same spec (dense gen reads R.geo / R.spec only,
so a lightweight shim with the dense geometry suffices -- the dense family guard
is recipe-level, not gen-level).
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # designs/lfm2moe

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def generate(R, out: Path = HERE) -> int:
    root = out.parent  # designs/
    lx = _load("layer_x_gen", root / "layer_x" / "gen_kernels.py")
    n = lx.generate(R, out=root / "layer_x")
    dense = _load("dense_gen", root / "dense" / "gen_kernels.py")
    # Dense TUs for dx/cx at this geometry (the leading dense layers).
    from recipes.load import current_spec  # noqa: E402
    from recipes import dense as D  # noqa: E402
    spec = current_spec()
    geo = D.geometry(spec)
    class _Shim:
        def __init__(self, spec, geo):
            self.spec = spec
            self.geo = geo
    n += dense.generate(_Shim(spec=spec, geo=geo), out=root / "dense")
    return n
