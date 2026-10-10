"""Write topology.json per kernel dir and viz.json beside a kernel set's manifest.json."""
from __future__ import annotations

import json
from pathlib import Path

from . import explain
from .timeline import decode_timeline
from .topology import extract_prj

VIZ_VERSION = 1
DESIGNS = Path(__file__).resolve().parents[1] / "designs"


class VizError(ValueError):
    pass


def model_summary(m: dict) -> dict:
    s = m.get("spec") or {}
    moe = (m.get("layout") or {}).get("moe") or {}
    keys = ("hidden", "num_layers", "vocab", "num_heads", "num_kv_heads", "head_dim", "quant",
            "num_experts", "experts_per_tok", "lin_key_heads", "lin_value_heads")
    out = {k: s[k] for k in keys if k in s}
    out.update({"name": (s.get("extra") or {}).get("model") or m.get("family"), "family": m.get("family"),
                "experts": moe.get("experts"), "topk": moe.get("topk")})
    return out


def build_viz(set_dir: Path, designs: Path = DESIGNS, build_dirs: dict | None = None) -> tuple[dict, dict]:
    """(viz.json content, {set dir name: topology}) for the set; refuses a build that is not the set's."""
    set_dir = Path(set_dir)
    m = json.loads((set_dir / "manifest.json").read_text(encoding="utf-8"))
    topo_of_dir: dict[str, dict] = {}
    for name, b in m.get("builds", {}).items():
        bdir = Path((build_dirs or {}).get(name) or Path(designs) / b["build_dir"])
        mine, built = set_dir / name / "insts.bin", bdir / "insts.bin"
        if not built.is_file() or not mine.is_file() or mine.read_bytes() != built.read_bytes():
            raise VizError(f"{name}: {bdir} is not the build of this set's {name}/insts.bin")
        try:
            topo_of_dir[name] = extract_prj(bdir / "final.prj")
        except ValueError as e:
            raise VizError(f"{name}: {e}") from e
    topos = {k: topo_of_dir[Path(kd["insts"]).parent.as_posix()] for k, kd in m["kernels"].items()
             if Path(kd["insts"]).parent.as_posix() in topo_of_dir}
    return assemble(m, topos), topo_of_dir


def assemble(m: dict, topos: dict[str, dict]) -> dict:
    """viz.json from a manifest and each kernel's topology."""
    kernels = {}
    for k, kd in m["kernels"].items():
        d = Path(kd["insts"]).parent.as_posix()
        b = m.get("builds", {}).get(d, {})
        kernels[k] = {"context": kd["context"], "dir": d, "design": b.get("design"), "patch": kd.get("patch")}
    for lt in m["layer_types"].values():
        for s in lt["program"]:
            if s["op"] == "run":
                kernels[s["kernel"]].setdefault("args", s.get("args"))
    for s in m.get("tail", []):
        if s["op"] == "run":
            kernels[s["kernel"]].setdefault("args", s.get("args"))
    return {
        "viz_version": VIZ_VERSION,
        "model": model_summary(m),
        "set": {"build_key": m.get("build_key"), "spec_hash": m.get("spec_hash"), "family": m.get("family")},
        "kernels": kernels,
        "layers": m["layers"],
        "layer_types": {lt: {"program": v["program"],
                             "state": ((v.get("buffers") or {}).get("state") or {}).get("kind")}
                        for lt, v in m["layer_types"].items()},
        "tail": m.get("tail", []),
        "decode": decode_timeline(m, topos),
    }


def write_viz(set_dir: Path, designs: Path = DESIGNS, build_dirs: dict | None = None) -> Path:
    """Write the set's viz files; on a refusal remove any stale viz.json so no old picture survives."""
    set_dir = Path(set_dir)
    out = set_dir / "viz.json"
    try:
        viz, topos = build_viz(set_dir, designs, build_dirs)
    except Exception:
        out.unlink(missing_ok=True)
        raise
    for name, t in topos.items():
        (set_dir / name / "topology.json").write_text(json.dumps(t, separators=(",", ":")), encoding="utf-8")
    out.write_text(json.dumps(viz, separators=(",", ":")), encoding="utf-8")
    if gap := explain.missing(viz, explain.load()):
        print(f"[viz] no explainer yet in src/viz/explain for: {', '.join(gap)}")
    return out
