#!/usr/bin/env python3
"""Derive EVERYTHING a new family recipe can get from a spec, as data.

    python utilities/derive-recipe.py --spec <name-or-path> --work <dir>
    oflm dev derive-recipe --spec qwen36moe-qwen3-coder-next

A family recipe is mostly a pure function of ModelSpec: geometry, layout,
pack ops, programs, manifest, catalogue points. The hand-written part is the
design directory and the kernel-source pattern when a genuinely new layer
type appears. This command derives the former and names the latter, so a
contributor starts from facts instead of a blank recipe module.

It does NOT write a recipe module (that would pretend the derivation is a
correct kernel). It writes `derivation.json` + `RECIPE.md` with:

  * hardware invariants (universal across every family here)
  * quant-derived constants (from spec.quant + the pool law)
  * capability-derived constants (experts / heads / conv from the spec)
  * the recipe template that fits (dense / moe-router / hybrid-conv / variant)
  * the design dir + kernel-source pattern to supply for a NEW layer type
  * the catalogue points build-new-spec would need, from its own preflight

Use with `oflm dev build-new-spec --spec <name> --dry-run` to get the live
missing-point list; this command records it under the work dir.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUILD_SPEC = ROOT / "utilities" / "build-new-spec.py"


def hardware_invariants() -> dict[str, Any]:
    """Universal NPU facts, identical in every implemented family."""
    return {
        "N_CORES": 8,
        "BAND_ROWS": 64,
        "ELEM": 4096,
        "STACK": "0x1800",
        "LM_BAND_ROWS": 128,
        "LM_TILE": 9216,
        "POOL_GRANULARITY_MB": 1,
    }


def quant_law(quant: str) -> dict[str, Any]:
    """Pool chunk law from the spec's quant (pools.cpp / gguf_pool.py)."""
    return {
        "q4_1": {"CH": 5120, "PER_CALL": 2, "CHUNK_VALUES": 8192, "CHUNK_ROWS": 32,
                 "CHUNK_COLS": 256, "note": "bf16 d|m planes, codes kept exactly"},
        "q4_1_f32": {"CH": 5120, "PER_CALL": 2, "CHUNK_VALUES": 8192, "CHUNK_ROWS": 32,
                     "CHUNK_COLS": 256, "note": "f32-scale GGUF-direct twin"},
        "q4_0": {"CH": 4608, "PER_CALL": 2, "CHUNK_VALUES": 8192, "CHUNK_ROWS": 32,
                 "CHUNK_COLS": 256, "note": "bf16 d plane then codes"},
        "q8_0": {"CH": 5120, "PER_CALL": 2, "note": "q8 half-tile 16x256"},
    }.get(quant, {"CH": 5120, "PER_CALL": 2,
                  "note": f"unknown quant {quant!r}; using the q4_1 law as a placeholder"})


def template_for(spec: dict[str, Any]) -> dict[str, str]:
    """Choose a recipe template by the spec's layer kinds."""
    moe = int(spec.get("num_experts") or 0) > 0
    conv = int(spec.get("conv_kernel") or 0) > 0
    gate = bool(spec.get("attn_gate"))
    types = set(spec.get("layer_types") or [])
    if moe and conv:
        return {"template": "hybrid-conv-moe", "closest": "qwen35moe/lfm2moe",
                "note": "two dispatches + conv; derive slot law from topk"}
    if moe:
        return {"template": "moe-router", "closest": "qwen36moe/flint/hyv3",
                "note": "host router + expert slots; TOPK/SLOTS from experts_per_tok"}
    if conv:
        return {"template": "hybrid-conv", "closest": "lfm2/lfm3",
                "note": "short-conv layers + attention; TAPS from conv_kernel"}
    if len(types) > 1:
        return {"template": "multi-kind-dense", "closest": "gemma4",
                "note": "per-kind sub-specs; one xclbin per layer kind"}
    return {"template": "dense", "closest": "qwen3/llama3/dense",
            "note": "one dispatch per layer"}


def capability_constants(spec: dict[str, Any]) -> dict[str, Any]:
    """Constants derivable from head/expert counts (the recipe's own arithmetic)."""
    out: dict[str, Any] = {}
    heads = int(spec.get("num_heads") or 0)
    kv = int(spec.get("num_kv_heads") or heads)
    if heads:
        # attnknobs.attn_cores(nh): largest divisor of nh <= 6 that the design can use.
        out["ATTN_CORES"] = max((c for c in (6, 4, 2, 1) if heads % c == 0), default=1)
        out["NHL"] = heads // out["ATTN_CORES"]
        out["QW"] = heads * int(spec.get("head_dim") or 0)
        out["KVW"] = kv * int(spec.get("head_dim") or 0)
    experts = int(spec.get("num_experts") or 0)
    topk = int(spec.get("experts_per_tok") or 0)
    if experts:
        out["TOPK_MAX"] = topk
        out["EXPERTS"] = experts
        # The slot table is built for topk + the shared-expert halves.
        shared = int(spec.get("shared_expert_intermediate") or 0)
        moe_ff = int(spec.get("moe_intermediate") or 0)
        out["SHARED_SLOTS"] = (shared // moe_ff) if moe_ff else 0
        out["SLOTS"] = topk + out["SHARED_SLOTS"]
        out["POOL_EXPERTS"] = experts + out["SHARED_SLOTS"]
    return out


def design_inputs(spec: dict[str, Any], template: dict[str, str]) -> dict[str, str]:
    """What a human must supply because it is NOT derivable from the spec."""
    return {
        "design": (f"designs/<new-family>/ ... or reuse {template['closest'].split('/')[0]}"),
        "kernel_source_pattern": (template["closest"].split("/")[0] +
                                  " (gen_kernels.py + *.cc/*.h; parameterize by the derived Geometry)"),
        "spec_target_repo": "an HF GGUF repo for numeric gates (--ref-model in build-new-family)",
        "note": ("A new layer kind needs a design; a new constant set on an existing kind "
                 "needs only the generated recipe + the existing design's gen_kernels.py."),
    }


def preflight(work: Path, spec_name: str, py: str, env: dict[str, str]) -> dict[str, Any]:
    """Record build-new-spec's own missing-point list (the catalogue gates)."""
    rc, out = 0, ""
    try:
        # Pass the FULL PATH: a candidate spec lives in recipes/candidates/,
        # and build-new-spec resolves a bare stem against recipes/specs/ only.
        p = subprocess.run([py, str(BUILD_SPEC), "--spec", spec_name, "--dry-run"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           cwd=str(ROOT), env=env)
        rc, out = p.returncode, p.stdout or ""
    except Exception as exc:  # noqa: BLE001
        out = str(exc)
    (work / "preflight.log").write_text(out, encoding="utf-8")
    points, walls = [], []
    for line in out.splitlines():
        s = line.strip()
        if s.endswith(":") and s.split(":", 1)[0] in (
                "router", "moe", "gemv_q4", "ln", "attn", "lm_head_q4", "lm_head_q8"):
            points.append(s)
        elif "the RECIPE does not support this shape" in s:
            walls.append(s)
        elif s.startswith(("router:", "moe:", "gemv_q4:", "ln:", "attn:", "lm_head_q4:")):
            points.append(s)
    return {"rc": rc, "missing_points": points, "recipe_walls": walls}


def write_report(work: Path, spec: dict[str, Any], report: dict[str, Any]) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    (work / "derivation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# Derived recipe: {spec.get('family')} / {spec.get('extra', {}).get('model', '?')}",
             "",
             "Everything below is computed from the ModelSpec plus universal NPU facts.",
             "It is a starting point for a recipe module, not a claim that the kernels work.",
             "",
             "## Template", "", f"- `{report['template']['template']}` "
             f"(closest implemented family: {report['template']['closest']})",
             f"- {report['template']['note']}", "",
             "## Hardware invariants", ""]
    for k, v in report["hardware"].items():
        lines.append(f"- `{k}` = `{v}`")
    lines += ["", "## Quant law", ""]
    for k, v in report["quant"].items():
        lines.append(f"- `{k}` = `{v}`")
    lines += ["", "## Capability constants", ""]
    for k, v in report["capabilities"].items():
        lines.append(f"- `{k}` = `{v}`")
    lines += ["", "## Inputs to supply", ""]
    for k, v in report["inputs"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Preflight (build-new-spec --dry-run)", ""]
    walls = report["preflight"].get("recipe_walls") or []
    if walls:
        lines.append("**The recipe cannot build this shape** — this is the port boundary:")
        lines.append("")
        for w in walls:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("### Catalogue points")
    lines.append("")
    for p in report["preflight"]["missing_points"]:
        lines.append(f"- {p}")
    if not report["preflight"]["missing_points"] and not walls:
        lines.append("- (none reported; see preflight.log)")
    lines += ["", "## Verify", "", "```sh",
              f"oflm dev build-new-spec --spec {work.name}",
              "```", ""]
    path = work / "RECIPE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="a checked-in spec name or path")
    ap.add_argument("--work", default="", help="artifact dir (default: build/derive-recipe/<spec>)")
    ap.add_argument("--python", default="", help="python for build-new-spec (default: sys.executable)")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes.load import load_spec                      # noqa: PLC0415
    sp = Path(a.spec)
    if not sp.is_file():
        stem = a.spec.removesuffix(".json")
        # A walled spec is parked in candidates/, not specs/ -- accept a bare
        # stem from either, with specs/ taking precedence (an active spec wins).
        for base in (ROOT / "open_kernels" / "recipes" / "specs",
                     ROOT / "open_kernels" / "recipes" / "candidates"):
            cand = base / (stem + ".json")
            if cand.is_file():
                sp = cand
                break
    if not sp.is_file():
        raise SystemExit(f"derive-recipe: no such spec: {a.spec}")
    spec = load_spec(sp).to_dict()
    work = Path(a.work) if a.work else ROOT / "build" / "derive-recipe" / sp.stem
    work.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "spec": a.spec, "family": spec.get("family"),
        "hardware": hardware_invariants(),
        "quant": quant_law(str(spec.get("quant") or "")),
        "template": template_for(spec),
        "capabilities": capability_constants(spec),
        "inputs": design_inputs(spec, template_for(spec)),
    }
    report["preflight"] = preflight(work, str(sp), a.python or sys.executable, dict(__import__("os").environ))
    path = write_report(work, spec, report)
    walls = report["preflight"].get("recipe_walls") or []
    wall_note = f"; RECIPE WALL: {walls[0].split(': ', 1)[-1]}" if walls else ""
    print(f"derive-recipe: {report['template']['template']} template, "
          f"{len(report['preflight']['missing_points'])} catalogue point(s){wall_note}")
    print(f"  report: {path}")
    print(f"  data:   {work / 'derivation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
