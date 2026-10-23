#!/usr/bin/env python3
"""Generate a candidate kernel spec for a HuggingFace model.

Usage:
    python utilities/gen-kernel-spec.py --repo Qwen/Qwen3.5-4B --out open_kernels/recipes/specs/qwen35-4b.json

Fetches config.json (and README frontmatter for base_model), derives dims and
family via the same heuristics as oflm-add/q4nx-build, and emits a spec JSON
that still needs manual routing/heuristic review before building.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS_DIR = ROOT / "open_kernels" / "recipes" / "specs"

# Reuse oflm-add alias tables if available.
sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
try:
    from oflm_add import CONFIG_MODEL_FAMILIES, FAMILY_ALIASES  # type: ignore
except Exception:
    CONFIG_MODEL_FAMILIES = {}
    FAMILY_ALIASES = []


def fetch_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename="config.json", local_dir="/tmp/hf_cfg")
    return json.loads(Path(p).read_text())


def family_from_config(cfg: dict) -> str | None:
    mt = cfg.get("model_type")
    if isinstance(mt, str):
        fam = CONFIG_MODEL_FAMILIES.get(mt.strip().lower())
        if fam:
            return fam
    for arch in (cfg.get("architectures") or []):
        low = arch.lower()
        for prefix, fam in FAMILY_ALIASES:
            if prefix.lower() in low:
                return fam
    return None


def base_model_from_readme(repo: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo, filename="README.md", local_dir="/tmp/hf_cfg")
        text = Path(p).read_text(errors="ignore")
    except Exception:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    fm = text[3:end]
    lines = fm.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("base_model:"):
            rest = lines[i].split(":", 1)[1].strip()
            if rest and not rest.startswith("[") and not rest.startswith("-"):
                val = rest.strip().strip('"').strip("'")
                if val and val.lower() not in ("null", "none", ""):
                    return val
            # list form
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("-"):
                val = lines[j].strip().lstrip("-").strip().strip('"').strip("'")
                if val and val.lower() not in ("null", "none", ""):
                    return val
                j += 1
        i += 1
    return None


def existing_spec_for_family_size(family: str, cfg: dict) -> str | None:
    # check if any spec already covers this hidden_size
    for sp in SPECS_DIR.glob("*.json"):
        try:
            d = json.loads(sp.read_text())
        except Exception:
            continue
        if d.get("family") == family and d.get("hidden") == cfg.get("hidden_size"):
            return sp.name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = fetch_config(args.repo)
    family = family_from_config(cfg)
    if not family:
        print(f"ERROR: could not derive family from config for {args.repo}", file=sys.stderr)
        sys.exit(1)

    bm = base_model_from_readme(args.repo)
    if bm:
        print(f"base_model detected: {bm}")
        try:
            base_cfg = fetch_config(bm)
            family = family_from_config(base_cfg) or family
            cfg = base_cfg
        except Exception as e:
            print(f"warn: could not fetch base config: {e}")

    existing = existing_spec_for_family_size(family, cfg)
    if existing:
        print(f"Existing spec covers this size: {existing} — linking instead of generating.")
        out = Path(args.out) if args.out else Path(f"/tmp/{family}-link.json")
        if out.is_file():
            # never clobber a real spec with a link pointer
            print(f"{out} already exists — keeping it")
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"link_to": existing, "base_model": bm}, indent=2))
        print(f"Wrote link file to {out}")
        return

    # Use an existing spec of the same family as a template for schema/routing.
    template = None
    for sp in SPECS_DIR.glob("*.json"):
        try:
            d = json.loads(sp.read_text())
        except Exception:
            continue
        if d.get("family") == family:
            template = d
            break
    if template:
        spec = json.loads(json.dumps(template))
    else:
        spec = {"family": family}
    spec["family"] = family
    spec["hidden"] = cfg.get("hidden_size", spec.get("hidden"))
    spec["num_layers"] = cfg.get("num_hidden_layers", spec.get("num_layers"))
    if "layer_types" in spec:
        spec["layer_types"] = ["dense"] * spec["num_layers"]
    spec["intermediate"] = cfg.get("intermediate_size", spec.get("intermediate"))
    spec["num_heads"] = cfg.get("num_attention_heads", spec.get("num_heads"))
    spec["num_kv_heads"] = cfg.get("num_key_value_heads", spec.get("num_kv_heads"))
    spec["vocab"] = cfg.get("vocab_size", spec.get("vocab"))
    rope_theta = cfg.get("rope_theta", spec.get("rope_theta"))
    spec["rope_theta"] = float(rope_theta) if rope_theta is not None else rope_theta
    head_dim = cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1)
    spec["head_dim"] = head_dim or spec.get("head_dim")
    spec["rotary_dim"] = head_dim or spec.get("rotary_dim")
    spec["extra"] = {"model_type": cfg.get("model_type"), "source": "hf_config", "model": args.repo.split("/")[-1] + "-NPU2"}
    out = Path(args.out) if args.out else SPECS_DIR / f"{family}-candidate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2))
    print(f"Wrote candidate spec to {out}")


if __name__ == "__main__":
    main()
