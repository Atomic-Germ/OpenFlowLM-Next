#!/usr/bin/env python3
"""Populate the base-model patch file from repo crawling.

    python utilities/gen-base-patch.py --repo unsloth/Llama-3.2-1B-Instruct-GGUF
    python utilities/gen-base-patch.py --from-specs        # every shipped spec's base

A GGUF cannot carry rope_scaling (Llama-3 bakes it into rope_freqs.weight) or
the real tokenizer id count (its `vocab` is the padded head row count). Those
live in ``open_kernels/recipes/patches.json``, keyed by GENERIC BASE NAME so a
finetune/quant inherits one entry. This script derives the values a repo's base
chain DOES state (config.json + tokenizer.json) and merges them in.

Hand-edited values are kept: a new entry is only added when the key is absent,
unless ``--refresh`` is given (then every fetched field is overwritten). A repo
whose base chain is gated or offline is flagged and left alone -- its patch must
be written by hand, and the script never clobbers a value it could not fetch.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utilities"))
sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
sys.path.insert(0, str(ROOT / "open_kernels"))

# gen-kernel-spec.py is a script, not a package; import it by file path so we
# reuse its exact fetch/base-chain helpers (and the same per-repo download cache).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("gen_kernel_spec", ROOT / "utilities" / "gen-kernel-spec.py")
gks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gks)

from recipes.model_patches import PATCHES_PATH, patch_key  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402


def load():
    try:
        return json.loads(PATCHES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(patches):
    PATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATCHES_PATH.write_text(json.dumps(dict(sorted(patches.items())), indent=2) + "\n",
                            encoding="utf-8")


def spec_bases_from_tree():
    """Generic base names of every shipped spec (its extra.model or link_to)."""
    bases = []
    for p in sorted((ROOT / "open_kernels" / "recipes" / "specs").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        model = d.get("extra", {}).get("model") or d.get("link_to")
        if model:
            bases.append(model)
    return bases


def derive(repo, gguf=""):
    """(key, fields, note) for a repo; fields may be empty on failure."""
    key = patch_key(gks.generic_model_name(repo, gguf))
    if not key or key == "NPU2":
        return None, {}, f"no generic base name for {repo}"
    base, cfg, rv = gks.spec_config_via_base_chain(repo)
    if cfg is None:
        # A repo with its own config.json (an unquantized base upload).
        try:
            cfg, rv = gks.spec_config(repo)
            base = repo
        except Exception as exc:                      # noqa: BLE001
            return key, {}, f"could not fetch config for {repo} or its base chain: {exc}"
    fields = {}
    rs = cfg.get("rope_scaling")
    if rs is not None:
        fields["rope_scaling"] = rs
    if rv is not None:
        fields["real_vocab"] = rv
    note = f"from {base}" + ("" if fields else " (no patchable fields in its config)")
    return key, fields, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", default=[],
                    help="a HF repo to crawl (repeatable)")
    ap.add_argument("--gguf", default="", help="local GGUF, used only for its general.name fallback")
    ap.add_argument("--from-specs", action="store_true",
                    help="crawl the base name of every shipped spec")
    ap.add_argument("--refresh", action="store_true",
                    help="overwrite fields already in the patch file")
    args = ap.parse_args()

    repos = list(args.repo)
    if args.from_specs:
        repos.extend(spec_bases_from_tree())
    if not repos:
        print("nothing to do: pass --repo or --from-specs", file=sys.stderr)
        return 2

    patches = load()
    changed = False
    for repo in repos:
        key, fields, note = derive(repo, args.gguf)
        if not key:
            print(f"skip: {note}")
            continue
        if not fields:
            print(f"skip {key}: {note}")
            continue
        entry = patches.get(key, {})
        added = []
        for name, value in fields.items():
            if args.refresh or entry.get(name) is None:
                entry[name] = value
                added.append(name)
        entry.setdefault("_source", repo)
        if added:
            patches[key] = entry
            changed = True
            print(f"{key}: set {', '.join(added)} ({note})")
        else:
            print(f"{key}: kept hand values ({note})")

    if changed:
        save(patches)
        print(f"wrote {PATCHES_PATH.relative_to(ROOT)}")
    else:
        print("no changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
