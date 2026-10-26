#!/usr/bin/env python3
"""What models have no kernel spec here, and what would adding one take?

    python utilities/spec-gaps.py                          # vs the official registry
    python utilities/spec-gaps.py --reference ../OpenFlowLM-Next   # vs another checkout
    python utilities/spec-gaps.py --reference ../OpenFlowLM-Next --missing-only

Two questions, and they are different:

  * the REGISTRY (src/model_list.json) says which models OpenFlowLM ships. A
    family/size there with no spec in open_kernels/recipes/specs has no open
    kernels: `oflm add` has nothing to link. This mode matches by the spec's
    `extra.model` against the registry's directory name.
  * another CHECKOUT says which sizes someone already worked out. This mode
    diffs the spec stems and, for each one only there, classifies it: does the
    recipe build it, which catalogue points does it need, and does this tree
    have a fixture that can validate each of those (utilities/op-gate.py)?

Exit status is 0 when there is nothing missing, 1 when there is, so it can gate
a release. Nothing is generated: adding a size is still one command
(utilities/build-new-spec.py --repo <hf-repo>), and this only says which.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "open_kernels"
SPECS = OPEN / "recipes" / "specs"
CANDIDATES = OPEN / "recipes" / "candidates"
REGISTRY = ROOT / "src" / "model_list.json"


def _load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _specs(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


def _model_of(p: Path) -> str:
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("extra", {}).get("model", "")
    except (OSError, ValueError):
        return ""


def _size_token(name: str) -> str:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name)
    return m.group(1).lower() if m else ""


def registry_gaps() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]],
                             list[tuple[str, str, str, str]]]:
    """([(family, size, model)] with no spec, [(family, size, model)] for other engines).

    Only the OPEN-KERNEL families can have a spec: a VL model, an embedding
    tower or a whisper set is served by its own engine (open_gemma3's vision,
    open_embedding, open_whisper), and listing it here as a gap would be noise
    that never goes away. The registry does not say which engine a model uses,
    so the family slug does: `details.family` is what `recipes/families.py`
    dispatches on, and a family it does not know is not this tree's business.
    """
    sys.path.insert(0, str(OPEN))
    from recipes.families import FAMILIES

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    have = {_model_of(p) for p in SPECS.glob("*.json")}
    have |= {_model_of(p) for p in CANDIDATES.glob("*.json")}
    from recipes.load import load_spec
    shipped = {}
    for p in sorted(SPECS.glob("*.json")):
        try:
            sp = load_spec(p)
        except Exception:                     # noqa: BLE001 - a report must not die on one file
            continue
        shipped.setdefault(sp.family, []).append((sp.extra.get("model", p.stem), _size_token(
            sp.extra.get("model", p.stem))))

    gaps, other, linked = [], [], []
    for fam, sizes in sorted(reg.get("models", {}).items()):
        if not isinstance(sizes, dict):
            continue
        for size, entry in sorted(sizes.items()):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "")
            det = entry.get("details", {})
            if not name or name in have or det.get("weights") == "gguf":
                continue          # present, or a GGUF install that links an existing set
            if entry.get("vlm"):
                other.append((fam, size, name))   # the vision tower is another engine's
                continue
            if det.get("family") not in FAMILIES:
                other.append((fam, size, name))
                continue
            # Same family and same parameter count as a shipped spec: a fine-tune or a
            # sibling that `oflm add` links to that set by kernel_hash -- it needs no
            # spec of its own unless its GEOMETRY differs (build-new-spec --dry-run says).
            tok = _size_token(name) or _size_token(size)
            same = [m for m, t in shipped.get(det.get("family"), []) if tok and t == tok]
            if same:
                linked.append((fam, size, name, same[0]))
            else:
                gaps.append((fam, size, name))
    return gaps, other, linked


def _kernel_hash(p: Path) -> str:
    try:
        sys.path.insert(0, str(OPEN))
        from recipes.load import load_spec
        return load_spec(p).kernel_hash()
    except Exception:                         # noqa: BLE001 - a report must not die on one file
        return ""


def _kernel_hashes(paths) -> dict[str, str]:
    out = {}
    for p in paths:
        h = _kernel_hash(p)
        if h:
            out.setdefault(h, p.stem)
    return out


def classify(spec_path: Path) -> str:
    """One line: builds here, needs these points, or hits this wall."""
    sys.path.insert(0, str(OPEN))
    from recipes import catalogue as C
    from recipes.families import for_spec
    from recipes.load import load_spec
    from recipes.catalogue import OpRangeError

    try:
        spec = load_spec(spec_path)
    except Exception as exc:                                  # noqa: BLE001 - report, do not raise
        return f"cannot load the spec: {type(exc).__name__}: {exc}"
    gates = _load_module("_op_gate", ROOT / "utilities" / "op-gate.py")
    with C.collecting() as missing:
        try:
            for_spec(spec).recipe(spec)
        except OpRangeError as exc:
            return f"RECIPE WALL: {str(exc).splitlines()[0]}"
    if not missing:
        return "builds here: every catalogue point is validated"
    auto, manual, delegated = [], [], []
    for op, kw in missing:
        if op in gates.GATE_FN:
            auto.append(op)
        elif op in gates.NOT_AUTOMATED and op == "attn":
            delegated.append(op)
        else:
            manual.append(f"{op} {kw}")
    bits = []
    if delegated:
        bits.append("attention gate")
    if auto:
        bits.append("gated: " + ", ".join(sorted(set(auto))))
    if manual:
        bits.append("NO FIXTURE (design step): " + "; ".join(manual))
    return "needs " + " + ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="", help="another checkout's root (or its recipes/specs dir)")
    ap.add_argument("--missing-only", action="store_true", help="hide what is already covered")
    a = ap.parse_args()

    ours = _specs(SPECS)
    problems = 0

    print(f"this tree: {len(ours)} spec(s) in open_kernels/recipes/specs"
          + (f", {len(_specs(CANDIDATES))} candidate(s)" if CANDIDATES.is_dir() else ""))
    gaps, other, linked = registry_gaps()
    if gaps:
        problems += len(gaps)
        print("\nREGISTRY: %d open-kernel model(s) with no spec (oflm add has no kernels to link)"
              % len(gaps))
        for fam, size, name in gaps:
            print(f"  {fam:16} {size:8} {name}")
    else:
        print("\nREGISTRY: every open-kernel model has a spec (or is a GGUF install)")
    if linked:
        print(f"\n  {len(linked)} model(s) with no spec of their own, served by a shipped set "
              f"(same family and size -- `oflm add` links by kernel_hash):")
        for fam, size, name, spec in linked:
            print(f"  link     {fam:16} {size:6} {name}  ->  {spec}")
    if other:
        print(f"  ({len(other)} other-engine model(s) skipped: VLM / embedding / whisper / "
              f"diffusion, not open-kernel families)")

    if a.reference:
        ref_dir = Path(a.reference)
        if (ref_dir / "open_kernels" / "recipes" / "specs").is_dir():
            ref_dir = ref_dir / "open_kernels" / "recipes" / "specs"
        ref = _specs(ref_dir)
        if not ref:
            print(f"\nREFERENCE: no specs found under {ref_dir}", file=sys.stderr)
            return 1
        only_there = sorted(set(ref) - set(ours))
        only_here = sorted(set(ours) - set(ref))
        print(f"\nREFERENCE {ref_dir}: {len(ref)} spec(s); {len(only_there)} only there, "
              f"{len(only_here)} only here")
        have_models = {_model_of(p) for p in ours.values()}
        have_hashes = _kernel_hashes(ours.values())
        for stem in only_there:
            model = _model_of(ref[stem])
            if model and model in have_models:
                # gen-kernel-spec.py's default output name (--out omitted): the same
                # model under a second file name, not a spec anyone needs.
                print(f"  dupe     {stem:22} same model as a shipped spec ({model})")
                continue
            if stem.endswith("-candidate"):
                # gen-kernel-spec.py's DEFAULT output name (--out omitted): a working
                # file someone left behind, not a spec. Fedora's tree has two.
                print(f"  candidate {stem:21} generator default name, not a shipped spec")
                continue
            h = _kernel_hash(ref[stem])
            if h and h in have_hashes:
                # A second NAME for one geometry: the same kernels, so no spec is
                # missing -- `oflm add` links this model to that set by kernel_hash.
                print(f"  dupe     {stem:22} same kernel geometry as {have_hashes[h]}")
                continue
            print(f"  MISSING  {stem:22} {classify(ref[stem])}")
            problems += 1
        for stem in only_here:
            print(f"  extra    {stem:22} (this tree has it, the reference does not)")

    if not a.missing_only:
        print("\ncovered (a spec claims the model):")
        for stem in sorted(ours):
            print(f"  ok       {stem}")

    print("\nadd one:  python utilities/build-new-spec.py --repo <hf-repo>   "
          "(then utilities/op-gate.py names any point with no fixture)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
