#!/usr/bin/env python3
"""Register NEW NAMES for an EXISTING family recipe (the mechanical half of a port).

    python utilities/register-family-alias.py --family qwen35 \
        --model-type deepseek_v32 --arch deepseek2 --name-prefix deepseek-v3

A shape-identical model (Bonsai -> qwen35) usually needs NO new recipe: its
config.json / GGUF header just name an architecture the registry does not know.
`build-new-family` discovers the geometry and picks the recipe; this command
writes the registry edits that must change in lockstep so the result works in
spec derivation AND install:

  * utilities/oflm-add/oflm_add/__init__.py
      CONFIG_MODEL_FAMILIES, GGUF_ARCH_FAMILIES, FAMILY_ALIASES
  * open_kernels/recipes/spec.py
      HF_FAMILIES, GGUF_FAMILIES, _FAMILY_OF  (aliased to the family's
      existing builder, so NO new spec-builder math is invented)

The edit is idempotent, reversible (--remove), and verified in-process: after
writing, the named keys resolve to the named family through the real lookup
functions. It does NOT add recipe logic, kernel designs, or catalogue points --
that is the research half in .agents/skills/open-family-addition/SKILL.md.

Never use this to force a model onto a recipe whose arithmetic does not fit:
the geometry wall in build-new-spec is the port boundary, not a naming gap.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADD = ROOT / "utilities" / "oflm-add" / "oflm_add" / "__init__.py"
SPEC = ROOT / "open_kernels" / "recipes" / "spec.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str, dry: bool) -> None:
    if not dry:
        path.write_text(text, encoding="utf-8")


def _dict_entry(text: str, table: str, key: str, value: str | None,
                remove: bool) -> str:
    """Add/remove a marked line inside a top-level dict literal.

    Added lines carry a `# oflm dev register-alias` marker, so --remove can
    delete exactly what this tool added without touching a pre-entry that
    happens to share a key or a neighbouring line.
    """
    m = re.search(rf"^{table}\s*=\s*\{{", text, re.M)
    if not m:
        raise SystemExit(f"register-family-alias: no table {table} in {ADD}")
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[start:i]
    marker = re.compile(
        rf'^\s*"{re.escape(key)}"\s*:\s*"[^"]+",\s*# oflm dev register-alias\s*\n', re.M)
    if remove:
        new_body, n = marker.subn("", body)
        if n == 0:
            return text                  # nothing this tool added here
        # Restore the original last entry: the marker line may have made the
        # previous entry end in ',' where it originally ended with '}'.
        new_body = new_body.rstrip()
        if new_body.endswith(","):
            new_body = new_body[:-1]
        return text[:start] + new_body + text[i:]
    if marker.search(body) or re.search(rf'"{re.escape(key)}"\s*:', body):
        return text                      # already present (idempotent)
    line = f'    "{key}": "{value}",  # oflm dev register-alias\n'
    if "\n" not in body:
        # Inline table: insert before the closing brace with a comma separator.
        new_body = body.rstrip() + f' "{key}": "{value}",  # oflm dev register-alias '
    else:
        stripped = body.rstrip()
        if not stripped.endswith(","):
            stripped += ","
        new_body = stripped + "\n" + line
    return text[:start] + new_body + text[i:]


def _alias_entry(text: str, prefix: str, family: str, remove: bool) -> str:
    """Add/remove a `("prefix", "family"),` line in FAMILY_ALIASES."""
    m = re.search(r"^FAMILY_ALIASES\s*=\s*\[", text, re.M)
    if not m:
        raise SystemExit("register-family-alias: no FAMILY_ALIASES table")
    start = m.end()
    i = text.index("]", start)
    body = text[start:i]
    pattern = re.compile(rf'^\s*\("{re.escape(prefix)}",\s*"[^"]+"\),\s*$', re.M)
    line = f'    ("{prefix}", "{family}"),\n'
    if remove:
        new_body = pattern.sub("", body)
    elif pattern.search(body):
        return text
    else:
        new_body = body + line
    return text[:start] + new_body + text[i:]


def _spec_alias(text: str, table: str, key: str, builder: str, remove: bool) -> str:
    return _dict_entry(text, table, key, builder, remove)


def builder_for(family: str) -> str | None:
    """The existing HF builder function registered for this family, if any.

    HF_FAMILIES is authoritative (it is what gen-kernel-spec uses); _FAMILY_OF
    is only a reverse index that may be incomplete in a WIP tree.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import spec as S                     # noqa: PLC0415
    for _key, fn in S.HF_FAMILIES.items():
        if getattr(fn, "__name__", "").startswith("_") and fn.__name__.endswith("_hf"):
            if S._FAMILY_OF.get(fn) == family:
                return fn.__name__
    # Fallback: a builder named after the family that is not yet reverse-indexed.
    candidate = f"_{family}_hf"
    if hasattr(S, candidate):
        return candidate
    return None


def gguf_builder_for(family: str, hf_builder: str) -> str:
    """The existing GGUF builder for this family (GGUF tables need their own)."""
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import spec as S                     # noqa: PLC0415
    for fn, fam in S._FAMILY_OF.items():
        if fam == family and fn.__name__.endswith("_gguf"):
            return fn.__name__
    candidate = f"_{family}_gguf"
    if hasattr(S, candidate):
        return candidate
    # A family whose GGUF and HF builders are the same function (plain dense
    # families often share one) -- the HF builder is then correct for both.
    return hf_builder


def register(family: str, *, model_types=(), arches=(), prefixes=(),
             dry: bool = False, remove: bool = False) -> dict:
    # Validate the family really has a recipe before touching anything.
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import families as F                  # noqa: PLC0415
    if family not in F.FAMILIES and family not in F.NOT_IMPLEMENTED:
        raise SystemExit(f"register-family-alias: {family!r} is not a known family "
                         f"(have {', '.join(F.FAMILIES)})")
    builder = builder_for(family)
    if not builder:
        raise SystemExit(f"register-family-alias: family {family!r} has no HF spec builder; "
                         "this tool only aliases names to an EXISTING builder -- the spec "
                         "builder itself is research (see open-family-addition)")
    gguf_builder = gguf_builder_for(family, builder)

    add = _read(ADD)
    for mt in model_types:
        add = _dict_entry(add, "CONFIG_MODEL_FAMILIES", mt, family, remove)
    for arch in arches:
        add = _dict_entry(add, "GGUF_ARCH_FAMILIES", arch, family, remove)
    for p in prefixes:
        add = _alias_entry(add, p, family, remove)

    spec = _read(SPEC)
    for mt in model_types:
        spec = _spec_alias(spec, "HF_FAMILIES", mt, builder, remove)
    for arch in arches:
        spec = _spec_alias(spec, "GGUF_FAMILIES", arch, gguf_builder, remove)
    # _FAMILY_OF already maps this family's EXISTING builders to the family
    # (that is how builder_for() found them). Aliasing new names never needs a
    # new mapping, so leave the reverse index untouched -- adding a duplicate
    # would make --remove delete both the copy and the original entry.

    _write(ADD, add, dry)
    _write(SPEC, spec, dry)
    return {"family": family, "builder": builder, "gguf_builder": gguf_builder,
            "model_types": list(model_types),
            "arches": list(arches), "prefixes": list(prefixes),
            "files": [str(ADD.relative_to(ROOT)), str(SPEC.relative_to(ROOT))]}


def _verify(family: str, model_types, arches) -> None:
    """After a real edit, the public lookups must resolve the new keys."""
    sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_oflm_add_verify", ADD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for mt in model_types:
        got = m.family_from_config_dict({"model_type": mt})
        if got != family:
            raise SystemExit(f"verify: model_type {mt!r} resolves to {got!r}, expected {family!r}")
    # GGUF arch lookup needs a file; check the table directly.
    for arch in arches:
        if m.GGUF_ARCH_FAMILIES.get(arch) != family:
            raise SystemExit(f"verify: arch {arch!r} not registered for {family!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="the EXISTING recipe family to alias to")
    ap.add_argument("--model-type", action="append", default=[],
                    help="config.json model_type naming the same layer (repeatable)")
    ap.add_argument("--arch", action="append", default=[],
                    help="GGUF general.architecture naming the same layer (repeatable)")
    ap.add_argument("--name-prefix", action="append", default=[],
                    help="repo-name prefix that should route to the family (repeatable)")
    ap.add_argument("--remove", action="store_true", help="remove the aliases instead")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()

    result = register(a.family, model_types=a.model_type, arches=a.arch,
                      prefixes=a.name_prefix, dry=a.dry_run, remove=a.remove)
    verb = "would remove" if a.remove else ("would register" if a.dry_run else "registered")
    print(f"register-family-alias: {verb} aliases for family {a.family} "
          f"(builder {result['builder']})")
    for key in ("model_types", "arches", "prefixes"):
        if result[key]:
            print(f"  {key}: {', '.join(result[key])}")
    for f in result["files"]:
        print(f"  {f}")
    if a.dry_run or a.remove:
        return 0
    _verify(a.family, a.model_type, a.arch)
    print("verified: the new keys resolve to the family through the real lookups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
