#!/usr/bin/env python3
"""Make a repo installable by geometry, doing the least work that suffices.

    python utilities/add-family.py --repo unsloth/granite-4.0-h-small-GGUF
    python utilities/add-family.py --repo <user>/<repo> --apply
    oflm dev add-family --repo <user>/<repo>

Everything is user-installed: there is no privileged "shipped" model set, and a
model is its GEOMETRY, not its repo name. `oflm add` matches an installed kernel
set by `kernel_hash`/`serves()` regardless of family name, so the COMMON case for
a new repo is that it already installs with zero registration. This tool works
out which case applies:

  * the geometry is already served by a kernel set -> nothing to add; report the
    set and stop. No spec, no recipe, no alias, no registry edit.
  * the geometry needs a new spec of a family the recipes already support ->
    hand the repo to `build-new-spec --repo` (SPEC/BUILD/GATE/CATALOGUE/REBUILD/
    VALIDATE); on failure that script parks the attempt in `candidates/` with a
    reason and edits no registry file.
  * a genuinely new architecture -> register a refusing-stub family across the
    tree (advanced `--name` mode), with family-doctor verifying a staged copy.
    The dev then fills only the arithmetic; the first failing command names the
    exact function.

Consistency, not convenience, is the point: the same geometry must always land
on the same kernels, whichever repo it arrives in. Where code exists somewhere
in the tree, it is reused everywhere possible -- a new family name that is just
an existing geometry gets an alias, never a duplicate recipe.

No engine switch case is added: SupportedModelFamily's `default` in
all_models.hpp already returns null with a clear "not a chat model this build
can run" error, which is the correct behaviour until a real engine exists.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SURFACE = ("recipe", "layout", "pack_plan", "programs", "builds",
           "hf_config_check", "manifest_layout", "KERNEL_SOURCES", "GEN_KERNELS")


def identifier(name):
    ident = re.sub(r"[^a-z0-9_]", "_", name.strip().lower()).strip("_")
    if not ident or not ident[0].isalpha():
        raise ValueError(f"--name {name!r} is not a usable family key (try e.g. 'mythos')")
    return ident


def load_gen_kernel_spec():
    """Import gen-kernel-spec.py by path to reuse its exact repo-crawl helpers."""
    import importlib.util
    path = ROOT / "utilities" / "gen-kernel-spec.py"
    spec = importlib.util.spec_from_file_location("gen_kernel_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def derive_repo_facts(repo, gguf=""):
    """(generic_name, spec_dict | None, family | None, note) for a repo.

    Uses the same helpers gen-kernel-spec / oflm add use, so a repo resolves to
    a geometry without any prior registration. `spec_dict` is None when the
    repo cannot be read offline; the note then says why.
    """
    gks = load_gen_kernel_spec()
    name = gks.generic_model_name(repo, gguf)
    try:
        cfg, rv, base = None, None, None
        try:
            cfg, rv = gks.spec_config(repo)
        except Exception:
            base, cfg, rv = gks.spec_config_via_base_chain(repo)
        if cfg is None and gguf and Path(gguf).is_file():
            from oflm_add import read_gguf_inventory
            inv = read_gguf_inventory(gguf, limit_mb=96)
            if inv:
                from recipes.spec import ModelSpec
                cfg = ModelSpec.from_gguf_metadata(inv[0]).to_dict()
        if cfg is None:
            return name, None, None, "could not read config.json, base chain, or local GGUF"
        family = gks.family_from_config(cfg) if "family" not in cfg else cfg["family"]
        return name, cfg, family, (f"from {base}" if base else "from repo config")
    except Exception as exc:                          # noqa: BLE001
        return name, None, None, f"derivation failed: {exc}"


def find_serving_set(spec_dict):
    """(set_name, set_dir) whose GEOMETRY serves spec_dict, or (None, None).

    Geometry only -- no name, no family. This is the 'zero registration needed'
    decision: if a set already serves it, oflm add will link it at install.
    """
    if not spec_dict:
        return None, None
    sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
    try:
        from oflm_add import (scan_kernel_sets, find_system_xclbin_root,
                              user_xclbin_dir, tree_xclbins)
        from recipes.spec import ModelSpec
        cand = ModelSpec.from_dict(spec_dict)
        roots = [find_system_xclbin_root(), user_xclbin_dir(None), tree_xclbins()]
        kdir, name, _ = scan_kernel_sets(cand, roots)
        return name, kdir
    except Exception:                                 # noqa: BLE001
        return None, None


def read(path):
    return path.read_text(encoding="utf-8")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def insert_once(text, anchor, addition, where="before"):
    """Idempotent insert relative to a unique anchor string.

    Returns (new_text, changed). If `addition` is already present, unchanged.
    """
    if addition.strip() and addition.strip() in text:
        return text, False
    if anchor not in text:
        raise ValueError(f"anchor not found: {anchor[:80]!r}")
    if text.count(anchor) != 1:
        raise ValueError(f"anchor is not unique ({text.count(anchor)} matches): {anchor[:80]!r}")
    if where == "before":
        new = text.replace(anchor, addition + anchor, 1)
    else:
        new = text.replace(anchor, anchor + addition, 1)
    return new, new != text


# ---------------------------------------------------------------- scaffold module
def scaffold(name, family, recipe_kind):
    reexports = ""
    if recipe_kind == "dense":
        reexports = (
            "\n# ---- re-export contract (dx.py reads these off the family module;\n"
            "#      missing ones die with AttributeError at export time) ----\n"
            "from . import dense as _dense\n"
            "band_bytes = _dense.band_bytes\n"
            "chunk_bytes = _dense.chunk_bytes\n"
            "attn_flags = _dense.attn_flags\n"
            "probe_env = _dense.probe_env\n"
            "DENSE_FAMILIES = _dense.DENSE_FAMILIES\n\n"
        )
    lines = [
        f'"""{family} open-kernel recipe (scaffold).',
        "",
        "REGISTERED BY `oflm dev add-family`. Every entry point below REFUSES until",
        "you implement this family's arithmetic. A scaffold that returned plausible",
        "numbers would emit kernels that drop a layer and still report parity against",
        "a replica making the same mistake. Fill each TODO; the next failing command",
        "names the exact function to implement.",
        '"""',
        "from __future__ import annotations",
        "",
        "from .spec import ModelSpec",
        "",
        f"FAMILY = {family!r}",
        reexports,
    ]
    for fn in SURFACE:
        if fn in ("KERNEL_SOURCES", "GEN_KERNELS"):
            lines.append(f"{fn} = ()  # TODO({family}): list the kernel sources")
            continue
        lines.append(f"def {fn}(spec: ModelSpec, *args, **kwargs):")
        lines.append(f'    raise NotImplementedError({f"{family}.{fn} is not implemented yet -- this is the next function to fill in"!r})')
        lines.append("")
    return "\n".join(lines)


def builder_stubs(name, family):
    """Two spec builders that refuse with a precise pointer."""
    hf = f'''def _{name}_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """TODO({family}): derive this family's geometry from config.json.

    Read the keys you need with `_need(cfg, key)` (it raises SpecError naming a
    missing key). Return a ModelSpec whose `family` is {family!r}. Keep fields
    that only this family uses zero-valued on every other family, and add them
    to the drop list in to_dict() so no shipped spec's hash moves.
    """
    raise SpecError("{family}: the HF spec builder is not implemented yet "
                    "(utilities/add-family.py stub -- implement _{name}_hf)")


'''
    gguf = f'''def _{name}_gguf(md: Mapping[str, Any]) -> ModelSpec:
    """TODO({family}): derive this family's geometry from GGUF metadata.

    The architecture is `general.architecture`; per-layer values ride as
    arrays -- check them against the declared pattern rather than guessing, and
    refuse variants by the keys that say so, never by a size heuristic.
    """
    raise SpecError("{family}: the GGUF spec builder is not implemented yet "
                    "(utilities/add-family.py stub -- implement _{name}_gguf)")


'''
    return hf, gguf


# ---------------------------------------------------------------- staged edits
def apply_edits(stage, name, family, recipe_kind, fields, adapter):
    """Make every mechanical edit in `stage`. Returns a list of edit descriptions."""
    edits = []
    recipes = stage / "open_kernels" / "recipes"
    addon = stage / "utilities" / "oflm-add" / "oflm_add" / "__init__.py"
    hpp = stage / "src" / "include" / "AutoModel" / "model_families.hpp"

    # 1. the recipe scaffold
    write(recipes / f"{name}.py", scaffold(name, family, recipe_kind))
    edits.append(f"wrote recipes/{name}.py")

    # 2. families.py: FAMILIES tuple + family_module dispatch
    p = recipes / "families.py"
    t = read(p)
    # Insert into the tuple right after its opening paren.
    t2 = t.replace('FAMILIES = ("', f'FAMILIES = ("{family}", "', 1)
    if t2 != t:
        t = t2; edits.append("families.py: added to FAMILIES")
    anchor = '    if name in NOT_IMPLEMENTED:\n'
    addition = (f'    if name == "{family}":\n'
                f'        from . import {name}\n'
                f'        return {name}\n')
    t2, ch = insert_once(t, anchor, addition)
    if ch:
        t = t2; edits.append("families.py: added family_module() branch")
    write(p, t)

    # 3. spec.py: builder stubs + map entries (+ optional spec fields)
    p = recipes / "spec.py"
    t = read(p)
    hf, gguf = builder_stubs(name, family)
    # Place the stubs just before HF_FAMILIES (all builders live above it).
    t2, ch = insert_once(t, 'HF_FAMILIES = {', hf + gguf)
    if ch:
        t = t2; edits.append("spec.py: added refusing spec builders")
    # Insert new keys before each map's closing brace (exact markers from the
    # tree). Idempotent: skip when this family's builder is already a value.
    marker = '"gemma4_text": _gemma4_unsupported}'
    if marker in t and f'_{name}_hf' not in t.split("HF_FAMILIES")[1].split("}")[0]:
        t = t.replace(marker,
                      f'"gemma4_text": _gemma4_unsupported,\n'
                      f'                "{name}": _{name}_hf}}', 1)
        edits.append("spec.py: registered HF model_type")
    marker = '"gemma4": _gemma4_gguf}'
    if marker in t and f'_{name}_gguf' not in t.split("GGUF_FAMILIES")[1].split("}")[0]:
        t = t.replace(marker,
                      f'"gemma4": _gemma4_gguf, "{name}": _{name}_gguf}}', 1)
        edits.append("spec.py: registered GGUF architecture")
    marker = '_gemma4_gguf: "gemma4"}'
    if marker in t and f'_{name}_hf: "{family}"' not in t:
        t = t.replace(marker,
                      f'_gemma4_gguf: "gemma4",\n'
                      f'               _{name}_hf: "{family}", _{name}_gguf: "{family}"}}', 1)
        edits.append("spec.py: registered builders in _FAMILY_OF")
    # ROLE_TENSORS: without an entry here, quant_map_from_gguf_types() refuses
    # the family with "no tensor-role table for family".
    marker = 'for _f in ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2", "k2", "gemma4"):'
    if marker in t and f'"{family}"' not in t.split(marker)[1][:120]:
        t = t.replace(marker, marker.replace('"gemma4")', f'"gemma4", "{family}")'), 1)
        edits.append("spec.py: added family to ROLE_TENSORS loop")
    # Optional new spec fields, dropped at their default so shipped hashes stay.
    for spec_field in fields:
        m = re.match(r"([a-z_][a-z0-9_]*):(\w+)=(.+)", spec_field)
        if not m:
            raise ValueError(f"--fields entry {spec_field!r} must be name:type=default")
        fname, ftype, fdefault = m.groups()
        anchor = '    norm_groups: int = 1\n'
        decl = f'    {fname}: {ftype} = {fdefault}\n'
        if decl not in t and anchor in t:
            t = t.replace(anchor, anchor + decl, 1)
            edits.append(f"spec.py: added field {fname}")
        # Add to the drop list so to_dict omits it at default.
        anchor = 'GEMMA4_FIELDS = ('
        if fname not in t.split("GEMMA4_FIELDS = (")[-1][:300]:
            t = t.replace(anchor, f'GEMMA4_FIELDS = ("{fname}", ', 1)
            edits.append(f"spec.py: added {fname} to the to_dict drop list")
    write(p, t)

    # 4. oflm-add tables
    if addon.is_file():
        t = read(addon)
        marker = 'GGUF_ARCH_FAMILIES = {'
        # Use the CONFIG_MODEL_FAMILIES closing entry as the anchor (stable).
        for mapname in ("CONFIG_MODEL_FAMILIES", "GGUF_ARCH_FAMILIES"):
            m = re.search(rf'{mapname} = \{{.*?\n\}}', t, re.S)
            if not m:
                raise ValueError(f"oflm-add: cannot find {mapname} declaration")
            block = m.group(0)
            if f'"{name}": "{family}"' not in block:
                addition = block[:-1] + f'    "{name}": "{family}",\n' + '}'
                t = t.replace(block, addition, 1)
                edits.append(f"oflm-add: added {mapname} entry")
        write(addon, t)

    # 5. frontend: alias the open-only family to an EXISTING adapter. No new
    # enum value and no engine switch case: SupportedModelFamily's `default`
    # already refuses a tag with no chat engine, and open families borrow an
    # adapter via `_shared_select_open_engine` (open-family-addition:76-80).
    if hpp.is_file():
        t = read(hpp)
        if f'SupportedModelFamily::{adapter}' not in t:
            raise ValueError(f"--adapter {adapter!r} is not a SupportedModelFamily enum value "
                             "in model_families.hpp (pick an existing adapter)")
        map_anchor = '        {"whisper-v3", SupportedModelFamily::error_whiper},'
        if f'{{"{name}", SupportedModelFamily::{adapter}}}' not in t and map_anchor in t:
            t = t.replace(map_anchor,
                          f'        {{"{name}", SupportedModelFamily::{adapter}}},  // open-only; '
                          f'alias to the adapter that owns this decoder\n' + map_anchor, 1)
            edits.append(f"model_families.hpp: aliased to {adapter} (no new enum)")
        write(hpp, t)
    return edits


def stage_tree():
    """Copy the parts family-doctor and the edits touch into a temp tree."""
    tmp = tempfile.mkdtemp(prefix="oflm-add-family-")
    stage = Path(tmp) / "tree"
    for rel in ("open_kernels/recipes", "src/include/AutoModel", "utilities/oflm-add/oflm_add"):
        shutil.copytree(ROOT / rel, stage / rel)
    # family-doctor needs these too.
    for rel in ("open_kernels/recipes/patches.json",):
        shutil.copy2(ROOT / rel, stage / rel)
    return stage, tmp


def run_doctor(stage):
    doctor = ROOT / "utilities" / "family-doctor.py"
    p = subprocess.run([sys.executable, str(doctor)], cwd=ROOT,
                       capture_output=True, text=True,
                       env={**os.environ, "OFLM_FAMILY_DOCTOR_ROOT": str(stage)})
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def register_family(name, recipe_kind, fields, adapter, apply):
    """Stage all mechanical edits, verify with family-doctor, optionally write."""
    stage, tmp = stage_tree()
    try:
        edits = apply_edits(stage, name, name, recipe_kind, fields, adapter)
        ok, out = run_doctor(stage)
        print("\n".join(f"add-family: {e}" for e in edits))
        print("\n[family-doctor on staged tree]\n" + out)
        if not ok:
            print("add-family: FAILED verification; the real tree was NOT touched")
            return 1
        if not apply:
            print("\nadd-family: staged verification passed; pass --apply to write it")
            return 0
        for rel in ("open_kernels/recipes", "src/include/AutoModel", "utilities/oflm-add/oflm_add"):
            src, dst = stage / rel, ROOT / rel
            for f in src.rglob("*"):
                if not f.is_file() or "__pycache__" in f.parts or f.suffix in (".pyc", ".so"):
                    continue
                target = dst / f.relative_to(src)
                try:
                    staged_text = read(f)
                except UnicodeDecodeError:
                    continue
                try:
                    current = read(target) if target.is_file() else None
                except UnicodeDecodeError:
                    current = None
                if current != staged_text:
                    write(target, staged_text)
        print(f"\nadd-family: family '{name}' registered and verified.")
        print("Remaining hand-work (the arithmetic only):")
        print(f"  - {name}/recipe entry points → first error names the function")
        print(f"  - _{name}_hf / _{name}_gguf in recipes/spec.py → SpecError names them")
        print(f"  Then: oflm dev build-new-spec --repo <repo>")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def handle_repo(repo, gguf, apply):
    """Repo-driven mode: do the least work that makes this geometry installable."""
    name, spec_dict, family, note = derive_repo_facts(repo, gguf)
    print(f"add-family: {repo}")
    print(f"  generic name : {name}")
    print(f"  derivation   : {note}")
    if spec_dict is None:
        print("  RESULT       : could not derive geometry from the repo offline.")
        print("                 Download the GGUF and pass --gguf <file>, or use --name.")
        return 1
    print(f"  family       : {family or '(none inferred)'}")
    set_name, set_dir = find_serving_set(spec_dict)
    if set_name:
        print(f"  RESULT       : geometry already served by kernel set '{set_name}'.")
        print("                 Nothing to add: `oflm add <repo>` links it by geometry.")
        print("                 No spec, recipe, alias or registry file was changed.")
        return 0
    # No set serves this geometry yet. build-new-spec owns the whole mechanical
    # flow; it writes a spec, links when a spec already covers the size, and
    # parks the attempt in candidates/ on failure WITHOUT editing a registry.
    print("  RESULT       : no installed set serves this geometry; delegating to build-new-spec.")
    cmd = [sys.executable, str(ROOT / "utilities" / "build-new-spec.py"), "--repo", repo]
    if gguf:
        cmd += ["--gguf", gguf]
    if not apply:
        print("                 (dry run; pass --apply to actually run it)")
        print("  would run    : " + " ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", help="a HF/ModelScope repo: derive its geometry and do the least work")
    ap.add_argument("--gguf", default="", help="local GGUF for --repo when the base config is gated")
    ap.add_argument("--name", help="advanced: family key to register directly (bypasses --repo)")
    ap.add_argument("--recipe", choices=("dense", "custom"), default="dense")
    ap.add_argument("--adapter", default="llama3",
                    help="an EXISTING SupportedModelFamily to alias this open-only family to; "
                         "no new enum value or engine switch case (per open-family-addition). "
                         "Change it if this family needs a different decoder adapter.")
    ap.add_argument("--fields", action="append", default=[],
                    help="optional new ModelSpec field name:type=default (repeatable)")
    ap.add_argument("--verify", action="store_true",
                    help="explicitly run family-doctor on the staged tree (always done; "
                         "accepted for clarity and scripting)")
    ap.add_argument("--apply", action="store_true",
                    help="write the staged, doctor-verified edits back to the tree")
    args = ap.parse_args()
    if args.repo:
        return handle_repo(args.repo, args.gguf, args.apply)
    if not args.name:
        ap.error("pass --repo <user>/<repo> (normal) or --name <key> (advanced)")
    try:
        name = identifier(args.name)
    except ValueError as exc:
        print(f"add-family: {exc}", file=sys.stderr)
        return 2
    return register_family(name, args.recipe, args.fields, args.adapter, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
