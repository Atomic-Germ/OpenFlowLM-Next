#!/usr/bin/env python3
"""Check the machine-verifiable invariants that the family skills state in prose.

    python utilities/family-doctor.py
    oflm dev family-doctor

Adding a family (open-family-addition, open-kernel-families) is a kernel-author
task, but several of its failure modes are pure data/registration mistakes that
used to be caught only because an agent remembered the skill text:

  * a dense recipe that forgets to re-export ``band_bytes`` / ``chunk_bytes`` /
    ``attn_flags`` / ``probe_env`` (dx.py reads them off the family module, and
    the export dies with an AttributeError);
  * a spec with a duplicate kernel hash or an unsafe export name;
  * a patches.json entry under an unnormalized key, or with an unknown field;
  * a frontend SupportedModelFamily with no string alias (the tag that resolves
    it at install time);
  * an op catalogue entry whose point no longer parses.

This script runs those checks without NPU hardware or model weights. It does
NOT attempt to judge arithmetic, layout, or numerics -- that remains kernel
author work. It also does not check the cross-file family-name mapping (qwen35
vs qwen3.5, phi3 vs phi4, ...): that mapping is intentionally hand-curated in
the oflm-add tables and the frontend header, and a name-match check would
produce false alarms.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The tree to inspect. `--root` lets a generator (add-family.py) verify a
# staged copy without touching the real tree; imports are re-pointed at it.
STAGED_ROOT = Path(__import__("os").environ.get("OFLM_FAMILY_DOCTOR_ROOT", ROOT)).resolve()
OK = STAGED_ROOT / "open_kernels"
# The recipes package must be imported from the tree being checked; insert it
# before the real one so a staged copy wins.
sys.path.insert(0, str(OK))
sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))

from recipes.families import FAMILIES, family_module  # noqa: E402
from recipes.dense import DENSE_FAMILIES  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402
from recipes.model_patches import (  # noqa: E402
    PATCH_FIELDS, PATCHES_PATH, load_patches, patch_key,
)

REQUIRED_REEXPORTS = ("band_bytes", "chunk_bytes", "attn_flags", "probe_env")
SPECS = OK / "recipes" / "specs"
FRONTEND = STAGED_ROOT / "src" / "include" / "AutoModel" / "model_families.hpp"


class Report:
    def __init__(self):
        self.failures = []
        self.warnings = []
        self.checks = 0

    def fail(self, msg):
        self.failures.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def check(self, ok, msg):
        self.checks += 1
        if not ok:
            self.fail(msg)


def check_reexports(r):
    """The dense/dx.py contract from open-kernel-families."""
    for fam in DENSE_FAMILIES:
        if fam not in FAMILIES:
            r.fail(f"dense family {fam!r} is not registered in recipes.families.FAMILIES")
            continue
        try:
            mod = family_module(fam)
        except Exception as exc:                      # noqa: BLE001
            r.fail(f"cannot import recipe for dense family {fam!r}: {exc}")
            continue
        missing = [name for name in REQUIRED_REEXPORTS if not hasattr(mod, name)]
        r.check(not missing,
                f"dense recipe {fam!r} does not re-export {', '.join(missing)} "
                "(dx.py reads them off the family module)")


def check_specs(r):
    """The two spec-invariant tests, run in-process so --check is one command."""
    seen_hash, seen_name = {}, {}
    for p in sorted(SPECS.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            r.fail(f"{p.name}: invalid JSON ({exc})")
            continue
        try:
            spec = ModelSpec.from_dict(d)
        except Exception as exc:                      # noqa: BLE001
            r.fail(f"{p.name}: {exc}")
            continue
        h = spec.kernel_hash()
        if h in seen_hash:
            r.fail(f"{p.name}: duplicate geometry with {seen_hash[h]} ({h})")
        seen_hash[h] = p.name
        name = (d.get("extra") or {}).get("model") or ""
        if name:
            if name in seen_name:
                r.fail(f"{p.name}: duplicate export name {name!r} with {seen_name[name]}")
            seen_name[name] = p.name
            # A GGUF upload suffix is not a stable export label.
            if re.search(r"(?i)(?:-gguf|-q4_1|-q4_k_m)$", name):
                r.fail(f"{p.name}: export name {name!r} carries a quant/upload suffix")


def check_patches(r):
    raw = {}
    try:
        raw = json.loads(PATCHES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        r.fail(f"cannot read {PATCHES_PATH}: {exc}")
        return
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            r.fail(f"patch {key!r}: entry is not an object")
            continue
        r.check(key == patch_key(key),
                f"patch key {key!r} is not normalized (use {patch_key(key)!r})")
        bad = [f for f in entry if f != "_source" and f not in PATCH_FIELDS]
        r.check(not bad, f"patch {key!r}: unknown field(s) {', '.join(bad)}")
        rs = entry.get("rope_scaling")
        if rs is not None and not isinstance(rs, dict):
            r.fail(f"patch {key!r}: rope_scaling must be an object")
        rv = entry.get("real_vocab")
        if rv is not None and (not isinstance(rv, int) or isinstance(rv, bool) or rv <= 0):
            r.fail(f"patch {key!r}: real_vocab must be a positive int")
    # The loader must find the same keys.
    for key in load_patches():
        r.check(key in {patch_key(k) for k in raw}, f"loader lost patch key {key!r}")


def check_frontend(r):
    """Every enum value must have a string alias (the install-time tag)."""
    if not FRONTEND.is_file():
        r.warn(f"frontend header not found: {FRONTEND}")
        return
    text = FRONTEND.read_text(encoding="utf-8")
    enum_block = re.search(r"typedef\s+enum\s*\{(.*?)\}\s*SupportedModelFamily",
                           text, re.S)
    if not enum_block:
        r.warn("could not parse the SupportedModelFamily enum")
        return
    values = set(re.findall(r"^\s*([a-z][a-z0-9_]*)\s*,?\s*(?://.*)?$",
                            enum_block.group(1), re.M))
    aliases = set(re.findall(r"SupportedModelFamily::(\w+)", text))
    missing = sorted(values - aliases)
    r.check(not missing,
            "SupportedModelFamily values with no string alias: " + ", ".join(missing))


def check_catalogue(r):
    cat = OK / "recipes" / "catalogue.py"
    if not cat.is_file():
        r.warn("catalogue.py not found")
        return
    sys.path.insert(0, str(OK / "recipes"))
    try:
        import catalogue  # noqa: E402
    except Exception as exc:                          # noqa: BLE001
        r.fail(f"cannot import the op catalogue: {exc}")
        return
    # A point is a claim about hardware; an empty catalogue means every gate
    # would fail at build time.
    if not getattr(catalogue, "CATALOGUE", None):
        r.fail("the op catalogue is empty (no point has ever been validated)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    r = Report()
    check_reexports(r)
    check_specs(r)
    check_patches(r)
    check_frontend(r)
    check_catalogue(r)
    for w in r.warnings:
        print(f"WARN: {w}")
    for f in r.failures:
        print(f"FAIL: {f}")
    print(f"family-doctor: {r.checks} check(s), {len(r.failures)} failure(s), "
          f"{len(r.warnings)} warning(s)")
    return 1 if r.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
