#!/usr/bin/env python3
"""Add a model SIZE to a family that already has kernels: one command.

    python utilities/build-new-spec.py --repo Qwen/Qwen3-0.6B

What it does, in order, and why each step exists:

  1. SPEC      the geometry comes from the model's config.json on HuggingFace
               (utilities/gen-kernel-spec.py). A fine-tune whose base model
               already has a spec LINKS to it instead -- a fine-tune never gets
               its own kernels.
  2. BUILD     the family's kernel set is compiled for the new size. This is
               where the catalogue refuses a geometry it has not seen, so the
               run is made with OPEN_KERNELS_UNVALIDATED=1: the build proves the
               design COMPILES at the new geometry, which is half the question.
  3. GATE      the attention kernel is RUN at the new geometry and scored
               against an fp64 oracle (open_kernels/designs/attn/: the
               production worker, synthetic inputs, no model). This is the
               other half: the numbers are right, not just the compile.
  4. CATALOGUE the validated point is written into open_kernels/recipes/
               catalogue.py, so the next build of this size needs no override.
  5. REBUILD   the kernel set is compiled again through the validated path,
               so what ships is a build the catalogue allows.

Only step 3 needs an NPU; everything else is a compile. Nothing here needs the
model's weights, a container or the engine.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "open_kernels" / "recipes" / "specs"
PROBE = ROOT / "open_kernels" / "designs" / "attn" / "probe_dense.py"
BUILD_DESIGN = ROOT / "open_kernels" / "build_design.py"
MAKE_TEST = ROOT / "open_kernels" / "designs" / "attn" / "make_test.py"
COMPARE = ROOT / "open_kernels" / "designs" / "attn" / "compare.py"
HARNESS = ROOT / "open_kernels" / "harness"
EXPORT = ROOT / "utilities" / "export-kernels.py"
GEN_SPEC = ROOT / "utilities" / "gen-kernel-spec.py"
ADD_POINT = ROOT / "utilities" / "catalogue-add-point.py"

# the fixture that validates each other op's new point, for the report
OP_FIXTURE = {
    "gemv_q4": "open_kernels/designs/gemv_q4/make_test.py (--K <width>)",
    "gemv_q8": "the OPEN-QUANT-Q8 procedure (no per-K fixture exists)",
    "ln": "open_kernels/designs/ln/make_test.py (--N <width>)",
    "lm_head_q4": "the q4 head has no fixture of its own: designs/lm_head_q8/make_test.py "
                  "is the model to port (the q4 head is the same kernel with another band split)",
    "lm_head_q8": "designs/lm_head_q8/make_test.py (--K <width>)",
}


def run(cmd, env=None, cwd=None, check=True, quiet=False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env or os.environ, cwd=cwd or str(ROOT))
    if check and p.returncode != 0:
        raise SystemExit(f"build-new-spec: FAILED: {' '.join(cmd)}")
    return p.returncode


def load_export_module():
    """export-kernels.py owns the venv and the toolchain PATH; reuse them."""
    spec = importlib.util.spec_from_file_location("_export_kernels", EXPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toolchain_env(mod) -> dict:
    """The PATH the IRON builds need: Peano (llvm-aie) and the XRT tools."""
    env = dict(os.environ)
    parts = []
    aie_bin = mod.llvm_aie_bin()
    if aie_bin:
        parts.append(str(aie_bin))
    parts.append(str(mod.XRT_BIN))
    parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(parts)
    lib = Path(mod.XRT_ROOT) / "lib"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(lib), env.get("LD_LIBRARY_PATH", "")])
    return env


def spec_geometry(spec_path: Path) -> dict:
    """The attention point this spec asks for, in the catalogue's key order."""
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import dense as D
    from recipes.load import load_spec
    spec = load_spec(spec_path)
    return {
        "point": (spec.head_dim, spec.num_heads, spec.num_kv_heads, spec.rotary_dim,
                  bool(spec.qk_norm), bool(spec.attn_gate),
                  bool(spec.qk_norm) and spec.family in D.QKNORM_POST_ROPE,
                  bool(D.qkv_bias(spec))),
        "family": spec.family,
        "name": spec_path.stem,
    }


def preflight(spec_path: Path) -> list[tuple[str, str]]:
    """What stands between this spec and a build.

    Two kinds, and the difference matters: the RECIPE's own limits (a width
    that does not fit a core's L1, say) are not a geometry the gate can
    validate -- the size is simply not supported by this recipe. The
    CATALOGUE's limits are: the attention one is what the gate exists for, and
    the others name the fixture that validates them.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import dense as D
    from recipes.load import load_spec
    from recipes.catalogue import OpRangeError
    spec = load_spec(spec_path)
    problems = []
    try:
        D.geometry(spec)
    except OpRangeError as exc:
        problems.append(("recipe", str(exc)))
    try:
        D.recipe(spec)
    except OpRangeError as exc:
        problems.append(("catalogue", str(exc)))
    return problems


def build_probe(py: Path, env: dict, spec_path: Path, work: Path, rows: int) -> Path:
    out = work / "probe"
    e = dict(env)
    e["ATTN_PROBE_SPEC"] = str(spec_path)
    e["ATTN_PROBE_ROWS"] = str(rows)
    run([py, BUILD_DESIGN, PROBE, out], env=e)
    return out


def harness_binary(work: Path) -> Path:
    """The N-buffer runner (open_kernels/harness). Built once, cached."""
    build = work / "harness"
    exe = build / "run_kernel"
    if not exe.is_file():
        run(["cmake", "-S", HARNESS, "-B", build, "-DCMAKE_BUILD_TYPE=Release"], quiet=True)
        run(["cmake", "--build", build, "-j", "8"], quiet=True)
    if not exe.is_file():
        raise SystemExit("build-new-spec: the harness did not build (open_kernels/harness)")
    return exe


def attn_gate(py: Path, env: dict, spec_path: Path, work: Path, rows: int) -> bool:
    """Build the attention probe at this geometry, run it, score it."""
    out = build_probe(py, env, spec_path, work, rows)
    run([py, MAKE_TEST, "--spec", spec_path, "--rows", rows, "--out", out], env=env)
    exe = harness_binary(work)
    run([exe, out / "attention.cfg"], env=env, cwd=out, quiet=True)
    rc = run([py, COMPARE, "--out", out], env=env, check=False)
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="the HuggingFace repo (Qwen/Qwen3-0.6B)")
    ap.add_argument("--spec-name", default="", help="the spec's stem (default: from the family and the size)")
    ap.add_argument("--rows", type=int, default=17, help="the attention gate's window length")
    ap.add_argument("--work", default="", help="where the build artifacts go (default: build/attn-gate/<spec>)")
    ap.add_argument("--skip-export", action="store_true", help="do not build the family's kernel set")
    ap.add_argument("--skip-gate", action="store_true", help="do not run the attention gate")
    ap.add_argument("--dry-run", action="store_true", help="stop after the spec")
    a = ap.parse_args()

    mod = load_export_module()
    py = mod.ensure_venv()
    env = toolchain_env(mod)

    name = a.spec_name
    if not name:
        sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
        try:
            from oflm_add import family_from_config_dict  # noqa: F401
        except Exception:
            pass
        name = a.repo.split("/")[-1].lower()
    spec_path = SPECS / f"{name}.json"

    # 1. the spec
    print(f"-- spec {name}", flush=True)
    existed = spec_path.is_file()
    rc = run([sys.executable, GEN_SPEC, "--repo", a.repo, "--out", spec_path], env=env, check=False)
    if rc != 0:
        print("build-new-spec: the spec could not be generated", file=sys.stderr)
        return rc
    if not spec_path.is_file():
        print("build-new-spec: no spec was written", file=sys.stderr)
        return 1
    written = json.loads(spec_path.read_text())
    if "link_to" in written:
        print(f"the base model already has kernels: {written['link_to']} -- nothing to build")
        if not existed:
            spec_path.unlink()
        return 0
    print(f"   {spec_path.relative_to(ROOT)}", flush=True)

    geo = spec_geometry(spec_path)
    print(f"   family {geo['family']}, attention point {geo['point']}", flush=True)
    problems = preflight(spec_path)
    fatal = False
    for kind, msg in problems:
        if kind == "recipe":
            fatal = True
            print(f"   the RECIPE does not support this shape: {msg}")
            print("      (a width that does not fit the design -- a different recipe, or a "
                  "different family)")
        else:
            op = msg.split(":")[0]
            print(f"   catalogue: {msg.splitlines()[0]}")
            hint = OP_FIXTURE.get(op)
            if hint:
                print(f"      validate it with {hint}")
    if fatal:
        # a spec the recipe cannot build would break every later export, so it
        # does not stay in the specs dir -- but it is not thrown away either:
        # a recipe that learns the shape (the FF split, say) picks it up.
        cand = ROOT / "open_kernels" / "recipes" / "candidates"
        cand.mkdir(parents=True, exist_ok=True)
        spec_path.rename(cand / spec_path.name)
        print(f"build-new-spec: the recipe cannot build this shape, so the spec moved to "
              f"{cand.relative_to(ROOT)} (the export would fail on it in the specs dir)",
              file=sys.stderr)
        return 1
    if a.dry_run:
        return 0

    work = Path(a.work) if a.work else ROOT / "build" / "attn-gate" / name
    work.mkdir(parents=True, exist_ok=True)

    # 2. the kernel set compiles at this geometry
    if not a.skip_export:
        print("-- build the kernel set (the catalogue's override is on: the geometry is not validated yet)",
              flush=True)
        e = dict(env, OPEN_KERNELS_UNVALIDATED="1")
        rc = run([py, EXPORT, "--specs", name, "--skip-bert"], env=e, check=False)
        if rc != 0:
            print("build-new-spec: the kernel set did not build at this geometry", file=sys.stderr)
            return rc

    # 3. the attention gate
    if not a.skip_gate:
        print("-- attention gate: the production worker at this geometry, against an fp64 oracle",
              flush=True)
        if not attn_gate(py, env, spec_path, work, a.rows):
            print("build-new-spec: the attention gate FAILED -- the catalogue was not touched",
                  file=sys.stderr)
            return 1
        print("   PASS", flush=True)

        # 4. the catalogue
        print("-- catalogue", flush=True)
        point = ", ".join(repr(v) for v in geo["point"])
        rc = run([sys.executable, ADD_POINT, "--op", "attn", "--point", point,
                  "--note", f"{geo['name']} (the attention gate, designs/attn/probe_dense.py)"],
                 env=env, check=False)
        if rc != 0:
            return rc

        # 5. the build through the validated path
        if not a.skip_export:
            print("-- rebuild through the validated path", flush=True)
            rc = run([py, EXPORT, "--specs", name, "--skip-bert"], env=env, check=False)
            if rc != 0:
                print("build-new-spec: the validated rebuild failed", file=sys.stderr)
                return rc

    print(f"\n{name} is ready: the spec, the built kernels and the catalogue entry.")
    print(f"  commit: {spec_path.relative_to(ROOT)} and open_kernels/recipes/catalogue.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
