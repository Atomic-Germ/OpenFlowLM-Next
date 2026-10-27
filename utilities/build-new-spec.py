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
  3. GATE      every point the catalogue is missing is RUN and scored before
               the build: the attention kernel against an fp64 oracle
               (designs/attn/), and each other op through its own fixture
               (utilities/op-gate.py: gemv_q4 at the new K, ln at the new width,
               ...). A point is written into the catalogue only after its
               fixture PASSES, so "the numbers are right" is checked, not
               assumed, for every geometry the new size introduces.
  4. CATALOGUE the validated point is written into open_kernels/recipes/
               catalogue.py, so the next build of this size needs no override.
  5. REBUILD   the kernel set is compiled again through the validated path,
               so what ships is a build the catalogue allows.
  6. REGISTER  the spec joins the frozen lists the tests enforce
               (utilities/register-spec.py): its spec_hash in test_lfm2.py and,
               when the name derived from the file stem is not the model it
               exports as, its name in test_spec_model_names.py. Mechanical
               edits, scripted -- never hand-written.

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
REGISTER = ROOT / "utilities" / "register-spec.py"
OP_GATE = ROOT / "utilities" / "op-gate.py"

def run(cmd, env=None, cwd=None, check=True, quiet=False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env or os.environ, cwd=cwd or str(ROOT))
    if check and p.returncode != 0:
        raise SystemExit(f"build-new-spec: FAILED: {' '.join(cmd)}")
    return p.returncode


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def preflight(spec_path: Path) -> tuple[list[str], list[tuple[str, dict]]]:
    """(the RECIPE's walls, the missing CATALOGUE points) in one pass.

    The recipe is the FAMILY's (`recipes/families.for_spec`), not `dense`:
    Qwen3.5 / LFM2 / the MoE each have their own, and asking `dense` about them
    raises "dense recipe given a 'qwen35' spec" -- which used to be printed as a
    catalogue problem and swallow the real validation of those families
    entirely.

    The two kinds are different: a RECIPE wall (a width that does not fit a
    core's L1) means this size is not supported by this recipe at all, and no
    gate can fix it. A missing CATALOGUE point is a claim the hardware has not
    proved yet -- utilities/op-gate.py proves it.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import catalogue as C
    from recipes.families import for_spec
    from recipes.load import load_spec
    from recipes.catalogue import OpRangeError
    spec = load_spec(spec_path)
    walls = []
    with C.collecting() as missing:
        try:
            for_spec(spec).recipe(spec)
        except OpRangeError as exc:
            walls.append(str(exc))
    return walls, missing


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


def attn_point_validated(spec) -> bool:
    """True when the spec's attention tuple is already in the catalogue.

    Not every family can be gated: designs/attn/probe_dense.py refuses a family with
    a q/k/v bias (Qwen2.5), and it should -- the probe compiles the production worker
    without that element. When the tuple has ALREADY been validated (by whatever
    procedure entered it), there is nothing for the gate to add, so the workflow
    skips it instead of failing a spec that is otherwise ready.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import dense as D
    from recipes.catalogue import CATALOGUE
    from recipes.load import load_spec
    s = spec if not isinstance(spec, Path) else load_spec(spec)
    tpl = CATALOGUE["attn"]
    kw = dict(head_dim=s.head_dim, num_heads=s.num_heads, num_kv_heads=s.num_kv_heads,
              rotary_dim=s.rotary_dim, qk_norm=s.qk_norm, attn_gate=s.attn_gate,
              qk_norm_post_rope=s.qk_norm and s.family in D.QKNORM_POST_ROPE,
              qkv_bias=D.qkv_bias(s))
    return any(c.value(kw) in c.points for c in tpl.combos)


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
    walls, missing = preflight(spec_path)
    for msg in walls:
        print(f"   the RECIPE does not support this shape: {msg}")
        print("      (a width that does not fit the design -- a different recipe, or a "
              "different family)")
    if missing:
        print(f"   the catalogue has not validated {len(missing)} point(s) this size needs:")
        for op, kw in missing:
            print(f"      {op}: {kw}")
    if walls:
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

    # 3a. every op EXCEPT attention: run its own fixture and write the point.
    # Attention is 3b, because its gate builds the model's worker, not a fixture.
    if missing:
        print("-- op gates: run each missing point's fixture before the build", flush=True)
        sys.path.insert(0, str(ROOT / "open_kernels"))
        from recipes.load import load_spec
        mod_gate = _load_module("_op_gate", OP_GATE)
        try:
            mod_gate.gate_points(load_spec(spec_path), spec_path, py, env,
                                 harness_binary(work), skip=("attn",))
        except mod_gate.GateError as exc:
            print(exc, file=sys.stderr)
            print("   (that op has no runnable fixture: the size needs a DESIGN step, not a "
                  "script run -- see utilities/op-gate.py NOT_AUTOMATED)", file=sys.stderr)
            return 1

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
    if not a.skip_gate and attn_point_validated(spec_path):
        # the tuple is already in the catalogue: nothing to run and nothing to add, and
        # some families (a q/k/v bias) are not expressible in the probe at all
        print("-- attention gate: this tuple is already validated in the catalogue; skipping",
              flush=True)
    elif not a.skip_gate:
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

    # 6. register the spec in the frozen test lists
    print("-- register the spec (utilities/register-spec.py)", flush=True)
    rc = run([sys.executable, REGISTER, name], env=env, check=False)
    if rc != 0:
        return rc

    print(f"\n{name} is ready: the spec, the built kernels and the catalogue entry.")
    print(f"  commit: {spec_path.relative_to(ROOT)}, open_kernels/recipes/catalogue.py and "
          "the two test files register-spec.py touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
