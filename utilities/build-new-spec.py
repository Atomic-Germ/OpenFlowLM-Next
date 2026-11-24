#!/usr/bin/env python3
"""Add a model SIZE to a family that already has kernels: one command.

    python utilities/build-new-spec.py --repo Qwen/Qwen3-0.6B

What it does, in order, and why each step exists:

  1. SPEC      the geometry comes from the model's config.json on HuggingFace
               (utilities/gen-kernel-spec.py), or from the GGUF header when the
               repo ships only GGUF weights (a quantized upload has no
               config.json -- the file names its own geometry). A fine-tune
               whose base model already has a spec LINKS to it instead -- a
               fine-tune never gets its own kernels.
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
  6. VALIDATE  utilities/register-spec.py checks the spec's geometry signature
               and export label against the shipped set (no duplicate shape,
               no duplicate export directory). It edits NO files: the contract
               is the geometry, so there is no hash/name list to commit. What
               you commit is the spec JSON and the catalogue points above.

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
CANDIDATES = ROOT / "open_kernels" / "recipes" / "candidates"
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


def to_candidates(spec_path: Path, reason: str, detail: str = "") -> None:
    """Move a spec that cannot be validated out of specs/ before it ships.

    CMake exports EVERY spec in specs/ on the next `cmake --build` — there is no
    per-spec opt-out. A spec left there after a failed gate would be compiled
    with an unvalidated geometry and break the build for everyone. The
    candidates/ dir is the parking lot: a recipe that learns the shape, or a
    fixture that gets written, picks it up later.

    A .reason file records WHY alongside it, so the next dev (or a script) does
    not have to rediscover the failure.
    """
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    if spec_path.is_file():
        spec_path.rename(CANDIDATES / spec_path.name)
    reason_path = CANDIDATES / f"{spec_path.stem}.reason"
    reason_path.write_text(
        f"{reason}\n" + (f"\n{detail.rstrip()}\n" if detail else ""),
        encoding="utf-8",
    )
    print(f"build-new-spec: {spec_path.name} moved to "
          f"{CANDIDATES.relative_to(ROOT)} ({reason}); see {reason_path.name}",
          file=sys.stderr)


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
    # The op gates run fixture scripts that import `open_kernels.q4_1_pack` and
    # `gguf_pool` as packages. op-gate.py's standalone toolchain() puts the repo
    # root on PYTHONPATH; this shared builder must do the same or a fixture fails
    # with ModuleNotFoundError and the spec is parked in candidates for an
    # ENVIRONMENT reason rather than a geometry/gate reason.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
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


def attn_gate_support(spec_path: Path) -> tuple[bool, str]:
    """Whether designs/attn/probe_dense.py can gate this spec, and why not.

    The probe covers the single-row and RB 1/2/4 block paths. A q/k/v-bias
    family used to be refused here (that point needed the manual OPEN-ATTN-QKV-BIAS
    whole-layer procedure), but the probe now streams the bias fifo the production
    worker uses, so those specs gate like any other. The capability is still
    decided from the spec BEFORE exporting kernels: a refusal at step 3b used to
    arrive after a full, successful export and read as a hard failure.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes.families import for_spec
    from recipes.load import load_spec
    s = load_spec(spec_path)
    mod = for_spec(s)
    if not hasattr(mod, "geometry"):
        # The attention probe is the DENSE worker's gate. MoE families expose
        # recipe()/layout() instead; their attention point enters through the
        # family bring-up procedure, not this probe. Say so instead of crashing.
        return False, (f"the attention probe only gates dense-family geometries; "
                       f"family {s.family!r} needs its family procedure")
    g = mod.geometry(s)
    if not hasattr(mod, "attn_flags"):
        return False, (f"family {s.family!r} has no attn_flags(), so the probe cannot build "
                       f"the kernel configuration this family ships")
    if g.RB not in (1, 2, 4):
        return False, f"RB={g.RB} (the probe covers only RB 1/2/4)"
    if getattr(g, "PTAB_ELEMS", 1) not in (1, 2):
        return False, f"a position record of {g.PTAB_ELEMS} elements (attn.h handles one or two)"
    return True, ""


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
    # require() enforces EVERY combo (the main attn tuple AND the later
    # Gemma-4 v_norm/scale/rope_pair group, whose default point is present for
    # everyone). any() let that default group vouch for an unvalidated main
    # tuple, so the gate was skipped and a plain export hit the catalogue wall.
    return all(c.value(kw) in c.points for c in tpl.combos)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="", help="the HuggingFace repo (Qwen/Qwen3-0.6B)")
    ap.add_argument("--gguf-file", default="",
                    help="a local GGUF whose header names the geometry (gated/offline repos); "
                         "paired with --repo for its base-model name")
    ap.add_argument("--spec", default="",
                    help="an EXISTING spec (stem or path) to build and gate: skips step 1. "
                         "For a local GGUF, pass --repo with --gguf-file instead")
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

    if not a.repo and not a.spec:
        ap.error("--repo or --spec is required")
    name = a.spec_name
    if a.spec:
        sp = Path(a.spec)
        # A spec stem can contain dots (qwen3-0.6b, qwen3-1.7b, lfm2-2.6b), so a
        # naif Path.stem would turn "qwen3-0.6b" into "qwen3-0". Treat an
        # existing path as a path, a name that already ends in .json as a file
        # name, and everything else as a bare stem to append .json to.
        if sp.is_file():
            spec_path = sp
        elif a.spec.endswith(".json"):
            spec_path = SPECS / a.spec
        else:
            spec_path = SPECS / f"{a.spec}.json"
        if not spec_path.is_file():
            print(f"build-new-spec: no such spec: {a.spec}", file=sys.stderr)
            return 1
        name = spec_path.stem
        print(f"-- spec {name} (given)", flush=True)
        goto_build = True
    else:
        goto_build = False
    # The spec NAME is a label, not geometry: it comes from the base model the
    # repo quantizes (gen-kernel-spec.py), never from the upload slug. Let the
    # generator write to a scratch path so it can choose the canonical name,
    # then move the result into the specs dir under that name. This keeps a
    # second repo of the same base -- e.g. unsloth/Llama-3.2-1B-Instruct-GGUF
    # after mradermacher/Llama-3.2-1B-Instruct-i1-GGUF -- on ONE spec, and
    # `oflm add` links any geometrically matching model by spec_hash anyway.
    generate = not goto_build and not name
    if generate:
        scratch = ROOT / "build" / "spec-scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        gen_out = scratch / f"{a.repo.split('/')[-1].lower()}.json"
    else:
        spec_path = SPECS / (f"{name}.json" if name else f"{a.repo.split('/')[-1].lower()}.json")
        gen_out = spec_path

    # 1. the spec -- the name is derived from the base model by gen-kernel-spec,
    # not from the repo slug, so report it only once it is known.
    print(f"-- spec from {a.repo or a.spec or a.gguf_file}", flush=True)
    existed = spec_path.is_file() if not generate else (SPECS / f"{gen_out.stem}.json").is_file()
    gen_cmd = [sys.executable, GEN_SPEC]
    if a.gguf_file:
        gen_cmd += ["--gguf", a.gguf_file]
    if a.repo:
        gen_cmd += ["--repo", a.repo]
    gen_cmd += ["--out", gen_out]
    rc = 0 if goto_build else run(gen_cmd, env=env, check=False)
    if rc != 0:
        print("build-new-spec: the spec could not be generated", file=sys.stderr)
        return rc
    if not gen_out.is_file():
        print("build-new-spec: no spec was written", file=sys.stderr)
        return 1
    written = json.loads(gen_out.read_text())
    if "link_to" in written:
        # The geometry is already served by an existing spec. The link file is
        # only a scratch artifact -- it must NOT become a shipped spec (it would
        # duplicate the geometry under a second name and break the export).
        print(f"the base model already has kernels: {written['link_to']} -- nothing to build")
        if gen_out.is_file():
            gen_out.unlink()
        return 0
    if a.dry_run and generate:
        # A dry run reports the derived spec; it never writes into specs/.
        name = gen_out.stem
        spec_path = gen_out
        print(f"   (dry run: {gen_out.relative_to(ROOT)})", flush=True)
    elif generate:
        # The generator picked the canonical filename (extra.model, lower-cased)
        # under scratch; move it into the specs dir under that stem.
        name = gen_out.stem
        spec_path = SPECS / f"{name}.json"
        if spec_path.is_file() and not existed:
            print(f"build-new-spec: {spec_path.name} already exists; keeping it", flush=True)
            gen_out.unlink()
        elif not spec_path.is_file():
            gen_out.rename(spec_path)
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
        # A spec the recipe cannot build would break every later export.
        to_candidates(spec_path, "recipe wall: the recipe cannot build this shape",
                      "\n".join(walls))
        return 1
    if a.dry_run:
        return 0

    work = Path(a.work) if a.work else ROOT / "build" / "attn-gate" / name
    work.mkdir(parents=True, exist_ok=True)

    # Decide attention-gate capability NOW, before any export. A family the probe
    # cannot gate still builds; saying so here is a notification, not a late hard
    # failure after a successful export.
    attn_validated = attn_point_validated(spec_path)
    # Only ask the probe about capability when there is actually a gate to run:
    # an already-validated tuple (or a family the probe does not build) must not
    # crash on a family module with no geometry() just to be skipped.
    attn_supported, attn_why = ((True, "") if attn_validated or a.skip_gate
                                else attn_gate_support(spec_path))
    if not attn_validated and not attn_supported and not a.skip_gate:
        # No scripted gate for this attention shape. Do NOT ship a spec with an
        # unvalidated point: the next cmake build would export it and the
        # catalogue wall would break the build for everyone. Park it.
        to_candidates(spec_path,
                      "no attention gate: the probe cannot validate this shape",
                      f"{attn_why}\n\nA scripted gate for this attention layout is needed "
                      "before this spec can ship (design task).")
        return 1

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
            msg = str(exc)
            if "has no automated gate" in msg or "NOT_AUTOMATED" in msg:
                to_candidates(spec_path,
                              "no op fixture: this size needs a DESIGN step",
                              msg)
            else:
                to_candidates(spec_path,
                              "op gate failed at this geometry",
                              msg + "\n\nA fixture that fits this geometry (e.g. a "
                              "large-K-capable layout) is needed before this spec can ship.")
            return 1

    # 2. the kernel set compiles at this geometry
    if not a.skip_export:
        print("-- build the kernel set (the catalogue's override is on: the geometry is not validated yet)",
              flush=True)
        e = dict(env, OPEN_KERNELS_UNVALIDATED="1")
        rc = run([py, EXPORT, "--specs", name, "--skip-bert", "--skip-diffusion"], env=e, check=False)
        if rc != 0:
            to_candidates(spec_path, "kernel build failed at this geometry",
                          f"export-kernels.py --specs {name} (OPEN_KERNELS_UNVALIDATED=1) "
                          f"exited {rc}")
            return rc

    # 3. the attention gate
    if not a.skip_gate and attn_validated:
        # the tuple is already in the catalogue: nothing to run and nothing to add
        print("-- attention gate: this tuple is already validated in the catalogue; skipping",
              flush=True)
    elif not a.skip_gate and not attn_supported:
        print(f"-- attention gate: skipped (notification above: {attn_why})", flush=True)
    elif not a.skip_gate:
        print("-- attention gate: the production worker at this geometry, against an fp64 oracle",
              flush=True)
        if not attn_gate(py, env, spec_path, work, a.rows):
            to_candidates(spec_path, "attention gate failed",
                          "the production worker at this geometry did not match the fp64 "
                          "oracle; the catalogue was not touched")
            return 1
        print("   PASS", flush=True)

        # 4. the catalogue
        print("-- catalogue", flush=True)
        point = ", ".join(repr(v) for v in geo["point"])
        rc = run([sys.executable, ADD_POINT, "--op", "attn", "--point", point,
                  "--note", f"{geo['name']} (the attention gate, designs/attn/probe_dense.py)"],
                 env=env, check=False)
        if rc != 0:
            to_candidates(spec_path, "catalogue point could not be added",
                          f"catalogue-add-point.py --op attn exited {rc}")
            return rc

        # 5. the build through the validated path
        if not a.skip_export:
            print("-- rebuild through the validated path", flush=True)
            rc = run([py, EXPORT, "--specs", name, "--skip-bert", "--skip-diffusion"], env=env, check=False)
            if rc != 0:
                to_candidates(spec_path, "validated rebuild failed",
                              f"export-kernels.py --specs {name} exited {rc}")
                return rc

    # 6. validate the spec's geometry signature and export label
    print("-- validate the spec (utilities/register-spec.py)", flush=True)
    rc = run([sys.executable, REGISTER, name], env=env, check=False)
    if rc != 0:
        to_candidates(spec_path, "spec validation failed",
                      f"register-spec.py {name} exited {rc} (duplicate geometry or export "
                      "name -- link to the existing spec instead)")
        return rc

    print(f"\n{name} is ready.", flush=True)
    print("  commit: " + ", ".join(str(p.relative_to(ROOT)) for p in
                                   [spec_path, ROOT / "open_kernels" / "recipes" / "catalogue.py"]),
          flush=True)
    print("  the next `cmake --build` exports its kernel set automatically; no test files "
          "or registry entries need editing.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
