#!/usr/bin/env python3
"""Validate the catalogue points a spec needs, by RUNNING each op's fixture.

    python utilities/op-gate.py --spec llama32-3b          # every missing point
    python utilities/op-gate.py --spec qwen3-8b --op ln    # one op
    python utilities/op-gate.py --spec qwen3-8b --dry-run  # say what it would run

A spec needs a catalogue point for every op it drives, and the catalogue only
holds points a kernel has actually been RUN at. This tool closes that loop
without a human in it:

  1. run the family's recipe under catalogue.collecting(), so ONE pass lists
     every point the spec is missing (not just the first one that would raise);
  2. for each missing point, run that op's own fixture: make_test.py generates
     vectors and an fp64 reference, build_design.py compiles the design at
     exactly that geometry, the harness runs it on the NPU, compare.py scores
     it;
  3. only on a PASS, write the point into recipes/catalogue.py with
     catalogue-add-point.py (values() parameters AND the whole-tuple
     combination, which are separate writes).

The per-op knowledge -- which fixture flags, which build env, how the compare
step is addressed -- lives in GATES below, in code, instead of in a docstring
somewhere. An op with no GATE is not gated silently: it is reported as NOT
AUTOMATED with what is missing, and the caller decides.

This is step 2b of utilities/build-new-spec.py. Run it on its own when a build
fails on one off-grid width (the message names the op and the value).
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "open_kernels"
DESIGNS = OPEN / "designs"
ADD_POINT = ROOT / "utilities" / "catalogue-add-point.py"
EXPORT = ROOT / "utilities" / "export-kernels.py"

# Ops whose point this tool can validate today: op -> what the gate knows.
#   fixture: argv tail for designs/<dir>/make_test.py (run with the gate's env)
#   env:     the variables the fixture AND the design read the geometry from
#   tag:     the fixture's file stem (its outputs are <name>_<tag>.bin / run_<tag>.cfg)
#   design:  the design entry point, relative to open_kernels/
#   compare: argv tail for designs/<dir>/compare.py
GATES = {
    # y[N] = W[N,K] @ x[K] from pool-order q4 chunks. A new weight width (K) is
    # the point a new model size asks for most often: every projection is a K.
    # The fixture synthesizes W at any N/K and takes the build knobs, so the gate
    # uses the recipe's own rs / per_call rather than the design's defaults.
    "gemv_q4": {
        "dir": "gemv_q4",
        "design": "designs/gemv_q4/gemv_q4.py",
        "compare": [],
    },
    # The RMSNorm. Its point is a COMBINATION (width, groups) plus a values()
    # width, because a grouped width can only be validated as that group count.
    "ln": {
        "dir": "ln",
        "design": "designs/ln/ln.py",
        "compare": [],
    },
    # The q4 vocab head: logits[N] = W[N, K] @ x[K], N = vocab. A new HIDDEN is a
    # new K here, and the head is the one op a new size trips after attention.
    "lm_head_q4": {
        "dir": "lm_head_q4",
        "design": "designs/lm_head_q4/lm_head_q4.py",
        "compare": [],
    },
}

# Ops with no runnable fixture in this tree. Naming what is missing is the
# point: a new spec that needs one of these is a DESIGN task, not a script run.
NOT_AUTOMATED = {
    "attn": "handled by build-new-spec's own attention gate (designs/attn/probe_dense.py)",
    "lm_head_q8": "designs/lm_head_q8/make_test.py compiles K as a module constant (2048), so it "
                  "cannot be pointed at another K; K 4096 / 5120 were validated through the "
                  "family's whole-layer runs",
    "gemv_q8": "no per-K fixture (the OPEN-QUANT-Q8 procedure validates it against a converted "
               "container's own q8 values)",
    "gemv_q4_prep_f32": "validated through the whole-layer MoE block only (K is fixed at 512)",
    "deltanet": "designs/deltanet/make_test.py reads a captured state, not a geometry: no per-shape run",
    "short_conv": "no fixture; (3, 2048) came from an oflm serve run (OPEN-SHORT-CONV-KERNEL)",
    "router": "no per-geometry fixture",
    "moe": "no per-geometry fixture (whole-layer MoE block only)",
}


class GateError(RuntimeError):
    """A point that the gates could not validate (and that a human must look at)."""


def _run(cmd, env=None, cwd=None, check=True, quiet=False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("   +", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else str(ROOT))
    if check and p.returncode != 0:
        raise GateError(f"op-gate: FAILED: {' '.join(cmd)}")
    return p.returncode


def toolchain():
    """The venv python and the PATH the builds need (export-kernels.py owns both)."""
    spec = importlib.util.spec_from_file_location("_export_kernels", EXPORT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import os
    env = dict(os.environ)
    parts = [str(mod.llvm_aie_bin()) if mod.llvm_aie_bin() else "", str(mod.XRT_BIN), env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(p for p in parts if p)
    lib = Path(mod.XRT_ROOT) / "lib"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(lib), env.get("LD_LIBRARY_PATH", "")])
    # The fixtures import `open_kernels.q4_1_pack` / `gguf_pool` as packages (they
    # were written to run from the repo root), so the repo root is on the path.
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    return mod.ensure_venv(), env


def missing_points(spec):
    """Every catalogue point this spec needs and does not have, in one pass."""
    from recipes import catalogue as C
    from recipes.families import for_spec

    with C.collecting() as missing:
        for_spec(spec).recipe(spec)
    return missing


def _add_param(op: str, param: str, value, note: str) -> None:
    _run([sys.executable, ADD_POINT, "--op", op, "--param", param, "--value", repr(value),
          "--note", note], quiet=True)


def _add_combo(op: str, point: tuple, note: str) -> None:
    _run([sys.executable, ADD_POINT, "--op", op, "--point", ", ".join(repr(v) for v in point),
          "--note", note], quiet=True)


def _values_params(op: str, kw: dict) -> list[str]:
    """The kw keys that are enumerated values() parameters AND out of range.

    A rule-based param (multiple_of / any_positive) cannot be extended by a
    point -- it either passes or the geometry is wrong -- so only the enumerated
    ones are written.
    """
    from recipes.catalogue import CATALOGUE
    tpl = CATALOGUE[op]
    grouped = {k for c in tpl.combos for k in c.keys}
    out = []
    for k in kw:
        param = tpl.params.get(k)
        if k in grouped or param is None or param.values is None:
            continue
        if kw[k] not in param.values:
            out.append(k)
    return out


def _combo_keys(op: str) -> list[tuple[str, ...]]:
    from recipes.catalogue import CATALOGUE
    return [c.keys for c in CATALOGUE[op].combos]


def _gemv_names(k: int, n: int, per_call: int) -> tuple[str, str]:
    """The fixture's own (tag, build dir) -- imported, not re-derived.

    designs/gemv_q4/make_test.py owns the naming (its outputs are
    <name>_<tag>.bin and run_<tag>.cfg, referencing build_<build>/final.xclbin);
    a second copy of the rule here is exactly how a gate ends up running the
    harness on the wrong xclbin.
    """
    import importlib.util
    f = DESIGNS / "gemv_q4" / "make_test.py"
    for path in (str(ROOT), str(OPEN)):          # the fixture imports open_kernels.<pkg>
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("_gemv_make_test", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build_names("share_down", k_override=k, n_override=n, per_call=per_call)


def _cleanup(design: Path, tag: str, cc_before: set, keep: bool) -> None:
    """Remove what a gate run leaves behind in the design dir.

    build_design.py generates one TU per (per_call, chunks-per-band, rs) entry
    point -- for an off-grid width that is dozens of files the repo does not
    track -- and the fixture writes its vectors and cfg beside them. The build
    dir (build_<tag>) is left alone: it is git-ignored and it is the record of
    what was run.
    """
    if keep:
        return
    for f in set(design.glob("*.cc")) - cc_before:
        f.unlink()
    for pat in (f"run_{tag}.cfg", f"*_{tag}.bin", f"y_{tag}.bin"):
        for f in design.glob(pat):
            f.unlink()


def gate_gemv_q4(spec, kw: dict, py, env, harness, note: str, dry: bool) -> None:
    """Run designs/gemv_q4 at this K and add the point.

    The catalogue key here is K; rs / per_call are the geometry the recipe will
    build with. The FIXTURE's own element buffers are laid out for the standalone
    design, which is less economical than the dense design that ships (dx streams
    through gen_kernels), so a per_call the model builds with can still overflow
    the standalone design's L1. When it does, the gate retries at the next
    smaller per_call: the kernel under test -- gemv_q4.h at this K -- is the
    same, and the difference is the harness's buffering, not the model's.
    """
    k = int(kw["K"])
    rs = int(kw.get("rs", 2))
    per_band = rs * k // 256
    tries = [int(kw.get("per_call", 2)), 1]
    d = DESIGNS / "gemv_q4"
    cc_before = set(d.glob("*.cc"))
    last: GateError | None = None
    for pc in dict.fromkeys(tries):
        if per_band % pc:
            continue
        n = 8 * 32 * rs                  # one band per core, 8 cores
        tag, build_dir = _gemv_names(k, n, pc)
        e = dict(env, GEMV_N=str(n), GEMV_K=str(k), GEMV_RS=str(rs), GEMV_CORES="8", GEMV_PER_CALL=str(pc))
        cmd_fixture = [py, d / "make_test.py", "--region", "share_down", "--N", n, "--K", k, "--per-call", pc]
        cmd_build = [py, ROOT / "open_kernels" / "build_design.py", d / "gemv_q4.py", d / f"build_{build_dir}"]
        if dry:
            print(f"   would run: {' '.join(map(str, cmd_fixture))}")
            print(f"   would build: {' '.join(map(str, cmd_build))}")
            print(f"   would add:  gemv_q4 K={k}")
            return
        _run(cmd_fixture, env=e, cwd=d, quiet=pc != tries[0])
        try:
            _run(cmd_build, env=e)
        except GateError as exc:
            last = exc
            print(f"   per_call={pc} does not fit the fixture's L1 at K={k}; trying the next")
            _cleanup(d, tag, cc_before, keep=dry)     # a failed build leaves TUs behind too
            continue
        _run([harness, d / f"run_{tag}.cfg"], env=e, cwd=d, quiet=True)
        _run([py, d / "compare.py", tag], env=e, cwd=d)
        for param in _values_params("gemv_q4", kw):
            _add_param("gemv_q4", param, kw[param], note)
        _cleanup(d, tag, cc_before, keep=dry)
        print(f"   PASS gemv_q4 K={k} (rs={rs}, per_call={pc})")
        return
    raise last or GateError(f"gemv_q4: no per_call in {tries} divides {per_band} chunks per band at K={k}")


def gate_ln(spec, kw: dict, py, env, harness, note: str, dry: bool) -> None:
    """Run designs/ln at this width / group count and add the point."""
    width = int(kw["width"])
    groups = int(kw.get("groups", 1))
    tag = f"w{width}g{groups}"
    d = DESIGNS / "ln"
    cc_before = set(d.glob("*.cc"))
    e = dict(env, LN_N=str(width), LN_GROUPS=str(groups), LN_EPS=f"{spec.norm_eps:g}",
             LN_BUILD=f"build_{tag}")
    if dry:
        print(f"   would run: ln fixture at N={width} groups={groups}")
        print(f"   would add:  ln width={width}, ({width}, {groups})")
        return
    _run([py, d / "make_test.py"], env=e, cwd=d)
    _run([py, ROOT / "open_kernels" / "build_design.py", d / "ln.py", d / f"build_{tag}"], env=e)
    _run([harness, d / "run.cfg"], env=e, cwd=d, quiet=True)
    _run([py, d / "compare.py"], env=e, cwd=d)
    _add_param("ln", "width", width, note)
    for keys in _combo_keys("ln"):
        _add_combo("ln", tuple(kw.get(k, 1) for k in keys), note)
    for f in ("x.bin", "add.bin", "w.bin", "ref_y.bin", "ref_xn.bin", "y_y.bin", "y_xn.bin", "run.cfg"):
        (d / f).unlink(missing_ok=True)
    _cleanup(d, tag, cc_before, keep=dry)
    print(f"   PASS ln width={width} (groups={groups})")


def gate_lm_head_q4(spec, kw: dict, py, env, harness, note: str, dry: bool) -> None:
    """Run designs/lm_head_q4 at this K (the new hidden) and add the point.

    The fixture covers 64-row bands of the head's pool; `--bands 64` is one
    band per row-group of 64 rows (4096 rows), which exercises the band law and
    the uneven split over the 8 cores without streaming the whole vocab.
    """
    k = int(kw["K"])
    bands = 64
    tag = f"k{k}_b{bands}"
    d = DESIGNS / "lm_head_q4"
    cc_before = set(d.glob("*.cc"))
    e = dict(env, LMHEAD_N=str(bands * 64), LMHEAD_K=str(k))
    cmd_fixture = [py, d / "make_test.py", "--k", k, "--bands", bands]
    cmd_build = [py, ROOT / "open_kernels" / "build_design.py", d / "lm_head_q4.py", d / f"build_{tag}"]
    if dry:
        print(f"   would run: {' '.join(map(str, cmd_fixture))}")
        print(f"   would build: {' '.join(map(str, cmd_build))}")
        print(f"   would add:  lm_head_q4 K={k}")
        return
    _run(cmd_fixture, env=e, cwd=d)
    _run(cmd_build, env=e)
    _run([harness, d / f"run_{tag}.cfg"], env=e, cwd=d, quiet=True)
    _run([py, d / "compare.py", tag], env=e, cwd=d)
    for param in _values_params("lm_head_q4", kw):
        _add_param("lm_head_q4", param, kw[param], note)
    _cleanup(d, tag, cc_before, keep=dry)
    print(f"   PASS lm_head_q4 K={k}")


GATE_FN = {"gemv_q4": gate_gemv_q4, "ln": gate_ln, "lm_head_q4": gate_lm_head_q4}


def gate_points(spec, spec_path: Path, py, env, harness, *, dry: bool = False,
                only: str = "", skip: tuple[str, ...] = ("attn",)) -> list[tuple[str, dict]]:
    """Validate every missing point (or just `only`'s), returning what was gated.

    `skip` is for an op the CALLER gates itself: build-new-spec.py runs the
    attention gate (it builds the model's own worker, not a fixture), so it
    skips "attn" here.

    Raises GateError for an op with no runnable fixture, naming it -- never
    silently skipping a point the build would then fail on.
    """
    missing = missing_points(spec)
    if only:
        missing = [(op, kw) for op, kw in missing if op == only]
    done = []
    for op, kw in missing:
        if op in skip:
            continue
        if _already_valid(op, kw):
            # the recipe asks for one width from several projections (q / o / up / down),
            # so the same point arrives once per call site; the first gate added it.
            continue
        fn = GATE_FN.get(op)
        if fn is None:
            raise GateError(f"op-gate: {op} {kw} has no automated gate: "
                            f"{NOT_AUTOMATED.get(op, 'no fixture for this op')}")
        note = f"{spec_path.stem} (utilities/op-gate.py: the {op} fixture at this geometry)"
        print(f"-- gate {op} {kw}", flush=True)
        fn(spec, kw, py, env, harness, note, dry)
        done.append((op, kw))
        if not dry:
            _reload_catalogue()
    return done


def _reload_catalogue():
    import importlib
    from recipes import catalogue as C
    importlib.reload(C)


def _already_valid(op: str, kw: dict) -> bool:
    """True when every point this call site would write is already in the catalogue.

    Reads the module fresh each time: catalogue-add-point.py edits the source in
    a subprocess, so a cached module would answer for the pre-edit file.
    """
    _reload_catalogue()
    from recipes.catalogue import CATALOGUE
    tpl = CATALOGUE.get(op)
    if tpl is None:
        return False
    for k in _values_params(op, kw):
        return False
    for c in tpl.combos:
        if c.value(kw) not in c.points:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="spec stem (qwen3-8b) or path to the JSON")
    ap.add_argument("--op", default="", help="only this op (gemv_q4, ln, ...)")
    ap.add_argument("--work", default="", help="where the harness is built (default: build/op-gate)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(OPEN))
    from recipes.load import load_spec

    p = Path(a.spec)
    spec_path = p if p.is_file() else OPEN / "recipes" / "specs" / f"{a.spec.removesuffix('.json')}.json"
    if not spec_path.is_file():
        print(f"op-gate: no such spec: {a.spec}", file=sys.stderr)
        return 1
    spec = load_spec(spec_path)

    missing = missing_points(spec)
    if not missing:
        print(f"{spec_path.stem}: every catalogue point it needs is already validated")
        return 0
    print(f"{spec_path.stem} needs {len(missing)} point(s):")
    for op, kw in missing:
        state = "" if op in GATE_FN else f"   [{NOT_AUTOMATED.get(op, 'no fixture')}]"
        print(f"   {op}: {kw}{state}")

    py, env = toolchain()
    work = Path(a.work) if a.work else ROOT / "build" / "op-gate"
    work.mkdir(parents=True, exist_ok=True)
    exe = work / "harness" / "run_kernel"
    if not exe.is_file() and not a.dry_run:
        _run(["cmake", "-S", OPEN / "harness", "-B", work / "harness", "-DCMAKE_BUILD_TYPE=Release"], quiet=True)
        _run(["cmake", "--build", work / "harness", "-j", "8"], quiet=True)

    try:
        done = gate_points(spec, spec_path, py, env, exe, dry=a.dry_run, only=a.op)
    except GateError as exc:
        print(exc, file=sys.stderr)
        return 1

    if a.dry_run:
        return 0
    # catalogue-add-point.py edited the source in a subprocess; this process
    # holds the pre-edit module, so re-read it before claiming the points landed.
    _reload_catalogue()
    # only this tool's own ops: "attn" is the build-new-spec gate, and --op means
    # the caller asked about one op and the others are not this run's business.
    again = [(op, kw) for op, kw in missing_points(spec)
             if op != "attn" and (not a.op or op == a.op)]
    if again:
        print("op-gate: still missing after the gates: " + ", ".join(f"{o} {k}" for o, k in again),
              file=sys.stderr)
        return 1
    print(f"\n{spec_path.stem}: {len(done)} point(s) validated and written to recipes/catalogue.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
