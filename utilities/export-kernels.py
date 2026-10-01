#!/usr/bin/env python3
"""Build every open NPU kernel xclbin set so `cmake --install` ships a full
distribution. Driven by the `export_kernels` CMake target (OFLM_BUILD_KERNELS=ON).
Safe to run repeatedly: each build's cache skips artifacts already exported on
the same toolchain.

Usage: utilities/export-kernels.py [--specs a,b,c] [--force]

Two kernel families share the one toolchain venv:

  * open_kernels (open_qwen36 + the dense families) -- compiled by
    open_kernels/export_qwen36_kernels.py, one command per recipe spec in
    open_kernels/recipes/specs/*.json. Compile-only; needs no NPU device.

  * open_npue (the BERT embedding design sets) -- built by
    npu_offload/gemm_rtp/export_gemm_rtp.py, one command per family in
    npu_offload/gemm_rtp/families.json, then verified against that file with
    npu_offload/gemm_rtp/check_design_sets.py. This one allocates NPU tensors
    (device="npu"), so it needs pyxrt and an installed NPU.

Requirements (assumed present, or set up here):
  * XRT installed at /opt/xilinx/xrt (xclbinutil/aiebu-asm on PATH, pyxrt).
  * ironvenv/ created here from ironvenv-requirements.txt (mlir-aie + Peano).
  * third_party/mlir-aie cloned here (best-effort; only used for toolchain.json
    version metadata).

Why Python 3.11: the installed XRT build ships pyxrt (its Python binding) for
3.11 only, and the open_npue export needs it. mlir-aie 1.4.2 and the llvm-aie
(Peano) wheel support 3.11, so one venv serves both families.

This SCRIPT is Linux-only -- its venv/path handling (`/opt/xilinx/xrt`,
`bin/python`, `lib/python*/site-packages/...`) is POSIX-specific, so the CMake
target that drives it is guarded to non-Windows builds. The underlying build
it drives is not: `open_kernels/export_qwen36_kernels.py` and
`build_design.py` have no OS-specific code, mlir-aie and Peano both ship
win_amd64 wheels, and a native Windows toolchain (`iron_setup.py`, a
downloaded XRT SDK zip -- see mlir-aie's `docs/buildHostWinNative.md`, no WSL
and no source build required) has built and run open_kernels sets on
hardware (`.opencode/skill/open-granite-kernels/SKILL.md`). `open_npue`'s
export additionally needs pyxrt and an installed NPU, which the Windows XRT
SDK also supplies (for its own pyxrt-compatible Python version).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO / "open_kernels" / "recipes" / "specs"
VENV = REPO / "ironvenv"
REQS = REPO / "ironvenv-requirements.txt"
EXPORT = REPO / "open_kernels" / "export_qwen36_kernels.py"

GEMM_RTP = REPO / "npu_offload" / "gemm_rtp"
BERT_EXPORT = GEMM_RTP / "export_gemm_rtp.py"
BERT_SPEC = GEMM_RTP / "families.json"
BERT_CHECK = GEMM_RTP / "check_design_sets.py"
XCLBINS = REPO / "src" / "xclbins"

# The open Whisper engine's kernel set. It has its own exporter rather than a
# recipes/specs entry (its set is a stream per encoder GEMM shape plus the
# separate FlashAttention context, not a per-family kernel list), so it needs
# its own call here -- otherwise `cmake --build` produces a distribution whose
# open Whisper engine has no kernels to load.
WHISPER_EXPORT = REPO / "open_kernels" / "export_whisper_kernels.py"
WHISPER_OUT = XCLBINS / "Whisper-V3-Turbo-NPU2" / "open_kernels"

XRT_ROOT = Path("/opt/xilinx/xrt")
XRT_BIN = XRT_ROOT / "bin"
XRT_PY = XRT_ROOT / "python"

# pyxrt is built for 3.11 by the installed XRT; pin the venv to match.
PYTHON = "3.11"


def venv_python() -> Path:
    return VENV / "bin" / "python"


def llvm_aie_bin() -> Path | None:
    matches = sorted(VENV.glob("lib/python*/site-packages/llvm-aie/bin"))
    return matches[-1] if matches else None


def venv_python_version(py: Path) -> str:
    out = subprocess.check_output(
        [str(py), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
    )
    return out.strip()


def ensure_venv() -> Path:
    py = venv_python()
    if py.is_file() and venv_python_version(py) != PYTHON:
        # XRT ships pyxrt for 3.11 only. A newer venv imports, mlir-aie then
        # picks CPUOnlyTensor, and the first BERT family (all-minilm) dies with
        # "Unsupported device: npu" -- a kernel error that is really a missing
        # binding. Recreate rather than build forty minutes of kernels that
        # cannot be verified on the device.
        got = venv_python_version(py)
        stale = VENV.with_name(f"ironvenv.py{got}")
        print(f"-- ironvenv is Python {got}, pyxrt needs {PYTHON}; moving it to {stale.name}",
              flush=True)
        if stale.exists():
            shutil.rmtree(stale)
        VENV.rename(stale)
        py = venv_python()
    if py.is_file():
        return py
    print(f"-- creating ironvenv (Python {PYTHON}, mlir-aie + Peano)", flush=True)
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", "--python", PYTHON, str(VENV)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(py), "-r", str(REQS)], check=True)
    else:
        py311 = shutil.which(f"python{PYTHON}")
        if not py311:
            print(f"FATAL: python{PYTHON} is not on PATH, and that is the only "
                  "interpreter XRT's pyxrt loads", file=sys.stderr)
            raise SystemExit(1)
        subprocess.run([py311, "-m", "venv", str(VENV)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(REQS)], check=True)
    return py


CATALOGUE = REPO / "src" / "model_list.json"
SPEC_GEN = REPO / "open_kernels" / "gen_catalogue_specs.py"
# Written next to the specs it describes: the catalogue digest and the count,
# so "are the derived specs current?" is answerable without a network fetch.
SPEC_STAMP = REPO / "open_kernels" / "recipes" / "specs" / ".stamp.json"


def _specs_current(specs: list[Path]) -> bool:
    """Whether the derived specs can be trusted, or must be rebuilt.

    Two independent conditions, and BOTH matter. "Every spec is newer than the
    catalogue" alone is not enough: deleting one file leaves the other 22 newer
    than the catalogue, so the check passes and the deleted model is never
    derived again -- an `oflm add` for it then matches no set and silently falls
    back to the closed engine, which is the exact symptom this work is about.
    So the count the catalogue implies is recorded in a stamp beside the specs,
    and a mismatch (or a missing stamp) regenerates.
    """
    if not specs or not CATALOGUE.exists():
        return False
    if specs[0].stat().st_mtime < CATALOGUE.stat().st_mtime:
        return False
    try:
        stamp = json.loads(SPEC_STAMP.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # The stamp is keyed on the catalogue's own content, so an edit to
    # model_list.json regenerates even if its mtime was not to move.
    digest = hashlib.sha256(CATALOGUE.read_bytes()).hexdigest()
    return stamp.get("catalogue_sha256") == digest and stamp.get("specs") == len(specs)


def ensure_specs(py: Path, force: bool) -> None:
    """Regenerate the per-(family, size) specs from the catalogue, when needed.

    The specs are DERIVED, not maintained: `gen_catalogue_specs.py` walks
    `model_list.json`, fetches each model's config.json, derives a ModelSpec and
    deduplicates by spec_hash. That is what replaced a hand-kept list of twelve
    files, which is how a model with a working recipe (Qwen3-0.6B) went unserved
    and how two specs ended up claiming one output directory.

    Regeneration is CONDITIONAL, and has to be. It reaches the network once per
    catalogue model, and `export_kernels` is an ALL target, so an unconditional
    regen would put 44 HTTPS round-trips in front of every incremental build --
    turning a no-op rebuild into a slow one, and a passing offline build into a
    failing one. So it runs when the catalogue has changed, when a spec is
    missing, or on --force; otherwise the existing derived specs stand. The
    staleness check is the catalogue's own mtime against the newest spec, which
    is the only input that changes what should be derived.
    """
    specs = sorted(SPECS_DIR.glob("*.json"))
    if not force and _specs_current(specs):
        print(f"-- specs: {len(specs)} derived spec(s) present and newer than the "
              f"catalogue; not regenerating")
        return
    print("-- specs: deriving one per (family, size) from the catalogue", flush=True)
    r = subprocess.run([str(py), str(SPEC_GEN), f"--out={SPECS_DIR}"])
    # Record what the catalogue said, so the next build can skip the fetch. Only
    # on success: a failed generation leaves a partial set, and stamping that
    # would make the next build trust it.
    if r.returncode == 0:
        try:
            n = len(sorted(SPECS_DIR.glob("*.json")))
            SPEC_STAMP.write_text(json.dumps(
                {"catalogue_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
                 "specs": n}, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    if r.returncode != 0:
        # A catalogue entry that is gated, or a model_type with no recipe, is
        # reported by the generator and does NOT fail it -- so a non-zero exit
        # here is a real failure (a collision, or the catalogue being unreadable).
        # Falling back to whatever specs exist keeps an incremental build working
        # rather than failing it on a metadata problem.
        print(f"WARNING: spec generation failed (exit {r.returncode}); using the "
              f"{len(sorted(SPECS_DIR.glob('*.json')))} spec(s) already present",
              file=sys.stderr)


def _derived_specs() -> list[str]:
    """Stems of the derived spec files, excluding the bookkeeping stamp.

    The stamp lives in the same directory because it describes those files, and
    a `*.json` glob that picked it up would hand the exporter a spec named
    `.stamp` -- which parses, is not a ModelSpec, and fails deep inside the
    design build instead of here.
    """
    if not SPECS_DIR.is_dir():
        return []
    return sorted(p.stem for p in SPECS_DIR.glob("*.json") if not p.name.startswith("."))


def spec_list(names: str) -> list[Path]:
    if names:
        specs: list[Path] = []
        missing: list[str] = []
        for name in names.split(","):
            name = name.strip()
            if not name:
                continue
            f = SPECS_DIR / name
            if not f.is_file():
                f = SPECS_DIR / (name + ".json")
            if f.is_file():
                specs.append(f)
            else:
                missing.append(name)
        if missing:
            # Named a spec that is not there. Specs are committed, so the fix is
            # usually to add the spec rather than to derive one; say what IS
            # there, rather than failing three lines later with a
            # FileNotFoundError from inside the exporter.
            have = _derived_specs()
            print(f"FATAL: no such spec: {', '.join(missing)}\n"
                  f"  Specs are committed under {SPECS_DIR}; add the spec there, or "
                  f"name one of the {len(have)} present:\n"
                  f"  {', '.join(have) if have else '(none yet)'}",
                  file=sys.stderr)
            raise SystemExit(1)
        return specs
    return [SPECS_DIR / f"{n}.json" for n in _derived_specs()]


def export_open_kernels(py: Path, specs: list[Path], force: bool) -> int:
    """open_qwen36 + dense families (compile-only, no NPU). Returns 0 on success."""
    failed: list[str] = []
    for spec in specs:
        name = spec.stem
        print(f"-- export {name}", flush=True)
        cmd = [str(py), str(EXPORT), "--spec", str(spec)]
        if force:
            cmd.append("--force")
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
            print(f"   FAILED: {name}", flush=True)
        else:
            print(f"   ok: {name}", flush=True)
    return len(failed)


def pyxrt_imports(py: Path) -> bool:
    env = os.environ.copy()
    probe = subprocess.run(
        [str(py), "-c", "import pyxrt"],
        env=env,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def export_bert_sets(py: Path, force: bool) -> int:
    """open_npue BERT design sets (needs pyxrt + NPU). Returns 0 on success."""
    if not pyxrt_imports(py):
        print("FATAL: pyxrt did not import. The BERT sets (all-minilm first) allocate "
              "NPU tensors, and iron falls back to a CPU tensor -- then fails with "
              f"'Unsupported device: npu' -- unless this is Python {PYTHON} and "
              f"{XRT_PY} is on PYTHONPATH. The installed XRT build ships pyxrt for "
              f"{PYTHON} only.", file=sys.stderr, flush=True)
        return 1
    spec = json.loads(BERT_SPEC.read_text(encoding="utf-8"))
    common = spec["common"]
    failed: list[str] = []
    for fam in spec["families"]:
        name = fam["name"]
        out = XCLBINS / name
        if (out / "gemm_rtp" / "design.json").is_file() and not force:
            print(f"-- bert {name}: already built (use --force to rebuild)", flush=True)
            continue
        print(f"-- bert {name} ({', '.join(fam['serves'])})", flush=True)
        cmd = [str(py), str(BERT_EXPORT), *fam["args"], *common, "--out", str(out)]
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
            print(f"   FAILED: {name}", flush=True)
        else:
            print(f"   ok: {name}", flush=True)

    # The built sets and the spec they were built from must agree.
    rc = subprocess.run([str(py), str(BERT_CHECK), "--xclbins", str(XCLBINS)]).returncode
    if rc != 0:
        failed.append("check_design_sets")
        print("   FAILED: check_design_sets (sets disagree with families.json)", flush=True)
    return len(failed)


def export_whisper_set(py: Path, force: bool) -> int:
    """open Whisper engine kernel set (needs pyxrt + NPU). Returns 0 on success."""
    if (WHISPER_OUT / "whisper_kernels.json").is_file() and not force:
        print("-- whisper: already built (use --force to rebuild)", flush=True)
        return 0
    print("-- whisper (Whisper-V3-Turbo-NPU2)", flush=True)
    cmd = [str(py), str(WHISPER_EXPORT), "--out", str(WHISPER_OUT)]
    if force:
        cmd.append("--force")
    if subprocess.run(cmd).returncode != 0:
        print("   FAILED: whisper", flush=True)
        return 1
    print("   ok: whisper", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specs", default="", help="comma-separated open_kernels spec names (empty = all)")
    ap.add_argument("--force", action="store_true", help="rebuild even when the build cache is current")
    a = ap.parse_args()

    py = ensure_venv()

    # ---- toolchain on PATH: Peano (llvm-aie wheel) first, then XRT ----------
    aie_bin = llvm_aie_bin()
    path_parts = []
    if aie_bin:
        path_parts.append(str(aie_bin))
    path_parts.append(str(XRT_BIN))
    path_parts.append(os.environ.get("PATH", ""))
    os.environ["PATH"] = os.pathsep.join(path_parts)

    # pyxrt (XRT's Python binding) lives beside the runtime, not in the venv.
    if XRT_PY.is_dir():
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(XRT_PY)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        )

    for tool in ("clang", "xclbinutil", "aiebu-asm"):
        if not shutil.which(tool):
            print(f"FATAL: {tool} not on PATH (Peano/XRT)", file=sys.stderr)
            return 1

    # ---- best-effort clone of third_party/mlir-aie (toolchain.json metadata)
    mlir_aie = REPO / "third_party" / "mlir-aie"
    if not mlir_aie.is_dir():
        print("-- cloning third_party/mlir-aie (best-effort)", flush=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Xilinx/mlir-aie", str(mlir_aie)],
                       capture_output=True)
    if mlir_aie.is_dir():
        os.environ["MLIR_AIE_ROOT"] = str(mlir_aie)

    # ---- the two kernel families ---------------------------------------------
    # The specs are derived from the catalogue, so make sure they exist and are
    # current before reading the list of them.
    if not a.specs:
        ensure_specs(py, a.force)
    specs = spec_list(a.specs)
    if not specs:
        print("FATAL: no kernel specs found", file=sys.stderr)
        return 1

    n_failed = export_open_kernels(py, specs, a.force)
    n_failed += export_whisper_set(py, a.force)
    n_failed += export_bert_sets(py, a.force)

    print(flush=True)
    if n_failed:
        print(f"kernel export failed: {n_failed} item(s)", flush=True)
        return 1
    print(f"kernel export ok: {len(specs)} open_kernels spec(s) + whisper + BERT design sets",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
