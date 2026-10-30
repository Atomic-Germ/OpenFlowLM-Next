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

  * the diffusion family (FLUX.2 [klein] 4B) -- built by
    open_kernels/export_dit_kernels.py: the dit_gemm/dit_fa/dit_ew streams and the
    VAE's conv/ew ones, composed into one full ELF per resolution. One family, one
    command, and it ships like the rest: `oflm image` finds it under the xclbins
    root, so a user never builds it. Its model is a bundle rather than a q4nx
    container, which is why it is a third family here and not a spec.

Requirements (assumed present, or set up here):
  * XRT installed at /opt/xilinx/xrt, or wherever XILINX_XRT points
    (xclbinutil/aiebu-asm on PATH, pyxrt under <prefix>/python).
  * ironvenv/ created here from ironvenv-requirements.txt (mlir-aie + Peano).
  * third_party/mlir-aie cloned here (best-effort; only used for toolchain.json
    version metadata).

The venv is created for the same Python minor version as the interpreter
running this script, because the open_npue export needs pyxrt and pyxrt is
built for one specific Python. mlir-aie and the llvm-aie (Peano) wheels ship
for that version too, so one venv serves both families.

This SCRIPT is Linux-only -- its venv/path handling (`$XILINX_XRT`,
`bin/python`, `lib/python*/site-packages/...`) is POSIX-specific, so the CMake
target that drives it is guarded to non-Windows builds. The underlying build
it drives is not: `open_kernels/export_qwen36_kernels.py` and
`build_design.py` have no OS-specific code, mlir-aie and Peano both ship
win_amd64 wheels, and a native Windows toolchain (`iron_setup.py`, a
downloaded XRT SDK zip -- see mlir-aie's `docs/buildHostWinNative.md`, no WSL
and no source build required) has built and run open_kernels sets on
hardware (`.agents/skills/open-granite-kernels/SKILL.md`). `open_npue`'s
export additionally needs pyxrt and an installed NPU, which the Windows XRT
SDK also supplies (for its own pyxrt-compatible Python version).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO / "open_kernels" / "recipes" / "specs"
VENV = Path(os.environ.get("OFLM_VENV_DIR", str(REPO / "ironvenv")))
REQS = REPO / "ironvenv-requirements.txt"
EXPORT = REPO / "open_kernels" / "export_qwen36_kernels.py"

GEMM_RTP = REPO / "npu_offload" / "gemm_rtp"
BERT_EXPORT = GEMM_RTP / "export_gemm_rtp.py"
BERT_SPEC = GEMM_RTP / "families.json"
BERT_CHECK = GEMM_RTP / "check_design_sets.py"
XCLBINS = REPO / "src" / "xclbins"

# XILINX_XRT is honoured when set so a relocated XRT (the Nix builds combine
# XRT with the amdxdna plugin in their own prefix) supplies pyxrt too.
XRT_ROOT = Path(os.environ.get("XILINX_XRT") or "/opt/xilinx/xrt")
XRT_BIN = XRT_ROOT / "bin"
XRT_PY = XRT_ROOT / "python"

# Build the venv for the same Python minor version as the interpreter running
# this script when the IRON wheels ship for it (cp311..cp314 today): the
# open_npue export needs pyxrt, and pyxrt is built for one specific Python. A
# newer interpreter than that (3.15) has no wheel, so fall back to 3.11.
PY_MIN, PY_MAX = 11, 14
PY = f"{sys.version_info.major}.{sys.version_info.minor}"
PYTHON = PY if PY_MIN <= sys.version_info.minor <= PY_MAX else "3.11"


def venv_python() -> Path:
    return VENV / "bin" / "python3"


def llvm_aie_bin() -> Path | None:
    matches = sorted(VENV.glob("lib/python*/site-packages/llvm-aie/bin"))
    return matches[-1] if matches else None


def ensure_venv() -> Path:
    py = venv_python()
    # Nix builds provide a pre-assembled IRON venv with the correct wheels.
    # Skip the network/pip-based setup when requested.
    if os.environ.get("OFLM_SKIP_VENV_SETUP") == "1" and py.is_file():
        return py
    if py.is_file():
        # If the existing venv can import the IRON "aie" package, trust it
        # (e.g. the Nix dev shell materialized it). Otherwise fall through and
        # recreate it.
        try:
            subprocess.run([str(py), "-c", "import aie"], check=True, capture_output=True)
            return py
        except subprocess.CalledProcessError:
            pass
    print("-- creating ironvenv (mlir-aie + Peano toolchain)", flush=True)
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", "--python", PYTHON, str(VENV)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(py), "-r", str(REQS)], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        # Ensure the venv has a python3 symlink; the nix-provided venv only
        # guarantees bin/python3, and export scripts below call python3.
        if not (VENV / "bin" / "python3").exists() and (VENV / "bin" / "python").exists():
            os.symlink("python", VENV / "bin" / "python3")
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(REQS)], check=True)
    return py


def spec_list(names: str) -> list[Path]:
    if names:
        specs: list[Path] = []
        for name in names.split(","):
            name = name.strip()
            if not name:
                continue
            f = SPECS_DIR / name
            if not f.is_file():
                f = SPECS_DIR / (name + ".json")
            specs.append(f)
        return specs
    return sorted(SPECS_DIR.glob("*.json"))


def export_open_kernels(py: Path, specs: list[Path], force: bool, jobs: int) -> int:
    """open_qwen36 + dense families (compile-only, no NPU). Returns 0 on success.

    Specs are independent, so they run in parallel. The inner --jobs is divided
    among the concurrent specs so the total process count stays near `jobs`.
    """
    failed: list[str] = []

    def build_spec(spec: Path) -> tuple[str, bool]:
        name = spec.stem
        print(f"-- export {name}", flush=True)
        inner_jobs = max(1, jobs // max(1, len(specs)))
        cmd = [str(py), str(EXPORT), "--spec", str(spec), "--jobs", str(inner_jobs)]
        if force:
            cmd.append("--force")
        rc = subprocess.run(cmd).returncode
        return name, rc == 0

    if jobs > 1 and len(specs) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(specs))) as ex:
            futures = {ex.submit(build_spec, spec): spec for spec in specs}
            for fut in as_completed(futures):
                name, ok = fut.result()
                if ok:
                    print(f"   ok: {name}", flush=True)
                else:
                    failed.append(name)
                    print(f"   FAILED: {name}", flush=True)
    else:
        for spec in specs:
            name, ok = build_spec(spec)
            if ok:
                print(f"   ok: {name}", flush=True)
            else:
                failed.append(name)
                print(f"   FAILED: {name}", flush=True)

    return len(failed)


# The sizes the schedules carry, and the edits: both sides of a bundle must name the same
# ones (q4nx/open_diffusion.py's RESOLUTIONS/EDITS), and the layout hash is computed from
# them, so a mismatch is refused at load rather than running the wrong streams.
DIT_RESOLUTIONS = "512,1024"
DIT_EDITS = "512,1024"


def diffusion_families(py: Path) -> list[str]:
    """The families the exporter can build, asked of the exporter itself.

    A hard-coded list here would ship whichever families someone remembered (it shipped only
    the 4B for a day); `--list-families` is the exporter's own answer.
    """
    out = subprocess.run([str(py), str(REPO / "open_kernels" / "export_dit_kernels.py"),
                          "--list-families"], capture_output=True, text=True)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def export_diffusion_sets(py: Path, force: bool, jobs: int = 4,
                          families: list[str] | None = None) -> int:
    """The diffusion kernel sets (FLUX.2 [klein], one per family). Failures count.

    Each is built in `build/dit-kernels/<family>` and INSTALLED into
    `src/xclbins/<family>/open_kernels`: the build directory carries every stream's xclbin
    (hundreds of MiB), the installed form is the full ELFs and their manifest (tens of MiB),
    and that is what ships and what the engine reads. The exporter keeps its own cache, so a
    second run is seconds.
    """
    script = REPO / "open_kernels" / "export_dit_kernels.py"
    failed = 0
    for fam in (families or diffusion_families(py)):
        scratch = REPO / "build" / "dit-kernels" / fam
        shipped = REPO / "src" / "xclbins" / fam / "open_kernels"
        print(f"-- diffusion: {fam} (resolutions {DIT_RESOLUTIONS}, edits {DIT_EDITS})", flush=True)
        cmd = [str(py), str(script), "--family", fam, "--resolutions", DIT_RESOLUTIONS,
               "--edits", DIT_EDITS, "--out", str(scratch), "--jobs", str(jobs)]
        if force:
            cmd.append("--force")
        if subprocess.run(cmd).returncode:
            failed += 1
            continue
        # --install copies the runtime files only; src/ must not carry the build tree
        if subprocess.run([str(py), str(script), "--family", fam, "--resolutions", DIT_RESOLUTIONS,
                           "--edits", DIT_EDITS, "--out", str(scratch),
                           "--install", str(shipped)]).returncode:
            failed += 1
    return failed


def export_bert_sets(py: Path, force: bool, jobs: int = 1) -> int:
    """open_npue BERT design sets (needs pyxrt + NPU). Returns 0 on success."""
    spec = json.loads(BERT_SPEC.read_text(encoding="utf-8"))
    common = spec["common"]
    failed: list[str] = []

    def build_family(fam: dict) -> tuple[str, bool]:
        name = fam["name"]
        out = XCLBINS / name
        if (out / "gemm_rtp" / "design.json").is_file() and not force:
            print(f"-- bert {name}: already built (use --force to rebuild)", flush=True)
            return name, True
        print(f"-- bert {name} ({', '.join(fam['serves'])})", flush=True)
        cmd = [str(py), str(BERT_EXPORT), *fam["args"], *common, "--out", str(out)]
        rc = subprocess.run(cmd).returncode
        return name, rc == 0

    families = spec["families"]
    # families.json: "qkv and attn_out here are IDENTICAL to BERT-h768-bfp16's
    # -- neither depends on --gated-ffn or --intermediate -- so these two
    # families share 8 of their 16 cache markers and must never be built
    # concurrently." Each family purges matching directories in the same
    # ~/.npu/cache, so two families sharing design geometry (same --hidden)
    # would delete each other's in-progress or retained artifacts if built at
    # the same time. Serialize families within one hidden size; different
    # hidden sizes still build in parallel.
    def hidden_of(fam: dict) -> str:
        args = fam["args"]
        return str(args[args.index("--hidden") + 1]) if "--hidden" in args else str(fam["name"])

    groups: dict[str, list[dict]] = {}
    for fam in families:
        groups.setdefault(hidden_of(fam), []).append(fam)

    def build_group(group: list[dict]) -> list[tuple[str, bool]]:
        results = []
        for fam in group:
            name, ok = build_family(fam)
            results.append((name, ok))
            if ok:
                print(f"   ok: {name}", flush=True)
            else:
                failed.append(name)
                print(f"   FAILED: {name}", flush=True)
        return results

    if jobs > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(groups))) as ex:
            # one worker per hidden size; families inside a group build
            # sequentially
            futures = {ex.submit(build_group, group): gname
                       for gname, group in groups.items()}
            for fut in as_completed(futures):
                fut.result()
    else:
        for group in groups.values():
            build_group(group)

    # The built sets and the spec they were built from must agree.
    rc = subprocess.run([str(py), str(BERT_CHECK), "--xclbins", str(XCLBINS)]).returncode
    if rc != 0:
        failed.append("check_design_sets")
        print("   FAILED: check_design_sets (sets disagree with families.json)", flush=True)
    return len(failed)


def ensure_toolchain():
    """Put the tree's MLIR-AIE toolchain on PATH when the caller has not.

    The export runs from CMake (the `export_kernels` target) as well as by hand, and a plain
    `cmake --workflow --preset linux-default` has not sourced `utilities/env.sh`. The toolchain
    lives in the tree's `ironvenv`: `aiecc`/`aie-opt` in `ironvenv/bin`, and PEANO_INSTALL_DIR
    from the llvm-aie package beside it (Peano stays off PATH -- its clang would shadow the
    system one -- and aiecc is handed it through that variable).

    Returns None when aiecc is already available, "" when nothing could be found, else a short
    description of what was added.
    """
    if shutil.which("aiecc"):
        return None
    root = Path(__file__).resolve().parents[1]
    venv_bin = root / "ironvenv" / "bin"
    if not (venv_bin / "aiecc").is_file():
        return ""
    os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")
    if not os.environ.get("PEANO_INSTALL_DIR"):
        for cand in sorted((root / "ironvenv" / "lib").glob("python3*/site-packages/llvm-aie")):
            os.environ["PEANO_INSTALL_DIR"] = str(cand)
            break
    return f"{venv_bin} (tree ironvenv)"


def main() -> int:
    added = ensure_toolchain()
    if added == "":
        print("export-kernels: aiecc is not on PATH and there is no ironvenv/bin/aiecc in the "
              "tree; source utilities/env.sh to build kernels", flush=True)
        return 2
    if added:
        print(f"export-kernels: using {added} for aiecc", flush=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specs", default="", help="comma-separated open_kernels spec names (empty = all)")
    ap.add_argument("--skip-specs", default="",
                    help="comma-separated open_kernels spec names to skip (e.g. a recipe with a known gap)")
    ap.add_argument("--force", action="store_true", help="rebuild even when the build cache is current")
    ap.add_argument("--skip-bert", action="store_true",
                    help="skip open_npue BERT embedding sets (they need an NPU at build time)")
    ap.add_argument("--bert-only", action="store_true",
                    help="only build open_npue BERT embedding sets, skip open_kernels dense specs")
    ap.add_argument("--skip-diffusion", action="store_true",
                    help="skip the FLUX.2 [klein] diffusion kernel set")
    ap.add_argument("--diffusion-only", action="store_true",
                    help="only build the diffusion kernel sets")
    ap.add_argument("--diffusion-family", default="",
                    help="comma-separated diffusion families (default: every family the "
                         "exporter can build -- FLUX.2-klein-4B-NPU2, -9B-NPU2, ...)")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                   help="parallel kernel builds (default: 0 = auto-detect CPU count)")
    a = ap.parse_args()

    if a.jobs <= 0:
        a.jobs = max(1, os.cpu_count() or 4)

    if a.skip_bert and a.bert_only:
        print("FATAL: --skip-bert and --bert-only are mutually exclusive", file=sys.stderr)
        return 1
    if a.diffusion_only:
        # --diffusion-only is the narrow run: no dense specs, no BERT
        a.skip_bert = True

    skip_set = {n.strip() for n in a.skip_specs.split(",") if n.strip()}

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

    # ---- the three kernel families -------------------------------------------
    n_failed = 0
    specs = []
    if not a.bert_only and not a.diffusion_only:
        specs = [s for s in spec_list(a.specs) if s.stem not in skip_set]
        if not specs:
            print("FATAL: no kernel specs found", file=sys.stderr)
            return 1
        n_failed += export_open_kernels(py, specs, a.force, a.jobs)

    if not a.skip_bert and not a.diffusion_only:
        n_failed += export_bert_sets(py, a.force, a.jobs)

    if not a.skip_diffusion:
        fams = [f.strip() for f in a.diffusion_family.split(",") if f.strip()] or None
        n_failed += export_diffusion_sets(py, a.force, a.jobs, fams)

    print(flush=True)
    if n_failed:
        print(f"kernel export failed: {n_failed} item(s)", flush=True)
        return 1
    if a.bert_only:
        print("kernel export ok: BERT design sets", flush=True)
    elif a.diffusion_only:
        print("kernel export ok: diffusion sets", flush=True)
    else:
        suffix = " + BERT design sets" if not a.skip_bert else ""
        if not a.skip_diffusion:
            suffix += " + diffusion sets"
        print(f"kernel export ok: {len(specs)} open_kernels spec(s){suffix}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
