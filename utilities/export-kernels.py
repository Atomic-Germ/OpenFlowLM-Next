#!/usr/bin/env python3
"""Build every open NPU kernel xclbin set so `cmake --install` ships a full
distribution. Driven by the `export_kernels` CMake target (FLM_BUILD_KERNELS=ON).
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
(Peano) wheel support 3.11, so one venv serves both families. The export is
Linux-only (it needs the XRT/Peano toolchain and, for open_npue, the NPU), so
the CMake target that drives this script is guarded to non-Windows builds.
"""
from __future__ import annotations

import argparse
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


def ensure_venv() -> Path:
    py = venv_python()
    if py.is_file():
        return py
    print("-- creating ironvenv (mlir-aie + Peano toolchain)", flush=True)
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", "--python", PYTHON, str(VENV)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(py), "-r", str(REQS)], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
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


def export_bert_sets(py: Path, force: bool) -> int:
    """open_npue BERT design sets (needs pyxrt + NPU). Returns 0 on success."""
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
    specs = spec_list(a.specs)
    if not specs:
        print("FATAL: no kernel specs found", file=sys.stderr)
        return 1

    n_failed = export_open_kernels(py, specs, a.force)
    n_failed += export_bert_sets(py, a.force)

    print(flush=True)
    if n_failed:
        print(f"kernel export failed: {n_failed} item(s)", flush=True)
        return 1
    print(f"kernel export ok: {len(specs)} open_kernels spec(s) + BERT design sets", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
