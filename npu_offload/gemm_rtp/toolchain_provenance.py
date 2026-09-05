# NpuEmbeddings -- toolchain provenance sidecar (T39, tasks/0106).
# SPDX-License-Identifier: MIT
#
# tasks/0102's audit found that kernel config hash `333c4d33` names TWO
# different instruction streams -- 0 `crrnd` instructions before the mlir-aie
# 1.3.4 -> 1.4.2 upgrade, 3 after -- and nothing in the build path could tell
# them apart; the audit had to fall back on .xclbin MTIME, in a repo whose
# trap 7c forbids exactly that, because no better source existed
# (research/notes/0009-toolchain-provenance.md).
#
# This writes a small toolchain.json NEXT TO each design.json the export
# tools already emit, capturing three strings that are already resident or a
# cheap call away: the mlir_aie package version, the Peano (llvm-aie) package
# version, and `git rev-parse HEAD` of C:\dev\mlir-aie. Nothing here is
# interpreted by the runtime -- it only reports it.
#
# DELIBERATE DEPARTURE from note 0009's proposal: the note also suggested
# tools/pack_npue.py copy a design's toolchain.json into the .npue container
# header. That is NOT done here. A .npue is packed from HuggingFace weights
# and never invokes mlir-aie at all, so a design's toolchain is not a
# property of the container -- the relationship between designs and
# containers is many-to-many in both directions (tasks/0104 gave MiniLM and
# bge-small different designs at the same geometry). Baking a design's build
# string into a container would assert a 1:1 relationship that does not
# exist, the same mistake the datapath field would have been if it had been
# put in the container instead of design.json. toolchain.json lives ONLY next
# to design.json, in the artifacts directory.
#
# Never fails a build: an export must not die because provenance couldn't be
# established. A field that can't be read is recorded as "unavailable", never
# guessed at and never a raised exception.

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MLIR_AIE_ROOT = Path(r"C:\dev\mlir-aie")


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:
        return "unavailable"


def _git_head(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "unavailable"
        head = out.stdout.strip()
        return head if head else "unavailable"
    except Exception:
        return "unavailable"


def write_toolchain_json(out_dir: Path,
                         mlir_aie_root: Path = MLIR_AIE_ROOT) -> dict:
    """Write toolchain.json next to design.json in `out_dir`.

    Three strings, all best-effort: a failure to read any one of them
    degrades that field to "unavailable" rather than raising, because
    provenance is not worth failing an export over.
    """
    info = {
        "mlir_aie_version": _pkg_version("mlir_aie"),
        "peano_version": _pkg_version("llvm-aie"),
        "mlir_aie_git_head": _git_head(mlir_aie_root),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "toolchain.json").write_text(json.dumps(info, indent=2),
                                            encoding="utf-8")
    return info
