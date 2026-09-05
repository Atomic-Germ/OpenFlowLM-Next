r"""Build the six open kernel sets the Qwen3.6-MoE engine loads, from source.

    source ~/ironenv142/bin/activate          # mlir-aie 1.4.2 + Peano, xclbinutil/aiebu-asm on PATH
    python open_kernels/export_qwen36_kernels.py [--out DIR] [--only lx0,ln] [--no-build] [--check FILE]

The compiled kernels are NOT checked in (see .gitignore): this script is how
they come to exist. It runs build_design.py once per set with the design's
compile-time knobs, then copies final.xclbin + insts.bin into
<out>/<name>/ -- by default src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/,
which is where src/open_qwen36/engine.cpp looks for them (and what
`install(DIRECTORY xclbins ...)` ships in the package). toolchain.json beside
them records the mlir-aie / Peano versions, the git commit of this tree and
the sha256 of every file, so a binary can always be traced to its source.

--check DIR compares the fresh build against a previous one (a directory with
the same <name>/final.xclbin, <name>/insts.bin layout) and exits non-zero on
any real difference: the "is the source really the source?" test. insts.bin
must be byte-identical. final.xclbin is compared after masking the fields
xclbinutil stamps per build -- the axlf header's unique id, timestamp and
UUID, the PDI UUID inside the AIE_PARTITION section, and the JSON
"XCLBIN_MIRROR_DATA" tail that repeats them -- because every other byte (the
AIE core ELFs, the CDOs, the metadata sections) is expected to match. A
rebuild on the same toolchain here differed in exactly those ~80 bytes per
xclbin and nowhere else (see src/open_qwen36/README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # open_kernels/
REPO = HERE.parent
DESIGNS = HERE / "designs"
DEFAULT_OUT = REPO / "src" / "xclbins" / "Qwen3.6-35B-A3B-NPU2" / "open_kernels"

# name -> (design source, build dir, compile-time environment)
# The build dirs are the ones the batch harness (model/make_decode.py) and the
# per-design make_test scripts already name, so a run of this script also
# leaves every design testable in place.
SETS = {
    "lx0":        (DESIGNS / "layer_x" / "lx.py",              DESIGNS / "layer_x" / "build_lx0",     {"LX_PART": "0"}),
    "lx1":        (DESIGNS / "layer_x" / "lx.py",              DESIGNS / "layer_x" / "build_lx1",     {"LX_PART": "1"}),
    "ax0":        (DESIGNS / "layer_x" / "ax.py",              DESIGNS / "layer_x" / "build_ax0",     {"AX_PART": "0"}),
    "ax1":        (DESIGNS / "layer_x" / "ax.py",              DESIGNS / "layer_x" / "build_ax1",     {"AX_PART": "1"}),
    "ln":         (DESIGNS / "ln" / "ln.py",                   DESIGNS / "ln" / "build",              {}),
    "lm_head_q8": (DESIGNS / "lm_head_q8" / "lm_head_q8.py",   DESIGNS / "lm_head_q8" / "build_full", {"LMHEAD_N": "248320", "LMHEAD_CORES": "8"}),
}
# knobs that must NOT leak in from the caller's shell
CLEAR = ("LX_PART", "LX_STOP", "AX_PART", "LMHEAD_N", "LMHEAD_CORES")
FILES = ("final.xclbin", "insts.bin")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def xclbin_volatile_ranges(x: bytes) -> list[tuple[int, int]]:
    """Byte ranges of an xclbin that legitimately change from build to build.

    axlf layout (xrt/include/xclbin.h): m_uniqueId @296, header.m_timeStamp
    @312, header.uuid @416, m_numSections @448, section headers @456 (40 B
    each: kind u32, name[16], pad, offset u64, size u64). Kind 32 is
    AIE_PARTITION: its aie_pdi array (array_offset @+120, 96 B entries) holds a
    per-build UUID per PDI, and each PDI image is a bootgen boot image whose
    image header carries a per-build unique id (@+0x98) and the header checksum
    that covers it (@+0xCC). The bytes after the last section are xclbinutil's
    XCLBIN_MIRROR_DATA JSON, a copy of the header. Nothing else -- the CDOs,
    the AIE core ELFs, the metadata sections -- may differ."""
    if x[:7] != b"xclbin2":
        return []
    ranges = [(296, 304), (312, 320), (416, 432)]
    n = struct.unpack_from("<I", x, 448)[0]
    end = 456 + n * 40
    for i in range(n):
        off = 456 + i * 40
        kind = struct.unpack_from("<I", x, off)[0]
        so, sz = struct.unpack_from("<QQ", x, off + 24)
        end = max(end, so + sz)
        if kind == 32:
            n_pdi, off_pdi = struct.unpack_from("<II", x, so + 120)
            for k in range(n_pdi):
                e = so + off_pdi + k * 96
                ranges.append((e, e + 16))                               # aie_pdi.uuid
                img_size, img_off = struct.unpack_from("<II", x, e + 16)  # aie_pdi.pdi_image
                if img_size >= 0xD0:
                    ranges.append((so + img_off + 0x98, so + img_off + 0x9C))   # image header unique id
                    ranges.append((so + img_off + 0xCC, so + img_off + 0xD0))   # image header checksum
    ranges.append((end, len(x)))
    return ranges


def xclbin_equivalent(a: bytes, b: bytes) -> tuple[bool, str]:
    """True when a and b differ only in build stamps (UUIDs / timestamps / ids)."""
    if len(a) != len(b):
        return False, f"sizes differ ({len(a)} vs {len(b)} B)"
    ranges = xclbin_volatile_ranges(a)
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if not diff:
        return True, "byte-identical"
    stray = [i for i in diff if not any(lo <= i < hi for lo, hi in ranges)]
    if stray:
        return False, f"{len(stray)} bytes differ outside the build-stamp fields (first @{stray[0]})"
    return True, f"{len(diff)} bytes differ, all build stamps (uuid/timestamp/unique id)"


def pkg_version(name: str) -> str:
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:
        return "unavailable"


def git_head(root: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def build(name: str) -> Path:
    src, out, knobs = SETS[name]
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update(knobs)
    t0 = time.time()
    knob_str = " ".join(f"{k}={v}" for k, v in knobs.items())
    print(f"[{name}] {knob_str} python build_design.py {src.relative_to(HERE).as_posix()} "
          f"{out.relative_to(HERE).as_posix()}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / "build_design.py"), str(src), str(out)], env=env, cwd=str(HERE))
    if r.returncode != 0:
        sys.exit(f"[{name}] build FAILED ({r.returncode})")
    print(f"[{name}] built in {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"destination (default {DEFAULT_OUT.relative_to(REPO).as_posix()})")
    ap.add_argument("--only", default=",".join(SETS), help="comma-separated subset of " + ",".join(SETS))
    ap.add_argument("--no-build", action="store_true", help="copy/hash the existing build dirs without rebuilding")
    ap.add_argument("--check", metavar="DIR",
                    help="a previous export (or a shipped xclbins/<model>/open_kernels dir) to compare "
                         "against; non-zero exit on any difference beyond the per-build UUID/timestamp stamps")
    a = ap.parse_args()
    names = [n.strip() for n in a.only.split(",") if n.strip()]
    bad = [n for n in names if n not in SETS]
    if bad:
        sys.exit(f"unknown set(s) {bad}; choose from {list(SETS)}")

    out_root = Path(a.out).resolve()
    hashes: dict[str, str] = {}
    for n in names:
        bdir = SETS[n][1] if a.no_build else build(n)
        dst = out_root / n
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            srcf = bdir / f
            if not srcf.is_file():
                sys.exit(f"[{n}] {srcf} missing after build")
            shutil.copyfile(srcf, dst / f)
            key = f"{n}/{f}"
            hashes[key] = sha256(dst / f)
            print(f"  {hashes[key]}  {key}  ({(dst / f).stat().st_size} B)")

    info = {
        "model": "Qwen3.6-35B-A3B-NPU2",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mlir_aie_version": pkg_version("mlir_aie"),
        "peano_version": pkg_version("llvm-aie"),
        "mlir_aie_git_head": git_head(Path(os.environ.get("MLIR_AIE_ROOT", Path.home() / "mlir-aie"))),
        "source_git_head": git_head(REPO),
        "sets": {n: {"design": SETS[n][0].relative_to(REPO).as_posix(), "env": SETS[n][2]} for n in names},
        "sha256": hashes,
    }
    # Keep what an earlier run recorded when this one did not build everything:
    # a subset rebuild merges its sets/hashes in, and --no-build (which may run
    # outside the toolchain venv) keeps the recorded versions instead of
    # overwriting them with "unavailable".
    tj = out_root / "toolchain.json"
    if tj.is_file() and (a.no_build or len(names) < len(SETS)):
        try:
            old = json.loads(tj.read_text(encoding="utf-8"))
            for k in ("sets", "sha256"):
                merged = dict(old.get(k, {}))
                merged.update(info[k])
                info[k] = merged
            if a.no_build:
                for k in ("built", "mlir_aie_version", "peano_version", "mlir_aie_git_head", "source_git_head"):
                    info[k] = old.get(k, info[k])
        except Exception:
            pass
    tj.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out_root}  (toolchain.json: mlir-aie {info['mlir_aie_version']}, Peano {info['peano_version']})")

    if a.check:
        ref = Path(a.check)
        bad = 0
        for key in hashes:
            mine, theirs = out_root / key, ref / key
            if not theirs.is_file():
                print(f"MISSING  {key}: not in {ref}")
                bad += 1
                continue
            x, y = mine.read_bytes(), theirs.read_bytes()
            if key.endswith("insts.bin"):
                ok, note = (x == y), ("byte-identical" if x == y else f"differ ({len(x)} vs {len(y)} B)")
            else:
                ok, note = xclbin_equivalent(x, y)
            print(f"{'same    ' if ok else 'MISMATCH'} {key}: {note}")
            bad += not ok
        if bad:
            print(f"CHECK FAILED: {bad} of {len(hashes)} files differ from {ref}")
            return 1
        print(f"CHECK OK: all {len(hashes)} files match {ref} (instruction streams byte-identical, "
              f"xclbins identical apart from build stamps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
