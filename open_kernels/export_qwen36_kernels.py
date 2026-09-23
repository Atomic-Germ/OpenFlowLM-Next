r"""Build the open kernel set a Qwen3.6-MoE model's engine loads, from source,
with the manifest.json the engine reads beside it.

    source ~/ironenv142/bin/activate          # mlir-aie 1.4.2 + Peano, xclbinutil/aiebu-asm on PATH
    python open_kernels/export_qwen36_kernels.py [--model-dir DIR | --spec FILE] [--out DIR]
                                                 [--only lx0,ln] [--no-build] [--force] [--check DIR]

The compiled kernels are NOT checked in (see .gitignore): this script is how
they come to exist. It derives the ModelSpec (recipes/spec.py) from the
model's config.json (default: the checked-in 27B spec), regenerates the
whole-layer designs' kernel TUs for it (designs/layer_x/gen_kernels.py), runs
build_design.py once per kernel set the recipe names (recipes/qwen36moe.py
`builds`) with OPEN_KERNELS_SPEC pointing at that spec, and copies
final.xclbin + insts.bin into <out>/<name>/ -- by default
src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/, which is where
src/open_qwen36/engine.cpp looks for them (and what `install(DIRECTORY
xclbins ...)` ships in the package). Beside them:

  manifest.json    everything the engine reads (recipes/manifest.py): layouts, contexts,
                   kernels, per-layer programs, packing plan, the spec and its hashes
  toolchain.json   the mlir-aie / Peano versions, the git commit of this tree and the
                   sha256 of every file, so a binary can always be traced to its source

The build cache (OPEN-BUILD-CACHE): the manifest's `build_key` hashes the
recipe sources, every kernel source the designs include, the spec and the
quant format. When <out>/manifest.json already carries the key this run
computes, nothing is rebuilt (--force overrides); any change to those inputs
rebuilds.

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
XCLBINS = REPO / "src" / "xclbins"
DEFAULT_MODEL = "Qwen3.6-35B-A3B-NPU2"
sys.path.insert(0, str(HERE))
from recipes.cache import build_key  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes.load import default_spec, load_spec, spec_from_model_dir  # noqa: E402
from recipes.manifest import dumps, manifest  # noqa: E402

# knobs that must NOT leak in from the caller's shell
CLEAR = ("LX_PART", "LX_STOP", "AX_PART", "UX_PART", "LMHEAD_N", "LMHEAD_CORES", "OPEN_KERNELS_SPEC")
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


def build(name: str, sets: dict, spec_file: Path) -> Path:
    src, out, knobs = DESIGNS / sets[name]["design"], DESIGNS / sets[name]["build_dir"], sets[name]["env"]
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update(knobs)
    env["OPEN_KERNELS_SPEC"] = str(spec_file)
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
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--model-dir", help="derive the spec from this model's config.json (+ tokenizer.json)")
    g.add_argument("--spec", help="a ModelSpec JSON (default: recipes/specs/qwen36-35b-a3b.json)")
    ap.add_argument("--out", default=None,
                    help="destination (default src/xclbins/<model name>/open_kernels)")
    ap.add_argument("--only", default=None, help="comma-separated subset of the recipe's kernel sets")
    ap.add_argument("--no-build", action="store_true", help="copy/hash the existing build dirs without rebuilding")
    ap.add_argument("--force", action="store_true", help="rebuild even when <out>/manifest.json has this build key")
    ap.add_argument("--max-ctx", type=int, default=4096, help="the manifest's default context capacity")
    ap.add_argument("--check", metavar="DIR",
                    help="a previous export (or a shipped xclbins/<model>/open_kernels dir) to compare "
                         "against; non-zero exit on any difference beyond the per-build UUID/timestamp stamps")
    a = ap.parse_args()

    # ---- the spec, its recipe, and the kernel sets that recipe names
    if a.model_dir:
        spec = spec_from_model_dir(Path(a.model_dir))
    elif a.spec:
        spec = load_spec(Path(a.spec))
    else:
        spec = default_spec()
    F = for_spec(spec)
    F.recipe(spec)                                     # refuses a spec outside the validated points
    sets = F.builds(spec)
    names = [n.strip() for n in a.only.split(",") if n.strip()] if a.only else list(sets)
    bad = [n for n in names if n not in sets]
    if bad:
        sys.exit(f"unknown set(s) {bad}; the recipe builds {list(sets)}")
    key = build_key(spec)
    out_root = Path(a.out).resolve() if a.out else (XCLBINS / spec.extra.get("model", DEFAULT_MODEL) / "open_kernels")
    out_root.mkdir(parents=True, exist_ok=True)
    spec_file = out_root / "spec.json"
    spec_file.write_text(spec.to_json(), encoding="utf-8", newline="\n")
    print(f"spec {spec.spec_hash()[:19]} ({spec.extra.get('model', spec.family)}), build key {key[:19]}")

    # ---- the cache: an export that already carries this key is current
    mj = out_root / "manifest.json"
    if not a.force and not a.no_build and mj.is_file():
        try:
            old = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        have = all((out_root / n / f).is_file() for n in names for f in FILES)
        if old.get("build_key") == key and have:
            print(f"up to date: {mj} already has build key {key[:19]} and every file; --force rebuilds")
            a.no_build = True

    # ---- the whole-layer designs' kernel TUs for this spec, then the builds
    if not a.no_build:
        os.environ["OPEN_KERNELS_SPEC"] = str(spec_file)
        gen_dir = HERE / F.GEN_KERNELS
        import importlib.util
        gspec = importlib.util.spec_from_file_location("gen_kernels", gen_dir)
        gen = importlib.util.module_from_spec(gspec)
        gspec.loader.exec_module(gen)
        gen.generate(F.recipe(spec))
    hashes: dict[str, str] = {}
    for n in names:
        bdir = DESIGNS / sets[n]["build_dir"] if a.no_build else build(n, sets, spec_file)
        dst = out_root / n
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            srcf = bdir / f
            if not srcf.is_file():
                sys.exit(f"[{n}] {srcf} missing after build")
            if srcf.resolve() != (dst / f).resolve():
                shutil.copyfile(srcf, dst / f)
            k = f"{n}/{f}"
            hashes[k] = sha256(dst / f)
            print(f"  {hashes[k]}  {k}  ({(dst / f).stat().st_size} B)")

    # ---- the manifest and the toolchain record
    m = manifest(spec, a.max_ctx, key)
    mj.write_text(dumps(m), encoding="utf-8", newline="\n")
    info = {
        "model": spec.extra.get("model", spec.family),
        "spec_hash": spec.spec_hash(),
        "build_key": key,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mlir_aie_version": pkg_version("mlir_aie"),
        "peano_version": pkg_version("llvm-aie"),
        "mlir_aie_git_head": git_head(Path(os.environ.get("MLIR_AIE_ROOT", Path.home() / "mlir-aie"))),
        "source_git_head": git_head(REPO),
        "sets": {n: {"design": (DESIGNS / sets[n]["design"]).relative_to(REPO).as_posix(), "env": sets[n]["env"]}
                 for n in names},
        "sha256": hashes,
    }
    # Keep what an earlier run recorded when this one did not build everything:
    # a subset rebuild merges its sets/hashes in, and --no-build (which may run
    # outside the toolchain venv) keeps the recorded versions instead of
    # overwriting them with "unavailable".
    tj = out_root / "toolchain.json"
    if tj.is_file() and (a.no_build or len(names) < len(sets)):
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
    print(f"-> {out_root}  (manifest.json + toolchain.json: mlir-aie {info['mlir_aie_version']}, "
          f"Peano {info['peano_version']})")

    if a.check:
        ref = Path(a.check)
        bad = 0
        for key_ in hashes:
            mine, theirs = out_root / key_, ref / key_
            if not theirs.is_file():
                print(f"MISSING  {key_}: not in {ref}")
                bad += 1
                continue
            x, y = mine.read_bytes(), theirs.read_bytes()
            if key_.endswith("insts.bin"):
                ok, note = (x == y), ("byte-identical" if x == y else f"differ ({len(x)} vs {len(y)} B)")
            else:
                ok, note = xclbin_equivalent(x, y)
            print(f"{'same    ' if ok else 'MISMATCH'} {key_}: {note}")
            bad += not ok
        rm = ref / "manifest.json"
        if rm.is_file():
            try:
                theirs = json.loads(rm.read_text(encoding="utf-8"))
                mine_ = json.loads(mj.read_text(encoding="utf-8"))
                for d in (theirs, mine_):
                    d.pop("build_key", None)
                ok = theirs == mine_
                print(f"{'same    ' if ok else 'MISMATCH'} manifest.json: {'equal apart from the build key' if ok else 'differs'}")
                bad += not ok
            except Exception as e:
                print(f"MISMATCH manifest.json: {e}")
                bad += 1
        if bad:
            print(f"CHECK FAILED: {bad} of {len(hashes)} files differ from {ref}")
            return 1
        print(f"CHECK OK: all {len(hashes)} files match {ref} (instruction streams byte-identical, "
              f"xclbins identical apart from build stamps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
