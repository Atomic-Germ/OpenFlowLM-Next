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
import concurrent.futures
import contextlib
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
sys.path.insert(0, str(HERE))
from recipes.cache import build_key  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes.load import default_spec, load_spec, spec_from_model_dir  # noqa: E402
from recipes.manifest import dumps, manifest  # noqa: E402

# knobs that must NOT leak in from the caller's shell
CLEAR = ("LX_PART", "LX_STOP", "AX_PART", "UX_PART", "LMHEAD_N", "LMHEAD_CORES", "OPEN_KERNELS_SPEC")
FILES = ("final.xclbin", "insts.bin")


# ————————————————————————————————————————————————————————————————
# Content-addressed build cache
# ————————————————————————————————————————————————————————————————
# 50 build_dir paths are shared between specs: attn_block/build_s256 is wanted
# by 7 specs, and the last full export ran 340 builds for 130 unique designs.
# The shared designs (attn_block, gemm_q4_prefill, lm_head_q4, ln) have no
# gen_kernels.py and do not read OPEN_KERNELS_SPEC, so their output is a
# function of (design sources, compile-time knobs) alone -- the build_dir name
# already encodes the spec-dependent parts. Hash those, write the hash next to
# the output, and the next spec to want the design reuses it instead of
# rebuilding an identical xclbin.
#
# The key covers the design's source files, so a design that DOES generate
# per-spec sources (dense, layer_x) changes its key when the spec changes and
# is rebuilt -- which is correct, and is also why Task 30 isolates those
# sources per spec in the first place.

KEY_FILE = ".build_key"


def build_content_key(design: Path, out: Path, env: dict) -> str:
    """Hash of everything that determines the build's output.

    Covers the design's source files (excluding build outputs and caches) and
    the compile-time knobs. Two builds with the same key produce the same
    xclbin, so the second one can reuse the first one's output.
    """
    h = hashlib.sha256()
    h.update(str(out.relative_to(DESIGNS)).encode())
    h.update(b"\0")
    for f in sorted(design.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(design)
        top = rel.parts[0] if rel.parts else ""
        if top.startswith("build") or top in ("__pycache__", ".staging", ".locks"):
            continue
        if f.suffix in (".xclbin", ".bin", ".elf", ".o", ".prj"):
            continue
        h.update(rel.as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    for k in sorted(env):
        h.update(f"{k}={env[k]}".encode())
        h.update(b"\0")
    return h.hexdigest()


# ————————————————————————————————————————————————————————————————
# Per-design build lock
# ————————————————————————————————————————————————————————————————
# 50 build_dir paths are shared between specs (e.g. lm_head_q4/build_151936 is
# wanted by qwen25-3b, qwen3-4b and qwen3-8b). build_design.py wipes its work
# dir (final.prj) before compiling, so two specs building the same design at
# the same time delete each other's objects and fail with "unable to rename
# temporary ... .o.tmp". The export runs specs in parallel, so the collision is
# the default, not the exception. One lock per build_dir makes the shared
# designs serialize while everything else stays parallel.

LOCKS = DESIGNS / ".locks"


@contextlib.contextmanager
def design_lock(out: Path):
    """Hold an exclusive cross-process lock for one build directory."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    # The build dir is designs/<design>/<build_dir>; flatten it for the lock name.
    key = str(out.relative_to(DESIGNS)).replace("/", "__").replace("\\", "__")
    lock_path = LOCKS / f"{key}.lock"
    fh = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


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


PROGRAM_MEMORY = 16384                                  # bytes per AIE2P core


def fullest_core(bdir: Path) -> int | None:
    """The largest .text among a finished build's core ELFs (aiecc keeps them in final.prj),
    or None when they or Peano's llvm-size cannot be found."""
    try:
        from aie.utils.config import peano_install_dir
        size = Path(peano_install_dir()) / "bin" / "llvm-size"
        used = []
        for elf in (bdir / "final.prj").glob("elfs_*/*.elf"):
            r = subprocess.run([str(size), "-A", str(elf)], capture_output=True, text=True, timeout=30)
            used += [int(l.split()[1]) for l in r.stdout.splitlines() if l.startswith(".text")]
        return max(used) if used else None
    except Exception:
        return None


def stage_design_tree(stage_root: Path) -> None:
    """Copy the design sources into a per-spec stage.

    The stage replaces the designs/ directory: a design's HERE.parent is the
    designs dir (its siblings -- gemv_q4, attn, ln), HERE.parent.parent is
    open_kernels (include/), and HERE.parent.parent.parent is the repo
    (npu_offload/). The stage therefore has to sit one level under
    open_kernels/ for those to resolve the same way, and it holds the whole
    tree because the designs read each other.
    """
    def _ignore(dirpath, names):
        skipped = set()
        for n in names:
            p = Path(dirpath) / n
            if n == "__pycache__" or n.startswith(".staging"):
                skipped.add(n)
            elif p.is_dir() and n.startswith("build"):
                skipped.add(n)
        return skipped

    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DESIGNS, stage_root, ignore=_ignore, dirs_exist_ok=True)


def _build_one(name: str, sets: dict, spec_file: Path, stage_root: Path | None = None) -> tuple[str, Path]:
    """Build one kernel set. Worker for the parallel build; returns (name, bdir).

    stage_root, when given, is a per-spec mirror of the design tree: the design
    sources and the generated kernel TUs are copied into it, so this spec
    compiles its own sources and no other spec can change them mid-build. The
    mirror keeps design_path relative to the stage root, so the staged path has
    the same shape as the real one.
    """
    design_path = sets[name]["design"]
    src = (stage_root / design_path) if stage_root else (DESIGNS / design_path)

    out = DESIGNS / sets[name]["build_dir"]
    knobs = sets[name]["env"]
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update(knobs)
    env["OPEN_KERNELS_SPEC"] = str(spec_file)
    t0 = time.time()
    knob_str = " ".join(f"{k}={v}" for k, v in knobs.items())
    print(f"[{name}] {knob_str} python build_design.py {src.relative_to(HERE).as_posix()} "
          f"{out.relative_to(HERE).as_posix()}", flush=True)
    # The lock keeps two specs out of the same build dir at once: the designs
    # are shared, and build_design.py wipes its work dir before compiling.
    with design_lock(out):
        # The cache lives next to the output, so it is covered by the same lock
        # as the build: a spec that finds a current key reuses the xclbin
        # instead of rebuilding an identical one.
        key = build_content_key(src.parent, out, knobs)
        marker = out / KEY_FILE
        have = all((out / f).is_file() for f in FILES)
        if have and marker.is_file() and marker.read_text(encoding="utf-8").strip() == key:
            print(f"[{name}] up to date (build key {key[:19]}), reusing", flush=True)
            return name, out

        # Passed through as it comes, and watched for the one failure a user cannot act on: aiecc
        # reports a core whose program does not fit only as an invalid ELF.
        p = subprocess.Popen([sys.executable, str(HERE / "build_design.py"), str(src), str(out)], env=env,
                             cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail, overflow = b"", False
        for chunk in iter(lambda: p.stdout.read1(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            tail = (tail + chunk)[-4096:]
            overflow = overflow or b"Overflow of program memory" in tail
        if p.wait() != 0:
            if overflow:
                raise RuntimeError(f"[{name}] build FAILED: a core's program is larger than its {PROGRAM_MEMORY} B of "
                                   "program memory. This is a bug in the kernels for this model, not in your setup; "
                                   "please report it with this log.")
            # Repeat the final compiler output in the exception: the outer exporter
            # continues with other models, and parallel workers can interleave logs.
            detail = tail.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"[{name}] build FAILED ({p.returncode})\n"
                               f"Last build output:\n{detail or '(no output)'}")
        marker.write_text(key + "\n", encoding="utf-8")
    print(f"[{name}] built in {time.time() - t0:.0f}s", flush=True)
    used = fullest_core(out)
    if used:
        print(f"[{name}] program memory: fullest core {used} of {PROGRAM_MEMORY} B ({PROGRAM_MEMORY - used} B free)")
    return name, out


def build(name: str, sets: dict, spec_file: Path, stage_root: Path | None = None) -> Path:
    """Sequential build of a single set (kept for callers that want it)."""
    _, out = _build_one(name, sets, spec_file, stage_root)
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
    # Sequential by default: each build drives aiecc over the whole design, so
    # running several at once needs several times the memory of one. Callers that
    # know their machine (the Nix package build passes NIX_BUILD_CORES) opt in.
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="kernel-set builds to run in parallel within this spec (default: 1)")
    a = ap.parse_args()

    # ---- the spec, its recipe, and the kernel sets that recipe names
    if a.model_dir:
        spec = spec_from_model_dir(Path(a.model_dir))
    elif a.spec:
        spec = load_spec(Path(a.spec))
    else:
        spec = default_spec()
    F = for_spec(spec)
    try:
        F.recipe(spec)                               # refuses a spec outside the validated points
    except Exception as e:
        from recipes.catalogue import OpRangeError
        if isinstance(e, OpRangeError):
            sys.exit(
                f"{e}\n\nThis is a *catalogue* wall, not a correctness verdict: the recipe is\n"
                f"dim-correct for the kernel template, it just has not been built and fixture-tested\n"
                f"at that (family, K, ...). For an experimenter the path through is deliberate:\n"
                f"\n"
                f"  OPEN_KERNELS_UNVALIDATED=1 python {sys.argv[0]} ...\n"
                f"\n"
                f"to accept the off-grid point (you are saying you accept the build failing\n"
                f"or producing an untested kernel), and the build's make compare fixture will\n"
                f"tell you whether it matches. The point gets validated (and added to the\n"
                f"catalogue) once its fixture passes."
            )
        raise
    sets = F.builds(spec)
    names = [n.strip() for n in a.only.split(",") if n.strip()] if a.only else list(sets)
    bad = [n for n in names if n not in sets]
    if bad:
        sys.exit(f"unknown set(s) {bad}; the recipe builds {list(sets)}")
    key = build_key(spec)
    if not a.out and not spec.extra.get("model"):
        # It used to fall back to the 35B's name, so exporting a spec without one wrote
        # over the flagship's shipped set. Say so instead of guessing a published name.
        sys.exit(f"{a.spec or a.model_dir or 'the default spec'}: no extra.model, so there is "
                 f"no export destination. It is the src/xclbins/<name> directory, and the name "
                 f"`oflm add` links a converted model's kernels by. Add it to the spec, or "
                 f"pass --out.")
    out_root = Path(a.out).resolve() if a.out else (XCLBINS / spec.extra["model"] / "open_kernels")
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

    # ---- invalidate per-design cache when --force is used
    if a.force:
        for n in names:
            bdir = DESIGNS / sets[n]["build_dir"]
            marker = bdir / KEY_FILE
            if marker.is_file():
                marker.unlink()

    # ---- the whole-layer designs' kernel TUs for this spec, then the builds
    # The generated kernel TUs differ per spec (dense/gen_kernels.py and
    # layer_x/gen_kernels.py read the spec), and they land in a design dir
    # that 12 and 3 specs respectively share. Writing them there meant one
    # spec's gen could overwrite the sources another spec was compiling, and
    # the xclbin that came out was silently wrong. So each spec stages its
    # own copy of the design tree and compiles from that.
    stage_root = HERE / f".staging_{spec.spec_hash()[7:26]}"  # strip "sha256:", keep 19 chars

    if not a.no_build:
        os.environ["OPEN_KERNELS_SPEC"] = str(spec_file)
        gen_dir = HERE / F.GEN_KERNELS
        import importlib.util
        gspec = importlib.util.spec_from_file_location("gen_kernels", gen_dir)
        gen = importlib.util.module_from_spec(gspec)
        gspec.loader.exec_module(gen)
        # Stage the design tree once, before the parallel builds: the workers
        # would otherwise copy the same file over each other and one could
        # read a half-written source.
        stage_design_tree(stage_root)
        # Generate into the staged copy of the design dir the generator targets.
        gen_design_dir = gen_dir.parent.relative_to(DESIGNS)
        (stage_root / gen_design_dir).mkdir(parents=True, exist_ok=True)
        gen.generate(F.recipe(spec), out=stage_root / gen_design_dir)
    
    # Build the requested sets in parallel, then copy outputs in deterministic order.
    bdirs: dict[str, Path] = {}
    if a.no_build:
        for n in names:
            bdirs[n] = DESIGNS / sets[n]["build_dir"]
    else:
        workers = max(1, min(a.jobs, len(names)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_build_one, n, sets, spec_file, stage_root) for n in names]
            for future in concurrent.futures.as_completed(futures):
                n, bdir = future.result()
                bdirs[n] = bdir

    hashes: dict[str, str] = {}
    for n in names:
        bdir = bdirs[n]
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
