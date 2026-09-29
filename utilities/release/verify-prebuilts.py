#!/usr/bin/env python3
"""Check that a tree carries the prebuilts a release of it needs.

The kernels and the Windows XRT inputs cannot be built by CI: they need an NPU.
They are built on a machine that has one, by utilities/release/stage-prebuilts.sh
and utilities/release/stage-prebuilts-win.ps1, committed to the staging branch,
and a release is tagged from that branch. So by the time anything is packaged,
they are in the tree already, and this asks the only question left to ask:

    are these the binaries THIS source was built for?

The three ways to get it wrong, all of which otherwise show up much later and
much less clearly:

  * a source-only commit landed on staging after the kernels were built, so the
    manifest records a different open_kernels/ or npu_offload/ tree than the tag
    has. The package loads and then fails to find a kernel, or worse, runs a
    kernel compiled for a different source revision.
  * a tag was cut from main, which has no binaries in it at all.
  * a cancelled NPU build left a zero-byte xclbin in the tree, and the failure
    arrives on a user's machine as a driver error naming no file and no release.

Run it locally before pushing a tag -- the answer is the same one the release
workflow will give you, without waiting for it:

    utilities/release/verify-prebuilts.py
    utilities/release/verify-prebuilts.py --version 0.1.0 --manifest /tmp/m.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

MANIFEST = "prebuilts/manifest.json"
WINDOWS_INCLUDE = os.path.join("prebuilts", "win", "xrt-include", "xrt", "xrt_device.h")
WINDOWS_LIB = os.path.join("prebuilts", "win", "xrt-lib", "xrt_coreutil.lib")


class Failure(Exception):
    """A reason not to release. The message is the whole point of this script."""


def git(*args: str, cwd: str = ".") -> str:
    out = subprocess.run(("git", *args), cwd=cwd, check=True,
                         capture_output=True, text=True).stdout
    return out.strip()


def version_from_presets(root: str) -> str:
    """OFLM_VERSION, which both presets have to agree on."""
    found = {}
    for rel in ("CMakePresets.json", os.path.join("src", "CMakePresets.json")):
        presets = json.load(open(os.path.join(root, rel)))["configurePresets"]
        common = next(p for p in presets if p["name"] == "common-default")
        found[rel] = common["cacheVariables"]["OFLM_VERSION"]
    if len(set(found.values())) != 1:
        raise Failure("the presets disagree about OFLM_VERSION: "
                      + ", ".join(f"{k}={v}" for k, v in sorted(found.items()))
                      + ". A release is the commit that bumps both.")
    return next(iter(found.values()))


def staged_kernel_files(root: str) -> list[str]:
    """Every built kernel file in src/xclbins, exactly as the staging script counts them.

    The rule is the one in .gitignore, asked of git rather than reimplemented
    here: a path the rules would ignore under src/xclbins is a built kernel.
    --no-index matters -- after the first staging run those files are TRACKED,
    and asking about untracked files alone finds nothing the second time round.
    """
    all_files = []
    for dirpath, _, names in os.walk(os.path.join(root, "src", "xclbins")):
        all_files += [os.path.join(dirpath, n) for n in names]
    rel = [os.path.relpath(p, root) for p in all_files]
    out = subprocess.run(
        ("git", "check-ignore", "--no-index", "--stdin", "-z"),
        cwd=root, input="\0".join(rel) + "\0",
        capture_output=True, text=True).stdout
    return sorted(p for p in out.split("\0") if p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="the tree to check (default: here)")
    ap.add_argument("--version", help="the version being released "
                                      "(default: OFLM_VERSION from the presets)")
    ap.add_argument("--manifest", default=MANIFEST, help=f"default: {MANIFEST}")
    ap.add_argument("--allow-missing", action="store_true",
                    help="warn instead of failing when there is no manifest: a "
                         "package that will load no open model, on purpose")
    args = ap.parse_args()
    root = args.root

    version = args.version or version_from_presets(root)
    path = args.manifest if os.path.isabs(args.manifest) else os.path.join(root, args.manifest)

    if not os.path.exists(path):
        if args.allow_missing:
            print(f"WARNING: no {args.manifest}: building with no open kernels")
            return 0
        raise Failure(
            f"there is no {args.manifest}.\n"
            "The kernel sets need an NPU, so they are built by\n"
            "  utilities/release/stage-prebuilts.sh       (Linux, with an NPU)\n"
            "  utilities/release/stage-prebuilts-win.ps1 (Windows, with an NPU)\n"
            "and committed to the staging branch, which the tag is then cut from.\n"
            "This tree looks like main, which has no binaries in it. See RELEASE.md.")

    doc = json.load(open(path))
    if doc.get("schema") != 2:
        raise Failure(f"{args.manifest} is schema {doc.get('schema')!r}, not 2. It was "
                      "written by an older pair of scripts; re-stage both platforms "
                      "with the scripts in this tree.")
    platforms = doc.get("platforms") or {}
    linux = platforms.get("linux")
    if not linux:
        raise Failure(f"{args.manifest} has no platforms.linux section, so no machine "
                      "has staged the open kernels. Run "
                      "utilities/release/stage-prebuilts.sh on an NPU machine and "
                      "push staging.")

    # 1. staged for this version
    if linux.get("for_version") != version:
        raise Failure(f"the kernels were staged for {linux.get('for_version')!r} but "
                      f"this release is {version!r}. Merge main into staging, re-stage "
                      "the kernels there, and tag again.")

    # 2. built from this source
    trees = linux.get("source_trees") or {}
    for key in ("open_kernels", "npu_offload"):
        actual = git("rev-parse", f"HEAD:{key}", cwd=root)
        if trees.get(key) != actual:
            raise Failure(
                f"{key}/ in this tree is {actual[:12]}, but the kernels were built "
                f"from {str(trees.get(key))[:12]}. Something was committed to staging "
                "after the kernels were built, so the package would ship kernels for "
                "a different engine. Re-stage them on the NPU machine.")

    # 3. the files the manifest claims are here, and are not empty
    kernels = staged_kernel_files(root)
    claimed = linux.get("kernel_files")
    if int(claimed) != len(kernels):
        raise Failure(f"the manifest claims {claimed} kernel file(s) and this tree has "
                      f"{len(kernels)}: the staging script and this check have to "
                      "agree on what a kernel file is.")
    for f in kernels:
        if os.path.getsize(os.path.join(root, f)) == 0:
            raise Failure(f"{f} is empty. A cancelled NPU build leaves a zero-byte "
                          "xclbin behind, and the failure that reports it arrives on "
                          "a user's machine as a driver error naming no file. "
                          "Re-stage the kernels on the NPU machine.")
    families = linux.get("families") or []
    if not families:
        raise Failure("the manifest lists no kernel families, so the engines find "
                      "nothing to load.")
    print(f"  linux:   {len(kernels)} kernel file(s), staged {linux.get('staged_utc')} "
          f"on XRT {linux.get('xrt', '?')}, driver {linux.get('npu_driver', '?')}")
    for family in families:
        n = sum(1 for f in kernels if f.split(os.sep)[2] == family)
        print(f"           {family:<32} {n} file(s)")
    unclaimed = sorted({f.split(os.sep)[2] for f in kernels} - set(families))
    if unclaimed:
        raise Failure(f"these families have kernel files in the tree but are not in "
                      f"the manifest's families list: {', '.join(unclaimed)}. The "
                      "families list is what the engines search.")

    # 4. the Windows inputs, if any were staged: all of them or none
    windows = platforms.get("windows")
    have_xrt = os.path.exists(os.path.join(root, WINDOWS_INCLUDE))
    have_lib = os.path.exists(os.path.join(root, WINDOWS_LIB))
    if windows and not (have_xrt and have_lib):
        missing = [p for p, there in ((WINDOWS_INCLUDE, have_xrt),
                                      (WINDOWS_LIB, have_lib)) if not there]
        raise Failure("the manifest has a platforms.windows section, so an XRT "
                      f"machine staged its side, but this tree is missing {', '.join(missing)}. "
                      "A half-staged Windows side is a release that cannot build an "
                      "MSI. Re-run utilities/release/stage-prebuilts-win.ps1.")
    if windows:
        if windows.get("for_version") != version:
            raise Failure(f"the XRT inputs were staged for "
                          f"{windows.get('for_version')!r} but this release is "
                          f"{version!r}.")
        lib_bytes = os.path.getsize(os.path.join(root, WINDOWS_LIB))
        if windows.get("import_lib_bytes") not in (None, lib_bytes):
            raise Failure(f"{WINDOWS_LIB} is {lib_bytes} bytes here and the manifest "
                          f"says {windows['import_lib_bytes']}. One of them is a "
                          "different file: the import library is made from the "
                          "driver's own DLL, so a mismatch means the wrong driver.")
        print(f"  windows: {windows.get('header_files', '?')} XRT headers, "
              f"xrt_coreutil.lib {lib_bytes} bytes, staged "
              f"{windows.get('staged_utc')} for XRT {windows.get('xrt_version')}")
    else:
        print(f"  windows: nothing staged (no platforms.windows section). The MSI job "
              f"will be skipped; the Linux packages are unaffected.")

    other = sorted(set(platforms) - {"linux", "windows"})
    if other:
        raise Failure(f"the manifest has platform sections this tree does not know "
                      f"how to check: {', '.join(other)}. Either a newer script wrote "
                      "it, or it was hand-edited.")
    print(f"OK: {args.manifest} and this tree agree, for {version}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git {' '.join(e.cmd)} failed: {e.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
