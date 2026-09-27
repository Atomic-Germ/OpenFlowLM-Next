#!/usr/bin/env bash
#
# Fetch and verify the release prebuilts, then lay them out for a build.
#
#   utilities/release/fetch-prebuilts.sh --version X.Y.Z [--branch npu-prebuilts]
#                                         [--ref <sha>] [--dest DIR]
#                                         [--no-extract] [--allow-missing]
#
# Called by .github/workflows/release.yml, and usable by hand to reproduce a CI
# package on a laptop. It does four things, in this order, and stops at the first
# failure that matters:
#
#   1. fetches the prebuilts branch (an orphan: no common ancestor with the tag)
#      at --ref, or its tip;
#   2. checks the manifest's `version` against --version;
#   3. checks the manifest's source TREE hashes against this checkout's, which is
#      what stops a release from shipping kernels generated from other source
#      than the engine loading them (see stage-prebuilts.sh for why trees);
#   4. verifies every payload file against SHA256SUMS and extracts.
#
# Step 3 is the whole reason this is a script and not a `tar xf`. A prebuilt
# bundle is a large opaque blob; without it, a release built from a stale branch
# tip would ship a mismatched kernel set and the failure would surface as a
# wrong number from a model, days later, on a user's NPU.
#
# Layout after a successful run (--dest defaults to the repo root):
#   src/xclbins/<family>/...        the built kernel sets, where CMake and WiX
#                                   both look for them
#   prebuilts/win/{xrt-include,xrt-lib}   the Windows NPU runtime deps, if the
#                                   bundle carried them
#
# --allow-missing downgrades a missing bundle to a warning and exits 0, for a
# dry run or a Linux-only packaging job; the release workflow uses it only for
# `workflow_dispatch` with skip_prebuilts set.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${OFLM_PREBUILDS_BRANCH:-npu-prebuilts}"

VERSION=""
REF=""
DEST=""
EXTRACT=1
ALLOW_MISSING=0

die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --version)       VERSION="${2:?--version needs a value}"; shift 2 ;;
        --branch)        BRANCH="${2:?--branch needs a value}"; shift 2 ;;
        --ref)           REF="${2:?--ref needs a value}"; shift 2 ;;
        --dest)          DEST="${2:?--dest needs a value}"; shift 2 ;;
        --no-extract)    EXTRACT=0; shift ;;
        --allow-missing) ALLOW_MISSING=1; shift ;;
        -h|--help)       sed -n '3,30p' "$0"; exit 0 ;;
        *)               die "unknown argument: $1" ;;
    esac
done

cd "$REPO"
[ -n "$VERSION" ] || die "--version is required"
[ -n "$DEST" ] || DEST="$REPO"

# ------------------------------------------------------------------- fetch
echo "==> fetching $BRANCH${REF:+ at $REF}"
FETCH_ARGS=(origin "$BRANCH")
[ -n "$REF" ] && FETCH_ARGS+=("$REF")
if ! git fetch --quiet --depth 1 "${FETCH_ARGS[@]}"; then
    if [ "$ALLOW_MISSING" = 1 ]; then
        echo "WARNING: $BRANCH is not fetchable; continuing without prebuilts." >&2
        echo "WARNING: the package built from this will ship no open kernels." >&2
        exit 0
    fi
    die "cannot fetch $BRANCH. It is an orphan branch that must be created once
       (see the error text in utilities/release/stage-prebuilts.sh) and then
       pushed by a run of that script on an NPU machine." >&2
fi
COMMIT="$(git rev-parse --short "FETCH_HEAD^{commit}")"
echo "==> $BRANCH = $COMMIT"

# An orphan branch is a tree like any other, but it has to be read WITHOUT
# checking it out: the working tree must stay on the tag, and `git checkout
# <commit> -- <path>` would rewrite the index of that tree as well. `git archive`
# reads a tree and writes a tarball, touching neither.
BUNDLE_SRC="$(mktemp -d)"
trap 'rm -rf "$BUNDLE_SRC"' EXIT
# Ask git whether the version is there before asking it to extract it. A
# missing pathspec otherwise prints "fatal: pathspec ... did not match" and then
# a tar error, and the one line an operator needs to read is the one at the
# bottom, after two lines of noise that look like the real problem.
if git cat-file -e "FETCH_HEAD:prebuilts/$VERSION" 2>/dev/null; then
    git archive --format=tar "FETCH_HEAD" "prebuilts/$VERSION" | tar -x -C "$BUNDLE_SRC"
else
    git archive --format=tar "FETCH_HEAD" prebuilts | tar -x -C "$BUNDLE_SRC" || true
fi
BUNDLE="$BUNDLE_SRC/prebuilts/$VERSION"
if [ ! -d "$BUNDLE" ]; then
    have="$(ls -1 "$BUNDLE_SRC/prebuilts" 2>/dev/null | tr '\n' ' ')"
    if [ "$ALLOW_MISSING" = 1 ]; then
        # Loud, because the alternative is a release that looks fine and ships a
        # package no open model can load. Two lines, in the log, in caps.
        echo "WARNING: prebuilts/$VERSION IS NOT ON $BRANCH (it has:${have:- nothing})" >&2
        echo "WARNING: THIS PACKAGE WILL SHIP NO OPEN KERNELS. verify-package.sh will" >&2
        echo "WARNING: fail on it, which is the only thing standing between this and a" >&2
        echo "WARNING: published release that cannot load an open model." >&2
        exit 0
    fi
    die "prebuilts/$VERSION is not on $BRANCH (it has:${have:- nothing})"
fi

# ---------------------------------------------------------------- manifest
MANIFEST="$BUNDLE/manifest.json"
[ -f "$MANIFEST" ] || die "$MANIFEST is missing from the bundle"
python3 - "$MANIFEST" "$VERSION" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
want = sys.argv[2]
got = doc.get("version")
if got != want:
    sys.exit(f"ERROR: the prebuilts bundle is for version {got!r}, not {want!r}. "
             f"Re-run utilities/release/stage-prebuilts.sh on the NPU machine for {want}.")
if doc.get("schema") != 1:
    sys.exit(f"ERROR: unsupported prebuilts manifest schema {doc.get('schema')!r}; "
             f"this checkout of fetch-prebuilts.sh understands schema 1.")
print(f"    built {doc.get('created_utc')} from {doc['source']['commit'][:12]}")
print(f"    {doc.get('n_xclbin_files')} kernel files, "
      f"windows deps: {doc.get('windows_deps')}")
print(f"    npu driver: {doc.get('toolchain', {}).get('npu_driver')}")
PY

# ------------------------------------------------------------- source trees
python3 - "$MANIFEST" <<'PY'
import json, subprocess, sys
doc = json.load(open(sys.argv[1]))
bad = []
for path, recorded in doc["source"]["trees"].items():
    try:
        actual = subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], text=True).strip()
    except subprocess.CalledProcessError:
        actual = "absent"
    if actual != recorded:
        bad.append(f"  {path}: prebuilts {recorded[:12]} vs this checkout {actual[:12]}")
if bad:
    sys.exit("ERROR: the prebuilts were built from different source than this tag:\n"
             + "\n".join(bad)
             + "\n\nOpen kernels are generated per spec and the closed/xclbin ABI moves with\n"
               "the source, so a mismatch means the package would load kernels the engine\n"
               "was not built for. Re-run utilities/release/stage-prebuilts.sh (without\n"
               "--no-build) from the release commit and push it again.")
print("    source trees match the tag")
PY

# ----------------------------------------------------------------- payload
( cd "$BUNDLE/payload" && sha256sum --quiet -c ../SHA256SUMS ) \
    || die "prebuilt payload failed SHA256SUMS verification (the bundle is corrupt or truncated)"

if [ "$EXTRACT" = 0 ]; then
    echo "==> --no-extract: payload verified, nothing unpacked"
    exit 0
fi

# tar -x into the repo: the archive's members are already src/xclbins/... paths,
# so this is where the kernels belong. Existing files are the closed sets git
# tracks, which the tar does not contain, so nothing is overwritten.
echo "==> extracting xclbins into $DEST/src/xclbins"
tar -C "$DEST" -xzf "$BUNDLE/payload/xclbins.tar.gz"
EXTRACTED="$(find "$DEST/src/xclbins" -name '*.xclbin' | wc -l)"
echo "==> $EXTRACTED xclbins present under src/xclbins"

if [ -d "$BUNDLE/payload/win" ]; then
    mkdir -p "$DEST/prebuilts/win"
    cp -a "$BUNDLE/payload/win/." "$DEST/prebuilts/win/"
    echo "==> windows deps: $(ls "$DEST/prebuilts/win" | tr '\n' ' ')"
fi

# The build tree links src/xclbins next to the executable; a leftover from an
# earlier build would shadow what we just unpacked.
if [ -L "$DEST/build-debug/xclbins" ] || [ -L "$DEST/build/xclbins" ]; then
    echo "==> removing a stale build/xclbins link (it would shadow the prebuilts)"
    rm -f "$DEST/build-debug/xclbins" "$DEST/build/xclbins"
fi
