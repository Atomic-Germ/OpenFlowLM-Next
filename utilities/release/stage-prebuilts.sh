#!/usr/bin/env bash
#
# Stage the release prebuilts: the NPU artifacts the release CI cannot build.
#
#   utilities/release/stage-prebuilts.sh [--version X.Y.Z] [--dest DIR]
#                                        [--no-build] [--no-push]
#
# WHAT THIS IS FOR. The tag-triggered release workflow (.github/workflows/
# release.yml) has no NPU, so every kernel that needs one has to be built on a
# machine that does and handed to CI as a payload. This script is that machine.
# It produces `prebuilts/<version>/` -- an xclbin tarball, an optional Windows
# dependency bundle, SHA256SUMS and a manifest.json -- and force-pushes it to the
# orphan branch `npu-prebuilts`. That branch is the transport: one overwritten
# copy of the binaries, no bloat in the main history, and the release workflow
# fetches it by ref (utilities/release/fetch-prebuilts.sh).
#
# WHY A MANIFEST AND NOT "WHATEVER IS THERE". The manifest records the source
# TREE hashes of the two directories the kernels are generated from
# (open_kernels/ and npu_offload/). The release workflow compares them against
# the tag it was triggered by, so a release cannot ship kernels generated from
# different source than the engine that loads them. Tree hashes, not a commit:
# a release adds the version-bump commit after the build, so the commit moves
# while these stay put, and a rebuild that changes nothing is still a match.
#
# THE COLLECTION RULE IS .gitignore. Everything under src/xclbins that git
# ignores is a built kernel; everything git tracks is a closed set that already
# ships in the repo. So the payload is `git ls-files -o -i --exclude-standard --
# src/xclbins`: the BERT design sets (src/xclbins/BERT-h*) and the open kernel
# sets (src/xclbins/*/open_kernels*), which is exactly what .gitignore excludes.
# A new family's kernels is picked up with no edit here.
#
#   --dest DIR    assemble the bundle in DIR (default: a fresh mktemp dir)
#   --no-build    collect what is already in src/xclbins; do not run the export
#   --no-push     assemble and verify only, leaving the branch alone
#
# Windows dependencies (XRT headers, xrt_coreutil.lib) are captured by
# utilities/release/stage-prebuilts-win.ps1 into <dest>/win. Run that first
# when the release ships an MSI, then point this script at the same --dest.
# Without it the release still builds; the workflow marks the MSI job skipped
# (or fails it, with --require-windows) rather than shipping an MSI with no NPU
# runtime to link against.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${OFLM_PREBUILDS_BRANCH:-npu-prebuilts}"
KEEP_VERSIONS=2

VERSION=""
DEST=""
DO_BUILD=1
DO_PUSH=1
REQUIRE_WINDOWS=0

die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --version)         VERSION="${2:?--version needs a value}"; shift 2 ;;
        --dest)            DEST="${2:?--dest needs a value}"; shift 2 ;;
        --no-build)        DO_BUILD=0; shift ;;
        --no-push)         DO_PUSH=0; shift ;;
        --require-windows) REQUIRE_WINDOWS=1; shift ;;
        -h|--help)         sed -n '3,40p' "$0"; exit 0 ;;
        *)                 die "unknown argument: $1" ;;
    esac
done

cd "$REPO"

# ------------------------------------------------------------------- version
# Default to the version the tree is about to be tagged with: OFLM_VERSION in
# the presets, the documented single source of truth (docs/semantic-versioning.md).
if [ -z "$VERSION" ]; then
    VERSION="$(python3 - "$REPO" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
for rel in ("CMakePresets.json", "src/CMakePresets.json"):
    presets = json.loads((root / rel).read_text())["configurePresets"]
    print(next(p for p in presets if p["name"] == "common-default")
            ["cacheVariables"]["OFLM_VERSION"])
PY
)"
fi
case "$VERSION" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) die "version '$VERSION' is not MAJOR.MINOR.PATCH. The release workflow only
       accepts a plain X.Y.Z tag: an RPM or a DEB version field cannot carry a
       -rc suffix, and a tag that cannot be packaged is not a release." ;;
esac
echo "==> version: $VERSION"

[ -n "$DEST" ] || DEST="$(mktemp -d)/oflm-prebuilts-$VERSION"
mkdir -p "$DEST"
BUNDLE="$DEST/prebuilts/$VERSION"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"

# Fail on a missing prebuilts branch BEFORE the twenty-minute kernel build, not
# after it. The branch is an orphan (no common ancestor with main), so the
# initial creation is a one-time manual step.
if [ "$DO_PUSH" = 1 ]; then
    git fetch --quiet origin "$BRANCH" 2>/dev/null || true
fi
if [ "$DO_PUSH" = 1 ] && ! git rev-parse --verify --quiet "refs/remotes/origin/$BRANCH" >/dev/null; then
    die "origin/$BRANCH does not exist. Create the orphan branch once, on any
       machine with the repo:
           git fetch origin $BRANCH || git checkout --orphan $BRANCH
           git rm -rf --cached . ; rm -rf ./* ; git commit --allow-empty -m 'prebuilts: orphan root'
           git push origin $BRANCH
           git checkout -" >&2
fi

# --------------------------------------------------------------------- build
if [ "$DO_BUILD" = 1 ]; then
    command -v xclbinutil >/dev/null || die \
        "xclbinutil is not on PATH, so XRT is not installed (expected /opt/xilinx/xrt)"
    if ! ls /dev/accel/accel[0-9]* >/dev/null 2>&1; then
        echo "WARNING: no /dev/accel/accel* device. The open_kernels sets compile" >&2
        echo "         without one, but the BERT design sets allocate NPU tensors" >&2
        echo "         (device=\"npu\") and will fail. Continuing." >&2
    fi
    echo "==> building every kernel set (long; the BERT sets need the NPU)"
    python3 utilities/export-kernels.py --force
fi

# ------------------------------------------------------------------ collect
# -i is the load-bearing flag: without it git lists every untracked file, and a
# scratch file in the working tree would ship in the payload.
mapfile -t XCLBINS < <(git ls-files --others --ignored --exclude-standard -- src/xclbins)
[ "${#XCLBINS[@]}" -gt 0 ] || die "no built kernels under src/xclbins (run without --no-build)"
echo "==> ${#XCLBINS[@]} built kernel files to ship"

HAVE_WINDOWS=0
if [ -d "$DEST/win" ]; then HAVE_WINDOWS=1; fi
if [ "$REQUIRE_WINDOWS" = 1 ] && [ "$HAVE_WINDOWS" = 0 ]; then
    die "--require-windows, but $DEST/win does not exist. Run
       utilities/release/stage-prebuilts-win.ps1 -Dest $DEST first." >&2
fi

PAYLOAD="$BUNDLE/payload"
mkdir -p "$PAYLOAD"
# tar keeps the src/xclbins/<family>/... layout, which is what both consumers
# expect after extraction: CMake's `install(DIRECTORY xclbins ...)` and the WiX
# `<Files Include="$(XclbinsDir)\**">`.
tar -C "$REPO" -czf "$PAYLOAD/xclbins.tar.gz" --null -T <(printf '%s\0' "${XCLBINS[@]}")
if [ "$HAVE_WINDOWS" = 1 ]; then
    cp -a "$DEST/win" "$PAYLOAD/win"
fi

# ----------------------------------------------------------------- manifest
SRC_COMMIT="$(git rev-parse HEAD)"
TREE_OPEN_KERNELS="$(git rev-parse HEAD:open_kernels 2>/dev/null || echo absent)"
TREE_NPU_OFFLOAD="$(git rev-parse HEAD:npu_offload 2>/dev/null || echo absent)"

# Recorded, not verified: the answer to "which toolchain produced this", not an
# input to any check.
NPU_DRIVER="unknown"
if [ -r /opt/xilinx/xrt/version.txt ]; then
    NPU_DRIVER="$(tr -d '\r' < /opt/xilinx/xrt/version.txt | head -1)"
fi
XRT_VERSION="$(xclbinutil --version 2>/dev/null | head -1 || echo unknown)"

( cd "$PAYLOAD" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS )
cp "$PAYLOAD/SHA256SUMS" "$BUNDLE/SHA256SUMS"

VERSION="$VERSION" SRC_COMMIT="$SRC_COMMIT" \
TREE_OPEN_KERNELS="$TREE_OPEN_KERNELS" TREE_NPU_OFFLOAD="$TREE_NPU_OFFLOAD" \
NPU_DRIVER="$NPU_DRIVER" XRT_VERSION="$XRT_VERSION" \
HAVE_WINDOWS="$HAVE_WINDOWS" NXCLBINS="${#XCLBINS[@]}" \
python3 - "$BUNDLE/manifest.json" <<'PY'
import datetime, hashlib, json, os, sys

out = sys.argv[1]
payload = os.path.join(os.path.dirname(out), "payload")

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

artifacts = {name: {"sha256": digest(os.path.join(payload, name)),
                    "bytes": os.path.getsize(os.path.join(payload, name))}
             for name in sorted(os.listdir(payload))
             if os.path.isfile(os.path.join(payload, name))}

doc = {
    "schema": 1,
    "version": os.environ["VERSION"],
    "created_utc": datetime.datetime.now(datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": {
        "commit": os.environ["SRC_COMMIT"],
        "trees": {
            "open_kernels": os.environ["TREE_OPEN_KERNELS"],
            "npu_offload": os.environ["TREE_NPU_OFFLOAD"],
        },
    },
    "toolchain": {
        "npu_driver": os.environ["NPU_DRIVER"],
        "xrt": os.environ["XRT_VERSION"],
    },
    "artifacts": artifacts,
    "windows_deps": os.environ["HAVE_WINDOWS"] == "1",
    "n_xclbin_files": int(os.environ["NXCLBINS"]),
}
with open(out, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")
print("wrote", out)
PY

echo "==> bundle at $BUNDLE"
du -sh "$BUNDLE"
if [ "$DO_PUSH" = 0 ]; then
    echo "==> --no-push: $BRANCH left alone"
    exit 0
fi

# --------------------------------------------------------------------- push
echo "==> staging onto $BRANCH (scratch worktree)"
# Deliberately not committed to the working branch: that branch is the release
# commit, which carries no binaries. A prebuilt committed there by accident ends
# up in the tag and in main's history forever.
WORKTREE="$(mktemp -d)/wt"
git worktree add --detach "$WORKTREE" "origin/$BRANCH" >/dev/null
trap 'git worktree remove --force "$WORKTREE" 2>/dev/null || true' EXIT

rm -rf "${WORKTREE:?}/prebuilts/$VERSION"
mkdir -p "$WORKTREE/prebuilts"
cp -a "$BUNDLE" "$WORKTREE/prebuilts/$VERSION"
# Rolling window, not an archive: keep the newest KEEP_VERSIONS so an older tag
# can still be rebuilt, drop the rest before they bloat the branch.
ls -1 "$WORKTREE/prebuilts" | sort -V | head -n "-$KEEP_VERSIONS" \
    | while read -r old; do rm -rf "${WORKTREE:?}/prebuilts/$old"; done

git -C "$WORKTREE" add -A
git -C "$WORKTREE" -c user.name="oflm release" -c user.email="release@openflowlm.com" \
    commit -q -m "prebuilts: $VERSION (${#XCLBINS[@]} kernel files, windows deps: $HAVE_WINDOWS)"
REF="$(git -C "$WORKTREE" rev-parse HEAD)"
# --force: the branch is rewritten every release, and a re-run for the same
# version must replace the payload rather than pile up a second copy.
git -C "$WORKTREE" push --force origin "HEAD:$BRANCH"
echo "==> $BRANCH is now at $REF"
echo "$REF"
