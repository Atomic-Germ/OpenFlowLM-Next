#!/usr/bin/env bash
#
# Stage the Linux prebuilts for a release: build every open kernel set on a
# machine that has an NPU, check the results in, and push them to `staging`.
#
#   utilities/release/stage-prebuilts.sh [--version X.Y.Z] [--no-build]
#                                        [--no-commit] [--no-push]
#                                        [--branch NAME] [--message TEXT]
#
# WHY A SCRIPT AND NOT A CI JOB. .github/workflows/release.yml has no NPU. The
# BERT design sets allocate NPU tensors (device="npu"), so `export-kernels.py`
# needs a card, and the XRT import library the MSI links against comes out of the
# Windows driver's own DLL. Both are built by a person on a machine that has the
# thing, and both end up in the `staging` branch, which is what a release is
# tagged from. The tag therefore *is* the commit that has the binaries: there is
# no payload to fetch, no hash to reconcile, and no way for a release to ship
# kernels generated from source other than the source in the tag.
#
# THE CONTRACT WITH THE OTHER MACHINE. utilities/release/stage-prebuilts-win.ps1
# runs the same dance on Windows for the XRT headers and xrt_coreutil.lib, and
# the two of you take turns on the same branch. What is shared is the layout and
# the manifest, so neither script has to know what the other staged:
#
#   src/xclbins/<family>/open_kernels/**   this script (Linux kernels)
#   prebuilts/win/**                       stage-prebuilts-win.ps1 (XRT inputs)
#   prebuilts/manifest.json                both, one section each
#
# The manifest records the source TREE hashes of the two directories the kernels
# are generated from (open_kernels/ and npu_offload/). The release workflow
# compares them against the tag, so a source-only commit pushed to staging after
# the kernels were built fails the release instead of shipping kernels the engine
# was not built for. Tree hashes, not a commit hash: the version bump lands on
# main and is merged in, so the commit moves while these stay put, and a rebuild
# that changes nothing is still a match.
#
# WHY `git add -f`. .gitignore excludes the built kernels (they are build
# products on a dev box, and main must never carry them), but a tracked file is
# never ignored -- so staging them is an explicit, auditable `add -f` of exactly
# the paths this script owns, rather than a branch-scoped .gitignore edit that
# would make src/ differ between branches.
#
#   --no-build    stage what is already in src/xclbins; do not run the export.
#                 Also the flag to pass if you just want to see what would be
#                 committed.
#   --no-commit   stage the files and write the manifest, but do not commit.
#   --no-push     commit locally, do not push.
#   --branch      the staging branch (default: staging)
#   --message     commit message (default: one naming the platform and count)
#
# GUARDS. The branch must be the staging branch, and the working tree must be
# clean: a release commit that is one developer's uncommitted source edit is a
# release nobody reviewed, and a CI run that never tested that source is worse.
# The build products themselves are ignored, so a dirty tree means something else
# is going on and the script stops.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${OFLM_STAGING_BRANCH:-staging}"
PLATFORM="linux"
MANIFEST="prebuilts/manifest.json"

VERSION=""
DO_BUILD=1
DO_COMMIT=1
DO_PUSH=1
MESSAGE=""

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "==> $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --version)  VERSION="${2:?--version needs a value}"; shift 2 ;;
        --branch)   BRANCH="${2:?--branch needs a value}"; shift 2 ;;
        --message)  MESSAGE="${2:?--message needs a value}"; shift 2 ;;
        --no-build) DO_BUILD=0; shift ;;
        --no-commit) DO_COMMIT=0; shift ;;
        --no-push)  DO_PUSH=0; shift ;;
        -h|--help)  sed -n '3,49p' "$0"; exit 0 ;;
        *)          die "unknown argument: $1" ;;
    esac
done

cd "$REPO"

# ------------------------------------------------------------------- version
# Default to the version the tree is about to be tagged with: OFLM_VERSION in
# the presets, the documented single source of truth (docs/semantic-versioning.md).
if [ -z "$VERSION" ]; then
    # Both presets, and they have to agree. The release workflow checks this too,
    # but a disagreement found here is found twenty minutes before a forty-minute
    # kernel build rather than in CI afterwards -- and the value has to be a
    # single line, because it goes into the manifest and the commit message.
    VERSION="$(python3 - <<'PY'
import json, pathlib, sys
found = {}
for rel in ("CMakePresets.json", "src/CMakePresets.json"):
    presets = json.loads(pathlib.Path(rel).read_text())["configurePresets"]
    common = next(p for p in presets if p["name"] == "common-default")
    found[rel] = common["cacheVariables"]["OFLM_VERSION"]
if len(set(found.values())) != 1:
    sys.exit("ERROR: the presets disagree about OFLM_VERSION: "
             + ", ".join(f"{k}={v}" for k, v in sorted(found.items()))
             + ". One of them was not bumped.")
print(next(iter(found.values())))
PY
)" || die "could not read OFLM_VERSION from the presets"
fi
case "$VERSION" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) die "version '$VERSION' is not MAJOR.MINOR.PATCH. A release tag with a
       pre-release suffix in it cannot be packaged as an RPM or a DEB." ;;
esac
say "version: $VERSION"
say "branch:  $BRANCH"

# ------------------------------------------------------------------- guards
current="$(git branch --show-current)"
[ "$current" = "$BRANCH" ] || die "you are on '$current', not '$BRANCH'.
   The prebuilts are committed to a branch that is not main, and a release is
   tagged from it. One-time setup, from a clean clone of main:

       git checkout -b $BRANCH
       git push -u origin $BRANCH

   Then merge main into it at the start of each release cycle:
       git merge main"

# Changes under the paths this script owns are the whole point of it -- a rebuild
# that drops a kernel, adds a family, or updates a spec -- so they are allowed,
# and they are what the commit below is made of. Everything else is refused: a
# release commit that happens to carry someone's uncommitted source edit is a
# release nobody reviewed and no CI run tested.
OWNED='^(src/xclbins/[^/]+/open_kernels[^/]*/|src/xclbins/BERT-h[^/]*/|prebuilts/)'
FOREIGN="$(git status --porcelain | sed 's/^...//' | grep -Ev "$OWNED" || true)"
if [ -n "$FOREIGN" ]; then
    echo "$FOREIGN" | sed 's/^/  /' >&2
    die "the working tree has changes outside the prebuilt kernels and
       prebuilts/. Commit or stash them first: a release must not be the commit
       that happens to carry uncommitted work."
fi
if [ -n "$(git status --porcelain)" ]; then
    say "$(git status --porcelain | wc -l) change(s) under the prebuilt paths"
fi

git fetch --quiet origin "$BRANCH" || die "cannot fetch origin/$BRANCH"
if ! git merge --ff-only --quiet "origin/$BRANCH" 2>/dev/null; then
    if [ -n "$(git rev-parse HEAD)" ] && [ "$(git rev-list --count "origin/$BRANCH"..HEAD)" != 0 ]; then
        die "you have commits that origin/$BRANCH does not. The other developer
       probably pushed first. Rebase onto it and re-run:
           git pull --rebase origin $BRANCH"
    fi
fi
say "up to date with origin/$BRANCH ($(git rev-parse --short "origin/$BRANCH"))"

# --------------------------------------------------------------------- build
if [ "$DO_BUILD" = 1 ]; then
    command -v xclbinutil >/dev/null || die \
        "xclbinutil is not on PATH, so XRT is not installed (expected /opt/xilinx/xrt)"
    if ! ls /dev/accel/accel[0-9]* >/dev/null 2>&1; then
        die "no /dev/accel/accel* device. The open_kernels sets compile without
       one, but the BERT design sets allocate NPU tensors (device=\"npu\") and
       will fail, so a bundle without them is not a release. Run this on the NPU
       machine."
    fi
    say "building every kernel set (long; the BERT sets need the NPU)"
    python3 utilities/export-kernels.py --force
else
    say "--no-build: staging what is already in src/xclbins"
fi

# ------------------------------------------------------------------ collect
# The collection rule is .gitignore: everything under src/xclbins that the rules
# would ignore is a built kernel, and everything that is not is a closed set
# which already ships in the repo. A new family is picked up with no edit here.
#
# `check-ignore --no-index` rather than `ls-files --others --ignored`: after the
# first staging run those kernels are TRACKED on this branch, and "--others" only
# ever lists untracked files, so the rule finds nothing the second time round and
# the script reports "no built kernels" with a full src/xclbins sitting right
# there. --no-index asks the question that is actually being asked -- would this
# path be ignored if it were untracked -- and it answers it for tracked files too.
mapfile -t ALL_XCLBINS < <(find src/xclbins -type f)
mapfile -t XCLBINS < <(
    git check-ignore --no-index --stdin -z < <(printf '%s\0' "${ALL_XCLBINS[@]}") \
        | tr '\0' '\n' || true
)
[ "${#XCLBINS[@]}" -gt 0 ] || die "no built kernels under src/xclbins. Run without
       --no-build, or check that .gitignore still excludes them (the rule is
       src/xclbins/*/open_kernels*/ and src/xclbins/BERT-h*/)."
say "${#XCLBINS[@]} built kernel files to stage"

# ------------------------------------------------------------------- verify
# Shape, not a load test: every kernel directory has to carry both halves of a
# compiled kernel, and the per-family spec has to parse. A build that died
# half-way through leaves a directory with a spec and no final.xclbin, and
# without this it is committed and shipped.
python3 - "${XCLBINS[@]}" <<'PY' || die "the built kernels are not complete (see above)"
import json, pathlib, sys

files = sys.argv[1:]
by_kernel: dict[str, set[str]] = {}
for f in files:
    p = pathlib.Path(f)
    # src/xclbins/<family>/open_kernels/<kernel>/{final.xclbin,insts.bin}
    if p.parent.name == "open_kernels" or p.parent.parent.name != "open_kernels":
        continue
    by_kernel.setdefault(str(p.parent), set()).add(p.name)

bad = []
for kernel, names in sorted(by_kernel.items()):
    for required in ("final.xclbin", "insts.bin"):
        if required not in names:
            bad.append(f"{kernel}: no {required} (has: {', '.join(sorted(names)) or 'nothing'})")

for spec in sorted(p for p in files if p.endswith("spec.json")):
    try:
        json.loads(pathlib.Path(spec).read_text())
    except Exception as exc:  # noqa: BLE001 - the message is the point
        bad.append(f"{spec}: does not parse ({exc})")

if not by_kernel:
    bad.append("no <family>/open_kernels/<kernel>/ directories at all; is the "
               "export writing somewhere else?")

if bad:
    print("\n".join(f"  INCOMPLETE: {b}" for b in bad), file=sys.stderr)
    sys.exit(1)
print(f"    {len(by_kernel)} kernel directories, each with final.xclbin and insts.bin")
PY

FAMILIES="$(printf '%s\n' "${XCLBINS[@]}" | cut -d/ -f3 | sort -u | tr '\n' ' ')"
say "families: $FAMILIES"

# ----------------------------------------------------------------- manifest
mkdir -p prebuilts
NPU_DRIVER="unknown"
[ -r /opt/xilinx/xrt/version.txt ] && NPU_DRIVER="$(tr -d '\r' < /opt/xilinx/xrt/version.txt | head -1)"
# xclbinutil indents its version banner; this field is read by a human six months
# from now, so squeeze the whitespace rather than storing it.
XRT_VERSION="$(xclbinutil --version 2>/dev/null | head -1 | tr -s '[:space:]' ' ' || echo unknown)"
XRT_VERSION="${XRT_VERSION# }"
XRT_VERSION="${XRT_VERSION% }"

# Read-modify-write: this script owns the "linux" section and nothing else, so it
# never clobbers the section stage-prebuilts-win.ps1 wrote.
TREE_OPEN="$(git rev-parse HEAD:open_kernels)"
TREE_NPU="$(git rev-parse HEAD:npu_offload)"

# Read-modify-write: this script owns the "linux" section and nothing else, so it
# never clobbers the section stage-prebuilts-win.ps1 wrote.
VERSION="$VERSION" PLATFORM="$PLATFORM" NPU_DRIVER="$NPU_DRIVER" \
XRT_VERSION="$XRT_VERSION" NXCLBINS="${#XCLBINS[@]}" FAMILIES="$FAMILIES" \
TREE_OPEN="$TREE_OPEN" TREE_NPU="$TREE_NPU" MANIFEST="$MANIFEST" python3 - <<'PY'
import datetime, json, os, sys

out = os.environ["MANIFEST"]
try:
    doc = json.load(open(out))
except FileNotFoundError:
    doc = {"schema": 2, "platforms": {}}
if doc.get("schema") != 2:
    sys.exit(f"ERROR: {out} is schema {doc.get('schema')!r} but this script writes "
             "schema 2. Delete it and re-stage both platforms, or check out the "
             "version of stage-prebuilts-win.ps1 that agrees with this one.")
platforms = doc.setdefault("platforms", {})

platforms[os.environ["PLATFORM"]] = {
    "staged_utc": datetime.datetime.now(datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "for_version": os.environ["VERSION"],
    "npu_driver": os.environ["NPU_DRIVER"],
    "xrt": os.environ["XRT_VERSION"],
    "kernel_files": int(os.environ["NXCLBINS"]),
    "families": os.environ["FAMILIES"].split(),
    # Recorded, and checked by the release workflow against the tag: a
    # source-only commit pushed to staging after the kernels were built fails the
    # release instead of shipping kernels the engine was not built for.
    "source_trees": {"open_kernels": os.environ["TREE_OPEN"],
                     "npu_offload": os.environ["TREE_NPU"]},
}
doc["platforms"] = dict(sorted(platforms.items()))
with open(out, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")
print("    " + out + ": " + ", ".join(doc["platforms"]))
PY

cat "$MANIFEST"

# ------------------------------------------------------------------- commit
# -f, because .gitignore excludes the kernels by design (see the header), and
# only these two paths: the guard above proved the tree is otherwise clean, so
# this cannot pick up anything the script did not build.
git add -f -- src/xclbins
git add -- "$MANIFEST"
say "staged $(git diff --cached --name-only | wc -l) path(s)"

if [ "$DO_COMMIT" = 0 ]; then
    say "--no-commit: leaving the index staged for you to look at"
    git diff --cached --stat
    exit 0
fi

[ -n "$MESSAGE" ] || MESSAGE="prebuilts(linux): ${#XCLBINS[@]} kernel files for $VERSION"
git -c user.name="${GIT_AUTHOR_NAME:-oflm release}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-release@openflowlm.com}" \
    commit -q -m "$MESSAGE"
COMMIT="$(git rev-parse HEAD)"
say "committed $COMMIT on $BRANCH"

# --------------------------------------------------------------------- push
if [ "$DO_PUSH" = 0 ]; then
    say "--no-push: $BRANCH is at $COMMIT locally"
    exit 0
fi

# The other developer's push can land during a forty-minute kernel build, and
# both of us touch prebuilts/manifest.json -- so a rejected push is the normal
# case, not the exception. Merge theirs in; if the manifest is the conflict,
# resolve it the only way that can be right, and without making them rebuild the
# kernels: take the file from origin, then re-apply our section on top of it.
#
# Taking it from origin explicitly (git show <ref>:<path>) rather than with
# --ours/--theirs, because those two mean opposite things in a merge and a
# rebase, and this script should not be able to get that backwards. A merge and
# not a rebase, because a merge resolves without opening an editor.
if ! git push --quiet origin "HEAD:$BRANCH"; then
    say "push rejected: origin/$BRANCH moved. Merging it in..."
    git fetch --quiet origin "$BRANCH"
    if ! git merge --no-edit "origin/$BRANCH"; then
        UNMERGED="$(git diff --name-only --diff-filter=U)"
        case "$UNMERGED" in
            *"$MANIFEST"*) ;;
            *) die "merge failed on
       $UNMERGED
       which this script does not know how to resolve. Merge by hand:
           git merge origin/$BRANCH" ;;
        esac
        say "manifest conflict: taking origin's sections, re-applying linux"
        git show "origin/$BRANCH:$MANIFEST" > "$MANIFEST" \
            || die "origin/$BRANCH has no $MANIFEST to merge with. Merge by hand."
        MANIFEST="$MANIFEST" PLATFORM="$PLATFORM" VERSION="$VERSION" \
        NPU_DRIVER="$NPU_DRIVER" XRT_VERSION="$XRT_VERSION" \
        NXCLBINS="${#XCLBINS[@]}" FAMILIES="$FAMILIES" \
        TREE_OPEN="$TREE_OPEN" TREE_NPU="$TREE_NPU" python3 - <<'PY'
# Same read-modify-write as before, on the file that just came from origin: add
# our key, keep every other key, sort. Shared with the block above on purpose --
# two copies of "put my section in" that could drift is a worse failure than a
# little repetition.
import datetime, json, os, sys

out = os.environ["MANIFEST"]
doc = json.load(open(out))
if doc.get("schema") != 2:
    sys.exit(f"ERROR: {out} is schema {doc.get('schema')!r} on origin/$BRANCH but "
             "this script writes schema 2.")
platforms = doc.setdefault("platforms", {})
platforms[os.environ["PLATFORM"]] = {
    "staged_utc": datetime.datetime.now(datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "for_version": os.environ["VERSION"],
    "npu_driver": os.environ["NPU_DRIVER"],
    "xrt": os.environ["XRT_VERSION"],
    "kernel_files": int(os.environ["NXCLBINS"]),
    "families": os.environ["FAMILIES"].split(),
    "source_trees": {"open_kernels": os.environ["TREE_OPEN"],
                     "npu_offload": os.environ["TREE_NPU"]},
}
doc["platforms"] = dict(sorted(platforms.items()))
with open(out, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")
PY
        git add -- "$MANIFEST"
        git commit --no-edit >/dev/null \
            || die "could not finish the merge after resolving the manifest.
       Merge by hand: git merge origin/$BRANCH"
    fi
    git push --quiet origin "HEAD:$BRANCH" \
        || die "push failed again. Something else moved; pull and re-run."
fi
say "pushed $BRANCH"

cat <<EOF

Kernels are on $BRANCH. If the other machine has not staged its side yet, the
XRT headers and xrt_coreutil.lib the MSI needs come from the Windows box with
the NPU driver:

    git checkout $BRANCH && git pull
    utilities\\release\\stage-prebuilts-win.ps1

When prebuilts/manifest.json has a section for both platforms, tag from this
branch -- not from main, which has no binaries in it:

    git tag -a v$VERSION -m 'OpenFlowLM $VERSION' && git push origin v$VERSION
EOF
